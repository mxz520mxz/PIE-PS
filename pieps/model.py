"""PIE encoder, differentiable guide cues, Transformer reliability and refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, knn_graph
from torch_scatter import scatter_max
from torch.utils.checkpoint import checkpoint

from pieps.layers.components import Cartesian
from pieps.layers.conv import Layer
from pieps.layers.ev_tgn import EV_TGN

class TransformerSelfConsistencyBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4, ff_mult=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask):
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PixelGNNStack(nn.Module):
    def __init__(self, in_dim=64, hid_dim=128, num_layers=5, k=8):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hid_dim))
        self.norms.append(nn.LayerNorm(hid_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.norms.append(nn.LayerNorm(hid_dim))
        self.convs.append(GCNConv(hid_dim, in_dim))
        self.norms.append(nn.LayerNorm(in_dim))


class PIEPS(nn.Module):
    def __init__(
        self,
        args,
        height=256,
        width=256,
        feat_dim=7,
        enc_dim=64,
        gnn_hid=128,
        pixel_gnn_layers=5,
        k=8,
        time_scale=0.3,
        scorer_hidden_dim=64,
        scorer_num_layers=2,
        scorer_num_heads=4,
        scorer_pixel_chunk=1024,
        c_eps=1e-6,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.time_scale = time_scale
        self.c_eps = c_eps
        self.feat_dim = feat_dim
        self.events_to_graph = EV_TGN(args)
        effective_radius = 2 * float(int(args.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        event_out_ch = int(args.base_width * 32)
        self.conv_block1 = Layer(feat_dim + 2, event_out_ch, args=args)

        self.raw_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.gnn_mlp = nn.Sequential(
            nn.Linear(event_out_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU(),
        )
        coarse_stack = PixelGNNStack(
            in_dim=enc_dim, hid_dim=gnn_hid, num_layers=pixel_gnn_layers, k=k
        )
        self.pixel_convs = coarse_stack.convs
        self.pixel_norms = coarse_stack.norms
        self.pixel_gnn_layers = pixel_gnn_layers
        self.pixel_gnn_k = k
        self.normal_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        scorer_in_dim = feat_dim + 64 + 9
        self.scorer_mlp = nn.Sequential(
            nn.Linear(scorer_in_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, scorer_hidden_dim),
            nn.ReLU(),
        )
        self.scorer_pixel_chunk = scorer_pixel_chunk
        self.checkpoint_scorer = False
        self.scorer_layers = nn.ModuleList([
            TransformerSelfConsistencyBlock(scorer_hidden_dim, scorer_num_heads)
            for _ in range(scorer_num_layers)
        ])
        self.scorer_head = nn.Linear(scorer_hidden_dim, 1)

        self.refine_proj = nn.Sequential(
            nn.Linear(128, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU(),
        )
        refine_stack = PixelGNNStack(
            in_dim=enc_dim, hid_dim=gnn_hid, num_layers=pixel_gnn_layers, k=k
        )
        self.refine_pixel_convs = refine_stack.convs
        self.refine_pixel_norms = refine_stack.norms
        self.refine_normal_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self._copy_coarse_to_refine()

    def _copy_coarse_to_refine(self):
        self.refine_proj.load_state_dict(self.proj.state_dict(), strict=True)
        for refine_conv, coarse_conv in zip(self.refine_pixel_convs, self.pixel_convs):
            refine_conv.load_state_dict(coarse_conv.state_dict(), strict=True)
        for refine_norm, coarse_norm in zip(self.refine_pixel_norms, self.pixel_norms):
            refine_norm.load_state_dict(coarse_norm.state_dict(), strict=True)
        self.refine_normal_head.load_state_dict(self.normal_head.state_dict(), strict=True)

    def _pixel_gnn_forward(self, x, pixel_pos, convs, norms):
        if x.shape[0] <= 1:
            return x
        k_eff = min(self.pixel_gnn_k, max(x.shape[0] - 1, 1))
        edge_index = knn_graph(pixel_pos, k=k_eff, loop=False)
        residual = x
        for i, (conv, norm) in enumerate(zip(convs, norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.pixel_gnn_layers - 1:
                x = F.relu(x)
        return x + residual

    def _encode_observations(self, pixel_feats, pixel_mask, pixel_pos, pixel_t):
        device = pixel_feats.device
        n_pix, n_obs, feat_dim = pixel_feats.shape
        pos_xy = pixel_pos.unsqueeze(1).expand(-1, n_obs, -1)
        pos_t = (pixel_t * self.time_scale).unsqueeze(-1)

        flat_mask = pixel_mask.reshape(-1) > 0
        flat_feats = pixel_feats.reshape(-1, feat_dim)[flat_mask]
        flat_pos = torch.cat([pos_xy, pos_t], dim=-1).reshape(-1, 3)[flat_mask]
        raw_feats = flat_feats.clone()

        flat_pix = (
            torch.arange(n_pix, device=device)
            .unsqueeze(1)
            .expand(-1, n_obs)
            .reshape(-1)[flat_mask]
        )
        flat_obs = (
            torch.arange(n_obs, device=device)
            .unsqueeze(0)
            .expand(n_pix, -1)
            .reshape(-1)[flat_mask]
        )

        data = Data(
            x=flat_feats,
            pos=flat_pos,
            width=torch.tensor(self.width, device=device),
            height=torch.tensor(self.height, device=device),
            time_window=torch.tensor(1000, device=device),
        )
        data = Batch.from_data_list([data])
        data = self.events_to_graph(data, reset=True)
        data = self.edge_attrs(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        data.x = torch.cat([data.x, data.pos[:, :2]], dim=1)
        data = self.conv_block1(data)

        h_raw = self.raw_mlp(raw_feats)
        h_gnn = self.gnn_mlp(data.x)
        h_obs = torch.cat([h_raw, h_gnn], dim=-1)

        h_obs_padded = pixel_feats.new_zeros(n_pix, n_obs, h_obs.shape[-1])
        h_obs_padded[flat_pix, flat_obs] = h_obs
        return {
            "flat_pix": flat_pix,
            "flat_obs": flat_obs,
            "h_obs": h_obs,
            "h_obs_padded": h_obs_padded,
        }

    def _aggregate_mean_max(self, h_obs, flat_pix, n_pix):
        device = h_obs.device
        h_dim = h_obs.shape[1]
        h_mean = torch.zeros(n_pix, h_dim, device=device)
        counts = torch.zeros(n_pix, 1, device=device)
        h_mean.index_add_(0, flat_pix, h_obs)
        counts.index_add_(0, flat_pix, torch.ones(h_obs.shape[0], 1, device=device))
        h_mean = h_mean / counts.clamp(min=1.0)

        h_max_raw, _ = scatter_max(h_obs, flat_pix, dim=0, dim_size=n_pix)
        h_max = torch.where(h_max_raw > -1e8, h_max_raw, torch.zeros_like(h_max_raw))
        return h_mean, h_max

    def _build_scoring_inputs(
        self,
        pixel_feats,
        pixel_mask,
        pixel_l_k,
        pixel_l_k1,
        pixel_p_k1,
        h_obs_padded,
        coarse_normal,
    ):
        mask_bool = pixel_mask > 0
        # Final-normal supervision must reach the guide branch through these cues.
        n0 = coarse_normal
        a_k = (n0.unsqueeze(1) * pixel_l_k).sum(dim=-1)
        a_k1 = (n0.unsqueeze(1) * pixel_l_k1).sum(dim=-1)
        l_mid = 0.5 * (pixel_l_k + pixel_l_k1)
        a_mid = (n0.unsqueeze(1) * l_mid).sum(dim=-1)
        delta_a = a_k1 - a_k
        sign_p = torch.where(pixel_p_k1 >= 0, 1.0, -1.0)
        p_delta_a = sign_p * delta_a

        valid_c_hat = mask_bool & (a_k > self.c_eps) & (a_k1 > self.c_eps)
        c_hat = sign_p * torch.log(
            a_k1.clamp(min=self.c_eps) / a_k.clamp(min=self.c_eps)
        )
        c_hat = torch.where(valid_c_hat, c_hat, torch.zeros_like(c_hat))

        scene_fill = torch.full_like(c_hat, float("nan"))
        c_hat_nan = torch.where(valid_c_hat, c_hat, scene_fill)
        if valid_c_hat.any():
            scene_median = torch.nanmedian(c_hat_nan).detach()
            if not torch.isfinite(scene_median):
                scene_median = torch.zeros((), device=c_hat.device, dtype=c_hat.dtype)
        else:
            scene_median = torch.zeros((), device=c_hat.device, dtype=c_hat.dtype)
        pixel_median = torch.nanmedian(c_hat_nan, dim=1).values.detach()
        scene_expand = torch.full_like(pixel_median, scene_median)
        pixel_median = torch.where(torch.isfinite(pixel_median), pixel_median, scene_expand)

        c_hat_scene_dev = torch.where(
            valid_c_hat,
            torch.abs(c_hat - scene_median),
            torch.zeros_like(c_hat),
        )
        c_hat_pixel_dev = torch.where(
            valid_c_hat,
            torch.abs(c_hat - pixel_median.unsqueeze(1)),
            torch.zeros_like(c_hat),
        )

        valid_c_hat_f = valid_c_hat.float()
        physical_cues = torch.cat(
            [
                a_k.unsqueeze(-1),
                a_k1.unsqueeze(-1),
                a_mid.unsqueeze(-1),
                delta_a.unsqueeze(-1),
                p_delta_a.unsqueeze(-1),
                valid_c_hat_f.unsqueeze(-1),
                c_hat.unsqueeze(-1),
                c_hat_scene_dev.unsqueeze(-1),
                c_hat_pixel_dev.unsqueeze(-1),
            ],
            dim=-1,
        )
        score_in = torch.cat([pixel_feats, h_obs_padded, physical_cues], dim=-1)
        return {
            "score_in": score_in,
            "valid_c_hat": valid_c_hat_f,
            "c_hat": c_hat,
            "c_hat_scene_dev": c_hat_scene_dev,
            "c_hat_pixel_dev": c_hat_pixel_dev,
        }

    def _scorer_chunk(self, score_in, pixel_mask):
        x = self.scorer_mlp(score_in)
        for layer in self.scorer_layers:
            x = layer(x, ~(pixel_mask > 0))
        return self.scorer_head(x).squeeze(-1)

    def _score_observations(self, score_in, pixel_mask):
        if self.training and self.checkpoint_scorer:
            # Recompute stateless scorer activations during backward. Every PIE
            # is retained; the scene graph and its BatchNorm are not recomputed.
            chunk_size = int(self.scorer_pixel_chunk) or score_in.shape[0]
            logit = torch.cat([
                checkpoint(self._scorer_chunk, score_in[start:start + chunk_size],
                           pixel_mask[start:start + chunk_size], use_reentrant=False)
                for start in range(0, score_in.shape[0], chunk_size)
            ], dim=0)
            return torch.where(pixel_mask > 0, logit, torch.zeros_like(logit)), torch.sigmoid(logit) * pixel_mask
        x = self.scorer_mlp(score_in)
        key_padding_mask = ~(pixel_mask > 0)
        chunk_size = int(self.scorer_pixel_chunk) if self.scorer_pixel_chunk else 0
        if chunk_size <= 0:
            for layer in self.scorer_layers:
                x = layer(x, key_padding_mask)
        else:
            chunks = []
            for start in range(0, x.shape[0], chunk_size):
                end = min(start + chunk_size, x.shape[0])
                chunk_x = x[start:end]
                chunk_mask = key_padding_mask[start:end]
                for layer in self.scorer_layers:
                    chunk_x = layer(chunk_x, chunk_mask)
                chunks.append(chunk_x)
            x = torch.cat(chunks, dim=0)
        logit = self.scorer_head(x).squeeze(-1)
        weight = torch.sigmoid(logit) * pixel_mask
        masked_logit = torch.where(pixel_mask > 0, logit, torch.zeros_like(logit))
        return masked_logit, weight

    def forward(
        self,
        pixel_feats,
        pixel_mask,
        pixel_pos,
        pixel_t,
        pixel_l_k,
        pixel_l_k1,
        pixel_p_k1,
        return_aux=False,
    ):
        n_pix = pixel_feats.shape[0]
        encoded = self._encode_observations(pixel_feats, pixel_mask, pixel_pos, pixel_t)
        coarse_mean, coarse_max = self._aggregate_mean_max(
            encoded["h_obs"], encoded["flat_pix"], n_pix
        )
        coarse_x = self.proj(torch.cat([coarse_mean, coarse_max], dim=-1))
        coarse_x = self._pixel_gnn_forward(
            coarse_x, pixel_pos, self.pixel_convs, self.pixel_norms
        )
        coarse_n = self.normal_head(coarse_x)
        coarse_n = F.normalize(coarse_n, p=2, dim=1)
        coarse_flip = (coarse_n[:, 2] < 0).float().unsqueeze(1)
        coarse_n = coarse_n * (1 - 2 * coarse_flip)

        scoring = self._build_scoring_inputs(
            pixel_feats,
            pixel_mask,
            pixel_l_k,
            pixel_l_k1,
            pixel_p_k1,
            encoded["h_obs_padded"],
            coarse_n,
        )
        obs_logit, obs_weight = self._score_observations(
            scoring["score_in"], pixel_mask
        )

        h_gate = obs_weight.unsqueeze(-1) * encoded["h_obs_padded"]
        weight_sum = obs_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        refine_mean = h_gate.sum(dim=1) / weight_sum
        gate_mask = (pixel_mask > 0).unsqueeze(-1)
        h_gate_masked = h_gate.masked_fill(~gate_mask, -1e9)
        refine_max_raw = h_gate_masked.max(dim=1).values
        refine_max = torch.where(
            refine_max_raw > -1e8,
            refine_max_raw,
            torch.zeros_like(refine_max_raw),
        )
        refine_x = self.refine_proj(torch.cat([refine_mean, refine_max], dim=-1))
        refine_x = self._pixel_gnn_forward(
            refine_x, pixel_pos, self.refine_pixel_convs, self.refine_pixel_norms
        )
        final_n = self.refine_normal_head(refine_x)
        final_n = F.normalize(final_n, p=2, dim=1)
        final_flip = (final_n[:, 2] < 0).float().unsqueeze(1)
        final_n = final_n * (1 - 2 * final_flip)

        if not return_aux:
            return final_n

        aux = {
            "coarse_normal": coarse_n,
            "obs_weight": obs_weight,
            "obs_logit": obs_logit,
            "valid_c_hat": scoring["valid_c_hat"],
            "c_hat": scoring["c_hat"],
            "c_hat_scene_dev": scoring["c_hat_scene_dev"],
            "c_hat_pixel_dev": scoring["c_hat_pixel_dev"],
        }
        return final_n, aux
