"""Model configuration, checkpoint loading and the single training objective."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

DEFAULT_MODEL = dict(height=256, width=256, enc_dim=64, pixel_gnn_layers=5,
                     pixel_k=8, time_scale=0.3, scorer_hidden_dim=64,
                     scorer_num_heads=4, scorer_num_layers=2, scorer_pixel_chunk=1024,
                     radius=0.05, max_neighbors=8, base_width=0.5, activation="relu",
                     edge_attr_dim=2, aggr="sum", kernel_size=5, c_eps=1e-6)
OBJECTIVE = "final_plus_coarse_normal_angular_degrees"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_model_config(config):
    unknown = set(config) - set(DEFAULT_MODEL)
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    result = dict(DEFAULT_MODEL, **config)
    for key, default in DEFAULT_MODEL.items():
        value = result[key]
        if isinstance(default, int) and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{key} must be a positive integer")
        if isinstance(default, float) and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if (result["height"], result["width"], result["edge_attr_dim"]) != (256, 256, 2):
        raise ValueError("Supported geometry is 256 x 256 with 2-D edge attributes")
    if result["pixel_gnn_layers"] < 2 or result["scorer_hidden_dim"] % result["scorer_num_heads"]:
        raise ValueError("Invalid GNN depth or attention head dimensions")
    if result["activation"] != "relu" or result["aggr"] != "sum":
        raise ValueError("The supported graph layer uses relu activation and sum aggregation")
    return result


def build_model(config, device):
    from pieps.model import PIEPS
    config = canonical_model_config(config)
    args = SimpleNamespace(**config)
    model_keys = ("height", "width", "enc_dim", "pixel_gnn_layers", "time_scale",
                  "scorer_hidden_dim", "scorer_num_heads", "scorer_num_layers",
                  "scorer_pixel_chunk", "c_eps")
    return PIEPS(args, feat_dim=7, gnn_hid=config["enc_dim"] * 2, k=config["pixel_k"],
                 **{key: config[key] for key in model_keys}).to(device)


def load_model(path, device):
    # Checkpoints, like converted pickle scenes, must come from a trusted source.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" in payload:
        config = canonical_model_config(payload["model_config"])
    else:
        # Only import the model dimensions from compatible existing checkpoints.
        previous = payload.get("config", {})
        if previous.get("feature", "raw") != "raw":
            raise ValueError("Only raw PIE feature checkpoints are supported")
        config = canonical_model_config({k: v for k, v in previous.items() if k in DEFAULT_MODEL})
    model = build_model(config, device)
    # Ignore unused dense-pooling parameters in compatible checkpoints.
    unused = {f"conv_block1.{block}.conv.{suffix}"
              for block in ("conv_block1", "conv_block2")
              for suffix in ("attn_key.weight", "attn_key.bias", "attn_query")}
    state = {key: value for key, value in payload["model"].items() if key not in unused}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def final_normal_loss(pred, target, valid):
    valid = valid.bool()
    if not valid.any():
        raise ValueError("No valid supervised pixels in this scene")
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosine = (pred * target).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.acos(cosine[valid]) * (180.0 / torch.pi)).mean()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def new_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cuda_device(name):
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The event-graph implementation requires an NVIDIA CUDA GPU")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    return device


def coarse_aux_loss(prediction, coarse, target, valid, weight):
    final = final_normal_loss(prediction, target, valid)
    guide = final_normal_loss(coarse, target, valid)
    return final + weight * guide, final, guide
