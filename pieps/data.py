"""Converted PIE scenes. Targets are separate from the seven network inputs."""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_KEYS = ("pixel_feats", "pixel_mask", "pixel_pos", "pixel_t",
              "pixel_l_k", "pixel_l_k1", "pixel_p_k1")


def load_scene(path, require_target=False, time_prefix_ratio=1.0):
    if not 0 < time_prefix_ratio <= 1:
        raise ValueError("time_prefix_ratio must be in (0, 1]")
    with Path(path).open("rb") as stream:
        scene = pickle.load(stream)  # Converted scene files must come from a trusted source.
    h, w = int(scene["H"]), int(scene["W"])
    if (h, w) != (256, 256):
        raise ValueError("The supported scene geometry is 256 x 256")
    mask = np.asarray(scene["mask"])
    e = np.asarray(scene["E"])
    if mask.shape != (h, w) or e.ndim != 2 or e.shape[1] < 5 or not len(e):
        raise ValueError("Expected a nonempty E array and a 256 x 256 object mask")
    if not np.isfinite(mask).all() or not (mask > 0).any():
        raise ValueError("Object mask must be finite and nonempty")
    lights = [np.asarray(scene[k], dtype=np.float32) for k in ("L_k", "L_k1")]
    polarity = np.asarray(scene["p_k1"], dtype=np.float32)
    if any(x.shape != (len(e), 3) for x in lights) or polarity.shape != (len(e),):
        raise ValueError("Lighting and polarity must align with E")
    if not all(np.isfinite(x).all() for x in [e[:, [0, 3, 4]], *lights, polarity]):
        raise ValueError("Non-finite model input")
    if np.any(e[:, 0] != np.floor(e[:, 0])) or np.any((e[:, 0] < 0) | (e[:, 0] >= h * w)):
        raise ValueError("Invalid flattened pixel index")
    if time_prefix_ratio < 1:
        count = max(1, int(np.ceil(len(e) * time_prefix_ratio)))
        indices = np.argsort(e[:, 3], kind="mergesort")[:count]
        e, lights, polarity = e[indices], [x[indices] for x in lights], polarity[indices]
    pix = e[:, 0].astype(np.int64)
    pe, times = e[:, 4].astype(np.float32), e[:, 3].astype(np.float32)
    feats = np.concatenate([pe[:, None], *lights], axis=1).astype(np.float32)
    order = np.lexsort((times, pix))
    unique, starts, counts = np.unique(pix[order], return_index=True, return_counts=True)
    n_pix, n_obs = len(unique), int(counts.max())
    padded_feats = np.zeros((n_pix, n_obs, 7), dtype=np.float32)
    padded_mask = np.zeros((n_pix, n_obs), dtype=np.float32)
    padded_time = np.zeros_like(padded_mask)
    padded_lights = [np.zeros((n_pix, n_obs, 3), dtype=np.float32) for _ in lights]
    padded_polarity = np.zeros_like(padded_mask)
    feats, times, lights, polarity = feats[order], times[order], [x[order] for x in lights], polarity[order]
    for row, (start, count) in enumerate(zip(starts, counts)):
        end = start + count
        padded_feats[row, :count] = feats[start:end]
        padded_mask[row, :count] = 1
        padded_time[row, :count] = times[start:end]
        for target, value in zip(padded_lights, lights):
            target[row, :count] = value[start:end]
        padded_polarity[row, :count] = polarity[start:end]
    # Preserve the trained model's per-scene normalization and pixel ordering.
    valid_time = padded_time[padded_mask > 0]
    t_min, t_max = valid_time.min(), valid_time.max()
    if t_max - t_min > 1e-8:
        padded_time = (padded_time - t_min) / (t_max - t_min)
    padded_time *= padded_mask
    pe_col = padded_feats[:, :, 0]
    valid_pe = pe_col[padded_mask > 0]
    pe_min, pe_max = valid_pe.min(), valid_pe.max()
    if pe_max - pe_min > 1e-8:
        padded_feats[:, :, 0] = 2 * (pe_col - pe_min) / (pe_max - pe_min) - 1
        padded_feats[:, :, 0] *= padded_mask
    y, x = unique // w, unique % w
    positions = np.stack([x.astype(np.float32) / w, y.astype(np.float32) / h], axis=1)
    batch = dict(zip(INPUT_KEYS, map(torch.from_numpy, [
        padded_feats, padded_mask, positions, padded_time, *padded_lights, padded_polarity,
    ])))
    batch.update(pixel_idx=torch.from_numpy(unique), pixel_valid=torch.from_numpy(mask[y, x] > 0),
                 H=h, W=w, object_mask=mask, scene_name=Path(path).stem)
    if require_target and "n_gt" not in scene:
        raise ValueError("Training/evaluation requires n_gt; inference does not")
    if "n_gt" in scene:
        target = np.asarray(scene["n_gt"], dtype=np.float32)
        if target.shape != (h, w, 3):
            raise ValueError("n_gt must have shape (H, W, 3)")
        if require_target:
            normals = target[mask > 0]
            if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) < 1e-8):
                raise ValueError("Ground-truth normals must be finite and nonzero on the object mask")
        batch["pixel_n_gt"] = torch.from_numpy(target[y, x])
        batch["normal_gt"] = target
    return batch


class SceneDataset(Dataset):
    def __init__(self, paths, require_target=False, time_prefix_ratio=1.0):
        self.paths = list(map(Path, paths))
        self.require_target = require_target
        self.time_prefix_ratio = time_prefix_ratio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_scene(self.paths[index], self.require_target, self.time_prefix_ratio)


def model_inputs(batch, device):
    return [batch[key].to(device) for key in INPUT_KEYS]
