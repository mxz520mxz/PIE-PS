"""Predict normal maps from converted observations and calibrated lighting."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pieps.data import load_scene, model_inputs
from pieps.metrics import build_dense_pred_map, dense_angle_mae, fill_mask_holes
from pieps.runtime import cuda_device, load_model, new_output_dir, sha256, write_json


def normal_image(normal, mask):
    rgb = np.clip((normal + 1) * 127.5, 0, 255).astype(np.uint8)
    rgb[np.asarray(mask) <= 0] = 0
    return Image.fromarray(rgb)


@torch.no_grad()
def predict_scene(model, batch, device, warmup=False):
    inputs = model_inputs(batch, device)
    if warmup:
        model(*inputs)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    pred = model(*inputs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite predicted normals")
    raw = build_dense_pred_map(pred, batch["pixel_idx"].numpy(), batch["H"], batch["W"])
    filled = fill_mask_holes(raw, batch["object_mask"])
    report = {"pixels_with_observations": len(pred), "observations": int(batch["pixel_mask"].sum()),
              "forward_runtime_sec": elapsed,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}
    if "normal_gt" in batch:
        report["mae_raw"] = dense_angle_mae(raw, batch["normal_gt"], batch["object_mask"])
        report["mae_filled"] = dense_angle_mae(filled, batch["normal_gt"], batch["object_mask"])
    return raw, filled, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-prefix-ratio", type=float, default=1.0)
    args = parser.parse_args()
    output = new_output_dir(args.output_dir)
    device = cuda_device(args.device)
    model, config = load_model(args.checkpoint, device)
    batch = load_scene(args.scene, time_prefix_ratio=args.time_prefix_ratio)
    raw, filled, report = predict_scene(model, batch, device, warmup=True)
    np.savez_compressed(output / "normals.npz", pred_normal_raw=raw, pred_normal_filled=filled,
                        mask=batch["object_mask"], pixel_idx=batch["pixel_idx"].numpy())
    normal_image(raw, batch["object_mask"]).save(output / "normal_raw.png")
    normal_image(filled, batch["object_mask"]).save(output / "normal_filled.png")
    if "normal_gt" in batch:
        normal_image(batch["normal_gt"], batch["object_mask"]).save(output / "normal_gt.png")
    report.update(scene=args.scene.name, checkpoint_sha256=sha256(args.checkpoint),
                  model_config=config, time_prefix_ratio=args.time_prefix_ratio,
                  timing_scope="one warmed GPU forward; excludes loading, preprocessing, filling and export")
    write_json(output / "metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
