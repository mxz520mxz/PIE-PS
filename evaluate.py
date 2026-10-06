"""Evaluate a selected checkpoint on an explicit validation or test partition."""
import argparse
import json
from pathlib import Path

import numpy as np

from infer import normal_image, predict_scene
from pieps.data import SceneDataset
from pieps.runtime import cuda_device, load_model, new_output_dir, sha256, write_json
from pieps.splits import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()
    splits = load_splits(args.manifest, args.data_root)
    output = new_output_dir(args.output_dir)
    device = cuda_device(args.device)
    model, config = load_model(args.checkpoint, device)
    dataset = SceneDataset(splits[args.split], require_target=True)
    rows = []
    for index, batch in enumerate(dataset):
        raw, filled, row = predict_scene(model, batch, device, warmup=index == 0)
        # An index avoids overwriting scenes that share a basename across datasets.
        name = f"{index:05d}_{batch['scene_name']}"
        row.update(scene=name, scene_sha256=sha256(dataset.paths[index]))
        rows.append(row)
        if args.save_maps:
            np.savez_compressed(output / f"{name}.npz", pred_normal_raw=raw, pred_normal_filled=filled)
            normal_image(filled, batch["object_mask"]).save(output / f"{name}.png")
        print(json.dumps(row), flush=True)
    report = {"rows": rows, "scene_count": len(rows), "split": args.split,
              "mean_mae_raw": float(np.mean([r["mae_raw"] for r in rows])),
              "mean_mae_filled": float(np.mean([r["mae_filled"] for r in rows])),
              "checkpoint_sha256": sha256(args.checkpoint), "model_config": config,
              "time_prefix_ratio": 1.0,
              "timing_scope": "GPU forward; excludes loading, preprocessing, filling and export"}
    write_json(output / "metrics.json", report)


if __name__ == "__main__":
    main()
