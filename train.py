"""Jointly train all modules using final and coarse normal angular error and validation scheduling."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pieps.data import SceneDataset, model_inputs
from pieps.runtime import (OBJECTIVE, build_model, canonical_model_config, cuda_device,
                           coarse_aux_loss, final_normal_loss, new_output_dir, sha256, write_json)
from pieps.splits import load_splits
from pieps.training import (TRAINER, ValidationSchedule, capture_rng, restore_training_state,
                            validate_training_config)


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"model", "training"}:
        raise ValueError("Config must contain model and training mappings")
    config["model"] = canonical_model_config(config["model"])
    validate_training_config(config["training"])
    return config


@torch.no_grad()
def validate(model, dataset, device):
    # no_grad keeps reusable CUDA graph buffers compatible with the next training step.
    model.eval()
    errors = []
    for batch in dataset:
        pred = model(*model_inputs(batch, device))
        error = final_normal_loss(pred, batch["pixel_n_gt"].to(device), batch["pixel_valid"].to(device))
        if not torch.isfinite(error):
            raise FloatingPointError("Non-finite validation error")
        errors.append(float(error))
    model.train()
    return float(np.mean(errors))


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path, help="Resume last.pth with the same configuration and state")
    parser.add_argument("--stop-after-epoch", type=int, help="Pause after this epoch without changing the learning-rate schedule")
    args = parser.parse_args()
    config = read_config(args.config)
    tc = config["training"]
    stop = tc["epochs"] if args.stop_after_epoch is None else args.stop_after_epoch
    if not 1 <= stop <= tc["epochs"]:
        parser.error("--stop-after-epoch must be between 1 and configured epochs")
    device = cuda_device(args.device)
    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])
    torch.cuda.manual_seed_all(tc["seed"])
    # Path/content overlap is checked, but test targets and metrics are never used here.
    splits = load_splits(args.manifest, args.data_root)
    fingerprint = {split: [sha256(p) for p in paths] for split, paths in splits.items()}
    train_ds = SceneDataset(splits["train"], require_target=True, time_prefix_ratio=tc["time_prefix_ratio"])
    val_ds = SceneDataset(splits["validation"], require_target=True)
    model = build_model(config["model"], device)
    model.checkpoint_scorer = tc["activation_checkpointing"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    provenance = {"initialization": "random", "coarse_loss_weight": tc["coarse_loss_weight"]}
    start, best, best_epoch = 0, float("inf"), None
    source = None
    if args.resume:
        source = torch.load(args.resume, map_location="cpu", weights_only=False)
        if source.get("trainer") != TRAINER or source.get("objective") != OBJECTIVE:
            raise ValueError("Checkpoint uses a different training objective or trainer")
        if source["recipe"] != config or source["split_fingerprints"] != fingerprint:
            raise ValueError("Resume configuration or dataset contents differ")
        if args.resume.resolve().parent != args.output_dir.resolve() or args.resume.name != "last.pth":
            raise ValueError("Resume last.pth into its own output directory to preserve the selected best checkpoint")
        if source["best_epoch"] is not None and not (args.output_dir / "best.pth").is_file():
            raise ValueError("The selected best checkpoint is missing from the run directory")
        restore_training_state(model, optimizer, source, device)
        provenance = source["training_provenance"]
        start, best, best_epoch = source["epoch"], source["best_validation_mae"], source["best_epoch"]

    scheduler = ValidationSchedule(optimizer, tc)
    if args.resume:
        scheduler.load_state_dict(source["scheduler"])
        if scheduler.stopped:
            raise ValueError("Run has already finished by early stopping")
    else:
        new_output_dir(args.output_dir)
    del source
    if start >= stop:
        raise ValueError("The requested stopping epoch has already been completed")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Single-stage training requires all parameters to be trainable")
    write_json(args.output_dir / "run.json", {"objective": OBJECTIVE, "trainer": TRAINER,
               "initialization": provenance["initialization"], "training_provenance": provenance,
               "recipe": config, "split_fingerprints": fingerprint,
               "split_counts": {k: len(v) for k, v in splits.items()},
               "selection_metric": "validation mean scene sparse MAE (degrees)",
               "torch_version": torch.__version__, "cuda_version": torch.version.cuda})
    print(json.dumps({"objective": OBJECTIVE, "training_provenance": provenance,
                      "start_epoch": start, "stop_epoch": stop,
                      "split_counts": {k: len(v) for k, v in splits.items()}}), flush=True)

    def checkpoint_payload(completed_epochs):
        return {"format_version": 3, "trainer": TRAINER, "phase": "joint",
                "model": model.state_dict(), "model_config": config["model"], "objective": OBJECTIVE,
                "initialization": provenance["initialization"], "training_provenance": provenance,
                "recipe": config, "epoch": completed_epochs, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_validation_mae": best, "best_epoch": best_epoch,
                "split_fingerprints": fingerprint, **capture_rng(device)}

    model.train()
    for epoch in range(start, stop):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        order = list(range(len(train_ds)))
        random.shuffle(order)
        losses, coarse_losses, total_losses = [], [], []
        lr = optimizer.param_groups[0]["lr"]
        for step, index in enumerate(order, 1):
            batch = train_ds[index]
            if step <= 3:
                print(json.dumps({"epoch": epoch + 1, "step": step, "scene_index": index,
                                  "pixels": len(batch["pixel_pos"]),
                                  "observations": int(batch["pixel_mask"].sum())}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model(*model_inputs(batch, device), return_aux=True)
            loss, final_loss, coarse_loss = coarse_aux_loss(
                prediction, aux["coarse_normal"], batch["pixel_n_gt"].to(device),
                batch["pixel_valid"].to(device), tc["coarse_loss_weight"])
            del aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in training scene {index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(final_loss.detach()))
            coarse_losses.append(float(coarse_loss.detach()))
            total_losses.append(float(loss.detach()))
            if step <= 3 or step % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "steps": len(order),
                                  "train_loss_deg": float(np.mean(losses)),
                                  "train_coarse_loss_deg": float(np.mean(coarse_losses)),
                                  "train_total_loss_deg": float(np.mean(total_losses))}), flush=True)
            del prediction, loss, final_loss, coarse_loss, batch
        validation = validate(model, val_ds, device)
        improved = validation < best
        if improved:
            best, best_epoch = validation, epoch + 1
        decision = scheduler.step(validation)
        payload = checkpoint_payload(epoch + 1)
        atomic_save(payload, args.output_dir / "last.pth")
        if improved:
            atomic_save(payload, args.output_dir / "best.pth")
        del payload
        row = {"epoch": epoch + 1, "train_loss_deg": float(np.mean(losses)), "validation_mae_deg": validation,
               "train_coarse_loss_deg": float(np.mean(coarse_losses)),
               "train_total_loss_deg": float(np.mean(total_losses)),
               "phase": "joint", "learning_rate": lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started, "best_epoch": best_epoch,
               "best_validation_mae_deg": best,
               "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, **decision}
        with (args.output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        reason = ("early_stopping" if scheduler.stopped else "max_epochs" if epoch + 1 == tc["epochs"]
                  else "paused" if epoch + 1 == stop else None)
        write_json(args.output_dir / "status.json", dict(row, completed=reason in ("early_stopping", "max_epochs"),
                                                       stop_reason=reason))
        print(json.dumps(row), flush=True)
        if scheduler.stopped:
            break


if __name__ == "__main__":
    main()
