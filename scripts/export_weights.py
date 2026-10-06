"""Export model weights and architecture metadata without training state or paths."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pieps.runtime import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, config = load_model(args.checkpoint, torch.device("cpu"))
    payload = {"format_version": 1, "model": model.state_dict(), "model_config": config,
               "training_objective": original.get("objective", original.get("training_objective", "unspecified"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
