"""Explicit dataset partitions; no benchmark-based checkpoint selection."""
import json
from pathlib import Path

from pieps.runtime import sha256


def load_splits(manifest, data_root, verify_hashes=True):
    manifest = json.loads(Path(manifest).read_text())
    if set(manifest) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain exactly train, validation and test lists")
    root = Path(data_root).resolve()
    result, paths_seen, hashes_seen = {}, {}, {}
    for split, names in manifest.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f"{split} must be a nonempty list")
        result[split] = []
        for name in names:
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Scene paths must be relative to data_root")
            path = (root / name).resolve()
            if not path.is_file() or path.suffix != ".pkl":
                raise ValueError(f"Missing converted scene: {name}")
            if path in paths_seen:
                raise ValueError(f"Duplicate scene path in {paths_seen[path]} and {split}: {name}")
            paths_seen[path] = split
            if verify_hashes:
                digest = sha256(path)
                if digest in hashes_seen:
                    raise ValueError(f"Duplicate scene content: {name} and {hashes_seen[digest]}")
                hashes_seen[digest] = name
            result[split].append(path)
    return result
