# PIE-PS

Photometric Stereo from Physical Irradiance Event Streams.

The model encodes PIE observations, estimates coarse normals, scores observations
with a Transformer, and predicts refined normals from weighted features.

## Installation

Requirements: Linux, Python 3.12, an NVIDIA GPU, and the CUDA 11.8 development toolkit.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
CUDA_HOME=/usr/local/cuda-11.8 bash scripts/install.sh
```

The installation script builds the event-graph CUDA extension and installs
PyTorch 2.4.1 and the required PyTorch Geometric extensions.

## Data

Download [PIE-PS-data-20261006.tar.gz](https://drive.google.com/file/d/1zBo8sxxHnNkPsg99rUcTNHhzJhi_t2XB/view) from Google Drive.

Extract `PIE-PS-data-20261006.tar.gz` beside the source directory. The archive
contains only scene files under `PIE-PS-data/dataset/`:

```text
dataset/
  train/       103 scenes
  validation/   23 scenes
  test/         42 scenes
  benchmark/     6 additional resimulated scenes
```

[configs/splits.json](configs/splits.json) defines the training, validation, and
test partitions, with paths relative to the dataset directory. The evaluation
command below uses the 42 test scenes.

For custom scenes, follow [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) and provide a
manifest with `train`, `validation`, and `test` lists. Define partitions by object
or capture sequence. Repeated paths and byte-identical files are rejected.
Load scene files and checkpoints from trusted sources.

## Training

```bash
python train.py --config configs/train.yaml --manifest configs/splits.json \
  --data-root ../PIE-PS-data/dataset --output-dir outputs/train
```

All modules train jointly from random initialization. The objective is final-normal
angular error plus `0.3` times coarse-normal angular error; the coefficient is
configured by `coarse_loss_weight`. Both terms use ground-truth normals on valid pixels.

The default configuration uses Adam, a maximum of 400 epochs, an initial learning
rate of `1e-4`, and a minimum of `1e-6`. A validation-driven scheduler reduces the
learning rate; early stopping applies after 40 epochs without sufficient improvement
at the minimum learning rate. One complete scene is used per optimizer step.

`best.pth` is selected using final-normal validation MAE. `last.pth` contains the
model, optimizer, scheduler, and random states. Resume with the same configuration:

```bash
python train.py --config configs/train.yaml --manifest configs/splits.json \
  --data-root ../PIE-PS-data/dataset --output-dir outputs/train \
  --resume outputs/train/last.pth
```

## Inference and evaluation

```bash
python infer.py --checkpoint outputs/train/best.pth \
  --scene ../PIE-PS-data/dataset/test/piesim_non_circle/blob_000000.pkl \
  --output-dir outputs/inference

python evaluate.py --checkpoint outputs/train/best.pth \
  --manifest configs/splits.json --data-root ../PIE-PS-data/dataset --split test \
  --output-dir outputs/test --save-maps
```

Inference returns raw and filled normal maps. Ground-truth normals are optional for
inference and required for evaluation. Evaluation reports scene-averaged angular
error over the object mask. Validation measures observed object pixels; filled
normals interpolate pixels without predictions.

## Tests

```bash
python -m unittest discover -s tests -v
```

To export model weights without optimizer state or run metadata:

```bash
python scripts/export_weights.py --checkpoint outputs/train/best.pth \
  --output weights/pieps.pth
```
