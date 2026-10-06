#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install --no-index torch-scatter==2.1.2 torch-cluster==1.6.3 torch-spline-conv==1.2.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
"$PYTHON" -m pip install --no-build-isolation --no-deps -e .
"$PYTHON" -m pip check
