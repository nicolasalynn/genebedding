#!/usr/bin/env bash
# Borzoi (track prediction model)
# Official: https://github.com/johahi/borzoi-pytorch
# Requires torch>=2.6 for torch.library.wrap_triton (used by flash-attn 2.8+)
set -e
CONDA_ENV="${CONDA_ENV:-borzoi}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/detect_cuda.sh"
# torch>=2.6 required for torch.library.wrap_triton (cu121 maxes at 2.5)
require_cuda_min 124 "borzoi needs torch>=2.6 (only available on cu124+)"

echo "=== Borzoi env: $CONDA_ENV (CUDA_VERSION=$CUDA_VERSION) ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.6" torchvision torchaudio --index-url "$CUDA_INDEX"
pip install "borzoi-pytorch>=0.4"
# flash-attn from source (adapts to detected CUDA + torch version)
pip install flash-attn --no-build-isolation
pip install seqmat pyarrow
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
