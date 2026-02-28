#!/usr/bin/env bash
# Borzoi (track prediction model)
# Official: https://github.com/johahi/borzoi-pytorch
# Requires torch>=2.6 for torch.library.wrap_triton (used by flash-attn 2.8+)
set -e
CONDA_ENV="${CONDA_ENV:-borzoi}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Borzoi env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# cu124 has torch 2.6 (cu121 maxes at 2.5 which lacks wrap_triton)
pip install "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "borzoi-pytorch>=0.4"
# flash-attn pre-built wheel (must match torch version + ABI)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
