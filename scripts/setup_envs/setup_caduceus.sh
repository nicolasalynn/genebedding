#!/usr/bin/env bash
# Caduceus (Mamba-based DNA model)
# Official: https://github.com/kuleshov-group/caduceus
# Requires transformers<4.46 (tie_weights API changed in 4.46+)
set -e
CONDA_ENV="${CONDA_ENV:-caduceus}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Caduceus env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Caduceus uses tie_weights(recompute_mapping=...) removed in transformers 4.46+
pip install "transformers>=4.30,<4.46" mamba_ssm
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
