#!/usr/bin/env bash
# DNABERT-2 (zhihan1996/DNABERT-2-117M)
# Official: https://github.com/MAGICS-LAB/DNABERT_2
# Requires trust_remote_code=True; pinned transformers to avoid config_class mismatch
set -e
CONDA_ENV="${CONDA_ENV:-dnabert}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== DNABERT-2 env: $CONDA_ENV ==="

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# DNABERT-2 requires transformers==4.29.2 to avoid config_class mismatch
# See: https://github.com/MAGICS-LAB/DNABERT_2/issues/38
pip install "transformers==4.29.2" einops
# Uninstall triton: DNABERT-2's bundled Triton flash-attn is incompatible with modern triton.
# Without triton, it auto-falls-back to standard PyTorch attention.
pip uninstall triton -y 2>/dev/null || true
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
