#!/usr/bin/env bash
# Evo2 (7b_base, 7b, 1b) by Arc Institute
# Official: https://github.com/ArcInstitute/evo2
# Requires Python 3.12, transformer-engine, flash-attn
set -e
CONDA_ENV="${CONDA_ENV:-evo2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Evo2 env: $CONDA_ENV ==="

# Evo2 requires Python 3.12
conda create -n "$CONDA_ENV" python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# Install torch first (cu128 gives latest; conda-forge transformer-engine may downgrade)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# transformer-engine from conda-forge (avoids empty meta-package from PyPI)
conda install -c conda-forge transformer-engine-torch=2.3.0 -y
# flash-attn (build from source with psutil/ninja for speed)
pip install psutil ninja
pip install flash-attn --no-build-isolation
# evo2 + core deps
pip install evo2 matplotlib scipy scikit-learn pandas tqdm
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
