#!/usr/bin/env bash
# Borzoi (track prediction). Uses borzoi-pytorch; default repo johahi/flashzoi-replicate-0.
set -e
CONDA_ENV="${CONDA_ENV:-borzoi}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  Borzoi env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install "borzoi-pytorch>=0.4"
# flash-attn: pre-built wheels (no nvcc needed). Try torch2.5 then 2.4; else build from source.
FLASH_WHEEL_2_5="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_WHEEL_2_4="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if ! pip install "$FLASH_WHEEL_2_5" 2>/dev/null; then
  if ! pip install "$FLASH_WHEEL_2_4" 2>/dev/null; then
    echo "Pre-built flash-attn wheels failed; trying build from source (requires nvcc, ~5 min)..."
    pip install ninja packaging
    MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>/dev/null || true
  fi
fi
pip install -e .

echo ""
echo ">>> Smoke test: BorzoiWrapper (downloads weights on first run)"
python -c "
from genebeddings.wrappers import BorzoiWrapper
w = BorzoiWrapper(repo='johahi/flashzoi-replicate-0')
e = w.embed('ACGT' * 2000, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
