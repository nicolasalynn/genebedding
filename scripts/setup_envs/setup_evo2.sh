#!/usr/bin/env bash
# Evo2 (7b_base, 7b, 1b). Weights handled by evo2 package.
set -e
CONDA_ENV="${CONDA_ENV:-evo2}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  Evo2 env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install evo2
pip install -e .

echo ""
echo ">>> Smoke test: Evo2Wrapper(7b_base)"
python -c "
from genebeddings.wrappers import Evo2Wrapper
w = Evo2Wrapper(model='7b_base')
e = w.embed('ACGT' * 50, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
