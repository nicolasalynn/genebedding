#!/usr/bin/env bash
# HyenaDNA (LongSafari). Weights from HuggingFace on first use.
set -e
CONDA_ENV="${CONDA_ENV:-hyenadna}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== HyenaDNA env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install "transformers>=4.30"
pip install seqmat pyarrow
pip install -e .

echo ""
echo ">>> Smoke test: HyenaDNAWrapper (medium-160k; may download from HF)"
python -c "
from genebeddings.wrappers import HyenaDNAWrapper
w = HyenaDNAWrapper(model='medium-160k')
e = w.embed('ACGT' * 100, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
