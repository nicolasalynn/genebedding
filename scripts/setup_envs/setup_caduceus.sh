#!/usr/bin/env bash
# Caduceus (Mamba-based). Weights from HuggingFace on first use.
set -e
CONDA_ENV="${CONDA_ENV:-caduceus}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  Caduceus env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
# Pin transformers: Caduceus uses tie_weights(recompute_mapping=...) removed in 4.46+
pip install "transformers>=4.30,<4.46" mamba_ssm
pip install -e .

echo ""
echo ">>> Smoke test: CaduceusWrapper (downloads from HF on first run)"
python -c "
from genebeddings.wrappers import CaduceusWrapper
w = CaduceusWrapper()
e = w.embed('ACGT' * 100, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
