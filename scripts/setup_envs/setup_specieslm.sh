#!/usr/bin/env bash
# SpeciesLM (gagneurlab). Weights from HuggingFace on first use.
set -e
CONDA_ENV="${CONDA_ENV:-specieslm}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== SpeciesLM env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.30,<4.46"
pip install seqmat pyarrow
pip install -e .

echo ""
echo ">>> Smoke test: SpeciesLMWrapper"
python -c "
from genebeddings.wrappers import SpeciesLMWrapper
w = SpeciesLMWrapper()
e = w.embed('ACGT' * 50, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
