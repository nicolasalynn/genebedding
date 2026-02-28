#!/usr/bin/env bash
# SpliceAI / OpenSpliceAI. Uses spliceai-pytorch or OpenSpliceAI; set OPENSPLICEAI_MODEL_DIR for weights.
set -e
CONDA_ENV="${CONDA_ENV:-spliceai}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  SpliceAI env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install spliceai-pytorch
pip install -e .

echo ""
echo ">>> Note: set OPENSPLICEAI_MODEL_DIR to a dir with OpenSpliceAI checkpoints (e.g. openspliceai-mane/10000nt)"
echo ">>> Or use SpliceAIWrapper with model_dir=... when calling. Weights are not auto-downloaded."
python -c "
from genebeddings.wrappers import SpliceAIWrapper
print('  SpliceAIWrapper import OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
