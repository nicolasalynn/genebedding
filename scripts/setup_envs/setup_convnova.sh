#!/usr/bin/env bash
# ConvNova (CNN backbone). Optional: place config/checkpoint in genebeddings/assets/convnova/.
set -e
CONDA_ENV="${CONDA_ENV:-convnova}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  ConvNova env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install "transformers>=4.30" omegaconf
pip install -e .

echo ""
echo ">>> Optional: add genebeddings/assets/convnova/convnova.yaml and last.backbone.pth for pretrained weights."
python -c "
from genebeddings.wrappers import ConvNovaWrapper
w = ConvNovaWrapper()
e = w.embed('ACGT' * 50, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
