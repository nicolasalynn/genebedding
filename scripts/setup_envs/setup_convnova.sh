#!/usr/bin/env bash
# ConvNova (CNN backbone). Optional: place config/checkpoint in genebeddings/assets/convnova/.
set -e
CONDA_ENV="${CONDA_ENV:-convnova}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/detect_cuda.sh"

echo "=== ConvNova env: $CONDA_ENV (CUDA_VERSION=$CUDA_VERSION) ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "$CUDA_INDEX"
pip install "transformers>=4.30" omegaconf
pip install seqmat pyarrow
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
