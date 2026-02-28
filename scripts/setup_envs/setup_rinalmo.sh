#!/usr/bin/env bash
# RiNALMo (RNA language model). Install from https://github.com/lbcb-sci/RiNALMo:
#   clone, pip install ., flash-attn==2.3.2; weights via get_pretrained_model().
set -e
CONDA_ENV="${CONDA_ENV:-rinalmo}"
CUDA_VERSION="${CUDA_VERSION:-121}"
CLONE_DIR="${RINALMO_CLONE_DIR:-$HOME/RiNALMo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  RiNALMo env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION  Clone: $CLONE_DIR"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# RiNALMo: install from GitHub (no PyPI package). Per https://github.com/lbcb-sci/RiNALMo
if [ ! -d "$CLONE_DIR/.git" ]; then
  rm -rf "$CLONE_DIR"
  git clone https://github.com/lbcb-sci/RiNALMo.git "$CLONE_DIR"
fi
pip install "$CLONE_DIR"
pip install flash-attn==2.3.2

cd "$REPO_ROOT"
pip install -e .

echo ""
echo ">>> Smoke test: RiNALMoWrapper (giga-v1)"
python -c "
from genebeddings.wrappers import RiNALMoWrapper
w = RiNALMoWrapper(model_name='giga-v1')
e = w.embed('ACGT' * 50, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
