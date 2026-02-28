#!/usr/bin/env bash
# Nucleotide Transformer family (all NT v1/v2 models: nt50_3mer, nt50_multi, ..., nt2500_okgp, v2-*)
# Weights: HuggingFace InstaDeepAI/nucleotide-transformer-* (downloaded on first use)
set -e
CONDA_ENV="${CONDA_ENV:-nt}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  Nucleotide Transformer env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION  Repo: $REPO_ROOT"
echo "============================================"

# Create conda env (Python 3.10)
conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# PyTorch with CUDA
pip install --upgrade pip setuptools wheel
pip install "torch>=2.0" "torchvision" "torchaudio" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# Transformers + genebeddings[nt]
pip install "transformers>=4.30"
pip install -e ".[nt]"

# Smoke test (downloads HF model on first run)
echo ""
echo ">>> Smoke test: NTWrapper(nt500_multi)"
python -c "
from genebeddings.wrappers import NTWrapper
w = NTWrapper(model='nt500_multi')
e = w.embed('ACGT' * 100, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
