#!/usr/bin/env bash
# Nucleotide Transformer v3 (NTv3) by InstaDeep
# Official: https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3
# U-Net architecture, single-base tokenization, up to 1Mb context
# Requires transformers >= 4.55.0 and trust_remote_code=True
set -e
CONDA_ENV="${CONDA_ENV:-ntv3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== NTv3 env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo $HOME/miniconda3)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA
CUDA_VERSION="${CUDA_VERSION:-cu121}"
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# NTv3 requires transformers >= 4.55.0 for custom code support
pip install "transformers>=4.55.0" tokenizers accelerate

# Standard dependencies for genebeddings
pip install seqmat scipy scikit-learn pyarrow matplotlib pandas tqdm numpy

# Install genebeddings (non-editable)
pip install .

echo "Done. Activate with: conda activate $CONDA_ENV"
echo "Test with: python -c \"from genebeddings.wrappers import NTv3Wrapper; print('OK')\""
