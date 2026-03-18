#!/usr/bin/env bash
# Distillation environment: PyTorch (teacher) + JAX/Haiku (student)
# Needs both frameworks since teacher runs in PyTorch and student trains in JAX.
set -e
CONDA_ENV="${CONDA_ENV:-distill}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Distillation env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo $HOME/miniconda3)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA (for teacher model inference)
CUDA_VERSION="${CUDA_VERSION:-cu121}"
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# JAX with CUDA (for student model training)
pip install "jax[cuda12]"

# Haiku + Optax (JAX neural network libraries)
pip install dm-haiku optax

# HuggingFace (model loading + progress tracking)
pip install "transformers>=4.55.0" tokenizers accelerate huggingface_hub

# Genomics dependencies
pip install seqmat pysam scipy scikit-learn numpy pandas tqdm

# Install genebeddings
pip install .

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
echo ""
echo "Usage:"
echo "  python -m scripts.distillation.distill_jax \\"
echo "      --teacher ntv3_100m_post \\"
echo "      --fasta /path/to/hg38.fa \\"
echo "      --output-dir /path/to/output \\"
echo "      --hf-repo your-username/model-name"
