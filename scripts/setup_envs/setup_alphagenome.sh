#!/usr/bin/env bash
# AlphaGenome (JAX-based DNA foundation model by Google DeepMind)
# Official: https://github.com/google-deepmind/alphagenome
# Requires Python 3.11+, JAX with CUDA 12
set -e
CONDA_ENV="${CONDA_ENV:-alphagenome}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== AlphaGenome env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# JAX with CUDA 12
pip install "jax[cuda12]"
# AlphaGenome and its dependencies
pip install alphagenome
pip install dm-haiku chex jmp orbax-checkpoint einshape "etils[epath]" \
  huggingface_hub jaxtyping kagglehub anndata pyfaidx pyranges pandas \
  pyarrow numpy aiohttp requests fsspec hatchling
# Clone and install alphagenome_research (for the model apply functions)
CLONE_DIR="${ALPHAGENOME_CLONE_DIR:-$HOME/alphagenome_research}"
if [ ! -d "$CLONE_DIR/.git" ]; then
  rm -rf "$CLONE_DIR"
  git clone https://github.com/google-deepmind/alphagenome_research.git "$CLONE_DIR"
fi
pip install "$CLONE_DIR"
# Non-editable install so genebeddings is importable
pip install seqmat scipy scikit-learn pyarrow
pip install .

echo "Done. Activate with: conda activate $CONDA_ENV"
