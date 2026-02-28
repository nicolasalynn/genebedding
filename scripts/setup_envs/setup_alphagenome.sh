#!/usr/bin/env bash
# AlphaGenome (JAX-based). Weights: HuggingFace or Kaggle on first use.
# Requires Python 3.11+ and CUDA 12. Installs alphagenome_research from GitHub.
set -e
CONDA_ENV="${CONDA_ENV:-alphagenome}"
CLONE_DIR="${ALPHAGENOME_CLONE_DIR:-$HOME/alphagenome_research}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  AlphaGenome env: $CONDA_ENV"
echo "  Clone dir: $CLONE_DIR"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel

# JAX with CUDA 12
pip install "jax[cuda12]"

# Dependencies expected by alphagenome_research
pip install dm-haiku chex jmp orbax-checkpoint einshape "etils[epath]" \
  huggingface_hub jaxtyping kagglehub anndata pyfaidx pyranges pandas \
  pyarrow tensorflow numpy aiohttp requests fsspec alphagenome hatchling

# Clone and install alphagenome_research (no PyPI package)
if [ ! -d "$CLONE_DIR/.git" ]; then
  rm -rf "$CLONE_DIR"
  git clone https://github.com/google-deepmind/alphagenome_research.git "$CLONE_DIR"
fi
pip install -e "$CLONE_DIR"

# Genebeddings (no NT/borzoi deps needed for this env)
pip install -e .

# Smoke test (may download weights)
echo ""
echo ">>> Smoke test: AlphaGenomeWrapper"
python -c "
from genebeddings.wrappers import AlphaGenomeWrapper
w = AlphaGenomeWrapper(model_version='fold_0', source='huggingface')
e = w.embed('ACGT' * 256, pool='mean')
print(f'  embed shape: {e.shape}')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
