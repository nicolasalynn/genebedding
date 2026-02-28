#!/usr/bin/env bash
# Combined env for all "main" profile models (everything except AlphaGenome and Evo2).
# These share compatible dependencies so they can coexist in one env.
set -e
CONDA_ENV="${CONDA_ENV:-genebeddings_main}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Main combined env: $CONDA_ENV ==="

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# transformers pinned <4.46 for caduceus/dnabert compat; ==4.29.2 would be safest for dnabert
# but caduceus needs >=4.30, so we use 4.45.x as the best compromise for the combined env
pip install "transformers>=4.30,<4.46" omegaconf einops mamba_ssm
pip install "borzoi-pytorch>=0.4" spliceai-pytorch
# RiNALMo from GitHub + flash-attn
pip install "git+https://github.com/lbcb-sci/RiNALMo.git"
pip install flash-attn --no-build-isolation
# git-lfs for HyenaDNA model download
sudo apt-get install -y git-lfs 2>/dev/null && git lfs install || true
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
