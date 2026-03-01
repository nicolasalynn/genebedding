#!/usr/bin/env bash
# RiNALMo (RNA language model)
# Official: https://github.com/lbcb-sci/RiNALMo
# Not on PyPI - must install from GitHub. Requires flash-attn==2.3.2.
# We pin torch==2.1 to match official RiNALMo requirements and ensure
# flash-attn 2.3.2 compiles cleanly (newer torch breaks the build).
set -e
CONDA_ENV="${CONDA_ENV:-rinalmo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== RiNALMo env: $CONDA_ENV ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# torch 2.1 (official RiNALMo requirement; flash-attn 2.3.2 won't compile against newer)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# RiNALMo from GitHub (no PyPI package)
pip install "git+https://github.com/lbcb-sci/RiNALMo.git"
# flash-attn 2.3.2 (pinned; newer versions break RiNALMo's RotaryEmbedding/unpad_input API)
pip install ninja psutil
pip install flash-attn==2.3.2 --no-build-isolation
# numpy<2 required (torch 2.1 is incompatible with numpy 2.x)
pip install "numpy<2"
pip install seqmat pyarrow
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
