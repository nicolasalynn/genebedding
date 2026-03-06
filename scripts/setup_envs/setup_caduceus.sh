#!/usr/bin/env bash
# Caduceus (Mamba-based DNA model)
# Official: https://github.com/kuleshov-group/caduceus
# Requires transformers<4.46 (tie_weights API changed in 4.46+)
set -e
CONDA_ENV="${CONDA_ENV:-caduceus}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/detect_cuda.sh"

echo "=== Caduceus env: $CONDA_ENV (CUDA_VERSION=$CUDA_VERSION) ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel

# mamba_ssm compiles CUDA kernels → needs nvcc + matching toolkit
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$CUDA_INDEX"

# Caduceus uses tie_weights(recompute_mapping=...) removed in transformers 4.46+
pip install "transformers>=4.30,<4.46"

# Build mamba_ssm with the installed torch (not pip's isolated env torch)
pip install mamba_ssm --no-build-isolation
pip install seqmat pyarrow scipy scikit-learn matplotlib pandas tqdm
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
