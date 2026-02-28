#!/usr/bin/env bash
# Nucleotide Transformer (all NT v1/v2 models)
# Official: https://github.com/instadeepai/nucleotide-transformer
# Weights: HuggingFace InstaDeepAI/nucleotide-transformer-* (downloaded on first use)
set -e
CONDA_ENV="${CONDA_ENV:-nt}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Nucleotide Transformer env: $CONDA_ENV ==="

# Initialize conda (works in non-interactive shells where conda isn't in PATH)
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# torch>=2.6 required (CVE-2025-32434 blocks torch.load in older versions)
# cu121 only has torch 2.5; cu128 has 2.6+ which we need for the torch.load CVE fix
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Latest transformers supports safetensors loading natively
pip install transformers
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
