#!/usr/bin/env bash
# Nucleotide Transformer (all NT v1/v2 models)
# Official: https://github.com/instadeepai/nucleotide-transformer
# Weights: HuggingFace InstaDeepAI/nucleotide-transformer-* (downloaded on first use)
set -e
CONDA_ENV="${CONDA_ENV:-nt}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/detect_cuda.sh"
# torch>=2.6 required (CVE-2025-32434 blocks torch.load in older versions)
# cu121 only has torch 2.5; cu124+ has 2.6+
require_cuda_min 124 "NT needs torch>=2.6 for CVE-2025-32434 fix (only available on cu124+)"

echo "=== Nucleotide Transformer env: $CONDA_ENV (CUDA_VERSION=$CUDA_VERSION) ==="

# Initialize conda (works in non-interactive shells where conda isn't in PATH)
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url "$CUDA_INDEX"
# transformers <4.46 required: NT v2 remote code uses find_pruneable_heads_and_indices,
# which was removed in transformers 5.x
pip install "transformers>=4.40,<4.46"
pip install seqmat pyarrow scipy scikit-learn matplotlib pandas tqdm
pip install -e .

echo "Done. Activate with: conda activate $CONDA_ENV"
