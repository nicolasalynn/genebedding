#!/usr/bin/env bash
# OpenSpliceAI (splice site predictor)
# Official: https://github.com/Kuanhao-Chao/OpenSpliceAI
# Models hosted at ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/
set -e
CONDA_ENV="${CONDA_ENV:-spliceai}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/detect_cuda.sh"

echo "=== OpenSpliceAI env: $CONDA_ENV (CUDA_VERSION=$CUDA_VERSION) ==="

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url "$CUDA_INDEX"
pip install openspliceai
pip install seqmat pyarrow
pip install -e .

# Download pre-trained human MANE 10000nt model (5 ensemble checkpoints, ~14MB)
MODEL_DIR="${OPENSPLICEAI_MODEL_DIR:-$HOME/openspliceai_models/OSAI-MANE/10000nt}"
mkdir -p "$MODEL_DIR"
FTP_BASE="ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/OSAI-MANE/10000nt"
for f in model_10000nt_rs10.pt model_10000nt_rs11.pt model_10000nt_rs12.pt model_10000nt_rs13.pt model_10000nt_rs14.pt; do
  if [ ! -f "$MODEL_DIR/$f" ]; then
    echo "  Downloading $f ..."
    curl -s -o "$MODEL_DIR/$f" "$FTP_BASE/$f"
  fi
done
echo "Models in: $MODEL_DIR"

echo "Done. Activate with: conda activate $CONDA_ENV"
echo "Set: export OPENSPLICEAI_MODEL_DIR=$MODEL_DIR"
