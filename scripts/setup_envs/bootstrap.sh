#!/usr/bin/env bash
# Bootstrap a fresh Lambda Labs (or similar) GPU instance.
# Run ONCE before any setup_*.sh scripts.
#
# Usage:
#   # Minimal (will prompt for HF token):
#   bash bootstrap.sh
#
#   # Non-interactive:
#   bash bootstrap.sh --hf-token hf_abc123... --paper-root ~/data/epistasis_paper
#
#   # Clone repo for you (HTTPS with GitHub PAT):
#   bash bootstrap.sh --hf-token hf_... --clone-repo https://github.com/nicolasalynn/dlm_wrappers.git
#
# All flags can also be set via env vars:
#   HF_TOKEN, EPISTASIS_PAPER_ROOT, REPO_URL
set -e

# ── Parse args ──────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-}"
PAPER_ROOT="${EPISTASIS_PAPER_ROOT:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --hf-token)     HF_TOKEN="$2"; shift 2 ;;
    --paper-root)   PAPER_ROOT="$2"; shift 2 ;;
    --clone-repo)   REPO_URL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Prompt for HF token if not provided ─────────────────────────────
if [ -z "$HF_TOKEN" ]; then
  echo -n "HuggingFace token (hf_...): "
  read -r HF_TOKEN
  if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HuggingFace token is required."
    exit 1
  fi
fi

# Default paper root for Lambda/cloud instances
PAPER_ROOT="${PAPER_ROOT:-$HOME/data/epistasis_paper}"

# ── tmux check ──────────────────────────────────────────────────────
if [ -z "$TMUX" ] && [ -z "$STY" ]; then
  echo ""
  echo "WARNING: You are NOT in tmux or screen."
  echo "  This script takes a long time (seqmat hg38 download, etc.)."
  echo "  If your SSH drops, the setup will be killed."
  echo ""
  echo "  Recommended: tmux new -s bootstrap"
  echo ""
  echo -n "Continue anyway? [y/N] "
  read -r yn
  case "$yn" in
    [Yy]*) ;;
    *) echo "Aborted. Run: tmux new -s bootstrap"; exit 0 ;;
  esac
fi

echo "=============================================="
echo "  Bootstrap: fresh instance setup"
echo "=============================================="

# ── 1. System packages ─────────────────────────────────────────────
echo ""
echo ">>> [1/7] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
  build-essential gcc g++ make \
  git git-lfs curl wget tmux \
  libssl-dev libffi-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncurses5-dev libgdbm-dev liblzma-dev
git lfs install

# ── 2. Miniconda ────────────────────────────────────────────────────
echo ""
echo ">>> [2/7] Installing Miniconda..."
CONDA_BASE="$HOME/miniconda3"
if [ -d "$CONDA_BASE" ]; then
  echo "  Miniconda already installed at $CONDA_BASE, skipping."
else
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_BASE"
  rm /tmp/miniconda.sh
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda init bash 2>/dev/null || true
conda config --set auto_activate_base false

# ── 3. Clone repo (optional) ───────────────────────────────────────
if [ -n "$REPO_URL" ]; then
  echo ""
  echo ">>> [3/7] Cloning repo..."
  CLONE_DEST="$HOME/genebeddings"
  if [ -d "$CLONE_DEST/.git" ]; then
    echo "  Repo already cloned at $CLONE_DEST, pulling latest..."
    git -C "$CLONE_DEST" pull
  else
    git clone "$REPO_URL" "$CLONE_DEST"
  fi
  echo "  Repo at: $CLONE_DEST"
else
  echo ""
  echo ">>> [3/7] Skipping repo clone (no --clone-repo). Assuming repo already cloned."
fi

# ── 4. HuggingFace token ───────────────────────────────────────────
echo ""
echo ">>> [4/7] Setting up HuggingFace token..."
echo "$HF_TOKEN" > ~/.hf_token
chmod 600 ~/.hf_token

# Helper: add/replace an export line in ~/.bashrc
_bashrc_set() {
  local var="$1" val="$2"
  sed -i "/^export ${var}=/d" ~/.bashrc 2>/dev/null || true
  echo "export ${var}=${val}" >> ~/.bashrc
}

_bashrc_set HF_TOKEN "$HF_TOKEN"
_bashrc_set HUGGING_FACE_HUB_TOKEN "$HF_TOKEN"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── 5. EPISTASIS_PAPER_ROOT + OPENSPLICEAI_MODEL_DIR ────────────────
echo ""
echo ">>> [5/7] Setting up project env vars..."

_bashrc_set EPISTASIS_PAPER_ROOT "$PAPER_ROOT"
export EPISTASIS_PAPER_ROOT="$PAPER_ROOT"
mkdir -p "$PAPER_ROOT"
echo "  EPISTASIS_PAPER_ROOT=$PAPER_ROOT"

OPENSPLICEAI_MODEL_DIR="$HOME/openspliceai_models/OSAI-MANE/10000nt"
_bashrc_set OPENSPLICEAI_MODEL_DIR "$OPENSPLICEAI_MODEL_DIR"
export OPENSPLICEAI_MODEL_DIR
echo "  OPENSPLICEAI_MODEL_DIR=$OPENSPLICEAI_MODEL_DIR"

# ── 6. Seqmat + hg38 genome data ───────────────────────────────────
echo ""
echo ">>> [6/7] Installing seqmat and downloading hg38 genome (this takes a while)..."
SEQMAT_DATA_DIR="$HOME/data/seqmat"

# Temporary conda env just for the data download, cleaned up after
conda create -n _seqmat_tmp python=3.10 -y -q
conda activate _seqmat_tmp
pip install -q --upgrade pip seqmat
seqmat setup --path "$SEQMAT_DATA_DIR" --organism hg38 --force
conda deactivate
conda env remove -n _seqmat_tmp -y

_bashrc_set SEQMAT_DATA_DIR "$SEQMAT_DATA_DIR"
export SEQMAT_DATA_DIR
echo "  SEQMAT_DATA_DIR=$SEQMAT_DATA_DIR"

# ── 7. GPU info + summary ──────────────────────────────────────────
echo ""
echo ">>> [7/7] System info"
echo "=============================================="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
nvidia-smi 2>/dev/null | grep "CUDA Version" || true
echo ""
echo "conda:  $(conda --version)"
echo "gcc:    $(gcc --version | head -1)"
echo ""

# Show what detect_cuda.sh would pick
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/detect_cuda.sh" ]; then
  source "$SCRIPT_DIR/detect_cuda.sh"
fi

echo ""
echo "=============================================="
echo "  Env vars persisted to ~/.bashrc:"
echo "=============================================="
echo "  HF_TOKEN=hf_...$(echo "$HF_TOKEN" | tail -c 5)"
echo "  HUGGING_FACE_HUB_TOKEN=(same)"
echo "  SEQMAT_DATA_DIR=$SEQMAT_DATA_DIR"
echo "  EPISTASIS_PAPER_ROOT=$PAPER_ROOT"
echo "  OPENSPLICEAI_MODEL_DIR=$OPENSPLICEAI_MODEL_DIR"
echo ""
echo "=============================================="
echo "  Bootstrap done. Next steps:"
echo "=============================================="
echo "  source ~/.bashrc"
if [ -n "$REPO_URL" ]; then
  echo "  cd ~/genebeddings"
fi
echo "  # Create all envs + validate (takes 1-2 hrs):"
echo "  tmux new -s setup"
echo "  bash scripts/validate_deployment.sh"
echo ""
echo "  # Or just create envs:"
echo "  bash scripts/setup_envs/test_all_envs.sh"
echo ""
echo "  # Then run the pipeline:"
echo "  bash scripts/run_pipeline_cluster.sh"
echo "=============================================="
