#!/usr/bin/env bash
# Create a GPU-suffixed conda env from any setup script and register it as a Jupyter kernel.
#
# Usage:
#   bash setup_gpu_env.sh <setup_script> <gpu_suffix>
#
# Examples:
#   bash setup_gpu_env.sh setup_borzoi.sh h100
#   bash setup_gpu_env.sh setup_caduceus.sh a100
#   bash setup_gpu_env.sh setup_evo2.sh h100
#
# This creates e.g. "borzoi_h100" env and registers a "Py (borzoi_h100)" Jupyter kernel.
# Kernel is installed to KERNEL_PREFIX (default: $HOME/jupyter-kernels) for use with
# custom JUPYTER_PATH setups, and also --user as fallback.

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <setup_script> <gpu_suffix>"
    echo "  e.g.: $0 setup_borzoi.sh h100"
    exit 1
fi

SETUP_SCRIPT="$1"
GPU_SUFFIX="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve setup script path
if [ ! -f "$SETUP_SCRIPT" ]; then
    SETUP_SCRIPT="$SCRIPT_DIR/$SETUP_SCRIPT"
fi
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "ERROR: Setup script not found: $1"
    exit 1
fi

# Extract default env name from the script (e.g. CONDA_ENV="${CONDA_ENV:-borzoi}")
DEFAULT_ENV=$(grep -oP 'CONDA_ENV="\$\{CONDA_ENV:-\K[^}]+' "$SETUP_SCRIPT")
if [ -z "$DEFAULT_ENV" ]; then
    echo "ERROR: Could not detect default CONDA_ENV from $SETUP_SCRIPT"
    exit 1
fi

export CONDA_ENV="${DEFAULT_ENV}_${GPU_SUFFIX}"
echo "=== Creating env: $CONDA_ENV (from $SETUP_SCRIPT) ==="

# Run the original setup script
bash "$SETUP_SCRIPT"

# Register Jupyter kernel
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install -q ipykernel

KERNEL_PREFIX="${KERNEL_PREFIX:-$HOME/jupyter-kernels}"
python -m ipykernel install \
    --prefix="$KERNEL_PREFIX" \
    --name="$CONDA_ENV" \
    --display-name="Py ($CONDA_ENV)"

# Also install --user as fallback
python -m ipykernel install \
    --user \
    --name="$CONDA_ENV" \
    --display-name="Py ($CONDA_ENV)"

echo "=== Done: $CONDA_ENV ==="
echo "Kernel registered as 'Py ($CONDA_ENV)'"
echo "Activate with: conda activate $CONDA_ENV"
