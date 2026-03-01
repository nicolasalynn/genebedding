#!/usr/bin/env bash
# Auto-detect system CUDA version and map to a PyTorch wheel suffix.
#
# Source this from any setup script:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/detect_cuda.sh"
#
# Exports:
#   CUDA_VERSION  – e.g. "121", "124", "128"
#   CUDA_INDEX    – e.g. "https://download.pytorch.org/whl/cu121"
#
# Override: CUDA_VERSION=128 bash setup_foo.sh
#
# Helpers (call AFTER sourcing):
#   require_cuda_min 124 "reason"   – exit 1 if CUDA_VERSION < 124
#   cap_cuda_max 121                – silently cap CUDA_VERSION to 121

# ── Available PyTorch CUDA wheel suffixes (ordered) ─────────────────
KNOWN_CUDA_VERSIONS=(118 121 124 128)

# ── Detection ───────────────────────────────────────────────────────
if [ -z "$CUDA_VERSION" ]; then
    # Parse "CUDA Version: X.Y" from nvidia-smi
    _DRIVER_CUDA=$(nvidia-smi 2>/dev/null \
        | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
        | head -1 \
        | grep -oE '[0-9]+\.[0-9]+')

    if [ -z "$_DRIVER_CUDA" ]; then
        echo "detect_cuda: WARNING – nvidia-smi not found or no GPU; defaulting to CUDA_VERSION=121"
        CUDA_VERSION=121
    else
        _MAJOR=$(echo "$_DRIVER_CUDA" | cut -d. -f1)
        _MINOR=$(echo "$_DRIVER_CUDA" | cut -d. -f2)
        _DRIVER_NUM=$(( _MAJOR * 10 + _MINOR ))   # e.g. 12.4 → 124

        # Pick the highest known wheel version ≤ driver version
        CUDA_VERSION=121   # safe fallback
        for _v in "${KNOWN_CUDA_VERSIONS[@]}"; do
            if [ "$_v" -le "$_DRIVER_NUM" ]; then
                CUDA_VERSION="$_v"
            fi
        done
        echo "detect_cuda: driver CUDA ${_DRIVER_CUDA} → CUDA_VERSION=${CUDA_VERSION}"
    fi
fi

export CUDA_VERSION
export CUDA_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# ── Helpers ─────────────────────────────────────────────────────────

# Enforce a minimum CUDA wheel version. Call after sourcing.
#   require_cuda_min 124 "borzoi needs torch>=2.6 (only available on cu124+)"
require_cuda_min() {
    local min_ver="$1"
    local reason="${2:-this script requires cu${min_ver}+}"
    if [ "$CUDA_VERSION" -lt "$min_ver" ]; then
        echo "detect_cuda: ERROR – CUDA_VERSION=${CUDA_VERSION} < minimum ${min_ver}."
        echo "  ${reason}"
        echo "  Either upgrade your NVIDIA driver or set CUDA_VERSION=${min_ver} manually."
        exit 1
    fi
}

# Silently cap CUDA_VERSION to a maximum (for packages with old torch pins).
#   cap_cuda_max 121   # torch==2.1.0 only has cu118/cu121 wheels
cap_cuda_max() {
    local max_ver="$1"
    if [ "$CUDA_VERSION" -gt "$max_ver" ]; then
        echo "detect_cuda: capping CUDA_VERSION ${CUDA_VERSION} → ${max_ver} (torch pin requires it)"
        CUDA_VERSION="$max_ver"
        CUDA_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
    fi
}
