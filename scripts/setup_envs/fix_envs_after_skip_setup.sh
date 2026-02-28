#!/usr/bin/env bash
# Repair all conda envs so test_all_envs.sh --skip-setup can pass.
# Run this ON the Lambda (or GPU) machine from repo root or scripts/setup_envs.
# No -q: failures will be visible. Exits on first error (set -e).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
CUDA_INDEX="https://download.pytorch.org/whl/cu121"

echo "=============================================="
echo "  Repairing envs (no --skip-setup was run)"
echo "  REPO_ROOT=$REPO_ROOT"
echo "=============================================="
if [ ! -f ~/.hf_token ]; then
  echo "WARNING: ~/.hf_token not found. NT and AlphaGenome will fail until you run: echo YOUR_TOKEN > ~/.hf_token && chmod 600 ~/.hf_token"
fi
if ! command -v git-lfs &>/dev/null; then
  echo "WARNING: git-lfs not installed. HyenaDNA in genebeddings_main will fail. Run: sudo apt-get install -y git-lfs && git lfs install"
fi
echo ""

run_in() {
  local env="$1"
  shift
  echo ">>> $env: $*"
  if ! conda activate "$env" 2>/dev/null; then
    echo "  (env $env not found, skipping)"
    return 0
  fi
  "$@"
  conda deactivate 2>/dev/null || true
}

# nt: torch>=2.6 required for torch.load CVE-2025-32434
run_in nt pip install "torch>=2.6" "torchvision" "torchaudio" --index-url "$CUDA_INDEX"

# evo2: package missing if env was never set up
run_in evo2 pip install evo2

# borzoi: torch>=2.5 for wrap_triton
run_in borzoi pip install "torch>=2.5" --index-url "$CUDA_INDEX"

# caduceus: tie_weights(recompute_mapping) removed in transformers 4.46+
run_in caduceus pip install "transformers>=4.30,<4.46" --force-reinstall

# dnabert: BertConfig.is_decoder removed in 4.46+
run_in dnabert pip install "transformers>=4.30,<4.46" --force-reinstall

# specieslm: transformers
run_in specieslm pip install "transformers>=4.30,<4.46"

# rinalmo: torch first, then flash-attn (needs --no-build-isolation to see torch), then rinalmo
run_in rinalmo bash -c 'pip install "torch>=2.0" --index-url "'"$CUDA_INDEX"'"; pip install flash-attn==2.3.2 --no-build-isolation; pip install "git+https://github.com/lbcb-sci/RiNALMo.git"'

# genebeddings_main: full stack in order, then genebeddings
run_in genebeddings_main bash -c 'cd "'"$REPO_ROOT"'" && pip install "torch>=2.6" --index-url "'"$CUDA_INDEX"'"; pip install "transformers>=4.30,<4.46" omegaconf; pip install flash-attn==2.3.2 --no-build-isolation; pip install "git+https://github.com/lbcb-sci/RiNALMo.git" borzoi-pytorch spliceai-pytorch; pip install -e .'

echo ""
echo "=============================================="
echo "  Repair done. Run tests:"
echo "  cd $REPO_ROOT && bash scripts/setup_envs/test_all_envs.sh --skip-setup"
echo "  Ensure ~/.hf_token has a valid token (NT, AlphaGenome). Install git-lfs for HyenaDNA in main."
echo "=============================================="
