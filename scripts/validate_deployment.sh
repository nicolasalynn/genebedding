#!/usr/bin/env bash
# Master deployment validation script.
# Runs 4 stages end-to-end: env creation, wrapper validation (rich table),
# short smoke test, and full fas-analysis smoke test.
#
# Usage:
#   bash scripts/validate_deployment.sh [OPTIONS]
#
# Options:
#   --skip-setup         Skip env creation (stage 1)
#   --only-wrappers      Stop after wrapper check (stages 1-2 only)
#   --skip-smoke         Skip both smoke tests (stages 1-2 only)
#   --only-short-smoke   Stop after short smoke test (stages 1-3)
#
# Environment variables (passed through to sub-scripts):
#   OPENSPLICEAI_MODEL_DIR  Path to OpenSpliceAI model weights
#   HF_TOKEN                HuggingFace token for gated models (NT, AlphaGenome)
#   EPISTASIS_PAPER_ROOT    Root for epistasis paper data paths
#   SEQMAT_DATA_DIR         Path to hg38 genome data for seqmat
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Parse arguments ──────────────────────────────────────────────────
SKIP_SETUP=false
ONLY_WRAPPERS=false
SKIP_SMOKE=false
ONLY_SHORT_SMOKE=false

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup)       SKIP_SETUP=true; shift ;;
    --only-wrappers)    ONLY_WRAPPERS=true; shift ;;
    --skip-smoke)       SKIP_SMOKE=true; shift ;;
    --only-short-smoke) ONLY_SHORT_SMOKE=true; shift ;;
    -h|--help)
      head -20 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Load credentials ─────────────────────────────────────────────────
if [ -f ~/.hf_token ]; then
  export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token)}"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ── Source conda ──────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Install conda or set CONDA_BASE."
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────
stage_header() {
  local num="$1" title="$2"
  echo ""
  echo "############################################################"
  echo "  STAGE $num: $title"
  echo "############################################################"
  echo ""
}

# ── Stage 1: Create environments ─────────────────────────────────────
if [ "$SKIP_SETUP" = true ]; then
  echo "[Stage 1] Skipped (--skip-setup)"
else
  stage_header 1 "Create conda environments"
  bash "$SCRIPT_DIR/setup_envs/test_all_envs.sh"
  echo "[Stage 1] Done."
fi

# ── Stage 2: Wrapper validation table ────────────────────────────────
stage_header 2 "Wrapper validation (rich table)"
WRAPPER_EXIT=0
bash "$SCRIPT_DIR/setup_envs/test_all_envs.sh" --skip-setup --report-params || WRAPPER_EXIT=$?

if [ "$WRAPPER_EXIT" -ne 0 ]; then
  echo ""
  echo "ERROR: Some wrappers failed (exit $WRAPPER_EXIT). Aborting — fix wrapper issues before running smoke tests."
  exit "$WRAPPER_EXIT"
fi
echo "[Stage 2] All wrappers passed."

# Stop early if requested
if [ "$ONLY_WRAPPERS" = true ] || [ "$SKIP_SMOKE" = true ]; then
  echo ""
  echo "Done (stopped after wrapper validation)."
  exit 0
fi

# ── Stage 3: Short smoke test ────────────────────────────────────────
stage_header 3 "Short smoke test (all models, 5 rows/source)"
bash "$SCRIPT_DIR/run_pipeline_cluster.sh" --smoke-test
echo "[Stage 3] Short smoke test passed."

if [ "$ONLY_SHORT_SMOKE" = true ]; then
  echo ""
  echo "Done (stopped after short smoke test)."
  exit 0
fi

# ── Stage 4: Full fas-analysis smoke test ────────────────────────────
stage_header 4 "Full fas-analysis smoke test (all models + all fas_exon rows)"
bash "$SCRIPT_DIR/run_pipeline_cluster.sh" --smoke-test-full
echo "[Stage 4] Full fas-analysis smoke test passed."

echo ""
echo "############################################################"
echo "  ALL STAGES PASSED"
echo "############################################################"
