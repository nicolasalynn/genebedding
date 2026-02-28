#!/usr/bin/env bash
# Run the full epistasis pipeline on a Lambda (or multi-GPU) cluster.
#
# Prereqs: clone repo, run all setup_*.sh so conda envs exist (nt, main, evo2, alphagenome, etc.).
# Set EPISTASIS_PAPER_ROOT and data/embeddings paths (or use paper_data_config default).
#
# Execution: one tool at a time per env. Phase 1 runs embed for each conda profile (main, evo2, alphagenome).
# Phase 2 runs metrics once (cov_inv from null, recompute, save one parquet per tool to output/sheets/).
#
# Usage:
#   cd /path/to/genebeddings
#   # Optional: verify GPU and wrappers first (one tool, 20 rows per source)
#   bash scripts/run_pipeline_cluster.sh --quick-test --env-profile main
#   # Full run (or run embed per profile then metrics)
#   bash scripts/run_pipeline_cluster.sh 2>&1 | tee pipeline.log
#
# In another terminal, monitor progress and GPU:
#   bash scripts/monitor_pipeline.sh
#
# Options:
#   --env-profile PROFILE   Run only this profile's embed phase (main|evo2|alphagenome). Omit to run all.
#   --phase metrics        Run only the metrics phase (sheets). Default: embed for all profiles then metrics.
#   --skip-metrics         After embed phases, do not run metrics.
#   --quick-test           One tool, 20 rows per source; verify GPU and wrappers before full run.
#   --smoke-test           ALL tools, 5 rows per source; verify every model end-to-end (embed + metrics).
#   --smoke-test-full      ALL tools, 5 rows per source + ALL fas_exon rows; full splicing validation.
#   --dry-run              Print commands, do not run.
#
# Monitor in another terminal: ./scripts/monitor_pipeline.sh
# (Uses output_base/pipeline_status.json by default; set PIPELINE_STATUS_FILE to override.)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_PROFILE=""
PHASE=""
SKIP_METRICS=""
QUICK_TEST=""
SMOKE_TEST=""
SMOKE_TEST_FULL=""
DRY_RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env-profile)      ENV_PROFILE="$2"; shift 2 ;;
    --phase)            PHASE="$2"; shift 2 ;;
    --skip-metrics)     SKIP_METRICS=1; shift ;;
    --quick-test)       QUICK_TEST=1; shift ;;
    --smoke-test)       SMOKE_TEST=1; shift ;;
    --smoke-test-full)  SMOKE_TEST_FULL=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -f ~/.hf_token ]; then export HF_TOKEN=$(cat ~/.hf_token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi

# Map env profile names to conda env names (setup scripts may use different names)
env_conda_name() {
  case "$1" in
    main) echo "${MAIN_CONDA_ENV:-genebeddings_main}" ;;
    *)    echo "$1" ;;
  esac
}

# Ensure EPISTASIS_PAPER_ROOT is set (required for output paths)
if [ -z "$EPISTASIS_PAPER_ROOT" ]; then
  echo "WARNING: EPISTASIS_PAPER_ROOT not set; using default from paper_data_config.py"
fi

# Ensure seqmat data is available (hg38 genome FASTA)
if [ -z "$SEQMAT_DATA_DIR" ]; then
  echo "WARNING: SEQMAT_DATA_DIR not set. seqmat needs hg38 genome data."
  echo "  Run once:  pip install seqmat && seqmat setup --path /path/to/data --organism hg38"
  echo "  Then set:  export SEQMAT_DATA_DIR=/path/to/data"
fi

# Status file for monitor_pipeline.sh (default: repo root; override with PIPELINE_STATUS_FILE)
if [ -z "$PIPELINE_STATUS_FILE" ]; then
  export PIPELINE_STATUS_FILE="$REPO_ROOT/pipeline_status.json"
fi

run_cmd() {
  if [ -n "$DRY_RUN" ]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

# Embed phase: one env at a time (one tool at a time within each)
if [ "$PHASE" = "metrics" ]; then
  :
else
  EMBED_EXTRA=""
  [ -n "$QUICK_TEST" ] && EMBED_EXTRA="--quick-test $EMBED_EXTRA"
  [ -n "$SMOKE_TEST" ] && EMBED_EXTRA="--smoke-test $EMBED_EXTRA"
  [ -n "$SMOKE_TEST_FULL" ] && EMBED_EXTRA="--smoke-test-full $EMBED_EXTRA"
  if [ -n "$ENV_PROFILE" ]; then
    CONDA_NAME=$(env_conda_name "$ENV_PROFILE")
    echo "=== Embed phase: env_profile=$ENV_PROFILE (conda env: $CONDA_NAME) ${QUICK_TEST:+[quick-test]}${SMOKE_TEST:+[smoke-test]}${SMOKE_TEST_FULL:+[smoke-test-full]} ==="
    conda activate "$CONDA_NAME" || { echo "Env $CONDA_NAME not found"; exit 1; }
    run_cmd python -m notebooks.processing.run_everything --phase embed --env-profile "$ENV_PROFILE" $EMBED_EXTRA
    conda deactivate 2>/dev/null || true
  else
    for profile in main evo2 alphagenome; do
      CONDA_NAME=$(env_conda_name "$profile")
      echo "=== Embed phase: env_profile=$profile (conda env: $CONDA_NAME) ${QUICK_TEST:+[quick-test]}${SMOKE_TEST:+[smoke-test]}${SMOKE_TEST_FULL:+[smoke-test-full]} ==="
      if conda activate "$CONDA_NAME" 2>/dev/null; then
        run_cmd python -m notebooks.processing.run_everything --phase embed --env-profile "$profile" $EMBED_EXTRA
        conda deactivate 2>/dev/null || true
      else
        echo "  (env $CONDA_NAME not found, skipping)"
      fi
    done
  fi
fi

# Metrics phase: cov_inv from null, recompute, save sheets
if [ "$PHASE" = "embed" ] && [ -z "$ENV_PROFILE" ]; then
  :
elif [ -n "$SKIP_METRICS" ]; then
  echo "Skipping metrics phase (--skip-metrics)"
else
  CONDA_NAME=$(env_conda_name "main")
  echo "=== Metrics phase: cov_inv + sheets (conda env: $CONDA_NAME) ==="
  conda activate "$CONDA_NAME" || { echo "Env $CONDA_NAME not found for metrics phase"; exit 1; }
  run_cmd python -m notebooks.processing.run_everything --phase metrics
  conda deactivate 2>/dev/null || true
fi

echo "Done. Sheets in output_base/sheets/ (epistasis_metrics_<tool>.parquet)"
