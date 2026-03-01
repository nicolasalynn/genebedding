#!/usr/bin/env bash
# Run the full epistasis pipeline on a Lambda (or multi-GPU) cluster.
#
# Prereqs: clone repo, run all setup_*.sh so conda envs exist (one per wrapper).
# Set EPISTASIS_PAPER_ROOT and data/embeddings paths (or use paper_data_config default).
#
# Execution: one env at a time (one tool at a time within each). Phase 1 loops 12 conda
# envs (nt, convnova, mutbert, ..., evo2, alphagenome). Phase 2 runs metrics once.
#
# Usage:
#   cd /path/to/genebeddings
#   # Optional: verify GPU and wrappers first (one tool, 20 rows per source)
#   bash scripts/run_pipeline_cluster.sh --quick-test --env-profile nt
#   # Full run (or run embed per profile then metrics)
#   bash scripts/run_pipeline_cluster.sh 2>&1 | tee pipeline.log
#
# In another terminal, monitor progress and GPU:
#   bash scripts/monitor_pipeline.sh
#
# Options:
#   --env-profile PROFILE   Run only this profile's embed phase. Omit to run all 12.
#   --phase metrics        Run only the metrics phase (sheets). Default: embed for all profiles then metrics.
#   --skip-metrics         After embed phases, do not run metrics.
#   --quick-test           One tool, 20 rows per source; verify GPU and wrappers before full run.
#   --smoke-test           ALL tools, 5 rows per source; verify every model end-to-end (embed + metrics).
#   --smoke-test-full      ALL tools, 5 rows per source + ALL fas_exon rows; full splicing validation.
#   --model-key KEY         Run only this model key (must belong to --env-profile).
#   --sources SRC [SRC..]  Run only these source names (e.g. okgp_chr12 fas_exon).
#   --gpus N               Number of GPUs to use for parallel embed (default: auto-detect).
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
MODEL_KEY=""
SOURCES_ARGS=""
DRY_RUN=""
GPU_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env-profile)      ENV_PROFILE="$2"; shift 2 ;;
    --phase)            PHASE="$2"; shift 2 ;;
    --skip-metrics)     SKIP_METRICS=1; shift ;;
    --quick-test)       QUICK_TEST=1; shift ;;
    --smoke-test)       SMOKE_TEST=1; shift ;;
    --smoke-test-full)  SMOKE_TEST_FULL=1; shift ;;
    --model-key)        MODEL_KEY="$2"; shift 2 ;;
    --sources)          shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SOURCES_ARGS="$SOURCES_ARGS $1"; shift; done ;;
    --gpus)             GPU_OVERRIDE="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

# Source conda — try conda info first, then common install locations
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
if [ -f ~/.hf_token ]; then export HF_TOKEN=$(cat ~/.hf_token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi

# All env profiles (one per wrapper conda env). Profile name = conda env name.
ALL_PROFILES=(nt convnova mutbert hyenadna caduceus borzoi dnabert rinalmo specieslm spliceai alphagenome evo2)

# env_conda_name: profile name = conda env name (no mapping needed)
env_conda_name() { echo "$1"; }

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

# GPU detection: auto-detect via nvidia-smi, override with --gpus N
if [ -n "$GPU_OVERRIDE" ]; then
  NUM_GPUS="$GPU_OVERRIDE"
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  NUM_GPUS=${NUM_GPUS:-1}
fi
[ "$NUM_GPUS" -lt 1 ] 2>/dev/null && NUM_GPUS=1
echo "GPUs available: $NUM_GPUS"

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
  [ -n "$MODEL_KEY" ] && EMBED_EXTRA="--model-key $MODEL_KEY $EMBED_EXTRA"
  [ -n "$SOURCES_ARGS" ] && EMBED_EXTRA="--sources $SOURCES_ARGS $EMBED_EXTRA"
  if [ -n "$ENV_PROFILE" ]; then
    CONDA_NAME=$(env_conda_name "$ENV_PROFILE")
    echo "=== Embed phase: env_profile=$ENV_PROFILE (conda env: $CONDA_NAME) ${QUICK_TEST:+[quick-test]}${SMOKE_TEST:+[smoke-test]}${SMOKE_TEST_FULL:+[smoke-test-full]} ==="
    conda activate "$CONDA_NAME" || { echo "Env $CONDA_NAME not found"; exit 1; }
    run_cmd python -m notebooks.processing.run_everything --phase embed --env-profile "$ENV_PROFILE" $EMBED_EXTRA
    conda deactivate 2>/dev/null || true
  else
    # Print GPU slot assignments
    echo ""
    echo "GPU slot assignments (${#ALL_PROFILES[@]} profiles across $NUM_GPUS GPU(s)):"
    for i in "${!ALL_PROFILES[@]}"; do
      gpu=$((i % NUM_GPUS))
      echo "  GPU $gpu: ${ALL_PROFILES[$i]}"
    done
    echo ""

    if [ -n "$DRY_RUN" ]; then
      # In dry-run mode, print what would be run per GPU slot
      for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        # Skip GPUs with no profiles assigned
        GPU_HAS_WORK=0
        for i in "${!ALL_PROFILES[@]}"; do
          [ $((i % NUM_GPUS)) -eq "$gpu" ] && GPU_HAS_WORK=1 && break
        done
        [ "$GPU_HAS_WORK" -eq 0 ] && continue

        echo "[DRY-RUN] GPU $gpu:"
        for i in "${!ALL_PROFILES[@]}"; do
          if [ $((i % NUM_GPUS)) -eq "$gpu" ]; then
            profile="${ALL_PROFILES[$i]}"
            CONDA_NAME=$(env_conda_name "$profile")
            echo "  CUDA_VISIBLE_DEVICES=$gpu conda run -n $CONDA_NAME --no-banner python -m notebooks.processing.run_everything --phase embed --env-profile $profile $EMBED_EXTRA"
          fi
        done
      done

    elif [ "$NUM_GPUS" -eq 1 ]; then
      # Single GPU: sequential loop (current behavior)
      for profile in "${ALL_PROFILES[@]}"; do
        CONDA_NAME=$(env_conda_name "$profile")
        echo "=== Embed phase: env_profile=$profile (conda env: $CONDA_NAME) ${QUICK_TEST:+[quick-test]}${SMOKE_TEST:+[smoke-test]}${SMOKE_TEST_FULL:+[smoke-test-full]} ==="
        if conda activate "$CONDA_NAME" 2>/dev/null; then
          run_cmd python -m notebooks.processing.run_everything --phase embed --env-profile "$profile" $EMBED_EXTRA
          conda deactivate 2>/dev/null || true
        else
          echo "  (env $CONDA_NAME not found, skipping)"
        fi
      done

    else
      # Multi-GPU: parallel subshells with conda run
      LOG_DIR="$REPO_ROOT/logs/embed_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "$LOG_DIR"
      echo "Embed logs: $LOG_DIR"
      echo ""

      SLOT_PIDS=()
      for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        # Collect profiles assigned to this GPU slot
        GPU_HAS_WORK=0
        for i in "${!ALL_PROFILES[@]}"; do
          [ $((i % NUM_GPUS)) -eq "$gpu" ] && GPU_HAS_WORK=1 && break
        done
        [ "$GPU_HAS_WORK" -eq 0 ] && continue

        (
          set +e
          export CUDA_VISIBLE_DEVICES=$gpu
          export PIPELINE_STATUS_FILE="$LOG_DIR/gpu${gpu}_status.json"
          for i in "${!ALL_PROFILES[@]}"; do
            if [ $((i % NUM_GPUS)) -eq "$gpu" ]; then
              profile="${ALL_PROFILES[$i]}"
              CONDA_NAME=$(env_conda_name "$profile")
              echo "=== [GPU $gpu] Embed: env_profile=$profile (conda env: $CONDA_NAME) ${QUICK_TEST:+[quick-test]}${SMOKE_TEST:+[smoke-test]}${SMOKE_TEST_FULL:+[smoke-test-full]} ==="
              conda run -n "$CONDA_NAME" --no-banner \
                python -m notebooks.processing.run_everything --phase embed --env-profile "$profile" $EMBED_EXTRA \
                && echo "PROFILE_RESULT:${profile}:PASS" \
                || echo "PROFILE_RESULT:${profile}:FAIL"
            fi
          done
        ) > "$LOG_DIR/gpu${gpu}.log" 2>&1 &
        SLOT_PIDS+=($!)
      done

      echo "Waiting for ${#SLOT_PIDS[@]} GPU slots to finish..."
      SLOT_FAIL=0
      for pid in "${SLOT_PIDS[@]}"; do
        wait "$pid" || SLOT_FAIL=$((SLOT_FAIL + 1))
      done

      # Summary table
      echo ""
      echo "=== Embed Summary ==="
      printf "%-20s %-6s %-8s\n" "PROFILE" "GPU" "STATUS"
      printf "%-20s %-6s %-8s\n" "-------" "---" "------"
      for i in "${!ALL_PROFILES[@]}"; do
        profile="${ALL_PROFILES[$i]}"
        gpu=$((i % NUM_GPUS))
        logfile="$LOG_DIR/gpu${gpu}.log"
        if grep -q "PROFILE_RESULT:${profile}:PASS" "$logfile" 2>/dev/null; then
          status="PASS"
        elif grep -q "PROFILE_RESULT:${profile}:FAIL" "$logfile" 2>/dev/null; then
          status="FAIL"
        else
          status="SKIP"
        fi
        printf "%-20s %-6s %-8s\n" "$profile" "$gpu" "$status"
      done
      echo ""
      echo "Logs: $LOG_DIR"

      # Count failures
      FAIL_COUNT=$(grep -rh "PROFILE_RESULT:.*:FAIL" "$LOG_DIR"/ 2>/dev/null | wc -l)
      if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "WARNING: $FAIL_COUNT profile(s) failed. Check logs for details."
      fi
    fi
  fi
fi

# Metrics phase: cov_inv from null, recompute, save sheets
if [ "$PHASE" = "embed" ] && [ -z "$ENV_PROFILE" ]; then
  :
elif [ -n "$SKIP_METRICS" ]; then
  echo "Skipping metrics phase (--skip-metrics)"
else
  CONDA_NAME=$(env_conda_name "nt")
  echo "=== Metrics phase: cov_inv + sheets (conda env: $CONDA_NAME) ==="
  conda activate "$CONDA_NAME" || { echo "Env $CONDA_NAME not found for metrics phase"; exit 1; }
  run_cmd python -m notebooks.processing.run_everything --phase metrics
  conda deactivate 2>/dev/null || true
fi

echo "Done. Sheets in output_base/sheets/ (epistasis_metrics_<tool>.parquet)"
