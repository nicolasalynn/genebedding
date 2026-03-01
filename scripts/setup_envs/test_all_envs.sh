#!/usr/bin/env bash
# 1. Create each environment (run setup script), 2. Activate it, 3. Run wrapper test on a test sequence.
# Outputs a report showing which envs/wrappers worked.
# Use --skip-setup to only run tests in existing envs (faster for iteration).
# Use --report-params to output enriched table with embedding dim, param count, context length.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_SETUP=false
REPORT_PARAMS=false
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=true; shift ;;
    --report-params) REPORT_PARAMS=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Load HF token so NT/AlphaGenome and gated models work when run directly on Lambda
if [ -f ~/.hf_token ]; then export HF_TOKEN=$(cat ~/.hf_token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi

# Initialize conda (works in non-interactive shells)
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Report file (append each result)
REPORT_FILE="${REPORT_FILE:-$SCRIPT_DIR/test_all_envs_report.txt}"
LOG_DIR="${SETUP_LOG_DIR:-$SCRIPT_DIR/setup_logs}"
mkdir -p "$LOG_DIR"
echo "ENV|WRAPPER|MODEL|STATUS|DETAIL" > "$REPORT_FILE"

# Each row: "setup_script|env_name|wrapper_key[|model_key1|model_key2|...]"
# If model keys are present, wrapper is tested once per model key (--model <key>).
# If no model keys, wrapper is tested once with its default model.
ENVS=(
  "setup_nucleotide_transformer.sh|nt|nt|nt50_3mer|nt50_multi|nt100_multi|nt250_multi|nt500_multi|nt500_ref|nt2500_multi|nt2500_okgp"
  "setup_alphagenome.sh|alphagenome|alphagenome"
  "setup_evo2.sh|evo2|evo2"
  "setup_spliceai.sh|spliceai|spliceai"
  "setup_convnova.sh|convnova|convnova"
  "setup_borzoi.sh|borzoi|borzoi"
  "setup_mutbert.sh|mutbert|mutbert"
  "setup_hyenadna.sh|hyenadna|hyenadna"
  "setup_caduceus.sh|caduceus|caduceus"
  "setup_dnabert.sh|dnabert|dnabert"
  "setup_rinalmo.sh|rinalmo|rinalmo"
  "setup_specieslm.sh|specieslm|specieslm"
)

# Collect TSV lines for rich table (report-params mode)
TSV_LINES=()

run_one_test() {
  local env_name="$1"
  local wrapper_key="$2"
  local model_key="$3"    # may be empty
  local extra_args=""

  if [ -n "$model_key" ]; then
    extra_args="--model $model_key"
  fi
  if [ "$REPORT_PARAMS" = true ]; then
    extra_args="$extra_args --report-params"
  fi

  local stdout_file stderr_file
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)

  python "$SCRIPT_DIR/run_wrapper_test.py" --wrapper "$wrapper_key" $extra_args \
    >"$stdout_file" 2>"$stderr_file" || true

  local stdout_content stderr_content
  stdout_content=$(cat "$stdout_file")
  stderr_content=$(cat "$stderr_file")
  rm -f "$stdout_file" "$stderr_file"

  # Display key for this test
  local display_key="${model_key:-$wrapper_key}"

  if [ "$REPORT_PARAMS" = true ]; then
    # Last line of stdout is our TSV; earlier lines may be model-loading logs (e.g. evo2)
    if [ -n "$stdout_content" ]; then
      echo "$stdout_content" | tail -1
    else
      # No stdout means crash before any output
      printf '%s\tFAIL\tN/A\tN/A\tN/A\n' "$display_key"
    fi
  else
    # Legacy mode: last line of stdout has "key OK shape=..."
    local last_line
    last_line=$(echo "$stdout_content" | tail -1)
    if echo "$last_line" | grep -q "OK"; then
      echo "OK|${last_line}"
    else
      echo "FAIL|${stderr_content}"
    fi
  fi
}

echo "=============================================="
echo "  test_all_envs: create envs + run wrapper tests"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  SKIP_SETUP=$SKIP_SETUP  REPORT_PARAMS=$REPORT_PARAMS"
echo "=============================================="

TOTAL=0
PASSED=0

for row in "${ENVS[@]}"; do
  # Parse: setup_script|env_name|wrapper_key[|model1|model2|...]
  IFS='|' read -ra fields <<< "$row"
  setup_script="${fields[0]}"
  env_name="${fields[1]}"
  wrapper_key="${fields[2]}"
  model_keys=("${fields[@]:3}")  # remaining fields are model keys (may be empty)

  echo ""
  echo "--- Environment: $env_name ---"

  if [ "$SKIP_SETUP" = false ]; then
    echo "  Running $setup_script ..."
    SETUP_LOG="$LOG_DIR/setup_${env_name}.log"
    if ! bash "$SCRIPT_DIR/$setup_script" 2>&1 | tee "$SETUP_LOG"; then
      tail -100 "$SETUP_LOG" > "$LOG_DIR/setup_${env_name}_last100.txt"
      echo "  Setup failed for $env_name (last 100 lines in $LOG_DIR/setup_${env_name}_last100.txt)" >&2
      if [ ${#model_keys[@]} -gt 0 ]; then
        for mk in "${model_keys[@]}"; do
          echo "${env_name}|${wrapper_key}|${mk}|SETUP_FAIL|see $LOG_DIR/setup_${env_name}_last100.txt" >> "$REPORT_FILE"
          ((TOTAL++)) || true
        done
      else
        echo "${env_name}|${wrapper_key}||SETUP_FAIL|see $LOG_DIR/setup_${env_name}_last100.txt" >> "$REPORT_FILE"
        ((TOTAL++)) || true
      fi
      continue
    fi
  fi

  conda activate "$env_name" || { echo "  conda activate $env_name failed" >&2; continue; }

  if [ ${#model_keys[@]} -gt 0 ]; then
    # Test each model variant
    for mk in "${model_keys[@]}"; do
      ((TOTAL++)) || true
      result=$(run_one_test "$env_name" "$wrapper_key" "$mk")

      if [ "$REPORT_PARAMS" = true ]; then
        # result is TSV line with real tabs: key<TAB>STATUS<TAB>emb_dim<TAB>params<TAB>context_bp
        TSV_LINES+=("$(printf '%s\t%s' "$env_name" "$result")")
        local_status=$(printf '%s' "$result" | cut -f2)
        echo "  $mk: $local_status"
        if [ "$local_status" = "OK" ] || [ "$local_status" = "SKIP" ]; then
          ((PASSED++)) || true
        fi
        echo "${env_name}|${wrapper_key}|${mk}|${local_status}|${result}" >> "$REPORT_FILE"
      else
        status="${result%%|*}"
        detail="${result#*|}"
        echo "  $mk: $status $detail"
        if [ "$status" = "OK" ]; then
          ((PASSED++)) || true
        fi
        echo "${env_name}|${wrapper_key}|${mk}|${status}|${detail}" >> "$REPORT_FILE"
      fi
    done
  else
    # Test wrapper with default model (no --model flag)
    ((TOTAL++)) || true
    result=$(run_one_test "$env_name" "$wrapper_key" "")

    if [ "$REPORT_PARAMS" = true ]; then
      TSV_LINES+=("$(printf '%s\t%s' "$env_name" "$result")")
      local_status=$(printf '%s' "$result" | cut -f2)
      echo "  $wrapper_key: $local_status"
      if [ "$local_status" = "OK" ] || [ "$local_status" = "SKIP" ]; then
        ((PASSED++)) || true
      fi
      echo "${env_name}|${wrapper_key}||${local_status}|${result}" >> "$REPORT_FILE"
    else
      status="${result%%|*}"
      detail="${result#*|}"
      echo "  $wrapper_key: $status $detail"
      if [ "$status" = "OK" ]; then
        ((PASSED++)) || true
      fi
      echo "${env_name}|${wrapper_key}||${status}|${detail}" >> "$REPORT_FILE"
    fi
  fi

  conda deactivate 2>/dev/null || true
done

echo ""
echo "=============================================="

if [ "$REPORT_PARAMS" = true ]; then
  # Rich table output from collected TSV lines
  echo "  DEPLOYMENT WRAPPER REPORT"
  echo "=============================================="
  printf "%-14s %-16s %-6s %8s %16s %10s\n" "ENV" "MODEL" "STATUS" "EMB_DIM" "PARAMS" "CONTEXT_BP"
  printf "%-14s %-16s %-6s %8s %16s %10s\n" "---" "-----" "------" "-------" "------" "----------"
  for tsv_line in "${TSV_LINES[@]}"; do
    # tsv_line has real tabs: env<TAB>key<TAB>STATUS<TAB>emb_dim<TAB>params<TAB>context_bp
    env_col=$(printf '%s' "$tsv_line" | cut -f1)
    model_col=$(printf '%s' "$tsv_line" | cut -f2)
    status_col=$(printf '%s' "$tsv_line" | cut -f3)
    emb_col=$(printf '%s' "$tsv_line" | cut -f4)
    params_col=$(printf '%s' "$tsv_line" | cut -f5)
    ctx_col=$(printf '%s' "$tsv_line" | cut -f6)
    # Format params with commas if numeric
    if [[ "$params_col" =~ ^[0-9]+$ ]]; then
      params_col=$(printf "%'d" "$params_col")
    fi
    printf "%-14s %-16s %-6s %8s %16s %10s\n" "$env_col" "$model_col" "$status_col" "$emb_col" "$params_col" "$ctx_col"
  done
else
  echo "  REPORT"
  echo "=============================================="
  printf "%-14s %-16s %-6s %s\n" "ENV" "WRAPPER" "STATUS" "DETAIL"
  printf "%-14s %-16s %-6s %s\n" "---" "-------" "------" "------"
  tail -n +2 "$REPORT_FILE" | while IFS='|' read -r env_name wrapper_key model_key status detail; do
    display_key="${model_key:-$wrapper_key}"
    # Truncate long detail for display
    [ ${#detail} -gt 50 ] && detail="${detail:0:47}..."
    printf "%-14s %-16s %-6s %s\n" "$env_name" "$display_key" "$status" "$detail"
  done
fi

echo "=============================================="
echo "  Total: $PASSED / $TOTAL passed"
echo "  Full report: $REPORT_FILE"
echo "=============================================="

[ "$PASSED" -eq "$TOTAL" ] && exit 0 || exit 1
