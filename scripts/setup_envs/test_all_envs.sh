#!/usr/bin/env bash
# 1. Create each environment (run setup script), 2. Activate it, 3. Run wrapper test on a test sequence.
# Outputs a report showing which envs/wrappers worked.
# Use --skip-setup to only run tests in existing envs (faster for iteration).
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_SETUP=false
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Load HF token so NT/AlphaGenome and gated models work when run directly on Lambda
if [ -f ~/.hf_token ]; then export HF_TOKEN=$(cat ~/.hf_token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi

# conda init
source "$(conda info --base)/etc/profile.d/conda.sh"

# Report file (append each result)
REPORT_FILE="${REPORT_FILE:-$SCRIPT_DIR/test_all_envs_report.txt}"
LOG_DIR="${SETUP_LOG_DIR:-$SCRIPT_DIR/setup_logs}"
mkdir -p "$LOG_DIR"
echo "ENV|WRAPPER|STATUS|DETAIL" > "$REPORT_FILE"

# Each row: "setup_script|conda_env|wrapper_key"
# Order: per-env setup script, default env name, then one or more wrapper keys to test in that env
ENVS=(
  "setup_nucleotide_transformer.sh|nt|nt"
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
  "setup_main.sh|genebeddings_main|nt|convnova|mutbert|hyenadna|caduceus|dnabert|rinalmo|specieslm"
)
# Note: main env does not test borzoi/spliceai in this list to keep runtime down; add if desired.

run_one_test() {
  local env_name="$1"
  local wrapper_key="$2"
  local out
  local code
  out=$(python "$SCRIPT_DIR/run_wrapper_test.py" --wrapper "$wrapper_key" 2>&1) || true
  code=$?
  if [ $code -eq 0 ]; then
    echo "OK|${out}"
  else
    echo "FAIL|${out}"
  fi
}

echo "=============================================="
echo "  test_all_envs: create envs + run wrapper tests"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  SKIP_SETUP=$SKIP_SETUP"
echo "=============================================="

TOTAL=0
PASSED=0

for row in "${ENVS[@]}"; do
  IFS='|' read -r setup_script env_name rest <<< "$row"
  IFS='|' read -ra wrapper_keys <<< "$rest"

  echo ""
  echo "--- Environment: $env_name ---"

  if [ "$SKIP_SETUP" = false ]; then
    echo "  Running $setup_script ..."
    SETUP_LOG="$LOG_DIR/setup_${env_name}.log"
    if ! bash "$SCRIPT_DIR/$setup_script" 2>&1 | tee "$SETUP_LOG"; then
      # Capture last 100 lines for debugging; report points to log
      tail -100 "$SETUP_LOG" > "$LOG_DIR/setup_${env_name}_last100.txt"
      echo "  Setup failed for $env_name (last 100 lines in $LOG_DIR/setup_${env_name}_last100.txt)" >&2
      for w in "${wrapper_keys[@]}"; do
        echo "${env_name}|${w}|SETUP_FAIL|see $LOG_DIR/setup_${env_name}_last100.txt" >> "$REPORT_FILE"
        ((TOTAL++)) || true
      done
      continue
    fi
  fi

  conda activate "$env_name" || { echo "  conda activate $env_name failed" >> "$REPORT_FILE"; continue; }

  for w in "${wrapper_keys[@]}"; do
    ((TOTAL++)) || true
    result=$(run_one_test "$env_name" "$w")
    status="${result%%|*}"
    detail="${result#*|}"
    echo "  $w: $status $detail"
    if [ "$status" = "OK" ] && echo "$detail" | grep -q "shape="; then
      ((PASSED++)) || true
    fi
    echo "${env_name}|${w}|${status}|${detail}" >> "$REPORT_FILE"
  done

  conda deactivate 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "  REPORT"
echo "=============================================="
printf "%-22s %-12s %-6s %s\n" "ENV" "WRAPPER" "STATUS" "DETAIL"
printf "%-22s %-12s %-6s %s\n" "---" "-------" "-----" "------"
tail -n +2 "$REPORT_FILE" | while IFS='|' read -r env_name wrapper_key status detail; do
  # Truncate long detail for display
  [ ${#detail} -gt 50 ] && detail="${detail:0:47}..."
  printf "%-22s %-12s %-6s %s\n" "$env_name" "$wrapper_key" "$status" "$detail"
done
echo "=============================================="
echo "  Total: $PASSED / $TOTAL passed"
echo "  Full report: $REPORT_FILE"
echo "=============================================="

[ "$PASSED" -eq "$TOTAL" ] && exit 0 || exit 1
