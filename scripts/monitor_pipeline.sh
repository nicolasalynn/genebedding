#!/usr/bin/env bash
# Monitor pipeline progress and GPU utilization. Run in a second terminal while
# the pipeline runs (e.g. scripts/run_pipeline_cluster.sh 2>&1 | tee pipeline.log).
#
# Usage:
#   ./scripts/monitor_pipeline.sh
#   ./scripts/monitor_pipeline.sh --status-file /path/to/output_base/pipeline_status.json
#   PIPELINE_STATUS_FILE=/path/to/pipeline_status.json ./scripts/monitor_pipeline.sh
#
# Refreshes every 2 seconds. Press Ctrl+C to stop.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATUS_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --status-file) STATUS_FILE="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

if [ -z "$STATUS_FILE" ]; then
  STATUS_FILE="${PIPELINE_STATUS_FILE:-$REPO_ROOT/pipeline_status.json}"
fi

echo "Monitoring: progress -> $STATUS_FILE, GPU -> nvidia-smi (every 2s). Ctrl+C to stop."
echo ""

while true; do
  printf '\n--- %s ---\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ -f "$STATUS_FILE" ]; then
    echo "[Pipeline status]"
    cat "$STATUS_FILE" 2>/dev/null | head -30
    echo ""
  else
    echo "[Pipeline status] (no file at $STATUS_FILE)"
  fi
  echo "[GPU]"
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  sleep 2
done
