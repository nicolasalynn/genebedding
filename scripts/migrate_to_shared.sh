#!/usr/bin/env bash
# Safely migrate embedding DBs and parquets from local instance storage
# to shared NFS mount. Safe to run from multiple instances — never overwrites
# existing files on the destination.
#
# Usage:
#   bash scripts/migrate_to_shared.sh
#   bash scripts/migrate_to_shared.sh --dry-run   # preview what would be copied
#
# After migration, set EPISTASIS_PAPER_ROOT to the shared mount so future
# pipeline runs write directly there (see bottom of script).
set -e

SHARED_BASE="${SHARED_STORAGE:-/lambda/nfs/lambda-nicolas-sandbox}"
LOCAL_ROOT="${EPISTASIS_PAPER_ROOT:-$HOME/data/epistasis_paper}"
SHARED_ROOT="$SHARED_BASE/epistasis_paper"

DRY_RUN=""
[ "$1" = "--dry-run" ] && DRY_RUN=1

if [ ! -d "$SHARED_BASE" ]; then
  echo "ERROR: Shared mount not found at $SHARED_BASE"
  echo "  Set SHARED_STORAGE=/path/to/mount and retry."
  exit 1
fi

if [ ! -d "$LOCAL_ROOT" ]; then
  echo "ERROR: Local data root not found at $LOCAL_ROOT"
  echo "  Set EPISTASIS_PAPER_ROOT=/path/to/local/data and retry."
  exit 1
fi

echo "=== Migrate to shared storage ==="
echo "  From: $LOCAL_ROOT"
echo "  To:   $SHARED_ROOT"
echo ""

# Create destination structure
if [ -z "$DRY_RUN" ]; then
  mkdir -p "$SHARED_ROOT"
fi

# rsync with --ignore-existing: never overwrite files already on destination.
# This is safe to run from multiple instances — first writer wins, subsequent
# runs skip files that already exist.
#
# Flags:
#   -a           archive mode (preserves permissions, timestamps, etc.)
#   -v           verbose (show what's being copied)
#   --progress   show per-file progress for large DBs
#   --ignore-existing  NEVER overwrite existing files on destination
RSYNC_FLAGS="-av --progress --ignore-existing"

if [ -n "$DRY_RUN" ]; then
  RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
  echo "[DRY-RUN] Would copy the following files:"
  echo ""
fi

# Migrate embeddings (DBs, npz packs, parquets)
if [ -d "$LOCAL_ROOT/embeddings" ]; then
  echo "--- Migrating embeddings ---"
  rsync $RSYNC_FLAGS "$LOCAL_ROOT/embeddings/" "$SHARED_ROOT/embeddings/"
  echo ""
fi

# Migrate data (CSVs, TSVs)
if [ -d "$LOCAL_ROOT/data" ]; then
  echo "--- Migrating data ---"
  rsync $RSYNC_FLAGS "$LOCAL_ROOT/data/" "$SHARED_ROOT/data/"
  echo ""
fi

# Migrate sheets if they exist
if [ -d "$LOCAL_ROOT/embeddings/sheets" ]; then
  echo "--- Migrating sheets ---"
  rsync $RSYNC_FLAGS "$LOCAL_ROOT/embeddings/sheets/" "$SHARED_ROOT/embeddings/sheets/"
  echo ""
fi

if [ -n "$DRY_RUN" ]; then
  echo "[DRY-RUN] No files were copied. Remove --dry-run to execute."
else
  echo "=== Migration complete ==="
  echo ""
  echo "File counts:"
  echo "  Local:  $(find "$LOCAL_ROOT" -type f | wc -l) files"
  echo "  Shared: $(find "$SHARED_ROOT" -type f | wc -l) files"
  echo ""
  echo "To use shared storage going forward, add to ~/.bashrc:"
  echo ""
  echo "  export EPISTASIS_PAPER_ROOT=$SHARED_ROOT"
  echo ""
  echo "Then restart your shell or run: source ~/.bashrc"
fi
