#!/usr/bin/env bash
# snapshot_baseline.sh -- reproduce the Phase 0 baseline snapshot.
# Run from the repo root (plofa/): bash scripts/snapshot_baseline.sh
# Creates baseline/backup/ with copies of all brain dirs and season ledger files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[snapshot_baseline] repo root: ${REPO_ROOT}"
mkdir -p baseline/backup

for dir in brains brains_def brains_team brains_trainer; do
    if [ -d "${dir}" ]; then
        echo "  copying ${dir}/ ..."
        cp -r "${dir}" "baseline/backup/${dir}"
    else
        echo "  SKIP: ${dir}/ does not exist"
    fi
done

for f in season_state.json season_stats.json manager_state.json referee_state.json; do
    if [ -f "${f}" ]; then
        echo "  copying ${f} ..."
        cp "${f}" "baseline/backup/${f}"
    else
        echo "  WARNING: ${f} not found"
    fi
done

echo "[snapshot_baseline] done."
ls -la baseline/backup/