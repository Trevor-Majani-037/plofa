#!/usr/bin/env bash
# snapshot_manager_baseline.sh -- Phase 0: snapshot the CURRENT manager-related
# source files BEFORE the Manager Brains project touches them.
#
# Run from the repo root (plofa/): bash scripts/snapshot_manager_baseline.sh
# Output: baseline_manager/backup/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/baseline_manager/backup"

cd "${REPO_ROOT}"
echo "[snapshot_manager_baseline] repo root: ${REPO_ROOT}"
echo "[snapshot_manager_baseline] output:    ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

for f in manager_state.json manager_profile.py tactical_ai.py squad_manager.py training_system.py; do
    if [ -f "${f}" ]; then
        echo "  copying ${f} ..."
        cp "${f}" "${OUT_DIR}/${f}"
    else
        echo "  WARNING: ${f} not found"
    fi
done

echo "[snapshot_manager_baseline] done."
ls -la "${OUT_DIR}"