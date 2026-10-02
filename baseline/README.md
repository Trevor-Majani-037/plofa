# Baseline Snapshot — Phase 0

Generated: 2026-09-11

## Purpose

This directory is the immutable reference for the neural-perception-upgrade project.
All later phases (hot-loop fix, perception gate, sensor layout v2, brain retraining)
are measured against these numbers.

## Contents

### backup/
- `brains/`        — production T1 brain set (11 positions, evolved GK-CF, surrogate_pos.json)
- `brains_def/`    — DefensiveActionBrain (ACTION.json)
- `brains_team/`   — TeamPressBrain (XI.json)
- `brains_trainer/`— NOT PRESENT (directory did not exist at snapshot time)
- `season_state.json`  — season ledger after MD1-3 (READ-ONLY, never modify)
- `season_stats.json`  — season stats after MD1-3 (READ-ONLY, never modify)
- `manager_state.json` — manager state snapshot
- `referee_state.json` — referee state snapshot

### matches.txt
Three scratch matches (pitch_replay.run_scratch_match, seeds 42/43/44).
No season state written. Used as determinism reference for all later phases.

### evolution_timing.txt
Wall time for `evolve_brains.py --position GK --generations 40 --seed 42`.
This is the Phase 1 baseline: the hot-loop fix must produce an identical
gen_best list AND reduce 160-gen time to <= 8 min.

### scripts/snapshot_baseline.sh
Bash script to recreate this backup from the repo root.

## Reproducibility

To re-run the baseline matches:
    python -c "from pitch_replay import run_scratch_match; run_scratch_match(seed=42)"
    python -c "from pitch_replay import run_scratch_match; run_scratch_match(seed=43)"
    python -c "from pitch_replay import run_scratch_match; run_scratch_match(seed=44)"

To re-run evolution timing:
    python evolve_brains.py --position GK --generations 40 --seed 42

## Notes

- brains_trainer/ was absent at checkpoint time.
- run_match.py HOME_TEAM=Hartwell City, AWAY_TEAM=Thornfield United.
- Season ledger files contain MD1-3 data and must NEVER be overwritten.