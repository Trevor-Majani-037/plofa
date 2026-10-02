# baseline_manager/ — Static-Manager Baseline (reproducible)

20-match snapshot captured against the CURRENT engine with
`USE_MANAGER_BRAIN=False` and static `ManagerProfile` behaviour
(seeds 42–61). A fresh capture was taken at Phase 6 because the
Phase-0 file proved irreproducible: its capture ran WITHOUT the fixed
`PYTHONHASHSEED` that the engine's cross-process determinism requires
(see `philosophy_proof.py`), and the engine has drifted since.

## What's here

| Path | Purpose |
|------|---------|
| `backup/` | Pre-edit copies of `manager_state.json`, `manager_profile.py`, `tactical_ai.py`, `squad_manager.py`, `training_system.py` |
| `matches.txt` | One JSON object per line: `seed`, `score_str`, `home_goals`, `away_goals`, `home_xg`, `away_xg`, `home_possession_pct`, `elapsed` |
| `summary.txt` | Aggregate stats: mean goals, mean xG, draw rate, home win rate |

## How the baseline was captured

20 scratch matches via `pitch_replay.run_scratch_match(seed=N)` for
N in 42..61. Static `ManagerProfile` managers are wired by
`ManagerPool` — no Manager Brain is active. The Phase-0 `MANAGER_TRACE`
instrumentation was removed by the Phase 6 spec, so no per-match `trace`
is recorded.

## Reproduce

Requires a fixed hash seed (the engine is only cross-process
deterministic under one — `philosophy_proof.py`):

```powershell
$env:PYTHONHASHSEED="0"
py -3 scripts\capture_manager_baseline.py
Remove-Item Env:\PYTHONHASHSEED
```

## Determinism

The capture script runs seed 42 twice and asserts identical `score_str`.
Verified reproducible across fresh processes under `PYTHONHASHSEED=0`
(e.g. seed 42 = 4-2, xG 1.57–0.61 every time).
