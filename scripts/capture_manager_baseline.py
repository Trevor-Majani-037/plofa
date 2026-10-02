"""
Phase 6 baseline (re)capture — reproducible protocol.

Runs pitch_replay.run_scratch_match(seed=N) for N in 42..61 WITH static
managers wired via ManagerPool and USE_MANAGER_BRAIN=False, then writes
baseline_manager/matches.txt and baseline_manager/summary.txt.

Determinism: the engine's cross-process RNG depends on PYTHONHASHSEED
(see philosophy_proof.py).  This capture therefore REQUIRES a fixed
hash seed: run with `PYTHONHASHSEED=0`.

Seed 42 is run twice; the second run's score_str must match the first
exactly.

NOTE: the temporary MANAGER_TRACE instrumentation used by the Phase-0
capture was removed by the Phase 6 spec, so per-match `trace` is no
longer recorded.
"""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("PYTHONHASHSEED") != "0":
    print("ERROR: run with PYTHONHASHSEED=0 for cross-process reproducibility.")
    sys.exit(2)

import match_engine as _me
# Baseline must ALWAYS capture the static (flag-OFF) behaviour regardless of
# the module default (ON since 2026-09-20) — pin it off for this process.
_me.USE_MANAGER_BRAIN = False
from pitch_replay import run_scratch_match

SEEDS = list(range(42, 62))   # 42..61 inclusive = 20 matches
RESULTS = []


def capture_match(seed):
    t0 = time.time()
    result = run_scratch_match(seed=seed, verbose=False)
    elapsed = time.time() - t0
    return dict(
        seed=seed,
        score_str=result.summary(),
        home_goals=int(result.home_goals),
        away_goals=int(result.away_goals),
        home_xg=round(result.home_xg, 2),
        away_xg=round(result.away_xg, 2),
        home_possession_pct=round(result.home_possession_pct, 1),
        elapsed=round(elapsed, 1),
    )


print("=" * 60)
print("Phase 6 baseline recapture (PYTHONHASHSEED=0, flag OFF)")

# Determinism check: seed 42 twice
det_a = capture_match(42)
det_b = capture_match(42)
assert det_a["score_str"] == det_b["score_str"], (
    f"DETERMINISM FAIL: {det_a['score_str']!r} vs {det_b['score_str']!r}")
print(f"  [det] seed 42 identical: {det_a['score_str']!r}")

for n in SEEDS:
    r = capture_match(n)
    RESULTS.append(r)
    print(f"  seed={n:>2d}  {r['score_str']:<52s}  xG={r['home_xg']}-{r['away_xg']}  "
          f"poss={r['home_possession_pct']}  {r['elapsed']}s")

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline_manager")
matches_path = os.path.join(base_dir, "matches.txt")
with open(matches_path, "w", encoding="utf-8") as f:
    for r in RESULTS:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

goals = [a + b for a, b in [(r["home_goals"], r["away_goals"]) for r in RESULTS]]
xg = [a + b for a, b in [(r["home_xg"], r["away_xg"]) for r in RESULTS]]
draws = sum(1 for r in RESULTS if r["home_goals"] == r["away_goals"])
home_wins = sum(1 for r in RESULTS if r["home_goals"] > r["away_goals"])
n = len(RESULTS)
summary = (
    f"mean_goals_per_match={round(sum(goals)/n, 2)}\n"
    f"mean_xg_per_match={round(sum(xg)/n, 2)}\n"
    f"draw_rate={round(draws/n, 4)}\n"
    f"home_win_rate={round(home_wins/n, 4)}\n"
)
with open(os.path.join(base_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\n  Wrote {n} lines to {matches_path}")
print(f"  Aggregate: goals={summary.splitlines()[0].split('=')[1]}  "
      f"xG={summary.splitlines()[1].split('=')[1]}  "
      f"draw_rate={summary.splitlines()[2].split('=')[1]}  "
      f"home_win_rate={summary.splitlines()[3].split('=')[1]}")
print("=" * 60)