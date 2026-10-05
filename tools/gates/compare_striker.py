"""Compare STRIKER output between two full-XI brain sets in real matches.

Runs identical seeds for both sets (neural full-XI each), then reports:
  • home-team goals per match and total
  • who scored (per player name)
  • ST shots (on/off/blocked/woodwork/penalties) and ST goals
  • per-position shot distribution from the home XI

Usage:
    python compare_striker.py <dir_a> <dir_b> --matches 7 --seed 21 [--label-a T1 --label-b T2]
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict

from match_engine import EventType
from validate_neural_xl import _run_neural

SHOT_TYPES = {
    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED,
    EventType.HIT_WOODWORK, EventType.FREEKICK_DIRECT, EventType.PENALTY_SCORED,
    EventType.PENALTY_MISSED,
}
GOAL_TYPES = {EventType.GOAL, EventType.PENALTY_SCORED}


def home_events(result):
    goals = defaultdict(int)      # scorer -> count
    shots = defaultdict(int)      # shooter -> count
    shot_g = defaultdict(int)     # shooter -> goals from shot events
    home_name = result.config.home_team
    for e in result.timeline:
        if e.team != home_name:
            continue
        name = e.player or "?"
        if e.event_type in GOAL_TYPES:
            goals[name] += 1
        elif e.event_type in SHOT_TYPES:
            shots[name] += 1
    return goals, shots


def run_set(brains_dir: str, n_matches: int, seed: int,
            home_style: str, away_style: str) -> dict:
    print(f"\n=== {os.path.basename(brains_dir)} ===")
    scores, goals_total, all_goals, all_shots = [], 0, defaultdict(int), defaultdict(int)
    poss = []
    t0 = time.time()
    for m in range(n_matches):
        s = seed + m * 100
        random.seed(s)   # identical stream to validate_neural_xl.main()
        result = _run_neural(brains_dir, s, home_style, away_style)[0]
        g, sh = home_events(result)
        scores.append(result.score_str)
        goals_total += result.home_goals
        poss.append(result.home_possession_pct)
        for k, v in g.items():
            all_goals[k] += v
        for k, v in sh.items():
            all_shots[k] += v
        print(f"  m{m+1}: {result.score_str}  home_goals={result.home_goals} "
              f"poss={result.home_possession_pct:.1f}")
    print(f"  TOTAL goals: {goals_total}")
    print(f"  mean possession: {statistics.mean(poss):.1f}")
    if all_goals:
        print("  SCORERS: " + ", ".join(f"{k} {v}" for k, v in sorted(all_goals.items())))
    print("  SHOOTERS: " + ", ".join(f"{k} {v}" for k, v in sorted(all_shots.items())))
    elapsed = time.time() - t0
    return {
        "goals": goals_total, "scores": scores,
        "scorers": dict(all_goals), "shooters": dict(all_shots),
        "poss": statistics.mean(poss), "elapsed_s": round(elapsed, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("brains_a", help="Full-XI brain dir (e.g. brains)")
    p.add_argument("brains_b", help="Full-XI brain dir (e.g. brains_retrain2)")
    p.add_argument("--matches", type=int, default=7)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--label-a", default="T1")
    p.add_argument("--label-b", default="T2")
    args = p.parse_args()

    for d in (args.brains_a, args.brains_b):
        if not os.path.isdir(d):
            sys.exit(f"! missing brain dir: {d}")
    os.environ.setdefault("PYTHONPATH", ".")

    a = run_set(args.brains_a, args.matches, args.seed,
                args.home_style, args.away_style)
    b = run_set(args.brains_b, args.matches, args.seed,
                args.home_style, args.away_style)

    A = f"[{args.label_a} {os.path.basename(args.brains_a)}]"
    B = f"[{args.label_b} {os.path.basename(args.brains_b)}]"
    print("\n" + "=" * 64)
    print(f"{'metric':<28}{A:>28}{B:>28}")
    print("-" * 64)
    print(f"{'home goals':<28}{a['goals']:>28}{b['goals']:>28}")
    print(f"{'ST goals':<28}{a['scorers'].get('ST',0):>28}{b['scorers'].get('ST',0):>28}")
    print(f"{'ST shots':<28}{a['shooters'].get('ST',0):>28}{b['shooters'].get('ST',0):>28}")
    print(f"{'ST shot share %':<28}"
          f"{100*a['shooters'].get('ST',0)/max(1,sum(a['shooters'].values())):>27.1f}%"
          f"{100*b['shooters'].get('ST',0)/max(1,sum(b['shooters'].values())):>27.1f}%")
    print(f"{'mean possession':<28}{a['poss']:>28.1f}{b['poss']:>28.1f}")
    print(f"{'elapsed':<28}{a['elapsed_s']:>28.1f}{b['elapsed_s']:>28.1f}")
    print("=" * 64)
    print("SCORERS A:", ", ".join(f"{k} {v}" for k, v in sorted(a['scorers'].items())) or "none")
    print("SCORERS B:", ", ".join(f"{k} {v}" for k, v in sorted(b['scorers'].items())) or "none")
    print("SHOOTERS A:", ", ".join(f"{k} {v}" for k, v in sorted(a['shooters'].items())))
    print("SHOOTERS B:", ", ".join(f"{k} {v}" for k, v in sorted(b['shooters'].items())))


if __name__ == "__main__":
    main()