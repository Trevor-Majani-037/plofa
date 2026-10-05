"""Gate two full-XI brain dirs against each other in REAL matches.

Identical seeds per match (seed + m*100, the validate_neural_xl
convention), home=Probe FC balanced vs away=Rival FC fluid_counter.
Both arms use the same brains architecture (challenger/incumbent, not
neural-vs-heuristic), so this is a clean head-to-head for trainergate /
surrogate-v2 experiments.

Usage:
    python gate_xl.py brains brains_trainer/challenger_v2 --matches 7 --seed 21
"""
from __future__ import annotations

import argparse
import os
import random
import statistics

from match_probe import _pin_heuristic, _restore_neural  # noqa: F401  (kept for parity)
from validate_neural_xl import _run_neural


def gate(brains_a: str, brains_b: str, n_matches: int, seed: int,
         home_style: str = "balanced", away_style: str = "fluid_counter",
         name_a: str = "A", name_b: str = "B") -> dict:
    af, bf = [], []
    print(f"GATE {name_a} {brains_a}  vs  {name_b} {brains_b}")
    print(f"  {n_matches} matches, seed {seed}, {home_style} vs {away_style}\n")
    for m in range(n_matches):
        s = seed + m * 100
        random.seed(s)
        result_a, fa = _run_neural(brains_a, s, home_style, away_style)
        random.seed(s)
        result_b, fb = _run_neural(brains_b, s, home_style, away_style)
        af.append(fa)
        bf.append(fb)
        print(f"  m{m+1}: [{name_a}] {result_a.score_str} fit={fa['fitness']:.3f} "
              f"goals={fa['goals']} poss={fa.get('possession_pct')}   |   "
              f"[{name_b}] {result_b.score_str} fit={fb['fitness']:.3f} "
              f"goals={fb['goals']} poss={fb.get('possession_pct')}")

    def _avg(lst, k):
        return round(sum(x[k] for x in lst) / len(lst), 4)

    summary = {
        "matches": n_matches, "seed": seed,
        f"{name_a}_fitness": _avg(af, "fitness"),
        f"{name_b}_fitness": _avg(bf, "fitness"),
        f"{name_a}_goals": sum(x["goals"] for x in af),
        f"{name_b}_goals": sum(x["goals"] for x in bf),
        f"{name_a}_possession": _avg(af, "possession_pct"),
        f"{name_b}_possession": _avg(bf, "possession_pct"),
    }
    summary["fitness_delta"] = round(
        summary[f"{name_a}_fitness"] - summary[f"{name_b}_fitness"], 4)
    summary["goal_diff"] = summary[f"{name_a}_goals"] - summary[f"{name_b}_goals"]
    wins = sum(1 for fa, fb in zip(af, bf) if fa["fitness"] > fb["fitness"])
    draws = sum(1 for fa, fb in zip(af, bf) if fa["fitness"] == fb["fitness"])
    summary[f"{name_a}_wins"] = wins
    summary[f"{name_b}_wins"] = len(af) - wins - draws
    print(f"\n  RESULT: {name_a} fitness {summary[f'{name_a}_fitness']} vs "
          f"{name_b} {summary[f'{name_b}_fitness']} (delta "
          f"{summary['fitness_delta']:+.4f}); goals {summary[f'{name_a}_goals']} "
          f"vs {summary[f'{name_b}_goals']} (diff {summary['goal_diff']:+d}); "
          f"won {wins}/{len(af)}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("incumbent", help="Baseline brain dir (e.g. brains = T1).")
    p.add_argument("challenger", help="Challenger brain dir.")
    p.add_argument("--matches", type=int, default=7)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--name-a", default="incumbent")
    p.add_argument("--name-b", default="challenger")
    args = p.parse_args()
    summary = gate(args.incumbent, args.challenger, args.matches, args.seed,
                   args.home_style, args.away_style,
                   args.name_a, args.name_b)
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()