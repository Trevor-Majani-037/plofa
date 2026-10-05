"""Step-5 gate — role-specific perception (PLOFA V2 audit §I.5).

Compares the SAME T1 XI under three perception regimes at identical seeds:

  IDENTITY   : PerceptionConfig(enabled=False) — the production league state
               (validate_neural_xl neural arm semantics).
  ROLEBLOCKS : PerceptionConfig(enabled=True, role_blocks=True) — imperfect
               range/FOV/noise + position-specific profiles + relevance
               top-k.  Same T1 brains, no re-training; the study measures
               whether real football survives when players stop seeing
               omniscient geometry.
  HEURISTIC  : the hand-calibrated DecisionBrain baseline (the old system),
               for the same-gate anchor lineage.

Every arm runs the full 11+GK XI vs the same opponent (Probe FC balanced vs
Rival FC fluid_counter).  Seeding matches validate_neural_xl exactly:
`random.seed(seed + m*100)` per arm per match — _run_neural does NOT seed
internally, so skipping the explicit seed invalidates the comparison.

The role-block perception is a CONDITION TEST, not a promotion: brains are
unchanged, nothing is trained.  A real on-pitch win would justify evolving
challengers that TRAIN under the same imperfect perception for a future
champion; a signficant drop pins perception behind its flags permanently.

Usage:
  .venv\\Scripts\\python.exe validate_perception.py --matches 6 --seed 21
"""
from __future__ import annotations

import argparse
import json
import random
import time

from perception import PerceptionConfig, set_perception, get_perception_config
import validate_neural_xl


def _run_with(brains_dir: str, seed: int, home_style: str, away_style: str,
              cfg: PerceptionConfig, label: str, match_idx: int):
    """Seed + run one arm under a fixed perception config, restoring after."""
    saved = get_perception_config()
    set_perception(cfg)
    try:
        random.seed(seed)
        result, fitness = validate_neural_xl._run_neural(
            brains_dir, seed, home_style, away_style)
        print(f"  {label:11s} m{match_idx}: {result.score_str}  "
              f"fit={fitness['fitness']:.3f}  goals={fitness['goals']}  "
              f"possession={fitness.get('possession_pct')}%  bound={fitness.get('_n_bound')}")
        return result, fitness
    finally:
        set_perception(saved)


def main():
    p = argparse.ArgumentParser(description="Step-5 role-perception gate.")
    p.add_argument("--brains", default="brains")
    p.add_argument("--matches", type=int, default=6)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--include-heuristic", action="store_true",
                   help="Also run the DecisionBrain baseline arm for anchor.")
    args = p.parse_args()

    arms = [
        ("identity", PerceptionConfig(enabled=False)),
        ("roleblocks", PerceptionConfig(enabled=True, role_blocks=True,
                                        seed=0)),
    ]
    if args.include_heuristic:
        arms = [("heuristic", None)] + arms

    n_arms = len(arms)
    print(f"=== Step-5 perception gate: {args.matches} matches x {n_arms} arms ===")
    print("(same T1 brains; perception is a CONDITION, not a retrain)\n")

    results = {name: [] for name, _ in arms}
    t0 = time.time()
    for m in range(args.matches):
        seed = args.seed + m * 100
        for name, cfg in arms:
            if name == "heuristic":
                random.seed(seed + 7)
                r, hf = validate_neural_xl._run_heuristic(
                    seed + 7, args.home_style, args.away_style)
                print(f"  heuristic   m{m+1}: {r.score_str}  "
                      f"fit={hf['fitness']:.3f}  goals={hf['goals']}  "
                      f"possession={hf.get('possession_pct')}%")
                results[name].append(hf)
            else:
                r, nf = _run_with(args.brains, seed, args.home_style,
                                  args.away_style, cfg, name, m + 1)
                results[name].append(nf)

    def _avg(lst, key):
        return round(sum(f[key] for f in lst) / len(lst), 4)

    summary = {"elapsed_s": round(time.time() - t0, 1)}
    for name, _ in arms:
        fits = results[name]
        summary[f"{name}_fitness"] = _avg(fits, "fitness")
        summary[f"{name}_goals"] = sum(f["goals"] for f in fits)
        summary[f"{name}_possession"] = _avg(fits, "possession_pct")

    if args.include_heuristic:
        summary["roleblocks_vs_heuristic_goal_diff"] = (
            summary["roleblocks_goals"] - summary["heuristic_goals"])
        summary["identity_vs_heuristic_goal_diff"] = (
            summary["identity_goals"] - summary["heuristic_goals"])
    summary["roleblocks_vs_identity_fitness_delta"] = round(
        summary["roleblocks_fitness"] - summary["identity_fitness"], 4)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()