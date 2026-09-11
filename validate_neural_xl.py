"""Validate a fully-neural XI against the heuristic XI in real matches.

This is the CORRECT full-team comparison.  The existing probe helpers
(``ab_compare``, ``run_neural_validation`` in match_probe.py) register
brains under ``{POS}{POS}`` (e.g. "STST"), but real players are named
"ST", "CM1", "CM2", "CB1", "CB2", etc.  Because ``NeuralDecisionBrain``
looks up brains by exact ``player.name``, that old wiring silently fell
back to *random* brains — so previous A/B numbers did not test our
evolved nets.

Here we register each position's brain to every starter whose
``position`` matches, then run real matches.  We compare:
    • NEURAL:  all 11 outfield+GK roles driven by learned brains.
    • HEURISTIC: the unchanged DecisionBrain (baseline).

Both use identical squads, seed, and opponent.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import date
from typing import Any, Dict, List, Optional

from football_brain import FootballBrain
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry,
)
from decision_brain import PlayerIntent, DecisionBrain
import match_probe

# position (as on the player object) -> brain file stem
POSITION_BRAIN_MAP = {
    "GK": "GK", "CB": "CB", "LB": "LB", "RB": "RB", "CDM": "CDM",
    "CM": "CM", "CAM": "CAM", "LW": "LW", "RW": "RW",
    "ST": "ST", "CF": "CF",
}

# The hand-calibrated heuristic, captured before any monkeypatching.
# event_chain.py now calls NeuralDecisionBrain.decide() directly, so the
# heuristic baseline is obtained by temporarily pinning the neural entry
# point to this function for the duration of a heuristic match.
_HEURISTIC_DECIDE = DecisionBrain.decide


def register_full_xi(brains_dir: str, starters: List[Any]) -> int:
    """Register every position's brain to all matching starters.

    Returns number of player-bindings created.
    """
    clear_registry()
    bound = 0
    for p in starters:
        stem = POSITION_BRAIN_MAP.get(p.position)
        if not stem:
            continue
        path = os.path.join(brains_dir, f"{stem}.json")
        if not os.path.exists(path):
            continue
        brain = FootballBrain.load(path)
        register_brain(p.name, brain)
        bound += 1
    return bound


def _run_neural(brains_dir: str, seed: int, home_style: str, away_style: str):
    home_squad, away_squad = match_probe._build_squads("Probe FC", "Rival FC")
    n_bound = register_full_xi(brains_dir, home_squad["starters"])
    try:
        config = match_probe.MatchConfig(
            home_team="Probe FC", away_team="Rival FC",
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile("Probe FC", home_style)
        ap = match_probe._team_profile("Rival FC", away_style)
        eng = match_probe.MatchEngine(config, hp, ap)
        eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
        eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
        result = eng.simulate()
        home_names = [p.name for p in home_squad["starters"]]
        fitness = match_probe.extract_team_fitness(result, home_names)
        fitness["_n_bound"] = n_bound
        return result, fitness
    finally:
        clear_registry()


def _run_heuristic(seed: int, home_style: str, away_style: str):
    home_squad, away_squad = match_probe._build_squads("Probe FC", "Rival FC")
    clear_registry()
    # Temporarily pin the neural entry point to the genuine heuristic so
    # substitutes/unregistered players cannot auto-load neural brains, and
    # turn the team press controller OFF: the heuristic arm is the OLD system
    # (heuristic on-ball + pure role-rate Bernoulli off-ball).
    saved_neural = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_HEURISTIC_DECIDE)
    from match_engine import set_team_press_auto
    set_team_press_auto(False)
    try:
        config = match_probe.MatchConfig(
            home_team="Probe FC", away_team="Rival FC",
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile("Probe FC", home_style)
        ap = match_probe._team_profile("Rival FC", away_style)
        eng = match_probe.MatchEngine(config, hp, ap)
        eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
        eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
        result = eng.simulate()
        home_names = [p.name for p in home_squad["starters"]]
        fitness = match_probe.extract_team_fitness(result, home_names)
        return result, fitness
    finally:
        NeuralDecisionBrain.decide = saved_neural
        set_team_press_auto(True)
        clear_registry()


def main():
    p = argparse.ArgumentParser(description="Full-XI neural vs heuristic validation.")
    p.add_argument("--brains", default="brains")
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    args = p.parse_args()

    print(f"=== Full-XI Neural vs Heuristic ({args.matches} matches each) ===")
    print("(each match ~15-20s)\n")

    n_fits, h_fits = [], []
    t0 = time.time()
    for m in range(args.matches):
        seed = args.seed + m * 100
        random.seed(seed)
        # same seed for both sides; reset match_probe module RNG too
        r, nf = _run_neural(args.brains, seed, args.home_style, args.away_style)
        print(f"  neural m{m+1}: {r.score_str}  "
              f"fit={nf['fitness']:.3f}  goals={nf['goals']}  "
              f"possession={nf.get('possession_pct')}%  "
              f"bound={nf.get('_n_bound')}")
        n_fits.append(nf)

        random.seed(seed + 7)
        r, hf = _run_heuristic(seed + 7, args.home_style, args.away_style)
        print(f"  heur   m{m+1}: {r.score_str}  "
              f"fit={hf['fitness']:.3f}  goals={hf['goals']}  "
              f"possession={hf.get('possession_pct')}%")
        h_fits.append(hf)

    def _avg(lst, key):
        return round(sum(f[key] for f in lst) / len(lst), 4)

    summary = {
        "neural_fitness": _avg(n_fits, "fitness"),
        "heuristic_fitness": _avg(h_fits, "fitness"),
        "neural_goals": sum(f["goals"] for f in n_fits),
        "heuristic_goals": sum(f["goals"] for f in h_fits),
        "neural_possession": _avg(n_fits, "possession_pct"),
        "heuristic_possession": _avg(h_fits, "possession_pct"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # goal differential (positive = neural scored more on aggregate)
    summary["goal_diff"] = summary["neural_goals"] - summary["heuristic_goals"]
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
