"""Real-match intent-rate audit for two brain XIs.

Runs identical same-opponent matches (Probe FC vs Rival FC, same seeds per
arm, caller-seeded like every gate) for each brains dir and reports the
chosen-intent distribution per position.  Purpose: verify a challenger's
off-ball/vertical profile on the pitch (crosses, through-balls, shots)
rather than trusting static argmax shares.

Usage:
    python audit_intents.py --dirs brains,brains_v2 --matches 3 --seed 21
"""
from __future__ import annotations

import argparse
import random

from brain_integration import NeuralDecisionBrain


def audit(brains_dir: str, n_matches: int, seed: int,
          home_style: str = "balanced", away_style: str = "fluid_counter"):
    from validate_neural_xl import _run_neural

    counts = {}
    original = NeuralDecisionBrain.decide

    def wrapped(player, x, y, teammates, defenders, position_engine, team_profile,
                under_pressure, attacks_right, game_state, minute=45.0,
                soul=None, record_trace=False):
        d = original(player, x, y, teammates, defenders, position_engine,
                     team_profile, under_pressure, attacks_right, game_state,
                     minute=minute, soul=soul, record_trace=record_trace)
        name = getattr(player, "name", "?")
        pos = getattr(player, "position", "?")
        key = (pos, d.intent.value)
        counts[key] = counts.get(key, 0) + 1
        return d

    NeuralDecisionBrain.decide = staticmethod(wrapped)
    try:
        for m in range(n_matches):
            s = seed + m * 100
            random.seed(s)
            _run_neural(brains_dir, s, home_style, away_style)
    finally:
        NeuralDecisionBrain.decide = original
    return counts


def _tabulate(counts, positions):
    intents = ["PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
               "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION"]
    per_pos = {p: {i: 0 for i in intents} for p in positions}
    tot = 0
    for (pos, intent), c in counts.items():
        if pos in per_pos and intent in per_pos[pos]:
            per_pos[pos][intent] = c
            tot += c
    print(f"{'pos':5s} | " + " | ".join(f"{i[:5]}" for i in intents))
    print("-" * (5 + len(intents) * 8))
    for p in positions:
        row = per_pos[p]
        base = max(1, sum(row.values()))
        cells = " | ".join(f"{100*row[i]/base:4.1f}%" for i in intents)
        print(f"{p:5s} | {cells}  (n={sum(row.values())})")
    for label in ("CROSS", "THROUGH_BALL", "SHOOT"):
        agg = sum(counts.get((p, label), 0) for p in positions)
        print(f"team {label:15s}: {agg}  ({100*agg/tot:.1f}% of {tot} decisions)")
    return per_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", default="brains,brains_v2")
    p.add_argument("--matches", type=int, default=3)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    args = p.parse_args()

    positions = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
    for d in args.dirs.split(","):
        c = audit(d.strip(), args.matches, args.seed,
                  args.home_style, args.away_style)
        print(f"\n=== {d.strip()} ({args.matches} matches, seed {args.seed}) ===")
        _tabulate(c, positions)


if __name__ == "__main__":
    main()