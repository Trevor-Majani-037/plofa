"""Real-match audit of the schema-v2 role feature blocks.

For every real on-ball `NeuralDecisionBrain.decide` touch (same frame the
brain saw — same teammates/defenders/engine/attacks_right that fed its
sensors), compute that player's role feature block and aggregate stats.

Purpose: eyeball the v2 features on REAL match geometry before wiring them
into any schema/brain.  A feature that is constant (std ~ 0) or stuck at a
boundary on real touches carries no signal for the opponent the league
faces; a family whose block always reads empty (all zeros / all ones) is a
menu to rethink, not to bake into a challenger's input size.

Usage:
    python audit_role_features.py --matches 2 --seed 21 --dirs brains

Read-only hat: only brains' decide is captured; engine/season untouched —
this is the same pattern as audit_intents.py / consequence_probe.py.
Outputs: console per-family table + value_experiments/role_features_audit.json
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

import numpy as np

from brain_integration import NeuralDecisionBrain
import role_features as rf
from role_features import MENU_NAMES, _role_family


def audit(brains_dir: str, n_matches: int, seed: int,
          home_style: str, away_style: str):
    """Run n neutral matches capturing per-family role features.

    Returns {family: {feature: {'n', 'mean', 'min', 'max', 'std',
    'zero_frac', 'one_frac'}}} plus total touch count.
    """
    from validate_neural_xl import _run_neural

    # family -> list of per-touch feature lists
    raw: dict[str, list[list[float]]] = defaultdict(list)
    original = NeuralDecisionBrain.decide

    def wrapped(player, x, y, teammates, defenders, position_engine, team_profile,
                under_pressure, attacks_right, game_state, minute=45.0,
                soul=None, record_trace=False):
        d = original(player, x, y, teammates, defenders, position_engine,
                     team_profile, under_pressure, attacks_right, game_state,
                     minute=minute, soul=soul, record_trace=record_trace)
        try:
            blk = rf.role_block(player, x, y, teammates, defenders,
                                position_engine, attacks_right)
            if blk is not None:
                fam = _role_family(getattr(player, "position", ""))
                raw[fam].append(list(blk))
        except Exception:
            pass  # audit must never break the match
        return d

    NeuralDecisionBrain.decide = staticmethod(wrapped)
    try:
        for m in range(n_matches):
            s = seed + m * 100
            random.seed(s)
            _run_neural(brains_dir, s, home_style, away_style)
    finally:
        NeuralDecisionBrain.decide = original

    stats: dict[str, dict] = {}
    for fam, rows in raw.items():
        arr = np.asarray(rows, dtype=np.float64)
        names = MENU_NAMES[fam]
        stats[fam] = {
            "touches": int(arr.shape[0]),
            "features": {
                names[i]: {
                    "mean": float(arr[:, i].mean()),
                    "min": float(arr[:, i].min()),
                    "max": float(arr[:, i].max()),
                    "std": float(arr[:, i].std()),
                    "zero_frac": float((arr[:, i] < 1e-9).mean()),
                    "one_frac": float((arr[:, i] > 1 - 1e-9).mean()),
                }
                for i in range(arr.shape[1])
            },
        }
    return stats


def _tabulate(stats: dict[str, dict]):
    fams = [f for f in ("GK", "CB", "FB", "DM", "CM", "AM", "WING", "ST")
            if f in stats]
    for fam in fams:
        s = stats[fam]
        n = s["touches"]
        header = (f"=== {fam}  (n={n} touches) ===")
        print(header)
        print(f"{'feature':<16}{'mean':>7}{'min':>7}{'max':>7}{'std':>7}"
              f"{'zero%':>8}{'one%':>8}  signal")
        for fname, st in sorted(s["features"].items()):
            signal = (st["std"] > 0.05
                      and st["zero_frac"] < 0.85
                      and st["one_frac"] < 0.85)
            print(f"{fname:<16}{st['mean']:7.3f}{st['min']:7.3f}"
                  f"{st['max']:7.3f}{st['std']:7.3f}"
                  f"{100*st['zero_frac']:7.1f}%{100*st['one_frac']:7.1f}%"
                  f"  {'yes' if signal else '--'}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", default="brains")
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--out", default="value_experiments/role_features_audit.json")
    args = p.parse_args()

    for d in args.dirs.split(","):
        d = d.strip()
        stats = audit(d, args.matches, args.seed,
                      args.home_style, args.away_style)
        print(f"### {d}  ({args.matches} matches, seed {args.seed}, "
              f"{args.home_style} vs {args.away_style})")
        total = sum(s["touches"] for s in stats.values())
        print(f"total on-ball touches captured: {total}\n")
        _tabulate(stats)

        import json, os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        payload = {
            "brains_dir": d, "matches": args.matches, "seed": args.seed,
            "home_style": args.home_style, "away_style": args.away_style,
            "total_touches": total,
            "families": stats,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"audit saved -> {args.out}")


if __name__ == "__main__":
    main()