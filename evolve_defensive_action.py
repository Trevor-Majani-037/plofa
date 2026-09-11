"""Evolve and evaluate the defensive-action controller (CLI).

Usage:
  python evolve_defensive_action.py [--generations 40] [--pop 32] [--seed 123]
                                    [--states 600]
  python evolve_defensive_action.py --collect-only --matches 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from defensive_action_evolution import (
    evolve_defensive_action, distribution_report, EvolutionResult)
from football_brain import DefensiveActionBrain, DEFENSIVE_ACTIONS
from defensive_action_probe import DefensiveActionSurrogate, collect_defensive_samples

BRAIN_OUT = "brains_def/ACTION.json"


def _load_samples(path: str, weighted: float = 1.0):
    with open(path, "r") as f:
        rows = json.load(f)
    moments = [np.asarray(r["sensors"], dtype=np.float64) for r in rows]
    styles = [r.get("style", "") for r in rows]
    if weighted < 1.0:
        keep = np.random.default_rng(0).random(len(rows)) < weighted
        moments = [m for m, k in zip(moments, keep) if k]
        styles = [s for s, k in zip(styles, keep) if k]
    return moments, styles


def main():
    ap = argparse.ArgumentParser(description="Defensive-action evolution CLI.")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--states", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--heuristic-samples", type=str,
                    default="brains_def/samples_heuristic.json")
    ap.add_argument("--forced-samples", type=str,
                    default="brains_def/samples_forced.json")
    ap.add_argument("--heur-weight", type=float, default=0.5,
                    help="Under-sample the natural (clearance-heavy) set so "
                         "the distribution stays balanced.")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--collect-seed", type=int, default=21)
    ap.add_argument("--fast", action="store_true", help="small smoke run")
    args = ap.parse_args()

    if args.fast:
        args.generations = 8
        args.pop = 12
        args.states = 100

    if args.collect_only:
        print("Collecting heuristic + forced counterfactual matches...")
        from defensive_action_probe import collect_defensive_samples
        for out_s, intervene in (
                (args.heuristic_samples, None),
                (args.forced_samples,
                 ["tackle", "interception", "clearance", "block"])):
            rows = collect_defensive_samples(
                n_matches=args.matches, seed=args.collect_seed,
                away_styles=["fluid_counter"],
                intervene_actions=intervene)
            if rows:
                os.makedirs(os.path.dirname(os.path.abspath(out_s)) or ".",
                            exist_ok=True)
                with open(out_s, "w") as f:
                    json.dump(rows, f)
                print(f"  wrote {len(rows)} samples -> {out_s}")
        return

    # ── load + merge dataset ────────────────────────────────
    heur_moments, heur_styles = [], []
    forced_moments, forced_styles = [], []
    if os.path.exists(args.heuristic_samples):
        heur_moments, heur_styles = _load_samples(args.heuristic_samples,
                                                  args.heur_weight)
    if os.path.exists(args.forced_samples):
        forced_moments, forced_styles = _load_samples(args.forced_samples, 1.0)

    moments = heur_moments + forced_moments
    styles = heur_styles + forced_styles
    if not moments:
        print("No samples found. Run the collector first, e.g.\n"
              "  python defensive_action_probe.py --matches 6\n"
              "  python defensive_action_probe.py --matches 12 "
              "--intervene tackle,interception,clearance,block")
        sys.exit(1)
    print(f"dataset: {len(moments)} moments "
          f"({len(heur_moments)} heuristic + {len(forced_moments)} forced)")
    from collections import Counter
    print("styles:", Counter(styles))

    # ── surrogate from merged real-match data ───────────────
    rows = []
    for path, w in ((args.heuristic_samples, args.heur_weight),
                    (args.forced_samples, 1.0)):
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            if w < 1.0:
                keep = np.random.default_rng(0).random(len(r)) < w
                r = [x for x, k in zip(r, keep) if k]
            rows.extend(r)
    surrogate = DefensiveActionSurrogate().fit(rows)
    print(f"surrogate: {surrogate.report()}")

    # ── evolve ──────────────────────────────────────────────
    print(f"\nevolving DefensiveActionBrain "
          f"{args.generations} gen x {args.pop} pop x {args.states} states...")
    res: EvolutionResult = evolve_defensive_action(
        moments, styles, surrogate=surrogate,
        population_size=args.pop, generations=args.generations,
        n_states=args.states, seed=args.seed)
    print(f"\nbest fitness {res.best_fitness:.4f}")

    best = res.best_brain
    os.makedirs(os.path.dirname(os.path.abspath(BRAIN_OUT)) or ".", exist_ok=True)
    best.save(BRAIN_OUT)
    print(f"saved -> {BRAIN_OUT}  ({repr(best)})")

    print("\ndistribution over recorded moments:")
    rep = distribution_report(best, moments)
    print(f"  n={rep['n']}  {rep['counts']}")

    print("\nfavourite act by bucket sample (feasible-only deep moments):")
    for m in moments:
        if m[19] < 0.5:
            continue
        print(f"  {surrogate.bucket(m)} -> {best.predict(m)}")


if __name__ == "__main__":
    main()