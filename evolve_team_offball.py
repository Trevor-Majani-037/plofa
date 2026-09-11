"""Evolve the shared XI press-engagement controller (TeamPressBrain).

CLI usage:
    python evolve_team_offball.py                            # all samples, defaults
    python evolve_team_offball.py --fast --seed 123          # quick smoke
    python evolve_team_offball.py --style fluid_counter      # style-pinned

Output: brains_team/XI.json — ONE controller shared by the whole XI, mapping
a 24-d team-state vector to engagement g in [0,1] which scales the
per-position press Bernoulli in the engine (prob = _PRESS_PROB[pos] * g).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from team_offball_probe import TeamPressSurrogate
from team_offball_evolution import evolve_team_press


def parse_args():
    p = argparse.ArgumentParser(description="Evolve the XI press-engagement controller.")
    p.add_argument("--samples", type=str, default="brains_team/samples.json")
    p.add_argument("--surrogate", type=str, default="brains_team/surrogate.json")
    p.add_argument("--style", type=str, default="",
                   help="Restrict training moment style ('' = all).")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--states", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="brains_team/XI.json")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.generations = min(args.generations, 10)
        args.population = min(args.population, 16)
        args.states = min(args.states, 150)

    if not os.path.exists(args.samples):
        print(f"! samples not found: {args.samples}")
        print("  run first:  python team_offball_probe.py --matches 6")
        return

    with open(args.samples) as f:
        rows = json.load(f)
    if args.style:
        rows = [r for r in rows if r.get("style") == args.style]
    moments = [np.asarray(r["sensors"], dtype=np.float64) for r in rows]

    surrogate = TeamPressSurrogate.load(args.surrogate) \
        if os.path.exists(args.surrogate) else None
    if surrogate is None:
        print(f"! surrogate not found at {args.surrogate} — neutral fitness")
    else:
        print(f"Using team surrogate ({surrogate.report()}); "
              f"{len(moments)} real moments")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    start = time.time()
    res = evolve_team_press(
        moments, style=args.style,
        population_size=args.population,
        generations=args.generations,
        n_states=args.states,
        seed=args.seed,
        verbose=True,
        surrogate=surrogate,
    )
    res.best_brain.save(args.out)
    print(f"\nteam controller fitness={res.best_fitness:.4f} "
          f"({time.time()-start:.1f}s) -> {args.out}")
    print("Load: from football_brain import TeamPressBrain; "
          "b = TeamPressBrain.load('brains_team/XI.json')")


if __name__ == "__main__":
    main()