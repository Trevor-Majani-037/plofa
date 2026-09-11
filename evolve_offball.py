"""Evolve off-ball press-gate consciences (OffBallBrain) per position.

CLI usage:
    python evolve_offball.py --position CM
    python evolve_offball.py --positions CM,LW,RW --generations 40 --population 32
    python evolve_offball.py  # all outfield positions, defaults

Output: brains_offball/<POSITION>.json (serialized OffBallBrain) driven by
the learnt off-ball surrogate (brains_offball/surrogate.json) so evolution
optimizes the defensive-episode outcome that press commits actually earn.

Fitness grounding: synthetic_conscience_fitness (offball_evolution.py) scores
each net against the surrogate's expected success of its chosen press/hold
decision in a defending state where the ball is inside the chase trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from offball_evolution import evolve_conscience
from football_brain import OffBallBrain

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve per-position off-ball press-gate consciences."
    )
    p.add_argument("--position", type=str, default=None,
                   help="Single position to evolve (default: all).")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated positions (e.g. CM,LW,RW).")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--states", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="brains_offball",
                   help="Output dir for per-position press-gate JSON files.")
    p.add_argument("--surrogate", type=str, default="brains_offball/surrogate.json",
                   help="Path to the learnt OffBallSurrogate JSON.")
    p.add_argument("--fast", action="store_true",
                   help="Reduced defaults for a quick smoke test.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.fast:
        args.generations = min(args.generations, 10)
        args.population = min(args.population, 16)
        args.states = min(args.states, 100)

    if args.position:
        positions = [args.position.strip().upper()]
    elif args.positions:
        positions = [s.strip().upper() for s in args.positions.split(",") if s.strip()]
    else:
        positions = ALL_POSITIONS

    surrogate = None
    if args.surrogate and os.path.exists(args.surrogate):
        from offball_probe import OffBallSurrogate
        surrogate = OffBallSurrogate.load(args.surrogate)
        print(f"Using off-ball surrogate: {args.surrogate} "
              f"({len(surrogate.table)} positions)")
    else:
        print(f"! surrogate {args.surrogate} not found — evolving with neutral "
              f"fitness (no press/hold grounding).")

    os.makedirs(args.out, exist_ok=True)
    results = {}

    for pos in positions:
        if pos not in ALL_POSITIONS:
            print(f"  ! unknown position '{pos}', skipping. Valid: {ALL_POSITIONS}")
            continue

        print(f"\n=== Evolving {pos} press-gate conscience ===")
        start = time.time()
        res = evolve_conscience(
            position=pos,
            population_size=args.population,
            generations=args.generations,
            n_states=args.states,
            seed=args.seed,
            verbose=True,
            surrogate=surrogate,
        )
        elapsed = time.time() - start

        path = os.path.join(args.out, f"{pos}.json")
        res.best_brain.save(path)
        results[pos] = {
            "best_fitness": round(res.best_fitness, 4),
            "generations": args.generations,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 1),
            "saved_to": path,
        }
        print(f"  {pos}: fitness={res.best_fitness:.4f} ({elapsed:.1f}s) -> {path}")

    manifest = os.path.join(args.out, "_manifest_offball.json")
    with open(manifest, "w") as f:
        json.dump({
            "positions": positions,
            "generations": args.generations,
            "population": args.population,
            "states": args.states,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary written to {manifest}")
    print("Load with: from football_brain import OffBallBrain; "
          "b = OffBallBrain.load('brains_offball/CM.json')")


if __name__ == "__main__":
    main()