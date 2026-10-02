"""
PLOFA 26/27 — MANAGER BRAIN EVOLUTION CLI (Phase 7)
====================================================
evolve_manager.py

Collects the static-manager curriculum, fits the surrogate, evolves a
ManagerBrain with the same GA operator set as brain_evolution, and saves
the best brain.

Usage:
    python evolve_manager.py --manager test_manager --matches 6 \
        --generations 40 --population 32 --states 0 --seed 42 \
        --out manager_brains/v1
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

from manager_collection import collect_manager_samples
from manager_evolution import evolve, posture_breakdown
from manager_surrogate import ManagerSurrogate


def _random_sensors(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        s = [rng.uniform(0.0, 1.0) for _ in range(12)]
        s[0] = rng.uniform(-1.0, 1.0)   # score_diff
        s[4] = rng.uniform(-1.0, 1.0)   # recent_goal_impact
        out.append(np.asarray(s, dtype=np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve a ManagerBrain")
    ap.add_argument("--manager", default="test_manager")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--states", type=int, default=0,
                    help="extra random sensor states blended into the eval corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="manager_brains/v1")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 64)
    print(f"collecting {args.matches} static-manager matches (flag OFF)")
    samples = collect_manager_samples(n_matches=args.matches, seed=args.seed)
    print(f"  sampled {len(samples)} (sensors, posture, outcome_points, outcome) rows")

    surrogate = ManagerSurrogate()
    surrogate.fit(samples)
    cov = surrogate.coverage()
    print(f"  surrogate coverage: {cov['buckets']} buckets, "
          f"{cov['bucket_posture_pairs']} (bucket, posture) pairs, "
          f"{cov['postures']} postures")

    eval_samples = list(samples)
    if args.states > 0:
        rng = random.Random(args.seed + 9000)
        synth = [(s, "BALANCED", 0.5, "SYNTH") for s in _random_sensors(rng, args.states)]
        surrogate.fit(samples + synth)
        eval_samples = samples + synth
        print(f"  blended {args.states} synthetic states (eval corpus now "
              f"{len(eval_samples)} rows)")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"evolving {args.population} brains x {args.generations} generations")
    res = evolve(
        samples=eval_samples,
        population_size=args.population,
        generations=args.generations,
        surrogate=surrogate,
        seed=args.seed,
    )
    print(f"  best_fitness={res.best_fitness:.4f}  "
          f"trajectory={res.generation[0]:.4f} -> {res.generation[-1]:.4f}  "
          f"mean={res.mean[-1]:.4f}")
    print(f"  posture breakdown over samples: {posture_breakdown(res.best_brain, eval_samples)}")

    path = os.path.join(out_dir, f"{args.manager}.json")
    res.best_brain.save(path)
    print(f"  saved {path}")

    surrogate_path = os.path.join(out_dir, "surrogate_refit.json")
    surrogate.save(surrogate_path)
    print(f"  saved {surrogate_path}  ({time.time()-t0:.1f}s total)")
    print("=" * 64)


if __name__ == "__main__":
    main()