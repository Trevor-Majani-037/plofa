"""Parallel evolution of all 11 positional brains on a fitness surrogate.

Because each position's evolution is independent, we launch several
subprocesses at once (one per position) to use the machine's CPUs.
Each position writes its own brains/<POS>.json file.

Usage:
    python evolve_all_parallel.py [--surrogate PATH] [--states N]
                                  [--generations N] [--population N]
                                  [--workers N] [--seed S]

The surrogate is shared read-only across all workers (fast matches are
not run during evolution, only during validation).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]


def _run_position(args_tuple):
    pos, surrogate, states, generations, population, seed, out, goal_bias = args_tuple
    cmd = [
        sys.executable, "evolve_brains.py",
        "--position", pos,
        "--surrogate", surrogate,
        "--states", str(states),
        "--generations", str(generations),
        "--population", str(population),
        "--seed", str(seed),
        "--goal-bias", str(goal_bias),
        "--out", out,
    ]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        # last summary line: "  POS: fitness=... (Xs) -> path"
        summary = ""
        for line in proc.stdout.splitlines():
            if f"{pos}:" in line and "fitness=" in line:
                summary = line.strip()
        elapsed = time.time() - start
        ok = proc.returncode == 0
        return {
            "pos": pos, "ok": ok, "elapsed_s": round(elapsed, 1),
            "summary": summary,
            "err": (proc.stderr[-800:] if proc.stderr and not ok else ""),
        }
    except Exception as e:  # noqa: BLE001
        return {"pos": pos, "ok": False, "elapsed_s": round(time.time() - start, 1),
                "summary": "", "err": str(e)}


def main():
    p = argparse.ArgumentParser(description="Parallel-evolve all positional brains.")
    p.add_argument("--surrogate", default="brains/surrogate.json")
    p.add_argument("--states", type=int, default=300)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--goal-bias", type=float, default=0.0,
                   help="Fraction of states drawn from scoring situations "
                        "(passed through to evolve_brains --goal-bias).")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated subset; default all 11.")
    p.add_argument("--out", default="brains")
    args = p.parse_args()

    if not os.path.exists(args.surrogate):
        print(f"! surrogate not found: {args.surrogate}")
        sys.exit(1)

    positions = ([s.strip().upper() for s in args.positions.split(",") if s.strip()]
                 if args.positions else ALL_POSITIONS)

    print(f"Evolving {len(positions)} positions with {args.workers} workers "
          f"(surrogate={args.surrogate})")
    tasks = [(pos, args.surrogate, args.states, args.generations, args.population,
              args.seed, args.out, args.goal_bias) for pos in positions]

    results = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_position, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            pos = r["pos"]
            results[pos] = r
            status = "OK " if r["ok"] else "ERR"
            print(f"[{status}] {pos}: {r['elapsed_s']}s  {r['summary']}")
            if r["err"]:
                print(f"       {r['err']}")
    elapsed = time.time() - start

    manifest = os.path.join(args.out, "_manifest_all.json")
    with open(manifest, "w") as f:
        json.dump({
            "surrogate": args.surrogate,
            "states": args.states, "generations": args.generations,
            "population": args.population, "seed": args.seed,
            "goal_bias": args.goal_bias,
            "total_elapsed_s": round(elapsed, 1), "results": results,
        }, f, indent=2)
    print(f"\nDone in {elapsed/60:.1f} min. Manifest: {manifest}")


if __name__ == "__main__":
    main()
