"""A REAL-ROSTER GATE: the same matches, two brain sets, the real squads.

WHY THIS EXISTS
---------------
`run_match.py` is the runner a human watches — the real roster, the real
souls, Percy's oversized brain, the real fixture — and it **seeds nothing**,
so every invocation is a DIFFERENT match. That is perfect for sampling and
useless for a comparison: the v7 set's 4-5 and the shipped set's 4-1 were
two different games.

The scratch harness (`tools/diag/_diag_v7_gate.py`) IS a controlled pair, but
it plays the synthetic "Probe FC vs Rival FC" squads with generic names — a
different world from the one the season is actually played in, and one where
the names collide (which is what emptied the pass-network panel).

So this probe runs the REAL fixture — Hartwell City v Thornfield United, the
real squads, the real souls — with the SAME seed for both arms, so each arm
plays the identical match and only the brain set differs.

⚠️  ONE PROCESS PER MATCH.  The brain registry, the per-position brain cache
and the mind registry are MODULE state: the first version of this gate ran
both arms in one process and the second arm inherited the first arm's
brains, so the two arms produced byte-identical matches (124 shots each,
11 goals each) and the gate reported "the brains make no difference". They
make a very large difference. Each match is therefore its own subprocess
(`_diag_roster_one.py`), and the metrics come from the runner's own export.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_roster_gate.py \\
        brains_v7 brains --seeds 21,22,23,24,25
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "tools", "diag", "_diag_roster_one.py")


def run_one(brains_dir: str, seed: int) -> dict:
    p = subprocess.run(
        [sys.executable, WORKER, brains_dir, str(seed)],
        cwd=ROOT, capture_output=True, text=True,
        # THE LOCALE'S CODEC IS NOT UTF-8 ON THIS MACHINE (cp1252), and the
        # runner prints emoji — so the default `text=True` decoding raises
        # UnicodeDecodeError in the reader thread and the gate dies with a
        # NoneType stdout.  Decode explicitly.
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no result from {brains_dir} seed {seed}:\n"
                           f"{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("brains_a")
    ap.add_argument("brains_b")
    ap.add_argument("--seeds", default="21,22,23,24,25")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    for arm, brains in ((args.label_a, args.brains_a), (args.label_b, args.brains_b)):
        rows = []
        for seed in seeds:
            r = run_one(brains, seed)
            rows.append(r)
            print(f"  {arm:<4} seed {seed}: {r['score'][0]}-{r['score'][1]}  "
                  f"shots {r['shots']}  xG {r['xg']:.2f}  "
                  f"offsides {r['offsides']}")
        results[arm] = rows

    a, b = results[args.label_a], results[args.label_b]
    print()
    print("=" * 74)
    print(f"{'metric':<36}{args.label_a:>19}{args.label_b:>19}")
    print("-" * 74)
    for key, label in (("home_goals", "home goals"),
                       ("away_goals", "goals conceded"),
                       ("shots", "shots"),
                       ("offsides", "offsides")):
        print(f"{label:<36}{sum(r[key] for r in a):>19}"
              f"{sum(r[key] for r in b):>19}")
    for key, label in (("poss", "possession % (mean)"),
                       ("xg", "xG (mean)")):
        print(f"{label:<36}{sum(r[key] for r in a) / len(a):>19.1f}"
              f"{sum(r[key] for r in b) / len(b):>19.1f}")
    for arm, rows in ((args.label_a, a), (args.label_b, b)):
        w = sum(1 for r in rows if r["score"][0] > r["score"][1])
        d = sum(1 for r in rows if r["score"][0] == r["score"][1])
        print(f"{'wins/draws/losses (' + arm + ')':<36}"
              f"{w:>10} / {d:>3} / {len(rows) - w - d:<3}")
    print("=" * 74)
    print(f"seeds {seeds} — each arm played the IDENTICAL matches "
          f"(one process per match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
