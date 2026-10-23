"""THE REAL-ROSTER GATE for the situational set — the shipping question.

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_situational_roster_gate.py \
        brains_v8 --seeds 21,22,23,24,25,26,27

WHY A SEPARATE PROBE
--------------------
The SCRATCH harness (``_diag_v7_gate.py``) compares brain SETS against
themselves — its fixture gives both teams the same generic names
(GK, CB1, ...), so a brain set bound to the home XI is also found by the
away XI.  That is comparable in kind with the historical gates, but it is
NOT "our team against a fixed opponent", and the AGENTS.md verdict is that
**the real-roster gate answers the shipping question**.

So this runs two REAL clubs (Natrican v Tryox City by default), binds the
challenger **by name to the home XI only**, and leaves the auto-load on
the shipped set so **the away side is the same control in both arms**.

ONE PROCESS PER MATCH
---------------------
The brain registry, the per-position brain cache and the mind registry
are MODULE state.  The first version of the real-roster gate ran both
arms in ONE process and the second arm inherited the first arm's brains:
the two arms produced BYTE-IDENTICAL matches and the gate reported
"the brains make no difference".  They make a very large difference.  So
each match here is a SUBPROCESS.

THE CONTROL ARM
----------------
The control binds the SHIPPED set to the home XI, so the control is
"shipped vs shipped" and the challenger arm is "new vs shipped" — the
away side is the shipped set in BOTH arms, so the only difference
between the arms is the home XI's brain.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))

RUNNER = os.path.join(ROOT, "tools", "diag", "_diag_roster_one.py")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable


def run_one(brains: str, seed: int, kind: str, home: str, away: str,
            perception: bool) -> dict:
    """One match, in its OWN process."""
    proc = subprocess.run(
        [PY, RUNNER, brains, str(seed), home, away, kind, "on" if perception else "off"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"match failed (seed {seed}, {kind}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no result line for seed {seed} ({kind}):\n"
                       f"{proc.stdout[-2000:]}")


def read_geometry(out_dir: str, home: str) -> dict:
    """The pass geometry and the shape, read off the export."""
    import glob
    import math

    jsons = glob.glob(os.path.join(out_dir, "*.json"))
    data = None
    for j in jsons:
        with open(j, encoding="utf-8") as fh:
            d = json.load(fh)
        if "timeline" in d:
            data = d
            break
    if data is None:
        return {}
    timeline = data.get("timeline", [])
    fwd = sq = back = 0
    lengths = []
    advances = []
    for e in timeline:
        if e.get("team") != home:
            continue
        if e.get("type") not in ("PASS", "PROGRESSIVE_PASS", "THROUGH_BALL",
                                  "SWITCH_OF_PLAY", "CROSS_ATTEMPT",
                                  "CROSS_SUCCESS"):
            continue
        # THE EXPORT'S FIELD NAMES ARE x / y, not location_x / location_y
        # (it is the PLOFA format, not StatsBomb's) — reading the wrong
        # key is the documented trap and it produces a clean, empty
        # answer rather than an error.
        x0 = e.get("x")
        x1 = e.get("end_x")
        y0 = e.get("y")
        y1 = e.get("end_y")
        if x0 is None or x1 is None or y0 is None or y1 is None:
            continue
        # The progression band: forward if it closes >=10% of the
        # distance to the opponent's goal.  The EXPORT's frame is
        # "home attacks right" (the PLOFA export's own convention), so
        # the home team's goal is at x=105.
        gx = 105.0
        d0 = math.hypot(gx - float(x0), 34.0 - float(y0))
        d1 = math.hypot(gx - float(x1), 34.0 - float(y1))
        prog = ((d0 - d1) / d0) if d0 > 0 else 0.0
        if prog >= 0.10:
            fwd += 1
        elif prog <= -0.10:
            back += 1
        else:
            sq += 1
        lengths.append(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
        advances.append(float(x1) - float(x0))
    total = max(1, fwd + sq + back)
    lengths.sort()
    advances.sort()
    return {
        "forward_pct": 100.0 * fwd / total,
        "square_pct": 100.0 * sq / total,
        "backward_pct": 100.0 * back / total,
        "median_pass_len": (lengths[len(lengths) // 2] if lengths else 0.0),
        "median_advance": (advances[len(advances) // 2] if advances else 0.0),
        "passes": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("brains")
    ap.add_argument("--seeds", default="21,22,23,24,25,26,27")
    ap.add_argument("--home", default="Natrican")
    ap.add_argument("--away", default="Tryox City")
    ap.add_argument("--control-dir", default="brains")
    ap.add_argument("--perception", choices=("off", "on"), default="off",
                    help="'off' = identity regime (default), 'on' = production condition")
    A = ap.parse_args()

    seeds = [int(s) for s in A.seeds.split(",") if s.strip()]
    out_dir = os.path.join(ROOT, "plofa_output", "_roster_gate")

    rows = []
    perception = A.perception == "on"
    for seed in seeds:
        for label, brains, kind in (("control", A.control_dir, "mlp"),
                                    ("challenger", A.brains, "situational")):
            res = run_one(brains, seed, kind, A.home, A.away, perception)
            geo = read_geometry(
                os.path.join(out_dir, f"seed{seed}_{os.path.basename(brains)}"),
                A.home)
            res["label"] = label
            res["seed"] = seed
            res.update(geo)
            rows.append(res)
            print(f"seed {seed} {label:11s} {res.get('score', '?')}  "
                  f"shots {res.get('shots')}  poss {res.get('poss'):.1f}  "
                  f"xg {res.get('xg', 0):.2f}  "
                  f"fwd {geo.get('forward_pct', 0):.1f}%  "
                  f"med_len {geo.get('median_pass_len', 0):.1f}m",
                  flush=True)

    # the summary
    print()
    for label in ("control", "challenger"):
        arm = [r for r in rows if r["label"] == label]
        if not arm:
            continue
        n = len(arm)
        goals = sum(r.get("home_goals", 0) for r in arm)
        conceded = sum(r.get("away_goals", 0) for r in arm)
        wins = sum(1 for r in arm
                   if r.get("home_goals", 0) > r.get("away_goals", 0))
        draws = sum(1 for r in arm
                    if r.get("home_goals", 0) == r.get("away_goals", 0))
        losses = n - wins - draws
        poss = sum(r.get("poss", 0.0) for r in arm) / n
        xg = sum(r.get("xg", 0.0) for r in arm) / n
        fwd = sum(r.get("forward_pct", 0.0) for r in arm) / n
        bwd = sum(r.get("backward_pct", 0.0) for r in arm) / n
        mlen = sum(r.get("median_pass_len", 0.0) for r in arm) / n
        madv = sum(r.get("median_advance", 0.0) for r in arm) / n
        print(f"{label:11s} {wins}W {draws}D {losses}L  "
              f"GF {goals} GA {conceded}  xG {xg:.2f}  poss {poss:.1f}%  "
              f"fwd {fwd:.1f}%  bwd {bwd:.1f}%  "
              f"med_len {mlen:.1f}m  med_adv {madv:+.2f}m")

    with open(os.path.join(ROOT, "artifacts",
                           "_diag_situational_roster_gate.json"),
              "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
