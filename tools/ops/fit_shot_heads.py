"""Fit the per-position shot read-outs and write them where the game can find them.

Each position is fitted independently and each one is judged on its own numbers.
A position that fails is NOT written — the decision path will simply find no file
and behave exactly as it does today, so a refusal costs nothing and cannot be
mistaken for a success.

Outputs, under ``brains/shot_heads/``:
    <POS>.json          the artifact (33 params + provenance)
    _report.json        every position's trials, tier and verdict

Usage:
    python tools/ops/fit_shot_heads.py
    python tools/ops/fit_shot_heads.py --positions ST,CF,LW,RW,CAM --states 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import shot_head  # noqa: E402
from situational_brain import SituationalBrain  # noqa: E402

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit shot read-outs.")
    ap.add_argument("--positions", default=",".join(ALL_POSITIONS))
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--out", default=os.path.join("brains", "shot_heads"))
    args = ap.parse_args()

    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    os.makedirs(args.out, exist_ok=True)

    reports = {}
    for pos in positions:
        path = os.path.join(args.brains, f"{pos}.json")
        if not os.path.exists(path):
            print(f"{pos:4s} SKIP    no brain at {path}")
            continue
        brain = SituationalBrain.load(path)
        t0 = time.time()
        head, rep = shot_head.fit(brain, pos, n_states=args.states, seed=args.seed)
        rep["elapsed_s"] = round(time.time() - t0, 1)
        reports[pos] = rep

        tier = rep.get("tier")
        out = os.path.join(args.out, f"{pos}.json")
        if head is not None:
            head.save(out)
            print(f"{pos:4s} {tier:12s} -> {out}")
        else:
            # A REFUSED position must have NO head, not the last one that
            # happened to pass.  The contract is "no file and the position is
            # untouched", and leaving a stale artifact behind breaks it in the
            # worst way: the fitter says no, the loader still finds a file, and
            # the refused mechanism keeps playing.  That is precisely how the
            # beta-4.0 ST head survived one refit after it was known to be bad.
            if os.path.exists(out):
                os.remove(out)
                print(f"{pos:4s} {tier:12s} (stale head REMOVED from {out})")
            else:
                print(f"{pos:4s} {tier:12s} (nothing written)")
        print(f"      {rep['verdict']}")

    with open(os.path.join(args.out, "_report.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)

    written = sorted(p for p, r in reports.items() if r.get("tier") in
                     ("ACCEPTED", "PROVISIONAL"))
    print(f"\nwrote {len(written)} head(s): {', '.join(written) or '(none)'}")
    print(f"report -> {os.path.join(args.out, '_report.json')}")
    print("\nThese heads are INERT until PLOFA_SHOT_HEAD=1 is set. A PROVISIONAL")
    print("head is not promotable without a real-match gate.")


if __name__ == "__main__":
    main()
