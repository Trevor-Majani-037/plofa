"""DO THE BRAINS PLAY REAL FOOTBALL IF THEY ARE THE ONLY DECIDERS?

Runs ONE match in ONE process and dumps metrics as JSON. The arm is an argv
flag, never an in-process toggle, because an A/B across two `simulate()` calls
in one process is INVALID: the second inherits the first's module-level brain
caches (`brain_integration._brain_registry`, `_pos_brain_cache`,
`cognition.mind._minds`), which is the seed-reproducibility bug biting exactly
where it hurts. Two arms therefore means two processes and an out-of-band diff.

    python _diag_brain_only.py --arm deciders   > out_deciders.json
    python _diag_brain_only.py --arm brain      > out_brain.json

`deciders` = today's engine, POLICY_INTENT_AUTHORITY False.
`brain`     = the neural brain owns receiver selection AND the delivery class;
              the TacticalPhase regression order and the wide-combo override
              no longer force `is_prog = False` over a sampled intent.

The metrics that matter most are the ones Checkpoint 32b already found
breaking when receiver authority was switched on: GK receptions and total event
count. If those collapse again the honest answer is that the deciders earn
their place, and the interesting number becomes WHICH of them earns it.
"""
import argparse
import json
import math
import random
from collections import Counter

import numpy as np

from _diag_watch import build

PASS_TYPES = ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY")


def geo(dx):
    return "forward" if dx > 5.0 else ("backward" if dx < -5.0 else "square")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("deciders", "brain"), required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="write the JSON here instead of stdout")
    a = ap.parse_args()

    if a.arm == "brain":
        from event_chain import PossessionChain
        PossessionChain.POLICY_INTENT_AUTHORITY = True

    out = []
    for m in range(a.matches):
        eng, hr, ar = build()
        eng.enable_virtual_gps(0.1)
        random.seed(a.seed + m * 100)
        # The engine narrates every goal and card to stdout, which makes
        # stdout unparseable as JSON. Swallow it for the duration of the match
        # and restore it before printing, rather than asking a shell to fish
        # the JSON out of the noise.
        import io
        import sys as _sys
        real = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            res = eng.simulate()
        finally:
            _sys.stdout = real
        H, A = eng.config.home_team, eng.config.away_team

        passes, intents, matrix, phases, recycles = [], Counter(), Counter(), \
            Counter(), Counter()
        gk_receipts = 0
        third = Counter()
        for e in res.timeline:
            t = getattr(e, "event_type", None)
            if t is None:
                continue
            md = getattr(e, "metadata", None) or {}
            if t.name in PASS_TYPES:
                x0 = getattr(e, "location_x", None)
                x1 = getattr(e, "end_x", None)
                if x0 is not None and x1 is not None:
                    minute = getattr(e, "minute", 0) or 0
                    right = (getattr(e, "team", "") == H) == (minute < 45)
                    dx = (x1 - x0) * (1.0 if right else -1.0)
                    passes.append(dict(
                        dx=dx,
                        dist=float(md.get("pass_length_m") or math.hypot(
                            x1 - x0, (getattr(e, "end_y", 0) or 0)
                            - (getattr(e, "location_y", 0) or 0))),
                        is_prog=bool(md.get("is_progressive")),
                        ax1=(x1 if right else 105.0 - x1),
                    ))
                    intents[(md.get("active_brain") or {}).get("intent", "?")] += 1
                    matrix[(md.get("attacking_matrix") or {}).get("action", "?")] += 1
                    phases[md.get("possession_phase", "?")] += 1
                    if md.get("recycle"):
                        recycles[md["recycle"]] += 1
                    third[md.get("end_third", "?")] += 1
            if t.name == "BALL_RECEIPT" and getattr(e, "player", "") and \
                    eng.position_engine.states.get(
                        getattr(e, "player", "")) is not None and \
                    eng.position_engine.states[
                        getattr(e, "player", "")].position == "GK":
                gk_receipts += 1

        d = [p["dx"] for p in passes]
        L = [p["dist"] for p in passes]
        g = Counter(geo(x) for x in d)
        n = len(passes) or 1
        out.append(dict(
            arm=a.arm, seed=a.seed + m * 100, score=res.score_str,
            possession=round(res.home_possession_pct, 1),
            xg=[round(res.home_xg, 2), round(res.away_xg, 2)],
            events=len(res.timeline),
            gk_receipts=gk_receipts,
            passes=len(passes),
            fwd=round(100 * g["forward"] / n, 1),
            sq=round(100 * g["square"] / n, 1),
            back=round(100 * g["backward"] / n, 1),
            median_dx=round(float(np.median(d)) if d else 0.0, 1),
            median_len=round(float(np.median(L)) if L else 0.0, 1),
            prog_pct=round(100 * sum(1 for p in passes if p["is_prog"]) / n, 1),
            final_third=round(100 * sum(1 for p in passes if p["ax1"] > 70) / n, 1),
            intents=dict(intents.most_common()),
            matrix=dict(matrix.most_common()),
            recycles=dict(recycles.most_common()),
            thirds=dict(third.most_common()),
        ))
    blob = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    else:
        print(blob)


if __name__ == "__main__":
    main()
