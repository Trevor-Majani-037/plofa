"""THE WHY: why can't a real centre-back's pass map be reproduced here?

The premise test is over. `_diag_cb_corridor_ab.py` showed CBs ALREADY pass
same-side 94.5% of the time, so "same-side corridor" was never the gap. The
A/B that followed showed the only real discriminator is:

    before:  completion 93.1%   forward share 49.5%   advance -0.0 m
    after :  completion 90.7%   forward share 55.2%   advance +0.9 m

Real elite CBs (Le Normand 27/28 in the opposition half at 96%; Laporte 31/33
at 94%) hold BOTH: very high completion AND forward progression. Mine complete
almost as much but go NOWHERE -- median advance 0.0 m, under half forward.

So the question is NOT "which teammate". It is:

    WHY is there no forward pass available to a CB in this engine?

Three candidate answers, measured not guessed:

  A. SHAPE — nobody is positioned AHEAD of the CB, so there is nothing forward
     to pass to. Earlier measurement: only 13% of CBs in their own half had a
     same-side forward option available AT ALL.
  B. REST DEFENCE — the 2026-09-29 invariant clamps the shallowest man ahead of
     the ball to 12 m BEHIND it. If that is clamping the man who would receive
     a CB's forward pass, the rule would be actively suppressing the answer.
  C. LENGTH — the pass itself is aimed too far/long to be a short progressive
     ball, so "forward" is unavailable by construction.

This measures, per CB decision in own half: how many teammates are ahead, how
far, and whether rest-defence clamping is the reason a candidate is not ahead.

Observation-only. Run:  .venv\\Scripts\\python.exe _diag_cb_forward_why.py
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

import brain_integration
import position_engine
from _diag_chance_coords import build_pair
from _diag_receiver_counterfactual import _clamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest-defence", choices=("on", "off"), default="on")
    ap.add_argument("--opp", choices=("balanced", "lowblock"), default="lowblock",
                    help="the REFERENCE data is a CB vs a LOW BLOCK -- "
                         "Le Normand 27/28 in the opposition half at 96%%, "
                         "Laporte 31/33 at 94%% were produced against ten men "
                         "behind the ball, NOT against a balanced or high-")
    ap.add_argument("--opp-high-line", action="store_true",
                    help="opp plays ATTACKING (the deliberately WRONG frame, "
                         "kept so the comparison can be shown to be unfair)")
    args = ap.parse_args()

    """Wrap AttackingMatrix._build_options: at the moment a CB considers his
    options, record who is ahead of him and on which side. This is the live
    candidate set at the live instant -- reading positions after simulate()
    would attribute final-whistle state to a mid-match moment."""
    import attacking_matrix

    STATS = Counter()
    GAPS = []
    CLAMPED = Counter()

    if args.rest_defence == "off":
        position_engine.PositionEngine.REST_DEFENCE_ENABLED = False

    from match_engine import PlayingStyle as PS, TeamStyle as TS
    if args.opp == "lowblock":
        away_style, away_play = TS.PARK_THE_BUS, PS.LOW_BLOCK
    else:
        away_style, away_play = TS.BALANCED, PS.MIXED
    if args.opp_high_line:
        away_style, away_play = TS.ATTACKING, PS.HIGH_PRESS

    raw = attacking_matrix.AttackingMatrix.__dict__["_build_options"]

    def build(*a, **kw):
        opts = raw.__func__(*a, **kw)
        try:
            carrier = a[0]
            if not str(getattr(carrier, "position", "")).rstrip("0123456789") == "CB":
                return opts
            x, y = a[3], a[4]
            teammates = a[1]
            pe = a[5]
            ar = a[6]
            STATS["cb_touches"] += 1
            if not ar:
                STATS["cb_away_side"] += 1
            fwd_same = fwd_far = back = 0
            best_fwd = None
            for tm in teammates or []:
                if str(getattr(tm, "position", "")).rstrip("0123456789") == "GK":
                    continue
                pxy = pe.tracked_position(tm.name) if hasattr(pe, "tracked_position") else None
                if not pxy:
                    continue
                tx, ty = pxy
                prog = (tx - x) if ar else (x - tx)
                same = (ty < 34.0) == (y < 34.0)
                if prog > 2.0:
                    if same:
                        fwd_same += 1
                        if best_fwd is None or prog > best_fwd:
                            best_fwd = prog
                    else:
                        fwd_far += 1
                elif prog < -2.0:
                    back += 1
            STATS["cb_touches_with_fwd_same"] += 1 if fwd_same else 0
            STATS["cb_touches_with_fwd_any"] += 1 if (fwd_same or fwd_far) else 0
            STATS["fwd_same_total"] += fwd_same
            STATS["fwd_far_total"] += fwd_far
            STATS["behind_total"] += back
            if best_fwd is not None:
                GAPS.append(best_fwd)
        except Exception as exc:  # noqa: BLE001
            STATS[f"err:{type(exc).__name__}"] += 1
        return opts

    attacking_matrix.AttackingMatrix._build_options = staticmethod(build)
    try:
        random.seed(4000)
        res = build_pair("Oxton", "Natrican",
                         away_team_style=away_style,
                         away_playing_style=away_play).simulate()
    finally:
        attacking_matrix.AttackingMatrix._build_options = raw
        position_engine.PositionEngine.REST_DEFENCE_ENABLED = True

    lines = []

    def p(m=""):
        print(m)
        lines.append(m)

    p(f"  REST_DEFENCE_ENABLED = {args.rest_defence}")
    p(f"  OPPONENT FRAME = {args.opp}"
      + (" + HIGH LINE (deliberately WRONG frame)"
         if args.opp_high_line else ""))
    p(f"  match {res.home_goals}-{res.away_goals}, {len(res.timeline)} events")
    p()
    p(f"  CB touches observed            : {STATS['cb_touches']}")
    p(f"  with >=1 SAME-SIDE man AHEAD   : {STATS['cb_touches_with_fwd_same']} "
      f"({100.0*STATS['cb_touches_with_fwd_same']/max(STATS['cb_touches'],1):.1f}%)")
    p(f"  with >=1 man AHEAD (any side)  : {STATS['cb_touches_with_fwd_any']} "
      f"({100.0*STATS['cb_touches_with_fwd_any']/max(STATS['cb_touches'],1):.1f}%)")
    p()
    p(f"  mean men ahead, same side      : {STATS['fwd_same_total']/max(STATS['cb_touches'],1):.2f}")
    p(f"  mean men ahead, far side       : {STATS['fwd_far_total']/max(STATS['cb_touches'],1):.2f}")
    p(f"  mean men BEHIND                : {STATS['behind_total']/max(STATS['cb_touches'],1):.2f}")
    p()
    if GAPS:
        g = sorted(GAPS)
        p(f"  nearest same-side man AHEAD by : median {g[len(g)//2]:.1f} m, "
          f"min {g[0]:.1f}, max {g[-1]:.1f}")
        p(f"  (real: a CB's pivot sits roughly 10-25 m ahead of him)")
    p()
    share = STATS["cb_touches_with_fwd_same"] / max(STATS["cb_touches"], 1)
    p(f"  men BEHIND per CB touch    : "
      f"{STATS['behind_total']/max(STATS['cb_touches'],1):.2f}")
    p()
    p("  COMPARE ACROSS ARMS: mean men AHEAD same-side, and mean BEHIND.")
    p("  If turning REST_DEFENCE off raises men-AHEAD and lowers men-BEHIND,")
    p("  the rule added 2026-09-29 is manufacturing the too-deep shape.")
    p("  'SHAPE IS NOT THE BLOCKER' below only tested AVAILABILITY, not")
    p("  PROPORTION -- the pivot can exist and still be a minority option.")
    if STATS:
        errs = {k: v for k, v in STATS.items() if k.startswith("err:")}
        if errs:
            p(f"  (probe errors: {errs})")

    tag = f"{args.rest_defence}_{args.opp}" + ("_highline" if args.opp_high_line else "")
    with open(f"_diag_cb_forward_why_{tag}.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote _diag_cb_forward_why_{tag}.txt")


if __name__ == "__main__":
    main()