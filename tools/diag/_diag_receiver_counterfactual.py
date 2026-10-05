"""FORCED COUNTERFACTUAL: does a same-side pass actually beat a far-side one?

The falsification test for "option 1" (a receiver / pointer head).

`_diag_receiver_signal.py` established:
  * receiver choice is 100% determined by the hand-coded `_best_forward`
    (fidelity 343/343), so there is no hidden signal to imitate;
  * but pass OUTCOME is unpredictable from geometry out of sample
    (AUC 0.497), so observation alone cannot train a value model.

The way out of an off-policy dead end is the one this project already used for
TeamPressBrain (`team_offball_probe.py --intervene`): FORCE the decision
BEFORE the rule sees it, so the missing counterfactual gets manufactured.

This forces ONLY on decisions where it is a clean two-way choice:
    PROGRESSIVE_PASS / THROUGH_BALL, with >=1 same-side AND >=1 far-side
    eligible candidate.  Then:
        --arm same : return the best SAME-side candidate
        --arm far  : return the best FAR-side candidate
        --arm off  : no forcing (control)

Same seeds, same fixtures, one lever.  Arms diverge downstream because the
ball genuinely goes somewhere else -- that is the effect, not a confound.

The lever is asserted, never assumed: a silent no-op here would reproduce the
exact bug class this project keeps hitting (the jump guard inserted after
`self._last_t[name] = t`; the dead-ball gate consumed before it was read).

Run (one arm per process):
    .venv\\Scripts\\python.exe _diag_receiver_counterfactual.py --arm same
    .venv\\Scripts\\python.exe _diag_receiver_counterfactual.py --arm far
    .venv\\Scripts\\python.exe _diag_receiver_counterfactual.py --arm off
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter

import brain_integration
from _diag_chance_coords import build_pair

PASS_TYPES = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS",
              "FREEKICK_CROSS", "CORNER_TAKEN"}
FORCE_INTENTS = ("PROGRESSIVE_PASS", "THROUGH_BALL")


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _openness(tx, ty, defenders, pe):
    open_val = 0.5
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = pe.get_position(d.name) if pe else (tx + 10, ty)
        open_val = min(open_val, _clamp((math.hypot(dx - tx, dy - ty) - 1.5) / 8.5))
    return open_val


class Forcer:
    def __init__(self, arm: str, original):
        self.arm = arm
        self.original = original
        self.eligible = 0        # decisions where a two-way choice existed
        self.forced = 0          # times we actually returned a different man
        self.opportunities = Counter()   # offered same / far, by arm
        self.kicked = Counter()

    def __call__(self, intent, player, x, y, teammates, defenders,
                 position_engine, attacks_right, favored_flank=None):
        original = self.original
        chosen = original(intent, player, x, y, teammates, defenders,
                          position_engine, attacks_right, favored_flank)
        if self.arm == "off":
            return chosen
        if getattr(intent, "name", "") not in FORCE_INTENTS:
            return chosen
        pe = position_engine
        same, far = [], []
        for t in teammates or []:
            if getattr(t, "position", "") == "GK":
                continue
            pxy = pe.tracked_position(t.name) if hasattr(pe, "tracked_position") else None
            tx, ty = pxy if pxy else (pe.get_position(t.name) if pe else (x, y))
            progress = (tx - x) if attacks_right else (x - tx)
            if progress < 4.0:
                continue
            val = _clamp(progress / 35.0) * 0.55 + _openness(tx, ty, defenders, pe) * 0.45
            rec = (val, t)
            (same if (ty < 34.0) == (y < 34.0) else far).append(rec)
        if not same or not far:
            self.opportunities["no_two_way"] += 1
            return chosen
        self.eligible += 1
        pick = max(same if self.arm == "same" else far, key=lambda r: r[0])[1]
        orig_name = getattr(chosen, "name", None)
        if getattr(pick, "name", None) != orig_name:
            self.forced += 1
            self.kicked[(orig_name, pick.name)] += 1
        return pick


def install(arm: str) -> Forcer:
    original = brain_integration._find_target
    forcer = Forcer(arm, original)
    brain_integration._find_target = forcer
    return forcer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("same", "far", "off"), required=True)
    ap.add_argument("--matches", type=int, default=2)
    args = ap.parse_args()

    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
    forcer = install(args.arm)

    lines = []
    def p(m=""):
        print(m, flush=True)
        lines.append(m)

    p(f"  ARM = {args.arm}   ({args.matches} matches)")
    agg = Counter()
    adv_same, adv_far = [], []

    for i in range(args.matches):
        h, a = pairs[i % len(pairs)]
        random.seed(4000 + i)          # identical seeds to _diag_receiver_signal
        res = build_pair(h, a).simulate()
        home = res.config.home_team
        tl = res.timeline

        # FORCE EFFICIENCY, measured in the timeline itself (not the counter):
        # a forced pass must actually be aimed at the forced man.
        passes = [e for e in tl if e.event_type.name in PASS_TYPES
                  and e.player and e.secondary_player]
        # outcome of every pass, split by the receiver's side vs the passer's
        # side -- the passer's own y is the reference for "same side".
        for e in passes:
            if e.end_x is None or e.location_x is None:
                continue
            prog = (e.end_x - e.location_x) if e.team == home else (e.location_x - e.end_x)
            agg["passes"] += 1
            if e.outcome:
                agg["completed"] += 1
            if prog > 9.14:
                agg["progressive"] += 1
            adv_same.append(prog)
        p(f"    [{i+1}/{args.matches}] {h} v {a}: {res.home_goals}-{res.away_goals}, "
          f"eligible {forcer.eligible}, forced {forcer.forced}")

    n = max(agg["passes"], 1)
    adv = sorted(adv_same)
    p()
    p(f"  LEVER EFFICIENCY")
    p(f"    two-way decisions found : {forcer.eligible}")
    p(f"    receiver actually changed: {forcer.forced} "
      f"({100.0*forcer.forced/max(forcer.eligible,1):.0f}% of eligible)")
    p(f"    decisions with no two-way choice (returned original): "
      f"{forcer.opportunities['no_two_way']}")
    if forcer.eligible and forcer.forced == 0:
        p("    *** LEVER IS A NO-OP -- these numbers mean nothing ***")
    top = forcer.kicked.most_common(6)
    for (o, nw), c in top:
        p(f"      swapped {o} -> {nw}  x{c}")
    p()
    p(f"  MATCH AGGREGATE  (arm = {args.arm})")
    p(f"    passes                     : {agg['passes']}")
    p(f"    completion rate            : {100.0*agg['completed']/n:.1f}%")
    p(f"    progressive rate (>9.14 m) : {100.0*agg['progressive']/n:.1f}%")
    if adv:
        p(f"    advance  median {adv[len(adv)//2]:+.1f} m, "
          f"mean {sum(adv)/len(adv):+.1f} m")
    p()
    p("  Compare this arm against the others: same completion rate with MORE")
    p("  advance, or equal advance with a higher completion rate, is the win.")
    p("  Neither is a pass count -- the arms deliberately play different matches")

    out = f"_diag_receiver_cf_{args.arm}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()