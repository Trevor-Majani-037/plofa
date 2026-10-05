"""DECISIVE: does changing `_find_target`'s return change the MATCH at all?

Two arms, ONE process, NO importlib.reload (the previous lever probe reloaded
`brain_integration`, which re-executes the module and rebinds
`NeuralDecisionBrain`; `event_chain` holds the OLD class, so the patch may have
been installed on an object nothing called -- a probe bug that reads exactly
like a real "the lever is inert" finding).

This also records, at the CONSUMPTION site, whether the name the policy returned
is the name that received the ball. If they agree, the lever is live and the
earlier identical digest was this probe's own bug.

Run:  .venv\\Scripts\\python.exe _diag_lever_decisive.py
"""
from __future__ import annotations

import hashlib
import random

import brain_integration
from _diag_chance_coords import build_pair
from _diag_receiver_counterfactual import Forcer

PASS_TYPES = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS"}


def digest(tl):
    h = hashlib.sha256()
    for e in tl:
        h.update(
            f"{e.event_type.name}|{e.team}|{e.player}|{e.secondary_player}|"
            f"{e.location_x:.2f},{e.location_y:.2f}|"
            f"{'' if e.end_x is None else f'{e.end_x:.2f}'}|{int(bool(e.outcome))}"
            .encode()
        )
    return h.hexdigest()[:16]


def run(arm, seed):
    original = brain_integration._find_target
    forcer = Forcer(arm, original)
    brain_integration._find_target = forcer
    try:
        random.seed(seed)
        res = build_pair("Oxton", "Natrican").simulate()
    finally:
        brain_integration._find_target = original
    return res, forcer


def main():
    lines = []
    def p(m=""):
        print(m)
        lines.append(m)

    res_off, f_off = run("off", 4000)
    res_far, f_far = run("far", 4000)

    d_off, d_far = digest(res_off.timeline), digest(res_far.timeline)
    p(f"  arm off : {len(res_off.timeline):>5} events  digest {d_off}  "
      f"score {res_off.home_goals}-{res_off.away_goals}  forced {f_off.forced}")
    p(f"  arm far : {len(res_far.timeline):>5} events  digest {d_far}  "
      f"score {res_far.home_goals}-{res_far.away_goals}  forced {f_far.forced}")
    p()
    same = d_off == d_far
    p(f"  TIMELINES IDENTICAL? {same}")
    if not same:
        p("  >>> THE LEVER IS LIVE. The earlier identical digest was the")
        p("      importlib.reload probe bug, NOT an inert lever.")
        p(f"      ({f_far.forced} receivers swapped and the match changed.)")
    else:
        p("  >>> LEVER STILL INERT with no reload involved. The forced target is")
        p("      genuinely discarded after `_find_target` returns.")

    # Cross-check: did the forced men actually receive passes in the far arm?
    p()
    p("  Did the forced receivers actually get the ball?")
    got = 0
    for (orig, new), c in f_far.kicked.most_common(8):
        hits = [e for e in res_far.timeline
                if e.event_type.name in PASS_TYPES and e.player == orig
                and e.secondary_player == new]
        got += len(hits)
        if hits:
            p(f"    {orig:<20} -> {new:<20} {len(hits)} pass(es)")
    p(f"  passes delivered to a forced receiver: {got}")
    p("  (arm off, same pairs, as the control:)")
    got0 = 0
    for (orig, new), c in f_far.kicked.most_common(8):
        hits = [e for e in res_off.timeline
                if e.event_type.name in PASS_TYPES and e.player == orig
                and e.secondary_player == new]
        got0 += len(hits)
    p(f"  passes delivered to the same pairs in arm off: {got0}")

    with open("_diag_lever_decisive.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_lever_decisive.txt")


if __name__ == "__main__":
    main()