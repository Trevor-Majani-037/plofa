"""DID THE FORCING LEVER ACTUALLY REACH THE PASS, OR ONLY A FIELD NOTHING READS?

Arm `far` reported 152 receiver changes, yet its match aggregate was
IDENTICAL to arm `off` in every digit (1460 passes, 92.2%, 20.8%, +1.2/+2.0
median).  Those two facts cannot both be true.  Either the lever changed a
field that nothing downstream consumes, or the aggregate was computed from
something the lever cannot touch.

This is the project's standing trap: `PLOFA-...export` calls it "DECISION IS
NOT EXECUTION", and the chance-creation work recorded "a fixed field is not a
fixed feature" -- you changed `goal_assistant` and the GOAL event's
`secondary_player` still carried the fabrication.

So: measure whether the timeline itself differs, and whether any forced
receiver actually appears as the receiver of a pass.

Run:  .venv\\Scripts\\python.exe _diag_receiver_lever.py
"""
from __future__ import annotations

import hashlib
import random

import brain_integration
from _diag_chance_coords import build_pair
from _diag_receiver_counterfactual import Forcer, install

PASS_TYPES = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS",
              "FREEKICK_CROSS", "CORNER_TAKEN"}


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


def run(arm, seed, home="Oxton", away="Natrican"):
    # fresh import state each arm
    import importlib
    import brain_integration as bi
    importlib.reload(bi)
    original = bi._find_target
    forcer = Forcer(arm, original)
    bi._find_target = forcer
    try:
        random.seed(seed)
        res = build_pair(home, away).simulate()
    finally:
        bi._find_target = original
    return res, forcer


def main():
    lines = []

    def p(m=""):
        print(m)
        lines.append(m)

    res_off, _ = run("off", 4000)
    res_far, forcer_far = run("far", 4000)

    d_off = digest(res_off.timeline)
    d_far = digest(res_far.timeline)
    p(f"  arm off : {len(res_off.timeline)} events, digest {d_off}, "
      f"score {res_off.home_goals}-{res_off.away_goals}")
    p(f"  arm far : {len(res_far.timeline)} events, digest {d_far}, "
      f"score {res_far.home_goals}-{res_far.away_goals}")
    p(f"  lever   : {forcer_far.eligible} eligible, {forcer_far.forced} forced")
    p()
    p(f"  TIMELINES IDENTICAL? {d_off == d_far}")
    if d_off == d_far:
        p("  *** THE LEVER NEVER REACHED THE MATCH. ***")
        p("  Forcing `_find_target` changed a value that no code path reads")
        p("  when it builds the pass. The pass receiver is chosen elsewhere.")

    # Did any forced receiver actually RECEIVE a pass from that carrier?
    p()
    p("  Did the forced men receive the ball?")
    swaps = forcer_far.kicked.most_common(10)
    got = 0
    checked = 0
    for (orig, new), _c in swaps:
        hits = [e for e in res_far.timeline
                if e.event_type.name in PASS_TYPES and e.player == orig
                and e.secondary_player == new]
        got += len(hits)
        checked += 1
        if hits:
            p(f"    {orig} -> {new}: {len(hits)} pass(es)")
    p(f"  forced swaps checked {checked}, passes actually delivered to the "
      f"forced man: {got}")

    # Where DOES the receiver come from? Name the field.
    p()
    p("  Does the engine use PlayerDecision.target at all?")
    src = open("event_chain.py", encoding="utf-8", errors="replace").read()
    n_target = src.count(".target")
    n_dp = src.count("_pass_destination")
    n_bf = src.count("_best_forward")
    p(f"    occurrences of '.target' in event_chain.py      : {n_target}")
    p(f"    occurrences of '_pass_destination'              : {n_dp}")
    p(f"    occurrences of '_best_forward'                  : {n_bf}")
    p("    (a second, independent target lookup would explain an inert lever)")

    with open("_diag_receiver_lever.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_receiver_lever.txt")


if __name__ == "__main__":
    main()