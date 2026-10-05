"""DIRECT: did the man `_find_target` returned become the man who received?

Two recorders, one match, joined on the player's name:

  A. `_find_target` -> the name the POLICY asked for (forced arm)
  B. `PossessionChain._pass_destination_to_receiver` -> the name the ball was
     actually aimed at, i.e. the receiver that governed the endpoint

This is the join `_diag_receiver_lever.py` could not make, because it looked
for the forced man in the TIMELINE (secondary_player) rather than at the
function that consumes the choice. Run:  .venv\\Scripts\\python.exe _diag_target_join.py
"""
from __future__ import annotations

import random
from collections import Counter

import brain_integration
import event_chain
from _diag_chance_coords import build_pair
from _diag_receiver_counterfactual import Forcer

POLICY_ASKED = Counter()      # carrier -> name the policy returned
ACTUAL_RECEIVER = Counter()   # carrier -> name the ball was aimed at
CALLS = Counter()


def main():
    lines = []
    def p(m=""):
        print(m)
        lines.append(m)

    original_find = brain_integration._find_target
    forcer = Forcer("far", original_find)

    def recording_find(*a, **kw):
        r = forcer(*a, **kw)
        try:
            carrier = getattr(a[1], "name", None)
            if carrier:
                POLICY_ASKED[(carrier, getattr(r, "name", None))] += 1
        except Exception:
            pass
        return r

    brain_integration._find_target = recording_find

    cls = event_chain.PossessionChain
    raw_dest = cls.__dict__["_pass_destination_to_receiver"]

    def recording_dest(cls_, receiver, x, y, *a, **kw):
        CALLS["dest"] += 1
        name = getattr(receiver, "name", None)
        if name:
            ACTUAL_RECEIVER[name] += 1
        if isinstance(raw_dest, classmethod):
            return raw_dest.__func__(cls_, receiver, x, y, *a, **kw)
        return raw_dest(receiver, x, y, *a, **kw)

    cls._pass_destination_to_receiver = classmethod(recording_dest)

    try:
        random.seed(4000)
        res = build_pair("Oxton", "Natrican").simulate()
    finally:
        brain_integration._find_target = original_find
        cls._pass_destination_to_receiver = raw_dest

    p(f"  _pass_destination_to_receiver calls : {CALLS['dest']}")
    p(f"  policy asks recorded               : {sum(POLICY_ASKED.values())}")
    p(f"  distinct receivers aimed at         : {len(ACTUAL_RECEIVER)}")
    p(f"  forcer: eligible {forcer.eligible}, changed {forcer.forced}")
    p()

    # The forced names, as a set.
    forced_new = {new for (_o, new), c in forcer.kicked.items()}
    forced_orig = {o for (o, _n), c in forcer.kicked.items()}
    aimed = set(ACTUAL_RECEIVER)

    p(f"  names the forcing arm SWAPPED TO   : {len(forced_new)}")
    p(f"  ...of those, ever AIMED the ball   : "
      f"{len(forced_new & aimed)}")
    for n in sorted(forced_new)[:10]:
        p(f"      {n:<22} aimed {ACTUAL_RECEIVER.get(n, 0)} times")
    p()
    p(f"  names the arm swapped AWAY FROM    : {len(forced_orig)}")
    p(f"  ...of those, ever AIMED the ball   : "
      f"{len(forced_orig & aimed)}")
    p()
    if forced_new & aimed:
        p("  >>> the forced receivers DID receive the ball -> the lever is live")
        p("      and an identical digest elsewhere has a different cause.")
    else:
        p("  >>> NOT ONE forced receiver ever received the ball.")
        p("      The name `_find_target` returns is discarded before the")
        p("      endpoint is computed.")

    with open("_diag_target_join.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_target_join.txt")


if __name__ == "__main__":
    main()