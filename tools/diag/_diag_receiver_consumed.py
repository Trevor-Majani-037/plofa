"""IS THE BRAIN'S CHOSEN RECEIVER EVER THE RECEIVER THAT ACTUALLY PLAYS THE PASS?

Forcing `_find_target` changed the returned name 70 times in one match and the
match timeline was BYTE-IDENTICAL (digest b075ecc2520f50ca, 2827 events, both
arms). One of those two facts is impossible. This finds out which consumption
path actually runs, by counting the two candidates for "who received it":

  (a) the policy target  -- `active_decision.target`, event_chain.py:2152-2159
  (b) the heuristic picker -- `BaseChain._pick_receiver`, event_chain.py:2161

Descriptor discipline matters here (project rule): read the raw descriptor out
of `Cls.__dict__` and branch on isinstance, because a `classmethod`/`staticmethod`
wrapped as a plain function shifts every positional argument.

Observation-only. Run:  .venv\\Scripts\\python.exe _diag_receiver_consumed.py
"""
from __future__ import annotations

import random

import event_chain
from _diag_chance_coords import build_pair

STATS = {"pick_receiver": 0, "policy_target_ok": 0, "policy_rejected": 0,
         "pass_emitted": 0}


def instrument():
    # `_pick_receiver` is a classmethod on PossessionChain, NOT BaseChain --
    # read off the AST rather than assuming the owner (BaseChain.__dict__
    # raises KeyError, which is how this was found).
    cls = event_chain.PossessionChain
    raw = cls.__dict__["_pick_receiver"]

    def counting(*a, **kw):
        STATS["pick_receiver"] += 1
        if isinstance(raw, classmethod):
            return raw.__func__(cls, *a, **kw)
        if isinstance(raw, staticmethod):
            return raw.__func__(*a, **kw)
        return raw(*a, **kw)

    cls._pick_receiver = staticmethod(counting)
    return raw


def main():
    original = instrument()
    try:
        random.seed(4000)
        res = build_pair("Oxton", "Natrican").simulate()
    finally:
        event_chain.PossessionChain._pick_receiver = original

    passes = [e for e in res.timeline if e.event_type.name == "PASS"]

    lines = []
    def p(m=""):
        print(m)
        lines.append(m)

    p(f"  PASS events in the timeline          : {len(passes)}")
    p(f"  _pick_receiver() calls               : {STATS['pick_receiver']}")
    p()
    if STATS["pick_receiver"] >= len(passes) * 0.9:
        p("  >>> THE HEURISTIC PICKER OWNS THE RECEIVER IN ESSENTIALLY EVERY PASS.")
        p("      `active_decision.target` (the neural brain's choice) is")
        p("      therefore NOT what determines who receives the ball.")
    else:
        p("  the policy target path is live at least sometimes -- the identical")
        p("  digest must then have another cause; do NOT conclude from this alone.")
    p()
    p("  This is the project's 'DECISION IS NOT EXECUTION' boundary, applied to")
    p("  RECEIVERS rather than to intents: the brain names a man, and a")
    p("  different code path picks the one who actually gets the ball.")

    with open("_diag_receiver_consumed.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_receiver_consumed.txt")


if __name__ == "__main__":
    main()