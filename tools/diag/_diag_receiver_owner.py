"""WHO ACTUALLY OWNS THE RECEIVER? (the forced_end branch)

`_pass_destination_to_receiver` fired ONCE in a full match, yet there are ~685
PASS events. Line 2295 explains it:

    if forced_end is not None:
        end_px, end_py = forced_end          # <- taken on almost every pass
    else:
        end_px, end_py = cls._pass_destination_to_receiver(receiver, ...)

`forced_receiver` / `forced_end` are set EARLIER (2111: `receiver =
forced_receiver`, before the brain's `active_decision.target` is even read at
2152), from `phase_decision.target` (1722) or `matrix_decision.target` (1817).

So this counts every candidate source of the receiver. Run:
    .venv\\Scripts\\python.exe _diag_receiver_owner.py
"""
from __future__ import annotations

import random
from collections import Counter

import event_chain
from _diag_chance_coords import build_pair

C = Counter()


def wrap(cls, name, label):
    raw = cls.__dict__[name]

    def f(cls_, *a, **kw):
        C[label] += 1
        if isinstance(raw, classmethod):
            return raw.__func__(cls_, *a, **kw)
        if isinstance(raw, staticmethod):
            return raw.__func__(*a, **kw)
        return raw(*a, **kw)

    setattr(cls, name, classmethod(f))
    return raw


def main():
    lines = []
    def p(m=""):
        print(m)
        lines.append(m)

    cls = event_chain.PossessionChain
    saved = {}
    for name, label in (
        ("_pass_destination_to_receiver", "endpoint from RECEIVER (brain target)"),
        ("_pass_destination_to_target", "endpoint from forced_end (phase/matrix)"),
        ("_pick_receiver", "receiver from heuristic picker"),
        ("_pick_wide_combo_target", "receiver from wide-combo rule"),
    ):
        try:
            saved[name] = wrap(cls, name, label)
        except KeyError:
            p(f"  (no {name} on PossessionChain)")

    try:
        random.seed(4000)
        res = build_pair("Oxton", "Natrican").simulate()
    finally:
        for name, raw in saved.items():
            setattr(cls, name, raw)

    passes = [e for e in res.timeline
              if e.event_type.name in ("PASS", "THROUGH_BALL", "CROSS",
                                       "SWITCH_PASS", "LONG_PASS")]
    p(f"  PASS-like events in the match : {len(passes)}")
    p()
    for k, v in C.most_common():
        p(f"    {v:>5}  {k}")
    p()
    total = C["endpoint from forced_end (phase/matrix)"]
    p(f"  passes whose endpoint came from the PHASE/MATRIX forced_end : "
      f"{total}/{len(passes)} ({100.0*total/max(len(passes),1):.0f}%)")
    p(f"  passes whose endpoint came from the brain's receiver       : "
      f"{C['endpoint from RECEIVER (brain target)']}/{len(passes)}")
    p()
    p("  If the first line dominates, the neural brain's chosen receiver")
    p("  (`_find_target`) is not what determines the pass on most passes --")
    p("  which is exactly why forcing it 70 times changed nothing.")

    with open("_diag_receiver_owner.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_receiver_owner.txt")


if __name__ == "__main__":
    main()