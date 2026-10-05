"""Corner outcomes over many seeds, aggregated — and a real A/B on the delivery.

The first A/B answered with 3 corners in one arm and 4 in the other, which is
not evidence of anything. Corner counts per match are small (4-12), so a single
match can never carry this question; the honest unit is corners across matches.

Runs all seeds in ONE process, which is now legitimate: the project's standing
"in-process A/B inherits the brain caches" rule was RETRACTED — `_diag_seed_repro.py`'s
`arm none` is byte-identical to a single call. Separate processes were habit,
not necessity.

`--arm off` zeroes the lead, the execution error and the arrival residual,
reproducing the pre-work behaviour: the cross aimed at the receiver's own feet
and a man who stops exactly on his mark.

Usage: python _diag_corner_multi.py on|off [seeds...]
"""
import io
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402
import position_engine  # noqa: E402
from event_chain import EventType  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else "on"
seeds = [int(s) for s in sys.argv[2:]] or [31, 37, 43, 49, 61, 67, 73, 79]
if arm == "off":
    event_chain.CORNER_LEAD_M = 0.0
    event_chain.CORNER_ERROR_M = 0.0
    event_chain.CORNER_ERROR_MAX_M = 0.0
    position_engine.PositionEngine.SET_PIECE_ARRIVAL_ERROR_M = 0.0

agg = Counter()
per_seed = []
_in_chain = [False]

_gen = event_chain.SetPieceChain.generate.__func__


def generate(cls, *a, **kw):
    res = _gen(cls, *a, **kw)
    if _in_chain[0]:
        return res
    _in_chain[0] = True
    try:
        evs = list(getattr(res, "events", []) or [])
        types = Counter(getattr(getattr(e, "event_type", None), "name", "?")
                        for e in evs)
        if not types.get("CORNER_TAKEN"):
            return res
        clear = [e for e in evs if e.event_type == EventType.CLEARANCE]
        headed = [c for c in clear if (c.metadata or {}).get("headed")]
        shots = (types.get("SHOT_ON_TARGET", 0) + types.get("SHOT_OFF_TARGET", 0)
                 + types.get("SHOT_BLOCKED", 0))
        agg["corners"] += 1
        agg["clearance"] += len(clear)
        agg["headed"] += len(headed)
        agg["foot"] += len(clear) - len(headed)
        agg["shots"] += shots
        agg["goals"] += types.get("GOAL", 0)
        agg["corner_won"] += types.get("CORNER_WON", 0)
        if not clear:
            agg["no_clearance"] += 1
        if not shots:
            agg["no_shot"] += 1
        per_seed.append((len(clear), len(headed), shots))
    finally:
        _in_chain[0] = False
    return res


event_chain.SetPieceChain.generate = classmethod(generate)

real = sys.stdout
rows = []
for s in seeds:
    eng, _, _ = build()
    random.seed(s)
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real
    rows.append((s, res.score_str, len(res.timeline)))

c = agg["corners"]
L = [f"ARM {arm.upper()} | {len(seeds)} seeds | "
     + "  ".join(f"{s}:{sc}" for s, sc, _ in rows), ""]
L.append(f"corners measured          {c}")
L.append(f"  ended in any clearance   {agg['clearance']:>4}"
         f"   ({100 * agg['clearance'] / max(1, c):.0f}%)")
L.append(f"    HEADED clearances       {agg['headed']:>4}"
         f"   ({100 * agg['headed'] / max(1, c):.0f}% of corners)")
L.append(f"    foot  clearances        {agg['foot']:>4}"
         f"   ({100 * agg['foot'] / max(1, c):.0f}%)")
L.append(f"  produced NO clearance    {agg['no_clearance']:>4}"
         f"   ({100 * agg['no_clearance'] / max(1, c):.0f}%)")
L.append(f"  produced a shot          {agg['shots']:>4}"
         f"   ({agg['shots'] / max(1, c):.2f} per corner)")
L.append(f"  produced NO shot         {agg['no_shot']:>4}"
         f"   ({100 * agg['no_shot'] / max(1, c):.0f}%)")
L.append(f"  produced a goal          {agg['goals']:>4}")
L.append(f"  corner WON (from a def)  {agg['corner_won']:>4}")
text = "\n".join(L)
open(f"_diag_corner_multi_{arm}.txt", "w", encoding="utf-8").write(text)
print(text)