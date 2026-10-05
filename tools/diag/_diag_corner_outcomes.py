"""Corner outcomes, and a real A/B on the delivery change.

THREE PROBE ERRORS FIXED HERE, ALL THE SAME SHAPE (reading a key that is not
the one the engine writes):
  1. `ball_aerial` -- not a `MatchEvent` field, and the corner branch passes it
     as a kwarg that does not survive as metadata either.
  2. `clearance_kind` -- a real helper exists (`_clearance_kind`) but the
     set-piece chain never calls it, so nothing carries that name.
  3. `headed` -- THIS is the field, in metadata. It was there the whole time.

THE A/B
`--arm off` sets the lead and the execution error to zero AND the arrival
residual to zero, which reproduces the pre-work behaviour of "the cross is
aimed at the receiver's own feet and he stops exactly on his mark". `--arm on`
is what ships.

This is only trustworthy now: matches replay byte-identically across processes
(see REPRODUCIBILITY in AGENTS.md), so the two arms differ ONLY by these
constants. Before that fix, per-arm swings included hash-order noise.

Usage: python _diag_corner_outcomes.py <seed> [on|off]
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

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
arm = sys.argv[2] if len(sys.argv) > 2 else "on"
if arm == "off":
    event_chain.CORNER_LEAD_M = 0.0
    event_chain.CORNER_ERROR_M = 0.0
    event_chain.CORNER_ERROR_MAX_M = 0.0
    position_engine.PositionEngine.SET_PIECE_ARRIVAL_ERROR_M = 0.0

out = []
stats = Counter()
_meta_keys = Counter()

_gen = event_chain.SetPieceChain.generate.__func__


def generate(cls, *a, **kw):
    res = _gen(cls, *a, **kw)
    evs = list(getattr(res, "events", []) or [])
    kinds = Counter()
    for e in evs:
        if e.event_type != EventType.CLEARANCE:
            continue
        md = e.metadata or {}
        _meta_keys.update(md.keys())
        # the field the engine actually writes
        kinds["headed" if md.get("headed") else "foot"] += 1
    types = Counter(getattr(getattr(e, "event_type", None), "name", "?")
                    for e in evs)
    corner = types.get("CORNER_TAKEN", 0) > 0
    if corner:
        stats["corners"] += 1
        stats["clear_from_corner"] += kinds.total()
        for k, v in kinds.items():
            stats[f"corner_{k}"] += v
        stats["shot_from_corner"] += types.get("SHOT_ON_TARGET", 0) + \
            types.get("SHOT_OFF_TARGET", 0) + types.get("SHOT_BLOCKED", 0)
        stats["goal_from_corner"] += types.get("GOAL", 0)
    out.append((corner, dict(kinds), dict(types)))
    return res


event_chain.SetPieceChain.generate = classmethod(generate)

eng, _, _ = build()
random.seed(seed)
real = sys.stdout
sys.stdout = io.StringIO()
try:
    res = eng.simulate()
finally:
    sys.stdout = real

L = [f"ARM {arm.upper()} | seed {seed} | {res.score_str} | "
     f"events {len(res.timeline)}", ""]
cc = stats["corners"]
L.append(f"corners (chains that took one) {cc}")
L.append(f"  ended in a clearance        {stats['clear_from_corner']}"
         f"  ({100 * stats['clear_from_corner'] / max(1, cc):.0f}% of corners)")
L.append(f"    HEADED clearances          {stats['corner_headed']}")
L.append(f"    foot  clearances           {stats['corner_foot']}")
L.append(f"  produced a shot             {stats['shot_from_corner']}")
L.append(f"  produced a goal             {stats['goal_from_corner']}")
L.append("")
L.append("metadata keys on CLEARANCE events:")
L.append("  " + ", ".join(sorted(_meta_keys)))
L.append("")
for i, (corner, kinds, types) in enumerate(out):
    if not corner:
        continue
    L.append(f"  corner {i}: {kinds or '-'}  "
             f"types={sorted(t for t in types if types[t])}")
text = "\n".join(L)
open(f"_diag_corner_outcomes_{arm}.txt", "w", encoding="utf-8").write(text)
print(text)