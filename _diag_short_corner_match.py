"""Do short corners actually fire, and does the branch survive a real match?

One match. Counts:
  * how many corners were taken with a SHORT routine (the new mechanic), as a
    share of all corners -- `SHORT_CORNER` is in all eight style pools, so it
    should be a meaningful fraction, not a rarity;
  * that a short corner skipped the packed box and took a low ball;
  * what those corners produced, so the routine can be judged on football
    rather than on existing.

Real football: teams work a short corner on roughly 10-20% of corners, more
when chasing a game late. If the engine produces ~0% the branch is dead; if it
produces 50%+ the pool weighting is wrong.

Usage: python _diag_short_corner_match.py <seed>
"""
import io
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402
import set_piece_routines as spr  # noqa: E402
from event_chain import EventType  # noqa: E402

stats = Counter()
short_recv_dist = []

_cd = spr.corner_delivery
_hits = [False]


def corner_delivery(routine):
    p = _cd(routine)
    if _hits[0]:
        stats["corners"] += 1
        if p.get("short"):
            stats["short_corners"] += 1
    return p


spr.corner_delivery = corner_delivery
event_chain.corner_delivery = corner_delivery

# the chain reads the module-level name, so rebind it too
_box = event_chain.SetPieceChain._corner_box_occupancy.__func__
_box_calls = [0]


def box_probe(cls, **kw):
    if _hits[0]:
        _box_calls[0] += 1
        if kw.get("_short_probe"):
            pass
    return _box(cls, **kw)


_gen = event_chain.SetPieceChain.generate.__func__


def generate(cls, *a, **kw):
    _hits[0] = True
    before = _box_calls[0]
    res = _gen(cls, *a, **kw)
    try:
        evs = list(getattr(res, "events", []) or [])
        types = Counter(getattr(getattr(e, "event_type", None), "name", "?")
                        for e in evs)
        if types.get("CORNER_TAKEN"):
            for e in evs:
                if e.event_type == EventType.CORNER_TAKEN:
                    stats["routine_" + str((e.metadata or {}).get("routine"))] += 1
                    break
            for t in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL"):
                stats["from_corner_" + t] += types.get(t, 0)
            stats["from_corner_CLEARANCE"] += types.get("CLEARANCE", 0)
    finally:
        _hits[0] = False
    return res


event_chain.SetPieceChain.generate = classmethod(generate)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
eng, _, _ = build()
random.seed(seed)
real = sys.stdout
sys.stdout = io.StringIO()
try:
    res = eng.simulate()
finally:
    sys.stdout = real

c = stats["corners"]
print(f"seed {seed} | {res.score_str} | events {len(res.timeline)}")
print(f"corners measured      {c}")
print(f"SHORT corners         {stats['short_corners']}"
      f"  ({100 * stats['short_corners'] / max(1, c):.0f}% of corners)")
print()
print("routine actually stamped on CORNER_TAKEN:")
for k in sorted(stats):
    if k.startswith("routine_"):
        print(f"   {k[8:]:<18} {stats[k]}")
print()
tot_shots = sum(stats[f"from_corner_{t}"] for t in
                ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL"))
print(f"from corners: shots {tot_shots} | goals {stats['from_corner_GOAL']} "
      f"| clearances {stats['from_corner_CLEARANCE']}")