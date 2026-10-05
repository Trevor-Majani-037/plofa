"""FOOTBALL-WISE before/after, attributed to the exact emitting call site.

Corrects two defects in the first split attempt (`_diag_football_sites.py`),
both of which would have produced a wrong table rather than an absent one:

  1. The BEFORE arm lost attribution entirely. `make_event_before` called the
     RAW `_orig_make`, bypassing the wrapper that records `SITE[id(ev)]`, and
     then replaced the recording wrapper on the class. Every event in the
     before arm would have been filed under "?".
  2. Snap distance was attributed through a mutable "current site" variable.
     That is unsound: `_absorb_chain`'s snap-to-event runs OUTSIDE
     `make_event`, so the variable still held whatever the last event was and
     would have labelled ~91% of the metres with a site that did not write
     them. A metric that attributes distance to the wrong line is worse than
     no metric, because it looks like an attribution.

Both are fixed by using the identity join `_diag_snap_origin.py` already
proved: `make_event` records `id(event) -> "file:lineno"`, and inside a
wrapped `record_touch` the caller's frame local `event` is read and looked up
by `id()`. That is exact — no inference, no mutable carry — and it is why the
earlier probe could name a single line as the whole of 602 m.

`make_event` is a `@staticmethod`: read it out of `BaseChain.__dict__` and
wrap it AS a staticmethod, or the descriptor protocol hands the wrapper
`minute` as `self`.

The pooled figures from `_diag_football_ab.py` that this supersedes:
  press_x_min -0.5 (before) / -1.0 (after)  -- so the "no press below x=55"
  claim was FALSE as stated, because that probe counted presses from every
  chain while only restoring TransitionChain's location. The claim is only
  ever testable on TransitionChain's own presses, which is what this does.
"""
import argparse
import collections
import json
import random
import statistics
import sys

sys.path.insert(0, ".")

from event_chain import BaseChain  # noqa: E402
from position_engine import PositionEngine  # noqa: E402

BIG = 12.0

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("before", "after"), required=True)
ap.add_argument("--matches", type=int, default=3)
ap.add_argument("--out", default=None)
A = ap.parse_args()

SITE = {}
_RAW = BaseChain.__dict__["make_event"]


def _fname(frame):
    return f"{frame.f_code.co_filename.split(chr(92))[-1]}:{frame.f_lineno}"


# ── record then optionally restore, in ONE wrapper, so the before arm
#    cannot bypass attribution (defect 1) ─────────────────────────
def _make(*a, **kw):
    ev = _RAW(*a, **kw)
    SITE[id(ev)] = _fname(sys._getframe(1))
    if A.arm == "before":
        md = ev.metadata or {}
        # KEY ON THE CHAIN, NOT THE EVENT TYPE. The first version of this
        # restore fired on every non-counterpress PRESS, and so rewrote
        # PossessionChain's press too -- which is `ball +/- 3 m`
        # (`event_chain.py:1560`) and was NEVER fabricated. That silently
        # moved 84 correct presses per match into the band, which both
        # fabricated the "deep presses rose" finding and manufactured the
        # snap-distance drop by teleporting 84 pressers per match. The
        # `cls` local names the chain that emitted the event, which is a
        # stable identifier where a line number is not.
        cls = sys._getframe(1).f_locals.get("cls")
        owner = getattr(cls, "__name__", "")
        if (ev.event_type.name == "PRESS"
                and owner == "TransitionChain"
                and not md.get("counterpress")):
            ev.location_x = random.uniform(55, 85)
            ev.location_y = random.uniform(10, 58)
        elif ev.event_type.name == "PASS" and md.get("press_resistance"):
            ev.location_x, ev.location_y = 50.0, 34.0
    return ev


BaseChain.make_event = staticmethod(_make)

# ── snap distance via the id join, NOT a mutable carry (defect 2) ──
_orig_touch = PositionEngine.record_touch
SNAP = collections.defaultdict(lambda: [0, 0.0])


def traced_touch(self, name, x, y, minute=0, **kw):
    st = self.states.get(name)
    if st is not None:
        jump = ((x - st.current_x) ** 2 + (y - st.current_y) ** 2) ** 0.5
        if jump >= BIG:
            ev = sys._getframe(1).f_locals.get("event")
            site = SITE.get(id(ev), "no-event-in-frame")
            s = SNAP[site]
            s[0] += 1
            s[1] += jump
    return _orig_touch(self, name, x, y, minute, **kw)


PositionEngine.record_touch = traced_touch

from _diag_chance_coords import build_pair  # noqa: E402

SHOT_TYPES = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED"}
rows = []
PX = collections.defaultdict(list)
PCOUNT = collections.Counter()

for m in range(A.matches):
    random.seed(9100 + m * 100)
    SNAP.clear()
    PCOUNT.clear()
    PX.clear()
    res = build_pair("Oxton", "Natrican").simulate()
    for e in res.timeline:
        if e.event_type.name != "PRESS":
            continue
        site = SITE.get(id(e), "not-from-make_event")
        PCOUNT[site] += 1
        PX[site].append(e.location_x)
    rows.append({
        "events": len(res.timeline),
        "goals": len([e for e in res.timeline
                      if e.event_type.name == "GOAL"]),
        "shots": len([e for e in res.timeline
                      if e.event_type.name in SHOT_TYPES]),
        "on_target": len([e for e in res.timeline
                          if e.event_type.name == "SHOT_ON_TARGET"]),
        "poss": res.home_possession_pct,
    })
    print(f"  match {m + 1}: {rows[-1]}", flush=True)

print()
print("=" * 78)
print(f"ARM = {A.arm.upper()}   ({A.matches} matches)")
print("=" * 78)
unattributed = PCOUNT.get("not-from-make_event", 0)
print(f"PRESS events attributed: {sum(PCOUNT.values()) - unattributed}"
      f"   unattributed: {unattributed}")
print()
print(f"  {'emitting site':<24} {'n':>5} {'x_min':>7} {'x_med':>7} "
      f"{'x_max':>7} {'x<55':>6}")
for site, n in PCOUNT.most_common():
    xs = sorted(PX[site])
    deep = sum(1 for v in xs if v < 55.0)
    print(f"  {site:<24} {n:>5} {xs[0]:>7.1f} {statistics.median(xs):>7.1f}"
          f" {xs[-1]:>7.1f} {deep:>6}")

print()
print(f"Snap distance (>= {BIG:.0f} m) by the site that wrote the event")
print(f"  {'site':<24} {'jumps':>7} {'metres':>9} {'per match':>10}")
tot_n = tot_m = 0
for site, (n, mm) in sorted(SNAP.items(), key=lambda kv: -kv[1][1]):
    print(f"  {site:<24} {n:>7} {mm:>9.0f} {mm / A.matches:>10.0f}")
    tot_n += n
    tot_m += mm
print(f"  {'TOTAL':<24} {tot_n:>7} {tot_m:>9.0f} {tot_m / A.matches:>10.0f}")

out = A.out or f"_diag_football_site_{A.arm}.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "arm": A.arm, "matches": rows,
        "press_sites": {s: {"n": n, "x_min": min(PX[s]),
                            "x_med": statistics.median(PX[s]),
                            "x_max": max(PX[s]),
                            "deep": sum(1 for v in PX[s] if v < 55.0)}
                        for s, n in PCOUNT.items()},
        "snap_sites": {s: {"jumps": v[0], "m": round(v[1], 1)}
                       for s, v in SNAP.items()},
    }, fh, indent=2)
print(f"\n  -> {out}")