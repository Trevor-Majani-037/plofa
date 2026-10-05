"""FOOTBALL-WISE before/after for the centre-spot location fix (2026-10-04).

What actually changed in the football, not just in the data.

The fix altered two coordinates on the live path, and only two (the penalty,
injury, substitution, VAR and celebration changes are post-play or non-on-ball,
so they move no position):

  * `TransitionChain`'s PRESS location — was `random.uniform(55, 85)` /
    `random.uniform(10, 58)`, INDEPENDENT of where the ball was.
  * the "player played through" PASS — carried no location, so it inherited the
    (50.0, 34.0) centre-spot default and `_absorb_chain` snapped the man there.

Both feed `PositionEngine.record_touch`, which BANKS the jump into
`minute_touch_distance` (`position_engine.py:907-910`). So both move a real
player and both inflate real distance.

  --arm before   restores the old behaviour at the event boundary, by wrapping
                 `make_event`. That is the only place the two old behaviours
                 were observable, and it keeps the comparison to exactly the
                 change under test rather than reverting the file.
  --arm after    runs the code as it stands.

ONE PROCESS PER ARM, and one match per process-seed pair. A/B across two
`simulate()` calls in a single process is invalid here: module-level brain and
mind caches survive `simulate()`, so the second match inherits the first's
state. This is the project's standing reproducibility blocker.

WHAT THE NUMBERS CAN AND CANNOT SUPPORT. A match is not reproducible from
`random.seed` (AGENTS.md, seed-reproducibility item), and the "before" arm
draws from `random` at a slightly different point in the stream than the
original code did. So differences in GOALS, SHOTS or POSSESSION between the
arms are NOISE at this sample size and are printed only so nobody mistakes
their absence for a claim. The metrics that ARE exact are the structural ones:
where presses are recorded, and how much distance is banked that no player
covered. Those are counts over a whole match, not a coin flip.
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

# ── distance banked from large snaps ──────────────────────────────
SNAP_METRES = [0.0]
SNAP_N = [0]
_orig_touch = PositionEngine.record_touch


def traced_touch(self, name, x, y, minute=0, **kw):
    st = self.states.get(name)
    if st is not None:
        jump = ((x - st.current_x) ** 2 + (y - st.current_y) ** 2) ** 0.5
        if jump >= BIG:
            SNAP_N[0] += 1
            SNAP_METRES[0] += jump
    return _orig_touch(self, name, x, y, minute, **kw)


PositionEngine.record_touch = traced_touch

# ── the "before" arm ──────────────────────────────────────────────
# `make_event` is a staticmethod: wrap it AS one, or the descriptor protocol
# hands the wrapper `minute` as `self`.
if A.arm == "before":
    _orig_make = BaseChain.__dict__["make_event"]

    def make_event_before(*a, **kw):
        ev = _orig_make(*a, **kw)
        fr = sys._getframe(1)
        md = ev.metadata or {}
        cls = fr.f_locals.get("cls")
        owner = getattr(cls, "__name__", "")
        # KEY ON THE CHAIN, NOT THE EVENT TYPE -- see _diag_press_band.py.
        # Firing on every non-counterpress PRESS also rewrote
        # PossessionChain's press, which is `ball +/- 3 m` and was never
        # fabricated, so the pooled numbers this probe first printed were
        # partly an artefact of the probe itself.
        if (ev.event_type.name == "PRESS"
                and owner == "TransitionChain"
                and not md.get("counterpress")):
            # the old draw, taken from the football stream as it was
            ev.location_x = random.uniform(55, 85)
            ev.location_y = random.uniform(10, 58)
        elif (ev.event_type.name == "PASS"
                and md.get("press_resistance")):
            # the old code passed no location at all -> MatchEvent's default
            ev.location_x, ev.location_y = 50.0, 34.0
        return ev

    BaseChain.make_event = staticmethod(make_event_before)

from _diag_chance_coords import build_pair  # noqa: E402

SHOT_TYPES = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED"}
rows = []
for m in range(A.matches):
    random.seed(9100 + m * 100)
    SNAP_N[0] = 0
    SNAP_METRES[0] = 0.0
    eng = build_pair("Oxton", "Natrican")
    res = eng.simulate()

    tl = res.timeline
    presses = [e for e in tl if e.event_type.name == "PRESS"]
    px = sorted(e.location_x for e in presses)
    deep = sum(1 for v in px if v < 55.0)          # own-half / counterpress
    corner = sum(1 for e in presses
                 if e.location_y < 12.0 or e.location_y > 56.0)
    rows.append({
        "events": len(tl),
        "goals": len([e for e in tl if e.event_type.name == "GOAL"]),
        "shots": len([e for e in tl if e.event_type.name in SHOT_TYPES]),
        "on_target": len([e for e in tl
                          if e.event_type.name == "SHOT_ON_TARGET"]),
        "poss": res.home_possession_pct,   # a @property, not a method
        "presses": len(presses),
        "press_x_min": round(px[0], 1) if px else None,
        "press_x_p25": round(px[len(px) // 4], 1) if px else None,
        "press_x_med": round(statistics.median(px), 1) if px else None,
        "press_x_p75": round(px[3 * len(px) // 4], 1) if px else None,
        "press_x_max": round(px[-1], 1) if px else None,
        "presses_deep": deep,
        "presses_corner_channel": corner,
        "snap_n": SNAP_N[0],
        "snap_m": round(SNAP_METRES[0]),
    })
    print(f"  match {m + 1}: {rows[-1]}", flush=True)


def tot(k):
    return sum(r[k] for r in rows)


def med(k):
    vals = [r[k] for r in rows if r[k] is not None]
    return round(statistics.median(vals), 1) if vals else None


summary = {
    "arm": A.arm,
    "matches": A.matches,
    "events_mean": round(tot("events") / len(rows)),
    "goals": tot("goals"),
    "shots": tot("shots"),
    "on_target": tot("on_target"),
    "poss_mean": round(tot("poss") / len(rows), 1),
    "presses": tot("presses"),
    "press_x_min": min(r["press_x_min"] for r in rows
                       if r["press_x_min"] is not None),
    "press_x_median": med("press_x_med"),
    "press_x_max": max(r["press_x_max"] for r in rows
                       if r["press_x_max"] is not None),
    "presses_deep": tot("presses_deep"),
    "presses_corner_channel": tot("presses_corner_channel"),
    "snap_n": tot("snap_n"),
    "snap_m": tot("snap_m"),
}

out = A.out or f"_diag_football_{A.arm}.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"summary": summary, "matches": rows}, fh, indent=2)

print("\n" + "=" * 66)
print(f"ARM = {A.arm.upper()}   ({A.matches} matches) -> {out}")
print("=" * 66)
for k, v in summary.items():
    if k in ("arm", "matches"):
        continue
    print(f"  {k:<26} {v}")
print()
print("  STRUCTURAL (exact, count-based)      press_x_min / presses_deep /")
print("                                       snap_m")
print("  NOISE-ONLY (not causal at this n)    goals / shots / poss_mean / events_mean")
