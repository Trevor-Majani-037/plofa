"""
SHOT-ORIGIN COHERENCE PROBE (2026-10-02)
========================================
QUESTION
--------
The causal-shooter fix makes the shooter the man who actually had the ball.
Does the shot's recorded ORIGIN actually agree with where that man was?

WHY THIS PROBE HAD TO BE REWRITTEN
----------------------------------
The first version of this probe compared each shot's origin against
`position_engine.tracked_position(shooter)` read AFTER `simulate()` returned.
That reports the player's position at the FINAL WHISTLE, not at the shot, so
it produced a "median 39.5 m disagreement" that was nothing but end-of-match
drift. A post-match read of a mutated engine, attributed to a mid-match
moment — the same shape of error as a post-match fact on the engine being
silently lost by a reader of the result.

It also used a 40-EVENT lookback to find "the ball's last position", which
reaches straight across possession changes and set pieces. Also invalid.

So this version:
  * snapshots every attacker's TRACKED POSITION AT CALL TIME, by wrapping
    `AttackChain.generate` — the value is read while the match is live,
  * takes the ball position from `PossessionChain.shoot_x/shoot_y`, which the
    engine itself hands to the shot, not from a search backwards through the
    timeline,
  * reports only shots the engine actually dispatched, and says how many.

    .venv\\Scripts\\python.exe _diag_shot_origin.py [n]
"""

import math
import random
import sys
import threading

from _diag_chance_coords import build_pair

PAIRS = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
         ("Justice", "Triumpher")]

SHOT_TYPES = {
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SHOT_SAVED",
    "SHOT_POST", "SHOT_WOODWORK", "SHOT_MISS",
}

# ── EXACT pairing ───────────────────────────────────────────────
# Two earlier versions of this probe matched a shot to a position by
# `(minute, team)` and then by "the last AttackChain call before this shot's
# clock". Both are wrong: a minute holds many sequences, and set-piece shots
# never pass through AttackChain at all, so the lookup returns an unrelated
# earlier possession. That produced "shooter was 19 m away" and even
# "shooter was at (50.0, 7.4)" — a formation anchor, not a live position.
#
# This version tags the ACTUAL event objects the chain returned, keyed by
# `id()`. The engine appends those same objects to the timeline, so object
# identity is an exact join with no ordering heuristic and no clock.
TAGS = {}               # id(event) -> {"snap": {...}, "ball": (x, y) or None}
PENDING = {}            # team -> ball position from the shoot decision
_lock = threading.Lock()

SHOT_TYPES = {
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SHOT_SAVED",
    "SHOT_POST", "SHOT_WOODWORK", "SHOT_MISS",
}


def _install():
    from event_chain import AttackChain
    original = AttackChain.generate.__func__      # unwrap classmethod

    def wrapper(cls, minute, attacking_team, *a, **kw):
        pe = kw.get("position_engine")
        snap = {}
        if pe is not None:
            # a = (defending_team, att_players, def_players, tp, dp, state, ...)
            for p in (a[1] or []):
                name = getattr(p, "name", "")
                try:
                    at = pe.tracked_position(name)
                except Exception:
                    at = None
                if at is not None:
                    snap[name] = at
        with _lock:
            ball = PENDING.pop(attacking_team, None)

        res = original(cls, minute, attacking_team, *a, **kw)

        # tag the very objects the chain produced
        with _lock:
            for e in getattr(res, "events", []) or []:
                if e.event_type.name in SHOT_TYPES:
                    TAGS[id(e)] = {"snap": snap, "ball": ball,
                                   "via_attack_chain": True}
        return res

    AttackChain.generate = classmethod(wrapper)


def _capture_ball_positions():
    """`PossessionChain` already records the ball position it hands to the
    shot (`shoot_x`/`shoot_y`). Park it so the immediately following
    AttackChain call can claim it."""
    from event_chain import PossessionChain
    original = PossessionChain.generate.__func__

    def wrapper(cls, minute, attacking_team, *a, **kw):
        res = original(cls, minute, attacking_team, *a, **kw)
        if getattr(res, "shoot_decision", False):
            with _lock:
                PENDING[attacking_team] = (res.shoot_x, res.shoot_y)
        return res

    PossessionChain.generate = classmethod(wrapper)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def run_match(i):
    h, a = PAIRS[i % len(PAIRS)]
    random.seed(3000 + i)
    print(f"  [{i+1}] {h} v {a} ...", flush=True)
    eng = build_pair(h, a)
    res = eng.simulate()

    rows = []
    for e in (getattr(res, "timeline", []) or []):
        if e.event_type.name not in SHOT_TYPES:
            continue
        tag = TAGS.get(id(e))
        snap = (tag or {}).get("snap") or {}
        at = snap.get(e.player)
        ball = (tag or {}).get("ball")

        sx, sy = e.location_x, e.location_y
        rows.append({
            "minute": e.minute, "team": e.team, "shooter": e.player,
            "via": bool(tag),
            "shot": (round(sx, 1), round(sy, 1)),
            "at_shot": (round(at[0], 1), round(at[1], 1)) if at else None,
            "ball": (round(ball[0], 1), round(ball[1], 1)) if ball else None,
            "d_man": math.hypot(sx - at[0], sy - at[1]) if at else None,
            "d_ball": math.hypot(sx - ball[0], sy - ball[1]) if ball else None,
        })
    print(f"      score {res.home_goals}-{res.away_goals}, {len(rows)} shots, "
          f"{len([r for r in rows if r['via']])} from an AttackChain call",
          flush=True)
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    _install()
    _capture_ball_positions()
    rows = []
    for i in range(n):
        rows.extend(run_match(i))

    print("\n" + "=" * 74)
    print(f"=== SHOT ORIGIN vs THE SHOOTER'S LIVE POSITION ({len(rows)} shots) ===\n")

    traced = [r for r in rows if r["via"]]
    print(f"  shots from an AttackChain call (origin should be the man) "
          f"{len(traced)}/{len(rows)}")
    if not traced:
        print("  no AttackChain shots — nothing to conclude")
        return rows

    dm = [r["d_man"] for r in traced if r["d_man"] is not None]
    print(f"  |shot origin - his live position at the shot|  (n={len(dm)})")
    print(f"      median {pct(dm,.5):6.2f} m   p90 {pct(dm,.9):6.2f} m   "
          f"max {max(dm):6.2f} m")
    for tol in (1.0, 3.0, 5.0, 10.0):
        k = sum(1 for d in dm if d <= tol)
        print(f"      within {tol:4.1f} m : {k:3d}/{len(dm)}  ({100*k/len(dm):.0f}%)")

    db = [r["d_ball"] for r in traced if r["d_ball"] is not None]
    if db:
        print(f"  |shot origin - the ball position handed to the shot|  (n={len(db)})")
        print(f"      median {pct(db,.5):6.2f} m   p90 {pct(db,.9):6.2f} m   "
              f"max {max(db):6.2f} m")

    print("\n  THE SHOTS WHERE THE ORIGIN MATCHES NEITHER (both >5 m):")
    bad = [r for r in traced
           if (r["d_man"] if r["d_man"] is not None else 1e9) > 5.0
           and (r["d_ball"] if r["d_ball"] is not None else 1e9) > 5.0]
    if not bad:
        print("    none — every origin agrees with the man or the ball")
    for r in bad[:20]:
        print(f"    {r['minute']:>3}' {r['team']:<13} {r['shooter']:<17} "
              f"shot {str(r['shot']):<14} man {str(r['at_shot']):<14} "
              f"ball {str(r['ball']):<14} "
              f"d_man {(r['d_man'] or -1):6.1f}  d_ball {(r['d_ball'] or -1):6.1f}")
    return rows


if __name__ == "__main__":
    main()