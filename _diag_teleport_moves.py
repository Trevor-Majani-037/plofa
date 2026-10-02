"""
HOW MUCH PLAYER MOVEMENT IS A SINGLE-CALL TELEPORT? (2026-10-02)
==============================================================
`PositionEngine.record_touch(name, x, y, minute)` is how the engine MOVES a
player. If the caller hands it a coordinate that is not where the man was, the
engine books the difference as distance covered and persists the invented
position, so every later read inherits it.

One live caller does exactly that. `event_chain.py:3404`, the cross receiver:

    rx, ry = position_engine.get_position(cross_receiver.name)
    rx = cls.clamp_attack_x(rx + random.uniform(1.0, 4.0), 85.0, 100.0, attacks_right)
    ...
    position_engine.record_touch(cross_receiver.name, rx, ry, minute)

Aiming a cross INSIDE the box is correct — you do aim at the box. The
fabrication is that the clamp result is then written back as the man's
position, so a winger standing at x=40 is recorded at x=85. That is up to 45 m
of movement that never happened, banked at full value.

`record_touch` is also legitimately used for goal kicks, restarts and corner
placement, where a big jump is real. So this does not just histogram every
call: it tags each jump with WHO made it and in which context, and separates
the deliberate placements from the clamps.

    .venv\\Scripts\\python.exe _diag_teleport_moves.py [n]
"""

import math
import random
import sys
import threading

from _diag_chance_coords import build_pair

PAIRS = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
         ("Justice", "Triumpher")]

MOVES = []                # (name, dx, dy, dist, context)
_lock = threading.Lock()
_CTX = ["unknown"]

BIG = 12.0                # metres; a man covers 1.2m in a tenth of a second


def _install():
    from position_engine import PositionEngine
    import event_chain as EC
    from match_engine import MatchEngine

    rt = PositionEngine.record_touch

    def wrapper(self, player_name, x, y, minute=0, *a, **kw):
        # `states[name]` is a PlayerSpatialState; the coordinates are
        # current_x / current_y. `.position` is the POSITION LABEL ("ST"), not
        # a point — the same trap as `get_position` returning pitch centre.
        st = self.states.get(getattr(player_name, "name", player_name))
        ox = getattr(st, "current_x", None) if st is not None else None
        oy = getattr(st, "current_y", None) if st is not None else None
        res = rt(self, player_name, x, y, minute, *a, **kw)
        if ox is not None and x is not None:
            d = math.hypot(x - ox, y - oy)
            if d >= BIG:
                with _lock:
                    MOVES.append((getattr(player_name, "name", player_name),
                                  ox, oy, x, y, d, _CTX[-1]))
        return res

    PositionEngine.record_touch = wrapper

    # ── attribute every move to the chain that made it ──
    # Without this the histogram is unattributed: a goal kick, a corner
    # placement and a cross-receiver clamp all look like "a man moved a long
    # way", and only the first two are legitimate. Wrapping each chain's
    # `generate` and pushing a context label is the cheapest honest split.
    def tag(cls, label):
        orig = cls.generate.__func__

        def wrapped(c, *a, **kw):
            _CTX.append(label)
            try:
                return orig(c, *a, **kw)
            finally:
                _CTX.pop()

        cls.generate = classmethod(wrapped)

    tag(EC.PossessionChain, "possession")
    tag(EC.AttackChain, "attack_chain")
    if hasattr(EC, "SetPieceChain"):
        tag(EC.SetPieceChain, "set_piece")
    if hasattr(EC, "TransitionChain"):
        tag(EC.TransitionChain, "transition")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    _install()
    for i in range(n):
        h, a = PAIRS[i % len(PAIRS)]
        random.seed(3000 + i)
        print(f"  [{i+1}] {h} v {a} ...", flush=True)
        eng = build_pair(h, a)
        res = eng.simulate()
        print(f"      score {res.home_goals}-{res.away_goals}", flush=True)

    total = len(MOVES)
    print("\n" + "=" * 74)
    print(f"=== SINGLE-CALL TELEPORTS >= {BIG} m  ({n} match) ===\n")
    print(f"  {total} record_touch calls moved a man >= {BIG} m in one call")

    if not MOVES:
        print("  none")
        return MOVES

    by_ctx = {}
    for name, ox, oy, nx, ny, d, ctx in MOVES:
        by_ctx.setdefault(ctx, []).append(d)
    print(f"\n  BY CONTEXT (context is whatever the instrumented caller set):")
    for ctx, ds in sorted(by_ctx.items(), key=lambda kv: -len(kv[1])):
        ds.sort()
        print(f"    {ctx:<34} n={len(ds):3d}  "
              f"median {ds[len(ds)//2]:6.1f} m  max {ds[-1]:6.1f} m  "
              f"total {sum(ds):8.0f} m")

    dists = sorted(d for *_, d, _ in MOVES)
    print(f"\n  ALL: median {dists[len(dists)//2]:.1f} m   "
          f"max {dists[-1]:.1f} m   sum {sum(dists):.0f} m "
          f"(~{sum(dists)/105.0:.1f} full pitches of invented movement)")

    print(f"\n  THE {min(15, total)} BIGGEST:")
    for name, ox, oy, nx, ny, d, ctx in sorted(MOVES, key=lambda t: -t[5])[:15]:
        print(f"    {name:<18} ({ox:6.1f},{oy:5.1f}) -> ({nx:6.1f},{ny:5.1f})  "
              f"{d:6.1f} m   ctx={ctx}")
    return MOVES


if __name__ == "__main__":
    main()