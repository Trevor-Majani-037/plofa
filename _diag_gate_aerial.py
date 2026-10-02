"""Gate for the aerial-duel fix. Two properties, both milliseconds.

P1  ARGUMENT ORDER MUST NOT DECIDE IT.
    The same duel, the two men passed in the opposite order, must give the
    same winner. Before the fix this returned ATTACKER / DEFENDER.

P2  SAMPLING RESOLUTION MUST NOT DECIDE IT.
    The same duel at sample_step 0.1 and 0.01 must give the same winner.
    Before the fix this returned ATTACKER / DEFENDER.

Neither is a claim about who SHOULD win a corner -- that is football. They
say only that the answer is a function of the players, not of the arithmetic.
"""
import sys

sys.path.insert(0, ".")
from geometry_engine import (BallFlight, MovingPlayer, Vec2, Vec3,
                             resolve_aerial_delivery)

C = Vec3(105.0, 4.0, 0.5)
T = (94.5, 40.0)


def mp(name, x, y, pace=78.0):
    return MovingPlayer(player=type("P", (), {"name": name})(),
                        position=Vec2(x, y), pace=pace, acceleration=6.0,
                        reaction_time=0.18, control_radius=1.05,
                        jump_height=0.62, standing_reach=1.80)


def duel(a_xy, d_xy, step, swap=False, target=T):
    f = BallFlight(start=C, target=Vec3(target[0], target[1], 0.0),
                   duration=1.3, apex_z=2.1)
    a, d = mp("ATTACKER", *a_xy), mp("DEFENDER", *d_xy)
    r = resolve_aerial_delivery(f, [d], [a], sample_step=step) if swap \
        else resolve_aerial_delivery(f, [a], [d], sample_step=step)
    return getattr(getattr(getattr(r, "winner", None), "player", None),
                   "name", "") or "nobody"


CASES = [
    ("identical spot (the pure tie)", (T[0], T[1]), (T[0], T[1])),
    ("goal-side marking, as mark() leaves it", (T[0] - 1.4, T[1]),
     (T[0] + 1.4, T[1])),
    ("defender out on the flight line", (T[0], T[1]), (99.0, 22.0)),
    ("attacker 4 m off his slot", (T[0] - 4.0, T[1]), (T[0] + 1.4, T[1])),
    ("defender 4 m off his marker", (T[0] - 1.4, T[1]), (T[0] - 1.4, T[1] + 4)),
]

print("P1  ARGUMENT ORDER")
bad1 = 0
for label, a, d in CASES:
    n, s = duel(a, d, 0.1), duel(a, d, 0.1, swap=True)
    ok = "ok " if n == s else "FAIL"
    bad1 += n != s
    print(f"    {ok} {label:<40} normal {n:>8} | swapped {s:>8}")
print()

print("P2  SAMPLING RESOLUTION (0.1 vs 0.01)")
bad2 = 0
for label, a, d in CASES:
    coarse, fine = duel(a, d, 0.1), duel(a, d, 0.01)
    ok = "ok " if coarse == fine else "FAIL"
    bad2 += coarse != fine
    print(f"    {ok} {label:<40} 0.100 {coarse:>8} | 0.010 {fine:>8}")
print()

print(f"argument-order failures: {bad1}/{len(CASES)}")
print(f"resolution failures   : {bad2}/{len(CASES)}")
print()
print("Sanity: a defender clearly on top should still win -- the fix must not")
print("have turned every duel into a coin toss.")
for gap in (0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0):
    w = duel((T[0] + gap, T[1]), (T[0] - gap, T[1]), 0.1)
    print(f"    defender {gap:4.1f} m nearer the ball -> {w:>8}")