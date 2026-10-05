"""Is the 93-100% attacker corner win rate a consequence of goal-side marking?

DECISIVE AND CHEAP. This calls `resolve_aerial_delivery` directly with
hand-placed players. No match, no engine, no suites -- it runs in milliseconds
and the CPU never notices.

THE HYPOTHESIS UNDER TEST
`resolve_aerial_delivery` picks the contestant who can reach the ball at the
EARLIEST point on its path (`if candidates: break` on the first sample anyone
can reach, then `sort(key=(time_s, -reach))[0]`). Meanwhile `_corner_box_occupancy`
marks each defender 1.4 m GOAL-SIDE of his man:

    out[dname] = at(max(1.5, depth - 1.4), ay + jitter)

Goal-side is "further along the ball's journey", because the ball travels from
the corner flag inward toward the goal. So the attacker stands essentially ON
the ball's destination and his marker stands 1.4 m short of it -- a deficit the
arrival race converts into a loss, every time.

If that is the mechanism, then:
  A. goal-side marking          -> attacker wins
  B. zero offset (same spot)    -> attacker STILL wins  (they are identical)
  C. defender ON the flight line-> defender wins      (the real fix)
  D. defender goal-side AND the cross is delivered SHORT of the attacker
     -> defender wins           (this is what "remove aim" would buy)
D is the one that matters most: it says the bias is not about marking quality
at all, it is about the ball being aimed at a specific man.
"""
import sys

sys.path.insert(0, ".")
from geometry_engine import (BallFlight, MovingPlayer, Vec2, Vec3,  # noqa: E402
                             resolve_aerial_delivery)

GOAL_X = 105.0          # attacking right
CORNER = Vec3(GOAL_X, 4.0, 0.5)      # the corner flag
SLOT = (94.5, 40.0)     # receiver slot: depth 10.5 from goal, y=40 (near band)


def mp(name, x, y, pace=78.0):
    return MovingPlayer(player=type("P", (), {"name": name})(),
                        position=Vec2(x, y), pace=pace, acceleration=6.0,
                        reaction_time=0.18, control_radius=1.05,
                        jump_height=0.62, standing_reach=1.80)


def duel(target, att_xy, def_xy, apex=2.1, duration=1.3):
    f = BallFlight(start=CORNER, target=Vec3(target[0], target[1], 0.0),
                   duration=duration, apex_z=apex)
    a = [mp("ATTACKER", *att_xy)]
    d = [mp("DEFENDER", *def_xy)]
    r = resolve_aerial_delivery(f, a, d)
    w = getattr(getattr(r, "winner", None), "player", None)
    return (getattr(w, "name", "") or "nobody"), getattr(r, "outcome", "")


# depth measured from the goal line; goal-side means a SMALLER depth
def goal_side(dx, y):
    return (GOAL_X - (dx - 1.4), y)


ATT = SLOT                                   # the attacker is ON the ball
DEF_GOALSIDE = goal_side(10.5, 40.0)         # exactly what mark() produces

print("CROSS AIMED AT THE RECEIVER'S SLOT  (today's behaviour)")
print(f"  attacker at slot {ATT}, defender goal-side {DEF_GOALSIDE}")
print(f"    -> {duel(SLOT, ATT, DEF_GOALSIDE)}")
print(f"  A/B defender at the attacker's OWN spot {ATT}")
print(f"    -> {duel(SLOT, ATT, ATT)}")
print(f"  C  defender ON the flight line, out in front (99.0, 22.0)")
print(f"    -> {duel(SLOT, ATT, (99.0, 22.0))}")
print()
print("CROSS DELIVERED SHORT OF THE RECEIVER  (what removing the aim buys)")
SHORT = (96.8, 36.5)
print(f"  ball lands {SHORT}, attacker still standing on {ATT}")
print(f"    -> {duel(SHORT, ATT, DEF_GOALSIDE)}")
print(f"  and if the attacker mis-runs 4 m goal-side to {(GOAL_X - 14.5, 40.0)}")
print(f"    -> {duel(SLOT, (GOAL_X - 14.5, 40.0), DEF_GOALSIDE)}")
print()
print("HOW BIG DOES THE EXECUTION ERROR HAVE TO BE?")
print("  sweeping the ball's endpoint across the box, everything else fixed:")
print()
print("   target            goal-side defender      attacker wins?")
for i in range(9):
    tx = 92.0 + i * 1.6
    ty = 36.0 + (i % 3) * 1.5
    w, _ = duel((tx, ty), ATT, DEF_GOALSIDE)
    print(f"   ({tx:5.1f},{ty:5.1f})        ({DEF_GOALSIDE[0]:5.1f},"
          f"{DEF_GOALSIDE[1]:4.1f})            {'YES' if w == 'ATTACKER' else 'no '} ({w})")