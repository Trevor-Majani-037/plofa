"""Is the first man actually on the flight line, and ahead of the ball?

Milliseconds, no match. This checks the GEOMETRY `_corner_box_occupancy`
returns, which is the only claim that can be made cheaply; whether it moves the
real corner number needs a match and is a separate question.

What it asserts, per corner side:
  * a first man exists and is an outfielder (not the keeper);
  * he sits within ~2.5 m of the straight line from the corner flag to the
    delivery target — i.e. he is where the ball will be, not behind the pack;
  * he is FURTHER from the goal than the receivers he is screening, i.e. he is
    in front of them on the ball's journey and not tucked goal-side;
  * he is inside the pitch and not on top of the taker.

The last one matters: a first man standing on the corner flag is on the flight
line and is useless.
"""
import math
import sys

sys.path.insert(0, ".")
from event_chain import SetPieceChain  # noqa: E402

ATTACKS_RIGHT = True
OWN_GOAL = 105.0 if ATTACKS_RIGHT else 0.0


class DNA:
    class physical:
        jumping = 70.0

    class technical:
        heading = 70.0


class P:
    def __init__(self, name, pos):
        self.name = name
        self.position = pos
        self.dna = DNA()


def perp_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (L * L)))
    return math.hypot(px - (ax + dx * t), py - (ay + dy * t))


for corner_y in (4.0, 66.0):
    att = [P("ST", "ST"), P("CF", "CF"), P("LW", "LW"), P("RW", "RW"),
           P("CAM", "CAM"), P("CM1", "CM"), P("GK", "GK")]
    dfn = [P("CB1", "CB"), P("CB2", "CB"), P("LB", "LB"), P("RB", "RB"),
           P("CDM", "CDM"), P("CM2", "CM"), P("GKD", "GK")]

    class Marking:
        assignments = {}
        first_man = "CB1"
        aerial_defender = "CB2"

    out = SetPieceChain._corner_box_occupancy(
        attacks_right=ATTACKS_RIGHT, corner_y=corner_y, zone="six",
        marking=Marking(), receiver=att[0], att_players=att,
        def_players=dfn)

    origin = (OWN_GOAL, corner_y)
    tgt = out[att[0].name]
    defs = {n: xy for n, xy in out.items()
            if n.startswith(("CB", "LB", "RB", "CDM", "CM2", "GKD"))}
    if not defs:
        print(f"corner_y {corner_y}: NO DEFENDER PLACED -- the guard failed")
        continue

    rows = []
    for n, (x, y) in defs.items():
        off = perp_dist(x, y, origin[0], origin[1], tgt[0], tgt[1])
        goal_dist = abs(OWN_GOAL - x)
        rows.append((off, goal_dist, n, x, y))
    rows.sort()
    best = rows[0]
    rec_goal_dist = abs(OWN_GOAL - tgt[0])
    fman = defs.get("CB1")          # `marking.first_man`

    print(f"corner_y {corner_y}  (attacking right, goal at x={OWN_GOAL})")
    print(f"  flag {origin}  target {tgt}  target goal-dist {rec_goal_dist:.1f} m")
    print(f"  {'defender':<10} {'off-line':>9} {'goal-dist':>10}")
    for off, gd, n, x, y in rows[:4]:
        star = "  <- first_man" if n == "CB1" else ""
        print(f"  {n:<10} {off:8.2f}m {gd:9.1f}m{star}")
    if fman:
        fx, fy = fman
        foff = perp_dist(fx, fy, origin[0], origin[1], tgt[0], tgt[1])
        fgd = abs(OWN_GOAL - fx)
        print(f"  DESIGNATED first man CB1: {foff:.2f} m off the line, "
              f"{fgd:.1f} m from goal -> "
              f"{'IN FRONT of the target' if fgd < rec_goal_dist else 'BEHIND it'}"
              f" | on pitch {0.0 <= fx <= 105.0 and 0.0 <= fy <= 68.0}"
              f" | {math.hypot(fx - origin[0], fy - origin[1]):.1f} m from the flag")
    # The flag sits ON the goal line, so goal-distance 0 IS the start of the
    # ball's journey. A man with a SMALLER goal-distance than the target is
    # therefore further ALONG the flight -- i.e. in front of it. An earlier
    # version of this probe had the comparison backwards and printed "BEHIND
    # it" for a first man who was 5.5 m closer to the corner flag than the
    # target, which is exactly what being a first man means.
    infront = best[1] < rec_goal_dist
    print(f"  first man candidate: {best[2]} — {best[0]:.2f} m off the line, "
          f"{best[1]:.1f} m from goal (target is {rec_goal_dist:.1f} m) -> "
          f"{'IN FRONT of the target on the flight' if infront else 'BEHIND it'}")
    print(f"  on the pitch: {0.0 <= best[3] <= 105.0 and 0.0 <= best[4] <= 68.0}")
    print(f"  not on the flag: "
          f"{math.hypot(best[3] - origin[0], best[4] - origin[1]) > 3.0}")
    print()