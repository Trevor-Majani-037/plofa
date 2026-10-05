"""IS THE WALL A FLAT LINE? — direct check on `_build_freekick_wall`.

(2026-10-01) `_diag_fk_wall.py` reported a lateral spread of 8.01 m for a
4-man wall spaced 0.55 m apart. A 4-man wall at 0.55 m spacing presents
3 * 0.55 = 1.65 m of face. 8.01 m is arithmetically impossible from that
placement, so one of the two numbers is wrong: either the wall really is
scattered, or the probe is measuring along the wrong axis.

Paying 90 s a match to answer that is wasteful, and a match also conflates
the wall with everything else. This builds one wall directly, with every
defender standing at a KNOWN position, and checks the geometry exactly.

Run:  .venv\\Scripts\\python.exe _diag_wall_shape.py
"""
import math

from event_chain import SetPieceChain
from position_engine import PositionEngine

# A clean picture: kicker at 84, attacking right, so the goal is at x=105 and
# the near post (y=30.34) is the one the wall must screen.
FK_X, FK_Y = 84.0, 30.0


class _FakeDNA:
    class physical:
        pace = 70.0


class _FakePlayer:
    def __init__(self, name, x, y, pos="CB"):
        self.name = name
        self.position = pos
        self.dna = _FakeDNA()


def main():
    pe = PositionEngine()
    # Four defenders already close to the ball, so the bounded approach is
    # not the thing under test here.
    spots = [(78.0, 26.0), (80.0, 33.0), (77.0, 38.0), (81.0, 30.0)]
    defs = []
    for i, (x, y) in enumerate(spots, 1):
        p = _FakePlayer(f"DEF{i}", x, y)
        pe.record_touch(p.name, x, y, 30)
        defs.append(p)

    gk = _FakePlayer("GKG", 96.0, 34.0, pos="GK")
    pe.record_touch(gk.name, 96.0, 34.0, 30)

    mps, gk_mp, reach = SetPieceChain._build_freekick_wall(
        FK_X, FK_Y, defs, True, pe, 30, gk)

    print(f"men in wall: {len(mps)}   (expected 4)")
    print(f"reach log : {['%.2f' % r for r in reach]}")

    goal_x = 105.0
    near_y = 30.34 if FK_Y < 34 else 37.66
    dx, dy = goal_x - FK_X, near_y - FK_Y
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux

    print(f"\nball-to-near-post unit vector  u=({ux:.4f}, {uy:.4f})")
    print(f"wall-face unit vector         p=({px:.4f}, {py:.4f})")
    print(f"ball->near-post length        {L:.2f} m\n")

    print(f"{'man':<6}{'x':>8}{'y':>8}{'dist':>8}{'along':>8}{'lateral':>9}")
    dists, laterals = [], []
    for m in mps:
        vx, vy = m.position.x - FK_X, m.position.y - FK_Y
        d = math.hypot(vx, vy)
        along = vx * ux + vy * uy
        lat = vx * px + vy * py
        dists.append(d)
        laterals.append(lat)
        nm = getattr(m.player, "name", "?")
        print(f"{nm:<6}{m.position.x:>8.2f}{m.position.y:>8.2f}"
              f"{d:>8.2f}{along:>8.2f}{lat:>9.2f}")

    spread = max(laterals) - min(laterals)
    print(f"\ndistance from ball : mean {sum(dists)/len(dists):.2f} m"
          f"  min {min(dists):.2f}  max {max(dists):.2f}"
          f"   [want ~9.15]")
    print(f"LATERAL SPREAD     : {spread:.2f} m"
          f"   [4 men x 0.55 m spacing = 1.65 m]")
    if spread > 3.0:
        print("\n  -> NOT A FLAT WALL. The men are not standing shoulder to")
        print("     shoulder, so the blocker set handed to resolve_shot is a")
        print("     handful of scattered bodies, not a screen.")
    else:
        print("\n  -> flat wall: the geometry helper is correct.")


if __name__ == "__main__":
    main()
