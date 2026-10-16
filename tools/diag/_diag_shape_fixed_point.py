"""THE FIXED POINT OF THE LIVE SHAPE CHAIN (2026-10-14) -- arithmetic, no match.

`MatchEngine._offball_move_player` line 2757:

    ax, ay = self._minute_start_snapshot.get(pname, (st.current_x, st.current_y))

and line 3806-3809 refreshes that dict every minute from
`PositionEngine.snapshot_positions(team)` -- the player's TRACKED position, NOT
`state.home_x`. Then line 2809:

    tx = ax + (ball_x - ax) * 0.14

So the shape target is 86% "where you already were" and 14% "toward the ball".
There is no formation post in the x term at all. This script computes where that
recursion settles, for a home-anchored chain and for the persistence chain, so
the measurement in `_diag_wide_depth_live.py` section 7 has a prediction to be
checked against rather than a story told after the fact.

x_{n+1} = (1-w)*ax + w*ball  with ax = x_n  (persistence)
      => x* = ball                            -- for ANY starting position
      => with ax = home (formation-anchored)
      => x* = (1-w)*home + w*ball
Then the run-target blend on top: x <- x + (run_x - x)*b.

Run:  python _diag_shape_fixed_point.py
"""
W_COMPACT = 0.14
BLEND_WIDE = 0.30


def fixed_point(home, ball, run_x=None, blend=0.0, w=W_COMPACT, iters=400):
    x = home
    for _ in range(iters):
        x = (1.0 - w) * x + w * ball          # persistence chain, ax = x_n
        if run_x is not None and blend > 0.0:
            x = x + (run_x - x) * blend
    return x


print("=" * 92)
print("WHERE THE LIVE CHAIN SETTLES -- ball at 70 m, 400 iterations")
print("=" * 92)
print(f"{'role':<8}{'home':>7}{'start':>7}{'settles':>10}"
      f"{'+run(84)':>11}{'with overlap':>15}")
for role, home in (("GK", 6.0), ("CB", 22.0), ("LB", 22.0), ("CDM", 40.0),
                   ("CM", 52.0), ("CAM", 66.0), ("LW", 82.0), ("ST", 88.0)):
    s = fixed_point(home, 70.0)
    sr = fixed_point(home, 70.0, run_x=84.0, blend=BLEND_WIDE)
    print(f"{role:<8}{home:>7.0f}{home:>7.0f}{s:>10.1f}{sr:>11.1f}"
          f"{'(blend .30)':>15}")
print()
print("Read the 'settles' column: EVERY outfield role converges to the ball's")
print("x, from any starting position. The chain has no fixed point other than")
print("the ball. That is the whole mechanism -- a persistence term with no")
print("restoring force toward the formation.")
print()

print("=" * 92)
print("SAME RECURSION WITH ax = state.home_x (what the dead drift_minute does)")
print("=" * 92)
print(f"{'role':<8}{'home':>7}{'ball 45':>9}{'ball 70':>9}{'ball 92':>9}")
for role, home in (("CB", 22.0), ("LB", 22.0), ("CDM", 40.0), ("CM", 52.0),
                   ("CAM", 66.0), ("LW", 82.0), ("ST", 88.0)):
    out = []
    for ball in (45.0, 70.0, 92.0):
        x = home
        for _ in range(400):
            x = (1.0 - W_COMPACT) * home + W_COMPACT * ball
        out.append(x)
    print(f"{role:<8}{home:>7.0f}{out[0]:>9.1f}{out[1]:>9.1f}{out[2]:>9.1f}")
print()
print("Home-anchored, a CB settles at 25-31 and a CB at 60 -- a SHAPE that")
print("responds to the ball. Compare the persistence table: 70 and 70.")
print()

print("=" * 92)
print("WHY THE CB CEILING LOOKS LIKE A FIX AND IS NOT ONE")
print("=" * 92)
print("cb_advance_clamp is a one-sided MAX at CB_MAX_ADVANCE_NX = 58.0.")
print("Fed the persistence chain above, every CB target exceeds it, so the")
print("clamp is permanently saturated and every CB sits ON the ceiling:")
for home in (22.0, 24.0, 26.0):
    print(f"   home {home:.0f} -> unclamped settles "
          f"{fixed_point(home, 70.0):.1f}  ->  clamped sits at 58.0")
print()
print("Measured live (seed 777, 144,971 CB samples): p25 55.5  p50 56.8  p75 57.9.")
print("That is a distribution with its mode ON the clamp. It is not a back")
print("four holding a line; it is three players parked at a ceiling, and the")
print("clamp is what is holding them there.")
