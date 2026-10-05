"""What ARE the shot coordinates, actually?

The shot invariant fails on real data under BOTH frame conventions — half-based
and per-acting-team — so the assumption underneath both is wrong. That
assumption is that `location_x`/`end_x` on a shot event live in the same global
pitch frame as everything else.

Rather than try a third convention on theory, print what the engine actually
emits: raw coordinates, the engine's own attacking-direction state, and the
ball's own position at that moment (the GPS is a separate, trusted record).
If the shot events disagree with the GPS about where the ball was, the shot
coordinates are in a different frame and no amount of mirroring will fix it.
"""
import io
import random
import sys

SHOTS = ("SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
         "GOAL")


def main():
    sys.path.insert(0, ".")
    from _diag_watch import build

    eng, _, _ = build()
    eng.enable_virtual_gps(0.1)
    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    tar = eng.position_engine.team_attacks_right
    H, A = eng.config.home_team, eng.config.away_team
    print(f"SCORE {res.score_str}")
    print(f"team_attacks_right (1st half): {tar}")

    # GPS: where the engine actually put the ball, per second
    gps = [(s["t"], s["x"], s["y"]) for s in eng.gps.samples
           if s["player"] == "__ball__"]

    def ball_at(clock_s):
        best = None
        for t, x, y in gps:
            if t <= clock_s:
                best = (t, x, y)
            else:
                break
        return best

    print(f"\n{'min':>4} {'team':<10}{'type':<18}{'raw_at':>8}{'raw_to':>8}"
          f"{'gps_ball':>10}{'teamR':>7}")
    for e in res.timeline:
        t = getattr(e, "event_type", None)
        if t is None or t.name not in SHOTS:
            continue
        team = getattr(e, "team", "")
        clock = (getattr(e, "minute", 0) or 0) * 60 + \
            (getattr(e, "second", 0) or 0)
        b = ball_at(clock)
        gx = f"{b[1]:.1f}" if b else "-"
        print(f"{getattr(e,'minute',0):>4} {team:<10}{t.name:<18}"
              f"{getattr(e,'location_x',0) or 0:>8.1f}"
              f"{getattr(e,'end_x',0) or 0:>8.1f}"
              f"{gx:>10}{str(tar.get(team)):>7}")

    # Where does the GPS put each team's shots relative to their own goal?
    print("\nwhere each team's shots happen, per the GPS (the trusted record):")
    for team, goal in ((H, 105.0), (A, 0.0)):
        xs = []
        for e in res.timeline:
            t = getattr(e, "event_type", None)
            if t is None or t.name not in SHOTS or \
                    getattr(e, "team", "") != team:
                continue
            clock = (getattr(e, "minute", 0) or 0) * 60 + \
                (getattr(e, "second", 0) or 0)
            b = ball_at(clock)
            if b:
                xs.append(b[1])
        if xs:
            print(f"   {team}: n={len(xs)} gps_x min={min(xs):.1f} "
                  f"max={max(xs):.1f}  (their target is x={goal})")


if __name__ == "__main__":
    main()
