"""IS THE SHOT INTEGRATED PER TICK, OR ONLY SOLVED?

`aim_shot_flight` returns a `BallFlight` (start + target, scalars).
`episode.resolve_shot(...)` returns an outcome, a goal point and a
`flight_time`. `physics_meta` reports `ticks` and `elapsed_s` for the episode.

So: if the shot is STEPPED, its flight appears in the episode trace and
`elapsed_s` / `ticks` for the shot action should match the flight time — about
0.7-1.3 s for a 25 m effort at 20-34 m/s. If the shot is only SOLVED, the
episode's tick count reflects the rest of the possession and the shot
contributes no samples of its own.

Print the physics block for every shot, plus the flight arithmetic, so the
answer is arithmetic rather than a reading of somebody's comment.
"""
import io
import random
import sys

SHOTS = ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL",
         "HIT_WOODWORK", "SAVE")


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
    print(f"SCORE {res.score_str}")

    n_physics = n_shot = 0
    print(f"\n{'min':>4} {'type':<17}{'ticks':>7}{'elapsed':>9}{'flight_s':>9}"
          f"{'launch':>8}{'avg_mps':>9}{'dist_m':>8}{'flight_calc':>12}")
    rows = []
    for e in res.timeline:
        t = getattr(e, "event_type", None)
        if t is None or t.name not in SHOTS:
            continue
        n_shot += 1
        md = getattr(e, "metadata", None) or {}
        ph = md.get("physics") or {}
        if ph:
            n_physics += 1
        launch = ph.get("shot_speed_mps") or 0.0
        ft = ph.get("flight_time_s") or 0.0
        lx, ly = getattr(e, "location_x", 0) or 0, getattr(e, "location_y", 0) or 0
        ex, ey = getattr(e, "end_x", 0) or 0, getattr(e, "end_y", 0) or 0
        dist = ((ex - lx) ** 2 + (ey - ly) ** 2) ** 0.5
        calc = dist / launch if launch else 0.0
        rows.append((t.name, ph.get("ticks"), ph.get("elapsed_s"),
                     ph.get("flight_s"), launch, dist, calc))
        if len(rows) <= 18:
            print(f"{getattr(e,'minute',0):>4} {t.name:<17}"
                  f"{str(ph.get('ticks','-')):>7}{str(ph.get('elapsed_s','-')):>9}"
                  f"{str(ph.get('flight_time_s','-')):>9}"
                  f"{launch:>8.1f}{str(ph.get('ball_speed_mps','-')):>9}"
                  f"{dist:>8.1f}{calc:>12.2f}")

    print(f"\nshot-family events: {n_shot}, carrying a physics block: {n_physics}")
    if not rows:
        print("NO physics metadata on any shot event.")
        return

    import statistics as st
    ticks = [r[1] for r in rows if isinstance(r[1], (int, float))]
    elapsed = [r[2] for r in rows if isinstance(r[2], (int, float))]
    flights = [r[3] for r in rows if isinstance(r[3], (int, float))]
    calc = [r[6] for r in rows if r[6]]
    print(f"episode ticks on a shot:   median {st.median(ticks) if ticks else '-'}"
          f"  range {min(ticks) if ticks else '-'}..{max(ticks) if ticks else '-'}")
    print(f"episode elapsed_s:         median {st.median(elapsed) if elapsed else '-'}")
    print(f"declared flight_time_s:    median {st.median(flights) if flights else '-'}")
    print(f"distance/launch_speed:     median {st.median(calc) if calc else '-'} s"
          f"   <- what the flight SHOULD take")
    print()
    a = st.median(elapsed) if elapsed else 0
    b = st.median(flights) if flights else 0
    c = st.median(calc) if calc else 0
    if b and c and abs(a - c) / max(c, 0.01) < 0.35:
        print("VERDICT: episode elapsed on a shot MATCHES the shot's own flight")
        print("time -> the shot IS being stepped tick by tick, and the points")
        print("exist in the episode trace. They are simply not exported.")
    elif b and c and b > 0 and abs(b - c) / max(c, 0.01) < 0.35:
        print("VERDICT: the DECLARED flight time matches the arithmetic, but the")
        print("episode did not spend that long -> solved, not stepped.")
    else:
        print("VERDICT: inconclusive from these three numbers alone; compare")
        print("elapsed, declared flight time and distance/launch above.")


if __name__ == "__main__":
    main()
