"""DOES A REAL SHOT TRAJECTORY EXIST?

The infrastructure is there: `MatchResult.full_match_ball_path` is a per-0.1 s
ball track, `PossessionChain` fills its per-chain `ball_path` with real
positions, and `pitch_replay` plays it back. But two comments in
`match_engine.py` claim that shot chains do NOT export a path and fall through
to `_synthesize_ball_path` — which would make the dotted line on the shot map
synthetic at BOTH ends: `exporter._shot_trajectory` invents the destination,
and the ball path during the shot would be an interpolation.

Comments have been wrong repeatedly, so measure instead:

  * how many ball samples fall inside each shot event's time window
  * the implied speed profile — a struck ball DECELERATES; a lerp is constant
  * straightness — an interpolation is exactly straight, a struck ball is not
    quite
  * where the path ends versus where the shot's own recorded outcome says it
    should end

If the samples are real, the plot can use them. If they are synthetic, the
honest answer is that the trajectory does not exist and should not be faked.
"""
import io
import random
import sys

import numpy as np

SHOTS = ("SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
         "GOAL", "HIT_WOODWORK", "SAVE")


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

    path = list(getattr(res, "full_match_ball_path", None) or [])
    print(f"full_match_ball_path: {len(path)} samples")
    if path:
        print(f"  sample keys: {sorted(path[0].keys())}")
        xs = [p.get("x") for p in path if p.get("x") is not None]
        print(f"  x range {min(xs):.1f}..{max(xs):.1f}, "
              f"t range {min(p['t'] for p in path):.0f}.."
              f"{max(p['t'] for p in path):.0f}")

    ts = np.array([p["t"] for p in path], dtype=float)
    bx = np.array([p.get("x", np.nan) for p in path], dtype=float)
    by = np.array([p.get("y", np.nan) for p in path], dtype=float)
    src = {}
    for p in path:
        src[p.get("source", "?")] = src.get(p.get("source", "?"), 0) + 1
    print(f"  by 'source': {src}")

    print(f"\n{'min':>4} {'team':<10}{'type':<17}{'n':>4}{'dur_s':>7}"
          f"{'v_in':>7}{'v_out':>7}{'straight':>10}{'end_x':>8}{'rec_end_x':>10}")
    rows = []
    for e in res.timeline:
        t = getattr(e, "event_type", None)
        if t is None or t.name not in SHOTS:
            continue
        t0 = (getattr(e, "minute", 0) or 0) * 60 + (getattr(e, "second", 0) or 0)
        i0 = int(np.searchsorted(ts, t0))
        i1 = int(np.searchsorted(ts, t0 + 3.0))
        seg = [(ts[i], bx[i], by[i]) for i in range(i0, i1)
               if not (np.isnan(bx[i]) or np.isnan(by[i]))]
        if len(seg) < 4:
            continue
        a = np.array(seg)
        d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
        dt = np.diff(a[:, 0])
        good = dt > 0
        v = d[good] / dt[good]
        # straightness: deviation from the straight origin->end line
        p0, p1 = a[0, 1:], a[-1, 1:]
        L = np.hypot(*(p1 - p0))
        dev = 0.0
        if L > 1:
            n = (p1 - p0) / L
            dev = float(np.max(np.abs(
                (a[:, 1] - p0[0]) * n[1] - (a[:, 2] - p0[1]) * n[0])))
        rows.append(dict(
            minute=getattr(e, "minute", 0), team=getattr(e, "team", ""),
            type=t.name, n=len(seg), dur=a[-1, 0] - a[0, 0],
            v_in=float(v[0]) if len(v) else 0.0,
            v_out=float(v[-1]) if len(v) else 0.0, dev=dev,
            end_x=float(a[-1, 1]), rec_end_x=getattr(e, "end_x", None)))
    for r in rows[:16]:
        rec = f"{r['rec_end_x']:.1f}" if r["rec_end_x"] is not None else "-"
        print(f"{r['minute']:>4} {r['team']:<10}{r['type']:<17}{r['n']:>4}"
              f"{r['dur']:>7.2f}{r['v_in']:>7.1f}{r['v_out']:>7.1f}"
              f"{r['dev']:>10.2f}{r['end_x']:>8.1f}{rec:>10}")

    if not rows:
        print("\nNO shot event had 4+ ball samples within 3 s — the shot "
              "window is not covered by the recorded path at all.")
        return

    devs = [r["dev"] for r in rows]
    decel = sum(1 for r in rows if r["v_out"] < r["v_in"] * 0.95)
    print(f"\nshots with >=4 samples: {len(rows)}")
    print(f"  max deviation from a straight line: median {np.median(devs):.3f} m,"
          f" max {max(devs):.3f} m")
    print(f"    (a lerp/synthesis gives ~0.000 on every shot)")
    print(f"  shots that DECELERATE (v_out < 0.95 * v_in): {decel}/{len(rows)}")
    print(f"    (a struck ball always decelerates; a constant-speed lerp does "
          f"not)")
    match = sum(1 for r in rows if r["rec_end_x"] is not None
                and abs(r["end_x"] - r["rec_end_x"]) < 3.0)
    tot = sum(1 for r in rows if r["rec_end_x"] is not None)
    print(f"  path end within 3 m of the event's recorded end_x: "
          f"{match}/{tot}")
    print("\nVERDICT: compare the deviation and deceleration numbers above.")
    print("Near-zero deviation on every row means the window is interpolated,")
    print("not recorded, and the true shot trajectory does not exist.")


if __name__ == "__main__":
    main()
