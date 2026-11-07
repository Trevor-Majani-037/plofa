"""Forensics: why broadcast players teleport in PLOFA replay.

Reads plofa_broadcast/replay.json (saved ReplayData) and quantifies:
  1. player speed distribution between consecutive recorded samples
  2. whether big jumps land ON the ball (touch_at/event sync) or
     land on a shape anchor (drift/block pull)
  3. team shape spread out of possession vs in possession
  4. bench/ghost samples before substitutions

Read-only. Prints ASCII-safe output.
Run:  .\\.venv\\Scripts\\python.exe -u tools\\diag\\_diag_broadcast_teleport.py
"""

import json
import math
import os
import statistics
from collections import Counter

REPLAY = os.path.join(os.path.dirname(__file__), "..", "..", "plofa_broadcast", "replay.json")
REPLAY = os.path.normpath(REPLAY)


def ascii_safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def main():
    with open(REPLAY, encoding="utf-8") as f:
        d = json.load(f)
    pp = d.get("player_path", {}) or {}
    ball = d.get("ball", []) or []
    frames = d.get("frames", []) or []
    print("[replay] players:", len(pp), "ball samples:", len(ball), "frames:", len(frames))
    if not pp:
        print("[replay] no player_path - nothing to analyse")
        return

    # ball lookup by time (payload ball is monotonic t,x,y)
    bt = [p["t"] for p in ball]
    bx = [p["x"] for p in ball]
    by = [p["y"] for p in ball]

    def ball_at(t):
        import bisect
        i = bisect.bisect_left(bt, t)
        if i == 0:
            return (bx[0], by[0])
        if i >= len(bt):
            return (bx[-1], by[-1])
        t0, t1 = bt[i - 1], bt[i]
        f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (bx[i - 1] + (bx[i] - bx[i - 1]) * f,
                by[i - 1] + (by[i] - by[i - 1]) * f)

    speeds = []
    jumps = []  # (t, name, dist, dt, v, d_ball_before, d_ball_after)
    per_player = Counter()
    for name, path in pp.items():
        per_player[name] = len(path)
        for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
            dt = float(t1) - float(t0)
            if dt <= 0:
                continue
            dist = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            v = dist / dt
            speeds.append(v)
            if dist > 15:
                b0 = ball_at(float(t0))
                b1 = ball_at(float(t1))
                d0 = math.hypot(float(x0) - b0[0], float(y0) - b0[1])
                d1 = math.hypot(float(x1) - b1[0], float(y1) - b1[1])
                jumps.append((float(t1), name, round(dist, 1), round(dt, 2),
                              round(v, 1), round(d0, 1), round(d1, 1)))

    speeds.sort()
    def pct(q):
        return speeds[min(len(speeds) - 1, int(q * len(speeds)))] if speeds else 0
    print("[speeds] n=%d  p50=%.2f  p90=%.2f  p99=%.2f  p99.9=%.2f  max=%.2f  m/s (real sprint ~7-10)" % (
        len(speeds), pct(0.50), pct(0.90), pct(0.99), pct(0.999), speeds[-1] if speeds else 0))
    print("[speeds] >8 m/s: %d (%.2f%%)  >12: %d  >25: %d" % (
        sum(1 for v in speeds if v > 8), 100 * sum(1 for v in speeds if v > 8) / max(1, len(speeds)),
        sum(1 for v in speeds if v > 12), sum(1 for v in speeds if v > 25)))
    jumps.sort(key=lambda j: -j[2])
    print("[jumps] >15m between consecutive samples: %d" % len(jumps))
    on_ball = [j for j in jumps if j[6] < 3]
    off_ball = [j for j in jumps if j[6] >= 3 and j[5] >= 3]
    print("[jumps] land on ball (<3m): %d  fly off-ball (>=3m to >=3m): %d" % (len(on_ball), len(off_ball)))
    print("[worst 12 jumps]  t, player, dist, dt, v, d_ball_before, d_ball_after")
    for j in jumps[:12]:
        print("  %7.1fs  %-18s %5.1fm in %4.2fs = %5.1f m/s  ball %5.1f -> %5.1f" % (
            j[0], ascii_safe(j[1]), j[2], j[3], j[4], j[5], j[6]))
    print("[worst off-ball jumps (not ball arrivals)]")
    for j in sorted(off_ball, key=lambda j: -j[2])[:8]:
        print("  %7.1fs  %-18s %5.1fm in %4.2fs = %5.1f m/s  ball %5.1f -> %5.1f" % (
            j[0], ascii_safe(j[1]), j[2], j[3], j[4], j[5], j[6]))

    print("[samples per player] min=%d max=%d  first sample times:" % (
        min(per_player.values()), max(per_player.values())))
    for name in list(per_player)[:4]:
        path = pp[name]
        print("  %-18s n=%d  first t=%.1f last t=%.1f" % (
            ascii_safe(name), per_player[name], path[0][0], path[-1][0]))

    gaps = []
    for name, path in pp.items():
        for (t0, _, _), (t1, _, _) in zip(path, path[1:]):
            dt = float(t1) - float(t0)
            if dt > 1.0:
                gaps.append((dt, name, float(t1)))
    gaps.sort(reverse=True)
    print("[gaps] n>1s=%d  max gap=%.1fs" % (len(gaps), gaps[0][0] if gaps else 0))
    print("[worst gaps]  dt, player, t_end")
    for dt, name, te in gaps[:10]:
        print("  %6.2fs  %-18s cont t=%.1f" % (dt, ascii_safe(name), te))
    inst = [j for j in jumps if j[3] <= 1.0]
    print("[instant teleports] dt<=1s & dist>15m: %d (%.2f%% of pairs)" % (len(inst), 100*len(inst)/max(1,len(speeds))))
    print("[worst instant teleports]")
    for j in sorted(inst, key=lambda j: -j[2])[:12]:
        print("  %7.1fs  %-18s %5.1fm in %4.2fs = %5.1f m/s  ball %5.1f -> %5.1f" % (
            j[0], ascii_safe(j[1]), j[2], j[3], j[4], j[5], j[6]))
    on_inst = [j for j in inst if j[6] < 3]
    off_inst = [j for j in inst if j[6] >= 3 and j[5] >= 3]
    print("[instant] land on ball: %d  fly off-ball: %d" % (len(on_inst), len(off_inst)))
    # annotate off-ball teleports with nearest timeline event
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "plofa_broadcast", "broadcast_data.json"), encoding="utf-8") as _ef:
            _evs = json.load(_ef).get("events", []) or []
    except Exception:
        _evs = []
    import bisect as _bis
    _et = [e["t"] for e in _evs]
    ctx = Counter()
    for j in off_inst:
        i = _bis.bisect_left(_et, j[0])
        best = None
        for cand in (i - 1, i):
            if 0 <= cand < len(_evs):
                if best is None or abs(_evs[cand]["t"] - j[0]) < abs(_evs[best]["t"] - j[0]):
                    best = cand
        if best is not None and abs(_evs[best]["t"] - j[0]) < 3.0:
            ctx[_evs[best]["type"]] += 1
        else:
            ctx["NO EVENT WITHIN 3s"] += 1
    print("[off-ball teleport context] nearest event type within 3s:", ctx.most_common(10))
    for j in sorted(off_inst, key=lambda j: -j[2])[:8]:
        print("  OFF %7.1fs  %-18s %5.1fm in %4.2fs = %5.1f m/s  ball %5.1f -> %5.1f" % (
            j[0], ascii_safe(j[1]), j[2], j[3], j[4], j[5], j[6]))

    print("[gap dt histogram >1s] " + str(Counter(round(g[0]) for g in gaps).most_common(8)))

    # ── SHAPE: defending line depths (metres from own goal), per frame ──
    # Home attacks right (own goal x=0); away attacks left (own goal x=105).
    BACK = ("CB", "LB", "RB"); MID = ("CDM", "CM", "CAM"); FRONT = ("ST", "CF", "LW", "RW")
    for side, own_goal in (("home", 0.0), ("away", 105.0)):
        att = (side == "home")
        back, mid, front, width = [], [], [], []
        def_depths = []
        for fr in frames:
            poss = fr.get("possession_team", "")
            defending = (poss != (d["home_team"] if side == "home" else d["away_team"]))
            if not defending:
                continue
            rows = fr.get(side, [])
            def _mean(cands):
                xs = [r["x"] for r in rows if r.get("position") in cands]
                return (sum(xs) / len(xs)) if xs else None
            ys = [r["y"] for r in rows]
            if _mean(BACK) is not None:
                back.append(_mean(BACK) if att else 105 - _mean(BACK))
                mid.append(_mean(MID) if att else 105 - _mean(MID))
                front.append(_mean(FRONT) if att else 105 - _mean(FRONT))
                width.append(max(ys) - min(ys) if ys else 0)
        def _s(v):
            return ("n=%d mean=%.1f sd=%.1f p10=%.1f p90=%.1f" % (
                len(v), statistics.mean(v), statistics.pstdev(v),
                sorted(v)[len(v)//10], sorted(v)[-len(v)//10])) if v else "n=0"
        print("[shape-def] %s defending (own-goal depth, metres) back=%s mid=%s front=%s width=%s" % (
            side, _s(back), _s(mid), _s(front), _s(width)))
    if frames:
        lens = sorted(len(f.get("home", []) + f.get("away", [])) for f in frames)
        print("[frames] home+away rows per minute: min=%d p50=%d max=%d" % (lens[0], lens[len(lens)//2], lens[-1]))
    # shape spread: collect out-of-sample team centroids per state chunk
    # (use frames possession_team)
    import bisect as _b
    for side in ("home", "away"):
        xs, ys = [], []
        n_in = n_out = 0
        for fr in frames:
            rows = fr.get(side, [])
            if not rows:
                continue
            for r in rows:
                xs.append(r["x"]); ys.append(r["y"])
                if fr.get("possession_team") == d["home_team" if side == "home" else "away_team"]:
                    n_in += 1
                else:
                    n_out += 1
        if xs:
            print("[shape] %s x mean=%.1f sd=%.1f | y mean=%.1f sd=%.1f | rows in=%d out=%d" % (
                side, statistics.mean(xs), statistics.pstdev(xs),
                statistics.mean(ys), statistics.pstdev(ys), n_in, n_out))


if __name__ == "__main__":
    main()
