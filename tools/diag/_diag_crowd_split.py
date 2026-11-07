"""Split broadcast crowding into TEAMMATE vs OPPONENT pairs.

Teammates must never share a coordinate (that is a rendering bug). Opponents
standing close is football (a duel, a marker). Team is recovered from
replay.json's player_team map, falling back to the event feed.

Usage:
    python tools/diag/_diag_crowd_split.py [path/to/replay.json]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


STEP = 0.5
CLOSE = 1.6


def series_at(samples, t):
    lo, hi, best = 0, len(samples) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            best = samples[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main(path: str) -> None:
    d = json.load(open(path, encoding="utf-8"))
    team = dict(d.get("player_team") or {})
    # replay.json does not carry a player->team map: derive it from the
    # per-minute frames, which list the home and away XI (subs included).
    for f in d.get("frames") or []:
        for side in ("home", "away"):
            for row in f.get(side) or []:
                team.setdefault(row.get("player"), f.get(side + "_team")
                                or d.get(side + "_team"))
    players = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
               for n, rows in d["player_path"].items()}
    end_t = max(s[-1][0] for s in players.values())

    mate_hist = defaultdict(int)
    opp_hist = defaultdict(int)
    worst_mate = []
    samples = 0
    mate_overlap_samples = 0
    t = 0.0
    while t <= end_t:
        pos = []
        for n, s in players.items():
            r = series_at(s, t)
            if r is not None:
                pos.append((n, r[1], r[2]))
        if len(pos) < 2:
            t += STEP
            continue
        samples += 1
        nm = no = 0
        closest = None
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dd = math.hypot(pos[i][1] - pos[j][1],
                                pos[i][2] - pos[j][2])
                if dd >= CLOSE:
                    continue
                same = team.get(pos[i][0], "?") == team.get(pos[j][0], "?")
                if same:
                    nm += 1
                    if closest is None or dd < closest[1]:
                        closest = (pos[i][0] + " / " + pos[j][0], dd)
                else:
                    no += 1
        mate_hist[nm] += 1
        opp_hist[no] += 1
        if nm:
            mate_overlap_samples += 1
        if closest is not None:
            worst_mate.append((t, closest[1], closest[0]))
        t += STEP

    worst_mate.sort(key=lambda r: r[1])
    print(f"file {path}")
    print(f"duration {end_t/60.0:.1f} min, {samples} samples @ {STEP}s")
    print(f"TEAMMATE pairs closer than {CLOSE} m")
    print(f"  samples with >=1 teammate overlap: {mate_overlap_samples} "
          f"({100.0*mate_overlap_samples/max(1,samples):.1f}%)")
    for k in sorted(mate_hist):
        print(f"    {k} pairs: {mate_hist[k]} samples")
    print(f"OPPONENT pairs closer than {CLOSE} m")
    for k in sorted(opp_hist):
        print(f"    {k} pairs: {opp_hist[k]} samples")
    print("closest teammate pairs:")
    for tt, dd, names in worst_mate[:8]:
        print(f"    {int(tt)//60:02d}:{tt%60:04.1f} {dd:4.2f} m  {names}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "plofa_broadcast/replay.json")
