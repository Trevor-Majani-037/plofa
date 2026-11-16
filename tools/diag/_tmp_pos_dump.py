import json, math, sys

d = json.load(open("plofa_broadcast/replay.json", encoding="utf-8"))
pp = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
      for n, rows in d["player_path"].items()}


def at(t, n):
    s = pp[n]
    lo, hi = 0, len(s) - 1
    best = None
    while lo <= hi:
        m = (lo + hi) // 2
        if s[m][0] <= t:
            best = s[m]
            lo = m + 1
        else:
            hi = m - 1
    return best


for T in (568.0, 1875.5, 4355.0):
    pos = [(n,) + tuple(at(T, n)[1:]) for n in pp if at(T, n) is not None]
    print("=" * 60)
    print("t", T, "players", len(pos))
    pairs = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dd = math.hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2])
            if dd < 1.5:
                pairs.append((pos[i][0], pos[j][0], round(dd, 2)))
    print("pairs <1.5 m:", len(pairs))
    for p in pairs:
        print("   ", p)
    for n, x, y in sorted(pos, key=lambda p: (-p[2], p[1])):
        print(f"    {n:24s} {x:6.1f} {y:6.1f}")
