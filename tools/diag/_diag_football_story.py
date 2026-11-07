"""Football story of the current replay, in the terms a coach uses.

Reads the per-minute frames (which carry cumulative physics per player) plus
the ball path, and prints:

  * the score and the possession share the sim believes,
  * who ran, how far, how often they sprinted, and their measured top speed,
  * where each team held its line (mean x of the back four, by half),
  * how much of the match the ball spent in each third,
  * what the keepers actually did.

Usage:
    python tools/diag/_diag_football_story.py [replay.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

THIRD = 105.0 / 3.0


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
    d = json.load(open(path, encoding="utf-8"))

    home, away = d["home_team"], d["away_team"]
    print(f"{home} {d['home_goals']}-{d['away_goals']} {away}   "
          f"{d['duration_s'] / 60.0:.0f} min")
    print(f"possession (sim)  {home} {d['home_possession']:.1f}% / "
          f"{away} {100.0 - d['home_possession']:.1f}%")

    last = d["frames"][-1]
    rows = [(home, r) for r in last["home"]] + [(away, r) for r in last["away"]]

    # `physics_*` in a frame is that MINUTE's accumulation, reset each minute
    # (`reset_minute_physics`). Summing across frames is the only way to get a
    # match total; reading the last frame gives one minute's running and looks
    # absurd (0.12 km per outfielder).
    totals = {}
    for f in d["frames"]:
        for team, side in ((home, "home"), (away, "away")):
            for r in f[side]:
                acc = totals.setdefault(r["player"], {
                    "player": r["player"], "team": team,
                    "position": r["position"],
                    "m": 0.0, "sprints": 0, "hi": 0, "top": 0.0})
                acc["m"] += r["physics_distance_m"]
                acc["sprints"] += int(r["physics_sprint_count"])
                acc["hi"] += int(r["physics_high_speed_sprint_count"])
                acc["top"] = max(acc["top"], r["physics_top_speed_mps"])

    print("\ndistance covered (km) -- a full 90 for an outfielder is 9-12")
    for team in (home, away):
        tr = [a for a in totals.values()
              if a["team"] == team and a["position"] != "GK"]
        tot = sum(a["m"] for a in tr) / 1000.0
        top = sorted(tr, key=lambda a: -a["m"])[:3]
        print(f"  {team:22s} total {tot:6.2f} km   "
              f"avg {tot / max(len(tr), 1):5.2f} km/player")
        for a in top:
            print(f"      {a['player']:20s} {a['m'] / 1000.0:5.2f} km  "
                  f"sprints {a['sprints']:3d}  top {a['top']:4.2f} m/s")

    print("\nsprints per player (>= high intensity)")
    for team in (home, away):
        tr = [a for a in totals.values()
              if a["team"] == team and a["position"] != "GK"]
        print(f"  {team:22s} {sum(a['hi'] for a in tr):4d} high-speed sprints")

    print("\nline height: mean x of the back four, by half "
          "(attacking right in 1st half for the home side)")
    for minute in (20, 45, 70, 90):
        f = d["frames"][min(minute, len(d["frames"])) - 1]
        out = []
        for side in ("home", "away"):
            back = [r["x"] for r in f[side]
                    if r["position"] in ("LB", "RB", "CB")]
            if back:
                out.append(f"{side} {sum(back) / len(back):5.1f}")
        print(f"  {minute:3d}'  " + "   ".join(out))

    ball = d["ball"]
    thirds = {"own": 0, "middle": 0, "final": 0}
    for row in ball:
        # `ball` rows are dicts (t, x, y, kind, team), not tuples: the replay
        # carries the kind and the team because the broadcast draws them.
        x = float(row["x"])
        k = "own" if x < THIRD else ("final" if x > 2 * THIRD else "middle")
        thirds[k] += 1
    n = max(len(ball), 1)
    print("\nwhere the ball lived (absolute thirds of the pitch)")
    for k, v in thirds.items():
        print(f"  {k:7s} {100.0 * v / n:5.1f}%")

    print("\nkeepers")
    for team in (home, away):
        gk = next((a for a in totals.values()
                   if a["team"] == team and a["position"] == "GK"), None)
        if gk:
            print(f"  {team:22s} {gk['player']:16s} "
                  f"{gk['m'] / 1000.0:5.2f} km  "
                  f"top {gk['top']:4.2f} m/s")


if __name__ == "__main__":
    main()
