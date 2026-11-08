"""What are the bodies actually DOING? The five locomotion bands, measured.

The sprint COUNT is a derived quantity with a 1 s sustain rule and a cooldown.
The banded TIME is the primary measurement: every 0.1 s tick is classed by the
speed the body actually travelled at, using Di Salvo's published GPS edges:

    standing   < 0.5 m/s
    walking    0.5 - 2.0 m/s
    jogging    2.0 - 4.2 m/s
    running    4.2 - 7.0 m/s
    sprinting  >= 7.0 m/s

If the bands are healthy but the sprint count is low, the movement is fine
and the counter is strict. If the bands are dominated by standing, the players
are not moving and the count is telling the truth.

Reference shares for a Premier League outfielder over 90 min (Di Salvo et al.,
broadly): standing ~17-22%, walking ~35-42%, jogging ~18-25%,
running ~10-16%, sprinting ~2-5%.

Usage:
    python tools/diag/_diag_locomotion_bands.py [replay.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

BANDS = ("standing", "walk", "jog", "running", "sprint")
LABEL = {"standing": "standing", "walk": "walking", "jog": "jogging",
         "running": "running", "sprint": "sprinting"}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
    d = json.load(open(path, encoding="utf-8"))
    duration = float(d["duration_s"])

    acc = defaultdict(lambda: {"m": 0.0, "pos": "", "team": "",
                               **{b: 0.0 for b in BANDS}})
    for f in d["frames"]:
        for side in ("home", "away"):
            for r in f[side]:
                a = acc[r["player"]]
                a["pos"] = r["position"]
                a["team"] = side
                a["m"] += r["physics_distance_m"]
                for b in BANDS:
                    a[f"physics_{b}_time_s"] = 0.0  # noqa: E113 (reset below)
                    a[b] += r[f"physics_{b}_time_s"]

    print(f"{d['home_team']} {d['home_goals']}-{d['away_goals']} {d['away_team']}"
          f"   {duration / 60.0:.0f} min")
    print(f"match seconds available per player: {duration:.0f} s\n")

    hdr = f"{'player':20s} {'pos':4s} {'km':>5s} " + " ".join(
        f"{LABEL[b][:4]:>6s}" for b in BANDS) + "   booked%"
    print(hdr)
    print("-" * len(hdr))

    tot = {b: 0.0 for b in BANDS}
    n_players = 0
    for name, a in sorted(acc.items(), key=lambda kv: -kv[1]["m"]):
        booked = sum(a[b] for b in BANDS)
        for b in BANDS:
            tot[b] += a[b]
        n_players += 1
        pct = {b: (100.0 * a[b] / booked if booked else 0.0) for b in BANDS}
        print(f"{name:20s} {a['pos']:4s} {a['m'] / 1000.0:5.2f} "
              + " ".join(f"{pct[b]:5.1f}%" for b in BANDS)
              + f"   {100.0 * booked / duration:5.1f}%")

    grand = sum(tot.values()) or 1.0
    print("-" * len(hdr))
    print(f"{'WHOLE MATCH':20s} {'':4s} {'':5s} "
          + " ".join(f"{100.0 * tot[b] / grand:5.1f}%" for b in BANDS)
          + f"   {100.0 * grand / (duration * n_players):5.1f}%")

    print("\ninterpretation")
    print(f"  booked seconds, all players: {grand:,.0f} "
          f"of {duration * n_players:,.0f} possible "
          f"({100.0 * grand / (duration * n_players):.1f}%)")
    for b in BANDS:
        print(f"  {LABEL[b]:10s} {tot[b] / max(n_players, 1) / 60.0:6.1f} min/player"
              f"   {100.0 * tot[b] / grand:5.1f}%")


if __name__ == "__main__":
    main()
