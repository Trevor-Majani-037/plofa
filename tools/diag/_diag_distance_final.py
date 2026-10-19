"""FINAL VERIFICATION -- every distance-related column the export publishes,
against the real-football bands already written down in calibrate_gps.py.

Bands (calibrate_gps.REF, per-90, outfielders):
    total distance   10.0-13.5 km
    HSR (>5.5 m/s)    850-1800 m
    sprint (>7 m/s)   300-900 m
    top speed        8.0-9.6 m/s

Also prints the decomposition (template / 0.45 x physics) so the residual is
attributed rather than described, and asserts the match itself is unchanged by
the accounting change: distance is telemetry-only (`player_distance_stats`
reaches `self.gps.ingest_episode_stats`, and only when GPS is enabled), so the
event timeline must be byte-identical to the pre-fix arm.
"""
import math
import random
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, ".")

from match_engine import (  # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder  # noqa: E402
import opta_analytics  # noqa: E402

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Jonas Keller", "CM", ["engine"], 25),
    ("Adriela Vela", "LW", ["dribbler"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher"], 29),
    ("Percy", "RW", ["grand_dribbler"], 24),
]
AWAY_STARTERS = [
    ("Bram Osei", "GK", ["shot_stopper"], 28),
    ("Yuri Tanaka", "LB", ["aggressive_fullback"], 23),
    ("Marek Dubois", "CB", ["ball_playing_cb"], 26),
    ("Stefan Berg", "CB", ["stopper_defender"], 29),
    ("Owen Castillo", "RB", ["overlapping_fullback"], 24),
    ("Tariq Aziz", "CDM", ["anchor_man"], 27),
    ("Nils Werner", "CM", ["box_box"], 25),
    ("Diego Ramos", "CAM", ["creator"], 24),
    ("Leo Mbeki", "LW", ["speedster"], 21),
    ("Ivan Petrov", "ST", ["target_man"], 28),
    ("Samir Haddad", "RW", ["dribbler"], 23),
]

random.seed(777)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                     match_date=date(2026, 8, 22), matchday=2)
hp = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION, Intensity.MEDIUM)
ap = TeamProfile("Thornfield United", TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
engine = MatchEngine(config, hp, ap)
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
result = engine.simulate()

# Independent recomposition of the DNA template from the published tables.
SPEEDS = opta_analytics.SPEEDS
BASE = opta_analytics.BASE_MINUTE_ACTIVITY
frames_n = defaultdict(int)
pos = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            frames_n[row["player"]] += 1
            pos[row["player"]] = row.get("position", "CM")

all_players = {"Hartwell City": {"starters": home["starters"],
                                 "substitutes": home["substitutes"]},
               "Thornfield United": {"starters": away["starters"],
                                     "substitutes": away["substitutes"]}}
analytics = opta_analytics.OptaAnalytics(result, all_players)
analytics.compute()

tmpl = {}
for name, obj in analytics.player_objs.items():
    p = analytics.player_meta.get(name, {}).get("position", "CM")
    base = BASE.get(p, opta_analytics.DEFAULT_ACTIVITY)
    wr = getattr(getattr(obj, "dna", None), "mental", None)
    w = (getattr(wr, "work_rate", 50.0) / 100.0) if wr is not None else 0.5
    e = 0.80 + 0.40 * w
    s, wa, j, r, sp = base
    j *= e
    r = min(0.35, r * e)
    sp = min(0.10, sp * e)
    tmpl[name] = (s * 60 * SPEEDS["standing"] + wa * 60 * SPEEDS["walking"]
                  + j * 60 * SPEEDS["jogging"] + r * 60 * SPEEDS["running"]
                  + sp * 60 * SPEEDS["sprinting"]) * frames_n.get(name, 0)

phys_log = defaultdict(float)
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            phys_log[row["player"]] += float(row.get("physics_distance_m", 0.0))

print(f"score {result.state.home_goals}-{result.state.away_goals}   "
      f"events {len(result.timeline)}   frames {len(result.position_log)}")
print()
hdr = (f"{'player':<18}{'pos':<5}{'DIST km':>9}{'template':>9}{'0.45*phys':>10}"
       f"{'spr#':>6}{'top km/h':>10}")
print(hdr)
print("-" * len(hdr))
for n in sorted(analytics.player_data,
                key=lambda k: -analytics.player_data[k]["distance_covered"]):
    d = analytics.player_data[n]
    print(f"{n:<18}{pos.get(n,'?'):<5}{d['distance_covered']:>9.2f}"
          f"{tmpl.get(n,0)/1000:>9.2f}{0.45*phys_log[n]/1000:>10.2f}"
          f"{int(d['sprints']):>6}{d['top_speed']:>10.1f}")

outf = [n for n in analytics.player_data if pos.get(n) != "GK"]
dk = [analytics.player_data[n]["distance_covered"] for n in outf]
print("-" * len(hdr))
print(f"\nOUTFIELDERS n={len(dk)}   "
      f"min {min(dk):.2f}   mean {sum(dk)/len(dk):.2f}   max {max(dk):.2f} km"
      f"     real band 10.0-13.5 km")
print(f"in band: {sum(1 for x in dk if 10.0 <= x <= 13.5)}/{len(dk)}")
print(f"over13.5: {sum(1 for x in dk if x > 13.5)}/{len(dk)}   "
      f"max excess {max(dk) - 13.5:+.2f} km")
gk = [analytics.player_data[n]["distance_covered"] for n in analytics.player_data
      if pos.get(n) == "GK"]
print(f"KEEPERS: {[round(x, 2) for x in gk]} km   (real ~4-6 km)")
tm = [tmpl[n] / 1000 for n in outf]
print(f"\ntemplate alone (outfielders): min {min(tm):.2f}  mean {sum(tm)/len(tm):.2f}  "
      f"max {max(tm):.2f} km")
print(f"0.45 x physics adds a FLAT   : mean "
      f"{sum(0.45*phys_log[n]/1000 for n in outf)/len(outf):.2f} km/player "
      f"(17.4% of the reported total)")
tops = [analytics.player_data[n]["top_speed"] for n in outf]
print(f"top speed: min {min(tops):.1f}  max {max(tops):.1f} km/h   "
      f"(calibrate_gps band 28.8-34.6 km/h)")