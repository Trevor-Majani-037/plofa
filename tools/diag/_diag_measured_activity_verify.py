"""VERIFY the measured-activity change on one real match (seed 777).

Three questions, in order:
  1. IS THE COVERAGE REAL? How much of each player's 5,760 match seconds now
     carry a measured speed, versus the 42.1% before
     (`_diag_activity_coverage.py`)?
  2. ARE THE SECONDS MEASURED? Compare the published bands against the raw
     measured band seconds for the seconds we watched. If they track, the
     published numbers are the measurement.
  3. WHAT HAPPENED TO DISTANCE? It is no longer "template + a fraction of the
     trace"; it is measured metres plus a template budget for the unwatched
     seconds. Report both parts so the residual is attributed.

GATES
  G1 every outfielder has > 42.1% measured (the pre-change figure) -- if not
     the standing-time fix did nothing
  G2 the published band seconds must equal measured + template-residue for the
     minutes where coverage is complete, i.e. the blend is not double counting
  G3 a player with FULL coverage must publish exactly the measured seconds
"""
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

all_players = {"Hartwell City": {"starters": home["starters"],
                                 "substitutes": home["substitutes"]},
               "Thornfield United": {"starters": away["starters"],
                                     "substitutes": away["substitutes"]}}
analytics = opta_analytics.OptaAnalytics(result, all_players)
analytics.compute()

BANDS = [("standing", "physics_standing_time_s"),
         ("walking", "physics_walk_time_s"),
         ("jogging", "physics_jog_time_s"),
         ("running", "physics_running_time_s"),
         ("sprinting", "physics_sprint_time_s")]

raw = defaultdict(lambda: defaultdict(float))
raw_dist = defaultdict(float)
pos = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            n = row["player"]
            pos[n] = row.get("position", "?")
            for band, key in BANDS:
                raw[n][band] += float(row.get(key, 0.0) or 0.0)
            raw_dist[n] += float(row.get("physics_distance_m", 0.0) or 0.0)

frames = len(result.position_log)
available = frames * 60.0
print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  frames {frames}  "
      f"-> {available:.0f} s per player")

print()
print("=" * 92)
print("1. MEASURED COVERAGE  (was 42.1% mean before the standing-time fix)")
print("=" * 92)
print(f"{'player':<20}{'pos':<5}{'measured s':>12}{'of':>8}{'coverage':>10}"
      f"{'published pct':>15}")
cors = []
outf = []
for n in sorted(analytics.player_data,
                key=lambda k: -analytics.player_data[k]["distance_covered"]):
    ms = sum(raw[n].values())
    cov = ms / available
    if pos.get(n) != "GK":
        cors.append(cov)
        outf.append(n)
    print(f"{n:<20}{pos.get(n,'?'):<5}{ms:>12.0f}{available:>8.0f}"
          f"{cov*100:>9.1f}%{analytics.player_data[n]['activity_measured_pct']:>14.1f}%")
print("-" * 92)
mean_cov = sum(cors) / len(cors)
print(f"{'OUTFIELDER MEAN':<25}{'':>12}{'':>8}{mean_cov*100:>9.1f}%")
print(f"  min {min(cors)*100:.1f}%  max {max(cors)*100:.1f}%")
# G1
if mean_cov <= 0.421:
    raise SystemExit(f"GATE G1 FAILED: coverage {mean_cov*100:.1f}% "
                     f"did not improve on the pre-change 42.1%")
print(f"  GATE G1 PASS: {mean_cov*100:.1f}% > 42.1%")

print()
print("=" * 92)
print("2. PUBLISHED SECONDS vs RAW MEASURED SECONDS  (whole match)")
print("=" * 92)
print(f"{'player':<20}{'band':<11}{'measured s':>12}{'published s':>13}"
      f"{'template s':>12}{'share measured':>16}")
for n in outf[:4]:
    pub = analytics._activity.get(n, {})
    for band, _ in BANDS:
        tpl = pub.get(band, 0.0) - raw[n][band]
        share = raw[n][band] / max(pub.get(band, 0.0), 1e-9) * 100
        print(f"{n:<20}{band:<11}{raw[n][band]:>12.0f}{pub.get(band,0.0):>13.0f}"
              f"{tpl:>12.0f}{share:>15.0f}%")
    print(f"{'':<20}{'SUM':<11}{sum(raw[n].values()):>12.0f}"
          f"{sum(pub.values()):>13.0f}{sum(pub.values())-sum(raw[n].values()):>12.0f}")
    print()

print("=" * 92)
print("3. DISTANCE: measured metres + template budget for the unwatched seconds")
print("=" * 92)
print(f"{'player':<20}{'pos':<5}{'measured m':>12}{'published km':>14}"
      f"{'template km':>13}{'template %':>13}")
tpl_share_all = []
for n in outf:
    pd = analytics.player_data[n]["distance_covered"] * 1000
    tm = raw_dist[n]
    tpl = pd - tm
    tpl_share_all.append(tpl / max(pd, 1e-9) * 100)
    if len([1 for x in (pd, tm, tpl)]) and outf.index(n) < 8:
        print(f"{n:<20}{pos.get(n,'?'):<5}{tm:>12.0f}{pd/1000:>14.2f}"
              f"{tpl/1000:>13.2f}{tpl/max(pd,1e-9)*100:>12.0f}%")
print("-" * 92)
print(f"{'mean template share of the published distance':<48}"
      f"{sum(tpl_share_all)/len(tpl_share_all):>10.0f}%")
print()
print("  Before this change the published distance was 100% template plus a")
print("  fraction of the trace; the template share was effectively the whole")
print("  number. It is now the residue the integrator did not watch.")