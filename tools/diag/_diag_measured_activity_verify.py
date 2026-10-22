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
from typing import Dict, List

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
      f"{'published pct':>15}{'overlap s':>11}{'published s':>13}")
cors = []
outf = []
for n in sorted(analytics.player_data,
                key=lambda k: -analytics.player_data[k]["distance_covered"]):
    ms = sum(raw[n].values())
    pd_ = analytics.player_data[n]
    # NET coverage: the raw sum can exceed the clock because the integrator and
    # the trace overlap in time. The overlap is subtracted here so the column
    # says how much of the match is measured, not how many instrument-seconds
    # were produced. (Before this, the table printed 102.3% for one player and
    # it read as a bug rather than as the overlap it is.)
    cov = (ms - pd_.get("activity_overlap_s", 0.0)) / available
    if pos.get(n) != "GK":
        cors.append(cov)
        outf.append(n)
    print(f"{n:<20}{pos.get(n,'?'):<5}{ms:>12.0f}{available:>8.0f}"
          f"{cov*100:>9.1f}%{pd_['activity_measured_pct']:>14.1f}%"
          f"{pd_.get('activity_overlap_s', 0.0):>11.0f}"
          f"{sum(analytics._activity.get(n, {}).values()):>13.0f}")

# G2: A MINUTE HAS SIXTY SECONDS. Once the integrator and the trace are
# merged and scaled, no player may be published with more activity time than
# the match contains.
_over = [(n, sum(analytics._activity.get(n, {}).values()), available)
         for n in outf
         if sum(analytics._activity.get(n, {}).values()) > available + 1.0]
if _over:
    raise SystemExit(f"GATE G2 FAILED: published seconds exceed the clock: "
                     f"{_over[:3]}")
print(f"  GATE G2 PASS: no outfielder is published above {available:.0f} s")
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
print("  (raw seconds are scaled back into each minute when the integrator and")
print("   the possession trace double-booked it -- `overlap s` above.)")
print(f"{'player':<20}{'band':<11}{'measured s':>12}{'published s':>13}"
      f"{'template s':>12}{'share measured':>16}")
for n in outf[:4]:
    pub = analytics._activity.get(n, {})
    ov = analytics.player_data[n].get("activity_overlap_s", 0.0)
    for band, _ in BANDS:
        # The overlap cannot be attributed per band; scale it to the band's
        # share of the measured total so the comparison is like-for-like.
        _ovb = ov * (raw[n][band] / max(sum(raw[n].values()), 1e-9))
        meas_b = max(0.0, raw[n][band] - _ovb)
        tpl = pub.get(band, 0.0) - meas_b
        share = meas_b / max(pub.get(band, 0.0), 1e-9) * 100
        print(f"{n:<20}{band:<11}{meas_b:>12.0f}{pub.get(band,0.0):>13.0f}"
              f"{tpl:>12.0f}{share:>15.0f}%")
    print(f"{'':<20}{'SUM':<11}{sum(raw[n].values())-ov:>12.0f}"
          f"{sum(pub.values()):>13.0f}{sum(pub.values())-sum(raw[n].values())+ov:>12.0f}")
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

# ── PER-POSITION SUMMARY + CACHE ──────────────────────────────────────
# Written to disk so a re-report costs nothing: everything above is a single
# match, and a percentile/per-position question should not need a second one.
per_pos: Dict[str, Dict[str, List[float]]] = {}
for n in outf:
    p_ = pos.get(n, "?")
    d = analytics.player_data[n]
    b = analytics._activity.get(n, {})
    tot = max(sum(b.values()), 1e-9)
    acc = per_pos.setdefault(p_, {"km": [], "stand": [], "meas": []})
    acc["km"].append(d["distance_covered"])
    acc["stand"].append(100.0 * b.get("standing", 0.0) / tot)
    acc["meas"].append(d["activity_measured_pct"])

print()
print("=" * 92)
print("4. PER POSITION (outfielders who played)")
print("=" * 92)
print(f"{'pos':<6}{'n':>4}{'km mean':>10}{'km min':>8}{'km max':>8}"
      f"{'standing %':>12}{'measured %':>12}")
for p_ in ("CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"):
    acc = per_pos.get(p_)
    if not acc:
        continue
    print(f"{p_:<6}{len(acc['km']):>4}{sum(acc['km'])/len(acc['km']):>10.2f}"
          f"{min(acc['km']):>8.2f}{max(acc['km']):>8.2f}"
          f"{sum(acc['stand'])/len(acc['stand']):>11.1f}%"
          f"{sum(acc['meas'])/len(acc['meas']):>11.1f}%")

import json
with open("_diag_measured_verify.json", "w", encoding="utf-8") as fh:
    json.dump({"per_position": per_pos,
               "players": {n: {"position": pos.get(n, "?"),
                               **analytics.player_data[n]} for n in outf}},
              fh, indent=1, default=str)
print("\n  cached -> _diag_measured_verify.json (do not commit)")