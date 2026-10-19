"""DISTANCE AUDIT PART 3 -- close the decomposition.

PART 1 decomposed the number as:
    REPORTED = DNA template + 0.45 x physics
PART 2 found writer (b) (the PossessionEpisode race trace) was booking 405 km
of unphysical geometry, and PART 3's fix capped it at physical reach, which
dropped writer (b) 405 -> 74 km and REPORTED 364 -> 290 km.

BUT the derived `template` column MOVED 6.77 -> 10.16 km/player across that
fix, and `baseline_dist` is computed in opta_analytics BEFORE any physics term
touches it, from a per-position DNA table that cannot change when a distance
ceiling is added. So either the derivation is wrong or there is a THIRD writer
into `state.physics_distance_m` that this probe never instrumented.

This probe does not assume either. It attributes every `record_physics_distance`
call to the CODE THAT MADE IT (the caller frame's function name -- the
`_diag_touch_sites.py` join AGENTS.md recommends over inferring intent), and it
recomputes the DNA template independently from the published tables.

GATES: the wrapper must fire (a non-zero call count) and the two independently
computed quantities must bracket the reported total.
"""
import inspect
import random
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, ".")

from match_engine import (  # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder  # noqa: E402
from position_engine import PositionEngine  # noqa: E402
import opta_analytics  # noqa: E402

if not hasattr(PositionEngine, "record_physics_distance"):
    raise SystemExit("ANCHOR MISSING: PositionEngine.record_physics_distance")

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

writer_m = defaultdict(lambda: defaultdict(float))
writer_n = defaultdict(lambda: defaultdict(int))

_orig = PositionEngine.record_physics_distance


def _rec(self, player_name, distance_m=0.0, *a, **kw):
    # Attribute to the CODE that called us, not to a guess about intent.
    f = inspect.currentframe().f_back
    site = "?"
    for _ in range(6):
        if f is None:
            break
        site = f"{f.f_code.co_filename.split(chr(92))[-1]}:{f.f_lineno} {f.f_code.co_name}"
        break
    writer_m[player_name][site] += float(distance_m)
    writer_n[player_name][site] += 1
    return _orig(self, player_name, distance_m, *a, **kw)


PositionEngine.record_physics_distance = _rec

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

total_calls = sum(sum(v.values()) for v in writer_n.values())
if total_calls == 0:
    raise SystemExit("VACUOUS: record_physics_distance never fired")

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  frames {len(result.position_log)}")
print(f"\nWRITERS into state.physics_distance_m -- attributed by CALL SITE:")
sites = sorted({s for p in writer_m for s in writer_m[p]},
               key=lambda s: -sum(writer_m[p][s] for p in writer_m))
grand = 0.0
for s in sites:
    km = sum(writer_m[p][s] for p in writer_m) / 1000
    calls = sum(writer_n[p][s] for p in writer_n)
    grand += km
    print(f"  {km:>9.2f} km  {calls:>8} calls  {s}")
print(f"  {'-'*9}  --------")
print(f"  {grand:>9.2f} km  {total_calls:>8} calls  TOTAL")

# Independently recompute the DNA template from the published tables.
SPEEDS = opta_analytics.SPEEDS
BASE = opta_analytics.BASE_MINUTE_ACTIVITY
DEFAULT = opta_analytics.DEFAULT_ACTIVITY
frames_n = defaultdict(int)
pos = {}
for frame in result.position_log:
    for side in ("home", "away"):
        for row in frame.get(side, []):
            frames_n[row["player"]] += 1
            pos[row["player"]] = row.get("position", "CM")

tmpl = defaultdict(float)
for name, obj in analytics.player_objs.items():
    p = analytics.player_meta.get(name, {}).get("position", "CM")
    base = BASE.get(p, DEFAULT)
    wr = getattr(getattr(obj, "dna", None), "mental", None)
    w = (getattr(wr, "work_rate", 50.0) / 100.0) if wr is not None else 0.5
    effort = 0.80 + 0.40 * w
    stand, walk, jog, run, sprint = base
    jog *= effort
    run = min(0.35, run * effort)
    sprint = min(0.10, sprint * effort)
    per_min = (stand * 60 * SPEEDS["standing"] + walk * 60 * SPEEDS["walking"]
               + jog * 60 * SPEEDS["jogging"] + run * 60 * SPEEDS["running"]
               + sprint * 60 * SPEEDS["sprinting"])
    tmpl[name] = per_min * frames_n.get(name, 0)

print()
print(f"{'player':<20}{'pos':<5}{'REPORTED':>9}{'template':>9}{'0.45*phys':>10}"
      f"{'0.45*physLOG':>13}{'delta':>8}")
print("-" * 74)
rep = tm = pl = 0.0
for name in sorted(analytics.player_data,
                   key=lambda k: -analytics.player_data[k]["distance_covered"]):
    d = analytics.player_data[name]["distance_covered"]
    raw = analytics._distances.get(name, 0.0)
    phys = sum(writer_m[name].values())
    phys_log = sum(float(r.get("physics_distance_m", 0.0))
                   for f in result.position_log for side in ("home", "away")
                   for r in f.get(side, []) if r["player"] == name)
    print(f"{name:<20}{pos.get(name,'?'):<5}{d:>9.2f}{tmpl[name]/1000:>9.2f}"
          f"{0.45*phys/1000:>10.2f}{0.45*phys_log/1000:>13.2f}"
          f"{(raw-tmpl[name]-0.45*phys_log)/1000:>8.3f}")
    rep += d
    tm += tmpl[name] / 1000
    pl += 0.45 * phys_log / 1000
print("-" * 74)
print(f"{'TOTAL':<25}{rep:>9.2f}{tm:>9.2f}{'':>10}{pl:>13.2f}{rep-tm-pl:>8.3f}")
print()
print(f"REPORTED {rep:.2f} km | template {tm:.2f} | 0.45*physics {pl:.2f} | "
      f"unexplained {rep-tm-pl:+.3f} km")
print(f"mean per player: template {tm/len(analytics.player_data):.2f} km  "
      f"REPORTED {rep/len(analytics.player_data):.2f} km")
outf = [n for n in analytics.player_data if pos.get(n) == "GK"]
outf_km = [analytics.player_data[n]["distance_covered"] for n in outf]
out_km = [analytics.player_data[n]["distance_covered"] for n in analytics.player_data
          if pos.get(n) != "GK"]
print(f"\noutfielders n={len(out_km)}: min {min(out_km):.2f}  "
      f"mean {sum(out_km)/len(out_km):.2f}  max {max(out_km):.2f} km   "
      f"(real band 10.5-13.0)")
print(f"keepers     n={len(outf_km)}: {[round(x,2) for x in outf_km]}  "
      f"(real ~4-6 km)")