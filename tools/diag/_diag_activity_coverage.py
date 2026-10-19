"""CAN THE ACTIVITY SECONDS COME FROM RAW DATA? -- measure coverage first.

THE QUESTION
------------
`standing_seconds` .. `sprinting_seconds` are currently a per-position TABLE
(`BASE_MINUTE_ACTIVITY`) scaled by work_rate, perturbed by telemetry, then
renormalised to 60 s. The user wants them measured. That is only possible if
the engine holds a real speed for most of a player's 5,400 match seconds.

WHAT ALREADY EXISTS
-------------------
`MatchEngine._record_offball_distance(pname, moved, spd, tgt, DT)` is called
ONCE PER 10 Hz TICK for every off-ball player, with a MEASURED `spd`. So the
raw speed stream exists. `PositionEngine.record_physics_distance` buckets the
tick's `duration_s` into walk / jog / sprint by speed -- and has NO STANDING
BAND, which is why a standing second is unrepresentable today.

THE TWO GATES THIS PROBE MUST PASS BEFORE ANY REDESIGN
-----------------------------------------------------
  G1 COVERAGE: what fraction of a player's 5,400 match seconds actually has a
     measured speed? If it is 40%, publishing measured seconds means publishing
     40% of the truth and guessing the other 60% -- which is what we are trying
     to stop doing.
  G2 BANDS: what does the measured speed distribution actually look like? A
     band set has to be justified by the data, not chosen to make a total come
     out right.

Attribution is by CALL SITE (the `_diag_touch_sites.py` frame join), not by
inferring intent.
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

off_s = defaultdict(float)        # off-ball integrator seconds
off_band = defaultdict(lambda: defaultdict(float))   # off-ball seconds by speed
ep_s = defaultdict(float)         # episode trace band seconds
ep_band = defaultdict(lambda: defaultdict(float))
slot_s = defaultdict(float)        # set-piece run-in seconds
off_ticks = defaultdict(int)

_orig = PositionEngine.record_physics_distance


def _rec(self, player_name, distance_m=0.0, duration_s=0.0, speed_mps=0.0,
         walk_time_s=0.0, jog_time_s=0.0, sprint_time_s=0.0, *a, **kw):
    f = inspect.currentframe().f_back
    site = f.f_code.co_name
    dur = float(duration_s)
    if site == "_record_offball_distance":
        off_s[player_name] += dur
        off_ticks[player_name] += 1
        v = float(speed_mps)
        if v < 0.5:
            k = "<0.5 still"
        elif v < 2.0:
            k = "0.5-2 walk"
        elif v < 4.2:
            k = "2-4.2 jog"
        elif v < 5.5:
            k = "4.2-5.5 mod"
        elif v < 7.0:
            k = "5.5-7 high"
        else:
            k = "7+ sprint"
        off_band[player_name][k] += dur
    elif site == "_accumulate_physics_stats":
        ep_s[player_name] += walk_time_s + jog_time_s + sprint_time_s
        ep_band[player_name]["walk"] += walk_time_s
        ep_band[player_name]["jog"] += jog_time_s
        ep_band[player_name]["sprint"] += sprint_time_s
    elif site == "advance_to_slots":
        slot_s[player_name] += dur
    return _orig(self, player_name, distance_m, duration_s, speed_mps,
                 walk_time_s, jog_time_s, sprint_time_s, *a, **kw)


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

frames = len(result.position_log)
available = frames * 60.0

if sum(off_s.values()) <= 0:
    raise SystemExit("VACUOUS: the off-ball integrator never recorded a speed")

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  frames {frames}  "
      f"-> {available:.0f} s per player of wall clock")

print()
print("=" * 80)
print("G1 COVERAGE -- seconds with a MEASURED speed, per player")
print("=" * 80)
print(f"{'player':<20}{'pos':<5}{'available':>11}{'offball':>10}{'episode':>10}"
      f"{'run-in':>8}{'observed':>10}{'coverage':>10}")
pos = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            pos[row["player"]] = row.get("position", "?")
covs = []
for n in sorted(off_s, key=lambda k: -off_s[k]):
    if pos.get(n) == "GK":
        continue
    obs = off_s[n] + ep_s[n] + slot_s[n]
    cov = obs / available
    covs.append(cov)
    print(f"{n:<20}{pos.get(n,'?'):<5}{available:>11.0f}{off_s[n]:>10.0f}"
          f"{ep_s[n]:>10.0f}{slot_s[n]:>8.0f}{obs:>10.0f}{cov*100:>9.1f}%")
print(f"{'MEAN':<25}{'':>11}{'':>10}{'':>10}{'':>8}"
      f"{'':>10}{sum(covs)/len(covs)*100:>9.1f}%")
print(f"  -> a fully-measured 90' needs 100%. Off-ball is the missing majority.")

print()
print("=" * 80)
print("G2 BANDS -- what the measured off-ball speed stream actually looks like")
print("=" * 80)
agg = defaultdict(float)
for n in off_band:
    for k, v in off_band[n].items():
        agg[k] += v
tot = sum(agg.values())
for k in ("<0.5 still", "0.5-2 walk", "2-4.2 jog", "4.2-5.5 mod",
          "5.5-7 high", "7+ sprint"):
    print(f"  {k:<16}{agg[k]:>10.0f} s{agg[k]/tot*100:>8.1f}%"
          f"{agg[k]/max(tot,1e-9)*available:>12.0f} s of a 5400 s match")
print()
print("episode trace band split (mean per outfielder):")
outf = [n for n in ep_band if pos.get(n) not in ("GK",) and ep_s[n] > 0]
if outf:
    for k in ("walk", "jog", "sprint"):
        m = sum(ep_band[n][k] for n in outf) / len(outf)
        print(f"  {k:<10}{m:>8.0f} s")
print()
print("NOTE: off-ball seconds exist only for players NOT on the ball this tick")
print("(`_offball_move_player` returns early for them). The episode trace is the")
print("on-ball half. Neither alone covers a match -- hence the coverage gap.")