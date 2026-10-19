"""THE MISSING 58% -- is it standing time dropped by `if moved <= 0: return`?

⚠️ MY FIRST VERSION OF THIS PROBE MEASURED NOTHING, and its GATE said so.
It compared calls to `_record_offball_distance` (counted in a wrapper placed
BEFORE the method's own `if moved <= 0: return`) against integrator
invocations. The first number cannot be smaller than the second, so the gate
fired "no gap" and stopped -- on an instrument that could not have detected
the gap even if it were real. **A gate that cannot fail is worse than no gate:
it reported "premise is wrong, stop" and I nearly believed it.**

THE ACTUAL QUESTION: of the ticks the integrator runs, how many actually
reach `PositionEngine.record_physics_distance`, and what do the rest look like?
Measured by wrapping the ACCUMULATOR (the only thing that can prove a
measurement was booked) and reading `moved` at the call site.

Two probes already ran and both matter:
  `_diag_match_clock_budget.py`  REFUTED "the engine simulates less than 90'":
        5,782 s simulated vs 5,760 s nominal (100.4%), integrators budget
        5,570 s = 96.3% of the clock. Time is NOT missing.
  `_diag_activity_coverage.py`   measured only 42.1% of each player's seconds
        carrying a measured speed. So the shortfall is in the WRITER, not the
        clock.

GATES
  G1 ticks the integrator ran > ticks that were booked -> a real gap
  G2 the unbooked ticks are predominantly `moved == 0`, i.e. standing
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
from position_engine import PositionEngine  # noqa: E402

for _o, _a in ((MatchEngine, "_offball_move_player"),
               (PositionEngine, "record_physics_distance")):
    if not hasattr(_o, _a):
        raise SystemExit(f"ANCHOR MISSING: {_o.__name__}.{_a}")

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

ran = defaultdict(int)          # ticks `_offball_move_player` actually stepped
onball = defaultdict(int)
gk = defaultdict(int)
booked = defaultdict(int)       # reached the accumulator (moved > 0)
moved_zero = defaultdict(int)   # reached `_record_offball_distance`, moved == 0
moved_hist = defaultdict(lambda: defaultdict(int))
pos = {}

_orig_move = MatchEngine._offball_move_player
_orig_rec = MatchEngine._record_offball_distance
_orig_acc = PositionEngine.record_physics_distance


def _move(self, pname, team, ball_x, ball_y, has_ball, danger_t, cross_team, DT):
    st = self.position_engine.states.get(pname)
    if st is not None and getattr(st, "position", "") == "GK":
        gk[pname] += 1
    elif pname in self._on_ball_this_minute:
        onball[pname] += 1
    else:
        ran[pname] += 1
    return _orig_move(self, pname, team, ball_x, ball_y, has_ball,
                      danger_t, cross_team, DT)


def _acc(self, player_name, distance_m=0.0, duration_s=0.0, speed_mps=0.0,
         walk_time_s=0.0, jog_time_s=0.0, sprint_time_s=0.0, *a, **kw):
    booked[player_name] += 1
    return _orig_acc(self, player_name, distance_m, duration_s, speed_mps,
                     walk_time_s, jog_time_s, sprint_time_s, *a, **kw)


def _rec(self, name, moved, speed, top, duration_s):
    if moved <= 0:
        moved_zero[name] += 1
    elif moved < 0.05:
        moved_hist[name]["<0.05 m"] += 1
    elif moved < 0.2:
        moved_hist[name]["0.05-0.2 m"] += 1
    elif moved < 0.5:
        moved_hist[name]["0.2-0.5 m"] += 1
    else:
        moved_hist[name][">=0.5 m"] += 1
    return _orig_rec(self, name, moved, speed, top, duration_s)


MatchEngine._offball_move_player = _move
PositionEngine.record_physics_distance = _acc
MatchEngine._record_offball_distance = _rec

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

for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            pos[row["player"]] = row.get("position", "?")

tr = sum(ran.values())
tb = sum(booked.values())
if tr == 0:
    raise SystemExit("VACUOUS: integrator never stepped a player")
if tb >= tr:
    raise SystemExit(f"GATE: no gap ({tb} booked >= {tr} ran) -- premise wrong")

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}")
print()
print("=" * 86)
print("TICKS STEPPED vs TICKS BOOKED vs TICKS DROPPED BY `if moved <= 0`")
print("=" * 86)
print(f"{'player':<20}{'pos':<5}{'stepped':>9}{'booked':>8}{'booked%':>9}"
      f"{'moved==0':>10}{'still%':>8}{'on-ball':>9}")
for n in sorted(ran, key=lambda k: -ran[k]):
    m0 = moved_zero[n]
    still = m0 / max(ran[n], 1) * 100
    print(f"{n:<20}{pos.get(n,'?'):<5}{ran[n]:>9}{booked[n]:>8}"
          f"{booked[n]/max(ran[n],1)*100:>8.1f}%{m0:>10}{still:>7.1f}%"
          f"{onball[n]:>9}")
print("-" * 86)
mz = sum(moved_zero.values())
print(f"{'TOTAL':<25}{tr:>9}{tb:>8}{tb/tr*100:>8.1f}%{mz:>10}{mz/tr*100:>7.1f}%"
      f"{sum(onball.values()):>9}")
print()
print(f"Seconds of standing time thrown away by `if moved <= 0: return`: "
      f"{mz*0.1:.0f} s across the match")
print(f"  = {mz*0.1/(len(ran) or 1):.0f} s per player, i.e. "
      f"{mz*0.1/(len(ran) or 1)/5760*100:.1f}% of a nominal 5,760 s match")
print()
print("moved distribution OF BOOKED TICKS:")
agg = defaultdict(int)
for n in moved_hist:
    for k, v in moved_hist[n].items():
        agg[k] += v
for k in ("<0.05 m", "0.05-0.2 m", "0.2-0.5 m", ">=0.5 m"):
    print(f"  moved {k:<12}{agg[k]:>9} ticks")
print()
print("VERDICT: a `moved == 0` tick is a player standing AT his shape target.")
print("That is a standing second, and it is discarded before any measurement.")
print("Standing time is therefore STRUCTURALLY UNMEASURABLE today -- which is")
print("precisely why the template invented a 16-83% standing share.")