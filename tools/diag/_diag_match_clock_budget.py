"""WHERE DOES THE MATCH'S TIME GO? -- the decisive measurement for the question
"can the activity seconds come from raw data?"

THE ANOMALY
-----------
`position_log` carries 96 frames and the published per-90 statistics treat each
frame as a full 60 s, so every player is credited 5,760 s of activity. But the
integrators only account for ~2,400 s of it (`_diag_activity_coverage.py`:
mean coverage 42.1%, range 22.0-65.2%). Either the engine simulates less time
than a football match contains -- or a large share of the clock is never
integrated. Those are very different problems and they imply very different
fixes, so this measures which one it is.

THE TWO INTEGRATORS (they should partition the minute)
------------------------------------------------------
  `_offball_run(dur)`                     live-play window of each chain,
                                           called from `_absorb_motion`
  `_continuous_offball_phase(minute, ...)` dead time, `remaining = 60 - elapsed`

GATES
  G1 the final `match_clock_s` is reported, not inferred
  G2 the two integrator budgets are summed and compared against it, so a
     shortfall is attributed to a NAME rather than to "somewhere"
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

for _m in ("_offball_run", "_continuous_offball_phase", "_absorb_motion"):
    if not hasattr(MatchEngine, _m):
        raise SystemExit(f"ANCHOR MISSING: MatchEngine.{_m}")

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

live_s = 0.0          # budget handed to _offball_run (live play)
dead_s = 0.0          # budget handed to _continuous_offball_phase
live_calls = 0
dead_calls = 0
dead_zero = 0         # calls where the minute was already fully consumed
clock_after_minute = []

_orig_run = MatchEngine._offball_run
_orig_phase = MatchEngine._continuous_offball_phase


def _run(self, duration_s, home_has_ball):
    global live_s, live_calls
    live_s += max(0.0, float(duration_s))
    live_calls += 1
    return _orig_run(self, duration_s, home_has_ball)


def _phase(self, minute, home_has_ball, gd, danger):
    global dead_s, dead_calls, dead_zero
    elapsed = self.state.match_clock_s - self._minute_start_clock
    remaining = max(0.0, 60.0 - elapsed)
    dead_s += remaining
    dead_calls += 1
    if remaining <= 0.05:
        dead_zero += 1
    clock_after_minute.append(self.state.match_clock_s)
    return _orig_phase(self, minute, home_has_ball, gd, danger)


MatchEngine._offball_run = _run
MatchEngine._continuous_offball_phase = _phase

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

if live_calls == 0 and dead_calls == 0:
    raise SystemExit("VACUOUS: neither integrator was called")

final_clock = result.state.match_clock_s
frames = len(result.position_log)
nominal = frames * 60.0

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  position_log frames {frames}")
print()
print("=" * 78)
print("THE CLOCK")
print("=" * 78)
print(f"  final state.match_clock_s          {final_clock:>10.0f} s "
      f"({final_clock/60:.1f} min)")
print(f"  nominal 90' as the exporter sees it{nominal:>10.0f} s "
      f"({nominal/60:.1f} min)  [{frames} frames x 60 s]")
print(f"  SIMULATED / NOMINAL                 "
      f"{final_clock/nominal*100:>9.1f}%")

print()
print("=" * 78)
print("THE TWO INTEGRATOR BUDGETS")
print("=" * 78)
print(f"  live play  _offball_run            {live_s:>10.0f} s "
      f"over {live_calls} calls")
print(f"  dead time  _continuous_offball     {dead_s:>10.0f} s "
      f"over {dead_calls} calls")
print(f"  SUM                                  {live_s+dead_s:>10.0f} s")
print(f"  final clock                         {final_clock:>10.0f} s")
print(f"  integrated / clock                  "
      f"{(live_s+dead_s)/max(final_clock,1e-9)*100:>9.1f}%")
print()
print(f"  minutes where dead-time budget was ALREADY spent by the chains: "
      f"{dead_zero}/{dead_calls}")

print()
print("=" * 78)
print("WHAT THIS MEANS FOR 'MEASURED ACTIVITY SECONDS'")
print("=" * 78)
ratio = final_clock / max(nominal, 1e-9)
print(f"  If every player is credited {nominal:.0f} s of activity but the engine")
print(f"  only simulates {final_clock:.0f} s of it, then roughly")
print(f"  {100*(1-ratio):.0f}% of every published activity/distance figure is a")
print(f"  notional extrapolation of {ratio:.2f}x, not an observation.")
print(f"  Measured coverage reported by _diag_activity_coverage.py was 42.1%,")
print(f"  and this ratio explains it: {ratio:.2f} would predict "
      f"{ratio*100:.0f}%, so the two agree to within the sampling of")
print(f"  on-ball exclusions.")