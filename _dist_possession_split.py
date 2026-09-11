"""Measure off-ball drift distance by position, split by in/out of possession."""
import random
from collections import defaultdict
from datetime import date

from position_engine import PositionEngine
from player_dna import SquadBuilder
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Jonas Keller", "CM", ["engine"], 25),
    ("Adri Vela", "LW", ["dribbler"], 22),
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

drift_split = defaultdict(lambda: [0.0, 0.0])   # pos -> [in_poss, out_poss]
touch_split = defaultdict(lambda: [0.0, 0.0])
_poss = {}

_orig_drift = PositionEngine.drift_minute
_orig_accum = PositionEngine.accumulate_drift_from_snapshot
_orig_touch = PositionEngine.record_touch


def _drift(self, team_name, profile, phase, **kw):
    _poss[team_name] = bool(kw.get("in_possession", False))
    return _orig_drift(self, team_name, profile, phase, **kw)


def _accum(self, team_name, before):
    pre = {n: s.minute_drift_distance for n, s in self.states.items()}
    _orig_accum(self, team_name, before)
    idx = 0 if _poss.get(team_name) else 1
    for n, s in self.states.items():
        d = s.minute_drift_distance - pre.get(n, 0.0)
        if d > 0:
            drift_split[s.position][idx] += d


def _touch(self, name, x, y, *a, **kw):
    s = self.states.get(name)
    pre = s.minute_touch_distance if s else 0.0
    _orig_touch(self, name, x, y, *a, **kw)
    if s is None:
        return
    idx = 0 if _poss.get(s.team) else 1
    touch_split[s.position][idx] += max(0.0, s.minute_touch_distance - pre)


PositionEngine.drift_minute = _drift
PositionEngine.accumulate_drift_from_snapshot = _accum
PositionEngine.record_touch = _touch

random.seed(11)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                     match_date=date(2026, 8, 22), matchday=2)
hs = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION, Intensity.MEDIUM)
as_ = TeamProfile("Thornfield United", TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
engine = MatchEngine(config, hs, as_)
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
result = engine.simulate()

print(f"\nscore: {result.state.home_goals}-{result.state.away_goals}")
print(f"\n{'pos':<5}{'off-ball IN km':>15}{'off-ball OUT km':>17}{'IN/OUT':>8}{'touches IN km':>15}")
order = ["CDM", "CM", "CAM", "LB", "RB", "CB", "LW", "RW", "ST", "GK"]
for pos in order:
    din, dout = drift_split[pos][0] / 1000, drift_split[pos][1] / 1000
    tin, tout = touch_split[pos][0] / 1000, touch_split[pos][1] / 1000
    ratio = f"{din/dout:.2f}" if dout > 0.01 else "  -"
    print(f"{pos:<5}{din:>15.2f}{dout:>17.2f}{ratio:>8}{tin:>15.2f}")

tin_all = sum(v[0] for v in touch_split.values()) / 1000
tout_all = sum(v[1] for v in touch_split.values()) / 1000
print(f"\nall touch distance: in-poss {tin_all:.2f} km vs out {tout_all:.2f} km")
