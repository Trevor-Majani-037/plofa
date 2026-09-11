import random
from collections import defaultdict
from datetime import date

from player_dna import SquadBuilder
from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle,
    PlayingStyle, Intensity, EventType,
)

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Kofi Mensah", "CAM", ["creator"], 24),
    ("Adri Vela", "LW", ["dribbler"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher"], 29),
    ("Percy", "RW", ["grand_dribbler"], 24),
]

AWAY_STARTERS = [
    ("A GK", "GK", ["sweeper_keeper"], 27),
    ("A LB", "LB", ["defensive_fullback"], 25),
    ("A CB1", "CB", ["no_nonsense_cb"], 27),
    ("A CB2", "CB", ["stopper_defender"], 29),
    ("A RB", "RB", ["defensive_fullback"], 25),
    ("A CDM", "CDM", ["ball_winning_mid"], 26),
    ("A CM1", "CM", ["box_box"], 26),
    ("A CAM", "CAM", ["creator"], 23),
    ("A LW", "LW", ["dribbler"], 22),
    ("A ST", "ST", ["clinical_finisher"], 28),
    ("A RW", "RW", ["crosser"], 23),
]

random.seed(7)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
away = SquadBuilder.build("Away", AWAY_STARTERS)
config = MatchConfig(home_team="Hartwell City", away_team="Away",
                     match_date=date(2026, 8, 16), matchday=1)
hs = TeamProfile("Hartwell City", TeamStyle.ATTACKING,
                 PlayingStyle.HIGH_PRESS, Intensity.HIGH)
as_ = TeamProfile("Away", TeamStyle.FLUID_COUNTER,
                  PlayingStyle.COUNTER, Intensity.MEDIUM)
engine = MatchEngine(config, hs, as_)
engine.quiet = True
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Away", away["starters"], away["substitutes"])
res = engine.simulate()

evs = res.timeline
H, A = "Hartwell City", "Away"


def team_count(event_type, team=None, outcome=None):
    n = 0
    for e in evs:
        if e.event_type is event_type:
            if team and e.team != team:
                continue
            if outcome is not None and e.outcome is not outcome:
                continue
            n += 1
    return n


def tackle_detail():
    by = defaultdict(lambda: defaultdict(int))  # technique -> outcome counts
    per_player = defaultdict(lambda: defaultdict(int))
    for e in evs:
        if e.event_type in (EventType.TACKLE_WON, EventType.TACKLE_LOST):
            tech = (e.metadata or {}).get("technique") or "press?"
            if (e.metadata or {}).get("from_pressure"):
                tech = "press"
            by[tech][e.event_type.name] += 1
            per_player[e.player][e.event_type.name] += 1
    return by, per_player


by, per_player = tackle_detail()

print("=" * 62)
print(f"{H}  {res.home_goals} - {res.away_goals}  {A}")
print(f"xG:  {res.home_xg} - {res.away_xg}")
print(f"possession:  {res.home_possession_pct}% - {res.away_possession_pct}%")
print("-" * 62)
print(f"Shots (on/off/blkd):  H {team_count(EventType.SHOT_ON_TARGET, H)}+{team_count(EventType.SHOT_OFF_TARGET, H)}/{team_count(EventType.SHOT_BLOCKED, H)}   "
      f"A {team_count(EventType.SHOT_ON_TARGET, A)}+{team_count(EventType.SHOT_OFF_TARGET, A)}/{team_count(EventType.SHOT_BLOCKED, A)}")
print(f"Blocks:               H {team_count(EventType.BLOCK, H)}  A {team_count(EventType.BLOCK, A)}")
print(f"Through-balls (A/C):  H {team_count(EventType.THROUGH_BALL, H)}/{team_count(EventType.THROUGH_BALL, H, True)}   "
      f"A {team_count(EventType.THROUGH_BALL, A)}/{team_count(EventType.THROUGH_BALL, A, True)}")
print(f"Tackles (W/L):        H {team_count(EventType.TACKLE_WON, H)}/{team_count(EventType.TACKLE_LOST, H)}   "
      f"A {team_count(EventType.TACKLE_WON, A)}/{team_count(EventType.TACKLE_LOST, A)}")
print(f"Fouls:                H {team_count(EventType.FOUL_COMMITTED, H)}  A {team_count(EventType.FOUL_COMMITTED, A)}")
print(f"Yellow/Red:           H {team_count(EventType.YELLOW_CARD, H)}/{team_count(EventType.RED_CARD, H)}   "
      f"A {team_count(EventType.YELLOW_CARD, A)}/{team_count(EventType.RED_CARD, A)}")
print("-" * 62)
print("Tackle technique:")
for tech, m in sorted(by.items()):
    print(f"   {tech:<10} won={m.get('TACKLE_WON', 0)}  lost={m.get('TACKLE_LOST', 0)}")
print("Top tacklers:")
for name, m in sorted(per_player.items(), key=lambda kv: -(kv[1]["TACKLE_WON"] + kv[1]["TACKLE_LOST"]))[:5]:
    print(f"   {name:<20} won={m.get('TACKLE_WON', 0)} lost={m.get('TACKLE_LOST', 0)}")