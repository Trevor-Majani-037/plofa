import random
from datetime import date

from player_dna import SquadBuilder
from match_engine import (MatchEngine, MatchConfig, TeamProfile, TeamStyle,
                          PlayingStyle, Intensity)
from sequence_engine import SequenceTracker

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

random.seed(7)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
away = SquadBuilder.build("Away", [("P0", "GK", [], 25)] +
                          [(f"P{i}", "CB", [], 25) for i in range(1, 11)])
config = MatchConfig(home_team="Hartwell City", away_team="Away",
                     match_date=date(2026, 8, 16), matchday=1)
hs = TeamProfile("Hartwell City", TeamStyle.ATTACKING, PlayingStyle.HIGH_PRESS, Intensity.HIGH)
as_ = TeamProfile("Away", TeamStyle.FLUID_COUNTER, PlayingStyle.COUNTER, Intensity.MEDIUM)
engine = MatchEngine(config, hs, as_)
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Away", away["starters"], away["substitutes"])
result = engine.simulate()

st = SequenceTracker("Hartwell City", "Away", result.timeline)
st.compute_metrics()

print("\n=== AWAY SHOT-ENDING SEQUENCES (raw events) ===")
for s in st.team_sequences("Away"):
    if not s.ends_in_shot:
        continue
    print(f"\n--- {s.shot.event_type.name} {s.shot.minute}' start={s.started_by.name} ---")
    for e in s.events:
        print(f"  {e.event_type.name:<22} {e.player:<12} loc=({e.location_x},{e.location_y}) "
              f"end=({e.end_x},{e.end_y}) outcome={e.outcome}")
    print(f"  forward_movement_ratio={s.forward_movement_ratio}")
