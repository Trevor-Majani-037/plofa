"""Diagnose GK activity across multiple seeds."""
import json
import random
from datetime import date
from match_engine import MatchEngine, MatchConfig, MatchState, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder

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
    ("Pavel Renko", "GK", ["sweeper_keeper"], 31),
    ("Bart Kuipers", "CB", ["stopper_defender"], 28),
    ("Ciro Mancini", "CB", [], 26),
    ("Demi Adeola", "CDM", ["ball_winner", "regista"], 27),
    ("Finn Larsson", "CM", ["press_resistant"], 25),
    ("Kwame Asante", "CAM", ["playmaker", "creator"], 23),
    ("Bruno Reis", "LW", ["speedster", "counter_attacker"], 24),
    ("Nico Strauss", "ST", ["fox_in_box", "cold_blooded"], 27),
    ("Tariq El-Amin", "RW", ["dribbler"], 22),
    ("Jide Afolabi", "LB", [], 26),
    ("Lee Sung-jin", "RB", ["overlapping_fullback"], 28),
]

results = []
for seed in range(24):
    random.seed(seed)
    home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
    away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
    config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                         match_date=date(2026, 8, 16), matchday=1)
    hs = TeamProfile("Hartwell City", TeamStyle.ATTACKING, PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    as_ = TeamProfile("Thornfield United", TeamStyle.FLUID_COUNTER, PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hs, as_)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
    result = engine.simulate()

    home_gk_events = []
    away_gk_events = []
    for e in result.timeline:
        if e.player == "Pavel Renko":
            away_gk_events.append(e)
        if e.player == "Keano Walsh":
            home_gk_events.append(e)

    home_total = len(home_gk_events)
    away_total = len(away_gk_events)
    results.append({
        "seed": seed,
        "home_total": home_total,
        "away_total": away_total,
    })

with open("gk_diag_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done.")
print(f"Seeds with home GK events: {sum(1 for r in results if r['home_total'] > 0)}")
print(f"Seeds with away GK events: {sum(1 for r in results if r['away_total'] > 0)}")
print(f"Home GK event counts: {[r['home_total'] for r in results if r['home_total'] > 0]}")
print(f"Away GK event counts: {[r['away_total'] for r in results if r['away_total'] > 0]}")
