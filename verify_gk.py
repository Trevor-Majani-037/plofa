"""Quick GK pass activity probe for Keano Walsh and Pavel Renko."""
import random
from datetime import date
from match_engine import MatchEngine, MatchConfig, MatchState, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from exporter import StatAccumulator

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


def build_match(seed=1):
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
    return engine, home, away


def gk_pass_count(result, player_name):
    count = 0
    for e in result.timeline:
        if e.player == player_name and e.event_type.value in ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL"):
            count += 1
    return count


seeds = list(range(24))
keano_counts = []
renko_counts = []

for seed in seeds:
    engine, home, away = build_match(seed)
    result = engine.simulate()
    all_players = {"Hartwell City": home, "Thornfield United": away}
    acc = StatAccumulator(result, all_players)
    
    keano_passes = acc.stats.get("Keano Walsh", {}).get("passes_completed", 0)
    renko_passes = acc.stats.get("Pavel Renko", {}).get("passes_completed", 0)
    
    keano_counts.append(keano_passes)
    renko_counts.append(renko_passes)
    
    print(f"seed={seed:2d}: Keano={keano_passes:3d}, Renko={renko_passes:3d}")

print()
print(f"Keano Walsh: min={min(keano_counts)}, max={max(keano_counts)}, avg={sum(keano_counts)/len(keano_counts):.1f}")
print(f"Pavel Renko: min={min(renko_counts)}, max={max(renko_counts)}, avg={sum(renko_counts)/len(renko_counts):.1f}")

keano_in_range = 15 <= sum(keano_counts)/len(keano_counts) <= 30
renko_in_range = 9 <= sum(renko_counts)/len(renko_counts) <= 30
print(f"Keano in 15-30: {keano_in_range}")
print(f"Renko in 9-30: {renko_in_range}")
