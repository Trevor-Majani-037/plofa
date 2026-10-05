import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from football_brain import FootballBrain, OUTPUT_SIZE
from decision_brain import PlayerIntent
from brain_sensors import extract_sensors

# Quick mock of original synthetic_fitness to test parity
def orig_random_game_state(rng, position):
    x = rng.uniform(40, 95)
    y = rng.uniform(2, 66)
    attacks_right = rng.random() < 0.5
    under_pressure = rng.random() < 0.4
    minute = rng.uniform(0, 90)
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1", "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = type("GS", (), {"name": rng.choice(game_state_names)})()
    n_teammates = rng.randint(3, 8)
    teammates = []
    for _ in range(n_teammates):
        tx = x + (rng.uniform(-5, 25) if attacks_right else rng.uniform(-25, 5))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, y + rng.uniform(-18, 18)))
        teammates.append((tx, ty))
    n_defenders = rng.randint(1, 6)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, x + rng.uniform(-8, 8)))
        dy = max(0, min(68, y + rng.uniform(-12, 12)))
        defenders.append((dx, dy))
    return {"x": x, "y": y, "attacks_right": attacks_right, "under_pressure": under_pressure, "minute": minute, "game_state": game_state, "teammates": teammates, "defenders": defenders}

class GS:
    __slots__ = ('name',)
    def __init__(self, name):
        self.name = name

def new_random_game_state(rng, position):
    x = rng.uniform(40, 95)
    y = rng.uniform(2, 66)
    attacks_right = rng.random() < 0.5
    under_pressure = rng.random() < 0.4
    minute = rng.uniform(0, 90)
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1", "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = GS(rng.choice(game_state_names))
    n_teammates = rng.randint(3, 8)
    teammates = []
    for _ in range(n_teammates):
        tx = x + (rng.uniform(-5, 25) if attacks_right else rng.uniform(-25, 5))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, y + rng.uniform(-18, 18)))
        teammates.append((tx, ty))
    n_defenders = rng.randint(1, 6)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, x + rng.uniform(-8, 8)))
        dy = max(0, min(68, y + rng.uniform(-12, 12)))
        defenders.append((dx, dy))
    return {"x": x, "y": y, "attacks_right": attacks_right, "under_pressure": under_pressure, "minute": minute, "game_state": game_state, "teammates": teammates, "defenders": defenders}

rng = random.Random(42)
for _ in range(10):
    r1 = orig_random_game_state(rng, "ST")
rng = random.Random(42)
for _ in range(10):
    r2 = new_random_game_state(rng, "ST")

print(f"Parity check: {r1['x'] == r2['x'] and r1['game_state'].name == r2['game_state'].name}")