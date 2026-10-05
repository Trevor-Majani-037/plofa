import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from football_brain import FootballBrain
from brain_evolution import synthetic_fitness, evolve

# generate batched corpus
corpus = []
for i in range(1):
    rng = random.Random(42)
    brain_sensors = []
    from brain_evolution import random_game_state, _DummyPositionEngine, _coords_to_players
    from brain_sensors import extract_sensors
    for _ in range(400):
        st = random_game_state(rng, "GK")
        all_names = {}
        for idx_t, (tx, ty) in enumerate(st["teammates"]): all_names[f"t{idx_t}"] = (tx, ty)
        for idx_d, (dx, dy) in enumerate(st["defenders"]): all_names[f"d{idx_d}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)
        teammates = _coords_to_players(st["teammates"], "t")
        defenders = _coords_to_players(st["defenders"], "d")
        s = extract_sensors(None, st["x"], st["y"], teammates, defenders, pe, st["under_pressure"], st["attacks_right"], st["game_state"], st["minute"])
        brain_sensors.append(s)
    corpus.append(np.array(brain_sensors))

brain = FootballBrain.random(42)
new_f = synthetic_fitness(brain, "GK", batched_sensors=corpus[0])
print("New fitness scale component:", new_f)