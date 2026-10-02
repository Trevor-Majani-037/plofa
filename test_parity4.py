import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')

from brain_evolution import random_game_state, random_game_state_scoring, _DummyPositionEngine, _coords_to_players, GameState
from brain_sensors import extract_sensors

# 1. Old state generation
def old_gen():
    rng = random.Random(42)
    s_list = []
    for _ in range(5):
        st = random_game_state(rng, "GK")
        all_names = {}
        for i, (tx, ty) in enumerate(st["teammates"]): all_names[f"t{i}"] = (tx, ty)
        for i, (dx, dy) in enumerate(st["defenders"]): all_names[f"d{i}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)
        teammates = _coords_to_players(st["teammates"], "t")
        defenders = _coords_to_players(st["defenders"], "d")
        s = extract_sensors(None, st["x"], st["y"], teammates, defenders, pe, st["under_pressure"], st["attacks_right"], st["game_state"], st["minute"])
        s_list.append(s)
    return s_list

# 2. Hoisted state generation
def new_gen():
    rng = random.Random(42)
    s_list = []
    for _ in range(5):
        if False: pass
        else:
            st = random_game_state(rng, "GK")

        t_coords = st["teammates"]
        d_coords = st["defenders"]
        all_names = {}
        for idx_t, (tx, ty) in enumerate(t_coords):
            all_names[f"t{idx_t}"] = (tx, ty)
        for idx_d, (dx, dy) in enumerate(d_coords):
            all_names[f"d{idx_d}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)

        teammates = _coords_to_players(t_coords, "t")
        defenders = _coords_to_players(d_coords, "d")

        s = extract_sensors(
            None, st["x"], st["y"], teammates, defenders, pe,
            st["under_pressure"], st["attacks_right"], st["game_state"],
            st["minute"],
        )
        s_list.append(s)
    return s_list

o = old_gen()
n = new_gen()
for i in range(len(o)):
    print(f"Match {i}:", np.allclose(o[i], n[i]))