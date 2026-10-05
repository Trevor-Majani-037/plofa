import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from football_brain import FootballBrain, OUTPUT_SIZE
from decision_brain import PlayerIntent
from brain_sensors import extract_sensors
from brain_evolution import _context_reward, random_game_state, random_game_state_scoring, _DummyPositionEngine, _coords_to_players, _INTENT_BY_IDX

def old_synthetic_fitness(brain, player_position, n_states, seed):
    rng = random.Random(seed)
    rewards = []; confidences = []; penalties = 0.0
    intent_counts = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    for _ in range(n_states):
        st = random_game_state(rng, player_position)
        all_names = {}
        for i, (tx, ty) in enumerate(st["teammates"]): all_names[f"t{i}"] = (tx, ty)
        for i, (dx, dy) in enumerate(st["defenders"]): all_names[f"d{i}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)
        teammates = _coords_to_players(st["teammates"], "t")
        defenders = _coords_to_players(st["defenders"], "d")
        sensors = extract_sensors(None, st["x"], st["y"], teammates, defenders, pe, st["under_pressure"], st["attacks_right"], st["game_state"], st["minute"])
        
        probs = brain.forward(sensors)
        idx = int(np.argmax(probs))
        intent = _INTENT_BY_IDX[idx]
        intent_counts[idx] += 1.0
        reward = _context_reward(intent, player_position, sensors)
        confidence = float(probs[idx])
        rewards.append(reward)
        confidences.append(confidence)

        sorted_p = np.sort(probs)[::-1]
        if len(sorted_p) > 1 and (sorted_p[0] - sorted_p[1]) < 0.02:
            penalties += 0.1
    
    mean_reward = statistics.mean(rewards)
    mean_conf = statistics.mean(confidences)
    return mean_reward * (0.5 + 0.5 * mean_conf) - penalties / n_states

brain = FootballBrain.random(42)
old_f = old_synthetic_fitness(brain, "GK", 400, 42)
print("Old:", old_f)