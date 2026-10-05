import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from football_brain import FootballBrain
from decision_brain import PlayerIntent

brain = FootballBrain.random(42)
sensors_list = [np.random.rand(24) for _ in range(5)]

# Loop
out_loop = []
for s in sensors_list:
    out_loop.append(brain.forward(s))

# Batched
sensors_arr = np.stack(sensors_list)
out_batched = brain.forward(sensors_arr)

print("Loop:")
for o in out_loop: print(o)
print("Batched:")
print(out_batched)
print("Match?", np.allclose(np.stack(out_loop), out_batched))