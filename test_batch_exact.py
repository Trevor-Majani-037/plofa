import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from football_brain import FootballBrain

brain = FootballBrain.random(42)
sensors_list = [np.random.rand(24) for _ in range(5)]
sensors_arr = np.stack(sensors_list)
out_loop = np.stack([brain.forward(s) for s in sensors_list])
out_batched = brain.forward(sensors_arr)
print("Max diff:", np.max(np.abs(out_loop - out_batched)))
print("Exact match?", np.array_equal(out_loop, out_batched))