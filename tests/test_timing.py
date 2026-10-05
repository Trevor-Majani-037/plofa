import time
import sys
sys.path.insert(0, '.')
from brain_evolution import evolve

t0 = time.time()
res = evolve(
    position="GK",
    population_size=16,
    generations=10,
    n_states=100,
    seed=42,
    verbose=True
)
print(f"Time: {time.time()-t0:.2f}s")