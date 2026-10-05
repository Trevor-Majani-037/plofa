import time, random, sys
sys.path.insert(0, '.')
from brain_evolution_old import evolve

evolve("GK", 32, 10, 400, 42, verbose=True)