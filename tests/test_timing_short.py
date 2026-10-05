import time, random, sys
sys.path.insert(0, '.')
from brain_evolution import evolve

evolve("GK", 16, 2, 100, 42, verbose=True)