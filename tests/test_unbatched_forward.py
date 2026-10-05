import time, random, statistics, math
import numpy as np
import sys
sys.path.insert(0, '.')
from brain_evolution import evolve

evolve("GK", 32, 2, 400, 42, verbose=True)