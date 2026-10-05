import time, sys
sys.path.insert(0, '.')
import sys_patch
sys.modules["brain_evolution"] = __import__("brain_evolution_old")
import evolve_brains
evolve_brains.main(["--position", "GK", "--generations", "40", "--seed", "42"])