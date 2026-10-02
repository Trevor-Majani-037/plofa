#!/usr/bin/env python3
"""CLOSED-LOOP GA — engine-outcome fitness, multi-match variance control.

Process: pop 8 x gen 3 x 3 eval matches = 72 real matches (~30 min) plus a
6-match champion-vs-incumbent shootout on FRESH seeds (~5 min).

Variance control: every brain faces the SAME eval seeds (GA_SEED + m*100),
so the GA compares brains under identical match conditions — the fitness
deltas are genuine differences, and 3-match averaging suppresses the
single-scorer lottery the 1-match proof suffered from.

Decision rule baked in: the champion is saved to brains/ST_engine.json ONLY
if it beats the incumbent brains/ST.json on the fresh-seed shootout.
Otherwise nothing is written (explicitly scrap).
"""
import math
import statistics
import time

from brain_evolution import evolve, extract_outcome_signals, outcome_fitness
from football_brain import FootballBrain
from match_probe import build_probe_engine

POSITION = "ST"
POP = 8
GENS = 3
EVAL_MATCHES = 3                 # matches per brain per generation
INPUT_SIZE = 32                  # matches the current v2 role-features ST lineage
TARGET = "ST"                    # real on-pitch name on the probe 4-3-3
GA_SEED = 11                     # base for the eval seeds (same for every brain)
SHOOTOUT_SEEDS = [9001 + m * 100 for m in range(6)]   # fresh, never used in GA


def outcome_scorer(brain: FootballBrain) -> float:
    """3-match averaged outcome fitness on the GA's fixed eval seeds."""
    fits = []
    for m in range(EVAL_MATCHES):
        build = build_probe_engine(TARGET, seed=GA_SEED + m * 100)
        result = build(brain)
        sig = extract_outcome_signals(result, target_player=TARGET,
                                      team=getattr(build, "team", None))
        fits.append(outcome_fitness(sig))
    return statistics.mean(fits)


def shootout(brain: FootballBrain) -> list[float]:
    fits = []
    for s in SHOOTOUT_SEEDS:
        build = build_probe_engine(TARGET, seed=s)
        result = build(brain)
        sig = extract_outcome_signals(result, target_player=TARGET,
                                      team=getattr(build, "team", None))
        fits.append(outcome_fitness(sig))
    return fits


def main():
    t0 = time.time()
    res = evolve(position=POSITION, population_size=POP, generations=GENS,
                 seed=GA_SEED, input_size=INPUT_SIZE, fitness_fn=outcome_scorer,
                 verbose=True)
    elapsed_ga = time.time() - t0
    print(f"\nGA finished in {elapsed_ga/60:.1f} min")
    print(f"gen_best = {[round(g, 4) for g in res.generation]}")
    print(f"gen_mean = {[round(g, 4) for g in res.mean]}")

    champion = res.best_brain
    incumbent = FootballBrain.load("brains/ST.json")

    print("\n=== Shootout (6 fresh seeds each, ~5 min) ===")
    f_inc = shootout(incumbent)
    f_champ = shootout(champion)
    avg_inc = statistics.mean(f_inc)
    avg_champ = statistics.mean(f_champ)
    delta = avg_champ - avg_inc
    print(f"incumbent: {[round(x, 3) for x in f_inc]}  avg={avg_inc:.4f}")
    print(f"champion : {[round(x, 3) for x in f_champ]}  avg={avg_champ:.4f}")
    print(f"delta    : {delta:+.4f}")

    if delta > 0:
        champion.save("brains/ST_engine.json")
        print(f"KEEP  — champion saved -> brains/ST_engine.json")
    else:
        print("SCRAP — champion does not beat incumbent on fresh seeds;")
        print("        nothing written, brains/ST_engine.json stays deleted.")
    print(f"\nTotal wall time ~{(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()