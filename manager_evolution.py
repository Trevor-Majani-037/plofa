"""
PLOFA 26/27 — MANAGER BRAIN EVOLUTION (Phase 7)
====================================================
manager_evolution.py

GA evolution of ManagerBrain over the collected STATIC-manager curriculum,
mirroring brain_evolution.evolve's operator set (elitism, crossover,
mutation, annealed decay) but with a surrogate-driven fitness:

    synthetic_fitness(brain, samples, surrogate):
        per sample:  forward -> posture_probs -> argmax posture
                     reward   = surrogate.expected_points(sensors, posture)
        fitness     = mean(reward) * (0.5 + 0.5 * mean_confidence)
        - dominance penalty when a single posture exceeds 60% share
        - entropy bonus favouring context-dependent posture variety
        scaled to [0, 1].

The forward pass is vectorised across ALL samples in one call
(batch matmul), then rewards/penalties are accumulated per sample.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from manager_brain import ManagerBrain, MANAGER_POSTURE_LABELS


@dataclass
class EvolutionResult:
    best_brain: ManagerBrain
    best_fitness: float
    generation: List[float]   # best fitness per generation
    mean: List[float]         # mean fitness per generation


def _posture_index(posture: str) -> int:
    try:
        return MANAGER_POSTURE_LABELS.index(posture)
    except ValueError:
        return 1  # unknown posture -> BALANCED


def synthetic_fitness(brain: ManagerBrain,
                      samples: Sequence[Any],
                      surrogate: Optional[Any] = None) -> float:
    """Score a manager brain over the collected samples.

    Parameters
    ----------
    brain : ManagerBrain
    samples : sequence of (sensors, posture, outcome_points, match_outcome)
    surrogate : ManagerSurrogate supplying expected_points(sensors, posture)

    Returns a scalar in [0, 1].
    """
    if not samples:
        return 0.0

    sensors_all = np.asarray([np.asarray(s[0], dtype=np.float64) for s in samples],
                             dtype=np.float64)
    n = len(samples)
    if sensors_all.ndim == 1:
        sensors_all = sensors_all[None, :]

    probs = brain.forward(sensors_all)["posture_probs"]  # (n, 3)

    rewards = np.empty(n, dtype=np.float64)
    confidences = np.empty(n, dtype=np.float64)
    posture_counts = np.zeros(3, dtype=np.float64)
    penalties = 0.0

    for i in range(n):
        p = probs[i]
        idx = int(np.argmax(p))
        posture = MANAGER_POSTURE_LABELS[idx]
        posture_counts[idx] += 1.0

        if surrogate is not None:
            reward = surrogate.expected_points(sensors_all[i], posture)
        else:
            reward = float(samples[i][2])
        rewards[i] = reward
        confidences[i] = p[idx]

        sorted_p = np.sort(probs[i])[::-1]
        if len(sorted_p) > 1 and (sorted_p[0] - sorted_p[1]) < 0.02:
            penalties += 0.1

    mean_reward = float(rewards.mean())
    mean_conf = float(confidences.mean())
    fitness = mean_reward * (0.5 + 0.5 * mean_conf) - penalties / n

    # dominance penalty: one posture owning > 60% of decisions is collapsed
    shares = posture_counts / posture_counts.sum()
    max_share = float(shares.max())
    dom_pen = max(0.0, max_share - 0.60) * 1.5

    # entropy bonus: variety across the 3 postures
    pos = shares[shares > 0]
    entropy = -float(np.sum(pos * np.log(pos + 1e-12)))
    bdiv = entropy / math.log(3)

    scaled = (fitness - dom_pen) * (1.0 + 0.5 * bdiv) / 1.5
    return max(0.0, min(1.0, scaled))


def _clone_brain(brain: ManagerBrain) -> ManagerBrain:
    return ManagerBrain(
        brain.w1.copy(), brain.b1.copy(),
        brain.w2.copy(), brain.b2.copy(),
        brain.w3.copy(), brain.b3.copy(),
    )


def evolve(
    samples: Sequence[Any],
    population_size: int = 32,
    generations: int = 40,
    surrogate: Optional[Any] = None,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 0,
    verbose: bool = True,
) -> EvolutionResult:
    """Evolve a ManagerBrain over fixed samples (batch forward per brain).

    Same operator set as brain_evolution.evolve: elitism preserves the top
    ``elitism`` fraction each generation; the rest come from crossover of
    the top ``crossover_top`` parents plus annealed mutation.
    """
    rng = _random.Random(seed)
    pop = [ManagerBrain.random(seed=i) for i in range(population_size)]

    best_overall: Optional[ManagerBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []

    mutate_rate = base_mutate_rate
    min_mutate_rate = 0.05

    for gen in range(generations):
        fitnesses = [synthetic_fitness(b, samples, surrogate) for b in pop]

        best_f = max(fitnesses)
        mean_f = sum(fitnesses) / len(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            best_overall = _clone_brain(pop[best_idx])

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        order = sorted(range(population_size),
                       key=lambda i: fitnesses[i], reverse=True)

        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]

        n_parents = max(2, int(population_size * crossover_top))
        parents = [pop[i] for i in order[:n_parents]]

        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2).mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate = max(min_mutate_rate, mutate_rate * mutate_decay)

    best = best_overall if best_overall is not None else pop[0]
    return EvolutionResult(
        best_brain=best,
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )


def posture_breakdown(brain: ManagerBrain,
                      samples: Sequence[Any]) -> dict:
    """Share of each posture a brain picks over the samples."""
    sensors_all = np.asarray(
        [np.asarray(s[0], dtype=np.float64) for s in samples], dtype=np.float64)
    if sensors_all.ndim == 1:
        sensors_all = sensors_all[None, :]
    probs = brain.forward(sensors_all)["posture_probs"]
    idx = np.argmax(probs, axis=-1)
    counts = np.bincount(idx.ravel(), minlength=3)
    n = int(counts.sum()) or 1
    return {k: float(counts[i] / n) for i, k in enumerate(MANAGER_POSTURE_LABELS)}