"""Defensive-action evolution — GA for the XI defensive-action controller.

DefensiveActionBrain reads a 24-d defensive-state vector and outputs a
softmax over [tackle, interception, clearance, block].  Fitness is grounded
in real-match DefensiveActionSurrogate data collected by
defensive_action_probe.py.

Pseudo-evolution: states are drawn from REAL collected defensive moments
(sensor vectors recorded before each contest), so the controller trains on
situations it will actually face.  Reward:

  * evidence cell (>= _EVIDENCE_MIN observed outcomes) for state bucket AND
    action: surrogate.expected_success(sensors, action) — has this action
    actually won the ball / cleared danger in states like this one in real
    matches (including the forced counterfactuals)?
  * silent/weak cell: shrink toward the bucket-wide expected success (action
    first tried where we have no real counterfactual is scored by the average
    of what siblings DID achieve, minus a small risk premium).

A net that collapses to one action is penalised exactly like the on-ball
evolution: dominance penalty (top action share above 0.50) plus a breadth
bonus that rewards using several acts where the surrogate says they are
nearly equal.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from football_brain import DefensiveActionBrain, DEFENSIVE_ACTIONS

_EVIDENCE_MIN = 3


def synthetic_fitness(
    brain: DefensiveActionBrain,
    moments: List[np.ndarray],
    styles: List[str],
    surrogate: Any,
    n_states: int = 600,
    seed: int = 0,
    dominance_threshold: float = 0.50,
    dominance_scale: float = 1.5,
    breadth_bonus: float = 1.0,
    exploration_premium: float = 0.0,
) -> float:
    """Score a defensive-action brain over real recorded moments."""
    rng = random.Random(seed)
    rewards: List[float] = []
    picks: List[str] = []
    n = len(moments)
    if n == 0:
        return 0.0

    for _ in range(n_states):
        i = rng.randrange(n)
        sensors = moments[i]
        style = styles[i] if i < len(styles) else ""
        action = brain.predict(sensors)
        picks.append(action)

        if surrogate is not None:
            base = surrogate.expected_success(sensors, action, style)
            support = surrogate.support(sensors, action, style)
            if support < _EVIDENCE_MIN:
                # Weak evidence: shrink toward bucket average (what the bucket
                # overall achieves), with a small exploration premium so novel
                # actions aren't instantly doomed in silent cells.
                bucket_avg = surrogate.expected_success(sensors, action, style)
                reward = bucket_avg + exploration_premium
            else:
                reward = base
        else:
            reward = 0.5
        rewards.append(reward)

    fitness = statistics.mean(rewards)

    # diversity: dominance penalty + breadth bonus (mirror  on-ball scheme)
    counts = {a: 0 for a in DEFENSIVE_ACTIONS}
    for a in picks:
        counts[a] += 1
    top_share = max(counts.values()) / len(picks)
    n_used = sum(1 for a in counts.values() if a > 0)
    bdiv = (n_used - 1) / (len(DEFENSIVE_ACTIONS) - 1)
    dom_pen = max(0.0, top_share - dominance_threshold) * dominance_scale

    scaled = (fitness - dom_pen) * (1.0 + breadth_bonus * bdiv) / 1.5
    return max(0.0, min(1.0, scaled))


@dataclass
class EvolutionResult:
    best_brain: DefensiveActionBrain
    best_fitness: float
    generation: List[float]
    mean: List[float]


def evolve_defensive_action(
    moments: List[np.ndarray],
    styles: Optional[List[str]] = None,
    surrogate: Optional[Any] = None,
    population_size: int = 32,
    generations: int = 40,
    n_states: int = 600,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 0,
    verbose: bool = True,
) -> EvolutionResult:
    rng = random.Random(seed)
    if styles is None:
        styles = [""] * len(moments)
    pop = [DefensiveActionBrain.random(seed=i + seed * 1000)
           for i in range(population_size)]

    best_overall: Optional[DefensiveActionBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []
    mutate_rate = base_mutate_rate

    for gen in range(generations):
        fitnesses = []
        for i, brain in enumerate(pop):
            f = synthetic_fitness(
                brain, moments, styles, surrogate=surrogate, n_states=n_states,
                seed=seed + gen * 1000 + i)
            fitnesses.append(f)

        best_f = max(fitnesses)
        mean_f = statistics.mean(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            best_overall = DefensiveActionBrain.deserialize(
                pop[best_idx].serialize())

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        order = sorted(range(population_size),
                       key=lambda i: fitnesses[i], reverse=True)
        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]
        parents = [pop[i] for i in
                   order[:max(2, int(population_size * crossover_top))]]

        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2).mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate *= mutate_decay

    return EvolutionResult(
        best_brain=best_overall if best_overall is not None else pop[0],
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )


def distribution_report(brain: DefensiveActionBrain,
                        moments: List[np.ndarray]) -> Dict[str, Any]:
    picks = [brain.predict(s) for s in moments]
    return {"n": len(picks),
            "counts": {a: picks.count(a) for a in DEFENSIVE_ACTIONS}}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "brains_def/samples_forced.json"
    with open(path, "r") as f:
        rows = json.load(f)
    moments = [np.asarray(r["sensors"], dtype=np.float64) for r in rows]
    styles = [r.get("style", "") for r in rows]
    print(f"loaded {len(moments)} moments from {path}")
    print("action mix:", {a: sum(1 for r in rows if r["action"] == a)
                          for a in DEFENSIVE_ACTIONS})
    mr = evolve_defensive_action(moments, styles)
    print(f"\nbest fitness {mr.best_fitness:.4f}")
    print("distribution over recorded moments:",
          distribution_report(mr.best_brain, moments))