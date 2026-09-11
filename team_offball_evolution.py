"""Team press-engagement evolution — GA for the shared XI controller.

TeamPressBrain reads a 24-d TEAM-state vector and outputs engagement g in
[0,1].  The fitness is grounded in real-match TeamPressSurrogate data (the
evidence-gated scheme that worked for the conscience, but now the decision is
the UNIT's engagement band — sit/med/press — which the individual probe
proved was the football-correct lever).

States are drawn PSEUDO-EVOLUTION style from the collected match moments
(real team sensor vectors), so the controller trains on situations it will
actually see.  Reward:

  * evidence cell (>= _EVIDENCE_MIN both for the band AND another band in the
    same team-state bucket): surrogate.expected_success(band(g))
  * silent cell: 1 - |g - median_effort(bucket)|  (reproduce what the
    heuristic unit typically did where we have no counterfactual)

band(g): g >= 0.6 -> press (% of the unit committing), g >= 0.2 -> med,
else sit.  A net that collapses to always-press is penalised in silent cells
(no counterfactual => must reproduce the observed engagement), and in
evidence cells the surrogate says whether HIGH or LOW engagement genuinely
paid off.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from football_brain import TeamPressBrain

_EVIDENCE_MIN = 5


def _band(g: float) -> str:
    if g >= 0.60:
        return "press"
    if g >= 0.20:
        return "med"
    return "sit"


def _median_effort(surrogate: Any, sensors: np.ndarray, style: str) -> float:
    """Median observed engagement for the state's bucket (heuristic prior)."""
    key = surrogate.bucket(sensors)
    row = None
    for s in (style, ""):
        row = surrogate.counts.get(s, {}).get(key)
        if row:
            break
    if not row:
        return 0.5
    # counts table maps band -> n; engage levels conventionally 0.05/0.4/0.85
    scale = {"sit": 0.05, "med": 0.40, "press": 0.85}
    tot = sum(row.values())
    if tot == 0:
        return 0.5
    eff = sum(row.get(b, 0) * scale.get(b, 0.4) for b in row) / tot
    return eff


def team_synthetic_fitness(
    brain: TeamPressBrain,
    moments: List[np.ndarray],
    style: str = "",
    n_states: int = 600,
    seed: int = 0,
    surrogate: Optional[Any] = None,
) -> float:
    """Score a team controller over real recorded team moments."""
    rng = random.Random(seed)
    rewards: List[float] = []
    n = len(moments)
    if n == 0:
        return 0.0

    for _ in range(n_states):
        sensors = moments[rng.randrange(n)]
        g = brain.forward(sensors)
        band = _band(g)

        if surrogate is not None and surrogate.is_evidence_cell(sensors, style,
                                                                _EVIDENCE_MIN):
            reward = surrogate.expected_success(sensors, band, style)
        elif surrogate is not None:
            prior = _median_effort(surrogate, sensors, style)
            reward = 1.0 - abs(g - prior)
        else:
            reward = 0.5
        rewards.append(reward)

    return max(0.0, min(1.0, statistics.mean(rewards)))


@dataclass
class TeamEvolutionResult:
    best_brain: TeamPressBrain
    best_fitness: float
    generation: List[float]
    mean: List[float]


def evolve_team_press(
    moments: List[np.ndarray],
    style: str = "",
    population_size: int = 32,
    generations: int = 40,
    n_states: int = 600,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 0,
    verbose: bool = True,
    surrogate: Optional[Any] = None,
) -> TeamEvolutionResult:
    rng = random.Random(seed)
    pop = [TeamPressBrain.random(seed=i) for i in range(population_size)]

    best_overall: Optional[TeamPressBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []
    mutate_rate = base_mutate_rate

    for gen in range(generations):
        fitnesses = []
        for i, brain in enumerate(pop):
            f = team_synthetic_fitness(
                brain, moments, style=style, n_states=n_states,
                seed=seed + gen * 1000 + i, surrogate=surrogate)
            fitnesses.append(f)

        best_f = max(fitnesses)
        mean_f = statistics.mean(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            best_overall = TeamPressBrain.deserialize(pop[best_idx].serialize())

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        order = sorted(range(population_size),
                       key=lambda i: fitnesses[i], reverse=True)
        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]
        parents = [pop[i] for i in order[:max(2, int(population_size * crossover_top))]]

        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2).mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate *= mutate_decay

    return TeamEvolutionResult(
        best_brain=best_overall if best_overall is not None else pop[0],
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )