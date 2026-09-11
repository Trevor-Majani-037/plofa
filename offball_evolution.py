"""Off-ball conscience evolution — GA for the binary press/hold gate.

The conscience net (OffBallBrain, 24->32->32->1) replaces the per-position
Bernoulli press trigger in MatchEngine._offball_move_player.  Evolution is
grounded in real-match data: the OffBallSurrogate (offball_probe.py) maps
(position, situation bucket, decision) -> expected defensive-episode score,
learned from collected press decisions against the heuristic baseline.

Synthetic fitness resembles brain_evolution.synthetic_fitness but for the
binary decision: for each random-but-realistic off-ball state it asks the
net's press probability p, picks the argmax decision, and rewards the
surrogate's expected success of that decision weighted by confidence (a net
that commits firmly to the good choice scores higher).  An indecision
penalty applies near p ~ 0.5.

Outcome state generation mirrors what the live engine actually presents at
the decision point: the player is defending (team_possession=0), ball within
~12.5 m of the runner, with teammates/defenders around the ball and runner.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain_sensors import extract_offball_sensors
from football_brain import OffBallBrain, INPUT_SIZE


# ─────────────────────────────────────────────────────────────
# SYNTHETIC OFF-BALL STATE GENERATION
# ─────────────────────────────────────────────────────────────

def random_offball_state(rng: random.Random, position: str) -> Dict[str, Any]:
    """Generate a random-but-realistic state for a DEFENDING off-ball player.

    Ball sits within the chase-trigger radius of the runner.  All coordinates
    are metres on a 105x68 pitch; the runner forwards are measured from the
    ball, defenders cluster around the ball carrier (the runner presses a
    carrier, not a loose ball).
    """
    pos = position
    # Home-zone x for the runner (defending shape).
    if pos in ("ST", "CF", "LW", "RW", "CAM"):
        rx = rng.uniform(55, 100)
    elif pos in ("CM", "CDM"):
        rx = rng.uniform(35, 75)
    else:  # CB, LB, RB, GK-outfield
        rx = rng.uniform(15, 60)
    ry = rng.uniform(5, 63)

    attacks_right = rng.random() < 0.5
    # Ball within trigger radius 12.5 m of the runner, in the general
    # direction of the enemy goal.
    direction = 1.0 if attacks_right else -1.0
    bx = rx + direction * rng.uniform(0, 12.0)
    by = ry + rng.uniform(-8, 8)
    bx = max(1.0, min(104.0, bx))
    by = max(1.0, min(67.0, by))

    minute = rng.uniform(5, 90)
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1",
                        "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = type("GS", (), {"name": rng.choice(game_state_names)})()

    # Teammates: already goal-side of ball or goal-side of runner.
    n_teammates = rng.randint(3, 8)
    teammates = []
    for _ in range(n_teammates):
        tx = bx + (rng.uniform(-15, 5) if attacks_right else rng.uniform(-5, 15))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, by + rng.uniform(-20, 20)))
        teammates.append((tx, ty))

    # Defenders (opposition ball carrier's team): cluster around the ball.
    n_defenders = rng.randint(1, 5)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, bx + rng.uniform(-8, 8)))
        dy = max(0, min(68, by + rng.uniform(-12, 12)))
        defenders.append((dx, dy))

    return {
        "rx": rx, "ry": ry, "bx": bx, "by": by,
        "attacks_right": attacks_right, "minute": minute,
        "game_state": game_state, "teammates": teammates, "defenders": defenders,
    }


class _DummyPositionEngine:
    def __init__(self, positions: Dict[str, Tuple[float, float]]):
        self.positions = positions
    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def _coords_to_players(coords: List[Tuple[float, float]], prefix: str):
    class FakePlayer:
        def __init__(self, name, position):
            self.name = name
            self.position = position
    return [FakePlayer(f"{prefix}{i}", "CM") for i, (x, y) in enumerate(coords)]


def _sensors_for(state: Dict[str, Any], position: str) -> np.ndarray:
    t_coords = state["teammates"]
    d_coords = state["defenders"]
    all_names = {}
    for i, (tx, ty) in enumerate(t_coords):
        all_names[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(d_coords):
        all_names[f"d{i}"] = (dx, dy)
    pe = _DummyPositionEngine(all_names)
    teammates = _coords_to_players(t_coords, "t")
    defenders = _coords_to_players(d_coords, "d")

    class _Runner:
        def __init__(self):
            self.position = position
            self.dna = None

    return extract_offball_sensors(
        _Runner(), state["rx"], state["ry"], state["bx"], state["by"],
        teammates=teammates, defenders=defenders, position_engine=pe,
        attacks_right=state["attacks_right"], game_state=state["game_state"],
        minute=state["minute"],
    )


# ─────────────────────────────────────────────────────────────
# SYNTHETIC FITNESS — evidence-gated conscience reward
# ─────────────────────────────────────────────────────────────

# Heuristic off-ball press probabilities (match_engine._PRESS_PROB).  The
# conscience may ONLY override these where decision-local data supports it;
# elsewhere it is rewarded for reproducing the heuristic frequency, so an
# always-press collapse is impossible.
_HEUR_PRESS_PROB = {
    "GK": 0.0, "CB": 0.80, "LB": 0.80, "RB": 0.80,
    "CDM": 0.75, "CM": 0.75, "CAM": 0.60,
    "LW": 0.45, "RW": 0.45, "ST": 0.50, "CF": 0.50,
}

# Both decisions need at least this many real samples in the cell before the
# conscience is allowed to override the heuristic (evidence gate).
_EVIDENCE_MIN = 5


def synthetic_conscience_fitness(
    brain: OffBallBrain,
    player_position: str,
    n_states: int = 400,
    seed: int = 0,
    surrogate: Optional[Any] = None,
) -> float:
    """Score a conscience brain over random off-ball defending states.

    Evidence-gated reward (fixes the off-policy collapse where the old fitness
    rewarded "press" in cells where the heuristic -- which pressed 67% of the
    time -- created all the data):

      * evidence cell (both press and hold >= _EVIDENCE_MIN samples):
          reward = surrogate.expected_success(decided)
        -> the net learns WHICH decision is genuinely better there.
      * silent cell (no decision-local counterfactual):
          reward = 1 - |p - _HEUR_PRESS_PROB[pos]|   (smooth, maximized at the
          heuristic press frequency) -> the net reproduces the calibrated
          heuristic instead of committing blindly.

    Without a surrogate the reward is the neutral 0.5 for every state.
    """
    rng = random.Random(seed)
    heur = _HEUR_PRESS_PROB.get(player_position, 0.6)
    rewards: List[float] = []

    for _ in range(n_states):
        st = random_offball_state(rng, player_position)
        sensors = _sensors_for(st, player_position)
        p = brain.forward(sensors)
        decision = "press" if p > 0.5 else "hold"

        if surrogate is not None and surrogate.is_evidence_cell(
            sensors, player_position, min_both=_EVIDENCE_MIN):
            reward = surrogate.expected_success(sensors, decision, player_position)
        elif surrogate is not None:
            reward = 1.0 - abs(p - heur)
        else:
            reward = 0.5
        rewards.append(reward)

    if not rewards:
        return 0.0
    return max(0.0, min(1.0, statistics.mean(rewards)))


# ─────────────────────────────────────────────────────────────
# POPULATION EVOLUTION
# ─────────────────────────────────────────────────────────────

@dataclass
class OffBallEvolutionResult:
    best_brain: OffBallBrain
    best_fitness: float
    generation: List[float]
    mean: List[float]


def evolve_conscience(
    position: str,
    population_size: int = 32,
    generations: int = 40,
    n_states: int = 400,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 0,
    verbose: bool = True,
    surrogate: Optional[Any] = None,
) -> OffBallEvolutionResult:
    rng = random.Random(seed)
    pop = [OffBallBrain.random(seed=i) for i in range(population_size)]

    best_overall: Optional[OffBallBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []

    mutate_rate = base_mutate_rate

    for gen in range(generations):
        fitnesses = []
        for brain in pop:
            f = synthetic_conscience_fitness(
                brain, position, n_states=n_states,
                seed=seed + gen * 1000 + len(fitnesses),
                surrogate=surrogate)
            fitnesses.append(f)

        best_f = max(fitnesses)
        mean_f = statistics.mean(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            data = pop[best_idx].serialize()
            best_overall = OffBallBrain.deserialize(data)

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        order = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]
        n_parents = max(2, int(population_size * crossover_top))
        parents = [pop[i] for i in order[:n_parents]]

        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2)
            child = child.mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate *= mutate_decay

    return OffBallEvolutionResult(
        best_brain=best_overall if best_overall is not None else pop[0],
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )