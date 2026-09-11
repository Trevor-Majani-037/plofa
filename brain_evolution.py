"""BrainEvolution — genetic algorithm engine for FootballBrain.

This module grows the population of per-player neural networks through
simulated matches.  Fitness is derived from how well each brain makes
decisions across a large sample of realistic game states.

Two evaluation modes:
  1. SYNTHETIC (default, fast)  — scores a brain against thousands of
     random-but-realistic match states generated from geometric
     priors.  No full match engine needed.  Great for fast iteration.
  2. FULL_MATCH                  — hooks into the real MatchEngine for a
     more accurate but far slower fitness signal.  Call
     ``evaluate_full_match(brain, ...)`` yourself with your own
     engine wiring; the population runner here stays pluggable.

Design
------
    population_of 32 brains →
    for each generation:
        score every brain (fitness)
        elitism: keep top 20% unchanged
        crossover top 50% to fill new population
        mutate the non-elite with decaying rate
        save best of generation
    → final best brain

Position-aware rewards: a ST evolves toward shots and through-balls, a
CB toward safe passes and recycling, a winger toward crosses and
drives.  Powerful right-side rewards are, by design, asymmetric with
left-side to force variety.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from football_brain import FootballBrain, INPUT_SIZE, OUTPUT_SIZE
from decision_brain import PlayerIntent
from brain_sensors import extract_sensors


# ─────────────────────────────────────────────────────────────
# POSITION PROFILES → fitness reward weights
# ─────────────────────────────────────────────────────────────

# reward is applied for each intent the brain chooses; it biases which
# intents a player gets credit for.  All intents still possible — just
# weighted differently by role.
POSITION_REWARDS: Dict[str, Dict[PlayerIntent, float]] = {
    "ST": {PlayerIntent.SHOOT: 1.0, PlayerIntent.THROUGH_BALL: 0.8,
           PlayerIntent.PROGRESSIVE_PASS: 0.6, PlayerIntent.CARRY: 0.5,
           PlayerIntent.CROSS: 0.4, PlayerIntent.SAFE_PASS: 0.2},
    "CF": {PlayerIntent.SHOOT: 0.9, PlayerIntent.THROUGH_BALL: 0.9,
           PlayerIntent.PROGRESSIVE_PASS: 0.8, PlayerIntent.CARRY: 0.6,
           PlayerIntent.CROSS: 0.4, PlayerIntent.SAFE_PASS: 0.3},
    "LW": {PlayerIntent.CROSS: 1.0, PlayerIntent.DRIBBLE: 0.9,
           PlayerIntent.CARRY: 0.8, PlayerIntent.PROGRESSIVE_PASS: 0.5,
           PlayerIntent.THROUGH_BALL: 0.5, PlayerIntent.SHOOT: 0.5,
           PlayerIntent.SAFE_PASS: 0.3},
    "RW": {PlayerIntent.CROSS: 1.0, PlayerIntent.DRIBBLE: 0.9,
           PlayerIntent.CARRY: 0.8, PlayerIntent.PROGRESSIVE_PASS: 0.5,
           PlayerIntent.THROUGH_BALL: 0.5, PlayerIntent.SHOOT: 0.5,
           PlayerIntent.SAFE_PASS: 0.3},
    "CAM": {PlayerIntent.THROUGH_BALL: 1.0, PlayerIntent.PROGRESSIVE_PASS: 0.9,
            PlayerIntent.SWITCH: 0.6, PlayerIntent.SHOOT: 0.7,
            PlayerIntent.CARRY: 0.5, PlayerIntent.SAFE_PASS: 0.4},
    "CM": {PlayerIntent.PROGRESSIVE_PASS: 0.9, PlayerIntent.SWITCH: 0.8,
           PlayerIntent.SAFE_PASS: 0.8, PlayerIntent.RECYCLE: 0.7,
           PlayerIntent.PROTECT_POSSESSION: 0.5, PlayerIntent.CARRY: 0.4,
           PlayerIntent.THROUGH_BALL: 0.6},
    "CDM": {PlayerIntent.SAFE_PASS: 1.0, PlayerIntent.RECYCLE: 0.9,
            PlayerIntent.PROTECT_POSSESSION: 0.8, PlayerIntent.SWITCH: 0.5,
            PlayerIntent.PROGRESSIVE_PASS: 0.5, PlayerIntent.CARRY: 0.3},
    "CB": {PlayerIntent.SAFE_PASS: 1.0, PlayerIntent.RECYCLE: 1.0,
           PlayerIntent.PROTECT_POSSESSION: 0.8, PlayerIntent.SWITCH: 0.4,
           PlayerIntent.PROGRESSIVE_PASS: 0.3, PlayerIntent.CARRY: 0.2},
    "LB": {PlayerIntent.SAFE_PASS: 0.9, PlayerIntent.CROSS: 0.8,
           PlayerIntent.RECYCLE: 0.7, PlayerIntent.PROGRESSIVE_PASS: 0.6,
           PlayerIntent.CARRY: 0.5, PlayerIntent.SWITCH: 0.5},
    "RB": {PlayerIntent.SAFE_PASS: 0.9, PlayerIntent.CROSS: 0.8,
           PlayerIntent.RECYCLE: 0.7, PlayerIntent.PROGRESSIVE_PASS: 0.6,
           PlayerIntent.CARRY: 0.5, PlayerIntent.SWITCH: 0.5},
    "GK": {PlayerIntent.SAFE_PASS: 1.0, PlayerIntent.RECYCLE: 0.9,
           PlayerIntent.PROTECT_POSSESSION: 0.8, PlayerIntent.PROGRESSIVE_PASS: 0.3},
    # default for unknown positions
    "": {PlayerIntent.SAFE_PASS: 0.7, PlayerIntent.PROGRESSIVE_PASS: 0.7,
         PlayerIntent.CARRY: 0.5, PlayerIntent.RECYCLE: 0.6},
}


def position_rewards(position: str) -> Dict[PlayerIntent, float]:
    return POSITION_REWARDS.get(position, POSITION_REWARDS[""])


# context-dependent reward scaling: sensor index → (intent, scale_fn)
# makes SHOOT only valuable near goal, CROSS only valuable in crossing zone,
# THROUGH_BALL only valuable with open teammates.
def _context_reward(intent: PlayerIntent, position: str,
                    sensors: np.ndarray) -> float:
    """Reward adjusted by sensor context — prevents always-SHOOT collapse.

    Each intent is scaled by relevant sensors so the brain MUST condition
    on context to get high fitness.  Creates natural complementarity:
    shoot near goal, pass when far, cross from wide zones, etc.
    """
    base = POSITION_REWARDS.get(position, POSITION_REWARDS[""]).get(intent, 0.0)

    near_goal = float(sensors[11])
    fwd_open = float(sensors[8])
    fwd_dist = float(sensors[7])
    crossing = float(sensors[10])
    space = float(sensors[9])
    def_5 = float(sensors[5])
    def_press = 1.0 - def_5

    if intent == PlayerIntent.SHOOT:
        if position in ("ST", "CF", "LW", "RW", "CAM"):
            return base * near_goal + 0.12 * (1.0 - near_goal)
    elif intent == PlayerIntent.THROUGH_BALL:
        fwd_avail = min(1.0, fwd_dist * 2.5)
        near_pen = 0.30 + 0.70 * (1.0 - near_goal)
        return base * fwd_avail * (0.20 + 0.80 * fwd_open) * near_pen * def_press
    elif intent == PlayerIntent.CROSS:
        return base * (0.10 + 0.90 * crossing) * def_press
    elif intent == PlayerIntent.PROGRESSIVE_PASS:
        near_pen = 0.40 + 0.60 * (1.0 - near_goal)
        return base * (0.20 + 0.80 * fwd_open) * near_pen * (0.5 + 0.5 * def_press)
    elif intent == PlayerIntent.CARRY:
        return base * (0.15 + 0.85 * space)
    elif intent == PlayerIntent.DRIBBLE:
        return base * (0.15 + 0.85 * space)
    return base


# ─────────────────────────────────────────────────────────────
# SYNTHETIC GAME-STATE GENERATION
# ─────────────────────────────────────────────────────────────

def random_game_state(rng: random.Random, position: str) -> Dict[str, Any]:
    """Generate a random-but-realistic match state for a given position.

    Returns a dict with keys: x, y, attackers_right, under_pressure,
    minute, game_state, plus lists of teammate/defender coordinates.
    """
    # x range depends on position: attackers live higher up
    if position in ("ST", "CF", "LW", "RW", "CAM"):
        x = rng.uniform(40, 95)
    elif position in ("CM", "CDM"):
        x = rng.uniform(25, 70)
    elif position in ("CB", "LB", "RB"):
        x = rng.uniform(10, 50)
    else:
        x = rng.uniform(30, 80)
    y = rng.uniform(2, 66)

    attacks_right = rng.random() < 0.5
    under_pressure = rng.random() < 0.4
    minute = rng.uniform(0, 90)
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1",
                        "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = type("GS", (), {"name": rng.choice(game_state_names)})()

    # teammates -- position-correct density
    n_teammates = rng.randint(3, 8)
    teammates = []
    for _ in range(n_teammates):
        tx = x + (rng.uniform(-5, 25) if attacks_right else rng.uniform(-25, 5))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, y + rng.uniform(-18, 18)))
        teammates.append((tx, ty))

    # defenders -- cluster around ball
    n_defenders = rng.randint(1, 6)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, x + rng.uniform(-8, 8)))
        dy = max(0, min(68, y + rng.uniform(-12, 12)))
        defenders.append((dx, dy))

    return {
        "x": x, "y": y, "attacks_right": attacks_right,
        "under_pressure": under_pressure, "minute": minute,
        "game_state": game_state, "teammates": teammates, "defenders": defenders,
    }


def random_game_state_scoring(rng: random.Random, position: str) -> Dict[str, Any]:
    """Generate a random state biased toward SCORING situations.

    This is the fix for the v3 retrain collapse: evolution on fully-random
    states never sees enough final-third / goal-close chances, so the rare
    scoring intents (ST SHOOT, CM through-balls, winger crosses) get
    optimised out of the argmax.  These states push the carrier into
    attacking positions (final third, goal-close, central) so evolution
    actually learns when to SHOOT / cross / play the killer pass.

    Same return layout as ``random_game_state`` (the x/y bands differ).
    """
    if position in ("ST", "CF", "LW", "RW", "CAM"):
        x = rng.uniform(78, 96)          # attackers planted high
    elif position in ("CM", "CDM"):
        x = rng.uniform(70, 88)          # deep runners arriving late
    else:
        x = rng.uniform(72, 92)          # everyone else pushes up
    y = rng.uniform(14, 54)

    attacks_right = rng.random() < 0.5
    under_pressure = rng.random() < 0.45   # slightly more pressure near goal
    minute = rng.uniform(55, 90)           # attacking phase
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1", "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = type("GS", (), {"name": rng.choice(game_state_names)})()

    # teammates: mix of support (behind) and runners (ahead in the box)
    n_teammates = rng.randint(3, 6)
    teammates = []
    for _ in range(n_teammates):
        tx = x + (rng.uniform(-6, 14) if attacks_right else rng.uniform(-14, 6))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, y + rng.uniform(-14, 14)))
        teammates.append((tx, ty))

    # defenders: packed around the ball (box is crowded)
    n_defenders = rng.randint(2, 6)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, x + rng.uniform(-6, 6)))
        dy = max(0, min(68, y + rng.uniform(-8, 8)))
        defenders.append((dx, dy))

    return {
        "x": x, "y": y, "attacks_right": attacks_right,
        "under_pressure": under_pressure, "minute": minute,
        "game_state": game_state, "teammates": teammates, "defenders": defenders,
    }


class _DummyPositionEngine:
    """Minimal position_engine stand-in for synthetic evaluation."""
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


# ─────────────────────────────────────────────────────────────
# POPULATION DIVERSITY (anti-collapse)
# ─────────────────────────────────────────────────────────────

def _compute_pop_diversity(pop: List[FootballBrain]) -> float:
    """Average L2 distance from each brain to the population centroid.

    Returns a non-negative scalar: 0 when every brain has identical
    weights (full collapse), higher when the population is spread.
    """
    flat = np.array([np.concatenate([
        b.w1.ravel(), b.b1.ravel(),
        b.w2.ravel(), b.b2.ravel(),
        b.w3.ravel(), b.b3.ravel(),
    ]) for b in pop])
    centroid = flat.mean(axis=0)
    dists = np.linalg.norm(flat - centroid, axis=1)
    return float(dists.mean())


def _apply_sharing(fitnesses: List[float], pop: List[FootballBrain],
                   sigma: float = 10.0) -> List[float]:
    """Fitness sharing (Goldberg & Richardson 1987).

    Divides each brain's fitness by its niche count — the sum of
    similarity to all other brains using a triangular kernel.
    Penalises crowded regions of weight space, preserving diversity.
    sigma controls the niche radius in L2 weight-distance units.
    """
    flat = np.array([np.concatenate([
        b.w1.ravel(), b.b1.ravel(),
        b.w2.ravel(), b.b2.ravel(),
        b.w3.ravel(), b.b3.ravel(),
    ]) for b in pop])
    # pairwise L2 distances
    sq = np.sum(flat ** 2, axis=1)
    dsq = sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T
    dsq = np.maximum(dsq, 0.0)
    dists = np.sqrt(dsq)

    shared = np.array(fitnesses, dtype=float)
    for i in range(len(pop)):
        niche = float(np.sum(np.maximum(0.0, 1.0 - dists[i] / sigma)))
        if niche > 0:
            shared[i] /= niche
    return shared.tolist()


# ─────────────────────────────────────────────────────────────
# FITNESS SCORING (synthetic)
# ─────────────────────────────────────────────────────────────

def synthetic_fitness(
    brain: FootballBrain,
    player_position: str,
    n_states: int = 400,
    seed: int = 0,
    surrogate: Optional[Any] = None,
    pop_diversity: float = 0.0,
    goal_bias: float = 0.0,
) -> float:
    """Score a brain by running it over many random game states.

    The per-state reward is either the position reward table (hand-made)
    or — when a ``surrogate`` (a ``FitnessSurrogate``) is supplied — the
    surrogate's learned expected-success for the chosen intent given the
    sensor state.  Using a surrogate grounds evolution in what actually
    wins real matches instead of our guessed preferences.

    Fitness = mean reward of chosen intents, weighted by the network's
    own confidence (a brain that commits firmly to a rewarding intent
    scores higher than one that's always wishy-washy), minus a penalty
    for indecision (all outputs near-uniform = low confidence).

    Two diversity bonuses prevent population collapse:
      - ``pop_diversity``: average L2 distance to population centroid,
        set by evolve().  Rewards brains that are different from the herd.
      - Behavioral entropy: Shannon entropy of the marginal intent
        distribution across all states.  Rewards brains that adapt their
        choices to context rather than always picking the same intent.

    Returns a scalar in roughly [0, 1].
    """
    rng = random.Random(seed)
    rewards: List[float] = []
    confidences: List[float] = []
    penalties = 0.0
    intent_counts = np.zeros(OUTPUT_SIZE, dtype=np.float64)

    for _ in range(n_states):
        if goal_bias > 0.0 and rng.random() < goal_bias:
            st = random_game_state_scoring(rng, player_position)
        else:
            st = random_game_state(rng, player_position)

        t_coords = st["teammates"]
        d_coords = st["defenders"]
        all_names = {}
        for i, (tx, ty) in enumerate(t_coords):
            all_names[f"t{i}"] = (tx, ty)
        for i, (dx, dy) in enumerate(d_coords):
            all_names[f"d{i}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)

        teammates = _coords_to_players(t_coords, "t")
        defenders = _coords_to_players(d_coords, "d")

        sensors = extract_sensors(
            None, st["x"], st["y"], teammates, defenders, pe,
            st["under_pressure"], st["attacks_right"], st["game_state"],
            st["minute"],
        )
        probs = brain.forward(sensors)
        idx = int(np.argmax(probs))
        intent = _INTENT_BY_IDX[idx]
        intent_counts[idx] += 1.0

        if surrogate is not None:
            reward = surrogate.expected_success(sensors, intent, player_position)
        else:
            reward = _context_reward(intent, player_position, sensors)
        confidence = float(probs[idx])
        rewards.append(reward)
        confidences.append(confidence)

        sorted_p = np.sort(probs)[::-1]
        if len(sorted_p) > 1 and (sorted_p[0] - sorted_p[1]) < 0.02:
            penalties += 0.1

    if not rewards:
        return 0.0

    mean_reward = statistics.mean(rewards)
    mean_conf = statistics.mean(confidences)
    fitness = mean_reward * (0.5 + 0.5 * mean_conf) - penalties / n_states

    # Behavioral diversity bonus: Shannon entropy of intent marginal.
    # A brain that always picks the same intent regardless of state gets 0;
    # one that spreads across intents based on context gets up to 1.0.
    intent_probs = intent_counts / intent_counts.sum()
    intent_probs = intent_probs[intent_probs > 0]
    entropy = -float(np.sum(intent_probs * np.log(intent_probs + 1e-12)))
    max_entropy = math.log(OUTPUT_SIZE)
    bdiv = entropy / max_entropy  # normalised to [0, 1]

    # Dominance penalty: if one intent exceeds 50% of states, penalise
    # proportionally.  This prevents the surrogate from collapsing the
    # brain to a single high-value intent (e.g. SWITCH for CM/CF).
    max_share = float(intent_counts.max() / intent_counts.sum())
    dom_pen = max(0.0, max_share - 0.50) * 1.5  # 0 at 50%, 0.75 at 100%

    # Effective-usage penalty: a brain can evade the single-intent rule by
    # collapsing onto TWO intents split ~50/50 (predicted for CAM: CARRY +
    # SWITCH, other 8 intents almost never firing).  So additionally require
    # a spread of behaviour: count intents individually clearing 5% of the
    # batch; if fewer than 6 clear it, apply a proportional penalty (0 at 6+,
    # 0.75 when only 3 intents ever fire, 1.5 at ... clamped by fitness scale).
    if intent_counts.sum() > 0:
        shares = intent_counts / intent_counts.sum()
        n_effective = int(np.sum(shares > 0.05))
        if n_effective < 6:
            shortfall = (6 - n_effective) / 6.0
            dom_pen += shortfall * 0.75  # 0 at 6+, 0.375 at 3, 0.75 at 0

    scaled = (fitness - dom_pen) * (1.0 + 1.0 * bdiv) / 1.5
    return max(0.0, min(1.0, scaled))


# need intent-by-index mapping (mirror brain_integration but avoid a
# heavy circular import)
from football_brain import INTENT_LABELS  # noqa: E402
from decision_brain import PlayerIntent as _PI  # noqa: E402
_INTENT_BY_IDX = [_PI(label) for label in INTENT_LABELS]


# ─────────────────────────────────────────────────────────────
# POPULATION EVOLUTION
# ─────────────────────────────────────────────────────────────

@dataclass
class EvolutionResult:
    best_brain: FootballBrain
    best_fitness: float
    generation: List[float]     # best fitness per generation
    mean: List[float]           # mean fitness per generation


def evolve(
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
    goal_bias: float = 0.0,
) -> EvolutionResult:
    """Run a full evolution run for a single positional brain.

    Parameters
    ----------
    position : str
        Which role this brain plays (ST, CAM, CB, etc.).  Determines
        fitness reward weights.
    population_size : int
        Number of candidate brains in the population.
    generations : int
        Number of evolution generations.
    n_states : int
        Number of random game states to evaluate each brain against.
    elitism : float
        Fraction of population preserved unchanged each generation.
    crossover_top : float
        Fraction of top performers used as crossover parents.
    base_mutate_rate : float
        Initial mutation rate (fraction of weights perturbed).
    mutate_decay : float
        Per-generation multiplier on mutation rate (annealing).
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Print per-generation progress.

    Returns
    -------
    EvolutionResult with best brain + history.
    """
    rng = random.Random(seed)
    pop = [FootballBrain.random(seed=i) for i in range(population_size)]

    best_overall: Optional[FootballBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []

    mutate_rate = base_mutate_rate
    min_mutate_rate = 0.05  # floor: never stop exploring
    init_pop_div = max(_compute_pop_diversity(pop), 1e-6)

    for gen in range(generations):
        # population diversity (anti-collapse bonus), normalised by init
        pop_div = _compute_pop_diversity(pop) / init_pop_div

        # evaluate
        fitnesses = []
        for brain in pop:
            f = synthetic_fitness(brain, position, n_states=n_states,
                                  seed=seed + gen * 1000 + len(fitnesses),
                                  surrogate=surrogate,
                                  pop_diversity=pop_div,
                                  goal_bias=goal_bias)
            fitnesses.append(f)

        # fitness sharing — divide by niche count to penalise convergence
        shared = _apply_sharing(fitnesses, pop)

        best_f = max(fitnesses)
        mean_f = statistics.mean(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            data = pop[best_idx].serialize()
            best_overall = FootballBrain.deserialize(data)

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        # ordering by SHARED fitness desc (diversity-aware selection)
        order = sorted(range(population_size), key=lambda i: shared[i], reverse=True)

        # elitism: keep top N unchanged
        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]

        # fill rest with crossover + mutation
        n_parents = max(2, int(population_size * crossover_top))
        parents = [pop[i] for i in order[:n_parents]]

        while len(next_pop) < population_size:
            # tournament-ish selection among parents
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2)
            # mutation with annealed rate
            child = child.mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate = max(min_mutate_rate, mutate_rate * mutate_decay)

    return EvolutionResult(
        best_brain=best_overall if best_overall is not None else pop[0],
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )


# ─────────────────────────────────────────────────────────────
# FULL-MATCH FITNESS (hook for real engine)
# ─────────────────────────────────────────────────────────────

def evaluate_full_match(brain: FootballBrain, build_engine: Callable[[], Any],
                        n_matches: int = 1) -> float:
    """Evaluate a brain using the real MatchEngine.

    Parameters
    ----------
    brain : FootballBrain
        The brain to evaluate.
    build_engine : Callable[[], Any]
        A zero-arg callable that builds a configured MatchEngine with
        this brain registered to the target player, runs simulate(),
        and returns the result object.  This keeps the heavy engine
        wiring out of this module — you supply it (see evolve_brains.py).
    n_matches : int
        Number of matches to average over.

    Returns
    -------
    float — normalised fitness from match outcomes.
    """
    scores = []
    for _ in range(n_matches):
        result = build_engine()
        # result has .score_home, .score_away, possession etc. — adapt
        # to the actual MatchResult interface.
        home = getattr(result, "score_home", getattr(result, "home_goals", 0))
        away = getattr(result, "score_away", getattr(result, "away_goals", 0))
        scores.append(float(home) / (float(away) + 1.0))
    return statistics.mean(scores)
