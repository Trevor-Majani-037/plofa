"""BrainEvolution — genetic algorithm engine for FootballBrain.

This module grows the population of per-player neural networks through
simulated matches.  Fitness is derived from how well each brain makes
decisions across a large sample of realistic game states.

Two evaluation modes:
  1. SYNTHETIC (default, fast)  — scores a brain against thousands of
     random-but-realistic match states generated from geometric
     priors.  No full match engine needed.  Great for fast iteration.
  2. FULL_MATCH (outcome-driven) — hooks into the real MatchEngine for an
     accurate but far slower fitness signal measured from the ENGINE'S OWN
     OUTCOMES: team xG difference, goal difference, possession, the target
     player's turnover rate and his own chance production.  NO hand-authored
     intention rewards — the brain is scored by what actually happens on the
     pitch as the engine's calibrated probability model decides it.  Fitness
     functions: ``extract_outcome_signals`` + ``outcome_fitness``, averaged
     by ``evaluate_full_match``.  Supply your own engine wiring via
     ``build_engine(brain) -> MatchResult`` (see match_probe.build_probe_engine),
     or hand ``outcome_fitness`` to ``evolve(..., fitness_fn=...)`` for a short
     in-engine refinement loop.

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
from perception import perceive, PerceptionConfig
from role_features import V2_INPUT_D, _role_family, build_v2_vector, role_block
from tactics_context import (
    V3_INPUT_D, build_v3_vector, random_tactics_context,
)


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

class GameState:
    __slots__ = ('name',)
    def __init__(self, name):
        self.name = name

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
    game_state = GameState(rng.choice(game_state_names))

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
    game_state = GameState(rng.choice(game_state_names))

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


def _clone_brain(brain: FootballBrain) -> FootballBrain:
    """Copy a brain by cloning its weight arrays (no schema round-trip).

    The GA only needs a weight copy for the "best of generation" snapshot.
    A serialize→deserialize round-trip would validate the mid-run brain
    against its *declared* meta: a schema-v2 role-features chromosome
    (32-wide input) carries the default v1_24d meta until the final result
    is stamped, so validating at this point wrongly rejects it.
    """
    return FootballBrain(
        brain.w1.copy(), brain.b1.copy(),
        brain.w2.copy(), brain.b2.copy(),
        brain.w3.copy(), brain.b3.copy(),
    )


class _FakePlayer:
    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position


def _coords_to_players(coords: List[Tuple[float, float]], prefix: str):
    """v1-style fake players — every actor labelled 'CM' (legacy behaviour).

    Kept byte-identical to the historic corpus generator so a v1 evolution
    run reproduces exactly; role-labelled actors are only used for the
    schema-v2 role-feature corpus (see ``_role_coords_to_players``)."""
    return [_FakePlayer(f"{prefix}{i}", "CM") for i, (x, y) in enumerate(coords)]


# Position-labelled actor pools for the role-feature corpus.  rng.sample
# consumes rng *only* in role mode, so the legacy (role-off) corpus path is
# untouched and remains byte-identical to earlier runs.
_OUTFIELD_TM_POOL = ["CB", "CB", "LB", "RB", "CDM", "CM", "CM",
                     "CAM", "LW", "ST", "RW"]
_OPP_POOL = ["GK", "CB", "CB", "LB", "RB", "CDM", "CM", "CM",
             "CAM", "LW", "ST", "RW"]


def _role_coords_to_players(coords: List[Tuple[float, float]], prefix: str,
                            pool: List[str], rng: random.Random):
    """Fake players with role-labelled positions (for v2 features).

    Role features filter actors by position (opp forwards/wides, our
    backline), so a synthetic corpus must tag each fake actor with a
    plausible canonical position.  Deterministic per seed via ``rng``."""
    labels = rng.sample(pool, len(coords))
    return [_FakePlayer(f"{prefix}{i}", labels[i]) for i, (x, y) in enumerate(coords)]


def _carrier(position: str) -> _FakePlayer:
    """The synthetic ball-carrier — carries only the position label (the
    role block reads ``player.position``; DNA stays unset so the shared
    v1 block's self/DNA features read the same defaults as ``player=None``)."""
    return _FakePlayer(f"carrier_{position}", position)


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

def generate_state_corpus(
    position: str,
    n_states: int,
    seed: int,
    goal_bias: float = 0.0,
    input_size: int = INPUT_SIZE,
    perception_config: Optional[PerceptionConfig] = None,
) -> np.ndarray:
    """Build a deterministic (n_states, input_size) sensor corpus.

    A fraction ``goal_bias`` of the states are drawn from scoring
    situations (final-third / goal-close / central) so rare high-value
    intents (ST SHOOT, CM through-balls, winger crosses) are represented
    in the argmax landscape.  Identical RNG sequence per (position,
    n_states, seed, goal_bias).

    ``input_size`` defaults to the v1 width and reproduces the legacy
    corpus byte-for-byte.  For a schema-v2 role-features brain the caller
    passes the role width (24 + 7 / 24 + 8): the shared 24-d block is
    unchanged and the role tail is derived from role-``position``-labelled
    fake actors, read through the SAME perceived scene when
    ``perception_config`` is enabled (train-through-imperfect-perception
    fidelity).  Unknown-role positions with an oversized ``input_size``
    zero-pad the tail (never crashes, learns nothing from dead inputs).
    """
    rng = random.Random(seed)
    role_mode = input_size != INPUT_SIZE
    tactics_mode = input_size in set(V3_INPUT_D.values())
    sensors = np.empty((n_states, input_size), dtype=np.float64)
    for i in range(n_states):
        if goal_bias > 0.0 and rng.random() < goal_bias:
            st = random_game_state_scoring(rng, position)
        else:
            st = random_game_state(rng, position)

        t_coords = st["teammates"]
        d_coords = st["defenders"]
        all_names = {}
        for idx_t, (tx, ty) in enumerate(t_coords):
            all_names[f"t{idx_t}"] = (tx, ty)
        for idx_d, (dx, dy) in enumerate(d_coords):
            all_names[f"d{idx_d}"] = (dx, dy)
        pe = _DummyPositionEngine(all_names)

        if role_mode:
            teammates = _role_coords_to_players(t_coords, "t",
                                                _OUTFIELD_TM_POOL, rng)
            defenders = _role_coords_to_players(d_coords, "d",
                                                _OPP_POOL, rng)
        else:
            teammates = _coords_to_players(t_coords, "t")
            defenders = _coords_to_players(d_coords, "d")

        if role_mode:
            carrier = _carrier(position)
            if perception_config is not None and perception_config.enabled:
                shared, scene = perceive(
                    carrier, st["x"], st["y"], teammates, defenders, pe,
                    st["under_pressure"], st["attacks_right"], st["game_state"],
                    st["minute"], config=perception_config, return_scene=True)
            else:
                shared = extract_sensors(
                    None, st["x"], st["y"], teammates, defenders, pe,
                    st["under_pressure"], st["attacks_right"], st["game_state"],
                    st["minute"],
                )
                scene = {"teammates": teammates, "defenders": defenders,
                         "position_engine": pe}
            role = role_block(carrier, st["x"], st["y"],
                              scene["teammates"], scene["defenders"],
                              scene["position_engine"], st["attacks_right"])
            if tactics_mode:
                # v3: append a synthetic manager-instruction block sampled
                # from the SAME rng sequence (rolls AFTER the geometry so
                # the v2 / v1 corpus stays byte-identical).
                tactics = random_tactics_context(rng)
                vec = build_v3_vector(shared, role, tactics)
            else:
                vec = build_v2_vector(shared, role)
            if vec.shape[0] != input_size:  # unknown role -> zero-pad tail
                padded = np.zeros(input_size, dtype=np.float64)
                padded[:min(vec.shape[0], input_size)] = vec[:input_size]
                vec = padded
            sensors[i] = vec
        else:
            sensors[i] = extract_sensors(
                None, st["x"], st["y"], teammates, defenders, pe,
                st["under_pressure"], st["attacks_right"], st["game_state"],
                st["minute"],
            )
    return sensors


def synthetic_fitness(
    brain: FootballBrain,
    player_position: str,
    batched_sensors: Optional[np.ndarray] = None,
    surrogate: Optional[Any] = None,
    pop_diversity: float = 0.0,
    n_states: int = 400,
    seed: int = 0,
    goal_bias: float = 0.0,
    input_size: int = INPUT_SIZE,
    perception_config: Optional[PerceptionConfig] = None,
) -> float:
    """Score a brain over a batch of sensor states.

    Either pass a pre-generated ``batched_sensors`` array of shape
    (n_states, input_size) or supply ``n_states``/``seed``/``goal_bias``
    and the corpus is generated deterministically here.  The network
    forward pass is vectorised (one batched matmul per brain instead of
    one Python call per state).

    Per-state reward is either the position reward table (hand-made) or
    — when a ``surrogate`` (a ``FitnessSurrogate``) is supplied — the
    surrogate's learned expected-success for the chosen intent given the
    sensor state.

    Fitness = mean reward of chosen intents, weighted by the network's
    own confidence, minus penalties for indecision / single-intent or
    two-intent collapse (anti-collapse stabilisers), then scaled by a
    behavioural-entropy bonus.  Returns a scalar in roughly [0, 1].

    ``input_size`` / ``perception_config`` forward to the corpus builder
    (schema-v2 role-features width and train-through-imperfect-perception).
    With a role-features brain the surrogate only ever sees the shared
    24-d block (its buckets are all index < 24), so the role tail never
    distorts a learned success table / critic.
    """
    if batched_sensors is None:
        batched_sensors = generate_state_corpus(
            player_position, n_states, seed, goal_bias,
            input_size=input_size, perception_config=perception_config)
    sensors_all = np.asarray(batched_sensors, dtype=np.float64)
    n_states = len(sensors_all)
    if n_states == 0:
        return 0.0

    probs_all = brain.forward(sensors_all)  # (N, OUTPUT_SIZE), one matmul pass

    rewards = np.empty(n_states, dtype=np.float64)
    confidences = np.empty(n_states, dtype=np.float64)
    intent_counts = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    penalties = 0.0

    for i in range(n_states):
        probs = probs_all[i]
        idx = int(np.argmax(probs))
        intent = _INTENT_BY_IDX[idx]
        intent_counts[idx] += 1.0

        if surrogate is not None:
            s_in = sensors_all[i][:24] if input_size != INPUT_SIZE else sensors_all[i]
            reward = surrogate.expected_success(s_in, intent, player_position)
        else:
            reward = _context_reward(intent, player_position, sensors_all[i])
        rewards[i] = reward
        confidences[i] = probs[idx]

        sorted_p = np.sort(probs)[::-1]
        if len(sorted_p) > 1 and (sorted_p[0] - sorted_p[1]) < 0.02:
            penalties += 0.1

    mean_reward = float(rewards.mean())
    mean_conf = float(confidences.mean())
    fitness = mean_reward * (0.5 + 0.5 * mean_conf) - penalties / n_states

    intent_probs = intent_counts / intent_counts.sum()
    intent_probs = intent_probs[intent_probs > 0]
    entropy = -float(np.sum(intent_probs * np.log(intent_probs + 1e-12)))
    max_entropy = math.log(OUTPUT_SIZE)
    bdiv = entropy / max_entropy

    max_share = float(intent_counts.max() / intent_counts.sum())
    dom_pen = max(0.0, max_share - 0.50) * 1.5
    if intent_counts.sum() > 0:
        shares = intent_counts / intent_counts.sum()
        n_effective = int(np.sum(shares > 0.05))
        if n_effective < 6:
            shortfall = (6 - n_effective) / 6.0
            dom_pen += shortfall * 0.75

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
    input_size: int = INPUT_SIZE,
    perception_config: Optional[PerceptionConfig] = None,
    role_family: Optional[str] = None,
    fitness_fn: Optional[Callable[[FootballBrain], float]] = None,
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
    surrogate : optional
        Learned FitnessSurrogate / CriticSurrogate driving expected-success.
    goal_bias : float
        Fraction of sampled states drawn from scoring situations.
    input_size : int
        Sensor width this brain consumes — INPUT_SIZE (24, v1) or the
        schema-v2 role-features width (24 + 7 / 24 + 8).  When larger
        than 24 the initial population and the synthetic corpus are built
        role-aware and the saved best brain carries a ``v2_role_features``
        meta block.
    perception_config : optional
        When enabled, the synthetic corpus is generated THROUGH the
        imperfect perception layer (training fidelity for a challenger
        that will gate under the closed Step-4 rule).
    role_family : optional
        Informational role family stamped into the v2 meta (defaults to
        ``_role_family(position)`` when ``input_size`` > 24).
    fitness_fn : optional
        Overrides the fitness signal entirely.  When given, each brain is
        scored ``fitness_fn(brain)`` instead of ``synthetic_fitness``; the
        synthetic corpus is never generated and ``surrogate``/``goal_bias``
        are ignored.  Use it to drive a SHORT in-engine refinement loop
        (population of ~4-8, few generations, with ``evaluate_full_match``
        behind the callable) — a full 32x40 real-match GA is ~7 h/position.

    Returns
    -------
    EvolutionResult with best brain + history.
    """
    rng = random.Random(seed)
    pop = [FootballBrain.random(seed=i, input_size=input_size)
           for i in range(population_size)]

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

        # PRE-GENERATE STATE CORPUS (vectorised forward + identical RNG
        # sequence per brain: seed + gen * 1000 + i) UNLESS an external
        # fitness_fn owns the signal.
        if fitness_fn is None:
            corpus_sensors = [
                generate_state_corpus(position, n_states, seed + gen * 1000 + i,
                                      goal_bias=goal_bias, input_size=input_size,
                                      perception_config=perception_config)
                for i in range(population_size)
            ]

        # evaluate
        fitnesses = []
        for i, brain in enumerate(pop):
            if fitness_fn is not None:
                f = float(fitness_fn(brain))
            else:
                f = synthetic_fitness(brain, position, batched_sensors=corpus_sensors[i],
                                      surrogate=surrogate,
                                      pop_diversity=pop_div,
                                      input_size=input_size)
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
            best_overall = _clone_brain(pop[best_idx])

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

    best = best_overall if best_overall is not None else pop[0]
    if input_size != INPUT_SIZE:
        # Stamp the schema lineage so loaders route this brain to the
        # right sensor builder (and reject it to a v1-only loader).
        from brain_schema import brain_meta_dict
        if input_size in set(V3_INPUT_D.values()):
            sensor_schema = "v3_tactics_context"
        else:
            sensor_schema = "v2_role_features"
        best.meta = brain_meta_dict(
            training_method="ga_surrogate",
            sensor_schema=sensor_schema,
            role_family=role_family or _role_family(position),
        )
    return EvolutionResult(
        best_brain=best,
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )


# ─────────────────────────────────────────────────────────────
# FULL-MATCH FITNESS — engine outcomes, not intention rewards
# ─────────────────────────────────────────────────────────────

# Signal weights: the fitness blend when a candidate brain is scored by the
# real engine.  All terms come from MatchEngine outcomes — none of the
# hand-authored POSITION_REWARDS leak in here.  xG diff carries the most
# weight because it is the engine's own expected-value/chance-quality model
# (lower variance than goals) and is directly "how good were the chances".
OUTCOME_DEFAULT_WEIGHTS: Dict[str, float] = {
    "xg_diff": 0.45,     # engine threat-model yield (chance quality)
    "turnover": 0.20,    # retention: fewer brain-attributed errors -> better
    "goal_diff": 0.15,   # sparse but real
    "possession": 0.10,  # cumulative control
    "chances": 0.10,     # the candidate's own shot production
}

# Event names that count as a player "creating a chance" for himself.
_OUTCOME_SHOT_EVENTS = {
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "HIT_WOODWORK",
    "GOAL", "FREEKICK_DIRECT", "PENALTY_SCORED", "PENALTY_MISSED",
}


@dataclass
class OutcomeFitnessConfig:
    """Normalisation + weighting knobs for outcome-driven fitness.

    Each signal is mapped into an achievement in [0, 1] and blended by
    ``weights``; the final scalar is then renormalised by the total weight.

    - xg_diff / goal_diff are centred at 0 (0.5 => parity) and saturate at
      the respective cap (an xG margin of ``xg_cap`` scores 1.0).
    - possession maps linearly from 0-100%.
    - turnover maps linearly from 0 to ``turnover_tolerance`` (an error on
      > 20% of one's own decided touches scores 0 on that axis).
    - chances: the candidate's own shot-esque events, saturating at
      ``chances_cap``.
    """
    weights: Dict[str, float] = field(default_factory=lambda: dict(OUTCOME_DEFAULT_WEIGHTS))
    xg_cap: float = 3.2
    goal_cap: float = 4.0
    turnover_tolerance: float = 0.20
    chances_cap: int = 8


def extract_outcome_signals(result: Any,
                           target_player: Optional[str] = None,
                           team: Optional[str] = None) -> Dict[str, float]:
    """Pull pure engine-outcome scalars for one candidate slot from a MatchResult.

    ``target_player`` (optional) is the candidate's exact name.  When given,
    the side the candidate plays for is inferred from his own timeline rows
    (falls back to home), and TWO brain-attributed signals are produced from
    the CARRY ``active_brain`` metadata: ``turnover_rate`` (fraction of his
    decided touches flagged ``is_error``) and ``own_shots`` (his shot-esque
    events).  When omitted, turnover_rate is 0 and own_shots is empty — the
    team-wide outcome signals still work.

    ``team`` (optional, ``"home"`` or ``"away"``) overrides the automatic
    side inference.  Use it when both squads share player names (the probe
    template names both STs ``"ST"``) so the timeline scan doesn't pick the
    wrong side's first occurrence.

    Returns a dict with keys: team, xg_diff, goal_diff, possession,
    turnover_rate, n_touches, n_errors, own_shots.
    """
    home_goals = float(getattr(result, "home_goals", 0) or 0)
    away_goals = float(getattr(result, "away_goals", 0) or 0)
    home_xg = float(getattr(result, "home_xg", 0.0) or 0.0)
    away_xg = float(getattr(result, "away_xg", 0.0) or 0.0)
    poss_home = float(getattr(result, "home_possession_pct", 50.0) or 50.0)
    timeline = list(getattr(result, "timeline", []) or [])

    def _name(ev: Any) -> str:
        pl = getattr(ev, "player", None)
        if isinstance(pl, str):
            return pl
        return str(getattr(pl, "name", "") or "")

    # side inference from the candidate's own rows; home is the fallback
    if team is not None:
        pass  # caller knows the side (e.g. duplicate names across squads)
    elif target_player:
        for ev in timeline:
            if _name(ev) == target_player:
                t = getattr(ev, "team", "") or "home"
                team = t if t else "home"
                break
    team = team or "home"
    team = str(team).title() if str(team).lower() in ("home", "away") else str(team)

    if str(team).lower().startswith(("home", "left", "1")):
        team_goals, opp_goals, team_xg, opp_xg, team_poss = (
            home_goals, away_goals, home_xg, away_xg, poss_home)
    else:
        team_goals, opp_goals, team_xg, opp_xg, team_poss = (
            away_goals, home_goals, away_xg, home_xg, 100.0 - poss_home)

    touches = errors = own_shots = 0
    if target_player:
        for ev in timeline:
            if _name(ev) != target_player:
                continue
            md = getattr(ev, "metadata", None) or {}
            if "active_brain" in md:
                touches += 1
                ab = md.get("active_brain") or {}
                if ab.get("is_error"):
                    errors += 1
            etype = getattr(getattr(ev, "event_type", None), "name", "")
            if etype in _OUTCOME_SHOT_EVENTS:
                own_shots += 1

    return {
        "team": team,
        "xg_diff": team_xg - opp_xg,
        "goal_diff": team_goals - opp_goals,
        "possession": team_poss,
        "turnover_rate": (errors / touches) if touches else 0.0,
        "n_touches": touches,
        "n_errors": errors,
        "own_shots": own_shots,
    }


def outcome_fitness(signals: Dict[str, float],
                    config: Optional[OutcomeFitnessConfig] = None) -> float:
    """Blend extracted outcome signals into a single fitness in [0, 1]."""
    cfg = config or OutcomeFitnessConfig()
    w = {k: float(cfg.weights.get(k, 0.0)) for k in OUTCOME_DEFAULT_WEIGHTS}
    total = sum(w.values())
    if total <= 0.0:
        return 0.0

    xg_sig = float(min(1.0, max(0.0, 0.5 + signals["xg_diff"] / cfg.xg_cap)))
    goal_sig = float(min(1.0, max(0.0, 0.5 + signals["goal_diff"] / cfg.goal_cap)))
    poss_sig = float(min(1.0, max(0.0, signals["possession"] / 100.0)))
    tol = max(1e-6, cfg.turnover_tolerance)
    retain_sig = float(min(1.0, max(0.0, 1.0 - signals["turnover_rate"] / tol)))
    chance_sig = float(min(1.0, signals.get("own_shots", 0) / max(1, cfg.chances_cap)))

    blended = (
        w["xg_diff"] * xg_sig + w["turnover"] * retain_sig
        + w["goal_diff"] * goal_sig + w["possession"] * poss_sig
        + w["chances"] * chance_sig
    )
    return float(min(1.0, max(0.0, blended / total)))


def evaluate_full_match(brain: FootballBrain,
                        build_engine: Callable[[FootballBrain], Any],
                        n_matches: int = 3,
                        config: Optional[OutcomeFitnessConfig] = None,
                        target_player: Optional[str] = None,
                        team: Optional[str] = None) -> float:
    """Score a candidate brain by the real engine's own outcomes.

    Parameters
    ----------
    brain : FootballBrain
        The candidate to evaluate.
    build_engine : Callable[[FootballBrain], Any]
        A callable that takes the candidate brain, wires it into a
        MatchEngine (registering it under the exact on-pitch player name so
        ``get_brain`` resolves it), runs ``simulate()`` and returns the
        MatchResult.  Provide it with ``match_probe.build_probe_engine`` or
        your own closure.  The brain is passed per-call so the SAME wiring
        can evaluate every population member (usable as ``evolve``'s
        ``fitness_fn``).
    n_matches : int
        Number of real matches to average over.  Each takes ~15-20s.
    config : optional
        ``OutcomeFitnessConfig`` normalisation/weighting.
    target_player : optional
        Exact name of the candidate's on-pitch player, used to attribute
        turnovers and own-chance creation (and infer his side).
    team : optional
        Which side the candidate plays for (``"home"``/``"away"``).
        When given, skips the fragile timeline inference — needed when both
        squads reuse the same player name (probe template).

    Returns
    -------
    float — averaged outcome fitness in [0, 1].
    """
    cfg = config or OutcomeFitnessConfig()
    fits = []
    n = max(1, int(n_matches))
    for _ in range(n):
        result = build_engine(brain)
        signals = extract_outcome_signals(result, target_player=target_player,
                                          team=team)
        fits.append(outcome_fitness(signals, cfg))
    return statistics.mean(fits) if fits else 0.0
