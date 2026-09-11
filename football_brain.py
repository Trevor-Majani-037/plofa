"""Per-player feedforward neural network — the evolved football brain.

Architecture
------------
    Input(24) → Dense(32, ReLU) → Dense(32, ReLU) → Dense(10, Softmax)

Each player owns a separate FootballBrain instance.  The weights are
identical in *shape* across all players but differ in *value* — seeded
by the player's DNA attributes and refined through evolutionary
training (see brain_evolution.py).

No gradient learning.  Pure numpy forward pass.  Evolution via
genetic algorithm handles weight optimization.

Design contract
---------------
    • forward(sensor_vector) → intent probabilities (10-float softmax)
    • mutate(rate, strength) → new brain with perturbed weights
    • crossover(other) → new brain blended from two parents
    • serialize() / deserialize() → JSON round-trip
    • from_dna(player_dna) → brain seeded from player attributes
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

INPUT_SIZE = 24
HIDDEN_1 = 32
HIDDEN_2 = 32
OUTPUT_SIZE = 10

INTENT_LABELS = [
    "PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
    "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION",
]


# ─────────────────────────────────────────────────────────────
# ACTIVATIONS
# ─────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────
# WEIGHT INITIALIZATION
# ─────────────────────────────────────────────────────────────

def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialization — good default for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)


def _dna_seeded_weights(
    vision: float, composure: float, decisions: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate initial weights biased by player DNA attributes.

    Higher vision → wider input-to-hidden receptive field (larger magnitudes
    on vision-related input rows).  Higher composure → smaller initial
    hidden-layer magnitudes (calmer internal state).  Higher decisions →
    slightly sharper output layer (more decisive softmax).
    """
    vision_scale = 0.8 + vision * 0.4
    composure_scale = 1.2 - composure * 0.4
    decisions_scale = 0.9 + decisions * 0.2

    w1 = _he_init(INPUT_SIZE, HIDDEN_1, rng) * vision_scale
    b1 = np.zeros(HIDDEN_1, dtype=np.float64)
    w2 = _he_init(HIDDEN_1, HIDDEN_2, rng) * composure_scale
    b2 = np.zeros(HIDDEN_2, dtype=np.float64)
    w3 = _he_init(HIDDEN_2, OUTPUT_SIZE, rng) * decisions_scale
    b3 = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    return w1, b1, w2, b2, w3, b3


# ─────────────────────────────────────────────────────────────
# FOOTBALL BRAIN
# ─────────────────────────────────────────────────────────────

class FootballBrain:
    """A small feedforward neural network that acts as one player's brain.

    The network takes 24 vision-sensor floats (ball position, nearest
    defender distance, teammate openness, fatigue, etc.) and outputs
    10 intent probabilities — one per action type in PlayerIntent.

    Weights are initialized from the player's DNA attributes and
    refined through evolutionary training.  The forward pass is
    deterministic given the same weights and inputs — all stochasticity
    comes from the sensor layer (which has access to live match geometry
    that varies per touch).
    """

    __slots__ = ("w1", "b1", "w2", "b2", "w3", "b3", "_rng")

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
        self._rng = np.random.default_rng(seed)

    # ── Forward pass ────────────────────────────────────────────

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Run the vision vector through the network.

        Parameters
        ----------
        sensors : np.ndarray, shape (24,)
            Normalised sensor vector from brain_sensors.extract_sensors().

        Returns
        -------
        np.ndarray, shape (10,)
            Softmax probability distribution over the 10 intents.
        """
        h = _relu(sensors @ self.w1 + self.b1)
        h = _relu(h @ self.w2 + self.b2)
        logits = h @ self.w3 + self.b3
        return _softmax(logits)

    def predict(self, sensors: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Forward pass + argmax.

        Returns (intent_index, probability, full_distribution).
        """
        probs = self.forward(sensors)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs

    # ── DNA seeding ─────────────────────────────────────────────

    @classmethod
    def from_dna(cls, player: Any, seed: Optional[int] = None) -> FootballBrain:
        """Create a brain whose initial weights are biased by player DNA.

        Parameters
        ----------
        player : PlayerProfile
            Must have .dna.mental.vision/composure/decisions.
        seed : int, optional
            RNG seed for reproducibility.
        """
        dna = getattr(player, "dna", None)
        mental = getattr(dna, "mental", None)

        def _attr(obj, path: str, default: float) -> float:
            cur = obj
            for part in path.split("."):
                if cur is None:
                    return default
                cur = getattr(cur, part, None)
            return float(cur) if cur is not None else default

        vision = _attr(mental, "vision", 55.0) / 100.0
        composure = _attr(mental, "composure", 55.0) / 100.0
        decisions = _attr(mental, "decisions", 55.0) / 100.0

        rng = np.random.default_rng(seed)
        w1, b1, w2, b2, w3, b3 = _dna_seeded_weights(vision, composure, decisions, rng)
        return cls(w1, b1, w2, b2, w3, b3, seed=seed)

    # ── Evolution operators ─────────────────────────────────────

    def mutate(self, rate: float = 0.15, strength: float = 0.3) -> FootballBrain:
        """Return a new brain with randomly perturbed weights.

        Parameters
        ----------
        rate : float
            Fraction of weights to mutate (0..1).
        strength : float
            Standard deviation of the Gaussian perturbation, relative
            to the current weight magnitude.
        """
        rng = np.random.default_rng()
        mask = lambda shape: rng.random(shape) < rate

        def _perturb(w: np.ndarray) -> np.ndarray:
            m = mask(w.shape)
            noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
            return w + m * noise

        return FootballBrain(
            _perturb(self.w1), _perturb(self.b1),
            _perturb(self.w2), _perturb(self.b2),
            _perturb(self.w3), _perturb(self.b3),
        )

    def crossover(self, other: FootballBrain) -> FootballBrain:
        """Uniform crossover — each weight comes from either parent."""
        rng = np.random.default_rng()

        def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b)

        return FootballBrain(
            _cross(self.w1, other.w1), _cross(self.b1, other.b1),
            _cross(self.w2, other.w2), _cross(self.b2, other.b2),
            _cross(self.w3, other.w3), _cross(self.b3, other.b3),
        )

    def blend(self, other: FootballBrain, alpha: float = 0.5) -> FootballBrain:
        """Blend weights: self * alpha + other * (1 - alpha)."""
        a, b = alpha, 1.0 - alpha
        return FootballBrain(
            self.w1 * a + other.w1 * b, self.b1 * a + other.b1 * b,
            self.w2 * a + other.w2 * b, self.b2 * a + other.b2 * b,
            self.w3 * a + other.w3 * b, self.b3 * a + other.b3 * b,
        )

    # ── Random brain (for initial population) ───────────────────

    @classmethod
    def random(cls, seed: Optional[int] = None) -> FootballBrain:
        """Create a brain with random He-initialized weights."""
        rng = np.random.default_rng(seed)
        w1 = _he_init(INPUT_SIZE, HIDDEN_1, rng)
        b1 = np.zeros(HIDDEN_1, dtype=np.float64)
        w2 = _he_init(HIDDEN_1, HIDDEN_2, rng)
        b2 = np.zeros(HIDDEN_2, dtype=np.float64)
        w3 = _he_init(HIDDEN_2, OUTPUT_SIZE, rng)
        b3 = np.zeros(OUTPUT_SIZE, dtype=np.float64)
        return cls(w1, b1, w2, b2, w3, b3, seed=seed)

    # ── Serialization ───────────────────────────────────────────

    def serialize(self) -> Dict[str, Any]:
        """Pack weights into a JSON-serializable dict."""
        return {
            "arch": [INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE],
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> FootballBrain:
        """Load a brain from a serialized dict."""
        return cls(
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
        )

    def save(self, path: str) -> None:
        """Write brain weights to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> FootballBrain:
        """Read brain weights from a JSON file."""
        with open(path, "r") as f:
            return cls.deserialize(json.load(f))

    # ── Info ────────────────────────────────────────────────────

    @property
    def param_count(self) -> int:
        """Total number of learnable parameters."""
        return (
            self.w1.size + self.b1.size
            + self.w2.size + self.b2.size
            + self.w3.size + self.b3.size
        )

    def __repr__(self) -> str:
        return (
            f"FootballBrain("
            f"{INPUT_SIZE}>{HIDDEN_1}>{HIDDEN_2}>{OUTPUT_SIZE}, "
            f"{self.param_count} params)"
        )


# ─────────────────────────────────────────────────────────────
# OFF-BALL CONSCIENCE BRAIN (binary press/hold gate)
# ─────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OffBallBrain:
    """24->32->32->1 feedforward net that gates OFF-BALL press decisions.

    Reads the same 24-d sensor vector (ball 0-1 / runner 2-3 decoupled) and
    emits a single sigmoid logit: p(press).  The deterministic 10 Hz shape
    integration in ``MatchEngine._offball_move_player`` stays authoritative;
    this net only replaces the ``cst['allow'] = 1.0 if random < _PRESS_PROB[pos]
    else -1.0`` Bernoulli (match_engine.py:1996).  Press when output > 0.5.

    Evolved per position against the off-ball surrogate (offball_probe.
    OffBallSurrogate) — same GA operators as FootballBrain, no gradients.
    """

    __slots__ = ("w1", "b1", "w2", "b2", "w3", "b3")

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3

    def forward(self, sensors: np.ndarray) -> float:
        """Sigmoid probability of pressing, given a 24-d off-ball vector."""
        h = _relu(sensors @ self.w1 + self.b1)
        h = _relu(h @ self.w2 + self.b2)
        logit = float((h @ self.w3 + self.b3)[0])
        return float(_sigmoid(np.asarray(logit, dtype=np.float64)))

    def decide(self, sensors: np.ndarray) -> bool:
        """True -> commit the press burst; False -> hold shape."""
        return self.forward(sensors) > 0.5

    def mutate(self, rate: float = 0.15, strength: float = 0.3) -> "OffBallBrain":
        rng = np.random.default_rng()
        mask = lambda shape: rng.random(shape) < rate

        def _perturb(w: np.ndarray) -> np.ndarray:
            m = mask(w.shape)
            noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
            return w + m * noise

        return OffBallBrain(
            _perturb(self.w1), _perturb(self.b1),
            _perturb(self.w2), _perturb(self.b2),
            _perturb(self.w3), _perturb(self.b3),
        )

    def crossover(self, other: "OffBallBrain") -> "OffBallBrain":
        rng = np.random.default_rng()

        def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b)

        return OffBallBrain(
            _cross(self.w1, other.w1), _cross(self.b1, other.b1),
            _cross(self.w2, other.w2), _cross(self.b2, other.b2),
            _cross(self.w3, other.w3), _cross(self.b3, other.b3),
        )

    def blend(self, other: "OffBallBrain", alpha: float = 0.5) -> "OffBallBrain":
        a, b = alpha, 1.0 - alpha
        return OffBallBrain(
            self.w1 * a + other.w1 * b, self.b1 * a + other.b1 * b,
            self.w2 * a + other.w2 * b, self.b2 * a + other.b2 * b,
            self.w3 * a + other.w3 * b, self.b3 * a + other.b3 * b,
        )

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "OffBallBrain":
        rng = np.random.default_rng(seed)
        w1 = _he_init(INPUT_SIZE, HIDDEN_1, rng)
        b1 = np.zeros(HIDDEN_1, dtype=np.float64)
        w2 = _he_init(HIDDEN_1, HIDDEN_2, rng)
        b2 = np.zeros(HIDDEN_2, dtype=np.float64)
        w3 = _he_init(HIDDEN_2, 1, rng)
        b3 = np.zeros(1, dtype=np.float64)
        return cls(w1, b1, w2, b2, w3, b3)

    def serialize(self) -> Dict[str, Any]:
        return {
            "arch": [INPUT_SIZE, HIDDEN_1, HIDDEN_2, 1],
            "kind": "offball_press_gate",
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "OffBallBrain":
        return cls(
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OffBallBrain":
        with open(path, "r") as f:
            return cls.deserialize(json.load(f))

    @property
    def param_count(self) -> int:
        return (
            self.w1.size + self.b1.size
            + self.w2.size + self.b2.size
            + self.w3.size + self.b3.size
        )

    def __repr__(self) -> str:
        return (
            f"OffBallBrain("
            f"{INPUT_SIZE}>{HIDDEN_1}>{HIDDEN_2}>1, "
            f"{self.param_count} params)"
        )


class TeamPressBrain(OffBallBrain):
    """Shared XI-wide press-engagement controller.

    Same 24->32->32->1 topology as OffBallBrain, but consumes a TEAM-level
    24-d sensor vector (ball zone vs our goal, danger, unit density, opp
    support, shape compactness, score/minutes/phase) and emits ONE engagement
    scalar g in [0,1] for the whole unit:

        g ~ 1  -> every near-ball defender presses at his role rate at the
                  same time (a coordinated collective press; the per-position
                  Bernoulli at match_engine.py:1996 becomes
                  ``prob = _PRESS_PROB[pos] * g``)
        g ~ 0  -> nobody leaves the shape; the block compacts (SIT).

    This is the team-level lever the individual-conscience probe proved was
    the right size: pressing is a unit act, not a per-shirt coin flip.
    Evolved against the TeamPressSurrogate (team-state bucket -> engagement
    band -> expected defensive-opportunity success).
    """

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        d["kind"] = "team_press_engagement"
        return d

    @classmethod
    def load(cls, path: str) -> "TeamPressBrain":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
        )


# Defensive action labels — output index order of DefensiveActionBrain.
DEFENSIVE_ACTIONS = ("tackle", "interception", "clearance", "block")


class DefensiveActionBrain:
    """Shared XI defensive-action controller (24->32->32->4).

    When a defensive contest erupts (a real press/tackle/challenge moment in
    open play or a deep recovery), the engine must pick WHICH act to deploy:
    a standing/sliding TACKLE, an INTERCEPTION pass lane, a CLEARANCE away
    from our goal, or throwing the body at a shot BLOCK.  The old code chose
    from four hand-tuned weight rows keyed only on ``danger``
    (match_engine._danger_scaled_action_weights).  This net replaces that
    table: it reads a 24-d DEFENSIVE-state vector (ball zone vs our goal,
    danger, ball aerial/ground, contest locality, attacker/defender density
    at the ball, who the nearest defender is, unit compactness, score/phase,
    clearance feasibility) and emits a softmax over
    [tackle, interception, clearance, block].

    Feasibility SAFETY GRIP (not a weight): the engine still refuses
    clearance/block far from the defending goal (a "clearance" 60 m from
    your own line is just a turnover — same as the AttackingMatrix never
    lets the on-ball brain fire a shot from its own half).  Within the
    feasible zone, THIS brain is the sole selector.

    Evolved against the defensive-action surrogate (defensive_action_probe.
    DefensiveActionSurrogate), same GA operators as FootballBrain, numpy only.
    """

    __slots__ = ("w1", "b1", "w2", "b2", "w3", "b3")

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Softmax probabilities over DEFENSIVE_ACTIONS (4-d, sums to 1)."""
        h = _relu(sensors @ self.w1 + self.b1)
        h = _relu(h @ self.w2 + self.b2)
        logits = h @ self.w3 + self.b3
        z = logits - float(np.max(logits))
        e = np.exp(z)
        return e / e.sum()

    def predict(self, sensors: np.ndarray) -> str:
        """Argmax action label."""
        return DEFENSIVE_ACTIONS[int(np.argmax(self.forward(sensors)))]

    def mutate(self, rate: float = 0.15, strength: float = 0.3) -> "DefensiveActionBrain":
        rng = np.random.default_rng()
        mask = lambda shape: rng.random(shape) < rate

        def _perturb(w: np.ndarray) -> np.ndarray:
            m = mask(w.shape)
            noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
            return w + m * noise

        return DefensiveActionBrain(
            _perturb(self.w1), _perturb(self.b1),
            _perturb(self.w2), _perturb(self.b2),
            _perturb(self.w3), _perturb(self.b3),
        )

    def crossover(self, other: "DefensiveActionBrain") -> "DefensiveActionBrain":
        rng = np.random.default_rng()

        def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b)

        return DefensiveActionBrain(
            _cross(self.w1, other.w1), _cross(self.b1, other.b1),
            _cross(self.w2, other.w2), _cross(self.b2, other.b2),
            _cross(self.w3, other.w3), _cross(self.b3, other.b3),
        )

    def blend(self, other: "DefensiveActionBrain", alpha: float = 0.5) -> "DefensiveActionBrain":
        a, b = alpha, 1.0 - alpha
        return DefensiveActionBrain(
            self.w1 * a + other.w1 * b, self.b1 * a + other.b1 * b,
            self.w2 * a + other.w2 * b, self.b2 * a + other.b2 * b,
            self.w3 * a + other.w3 * b, self.b3 * a + other.b3 * b,
        )

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "DefensiveActionBrain":
        rng = np.random.default_rng(seed)
        w1 = _he_init(INPUT_SIZE, HIDDEN_1, rng)
        b1 = np.zeros(HIDDEN_1, dtype=np.float64)
        w2 = _he_init(HIDDEN_1, HIDDEN_2, rng)
        b2 = np.zeros(HIDDEN_2, dtype=np.float64)
        w3 = _he_init(HIDDEN_2, len(DEFENSIVE_ACTIONS), rng)
        b3 = np.zeros(len(DEFENSIVE_ACTIONS), dtype=np.float64)
        return cls(w1, b1, w2, b2, w3, b3)

    def serialize(self) -> Dict[str, Any]:
        return {
            "arch": [INPUT_SIZE, HIDDEN_1, HIDDEN_2, len(DEFENSIVE_ACTIONS)],
            "kind": "defensive_action",
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "DefensiveActionBrain":
        return cls(
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DefensiveActionBrain":
        with open(path, "r") as f:
            return cls.deserialize(json.load(f))

    @property
    def param_count(self) -> int:
        return (
            self.w1.size + self.b1.size
            + self.w2.size + self.b2.size
            + self.w3.size + self.b3.size
        )

    def __repr__(self) -> str:
        return (
            f"DefensiveActionBrain("
            f"{INPUT_SIZE}>{HIDDEN_1}>{HIDDEN_2}>{len(DEFENSIVE_ACTIONS)}, "
            f"{self.param_count} params)"
        )


def extract_team_sensors(engine: Any, team: str, ball_x: float, ball_y: float,
                         danger_t: float) -> np.ndarray:
    """24-d TEAM-state vector (0..1) feeding the TeamPressBrain.

    Standalone helper (numpy-only, no match_engine import) so both
    team_offball_probe.py and the live wiring in match_engine.py use the
    SAME feature layout. Index map:

      0 ball_x/105, 1 ball_y/68
      2 ball projection toward OUR goal line (0 our goal .. 1 opp goal)
      3 danger/100
      4 score diff -3..3 -> 0..1 (from OUR side)
      5 minute/90
      6 phase / 4.0
      7 our shape compactness (mean dist of our outfielders to centroid /35)
      8 our outfielders within 15 m of ball /11
      9 opp outfielders within 15 m of ball /11
     10 our outfield centroid x/105
     11 opp outfield centroid x/105
     12-14 ball zone one-hots own/mid/final third (vs our goal)
     15 opp outfielders in our final third /11
     16 our deepest defender projection/105 (goal-side), 1-own-goal-side
     17 space behind our block: clamped opp_centroid - our_centroid
     18 attacks_right (1 if our team attacks right)
     19 opp forwards within 25 m of OUR goal line /4
     20-23 0 (spare)
    """
    pe = engine.position_engine
    ours = [nm for nm in pe.team_rosters.get(team, []) if
            pe.states.get(nm) is not None and
            getattr(pe.states[nm], "position", "") != "GK"]
    opp = [nm for t, names in pe.team_rosters.items() if t != team for nm in names]
    opp = [nm for nm in opp if pe.states.get(nm) is not None and
           getattr(pe.states[nm], "position", "") != "GK"]

    o_x = [pe.states[nm].current_x for nm in ours]
    o_y = [pe.states[nm].current_y for nm in ours]
    d_x = [pe.states[nm].current_x for nm in opp]
    d_y = [pe.states[nm].current_y for nm in opp]

    attr = pe.team_attacks_right.get(team, True)
    own_goal_x = 105.0 if not attr else 0.0
    proj = abs(ball_x - own_goal_x) / 105.0

    centroid_ox = (sum(o_x) / len(o_x)) if o_x else 0.0
    centroid_oy = (sum(o_y) / len(o_y)) if o_y else 34.0
    compact = (sum(((x - centroid_ox) ** 2 + (y - centroid_oy) ** 2) ** 0.5
                   for x, y in zip(o_x, o_y)) / len(o_x)) if o_x else 40.0
    opp_centroid_x = (sum(d_x) / len(d_x)) if d_x else 50.0

    near_ball_ours = sum(1 for x, y in zip(o_x, o_y)
                         if ((x - ball_x) ** 2 + (y - ball_y) ** 2) ** 0.5 <= 15.0)
    near_ball_opp = sum(1 for x, y in zip(d_x, d_y)
                        if ((x - ball_x) ** 2 + (y - ball_y) ** 2) ** 0.5 <= 15.0)

    zone = proj
    own_third = 1.0 if zone < 0.33 else 0.0
    final_third = 1.0 if zone > 0.66 else 0.0
    mid_third = 1.0 if 0.33 <= zone <= 0.66 else 0.0

    opp_in_final = sum(1 for x in d_x
                       if abs(x - own_goal_x) / 105.0 > 0.66) if d_x else 0

    deepest = max((abs(x - own_goal_x) / 105.0) for x in o_x) if o_x else 1.0
    space_behind = max(0.0, min(1.0,
                                (opp_centroid_x - centroid_ox) / 105.0 + 0.5))

    score_diff = engine.state.home_goals - engine.state.away_goals
    if team == getattr(engine.config, "away_team", ""):
        score_diff = -score_diff
    sd = max(-3.0, min(3.0, score_diff))
    phase = 0.0
    try:
        phase = float(engine.state.phase.value) / 4.0
    except Exception:
        phase = 0.2

    opp_goalmouth_25 = sum(1 for x in d_x
                           if abs(x - own_goal_x) <= 25.0) if d_x else 0

    s = np.zeros(24, dtype=np.float64)
    s[0] = ball_x / 105.0
    s[1] = ball_y / 68.0
    s[2] = proj
    s[3] = max(0.0, min(1.0, danger_t / 100.0))
    s[4] = (sd + 3.0) / 6.0
    s[5] = min(1.0, float(engine.state.minute) / 90.0)
    s[6] = phase
    s[7] = min(1.0, compact / 35.0)
    s[8] = near_ball_ours / 11.0
    s[9] = near_ball_opp / 11.0
    s[10] = centroid_ox / 105.0
    s[11] = opp_centroid_x / 105.0
    s[12] = own_third
    s[13] = mid_third
    s[14] = final_third
    s[15] = opp_in_final / 11.0
    s[16] = deepest
    s[17] = space_behind
    s[18] = 1.0 if attr else 0.0
    s[19] = min(1.0, opp_goalmouth_25 / 4.0)
    return s


def extract_defensive_sensors(
    engine: Any,
    defending_team: str,
    attacking_team: str,
    ball_x: float, ball_y: float,
    danger_level: float = 0.0,
    ball_aerial: bool = False,
    contest_x: Optional[float] = None,
    contest_y: Optional[float] = None,
    opponent_distance: Optional[float] = None,
    press_occurred: bool = False,
) -> np.ndarray:
    """24-d DEFENSIVE-state vector (0..1) feeding the DefensiveActionBrain.

    Composed of the shared team-state layout plus contest-LOCAL features that
    the action-type selector genuinely needs.  Index map:

      0 ball_x/105, 1 ball_y/68
      2 ball projection toward OUR goal line (0 our goal .. 1 opp goal)
      3 danger/100
      4 score diff -3..3 -> 0..1 (from OUR side)
      5 minute/90
      6 phase / 4.0
      7 our shape compactness (mean dist to centroid /35)
      8 our outfielders within 15 m of ball /11
      9 opp outfielders within 15 m of ball /11
     10 contest point distance to OUR goal line /105  (clearance feasibility)
     11 ball AERIAL (1) vs ground (0)
     12 nearest-defender distance to the ball (from contest point / 20)
     13 defender:attacker density ratio at contest (0.2 .. 1.5 -> 0..1)
     14 our deepest defender's line (1 = close to own goal)
     15 "a press event already happened this sequence" (1) / not (0)
     16 opp forwards within 25 m of OUR goal /4
     17 space behind our block (opp centroid - our centroid clamp)
     18 attacks_right for defending team
     19 clearance/block FEASIBLE here (ball in our defensive ~40 % / danger)
     20 nearest defender role: CB (1)
     21 nearest defender role: FB  (LB/RB) (1)
     22 nearest defender role: MID (CDM/CM/CAM) (1)
     23 defender unit average stamina /100
    """
    pe = engine.position_engine
    ours = [nm for nm in pe.team_rosters.get(defending_team, []) if
            pe.states.get(nm) is not None and
            getattr(pe.states[nm], "position", "") != "GK"]
    opp = [nm for t, names in pe.team_rosters.items() if t != defending_team for nm in names]
    opp = [nm for nm in opp if pe.states.get(nm) is not None and
           getattr(pe.states[nm], "position", "") != "GK"]

    o_x = [pe.states[nm].current_x for nm in ours]
    o_y = [pe.states[nm].current_y for nm in ours]
    d_x = [pe.states[nm].current_x for nm in opp]
    d_y = [pe.states[nm].current_y for nm in opp]

    attr = pe.team_attacks_right.get(defending_team, True)
    own_goal_x = 105.0 if not attr else 0.0
    proj = abs(ball_x - own_goal_x) / 105.0

    centroid_ox = (sum(o_x) / len(o_x)) if o_x else 0.0
    centroid_oy = (sum(o_y) / len(o_y)) if o_y else 34.0
    compact = (sum(((x - centroid_ox) ** 2 + (y - centroid_oy) ** 2) ** 0.5
                   for x, y in zip(o_x, o_y)) / len(o_x)) if o_x else 40.0
    opp_centroid_x = (sum(d_x) / len(d_x)) if d_x else 50.0

    near_ball_ours = sum(1 for x, y in zip(o_x, o_y)
                         if ((x - ball_x) ** 2 + (y - ball_y) ** 2) ** 0.5 <= 15.0)
    near_ball_opp = sum(1 for x, y in zip(d_x, d_y)
                        if ((x - ball_x) ** 2 + (y - ball_y) ** 2) ** 0.5 <= 15.0)

    cx = contest_x if contest_x is not None else ball_x
    cy = contest_y if contest_y is not None else ball_y
    contest_to_goal = abs(cx - own_goal_x) / 105.0

    nearest_def_dist = 20.0
    nearest_role = "MID"
    for nm, x, y in zip(ours, o_x, o_y):
        d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if d < nearest_def_dist:
            nearest_def_dist = d
            nearest_role = getattr(pe.states[nm], "position", "MID")

    def_ratio = 1.0
    if near_ball_opp > 0:
        def_ratio = near_ball_ours / max(1, near_ball_opp)
    density_ratio = max(0.0, min(1.0, (def_ratio - 0.2) / 1.3))

    deepest = max((abs(x - own_goal_x) / 105.0) for x in o_x) if o_x else 1.0
    space_behind = max(0.0, min(1.0,
                                (opp_centroid_x - centroid_ox) / 105.0 + 0.5))

    score_diff = engine.state.home_goals - engine.state.away_goals
    if defending_team == getattr(engine.config, "away_team", ""):
        score_diff = -score_diff
    sd = max(-3.0, min(3.0, score_diff))
    phase = 0.0
    try:
        phase = float(engine.state.phase.value) / 4.0
    except Exception:
        phase = 0.2

    opp_goalmouth_25 = sum(1 for x in d_x
                           if abs(x - own_goal_x) <= 25.0) if d_x else 0

    feasible_zones = proj < 0.60  # within ~ our defensive 60%
    feasible = 1.0 if (feasible_zones or danger_level >= 30.0) else 0.0

    role_cb = 1.0 if nearest_role in ("CB", "GK") else 0.0
    role_fb = 1.0 if nearest_role in ("LB", "RB", "LWB", "RWB") else 0.0
    role_mid = 1.0 if nearest_role in ("CDM", "CM", "CAM") else 0.0

    stamina_avg = 0.7
    try:
        stamina_vals = [
            engine.sub_controller.stamina[p.name].current_stamina
            for p in (pe.team_rosters.get(defending_team, []) or []) if
            hasattr(engine, "sub_controller") and engine.sub_controller and
            getattr(engine.sub_controller, "stamina", None) and
            p.name in getattr(engine.sub_controller, "stamina", {})
        ]
        if stamina_vals:
            stamina_avg = sum(stamina_vals) / len(stamina_vals) / 100.0
    except Exception:
        stamina_avg = 0.7

    s = np.zeros(24, dtype=np.float64)
    s[0] = ball_x / 105.0
    s[1] = ball_y / 68.0
    s[2] = proj
    s[3] = max(0.0, min(1.0, danger_level / 100.0))
    s[4] = (sd + 3.0) / 6.0
    s[5] = min(1.0, float(engine.state.minute) / 90.0)
    s[6] = phase
    s[7] = min(1.0, compact / 35.0)
    s[8] = near_ball_ours / 11.0
    s[9] = near_ball_opp / 11.0
    s[10] = contest_to_goal
    s[11] = 1.0 if ball_aerial else 0.0
    s[12] = min(1.0, nearest_def_dist / 20.0)
    s[13] = density_ratio
    s[14] = deepest
    s[15] = 1.0 if press_occurred else 0.0
    s[16] = min(1.0, opp_goalmouth_25 / 4.0)
    s[17] = space_behind
    s[18] = 1.0 if attr else 0.0
    s[19] = feasible
    s[20] = role_cb
    s[21] = role_fb
    s[22] = role_mid
    s[23] = max(0.0, min(1.0, stamina_avg))
    return s
