"""
PLOFA 26/27 — MANAGER BRAIN
====================================================
manager_brain.py

A small feedforward neural network that serves as a manager's decision core.

Architecture: 12 sensors → 32 hidden → 32 hidden → 5 raw outputs.
  - outputs[0:3]  → softmax → posture probabilities (DEFEND / BALANCED / ATTACK)
  - outputs[3]    → sigmoid → pressing intensity
  - outputs[4]    → sigmoid → sub urgency

Not wired into the engine yet (Phase 10 toggle).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Constants ───────────────────────────────────────────────────────

MANAGER_ARCH = (12, 32, 32, 5)
MANAGER_POSTURE_LABELS = ("DEFEND", "BALANCED", "ATTACK")

SENSOR_VERSION = 1
ARCH_VERSION = 1


# ── Micro-nn helpers (standalone, no external imports) ───────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)


def _dna_seeded_weights(
    vision: float, composure: float, decisions: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate initial weights biased by a manager's DNA attributes.

    Vision biases w1 (sensory layer), composure biases w2 (internal state),
    decisions biases w3 (action selection) — matching football_brain.py's
    pattern but adapted for 12 inputs and 5 outputs.
    """
    vision_scale = 0.8 + vision * 0.4
    composure_scale = 1.2 - composure * 0.4
    decisions_scale = 0.9 + decisions * 0.2

    w1 = _he_init(12, 32, rng) * vision_scale
    b1 = np.zeros(32, dtype=np.float64)
    w2 = _he_init(32, 32, rng) * composure_scale
    b2 = np.zeros(32, dtype=np.float64)
    w3 = _he_init(32, 5, rng) * decisions_scale
    b3 = np.zeros(5, dtype=np.float64)
    return w1, b1, w2, b2, w3, b3


# ─────────────────────────────────────────────────────────────────────
# MANAGER BRAIN
# ─────────────────────────────────────────────────────────────────────

class ManagerBrain:
    """A small feedforward network for a manager's in-match decisions.

    Takes the 12-float manager sensor vector and produces posture
    probabilities, pressing intensity and sub urgency.

    Not wired into the engine yet (USE_MANAGER_BRAIN toggle, Phase 10).
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

    # ── Forward pass ────────────────────────────────────────────

    def forward(self, sensors: np.ndarray) -> dict:
        """Run the sensor vector(s) through the network.

        Parameters
        ----------
        sensors : np.ndarray
            Shape (12,) for a single reading or (N, 12) for a batch.

        Returns
        -------
        dict with keys:
            posture_probs : np.ndarray  (3,) or (N, 3)
            pressing      : float       (scalar) or (N,) array
            sub_urgency   : float       (scalar) or (N,) array
        """
        single = sensors.ndim == 1
        x = np.asarray(sensors, dtype=np.float64)
        if single:
            x = x[None, :]

        h = _relu(x @ self.w1 + self.b1)
        h = _relu(h @ self.w2 + self.b2)
        logits = h @ self.w3 + self.b3        # (..., 5)

        posture_probs = _softmax(logits[..., 0:3])
        pressing = _sigmoid(logits[..., 3])
        sub_urgency = _sigmoid(logits[..., 4])

        return {
            "posture_probs": posture_probs[0] if single else posture_probs,
            "pressing": float(pressing[0]) if single else pressing,
            "sub_urgency": float(sub_urgency[0]) if single else sub_urgency,
        }

    # ── Evolution operators ─────────────────────────────────────

    def mutate(self, rate: float = 0.15, strength: float = 0.3) -> ManagerBrain:
        rng = np.random.default_rng()

        def _perturb(w: np.ndarray) -> np.ndarray:
            mask = rng.random(w.shape) < rate
            noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
            return w + mask * noise

        return ManagerBrain(
            _perturb(self.w1), _perturb(self.b1),
            _perturb(self.w2), _perturb(self.b2),
            _perturb(self.w3), _perturb(self.b3),
        )

    def crossover(self, other: ManagerBrain) -> ManagerBrain:
        rng = np.random.default_rng()

        def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b)

        return ManagerBrain(
            _cross(self.w1, other.w1), _cross(self.b1, other.b1),
            _cross(self.w2, other.w2), _cross(self.b2, other.b2),
            _cross(self.w3, other.w3), _cross(self.b3, other.b3),
        )

    def blend(self, other: ManagerBrain, alpha: float = 0.5) -> ManagerBrain:
        a, b = alpha, 1.0 - alpha
        return ManagerBrain(
            self.w1 * a + other.w1 * b, self.b1 * a + other.b1 * b,
            self.w2 * a + other.w2 * b, self.b2 * a + other.b2 * b,
            self.w3 * a + other.w3 * b, self.b3 * a + other.b3 * b,
        )

    # ── Random / DNA seeding ────────────────────────────────────

    @classmethod
    def random(cls, seed: Optional[int] = None) -> ManagerBrain:
        rng = np.random.default_rng(seed)
        w1 = _he_init(12, 32, rng)
        b1 = np.zeros(32, dtype=np.float64)
        w2 = _he_init(32, 32, rng)
        b2 = np.zeros(32, dtype=np.float64)
        w3 = _he_init(32, 5, rng)
        b3 = np.zeros(5, dtype=np.float64)
        return cls(w1, b1, w2, b2, w3, b3)

    @classmethod
    def from_dna(cls, player_dna: Any, seed: Optional[int] = None) -> ManagerBrain:
        """Create a brain whose initial weights are biased by a player's DNA.

        Accepts either a PlayerDNA object (with .mental attributes) or a
        Player object (accessed via .dna).  Returns a ManagerBrain whose
        input weights are vision-biased, hidden weights are composure-biased
        and output weights are decisions-biased.
        """
        dna = getattr(player_dna, "dna", player_dna)  # accept player or dna
        mental = getattr(dna, "mental", None)

        def _attr(obj: Any, path: str, default: float) -> float:
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
        w1, b1, w2, b2, w3, b3 = _dna_seeded_weights(
            vision, composure, decisions, rng,
        )
        return cls(w1, b1, w2, b2, w3, b3)

    # ── Serialization ───────────────────────────────────────────

    def serialize(self) -> Dict[str, Any]:
        return {
            "arch": list(MANAGER_ARCH),
            "kind": "manager_brain",
            "sensor_version": SENSOR_VERSION,
            "arch_version": ARCH_VERSION,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> ManagerBrain:
        kind = data.get("kind", "")
        if kind != "manager_brain":
            raise ValueError(f"expected kind='manager_brain', got {kind!r}")

        arch = data.get("arch")
        if arch != list(MANAGER_ARCH):
            raise ValueError(
                f"arch mismatch: expected {list(MANAGER_ARCH)}, got {arch}")

        sv = data.get("sensor_version")
        if sv != SENSOR_VERSION:
            raise ValueError(
                f"sensor_version mismatch: expected {SENSOR_VERSION}, got {sv}")

        av = data.get("arch_version")
        if av != ARCH_VERSION:
            raise ValueError(
                f"arch_version mismatch: expected {ARCH_VERSION}, got {av}")

        arrays = [
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
        ]
        return cls(*arrays)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> ManagerBrain:
        with open(path, "r", encoding="utf-8") as f:
            return cls.deserialize(json.load(f))

    # ── Info ────────────────────────────────────────────────────

    @property
    def param_count(self) -> int:
        return (
            self.w1.size + self.b1.size
            + self.w2.size + self.b2.size
            + self.w3.size + self.b3.size
        )

    def __repr__(self) -> str:
        return f"ManagerBrain(arch={list(MANAGER_ARCH)}, {self.param_count} params)"