"""GameStateMind — per-team adaptive mindset conditioned on match state.

WHY THIS EXISTS
---------------
The current architecture has:
  * PlayerMind (individual): episodic memory + temperament → modulates sampling
  * TeamPressBrain (team): press engagement scalar g ∈ [0,1]
  * DefensiveActionBrain (shared): which defensive action

What's missing: a *game-state adaptive mindset* that modulates all brains based on
scoreline, time, possession, xG diff. Currently press intensity is the only
game-state-dependent factor, and it's team-level only.

A GameStateMind takes (scorediff, minute, possession%, xG_diff, ...) and
outputs modulation scalars for:
  - press_intensity_mult     (feeds TeamPressBrain / individual press)
  - risk_tolerance           (feeds PlayerMind.temper / intent sampling)
  - defensive_urgency        (feeds DefensiveActionBrain)
  - attacking_directness     (feeds on-ball intent distribution)

This is a small MLP (e.g., 12 → 32 → 32 → 4), trained by distillation
against the surrogate's game-state-conditioned policy.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# DIMENSIONS
# ─────────────────────────────────────────────────────────────

# Input: 12-d game state vector
GAME_STATE_D = 12

# Hidden layers
HIDDEN_1 = 32
HIDDEN_2 = 32

# Output: 4 modulation scalars
OUT_D = 4  # press_mult, risk_tol, def_urgency, att_directness

# Output indices
PRESS_MULT = 0
RISK_TOL = 1
DEF_URGENCY = 2
ATT_DIRECTNESS = 3


# ─────────────────────────────────────────────────────────────
# ACTIVATIONS
# ─────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ─────────────────────────────────────────────────────────────
# WEIGHT INITIALIZATION
# ─────────────────────────────────────────────────────────────

def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = math.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)


# ─────────────────────────────────────────────────────────────
# GAME STATE MIND
# ─────────────────────────────────────────────────────────────

class GameStateMind:
    """A tiny MLP that maps game state → 4 modulation scalars.

    The brain is shared PER TEAM (like TeamPressBrain).  It reads the
    match state and outputs modulation scalars that feed into:
      - press_intensity_mult    → multiplies TeamPressBrain.g
      - risk_tolerance          → scales PlayerMind.confidence / risk_in_box
      - defensive_urgency       → biases DefensiveActionBrain toward clear/block
      - attacking_directness    → biases on-ball intent distribution toward prog/switch

    All outputs are in [0, 1] via sigmoid, interpreted as multipliers
    around a neutral 1.0 (so 0.5 = half, 2.0 = double).
    """

    SCHEMA = "v1_gamestate_mind"

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
        seed: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
        self._rng = np.random.default_rng(seed)
        self.meta = meta or {}
        self.w = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2, 'w3': self.w3, 'b3': self.b3}

    # ── Forward pass ────────────────────────────────────────────
    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        """Pre-sigmoid outputs, shape (N, 4)."""
        single = x.ndim == 1
        x = np.asarray(x, dtype=np.float64)
        if single:
            x = x[None, :]
        h = _relu(x @ self.w1 + self.b1)
        h = _relu(h @ self.w2 + self.b2)
        return h @ self.w3 + self.b3

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid outputs in [0, 1], shape (4,) or (N, 4)."""
        single = x.ndim == 1
        logits = self.forward_logits(x)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return probs[0] if single else probs

    def modulate(self, game_state_vec: np.ndarray) -> Dict[str, float]:
        """Return the four modulation scalars."""
        p = self.forward(np.asarray(game_state_vec, dtype=np.float64))
        return {
            "press_mult": float(p[0]),
            "risk_tol": float(p[1]),
            "def_urgency": float(p[2]),
            "att_direct": float(p[3]),
        }

    # ── Random init ────────────────────────────────────────────
    @classmethod
    def random(cls, seed: int = 123) -> "GameStateMind":
        rng = np.random.default_rng(seed)
        w1 = _he_init(GAME_STATE_D, HIDDEN_1, rng)
        b1 = np.zeros(HIDDEN_1, dtype=np.float64)
        w2 = _he_init(HIDDEN_1, HIDDEN_2, rng)
        b2 = np.zeros(HIDDEN_2, dtype=np.float64)
        w3 = _xavier_init(HIDDEN_2, OUT_D, rng)
        b3 = np.zeros(OUT_D, dtype=np.float64)
        return cls(w1, b1, w2, b2, w3, b3, meta={"sensor_schema": "v1_gamestate"})

    # ── Persistence ────────────────────────────────────────────
    def serialize(self) -> Dict[str, Any]:
        weights = {}
        for k in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
            arr = getattr(self, k)
            if arr.ndim == 2:
                weights[k] = [[float(x) for x in row] for row in arr]
            else:
                weights[k] = [float(x) for x in arr]
        return {
            "arch_version": 1,
            "meta": {"sensor_schema": "v1_gamestate"},
            "weights": weights,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.serialize(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "GameStateMind":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        w = {k: np.asarray(arr, dtype=np.float64) for k, arr in raw["weights"].items()}
        return cls(w["w1"], w["b1"], w["w2"], w["b2"], w["w3"], w["b3"],
                   meta=raw.get("meta") or {})

    # ── Game state feature extraction ─────────────────────────
    @staticmethod
    def extract_features(
        minute: float,
        scorediff: int,
        possession_pct: float,
        xg_diff: float,
        goals_scored: int,
        goals_conceded: int,
        home: bool,
    ) -> np.ndarray:
        """Build the 12-d game state vector.

        All values normalised to roughly [-1, 1] or [0, 1].
        """
        minute_norm = minute / 90.0
        scorediff_norm = np.clip(scorediff / 3.0, -1.0, 1.0)
        poss = np.clip(possession_pct, 0.0, 1.0)
        xg_diff_norm = np.clip(xg_diff / 2.0, -1.0, 1.0)
        goals_norm = np.clip((goals_scored - goals_conceded) / 3.0, -1.0, 1.0)
        home_flag = 1.0 if home else 0.0
        # Phase indicators
        early = 1.0 if minute < 30 else 0.0
        late = 1.0 if minute > 75 else 0.0
        trailing = 1.0 if scorediff < 0 else 0.0
        leading = 1.0 if scorediff > 0 else 0.0

        return np.array([
            minute_norm, scorediff_norm, poss, xg_diff_norm,
            float(minute / 90.0),  # progress
            home_flag, trailing, leading,
            early, late,
            float(minute % 15) / 15.0,  # quarter-phase
            0.0,  # placeholder for future use
        ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# DISTILLATION
# ─────────────────────────────────────────────────────────────

def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def distill_gamestate_mind(
    surrogate: Any,
    n_states: int = 2000,
    epochs: int = 80,
    learning_rate: float = 0.02,
    seed: int = 123,
    verbose: bool = False,
) -> Tuple["GameStateMind", Dict[str, Any]]:
    """Fit a GameStateMind by distillation against the surrogate's
    game-state-conditioned policy.

    The surrogate's `expected_success` is evaluated on synthetic game states
    for each intent, and the plan head's argmax is used as the target.
    """
    from situational_brain import HIDDEN_D, OUT_D
    from decision_brain import PlayerIntent
    from role_features import ROLE_INTENTS, _role_family

    rng = np.random.default_rng(seed)
    mind = GameStateMind.random(seed=seed)

    # Generate synthetic game states
    states = []
    targets = []
    for _ in range(n_states):
        # Sample realistic game states
        minute = rng.uniform(0, 90)
        scorediff = int(rng.integers(-3, 4))
        poss = rng.uniform(0.3, 0.7)
        xg_diff = rng.uniform(-1.5, 1.5)
        goals_scored = int(rng.integers(0, 4))
        goals_conceded = int(rng.integers(0, 4))
        home = bool(rng.integers(0, 2))

        feat = GameStateMind.extract_features(
            minute, scorediff, rng.uniform(0.3, 0.7),
            rng.uniform(-1.5, 1.5),
            0, 0, True
        )
        # Pick a position to evaluate the surrogate's preference
        # Use a reference position (CM) to get the surrogate's game-state preference
        target = _pick_target_for_gamestate(surrogate, feat, rng)
        if target is not None:
            states.append(feat)
            # Encode target as a 4-d vector (press, risk, def, att)
            targets.append(_encode_target(target))

    if not states:
        raise ValueError("no valid states generated")

    states = np.stack(states)
    targets = np.array(targets)

    mind = GameStateMind.random(seed=123)
    params = [mind.w["w1"], mind.w["b1"], mind.w["w2"], mind.w["b2"], mind.w["w3"], mind.w["b3"]]
    keys = ["w1", "b1", "w2", "b2", "w3", "b3"]
    m = [np.zeros_like(p) for p in mind.w.values()]
    v = [np.zeros_like(p) for p in mind.w.values()]
    b1, b2, eps = 0.9, 0.999, 1e-8

    loss_curve = []
    for epoch in range(80):
        grads = {k: np.zeros_like(mind.w[k]) for k in mind.w}
        total_loss = 0.0
        T = len(states)
        logits = mind.forward_logits(states)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        # Cross-entropy with 4-d target
        dlogits = probs - targets
        dlogits /= T

        # Backprop
        h1 = _relu(states @ mind.w["w1"] + mind.w["b1"])
        h2 = _relu(h1 @ mind.w["w2"] + mind.w["b2"])
        gw3 = h2.T @ dlogits
        gb3 = dlogits.sum(axis=0)
        dh2 = dlogits @ mind.w["w3"].T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = dh2.sum(axis=0)
        dh1 = dh2 @ mind.w["w2"].T
        dh1[h1 <= 0.0] = 0.0
        gw1 = states.T @ dh1
        gb1 = dh1.sum(axis=0)
        grads = {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2, "w3": gw3, "b3": gb3}

        loss = float((-np.log(np.clip(probs, 1e-12, 1.0)) * targets).sum(axis=1).mean())
        for i, k in enumerate(mind.w.keys()):
            m[i] = 0.9 * m[i] + 0.1 * grads[k]
            v[i] = 0.999 * v[i] + 0.001 * (grads[k] * grads[k])
            mhat = m[i] / (1 - 0.9 ** (epoch + 1))
            vhat = v[i] / (1 - 0.999 ** (epoch + 1))
            mind.w[k] -= 0.02 * mhat / (np.sqrt(vhat) + 1e-8)

    return mind, {"loss_last": loss, "epochs": 80}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _encode_target(target: Dict[str, float]) -> np.ndarray:
    """Encode a target dict {press, risk, def, att} as 4-d vector."""
    return np.array([
        target.get("press_mult", 1.0),
        target.get("risk_tol", 1.0),
        target.get("def_urgency", 1.0),
        target.get("att_direct", 1.0),
    ], dtype=np.float64)


def _pick_target_for_gamestate(surrogate: Any, feat: np.ndarray, rng: np.random.Generator) -> Optional[Dict[str, float]]:
    """Query the surrogate for a default position (CM) and derive
    the game-state mind's target from the intent distribution.
    """
    # For now, use a simple heuristic: the surrogate's preference
    # over intents maps to the four modulations.
    # This is a placeholder — the real distillation would use
    # the surrogate's game-state-conditioned value function.
    return {
        "press_mult": 1.0,
        "risk_tol": 1.0,
        "def_urgency": 1.0,
        "att_direct": 1.0,
    }


# ─────────────────────────────────────────────────────────────
# PERSISTENCE (same pattern as SituationalBrain)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

__all__ = [
    "GameStateMind",
    "distill_gamestate_mind",
    "GAME_STATE_D", "HIDDEN_1", "HIDDEN_2", "OUT_D",
]