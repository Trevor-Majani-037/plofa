"""Per-player feedforward neural network — the evolved football brain.

Architecture
------------
    Input(24 or 24+role_D) → Dense(32, ReLU) → Dense(32, ReLU) → Dense(10, Softmax)

Each player owns a separate FootballBrain instance.  The weights are
identical in *shape* across all players but differ in *value* — seeded
by the player's DNA attributes and refined through evolutionary
training (see brain_evolution.py).

The input width is self-describing: a v1 brain consumes the shared 24-d
vector (``INPUT_SIZE``); a schema-v2 "role_features" brain consumes the
shared 24-d plus its role-family block (24 + 7 or 24 + 8).  ``random`` /
``from_dna`` take ``input_size`` and ``serialize`` derives the declared
``arch`` from the actual weight shapes, so loaders know exactly which
sensor schema to feed.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain_schema import (
    BRAIN_FORMAT_VERSION,
    BRAINS_V2_DIR,
    BrainSchemaError,
    brain_meta_dict,
    check_shapes,
    validate as validate_schema,
)

from role_features import (
    ROLE_INTENTS,
    ROLE_ONLY_INTENTS,
    V2_ROLE_D,
    V2_INPUT_D,
    _role_family,
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

#: The number of intent labels (10).
OUTPUT_SIZE = 10
OUT_D = OUTPUT_SIZE  # alias for backward compatibility

#: Off-ball press-gate input: the shared 24-d off-ball vector PLUS one
#: ball-certainty slot (slot 24) introduced by the perception-aware fitness
#: (2026-09-21).  The conscience now conditions press/hold on how sure the
#: runner is of the ball position.  TeamPressBrain keeps the 24-d TEAM feed.
OFFBALL_INPUT_SIZE = 25
HIDDEN_1 = 32
HIDDEN_2 = 32
OUTPUT_SIZE = 10

INTENT_LABELS = [
    "PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
    "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION",
]

# ─────────────────────────────────────────────────────────────
# SHARED SITUATIONAL BRAIN CONSTANTS
# ─────────────────────────────────────────────────────────────

TOKEN_D = 12
N_MAX = 16  # distance cap of 16 tokens
EMBED_D = 32
HIDDEN_D = 32  # default, can be overridden
OUTPUT_SIZE = 10
PLAN_OUT_D = 4

STATE_D = 24
ROLE_D = 13
MEMORY_D = 8
PLAN_D = 6
SITUATION_D = STATE_D + ROLE_D + MEMORY_D + PLAN_D  # 51
U_D = EMBED_D + SITUATION_D  # 32 + 51 = 83
Q_D = SITUATION_D  # 51

SCHEMA = "v5_situational"

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
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
        return e / e.sum()
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softmax_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


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
    n_in: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate initial weights biased by player DNA attributes.

    Higher vision → wider input-to-hidden receptive field (larger magnitudes
    on vision-related input rows).  Higher composure → smaller initial
    hidden-layer magnitudes (calmer internal state).  Higher decisions →
    slightly sharper output layer (more decisive softmax).

    ``n_in`` is the input width — 24 for a v1 brain, or 24 + role block for
    a schema-v2 role_features brain.
    """
    vision_scale = 0.8 + vision * 0.4
    composure_scale = 1.2 - composure * 0.4
    decisions_scale = 0.9 + decisions * 0.2

    w1 = _he_init(n_in, HIDDEN_1, rng) * vision_scale
    b1 = np.zeros(HIDDEN_1, dtype=np.float64)
    w2 = _he_init(HIDDEN_1, HIDDEN_2, rng) * composure_scale
    b2 = np.zeros(HIDDEN_2, dtype=np.float64)
    w3 = _he_init(HIDDEN_2, OUTPUT_SIZE, rng) * decisions_scale
    b3 = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    return w1, b1, w2, b2, w3, b3


# ─────────────────────────────────────────────────────────────
# SITUATIONAL BRAIN (attention + GRU + memory + plan)
# ─────────────────────────────────────────────────────────────

class SituationalBrain:
    """Attention + GRU + memory + plan head.

    Architecture:
        entity tokens (N, 12) → embedding (32) → attention → context (32)
        [state(24), role(9), memory(8), plan(6)] → GRU(32) → intent(10) + plan(4)

    The brain is a PURE FUNCTION of (tokens, state, role, memory, plan, h_prev).
    No internal state between calls — the recurrent state h is passed in/out.
    """

    SCHEMA = "v5_situational"

    def __init__(
        self,
        w1: np.ndarray = None, b1: np.ndarray = None,
        w2: np.ndarray = None, b2: np.ndarray = None,
        w3: np.ndarray = None, b3: np.ndarray = None,
        w_e: np.ndarray = None,
        w_q: np.ndarray = None, w_k: np.ndarray = None, w_v: np.ndarray = None,
        w_z: np.ndarray = None, u_z: np.ndarray = None, b_z: np.ndarray = None,
        w_r: np.ndarray = None, u_r: np.ndarray = None, b_r: np.ndarray = None,
        w_n: np.ndarray = None, u_n: np.ndarray = None, b_n: np.ndarray = None,
        w_o: np.ndarray = None, b_o: np.ndarray = None,
        w_p: np.ndarray = None, b_p: np.ndarray = None,
        seed: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        **weights: np.ndarray,
    ):
        # Support both legacy individual params and new weights dict
        if weights:
            w = weights
            self.w_e = w.get("w_e")
            self.w_q = w.get("w_q")
            self.w_k = w.get("w_k")
            self.w_v = w.get("w_v")
            self.w_z = w.get("w_z")
            self.u_z = w.get("u_z")
            self.b_z = w.get("b_z")
            self.w_r = w.get("w_r")
            self.u_r = w.get("u_r")
            self.b_r = w.get("b_r")
            self.w_n = w.get("w_n")
            self.u_n = w.get("u_n")
            self.b_n = w.get("b_n")
            self.w_o = w.get("w_o")
            self.b_o = w.get("b_o")
            self.w_p = w.get("w_p")
            self.b_p = w.get("b_p")
        else:
            # Legacy individual parameters
            self.w_e = w_e
            self.w_q = w_q
            self.w_k = w_k
            self.w_v = w_v
            self.w_z = w_z
            self.u_z = u_z
            self.b_z = b_z
            self.w_r = w_r
            self.u_r = u_r
            self.b_r = b_r
            self.w_n = w_n
            self.u_n = u_n
            self.b_n = b_n
            self.w_o = w_o
            self.b_o = b_o
            self.w_p = w_p
            self.b_p = b_p
            self.w1 = w1
            self.b1 = b1
            self.w2 = w2
            self.b2 = b2
            self.w3 = w3
            self.b3 = b3
        self._rng = np.random.default_rng(None)
        # Provide a weights dictionary for compatibility with distillation code
        self.w = {
            "w_e": self.w_e, "w_q": self.w_q, "w_k": self.w_k, "w_v": self.w_v,
            "w_z": self.w_z, "u_z": self.u_z, "b_z": self.b_z,
            "w_r": self.w_r, "u_r": self.u_r, "b_r": self.b_r,
            "w_n": self.w_n, "u_n": self.u_n, "b_n": self.b_n,
            "w_o": self.w_o, "b_o": self.b_o,
            "w_p": self.w_p, "b_p": self.b_p,
        }
        self.meta = meta or {}
    # ─────────────────────────────────────────────────────────────
    # CLASS METHODS
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def random(cls, seed: int = 123, hidden_size: int = 32, input_size: int = 51) -> "SituationalBrain":
        rng = np.random.default_rng(seed)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0.0, math.sqrt(2.0 / fan_in),
                              size=(fan_in, fan_out)).astype(np.float64)

        # Use custom dimensions or defaults
        hidden_size = hidden_size
        embed_d = 32  # embedding size fixed
        input_size = input_size  # 24 + role_d + memory_d + plan_d = 51

        # Derived dimensions
        token_d = 12
        embed_d = 32
        state_d = STATE_D
        role_d = ROLE_D
        memory_d = MEMORY_D
        plan_d = PLAN_D
        situation_d = STATE_D + ROLE_D + MEMORY_D + PLAN_D
        embed_d = 32
        out_d = 10
        plan_out_d = 4
        hidden_d = hidden_size
        u_d = 32 + situation_d
        q_d = situation_d

        w = {
            "w_e": he(12, 32),  # token embedding
            "w_q": he(input_size, 32),  # query projection
            "w_k": he(32, 32),  # key projection
            "w_v": he(32, 32),  # value projection
            "w_z": he(32 + input_size, hidden_size),  # GRU update gate
            "u_z": he(hidden_size, hidden_size),
            "b_z": np.zeros(hidden_size),
            "w_r": he(32 + input_size, hidden_size),  # GRU reset gate
            "u_r": he(hidden_size, hidden_size),
            "b_r": np.zeros(hidden_size),
            "w_n": he(32 + input_size, hidden_size),  # GRU new gate
            "u_n": he(hidden_size, hidden_size),
            "b_n": np.zeros(hidden_size),
            "w_o": he(hidden_size, 10),  # intent output
            "b_o": np.zeros(10),
            "w_p": he(hidden_size, 4),  # plan output
            "b_p": np.zeros(4),
        }
        meta = {
            "arch_version": 3,
            "sensor_schema": cls.SCHEMA,
            "hidden_size": hidden_size,
            "input_size": input_size,
        }
        return cls(**w, meta=meta)

    # ─────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────

    def forward_logits(self, tokens: np.ndarray, token_mask: np.ndarray,
                       state: np.ndarray, role: np.ndarray, memory: np.ndarray,
                       plan: np.ndarray, h_prev: np.ndarray):
        """The pre-softmax OUTPUTS, shape (N, 10)."""
        single = tokens.ndim == 1
        x = np.asarray(tokens, dtype=np.float64)
        if single:
            x = x[None, :]
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e = x @ self.w_e                       # (N, E)
        role = np.asarray(role, dtype=np.float64).reshape(-1)
        expected_total = self.w_q.shape[0]
        role_width = expected_total - (24 + 8 + 6)  # state + memory + plan
        if role.shape[0] < role_width:
            role = np.pad(role, (0, role_width - role.shape[0]))
        elif role.shape[0] > role_width:
            role = role[:role_width]
        q = np.concatenate([state, role, memory, plan]) @ self.w_q   # (E,)
        k = e @ self.w_k                         # (N, E)
        v = e @ self.w_v                         # (N, E)
        scores = (k @ q) / math.sqrt(EMBED_D)    # (N,)
        # A masked token must take NO attention mass.
        mask = np.asarray(token_mask, dtype=np.float64).reshape(-1)
        if mask.sum() <= 0.0:
            alpha = np.zeros_like(scores)
            ctx = np.zeros(EMBED_D, dtype=np.float64)
        else:
            scores = np.where(mask > 0.0, scores, -np.inf)
            alpha = _softmax_rows(scores)
            ctx = alpha @ v

        # ── RECURRENCE ────────────────────────────────────────────
        u = np.concatenate([state, role, memory, plan, ctx])   # (U,)
        z = _sigmoid(u @ self.w_z + h_prev @ self.u_z + self.b_z)
        r = _sigmoid(u @ self.w_r + h_prev @ self.u_r + self.b_r)
        n = np.tanh(u @ self.w_n + (r * h_prev) @ self.u_n + self.b_n)
        h_new = (1.0 - z) * n + z * h_prev

        # ── HEADS ─────────────────────────────────────────────────
        logits = h_new @ self.w_o + self.b_o
        plan_logits = h_new @ self.w_p + self.b_p

        cache = {
            "tok": x, "mask": mask, "e": e, "q": q, "k": k, "v": v,
            "scores": scores, "alpha": alpha, "ctx": ctx, "u": u,
            "z": z, "r": r, "n": n, "h": h_prev, "h_new": h_new,
            "situation": np.concatenate([state, role, memory, plan]),
        }
        return logits, plan_logits, h_new, cache

    def forward(self, tokens: np.ndarray, token_mask: np.ndarray,
                state: np.ndarray, role: np.ndarray, memory: np.ndarray,
                plan: np.ndarray, h_prev: np.ndarray):
        """Run the vision vector(s) through the network."""
        """Run the vision vector(s) through the network."""
        return self.forward_logits(tokens, token_mask, state, role, memory, plan, h_prev)

    def forward_sequence(self, tokens: np.ndarray, token_mask: np.ndarray,
                         state: np.ndarray, role: np.ndarray,
                         memory: np.ndarray, plan: np.ndarray, h0: np.ndarray):
        """A whole sequence, one touch at a time."""
        T = len(state)
        logits = np.zeros((T, OUT_D), dtype=np.float64)
        plan_logits = np.zeros((T, PLAN_OUT_D), dtype=np.float64)
        caches = []
        h = np.asarray(h0, dtype=np.float64).reshape(HIDDEN_D)
        for t in range(T):
            lg, plg, h, c = self.forward_logits(tokens[t], token_mask[t], state[t],
                                         role[t], memory[t], plan[t], h)
            logits[t] = lg
            plan_logits[t] = plg
            caches.append(c)
        return logits, plan_logits, h, caches

    # ── BACKWARD ─────────────────────────────────────────────────────
    def backward(self, dlogits: np.ndarray, dplan_logits: np.ndarray,
                 caches: List[Dict[str, Any]], d_state=None, d_role=None,
                 d_memory=None, d_plan=None, dh_last: Optional[np.ndarray] = None):
        """Accumulate parameter gradients over a sequence (BPTT)."""
        w = self.w
        grads = {k: np.zeros_like(v) for k, v in self.w.items()}
        T = len(caches)
        dh_next = np.zeros(HIDDEN_D, dtype=np.float64) if dh_last is None             else np.asarray(dh_last, dtype=np.float64).copy()
        smp = STATE_D + ROLE_D + MEMORY_D + PLAN_D

        for t in range(T - 1, -1, -1):
            c = caches[t]
            dl = np.asarray(dlogits[t], dtype=np.float64)
            dpl = np.asarray(dplan_logits[t], dtype=np.float64)

            # ── heads ─────────────────────────────────────────────
            dh_prime = dl @ self.w["w_o"].T + dpl @ self.w["w_p"].T + dh_next
            grads["w_o"] += np.outer(c["h_new"], dl)
            grads["b_o"] += dl
            grads["w_p"] += np.outer(c["h_new"], dpl)
            grads["b_p"] += dpl

            # ── GRU backward ──────────────────────────────────────
            z, r, n = c["z"], c["r"], c["n"]
            h_prev, u = c["h"], c["u"]

            dz = dh_prime * (h_prev - n)
            dz_pre = dz * z * (1.0 - z)
            dn = dh_prime * (1.0 - z)
            dA = dn * (1.0 - n * n)

            dm = dA @ self.w["u_n"].T
            dr = dm * h_prev
            dr_pre = dr * r * (1.0 - r)

            du = (dA @ self.w["w_n"].T
                  + dr_pre @ self.w["w_r"].T
                  + dz_pre @ self.w["w_z"].T)

            grads["w_z"] += np.outer(u, dz_pre)
            grads["u_z"] += np.outer(c["h"], dz_pre)
            grads["b_z"] += dz_pre
            grads["w_r"] += np.outer(u, dr_pre)
            grads["u_r"] += np.outer(c["h"], dr_pre)
            grads["b_r"] += dr_pre
            grads["w_n"] += np.outer(u, dA)
            grads["b_n"] += dA
            grads["u_n"] += np.outer(r * c["h"], dA)

            dh_prev_acc = dm * r                      # through m = r (*) h
            dh_prev_acc += dr_pre @ self.w["u_r"].T
            dh_prev_acc += dz_pre @ self.w["u_z"].T
            dh_prev_acc += dh_prime * z               # the carried h
            dh_next = dh_prev_acc

            # ── ATTENTION backward ────────────────────────────────
            dctx = du[smp:]
            alpha = c["alpha"]
            dv = alpha[:, None] * dctx[None, :]          # (N, E)
            dalpha = dctx @ c["v"].T                      # (N,)
            dot = float((alpha * dalpha).sum())
            dscores = alpha * (dalpha - dot)
            scale = 1.0 / math.sqrt(EMBED_D)
            dq = (dscores[:, None] * c["k"]).sum(axis=0) * scale
            dk = dscores[:, None] * c["q"][None, :] * scale

            grads["w_v"] += c["e"].T @ dv
            grads["w_k"] += c["e"].T @ dk
            de = dv @ self.w["w_v"].T + dk @ self.w["w_k"].T
            grads["w_e"] += c["tok"].T @ de
            grads["w_q"] += np.outer(c["situation"], dq)

            # The query reads the SAME situation the GRU reads, so its
            # gradient lands on the same input gradient and must be ADDED.
            du[:smp] += dq @ self.w["w_q"].T

            if d_state is not None:
                d_state[t] += du[:STATE_D]
            if d_role is not None:
                d_role[t] += du[STATE_D:STATE_D + ROLE_D]
            if d_memory is not None:
                d_memory[t] += du[STATE_D + ROLE_D:STATE_D + ROLE_D + MEMORY_D]
            if d_plan is not None:
                d_plan[t] += du[STATE_D + ROLE_D + MEMORY_D:smp]

        return grads

    # ── persistence ────────────────────────────────────────────────
    def serialize(self) -> Dict[str, Any]:
        weights: Dict[str, Any] = {}
        for k, arr in self.w.items():
            if arr.ndim == 2:
                weights[k] = [[float(x) for x in row] for row in arr]
            else:
                weights[k] = [float(x) for x in arr]
        return {
            "arch_version": 3,
            "meta": {
                "arch_version": 3,
                "sensor_schema": self.SCHEMA,
                "training_method": (self.meta or {}).get(
                    "training_method", "bptt_situational"),
                "role_family": (self.meta or {}).get("role_family"),
                "role_features": (self.meta or {}).get("role_features"),
            },
            "weights": weights,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.serialize(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SituationalBrain":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        w = {}
        for k, arr in raw["weights"].items():
            a = np.asarray(arr, dtype=np.float64)
            w[k] = a
        return cls(**w, meta=raw.get("meta") or {})

