"""SituationalBrain — a brain that attends, remembers and plans.

WHY A NEW BRAIN AND NOT A PATCHED OLD ONE
------------------------------------------
The shipped brain (``football_brain.FootballBrain``) is a PURE FUNCTION:
``relu(x@w1+b1) -> relu(h@w2+b2) -> h@w3+b3``, 24-ish inputs, 10 intents
out.  It has no recurrence (the same situation always gives the same
answer), no memory (nothing about what just happened to this player is in
the vector) and no attention (the vector is a HAND-POOLED summary —
"nearest defender", "best forward teammate" — so the network never sees
WHO it is looking at, only two pooled scalars about them).

Adding those three to the existing class would change its input width,
which invalidates every shipped brain (the schema discipline in
``role_features.align_role_tail`` exists precisely because that is
dangerous).  So this is a SEPARATE class with its own schema
(``v5_situational``), and the shipped ``brains/`` set is untouched.

THE ARCHITECTURE
----------------
::

    tokens (N, 12)          the players this player could attend to
        │  W_e
        ▼
      e_j (32) ──────────────┐
        │                    │  W_k -> k_j, W_v -> v_j
        │                    │
    q = W_q · [state, memory, plan] (32)
        │                    │
        └──► α = softmax(q·k_j/√E) over the VISIBLE tokens
                             │
                             ▼
                    c = Σ α_j v_j          (32)   THE CONTEXT VECTOR
                             │
    u = [state(24), memory(8), plan(6), c(32)]      (70)
                             │
                       GRU(u, h_prev)                (32)
                             │  h  (the RECURRENT state)
              ┌──────────────┴──────────────┐
              ▼                             ▼
      logits (10)  the INTENT          plan_logits (4)  the PLAN

WHAT EACH PIECE IS FOR
----------------------
ATTENTION
    The 24-d shared block pools the scene into fixed scalars, so "the
    winger who is 2 m in front of me" and "the winger 25 m away on the
    far side" are the same input.  The attention block lets the network
    CHOOSE which players matter and read their features directly.  It is
    real multi-head-free attention over a variable-size entity set, with a
    MASK so padded tokens take no attention mass.

RECURRENCE
    A GRU cell.  ``h`` is carried from touch to touch, so the decision
    depends on what the player has been doing — which is the difference
    between "a striker" and "a striker who has just missed two chances".
    The state lives with the PLAYER (``player_state.PlayerNeuralState``),
    not on the brain, because the brain objects are cached per POSITION
    and a state on the brain would be shared by eleven players.

PLANNING
    A second head that proposes which possession phase the team should be
    executing.  It is supervised by the engine's OWN phase
    (``possession_phases.PossessionPhase``) rather than by a new
    hand-written table, so the plan head learns to name the thing the
    engine already does.  The plan is adopted with a HYSTERESIS
    (``player_state.TeamPlan.propose``), so one stray confident touch
    cannot rewrite the team's objective.

MEMORY
    The 8-d block from ``memory_features`` — what this player remembers
    of his own match so far.  It is an INPUT, so the network can learn
    how much a missed chance should change the next decision, rather than
    having that rule written by hand.

PURE NUMPY
----------
Forward and backward are written out in numpy, like ``brain_distill``'s
trainer (which already back-propagates a cross-entropy through the
softmax by hand).  No ML framework, no gradient library.  The backward
pass is the whole point of this file: without it the network could not be
trained at all, and the project has no other way to fit a net.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# DIMENSIONS
# ─────────────────────────────────────────────────────────────

#: Per-entity features.  Twelve, and every one of them is a REAL number
#: about a real player — no placeholder slots.
TOKEN_D = 12

#: How many entities the attention can look at at once.  The nearest
#: ``N_MAX`` players by distance, padded.
N_MAX = 16

#: The attention embedding width.
EMBED_D = 32

#: The recurrent width.
HIDDEN_D = 32

#: The shared 24-d block.
STATE_D = 24

#: The role block, at a FIXED width of 9 (the widest family's menu).
#: The families emit 7/8/9 features and they mean different things in each
#: slot — which is exactly right, because the brain is per position.  A
#: fixed width keeps one network shape for all eleven positions.
ROLE_D = 9

#: The memory block width (``memory_features.MEMORY_FEATURE_D``).
MEMORY_D = 8

#: The plan summary width: one-hot over the phases, plus confidence and a
#: "has a plan" flag (``player_state.TeamPlan.summary``).
PLAN_D = 6

#: The intent count.
OUT_D = 10

#: The number of plan phases (``player_state.PLAN_COUNT``).
PLAN_OUT_D = 4

#: Everything the network reads about the situation, before the context
#: vector: the shared block, the role block, the memory block and the
#: plan.
SITUATION_D = STATE_D + ROLE_D + MEMORY_D + PLAN_D

#: The concatenated input to the GRU.
U_D = SITUATION_D + EMBED_D

#: The query is built from the situation, NOT from the context it is about
#: to produce.
Q_D = SITUATION_D


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softmax_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def _softmax(x: np.ndarray) -> np.ndarray:
    """A one-dimensional softmax."""
    return _softmax_rows(np.asarray(x, dtype=np.float64).reshape(-1))


class SituationalBrain:
    """Attention + recurrence + a plan head, in numpy.

    The weights are named so that a saved brain is legible: ``w_e`` is the
    token embedding, ``w_q``/``w_k``/``w_v`` are the attention, ``w_z``/
    ``u_z``/``w_r``/``u_r``/``w_n``/``u_n`` are the GRU, ``w_o``/``b_o``
    are the intent head and ``w_p``/``b_p`` are the plan head.
    """

    SCHEMA = "v5_situational"

    def __init__(self, weights: Dict[str, np.ndarray], meta: Optional[Dict[str, Any]] = None):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.meta = meta or {}
        self._check_shapes()

    # ── shapes ─────────────────────────────────────────────────────
    def _check_shapes(self) -> None:
        w = self.w
        expected = {
            "w_e": (TOKEN_D, EMBED_D),
            "w_q": (Q_D, EMBED_D),
            "w_k": (EMBED_D, EMBED_D),
            "w_v": (EMBED_D, EMBED_D),
            "w_z": (U_D, HIDDEN_D),
            "u_z": (HIDDEN_D, HIDDEN_D),
            "b_z": (HIDDEN_D,),
            "w_r": (U_D, HIDDEN_D),
            "u_r": (HIDDEN_D, HIDDEN_D),
            "b_r": (HIDDEN_D,),
            "w_n": (U_D, HIDDEN_D),
            "u_n": (HIDDEN_D, HIDDEN_D),
            "b_n": (HIDDEN_D,),
            "w_o": (HIDDEN_D, OUT_D),
            "b_o": (OUT_D,),
            "w_p": (HIDDEN_D, PLAN_OUT_D),
            "b_p": (PLAN_OUT_D,),
        }
        for k, shape in expected.items():
            if k not in w:
                raise ValueError(f"missing weight {k!r}")
            if tuple(w[k].shape) != shape:
                raise ValueError(
                    f"weight {k!r} has shape {tuple(w[k].shape)}, expected {shape}")

    @property
    def hidden_size(self) -> int:
        return HIDDEN_D

    # ── random init ────────────────────────────────────────────────
    @classmethod
    def random(cls, seed: int = 123) -> "SituationalBrain":
        rng = np.random.default_rng(seed)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0.0, math.sqrt(2.0 / fan_in),
                              size=(fan_in, fan_out)).astype(np.float64)

        w = {
            "w_e": he(TOKEN_D, EMBED_D),
            "w_q": he(Q_D, EMBED_D),
            "w_k": he(EMBED_D, EMBED_D),
            "w_v": he(EMBED_D, EMBED_D),
            "w_z": he(U_D, HIDDEN_D),
            "u_z": he(HIDDEN_D, HIDDEN_D),
            "b_z": np.zeros(HIDDEN_D),
            "w_r": he(U_D, HIDDEN_D),
            "u_r": he(HIDDEN_D, HIDDEN_D),
            "b_r": np.zeros(HIDDEN_D),
            "w_n": he(U_D, HIDDEN_D),
            "u_n": he(HIDDEN_D, HIDDEN_D),
            "b_n": np.zeros(HIDDEN_D),
            "w_o": he(HIDDEN_D, OUT_D),
            "b_o": np.zeros(OUT_D),
            "w_p": he(HIDDEN_D, PLAN_OUT_D),
            "b_p": np.zeros(PLAN_OUT_D),
        }
        return cls(w, meta={"arch_version": 3, "sensor_schema": cls.SCHEMA})

    # ── forward ────────────────────────────────────────────────────
    def forward(self, tokens: np.ndarray, token_mask: np.ndarray,
                state: np.ndarray, role: np.ndarray, memory: np.ndarray,
                plan: np.ndarray, h_prev: np.ndarray):
        """One decision.

        Parameters
        ----------
        tokens : (N, TOKEN_D)          the visible entities
        token_mask : (N,)              1.0 for a real entity, 0.0 for padding
        state : (STATE_D,)             the shared 24-d block
        role : (ROLE_D,)               the player's own role block
        memory : (MEMORY_D,)           the memory block
        plan : (PLAN_D,)               the plan summary
        h_prev : (HIDDEN_D,)           the recurrent state carried in

        Returns
        -------
        (logits, plan_logits, h_new, cache)
        """
        w = self.w
        tok = np.asarray(tokens, dtype=np.float64).reshape(-1, TOKEN_D)
        mask = np.asarray(token_mask, dtype=np.float64).reshape(-1)
        st = np.asarray(state, dtype=np.float64).reshape(STATE_D)
        ro = np.asarray(role, dtype=np.float64).reshape(ROLE_D)
        mem = np.asarray(memory, dtype=np.float64).reshape(MEMORY_D)
        pl = np.asarray(plan, dtype=np.float64).reshape(PLAN_D)
        h = np.asarray(h_prev, dtype=np.float64).reshape(HIDDEN_D)
        situation = np.concatenate([st, ro, mem, pl])

        # ── ATTENTION ────────────────────────────────────────────
        e = tok @ w["w_e"]                       # (N, E)
        q = situation @ w["w_q"]                  # (E,)
        k = e @ w["w_k"]                         # (N, E)
        v = e @ w["w_v"]                         # (N, E)
        scores = (k @ q) / math.sqrt(EMBED_D)    # (N,)
        # A masked token must take NO attention mass.  Doing it by adding
        # -inf before the softmax is the standard way, but with an all-zero
        # mask every score would be -inf and the softmax would be NaN — so
        # the "no entities at all" case is special-cased to a zero context.
        if mask.sum() <= 0.0:
            alpha = np.zeros_like(scores)
            ctx = np.zeros(EMBED_D, dtype=np.float64)
        else:
            scores = np.where(mask > 0.0, scores, -np.inf)
            alpha = _softmax_rows(scores)
            ctx = alpha @ v

        # ── RECURRENCE ────────────────────────────────────────────
        u = np.concatenate([situation, ctx])    # (U,)
        z = _sigmoid(u @ w["w_z"] + h @ w["u_z"] + w["b_z"])
        r = _sigmoid(u @ w["w_r"] + h @ w["u_r"] + w["b_r"])
        n = np.tanh(u @ w["w_n"] + (r * h) @ w["u_n"] + w["b_n"])
        h_new = (1.0 - z) * n + z * h

        # ── HEADS ─────────────────────────────────────────────────
        logits = h_new @ w["w_o"] + w["b_o"]
        plan_logits = h_new @ w["w_p"] + w["b_p"]

        cache = {
            "tok": tok, "mask": mask, "e": e, "q": q, "k": k, "v": v,
            "scores": scores, "alpha": alpha, "ctx": ctx, "u": u,
            "z": z, "r": r, "n": n, "h": h, "h_new": h_new,
            "situation": situation,
        }
        return logits, plan_logits, h_new, cache

    def forward_sequence(self, tokens: np.ndarray, token_mask: np.ndarray,
                         state: np.ndarray, role: np.ndarray,
                         memory: np.ndarray, plan: np.ndarray, h0: np.ndarray):
        """A whole sequence, one touch at a time.

        Shapes: tokens (T, N, TOKEN_D), token_mask (T, N), state (T, 24),
        role (T, 9), memory (T, 8), plan (T, 6), h0 (32,).

        Returns (logits (T, 10), plan_logits (T, K), h (32,), caches list).
        """
        T = len(state)
        logits = np.zeros((T, OUT_D), dtype=np.float64)
        plan_logits = np.zeros((T, PLAN_OUT_D), dtype=np.float64)
        caches = []
        h = np.asarray(h0, dtype=np.float64).reshape(HIDDEN_D)
        for t in range(T):
            lg, plg, h, c = self.forward(tokens[t], token_mask[t], state[t],
                                         role[t], memory[t], plan[t], h)
            logits[t] = lg
            plan_logits[t] = plg
            caches.append(c)
        return logits, plan_logits, h, caches

    # ── backward ───────────────────────────────────────────────────
    def backward(self, dlogits: np.ndarray, dplan_logits: np.ndarray,
                 caches: List[Dict[str, Any]], d_state=None, d_role=None,
                 d_memory=None, d_plan=None, dh_last: Optional[np.ndarray] = None):
        """Accumulate parameter gradients over a sequence (BPTT).

        ``dlogits`` and ``dplan_logits`` are (T, OUT_D) / (T, PLAN_OUT_D).
        The optional input-gradient arguments are filled IN PLACE when
        supplied, so a caller can also learn the inputs.

        Returns the gradient dict, keyed by the same names as ``self.w``.

        THE GRU, WRITTEN OUT
        ---------------------
        Forward::

            z = sig(W_z u + U_z h + b_z)
            r = sig(W_r u + U_r h + b_r)
            m = r (*) h
            n = tanh(W_n u + U_n m + b_n)
            h' = (1 - z) (*) n + z (*) h

        The two terms that use ``dh'`` DIRECTLY (the ``(1-z) n`` and the
        ``z h``) use the ORIGINAL ``dh'``, not the running accumulation —
        using the accumulated one would silently charge the earlier gates
        for the later gates' share.  Everything else accumulates.
        """
        w = self.w
        grads = {k: np.zeros_like(v) for k, v in w.items()}
        T = len(caches)
        dh_next = np.zeros(HIDDEN_D, dtype=np.float64) if dh_last is None \
            else np.asarray(dh_last, dtype=np.float64).copy()
        sit = SITUATION_D

        for t in range(T - 1, -1, -1):
            c = caches[t]
            dl = np.asarray(dlogits[t], dtype=np.float64)
            dpl = np.asarray(dplan_logits[t], dtype=np.float64)

            # ── heads ─────────────────────────────────────────────
            dh_prime = dl @ w["w_o"].T + dpl @ w["w_p"].T + dh_next
            grads["w_o"] += np.outer(c["h_new"], dl)
            grads["b_o"] += dl
            grads["w_p"] += np.outer(c["h_new"], dpl)
            grads["b_p"] += dpl

            # ── GRU backward ──────────────────────────────────────
            z, r, n = c["z"], c["r"], c["n"]
            h_prev, u = c["h"], c["u"]

            # h' = (1 - z) (*) n + z (*) h   =>   dh'/dz = h - n
            # (the sign matters: getting it backwards turns the update gate
            # into an "anti-update" gate, and the loss still falls, so only
            # a finite-difference check catches it).
            dz = dh_prime * (h_prev - n)
            dz_pre = dz * z * (1.0 - z)
            dn = dh_prime * (1.0 - z)
            dA = dn * (1.0 - n * n)

            dm = dA @ w["u_n"].T
            dr = dm * h_prev
            dr_pre = dr * r * (1.0 - r)

            du = (dA @ w["w_n"].T
                  + dr_pre @ w["w_r"].T
                  + dz_pre @ w["w_z"].T)

            grads["w_z"] += np.outer(u, dz_pre)
            grads["u_z"] += np.outer(h_prev, dz_pre)
            grads["b_z"] += dz_pre
            grads["w_r"] += np.outer(u, dr_pre)
            grads["u_r"] += np.outer(h_prev, dr_pre)
            grads["b_r"] += dr_pre
            grads["w_n"] += np.outer(u, dA)
            grads["b_n"] += dA
            grads["u_n"] += np.outer(r * h_prev, dA)

            dh_prev_acc = dm * r                      # through m = r (*) h
            dh_prev_acc += dr_pre @ w["u_r"].T
            dh_prev_acc += dz_pre @ w["u_z"].T
            dh_prev_acc += dh_prime * z               # the carried h
            dh_next = dh_prev_acc

            # ── ATTENTION backward ────────────────────────────────
            # The context vector is the LAST block of the GRU's input, so
            # its gradient is the tail of `du`.
            dctx = du[sit:]
            alpha = c["alpha"]
            dv = alpha[:, None] * dctx[None, :]          # (N, E)
            dalpha = dctx @ c["v"].T                      # (N,)
            # THE FULL SOFTMAX JACOBIAN.
            #
            # The short form ``alpha * (1 - alpha) * dalpha`` is only valid
            # when ``sum(alpha * dalpha) == 0`` — i.e. when the loss
            # depends on alpha only through a shift-invariant function.
            # Here the loss depends on the CONTEXT ``ctx = alpha @ v``, so
            # ``sum(alpha * dalpha) = dctx . ctx``, which is generally NOT
            # zero.  The short form silently drops a term and the network
            # trains against the wrong gradient.
            dot = float((alpha * dalpha).sum())
            dscores = alpha * (dalpha - dot)
            scale = 1.0 / math.sqrt(EMBED_D)
            dq = (dscores[:, None] * c["k"]).sum(axis=0) * scale
            dk = dscores[:, None] * c["q"][None, :] * scale

            # ORIENTATION MATTERS, AND SQUARE MATRICES HIDE IT.
            #
            #   v[n, j] = sum_i e[n, i] * W_v[i, j]
            #
            # so  dW_v[i, j] = sum_n dv[n, j] * e[n, i]  ==  e.T @ dv.
            #
            # The "obvious" ``dv.T @ e`` gives sum_n dv[n, i] * e[n, j] —
            # the TRANSPOSE of the right answer.  Because W_v is SQUARE
            # (EMBED_D x EMBED_D) the shapes match perfectly and numpy
            # raises nothing; only a finite-difference check finds it.
            grads["w_v"] += c["e"].T @ dv
            grads["w_k"] += c["e"].T @ dk
            de = dv @ w["w_v"].T + dk @ w["w_k"].T
            grads["w_e"] += c["tok"].T @ de
            grads["w_q"] += np.outer(c["situation"], dq)

            # The query reads the SAME situation the GRU reads, so its
            # gradient lands on the same input gradient and must be ADDED,
            # not overwrite it.
            du[:sit] += dq @ w["w_q"].T

            if d_state is not None:
                d_state[t] += du[:STATE_D]
            if d_role is not None:
                d_role[t] += du[STATE_D:STATE_D + ROLE_D]
            if d_memory is not None:
                d_memory[t] += du[STATE_D + ROLE_D:STATE_D + ROLE_D + MEMORY_D]
            if d_plan is not None:
                d_plan[t] += du[STATE_D + ROLE_D + MEMORY_D:sit]

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
        return cls(w, meta=raw.get("meta") or {})
