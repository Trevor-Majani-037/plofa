"""SITUATIONAL DISTILLATION — train the four-capability brain by BPTT.

WHAT THIS IS
------------
The same idea as ``brain_distill.py`` — fit the network to the SURROGATE'S
OWN POLICY rather than to a reward — extended to a network that has a
recurrent core, an attention block and a second head.

The target for every step is still "the intent the surrogate's table
prefers in this bucket, over the role's vocabulary".  That matters: the
CENSUS compares a trained set's intent mix against the table's mix, and
that comparison stays meaningful only if the objective is the same one.
What changes is the model, not the goal.

THE THREE ADDITIONS
-------------------
RECURRENCE
    The training signal now spans a SEQUENCE of touches, so the loss is
    back-propagated THROUGH TIME (BPTT) rather than through a single
    forward pass.  The hidden state is carried from step to step, exactly
    as it is at match time, and it is RESET at the start of every
    sequence — because a possession is not a continuation of the last one.

ATTENTION
    The attention reads a SET of entity tokens, so the sequence carries
    the token set and its mask, and the padded tokens take no attention
    mass (the mask is applied inside the softmax).

PLANNING
    A second head, supervised by the engine's OWN possession phase.  It
    is a SEPARATE loss term, weighted ``PLAN_LOSS_W``, because the plan
    is a different kind of answer (which phase are we in) from the
    intent (what do I do now).

WHY NOT TRAIN THE PLAN HEAD FROM THE TABLE
-------------------------------------------
Because the table says nothing about phases.  The engine's phase machine
(``possession_phases``) already names the phase, it is pure geometry, and
it is the thing the team is actually playing — so it is the honest
supervision signal for "which phase is this".

THE LOSS, EXACTLY
-----------------
::

    L = mean_t CE(intent_logits[t], intent_target[t])
      + PLAN_LOSS_W * mean_t CE(plan_logits[t], plan_target[t])

The plan term is a small addition, not a replacement: the intent is what
the player DOES, the plan is what he is TRYING to do.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from decision_brain import PlayerIntent
from role_features import ROLE_INTENTS, _role_family
from situational_brain import (
    HIDDEN_D, MEMORY_D, OUT_D, PLAN_D, PLAN_OUT_D, ROLE_D, STATE_D,
    TOKEN_D, SituationalBrain,
)


#: How much the plan head counts for, relative to the intent head.
PLAN_LOSS_W = 0.35


def _vocabulary(position: str) -> List[PlayerIntent]:
    fam = _role_family(position) or ""
    names = ROLE_INTENTS.get(fam, frozenset())
    return [i for i in PlayerIntent if i.value in names]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def intent_targets(surrogate: Any, state: np.ndarray, position: str) -> np.ndarray:
    """The table's preferred intent for each state, as an index.

    Identical in spirit to ``brain_distill._targets`` — the SAME objective,
    so the census comparison remains apples-to-apples.
    """
    vocab = _vocabulary(position)
    if not vocab:
        raise ValueError(f"no role vocabulary for position {position!r}")
    out = np.zeros(len(state), dtype=np.int64)
    for i, row in enumerate(state):
        best = None
        best_v = None
        for intent in vocab:
            v = float(surrogate.expected_success(row[:24], intent, position))
            if best_v is None or v > best_v:
                best, best_v = intent, v
        if best is None:
            continue
        out[i] = [p.value for p in PlayerIntent].index(best.value)
    return out


class Sequence:
    """One player's run of touches — the unit BPTT trains over."""

    __slots__ = ("tokens", "token_mask", "state", "role", "memory", "plan",
                 "intent", "phase", "position")

    def __init__(self, tokens, token_mask, state, role, memory, plan,
                 intent, phase, position):
        self.tokens = np.asarray(tokens, dtype=np.float64)
        self.token_mask = np.asarray(token_mask, dtype=np.float64)
        self.state = np.asarray(state, dtype=np.float64)
        self.role = np.asarray(role, dtype=np.float64)
        self.memory = np.asarray(memory, dtype=np.float64)
        self.plan = np.asarray(plan, dtype=np.float64)
        self.intent = np.asarray(intent, dtype=np.int64)
        self.phase = np.asarray(phase, dtype=np.int64)
        self.position = position

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def check(self) -> None:
        """Every array must agree on the length, or the BPTT is nonsense."""
        n = self.state.shape[0]
        for name in ("tokens", "token_mask", "role", "memory", "plan",
                     "intent", "phase"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"sequence length mismatch: {name} has {arr.shape[0]} "
                    f"steps, state has {n}")
        if self.state.shape[1] != STATE_D:
            raise ValueError(f"state width {self.state.shape[1]} != {STATE_D}")
        if self.role.shape[1] != ROLE_D:
            raise ValueError(f"role width {self.role.shape[1]} != {ROLE_D}")
        if self.memory.shape[1] != MEMORY_D:
            raise ValueError(f"memory width {self.memory.shape[1]} != {MEMORY_D}")
        if self.plan.shape[1] != PLAN_D:
            raise ValueError(f"plan width {self.plan.shape[1]} != {PLAN_D}")
        if self.tokens.shape[2] != TOKEN_D:
            raise ValueError(f"token width {self.tokens.shape[2]} != {TOKEN_D}")


def load_sequences(path: str) -> List[Sequence]:
    """Read a situational corpus (the format the collector writes)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: List[Sequence] = []
    for entry in raw.get("sequences", []):
        steps = entry["steps"]
        if not steps:
            continue
        seq = Sequence(
            tokens=[s["tokens"] for s in steps],
            token_mask=[s["token_mask"] for s in steps],
            state=[s["state"] for s in steps],
            role=[s["role"] for s in steps],
            memory=[s["memory"] for s in steps],
            plan=[s["plan"] for s in steps],
            intent=[s["intent"] for s in steps],
            phase=[s["phase"] for s in steps],
            position=entry.get("position", ""),
        )
        seq.check()
        out.append(seq)
    return out


def distill_situational(
    position: str,
    surrogate: Any,
    sequences: List[Sequence],
    epochs: int = 60,
    learning_rate: float = 0.01,
    seed: int = 123,
    plan_loss_w: float = PLAN_LOSS_W,
    verbose: bool = False,
) -> Tuple[SituationalBrain, Dict[str, Any]]:
    """Fit one position's situational brain by BPTT over its sequences.

    Returns the brain and a report (the loss curve, the agreement with
    the table, and the intent mix of the result).
    """
    if not sequences:
        raise ValueError("no sequences to train on")
    rng = np.random.default_rng(seed)
    brain = SituationalBrain.random(seed=seed)

    params = [brain.w[k] for k in sorted(brain.w)]
    keys = sorted(brain.w)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    # The intent targets come from the SURROGATE, not from the corpus, so
    # the corpus does not have to carry them.
    for seq in sequences:
        seq.intent = intent_targets(surrogate, seq.state, position)
    n_steps = sum(len(s) for s in sequences)

    loss_curve = []
    agree_curve = []
    for epoch in range(epochs):
        grads = {k: np.zeros_like(brain.w[k]) for k in keys}
        total_loss = 0.0
        agree_hits = 0
        agree_n = 0
        for seq in sequences:
            T = len(seq)
            logits, plan_logits, _h, caches = brain.forward_sequence(
                seq.tokens, seq.token_mask, seq.state, seq.role,
                seq.memory, seq.plan, np.zeros(HIDDEN_D))
            logp = _log_softmax(logits)
            plan_logp = _log_softmax(plan_logits)
            nll = -(logp[np.arange(T), seq.intent])
            pnll = -(plan_logp[np.arange(T), seq.phase])
            total_loss += float((nll + plan_loss_w * pnll).sum())

            probs = np.exp(logp)
            agree_hits += int((probs.argmax(axis=1) == seq.intent).sum())
            agree_n += T

            dlogits = probs.copy()
            dlogits[np.arange(T), seq.intent] -= 1.0
            dlogits /= T
            pprobs = np.exp(plan_logp)
            dplan = pprobs.copy()
            dplan[np.arange(T), seq.phase] -= 1.0
            dplan *= plan_loss_w / T

            g = brain.backward(dlogits, dplan, caches)
            for k in keys:
                grads[k] += g[k]

        loss_curve.append(total_loss / max(1, n_steps))
        agree_curve.append(agree_hits / max(1, agree_n))

        step = epoch + 1
        for i, k in enumerate(keys):
            gi = grads[k]
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * (gi * gi)
            mhat = m[i] / (1 - b1 ** step)
            vhat = v[i] / (1 - b2 ** step)
            brain.w[k] -= learning_rate * mhat / (np.sqrt(vhat) + eps)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:>4}  loss {loss_curve[-1]:.4f}  "
                  f"agree {agree_curve[-1]:.3f}")

    # The final mix, over the corpus.
    labels = [p.value for p in PlayerIntent]
    mix: Dict[str, int] = {}
    for seq in sequences:
        logits, _pl, _h, _c = brain.forward_sequence(
            seq.tokens, seq.token_mask, seq.state, seq.role, seq.memory,
            seq.plan, np.zeros(HIDDEN_D))
        for idx in logits.argmax(axis=1):
            mix[labels[int(idx)]] = mix.get(labels[int(idx)], 0) + 1
    total = max(1, sum(mix.values()))
    report = {
        "position": position,
        "sequences": len(sequences),
        "steps": n_steps,
        "epochs": epochs,
        "seed": seed,
        "loss_first": loss_curve[0] if loss_curve else None,
        "loss_last": loss_curve[-1] if loss_curve else None,
        "agreement_first": agree_curve[0] if agree_curve else None,
        "agreement": agree_curve[-1] if agree_curve else None,
        "mix": {k: round(v / total, 4) for k, v in sorted(mix.items())},
    }
    return brain, report
