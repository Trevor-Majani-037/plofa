"""GAMESTATE MIND DISTILLATION — train the game-state mind.

The GameStateMind takes a 12-d game state vector and outputs 4 modulation
scalars (press, risk, def, att).  We train it by distillation against
the surrogate's game-state-conditioned policy.

The target for each game state is derived from the surrogate's intent
distribution: the surrogate's intent preferences are mapped to the four
modulation scalars via a heuristic mapping.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gamestate_mind import GameStateMind, GAME_STATE_D, HIDDEN_1, HIDDEN_2, OUT_D
from critic_surrogate import load_any_surrogate


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _intent_to_mods(intent_dist: Dict[str, float]) -> np.ndarray:
    """Heuristic mapping from intent distribution to 4 modulation scalars."""
    press_weights = {
        "PROGRESSIVE_PASS": 1.2, "THROUGH_BALL": 1.3, "SWITCH": 1.1,
        "CARRY": 1.1, "DRIBBLE": 1.15, "CROSS": 1.1,
        "SHOOT": 1.05, "SAFE_PASS": 0.9, "RECYCLE": 0.85,
        "PROTECT_POSSESSION": 0.8,
    }
    risk_weights = {
        "PROGRESSIVE_PASS": 1.2, "THROUGH_BALL": 1.2, "SWITCH": 1.1,
        "CARRY": 1.1, "DRIBBLE": 1.1, "CROSS": 1.05,
        "SHOOT": 0.9, "SAFE_PASS": 0.7, "RECYCLE": 0.7,
        "PROTECT_POSSESSION": 0.6,
    }
    def_weights = {
        "PROGRESSIVE_PASS": 0.9, "THROUGH_BALL": 0.85, "SWITCH": 1.05,
        "CARRY": 1.0, "DRIBBLE": 0.95, "CROSS": 1.1,
        "SHOOT": 0.8, "SAFE_PASS": 1.2, "RECYCLE": 1.25,
        "PROTECT_POSSESSION": 1.3,
    }
    att_weights = {
        "PROGRESSIVE_PASS": 1.3, "THROUGH_BALL": 1.4, "SWITCH": 1.2,
        "CARRY": 1.1, "DRIBBLE": 1.05, "CROSS": 1.15,
        "SHOOT": 1.0, "SAFE_PASS": 0.8, "RECYCLE": 0.9,
        "PROTECT_POSSESSION": 0.7,
    }

    total = sum(intent_dist.values()) or 1.0
    press = sum(press_weights.get(k, 1.0) * v for k, v in intent_dist.items()) / total
    risk = sum(risk_weights.get(k, 1.0) * v for k, v in intent_dist.items()) / total
    def_ = sum(def_weights.get(k, 1.0) * v for k, v in intent_dist.items()) / total
    att = sum(att_weights.get(k, 1.0) * v for k, v in intent_dist.items()) / total

    return np.array([
        np.clip(0.5 + 0.5 * press, 0.3, 2.0),
        np.clip(0.5 + 0.5 * (1.0 - risk), 0.3, 2.0),
        np.clip(0.5 + 0.5 * def_, 0.3, 2.0),
        np.clip(0.5 + 0.5 * att, 0.3, 2.0),
    ], dtype=np.float64)


def _surrogate_intent_dist(surrogate, game_state_vec, position="CM"):
    """Get the surrogate's intent distribution for a game state."""
    from decision_brain import PlayerIntent
    from role_features import ROLE_INTENTS

    fam = "CM"
    vocab = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, frozenset())]

    intent_dist = {}
    # Use the surrogate's expected_success with a zero sensor vector
    # (the game state is encoded in the first 24 dims)
    sv = np.zeros(47, dtype=np.float64)
    sv[:12] = game_state_vec[:12]

    for intent in vocab:
        v = float(surrogate.expected_success(sv, intent, position))
        intent_dist[intent.value] = max(0.0, v)

    total = sum(intent_dist.values()) or 1.0
    return {k: v / total for k, v in intent_dist.items()}


def distill_gamestate_mind(
    surrogate,
    n_states: int = 2000,
    epochs: int = 80,
    learning_rate: float = 0.02,
    seed: int = 123,
    verbose: bool = False,
) -> Tuple[GameStateMind, Dict[str, Any]]:
    """Fit a GameStateMind by distillation over synthetic game states."""
    rng = np.random.default_rng(seed)
    mind = GameStateMind.random(seed=seed)

    # Generate synthetic game states
    states = []
    targets = []
    for _ in range(n_states):
        minute = rng.uniform(0, 90)
        scorediff = int(rng.integers(-3, 4))
        poss = rng.uniform(0.3, 0.7)
        xg_diff = rng.uniform(-1.5, 1.5)
        home = bool(rng.integers(0, 2))

        feat = GameStateMind.extract_features(
            minute, scorediff, poss, xg_diff, 0, 0, home
        )
        # Get surrogate's intent distribution
        intent_dist = _surrogate_intent_dist(surrogate, feat)
        mods = _intent_to_mods(intent_dist)
        target = mods

        states.append(feat)
        targets.append(target)

    states = np.stack(states)
    targets = np.stack(targets)

    # Training loop (Adam)
    mind = GameStateMind.random(seed=seed)
    m = [np.zeros_like(p) for p in [mind.w1, mind.b1, mind.w2, mind.b2, mind.w3, mind.b3]]
    v = [np.zeros_like(p) for p in [mind.w1, mind.b1, mind.w2, mind.b2, mind.w3, mind.b3]]
    b1_adam, b2_adam, eps = 0.9, 0.999, 1e-8

    loss_curve = []
    for epoch in range(epochs):
        # Forward
        h1 = _relu(states @ mind.w1 + mind.b1)
        h2 = _relu(h1 @ mind.w2 + mind.b2)
        logits = h2 @ mind.w3 + mind.b3
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))

        # MSE loss
        diff = probs - targets
        loss = float(np.mean(diff ** 2))

        # Backward
        dlogits = 2.0 * diff / len(states)
        gw3 = h2.T @ dlogits
        gb3 = dlogits.sum(axis=0)
        dh2 = dlogits @ mind.w3.T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = dh2.sum(axis=0)
        dh1 = dh2 @ mind.w2.T
        dh1[h1 <= 0.0] = 0.0
        gw1 = states.T @ dh1
        gb1 = dh1.sum(axis=0)

        grads = [gw1, gb1, gw2, gb2, gw3, gb3]
        params = [mind.w1, mind.b1, mind.w2, mind.b2, mind.w3, mind.b3]

        for i in range(6):
            m[i] = b1_adam * m[i] + (1 - b1_adam) * grads[i]
            v[i] = b2_adam * v[i] + (1 - b2_adam) * (grads[i] * grads[i])
            mhat = m[i] / (1 - b1_adam ** (epoch + 1))
            vhat = v[i] / (1 - b2_adam ** (epoch + 1))
            params[i] -= learning_rate * mhat / (np.sqrt(vhat) + eps)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}  loss {loss:.6f}")

    report = {"epochs": epochs, "loss_last": loss, "n_states": n_states}
    return mind, report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", required=True)
    p.add_argument("--out", default="brains/gamestate_mind.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--states", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    sur = load_any_surrogate(args.surrogate)
    mind, report = distill_gamestate_mind(sur, n_states=args.states, epochs=args.epochs, seed=args.seed, verbose=True)
    mind.save(args.out)
    print(json.dumps(report, indent=2))
