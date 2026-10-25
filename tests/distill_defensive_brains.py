#!/usr/bin/env python
"""Distill situational defensive brains from surrogate."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_FAMILY_MAP = {
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "DM", "CM": "DM", "CAM": "DM",
    "GK": "GK",
}

DEF_ACTIONS = ["tackle", "interception", "clearance", "block"]
ACTION_TO_IDX = {a: i for i, a in enumerate(DEF_ACTIONS)}

# Surrogate bucket function
def bucket_from_state(state: np.ndarray) -> str:
    if len(state) < 24:
        return "Z0D0A0O0E0F0M0"

    pressure = state[16] if len(state) > 16 else 0
    zone = 2 if pressure > 0.7 else (1 if pressure > 0.3 else 0)

    final_third = state[12] if len(state) > 12 else 0
    danger = 2 if final_third > 0.7 else (1 if final_third > 0.3 else 0)

    own_half = state[13] if len(state) > 13 else 0
    attack_dir = 1 if own_half > 0.5 else 0

    space = state[9] if len(state) > 9 else 0
    opp_near = 2 if space < 0.3 else (1 if space < 0.7 else 0)

    central = state[15] if len(state) > 15 else 0
    engage = 1 if central > 0.5 else 0

    goal_close = state[14] if len(state) > 14 else 0
    foul = 1 if goal_close > 0.5 else 0

    momentum = 1 if pressure > 0.5 else 0

    return f"Z{zone}D{danger}A{attack_dir}O{opp_near}E{engage}F{foul}M{momentum}"


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════════

class SituationalDefensiveBrain:
    """Situational brain for defensive actions: 51 -> 64 -> 64 -> 4."""

    def __init__(self, input_dim: int = 51, hidden: int = 64, output_dim: int = 4, seed: int = 123):
        rng = np.random.default_rng(seed)
        # Xavier initialization
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, hidden)).astype(np.float64)
        self.b2 = np.zeros(hidden, dtype=np.float64)
        self.w3 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, output_dim)).astype(np.float64)
        self.b3 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (..., input_dim). Returns logits shape: (..., output_dim)."""
        h1 = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)  # ReLU
        logits = h2 @ self.w3 + self.b3
        return logits

    def forward_with_intermediates(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning intermediates for BPTT."""
        h1_pre = x @ self.w1 + self.b1
        h1 = np.maximum(0, h1_pre)
        h2_pre = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, h2_pre)
        logits = h2 @ self.w3 + self.b3
        return logits, h1, h2, h1_pre, h2_pre

    def predict(self, x: np.ndarray) -> int:
        """Predict action index (argmax)."""
        logits = self.forward(x)
        return int(np.argmax(logits))

    def save(self, path: str):
        np.savez(path,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3,
                 meta=json.dumps({
                     "input_dim": self.w1.shape[0],
                     "hidden": self.w1.shape[1],
                     "output_dim": self.w3.shape[1],
                 }))

    @classmethod
    def load(cls, path: str) -> "SituationalDefensiveBrain":
        data = np.load(path, allow_pickle=True)
        meta = json.loads(data["meta"].item())
        brain = cls(meta["input_dim"], meta["hidden"], meta["output_dim"])
        brain.w1 = data["w1"]
        brain.b1 = data["b1"]
        brain.w2 = data["w2"]
        brain.b2 = data["b2"]
        brain.w3 = data["w3"]
        brain.b3 = data["b3"]
        return brain


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS AND GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax with numerical stability."""
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross entropy loss. targets shape: (batch, num_classes) one-hot."""
    probs = softmax(logits)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.mean(np.sum(targets * np.log(probs), axis=-1))


def backward(logits: np.ndarray, targets: np.ndarray,
             h1: np.ndarray, h2: np.ndarray, h1_pre: np.ndarray, h2_pre: np.ndarray,
             x: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> Tuple:
    """Compute gradients for one batch."""
    batch_size = logits.shape[0]

    # Softmax gradient
    probs = softmax(logits)
    dlogits = (probs - targets) / batch_size  # (batch, 4)

    # Layer 3
    dw3 = h2.T @ dlogits
    db3 = np.sum(dlogits, axis=0)

    # Layer 2
    dh2 = dlogits @ w3.T
    dh2_pre = dh2 * (h2_pre > 0)  # ReLU gradient
    dw2 = h1.T @ dh2_pre
    db2 = np.sum(dh2_pre, axis=0)

    # Layer 1
    dh1 = dh2_pre @ w2.T
    dh1_pre = dh1 * (h1_pre > 0)  # ReLU gradient
    dw1 = x.T @ dh1_pre
    db1 = np.sum(dh1_pre, axis=0)

    return dw1, db1, dw2, db2, dw3, db3


# ═══════════════════════════════════════════════════════════════════════════════
# DISTILLER
# ═══════════════════════════════════════════════════════════════════════════════

class DefensiveDistiller:
    """Distill situational brain from surrogate."""

    def __init__(self, surrogate_path: str, family: str, seed: int = 123):
        self.family = family
        with open(surrogate_path, "r") as f:
            self.surrogate = json.load(f)

        # Build training data from surrogate
        self.train_states: List[np.ndarray] = []
        self.train_targets: List[np.ndarray] = []

        self._build_training_data()

        # Create brain
        self.brain = SituationalDefensiveBrain(input_dim=51, hidden=64, output_dim=4, seed=seed)
        self.lr = 0.001

    def _build_training_data(self):
        """Build training pairs from surrogate."""
        table = self.surrogate.get("table", {})

        # Use family-specific table and global fallback
        family_table = table.get(self.family, {})
        global_table = table.get("", {})

        for bucket, actions in family_table.items():
            # Create a representative state for this bucket
            state = self._bucket_to_state(bucket)
            if state is None:
                continue

            # Get action values
            action_vals = {}
            for action, val in actions.items():
                action_vals[action] = val

            # Also check global fallback for missing actions
            if bucket in global_table:
                for action, val in global_table[bucket].items():
                    if action not in action_vals:
                        action_vals[action] = val

            if not action_vals:
                continue

            # Find best action
            best_action = max(action_vals, key=action_vals.get)
            best_idx = ACTION_TO_IDX[best_action]

            # Build full 51-dim input
            # state(24) + role(13) + memory(8) + plan(6) = 51
            role_block = self._family_role_block()
            memory_block = np.zeros(8, dtype=np.float64)
            plan_block = np.zeros(6, dtype=np.float64)
            state_vec = self._bucket_to_state(bucket)
            if state_vec is None:
                state_vec = np.zeros(24, dtype=np.float64)

            full_input = np.concatenate([state_vec, role_block, memory_block, plan_block])
            assert len(full_input) == 51

            # Target is one-hot for best action
            target = np.zeros(4, dtype=np.float64)
            target[best_idx] = 1.0

            self.train_states.append(full_input)
            self.train_targets.append(target)

    def _bucket_to_state(self, bucket: str) -> np.ndarray:
        """Convert bucket string to approximate state vector."""
        # Parse bucket: Z{zone}D{danger}A{attack}O{opp}E{engage}F{foul}M{momentum}
        import re
        match = re.match(r"Z(\d)D(\d)A(\d)O(\d)E(\d)F(\d)M(\d)", bucket)
        if not match:
            return np.zeros(24, dtype=np.float64)

        zone, danger, attack, opp, engage, foul, momentum = map(int, match.groups())

        state = np.zeros(24, dtype=np.float64)
        # Map bucket values back to approximate state features
        state[16] = zone / 2.0 * 0.7  # pressure
        state[12] = danger / 2.0 * 0.7  # final_third
        state[13] = 0.8 if attack else 0.2  # own_half
        state[9] = 0.2 if opp == 2 else (0.5 if opp == 1 else 0.8)  # space
        state[15] = 0.8 if engage else 0.2  # central
        state[14] = 0.8 if foul else 0.2  # goal_close
        state[11] = 0.8 if momentum else 0.2  # momentum proxy

        return state

    def _family_role_block(self) -> np.ndarray:
        """Create role block for this family (13 dims to match ROLE_D)."""
        block = np.zeros(13, dtype=np.float64)
        # Map family to action vocab
        fam_vocab = {
            "CB": {"RECYCLE", "PROGRESSIVE_PASS", "SAFE_PASS", "PROTECT_POSSESSION", "SWITCH", "CARRY", "DRIBBLE", "THROUGH_BALL"},
            "FB": {"RECYCLE", "PROGRESSIVE_PASS", "SAFE_PASS", "PROTECT_POSSESSION", "SWITCH", "CROSS", "DRIBBLE", "CARRY", "THROUGH_BALL"},
            "DM": {"RECYCLE", "PROGRESSIVE_PASS", "SAFE_PASS", "PROTECT_POSSESSION", "SWITCH", "CARRY", "DRIBBLE", "THROUGH_BALL"},
            "GK": {"RECYCLE", "DRIBBLE", "SAFE_PASS", "PROTECT_POSSESSION", "THROUGH_BALL", "SWITCH", "CARRY"},
        }
        from decision_brain import PlayerIntent
        vocab = fam_vocab.get(self.family, frozenset())
        for i, intent in enumerate(PlayerIntent):
            if intent.value in vocab:
                block[i] = 1.0
        return block

    def distill(self, epochs: int = 80, batch_size: int = 32) -> List[float]:
        """Run distillation training."""
        n = len(self.train_states)
        if n == 0:
            print(f"No training data for {self.family}")
            return []

        print(f"Distilling {self.family}: {n} samples, {epochs} epochs")

        losses = []
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                x_batch = np.stack([self.train_states[j] for j in batch_idx])
                y_batch = np.stack([self.train_targets[j] for j in batch_idx])

                # Forward
                logits, h1, h2, h1_pre, h2_pre = self.brain.forward_with_intermediates(x_batch)

                # Loss
                loss = cross_entropy_loss(logits, y_batch)
                epoch_loss += loss

                # Backward
                dw1, db1, dw2, db2, dw3, db3 = backward(
                    logits, y_batch, h1, h2, h1_pre, h2_pre,
                    x_batch, self.brain.w1, self.brain.w2, self.brain.w3
                )

                # Update
                self.brain.w1 -= self.lr * dw1
                self.brain.b1 -= self.lr * db1
                self.brain.w2 -= self.lr * dw2
                self.brain.b2 -= self.lr * db2
                self.brain.w3 -= self.lr * dw3
                self.brain.b3 -= self.lr * db3

                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            losses.append(avg_loss)

            if epoch % 10 == 0:
                # Accuracy
                preds = np.array([self.brain.predict(s) for s in self.train_states])
                targets = np.argmax(self.train_targets, axis=1)
                acc = np.mean(preds == targets)
                print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.3f}")

        return losses


def main():
    surrogate_path = "brains_def/surrogate_situational.json"
    out_dir = "brains_def/situational"
    os.makedirs(out_dir, exist_ok=True)

    for family in ["CB", "FB", "DM", "GK"]:
        print(f"\n=== Distilling {family} ===")
        distiller = DefensiveDistiller(surrogate_path, family)
        losses = distiller.distill(epochs=80, batch_size=16)

        # Save brain
        out_path = os.path.join(out_dir, f"{family}.npz")
        distiller.brain.save(out_path)
        print(f"Saved {family} brain to {out_path}")

    print("\nAll brains distilled!")


if __name__ == "__main__":
    import re
    main()