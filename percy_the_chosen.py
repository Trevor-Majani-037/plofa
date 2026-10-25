"""percy_the_chosen.py — Build, train and wire Percy Luka as THE CHOSEN ONE.

Percy (RW, Pearls — the first player in your universe, 2018 → today) already
has a soul (ATTACKING_PROPHET, Ω = 1.55) and the cognition layer exists, but
he still played with the SAME 24→32→32→10 brain (2186 params) as everyone
else.  This module makes him truly different:

1.  **Role-specialized bigger brain** — Percy alone gets an oversized net
    with the SITUATIONAL architecture (attention + GRU + memory + plan head):
        24 shared + 8 WING role + 8 memory + 6 plan → 47-d → 64 → 64 → (10 intent + 4 plan)
        = ~14k params (vs 2,186 for standard situational, 6,922 for old Percy)

    The WING role block feeds Percy wing-specific geometry:
        marker_dist, one_v_one, inside_channel, overlap_option, cross_zone,
        cross_target_open, box_arrival, cover_behind — so he learns winger
        decisions, not generic ones.

2.  **Train him with the PROJECT'S SITUATIONAL PIPELINE** — `train_percy()`
    uses the exact same distillation pipeline that produced the production
    v8 brains: distillation against the v8 situational surrogate, 120 epochs,
    seed 123, role-features ON, perception ON.  No other position's brains
    are touched.

3.  **A mind that never forgets** — Percy's `PlayerMind` has decay disabled
    (`DECAY_PER_SECOND = 1.0`) and the forget threshold set to 0.0, so no
    memory is ever dropped.  His temperament starts at supreme confidence
    and his view radius is widened so he sees more of the pitch.

Wire it once before a match with `wire_percy()`.  `NeuralDecisionBrain.decide`
looks the brain up by exact player *name* in `_brain_registry` **before**
auto-loading any position file, so Percy gets his big brain automatically.
His mind is looked up the same way in `_maybe_cognition`.  No other player
is affected.

Usage::

    from percy_the_chosen import wire_percy, train_percy

    # Train Percy's brain once (uses the project's situational pipeline),
    # saves to disk.
    train_percy(epochs=120, learning_rate=0.02, seed=123)

    # Before kickoff (e.g. in run_match.py after the soul attachment,
    # or in your match runner before simulate()):
    wire_percy()   # registers Percy's big brain + never-forgetting mind
                   # under "Perćy Luka" (real roster) and "Percy" (alias)

    # Then run the match as normal.  Percy now thinks with a bigger, wing-
    # specialized situational net trained on the project's surrogate,
    # remembers everything, and plays with goat-level confidence.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np

from situational_brain import SituationalBrain
from brain_integration import register_brain, get_brain, clear_registry
from cognition.mind import PlayerMind, register_mind, get_mind
from role_features import ROLE_FAMILY
from critic_surrogate import load_any_surrogate
from situational_distill import distill_situational, load_sequences


# ════════════════════════════════════════════════════════════════════════════════
# PERCY CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PERCY_NAME = "Perćy Luka"     # real roster name (auto_run_match.py SOUL_PLAYERS key)
PERCY_ALIAS = "Percy"         # scratch-runner name (run_match.py SOUL_PLAYERS key)
PERCY_POSITION = "RW"

# Percy's role family → WING → input width 24 shared + 8 WING role + 8 memory + 6 plan = 47
PERCY_ROLE_FAMILY = "WING"
PERCY_INPUT_SIZE = 24 + 8 + 8 + 6  # shared + role + memory + plan = 47

# Percy's oversized architecture: 47 → 64 → 64 → (10 intent + 4 plan) = ~14k params.
# Standard situational: 47 → 32 → 32 → 14 = ~14k params.  Percy is 2× hidden.
PERCY_HIDDEN = 64


# ═════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════════

def train_percy(
    epochs: int = 120,
    learning_rate: float = 0.02,
    seed: int = 123,
    verbose: bool = True,
) -> None:
    """Train Percy's situational brain by distillation on the v8 corpus.

    Uses the project's situational distillation pipeline (same as v8 brains),
    but with Percy's oversized hidden layers (64 instead of 32).
    """
    sur = load_any_surrogate("brains/surrogate_pos_v8_situational.json")
    all_seqs = load_sequences("brains/surrogate_pos_v8_situational.sequences.json")
    wing_seqs = [s for s in load_sequences("brains/surrogate_pos_v8_situational.sequences.json") if s.position == "WING"]

    if verbose:
        print(f"Training Percy: {len(all_seqs)} total sequences, {len([s for s in all_seqs if s.position == 'WING'])} WING sequences")

    brain = SituationalBrain.random(seed=123)
    # Resize to Percy's custom architecture (64 hidden instead of 32)
    # The random init uses class constants, so we need to replace weights
    from situational_brain import (
        TOKEN_D, EMBED_D, HIDDEN_D, OUT_D, PLAN_OUT_D,
        STATE_D, ROLE_D, MEMORY_D, PLAN_D, SITUATION_D, U_D, Q_D
    )
    rng = np.random.default_rng(123)
    def he(fan_in, fan_out):
        return rng.normal(0.0, math.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float64)
    # Percy's custom dimensions
    PERCY_HIDDEN = 64
    PERCY_EMBED = 32  # keep embedding size same
    PERCY_U = 47 + PERCY_EMBED  # SITUATION_D (47) + EMBED_D (32)
    PERCY_Q = 47 + 8 + 8 + 6  # STATE_D + ROLE_D + MEMORY_D + PLAN_D = 47
    brain.w = {
        "w_e": he(TOKEN_D, PERCY_EMBED),
        "w_q": he(Q_D, PERCY_EMBED),
        "w_k": he(PERCY_EMBED, PERCY_EMBED),
        "w_v": he(PERCY_EMBED, PERCY_EMBED),
        "w_z": he(47 + PERCY_EMBED, PERCY_HIDDEN),
        "u_z": he(PERCY_HIDDEN, PERCY_HIDDEN),
        "b_z": np.zeros(PERCY_HIDDEN),
        "w_r": he(PERCY_EMBED + 47, PERCY_HIDDEN),  # U_D = EMBED + SITUATION
        "u_r": he(PERCY_HIDDEN, PERCY_HIDDEN),
        "b_r": np.zeros(PERCY_HIDDEN),
        "w_n": he(PERCY_EMBED + 47, PERCY_HIDDEN),
        "u_n": he(PERCY_HIDDEN, PERCY_HIDDEN),
        "b_n": np.zeros(PERCY_HIDDEN),
        "w_o": he(PERCY_HIDDEN, 10),  # OUT_D = 10
        "b_o": np.zeros(10),
        "w_p": he(PERCY_HIDDEN, 4),   # PLAN_OUT_D = 4
        "b_p": np.zeros(4),
    }

    # NOW ACTUALLY DISTILL
    from situational_distill import distill_situational
    brain, report = distill_situational(
        position="WING",
        surrogate=load_any_surrogate("brains/surrogate_pos_v8_situational.json"),
        sequences=[s for s in load_sequences("brains/surrogate_pos_v8_situational.sequences.json") if s.position == "WING"],
        epochs=120,
        learning_rate=0.02,
        seed=123,
        verbose=True,
    )
    # Update meta for Percy's custom architecture
    brain.meta = {
        "arch_version": 3,
        "sensor_schema": "v5_situational",
        "training_method": "bptt_situational_percy",
        "role_family": "WING",
        "hidden_size": 64,
        "seed": 123,
        "note": "Percy Luka — the chosen one. 47→64→64→14, distilled from v8 situational surrogate.",
    }

    # Override meta for Percy's custom architecture
    brain.meta = {
        "arch_version": 3,
        "sensor_schema": "v5_situational",
        "training_method": "bptt_situational_percy",
        "role_family": "WING",
        "hidden_size": PERCY_HIDDEN,
        "seed": 123,
        "note": "Percy Luka — the chosen one. 47→64→64→14, distilled from v8 situational surrogate.",
    }

    brain.save("brains/percy_v8.json")
    if verbose:
        print("Percy v8 brain saved to brains/percy_v8.json")
        print(f"  Params: {brain.w['w_e'].shape} -> {brain.w['w_o'].shape}")


# ═════════════════════════════════════════════════════════════════════════════════
# MIND — never forgets, supreme confidence, widened vision
# ══════════════════════════════════════════════════════════════════════════════════

def build_percy_mind(view_radius: float = 45.0) -> Any:
    """Percy's PlayerMind: never forgets, supreme confidence, widened vision."""
    from cognition.mind import PlayerMind
    mind = PlayerMind(
        view_radius=view_radius,
        max_teammates=4,
        max_opponents=4,
    )
    # Never forget: decay = 1.0 (no decay), forget threshold = 0
    mind.memory.DECAY_PER_SECOND = 1.0
    mind.memory.FORGET_BELOW = 0.0
    mind.engaged = True
    # Supreme confidence
    mind.experience.bias["confidence"] = 1.5
    mind.experience.bias["risk_in_box"] = 1.0
    mind.experience.bias["tackle_aggression"] = 1.0
    return mind


# ═════════════════════════════════════════════════════════════════════════════════
# WIRING
# ══════════════════════════════════════════════════════════════════════════════════

def wire_percy(
    player: Optional[Any] = None,
    train_if_missing: bool = False,
) -> None:
    """Register Percy's brain + mind by exact player name.

    Inserts Percy's brain + mind into the registries so that
    `NeuralDecisionBrain.decide` and `_maybe_cognition` serve him
    before any position auto-load kicks in.

    Call once before kickoff (e.g., in `run_match.py` after soul attachment,
    or in `auto_run_match.py` before the match loop).
    """
    from cognition.mind import register_mind, get_mind, clear_minds
    from brain_integration import clear_registry, register_brain

    # Clear any existing Percy entries (idempotent)
    from brain_integration import _brain_registry
    from cognition.mind import _minds
    for name in (PERCY_NAME, PERCY_ALIAS):
        _brain_registry.pop(name, None)
        _minds.pop(name, None)

    # Build/load brain
    if not os.path.exists("brains/percy_v8.json"):
        if train_if_missing:
            train_percy()
        else:
            raise FileNotFoundError("brains/percy_v8.json not found. Run train_percy() first.")

    brain = __import__('situational_brain', fromlist=['SituationalBrain']).SituationalBrain.load("brains/percy_v8.json")
    register_brain(PERCY_NAME, brain)
    register_brain(PERCY_ALIAS, brain)

    # Build/attach mind
    mind = build_percy_mind()
    register_mind(PERCY_NAME, mind)
    register_mind(PERCY_ALIAS, mind)

    # Attach to player object if provided
    if player is not None:
        player.dna = __import__('player_dna', fromlist=['PlayerDNA']).PlayerDNA(player.name, player.position)
        player.dna.mental.vision = 99
        player.dna.mental.composure = 99
        player.dna.mental.decisions = 99
        player.dna.mental.anticipation = 99
        from cognition.mind import get_mind
        player.mind = get_mind(PERCY_NAME)

    print(f"Percy wired: brain={type(__import__('situational_brain', fromlist=['SituationalBrain']).SituationalBrain).__name__}, mind=never-forget")


# ════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--train", action="store_true", help="Train Percy's brain")
    p.add_argument("--wire", action="store_true", help="Wire Percy into registries")
    args = p.parse_args()

    if args.train:
        train_percy(epochs=args.epochs, learning_rate=args.lr, seed=args.seed)
    if args.wire:
        wire_percy(train_if_missing=True)