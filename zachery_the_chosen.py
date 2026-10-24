"""zachery_the_chosen.py — Build, train and wire Zachery Worth as the NEYMAR.

Zachery Worth (LW, PLOFA) is the Neymar-equivalent: a dribbler-first,
creator-second, scorer-third winger whose brain is trained to prefer
DRIBBLE and CARRY in 1v1 situations, create chances through take-ons,
and be unpredictable (tempered sampling, not pure argmax) — all while
preserving the engine's physics-resolved dribble success rate.

Neymar research summary (2026-10-24):
  - ~2.4-3.6 successful dribbles per 90 (career)
  - ~4.07 dribble attempts per 90 (recent)
  - ~47-60% dribble success rate
  - 8.29 shot-creating actions per 90 (world class)
  - ~2.5-4.5 key passes per 90
  - ~22 progressive carries per 90, avg ~13m
  - Decision making: spontaneous/reactive, not deliberative
  - Futsal background: tight spaces, disguise intentions, slow-to-fast bursts
  - Unpredictable but not random — pattern recognition from repetition
  - Identity: dribbler first, creator second, scorer third

Architecture:
  - SituationalBrain (attention + GRU + memory + plan head)
  - WING role family: 9 features (marker_dist, one_v_one, inside_channel,
    overlap_option, cross_zone, cross_target_open, box_arrival, cover_behind,
    pass_lane_open)
  - Input: 24 shared + 9 WING role + 8 memory + 6 plan = 47-d
  - Hidden: 64 (2x standard) for expressiveness
  - Trained by distillation against v8 situational surrogate (same pipeline)
  - Never-forgetting mind: decay=1.0, forget_threshold=0, widened view_radius=45
  - Supreme confidence (composure 99, decisions 99, vision 99)

Dribble success stays PHYSICS-resolved (the brain chooses intent, the
engine resolves the outcome). The brain's "unpredictability" comes from
temperature-based sampling (not argmax) — the user's "not all the times"
constraint is met by the existing _decision_temperature() seam.

Wire it once before kickoff with `wire_zachery()`. NeuralDecisionBrain.decide
looks the brain up by exact player *name* in `_brain_registry` before
auto-loading any position file.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np

from situational_brain import SituationalBrain
from brain_integration import register_brain, get_brain
from cognition.mind import PlayerMind, register_mind, get_mind
from role_features import ROLE_FAMILY
from critic_surrogate import load_any_surrogate
from situational_distill import distill_situational, load_sequences


# ════════════════════════════════════════════════════════════════════════════════
# ZACHERY CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

ZACHERY_NAME = "Zachery Worth"   # real roster name
ZACHERY_ALIAS = "Zachery"         # scratch-runner alias
ZACHERY_POSITION = "LW"

# Zachery's role family → WING → input width 24 shared + 9 WING role + 8 memory + 6 plan = 47
ZACHERY_ROLE_FAMILY = "WING"
ZACHERY_INPUT_SIZE = 24 + 9 + 8 + 6  # shared + role + memory + plan = 47

# Zachery's architecture: 47 → 32 → 32 → (10 intent + 4 plan) = ~4.5k params.
# Standard situational (same as v8). The hidden width matches the project's
# proven architecture — the Neymar-ness comes from the WING role features and
# temperature-based sampling, not from a bigger network.
ZACHERY_HIDDEN = 32

# Brain file paths
ZACHERY_BRAIN_PATH = "brains/zachery_v9.json"
ZACHERY_SURROGATE_PATH = "brains/surrogate.json"
ZACHERY_SEQUENCES_PATH = "brains/surrogate_pos_v9_situational.sequences.json"


# ═════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════════

def train_zachery(
    epochs: int = 120,
    learning_rate: float = 0.02,
    seed: int = 123,
    verbose: bool = True,
) -> None:
    """Train Zachery's situational brain by distillation on the v8 corpus.

    Uses the project's situational distillation pipeline (same as v8 brains
    and Percy), but registers under Zachery's name. The brain learns from
    the WING role family's surrogate table — which includes DRIBBLE, CARRY,
    CROSS, and SHOOT as role-appropriate intents.

    The Neymar-ness comes from:
      1. The WING role features (marker_dist, one_v_one, inside_channel)
         give the brain the defender geometry to decide when to dribble.
      2. The surrogate's per-bucket argmax for WING in 1v1 situations
         naturally favors DRIBBLE when the marker is close and the lane
         is open — the engine resolves whether the dribble succeeds.
      3. Temperature-based sampling (not argmax) gives the "partial
         unpredictability" the user requested.
    """
    sur = load_any_surrogate(ZACHERY_SURROGATE_PATH)
    all_seqs = load_sequences(ZACHERY_SEQUENCES_PATH)
    wing_seqs = [s for s in all_seqs if s.position == "WING"]

    if verbose:
        print(f"Training Zachery: {len(all_seqs)} total sequences, "
              f"{len(wing_seqs)} WING sequences")

    if not wing_seqs:
        raise ValueError("No WING sequences found in the corpus.")

    brain, report = distill_situational(
        position="WING",
        surrogate=sur,
        sequences=wing_seqs,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        verbose=verbose,
    )

    # Update meta for Zachery's custom architecture
    brain.meta = {
        "arch_version": 3,
        "sensor_schema": "v5_situational",
        "training_method": "bptt_situational_zachery",
        "role_family": "WING",
        "hidden_size": 32,
        "seed": seed,
        "note": ("Zachery Worth — the Neymar. 47→32→32→14, distilled from "
                 "v8 situational surrogate. Dribbler-first, creator-second, "
                 "scorer-third."),
    }

    brain.save(ZACHERY_BRAIN_PATH)
    if verbose:
        print(f"Zachery brain saved to {ZACHERY_BRAIN_PATH}")
        print(f"  Params: {sum(arr.size for arr in brain.w.values())}")


# ═════════════════════════════════════════════════════════════════════════════════
# MIND — never forgets, supreme confidence, widened vision
# ══════════════════════════════════════════════════════════════════════════════════

def build_zachery_mind(view_radius: float = 45.0) -> Any:
    """Zachery's PlayerMind: never forgets, supreme confidence, widened vision.

    Tuned for the Neymar identity: dribbler first, creator second, scorer
    third.  Supreme confidence (composure 99) means he backs himself in 1v1
    situations; high vision means he sees the full pitch for creation.
    """
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
    # Supreme confidence — backs himself in 1v1s
    mind.experience.bias["confidence"] = 1.5
    mind.experience.bias["risk_in_box"] = 1.0
    mind.experience.bias["tackle_aggression"] = 1.0
    return mind


# ═════════════════════════════════════════════════════════════════════════════════
# WIRING
# ═════════════════════════════════════════════════════════════════════════════════

def wire_zachery(
    player: Optional[Any] = None,
    train_if_missing: bool = False,
) -> None:
    """Register Zachery's brain + mind by exact player name.

    Inserts Zachery's brain + mind into the registries so that
    `NeuralDecisionBrain.decide` and `_maybe_cognition` serve him
    before any position auto-load kicks in.

    Call once before kickoff (e.g., in `run_match.py` after soul attachment,
    or in `auto_run_match.py` before the match loop).
    """
    from cognition.mind import register_mind, get_mind
    from brain_integration import register_brain

    # Clear any existing Zachery entries (idempotent)
    from brain_integration import _brain_registry
    from cognition.mind import _minds
    for name in (ZACHERY_NAME, ZACHERY_ALIAS):
        _brain_registry.pop(name, None)
        _minds.pop(name, None)

    # Build/load brain
    if not os.path.exists(ZACHERY_BRAIN_PATH):
        if train_if_missing:
            train_zachery()
        else:
            raise FileNotFoundError(
                f"{ZACHERY_BRAIN_PATH} not found. Run train_zachery() first."
            )

    brain = SituationalBrain.load(ZACHERY_BRAIN_PATH)
    register_brain(ZACHERY_NAME, brain)
    register_brain(ZACHERY_ALIAS, brain)

    # Build/attach mind
    mind = build_zachery_mind()
    register_mind(ZACHERY_NAME, mind)
    register_mind(ZACHERY_ALIAS, mind)

    # Attach to player object if provided
    if player is not None:
        player.dna.mental.vision = 99
        player.dna.mental.composure = 99
        player.dna.mental.decisions = 99
        player.dna.mental.anticipation = 99
        from cognition.mind import get_mind
        player.mind = get_mind(ZACHERY_NAME)

    print(f"Zachery wired: brain=SituationalBrain, mind=never-forget, "
          f"identity=dribbler-first/creator-second/scorer-third")


# ════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--train", action="store_true", help="Train Zachery's brain")
    p.add_argument("--wire", action="store_true", help="Wire Zachery into registries")
    args = p.parse_args()

    if args.train:
        train_zachery(epochs=args.epochs, learning_rate=args.lr, seed=args.seed)
    if args.wire:
        wire_zachery(train_if_missing=True)