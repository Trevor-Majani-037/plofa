"""NeuralDecisionBrain — drop-in replacement for DecisionBrain.

Same external contract: ``decide(player, x, y, ...)`` returns a
``PlayerDecision``.  Internally it:

    1. Builds the 24-float sensor vector (brain_sensors).
    2. Runs the forward pass through the player's FootballBrain.
    3. Maps the 10-output softmax to a PlayerDecision with all the
       explainability fields the rest of the engine expects.

The existing deterministic layers (AttackingMatrix shot gate,
TacticalPhase regression orders, wide-combo override) remain the
authoritative override for shots and structural resets — this brain
never fires a shot itself, exactly like the heuristic DecisionBrain.

Integration point in event_chain.py (line 1594):
    Old:  from decision_brain import DecisionBrain
    New:  from brain_integration import NeuralDecisionBrain
    Call: NeuralDecisionBrain.decide(...)  — same arguments, same return.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional

from football_brain import FootballBrain, INTENT_LABELS
from brain_sensors import extract_sensors, _get_attr, _fatigue_estimate, _clamp
from decision_brain import (
    PlayerDecision, PlayerIntent, CARRY_LIKE_INTENTS,
)


# ─────────────────────────────────────────────────────────────
# INTENT MAPPING
# ─────────────────────────────────────────────────────────────

_INTENT_BY_INDEX = [PlayerIntent(label) for label in INTENT_LABELS]

# Index mapping from FootballBrain output to PlayerIntent
_IDX = {label: i for i, label in enumerate(INTENT_LABELS)}

# Always-visible intents (fallbacks that the heuristic brain never hides)
_ALWAYS_VISIBLE = frozenset({
    PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE,
    PlayerIntent.PROTECT_POSSESSION, PlayerIntent.CARRY,
})

# Risky intents (for explainability)
_RISKY = frozenset({
    PlayerIntent.THROUGH_BALL, PlayerIntent.DRIBBLE, PlayerIntent.SWITCH,
    PlayerIntent.CROSS, PlayerIntent.SHOOT,
})

# Human-readable reasons per intent
_REASONS = {
    PlayerIntent.PROGRESSIVE_PASS: "neural: line-breaking option into an advanced teammate",
    PlayerIntent.SAFE_PASS: "neural: reliable short option under the circumstances",
    PlayerIntent.THROUGH_BALL: "neural: a run he trusts and a lane he believes is open",
    PlayerIntent.SWITCH: "neural: congestion on this side, space on the far side",
    PlayerIntent.CARRY: "neural: space to advance into",
    PlayerIntent.DRIBBLE: "neural: backs himself to beat the marker",
    PlayerIntent.CROSS: "neural: delivery window from the wide crossing zone",
    PlayerIntent.SHOOT: "neural: believes the window is there",
    PlayerIntent.RECYCLE: "neural: resets the picture rather than force it",
    PlayerIntent.PROTECT_POSSESSION: "neural: shields it and buys time for support",
}


# ─────────────────────────────────────────────────────────────
# BRAIN REGISTRY (per-player brain instances)
# ─────────────────────────────────────────────────────────────

_brain_registry: Dict[str, FootballBrain] = {}

# Directory holding per-position brain files (<POSITION>.json).  When a
# player of a given position asks for a brain that isn't registered, we
# auto-load from this directory (cached per position) so every player in
# any match gets a trained brain without per-run wiring.
BRAIN_DIR: str = os.environ.get("PLOFA_BRAIN_DIR", "brains")
_pos_brain_cache: Dict[str, FootballBrain] = {}


def set_brain_dir(path: str) -> None:
    global BRAIN_DIR
    BRAIN_DIR = path
    _pos_brain_cache.clear()


def _brain_for_position(position: str) -> Optional[FootballBrain]:
    """Load and cache the brain for a position, or None if unavailable."""
    if position in _pos_brain_cache:
        return _pos_brain_cache[position]
    if not position:
        return None
    path = os.path.join(BRAIN_DIR, f"{position}.json")
    if not os.path.exists(path):
        return None
    try:
        brain = FootballBrain.load(path)
    except Exception:
        return None
    _pos_brain_cache[position] = brain
    return brain


def register_brain(player_name: str, brain: FootballBrain) -> None:
    """Bind a FootballBrain to a player name."""
    _brain_registry[player_name] = brain


def get_brain(player_name: str) -> Optional[FootballBrain]:
    """Look up a player's brain, or None if not registered."""
    return _brain_registry.get(player_name)


def clear_registry() -> None:
    """Remove all registered brains (for testing)."""
    _brain_registry.clear()


def load_brains_from_file(path: str) -> int:
    """Load a JSON file mapping player_name → serialized brain.

    Expected format: {"PlayerName": {brain dict}, ...}
    Returns the number of brains loaded.
    """
    import json
    with open(path, "r") as f:
        data = json.load(f)
    count = 0
    for name, brain_data in data.items():
        register_brain(name, FootballBrain.deserialize(brain_data))
        count += 1
    return count


def save_brains_to_file(path: str) -> None:
    """Write all registered brains to a JSON file."""
    import json
    data = {name: brain.serialize() for name, brain in _brain_registry.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────
# RISK ESTIMATION (lightweight heuristic for explainability)
# ─────────────────────────────────────────────────────────────

def _estimate_risk(intent: PlayerIntent, near_def_dist: float, fatigue: float) -> float:
    """Crude risk estimate for the chosen intent — not used for selection,
    only for the PlayerDecision.risk_level field."""
    base = {
        PlayerIntent.SAFE_PASS: 0.12,
        PlayerIntent.RECYCLE: 0.08,
        PlayerIntent.PROTECT_POSSESSION: 0.15,
        PlayerIntent.CARRY: 0.20,
        PlayerIntent.PROGRESSIVE_PASS: 0.35,
        PlayerIntent.SWITCH: 0.45,
        PlayerIntent.DRIBBLE: 0.40,
        PlayerIntent.THROUGH_BALL: 0.55,
        PlayerIntent.CROSS: 0.50,
        PlayerIntent.SHOOT: 0.60,
    }.get(intent, 0.30)
    if near_def_dist is not None and near_def_dist < 3.0:
        base += 0.10
    base += fatigue * 0.10
    return _clamp(base, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# DECISION QUALITY (vs the full probability distribution)
# ─────────────────────────────────────────────────────────────

def _decision_quality(chosen_idx: int, probs: list) -> float:
    """How confident was the network?  Peaked distribution = high quality."""
    chosen_prob = probs[chosen_idx]
    # Sort descending — if the top two are close, quality is lower
    sorted_probs = sorted(probs, reverse=True)
    if len(sorted_probs) < 2:
        return 1.0
    gap = sorted_probs[0] - sorted_probs[1]
    return _clamp(0.5 + gap * 2.0)


def _decision_temperature(player: Any, under_pressure: bool, fatigue: float) -> float:
    """Temperature for softmax sampling — returns 0 for pure argmax.

    Scaled by the player's decisions/composure DNA: an elite decision-maker
    samples close to argmax (low temp), a flustered/limited one samples more
    broadly.  Pressure and fatigue raise the temperature (worse judgement).
    """
    mental = getattr(getattr(player, "dna", None), "mental", None)
    decisions = _get_attr(mental, "decisions", 55.0)
    composure = _get_attr(mental, "composure", 55.0)
    # base is 0 (argmax) for a 100-decisions player, rising to ~0.4 for a
    # very low-decisions player.  Graduated across the attribute range.
    base = max(0.0, (100.0 - decisions) / 250.0)
    if under_pressure:
        base *= (1.0 + (1.0 - composure / 100.0) * 0.8)
    base *= (1.0 + fatigue * 0.5)
    return base


def _sample_from_probs(probs: Any, temperature: float) -> int:
    """Sample an intent index from a probability distribution.

    temperature == 0 → argmax (deterministic).  temperature > 0 → softmax
    sampling with sharper distribution, giving variety across touches.
    """
    import numpy as _np
    p = _np.asarray(probs, dtype=_np.float64)
    if temperature <= 1e-6:
        return int(_np.argmax(p))
    # renormalise with temperature to sharpen/soften
    tempered = _np.exp(_np.log(p + 1e-12) / temperature)
    tempered = tempered / tempered.sum()
    r = random.random()
    cum = 0.0
    for i, prob in enumerate(tempered):
        cum += prob
        if r <= cum:
            return i
    return int(_np.argmax(tempered))


def _get_attr(obj: Any, path: str, default: float) -> float:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return float(cur) if cur is not None else default


# ─────────────────────────────────────────────────────────────
# NeuralDecisionBrain
# ─────────────────────────────────────────────────────────────

class NeuralDecisionBrain:
    """Drop-in replacement for DecisionBrain using a per-player
    feedforward neural network.

    Same ``decide()`` signature, same ``PlayerDecision`` return type.
    The only difference is HOW the intent is selected: a learned
    forward pass instead of hand-crafted heuristic scoring + softmax.
    """

    @staticmethod
    def decide(
        player: Any,
        x: float,
        y: float,
        teammates: List[Any],
        defenders: List[Any],
        position_engine: Any,
        team_profile: Any,
        under_pressure: bool,
        attacks_right: bool,
        game_state: Any,
        minute: float = 45.0,
        soul: Any = None,
        record_trace: bool = False,
    ) -> PlayerDecision:
        # ── Look up the player's brain ─────────────────────────
        brain = get_brain(getattr(player, "name", ""))
        if brain is None:
            # Auto-load a trained brain for this player's position if one
            # is available (cached per position).  Only fall back to a
            # random brain if no trained net exists for the role.
            brain = _brain_for_position(getattr(player, "position", ""))
            if brain is not None:
                register_brain(getattr(player, "name", ""), brain)
        if brain is None:
            # Fallback: generate a random brain on the fly so the
            # match never crashes.  In production all players should
            # be registered before kickoff.
            brain = FootballBrain.random()
            register_brain(getattr(player, "name", "unknown"), brain)

        # ── Build sensor vector ────────────────────────────────
        score_diff = 0
        gs_name = getattr(game_state, "name", "LEVEL")
        if "HOME_AHEAD" in gs_name or "HOME_CHASE" in gs_name:
            score_diff = 1
        elif "AWAY_AHEAD" in gs_name or "AWAY_CHASE" in gs_name:
            score_diff = -1

        sensors = extract_sensors(
            player, x, y, teammates, defenders, position_engine,
            under_pressure, attacks_right, game_state, minute,
            team_possession=True, score_diff=score_diff,
        )

        # ── Forward pass ───────────────────────────────────────
        probs = brain.forward(sensors)
        fatigue = _fatigue_estimate(player, minute)

        # Temperature-based probabilistic sampling (not argmax): a high
        # composure / decisions player commits more firmly to the
        # network's top pick; a flustered or low-decisions player is
        # more likely to sample a lower-probability option.  This gives
        # the same bounded-variety feel the heuristic DecisionBrain had,
        # but the probabilities themselves come from the learned net.
        temperature = _decision_temperature(player, under_pressure, fatigue)
        chosen_idx = _sample_from_probs(probs, temperature)
        chosen_prob = float(probs[chosen_idx])
        intent = _INTENT_BY_INDEX[chosen_idx]

        # ── Build PlayerDecision ───────────────────────────────
        near_def = _nearest_defender_dist(x, y, defenders, position_engine)

        risk = _estimate_risk(intent, near_def, fatigue)
        quality = _decision_quality(chosen_idx, probs.tolist())
        action = "CARRY" if intent in CARRY_LIKE_INTENTS else "PASS"

        # Perceived intents: every intent with probability > 5%
        perceived = frozenset(
            _INTENT_BY_INDEX[i] for i, p in enumerate(probs) if p > 0.05
        )

        # Find best target (same logic as heuristic brain for backward compat)
        target = _find_target(
            intent, player, x, y, teammates, defenders,
            position_engine, attacks_right,
        )

        trace = None
        if record_trace:
            trace = {
                "player": getattr(player, "name", "?"),
                "fatigue": round(fatigue, 3),
                "sensor_vector": sensors.tolist(),
                "output_probs": {
                    label: round(float(probs[i]), 4)
                    for i, label in enumerate(INTENT_LABELS)
                },
                "chosen_idx": chosen_idx,
                "choice_probability": round(chosen_prob, 4),
            }

        return PlayerDecision(
            intent=intent,
            action=action,
            confidence=round(_clamp(chosen_prob), 3),
            reason=_REASONS.get(intent, "neural: on-ball decision"),
            risk_level=round(risk, 3),
            decision_quality=round(quality, 3),
            evaluation_error=0.0,  # neural nets don't have an "objective" to compare against at decision time
            is_error=False,
            target=target,
            perceived_intents=perceived,
            trace=trace,
        )


# ─────────────────────────────────────────────────────────────
# TARGET SELECTION (finds best teammate for the chosen intent)
# ─────────────────────────────────────────────────────────────

def _find_target(
    intent: PlayerIntent,
    player: Any, x: float, y: float,
    teammates: List[Any], defenders: List[Any],
    position_engine: Any, attacks_right: bool,
) -> Any:
    """Find the most relevant teammate for the chosen intent.
    This is a lightweight geometric lookup, not a scoring system."""
    if intent in (PlayerIntent.CARRY, PlayerIntent.DRIBBLE,
                  PlayerIntent.PROTECT_POSSESSION):
        return None

    if intent == PlayerIntent.SAFE_PASS:
        return _nearest_teammate(x, y, teammates, position_engine)

    if intent == PlayerIntent.RECYCLE:
        return _deepest_teammate(x, y, teammates, position_engine, attacks_right)

    if intent == PlayerIntent.PROGRESSIVE_PASS:
        return _best_forward(x, y, teammates, defenders, position_engine, attacks_right)

    if intent == PlayerIntent.THROUGH_BALL:
        return _best_forward(x, y, teammates, defenders, position_engine, attacks_right)

    if intent == PlayerIntent.SWITCH:
        return _wide_teammate(x, y, teammates, defenders, position_engine)

    return None


def _nearest_teammate(x, y, teammates, position_engine):
    best, best_dist = None, 999.0
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = position_engine.get_position(t.name) if position_engine else (x, y)
        d = math.hypot(tx - x, ty - y)
        if 1.0 < d < 22.0 and d < best_dist:
            best_dist = d
            best = t
    return best


def _deepest_teammate(x, y, teammates, position_engine, attacks_right):
    best, best_depth = None, -999.0
    for t in teammates or []:
        tx, ty = position_engine.get_position(t.name) if position_engine else (x, y)
        depth = (tx - x) if attacks_right else (x - tx)
        if depth > best_depth:
            best_depth = depth
            best = t
    return best


def _best_forward(x, y, teammates, defenders, position_engine, attacks_right):
    best, best_val = None, -1.0
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = position_engine.get_position(t.name) if position_engine else (x, y)
        progress = (tx - x) if attacks_right else (x - tx)
        if progress < 4.0:
            continue
        # rough openness
        open_val = 0.5
        for d in defenders or []:
            if getattr(d, "position", "") == "GK":
                continue
            dx, dy = position_engine.get_position(d.name) if position_engine else (tx + 10, ty)
            dist = math.hypot(dx - tx, dy - ty)
            open_val = min(open_val, _clamp((dist - 1.5) / 8.5))
        val = _clamp(progress / 35.0) * 0.55 + open_val * 0.45
        if val > best_val:
            best_val = val
            best = t
    return best


def _wide_teammate(x, y, teammates, defenders, position_engine):
    best, best_val = None, -1.0
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = position_engine.get_position(t.name) if position_engine else (x, y)
        width_gap = abs(ty - y)
        if width_gap < 22.0:
            continue
        open_val = 0.5
        for d in defenders or []:
            if getattr(d, "position", "") == "GK":
                continue
            dx, dy = position_engine.get_position(d.name) if position_engine else (tx + 10, ty)
            dist = math.hypot(dx - tx, dy - ty)
            open_val = min(open_val, _clamp((dist - 1.5) / 8.5))
        val = _clamp(width_gap / 55.0) * 0.5 + open_val * 0.5
        if val > best_val:
            best_val = val
            best = t
    return best


def _nearest_defender_dist(x, y, defenders, position_engine):
    best = None
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        if position_engine is None:
            continue
        try:
            dx, dy = position_engine.get_position(d.name)
        except Exception:
            continue
        dist = math.hypot(dx - x, dy - y)
        if best is None or dist < best:
            best = dist
    return best


# needed for numpy import in decide()
import numpy as np  # noqa: E402
