"""PlayerMind — a PLOFA player's subjective self.

Composes the three TOLAND cognition pillars on top of PLOFA's own neural
brains:

    * senses      — FOV gate: which actors survive into the sensor block
    * memory      — episodic recollection of match events (theirs + against)
    * experience  — temperament biases (risk_in_box, tackle_aggression,
                    confidence) folded from memory into decision sampling

The mind never generates a decision by itself.  It sits BETWEEN the
forward pass of the evolved ``FootballBrain`` and the temperature sampling:
it decides *what the player perceives* and *how boldly he samples the
distribution the brain produced*.  Same brain, same geometry, two different
players → different choices, because one remembers.

A player with no registered mind (``engaged=False``) is completely
unaffected — the decision path is byte-identical to vanilla
``NeuralDecisionBrain``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .senses import VisionSystem
from .memory import MemorySystem, MemoryEvent
from .experience import ExperienceModel


# ─────────────────────────────────────────────────────────────
# EVENT → MEMORY TAG MAPPING (duck-typed on event_type.name)
# ─────────────────────────────────────────────────────────────
# PLOFA MatchEvents carry ``event_type`` (an EventType enum), ``player``
# (the primary actor), ``secondary_player`` and ``metadata``.  The observer
# is written duck-typed so the cognition package stays import-light: it
# matches on the type name string and never imports the match engine.
# ─────────────────────────────────────────────────────────────

# tag → (event-type names tracked for THIS player)
_TAG_EVENTS = {
    "foul_in_box": {
        "FOUL_COMMITTED",
    },
    # TOLAND's verbatim tag name: ExperienceModel.apply_memory recalls
    # ``clean_tackle_win`` so wins feed confidence/aggression upward.
    "clean_tackle_win": {
        "TACKLE_WON", "INTERCEPTION", "PRESS_SUCCESS",
        "BALL_RECOVERY", "CLEARANCE", "BLOCK",
    },
    "lost_duel": {
        "TACKLE_LOST", "DRIBBLE_FAIL", "DISPOSSESSED",
        "MISCONTROL",
    },
    "big_chance_missed": {
        "SHOT_OFF_TARGET", "SHOT_BLOCKED", "HIT_WOODWORK",
        "PENALTY_MISSED",
    },
    "big_chance_scored": {
        "GOAL", "PENALTY_SCORED",
    },
    "conceded": {
        "SAVE",  # keeper gets credit for a save
    },
}

# Default severity weights per tag type.
_TAG_SEVERITY = {
    "foul_in_box": 1.2,
    "clean_tackle_win": 1.0,
    "lost_duel": 1.0,
    "big_chance_missed": 1.6,
    "big_chance_scored": 2.0,
    "conceded": 1.0,
}


class PlayerMind:
    """One player's senses + memory + temperament."""

    def __init__(
        self,
        view_radius: float = 25.0,
        max_teammates: int = 2,
        max_opponents: int = 2,
    ) -> None:
        self.senses = VisionSystem(
            view_radius=view_radius,
            max_teammates=max_teammates,
            max_opponents=max_opponents,
        )
        self.memory = MemorySystem()
        self.experience = ExperienceModel()
        self.engaged = True

    # ── PERCEPTION GATE ─────────────────────────────────────
    def perceive(
        self,
        player: Any,
        x: float, y: float,
        teammates: List[Any],
        defenders: List[Any],
        position_engine: Any,
    ):
        """Filter the scene to what this player can actually see."""
        return self.senses.filter_frame(
            player, x, y, teammates, defenders, position_engine,
        )

    # ── MEMORY OBSERVER ─────────────────────────────────────
    def observe(self, event: Any, dt: float = 1.0) -> None:
        """Consume a PLOFA MatchEvent (duck-typed) into episodic memory.

        The observer dispatcher (cognition_brain._observe_event) already
        routed the event to THIS mind's owner, so we only need to classify
        the type name and remember it.  The engine's ``_absorb_chain`` calls
        this for every event once ``engage_mind_observer()`` is on.
        """
        type_name = getattr(getattr(event, "event_type", None), "name", "")
        for tag, names in _TAG_EVENTS.items():
            if type_name in names:
                self._remember(tag, event, dt)

    def _remember(self, tag: str, event: Any, dt: float) -> None:
        meta = getattr(event, "metadata", None) or {}
        x = meta.get("x")
        y = meta.get("y")
        if not isinstance(x, (int, float)):
            x = getattr(event, "location_x", None)
        if not isinstance(y, (int, float)):
            y = getattr(event, "location_y", None)
        location = (x, y) if (isinstance(x, (int, float)) and
                              isinstance(y, (int, float))) else None
        severity = _TAG_SEVERITY.get(tag, 1.0)
        self.memory.store(tag, location=location, severity=severity)
        self.memory.decay(dt)
        # Memory → temperament immediately: a scar changes behaviour now.
        self.experience.apply_memory(self.memory)

    # ── TEMPERAMENT → SAMPLING ──────────────────────────────
    def temper(
        self,
        probs: Any,
        labels: List[str],
        under_pressure: bool = False,
        fatigue: float = 0.0,
    ):
        """Distort the neural distribution before sampling.

        * confidence  → peaks (<1.0 flat — indecisive) or sharpens
                        (>1.0 firm) the distribution.
        * risk_in_box → a fearful player who has been burned in the box
                        backs away from the high-risk intents and leans
                        into the safe ones.

        Returns a probability vector of the same shape, renormalised.
        """
        import numpy as np

        b = self.experience.bias
        risk = b["risk_in_box"]
        conf = b["confidence"]
        if abs(risk - 1.0) < 1e-9 and abs(conf - 1.0) < 1e-9:
            return probs

        p = np.asarray(probs, dtype=np.float64)
        alpha = min(2.0, max(0.3, conf))

        # Fear dampens the high-risk intents toward safety.
        risk_intents = {"THROUGH_BALL", "DRIBBLE", "SWITCH", "CROSS", "SHOOT"}
        safe_intents = {"SAFE_PASS", "PROTECT_POSSESSION", "RECYCLE"}

        mults = []
        for label in labels:
            m = 1.0
            if label in risk_intents:
                m = 0.5 + 0.5 * risk
            elif label in safe_intents:
                m = 1.0 + 0.5 * (1.0 - risk)
            mults.append(m)
        mults = np.asarray(mults, dtype=np.float64)

        p = (np.power(p + 1e-12, alpha)) * mults
        total = p.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray(probs, dtype=np.float64)
        return p / total


# ── PERSISTENCE ACROSS MATCHDAYS ─────────────────────
    def to_state(self) -> Dict[str, Any]:
        """Serialise memory weights + temperament + perception config.

        Locations are in-match only and are dropped; what survives between
        matchdays is the emotional weight of each scar (which already fell
        to this level under the in-match 0.999/s decay) and the temperament
        it produced.
        """
        return {
            "config": {
                "view_radius": self.senses.view_radius,
                "max_teammates": self.senses.max_teammates,
                "max_opponents": self.senses.max_opponents,
                "engaged": self.engaged,
            },
            "memory_tags": self.memory.recall_all(),
            "temperament": dict(self.experience.bias),
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "PlayerMind":
        """Rebuild a mind from a persisted SeasonState cognition record."""
        cfg = data.get("config", {})
        mind = cls(
            view_radius=cfg.get("view_radius", 25.0),
            max_teammates=cfg.get("max_teammates", 2),
            max_opponents=cfg.get("max_opponents", 2),
        )
        mind.engaged = bool(cfg.get("engaged", True))
        tags = data.get("memory_tags", {}) or {}
        for tag, weight in tags.items():
            if isinstance(weight, (int, float)) and weight > 0:
                mind.memory.events.append(
                    MemoryEvent(tag, None, float(weight))
                )
        temperament = data.get("temperament")
        if isinstance(temperament, dict):
            for key in mind.experience.bias:
                val = temperament.get(key)
                if isinstance(val, (int, float)):
                    mind.experience.bias[key] = float(val)
        return mind


# ─────────────────────────────────────────────────────────────
# MIND REGISTRY (per player name, mirroring the brain registry)
# ─────────────────────────────────────────────────────────────

_minds: Dict[str, PlayerMind] = {}


def register_mind(player_name: str, mind: PlayerMind) -> None:
    """Bind a PlayerMind to a player name."""
    _minds[player_name] = mind


def get_mind(player_name: str) -> Optional[PlayerMind]:
    """Look up a player's mind, or None if not registered."""
    return _minds.get(player_name)


def clear_minds() -> None:
    """Remove all registered minds (for testing)."""
    _minds.clear()


def new_mind(**kwargs) -> PlayerMind:
    """Convenience factory for a fresh, engaged mind."""
    return PlayerMind(**kwargs)