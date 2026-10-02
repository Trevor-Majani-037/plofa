"""
PLOFA 26/27 — COACH-TO-PLAYER INSTRUCTIONS (Phase 8 v1)
======================================================
Gives the live brain-manager a say at the INDIVIDUAL level — "shoot less",
"stay wide", "target their left" — without retraining any neural net.

Design (same bias-layer discipline as every other manager lever):
  * Instructions are DERIVED from the manager's CURRENT stored posture
    (no decide() poll, no dwell disturbance, no RNG — pure read).
  * They BIAS, never command: intent bonuses ride log-space on the
    carrier's forward-pass distribution pre-sample (the brain stays the
    sole chooser); flank focus adds a small bonus in target selection;
    width is a small anchor delta beside stance/pattern deltas.
  * BALANCED posture, static managers, and flag OFF all yield None —
    byte-identical to the pre-instruction engine.
  * Maverick tax: low-decisions/composure players scale the intent bias
    down (down to 30%) — some players simply don't listen well.

v1 instruction set (posture-derived, team-wide, role-scoped at apply):
  ATTACK → SHOOT+/CROSS+/THROUGH_BALL+/DRIBBLE+, STAY_WIDE, focus flank
            = the team's current attack-pattern flank (its overload side).
  DEFEND → SHOOT-/CROSS-, SAFE+/RECYCLE+/PROTECT+, TUCK_IN, no focus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CoachInstructions:
    """One team's current coach instruction (all fields deterministic)."""
    intent_bias: Dict[str, float] = field(default_factory=dict)
    favored_flank: Optional[str] = None   # "L" / "R" / None (normalised)
    width_cmd: float = 0.0                # +1 stay wide, -1 tuck in, 0 none
    source: str = "BALANCED"


# Logit bonuses on the carrier's intent distribution (small vs the
# learned net; SHOOT_MORE is additionally capped downstream by the
# AttackingMatrix shot gate, SHOOT_LESS always bites).
_ATTACK_BIAS: Dict[str, float] = {
    "SHOOT": 0.25, "CROSS": 0.20, "THROUGH_BALL": 0.15, "DRIBBLE": 0.10,
}
_DEFEND_BIAS: Dict[str, float] = {
    "SHOOT": -0.30, "CROSS": -0.15, "SAFE_PASS": 0.15,
    "RECYCLE": 0.15, "PROTECT_POSSESSION": 0.20,
}


def instructions_for_manager(manager: Any,
                             team_pattern: Any = None) -> Optional[CoachInstructions]:
    """Derive the live coach instruction for one team's manager.

    Pure read of ``manager._current_posture`` — never polls decide().
    Returns None unless the brain path is on AND a live (decide-capable)
    manager is wired AND its posture is ATTACK/DEFEND.
    """
    if manager is None or not hasattr(manager, "decide"):
        return None
    import match_engine as _me
    if not bool(getattr(_me, "USE_MANAGER_BRAIN", False)):
        return None
    posture = getattr(manager, "_current_posture", "BALANCED")
    if posture == "ATTACK":
        flank: Optional[str] = None
        if team_pattern is not None:
            try:
                from attack_patterns import favored_flank as _ff
                flank = _ff(team_pattern)
            except Exception:
                flank = None
        return CoachInstructions(intent_bias=dict(_ATTACK_BIAS),
                                 favored_flank=flank,
                                 width_cmd=1.0, source="ATTACK")
    if posture == "DEFEND":
        return CoachInstructions(intent_bias=dict(_DEFEND_BIAS),
                                 favored_flank=None,
                                 width_cmd=-1.0, source="DEFEND")
    return None


def listener_scale(player: Any) -> float:
    """Maverick tax: 0.3 (ignores the coach) .. 1.0 (model professional).

    From the carrier's own decisions + composure — the same mental
    attributes that already govern his noise and temperature.
    """
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)
    try:
        dec = float(getattr(mental, "decisions", 55.0)) / 100.0
        com = float(getattr(mental, "composure", 55.0)) / 100.0
    except Exception:
        return 1.0
    idx = dec * 0.5 + com * 0.5
    return max(0.3, min(1.0, 0.3 + 0.7 * idx))
