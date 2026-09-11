"""
PLOFA 26/27 — TACTICAL SHAPES MODULE
====================================
tactical_shapes.py

Feature #1: in-match FORMATION / SHAPE shifts.

tactical_ai.py already moves the numeric dials (press intensity, tempo,
directness, defensive line, shot volume) as a manager reacts to the
scoreline. What it explicitly does NOT do (see its wiring guide) is change
WHERE each role rests — the actual 11 bodies on the pitch. This module closes
that gap: it turns a game state into a *formation stance* (baseline / pushing
/ all-out chase / protect-lead / see-it-out / tense-level) and expresses each
stance as a set of per-role home-position DELTAS that the PositionEngine
applies additively on top of the formation-computed home anchor.

Real football mirrors:
  - All-out chase (two goals down, late): fullbacks pushed up to the final
    third, midfield step-on, wingers high and wide, CAM up beside the striker
    — the classic 4-2-4 / 3-4-3.
  - See-it-out (two goals up, late): wingers drop to wide midfield, a
    compact 5-4-1-ish block forms, the lone striker stays isolated as the out
    ball.
  - Man down: the whole block tucks toward the centre and sags — never a
    stretched shape with a hole.
  - Fatigue: a subtle collective sag as legs go.

PURE-DATA MODULE. It imports NOTHING from the engine at module load (all
engine imports are lazy, inside functions) so position_engine, tactical_ai
and match_engine can all import it without creating an import cycle.

Coordinate convention — ATTACKING-RIGHT normalised space, matching
position_engine.BASE_HOME_POSITIONS:
    +x  = toward the opponent goal (10.5 m/grid unit)
    low y  = own-left flank channel (LW / LB)
    high y = own-right flank channel (RW / RB)
The PositionEngine mirrors both delta signs for teams that attack left.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

# Roles that carry spatial deltas. Anything not listed keeps its anchor.
ALL_ROLES = ("GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF")


class FormationStance(Enum):
    """The shape a team takes as a reaction to the current game state."""
    BASELINE = "baseline"          # authored shape, no in-match adjustment
    PUSHING = "pushing"            # one goal down, chasing late
    ALL_OUT_CHASE = "all_out_chase"  # two-plus goals down, late: men forward
    PROTECT_LEAD = "protect_lead"  # one goal up, late: cautious block
    SEE_IT_OUT = "see_it_out"      # two-plus goals up, late: see the game out
    TENSE_LEVEL = "tense_level"    # level, late: both sides push for one goal


# ─────────────────────────────────────────────
# PER-ROLE HOME DELTAS (dx forward m, dy flank m)
# ─────────────────────────────────────────────

# dy sign convention: positive = toward the team's RIGHT channel (high y),
# negative = toward the team's LEFT channel (low y). For wide roles (LW/LB
# rest LOW) "stay wide" therefore means negative dy; for RW/RB "stay wide"
# means positive dy. Each table lists every role for readability.

STANCE_ROLE_DELTAS: Dict[FormationStance, Dict[str, Tuple[float, float]]] = {
    FormationStance.BASELINE: {r: (0.0, 0.0) for r in ALL_ROLES},

    # One goal down late — a controlled push: wingers stay high & wide,
    # fullbacks climb into the middle third, midfield steps up.
    FormationStance.PUSHING: {
        "GK":  (0.0,  0.0),
        "CB":  (1.5,  0.0),
        "LB":  (10.0,  0.5),
        "RB":  (10.0, -0.5),
        "CDM": (4.0,  0.0),
        "CM":  (5.0,  0.0),
        "CAM": (4.0,  0.0),
        "LW":  (7.0, -0.5),
        "RW":  (7.0,  0.5),
        "ST":  (2.0,  0.0),
        "CF":  (2.0,  0.0),
    },

    # Two-plus goals down late — the all-out chase shape: fullbacks in the
    # final third, CAM beside the striker, CM steps into attack, wingers hug
    # the touchline and flood the box.
    FormationStance.ALL_OUT_CHASE: {
        "GK":  (1.0,  0.0),
        "CB":  (3.5,  0.0),
        "LB":  (17.0, 1.0),
        "RB":  (17.0, -1.0),
        "CDM": (6.0,  0.0),
        "CM":  (9.0,  1.5),
        "CAM": (7.0,  0.0),
        "LW":  (10.0, -1.0),
        "RW":  (10.0,  1.0),
        "ST":  (5.0,  0.0),
        "CF":  (5.0,  0.0),
    },

    # One goal up, late — the block drops, wingers come deeper to the wide
    # midfield slots, the shape narrows slightly.
    FormationStance.PROTECT_LEAD: {
        "GK":  (0.0,  0.0),
        "CB":  (-3.0, 0.0),
        "LB":  (-7.0, 0.0),
        "RB":  (-7.0, 0.0),
        "CDM": (-6.0, 0.0),
        "CM":  (-8.0, 0.0),
        "CAM": (-9.0, 0.0),
        "LW":  (-14.0,  1.5),
        "RW":  (-14.0, -1.5),
        "ST":  (-5.0, 0.0),
        "CF":  (-6.0, 0.0),
    },

    # Two-plus goals up, late — see the game out: wingers drop to full-back-
    # adjacent wide-mid roles, everything sags into a compact deep block and
    # the lone striker stays isolated as the out ball (net ST shift still
    # backwards, but noticeably less than the midfield's).
    FormationStance.SEE_IT_OUT: {
        "GK":  (0.0,  0.0),
        "CB":  (-5.0, 0.0),
        "LB":  (-10.0, 0.0),
        "RB":  (-10.0, 0.0),
        "CDM": (-8.0, 0.0),
        "CM":  (-11.0, 0.0),
        "CAM": (-12.0, 0.0),
        "LW":  (-20.0,  2.5),
        "RW":  (-20.0, -2.5),
        "ST":  (-8.0, 0.0),
        "CF":  (-9.0, 0.0),
    },

    # Level and late — both sides know one goal wins it: a mild push with
    # slightly advanced wingers and fullbacks.
    FormationStance.TENSE_LEVEL: {
        "GK":  (0.0,  0.0),
        "CB":  (1.0,  0.0),
        "LB":  (6.0,  0.5),
        "RB":  (6.0, -0.5),
        "CDM": (2.0,  0.0),
        "CM":  (3.0,  0.0),
        "CAM": (3.0,  0.0),
        "LW":  (4.0, -0.5),
        "RW":  (4.0,  0.5),
        "ST":  (2.0,  0.0),
        "CF":  (2.0,  0.0),
    },
}

# ─────────────────────────────────────────────
# PER-ROLE MODIFIER DELTAS (stack additively on the stance)
# ─────────────────────────────────────────────

# A short-handed team NEVER stretches: wide men tuck into the half-spaces,
# the whole block sags a few metres and the striker stays higher than a
# midfielder (the counter outlet must exist or the ball never leaves).
MAN_DOWN_ROLE_DELTAS: Dict[str, Tuple[float, float]] = {
    "GK":  (0.0,  0.0),
    "CB":  (-2.0,  1.0),
    "LB":  (-5.0,  8.0),
    "RB":  (-5.0, -8.0),
    "CDM": (-6.0,  1.5),
    "CM":  (-8.0,  2.0),
    "CAM": (-9.0,  2.0),
    "LW":  (-9.0,  8.0),
    "RW":  (-9.0, -8.0),
    "ST":  (-3.0,  0.0),
    "CF":  (-4.0,  1.0),
}

# Legs go, shape sags: a subtle collective retreat, no dramatic reshuffle.
FATIGUED_ROLE_DELTAS: Dict[str, Tuple[float, float]] = {
    "GK":  (0.0, 0.0),
    "CB":  (-1.0, 0.0),
    "LB":  (-3.0, 0.0),
    "RB":  (-3.0, 0.0),
    "CDM": (-2.0, 0.0),
    "CM":  (-3.0, 0.0),
    "CAM": (-3.0, 0.0),
    "LW":  (-3.0, 0.0),
    "RW":  (-3.0, 0.0),
    "ST":  (-2.0, 0.0),
    "CF":  (-2.0, 0.0),
}


def stance_delta_roles(
    stance: FormationStance,
    own_red_cards: int = 0,
    avg_stamina: float = 100.0,
) -> Dict[str, Tuple[float, float]]:
    """Composite per-role home deltas for a stance (+ man-down / fatigue).

    Returns a dict keyed by role label; empty values are omitted so callers
    can iterate directly. Stays a pure function of its inputs so the match
    engine gets a stable, idempotent shape per (stance, cards, stamina)."""
    out: Dict[str, Tuple[float, float]] = {}
    base = STANCE_ROLE_DELTAS.get(stance, STANCE_ROLE_DELTAS[FormationStance.BASELINE])
    for role in ALL_ROLES:
        dx, dy = base.get(role, (0.0, 0.0))
        if own_red_cards > 0:
            mdx, mdy = MAN_DOWN_ROLE_DELTAS.get(role, (0.0, 0.0))
            dx += mdx
            dy += mdy
        if avg_stamina < 75.0:
            fdx, fdy = FATIGUED_ROLE_DELTAS.get(role, (0.0, 0.0))
            dx += fdx
            dy += fdy
        if dx != 0.0 or dy != 0.0:
            out[role] = (round(dx, 1), round(dy, 1))
    return out


def formation_stance_for(
    profile,
    state,
    team_name: str,
    home_team: str,
    manager=None,
    minute: Optional[int] = None,
) -> FormationStance:
    """Pick the formation stance for a team from the live match state.

    Mirrors the posture thresholds of tactical_ai.TacticalAI.adjust() (the
    same scoreline/clock/manger lenses) so the SHAPE and the DIALS always
    agree — a manager who chases harder also throws men forward, a manager
    who protects also locks the block.

    `profile` duck-types TeamProfile (only .style is read); `state`
    duck-types MatchState (.goal_difference, .minute).
    """
    gd = state.goal_difference if team_name == home_team else -state.goal_difference
    minute = state.minute if minute is None else minute

    # Manager bias layer — identical lenses to tactical_ai.py (aggressive
    # managers chase earlier / protect later).
    chase_min, push_min, protect_min, lead_min = 60, 70, 70, 80
    if manager is not None:
        chase_min = max(45, 60 - manager.chase_shift())
        push_min = max(50, 70 - manager.chase_shift() * 0.7)
        protect_min = max(50, 70 - manager.protect_shift())
        lead_min = max(60, 80 - manager.protect_shift())

    # Stance selection — same decision order as TacticalAI.adjust().
    if gd <= -2 and minute >= chase_min:
        return FormationStance.ALL_OUT_CHASE
    if gd == -1 and minute >= push_min:
        return FormationStance.PUSHING
    if gd >= 2 and minute >= protect_min:
        return FormationStance.SEE_IT_OUT
    if gd == 1 and minute >= lead_min:
        return FormationStance.PROTECT_LEAD
    if gd == 0 and minute >= 80:
        return FormationStance.TENSE_LEVEL

    # The very first exchanges are a feel-out period handled by the dials;
    # the shape stays authored.
    return FormationStance.BASELINE