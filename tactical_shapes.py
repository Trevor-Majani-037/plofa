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


# ─────────────────────────────────────────────────────────────
# DEFENSIVE STYLE TEMPERAMENT (2026-10-16)
#
# `formation_stance_for` takes a `profile` and its docstring claimed
# "only .style is read". The body reads `state.goal_difference` and
# `state.minute` and NEVER `.style`. Consequence: the ATTACKING half of the
# shape was style-driven all along (`pattern_for` picks from
# `_STYLE_PATTERN_POOLS`) and the DEFENSING half was not, so a park-the-bus
# side and a gegenpressing side held an identical back line until the
# scoreline happened to change. Measured: two frames (BALANCED/MIXED vs
# FLUID_COUNTER/COUNTER) produced per-role lines within 0.3 m of each other.
#
# This is the missing half. It is a persistent TEMPERAMENT added on top of
# whatever stance the game state selects, NOT a replacement for it — so a
# manager who goes 2-0 down and chases still throws men forward; a
# park-the-bus side simply starts every situation from deeper.
#
# MAGNITUDE, chosen against the tables above: the game-state stance deltas run
# -20 m (SEE_IT_OUT, LW) to +17 m (ALL_OUT_CHASE, LB). This table spans
# -10 to +5, deliberately UNDER it. A style bias able to outvote SEE_IT_OUT
# would make a side abandon its identity to protect a lead, which is the
# opposite of what protecting a lead is for.
#
# SHAPE, borrowed from SEE_IT_OUT rather than a flat drop: the back line and
# midfield sink while the STRIKER holds. A block whose front pair drop with
# the defenders has no outlet, which is exactly why the existing SEE_IT_OUT
# moves ST only -8 while moving LW -20.
#
# dy sign convention follows the tables above: for wide roles resting LOW,
# positive dy moves INWARD (a narrowing block), negative dy moves OUTWARD
# (a pressing side staying wide). PUSHING is the worked example -- LW
# (7.0, -0.5) -- and SEE_IT_OUT the other -- LW (-20.0, 2.5).
_STYLE_DEFENSE_ARCHETYPE: Dict[str, str] = {
    "park_the_bus": "sunk", "ultra_defensive": "sunk",
    "defensive": "low", "fluid_counter": "low", "route_one": "low",
    "balanced": "neutral", "wing_play": "neutral",
    "tiki_taka": "high", "structured_possession": "high",
    "vertical_tiki_taka": "high", "gegenpressing": "high",
    "attacking": "high", "ultra_attacking": "high",
}

_STYLE_DEFENSE_ROLE_DELTAS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # A side that will not leave its own third. Deepest here, and the widest
    # lateral move, because dropping without narrowing leaves the half-spaces
    # open -- which is the exact failure `defensive_block_target`'s own
    # docstring records ("49.6 m wide on a 68 m pitch ... both half-spaces
    # open").
    "sunk": {
        "GK": (0.0, 0.0), "CB": (-6.0, 0.0),
        "LB": (-8.0, 3.0), "RB": (-8.0, -3.0),
        "CDM": (-6.0, 0.0), "CM": (-8.0, 0.0), "CAM": (-6.0, 0.0),
        "LW": (-10.0, 4.0), "RW": (-10.0, -4.0),
        "ST": (-3.0, 0.0), "CF": (-4.0, 0.0),
    },
    # A normal defensive side: sits a few metres off its attacking post.
    "low": {
        "GK": (0.0, 0.0), "CB": (-3.0, 0.0),
        "LB": (-4.0, 1.5), "RB": (-4.0, -1.5),
        "CDM": (-3.0, 0.0), "CM": (-4.0, 0.0), "CAM": (-3.0, 0.0),
        "LW": (-5.0, 2.0), "RW": (-5.0, -2.0),
        "ST": (-2.0, 0.0), "CF": (-2.0, 0.0),
    },
    "neutral": {},
    # Possession and pressing sides hold a higher line. The midfielder push is
    # the largest term: the front of the press IS the midfield line, so a
    # gegenpressing side that defends deep has not pressed at all.
    "high": {
        "GK": (0.0, 0.0), "CB": (3.0, 0.0),
        "LB": (4.0, -1.0), "RB": (4.0, 1.0),
        "CDM": (4.0, 0.0), "CM": (5.0, 0.0), "CAM": (3.0, 0.0),
        "LW": (5.0, -2.0), "RW": (5.0, 2.0),
        "ST": (2.0, 0.0), "CF": (2.0, 0.0),
    },
}


def style_defense_deltas(style: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """The defensive-shape temperament for a TeamStyle value.

    Pure lookup, no defaults invented: an unrecognised or absent style gets
    zero deltas, i.e. exactly the pre-2026-10-16 behaviour. That matters
    because a probe that reaches this with a None style and gets a made-up
    "average" would be measuring its own assumption.
    """
    if not style:
        return {}
    arch = _STYLE_DEFENSE_ARCHETYPE.get(style, "neutral")
    return _STYLE_DEFENSE_ROLE_DELTAS.get(arch, {})


def stance_delta_roles(
    stance: FormationStance,
    own_red_cards: int = 0,
    avg_stamina: float = 100.0,
    style: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Composite per-role home deltas for a stance (+ man-down / fatigue / style).

    Returns a dict keyed by role label; empty values are omitted so callers
    can iterate directly. Stays a pure function of its inputs so the match
    engine gets a stable, idempotent shape per (stance, cards, stamina, style).

    Stack order is deliberate: style is the floor (who this team IS), then
    the game-state stance (what the situation demands), then man-down and
    fatigue (what the match has taken out of them). All additive, so a
    park-the-bus side two goals up sits deeper than a balanced side two
    goals up -- while both still execute SEE_IT_OUT.
    """
    out: Dict[str, Tuple[float, float]] = {}
    base = STANCE_ROLE_DELTAS.get(stance, STANCE_ROLE_DELTAS[FormationStance.BASELINE])
    style_base = style_defense_deltas(style)
    for role in ALL_ROLES:
        dx, dy = base.get(role, (0.0, 0.0))
        sdx, sdy = style_base.get(role, (0.0, 0.0))
        dx += sdx
        dy += sdy
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

    `profile` is accepted for interface symmetry with its siblings and is
    passed by the caller, but this function does NOT read it. It used to have
    a docstring asserting that it read the team's style, which was false and
    cost a real capability: the stance is chosen from the SCORELINE and the
    CLOCK alone, so every team defended identically until minute 60. The
    style's influence on the defensive shape now lives in
    `style_defense_deltas`, which the caller passes to `stance_delta_roles`
    alongside the stance -- stacked under it, not competing with it. That
    split is deliberate: a manager chasing late must be able to throw men
    forward regardless of which club he manages.
    """
    gd = state.goal_difference if team_name == home_team else -state.goal_difference
    minute = state.minute if minute is None else minute

    # Manager bias layer — identical lenses to tactical_ai.py (aggressive
    # managers chase earlier / protect later). Applied ONLY to managers that
    # expose the Phase 0–5 surface (chase_shift/protect_shift). A Phase 6
    # brain Manager (manager_profile.Manager) has no such shifts — its
    # posture is decided by the neural pipeline, so the threshold lenses
    # are skipped for it (no crash on a missing attribute) and its say
    # comes AFTER stance selection as a one-step nudge (Phase 8).
    chase_min, push_min, protect_min, lead_min = 60, 70, 70, 80
    if manager is not None and hasattr(manager, "chase_shift"):
        chase_min = max(45, 60 - manager.chase_shift())
        push_min = max(50, 70 - manager.chase_shift() * 0.7)
        protect_min = max(50, 70 - manager.protect_shift())
        lead_min = max(60, 80 - manager.protect_shift())

    # Stance selection — same decision order as TacticalAI.adjust().
    if gd <= -2 and minute >= chase_min:
        stance = FormationStance.ALL_OUT_CHASE
    elif gd == -1 and minute >= push_min:
        stance = FormationStance.PUSHING
    elif gd >= 2 and minute >= protect_min:
        stance = FormationStance.SEE_IT_OUT
    elif gd == 1 and minute >= lead_min:
        stance = FormationStance.PROTECT_LEAD
    elif gd == 0 and minute >= 80:
        stance = FormationStance.TENSE_LEVEL
    else:
        # The very first exchanges are a feel-out period handled by the
        # dials; the shape stays authored.
        return FormationStance.BASELINE

    # ── BRAIN-MANAGER SAY (Phase 8, opt-in) ──────────────────────
    # A live Manager's CURRENT posture nudges the shape one step along
    # the attack order (dwell-guarded upstream, so no flicker):
    #   ATTACK  → one step more attacking (chase harder, throw men forward)
    #   DEFEND  → one step more defensive (lock the block earlier)
    #   BALANCED / no live manager / flag OFF → scoreline shape stands.
    # Static profiles (no .decide) never enter this branch.
    if manager is not None and hasattr(manager, "decide"):
        import match_engine as _me
        if bool(getattr(_me, "USE_MANAGER_BRAIN", False)):
            posture = getattr(manager, "_current_posture", "BALANCED")
            if posture in ("ATTACK", "DEFEND"):
                order = (
                    FormationStance.ALL_OUT_CHASE,
                    FormationStance.PUSHING,
                    FormationStance.TENSE_LEVEL,
                    FormationStance.BASELINE,
                    FormationStance.PROTECT_LEAD,
                    FormationStance.SEE_IT_OUT,
                )
                idx = order.index(stance)
                if posture == "ATTACK":
                    stance = order[max(0, idx - 1)]
                else:
                    stance = order[min(len(order) - 1, idx + 1)]
    return stance