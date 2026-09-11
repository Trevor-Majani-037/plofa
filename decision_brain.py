"""Bounded-rationality on-ball intent selection for PLOFA.

This module answers exactly one question for the player currently on the
ball: "What would THIS player -- with THIS vision, composure, decision-
making, tendencies and Soul, fatigued and pressured to THIS degree, in
THIS scoreline -- probably perceive and choose to do right now?"

It never answers "what is the objectively best action here?" and it never
decides whether the chosen action succeeds. Existing systems keep doing
what they already do:

    * AttackingMatrix / TacticalPhase / wide-combo overrides remain the
      authoritative, calibrated deterministic layer for shot-taking and
      structural resets. This brain only runs for touches those systems
      have left open, and never fires a shot itself (see integration
      notes in event_chain.py at the call site).
    * PositionEngine, pitch_control, the possession-physics episode
      (resolve_pass / resolve_dribble / resolve_long_pass) and the
      defensive systems remain solely responsible for EXECUTION and
      OUTCOME. This module never rolls a pass-completion or tackle dice
      -- it only proposes an intent label and (optionally) a preferred
      target, which the existing execution code consumes exactly like it
      already consumed ActivePlayBrain's CARRY/PASS binary.

Design contract
----------------
1. Ten candidate intents (PlayerIntent): PROGRESSIVE_PASS, SAFE_PASS,
   THROUGH_BALL, SWITCH, CARRY, DRIBBLE, CROSS, SHOOT, RECYCLE,
   PROTECT_POSSESSION.
2. Every candidate gets an OBJECTIVE value (cheap geometric/DNA estimate
   of how good the option really is, given real space/marking/distance
   data from PositionEngine) and a PERCEIVED value (the objective value
   distorted by player-specific noise and bias). Selection samples over
   PERCEIVED values of only the candidates a player actually notices --
   never argmax, never over the objective values directly.
3. Perception is bounded: a candidate whose objective value sits below a
   player-specific visibility floor (driven by vision/decisions/
   anticipation) is dropped before selection -- the player never
   considered it. This is what makes "fails to recognize an option
   entirely" possible, not just "recognizes it but rates it wrong."
4. decision_quality and evaluation_error describe the DECISION only.
   Nothing here touches success probability -- that stays downstream.
5. Everything is derived from data already produced elsewhere in the
   engine (PositionEngine geometry, PlayerDNA, PlayerSoul, TeamProfile
   style, MatchState/game_state) -- no parallel match engine, no shadow
   geometry system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# INTENTS
# ─────────────────────────────────────────────────────────────

class PlayerIntent(Enum):
    PROGRESSIVE_PASS   = "PROGRESSIVE_PASS"
    SAFE_PASS          = "SAFE_PASS"
    THROUGH_BALL       = "THROUGH_BALL"
    SWITCH             = "SWITCH"
    CARRY              = "CARRY"
    DRIBBLE            = "DRIBBLE"
    CROSS              = "CROSS"
    SHOOT              = "SHOOT"
    RECYCLE            = "RECYCLE"
    PROTECT_POSSESSION = "PROTECT_POSSESSION"


# Intents that resolve, downstream, as a "carry the ball" touch rather
# than a release of possession. Kept in one place so the event_chain
# integration and any future consumer agree on the mapping.
CARRY_LIKE_INTENTS = frozenset({PlayerIntent.CARRY, PlayerIntent.DRIBBLE})


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _get(obj: Any, path: str, default: float) -> float:
    """Dotted-path getattr with a default, so a missing DNA sub-object
    never raises mid-match."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return float(cur) if cur is not None else default


# ─────────────────────────────────────────────────────────────
# EXPLAINABILITY RECORDS
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionCandidate:
    """A single option as it objectively exists on the pitch."""
    intent: PlayerIntent
    objective_value: float          # 0..~1.3, cheap EV-style estimate
    risk: float                     # 0..1, how much can go wrong
    target: Any = None              # teammate PlayerProfile, if relevant
    note: str = ""                  # human-readable reason for the value


@dataclass(frozen=True)
class PerceivedCandidate:
    """The same option as distorted through one player's eyes."""
    candidate: ActionCandidate
    perceived_value: float
    noise: float
    bias: float


@dataclass(frozen=True)
class PlayerDecision:
    """Result of DecisionBrain.decide().

    `action` stays a legacy "CARRY"/"PASS" label so the existing
    event_chain call site (which only ever checked
    `active_decision.action == "CARRY"`) keeps working unmodified.
    `intent` carries the full ten-way choice for any caller that wants
    to use it.
    """
    intent: PlayerIntent
    action: str                     # "CARRY" | "PASS" (legacy compat)
    confidence: float                # 0..1, how peaked the belief was
    reason: str
    risk_level: float                # 0..1, risk of the CHOSEN option
    decision_quality: float          # 0..1, rank-based, vs objective truth
    evaluation_error: float          # |perceived - objective| for chosen
    is_error: bool                   # chosen option was not top-2 objectively
    target: Any = None               # preferred teammate, if any
    perceived_intents: frozenset = field(default_factory=frozenset)
    trace: Optional[Dict[str, Any]] = None   # full candidate dump, opt-in


# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# (deliberately self-contained rather than imported from event_chain's
#  BaseChain, since event_chain imports THIS module -- importing back
#  would be circular. These are intentionally cheap approximations for
#  PERCEPTION/PREFERENCE only; the real geometry authority for EXECUTION
#  remains PositionEngine + the possession-physics episode.)
# ─────────────────────────────────────────────────────────────

def _pos(position_engine, name: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if position_engine is None:
        return fallback
    try:
        return position_engine.get_position(name)
    except Exception:
        return fallback


def _nearest_defender_dist(x: float, y: float, defenders: List[Any],
                            position_engine: Any) -> Optional[float]:
    best = None
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = _pos(position_engine, d.name, (x + 10, y))
        dist = math.hypot(dx - x, dy - y)
        if best is None or dist < best:
            best = dist
    return best


def _defenders_within(x: float, y: float, defenders: List[Any],
                       position_engine: Any, radius: float) -> int:
    count = 0
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = _pos(position_engine, d.name, (x + 999, y))
        if math.hypot(dx - x, dy - y) <= radius:
            count += 1
    return count


def _teammate_openness(tx: float, ty: float, defenders: List[Any],
                        position_engine: Any) -> float:
    """0 (smothered) .. 1 (wide open), from nearest defender to a point."""
    d = _nearest_defender_dist(tx, ty, defenders, position_engine)
    if d is None:
        return 1.0
    return _clamp((d - 1.5) / 8.5)


def _best_forward_teammate(
    x: float, y: float, teammates: List[Any], defenders: List[Any],
    position_engine: Any, attacks_right: bool,
) -> Optional[Tuple[Any, float, float, float]]:
    """Best (teammate, tx, ty, combined_value) among teammates who are
    meaningfully further forward than the ball, weighted by their space."""
    best = None
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = _pos(position_engine, t.name, (x, y))
        progress = (tx - x) if attacks_right else (x - tx)
        if progress < 4.0:
            continue
        openness = _teammate_openness(tx, ty, defenders, position_engine)
        value = _clamp(progress / 35.0) * 0.55 + openness * 0.45
        if best is None or value > best[3]:
            best = (t, tx, ty, value)
    return best


def _wide_switch_teammate(
    x: float, y: float, teammates: List[Any], defenders: List[Any],
    position_engine: Any,
) -> Optional[Tuple[Any, float, float, float]]:
    """Best far-side teammate for a switch of play: opposite half of the
    pitch width, not much further back than the ball."""
    best = None
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = _pos(position_engine, t.name, (x, y))
        width_gap = abs(ty - y)
        if width_gap < 22.0:
            continue
        openness = _teammate_openness(tx, ty, defenders, position_engine)
        value = _clamp(width_gap / 55.0) * 0.5 + openness * 0.5
        if best is None or value > best[3]:
            best = (t, tx, ty, value)
    return best


def _nearest_safe_teammate(
    x: float, y: float, teammates: List[Any], defenders: List[Any],
    position_engine: Any,
) -> Optional[Tuple[Any, float, float, float]]:
    """Closest, low-risk short option -- the 'always available' out
    ball. Real football always has one of these; the objective value is
    intentionally never zero."""
    best = None
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = _pos(position_engine, t.name, (x, y))
        dist = math.hypot(tx - x, ty - y)
        if dist > 22.0 or dist < 1.0:
            continue
        openness = _teammate_openness(tx, ty, defenders, position_engine)
        value = _clamp(1.0 - dist / 22.0) * 0.5 + openness * 0.5
        if best is None or value > best[3]:
            best = (t, tx, ty, value)
    return best


def _deepest_teammate(
    x: float, y: float, teammates: List[Any], position_engine: Any,
    attacks_right: bool,
) -> Optional[Tuple[Any, float, float]]:
    best = None
    for t in teammates or []:
        tx, ty = _pos(position_engine, t.name, (x, y))
        depth = (x - tx) if attacks_right else (tx - x)
        if best is None or depth > best[2]:
            best = (t, tx, depth)
    if best is None:
        return None
    t, tx, _ = best
    ty = _pos(position_engine, t.name, (x, y))[1]
    return (t, tx, ty)


def _goal_distance(x: float, y: float, attacks_right: bool) -> float:
    gx = 105.0 if attacks_right else 0.0
    return math.hypot(gx - x, 34.0 - y)


def _fatigue_estimate(player: Any, minute: float) -> float:
    """Approximate 0 (fresh) .. 1 (exhausted) in-match fatigue.

    NOTE (honest limitation): PossessionChain does not currently receive
    the match_engine's live per-player stamina (sub_controller.stamina),
    only the static DNA stamina attribute and the current minute. This
    is a deliberately conservative proxy, not a claim that live stamina
    is wired through. Threading `sub_controller.stamina` into
    PossessionChain.generate()/ChainDispatcher.possession() is a
    follow-up (see integration notes) that would let this call use the
    real number instead of the estimate.
    """
    stamina_attr = _get(getattr(player, "dna", None), "physical.stamina", 65.0) / 100.0
    match_frac = _clamp(minute / 90.0, 0.0, 1.15)
    live = match_frac * (1.35 - stamina_attr)
    carry_over = _get(getattr(player, "dna", None), "form.fatigue_level", 0.0) / 100.0
    return _clamp(live * 0.8 + carry_over * 0.25)


# ─────────────────────────────────────────────────────────────
# CANDIDATE GENERATION (OBJECTIVE VALUES)
# ─────────────────────────────────────────────────────────────

def _generate_candidates(
    player: Any, x: float, y: float, teammates: List[Any], defenders: List[Any],
    position_engine: Any, team_profile: Any, under_pressure: bool,
    attacks_right: bool, game_state: Any, minute: float,
) -> List[ActionCandidate]:
    dna = getattr(player, "dna", None)
    technical = getattr(dna, "technical", None)
    mental = getattr(dna, "mental", None)
    physical = getattr(dna, "physical", None)
    passing = getattr(dna, "passing", None)
    tendencies = getattr(dna, "tendencies", None)
    position = getattr(player, "position", "")

    near_def = _nearest_defender_dist(x, y, defenders, position_engine)
    space = 1.0 if near_def is None else _clamp((near_def - 1.5) / 8.5)
    tight_marked = near_def is not None and near_def < 3.0

    final_third = x > 70.0 if attacks_right else x < 35.0
    own_half = x < 52.5 if attacks_right else x > 52.5
    wide_channel = position in ("LW", "RW", "LB", "RB")
    near_byline = (x > 80.0) if attacks_right else (x < 25.0)

    dist_goal = _goal_distance(x, y, attacks_right)
    central_lane = abs(y - 34.0) < 18.0

    candidates: List[ActionCandidate] = []

    # ── SAFE_PASS: the always-available out-ball ────────────────
    safe = _nearest_safe_teammate(x, y, teammates, defenders, position_engine)
    safe_val = 0.55 + (0.15 if under_pressure else 0.0) + (0.10 if own_half else 0.0)
    safe_target = None
    if safe is not None:
        safe_target, _, _, openness = safe
        safe_val += openness * 0.20
    candidates.append(ActionCandidate(
        PlayerIntent.SAFE_PASS, _clamp(safe_val, 0.05, 1.2), risk=0.12,
        target=safe_target, note="nearby low-risk option",
    ))

    # ── RECYCLE: pass it back, reset the picture ────────────────
    deep = _deepest_teammate(x, y, teammates, position_engine, attacks_right)
    plays_safe = _get(tendencies, "plays_safe", 0.5)
    recycle_val = 0.25 + plays_safe * 0.35 + (0.25 if under_pressure else 0.0)
    if not own_half:
        recycle_val -= 0.10
    candidates.append(ActionCandidate(
        PlayerIntent.RECYCLE, _clamp(recycle_val, 0.05, 1.1), risk=0.08,
        target=deep[0] if deep else None, note="reset possession backward/sideways",
    ))

    # ── PROTECT_POSSESSION: shield it, don't release ────────────
    composure = _get(mental, "composure", 50.0) / 100.0
    strength = _get(physical, "strength", 55.0) / 100.0
    fatigue = _fatigue_estimate(player, minute)
    protect_val = 0.15 + composure * 0.20 + strength * 0.15
    game_state_name = getattr(game_state, "name", "LEVEL")
    if game_state_name in {"HOME_CRUISE", "AWAY_CRUISE"}:
        protect_val += 0.20  # game management, late-lead style
    protect_val += fatigue * 0.22  # conserve energy rather than force a risky pass
    if under_pressure:
        protect_val -= 0.10  # hard to shield the ball when someone is on you
    candidates.append(ActionCandidate(
        PlayerIntent.PROTECT_POSSESSION, _clamp(protect_val, 0.02, 1.0), risk=0.15,
        note="hold the ball up, buy time for support",
    ))

    # ── PROGRESSIVE_PASS ─────────────────────────────────────────
    fwd = _best_forward_teammate(x, y, teammates, defenders, position_engine, attacks_right)
    vision = _get(mental, "vision", 55.0) / 100.0
    decisions_attr = _get(mental, "decisions", 55.0) / 100.0
    short_passing = _get(passing, "short_passing", 55.0) / 100.0
    prog_val = 0.0
    prog_target = None
    if fwd is not None:
        prog_target, _, _, fwd_value = fwd
        prog_val = 0.20 + fwd_value * 0.55 + vision * 0.15 + short_passing * 0.10
        if final_third:
            prog_val += 0.10
    candidates.append(ActionCandidate(
        PlayerIntent.PROGRESSIVE_PASS, _clamp(prog_val, 0.0, 1.25), risk=0.35,
        target=prog_target, note="line-breaking pass to advanced teammate",
    ))

    # ── THROUGH_BALL: needs vision + a run to find + a lane ─────
    through_balls_attr = _get(passing, "through_balls", 45.0) / 100.0
    tb_tendency = _get(tendencies, "plays_through_ball", 0.10)
    tb_val = 0.0
    if fwd is not None and final_third and not tight_marked:
        _, _, _, fwd_value = fwd
        tb_val = (0.10 + fwd_value * 0.4 + through_balls_attr * 0.35
                  + vision * 0.25 + tb_tendency * 0.6)
    candidates.append(ActionCandidate(
        PlayerIntent.THROUGH_BALL, _clamp(tb_val, 0.0, 1.3), risk=0.55,
        target=fwd[0] if fwd is not None else None,
        note="line-breaking ball into space behind the defense",
    ))

    # ── SWITCH: needs a genuinely far, open outlet ───────────────
    switch = _wide_switch_teammate(x, y, teammates, defenders, position_engine)
    switch_play_attr = _get(passing, "switch_play", 50.0) / 100.0
    switch_tendency = _get(tendencies, "switches_play", 0.08)
    congestion = _defenders_within(x, y, defenders, position_engine, 8.0)
    switch_val = 0.0
    switch_target = None
    if switch is not None:
        switch_target, _, _, sw_value = switch
        switch_val = (0.10 + sw_value * 0.45 + switch_play_attr * 0.25
                      + switch_tendency * 0.8 + min(congestion, 3) * 0.06)
    candidates.append(ActionCandidate(
        PlayerIntent.SWITCH, _clamp(switch_val, 0.0, 1.2), risk=0.45,
        target=switch_target, note="diagonal to relieve congestion / exploit the far side",
    ))

    # ── CARRY: open space to advance into ────────────────────────
    dribbling = _get(technical, "dribbling", 55.0) / 100.0
    ball_control = _get(technical, "ball_control", 55.0) / 100.0
    attempts_dribble = _get(tendencies, "attempts_dribble", 0.25)
    carry_val = 0.15 + space * 0.45 + ball_control * 0.15 + dribbling * 0.10
    if final_third:
        carry_val += 0.08
    if under_pressure:
        carry_val -= 0.30
    carry_val -= fatigue * 0.20
    candidates.append(ActionCandidate(
        PlayerIntent.CARRY, _clamp(carry_val, 0.0, 1.2), risk=0.20 + (0.15 if under_pressure else 0),
        note="drive with the ball into open space",
    ))

    # ── DRIBBLE: a specific defender to beat, not just space ─────
    dribble_val = 0.0
    if tight_marked:
        confidence = _get(getattr(dna, "form", None), "confidence", 50.0) / 100.0
        dribble_val = (0.10 + dribbling * 0.45 + attempts_dribble * 0.55
                       + confidence * 0.15 - fatigue * 0.15)
        if wide_channel:
            dribble_val += 0.08
    candidates.append(ActionCandidate(
        PlayerIntent.DRIBBLE, _clamp(dribble_val, 0.0, 1.25),
        risk=0.40 + (0.10 if fatigue > 0.6 else 0),
        note="take the marker on 1v1",
    ))

    # ── CROSS: wide, near the byline, with delivery quality ─────
    crossing = _get(technical, "crossing", 50.0) / 100.0
    crosses_tendency = _get(tendencies, "crosses_from_wide", 0.40)
    cross_val = 0.0
    if wide_channel and near_byline:
        cross_val = 0.15 + crossing * 0.45 + crosses_tendency * 0.45
    candidates.append(ActionCandidate(
        PlayerIntent.CROSS, _clamp(cross_val, 0.0, 1.2), risk=0.50,
        note="deliver from the wide crossing zone",
    ))

    # ── SHOOT: only meaningful within realistic shooting range ──
    finishing = _get(technical, "finishing", 50.0) / 100.0
    long_shots = _get(technical, "long_shots", 40.0) / 100.0
    shoots_distance_tendency = _get(tendencies, "shoots_from_distance", 0.15)
    shoot_val = 0.0
    if dist_goal < 32.0 and central_lane:
        range_factor = _clamp(1.0 - dist_goal / 32.0)
        in_box = dist_goal < 18.0
        shoot_val = range_factor * (finishing * 0.6 + (0.3 if in_box else long_shots * 0.6))
        if not in_box:
            shoot_val *= (0.35 + shoots_distance_tendency * 1.5)
        if tight_marked:
            shoot_val *= 0.7
    candidates.append(ActionCandidate(
        PlayerIntent.SHOOT, _clamp(shoot_val, 0.0, 1.3), risk=0.60,
        note="pull the trigger",
    ))

    # Style nudges (reuse TeamProfile.style, same categories ActivePlayBrain
    # already used, instead of inventing a new tactical taxonomy).
    style = getattr(getattr(team_profile, "style", None), "value", "")
    adjusted: List[ActionCandidate] = []
    for c in candidates:
        v = c.objective_value
        if style in {"tiki_taka", "structured_possession", "possession"}:
            if c.intent in (PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE, PlayerIntent.PROGRESSIVE_PASS):
                v += 0.06
        elif style in {"fluid_counter", "route_one", "attacking", "ultra_attacking"}:
            if c.intent in (PlayerIntent.CARRY, PlayerIntent.THROUGH_BALL, PlayerIntent.SWITCH):
                v += 0.05
        adjusted.append(ActionCandidate(c.intent, _clamp(v, 0.0, 1.35), c.risk, c.target, c.note))

    return adjusted


# ─────────────────────────────────────────────────────────────
# PERCEPTION (BOUNDED RATIONALITY)
# ─────────────────────────────────────────────────────────────

def _archetype_risk_bias(soul: Any, intent: PlayerIntent) -> float:
    """Soul archetypes change PERCEPTION and PREFERENCE only -- never a
    free success bonus (that stays with the existing execution-time Soul
    multipliers applied elsewhere, e.g. dribble_success_mult)."""
    if soul is None:
        return 0.0
    name = getattr(getattr(soul, "archetype", None), "name", "")
    profile = getattr(soul, "profile", None)
    if profile is None:
        return 0.0
    bias = 0.0
    if intent == PlayerIntent.THROUGH_BALL:
        bias += (getattr(profile, "through_ball_mult", 1.0) - 1.0) * 0.5
    elif intent in (PlayerIntent.CARRY, PlayerIntent.DRIBBLE):
        bias += (getattr(profile, "dribble_attempt_mult", 1.0) - 1.0) * 0.5
        if intent is PlayerIntent.CARRY:
            bias += (getattr(profile, "carry_frequency_mult", 1.0) - 1.0) * 0.3
    elif intent == PlayerIntent.CROSS:
        bias += (getattr(profile, "cross_quality_mult", 1.0) - 1.0) * 0.4
    elif intent == PlayerIntent.SHOOT:
        bias += (getattr(profile, "shot_frequency_mult", 1.0) - 1.0) * 0.4
    # A soul that "reads the game early" perceives forward-risk options
    # more clearly rather than more optimistically -- reduces noise
    # elsewhere, handled in _noise_scale via can_read_game_early.
    return bias


def _preference_bias(player: Any, intent: PlayerIntent) -> float:
    """DNA tendency-driven amplification: a player who tends to attempt
    an action perceives it as more attractive than pure geometry says --
    this is preference, not extra competence."""
    dna = getattr(player, "dna", None)
    tendencies = getattr(dna, "tendencies", None)
    if intent is PlayerIntent.CARRY or intent is PlayerIntent.DRIBBLE:
        return (_get(tendencies, "attempts_dribble", 0.25) - 0.25) * 0.6
    if intent is PlayerIntent.THROUGH_BALL:
        return (_get(tendencies, "plays_through_ball", 0.10) - 0.10) * 1.2
    if intent is PlayerIntent.SWITCH:
        return (_get(tendencies, "switches_play", 0.08) - 0.08) * 1.5
    if intent is PlayerIntent.CROSS:
        return (_get(tendencies, "crosses_from_wide", 0.40) - 0.40) * 0.5
    if intent in (PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE):
        return (_get(tendencies, "plays_safe", 0.50) - 0.50) * 0.4
    if intent is PlayerIntent.SHOOT:
        return (_get(tendencies, "shoots_from_distance", 0.15) - 0.15) * 0.5
    return 0.0


_RISKY_INTENTS = frozenset({
    PlayerIntent.THROUGH_BALL, PlayerIntent.DRIBBLE, PlayerIntent.SWITCH,
    PlayerIntent.CROSS, PlayerIntent.SHOOT,
})


def _noise_scale(player: Any, soul: Any, under_pressure: bool, fatigue: float) -> float:
    """Std-dev of the evaluation-error noise. Lower decisions/composure/
    anticipation -> noisier judgement. Pressure and fatigue amplify it.
    This is the "imperfect information / decision-making ability" knob."""
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)
    decisions_attr = _get(mental, "decisions", 55.0) / 100.0
    composure = _get(mental, "composure", 55.0) / 100.0
    anticipation = _get(mental, "anticipation", 55.0) / 100.0

    base = 0.34 * (1.0 - (decisions_attr * 0.5 + anticipation * 0.3 + composure * 0.2))
    base = max(0.05, base)
    if under_pressure:
        base *= (1.55 - composure * 0.55)
    base *= (1.0 + fatigue * 0.45)
    if soul is not None and getattr(getattr(soul, "profile", None), "can_read_game_early", False):
        base *= 0.75
    return max(0.04, base)


def _visibility_floor(player: Any) -> float:
    """Objective value below this, for a given player, is simply not
    perceived. Elite vision/decisions/anticipation players notice more
    marginal options; limited players only see the obvious stuff."""
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)
    vision = _get(mental, "vision", 55.0) / 100.0
    decisions_attr = _get(mental, "decisions", 55.0) / 100.0
    anticipation = _get(mental, "anticipation", 55.0) / 100.0
    perception_index = vision * 0.45 + decisions_attr * 0.30 + anticipation * 0.25
    # 90+ perception index -> floor near 0.02 (misses almost nothing).
    # 40 perception index -> floor near 0.24 (misses genuinely useful,
    # low-salience options -- e.g. the low-vision player who never sees
    # the through-ball lane).
    return _clamp(0.30 - perception_index * 0.28, 0.02, 0.30)


def _perceive(
    candidates: List[ActionCandidate], player: Any, soul: Any,
    under_pressure: bool, fatigue: float,
) -> List[PerceivedCandidate]:
    floor = _visibility_floor(player)
    noise_scale = _noise_scale(player, soul, under_pressure, fatigue)
    perceived: List[PerceivedCandidate] = []
    for c in candidates:
        # SAFE_PASS and RECYCLE and PROTECT_POSSESSION are the reliable
        # "always considered" fallbacks -- even a limited player knows he
        # can pass it short or backward. Everything else is subject to
        # the visibility floor.
        always_visible = c.intent in (
            PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE, PlayerIntent.PROTECT_POSSESSION,
            PlayerIntent.CARRY,
        )
        if not always_visible and c.objective_value < floor:
            continue

        bias = _preference_bias(player, c.intent) + _archetype_risk_bias(soul, c.intent)
        noise_std = noise_scale
        if c.intent in _RISKY_INTENTS:
            # Creative/high-tendency players skew noise positive on risky
            # options (they overrate the spectacular ball); conservative
            # players skew negative on the same options. plays_safe is
            # the cleanest single conservatism signal already in DNA.
            dna = getattr(player, "dna", None)
            plays_safe = _get(getattr(dna, "tendencies", None), "plays_safe", 0.5)
            skew = (0.5 - plays_safe) * noise_std * 0.9
            noise = random.gauss(skew, noise_std)
        else:
            noise = random.gauss(0.0, noise_std * 0.6)

        perceived_value = max(0.0, c.objective_value * (1.0 + bias) + noise)
        perceived.append(PerceivedCandidate(c, perceived_value, noise, bias))
    return perceived


# ─────────────────────────────────────────────────────────────
# SELECTION (PROBABILISTIC, NEVER ARGMAX)
# ─────────────────────────────────────────────────────────────

def _softmax_sample(
    perceived: List[PerceivedCandidate], temperature: float,
) -> Tuple[PerceivedCandidate, float]:
    temperature = max(0.05, temperature)
    weights = [math.exp(p.perceived_value / temperature) for p in perceived]
    total = sum(weights)
    if total <= 0:
        choice = random.choice(perceived)
        return choice, 1.0 / len(perceived)
    probs = [w / total for w in weights]
    roll = random.random()
    cumulative = 0.0
    for pc, p in zip(perceived, probs):
        cumulative += p
        if roll <= cumulative:
            return pc, p
    return perceived[-1], probs[-1]


def _decision_temperature(player: Any, under_pressure: bool, fatigue: float) -> float:
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)
    decisions_attr = _get(mental, "decisions", 55.0) / 100.0
    composure = _get(mental, "composure", 55.0) / 100.0
    base = 0.28 - decisions_attr * 0.14
    if under_pressure:
        base *= (1.4 - composure * 0.3)
    base *= (1.0 + fatigue * 0.3)
    return max(0.06, base)


class DecisionBrain:
    """Chooses the on-ball carrier's next intent from bounded, imperfect
    perception of the current match state. See module docstring."""

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
        if soul is None:
            try:
                from player_soul import SoulApplicator
                soul = SoulApplicator.get_soul(player)
            except Exception:
                soul = None

        fatigue = _fatigue_estimate(player, minute)

        candidates = _generate_candidates(
            player, x, y, teammates, defenders, position_engine, team_profile,
            under_pressure, attacks_right, game_state, minute,
        )
        perceived = _perceive(candidates, player, soul, under_pressure, fatigue)
        if not perceived:
            # Should not happen (CARRY/SAFE_PASS/RECYCLE/PROTECT are
            # always visible), but never crash the match sim over a
            # decision-layer edge case.
            fallback = candidates[0]
            perceived = [PerceivedCandidate(fallback, fallback.objective_value, 0.0, 0.0)]

        temperature = _decision_temperature(player, under_pressure, fatigue)
        chosen, choice_prob = _softmax_sample(perceived, temperature)

        # decision_quality: was the chosen option actually a good one,
        # judged against the OBJECTIVE ranking of every option that
        # existed (whether perceived or not)? This is intentionally
        # computed against the full candidate set, not just what the
        # player perceived -- a player who never sees the best option
        # and picks his best PERCEIVED one can still be scored as having
        # made an objectively mediocre decision.
        ranked = sorted(candidates, key=lambda c: c.objective_value, reverse=True)
        rank = next((i for i, c in enumerate(ranked) if c.intent == chosen.candidate.intent), len(ranked) - 1)
        decision_quality = _clamp(1.0 - rank / max(1, len(ranked) - 1))
        is_error = rank >= 2  # not top-2 objectively -> a real mistake, not just a stylistic pick

        evaluation_error = abs(chosen.perceived_value - chosen.candidate.objective_value)

        intent = chosen.candidate.intent
        action = "CARRY" if intent in CARRY_LIKE_INTENTS else "PASS"

        reasons = {
            PlayerIntent.PROGRESSIVE_PASS: "line-breaking option into an advanced teammate",
            PlayerIntent.SAFE_PASS: "reliable short option under the circumstances",
            PlayerIntent.THROUGH_BALL: "a run he trusts and a lane he believes is open",
            PlayerIntent.SWITCH: "congestion on this side, space on the far side",
            PlayerIntent.CARRY: "space to advance into",
            PlayerIntent.DRIBBLE: "backs himself to beat the marker",
            PlayerIntent.CROSS: "delivery window from the wide crossing zone",
            PlayerIntent.SHOOT: "believes the window is there",
            PlayerIntent.RECYCLE: "resets the picture rather than force it",
            PlayerIntent.PROTECT_POSSESSION: "shields it and buys time for support",
        }

        trace = None
        if record_trace:
            trace = {
                "player": getattr(player, "name", "?"),
                "fatigue": round(fatigue, 3),
                "temperature": round(temperature, 3),
                "visibility_floor": round(_visibility_floor(player), 3),
                "choice_probability": round(choice_prob, 3),
                "candidates": [
                    {
                        "intent": c.intent.value,
                        "objective_value": round(c.objective_value, 3),
                        "risk": round(c.risk, 3),
                        "target": getattr(c.target, "name", None),
                        "note": c.note,
                    }
                    for c in candidates
                ],
                "perceived": [
                    {
                        "intent": p.candidate.intent.value,
                        "perceived_value": round(p.perceived_value, 3),
                        "noise": round(p.noise, 3),
                        "bias": round(p.bias, 3),
                    }
                    for p in perceived
                ],
            }

        return PlayerDecision(
            intent=intent,
            action=action,
            confidence=round(_clamp(choice_prob), 3),
            reason=reasons.get(intent, "on-ball decision"),
            risk_level=round(chosen.candidate.risk, 3),
            decision_quality=round(decision_quality, 3),
            evaluation_error=round(evaluation_error, 3),
            is_error=is_error,
            target=chosen.candidate.target,
            perceived_intents=frozenset(p.candidate.intent for p in perceived),
            trace=trace,
        )
