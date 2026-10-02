"""
PLOFA 26/27 — TACTICAL AI MODULE
====================================
tactical_ai.py

Closes a technical gap flagged directly: "no tactical adjustments... teams
don't adapt in-game... a park-the-bus team plays the same statistical
profile in minute 1 and minute 89 regardless of the score."

What already existed before this module:
    MomentumEngine.get_game_state_modifier() scales shot PROBABILITY by
    scoreline — that's real, and stays. What was missing was any change
    to the team's actual TACTICAL SHAPE (press intensity, tempo,
    directness, defensive line) in response to the match state — i.e.
    a manager actually doing something, not just "tired legs try harder".

Design:
    This is intentionally NOT a rewrite of TeamProfile. It computes a
    small set of ADDITIVE adjustments on top of the team's authored style,
    representing real in-match management: throwing men forward when
    chasing, shutting up shop when ahead late, matching an opponent's
    press when being overrun. MatchEngine calls `TacticalAI.adjust()`
    once per minute per team and uses the returned EffectiveTactics in
    place of the raw profile fields for that minute's sequences.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import match_engine as _match_engine
from match_engine import TeamProfile, MatchState, TeamStyle
from tactical_shapes import FormationStance, formation_stance_for


@dataclass
class EffectiveTactics:
    """The live, in-match-adjusted version of a team's tactical dials.
    Same fields TeamProfile exposes to PossessionEngine/AttackChain, so
    callers can use this in place of the static TeamProfile without any
    other code changes."""
    style: Optional[TeamStyle]
    press_intensity: float
    tempo: float
    directness: float
    defensive_line: float
    shots_per_sequence: float
    big_chance_ratio: float
    press_success_rate: float
    possession_target: float

    # Checkpoint 30 — club-philosophy carry-over (None-safe neutral 0.0 when
    # the profile has no philosophy).  Lever B/C read these off the live
    # EffectiveTactics inside PossessionEngine.sequence_length.
    patience: float = 0.0
    starve: float = 0.0

    # Diagnostic tag so exports/commentary can say WHY (e.g. "chasing_2_late")
    posture: str = "baseline"

    # Feature #1 — the live FORMATION stance that goes with those dials
    # (all-out chase shape, seeing it out, etc.). Read directly by the
    # PositionEngine / exports for this minute.
    stance: FormationStance = FormationStance.BASELINE


def _effective_from_posture_pressing(
    profile: TeamProfile,
    state: MatchState,
    team_name: str,
    home_team: str,
    red_cards_against: int,
    avg_stamina: float,
    posture: str,
    pressing: float,
) -> EffectiveTactics:
    """Build EffectiveTactics from a posture + pressing pair.

    Shared core for the live-brain path (which polls ``Manager.decide()``
    first) and the stored-reading helpers (which reuse the manager's
    current instruction without polling). Physical modifiers are shared:
    man-down, cagey opening, fatigue are match FACTS, not choices.
    """
    press   = profile.press_intensity
    tempo   = profile.tempo
    direct  = profile.directness
    def_line = profile.defensive_line
    shots   = profile.shots_per_sequence
    big_ch  = profile.big_chance_ratio
    press_succ = profile.press_success_rate
    poss_target = profile.possession_target

    if posture == "ATTACK":
        u = pressing
        press = min(1.0, press * (1 + 0.35 * u))
        tempo = min(1.0, tempo * (1 + 0.30 * u))
        direct = min(1.0, direct * (1 + 0.40 * u))
        def_line = min(1.0, def_line * (1 + 0.25 * u))
        shots = shots * (1 + 0.45 * u)
        big_ch = big_ch * (1 - 0.10 * u)   # more shots, lower avg quality
        poss_target = min(75, poss_target * (1 + 0.15 * u))
        posture_tag = "brain_ATTACK"
        stance = FormationStance.ALL_OUT_CHASE
    elif posture == "DEFEND":
        c = pressing
        press = press * (1 - 0.35 * c)
        tempo = tempo * (1 - 0.30 * c)
        direct = direct * (1 - 0.15 * c)   # keep the ball, don't rush
        def_line = def_line * (1 - 0.30 * c)  # drop deeper
        shots = shots * (1 - 0.35 * c)
        poss_target = poss_target * (1 - 0.05 * c)
        posture_tag = "brain_DEFEND"
        stance = FormationStance.SEE_IT_OUT
    else:  # BALANCED — mild modulation anchored at the authored profile
        drift = (pressing - 0.5) * 0.10
        press = min(1.0, press * (1 + drift))
        tempo = min(1.0, tempo * (1 + drift))
        posture_tag = "brain_BALANCED"
        stance = FormationStance.BASELINE

    # ── PHYSICAL MODIFIERS (shared with the static path) ─────────
    if red_cards_against > 0:
        man_down_factor = 1.0 - 0.12 * red_cards_against
        press = press * man_down_factor
        def_line = def_line * man_down_factor
        tempo = tempo * man_down_factor
        poss_target = poss_target * man_down_factor
        posture_tag += "+man_down"
    if state.minute <= 10:
        press = press * 0.90
        direct = direct * 0.92
    if avg_stamina < 75.0:
        fatigue_factor = min(1.0, (75.0 - avg_stamina) / 25.0)  # 0.0@75, 1.0@50
        press = press * (1.0 - 0.40 * fatigue_factor)
        tempo = tempo * (1.0 - 0.25 * fatigue_factor)
        def_line = def_line * (1.0 - 0.30 * fatigue_factor)
        posture_tag += "+fatigued"

    return EffectiveTactics(
        style=profile.style if hasattr(profile, 'style') else None,
        press_intensity=round(min(1.0, max(0.05, press)), 4),
        tempo=round(min(1.0, max(0.10, tempo)), 4),
        directness=round(min(1.0, max(0.05, direct)), 4),
        defensive_line=round(min(1.0, max(0.05, def_line)), 4),
        shots_per_sequence=round(max(0.02, shots), 4),
        big_chance_ratio=round(min(0.80, max(0.15, big_ch)), 4),
        press_success_rate=round(min(0.55, max(0.05, press_succ)), 4),
        possession_target=round(min(80, max(20, poss_target)), 2),
        patience=round(float(getattr(profile, "patience", 0.0) or 0.0), 4),
        starve=round(float(getattr(profile, "starve", 0.0) or 0.0), 4),
        posture=posture_tag,
        stance=stance,
    )


def _effective_from_brain_decision(
    profile: TeamProfile,
    state: MatchState,
    team_name: str,
    home_team: str,
    red_cards_against: int,
    avg_stamina: float,
    manager,
    engine,
) -> EffectiveTactics:
    """Build EffectiveTactics from a live Manager.decide() decision.

    Phase 6 (opt-in USE_MANAGER_BRAIN): instead of the static posture
    thresholds, the full decision-maker (manager_profile.Manager — brain +
    mind + memory) announces a posture (DEFEND/BALANCED/ATTACK) and a
    pressing intensity 0..1. We translate those into the same dials the
    static path drives, so everything downstream (PossessionEngine,
    AttackChain, exports) keeps consuming EffectiveTactics unchanged.
    """
    decision = manager.decide(engine, team_name)
    posture = decision.get("posture", "BALANCED")
    pressing = min(1.0, max(0.0, float(decision.get("pressing", 0.5))))
    return _effective_from_posture_pressing(
        profile, state, team_name, home_team,
        red_cards_against, avg_stamina, posture, pressing,
    )


def brain_stored_possession_target(profile, manager) -> Optional[float]:
    """Possession target under the manager's CURRENT stored instruction.

    Phase 8: lets the live coach move the ball-share dial. Pure read of
    ``manager._current_posture`` / ``_current_pressing`` — no ``decide()``
    poll, so no dwell disturbance and no extra mind convictions. Uses the
    SAME posture math as the brain path (ATTACK lifts, DEFEND drops,
    BALANCED leaves the authored target alone).

    Returns None when the brain path is off or no live manager is wired,
    so flag-OFF callers stay byte-identical.
    """
    if not bool(getattr(_match_engine, "USE_MANAGER_BRAIN", False)):
        return None
    if manager is None or not hasattr(manager, "decide"):
        return None
    posture = getattr(manager, "_current_posture", None)
    if posture not in ("ATTACK", "DEFEND", "BALANCED"):
        return None
    pressing = min(1.0, max(0.0, float(
        getattr(manager, "_current_pressing", 0.5))))
    base = float(profile.possession_target)
    if posture == "ATTACK":
        return min(75, base * (1 + 0.15 * pressing))
    if posture == "DEFEND":
        return base * (1 - 0.05 * pressing)
    return base


# ── MANAGER TRACE REMOVED (Phase 6) ─────────────────────────────
# The Phase 0 temporary instrumentation (MANAGER_TRACE module list +
# _manager_trace()) has been retired now that the Manager Brains wiring
# lands — the brain path returns its own decision posture directly and
# no longer needs a side-channel trace list.

# ── MANAGER COLLECTION HOOK (Phase 7, opt-in) ────────────────────
# Set by manager_collection.collect_manager_samples() to record the
# STATIC manager's (sensors, posture) decisions as a training curriculum.
# None (default) = zero engine impact. Callable(team_name, state,
# posture, engine).
_COLLECTION_HOOK = None


class TacticalAI:
    """
    Stateless — recomputed every minute from live MatchState, so a team's
    posture updates immediately as the scoreline/clock changes (a goal
    conceded in the 85th minute flips a "see it out" posture back to
    "push" instantly, exactly like a real manager's touchline reaction).
    """

    @staticmethod
    def adjust(profile: TeamProfile, state: MatchState, team_name: str,
               home_team: str, red_cards_against: int = 0, avg_stamina: float = 100.0,
               manager=None, engine=None) -> EffectiveTactics:
        gd = state.goal_difference if team_name == home_team else -state.goal_difference
        minute = state.minute

        # ── MANAGER-BRAIN PATH (Phase 6, opt-in) ─────────────────
        # Master switch ON + a real Manager (has .decide) + the engine
        # supplied → the FULL decision-maker drives the dials. Polled at
        # EVERY trigger event (not every minute); the 180 s posture dwell
        # is enforced inside Manager.decide itself. Returns early so the
        # static thresholds below are left untouched.
        if (bool(getattr(_match_engine, "USE_MANAGER_BRAIN", False))
                and engine is not None and manager is not None
                and hasattr(manager, "decide")):
            return _effective_from_brain_decision(
                profile, state, team_name, home_team,
                red_cards_against, avg_stamina, manager, engine,
            )

        press   = profile.press_intensity
        tempo   = profile.tempo
        direct  = profile.directness
        def_line = profile.defensive_line
        shots   = profile.shots_per_sequence
        big_ch  = profile.big_chance_ratio
        press_succ = profile.press_success_rate
        poss_target = profile.possession_target
        posture = "baseline"

        # ── MANAGER BIAS LAYER ────────────────────────────────────
        # A manager is not a new decision-maker — it shifts the THRESHOLDS
        # at which the existing TacticalAI postures fire. An aggressive
        # manager chases earlier and protects later; a cautious one does
        # the opposite. `manager` duck-types the small surface this needs:
        #   .chase_shift()   -> minutes earlier an attacker chases
        #   .protect_shift() -> minutes earlier/later a leader parks up
        #   .risk_tolerance  -> 0 cautious, 1 aggressive
        chase_min = 60
        push_min = 70
        protect_min = 70
        lead_min = 80
        risq = 0.0
        if manager is not None and hasattr(manager, "chase_shift"):
            chase_min = max(45, 60 - manager.chase_shift())
            push_min = max(50, 70 - manager.chase_shift() * 0.7)
            protect_min = max(50, 70 - manager.protect_shift())
            lead_min = max(60, 80 - manager.protect_shift())
            # An attacking manager also pushes possession harder while chasing.
            risq = max(0.0, min(1.0, float(getattr(manager, "risk_tolerance", 0.0))))

        # ── CHASING THE GAME ────────────────────────────────────
        if gd <= -2 and minute >= chase_min:
            urgency = min(1.0, (minute - chase_min) / 25.0)   # ramps up after chase_min
            press    = min(1.0, press * (1 + 0.35 * urgency))
            tempo    = min(1.0, tempo * (1 + 0.30 * urgency))
            direct   = min(1.0, direct * (1 + 0.40 * urgency))
            def_line = min(1.0, def_line * (1 + 0.25 * urgency))
            shots    = shots * (1 + 0.45 * urgency)
            big_ch   = big_ch * (1 - 0.10 * urgency)    # more shots, lower avg quality
            poss_target = min(75, poss_target * (1 + 0.15 * urgency + 0.10 * risq))
            posture = "all_out_chase"

        elif gd == -1 and minute >= push_min:
            urgency = min(1.0, (minute - push_min) / 20.0)
            press  = min(1.0, press * (1 + 0.18 * urgency))
            tempo  = min(1.0, tempo * (1 + 0.15 * urgency))
            direct = min(1.0, direct * (1 + 0.20 * urgency))
            shots  = shots * (1 + 0.22 * urgency)
            posture = "pushing"

        # ── PROTECTING A LEAD ────────────────────────────────────
        elif gd >= 2 and minute >= protect_min:
            caution = min(1.0, (minute - protect_min) / 20.0)
            press    = press * (1 - 0.35 * caution)
            tempo    = tempo * (1 - 0.30 * caution)
            direct   = direct * (1 - 0.15 * caution)     # keep the ball, don't rush
            def_line = def_line * (1 - 0.30 * caution)   # drop deeper
            shots    = shots * (1 - 0.35 * caution)
            poss_target = poss_target * (1 - 0.05 * caution)
            posture = "see_it_out"

        elif gd == 1 and minute >= lead_min:
            caution = min(1.0, (minute - lead_min) / 10.0)
            press    = press * (1 - 0.20 * caution)
            def_line = def_line * (1 - 0.18 * caution)
            shots    = shots * (1 - 0.20 * caution)
            posture = "protect_lead"

        # ── LEVEL, LATE ──────────────────────────────────────────
        elif gd == 0 and minute >= 80:
            # Both sides know one goal wins it — mild extra intensity
            press = min(1.0, press * 1.08)
            shots = shots * 1.10
            posture = "tense_level"

        # ── DOWN TO 10 (OR FEWER) MEN ─────────────────────────────
        if red_cards_against > 0:
            man_down_factor = 1.0 - 0.12 * red_cards_against
            press = press * man_down_factor
            def_line = def_line * man_down_factor
            tempo = tempo * man_down_factor
            poss_target = poss_target * man_down_factor
            posture += "+man_down"

        # ── OPENING PUSH (first 10 minutes: cagey feel-out) ───────
        if minute <= 10:
            press = press * 0.90
            direct = direct * 0.92

        # ── FATIGUE (STAMINA) LOOP ────────────────────────────────
        if avg_stamina < 75.0:
            fatigue_factor = min(1.0, (75.0 - avg_stamina) / 25.0)  # 0.0 at 75, 1.0 at 50
            press = press * (1.0 - 0.40 * fatigue_factor)
            tempo = tempo * (1.0 - 0.25 * fatigue_factor)
            def_line = def_line * (1.0 - 0.30 * fatigue_factor)
            posture += "+fatigued"

        # Feature #1 — the shape that goes with this posture. Same scoreline /
        # clock / manager lenses, so an all-out-chase set of dials ALWAYS
        # arrives with the all-out-chase formation.
        stance = formation_stance_for(
            profile, state, team_name, home_team, manager=manager
        )

        if _COLLECTION_HOOK is not None:
            _COLLECTION_HOOK(team_name, state, posture, engine)

        return EffectiveTactics(
            style=profile.style if hasattr(profile, 'style') else None,
            press_intensity=round(min(1.0, max(0.05, press)), 4),
            tempo=round(min(1.0, max(0.10, tempo)), 4),
            directness=round(min(1.0, max(0.05, direct)), 4),
            defensive_line=round(min(1.0, max(0.05, def_line)), 4),
            shots_per_sequence=round(max(0.02, shots), 4),
            big_chance_ratio=round(min(0.80, max(0.15, big_ch)), 4),
            press_success_rate=round(min(0.55, max(0.05, press_succ)), 4),
            possession_target=round(min(80, max(20, poss_target)), 2),
            patience=round(float(getattr(profile, "patience", 0.0) or 0.0), 4),
            starve=round(float(getattr(profile, "starve", 0.0) or 0.0), 4),
            posture=posture,
            stance=stance,
        )


# ─────────────────────────────────────────────
# WIRING GUIDE
# ─────────────────────────────────────────────

WIRING_GUIDE = """
WIRING tactical_ai.py INTO match_engine.py
═════════════════════════════════════════════

In MatchEngine._simulate_minute(), right after computing home_poss/away_poss,
compute effective tactics and use THOSE for the rest of the minute instead
of self.home_profile / self.away_profile directly:

    from tactical_ai import TacticalAI

    home_tactics = TacticalAI.adjust(
        self.home_profile, self.state, home_team, home_team,
        red_cards_against=self.state.home_red_cards)
    away_tactics = TacticalAI.adjust(
        self.away_profile, self.state, away_team, home_team,
        red_cards_against=self.state.away_red_cards)

Then wherever the loop currently reads e.g. `att_profile.shots_per_sequence`
or `def_profile.press_intensity`, read `home_tactics.shots_per_sequence` /
`away_tactics.press_intensity` instead (swap in att_tactics/def_tactics
depending which team is attacking that sequence). PossessionEngine.
sequence_length() and calculate_possession_split() can take the same
EffectiveTactics object since it duck-types every field TeamProfile
exposes to them.

This is intentionally a thin layer: it doesn't change WHO plays or the
formation, just HOW urgently/high/direct they play — which is exactly
the lever a real manager pulls most often in-game, before the bigger
hammer of an actual substitution (already handled by squad_manager.py).
"""
