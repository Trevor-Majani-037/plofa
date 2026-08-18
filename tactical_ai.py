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

from match_engine import TeamProfile, MatchState, TeamStyle


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

    # Diagnostic tag so exports/commentary can say WHY (e.g. "chasing_2_late")
    posture: str = "baseline"


class TacticalAI:
    """
    Stateless — recomputed every minute from live MatchState, so a team's
    posture updates immediately as the scoreline/clock changes (a goal
    conceded in the 85th minute flips a "see it out" posture back to
    "push" instantly, exactly like a real manager's touchline reaction).
    """

    @staticmethod
    def adjust(profile: TeamProfile, state: MatchState, team_name: str,
               home_team: str, red_cards_against: int = 0, avg_stamina: float = 100.0) -> EffectiveTactics:
        gd = state.goal_difference if team_name == home_team else -state.goal_difference
        minute = state.minute

        press   = profile.press_intensity
        tempo   = profile.tempo
        direct  = profile.directness
        def_line = profile.defensive_line
        shots   = profile.shots_per_sequence
        big_ch  = profile.big_chance_ratio
        press_succ = profile.press_success_rate
        poss_target = profile.possession_target
        posture = "baseline"

        # ── CHASING THE GAME ────────────────────────────────────
        if gd <= -2 and minute >= 60:
            urgency = min(1.0, (minute - 60) / 25.0)   # ramps up 60'->85'
            press    = min(1.0, press * (1 + 0.35 * urgency))
            tempo    = min(1.0, tempo * (1 + 0.30 * urgency))
            direct   = min(1.0, direct * (1 + 0.40 * urgency))
            def_line = min(1.0, def_line * (1 + 0.25 * urgency))
            shots    = shots * (1 + 0.45 * urgency)
            big_ch   = big_ch * (1 - 0.10 * urgency)    # more shots, lower avg quality
            poss_target = min(75, poss_target * (1 + 0.15 * urgency))
            posture = "all_out_chase"

        elif gd == -1 and minute >= 70:
            urgency = min(1.0, (minute - 70) / 20.0)
            press  = min(1.0, press * (1 + 0.18 * urgency))
            tempo  = min(1.0, tempo * (1 + 0.15 * urgency))
            direct = min(1.0, direct * (1 + 0.20 * urgency))
            shots  = shots * (1 + 0.22 * urgency)
            posture = "pushing"

        # ── PROTECTING A LEAD ────────────────────────────────────
        elif gd >= 2 and minute >= 70:
            caution = min(1.0, (minute - 70) / 20.0)
            press    = press * (1 - 0.35 * caution)
            tempo    = tempo * (1 - 0.30 * caution)
            direct   = direct * (1 - 0.15 * caution)     # keep the ball, don't rush
            def_line = def_line * (1 - 0.30 * caution)   # drop deeper
            shots    = shots * (1 - 0.35 * caution)
            poss_target = poss_target * (1 - 0.05 * caution)
            posture = "see_it_out"

        elif gd == 1 and minute >= 80:
            caution = min(1.0, (minute - 80) / 10.0)
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
            posture=posture,
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
