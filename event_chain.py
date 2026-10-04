"""
PLOFA 26/27 — EVENT CHAIN MODULE
==================================
event_chain.py

Philosophy:
    In real football, nothing happens in isolation.
    A dribble BECOMES a carry. A carry BECOMES a shot attempt.
    A press BECOMES a turnover. A turnover BECOMES a counter.
    A corner BECOMES a header. A header BECOMES a goal.

    This module models those causal chains explicitly.
    Every chain produces a sequence of MatchEvents.
    The StatAccumulator reads those events to build player stats.

    The MatchEngine calls these chains during simulation.
    Each chain returns a list of events — the engine adds them to the timeline.

Chain Types:
    PossessionChain     — Build-up play: passes, carries, progressive actions
    AttackChain         — Chance creation → shot → outcome
    SetPieceChain       — Corners, free kicks, penalties
    TransitionChain     — Press → turnover → counter-attack
    DefensiveChain      — Tackle, interception, clearance, block
    DisciplineChain     — Foul → card → consequences
    SubstitutionChain   — Player change with tactical narrative
"""

from __future__ import annotations
import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING, NamedTuple

from match_engine import (
    MatchEvent, EventType, SituationType, MatchPhase,
    GameState, XGEngine, MatchState
)
from player_dna import PlayerDNA, PlayerProfile, DNAFactory, BehavioralTendencies
from position_engine import PositionEngine, ELLIPSE_COMPOSE_FLOOR
from block_awareness import BlockNavigationEngine
from geometry_engine import (
    MovingPlayer, Vec2, Vec3, BallSpin, make_flight, make_ballistic_flight,
    resolve_aerial_delivery,
    resolve_dribble, resolve_ground_pass, resolve_shot, resolve_tackle,
    body_mass, body_balance,
)
from set_piece_routines import (
    SetPieceRoutine, corner_delivery, receiver_weight,
)
from possession_physics import (
    PossessionEpisode, aim_shot_flight, ground_pass_speed, shot_speed,
    header_shot_speed,
    sprint_speed, acceleration, reaction_time, control_radius, tackle_radius,
    jump_height, standing_reach, dive_extension, dive_speed, aerial_delivery_speed,
    delivery_spin, rolling_decel_for,
)
from attacking_matrix import (
    AttackingMatrix,
    nearest_defender_dist,
    lane_clearance,
)
from typing_extensions import TypedDict
from winger_behavior import (
    WingerBehaviorEngine,

    WingerSpatialProfile,
)
from fullback_behavior import FullbackBehaviorEngine
from possession_phases import (
    PossessionPhase,
    TacticalDirective,
    PossessionPhaseEngine,
    GKSnapshot,
    TeammateSnapshot,
    PossessionDecision,
    possession_phase_for,
)
from attack_patterns import (
    AttackPattern,
    favored_flank,
    flank_bias_multiplier,
)
from cross_detector import detect_cross
from long_pass_detector import detect_long_pass
from pass_classifier import classify_pass, apply_wind_deflection
from weather_physics import WeatherPhysics, WeatherCondition
from pressing_profiles import (
    PressingProfile,
    PROFILES as PRESS_PROFILES,
    resolve_profile,
    engagement_allows,
    cover_shadow_clearance,
    cover_shadow_blocked,
    in_cover_shadow,
    COVER_SHADOW_BLOCK_THRESHOLD,
    TRAP_RANGE_M,
    TRAP_CONVERSION_PROB,
    is_trap_profile,
    trap_present,
)
from active_play_brain import ActivePlayBrain  # noqa: F401 (legacy import kept for compat)
from decision_brain import PlayerIntent
from brain_integration import NeuralDecisionBrain
from threat_engine import (
    danger_after_clearance,
    calculate_relative_ball_angle,
    defender_facing_point,
    orientation_zone,
    clearance_failure_multiplier,
    clearance_foot_for_angle,
    apply_width_bias,
    own_goal_probability,
)
from marking import SetPieceMarkingEngine, MarkingEngine

# Checkpoint 25 — pass-execution reaction radius: a defender only cuts out a
# ball genuinely within reach (0.8m), tighter than the 1.2m planning radius
# the decision layer uses to AVOID corridors.
LANE_REACTION_DIST = 0.8
SIX_YARD_BOX_DEPTH_M = 6 * 0.9144
# Checkpoint 32 — delivery-range re-coupling. _pass_destination_to_receiver
# originally capped a pass at `pass_dist` (calibrated when teammates sat
# ~15m apart). The later shape/stretch layer spaces possession options ~2x
# wider (median ~26m), so a short pass capped at ~12m died off the
# receiver's feet (~48% underhit) and possessions collapsed. This multiplier
# scales the DELIVERY reach toward the receiver's separation so the pass
# arrives in front of him, while the soft 30m ceiling keeps a short intent
# from pinging an absurdly far runner (long passes are unaffected — their
# pass_dist already exceeds spacing).
REACH_SPACING_MULT = 1.0
REACH_SPACING_TARGET = 0.93
REACH_SPACING_CEIL_M = 30.0


def is_goalkeeper_run_out(goal_line_x: float, contact_x: float) -> bool:
    """Return whether the keeper's aerial contact was beyond the six-yard box."""
    return abs(contact_x - goal_line_x) > SIX_YARD_BOX_DEPTH_M

# Lazy import to avoid circular dependency
_soul_applicator = None
def _get_soul_applicator():
    global _soul_applicator
    if _soul_applicator is None:
        from player_soul import SoulApplicator
        _soul_applicator = SoulApplicator
    return _soul_applicator

if TYPE_CHECKING:
    from match_engine import TeamProfile


# ─────────────────────────────────────────────
# OFFSIDE CALIBRATION (realism tuning)
# ─────────────────────────────────────────────
# A purely geometric "receiver ahead of the second-last defender" test
# fires ~60+ times per match (measured) because the simulated attacking
# line routinely sits many metres past a deep defending line at the moment
# of release. Real football sees ~3-8 offsides per game. Two realisms that
# were missing close that gap:
#
#   * Law 11 "level = onside" tolerance: a receiver LEVEL with (or within
#     OFFSIDE_LEVEL_TOL_M of) the second-last defender is NOT offside.
#
#   * Run-timing / flag discipline: even when decisively beyond the line,
#     the attacker usually times the run so the ball meets them as they
#     arrive level — so the flag only goes up a fraction of the time,
#     rising with how decisively the receiver is beyond the second-last
#     defender. Culturally, most "way past the line" receivers would never
#     be played to (the pass would be dead before it arrived); treating a
#     large margin with only a modest call-rate is therefore the realistic
#     proxy and is what keeps per-match offsides near the 3-8 band instead
#     of ~60.
OFFSIDE_LEVEL_TOL_M   = 1.5   # within 1.5m of the line -> level, onside
OFFSIDE_FULL_MARGIN_M = 8.0   # beyond this, "decisiveness" saturates at 1.0
OFFSIDE_CALL_FLOOR    = 0.02  # flag probability when barely past the line
OFFSIDE_CALL_PEAK     = 0.06  # flag probability when decisively beyond


class OffsideVerdict(NamedTuple):
    """Outcome of a Law 11 check on a completed pass.

    called   — (x, y) when the flag rose: the pass is penalised at once.
    uncalled — (x, y) when the receiver WAS in an offside position but the
               flag stayed down (delayed flag / linesman missed it): play
               continues, and the receiver is stamped on the ChainResult so
               a later goal by the same player can go to VAR review.
    Either may be None; both are None when no offside position existed.
    """
    called:   Optional[Tuple[float, float]] = None
    uncalled: Optional[Tuple[float, float]] = None


# ─────────────────────────────────────────────
# SHOT-SPEED STAMPING — Opta-style launch velocity
# ─────────────────────────────────────────────
# Every shot-producing chain attaches the ball's launch velocity (m/s and
# km/h) plus flight time to the event metadata, so analytics can report
# "average shot speed" the way Opta does. Additive only: the raw timeline
# events are unchanged otherwise (physics metadata is untouched).

SHOT_SPEED_EVENT_TYPES = frozenset({
    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED,
    EventType.GOAL, EventType.HIT_WOODWORK, EventType.PENALTY_SCORED,
    EventType.PENALTY_MISSED,
})


def _stamp_shot_speed(ev: MatchEvent, mps: float, flight_s: Optional[float] = None) -> None:
    """Attach a launch velocity to an event's metadata (m/s + km/h + flight s)."""
    md = ev.metadata
    md.setdefault("shot_speed_mps", round(mps, 2))
    md.setdefault("shot_speed_kmh", round(mps * 3.6, 1))
    if flight_s is not None:
        md.setdefault("shot_flight_s", round(flight_s, 2))


def _shot_speed_for_player(player: "PlayerProfile", body_part: str = "foot") -> float:
    """Launch velocity the player's DNA would generate for this body part.

    Foot: 20-34 m/s (struck). Head: 5-12 m/s (redirected, far slower).
    Used by chains that don't run the full ballistics solver.
    """
    dna = getattr(player, "dna", None)
    physical = getattr(dna, "physical", None)
    technical = getattr(dna, "technical", None)
    strength = float(getattr(physical, "strength", 60.0))
    finishing = float(getattr(technical, "finishing", 50.0))
    if body_part == "head":
        return header_shot_speed(strength * 0.5 + finishing * 0.5)
    return shot_speed(strength * 0.4 + finishing * 0.6)


# ─────────────────────────────────────────────
# CHAIN RESULT — What every chain returns
# ─────────────────────────────────────────────

@dataclass
class ChainResult:
    """
    The output of an event chain.
    Contains events generated and key outcome flags.
    """
    events: List[MatchEvent] = field(default_factory=list)

    # Outcome flags (read by MatchEngine to update state)
    goal_scored: bool        = False
    goal_team: str           = ""
    goal_scorer: str         = ""
    goal_assistant: str      = ""
    xg_generated: float      = 0.0
    xa_generated: float      = 0.0
    shot_on_target: bool     = False
    possession_lost: bool    = False
    card_issued: bool        = False
    card_type: str           = ""    # "yellow" / "red"
    carded_player: str       = ""
    carded_team: str         = ""
    penalty_won: bool        = False
    corner_won: bool         = False
    corner_team: str         = ""    # which team the won corner belongs to
    foul_committed: bool     = False
    
    # Restart flags (Checkpoint 7 - out-of-bounds detection)
    restart_required: bool   = False
    restart_type: str        = ""    # "throw_in" / "goal_kick"
    restart_team: str        = ""    # which team takes the restart
    restart_x: float         = 0.0   # restart location x
    restart_y: float         = 0.0   # restart location y
    
    # Delayed offside flag (set when VAR rules a goal out for offside)
    delayed_offside: bool    = False

    # VAR review stamp — a pass found this receiver in an offside position
    # but the flag stayed down. If the SAME player scores later in this
    # possession, MatchEngine._maybe_var_overturn reviews the goal.
    unflagged_offside_player: str = ""
    unflagged_offside_x: float    = 0.0
    unflagged_offside_y: float    = 0.0

    # Checkpoint 19 — Offside detection: offside location for free kick placement
    offside_detected: bool   = False
    offside_x: float         = 0.0   # where the offside occurred
    offside_y: float         = 0.0
    offside_player: str      = ""    # the player who was offside
    offside_team: str        = ""    # team that was offside (attacking team)
    
    # Own-goal flag (set by DefensiveChain on a critical clearance failure)
    own_goal: bool           = False

    # Checkpoint 10 — Attacking Matrix hand-off fields. Set by PossessionChain
    # when the carrier's per-touch matrix decision is SHOOT. MatchEngine reads
    # these to dispatch the existing AttackChain shot pipeline anchored at the
    # carrier's position, and uses shot_taken to stop the independent shot_prob
    # path double-firing for the same sequence.
    shoot_decision: bool     = False
    shoot_player: str        = ""
    # WHO PASSED THE BALL TO `shoot_player`, or "" if nobody did (the carrier
    # won it himself and dribbled it in). That is a real football distinction:
    # the goal is UNASSISTED. Previously the assist was `_pick_creator`, a
    # role-and-distance-weighted random draw over the squad, so it named a man
    # who frequently appears nowhere in the run-up.
    shoot_assister: str      = ""
    shoot_x: float           = 0.0
    shoot_y: float           = 0.0
    shoot_under_pressure: bool = False
    shot_taken: bool         = False

    # ── WHO HOLDS THE BALL WHEN THE EPISODE ENDS ──────────────────────────
    # `shoot_player` above is only populated at the three SHOOT sites, so it
    # says nothing for the far more common case of a possession that ends with
    # the ball still at a player's feet and the shot decided LATER, by
    # MatchEngine's per-minute shot funnel (match_engine.py:5276). That funnel
    # was passing no names at all, so AttackChain fell back to `_pick_shooter` —
    # a role- and distance-weighted RANDOM DRAW over the squad. Measured: the
    # engine's own ledger could find a real setup pass for 30 of 50 shots while
    # the engine named a creator for 2, because the identity was sitting here
    # and was never carried out.
    #
    # `ball_carrier` is who holds it; `ball_carrier_passed_by` is who delivered
    # it to him, or "" if nobody did (he won it himself). Same derivation and
    # same honesty rule as `shoot_assister`: the passer is the previous value of
    # the carrier at each carrier change, and there are no other assignments.
    # "" is a real answer, not a gap to be filled with a plausible name.
    ball_carrier: str        = ""
    ball_carrier_passed_by: str = ""

    # Possession-time tracking (real minutes held, not a probability roll)
    sequence_duration_s: float = 0.0
    player_possession_s: Dict[str, float] = field(default_factory=dict)

    # Whole-possession continuous ball path (persistent across the episode,
    # unlike the per-action trajectory). Each entry: {t,x,y,kind,p}.
    ball_path: List[Any] = field(default_factory=list)

    # Real per-player distance / sprint accounting measured by the underlying
    # PossessionEpisode's 10 Hz trace (actual physics integration, not a roll).
    # Populated by any chain that resolves a PossessionEpisode. Maps player name
    # -> {"distance_m", "sprint_distance_m", "sprint_count", "top_speed_mps"}.
    player_distance_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(self, event: MatchEvent):
        self.events.append(event)
        return self


def mark_var_disallowed(result: ChainResult, minute: int, team: str,
                        phase: MatchPhase, game_state: GameState,
                        player: str, offside_x: float, offside_y: float) -> None:
    """Rule out a goal for offside — the VAR overturn.

    Keeps goal_scored=True so MatchEngine._absorb_chain emits the
    VAR_DISALLOWED_GOAL event and skips the score/momentum/kickoff updates;
    strips GOAL from the timeline (the ball never legally crossed the line);
    records the OFFSIDE for stats; and arms offside_* so the free kick is
    queued at the position of the offence by the existing offside path.
    """
    if not result.goal_scored or result.delayed_offside:
        return
    result.delayed_offside = True
    result.offside_detected = True
    result.offside_x = offside_x
    result.offside_y = offside_y
    result.offside_player = player
    result.offside_team = team
    result.possession_lost = True
    result.events = [e for e in result.events if e.event_type != EventType.GOAL]
    result.add(MatchEvent(
        minute=minute, second=0, event_type=EventType.OFFSIDE,
        team=team, player=player, phase=phase, game_state=game_state,
        location_x=offside_x, location_y=offside_y, outcome=False,
        metadata={"var_disallow": True, "offside_x": offside_x,
                  "offside_y": offside_y},
    ))


# ─────────────────────────────────────────────
# PITCH ZONES — Consistent spatial model
# ─────────────────────────────────────────────

class PitchZone:
    """
    Standard 105×68m pitch divided into zones.
    x=0: own goal line, x=105: opponent goal line
    y=0: left touchline, y=68: right touchline
    """
    # Zone x-ranges (meters from own goal)
    DEF_THIRD   = (0,   35)
    MID_THIRD   = (35,  70)
    ATT_THIRD   = (70,  105)
    FINAL_THIRD = (70,  105)  # alias

    # Box coordinates
    # Checkpoint 6 fix: x=105 is the goal line itself — a shot "originating"
    # there means standing inside the goal. Capped at 104.3 so generated
    # shot locations always sit a plausible half-stride before the line,
    # never on top of it (this is what was producing goals that appeared
    # to be scored from the goal-kick/goal line in exports).
    GOAL_LINE     = 105.0
    PENALTY_AREA  = (83, 104.3)   # 18-yard box
    SIX_YARD_BOX  = (99, 104.3)
    PENALTY_SPOT  = (94.0, 34.0)

    # Width zones
    LEFT_CHANNEL  = (0,   23)
    CENTRAL       = (23,  45)
    RIGHT_CHANNEL = (45,  68)

    @staticmethod
    def random_in(x_range: Tuple, y_range: Tuple = (5, 63)) -> Tuple[float, float]:
        return (
            round(random.uniform(*x_range), 1),
            round(random.uniform(*y_range), 1),
        )

    @staticmethod
    def zone_name(x: float, attacks_right: bool = True) -> str:
        if attacks_right:
            if x < 35:   return "def_third"
            if x < 70:   return "mid_third"
            if x < 83:   return "att_third"
            if x < 99:   return "penalty_area"
            return "six_yard_box"
        else:
            if x > 70:   return "def_third"
            if x > 35:   return "mid_third"
            if x > 22:   return "att_third"
            if x > 6:    return "penalty_area"
            return "six_yard_box"

    @staticmethod
    def is_in_box(x: float, attacks_right: bool = True) -> bool:
        return x >= 83 if attacks_right else x <= 22

    @staticmethod
    def xg_zone(x: float, y: float, attacks_right: bool = True) -> str:
        if attacks_right:
            if x >= 99:  return "six_yard_box"
            if x >= 83:  return "inside_box"
            if x >= 70:  return "edge_of_box"
            return "outside_box"
        else:
            if x <= 6:   return "six_yard_box"
            if x <= 22:  return "inside_box"
            if x <= 35:  return "edge_of_box"
            return "outside_box"


# ─────────────────────────────────────────────
# BASE CHAIN — Shared helpers
# ─────────────────────────────────────────────

class BaseChain:
    """Shared utilities for all chain classes."""

    @staticmethod
    def make_event(
        minute: int, event_type: EventType,
        team: str, player: str,
        phase: MatchPhase, game_state: GameState,
        **kwargs
    ) -> MatchEvent:
        return MatchEvent(
            minute=minute,
            second=kwargs.pop("second", random.randint(0, 59)),
            event_type=event_type,
            team=team,
            player=player,
            phase=phase,
            game_state=game_state,
            **kwargs
        )

    @staticmethod
    def _gk_shot_motion(gk: PlayerProfile, ball_x: float, ball_y: float,
                        attacks_right: bool) -> MovingPlayer:
        """Keeper start position for a shot: angle-bisecting depth.

        Real keepers stand on the bisector of the angle between the ball and
        the two posts, stepping off the line as distance grows. The resulting
        position is what the dive-reach model resolves against — a keeper
        covering the mouth centrally saves central shots, while a well-placed
        shot to a corner has to beat his lateral dive.
        """
        import math as _m
        goal_line = 105.0 if attacks_right else 0.0
        dist_to_goal = _m.hypot(goal_line - ball_x, 34.0 - ball_y)
        # Shot-facing depth: keepers stay within ~0.5-2.8m of the line when a
        # shot is struck (they cut angles early but are near the line when the
        # shot arrives). Deeper balls can be a touch further off the line.
        if dist_to_goal < 5.0:
            start_x = 0.5
        elif dist_to_goal < 12.0:
            start_x = 1.2
        elif dist_to_goal < 20.0:
            start_x = 1.8
        elif dist_to_goal < 30.0:
            start_x = 2.4
        else:
            start_x = 2.8
        start_x = min(3.2, max(0.4, start_x))
        # Angle bisection between the posts from the ball.
        angle_left = _m.atan2(30.34 - ball_y, goal_line - ball_x)
        angle_right = _m.atan2(37.66 - ball_y, goal_line - ball_x)
        bisector = (angle_left + angle_right) / 2.0
        gk_x = goal_line - start_x if attacks_right else goal_line + start_x
        raw_y = ball_y + _m.tan(bisector) * (gk_x - ball_x)
        # Acute angles: keepers shade the near post hard (cutting the
        # shooting lane) at the cost of leaving the far post open. A truly
        # central ball is met centrally.
        near_post = 30.34 if (ball_y < 34.0) else 37.66
        angle_frac = max(0.0, min(1.0, abs(34.0 - ball_y) / 11.0))
        if abs(34.0 - ball_y) > 2.0:
            raw_y += (near_post - 34.0) * (0.35 + 0.45 * angle_frac)
        # Keepers don't track perfectly — anticipation lag and shot disguise
        # leave a bounded positional error the geometry then resolves against
        # (a well-placed shot to the open side beats him).
        start_y = max(30.24, min(37.76, raw_y + random.gauss(0.0, 0.65)))
        motion = BaseChain._moving_player(gk, None)
        from geometry_engine import MovingPlayer as _MP
        return _MP(
            player=gk, position=Vec2(gk_x, start_y),
            pace=motion.pace, acceleration=motion.acceleration,
            reaction_time=motion.reaction_time,
            control_radius=motion.control_radius,
            tackle_radius=motion.tackle_radius,
            is_goalkeeper=True,
            jump_height=motion.jump_height,
            standing_reach=motion.standing_reach,
            dive_reach=motion.dive_reach,
             dive_speed=motion.dive_speed,
         )

    @staticmethod
    def _accumulate_physics_stats(
        episode,
        position_engine: Optional[PositionEngine],
        minute: int,
    ) -> Dict[str, Dict[str, float]]:
        """Extract per-player distance/sprint stats from episode trace and
        feed them to PositionEngine for per-minute activity aggregation.
        Returns the stats dict so the caller can also flag on-ball players."""
        if episode is None or position_engine is None:
            return {}
        try:
            stats = episode.calculate_distance_stats()
        except Exception:
            return {}
        for player_name, s in stats.items():
            position_engine.record_physics_distance(
                player_name,
                distance_m=s.get("distance_m", 0.0),
                walk_time_s=s.get("walk_time_s", 0.0),
                jog_time_s=s.get("jog_time_s", 0.0),
                sprint_time_s=s.get("sprint_time_s", 0.0),
                sprint_count=s.get("sprint_count", 0.0),
                high_speed_sprint_count=s.get("high_speed_sprint_count", 0.0),
                top_speed_mps=s.get("top_speed_mps", 0.0),
            )
        return dict(stats)

    @staticmethod
    def _inherit_restart(source: "ChainResult", target: "ChainResult") -> None:
        """Fold a sub-chain's restart (throw-in / goal-kick) into its parent.

        Sub-chains that only forward events (DefensiveChain inside
        PossessionChain / CornerChain) would otherwise lose the restart the
        engine needs to queue — e.g. a clearance to the touchline that puts
        the ball out for a throw-in."""
        if not getattr(source, "restart_required", False):
            return
        target.restart_required = True
        target.restart_type = source.restart_type
        target.restart_team = getattr(source, "restart_team", "")
        target.restart_x = getattr(source, "restart_x", None)
        target.restart_y = getattr(source, "restart_y", None)
        target.possession_lost = True

    @staticmethod
    def _moving_player(player: PlayerProfile, position_engine: Optional[PositionEngine]) -> MovingPlayer:
        """Build the short-window kinematic state used to resolve actions.

        Attributes are mapped to real football ranges (5.0-9.2 m/s sprint,
        3.0-6.0 m/s^2 acceleration, 0.15-0.42 s reaction, 0.7-1.6 m control
        radius, 0.9-1.9 m tackle radius, calibrated reach/dive envelopes).
        """
        if position_engine is not None:
            x, y = position_engine.get_position(player.name)
        else:
            x = getattr(player, "position_x", 52.5)
            y = getattr(player, "position_y", 34.0)
        dna = player.dna
        pace = float(getattr(getattr(dna, "physical", None), "pace", 60.0))
        accel_attr = float(getattr(getattr(dna, "physical", None), "acceleration", 60.0))
        ball_control = float(getattr(getattr(dna, "technical", None), "ball_control", 55.0))
        tackling = float(getattr(getattr(dna, "defending", None), "tackling", 40.0))
        anticipation = float(getattr(getattr(dna, "mental", None), "anticipation",
                                      getattr(getattr(dna, "mental", None), "vision", 50.0)))
        jumping = float(getattr(getattr(dna, "physical", None), "jumping", 50.0))
        is_gk = getattr(player, "position", "") == "GK"
        reflexes = 60.0
        diving = 60.0
        if is_gk:
            gk_attrs = getattr(dna, "gk_attrs", None)
            reflexes = float(getattr(gk_attrs, "reflexes", 60.0)) if gk_attrs else 60.0
            diving = float(getattr(gk_attrs, "diving", 60.0)) if gk_attrs else 60.0
        # Body identity (#4): mass is pure strength; balance is the leverage
        # converting mass at contact (strength + agility + ball control).
        strength_attr = float(getattr(getattr(dna, "physical", None), "strength", 50.0))
        agility_attr = float(getattr(getattr(dna, "physical", None), "agility", 50.0))
        body_mass_kg = body_mass(strength_attr)
        balance_val = body_balance(strength_attr, agility_attr, ball_control)
        return MovingPlayer(
            player=player,
            position=Vec2(x, y),
            pace=pace,
            acceleration=acceleration(accel_attr),
            reaction_time=reaction_time(anticipation, reflexes if is_gk else 60.0),
            control_radius=control_radius(ball_control),
            tackle_radius=tackle_radius(tackling),
            is_goalkeeper=is_gk,
            jump_height=jump_height(jumping),
            standing_reach=standing_reach(jumping, is_gk),
            dive_reach=dive_extension(diving) if is_gk else 0.6,
            dive_speed=dive_speed(reflexes) if is_gk else 2.2,
            body_mass_kg=body_mass_kg,
            balance=balance_val,
        )

    @classmethod
    def _dribble_confirmation_gate(
        cls,
        geometric_outcome: str,
        attacker: Optional[PlayerProfile],
        tackler: Optional[PlayerProfile],
    ) -> str:
        """Layer a skill-differential confirmation gate onto dribble geometry.

        Geometry remains the PRIMARY authority for who physically reaches the
        ball — but, exactly like the pass interception roll and the shot's
        Checkpoint 28 worldie gate, it is no longer the ONLY voice. A defender
        who geometrically wins the race can still be shrugged off by a gifted
        dribbler; and a carrier who geometrically kept the ball can still be
        hauled down by a last-ditch stretch tackle.

        Attack-side signal blends the close-control dribbling traits; defence
        blends the tackling/reading traits. The differential shifts a base
        probability either way, producing both halves of the gate:

            * ``tackled``  -> may flip to ``retained`` (shrug-off) — the
                              attacker's skill beat a defender who got there.
            * ``retained`` -> may flip to ``tackled`` (stretch tackle) — the
                              defender snatched the ball despite losing the
                              geometric race.
        """
        if attacker is None or tackler is None:
            return geometric_outcome

        a_tech = getattr(attacker, "dna", None)
        t_tech = getattr(tackler, "dna", None)
        if a_tech is None or t_tech is None:
            return geometric_outcome

        attacking_skill = (
            float(getattr(getattr(a_tech, "technical", None), "dribbling", 50.0)) * 0.50
            + float(getattr(getattr(a_tech, "physical", None), "agility", 50.0)) * 0.25
            + float(getattr(getattr(a_tech, "technical", None), "ball_control", 50.0)) * 0.25
        ) / 100.0

        defending_skill = (
            float(getattr(getattr(t_tech, "defending", None), "tackling", 50.0)) * 0.6
            + float(getattr(getattr(t_tech, "mental", None), "anticipation", 50.0)) * 0.4
        ) / 100.0

        differential = attacking_skill - defending_skill

        if geometric_outcome == "tackled":
            # Defender geometrically got there — but can the attacker still
            # shrug it off? Mirrors the pass intercept_prob roll (a skilled
            # passer beats a defender positioned to intercept).
            shrug_prob = 0.10 + differential * 0.40
            shrug_prob = max(0.05, min(0.60, shrug_prob))
            if random.random() < shrug_prob:
                return "retained"
            return "tackled"

        # geometric_outcome == "retained": the carrier kept it — but a
        # last-ditch stretch tackle can still snatch it back (the mirror of a
        # marginal shot being clawed back by a worldie save).
        stretch_prob = 0.04 - differential * 0.35
        stretch_prob = max(0.02, min(0.35, stretch_prob))
        if random.random() < stretch_prob:
            return "tackled"
        return "retained"

    @classmethod
    def _aerial_pass_gate(
        cls,
        geometric_outcome: str,
        winner_player: Optional[PlayerProfile],
        passer: PlayerProfile,
    ) -> str:
        """DECISION (documented, for audit): long balls get a Checkpoint-28
        confirmation gate — YES, deliberately.

        Geometry (the ballistic race + landing-point duel in
        ``episode.resolve_long_pass``) stays the PRIMARY authority for who
        physically wins the descent. The gate is a SECOND, attribute-based
        voice layered on top for the contested case only, mirroring exactly
        where the shots got theirs (Checkpoint 28 worldie gate), the ground
        passes got theirs (the intercept_prob skill roll), and the dribbles
        got theirs (_dribble_confirmation_gate): one voice is not the whole
        story for a contested action in this engine.

        Shape: the winning marker geometrically arrived first, but to convert
        that into an actual interception he must demonstrate the defensive
        attributes to read and win it against a lofted delivery of real
        quality. A soft, short-hoofed long passer is worse off than an elite
        diagonal router, so the differential (passer long-passing vs marker
        tackling/anticipation) moves a base probability — the same 0.15±0.35
        band the ground-pass interception roll and the old aerial fallback
        used, so long-ball interception rates stay in the same calibrated
        range while now being *distributed by the defensive attributes that
        actually matter in the air*.

        Returns the possibly-flipped geometric outcome (``received`` /
        ``intercepted``). Geometry-pure outcomes (``underhit``) pass through
        untouched — a ball nobody could physically reach is never 'revived'
        by skill on either side.
        """
        if geometric_outcome != "intercepted" or winner_player is None:
            return geometric_outcome

        int_dna = getattr(winner_player, "dna", None)
        if int_dna is None:
            return geometric_outcome

        int_tackling = float(getattr(getattr(int_dna, "defending", None), "tackling", 50.0))
        int_anticipation = float(getattr(getattr(int_dna, "mental", None), "anticipation", 50.0))
        int_skill = (int_tackling * 0.4 + int_anticipation * 0.6) / 100.0
        passing_attrs = getattr(getattr(passer, "dna", None), "passing", None)
        passer_long = float(getattr(passing_attrs, "long_passing", 55.0)) if passing_attrs is not None else 55.0
        passer_skill = passer_long / 100.0
        intercept_prob = 0.15 + (int_skill - passer_skill) * 0.35
        intercept_prob = max(0.08, min(0.50, intercept_prob))
        if random.random() < intercept_prob:
            return "intercepted"
        return "received"

    @classmethod
    def _tackle_attacker_resistance(
        cls,
        attacker: Optional[PlayerProfile],
        defender: Optional[PlayerProfile],
    ) -> float:
        """How much a skilled carrier erodes a defender's tackle rate.

        Mirrors the attacking_skill/defending_skill differential that
        _dribble_confirmation_gate layers onto the geometric dribble race
        (the physics-race path), so the two tackle systems speak the same
        language: a high-dribbling attacker measurably reduces the defender's
        chance of winning the ball, the same way they already shrug off a
        defender who geometrically got there.

        Attack blend (0.5/0.25/0.25, matching the gate's shape): dribbling,
        agility, strength. Defence blend (0.6/0.4): tackling, anticipation.
        """
        a_tech = getattr(attacker, "dna", None)
        t_tech = getattr(defender, "dna", None)

        attacking_skill = (
            float(getattr(getattr(a_tech, "technical", None), "dribbling", 50.0)) * 0.50
            + float(getattr(getattr(a_tech, "physical", None), "agility", 50.0)) * 0.25
            + float(getattr(getattr(a_tech, "physical", None), "strength", 50.0)) * 0.25
        ) / 100.0
        defending_skill = (
            float(getattr(getattr(t_tech, "defending", None), "tackling", 50.0)) * 0.6
            + float(getattr(getattr(t_tech, "mental", None), "anticipation", 50.0)) * 0.4
        ) / 100.0

        differential = attacking_skill - defending_skill
        resistance = 0.05 + differential * 0.28
        return max(0.0, min(0.30, resistance))

    # ── REACTIVE BLOCKS (pass / cross) ─────────────────────────
    # Opta split: a block is a shot stopped by an outfield body; a pass or
    # cross cut out by a body with no shot involved is a blocked pass/cross
    # (reactive, less read than an interception). Geometry: defender must be
    # inside the ball's travel corridor but arrive too late to control it —
    # close enough to throw a leg/body in the way, not to intercept clean.
    # Physics inputs: lane distance, ball speed, defender block radius
    # (tackling + bravery + strength) and reaction.
    BLOCK_CORRIDOR_M = 2.2

    @classmethod
    def _point_seg_dist(cls, px: float, py: float,
                        ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        if dx == 0.0 and dy == 0.0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    @classmethod
    def _block_radius(cls, defender: PlayerProfile) -> float:
        dna = getattr(defender, "dna", None)
        tackling = float(getattr(getattr(dna, "defending", None), "tackling", 50.0)) if dna else 50.0
        bravery = float(getattr(getattr(dna, "mental", None), "bravery", 50.0)) if dna else 50.0
        strength = float(getattr(getattr(dna, "physical", None), "strength", 50.0)) if dna else 50.0
        return 0.8 + (tackling / 100.0) * 0.7 + (bravery / 100.0) * 0.5 + (strength / 100.0) * 0.3

    @classmethod
    def _pick_pass_blocker(cls, sx: float, sy: float, ex: float, ey: float,
                           def_players, position_engine,
                           exclude: str = ""):
        """Nearest defender inside the travel corridor (mid-third of flight).

        Reactive blockers cluster in the middle of the ball path — a man
        standing on the passer's toes or the receiver's chest intercepts or
        presses instead. Returns (defender, lane_dist) or (None, inf).
        """
        best, best_d = None, float("inf")
        for d in (def_players or []):
            if getattr(d, "position", "") == "GK":
                continue
            if getattr(d, "name", "") == exclude:
                continue
            if position_engine is not None:
                try:
                    dx, dy = position_engine.get_position(d.name)
                except Exception:
                    continue
            else:
                dx, dy = getattr(d, "position_x", 52.5), getattr(d, "position_y", 34.0)
            # must be downfield of the release, not behind it
            seg_len = math.hypot(ex - sx, ey - sy)
            if seg_len < 1.0:
                continue
            t = ((dx - sx) * (ex - sx) + (dy - sy) * (ey - sy)) / (seg_len * seg_len)
            if t < 0.15 or t > 0.85:
                continue
            lane = cls._point_seg_dist(dx, dy, sx, sy, ex, ey)
            if lane <= cls.BLOCK_CORRIDOR_M and lane < best_d:
                best, best_d = d, lane
        return best, best_d

    @classmethod
    def _pass_block_probability(cls, lane_dist: float, ball_speed_mps: float,
                                defender: PlayerProfile,
                                pass_dist_m: float) -> float:
        radius = cls._block_radius(defender)
        # tight to the lane + slow ball + short pass = blockable
        lane_factor = max(0.0, 1.0 - lane_dist / cls.BLOCK_CORRIDOR_M)
        speed_factor = max(0.25, min(1.0, 18.0 / max(8.0, ball_speed_mps)))
        dist_factor = max(0.4, min(1.0, 20.0 / max(6.0, pass_dist_m)))
        reach = max(0.0, min(1.0, (radius - lane_dist) / radius + 0.35))
        prob = 0.10 + 0.55 * lane_factor * speed_factor * dist_factor * (0.4 + 0.6 * reach)
        return max(0.02, min(0.45, prob))

    @classmethod
    def _foot_for_pass(cls, player: PlayerProfile, from_x: float, from_y: float,
                       to_x: float, to_y: float, attacks_right: bool) -> str:
        """Which foot a player passes with, driven by body angle + footedness.

        The pass direction is expressed relative to the player's facing (the
        direction of attack). Within ±30° the preferred foot is used; a pass
        forced out to the flank is met with the foot on that side of the body
        — the same dead-zone model already used for clearances.
        """
        dx = to_x - from_x
        dy = to_y - from_y
        lateral = dy if attacks_right else -dy
        norm = abs(dx) if abs(dx) > 1e-6 else 1e-6
        angle_deg = math.degrees(math.atan2(lateral, norm))
        return clearance_foot_for_angle(angle_deg, player.dna.preferred_foot)

    @staticmethod
    def pick_weighted(
        players: List[PlayerProfile],
        weight_fn,
        exclude: str = None
    ) -> Optional[PlayerProfile]:
        pool = [p for p in players if p.name != exclude]
        if not pool:
            return None
        weights = [max(0.1, weight_fn(p)) for p in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    @staticmethod
    def pick_weighted_spatial(
        players: List[PlayerProfile],
        weight_fn,
        position_engine: Optional[PositionEngine],
        at_x: float,
        at_y: float,
        exclude: str = None,
        spatial_exponent: float = 1.0,
    ) -> Optional[PlayerProfile]:
        """
        Checkpoint 5: spatially-grounded version of pick_weighted().
        Multiplies the label-based weight by the player's real-time
        positional plausibility for an action happening at (at_x, at_y).

        spatial_exponent (>1.0) sharpens the proximity dominance — used by
        the builder pick so the player actually standing on the ball wins
        even when their label weight is lower than a nearby midfielder's.

        Falls back to pure label weighting if no position_engine is wired
        in (e.g. old call sites / tests) — nothing breaks.
        """
        pool = [p for p in players if p.name != exclude]
        if not pool:
            return None
        weights = []
        for p in pool:
            label_w = max(0.1, weight_fn(p))
            if position_engine is not None:
                plaus = position_engine.plausibility_at(p.name, at_x, at_y)
            else:
                plaus = 1.0
            if spatial_exponent != 1.0:
                plaus = plaus ** spatial_exponent
            weights.append(max(0.02, label_w * plaus))
        return random.choices(pool, weights=weights, k=1)[0]

    @staticmethod
    def _outfield_players(players: List[PlayerProfile]) -> List[PlayerProfile]:
        """Filter goalkeepers out of an outfield selection pool.

        GKs are the first protectors of the goal — they never press in
        midfield, tackle, intercept, clear, block, carry counters, or
        shoot. Excluding them at the source stops a GK from racking up
        outfield stats (e.g. leading the league in interceptions) at
        unrealistic midfield locations.
        """
        return [p for p in players if p.position != "GK"]

    @classmethod
    def _marking_tightness(
        cls,
        receiver: PlayerProfile,
        ball_x: float,
        ball_y: float,
        def_players: List[PlayerProfile],
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
    ) -> float:
        """
        How tightly marked is this receiver right now? 0.0 = completely free,
        1.0 = smothered.

        Uses the PositionEngine's live spatial state for BOTH the receiver and
        every defender. A defender within ~2m of the receiver who is also
        goalside (between the receiver and their own goal) is a tight mark.
        No position_engine = no marking model (safe fallback, returns 0).
        """
        if position_engine is None or not def_players:
            return 0.0

        rx, ry = position_engine.get_position(receiver.name)
        # Goalside direction: the goal the receiver attacks is at 105 (attacking right)
        # or 0 (attacking left). A defender standing between receiver and goal is
        # closer to that goal line than the receiver is.
        gx = 105.0 if attacks_right else 0.0

        nearest_dist = None
        for d in def_players:
            if d.position == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            dist = math.hypot(dx - rx, dy - ry)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist

        if nearest_dist is None:
            return 0.0

        # Distance tightness: within 1.5m ~= 1.0, beyond 10m ~= 0.0
        tight = 1.0 - max(0.0, min(1.0, (nearest_dist - 1.5) / 8.5))

        # Goalside multiplier: a defender standing between the receiver and
        # their own goal is a much tighter mark than one chasing from behind.
        goalside = False
        for d in def_players:
            if d.position == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            # Defender closer to their own goal line than receiver = goalside
            if abs(gx - dx) < abs(gx - rx):
                goalside = True
                break

        if goalside:
            tight = min(1.0, tight * 1.15)
        else:
            tight *= 0.55  # chasing from behind = far less dangerous

        return max(0.0, min(1.0, tight))

    @staticmethod
    def position_weight(pos: str, preferred: List[str], weight: float = 4.0) -> float:
        return weight if pos in preferred else 1.0

    # ── DIRECTION-AWARE PITCH HELPERS ──────────────────────────────
    # In real football, home team attacks right (toward x=105) and
    # away team attacks left (toward x=0). All spatial calculations
    # must account for which direction the acting team attacks.

    @staticmethod
    def fwd(x: float, advance: float, attacks_right: bool) -> float:
        return x + advance if attacks_right else x - advance

    @staticmethod
    def clamp_x(x: float, attacks_right: bool) -> float:
        return max(2.0, min(103.0, x))

    @staticmethod
    def goal_x(attacks_right: bool) -> float:
        return 105.0 if attacks_right else 0.0

    @staticmethod
    def goal_dist(x: float, y: float, attacks_right: bool) -> float:
        gx = 105.0 if attacks_right else 0.0
        return ((x - gx) ** 2 + (y - 34.0) ** 2) ** 0.5

    @staticmethod
    def angle_to_goal(x: float, y: float, attacks_right: bool) -> float:
        import math
        gx = 105.0 if attacks_right else 0.0
        dist_from_line = abs(gx - x)
        y_off = abs(y - 34.0)
        if dist_from_line > 0.01:
            return math.degrees(math.atan2(y_off, dist_from_line))
        return 90.0

    @staticmethod
    def is_attacking_third(x: float, attacks_right: bool) -> bool:
        if attacks_right:
            return x >= 70.0
        return x <= 35.0

    @staticmethod
    def is_final_third(x: float, attacks_right: bool) -> bool:
        return BaseChain.is_attacking_third(x, attacks_right)

    @staticmethod
    def is_deep_attack(x: float, attacks_right: bool) -> bool:
        if attacks_right:
            return x >= 80.0
        return x <= 25.0

    @staticmethod
    def mirror_x(x: float, attacks_right: bool) -> float:
        return x if attacks_right else 105.0 - x

    @classmethod
    def clamp_attack_x(cls, x: float, lo: float, hi: float,
                       attacks_right: bool) -> float:
        """Clamp an x into a band written in ATTACKING-RIGHT terms, and return
        it in the live frame.

        The bands (85-102 = the box, 85-100, 22-46 etc.) all read "near the
        goal being attacked". Clamping a live-frame x straight against them is
        only correct when attacking right: an away-team contact at x=15 gets
        dragged to x=85, which is the far end of their OWN half — so the
        header, the rebound, the scramble and the follow-up all resolve at the
        wrong goal. Measured: the away team's shots came out bimodal, roughly
        half at the correct end and half at x=85-97, the exact value of the
        hard-coded clamp. Mirror into the attacking frame, clamp, mirror back.
        """
        return cls.mirror_x(max(lo, min(hi, cls.mirror_x(x, attacks_right))),
                            attacks_right)

    @staticmethod
    def penalty_spot_x(attacks_right: bool) -> float:
        return 94.0 if attacks_right else 11.0

    @classmethod
    def _control_quality(
        cls,
        receiver: PlayerProfile,
        ball_speed_mps: float = 0.0,
        is_airborne: bool = False,
        under_pressure: bool = False,
        pass_distance_m: float = 0.0,
        pressure_level: float = 0.0,
    ) -> float:
        """
        Return miscontrol probability for a ball receipt.

        Factors:
        - Ball speed (faster = harder to control)
        - Aerial ball (lofted pass = harder to control)
        - Pressure (defender closing down)
        - Receiver's first_touch / ball_control attribute
        - Pass distance (longer = harder to judge)
        """
        base = 0.06  # baseline miscontrol ~6%
        # Ball speed penalty: 15 m/s+ is a driven pass
        if ball_speed_mps > 18.0:
            base += 0.08
        elif ball_speed_mps > 14.0:
            base += 0.04
        # Aerial ball penalty
        if is_airborne:
            base += 0.06
        # Pressure penalty
        if under_pressure:
            base += 0.05 + 0.03 * (pressure_level / 100.0)
        # Pass distance penalty
        if pass_distance_m > 25.0:
            base += 0.05
        elif pass_distance_m > 15.0:
            base += 0.02
        # Receiver skill: first_touch / ball_control reduces miscontrol
        first_touch = getattr(receiver.dna.technical, "first_touch", 55.0)
        ball_control = getattr(receiver.dna.technical, "ball_control", 55.0)
        control_skill = (first_touch + ball_control) / 200.0
        base *= max(0.3, 1.0 - control_skill * 0.5)
        if WeatherPhysics.enabled:
            ctrl_mult = WeatherPhysics.dribble_control_mult()
            base += (1.0 - ctrl_mult) * 0.15
        return max(0.02, min(0.45, base))

    @classmethod
    def _carry_distance_advance(
        cls, player: PlayerProfile, x: float, profile,
        is_micro: bool = False, is_counter: bool = False
    ) -> Tuple[float, float]:
        """Return (distance, forward_advance_ratio) driven by DNA + context.

        Carrying model is context-gated: long box-to-box carries are
        essentially counter-attack actions. In build-up play a player
        drives the ball a short-to-medium touch (up to ~15m for a
        skilled, direct carrier); only an actual counter (is_counter=True)
        produces the 10-40m slalom the telemetry counts as a true run.
        """
        carry_skill = (player.dna.physical.pace + player.dna.technical.dribbling + player.dna.technical.ball_control) / 3

        if is_counter:
            base = 10 + (carry_skill / 100) * 25
        elif is_micro:
            base = 1 + (carry_skill / 100) * 5
        else:
            base = 2 + (carry_skill / 100) * 12

        pos_mult = 1.25 if x < 35 else (0.80 if x > 70 else 1.10)
        base *= pos_mult

        stam = player.dna.physical.stamina / 100.0
        base *= (0.70 + stam * 0.30)

        style_mult = {
            "route_one": 1.30, "fluid_counter": 1.25, "direct": 1.20,
            "attacking": 1.10, "ultra_attacking": 1.15, "gegenpressing": 1.10,
            "balanced": 1.0, "wing_play": 1.05, "vertical_tiki_taka": 1.0,
            "defensive": 0.85, "park_the_bus": 0.75, "ultra_defensive": 0.80,
            "tiki_taka": 0.70, "structured_possession": 0.80
        }
        if hasattr(profile, 'style'):
            sm = style_mult.get(profile.style.value, 1.0)
        else:
            sm = 1.0
        base *= sm

        if WeatherPhysics.enabled:
            base *= WeatherPhysics.carry_distance_mult()

        lo, hi = (1, 6) if is_micro else (10, 40) if is_counter else (2, 16)
        dist = max(lo, min(hi, base))

        adv = 0.40 + (carry_skill / 200)
        adv = max(0.20, min(0.90, adv))

        return dist, adv

    @classmethod
    def _pick_aerial_threat(cls, players, exclude=None) -> Optional[PlayerProfile]:
        outfield = [p for p in players if p.position != "GK"] or players
        return cls.pick_weighted(
            outfield,
            lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2,
            exclude=exclude
        )

    @classmethod
    def _pick_set_piece_target(cls, players, zone: Optional[str],
                               exclude=None) -> Optional[PlayerProfile]:
        """Routine-zone target pick for a dead-ball delivery.

        ``receiver_weight`` (set_piece_routines) rescales the same weighted
        pick so the scheme selects its own type of attacker: the big leapers
        for the posts, the late-arriving finisher for the penalty spot, and
        the press-resistant feet for the short-corner edge. ``None`` zone
        keeps the classic aerial-threat weight exactly."""
        outfield = [p for p in players if p.position != "GK"] or players
        return cls.pick_weighted(
            outfield,
            lambda p: receiver_weight(zone, p),
            exclude=exclude,
        )

    @classmethod
    def _pick_aerial_defender(cls, players) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            players,
            lambda p: (p.dna.physical.jumping + p.dna.defending.clearances) / 2
        )

    @classmethod
    def _pick_gk_player(cls, players) -> Optional[PlayerProfile]:
        gks = [p for p in players if p.position == "GK"]
        return gks[0] if gks else None


# ─────────────────────────────────────────────
# 1. POSSESSION CHAIN
# Build-up play: passes, carries, progressive actions
# ─────────────────────────────────────────────

# Real corner/free-kick setup time: the run into the box, jostling and
# marking before the cross. Measured dead time was 0.0 s, so players were
# being TELEPORTED into position.
#
# 7.0 s, not 5.0: the median gap from a player's own position to his slot is
# 41 m, and a corner run-in is a sprint at ~6 m/s, so 5.0 s closed 30 m and
# left a median 14.7 m of ground -- only 24% of slot assignments had actually
# arrived when the ball was crossed. A real corner is typically taken 5-10 s
# after it is awarded, so 7.0 s is inside the real band and lets most of the
# box get there. Players who genuinely cannot make it still do not, which is
# correct.
SET_PIECE_JOSTLE_S = 7.0
# How close to his assigned slot a player must be to count as having made
# it into the box at the moment the cross is played. Measured: 62% of
# assignments arrive inside 3 m, 26% are still >15 m out and are correctly
# NOT offered to the aerial duel.
SET_PIECE_ARRIVED_M = 3.0


class PossessionChain(BaseChain):
    """
    Models a possession sequence from winning the ball
    to either losing it or transitioning to an attack.

    Sequence structure:
        build_up phase  → short passes in own half
        progression     → carries/long passes into midfield
        final_third     → key passes, through balls, crosses
    """

# The player policy owns the *choice* of what to attempt.  Tactical
    # phases, the attacking matrix and role behaviour remain valuable, but
    # they are advice/feasibility layers: they must not silently replace a
    # sampled player intent with a different pass or shot.
    #
    # Checkpoint 32b — the forced-receiver layer is the DELIVERY GUARANTEE.
    # Disabling it (True) let the neural brain own receiver selection
    # alone, which (a) fed attackers every touch at the expense of the
    # structural build-up and (b) stopped the phase engine's
    # RELEASE_TO_GK / EMERGENCY_DROP_TO_GK directives from reaching the
    # keeper — GK receptions collapsed to ~4-11/match and match events
    # dropped from ~3000-3700 to ~1900-2000.  With it enabled (False) the
    # phase/matrix/wide-combo forced receivers co-exist with the neural
    # brain: the policy still chooses intent, deterministic layers still
    # guarantee deliveries and keeper involvement.
    POLICY_INTENT_AUTHORITY: bool = False

    @classmethod
    def generate(
        cls,
        minute: int,
        attacking_team: str,
        players: List[PlayerProfile],
        team_profile: "TeamProfile",
        state: MatchState,
        sequence_length: int,
        defending_players: List[PlayerProfile] = None,
        position_engine: Optional[PositionEngine] = None,
        context_x: Optional[float] = None,
        context_y: Optional[float] = None,
        attacks_right: bool = True,
        def_press_intensity: Optional[float] = None,
        def_style_key: Optional[str] = None,
        att_style_key: Optional[str] = None,
        counterpress: Optional[Dict[str, Any]] = None,
        coach_instructions: Any = None,
    ) -> ChainResult:
        """
        Full StatsBomb-level possession sequence.

        Real football atomic event pattern per pass:
            CARRY (ball brought to passing position, 2-8m)
            → PASS (ball leaves foot)
            → BALL_RECEIPT (receiver controls it)
            → [PRESSURE if defender closes down]
            → [MISCONTROL if first touch fails]
            → next action...

        This generates ~8-15 events per sequence,
        matching StatsBomb's 1500-3400 events per match
        at 2-4 sequences per minute.
        """
        result = ChainResult()
        phase = state.phase
        game_state = state.game_state

        # Starting location: where possession was won
        if context_x is not None and context_y is not None:
            x, y = context_x, context_y
        else:
            x, y = cls._starting_position(team_profile)

        # Track who currently has the ball
        # Checkpoint 5: builder pick is now grounded in real spatial plausibility
        # at the sequence's starting coordinates, not just a flat label weight.
        last_player = cls._pick_builder(players, position_engine, x, y)

        # Who last PASSED to the man who currently has the ball. "" means
        # nobody did — he won the ball himself. This is the seed of an honest
        # assist: a goal by a man who received it from a team-mate is ASSISTED
        # by that team-mate, and a goal by a man who dribbled it in from
        # halfway is UNASSISTED, which is a real and common outcome, not a gap
        # to be filled with a plausible-looking name.
        #
        # It is free: the passer is the PREVIOUS value of `last_player` at every
        # point where the carrier changes. Those are the only four assignments
        # to `last_player` in this module, and all three mid-sequence ones are
        # pass-derived (a completed pass, a completed through ball, and an
        # attacker winning a cross — where crediting the CROSS TAKER is exactly
        # what the Laws award).
        last_passer = ""

        if position_engine is not None:
            position_engine.record_touch(last_player.name, x, y, minute)

        def_players = defending_players or []

        # ── PRESSING PROFILE (geometric 30° cover-shadow) ─────────────
        # The defending team's structural pressing identity drives both the
        # press probability (per-third zones scaled by the live adjusted
        # intensity) and the cover-shadow geometry that chokes forward
        # lanes — which the phase engine reads to trigger the GK Emergency
        # Phase Regression. An authored pressing style wins outright;
        # otherwise the live press_intensity band decides (so TacticalAI's
        # fatigue-driven intensity drops can pull a team down a pressing
        # tier as the match wears on).
        _def_press_i = (def_press_intensity if def_press_intensity is not None
                        else getattr(team_profile, "press_intensity", 0.5))
        press_profile = resolve_profile(_def_press_i, def_style_key)
        press_cfg = PRESS_PROFILES[press_profile]
        att_style_key = att_style_key or getattr(
            getattr(team_profile, "style", None), "value", "balanced"
        )

        # ── TACTICAL POSSESSION PHASES (Checkpoint 14) ──────────────
        # Track the current geometric phase across the sequence and run the
        # phase engine each touch. The engine treats the team's own keeper as
        # a permanent overload anchor of build-up play — the safety valve the
        # whole regression machine revolves around.
        current_phase = possession_phase_for(x, y, attacks_right)

        # ── POSSESSION-PHYSICS EPISODE (Checkpoint 27) ─────────────
        # A shared 0.1 s clock for the whole possession. Every pass, carry,
        # dribble, duel and (handed to AttackChain) shot is resolved in
        # continuous motion on this clock; the trace is consumable by the
        # exporter/analytics without changing the event schema.
        episode = PossessionEpisode()
        episode.set_ball(x, y)
        if position_engine is not None:
            episode.register([cls._moving_player(p, position_engine) for p in players])
            episode.register([cls._moving_player(p, position_engine) for p in def_players])

        for step in range(sequence_length):
            if result.possession_lost:
                break

            is_final_step = (step == sequence_length - 1)

            # ── 1. MICRO-CARRY BEFORE ACTION ──────────────────────
            # In real football, players carry the ball 2-6m between
            # receiving and their next action. StatsBomb logs these.
            # Rate: ~55% of actions are preceded by a micro-carry.
            # (Not every action — first touch directly into pass is common)
            if x > 5 and step > 0 and random.random() < 0.55:
                carry_dist, adv_ratio = cls._carry_distance_advance(
                    last_player, x, team_profile, is_micro=True
                )
                advance = carry_dist * (adv_ratio - 0.30)
                # Checkpoint 24 — the micro-carry was a 55%-per-step,
                # always-successful forward escalator: wingers received at
                # the edge and WALKED to the goal line untouched, then every
                # disposal from there stamped as a cross (20-30/match).
                # In traffic the dribbler steps sideways or gets stopped;
                # the byline cap keeps him at the cutback station; stepping
                # into a defender risks the dispossession real dribblers
                # suffer constantly.
                if last_player.position in ("LW", "RW"):
                    deep = (x > 85.0) if attacks_right else (x < 20.0)
                    if deep:
                        advance *= 0.4   # he pulls up at the cutback station
                end_cx = cls.clamp_x(x + (advance if attacks_right else -advance), attacks_right)
                if last_player.position in ("LW", "RW"):
                    end_cx = min(end_cx, 97.0) if attacks_right else max(end_cx, 8.0)
                # Checkpoint 18 wiring — a winger's micro-carry is steered
                # back onto its flank channel (small noise + touchline bias)
                # instead of being a pure random lateral walk (the old source
                # of the "inverted-10" pass map). Everyone else keeps legacy.
                _m_mode, _m_anchor, _m_bias = cls._winger_carry_steering(
                    last_player, x, y, attacks_right, False,
                    def_players, position_engine, commit_rolls=False,
                )
                # Fullbacks get the same touchline re-assertion on their
                # micro-carries — flank commitment must survive the little
                # touches too, not just the long carries (Checkpoint 30).
                if (_m_anchor is None
                        and last_player.position in ("LB", "RB")):
                    _m_mode, _m_anchor, _m_bias = cls._fullback_carry_steering(
                        last_player, x, y, attacks_right, False,
                        def_players, position_engine, commit_rolls=False,
                    )
                if _m_anchor is not None:
                    end_cy = y + _m_bias + (0.5 - random.random()) * 3
                else:
                    # PITCH WIDTH FIX: Increased from 6 to 10 to allow wider micro-carries
                    end_cy = y + (0.5 - random.random()) * 10
                end_cy = max(2, min(66, end_cy))

                # Stepping into an occupied defender ends the run — now decided
                # by geometry on the shared possession clock ("both": the ball
                # visibly travels AND race-to-ball decides the outcome).
                micro_lost = False
                if position_engine is not None:
                    mover = cls._moving_player(last_player, position_engine)
                    defs = (
                        [cls._moving_player(d, position_engine) for d in def_players
                         if getattr(d, "position", "") != "GK"]
                        if def_players else []
                    )
                    mc_res = episode.resolve_dribble(
                        Vec2(x, y), Vec2(end_cx, end_cy), mover, defs
                    )
                    _mc_tackler = getattr(mc_res.tackler, "player", None)
                    _mc_out = cls._dribble_confirmation_gate(
                        mc_res.outcome, last_player, _mc_tackler
                    )
                    micro_lost = _mc_out != "retained"
                    episode.set_ball(end_cx, end_cy)
                else:
                    # No physics feed: record the carry as a timed move so the
                    # whole-possession ball_path still spans it.
                    episode.record_ball_move(x, y, end_cx, end_cy, 6.0, "carry")

                result.add(cls.make_event(
                    minute, EventType.CARRY, attacking_team, last_player.name,
                    phase, game_state,
                    location_x=x, location_y=y,
                    end_x=end_cx, end_y=end_cy,
                    outcome=not micro_lost,
                    metadata={
                        "progressive": False,
                        "distance": round(carry_dist, 1),
                        "micro_carry": True,
                    }
                ))
                if micro_lost:
                    result.add(cls.make_event(
                        minute, EventType.DISPOSSESSED, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=end_cx, location_y=end_cy,
                        outcome=False,
                    ))
                    result.possession_lost = True
                    break
                x, y = end_cx, end_cy

            # ── 2. PRESSURE CHECK ──────────────────────────────────
            # Real StatsBomb: ~30% of passes are made under pressure.
            # Pressure events are logged as single events on the defender.
            # Under pressure = lower pass completion probability.
            #
            # Checkpoint 14: pressing now reaches the OWN THIRD. Previously
            # the guard `x > 30` meant build-up could never be pressed, so
            # the "forced back to the keeper" trigger could not fire (a
            # defender can't press a CB who is never under pressure).
            #
            # Checkpoint 15 (pressing profiles): the per-third base
            # probability now comes from the defending team's PRESSING
            # PROFILE (ultra-high gegenpress vs mid-block trap vs low-block
            # contain), gated by the profile's line of engagement and scaled
            # by the live (TacticalAI-adjusted, fatigue-aware) intensity.
            # A gegenpressing side presses the keeper/CB build-up while a
            # parked-bus side doesn't waste the energy — exactly what
            # generates the modern GK-as-overload-anchor stats.
            under_pressure = False
            pressure_player = None
            if def_players:
                press_intensity = _def_press_i
                nx = x if attacks_right else (105.0 - x)
                engaged_normal = engagement_allows(nx, press_profile)
                # ── COUNTERPRESS BURST (P2) ────────────────────────
                # Override the engagement gate when the defending team is
                # counterpressing near the recovery zone.  Within
                # COUNTERPRESS_RANGE_M of the ball-lost spot the team
                # presses regardless of the static line, and its press
                # probability is boosted by COUNTERPRESS_INTENSITY_MULT.
                cp_boost = 1.0
                if counterpress and counterpress.get("active") and state is not None:
                    cp_dist = math.hypot(
                        x - counterpress["x"], y - counterpress["y"]
                    )
                    if cp_dist <= state.COUNTERPRESS_RANGE_M:
                        engaged_normal = True
                        cp_boost = state.COUNTERPRESS_INTENSITY_MULT
                if engaged_normal:
                    press_zone_prob = press_cfg.zone_probs.get(
                        "box" if x > 83 else
                        "att_third" if x > 70 else
                        "mid_third" if x > 35 else "own_third",
                        0.20
                    )
                    # Scale by live press intensity (from the defending team)
                    press_prob = min(0.75, press_zone_prob * (0.6 + 0.6 * press_intensity) * cp_boost)
                    # A press only commits when a defender is actually within
                    # the profile's engagement range of the carrier.
                    near_def = nearest_defender_dist(x, y, def_players, position_engine)
                    if near_def is not None:
                        press_prob *= min(1.0, press_cfg.engagement_range_m / max(3.0, near_def))

                    if random.random() < press_prob:
                        under_pressure = True
                        # Bug fix (GK positional regression): this was the
                        # ONLY selection function in this whole file using
                        # flat pick_weighted() with no spatial plausibility
                        # check at all -- every other pick (_pick_builder,
                        # _pick_receiver, shooters, creators) is spatially
                        # grounded. That let a GK's flat 0.1 weight win
                        # occasionally even at x>70 (near the opponent's box),
                        # which a real keeper never does. GK now gets its own
                        # sharply-tapered weight on top of spatial plausibility,
                        # same pattern as the other two fixes in this pass.
                        def _press_weight(p: PlayerProfile) -> float:
                            if p.position == "GK":
                                return 0.5 if x <= 25 else 0.01
                            return {
                                "CDM": 3.5, "CM": 3.0, "CAM": 2.5,
                                "LW": 2.2, "RW": 2.2, "ST": 2.0,
                                "CB": 1.5, "LB": 1.2, "RB": 1.2,
                            }.get(p.position, 1.0)

                        pressure_player = cls.pick_weighted_spatial(
                            def_players, _press_weight, position_engine, x, y,
                        )
                        if pressure_player:
                            result.add(cls.make_event(
                                minute, EventType.PRESS, attacking_team, pressure_player.name,
                                phase, game_state,
                                secondary_player=last_player.name,
                                location_x=x + random.uniform(-3, 3),
                                location_y=y + random.uniform(-3, 3),
                                outcome=False,  # Outcome determined by what follows
                                metadata={
                                    "pressing": True,
                                    "zone_x": round(x, 1),
                                    "press_profile": press_profile.value,
                                    "press_tax": press_cfg.stamina_tax,
                                    "cover_shadow": True,
                                    "counterpress": bool(cp_boost > 1.0),
                                }
                            ))

            # ── 3. MAIN ACTION: PASS or CARRY or DRIBBLE ──────────
            # Checkpoint 10 — Attacking Matrix: every touch is evaluated as a
            # dynamic spatial network (shooting window, passing corridors,
            # teammate strategic value). A SHOOT resolution hands off to the
            # existing AttackChain shot pipeline (MatchEngine dispatches it
            # anchored at this touch); pass resolutions force the receiver and
            # aim the ball at their live position. No position engine wired in
            # => the matrix falls back and every existing selection routine
            # runs unchanged.
            #
            # Checkpoint 14 — Tactical Possession Phases: the phase engine
            # runs FIRST. When the geometric phase hits a dead-end (forward
            # routes congested / carrier pressed) it orders a REGRESSION —
            # recycle backward or emergency drop to the keeper — and that
            # directive OVERRIDES the forward-looking matrix. Without this a
            # bottled-up winger forces a low-probability cross instead of
            # resetting the phase to the goalkeeper, and keepers never see
            # the ball.
            matrix_decision = None
            forced_receiver = None
            forced_end = None
            matrix_meta = None
            regression_mode = None    # None | "drop_to_gk" | "recycle" | "wing_switch" | "circulation"
            wide_combo_mode = False   # Checkpoint 24: wide combination pass
            phase_decision = None

            if position_engine is not None:
                current_phase, phase_decision = cls._tactical_phase_step(
                    last_player, players, def_players, x, y, current_phase,
                    under_pressure, attacks_right, team_profile, position_engine,
                    att_style_key=att_style_key,
                    def_style_key=def_style_key,
                    def_press_intensity=_def_press_i,
                    counterpress=counterpress,
                    state=state,
                )
                if phase_decision is not None and phase_decision.directive in (
                    TacticalDirective.RECYCLE_BACKWARD,
                    TacticalDirective.RELEASE_TO_GK,
                    TacticalDirective.EMERGENCY_DROP_TO_GK,
                ):
                    regression_mode = "drop_to_gk" if phase_decision.regress_to_gk else "recycle"
                elif phase_decision is not None and phase_decision.directive == TacticalDirective.WING_SWITCH:
                    regression_mode = "wing_switch"
                elif (phase_decision is not None
                        and phase_decision.directive == TacticalDirective.SUSTAIN_CIRCULATION
                        and phase_decision.target is not None):
                    # Checkpoint 23 — tempo circulation: a DELIBERATE support
                    # pass (lateral or backward) to the phase engine's chosen
                    # target, taken while forward lanes were open. Distinct
                    # from "recycle" (which is forced by congestion) so
                    # analytics can tell patience from bailout.
                    regression_mode = "circulation"

            # A phase reset is tactical advice in policy-authority mode.  It
            # stays in telemetry through phase_decision, but cannot overwrite
            # the carrier's selected action or target below.
            phase_recommendation = regression_mode
            if cls.POLICY_INTENT_AUTHORITY:
                regression_mode = None

            if (not cls.POLICY_INTENT_AUTHORITY
                    and regression_mode is not None and phase_decision is not None
                    and phase_decision.target is not None):
                # SHOT-BEATS-REGRESSION: the phase engine orders a structural
                # reset when forward lanes are congested, but the carrier with
                # a genuinely shootable window pulls the trigger instead of
                # recycling back — a striker on the ball in the box is never
                # forced into a 90m back-pass to the keeper. The matrix is
                # deterministic (no RNG), so consulting it here is free.
                if position_engine is not None:
                    shot_decision = AttackingMatrix.decide(
                        last_player,
                        [p for p in players if p.name != last_player.name],
                        def_players, x, y,
                        position_engine=position_engine,
                        attacks_right=attacks_right,
                        team_profile=team_profile,
                        under_pressure=under_pressure,
                        danger_level=min(100.0, max(0.0, (70.0 - abs(x - (0.0 if attacks_right else 105.0))) / 70.0 * 100.0)),
                    )
                    if (shot_decision is not None and not shot_decision.fallback
                            and shot_decision.action == "SHOOT"):
                        # FIX (scoreline realism): the take-prob gate floor is
                        # raised from 0.60 to 0.70 so only genuinely high-value
                        # windows are pulled the trigger on. A marginal 0.62
                        # window now recycles instead of shooting, cutting the
                        # inflated shot volume that fed 8-8 / 10-4 scorelines.
                        #
                        # CALIBRATION (2026-09-17): floor returned to 0.70
                        # (was 0.60) after the per-minute funnel was found to
                        # be inert — the gates are the real volume control.
                        take_prob = max(0.0, min(1.0, (shot_decision.shot_score - 0.63) / 0.30))
                        if random.random() < take_prob:
                            result.add(cls.make_event(
                                minute, EventType.CARRY, attacking_team, last_player.name,
                                phase, game_state,
                                location_x=x, location_y=y,
                                end_x=x, end_y=y,
                                outcome=True,
                                metadata={
                                    "attacking_matrix": {
                                        "action": "SHOOT",
                                        "reason": shot_decision.reason,
                                        "scenario": shot_decision.scenario,
                                        "shot_score": round(shot_decision.shot_score, 3),
                                        "shot_taken": round(take_prob, 2),
                                    },
                                    "shot_intent": True,
                                }
                            ))
                            result.shoot_decision = True
                            result.shoot_player = last_player.name
                            result.shoot_assister = last_passer
                            result.shoot_x = x
                            result.shoot_y = y
                            result.shoot_under_pressure = under_pressure
                            result.shot_taken = True
                            break
                target_player = next(
                    (p for p in players if p.name == phase_decision.target), None
                )
                # ── CHECKPOINT 31: WINGERS DON'T ORCHESTRATE RESETS ──
                # A far-target circulation/switch order handed to a
                # touchline winger becomes a 25-40m diagonal he almost
                # never hits (Doku: ~1 long ball per 90). Downgrade most
                # of these to his normal short game; keep ~25% as the
                # genuine rarity.
                if target_player is not None:
                    _tx, _ty = position_engine.get_position(target_player.name)
                    if (last_player.position in ("LW", "RW")
                            and math.hypot(_tx - x, _ty - y) > 18.0
                            and random.random() < 0.75):
                        regression_mode = None
                        phase_decision = None
                        target_player = None
                if target_player is not None and not cls.POLICY_INTENT_AUTHORITY:
                    tx, ty = position_engine.get_position(target_player.name)
                    forced_receiver = target_player
                    forced_end = cls._pass_destination_to_target(tx, ty, attacks_right)
                    matrix_meta = {
                        "possession_phase": phase_decision.phase.value,
                        "phase_directive": phase_decision.directive.value,
                        "phase_reason": phase_decision.reason,
                        "recycle": regression_mode,
                    }
            if position_engine is not None:
                matrix_decision = AttackingMatrix.decide(
                    last_player,
                    [p for p in players if p.name != last_player.name],
                    def_players, x, y,
                    position_engine=position_engine,
                    attacks_right=attacks_right,
                    team_profile=team_profile,
                    under_pressure=under_pressure,
                    danger_level=min(100.0, max(0.0, (70.0 - abs(x - (0.0 if attacks_right else 105.0))) / 70.0 * 100.0)),
                    block_shape=(state.away_block if attacks_right else state.home_block),
                )

                if matrix_decision is not None and not matrix_decision.fallback:
                    matrix_meta = {
                        "attacking_matrix": {
                            "action": matrix_decision.action,
                            "reason": matrix_decision.reason,
                            "scenario": matrix_decision.scenario,
                            "shot_score": round(matrix_decision.shot_score, 3),
                        }
                    }
                    if (not cls.POLICY_INTENT_AUTHORITY
                            and matrix_decision.action == "SHOOT"):
                        # Take-probability gate: the matrix flags a shootable
                        # window (deterministic decision), but the player only
                        # pulls the trigger when the chance clearly beats the
                        # elite bar — a marginal 0.52 window is squared/recycled,
                        # a 1.0+ sitter is always taken. This keeps per-match
                        # shot volume in the same band as the pre-feature
                        # shot_prob path while the DECISION logic stays pure.
                        #
                        # FIX (scoreline realism): the gate floor is raised from
                        # 0.60 to 0.70 so only genuinely high-value windows are
                        # pulled the trigger on — marginal windows recycle through
                        # the pass network instead of inflating shot volume.
                        #
                        # CALIBRATION (2026-09-17): floor returned to 0.70 — see note above.
                        take_prob = max(0.0, min(1.0, (matrix_decision.shot_score - 0.63) / 0.30))
                        if random.random() >= take_prob:
                            matrix_decision = None  # recycle: run existing logic
                        else:
                            result.add(cls.make_event(
                                minute, EventType.CARRY, attacking_team, last_player.name,
                                phase, game_state,
                                location_x=x, location_y=y,
                                end_x=x, end_y=y,
                                outcome=True,
                                metadata={
                                    "attacking_matrix": {
                                        "action": "SHOOT",
                                        "reason": matrix_decision.reason,
                                        "scenario": matrix_decision.scenario,
                                        "shot_score": round(matrix_decision.shot_score, 3),
                                        "shot_taken": round(take_prob, 2),
                                    },
                                    "shot_intent": True,
                                }
                            ))
                            result.shoot_decision = True
                            result.shoot_player = last_player.name
                            result.shoot_assister = last_passer
                            result.shoot_x = x
                            result.shoot_y = y
                            result.shoot_under_pressure = under_pressure
                            result.shot_taken = True
                            break
                    if (not cls.POLICY_INTENT_AUTHORITY and matrix_decision is not None
                            and matrix_decision.is_pass and matrix_decision.target is not None):
                        forced_receiver = matrix_decision.target
                        forced_end = cls._pass_destination_to_target(
                            matrix_decision.target_x, matrix_decision.target_y,
                            attacks_right,
                        )
                        # Checkpoint 31 — winger matrix deliveries cap at
                        # 20m of travel: past that he clips the ball into
                        # the target's channel instead of hitting a 30m+
                        # diagonal (the Doku profile has no such pass).
                        if last_player.position in ("LW", "RW"):
                            _fx, _fy = forced_end
                            _dx, _dy = _fx - x, _fy - y
                            _fd = math.hypot(_dx, _dy)
                            if _fd > 20.0:
                                forced_end = (
                                    x + _dx / _fd * 20.0,
                                    y + _dy / _fd * 20.0,
                                )

            # ── CHECKPOINT 24: WIDE COMBINATION OVERRIDE ────────────
            # The matrix's option values are progress/depth-biased, so for a
            # WIDE carrier it kept choosing box-seekers and far runners —
            # geometrically stamped as crosses (20-25/match) and long balls.
            # A real winger's default with the ball on the flank is the
            # short game. Intercept here (shoot decisions have already
            # broken out above; through balls fire later and still can).
            if (not cls.POLICY_INTENT_AUTHORITY and (forced_receiver is None or
                    (matrix_decision is not None and matrix_decision.is_pass) or
                    (phase_decision is not None
                     and getattr(phase_decision.directive, "value", "") == "progress"))):
                if (position_engine is not None
                        and last_player.position in ("LW", "RW", "LB", "RB")
                        and regression_mode is None
                        and not cls.POLICY_INTENT_AUTHORITY):
                    _combo_target = cls._pick_wide_combo_target(
                        last_player, players, x, y,
                        position_engine, def_players, attacks_right,
                    )
                    if _combo_target is not None:
                        forced_receiver = _combo_target
                        _ctx, _cty = position_engine.get_position(_combo_target.name)
                        forced_end = cls._pass_destination_to_target(
                            _ctx, _cty, attacks_right)
                        wide_combo_mode = True
                        matrix_decision = None

            # The keeper on the ball is a distribution touch — force the pass.
            gk_distribution = (last_player.position == "GK")

            # ── ACTIVE DECISION BRAIN (bounded-rationality intent) ────
            # This is the player's on-ball INTENT, not the outcome.
            # ── NEURAL DECISION BRAIN (live) ──────────────────
            # The hand-calibrated DecisionBrain has been retired; this
            # call site now routes through NeuralDecisionBrain, a
            # per-player feedforward net evolved by genetic algorithm.
            # The policy is sampled before execution.  It selects the
            # attempted action; deterministic systems below only establish
            # whether that attempt is physically feasible and what happened.
            #
            # The 10 intent candidates (safe/progressive pass, through
            # ball, switch, carry, dribble, cross, shoot, recycle,
            # protect possession) are generated from live geometry +
            # DNA + Soul, distorted by this specific player's
            # vision/composure/decisions/anticipation/fatigue/pressure,
            # and sampled probabilistically by network temperature —
            # never argmax, never 100% accurate, never identical across
            # players.  Each player's brain auto-loads from
            # BRAIN_DIR/<POSITION>.json on first touch.
            active_decision = NeuralDecisionBrain.decide(
                last_player, x, y,
                [p for p in players if p.name != last_player.name],
                def_players, position_engine, team_profile, under_pressure,
                attacks_right, game_state, minute=minute,
                coach_instructions=coach_instructions,
            )

            # A policy-selected shot is permitted only from a real shooting
            # position.  This is deliberately a geometry feasibility check,
            # not an alternate action selector: the matrix contributes its
            # shot window score but cannot decide to shoot on the player's
            # behalf.  AttackChain remains the outcome authority for xG,
            # keeper interaction, blocks, rebounds and restarts.
            policy_shot_score = float(getattr(matrix_decision, "shot_score", 0.0) or 0.0)
            policy_shot_feasible = (
                active_decision.intent is PlayerIntent.SHOOT
                and last_player.position != "GK"
                and policy_shot_score >= 0.20
            )
            if policy_shot_feasible:
                result.add(cls.make_event(
                    minute, EventType.CARRY, attacking_team, last_player.name,
                    phase, game_state,
                    location_x=x, location_y=y, end_x=x, end_y=y,
                    outcome=True,
                    metadata={
                        "shot_intent": True,
                        "decision_authority": "player_policy",
                        "execution_feasibility": {
                            "shot_score": round(policy_shot_score, 3),
                            "minimum_shot_score": 0.20,
                        },
                        "active_brain": {
                            "action": active_decision.action,
                            "intent": active_decision.intent.value,
                            "confidence": active_decision.confidence,
                            "reason": active_decision.reason,
                            "decision_quality": active_decision.decision_quality,
                            "is_error": active_decision.is_error,
                        },
                    },
                ))
                result.shoot_decision = True
                result.shoot_player = last_player.name
                result.shoot_assister = last_passer
                result.shoot_x = x
                result.shoot_y = y
                result.shoot_under_pressure = under_pressure
                result.shot_taken = True
                break

            # ── CHECKPOINT 18: MODERN WINGER CARRY STEERING ──────────
            # The Winger Behaviour Engine's on-the-ball geometry is now live:
            #   - should_drive_byline  → commit to the touchline→byline corridor
            #   - should_cut_inside    → deliberate diagonal into an OPEN
            #                            half-space (inverted wingers)
            #   - carry_direction_bias → pull a drifted carry back onto the flank
            # These helpers were previously dead code, so every winger carry was
            # a pure random lateral walk in y — the root cause of the "inverted
            # 10" pass maps. Formation-corrected anchor (home_y, Checkpoint 21e)
            # keeps mirrored (attacking-left) wingers on the correct side.
            winger_drive_mode = None
            winger_anchor_y = None
            winger_bias = 0.0
            if (not gk_distribution and regression_mode is None
                    and position_engine is not None
                    and last_player.position in ("LW", "RW")):
                winger_drive_mode, winger_anchor_y, winger_bias = cls._winger_carry_steering(
                    last_player, x, y, attacks_right, under_pressure,
                    def_players, position_engine,
                )

            # Fullback flank commitment on the ball — same role the winger
            # steering fills for wide men: a fullback who carries should hug
            # his touchline / overlap lane, not wander across midfield. This
            # was the missing half of "fullbacks stretch the pitch": their
            # carries were a pure random lateral walk.
            fb_drive_mode = None
            fb_anchor_y = None
            fb_bias = 0.0
            if (not gk_distribution and regression_mode is None
                    and position_engine is not None
                    and last_player.position in ("LB", "RB")):
                fb_drive_mode, fb_anchor_y, fb_bias = cls._fullback_carry_steering(
                    last_player, x, y, attacks_right, under_pressure,
                    def_players, position_engine,
                )

            carry_prob = cls._carry_probability(last_player, x, team_profile)

            # Longer carry (progression attempt, not micro-carry)
            can_carry = x < 88 if attacks_right else x > 17
            # A winger who commits to a drive/cut carries the ball instead of
            # settling for a safe pass — the instinct gates carry-vs-pass in
            # the final third (Checkpoint 18).
            choose_carry = active_decision.action == "CARRY"
            if winger_drive_mode is not None:
                carry_prob = max(carry_prob, 0.80)
            if (choose_carry and can_carry and not gk_distribution):
                carry_dist, adv_ratio = cls._carry_distance_advance(
                    last_player, x, team_profile
                )
                # DNA-driven carry speed for the whole-possession ball path.
                _carry_speed = sprint_speed(float(getattr(
                    getattr(last_player.dna, "physical", None), "pace", 60.0))) * 0.68
                raw_advance = carry_dist * adv_ratio
                new_x = cls.clamp_x(x + (raw_advance if attacks_right else -raw_advance), attacks_right)
                # Checkpoint 24 — a winger carry ends at the cutback station,
                # never ON the goal line; beyond ~97 the ball is out or the
                # fullback has forced the corner.
                if last_player.position in ("LW", "RW"):
                    new_x = min(new_x, 97.0) if attacks_right else max(new_x, 8.0)
                if winger_drive_mode == "byline":
                    # Drive the touchline→byline corridor: hug the line while
                    # advancing (the modern winger's runway).
                    new_y = y + (winger_anchor_y - y) * 0.35
                    new_y += (0.5 - random.random()) * 2.0
                elif winger_drive_mode == "cut_inside":
                    # Deliberate diagonal into the (geometry-verified open)
                    # half-space — a real inverted-winger cut, not a wander.
                    cut_target = winger_anchor_y + (1.0 if winger_anchor_y < 34.0 else -1.0) * 10.0
                    new_y = y + (cut_target - y) * 0.30
                    new_y += (0.5 - random.random()) * 2.0
                elif winger_anchor_y is not None:
                    # Normal winger carry: touchline recovery bias + much
                    # smaller random noise, so the flank re-asserts itself.
                    new_y = y + winger_bias
                    new_y += (0.5 - random.random()) * (
                        2 + (last_player.dna.technical.ball_control / 100) * 4
                    )
                elif fb_drive_mode == "overlap_hold":
                    # Advanced fullback carrying down his overlap lane: hug
                    # the touchline while progressing (the modern FB runway).
                    new_y = y + (fb_anchor_y - y) * 0.35
                    new_y += (0.5 - random.random()) * 2.0
                elif fb_anchor_y is not None:
                    # Fullback carry: touchline recovery bias + small noise,
                    # so the flank re-asserts itself just like a winger's.
                    new_y = y + fb_bias
                    new_y += (0.5 - random.random()) * (
                        2 + (last_player.dna.technical.ball_control / 100) * 4
                    )
                else:
                    # PITCH WIDTH FIX: Increased from 4 + ball_control * 8 (max 12m) to
                    # 6 + ball_control * 14 (max 20m) to allow wider diagonal runs.
                    vert_range = 6 + (last_player.dna.technical.ball_control / 100) * 14
                    new_y = y + (0.5 - random.random()) * vert_range
                new_y = max(2, min(66, new_y))
                is_prog = (new_x - x) > 9.14 if attacks_right else (x - new_x) > 9.14

                # "Both" (visible travel + timing-based outcomes): when a
                # position engine is wired in, EVERY carry is resolved on the
                # shared 0.1 s possession clock. The ball travels and race-to-ball
                # decides retention — soul/DNA still shape it through control &
                # tackle radii and speed inside resolve_dribble.
                carry_resolution = None
                if position_engine is not None:
                    mover = cls._moving_player(last_player, position_engine)
                    defs = (
                        [cls._moving_player(d, position_engine) for d in def_players
                         if getattr(d, "position", "") != "GK"]
                        if def_players else []
                    )
                    carry_resolution = episode.resolve_dribble(
                        Vec2(x, y), Vec2(new_x, new_y), mover, defs
                    )
                    _carry_tackler = getattr(carry_resolution.tackler, "player", None)
                    _carry_out = cls._dribble_confirmation_gate(
                        carry_resolution.outcome, last_player, _carry_tackler
                    )
                    carry_success = _carry_out == "retained"
                else:
                    carry_success = random.random() < 0.82

                fail_x = x + (carry_dist * 0.3 if attacks_right else -carry_dist * 0.3)
                result.add(cls.make_event(
                    minute, EventType.CARRY, attacking_team, last_player.name,
                    phase, game_state,
                    location_x=x, location_y=y,
                    end_x=new_x if carry_success else cls.clamp_x(fail_x, attacks_right),
                    end_y=new_y,
                    outcome=carry_success,
                    metadata={
                        "progressive": is_prog,
                        "distance": round(carry_dist, 1),
                        "active_brain": {
                            "action": active_decision.action,
                            "intent": active_decision.intent.value,
                            "confidence": active_decision.confidence,
                            "reason": active_decision.reason,
                            "decision_quality": active_decision.decision_quality,
                            "is_error": active_decision.is_error,
                        },
                        "decision_authority": "player_policy",
                        "phase_recommendation": phase_recommendation,
                    }
                ))

                if carry_success:
                    _from_x, _from_y = x, y
                    x, y = new_x, new_y
                    episode.set_ball(x, y)
                    if position_engine is None:
                        episode.record_ball_move(_from_x, _from_y, new_x, new_y, _carry_speed, "carry")
                    if position_engine is not None:
                        position_engine.record_touch(last_player.name, x, y, minute)
                else:
                    if carry_resolution is not None:
                        _from_x, _from_y = x, y
                        x, y = carry_resolution.contact_point.x, carry_resolution.contact_point.y
                        episode.set_ball(x, y)
                        if position_engine is None:
                            episode.record_ball_move(_from_x, _from_y, x, y, _carry_speed, "carry")
                    # Lost carry → dispossession or turnover
                    result.add(cls.make_event(
                        minute, EventType.DISPOSSESSED, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=False,
                    ))
                    result.possession_lost = True
                    break

            # ── PASS ──────────────────────────────────────────────
            else:
                receiver = forced_receiver
                if receiver is None:
                    # ── CHECKPOINT 24: WIDE COMBINATION PASS ─────────
                    # A wide carrier in the final third who isn't crossing,
                    # shooting or taking his man on plays the SHORT game:
                    # cutback to the edge, lateral to the CAM, recycle to
                    # the overlapping fullback. This is 80% of a real
                    # winger's pass map (Doku: 37/39 short, 95%) — without
                    # it the engine's only wide outcomes were crosses and
                    # long diagonals. Byline drivers combine less (they'd
                    # rather carry/cross); combinators combine more.
                    if (position_engine is not None
                            and last_player.position in ("LW", "RW", "LB", "RB")
                            and regression_mode is None and not gk_distribution):
                        _combo_target = cls._pick_wide_combo_target(
                            last_player, players, x, y,
                            position_engine, def_players, attacks_right,
                        )
                        if _combo_target is not None:
                            receiver = _combo_target
                            wide_combo_mode = True
                if receiver is None:
                    if gk_distribution:
                        receiver = cls._pick_gk_distribution(
                            last_player, players, x, y, team_profile,
                            position_engine=position_engine,
                            def_players=def_players, attacks_right=attacks_right,
                        )
                        if receiver is None:
                            receiver = cls._pick_gk_distribution(
                                last_player, players, x, y, team_profile,
                                position_engine=position_engine,
                                def_players=def_players, attacks_right=attacks_right,
                                att_style_key=att_style_key,
                            )
                    else:
                        # The policy owns the intended receiver for every
                        # pass-like intent.  _find_target already selects a
                        # target appropriate to SAFE/RECYCLE/PROGRESSIVE/
                        # THROUGH/SWITCH; the heuristic picker is only a
                        # feasibility fallback when the policy has none.
                        brain_target = active_decision.target
                        if (brain_target is not None
                                and active_decision.intent in (
                                    PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE,
                                    PlayerIntent.PROGRESSIVE_PASS,
                                    PlayerIntent.THROUGH_BALL, PlayerIntent.SWITCH)
                                and any(p.name == brain_target.name for p in players)):
                            receiver = brain_target
                        else:
                            receiver = cls._pick_receiver(
                                players, last_player, x, team_profile,
                                position_engine=position_engine, y=y,
                                def_players=def_players, attacks_right=attacks_right,
                                possession_phase=current_phase if phase_decision else None,
                                match_state=state,  # Checkpoint 29 — feeds block navigation
                            )
                if not receiver:
                    break

                # ── CHECKPOINT 20: ATTACKING PROPHET SCENARIO EVALUATION ──
                # Elite playmakers (ATTACKING_PROPHET souls) evaluate multiple
                # geometric scenarios before acting. If the default receiver
                # pick is suboptimal and a better option exists, override it.
                # Only when the phase engine has NOT issued a directive:
                # a forced regression/circulation/wing-switch target is a
                # structural team order — even a genius obeys the reset (and
                # the Checkpoint 23 circulation web is exactly how the real
                # Modrics of the world play).
                if (receiver is not None and position_engine is not None
                        and regression_mode is None and not wide_combo_mode
                        and active_decision.target is None):
                    _soul = _get_soul_applicator().get_soul(last_player)
                    if (_soul is not None
                            and getattr(_soul.archetype, 'name', '') == 'ATTACKING_PROPHET'
                            and random.random() < 0.55):
                        from player_soul import SoulScenarioCalculator
                        scenario_options = SoulScenarioCalculator.evaluate_pass_options(
                            last_player, players, def_players or [],
                            x, y, position_engine, attacks_right,
                        )
                        if scenario_options:
                            top_scenario = scenario_options[0]
                            top_target = top_scenario.get('target')
                            if (top_target is not None
                                    and top_target.name != receiver.name
                                    and top_scenario.get('score', 0.0) > 0.35):
                                receiver = top_target

                # ── CHECKPOINT 42: INTENT AUTHORITY OVER THE DELIVERY CLASS ──
                # `POLICY_INTENT_AUTHORITY` was wired to the FIVE receiver-
                # selection sites above (1556/1559/1665/1709/1737) and never to
                # these two. So the "brain decides" experiment had only ever
                # switched off the brain's choice of RECEIVER, not its choice
                # of what KIND of pass to play: both branches below still ran
                # `is_prog = False` over the top of a sampled intent, which is
                # exactly the observed failure — a match in which the neural
                # brain chose PROGRESSIVE_PASS 212 times and the ball moved a
                # median of +0.5 m, with `matrix PROGRESSIVE_PASS` passes
                # travelling BACKWARDS more often than forwards (36% back vs
                # 28% fwd, median -1.4 m) and `RECYCLE_PASS` going further
                # forward (+1.1 m) than the thing named progressive.
                #
                # Gating these two is what makes the switch mean what its name
                # says. It is the last piece, not a new mechanism: with the
                # flag off, behaviour is byte-identical to today.
                if regression_mode is not None and not cls.POLICY_INTENT_AUTHORITY:
                    # Checkpoint 14 — a regression pass is a DELIBERATE
                    # backward reset: short, safe, never flagged progressive.
                    # The only exception is a direct wing-to-keeper recovery
                    # diagonal, which is a genuine long pass.
                    is_switch = False
                    is_prog = False
                    long_intent = False
                    if regression_mode == "drop_to_gk":
                        _rx, _ry = position_engine.get_position(receiver.name)
                        if math.hypot(_rx - x, _ry - y) > 25.0:
                            long_intent = True
                    elif regression_mode == "circulation":
                        # Checkpoint 23 — tempo circulation passes are aimed
                        # at a real support target; a far-side diagonal
                        # (25m+) is a genuine switch of play and must be
                        # weighted (and completed) like one, not like a 5m
                        # square ball. Shorter support passes stay safe/short.
                        _rx, _ry = position_engine.get_position(receiver.name)
                        _cdist = math.hypot(_rx - x, _ry - y)
                        if _cdist > 25.0:
                            long_intent = True
                            if abs(_ry - y) > 15.0:
                                is_switch = True
                elif wide_combo_mode and not cls.POLICY_INTENT_AUTHORITY:
                    # Checkpoint 24 — the wide combination pass (cutback /
                    # short lateral / recycle to the overlapping fullback):
                    # deliberately short and safe, never progressive-flagged.
                    is_switch = False
                    is_prog = False
                    long_intent = False
                else:
                    long_intent = cls._should_be_long_pass(last_player, x, team_profile)
                    prog_zone = (35 < x < 80) if attacks_right else (25 < x < 70)
                    is_prog   = cls._should_be_progressive(last_player, x, team_profile, prog_zone)
                    switch_threshold = last_player.dna.tendencies.switches_play
                    # Intent determines delivery class.  Tactical tendencies
                    # shape the *policy distribution* upstream; once the
                    # player has selected an intent, they do not get replaced
                    # by a second random heuristic.
                    if active_decision.intent is PlayerIntent.SWITCH:
                        is_switch, long_intent, is_prog = True, True, True
                    elif active_decision.intent is PlayerIntent.PROGRESSIVE_PASS:
                        is_switch, is_prog = False, True
                    elif active_decision.intent is PlayerIntent.THROUGH_BALL:
                        is_switch, is_prog = False, True
                    elif active_decision.intent in (
                            PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE,
                            PlayerIntent.PROTECT_POSSESSION):
                        is_switch, long_intent, is_prog = False, False, False
                    else:
                        is_switch = random.random() < switch_threshold

                # ── CHECKPOINT 14: PHASE TELEMETRY STAMP ──────────────
                # Every pass is stamped with the tactical phase it was played
                # from, the engine's directive, whether it was a deliberate
                # regression, and (for keeper distributions) who launched it.
                # Analytics (sequence engine / xT / PVA) can now group passes
                # by phase and spot the classic CB→LB→LW→(blocked)→LB→CB→GK
                # structural reset sequences.
                phase_pass_meta = {}
                if position_engine is not None and phase_decision is not None:
                    phase_pass_meta = {
                        "possession_phase": current_phase.value,
                        "phase_directive": phase_decision.directive.value,
                        "phase_reason": phase_decision.reason,
                        "recycle": regression_mode,
                        "phase_recommendation": phase_recommendation,
                    }
                if gk_distribution:
                    phase_pass_meta["gk_distribution"] = True

                marking = cls._marking_tightness(
                    receiver, x, y, def_players, position_engine, attacks_right
                )

                pass_dist = cls._pass_distance(last_player, x, long_intent, is_prog, under_pressure, team_profile)

                if forced_end is not None:
                    end_px, end_py = forced_end
                else:
                    # Checkpoint 21 — aim the ball at the RECEIVER'S LIVE
                    # POSITION instead of at a random forward vector. The
                    # receiver controls the endpoint, so the delivery and the
                    # next event's starting point are the same player/place;
                    # a winger whose post is the touchline ends up ON the
                    # touchline after the pass and nobody gets teleported into
                    # the central clump.
                    end_px, end_py = cls._pass_destination_to_receiver(
                        receiver, x, y, pass_dist,
                        position_engine, attacks_right,
                        long_intent=long_intent,
                    )

                # ── Aerodynamic wind deflection (Weather Physics) ────────
                wind_drift_m = 0.0
                # Cross detection runs below, after the event type is chosen;
                # before then only the delivery mode is known.
                delivery_airborne = bool(is_switch or long_intent)
                active_weather = getattr(state, "weather", None)
                if WeatherPhysics.is_active(active_weather):
                    end_px, end_py, wind_drift_m = WeatherPhysics.pass_lateral_deflection(
                        end_px, end_py, x, y,
                        weather=active_weather,
                        is_airborne=delivery_airborne,
                    )

                # ── RACE-TO-BALL (Checkpoint 27) ───────────────────
                # Geometry is the outcome authority for ALL passes. There
                # is deliberately NO completion roll layered on top: a
                # defender who physically reaches the travelling ball inside
                # their control radius wins it; a receiver who can't arrive
                # in the first-touch window loses it. Continuous 0.1 s ticking
                # on the shared possession clock (episode.resolve_pass for
                # ground passes, episode.resolve_aerial for lofted/long balls).
                interceptor = None
                geometry_meta = {}
                if position_engine is not None and receiver is not None:
                    receiver_motion = cls._moving_player(receiver, position_engine)
                    defender_motion = [
                        cls._moving_player(defender, position_engine)
                        for defender in def_players
                        if getattr(defender, "position", "") != "GK"
                    ]
                    gk = cls._pick_gk_player(def_players)
                    passing = float(getattr(last_player.dna.passing, "short_passing", 55.0))
                    switch_play = float(getattr(last_player.dna.passing, "switch_play", 50.0))
                    # Calibrated kick speeds: short balls 10-23 m/s; a driven
                    # progressive ball gains pace; switches and long balls use
                    # aerial_delivery_speed (lofted trajectory).
                    if is_switch or long_intent:
                        ball_speed = aerial_delivery_speed(
                            float(getattr(last_player.dna.passing, "long_passing", 55.0))
                        )
                    else:
                        ball_speed = ground_pass_speed(
                            passing, driven=3.0 if is_prog else 0.0
                        )
                        if under_pressure:
                            ball_speed += 1.0

                    # Long balls / switches use TRUE ballistic race;
                    # short passes use ground race.
                    if is_switch or long_intent:
                        aerial_height = 2.0 + random.uniform(0.5, 1.5)
                        _lst = list(defender_motion)
                        # A sweeper goalkeeper may genuinely leave
                        # his line to claim a long ball deep inside
                        # his own box. He enters the race only
                        # when the delivery is aimed at that
                        # genuine sweep zone — geometry then
                        # decides if he beats the receiver.
                        _deep = (end_px >= 83.0) if attacks_right else (end_px <= 22.0)
                        if gk is not None and _deep:
                            _lst.append(cls._moving_player(gk, position_engine))
                        episode.set_ball(x, y)
                        # Checkpoint 7 — the long ball carries real spin:
                        # a whipped out-swinger curls, a lofted pass floats on
                        # backspin. The bend is geometry (Magnus), zero at the
                        # landing point so the deflection below stays caught.
                        _passer_lp = float(getattr(getattr(last_player, "dna", None), "passing", None) and getattr(last_player.dna.passing, "long_passing", 55.0))
                        long_spin = delivery_spin(
                            _passer_lp,
                            kind="back" if random.random() < 0.35 else "side",
                        )
                        # weather=None: chain already deflected end_px/end_py
                        # at the Checkpoint weather block above (k_drag=0.09
                        # for airborne); resolve_long_pass must not apply
                        # a second deflection.
                        aerial_pass_res = episode.resolve_long_pass(
                            Vec2(x, y), Vec2(end_px, end_py),
                            ball_speed,
                            receiver_motion, _lst,
                            landing_z=aerial_height,
                            weather=None,
                            pressure_level=1.0 if under_pressure else 0.0,
                            spin=long_spin,
                        )
                        interceptor = None
                        if aerial_pass_res.outcome == "received":
                            success = True
                            gate = "none"
                        elif aerial_pass_res.outcome == "intercepted":
                            # ── CHECKPOINT-28-STYLE ATTRIBUTE GATE ──────
                            # Geometry said the defender won the landing
                            # point; the gate then asks whether his
                            # defensive attributes beat the passer's long
                            # passing quality. See
                            # PossessionChain._aerial_pass_gate.
                            gated_outcome = cls._aerial_pass_gate(
                                "intercepted",
                                aerial_pass_res.winner.player
                                if aerial_pass_res.winner is not None else None,
                                last_player,
                            )
                            gate = "checkpoint28"
                            if gated_outcome == "intercepted":
                                success = False
                                int_player = aerial_pass_res.winner.player
                                interceptor = (
                                    int_player,
                                    aerial_pass_res.contact_point.x,
                                    aerial_pass_res.contact_point.y,
                                )
                            else:
                                success = True
                        else:
                            success = False
                            gate = "none"
                        geometry_meta = {
                            "resolution": "aerial_race",
                            "ball_speed_mps": round(aerial_pass_res.launch_speed or ball_speed, 2),
                            "ball_travel_s": round(aerial_pass_res.ball_travel_time, 2),
                            "receiver_arrival_s": round(aerial_pass_res.receiver_arrival_time, 2),
                            "kinematic_outcome": aerial_pass_res.outcome,
                            "physics": episode.physics_meta("long_pass"),
                            "aerial_height_m": round(aerial_pass_res.apex_height or aerial_height, 2),
                            "launch_speed_mps": round(aerial_pass_res.launch_speed, 2),
                            "launch_angle_deg": round(aerial_pass_res.launch_angle_deg, 1),
                            "wind_drift_m": round(wind_drift_m, 2),
                            "spin_rate_rps": round(aerial_pass_res.spin_rate, 2),
                            "spin_kind": aerial_pass_res.spin_kind or "none",
                            "gate": gate,
                        }
                    else:
                        # Ground pass race
                        pass_resolution = episode.resolve_pass(
                            Vec2(x, y), Vec2(end_px, end_py), receiver_motion,
                            defender_motion, ball_speed,
                            rolling_decel=rolling_decel_for(active_weather),
                        )
                        if pass_resolution.outcome == "intercepted" and pass_resolution.interceptor is not None:
                            int_dna = pass_resolution.interceptor.player.dna
                            int_tackling = float(getattr(getattr(int_dna, "defending", None), "tackling", 50.0))
                            int_anticipation = float(getattr(getattr(int_dna, "mental", None), "anticipation", 50.0))
                            int_skill = (int_tackling * 0.4 + int_anticipation * 0.6) / 100.0
                            passer_skill = passing / 100.0
                            intercept_prob = 0.15 + (int_skill - passer_skill) * 0.35
                            intercept_prob = max(0.08, min(0.50, intercept_prob))
                            if random.random() < intercept_prob:
                                success = False
                                interceptor = (
                                    pass_resolution.interceptor.player,
                                    pass_resolution.contact_point.x,
                                    pass_resolution.contact_point.y,
                                )
                            else:
                                success = True
                        else:
                            success = pass_resolution.outcome == "received"
                        geometry_meta = {
                            "resolution": "race_to_ball",
                            "ball_speed_mps": round(ball_speed, 2),
                            "ball_travel_s": round(pass_resolution.ball_travel_time, 2),
                            "receiver_arrival_s": round(pass_resolution.receiver_arrival_time, 2),
                            "kinematic_outcome": pass_resolution.outcome,
                            "physics": episode.physics_meta("ground_pass"),
                        }
                        if pass_resolution.interceptor_arrival_time is not None:
                            geometry_meta["interceptor_arrival_s"] = round(
                                pass_resolution.interceptor_arrival_time, 2
                            )
                else:
                    # LONG BALLS / NO GEOMETRY FEED → bring long balls into the
                    # 3D aerial system. Even without a live position engine we
                    # resolve the lofted pass as a ballistic race: the ball
                    # travels a height/arc over time and the landing-point duel
                    # is settled by a receiver's jump/reach against a contesting
                    # defender's aerial geometry (jump timing + vertical reach).
                    _fallback = False
                    try:
                        receiver_motion = cls._moving_player(receiver, position_engine)
                        defender_motion = [
                            cls._moving_player(defender, position_engine)
                            for defender in (def_players or [])
                            if getattr(defender, "position", "") != "GK"
                        ]
                        gk = cls._pick_gk_player(def_players or [])
                        _fallback_deep = False
                        aerial_height = 2.0 + random.uniform(0.5, 1.5)
                        _lst = list(defender_motion)
                        # Sweeper scenario: deep delivery into the
                        # goalkeeper's own box.
                        _fallback_deep = (end_px >= 83.0) if attacks_right else (end_px <= 22.0)
                        if gk is not None and _fallback_deep:
                            _lst.append(cls._moving_player(gk, position_engine))
                        episode.set_ball(x, y)
                        # True ballistic swing: long balls carry Magnus from the
                        # passer's long-passing quality (backspin floats a
                        # loft, side spins a switch around the defender snake).
                        _fb_lp = float(getattr(getattr(last_player, "dna", None), "passing", None) and getattr(last_player.dna.passing, "long_passing", 55.0))
                        fallback_spin = delivery_spin(
                            _fb_lp,
                            kind="back" if random.random() < 0.35 else "side",
                        )
                        # weather=None: the chain already deflected end_px/
                        # end_py via pass_lateral_deflection (airborne k_drag)
                        # at the Checkpoint weather block above; resolve_long
                        # _pass must not apply a second deflection.
                        aerial_pass_res = episode.resolve_long_pass(
                            Vec2(x, y), Vec2(end_px, end_py),
                            aerial_delivery_speed(
                                float(getattr(last_player.dna.passing, "long_passing", 55.0))
                            ),
                            receiver_motion, _lst,
                            landing_z=aerial_height,
                            weather=None,
                            pressure_level=1.0 if under_pressure else 0.0,
                            spin=fallback_spin,
                        )
                        interceptor = None

                        if aerial_pass_res.outcome == "received":
                            success = True
                            gate = "none"
                        elif aerial_pass_res.outcome == "intercepted":
                            # ── CHECKPOINT-28-STYLE ATTRIBUTE GATE ──────
                            # DECISION (documented): long balls DO get the
                            # attribute confirmation gate, exactly like shots,
                            # ground passes and dribbles. Geometry decided the
                            # defender physically won the landing point; the
                            # gate then asks whether that marker's defensive
                            # attributes (tackling/anticipation) beat the
                            # passer's long-passing quality — the same band as
                            # the ground-pass interception roll. See
                            # PossessionChain._aerial_pass_gate.
                            gated_outcome = cls._aerial_pass_gate(
                                "intercepted",
                                aerial_pass_res.winner.player
                                if aerial_pass_res.winner is not None else None,
                                last_player,
                            )
                            gate = "checkpoint28"
                            if gated_outcome == "intercepted":
                                success = False
                                int_player = aerial_pass_res.winner.player
                                interceptor = (
                                    int_player,
                                    aerial_pass_res.contact_point.x,
                                    aerial_pass_res.contact_point.y,
                                )
                            else:
                                success = True
                        else:
                            success = False
                            gate = "none"
                        geometry_meta = {
                            "resolution": "aerial_race",
                            "ball_speed_mps": round(aerial_pass_res.launch_speed or 15.0, 2),
                            "ball_travel_s": round(aerial_pass_res.ball_travel_time, 2),
                            "receiver_arrival_s": round(aerial_pass_res.receiver_arrival_time, 2),
                            "kinematic_outcome": aerial_pass_res.outcome,
                            "physics": episode.physics_meta("long_pass"),
                            "aerial_height_m": round(aerial_pass_res.apex_height or aerial_height, 2),
                            "launch_speed_mps": round(aerial_pass_res.launch_speed, 2),
                            "launch_angle_deg": round(aerial_pass_res.launch_angle_deg, 1),
                            "wind_drift_m": round(wind_drift_m, 2),
                            "spin_rate_rps": round(aerial_pass_res.spin_rate, 2),
                            "spin_kind": aerial_pass_res.spin_kind or "none",
                            "gate": gate,
                        }
                    except Exception:
                        _fallback = True

                    if _fallback:
                        # If anything above was malformed, fall back to the
                        # legacy probability delivery so the possession survives.
                        success = cls._pass_success(
                            last_player, long_intent, under_pressure,
                            receiver=receiver, marking=marking,
                            confidence=last_player.dna.form.confidence,
                        )
                        episode.record_ball_move(
                            x, y, end_px, end_py, 15.0, "pass",
                            rolling_decel=rolling_decel_for(active_weather),
                        )

                if is_switch and long_intent:
                    etype = EventType.SWITCH_OF_PLAY
                elif is_prog and success and abs(end_px - x) > 9.14:
                    etype = EventType.PROGRESSIVE_PASS
                else:
                    etype = EventType.PASS

                pass_advance = end_px - x
                if not attacks_right:
                    pass_advance = -pass_advance

                # ── CHECKPOINT 11: GEOMETRIC CROSS DETECTION ─────────
                # Data providers do not classify a delivery by intent — a
                # "generic" pass that starts wide and lands in (or flashes
                # through) the opponent box IS a cross. Run the pure
                # geometric detector over every pass so qualifying deliveries
                # are stamped StatsBomb-style (`cross: true`, `is_airborne`)
                # regardless of the etype the engine chose for it.
                _cr = detect_cross(x, y, end_px, end_py, attacks_right,
                                   event_type=etype.name)
                delivery_airborne = delivery_airborne or _cr.airborne

                # ── CHECKPOINT 12: GEOMETRIC LONG PASS DETECTION ──────
                # Opta does NOT classify a pass as long from the passer's
                # intent either — any ground or airborne pass that covers
                # >= 35 yd (32 m) over the pitch surface is a Long Pass,
                # regardless of what the decision loop intended. `long_intent`
                # above only shapes how far the player tries to hit it; the
                # recorded `is_long` stamp is the geometric verdict on the
                # ACTUAL start→end delivery. Crosses, uncontrolled clearances
                # and throw-ins are excluded by provider category.
                _lp = detect_long_pass(
                    x, y, end_px, end_py,
                    event_type=etype.name,
                    is_cross=_cr.is_cross,
                    is_airborne=delivery_airborne,
                )
                is_long = _lp.is_long_pass

                # ── CHECKPOINT 13: FULL OPTA PASS CLASSIFICATION ─────
                # Beyond long/short and cross, stamp the complete Opta-style
                # pass taxonomy: pass type (chipped/launch/through ball/…),
                # length class, direction, origin half/third/channel. This is
                # a pure geometric+flag predicate chained on the same delivery.
                body_part = cls._foot_for_pass(
                    last_player, x, y, end_px, end_py, attacks_right)
                _pc = classify_pass(
                    x, y, end_px, end_py,
                    signed_dx=pass_advance,
                    is_cross=_cr.is_cross,
                    is_airborne=delivery_airborne,
                    is_headed=(body_part == "head"),
                    under_pressure=under_pressure,
                    attacks_right=attacks_right,
                    wind_drift_m=wind_drift_m,
                )

                if success and WeatherPhysics.is_active(active_weather):
                    acc_mult = WeatherPhysics.pass_accuracy_mult(
                        active_weather,
                        is_long=bool(is_long),
                        is_airborne=delivery_airborne,
                    )
                    if acc_mult < 1.0 and random.random() > acc_mult:
                        success = False

                pass_energy = cls._pass_energy_cost(
                    last_player, x, y, end_px, end_py,
                    is_long=is_long, under_pressure=under_pressure,
                    marking=marking,
                )

                # ── REACTIVE PASS/CROSS BLOCK ────────────────────
                # Opta: pass cleanly played then stopped by a body = blocked
                # pass (or blocked cross if geometric cross). Only converts
                # clean completions — intercepted/underhit balls stay
                # interceptions/turnovers. Physics: lane distance + ball
                # speed + defender block radius.
                pass_blocker = None
                pass_block_lane = 0.0
                pass_block_type = "cross" if _cr.is_cross else "pass"
                if (success and interceptor is None
                        and not gk_distribution
                        and (geometry_meta.get("kinematic_outcome") != "underhit")):
                    _blk, _lane = cls._pick_pass_blocker(
                        x, y, end_px, end_py, def_players,
                        position_engine, exclude=receiver.name,
                    )
                    if _blk is not None:
                        try:
                            _bs = float(ball_speed)
                        except Exception:
                            _bs = 15.0
                        try:
                            _pdm = float(_lp.distance_m)
                        except Exception:
                            _pdm = math.hypot(end_px - x, end_py - y)
                        _block_p = cls._pass_block_probability(
                            _lane, _bs, _blk, _pdm,
                        )
                        # ── CONTINUOUS PHYSICS GATE (opt-in) ───────────
                        # The geometric model above asks "is he near the
                        # lane?". It never asks "can he GET there before the
                        # ball does?" — so a defender twenty metres back along
                        # the line, facing the wrong way, still gets a floor
                        # chance against a 25 m/s pass.
                        #
                        # This attenuates that probability when the physics say
                        # he physically cannot arrive. It never replaces it: the
                        # engine's number is kept, only scaled down, and floored
                        # at the engine's own 0.02. Swapping in a hard
                        # arrival-time contest instead would turn a tuned
                        # probability into a guaranteed winner and move every
                        # scoreline in a season calibrated around it.
                        #
                        # Inert unless MatchConfig.physics_enabled is True; the
                        # adapter slot is None otherwise, so this is one dict
                        # lookup on a path that already computes a probability.
                        try:
                            from physics.adapter import active_adapter
                            _phys = active_adapter()
                        except Exception:
                            _phys = None
                        if _phys is not None:
                            try:
                                _phys.sync_clock(state.match_clock_s)
                                _block_p, _why = _phys.block_probability(
                                    _block_p, _blk.name, (x, y),
                                    (end_px, end_py), _bs, _lane,
                                )
                                _block_p = min(_block_p, cls._pass_block_probability(
                                    _lane, _bs, _blk, _pdm))
                            except Exception:
                                _block_p = cls._pass_block_probability(
                                    _lane, _bs, _blk, _pdm)
                        if random.random() < _block_p:
                            success = False
                            pass_blocker = _blk
                            pass_block_lane = _lane

                result.add(cls.make_event(
                    minute, etype, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=receiver.name,
                    location_x=x, location_y=y,
                    end_x=end_px, end_y=end_py,
                    outcome=success,
                    metadata={
                        "decision_authority": "player_policy",
                        "active_brain": {
                            "action": active_decision.action,
                            "intent": active_decision.intent.value,
                            "confidence": active_decision.confidence,
                            "reason": active_decision.reason,
                            "decision_quality": active_decision.decision_quality,
                            "is_error": active_decision.is_error,
                        },
                        "is_long": is_long,
                        "pass_length_m": round(_lp.distance_m, 1),
                        "pass_length_yards": round(_lp.distance_yards, 1),
                        "pass_height": _lp.height,
                        "is_progressive": is_prog,
                        "under_pressure": under_pressure,
                        "pass_advance": pass_advance,
                        "body_part": body_part,
                        "cross": _cr.is_cross,
                        "is_airborne": delivery_airborne,
                        "cross_origin": _cr.origin_zone,
                        "cross_dest": _cr.destination_zone,
                        "pass_type": _pc.pass_type,
                        "length_class": _pc.length_class,
                        "wind_drift_m": round(wind_drift_m, 2),
                        "pass_direction": _pc.direction,
                        "start_half": _pc.start_half,
                        "end_half": _pc.end_half,
                        "start_third": _pc.start_third,
                        "end_third": _pc.end_third,
                        "pass_channel": _pc.channel,
                        "pass_energy_cost": round(pass_energy, 3),
                        "geometry": geometry_meta,
                        **(phase_pass_meta or {}),
                        **(matrix_meta or {}),
                    }
                ))

                if success:
                    # ── CHECKPOINT 19: OFFSIDE DETECTION ─────────────
                    # A pass that puts the receiver in an offside position
                    # (in opponent's half, ahead of second-last defender,
                    # ahead of the ball) is penalised. The free kick is
                    # placed at the offside location, not a random zone.
                    offside_loc, offside_missed = cls._check_offside(
                        last_player, receiver, x, y, end_px, end_py,
                        def_players, position_engine, attacks_right,
                    )
                    if offside_loc is not None:
                        ox, oy = offside_loc
                        result.add(cls.make_event(
                            minute, EventType.OFFSIDE, attacking_team, receiver.name,
                            phase, game_state,
                            location_x=ox, location_y=oy,
                            outcome=False,
                            metadata={
                                "passer": last_player.name,
                                "offside_x": ox,
                                "offside_y": oy,
                            }
                        ))
                        result.offside_detected = True
                        result.offside_x = ox
                        result.offside_y = oy
                        result.offside_player = receiver.name
                        result.offside_team = attacking_team
                        result.possession_lost = True
                        break

                    if offside_missed is not None:
                        # Delayed offside: the receiver stood in an offside
                        # position but the flag stayed down — play continues.
                        # Stamped so a later goal by THIS player in this
                        # possession goes to VAR (MatchEngine._maybe_var_overturn).
                        result.unflagged_offside_player = receiver.name
                        result.unflagged_offside_x, result.unflagged_offside_y = offside_missed

                    # ── BALL RECEIPT ─────────────────────────────
                    # StatsBomb logs a BALL_RECEIPT event for every
                    # completed pass. This is where ~900 extra events come from.
                    # Receiver may miscontrol — geometry-aware when position_engine
                    # is available (ball speed, pressure, distance, skill).
                    miscontrol_prob = 0.08
                    if position_engine is not None:
                        ball_speed = geometry_meta.get("ball_speed_mps", 0.0)
                        pass_dist = geometry_meta.get("ball_travel_s", 0.0) * ball_speed if ball_speed > 0 else 0.0
                        # pressure_level: 0..100 scale using def_press_intensity
                        pressure_level = float(def_press_intensity or 0.0) * 100.0
                        miscontrol_prob = cls._control_quality(
                            receiver,
                            ball_speed_mps=ball_speed,
                            is_airborne=geometry_meta.get("resolution") == "aerial_race",
                            under_pressure=under_pressure,
                            pass_distance_m=pass_dist,
                            pressure_level=pressure_level,
                        )
                    # ── CONTINUOUS PHYSICS: ball flight time (opt-in) ────
                    # The pass and the touch that ends it are stamped at the
                    # same instant above, so a 40 m ball in behind is logged as
                    # arriving at the moment it was struck. The flight time is
                    # already known — it is in geometry_meta — but it was only
                    # ever used to recover pass DISTANCE for the miscontrol
                    # model. Nothing acted on it as time.
                    #
                    # Shifting the receiving event forward makes the timeline a
                    # recording rather than a log, and because the engine
                    # derives the match clock from its event timestamps
                    # (`match_clock_s += dur`), the clock inherits the delay
                    # for free. Defenders then have real time to close the
                    # gap before the next action.
                    _touch_minute, _touch_second = minute, getattr(state, "second", 0)
                    try:
                        from physics.adapter import active_adapter
                        _fphys = active_adapter()
                    except Exception:
                        _fphys = None
                    if _fphys is not None:
                        try:
                            # Distance computed HERE, not taken from `_pdm`.
                            # `_pdm` is bound inside the reactive-block branch
                            # above, so a completed pass that never entered that
                            # branch would reach this line with it undefined —
                            # raise NameError, get swallowed by the except
                            # below, and silently never shift the clock. The
                            # pass origin and destination are both in scope at
                            # the receiving event, so derive it from those.
                            _leg_m = math.hypot(end_px - x, end_py - y)
                            _t, _why = _fphys.pass_flight_time(
                                _leg_m, geometry_meta.get("ball_speed_mps", 0.0))
                            if _t > 0.0:
                                _adv = int(_t)
                                _touch_minute = minute + _adv // 60
                                _touch_second = int(
                                    getattr(state, "second", 0)) + _adv % 60
                                if _touch_second >= 60:
                                    _touch_second -= 60
                                    _touch_minute += 1
                        except Exception:
                            pass

                    miscontrol = random.random() < miscontrol_prob
                    if miscontrol:
                        result.add(cls.make_event(
                            _touch_minute, EventType.MISCONTROL, attacking_team, receiver.name,
                            phase, game_state,
                            second=_touch_second,
                            location_x=end_px, location_y=end_py,
                            outcome=False,
                            metadata={"from_pass": True}
                        ))
                        result.possession_lost = True
                        break
                    else:
                        result.add(cls.make_event(
                            _touch_minute, EventType.BALL_RECEIPT, attacking_team, receiver.name,
                            phase, game_state,
                            second=_touch_second,
                            location_x=end_px, location_y=end_py,
                            outcome=True,
                        ))

                    # Update position and ball carrier
                    x, y = end_px, end_py
                    last_passer = last_player.name   # the man who played it
                    last_player = receiver
                    if position_engine is not None:
                        position_engine.record_touch(receiver.name, x, y, minute)
                    episode.set_ball(x, y)

                else:
                    # Failed pass — log miscontrol/turnover
                    if interceptor is not None:
                        # Checkpoint 25 — the lane choked the pass: credit the
                        # shadowing defender with a real INTERCEPTION at the
                        # point his cone caught the ball, instead of the ball
                        # silently teleporting back to the passer as a turnover.
                        idf, ix, iy = interceptor
                        result.add(cls.make_event(
                            minute, EventType.INTERCEPTION, idf.team_name, idf.name,
                            phase, game_state,
                            secondary_player=last_player.name,
                            location_x=ix, location_y=iy,
                            outcome=True,
                            metadata={"intercepted_pass_from": last_player.name},
                        ))
                        # Genuine sweeper run-out: the goalkeeper
                        # left his line and beat the attacker to a
                        # ball deep inside his own box. A sweep is
                        # a run-out and is not counted as a shot
                        # save.
                        if getattr(idf, "position", "") == "GK":
                            _deep = (ix >= 83.0) if attacks_right else (ix <= 22.0)
                            if _deep:
                                result.add(cls.make_event(
                                    minute, EventType.SAVE,
                                    idf.team_name, idf.name,
                                    phase, game_state, outcome=True,
                                    location_x=ix, location_y=iy,
                                    metadata={"type": "gk_sweep", "runs_out": True, "contact_z": 0.0},
                                ))
                    if pass_blocker is not None:
                        # Reactive block: body in the lane, no possession won.
                        # Pass is incomplete; blocker gets a blocked pass/cross.
                        bx = x + (end_px - x) * 0.5
                        by = y + (end_py - y) * 0.5
                        result.add(cls.make_event(
                            minute, EventType.BLOCK,
                            pass_blocker.team_name, pass_blocker.name,
                            phase, game_state,
                            secondary_player=last_player.name,
                            location_x=bx, location_y=by,
                            outcome=True,
                            metadata={
                                "blocked_type": pass_block_type,
                                "blocked_pass_from": last_player.name,
                                "lane_dist_m": round(pass_block_lane, 2),
                                "physics": episode.physics_meta("pass_block"),
                            },
                        ))
                    result.add(cls.make_event(
                        minute, EventType.TURNOVER, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=False,
                    ))
                    result.possession_lost = True
                    break

            # ── 4. DUEL (contested possession, ~15% of sequences) ──
            # Ground duels happen when defender challenges carrier
            if (not result.possession_lost and under_pressure
                    and pressure_player and random.random() < 0.35):
                duel_type = EventType.AERIAL_DUEL if (
                    random.random() < 0.20 and x > 50
                ) else EventType.GROUND_DUEL

                if duel_type == EventType.GROUND_DUEL and position_engine is not None:
                    # Contact-window contest on the shared clock: the
                    # challenger must beat the carrier's escape timing into
                    # the tackle envelope (movement + tackle radius + window).
                    att_wins = episode.resolve_duel_contest(
                        cls._moving_player(last_player, position_engine),
                        cls._moving_player(pressure_player, position_engine),
                        x, y,
                    )
                    duel_physics = episode.physics_meta("ground_duel")
                elif position_engine is not None:
                    # Aerial duel: 3D trajectory + vertical reach geometry
                    aerial_height = 1.5 + random.uniform(0, 1.0)
                    aerial_speed = aerial_delivery_speed(
                        float(getattr(last_player.dna.passing, "long_passing", 55.0))
                    )
                    aerial_target_x = x + random.uniform(-3, 3)
                    aerial_target_y = y + random.uniform(-3, 3)
                    if math.hypot(aerial_target_x - x, aerial_target_y - y) < 0.5:
                        # Keep the flight's range physically launchable; a ~0 m
                        # span would hit the 45 deg max-range fallback.
                        aerial_target_x += 0.5
                    flight = make_ballistic_flight(
                        Vec3(x, y, 0.05),
                        Vec3(aerial_target_x, aerial_target_y, aerial_height),
                        aerial_speed,
                        loft=0.0,
                    )
                    aerial_attackers = [
                        cls._moving_player(last_player, position_engine)
                    ]
                    aerial_defenders = [
                        cls._moving_player(p, position_engine)
                        for p in def_players
                        if getattr(p, "position", "") != "GK"
                    ]
                    if pressure_player is not None:
                        aerial_defenders.append(cls._moving_player(pressure_player, position_engine))
                    
                    aerial_resolution = episode.resolve_aerial(flight, aerial_attackers, aerial_defenders)
                    att_wins = (
                        aerial_resolution.outcome in ("controlled", "contested")
                        and aerial_resolution.winner in aerial_attackers
                    )
                    duel_physics = episode.physics_meta("aerial_duel")
                else:
                    att_wins = random.random() < (
                        DNAFactory.get_aerial_success_rate(last_player.dna)
                    )
                    duel_physics = None

                result.add(cls.make_event(
                    minute, duel_type, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=pressure_player.name,
                    location_x=x, location_y=y,
                    outcome=att_wins,
                    metadata={"duel_type": duel_type.name,
                              "physics": duel_physics}
                ))

                # ── SECOND PRESSER / TRAP (P3) ──────────────────────
                # A second defender converging on the carrier's escape lane
                # closes the exit the primary presser leaves open.  A trap
                # profile (mid-block trap / gegenpress) running with an extra
                # defender within TRAP_RANGE_M of the carrier forces the
                # presser's win — the ball is trapped and stolen rather than
                # dribbled out of trouble.
                trap_win = False
                if is_trap_profile(def_style_key) and trap_present(
                    def_players, pressure_player, x, y, position_engine,
                ) and random.random() < TRAP_CONVERSION_PROB:
                    trap_win = True
                if trap_win:
                    # The trap steals the ball: credit a clean interception at
                    # the carrier's location by the presser who made the trap.
                    result.events[-1].outcome = False
                    result.add(cls.make_event(
                        minute, EventType.INTERCEPTION,
                        pressure_player.team_name, pressure_player.name,
                        phase, game_state,
                        secondary_player=last_player.name,
                        location_x=x, location_y=y,
                        outcome=True,
                        metadata={"trap": True},
                    ))
                    result.possession_lost = True
                    break

                if not att_wins:
                    result.possession_lost = True
                    break

            # ── 5. THROUGH BALL (final step, vision players) ───────
            # A through ball is opportunity-checked, not outcome-decided:
            # it only happens when there is a forward RUNNER past an OPEN
            # lateral channel in the defensive line (CB/RB, CB-CB, LB-CB)
            # and SPACE behind that line to run into. We first pick the
            # runner, then score the lane-through-the-line opportunity and
            # gate on it (modulated by the passer's tendency + vision). The
            # actual SUCCESS is left to physics — resolve_ground_pass races
            # the ball against the defenders AND the sweeper keeper, so a
            # through ball only completes when it genuinely bypasses everyone.
            through_zone = x > 50 if attacks_right else x < 55
            receiver = None
            tb_opportunity = 0.0
            if is_final_step and through_zone and not result.possession_lost:
                policy_through_target = active_decision.target
                if (active_decision.intent is PlayerIntent.THROUGH_BALL
                        and policy_through_target is not None
                        and any(p.name == policy_through_target.name for p in players)):
                    receiver = policy_through_target
                else:
                    receiver = cls._pick_receiver(
                        players, last_player, x, team_profile,
                        preferred_positions=["ST", "CF", "LW", "RW", "CAM"],
                        position_engine=position_engine, y=y,
                    )
                if receiver:
                    tb_opportunity = cls._through_ball_opportunity(
                        last_player, receiver, x, y, attacks_right,
                        def_players, position_engine,
                    )
                    # The policy selects whether to dare the lane.  Geometry
                    # still decides whether a lane exists, and the ensuing
                    # race resolves the outcome; a sampled THROUGH_BALL is
                    # no longer demoted to a tendency-weighted coin flip.
                    tendency = float(getattr(
                        getattr(last_player.dna, "tendencies", None),
                        "plays_through_ball", 0.10))
                    _vis = float(getattr(
                        getattr(last_player.dna, "mental", None), "vision", 60.0)
                        or 60.0)
                    skill = min(1.0, tendency * 1.2 + (_vis - 45.0) / 220.0)
                    if active_decision.intent is PlayerIntent.THROUGH_BALL:
                        attempt = 1.0 if tb_opportunity > 0.0 else 0.0
                    else:
                        # Non-through intents may still exploit an obvious
                        # lane as a secondary emergent behaviour.
                        attempt = skill * (0.35 + tb_opportunity * 0.65)
                    if random.random() >= attempt:
                        receiver = None
            if receiver:
                if position_engine is not None:
                    rx, ry = position_engine.get_position(receiver.name)
                    end_tx = cls.clamp_x(rx, attacks_right)
                    end_ty = max(5.0, min(63.0, ry))
                    tb_dist = math.hypot(end_tx - x, end_ty - y)
                    ball_speed = ground_pass_speed(
                        float(getattr(last_player.dna.passing, "short_passing", 55.0)),
                        driven=5.0
                    )
                    # ── GROUND PASS / THROUGH BALL ──
                    # Real race-to-ball: the GK may genuinely
                    # sweep a through ball off his line. He enters
                    # the race only when the ball is delivered
                    # deep inside his own box — the authentic
                    # sweeper scenario — and his live position
                    # (position_engine) is where he stands now.
                    gk = cls._pick_gk_player(def_players)
                    deep_into_box = (end_tx >= 83.0) if attacks_right else (end_tx <= 22.0)
                    tb_defenders = [
                        cls._moving_player(d, position_engine)
                        for d in def_players
                        if getattr(d, "position", "") != "GK"
                    ]
                    if gk is not None and deep_into_box:
                        tb_defenders.append(cls._moving_player(gk, position_engine))
                    tb_resolution = episode.resolve_ground_pass(
                        Vec2(x, y), Vec2(end_tx, end_ty),
                        cls._moving_player(receiver, position_engine),
                        tb_defenders,
                        ball_speed,
                        rolling_decel=rolling_decel_for(
                            getattr(state, "weather", None)
                        ),
                    )
                    tb_success = tb_resolution.outcome == "received"
                else:
                    tb_success, end_tx, end_ty, tb_dist, tb_prob = cls._generate_through_ball(
                        last_player, receiver, x, y,
                        minute, phase, game_state, attacks_right, team_profile,
                    )
                result.add(cls.make_event(
                    minute, EventType.THROUGH_BALL, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=receiver.name,
                    location_x=x, location_y=y,
                    end_x=end_tx, end_y=end_ty,
                    outcome=tb_success,
                    metadata={"decision_authority": "player_policy" if active_decision.intent is PlayerIntent.THROUGH_BALL else "emergent_opportunity",
                              "distance": round(tb_dist, 1),
                              "pass_type": "through ball",
                              "body_part": cls._foot_for_pass(
                                  last_player, x, y, end_tx, end_ty, attacks_right)}
                ))
                if tb_success:
                    result.add(cls.make_event(
                        minute, EventType.BALL_RECEIPT, attacking_team, receiver.name,
                        phase, game_state,
                        location_x=end_tx, location_y=end_ty,
                        outcome=True,
                    ))
                    last_passer = last_player.name   # the man who played it
                    last_player = receiver
                    x, y = end_tx, end_ty
                    if position_engine is not None:
                        position_engine.record_touch(receiver.name, x, y, minute)
                else:
                    if position_engine is not None and tb_resolution.interceptor is not None:
                        idf = tb_resolution.interceptor.player
                        result.add(cls.make_event(
                            minute, EventType.INTERCEPTION,
                            getattr(idf, "team_name", "") or getattr(idf, "team", ""),
                            getattr(idf, "name", ""),
                            phase, game_state,
                            secondary_player=last_player.name,
                            location_x=tb_resolution.contact_point.x,
                            location_y=tb_resolution.contact_point.y,
                            outcome=True,
                            metadata={"intercepted_pass_from": last_player.name},
                        ))
                        # A genuine sweeper run-out: the goalkeeper
                        # left his line and beat the attacker to a
                        # through ball deep in his own box. Recorded
                        # as a SAVE with type "gk_sweep" so the
                        # exporter counts it as runs_out and only
                        # runs_out.
                        if getattr(idf, "position", "") == "GK":
                            result.add(cls.make_event(
                                minute, EventType.SAVE,
                                getattr(idf, "team_name", ""), idf.name,
                                phase, game_state, outcome=True,
                                location_x=tb_resolution.contact_point.x,
                                location_y=tb_resolution.contact_point.y,
                                metadata={
                                    "type": "gk_sweep",
                                    "runs_out": True,
                                    "contact_z": 0.0,
                                },
                            ))
                    result.add(cls.make_event(
                        minute, EventType.TURNOVER, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=False,
                    ))
                    result.possession_lost = True
                    break

            # ── 6. DRIBBLE (wide players, attacking third) ─────────
            # STRICT OPTA/STATSBOMB DEFINITION (Checkpoint refinement):
            # A dribble completed ONLY occurs when an attacking player actively
            # bypasses an ENGAGED defender using technical skill, pace, or feint
            # while maintaining control. Simply running into open space does NOT count.
            # 
            # Key changes from previous version:
            # 1. DEFENDER MUST BE NEARBY (within tackling radius ~3-5m)
            # 2. Dribble attempt only triggers if defender is actively engaged
            # 3. "Heavy touch" failure mode — pushing ball too far = unsuccessful
            # 4. Lower base attempt rate when defenders aren't pressing
            dribble_zone = x > 45 if attacks_right else x < 60
            if (not result.possession_lost and dribble_zone
                    and last_player.position in ("LW", "RW", "CAM", "ST", "CF")):
                
                # Find NEAREST defender (not just weighted random)
                nearest_defender = None
                min_dist = float('inf')
                if def_players and position_engine is not None:
                    for defender in def_players:
                        def_pos = position_engine.get_position(defender.name)
                        if def_pos:
                            dx = def_pos[0] - x
                            dy = def_pos[1] - y
                            dist = (dx**2 + dy**2) ** 0.5
                            if dist < min_dist:
                                min_dist = dist
                                nearest_defender = defender
                
                # CRITICAL: Only attempt dribble if defender is ENGAGED (within ~5m tackling radius)
                defender_engaged = min_dist < 5.0 if nearest_defender else False
                
                # Dribble attempt rate depends on defender proximity and player tendency
                base_attempt_rate = last_player.dna.tendencies.attempts_dribble
                if defender_engaged:
                    # Defender pressing → higher dribble attempt rate (skill expression)
                    attempt_rate = base_attempt_rate * 1.2
                else:
                    # Open space → much lower rate (most are just carries, not dribbles)
                    attempt_rate = base_attempt_rate * 0.15

                # ── CHECKPOINT 18: MODERN WINGER 1v1 ISOLATION ──────
                # Modern wingers (Vini Jr, Saka, Doku) are told to attack the
                # fullback 1v1 on the flank — the touchline→byline corridor is
                # their runway. When a winger is isolated against the opposing
                # fullback, their dribble attempt rate spikes dramatically.
                # This is the "isolation thirst" — the winger's DNA tendency
                # to take on the fullback when the geometry says the 1v1 is on.
                if last_player.position in ("LW", "RW") and position_engine is not None:
                    winger_profile = position_engine.winger_registry.get(last_player.name)
                    if winger_profile is not None:
                        isolated, fb, fb_dist = winger_profile.fullback_isolation(
                            x, y, def_players, position_engine, attacks_right
                        )
                        if isolated:
                            # Isolated 1v1 → the winger attacks the fullback
                            # with their isolation thirst driving the attempt
                            attempt_rate = max(
                                attempt_rate,
                                last_player.dna.tendencies.attacks_fullback_1v1 * 0.85
                            )
                            # Checkpoint 24 — an isolated fullback standing
                            # 5-8m off IS the engagement a touchline winger
                            # attacks (he doesn't wait to be grabbed). Doku
                            # attempts ~10 take-ons per 90; requiring a
                            # defender inside 5m starved attempts to ~1-5.
                            if fb is not None and fb_dist < 7.0:
                                # Checkpoint 32 — the defending fullback's own
                                # 1v1 decision. A cover-aware FB told to JOCKEY
                                # holds the line and does NOT become the engaged
                                # defender on a 5-8m "isolation" — the winger
                                # doesn't auto-attempt into a standing defender.
                                # An engaging FB keeps the existing aggressive
                                # contact behaviour. A truly-close (<5m) winger
                                # is contested either way.
                                fb_force_engage = True
                                if position_engine is not None:
                                    fb_profile = position_engine.fullback_registry.get(fb.name)
                                    if fb_profile is not None:
                                        try:
                                            _wp = getattr(getattr(last_player.dna, "physical", None), "pace", 60.0)
                                            _fp = getattr(getattr(fb.dna, "physical", None), "pace", 60.0)
                                            decision = FullbackBehaviorEngine.defend_engagement(
                                                fb_profile, x, y, [last_player], def_players,
                                                position_engine, attacks_right,
                                                winger_pace_advantage=(_wp > _fp),
                                            )
                                            if decision == "jockey" and fb_dist >= 5.0:
                                                fb_force_engage = False
                                        except Exception:
                                            fb_force_engage = True
                                if fb_force_engage:
                                    defender_engaged = True
                                    if nearest_defender is None or fb_dist < min_dist:
                                        nearest_defender = fb
                                        min_dist = fb_dist
                                    attempt_rate = max(
                                        attempt_rate,
                                        last_player.dna.tendencies.attacks_fullback_1v1
                                        * (0.6 + 0.5 * winger_profile.isolation_thirst)
                                    )
                
                if random.random() < attempt_rate and defender_engaged:
                    # Dribble vs nearest engaged defender
                    marker = nearest_defender
                    # The dribble target remains a tactical decision, but the
                    # defender's response is resolved by reach and tackle
                    # radius over the actual movement window.
                    drb_adv = 3.0 + last_player.dna.technical.dribbling / 20.0
                    end_drb_x = cls.clamp_x(x + (drb_adv if attacks_right else -drb_adv), attacks_right)
                    if last_player.position in ("LW", "RW"):
                        end_drb_x = min(end_drb_x, 97.0) if attacks_right else max(end_drb_x, 8.0)
                    _d_mode, _d_anchor, _d_bias = cls._winger_carry_steering(
                        last_player, x, y, attacks_right, under_pressure,
                        def_players, position_engine,
                    )
                    if _d_mode == "byline":
                        end_drb_y = y + (_d_anchor - y) * 0.35
                    elif _d_mode == "cut_inside":
                        _cut = _d_anchor + (1.0 if _d_anchor < 34.0 else -1.0) * 10.0
                        end_drb_y = y + (_cut - y) * 0.30
                    elif _d_anchor is not None:
                        end_drb_y = y + _d_bias
                    else:
                        end_drb_y = y
                    end_drb_y = max(5, min(63, end_drb_y))

                    if position_engine is not None:
                        dribble_resolution = episode.resolve_dribble(
                            Vec2(x, y), Vec2(end_drb_x, end_drb_y),
                            cls._moving_player(last_player, position_engine),
                            [
                                cls._moving_player(defender, position_engine)
                                for defender in def_players
                                if getattr(defender, "position", "") != "GK"
                            ],
                        )
                        _drb_out = cls._dribble_confirmation_gate(
                            dribble_resolution.outcome, last_player, marker
                        )
                        drb_success = _drb_out == "retained"
                        if not drb_success:
                            end_drb_x = dribble_resolution.contact_point.x
                            end_drb_y = dribble_resolution.contact_point.y
                    else:
                        drb_success = random.random() < DNAFactory.get_dribble_success_rate(last_player.dna)

                    if drb_success:
                        # Retained: the carrier reaches the selected target.
                        pass
                        
                        result.add(cls.make_event(
                            minute,
                            EventType.DRIBBLE_SUCCESS,
                            attacking_team, last_player.name,
                            phase, game_state,
                            secondary_player=marker.name if marker else None,
                            location_x=x, location_y=y,
                            end_x=end_drb_x, end_y=end_drb_y,
                            outcome=True,
                            metadata={
                                "dribbled_past": True,
                                "defender_distance": round(min_dist, 2),
                                "beat_defender": marker.name if marker else "unknown",
                                "resolution": "contact_window" if position_engine is not None else "legacy_dribble",
                                "physics": episode.physics_meta("dribble") if position_engine is not None else None,
                            },
                        ))
                        x, y = end_drb_x, end_drb_y
                        if position_engine is not None:
                            position_engine.record_touch(last_player.name, x, y, minute)
                        episode.set_ball(x, y)
                    else:
                        # Geometry places the turnover at the tackle contact.
                        fail_reason = "tackled"
                        
                        result.add(cls.make_event(
                            minute,
                            EventType.DRIBBLE_FAIL,
                            attacking_team, last_player.name,
                            phase, game_state,
                            secondary_player=marker.name if marker else None,
                            location_x=x, location_y=y,
                            end_x=end_drb_x, end_y=end_drb_y,
                            outcome=False,
                            metadata={
                                "failure_reason": fail_reason,
                                "defender_distance": round(min_dist, 2) if marker else None,
                                "resolution": "contact_window" if position_engine is not None else "legacy_dribble",
                                "physics": episode.physics_meta("dribble") if position_engine is not None else None,
                            },
                        ))
                        result.possession_lost = True

            # ── 7. CROSS (wide players near byline) ────────────────
            # Checkpoint 24 — real wingers deliver 2-6 crosses per 90, not
            # 12-35. The trigger zone is the byline corridor (x>80), and the
            # per-touch delivery roll is an order of magnitude lower: a
            # winger's default in the wide final third is to COMBINE (short
            # lateral/cutback) or carry — the cross is the exception, driven
            # by the winger's own cross_instinct via should_cross().
            cross_zone = x > 80 if attacks_right else x < 25
            cross_prob = 0.07  # default cross probability
            # ── CHECKPOINT 18: MODERN WINGER CROSS TIMING ──────────
            # Modern wingers (Saka, Vini, Salah) deliver from the dangerous
            # wide crossing zone — the touchline→byline corridor. Their cross
            # instinct (from DNA) drives WHEN they deliver: a crosser whips
            # it in early from the crossing zone, while an inverted winger
            # carries on toward the byline before cutting back. The winger
            # behavior engine reads the geometry and the player's profile to
            # decide if this is the right moment to deliver.
            if last_player.position in ("LW", "RW") and position_engine is not None:
                winger_profile = position_engine.winger_registry.get(last_player.name)
                if winger_profile is not None:
                    if WingerBehaviorEngine.should_cross(
                        winger_profile, x, y, attacks_right, under_pressure
                    ):
                        cross_prob = 0.25  # winger instinct says deliver now
                    else:
                        cross_prob = 0.04  # winger carries on instead
            # A policy CROSS is an attempted delivery when geometry permits;
            # role behaviour is a fallback generator for players who did not
            # select it, not a veto over the selected intent.
            policy_cross = active_decision.intent is PlayerIntent.CROSS
            if (not result.possession_lost and cross_zone
                    and last_player.position in ("LW", "RW", "LB", "RB")
                    and (policy_cross or random.random() < cross_prob)):
                cross_skill = last_player.dna.technical.crossing / 100.0
                zone, end_tx, end_ty, raw_tx, raw_ty = cls._generate_cross_destination(
                    x, y, attacks_right, cross_skill
                )
                # Checkpoint 11 — even the engine's OWN cross decision is
                # validated by the pure geometric detector (origin wide +
                # destination in/flashing through the box). A "cross" whipped
                # from a central position is stamped `cross: false` exactly
                # as Opta/StatsBomb would refuse to tag it.
                _cr = detect_cross(x, y, end_tx, end_ty, attacks_right,
                                   event_type="CROSS_ATTEMPT")
                
                # ── GEOMETRY-FIRST CROSS RESOLUTION ─────────────────
                # The cross trajectory is a 3D aerial flight; the outcome is
                # decided by who reaches the contact point first (vertical
                # reach + horizontal speed + timing), NOT a success roll.
                cross_success = False
                cross_resolution = None
                cross_origin_x, cross_origin_y = x, y
                gk = cls._pick_gk_player(def_players)
                # ── WHO GETS THE BALL ────────────────────────────────
                # The receiver was picked by ABILITY ALONE from the whole
                # squad (`_pick_aerial_threat` weights DNA jumping + heading
                # and knows nothing about geometry), and then three lines
                # later his real position was rewritten:
                #     rx, ry = get_position(receiver)
                #     rx = clamp_attack_x(rx + uniform(1,4), 85, 100, …)
                #     ry = max(22, min(46, ry))
                #     record_touch(receiver, rx, ry, minute)
                # That is two fabrications in three lines. He was often 30-50 m
                # upfield; the clamp teleported him into the box AND banked the
                # gap as distance covered (77 record_touch jumps / 2,766 m
                # never walked, over two matches), and because `_moving_player`
                # reads the position engine he was then scored as ALREADY
                # there — so `resolve_aerial_delivery` charged him no movement
                # cost and he won the header for free. A man who never ran into
                # the box got the ball because he was moved there on paper.
                #
                # Now: picked by ability × PLAUSIBILITY. `pick_weighted_spatial`
                # already existed for exactly this — "multiplies the
                # label-based weight by the player's real-time positional
                # plausibility for an action happening at (at_x, at_y)". He
                # then stays exactly where he is and the resolver charges him
                # the real distance, so a header won is one actually contested.
                cross_receiver = cls.pick_weighted_spatial(
                    players,
                    lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2,
                    position_engine, end_tx, end_ty,
                    exclude=last_player.name, spatial_exponent=1.5,
                )
                cross_defender = cls._pick_aerial_defender(def_players)
                # The AIM, in the box. Aiming a cross inside the box is correct
                # football, so it stays — it is only ever a target now, and it
                # belongs to nobody's tracked position.
                rx = max(84.0, min(102.0, end_tx))
                ry = max(24.0, min(44.0, end_ty))
                end_tx, end_ty = rx, ry
                # NOTE: the old defender placement
                #     dx2, dy2 = get_position(defender)
                #     dy2 = max(24, min(44, dy2 + uniform(-2,2)))
                #     record_touch(defender, dx2, dy2, minute)
                # is GONE, and it went for a reason worth stating: `dx2` and
                # `dy2` were WRITE-ONLY. Nothing after that block ever read
                # them (grep: five occurrences, all assignments) — the target
                # was derived from `rx`/`ry`, not from the defender. So its
                # entire effect was to teleport a defender — up to 14 m in y
                # for one standing wide — and bank the gap as distance he
                # covered. A fabrication with no consumer, which is exactly
                # why it survived so long: it read like it was placing a
                # marker.
                if position_engine is not None:
                    cross_height = 1.2 + cross_skill * 1.2
                    cross_speed = aerial_delivery_speed(
                        float(getattr(last_player.dna.passing, "long_passing", 55.0))
                    )
                    # Whipped delivery: the winger's crossing quality spins the
                    # ball into the corridor in front of the runner (side-spin
                    # Magnus bend). Aimed at the landing point, so the curl is
                    # visible mid-flight but the resolver's landing stays exact.
                    cross_spin = delivery_spin(
                        float(getattr(last_player.dna.passing, "crossing", 55.0)),
                        kind="side",
                    )
                    flight = make_ballistic_flight(
                        Vec3(x, y, 0.05),
                        Vec3(end_tx, end_ty, cross_height),
                        cross_speed,
                        loft=0.0,
                        spin=cross_spin,
                    )
                    cross_attackers = [
                        cls._moving_player(cross_receiver, position_engine)
                    ] if cross_receiver else []
                    cross_defenders = []
                    if cross_defender:
                        cross_defenders.append(cls._moving_player(cross_defender, position_engine))
                    if gk is not None:
                        cross_defenders.append(cls._moving_player(gk, position_engine))
                    episode.register(cross_attackers + cross_defenders)
                    cross_resolution = episode.resolve_aerial(flight, cross_attackers, cross_defenders)
                    
                    if cross_resolution.outcome in ("controlled", "contested"):
                        winner_is_attacker = cross_resolution.winner in cross_attackers
                        cross_success = winner_is_attacker
                        if cross_resolution.winner is not None:
                            winner_profile = cross_resolution.winner.player
                            winner_name = getattr(winner_profile, "name", str(winner_profile))
                            episode.update_player_position(
                                winner_name,
                                cross_resolution.contact_point.x,
                                cross_resolution.contact_point.y,
                            )
                        if cross_success:
                            x, y = cross_resolution.contact_point.x, cross_resolution.contact_point.y
                            episode.set_ball(x, y)
                            if position_engine is not None and cross_resolution.winner is not None:
                                position_engine.record_touch(winner_name, x, y, minute)
                            # The cross TAKER gets the assist on a headed goal,
                            # which is what the Laws award and what a human
                            # would write down — so the passer of record is the
                            # man who crossed, not the man who won the header.
                            if getattr(cross_resolution.winner, "player", None) is not None:
                                last_passer = last_player.name
                            last_player = next(
                                (p for p in players if getattr(p, "name", "") == winner_name),
                                last_player,
                            )
                        elif gk is not None and cross_resolution.winner is not None and getattr(cross_resolution.winner.player, "name", "") == gk.name:
                            # ── GK WINS THE HIGH BALL → CLAIM / PUNCH ────
                            # The keeper genuinely arrived and won the aerial
                            # race on an open-play cross. That is a real
                            # high-ball take: he decides to claim it cleanly
                            # or punch it clear, exactly like the corner path.
                            cpx = max(83.0, min(104.5, cross_resolution.contact_point.x)) if attacks_right else max(0.5, min(22.0, cross_resolution.contact_point.x))
                            cpy = max(24.0, min(44.0, cross_resolution.contact_point.y))
                            contact_z = cross_resolution.contact_point.z
                            contested = cross_resolution.outcome == "contested"
                            challenger_present = cross_resolution.challenger is not None
                            gk_action, claim_height = GoalkeeperEngine.decide_high_ball(
                                gk, contact_z, contested, challenger_present,
                            )
                            # A run-out is a genuine sweep: the keeper left his
                            # goal-mouth and beat an attacker to a LIVE open-play
                            # cross. The win point IS the evidence — beyond the
                            # six-yard box depth (5.5m) the keeper physically
                            # committed to claim/punch it ahead of his own line,
                            # i.e. a genuine sweep-keeper exit.
                            goal_line_x = 105.0 if attacks_right else 0.0
                            runs_out = is_goalkeeper_run_out(goal_line_x, cpx)
                            event_type_md = "gk_punch" if gk_action == "punch" else "gk_claim"
                            result.add(cls.make_event(
                                minute, EventType.SAVE, defending_players[0].team_name if defending_players else attacking_team,
                                gk.name, phase, game_state, outcome=True,
                                location_x=cpx, location_y=cpy,
                                metadata={
                                    "type": event_type_md,
                                    "claim_height": claim_height,
                                    "contested": contested,
                                    "runs_out": runs_out,
                                    "contact_z": round(contact_z, 2),
                                },
                            ))
                            x, y = cpx, cpy
                            episode.set_ball(x, y)
                            result.possession_lost = True
                        else:
                            cx = cross_resolution.contact_point.x
                            cy = cross_resolution.contact_point.y
                            clearance_winner = cross_resolution.winner
                            clearance_def_team = defending_players[0].team_name if defending_players else None
                            clearance_att_team = attacking_team
                            clearance_team = (
                                clearance_def_team if clearance_winner in cross_defenders else clearance_att_team
                            )
                            # Swap attacker/defender player lists for DefensiveChain
                            # depending on which team won the aerial duel.
                            if clearance_team == clearance_def_team:
                                chain_def_players = defending_players
                                chain_att_players = players
                            else:
                                chain_def_players = players
                                chain_att_players = defending_players
                            clearance_result = DefensiveChain.generate(
                                minute, clearance_team,
                                clearance_att_team if clearance_team == clearance_def_team else clearance_def_team,
                                chain_def_players, chain_att_players,
                                state, action_type="clearance",
                                context_x=cx, context_y=cy,
                                attacks_right=attacks_right,
                                danger_level=state.threat.danger_at(clearance_team) if hasattr(state, 'threat') else 50.0,
                                ball_aerial=True,
                                own_goal_x=105.0 if attacks_right else 0.0,
                                position_engine=position_engine,
                                ball_z=cross_resolution.contact_point.z,
                                defender_facing_x=cx, defender_facing_y=cy,
                                opponent_distance=0.5,
                            )
                            for e in clearance_result.events:
                                result.add(e)
                            cls._inherit_restart(clearance_result, result)
                            x, y = cx, cy
                            episode.set_ball(x, y)
                            result.possession_lost = True
                    else:
                        x, y = cross_resolution.contact_point.x, cross_resolution.contact_point.y
                        episode.set_ball(x, y)
                        result.possession_lost = True
                else:
                    cross_success = random.random() < (0.30 + cross_skill * 0.35)
                
                result.add(cls.make_event(
                    minute, EventType.CROSS_ATTEMPT, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=(
                        winner_name
                        if cross_success and cross_resolution is not None
                        and cross_resolution.winner is not None
                        else None
                    ),
                    location_x=cross_origin_x, location_y=cross_origin_y,
                    end_x=end_tx, end_y=end_ty,
                    outcome=cross_success,
                    metadata={
                        "decision_authority": "player_policy" if policy_cross else "role_fallback",
                        "open_play": True,
                        "target_zone": zone,
                        "cross_skill": round(cross_skill, 3),
                        "raw_target_x": round(raw_tx, 1),
                        "raw_target_y": round(raw_ty, 1),
                        "body_part": cls._foot_for_pass(
                            last_player, x, y, end_tx, end_ty, attacks_right),
                        "cross": _cr.is_cross,
                        "is_airborne": _cr.airborne,
                        "cross_origin": _cr.origin_zone,
                        "cross_dest": _cr.destination_zone,
                        "resolution": "aerial_trajectory" if position_engine is not None else "legacy_roll",
                    }
                ))
                if cross_success:
                    result.add(cls.make_event(
                        minute, EventType.CROSS_SUCCESS, attacking_team, last_player.name,
                        phase, game_state,
                        secondary_player=(
                            winner_name
                            if cross_resolution is not None
                            and cross_resolution.winner is not None
                            else None
                        ),
                        location_x=x, location_y=y,
                        outcome=True,
                        metadata={"open_play": True}
                    ))
                elif not cross_success and result.possession_lost and cross_resolution is not None:
                    if cross_resolution.winner is not None:
                        winner_profile = cross_resolution.winner.player
                        is_defender = cross_resolution.winner in cross_defenders
                        clearance_def_team = defending_players[0].team_name if defending_players else None
                        clearance_team = clearance_def_team if is_defender else attacking_team
                        result.add(cls.make_event(
                            minute, EventType.CLEARANCE, clearance_team, winner_profile.name,
                            phase, game_state,
                            location_x=x, location_y=y,
                            outcome=True,
                            metadata={"from_cross": True, "resolution": "aerial_trajectory",
                                      "cross_skill": round(cross_skill, 3),
                                      "defender_clearance": is_defender}
                        ))
                        if is_defender:
                            # Cross block: delivery into the box stopped before
                            # reaching its target (aerial physics already won
                            # by the defender above). Counts as a block too.
                            result.add(cls.make_event(
                                minute, EventType.BLOCK,
                                clearance_team, winner_profile.name,
                                phase, game_state,
                                secondary_player=last_player.name,
                                location_x=x, location_y=y,
                                outcome=True,
                                metadata={
                                    "blocked_type": "cross",
                                    "blocked_cross_from": last_player.name,
                                    "from_cross": True,
                                    "physics": episode.physics_meta("cross_block"),
                                },
                            ))
                    result.add(cls.make_event(
                        minute, EventType.TURNOVER, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=False,
                    ))

        # ── OUT-OF-BOUNDS DETECTION (Checkpoint 7) ────────────────
        # Check if ball went out of bounds and emit appropriate restart
        if not result.possession_lost and not result.corner_won:
            # Throw-in detection: (y < 2 or y > 66) AND x < 105
            if (y < 2.0 or y > 66.0) and x < 105.0:
                result.restart_required = True
                result.restart_type = "throw_in"
                # Award to team that DIDN'T touch last (opposing team)
                # In PossessionChain, attacking_team had last touch
                result.restart_team = ""  # Will be determined by MatchEngine
                result.restart_x = x
                result.restart_y = 0.0 if y < 2.0 else 68.0
                result.possession_lost = True  # Ball is out, possession ends
                
            # Goal kick detection: x ≥ 105 AND (y < 30.34 or y > 37.66)
            # This means ball crossed goal line but not between posts
            elif x >= 105.0 and (y < 30.34 or y > 37.66):
                result.restart_required = True
                result.restart_type = "goal_kick"
                # Award to defending team (opposite of attacking_team)
                result.restart_team = ""  # Will be determined by MatchEngine
                result.restart_x = random.uniform(8, 18)  # GK position
                result.restart_y = 34.0  # Center of goal area
                result.possession_lost = True  # Ball is out, possession ends

        # ── PHYSICS TRACE STAMP (Checkpoint 27) ─────────────────────
        # The continuous 0.1 s motion trace is attached to the final event of
        # the sequence so exporter/analytics consumers can recover the whole
        # episode's movement without changing the event schema.
        if episode.trace and result.events:
            last_evt = result.events[-1]
            if last_evt.metadata.get("physics") is None:
                last_evt.metadata["physics"] = episode.physics_meta("possession_episode")
            last_evt.metadata["motion_trace"] = episode.condensed_trace()
            last_evt.metadata["ball_motion"] = episode.ball_path

        # ── POSSESSION-TIME (real minutes held, not a probability) ─────
        # episode.elapsed is the 0.1 s clock advanced continuously across every
        # pass, carry, dribble and duel in this sequence — the team's actual
        # time on the ball. Split it across the players with an on-ball action
        # (carry/pass/dribble) for per-player possession minutes.
        result.sequence_duration_s = round(episode.elapsed, 2)
        result.ball_path = episode.ball_path
        # Real per-player distance / sprint accounting from the episode's 10 Hz
        # trace (actual physics integration). This is what lets the unified
        # timeline report physically-measured distance instead of a snapshot.
        try:
            result.player_distance_stats = dict(episode.calculate_distance_stats())
        except Exception:
            result.player_distance_stats = {}
        _touch_types = (EventType.CARRY, EventType.PASS, EventType.PROGRESSIVE_PASS,
                        EventType.SWITCH_OF_PLAY)
        _touch_counts: Dict[str, int] = {}
        for _ev in result.events:
            if _ev.event_type in _touch_types:
                _touch_counts[_ev.player] = _touch_counts.get(_ev.player, 0) + 1
        _total_touches = sum(_touch_counts.values())
        if _total_touches > 0:
            _per = result.sequence_duration_s / _total_touches
            for _p, _c in _touch_counts.items():
                result.player_possession_s[_p] = round(_c * _per, 2)

        # ── HAND OFF THE BALL CARRIER ───────────────────────────────────────
        # Only while the attacking team still HAS the ball. A possession that
        # ended in a turnover, a throw-in or a goal kick has no carrier to
        # name, and naming one anyway would put a player's name on a shot taken
        # from a ball he is not touching — the exact defect this whole chain of
        # work exists to remove.
        #
        # `last_passer` is the passer OF RECORD for `last_player`; if they are
        # the same man the ball came from a turnover we do not describe, so the
        # field stays empty and the strike is honestly unassisted.
        if not result.possession_lost and last_player is not None:
            result.ball_carrier = getattr(last_player, "name", "") or ""
            result.ball_carrier_passed_by = (
                "" if result.ball_carrier == last_passer else last_passer
            )

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)
        return result

    # ── HELPERS ───────────────────────────────────────────────

    @classmethod
    def _starting_position(cls, profile, state=None) -> Tuple[float, float]:
        """
        Checkpoint 7 -- where does this team typically start sequences?

        Previously this drew a fresh, independent random zone from team
        style ALONE, every single sequence -- the ball had no memory of
        where it actually was, so it "teleported" an average of ~33m
        between sequences (measured on a real match run: 28% of sequence
        starts jumped >40m from where the ball last was, max jump 91m on
        a 105m pitch). That's how a striker could get selected as a
        possession "builder" deep in his own box purely because a random
        draw happened to land there.

        Now the team's style-typical zone is a SECONDARY nudge on top of
        a PRIMARY anchor: state.last_ball_x/y -- the last real location
        the ball was actually seen at, kept truthful event-by-event in
        MatchEngine._absorb_chain. Continuity is the dominant signal (a
        sequence starts close to where the ball last was), while style
        still shapes build-up tendency on top (a park-the-bus side still
        generally settles deeper even from the same recovery point,
        exactly like a real low block retreating to reorganize rather
        than instantly pushing out).

        Falls back to the old pure style-random behavior when no state
        is supplied (e.g. a standalone/test call) -- nothing breaks.
        
        Accepts both TeamProfile (has .style) and EffectiveTactics (no .style).
        For EffectiveTactics, use defensive_line as a proxy for style depth.
        """
        from match_engine import TeamStyle
        style_x_range = (15, 45)
        # Handle both TeamProfile and EffectiveTactics
        if hasattr(profile, 'style'):
            if profile.style in (TeamStyle.PARK_THE_BUS, TeamStyle.ULTRA_DEFENSIVE):
                style_x_range = (5, 30)
            elif profile.style in (TeamStyle.TIKI_TAKA, TeamStyle.STRUCTURED_POSSESSION):
                style_x_range = (20, 50)
            elif profile.style in (TeamStyle.FLUID_COUNTER, TeamStyle.ROUTE_ONE):
                style_x_range = (10, 40)
        elif hasattr(profile, 'defensive_line'):
            # EffectiveTactics: use defensive_line as proxy for style depth
            if profile.defensive_line < 0.25:
                style_x_range = (5, 30)   # Very defensive
            elif profile.defensive_line < 0.45:
                style_x_range = (10, 40)  # Defensive
            elif profile.defensive_line > 0.65:
                style_x_range = (20, 50)  # Attacking/possession
            # else: default (15, 45)
        style_x = random.uniform(*style_x_range)
        style_y = random.uniform(5, 63)

        last_x = getattr(state, "last_ball_x", None) if state is not None else None
        last_y = getattr(state, "last_ball_y", None) if state is not None else None
        if last_x is None or last_y is None:
            return round(style_x, 1), round(style_y, 1)

        # 75/25 blend: continuity dominates, style nudges. Then jitter --
        # a sequence doesn't restart from the EXACT same coordinate every
        # time; the loose ball settles a few meters off as players jostle.
        anchor_x = last_x * 0.75 + style_x * 0.25
        anchor_y = last_y * 0.75 + style_y * 0.25
        x = max(2.0, min(103.0, anchor_x + random.uniform(-6, 6)))
        y = max(2.0, min(66.0, anchor_y + random.uniform(-8, 8)))
        return round(x, 1), round(y, 1)

    @classmethod
    def _pick_builder(
        cls, players: List[PlayerProfile],
        position_engine: Optional[PositionEngine] = None,
        x: float = 25.0, y: float = 34.0,
    ) -> Optional[PlayerProfile]:
        outfield_preferred = ["CB", "CDM", "CM", "LB", "RB"]

        # Throttle GK options: exclude GK from builder selection so keepers
        # do not dominate deep build-up touches. Modern build-up starts with
        # CBs/CDMs; the keeper remains the phase-engine safety valve, not the
        # routine first touch.
        candidates = [p for p in players if p.position != "GK"]

        def label_weight(p: PlayerProfile) -> float:
            return 3.0 if p.position in outfield_preferred else 0.8

        return cls.pick_weighted_spatial(
            candidates, label_weight, position_engine, x, y,
            spatial_exponent=2.0,
        )

    # ── TACTICAL POSSESSION PHASES (Checkpoint 14/15) ─────────────
    @classmethod
    def _tactical_phase_step(
        cls,
        carrier: PlayerProfile,
        players: List[PlayerProfile],
        def_players: List[PlayerProfile],
        x: float,
        y: float,
        current_phase: PossessionPhase,
        under_pressure: bool,
        attacks_right: bool,
        team_profile: "TeamProfile",
        position_engine: PositionEngine,
        att_style_key: Optional[str] = None,
        def_style_key: Optional[str] = None,
        def_press_intensity: Optional[float] = None,
        counterpress: Optional[Dict[str, Any]] = None,
        state: Optional[MatchState] = None,
    ) -> Tuple[PossessionPhase, Optional[PossessionDecision]]:
        """
        Run the possession-phase engine for this touch and return
        (new_phase, decision).

        Teammate snapshots are built from live PositionEngine state: each
        outfield teammate's marking tightness is measured, and the keeper is
        wrapped as the engine's GK overload anchor so deep restarts under
        pressure trigger the RELEASE→GK regress chain.

        Checkpoint 15: the 30° cover-shadow geometry of the defending
        profile now gates the corridors. A teammate whose lane runs through
        a defender's cover-shadow cone is NOT a passing option — that is
        what chokes the forward lanes and fires the GK Emergency Phase
        Regression against a high press. The keeper's own lane gate is the
        same combined clearance, so a gegenpress that stands a man on the
        keeper still cuts the safety valve open only when the cone geometry
        actually allows the pass.
        """
        def_style_key = def_style_key or "balanced"
        def_press_i = (def_press_intensity if def_press_intensity is not None
                       else getattr(team_profile, "press_intensity", 0.5))
        press_profile = resolve_profile(def_press_i, def_style_key)
        engaged = engagement_allows(
            x if attacks_right else (105.0 - x), press_profile
        )

        # ── COUNTERPRESS BURST OVERRIDE (P2) ─────────────────────────
        # When a team has just lost possession, it presses the recovery
        # zone for ~8 s, overriding the normal static engagement line
        # within COUNTERPRESS_RANGE_M of the ball-lost location.  This
        # keeps the cover-shadow geometry active and forward lanes choked
        # even when the ball is in the team's own half — exactly what a
        # real counterpress does in the first seconds after a turnover.
        cp_blocked_lanes = False
        if counterpress and counterpress.get("active") and state is not None:
            cp_dist = math.hypot(x - counterpress["x"], y - counterpress["y"])
            if cp_dist <= state.COUNTERPRESS_RANGE_M:
                engaged = True
                cp_blocked_lanes = True

        teammates = []
        for p in players:
            if p.name == carrier.name or p.position == "GK":
                continue
            tx, ty = position_engine.get_position(p.name)
            marking = cls._marking_tightness(
                p, x, y, def_players, position_engine, attacks_right
            )
            lane_blocked = cover_shadow_blocked(
                x, y, tx, ty, def_players, position_engine,
                press_profile, engaged=engaged,
            )
            teammates.append(TeammateSnapshot(p.name, p.position, tx, ty, marking,
                                              lane_blocked=lane_blocked))

        gk = next((p for p in players if p.position == "GK"), None)
        if gk is None:
            return current_phase, None
        gx, gy = position_engine.get_position(gk.name)
        # A back-pass to the keeper is a short, central reset — it needs
        # only a half-clear corridor, not the fully-clean lane a forward
        # pass demands. 0.3 (vs the usual 0.5) keeps the gate meaningful
        # without strangling routine build-up recirculation.
        lane_open = cover_shadow_clearance(
            x, y, gx, gy, def_players, position_engine,
            press_profile, engaged=engaged,
        ) >= COVER_SHADOW_BLOCK_THRESHOLD
        gk_snap = GKSnapshot(gk.name, gx, gy, lane_open)

        style_key = att_style_key or getattr(
            getattr(team_profile, "style", None), "value", "balanced"
        )
        # Checkpoint 23 — carrier IQ (vision-weighted) modulates how often the
        # tempo-circulation roll fires: elite readers of the game sustain a
        # touch less because they spot the vertical ball earlier.
        carrier_iq = 0.70
        carrier_dna = getattr(carrier, "dna", None)
        if carrier_dna is not None:
            carrier_iq = (
                carrier_dna.mental.vision * 0.6
                + carrier_dna.mental.composure * 0.4
            ) / 100.0
        # Feature #2 — the carrier team's live attack pattern (set per minute
        # by the MatchEngine from the team's identity + game state). The
        # possession engine uses its favoured flank / pattern identity to bias
        # circulation and the switch; receivers on that flank draw a mild
        # boost in _pick_receiver. A defending team carries NONE.
        _pattern = None
        if state is not None and position_engine is not None:
            _carrier_state = position_engine.states.get(getattr(carrier, "name", ""))
            if _carrier_state is not None:
                _pattern = (
                    getattr(state, "team_patterns", {}) or {}
                ).get(_carrier_state.team)
        engine = PossessionPhaseEngine(
            gk_snap, style_key=style_key, carrier_iq=carrier_iq,
            favored_flank=favored_flank(_pattern),
            pattern_key=(_pattern.value if _pattern is not None else None),
        )
        decision = engine.decide(
            current_phase, x, y, carrier.position, teammates,
            under_pressure=under_pressure,
            attacks_right=attacks_right,
        )
        return decision.phase, decision

    @classmethod
    def _gk_danger_level(
        cls,
        gk: PlayerProfile,
        x: float,
        y: float,
        def_players: List[PlayerProfile],
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
    ) -> float:
        """GK's own threat assessment: 0 (calm) to 100 (panic).

        Mirrors the same geometric inputs the rest of the defence reads
        from `threat_engine`, but localised to the keeper's position and
        the attackers immediately around him. Used by `_pick_gk_distribution`
        so that a keeper under pressure does not try to play out from the
        back the way a calm keeper would.
        """
        own_goal_x = 0.0 if attacks_right else 105.0

        # Proximity danger — closer to own goal = more danger (60 % weight).
        dist_to_goal = abs(x - own_goal_x)
        proximity_danger = max(
            0.0, min(100.0, (70.0 - dist_to_goal) / 70.0 * 100.0)
        )

        if not def_players or position_engine is None:
            return proximity_danger * 0.5

        gx, gy = position_engine.get_position(gk.name)
        nearby_attackers = 0
        nearest_attacker_dist: Optional[float] = None
        for d in def_players:
            if getattr(d, "position", None) == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            dist = math.hypot(dx - gx, dy - gy)
            if dist < 15.0:
                nearby_attackers += 1
            if nearest_attacker_dist is None or dist < nearest_attacker_dist:
                nearest_attacker_dist = dist

        # Pressure danger — how close is the nearest attacker (40 % weight).
        pressure_danger = 0.0
        if nearest_attacker_dist is not None:
            if nearest_attacker_dist < 1.5:
                pressure_danger = 80.0
            elif nearest_attacker_dist < 5.0:
                pressure_danger = 50.0
            elif nearest_attacker_dist < 15.0:
                pressure_danger = 20.0

        danger = proximity_danger * 0.6 + pressure_danger * 0.4
        if nearby_attackers >= 2:
            danger = min(100.0, danger + 15.0)
        if nearby_attackers >= 3:
            danger = min(100.0, danger + 10.0)

        return max(0.0, min(100.0, danger))

    @classmethod
    def _pick_gk_distribution(
        cls,
        gk: PlayerProfile,
        players: List[PlayerProfile],
        x: float,
        y: float,
        profile: "TeamProfile",
        position_engine: Optional[PositionEngine] = None,
        def_players: Optional[List[PlayerProfile]] = None,
        attacks_right: bool = True,
        att_style_key: Optional[str] = None,
    ) -> Optional[PlayerProfile]:
        """
        The keeper's deliberate distribution target.

        Possession-style sides play out to a deep anchor (CB/LB/RB/CDM) so
        the ball restarts build-up intent; direct sides (route-one, fluid
        counter) launch to a wide/forward outlet. The keeper doubles as the
        phase engine's GK overload anchor, so a return to him resets the
        phase machine rather than resetting to a random lob upfield.

        Checkpoint 16 — GK threat awareness: the keeper now reads the same
        danger level defenders use. Under HIGH/CRITICAL danger he overrides
        the team's possession preference and launches long, because a keeper
        who tries to play out from the back when his own goal is under real
        pressure is a liability.
        """
        short_roles = ("CB", "LB", "RB", "CDM")
        direct_roles = ("LW", "RW", "ST", "CF", "CAM")

        style_key = att_style_key or getattr(
            getattr(profile, "style", None), "value", "balanced"
        )
        is_direct = style_key in ("route_one", "fluid_counter", "direct",
                                   "ultra_attacking", "attacking")

        # GK threat awareness: under HIGH/CRITICAL danger the keeper clears
        # it long regardless of team style — safety first.
        gk_danger = cls._gk_danger_level(
            gk, x, y, def_players or [], position_engine, attacks_right
        )
        danger_override = gk_danger >= 60.0

        # Under pressure the keeper clears it long (route-one instinct);
        # a threatened GK (HIGH/CRITICAL danger) also launches long even if
        # his team is a possession side.
        launch = is_direct or danger_override or x > 25.0
        if launch:
            return cls.pick_weighted(
                [p for p in players if p.position != "GK"],
                lambda p: 2.2 if p.position in direct_roles else 0.15,
                exclude=gk.name,
            )
        return cls.pick_weighted_spatial(
            [p for p in players if p.position != "GK"],
            lambda p: 3.0 if p.position in short_roles else 0.2,
            position_engine, x, y,
            exclude=gk.name,
        )

    @classmethod
    def _pick_receiver(
        cls,
        players: List[PlayerProfile],
        passer: PlayerProfile,
        x: float,
        profile: "TeamProfile",
        preferred_positions: List[str] = None,
        position_engine: Optional[PositionEngine] = None,
        y: float = 34.0,
        def_players: Optional[List[PlayerProfile]] = None,
        attacks_right: bool = True,
        possession_phase: Optional[PossessionPhase] = None,
        match_state: Optional["MatchState"] = None,
    ) -> Optional[PlayerProfile]:
        # Closer to goal = higher chance of forward player receiving.
        # VERTICALITY RE-BALANCE: the gradient was min(5.0, 1 + (x/105)*4) —
        # a forward label was worth up to 5x whenever the ball was near the
        # box, which made "pass it to the deepest forward" the winner on
        # almost every touch even from midfield. The gradient is compressed
        # (cap 4.4x) so forward preference survives but support/lateral
        # outlets compete properly — less forced verticality, same shape.
        fwd_weight = min(4.4, 1.0 + (x / 105) * 3.4)
        fwd_pos = preferred_positions or ["CAM", "LW", "RW", "ST", "CF", "CM"]

        # Confidence gates how willing the passer is to attempt a pass into a
        # tightly-marked receiver. Low confidence → shy away from marked
        # targets (strong weight penalty); high confidence → still try the
        # risky forward pass (weak penalty).
        confidence = getattr(getattr(passer, "dna", None), "form", None)
        conf = (confidence.confidence / 100.0) if confidence is not None else 0.5
        confidence_factor = 0.85 - conf * 0.55   # 0.85 (low conf) .. 0.30 (high conf)

        # Bug fix (GK positional regression): GK used to share the flat
        # "everyone else" weight (1.0) with CB/LB/RB/CDM here. Measured
        # effect: GK was receiving passes as far forward as x=66.9,
        # accounting for ~65% of his total logged touches across a match
        # and the single biggest contributor to his average position
        # drifting past a believable deep-keeper range. The spatial
        # plausibility multiplier alone wasn't suppressing this enough at
        # long range (it has a soft floor, by design, for genuine outlier
        # plays) — GK needs his own sharply-tapering weight on top of it,
        # not parity with outfield defenders.
        def label_weight(p: PlayerProfile) -> float:
            if p.position == "GK":
                # Clever & realistic GK involvement:
                # Modern GKs (like Ederson, Alisson) are active sweepers and release valves.
                # When the ball is in the defensive third, they are a strong outlet.
                # Even in the middle third (up to x=50), they are a viable back-pass option to relieve pressure.
                if x <= 35:
                    # Checkpoint 14: in a REGROUP phase the whole point is a
                    # deliberate regression — the keeper is THE structural
                    # reset receiver, so his weight is lifted well above the
                    # generic "solid back-pass option".
                    if possession_phase == PossessionPhase.REGROUP_BUILD_UP:
                        return 2.6
                    return 1.8   # Solid back-pass option in own third
                elif x <= 50:
                    return 0.5   # Occasional release valve from midfield
                else:
                    return 0.05  # Rare, but possible (e.g. extreme high line)

            # FIX: Taper Center-Back weight when passing INTO/INSIDE the box (x >= 83)
            # CBs can build up in mid third, but shouldn't be primary receivers inside the box!
            if p.position == "CB":
                if x >= 83:
                    base = 0.05  # Extremely rare for CB to receive inside opponent box in open play
                elif x >= 70:
                    base = 0.4   # Low weight in the final third
                else:
                    base = 1.8   # Normal/high weight in own half & mid third
            else:
                # Checkpoint 22 — winger channel-discipline gate. Real
                # wingers get a "forward label" bonus (fwd_weight) for
                # good reason: they SHOULD be a prime target once play
                # reaches the final third. But that bonus was being
                # applied purely from the position LABEL ("LW"/"RW" is in
                # fwd_pos"), with zero regard for whether the winger is
                # actually in his flank channel right now — so during
                # ordinary central circulation (MIDFIELD_CIRCULATION),
                # a winger who had drifted centrally kept winning the
                # same forward bonus as a genuinely central CAM/CM, and
                # position_engine.receive_option_quality()'s discipline
                # penalty (a soft ~0.3-0.4x multiplier, not a hard
                # exclusion) wasn't enough on its own to outweigh a 5x
                # label bonus. That combination is what was producing a
                # winger's pass map fanning from a central hub near the
                # halfway line instead of the touchline — a genuine,
                # traced bug, not a fabricated stat artifact.
                #
                # Fix: for LW/RW specifically, only award the FULL
                # fwd_weight when the player's real, live position
                # (from position_engine, via winger_behavior.py's own
                # WingerSpatialProfile.flank_channel()) says he's
                # actually wide right now. Out of his channel, he's
                # tapered to a modest fraction of the bonus — still a
                # viable option ahead of the ball (matches how a CB
                # tapers rather than zeroes near the box above), just no
                # longer specially rewarded for a label he isn't
                # currently living up to. Falls back to the old
                # unconditional fwd_weight when no position_engine or no
                # registered winger profile is available, so nothing
                # breaks for callers that don't supply either.
                if p.position in ("LW", "RW") and position_engine is not None:
                    wp = position_engine.winger_registry.get(p.name)
                    if wp is not None:
                        cur_x, cur_y = position_engine.get_position(p.name)
                        if wp.flank_channel(cur_y):
                            base = fwd_weight
                        else:
                            base = 1.0 + (fwd_weight - 1.0) * 0.25
                    else:
                        base = fwd_weight
                # FULLBACK OVERLAP FIX: Overlapping fullbacks (LB/RB advanced
                # into the final third) should be attractive passing options
                # for wingers, providing width + support. Without this boost,
                # fullbacks get only base=1.0 weight even when perfectly
                # positioned on the overlap, making wingers recycle sideways
                # instead of playing the support runner.
                elif p.position in ("LB", "RB") and position_engine is not None:
                    cur_x, cur_y = position_engine.get_position(p.name)
                    # Advanced into attacking territory (x > 60 for attacking right)
                    advanced = (cur_x > 60.0 if attacks_right else cur_x < 45.0)
                    if advanced:
                        # Give them a healthy forward-player bonus when overlapping
                        base = fwd_weight * 0.70  # 70% of winger/CAM weight
                    else:
                        base = 1.0  # Normal midfielder weight in own half
                else:
                    base = fwd_weight if p.position in fwd_pos else 1.0

            # Marking: tightly-marked receivers are harder to find. The
            # weight penalty scales with how confident the passer is — a
            # confident creator still threads the pass to a marked forward.
            tight = cls._marking_tightness(
                p, x, y, def_players or [], position_engine, attacks_right
            )
            if tight > 0:
                base *= max(0.05, 1.0 - tight * confidence_factor)

            # Checkpoint 21 — anti-clustering receive weighting: a pass is
            # aimed at a teammate who is IN POSITION — at (or running toward)
            # their formation post, in a reachable passing relationship to
            # the ball. This replaces the old near-ball ellipse (which
            # rewarded whoever stood closest to the ball and therefore fed
            # the mid-pitch clump). The GK is skipped — his weight is already
            # governed by the explicit back-pass rules above.
            if p.position != "GK" and position_engine is not None:
                quality = position_engine.receive_option_quality(p.name, x, y, attacks_right)
                base *= (ELLIPSE_COMPOSE_FLOOR + (1.0 - ELLIPSE_COMPOSE_FLOOR) * quality)

            # ── CHECKPOINT 29: BLOCK NAVIGATION ───────────────────────
            # Determine which team's block the passer faces:
            #   attacks_right=True  → passer goes right → opponent holds right goal
            #                         → opponent is the away team → away_block
            #   attacks_right=False → passer goes left  → opponent holds left goal
            #                         → opponent is the home team → home_block
            block_shape = None
            if match_state is not None:
                block_shape = (match_state.away_block if attacks_right
                               else match_state.home_block)

            if block_shape is not None and position_engine is not None:
                px, py = position_engine.get_position(passer.name)
                rx, ry = position_engine.get_position(p.name)

                # Derive passer attributes from DNA for accurate orbital scoring
                _pmt = getattr(getattr(passer, 'dna', None), 'mental', None)
                _vision     = float(getattr(_pmt, 'vision',    60.0)) if _pmt else 60.0
                _composure  = float(getattr(_pmt, 'composure', 50.0)) if _pmt else 50.0
                # High composure → patient → prefers safe orbital routes (low risk tolerance)
                _risk_tol   = max(0.10, min(0.90, 1.0 - _composure / 100.0))

                # Does this pass go around the block or through it?
                orbital = BlockNavigationEngine.pass_orbital_score(
                    (px, py), (rx, ry), block_shape,
                    passer_vision=_vision,
                    passer_risk_tolerance=_risk_tol,
                    passer_position=passer.position,
                    is_progressive=(rx > px + 8 if attacks_right else rx < px - 8),
                )
                base *= orbital

                # Bonus if receiver is sitting in an open channel
                channel_mult = BlockNavigationEngine.channel_bonus((rx, ry), block_shape)
                base *= channel_mult

                # Volume-recycler bonus: high-vision CMs/CDMs prefer to keep the
                # ball moving around the edges of the block (Modric/Tanaka pattern)
                recycle = BlockNavigationEngine.recycle_tendency_score(
                    passer, p, block_shape, receiver_pos=(rx, ry)
                )
                base *= recycle


            # ── CHECKPOINT 20: OFFSIDE-POSITION GUARD ────────────────
            # Real playmakers never feed a teammate who is standing in an
            # offside POSITION (beyond the second-last defender and ahead of
            # the ball) — the ball would be dead before it arrives. They
            # recycle to an onside option instead. This is a strong weight
            # penalty, not a hard exclusion, so a rare timed run off the line
            # still has a chance. The GK's back-pass role is untouched.
            if p.position != "GK" and position_engine is not None:
                rx, _ry = position_engine.get_position(p.name)
                second_last = cls._second_last_defender_x(
                    def_players or [], position_engine, attacks_right
                )
                if cls._in_offside_position(rx, x, second_last, attacks_right):
                    base *= 0.10
                

            return base

        # ── CHECKPOINT 20: OFFSIDE-POSITION GUARD (hard filter) ─────
        # Real playmakers never feed a teammate who is standing in an
        # offside POSITION (beyond the second-last defender and ahead of
        # the ball) — the ball would be dead before it arrives. Exclude
        # offside-position receivers from the candidate pool entirely when
        # an onside option exists. The GK is exempt (his back-pass role is
        # governed by the rules above), and if every outfield option is
        # offside we fall back to the full pool — a team pinned high still
        # has to play somewhere, and the Law-11 check remains the backstop.
        candidates = players
        if position_engine is not None and def_players:
            second_last = cls._second_last_defender_x(
                def_players, position_engine, attacks_right
            )
            if second_last is not None:
                onside = [
                    p for p in players
                    if p.name == passer.name
                    or p.position == "GK"
                    or not cls._in_offside_position(
                        position_engine.get_position(p.name)[0], x,
                        second_last, attacks_right)
                ]
                if any(p.name != passer.name for p in onside):
                    candidates = onside

        # Checkpoint 21 — plain weighted draw over the scored options. The
        # option quality above ALREADY encodes reachability / direction /
        # post-discipline, so no extra near-ball plausibility multiply is
        # applied here — that term is what pinned the pass to the central
        # clump and let the ball never leave the middle of the pitch.
        #
        # Feature #2 — PATTERN FLANK PULL: when this team is leaning on a side
        # overload / wing isolation, receivers already standing on that flank
        # draw a mild weight bump (and the far side a small tax). This is the
        # player-level half of the overload: the ball LIVES on the pattern's
        # flank (3v2s are created there) until the far lane truly opens for
        # the switch. It is deliberately weak (1.18 / 0.88) so the geometric
        # weighting above keeps its authority.
        _base_labeller = label_weight
        _fav = None
        _pteam = None
        if match_state is not None and position_engine is not None:
            _pst = position_engine.states.get(getattr(passer, "name", ""))
            if _pst is not None:
                _pteam = _pst.team
                _pattern = (getattr(match_state, "team_patterns", {}) or {}).get(_pteam)
                _fav = favored_flank(_pattern)

        def label_weight(p: PlayerProfile) -> float:
            w = _base_labeller(p)
            if _fav and position_engine is not None:
                try:
                    _r = position_engine.states.get(p.name)
                    if _r is not None and _r.team == _pteam:
                        _ry = position_engine.get_position(p.name)[1]
                        w *= flank_bias_multiplier(_ry, attacks_right, _fav)
                except Exception:
                    pass
            return w

        return cls.pick_weighted(
            candidates,
            label_weight,
            exclude=passer.name,
        )

    @classmethod
    def _carry_probability(cls, player: PlayerProfile, x: float, profile: "TeamProfile") -> float:
        """Higher dribbling + further forward = more carries."""
        base = player.dna.tendencies.attempts_dribble
        position_bonus = {
            "LW": 0.15, "RW": 0.15, "CAM": 0.10,
            "CM": 0.05, "ST": 0.08, "CF": 0.08,
        }.get(player.position, 0.0)
        # More carries in midfield/attack, fewer in own half
        territory_mult = 0.5 if x < 35 else (1.2 if x > 65 else 1.0)
        return min(0.55, (base + position_bonus) * territory_mult)

    @classmethod
    def _winger_carry_steering(
        cls,
        player: PlayerProfile,
        x: float,
        y: float,
        attacks_right: bool,
        under_pressure: bool,
        def_players: Optional[List[PlayerProfile]],
        position_engine: Optional[PositionEngine],
        commit_rolls: bool = True,
    ) -> Tuple[Optional[str], Optional[float], float]:
        """
        Checkpoint 18 wiring — the Winger Behaviour Engine's on-the-ball
        steering, made LIVE in the event chain.

        Returns (drive_mode, anchor_y, bias):
            drive_mode : None | "byline" | "cut_inside"
                "byline"     → commit to driving the touchline→byline corridor
                "cut_inside" → deliberate diagonal into the half-space, gated
                               on the half-space actually being open
            anchor_y   : formation-corrected touchline anchor (home_y). This
                follows the FORMATION (Checkpoint 21e), never the position
                name, so a mirrored (attacking-left) winger is steered to the
                correct side of the pitch.
            bias       : lateral carry bias (metres) from carry_direction_bias
                that pulls a drifted carry back onto the flank channel.

        commit_rolls=False skips the drive/cut rolls (used for micro-carries,
        where only the anchor + bias are wanted).
        """
        if (position_engine is None
                or getattr(player, "position", None) not in ("LW", "RW")):
            return None, None, 0.0
        profile = position_engine.winger_registry.get(player.name)
        if profile is None:
            return None, None, 0.0
        state = position_engine.states.get(player.name)
        anchor_y = state.home_y if state is not None else profile.touchline_anchor_y

        drive_mode = None
        if commit_rolls:
            isolated = False
            if def_players:
                try:
                    isolated, _fb, _d = profile.fullback_isolation(
                        x, y, def_players, position_engine, attacks_right,
                    )
                except Exception:
                    isolated = False
            try:
                if WingerBehaviorEngine.should_drive_byline(
                    profile, x, y, attacks_right,
                    isolated=isolated,
                    under_pressure=under_pressure,
                    defenders=def_players, position_engine=position_engine,
                    anchor_y=anchor_y,
                ):
                    drive_mode = "byline"
                elif WingerBehaviorEngine.should_cut_inside(
                    profile, x, y, attacks_right,
                    defenders=def_players, position_engine=position_engine,
                    anchor_y=anchor_y,
                ):
                    drive_mode = "cut_inside"
            except Exception:
                drive_mode = None

        bias = 0.0
        try:
            bias = WingerBehaviorEngine.carry_direction_bias(
                profile, x, y, attacks_right,
                defenders=def_players, position_engine=position_engine,
                anchor_y=anchor_y,
            )
        except Exception:
            bias = 0.0
        return drive_mode, anchor_y, bias

    @classmethod
    def _fullback_carry_steering(
        cls,
        player: PlayerProfile,
        x: float,
        y: float,
        attacks_right: bool,
        under_pressure: bool,
        def_players: Optional[List[PlayerProfile]],
        position_engine: Optional[PositionEngine],
        commit_rolls: bool = True,
    ) -> Tuple[Optional[str], Optional[float], float]:
        """
        Fullback flank commitment on the ball (mirror of
        _winger_carry_steering, but through the FullbackBehaviourEngine).

        Returns (drive_mode, anchor_y, bias):
            drive_mode : None | "overlap_hold"
                "overlap_hold" → an advanced fullback carries ON the touchline
                    (his overlap lane), not a random walk across midfield.
            anchor_y   : formation-corrected touchline anchor (home_y), so a
                mirrored (attacking-left) fullback is steered to the correct
                side of the pitch.
            bias       : lateral carry bias (metres) from
                FullbackBehaviorEngine.carry_direction_bias that pulls a
                drifted carry back onto the flank channel.
        """
        if (position_engine is None
                or getattr(player, "position", None) not in ("LB", "RB")):
            return None, None, 0.0
        profile = position_engine.fullback_registry.get(player.name)
        if profile is None:
            return None, None, 0.0
        state = position_engine.states.get(player.name)
        anchor_y = state.home_y if state is not None else profile.touchline_anchor_y

        drive_mode = None
        if commit_rolls:
            try:
                if FullbackBehaviorEngine.should_advance(
                    profile, x, y, attacks_right,
                    x, y,
                    in_possession=True,
                    under_pressure=under_pressure,
                    anchor_y=anchor_y,
                ):
                    drive_mode = "overlap_hold"
            except Exception:
                drive_mode = None

        bias = 0.0
        try:
            bias = FullbackBehaviorEngine.carry_direction_bias(
                profile, x, y, attacks_right,
                in_possession=True, defenders=def_players,
                position_engine=position_engine, anchor_y=anchor_y,
            )
        except Exception:
            bias = 0.0
        return drive_mode, anchor_y, bias

    @classmethod
    def _pick_wide_combo_target(
        cls,
        carrier: PlayerProfile,
        players: List[PlayerProfile],
        x: float, y: float,
        position_engine,
        def_players: Optional[List[PlayerProfile]],
        attacks_right: bool,
    ) -> Optional[PlayerProfile]:
        """
        Checkpoint 24 — the wide combination pass. A wide carrier (winger or
        fullback) in the final-third flank channel looks for the short game:
        cutback to the penalty-spot/edge area, lateral to the CAM/CM, or a
        recycle to the overlapping fullback. These are the passes that make
        up the dense flank web of a real winger's map (Doku, Saka, Vini).

        Archetype-modulated: byline drivers (traditional wingers) combine
        less often — their instinct is to carry and deliver; inverted and
        playmaking wide men combine more. Returns None when no credible
        short option exists (the caller falls back to normal selection).
        """
        in_flank = (y < 24) if carrier.position in ("LW", "LB") else (y > 44)
        if not in_flank:
            return None
        final_third = (x > 70) if attacks_right else (x < 35)
        middle_third = (x > 45) if attacks_right else (x < 60)
        if not middle_third:
            return None  # deep in own half the wide man's pass is structural

        wprof = position_engine.winger_registry.get(carrier.name)
        byline = wprof.byline_instinct if wprof is not None else 0.55
        if final_third:
            combo_prob = 0.78 - 0.40 * byline
        else:
            combo_prob = 0.62 - 0.25 * byline
        if random.random() > combo_prob:
            return None

        sign = 1.0 if attacks_right else -1.0
        role_w = {"CAM": 1.20, "CM": 1.10, "CDM": 0.90, "LB": 1.05, "RB": 1.05,
                  "ST": 0.60, "CF": 0.60, "CB": 0.50, "LW": 0.15, "RW": 0.15,
                  "GK": 0.0}
        same_side_fb = {"LW": "LB", "LB": "LB", "RW": "RB", "RB": "RB"}[carrier.position]

        candidates: List[PlayerProfile] = []
        weights: List[float] = []
        # Checkpoint 31 — a winger's combination web decays HARDER with
        # distance (Doku 37/39 under 15m): near options dominate, but the
        # window stays 24m so a stretched block doesn't push the pass into
        # the LONG fallback paths (which are exactly what we're avoiding).
        is_winger = carrier.position in ("LW", "RW")
        dist_decay = 6.0 if is_winger else 9.0
        for t in players:
            if t.name == carrier.name:
                continue
            w = role_w.get(t.position, 0.4)
            if w <= 0.0:
                continue
            tx, ty = position_engine.get_position(t.name)
            d = math.hypot(tx - x, ty - y)
            if d < 3.0 or d > 24.0:
                continue
            ahead = (tx - x) * sign
            if ahead > 8.0:
                continue  # beyond that it's a through ball, not a combination
            # A wide-origin pass INTO the box is a cross/cutback by provider
            # definition — real wingers play 2-4 of those a match (they're
            # the cross mechanism's job), not 20+. Keep a trickle only.
            box_line = 88.0 if attacks_right else 17.0
            if (tx > box_line) if attacks_right else (tx < box_line):
                w *= 0.15
            # marking: skip smothered targets, discount pressed ones
            nearest_def = 99.0
            if def_players:
                nearest_def = min(
                    (math.hypot(position_engine.get_position(dp.name)[0] - tx,
                                position_engine.get_position(dp.name)[1] - ty)
                     for dp in def_players if getattr(dp, 'position', None) != 'GK'),
                    default=99.0,
                )
            if nearest_def < 2.5:
                continue
            if nearest_def < 4.0:
                w *= 0.5
            if t.position == same_side_fb:
                w *= 1.25  # the overlap/underlap recycle
            w *= 1.0 / (1.0 + d / dist_decay)
            if abs(ty - y) >= 3.0:
                w *= 1.30  # laterals and cutback diagonals are the shape
            candidates.append(t)
            weights.append(w)
        if not candidates:
            return None
        return random.choices(candidates, weights=weights, k=1)[0]

    @classmethod
    def _should_be_long_pass(cls, player: PlayerProfile, x: float, profile) -> bool:
        from match_engine import TeamStyle
        # Handle both TeamProfile and EffectiveTactics
        if hasattr(profile, 'style') and profile.style == TeamStyle.ROUTE_ONE:
            return random.random() < 0.55
        # Checkpoint 24 — wingers RECEIVE switches and diagonals; they almost
        # never LAUNCH them (Doku: 1 long ball per 90). A winger trying to
        # hit a 35m ball is the quarterback's job description, not his.
        if getattr(player, "position", "") in ("LW", "RW"):
            return random.random() < (0.05 if x < 40 else 0.03)
        if x < 40:
            return random.random() < 0.20
        return random.random() < 0.12

    @classmethod
    def _pass_success(cls, player: PlayerProfile, is_long: bool, under_pressure: bool,
                       receiver: Optional[PlayerProfile] = None,
                       chemistry=None,
                       marking: float = 0.0,
                       confidence: Optional[float] = None,
                       lane_mult: float = 1.0,
                       is_airborne: bool = False,
                       weather: Optional[WeatherCondition] = None) -> bool:
        prob = DNAFactory.get_pass_accuracy(player.dna, is_long=is_long, under_pressure=under_pressure)
        # Chemistry modifier: if both players have chemistry data, multiply
        # pass accuracy by the chemistry multiplier (0.90x to 1.14x).
        if receiver is not None:
            # Resolve the active chemistry from the live registry unless the
            # caller passed one explicitly. Falls back to the passer's team.
            from squad_chemistry import active_for
            chem = chemistry if chemistry is not None else active_for(getattr(player, "team_name", None))
            if chem is not None:
                chem_mult = chem.pass_chemistry_mult(player.name, receiver.name)
                prob = min(0.95, prob * chem_mult)
        # Marking penalty: a pass into a tightly-marked receiver is harder to
        # complete, but the effect is modest — a smothered pass drops ~25%
        # off base accuracy at most, not half. Confidence modulates the
        # damage: a confident passer threads it through, a low-confidence one
        # leaves the pass short and the defender intercepts.
        if marking > 0:
            if confidence is None:
                conf = getattr(getattr(player, "dna", None), "form", None)
                conf_val = (conf.confidence / 100.0) if conf is not None else 0.5
            else:
                conf_val = confidence / 100.0
            resilience = 0.55 + conf_val * 0.45    # 0.55 (low conf) .. 1.0 (high conf)
            prob *= max(0.62, 1.0 - marking * 0.38 * (1.30 - resilience))
        # Checkpoint 25 — the pass LANE bites in execution, not just in
        # target selection. Until now a pass through a defender's cover
        # shadow completed at the same rate as an open one — line-breaking
        # balls were free. A choked corridor cuts completion sharply.
        prob *= lane_mult
        if WeatherPhysics.enabled or weather is not None:
            prob *= WeatherPhysics.pass_accuracy_mult(weather, is_long=is_long, is_airborne=is_airborne)
        prob = _get_soul_applicator().modify_pass_accuracy(player, prob)
        return random.random() < prob

    @classmethod
    def _lane_interceptor(cls, x: float, y: float, ex: float, ey: float,
                          def_players, position_engine):
        """The defender standing ON the pass lane who cuts the ball out.

        Returns (defender, intercept_x, intercept_y) at his projection
        point on the corridor, or None when nobody is within
        LANE_REACTION_DIST. The interception belongs to the man whose body
        the carrier tried to play through.
        """
        if position_engine is None or not def_players:
            return None
        seg_len = math.hypot(ex - x, ey - y)
        if seg_len < 1e-6:
            return None
        best = None
        for d in def_players:
            if getattr(d, "position", None) == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            t = ((dx - x) * (ex - x) + (dy - y) * (ey - y)) / (seg_len * seg_len)
            t = max(0.0, min(1.0, t))
            px = x + t * (ex - x)
            py = y + t * (ey - y)
            dist = math.hypot(dx - px, dy - py)
            if dist <= LANE_REACTION_DIST and (best is None or dist < best[0]):
                best = (dist, d, px, py)
        if best is None:
            return None
        _, d, px, py = best
        return d, px, py

    @classmethod
    def _through_ball_opportunity(
        cls,
        passer: PlayerProfile,
        receiver: PlayerProfile,
        x: float, y: float,
        attacks_right: bool,
        def_players: List[PlayerProfile],
        position_engine: Optional[PositionEngine],
    ) -> float:
        """Score the OPPORTUNITY for a genuine through ball (0..1).

        A through ball is a pass DELIVERED THROUGH THE DEFENSIVE LINE (a
        lateral channel between line defenders — CB/RB, CB-CB, LB-CB — or
        round a windmill of them) into the SPACE BEHIND that line, to a
        receiver who is running into/onto that space before the defence can
        turn. It must bypass everyone (the line AND the goalkeeper's sweeper
        race — the latter is handled by ``resolve_ground_pass``).

        This returns how open that whole picture is RIGHT NOW:
          * line_gap   — the widest open lateral channel through the back
                         line (normalised 0..1). A back line with no gap
                         (a locked block) scores ~0.
          * behind     — how much room is BEHIND the offside line for the
                         receiver to run into (0..1, reusing the same
                         second-last-defender geometry as Law 11).
          * receiver   — how "toward the open channel" the receiver is
                         running (his lateral bias into the gap + being
                         level with/behind the line but onside).

        A zero/no-data case returns 0.5 (neutral) so the downstream
        tendency/skill weighting stays the arbiter when we can't see space.
        """
        if position_engine is None or not def_players or receiver is None:
            return 0.5
        sign = 1.0 if attacks_right else -1.0
        goal_x = 105.0 if attacks_right else 0.0

        # 1) Back line ahead of the ball — outfield defenders between the
        #    ball and their own goal (CB/LB/RB hold the line; a CDM that has
        #    dropped in front is not part of the line to break).
        line = []
        for d in def_players:
            if getattr(d, "position", None) in ("GK",):
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, dy = position_engine.get_position(dname)
            ahead = (dx > x) if attacks_right else (dx < x)
            # Only defenders goal-side of the ball and roughly in the back
            # four's band count toward the line; exclude advanced CDMs.
            pos = getattr(d, "position", "")
            if ahead and pos in ("CB", "LB", "RB", "CDM"):
                line.append((dx, dy))
        if not line:
            return 0.5

        # 2) Sort the line by distance to the defended goal (deepest first).
        line.sort(key=lambda pt: abs(goal_x - pt[0]))

        # 3) Lateral gaps between adjacent line bodies (normalised: >7m wide
        #    is a real channel to thread; <2.5m is shut). Gaps run between
        #    every neighbouring y in the line — this is what captures the
        #    CB-RB / CB-CB / LB-CB seams.
        ys = sorted(p[1] for p in line)
        widest = 2.5
        for a, b in zip(ys, ys[1:]):
            gap = b - a
            widest = max(widest, gap)
        line_gap = max(0.0, min(1.0, (widest - 2.5) / (7.0 - 2.5)))

        # 4) Space behind the offside line for the run. Reuse the same
        #    second-last-defender geometry as the Law 11 check.
        second_last = cls._second_last_defender_x(
            def_players, position_engine, attacks_right)
        behind = 0.5
        rx, _ry = position_engine.get_position(receiver.name)
        if second_last is not None:
            if attacks_right:
                window = rx - second_last        # positive = clearly beyond the line
            else:
                window = second_last - rx
            behind = max(0.0, min(1.0, window / 15.0))

        # 5) Receiver running toward the open channel: his lateral bias into
        #    the widest gap and his vertical level vs the line. A receiver
        #    camped ON the line can't run onto a through ball.
        widest_band = 0.5
        if len(ys) >= 2:
            # Find the largest adjacent-gap midpoint and how close the
            # receiver's y is to it.
            best_mid, best_span = None, 0.0
            for a, b in zip(ys, ys[1:]):
                span = b - a
                if span > best_span:
                    best_span, best_mid = span, (a + b) / 2.0
            if best_mid is not None:
                near = max(0.0, 1.0 - abs(_ry - best_mid) / 8.0)
                widest_band = max(0.0, min(1.0, 0.5 + near * 0.5))
        if line_gap <= 0.05:
            widest_band = max(0.0, widest_band - 0.3)

        # 6) Combine — all three must be present for a real through-ball
        #    chance; a flat mid-tight block suppresses all of them.
        opportunity = line_gap * 0.45 + behind * 0.35 + widest_band * 0.20
        return max(0.0, min(1.0, opportunity))

    @staticmethod
    def _second_last_defender_x(
        def_players: List[PlayerProfile],
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
    ) -> Optional[float]:
        """
        X-coordinate of the second-last defender (Law 11's offside line).

        For a team attacking right the last defender is the smallest x; for a
        team attacking left it is the largest x. Excludes the GK, who is the
        "last" defender by position but never counts toward the offside line.
        Returns None when there is no position engine or fewer than two
        outfield defenders are known.
        """
        if position_engine is None or not def_players:
            return None
        xs = []
        for d in def_players:
            if getattr(d, "position", None) == "GK":
                continue
            dx, _dy = position_engine.get_position(d.name)
            xs.append(dx)
        if len(xs) < 2:
            return None
        xs.sort() if attacks_right else xs.sort(reverse=True)
        return xs[1]

    @staticmethod
    def _in_offside_position(
        player_x: float,
        ball_x: float,
        second_last_x: Optional[float],
        attacks_right: bool,
    ) -> bool:
        """
        Law 11 position check for a player currently standing at player_x.

        True when the player is in the opponent's half, ahead of the ball,
        and ahead of the second-last defender — i.e. in an offside POSITION.
        (Being in an offside position is not an offence by itself; this only
        feeds decision logic that avoids passing to such a player.)
        """
        if second_last_x is None:
            return False
        in_opp_half = player_x > 52.5 if attacks_right else player_x < 52.5
        ahead_ball = player_x > ball_x if attacks_right else player_x < ball_x
        ahead_line = player_x > second_last_x if attacks_right else player_x < second_last_x
        return in_opp_half and ahead_ball and ahead_line

    @classmethod
    def _check_offside(
        cls,
        passer: PlayerProfile,
        receiver: PlayerProfile,
        x: float, y: float,
        end_x: float, end_y: float,
        def_players: List[PlayerProfile],
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
    ) -> OffsideVerdict:
        """
        Check if a completed pass leaves the receiver in an offside position.

        Offside conditions (Law 11):
            1. Receiver is in the opponent's half at the moment the ball is played.
            2. Receiver is ahead of the second-last defender.
            3. Receiver is ahead of the ball.

        The three conditions are judged against the RECEIVER's live position
        at the moment the ball is played (per Law 11), NOT against where the
        pass destination ends up — a runner starting onside and chasing a
        through ball beyond the line is onside, exactly as in real football.

        Returns an OffsideVerdict: called=(x, y) when the flag rose,
        uncalled=(x, y) when the receiver was beyond the line but the flag
        stayed down (delayed offside — VAR material), both None otherwise.
        The location is the receiver's position at the moment of the pass.
        """
        if position_engine is None:
            return OffsideVerdict()

        # Law 11 is judged at the moment the ball is played, on the
        # receiver's LIVE position — not on where the pass destination
        # ends up. A striker starting onside and running onto a through
        # ball beyond the line is onside, even when the ball lands beyond the line.
        rx, ry = position_engine.get_position(receiver.name)

        # Condition 1: receiver must be in the opponent's half
        opp_half = rx > 52.5 if attacks_right else rx < 52.5
        if not opp_half:
            return OffsideVerdict()

        # Condition 3: receiver must be ahead of the ball
        ball_ahead = (rx > x) if attacks_right else (rx < x)
        if not ball_ahead:
            return OffsideVerdict()

        second_last_x = cls._second_last_defender_x(
            def_players, position_engine, attacks_right
        )
        if second_last_x is None:
            return OffsideVerdict()

        # Condition 2: receiver must be ahead of second-last defender
        receiver_ahead = (rx > second_last_x) if attacks_right else (rx < second_last_x)
        if not receiver_ahead:
            return OffsideVerdict()

        # ── CALIBRATED OFF-SIDE DECISION (see OFFSIDE_* constants) ──
        # Margin in metres by which the receiver is beyond the offside line.
        margin = (rx - second_last_x) if attacks_right else (second_last_x - rx)

        # Law 11: being LEVEL with the second-last defender is NOT offside. A
        # small "benefit of the doubt" tolerance keeps a receiver level with
        # (or a centimetre past) the line onside, exactly as referees call it.
        if margin <= OFFSIDE_LEVEL_TOL_M:
            return OffsideVerdict()

        # Run-timing / flag discipline: the flag rises with how decisively the
        # receiver is beyond the line, but it is not a certainty even then —
        # most beyond-the-line deliveries never arrive because the pass would
        # be dead before the receiver gets there. This is what brings the
        # per-match total from ~60 back to the realistic 3-8.
        decisiveness = (margin - OFFSIDE_LEVEL_TOL_M) / (
            OFFSIDE_FULL_MARGIN_M - OFFSIDE_LEVEL_TOL_M)
        decisiveness = max(0.0, min(1.0, decisiveness))
        p_offside = OFFSIDE_CALL_FLOOR + (
            OFFSIDE_CALL_PEAK - OFFSIDE_CALL_FLOOR) * decisiveness
        if random.random() > p_offside:
            # The flag stayed down — a clear offside position left uncalled.
            # Play continues; stamp for a possible VAR review if he scores.
            return OffsideVerdict(uncalled=(rx, ry))

        return OffsideVerdict(called=(rx, ry))

    @classmethod
    def _generate_through_ball(
        cls, passer: PlayerProfile, receiver: PlayerProfile,
        x: float, y: float,
        minute: int, phase: MatchPhase, game_state: GameState,
        attacks_right: bool, profile,
    ) -> Tuple[bool, float, float, float, float]:
        """Calculate through ball destination and success. Returns (success, end_x, end_y, dist, tb_prob)."""
        vision = passer.dna.mental.vision / 100.0
        pass_skill = (passer.dna.passing.short_passing + passer.dna.passing.long_passing) / 200

        space_ahead = (105 - x) if attacks_right else x
        base_dist = 7 + vision * 18
        space_factor = min(1.0, space_ahead / 60)
        dist = base_dist * (0.7 + space_factor * 0.3)
        dist = max(5, min(space_ahead * 0.7, dist))

        receiver_pos = receiver.position
        is_wide = receiver_pos in ("LW", "RW")
        if is_wide:
            if attacks_right:
                end_ty = random.uniform(14, 24) if y < 34 else random.uniform(44, 54)
            else:
                end_ty = random.uniform(44, 54) if y < 34 else random.uniform(14, 24)
        else:
            spread = 12 * (1 - pass_skill * 0.5)
            end_ty = y + (0.5 - random.random()) * spread
            end_ty = max(18, min(50, end_ty))

        end_tx = cls.clamp_x(x + (dist if attacks_right else -dist), attacks_right)
        end_ty = max(5, min(63, end_ty))

        base_prob = DNAFactory.get_pass_accuracy(passer.dna, is_long=False, under_pressure=False)
        through_difficulty = 0.60 + vision * 0.25
        receiver_pace_bonus = (receiver.dna.physical.pace / 100.0) * 0.06
        tb_prob = base_prob * through_difficulty + receiver_pace_bonus
        tb_prob = max(0.10, min(0.85, tb_prob))
        tb_success = random.random() < tb_prob

        return tb_success, end_tx, end_ty, dist, tb_prob

    @classmethod
    def _generate_cross_destination(
        cls, x: float, y: float, attacks_right: bool,
        crossing_skill: float
    ) -> Tuple[str, float, float, float, float]:
        """Pick cross target zone and compute destination coords. Returns (zone_label, end_x, end_y, raw_end_x, raw_end_y)."""
        norm_x = x if attacks_right else 105 - x
        dist_to_byline = 104 - norm_x

        spread = max(2.0, 6.0 * (1 - crossing_skill * 0.6))

        if dist_to_byline > 25:
            zones = [("near_post", 25), ("penalty_spot", 35), ("far_post", 25), ("cutback", 15)]
        elif dist_to_byline > 10:
            zones = [("near_post", 35), ("penalty_spot", 20), ("far_post", 25), ("cutback", 20)]
        else:
            zones = [("near_post", 45), ("far_post", 25), ("penalty_spot", 15), ("cutback", 15)]

        weights = [w for _, w in zones]
        zone_labels = [z for z, _ in zones]
        zone = random.choices(zone_labels, weights=weights)[0]

        targets = {
            "near_post":    (92 + (norm_x - 75) * 0.15, 33 - (y - 34) * 0.1),
            "far_post":     (98 + max(0, (norm_x - 85)) * 0.15, 23 + (y - 34) * 0.05),
            "penalty_spot": (85, 35 + (y - 34) * 0.1),
            "cutback":      (78 + (norm_x - 78) * 0.2, 35 + (y - 34) * 0.1),
        }

        raw_tx, raw_ty = targets[zone]
        end_tx = raw_tx + random.gauss(0, spread * 0.5)
        end_ty = raw_ty + random.gauss(0, spread)
        end_tx = max(70, min(104, end_tx))
        end_ty = max(8, min(60, end_ty))

        if not attacks_right:
            end_tx = 105 - end_tx

        return zone, end_tx, end_ty, raw_tx, raw_ty

    @classmethod
    def _pass_distance(
        cls, player: PlayerProfile, x: float,
        is_long: bool, is_progressive: bool,
        under_pressure: bool, profile
    ) -> float:
        pass_skill = (player.dna.passing.short_passing + player.dna.passing.long_passing) / 2
        base = 3 + (pass_skill / 100) * 14

        pos_mult = 1.15 if x < 35 else (0.85 if x > 70 else 1.0)
        base *= pos_mult

        if under_pressure:
            base *= 0.65

        style_mult = {
            "tiki_taka": 0.70, "structured_possession": 0.75,
            "possession": 0.80, "defensive": 0.85, "park_the_bus": 0.80,
            "ultra_defensive": 0.85, "balanced": 1.0,
            "fluid_counter": 1.10, "attacking": 1.10, "ultra_attacking": 1.15,
            "gegenpressing": 1.0, "wing_play": 1.05, "vertical_tiki_taka": 1.0,
            "route_one": 1.60, "direct": 1.30
        }
        if hasattr(profile, 'style'):
            sm = style_mult.get(profile.style.value, 1.0)
        else:
            sm = 1.0
        base *= sm

        if is_long:
            return max(15, min(50, base * 2.8))
        # ── CHECKPOINT 31: WINGER SHORT-GAME PROFILE ─────────────
        # The real modern winger's pass map (Doku: 37/39 under 15m) is a
        # dense flank web of cutbacks, laterals and fullback recycles.
        # PLOFA wingers were medians 12.5-15m with only ~55% under 15m.
        # Shrink their base and cap the progressive stretch so winger
        # distribution lives in the 4-16m band; every other role unchanged.
        if getattr(player, "position", "") in ("LW", "RW"):
            base *= 0.78
            if under_pressure:
                base *= 0.80   # squeezed on the touchline he still recycles short
            if is_progressive:
                return max(5, min(16, base * 1.35))
        if is_progressive:
            return max(5, min(22, base * 1.35))
        return max(2, min(20, base))

    @classmethod
    def _pass_energy_cost(
        cls,
        player: PlayerProfile,
        x: float, y: float,
        end_x: float, end_y: float,
        is_long: bool,
        under_pressure: bool,
        marking: float = 0.0,
    ) -> float:
        """
        Compute the relative energy cost of a pass (0.0 — 1.0+).

        A long pass in a difficult position/angle uses more energy/mental
        battery than a short pass in the same position. This models the
        real-life cost: playing a 40m diagonal under pressure drains more
        than a 5m safe pass, and a highly-marked receiver demands more
        mental focus from the passer.

        The composite is then modulated by the passer's DNA: composure
        and vision reduce the mental battery cost.
        """
        dist = math.hypot(end_x - x, end_y - y)

        dist_factor = min(1.0, dist / 50.0)

        if dist > 1.0:
            dx = abs(end_x - x)
            dy = abs(end_y - y)
            angle_ratio = dy / max(1.0, dx)
            angle_factor = min(1.0, angle_ratio * 1.5)
        else:
            angle_factor = 0.0

        marking_factor = max(0.0, min(1.0, marking)) * 0.5
        type_factor = 0.3 if is_long else 0.1
        pressure_factor = 0.2 if under_pressure else 0.0

        energy = (
            dist_factor * 0.35 +
            angle_factor * 0.25 +
            marking_factor * 0.15 +
            type_factor * 0.15 +
            pressure_factor * 0.10
        )

        if hasattr(player, 'dna') and player.dna is not None:
            composure = getattr(getattr(player.dna, 'mental', None), 'composure', 50.0) / 100.0
            vision = getattr(getattr(player.dna, 'mental', None), 'vision', 50.0) / 100.0
            mental_resilience = (composure + vision) / 2.0
            energy *= max(0.5, 1.0 - mental_resilience * 0.4)

        return max(0.0, min(1.0, energy))

    @classmethod
    def _should_be_progressive(
        cls, player: PlayerProfile, x: float, profile,
        in_zone: bool
    ) -> bool:
        if not in_zone:
            return False
        vision_roll = player.dna.mental.vision / 100.0
        composure_roll = player.dna.mental.composure / 100.0
        safe_penalty = player.dna.tendencies.plays_safe * 0.3
        # VERTICALITY RE-BALANCE: the old coefficients (vision 0.5 +
        # composure 0.2, clamp 0.60) fired a "drive it forward" intent on
        # most touches in the progression zone. The bar is raised — vision
        # weighs less, safety weighs more, and the cap drops to 0.48 — so a
        # forward intent is a genuine decision by an elite passer rather
        # than the default exit from midfield.
        base_prob = vision_roll * 0.42 + composure_roll * 0.16 - safe_penalty * 1.15
        if hasattr(profile, 'style'):
            fast_styles = {"fluid_counter", "gegenpressing", "ultra_attacking", "route_one", "direct"}
            if profile.style.value in fast_styles:
                base_prob += 0.09
        # Confidence modulates forward-pass boldness: a player low on
        # confidence turns the ball back / sideways rather than attempting
        # the risky forward pass; a confident one commits to it.
        conf = getattr(getattr(player, "dna", None), "form", None)
        if conf is not None:
            conf_mult = 0.50 + (conf.confidence / 100.0) * 0.85   # 0.50 .. 1.35
            base_prob *= conf_mult
        base_prob = max(0.06, min(0.48, base_prob))
        return random.random() < base_prob

    @classmethod
    def _pass_destination(
        cls, player: PlayerProfile,
        x: float, y: float, pass_dist: float,
        profile, attacks_right: bool,
        is_long: bool, is_prog: bool
    ) -> Tuple[float, float]:
        pos_fwd_bias = {
            "GK": 0.80, "CB": 0.80, "CDM": 0.72,
            "LB": 0.68, "RB": 0.68,
            "CM": 0.75, "CAM": 0.80,
            "LW": 0.72, "RW": 0.72,
            "ST": 0.55, "CF": 0.58,
        }.get(player.position, 0.65)

        pos_factor = 1.0 - (x / 105) * 0.15
        safety_factor = 1.0 - player.dna.tendencies.plays_safe * 0.08

        style_dir = {
            "tiki_taka": 0.82, "structured_possession": 0.86,
            "possession": 0.86, "defensive": 0.90, "park_the_bus": 0.82,
            "ultra_defensive": 0.86, "balanced": 1.0,
            "fluid_counter": 1.10, "attacking": 1.06, "ultra_attacking": 1.10,
            "gegenpressing": 1.02, "wing_play": 1.04, "vertical_tiki_taka": 1.02,
            "route_one": 1.18, "direct": 1.14
        }
        if hasattr(profile, 'style'):
            style_mod = style_dir.get(profile.style.value, 1.0)
        else:
            style_mod = 1.0

        fwd_prob = pos_fwd_bias * pos_factor * style_mod * safety_factor
        fwd_prob = max(0.35, min(0.85, fwd_prob))

        comp = player.dna.mental.composure / 100.0
        vision = player.dna.mental.vision / 100.0
        plays_safe = player.dna.tendencies.plays_safe

        if random.random() < fwd_prob:
            advance_ratio = 0.20 + vision * 0.50
            advance_ratio = max(0.15, min(0.70, advance_ratio)) * comp
            dx = pass_dist * advance_ratio
        else:
            # Non-forward: sideway vs backward depends on safety + position
            if plays_safe > 0.55:
                # Safe player recycles sideways
                if x > 70:
                    dx = pass_dist * random.uniform(-0.20, 0.08)
                elif x < 35:
                    dx = pass_dist * random.uniform(-0.15, 0.10)
                else:
                    dx = pass_dist * random.uniform(-0.12, 0.10)
            else:
                # Adventurous player may switch or recycle deeper
                if x > 70:
                    dx = pass_dist * random.uniform(-0.35, 0.08)
                elif x < 35:
                    dx = pass_dist * random.uniform(-0.20, 0.10)
                else:
                    dx = pass_dist * random.uniform(-0.25, 0.10)

        end_px = cls.clamp_x(x + (dx if attacks_right else -dx), attacks_right)

        # PITCH WIDTH FIX: Increased from 4 + vert_skill * 16 (max 20m) to 
        # 6 + vert_skill * 22 (max 28m) to allow wider cross-field switches
        # and better pitch stretching for skilled passers.
        vert_skill = (player.dna.passing.short_passing + player.dna.technical.ball_control) / 200
        vert_range = 6 + vert_skill * 22
        end_py = y + (0.5 - random.random()) * vert_range
        end_py = max(2, min(66, end_py))

        # Preserve flank width for wide fullbacks on non-forward reset passes.
        # This avoids a safe fullback recycle pushing the ball unnaturally toward
        # the central channel when the pass is meant to be a sideways/backward option.
        # PITCH WIDTH FIX: Further relaxed from 18/50 to 12/56 to allow wider fullback play.
        is_forward = (dx > 0) if attacks_right else (dx < 0)
        if player.position in ("LB", "RB") and not is_forward:
            if y < 12.0:
                end_py = max(end_py, y - 3.0)
                end_py = min(end_py, 14.0)
            elif y > 56.0:
                end_py = min(end_py, y + 3.0)
                end_py = max(end_py, 54.0)

        # ── CHECKPOINT 18: WINGER FLANK PRESERVATION ──────────────
        # MODERN WINGER FIX: wingers are touchline-hugging flank attackers,
        # NOT drifting #10s. The middle of the pitch is always full — a #10
        # owns that space — and a winger who drifts inside leaves his flank
        # open. When a winger receives a pass, the destination must stay on
        # their flank channel, not drift toward midfield. This is one of the
        # key reasons wingers were ending up next to the CAM — every pass
        # pulled them central.
        # PITCH WIDTH FIX: Further relaxed caps from 18/50 to 12/56 to allow wingers
        # to reach touchline zones (0-10m, 58-68m) for realistic wing play.
        if player.position in ("LW", "RW"):
            flank_keep = 0.55 if not is_forward else 0.40
            if player.position == "LW":
                # Left winger: destination stays in the left flank channel
                if y < 26.0:
                    # Already wide — keep it wide (relaxed from 18.0 to 12.0)
                    end_py = min(end_py, 12.0)
                elif end_py > 12.0:
                    # Drifted central — push the destination back to the flank
                    end_py = y - (y - 12.0) * flank_keep
                    end_py = max(3.0, min(12.0, end_py))
            else:
                # Right winger: destination stays in the right flank channel
                if y > 42.0:
                    # Already wide — keep it wide (relaxed from 50.0 to 56.0)
                    end_py = max(end_py, 56.0)
                elif end_py < 56.0:
                    # Drifted central — push the destination back to the flank
                    end_py = y + (56.0 - y) * flank_keep
                    end_py = max(46.0, min(62.0, end_py))

        return end_px, end_py

    @classmethod
    def _pass_destination_to_receiver(
        cls, receiver: PlayerProfile,
        x: float, y: float, pass_dist: float,
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
        long_intent: bool = False,
    ) -> Tuple[float, float]:
        """
        Checkpoint 21 — anti-clustering pass delivery.

        The old `_pass_destination` aimed the ball at a RANDOM vector: a
        forward stride of `pass_dist * (0.20..0.70)` plus a lateral jitter of
        ±2-10m. The receiver never had to be near the landing point, so the
        receiver was then teleported to wherever the vector happened to land
        — and because that vector only moved forward with a tiny lateral
        spread, the whole team crept up the central channel over 90 minutes.

        Instead the ball is aimed at the receiver's LIVE position from the
        position engine. The receiver controls the endpoint, so the delivery
        and the next event's starting point agree. A winger whose post is the
        touchline ends up ON the touchline; a fullback recycling stays on his
        flank. `pass_dist` (the intended weight of the pass) caps how far the
        ball can travel so a short pass still cannot reach a far receiver.
        """
        if position_engine is not None:
            rx, ry = position_engine.get_position(receiver.name)
            dx = rx - x
            dy = ry - y
        else:
            # No position engine → no live receiver coordinates exist
            # (PlayerProfile carries no home xy). Legacy pre-Checkpoint-21
            # behaviour: a conservative forward stride with lateral jitter.
            stride = max(3.0, min(pass_dist, 18.0))
            lead_dir = 1.0 if attacks_right else -1.0
            dx = stride * 0.5 * lead_dir
            dy = (0.5 - random.random()) * 8.0
        d = math.hypot(dx, dy) or 1.0
        capped = min(d, max(pass_dist, 3.0))
        # Checkpoint 32 — delivery-range re-coupling. The cap above was
        # calibrated when teammates sat ~15m apart; the shape/stretch layer
        # now spaces possession options ~2x wider (median ~26m), so a pass
        # capped at pass_dist (~12m) died off the receiver's feet (~48%
        # underhit) and possessions collapsed. Short/progressive intents get
        # their delivery reach scaled toward the receiver's separation,
        # bounded by the soft ceiling so a safe pass cannot ping an
        # absurdly far runner. Long intent already carries enough range
        # (pass_dist >= spacing) and is left untouched.
        if not long_intent:
            capped = min(d, min(max(capped * REACH_SPACING_MULT,
                                    d * REACH_SPACING_TARGET),
                                REACH_SPACING_CEIL_M))
        # Lead the receiver: put the ball slightly in front of him so the
        # pass is a delivery, not a teleport. Longer balls lead further.
        # "In front" means in the direction of attack.
        lead = 2.0 if capped >= 22.0 else (0.8 if capped >= 10.0 else 0.0)
        lead_dir = 1.0 if attacks_right else -1.0
        end_px = x + (dx / d) * capped + lead * lead_dir
        end_py = y + (dy / d) * capped

        # ── CHECKPOINT 31: TOUCHLINE DELIVERY BAND FOR WIDE PLAYERS ──
        # This function aims at the receiver's LIVE position — so a winger
        # who had drifted into the half-space got fed THERE, re-planting him
        # central on every reception. That feedback loop is what made the
        # winger heat maps read like RM/LM (|y-centre| ≈ 15-17 vs an anchor
        # at 24). Real build-up feeds the touchline: the delivery itself
        # restores the width. Formation-corrected via home_y (Checkpoint
        # 21e), so mirrored attacking-left wingers get their own flank.
        #
        # FULLBACK FLANK COMMITMENT — the same loop bit the fullbacks: their
        # receptions were aimed at wherever they drifted, re-planting them in
        # the half-space instead of on the line. A fullback IS the width on
        # his side (overlap support + stretch), so he is fed the same
        # touchline band as the winger. The one carve-out: an INVERTED
        # fullback's deliberate tuck pocket ('underlapping builds up in the
        # half-space') is honoured, so the delivery follows the run instead
        # of yanking him back to the touchline.
        if position_engine is not None:
            _rstate = getattr(position_engine, "states", {}).get(receiver.name)
            if (_rstate is not None
                    and getattr(_rstate, "position", None) in ("LW", "RW", "LB", "RB")):
                r_home = _rstate.home_y
                r_sign = -1.0 if r_home <= 34.0 else 1.0   # toward his touchline
                band_far = r_home + r_sign * 3.0    # may sit 3m beyond anchor
                band_near = r_home - r_sign * 7.0   # inner edge: 7m infield
                if getattr(_rstate, "position", None) in ("LB", "RB"):
                    # Inverted fullbacks step inside during build-up — let
                    # their pocket (deep + infield) survive the band.
                    _fbreg = getattr(position_engine, "fullback_registry", None)
                    _fb_prof = (_fbreg.get(receiver.name)
                                if _fbreg is not None else None)
                    if (_fb_prof is not None
                            and _fb_prof.tuck_instinct > 0.40
                            and _fb_prof.in_tuck_zone(rx, attacks_right)):
                        band_near = r_home - r_sign * 14.0
                band_lo, band_hi = sorted((band_near, band_far))
                # Gentle pull into the band — a deliberate cut-inside run is
                # still followed, but a passive drift is corrected.
                if end_py < band_lo:
                    end_py += (band_lo - end_py) * 0.75
                elif end_py > band_hi:
                    end_py += (band_hi - end_py) * 0.75

        end_px = cls.clamp_x(end_px, attacks_right)
        return end_px, max(2, min(66, end_py))

    @classmethod
    def _pass_destination_to_target(
        cls, target_x: float, target_y: float,
        attacks_right: bool = True,
    ) -> Tuple[float, float]:
        """
        Checkpoint 10 — Attacking Matrix passes are aimed at a chosen target's
        live spatial position instead of a style-random vector. Small jitter is
        applied so a pass is never a perfect teleport onto the receiver's feet.
        """
        end_px = target_x + random.uniform(-1.5, 1.5)
        end_py = target_y + random.uniform(-1.5, 1.5)
        end_px = cls.clamp_x(end_px, attacks_right)
        return end_px, max(2, min(66, end_py))

    @classmethod
    def _generate_carry(
        cls, minute: int, team: str, player: PlayerProfile,
        x: float, y: float, phase: MatchPhase, game_state: GameState,
        profile: "TeamProfile"
    ) -> Tuple[MatchEvent, float, float, bool]:
        """Generate a carry event and return (event, new_x, new_y, success)."""
        dist, adv_ratio = cls._carry_distance_advance(player, x, profile)
        new_x = min(103, x + dist * adv_ratio)
        # PITCH WIDTH FIX: Increased from 4 + ball_control * 8 (max 12m) to
        # 6 + ball_control * 14 (max 20m) to allow wider diagonal runs.
        vert_range = 6 + (player.dna.technical.ball_control / 100) * 14
        new_y = y + (0.5 - random.random()) * vert_range
        new_y = max(2, min(66, new_y))

        is_progressive = new_x > x + 10
        drb_prob = DNAFactory.get_dribble_success_rate(player.dna)
        drb_prob = _get_soul_applicator().modify_dribble_success(player, drb_prob)
        success = random.random() < drb_prob

        event = cls.make_event(
            minute, EventType.CARRY, team, player.name,
            phase, game_state,
            location_x=x, location_y=y,
            end_x=new_x if success else x + dist * 0.3,
            end_y=new_y,
            outcome=success,
            metadata={"progressive": is_progressive, "distance": round(dist, 1)}
        )
        return event, new_x, new_y, success

    @classmethod

    @classmethod
    def _make_turnover(
        cls, minute: int, losing_team: str, player: PlayerProfile,
        x: float, y: float, phase: MatchPhase, game_state: GameState
    ) -> MatchEvent:
        return cls.make_event(
            minute, EventType.TURNOVER, losing_team, player.name,
            phase, game_state,
            location_x=x, location_y=y,
            outcome=False,
        )


# ─────────────────────────────────────────────
# 2. ATTACK CHAIN
# Chance creation → dribble/cross/through ball → shot → outcome
# ─────────────────────────────────────────────

class AttackChain(BaseChain):
    """
    Models the final third attack sequence:
        position in danger zone
        → chance created (key pass / cross / through ball / individual)
        → shot attempt
        → goal / save / miss / block
    """

    @staticmethod
    def _select_action_from_position(x: float, y: float, player_position: str, attacks_right: bool = True) -> str:
        """
        Geometry-aware shot selector.
        Rejects shooting from impossible positions/angles and intelligently
        biases toward crossing/passing instead.

        Checkpoint 10: the pure angle geometry is delegated to the Attacking
        Matrix (`attacking_matrix.shooting_angle_degrees`) so the pitch-level
        selector and the matrix share one source of truth. The "shoot"/"pass"/
        "cross"/"dribble" contract and branch structure are unchanged.

        Args:
            x: X-coordinate (meters from own goal line)
            y: Y-coordinate (meters from left touchline)
            player_position: Player's position (e.g., "LW", "RW", "ST")
            attacks_right: True when attacking the right-hand goal

        Returns:
            Action string: "shoot", "pass", "cross", or "dribble"
        """
        from attacking_matrix import shooting_angle_degrees

        goal_line = 105.0 if attacks_right else 0.0

        # Reject shooting behind goal line (x ≥ 105)
        if x >= 105.0:
            # Return "pass" if central, "cross" if wide
            if 20 < y < 48:
                return "pass"
            else:
                return "cross"

        # Angle to goal centre line (0° = central, 90° = level with goal line)
        angle_degrees = shooting_angle_degrees(x, y, attacks_right=attacks_right)

        # Acute angle logic: angle > 70° → return "cross" or "pass"
        if angle_degrees > 70.0:
            # Very acute angle, reject shooting
            if y < 20 or y > 48:
                return "cross"
            else:
                return "pass"

        # Moderate angle (60-70°): weighted choice 80% cross, 20% shot
        if angle_degrees > 60.0:
            choices = ["cross", "shot"]
            weights = [0.80, 0.20]
            return random.choices(choices, weights=weights, k=1)[0]

        # Wide positions near byline (x > 95, y < 20 or y > 48)
        if x > 95 and player_position in ["LW", "RW", "LB", "RB"]:
            if y < 20 or y > 48:
                choices = ["cross", "pass", "dribble", "shot"]
                weights = [0.65, 0.20, 0.10, 0.05]
                return random.choices(choices, weights=weights, k=1)[0]

        # ── CALIBRATION: distance propensity ────────────────────────
        # Real football shot frequency decays with distance: ~55-60% of
        # attempts come from inside the penalty area, ~20% from 16-20m,
        # ~15% from 20-25m, and only ~8-10% beyond 25m. The shot ORIGIN is
        # state-bound to the shooter's live position, so this selector is
        # the discipline gate: it turns pot-shots from range back into
        # passes instead of letting the release become a 30-60m cannonball.
        dist = BaseChain.goal_dist(x, y, attacks_right)
        if dist > 30.0:
            # beyond ~30m: a genuine rarity (~2% of attempts)
            return random.choices(["pass", "shoot"], weights=[0.98, 0.02], k=1)[0]
        if dist > 26.0:
            return random.choices(["pass", "shoot"], weights=[0.94, 0.06], k=1)[0]
        if dist > 22.0:
            return random.choices(["pass", "shoot"], weights=[0.85, 0.15], k=1)[0]
        if dist > 19.0:
            return random.choices(["pass", "shoot"], weights=[0.60, 0.40], k=1)[0]

        # Realistic shooting position
        return "shoot"

    @classmethod
    def generate(
        cls,
        minute: int,
        attacking_team: str,
        defending_team: str,
        att_players: List[PlayerProfile],
        def_players: List[PlayerProfile],
        team_profile: "TeamProfile",
        def_profile: "TeamProfile",
        state: MatchState,
        situation: SituationType,
        context_x: float = None,
        context_y: float = None,
        position_engine: Optional[PositionEngine] = None,
        attacks_right: bool = True,
        shooter_name: str = "",
        assister_name: str = "",
    ) -> ChainResult:
        result = ChainResult()
        phase  = state.phase
        gs     = state.game_state

        default_anchor = 88.0 if attacks_right else 17.0
        anchor_x = context_x if context_x is not None else default_anchor
        anchor_y = context_y if context_y is not None else 34.0
        # The man who actually had the ball, when the caller knows it. This is
        # the ONLY causal link the chain has to the possession sequence: with
        # it, the shot is caused by the play, so the pass that delivered the
        # ball really is the key pass and its passer really is the assist.
        # Without it the shooter is a role-and-distance-weighted DRAW over the
        # squad, which is how a shot came to be taken 22 m from the ball by a
        # man who never touched it, credited to a third player who appears
        # nowhere in the run-up.
        shooter = cls._named_shooter(att_players, shooter_name) or \
                  cls._pick_shooter(att_players, position_engine, anchor_x, anchor_y)
        # ── THE CREATOR OF THE CHANCE IS THE PLAYER WHO DELIVERED THE BALL ──
        # `assister_name` is the REAL passer, carried in from `PossessionChain`
        # (the previous value of `last_player` at the carrier change). It used
        # to come from `_pick_creator`, a role- and distance-weighted RANDOM
        # DRAW over the whole attacking squad. Because the key pass's ORIGIN is
        # this player's tracked position, the draw did not merely mis-name the
        # creator — it fabricated the GEOMETRY of the key pass, and the man it
        # picked had frequently never touched the ball.
        #
        # NO FALLBACK, and that is what preserves the relationship between the
        # three quantities the user cares about. `ChanceCreationLedger` awards
        # a chance created only to a completed PASS that results in a shot
        # ("Every completed pass that directly results in a shot = 1 Chance
        # Created"), and its `_find_setup_pass` deliberately returns None for a
        # dribble or a loose ball. Crediting the SHOOTER here as the creator of
        # his own solo run would make the engine's CHANCE_CREATED and the
        # ledger's backward scan disagree about the SAME shot — the two sources
        # would credit different players, or one and not the other. So with no
        # passer there is no creator and no CHANCE_CREATED event at all; the
        # shot remains in the timeline and the ledger still records it, with no
        # creator, exactly as Opta counts an unassisted strike.
        creator_name = cls._resolve_assister(att_players, assister_name, shooter)
        creator = cls._named_outfielder(att_players, creator_name) if creator_name else None
        gk      = cls._pick_gk(def_players)

        if not shooter:
            return result

        # ── POSSESSION-PHYSICS EPISODE (Checkpoint 27) ─────────────
        # A fresh 0.1 s clock for this chance: the pre-shot dribble, the
        # shot trajectory and the keeper's dive timing all resolve in
        # continuous motion on this episode.
        episode = PossessionEpisode()
        if position_engine is not None:
            episode.register([
                cls._moving_player(p, position_engine) for p in def_players
                if getattr(p, "position", "") != "GK"
            ])
            episode.register([cls._moving_player(gk, position_engine)]) if gk else None
            episode.register([cls._moving_player(shooter, position_engine)])

        # Shot location: use spatial anchor if provided (counter-attack continuity)
        in_attacking_half = (context_x is not None and context_x > 60) if attacks_right else (context_x is not None and context_x < 45)
        if context_x is not None and in_attacking_half:
            x_adv = random.uniform(-2, 5)
            x = cls.clamp_x(context_x + (x_adv if attacks_right else -x_adv), attacks_right)
            y = (context_y or 34) + random.uniform(-4, 4)
            y = max(5, min(63, y))
        elif (situation in (SituationType.OPEN_PLAY, SituationType.FAST_BREAK)
                and position_engine is not None
                and shooter.name in position_engine.states):
            # STATE-BOUND SHOT ORIGIN (2026-09-13): the shot starts at the
            # shooter's ACTUAL live position on the pitch — the distribution
            # emerges from play instead of being imposed by a calibrated
            # distance draw. Only the geometry gate below
            # (_select_action_from_position) filters it, so a midfield
            # release becomes a genuine 35-45m strike (handled by real xG /
            # keeper physics) and a behind-goal/acute-angle spot reverts to a
            # pass or cross. A small strike-pocket modifier (0.5-2.5m ahead
            # toward goal, minor lateral jitter) represents the contact point
            # being a touch in front of the planted foot, not a teleport.
            s_x, s_y = position_engine.get_position(shooter.name)
            strike_pocket = random.uniform(0.5, 2.5)
            x = cls.clamp_x(
                s_x + (strike_pocket if attacks_right else -strike_pocket),
                attacks_right,
            )
            y = max(3.0, min(65.0, s_y + random.uniform(-1.5, 1.5)))
            x, y = round(x, 1), round(y, 1)
        else:
            x, y = cls._shot_location(situation, team_profile, attacks_right=attacks_right)
        zone = PitchZone.xg_zone(x, y, attacks_right=attacks_right)

        # ── RANGE SETTLEMENT (CALIBRATION 2026-09-17) ────────────────
        # A committed final-third attack doesn't always reach the box —
        # real teams settle ~30% of open-play chances from 20-28m when the
        # box-entry is denied (shot quality is then priced honestly by the
        # XGEngine distance sharpening). Without this the funnel origin
        # (state-bound shooter positions already deep) over-concentrates in
        # the six-yard/apron ring and inflates xG far above the real band.
        settle_from_range = False
        if situation in (SituationType.OPEN_PLAY, SituationType.FAST_BREAK) \
                and random.random() < 0.30:
            settle_from_range = True
            target_dist = random.uniform(20.0, 28.0)
            gx_goal = 105.0 if attacks_right else 0.0
            x = round(gx_goal - (target_dist if attacks_right else -target_dist), 1)
            y = round(max(15.0, min(53.0, 34.0 + random.uniform(-9.0, 9.0))), 1)
            zone = PitchZone.xg_zone(x, y, attacks_right=attacks_right)

        # ── GEOMETRY-AWARE SHOT SELECTOR (Checkpoint 10) ─────────
        # Only shoot where the pitch geometry allows it: acute byline angles
        # are crossed/passed instead of blasted at the keeper from 2m out.
        # A cleared byline cross becomes a corner; otherwise possession turns
        # over with no shot.
        if situation != SituationType.PENALTY:
            shot_action = ("shoot" if settle_from_range
                           else cls._select_action_from_position(
                x, y, shooter.position, attacks_right=attacks_right
            ))
            if shot_action != "shoot":
                result.possession_lost = True
                if (shot_action == "cross"
                        and cls.goal_dist(x, y, attacks_right) < 8.0):
                    result.corner_won = True
                    result.corner_team = attacking_team
                return result

        if position_engine is not None:
            position_engine.record_touch(shooter.name, x, y, minute)
            # The creator's position was previously written as `x - 8, y` — an
            # invented spot 8 m behind the shooter, not anywhere he was
            # tracked. It looked like a real coordinate and it MOVED him there,
            # so every downstream position read inherited the fiction. Credit
            # the creator at his own tracked position; if he has none, credit
            # nothing. A touch we cannot place is not a touch we can place
            # somewhere convenient.
            if creator:
                _creator_at = position_engine.tracked_position(creator.name)
                if _creator_at is not None:
                    position_engine.record_touch(creator.name,
                                                 _creator_at[0], _creator_at[1], minute)

        # Body part (Checkpoint 6: now angle/channel-aware)
        body_part = cls._body_part(shooter, situation, y)

        # Is it a big chance?
        is_big = cls._is_big_chance(situation, zone, team_profile)

        # Under pressure?
        under_pressure = cls._is_under_pressure(x, def_profile, state)

        # ── CHANCE CREATION EVENT ─────────────────────────────
        # ── CHANCE CREATION EVENT ─────────────────────────────
        # This event used to carry `location_x = x - random.uniform(5, 20)`:
        # a start point drawn from the global football RNG, 5-20 m behind the
        # shot, matching no pass that player ever made (measured 15/15 outside
        # any real pass origin, all 15 inside the [5,20] draw band). The end
        # was the shot's taken location, so the pair asserted "the key pass
        # ended exactly where the shot was struck" — which is wrong whenever
        # the receiver carried, and the carry is modelled explicitly
        # elsewhere. Both ends are now the real tracked positions: the
        # creator's, and the shooter's. When the creator has never been
        # tracked the origin is omitted and `origin_known` says so, because
        # an absent field is honest and a plausible coordinate is not.
        creation_event = None
        # ── STREAM PARITY — DO NOT DELETE ─────────────────────────────────────
        # This draw used to be the argument of the fabricated origin
        # (`x - random.uniform(5, 20)`). Removing the fabrication removed the
        # draw with it, and that is NOT a cosmetic change: it shifts every
        # subsequent number in the global football stream, so the rest of the
        # match plays out differently. Proven, not assumed — restoring this one
        # line turns `test_match_crosses_stamped_geometrically` from fail to
        # pass with no behavioural change at all, and every calibration figure
        # in AGENTS.md was taken on the stream this preserves.
        #
        # It is drawn HERE, unconditionally, and deliberately OUTSIDE the
        # `if creator` guard below. It used to sit inside that block, which was
        # safe only because `_pick_creator` was a weighted draw that ALWAYS
        # returned somebody, so the guard never skipped it. Now that `creator`
        # is the real passer and is legitimately empty for an unassisted
        # strike, a draw left inside the guard would be skipped on exactly
        # those shots and silently desynchronise the stream for the rest of
        # the match. Consuming it at a fixed point in the sequence is the
        # whole point of ballast: same position in the stream, every time.
        #
        # The correct long-term fix is the documented one: make a match
        # reproducible from `random.seed` (module-level brain/mind caches
        # survive `simulate()`), then this shim can go. Until then it stays,
        # named so it is not mistaken for dead code and "cleaned up".
        _STREAM_PARITY_DRAW = random.uniform(5.0, 20.0)   # noqa: F841
        if creator and situation != SituationType.PENALTY:
            creation_type = cls._creation_type(creator, situation, team_profile)

            _origin = (position_engine.tracked_position(creator.name)
                       if position_engine is not None else None)
            _origin_known = _origin is not None
            # MatchEvent.location_x is typed float, so an untracked creator
            # cannot be represented as "absent" in memory. It falls back to
            # the shot's own coordinates, which is why `origin_known` is
            # stamped: a consumer that does not check the flag would read a
            # zero-length key pass, which is a different lie, not a smaller
            # one. Every live match registers the whole XI, so this branch is
            # a safety net rather than a path — assert that in the tests.
            _ox, _oy = (_origin if _origin_known else (x, y))

            creation_event = cls.make_event(
                minute,
                EventType.BIG_CHANCE_CREATED if is_big else EventType.CHANCE_CREATED,
                attacking_team, creator.name,
                phase, gs,
                secondary_player=shooter.name,
                location_x=_ox,
                location_y=_oy,
                end_x=x, end_y=y,
                situation=situation,
                xa=0.0,   # backfilled below once the final shot xG is known
                outcome=True,
                metadata={"creation_type": creation_type, "is_big_chance": is_big,
                          "origin_known": _origin_known,
                          "origin_source": "tracked_position" if _origin_known
                                           else "untracked_fallback"}
            )
            result.add(creation_event)

        # ── DRIBBLE BEFORE SHOT? ──────────────────────────────
        # STRICT OPTA/STATSBOMB DEFINITION: Dribble only counts if beating
        # an ENGAGED defender. Pre-shot scenarios inherently have defensive
        # pressure, so this is valid — but still needs defender reference.
        if (situation == SituationType.OPEN_PLAY
                and random.random() < shooter.dna.tendencies.attempts_dribble * 0.4):

            # Pick nearest defender for the dribble duel (geometric, not a
            # weighted pick) so the tackle race is against the real pressure.
            nearest_defender = None
            if position_engine is not None and def_players:
                _min_d = float("inf")
                for _d in def_players:
                    if getattr(_d, "position", "") == "GK":
                        continue
                    _pos = position_engine.get_position(_d.name)
                    _dist = math.hypot(_pos[0] - x, _pos[1] - y)
                    if _dist < _min_d:
                        _min_d = _dist
                        nearest_defender = _d
            if nearest_defender is None:
                nearest_defender = cls.pick_weighted(
                    def_players,
                    lambda p: {
                        "CB": 3.5, "CDM": 2.5, "CM": 2.0, "LB": 1.5, "RB": 1.5,
                    }.get(p.position, 0.5),
                )

            # The carrier targets the shooting pocket; the outcome is a
            # continuous contact-window race (movement + tackle radius),
            # NOT a dribble-success roll.
            drb_start_x = x - 5
            drb_end_x = x
            drb_physics = None
            if position_engine is not None:
                dribble_res = episode.resolve_dribble(
                    Vec2(x, y), Vec2(x + (3.0 if attacks_right else -3.0), y),
                    cls._moving_player(shooter, position_engine),
                    [
                        cls._moving_player(d, position_engine)
                        for d in def_players
                        if getattr(d, "position", "") != "GK"
                    ],
                )
                _drb_out = cls._dribble_confirmation_gate(
                    dribble_res.outcome, shooter, nearest_defender
                )
                success = _drb_out == "retained"
                drb_end_x = dribble_res.contact_point.x
                drb_physics = episode.physics_meta("pre_shot_dribble")
                is_heavy_touch = False
            else:
                drb_prob = DNAFactory.get_dribble_success_rate(shooter.dna)
                if nearest_defender:
                    def_tackle_skill = nearest_defender.dna.defending.tackling / 100.0
                    drb_prob -= (def_tackle_skill * 0.18)
                drb_prob = _get_soul_applicator().modify_dribble_success(shooter, drb_prob)
                drb_prob = max(0.15, min(0.80, drb_prob))
                success = random.random() < drb_prob
                ball_control_quality = shooter.dna.technical.ball_control / 100.0
                heavy_touch_risk = 0.15 * (1.0 - ball_control_quality)
                is_heavy_touch = random.random() < heavy_touch_risk

            if success and not is_heavy_touch:
                # TRUE SUCCESS: Beat defender, create better angle
                result.add(cls.make_event(
                    minute,
                    EventType.DRIBBLE_SUCCESS,
                    attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=nearest_defender.name if nearest_defender else None,
                    location_x=drb_start_x, location_y=y,
                    end_x=x, end_y=y,
                    outcome=True,
                    metadata={
                        "dribbled_past": True,
                        "pre_shot_dribble": True,
                        "beat_defender": nearest_defender.name if nearest_defender else "unknown",
                        "resolution": "contact_window" if position_engine is not None else "legacy_dribble",
                        "physics": drb_physics,
                    },
                ))
                # Successful dribble → better shot position
                drb_shot_adv = random.uniform(2, 6)
                x = cls.clamp_x(x + (drb_shot_adv if attacks_right else -drb_shot_adv), attacks_right)
                under_pressure = False  # Beat defender
            else:
                # FAILURE: Tackled or heavy touch
                fail_reason = "heavy_touch" if is_heavy_touch else "tackled"
                result.add(cls.make_event(
                    minute,
                    EventType.DRIBBLE_FAIL,
                    attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=nearest_defender.name if nearest_defender else None,
                    location_x=drb_start_x, location_y=y,
                    end_x=drb_end_x, end_y=y,
                    outcome=False,
                    metadata={
                        "failure_reason": fail_reason,
                        "pre_shot_dribble": True,
                        "resolution": "contact_window" if position_engine is not None else "legacy_dribble",
                        "physics": drb_physics,
                    },
                ))
                result.possession_lost = True
                # This early return happens BEFORE the final xG is computed
                # further down, which is where creation_event.xa normally
                # gets backfilled. Without this, every chance ending in a
                # failed dribble left xa stuck at its 0.0 placeholder.
                if creation_event is not None:
                    fallback_xg = XGEngine.calculate(
                        zone=zone, body_part=body_part, situation=situation,
                        under_pressure=under_pressure, is_big_chance=is_big,
                        shot_x=x, shot_y=y, attacks_right=attacks_right,
                    )
                    creation_event.xa = fallback_xg
                    result.xa_generated = fallback_xg
                return result

        # ── CALCULATE xG ─────────────────────────────────────
        xg = XGEngine.calculate(
            zone=zone,
            body_part=body_part,
            situation=situation,
            under_pressure=under_pressure,
            is_big_chance=is_big,
            first_time_shot=random.random() < 0.30,
            shot_x=x,
            shot_y=y,
            attacks_right=attacks_right,
        )
        # Checkpoint 6: angle-dependent finishing — a weak-foot shot forced
        # across the body from the wrong channel is genuinely harder than
        # the flat weak_foot attribute alone implies; a natural-side strike
        # gets a small quality bump. Headers/central shots are untouched.
        xg = round(xg * cls._angle_difficulty_mult(shooter, body_part, y), 4)
        if is_big:
            xg = _get_soul_applicator().modify_big_chance_conversion(shooter, xg)
            xg = round(xg, 4)
        result.xg_generated = xg

        # xA = xG exactly, evaluated at the FINAL shot quality — not a second,
        # independently-rolled calculation missing shot_x/shot_y and the
        # angle-difficulty multiplier. This is what made a passer's xA
        # diverge from the shooter's own xG on the same shot.
        if creation_event is not None:
            creation_event.xa = xg
            result.xa_generated = xg

        # ── SHOT OUTCOME (Checkpoint 27) ─────────────────────
        # Geometry is the outcome authority for shots: the shooter aims at a
        # goal-plane target with skill-driven placement, the ball flies a 3D
        # trajectory, and the keeper's reaction + dive envelope (reach, dive
        # speed, vertical reach) decides goal vs save. On-target/off-target/
        # woodwork/block are DERIVED from the crossing geometry, not rolled.
        gk_motion = (
            cls._gk_shot_motion(gk, x, y, attacks_right)
            if gk is not None else None
        )
        blockers = []
        if position_engine is not None:
            blockers = [
                cls._moving_player(d, position_engine)
                for d in def_players
                if getattr(d, "position", "") != "GK"
            ]
        flight = aim_shot_flight(
            shooter.dna, x, y, body_part, attacks_right,
            under_pressure=under_pressure,
        )
        episode.set_ball(x, y)
        shot_res = episode.resolve_shot(flight, gk_motion, blockers, attacks_right=attacks_right)
        shot_out = shot_res.outcome  # goal | saved | woodwork | wide | blocked
        shot_physics = episode.physics_meta("shot")
        # Opta-style shot speed: the launch velocity baked into the flight
        # (20-34 m/s for a struck ball; headers are far slower). Reported as
        # the speed the ball leaves the boot/noggin — not the average over the
        # arc (which is ball_speed_mps, kept below for continuity).
        launch_speed = getattr(flight, "initial_speed", 0.0) or 0.0
        if body_part == "head" or launch_speed < 12.0:
            launch_speed = _shot_speed_for_player(shooter, body_part)
        shot_physics.update({
            "flight_s": round(shot_res.flight_time, 2),
            "shot_speed_mps": round(launch_speed, 2),
            "shot_speed_kmh": round(launch_speed * 3.6, 1),
            "flight_time_s": round(flight.duration, 2),
            "target_y": round(shot_res.goal_point.y, 2),
            "target_z": round(shot_res.goal_point.z, 2),
            "ball_speed_mps": round(math.hypot(
                flight.target.x - flight.start.x,
                flight.target.y - flight.start.y) / max(0.01, flight.duration), 2),
        })
        shot_x_end = shot_res.goal_point.x
        shot_y_end = shot_res.goal_point.y

        # ── KEEPER SPILL (CALIBRATION 2026-09-17) ─────────────────────
        # Real keepers parry far more than they catch, and a parried strike
        # squirms home disturbingly often (~41% of on-target shots become
        # goals league-wide). PLOFA's geometry saved ~78% of on-target beats
        # (goals/SOT ~22%) because the dive envelope treats every parry as a
        # clean outcome. A parried (not smothered) attempt that the physics
        # sent to "saved" now has a real spill chance — the goal is scored
        # through the normal conversion gate below, which keeps finishing and
        # chance quality meaningful (a scuffed apron strike that was parried
        # still faces its worldie-skimmer).
        # CALIBRATION (2026-09-17): spill 0.42 -> 0.47 to lift goals/xG
        # from ~1.09 toward the real ~1.148 (checked against the corrected
        # deterministic probe corpus).
        # CALIBRATION (2026-09-17): spill 0.42 -> 0.47 to lift goals/xG
        # from ~1.09 toward the real ~1.148 (measured on the corrected
        # deterministic probe corpus).
        spill_goal = False
        if shot_out == "saved" and shot_res.goalkeeper is not None:
            keep_dist = math.hypot(
                shot_y_end - shot_res.goalkeeper.position.y,
                shot_x_end - shot_res.goalkeeper.position.x,
            )
            parried = keep_dist > (getattr(shot_res.goalkeeper, "control_radius", 0.8) + 0.4)
            goalline_smother = abs(shot_x_end - flight.start.x) < 2.0
            if parried and not goalline_smother and random.random() < 0.47:
                spill_goal = True
                shot_out = "goal"

        if shot_out == "goal":
            # ── HYBRID CONVERSION GATE (Checkpoint 28) ─────────────────
            # Geometry is still the PRIMARY authority: it has already decided
            # this shot placed the ball where the keeper's dive envelope could
            # NOT reach it — the optimal, "keeper beaten" case. We keep that
            # as-is for genuine clearcuts. But a scuffed low-quality effort
            # (tiny xG) that STARTS as a clean-geometry strike can still be
            # clawed back by the keeper ("worldie"): the final goal/save now
            # also respects chance quality and finishing, so high-value
            # chances and clinical finishers convert and junk shots don't
            # bang in at the same fixed rate (which is what let blowouts and
            # high-volume teams run away).
            finishing = float(getattr(
                getattr(getattr(shooter, "dna", None), "technical", None),
                "finishing", 50.0))
            fin_signal = 0.80 + (finishing / 100.0) * 0.40   # 0.82 -> 1.20
            if is_big:
                fin_signal += 0.10
            gate_xg = max(0.0, min(0.99, xg * fin_signal))
            # Captured inspite of the geometry (keeper worldie) probability
            # is low for quality chances, high for hope-shots.
            worldie_p = max(0.0, 1.0 - gate_xg)
            # CALIBRATION (2026-09-17): conversion gate re-tuned. The old
            # `gate_xg >= 0.40 or random >= worldie_p*0.5` clawed back ~46%
            # of geometry goals on sub-0.2 xG chances — which is why goals
            # (1.06/team) trailed xG (1.84/team) far below the real
            # goals-for-xG ratio (~1.15). xG is now priced honestly per shot
            # (zone base + distance sharpening), so a shot that BEATS the
            # keeper geometrically should score; only scuffed hope-shots keep
            # a real worldie risk. Threshold dropped to 0.15 and the clawback
            # halved.
            converted = gate_xg >= 0.15 or random.random() >= worldie_p * 0.30

            if converted:
                result.shot_on_target = True
                result.add(cls.make_event(
                    minute, EventType.SHOT_ON_TARGET, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=gk.name if gk else None,
                    location_x=x, location_y=y,
                    end_x=shot_x_end, end_y=shot_y_end,
                    situation=situation, xg=xg, body_part=body_part,
                    outcome=True,
                    metadata={"zone": zone, "is_big_chance": is_big,
                              "physics": shot_physics, "resolution": "trajectory"}
                ))
                result.goal_scored   = True
                result.goal_team     = attacking_team
                result.goal_scorer   = shooter.name
                result.goal_assistant = cls._resolve_assister(att_players, assister_name, shooter)

                result.add(cls.make_event(
                    minute, EventType.GOAL, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=(cls._resolve_assister(
                        att_players, assister_name, shooter) or None),
                    location_x=x, location_y=y,
                    end_x=shot_x_end, end_y=shot_y_end,
                    situation=situation,
                    xg=xg,
                    body_part=body_part,
                    outcome=True,
                    metadata={
                        "zone": zone,
                        "is_big_chance": is_big,
                        "body_part": body_part,
                        "physics": shot_physics,
                        "resolution": "trajectory",
                    }
                ))
            else:
                # Keeper clawed it back — still an on-target save (worldie).
                result.shot_on_target = True
                result.add(cls.make_event(
                    minute, EventType.SHOT_ON_TARGET, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=gk.name if gk else None,
                    location_x=x, location_y=y,
                    end_x=shot_x_end, end_y=shot_y_end,
                    situation=situation, xg=xg, body_part=body_part,
                    outcome=True,
                    metadata={"zone": zone, "is_big_chance": is_big,
                              "physics": shot_physics, "resolution": "trajectory",
                              "worldie": True}
                ))
                result.add(cls.make_event(
                    minute, EventType.SAVE, defending_team,
                    gk.name if gk else "GK",
                    phase, gs,
                    secondary_player=shooter.name,
                    location_x=x, location_y=y,
                    end_x=shot_x_end, end_y=shot_y_end,
                    xg=xg,
                    outcome=True,
                    metadata={"zone": zone, "is_big_chance": is_big,
                              "physics": shot_physics, "resolution": "trajectory",
                              "worldie": True}
                ))

        elif shot_out == "saved":
            result.shot_on_target = True
            result.add(cls.make_event(
                minute, EventType.SHOT_ON_TARGET, attacking_team, shooter.name,
                phase, gs,
                secondary_player=gk.name if gk else None,
                location_x=x, location_y=y,
                end_x=shot_x_end, end_y=shot_y_end,
                situation=situation,
                xg=xg,
                body_part=body_part,
                outcome=True,
                metadata={"zone": zone, "is_big_chance": is_big,
                          "physics": shot_physics, "resolution": "trajectory"}
            ))

            # Save — reach/dive geometry decided the keeper GOT there. Whether
            # it's a parry or a clean catch is derived from how central the
            # shot crossed relative to the keeper's body, not a flat roll.
            is_parry = True
            if gk_motion is not None:
                keep_dist = math.hypot(
                    shot_y_end - gk_motion.position.y,
                    shot_x_end - gk_motion.position.x,
                )
                is_parry = keep_dist > (gk_motion.control_radius + 0.6)
            save_type = "parry" if is_parry else "catch"
            is_goalline_save = abs(shot_x_end - flight.start.x) < 2.0
            angle_from_center = abs(y - 34.0)
            is_wide_shot = angle_from_center > 15.0
            is_close_range = x > 95.0 if attacks_right else x < 10.0

            shot_difficulty = xg * (1.0 + (0.4 if under_pressure else 0.0))
            corner_from_parry = False
            if is_parry:
                # CALIBRATION (2026-09-17): parry->corner base lowered 0.55
                # -> 0.38 -> 0.34 -> 0.30 (real ~0.40) with tighter range/
                # angle adders and a lower cap so corner volume lands near
                # the real ~4.6/match.
                corner_prob = 0.30
                if is_wide_shot:
                    corner_prob += 0.16
                if is_close_range:
                    corner_prob += 0.12
                if shot_difficulty > 0.5:
                    corner_prob += 0.08
                corner_from_parry = random.random() < min(0.76, corner_prob)

            result.add(cls.make_event(
                minute, EventType.SAVE, defending_team,
                gk.name if gk else "GK",
                phase, gs,
                secondary_player=shooter.name,
                location_x=x, location_y=y,
                end_x=shot_x_end, end_y=shot_y_end,
                xg=xg,
                outcome=True,
                metadata={
                    "zone": zone,
                    "is_big_chance": is_big,
                    "save_type": save_type,
                    "parried": is_parry,
                    "deflection_angle": angle_from_center if is_parry else 0,
                    "goalline_save": is_goalline_save,
                    "physics": shot_physics,
                    "resolution": "trajectory",
                }
            ))

            if corner_from_parry:
                result.corner_won = True
                result.corner_team = attacking_team
                corner_y = 0.0 if y < 34.0 else 68.0
                result.add(cls.make_event(
                    minute, EventType.CORNER_WON, attacking_team, shooter.name,
                    phase, gs,
                    location_x=105 if attacks_right else 0,
                    location_y=corner_y,
                    metadata={"from_save_deflection": True, "gk_parry": True}
                ))

        elif shot_out == "woodwork":
            rebound_in = random.random() < cls._woodwork_rebound_prob(
                x, y, shot_x_end, shot_y_end,
                gk_motion, xg, attacks_right,
            )
            # CALIBRATION (2026-09-17): woodwork->corner 0.60 -> 0.45 -> 0.40 -> 0.36.
            goes_for_corner = not rebound_in and random.random() < 0.36

            result.add(cls.make_event(
                minute, EventType.HIT_WOODWORK, attacking_team, shooter.name,
                phase, gs,
                secondary_player=gk.name if gk else None,
                location_x=x, location_y=y,
                end_x=shot_x_end, end_y=shot_y_end,
                situation=situation, xg=xg, body_part=body_part,
                outcome=rebound_in,
                metadata={
                    "zone": zone,
                    "is_big_chance": is_big,
                    "rebound_in": rebound_in,
                    "corner_awarded": goes_for_corner,
                    "physics": shot_physics,
                    "resolution": "trajectory",
                }
            ))
            if rebound_in:
                result.goal_scored    = True
                result.goal_team      = attacking_team
                result.goal_scorer    = shooter.name
                result.goal_assistant = cls._resolve_assister(att_players, assister_name, shooter)
                result.add(cls.make_event(
                    minute, EventType.GOAL, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=(cls._resolve_assister(
                        att_players, assister_name, shooter) or None),
                    location_x=x, location_y=y,
                    end_x=shot_x_end, end_y=shot_y_end,
                    situation=situation, xg=xg, body_part=body_part,
                    outcome=True,
                    metadata={"zone": zone, "is_big_chance": is_big,
                              "body_part": body_part, "via_woodwork": True,
                              "physics": shot_physics}
                ))
            elif goes_for_corner:
                result.corner_won = True
                result.corner_team = attacking_team
                corner_y = 0.0 if y < 34.0 else 68.0
                result.add(cls.make_event(
                    minute, EventType.CORNER_WON, attacking_team, shooter.name,
                    phase, gs,
                    location_x=105 if attacks_right else 0,
                    location_y=corner_y,
                    metadata={"from_woodwork": True}
                ))

        elif shot_out == "blocked":
            blocker = shot_res.blocker.player if shot_res.blocker is not None else None
            result.add(cls.make_event(
                minute, EventType.SHOT_BLOCKED, attacking_team, shooter.name,
                phase, gs,
                secondary_player=getattr(blocker, "name", None),
                location_x=x, location_y=y,
                end_x=shot_res.goal_point.x, end_y=shot_res.goal_point.y,
                xg=xg,
                outcome=False,
                metadata={"physics": shot_physics, "resolution": "trajectory"}
            ))
            # CALIBRATION (2026-09-17): blocked-shot corner 0.55 -> 0.45 -> 0.42 -> 0.38.
            corner_awarded = random.random() < 0.38
            if corner_awarded:
                result.corner_won = True
                result.corner_team = attacking_team
                corner_y = 0.0 if y < 34.0 else 68.0
                result.add(cls.make_event(
                    minute, EventType.CORNER_WON, attacking_team, shooter.name,
                    phase, gs,
                    location_x=105 if attacks_right else 0,
                    location_y=corner_y,
                    metadata={"from_shot_block": True}
                ))
            elif blocker:
                result.add(cls.make_event(
                    minute, EventType.BALL_RECOVERY, defending_team, blocker.name,
                    phase, gs,
                    location_x=x, location_y=y,
                    outcome=True,
                    metadata={"after_block": True}
                ))

        else:  # wide
            result.add(cls.make_event(
                minute, EventType.SHOT_OFF_TARGET, attacking_team, shooter.name,
                phase, gs,
                location_x=x, location_y=y,
                end_x=shot_x_end, end_y=shot_y_end,
                xg=xg,
                outcome=False,
                metadata={"physics": shot_physics, "resolution": "trajectory"}
            ))

        # ── PHYSICS TRACE STAMP (Checkpoint 27) ─────────────────────
        if episode.trace and result.events:
            last_evt = result.events[-1]
            last_evt.metadata["physics"] = shot_physics
            last_evt.metadata["motion_trace"] = episode.condensed_trace()

        # ── SHOT-SPEED STAMP (Opta-style) ─────────────────────────
        # Every shot event carries the launch velocity + flight time at the
        # metadata TOP level (consistent with corner/penalty/FK shots below),
        # so the average-shot-speed aggregator reads one source of truth.
        for _ev in result.events:
            if _ev.event_type in SHOT_SPEED_EVENT_TYPES:
                _stamp_shot_speed(_ev, launch_speed, flight.duration)

        # ── OUT-OF-BOUNDS DETECTION (Checkpoint 7) ────────────────
        # Check if shot went out of bounds and emit appropriate restart
        # Get final ball position from last event
        if result.events:
            last_event = result.events[-1]
            final_x = last_event.end_x if last_event.end_x is not None else last_event.location_x
            final_y = last_event.end_y if last_event.end_y is not None else last_event.location_y
            
            # Only check if no goal scored and no corner won
            if not result.goal_scored and not result.corner_won:
                # Throw-in detection: (y < 2 or y > 66) AND x < 105
                if (final_y < 2.0 or final_y > 66.0) and final_x < 105.0:
                    result.restart_required = True
                    result.restart_type = "throw_in"
                    # Award to team that DIDN'T touch last (defending team)
                    result.restart_team = defending_team
                    result.restart_x = final_x
                    result.restart_y = 0.0 if final_y < 2.0 else 68.0
                    result.possession_lost = True
                    
                # Goal kick detection: x ≥ 105 AND (y < 30.34 or y > 37.66)
                elif final_x >= 105.0 and (final_y < 30.34 or final_y > 37.66):
                    result.restart_required = True
                    result.restart_type = "goal_kick"
                    # Award to defending team
                    result.restart_team = defending_team
                    result.restart_x = random.uniform(8, 18)
                    result.restart_y = 34.0
                    result.possession_lost = True

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)
        return result

    # ── HELPERS ───────────────────────────────────────────────

    @classmethod
    def _named_outfielder(
        cls, players: List[PlayerProfile], name: str,
    ) -> Optional[PlayerProfile]:
        """Resolve an explicitly-named outfielder, or None.

        The shared lookup behind both `_named_shooter` and `_resolve_assister`.
        Deliberately strict: the name must match a profile in the squad we were
        GIVEN, and must be an outfielder. A stale name (a substituted player, a
        team-mate absent from this list) returns None, so the caller falls back
        rather than crediting a ghost.
        """
        if not name:
            return None
        for p in cls._outfield_players(players):
            if getattr(p, "name", "") == name:
                return p
        return None

    @classmethod
    def _named_shooter(
        cls, players: List[PlayerProfile], name: str,
    ) -> Optional[PlayerProfile]:
        """Resolve an explicitly-named shooter, or None.

        Deliberately strict: the name must match a profile in the squad we were
        GIVEN, and must be an outfielder. A stale name (a substituted player, a
        team-mate absent from this list) returns None, so the caller falls back
        to the weighted pick rather than shooting with a ghost.
        """
        return cls._named_outfielder(players, name)

    @classmethod
    def _resolve_assister(
        cls, players: List[PlayerProfile], name: str,
        shooter: Optional[PlayerProfile],
    ) -> str:
        """The real assister's name, or "" for a genuinely unassisted goal.

        Deliberately NO fallback. The old behaviour was
        `creator.name if creator else ""` where `creator` was a role- and
        distance-weighted random draw over the squad — so the engine named an
        assister who frequently appears nowhere in the run-up, and the Goals
        sheet disagreed with the chance-creation ledger by construction.

        Returning "" is not a gap, it is the answer: a man who wins the ball and
        dribbles it in from 40 yards has NO assist, and that is common in real
        football. An absent field is omitted, never invented.

        The name must resolve to an outfielder in the squad we were given, and
        must not be the shooter (nobody assists himself).
        """
        if not name:
            return ""
        if shooter is not None and name == getattr(shooter, "name", ""):
            return ""
        found = cls._named_outfielder(players, name)
        return getattr(found, "name", "") if found else ""

    @classmethod
    def _pick_shooter(
        cls, players: List[PlayerProfile],
        position_engine: Optional[PositionEngine] = None,
        x: float = 88.0, y: float = 34.0,
    ) -> Optional[PlayerProfile]:
        return cls.pick_weighted_spatial(
            cls._outfield_players(players),
            lambda p: {
                "ST": 6.0, "CF": 5.5, "LW": 4.0, "RW": 4.0,
                "CAM": 3.0, "CM": 1.5, "CDM": 0.5,
                "CB": 0.3, "LB": 0.4, "RB": 0.4, "GK": 0.0,
            }.get(p.position, 1.0),
            position_engine, x, y,
        )

    @classmethod
    def _pick_gk(cls, def_players: List[PlayerProfile]) -> Optional[PlayerProfile]:
        gks = [p for p in def_players if p.position == "GK"]
        return gks[0] if gks else None

    @classmethod
    def _pick_blocker(cls, def_players: List[PlayerProfile]) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            def_players,
            lambda p: {
                "CB": 4.0, "CDM": 3.0, "CM": 2.5,
                "LB": 2.0, "RB": 2.0, "GK": 0.0,
            }.get(p.position, 1.0)
        )

    @classmethod
    def _shot_location(cls, situation: SituationType, profile: "TeamProfile",
                        attacks_right: bool = True) -> Tuple[float, float]:
        if situation == SituationType.PENALTY:
            return (cls.penalty_spot_x(attacks_right), 34.0)
        if situation == SituationType.CORNER:
            return PitchZone.random_in(
                cls.mirror_x(88, attacks_right) if attacks_right else cls.mirror_x((88, 102), attacks_right),
                (24, 44)
            ) if attacks_right else (
                round(random.uniform(3, 17), 1),
                round(random.uniform(24, 44), 1),
            )
        m = lambda lo, hi: (105 - hi, 105 - lo) if not attacks_right else (lo, hi)
        if situation == SituationType.DIRECT_FREEKICK:
            lo, hi = m(72, 88)
            return PitchZone.random_in((lo, hi), (20, 48))
        if situation in (SituationType.FAST_BREAK, SituationType.OPEN_PLAY):
            # Realistic shot-distance distribution: most shots are struck from
            # 12-22m (peak ~15m), tapering toward point-blank and long range.
            # A straight uniform(78,103) over-fed the geometry point-blank
            # chances at x~101-103 where the flight time (0.2s) makes the
            # keeper unreachable — producing near-automatic goals.
            if situation == SituationType.FAST_BREAK:
                dist = max(7.0, random.gauss(11.5, 4.0))
            else:
                dist = max(7.0, min(32.0, -14.0 * math.log(1.0 - random.random()) + 3.0))
            dist = min(32.0, dist)
            x = (105.0 - dist) if attacks_right else dist
            spread = 3.0 + dist * 0.33
            y = 34.0 + random.uniform(-spread, spread)
            y = max(3.0, min(65.0, y))
            return round(x, 1), round(y, 1)

    @classmethod
    def _body_part(cls, shooter: PlayerProfile, situation: SituationType, y: float = 34.0) -> str:
        pos = shooter.position
        if situation == SituationType.CORNER:
            return random.choices(["head", "right_foot", "left_foot"], weights=[60, 25, 15])[0]
        if pos in ["CB", "LB", "RB"] and situation in [SituationType.CORNER, SituationType.CROSSED_FREEKICK]:
            return random.choices(["head", "right_foot", "left_foot"], weights=[70, 20, 10])[0]

        # Footedness effect
        foot = shooter.dna.preferred_foot
        wf   = shooter.dna.technical.weak_foot / 100.0

        # Checkpoint 6: channel/angle awareness. A right-footer cutting in
        # from the LEFT channel (y < 34, wide left) gets a natural,
        # open-body strike onto their right foot — the classic inside-cut.
        # A left-footer gets that same natural angle from the RIGHT channel
        # (y > 34). The opposite channel forces an across-body or weak-foot
        # connection, which is genuinely harder — reflected here as a
        # reduced natural-foot weight (pushing more shots onto the weak
        # foot or a header) rather than pretending angle doesn't exist.
        channel_offset = y - 34.0   # negative = left channel, positive = right
        if foot == "right":
            natural_side = channel_offset < -4.0     # left channel favors right foot
            awkward_side = channel_offset > 4.0       # right channel is across-body for a righty
        else:
            natural_side = channel_offset > 4.0        # right channel favors left foot
            awkward_side = channel_offset < -4.0

        natural_w = 55 if natural_side else (35 if awkward_side else 45)
        weak_w = 45 * wf if not awkward_side else 60 * wf

        if foot == "right":
            return random.choices(["right_foot", "left_foot", "head"],
                                   weights=[natural_w, weak_w, 20])[0]
        else:
            return random.choices(["left_foot", "right_foot", "head"],
                                   weights=[natural_w, weak_w, 20])[0]

    @classmethod
    def _woodwork_rebound_prob(
        cls,
        shot_x: float,
        shot_y: float,
        shot_x_end: float,
        shot_y_end: float,
        gk_motion,
        xg: float,
        attacks_right: bool,
    ) -> float:
        """
        Geometry-aware rebound probability for woodwork hits.

        Factors:
        - GK proximity to the shot impact point (closer = less rebound)
        - Shot xG quality (higher quality = harder to rebound from)
        - Shot distance from goal (closer range = more rebound)
        """
        rebound_base = 0.15
        if gk_motion is not None:
            gk_dist = math.hypot(
                shot_x_end - gk_motion.position.x,
                shot_y_end - gk_motion.position.y,
            )
            if gk_dist < 2.0:
                rebound_base -= 0.10
            elif gk_dist < 4.0:
                rebound_base -= 0.05
        if xg > 0.30:
            rebound_base += 0.05
        dist = math.hypot(shot_x_end - shot_x, shot_y_end - shot_y)
        if dist < 8.0:
            rebound_base += 0.05
        return max(0.05, min(0.35, rebound_base))

    @classmethod
    def _angle_difficulty_mult(cls, shooter: PlayerProfile, body_part: str, y: float) -> float:
        """
        Checkpoint 6: a left-footer's chances of scoring with their right
        foot (or vice versa) genuinely depend on the angle they're shooting
        from, not just a flat weak-foot number. Shooting off your natural
        side (right foot from the left channel, left foot from the right
        channel) is the easy, open-body strike — full quality. Being forced
        across your body (weak foot AND the wrong channel for it) is a real
        finishing penalty on top of the raw weak_foot attribute. Headers and
        central shots are unaffected — angle only matters for foot choice.
        """
        if body_part == "head":
            return 1.0
        foot = shooter.dna.preferred_foot
        wf = shooter.dna.technical.weak_foot / 100.0
        channel_offset = y - 34.0
        is_weak_foot_shot = (
            (body_part == "left_foot" and foot == "right") or
            (body_part == "right_foot" and foot == "left")
        )
        if not is_weak_foot_shot:
            # Natural foot — small bonus if also the natural open-body
            # channel for it, neutral otherwise.
            if foot == "right" and channel_offset < -4.0:
                return 1.06
            if foot == "left" and channel_offset > 4.0:
                return 1.06
            return 1.0
        # Weak-foot shot: penalty scales with how far it is from that
        # foot's natural channel, softened by how good the weak foot is.
        if foot == "right":
            off_natural = channel_offset < -4.0   # weak (left) foot but on the right-footer's easy side
        else:
            off_natural = channel_offset > 4.0
        base_penalty = 0.80 if off_natural else 0.92
        # A strong weak-foot (high weak_foot attribute) closes most of the gap
        return round(min(1.0, base_penalty + (1.0 - base_penalty) * wf), 3)

    @classmethod
    def _is_big_chance(cls, situation: SituationType, zone: str, profile: "TeamProfile") -> bool:
        base = profile.big_chance_ratio
        if situation == SituationType.PENALTY:      return True
        if situation == SituationType.FAST_BREAK:   base *= 1.3
        if zone == "six_yard_box":                  base *= 1.4
        if zone in ("inside_box", "penalty_area"):  base *= 1.1
        return random.random() < min(0.80, base)

    @classmethod
    def _is_under_pressure(cls, x: float, def_profile: "TeamProfile", state: MatchState) -> bool:
        base = def_profile.press_intensity * 0.5
        if x >= 83:  base *= 0.7  # Hard to press effectively in the box
        return random.random() < base

    @classmethod
    def _creation_type(cls, creator: PlayerProfile, situation: SituationType, profile: "TeamProfile") -> str:
        if situation in (SituationType.CORNER, SituationType.CROSSED_FREEKICK):
            return "cross"
        if situation == SituationType.DIRECT_FREEKICK:
            return "free_kick"
        # Open play creation
        if creator.position in ("LB", "RB"):
            return random.choices(["cross", "through_ball", "key_pass"], weights=[55, 15, 30])[0]
        if creator.position in ("LW", "RW"):
            # Modern wingers create from WIDE — cross/cut-back dominates.
            # The middle of the pitch is always full; the winger's creative
            # output comes from the touchline→byline corridor (Saka, Vini,
            # Salah all create their key passes from wide positions).
            return random.choices(["cross", "cut_back", "key_pass"], weights=[50, 30, 20])[0]
        return random.choices(["key_pass", "through_ball", "cut_back"], weights=[50, 30, 20])[0]

    @classmethod
    def _shot_on_target_prob(
        cls, xg: float, shooter: PlayerProfile, situation: SituationType
    ) -> float:
        # xG drives on-target probability
        base = 0.20 + (xg * 0.65)
        # Composure and finishing improve it
        comp = shooter.dna.mental.composure / 100.0
        fin  = shooter.dna.technical.finishing / 100.0
        base += (comp + fin) * 0.05
        base = _get_soul_applicator().modify_shot_quality(shooter, base)
        if WeatherPhysics.enabled:
            base *= WeatherPhysics.shot_accuracy_mult()
        # Penalty: always on target (unless catastrophic miss)
        if situation == SituationType.PENALTY:
            return min(0.97, base * 1.5)
        return min(0.92, max(0.08, base))


# ─────────────────────────────────────────────
# 3. SET PIECE CHAIN
# Corners, free kicks, penalties — own sub-simulations
# ─────────────────────────────────────────────

class SetPieceChain(BaseChain):
    """
    Models set piece sequences in full.

    Corner → delivery → aerial duel → headed shot / clearance / second ball
    Free kick → direct / crossed → shot / header / deflection
    Penalty → spot kick ritual → goal / save / miss
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        attacking_team: str,
        defending_team: str,
        att_players: List[PlayerProfile],
        def_players: List[PlayerProfile],
        state: MatchState,
        situation: SituationType,
        attacks_right: bool = True,
        context_x: Optional[float] = None,
        context_y: Optional[float] = None,
        position_engine=None,
        routine: Optional[SetPieceRoutine] = None,
    ) -> ChainResult:
        if situation == SituationType.PENALTY:
            return cls._penalty_chain(minute, attacking_team, defending_team,
                                       att_players, def_players, state, attacks_right)
        elif situation == SituationType.CORNER:
            return cls._corner_chain(minute, attacking_team, defending_team,
                                      att_players, def_players, state, attacks_right,
                                      position_engine=position_engine,
                                      routine=routine)
        else:
            return cls._freekick_chain(minute, attacking_team, defending_team,
                                        att_players, def_players, state, situation, attacks_right,
                                        context_x=context_x, context_y=context_y,
                                        position_engine=position_engine,
                                        routine=routine)

    @classmethod
    def _penalty_chain(cls, minute, att_team, def_team,
                        att_players, def_players, state,
                        attacks_right=True) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        ps_x = cls.penalty_spot_x(attacks_right)

        taker = cls._pick_sp_taker(att_players, situation="penalty")
        gk    = cls._pick_gk_player(def_players)

        result.add(cls.make_event(
            minute, EventType.PENALTY_WON, att_team, taker.name,
            phase, gs, location_x=ps_x,
            location_y=PitchZone.PENALTY_SPOT[1]
        ))

        pen_quality = taker.dna.technical.penalty_taking / 100.0
        pen_prob    = 0.60 + (pen_quality * 0.30)

        # Opta-style shot speed: penalties are struck at full power from the
        # spot (≈11 m to the goal line at 20-35 m/s → sub-half-second flight).
        pen_launch = _shot_speed_for_player(taker, "foot")
        pen_flight = round(11.0 / pen_launch, 2)
        pen_speed_md = {
            "pen_prob": round(pen_prob, 3),
            "shot_speed_mps": round(pen_launch, 2),
            "shot_speed_kmh": round(pen_launch * 3.6, 1),
            "shot_flight_s": pen_flight,
        }

        if gk:
            gk_reflex = gk.dna.gk_attrs.reflexes / 100.0
            pen_prob -= gk_reflex * 0.05

        pen_prob = max(0.55, min(0.92, pen_prob))

        if random.random() < pen_prob:
            result.add(cls.make_event(
                minute, EventType.PENALTY_SCORED, att_team, taker.name,
                phase, gs,
                location_x=ps_x,
                location_y=PitchZone.PENALTY_SPOT[1],
                xg=0.79, outcome=True,
                body_part=random.choice(["right_foot", "left_foot"]),
                metadata=dict(pen_speed_md)
            ))
            result.goal_scored    = True
            result.goal_team      = att_team
            result.goal_scorer    = taker.name
            result.goal_assistant = ""
            result.xg_generated   = 0.79
        else:
            result.add(cls.make_event(
                minute, EventType.PENALTY_MISSED, att_team, taker.name,
                phase, gs,
                location_x=ps_x,
                location_y=PitchZone.PENALTY_SPOT[1],
                xg=0.79, outcome=False,
                metadata={**pen_speed_md, "saved_by": gk.name if gk else "GK"}
            ))
            if gk:
                result.add(cls.make_event(
                    minute, EventType.SAVE, def_team, gk.name,
                    phase, gs, xg=0.79, outcome=True,
                    location_x=ps_x,
                    location_y=PitchZone.PENALTY_SPOT[1],
                    metadata={"penalty_save": True}
                ))

        return result

    # ── CORNER BOX OCCUPANCY (2026-10-02) ─────────────────────────────────
    #
    # `_corner_chain` builds a genuine corner-defence grid via
    # `SetPieceMarkingEngine` — aerial man on the top threat, a first man on
    # the near-post line, GK on the defended side — and then applied it to
    # exactly THREE players: the receiver, one marker, and the keeper. The
    # other eighteen outfielders were never repositioned and held the live
    # defensive shape, which sits OUTSIDE the box. So a corner was resolved as
    # a 1-v-1 aerial duel in an empty box, and nothing suppressed the header
    # except the keeper. Geometry alone does not fix that: what suppresses
    # corner goals in real football is a pack of seven defenders dragging six
    # attackers, so the pack has to exist before the crossing geometry means
    # anything.
    #
    # Real occupancy, which is the target: 5-7 attackers between the penalty
    # spot and the six-yard line, one on each post, a couple at the top of the
    # box for the second phase; 6-8 defenders, each GOAL-SIDE of the man he is
    # marking; the keeper on his line.
    #
    # Returns {name: (x, y)} in RAW pitch coordinates and writes NOTHING, so
    # the decision and the application stay separable and the caller decides
    # how hard to pull. Only draws randomness for the small positional jitter
    # that real bodies have inside a fixed slot.
    @classmethod
    def _corner_box_occupancy(
        cls,
        *,
        attacks_right: bool,
        corner_y: float,
        zone: Optional[str],
        marking,
        receiver,
        att_players,
        def_players,
    ) -> Dict[str, Tuple[float, float]]:
        """Fill the box for a corner. Depth is measured FROM the goal being
        attacked, so one slot table serves a team attacking either way."""
        own_goal_x = 105.0 if attacks_right else 0.0

        def at(depth: float, y: float) -> Tuple[float, float]:
            return ((own_goal_x - depth, y) if attacks_right
                    else (own_goal_x + depth, y))

        near = (38.0, 43.0) if corner_y < 34 else (25.0, 30.0)
        far = (25.0, 30.0) if corner_y < 34 else (38.0, 43.0)
        # (depth from the goal line, y band, slot label)
        slots: List[Tuple[float, Tuple[float, float], str]] = [
            (13.5, (31.5, 36.5), "six"),
            (10.5, near, "near"),
            (10.5, far, "far"),
            (15.0, (32.0, 36.0), "penalty"),
            (17.5, (30.0, 38.0), "second"),
            (25.0, (29.0, 39.0), "edge"),
        ]

        def aerial(p) -> float:
            return (0.6 * getattr(p.dna.physical, "jumping", 60.0)
                    + 0.4 * getattr(p.dna.technical, "heading", 60.0))

        by_name = {p.name: p for p in att_players}
        out: Dict[str, Tuple[float, float]] = {}

        # 1. the receiver takes the slot the routine asked for
        receiver_slot = next((s for s in slots if s[2] == zone), None) \
            or slots[0]
        if receiver is not None:
            d, band, _ = receiver_slot
            out[receiver.name] = at(d + random.uniform(-1.2, 1.2),
                                    random.uniform(*band))

        # 2. the best remaining aerial threats take the remaining slots
        used = {receiver.name} if receiver is not None else set()
        pool = sorted((p for p in att_players
                       if getattr(p, "position", "") != "GK"
                       and getattr(p, "name", "") not in used),
                      key=lambda p: -aerial(p))
        for p, (d, band, _lab) in zip(pool, [s for s in slots
                                             if s is not receiver_slot]):
            out[p.name] = at(d + random.uniform(-1.2, 1.2),
                             random.uniform(*band))

        # 3. every attacker in the box gets a defender GOAL-SIDE of him. This
        #    is the part that actually suppresses corner goals: the header is
        #    contested because somebody stands between the man and the goal,
        #    not because a probability said so.
        marked: set = set()

        def mark(dname: str, aname: str) -> bool:
            if dname in marked or dname in out or aname not in out:
                return False
            ax, ay = out[aname]
            depth = abs(own_goal_x - ax)
            out[dname] = at(max(1.5, depth - 1.4),
                            ay + random.uniform(-1.2, 1.2))
            marked.add(dname)
            return True

        # 3a. the marking grid's own assignments take precedence
        for dname, aname in (getattr(marking, "assignments", None) or {}).items():
            if aname in out:
                mark(dname, aname)
            else:                      # a zonal slot rather than a man
                lab = str(aname).replace("_zone", "")
                s = next((x for x in slots if x[2] == lab), None)
                if s and dname not in out:
                    out[dname] = at(s[0], random.uniform(*s[1]))
                    marked.add(dname)

        # 3b. then pair whatever is left, best aerial defenders first
        def_pool = sorted((p for p in def_players
                           if getattr(p, "position", "") != "GK"
                           and getattr(p, "name", "") not in marked),
                          key=lambda p: -aerial(p))
        loose = [n for n in out
                 if n not in marked
                 and getattr(by_name.get(n), "position", "") != "GK"]
        for d in def_pool:
            if not loose:
                break
            mark(d.name, loose.pop(0))

        return out

    @classmethod
    def _corner_chain(cls, minute, att_team, def_team,
                       att_players, def_players, state,
                       attacks_right=True, position_engine=None,
                       routine: Optional[SetPieceRoutine] = None,
                       delivery_origin: Optional[Tuple[float, float]] = None,
                       is_corner: bool = True,
                       ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        episode = None

        def _moving_player_local(name):
            return cls._moving_player(
                next((p for p in att_players + def_players if getattr(p, "name", "") == name), None),
                position_engine,
            )

        corner_x = 105.0 if attacks_right else 0.0
        corner_y  = random.choice([1.0, 67.0])
        # Law 11 judgement origin: WHERE the ball is struck from. Defaults to
        # the corner arc; a crossed free kick passes its own spot instead.
        delivery_x, delivery_y = (
            delivery_origin if delivery_origin is not None else (corner_x, corner_y)
        )
        # Which side the ball is delivered FROM follows the delivery spot, not
        # the corner arc. It picks the taker (`_pick_sp_taker`), the corner-
        # marking grid, and the in/out-swing sign below. Reading it off
        # `corner_y` meant a crossed free kick on the left flank was struck
        # from a RIGHT corner arc — a random one — while being offside-judged
        # from the foul spot. Two origins for one delivery. For a real corner
        # the two are the same value, so corners are unaffected.
        # NOTE: the `random.choice` above is still drawn unconditionally so the
        # football RNG stream is byte-identical to before this fix.
        corner_side = "right" if delivery_y > 34 else "left"
        # Snapshot every player's position at the instant the ball is played
        # (the Law 11 judgement moment) — captured before any chain movement
        # or record_touch overwrites a position.
        delivery_pos: Dict[str, Tuple[float, float]] = {}
        if position_engine is not None:
            delivery_pos = {
                p.name: position_engine.get_position(p.name)
                for p in (att_players + def_players)
                if getattr(p, "name", None)
            }
        taker    = cls._pick_sp_taker(att_players, situation="corner", corner_side=corner_side)
        # Feature #3 — committed attacking routine: the scheme decides WHO
        # attacks the delivery and WHERE it is aimed. Baseline (routine=None)
        # keeps the classic pure aerial-threat pick.
        params = corner_delivery(routine)
        zone = params.get("target_zone")
        receiver = cls._pick_set_piece_target(att_players, zone, exclude=taker.name)
        gk       = cls._pick_gk_player(def_players)

        # ── CHECKPOINT P1: SET-PIECE DEFENSIVE MARKING GRID ────────────────
        # Replace the single "pick one jumping CB" aerial duel with a real
        # corner-defence assignment: the best-suited aerial CB carries the
        # opponent's biggest aerial threat, a first man guards the near-post
        # line, and the GK starts on the defended side of the away swing.
        own_goal_x = 105.0 if attacks_right else 0.0
        sp_defenders = [
            (p.name, p.position,
             getattr(p.dna.physical, "jumping", 60.0),
             getattr(p.dna.technical, "heading", 60.0))
            for p in def_players
        ]
        sp_attackers = [
            (p.name, p.position,
             getattr(p.dna.physical, "jumping", 60.0),
             getattr(p.dna.technical, "heading", 60.0))
            for p in att_players if p.position != "GK"
        ]
        sp = SetPieceMarkingEngine.build(
            sp_defenders, sp_attackers, own_goal_x, corner_side=corner_side)
        aerial_defender_name = sp.aerial_defender
        defender = next((p for p in def_players
                         if p.name == aerial_defender_name), None)
        if defender is None:
            defender = cls._pick_aerial_defender(def_players)

        # Corner taken event — outcome is determined by the aerial physics
        # below, not a pre-roll. Start as True; if no one wins it cleanly
        # we will re-evaluate after resolve_aerial.
        # It carries NO end point. The delivery target is not known until the
        # flight is built below, and the contact point is not known until the
        # aerial resolves, so both are stamped afterwards onto this same event
        # (`_stamp_delivery_endpoint`). Emitting it with an invented end here
        # is what made every corner a zero-length "key pass".
        corner_event = cls.make_event(
            minute, EventType.CORNER_TAKEN, att_team, taker.name,
            phase, gs,
            location_x=delivery_x, location_y=delivery_y,
            outcome=True,
            situation=SituationType.CORNER,
        )
        result.add(corner_event)

        # ── PLAYER POSITIONING ─────────────────────────────────────
        # Use position_engine when available so players are near their
        # realistic goalmouth positions. Fall back to fixed zones.
        if position_engine is not None:
            # Ensure the position engine has been updated for this minute.
            position_engine.update_pitch_control(att_team, def_team, minute=minute)

            def _pos(name):
                return position_engine.get_position(name)

            # Attacking target: near the penalty spot / front post
            if receiver:
                rx, ry = _pos(receiver.name)

                def _zone_band(_zone, _corner_y):
                    # near/far swap with the corner side; six/penalty/edge
                    # stay central in the box.
                    if _zone == "near":
                        return (24.0, 30.0) if _corner_y < 34 else (38.0, 44.0)
                    if _zone == "far":
                        return (38.0, 44.0) if _corner_y < 34 else (24.0, 30.0)
                    if _zone == "six":
                        return (31.0, 37.0)
                    if _zone == "penalty":
                        return (32.0, 36.0)
                    return (26.0, 36.0)  # edge (short-corner pull-back)

                def _zone_x(_zone, _att_right):
                    if _zone == "six":
                        return (93.0 + random.uniform(0.0, 2.0)) if _att_right \
                            else (12.0 - random.uniform(0.0, 2.0))
                    if _zone == "edge":
                        return (84.0 + random.uniform(0.0, 3.0)) if _att_right \
                            else (21.0 - random.uniform(0.0, 3.0))
                    return (88.0 + random.uniform(0.0, 6.0)) if _att_right \
                        else (17.0 - random.uniform(0.0, 6.0))

                if zone is not None:
                    # Feature #3: the routine plants the target at its own
                    # box zone, not whichever way the receiver drifted.
                    low, high = _zone_band(zone, corner_y)
                    rx = _zone_x(zone, attacks_right)
                    ry = random.uniform(low, high)
                else:
                    # Keep them in the box — nudge if they drifted too far.
                    if attacks_right and rx < 85:
                        rx = 88.0 + random.uniform(0.0, 6.0)
                    elif not attacks_right and rx > 20:
                        rx = 17.0 - random.uniform(0.0, 6.0)
                    ry = max(22.0, min(46.0, ry))
                if position_engine is not None:
                    position_engine.record_touch(receiver.name, rx, ry, minute)
            else:
                rx = 92.0 if attacks_right else 13.0
                ry = random.uniform(28.0, 40.0)

            # Defender marker: near the attacker but slightly offset
            if defender:
                dx, dy = _pos(defender.name)
                dy = max(24.0, min(44.0, dy + random.uniform(-2.0, 2.0)))
                if position_engine is not None:
                    position_engine.record_touch(defender.name, dx, dy, minute)
            else:
                dx = rx + random.uniform(-1.5, 1.5)
                dy = ry + random.uniform(-2.0, 2.0)

            # GK: on the goal line, central
            if gk:
                gkx = 104.3 if attacks_right else 0.7
                gky = 34.0 + random.uniform(-3.0, 3.0)
                if position_engine is not None:
                    position_engine.record_touch(gk.name, gkx, gky, minute)
            else:
                gkx = 104.3 if attacks_right else 0.7
                gky = 34.0
        else:
            # Fallback positions when position_engine is unavailable
            rx = 92.0 if attacks_right else 13.0
            ry = random.uniform(28.0, 40.0)
            dx = rx + random.uniform(-1.5, 1.5)
            dy = ry + random.uniform(-2.0, 2.0)
            gkx = 104.3 if attacks_right else 0.7
            gky = 34.0

        # ── BOX OCCUPANCY: place the WHOLE unit, not three players ────────
        # Applied AFTER the receiver/marker/GK writes above, so the pack wins
        # any conflict, and BEFORE the 3D flight, so the aerial duel happens in
        # the positions the marking grid asked for. Gated on `is_corner`: a
        # crossed free kick reaches this function too (23 of the 28 calls in a
        # typical match) and must NOT get a corner's box.
        if is_corner:
            try:
                box = cls._corner_box_occupancy(
                    attacks_right=attacks_right,
                    corner_y=corner_y,
                    zone=zone,
                    marking=sp,
                    receiver=receiver,
                    att_players=att_players,
                    def_players=def_players,
                )
            except Exception:
                # A set piece must never take a match down with it. Falling
                # back to the old three-player set-up is wrong but playable;
                # crashing is neither.
                box = {}
            if box and position_engine is not None:
                # NO PLACEMENT. The box is opened as a PENDING WINDOW and the
                # players walk themselves into it over `SET_PIECE_JOSTLE_S`.
                #
                # Placing them here (the previous behaviour, via
                # `set_piece_place`) meant they were ALREADY on their slots
                # before the integrator's first tick, so the window was holding
                # a position nobody had to move to: measured median distance
                # from slot to player was 0.0 m across 156 slot-assignments in
                # 12 corners, and 0.03 of them closed any ground at all. The
                # window, the duration and the dead-ball run suppression were
                # all live and all inert.
                #
                # `set_piece_place` is still the correct call for the free-kick
                # WALL, where the men genuinely are standing still and the
                # arrangement is the point. A corner box is a race, not a pose.
                #
                # `set_piece_place`'s own docstring is right that hiding the
                # displacement would "move the lie to a different column" --
                # it is not hidden, it is counted ONCE by
                # `record_physics_distance` in `_offball_move_player`, now
                # with real elapsed time and a real speed instead of a 30 m
                # teleport in 0.0 s.
                # Run them in FIRST, inside the chain, because the
                # delivery below is resolved in this same call and a
                # window opened for the post-chain integrator arrives
                # too late to affect it -- the header would be contested
                # by an empty box. `advance_to_slots` moves them at a
                # sprint with real elapsed time and books the distance
                # once, honestly.
                position_engine.advance_to_slots(
                    box, SET_PIECE_JOSTLE_S,
                    exclude={getattr(taker, "name", None)})

                # The window then HOLDS the box through the post-corner
                # integration, so they are not immediately dragged back
                # out to their shape anchors, and run sampling is
                # suppressed while it is open.
                position_engine.open_setpiece_window(
                    box, SET_PIECE_JOSTLE_S)
                if receiver is not None and receiver.name in box:
                    rx, ry = box[receiver.name]

        # ── 3D FLIGHT ──────────────────────────────────────────────
        # The ball always leaves the taker's boot. Delivery quality is encoded
        # in the flight parameters, not a separate success roll.
        #
        # KEY FIX: the target height must be WITHIN players' vertical reach
        # (~2.1–2.4 m for most outfielders). The old code set corner_height
        # as high as 3.9 m, which made every corner a "drops" because no one
        # could physically reach the delivery point.
        cross_quality = taker.dna.technical.crossing / 100.0
        # Good crossers (80+) put it in the 2.0–2.4 m sweet spot.
        # Poor crossers (0) float it 2.6–3.0 m (too high) or drive it 1.6–1.9 m (too low).
        corner_height = 2.0 + (1.0 - cross_quality) * random.uniform(0.3, 0.9)
        # Feature #3: routines only bend the flight of a crosser who CAN
        # bend it — a poor crosser's ball stays a poor crosser's ball.
        corner_height = max(2.0, min(3.0, corner_height + params.get("height_bias", 0.0) * cross_quality))
        corner_speed = aerial_delivery_speed(
            float(getattr(taker.dna.passing, "long_passing", 55.0))
        )
        # Target is slightly ahead of the attacker so they run onto it
        target_x = rx + random.uniform(-1.0, 1.0)
        target_y = ry + random.uniform(-1.0, 1.0)
        # The flight arc is now ballistic: launch angle and apex fall out of
        # the taker's crossing speed, not a tuned apex.
        # An in-swinger bends toward the goalmouth (side spin from the taker's
        # crossing quality) — the classic corner that sucks the keeper toward
        # the six-yard box even before the aerial duel is contested.
        corner_spin = delivery_spin(
            float(getattr(taker.dna.passing, "crossing", 55.0)),
            kind="side",
        )
        # Feature #3: an out-swinging corner curls AWAY from the goalmouth
        # (the far-post hunted delivery), the in-swinger toward it. The sign
        # is made deterministic per corner side; the RNG draw above is kept
        # for stream parity.
        if params.get("swing") == "out":
            out_sign = -1.0 if delivery_y < 34 else 1.0
            corner_spin = BallSpin(corner_spin.rate, corner_spin.kind, out_sign)
        flight = make_ballistic_flight(
            Vec3(delivery_x, delivery_y, 0.05),
            Vec3(target_x, target_y, corner_height),
            corner_speed,
            loft=0.0,
            spin=corner_spin,
        )

        # ── AERIAL DUEL — outcome authority ────────────────────────
        episode = PossessionEpisode()
        attacker_mp = _moving_player_local(receiver.name) if receiver else None
        defender_mp = _moving_player_local(defender.name) if defender else None
        gk_mp = _moving_player_local(gk.name) if gk else None

        # EVERY player who reached the box, not just the receiver.
        # This is step 3 and it is the step that makes steps 1 and 2 mean
        # anything: until now the aerial duel was offered THREE players
        # (receiver + one defender + keeper) while the box held thirteen,
        # so the crowding and the running were cosmetic.
        _arrived = (position_engine.arrived_setpiece_players(
            SET_PIECE_ARRIVED_M) if position_engine is not None else set())

        attackers = []
        for _p in att_players:
            if _p.name not in _arrived:
                continue
            _mp = _moving_player_local(_p.name)
            if _mp is not None:
                attackers.append(_mp)
        if not attackers and attacker_mp is not None:
            # Nobody made the box. Still resolve against the receiver rather
            # than declaring the corner uncontested -- the ball was aimed at
            # him and he still has to be beaten to win it.
            attackers = [attacker_mp]

        defenders = []
        for _p in def_players:
            if _p.name in _arrived:
                _mp = _moving_player_local(_p.name)
                if _mp is not None:
                    defenders.append(_mp)
        if defender_mp is not None and defender_mp not in defenders:
            defenders.append(defender_mp)
        if gk_mp:
            defenders.append(gk_mp)
        episode.register(attackers + defenders)
        aerial = episode.resolve_aerial(flight, attackers, defenders)

        # ── STAMP THE DELIVERY'S REAL ENDPOINT ─────────────────────
        # The corner event was emitted above without one. Now that the flight
        # exists, the delivery endpoint is the point the ball actually reached:
        # the contact point if anyone got to it, otherwise the aimed target.
        # Both are tracked physics, and `secondary_player` names the man the
        # ball was aimed at — which is what lets the chance-creation ledger
        # link a corner to a shot by that player instead of guessing.
        _contact = getattr(aerial, "contact_point", None) if aerial else None
        if _contact is not None:
            _end_x, _end_y = _contact.x, _contact.y
            _end_src = "contact"
        else:
            _end_x, _end_y = target_x, target_y
            _end_src = "aimed_target"
        corner_event.end_x = _end_x
        corner_event.end_y = _end_y
        if receiver is not None:
            corner_event.secondary_player = receiver.name
        corner_event.metadata = {
            **(corner_event.metadata or {}),
            "delivery_end_x": round(_end_x, 2),
            "delivery_end_y": round(_end_y, 2),
            "delivery_end_source": _end_src,
            "delivery_target": receiver.name if receiver is not None else None,
        }

        # Determine who won, if anyone.
        winner_name = ""
        winner_team = ""
        if aerial.winner is not None:
            winner_name = getattr(aerial.winner.player, "name", "")
            if winner_name in [p.name for p in att_players]:
                winner_team = att_team
            elif winner_name in [p.name for p in def_players]:
                winner_team = def_team

        # outcome authority branches
        # Who actually heads it. With a real box contested, that is
        # whoever won the aerial -- not necessarily the player the
        # routine aimed at. Requiring the winner to BE the receiver (the
        # previous test) would have filed every other attacker header as a
        # loose ball the moment step 3 let more than one attacker compete.
        headerer = None
        if winner_name and winner_team == att_team:
            headerer = next((p for p in att_players
                             if p.name == winner_name), None)
        if headerer is None and attacker_mp is not None:
            headerer = receiver
        att_wins = (
            aerial.outcome in ("controlled", "contested")
            and headerer is not None
        )
        def_wins = (
            aerial.outcome in ("controlled", "contested")
            and not att_wins
            and winner_team == def_team
        )
        gk_wins  = gk is not None and winner_name == gk.name

        # Corner delivery quality: only an attacker winning the aerial counts
        # as a "completed" corner. Defender/GK wins or a loose ball = failure.
        corner_delivery_success = att_wins

        # Re-emit CORNER_TAKEN with the correct outcome now that physics
        # has decided it. The exporter uses this for completion stats.
        for e in result.events:
            if e.event_type == EventType.CORNER_TAKEN:
                e.outcome = corner_delivery_success
                _md = dict(e.metadata or {})
                if routine is not None:
                    _md["routine"] = routine.value
                    _md["crowd"] = round(params.get("crowd", 0.0), 2)
                e.metadata = _md
                break

        # AERIAL_DUEL event
        result.add(cls.make_event(
            minute, EventType.AERIAL_DUEL, att_team,
            receiver.name if receiver else taker.name,
            phase, gs,
            secondary_player=defender.name if defender else (gk.name if gk else None),
            location_x=aerial.contact_point.x if aerial else target_x,
            location_y=aerial.contact_point.y if aerial else target_y,
            outcome=att_wins,
            metadata={
                "physics": episode.physics_meta("corner_aerial") if episode else None,
                "resolution": "aerial_trajectory" if position_engine is not None else "legacy_roll",
                "winner": winner_name,
                "winner_team": winner_team,
                "set_piece_marking": sp.as_dict() if sp else None,
                **({"routine": routine.value, "target_zone": zone}
                   if routine is not None else {}),
            }
        ))

        # Update position of whoever touched it
        if winner_name and position_engine is not None:
            cx = aerial.contact_point.x if aerial else target_x
            cy = aerial.contact_point.y if aerial else target_y
            position_engine.record_touch(winner_name, cx, cy, minute)

        # ── ATTACKER WINS → HEADER SHOT ───────────────────────────
        if att_wins and headerer:
            # Feature #3: a committed pile (six-yard / spotted crowd) raises
            # the odds of a BIG_CHANCE-grade header the same way numbers in
            # the box do in real football.
            big_odds = 0.35 + params.get("crowd", 0.0) * 0.15
            xg = XGEngine.calculate(
                zone="inside_box", body_part="head",
                situation=SituationType.CORNER,
                is_big_chance=random.random() < big_odds
            )
            result.xg_generated = xg
            result.xa_generated = xg

            shot_x = aerial.contact_point.x if aerial else target_x
            shot_y = aerial.contact_point.y if aerial else target_y
            shot_x = cls.clamp_attack_x(
                shot_x + random.uniform(-2.0, 2.0), 85.0, 102.0, attacks_right)
            shot_y = max(26.0, min(42.0, shot_y + random.uniform(-2.0, 2.0)))

            sot_prob = 0.30 + xg * 0.4
            if random.random() < sot_prob:
                result.shot_on_target = True
                result.add(cls.make_event(
                    minute, EventType.SHOT_ON_TARGET, att_team, headerer.name,
                    phase, gs, xg=xg, body_part="head",
                    situation=SituationType.CORNER, outcome=True,
                    secondary_player=gk.name if gk else None,
                    location_x=shot_x,
                    location_y=shot_y,
                ))

                is_goal, positioning = GoalkeeperEngine.evaluate_save(
                    xg, DNAFactory.get_shooter_quality(headerer.dna),
                    shot_x, shot_y, gk,
                    state.last_ball_x, state.last_ball_y
                )
                if is_goal:
                    result.goal_scored    = True
                    result.goal_team      = att_team
                    result.goal_scorer    = headerer.name
                    result.goal_assistant = taker.name
                    result.add(cls.make_event(
                        minute, EventType.GOAL, att_team, headerer.name,
                        phase, gs, xg=xg, body_part="head",
                        situation=SituationType.CORNER, outcome=True,
                        secondary_player=taker.name,
                        location_x=shot_x,
                        location_y=shot_y,
                    ))
                else:
                    is_goalline_save = (
                        positioning.get("start_x") is not None
                        and positioning["start_x"] <= 2.0
                    )
                    if gk:
                        result.add(cls.make_event(
                            minute, EventType.SAVE, def_team, gk.name,
                            phase, gs, xg=xg, outcome=True,
                            location_x=shot_x,
                            location_y=shot_y,
                            metadata={"goalline_save": is_goalline_save}
                        ))
                    # ── REBOUND / SECOND BALL AFTER SAVE ──────────
                    if random.random() < 0.12:
                        rebound_player = cls._pick_aerial_threat(att_players, exclude=headerer.name)
                        if rebound_player:
                            rebound_x = cls.clamp_attack_x(
                                shot_x + random.uniform(-3.0, 3.0), 85.0, 100.0,
                                attacks_right)
                            rebound_y = max(24.0, min(44.0, shot_y + random.uniform(-3.0, 3.0)))
                            result.add(cls.make_event(
                                minute, EventType.BALL_RECOVERY, att_team, rebound_player.name,
                                phase, gs,
                                location_x=rebound_x, location_y=rebound_y,
                                outcome=True,
                                metadata={"loose_ball": True, "second_phase": True}
                            ))
                            rebound_xg = XGEngine.calculate(
                                zone="inside_box", body_part="right_foot",
                                situation=SituationType.CORNER,
                            ) * 0.60
                            rebound_sot = random.random() < (0.25 + rebound_xg * 0.3)
                            if rebound_sot:
                                is_goal2, _ = GoalkeeperEngine.evaluate_save(
                                    rebound_xg, DNAFactory.get_shooter_quality(rebound_player.dna),
                                    rebound_x, rebound_y, gk,
                                    state.last_ball_x, state.last_ball_y,
                                )
                                result.add(cls.make_event(
                                    minute, EventType.SHOT_ON_TARGET if is_goal2 or random.random() < 0.5 else EventType.SHOT_OFF_TARGET,
                                    att_team, rebound_player.name,
                                    phase, gs, xg=rebound_xg, body_part="right_foot",
                                    situation=SituationType.CORNER, outcome=is_goal2,
                                    location_x=rebound_x, location_y=rebound_y,
                                ))
                                if is_goal2:
                                    result.goal_scored    = True
                                    result.goal_team      = att_team
                                    result.goal_scorer    = rebound_player.name
                                    result.goal_assistant = taker.name
                                    result.add(cls.make_event(
                                        minute, EventType.GOAL, att_team, rebound_player.name,
                                        phase, gs, xg=rebound_xg, body_part="right_foot",
                                        situation=SituationType.CORNER, outcome=True,
                                        secondary_player=taker.name,
                                        location_x=rebound_x, location_y=rebound_y,
                                        metadata={"second_phase": True}
                                    ))
            else:
                result.add(cls.make_event(
                    minute, EventType.SHOT_OFF_TARGET, att_team, headerer.name,
                    phase, gs, xg=xg, outcome=False,
                    location_x=shot_x,
                    location_y=shot_y,
                ))
                if random.random() < 0.10:
                    second_attacker = cls._pick_aerial_threat(att_players, exclude=taker.name)
                    if second_attacker and second_attacker != headerer:
                        scramble_x = cls.clamp_attack_x(
                                shot_x + random.uniform(-2.0, 2.0), 85.0, 100.0,
                                attacks_right)
                        scramble_y = max(24.0, min(44.0, shot_y + random.uniform(-2.0, 2.0)))
                        result.add(cls.make_event(
                            minute, EventType.BALL_RECOVERY, att_team, second_attacker.name,
                            phase, gs,
                            location_x=scramble_x, location_y=scramble_y,
                            outcome=True,
                            metadata={"loose_ball": True}
                        ))
        # ── DEFENDER WINS → CLEARANCE ─────────────────────────────
        elif def_wins and not gk_wins:
            # Route through DefensiveChain so the clearance gets the
            # full biomechanical treatment: headed/foot split, success
            # rate, danger relief, and all metadata the exporter/threat
            # engine expect.
            clearance_result = DefensiveChain.generate(
                minute, def_team, att_team,
                def_players, att_players,
                state, action_type="clearance",
                context_x=dx, context_y=dy,
                attacks_right=attacks_right,
                danger_level=state.threat.danger_at(def_team) if hasattr(state, 'threat') else 50.0,
                ball_aerial=True,
                own_goal_x=105.0 if attacks_right else 0.0,
                position_engine=position_engine,
                ball_z=aerial.contact_point.z if aerial else 2.0,
                defender_facing_x=dx, defender_facing_y=dy,
                opponent_distance=math.hypot(
                    dx - aerial.contact_point.x,
                    dy - aerial.contact_point.y,
                ) if aerial else 2.0,
            )
            for e in clearance_result.events:
                result.add(e)
            cls._inherit_restart(clearance_result, result)

        # ── GK WINS → CLAIM / PUNCH / RUNS OUT ───────────────────────
        elif gk_wins and gk:
            # The keeper's win is a physical high-ball take: he gets there at
            # the actual aerial contact point (wherever the delivery carried),
            # not some placeholder beside the goal line.
            cpx = aerial.contact_point.x if aerial is not None else target_x
            cpy = aerial.contact_point.y if aerial is not None else target_y
            cpx = max(83.0, min(104.5, cpx)) if attacks_right else max(0.5, min(22.0, cpx))
            cpy = max(24.0, min(44.0, cpy))
            contact_z = aerial.contact_point.z if aerial is not None else corner_height
            contested = aerial is not None and aerial.outcome == "contested"
            challenger_present = aerial is not None and aerial.challenger is not None
            action, claim_height = GoalkeeperEngine.decide_high_ball(
                gk, contact_z, contested, challenger_present,
            )

            event_type_md = "gk_punch" if action == "punch" else "gk_claim"
            result.add(cls.make_event(
                minute, EventType.SAVE, def_team, gk.name,
                phase, gs, outcome=True,
                location_x=cpx,
                location_y=cpy,
                metadata={
                    "type": event_type_md,
                    "corner_followup": "gk_punch" if action == "punch" else "gk_claim",
                    "corner_side": corner_side,
                    "claim_height": claim_height,
                    "contested": contested,
                    "contact_z": round((aerial.contact_point.z if aerial is not None else corner_height), 2),
                },
            ))

        # ── NO CLEAR WINNER → BALL FALLS LOOSE ────────────────────
        else:
            loose_x = cls.clamp_attack_x(
                target_x + random.uniform(-2.0, 2.0), 85.0, 100.0, attacks_right)
            loose_y = max(24.0, min(44.0, target_y + random.uniform(-2.0, 2.0)))
            result.add(cls.make_event(
                    minute, EventType.BALL_RECOVERY, att_team,
                    receiver.name if receiver else taker.name,
                    phase, gs,
                    location_x=loose_x, location_y=loose_y,
                    outcome=True,
                    metadata={"loose_ball": True, "corner_followup": "loose"},
                ))

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)

        # ── SHOT-SPEED STAMP (Opta-style) ─────────────────────────
        # Corner headers/rebound shots carry the SHOT event's body_part, so the
        # stamp loop derives each effort's launch velocity (head ≈ far slower
        # than a struck ball) and its ~flight time to the goal line.
        _goal_line_x = 105.0 if attacks_right else 0.0
        for _ev in result.events:
            if _ev.event_type in SHOT_SPEED_EVENT_TYPES:
                _bp = getattr(_ev, "body_part", "head")
                _shooter = next(
                    (p for p in att_players if p.name == _ev.player), receiver
                )
                _spd = _shot_speed_for_player(_shooter, "head" if _bp == "head" else "foot")
                _stamp_shot_speed(
                    _ev, _spd,
                    max(0.1, abs(_goal_line_x - _ev.location_x) / _spd),
                )

        # ── VAR / LAW 11 — offside at the moment the ball was played ──
        # A goal by a player who stood in an offside position when the
        # delivery was struck is ruled out (positions come from the entry
        # snapshot, not live — judgement happens at the kick, not the header).
        # For a CORNER the ball sits ON the goal line so nobody can be ahead
        # of it — the geometry itself encodes Law 11's direct-corner
        # exemption. Crossed free kicks (delivery_origin = the foul spot) are
        # the real case this catches: a marker caught creeping beyond the
        # line when the cross was whipped in.
        if result.goal_scored and delivery_pos and result.goal_scorer in delivery_pos:
            _scorer_del_x, _scorer_del_y = delivery_pos[result.goal_scorer]
            _def_xs = [
                delivery_pos[d.name][0] for d in def_players
                if getattr(d, "position", None) != "GK"
                and getattr(d, "name", None) in delivery_pos
            ]
            if len(_def_xs) >= 2:
                _def_xs.sort() if attacks_right else _def_xs.sort(reverse=True)
                _second_last = _def_xs[1]
                if PossessionChain._in_offside_position(
                        _scorer_del_x, delivery_x, _second_last, attacks_right):
                    _margin = ((_scorer_del_x - _second_last) if attacks_right
                               else (_second_last - _scorer_del_x))
                    if _margin > OFFSIDE_LEVEL_TOL_M:
                        mark_var_disallowed(
                            result, minute, att_team, phase, gs,
                            result.goal_scorer, _scorer_del_x, _scorer_del_y,
                        )

        # The jostling window the players were walked in over. Reported
        # so `_absorb_motion` integrates the box for that long instead of
        # the synthesized ball-travel time; the measured dead time was
        # 0.0 s, which is why they were being teleported into position.
        result.sequence_duration_s = SET_PIECE_JOSTLE_S
        return result

    # ── DIRECT FREE KICK WALL ─────────────────────────────────────────
    # Law 12: the wall stands 9.15 m from the ball, between the ball and the
    # goal, facing the kicker. Before this, `_freekick_chain`'s direct branch
    # shot at an undefended goal — the wall was never built and the direct
    # branch never ran at all (see the foul-awarded free kick award in
    # match_engine.py; before it, 100% of free kicks came from the offside
    # queue and every one became a cross).
    #
    # The men are the DEFENDING outfielders, chosen as the ones already
    # nearest the ball, and they are placed by a BOUNDED approach: nobody
    # covers more than `speed * APPROACH_S` from where he stands. A defender
    # 40 m away genuinely cannot get into the wall, and the helper returns
    # where each man actually ended up, so the blocker set handed to the
    # geometry engine is the wall that exists, not the wall that was ordered.
    #
    # APPROACH_S is the dead-ball window, and 2.5 s was simply the wrong
    # number: measured over 2 real matches it produced walls whose men sat a
    # mean 17.2 m from the ball (median 16.7, only 25% inside 10.5 m) and
    # spread 30.4 m laterally. A 30 m-wide "wall" is not a wall, and the
    # 30 m came from the budget itself — a man who runs out of travel stops
    # partway along the line from where he stood, and those partway points
    # are scattered all over the pitch. So the defect was the budget, not the
    # placement maths.
    #
    # Real direct free kicks get roughly 8-12 s of organisation: the referee
    # signals, the taker steps back, defenders jog across and the wall forms.
    # 9.0 s at the engine's own top speeds (5.0 + pace*0.042 m/s, 7.3-8.7 in
    # practice) is a 65-78 m budget, which every outfielder comfortably
    # covers from anywhere on the pitch — and that is the point. The bound is
    # retained deliberately rather than removed: a keeper 60 m upfield
    # genuinely cannot be in the wall, and the helper must keep reporting the
    # wall that exists rather than the wall that was ordered. What changes is
    # that for an ordinary dead ball, the honest answer is the full wall.
    WALL_DISTANCE_M = 9.15
    WALL_SETUP_S = 9.0
    WALL_MIN_MEN = 4
    # Shoulder-to-shoulder spacing. 0.55 m per man gives a 4-man wall about
    # 1.65 m of face, which is what a real wall presents to a striker.
    WALL_SPACING_M = 0.55

    @classmethod
    def _build_freekick_wall(cls, fk_x, fk_y, def_players, attacks_right,
                             position_engine, minute, gk):
        """Place the wall. Returns (wall_mps, gk_mp, reached) where `wall_mps`
        are MovingPlayers at the positions they could actually reach."""
        import math as _m

        goal_x = 105.0 if attacks_right else 0.0
        near_y = 30.34 if fk_y < 34 else 37.66
        dx, dy = goal_x - fk_x, near_y - fk_y
        length = _m.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        # unit vector ALONG the wall face
        px, py = -uy, ux

        # Nearest outfielders make the wall, keeper excluded.
        pool = [p for p in def_players
                if getattr(p, "position", "") != "GK" and p is not gk]
        def _dist(p):
            if position_engine is None:
                return 0.0
            try:
                ex, ey = position_engine.get_position(p.name)
            except Exception:
                return 1e9
            return _m.hypot(ex - fk_x, ey - fk_y)
        pool.sort(key=_dist)
        men = pool[:max(cls.WALL_MIN_MEN, 4)]

        wall_mps = []
        # Centre the wall on the ball-to-near-post line, then fan the men out
        # along the face at shoulder width. For 4 men this presents ~1.65 m
        # of face; the target is a flat line, and a wide spread here would
        # mean the men are not actually standing together.
        n = len(men)
        reach_log = []
        for i, p in enumerate(men):
            lateral = (i - (n - 1) / 2.0) * cls.WALL_SPACING_M
            tx = fk_x + ux * cls.WALL_DISTANCE_M + px * lateral
            ty = fk_y + uy * cls.WALL_DISTANCE_M + py * lateral
            ty = max(1.0, min(67.0, ty))
            if position_engine is not None:
                try:
                    sx, sy = position_engine.get_position(p.name)
                except Exception:
                    sx, sy = fk_x, fk_y
                pace = float(getattr(getattr(p.dna, "physical", None),
                                     "pace", 60.0))
                top_speed = 5.0 + max(0.0, min(100.0, pace)) * 0.042
                budget = top_speed * cls.WALL_SETUP_S
                d = _m.hypot(tx - sx, ty - sy)
                if d > budget and d > 0.0:
                    # Cannot get there: stop at the edge of what he can cover.
                    scale = budget / d
                    tx, ty = sx + (tx - sx) * scale, sy + (ty - sy) * scale
                reach_log.append(_m.hypot(tx - sx, ty - sy))
                # set_piece_place, NOT record_touch: record_touch applies the
                # wide-role flank hold and the GK box anchor after writing,
                # and those corrections dismantle a wall (see
                # PositionEngine.set_piece_place for the measurement).
                position_engine.set_piece_place(p.name, tx, ty, minute)
            wall_mps.append(cls._moving_player(p, position_engine))

        # Keeper sets his line. Against a wall a keeper does NOT stand on his
        # goal line — he comes off it and stands on the bisector of the angle
        # between the two posts, which is the only spot from which he can see
        # both the near post and the far post past the wall's edge. The
        # bisector from the BALL to the goal-mouth centre is the same line by
        # construction, so the keeper stands on it, just short of his line.
        gk_mp = None
        if gk is not None:
            gk_line_x = 101.0 if attacks_right else 4.0
            # Bisector: from the ball to the centre of the goal mouth.
            bcx, bcy = fk_x, fk_y
            gcx, gcy = gk_line_x, 34.0
            bl = _m.hypot(gcx - bcx, gcy - bcy) or 1.0
            bux, buy = (gcx - bcx) / bl, (gcy - bcy) / bl
            # Stand 2.5 m off his line, toward the ball, i.e. forward of the
            # posts but behind the wall's back (the wall is 9.15 m out).
            gk_tx = gcx - bux * 2.5
            gk_ty = max(6.0, min(62.0, gcy - buy * 2.5))
            if position_engine is not None:
                try:
                    position_engine.set_piece_place(
                        gk.name, gk_tx, gk_ty, minute)
                except Exception:
                    pass
            gk_mp = cls._moving_player(gk, position_engine)
        return wall_mps, gk_mp, reach_log

    @classmethod
    def _freekick_chain(cls, minute, att_team, def_team,
                         att_players, def_players, state, situation,
                         attacks_right=True,
                         context_x: Optional[float] = None,
                         context_y: Optional[float] = None,
                         position_engine=None,
                         routine: Optional[SetPieceRoutine] = None) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        episode = None

        # Checkpoint 19 — offside free kicks: use the actual offside location
        # instead of a random zone. This ensures the free kick is placed
        # exactly where the offside occurred, matching real football laws.
        if context_x is not None and context_y is not None:
            fk_x = max(2.0, min(103.0, context_x))
            fk_y = max(2.0, min(66.0, context_y))
        else:
            fk_x = cls.mirror_x(random.uniform(72, 90), attacks_right)
            fk_y = random.uniform(20, 48)
        direct_range = fk_x > 78 if attacks_right else fk_x < 27
        fk_type = "direct" if situation == SituationType.DIRECT_FREEKICK and direct_range else "crossed"
        # Feature #3: a committed TRAINED_CROSS hands the ball to the box even
        # in the direct half-circle (very common in real football);
        # DIRECT_ATTEMPT strikes the goal-mouth.
        if (routine == SetPieceRoutine.TRAINED_CROSS
                and situation == SituationType.DIRECT_FREEKICK
                and direct_range):
            fk_type = "crossed"
        taker = cls._pick_sp_taker(att_players, situation="freekick", freekick_type=fk_type)
        gk    = cls._pick_gk_player(def_players)

        result.add(cls.make_event(
            minute, EventType.FREEKICK_WON, att_team, taker.name,
            phase, gs, location_x=fk_x, location_y=fk_y
        ))

        # Direct or crossed?
        direct_range = fk_x > 78 if attacks_right else fk_x < 27
        if situation == SituationType.DIRECT_FREEKICK and direct_range and fk_type == "direct":
            # Direct shot
            result.add(cls.make_event(
                minute, EventType.FREEKICK_DIRECT, att_team, taker.name,
                phase, gs, location_x=fk_x, location_y=fk_y,
                metadata={"routine": routine.value} if routine is not None else {},
            ))

            xg = XGEngine.calculate(
                zone=PitchZone.xg_zone(fk_x, fk_y, attacks_right=attacks_right),
                body_part=random.choice(["right_foot", "left_foot"]),
                situation=SituationType.DIRECT_FREEKICK,
            )
            result.xg_generated = xg

            # ── WALL, then GEOMETRY ──────────────────────────────────
            # The wall is built and the ball is struck THROUGH it. The old
            # branch rolled `sot_prob = 0.45 + fk/100*0.35` — 45-80% on
            # target from a dead ball into an EMPTY goal — and decided the
            # outcome with `GoalkeeperEngine.evaluate_save`, which knows
            # nothing about a wall. Now the flight is aimed at the goal plane
            # by `aim_shot_flight` and resolved against the keeper's dive
            # envelope AND the wall's swept blocker envelope
            # (`geometry_engine.resolve_shot`), so a ball driven into the
            # wall is BLOCKED and one lifted over the top is not.
            wall_mps, gk_mp, _reach = cls._build_freekick_wall(
                fk_x, fk_y, def_players, attacks_right, position_engine,
                minute, gk)

            fk_episode = PossessionEpisode()
            taker_mp = cls._moving_player(taker, position_engine)
            fk_episode.register([taker_mp] + wall_mps
                                + ([gk_mp] if gk_mp else []))
            flight = aim_shot_flight(
                taker.dna, fk_x, fk_y,
                random.choice(["right_foot", "left_foot"]),
                attacks_right, under_pressure=False,
            )
            shot_res = fk_episode.resolve_shot(
                flight, gk_mp, blockers=wall_mps,
                attacks_right=attacks_right)
            outcome = shot_res.outcome
            # The terminus rides on the event as end_x/end_y, so the shot map
            # draws the real flight rather than reconstructing one.
            end_x, end_y = shot_res.goal_point.x, shot_res.goal_point.y
            _blk = getattr(shot_res, "blocker", None)
            base_meta = {
                "trajectory": "physics",
                "wall_players": [getattr(m.player, "name", "") for m in wall_mps],
                "wall_distance_m": cls.WALL_DISTANCE_M,
                "flight_time_s": round(shot_res.flight_time, 3),
                "shot_speed_mps": round(shot_res.ball_speed_at_contact, 2),
                "resolution": outcome,
            }

            if outcome in ("goal", "saved"):
                result.shot_on_target = True
                result.add(cls.make_event(
                    minute, EventType.SHOT_ON_TARGET, att_team, taker.name,
                    phase, gs, xg=xg, outcome=True,
                    secondary_player=gk.name if gk else None,
                    location_x=fk_x, location_y=fk_y,
                    end_x=end_x, end_y=end_y, metadata=base_meta,
                ))
            if outcome == "goal":
                result.goal_scored  = True
                result.goal_team    = att_team
                result.goal_scorer  = taker.name
                result.add(cls.make_event(
                    minute, EventType.GOAL, att_team, taker.name,
                    phase, gs, xg=xg, outcome=True,
                    situation=SituationType.DIRECT_FREEKICK,
                    location_x=fk_x, location_y=fk_y,
                    end_x=end_x, end_y=end_y, metadata=base_meta,
                ))
            elif outcome == "saved" and gk:
                result.add(cls.make_event(
                    minute, EventType.SAVE, def_team, gk.name,
                    phase, gs, xg=xg, outcome=True,
                    location_x=fk_x, location_y=fk_y,
                    metadata={
                        "goalline_save": False,
                        "trajectory": "physics",
                        "gk_position_at_save": (
                            [round(shot_res.gk_position_at_save.x, 2),
                             round(shot_res.gk_position_at_save.y, 2)]
                            if shot_res.gk_position_at_save else None),
                        "gk_dive_time": round(shot_res.gk_dive_time, 3),
                    },
                ))
            else:
                # wide | blocked | woodwork. A ball into the wall is BLOCKED,
                # which is a different event from a shot off target and is
                # already a first-class type in this engine — collapsing the
                # two would hide the single most important thing a wall does.
                _blk_name = getattr(getattr(_blk, "player", None), "name", "")
                result.add(cls.make_event(
                    minute,
                    (EventType.SHOT_BLOCKED if outcome == "blocked"
                     else EventType.SHOT_OFF_TARGET),
                    att_team, taker.name,
                    phase, gs, xg=xg, outcome=False,
                    location_x=fk_x, location_y=fk_y,
                    end_x=end_x, end_y=end_y,
                    metadata={**base_meta, "blocked_by": _blk_name},
                ))

            result.player_distance_stats = cls._accumulate_physics_stats(
                fk_episode, position_engine, minute)
            for _ev in result.events:
                if _ev.event_type in SHOT_SPEED_EVENT_TYPES:
                    _stamp_shot_speed(
                        _ev, shot_res.ball_speed_at_contact,
                        max(0.1, shot_res.flight_time))
        else:
            # Crossed free kick — becomes like a corner
            fk_cross_event = cls.make_event(
                minute, EventType.FREEKICK_CROSS, att_team, taker.name,
                phase, gs, location_x=fk_x, location_y=fk_y,
                metadata={"routine": routine.value} if routine is not None else {},
            )
            result.add(fk_cross_event)
            # Resolve like a corner (Feature #3: the corner routine shapes the
            # crossed free kick too; the position engine stays threaded in).
            # A crossed free kick is NOT a corner: same aerial duel, but
            # no box to fill and a wall instead. `is_corner=False` keeps
            # the corner crowding out of this path.
            sub = cls._corner_chain(minute, att_team, def_team, att_players,
                                    def_players, state, attacks_right,
                                    position_engine=position_engine,
                                    routine=routine,
                                    delivery_origin=(fk_x, fk_y),
                                    is_corner=False)
            # Inherit events (minus the duplicate corner taken)
            result.events.extend(sub.events[1:])
            # The sub-chain's own CORNER_TAKEN is discarded as a duplicate, but
            # it is the one carrying the resolved delivery endpoint. Copy it
            # across so the crossed free kick is a tracked delivery like every
            # other pass, rather than an origin with no destination — which is
            # what left every set-piece key pass zero-length.
            if sub.events:
                _sub_delivery = sub.events[0]
                fk_cross_event.end_x = _sub_delivery.end_x
                fk_cross_event.end_y = _sub_delivery.end_y
                if _sub_delivery.secondary_player:
                    fk_cross_event.secondary_player = _sub_delivery.secondary_player
                fk_cross_event.metadata = {
                    **(fk_cross_event.metadata or {}),
                    "delivery_end_x": _sub_delivery.metadata.get("delivery_end_x"),
                    "delivery_end_y": _sub_delivery.metadata.get("delivery_end_y"),
                    "delivery_end_source": _sub_delivery.metadata.get(
                        "delivery_end_source"),
                    "delivery_target": _sub_delivery.metadata.get("delivery_target"),
                }
            result.goal_scored    = sub.goal_scored
            result.goal_team      = sub.goal_team
            result.goal_scorer    = sub.goal_scorer
            result.goal_assistant = taker.name  # Taker gets the assist
            result.xg_generated   = sub.xg_generated
            result.xa_generated   = sub.xa_generated
            result.shot_on_target = sub.shot_on_target
            # VAR review may have ruled the sub-goal out inside _corner_chain
            # — the offside flags must survive the hand-off or the engine
            # would credit a goal whose GOAL event was already stripped.
            result.delayed_offside = sub.delayed_offside
            result.offside_detected = sub.offside_detected
            result.offside_x        = sub.offside_x
            result.offside_y        = sub.offside_y
            result.offside_player   = sub.offside_player
            result.offside_team     = sub.offside_team
            result.unflagged_offside_player = sub.unflagged_offside_player
            result.unflagged_offside_x      = sub.unflagged_offside_x
            result.unflagged_offside_y      = sub.unflagged_offside_y

        # ── SHOT-SPEED STAMP (Opta-style) ─────────────────────────
        # Direct free-kick attempts: fired at pace — base shot velocity lifted
        # by the taker's dead-ball skill.
        for _ev in result.events:
            if _ev.event_type in SHOT_SPEED_EVENT_TYPES:
                _fk_lift = 0.90 + (float(getattr(
                    getattr(taker.dna, "technical", None), "free_kick", 50.0,
                )) / 100.0) * 0.30
                _spd = _shot_speed_for_player(taker, "foot") * _fk_lift
                _goal_x = 105.0 if attacks_right else 0.0
                _stamp_shot_speed(
                    _ev, _spd,
                    max(0.1, abs(_goal_x - _ev.location_x) / _spd),
                )

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)
        return result

    # ── HELPERS ───────────────────────────────────────────────

    @classmethod
    def _pick_sp_taker(cls, players: List[PlayerProfile], situation: str = "setpiece", **ctx) -> PlayerProfile:
        outfield = [p for p in players if p.position != "GK"] or players

        if situation == "corner":
            eligible = [p for p in outfield if p.position != "CB"]
            pool = eligible if eligible else outfield
        else:
            pool = outfield

        if situation == "penalty":
            return cls._pick_penalty_taker(pool)

        def _score(p: PlayerProfile) -> float:
            score = 1.0
            score *= (0.4 + p.dna.technical.free_kick / 100.0)

            creative_specs = {"creator", "grand_creator", "sup_vision", "playmaker", "dl_playmaker"}
            if any(spec in p.dna.specialties for spec in creative_specs):
                score *= 3.0

            if situation == "corner":
                if p.position in ("LW", "RW", "LB", "RB"):
                    score *= 2.5
                elif p.position in ("CAM", "CM", "CDM"):
                    score *= 1.2
                side = ctx.get("corner_side")
                if side == "right" and p.dna.preferred_foot == "right":
                    score *= 1.4
                elif side == "left" and p.dna.preferred_foot == "left":
                    score *= 1.4
                score *= (0.6 + p.dna.technical.crossing / 100.0)

            elif situation == "freekick":
                fk_type = ctx.get("freekick_type", "crossed")
                if fk_type == "direct":
                    score *= (0.5 + p.dna.technical.penalty_taking / 100.0)
                    score *= (0.6 + p.dna.technical.finishing / 100.0)
                else:
                    score *= (0.7 + p.dna.mental.vision / 100.0)
                    score *= (0.6 + p.dna.technical.crossing / 100.0)

            return max(score, 0.1)

        weights = [_score(p) for p in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    @staticmethod
    def _pick_penalty_taker(pool: List[PlayerProfile]) -> PlayerProfile:
        """
        Strict spot-kick hierarchy, fully deterministic (no random draw):
          1. ST/CF   — first choice, always
          2. LW/RW   — take over only when an ST exists but is NOT the team star
          3. CAM     — third choice
          4. CM/CDM  — almost never (only when no attacker exists at all)
          5. Fullbacks, then CB as absolute last resort — never ahead of a striker
        """
        def _best(cands: List[PlayerProfile]) -> Optional[PlayerProfile]:
            if not cands:
                return None
            return max(cands, key=lambda p: (
                p.dna.technical.penalty_taking,
                p.dna.mental.composure,
                p.dna.overall_rating,
            ))

        sts     = [p for p in pool if p.position in ("ST", "CF")]
        wingers = [p for p in pool if p.position in ("LW", "RW")]
        cams    = [p for p in pool if p.position == "CAM"]
        mids    = [p for p in pool if p.position in ("CM", "CDM")]

        star = max(pool, key=lambda p: p.dna.overall_rating)

        # ST is strictly first choice unless the team's star is someone else
        # and that star is a winger.
        if sts:
            if not wingers or star in sts or star not in wingers:
                taker = _best(sts)
                if taker is not None:
                    return taker

        for tier in (wingers, cams, mids):
            taker = _best(tier)
            if taker is not None:
                return taker

        fullbacks = [p for p in pool if p.position in ("LB", "RB")]
        cbs       = [p for p in pool if p.position == "CB"]
        return (_best(fullbacks) or _best(cbs) or pool[0])


# ─────────────────────────────────────────────
# 4. TRANSITION CHAIN
# Press → turnover → counter-attack
# ─────────────────────────────────────────────

class TransitionChain(BaseChain):
    """
    Models the press-win-counter cycle.
    The most dynamic and momentum-shifting chain in football.

    Press succeeds → ball won high → immediate counter
    Counter quality depends on: speed of players available,
    number of players forward, defending team's recovery.
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        pressing_team: str,
        retreating_team: str,
        press_players: List[PlayerProfile],
        retreat_players: List[PlayerProfile],
        press_profile: "TeamProfile",
        state: MatchState,
        position_engine: Optional[PositionEngine] = None,
        attacks_right: bool = True,
        counterpress: Optional[Dict[str, Any]] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        # Who presses?
        presser = cls._pick_presser(press_players)
        pressed = cls._pick_pressed_player(retreat_players)
        if not presser or not pressed:
            return result

        # ── WHERE THE PRESS HAPPENS ─────────────────────────────────
        # This used to be `random.uniform(55, 85)` / `random.uniform(10, 58)`.
        # A press happens where the ball is, so drawing a zone produced an
        # event at a place no player and no ball ever was — and because
        # `_absorb_chain` snaps the presser to `location_*`, the draw also
        # teleported the presser and banked the gap as distance he covered
        # (measured: 30 snaps / 843 m per match at `match_engine.py:5764`
        # alone, versus 159/159 already at the ball for the OTHER press site).
        #
        # The pressed man holds the ball, so his tracked position IS the press
        # location. `tracked_position` rather than `get_position`, because the
        # latter answers (50.0, 34.0) for an untracked player — the centre
        # spot, which reads as a measurement. When there is genuinely no
        # tracked state (no position engine attached), the draw is kept but
        # LABELLED, so a fabricated coordinate is never indistinguishable from
        # a measured one downstream. Same pattern as CHANCE_CREATED's
        # `origin_known` / `origin_source`.
        loc = (position_engine.tracked_position(pressed.name)
               if position_engine is not None else None)
        if loc is not None:
            press_x, press_y = loc
            loc_source = "tracked"
        else:
            press_x = random.uniform(55, 85)
            press_y = random.uniform(10, 58)
            loc_source = "untracked_draw"

        # ── COUNTERPRESS BURST (P2) ────────────────────────────────
        # When the pressing team has just lost possession, its first
        # defensive action is anchored at the recovery zone (right where
        # the ball was lost) and its press is more likely to succeed — the
        # "win it back immediately" hunt at the exact loss location.
        cp_boost = 1.0
        if counterpress and counterpress.get("active"):
            press_x = counterpress.get("x", press_x)
            press_y = counterpress.get("y", press_y)
            loc_source = "counterpress_zone"
            cp_boost = state.COUNTERPRESS_INTENSITY_MULT

        result.add(cls.make_event(
            minute, EventType.PRESS, pressing_team, presser.name,
            phase, gs,
            secondary_player=pressed.name,
            location_x=press_x,
            location_y=press_y,
            metadata={"counterpress": bool(cp_boost > 1.0),
                      "location_source": loc_source},
        ))

        # Press success?
        press_success_rate = min(
            0.96,
            press_profile.press_success_rate
            * (presser.dna.physical.pace / 100.0 * 0.3 + 0.7)
            * (1.0 - pressed.dna.press_resistance / 100.0 * 0.4)
            * cp_boost,
        )
        # Checkpoint 7: soul pressers (Pressing Evangelist, Sweeper Sage)
        # win the ball back more often — their defining trait. Also flows
        # into the counter that follows, so a pressing soul doesn't just
        # press more, they actually TRANSFORM more presses into chances.
        #press_success_rate = SoulApplicator.modify_press_success(
            #presser, press_success_rate, state, pressing_team)

        if random.random() < press_success_rate:
            # Emit the ACTUAL physical action that won the ball
            # (not a synthetic PRESS_SUCCESS — this is how StatsBomb logs it)
            ball_winning_action = random.choices(
                [EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.BALL_RECOVERY],
                weights=[0.40, 0.35, 0.25]
            )[0]
            result.add(cls.make_event(
                minute, ball_winning_action, pressing_team, presser.name,
                phase, gs,
                secondary_player=pressed.name,
                location_x=press_x,      # Spatial continuity: same coords as press
                location_y=press_y,
                outcome=True,
                metadata={"from_pressure": True}
            ))
            result.possession_lost = True  # Retreating team loses ball

            # ── COUNTER ATTACK — anchored at press coordinates ──
            counter_result = cls._generate_counter(
                minute, pressing_team, retreating_team,
                press_players, retreat_players, press_profile, state,
                anchor_x=press_x, anchor_y=press_y,   # spatial anchor
                position_engine=position_engine,
                attacks_right=attacks_right,
            )
            result.events.extend(counter_result.events)
            result.goal_scored    = counter_result.goal_scored
            result.goal_team      = counter_result.goal_team
            result.goal_scorer    = counter_result.goal_scorer
            result.goal_assistant = counter_result.goal_assistant
            result.xg_generated   = counter_result.xg_generated
            result.xa_generated   = counter_result.xa_generated
            result.shot_on_target = counter_result.shot_on_target
        else:
            # Press failed — player played through.
            # This event used to pass NO location at all, which is not the
            # same as passing none: `MatchEvent.location_x`/`location_y`
            # DEFAULT to (50.0, 34.0) — the centre spot
            # (`match_engine.py:592`) — and `_absorb_chain:5764`'s
            # `if event.location_x is not None` guard cannot see a default.
            # So the man who played through the press was recorded as having
            # materialised at the centre circle, and the gap banked as distance
            # he covered. Measured 24 such events per match, 22 of them
            # snapping >=12 m (602 m). `pressed` is the man ON THE BALL, so
            # the press location above is his actual position.
            result.add(cls.make_event(
                minute, EventType.PASS, retreating_team, pressed.name,
                phase, gs, outcome=True,
                location_x=press_x,
                location_y=press_y,
                metadata={"press_resistance": True,
                          "location_source": loc_source,
                          "body_part": "right_foot" if pressed.dna.preferred_foot == "right" else "left_foot"}
            ))

        return result

    @classmethod
    def _generate_counter(
        cls, minute, counter_team, defending_team,
        counter_players, def_players, counter_profile, state,
        anchor_x: float = None, anchor_y: float = None,
        position_engine: Optional[PositionEngine] = None,
        attacks_right: bool = True,
    ) -> ChainResult:
        """
        Fast break counter-attack chain.
        anchor_x/y: spatial anchor from the preceding press event.
        All carry/pass coordinates flow from this anchor to maintain
        spatial continuity in the event log.
        attacks_right: direction of the RETREATING team (the frame the
        transition was dispatched in). The counter team attacks the
        opposite way.
        """
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        counter_attacks_right = not attacks_right

        # Pick counter carrier (fast players)
        carrier = cls._pick_fast_player(counter_players)
        # ── WHO FINISHES THE COUNTER ──────────────────────────────────────
        # This used to be
        #     shooter = cls._pick_shooter(counter_players, exclude=carrier…)
        # which made TWO mistakes at once. It excluded the carrier, so the man
        # holding the ball was STRUCTURALLY INCAPABLE of scoring the counter —
        # and because `if shooter != carrier` then gated the only branch that
        # lets him shoot, the solo-run-and-shot branch below was UNREACHABLE
        # DEAD CODE. In the engine's own counters somebody else always
        # finished.
        #
        # Now: ~55% of counters are squared to a team-mate (the previous
        # rate), otherwise the man who carried it in goes himself. Both
        # branches are reachable and the carrier can score.
        #
        # The carrier pick itself is still ability-weighted rather than
        # tracked, because `DefensiveChain`'s ball-winner is not threaded into
        # this call. That is a ROLE pick ("who leads the break"), not a false
        # claim about where the ball is — the CARRY event records him at the
        # anchor coordinate, so his position is real. Recorded, not changed:
        # the fix is to thread the ball-winner, not to guess it here.
        _squares_it = random.random() < 0.55
        shooter = (cls._pick_shooter(counter_players, exclude=carrier.name)
                   if (_squares_it and carrier) else None)
        if not carrier:
            return result

        # Spatial anchor: counter starts from WHERE the ball was won
        # not from a randomised position. This ensures telemetry continuity.
        x = anchor_x if anchor_x is not None else random.uniform(55, 75)
        y = anchor_y if anchor_y is not None else random.uniform(15, 53)

        # Carry forward fast from the anchor. Direction is the counter team's
        # own attacking way (counter_attacks_right), so an away-side counter
        # breaks TOWARD the opponent goal (x to 0), not back toward its own.
        carry_dist, adv_ratio = cls._carry_distance_advance(
            carrier, x, counter_profile, is_counter=True
        )
        advance_x = carry_dist * adv_ratio if counter_attacks_right else -carry_dist * adv_ratio
        end_x = cls.clamp_x(x + advance_x, counter_attacks_right)
        vert_range = 4 + (carrier.dna.technical.ball_control / 100) * 8
        end_y = y + (0.5 - random.random()) * vert_range
        end_y = max(5, min(63, end_y))

        result.add(cls.make_event(
            minute, EventType.CARRY, counter_team, carrier.name,
            phase, gs,
            location_x=x, location_y=y,
            end_x=end_x, end_y=end_y,
            outcome=True,
            metadata={"counter": True, "distance": round(carry_dist, 1)}
        ))

        x, y = end_x, end_y

        # Pass to the team-mate, or go himself (see the pick above: the 55%
        # roll has already been taken, so this test is now just "was a
        # team-mate found").
        if shooter:
            pass_adv = random.uniform(6, 14) if counter_attacks_right else -random.uniform(6, 14)
            pass_end_x = cls.clamp_x(x + pass_adv, counter_attacks_right)
            pass_end_y = y + random.uniform(-6, 6)
            pass_end_y = max(5, min(63, pass_end_y))
            result.add(cls.make_event(
                minute, EventType.PASS, counter_team, carrier.name,
                phase, gs,
                secondary_player=shooter.name,
                location_x=x, location_y=y,
                end_x=pass_end_x, end_y=pass_end_y,
                outcome=True,
                metadata={
                    "counter_pass": True,
                    "body_part": cls._foot_for_pass(
                        carrier, x, y, pass_end_x, pass_end_y, counter_attacks_right),
                }
            ))
            # Attack chain anchored at pass end coordinates.
            # The two names are the WHOLE point: without them AttackChain
            # falls back to `_pick_shooter`, a role-weighted draw over the
            # team it was just handed — so it could name a shooter who is
            # neither the receiver of the pass above nor the man who carried
            # it in. The key pass would then name one receiver and the shot
            # would be credited to another, and the two would disagree.
            attack = AttackChain.generate(
                minute, counter_team, defending_team,
                counter_players, def_players,
                counter_profile, counter_profile,
                state, SituationType.FAST_BREAK,
                context_x=pass_end_x, context_y=pass_end_y,
                position_engine=position_engine,
                attacks_right=counter_attacks_right,
                shooter_name=shooter.name,
                assister_name=carrier.name,
            )
            result.events.extend(attack.events)
            result.goal_scored    = attack.goal_scored
            result.goal_team      = attack.goal_team
            result.goal_scorer    = attack.goal_scorer
            result.goal_assistant = attack.goal_assistant
            result.xg_generated   = attack.xg_generated
            result.xa_generated   = attack.xa_generated
            result.shot_on_target = attack.shot_on_target
            # Shooter / assister out, for the same reason as the solo branch.
            # The pair is (receiver of the pass above, the man who passed it),
            # which is exactly what `PossessionChain` publishes as
            # `shoot_player` / `shoot_assister`.
            result.shoot_player   = shooter.name
            result.shoot_assister  = carrier.name
        else:
            # Solo run and shot — anchored at carry end. He dribbled it in, so
            # there is NO assister: an empty string is the honest answer and it
            # is the same rule `_resolve_assister` follows everywhere else. A
            # counter finished off a solo carry is genuinely unassisted.
            attack = AttackChain.generate(
                minute, counter_team, defending_team,
                [carrier], def_players,
                counter_profile, counter_profile,
                state, SituationType.FAST_BREAK,
                context_x=x, context_y=y,
                position_engine=position_engine,
                attacks_right=counter_attacks_right,
                shooter_name=carrier.name,
                assister_name="",
            )
            result.events.extend(attack.events)
            result.goal_scored    = attack.goal_scored
            result.goal_team      = attack.goal_team
            result.goal_scorer    = attack.goal_scorer
            result.goal_assistant = attack.goal_assistant
            result.xg_generated   = attack.xg_generated
            result.shot_on_target = attack.shot_on_target
            # Carry the shooter out for the same reason the other branch does:
            # `_absorb_chain` and the ledger both read `shoot_player`, and a
            # ChainResult whose scorer is only on the GOAL event is the
            # "fixed field is not a fixed feature" trap again.
            result.shoot_player   = carrier.name

        return result

    @classmethod
    def _pick_presser(cls, players) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            cls._outfield_players(players),
            lambda p: (p.dna.physical.pace * 0.4 + p.dna.mental.work_rate * 0.6) / 100.0
            * {"ST": 2.5, "LW": 2.2, "RW": 2.2, "CAM": 1.8, "CM": 1.5}.get(p.position, 1.0)
        )

    @classmethod
    def _pick_pressed_player(cls, players) -> Optional[PlayerProfile]:
        # Real data: 85% of high-press targets are CB/CDM building from back.
        # GK is only pressed when they have the ball and no CB is available.
        # Weight distribution reflects this reality.
        return cls.pick_weighted(
            players,
            lambda p: {
                "CB":  4.5,   # Primary target: centre-backs building out
                "CDM": 3.5,   # Secondary: defensive mid receiving from CB
                "LB":  2.0,   # Fullbacks in possession
                "RB":  2.0,
                "GK":  0.8,   # Rarely pressed directly (clears long instead)
                "CM":  1.0,
            }.get(p.position, 0.5)
        )

    @classmethod
    def _pick_fast_player(cls, players) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            cls._outfield_players(players),
            lambda p: p.dna.physical.pace / 100.0
        )

    @classmethod
    def _pick_shooter(cls, players, exclude=None) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            cls._outfield_players(players),
            lambda p: {"ST": 5.0, "CF": 4.5, "LW": 3.0, "RW": 3.0, "CAM": 2.0}.get(p.position, 0.8),
            exclude=exclude
        )


# ─────────────────────────────────────────────
# 5. DEFENSIVE CHAIN
# Tackle, interception, clearance, block
# ─────────────────────────────────────────────

class DefensiveChain(BaseChain):
    """
    Models defensive actions.
    Called when defender intercepts / challenges / clears.

    Checkpoint 9 — defensive awareness:
        Defenders understand their own (x, y), the ball's (x, y), and the
        goalpost xy of the goal they defend (own_goal_x). They act on a
        shared DANGER LEVEL: the closer the ball is to own_goal_x, the more
        urgent and clearance-biased the reaction. Clearances are split into
        HEADED (aerial ball redirected with the head) and FOOT (kicked away
        with no intended possession) per Opta/StatsBomb, with
        attribute-driven success for each.
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        defending_team: str,
        attacking_team: str,
        def_players: List[PlayerProfile],
        att_players: List[PlayerProfile],
        state: MatchState,
        action_type: str = "tackle",
        context_x: float = None,
        context_y: float = None,
        attacks_right: bool = True,
        referee_strictness: float = 0.5,
        danger_level: float = 0.0,
        ball_aerial: bool = False,
        own_goal_x: float = 105.0,
        position_engine: Optional[PositionEngine] = None,
        ball_z: Optional[float] = None,
        defender_facing_x: Optional[float] = None,
        defender_facing_y: Optional[float] = None,
        opponent_distance: Optional[float] = None,
        stamina: Optional[float] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        defender = cls._pick_defender(
            def_players, action_type,
            position_engine=position_engine,
            x=context_x, y=context_y,
        )
        attacker = cls._pick_attacker(
            att_players,
            position_engine=position_engine,
            x=context_x, y=context_y,
        )
        if not defender:
            return result

        # Spatial continuity: use context coords if provided,
        # otherwise estimate from action type (reaction events happen near the ball)
        if context_x is not None:
            x = context_x + random.uniform(-3, 3)  # small positional noise
            y = (context_y or 34) + random.uniform(-4, 4)
        else:
            # Fallback: position-appropriate defaults
            x = random.uniform(20, 75) if action_type in ("tackle","interception") else random.uniform(75, 103)
            y = random.uniform(8, 60)
        x = max(0, min(105, x))
        y = max(0, min(68, y))

        if action_type == "tackle":
            # ── CRAFT THE TACKLE RATE (calibration anchor) ──────────
            # The DNA roll stays the season-long baseline for how reliably
            # this defender wins the ball; the PHYSICS layer below then
            # voices the live geometry (technique + who physically reached
            # the ball), so a sliding vs standing tackle is a real reach
            # decision, not just a color label on the same dice.
            tackle_rate = DNAFactory.get_tackle_success_rate(defender.dna)

            # 1) Attacker voice — a skilled carrier erodes the defender's
            #    edge. Same attacking-skill differential as the physics-race
            #    path (_dribble_confirmation_gate).
            if attacker is not None:
                tackle_rate -= cls._tackle_attacker_resistance(attacker, defender)

            # 2) Danger panic — a last-ditch challenge is rushed and
            #    mistimed (lower success, higher foul chance below).
            risk = max(0.0, min(1.0, danger_level / 100.0))
            tackle_rate -= 0.10 * risk

            # 3) Aggression/bravery tradeoff — committed tacklers dive in:
            #    less reliable, but when the crunch lands it wins clean more
            #    often (the conversion roll in the failure branch below).
            aggression = float(getattr(
                getattr(defender.dna, "tendencies", None),
                "tackles_aggressively", 0.40))
            bravery = float(getattr(
                getattr(defender.dna, "mental", None), "bravery", 60.0)) / 100.0
            commitment = max(0.0, min(1.0, 0.5 * aggression + 0.5 * bravery))
            tackle_rate *= (1.0 - 0.12 * commitment)

            tackle_rate = max(0.08, min(0.92, tackle_rate))

            # ── PHYSICS-FIRST TACKLE (slide vs standing) ────────────
            # DefensiveChain's tackle is now resolved through a real reach
            # race (geometry_engine.resolve_tackle). The defender's DNA
            # slide_tackle_aggression picks the technique:
            #   * STANDING — short tackle_radius, stays on his feet; wins
            #     only by cleanly reaching the ball, but a miss is clean and
            #     the recovery is quick (few fouls).
            #   * SLIDING  — the extended leg grows the reach envelope
            #     sharply (gets to balls a standing tackle can't), but the
            #     lunge commits the body: worse shoulder leverage, and a
            #     mistimed slide that clips the man is a physical foul.
            # The physics signal (who physically reached the ball) shifts the
            # calibrated tackle_rate, so a decisive geometry edge is rewarded
            # and a cleanly-beaten defender is punished — in proportion to
            # how tight the contest actually is (opponent_distance).
            slide_prob = float(getattr(
                getattr(defender.dna, "tendencies", None),
                "slide_tackle_aggression", aggression))
            technique = "standing"
            physics_won = None
            physics_foul = False
            physics_note = ""
            distance_weight = 0.0
            if (position_engine is not None and defender is not None
                    and attacker is not None):
                def_mp = cls._moving_player(defender, position_engine)
                att_mp = cls._moving_player(attacker, position_engine)
                sign = 1.0 if attacks_right else -1.0
                ball_start = Vec2(x, y)
                carrier_target = Vec2(
                    min(105.0, max(0.0, x + sign * 6.0)),
                    y + (34.0 - y) * 0.1,
                )
                tres = resolve_tackle(
                    def_mp, att_mp, ball_start, carrier_target,
                    slide_prob=slide_prob,
                )
                technique = tres.technique
                physics_won = tres.won
                physics_foul = tres.foul
                physics_note = tres.resolution_note
                if opponent_distance is not None:
                    distance_weight = max(
                        0.0, min(1.0, 1.0 - opponent_distance / 6.5))
                else:
                    distance_weight = 0.6

            success = random.random() < tackle_rate
            if physics_won is not None:
                # The live geometry edge nudges the calibrated roll; the
                # tightness weight keeps it from overriding DNA wholesale
                # when the contest is open.
                edge = 0.16 if physics_won else -0.16
                success = random.random() < max(
                    0.03, min(0.97, tackle_rate + edge * distance_weight))
            clean_won = False

            # ── FAILURE RESOLUTION (foul vs clean-win vs dribble-past) ──
            foul_committed = False
            if not success:
                # A mistimed SLIDE is the physical foul — the extended leg
                # clips the man. Standing misses are clean; a dirty profile
                # and last-ditch panic still lift the foul chance.
                if physics_foul:
                    foul_committed = random.random() < (0.55 + 0.30 * risk)
                else:
                    foul_base = 0.40
                    if defender.dna.tendencies.tackles_aggressively > 0.6:
                        foul_base *= 1.3
                    foul_base *= (1.0 + 0.60 * risk)
                    foul_base *= max(0.5, defender.dna.tendencies.commits_fouls * 2.0)
                    pos_mult = {"CDM": 1.4, "CB": 1.3, "CM": 1.15, "LB": 1.1, "RB": 1.1}.get(defender.position, 1.0)
                    foul_committed = random.random() < (foul_base * pos_mult)

                if not foul_committed:
                    # Clean win on the committed challenge: before conceding
                    # the dribble-past, an aggressive/brave tackler who dove
                    # in has a real chance the crunch actually won the ball.
                    conversion = max(0.0, min(0.30, commitment * 0.30 - 0.04))
                    if random.random() < conversion:
                        success = True
                        clean_won = True

            result.add(cls.make_event(
                minute,
                EventType.TACKLE_WON if success else EventType.TACKLE_LOST,
                defending_team, defender.name,
                phase, gs,
                secondary_player=attacker.name if attacker else None,
                location_x=x, location_y=y,
                outcome=success,
                metadata={
                    "danger_before": round(danger_level, 1),
                    "tackle_rate": round(tackle_rate, 3),
                    "committed": round(commitment, 2),
                    "clean_commitment": clean_won,
                    "technique": technique,
                    "physics_note": physics_note,
                },
            ))
            if not success:
                if foul_committed:
                    result.foul_committed = True
                    result.add(cls.make_event(
                        minute, EventType.FOUL_COMMITTED,
                        defending_team, defender.name,
                        phase, gs,
                        secondary_player=attacker.name if attacker else None,
                        location_x=x, location_y=y,
                        outcome=False,
                        metadata={"from_failed_tackle": True}
                    ))
                    if attacker:
                        result.add(cls.make_event(
                            minute, EventType.FOUL_WON,
                            attacking_team, attacker.name,
                            phase, gs,
                            secondary_player=defender.name,
                            location_x=x, location_y=y,
                            outcome=True,
                            metadata={"drew_foul": True}
                        ))
                    # Card roll from tackle foul — same conversion philosophy
                    # as DisciplineChain: ~13-15% of fouls become a card at a
                    # default ref, scaled by strictness and the tackler's
                    # aggression / discipline record.
                    card_prob = (
                        0.15
                        * (0.55 + 0.80 * referee_strictness)
                        * (defender.dna.tendencies.tackles_aggressively * 1.5 + 0.3)
                        * (0.6 + defender.dna.tendencies.commits_fouls)
                    )
                    if random.random() < card_prob:
                        # Straight-red tail scales with strictness too: a
                        # lenient ref almost never sends a player off for a bad
                        # challenge, a strict one will. Old code used a flat
                        # 0.04, so tackles produced reds every match no matter
                        # how lenient the referee was set.
                        straight_red_chance = 0.04 * (0.15 + referee_strictness)
                        is_red = random.random() < straight_red_chance
                        result.card_issued = True
                        result.card_type = "red" if is_red else "yellow"
                        result.carded_player = defender.name
                        result.carded_team = defending_team
                        result.add(cls.make_event(
                            minute,
                            EventType.RED_CARD if is_red else EventType.YELLOW_CARD,
                            defending_team, defender.name,
                            phase, gs,
                            secondary_player=attacker.name if attacker else None,
                            location_x=x, location_y=y,
                            metadata={"from_tackle": True, "reason": "straight_red" if is_red else "foul"}
                        ))
                else:
                    if attacker:
                        end_dx = cls.clamp_x(x + (3 + random.random() * 5
                                                   if attacks_right else -(3 + random.random() * 5)),
                                             attacks_right)
                        result.add(cls.make_event(
                            minute, EventType.DRIBBLE_SUCCESS,
                            attacking_team, attacker.name,
                            phase, gs,
                            secondary_player=defender.name,
                            location_x=x, location_y=y,
                            end_x=end_dx, end_y=y + (0.5 - random.random()) * 3,
                            outcome=True,
                            metadata={"dribbled_past": True, "paired_with_tackle": True}
                        ))

        elif action_type == "interception":
            base_int = defender.dna.defending.interceptions / 100.0 * 0.7 + 0.2
            success = random.random() < base_int
            if success:
                result.add(cls.make_event(
                    minute, EventType.INTERCEPTION,
                    defending_team, defender.name,
                    phase, gs,
                    location_x=x, location_y=y,
                    outcome=True,
                    metadata={"danger_before": round(danger_level, 1)},
                ))

        elif action_type == "clearance":
            # ── CHECKPOINT 9 + 10: SPATIAL CLEARANCE ENGINE ─────────
            # Opta/StatsBomb split clearances into HEADED and FOOT. The tool
            # is chosen on the Z-AXIS: a ball above hip height (Z > 1.2m)
            # is redirected with the head; a low ball (Z <= 1.2m) is met
            # with a foot. The defender then reads their own body orientation
            # to the ball (optimal / flank / blind), how contested the
            # attempt is (attacker distance), and their fatigue — each
            # amplifying P_fail. Failures are chaotic: a sliced kick, a lost
            # aerial duel, or a catastrophic OWN GOAL in the blind panic.
            ball_x = context_x if context_x is not None else x
            ball_y = context_y if context_y is not None else y

            clearance_kind = cls._clearance_kind(ball_aerial, ball_z)

            if defender_facing_x is not None and defender_facing_y is not None:
                f_x, f_y = defender_facing_x, defender_facing_y
            else:
                f_x, f_y = defender_facing_point(x, y, ball_x, ball_y, own_goal_x)
            rel_angle = calculate_relative_ball_angle(x, y, f_x, f_y, ball_x, ball_y)
            orient = orientation_zone(rel_angle)

            body_part = cls._clearance_body_part(defender, clearance_kind, rel_angle)
            base_rate = cls._clearance_success_rate(defender, clearance_kind, danger_level)
            fail_mult = clearance_failure_multiplier(rel_angle, opponent_distance, stamina)
            success_rate = max(0.05, min(0.92, base_rate / fail_mult))
            outcome = random.random() < success_rate

            # Chaotic failure: own goal (critical) vs slice / aerial loss.
            failure_cause = None
            if not outcome:
                og_p = own_goal_probability(rel_angle, opponent_distance, stamina, danger_level)
                if random.random() < og_p:
                    failure_cause = "own_goal"
                elif clearance_kind == "headed":
                    failure_cause = "aerial_loss" if random.random() < 0.65 else "slice"
                else:
                    failure_cause = "slice" if random.random() < 0.70 else "miscue"

            end_x, end_y, extra = cls._clearance_destination(
                clearance_kind, x, y, own_goal_x, danger_level, outcome
            )
            if outcome:
                end_y = apply_width_bias(end_x, end_y, own_goal_x)
            if not outcome and failure_cause != "own_goal":
                clearance_dest = extra.get("dest", "opponent")
                if clearance_dest == "corner":
                    result.corner_won = True   # Attacking team wins corner
                    result.corner_team = attacking_team
                    # Emit the CORNER_WON event
                    result.add(cls.make_event(
                        minute, EventType.CORNER_WON, attacking_team,
                        attacker.name if attacker else "Attacker",
                        phase, gs,
                        location_x=own_goal_x,  # Goal line
                        location_y=0.0 if y < 34.0 else 68.0,
                        metadata={"from_failed_clearance": True, "clearance_slice": True}
                    ))

            danger_after = danger_after_clearance(
                danger_level, own_goal_x, x, y, end_x, end_y, outcome
            )
            if failure_cause == "own_goal":
                danger_after = 100.0   # the threat was realised — danger PEAKS

            result.add(cls.make_event(
                minute, EventType.CLEARANCE,
                defending_team, defender.name,
                phase, gs,
                location_x=x, location_y=y,   # Spatial continuity: actual location
                end_x=end_x,
                end_y=end_y,
                outcome=outcome,
                body_part=body_part,
                metadata={
                    "clearance_type": clearance_kind,   # "headed" | "foot"
                    "headed": clearance_kind == "headed",
                    "danger_before": round(danger_level, 1),
                    "danger_after": danger_after,
                    "effective": outcome,
                    "relative_angle": round(rel_angle, 1),
                    "orientation_zone": orient,
                    "failure_cause": failure_cause,
                    "contested": round(opponent_distance, 2) if opponent_distance is not None else None,
                    "stamina": round(stamina, 1) if stamina is not None else None,
                    "ball_z": round(ball_z, 2) if ball_z is not None else None,
                    **extra,
                }
            ))

            # ── THROW-IN DETECTION (Checkpoint 8 extension) ────────
            # A clearance to the touchline (end_y at 0/68) puts the ball
            # dead for a throw-in. The clearing DEFENDING team touched the
            # ball last, so the throw is awarded to the ATTACKING team.
            # Without this the ball was silently going out of play with no
            # restart ever queued — throw-ins could never fire in a match.
            if extra.get("dest") == "touchline":
                result.restart_required = True
                result.restart_type = "throw_in"
                result.restart_team = attacking_team
                result.restart_x = end_x
                result.restart_y = end_y
                result.possession_lost = True

            if failure_cause == "own_goal":
                # The panic clearance redirects the ball into the defender's
                # own net. The ATTACKING team is credited; the defender's
                # name goes down as the own goal. The danger PEAKS.
                result.own_goal = True
                result.goal_scored = True
                result.goal_team = attacking_team
                result.goal_scorer = defender.name
                result.possession_lost = True
                result.add(cls.make_event(
                    minute, EventType.OWN_GOAL,
                    defending_team, defender.name,
                    phase, gs,
                    location_x=x, location_y=y,
                    outcome=False,
                    # Own goals never have an assisting player credited.
                    secondary_player=None,
                    metadata={
                        "own_goal": True,
                        "clearance_body_part": body_part,
                        "clearance_type": clearance_kind,
                        "orientation_zone": orient,
                        "relative_angle": round(rel_angle, 1),
                        "danger_before": round(danger_level, 1),
                    },
                ))

        elif action_type == "block":
            # Spatial continuity: blocks happen near the shot origin
            # Checkpoint: realistic block deflections with chaotic physics
            # ~65% of blocks result in corners or dangerous deflections
            block_roll = random.random()
            
            # Calculate chaotic deflection angle
            import math
            # Defender's body orientation creates unpredictable ricochets
            base_deflection_angle = random.uniform(-math.radians(60), math.radians(60))
            deflection_speed = random.uniform(12.0, 28.0)  # m/s
            
            # Determine deflection outcome
            # CALIBRATION (2026-09-17): block->corner 0.52 -> 0.44 -> 0.40 -> 0.37
            # so total corner volume drops toward the real ~4.6/match.
            if block_roll < 0.37:
                # Deflection for corner - most common outcome (increased from 45% to 52%)
                outcome = False
                result.corner_won = True
                result.corner_team = attacking_team
                metadata = {
                    "deflection": "corner",
                    "deflection_angle_degrees": round(math.degrees(base_deflection_angle), 1),
                    "deflection_speed": round(deflection_speed, 1),
                }
                # Emit CORNER_WON event
                result.add(cls.make_event(
                    minute, EventType.CORNER_WON, attacking_team,
                    attacker.name if attacker else "Attacker",
                    phase, gs,
                    location_x=own_goal_x,
                    location_y=0.0 if y < 34.0 else 68.0,
                    metadata={"from_defensive_block": True, "chaotic_deflection": True}
                ))
            elif block_roll < 0.68:
                # Dangerous deflection: loops to another attacker (increased range to 68%)
                outcome = False
                metadata = {
                    "deflection": "to_attacker",
                    "deflection_angle_degrees": round(math.degrees(base_deflection_angle), 1),
                    "second_ball": True,
                }
            else:
                # Clean block: defender controls the ricochet
                outcome = True
                metadata = {
                    "deflection": "safe",
                    "controlled_block": True,
                }

            metadata["danger_before"] = round(danger_level, 1)
            metadata["blocked_type"] = "shot"
            result.add(cls.make_event(
                minute, EventType.BLOCK,
                defending_team, defender.name,
                phase, gs,
                secondary_player=attacker.name if attacker else None,
                location_x=x, location_y=y,   # Spatial continuity
                outcome=outcome,
                metadata=metadata,
            ))

        # Ball recovery only follows a genuinely successful defensive action.
        # (Checkpoint 6 fix: interception previously counted as a clean win
        # unconditionally, even on a failed interception roll — now it
        # requires the actual success outcome, same as tackle/clearance/block.)
        clean_win = (
            (action_type == "tackle" and not result.foul_committed) or
            (action_type == "interception" and result.events and result.events[-1].outcome) or
            (action_type in ("clearance", "block") and
             result.events and result.events[-1].outcome)
        )
        if clean_win:
            result.add(cls.make_event(
                minute, EventType.BALL_RECOVERY,
                defending_team, defender.name,
                phase, gs,
                location_x=x, location_y=y,
                outcome=True,
            ))

        # Checkpoint 6: tell the engine the attacking team's possession of
        # this sequence is over — either the defense genuinely won the ball
        # (clean_win) or the ball went dead for a corner (corner_won). Without
        # this, the engine would carry on into the shot phase for a team that
        # just had the ball taken off it.
        if clean_win or result.corner_won:
            result.possession_lost = True

        return result

    @classmethod
    def _pick_defender(cls, players, action_type,
                       position_engine: Optional[PositionEngine] = None,
                       x: float = None, y: float = None) -> Optional[PlayerProfile]:
        preferred = {
            "tackle":        ["CB", "CDM", "CM", "LB", "RB"],
            "interception":  ["CB", "CDM", "LB", "RB"],
            "clearance":     ["CB", "LB", "RB", "CDM"],
            "block":         ["CB", "CDM", "CM"],
        }.get(action_type, ["CB", "CDM"])
        pool = cls._outfield_players(players)
        if position_engine is not None and x is not None and y is not None:
            # Checkpoint 9: the NEAREST defender reacts — a CB stranded on the
            # opposite side of the pitch cannot win this duel. Positional
            # plausibility at the ball's (x, y) grounds the pick.
            return cls.pick_weighted_spatial(
                pool,
                lambda p: 3.5 if p.position in preferred else 0.8,
                position_engine, x, y,
            )
        return cls.pick_weighted(
            pool,
            lambda p: 3.5 if p.position in preferred else 0.8
        )

    @classmethod
    def _pick_attacker(cls, players,
                       position_engine: Optional[PositionEngine] = None,
                       x: float = None, y: float = None) -> Optional[PlayerProfile]:
        weight_fn = lambda p: {"ST": 4.0, "LW": 3.0, "RW": 3.0, "CAM": 2.5}.get(p.position, 1.0)
        if position_engine is not None and x is not None and y is not None:
            return cls.pick_weighted_spatial(players, weight_fn, position_engine, x, y)
        return cls.pick_weighted(players, weight_fn)

    # ── CHECKPOINT 9: HEADED / FOOT CLEARANCE HELPERS ─────────────

    @staticmethod
    def _clearance_kind(ball_aerial: bool, ball_z: Optional[float] = None) -> str:
        """Z-AXIS tool selection: a ball above hip height (Z > 1.2m) is
        redirected with the HEAD; a low ball (Z <= 1.2m) is met with a FOOT.
        Falls back to the aerial-flag heuristic when no height is known."""
        if ball_z is not None:
            return "headed" if ball_z > 1.2 else "foot"
        return "headed" if ball_aerial else "foot"

    @classmethod
    def _clearance_body_part(cls, defender: PlayerProfile, clearance_kind: str,
                             rel_angle: float = 0.0) -> str:
        """Which body part clears the ball. Headed clearances always use the
        head. Foot clearances honour the defender's preferred foot inside the
        optimal dead-zone (±30°) and switch to the flank foot outside it —
        the sign of the relative angle dictates left vs right, mirroring a
        real centre-back's stance."""
        if clearance_kind == "headed":
            return "head"
        foot = getattr(getattr(defender, "dna", None), "preferred_foot", "right")
        return clearance_foot_for_angle(rel_angle, foot)

    @classmethod
    def _clearance_success_rate(cls, defender: PlayerProfile, clearance_kind: str,
                                danger_level: float) -> float:
        """
        Attribute-correct success for each clearance type, panicked slightly
        by CRITICAL danger (a panicked hoof is sloppier than a calm one).

        Headed:   aerial dominance (jump+heading+bravery), heading, composure,
                  clearing technique.
        Foot:     defending clearances, marking, composure, anticipation.
        """
        dna = getattr(defender, "dna", None)
        if dna is None:
            return 0.55
        if clearance_kind == "headed":
            base = (
                dna.aerial_dominance / 100.0 * 0.55
                + dna.technical.heading / 100.0 * 0.25
                + dna.mental.composure / 100.0 * 0.10
                + dna.defending.clearances / 100.0 * 0.10
            )
        else:
            base = (
                dna.defending.clearances / 100.0 * 0.40
                + dna.defending.marking / 100.0 * 0.15
                + dna.mental.composure / 100.0 * 0.25
                + dna.mental.anticipation / 100.0 * 0.20
            )
        panic = 1.0 - 0.12 * (max(0.0, min(1.0, danger_level / 100.0)))
        return max(0.15, min(0.90, base * panic))

    @classmethod
    def _clearance_destination(
        cls, clearance_kind: str,
        from_x: float, from_y: float,
        own_goal_x: float, danger_level: float,
        outcome: bool,
    ) -> Tuple[float, float, Dict]:
        """
        Where the cleared ball ends up — always AWAY from own_goal_x.

        Headed clearances are short and angled toward the touchlines (get it
        away, don't invite the second ball through the middle). Foot
        clearances are longer; under CRITICAL danger they become a big hoof,
        more likely to go out of play (safe) but also more likely to be
        scuffed. Failures leave the ball in the danger zone.
        
        CHECKPOINT: Panic clearances under extreme pressure frequently slice
        over the byline for corners, especially from wide defensive positions.
        """
        away = -1.0 if own_goal_x == 105.0 else 1.0   # sign: away from own goal
        extra: Dict = {"dest": "field"}

        if not outcome:
            # Scuffed / fails to clear — the ball stays in the danger zone.
            # Under HIGH danger (85+), failed clearances often slice backwards
            # over own byline for a corner
            import math
            
            # Calculate panic factor based on danger and position
            panic_factor = danger_level / 100.0
            dist_from_goal = abs(from_x - own_goal_x)
            is_wide_position = from_y < 20.0 or from_y > 48.0
            
            # Base corner chance for failed clearances
            corner_chance = 0.35  # Base 35%
            if danger_level >= 85.0:
                corner_chance += 0.25  # Panic increases slicing
            if dist_from_goal < 18.0:  # Very close to goal
                corner_chance += 0.20
            if is_wide_position:  # Wide defenders slice more often
                corner_chance += 0.15
            
            if random.random() < min(0.75, corner_chance):
                # Sliced backwards over own byline
                extra["dest"] = "corner"
                extra["slice_direction"] = "backwards"
                end_x = from_x + away * random.uniform(-8, -2)  # Goes backwards
                end_y = from_y + random.uniform(-12, 12)
            else:
                # Stays in danger zone but doesn't go out
                extra["dest"] = "opponent"
                end_x = from_x + away * random.uniform(2, 9)
                end_y = from_y + random.uniform(-8, 8)
            
            end_y = max(4, min(64, end_y))
            return round(end_x, 1), round(end_y, 1), extra

        if clearance_kind == "headed":
            # Short, safe, angled toward a touchline.
            # Headers can also deflect awkwardly for corners under pressure
            end_x = from_x + away * random.uniform(22, 45)
            
            # Check if clearance goes out
            if random.random() < 0.30:
                extra["dest"] = "touchline"
                end_y = 0.0 if random.random() < 0.5 else 68.0
            elif danger_level >= 80.0 and random.random() < 0.20:
                # High danger headers can deflect backwards
                extra["dest"] = "corner"
                extra["header_mishit"] = True
                end_x = from_x + away * random.uniform(-5, 10)
                end_y = random.uniform(6, 62)
            else:
                if random.random() < 0.65:
                    end_y = random.uniform(6, 18) if random.random() < 0.5 else random.uniform(50, 62)
                else:
                    end_y = random.uniform(24, 44)
        else:
            # Long, decisive hoof toward the safe midfield band.
            if danger_level >= 85:
                end_x = from_x + away * random.uniform(38, 60)   # big boot
                if random.random() < 0.35:
                    extra["dest"] = "touchline"
                    end_y = 0.0 if random.random() < 0.5 else 68.0
                else:
                    end_y = random.uniform(8, 60)
            else:
                end_x = from_x + away * random.uniform(30, 48)
                if random.random() < 0.20:
                    extra["dest"] = "touchline"
                    end_y = 0.0 if random.random() < 0.5 else 68.0
                else:
                    end_y = random.uniform(8, 60) if random.random() < 0.7 \
                        else random.uniform(24, 44)

        end_x = max(2.0, min(103.0, end_x))
        end_y = max(0.0, min(68.0, end_y))
        return round(end_x, 1), round(end_y, 1), extra


# ─────────────────────────────────────────────
# 6. DISCIPLINE CHAIN
# Foul → card → consequences
# ─────────────────────────────────────────────

class DisciplineChain(BaseChain):
    """
    Models foul → card → player reaction → game consequence.

    Consequences:
        Yellow → player on a booking (second = red)
        Red    → team down to 10, momentum swing
        Penalty → if in the box
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        fouling_team: str,
        fouled_team: str,
        fouling_players: List[PlayerProfile],
        fouled_players: List[PlayerProfile],
        state: MatchState,
        referee_strictness: float = 0.5,
        x: float = None,
        y: float = None,
        attacks_right: bool = True,
        booked_players: Dict[str, int] = None,
        box_penalty_chance: Optional[float] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        fouler = cls._pick_fouler(fouling_players)
        # A keeper "drew foul" is only realistic when he races out to the
        # edge of his box; the foul x here is a free draw (25–85) when the
        # chain is fired without context, so the keeper otherwise gets
        # logged as fouled 85m upfield — inflating his map node. Restrict
        # the victim pool to outfielders (the fouler can still be a GK: a
        # rush-out challenge that clips an attacker is a genuine keeper foul).
        victim = cls._pick_victim(
            [p for p in fouled_players if p.position != "GK"] or fouled_players
        )
        if not fouler:
            return result

        x = x or random.uniform(25, 85)
        y = y or random.uniform(5, 63)

        result.foul_committed = True

        result.add(cls.make_event(
            minute, EventType.FOUL_COMMITTED, fouling_team, fouler.name,
            phase, gs,
            secondary_player=victim.name if victim else None,
            location_x=x, location_y=y,
            metadata={"in_box": PitchZone.is_in_box(x, attacks_right=attacks_right)}
        ))

        # FOUL_WON event for the fouled player
        if victim:
            result.add(cls.make_event(
                minute, EventType.FOUL_WON, fouled_team, victim.name,
                phase, gs,
                secondary_player=fouler.name,
                location_x=x, location_y=y,
                outcome=True,
                metadata={"drew_foul": True}
            ))

        # Penalty if in the box?
        in_penalty_box = PitchZone.is_in_box(x, attacks_right=attacks_right)
        goal_mouth = (x < 103) if attacks_right else (x > 2)
        if in_penalty_box and goal_mouth:
            # PHYSICS-GROUNDED CONVERSION. A box foul alone is not a penalty —
            # the ref gives a spot kick when the defender's lunge actually
            # DENIES a clear scoring opportunity. The MatchEngine computes that
            # conviction from the live defensive danger (ball inside the box,
            # defenders scrambling) and passes it here. Defaults to the legacy
            # value only when the chain is called without context (tests /
            # standalone demo), so existing behaviour is unchanged.
            box_conv = 0.65 if box_penalty_chance is None else box_penalty_chance
            box_conv = max(0.0, min(1.0, box_conv))
            if random.random() < box_conv:
                result.penalty_won = True
                result.add(cls.make_event(
                    minute, EventType.PENALTY_WON, fouled_team,
                    victim.name if victim else "Unknown",
                    phase, gs, location_x=x, location_y=y,
                ))

        # Card probability
        from match_engine import PhaseEngine

        # Base probability influenced by player personality
        card_risk = 1.0
        if hasattr(fouler.dna, "personality") and fouler.dna.personality:
            card_risk = fouler.dna.personality.card_risk_mult

        # Check if player is already booked (second yellow risk)
        already_booked = False
        if booked_players is not None:
            already_booked = fouler.name in booked_players

        # Real football: roughly 15% of fouls become a card (a ~3.5-4
        # yellow match off ~25 fouls). Referee strictness is the main dial,
        # but it must reach a genuinely LOW rate at 0.0 (lenient) and only a
        # modestly higher one at 1.0 (strict) — the old (0.45 + 1.1*strict)
        # floor still booked ~45% of the baseline even from the most lenient
        # referee, which is what made cards (and the straight-red tail that
        # hangs off them) appear in essentially every match regardless of
        # strictness. The new floor is ~0.55x and the ceiling ~1.35x, so a
        # 0.0 ref is truly lenient and a 1.0 ref is firm without emptying a
        # whole team into the book.
        card_prob = (
            0.15
            * PhaseEngine.card_mult(phase)
            * (0.55 + 0.80 * referee_strictness)
            * (0.6 + fouler.dna.tendencies.commits_fouls)
            * card_risk
        )

        # Already booked players are much more likely to get a second yellow
        if already_booked:
            card_prob *= 1.8

        # Dangerous foul = higher card probability
        in_att_third = x > 70 if attacks_right else x < 35
        if in_att_third:
            card_prob *= 1.3
        if fouler.dna.tendencies.tackles_aggressively > 0.60:
            card_prob *= 1.15

        if random.random() < card_prob:
            # If already on a yellow, second yellow = automatic red
            if already_booked:
                is_straight_red = False  # It's a second-yellow red
                second_yellow = True
            else:
                # Straight-red chance is GATED by strictness: a lenient ref
                # (0.0) almost never reaches for a straight red, while a
                # strict ref (1.0) does so at roughly the old baseline. The
                # previous code used a flat 0.06 regardless of strictness, so
                # red cards were issued in basically every match no matter how
                # lenient the referee was supposed to be.
                straight_red_chance = 0.05 * (0.15 + referee_strictness)
                is_straight_red = random.random() < straight_red_chance
                second_yellow = False

            card_type = "red" if (is_straight_red or second_yellow) else "yellow"
            reason = "second_yellow" if second_yellow else ("straight_red" if is_straight_red else "foul")
            result.card_issued   = True
            result.card_type     = card_type
            result.carded_player = fouler.name
            result.carded_team   = fouling_team

            result.add(cls.make_event(
                minute,
                EventType.RED_CARD if card_type == "red" else EventType.YELLOW_CARD,
                fouling_team, fouler.name,
                phase, gs,
                location_x=x, location_y=y,
                metadata={"reason": reason}
            ))

        return result

    @classmethod
    def _pick_fouler(cls, players) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            players,
            lambda p: (
                p.dna.tendencies.commits_fouls * 3.0
                * {"CDM": 1.5, "CM": 1.2, "CB": 1.3, "LB": 1.1, "RB": 1.1}.get(p.position, 1.0)
            )
        )

    @classmethod
    def _pick_victim(cls, players) -> Optional[PlayerProfile]:
        return cls.pick_weighted(
            players,
            lambda p: (
                p.dna.tendencies.dives * 2.0
                + {"LW": 1.5, "RW": 1.5, "CAM": 1.2, "ST": 1.1}.get(p.position, 0.8)
            )
        )


# ─────────────────────────────────────────────
# 7. SUBSTITUTION CHAIN
# Player change with tactical context
# ─────────────────────────────────────────────

class SubstitutionChain(BaseChain):
    """
    Models a substitution event.
    Records who came on, who went off, and at what minute.
    Tactical context affects subsequent chain probabilities.
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        team: str,
        player_off: PlayerProfile,
        player_on: PlayerProfile,
        state: MatchState,
        reason: str = "tactical",  # "tactical" | "injury" | "chasing_game" | "protecting_lead"
        position_engine: Optional[PositionEngine] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        # A substitution happens WHERE THE OUTGOING PLAYER IS. This event used
        # to pass no location at all, which is not the same as passing none:
        # `MatchEvent.location_x`/`location_y` DEFAULT to (50.0, 34.0) — the
        # centre spot (`match_engine.py:592`) — so it was exported as though
        # every change happened on the centre circle.
        #
        # NOTE this chain has ZERO call sites: the live path builds the
        # SUBSTITUTION event directly in `MatchEngine._execute_substitution`,
        # which is why the same defect had to be fixed there too. Recording
        # that rather than deleting it — but do not read a green guard here as
        # evidence the live substitution path is covered. It is not.
        loc = (position_engine.tracked_position(player_off.name)
               if position_engine is not None else None)
        loc_x, loc_y = loc if loc is not None else (50.0, 34.0)

        result.add(cls.make_event(
            minute, EventType.SUBSTITUTION, team,
            player_off.name,
            phase, gs,
            secondary_player=player_on.name,
            location_x=loc_x,
            location_y=loc_y,
            metadata={
                "player_off": player_off.name,
                "player_on": player_on.name,
                "reason": reason,
                "position_off": player_off.position,
                "position_on":  player_on.position,
                "location_source": "tracked" if loc is not None
                                   else "untracked_default",
            }
        ))

        return result


# ─────────────────────────────────────────────
# CHAIN DISPATCHER — MatchEngine entry point
# ─────────────────────────────────────────────

class ChainDispatcher:
    """
    Single entry point for the MatchEngine to call chains.
    Picks the right chain based on context and returns ChainResult.
    """

    @staticmethod
    def possession(
        minute, attacking_team, players, team_profile, state, seq_length,
        defending_players=None, position_engine=None,
        context_x=None, context_y=None,
        attacks_right: bool = True,
        def_press_intensity: Optional[float] = None,
        def_style_key: Optional[str] = None,
        att_style_key: Optional[str] = None,
        counterpress: Optional[Dict[str, Any]] = None,
        coach_instructions: Any = None,
    ) -> ChainResult:
        return PossessionChain.generate(
            minute, attacking_team, players, team_profile, state, seq_length,
            defending_players=defending_players, position_engine=position_engine,
            context_x=context_x, context_y=context_y,
            attacks_right=attacks_right,
            def_press_intensity=def_press_intensity,
            def_style_key=def_style_key,
            att_style_key=att_style_key,
            counterpress=counterpress,
            coach_instructions=coach_instructions,
        )

    @staticmethod
    def attack(
        minute, att_team, def_team,
        att_players, def_players,
        att_profile, def_profile, state, situation,
        position_engine=None,
        context_x=None, context_y=None,
        delayed_offside=False,
        attacks_right: bool = True,
        shooter_name: str = "",
        assister_name: str = "",
    ) -> ChainResult:
        res = AttackChain.generate(
            minute, att_team, def_team,
            att_players, def_players,
            att_profile, def_profile, state, situation,
            context_x=context_x, context_y=context_y,
            position_engine=position_engine,
            attacks_right=attacks_right,
            shooter_name=shooter_name,
            assister_name=assister_name,
        )
        res.delayed_offside = delayed_offside
        return res

    @staticmethod
    def set_piece(
        minute, att_team, def_team,
        att_players, def_players, state, situation,
        attacks_right: bool = True,
        context_x: Optional[float] = None,
        context_y: Optional[float] = None,
        position_engine=None,
        routine: Optional[SetPieceRoutine] = None,
    ) -> ChainResult:
        return SetPieceChain.generate(
            minute, att_team, def_team,
            att_players, def_players, state, situation,
            attacks_right=attacks_right,
            context_x=context_x,
            context_y=context_y,
            position_engine=position_engine,
            routine=routine,
        )

    @staticmethod
    def transition(
        minute, pressing_team, retreating_team,
        press_players, retreat_players,
        press_profile, state, position_engine=None,
        attacks_right: bool = True,
        counterpress: Optional[Dict[str, Any]] = None,
    ) -> ChainResult:
        return TransitionChain.generate(
            minute, pressing_team, retreating_team,
            press_players, retreat_players,
            press_profile, state, position_engine=position_engine,
            attacks_right=attacks_right,
            counterpress=counterpress,
        )

    @staticmethod
    def defensive_action(
        minute, defending_team, attacking_team,
        def_players, att_players, state, action_type="tackle",
        context_x=None, context_y=None,
        attacks_right: bool = True,
        referee_strictness: float = 0.5,
        danger_level: float = 0.0,
        ball_aerial: bool = False,
        own_goal_x: float = 105.0,
        position_engine: Optional[PositionEngine] = None,
        ball_z: Optional[float] = None,
        defender_facing_x: Optional[float] = None,
        defender_facing_y: Optional[float] = None,
        opponent_distance: Optional[float] = None,
        stamina: Optional[float] = None,
    ) -> ChainResult:
        return DefensiveChain.generate(
            minute, defending_team, attacking_team,
            def_players, att_players, state, action_type,
            context_x=context_x, context_y=context_y,
            attacks_right=attacks_right,
            referee_strictness=referee_strictness,
            danger_level=danger_level,
            ball_aerial=ball_aerial,
            own_goal_x=own_goal_x,
            position_engine=position_engine,
            ball_z=ball_z,
            defender_facing_x=defender_facing_x,
            defender_facing_y=defender_facing_y,
            opponent_distance=opponent_distance,
            stamina=stamina,
        )

    @staticmethod
    def goal_kick(
        minute, kicking_team, defending_team,
        kick_players, def_players,
        team_profile, state, position_engine=None,
    ) -> ChainResult:
        return GoalKickChain.generate(
            minute, kicking_team, defending_team,
            kick_players, def_players,
            team_profile, state, position_engine=position_engine,
        )

    @staticmethod
    def throw_in(
        minute, throwing_team, defending_team,
        throw_players, def_players,
        team_profile, state, x, y,
        position_engine=None,
    ) -> ChainResult:
        return ThrowInChain.generate(
            minute, throwing_team, defending_team,
            throw_players, def_players,
            team_profile, state, x, y,
            position_engine=position_engine,
        )

    @staticmethod
    def discipline(
        minute, fouling_team, fouled_team,
        fouling_players, fouled_players, state,
        referee_strictness=0.5, x=None, y=None,
        attacks_right: bool = True,
        box_penalty_chance: Optional[float] = None,
    ) -> ChainResult:
        return DisciplineChain.generate(
            minute, fouling_team, fouled_team,
            fouling_players, fouled_players, state,
            referee_strictness, x, y,
            attacks_right=attacks_right,
            booked_players=state.booked_players,
            box_penalty_chance=box_penalty_chance,
        )

    @staticmethod
    def substitution(
        minute, team, player_off, player_on, state, reason="tactical",
        position_engine=None,
    ) -> ChainResult:
        return SubstitutionChain.generate(
            minute, team, player_off, player_on, state, reason,
            position_engine=position_engine,
        )

if __name__ == "__main__":
    from player_dna import SquadBuilder
    from match_engine import MatchState, MatchPhase, GameState, TeamProfile, TeamStyle, PlayingStyle, Intensity

    print("\n⛓️  PLOFA 26/27 — Event Chain Module Demo")
    print("="*55)

    # Build mini squads
    hartwell = SquadBuilder.build("Hartwell City", [
        ("Keano Walsh",  "GK",  ["sweeper_keeper"],              29),
        ("Emeka Obi",    "CB",  ["ball_playing_cb"],             27),
        ("Tavish Crane", "CB",  ["stopper_defender", "strong"],  30),
        ("Mateo Sanz",   "CDM", ["anchor_man", "interceptor"],   28),
        ("Kofi Mensah",  "CAM", ["creator", "sup_vision"],       24),
        ("Adri Vela",    "LW",  ["dribbler", "speedster"],       22),
        ("Dragan Novak", "ST",  ["clinical_finisher"],           29),
        ("Yusuf Hamid",  "RW",  ["grand_dribbler", "inverted"],  23),
        ("Luca Ferrini", "CM",  ["box_box"],                     26),
        ("Darius Frost", "LB",  ["aggressive_fullback"],         24),
        ("Rico Alves",   "RB",  ["overlapping_fullback"],        25),
    ], team_superstars=["Dragan Novak"], set_piece_takers=["Kofi Mensah"])

    thornfield = SquadBuilder.build("Thornfield United", [
        ("Pavel Renko",  "GK",  ["sweeper_keeper"],              31),
        ("Bart Kuipers", "CB",  ["stopper_defender"],            28),
        ("Ciro Mancini", "CB",  [],                              26),
        ("Demi Adeola",  "CDM", ["ball_winner", "regista"],      27),
        ("Finn Larsson", "CM",  ["press_resistant"],             25),
        ("Kwame Asante", "CAM", ["playmaker", "creator"],        23),
        ("Bruno Reis",   "LW",  ["speedster", "counter_attacker"], 24),
        ("Nico Strauss", "ST",  ["fox_in_box", "cold_blooded"],  27),
        ("Tariq El-Amin","RW",  ["dribbler"],                    22),
        ("Jide Afolabi", "LB",  [],                              26),
        ("Lee Sung-jin", "RB",  ["overlapping_fullback"],        28),
    ], set_piece_takers=["Kwame Asante"])

    hw_players = hartwell["starters"]
    tf_players = thornfield["starters"]
    for p in hw_players + tf_players:
        p.dna.minutes_played = 45  # Mid-match

    state = MatchState(minute=67, home_goals=1, away_goals=1,
                        momentum=12.0, possession_team="Hartwell City",
                        phase=MatchPhase.PEAK_INTENSITY)
    state.added_time = 0

    hw_profile = TeamProfile("Hartwell City", TeamStyle.ATTACKING,
                              PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    tf_profile = TeamProfile("Thornfield United", TeamStyle.FLUID_COUNTER,
                              PlayingStyle.COUNTER, Intensity.MEDIUM)

    tests = [
        ("⚽  Attack Chain (open play)",
         lambda: ChainDispatcher.attack(
             67, "Hartwell City", "Thornfield United",
             hw_players, tf_players, hw_profile, tf_profile,
             state, SituationType.OPEN_PLAY
         )),
        ("🏃  Transition Chain (press → counter)",
         lambda: ChainDispatcher.transition(
             72, "Thornfield United", "Hartwell City",
             tf_players, hw_players, tf_profile, state
         )),
        ("🚩  Corner Chain",
         lambda: ChainDispatcher.set_piece(
             78, "Hartwell City", "Thornfield United",
             hw_players, tf_players, state, SituationType.CORNER
         )),
        ("📋  Possession Chain (8 passes)",
         lambda: ChainDispatcher.possession(
             45, "Hartwell City", hw_players, hw_profile, state, 8
         )),
        ("🟨  Discipline Chain",
         lambda: ChainDispatcher.discipline(
             81, "Thornfield United", "Hartwell City",
             tf_players, hw_players, state, referee_strictness=0.6
         )),
    ]

    total_goals = 0
    for label, fn in tests:
        print(f"\n{label}")
        r = fn()
        for e in r.events:
            flag = {
                EventType.GOAL: "  ⚽ GOAL",
                EventType.SAVE: "  🧤 SAVE",
                EventType.YELLOW_CARD: "  🟨 YELLOW",
                EventType.RED_CARD: "  🟥 RED",
                EventType.SHOT_ON_TARGET: "  🎯 ON TARGET",
                EventType.SHOT_OFF_TARGET: "  ↗  OFF TARGET",
                EventType.SHOT_BLOCKED: "  🚫 BLOCKED",
                EventType.PRESS_SUCCESS: "  💥 PRESS WON",
                EventType.TURNOVER: "  ❌ TURNOVER",
                EventType.PENALTY_SCORED: "  ⚽ PENALTY SCORED",
                EventType.PENALTY_MISSED: "  ❌ PENALTY MISSED",
            }.get(e.event_type)
            if flag:
                print(f"    {flag}: {e.player} [{e.minute}']")
        print(f"  → {len(r.events)} events | "
              f"Goal: {r.goal_scorer if r.goal_scored else 'No'} | "
              f"xG: {r.xg_generated:.3f} | xA: {r.xa_generated:.3f}")
        if r.goal_scored:
            total_goals += 1

    print(f"\n✅ Event Chain module operational — {total_goals} goal(s) across all chain tests.")
    print("   Next: Stat Accumulator + Exporter\n")

class GoalkeeperEngine:
    """
    Goalkeeper mini-position engine (Checkpoint 7.5 fix).
    
    Three-layer GK intelligence:
        1. STARTING POSITION — How far off the goal line the GK stands.
           Depends on ball proximity (closer = narrower angle = deeper position),
           and GK's own sweeping/positioning tendency.
        2. ANGLE BISECTION — The GK bisects the angle between ball and both posts.
           The closer the ball, the tighter the GKs stance. The wider the angle,
           the more ground the GK must cover.
        3. REACTION TIME — The GK's reflexes and composure determine whether
           they can get to a shot in the unsaved portion of the goal.
    
    The result is that a well-positioned GK with good reflexes saves more than
    xG alone predicts, and a poorly-positioned or slow-reacting GK saves less.
    """

    @staticmethod
    def _get_gk_positioning(gk, ball_x: float, ball_y: float) -> dict:
        """
        Compute the GK's starting position and coverage parameters.
        
        Returns a dict with:
            start_x:  GK's distance from goal line (0 = on the line, 6 = off line)
            start_y:  GK's lateral position (0-68, relative to goal center at 34)
            angle_span:  The angle (in degrees) the GK must cover
            reaction_time: How fast the GK can react (0-1, higher = faster)
            reach: Effective reach radius in meters (height + jumping based)
        """
        import math
        
        # Goal post y-coordinates (inner edges at 30.34 and 37.66)
        POST_LEFT = 30.34
        POST_RIGHT = 37.66
        GOAL_MID = 34.0
        GOAL_LINE_X = 105.0
        
        # Distance from ball to goal
        dist_to_goal = math.sqrt((GOAL_LINE_X - ball_x) ** 2 + (ball_y - GOAL_MID) ** 2)
        
        # ── 1. STARTING POSITION (how far off the line) ────────────
        # GK stands further off line when ball is far (to cut down angle),
        # and drops back when ball is close (to cover the near-post gap).
        # Base: at dist 25m+ (long range), GK stands ~4-5m off line.
        # At dist < 5m (close range), GK on/near line.
        if dist_to_goal < 5.0:
            start_x = 0.5  # Virtually on the line
        elif dist_to_goal < 12.0:
            start_x = 1.5  # Edge of six-yard box
        elif dist_to_goal < 20.0:
            start_x = 3.0  # Middle of six-yard box
        elif dist_to_goal < 30.0:
            start_x = 4.5  # Towards penalty spot
        else:
            start_x = 6.0  # Well off line (sweeper territory)
        
        # Sweeper keeper / aggressive positioning bonus
        if hasattr(gk, 'dna') and hasattr(gk.dna, 'specialties'):
            if 'sweeper_keeper' in gk.dna.specialties:
                start_x += 1.5  # Push another 1.5m out
        start_x = min(8.0, max(0.0, start_x))
        
        # ── 2. LATERAL POSITION (y-axis bisection) ─────────────────
        # GK positions to bisect the angle between ball and both posts.
        # This is the optimal position for a given ball location.
        # Angle to left post
        angle_left = math.atan2(POST_LEFT - ball_y, GOAL_LINE_X - ball_x)
        # Angle to right post
        angle_right = math.atan2(POST_RIGHT - ball_y, GOAL_LINE_X - ball_x)
        # Bisector angle
        bisector_angle = (angle_left + angle_right) / 2.0
        # Project bisector to GK's starting line (x = 105 - start_x)
        gk_x = GOAL_LINE_X - start_x
        # y = ball_y + tan(bisector_angle) * (gk_x - ball_x)
        raw_y = ball_y + math.tan(bisector_angle) * (gk_x - ball_x)
        # Clamp to post width (GK can't be wider than the posts on the line)
        start_y = max(POST_LEFT - 0.5, min(POST_RIGHT + 0.5, raw_y))
        
        # ── 3. ANGLE SPAN ──────────────────────────────────────────
        # Total angle (degrees) the GK must cover from their position.
        # Wider = harder to reach both sides.
        angle_left_from_gk = math.atan2(POST_LEFT - start_y, start_x)
        angle_right_from_gk = math.atan2(POST_RIGHT - start_y, start_x)
        angle_span = abs(math.degrees(angle_right_from_gk - angle_left_from_gk))
        
        # ── 4. REACTION TIME ───────────────────────────────────────
        # Base reaction: 0.7 (average pro GK)
        base_reaction = 0.70
        if hasattr(gk, 'dna'):
            reflexes = gk.dna.gk_attrs.reflexes / 100.0
            composure = gk.dna.mental.composure / 100.0
            # Reflexes drive reaction time more than composure
            reaction_time = base_reaction * (0.4 + reflexes * 0.4 + composure * 0.2)
        else:
            reaction_time = base_reaction
        
        # ── 5. REACH ───────────────────────────────────────────────
        # Effective reach in meters: 2.5m base (arm span + dive distance)
        base_reach = 2.5
        if hasattr(gk, 'dna'):
            height_factor = gk.dna.physical.jumping / 100.0 * 0.8 + 0.4
            reach = base_reach * (0.7 + height_factor * 0.3)
        else:
            reach = base_reach

        if WeatherPhysics.enabled:
            reach *= WeatherPhysics.gk_reach_mult()
            reaction_time *= WeatherPhysics.gk_reaction_mult()

        return {
            'start_x': round(start_x, 1),
            'start_y': round(start_y, 1),
            'angle_span': round(angle_span, 1),
            'reaction_time': round(reaction_time, 2),
            'reach': round(reach, 2),
            'gk_x': round(gk_x, 1),
        }

    @staticmethod
    def _is_shot_savable(gk, shot_x: float, shot_y: float, positioning: dict) -> float:
        """
        Determine if the GK can reach this shot based on positioning.
        Returns the save probability multiplier [0.0, 1.5].
        
        The key insight: a GK who has bisected the angle correctly + the
        shot is within reach range = higher save chance than xG alone.
        A shot to the opposite corner that the GK has over-committed = 
        significantly lower save chance.
        """
        import math
        GOAL_MID = 34.0
        POST_LEFT = 30.34
        POST_RIGHT = 37.66
        GOAL_LINE_X = 105.0
        
        gk_y = positioning['start_y']
        gk_x = GOAL_LINE_X - positioning['start_x']
        reach = positioning['reach']
        reaction_time = positioning['reaction_time']
        
        # Determine if shot is to the GK's left or right
        shot_side = 'left' if shot_y < gk_y else 'right'
        goal_side = 'left' if shot_y < GOAL_MID else 'right'
        
        # Distance from GK starting position to shot location at goal line
        dy = abs(shot_y - gk_y)
        dx = GOAL_LINE_X - gk_x  # GK is this far from goal line
        dist_to_shot = math.sqrt(dy ** 2 + dx ** 2)
        
        # ── CAN THE GK REACH IT? ────────────────────────────────────
        # Effective reach: GK has reaction_time * reach meters of dive range
        # in the direction of the shot. A shot within that range gets full
        # attention; beyond it, the GK is stretching.
        effective_reach = reach * (0.8 + reaction_time * 0.4)
        
        if dist_to_shot <= effective_reach:
            # Within reach: GK has good chance, positioning matters
            # Better positioned (closer to shot line) = higher save
            reach_factor = 1.0  # Full reach capability
        else:
            # Beyond comfortable reach: GK must stretch
            overshoot = dist_to_shot - effective_reach
            # Exponential falloff: every 0.5m beyond reach is 20% harder
            reach_factor = max(0.2, 1.0 - overshoot * 0.4)
        
        # ── ANGLE SPAN FACTOR ──────────────────────────────────────
        # Wider angle span = more ground to cover = lower save chance
        # At 20° (distant shot), easy. At 60°+ (close range), hard.
        angle_span = positioning['angle_span']
        if angle_span < 25:
            angle_factor = 1.20  # Narrow angle, GK well-positioned
        elif angle_span < 35:
            angle_factor = 1.05
        elif angle_span < 50:
            angle_factor = 0.90
        elif angle_span < 65:
            angle_factor = 0.75
        else:
            angle_factor = 0.55  # Very wide angle, GK exposed
        
        # ── REACTION TIME FACTOR ───────────────────────────────────
        # Shot to the same side GK is positioned = easier
        # Shot across the body = harder
        if shot_side == goal_side:
            # Same side: GK is already leaning that way
            reaction_factor = 0.8 + reaction_time * 0.3
        else:
            # Across body: GK must change direction
            reaction_factor = 0.5 + reaction_time * 0.3
        
        # ── SHOT PLACEMENT QUALITY ─────────────────────────────────
        # Shots closer to the post = harder to save, whatever the xG
        post_distance = min(abs(shot_y - POST_LEFT), abs(shot_y - POST_RIGHT))
        if post_distance < 0.5:
            placement_factor = 0.6  # Top corner / post - extremely hard
        elif post_distance < 1.5:
            placement_factor = 0.75  # Side netting
        elif post_distance < 3.0:
            placement_factor = 0.9  # Decent placement
        else:
            placement_factor = 1.1  # Central - GK should save
        
        # ── COMBINED SAVE MULTIPLIER ───────────────────────────────
        # Higher = harder to score (easier to save)
        save_mult = reach_factor * angle_factor * reaction_factor * placement_factor
        return round(save_mult, 3)

    @staticmethod
    def decide_high_ball(gk, contact_z: float, contested: bool,
                         challenger_present: bool = False) -> Tuple[str, str]:
        """DECISION (audited, no flat roll): a goalkeeper winning a high ball
        (corner, crossed free kick, or open-play cross) must commit to one of
        two real handling choices — CLAIM it cleanly or PUNCH it clear.

        Inputs are all physical, real things that already happened in the
        simulation, not post-hoc stat heuristics:
            contact_z       — the 3D height the ball was won at (the delivery
                              was either within a two-handed secure range or
                              not)
            contested       — whether the aerial duel had a genuine challenger
                              physically arriving (traffic means a clean two-
                              handed catch is not available)
            challenger_present — the challenger exists at all (physics gate)

        The punch tendency genuinely rises with delivery height (above a
        keeper's secure two-handed reach the only safe way to clear a cross is
        a fist) and with contest traffic, and genuinely falls with the GK's own
        handling, aerial command and composure from his DNA.

        Returns (action, height) with action in {"claim", "punch"} and height
        in {"high", "medium", "low"} (the actual height at which the ball was
        taken).
        """
        # ── Claim height bands from the REAL contact height ──────────
        # A "high" claim is a full-extension take at or above the crossbar
        # zone (~2.4m+ for a 6'3" keeper); "medium" is a normal head/chest
        # take (~1.9-2.4m); "low" is gathered at waist or floor height.
        if contact_z >= 2.4:
            height = "high"
        elif contact_z >= 1.9:
            height = "medium"
        else:
            height = "low"

        # ── Chosen action: claim vs punch (multi-cause decision) ─────
        punch_tendency = 0.0
        if contested and challenger_present:
            punch_tendency += 0.40    # an attacker is in the arc — can't wrap cleanly
        elif challenger_present:
            punch_tendency += 0.20
        if contact_z >= 2.6:
            punch_tendency += 0.30    # delivery above secure two-handed reach
        elif contact_z >= 2.3:
            punch_tendency += 0.15

        if gk is not None and getattr(gk, "dna", None) is not None:
            gk_dna = gk.dna
            handling = float(getattr(getattr(gk_dna, "gk_attrs", None), "handling", 60.0)) / 100.0
            aerial   = float(getattr(getattr(gk_dna, "gk_attrs", None), "aerial_gk", 60.0)) / 100.0
            composure = float(getattr(getattr(gk_dna, "mental", None), "composure", 60.0)) / 100.0
            # Secure handlers / dominant aerial keepers / calm keepers punch less.
            punch_tendency -= handling * 0.22
            punch_tendency -= aerial   * 0.20
            punch_tendency -= composure * 0.08

        punch_tendency = max(0.10, min(0.80, punch_tendency))
        action = "punch" if random.random() < punch_tendency else "claim"
        return action, height

    @staticmethod
    def evaluate_save(xg: float, shooter_quality: float, shot_x: float, shot_y: float, gk, last_ball_x: float, last_ball_y: float):
        """
        Advanced GK engine replacing flat xG evaluation.
        
        Returns (is_goal, positioning) where is_goal is True if the shot
        beats the keeper, and positioning is the dict from _get_gk_positioning.
        
        Flow:
            1. Compute GK's starting position (angle bisection + depth)
            2. Determine if the shot is savable from that position
            3. Adjust effective xG by save probability
            4. Roll against the adjusted probability
        """
        if not gk:
            effective_prob = min(0.99, xg * shooter_quality)
            is_goal = random.random() < effective_prob
            return is_goal, {"start_x": None}
        
        positioning = GoalkeeperEngine._get_gk_positioning(gk, last_ball_x, last_ball_y)
        
        # Compute save multiplier
        save_mult = GoalkeeperEngine._is_shot_savable(gk, shot_x, shot_y, positioning)
        
        # ── CONTINUOUS PHYSICS GATE (opt-in) ───────────────────────────
        # The positioning model above is a good one — it bisects the angle,
        # sets depth by distance to goal, and derives reaction time from
        # reflexes and composure. What it cannot express is the BALL'S FLIGHT
        # TIME. `effective_reach = reach * (0.8 + reaction_time * 0.4)` gives a
        # keeper identical reach against a shot from 30 m as against one from
        # 8 m, even though the first arrives in ~1.2 s and the second in
        # ~0.35 s. A keeper beaten by a driven shot from 25 yards has usually
        # not been beaten on reach — he has been beaten on time.
        #
        # So this attenuates save_mult when the physics say he cannot get a
        # hand to it in the time the ball actually takes. It never adds credit.
        # Because the caller already floors save_mult at 1.0, pushing it toward
        # 1.0 converges on the raw xG — the engine's own documented safe
        # fallback — so this cannot inflate scorelines the way the inverted
        # multiplier described below once did.
        #
        # Inert unless MatchConfig.physics_enabled is True; the adapter slot is
        # None otherwise, so this is one dict lookup on a path that already
        # computes a save multiplier.
        try:
            from physics.adapter import active_adapter
            _phys = active_adapter()
        except Exception:
            _phys = None
        if _phys is not None:
            try:
                save_mult, _why = _phys.keeper_save_multiplier(
                    save_mult, shot_x, shot_y, last_ball_x, last_ball_y,
                    positioning)
            except Exception:
                # Physics is an enhancement. A keeper gate that cannot be
                # evaluated leaves the engine's own multiplier exactly as it
                # was — the shot is not re-decided, it just isn't adjusted.
                pass
        
        # Base probability: this is the chance the ball goes in (goal happens)
        base_prob = min(0.99, xg * shooter_quality)
        
        # The save_mult MODIFIES the xG: higher save_mult = lower goal probability.
        # save_mult of 1.0 = xG unchanged (neutral positioning)
        # save_mult of 1.5 = xG reduced by 33% (GK well-positioned)
        #
        # FIX (scoreline realism): the previous `base_prob / save_mult` was
        # mathematically inverted — it DIVIDED the xG by the save multiplier,
        # so a save_mult below 1.0 (a "badly positioned" GK) AMPLIFIED the
        # conversion rate instead of leaving it at the raw xG. Because the
        # four factors (reach × angle × reaction × placement) routinely
        # multiply to ~0.9-1.0, nearly every shot got its xG boosted, which
        # is what inflated scorelines to 8-8 / 10-4. A goalkeeper should only
        # ever REDUCE conversion below the raw xG, never raise it. Clamping
        # save_mult to a floor of 1.0 guarantees that: a well-positioned GK
        # (save_mult > 1) lowers the conversion, a neutral or badly-positioned
        # GK (save_mult <= 1) leaves it at the raw xG.
        adjusted_xg = base_prob / max(1.0, save_mult)
        
        # Clamp
        adjusted_xg = min(0.98, max(0.005, adjusted_xg))
        
        # Roll for goal
        is_goal = random.random() < adjusted_xg
        return is_goal, positioning


class GoalPhysicsEngine:
    """
    Handles physical trajectory calculations for shots.
    
    Determines whether a shot from a given (x,y) coordinate:
    - Is on frame (between the posts at y=30.34 and y=37.66 at x=105)
    - Hits the woodwork (within ~0.25m of a post or the crossbar)
    - Goes wide or high
    
    Also computes rebound angles for blocked/woodwork shots.
    """
    
    GOAL_LEFT = 30.34
    GOAL_RIGHT = 37.66
    GOAL_CENTER = 34.0
    GOAL_LINE_X = 105.0
    POST_RADIUS = 0.25  # How close to post is "woodwork"
    CROSSBAR_Y = 37.66  # Actually z-coordinate, approximated in y-plane
    GOAL_HEIGHT_M = 2.44  # Approximated as extra range above posts
    
    @staticmethod
    def calculate_intersection(start_x, start_y, target_y):
        """
        Simple 2D intersection check: does a line from (start_x, start_y)
        to (GOAL_LINE_X, target_y) pass between the posts?
        
        Returns (on_target: bool, exact_y: float)
        """
        if GoalPhysicsEngine.GOAL_LEFT <= target_y <= GoalPhysicsEngine.GOAL_RIGHT: #why is it GoalPhysicsEngine and not GoalkeeperEngine? Because this is about the shot, not the GK
            return True, target_y
        return False, target_y
    
    @staticmethod
    def is_on_target(shot_x: float, shot_y: float, shooter_position: str = "ST") -> bool:
        """
        Geometry-aware on-target probability.
        
        A shot from close range and central is more likely on target.
        A shot from wide angles or long range is less likely on target.
        
        Returns probability [0, 1] that the shot is on frame.
        """
        import math
        dx = max(1.0, GoalPhysicsEngine.GOAL_LINE_X - shot_x) #why is GoalPhysicsEngine and not GoalkeeperEngine? Because this is about the shot, not the GK
        dy = abs(shot_y - GoalPhysicsEngine.GOAL_CENTER)
        
        # Distance from goal
        dist = math.sqrt(dx ** 2 + dy ** 2)
        
        # Base: 50% on-target for average shot
        base = 0.50
        
        # Distance factor: closer = more on target
        dist_factor = max(0.3, 1.0 - (dist / 60.0) * 0.6)
        base *= dist_factor
        
        # Angle factor: wider angles produce more off-target shots
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 60:
            angle_factor = 0.6
        elif angle > 45:
            angle_factor = 0.75
        elif angle > 30:
            angle_factor = 0.85
        else:
            angle_factor = 1.0
        base *= angle_factor
        
        return min(0.92, max(0.08, base))
    
    @staticmethod
    def get_shot_outcome(shot_x: float, shot_y: float, on_target_roll: float) -> str:
        """
        Determine the outcome of a shot based on position and random roll.
        
        Returns: "goal" | "woodwork" | "save" | "wide" | "blocked"
        """
        import math
        dx = max(1.0, GoalkeeperEngine.GOAL_LINE_X - shot_x)
        dy = abs(shot_y - GoalkeeperEngine.GOAL_CENTER)
        dist = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.degrees(math.atan2(dy, dx))
        
        # Shots from > 40m are almost never on target
        on_target_prob = GoalkeeperEngine.is_on_target(shot_x, shot_y)
        
        if on_target_roll < on_target_prob:
            # On frame - between the posts
            # Check for woodwork (posts + crossbar)
            post_proximity = min(
                abs(shot_y - GoalkeeperEngine.GOAL_LEFT),
                abs(shot_y - GoalkeeperEngine.GOAL_RIGHT)
            )
            if post_proximity < GoalkeeperEngine.POST_RADIUS:
                return "woodwork"
            # Check crossbar (approximated by height factor)
            if dist < 15.0 and random.random() < 0.05:
                return "woodwork"
            return "save"  # Will be resolved by GK engine
        else:
            # Off frame
            # Near miss (woodwork adjacent)
            post_proximity = min(
                abs(shot_y - GoalkeeperEngine.GOAL_LEFT),
                abs(shot_y - GoalkeeperEngine.GOAL_RIGHT)
            )
            if post_proximity < 0.5 and random.random() < 0.3:
                return "woodwork"
            return "wide"
# ─────────────────────────────────────────────
# GOAL KICK CHAIN — Realistic Restart Mechanics
# ─────────────────────────────────────────────

class GoalKickChain(BaseChain):
    """
    Models realistic goal kick restarts.
    Short build-up vs Long Launch based on team style and footedness.
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        kicking_team: str,
        defending_team: str,
        kick_players: List[PlayerProfile],
        def_players: List[PlayerProfile],
        team_profile: "TeamProfile",
        state: MatchState,
        position_engine: Optional[PositionEngine] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        episode = None

        gk = next((p for p in kick_players if p.position == "GK"), None)
        gk_name = gk.name if gk else "GK"
        gk_foot = getattr(gk.dna, "preferred_foot", "right") if gk else "right"

        # Determine strategy: Short build-up or Long launch?
        from match_engine import TeamStyle
        style = getattr(team_profile, "style", TeamStyle.BALANCED)
        
        # Tiki-taka, possession, and vertical tiki-taka prefer short build-up
        short_prob = 0.85 if style in (TeamStyle.TIKI_TAKA, TeamStyle.STRUCTURED_POSSESSION, TeamStyle.VERTICAL_TIKI_TAKA) else (
            0.15 if style in (TeamStyle.ROUTE_ONE, TeamStyle.PARK_THE_BUS, TeamStyle.ULTRA_DEFENSIVE) else 0.50
        )

        is_short = random.random() < short_prob

        if is_short:
            # ── SHORT BUILD-UP (Play out from the back) ─────────────
            # Pick a deep defender (CB, LB, RB) standing near the box
            receiver = cls.pick_weighted(
                kick_players,
                lambda p: 3.5 if p.position in ("CB", "LB", "RB", "CDM") else 0.01,
                exclude=gk_name
            )
            if not receiver:
                receiver = kick_players[0]

            end_x = random.uniform(12.0, 22.0)
            end_y = random.uniform(12.0, 56.0)

            # GK short pass
            result.add(cls.make_event(
                minute, EventType.GOAL_KICK, kicking_team, gk_name,
                phase, gs,
                secondary_player=receiver.name,
                location_x=6.0, location_y=34.0,
                end_x=end_x, end_y=end_y,
                outcome=True,
                metadata={"short_build_up": True}
            ))

            result.add(cls.make_event(
                minute, EventType.BALL_RECEIPT, kicking_team, receiver.name,
                phase, gs,
                location_x=end_x, location_y=end_y,
                outcome=True,
            ))

            if position_engine:
                position_engine.record_touch(gk_name, 6.0, 34.0, minute)
                position_engine.record_touch(receiver.name, end_x, end_y, minute)

        else:
            # ── LONG LAUNCH (Goal kick into opponent/midfield half) ──
            # Target zone: x = 55 to 72m
            end_x = random.uniform(55.0, 72.0)

            # Footedness direction bias:
            # Left footed GK launches toward Right/Center (y = 30 to 58)
            # Right footed GK launches toward Left/Center (y = 10 to 38)
            if gk_foot == "left":
                end_y = random.uniform(30.0, 58.0)
            else:
                end_y = random.uniform(10.0, 38.0)

            result.add(cls.make_event(
                minute, EventType.GOAL_KICK, kicking_team, gk_name,
                phase, gs,
                location_x=6.0, location_y=34.0,
                end_x=end_x, end_y=end_y,
                outcome=True,
                metadata={"long_launch": True, "gk_foot": gk_foot}
            ))

            # Contested aerial duel at target zone
            target_att = cls.pick_weighted(
                kick_players,
                lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2 if p.position != "GK" else 0.1
            )
            target_def = cls.pick_weighted(
                def_players,
                lambda p: (p.dna.physical.jumping + p.dna.defending.clearances) / 2 if p.position != "GK" else 0.1
            )

            att_win = False
            aerial_physics = None
            if target_att and target_def:
                if position_engine is not None:
                    gk_aerial_height = 1.8 + random.uniform(0.3, 1.2)
                    gk_aerial_speed = aerial_delivery_speed(
                        float(getattr(gk.dna.passing, "long_passing", 55.0)) if gk else 55.0
                    )
                    flight = make_ballistic_flight(
                        Vec3(6.0, 34.0, 0.05),
                        Vec3(end_x, end_y, gk_aerial_height),
                        gk_aerial_speed,
                        loft=0.0,
                    )
                    aerial_attackers = [
                        cls._moving_player(target_att, position_engine)
                    ]
                    aerial_defenders = [
                        cls._moving_player(target_def, position_engine)
                    ]
                    episode = PossessionEpisode()
                    episode.register(aerial_attackers)
                    episode.register(aerial_defenders)
                    aerial_resolution = episode.resolve_aerial(flight, aerial_attackers, aerial_defenders)
                    att_win = (
                        aerial_resolution.outcome in ("controlled", "contested")
                        and aerial_resolution.winner in aerial_attackers
                    )
                    aerial_physics = episode.physics_meta("goal_kick_aerial")
                    
                    if aerial_resolution.winner is not None:
                        winner_name = getattr(aerial_resolution.winner.player, "name", "")
                        if winner_name:
                            episode.update_player_position(
                                winner_name,
                                aerial_resolution.contact_point.x,
                                aerial_resolution.contact_point.y,
                            )
                            if position_engine is not None:
                                position_engine.record_touch(winner_name, aerial_resolution.contact_point.x, aerial_resolution.contact_point.y, minute)
                else:
                    att_win = random.random() < 0.50

                result.add(cls.make_event(
                    minute, EventType.AERIAL_DUEL, kicking_team, target_att.name,
                    phase, gs,
                    secondary_player=target_def.name,
                    location_x=end_x, location_y=end_y,
                    outcome=att_win,
                    metadata={"from_goal_kick": True, "resolution": "aerial_trajectory" if position_engine is not None else "legacy_roll", "physics": aerial_physics}
                ))
                if not att_win:
                    result.possession_lost = True

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)
        return result


# ─────────────────────────────────────────────
# THROW-IN CHAIN — Standard & Brentford Long Throws
# ─────────────────────────────────────────────

class ThrowInChain(BaseChain):
    """
    Handles throw-in restarts.
    Wingbacks take throw-ins.
    In attacking third (x >= 80), long-throw specialist teams throw directly into box!
    """

    @classmethod
    def generate(
        cls,
        minute: int,
        throwing_team: str,
        defending_team: str,
        throw_players: List[PlayerProfile],
        def_players: List[PlayerProfile],
        team_profile: "TeamProfile",
        state: MatchState,
        x: float,
        y: float,
        position_engine: Optional[PositionEngine] = None,
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state
        episode = None

        # Wingbacks/Fullbacks always take throw-ins
        taker = cls.pick_weighted(
            throw_players,
            lambda p: 4.0 if p.position in ("LB", "RB", "LWB", "RWB") else 0.5
        ) or throw_players[0]

        from match_engine import TeamStyle
        style = getattr(team_profile, "style", TeamStyle.BALANCED)
        is_long_throw_team = style in (TeamStyle.ROUTE_ONE, TeamStyle.WING_PLAY, TeamStyle.ATTACKING)

        # Brentford / Stoke style long throw into box if x >= 80m
        if x >= 80.0 and is_long_throw_team and random.random() < 0.65:
            # ── LONG THROW-IN INTO THE BOX ──────────────────────────
            end_x = random.uniform(88.0, 98.0)
            end_y = random.uniform(22.0, 46.0)

            result.add(cls.make_event(
                minute, EventType.THROW_IN, throwing_team, taker.name,
                phase, gs,
                location_x=x, location_y=y,
                end_x=end_x, end_y=end_y,
                outcome=True,
                metadata={"long_throw_to_box": True}
            ))

            # Pick aerial threat in box
            receiver = cls.pick_weighted(
                cls._outfield_players(throw_players),
                lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2 if p.name != taker.name else 0.1
            )
            defender = cls.pick_weighted(
                def_players,
                lambda p: (p.dna.physical.jumping + p.dna.defending.clearances) / 2
            )

            if receiver and defender:
                att_win = False
                aerial_physics = None
                if position_engine is not None:
                    throw_aerial_height = 1.5 + random.uniform(0.3, 1.0)
                    throw_aerial_speed = aerial_delivery_speed(
                        float(getattr(taker.dna.passing, "long_passing", 55.0))
                    )
                    flight = make_ballistic_flight(
                        Vec3(x, y, 0.05),
                        Vec3(end_x, end_y, throw_aerial_height),
                        throw_aerial_speed,
                        loft=0.0,
                    )
                    aerial_attackers = [
                        cls._moving_player(receiver, position_engine)
                    ]
                    aerial_defenders = [
                        cls._moving_player(defender, position_engine)
                    ]
                    episode = PossessionEpisode()
                    episode.register(aerial_attackers)
                    episode.register(aerial_defenders)
                    aerial_resolution = episode.resolve_aerial(flight, aerial_attackers, aerial_defenders)
                    att_win = (
                        aerial_resolution.outcome in ("controlled", "contested")
                        and aerial_resolution.winner in aerial_attackers
                    )
                    aerial_physics = episode.physics_meta("throw_in_aerial")
                    
                    if aerial_resolution.winner is not None:
                        winner_name = getattr(aerial_resolution.winner.player, "name", "")
                        if winner_name:
                            episode.update_player_position(
                                winner_name,
                                aerial_resolution.contact_point.x,
                                aerial_resolution.contact_point.y,
                            )
                            if position_engine is not None:
                                position_engine.record_touch(winner_name, aerial_resolution.contact_point.x, aerial_resolution.contact_point.y, minute)
                else:
                    att_win = random.random() < 0.48

                result.add(cls.make_event(
                    minute, EventType.AERIAL_DUEL, throwing_team, receiver.name,
                    phase, gs,
                    secondary_player=defender.name,
                    location_x=end_x, location_y=end_y,
                    outcome=att_win,
                    metadata={"long_throw_box_scramble": True, "resolution": "aerial_trajectory" if position_engine is not None else "legacy_roll", "physics": aerial_physics}
                ))

                if att_win:
                    # Flick-on header chance or shot
                    result.add(cls.make_event(
                        minute, EventType.BALL_RECOVERY, throwing_team, receiver.name,
                        phase, gs,
                        location_x=end_x, location_y=end_y,
                        outcome=True,
                        metadata={"loose_ball": True}
                    ))
                else:
                    # Cleared by defender
                    result.add(cls.make_event(
                        minute, EventType.CLEARANCE, defending_team, defender.name,
                        phase, gs,
                        location_x=end_x, location_y=end_y,
                        end_x=random.uniform(50, 70), end_y=random.uniform(10, 58),
                        outcome=True
                    ))
                    result.possession_lost = True

        else:
            # ── STANDARD SHORT THROW-IN ─────────────────────────────
            _outfield = cls._outfield_players(throw_players)
            receiver = cls.pick_weighted(
                _outfield,
                lambda p: 3.0 if p.position in ("CM", "CAM", "LW", "RW", "ST") else 1.0,
                exclude=taker.name
            ) or (_outfield[0] if _outfield else None)

            end_x = max(2.0, min(103.0, x + random.uniform(-4, 6)))
            end_y = max(4.0, min(64.0, y + (5.0 if y < 34 else -5.0)))

            result.add(cls.make_event(
                minute, EventType.THROW_IN, throwing_team, taker.name,
                phase, gs,
                secondary_player=receiver.name if receiver else None,
                location_x=x, location_y=y,
                end_x=end_x, end_y=end_y,
                outcome=True
            ))

            if receiver:
                result.add(cls.make_event(
                    minute, EventType.BALL_RECEIPT, throwing_team, receiver.name,
                    phase, gs,
                    location_x=end_x, location_y=end_y,
                    outcome=True
                ))
            else:
                result.possession_lost = True

            if position_engine:
                position_engine.record_touch(taker.name, x, y, minute)
                if receiver:
                    position_engine.record_touch(receiver.name, end_x, end_y, minute)

        result.player_distance_stats = cls._accumulate_physics_stats(episode, position_engine, minute)
        return result
