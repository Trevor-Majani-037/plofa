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
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

from match_engine import (
    MatchEvent, EventType, SituationType, MatchPhase,
    GameState, XGEngine, MatchState
)
from player_dna import PlayerDNA, PlayerProfile, DNAFactory, BehavioralTendencies
from position_engine import PositionEngine, ELLIPSE_COMPOSE_FLOOR
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
from possession_phases import (
    PossessionPhase,
    TacticalDirective,
    PossessionPhaseEngine,
    GKSnapshot,
    TeammateSnapshot,
    PossessionDecision,
    possession_phase_for,
)
from cross_detector import detect_cross
from long_pass_detector import detect_long_pass
from pass_classifier import classify_pass
from pressing_profiles import (
    PressingProfile,
    PROFILES as PRESS_PROFILES,
    resolve_profile,
    engagement_allows,
    cover_shadow_clearance,
    cover_shadow_blocked,
    in_cover_shadow,
    COVER_SHADOW_BLOCK_THRESHOLD,
)
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

# Checkpoint 25 — pass-execution reaction radius: a defender only cuts out a
# ball genuinely within reach (0.8m), tighter than the 1.2m planning radius
# the decision layer uses to AVOID corridors.
LANE_REACTION_DIST = 0.8

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
    
    # Delayed offside flag (set by ChainDispatcher for VAR-style checks)
    delayed_offside: bool    = False

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
    shoot_x: float           = 0.0
    shoot_y: float           = 0.0
    shoot_under_pressure: bool = False
    shot_taken: bool         = False

    def add(self, event: MatchEvent):
        self.events.append(event)
        return self


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

    @staticmethod
    def penalty_spot_x(attacks_right: bool) -> float:
        return 94.0 if attacks_right else 11.0

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

        lo, hi = (1, 6) if is_micro else (10, 40) if is_counter else (2, 16)
        dist = max(lo, min(hi, base))

        adv = 0.40 + (carry_skill / 200)
        adv = max(0.20, min(0.90, adv))

        return dist, adv


# ─────────────────────────────────────────────
# 1. POSSESSION CHAIN
# Build-up play: passes, carries, progressive actions
# ─────────────────────────────────────────────

class PossessionChain(BaseChain):
    """
    Models a possession sequence from winning the ball
    to either losing it or transitioning to an attack.

    Sequence structure:
        build_up phase  → short passes in own half
        progression     → carries/long passes into midfield
        final_third     → key passes, through balls, crosses
    """

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
        if not last_player:
            return result

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

        for step in range(sequence_length):
            if result.possession_lost:
                break

            is_final_step = (step == sequence_length - 1)
            action_roll = random.random()

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
                if _m_anchor is not None:
                    end_cy = y + _m_bias + (0.5 - random.random()) * 3
                else:
                    end_cy = y + (0.5 - random.random()) * 6
                end_cy = max(2, min(66, end_cy))

                # Stepping into an occupied defender ends the run sometimes.
                micro_lost = False
                if position_engine is not None and def_players:
                    nearest_at_end = min(
                        (math.hypot(position_engine.get_position(d.name)[0] - end_cx,
                                    position_engine.get_position(d.name)[1] - end_cy)
                         for d in def_players if getattr(d, 'position', None) != 'GK'),
                        default=99.0,
                    )
                    if nearest_at_end < 2.5 and random.random() < 0.30:
                        micro_lost = True

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
                if engagement_allows(nx, press_profile):
                    press_zone_prob = press_cfg.zone_probs.get(
                        "box" if x > 83 else
                        "att_third" if x > 70 else
                        "mid_third" if x > 35 else "own_third",
                        0.20
                    )
                    # Scale by live press intensity (from the defending team)
                    press_prob = min(0.75, press_zone_prob * (0.6 + 0.6 * press_intensity))
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

            if regression_mode is not None and phase_decision is not None and phase_decision.target is not None:
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
                        take_prob = max(0.0, min(1.0, (shot_decision.shot_score - 0.70) / 0.30))
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
                            result.shoot_x = x
                            result.shoot_y = y
                            result.shoot_under_pressure = under_pressure
                            result.shot_taken = True
                            break
                target_player = next(
                    (p for p in players if p.name == phase_decision.target), None
                )
                if target_player is not None:
                    tx, ty = position_engine.get_position(target_player.name)
                    forced_receiver = target_player
                    forced_end = cls._pass_destination_to_target(tx, ty, attacks_right)
                    matrix_meta = {
                        "possession_phase": phase_decision.phase.value,
                        "phase_directive": phase_decision.directive.value,
                        "phase_reason": phase_decision.reason,
                        "recycle": regression_mode,
                    }
            elif position_engine is not None:
                matrix_decision = AttackingMatrix.decide(
                    last_player,
                    [p for p in players if p.name != last_player.name],
                    def_players, x, y,
                    position_engine=position_engine,
                    attacks_right=attacks_right,
                    team_profile=team_profile,
                    under_pressure=under_pressure,
                    danger_level=min(100.0, max(0.0, (70.0 - abs(x - (0.0 if attacks_right else 105.0))) / 70.0 * 100.0)),
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
                    if matrix_decision.action == "SHOOT":
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
                        take_prob = max(0.0, min(1.0, (matrix_decision.shot_score - 0.70) / 0.30))
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
                            result.shoot_x = x
                            result.shoot_y = y
                            result.shoot_under_pressure = under_pressure
                            result.shot_taken = True
                            break
                    if matrix_decision is not None and matrix_decision.is_pass and matrix_decision.target is not None:
                        forced_receiver = matrix_decision.target
                        forced_end = cls._pass_destination_to_target(
                            matrix_decision.target_x, matrix_decision.target_y,
                            attacks_right,
                        )

            # ── CHECKPOINT 24: WIDE COMBINATION OVERRIDE ────────────
            # The matrix's option values are progress/depth-biased, so for a
            # WIDE carrier it kept choosing box-seekers and far runners —
            # geometrically stamped as crosses (20-25/match) and long balls.
            # A real winger's default with the ball on the flank is the
            # short game. Intercept here (shoot decisions have already
            # broken out above; through balls fire later and still can).
            if (forced_receiver is None or
                    (matrix_decision is not None and matrix_decision.is_pass) or
                    (phase_decision is not None
                     and getattr(phase_decision.directive, "value", "") == "progress")):
                if (position_engine is not None
                        and last_player.position in ("LW", "RW", "LB", "RB")
                        and regression_mode is None):
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

            carry_prob = cls._carry_probability(last_player, x, team_profile)

            # Longer carry (progression attempt, not micro-carry)
            can_carry = x < 88 if attacks_right else x > 17
            # A winger who commits to a drive/cut carries the ball instead of
            # settling for a safe pass — the instinct gates carry-vs-pass in
            # the final third (Checkpoint 18).
            if winger_drive_mode is not None:
                carry_prob = max(carry_prob, 0.80)
            if (action_roll < carry_prob and can_carry and not under_pressure
                    and regression_mode is None and not gk_distribution):
                carry_dist, adv_ratio = cls._carry_distance_advance(
                    last_player, x, team_profile
                )
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
                else:
                    vert_range = 4 + (last_player.dna.technical.ball_control / 100) * 8
                    new_y = y + (0.5 - random.random()) * vert_range
                new_y = max(2, min(66, new_y))
                is_prog = (new_x - x) > 9.14 if attacks_right else (x - new_x) > 9.14

                if hasattr(last_player, "dna"):
                    dribble_success_rate = DNAFactory.get_dribble_success_rate(last_player.dna)
                    # Checkpoint 24 — carries were near-invincible
                    # (0.78 + 0.15·skill ≈ 93-98%), so every winger drive
                    # reached the byline and fed the cross trigger. Real
                    # carriers lose the ball constantly against a set block:
                    # Doku completes barely half his take-ons. Success now
                    # scales with the nearest defender's distance and gets
                    # harder the deeper into the block the carry goes.
                    carry_prob = 0.58 + dribble_success_rate * 0.12
                    if position_engine is not None and def_players:
                        nearest_def = min(
                            (math.hypot(position_engine.get_position(d.name)[0] - x,
                                        position_engine.get_position(d.name)[1] - y)
                             for d in def_players if getattr(d, 'position', None) != 'GK'),
                            default=99.0,
                        )
                        if nearest_def < 2.5:
                            carry_prob -= 0.22
                        elif nearest_def < 5.0:
                            carry_prob -= 0.10
                    if (x > 72) if attacks_right else (x < 33):
                        carry_prob -= 0.08  # final third: no free rides through a set block
                    carry_prob = max(0.30, min(0.90, carry_prob))
                    carry_prob = _get_soul_applicator().modify_dribble_success(last_player, carry_prob)
                    carry_success = random.random() < carry_prob
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
                    metadata={"progressive": is_prog, "distance": round(carry_dist, 1)}
                ))

                if carry_success:
                    x, y = new_x, new_y
                    if position_engine is not None:
                        position_engine.record_touch(last_player.name, x, y, minute)
                else:
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
                        receiver = cls._pick_receiver(
                            players, last_player, x, team_profile,
                            position_engine=position_engine, y=y,
                            def_players=def_players, attacks_right=attacks_right,
                            possession_phase=current_phase if phase_decision else None,
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
                        and regression_mode is None and not wide_combo_mode):
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

                if regression_mode is not None:
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
                elif wide_combo_mode:
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
                    is_switch = random.random() < last_player.dna.tendencies.switches_play
                    # Checkpoint 24 — wingers get switched TO, they don't
                    # orchestrate. Their far-side hail-mary is a rarity.
                    if last_player.position in ("LW", "RW"):
                        is_switch = is_switch and random.random() < 0.35

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
                    }
                if gk_distribution:
                    phase_pass_meta["gk_distribution"] = True

                marking = cls._marking_tightness(
                    receiver, x, y, def_players, position_engine, attacks_right
                )

                # ── CHECKPOINT 20: SPACE BATTLE ────────────────────────
                # When the receiver is closely marked, they must "battle" their
                # marker to get to space. This models real life: beating a
                # defender to receive a pass requires pace, vision, strength,
                # and stamina. If the space battle fails, the pass either goes
                # incomplete or the receiver can't get free.
                space_battle_failed = False
                if marking > 0.5 and receiver is not None and def_players:
                    try:
                        # Find the closest defender to the receiver
                        closest_defender = None
                        min_dist = float('inf')
                        for d in (def_players or []):
                            if getattr(d, 'position', None) == 'GK':
                                continue
                            dx, dy = position_engine.get_position(d.name) if position_engine else (0, 0)
                            dist = math.hypot(dx - x, dy - y)
                            if dist < min_dist:
                                min_dist = dist
                                closest_defender = d
                        if closest_defender is not None:
                            battle_prob = DNAFactory.get_space_battle_success(
                                receiver.dna, closest_defender.dna
                            )
                            space_battle_failed = random.random() > battle_prob
                    except Exception:
                        space_battle_failed = False

                if space_battle_failed:
                    # Receiver can't beat their marker — pass fails
                    result.add(cls.make_event(
                        minute, EventType.TURNOVER, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=False,
                        metadata={"space_battle_lost": True, "receiver": receiver.name}
                    ))
                    result.possession_lost = True
                    break

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
                    )

                # Checkpoint 25 — evaluate the corridor BEFORE rolling success.
                # Lane geometry was previously only a *selection* input; a
                # pass played through a defender's cover shadow completed at
                # full rate, so line-breakers were free. Now the corridor's
                # proximity model cuts completion (a ball through a defender's
                # feet is cut out ~70%, one 3m clear is untouched), and a
                # fully body-blocked failure resolves as a real INTERCEPTION
                # by the nearest defender. Long balls and switches travel
                # OVER the lines — exempt by design.
                lane_mult = 1.0
                interceptor = None
                pass_len = math.hypot(end_px - x, end_py - y)
                # Only medium/long ground balls travel far enough to BE a
                # line-breaking risk. Short combinations into a marked man
                # already fail via the marking penalty + space battle —
                # taxing them again here would double-count the marker.
                if (position_engine is not None and def_players
                        and not long_intent and pass_len >= 10.0):
                    lane_cl = lane_clearance(
                        x, y, end_px, end_py, def_players, position_engine,
                        block_dist=LANE_REACTION_DIST)
                    if lane_cl < 1.0:
                        # A ball through a defender's feet still squeaks
                        # through more often than not — reaction speed is
                        # not omniscience — so the floor is 0.70, not 0.30.
                        lane_mult = 0.75 + 0.25 * lane_cl
                        if lane_cl <= 0.0:
                            interceptor = cls._lane_interceptor(
                                x, y, end_px, end_py, def_players,
                                position_engine)

                success = cls._pass_success(last_player, long_intent, under_pressure,
                                            receiver=receiver, marking=marking,
                                            confidence=last_player.dna.form.confidence,
                                            lane_mult=lane_mult)

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
                    is_airborne=_cr.airborne,
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
                    is_airborne=_cr.airborne,
                    is_headed=(body_part == "head"),
                    under_pressure=under_pressure,
                    attacks_right=attacks_right,
                )

                pass_energy = cls._pass_energy_cost(
                    last_player, x, y, end_px, end_py,
                    is_long=is_long, under_pressure=under_pressure,
                    marking=marking,
                )

                result.add(cls.make_event(
                    minute, etype, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=receiver.name,
                    location_x=x, location_y=y,
                    end_x=end_px, end_y=end_py,
                    outcome=success,
                    metadata={
                        "is_long": is_long,
                        "pass_length_m": round(_lp.distance_m, 1),
                        "pass_length_yards": round(_lp.distance_yards, 1),
                        "pass_height": _lp.height,
                        "is_progressive": is_prog,
                        "under_pressure": under_pressure,
                        "pass_advance": pass_advance,
                        "body_part": body_part,
                        "cross": _cr.is_cross,
                        "is_airborne": _cr.airborne,
                        "cross_origin": _cr.origin_zone,
                        "cross_dest": _cr.destination_zone,
                        "pass_type": _pc.pass_type,
                        "length_class": _pc.length_class,
                        "pass_direction": _pc.direction,
                        "start_half": _pc.start_half,
                        "end_half": _pc.end_half,
                        "start_third": _pc.start_third,
                        "end_third": _pc.end_third,
                        "pass_channel": _pc.channel,
                        "pass_energy_cost": round(pass_energy, 3),
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
                    offside_loc = cls._check_offside(
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

                    # ── BALL RECEIPT ─────────────────────────────
                    # StatsBomb logs a BALL_RECEIPT event for every
                    # completed pass. This is where ~900 extra events come from.
                    # Receiver may miscontrol (~8% of receipts)
                    miscontrol = random.random() < 0.08
                    if miscontrol:
                        result.add(cls.make_event(
                            minute, EventType.MISCONTROL, attacking_team, receiver.name,
                            phase, game_state,
                            location_x=end_px, location_y=end_py,
                            outcome=False,
                            metadata={"from_pass": True}
                        ))
                        result.possession_lost = True
                        break
                    else:
                        result.add(cls.make_event(
                            minute, EventType.BALL_RECEIPT, attacking_team, receiver.name,
                            phase, game_state,
                            location_x=end_px, location_y=end_py,
                            outcome=True,
                        ))

                    # Update position and ball carrier
                    x, y = end_px, end_py
                    last_player = receiver
                    if position_engine is not None:
                        position_engine.record_touch(receiver.name, x, y, minute)

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

                att_wins = random.random() < (
                    DNAFactory.get_aerial_success_rate(last_player.dna)
                    if duel_type == EventType.AERIAL_DUEL
                    else DNAFactory.get_tackle_success_rate(last_player.dna) * 0.55 + 0.45
                )

                result.add(cls.make_event(
                    minute, duel_type, attacking_team, last_player.name,
                    phase, game_state,
                    secondary_player=pressure_player.name,
                    location_x=x, location_y=y,
                    outcome=att_wins,
                    metadata={"duel_type": duel_type.name}
                ))

                if not att_wins:
                    result.possession_lost = True
                    break

            # ── 5. THROUGH BALL (final step, vision players) ───────
            through_zone = x > 50 if attacks_right else x < 55
            if (is_final_step and through_zone and not result.possession_lost
                    and random.random() < last_player.dna.tendencies.plays_through_ball):
                receiver = cls._pick_receiver(
                    players, last_player, x, team_profile,
                    preferred_positions=["ST", "CF", "LW", "RW", "CAM"],
                    position_engine=position_engine, y=y,
                )
                if receiver:
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
                        metadata={"distance": round(tb_dist, 1), "tb_prob": round(tb_prob, 3),
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
                        last_player = receiver
                        x, y = end_tx, end_ty
                        if position_engine is not None:
                            position_engine.record_touch(receiver.name, x, y, minute)

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
                    drb_prob = DNAFactory.get_dribble_success_rate(last_player.dna)
                    
                    # Defender quality affects success rate
                    if marker:
                        def_tackle_skill = marker.dna.defending.tackling / 100.0
                        drb_prob -= (def_tackle_skill * 0.15)  # Good defenders reduce success
                    
                    drb_prob = _get_soul_applicator().modify_dribble_success(last_player, drb_prob)
                    drb_prob = max(0.15, min(0.85, drb_prob))  # Clamp to realistic range
                    
                    drb_success = random.random() < drb_prob
                    
                    # HEAVY TOUCH CHECK: Even if technically successful, did attacker push ball too far?
                    ball_control_quality = last_player.dna.technical.ball_control / 100.0
                    heavy_touch_risk = 0.12 * (1.0 - ball_control_quality)  # Poor control = higher risk
                    is_heavy_touch = random.random() < heavy_touch_risk
                    
                    if drb_success and not is_heavy_touch:
                        # TRUE SUCCESS: Beat defender, maintained control
                        drb_adv = random.uniform(3, 8)
                        end_drb_x = cls.clamp_x(x + (drb_adv if attacks_right else -drb_adv), attacks_right)
                        if last_player.position in ("LW", "RW"):
                            # Checkpoint 24 — same byline cap as carries.
                            end_drb_x = min(end_drb_x, 97.0) if attacks_right else max(end_drb_x, 8.0)
                        # Checkpoint 18 — a beating winger dribble continues the
                        # drive/cut shape instead of a random lateral wander.
                        _d_mode, _d_anchor, _d_bias = cls._winger_carry_steering(
                            last_player, x, y, attacks_right, under_pressure,
                            def_players, position_engine,
                        )
                        if _d_mode == "byline":
                            end_drb_y = y + (_d_anchor - y) * 0.35
                            end_drb_y += (0.5 - random.random()) * 2
                        elif _d_mode == "cut_inside":
                            _cut = _d_anchor + (1.0 if _d_anchor < 34.0 else -1.0) * 10.0
                            end_drb_y = y + (_cut - y) * 0.30
                            end_drb_y += (0.5 - random.random()) * 2
                        elif _d_anchor is not None:
                            end_drb_y = y + _d_bias + (0.5 - random.random()) * 2
                        else:
                            end_drb_y = y + (0.5 - random.random()) * 4
                        end_drb_y = max(5, min(63, end_drb_y))
                        
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
                                "beat_defender": marker.name if marker else "unknown"
                            },
                        ))
                        x, y = end_drb_x, end_drb_y
                        if position_engine is not None:
                            position_engine.record_touch(last_player.name, x, y, minute)
                    else:
                        # FAILURE: Either tackled by defender OR heavy touch
                        fail_reason = "heavy_touch" if is_heavy_touch else "tackled"
                        drb_fail_adv = random.uniform(1, 3) if is_heavy_touch else 0
                        end_drb_x = cls.clamp_x(x + (drb_fail_adv if attacks_right else -drb_fail_adv), attacks_right)
                        end_drb_y = y + (0.5 - random.random()) * 2
                        end_drb_y = max(5, min(63, end_drb_y))
                        
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
            if (not result.possession_lost and cross_zone
                    and last_player.position in ("LW", "RW", "LB", "RB")
                    and random.random() < cross_prob):
                cross_skill = last_player.dna.technical.crossing / 100.0
                zone, end_tx, end_ty, raw_tx, raw_ty = cls._generate_cross_destination(
                    x, y, attacks_right, cross_skill
                )
                cross_success = random.random() < (0.30 + cross_skill * 0.35)
                # Checkpoint 11 — even the engine's OWN cross decision is
                # validated by the pure geometric detector (origin wide +
                # destination in/flashing through the box). A "cross" whipped
                # from a central position is stamped `cross: false` exactly
                # as Opta/StatsBomb would refuse to tag it.
                _cr = detect_cross(x, y, end_tx, end_ty, attacks_right,
                                   event_type="CROSS_ATTEMPT")
                result.add(cls.make_event(
                    minute, EventType.CROSS_ATTEMPT, attacking_team, last_player.name,
                    phase, game_state,
                    location_x=x, location_y=y,
                    end_x=end_tx, end_y=end_ty,
                    outcome=cross_success,
                    metadata={
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
                    }
                ))
                if cross_success:
                    result.add(cls.make_event(
                        minute, EventType.CROSS_SUCCESS, attacking_team, last_player.name,
                        phase, game_state,
                        location_x=x, location_y=y,
                        outcome=True,
                        metadata={"open_play": True}
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
        engine = PossessionPhaseEngine(
            gk_snap, style_key=style_key, carrier_iq=carrier_iq
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
    ) -> Optional[PlayerProfile]:
        # Closer to goal = higher chance of forward player receiving
        fwd_weight = min(5.0, 1.0 + (x / 105) * 4.0)
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
            w *= 1.0 / (1.0 + d / 9.0)
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
                       lane_mult: float = 1.0) -> bool:
        prob = DNAFactory.get_pass_accuracy(player.dna, is_long=is_long, under_pressure=under_pressure)
        # Chemistry modifier: if both players have chemistry data, multiply
        # pass accuracy by the chemistry multiplier (0.90x to 1.14x).
        if chemistry is not None and receiver is not None:
            chem_mult = chemistry.pass_chemistry_mult(player.name, receiver.name)
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
    ) -> Optional[Tuple[float, float]]:
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

        Returns (offside_x, offside_y) if offside, else None.
        The offside location is the receiver's position at the moment of the pass.
        """
        if position_engine is None:
            return None

        # Law 11 is judged at the moment the ball is played, on the
        # receiver's LIVE position — not on where the pass destination
        # ends up. A striker starting onside and running onto a through
        # ball is onside, even when the ball lands beyond the line.
        rx, ry = position_engine.get_position(receiver.name)

        # Condition 1: receiver must be in opponent's half
        opp_half = rx > 52.5 if attacks_right else rx < 52.5
        if not opp_half:
            return None

        # Condition 3: receiver must be ahead of the ball
        ball_ahead = (rx > x) if attacks_right else (rx < x)
        if not ball_ahead:
            return None

        second_last_x = cls._second_last_defender_x(
            def_players, position_engine, attacks_right
        )
        if second_last_x is None:
            return None

        # Condition 2: receiver must be ahead of second-last defender
        receiver_ahead = (rx > second_last_x) if attacks_right else (rx < second_last_x)
        if not receiver_ahead:
            return None

        # ── CALIBRATED OFF-SIDE DECISION (see OFFSIDE_* constants) ──
        # Margin in metres by which the receiver is beyond the offside line.
        margin = (rx - second_last_x) if attacks_right else (second_last_x - rx)

        # Law 11: being LEVEL with the second-last defender is NOT offside. A
        # small "benefit of the doubt" tolerance keeps a receiver level with
        # (or a centimetre past) the line onside, exactly as referees call it.
        if margin <= OFFSIDE_LEVEL_TOL_M:
            return None

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
            return None   # the run was timed onside / the flag stayed down

        return (rx, ry)

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
        if is_progressive:
            return max(5, min(25, base * 1.5))
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
        base_prob = vision_roll * 0.5 + composure_roll * 0.2 - safe_penalty
        if hasattr(profile, 'style'):
            fast_styles = {"fluid_counter", "gegenpressing", "ultra_attacking", "route_one", "direct"}
            if profile.style.value in fast_styles:
                base_prob += 0.12
        # Confidence modulates forward-pass boldness: a player low on
        # confidence turns the ball back / sideways rather than attempting
        # the risky forward pass; a confident one commits to it.
        conf = getattr(getattr(player, "dna", None), "form", None)
        if conf is not None:
            conf_mult = 0.55 + (conf.confidence / 100.0) * 0.90   # 0.55 .. 1.45
            base_prob *= conf_mult
        base_prob = max(0.08, min(0.60, base_prob))
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

        vert_skill = (player.dna.passing.short_passing + player.dna.technical.ball_control) / 200
        vert_range = 4 + vert_skill * 16
        end_py = y + (0.5 - random.random()) * vert_range
        end_py = max(2, min(66, end_py))

        # Preserve flank width for wide fullbacks on non-forward reset passes.
        # This avoids a safe fullback recycle pushing the ball unnaturally toward
        # the central channel when the pass is meant to be a sideways/backward option.
        is_forward = (dx > 0) if attacks_right else (dx < 0)
        if player.position in ("LB", "RB") and not is_forward:
            if y < 22.0:
                end_py = max(end_py, y - 3.0)
                end_py = min(end_py, 24.0)
            elif y > 46.0:
                end_py = min(end_py, y + 3.0)
                end_py = max(end_py, 44.0)

        # ── CHECKPOINT 18: WINGER FLANK PRESERVATION ──────────────
        # MODERN WINGER FIX: wingers are touchline-hugging flank attackers,
        # NOT drifting #10s. The middle of the pitch is always full — a #10
        # owns that space — and a winger who drifts inside leaves his flank
        # open. When a winger receives a pass, the destination must stay on
        # their flank channel, not drift toward midfield. This is one of the
        # key reasons wingers were ending up next to the CAM — every pass
        # pulled them central.
        if player.position in ("LW", "RW"):
            flank_keep = 0.55 if not is_forward else 0.40
            if player.position == "LW":
                # Left winger: destination stays in the left flank channel
                if y < 26.0:
                    # Already wide — keep it wide
                    end_py = min(end_py, 22.0)
                elif end_py > 22.0:
                    # Drifted central — push the destination back to the flank
                    end_py = y - (y - 22.0) * flank_keep
                    end_py = max(6.0, min(22.0, end_py))
            else:
                # Right winger: destination stays in the right flank channel
                if y > 42.0:
                    # Already wide — keep it wide
                    end_py = max(end_py, 46.0)
                elif end_py < 46.0:
                    # Drifted central — push the destination back to the flank
                    end_py = y + (46.0 - y) * flank_keep
                    end_py = max(46.0, min(62.0, end_py))

        return end_px, end_py

    @classmethod
    def _pass_destination_to_receiver(
        cls, receiver: PlayerProfile,
        x: float, y: float, pass_dist: float,
        position_engine: Optional[PositionEngine],
        attacks_right: bool,
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
        # Lead the receiver: put the ball slightly in front of him so the
        # pass is a delivery, not a teleport. Longer balls lead further.
        # "In front" means in the direction of attack.
        lead = 2.0 if capped >= 22.0 else (0.8 if capped >= 10.0 else 0.0)
        lead_dir = 1.0 if attacks_right else -1.0
        end_px = x + (dx / d) * capped + lead * lead_dir
        end_py = y + (dy / d) * capped
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
        vert_range = 4 + (player.dna.technical.ball_control / 100) * 8
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
    def _generate_pass_event(
        cls, minute: int, team: str, passer: PlayerProfile,
        receiver: PlayerProfile, x: float, y: float,
        is_long: bool, is_prog: bool, is_switch: bool,
        success: bool, phase: MatchPhase, game_state: GameState
    ) -> MatchEvent:
        end_x = min(105, x + cls._pass_distance(is_long, is_prog) * (1.0 if success else 0.3))
        end_y = random.uniform(5, 63)

        etype = (
            EventType.PROGRESSIVE_PASS if is_prog and success
            else EventType.SWITCH_OF_PLAY if is_switch and success
            else EventType.PASS
        )

        # Checkpoint 11 — geometric cross qualifier on every generated pass.
        _cr = detect_cross(x, y, end_x, end_y, True,
                           event_type=etype.name)

        # Checkpoint 13 — full Opta pass classification stamp.
        signed_dx = end_x - x
        body_part = "right_foot" if passer.dna.preferred_foot == "right" else "left_foot"
        _pc = classify_pass(x, y, end_x, end_y,
                            signed_dx=signed_dx,
                            is_cross=_cr.is_cross,
                            is_airborne=_cr.airborne,
                            is_headed=(body_part == "head"))

        return cls.make_event(
            minute, etype, team, passer.name,
            phase, game_state,
            secondary_player=receiver.name if success else None,
            location_x=x, location_y=y,
            end_x=end_x, end_y=end_y,
            outcome=success,
            metadata={"is_long": is_long, "is_progressive": is_prog,
                      "body_part": body_part,
                      "cross": _cr.is_cross,
                      "is_airborne": _cr.airborne,
                      "cross_origin": _cr.origin_zone,
                      "cross_dest": _cr.destination_zone,
                      "pass_type": _pc.pass_type,
                      "length_class": _pc.length_class,
                      "pass_direction": _pc.direction,
                      "start_half": _pc.start_half,
                      "end_half": _pc.end_half,
                      "start_third": _pc.start_third,
                      "end_third": _pc.end_third,
                      "pass_channel": _pc.channel}
        )

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
    def _select_action_from_position(x: float, y: float, player_position: str) -> str:
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

        Returns:
            Action string: "shoot", "pass", "cross", or "dribble"
        """
        from attacking_matrix import shooting_angle_degrees

        # Reject shooting behind goal line (x ≥ 105)
        if x >= 105.0:
            # Return "pass" if central, "cross" if wide
            if 20 < y < 48:
                return "pass"
            else:
                return "cross"

        # Angle to goal centre line (0° = central, 90° = level with goal line)
        angle_degrees = shooting_angle_degrees(x, y, attacks_right=True)

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
    ) -> ChainResult:
        result = ChainResult()
        phase  = state.phase
        gs     = state.game_state

        default_anchor = 88.0 if attacks_right else 17.0
        anchor_x = context_x if context_x is not None else default_anchor
        anchor_y = context_y if context_y is not None else 34.0
        shooter = cls._pick_shooter(att_players, position_engine, anchor_x, anchor_y)
        creator = cls._pick_creator(
            att_players, exclude=shooter.name if shooter else None,
            position_engine=position_engine, x=anchor_x, y=anchor_y,
        )
        gk      = cls._pick_gk(def_players)

        if not shooter:
            return result

        # Shot location: use spatial anchor if provided (counter-attack continuity)
        in_attacking_half = (context_x > 60) if attacks_right else (context_x is not None and context_x < 45)
        if context_x is not None and in_attacking_half:
            x_adv = random.uniform(0, 12)
            x = cls.clamp_x(context_x + (x_adv if attacks_right else -x_adv), attacks_right)
            y = (context_y or 34) + random.uniform(-10, 10)
            y = max(5, min(63, y))
        else:
            x, y = cls._shot_location(situation, team_profile, attacks_right=attacks_right)
        zone = PitchZone.xg_zone(x, y, attacks_right=attacks_right)

        if position_engine is not None:
            position_engine.record_touch(shooter.name, x, y, minute)
            if creator:
                position_engine.record_touch(creator.name, x - 8, y, minute)

        # Body part (Checkpoint 6: now angle/channel-aware)
        body_part = cls._body_part(shooter, situation, y)

        # Is it a big chance?
        is_big = cls._is_big_chance(situation, zone, team_profile)

        # Under pressure?
        under_pressure = cls._is_under_pressure(x, def_profile, state)

        # ── CHANCE CREATION EVENT ─────────────────────────────
        # ── CHANCE CREATION EVENT ─────────────────────────────
        creation_event = None
        if creator and situation != SituationType.PENALTY:
            creation_type = cls._creation_type(creator, situation, team_profile)

            creation_event = cls.make_event(
                minute,
                EventType.BIG_CHANCE_CREATED if is_big else EventType.CHANCE_CREATED,
                attacking_team, creator.name,
                phase, gs,
                secondary_player=shooter.name,
                location_x=x - random.uniform(5, 20),
                location_y=y,
                end_x=x, end_y=y,
                situation=situation,
                xa=0.0,   # backfilled below once the final shot xG is known
                outcome=True,
                metadata={"creation_type": creation_type, "is_big_chance": is_big}
            )
            result.add(creation_event)

        # ── DRIBBLE BEFORE SHOT? ──────────────────────────────
        # STRICT OPTA/STATSBOMB DEFINITION: Dribble only counts if beating
        # an ENGAGED defender. Pre-shot scenarios inherently have defensive
        # pressure, so this is valid — but still needs defender reference.
        if (situation == SituationType.OPEN_PLAY
                and random.random() < shooter.dna.tendencies.attempts_dribble * 0.4):
            
            # Pick nearest defender for the dribble duel
            nearest_defender = None
            if def_players:
                # In final third, nearest defender is likely CB or pressing midfielder
                nearest_defender = cls.pick_weighted(
                    def_players,
                    lambda p: {
                        "CB": 3.5, "CDM": 2.5, "CM": 2.0, "LB": 1.5, "RB": 1.5,
                    }.get(p.position, 0.5),
                )
            
            drb_prob = DNAFactory.get_dribble_success_rate(shooter.dna)
            
            # Defender quality affects success
            if nearest_defender:
                def_tackle_skill = nearest_defender.dna.defending.tackling / 100.0
                drb_prob -= (def_tackle_skill * 0.18)  # Final third defending is crucial
            
            drb_prob = _get_soul_applicator().modify_dribble_success(shooter, drb_prob)
            drb_prob = max(0.15, min(0.80, drb_prob))
            
            success = random.random() < drb_prob
            
            # Heavy touch check — even more critical in tight box scenarios
            ball_control_quality = shooter.dna.technical.ball_control / 100.0
            heavy_touch_risk = 0.15 * (1.0 - ball_control_quality)
            is_heavy_touch = random.random() < heavy_touch_risk
            
            drb_start_x = x - 5
            
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
                    },
                ))
                # Successful dribble → better shot position
                drb_shot_adv = random.uniform(2, 6)
                x = cls.clamp_x(x + (drb_shot_adv if attacks_right else -drb_shot_adv), attacks_right)
                zone = PitchZone.xg_zone(x, y, attacks_right=attacks_right)
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
                    end_x=x, end_y=y,
                    outcome=False,
                    metadata={
                        "failure_reason": fail_reason,
                        "pre_shot_dribble": True,
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

        # ── SHOT OUTCOME ──────────────────────────────────────
        shot_on_target_prob = cls._shot_on_target_prob(xg, shooter, situation)

        if random.random() < shot_on_target_prob:
            # ── WOODWORK (Checkpoint 6) ────────────────────────────
            # Previously absent entirely — no woodwork category existed,
            # so a shot heading for the frame either went in or was saved.
            # ~7% of shots that would have been on-target instead cannon
            # off the post/bar. Most result in corners (60%) as the ball
            # deflects off at an angle; others stay in play as loose balls
            # or occasionally loop back in off a post/keeper/defender.
            if random.random() < 0.07:
                rebound_in = random.random() < 0.15
                goes_for_corner = not rebound_in and random.random() < 0.60
                
                result.add(cls.make_event(
                    minute, EventType.HIT_WOODWORK, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=gk.name if gk else None,
                    location_x=x, location_y=y,
                    situation=situation, xg=xg, body_part=body_part,
                    outcome=rebound_in,
                    metadata={
                        "zone": zone,
                        "is_big_chance": is_big,
                        "rebound_in": rebound_in,
                        "corner_awarded": goes_for_corner,
                    }
                ))
                if rebound_in:
                    result.goal_scored    = True
                    result.goal_team      = attacking_team
                    result.goal_scorer    = shooter.name
                    result.goal_assistant = creator.name if creator else ""
                    result.add(cls.make_event(
                        minute, EventType.GOAL, attacking_team, shooter.name,
                        phase, gs,
                        secondary_player=creator.name if creator else None,
                        location_x=x, location_y=y,
                        situation=situation, xg=xg, body_part=body_part,
                        outcome=True,
                        metadata={"zone": zone, "is_big_chance": is_big,
                                  "body_part": body_part, "via_woodwork": True}
                    ))
                elif goes_for_corner:
                    # Woodwork deflection sends ball out for corner
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
                return result

            result.shot_on_target = True

            # Emit shot on target
            result.add(cls.make_event(
                minute, EventType.SHOT_ON_TARGET, attacking_team, shooter.name,
                phase, gs,
                secondary_player=gk.name if gk else None,
                location_x=x, location_y=y,
                situation=situation,
                xg=xg,
                body_part=body_part,
                outcome=True,
                metadata={"zone": zone, "is_big_chance": is_big}
            ))

            # ── GOAL? ─────────────────────────────────────────
            shooter_quality = DNAFactory.get_shooter_quality(shooter.dna)
            is_goal, positioning = GoalkeeperEngine.evaluate_save(
                xg, shooter_quality, x, y, gk, state.last_ball_x, state.last_ball_y
            )
            if is_goal:
                result.goal_scored   = True
                result.goal_team     = attacking_team
                result.goal_scorer   = shooter.name
                result.goal_assistant = creator.name if creator else ""

                result.add(cls.make_event(
                    minute, EventType.GOAL, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=creator.name if creator else None,
                    location_x=x, location_y=y,
                    situation=situation,
                    xg=xg,
                    body_part=body_part,
                    outcome=True,
                    metadata={
                        "zone": zone,
                        "is_big_chance": is_big,
                        "body_part": body_part,
                    }
                ))
            else:
                # Save - now with realistic deflection physics
                shot_difficulty = xg * (1.0 + (0.4 if under_pressure else 0.0))
                angle_from_center = abs(y - 34.0)
                is_wide_shot = angle_from_center > 15.0
                is_close_range = x > 95.0 if attacks_right else x < 10.0

                parry_chance = min(0.80, 0.35 + shot_difficulty * 0.45 + (angle_from_center / 34.0) * 0.30)
                if is_close_range:
                    parry_chance *= 1.4
                if body_part == "head":
                    parry_chance *= 1.25

                is_parry = random.random() < parry_chance
                save_type = "parry" if is_parry else "catch"

                is_goalline_save = (
                    positioning.get("start_x") is not None
                    and positioning["start_x"] <= 2.0
                )
                
                # Parried saves can result in corners
                corner_from_parry = False
                if is_parry:
                    # Deflection vector calculation
                    # Shots from wide angles deflected around posts → corner
                    # Formula: wider shot + closer range = higher corner chance
                    corner_prob = 0.55  # Base 55% of parries go for corners (increased from 40%)
                    if is_wide_shot:
                        corner_prob += 0.25  # Wide shots more likely deflected out
                    if is_close_range:
                        corner_prob += 0.15  # Close range = less reaction time
                    if shot_difficulty > 0.5:
                        corner_prob += 0.15  # Difficult shots harder to control (increased from 0.10)
                    
                    corner_from_parry = random.random() < min(0.90, corner_prob)
                
                result.add(cls.make_event(
                    minute, EventType.SAVE, defending_team,
                    gk.name if gk else "GK",
                    phase, gs,
                    secondary_player=shooter.name,
                    location_x=x, location_y=y,
                    xg=xg,
                    outcome=True,
                    metadata={
                        "zone": zone,
                        "is_big_chance": is_big,
                        "save_type": save_type,
                        "parried": is_parry,
                        "deflection_angle": angle_from_center if is_parry else 0,
                        "goalline_save": is_goalline_save,
                    }
                ))
                
                if corner_from_parry:
                    result.corner_won = True
                    result.corner_team = attacking_team
                    # Determine which side the corner comes from based on shot position
                    corner_y = 0.0 if y < 34.0 else 68.0
                    result.add(cls.make_event(
                        minute, EventType.CORNER_WON, attacking_team, shooter.name,
                        phase, gs, 
                        location_x=105 if attacks_right else 0,
                        location_y=corner_y,
                        metadata={"from_save_deflection": True, "gk_parry": True}
                    ))

        else:
            # Shot off target or blocked?
            if random.random() < 0.38:
                blocker = cls._pick_blocker(def_players)
                result.add(cls.make_event(
                    minute, EventType.SHOT_BLOCKED, attacking_team, shooter.name,
                    phase, gs,
                    secondary_player=blocker.name if blocker else None,
                    location_x=x, location_y=y,
                    xg=xg,
                    outcome=False,
                ))
                corner_awarded = random.random() < 0.55
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
            else:
                result.add(cls.make_event(
                    minute, EventType.SHOT_OFF_TARGET, attacking_team, shooter.name,
                    phase, gs,
                    location_x=x, location_y=y,
                    xg=xg,
                    outcome=False,
                ))

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

        return result

    # ── HELPERS ───────────────────────────────────────────────

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
    def _pick_creator(
        cls, players: List[PlayerProfile], exclude: str = None,
        position_engine: Optional[PositionEngine] = None,
        x: float = 75.0, y: float = 34.0,
    ) -> Optional[PlayerProfile]:
        return cls.pick_weighted_spatial(
            players,
            lambda p: {
                "CAM": 5.5, "CM": 3.5, "LW": 3.0, "RW": 3.0,
                "CDM": 2.0, "LB": 2.0, "RB": 2.0,
                "CB": 0.5, "ST": 0.8, "GK": 0.1,
            }.get(p.position, 1.0),
            position_engine, x, y,
            exclude=exclude,
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
        if situation == SituationType.FAST_BREAK:
            lo, hi = m(88, 103)
            return PitchZone.random_in((lo, hi), (22, 46))
        # Open play
        lo, hi = m(78, 103)
        x = random.uniform(lo, hi)
        dist_from_line = (105.0 - x) if attacks_right else x
        max_spread = max(4.0, dist_from_line * 1.2)
        y = random.uniform(34.0 - max_spread, 34.0 + max_spread)
        y = max(0.0, min(68.0, y))
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
    ) -> ChainResult:
        if situation == SituationType.PENALTY:
            return cls._penalty_chain(minute, attacking_team, defending_team,
                                       att_players, def_players, state, attacks_right)
        elif situation == SituationType.CORNER:
            return cls._corner_chain(minute, attacking_team, defending_team,
                                      att_players, def_players, state, attacks_right)
        else:
            return cls._freekick_chain(minute, attacking_team, defending_team,
                                        att_players, def_players, state, situation, attacks_right,
                                        context_x=context_x, context_y=context_y)

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
                metadata={"pen_prob": round(pen_prob, 3)}
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
                metadata={"saved_by": gk.name if gk else "GK"}
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

    @classmethod
    def _corner_chain(cls, minute, att_team, def_team,
                       att_players, def_players, state,
                       attacks_right=True) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        corner_x = 105.0 if attacks_right else 0.0
        corner_y  = random.choice([1.0, 67.0])
        corner_side = "right" if corner_y > 34 else "left"
        taker    = cls._pick_sp_taker(att_players, situation="corner", corner_side=corner_side)
        receiver = cls._pick_aerial_threat(att_players, exclude=taker.name)
        defender = cls._pick_aerial_defender(def_players)
        gk       = cls._pick_gk_player(def_players)

        # Corner taken event
        result.add(cls.make_event(
            minute, EventType.CORNER_TAKEN, att_team, taker.name,
            phase, gs,
            location_x=corner_x, location_y=corner_y,
            situation=SituationType.CORNER,
        ))

        # Real football: ~65% of corners are cleared without incident (defense clears,
        # ball goes out for goal kick, or simple clearance). Only ~35% result in
        # any meaningful action (delivery received, aerial duel, shot attempt).
        # This matches real corner outcome distribution.
        corner_outcome_roll = random.random()

        # ~65%: Corner cleared without incident - ball goes out or is cleared
        # Geometry-based determination (no random weights):
        # Uses corner side, GK position, and defender positioning to decide outcome
        if corner_outcome_roll < 0.65:
            # Geometry-based clearance type determination:
            # - Corner side (left/right) influences clearance direction
            # - GK position relative to corner area affects claiming ability
            # - Defender proximity determines who wins the aerial duel

            # Step 1: Check if GK is positioned to claim the corner
            gk = cls._pick_gk_player(def_players)
            gk_claim_possible = False
            if gk:
                # GK can claim if within ~25m of corner area and centrally positioned
                gk_to_corner_dist = math.hypot(
                    gk.position_x - corner_x, gk.position_y - corner_y
                ) if hasattr(gk, 'position_x') and hasattr(gk, 'position_y') else 0
                gk_claim_possible = gk_to_corner_dist < 25.0 and abs(gk.position_y - 34.0) < 15.0 if hasattr(gk, 'position_y') else False

            # Step 2: Determine clearance type using geometry (not random weights)
            if gk_claim_possible and gk:
                # GK is positioned to claim → goalkeeper catches/claims the corner
                result.add(cls.make_event(
                    minute, EventType.SAVE, def_team, gk.name,
                    phase, gs, outcome=True,
                    location_x=random.uniform(99, 104.3),
                    location_y=random.uniform(24, 44),
                    metadata={"type": "corner_claim_geometry", "corner_followup": "gk_claim", "corner_side": corner_side},
                ))
            elif corner_side == "right":
                # Corner from right side → left-side defensive clearance
                # (defenders on right side clear to their left)
                defender = cls._pick_aerial_defender(def_players)
                result.add(cls.make_event(
                    minute, EventType.CLEARANCE, def_team,
                    defender.name if defender else "Unknown",
                    phase, gs,
                    location_x=random.uniform(85, 95),
                    location_y=random.uniform(18, 35),  # toward left side
                    outcome=True,
                    metadata={"corner_followup": "defensive_clearance", "corner_side": corner_side, "clearance_reason": "geometry_right_corner"},
                ))
            elif corner_side == "left":
                # Corner from left side → right-side defensive clearance
                # (defenders on left side clear to their right)
                defender = cls._pick_aerial_defender(def_players)
                result.add(cls.make_event(
                    minute, EventType.CLEARANCE, def_team,
                    defender.name if defender else "Unknown",
                    phase, gs,
                    location_x=random.uniform(85, 95),
                    location_y=random.uniform(35, 50),  # toward right side
                    outcome=True,
                    metadata={"corner_followup": "defensive_clearance", "corner_side": corner_side, "clearance_reason": "geometry_left_corner"},
                ))
            else:
                # Central corner or no clear side → defensive clearance to touch
                defender = cls._pick_aerial_defender(def_players)
                result.add(cls.make_event(
                    minute, EventType.CLEARANCE, def_team,
                    defender.name if defender else "Unknown",
                    phase, gs,
                    location_x=random.uniform(85, 105),
                    location_y=random.uniform(18, 50),
                    outcome=True,
                    metadata={"corner_followup": "defensive_clearance", "corner_side": corner_side, "clearance_reason": "geometry_central"},
                ))

            return result

        # ~35%: Corner results in some meaningful action (delivery, aerial duel, etc.)
        # Cross quality
        cross_quality = taker.dna.technical.crossing / 100.0
        delivery_success = random.random() < (0.55 + cross_quality * 0.30)

        if delivery_success and receiver:
            # Aerial duel in the box
            att_aerial = DNAFactory.get_aerial_success_rate(receiver.dna)
            def_aerial = DNAFactory.get_aerial_success_rate(defender.dna) if defender else 0.45
            att_wins   = random.random() < att_aerial / (att_aerial + def_aerial)

            result.add(cls.make_event(
                minute, EventType.AERIAL_DUEL, att_team, receiver.name,
                phase, gs,
                secondary_player=defender.name if defender else None,
                location_x=random.uniform(88, 102),
                location_y=random.uniform(24, 44),
                outcome=att_wins,
            ))

            if att_wins:
                # Shot from header
                xg = XGEngine.calculate(
                    zone="inside_box", body_part="head",
                    situation=SituationType.CORNER,
                    is_big_chance=random.random() < 0.35
                )
                result.xg_generated = xg
                result.xa_generated = xg  # xA = xG at moment of creation (correct standard)

                sot_prob = 0.30 + xg * 0.4
                if random.random() < sot_prob:
                    result.shot_on_target = True
                    shot_header_x = random.uniform(88, 100)
                    shot_header_y = random.uniform(27, 41)
                    result.add(cls.make_event(
                        minute, EventType.SHOT_ON_TARGET, att_team, receiver.name,
                        phase, gs, xg=xg, body_part="head",
                        situation=SituationType.CORNER, outcome=True,
                        secondary_player=gk.name if gk else None,
                        location_x=shot_header_x,
                        location_y=shot_header_y,
                    ))

                    is_goal, positioning = GoalkeeperEngine.evaluate_save(
                        xg, DNAFactory.get_shooter_quality(receiver.dna),
                        shot_header_x, shot_header_y, gk,
                        state.last_ball_x, state.last_ball_y
                    )
                    if is_goal:
                        result.goal_scored    = True
                        result.goal_team      = att_team
                        result.goal_scorer    = receiver.name
                        result.goal_assistant = taker.name
                        result.add(cls.make_event(
                            minute, EventType.GOAL, att_team, receiver.name,
                            phase, gs, xg=xg, body_part="head",
                            situation=SituationType.CORNER, outcome=True,
                            secondary_player=taker.name,
                            location_x=random.uniform(90, 102),
                            location_y=random.uniform(28, 40),
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
                                location_x=shot_header_x,
                                location_y=shot_header_y,
                                metadata={"goalline_save": is_goalline_save}
                            ))
                        # ── SECOND BALL CHAOS AFTER SAVE ──────────
                        # ~8% chance of a loose-ball scramble after a corner save
                        if random.random() < 0.08:
                            rebound_player = cls._pick_aerial_threat(att_players, exclude=receiver.name)
                            if rebound_player:
                                rebound_x = random.uniform(88, 100)
                                rebound_y = random.uniform(24, 44)
                                result.add(cls.make_event(
                                    minute, EventType.BALL_RECOVERY, att_team, rebound_player.name,
                                    phase, gs,
                                    location_x=rebound_x, location_y=rebound_y,
                                    outcome=True,
                                    metadata={"loose_ball": True, "second_phase": True}
                                ))
                                # Second shot from the scramble
                                rebound_xg = XGEngine.calculate(
                                    zone="inside_box", body_part="right_foot",
                                    situation=SituationType.CORNER,
                                    is_big_chance=random.random() < 0.25
                                ) * 0.75  # Reduced quality - scramble
                                rebound_sot = random.random() < (0.25 + rebound_xg * 0.3)
                                if rebound_sot and random.random() < 0.40:  # ~40% chance of SOT from scramble
                                    result.add(cls.make_event(
                                        minute, EventType.SHOT_ON_TARGET, att_team, rebound_player.name,
                                        phase, gs, xg=rebound_xg, body_part="right_foot",
                                        situation=SituationType.CORNER, outcome=True,
                                        location_x=rebound_x, location_y=rebound_y,
                                    ))
                                    # FIX (scoreline realism + GK consistency): the
                                    # scramble goal used to roll flat against the raw
                                    # rebound_xg — a second goal chance that completely
                                    # bypassed the goalkeeper's save evaluation. That
                                    # was (a) inconsistent with every other shot in the
                                    # game and (b) a measurable second-goal generator per
                                    # corner, inflating scorelines. It now goes through
                                    # the same corrected GoalkeeperEngine.evaluate_save
                                    # as the initial header, so the keeper's positioning
                                    # and reach apply identically.
                                    is_goal, positioning = GoalkeeperEngine.evaluate_save(
                                        rebound_xg, DNAFactory.get_shooter_quality(rebound_player.dna),
                                        rebound_x, rebound_y, gk, state.last_ball_x, state.last_ball_y,
                                    )
                                    if is_goal:
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
                                        minute, EventType.SHOT_OFF_TARGET, att_team, rebound_player.name,
                                        phase, gs, xg=rebound_xg, outcome=False,
                                        location_x=rebound_x, location_y=rebound_y,
                                    ))
                else:
                    result.add(cls.make_event(
                        minute, EventType.SHOT_OFF_TARGET, att_team, receiver.name,
                        phase, gs, xg=xg, outcome=False,
                        location_x=random.uniform(88, 102),
                        location_y=random.uniform(20, 48),
                    ))
                    # ── OFF-TARGET SECOND BALL CHAOS ──────────────
                    # ~15% chance of the ball falling to another attacker after a missed header
                    if random.random() < 0.15:
                        second_attacker = cls._pick_aerial_threat(att_players, exclude=taker.name)
                        if second_attacker and second_attacker != receiver:
                            scramble_xg = XGEngine.calculate(
                                zone="inside_box", body_part="right_foot",
                                situation=SituationType.CORNER,
                            ) * 0.65
                            result.add(cls.make_event(
                                minute, EventType.BALL_RECOVERY, att_team, second_attacker.name,
                                phase, gs,
                                location_x=random.uniform(88, 100),
                                location_y=random.uniform(24, 44),
                                outcome=True,
                                metadata={"loose_ball": True}
                            ))
                            if random.random() < 0.30:
                                result.corner_won = True
                                result.corner_team = att_team
                                result.add(cls.make_event(
                                    minute, EventType.CORNER_WON, att_team, second_attacker.name,
                                    phase, gs, location_x=105, location_y=random.choice([0.0, 68.0])
                                ))
                if not result.goal_scored and random.random() < 0.25 and delivery_success and receiver:
                    # ── GK PUNCH / SECOND BALL FROM PARTIAL CLAIM ──
                    # GK punches rather than cleanly catching ~25% of the time
                    punch_clear = cls._pick_aerial_defender(att_players) or cls._pick_aerial_threat(def_players, exclude=receiver.name)
                    if punch_clear:
                        punch_x = random.uniform(83, 96)
                        punch_y = random.uniform(24, 44)
                        punch_team = att_team if random.random() < 0.5 else def_team
                        result.add(cls.make_event(
                            minute, EventType.BALL_RECOVERY, punch_team, punch_clear.name,
                            phase, gs,
                            location_x=punch_x, location_y=punch_y,
                            outcome=True,
                            metadata={"gk_punch": True}
                        ))
            else:
                # Defender wins — clearance
                result.add(cls.make_event(
                    minute, EventType.CLEARANCE, def_team,
                    defender.name if defender else "Unknown",
                    phase, gs,
                    location_x=random.uniform(83, 100),
                    location_y=random.uniform(18, 50),
                    outcome=True,
                ))
        else:
            # Delivery fails — GK claims or goes out
            if gk and random.random() < 0.55:
                # Bug fix (GK positional regression): never set a location,
                # silently defaulted to (50, 34) — center circle. A high
                # claim happens in the six-yard-box/goal-mouth area.
                result.add(cls.make_event(
                    minute, EventType.SAVE, def_team, gk.name,
                    phase, gs, outcome=True,
                    location_x=random.uniform(99, 104.3),
                    location_y=random.uniform(24, 44),
                    metadata={"type": "high_claim"}
                ))

        return result

    @classmethod
    def _freekick_chain(cls, minute, att_team, def_team,
                         att_players, def_players, state, situation,
                         attacks_right=True,
                         context_x: Optional[float] = None,
                         context_y: Optional[float] = None) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

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
        taker = cls._pick_sp_taker(att_players, situation="freekick", freekick_type=fk_type)
        gk    = cls._pick_gk_player(def_players)

        result.add(cls.make_event(
            minute, EventType.FREEKICK_WON, att_team, taker.name,
            phase, gs, location_x=fk_x, location_y=fk_y
        ))

        # Direct or crossed?
        direct_range = fk_x > 78 if attacks_right else fk_x < 27
        if situation == SituationType.DIRECT_FREEKICK and direct_range:
            # Direct shot
            result.add(cls.make_event(
                minute, EventType.FREEKICK_DIRECT, att_team, taker.name,
                phase, gs, location_x=fk_x, location_y=fk_y
            ))

            xg = XGEngine.calculate(
                zone=PitchZone.xg_zone(fk_x, fk_y, attacks_right=attacks_right),
                body_part=random.choice(["right_foot", "left_foot"]),
                situation=SituationType.DIRECT_FREEKICK,
            )
            result.xg_generated = xg
            sot_prob = 0.45 + taker.dna.technical.free_kick / 100.0 * 0.35
            if random.random() < sot_prob:
                result.shot_on_target = True
                # Bug fix: this SHOT_ON_TARGET never set a location either,
                # defaulting to (50, 34). A direct free kick on target lands
                # in the goal-mouth area, not midfield.
                shot_x = cls.mirror_x(random.uniform(96, 104.3), attacks_right)
                shot_y = random.uniform(26, 42)
                result.add(cls.make_event(
                    minute, EventType.SHOT_ON_TARGET, att_team, taker.name,
                    phase, gs, xg=xg, outcome=True,
                    secondary_player=gk.name if gk else None,
                    location_x=shot_x, location_y=shot_y,
                ))
                is_goal, positioning = GoalkeeperEngine.evaluate_save(
                    xg, DNAFactory.get_shooter_quality(taker.dna),
                    shot_x, shot_y, gk, state.last_ball_x, state.last_ball_y
                )
                if is_goal:
                    result.goal_scored  = True
                    result.goal_team    = att_team
                    result.goal_scorer  = taker.name
                    result.add(cls.make_event(
                        minute, EventType.GOAL, att_team, taker.name,
                        phase, gs, xg=xg, outcome=True,
                        situation=SituationType.DIRECT_FREEKICK,
                        location_x=shot_x, location_y=shot_y,
                    ))
                elif gk:
                    is_goalline_save = (
                        positioning.get("start_x") is not None
                        and positioning["start_x"] <= 2.0
                    )
                    result.add(cls.make_event(
                        minute, EventType.SAVE, def_team, gk.name,
                        phase, gs, xg=xg, outcome=True,
                        location_x=shot_x, location_y=shot_y,
                        metadata={"goalline_save": is_goalline_save}
                    ))
            else:
                result.add(cls.make_event(
                    minute, EventType.SHOT_OFF_TARGET, att_team, taker.name,
                    phase, gs, xg=xg, outcome=False,
                ))
        else:
            # Crossed free kick — becomes like a corner
            result.add(cls.make_event(
                minute, EventType.FREEKICK_CROSS, att_team, taker.name,
                phase, gs, location_x=fk_x, location_y=fk_y
            ))
            # Resolve like a corner
            sub = cls._corner_chain(minute, att_team, def_team, att_players, def_players, state, attacks_right)
            # Inherit events (minus the duplicate corner taken)
            result.events.extend(sub.events[1:])
            result.goal_scored    = sub.goal_scored
            result.goal_team      = sub.goal_team
            result.goal_scorer    = sub.goal_scorer
            result.goal_assistant = taker.name  # Taker gets the assist
            result.xg_generated   = sub.xg_generated
            result.xa_generated   = sub.xa_generated
            result.shot_on_target = sub.shot_on_target

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
                elif p.dna.preferred_foot == "both":
                    score *= 1.2
                score *= (0.6 + p.dna.technical.crossing / 100.0)

            elif situation == "freekick":
                fk_type = ctx.get("freekick_type", "crossed")
                if fk_type == "direct":
                    score *= (0.5 + p.dna.technical.penalty_taking / 100.0)
                    score *= (0.6 + p.dna.technical.finishing / 100.0)
                else:
                    score *= (0.7 + p.dna.mental.vision / 100.0)
                    score *= (0.6 + p.dna.technical.crossing / 100.0)

            elif situation == "penalty":
                score *= (0.3 + p.dna.technical.penalty_taking / 100.0)
                score *= (0.7 + p.dna.mental.composure / 100.0)

            return max(score, 0.1)

        weights = [_score(p) for p in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    @classmethod
    def _pick_aerial_threat(cls, players, exclude=None) -> Optional[PlayerProfile]:
        # The keeper attacks corners/crosses only in emergency last-minute
        # scenarios — never as the routine aerial target. A sweeper GK with
        # high jumping used to win this pick and then get logged winning a
        # duel at x≈100, which dragged his StatsBomb-style average touch
        # position miles upfield and broke the GK-anchoring regression guard.
        outfield = [p for p in players if p.position != "GK"] or players
        return cls.pick_weighted(
            outfield,
            lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2,
            exclude=exclude
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
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        # Who presses?
        presser = cls._pick_presser(press_players)
        pressed = cls._pick_pressed_player(retreat_players)
        if not presser or not pressed:
            return result

        # StatsBomb standard: single PRESSURE event only.
        # Success is inferred from what happens next (interception/tackle/turnover).
        press_x = random.uniform(55, 85)
        press_y = random.uniform(10, 58)

        result.add(cls.make_event(
            minute, EventType.PRESS, pressing_team, presser.name,
            phase, gs,
            secondary_player=pressed.name,
            location_x=press_x,
            location_y=press_y,
        ))

        # Press success?
        press_success_rate = (
            press_profile.press_success_rate
            * (presser.dna.physical.pace / 100.0 * 0.3 + 0.7)
            * (1.0 - pressed.dna.press_resistance / 100.0 * 0.4)
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
            # Press failed — player played through
            result.add(cls.make_event(
                minute, EventType.PASS, retreating_team, pressed.name,
                phase, gs, outcome=True,
                metadata={"press_resistance": True,
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
        shooter  = cls._pick_shooter(counter_players, exclude=carrier.name if carrier else None)
        if not carrier:
            return result

        # Spatial anchor: counter starts from WHERE the ball was won
        # not from a randomised position. This ensures telemetry continuity.
        x = anchor_x if anchor_x is not None else random.uniform(55, 75)
        y = anchor_y if anchor_y is not None else random.uniform(15, 53)

        # Carry forward fast from the anchor
        carry_dist, adv_ratio = cls._carry_distance_advance(
            carrier, x, counter_profile, is_counter=True
        )
        end_x = min(103, x + carry_dist * adv_ratio)
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

        # Pass to shooter or shoot directly?
        if shooter and shooter != carrier and random.random() < 0.55:
            pass_end_x = min(105, x + random.uniform(6, 14))
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
            # Attack chain anchored at pass end coordinates
            attack = AttackChain.generate(
                minute, counter_team, defending_team,
                counter_players, def_players,
                counter_profile, counter_profile,
                state, SituationType.FAST_BREAK,
                context_x=pass_end_x, context_y=pass_end_y,
                position_engine=position_engine,
            )
            result.events.extend(attack.events)
            result.goal_scored    = attack.goal_scored
            result.goal_team      = attack.goal_team
            result.goal_scorer    = attack.goal_scorer
            result.goal_assistant = attack.goal_assistant
            result.xg_generated   = attack.xg_generated
            result.xa_generated   = attack.xa_generated
            result.shot_on_target = attack.shot_on_target
        else:
            # Solo run and shot — anchored at carry end
            attack = AttackChain.generate(
                minute, counter_team, defending_team,
                [carrier], def_players,
                counter_profile, counter_profile,
                state, SituationType.FAST_BREAK,
                context_x=x, context_y=y,
                position_engine=position_engine,
            )
            result.events.extend(attack.events)
            result.goal_scored    = attack.goal_scored
            result.goal_team      = attack.goal_team
            result.goal_scorer    = attack.goal_scorer
            result.xg_generated   = attack.xg_generated
            result.shot_on_target = attack.shot_on_target

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
            tackle_rate = DNAFactory.get_tackle_success_rate(defender.dna)
            success = tackle_rate > random.random()
            result.add(cls.make_event(
                minute,
                EventType.TACKLE_WON if success else EventType.TACKLE_LOST,
                defending_team, defender.name,
                phase, gs,
                secondary_player=attacker.name if attacker else None,
                location_x=x, location_y=y,
                outcome=success,
                metadata={"danger_before": round(danger_level, 1)},
            ))
            if not success:
                # Not every failed tackle is a foul — depends on aggression.
                # A foul off a bad challenge is the minority outcome (~40%
                # of failed tackles at a dirty-profile player, less for a
                # clean one), which keeps contextual tackle fouls realistic
                # on top of the per-minute foul flow.
                foul_base = 0.40
                if defender.dna.tendencies.tackles_aggressively > 0.6:
                    foul_base *= 1.3
                foul_base *= max(0.5, defender.dna.tendencies.commits_fouls * 2.0)
                pos_mult = {"CDM": 1.4, "CB": 1.3, "CM": 1.15, "LB": 1.1, "RB": 1.1}.get(defender.position, 1.0)
                is_foul = random.random() < (foul_base * pos_mult)

                if is_foul:
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
                        * (0.5 + referee_strictness)
                        * (defender.dna.tendencies.tackles_aggressively * 1.5 + 0.3)
                        * (0.6 + defender.dna.tendencies.commits_fouls)
                    )
                    if random.random() < card_prob:
                        is_red = random.random() < 0.04
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
            # Checkpoint 7: soul readers (Sweeper Sage, Defensive Purist)
            # intercept more — anticipation is their defining trait.
            #base_int = SoulApplicator.modify_interception_rate(
                #defender, base_int, state, defending_team)
            success = random.random() < base_int
            result.add(cls.make_event(
                minute, EventType.INTERCEPTION,
                defending_team, defender.name,
                phase, gs,
                location_x=x, location_y=y,
                outcome=success,
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
                    secondary_player=attacker.name if attacker else None,
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
            if block_roll < 0.52:
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
            result.add(cls.make_event(
                minute, EventType.BLOCK,
                defending_team, defender.name,
                phase, gs,
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
            if random.random() < 0.65:
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
        # yellow match off ~25 fouls). Referee strictness is the main dial
        # (~0.45x lenient → ~1.55x strict), with the fouler's discipline
        # record and personality nudging it on top.
        card_prob = (
            0.15
            * PhaseEngine.card_mult(phase)
            * (0.45 + 1.1 * referee_strictness)
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
                is_straight_red = random.random() < 0.06
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
    ) -> ChainResult:
        result = ChainResult()
        phase, gs = state.phase, state.game_state

        result.add(cls.make_event(
            minute, EventType.SUBSTITUTION, team,
            player_off.name,
            phase, gs,
            secondary_player=player_on.name,
            metadata={
                "player_off": player_off.name,
                "player_on": player_on.name,
                "reason": reason,
                "position_off": player_off.position,
                "position_on":  player_on.position,
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
    ) -> ChainResult:
        return PossessionChain.generate(
            minute, attacking_team, players, team_profile, state, seq_length,
            defending_players=defending_players, position_engine=position_engine,
            context_x=context_x, context_y=context_y,
            attacks_right=attacks_right,
            def_press_intensity=def_press_intensity,
            def_style_key=def_style_key,
            att_style_key=att_style_key,
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
    ) -> ChainResult:
        res = AttackChain.generate(
            minute, att_team, def_team,
            att_players, def_players,
            att_profile, def_profile, state, situation,
            context_x=context_x, context_y=context_y,
            position_engine=position_engine,
            attacks_right=attacks_right,
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
    ) -> ChainResult:
        return SetPieceChain.generate(
            minute, att_team, def_team,
            att_players, def_players, state, situation,
            attacks_right=attacks_right,
            context_x=context_x,
            context_y=context_y,
        )

    @staticmethod
    def transition(
        minute, pressing_team, retreating_team,
        press_players, retreat_players,
        press_profile, state, position_engine=None,
        attacks_right: bool = True,
    ) -> ChainResult:
        return TransitionChain.generate(
            minute, pressing_team, retreating_team,
            press_players, retreat_players,
            press_profile, state, position_engine=position_engine,
            attacks_right=attacks_right,
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
    def discipline(
        minute, fouling_team, fouled_team,
        fouling_players, fouled_players, state,
        referee_strictness=0.5, x=None, y=None,
        attacks_right: bool = True,
    ) -> ChainResult:
        return DisciplineChain.generate(
            minute, fouling_team, fouled_team,
            fouling_players, fouled_players, state,
            referee_strictness, x, y,
            attacks_right=attacks_right,
            booked_players=state.booked_players,
        )

    @staticmethod
    def substitution(
        minute, team, player_off, player_on, state, reason="tactical"
    ) -> ChainResult:
        return SubstitutionChain.generate(
            minute, team, player_off, player_on, state, reason
        )
    


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

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

            if target_att and target_def:
                att_win = random.random() < 0.50
                result.add(cls.make_event(
                    minute, EventType.AERIAL_DUEL, kicking_team, target_att.name,
                    phase, gs,
                    secondary_player=target_def.name,
                    location_x=end_x, location_y=end_y,
                    outcome=att_win,
                    metadata={"from_goal_kick": True}
                ))
                if not att_win:
                    result.possession_lost = True

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
                throw_players,
                lambda p: (p.dna.physical.jumping + p.dna.technical.heading) / 2 if p.name != taker.name else 0.1
            )
            defender = cls.pick_weighted(
                def_players,
                lambda p: (p.dna.physical.jumping + p.dna.defending.clearances) / 2
            )

            if receiver and defender:
                att_win = random.random() < 0.48
                result.add(cls.make_event(
                    minute, EventType.AERIAL_DUEL, throwing_team, receiver.name,
                    phase, gs,
                    secondary_player=defender.name,
                    location_x=end_x, location_y=end_y,
                    outcome=att_win,
                    metadata={"long_throw_box_scramble": True}
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
            receiver = cls.pick_weighted(
                throw_players,
                lambda p: 3.0 if p.position in ("CM", "CAM", "LW", "RW", "ST") else 1.0,
                exclude=taker.name
            ) or throw_players[0]

            end_x = max(2.0, min(103.0, x + random.uniform(-4, 6)))
            end_y = max(4.0, min(64.0, y + (5.0 if y < 34 else -5.0)))

            result.add(cls.make_event(
                minute, EventType.THROW_IN, throwing_team, taker.name,
                phase, gs,
                secondary_player=receiver.name,
                location_x=x, location_y=y,
                end_x=end_x, end_y=end_y,
                outcome=True
            ))

            result.add(cls.make_event(
                minute, EventType.BALL_RECEIPT, throwing_team, receiver.name,
                phase, gs,
                location_x=end_x, location_y=end_y,
                outcome=True
            ))

            if position_engine:
                position_engine.record_touch(taker.name, x, y, minute)
                position_engine.record_touch(receiver.name, end_x, end_y, minute)

        return result