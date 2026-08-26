"""
PLOFA 26/27 SEASON ENGINE
=========================
Match Simulation Engine — Core Module

Philosophy:
    The match PLAYS ITSELF. The score is a RESULT, not an input.
    Every stat is a CONSEQUENCE of simulated events, not a random draw.

Architecture:
    MatchEngine         — The simulation timeline (this file)
    player_dna.py       — Player archetypes & attribute system
    event_chain.py      — Causal event chains (dribble→carry→shot)
    stat_accumulator.py — Converts events into stats
    exporter.py         — Excel/CSV/JSON/SQLite output
"""

from __future__ import annotations
# squad_manager imported lazily inside methods to avoid circular imports
import math
import random
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from enum import Enum, auto
from datetime import date

from position_engine import PositionEngine
from threat_engine import ThreatEngine
from block_awareness import BlockShape, BlockDetector

# The match narrative prints emoji/unicode; on legacy consoles (cp1252 etc.)
# that raises UnicodeEncodeError mid-simulation. Reconfigure the streams to
# fall back to '?' rather than crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")  # type: ignore
    except Exception:
        pass


# ─────────────────────────────────────────────
# ENUMS — The language of the simulation
# ─────────────────────────────────────────────

class MatchPhase(Enum):
    """A match has psychological phases, not just minutes."""
    OPENING        = "opening"        # 1–15:  Feeling out, cautious
    FIRST_SPELL    = "first_spell"    # 16–30: First real pressure
    FIRST_HALF_END = "first_half_end" # 31–45: Late first-half push
    SECOND_OPEN    = "second_open"    # 46–60: Second half reset
    PEAK_INTENSITY = "peak_intensity" # 61–75: Match decided here most often
    FINAL_PUSH     = "final_push"     # 76–90: Desperation or control
    ADDED_TIME     = "added_time"     # 90+:   Chaos or calm

class GameState(Enum):
    """Who is in control right now?"""
    LEVEL       = auto()   # 0-0 or tied
    HOME_AHEAD  = auto()   # Home team leading
    AWAY_AHEAD  = auto()   # Away team leading
    HOME_CRUISE = auto()   # Home 2+ goals ahead, managing
    AWAY_CRUISE = auto()   # Away 2+ goals ahead, managing
    HOME_CHASE  = auto()   # Home chasing 2+ goals deficit
    AWAY_CHASE  = auto()   # Away chasing 2+ goals deficit

class EventType(Enum):
    """Every discrete thing that can happen in a match."""
    # Possession events
    POSSESSION_SEQUENCE  = auto()
    PASS                 = auto()
    CARRY                = auto()
    DRIBBLE_ATTEMPT      = auto()
    DRIBBLE_SUCCESS      = auto()
    DRIBBLE_FAIL         = auto()
    CROSS_ATTEMPT        = auto()
    CROSS_SUCCESS        = auto()
    THROUGH_BALL         = auto()
    PROGRESSIVE_PASS     = auto()
    SWITCH_OF_PLAY       = auto()

    # Transition events
    TURNOVER             = auto()
    INTERCEPTION         = auto()
    TACKLE_WON           = auto()
    TACKLE_LOST          = auto()
    CLEARANCE            = auto()
    BLOCK                = auto()
    RECOVERY             = auto()
    PRESS                = auto()
    PRESS_SUCCESS        = auto()

    # Chance events (the core chain)
    CHANCE_CREATED       = auto()
    BIG_CHANCE_CREATED   = auto()
    SHOT_ATTEMPT         = auto()
    SHOT_ON_TARGET       = auto()
    SHOT_OFF_TARGET      = auto()
    SHOT_BLOCKED         = auto()
    HIT_WOODWORK         = auto()   # Checkpoint 6: post/bar strike, previously absent entirely
    SAVE                 = auto()
    GOAL                 = auto()
    OWN_GOAL             = auto()
    PENALTY_WON          = auto()
    PENALTY_SCORED       = auto()
    PENALTY_MISSED       = auto()

    # Set piece events
    CORNER_WON           = auto()
    CORNER_TAKEN         = auto()
    FREEKICK_WON         = auto()
    FREEKICK_DIRECT      = auto()
    FREEKICK_CROSS       = auto()
    THROW_IN             = auto()
    GOAL_KICK            = auto()
    OFFSIDE              = auto()
    VAR_DISALLOWED_GOAL  = auto()
    KICKOFF              = auto()

    # Discipline events
    FOUL_COMMITTED       = auto()
    FOUL_WON             = auto()
    YELLOW_CARD          = auto()
    RED_CARD             = auto()

    # Physical events
    AERIAL_DUEL          = auto()
    GROUND_DUEL          = auto()
    SPRINT               = auto()

    # StatsBomb-standard atomic events
    BALL_RECEIPT         = auto()   # Logged for every completed pass receiver
    MISCONTROL           = auto()   # Failed first touch / bad control
    DISPOSSESSED         = auto()   # Player loses ball under pressure
    BALL_RECOVERY        = auto()   # Defensive recovery of loose ball
    FIFTY_FIFTY          = auto()   # Contested loose ball duel
    PRESSURE             = auto()   # Single pressure event (StatsBomb standard)

    # Match control events
    SUBSTITUTION         = auto()
    INJURY               = auto()
    ADDED_TIME_SIGNAL    = auto()


class SituationType(Enum):
    """How did a chance/goal originate?"""
    OPEN_PLAY       = "open_play"
    FAST_BREAK      = "fast_break"
    CORNER          = "corner"
    DIRECT_FREEKICK = "direct_freekick"
    CROSSED_FREEKICK = "crossed_freekick"
    PENALTY         = "penalty"
    THROW_IN        = "throw_in"
    OWN_GOAL        = "own_goal"


class TeamStyle(Enum):
    ULTRA_ATTACKING      = "ultra_attacking"
    ATTACKING            = "attacking"
    BALANCED             = "balanced"
    DEFENSIVE            = "defensive"
    ULTRA_DEFENSIVE      = "ultra_defensive"
    GEGENPRESSING        = "gegenpressing"
    TIKI_TAKA            = "tiki_taka"
    PARK_THE_BUS         = "park_the_bus"
    ROUTE_ONE            = "route_one"
    WING_PLAY            = "wing_play"
    VERTICAL_TIKI_TAKA   = "vertical_tiki_taka"
    FLUID_COUNTER        = "fluid_counter"
    STRUCTURED_POSSESSION = "structured_possession"


class PlayingStyle(Enum):
    POSSESSION          = "possession"
    COUNTER             = "counter"
    MIXED               = "mixed"
    DIRECT              = "direct"
    PATIENT_BUILD_UP    = "patient_build_up"
    HIGH_PRESS          = "high_press"
    LOW_BLOCK           = "low_block"
    TRANSITION_FOCUSED  = "transition_focused"


class Intensity(Enum):
    LOW       = "low"
    MEDIUM    = "medium"
    HIGH      = "high"
    VERY_HIGH = "very_high"


# ─────────────────────────────────────────────
# CORE DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class MatchEvent:
    """
    A single atomic event in the match timeline.
    Everything that happens is an event. Stats are derived FROM events.
    """
    minute: int
    second: int
    event_type: EventType
    team: str
    player: str                          # Primary actor
    secondary_player: Optional[str] = None  # Receiver, fouled player, etc.
    situation: SituationType = SituationType.OPEN_PLAY
    location_x: float = 50.0            # 0–105 (meters from home goal line)
    location_y: float = 34.0            # 0–68 (meters from left touchline)
    end_x: Optional[float] = None       # Where event ended (passes, carries)
    end_y: Optional[float] = None
    xg: float = 0.0                     # xG value if shot/chance
    xa: float = 0.0                     # xA value if assist action
    outcome: bool = True                # Did the action succeed?
    body_part: str = "right_foot"       # foot/head/other
    phase: MatchPhase = MatchPhase.OPENING
    game_state: GameState = GameState.LEVEL
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra context

    def __post_init__(self):
        # The event chains stamp the foot/head used for passes, through balls
        # and crosses inside metadata["body_part"]. Promote it to the field so
        # every consumer (exporter "Footed Passes"/"Footed Events" sheets,
        # threat engine, header detection) sees the REAL body part instead of
        # the "right_foot" dataclass default. Shots already pass body_part as
        # a top-level kwarg, so this is a no-op for them (same value).
        md_body = self.metadata.get("body_part")
        if md_body:
            self.body_part = md_body

    @property
    def is_shot(self) -> bool:
        return self.event_type in (
            EventType.SHOT_ON_TARGET,
            EventType.SHOT_OFF_TARGET,
            EventType.SHOT_BLOCKED,
            EventType.GOAL,
            EventType.PENALTY_SCORED,
            EventType.PENALTY_MISSED,
        )

    @property
    def is_defensive(self) -> bool:
        return self.event_type in (
            EventType.TACKLE_WON, EventType.TACKLE_LOST,
            EventType.INTERCEPTION, EventType.CLEARANCE,
            EventType.BLOCK, EventType.RECOVERY,
        )

    @property
    def distance_from_goal(self) -> float:
        """Euclidean distance from the attacking goal (x=105, y=34)."""
        return ((self.location_x - 105) ** 2 + (self.location_y - 34) ** 2) ** 0.5


@dataclass
class TeamProfile:
    """
    A team's identity for this match.
    DNA that shapes HOW they play, not just what numbers they produce.
    """
    name: str
    style: TeamStyle
    playing_style: PlayingStyle
    intensity: Intensity

    uses_false_nine: bool = False

    # Tactical DNA (0.0–1.0 scales)
    press_intensity: float = 0.5        # How aggressively they press
    defensive_line: float = 0.5         # 0=deep, 1=high line
    width: float = 0.5                  # 0=narrow, 1=wide
    tempo: float = 0.5                  # 0=slow, 1=fast
    directness: float = 0.5             # 0=patient, 1=direct

    # Derived probabilities (set during __post_init__)
    possession_target: float = 50.0     # Natural possession tendency
    shots_per_sequence: float = 0.15    # Chance a possession sequence ends in shot
    big_chance_ratio: float = 0.35      # % of chances that are "big"
    press_success_rate: float = 0.25    # % of presses that win ball

    def __post_init__(self):
        self._apply_style_dna()

    def _apply_style_dna(self):
        """Map style enum to tactical DNA values."""
        style_profiles = {
            TeamStyle.ULTRA_ATTACKING: {
                'press_intensity': 0.8, 'defensive_line': 0.8,
                'width': 0.7, 'tempo': 0.9, 'directness': 0.7,
                'possession_target': 55.0, 'shots_per_sequence': 0.15,
                'big_chance_ratio': 0.40, 'press_success_rate': 0.30,
            },
            TeamStyle.ATTACKING: {
                'press_intensity': 0.65, 'defensive_line': 0.65,
                'width': 0.6, 'tempo': 0.7, 'directness': 0.6,
                'possession_target': 52.0, 'shots_per_sequence': 0.13,
                'big_chance_ratio': 0.37, 'press_success_rate': 0.27,
            },
            TeamStyle.GEGENPRESSING: {
                'press_intensity': 0.95, 'defensive_line': 0.75,
                'width': 0.6, 'tempo': 0.95, 'directness': 0.65,
                'possession_target': 50.0, 'shots_per_sequence': 0.14,
                'big_chance_ratio': 0.38, 'press_success_rate': 0.40,
            },
            TeamStyle.TIKI_TAKA: {
                'press_intensity': 0.72, 'defensive_line': 0.70,
                'width': 0.5, 'tempo': 0.55, 'directness': 0.25,
                'possession_target': 70.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.30, 'press_success_rate': 0.38,
            },
            TeamStyle.BALANCED: {
                'press_intensity': 0.50, 'defensive_line': 0.50,
                'width': 0.5, 'tempo': 0.55, 'directness': 0.50,
                'possession_target': 50.0, 'shots_per_sequence': 0.11,
                'big_chance_ratio': 0.33, 'press_success_rate': 0.25,
            },
            TeamStyle.DEFENSIVE: {
                'press_intensity': 0.30, 'defensive_line': 0.30,
                'width': 0.4, 'tempo': 0.40, 'directness': 0.55,
                'possession_target': 42.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.28, 'press_success_rate': 0.18,
            },
            TeamStyle.ULTRA_DEFENSIVE: {
                'press_intensity': 0.15, 'defensive_line': 0.15,
                'width': 0.35, 'tempo': 0.30, 'directness': 0.60,
                'possession_target': 35.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.25, 'press_success_rate': 0.12,
            },
            TeamStyle.PARK_THE_BUS: {
                'press_intensity': 0.10, 'defensive_line': 0.10,
                'width': 0.30, 'tempo': 0.25, 'directness': 0.65,
                'possession_target': 32.0, 'shots_per_sequence': 0.04,
                'big_chance_ratio': 0.22, 'press_success_rate': 0.10,
            },
            TeamStyle.WING_PLAY: {
                'press_intensity': 0.55, 'defensive_line': 0.55,
                'width': 0.90, 'tempo': 0.65, 'directness': 0.60,
                'possession_target': 48.0, 'shots_per_sequence': 0.12,
                'big_chance_ratio': 0.35, 'press_success_rate': 0.22,
            },
            TeamStyle.ROUTE_ONE: {
                'press_intensity': 0.40, 'defensive_line': 0.40,
                'width': 0.55, 'tempo': 0.80, 'directness': 0.90,
                'possession_target': 38.0, 'shots_per_sequence': 0.09,
                'big_chance_ratio': 0.30, 'press_success_rate': 0.20,
            },
            TeamStyle.VERTICAL_TIKI_TAKA: {
                'press_intensity': 0.65, 'defensive_line': 0.65,
                'width': 0.55, 'tempo': 0.70, 'directness': 0.55,
                'possession_target': 60.0, 'shots_per_sequence': 0.12,
                'big_chance_ratio': 0.36, 'press_success_rate': 0.30,
            },
            TeamStyle.STRUCTURED_POSSESSION: {
                'press_intensity': 0.55, 'defensive_line': 0.55,
                'width': 0.5, 'tempo': 0.50, 'directness': 0.35,
                'possession_target': 62.0, 'shots_per_sequence': 0.09,
                'big_chance_ratio': 0.31, 'press_success_rate': 0.30,
            },
            TeamStyle.FLUID_COUNTER: {
                'press_intensity': 0.45, 'defensive_line': 0.40,
                'width': 0.65, 'tempo': 0.75, 'directness': 0.72,
                'possession_target': 43.0, 'shots_per_sequence': 0.12,
                'big_chance_ratio': 0.38, 'press_success_rate': 0.22,
            },
        }
        profile = style_profiles.get(self.style, style_profiles[TeamStyle.BALANCED])
        for attr, val in profile.items():
            setattr(self, attr, val)

        # Intensity modifier
        intensity_mult = {
            Intensity.LOW: 0.80,
            Intensity.MEDIUM: 1.00,
            Intensity.HIGH: 1.15,
            Intensity.VERY_HIGH: 1.30,
        }[self.intensity]

        self.press_intensity = min(1.0, self.press_intensity * intensity_mult)
        self.tempo = min(1.0, self.tempo * intensity_mult)
        self.press_success_rate = min(0.55, self.press_success_rate * intensity_mult)


@dataclass
class MatchConfig:
    """All the metadata about this match."""
    home_team: str
    away_team: str
    match_date: date = field(default_factory=date.today)
    matchday: int = 1
    season: str = "26/27"
    competition: str = "PLOFA"
    venue: str = "Unknown Stadium"
    stadium_capacity: int = 35000
    referee: str = "Unknown Referee"
    referee_strictness: float = 0.5    # 0=lenient, 1=strict
    is_derby: bool = False
    home_advantage: float = 0.08       # % boost to home team probabilities
    weather: str = "clear"             # clear, rain, wind, fog


# ─────────────────────────────────────────────
# MATCH STATE — Live state during simulation
# ─────────────────────────────────────────────

@dataclass
class MatchState:
    """
    The live state of the match at any given minute.
    This is what drives ALL probability calculations.
    """
    minute: int = 0
    second: int = 0
    home_goals: int = 0
    away_goals: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0

    # Momentum (−100 to +100: negative=away dominant, positive=home dominant)
    momentum: float = 0.0

    # Who has the ball right now
    possession_team: str = ""

    # Phase
    phase: MatchPhase = MatchPhase.OPENING

    # Red cards (affects team strength)
    home_red_cards: int = 0
    away_red_cards: int = 0

    # Substitutions made
    home_subs_made: int = 0
    away_subs_made: int = 0

    # Consecutive actions by same team (builds/breaks momentum)
    consecutive_home_possessions: int = 0
    consecutive_away_possessions: int = 0

    # Added time (decided at ~88th minute)
    added_time: int = 0

    # Checkpoint 6 — corner consistency: how many corners each team has won
    # and is owed the next set-piece sequence. Incremented by _absorb_chain()
    # when a ChainResult reports corner_won=True, decremented-and-consumed by
    # _simulate_minute() before the normal situation roll — this is what
    # makes corners an actual CONSEQUENCE of a blocked shot / clearance
    # rather than an independent random draw that happened to coincide.
    #
    # These are PER-TEAM COUNTERS rather than a single latch slot: a corner
    # won mid-sequence used to overwrite any earlier-pending corner from the
    # same sequence, silently dropping ~60% of legitimately-won corners (so
    # averages landed at ~3 instead of the 8-11 real-football range). Counting
    # queues let every won corner survive to be taken.
    pending_corners_home: int = 0
    pending_corners_away: int = 0

    # Checkpoint X — penalty causality: a foul the defending team commits
    # inside its OWN box is a spot-kick offence. When a DisciplineChain
    # reports penalty_won=True, the fouled team is owed the turn penalty
    # sequence and it is consumed (queued -> PenaltyChain) exactly like a
    # won corner. Per-TEAM counters so several won penalties in a minute
    # (rare but possible under a siege) all survive to be taken instead of
    # overwriting one another.
    pending_penalty_home: int = 0
    pending_penalty_away: int = 0

    # Per-defending-team tally of spot-kicks actually taken this match. Keeps
    # the box-penalty physics realistic: a siege can produce one, occasionally
    # two, but never a deluge (otherwise a single match turns into a spot-kick
    # carnival). MatchEngine refuses the box-foul→penalty conversion once a
    # team has already conceded this many.
    penalties_taken: Dict[str, int] = field(default_factory=dict)
    PENALTY_CAP_PER_TEAM: int = 2

    # Checkpoint 7 — persistent ball-state: the last REAL location the ball
    # was seen at (from an actual event's end_x/end_y, or location_x/y if no
    # end coords exist). Every new possession sequence anchors its starting
    # position off THIS instead of drawing an independent random zone —
    # this is what stops the ball "teleporting" between sequences. Defaults
    # to the center circle, which is also what it resets to after a goal
    # (kickoff) and at kickoff itself.
    last_ball_x: float = 52.5
    last_ball_y: float = 34.0
    
    pending_kickoff_for: str = ""
    first_half_kickoff_team: str = ""
    pending_second_half_kickoff: bool = False

    # Checkpoint 8 — restart causality: goal kicks and throw-ins
    pending_goal_kick_for: str = ""
    pending_throw_in_for: str = ""
    pending_restart_x: float = 0.0
    pending_restart_y: float = 0.0

    # Checkpoint 19 — offside free kicks: placed at the offside location
    pending_offside_fk_for: str = ""
    pending_offside_fk_x: float = 0.0
    pending_offside_fk_y: float = 0.0

    # Disciplinary tracking
    booked_players: Dict[str, int] = field(default_factory=lambda: {})
    sent_off_players: List[str] = field(default_factory=lambda: [])

    # Checkpoint 11 — cross situations: when a delivery is detected (a
    # CROSS_ATTEMPT/CROSS_SUCCESS/corner, OR any pass stamped `cross: true`
    # by the geometric CrossDetector), the attacking team's off-ball players
    # crash the box (PositionEngine.attacking_crash) and the defending
    # team's danger is forced HIGH/CRITICAL by the threat engine. Reset each
    # minute so a cross only shapes the block for the minute it happens in.
    cross_active: bool = False
    cross_team: str = ""
    cross_player: str = ""
    cross_x: float = 52.5
    cross_y: float = 34.0
    cross_attacks_right: bool = True
    # Realistic corner causality (not a random draw): whether a live cross
    # this minute has ALREADY been turned into a corner by the defender /
    # keeper putting the delivery behind. Guards so one uncontested cross
    # can only ever concede at most one corner — no double-counting.
    cross_corner_done: bool = False

    home_block: Optional[BlockShape] = None
    away_block: Optional[BlockShape] = None

    @property
    def goal_difference(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def game_state(self) -> GameState:
        gd = self.goal_difference
        if gd == 0:
            return GameState.LEVEL
        elif gd == 1:
            return GameState.HOME_AHEAD
        elif gd == -1:
            return GameState.AWAY_AHEAD
        elif gd >= 2:
            return GameState.HOME_CRUISE
        elif gd <= -2:
            return GameState.AWAY_CRUISE
        return GameState.LEVEL

    @property
    def score_str(self) -> str:
        return f"{self.home_goals}–{self.away_goals}"


# ─────────────────────────────────────────────
# PHASE ENGINE — Defines the psychological arc
# ─────────────────────────────────────────────

class PhaseEngine:
    """
    Manages match phases and their probability multipliers.

    Real football has rhythms. This models them.
    Goals are more likely in certain phases.
    Pressing is more intense in certain phases.
    Cards spike in certain phases.
    """

    PHASE_MINUTES = {
        MatchPhase.OPENING:        (1,  15),
        MatchPhase.FIRST_SPELL:    (16, 30),
        MatchPhase.FIRST_HALF_END: (31, 45),
        MatchPhase.SECOND_OPEN:    (46, 60),
        MatchPhase.PEAK_INTENSITY: (61, 75),
        MatchPhase.FINAL_PUSH:     (76, 90),
        MatchPhase.ADDED_TIME:     (91, 99),
    }

    # How likely a goal is in each phase relative to baseline
    # Real data: most goals 75-90, fewest 1-15
    GOAL_PROBABILITY_MULTIPLIERS = {
        MatchPhase.OPENING:        0.70,
        MatchPhase.FIRST_SPELL:    0.90,
        MatchPhase.FIRST_HALF_END: 1.10,   # Late first-half goals
        MatchPhase.SECOND_OPEN:    1.00,
        MatchPhase.PEAK_INTENSITY: 1.20,   # Most goals here
        MatchPhase.FINAL_PUSH:     1.35,   # Desperation/control
        MatchPhase.ADDED_TIME:     1.50,   # Chaos minutes
    }

    # How likely a card is in each phase
    CARD_PROBABILITY_MULTIPLIERS = {
        MatchPhase.OPENING:        0.60,
        MatchPhase.FIRST_SPELL:    0.80,
        MatchPhase.FIRST_HALF_END: 1.10,
        MatchPhase.SECOND_OPEN:    0.90,
        MatchPhase.PEAK_INTENSITY: 1.30,   # Frustration peak
        MatchPhase.FINAL_PUSH:     1.50,   # Desperation
        MatchPhase.ADDED_TIME:     1.80,   # Maximum tension
    }

    # Press intensity per phase
    PRESS_INTENSITY_MULTIPLIERS = {
        MatchPhase.OPENING:        0.80,
        MatchPhase.FIRST_SPELL:    1.00,
        MatchPhase.FIRST_HALF_END: 1.10,
        MatchPhase.SECOND_OPEN:    1.00,
        MatchPhase.PEAK_INTENSITY: 1.20,
        MatchPhase.FINAL_PUSH:     1.30,
        MatchPhase.ADDED_TIME:     1.40,
    }

    @classmethod
    def get_phase(cls, minute: int) -> MatchPhase:
        for phase, (start, end) in cls.PHASE_MINUTES.items():
            if start <= minute <= end:
                return phase
        return MatchPhase.FINAL_PUSH

    @classmethod
    def goal_mult(cls, phase: MatchPhase) -> float:
        return cls.GOAL_PROBABILITY_MULTIPLIERS.get(phase, 1.0)

    @classmethod
    def card_mult(cls, phase: MatchPhase) -> float:
        return cls.CARD_PROBABILITY_MULTIPLIERS.get(phase, 1.0)

    @classmethod
    def press_mult(cls, phase: MatchPhase) -> float:
        return cls.PRESS_INTENSITY_MULTIPLIERS.get(phase, 1.0)


# ─────────────────────────────────────────────
# MOMENTUM ENGINE — The heart of realism
# ─────────────────────────────────────────────

class MomentumEngine:
    """
    Momentum is the invisible force that makes football feel real.

    A goal shifts momentum. A red card shifts it harder.
    A near-miss builds it. A poor pass bleeds it.
    Crowd noise (home advantage) sustains it.

    Range: −100 (away dominance) to +100 (home dominance)
    Neutral: 0
    """

    @staticmethod
    def after_goal(state: MatchState, scoring_team: str, home_team: str) -> float:
        """Goal dramatically shifts momentum."""
        shift = random.uniform(18, 30)   # Big momentum swing
        if scoring_team == home_team:
            return min(100, state.momentum + shift)
        else:
            return max(-100, state.momentum - shift)

    @staticmethod
    def after_red_card(state: MatchState, carded_team: str, home_team: str) -> float:
        """Red card is a massive momentum shift."""
        shift = random.uniform(25, 40)
        if carded_team == home_team:
            return max(-100, state.momentum - shift)
        else:
            return min(100, state.momentum + shift)

    @staticmethod
    def after_save(state: MatchState, saving_team: str, home_team: str) -> float:
        """Big saves shift momentum toward the saving team."""
        shift = random.uniform(5, 12)
        if saving_team == home_team:
            return min(100, state.momentum + shift)
        else:
            return max(-100, state.momentum - shift)

    @staticmethod
    def natural_decay(state: MatchState, home_team: str, home_profile: TeamProfile) -> float:
        """
        Momentum naturally decays toward 0 (equilibrium),
        but home advantage creates a slight positive pull.
        """
        home_bias = home_profile.possession_target - 50.0  # +ve = possession team
        home_advantage_pull = 3.0  # Points pulled toward home advantage per phase

        decay_rate = 0.08  # 8% decay per event toward baseline
        baseline = home_advantage_pull

        new_momentum = state.momentum * (1 - decay_rate) + baseline * decay_rate
        return round(new_momentum, 2)

    @staticmethod
    def get_attacking_probability_modifier(state: MatchState, team: str, home_team: str) -> float:
        """
        Convert current momentum to an attack probability modifier.
        Home team benefits from positive momentum, away from negative.
        """
        if team == home_team:
            raw = state.momentum / 100.0
        else:
            raw = -state.momentum / 100.0

        # Scale: momentum gives max ±25% probability boost
        return 1.0 + (raw * 0.25)

    @staticmethod
    def get_game_state_modifier(state: MatchState, team: str, home_team: str) -> float:
        """
        Teams react differently based on scoreline.
        Losing teams push forward (↑ attack chance), winning teams hold (↓ attack chance).
        """
        gd = state.goal_difference if team == home_team else -state.goal_difference
        minute = state.minute
        late_game = minute >= 70

        # Base modifiers by goal difference
        if gd >= 2:
            # Cruising — conservative
            base = 0.75 if not late_game else 0.65
        elif gd == 1:
            # Protecting lead
            base = 0.88 if not late_game else 0.80
        elif gd == 0:
            # Level — normal
            base = 1.00
        elif gd == -1:
            # Chasing — push forward
            base = 1.12 if late_game else 1.05
        else:
            # Desperate — all out attack
            base = 1.35 if late_game else 1.15

        return base


# ─────────────────────────────────────────────
# XG ENGINE — Shot quality calculation
# ─────────────────────────────────────────────

class XGEngine:
    """
    Realistic xG calculation based on shot characteristics.
    Every shot has an xG value. Goals emerge from xG probabilities.
    """

    # Base xG by shot origin zone
    ZONE_XG = {
        "six_yard_box":    0.45,
        "penalty_spot":    0.65,
        "inside_box":      0.16,
        "edge_of_box":     0.06,
        "outside_box":     0.022,
        "long_range":      0.008,
    }

    # Body part multipliers
    BODY_PART_MULT = {
        "right_foot":  1.00,
        "left_foot":   0.95,
        "head":        0.70,   # Headers convert less despite good positions
        "other":       0.45,
    }

    # Situation multipliers
    SITUATION_MULT = {
        SituationType.OPEN_PLAY:        1.00,
        SituationType.FAST_BREAK:       1.18,   # Clear run on goal
        SituationType.CORNER:           0.75,
        SituationType.DIRECT_FREEKICK:  0.85,
        SituationType.CROSSED_FREEKICK: 0.80,
        SituationType.PENALTY:          0.79,   # Fixed, overrides zone
        SituationType.THROW_IN:         0.60,
    }

    # Pressure multiplier (defender breathing down neck)
    UNDER_PRESSURE_MULT = 0.65

    @classmethod
    def calculate_geometric(
        cls,
        x: float,
        y: float,
        body_part: str,
        situation: SituationType,
        under_pressure: bool = False,
        attacks_right: bool = True,
    ) -> float:
        """
        Continuous geometric xG model based on exact (x, y) coordinates.
        Used by tests and as an alternative API to the zone-based calculate().
        
        Model:
            - Base xG from distance-to-goal (exponential decay)
            - Angle multiplier (central shots worth more)
            - Body part modifier
            - Situation modifier
            - Pressure penalty
        """
        if situation == SituationType.PENALTY:
            return 0.79

        # Distance from attacking goal
        goal_x = 105.0 if attacks_right else 0.0
        dx = goal_x - x
        dy = 34.0 - y
        dist = (dx ** 2 + dy ** 2) ** 0.5

        # Base xG decays exponentially with distance
        # At 1m: ~0.70, at 10m: ~0.35, at 30m: ~0.05, at 60m: ~0.005
        base = 0.75 * (0.87 ** dist)

        # Angle factor: central (y≈34) is best, wider angles reduce xG
        angle_factor = max(0.15, 1.0 - (abs(dy) / 68.0) * 0.7)

        base *= angle_factor
        base *= cls.BODY_PART_MULT.get(body_part, 0.80)
        base *= cls.SITUATION_MULT.get(situation, 1.00)

        if under_pressure:
            base *= cls.UNDER_PRESSURE_MULT

        return round(min(0.99, base), 4)

    @classmethod
    def calculate(
        cls,
        zone: str,
        body_part: str,
        situation: SituationType,
        under_pressure: bool = False,
        is_big_chance: bool = False,
        first_time_shot: bool = False,
        shot_x: float | None = None,
        shot_y: float | None = None,
        attacks_right: bool = True,
    ) -> float:
        """
        Calculate xG for a shot.
        
        The zone xG values (e.g. 0.59 for six_yard_box) are the AVERAGE
        conversion rate for ALL shots from that zone — they already include
        fast breaks, big chances, first-time shots, etc. Multiplying by
        situation, big_chance, or first_time again would be double-counting
        the quality of the position.
        
        Only body_part and pressure are applied as modifiers because they
        genuinely change the physics of the shot (a header IS harder than
        a foot from the same spot; a shot under pressure IS harder).
        
        If shot_x and shot_y are provided, applies a geometric angle penalty:
        shots from extreme angles (very wide relative to distance from goal)
        have reduced xG because the goal opening is barely visible.
        The goal is 7.32m wide (y=30.34 to y=37.66).
        """

        if situation == SituationType.PENALTY:
            return 0.79

        base = cls.ZONE_XG.get(zone, 0.05)
        base *= cls.BODY_PART_MULT.get(body_part, 0.80)

        if under_pressure:
            base *= cls.UNDER_PRESSURE_MULT

        # ── GOAL POST GEOMETRIC AWARENESS ──────────────────────────
        # The goal is 7.32m wide (y=30.34 to y=37.66).
        # A shot from wide y-values at close range has a very narrow
        # angle to the goal — the posts block most of the opening.
        goal_x = 105.0 if attacks_right else 0.0
        if shot_x is not None and shot_y is not None:
            dx = max(1.0, abs(goal_x - shot_x))
            dy = abs(shot_y - 34.0)
            
            if dy > 0 and dx > 0:
                # Angle from center: how far off-center is the shot?
                angle_from_center = np.arctan2(dy, dx)
                # The effective goal width visible = 7.32 * cos(angle_from_center)
                # At 0° (dead center): full 7.32m visible
                # At 45°: only ~5.2m visible
                # At 60°: only ~3.7m visible
                # At 75°: only ~1.9m visible
                angle_penalty = max(0.15, np.cos(angle_from_center))
                base *= angle_penalty

        # Small random variation (±5% instead of ±10%) — keeps xG realistic
        noise = random.uniform(0.95, 1.05)
        base *= noise

        return round(min(base, 0.99), 4)

    @classmethod
    def does_goal_happen(cls, xg: float, shooter_quality: float = 1.0) -> bool:
        """
        Roll against xG to determine if goal is scored.
        shooter_quality: 1.0 = average, 1.2 = elite finisher, 0.8 = poor finisher
        """
        effective_xg = min(0.99, xg * shooter_quality)
        return random.random() < effective_xg


# ─────────────────────────────────────────────
# POSSESSION ENGINE — Who has the ball and for how long
# ─────────────────────────────────────────────

class PossessionEngine:
    """
    Models possession sequences realistically.

    A possession sequence is a chain of events from winning
    the ball to either losing it or creating a chance/shot.
    """

    @staticmethod
    def calculate_possession_split(
        home_profile: TeamProfile,
        away_profile: TeamProfile,
        state: MatchState,
        home_team: str,
    ) -> Tuple[float, float]:
        """
        Calculate current possession probability for each team.
        This is DYNAMIC — it changes with game state and momentum.
        """
        home_base = home_profile.possession_target / 100.0
        away_base = away_profile.possession_target / 100.0

        # Possession sides get a small style premium so elite tiki-taka
        # truly feels dominant against other possession-oriented teams.
        if getattr(home_profile, 'style', None) == TeamStyle.TIKI_TAKA:
            home_base *= 1.08
        elif getattr(home_profile, 'style', None) in (
                TeamStyle.STRUCTURED_POSSESSION, TeamStyle.VERTICAL_TIKI_TAKA):
            home_base *= 1.03
        if getattr(away_profile, 'style', None) == TeamStyle.TIKI_TAKA:
            away_base *= 1.08
        elif getattr(away_profile, 'style', None) in (
                TeamStyle.STRUCTURED_POSSESSION, TeamStyle.VERTICAL_TIKI_TAKA):
            away_base *= 1.03

        # Normalize (they don't sum to 1.0 since both want >50%)
        total = home_base + away_base
        home_base /= total
        away_base /= total

        # Game state modifier
        gd = state.goal_difference
        late_game = state.minute >= 70

        if gd >= 2 and late_game:
            # Home team killing time = more home possession
            home_base = min(0.75, home_base * 1.15)
        elif gd <= -2 and late_game:
            # Away team chasing = more home possession (home defending)
            home_base = min(0.80, home_base * 1.20)
        elif gd == -1 and late_game:
            # Away trailing late = they push, get more ball
            home_base = max(0.30, home_base * 0.90)

        # Momentum modifier (max ±8% swing)
        momentum_effect = state.momentum / 100.0 * 0.08
        home_base = max(0.20, min(0.80, home_base + momentum_effect))
        away_base = 1.0 - home_base

        # Red card penalty (10-man team gets less possession)
        if state.home_red_cards > 0:
            reduction = 0.07 * state.home_red_cards
            home_base = max(0.20, home_base - reduction)
            away_base = 1.0 - home_base
        if state.away_red_cards > 0:
            reduction = 0.07 * state.away_red_cards
            away_base = max(0.20, away_base - reduction)
            home_base = 1.0 - away_base

        return round(home_base, 4), round(away_base, 4)

    @staticmethod
    def sequence_length(team_profile: TeamProfile | EffectiveTactics, state: MatchState) -> int:
        """
        How many passes in a typical possession sequence for this team.
        Tiki-taka teams have long sequences, route one teams have short ones.
        Accepts both TeamProfile and EffectiveTactics (which has a .style attr).
        """
        # Resolve style from profile — TeamProfile has .style, EffectiveTactics
        # stores it as an attribute if created from adjust().
        style = getattr(team_profile, "style", None)
        if style is None:
            # Fallback for EffectiveTactics: use possession_target as proxy
            return random.randint(3, 8)
        # Checkpoint 23: ranges for possession-capable styles sit slightly
        # higher than they historically did. That is only realistic NOW —
        # the tempo-circulation directive lets a long sequence hover in the
        # middle third instead of marching box-to-box, so a 14-pass spell
        # looks like City circulating rather than a conveyor belt to a shot.
        # Direct/defensive styles are untouched: their short sequences ARE
        # their identity.
        base_length = {
            TeamStyle.TIKI_TAKA:           random.randint(9, 20),
            TeamStyle.STRUCTURED_POSSESSION: random.randint(7, 16),
            TeamStyle.VERTICAL_TIKI_TAKA:  random.randint(6, 13),
            TeamStyle.ATTACKING:           random.randint(5, 12),
            TeamStyle.BALANCED:            random.randint(4, 10),
            TeamStyle.GEGENPRESSING:       random.randint(4, 8),
            TeamStyle.FLUID_COUNTER:       random.randint(3, 7),
            TeamStyle.DEFENSIVE:           random.randint(2, 6),
            TeamStyle.WING_PLAY:           random.randint(4, 10),
            TeamStyle.ULTRA_ATTACKING:     random.randint(5, 11),
            TeamStyle.ROUTE_ONE:           random.randint(1, 4),
            TeamStyle.PARK_THE_BUS:        random.randint(1, 4),
            TeamStyle.ULTRA_DEFENSIVE:     random.randint(1, 3),
        }.get(team_profile.style, random.randint(4, 10))

        return base_length


# ─────────────────────────────────────────────
# THE MATCH ENGINE — The simulation core
# ─────────────────────────────────────────────

class MatchEngine:
    """
    The heart of PLOFA 26/27.

    Simulates a football match minute-by-minute, event-by-event.
    Everything emerges from probabilities that react to game state.

    Usage:
        engine = MatchEngine(config, home_profile, away_profile)
        engine.set_squad("Home FC", starters, subs)
        engine.set_squad("Away FC", starters, subs)
        result = engine.simulate()
        result.export_to_excel("matchday_1.xlsx")
    """

    def __init__(
        self,
        config: MatchConfig,
        home_profile: TeamProfile,
        away_profile: TeamProfile,
    ):
        self.config = config
        self.home_profile = home_profile
        self.away_profile = away_profile

        self.state = MatchState(
            possession_team=config.home_team  # Home team kicks off
        )

        self.timeline: List[MatchEvent] = []   # The complete match history
        self.squads: Dict[str, List] = {}      # {team_name: [Player objects]}
        self.active_players: Dict[str, List] = {}  # Currently on pitch

        # Accumulators (filled during simulation, read during export)
        self.event_counts: Dict[str, Dict] = {}
        self.goals: List[MatchEvent] = []
        self.cards: List[MatchEvent] = []
        self.subs: List[MatchEvent] = []

        # Squad manager — wired in via set_stamina_controller()
        self.sub_controller = None   # SubstitutionController or None

        # Set to True to silence per-match narrative prints (goal / card /
        # sub lines). Keeps seeded multi-sim verification logs parseable.
        self.quiet = False

        # Checkpoint 5 — Position Engine: persistent per-player spatial state,
        # causal drift, zone-grounded selection. One instance per match.
        self.position_engine = PositionEngine()

        # Checkpoint 9 — Threat Engine: live per-team danger level driven by
        # ball↔defended-goal geometry. Wired into _absorb_chain (live danger),
        # _simulate_minute (danger-scaled defensive contests) and _run_minute
        # (defensive_block coordination). Home defends x=0, away defends x=105.
        from threat_engine import ThreatEngine
        self.threat = ThreatEngine(config.home_team, config.away_team)

        # Opta telemetry — per-minute spatial + momentum logs filled during
        # _run_minute and consumed by the post-match analytics module.
        self.position_log: List[Dict] = []
        self.momentum_log: List[Dict] = []

    def _update_block_shapes(self, minute: int):
        """
        Checkpoint 29 — refresh both teams' defensive BlockShapes from their
        live spatial states. Called every minute before sequences run so pass
        selection can navigate around or through the opponent block.
        Home defends x=0 (deep = low x -> attacks_right=False),
        away defends x=105 (deep = high x -> attacks_right=True).
        """
        for team, attacks_right, attr in (
            (self.config.home_team, False, "home_block"),
            (self.config.away_team, True, "away_block"),
        ):
            names = [n for n in self.position_engine.team_rosters.get(team, [])
                     if n in self.position_engine.states]
            positions = {n: self.position_engine.get_position(n) for n in names}
            pos_map = {n: self.position_engine.states[n].position for n in names}
            shape = BlockDetector.detect(
                positions, pos_map, attacks_right=attacks_right, minute=minute
            )
            setattr(self.state, attr, shape)

    def set_squad(self, team_name: str, starters: list, substitutes: list = None):
        """Register a squad for the match."""
        if len(starters) != 11:
            raise ValueError(
                f"Team '{team_name}' must have exactly 11 starters, "
                f"got {len(starters)}."
            )
        self.squads[team_name] = {
            'starters': starters,
            'substitutes': substitutes or [],
        }
        self.active_players[team_name] = list(starters)

        # Give every starter a home position + live spatial state,
        # anchored to this team's actual tactical profile.
        profile = self.home_profile if team_name == self.config.home_team else self.away_profile
        attacks_right = (team_name == self.config.home_team)
        self.position_engine.initialize_team(team_name, starters, profile, attacks_right=attacks_right)

    def set_stamina_controller(self, controller):
        """
        Wire in a SubstitutionController from squad_manager.py.
        Call this after set_squad() for both teams.
        All starters are registered with their starting stamina.
        """
        self.sub_controller = controller
        # Register all starters
        for team, players in self.active_players.items():
            for p in players:
                starting = 100.0
                if hasattr(p, 'dna') and hasattr(p.dna, '_starting_stamina'):
                    starting = p.dna._starting_stamina
                controller.register_player(p, starting_stamina=starting)

    def simulate(self) -> "MatchResult":
        """
        Run the full match simulation.
        Returns a MatchResult containing all events and derived stats.
        Substitution logic and stamina tracking run in parallel.
        """
        self._initialize_simulation()

        def _run_minute(minute: int):
            self.state.minute = minute
            self.state.phase  = PhaseEngine.get_phase(minute)

            # ── KICKOFF TRIGGERS ────────────────────────────────────────
            if minute == 1 and not self.state.pending_kickoff_for:
                self.state.pending_kickoff_for = self.state.first_half_kickoff_team
            if minute == 46 and self.state.pending_second_half_kickoff:
                second_half_team = (
                    self.config.home_team if random.random() < 0.5
                    else self.config.away_team
                )
                self.state.pending_kickoff_for = second_half_team
                self.state.pending_second_half_kickoff = False

            # ── SUBSTITUTION CHECK (before the minute plays out) ──
            if self.sub_controller is not None:
                gd_home = self.state.home_goals - self.state.away_goals
                gd_away = -gd_home
                subs = self.sub_controller.process_minute(
                    minute, self.active_players, gd_home, gd_away
                )
                for sub in subs:
                    self._execute_substitution(sub, minute)

            # ── PER-MINUTE BASELINE STAMINA DRAIN (ALL ACTIVE) ────
            # Every player on the pitch loses stamina continuously
            # regardless of whether they appear in a discrete event.
            # This models the constant running, positioning and effort
            # that doesn't generate a logged event.
            if self.sub_controller is not None:
                for team, players in self.active_players.items():
                    team_style = (
                        self.home_profile.style.value
                        if team == self.config.home_team
                        else self.away_profile.style.value
                    )
                    intensity = (
                        self.home_profile.intensity.value
                        if team == self.config.home_team
                        else self.away_profile.intensity.value
                    )
                    # Intensity multiplier on baseline
                    intensity_mult = {
                        "low": 0.82, "medium": 1.00,
                        "high": 1.18, "very_high": 1.35
                    }.get(intensity, 1.0)

                    for player in players:
                        name = getattr(player, "name", "")
                        if not getattr(player, "_subbed_off", False):
                            state = self.sub_controller.stamina.get(name)
                            if state and not state.is_injured:
                                # Bug fix: intensity used to be applied AFTER
                                # the fact by reading back the player's
                                # CUMULATIVE "standing" drain-to-date and
                                # subtracting a fraction of that whole total,
                                # every minute — a compounding loop (bigger
                                # cumulative total -> bigger top-up -> even
                                # bigger cumulative total next minute) that
                                # produced >1000% "Total Drained" figures for
                                # high-intensity teams. Now intensity_mult is
                                # folded directly into the single drain call
                                # for THIS minute's marginal cost only.
                                state.drain_baseline(team_style, intensity_mult=intensity_mult)
                                state.update_performance_mult()

            # ── CHECKPOINT 29: OPPONENT BLOCK SHAPES ────────────────
            # Refresh both teams' defensive BlockShapes from the live
            # spatial states (as of the end of the previous minute) so
            # pass selection this minute can navigate the block.
            self._update_block_shapes(minute)
            # Feed the same shapes to the drift engine — the in-possession
            # team's CAMs/CMs/CFs occupy the block's half-space channels
            # (HalfSpaceMagnet) while pass selection orbits them.
            self.position_engine.set_block_context(
                self.state.home_block, self.state.away_block
            )

            # ── SIMULATE MINUTE ────────────────────────────────────
            self._simulate_minute(minute, TeamStyle)

            # ── POSITION ENGINE: CAUSAL DRIFT (Checkpoint 5/6.1) ────
            # Every player not touched THIS minute drifts back toward
            # their formation-anchored home position. Prevents "sticky"
            # displacement (e.g. a striker staying camped in his own
            # third indefinitely after one deep involvement).
            #
            # Checkpoint 6.1 fix: this now runs AFTER _simulate_minute,
            # using the ACTUAL sequence tally from the minute just played
            # (self._minute_home_seq / _minute_away_seq) rather than a
            # snapshot of state.possession_team taken BEFORE this minute's
            # sequences ran. Previously a team's attacking/defensive shape
            # was decided from whoever happened to hold the ball at the
            # very end of the PREVIOUS minute — up to a full minute stale,
            # despite possession flipping 2-4 times inside _simulate_minute
            # itself. This is what let a fullback's shape target stay
            # "advanced" for a minute or more after his team had actually
            # lost the ball. Ties (or a minute with zero sequences, e.g.
            # a single consumed corner) fall back to the current
            # possession_team snapshot rather than guessing.
            total_seq = self._minute_home_seq + self._minute_away_seq
            if total_seq > 0:
                home_has_ball = self._minute_home_seq >= self._minute_away_seq
            else:
                home_has_ball = self.state.possession_team == self.config.home_team

            gd_home_now = self.state.home_goals - self.state.away_goals
            home_opponents = self.active_players.get(self.config.away_team, [])
            away_opponents = self.active_players.get(self.config.home_team, [])

            # ── REAL MOVEMENT CAPTURE: snapshot before the off-ball phase ──
            # drift_minute + defensive_block + attacking_crash together are
            # this minute's entire off-ball movement step for each team.
            # Snapshotting before and diffing after (below) captures the
            # true net distance every player moved this minute from all
            # three sources combined, without instrumenting each one
            # individually.
            _home_pos_before = self.position_engine.snapshot_positions(self.config.home_team)
            _away_pos_before = self.position_engine.snapshot_positions(self.config.away_team)

            # Danger per team is needed by drift_minute (ball-side squeeze)
            # and defensive_block — compute once, reuse below.
            _danger = {
                self.config.home_team: self.threat.danger_at(self.config.home_team),
                self.config.away_team: self.threat.danger_at(self.config.away_team),
            }

            self.position_engine.drift_minute(
                self.config.home_team, self.home_profile,
                self.state.phase, game_state_gd=gd_home_now, minute=minute,
                in_possession=home_has_ball,
                ball_x=self.state.last_ball_x, ball_y=self.state.last_ball_y,
                opponent_players=home_opponents,
                danger_level=_danger[self.config.home_team],
            )
            self.position_engine.drift_minute(
                self.config.away_team, self.away_profile,
                self.state.phase, game_state_gd=-gd_home_now, minute=minute,
                in_possession=not home_has_ball,
                ball_x=self.state.last_ball_x, ball_y=self.state.last_ball_y,
                opponent_players=away_opponents,
                danger_level=_danger[self.config.away_team],
            )

            # ── CHECKPOINT 9: COORDINATED DEFENSIVE BLOCK ────────────
            # When a team is OUT of possession AND the ball is alive near
            # their own goalpost (danger ≥ 25), their back four + keeper
            # coordinate into a tight goal-side block: CBs narrow toward
            # the ball's y, fullbacks tuck in, and the keeper guards the
            # goal line. This is pure spatial intent on top of the shape
            # drift, so it only bites when the match state genuinely needs
            # it — otherwise the baseline formation stands untouched.
            for team, has_ball in (
                (self.config.home_team, home_has_ball),
                (self.config.away_team, not home_has_ball),
            ):
                danger = _danger[team]
                if danger < 25 or has_ball:
                    continue
                bx, by = self.state.last_ball_x, self.state.last_ball_y
                own_goal_x = self.threat.own_goal_x(team)
                pull = min(1.0, (danger - 25) / 65.0)   # deeper block as danger grows
                self.position_engine.defensive_block(
                    team, bx, by, own_goal_x, danger,
                    minute=minute, pull_strength=pull,
                )

            # ── CHECKPOINT 11: ATTACKING BOX CRASH ────────────────
            # When a cross was detected this minute, the attacking team's
            # off-ball forwards crash the box (near-side to the penalty spot,
            # far-side to the back post) instead of short-support drifting.
            # Runs AFTER the defensive block so both units converge on the
            # delivery — exactly the six-on-six box scramble a real whipped
            # cross produces.
            if self.state.cross_active and self.state.cross_team:
                self.position_engine.attacking_crash(
                    self.state.cross_team,
                    self.state.cross_x, self.state.cross_y,
                    self.state.cross_attacks_right,
                    minute=minute,
                    intensity=0.6,
                    carrier_name=self.state.cross_player,
                )

            # Close the off-ball movement snapshot: real net distance for
            # everything drift_minute/defensive_block/attacking_crash just
            # did this minute, folded into minute_drift_distance per player.
            self.position_engine.accumulate_drift_from_snapshot(
                self.config.home_team, _home_pos_before
            )
            self.position_engine.accumulate_drift_from_snapshot(
                self.config.away_team, _away_pos_before
            )

            # Checkpoint 26 — refresh the velocity-aware pitch-control
            # cache from the just-updated drift velocities. Consumers
            # (e.g. winger half-space openness) read it via
            # position_engine.pitch_control_result/field.
            self.position_engine.update_pitch_control(
                self.config.home_team, self.config.away_team, minute=minute,
            )

            # ── OPTA TELEMETRY LOGGING ─────────────────────────────
            # Per-minute snapshot of every player's live spatial state plus
            # the scoreline. Feeds the post-match analytics module (distance
            # covered, line positions, game-state minutes, momentum series).
            # Each player's row now also carries this minute's REAL movement
            # (touch + drift distance, touch count, peak touch jump) —
            # genuinely derived from the simulation, not an authored
            # baseline — which opta_analytics.py uses as its primary
            # distance/sprint signal.
            frame = {
                "minute": minute,
                "home": [],
                "away": [],
                "home_goals": self.state.home_goals,
                "away_goals": self.state.away_goals,
                "possession_team": self.state.possession_team,
                "phase": self.state.phase.value,
            }
            for team in (self.config.home_team, self.config.away_team):
                side = "home" if team == self.config.home_team else "away"
                for r in self.position_engine.snapshot(team):
                    activity = self.position_engine.pop_minute_activity(r["player"])
                    frame[side].append({
                        "player": r["player"],
                        "position": r["position"],
                        "x": r["current_x"],
                        "y": r["current_y"],
                        "distance_touch": activity["distance_touch"],
                        "distance_drift": activity["distance_drift"],
                        "distance_total": activity["distance_total"],
                        "touches": activity["touches"],
                        "peak_touch_jump": activity["peak_touch_jump"],
                        "physics_distance_m": activity.get("physics_distance_m", 0.0),
                        "physics_sprint_count": activity.get("physics_sprint_count", 0.0),
                        "physics_high_speed_sprint_count": activity.get("physics_high_speed_sprint_count", 0.0),
                        "physics_top_speed_mps": activity.get("physics_top_speed_mps", 0.0),
                    })
            self.position_log.append(frame)

            # ── MOMENTUM DECAY ─────────────────────────────────────
            self.state.momentum = MomentumEngine.natural_decay(
                self.state, self.config.home_team, self.home_profile
            )
            self.momentum_log.append({
                "minute": minute,
                "momentum": round(self.state.momentum, 2),
                "home_goals": self.state.home_goals,
                "away_goals": self.state.away_goals,
            })

        # First 90 minutes
        for minute in range(1, 91):
            _run_minute(minute)

            # ── HALF-TIME RECOVERY (minute 45) ──────────────────────
            # In real football the 15-minute half-time break does not fully
            # reset players, but it does provide acute recovery (~18% of
            # max stamina). This is applied here so the second half starts
            # with visibly fresher players, matching real match dynamics.
            if minute == 45 and self.sub_controller is not None:
                for team_players in self.active_players.values():
                    for p in team_players:
                        name = getattr(p, "name", "")
                        state = self.sub_controller.stamina.get(name)
                        if state and not state.is_injured:
                            state.half_time_recovery(recovery_pct=0.18)
                # Realism: players retreat to their own halves during the
                # break so the second half starts from clean shapes.
                self._reset_positions_to_halves()

        # Added time (decided after full 90)
        added = self._decide_added_time()
        for minute in range(91, 91 + added):
            self.state.phase = MatchPhase.ADDED_TIME
            _run_minute(minute)

        # Final whistle — set minutes for everyone still on pitch
        total_mins = 90 + added
        for team_players in self.active_players.values():
            for p in team_players:
                if (hasattr(p, "dna") and p.dna.minutes_played == 0
                        and getattr(p, "sub_in_minute", None) is None):
                    p.dna.minutes_played = total_mins

        # Also finalise substitutes who came on. Bench players carry a
        # pre-planned sub_in_minute from the roster (the "sub ~65'" tag),
        # so only credit minutes to those who actually entered the pitch —
        # otherwise unused bench players get phantom minutes and all-zero
        # statlines.
        for team_squad in self.squads.values():
            for p in team_squad.get("substitutes", []):
                if (hasattr(p, "dna") and p.dna.minutes_played == 0
                        and getattr(p, "_entered_pitch", False)
                        and getattr(p, "sub_in_minute", None) is not None):
                    p.dna.minutes_played = total_mins - p.sub_in_minute

        return MatchResult(
            config=self.config,
            state=self.state,
            timeline=self.timeline,
            goals=self.goals,
            cards=self.cards,
            subs=self.subs,
            squads=self.squads,
            threat=self.threat,
            position_log=self.position_log,
            momentum_log=self.momentum_log,
        )

    def _execute_substitution(self, sub: dict, minute: int):
        """
        Apply a substitution decided by SubstitutionController.
        Swaps player_off out of active_players, player_on in.
        Emits substitution event to timeline.
        """
        from event_chain import EventType as ET
        team      = sub["team"]
        name_off  = sub["player_off"]
        name_on   = sub["player_on"]

        # Find PlayerProfile objects
        player_off_obj = next(
            (p for p in self.active_players.get(team, [])
             if getattr(p, "name", "") == name_off), None
        )
        player_on_obj = next(
            (p for p in self.squads.get(team, {}).get("substitutes", [])
             if getattr(p, "name", "") == name_on), None
        )

        if player_off_obj is None or player_on_obj is None:
            return

        # Swap in active_players
        active = self.active_players[team]
        idx = next((i for i, p in enumerate(active)
                    if getattr(p, "name", "") == name_off), None)
        if idx is not None:
            active[idx] = player_on_obj
            # Mark actual pitch entry — the roster's pre-planned
            # sub_in_minute alone does NOT mean the player came on,
            # so final-whistle minutes must key off this flag.
            player_on_obj._entered_pitch = True

        # Set minutes
        if hasattr(player_off_obj, "dna"):
            player_off_obj.dna.minutes_played = minute
        if hasattr(player_on_obj, "dna") and player_on_obj.dna.minutes_played == 0:
            player_on_obj.dna.minutes_played = 0  # will be set at final whistle
            player_on_obj.sub_in_minute = minute

        # Checkpoint 5: give the incoming sub a fresh home position
        # anchored to their actual role, rather than inheriting nothing
        # (which would leave them with no spatial state at all).
        team_profile = self.home_profile if team == self.config.home_team else self.away_profile
        self.position_engine.register_substitute(team, player_on_obj, team_profile)

        # Update state sub counters
        if team == self.config.home_team:
            self.state.home_subs_made += 1
        else:
            self.state.away_subs_made += 1

        # Emit substitution event
        sub_event = MatchEvent(
            minute=minute,
            second=0,
            event_type=EventType.SUBSTITUTION,
            team=team,
            player=name_off,
            secondary_player=name_on,
            phase=self.state.phase,
            game_state=self.state.game_state,
            metadata={
                "reason":    sub.get("reason", "tactical"),
                "freshness": sub.get("freshness", 1.0),
                "stamina_at_exit": sub.get("stamina_at_exit", 0),
            }
        )
        self.timeline.append(sub_event)
        self.subs.append(sub_event)

        reason_icon = {
            "tactical": "🔄", "stamina": "😮‍💨",
            "injury": "🤕", "game_state": "♟️",
        }.get(sub.get("reason", "tactical"), "🔄")
        if not self.quiet:
            print(f"  {reason_icon} SUB {minute}' — {name_off} → {name_on} ({team}) "
              f"[{sub.get('reason','tactical')}]")

    def _initialize_simulation(self):
        """Set up initial state before the whistle."""
        if random.random() < 0.5:
            self.state.possession_team = self.config.home_team
            self.state.first_half_kickoff_team = self.config.home_team
        else:
            self.state.possession_team = self.config.away_team
            self.state.first_half_kickoff_team = self.config.away_team
        self.state.pending_second_half_kickoff = True

        # Apply home advantage to starting momentum
        home_crowd_factor = 5.0 if not self.config.is_derby else 8.0
        self.state.momentum = home_crowd_factor

    def _reset_positions_to_halves(self):
        """Reset positions for a kickoff restart.
        
        Physics: the KICKING team's attackers step up to the centre circle
        because they are the ones who initiate play. The defending team
        holds its defensive shape. A hard snap is correct here because
        the whistle gives everyone ~10 seconds to station themselves.
        """
        kickoff_team = self.state.pending_kickoff_for or self.state.possession_team
        defending_team = (
            self.config.away_team if kickoff_team == self.config.home_team
            else self.config.home_team
        )
        for team_name, team_players in self.active_players.items():
            for p in team_players:
                name = getattr(p, "name", "")
                state = self.position_engine.states.get(name)
                if not state:
                    continue
                if team_name != kickoff_team:
                    # Defending team: snap to home shape, but clamp to own half
                    # so forwards don't start the half camped in the opponent's
                    # half at kickoff.
                    state.current_x = state.home_x
                    state.current_y = state.home_y
                    own_goal_x = 0.0 if team_name == self.config.home_team else 105.0
                    if (team_name == self.config.home_team and state.current_x > 52.5) or (
                        team_name == self.config.away_team and state.current_x < 52.5
                    ):
                        state.current_x = own_goal_x + (52.5 - own_goal_x) * 0.5
                    continue
                # Kicking team: push attackers toward the centre circle
                pos = getattr(p, "position", "")
                if pos in ("ST", "LW", "RW", "CAM"):
                    # Attackers plant themselves just behind the centre spot
                    # so they can receive the restart and carry it forward.
                    state.current_x = 50.0 + random.uniform(-3.0, 3.0)
                    state.current_y = 34.0 + random.uniform(-8.0, 8.0)
                elif pos in ("CM", "CDM"):
                    # Midfielders hold the middle third, ready to receive
                    # the backward pass that every real kickoff starts with.
                    state.current_x = 42.0 + random.uniform(-4.0, 4.0)
                    state.current_y = 34.0 + random.uniform(-10.0, 10.0)
                else:
                    # Defenders and fullbacks stay deep — the safety valve.
                    state.current_x = state.home_x
                    state.current_y = state.home_y

    def _pick_kickoff_taker(self, team_name: str) -> str:
        """Pick a realistic kickoff taker from the active squad.
        
        Real football: the player who steps up is almost always an
        attacker or midfielder (CAM, CM, LW, RW, CDM) — the same
        players who naturally stand closest to the centre circle
        after the teams reset. CBs and fullbacks do not take kickoffs.
        """
        players = self.active_players.get(team_name, [])
        if not players:
            return "Kickoff Taker"

        preferred = ["CAM", "CM", "LW", "RW", "CDM"]
        candidates = [
            p for p in players
            if getattr(p, "position", "") in preferred
        ]
        if not candidates:
            candidates = list(players)

        def dist_to_centre(p):
            s = self.position_engine.states.get(getattr(p, "name", ""))
            if s:
                return ((s.current_x - 52.5) ** 2 + (s.current_y - 34.0) ** 2) ** 0.5
            return 999.0

        candidates.sort(key=dist_to_centre)
        return candidates[0].name

    def _decide_added_time(self) -> int:
        """Realistic added time based on match events."""
        # Base: 2-6 minutes
        # More goals, cards, and subs = more added time
        base = random.randint(2, 6)
        goal_bonus = len(self.goals) * 0.5
        card_bonus = len(self.cards) * 0.3
        sub_bonus = (self.state.home_subs_made + self.state.away_subs_made) * 0.4
        added = int(base + goal_bonus + card_bonus + sub_bonus)
        self.state.added_time = min(added, 12)  # Cap at 12
        return self.state.added_time

    def _simulate_minute(self, minute: int, style: TeamStyle):
        """
        Simulate a single minute of football.

        Real football generates 22-28 events per minute (StatsBomb standard).
        This method runs 2-4 possession sequences per minute, each generating
        8-15 events through the full causal chain:
            carry → pass → ball_receipt → pressure → carry → pass...

        Target: 1,500-3,400 total events per match.
        """
        from event_chain import ChainDispatcher

        phase     = self.state.phase
        home_team = self.config.home_team
        away_team = self.config.away_team

        # Checkpoint 6.1 — possession-share tracking for THIS minute.
        # Used by _run_minute (after this method returns) to decide each
        # team's attacking/defensive SHAPE for the drift that precedes the
        # next minute. Previously that decision used self.state.possession_team
        # captured BEFORE this method ran — i.e. whoever last had the ball at
        # the END of the PREVIOUS minute — even though possession flips 2-4
        # times inside this very method. Counting actual sequences here closes
        # that staleness gap.
        self._minute_home_seq = 0
        self._minute_away_seq = 0

        # Checkpoint 11 — a cross situation only shapes the current minute.
        self.state.cross_active = False
        self.state.cross_corner_done = False

        # ── POSSESSION SPLIT ──────────────────────────────────────────
        home_poss, away_poss = PossessionEngine.calculate_possession_split(
            self.home_profile, self.away_profile, self.state, home_team
        )

        # ── SEQUENCES PER MINUTE ──────────────────────────────────────
        # Real football: possession changes hands multiple times per minute.
        # High-tempo styles (gegenpressing, ultra-attacking) generate more.
        # Low-tempo styles (park-the-bus) generate fewer.
        # Checkpoint 23: proactive styles tick one extra sequence per minute.
        # Per-sequence shot probability is divided by n_sequences below, so
        # this raises PASS VOLUME (real matches: 450-700 passes/team, hub
        # midfielders 60-90 passes) without inflating shots or scorelines.
        # Combined with tempo circulation, the extra sequences are spent
        # weaving the middle-third web rather than producing more chances.
        base_sequences = {
            "ultra_attacking":      random.randint(4, 5),
            "attacking":            random.randint(4, 5),
            "gegenpressing":        random.randint(4, 5),
            "tiki_taka":            random.randint(4, 5),
            "vertical_tiki_taka":   random.randint(4, 5),
            "balanced":             random.randint(3, 4),
            "structured_possession": random.randint(3, 4),
            "fluid_counter":        random.randint(2, 4),
            "wing_play":            random.randint(3, 4),
            "defensive":            random.randint(2, 3),
            "route_one":            random.randint(2, 3),
            "park_the_bus":         random.randint(1, 3),
            "ultra_defensive":      random.randint(1, 2),
        }
        # Use the more active team's style to determine match tempo
        home_style = self.home_profile.style.value
        away_style = self.away_profile.style.value
        tempo_style = home_style if home_poss >= away_poss else away_style
        n_sequences = base_sequences.get(tempo_style, random.randint(2, 4))

        # Added time is more frantic
        if self.state.phase.value == "added_time":
            n_sequences = min(n_sequences + 1, 5)

        # ── SIMULATE EACH SEQUENCE ────────────────────────────────────
        for seq_idx in range(n_sequences):
            # ── KICKOFF (Start of half / After Goal) ──────────────────
            if self.state.pending_kickoff_for:
                kickoff_team = self.state.pending_kickoff_for
                self.state.pending_kickoff_for = ""
                self.state.possession_team = kickoff_team
                self.state.last_ball_x = 52.5
                self.state.last_ball_y = 34.0

                # Realism: snap both teams back to their formation halves
                # so the restart begins from a clean shape, not a scrambled
                # goal-sequence tail.
                self._reset_positions_to_halves()

                # Checkpoint 9 — ball back at centre circle: both teams'
                # danger returns to the low kickoff baseline.
                self.threat.on_kickoff(minute)
                
                taker = self._pick_kickoff_taker(kickoff_team)
                
                self.timeline.append(MatchEvent(
                    minute=minute, second=0,
                    event_type=EventType.KICKOFF,
                    team=kickoff_team,
                    player=taker,
                    location_x=52.5, location_y=34.0,
                    phase=self.state.phase, game_state=self.state.game_state
                ))
                
                # After kickoff, the sequence proceeds with kickoff_team in possession
                attacking_team = kickoff_team
                defending_team = home_team if kickoff_team == away_team else away_team
            else:
                pass
                
            # ── CHECKPOINT 6: CONSUME A PENDING CORNER FIRST ──────────
            # If a chain earlier this minute reported corner_won, that
            # team is OWED this sequence as an actual corner — not a
            # fresh random roll of possession/situation. This is what
            # makes corners a genuine consequence of a blocked shot or
            # defensive clearance rather than an independently-drawn
            # situation that merely happened to coincide.
            # Fix: anchor corner to state.last_ball_x/y so there's no teleport.
            pending_corner_home = self.state.pending_corners_home > 0
            pending_corner_away = self.state.pending_corners_away > 0
            if pending_corner_home or pending_corner_away:
                if pending_corner_home:
                    corner_team = home_team
                    self.state.pending_corners_home -= 1
                else:
                    corner_team = away_team
                    self.state.pending_corners_away -= 1
                corner_opponent = away_team if corner_team == home_team else home_team
                self.state.possession_team = corner_team
                if corner_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                # Anchor corner to last ball position — a corner doesn't
                # teleport the ball; it was won from a blocked shot/clearance
                # right there, and the set piece delivery is from that context.
                self.state.last_ball_x = min(105.0, max(83.0, self.state.last_ball_x))
                self.state.last_ball_y = max(5.0, min(63.0, self.state.last_ball_y))
                sp_result = ChainDispatcher.set_piece(
                    minute, corner_team, corner_opponent,
                    self.active_players.get(corner_team, []),
                    self.active_players.get(corner_opponent, []),
                    self.state, SituationType.CORNER,
                    position_engine=self.position_engine,
                )
                if self._absorb_chain(sp_result, minute): break
                continue

            # ── CHECKPOINT X: CONSUME A PENDING PENALTY FIRST ──────────
            # A foul the defending team committed inside its own box wins a
            # spot kick for the fouled side. Like won corners, this queues
            # the ACTUAL penalty sequence rather than leaving PENALTY_WON as
            # a dangling event. The fouled team kicks toward the same goal it
            # was attacking when the foul was drawn.
            pending_pen_home = self.state.pending_penalty_home > 0
            pending_pen_away = self.state.pending_penalty_away > 0
            if pending_pen_home or pending_pen_away:
                pen_team = home_team if pending_pen_home else away_team
                if pending_pen_home:
                    self.state.pending_penalty_home -= 1
                else:
                    self.state.pending_penalty_away -= 1
                pen_opponent = away_team if pen_team == home_team else home_team
                self.state.possession_team = pen_team
                if pen_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                # The conceding (defending/fouling) side is charged with the
                # spot — this is what the per-team CAP reads so a siege can't
                # cascade into a string of penalties against the same defence.
                self.state.penalties_taken[pen_opponent] = self.state.penalties_taken.get(pen_opponent, 0) + 1
                pen_result = ChainDispatcher.set_piece(
                    minute, pen_team, pen_opponent,
                    self.active_players.get(pen_team, []),
                    self.active_players.get(pen_opponent, []),
                    self.state, SituationType.PENALTY,
                    attacks_right=(pen_team == home_team),
                    position_engine=self.position_engine,
                )
                if self._absorb_chain(pen_result, minute): break
                continue

            # ── CHECKPOINT 8: RESTART SEQUENCES ─────────────────────────
            # Goal kick first (direct from shot-end on goal line), then
            # throw-in (wide of the posts, on the sideline).
            if self.state.pending_goal_kick_for:
                gk_team = self.state.pending_goal_kick_for
                self.state.pending_goal_kick_for = ""
                gk_opponent = home_team if gk_team == away_team else away_team
                self.state.possession_team = gk_team
                if gk_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                gk_result = ChainDispatcher.goal_kick(
                    minute, gk_team, gk_opponent,
                    self.active_players.get(gk_team, []),
                    self.active_players.get(gk_opponent, []),
                    self.home_profile if gk_team == home_team else self.away_profile,
                    self.state,
                    position_engine=self.position_engine,
                )
                if self._absorb_chain(gk_result, minute): break
                continue

            if self.state.pending_throw_in_for:
                throw_team = self.state.pending_throw_in_for
                self.state.pending_throw_in_for = ""
                throw_opponent = home_team if throw_team == away_team else away_team
                self.state.possession_team = throw_team
                if throw_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                throw_result = ChainDispatcher.throw_in(
                    minute, throw_team, throw_opponent,
                    self.active_players.get(throw_team, []),
                    self.active_players.get(throw_opponent, []),
                    self.home_profile if throw_team == home_team else self.away_profile,
                    self.state,
                    self.state.pending_restart_x,
                    self.state.pending_restart_y,
                    position_engine=self.position_engine,
                )
                if self._absorb_chain(throw_result, minute): break
                continue

            # Checkpoint 19 — offside free kicks: when a pass in open play
            # is detected as offside, the defending team gets a free kick
            # at the offside location (not a random zone).
            if self.state.pending_offside_fk_for:
                fk_team = self.state.pending_offside_fk_for
                self.state.pending_offside_fk_for = ""
                fk_opponent = home_team if fk_team == away_team else away_team
                self.state.possession_team = fk_team
                if fk_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                fk_result = ChainDispatcher.set_piece(
                    minute, fk_team, fk_opponent,
                    self.active_players.get(fk_team, []),
                    self.active_players.get(fk_opponent, []),
                    self.state, SituationType.DIRECT_FREEKICK,
                    attacks_right=(fk_team == home_team),
                    context_x=self.state.pending_offside_fk_x,
                    context_y=self.state.pending_offside_fk_y,
                    position_engine=self.position_engine,
                )
                # Stamp the actual offside location onto the FREEKICK_WON event
                # so the exporter records it at the correct coordinates.
                for ev in fk_result.events:
                    if ev.event_type == EventType.FREEKICK_WON:
                        ev.location_x = self.state.pending_offside_fk_x
                        ev.location_y = self.state.pending_offside_fk_y
                        break
                if self._absorb_chain(fk_result, minute): break
                continue

            # Decide which team has possession this sequence
            # weighted by possession split
            if random.random() < home_poss:
                attacking_team = home_team
                defending_team = away_team
            else:
                attacking_team = away_team
                defending_team = home_team

            self.state.possession_team = attacking_team
            if attacking_team == home_team:
                self._minute_home_seq += 1
            else:
                self._minute_away_seq += 1

            from tactical_ai import TacticalAI
            att_raw_profile = self.home_profile if attacking_team == home_team else self.away_profile
            def_raw_profile = self.away_profile if attacking_team == home_team else self.home_profile

            # Calculate average stamina for each team
            def get_avg_stamina(team_name):
                if not self.sub_controller: return 100.0
                players = self.active_players.get(team_name, [])
                staminas = [self.sub_controller.stamina[p.name].current_stamina for p in players if getattr(p, 'name', '') in self.sub_controller.stamina]
                return sum(staminas) / len(staminas) if staminas else 100.0

            att_avg_stamina = get_avg_stamina(attacking_team)
            def_avg_stamina = get_avg_stamina(defending_team)

            att_profile = TacticalAI.adjust(
                att_raw_profile, self.state, attacking_team, home_team,
                red_cards_against=self.state.home_red_cards if attacking_team != home_team
                                else self.state.away_red_cards,
                avg_stamina=att_avg_stamina
            )
            def_profile = TacticalAI.adjust(
                def_raw_profile, self.state, defending_team, home_team,
                red_cards_against=self.state.home_red_cards if defending_team != home_team
                                else self.state.away_red_cards,
                avg_stamina=def_avg_stamina
            )

            att_players = self.active_players.get(attacking_team, [])
            def_players = self.active_players.get(defending_team, [])

            if not att_players:
                continue

            # ── MOMENTUM & GAME STATE ────────────────────────────────
            momentum_mod    = MomentumEngine.get_attacking_probability_modifier(
                self.state, attacking_team, home_team)
            game_state_mod  = MomentumEngine.get_game_state_modifier(
                self.state, attacking_team, home_team)
            phase_goal_mult = PhaseEngine.goal_mult(phase)

            attacks_right = (attacking_team == home_team)

            # ── TRANSITION PRESS ─────────────────────────────────────
            # A dedicated transition event happens ~25% of sequences
            # (on top of the pressure events embedded in PossessionChain)
            press_prob = (
                def_profile.press_intensity
                * PhaseEngine.press_mult(phase)
                * 0.25
            )
            if random.random() < press_prob:
                trans_result = ChainDispatcher.transition(
                    minute, defending_team, attacking_team,
                    def_players, att_players, def_profile, self.state,
                    position_engine=self.position_engine,
                    attacks_right=attacks_right,
                )
                if self._absorb_chain(trans_result, minute): break
                if trans_result.possession_lost:
                    # Ball changes hands — next sequence is for other team
                    continue

            # ── POSSESSION SEQUENCE ──────────────────────────────────
            # Sequence length varies by style and game state
            seq_length = PossessionEngine.sequence_length(att_profile, self.state)

            # Pass defending players into possession chain so it can
            # generate realistic pressure events at the right locations.
            # Checkpoint 15: the DEFENDING team's live (TacticalAI-adjusted)
            # press intensity and pressing style are passed through so the
            # possession chain resolves the correct pressing profile for the
            # cover-shadow geometry and per-profile press probabilities
            # (previously def_press_intensity was never forwarded and the
            # chain fell back to the ATTACKING team's press_intensity — a bug).
            poss_result = ChainDispatcher.possession(
                minute, attacking_team, att_players,
                att_profile, self.state, seq_length,
                defending_players=def_players,
                position_engine=self.position_engine,
                context_x=self.state.last_ball_x,
                context_y=self.state.last_ball_y,
                attacks_right=attacks_right,
                def_press_intensity=def_profile.press_intensity,
                def_style_key=def_raw_profile.style.value,
                att_style_key=att_raw_profile.style.value,
            )
            if self._absorb_chain(poss_result, minute): break

            if poss_result.possession_lost:
                if self._defensive_recovery(
                    minute, poss_result, attacking_team, defending_team,
                    att_players, def_players, attacks_right, def_avg_stamina,
                ):
                    break
                continue  # Next sequence starts with other team

            # ── ATTACKING MATRIX SHOT HAND-OFF (Checkpoint 10) ──────────
            # The possession chain's per-touch matrix resolved SHOOT: the ball
            # carrier's touch already set the shot anchor. Dispatch the existing
            # AttackChain shot pipeline anchored at that position (reusing its
            # xG / body-part / angle-difficulty / GK / woodwork / corner /
            # restart logic unchanged), then hand control back to the loop —
            # the attack chain's outcome (goal / save / miss / block) already
            # determined possession for the next sequence. `continue` also
            # guarantees the independent shot_prob block below never fires a
            # second chance from the same sequence.
            if poss_result.shoot_decision:
                att_result = ChainDispatcher.attack(
                    minute, attacking_team, defending_team,
                    att_players, def_players,
                    att_profile, def_profile, self.state, SituationType.OPEN_PLAY,
                    position_engine=self.position_engine,
                    context_x=poss_result.shoot_x,
                    context_y=poss_result.shoot_y,
                    attacks_right=attacks_right,
                )
                if self._absorb_chain(att_result, minute): break
                continue

            # ── DEFENSIVE CONTEST (Checkpoint 6 + Checkpoint 9) ─────────
            # Causal gating fix: standalone tackles/interceptions/clearances/
            # blocks (DefensiveChain) previously never fired in open play at
            # all — this is what "a tackle = pressure but not vice versa"
            # actually requires. A defensive action here is now GATED behind
            # a real PRESS event having occurred earlier in this same
            # possession sequence (PossessionChain's embedded pressure
            # checks), and its frequency scales directly with the defending
            # team's press_intensity — a high-press side genuinely racks up
            # more tackles/clearances/blocks, not just more presses.
            #
            # Checkpoint 9 — the DANGER LEVEL now steers WHICH action the
            # defence reaches for: when the ball is close to their goalpost
            # xy, they get it away (clearance/block bias); when danger is
            # low they win it back (tackle/interception bias). The defender
            # picks the action closest to the ball, and clears an AERIAL
            # ball with a headed clearance vs a low ball with a foot one.
            pressure_occurred = any(
                e.event_type == EventType.PRESS for e in poss_result.events
            )
            if pressure_occurred:
                contest_prob = min(0.65, def_profile.press_intensity
                                    * PhaseEngine.press_mult(phase) * 0.55)
                if random.random() < contest_prob:
                    last_evt = poss_result.events[-1]
                    ctx_x = last_evt.end_x if last_evt.end_x is not None else last_evt.location_x
                    ctx_y = last_evt.end_y if last_evt.end_y is not None else last_evt.location_y
                    danger = self.threat.danger_at(defending_team)
                    own_goal_x = 105.0 if attacks_right else 0.0
                    action_type = random.choices(
                        ["tackle", "interception", "clearance", "block"],
                        weights=self._danger_scaled_action_weights(danger),
                    )[0]
                    # Clearances/blocks only make sense defending near their
                    # own goal — well upfield, fall back to tackle/interception.
                    clearance_zone = ctx_x > 70 if attacks_right else ctx_x < 35
                    if action_type in ("clearance", "block") and ctx_x is not None and not clearance_zone:
                        action_type = "tackle" if random.random() < 0.6 else "interception"
                    def_result = ChainDispatcher.defensive_action(
                        minute, defending_team, attacking_team,
                        def_players, att_players, self.state, action_type,
                        context_x=ctx_x, context_y=ctx_y,
                        attacks_right=attacks_right,
                        danger_level=danger,
                        ball_aerial=self._infer_aerial_ball(poss_result.events, last_evt),
                        own_goal_x=own_goal_x,
                        position_engine=self.position_engine,
                        ball_z=self._infer_ball_height(poss_result.events, last_evt),
                        defender_facing_x=self._defender_facing_at(
                            defending_team, ctx_x, ctx_y, own_goal_x)[0],
                        defender_facing_y=self._defender_facing_at(
                            defending_team, ctx_x, ctx_y, own_goal_x)[1],
                        opponent_distance=self._contest_distance(
                            defending_team, attacking_team, ctx_x, ctx_y),
                        stamina=def_avg_stamina,
                        referee_strictness=self.config.referee_strictness,
                    )
                    if self._absorb_chain(def_result, minute): break
                    if def_result.possession_lost:
                        continue  # Defense won the ball — no shot phase this sequence

            # ── DIRECT CLEARANCE (even without press) ─────────────
            if not poss_result.possession_lost:
                last_evt = poss_result.events[-1]
                ctx_x = last_evt.end_x if last_evt.end_x is not None else last_evt.location_x
                ctx_y = last_evt.end_y if last_evt.end_y is not None else last_evt.location_y
                dangerous_def = ctx_x > 80 if attacks_right else ctx_x < 25
                if dangerous_def and random.random() < 0.50:
                    def_result = ChainDispatcher.defensive_action(
                        minute, defending_team, attacking_team,
                        def_players, att_players, self.state, "clearance",
                        context_x=ctx_x, context_y=ctx_y,
                        attacks_right=attacks_right,
                        danger_level=self.threat.danger_at(defending_team),
                        ball_aerial=self._infer_aerial_ball(poss_result.events, last_evt),
                        own_goal_x=105.0 if attacks_right else 0.0,
                        position_engine=self.position_engine,
                        ball_z=self._infer_ball_height(poss_result.events, last_evt),
                        defender_facing_x=self._defender_facing_at(
                            defending_team, ctx_x, ctx_y, 105.0 if attacks_right else 0.0)[0],
                        defender_facing_y=self._defender_facing_at(
                            defending_team, ctx_x, ctx_y, 105.0 if attacks_right else 0.0)[1],
                        opponent_distance=self._contest_distance(
                            defending_team, attacking_team, ctx_x, ctx_y),
                        stamina=def_avg_stamina,
                        referee_strictness=self.config.referee_strictness,
                    )
                    if self._absorb_chain(def_result, minute): break
                    if def_result.possession_lost:
                        continue

            # ── GAME STATE: shot volume and quality modifiers ────────
            gd = self.state.home_goals - self.state.away_goals
            att_gd = gd if attacking_team == home_team else -gd

            shot_prob = (
                att_profile.shots_per_sequence
                * momentum_mod
                * game_state_mod
                * phase_goal_mult
                # Divide by n_sequences so total shots/game stays realistic
                # despite multiple sequences per minute.
                #
                # FIX (scoreline realism): the divisor was `n_sequences * 0.7`,
                # which under-divided and let an Attacking/High-Press team
                # (0.18-0.22 shots/sequence) rack up 40-55 shots/team/game —
                # far above the realistic 12-18 band. Raising it to
                # `n_sequences * 1.0` pulls per-sequence shot probability down
                # proportionally so total shots land in the real-football range.
                / max(1, n_sequences * 1.0)
            )

            if att_gd <= -2 and minute >= 60:
                shot_prob *= 1.30
                _xg_quality_mult = 0.80
            elif att_gd == -1 and minute >= 70:
                shot_prob *= 1.15
                _xg_quality_mult = 0.90
            elif att_gd >= 2:
                shot_prob *= 0.75
                _xg_quality_mult = 1.15
            elif att_gd == 1 and minute >= 75:
                shot_prob *= 0.85
                _xg_quality_mult = 1.05
            else:
                _xg_quality_mult = 1.0

            # FIX (scoreline realism): the scoreline governor `_xg_quality_mult`
            # was computed in every branch above but NEVER applied — grep showed
            # zero usages. This is what let a 6-0 runaway keep producing
            # high-quality chances at full volume, inflating scorelines to
            # 8-8 / 10-4. It is now multiplied into shot_prob so the scoreline
            # feeds back: a team 2+ down (0.80) creates fewer chances, a team
            # cruising 2+ up (1.15) creates more but is already ahead — this
            # compresses runaway scorelines toward realistic 2-3 goal margins.
            shot_prob *= _xg_quality_mult

            # Red card: 10-man team creates less
            if (attacking_team == home_team and self.state.home_red_cards > 0) or \
               (attacking_team == away_team and self.state.away_red_cards > 0):
                shot_prob *= 0.80

            # ── CHANCE / SHOT ────────────────────────────────────────
            # shot_taken guards against double-firing a shot in the same
            # sequence if the attacking matrix already resolved one (it would
            # normally be skipped via the continue above — this is belt and
            # braces).
            if random.random() < shot_prob and not poss_result.shot_taken:
                situation = self._determine_situation(att_profile, phase, style)

                if situation in (SituationType.CORNER, SituationType.DIRECT_FREEKICK,
                                  SituationType.CROSSED_FREEKICK, SituationType.PENALTY):
                    # Fix: anchor set piece to last ball state — the situation
                    # was generated from the ball's actual location, not a new
                    # independent random zone. Free kicks and corners happen
                    # where the ball was, not where a separate random draw lands.
                    if situation in (SituationType.DIRECT_FREEKICK, SituationType.CROSSED_FREEKICK):
                        fk_x = min(90.0, max(65.0, self.state.last_ball_x))
                        fk_y = max(15.0, min(53.0, self.state.last_ball_y))
                        self.state.last_ball_x = fk_x
                        self.state.last_ball_y = fk_y
                    sp_result = ChainDispatcher.set_piece(
                        minute, attacking_team, defending_team,
                        att_players, def_players, self.state, situation,
                        attacks_right=attacks_right,
                        position_engine=self.position_engine,
                    )
                    if self._absorb_chain(sp_result, minute): break
                else:
                    att_result = ChainDispatcher.attack(
                        minute, attacking_team, defending_team,
                        att_players, def_players,
                        att_profile, def_profile, self.state, situation,
                        position_engine=self.position_engine,
                        context_x=self.state.last_ball_x,
                        context_y=self.state.last_ball_y,
                        attacks_right=attacks_right,
                    )
                    if self._absorb_chain(att_result, minute): break


        # ── FOUL / DISCIPLINE ────────────────────────────────────
        # Rolled ONCE per minute (not once per sequence) so foul volume
        # stays realistic no matter how many sequences a minute produces
        # or how often possession changes hands. Real football: ~20-30
        # fouls and ~3-5 yellows per match (~0.25-0.3 fouls/minute).
        # The referee's strictness shapes how often a foul becomes a
        # card — a lenient ref doesn't make players foul less, he just
        # books fewer of them.
        last_attacker = self.state.possession_team or home_team
        fouling_team = away_team if last_attacker == home_team else home_team
        att_right = (last_attacker == home_team)
        foul_prob = (
            0.23
            * PhaseEngine.card_mult(phase)
        )
        if random.random() < foul_prob:
            # PHYSICS-ANCHORED FOUL LOCATION.
            # A defensive foul happens at the live engagement point — where
            # the defending team is actually challenging the ball — NOT at an
            # arbitrary random spot. We anchor to the ball; the foul drifts a
            # few metres around that contest (striker checked, second ball,
            # shoulder in the channel).
            lx = self.state.last_ball_x
            ly = self.state.last_ball_y
            foul_x = min(101.0, max(6.0, lx + random.gauss(0, 16.0)))
            foul_y = min(63.0, max(5.0, ly + random.gauss(0, 4.5)))

            # PHYSICS-GROUNDED PENALTY CONVICTION.
            # A foul inside the box is NOT automatically a spot-kick. The ref
            # only gives one when the defending team's lunge actually denied a
            # clear scoring opportunity — i.e. the ball really was deep in its
            # OWN box and live danger was high (defenders scrambling). We drive
            # that straight off the ThreatEngine's pure ball↔goal geometry, and
            # keep it deliberately scarce: a real penalty roughly every 3-5
            # matches, never one per match. A per-team cap stops a siege turning
            # into a spot-kick carnival.
            box_conviction = 0.0
            ball_in_deny_zone = (lx >= 84) if att_right else (lx <= 21)
            if ball_in_deny_zone:
                danger_def = self.threat.danger_at(fouling_team)
                taken = self.state.penalties_taken.get(fouling_team, 0)
                if taken < MatchState.PENALTY_CAP_PER_TEAM:
                    box_conviction = 0.04 + 0.09 * (danger_def / 100.0)   # ~0.04-0.13
                    if taken >= 1:
                        box_conviction *= 0.35   # a second spot is a rare table-tilt
                    box_conviction = min(0.5, box_conviction)
            disc_result = ChainDispatcher.discipline(
                minute, fouling_team, last_attacker,
                self.active_players.get(fouling_team, []),
                self.active_players.get(last_attacker, []),
                self.state,
                referee_strictness=self.config.referee_strictness,
                x=foul_x, y=foul_y,
                attacks_right=att_right,
                box_penalty_chance=box_conviction,
            )
            self._absorb_chain(disc_result, minute)

    def _danger_scaled_action_weights(self, danger: float) -> List[float]:
        """
        Checkpoint 9 — how the live danger level steers which defensive
        action the team reaches for:

            danger 0        → unchanged baseline (tackle-heavy)
            danger ≥ 30     → mild shift toward clearances
            danger ≥ 60     → clear the lines (clearance/block heavy)
            danger ≥ 85     → six-yard scramble: bodies on everything

        The danger-0 branch is byte-for-byte the pre-feature weights, which
        is what keeps the no-threat baseline statistically unchanged.
        """
        if danger >= 85:
            return [0.15, 0.08, 0.48, 0.29]   # tackle, interception, clearance, block
        if danger >= 60:
            return [0.20, 0.12, 0.44, 0.24]
        if danger >= 30:
            return [0.28, 0.25, 0.27, 0.20]
        return [0.32, 0.28, 0.22, 0.18]

    def _defensive_recovery(self, minute, poss_result, attacking_team,
                            defending_team, att_players, def_players,
                            attacks_right: bool, def_avg_stamina) -> bool:
        """Checkpoint 29 — defensive wins inside PossessionChain (physics
        race-to-ball interceptions, miscontrols, lost duels) previously died
        at the possession_lost continue and never reached the DefensiveChain
        dispatcher, so deep clearances/blocks collapsed to ~1/match. When a
        sequence is turned over in the defending third, the defence now gets
        a danger-scaled chance to hammer it away (clearance-heavy), feeding
        the same clearance/corner/own-goal pipeline as the contest path.
        Returns True when the match ended during the recovery."""
        if not poss_result.events:
            return False
        last_evt = poss_result.events[-1]
        if last_evt.event_type in (EventType.CLEARANCE, EventType.BLOCK):
            return False
        ctx_x = last_evt.end_x if last_evt.end_x is not None else getattr(last_evt, "location_x", None)
        ctx_y = last_evt.end_y if last_evt.end_y is not None else getattr(last_evt, "location_y", None)
        if ctx_x is None:
            return False
        deep_zone = ctx_x > 70.0 if attacks_right else ctx_x < 35.0
        if not deep_zone:
            return False
        danger = self.threat.danger_at(defending_team)
        recovery_prob = 0.22 + min(0.20, danger / 250.0)
        if random.random() >= recovery_prob:
            return False
        clearance_w = 0.70 + danger / 400.0
        action_type = random.choices(
            ["clearance", "block", "tackle"],
            weights=[clearance_w, 0.16, 0.12],
        )[0]
        own_goal_x = 105.0 if attacks_right else 0.0
        from event_chain import ChainDispatcher
        def_result = ChainDispatcher.defensive_action(
            minute, defending_team, attacking_team,
            def_players, att_players, self.state, action_type,
            context_x=ctx_x, context_y=ctx_y,
            attacks_right=attacks_right,
            danger_level=danger,
            ball_aerial=self._infer_aerial_ball(poss_result.events, last_evt),
            own_goal_x=own_goal_x,
            position_engine=self.position_engine,
            ball_z=self._infer_ball_height(poss_result.events, last_evt),
            defender_facing_x=self._defender_facing_at(
                defending_team, ctx_x, ctx_y, own_goal_x)[0],
            defender_facing_y=self._defender_facing_at(
                defending_team, ctx_x, ctx_y, own_goal_x)[1],
            opponent_distance=self._contest_distance(
                defending_team, attacking_team, ctx_x, ctx_y),
            stamina=def_avg_stamina,
            referee_strictness=self.config.referee_strictness,
        )
        return self._absorb_chain(def_result, minute)

    def _infer_aerial_ball(self, events, last_evt) -> bool:
        """
        Checkpoint 9 — is the ball currently in the air when the defence
        has to react? A cross / corner / free-kick cross / aerial duel /
        headed touch just before the defensive action means the defender
        clears with their HEAD; a low ball is cleared with the FOOT.

        Checkpoint 11 — the geometric CrossDetector's `is_airborne` stamp
        (set on every qualifying cross) is authoritative when present; a
        low driven cross is correctly routed to a FOOT clearance.
        """
        if last_evt is None:
            return False
        meta = getattr(last_evt, "metadata", None) or {}
        if meta.get("is_airborne") is True:
            return True
        if getattr(last_evt, "body_part", "") == "head":
            return True
        if last_evt.event_type in (
            EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
            EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS,
            EventType.AERIAL_DUEL,
        ):
            return True
        for e in reversed(events[-4:]):
            m = getattr(e, "metadata", None) or {}
            if m.get("is_airborne") is True:
                return True
            if getattr(e, "body_part", "") == "head":
                return True
            if e.event_type in (
                EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
                EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS,
            ):
                return True
        return False

    def _near_ball_counts(self, bx: float, by: float) -> Dict[str, int]:
        """
        Checkpoint 9 — how many outfield bodies from each team are within
        ~8m of the ball right now. Feeds the danger assessment's pressure
        factor (a striker unmarked at the penalty spot is worse than a
        5-on-1 scramble). Reads the Position Engine's live spatial state.
        """
        counts: Dict[str, int] = {}
        for team, players in self.active_players.items():
            n = 0
            for p in players:
                if getattr(p, "position", "") == "GK":
                    continue
                px, py = self.position_engine.get_position(p.name)
                if (px - bx) ** 2 + (py - by) ** 2 <= 64.0:   # 8m radius
                    n += 1
            counts[team] = n
        return counts

    def _infer_ball_height(self, events, last_evt) -> float:
        """
        Checkpoint 10 — the Z-AXIS. A ball the defence has to react to that
        came from a cross / corner / free-kick cross / aerial duel / headed
        touch is above hip height; a low ball is on the deck or a low bounce.
        This height (metres) is what picks the headed vs foot clearance tool
        (Z > 1.2m ⇒ head).
        """
        if self._infer_aerial_ball(events, last_evt):
            return round(random.uniform(1.4, 2.6), 2)
        return round(random.uniform(0.2, 1.1), 2)

    def _defender_facing_at(self, def_team: str, bx: float, by: float,
                            own_goal_x: float) -> Tuple[float, float]:
        """
        Checkpoint 10 — which way is the nearest defender facing when they
        react to the ball? A defender still goal-side of the ball faces the
        ball (Optimal zone). A defender who has been BEATEN (ball closer to
        their own goalpost xy than they are) is sprinting back and faces their
        own goal — the Blind/Panic zone where sliced clearances and own goals
        live.
        """
        from threat_engine import defender_facing_point
        best = None
        for p in self.active_players.get(def_team, []):
            if getattr(p, "position", "") == "GK":
                continue
            dx, dy = self.position_engine.get_position(p.name)
            d = (dx - bx) ** 2 + (dy - by) ** 2
            if best is None or d < best[0]:
                best = (d, dx, dy)
        if best is None:
            return None, None
        return defender_facing_point(best[1], best[2], bx, by, own_goal_x)

    def _contest_distance(self, def_team: str, att_team: str,
                          bx: float, by: float) -> Optional[float]:
        """
        Checkpoint 10 — how CONTESTED is the clearing defender? The distance
        (metres) from the defender nearest the ball to the nearest attacking
        player. 0.5m = fully contested (spec); under ~2m it starts amplifying
        P_fail. None when there's no attacker near enough to care.
        """
        near_def = []
        for p in self.active_players.get(def_team, []):
            if getattr(p, "position", "") == "GK":
                continue
            dx, dy = self.position_engine.get_position(p.name)
            if (dx - bx) ** 2 + (dy - by) ** 2 <= 225.0:   # within 15m of ball
                near_def.append((dx, dy))
        if not near_def:
            return None
        dx, dy = min(near_def, key=lambda q: (q[0] - bx) ** 2 + (q[1] - by) ** 2)
        best = None
        for p in self.active_players.get(att_team, []):
            if getattr(p, "position", "") == "GK":
                continue
            ax, ay = self.position_engine.get_position(p.name)
            d = math.hypot(ax - dx, ay - dy)
            if best is None or d < best:
                best = d
        return round(best, 2) if best is not None else None

    def _absorb_chain(self, chain_result, minute: int) -> bool:
        """
        Read a ChainResult and update the engine's timeline + match state.
        This is the single point where chain outputs become match facts.
        Also drains stamina from every player involved in each event.
        """
        from squad_manager import get_stamina_action
        # Add all events to the timeline + drain stamina
        for event in chain_result.events:
            self.timeline.append(event)

            # ── STAMINA DRAIN ──────────────────────────────────────
            # Checkpoint 15: pressing-profile fatigue tax. PRESS events
            # carry the defending profile's stamina_tax multiplier; a
            # gegenpress team pays 1.35x per press while a low block pays
            # 1.0x. That tax drains team stamina faster, TacticalAI lowers
            # the effective press intensity, and the press weakens — the
            # "press yourself into exhaustion" loop.
            press_tax = float(
                (getattr(event, "metadata", None) or {}).get("press_tax", 1.0)
            )
            if self.sub_controller is not None:
                # Primary actor
                if event.player:
                    action_key = get_stamina_action(event.event_type.name)
                    if action_key:
                        self.sub_controller.process_action(
                            event.player, action_key, event.team, minute,
                            drain_mult=press_tax,
                        )
                    else:
                        # Log missing mapping for debugging
                        print(f"  ⚠️ No stamina mapping for: {event.event_type.name}")
                
                # Secondary actor (duels, tackles, passes)
                if event.secondary_player:
                    action_key = get_stamina_action(event.event_type.name)
                    if action_key:
                        # Secondary actors get reduced drain (they're not the primary actor)
                        self.sub_controller.process_action(
                            event.secondary_player, action_key, event.team, minute,
                            drain_mult=press_tax, is_secondary=True,
                        )

            # Checkpoint 6.3 — pass energy cost: long/difficult passes drain
            # extra mental/physical energy from the passer. A 40m diagonal
            # under pressure costs more than a 5m safe pass, modelling the
            # real-life "mental battery" drain Enzo/Rice-level midfielders
            # manage by choosing the right pass at the right time.
            _meta = getattr(event, "metadata", None) or {}
            pass_energy = _meta.get("pass_energy_cost")
            if (pass_energy is not None and event.player
                    and self.sub_controller is not None):
                state = self.sub_controller.stamina.get(event.player)
                if state is not None and not state.is_injured:
                    extra_drain = 0.02 + float(pass_energy) * 0.12
                    state.drain("pass_energy", extra_drain)
                    state.update_performance_mult()

            # Checkpoint 5: keep the Position Engine's live spatial state
            # truthful — every event with real coordinates updates the
            # involved player(s)' current position, not just the ball's.
            if event.location_x is not None and event.location_y is not None:
                self.position_engine.record_touch(
                    event.player, event.location_x, event.location_y, minute
                )
                if event.secondary_player:
                    # Use end coordinates if present (e.g. pass receiver),
                    # else the same location (duels, presses).
                    sx = event.end_x if event.end_x is not None else event.location_x
                    sy = event.end_y if event.end_y is not None else event.location_y
                    self.position_engine.record_touch(
                        event.secondary_player, sx, sy, minute
                    )

            # Drain stamina for primary actor
            if self.sub_controller is not None and event.player:
                action_key = get_stamina_action(event.event_type.name)
                if action_key:
                    self.sub_controller.process_action(
                        event.player, action_key, event.team, minute
                    )
                # Drain secondary player too (e.g. aerial duel both sides)
                if event.secondary_player and event.event_type.name in (
                    "AERIAL_DUEL", "GROUND_DUEL", "TACKLE_WON", "TACKLE_LOST"
                ):
                    self.sub_controller.process_action(
                        event.secondary_player, action_key, event.team, minute
                    )

            # Checkpoint 7 — persistent ball-state: keep state.last_ball_x/y
            # truthful to whatever actually just happened, in event order,
            # so the LAST event of this chain is what the NEXT sequence's
            # starting position anchors off. Prefer end_x/end_y (where the
            # ball ended up after a pass/carry/clearance) and fall back to
            # location_x/y for events with no distinct end point (duels,
            # tackles, presses).
            bx = event.end_x if event.end_x is not None else event.location_x
            by = event.end_y if event.end_y is not None else event.location_y
            if bx is not None and by is not None:
                self.state.last_ball_x = bx
                self.state.last_ball_y = by

            # ── CHECKPOINT 11: CROSS SITUATION TRIGGER ──────────────
            # A detected cross delivery (engine CROSS_ATTEMPT/SUCCESS/corner
            # OR any pass the geometric CrossDetector stamped `cross: true`)
            # arms this minute's box-crash run for the attacking team. The
            # threat engine independently forces the defending danger to
            # HIGH/CRITICAL in observe_event via the same metadata. cross_x/y
            # is the DELIVERY ORIGIN (the wide crossing zone), which is what
            # the attacking_crash gate keys off.
            _meta = getattr(event, "metadata", None) or {}
            _etype = getattr(event.event_type, "name", "")
            if _etype in ("CROSS_ATTEMPT", "CROSS_SUCCESS", "CORNER_TAKEN") \
                    or _meta.get("cross"):
                ox = event.location_x if event.location_x is not None else bx
                oy = event.location_y if event.location_y is not None else by
                self.state.cross_active = True
                self.state.cross_team = event.team or ""
                self.state.cross_player = event.player or ""
                self.state.cross_x = ox
                self.state.cross_y = oy
                self.state.cross_attacks_right = (event.team == self.config.home_team)

                # Realistic corner causality (not a random draw): a live cross
                # into the box that is NOT converted is regularly put behind by
                # the defender/keeper for a corner — the single most common
                # corner origin in real football. Only awarded when the cross
                # is genuinely delivered (CROSS_SUCCESS) or contested, capped
                # at one per minute so an uncontested delivery can't pile up
                # corners. This raises corner volume with a CAUSAL source
                # instead of inflating the random CORNER situation weight.
                if (_etype == "CROSS_SUCCESS" or _meta.get("cross")) \
                        and not self.state.cross_corner_done \
                        and not chain_result.corner_won \
                        and event.team:
                    # Defenders typically clear the high-ball behind when they
                    # are under real pressure near their own goal (danger high).
                    danger = self.threat.danger_at(
                        self.config.home_team if event.team != self.config.home_team
                        else self.config.away_team
                    )
                    behind_prob = 0.15 + 0.22 * min(1.0, max(0.0, danger / 100.0))
                    if random.random() < behind_prob:
                        self.state.cross_corner_done = True
                        chain_result.corner_won = True
                        chain_result.corner_team = event.team
                        self.state.cross_active = False

            # Checkpoint 9 — Threat Engine: keep both teams' live danger level
            # truthful to the ball's actual position every single event. Near-
            # ball player counts (who has bodies on the ball) feed the pressure
            # factor of the danger assessment.
            self.threat.observe_event(
                event, minute,
                near_counts=self._near_ball_counts(bx, by),
            )


        # Checkpoint 6 — corner causality: a chain reporting corner_won is
        # no longer a discarded flag. It queues the ACTUAL next set-piece
        # sequence for the team that won it, consumed at the top of
        # _simulate_minute's sequence loop. Turned into a per-team COUNTER so
        # corners won in the SAME sequence can never overwrite (drop) each
        # other — each win survives until the loop takes it.
        if chain_result.corner_won and chain_result.corner_team:
            if chain_result.corner_team == self.config.home_team:
                self.state.pending_corners_home += 1
            else:
                self.state.pending_corners_away += 1

        # Checkpoint X — penalty causality: a foul a defending team committed
        # inside its OWN box is a spot-kick offence. The fouling team's box
        # foul means the FOULED side takes the kick, so we queue the fouled
        # team. The PENALTY_WON event's `.team` is the fouled (attacking)
        # side. This converts a won penalty into an ACTUAL spot-kick sequence
        # rather than leaving it as a dangling one-off timeline event.
        if chain_result.penalty_won:
            _fouled = ""
            for _ev in chain_result.events:
                if _ev.event_type == EventType.PENALTY_WON:
                    _fouled = _ev.team
                    break
            if _fouled:
                if _fouled == self.config.home_team:
                    self.state.pending_penalty_home += 1
                else:
                    self.state.pending_penalty_away += 1

        # Checkpoint 8 — restart causality: goal kicks and throw-ins
        # When a chain reports restart_required, queue the actual restart
        # chain (in _simulate_minute) rather than emitting a stub event.
        # This lets GoalKickChain and ThrowInChain model their full logic
        # (short vs. long build-up, footedness bias, Brentford long throws...)
        if chain_result.restart_required:
            restart_team = chain_result.restart_team
            if not restart_team:
                if chain_result.restart_type == "throw_in":
                    restart_team = (self.config.away_team if self.state.possession_team == self.config.home_team 
                                   else self.config.home_team)
                elif chain_result.restart_type == "goal_kick":
                    restart_team = (self.config.away_team if self.state.possession_team == self.config.home_team 
                                   else self.config.home_team)
            
            if chain_result.restart_type == "throw_in":
                self.state.pending_throw_in_for = restart_team
            elif chain_result.restart_type == "goal_kick":
                self.state.pending_goal_kick_for = restart_team
            self.state.pending_restart_x = chain_result.restart_x
            self.state.pending_restart_y = chain_result.restart_y

        # Checkpoint 19 — offside detection: queue a free kick at the
        # offside location for the defending team. The free kick is NOT
        # placed in a random zone — it is placed exactly where the
        # offside occurred, which is what the real laws of the game prescribe.
        if getattr(chain_result, 'offside_detected', False):
            offside_attacking_team = getattr(chain_result, 'offside_team', self.state.possession_team)
            defending_team = (
                self.config.away_team if offside_attacking_team == self.config.home_team
                else self.config.home_team
            )
            self.state.pending_offside_fk_for = defending_team
            self.state.pending_offside_fk_x = getattr(chain_result, 'offside_x', 0.0)
            self.state.pending_offside_fk_y = getattr(chain_result, 'offside_y', 0.0)


        # Goal
        if chain_result.goal_scored:
            if getattr(chain_result, 'delayed_offside', False):
                # VAR DISALLOWED GOAL
                if not self.quiet:
                    print(f"  ❌ GOAL RULED OUT (VAR/Offside)! {minute}' — {chain_result.goal_scorer}")
                self.timeline.append(MatchEvent(
                    minute=minute, second=0,
                    event_type=EventType.VAR_DISALLOWED_GOAL,
                    team=chain_result.goal_team,
                    player=chain_result.goal_scorer,
                    phase=self.state.phase, game_state=self.state.game_state
                ))
                # Free kick to opposing team (no goal)
                self.state.pending_kickoff_for = ""
            else:
                # Find the goal event already in timeline (including an
                # own goal — a critical clearance failure redirects the
                # ball into the defender's own net — and penalties, which
                # also set goal_scored but emit PENALTY_SCORED).
                goal_events = [e for e in chain_result.events
                               if e.event_type in (EventType.GOAL, EventType.OWN_GOAL,
                                                   EventType.PENALTY_SCORED)]
                for ge in goal_events:
                    self.goals.append(ge)
                if chain_result.goal_team == self.config.home_team:
                    self.state.home_goals += 1
                else:
                    self.state.away_goals += 1
                self.state.momentum = MomentumEngine.after_goal(
                    self.state, chain_result.goal_team, self.config.home_team
                )
                if getattr(chain_result, "own_goal", False):
                    if not self.quiet:
                        print(f"  🥅 OWN GOAL! {minute}' — {chain_result.goal_scorer} "
                              f"({chain_result.goal_team}) [{self.state.score_str}]")
                else:
                    if not self.quiet:
                        print(f"  ⚽ GOAL! {minute}' — {chain_result.goal_scorer} "
                              f"({chain_result.goal_team}) [{self.state.score_str}]")
                # Set up Kickoff for conceding team
                conceding_team = self.config.away_team if chain_result.goal_team == self.config.home_team else self.config.home_team
                self.state.pending_kickoff_for = conceding_team
                # Realism: snap both teams back to their halves so the restart
                # begins from clean defensive shapes rather than a scrambled
                # goal-mouth tail.
                self._reset_positions_to_halves()
                # Checkpoint 9 — the threat was realised: the conceding team's
                # danger PEAKS (a goal came from it), then resets at kickoff.
                self.threat.on_goal(conceding_team, minute)
                return True # Break sequence loop

        # Penalty scored (separate event type)
        pen_goals = [e for e in chain_result.events if e.event_type == EventType.PENALTY_SCORED]
        for pe_ev in pen_goals:
            if pe_ev not in self.goals:
                self.goals.append(pe_ev)
            if chain_result.goal_team == self.config.home_team:
                self.state.home_goals += 1
            else:
                self.state.away_goals += 1
            self.state.momentum = MomentumEngine.after_goal(
                self.state, chain_result.goal_team, self.config.home_team
            )
            if not self.quiet:
                print(f"  ⚽ PENALTY! {minute}' — {chain_result.goal_scorer} "
                      f"({chain_result.goal_team}) [{self.state.score_str}]")

            # xG accumulation
            # xG accumulation
        if chain_result.xg_generated > 0:
            # Attribute xG to whichever team actually took the shot, not to
            # self.state.possession_team. possession_team can be stale here —
            # e.g. a successful TransitionChain press flips the ball to the
            # pressing/counter-attacking team, but possession_team is only
            # updated for the NEXT sequence, not before this chain result is
            # absorbed. That silently misattributed every successful counter's
            # xG to the team that had just been dispossessed, causing
            # state.home_xg/away_xg (used by the shot-map PNG, summary PNG,
            # and console summary) to diverge from the per-event totals used
            # by the Excel/CSV/JSON exports (which read event.team directly).
            _SHOT_TYPES = (
                EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                EventType.SHOT_BLOCKED, EventType.GOAL,
                EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
                EventType.HIT_WOODWORK,
            )
            shot_event = next(
                (e for e in chain_result.events if e.event_type in _SHOT_TYPES),
                None,
            )
            xg_team = shot_event.team if shot_event is not None else self.state.possession_team
            if xg_team == self.config.home_team:
                self.state.home_xg += chain_result.xg_generated
            else:
                self.state.away_xg += chain_result.xg_generated

        # Cards
        if chain_result.card_issued:
            card_events = [
                e for e in chain_result.events
                if e.event_type in (EventType.YELLOW_CARD, EventType.RED_CARD)
            ]
            for ce in card_events:
                self.cards.append(ce)
                player_name = ce.player
                card_team = ce.team

                if ce.event_type == EventType.YELLOW_CARD:
                    # Track booking — first yellow for this player
                    self.state.booked_players[player_name] = self.state.booked_players.get(player_name, 0) + 1
                    if not self.quiet:
                        print(f"  🟨 YELLOW CARD! {minute}' — {player_name} ({card_team})")

                elif ce.event_type == EventType.RED_CARD:
                    # Check if it's a second yellow
                    was_booked = self.state.booked_players.get(player_name, 0) > 0
                    if was_booked:
                        if not self.quiet:
                            print(f"  🟥🟨 SECOND YELLOW! {minute}' — {player_name} ({card_team}) SENT OFF")
                    else:
                        if not self.quiet:
                            print(f"  🟥 RED CARD! {minute}' — {player_name} ({card_team}) SENT OFF")

                    # Credit minutes up to the sending-off BEFORE removal —
                    # the final-whistle pass only covers players still in the
                    # active pools, so skipping this leaves the red-carded
                    # player at minutes_played=0 and breaks every per-90 stat.
                    for _p in self.active_players.get(card_team, []):
                        if getattr(_p, "name", "") == player_name and hasattr(_p, "dna"):
                            _p.dna.minutes_played = minute

                    if card_team == self.config.home_team:
                        self.state.home_red_cards += 1
                    else:
                        self.state.away_red_cards += 1
                    # Remove player from the active pool
                    self.active_players[card_team] = [
                        p for p in self.active_players.get(card_team, [])
                        if p.name != player_name
                    ]

                    self.state.sent_off_players.append(player_name)
                    self.state.booked_players.pop(player_name, None)  # Clear booking record
                    if self.position_engine is not None:
                        self.position_engine.remove_player(card_team, player_name)
                    self.state.momentum = MomentumEngine.after_red_card(
                        self.state, card_team, self.config.home_team
                    )

        return False

    def _simulate_shot_sequence(
        self,
        minute: int,
        attacking_team: str,
        defending_team: str,
        attacking_profile: TeamProfile,
        phase: MatchPhase,
    ):
        """
        Simulate the chain: chance created → shot attempt → outcome.
        This is the causal chain at the heart of the engine.
        """
        # Pick a shooter (position-weighted)
        shooter = self._pick_player(
            attacking_team,
            preferred_positions=['ST', 'CF', 'LW', 'RW', 'CAM'],
            exclude_pos=['GK']
        )
        creator = self._pick_player(
            attacking_team,
            preferred_positions=['CAM', 'CM', 'LW', 'RW', 'CDM'],
            exclude_pos=['GK'],
            exclude_player=shooter
        )

        # Is this a big chance?
        is_big_chance = random.random() < attacking_profile.big_chance_ratio

        # Determine shot zone and situation
        situation = self._determine_situation(attacking_profile, phase)
        zone, body_part = self._determine_shot_characteristics(
            situation, attacking_profile
        )

        # Calculate xG
        under_pressure = random.random() < (defending_profile := self.away_profile if attacking_team == self.config.home_team else self.home_profile).press_intensity * 0.4
        xg = XGEngine.calculate(
            zone=zone,
            body_part=body_part,
            situation=situation,
            under_pressure=under_pressure,
            is_big_chance=is_big_chance,
            first_time_shot=random.random() < 0.35,
        )

        # Accumulate team xG
        if attacking_team == self.config.home_team:
            self.state.home_xg += xg
        else:
            self.state.away_xg += xg

        # Emit chance created event
        chance_type = EventType.BIG_CHANCE_CREATED if is_big_chance else EventType.CHANCE_CREATED
        loc = self._shot_location(zone)
        self._emit_event(
            minute=minute,
            event_type=chance_type,
            team=attacking_team,
            player=creator,
            secondary_player=shooter,
            situation=situation,
            location_x=loc[0],
            location_y=loc[1],
            xa=xg * 0.85,  # xA slightly less than xG
        )

        # ── DOES IT RESULT IN A SHOT ON TARGET? ─────────────
        shot_on_target_prob = 0.35 + (xg * 0.5)
        shot_on_target_prob = min(0.92, max(0.08, shot_on_target_prob))

        if random.random() < shot_on_target_prob:
            # Shot on target
            self._emit_event(
                minute=minute,
                event_type=EventType.SHOT_ON_TARGET,
                team=attacking_team,
                player=shooter,
                situation=situation,
                location_x=loc[0],
                location_y=loc[1],
                xg=xg,
                body_part=body_part,
            )

            # ── DOES IT GO IN? ───────────────────────────────
            # Shooter quality affects conversion
            shooter_quality = self._get_shooter_quality(shooter)
            if XGEngine.does_goal_happen(xg, shooter_quality):
                self._register_goal(
                    minute, attacking_team, shooter, creator,
                    situation, zone, body_part, xg, is_big_chance
                )
            else:
                # Save! Momentum shifts slightly
                self.state.momentum = MomentumEngine.after_save(
                    self.state, defending_team, self.config.home_team
                )
                self._emit_event(
                    minute=minute,
                    event_type=EventType.SAVE,
                    team=defending_team,
                    player=self._pick_player(defending_team, preferred_positions=['GK']),
                    secondary_player=shooter,
                    location_x=loc[0],
                    location_y=loc[1],
                    xg=xg,
                    body_part=body_part,
                    outcome=True,
                )
        else:
            # Shot off target or blocked
            if random.random() < 0.35:
                self._emit_event(
                    minute=minute,
                    event_type=EventType.SHOT_BLOCKED,
                    team=attacking_team,
                    player=shooter,
                    secondary_player=self._pick_player(
                        defending_team, preferred_positions=['CB', 'CDM', 'CM']
                    ),
                    xg=xg,
                )
            else:
                self._emit_event(
                    minute=minute,
                    event_type=EventType.SHOT_OFF_TARGET,
                    team=attacking_team,
                    player=shooter,
                    xg=xg,
                )

    def _register_goal(
        self,
        minute: int,
        team: str,
        scorer: str,
        creator: str,
        situation: SituationType,
        zone: str,
        body_part: str,
        xg: float,
        is_big_chance: bool,
    ):
        """Register a goal and update match state."""
        if team == self.config.home_team:
            self.state.home_goals += 1
        else:
            self.state.away_goals += 1

        goal_event = MatchEvent(
            minute=minute,
            second=random.randint(0, 59),
            event_type=EventType.GOAL,
            team=team,
            player=scorer,
            secondary_player=creator,
            situation=situation,
            location_x=self._shot_location(zone)[0],
            location_y=self._shot_location(zone)[1],
            xg=xg,
            body_part=body_part,
            phase=self.state.phase,
            game_state=self.state.game_state,
            metadata={
                'is_big_chance': is_big_chance,
                'score_after': self.state.score_str,
            }
        )
        self.timeline.append(goal_event)
        self.goals.append(goal_event)

        # Massive momentum shift after goal
        self.state.momentum = MomentumEngine.after_goal(
            self.state, team, self.config.home_team
        )

        if not self.quiet:
            print(f"  ⚽ GOAL! {minute}' — {scorer} ({team}) [{self.state.score_str}]")

    def _emit_press_event(self, minute: int, pressing_team: str, attacked_team: str):
        """Emit a pressing event."""
        presser = self._pick_player(
            pressing_team,
            preferred_positions=['ST', 'LW', 'RW', 'CAM', 'CM']
        )
        self._emit_event(
            minute=minute,
            event_type=EventType.PRESS,
            team=pressing_team,
            player=presser,
        )

    def _emit_event(self, minute: int, event_type: EventType, team: str,
                    player: str, **kwargs) -> MatchEvent:
        """Create and record an event."""
        event = MatchEvent(
            minute=minute,
            second=kwargs.pop('second', random.randint(0, 59)),
            event_type=event_type,
            team=team,
            player=player,
            phase=self.state.phase,
            game_state=self.state.game_state,
            **kwargs
        )
        self.timeline.append(event)
        return event

    # ── HELPER METHODS ───────────────────────────────────────

    def _pick_player(
        self,
        team: str,
        preferred_positions: List[str] = None,
        exclude_pos: List[str] = None,
        exclude_player: str = None,
    ) -> str:
        """Pick a player from the active squad, weighted by position."""
        players = self.active_players.get(team, [])
        if not players:
            return f"{team}_Unknown"

        # Filter
        candidates = [
            p for p in players
            if (exclude_pos is None or getattr(p, 'position', 'CM') not in exclude_pos)
            and (exclude_player is None or getattr(p, 'name', '') != exclude_player)
        ]

        if not candidates:
            candidates = players

        # Weight by preferred positions
        if preferred_positions:
            weights = []
            for p in candidates:
                pos = getattr(p, 'position', 'CM')
                if pos in preferred_positions:
                    weights.append(4.0)
                else:
                    weights.append(1.0)
            chosen = random.choices(candidates, weights=weights, k=1)[0]
        else:
            chosen = random.choice(candidates)

        return getattr(chosen, 'name', str(chosen))

    def _get_shooter_quality(self, shooter_name: str) -> float:
        """Get a shooter's finishing quality modifier."""
        for team_players in self.active_players.values():
            for p in team_players:
                if getattr(p, 'name', '') == shooter_name:
                    specs = getattr(p, 'specialties', [])
                    if 'clinical_finisher' in specs or 'fox_in_box' in specs:
                        return 1.25
                    elif 'poacher' in specs:
                        return 1.15
                    elif 'shooter' in specs:
                        return 1.10
        return 1.0

    def _determine_situation(
        self, profile, phase: MatchPhase, style: TeamStyle = None
    ) -> SituationType:
        """Determine how the chance was created."""
        # Base weights.
        # Checkpoint 6: CORNER's independent weight is deliberately small now.
        # Most corners are generated CAUSALLY (blocked shots, ineffective
        # clearances/blocks -> pending_corners_home/away, consumed at the top of
        # the sequence loop) rather than by this random draw. What remains
        # here is a residual for real-world corner sources this engine
        # doesn't model discretely yet (keeper tipping over, a cross
        # knocked behind under no direct pressure, etc.) — not the primary
        # source of corners anymore.
        weights = {
            SituationType.OPEN_PLAY:        58,
            SituationType.FAST_BREAK:       17,
            SituationType.CORNER:           5,
            SituationType.DIRECT_FREEKICK:  9,
            SituationType.CROSSED_FREEKICK: 8,
            # NOTE: SituationType.PENALTY is intentionally ABSENT from this
            # weights table. A penalty is NOT a generic shot situation — it is
            # always the consequence of a defending-team foul inside its own
            # box, which the DisciplineChain reports via `box_conviction` and
            # queues through `pending_penalty_*`. Leaving PENALTY here generated
            # "phantom" penalty kicks out of ordinary attacking moves (a kick
            # nobody fouled for), inflating counts and double-charging the
            # conceding defence. Penalties are now produced exclusively by the
            # box-foul conviction path below.
        }

        # Resolve style — profile may be EffectiveTactics (no .style attr) or TeamProfile
        team_style = getattr(profile, 'style', style) or TeamStyle.BALANCED
        # Style adjustments
        if team_style == TeamStyle.WING_PLAY:
            weights[SituationType.CORNER] += 3
        if team_style == TeamStyle.FLUID_COUNTER:
            weights[SituationType.FAST_BREAK] += 10

        if team_style == TeamStyle.ROUTE_ONE:
            weights[SituationType.CROSSED_FREEKICK] += 5

        # Late game = more corners and set pieces
        if phase in (MatchPhase.FINAL_PUSH, MatchPhase.ADDED_TIME):
            weights[SituationType.CORNER] += 2
            weights[SituationType.DIRECT_FREEKICK] += 3

        situations = list(weights.keys())
        wts = list(weights.values())
        return random.choices(situations, weights=wts, k=1)[0]

    def _determine_shot_characteristics(
        self, situation: SituationType, profile: TeamProfile
    ) -> Tuple[str, str]:
        """Return (zone, body_part) for a shot."""
        if situation == SituationType.PENALTY:
            return "penalty_spot", random.choice(["right_foot", "left_foot"])

        if situation == SituationType.CORNER:
            zone = random.choices(
                ["six_yard_box", "inside_box", "edge_of_box"],
                weights=[35, 50, 15]
            )[0]
            body_part = random.choices(["head", "right_foot", "left_foot"], weights=[55, 25, 20])[0]
            return zone, body_part

        if situation == SituationType.DIRECT_FREEKICK:
            zone = random.choices(
                ["edge_of_box", "outside_box"],
                weights=[60, 40]
            )[0]
            body_part = random.choice(["right_foot", "left_foot"])
            return zone, body_part

        # Open play / fast break
        if situation == SituationType.FAST_BREAK:
            zone = random.choices(
                ["six_yard_box", "inside_box", "edge_of_box"],
                weights=[25, 55, 20]
            )[0]
        else:
            zone = random.choices(
                ["six_yard_box", "inside_box", "edge_of_box", "outside_box"],
                weights=[12, 45, 28, 15]
            )[0]

        body_part = random.choices(
            ["right_foot", "left_foot", "head"],
            weights=[45, 35, 20]
        )[0]

        return zone, body_part

    def _shot_location(self, zone: str) -> Tuple[float, float]:
        """
        Convert zone name to pitch coordinates.

        Checkpoint 6 fix: six_yard_box and inside_box previously sampled
        x up to 105.0, which IS the goal line — a shot "from" there means
        standing inside the goal itself, and was the source of goals
        appearing to be scored from the goal-kick line in exports. Capped
        at 104.3 so the closest possible shot is a plausible half-stride
        out, never literally on the line.

        Checkpoint 8 fix: further capped to 103.0 (2m from goal line) so the
        closest possible shot is a realistic distance from goal. A half-stride
        (0.7m) is still essentially on the line — no player shoots from there
        in real football. Also narrowed y ranges to avoid physically impossible
        acute angles where the goal is barely visible.
        """
        locations = {
            "six_yard_box":   (random.uniform(99, 103.0), random.uniform(29, 39)),
            "penalty_spot":   (94.0, 34.0),
            "inside_box":     (random.uniform(88, 103.0), random.uniform(24, 44)),
            "edge_of_box":    (random.uniform(83, 90), random.uniform(24, 44)),
            "outside_box":    (random.uniform(70, 83), random.uniform(20, 48)),
            "long_range":     (random.uniform(55, 70), random.uniform(15, 53)),
        }
        return locations.get(zone, (85.0, 34.0))


# ─────────────────────────────────────────────
# MATCH RESULT — What simulation returns
# ─────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    The complete output of a simulated match.
    Contains the full event timeline and final state.
    Stat accumulation and export happen in the next module.
    """
    config: MatchConfig
    state: MatchState
    timeline: List[MatchEvent]
    goals: List[MatchEvent]
    cards: List[MatchEvent]
    subs: List[MatchEvent]
    squads: Dict
    threat: ThreatEngine
    position_log: List[Dict] = field(default_factory=list)
    momentum_log: List[Dict] = field(default_factory=list)

    @property
    def home_goals(self) -> int:
        return self.state.home_goals

    @property
    def away_goals(self) -> int:
        return self.state.away_goals

    @property
    def score_str(self) -> str:
        return (
            f"{self.config.home_team} {self.home_goals}–"
            f"{self.away_goals} {self.config.away_team}"
        )

    @property
    def home_xg(self) -> float:
        return round(self.state.home_xg, 2)

    @property
    def away_xg(self) -> float:
        return round(self.state.away_xg, 2)

    def summary(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"  {self.score_str}",
            f"  xG: {self.config.home_team} {self.home_xg} — {self.away_xg} {self.config.away_team}",
            f"  Goals: {len(self.goals)} | Cards: {len(self.cards)}",
            f"  Timeline events: {len(self.timeline)}",
            f"  Added time: {self.state.added_time}'",
            f"{'='*50}",
        ]
        for g in self.goals:
            assist = f" (assist: {g.secondary_player})" if g.secondary_player else ""
            lines.append(f"  ⚽ {g.minute}' {g.player}{assist} — {g.team}")
        for c in self.cards:
            icon = "🟥" if c.event_type == EventType.RED_CARD else "🟨"
            lines.append(f"  {icon} {c.minute}' {c.player} — {c.team}")
        return "\n".join(lines)