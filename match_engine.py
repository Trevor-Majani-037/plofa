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
from types import SimpleNamespace
from enum import Enum, auto
from datetime import date

from position_engine import PositionEngine
from threat_engine import ThreatEngine
from block_awareness import BlockShape, BlockDetector
from virtual_gps import VirtualGPS
from weather_physics import WeatherCondition, WeatherPhysics
from tactical_shapes import FormationStance
from attack_patterns import AttackPattern
from set_piece_routines import SetPieceRoutine
from ball_vision import movement_ball
from run_tracking import IntendedRunRecorder, RunTracker

# The match narrative prints emoji/unicode; on legacy consoles (cp1252 etc.)
# that raises UnicodeEncodeError mid-simulation. Reconfigure the streams to
# fall back to '?' rather than crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")  # type: ignore
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# MANAGER BRAINS — MASTER SWITCH (Phase 6 → default since Phase 10)
# ─────────────────────────────────────────────────────────────
# Since 2026-09-20 the live brain-manager (manager_profile.Manager with
# .decide) is the DEFAULT: when a real Manager is injected via set_managers
# its decision drives the tactical dials, coach instructions, possession
# target, formation stance and sub urgency.  Every engine hook guards on
# isinstance(., Manager), so runners that inject NO manager (custom gates,
# probes, validate harnesses) are byte-identical to the pre-Phase-6 static
# logic — this flag only changes behaviour for managers that are wired.
USE_MANAGER_BRAIN = True

# Phase 10 — position off-ball press-gate brains (brains_offball/*.json).
# True by default; a per-player Bernoulli uses the LW/RW/CM evolved brains
# when present and falls back to the static role rate otherwise.
_OFFBALL_POS_BRAIN_WIRED = True


def _brain_possession_profiles(home_prof, away_prof, home_mgr, away_mgr):
    """Phase 8 — coach's say on ball share (opt-in, flag-gated).

    When the brain path is on and a live Manager is wired, wrap each raw
    profile in a proxy carrying the coach's CURRENT possession target
    (stored posture/pressing — no decide() poll). ``style`` and
    ``has_identity`` pass through untouched so the philosophy feedback
    (Lever A) and style premiums behave exactly as before. Flag OFF (or
    static managers) returns the raw profiles unchanged.
    """
    if not USE_MANAGER_BRAIN:
        return home_prof, away_prof
    from tactical_ai import brain_stored_possession_target as _brain_poss
    out_h, out_a = home_prof, away_prof
    _ht = _brain_poss(home_prof, home_mgr)
    if _ht is not None:
        out_h = SimpleNamespace(
            possession_target=_ht, style=home_prof.style,
            has_identity=getattr(home_prof, "has_identity", False))
    _at = _brain_poss(away_prof, away_mgr)
    if _at is not None:
        out_a = SimpleNamespace(
            possession_target=_at, style=away_prof.style,
            has_identity=getattr(away_prof, "has_identity", False))
    return out_h, out_a


# ─────────────────────────────────────────────────────────────
# MANAGER COLLECTION CHECKPOINT HOOK (Phase 7, opt-in)
# ─────────────────────────────────────────────────────────────
# Set by manager_collection.collect_manager_samples() to sample the
# static manager's sensor view every 5 minutes EVEN when no tactical
# decision fired that minute. None (default) = zero engine impact.
# Callable(engine, minute).
_collection_checkpoint_hook = None


# ─────────────────────────────────────────────────────────────
# SHARED XI PRESS-ENGAGEMENT CONTROLLER (TeamPressBrain)
# ─────────────────────────────────────────────────────────────
# ONE 24->32->32->1 brain drives the WHOLE unit's off-ball press Bernoulli:
#     prob = _PRESS_PROB[pos] * g     at _offball_move_player.
# g in [0,1] is the controller's engagement scalar for the current team-state
# (ball zone vs our goal, danger, unit density, shape compactness).  Default
# disabled (g = 1.0 -> pure role-rate Bernoulli, the heuristic baseline);
# call set_team_press_brain(brain) / load_team_press_brain(path) to engage.
_TEAM_PRESS_BRAIN = None          # shared TeamPressBrain or None
_TEAM_PRESS_AUTO = True           # auto-load BRAINS_TEAM_DIR/XI.json on first tick
_TEAM_PRESS_SENSOR_IMPORT = None  # cached extract_team_sensors callable
_TEAM_PRESS_DIR = "brains_team"   # per-project controller directory
_TEAM_PRESS_LOADED = False


def set_team_press_dir(path: str) -> None:
    """Override the controller directory (default "brains_team")."""
    global _TEAM_PRESS_DIR, _TEAM_PRESS_LOADED
    _TEAM_PRESS_DIR = path
    _TEAM_PRESS_LOADED = False


def set_team_press_auto(auto_load: bool) -> None:
    """Enable/disable auto-loading of BRAINS_TEAM_DIR/XI.json (default True)."""
    global _TEAM_PRESS_AUTO
    _TEAM_PRESS_AUTO = bool(auto_load)


def set_team_press_brain(brain) -> None:
    """Engage the shared XI press controller (None disables -> g = 1.0)."""
    global _TEAM_PRESS_BRAIN
    _TEAM_PRESS_BRAIN = brain


def load_team_press_brain(path: str):
    """Load a TeamPressBrain JSON and engage it for every match."""
    from football_brain import TeamPressBrain
    brain = TeamPressBrain.load(path)
    set_team_press_brain(brain)
    return brain


def _ensure_team_press_loaded() -> None:
    global _TEAM_PRESS_BRAIN, _TEAM_PRESS_LOADED
    if _TEAM_PRESS_LOADED:
        return
    _TEAM_PRESS_LOADED = True
    if _TEAM_PRESS_AUTO and _TEAM_PRESS_BRAIN is None:
        import os as _os
        _path = _os.path.join(_TEAM_PRESS_DIR, "XI.json")
        if _os.path.exists(_path):
            try:
                load_team_press_brain(_path)
            except Exception:
                pass


#: The seed value that was in force before this process applied any per-match
#: seed, or ``None`` if we never have. Tracking the *value* rather than a
#: snapshot of the config object matters: perception is a process-wide global
#: that callers (and the Streamlit UI) may legitimately reconfigure at runtime,
#: and restoring a stale object would silently clobber their change.
_PRE_SEED_VALUE: Optional[int] = None


def _apply_perception_seed(seed: Optional[int]) -> None:
    """Point the process-wide perception config at this match's noise seed.

    Perception is a process-wide global (``get_perception_config``), so a naive
    "set it and move on" would leak one match's seed into every later match in
    the same process — exactly the kind of cross-match contamination this whole
    investigation is about.

    Only this function's own change is ever undone. ``seed=None`` restores the
    value that was in force before the first seeded match; if no seeded match
    has run, it does nothing at all, so the legacy path is genuinely untouched.
    Every other perception field is preserved, because the restore replaces only
    the ``seed`` attribute of whatever config is current.
    """
    global _PRE_SEED_VALUE
    try:
        from dataclasses import replace as _replace
        from perception import get_perception_config, set_perception
        current = get_perception_config()

        if seed is None:
            if _PRE_SEED_VALUE is None:
                return                      # we never touched it
            set_perception(_replace(current, seed=_PRE_SEED_VALUE))
            _PRE_SEED_VALUE = None
            return

        if _PRE_SEED_VALUE is None:
            _PRE_SEED_VALUE = int(current.seed)
        set_perception(_replace(current, seed=int(seed)))
    except Exception:
        # Perception seeding is an enhancement, never a hard dependency: if it
        # cannot be applied the match still plays.
        pass


def _reset_perception_seed_tracking() -> None:
    """Forget any remembered pre-seed value (tests, and long-lived processes)."""
    global _PRE_SEED_VALUE
    _PRE_SEED_VALUE = None


def _team_press_sensors(engine: Any, team: str, ball_x: float,
                        ball_y: float, danger_t: float) -> np.ndarray:
    global _TEAM_PRESS_SENSOR_IMPORT
    if _TEAM_PRESS_SENSOR_IMPORT is None:
        from football_brain import extract_team_sensors
        _TEAM_PRESS_SENSOR_IMPORT = extract_team_sensors
    return _TEAM_PRESS_SENSOR_IMPORT(engine, team, ball_x, ball_y, danger_t)


def _team_press_g(engine: Any, team: str, ball_x: float, ball_y: float,
                  danger_t: float) -> float:
    """Engagement scalar for this team-tick: brain(g) or 1.0 (heuristic)."""
    _ensure_team_press_loaded()
    brain = _TEAM_PRESS_BRAIN
    if brain is None:
        return 1.0
    try:
        return float(brain.forward(_team_press_sensors(
            engine, team, ball_x, ball_y, danger_t)))
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────
# COGNITION OBSERVER (episodic-memory feed)
# ─────────────────────────────────────────────────────────────
# The TOLAND merge's memory tap.  When engaged, every absorbed match event
# is handed to the PlayerMinds of the players involved (cognition_brain
# installs its dispatcher here) so episodic memory fills from real match
# facts with no parallel simulation.  Disabled by default — zero cost and
# zero behaviour change until engaged.
_COGNITION_OBSERVER = None


def set_cognition_observer(fn) -> None:
    """Engage a per-event cognition observer (None disables)."""
    global _COGNITION_OBSERVER
    _COGNITION_OBSERVER = fn


def _cognition_observe(event) -> None:
    obs = _COGNITION_OBSERVER
    if obs is None:
        return
    try:
        obs(event)
    except Exception:
        return


# ─────────────────────────────────────────────
# DEFENSIVE-ACTION CONTROLLER (DefensiveActionBrain)
# ─────────────────────────────────────────────
# ONE 24->32->32->4 brain picks WHICH defensive act to deploy on a contest
# moment: tackle / interception / clearance / block.  It replaces the
# hand-tuned _danger_scaled_action_weights table AND the hardcoded
# "clearance" / recovery weights at the three defensive re/action sites:
#   * open-play contest      (DefensiveChain dispatch, _simulate_minute)
#   * direct clearance       (ball dead inside our defensive third)
#   * defensive recovery     (deep block after a turnover)
# Default: auto-load brains_def/ACTION.json on first use; disabled ->
# the heuristic selection acts as the baseline.  A PHYSICS safety grip stays:
# clearance/block far from our goal is refused (a "clearance" 60 m from your
# own line is just a turnover) — feasibility is also a sensor (s[19]) so the
# net learns to stay out of that zone on its own.
_DEF_ACTION_BRAIN = None           # shared DefensiveActionBrain or None
_DEF_ACTION_AUTO = True            # auto-load BRAINS_DEF_DIR/ACTION.json
_DEF_ACTION_DIR = "brains_def"     # per-project controller directory
_DEF_ACTION_LOADED = False
_DEF_ACTION_SENSOR_IMPORT = None   # cached extract_defensive_sensors callable
_DEF_ACTION_FEASIBLE = {           # physics gate zones (ball near OUR goal)
    True: lambda cx: cx > 35.0,    # attacks_right: our goal is x=0
    False: lambda cx: cx < 70.0,   # attacks_left : our goal is x=105
}


def set_defensive_action_dir(path: str) -> None:
    global _DEF_ACTION_DIR, _DEF_ACTION_LOADED
    _DEF_ACTION_DIR = path
    _DEF_ACTION_LOADED = False


def set_defensive_action_auto(auto_load: bool) -> None:
    """Enable/disable auto-loading of BRAINS_DEF_DIR/ACTION.json (default True)."""
    global _DEF_ACTION_AUTO
    _DEF_ACTION_AUTO = bool(auto_load)


def set_defensive_action_brain(brain) -> None:
    """Engage the shared defensive-action controller (None disables -> heuristic)."""
    global _DEF_ACTION_BRAIN
    _DEF_ACTION_BRAIN = brain


def load_defensive_action_brain(path: str):
    """Load a DefensiveActionBrain JSON and engage it for every match."""
    from football_brain import DefensiveActionBrain
    brain = DefensiveActionBrain.load(path)
    set_defensive_action_brain(brain)
    return brain


def _ensure_def_action_loaded() -> None:
    global _DEF_ACTION_BRAIN, _DEF_ACTION_LOADED
    if _DEF_ACTION_LOADED:
        return
    _DEF_ACTION_LOADED = True
    if _DEF_ACTION_AUTO and _DEF_ACTION_BRAIN is None:
        import os as _os
        _path = _os.path.join(_DEF_ACTION_DIR, "ACTION.json")
        if _os.path.exists(_path):
            try:
                load_defensive_action_brain(_path)
            except Exception:
                pass


def _def_action_sensors(engine: Any, defending_team: str, attacking_team: str,
                        ball_x: float, ball_y: float, danger_level: float,
                        ball_aerial: bool, ctx_x: float, ctx_y: float,
                        opponent_distance: float, press_occurred: bool) -> np.ndarray:
    global _DEF_ACTION_SENSOR_IMPORT
    if _DEF_ACTION_SENSOR_IMPORT is None:
        from football_brain import extract_defensive_sensors
        _DEF_ACTION_SENSOR_IMPORT = extract_defensive_sensors
    return _DEF_ACTION_SENSOR_IMPORT(
        engine, defending_team, attacking_team, ball_x, ball_y,
        danger_level=danger_level, ball_aerial=ball_aerial,
        contest_x=ctx_x, contest_y=ctx_y,
        opponent_distance=opponent_distance,
        press_occurred=press_occurred,
    )


def _def_action_choice(engine: Any, defending_team: str, attacking_team: str,
                       danger_level: float, ctx_x: float, ctx_y: float,
                       ball_aerial: bool, opponent_distance: Optional[float],
                       attacks_right: bool, ball_x: float, ball_y: float,
                       press_occurred: bool = False,
                       fallback: Optional[str] = None,
                       site: str = "contest") -> str:
    """Pick the defensive action type for a contest moment.

    Neural brain engaged -> softmax argmax over
    [tackle, interception, clearance, block].  Otherwise -> the heuristic
    _danger_scaled_action_weights table (or ``fallback`` for the legacy
    hardcoded sites).  A PHYSICS grip refuses clearance/block far from our
    goal (feasibility); if refused, the pick narrows to tackle/interception.
    ``site`` labels the call origin ("contest", "direct_clearance",
    "recovery") for collector probes — not a decision input.
    """
    _ensure_def_action_loaded()
    feasible_fn = _DEF_ACTION_FEASIBLE.get(bool(attacks_right))
    feasible = True
    if ctx_x is not None and feasible_fn is not None:
        feasible = feasible_fn(ctx_x)
    # clamp to goal-side region so the physics gate makes real sense
    own_goal_x = 105.0 if not attacks_right else 0.0
    proj = abs(ctx_x - own_goal_x) / 105.0 if ctx_x is not None else 1.0

    action = None
    brain = _DEF_ACTION_BRAIN
    if brain is not None:
        try:
            sens = _def_action_sensors(
                engine, defending_team, attacking_team,
                ball_x, ball_y, danger_level, ball_aerial,
                ctx_x, ctx_y, opponent_distance or 0.0, press_occurred)
            action = brain.predict(sens)
        except Exception:
            action = None
    if action is None:
        if fallback is not None:
            action = fallback
        else:
            action = random.choices(
                ["tackle", "interception", "clearance", "block"],
                weights=_danger_scaled_action_weights_static(danger_level),
            )[0]

    if action in ("clearance", "block") and not feasible:
        action = "tackle" if random.random() < 0.6 else "interception"
    return action


# module-level mirror of MatchEngine._danger_scaled_action_weights so the
# controller hook can fall back without an engine instance.
def _danger_scaled_action_weights_static(danger: float) -> List[float]:
    if danger >= 85:
        return [0.15, 0.08, 0.48, 0.29]
    if danger >= 60:
        return [0.20, 0.12, 0.44, 0.24]
    if danger >= 30:
        return [0.28, 0.25, 0.27, 0.20]
    return [0.32, 0.28, 0.22, 0.18]


# Dedicated RNG for cosmetic, non-football randomness (e.g. how long a
# goal celebration lasts). Kept separate from the global random stream on
# purpose: seeded match reproductions must draw the same football sequence
# with or without the presentation layer switched on.
#
# Checkpoint 37: this must be SEEDED. An unseeded instance drew from
# OS entropy, so the celebration length added into state.match_clock_s
# diverged run-to-run even under a fixed match seed — the clock shift then
# cascaded into minute/phase/added-time logic and broke seeded match
# reproduction. A fixed seed keeps celebration lengths deterministic (and
# still decoupled from the global football stream).
_COSMETIC_RNG = random.Random(0x5EEDC05)


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
    GOAL_CELEBRATION     = auto()   # Post-goal pause; adds 10-30s to the match clock
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
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra context

    def __post_init__(self):
        # The event chains stamp the foot/head used for passes, through balls
        # and crosses inside metadata["body_part"]. Promote it to the field so
        # every consumer (exporter "Footed Passes"/"Footed Events" sheets,
        # threat engine, header detection) sees the REAL body part instead of
        # the "right_foot" dataclass default. Shots already pass body_part as
        # a top-level kwarg, so this is a no-op for them (same value).
        md_body = (self.metadata or {}).get("body_part")
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


# CHRONOGRAPHY ─────────────────────────────────────────
# The engine runs on one continuous global clock (state.match_clock_s). This
# additive layer records the exact clock span each possession chain occupied,
# then stamps the timeline events with their TRUE match-second (the "minute" is
# only a bucket; most chain events carry a placeholder random second). Reading
# only — it changes no outcome and mutates nothing.

@dataclass
class ChainClockMark:
    """Clock span a single absorbed possession chain occupied on the global clock."""
    minute: int
    start_clock: float
    end_clock: float
    motion_folded: bool   # False = goal chain early-returned before _absorb_motion
    events: "list[MatchEvent]"


@dataclass
class TimedEvent:
    """A timeline event stamped with its real match-second from the global clock."""
    minute: int
    second: float                   # 0–59.9 within the minute bucket
    match_clock_s: float            # seconds from kickoff (true global clock)
    duration: float                 # measured spacing to the next chain event (0 for last)
    event_type: str
    team: str
    player: str
    secondary_player: Optional[str] = None
    location_x: float = 0.0
    location_y: float = 0.0
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    goal: bool = False
    source_event_index: int = -1    # index into MatchResult.timeline

    @property
    def stamp(self) -> str:
        return f"{self.minute}:{self.second:04.1f}"

    @property
    def is_goal(self) -> bool:
        return self.event_type in ("GOAL", "OWN_GOAL", "PENALTY_SCORED")


@dataclass
class MatchChronology:
    events: "list[TimedEvent]"
    match_duration_s: float
    measured_play_s: float     # sum of folded chain spans (real ball-in-play time)
    dead_time_s: float         # restarts, dead minutes, goal celebrations, etc.
    n_chains: int
    n_unmarked_events: int     # timeline events outside any chain mark (kickoffs, stoppage)

    @property
    def play_share(self) -> float:
        if self.match_duration_s <= 0.0:
            return 0.0
        return round(100.0 * self.measured_play_s / self.match_duration_s, 1)


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
    compactness: float = 0.5            # 0=spread, 1=narrow/packed block

    # Derived probabilities (set during __post_init__)
    possession_target: float = 50.0     # Natural possession tendency
    shots_per_sequence: float = 0.15    # Chance a possession sequence ends in shot
    big_chance_ratio: float = 0.35      # % of chances that are "big"
    press_success_rate: float = 0.25    # % of presses that win ball

    # ── Club Philosophy layer (Checkpoint 30) ───────────────────────
    # When a ClubPhilosophy is attached, _apply_philosophy() pulls the
    # style DNA knobs toward the philosophy's targets (weighted by
    # identity_strength = magnitude × adherence).  None-safe: when
    # philosophy is None every new field stays neutral and no existing
    # behaviour changes.
    philosophy: Optional[Any] = None     # ClubPhilosophy or None
    hunger: float     = 0.0             # Cruyff dial (0..1)
    patience: float   = 0.0             # circulate-don't-force (0..1)
    starve: float     = 0.0             # opponent-squeeze intensity (0..1)
    has_identity: bool = False           # True when a philosophy was applied

    def __post_init__(self):
        self._apply_style_dna()
        self._apply_philosophy()
        # CALIBRATION (2026-09-17): funnel shot volume calibrated AFTER the
        # punchy per-style profiles: the style DNA + philosophy mix already
        # scale shotgun intent by style, and the AttackChain range-settle +
        # geometry selector now land the REAL mix mix. A ×0.82 trim brings
        # the volume (14.7 shots/team measured) down to the real band
        # (12.1) without touching per-style differentiation. Distance
        # discipline lives in the chain, so the trim sheds mostly
        # box-perimeter volume.
        self.shots_per_sequence = round(max(0.04, self.shots_per_sequence), 4)

    def _apply_style_dna(self):
        """Map style enum to tactical DNA values."""
        style_profiles = {
            TeamStyle.ULTRA_ATTACKING: {
                'press_intensity': 0.8, 'defensive_line': 0.8,
                'width': 0.7, 'tempo': 0.9, 'directness': 0.7,
                'compactness': 0.35,
                'possession_target': 55.0, 'shots_per_sequence': 0.15,
                'big_chance_ratio': 0.40, 'press_success_rate': 0.30,
            },
            TeamStyle.ATTACKING: {
                'press_intensity': 0.65, 'defensive_line': 0.65,
                'width': 0.6, 'tempo': 0.7, 'directness': 0.6,
                'compactness': 0.40,
                'possession_target': 52.0, 'shots_per_sequence': 0.13,
                'big_chance_ratio': 0.37, 'press_success_rate': 0.27,
            },
            TeamStyle.GEGENPRESSING: {
                'press_intensity': 0.95, 'defensive_line': 0.75,
                'width': 0.6, 'tempo': 0.95, 'directness': 0.65,
                'compactness': 0.45,
                'possession_target': 50.0, 'shots_per_sequence': 0.14,
                'big_chance_ratio': 0.38, 'press_success_rate': 0.40,
            },
            TeamStyle.TIKI_TAKA: {
                'press_intensity': 0.72, 'defensive_line': 0.70,
                'width': 0.5, 'tempo': 0.55, 'directness': 0.25,
                'compactness': 0.55,
                'possession_target': 70.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.30, 'press_success_rate': 0.38,
            },
            TeamStyle.BALANCED: {
                'press_intensity': 0.50, 'defensive_line': 0.50,
                'width': 0.5, 'tempo': 0.55, 'directness': 0.50,
                'compactness': 0.50,
                'possession_target': 50.0, 'shots_per_sequence': 0.11,
                'big_chance_ratio': 0.33, 'press_success_rate': 0.25,
            },
            TeamStyle.DEFENSIVE: {
                'press_intensity': 0.30, 'defensive_line': 0.30,
                'width': 0.4, 'tempo': 0.40, 'directness': 0.55,
                'compactness': 0.65,
                'possession_target': 42.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.28, 'press_success_rate': 0.18,
            },
            TeamStyle.ULTRA_DEFENSIVE: {
                'press_intensity': 0.15, 'defensive_line': 0.15,
                'width': 0.35, 'tempo': 0.30, 'directness': 0.60,
                'compactness': 0.75,
                'possession_target': 35.0, 'shots_per_sequence': 0.07,
                'big_chance_ratio': 0.25, 'press_success_rate': 0.12,
            },
            TeamStyle.PARK_THE_BUS: {
                'press_intensity': 0.10, 'defensive_line': 0.10,
                'width': 0.30, 'tempo': 0.25, 'directness': 0.65,
                'compactness': 0.90,
                'possession_target': 32.0, 'shots_per_sequence': 0.04,
                'big_chance_ratio': 0.22, 'press_success_rate': 0.10,
            },
            TeamStyle.WING_PLAY: {
                'press_intensity': 0.55, 'defensive_line': 0.55,
                'width': 0.90, 'tempo': 0.65, 'directness': 0.60,
                'compactness': 0.35,
                'possession_target': 48.0, 'shots_per_sequence': 0.12,
                'big_chance_ratio': 0.35, 'press_success_rate': 0.22,
            },
            TeamStyle.ROUTE_ONE: {
                'press_intensity': 0.40, 'defensive_line': 0.40,
                'width': 0.55, 'tempo': 0.80, 'directness': 0.90,
                'compactness': 0.70,
                'possession_target': 38.0, 'shots_per_sequence': 0.09,
                'big_chance_ratio': 0.30, 'press_success_rate': 0.20,
            },
            TeamStyle.VERTICAL_TIKI_TAKA: {
                'press_intensity': 0.65, 'defensive_line': 0.65,
                'width': 0.55, 'tempo': 0.70, 'directness': 0.55,
                'compactness': 0.50,
                'possession_target': 60.0, 'shots_per_sequence': 0.12,
                'big_chance_ratio': 0.36, 'press_success_rate': 0.30,
            },
            TeamStyle.STRUCTURED_POSSESSION: {
                'press_intensity': 0.55, 'defensive_line': 0.55,
                'width': 0.5, 'tempo': 0.50, 'directness': 0.35,
                'compactness': 0.50,
                'possession_target': 62.0, 'shots_per_sequence': 0.09,
                'big_chance_ratio': 0.31, 'press_success_rate': 0.30,
            },
            TeamStyle.FLUID_COUNTER: {
                'press_intensity': 0.45, 'defensive_line': 0.40,
                'width': 0.65, 'tempo': 0.75, 'directness': 0.72,
                'compactness': 0.45,
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

    def _apply_philosophy(self):
        """Blend the attached ClubPhilosophy into the style DNA (Checkpoint 30).

        Pulls every tactical knob toward the philosophy's target by a weight
        of ``identity_strength = magnitude × adherence``.  Completely inert
        when ``philosophy`` is None — the profile is byte-for-byte the
        style-default it would have been before this layer existed.

        The mapped knobs (README of what a philosophy "owns"):
          * hunger      → possession_target (Cruyff: we must have the ball)
          * patience    → shots_per_sequence, recovery willingness
          * starve      → opponent sequence-length squeeze (Lever B)
        """
        phi = self.philosophy
        if phi is None:
            self.hunger = 0.0
            self.patience = 0.0
            self.starve = 0.0
            self.has_identity = False
            return

        w = max(0.0, min(1.0, getattr(phi, "identity_strength", 0.0)))
        if w <= 0.0:
            self.hunger = 0.0
            self.patience = 0.0
            self.starve = 0.0
            self.has_identity = False
            return

        def _mix(base: float, target: float) -> float:
            return base * (1.0 - w) + target * w

        self.hunger = round(min(1.0, max(0.0, getattr(phi, "hunger", 0.0))), 4)
        self.patience = round(min(1.0, max(0.0, getattr(phi, "patience", 0.0))), 4)
        self.starve = round(min(1.0, max(0.0, getattr(phi, "starve", 0.0))), 4)

        self.press_intensity = round(min(1.0, max(0.05, _mix(
            self.press_intensity, getattr(phi, "press_target", 0.5)))), 4)
        self.defensive_line = round(min(1.0, max(0.05, _mix(
            self.defensive_line, getattr(phi, "line_target", 0.5)))), 4)
        self.width = round(min(1.0, max(0.05, _mix(
            self.width, getattr(phi, "width_target", 0.5)))), 4)
        self.tempo = round(min(1.0, max(0.05, _mix(
            self.tempo, getattr(phi, "tempo_target", 0.5)))), 4)
        self.directness = round(min(1.0, max(0.05, _mix(
            self.directness, getattr(phi, "directness_target", 0.5)))), 4)

        # Cruyff conversion: hunger → possession_target (18 at 0 → ~75 at 1)
        hunger_target = min(76.0, 18.0 + self.hunger * 58.0)
        self.possession_target = round(_mix(self.possession_target, hunger_target), 1)
        self.shots_per_sequence = round(max(0.03, _mix(
            self.shots_per_sequence, getattr(phi, "shots_target", 0.12))), 4)

        self.has_identity = True


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
    weather: Union[WeatherCondition, str] = "clear"  # clear, rain, wind, fog or WeatherCondition
    start_time: Optional[str] = None   # Kickoff time (e.g. "12:30", "15:00", "20:00")
    weather_enabled: bool = False      # Toggle physics effects (default False for zero regression)

    # ── Competition context (Phase 2–3 world layer additions) ──────────
    # Everything below is additive metadata: defaults reproduce today's
    # 26/27 behaviour exactly, and no engine logic reads these fields yet.
    competition_id: Optional[str] = None      # world-level competition identity
    competition_type: Optional[str] = None    # "league" | "cup" | "continental"
    stage_name: Optional[str] = None          # e.g. "Knockout", "Group Stage"
    round_name: Optional[str] = None          # e.g. "Quarter-final", "MD 12"
    leg: Optional[str] = None                 # "first" | "second" | None
    extra_time: bool = False
    penalties: bool = False
    aggregate: Optional[Tuple[int, int]] = None   # running aggregate (home, away)
    away_goals_rule: bool = False
    importance: float = 1.0                   # 0..inf; 1 = ordinary league game
    substitution_rules: Optional[dict] = None

    # ── Perception seeding (opt-in; None = exactly today's behaviour) ──────
    # ``PerceptionConfig.seed`` is 0 by default and nothing changed it, so the
    # noise a player gets from misreading the ball is byte-identical in every
    # match of every season — the same striker at the same minute produced the
    # same misread against every opponent. Setting this seeds perception from
    # the fixture instead: the same fixture replays identically, different
    # fixtures differ. ``None`` leaves the legacy behaviour untouched, which is
    # what keeps 26/27 output byte-identical.
    perception_seed: Optional[int] = None

    # ── Continuous physics (opt-in; False = exactly today's behaviour) ────
    # The physics layer lives in ``physics/`` and is purely additive: it answers
    # "how long does this physically take?" and never decides what should
    # happen. Defaulting to False is what keeps 26/27 output byte-identical —
    # with this off the engine never constructs a PhysicsWorld and never calls
    # the adapter, so there is nothing to change the existing code path.
    physics_enabled: bool = False
    #: record impossible-movement violations instead of silently tolerating
    #: them. Off by default: a season must not abort on one bad frame.
    physics_strict: bool = False


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

    # Manager Brains (Phase 6): the team credited with the MOST RECENT goal
    # (goal_team, NOT the own-goal-scoring defender's side). Stamped by
    # _absorb_chain when a goal chain lands, read by Manager.on_event to tell
    # on_goal_scored from on_goal_conceded. None when no goal has happened
    # (or a VAR-disallowed one was just cancelled).
    last_goal_team: Optional[str] = None

    # Who has the ball right now
    possession_team: str = ""

    # Feature #1/#2 — the live per-team formation stance (chasing shape, see-
    # it-out block ...) and attack pattern (overload side, box midfield ...)
    # for THIS minute. Set once per minute by _run_minute()/_simulate_minute(),
    # consumed by the possession/off-ball layers and read in exports/analytics.
    team_stances: Dict[str, FormationStance] = field(default_factory=dict)
    team_patterns: Dict[str, AttackPattern] = field(default_factory=dict)

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

    # Weather physics state
    weather: Optional[WeatherCondition] = None
    weather_enabled: bool = False

    # Possession-time tracking (Checkpoint — measured seconds held, vs the
    # probabilistic possession split used to award sequences). Drives real
    # possession % and per-player possession minutes.
    home_possession_s: float = 0.0
    away_possession_s: float = 0.0
    possession_time_by_player: Dict[str, float] = field(default_factory=dict)

    # Causal possession rights (Checkpoint — possession is not a memoryless
    # coin flip). When a team WINS the ball in open play it is OWED the next
    # open-play sequence; only a restart (kickoff / corner / penalty / goal
    # kick / throw-in / offside free kick) redirects the ball away from them.
    # Single-shot: consumed-and-cleared at the sequence team decision, so the
    # carry never stacks across turnovers.
    possession_winner: str = ""

    # Global continuous match timeline (Checkpoint — the macro loop is still
    # event/sequence-driven, but every chain's per-tick ball_path is offset
    # onto ONE monotonic 90' clock here, so the whole match reads as one
    # continuous, time-ordered ball trajectory instead of disconnected episodes.
    match_clock_s: float = 0.0
    match_ball_path: List[Dict[str, Any]] = field(default_factory=list)

    # Passive 5 Hz player-position recorder (reads position_engine.states only;
    # never mutates, never consumes RNG, changes no outcome).  keyed by player
    # name -> [(t, x, y)...] on the global clock; player_team maps name->team.
    player_path: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=dict)
    player_team: Dict[str, str] = field(default_factory=dict)

    # Added time (decided at ~88th minute)
    added_time: int = 0

    # ── Extra time + penalty shootout (audit §16 phase 5) ──────────────
    # ALL of the following default to "this match had neither". The engine only
    # reads or writes them when config.extra_time / config.penalties are set, so
    # a 26/27 league match takes exactly the same code path it always did and
    # produces byte-identical output (plan §19 / audit §17).
    in_extra_time: bool = False
    went_to_extra_time: bool = False
    extra_time_minutes: int = 0
    shootout_played: bool = False
    home_pens: int = 0
    away_pens: int = 0
    shootout_winner: str = ""      # "" until decided; then a team NAME
    shootout_rounds: List[Dict[str, Any]] = field(default_factory=list)

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

    # Foul-awarded free kicks. Before this, `pending_offside_fk_for` was the
    # ONLY queue feeding `_freekick_chain`, so every free kick in a match was
    # an offside restart and ordinary fouls (~22/match) awarded cards and
    # penalties but never a kick. Measured (_diag_fk_site.py): 13 of 13 free
    # kicks came from the offside queue, 0 from open play, 0 reached the
    # direct branch. That left the whole direct-free-kick path — and any wall
    # built on it — unreachable. A counter rather than a single slot, so two
    # fouls in one minute cannot overwrite each other (the same reasoning as
    # pending_corners_*).
    pending_foul_fk_home: int = 0
    pending_foul_fk_away: int = 0
    pending_foul_fk_x: float = 0.0
    pending_foul_fk_y: float = 0.0

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

    # ── Counterpress burst (P2) ────────────────────────────────────────
    # When a team loses possession, it briefly presses the recovery zone
    # (where the ball was lost) at elevated intensity for
    # COUNTERPRESS_WINDOW_S seconds of match clock, overriding the normal
    # static engagement line within COUNTERPRESS_RANGE_M of that zone.
    # Models the real-football pattern where the team that lost the ball
    # immediately hunts to win it back before the opponent can settle.
    COUNTERPRESS_WINDOW_S: float = 8.0
    COUNTERPRESS_RANGE_M: float = 18.0
    COUNTERPRESS_INTENSITY_MULT: float = 2.5
    counterpress_team: str = ""
    counterpress_until_s: float = -1.0
    counterpress_x: float = 0.0
    counterpress_y: float = 0.0

    def counterpress_active(self, team: str) -> bool:
        """True if *team* should be counterpressing right now."""
        return (
            self.counterpress_team == team
            and self.match_clock_s <= self.counterpress_until_s
            and self.counterpress_team != ""
        )

    def set_counterpress(self, team: str, x: float, y: float) -> None:
        """Arm the counterpress burst for *team* anchored at (x, y)."""
        self.counterpress_team = team
        self.counterpress_until_s = self.match_clock_s + self.COUNTERPRESS_WINDOW_S
        self.counterpress_x = x
        self.counterpress_y = y

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
        """Goal dramatically shifts momentum — with diminishing returns when already dominant.

        The 1st goal in a level match causes a full swing. But a 4th consecutive goal
        when momentum is already pinned near ±80 adds little extra — the dominance is
        already fully priced in. This prevents the momentum snowball from permanently
        locking out the losing team's attacking numbers.
        """
        shift = random.uniform(18, 30)   # Big momentum swing
        if scoring_team == home_team:
            current = state.momentum
            # Diminishing returns when already dominating
            if current >= 60:
                shift *= 0.50
            elif current >= 40:
                shift *= 0.75
            return min(100, current + shift)
        else:
            current = state.momentum
            # Diminishing returns when already dominating (away)
            if current <= -60:
                shift *= 0.50
            elif current <= -40:
                shift *= 0.75
            return max(-100, current - shift)

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
        Momentum naturally decays toward 0 (equilibrium).

        NOTE: the decay baseline is kept at 0 so the two teams are treated
        symmetrically. A previous version hard-seeded the baseline at +3.0
        (a permanent home pull), which — combined with the goal-driven
        momentum snowball — made home teams win almost every match and left
        away teams (and draws) statistically unable to happen. Any genuine
        home advantage should come from `config.home_advantage`, not from a
        one-sided momentum seed.
        """
        decay_rate = 0.08  # 8% decay per event toward baseline
        baseline = 0.0

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
    def get_game_state_modifier(state: MatchState, team: str, home_team: str,
                                composure: float = 1.0) -> float:
        """
        Teams react differently based on scoreline.
        Losing teams push forward (↑ attack chance), winning teams hold (↓ attack chance).

        Collapse penalty: a team losing 3+ goals doesn't get an ever-growing desperate
        boost — in real football their shape breaks down, morale collapses, and they
        create FEWER chances despite opening up. Each extra goal beyond -2 chips away
        at the desperation boost.

        `composure` (default 1.0): a manager's man-management fingerprint. A
        strong man-manager (>1.0) lifts a side that's behind so they don't
        collapse as fast; a weak one (<1.0) lets heads drop sooner. Applied
        only when the team is losing — a manager's steadiness matters most
        exactly when a team is on the ropes.
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
            base *= composure   # man-management steadies a chasing side
        else:
            # Desperate — but cap the boost based on how badly they're losing.
            # A team down 3+ has a broken shape and collapsing morale; they are
            # NOT generating 35% more creative chances — they're scrambling.
            deficit = abs(gd)
            boost = 1.35 if late_game else 1.15
            # Each extra goal beyond -2 reduces the desperate boost
            # -2: full boost (1.35), -3: ×0.88, -4: ×0.76, -5: ×0.64, -6: ×0.65 floor
            collapse_penalty = max(0.65, 1.0 - (deficit - 2) * 0.12)
            base = boost * collapse_penalty
            # A great man-manager holds the structure together and softens the
            # collapse penalty even at 3+ down (never lets it floor below ~0.72).
            base *= composure

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
        "six_yard_box":    0.48,
        "penalty_spot":    0.65,
        "inside_box":      0.13,
        "edge_of_box":     0.04,
        "outside_box":     0.02,
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

        # ── DISTANCE SHARPENING (CALIBRATION 2026-09-17) ──────────────
        # PLOFA's zone bands are coarse — "inside_box" spans 6-22m and
        # "edge_of_box" 22-35m — so the SAME zone price is applied to a
        # 9m tap-in and a 21m pile-driver. Real conversion decays steadily
        # with distance, so once the exact anchor is known we re-sharpen:
        #   >27m  → ×0.22    22-27m → ×0.41
        #   17-22m→ ×0.57     ≤17m   → unchanged
        if shot_x is not None and shot_y is not None:
            sx = abs(goal_x - shot_x)
            sy = abs(shot_y - 34.0)
            sdist = (sx * sx + sy * sy) ** 0.5
            if sdist >= 27.0:
                base *= 0.22
            elif sdist >= 22.0:
                base *= 0.41
            elif sdist >= 17.0:
                base *= 0.57

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

        # ── LEVER A — CLOSED-LOOP OWNERSHIP (Checkpoint 30) ─────────
        # When at least one side carries a club philosophy, the split is no
        # longer an open-loop starter prior: it feeds the ACTUAL measured
        # possession time back in, so a team that genuinely holds the ball
        # longer gets reinforced — real possession is a feedback loop
        # ("we have it → we keep it").  trust ramps up as the match wears
        # on (more measured samples), so the early game leans on the prior
        # and the identity converges the measured outcome onto its target.
        # Fully inert when no philosophy is present (byte-compatible).
        if (getattr(home_profile, "has_identity", False)
                or getattr(away_profile, "has_identity", False)):
            meas_total = state.home_possession_s + state.away_possession_s
            if meas_total >= 30.0:
                meas_home = state.home_possession_s / meas_total
                _trust = max(0.30, 0.55 - 0.00014 * meas_total)
                home_base = (1.0 - _trust) * home_base + _trust * meas_home
                away_base = 1.0 - home_base

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
    def sequence_length(team_profile: TeamProfile | EffectiveTactics,
                        state: MatchState,
                        oppressor_starve: float = 0.0) -> int:
        """
        How many passes in a typical possession sequence for this team.
        Tiki-taka teams have long sequences, route one teams have short ones.
        Accepts both TeamProfile and EffectiveTactics (which has a .style attr).

        Checkpoint 30 — two philosophy levers ride on top of the style roll:
          * Lever C (patience): a patient identity keeps the ball longer,
            stretching its own sequences (up to ~+45%).
          * Lever B (starve): the OPPONENT's hunger squeezes our retention
            (up to −35% from a maximal-starve defender, e.g. Gegenpressing).
        Both are inert when the profiles carry no philosophy (0.0 knobs).
        """
        # Resolve style from profile — TeamProfile has .style, EffectiveTactics
        # stores it as an attribute if created from adjust().
        style = getattr(team_profile, "style", None)
        # Owner-side patience + opponent-side starve (None-safe)
        patience = max(0.0, min(1.0, getattr(team_profile, "patience", 0.0) or 0.0))
        starve = max(0.0, min(1.0, oppressor_starve or 0.0))
        scale = (1.0 + 0.45 * patience) * (1.0 - 0.35 * starve)

        if style is None:
            # Fallback for EffectiveTactics: use possession_target as proxy
            return max(1, int(round(random.randint(3, 8) * scale)))
        # Checkpoint 23: ranges for possession-capable styles sit slightly
        # higher than they historically did. That is only realistic NOW —
        # the tempo-circulation directive lets a long sequence hover in the
        # middle third instead of marching box-to-box, so a 14-pass spell
        # looks like City circulating rather than a conveyor belt to a shot.
        # Direct/defensive styles are untouched: their short sequences ARE
        # their identity.
        #
        # CALIBRATION (2026-09-17): build-up ranges raised ~40% for the
        # possession-capable styles so per-team pass volume climbs toward
        # the real La Liga band (~530-550). Real teams cycle the ball far
        # more than PLOFA did (398 vs 538 passes/team). Tempo-circulation
        # keeps the extra touches in build-up rather than turning them into
        # shots.
        base_length = {
            TeamStyle.TIKI_TAKA:           random.randint(13, 28),
            TeamStyle.STRUCTURED_POSSESSION: random.randint(11, 22),
            TeamStyle.VERTICAL_TIKI_TAKA:  random.randint(9, 18),
            TeamStyle.ATTACKING:           random.randint(9, 18),
            TeamStyle.BALANCED:            random.randint(8, 16),
            TeamStyle.GEGENPRESSING:       random.randint(6, 12),
            TeamStyle.FLUID_COUNTER:       random.randint(5, 11),
            TeamStyle.DEFENSIVE:           random.randint(4, 9),
            TeamStyle.WING_PLAY:           random.randint(8, 16),
            TeamStyle.ULTRA_ATTACKING:     random.randint(8, 16),
            TeamStyle.ROUTE_ONE:           random.randint(1, 4),
            TeamStyle.PARK_THE_BUS:        random.randint(1, 4),
            TeamStyle.ULTRA_DEFENSIVE:     random.randint(1, 3),
        }.get(team_profile.style, random.randint(4, 10))

        return max(1, int(round(base_length * scale)))


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

    # ── Continuous physics: class-level defaults ───────────────────────
    # Declared here rather than only in _init_physics because
    # _initialize_simulation() runs from simulate(), not __init__. Without a
    # class-level default, `engine.physics` would raise AttributeError on a
    # freshly constructed engine — and "physics is off" must be a value you can
    # ask about, not an absence you have to guard.
    physics = None
    physics_unavailable = False


    # Causal possession carry (Checkpoint): how often a team that just WON
    # the ball gets the next open-play sequence. Probabilistic (not absolute)
    # so the engine still exercises its possession-target weighting and the
    # calibrated split (test_calibration 20-80%) stays enforceable — the carry
    # hands the ball back to the recovering side in real life, but a loose
    # challenge / second ball can still flip it straight back.
    POSSESSION_CARRY_PROB: float = 0.90

    def __init__(
        self,
        config: MatchConfig,
        home_profile: TeamProfile,
        away_profile: TeamProfile,
    ):
        self.config = config
        self.home_profile = home_profile
        self.away_profile = away_profile

        # ── Which block does each team hold? ──────────────────────
        # EVERY team gets a defensive block; only the AREA differs. The
        # discipline is one rule — "if you do not have the ball, do not leave a
        # space you can be put into" — expressed as three lines with explicit
        # depths, so the gaps between them are named numbers rather than
        # whatever the shape engine happens to produce.
        #
        # Derived from the team's own authored style via the same test
        # `pressing_profiles.profile_for_style` already uses, so there is one
        # source of truth for "how deep does this team sit". An earlier
        # version of this work gated the shape on a boolean that was true only
        # for LOW_BLOCK_CONTAIN, which meant a mid-block or high-block team
        # defended with the generic 7%-lateral shape — i.e. with no shape at
        # all, and with both half-spaces open.
        self._defensive_block: Dict[str, str] = {}
        try:
            from pressing_profiles import (PressingProfile,
                                           profile_for_style)
            _HEIGHT = {
                PressingProfile.LOW_BLOCK_CONTAIN: "low",
                PressingProfile.MID_BLOCK_TRAP: "mid",
                PressingProfile.ULTRA_HIGH_GEGENPRESS: "high",
            }
        except Exception:
            _HEIGHT = {}
        for _team, _profile in ((self.config.home_team, home_profile),
                                (self.config.away_team, away_profile)):
            if _profile is None:
                continue
            try:
                self._defensive_block[_team] = _HEIGHT.get(
                    profile_for_style(_profile.style.value),
                    PositionEngine.BLOCK_DEFAULT)
            except Exception:
                # No press profile available: fall back to the middle block.
                # Absent enhancement, never a failed match.
                self._defensive_block[_team] = "mid"
        #: Kept for callers and tests that ask "is this team parking the bus?",
        #: which is now a narrower question than "does it hold a block".
        self._low_block_teams: Dict[str, bool] = {
            t: (p == "low") for t, p in self._defensive_block.items()}

        # Reseed the cosmetic RNG per match. _COSMETIC_RNG is a module-level
        # singleton seeded only at import, and celebration_s (line ~4382) is
        # ADDED to match_clock_s — so without a per-match reseed, celebration
        # lengths drawn during match N leak into match N+1's clock inside the
        # same process and break deterministic replay for any harness that
        # runs several matches in one process (gate_xl, validate_neural_xl,
        # compare_striker, brain_self_trainer, ...). Reseeding to the same
        # fixed constant means every match starts from the same imported
        # state: byte-identical to a fresh process, and byte-identical across
        # in-process repeats.
        _COSMETIC_RNG.seed(0x5EEDC05)

        self.state = MatchState(
            possession_team=config.home_team  # Home team kicks off
        )

        # Resolve weather condition & toggle
        if isinstance(config.weather, str):
            self.weather_condition = WeatherCondition.from_string(config.weather)
        elif isinstance(config.weather, WeatherCondition):
            self.weather_condition = config.weather
        else:
            self.weather_condition = WeatherCondition.clear()

        self.state.weather = self.weather_condition
        self.state.weather_enabled = getattr(config, "weather_enabled", False)

        # Set active weather in WeatherPhysics engine
        WeatherPhysics.set_active_weather(
            self.weather_condition,
            enabled=self.state.weather_enabled,
        )

        self.timeline: List[MatchEvent] = []   # The complete match history
        # Chronography substrate: the exact global-clock span of every absorbed
        # possession chain. Read by MatchEngine.chronograph() after the whistle.
        self._chain_clock_marks: List[ChainClockMark] = []
        # Feature #1/#2 — match-clock instant each team's shape (stance or
        # pattern) last changed, so the off-ball integrator can animate the
        # actual RESHAPE instead of teleporting bodies at a minute boundary.
        self._shape_apply_clock: Dict[str, float] = {}
        self.squads: Dict[str, List] = {}      # {team_name: [Player objects]}
        self.active_players: Dict[str, List] = {}  # Currently on pitch

        # Accumulators (filled during simulation, read during export)
        self.event_counts: Dict[str, Dict] = {}
        # Checkpoint 39 — off-ball run tracking (six Gradient-style types:
        # advance/overlap/underlap/far_side/forward/support), sampled at the
        # 10 Hz integrator from both live and dead possession windows.
        self.run_tracker = RunTracker()
        # Intended runs: what the behaviour engines DECIDED, as distinct from
        # what RunTracker observes from the movement afterwards. The mode used
        # to be discarded at the position-layer boundary, which left the
        # exporter guessing from geometry.
        self.intended_runs = IntendedRunRecorder()
        self.goals: List[MatchEvent] = []
        self.cards: List[MatchEvent] = []
        self.subs: List[MatchEvent] = []

        # Squad manager — wired in via set_stamina_controller()
        self.sub_controller = None   # SubstitutionController or None

        # Managers (optional bias layer) — wired in via set_managers().
        self.home_manager = None
        self.away_manager = None

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

        # Unified-timeline carry state: the last ball position folded onto the
        # global ball path, so each successive action's synthesized segment
        # starts exactly where the previous one ended (continuity across
        # actions). Seeded at the centre circle (kickoff / post-goal reset).
        self._last_path_x: float = 52.5
        self._last_path_y: float = 34.0
        self._last_path_t: float = 0.0

        # Single-clock continuity support: every off-ball player is integrated
        # at 10 Hz for the whole match (not once per minute), and real
        # path-distance + sprint segments are accumulated into PositionEngine.
        self._on_ball_this_minute: set = set()
        self._sprint_state: Dict[str, Dict[str, bool]] = {}
        self._patrol: Dict[str, float] = {}
        self._chase_state: Dict[str, Dict[str, float]] = {}
        self._minute_start_snapshot: Dict[str, Tuple[float, float]] = {}
        self._top_speed_cache: Dict[str, float] = {}
        self._minute_start_clock: float = 0.0
        # Team-press engagement cache: {(team, tick_key) -> g} so the shared
        # controller's forward pass runs once per team per off-ball tick.
        self._team_press_g_cache: Dict[Tuple[str, float], float] = {}
        self._offball_press_cache: Dict[Tuple[str, float, float, float],
                                        Optional[float]] = {}
        # Team-wide per-TICK decisions taken once per team and then read by
        # every player in that team, because they are judgements about the
        # team and resolving them per player would give different answers
        # inside one tick (positions move as the tick integrates).
        self._offball_tick_seq: int = 0
        self._rest_defence_seq: int = -1
        self._rest_defence_cache: Dict[str, Optional[str]] = {}
        # Striker runs are cached per (minute, ball zone), not per tick: at
        # 10 Hz the run mode would flicker, and once per minute the striker
        # could not answer a change of situation. The key carries no RNG.
        self._striker_run_cache: Dict[Tuple[str, int, int],
                                      Dict[str, Tuple[float, float, float]]] = {}

        # Virtual GPS recorder — a 10 Hz per-tick position log that the
        # off-ball integrator feeds for verification/visualisation. Disabled
        # by default (zero overhead); turn on via enable_virtual_gps().
        self.gps: Optional[VirtualGPS] = None

    def enable_virtual_gps(self, tick_s: float = 0.1) -> VirtualGPS:
        """Enable the per-tick GPS recorder and return it. If already enabled
        returns the existing recorder. Must be called before simulate()."""
        if self.gps is None:
            self.gps = VirtualGPS(tick_s=tick_s)
        return self.gps

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

    def _avg_stamina(self, team_name: str) -> float:
        """Mean current stamina of a team's active XI (100.0 when no sub
        controller is wired). Drives the fatigue factor in the tactical
        shape layer, identical to the per-sequence closure used in
        _simulate_minute()."""
        if not self.sub_controller:
            return 100.0
        players = self.active_players.get(team_name, [])
        staminas = [
            self.sub_controller.stamina[p.name].current_stamina
            for p in players
            if getattr(p, "name", "") in self.sub_controller.stamina
        ]
        return sum(staminas) / len(staminas) if staminas else 100.0

    def _rest_defence_violator(
        self, team: str, ball_x: Optional[float], has_ball: bool,
    ) -> Optional[str]:
        """Positional-play rest defence, resolved ONCE PER TICK per team.

        Returns the name of the one outfield player to hold behind the ball, or
        None when the invariant already holds. None is the common case: in any
        ordinary shape the centre-backs and pivot are behind the ball, so this
        is a backstop against the whole team being beyond it, not a force that
        reshapes play every tick.

        Suppressed inside a low block, where the anchor substitution upstream
        already puts the entire unit behind the ball and the invariant cannot
        be violated. The block is a shape, and the block wins — the same
        precedence CK35's pitch-stretch rule gives it.
        """
        if not has_ball:
            return None
        if self._low_block_teams.get(team, False):
            return None
        seq = self._offball_tick_seq
        if seq != self._rest_defence_seq:
            self._rest_defence_seq = seq
            self._rest_defence_cache = {}
        if team not in self._rest_defence_cache:
            self._rest_defence_cache[team] = \
                self.position_engine.rest_defence_violator(
                    team, ball_x, has_ball,
                    self.position_engine.team_attacks_right.get(team, True),
                )
        return self._rest_defence_cache[team]

    def _striker_runs(
        self, team: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Striker run targets, decided per (minute, ball zone) per team.

        The cadence is the load-bearing part, and it is a trade-off between two
        failures. Per tick, the run mode flickers ten times a second and the
        decision is noise. Once per minute, the striker commits for a whole
        minute and cannot answer a change of situation - he would still be
        running in behind after the ball turned over at his feet.

        Keying on the ball's zone as well as the minute splits the difference:
        the decision re-rolls when the ball has moved a quarter of the pitch,
        which is a coarse proxy for "the situation materially changed", and is
        free because the run layer draws no global RNG (see
        ``PositionEngine._deterministic_rng``).

        In-possession only: out of possession decide_run declines anyway, but
        the guard is here so a stale cache can never steer a striker while his
        team is defending.
        """
        if not has_ball or ball_x is None:
            return {}
        attacks_right = self.position_engine.team_attacks_right.get(team, True)
        minute = self.state.minute
        nx = ball_x if attacks_right else 105.0 - ball_x
        zone = int(max(0.0, min(104.9, nx)) // 15.0)     # 7 buckets, own goal first
        key = (team, minute, zone)
        if key not in self._striker_run_cache:
            # The opponent's BlockShape, for the CM's orbit channels. The
            # shapes are refreshed once a minute by _update_block_shapes and
            # read here, not recomputed, so the CM orbits the block it can
            # actually see rather than one rebuilt per tick.
            opp_block = None
            if team == self.config.home_team:
                opp_block = self.state.away_block
            elif team == self.config.away_team:
                opp_block = self.state.home_block
            merged = dict(self.position_engine.striker_run_targets(
                team, ball_x, ball_y, has_ball, attacks_right,
                self._avg_stamina(team), key,
            ))
            merged.update(self.position_engine.offball_run_targets(
                team, ball_x, ball_y, has_ball, attacks_right,
                opp_block, minute,
            ))
            # Record the DECISION here, on the cache miss, which is exactly
            # once per real decision rather than once per 10 Hz tick. The
            # positional effect is applied by the caller; this is the intent,
            # and the two are only comparable because they are counted apart.
            for _name, _entry in merged.items():
                if len(_entry) == 4:
                    _st = self.position_engine.states.get(_name)
                    # The target and the player's CURRENT position go in with
                    # the mode: a decision that does not ask him to go
                    # anywhere is a positioning, not a run, and the recorder
                    # needs the geometry to tell those apart (see
                    # IntendedRunRecorder.MOVE_MIN_M).
                    self.intended_runs.record(
                        _name, getattr(_st, "position", ""), _entry[3],
                        _entry[1], _entry[2],
                        getattr(_st, "current_x", None),
                        getattr(_st, "current_y", None))
            self._striker_run_cache[key] = merged
        return self._striker_run_cache[key]

    def _sp_routine(self, team_name: str, situation: SituationType,
                    minute: int, players: list, fk_context=None):
        """Feature #3 — the committing-side's set-piece routine for a dead
        ball. Emergent from manager identity + squad aerial profile + score.
        Corner/crossed dead balls pick a corner routine; direct-range free
        kicks pick a free-kick scheme. Penalties and out-of-range kicks are
        None (engine default behaviour)."""
        from set_piece_routines import (
            aerial_presence, corner_routine_for, freekick_routine_for,
        )
        if situation == SituationType.PENALTY or not players:
            return None
        profile = self.home_profile if team_name == self.config.home_team \
            else self.away_profile
        style = getattr(profile, "style", None)
        style_name = style.value if style is not None else ""
        state = self.state
        gd = state.goal_difference if team_name == self.config.home_team \
            else -state.goal_difference
        chasing = gd <= -2 and state.minute >= 60
        protecting = gd >= 1 and state.minute >= 70
        if situation in (SituationType.CORNER, SituationType.CROSSED_FREEKICK):
            aerial = aerial_presence(players)
            return corner_routine_for(
                style_name, state, team_name, self.config.home_team, minute,
                aerial_score=aerial, chasing=chasing, protecting=protecting,
            )
        if situation == SituationType.DIRECT_FREEKICK:
            if fk_context is not None and fk_context[0] is not None:
                fk_x = max(2.0, min(103.0, fk_context[0]))
            else:
                fk_x = max(2.0, min(103.0, state.last_ball_x))
            attacks_right = (team_name == self.config.home_team)
            direct_range = fk_x > 78 if attacks_right else fk_x < 27
            taker_quality = 0.5
            from event_chain import SetPieceChain
            taker = SetPieceChain._pick_sp_taker(
                players, situation="freekick",
                freekick_type=("direct" if direct_range else "crossed"))
            if taker is not None:
                taker_quality = getattr(
                    getattr(taker.dna, "technical", None), "free_kick", 50.0) / 100.0
            return freekick_routine_for(
                style_name, state, team_name, self.config.home_team, minute,
                direct_range=direct_range, taker_free_kick=taker_quality,
                chasing=chasing, protecting=protecting,
            )
        return None

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
        if hasattr(controller, "set_weather") and getattr(self.state, "weather_enabled", False):
            controller.set_weather(self.weather_condition)
        # Register all starters
        for team, players in self.active_players.items():
            for p in players:
                starting = 100.0
                if hasattr(p, 'dna') and hasattr(p.dna, '_starting_stamina'):
                    starting = p.dna._starting_stamina
                controller.register_player(p, starting_stamina=starting)

    def set_managers(self, home_manager=None, away_manager=None):
        """Wire in a manager for the bias / brain layer.

        Phase 0–5 behaviour: ``ManagerProfile`` objects (duck-typing
        ``stubbornness()``, ``risk_tolerance``, ``chase_shift()``,
        ``protect_shift()``, ``man_management``).

        Phase 6 behaviour: real ``Manager`` objects (brain + mind +
        memory).  When ``USE_MANAGER_BRAIN`` is True the engine delegates
        to ``Manager.decide`` / ``on_event`` / ``end_of_match`` instead of
        the static TacticalAI posture thresholds.
        """
        self.home_manager = home_manager
        self.away_manager = away_manager

    def _credit_possession(
        self,
        team: str,
        seconds: float,
        player_map: Optional[Dict[str, float]] = None,
    ) -> None:
        """Accumulate measured possession time (seconds) for a team/players.

        Called after each possession sequence resolves. `seconds` is the
        episode's continuous 0.1 s clock elapsed during that team's spell on
        the ball — a real measure, not the probabilistic split used to decide
        who gets the next sequence.
        """
        if not seconds or seconds <= 0:
            return
        if team == self.config.home_team:
            self.state.home_possession_s += seconds
        else:
            self.state.away_possession_s += seconds
        if player_map:
            by_player = self.state.possession_time_by_player
            for name, secs in player_map.items():
                by_player[name] = by_player.get(name, 0.0) + secs

    def _arm_possession_winner(self, winner: str, result) -> None:
        """Causal possession carry (Checkpoint).

        A clean open-play turnover hands the ball to the recovering team for
        the NEXT sequence. Restart-bound losses don't arm a carry — the ball
        is dead, and the restart obligation consumed at the top of the next
        sequence (corner / penalty / goal kick / throw-in / offside free
        kick) redirects possession instead.
        """
        if getattr(result, "corner_won", False):
            return
        if getattr(result, "restart_required", False):
            return
        if getattr(result, "offside_detected", False):
            return
        self.state.possession_winner = winner

    def _resolve_sequence_attacker(
        self, home_poss: float, home_team: str, away_team: str,
    ) -> Tuple[str, str]:
        """Decide who attacks the next open-play sequence.

        Causal possession carry first: a team that just WON the ball in open
        play is OWED the sequence (~POSSESSION_CARRY_PROB), and the carry is
        single-shot — consumed and cleared here. Otherwise fall back to the
        possession-target weighted flip (home_poss is the home share, 0-1).

        Returns (attacking_team, defending_team).
        """
        winner_carried = False
        if self.state.possession_winner:
            if random.random() < self.POSSESSION_CARRY_PROB:
                attacker = self.state.possession_winner
                winner_carried = True
            self.state.possession_winner = ""  # single-shot rights
        if not winner_carried:
            attacker = home_team if random.random() < home_poss else away_team
        defender = away_team if attacker == home_team else home_team
        return attacker, defender

    def _synthesize_ball_path(self, events, start_x: float, start_y: float):
        """Build a continuous, time-ordered ball path for a chain that did NOT
        emit its own ``ball_path`` (every chain except PossessionChain today).

        Driven purely off the chain's REAL event coordinates (each event
        already carries the true ball location / end position from the
        simulation), interpolated at the shared 0.1 s clock used by
        PossessionEpisode. This is what makes the unified timeline continuous
        WITHIN every action and ACROSS actions: ball-moving events (passes,
        carries, shots, clearances, crosses) trace a smooth segment from the
        carried ball position to their destination, while contests (press /
        tackle / foul) anchor a marker at the live ball position instead of
        teleporting it to a default coordinate.

        Returns ``(points, duration_s)`` where ``points`` is a list of
        ``{"t","x","y","kind"}`` dicts with ``t`` local to this action (0-based).
        """
        if not events:
            return [], 0.0

        # Event families that actually move the ball.
        _MOVE = {
            "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
            "CROSS_ATTEMPT", "CROSS_SUCCESS", "FREEKICK_CROSS", "CORNER_TAKEN",
            "CARRY", "DRIBBLE_ATTEMPT", "DRIBBLE_SUCCESS", "DRIBBLE_FAIL",
            "CLEARANCE", "BLOCK",
            "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL",
            "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED", "HIT_WOODWORK",
            "BALL_RECEIPT",
        }
        # Restarts re-spot the ball (no connecting segment from the previous
        # action — the ball is physically placed at the restart coordinate).
        _RESTART = {
            "KICKOFF", "GOAL_KICK", "THROW_IN", "CORNER_TAKEN",
            "CORNER_WON", "FREEKICK_WON",
        }
        _SHOT = {
            "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL",
            "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED", "HIT_WOODWORK",
        }

        _PASS_SPEED = 14.0
        _CROSS_SPEED = 16.0
        _CARRY_SPEED = 5.5
        _CLEAR_SPEED = 18.0
        _SHOT_SPEED = 24.0
        _TICK = 0.1

        def _speed(name: str) -> float:
            if name in _SHOT:
                return _SHOT_SPEED
            if "CROSS" in name or name == "CORNER_TAKEN":
                return _CROSS_SPEED
            if "CARRY" in name or "DRIBBLE" in name:
                return _CARRY_SPEED
            if name in ("CLEARANCE", "BLOCK"):
                return _CLEAR_SPEED
            return _PASS_SPEED

        points: List[Dict[str, Any]] = []
        t = 0.0
        cx, cy = float(start_x), float(start_y)
        first = True

        for ev in events:
            name = getattr(getattr(ev, "event_type", None), "name", "")
            ex = getattr(ev, "end_x", None)
            ey = getattr(ev, "end_y", None)
            lx = getattr(ev, "location_x", None)
            ly = getattr(ev, "location_y", None)

            if name in _MOVE:
                # Real event geometry: a ball-moving event travels FROM its
                # own location (passer/carrier/shooter) TO its end coords
                # (receiver/box). We draw origin->destination, NOT
                # carried->destination — otherwise a pass whose origin is far
                # from the previous chain's end would draw a spurious
                # cross-pitch segment. Falls back to the carried position when
                # the event carries no usable coordinates.
                slx, sly = (float(lx), float(ly)) if (lx is not None and ly is not None) else (None, None)
                sex, sey = (float(ex), float(ey)) if (ex is not None and ey is not None) else (None, None)

                if slx is not None:
                    start = (slx, sly)
                else:
                    start = (cx, cy)
                if sex is not None:
                    target = (sex, sey)
                elif slx is not None:
                    target = (slx, sly)
                else:
                    target = (cx, cy)

                # Shots travel to the goal plane from the shooter's feet.
                if name in _SHOT:
                    goal_x = 105.0 if getattr(ev, "team", None) == self.config.home_team else 0.0
                    target = (float(goal_x), (sly if sly is not None else 34.0))

                # A restart (e.g. kickoff) re-spots the ball: begin here with
                # no connecting segment from the previous action.
                if first and name in _RESTART:
                    cx, cy = start
                    points.append({"t": round(t, 3), "x": round(cx, 2),
                                   "y": round(cy, 2), "kind": "restart"})
                    first = False
                    continue

                # Ball continuity within a chain: per-event coordinates come
                # from player positions that can drift a few metres from the
                # live ball — bridge the drift with a short carry so the ball
                # travels to its next passer rather than jumping on the log.
                carry_gap = math.hypot(start[0] - cx, start[1] - cy)
                if carry_gap > 1.5:
                    csteps = max(1, int(math.ceil(carry_gap / (_CARRY_SPEED * _TICK))))
                    for i in range(1, csteps + 1):
                        f = i / csteps
                        t += _TICK
                        points.append({"t": round(t, 3),
                                       "x": round(cx + (start[0] - cx) * f, 2),
                                       "y": round(cy + (start[1] - cy) * f, 2),
                                       "kind": "carry"})
                    cx, cy = start

                dist = math.hypot(target[0] - start[0], target[1] - start[1])
                spd = _speed(name)
                if dist <= 1e-6:
                    points.append({"t": round(t, 3), "x": round(start[0], 2),
                                   "y": round(start[1], 2), "kind": "static"})
                else:
                    steps = max(1, int(math.ceil(dist / (spd * _TICK))))
                    for i in range(1, steps + 1):
                        t += _TICK
                        f = i / steps
                        bx = start[0] + (target[0] - start[0]) * f
                        by = start[1] + (target[1] - start[1]) * f
                        points.append({"t": round(t, 3), "x": round(bx, 2),
                                       "y": round(by, 2), "kind": "move"})
                    cx, cy = target
            else:
                # Non-ball-moving event (press / tackle / foul / card / sub):
                # anchor a marker at the live ball position; it does not move
                # the ball, so no spurious jump to a default coordinate.
                if first and name in _RESTART:
                    if lx is not None and ly is not None:
                        cx, cy = float(lx), float(ly)
                    points.append({"t": round(t, 3), "x": round(cx, 2),
                                   "y": round(cy, 2), "kind": "restart"})
                    first = False
                    continue
                points.append({"t": round(t, 3), "x": round(cx, 2),
                               "y": round(cy, 2), "kind": "event"})
                t += _TICK

            first = False

        return points, round(t, 3)

    def _absorb_motion(self, chain_result) -> None:
        """Fold a chain's per-tick ``ball_path`` onto the SINGLE global match
        clock (Checkpoint — #1). Every possession/attack/defensive/set-piece
        chain emits its own continuously-timed ball_path; here we offset it by
        the running match clock and append to ``state.match_ball_path``, then
        advance the clock by the chain's measured (or synthesized) duration.
        The result is one monotonic, time-ordered ball trajectory for the whole
        match — the "broadcast replay" spine — even though the macro loop is
        still event/sequence-driven between chains.

        Chains that don't yet export a ``ball_path`` (shots, defensive actions,
        transitions, set pieces, fouls) are synthesized from their real event
        coordinates so the unified timeline stays continuous across ALL actions,
        not just possession spells.
        """
        bp = getattr(chain_result, "ball_path", None)
        dur = getattr(chain_result, "sequence_duration_s", 0.0) or 0.0

        # Fill the gap for chains that don't measure their own motion yet.
        if not bp:
            synth, synth_dur = self._synthesize_ball_path(
                getattr(chain_result, "events", []),
                self._last_path_x, self._last_path_y,
            )
            if synth:
                bp = synth
                # Only supply a duration if the chain did not measure
                # one. A set-piece chain that opened a jostling window
                # has stated its own elapsed time; overwriting it with
                # the synthesized travel time would silently shorten
                # the window and teleport the players again.
                if not dur:
                    dur = synth_dur

        team = None
        for ev in getattr(chain_result, "events", []):
            if getattr(ev, "team", None):
                team = ev.team
                break
        if team is None:
            team = self.state.possession_team

        prepend = 0.0
        if bp:
            base = self.state.match_clock_s
            # Continuity bridge — applied to EVERY consecutive gap in this
            # chain's ball path (the seam from the last chain's end AND any
            # internal jumps where a discrete action re-spots the ball): if
            # the ball would teleport farther than the fetch trigger, log it
            # physically travelling at ball-roll pace as a 'fetch' transition
            # so the unified timeline is a fluid replay, never a list of
            # disjoint possessions. Clock stays monotonic (float rounding is
            # clamped against the running cursor), and the persistent ball
            # state is synced to where the path actually ends.
            cursor = max(base, self._last_path_t)
            cvx, cvy = self._last_path_x, self._last_path_y
            for p in bp:
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                tt = base + float(p.get("t", 0.0))
                gap = math.hypot(x - cvx, y - cvy)
                if gap > self._FETCH_TRIGGER:
                    seg_dt = gap / self._FETCH_SPEED
                    steps = max(1, int(math.ceil(gap / (self._FETCH_SPEED * self._FETCH_TICK))))
                    end_t = max(cursor + seg_dt, tt)
                    for i in range(1, steps + 1):
                        f = i / steps
                        b_t = cursor + (end_t - cursor) * f
                        self.state.match_ball_path.append({
                            "t": round(b_t, 3),
                            "x": round(cvx + (x - cvx) * f, 2),
                            "y": round(cvy + (y - cvy) * f, 2),
                            "kind": "fetch",
                            "team": team,
                        })
                    cursor = end_t
                cvx, cvy = x, y
                if tt > cursor:
                    cursor = tt
                self.state.match_ball_path.append({
                    "t": round(cursor, 3),
                    "x": x,
                    "y": y,
                    "kind": p.get("kind", "move"),
                    "team": team,
                })
            # Carry the true last ball position forward for the next action so
            # its segment starts exactly where this one ended (cross-action
            # continuity), and keep the persistent ball state in agreement.
            _last = bp[-1]
            self._last_path_x = float(_last.get("x", self._last_path_x))
            self._last_path_y = float(_last.get("y", self._last_path_y))
            self._last_path_t = cursor
            self.state.last_ball_x = self._last_path_x
            self.state.last_ball_y = self._last_path_y
            # The global clock must absorb the bridging time actually consumed
            # (fetch segments push the tail beyond the chain's own duration).
            prepend = max(0.0, cursor - (base + float(bp[-1].get("t", 0.0))))

        if dur <= 0:
            # Final fallback: chains whose events carry no usable coordinates
            # still advance the clock monotonically by event volume.
            dur = max(0.0, len(getattr(chain_result, "events", [])) * 1.1)
        self.state.match_clock_s += dur + prepend
        # Keep off-ball players jogging through this episode's live window so
        # their distance/sprints are physically measured during play too (not
        # just dead time). The global clock was already advanced by `dur`.
        self._offball_run(dur, self.state.possession_team == self.config.home_team)

    # ── CONTINUOUS OFF-BALL INTEGRATOR (single 10 Hz clock) ──────────
    # Chasing (defending, ball within _CHASE_TRIGGER): instead of a constant
    # 80%-of-top-speed run that lasts forever, the player ACCELERATES into a
    # genuine burst (real top-end effort) that must let off after a few
    # seconds. This restores realistic top speeds (30+ km/h for athletes),
    # produces real sprint segments (>=7 m/s sustained) so CBs/CMs register
    # sprints, and stops endless high-speed running inflating distance for
    # wide/pressing roles.
    _CHASE_TRIGGER = 12.5    # ball within this distance (m) while defending
    _CHASE_RAMP = 0.55       # burst build factor per second of chase
    _CHASE_LETOFF = 6.0      # burst decay factor per second once spent
    _CHASE_BURST_T = 2.7     # max seconds a burst is fully sustained
    _CHASE_RESUSTAIN = 6.0    # ball this close -> re-trigger a fresh burst
    _CHASE_EFFORT = 0.95      # burst peak as fraction of player top speed

    # Ball-path fluidity: a chain whose ball starts farther than this from
    # where the last one ended (a restart re-spot, a set-piece placement) is
    # bridged by a 'fetch' segment — the ball physically travels back into
    # play at a ball-boy/roll-up pace instead of teleporting on the match log.
    _FETCH_TRIGGER = 8.0
    _FETCH_SPEED = 4.0
    _FETCH_TICK = 0.1

    # Probability a NEW press actually converts into a flat-out burst. Real
    # pressing is selective: central defenders/mids recover hard at high rate,
    # but wide forwards contain and pick their moments (~half), so they don't
    # fly in at every trigger and rack up real-world-unrealistic HSR volume.
    _PRESS_PROB = {
        "CB": 0.80, "DC": 0.80, "LCB": 0.80, "RCB": 0.80, "DMC": 0.75,
        "CDM": 0.75, "FB": 0.75, "LB": 0.75, "RB": 0.75, "CM": 0.75,
        "CAM": 0.60, "LM": 0.55, "RM": 0.55, "LW": 0.45, "RW": 0.45,
        "WF": 0.45, "ST": 0.50, "CF": 0.50, "SS": 0.55,
    }

    # Base sustained jog speed (m/s) by outfield role, used so off-ball
    # players cover realistic ground instead of idling at their shape target.
    _JOG_SPEED = {
        "GK": 1.0, "CB": 1.5, "DC": 1.5, "LCB": 1.5, "RCB": 1.5, "DMC": 1.5,
        "FB": 1.9, "LB": 1.9, "RB": 1.9,
        "CM": 2.0, "CDM": 2.0, "CAM": 2.0, "LM": 2.0, "RM": 2.0,
        "LW": 1.75, "RW": 1.75, "WF": 1.75,
        "ST": 2.0, "CF": 2.0, "SS": 2.0,
    }

    # ── LOW BLOCK: the defensive recovery pace ─────────────────────
    # A team that loses the ball drops into its block within a few seconds.
    # The generic jog ladder cannot express that: with the ball on the far side
    # it gives 0.54 m/s, which is fine for *holding* a shape you are already in
    # and useless for *reaching* one. Measured, the unit assembled 44 m from
    # its own goal when the block slot was 19 m — the approach never completed,
    # and then the hold-slot rule locked in whatever distance it had reached.
    #
    # Speed is proportional to the distance still owed and capped at a
    # fraction of the player's own top speed, so he eases in and stops rather
    # than jogging past the slot. Both numbers are fractions of his own pace
    # because this is a recovery run, not a sprint — 0.75 of top speed is a
    # strong jog back, not a burst.
    # Set-piece approach to a PENDING SLOT. Same reasoning and the same
    # shape as the low-block recovery below, because it is the same
    # problem: a destination the player is not currently in, which the
    # shape-holding trot can never reach. Not every defender makes it
    # before the cross, and that is correct.
    _SET_PIECE_APPROACH_GAIN = 1.1
    _SET_PIECE_APPROACH_TOP_FRAC = 0.75
    _LB_RECOVER_GAIN = 1.1
    _LB_RECOVER_TOP_FRAC = 0.75

    # Events that put a player ON the ball (so the off-ball integrator skips
    # them). Derived from events, not episode distance stats (which trace all
    # registered players).
    _ON_BALL_EVENTS = {
        "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
        "CARRY", "DRIBBLE_ATTEMPT", "DRIBBLE_SUCCESS", "DRIBBLE_FAIL",
        "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL",
        "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED", "HIT_WOODWORK",
        "CLEARANCE", "BLOCK", "CROSS_ATTEMPT", "CROSS_SUCCESS",
        "FREEKICK_CROSS", "CORNER_TAKEN", "BALL_RECEIPT", "INTERCEPTION",
        "TACKLE_WON", "TACKLE_LOST",
    }

    def _record_player_tick(self, t: float, home: str, away: str) -> None:
        """Passive player-position sample (read-only, no RNG, no outcome
        effect): snapshot every registered player's live coordinates onto
        ``state.player_path`` at ~5 Hz on the global clock, paired with the
        per-minute ball path so a 2D replay can show real compacted-shape
        motion instead of once-per-minute ghost dots."""
        if self.position_engine is None:
            return
        pp = self.state.player_path
        pt = self.state.player_team
        for team in (home, away):
            for name in self.position_engine.team_rosters.get(team, []):
                st = self.position_engine.states.get(name)
                if st is None:
                    continue
                if name not in pt:
                    pt[name] = team
                lst = pp.get(name)
                if lst is None:
                    lst = []
                    pp[name] = lst
                lst.append((round(float(t), 2),
                            round(float(st.current_x), 1),
                            round(float(st.current_y), 1)))

    def _record_offball_distance(self, name: str, moved: float,
                                 speed: float, top: float,
                                 duration_s: float) -> None:
        """Accumulate REAL per-tick off-ball movement into PositionEngine's
        physics store, counting sprint SEGMENTS with minimum duration and
        cooldown so micro-oscillations around the threshold don't inflate
        the count."""
        if moved <= 0:
            return
        st = self._sprint_state.setdefault(name, {
            "in_sprint": False, "in_hi": False,
            "sprint_run": 0, "hi_run": 0,
            "sprint_cooldown": 0, "hi_cooldown": 0,
        })
        sprint_inc = 0
        hi_inc = 0
        in_sprint = speed >= 7.0
        in_hi = speed >= 8.5
        # Sprint (>=7 m/s): count only after ≥3 consecutive ticks (0.3 s)
        # and only if cooldown has expired (≥5 ticks below threshold).
        if in_sprint:
            st["sprint_cooldown"] = 0
            st["sprint_run"] += 1
            if st["sprint_run"] == 3 and not st["in_sprint"]:
                sprint_inc = 1
                st["in_sprint"] = True
        else:
            st["sprint_run"] = 0
            st["in_sprint"] = False
            st["sprint_cooldown"] = 5
        # High-speed sprint (>=8.5 m/s): same logic, shorter window.
        if in_hi:
            st["hi_cooldown"] = 0
            st["hi_run"] += 1
            if st["hi_run"] == 2 and not st["in_hi"]:
                hi_inc = 1
                st["in_hi"] = True
        else:
            st["hi_run"] = 0
            st["in_hi"] = False
            st["hi_cooldown"] = 4
        self.position_engine.record_physics_distance(
            name, distance_m=moved, duration_s=duration_s, speed_mps=speed,
            sprint_count=sprint_inc,
            high_speed_sprint_count=hi_inc,
            top_speed_mps=speed if speed > 0 else 0.0,
        )

    def _offball_move_player(self, pname: str, team: str, ball_x: float,
                              ball_y: float, has_ball: bool, danger_t: float,
                              cross_team: str, DT: float) -> None:
        """One 10 Hz integration step for a single off-ball player.

        The player is pulled toward a live, ball-compacted shape target (home
        anchor + ball-side compression, plus defensive-block squeeze and
        attacking box-crash nudges). A sustained jog (with press-sprints when
        the ball is live and near while defending, and a small orbit once the
        shape is reached) means REAL ground is covered every tick — genuine
        physics integration, not a once-per-minute net delta. On-ball players
        are skipped by the caller.
        """
        st = self.position_engine.states.get(pname)
        if st is None or getattr(st, "position", "") == "GK":
            return
        if pname in self._on_ball_this_minute:
            return
        ax, ay = self._minute_start_snapshot.get(pname, (st.current_x, st.current_y))
        pos = getattr(st, "position", "")
        # OFF-BALL MOVEMENT HONESTY (2026-09-20): this player's INDIVIDUAL
        # physic-y read — the chase trigger + involvement gate — acts on
        # where HE believes the ball is, not the omniscient truth.  The
        # team's SHAPE compaction below keeps the true ball (a real team
        # shifts shape by voice, but the player who THINKS the ball is
        # elsewhere genuinely wastes the sprint / misses the close-down).
        _attacks_right = self.position_engine.team_attacks_right.get(team, True)
        bx, by = movement_ball(
            self, pname, pos, _attacks_right, ball_x, ball_y,
            float(self.state.minute))
        # Feature #1/#2 — RESHAPE WINDOW: when a formation stance or attack
        # pattern was just applied to this team, the off-ball anchor migrates
        # onto the player's NEW home post over ~90 s of match clock instead of
        # the team teleporting at a minute boundary. Weight decays to zero, so
        # ordinary ball-compacted shape takes over once the reshape is done.
        _applied = self._shape_apply_clock.get(team)
        if _applied is not None:
            _elapsed = self.state.match_clock_s - _applied
            if 0.0 <= _elapsed < 90.0:
                _rw = 1.0 - _elapsed / 90.0
                ax += (st.home_x - ax) * _rw * 0.20
                ay += (st.home_y - ay) * _rw * 0.20
        # ── DEFENSIVE BLOCK: substitute the ANCHOR, then let the chain compact it ──
        # A block is a SHAPE, and a shape is decided by the anchor, not by a
        # correction bolted on at the end. `tx = ax + (ball_x - ax) * 0.14`
        # compacts the team's home anchor toward the ball. Swapping the anchor
        # for the block position means the whole existing chain — including the
        # compaction, the danger pull and the live-spacing guard — then operates
        # on a block instead of fighting one.
        #
        # The first attempt steered the finished target by 0.85 at the end of
        # the chain. That is a blend against a competing value, and it lost: the
        # shape reached 47.5 m from its own goal when the block slot was 19 m.
        # Bumping the blend weight would not have fixed that; it would have
        # replaced the tuned build-up and spacing layers with a wall.
        #
        # Only while OUT of possession. In possession the team attacks with its
        # normal shape, which is what makes a block a platform rather than a
        # retreat — and the front two of a low block exist precisely to be the
        # outlet when it wins it back.
        _block = self._defensive_block.get(team) if not has_ball else None
        _in_block = _block is not None
        if _in_block:
            _lbt = self.position_engine.defensive_block_target(
                pname, ball_x, ball_y,
                self.position_engine.team_attacks_right.get(team, True),
                _block)
            if _lbt is not None:
                _lb_a, ax, ay = _lbt
        # Live shape target: home anchor compacted toward the ball.
        tx = ax + (ball_x - ax) * 0.14
        ty = ay + (ball_y - ay) * 0.07
        # Defensive block: out of possession + danger -> squeeze.
        #
        # Skipped inside a low block, and that is the whole difference between
        # the two. A mid-block team *surges* toward danger because the aim is to
        # win it back. A low block's aim is the opposite: hold the line, deny
        # the middle, and make the opponent come to you. Squeezing toward the
        # ball on every dangerous touch is the engine doing its job correctly
        # for the wrong shape.
        if (not has_ball) and danger_t >= 25 and not _in_block:
            pull = min(1.0, (danger_t - 25) / 65.0)
            tx += (ball_y - ty) * 0.25 * pull
            own_gx = 105.0 if team == self.config.away_team else 0.0
            tx += (own_gx - tx) * 0.10 * pull
        # Attacking box crash: cross in flight for this team.
        if cross_team == team:
            tx += (self.state.cross_x - tx) * 0.20
            ty += (self.state.cross_y - ty) * 0.20
        # ── LIVE RUN TARGETS (live) ────────────────────────────────────
        # The committed runs that the behavior engines have always computed but
        # which no match could reach, because their only consumer sat under
        # drift_minute and MatchEngine never calls drift_minute: striker in
        # behind / drop to link / post channel, winger byline / cut / box,
        # fullback overlap / underlap / tuck, CM drop / carry / late / orbit,
        # and the #10 pocket roam. A TARGET steer in the same shape as
        # CK36/37/38, so the jog integrator below still pace-caps the travel.
        #
        # PLACED BEFORE CK35/CK36 ON PURPOSE, which is the opposite of where
        # rest defence sits. Rest defence is a CONSTRAINT and therefore goes
        # last, so nothing can outvote it. A run target is only a PREFERENCE,
        # and CK35 (wide roles hold the touchline) and CK36 (CM triangle
        # socket) are older, tuned and covered by tests. Blending a new,
        # unproven layer after them silently disabled both - the triangle suite
        # caught exactly that on the first attempt. So a preference yields and
        # a constraint does not.
        runs = self._striker_runs(team, ball_x, ball_y, has_ball)
        pname_run = runs.get(pname)
        if pname_run is not None:
            # (blend, tx, ty, mode). The mode was recorded on the cache miss in
            # _striker_runs; all that is wanted here is the geometric steer.
            p_sr, srx, sry = pname_run[0], pname_run[1], pname_run[2]
            if p_sr > 0.0:
                tx += (srx - tx) * p_sr
                ty += (sry - ty) * p_sr
        # Checkpoint 35 — PITCH-STRETCH RULE: wide roles (LW/RW/LB/RB) are the
        # team's width providers. When the live ball sits on the central spine
        # (packed middle), steer the off-ball shape target toward the player's
        # touchline channel so width is actively re-asserted every tick
        # instead of letting central ball-compaction collapse the flanks. The
        # blend is spine-scaled and the ACTUAL movement stays pace-capped (the
        # jog integrator below travels toward the steered target); it is zero
        # when the ball is already wide. Final y is hard-clamped on the pitch.
        #
        # Suppressed inside a low block. Re-asserting the touchline channel
        # precisely when the ball is central is the direct opposite of what a
        # low block is for: the whole mechanism is that the unit NARROWS around
        # the spine to deny the middle, and this rule exists to stop it
        # narrowing. Both rules cannot hold at once, and for this shape the
        # block wins.
        stretch_w = 0.0 if _in_block else self.position_engine.wide_stretch_blend(
            pname, ball_y)
        if stretch_w > 0.0:
            ty += (getattr(st, "home_y", ty) - ty) * stretch_w
        # Checkpoint 36 — TRIANGLE SUPPORT RULE: while the team holds the ball,
        # midfielders steer their shape target into a passing-triangle socket —
        # a half-space support off the wide cluster on flank play (near-side CM
        # joins winger + full-back), or split around the ball with the CDM
        # pivot behind on central play. TARGET steer only: the jog integrator
        # below stays pace-capped, and the y-socket is always inside the pitch.
        tri = self.position_engine.midfielder_triangle_support(
            team, ball_x, ball_y, has_ball,
            self.position_engine.team_attacks_right.get(team, True))
        if pname in tri:
            p_tri, trix, triy = tri[pname]
            if p_tri > 0.0:
                tx += (trix - tx) * p_tri
                ty += (triy - ty) * p_tri
        # Checkpoint 37 — BACK-LINE BUILD-UP DROP: while the team builds from
        # the back, the ball-side CB sags toward a goal-side socket so the
        # pressed midfield has a short 8-15m reset instead of the 25-40m
        # heave to the keeper. TARGET steer only (jog integrator pace-caps).
        # Checkpoint 37/38 — BACK-LINE BUILD-UP DROP and PRESSURE-AWARE
        # BACK-LINE SPREAD.
        #
        # CK38 is NOT suppressed inside a low block, despite appearances.
        # Suppressing it was tried and measured: the block slot of 19 m then
        # arrived at the final target as 39.4 m, against 36.5 m with CK38
        # running. So CK38 was pulling the block DEEPER, not spreading it away
        # as its name suggests in this context, and the guard does the team no
        # harm here. The anchor stays overridden; these two stay on.
        bld = self.position_engine.backline_build_up_support(
            team, ball_x, ball_y, has_ball,
            self.position_engine.team_attacks_right.get(team, True))
        if bld and pname in bld:
            p_bl, blx, bly = bld[pname]
            if p_bl > 0.0:
                tx += (blx - tx) * p_bl
                ty += (bly - ty) * p_bl
        spr = self.position_engine.backline_spread_pressure(
            team, ball_x, ball_y, has_ball,
            self.position_engine.team_attacks_right.get(team, True))
        if spr and pname in spr:
            p_sp, spx, spy = spr[pname]
            if p_sp > 0.0:
                tx += (spx - tx) * p_sp
                ty += (spy - ty) * p_sp
        # (The low block no longer steers here. It substitutes the shape ANCHOR
        # upstream, before the generic compaction, so the whole chain operates
        # on a low block instead of fighting one. See the anchor substitution
        # above and `PositionEngine.low_block_target`.)
        tx = max(0.0, min(105.0, tx))
        ty = max(0.0, min(68.0, ty))
        cx, cy = st.current_x, st.current_y
        # Checkpoint 38 — LIVE-TICK SPACING GUARD: redirect this runner's
        # target away from any teammate currently inside LIVE_SEP_MIN of him,
        # so two players never stack on the same socket during a build-up
        # sequence (the per-minute graph repel is too slow at 10Hz).
        tx, ty = self.position_engine.live_spacing_redirect(
            team, cx, cy, tx, ty)
        # ── REST DEFENCE (positional play) ────────────────────────────
        # Last of the four superiority types with no representation here, and
        # the only one that is a CONSTRAINT. Every rule above is a preference
        # toward a socket, so a team whose preferences all point upfield can
        # legally end up with ten men ahead of the ball; this makes that state
        # unreachable.
        #
        # Applied LAST, after every other rule has had its say, because a
        # constraint has to be evaluated against the final target — clamping
        # earlier would just be undone by CK37/CK38/triangle support, which all
        # steer the same depth axis. A depth clamp, not a blend: blending
        # would let a preference outvote an invariant.
        #
        # Resolved once per tick per team (see _rest_defence_violator) and
        # applied to exactly one player, so a team that has genuinely nobody
        # back is not turned inside out trying to fix it.
        # A pending set-piece slot overrides the shape target AND rest
        # defence: during an attacking corner nobody should be pulled
        # behind the ball, least of all the striker standing on the last
        # man. Placed immediately before rest defence so it wins the way
        # an invariant wins, rather than blending as a preference would.
        _sp_slot = self.position_engine.setpiece_target(pname)
        if _sp_slot is not None:
            tx, ty = _sp_slot
        elif self._rest_defence_violator(team, ball_x, has_ball) == pname:
            tx, ty = self.position_engine.rest_defence_clamp(
                tx, ty, ball_x, _attacks_right)
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        tgt = self._top_speed_cache.get(pname, 7.0)
        jog = self._JOG_SPEED.get(pos, 1.8)
        ball_dist = math.hypot(bx - cx, by - cy)
        # Involvement gate: when the play is on the FAR side of the pitch a
        # player holds his shape at a light trot rather than tracking the
        # ball across it (real wingers/full-backs don't chase diagonally);
        # full jog / chase bursts only engage when the ball swings into his
        # zone. This is what keeps wide roles' totals around the real
        # ~11.5-12.5 km/90 instead of 16+.
        # OFF-BALL MOVEMENT HONESTY (2026-09-20): both this gate and the
        # chase trigger/resustain act on the PHYSICAL perceived ball (bx,
        # by) — honest effort: a player who believes the ball is far holds
        # his trot, a wrong belief wastes (or forfeits) the sprint.
        involved = abs(bx - self._minute_start_snapshot.get(pname, (cx, cy))[0]) < 30.0
        if not involved:
            jog = jog * 0.30
        chasing = (not has_ball) and ball_dist < self._CHASE_TRIGGER
        cst = self._chase_state.setdefault(pname, {"p": 0.0, "t": 0.0, "allow": 0.0})
        if chasing:
            if cst["p"] <= 0.0 and cst["allow"] == 0.0:
                _g_key = (team, round(self.state.match_clock_s * 10.0))
                g = self._team_press_g_cache.get(_g_key)
                if g is None:
                    g = _team_press_g(self, team, ball_x, ball_y, danger_t)
                    self._team_press_g_cache[_g_key] = g
                prob = self._PRESS_PROB.get(pos, 0.60) * g
                # Phase 10 — position off-ball brain: a LW/RW/CM runner with
                # an evolved press-gate brain replaces the static role rate
                # with its own p(press) over the same off-ball vector it was
                # trained on, still scaled by the team commitment.  Missing
                # position brain / any error => None => keep today's formula.
                if _OFFBALL_POS_BRAIN_WIRED:
                    _ob = self._offball_press_prob(
                        pname, pos, team, ball_x, ball_y, danger_t)
                    if _ob is not None:
                        prob = _ob * g
                cst["allow"] = 1.0 if random.random() < prob else -1.0
            cst["t"] += DT
            if cst["allow"] > 0:
                if cst["t"] < self._CHASE_BURST_T or ball_dist < self._CHASE_RESUSTAIN:
                    cst["p"] = min(1.0, cst["p"] + self._CHASE_RAMP * DT)
                else:
                    cst["p"] = max(0.0, cst["p"] - self._CHASE_LETOFF * DT)
                speed = jog + (tgt * self._CHASE_EFFORT - jog) * cst["p"]
            else:
                cst["p"] = 0.0
                speed = jog
        else:
            cst["p"] = max(0.0, cst["p"] - self._CHASE_LETOFF * DT)
            cst["allow"] = 0.0
            speed = jog
        arrive = 2.0
        _slot_dist = math.hypot(tx - cx, ty - cy)
        if _in_block and dist > arrive:
            # Dropping into the block is a DEFENSIVE RECOVERY, not a jog.
            #
            # The generic ladder above gives 0.54 m/s when the ball is on the
            # far side of the pitch, which is fine for holding a shape you are
            # already in and useless for reaching one you are not: measured, the
            # unit assembled 44 m from its own goal when the block slot was
            # 19 m, because the approach was too slow to ever complete and then
            # held position at whatever distance it had reached.
            #
            # A real team that loses the ball drops into its block within a few
            # seconds. Speed is scaled by how far the player still is from his
            # slot, so he covers the last long way and eases in — and, because
            # it decays to zero at the slot, he still arrives and STOPS rather
            # than jogging past it.
            _recover = min(tgt * self._LB_RECOVER_TOP_FRAC,
                           _slot_dist * self._LB_RECOVER_GAIN)
            speed = max(jog, _recover)

        if _sp_slot is not None and dist > arrive:
            # A corner slot 40 m away is not a shape to hold, it is a
            # race to run. Identical in form to the block recovery
            # above and for the identical stated reason: the generic
            # ladder is a shape integrator, so it trots at 0.54-0.71 m/s
            # and never arrives. Measured at 0.71 m/s over 5.0 s, a
            # 3.5 m gain on a 41.5 m gap -- 28 of 117 assignments moved
            # more than half a metre.
            _run = min(tgt * self._SET_PIECE_APPROACH_TOP_FRAC,
                       _slot_dist * self._SET_PIECE_APPROACH_GAIN)
            speed = max(speed, _run)

        if dist > arrive:
            step = min(speed * DT, dist)
            nx = cx + dx / dist * step
            ny = cy + dy / dist * step
            moved = step
        else:
            # Arrived at shape: patrol a small orbit so the player keeps jogging
            # (covering distance) instead of idling at his spot.
            #
            # A low block does NOT do this, and the post is explicit about why:
            # "positions are held, not players tracked, so movement does not
            # break the shape." An orbit means every player is permanently out
            # of position, so the block can never actually assemble — measured,
            # the unit sat 53 m from its own goal when the block slot was 19 m,
            # not because the target was wrong but because nobody ever stopped
            # moving to reach it.
            if _in_block:
                # Hold the slot. A token drift keeps distance accounting alive
                # without a player visibly wandering off his line.
                step = 0.12 * jog * DT
                nx, ny = cx, cy
                moved = 0.0
            else:
                ang = self._patrol.get(pname, random.random() * 6.283)
                ang += 0.6 * DT
                self._patrol[pname] = ang
                px = ax + 4.0 * math.cos(ang)
                py = ay + 4.0 * math.sin(ang)
                pdx, pdy = px - cx, py - cy
                pd = math.hypot(pdx, pdy)
                step = (jog * 0.8 * DT) if pd <= 0 else min(jog * 0.8 * DT, pd)
                nx = cx + (pdx / pd * step if pd > 0 else 0.0)
                ny = cy + (pdy / pd * step if pd > 0 else 0.0)
                moved = step
        spd = (moved / DT) if DT > 0 else 0.0

        # ── CONTINUOUS PHYSICS: physical movement (opt-in) ───────────
        # Everything above is the engine's own step, and it is left exactly as
        # it was. This block REPLACES the resulting position with a kinematic
        # one when — and only when — physics is switched on.
        #
        # Why replace rather than adjust: the step above assigns a scalar speed
        # and displaces the player along the straight line to his target, so
        # direction is recomputed from scratch every tick. A player at full
        # pace can reverse instantly, nothing limits how fast his heading
        # swings, `tgt` is DNA-only so his legs never feel fatigue, and a
        # changed target is acted on with no perception cost. None of those are
        # reachable by tweaking a coefficient, because the missing quantity is
        # *velocity state*, not a gain.
        #
        # The target itself is untouched. All that shaping — the shape engine,
        # wide stretch, triangle support, build-up drop, backline spread,
        # live-spacing redirect — still decides WHERE the player is trying to
        # be. The physics only decides how his body gets there. That split is
        # the reason this is safe to enable: the tuned shape logic is not
        # re-tuned, it is obeyed.
        #
        # Inert unless MatchConfig.physics_enabled is True, and a physics
        # failure leaves the engine's own step in place rather than dropping
        # the player.
        if self.physics_enabled and self.physics is not None:
            try:
                _phys_step = self.physics.offball_step(pname, (tx, ty), DT)
            except Exception:
                _phys_step = None
            if _phys_step is not None:
                nx, ny, moved, spd = _phys_step

        st.current_x, st.current_y = nx, ny
        # Real distance into both the physics store and the legacy drift
        # field (so distance_total stays the comprehensive real total).
        self._record_offball_distance(pname, moved, spd, tgt, DT)
        st.minute_drift_distance += moved

    def _offball_press_prob(self, pname, position, team, ball_x, ball_y,
                            danger_t) -> Optional[float]:
        """p(press) for this runner from the evolved position off-ball brain.

        Wraps offball_brain_wiring.offball_press_prob; any missing brain,
        malformed state or exception yields None (engine keeps the static
        Bernoulli).  Memoized per press window (reset when allow clears)."""
        try:
            import offball_brain_wiring as _obw
            cache = self._offball_press_cache
            key = (pname, round(ball_x), round(ball_y), round(danger_t, 1))
            hit = cache.get(key)
            if hit is None:
                hit = _obw.offball_press_prob(
                    self, pname, position, team, ball_x, ball_y, danger_t,
                    minute=self.state.minute,
                    score_diff=self.state.home_goals - self.state.away_goals,
                )
                cache[key] = hit
            return hit
        except Exception:
            return None

    def _sample_run_tracking(self, t: float, home_has_ball: bool) -> None:
        """Checkpoint 39 — feed the run tracker one 10 Hz sample of every
        in-possession outfield player (the tracker itself resets segments
        across possession flips and sample gaps)."""
        if self.run_tracker is None or self.position_engine is None:
            return
        bx, by = self.state.last_ball_x, self.state.last_ball_y
        if bx is None or by is None:
            return
        # A set piece is a DEAD BALL. Do not sample runs while one is
        # pending: the six observed types describe movement relative
        # to the ball during LIVE play, and filing "centre-back
        # shuffled to his marker" as a support run is precisely how
        # `support` became 95% of every run before.
        #
        # Distance is deliberately NOT suppressed — it accrues through
        # `record_physics_distance` in `_offball_move_player`, which
        # runs regardless. So a corner approach counts as distance
        # covered and NOT as a run, which is the answer.
        if getattr(self, "_dead_ball", False) or self.position_engine.setpiece_active():
            return
        home = self.config.home_team
        team = home if home_has_ball else self.config.away_team
        attacks_right = self.position_engine.team_attacks_right.get(team, True)
        for pname in self.position_engine.team_rosters.get(team, []):
            st = self.position_engine.states.get(pname)
            if st is None or getattr(st, "position", "") == "GK":
                continue
            if st.current_x is None or st.current_y is None:
                continue
            self.run_tracker.sample(
                t, pname, st.current_x, st.current_y, bx, by,
                attacks_right, team, True)

    def get_run_profile(self) -> Dict[str, Dict[str, int]]:
        """Checkpoint 39 — per-player off-ball run counts, six types:
        advance / overlap / underlap / far_side / forward / support."""
        return self.run_tracker.profile() if self.run_tracker else {}

    def _offball_run(self, duration_s: float, home_has_ball: bool) -> None:
        """Jog every off-ball player for ``duration_s`` of LIVE play (during an
        episode), without advancing the global clock — the episode already did.
        This is what makes off-ball distance real during possession/duels, not
        just during dead time."""
        if duration_s <= 0:
            return
        DT = 0.1
        ticks = max(1, int(round(duration_s / DT)))
        home, away = self.config.home_team, self.config.away_team
        ball_x, ball_y = self.state.last_ball_x, self.state.last_ball_y
        danger = {
            home: self.threat.danger_at(home),
            away: self.threat.danger_at(away),
        }
        cross_team = self.state.cross_team if self.state.cross_active else ""
        # GPS: use a local, continuously-advancing timebase through the live
        # window so the raw log stays monotonic at 10 Hz and aligned with the
        # global clock (the global clock was already advanced by `duration_s`,
        # and only advances once per chain, not per tick, so it would repeat).
        # The live window spans [clock - duration_s, clock].
        t0 = self.state.match_clock_s - duration_s
        # The pending set-piece window is consumed PER SUB-TICK inside the loop
        # below, not here. Consuming the whole budget at the top zeroed it
        # before the first `setpiece_target()` lookup, so a 5 s corner window
        # was live for exactly one 0.1 s step and then every player fell back
        # to his shape anchor -- measured 0.93 m/s against a 39 m median gap.
        # The flag is captured HERE because it describes the whole run.
        self._dead_ball = self.position_engine.setpiece_active()
        if self.gps is not None:
            self.gps.begin_minute(self.state.minute, t0)
        # On-ball players move via possession-episode traces that are ingested
        # separately; their end-of-chain position syncs are position jumps we
        # must not re-derive as distance/sprints from the position gap.
        skip = set(self._on_ball_this_minute) if self.gps is not None else None
        for i in range(ticks):
            # Spend the set-piece window as the integrated time actually
            # elapses, so `setpiece_target()` stays live for every one of the
            # `ticks` steps it was opened for and closes on the last one.
            if self.position_engine.setpiece_active():
                self.position_engine.tick_setpiece(DT)
            # Team-wide per-tick decisions (rest defence) are resolved once and
            # read by both teams below, so the tick needs an identity of its
            # own. match_clock_s does not supply one: this loop advances it
            # outside, so it is constant for every tick here.
            self._offball_tick_seq += 1
            for team in (home, away):
                has_ball = (team == home) == home_has_ball
                for pname in self.position_engine.team_rosters.get(team, []):
                    self._offball_move_player(
                        pname, team, ball_x, ball_y, has_ball,
                        danger.get(team, 0.0), cross_team, DT)
                # Checkpoint 37 — SWEEPER-KEEPER BUILD-OUT: while this team
                # possesses inside its own two-thirds the keeper slides to a
                # ~24m socket so the CB/CDM/FB always has a genuinely SHORT
                # back bump (the 30m gate keeps feeding healthy and short).
                if has_ball:
                    _pert = self.position_engine.team_attacks_right.get(team, True)
                    _gk_t = self.position_engine.gk_build_up_advance(
                        team, ball_x, ball_y, True, _pert)
                    if _gk_t is not None:
                        for _gn in self.position_engine.team_rosters.get(team, []):
                            _gs = self.position_engine.states.get(_gn)
                            if _gs is None or getattr(_gs, "position", "") != "GK":
                                continue
                            _gxp = self.position_engine.GK_BUILD_UP_PULL
                            _gs.current_x += (_gk_t[0] - _gs.current_x) * _gxp
                            _gs.current_y += (_gk_t[1] - _gs.current_y) * _gxp
                            break
            # Checkpoint 39 — OFF-BALL RUN TRACKING: sample the possessing
            # team's outfield players once per tick (classification happens
            # inside RunTracker against the previous sample).
            self._sample_run_tracking(t0 + (i + 1) * DT, home_has_ball)
            if i % 2 == 0:
                self._record_player_tick(t0 + (i + 1) * DT, home, away)
            if self.gps is not None:
                self.gps.record_tick(
                    self.state.minute, t0 + (i + 1) * DT, self.position_engine,
                    home, away, ball_x, ball_y,
                    skip_accumulate=skip)

    def _continuous_offball_phase(self, minute: int, home_has_ball: bool,
                                  gd_home_now: float,
                                  danger: Dict[str, float]) -> None:
        """Integrate every off-ball player at 10 Hz for the remainder of the
        minute so the whole match lives on one continuous clock.

        This phase covers the DEAD time between episodes; the live-play window
        is covered by ``_offball_run`` (called from ``_absorb_motion``). The
        clock advances by dt every tick and the ball is logged as 'rest', which
        fills the dead time so ``match_ball_path`` spans the full 90' with no
        gaps.
        """
        DT = 0.1
        elapsed = self.state.match_clock_s - self._minute_start_clock
        remaining = max(0.0, 60.0 - elapsed)
        ticks = max(1, int(round(remaining / DT)))

        home = self.config.home_team
        away = self.config.away_team
        cross_team = self.state.cross_team if self.state.cross_active else ""
        # On-ball players are moved by episode traces (ingested separately);
        # skip their end-of-chain position syncs here too (dead time has no
        # such snaps, but the sync lands on the FIRST dead tick).
        skip = set(self._on_ball_this_minute) if self.gps is not None else None

        for ti in range(ticks):
            ball_x = self.state.last_ball_x
            ball_y = self.state.last_ball_y
            self._offball_tick_seq += 1
            for team in (home, away):
                has_ball = (team == home) == home_has_ball
                for pname in self.position_engine.team_rosters.get(team, []):
                    self._offball_move_player(
                        pname, team, ball_x, ball_y, has_ball,
                        danger.get(team, 0.0), cross_team, DT)
            # Checkpoint 39 — keep run segments alive through dead-time
            # possession too (the tracker's GAP_S guard bridges any break).
            self._sample_run_tracking(self.state.match_clock_s, home_has_ball)
            self.state.match_clock_s += DT
            if ti % 10 == 0:
                self.state.match_ball_path.append({
                    "t": round(self.state.match_clock_s, 3),
                    "x": round(ball_x, 2), "y": round(ball_y, 2),
                    "kind": "rest", "team": self.state.possession_team or "",
                })
            if ti % 2 == 0:
                self._record_player_tick(self.state.match_clock_s, home, away)
            if self.gps is not None:
                self.gps.record_tick(
                    minute, self.state.match_clock_s, self.position_engine,
                    home, away, ball_x, ball_y,
                    skip_accumulate=skip)

    def _maybe_loose_ball(
        self, minute, shot_result, attacking_team, defending_team,
        att_players, def_players, attacks_right,
    ) -> None:
        """#4 — after a saved/blocked shot the ball is a LIVE loose entity that
        nearby players race for on the continuous clock (resolve_loose_ball).
        We resolve who wins it and fold the scramble onto the global timeline
        + possession clock, so second balls are no longer teleported away.
        """
        from possession_physics import PossessionEpisode
        from event_chain import ChainDispatcher, BaseChain, EventType as ET
        saved = any(
            getattr(e.event_type, "name", "")
            in ("SHOT_SAVED", "SHOT_BLOCKED")
            for e in shot_result.events
        )
        if not saved or self.position_engine is None:
            return
        sx = sy = None
        for e in shot_result.events:
            if getattr(e.event_type, "name", "") in ("SHOT_SAVED", "SHOT_BLOCKED"):
                sx = e.end_x if e.end_x is not None else e.location_x
                sy = e.end_y if e.end_y is not None else e.location_y
        if sx is None:
            sx, sy = self.state.last_ball_x, self.state.last_ball_y
        att_m = [BaseChain._moving_player(p, self.position_engine)
                 for p in att_players if getattr(p, "position", "") != "GK"]
        defe_m = [BaseChain._moving_player(p, self.position_engine)
                  for p in def_players if getattr(p, "position", "") != "GK"]
        ep = PossessionEpisode()
        winner = ep.resolve_loose_ball(sx, sy, att_m, defe_m)
        team = attacking_team if winner == "attack" else defending_team

        # Fold the scramble onto the single global match timeline (#1/#4).
        base = self.state.match_clock_s
        for p in ep.ball_path:
            self.state.match_ball_path.append({
                "t": round(base + float(p.get("t", 0.0)), 3),
                "x": float(p.get("x", sx)), "y": float(p.get("y", sy)),
                "kind": "loose", "team": team,
            })
        self.state.match_clock_s += ep.elapsed
        if ep.ball_path:
            _lp = ep.ball_path[-1]
            self._last_path_x = float(_lp.get("x", self._last_path_x))
            self._last_path_y = float(_lp.get("y", self._last_path_y))
        self._credit_possession(team, ep.elapsed, None)

        ball_ev = getattr(ET, "BALL_RECOVERY", None)
        if ball_ev is not None:
            self.timeline.append(MatchEvent(
                minute=minute, second=int(self.state.match_clock_s % 60),
                event_type=ball_ev, team=team, player="",
                location_x=sx, location_y=sy,
                phase=self.state.phase, game_state=self.state.game_state,
            ))

    def _run_minute(self, minute: int):
        """Kept for API symmetry; the real driver is the closure in simulate()."""
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────────
    # EXTRA TIME + PENALTY SHOOTOUT (audit §16 phase 5, plan §10/§11)
    # ─────────────────────────────────────────────────────────────
    #
    # Everything in this block is behind `config.extra_time` /
    # `config.penalties`, both of which default to False. A 26/27 league match
    # never enters it, so its output is byte-identical to before (plan §19).
    #
    # TWO DESIGN RULES, both load-bearing:
    #
    # 1. The shootout draws from `_COSMETIC_RNG`, the dedicated cosmetic
    #    stream, NOT the football RNG. A penalty shootout is the resolution of a
    #    tie, not part of the football — if it consumed the seeded sequence it
    #    would change every event after it and destroy reproducibility. This is
    #    the same discipline already used for goal-celebration durations.
    #
    # 2. Extra time runs through the SAME `_run_minute` closure as the first 90.
    #    There is no second, simplified "cup mode" — the same brains, chains,
    #    physics and substitutions play out, on tired legs.

    #: Real extra time is two 15-minute periods with a short break between.
    EXTRA_TIME_PERIOD_MINUTES = 15
    #: The break before the second ET period buys far less than half-time's 18%.
    EXTRA_TIME_BREAK_RECOVERY = 0.06
    #: Standard shootout: five kicks each, then sudden death.
    SHOOTOUT_INITIAL_KICKS = 5

    def _extra_time_needed(self) -> bool:
        """A tie is level after 90 (+added) and the competition allows ET."""
        return self.config.extra_time and \
            self.state.home_goals == self.state.away_goals

    def _play_extra_time(self, run_minute) -> int:
        """Play up to 30 minutes of extra time. Returns minutes actually played.

        Ends the moment either side leads by two, which is the real rule and
        also keeps a decided tie from grinding out dead minutes.
        """
        start = 90 + self.state.added_time + 1
        played = 0
        self.state.went_to_extra_time = True
        self.state.in_extra_time = True
        self.state.extra_time_minutes = 0

        for offset in range(self.EXTRA_TIME_PERIOD_MINUTES * 2):
            # the interval before the second period
            if offset == self.EXTRA_TIME_PERIOD_MINUTES and \
                    self.sub_controller is not None:
                for team_players in self.active_players.values():
                    for p in team_players:
                        state = self.sub_controller.stamina.get(
                            getattr(p, "name", ""))
                        if state and not state.is_injured:
                            state.half_time_recovery(
                                recovery_pct=self.EXTRA_TIME_BREAK_RECOVERY)

            minute = start + offset
            run_minute(minute)
            played += 1
            self.state.extra_time_minutes = played

            if abs(self.state.home_goals - self.state.away_goals) >= 2:
                break

        self.state.in_extra_time = False
        return played

    def _shootout_takers(self, team: str) -> List[Any]:
        """Kick order for one side: outfield starters first, then substitutes.

        Real teams send their best takers first; the squad is already ordered
        that way, so outfielders-before-keepers-then-bench is a faithful
        approximation without inventing a separate "penalty taker" ranking.
        """
        squad = (self.squads.get(team) or {})
        outfield, keepers = [], []
        for p in squad.get("starters", []):
            (keepers if str(getattr(p, "position", "")).upper() == "GK"
             else outfield).append(p)
        return outfield + keepers + list(squad.get("substitutes", []))

    def _run_penalty_shootout(self) -> Dict[str, Any]:
        """Decide a level tie from the spot. Returns the shootout record.

        Deterministic and self-contained: five kicks each, alternating with the
        home side first, stopping the instant the tie is mathematically decided,
        then sudden death. Drawn from `_COSMETIC_RNG` so it cannot perturb the
        seeded football sequence.
        """
        home = self.config.home_team
        away = self.config.away_team
        takers = {home: self._shootout_takers(home),
                  away: self._shootout_takers(away)}

        scored = {home: 0, away: 0}
        taken = {home: 0, away: 0}
        rounds: List[Dict[str, Any]] = []
        self.state.shootout_played = True

        def _kick(team: str, round_no: int, sudden: bool) -> bool:
            order = takers[team]
            idx = taken[team]
            taken[team] += 1
            player = order[idx] if idx < len(order) else None
            name = getattr(player, "name", f"{team} taker {idx + 1}")
            # a taken penalty goes in far more often than it is saved
            converted = _COSMETIC_RNG.random() < 0.76
            if converted:
                scored[team] += 1
            self.state.match_clock_s += 25  # ~25s per penalty, incl. the run-up
            minute, second = divmod(int(self.state.match_clock_s), 60)
            # A penalty is taken from the SPOT, 11 m from the goal line — not
            # from the centre circle this event used to inherit by default
            # (`MatchEvent.location_x` defaults to 50.0). Which end is a
            # convention here: the engine models no shootout goal, because a
            # real shootout has both teams kicking at the SAME one. So the
            # home-attacking end is used and the value is LABELLED, rather than
            # being indistinguishable from an observed coordinate.
            spot_x = 105.0 - 11.0
            self.timeline.append(MatchEvent(
                minute=minute, second=second,
                event_type=(EventType.PENALTY_SCORED if converted
                            else EventType.PENALTY_MISSED),
                team=team, player=name,
                phase=self.state.phase, game_state=self.state.game_state,
                location_x=spot_x, location_y=34.0,
                metadata={"round": round_no, "sudden_death": sudden,
                          "taker_index": idx,
                          "location_source": "shootout_spot_convention"},
            ))
            if not self.quiet:
                mark = "✔" if converted else "✘"
                print(f"  ⚽ {minute}' PEN {name} ({team}) {mark} "
                      f"[{scored[home]}-{scored[away]}]")
            return converted

        def _decided(n: int) -> bool:
            """Is the tie decided after n kicks each (or in sudden death)?"""
            remaining = self.SHOOTOUT_INITIAL_KICKS - n
            return (scored[home] > scored[away] + remaining
                    or scored[away] > scored[home] + remaining)

        # ── initial five each ──
        for n in range(self.SHOOTOUT_INITIAL_KICKS):
            _kick(home, n + 1, sudden=False)
            if _decided(n + 1):
                break
            _kick(away, n + 1, sudden=False)
            if _decided(n + 1):
                break
            rounds.append({"round": n + 1, "home": scored[home],
                           "away": scored[away], "sudden_death": False})

        winner = ""
        if not _decided(self.SHOOTOUT_INITIAL_KICKS):
            # ── sudden death: one each, until they differ ──
            n = self.SHOOTOUT_INITIAL_KICKS
            while not winner:
                n += 1
                before = (scored[home], scored[away])
                _kick(home, n, sudden=True)
                _kick(away, n, sudden=True)
                rounds.append({"round": n, "home": scored[home],
                               "away": scored[away], "sudden_death": True})
                if scored[home] != scored[away]:
                    winner = home if scored[home] > scored[away] else away
                assert (scored[home], scored[away]) != before or winner

        if not winner:
            # the loop above always terminates on a difference; if the initial
            # five decided it, settle it here
            winner = home if scored[home] > scored[away] else away

        self.state.home_pens = scored[home]
        self.state.away_pens = scored[away]
        self.state.shootout_winner = winner
        self.state.shootout_rounds = rounds
        return {
            "home_team": home,
            "away_team": away,
            "home_pens": scored[home],
            "away_pens": scored[away],
            "winner": winner,
            "kicks": {home: taken[home], away: taken[away]},
            "rounds": rounds,
        }

    def _resolve_winner_team(self) -> str:
        """The team that advanced, or "" for a draw / a league match.

        Deliberately empty for an ordinary league result: a league match has no
        winner concept, and reporting the leader as "the winner" would leak
        knockout semantics into every 26/27 row in the warehouse.
        """
        if not (self.config.extra_time or self.config.penalties
                or self.config.aggregate):
            return ""
        if self.state.shootout_winner:
            return self.state.shootout_winner
        hg, ag = self.state.home_goals, self.state.away_goals
        if hg > ag:
            return self.config.home_team
        if ag > hg:
            return self.config.away_team
        if self.config.away_goals_rule and self.config.aggregate:
            prior_h, prior_a = self.config.aggregate
            if prior_h + hg > prior_a + ag:
                return self.config.home_team
            if prior_a + ag > prior_h + hg:
                return self.config.away_team
        return ""

    def _resolve_aggregate(self) -> Optional[Tuple[int, int]]:
        """Running two-legged aggregate, or None for a single match.

        ``MatchContextFragment.aggregate`` carries the score BEFORE this leg, so
        the total is prior + this leg's goals (a shootout does not count — the
        modern UEFA convention, and the away-goals rule is dead anyway).
        """
        if not self.config.aggregate:
            return None
        prior_h, prior_a = self.config.aggregate
        return (prior_h + self.state.home_goals,
                prior_a + self.state.away_goals)

    def simulate(self) -> "MatchResult":
        """
        Run the full match simulation.
        Returns a MatchResult containing all events and derived stats.
        Substitution logic and stamina tracking run in parallel.
        """
        self._initialize_simulation()

        def _run_minute(minute: int):
            self.state.minute = minute
            # Extra time plays under the ADDED_TIME phase (minutes past 90 are
            # the highest-tension phase in the model). The guard is written so
            # that with `in_extra_time` False — i.e. every 26/27 match — this
            # evaluates to exactly the expression it always did.
            self.state.phase  = (MatchPhase.ADDED_TIME if self.state.in_extra_time
                                 else PhaseEngine.get_phase(minute))

            # ── MANAGER COLLECTION CHECKPOINT (Phase 7, opt-in) ──
            if _collection_checkpoint_hook is not None and minute % 5 == 0:
                _collection_checkpoint_hook(self, minute)

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
                # Phase 6 — push brain-manager sub urgency into the controller
                # so check_stamina_sub can modulate. At flag-off or when no
                # brain-manager is wired the dict stays empty → urgency_for
                # returns the neutral 0.5 (factor 1.0) so the base sub
                # probability is unchanged.
                if USE_MANAGER_BRAIN:
                    from manager_profile import Manager as _BrainManager
                    for _team, _mgr in (
                        (self.config.home_team, self.home_manager),
                        (self.config.away_team, self.away_manager),
                    ):
                        if _mgr is not None and isinstance(_mgr, _BrainManager) \
                                and hasattr(_mgr, "_current_urgency"):
                            self.sub_controller.set_manager_urgency(
                                _team, _mgr._current_urgency
                            )
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
                                weather_mult = (
                                    WeatherPhysics.stamina_drain_mult(self.weather_condition)
                                    if getattr(self.state, "weather_enabled", False) else 1.0
                                )
                                state.drain_baseline(team_style, intensity_mult=intensity_mult, weather_mult=weather_mult)
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

            # ── FEATURE #1/#2: IN-MATCH SHAPE (once per minute per team) ──
            # The manager reacts to the scoreline/clock by (1) adopting a
            # FORMATION STANCE (chasing shape, see-it-out block, man-down
            # compactness) and (2) committing the likely ball-carrier to an
            # ATTACK PATTERN (overload side, box midfield ...) for the next
            # ~5-minute chunk. Both are applied to the PositionEngine's home
            # anchors BEFORE the minute's sequences so the pass network /
            # width steering / restart snaps immediately reflect the new XI.
            from tactical_shapes import formation_stance_for
            from attack_patterns import pattern_for
            _poss_h, _poss_a = PossessionEngine.calculate_possession_split(
                *_brain_possession_profiles(
                    self.home_profile, self.away_profile,
                    self.home_manager, self.away_manager),
                self.state,
                self.config.home_team,
            )
            _gd_h = self.state.home_goals - self.state.away_goals
            for _team, _prof, _reds, _mgr, _is_home in (
                (self.config.home_team, self.home_profile,
                 self.state.home_red_cards, self.home_manager, True),
                (self.config.away_team, self.away_profile,
                 self.state.away_red_cards, self.away_manager, False),
            ):
                _stance = formation_stance_for(
                    _prof, self.state, _team, self.config.home_team,
                    manager=_mgr,
                )
                if self.position_engine.apply_formation_stance(
                    _team, _stance,
                    own_red_cards=_reds,
                    avg_stamina=self._avg_stamina(_team),
                ):
                    self._shape_apply_clock[_team] = self.state.match_clock_s
                self.state.team_stances[_team] = _stance

                # Posession-based pattern: the likely carrier of this minute
                # leans on its identity overload; the defending side holds a
                # pure defensive stance (no possession pattern).
                _likely_carrier = (self.config.home_team if _poss_h >= _poss_a
                                   else self.config.away_team)
                _chasing = (
                    (_gd_h <= -1 if _is_home else _gd_h >= 1)
                    and minute >= 60
                )
                _protecting = (
                    (_gd_h >= 1 if _is_home else _gd_h <= -1)
                    and minute >= 70
                )
                _pattern = pattern_for(
                    getattr(getattr(_prof, "style", None), "value", "balanced"),
                    self.state, _team, self.config.home_team, minute,
                    chasing=_chasing, protecting=_protecting,
                    manager=_mgr,
                )
                if _team != _likely_carrier:
                    _pattern = AttackPattern.NONE
                if self.position_engine.apply_attack_pattern(_team, _pattern):
                    self._shape_apply_clock[_team] = self.state.match_clock_s
                self.state.team_patterns[_team] = _pattern

                # Phase 8 v1 — coach width instruction ("stay wide" /
                # "tuck in", stored-posture read). Static / BALANCED /
                # flag OFF yields width 0 → apply_coach_width is a no-op.
                if USE_MANAGER_BRAIN:
                    from coach_instructions import instructions_for_manager as _instr_for
                    _ci = _instr_for(_mgr, _pattern)
                    if self.position_engine.apply_coach_width(
                            _team, _ci.width_cmd if _ci is not None else 0.0):
                        self._shape_apply_clock[_team] = self.state.match_clock_s

            # ── SIMULATE MINUTE ────────────────────────────────────
            # Single-clock bookkeeping: anchor the minute's start so the
            # off-ball integrator can measure the rest of the minute and keep
            # the global clock spanning the full 90'.
            self._minute_start_clock = self.state.match_clock_s
            self._on_ball_this_minute = set()
            self._sprint_state = {}
            self._patrol = {}
            self._chase_state = {}
            self._team_press_g_cache = {}
            self._offball_press_cache = {}
            self._minute_start_snapshot = {}
            for _t in (self.config.home_team, self.config.away_team):
                self._minute_start_snapshot.update(
                    self.position_engine.snapshot_positions(_t))
            if not self._top_speed_cache:
                for _tm, _pls in self.active_players.items():
                    for _p in _pls:
                        _pace = float(getattr(getattr(getattr(_p, "dna", None),
                                                     "physical", None), "pace", 60.0))
                        self._top_speed_cache[_p.name] = 5.0 + _pace * 0.042
                # Hand the SAME speeds to the run tracker, so its jump guard
                # uses the integrator's real limit. Without this the tracker
                # has no way to tell a 99 m possession-episode position sync
                # from a sprint, and fired run predicates off it (12.6% of
                # observed runs). Read from the engine rather than recomputed:
                # `top_speed_mpm` on the spatial state is a per-minute STEP
                # distance, not a speed, and conflating the two is what made two
                # earlier attempts report wildly wrong contamination figures.
                if self.run_tracker is not None:
                    self.run_tracker.set_top_speeds(self._top_speed_cache)
            self._simulate_minute(minute, TeamStyle)

            # ── CONTINUOUS OFF-BALL PHYSICS (single 10 Hz clock) ─────
            # Replace the old once-per-minute net drift with a per-tick
            # integrator: every off-ball player is moved toward a live
            # ball-compacted target each 0.1 s for the remainder of the
            # minute, and REAL path-distance + sprint SEGMENTS are recorded.
            # This makes off-ball distance/sprints physically measured (not a
            # snapshot delta) and keeps the whole match on one continuous
            # clock spanning the full 90'.
            total_seq = self._minute_home_seq + self._minute_away_seq
            if total_seq > 0:
                home_has_ball = self._minute_home_seq >= self._minute_away_seq
            else:
                home_has_ball = self.state.possession_team == self.config.home_team
            gd_home_now = self.state.home_goals - self.state.away_goals
            _danger = {
                self.config.home_team: self.threat.danger_at(self.config.home_team),
                self.config.away_team: self.threat.danger_at(self.config.away_team),
            }
            self._continuous_offball_phase(minute, home_has_ball, gd_home_now, _danger)

            # ── DEFENSIVE BLOCK COORDINATION (Checkpoint defensive-awareness) ──
            # The out-of-possession team with a live threat (danger ≥ 25, ball in
            # its own half) pulls its GK/CB/LB/RB/CDM line into a compact
            # goal-side block — ball-side CBs shift hardest, so the right/left
            # centre-back naturally covers the channel the ball is being played
            # into. No-op at low danger → baseline drift is untouched.
            blocking_team = (
                self.config.away_team if home_has_ball else self.config.home_team
            )
            attacking_team = (
                self.config.home_team if home_has_ball else self.config.away_team
            )
            def_att_right = self.position_engine.team_attacks_right.get(blocking_team, True)
            own_goal_x = 0.0 if def_att_right else 105.0
            block_danger = _danger.get(blocking_team, 0.0)
            if block_danger >= 25.0:
                block_profile = (
                    self.away_profile if blocking_team == self.config.away_team
                    else self.home_profile
                )
                self.position_engine.defensive_block(
                    blocking_team,
                    self.state.last_ball_x,
                    self.state.last_ball_y,
                    own_goal_x=own_goal_x,
                    danger_level=block_danger,
                    minute=minute,
                    defensive_line=getattr(block_profile, "defensive_line", 0.5),
                    compactness=getattr(block_profile, "compactness", 0.0),
                    attacking_team=attacking_team,
                )

            # Checkpoint 26 — refresh the velocity-aware pitch-control
            # cache from the just-updated drift velocities. Consumers
            # (e.g. winger half-space openness) read it via
            # position_engine.pitch_control_result/field.
            self.position_engine.update_pitch_control(
                self.config.home_team, self.config.away_team, minute=minute,
            )

            # ── GOALKEEPER ANCHOR INVARIANT ────────────────────────
            # Re-assert that no keeper finished the off-ball phase at the far
            # end of the pitch. The anchor itself only runs inside
            # `record_touch`, so it never saw the off-ball shape writes.
            # Measured pre-fix on a real match: 168 of 325 keeper writes at
            # x >= 40 (peaking 80-105), which is what pushed the home
            # keeper's average x to 26.5-27.6 against a ceiling of 25 in
            # tests.py::test_pass_network_positions_stay_realistic. Enforced
            # HERE, at the single boundary of the off-ball phase, so it is
            # one choke point rather than ~30 individual write sites, and a
            # shape rule added later cannot bypass it.
            for _t in (self.config.home_team, self.config.away_team):
                self.position_engine.enforce_gk_anchor(_t)

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
                "home_stance": getattr(
                    self.state.team_stances.get(self.config.home_team),
                    "value", "baseline",
                ),
                "away_stance": getattr(
                    self.state.team_stances.get(self.config.away_team),
                    "value", "baseline",
                ),
                "home_pattern": getattr(
                    self.state.team_patterns.get(self.config.home_team),
                    "value", "none",
                ),
                "away_pattern": getattr(
                    self.state.team_patterns.get(self.config.away_team),
                    "value", "none",
                ),
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
                        "physics_walk_time_s": activity.get("physics_walk_time_s", 0.0),
                        "physics_jog_time_s": activity.get("physics_jog_time_s", 0.0),
                        "physics_sprint_time_s": activity.get("physics_sprint_time_s", 0.0),
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

        # ── EXTRA TIME + PENALTY SHOOTOUT (audit §16 phase 5) ────────
        # Behind two config flags that both default to False, so a 26/27 league
        # match never reaches this code and its output is unchanged.
        if self.config.extra_time or self.config.penalties:
            if self._extra_time_needed():
                self._play_extra_time(_run_minute)
            if (self.config.penalties
                    and self.state.home_goals == self.state.away_goals):
                self._run_penalty_shootout()

        # Final whistle — any possession carry armed by the LAST sequence (a
        # turnover in the closing seconds that the next sequence never got to
        # consume) is moot: the match is over. Clear it so no dangling carry
        # leaks past full-time.
        self.state.possession_winner = ""

        # Final whistle — set minutes for everyone still on pitch.
        # Extra time counts towards minutes played, so a player who survives to
        # the end of a shootout-tied match is credited for it.
        total_mins = 90 + added + self.state.extra_time_minutes
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

        result = MatchResult(
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
            gps=self.gps,
            chronology=self.chronograph(),
            # ── knockout resolution (audit §16 phase 5) ──
            went_to_extra_time=self.state.went_to_extra_time,
            extra_time_minutes=self.state.extra_time_minutes,
            shootout=({
                "home_team": self.config.home_team,
                "away_team": self.config.away_team,
                "home_pens": self.state.home_pens,
                "away_pens": self.state.away_pens,
                "winner": self.state.shootout_winner,
                "rounds": self.state.shootout_rounds,
            } if self.state.shootout_played else None),
            winner_team=self._resolve_winner_team(),
            final_score=(self.state.home_goals, self.state.away_goals),
            aggregate=self._resolve_aggregate(),
        )
        # ── RUN TAXONOMIES, attached to the result ──────────────────────
        # Both live on the ENGINE, not the result, and neither was reachable
        # from an exporter that only sees the result. So they are attached here
        # rather than threaded through every call site.
        #
        #   run_profile           RunTracker's six GEOMETRICALLY OBSERVED types
        #                         (advance/overlap/underlap/far_side/forward/
        #                         support) - what the movement actually did.
        #   intended_run_profile  the behaviour engines' own vocabulary
        #                         (behind/cut/orbit/overlap/...) - what they
        #                         DECIDED. Set as plain attributes rather than
        #                         MatchResult fields so no caller that builds a
        #                         result by hand can be broken by them.
        #
        # This is the shape fact world.ingest already had to work around for
        # sub_controller: post-match facts that live on an object the result
        # does not carry are silently lost by an adapter reading the result
        # alone. Pinning it here is the fix, not another gap.
        result.run_profile = self.get_run_profile()
        result.intended_run_profile = self.intended_runs.profile()
        # Same shape trap as run_profile: a post-match fact that lives only on
        # the engine is lost by anything reading just the result, and the bar
        # that decides what counts as a run is exactly the thing you need to
        # see when the numbers look wrong.
        result.intended_run_diagnostics = self.intended_runs.diagnostics()

        # Phase 6 — Manager Brains end-of-match callback (opt-in): each wired
        # brain-manager closes its loop here — the mind updates composure/
        # stress from the final result and the episodic memory records the
        # conviction it committed to (see manager_profile.Manager.end_of_match).
        if USE_MANAGER_BRAIN:
            from manager_profile import Manager as _BrainManager
            for _team, _mgr in (
                (self.config.home_team, self.home_manager),
                (self.config.away_team, self.away_manager),
            ):
                if _mgr is not None and isinstance(_mgr, _BrainManager):
                    _mgr.end_of_match(result, _team)

        return result

    def chronograph(self) -> MatchChronology:
        """Stamp every chain event with its TRUE match-second on the single
        global clock.

        The engine keeps one continuous clock (``state.match_clock_s``) that the
        possession episodes advance via ``_absorb_motion``. The timeline's
        ``MatchEvent.second`` is only a bucket (mostly a placeholder random
        value); this recreates the real chronology afterwards using the exact
        window each chain actually occupied (recorded as ``_chain_clock_marks``).

        Purely additive: reads marks + timeline, mutates nothing, changes no
        outcome — a goal chain collapses to its chain-start clock because its
        motion was never folded (``motion_folded=False``).
        """
        idx_by_id = {id(e): i for i, e in enumerate(self.timeline)}
        events: List[TimedEvent] = []
        for mark in self._chain_clock_marks:
            n = len(mark.events)
            if n == 0:
                continue
            span = mark.end_clock - mark.start_clock
            for i, ev in enumerate(mark.events):
                if n == 1:
                    at = mark.start_clock + span * 0.5
                else:
                    at = mark.start_clock + span * i / (n - 1)
                if span > 0.0 and i < n - 1:
                    nxt = mark.start_clock + span * (i + 1) / (n - 1)
                    dur = nxt - at
                else:
                    dur = 0.0
                ev_type = getattr(ev.event_type, "name", str(ev.event_type))
                events.append(TimedEvent(
                    minute=mark.minute,
                    second=round(min(at % 60.0, 59.9), 1),
                    match_clock_s=round(at, 2),
                    duration=round(dur, 2),
                    event_type=ev_type,
                    team=ev.team,
                    player=ev.player,
                    secondary_player=ev.secondary_player,
                    location_x=ev.location_x,
                    location_y=ev.location_y,
                    end_x=ev.end_x,
                    end_y=ev.end_y,
                    goal=ev_type in ("GOAL", "OWN_GOAL", "PENALTY_SCORED"),
                    source_event_index=idx_by_id.get(id(ev), -1),
                ))
        events.sort(key=lambda te: (te.match_clock_s, te.minute))

        measured_play_s = sum(
            m.end_clock - m.start_clock
            for m in self._chain_clock_marks if m.motion_folded
        )
        match_duration_s = self.state.match_clock_s
        covered = {id(e) for m in self._chain_clock_marks for e in m.events}
        unmarked = sum(1 for e in self.timeline if id(e) not in covered)
        return MatchChronology(
            events=events,
            match_duration_s=round(match_duration_s, 2),
            measured_play_s=round(measured_play_s, 2),
            dead_time_s=round(max(0.0, match_duration_s - measured_play_s), 2),
            n_chains=len(self._chain_clock_marks),
            n_unmarked_events=unmarked,
        )

    # ── goal-mouth coordinate ───────────────────────────────────────
    def _goal_mouth(self, team: str) -> Tuple[float, float]:
        """The centre of the goal `team` attacks, in the home-attacks-right frame.

        Needed because `MatchEvent.location_x`/`location_y` DEFAULT to
        (50.0, 34.0) (`MatchEvent`, below) — the CENTRE SPOT. So an event built
        with no location does not read as absent, it reads as a measurement of
        the centre circle. A goal celebration or a goal ruled out for offside,
        recorded at the centre spot, is football nonsense that exports as
        truth. `team_attacks_right` is set once per team at `initialize_team`
        and never flipped at half-time (home is always True, away always
        False), which is the frame this whole file already uses.
        """
        right = True
        if self.position_engine is not None:
            right = self.position_engine.team_attacks_right.get(team, True)
        return (105.0, 34.0) if right else (0.0, 34.0)

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

        # ── WHERE the outgoing player was standing ──────────────────
        # Both the INJURY and the SUBSTITUTION event used to be built with no
        # location at all. That is NOT the same as "no location": `MatchEvent`
        # defaults `location_x`/`location_y` to (50.0, 34.0) — the CENTRE SPOT —
        # and nothing downstream can tell a default from a measurement. So
        # every substitution in a match was exported as having happened on the
        # centre circle. `tracked_position` rather than `get_position`, because
        # the latter answers (50.0, 34.0) for an untracked player — the same
        # value, for the same reason, one layer down.
        #
        # Computed BEFORE the injury block: the INJURY event is emitted first
        # on purpose (it is the cause), so it reads these two variables too.
        off_loc = (self.position_engine.tracked_position(name_off)
                   if self.position_engine is not None else None)
        off_x, off_y = off_loc if off_loc is not None else (50.0, 34.0)
        loc_source = "tracked" if off_loc is not None else "untracked_default"

        # Injury milestone: when the sub was forced by an in-match injury,
        # surface an INJURY event BEFORE the substitution so the timeline
        # (and exports) show the causality: the injury, then the change.
        # Build it first so it lands ahead of the SUBSTITUTION event below.
        ilabel = sub.get("reason", "tactical").lower()
        inj_event = None
        if ilabel == "injury":
            inj_meta = {}
            st_state = (self.sub_controller.stamina.get(name_off)
                        if self.sub_controller is not None else None)
            if st_state is not None:
                inj_meta.update({
                    "injury_type": st_state.injury_type or "unknown",
                    "injury_severity": round(st_state.injury_severity, 1),
                    "injury_minute": st_state.injury_minute,
                })
            inj_event = MatchEvent(
                minute=minute, second=0,
                event_type=EventType.INJURY,
                team=team,
                player=name_off,
                phase=self.state.phase,
                game_state=self.state.game_state,
                location_x=off_x,
                location_y=off_y,
                metadata={**inj_meta, "location_source": loc_source},
            )

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
            location_x=off_x,
            location_y=off_y,
            metadata={
                "reason":    sub.get("reason", "tactical"),
                "freshness": sub.get("freshness", 1.0),
                "stamina_at_exit": sub.get("stamina_at_exit", 0),
                "location_source": loc_source,
            }
        )
        if inj_event is not None:
            self.timeline.append(inj_event)
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
        self._chain_clock_marks.clear()

        # Opt-in perception seeding. A caller that wants replayable-but-varied
        # perception noise sets MatchConfig.perception_seed; everyone else
        # (including the live 26/27 path) keeps the legacy global config
        # untouched, so this is a no-op unless explicitly asked for.
        _apply_perception_seed(getattr(self.config, "perception_seed", None))

        # Opt-in continuous physics. Constructed here so the adapter's clock can
        # be seeded from the engine's own continuous clock (MatchState already
        # carries one), and only when asked for — with physics_enabled False the
        # engine never touches physics/ at all.
        self._init_physics()

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

    # ── CONTINUOUS PHYSICS (opt-in) ─────────────────────────────────────
    #
    # Everything below is additive and inert unless MatchConfig.physics_enabled
    # is True. With it False — the default, and what the live 26/27 season uses
    # — ``self.physics`` stays None and none of this is reachable, so the
    # existing code path is unchanged.
    #
    # The physics layer answers "how long does this physically take?" and
    # nothing else. What a player SHOULD do remains the brain's business, and
    # where a player IS remains PositionEngine's. This only supplies durations
    # and arrival times.

    def _init_physics(self) -> None:
        """Attach a physics adapter, if the config asked for one."""
        self.physics = None
        self.physics_unavailable = False
        # Clear the module slot FIRST, unconditionally. The possession chains
        # read it, and a stale adapter from the previous match would apply that
        # match's stamina and clock to this one — the id()-reuse class of bug,
        # in a single global.
        #
        # Looked up in sys.modules rather than imported. An earlier version did
        # `from physics.adapter import set_active_adapter` here, which meant the
        # DISABLED path imported the whole package — quietly breaking the
        # additive guarantee this engine relies on to leave 26/27 untouched.
        # If the module was never loaded there is no slot to clear, so there is
        # no reason to load it.
        try:
            import sys as _sys
            _mod = _sys.modules.get("physics.adapter")
            if _mod is not None:
                _mod.set_active_adapter(None)
        except Exception:
            pass
        if not getattr(self.config, "physics_enabled", False):
            return
        try:
            from physics.adapter import EngineAdapter
        except Exception:
            # Physics is an enhancement. If the package is unavailable the match
            # still plays — losing the layer is strictly better than losing a
            # season.
            self.physics_unavailable = True
            return

        self.physics = EngineAdapter(
            self,
            weather=getattr(self.config, "weather", None),
        )
        self.physics.sync_clock(self.state.match_clock_s)
        # Publish to the single module slot the possession chains read. Written
        # here and, crucially, ALSO written on the disabled path below, so the
        # slot can never outlive the match that filled it.
        from physics.adapter import set_active_adapter
        set_active_adapter(self.physics)

    @property
    def physics_enabled(self) -> bool:
        return getattr(self, "physics", None) is not None

    def resolve_pass_physically(self, start, end, kind: str = "short",
                                 *, intended: str = "",
                                 names=None):
        """Plan a pass in real coordinates and resolve it by arrival time.

        Returns the ``PassResolution``, or ``None`` when physics is disabled —
        so a caller can write the same code either way and let the flag decide.
        The engine does not yet route its own passes through here; this is the
        seam a caller uses, and the hook the internal routing will call.
        """
        if self.physics is None:
            return None
        self.physics.sync_clock(self.state.match_clock_s)
        return self.physics.resolve_pass(start, end, kind=kind,
                                         intended=intended, names=names)

    def physics_report(self) -> str:
        """Diagnostics for the physics layer, or a note that it is off."""
        if self.physics is None:
            return ("physics: disabled (MatchConfig.physics_enabled is False, "
                    "so this match ran entirely on the legacy path)")
        lines = [self.physics.report()]
        if getattr(self.config, "physics_strict", False):
            self.physics.world.assert_clean()
            lines.append("strict mode: no physics violations recorded")
        return "\n".join(lines)

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
        sub_bonus = (self.state.home_subs_made + self.state.away_subs_made) * 0.15
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
            *_brain_possession_profiles(
                self.home_profile, self.away_profile,
                self.home_manager, self.away_manager),
            self.state, home_team
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
            kickoff_this_seq = False
            if self.state.pending_kickoff_for:
                kickoff_team = self.state.pending_kickoff_for
                self.state.pending_kickoff_for = ""
                self.state.possession_team = kickoff_team
                self.state.possession_winner = ""  # restart owns possession
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
                
                # After kickoff, the sequence proceeds with kickoff_team in
                # possession. The kickoff team is NOT re-rolled by the normal
                # possession split below (that would randomise a restart away
                # ~50% of the time); it keeps the ball for this sequence.
                attacking_team = kickoff_team
                defending_team = home_team if kickoff_team == away_team else away_team
                kickoff_this_seq = True
                if kickoff_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
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
                self.state.possession_winner = ""  # restart owns possession
                if corner_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                # Anchor corner to last ball position — a corner doesn't
                # teleport the ball; it was won from a blocked shot/clearance
                # right there, and the set piece delivery is from that context.
                #
                # `attacks_right` was MISSING here, so it took the
                # `set_piece` default of True and every away-team corner was
                # resolved as if that team attacked the right-hand goal. Its
                # box grid, its taker, its delivery target, the contact point
                # and any loose-ball follow-up were all placed at the away
                # team's OWN end. Measured: 6 of 36 shots in a two-match
                # sample were taken on the shooting team's own half, all of
                # them the away side, split between the corner shot itself and
                # the open-play shot that followed the loose ball. The other
                # four `set_piece` call sites (penalty, offside FK, foul FK,
                # and the in-play branch) already passed it, which is why the
                # frame was right everywhere except corners.
                self.state.last_ball_x = min(105.0, max(83.0, self.state.last_ball_x))
                self.state.last_ball_y = max(5.0, min(63.0, self.state.last_ball_y))
                sp_result = ChainDispatcher.set_piece(
                    minute, corner_team, corner_opponent,
                    self.active_players.get(corner_team, []),
                    self.active_players.get(corner_opponent, []),
                    self.state, SituationType.CORNER,
                    attacks_right=(corner_team == home_team),
                    position_engine=self.position_engine,
                    routine=self._sp_routine(
                        corner_team, SituationType.CORNER, minute,
                        self.active_players.get(corner_team, [])),
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
                self.state.possession_winner = ""  # restart owns possession
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
                self.state.possession_winner = ""  # restart owns possession
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
                self.state.possession_winner = ""  # restart owns possession
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
                self.state.possession_winner = ""  # restart owns possession
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
                    routine=self._sp_routine(
                        fk_team, SituationType.DIRECT_FREEKICK, minute,
                        self.active_players.get(fk_team, []),
                        fk_context=(self.state.pending_offside_fk_x,
                                    self.state.pending_offside_fk_y)),
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

            # ── FOUL-AWARDED FREE KICK (consume) ────────────────────────
            # The counterpart to the foul award. Placed at the FOUL SPOT, not
            # at the offside spot, so `attacks_right` here is the foul victim's
            # direction and the spot is already in absolute pitch coordinates.
            # The chain decides direct-vs-crossed from the geometry
            # (event_chain._freekick_chain: x > 78 when attacking right),
            # which is why direct free kicks finally become reachable.
            pending_foul_home = self.state.pending_foul_fk_home > 0
            pending_foul_away = self.state.pending_foul_fk_away > 0
            if pending_foul_home or pending_foul_away:
                if pending_foul_home:
                    foul_fk_team = home_team
                    self.state.pending_foul_fk_home -= 1
                else:
                    foul_fk_team = away_team
                    self.state.pending_foul_fk_away -= 1
                foul_fk_opponent = (away_team if foul_fk_team == home_team
                                    else home_team)
                self.state.possession_team = foul_fk_team
                self.state.possession_winner = ""  # restart owns possession
                if foul_fk_team == home_team:
                    self._minute_home_seq += 1
                else:
                    self._minute_away_seq += 1
                fk_result = ChainDispatcher.set_piece(
                    minute, foul_fk_team, foul_fk_opponent,
                    self.active_players.get(foul_fk_team, []),
                    self.active_players.get(foul_fk_opponent, []),
                    self.state, SituationType.DIRECT_FREEKICK,
                    attacks_right=(foul_fk_team == home_team),
                    context_x=self.state.pending_foul_fk_x,
                    context_y=self.state.pending_foul_fk_y,
                    position_engine=self.position_engine,
                    routine=self._sp_routine(
                        foul_fk_team, SituationType.DIRECT_FREEKICK, minute,
                        self.active_players.get(foul_fk_team, []),
                        fk_context=(self.state.pending_foul_fk_x,
                                    self.state.pending_foul_fk_y)),
                )
                for ev in fk_result.events:
                    if ev.event_type == EventType.FREEKICK_WON:
                        ev.location_x = self.state.pending_foul_fk_x
                        ev.location_y = self.state.pending_foul_fk_y
                        break
                if self._absorb_chain(fk_result, minute): break
                continue

            # Decide which team has possession this sequence
            # weighted by possession split. The kickoff sequence keeps the ball
            # with the kicking team (set above) and is NOT re-rolled here.
            if not kickoff_this_seq:
                # Causal possession carry (Checkpoint): a team that just WON
                # the ball in open play is OWED this sequence — real possession
                # chains are causal, not a memoryless coin flip. Applied
                # ~POSSESSION_CARRY_PROB of the time so the possession-target
                # weighting and calibrated split stay enforceable.
                attacking_team, defending_team = self._resolve_sequence_attacker(
                    home_poss, home_team, away_team
                )
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
                avg_stamina=att_avg_stamina,
                manager=(self.home_manager if attacking_team == home_team
                         else self.away_manager),
                engine=self,
            )
            def_profile = TacticalAI.adjust(
                def_raw_profile, self.state, defending_team, home_team,
                red_cards_against=self.state.home_red_cards if defending_team != home_team
                                else self.state.away_red_cards,
                avg_stamina=def_avg_stamina,
                manager=(self.home_manager if defending_team == home_team
                         else self.away_manager),
                engine=self,
            )

            att_players = self.active_players.get(attacking_team, [])
            def_players = self.active_players.get(defending_team, [])

            if not att_players:
                continue

            # ── MOMENTUM & GAME STATE ────────────────────────────────
            att_manager = (self.home_manager if attacking_team == home_team
                           else self.away_manager)
            att_composure = getattr(att_manager, "composure_scale", lambda: 1.0)() \
                if att_manager is not None else 1.0
            momentum_mod    = MomentumEngine.get_attacking_probability_modifier(
                self.state, attacking_team, home_team)
            game_state_mod  = MomentumEngine.get_game_state_modifier(
                self.state, attacking_team, home_team, composure=att_composure)
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
                trans_cp = None
                if self.state.counterpress_active(defending_team):
                    trans_cp = {"active": True,
                                "x": self.state.counterpress_x,
                                "y": self.state.counterpress_y}
                trans_result = ChainDispatcher.transition(
                    minute, defending_team, attacking_team,
                    def_players, att_players, def_profile, self.state,
                    position_engine=self.position_engine,
                    attacks_right=attacks_right,
                    counterpress=trans_cp,
                )
                if self._absorb_chain(trans_result, minute): break
                if trans_result.possession_lost:
                    # Ball changes hands — the team that just lost it arms a
                    # counterpress burst at the recovery zone for the next few
                    # seconds of match clock. The recovering team is OWED the
                    # next sequence (real possession is causal), barring a
                    # restart redirect.
                    self._arm_possession_winner(defending_team, trans_result)
                    self.state.set_counterpress(
                        attacking_team,
                        self.state.last_ball_x,
                        self.state.last_ball_y,
                    )
                    # Next sequence starts with the recovering team
                    continue

            # ── POSSESSION SEQUENCE ──────────────────────────────────
            # Sequence length varies by style and game state
            seq_length = PossessionEngine.sequence_length(
                att_profile, self.state,
                oppressor_starve=getattr(def_profile, "starve", 0.0),
            )

            # Pass defending players into possession chain so it can
            # generate realistic pressure events at the right locations.
            # Checkpoint 15: the DEFENDING team's live (TacticalAI-adjusted)
            # press intensity and pressing style are passed through so the
            # possession chain resolves the correct pressing profile for the
            # cover-shadow geometry and per-profile press probabilities
            def_press = def_profile.press_intensity
            if getattr(self.state, "weather_enabled", False):
                def_press *= WeatherPhysics.pressing_intensity_mult(self.weather_condition)

            poss_cp = None
            if self.state.counterpress_active(defending_team):
                poss_cp = {"active": True,
                           "x": self.state.counterpress_x,
                           "y": self.state.last_ball_y}
            # Phase 8 v1 — coach-to-player instructions (opt-in): the
            # attacking team's live manager speaks to this sequence's
            # carriers (stored posture read — no decide() poll). Static /
            # BALANCED / flag OFF yields None → chain byte-identical.
            _coach_instr = None
            if USE_MANAGER_BRAIN:
                from coach_instructions import instructions_for_manager as _instr_for
                _att_mgr = (self.home_manager if attacking_team == home_team
                            else self.away_manager)
                _coach_instr = _instr_for(
                    _att_mgr, self.state.team_patterns.get(attacking_team))
            poss_result = ChainDispatcher.possession(
                minute, attacking_team, att_players,
                att_profile, self.state, seq_length,
                defending_players=def_players,
                position_engine=self.position_engine,
                context_x=self.state.last_ball_x,
                context_y=self.state.last_ball_y,
                attacks_right=attacks_right,
                def_press_intensity=def_press,
                def_style_key=def_raw_profile.style.value,
                att_style_key=att_raw_profile.style.value,
                counterpress=poss_cp,
                coach_instructions=_coach_instr,
            )
            self._credit_possession(
                attacking_team, poss_result.sequence_duration_s,
                poss_result.player_possession_s,
            )
            if self._absorb_chain(poss_result, minute): break

            if poss_result.possession_lost:
                # Turnover: the attacking team just lost the ball. The
                # recovering (defending) team is OWED the next open-play
                # sequence — real possession chains are causal. Arm their
                # counterpress burst anchored at the recovery zone (where the
                # ball ended) so the NEXT sequence's defending pressures spike
                # around that zone instead of reverting to the static line.
                self._arm_possession_winner(defending_team, poss_result)
                self.state.set_counterpress(
                    attacking_team,
                    self.state.last_ball_x,
                    self.state.last_ball_y,
                )
                if self._defensive_recovery(
                    minute, poss_result, attacking_team, defending_team,
                    att_players, def_players, attacks_right, def_avg_stamina,
                ):
                    break
                continue  # Next sequence starts with the recovering team (carried above)

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
                    # The possession chain already recorded WHO decided to
                    # shoot (`PossessionChain` sets `shoot_player` at all three
                    # SHOOT sites). Passing it makes the shot CAUSED by the
                    # play: the pass that delivered the ball is then the key
                    # pass and its passer the assist. It used to be dropped
                    # here, so the shooter was re-drawn by role weight and
                    # distance to the ball — which is how a goal ended up
                    # scored from 25 m by a man who never touched the ball.
                    shooter_name=poss_result.shoot_player,
                    # ...and the real passer, so the assist is TRUE TRACKING
                    # instead of a role-weighted random draw. "" means nobody
                    # passed to him, i.e. an UNASSISTED goal — which is a real
                    # and common outcome, and is recorded as absent rather than
                    # filled with a plausible name.
                    assister_name=poss_result.shoot_assister,
                )
                self._maybe_var_overturn(att_result, poss_result, minute, attacking_team)
                if self._absorb_chain(att_result, minute): break
                self._maybe_loose_ball(
                    minute, att_result, attacking_team, defending_team,
                    att_players, def_players, attacks_right,
                )
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
            # Checkpoint — TACKLE VOLUME fix (why teams were seeing too few
            # tackles):
            # Defensive contests were GATED behind a PRESS event having
            # happened in this exact sequence (the `pressure_occurred` guard
            # below). But a press is a narrow, probabilistically-scored event
            # (engagement range x zone prob x intensity), so many possessions
            # produced NO press roll at all — silently starving the entire
            # defensive chain (tackles + interceptions + blocks) of chances to
            # fire. That is precisely "a tackle = pressure but not vice versa":
            # proactive defending should generate its own contests, not wait
            # for a press event to happen to unlock them.
            #
            # Fix: the defensive contest now fires on its OWN probability,
            # proportional to the defending team's press_intensity (how
            # engaged the whole team is) and the phase multiplier — 
            # INDEPENDENT of whether a PRESS event happened to roll — so a
            # pressing side records tackles at realistic volume. A press event
            # in the sequence still nudges the chance up (pressing player IS
            # closer to the ball), but it is no longer a hard pre-requisite.
            pressure_occurred = any(
                e.event_type == EventType.PRESS for e in poss_result.events
            )
            base_contest = def_profile.press_intensity * PhaseEngine.press_mult(phase)
            # Whether a press event actually happened boosts the raw contest
            # chance (the pressing defender is tight to the ball).
            if pressure_occurred:
                base_contest += 0.18
            contest_prob = min(0.72, base_contest * 0.6)
            if random.random() < contest_prob:
                    last_evt = poss_result.events[-1]
                    ctx_x = last_evt.end_x if last_evt.end_x is not None else last_evt.location_x
                    ctx_y = last_evt.end_y if last_evt.end_y is not None else last_evt.location_y
                    danger = self.threat.danger_at(defending_team)
                    own_goal_x = 105.0 if attacks_right else 0.0
                    # Action type: learned DefensiveActionBrain when engaged,
                    # else the heuristic danger-scaled weights.  The physical
                    # clearance/block feasibility grip lives inside
                    # _def_action_choice (a "clearance" 60 m from our own goal
                    # line is a turnover, not an act).
                    action_type = _def_action_choice(
                        self, defending_team, attacking_team,
                        danger, ctx_x, ctx_y,
                        self._infer_aerial_ball(poss_result.events, last_evt),
                        self._contest_distance(defending_team, attacking_team,
                                               ctx_x, ctx_y),
                        attacks_right,
                        self.state.last_ball_x, self.state.last_ball_y,
                        press_occurred=pressure_occurred,
                    )
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
                        self._arm_possession_winner(defending_team, def_result)
                        continue  # Defense won the ball — next sequence is theirs

            # ── DIRECT CLEARANCE (even without press) ─────────────
            if not poss_result.possession_lost:
                last_evt = poss_result.events[-1]
                ctx_x = last_evt.end_x if last_evt.end_x is not None else last_evt.location_x
                ctx_y = last_evt.end_y if last_evt.end_y is not None else last_evt.location_y
                dangerous_def = ctx_x > 80 if attacks_right else ctx_x < 25
                if dangerous_def and random.random() < 0.50:
                    danger_now = self.threat.danger_at(defending_team)
                    # Same learned action controller as the open-play contest:
                    # the trigger (dead ball in our third) is a material FEASIBILITY
                    # fact, but WHICH act (clearance/block/tackle/interception) is
                    # the brain's call, not a hardcoded "clearance".
                    action_type = _def_action_choice(
                        self, defending_team, attacking_team,
                        danger_now, ctx_x, ctx_y,
                        self._infer_aerial_ball(poss_result.events, last_evt),
                        self._contest_distance(defending_team, attacking_team,
                                               ctx_x, ctx_y),
                        attacks_right,
                        self.state.last_ball_x, self.state.last_ball_y,
                        press_occurred=pressure_occurred,
                        fallback="clearance",
                        site="direct_clearance",
                    )
                    def_result = ChainDispatcher.defensive_action(
                        minute, defending_team, attacking_team,
                        def_players, att_players, self.state, action_type,
                        context_x=ctx_x, context_y=ctx_y,
                        attacks_right=attacks_right,
                        danger_level=danger_now,
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
                        self._arm_possession_winner(defending_team, def_result)
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
            elif att_gd >= 4:
                # Parking the bus — winning comfortably, barely attacking
                shot_prob *= 0.45
                _xg_quality_mult = 1.20
            elif att_gd == 3:
                # Well ahead — sitting deep, occasional break
                shot_prob *= 0.55
                _xg_quality_mult = 1.18
            elif att_gd >= 2:
                # Comfortable — conservative but not passive
                shot_prob *= 0.65
                _xg_quality_mult = 1.15
            elif att_gd == 1 and minute >= 75:
                shot_prob *= 0.82
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
            #
            # CALIBRATION (real-football shot origination): the per-minute
            # shot_prob funnel may only dispatch a shot when the ball is
            # INSIDE the attacking third (within ~42m of the goal it attacks).
            # Previously a team in midfield possession would fire the attack
            # chain from its own half and release 35-68m pot-shots at volume —
            # real teams take ~95% of their shots from the final third. The
            # distance propensity AT the contact point is then handled by the
            # AttackChain geometry selector (_select_action_from_position).
            ball_x = self.state.last_ball_x
            in_final_third = (
                (attacks_right and ball_x >= 63.0)
                or (not attacks_right and ball_x <= 42.0)
            )
            if random.random() < shot_prob and not poss_result.shot_taken \
                    and in_final_third:
                situation = self._determine_situation(att_profile, phase, style)

                if situation in (SituationType.CORNER, SituationType.DIRECT_FREEKICK,
                                  SituationType.CROSSED_FREEKICK, SituationType.PENALTY):
                    # Fix: anchor set piece to last ball state — the situation
                    # was generated from the ball's actual location, not a new
                    # independent random zone. Free kicks and corners happen
                    # where the ball was, not where a separate random draw lands.
                    if situation in (SituationType.DIRECT_FREEKICK, SituationType.CROSSED_FREEKICK):
                        # NOTE: this clamp is home-frame only and is NOT the
                        # reason direct free kicks never fire. Measured
                        # (_diag_fk_site.py, 1 match): 100% of free kicks reach
                        # the dispatcher via the OFFSIDE queue at 4627, which
                        # passes `pending_offside_fk_x` raw and unclamped;
                        # this branch is never taken. Ordinary fouls award
                        # cards/penalties but never queue a free kick at all.
                        fk_x = min(90.0, max(65.0, self.state.last_ball_x))
                        fk_y = max(15.0, min(53.0, self.state.last_ball_y))
                        self.state.last_ball_x = fk_x
                        self.state.last_ball_y = fk_y
                    sp_result = ChainDispatcher.set_piece(
                        minute, attacking_team, defending_team,
                        att_players, def_players, self.state, situation,
                        attacks_right=attacks_right,
                        context_x=(self.state.last_ball_x
                                   if situation in (SituationType.DIRECT_FREEKICK,
                                                    SituationType.CROSSED_FREEKICK) else None),
                        context_y=(self.state.last_ball_y
                                   if situation in (SituationType.DIRECT_FREEKICK,
                                                    SituationType.CROSSED_FREEKICK) else None),
                        position_engine=self.position_engine,
                        routine=self._sp_routine(
                            attacking_team, situation, minute, att_players),
                    )
                    if self._absorb_chain(sp_result, minute): break
                else:
                    # This funnel produced the BULK of the match's shots and it
                    # used to pass no names at all, so AttackChain drew the
                    # shooter by role weight and distance — a player who may
                    # never have touched the ball, shooting from a position he
                    # was not at. The possession episode that just ran knows
                    # exactly who holds the ball and who gave it to him, so
                    # pass both. This is the same causation the matrix hand-off
                    # above established; leaving it off here was why the engine
                    # could name a creator for 2 shots in a match while its own
                    # ledger found a real setup pass for 30.
                    #
                    # `ball_carrier` is empty when the possession did not end
                    # with the ball at a player's feet (a turnover, a throw-in, a
                    # goal kick). `_named_shooter` resolves "" to None, so the
                    # chain falls back exactly as it did before — we do not
                    # invent a name to fill the slot.
                    att_result = ChainDispatcher.attack(
                        minute, attacking_team, defending_team,
                        att_players, def_players,
                        att_profile, def_profile, self.state, situation,
                        position_engine=self.position_engine,
                        context_x=self.state.last_ball_x,
                        context_y=self.state.last_ball_y,
                        attacks_right=attacks_right,
                        shooter_name=poss_result.shoot_player
                                     or poss_result.ball_carrier,
                        assister_name=poss_result.shoot_assister
                                      or poss_result.ball_carrier_passed_by,
                    )
                    self._maybe_var_overturn(att_result, poss_result, minute, attacking_team)
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

            # ── FOUL -> FREE KICK AWARD ────────────────────────────────
            # A defensive foul is a free kick to the fouled side, taken from
            # the spot. This is the award that was missing: until now a foul
            # produced a card and maybe a penalty, and nothing else, so
            # `pending_offside_fk_for` was the only thing that ever fed
            # `_freekick_chain`. Queued (not taken inline) so it is consumed
            # at the top of the next minute's sequence loop, exactly like the
            # corner and penalty awards at 5637 / 5649 — an immediate restart
            # would restart play inside the same sequence that produced the
            # foul.
            #
            # A foul that CONVICTED a penalty is not also given a free kick:
            # the spot kick supersedes it, and `penalty_won` is the flag the
            # penalty award already keys off.
            #
            # Nor is a foul INSIDE the box given a direct free kick when the
            # referee declined to award a penalty. Measured on a real match,
            # this was awarding "direct free kicks" from x=101.0 — 4 m from
            # the goal line — because `foul_x` is clamped to 101.0 above and a
            # box foul that fails the conviction test still falls through
            # here. A dead ball 4 m out is a tap-in, not a free kick, and it
            # dragged the keeper to the goal line and dragged the wall on top
            # of him: `tests.py::test_pass_network_positions_stay_realistic`
            # failed with the home keeper averaging x=26.6 because of it.
            # In real football a box foul that is not a penalty is an INDIRECT
            # free kick, which cannot be shot; the honest model is to award
            # no set piece at all and let play restart, which is what leaving
            # the queue empty does. The foul and any card are still recorded.
            # Test the FOUL SPOT against the real penalty-area boundary
            # (x >= 88 attacking right / x <= 17 attacking left), NOT against
            # `ball_in_deny_zone` (x >= 84). The deny zone is a deliberately
            # generous "is there danger here" test used to scale penalty
            # CONVICTION; using it here as the box test excluded every
            # legitimate direct free kick taken from 78-84 m and cut the
            # measured rate to 1.5/match. A direct free kick from 80 m is
            # exactly the shot this whole branch exists for. The box is 88.
            in_box_foul = (foul_x >= 88.0) if att_right else (foul_x <= 17.0)

            # Only SOME fouls are awarded as a free kick. Awarding every one
            # produced 40 free kicks a match (22 fouls + 21 offside
            # restarts), and that volume — not the wall, and not the keeper
            # placement — is what broke
            # `tests.py::test_pass_network_positions_stay_realistic`: the home
            # keeper's average x rose to 27.6 against an asserted ceiling of
            # 25. Each awarded kick is a full restart that costs a sequence,
            # resets the shape and pulls the defensive block upfield, so
            # roughly doubling the restart count moves the keeper for reasons
            # that have nothing to do with set pieces. Isolated by disabling
            # this award alone, which made the test pass.
            #
            # Real football does not restart for every foul: many are played
            # on as an advantage, and the ones that do stop play tend to be in
            # the attacking half, where the restart actually matters. Keeping
            # the award to those is the honest restriction — it is a
            # consequence of where fouls happen, not a rate invented to make
            # a number fit.
            worth_restarting = (foul_x >= 63.0) if att_right else (foul_x <= 42.0)

            if (disc_result.foul_committed
                    and not disc_result.penalty_won
                    and not in_box_foul
                    and worth_restarting):
                if last_attacker == home_team:
                    self.state.pending_foul_fk_home += 1
                else:
                    self.state.pending_foul_fk_away += 1
                self.state.pending_foul_fk_x = foul_x
                self.state.pending_foul_fk_y = foul_y

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
        # Same learned action controller as the open-play contest: the deep
        # recovery is a genuine "get it away" moment, but the act (clearance
        # vs block vs winning it with a tackle/interception) is the brain's
        # call — not a hardcoded clearance-heavy table.
        action_type = _def_action_choice(
            self, defending_team, attacking_team,
            danger, ctx_x, ctx_y,
            self._infer_aerial_ball(poss_result.events, last_evt),
            self._contest_distance(defending_team, attacking_team,
                                   ctx_x, ctx_y),
            attacks_right,
            self.state.last_ball_x, self.state.last_ball_y,
            press_occurred=False,
            fallback="clearance",
            site="recovery",
        )
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

    def _maybe_var_overturn(self, att_result, poss_result, minute: int,
                            attacking_team: str) -> None:
        """VAR review for an open-play goal (delayed-offside overturn).

        If PossessionChain found the eventual scorer in an offside position
        earlier in THIS possession but the flag stayed down (the
        unflagged_offside_* stamp), the goal is reviewed and ruled out.
        Penalties and own goals are offside-exempt. Uses the shared
        event_chain.mark_var_disallowed so the timeline gets OFFSIDE (stats),
        GOAL is stripped, and the offside free kick is armed via the existing
        offside_detected path in _absorb_chain.
        """
        if att_result is None or not getattr(att_result, "goal_scored", False):
            return
        if getattr(att_result, "delayed_offside", False):
            return
        if getattr(att_result, "own_goal", False):
            return
        if not any(e.event_type == EventType.GOAL for e in att_result.events):
            return  # penalties / non-open-play goals are offside-exempt
        stamp = getattr(poss_result, "unflagged_offside_player", "") if poss_result else ""
        if not stamp or att_result.goal_scorer != stamp:
            return
        from event_chain import mark_var_disallowed
        mark_var_disallowed(
            att_result, minute, attacking_team,
            self.state.phase, self.state.game_state, stamp,
            getattr(poss_result, "unflagged_offside_x", 0.0),
            getattr(poss_result, "unflagged_offside_y", 0.0),
        )

    def _absorb_chain(self, chain_result, minute: int) -> bool:
        """
        Read a ChainResult and update the engine's timeline + match state.
        This is the single point where chain outputs become match facts.
        Also drains stamina from every player involved in each event.
        """
        from squad_manager import get_stamina_action
        # Phase 6 — Manager Brains: reset the "most recent goal" stamp, then
        # credit it from this chain BEFORE the event feed below so managers
        # can attribute GOAL/PENALTY_SCORED/OWN_GOAL events to the right
        # side (goal_team is the CREDITED team — for an own goal that is the
        # attacker, not the defender whose error put it in). A VAR-disallowed
        # goal (delayed_offside) is cancelled: no stamp.
        self.state.last_goal_team = None
        if chain_result.goal_scored and not getattr(chain_result, "delayed_offside", False):
            self.state.last_goal_team = chain_result.goal_team
        # Chronography: the global clock at the moment this chain starts being
        # absorbed. _absorb_motion (folded at the tail) will advance it by the
        # chain's physics duration; goal chains that early-return keep start==end.
        _chain_start_clock = self.state.match_clock_s
        # Track which players were actually ON THE BALL this chain so the
        # continuous off-ball integrator skips them (they're already moved by
        # the episode). NOTE: we use the chain's ball-action EVENTS, NOT the
        # episode's distance-stats dict — the episode traces every registered
        # player (both teams), so the stats dict would flag ~all 22 and freeze
        # the off-ball integrator for everyone.
        for ev in chain_result.events:
            if getattr(ev.event_type, "name", "") in self._ON_BALL_EVENTS:
                if getattr(ev, "player", None):
                    self._on_ball_this_minute.add(ev.player)
                if getattr(ev, "secondary_player", None):
                    self._on_ball_this_minute.add(ev.secondary_player)
        # Add all events to the timeline + drain stamina
        for event in chain_result.events:
            self.timeline.append(event)
            # Cognition tap: feed the event into the involved players' minds
            # (no-op unless engage_mind_observer() has been called).
            _cognition_observe(event)

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

            # Phase 6 — Manager Brains event feed (opt-in). After EACH event
            # the wired brain-managers (if any) are told what happened. The
            # mind ignores everything except goal outcomes, which it uses to
            # update composure/stress via Manager.on_event. Cheap no-op for
            # other event types.
            if USE_MANAGER_BRAIN:
                from manager_profile import Manager as _BrainManager
                _etype_name = getattr(event.event_type, "name", "")
                for _team, _mgr in (
                    (self.config.home_team, self.home_manager),
                    (self.config.away_team, self.away_manager),
                ):
                    if _mgr is not None and isinstance(_mgr, _BrainManager):
                        _mgr.on_event(_etype_name, self, _team)


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
                var_x, var_y = self._goal_mouth(chain_result.goal_team)
                if not self.quiet:
                    print(f"  ❌ GOAL RULED OUT (VAR/Offside)! {minute}' — {chain_result.goal_scorer}")
                self.timeline.append(MatchEvent(
                    minute=minute, second=0,
                    event_type=EventType.VAR_DISALLOWED_GOAL,
                    team=chain_result.goal_team,
                    player=chain_result.goal_scorer,
                    phase=self.state.phase, game_state=self.state.game_state,
                    # A goal ruled out for offside happened AT THE GOAL, not on
                    # the centre circle this used to default to. The offside
                    # itself was at `pending_offside_fk_x/y`, queued just above;
                    # the goal attempt is what this event records, so the goal
                    # mouth is the honest place.
                    location_x=var_x, location_y=var_y,
                    metadata={"location_source": "goal_mouth",
                              "offside_x": getattr(chain_result, 'offside_x', None),
                              "offside_y": getattr(chain_result, 'offside_y', None)},
                ))
                # The restart is the offside free kick queued by the
                # offside_detected block above — clear any kickoff so that
                # free kick is the only restart on the ball.
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
                self._chain_clock_marks.append(ChainClockMark(
                    minute=minute, start_clock=_chain_start_clock,
                    end_clock=self.state.match_clock_s, motion_folded=False,
                    events=list(chain_result.events),
                ))
                # GOAL CELEBRATION — the fixed real-world pause (10-30s) between
                # a goal and the center restart. The chain span above was folded
                # at the pre-celebration clock, so the celebration time lands in
                # dead_time rather than inflating the goal chain's measured play.
                # Drawn from the cosmetic RNG so presentation randomness never
                # perturbs the seeded football sequence.
                celebration_s = _COSMETIC_RNG.randint(10, 30)
                self.state.match_clock_s += celebration_s
                ce_min, ce_sec = divmod(int(self.state.match_clock_s), 60)
                # The scorer celebrates where he scored. This event used to
                # inherit the (50.0, 34.0) centre-spot default, so every
                # celebration in every match was exported on the centre circle.
                # Note this fires AFTER `_reset_positions_to_halves()`, so the
                # tracked positions are already the kickoff shape — the goal
                # mouth, not the engine's current state, is the truthful answer.
                cel_x, cel_y = self._goal_mouth(chain_result.goal_team)
                self.timeline.append(MatchEvent(
                    minute=ce_min, second=ce_sec,
                    event_type=EventType.GOAL_CELEBRATION,
                    team=chain_result.goal_team,
                    player=chain_result.goal_scorer,
                    phase=self.state.phase, game_state=self.state.game_state,
                    location_x=cel_x, location_y=cel_y,
                    metadata={"duration": celebration_s,
                              "location_source": "goal_mouth"},
                ))
                if not self.quiet:
                    print(f"  🎉 CELEBRATION ({celebration_s}s)"
                          f" @ {ce_min}'{ce_sec:02d}")
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

        # Checkpoint — fold this chain's continuous ball_path onto the single
        # global match clock so the whole 90' reads as one timeline (#1).
        self._absorb_motion(chain_result)

        # Feed the on-ball possession-episode movement into the virtual GPS so
        # its physical totals stay complete (the off-ball sampler only covers
        # the continuous integrator, not the episode's own physics trace).
        if self.gps is not None:
            pds = getattr(chain_result, "player_distance_stats", None) or {}
            if pds:
                self.gps.ingest_episode_stats(pds, minute)

        # Chronography: the chain's real span on the global clock (its motion is
        # folded above, so end_clock is strictly after start_clock).
        self._chain_clock_marks.append(ChainClockMark(
            minute=minute, start_clock=_chain_start_clock,
            end_clock=self.state.match_clock_s, motion_folded=True,
            events=list(chain_result.events),
        ))
        return False

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


# ─────────────────────────────────────────────
# MATCH RESULT — What simulation returns
# ─────────────────────────────────────────────

def _terrs(d: Dict[str, float]) -> str:
    if not d:
        return "n/a"
    return (f"{d.get('att_third', 0):.0f}/"
            f"{d.get('mid_third', 0):.0f}/"
            f"{d.get('def_third', 0):.0f}")


def _pcts(v: Optional[float]) -> str:
    return f"{v:.0f}%" if v is not None else "n/a"


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
    gps: Optional[VirtualGPS] = None
    chronology: Optional["MatchChronology"] = None
    # ── knockout resolution (audit §16 phase 5) ──
    went_to_extra_time: bool = False
    extra_time_minutes: int = 0
    shootout: Optional[Dict[str, Any]] = None
    winner_team: str = ""
    final_score: Optional[Tuple[int, int]] = None
    aggregate: Optional[Tuple[int, int]] = None

    @property
    def is_knockout_decided(self) -> bool:
        """True when a knockout tie has an outright winner (shootout or ET)."""
        return bool(self.winner_team)


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

    @property
    def home_possession_pct(self) -> float:
        tot = self.state.home_possession_s + self.state.away_possession_s
        return round(100.0 * self.state.home_possession_s / tot, 1) if tot > 0 else 50.0

    @property
    def away_possession_pct(self) -> float:
        return round(100.0 - self.home_possession_pct, 1)

    # ── GLOBAL CONTINUOUS TIMELINE (#1) ──────────────────────────
    @property
    def match_clock_s(self) -> float:
        return round(self.state.match_clock_s, 1)

    @property
    def full_match_ball_path(self) -> List[Dict[str, Any]]:
        """The whole 90' as one monotonic, time-ordered ball trajectory."""
        return self.state.match_ball_path

    @property
    def full_match_player_path(self) -> Dict[str, Any]:
        """~5 Hz per-player positions on the global clock (passive recorder).

        Returns {"path": {name: [(t, x, y), ...]}, "team": {name: team}}.
        """
        return {
            "path": self.state.player_path or {},
            "team": self.state.player_team or {},
        }

    # ── REAL PHYSICS DISTANCE / SPRINT TOTALS ──────────────────────
    def physics_totals(self) -> Dict[str, Dict[str, float]]:
        """Aggregate the per-minute physics store across the whole match.

        Distance/sprints here are derived from ACTUAL 10 Hz movement
        integration (on-ball via PossessionEpisode traces, off-ball via the
        continuous off-ball integrator) — not snapshot baselines. Returns a
        mapping player_name -> {distance_m, sprint_count,
        high_speed_sprint_count, top_speed_mps}.
        """
        out: Dict[str, Dict[str, float]] = {}
        for frame in self.position_log:
            for side in ("home", "away"):
                for row in frame.get(side, []):
                    name = row["player"]
                    agg = out.setdefault(name, {
                        "distance_m": 0.0, "sprint_count": 0.0,
                        "high_speed_sprint_count": 0.0, "top_speed_mps": 0.0,
                    })
                    agg["distance_m"] += float(row.get("physics_distance_m", 0.0))
                    agg["sprint_count"] += float(row.get("physics_sprint_count", 0.0))
                    agg["high_speed_sprint_count"] += float(
                        row.get("physics_high_speed_sprint_count", 0.0))
                    agg["top_speed_mps"] = max(
                        agg["top_speed_mps"],
                        float(row.get("physics_top_speed_mps", 0.0)))
        for agg in out.values():
            agg["distance_m"] = round(agg["distance_m"], 1)
            agg["sprint_count"] = round(agg["sprint_count"], 1)
            agg["high_speed_sprint_count"] = round(agg["high_speed_sprint_count"], 1)
            agg["top_speed_mps"] = round(agg["top_speed_mps"], 2)
        return out

    # ── POSSESSION ANALYTICS (#5) ────────────────────────────────
    def territorial_possession(self) -> Dict[str, Dict[str, float]]:
        """Share of ball-time spent in each third, per team (from the global
        continuous timeline). Answers 'territorial dominance' — not just who
        had the ball, but WHERE on the pitch they had it."""
        from collections import defaultdict
        counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total: Dict[str, float] = defaultdict(float)
        for pt in self.state.match_ball_path:
            x = pt.get("x", 52.5)
            team = pt.get("team", "")
            third = "def_third" if x < 35.0 else ("mid_third" if x < 70.0 else "att_third")
            counts[team][third] += 1.0
            total[team] += 1.0
        out: Dict[str, Dict[str, float]] = {}
        for team, d in counts.items():
            t = total[team] or 1.0
            out[team] = {k: round(100.0 * v / t, 1) for k, v in d.items()}
        return out

    def press_responsiveness(self) -> Dict[str, Optional[float]]:
        """Pressing intensity that actually converts: share of a team's press
        actions that are followed by a ball win (interception/tackle/recovery).
        Evaluated from the live timeline, not a roll."""
        from collections import defaultdict
        press = defaultdict(int)
        wins = defaultdict(int)
        for e in self.timeline:
            name = getattr(e.event_type, "name", "")
            if name == "PRESS":
                press[e.team] += 1
            elif name in ("INTERCEPTION", "TACKLE_WON", "BALL_RECOVERY"):
                wins[e.team] += 1
        out: Dict[str, Optional[float]] = {}
        teams = set(list(press.keys()) + list(wins.keys()))
        for team in teams:
            p = press.get(team, 0)
            w = wins.get(team, 0)
            out[team] = round(100.0 * w / p, 1) if p else None
        return out

    # ── SHOT SPEED (Opta-style) ─────────────────────────────────
    def shot_speed_stats(self) -> Dict[str, Any]:
        """Per-team average/max shot velocity across the match, Opta-style.

        Reads the launch velocity stamped on each shot-attempt event
        (``metadata["shot_speed_kmh"]`` — set by every shot-producing chain).
        GOAL events are excluded so a goal's paired SHOT_ON_TARGET isn't
        counted twice; headers count but contribute their (slower, ~5-12 m/s)
        velocities naturally.
        """
        attempt_types = frozenset({
            EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
            EventType.SHOT_BLOCKED, EventType.HIT_WOODWORK,
            EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
        })
        from collections import defaultdict
        by_team: Dict[str, list] = defaultdict(list)
        shots_total = 0
        shots_with_speed = 0
        for e in self.timeline:
            if e.event_type not in attempt_types:
                continue
            shots_total += 1
            md = e.metadata or {}
            speed = md.get("shot_speed_kmh")
            if speed is None:
                speed = (md.get("physics") or {}).get("shot_speed_kmh")
            if speed:
                shots_with_speed += 1
                by_team[e.team].append(float(speed))
        out: Dict[str, Any] = {}
        all_speeds: List[float] = []
        for team, vals in by_team.items():
            out[f"{team}"] = {
                "avg_kmh": round(sum(vals) / len(vals), 1),
                "max_kmh": round(max(vals), 1),
                "shots": len(vals),
            }
            all_speeds.extend(vals)
        out["match_avg_kmh"] = (
            round(sum(all_speeds) / len(all_speeds), 1) if all_speeds else None
        )
        out["shots_with_speed"] = shots_with_speed
        out["shots_total"] = shots_total
        out["coverage_pct"] = (
            round(100.0 * shots_with_speed / shots_total, 1) if shots_total else 0.0
        )
        return out

    def summary(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"  {self.score_str}",
            f"  xG: {self.config.home_team} {self.home_xg} — {self.away_xg} {self.config.away_team}",
            f"  Possession: {self.config.home_team} {self.home_possession_pct}% — "
            f"{self.away_possession_pct}% {self.config.away_team}",
            f"  Territorial (att%/mid%/def%): "
            f"{self.config.home_team} "
            f"{_terrs(self.territorial_possession().get(self.config.home_team, {}))} | "
            f"{self.config.away_team} "
            f"{_terrs(self.territorial_possession().get(self.config.away_team, {}))}",
            f"  Press responsiveness: "
            f"{_pcts(self.press_responsiveness().get(self.config.home_team))} / "
            f"{_pcts(self.press_responsiveness().get(self.config.away_team))}",
            f"  Goals: {len(self.goals)} | Cards: {len(self.cards)}",
            f"  Timeline events: {len(self.timeline)}",
            f"  Added time: {self.state.added_time}'",
            f"{'='*50}",
        ]
        for g in self.goals:
            assist = (f" (assist: {g.secondary_player})"
                      if g.secondary_player and g.event_type != EventType.OWN_GOAL else "")
            lines.append(f"  ⚽ {g.minute}' {g.player}{assist} — {g.team}")
        for c in self.cards:
            icon = "🟥" if c.event_type == EventType.RED_CARD else "🟨"
            lines.append(f"  {icon} {c.minute}' {c.player} — {c.team}")
        return "\n".join(lines)
