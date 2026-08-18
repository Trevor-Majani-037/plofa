"""
PLOFA 26/27 — PLAYER DNA MODULE
================================
player_dna.py

Philosophy:
    Every player has a DNA profile that determines WHO THEY ARE on the pitch.
    Stats don't get generated randomly — they emerge from a player's DNA
    interacting with match events.

    A 'clinical_finisher' doesn't just get a +20% shot multiplier.
    He has a higher finishing quality, lower miss rate on big chances,
    better composure under pressure, specific body part tendencies.

    DNA has three layers:
        1. ARCHETYPE    — The player's fundamental role (Poacher, Regista, Engine...)
        2. ATTRIBUTES   — 0-100 numeric ratings per skill domain
        3. TENDENCIES   — Behavioral probabilities (how often they attempt X)

Architecture:
    PlayerDNA          — Core DNA object attached to every player
    ArchetypeLibrary   — All PLOFA archetypes with their attribute templates
    DNAFactory         — Creates DNA from a player's position + specialty list
    PlayerProfile      — Full player object used by the simulation
"""

from __future__ import annotations
import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# ATTRIBUTE DOMAINS
# Each domain is a 0–100 rating
# ─────────────────────────────────────────────

@dataclass
class PhysicalAttributes:
    """What the body can do."""
    pace: float           = 60.0   # Raw sprint speed
    acceleration: float   = 60.0   # 0→top speed quickness
    stamina: float        = 65.0   # Endurance across 90 min
    strength: float       = 60.0   # Physical duels
    jumping: float        = 60.0   # Aerial ability
    agility: float        = 60.0   # Change of direction


@dataclass
class TechnicalAttributes:
    """Ball skills."""
    dribbling: float      = 60.0   # Carrying ball past opponents
    first_touch: float    = 62.0   # Receiving quality
    ball_control: float   = 62.0   # Keeping ball under pressure
    crossing: float       = 50.0   # Delivery from wide areas
    finishing: float      = 50.0   # Shot conversion
    long_shots: float     = 45.0   # Quality from distance
    heading: float        = 55.0   # Aerial finishing/defending
    free_kick: float      = 45.0   # Set piece delivery
    penalty_taking: float = 60.0   # Spot kick quality
    weak_foot: float      = 50.0   # Non-dominant foot quality


@dataclass
class MentalAttributes:
    """Decision making and psychology."""
    vision: float         = 60.0   # Seeing options
    composure: float      = 60.0   # Performance under pressure
    decisions: float      = 60.0   # Choosing right action
    positioning: float    = 60.0   # Finding the right space
    anticipation: float   = 60.0   # Reading the game
    work_rate: float      = 65.0   # Off-ball effort
    aggression: float     = 55.0   # Desire to win duels
    leadership: float     = 50.0   # Effect on teammates
    concentration: float  = 60.0   # Consistency over 90 min
    bravery: float        = 60.0   # Putting body on the line
    geometric_awareness: float = 50.0  # Reading spaces, covering teammates


@dataclass
class PassingAttributes:
    """Distribution quality."""
    short_passing: float  = 65.0   # Short range accuracy
    long_passing: float   = 55.0   # Long range accuracy
    through_balls: float  = 50.0   # Line-breaking passes
    switch_play: float    = 50.0   # Wide diagonal switches


@dataclass
class DefendingAttributes:
    """Defensive actions."""
    tackling: float       = 50.0   # Winning the ball
    marking: float        = 50.0   # Tracking runners
    interceptions: float  = 50.0   # Reading passes
    blocking: float       = 50.0   # Getting in the way
    clearances: float     = 50.0   # Getting ball away


@dataclass
class GoalkeeperAttributes:
    """GK-specific skills."""
    diving: float         = 60.0
    handling: float       = 60.0
    kicking: float        = 55.0
    reflexes: float       = 60.0
    positioning_gk: float = 60.0
    communication: float  = 55.0
    sweeping: float       = 50.0
    aerial_gk: float      = 55.0


# ─────────────────────────────────────────────
# BEHAVIORAL TENDENCIES
# How a player CHOOSES to act, not just what they can do
# ─────────────────────────────────────────────

@dataclass
class BehavioralTendencies:
    """
    Probabilities that shape what actions a player takes.
    These interact with match events to determine actual actions.
    """
    # ATTACKING TENDENCIES
    shoots_from_distance: float     = 0.15   # % of chances from outside box
    attempts_dribble: float         = 0.25   # % of carries that become dribbles
    cuts_inside: float              = 0.30   # Winger: cuts in vs crosses
    makes_runs_behind: float        = 0.35   # ST/LW/RW: runs in behind
    arrives_late: float             = 0.20   # CM/CDM: late box arrivals

    # MODERN WINGER TENDENCIES (Checkpoint 18)
    # Modern wingers (Vini Jr, Saka, Salah, Doku) are touchline-hugging
    # flank attackers, NOT drifting number-10s. The middle of the pitch is
    # always full — a #10 owns that space — and a winger who drifts inside
    # leaves his flank open and crowds his own teammates.
    hugs_touchline: float           = 0.70   # Stays wide on the flank channel
    drives_byline: float            = 0.50   # Attacks the touchline→byline corridor
    attacks_fullback_1v1: float     = 0.55   # Isolates the fullback and takes him on
    late_box_runs: float            = 0.45   # Late back-post arrivals (Saka/Vini)
    crosses_from_wide: float        = 0.45   # Delivers from the wide crossing zone

    # PASSING TENDENCIES
    plays_through_ball: float       = 0.10   # % of key passes that are through balls
    switches_play: float            = 0.08   # % of passes that are switches
    plays_safe: float               = 0.50   # % of passes that are backward/sideways
    presses_high: float             = 0.40   # How often presses opponent GK/CB

    # DEFENSIVE TENDENCIES
    tackles_aggressively: float     = 0.40   # Dives in vs stays on feet
    holds_position: float           = 0.60   # Stays in shape vs follows runner
    attacks_the_ball: float         = 0.55   # Goes for header vs waits

    # PHYSICAL TENDENCIES
    sprints_frequently: float       = 0.50   # High-intensity runs per opportunity
    conserves_energy: float         = 0.20   # Jogs when could sprint (early game)

    # RISK TENDENCIES
    commits_fouls: float            = 0.15   # Fouls vs lets player go
    dives: float                    = 0.05   # Wins soft fouls
    argues_with_ref: float          = 0.10   # Gets extra cautions


# ─────────────────────────────────────────────
# FORM & FATIGUE STATE
# Changes match-to-match based on simulation history
# ─────────────────────────────────────────────

@dataclass
class PlayerFormState:
    """
    Persistent across matches. Affects performance multipliers.
    This is the 'beyond the pitch' scaffold.
    """
    # Current form (0-100, 50 = neutral)
    confidence: float       = 50.0
    # Goals scored in last 5 matches (for hot streak detection)
    recent_goals: List[int]  = field(default_factory=lambda: [0,0,0,0,0])
    recent_ratings: List[float] = field(default_factory=lambda: [6.0]*5)

    # Injury state
    is_injured: bool         = False
    injury_type: str         = "none"
    matches_remaining_out: int = 0

    # Fatigue from previous match (0=fresh, 100=exhausted)
    fatigue_level: float     = 0.0

    # Minutes load (cumulative this season)
    season_minutes: int      = 0

    @property
    def form_multiplier(self) -> float:
        """Convert confidence to performance multiplier (0.80–1.20)."""
        return 0.80 + (self.confidence / 100.0) * 0.40

    @property
    def fatigue_multiplier(self) -> float:
        """Higher fatigue = worse performance (1.0 → 0.85 at max)."""
        return max(0.85, 1.0 - (self.fatigue_level / 100.0) * 0.15)

    @property
    def is_on_hot_streak(self) -> bool:
        return sum(self.recent_goals) >= 4 and self.confidence >= 70

    @property
    def is_in_slump(self) -> bool:
        return sum(self.recent_goals) == 0 and self.confidence <= 30

    def update_after_match(self, rating: float, goals: int, assists: int):
        """Update form state after a match."""
        self.recent_ratings.pop(0)
        self.recent_ratings.append(rating)
        self.recent_goals.pop(0)
        self.recent_goals.append(goals)

        # Confidence update
        if rating >= 7.5:
            self.confidence = min(100, self.confidence + 8)
        elif rating >= 7.0:
            self.confidence = min(100, self.confidence + 4)
        elif rating >= 6.5:
            self.confidence = min(100, self.confidence + 1)
        elif rating >= 6.0:
            pass  # Neutral
        elif rating >= 5.5:
            self.confidence = max(0, self.confidence - 4)
        else:
            self.confidence = max(0, self.confidence - 8)

        # Bonus for goals/assists
        self.confidence = min(100, self.confidence + goals * 3 + assists * 1.5)

        # Fatigue increases (resets with rest between matches)
        self.fatigue_level = min(100, self.fatigue_level + random.uniform(15, 30))


# ─────────────────────────────────────────────
# PLAYER DNA — The complete genetic profile
# ─────────────────────────────────────────────

@dataclass
class PlayerDNA:
    """
    The complete DNA of a PLOFA player.
    Every stat generated in a match is a function of this DNA
    interacting with match events.
    """
    name: str
    position: str
    age: int                = 24
    nationality: str        = "Tolandian"
    is_superstar: bool      = False

    # Attribute domains
    physical: PhysicalAttributes       = field(default_factory=PhysicalAttributes)
    technical: TechnicalAttributes     = field(default_factory=TechnicalAttributes)
    mental: MentalAttributes           = field(default_factory=MentalAttributes)
    passing: PassingAttributes         = field(default_factory=PassingAttributes)
    defending: DefendingAttributes     = field(default_factory=DefendingAttributes)
    gk_attrs: GoalkeeperAttributes     = field(default_factory=GoalkeeperAttributes)

    # Behavioral profile
    tendencies: BehavioralTendencies   = field(default_factory=BehavioralTendencies)

    # Form & fatigue (live state)
    form: PlayerFormState              = field(default_factory=PlayerFormState)

    # Identity
    specialties: List[str]             = field(default_factory=list)
    preferred_foot: str                = "right"
    footedness: str                    = "right"   # right/left/both
    is_set_piece_taker: bool           = False
    archetype: str                     = "generic"

    # Match state (reset each match)
    is_starter: bool                   = True
    sub_in_minute: Optional[int]       = None
    sub_out_minute: Optional[int]      = None
    minutes_played: int                = 0

    soul: Optional[PlayerSoul] = None

    # Live in-match performance multiplier (0.60–1.0).
    # Set every minute by MatchEngine from squad_manager's
    # PlayerStaminaState.performance_mult. The DNAFactory.get_* lookups
    # below read this so tired players ACTUALLY get worse — previously
    # stamina drained but nothing consumed the number, so a player at
    # 5% in the 88th minute took shots and won dribbles at kickoff rates.
    live_performance_mult: float       = 1.0

    # ── DERIVED PROPERTIES ────────────────────────────────────

    @property
    def overall_rating(self) -> float:
        """FIFA-style overall based on position."""
        pos = self.position
        if pos == 'GK':
            return np.mean([
                self.gk_attrs.diving, self.gk_attrs.reflexes,
                self.gk_attrs.handling, self.gk_attrs.positioning_gk,
                self.gk_attrs.kicking, self.gk_attrs.sweeping,
            ])
        elif pos in ['CB', 'LB', 'RB']:
            return np.mean([
                self.physical.pace * 0.8,
                self.physical.strength,
                self.defending.tackling * 1.3,
                self.defending.marking * 1.3,
                self.defending.interceptions * 1.2,
                self.mental.positioning * 1.1,
                self.passing.short_passing * 0.7,
            ]) / 1.07  # Normalize weights
        elif pos in ['CDM', 'CM']:
            return np.mean([
                self.defending.tackling,
                self.defending.interceptions,
                self.passing.short_passing * 1.2,
                self.passing.long_passing,
                self.mental.vision * 1.1,
                self.mental.positioning,
                self.physical.stamina * 1.1,
            ]) / 1.07
        elif pos == 'CAM':
            return np.mean([
                self.technical.dribbling,
                self.passing.through_balls * 1.3,
                self.passing.short_passing * 1.1,
                self.mental.vision * 1.3,
                self.technical.finishing * 0.9,
                self.mental.composure,
            ]) / 1.07
        elif pos in ['LW', 'RW']:
            return np.mean([
                self.physical.pace * 1.2,
                self.technical.dribbling * 1.3,
                self.technical.crossing,
                self.technical.finishing * 0.9,
                self.mental.decisions,
                self.physical.agility * 1.1,
            ]) / 1.07
        elif pos in ['ST', 'CF']:
            return np.mean([
                self.technical.finishing * 1.4,
                self.mental.composure * 1.2,
                self.physical.pace * 0.9,
                self.physical.strength * 0.8,
                self.technical.heading * 0.8,
                self.mental.positioning * 1.1,
            ]) / 1.07
        else:
            return np.mean([
                self.technical.ball_control,
                self.passing.short_passing,
                self.mental.decisions,
                self.physical.stamina,
            ])

    @property
    def effective_finishing(self) -> float:
        """Finishing quality adjusted for form, fatigue AND live stamina."""
        base = self.technical.finishing
        return base * self.form.form_multiplier * self.form.fatigue_multiplier * self.live_performance_mult

    @property
    def effective_passing(self) -> float:
        """Passing quality adjusted for form, fatigue AND live stamina."""
        base = (self.passing.short_passing + self.passing.long_passing) / 2
        return base * self.form.form_multiplier * self.form.fatigue_multiplier * self.live_performance_mult

    @property
    def effective_dribbling(self) -> float:
        """Dribbling quality adjusted for form AND live stamina."""
        return self.technical.dribbling * self.form.form_multiplier * self.live_performance_mult

    @property
    def press_resistance(self) -> float:
        """How well player retains ball under pressure (0-100)."""
        return (self.technical.ball_control + self.mental.composure) / 2

    @property
    def aerial_dominance(self) -> float:
        """Combined aerial threat/defending (0-100)."""
        return (self.physical.jumping + self.technical.heading + self.mental.bravery) / 3


# ─────────────────────────────────────────────
# ARCHETYPE LIBRARY
# Defines attribute templates for every player archetype
# ─────────────────────────────────────────────

class ArchetypeLibrary:
    """
    Every archetype is a TEMPLATE of attribute ranges.
    DNA Factory uses these to create individual players with
    variation within the archetype's natural range.

    Format: attribute_path -> (min, max)
    """

    ARCHETYPES: Dict[str, Dict] = {

        # ── GOALKEEPERS ──────────────────────────────────────

        "shot_stopper": {
            "description": "Traditional goalkeeper, elite reflexes, stays on line",
            "positions": ["GK"],
            "gk_attrs.diving":         (82, 94),
            "gk_attrs.reflexes":       (84, 96),
            "gk_attrs.handling":       (72, 84),
            "gk_attrs.positioning_gk": (78, 88),
            "gk_attrs.kicking":        (55, 70),
            "gk_attrs.sweeping":       (55, 68),
            "gk_attrs.aerial_gk":      (72, 84),
            "mental.composure":        (72, 88),
            "tendencies.attacks_the_ball": 0.45,
        },

        "sweeper_keeper": {
            "description": "Commands the area, initiates from back, high line",
            "positions": ["GK"],
            "gk_attrs.diving":         (72, 84),
            "gk_attrs.reflexes":       (74, 86),
            "gk_attrs.handling":       (78, 88),
            "gk_attrs.positioning_gk": (82, 92),
            "gk_attrs.kicking":        (78, 90),
            "gk_attrs.sweeping":       (84, 95),
            "gk_attrs.aerial_gk":      (80, 90),
            "passing.short_passing":   (68, 80),
            "passing.long_passing":    (70, 84),
            "physical.pace":           (55, 68),
            "mental.decisions":        (76, 88),
            "tendencies.attacks_the_ball": 0.65,
            "tendencies.presses_high": 0.55,
        },

        "distribution_gk": {
            "description": "Build-up specialist GK, plays out from the back",
            "positions": ["GK"],
            "gk_attrs.diving":         (72, 83),
            "gk_attrs.reflexes":       (72, 83),
            "gk_attrs.handling":       (74, 85),
            "gk_attrs.positioning_gk": (76, 87),
            "gk_attrs.kicking":        (84, 95),
            "gk_attrs.sweeping":       (72, 84),
            "gk_attrs.communication":  (82, 93),
            "passing.short_passing":   (78, 90),
            "passing.long_passing":    (80, 92),
            "mental.vision":           (72, 84),
            "tendencies.plays_safe":   0.70,
        },

        # ── CENTER BACKS ─────────────────────────────────────

        "ball_playing_cb": {
            "description": "Comfortable on the ball, initiates from deep",
            "positions": ["CB"],
            "defending.tackling":      (72, 84),
            "defending.marking":       (74, 85),
            "defending.interceptions": (72, 84),
            "defending.clearances":    (68, 80),
            "passing.short_passing":   (78, 90),
            "passing.long_passing":    (76, 88),
            "passing.switch_play":     (72, 86),
            "mental.vision":           (70, 82),
            "mental.composure":        (76, 88),
            "physical.strength":       (68, 80),
            "tendencies.plays_safe":   0.55,
            "tendencies.switches_play": 0.18,
        },

        "stopper_cb": {
            "description": "Dominant defender, wins everything, direct clearances",
            "positions": ["CB"],
            "defending.tackling":      (82, 93),
            "defending.marking":       (80, 91),
            "defending.blocking":      (78, 90),
            "defending.clearances":    (80, 92),
            "physical.strength":       (82, 94),
            "physical.jumping":        (80, 92),
            "technical.heading":       (80, 92),
            "mental.aggression":       (74, 87),
            "mental.bravery":          (80, 92),
            "passing.short_passing":   (55, 68),
            "tendencies.tackles_aggressively": 0.70,
            "tendencies.attacks_the_ball":     0.75,
        },

        "sweeper_cb": {
            "description": "Reads play, covers space, intercepts rather than tackles",
            "positions": ["CB"],
            "defending.interceptions": (82, 93),
            "defending.marking":       (78, 90),
            "defending.tackling":      (70, 82),
            "mental.anticipation":     (82, 93),
            "mental.positioning":      (82, 93),
            "physical.pace":           (68, 80),
            "mental.concentration":    (80, 91),
            "tendencies.holds_position":       0.75,
            "tendencies.tackles_aggressively": 0.30,
        },

        # ── FULLBACKS ─────────────────────────────────────────

        "attacking_fullback": {
            "description": "Wide threat, high crosses, gets forward constantly",
            "positions": ["LB", "RB"],
            "physical.pace":           (76, 88),
            "physical.stamina":        (78, 90),
            "technical.crossing":      (76, 88),
            "technical.dribbling":     (68, 80),
            "defending.tackling":      (68, 80),
            "defending.marking":       (66, 78),
            "passing.short_passing":   (70, 82),
            "mental.work_rate":        (80, 92),
            "tendencies.cuts_inside":  0.20,
            "tendencies.sprints_frequently": 0.75,
            "tendencies.presses_high": 0.55,
        },

        "inverted_fullback": {
            "description": "Tucks inside to midfield, builds from deep",
            "positions": ["LB", "RB"],
            "physical.pace":           (70, 82),
            "physical.stamina":        (76, 88),
            "technical.dribbling":     (70, 82),
            "passing.short_passing":   (76, 88),
            "passing.switch_play":     (72, 84),
            "mental.vision":           (70, 82),
            "mental.positioning":      (72, 84),
            "defending.tackling":      (66, 78),
            "tendencies.cuts_inside":  0.65,
            "tendencies.plays_safe":   0.55,
            "tendencies.switches_play": 0.20,
        },

        "defensive_fullback": {
            "description": "Stays back, first job is defending",
            "positions": ["LB", "RB"],
            "defending.tackling":      (76, 88),
            "defending.marking":       (78, 90),
            "defending.interceptions": (72, 84),
            "physical.strength":       (72, 84),
            "mental.concentration":    (76, 88),
            "mental.positioning":      (76, 88),
            "physical.pace":           (66, 78),
            "technical.crossing":      (52, 66),
            "tendencies.holds_position":       0.78,
            "tendencies.tackles_aggressively": 0.55,
            "tendencies.sprints_frequently":   0.40,
        },

        # ── DEFENSIVE MIDFIELDERS ─────────────────────────────

        "anchor": {
            "description": "Sits in front of defence, screens, rarely advances",
            "positions": ["CDM"],
            "defending.tackling":      (78, 90),
            "defending.interceptions": (78, 90),
            "defending.blocking":      (72, 84),
            "mental.positioning":      (80, 92),
            "mental.concentration":    (78, 90),
            "mental.geometric_awareness": (65, 76),
            "physical.strength":       (76, 88),
            "passing.short_passing":   (70, 82),
            "tendencies.holds_position":  0.82,
            "tendencies.plays_safe":      0.70,
            "tendencies.tackles_aggressively": 0.45,
        },

        "ball_winning_mid": {
            "description": "Presses relentlessly, wins ball high, aggressive",
            "positions": ["CDM", "CM"],
            "defending.tackling":      (80, 92),
            "defending.interceptions": (76, 88),
            "mental.aggression":       (78, 90),
            "mental.work_rate":        (82, 93),
            "mental.geometric_awareness": (60, 72),
            "physical.stamina":        (80, 92),
            "physical.strength":       (74, 86),
            "passing.short_passing":   (62, 74),
            "tendencies.tackles_aggressively": 0.72,
            "tendencies.presses_high":         0.68,
            "tendencies.commits_fouls":        0.22,
        },

        "regista": {
            "description": "Deep-lying playmaker, orchestrates from back",
            "positions": ["CDM", "CM"],
            "passing.short_passing":   (82, 94),
            "passing.long_passing":    (82, 94),
            "passing.through_balls":   (78, 90),
            "passing.switch_play":     (80, 92),
            "mental.vision":           (84, 95),
            "mental.decisions":        (80, 92),
            "mental.composure":        (80, 92),
            "mental.geometric_awareness": (80, 91),
            "technical.first_touch":   (80, 92),
            "physical.stamina":        (72, 84),
            "tendencies.plays_safe":   0.45,
            "tendencies.switches_play": 0.25,
            "tendencies.plays_through_ball": 0.20,
            "tendencies.presses_high": 0.20,
        },

        # ── CENTRAL MIDFIELDERS ───────────────────────────────

        "box_to_box": {
            "description": "Complete midfielder, both ends, covers every blade of grass",
            "positions": ["CM"],
            "physical.stamina":        (82, 94),
            "physical.pace":           (70, 82),
            "mental.work_rate":        (84, 95),
            "mental.geometric_awareness": (68, 80),
            "defending.tackling":      (70, 82),
            "defending.interceptions": (68, 80),
            "passing.short_passing":   (74, 86),
            "technical.finishing":     (62, 74),
            "tendencies.arrives_late": 0.40,
            "tendencies.sprints_frequently": 0.65,
            "tendencies.presses_high": 0.55,
        },

        "deep_playmaker": {
            "description": "Controls tempo from CM position, high pass volume",
            "positions": ["CM", "CDM"],
            "passing.short_passing":   (80, 92),
            "passing.long_passing":    (76, 88),
            "mental.vision":           (78, 90),
            "mental.decisions":        (78, 90),
            "mental.composure":        (76, 88),
            "mental.geometric_awareness": (78, 90),
            "physical.stamina":        (74, 86),
            "technical.first_touch":   (76, 88),
            "tendencies.plays_safe":   0.55,
            "tendencies.switches_play": 0.18,
        },

        "progressive_midfielder": {
            "description": "Carries ball forward, drives through lines",
            "positions": ["CM", "CAM"],
            "technical.dribbling":     (74, 86),
            "physical.pace":           (72, 84),
            "mental.vision":           (74, 86),
            "mental.decisions":        (72, 84),
            "mental.geometric_awareness": (74, 86),
            "passing.short_passing":   (74, 86),
            "passing.through_balls":   (72, 84),
            "physical.stamina":        (76, 88),
            "tendencies.attempts_dribble": 0.35,
            "tendencies.makes_runs_behind": 0.30,
        },

        # ── ATTACKING MIDFIELDERS ─────────────────────────────

        "classic_ten": {
            "description": "Pure number 10, vision, key passes, between the lines",
            "positions": ["CAM"],
            "mental.vision":           (84, 95),
            "mental.decisions":        (80, 92),
            "mental.composure":        (78, 90),
            "mental.geometric_awareness": (82, 93),
            "passing.through_balls":   (82, 94),
            "passing.short_passing":   (80, 92),
            "technical.first_touch":   (82, 94),
            "technical.dribbling":     (74, 86),
            "technical.finishing":     (68, 80),
            "tendencies.plays_through_ball": 0.25,
            "tendencies.plays_safe":   0.35,
        },

        "shadow_striker": {
            "description": "CAM who gets in the box, second striker tendencies",
            "positions": ["CAM"],
            "technical.finishing":     (76, 88),
            "mental.positioning":      (78, 90),
            "mental.anticipation":     (78, 90),
            "mental.geometric_awareness": (72, 84),
            "physical.pace":           (72, 84),
            "technical.dribbling":     (72, 84),
            "mental.composure":        (74, 86),
            "tendencies.makes_runs_behind": 0.50,
            "tendencies.arrives_late": 0.45,
            "tendencies.shoots_from_distance": 0.20,
        },

        # ── WINGERS ───────────────────────────────────────────

        "traditional_winger": {
            "description": "Wide, gets to the byline, whips crosses in",
            "positions": ["LW", "RW"],
            "physical.pace":           (80, 92),
            "physical.acceleration":   (80, 93),
            "technical.crossing":      (76, 88),
            "technical.dribbling":     (72, 84),
            "physical.stamina":        (74, 86),
            "mental.work_rate":        (74, 86),
            "tendencies.cuts_inside":  0.20,
            "tendencies.sprints_frequently": 0.72,
            "tendencies.attempts_dribble":   0.45,
            # Modern winger tendencies (Checkpoint 18)
            "tendencies.hugs_touchline":      0.90,
            "tendencies.drives_byline":       0.80,
            "tendencies.attacks_fullback_1v1": 0.60,
            "tendencies.late_box_runs":       0.45,
            "tendencies.crosses_from_wide":   0.70,
        },

        "inverted_winger": {
            "description": "Cuts inside to shoot, false winger, opposite foot",
            "positions": ["LW", "RW"],
            "physical.pace":           (78, 90),
            "physical.acceleration":   (78, 91),
            "technical.dribbling":     (78, 90),
            "technical.finishing":     (72, 84),
            "technical.long_shots":    (70, 82),
            "mental.composure":        (72, 84),
            "tendencies.cuts_inside":  0.72,
            "tendencies.shoots_from_distance": 0.28,
            "tendencies.attempts_dribble":     0.50,
            # Modern winger tendencies (Checkpoint 18)
            "tendencies.hugs_touchline":      0.85,
            "tendencies.drives_byline":       0.30,
            "tendencies.attacks_fullback_1v1": 0.75,
            "tendencies.late_box_runs":       0.55,
            "tendencies.crosses_from_wide":   0.30,
        },

        "pressing_winger": {
            "description": "Defensive duty winger, high press, wins ball high",
            "positions": ["LW", "RW"],
            "physical.pace":           (78, 90),
            "physical.stamina":        (80, 92),
            "physical.acceleration":   (78, 90),
            "mental.work_rate":        (82, 93),
            "mental.aggression":       (70, 82),
            "defending.tackling":      (62, 74),
            "tendencies.presses_high": 0.75,
            "tendencies.sprints_frequently": 0.70,
            "tendencies.cuts_inside":  0.35,
            # Modern winger tendencies (Checkpoint 18)
            "tendencies.hugs_touchline":      0.88,
            "tendencies.drives_byline":       0.55,
            "tendencies.attacks_fullback_1v1": 0.50,
            "tendencies.late_box_runs":       0.55,
            "tendencies.crosses_from_wide":   0.45,
        },

        # ── STRIKERS ─────────────────────────────────────────

        "poacher": {
            "description": "Lives in the box, minimal touches, finishes everything",
            "positions": ["ST", "CF"],
            "technical.finishing":     (84, 95),
            "mental.positioning":      (84, 95),
            "mental.anticipation":     (82, 94),
            "mental.composure":        (80, 92),
            "technical.heading":       (70, 82),
            "physical.pace":           (70, 82),
            "tendencies.makes_runs_behind": 0.65,
            "tendencies.shoots_from_distance": 0.08,
            "tendencies.attempts_dribble":     0.12,
        },

        "target_man": {
            "description": "Hold-up play, aerials, brings others into game",
            "positions": ["ST", "CF"],
            "physical.strength":       (82, 94),
            "physical.jumping":        (82, 94),
            "technical.heading":       (82, 94),
            "mental.bravery":          (80, 92),
            "technical.first_touch":   (74, 86),
            "passing.short_passing":   (68, 80),
            "technical.finishing":     (66, 78),
            "tendencies.attacks_the_ball": 0.78,
            "tendencies.makes_runs_behind": 0.25,
        },

        "complete_striker": {
            "description": "Does everything — goals, link-up, pressing",
            "positions": ["ST", "CF"],
            "technical.finishing":     (80, 91),
            "mental.positioning":      (78, 90),
            "physical.pace":           (76, 88),
            "physical.strength":       (72, 84),
            "technical.heading":       (72, 84),
            "mental.composure":        (76, 88),
            "passing.short_passing":   (70, 82),
            "mental.work_rate":        (74, 86),
            "tendencies.makes_runs_behind": 0.45,
            "tendencies.presses_high": 0.50,
        },

        "deep_lying_striker": {
            "description": "Drops deep, link-up play, assists as much as goals",
            "positions": ["ST", "CF"],
            "passing.through_balls":   (72, 84),
            "passing.short_passing":   (74, 86),
            "mental.vision":           (72, 84),
            "technical.first_touch":   (76, 88),
            "technical.dribbling":     (70, 82),
            "technical.finishing":     (68, 80),
            "tendencies.plays_through_ball": 0.22,
            "tendencies.attempts_dribble":   0.30,
            "tendencies.makes_runs_behind":  0.20,
        },

        "speedster_striker": {
            "description": "Pace above all, runs in behind, counter-attacking threat",
            "positions": ["ST", "CF", "LW", "RW"],
            "physical.pace":           (88, 97),
            "physical.acceleration":   (88, 97),
            "technical.finishing":     (70, 82),
            "mental.positioning":      (74, 86),
            "physical.stamina":        (74, 86),
            "tendencies.makes_runs_behind": 0.78,
            "tendencies.sprints_frequently": 0.80,
            "tendencies.shoots_from_distance": 0.10,
        },

        # ── GENERIC FALLBACK ─────────────────────────────────

        "generic": {
            "description": "Average player, no extreme strengths or weaknesses",
            "positions": [],  # Applies to any position
        },
    }

    # ── SPECIALTY → ARCHETYPE MAPPING ────────────────────────
    # Maps your existing specialty strings to DNA archetypes
    SPECIALTY_ARCHETYPE_MAP: Dict[str, str] = {
        # GK
        "sweeper_keeper":     "sweeper_keeper",
        # CB
        "ball_playing_cb":    "ball_playing_cb",
        "stopper_defender":   "stopper_cb",
        "no_nonsense_cb":     "stopper_cb",
        "sweeper_cb":         "sweeper_cb",
        "sweeper":            "sweeper_cb",
        # FB
        "aggressive_fullback": "attacking_fullback",
        "overlapping_fullback": "attacking_fullback",
        "underlapping_fullback": "inverted_fullback",
        # CDM
        "anchor_man":         "anchor",
        "ball_winner":        "ball_winning_mid",
        "interceptor":        "ball_winning_mid",
        "regista":            "regista",
        # CM
        "box_box":            "box_to_box",
        "grand_box_to_box":   "box_to_box",
        "engine":             "box_to_box",
        "playmaker":          "deep_playmaker",
        "dl_playmaker":       "deep_playmaker",
        "ball_progressor":    "progressive_midfielder",
        "press_breaker":      "progressive_midfielder",
        # CAM
        "creator":            "classic_ten",
        "grand_creator":      "classic_ten",
        "sup_vision":         "classic_ten",
        "deep_lying_forward": "shadow_striker",
        "late_runner":        "shadow_striker",
        # LW/RW
        "inverted":           "inverted_winger",
        "speedster":          "speedster_striker",
        "crosser":            "traditional_winger",
        "pressing_forward":   "pressing_winger",
        # ST
        "poacher":            "poacher",
        "fox_in_box":         "poacher",
        "target_man":         "target_man",
        "aerial_threat":      "target_man",
        "clinical_finisher":  "complete_striker",
        "cold_blooded":       "poacher",
        "dribbler":           "inverted_winger",
        "grand_dribbler":     "inverted_winger",
        # GK dist
        "distribution_gk":    "distribution_gk",
    }

    @classmethod
    def get_archetype_for_player(cls, position: str, specialties: List[str]) -> str:
        """Determine best archetype given position and specialties."""
        # Try to match specialty to archetype
        for spec in specialties:
            if spec in cls.SPECIALTY_ARCHETYPE_MAP:
                candidate = cls.SPECIALTY_ARCHETYPE_MAP[spec]
                arch_data = cls.ARCHETYPES.get(candidate, {})
                arch_positions = arch_data.get("positions", [])
                # Accept if position matches OR archetype has no position restriction
                if not arch_positions or position in arch_positions:
                    return candidate

        # Fallback by position
        position_defaults = {
            "GK":  "shot_stopper",
            "CB":  "stopper_cb",
            "LB":  "attacking_fullback",
            "RB":  "attacking_fullback",
            "CDM": "anchor",
            "CM":  "box_to_box",
            "CAM": "classic_ten",
            "LW":  "traditional_winger",
            "RW":  "traditional_winger",
            "ST":  "complete_striker",
            "CF":  "complete_striker",
        }
        return position_defaults.get(position, "generic")


# ─────────────────────────────────────────────
# DNA FACTORY — Builds PlayerDNA objects
# ─────────────────────────────────────────────

class DNAFactory:
    """
    Creates PlayerDNA objects from position + specialty list.

    Each player gets:
        1. Archetype determined from specialties
        2. Attributes set from archetype template with natural variation
        3. Tendencies shaped by archetype and specialties
        4. Age curve applied (peaks 26-29, declines after 32)
        5. Superstar boost if applicable
    """

    # Age curve: peak multiplier at each age bracket
    AGE_CURVE = {
        range(16, 20): 0.82,   # Youth: raw but unpolished
        range(20, 23): 0.90,   # Rising
        range(23, 27): 0.97,   # Approaching peak
        range(27, 30): 1.00,   # Peak years
        range(30, 33): 0.97,   # Experienced
        range(33, 36): 0.91,   # Declining
        range(36, 45): 0.82,   # Veteran
    }

    @classmethod
    def create(
        cls,
        name: str,
        position: str,
        specialties: List[str],
        age: int = 24,
        nationality: str = "Tolandian",
        preferred_foot: str = "right",
        is_superstar: bool = False,
        is_set_piece_taker: bool = False,
    ) -> PlayerDNA:
        """Build a complete PlayerDNA object."""

        archetype_key = ArchetypeLibrary.get_archetype_for_player(position, specialties)
        archetype = ArchetypeLibrary.ARCHETYPES.get(archetype_key, {})
        age_mult = cls._get_age_multiplier(age)

        # Build attribute domains
        physical  = cls._build_physical(archetype, age_mult, specialties)
        technical = cls._build_technical(archetype, age_mult, specialties, position)
        mental    = cls._build_mental(archetype, age_mult, specialties)
        passing   = cls._build_passing(archetype, age_mult, specialties)
        defending = cls._build_defending(archetype, age_mult, specialties, position)
        gk_attrs  = cls._build_gk(archetype, age_mult) if position == "GK" else GoalkeeperAttributes()
        tendencies = cls._build_tendencies(archetype, specialties, position)

        # Footedness
        if "two_footed" in specialties:
            footedness = "both"
        elif random.random() < 0.20:
            footedness = "left" if preferred_foot == "right" else "right"
        else:
            footedness = preferred_foot

        dna = PlayerDNA(
            name=name,
            position=position,
            age=age,
            nationality=nationality,
            is_superstar=is_superstar,
            physical=physical,
            technical=technical,
            mental=mental,
            passing=passing,
            defending=defending,
            gk_attrs=gk_attrs,
            tendencies=tendencies,
            specialties=specialties,
            preferred_foot=preferred_foot,
            footedness=footedness,
            is_set_piece_taker=is_set_piece_taker,
            archetype=archetype_key,
        )

        # Superstar boost
        if is_superstar:
            cls._apply_superstar_boost(dna)

        return dna

    # ── LEGACY ARCHETYPE BUILDER ────────────────────────────────
    # Pre-Checkpoint builder that took an archetype key directly. Rebuilt on
    # top of create() so old fixtures/tools (test_preservation_properties,
    # test_bug_exploration, test_restart_implementation, ...) keep working
    # after the archetype-by-specialty refactor.
    LEGACY_ARCHETYPE_ALIASES = {
        "target_forward":   "target_man",
        "complete_forward": "complete_striker",
        "inside_forward":   "inverted_winger",
    }

    @classmethod
    def create_archetype(
        cls,
        archetype_key: str,
        age: int = 26,
        name: str = None,
    ) -> PlayerDNA:
        """
        Build a PlayerDNA straight from an archetype key (or a specialty /
        legacy archetype name). Resolves the position from the archetype
        template and rebuilds through the standard create() path so every
        attribute pipeline lives in one place.
        """
        key = ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP.get(archetype_key)
        if key is None:
            key = cls.LEGACY_ARCHETYPE_ALIASES.get(archetype_key)
        key = key or archetype_key
        arch = ArchetypeLibrary.ARCHETYPES.get(key, {})
        if not arch:
            raise ValueError(f"Unknown archetype or specialty: {archetype_key}")
        position = (arch.get("positions") or ["ST"])[0]
        dna_name = name or " ".join(w.capitalize() for w in archetype_key.split("_"))
        return cls.create(
            name=dna_name,
            position=position,
            specialties=[key],
            age=age,
        )

    # ── ATTRIBUTE BUILDERS ────────────────────────────────────

    @classmethod
    def _attr(cls, archetype: Dict, key: str, default_range: Tuple, age_mult: float) -> float:
        """Get attribute value from archetype template with variation."""
        if key in archetype:
            lo, hi = archetype[key]
        else:
            lo, hi = default_range
        base = random.uniform(lo, hi)
        return round(min(99.0, max(30.0, base * age_mult)), 1)

    @classmethod
    def _build_physical(cls, arch: Dict, age_mult: float, specs: List[str]) -> PhysicalAttributes:
        p = PhysicalAttributes(
            pace         = cls._attr(arch, "physical.pace",         (55, 72), age_mult),
            acceleration = cls._attr(arch, "physical.acceleration",  (55, 72), age_mult),
            stamina      = cls._attr(arch, "physical.stamina",       (60, 76), age_mult),
            strength     = cls._attr(arch, "physical.strength",      (55, 72), age_mult),
            jumping      = cls._attr(arch, "physical.jumping",       (55, 72), age_mult),
            agility      = cls._attr(arch, "physical.agility",       (55, 72), age_mult),
        )
        # Specialty overrides
        if "speedster" in specs:
            p.pace         = min(99, p.pace * 1.22)
            p.acceleration = min(99, p.acceleration * 1.22)
        if "strong" in specs:
            p.strength = min(99, p.strength * 1.20)
            p.jumping  = min(99, p.jumping * 1.10)
        if "ironman" in specs:
            p.stamina = min(99, p.stamina * 1.15)
        if "engine" in specs:
            p.stamina = min(99, p.stamina * 1.12)
        return p

    @classmethod
    def _build_technical(cls, arch: Dict, age_mult: float, specs: List[str], pos: str) -> TechnicalAttributes:
        t = TechnicalAttributes(
            dribbling     = cls._attr(arch, "technical.dribbling",     (45, 68), age_mult),
            first_touch   = cls._attr(arch, "technical.first_touch",   (55, 72), age_mult),
            ball_control  = cls._attr(arch, "technical.ball_control",  (55, 72), age_mult),
            crossing      = cls._attr(arch, "technical.crossing",      (40, 62), age_mult),
            finishing     = cls._attr(arch, "technical.finishing",     (35, 60), age_mult),
            long_shots    = cls._attr(arch, "technical.long_shots",    (35, 58), age_mult),
            heading       = cls._attr(arch, "technical.heading",       (42, 65), age_mult),
            free_kick     = cls._attr(arch, "technical.free_kick",     (38, 60), age_mult),
            penalty_taking = cls._attr(arch, "technical.penalty_taking",(52, 70), age_mult),
            weak_foot     = cls._attr(arch, "technical.weak_foot",     (35, 60), age_mult),
        )
        if "clinical_finisher" in specs or "cold_blooded" in specs:
            t.finishing     = min(99, t.finishing * 1.22)
            t.penalty_taking = min(99, t.penalty_taking * 1.15)
        if "dribbler" in specs:
            t.dribbling = min(99, t.dribbling * 1.22)
            t.ball_control = min(99, t.ball_control * 1.10)
        if "grand_dribbler" in specs:
            t.dribbling = min(99, t.dribbling * 1.35)
        if "crosser" in specs:
            t.crossing = min(99, t.crossing * 1.20)
        if "set_piece_specialist" in specs:
            t.free_kick = min(99, t.free_kick * 1.25)
        if "two_footed" in specs:
            t.weak_foot = min(99, t.weak_foot * 1.40)
        if "long_shot_taker" in specs:
            t.long_shots = min(99, t.long_shots * 1.22)
        return t

    @classmethod
    def _build_mental(cls, arch: Dict, age_mult: float, specs: List[str]) -> MentalAttributes:
        # Mental attributes are LESS affected by age (experience compensates)
        mental_age = max(0.92, age_mult)
        m = MentalAttributes(
            vision        = cls._attr(arch, "mental.vision",        (50, 70), mental_age),
            composure     = cls._attr(arch, "mental.composure",     (50, 70), mental_age),
            decisions     = cls._attr(arch, "mental.decisions",     (50, 70), mental_age),
            positioning   = cls._attr(arch, "mental.positioning",   (50, 70), mental_age),
            anticipation  = cls._attr(arch, "mental.anticipation",  (50, 70), mental_age),
            work_rate     = cls._attr(arch, "mental.work_rate",     (55, 74), mental_age),
            aggression    = cls._attr(arch, "mental.aggression",    (45, 68), mental_age),
            leadership    = cls._attr(arch, "mental.leadership",    (38, 62), mental_age),
            concentration = cls._attr(arch, "mental.concentration", (50, 70), mental_age),
            bravery       = cls._attr(arch, "mental.bravery",       (50, 70), mental_age),
        )
        if "captain" in specs or "leadership" in specs:
            m.leadership  = min(99, m.leadership * 1.25)
            m.composure   = min(99, m.composure * 1.10)
        if "big_game_player" in specs:
            m.composure   = min(99, m.composure * 1.15)
        if "clutch" in specs:
            m.composure   = min(99, m.composure * 1.12)
            m.decisions   = min(99, m.decisions * 1.10)
        if "press_resistant" in specs:
            m.composure   = min(99, m.composure * 1.12)
            m.decisions   = min(99, m.decisions * 1.08)
        return m

    @classmethod
    def _build_passing(cls, arch: Dict, age_mult: float, specs: List[str]) -> PassingAttributes:
        mental_age = max(0.92, age_mult)  # Passing improves with experience
        p = PassingAttributes(
            short_passing = cls._attr(arch, "passing.short_passing", (55, 72), mental_age),
            long_passing  = cls._attr(arch, "passing.long_passing",  (45, 65), mental_age),
            through_balls = cls._attr(arch, "passing.through_balls", (40, 60), mental_age),
            switch_play   = cls._attr(arch, "passing.switch_play",   (40, 60), mental_age),
        )
        if "passer" in specs:
            p.short_passing = min(99, p.short_passing * 1.18)
            p.long_passing  = min(99, p.long_passing * 1.15)
        if "sup_vision" in specs or "regista" in specs:
            p.through_balls = min(99, p.through_balls * 1.25)
            p.switch_play   = min(99, p.switch_play * 1.20)
        if "creator" in specs or "grand_creator" in specs:
            p.through_balls = min(99, p.through_balls * 1.20)
        return p

    @classmethod
    def _build_defending(cls, arch: Dict, age_mult: float, specs: List[str], pos: str) -> DefendingAttributes:
        d = DefendingAttributes(
            tackling      = cls._attr(arch, "defending.tackling",      (35, 60), age_mult),
            marking       = cls._attr(arch, "defending.marking",       (35, 60), age_mult),
            interceptions = cls._attr(arch, "defending.interceptions", (35, 60), age_mult),
            blocking      = cls._attr(arch, "defending.blocking",      (35, 60), age_mult),
            clearances    = cls._attr(arch, "defending.clearances",    (35, 60), age_mult),
        )
        if "tackler" in specs:
            d.tackling    = min(99, d.tackling * 1.22)
        if "interceptor" in specs:
            d.interceptions = min(99, d.interceptions * 1.22)
        if "blocker" in specs:
            d.blocking    = min(99, d.blocking * 1.22)
        if "ball_winner" in specs:
            d.tackling    = min(99, d.tackling * 1.18)
            d.interceptions = min(99, d.interceptions * 1.15)
        return d

    @classmethod
    def _build_gk(cls, arch: Dict, age_mult: float) -> GoalkeeperAttributes:
        mental_age = max(0.92, age_mult)
        return GoalkeeperAttributes(
            diving         = cls._attr(arch, "gk_attrs.diving",         (60, 78), age_mult),
            handling       = cls._attr(arch, "gk_attrs.handling",       (60, 78), age_mult),
            kicking        = cls._attr(arch, "gk_attrs.kicking",        (55, 74), mental_age),
            reflexes       = cls._attr(arch, "gk_attrs.reflexes",       (60, 78), age_mult),
            positioning_gk = cls._attr(arch, "gk_attrs.positioning_gk", (60, 78), mental_age),
            communication  = cls._attr(arch, "gk_attrs.communication",  (55, 74), mental_age),
            sweeping       = cls._attr(arch, "gk_attrs.sweeping",       (52, 72), mental_age),
            aerial_gk      = cls._attr(arch, "gk_attrs.aerial_gk",     (58, 76), age_mult),
        )

    @classmethod
    def _build_tendencies(cls, arch: Dict, specs: List[str], pos: str) -> BehavioralTendencies:
        """Build behavioral tendencies from archetype + specialty."""
        t = BehavioralTendencies()

        # Apply archetype tendency overrides
        for key, val in arch.items():
            if key.startswith("tendencies.") and isinstance(val, float):
                attr = key.split(".")[1]
                if hasattr(t, attr):
                    setattr(t, attr, val)

        # Specialty tendency overrides
        if "dirty_player" in specs:
            t.commits_fouls = min(0.45, t.commits_fouls * 1.5)
            t.argues_with_ref = min(0.30, t.argues_with_ref * 1.4)
        if "foul_drawer" in specs:
            t.dives = min(0.20, t.dives * 1.5)
        if "engine" in specs or "workhorse" in specs:
            t.sprints_frequently = min(0.90, t.sprints_frequently * 1.25)
            t.presses_high       = min(0.80, t.presses_high * 1.20)
        if "press_leader" in specs:
            t.presses_high = min(0.85, t.presses_high * 1.35)
        if "captain" in specs:
            t.argues_with_ref = max(0.05, t.argues_with_ref * 0.60)

        # ── MODERN WINGER SPECIALTY OVERRIDES (Checkpoint 18) ──
        # Wingers are touchline-hugging flank attackers, not drifting #10s.
        # The middle of the pitch is always full — a #10 owns that space —
        # and a winger who drifts inside leaves his flank open.
        if pos in ("LW", "RW"):
            # All wingers default to hugging the touchline
            t.hugs_touchline = max(t.hugs_touchline, 0.75)
            # Wingers attack the fullback 1v1 by default
            t.attacks_fullback_1v1 = max(t.attacks_fullback_1v1, 0.55)

        if "crosser" in specs:
            t.drives_byline = min(0.95, t.drives_byline * 1.30)
            t.crosses_from_wide = min(0.95, t.crosses_from_wide * 1.30)
            t.hugs_touchline = min(0.98, t.hugs_touchline * 1.10)
        if "inverted" in specs:
            t.drives_byline = max(0.10, t.drives_byline * 0.50)
            t.crosses_from_wide = max(0.10, t.crosses_from_wide * 0.50)
            t.attacks_fullback_1v1 = min(0.95, t.attacks_fullback_1v1 * 1.20)
        if "speedster" in specs:
            t.drives_byline = min(0.95, t.drives_byline * 1.20)
            t.late_box_runs = min(0.95, t.late_box_runs * 1.25)
            t.hugs_touchline = min(0.98, t.hugs_touchline * 1.10)
        if "grand_dribbler" in specs or "dribbler" in specs:
            t.attacks_fullback_1v1 = min(0.98, t.attacks_fullback_1v1 * 1.25)
        if "pressing_forward" in specs:
            t.hugs_touchline = min(0.98, t.hugs_touchline * 1.10)

        return t

    @classmethod
    def _apply_superstar_boost(cls, dna: PlayerDNA):
        """Superstars are 10-20% better across the board."""
        boost = random.uniform(1.10, 1.20)
        for domain in [dna.physical, dna.technical, dna.mental, dna.passing, dna.defending]:
            for attr in vars(domain):
                val = getattr(domain, attr)
                if isinstance(val, float):
                    setattr(domain, attr, round(min(99.0, val * boost), 1))
        if dna.position == "GK":
            for attr in vars(dna.gk_attrs):
                val = getattr(dna.gk_attrs, attr)
                if isinstance(val, float):
                    setattr(dna.gk_attrs, attr, round(min(99.0, val * boost), 1))
        dna.mental.composure = min(99, dna.mental.composure * 1.08)  # Extra composure

    @classmethod
    def _get_age_multiplier(cls, age: int) -> float:
        for age_range, mult in cls.AGE_CURVE.items():
            if age in age_range:
                return mult
        return 0.85  # Very young or very old

    # ── SHOOTER QUALITY ───────────────────────────────────────

    @classmethod
    def get_shooter_quality(cls, dna: PlayerDNA) -> float:
        """
        Convert finishing attribute to xG multiplier.
        Used by the MatchEngine when rolling for goals. Now scales with
        live stamina so a drained striker converts worse than a fresh one.
        """
        base = dna.effective_finishing / 100.0
        # Scale to 0.70–1.40 range
        return 0.70 + (base * 0.70)

    @classmethod
    def get_pass_accuracy(cls, dna: PlayerDNA, is_long: bool = False, under_pressure: bool = False) -> float:
        """
        Convert passing attributes to completion probability.
        Now scales with live stamina via effective_passing.
        """
        if is_long:
            base = dna.passing.long_passing / 100.0
        else:
            base = dna.passing.short_passing / 100.0

        # Apply form, fatigue AND live stamina
        base *= dna.form.form_multiplier * dna.form.fatigue_multiplier * dna.live_performance_mult

        # Pressure reduces accuracy
        if under_pressure:
            composure_factor = dna.mental.composure / 100.0
            pressure_penalty = (1 - composure_factor) * 0.15
            base -= pressure_penalty

        # Scale: good passer 0.88–0.95, poor passer 0.65–0.78
        scaled = 0.55 + (base * 0.45)
        return round(min(0.98, max(0.50, scaled)), 3)

    @classmethod
    def get_dribble_success_rate(cls, dna: PlayerDNA) -> float:
        """Convert dribbling attribute to success probability.
        Now scales with live stamina via effective_dribbling."""
        base = dna.effective_dribbling / 100.0
        # Scale: elite dribbler 0.65–0.78, average 0.45–0.58
        return round(0.35 + (base * 0.45), 3)

    @classmethod
    def get_tackle_success_rate(cls, dna: PlayerDNA) -> float:
        """Convert tackling attribute to success probability.
        Scales with live stamina — tired defenders mistime tackles."""
        base = (dna.defending.tackling / 100.0) * dna.live_performance_mult
        return round(0.40 + (base * 0.45), 3)

    @classmethod
    def get_aerial_success_rate(cls, dna: PlayerDNA) -> float:
        """Convert aerial attributes to success probability."""
        base = ((dna.physical.jumping + dna.technical.heading + dna.mental.bravery) / 3.0) / 100.0
        return round(0.35 + (base * 0.45), 3)

    @classmethod
    def get_space_battle_success(cls, dna: PlayerDNA, defender_dna: Optional[PlayerDNA] = None) -> float:
        """
        Checkpoint 19 — space battle: when a marked player attempts to beat
        their marker to get to space (receiving a pass, making a run), the
        outcome depends on pace, vision, strength, and stamina vs the
        defender's tackling and marking.

        This models the real-life mechanic: "beating a defender to get to
        space requires pace and vision strength stamina."
        """
        if defender_dna is None:
            defender_dna = PlayerDNA(name="defender", position="CB")

        # Attacker's physical + mental package
        pace = dna.physical.pace / 100.0
        accel = dna.physical.acceleration / 100.0
        strength = dna.physical.strength / 100.0
        stamina = dna.live_performance_mult
        vision = dna.mental.vision / 100.0
        composure = dna.mental.composure / 100.0

        # Defender's defensive package
        tackle = defender_dna.defending.tackling / 100.0
        marking = defender_dna.defending.marking / 100.0
        def_stamina = defender_dna.live_performance_mult

        # Attacker advantage: pace + acceleration burst + strength shield + composure
        att_score = (pace * 0.30 + accel * 0.25 + strength * 0.20 +
                     composure * 0.15 + vision * 0.10) * stamina

        # Defender advantage: tackle + marking + positioning
        def_score = (tackle * 0.40 + marking * 0.35 +
                     (defender_dna.mental.positioning / 100.0) * 0.25) * def_stamina

        # Space battle: attacker wins if their advantage exceeds defender's
        # by enough margin. A truly elite dribbler (pace 85+, strength 75+)
        # beats an average CB ~60-70% of the time.
        raw = att_score - def_score + 0.15  # small base advantage for attacker initiative
        prob = 1.0 / (1.0 + math.exp(-raw * 8.0))  # sigmoid to 0-1

        return round(max(0.15, min(0.85, prob)), 3)


# ─────────────────────────────────────────────
# PLAYER PROFILE — Full match-ready object
# ─────────────────────────────────────────────

@dataclass
class PlayerProfile:
    """
    The complete player object used throughout the simulation.
    Wraps DNA with match-specific state.
    """
    dna: PlayerDNA
    team_name: str

    # Match state (populated during simulation)
    is_starter: bool          = True
    sub_in_minute: Optional[int]  = None
    sub_out_minute: Optional[int] = None

    @property
    def name(self) -> str:
        return self.dna.name

    @property
    def position(self) -> str:
        return self.dna.position

    @property
    def specialties(self) -> List[str]:
        return self.dna.specialties

    @property
    def minutes_played(self) -> int:
        return self.dna.minutes_played

    @minutes_played.setter
    def minutes_played(self, val: int):
        self.dna.minutes_played = val

    def __repr__(self) -> str:
        return (
            f"PlayerProfile({self.name}, {self.position}, "
            f"{self.team_name}, OVR={self.dna.overall_rating:.0f})"
        )


# ─────────────────────────────────────────────
# SQUAD BUILDER — Convenience class
# ─────────────────────────────────────────────

class SquadBuilder:
    """
    Builds full squads from simple tuples.
    Same interface as the old system, now produces PlayerProfile objects with DNA.

    Usage:
        starters = [
            ("Dragan Novak", "ST", ["clinical_finisher", "aerial_threat"], 28),
            ("Adri Vela",    "LW", ["dribbler", "speedster"],              23),
        ]
        squad = SquadBuilder.build("Hartwell City", starters, subs)
    """

    @staticmethod
    def build(
        team_name: str,
        starters: List[Tuple],
        substitutes: List[Tuple] = None,
        team_superstars: List[str] = None,
        set_piece_takers: List[str] = None,
    ) -> Dict[str, List[PlayerProfile]]:
        """
        Build a squad dict with 'starters' and 'substitutes'.

        Tuple format: (name, position, specialties, [age], [nationality], [preferred_foot])
        """
        superstars = team_superstars or []
        sp_takers  = set_piece_takers or []
        result = {'starters': [], 'substitutes': []}

        for entry in (starters or []):
            profile = SquadBuilder._build_player(
                entry, team_name, is_starter=True,
                superstars=superstars, sp_takers=sp_takers
            )
            result['starters'].append(profile)

        if len(result['starters']) != 11:
            raise ValueError(
                f"Team '{team_name}' must have exactly 11 starters, "
                f"got {len(result['starters'])}."
            )

        gk_count = sum(1 for p in result['starters'] if p.position == "GK")
        if gk_count != 1:
            raise ValueError(
                f"Team '{team_name}' starting XI must contain exactly 1 GK, got {gk_count}. "
                f"Starters: {[(p.name, p.position) for p in result['starters']]}"
            )

        for entry in (substitutes or []):
            # Sub tuples can have a 5th element: sub_in_minute
            sub_in = None
            if len(entry) > 4 and isinstance(entry[-1], int) and entry[-1] > 20:
                sub_in = entry[-1]
                entry  = entry[:-1]

            profile = SquadBuilder._build_player(
                entry, team_name, is_starter=False,
                superstars=superstars, sp_takers=sp_takers
            )
            profile.sub_in_minute = sub_in
            result['substitutes'].append(profile)

        return result

    @staticmethod
    def _build_player(
        entry: Tuple,
        team_name: str,
        is_starter: bool,
        superstars: List[str],
        sp_takers: List[str],
    ) -> PlayerProfile:
        """Parse a player tuple and build PlayerProfile."""
        name       = entry[0]
        position   = entry[1]
        specialties = list(entry[2]) if len(entry) > 2 else []
        age        = entry[3] if len(entry) > 3 else random.randint(19, 33)
        nationality = entry[4] if len(entry) > 4 else "Tolandian"
        pref_foot  = entry[5] if len(entry) > 5 else (
            "left" if random.random() < 0.15 else "right"
        )

        dna = DNAFactory.create(
            name=name,
            position=position,
            specialties=specialties,
            age=age,
            nationality=nationality,
            preferred_foot=pref_foot,
            is_superstar=(name in superstars),
            is_set_piece_taker=(name in sp_takers),
        )

        profile = PlayerProfile(dna=dna, team_name=team_name, is_starter=is_starter)
        return profile


# ─────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧬 PLOFA 26/27 — Player DNA Module Demo")
    print("="*55)

    # Build Hartwell City squad
    hartwell_starters = [
        ("Keano Walsh",   "GK",  ["sweeper_keeper"],                    29),
        ("Darius Frost",  "LB",  ["aggressive_fullback", "engine"],      24),
        ("Emeka Obi",     "CB",  ["ball_playing_cb"],                    27),
        ("Tavish Crane",  "CB",  ["stopper_defender", "strong"],         30),
        ("Rico Alves",    "RB",  ["overlapping_fullback"],               25),
        ("Mateo Sanz",    "CDM", ["anchor_man", "interceptor"],          28),
        ("Luca Ferrini",  "CM",  ["box_box", "engine"],                  26),
        ("Kofi Mensah",   "CAM", ["creator", "sup_vision"],              24),
        ("Adri Vela",     "LW",  ["dribbler", "speedster"],              22),
        ("Dragan Novak",  "ST",  ["clinical_finisher", "aerial_threat"], 29),
        ("Yusuf Hamid",   "RW",  ["grand_dribbler", "inverted"],         23),
    ]

    squad = SquadBuilder.build(
        team_name="Hartwell City",
        starters=hartwell_starters,
        team_superstars=["Dragan Novak", "Yusuf Hamid"],
        set_piece_takers=["Kofi Mensah", "Adri Vela"],
    )

    print(f"\n{'Player':<20} {'Pos':<5} {'Arch':<22} {'OVR':>5}  {'Pace':>5} {'Fin':>5} {'Vis':>5} {'Tck':>5}")
    print("-"*75)
    for p in squad['starters']:
        d = p.dna
        print(
            f"{p.name:<20} {p.position:<5} {d.archetype:<22} "
            f"{d.overall_rating:>5.1f}  "
            f"{d.physical.pace:>5.1f} "
            f"{d.technical.finishing:>5.1f} "
            f"{d.mental.vision:>5.1f} "
            f"{d.defending.tackling:>5.1f}"
        )

    print("\n🔬 Deep dive — Dragan Novak (Superstar ST):")
    novak = next(p for p in squad['starters'] if p.name == "Dragan Novak")
    d = novak.dna
    print(f"   Archetype:        {d.archetype}")
    print(f"   Overall:          {d.overall_rating:.1f}")
    print(f"   Finishing:        {d.technical.finishing:.1f} (effective: {d.effective_finishing:.1f})")
    print(f"   Shooter quality:  {DNAFactory.get_shooter_quality(d):.3f}×")
    print(f"   Composure:        {d.mental.composure:.1f}")
    print(f"   Tendency→dribble: {d.tendencies.attempts_dribble:.0%}")
    print(f"   Tendency→press:   {d.tendencies.presses_high:.0%}")
    print(f"   Superstar:        {d.is_superstar}")

    print("\n🔬 Deep dive — Kofi Mensah (Creator CAM):")
    kofi = next(p for p in squad['starters'] if p.name == "Kofi Mensah")
    d = kofi.dna
    print(f"   Archetype:        {d.archetype}")
    print(f"   Overall:          {d.overall_rating:.1f}")
    print(f"   Vision:           {d.mental.vision:.1f}")
    print(f"   Through balls:    {d.passing.through_balls:.1f}")
    print(f"   Pass accuracy:    {DNAFactory.get_pass_accuracy(d):.1%}")
    print(f"   Long pass acc:    {DNAFactory.get_pass_accuracy(d, is_long=True):.1%}")
    print(f"   Set piece taker:  {d.is_set_piece_taker}")

    print("\n✅ Player DNA module operational.")
    print("   Next: Stat Accumulator — events → StatsBomb-level player stats\n")

#where to add soul: Optional[PlayerSoul] = None? 


#you add it 
#where to add if soul_assignments and name in soul_assignments:
       #dna.soul = soul_assignments[name]?
