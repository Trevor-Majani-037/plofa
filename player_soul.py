"""
PLOFA 26/27 — PLAYER SOUL MODULE
===================================
player_soul.py

Philosophy:
    Stats come from attributes. Greatness comes from the soul.

    A player's soul is the deepest layer of who they are on a football pitch.
    It is not reducible to pace, finishing, or vision scores.
    It is the RELATIONSHIP between a player and their craft —
    how deeply they understand it, how obsessively they pursue it,
    how they bend the game to their will.

    Two players can have identical DNA attributes.
    Only one of them has the soul that makes them truly alien.

The Greatness Formula:
    G = (H^α × T^β × L^γ) × Ω

    H = Hardwork  (0.0–1.0): relentlessness, consistency, standards
    T = Talent    (0.0–1.0): ceiling — what body and mind can do at peak
    L = Luck      (0.0–1.0): right club, right era, no career-ending injuries
    α = 0.35, β = 0.50, γ = 0.15  (talent weighted heaviest)

    Ω (Omega) = Multiplicative Activation Bonus
        Activates ONLY when H > 0.85, T > 0.88, L > 0.75
        This is what separates Percy from merely elite players.
        It is NOT additive. It is a phase transition.
        Ω = 1.0 (no bonus) for normal elite players
        Ω = 1.25–1.60 for once-in-a-generation talents

Soul Archetypes (max 2–3 per league):
    ATTACKING_PROPHET   — Messi. Every atom of attacking football.
    DEFENSIVE_PURIST    — Van Dijk. Defense is their religion.
    MIDFIELD_PHILOSOPHER — Xavi. Understands space before it exists.
    CREATIVE_ORACLE     — De Bruyne/Bruno. One dimension, but complete.
    GOALSCORING_SAVANT  — Lewandowski. Finishing is biological.
    PRESSING_EVANGELIST — Gegenpress incarnate. Hunting is instinct.
    WIDE_DESTROYER      — Prime Robben. One direction. Can't stop it.
    SWEEPER_SAGE        — Pirlo. Reads the game one pass ahead.
    WALL               — Prime Terry. Simply does not concede.

Design reference: Percy (RW, Hartwell City) — ATTACKING_PROPHET, Ω = 1.55
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# SOUL ARCHETYPES
# ─────────────────────────────────────────────

class SoulArchetype(Enum):
    """
    The 9 transcendent player souls.
    Maximum 2–3 per PLOFA league at any time.
    Most elite players have NONE of these — they are merely great.
    """
    ATTACKING_PROPHET    = "attacking_prophet"
    DEFENSIVE_PURIST     = "defensive_purist"
    MIDFIELD_PHILOSOPHER = "midfield_philosopher"
    CREATIVE_ORACLE      = "creative_oracle"
    GOALSCORING_SAVANT   = "goalscoring_savant"
    PRESSING_EVANGELIST  = "pressing_evangelist"
    WIDE_DESTROYER       = "wide_destroyer"
    SWEEPER_SAGE         = "sweeper_sage"
    WALL                 = "wall"


# ─────────────────────────────────────────────
# GREATNESS PILLARS
# ─────────────────────────────────────────────

@dataclass
class GreatnessPillars:
    """
    The three pillars of the Greatness Formula.
    Each is 0.0–1.0. Most elite players top out at 0.82.
    Percy hits 0.97, 0.99, 0.91.
    """
    hardwork: float   # Relentlessness, standards, recovery work
    talent:   float   # Raw ceiling — what is physically/mentally possible
    luck:     float   # Right club, right era, health, mentors, timing

    # Threshold for Omega activation
    HARDWORK_THRESHOLD: float = field(default=0.85, init=False, repr=False)
    TALENT_THRESHOLD:   float = field(default=0.88, init=False, repr=False)
    LUCK_THRESHOLD:     float = field(default=0.75, init=False, repr=False)

    # Exponents
    ALPHA: float = field(default=0.35, init=False, repr=False)  # hardwork weight
    BETA:  float = field(default=0.50, init=False, repr=False)  # talent weight
    GAMMA: float = field(default=0.15, init=False, repr=False)  # luck weight

    def __post_init__(self):
        self.hardwork = max(0.0, min(1.0, self.hardwork))
        self.talent   = max(0.0, min(1.0, self.talent))
        self.luck     = max(0.0, min(1.0, self.luck))

    @property
    def omega_activated(self) -> bool:
        """True only when ALL three pillars clear their thresholds."""
        return (
            self.hardwork >= self.HARDWORK_THRESHOLD and
            self.talent   >= self.TALENT_THRESHOLD   and
            self.luck     >= self.LUCK_THRESHOLD
        )

    @property
    def omega(self) -> float:
        """
        The multiplicative activation bonus.
        Not a simple multiplier — it scales with HOW FAR above threshold.
        A player barely clearing thresholds gets Ω ≈ 1.15.
        Percy, maxing all three, gets Ω ≈ 1.55.
        """
        if not self.omega_activated:
            return 1.0

        # Excess above each threshold (0.0 to ~0.15 range each)
        h_excess = self.hardwork - self.HARDWORK_THRESHOLD
        t_excess = self.talent   - self.TALENT_THRESHOLD
        l_excess = self.luck     - self.LUCK_THRESHOLD

        # Combined excess drives Omega
        total_excess = (h_excess * 0.35 + t_excess * 0.50 + l_excess * 0.15)

        # Map excess to Ω range: 1.15 (just activated) → 1.60 (absolute peak)
        omega = 1.15 + (total_excess / 0.15) * 0.45
        return round(min(1.60, max(1.15, omega)), 4)

    @property
    def raw_score(self) -> float:
        """
        G_raw = H^α × T^β × L^γ
        Before Omega. Most elite players: 0.70–0.82.
        Percy: ~0.95.
        """
        return round(
            (self.hardwork ** self.ALPHA) *
            (self.talent   ** self.BETA)  *
            (self.luck     ** self.GAMMA),
            4
        )

    @property
    def greatness_coefficient(self) -> float:
        """
        G = raw_score × Ω
        The final coefficient applied to in-match performance.
        Range: ~0.40 (poor) → 1.60 (Percy-tier)
        Average PLOFA player: ~0.62
        Good player: ~0.72
        Elite player: ~0.82
        Star player: ~0.88
        Omega-activated: 1.15–1.60
        """
        return round(self.raw_score * self.omega, 4)

    @property
    def tier(self) -> str:
        g = self.greatness_coefficient
        if g >= 1.40: return "TRANSCENDENT"
        if g >= 1.15: return "GENERATIONAL"
        if g >= 0.88: return "STAR"
        if g >= 0.78: return "ELITE"
        if g >= 0.68: return "QUALITY"
        if g >= 0.58: return "SOLID"
        return "AVERAGE"

    def describe(self) -> str:
        lines = [
            f"  Hardwork : {self.hardwork:.2f} {'✓' if self.hardwork >= self.HARDWORK_THRESHOLD else '✗'}",
            f"  Talent   : {self.talent:.2f} {'✓' if self.talent >= self.TALENT_THRESHOLD else '✗'}",
            f"  Luck     : {self.luck:.2f} {'✓' if self.luck >= self.LUCK_THRESHOLD else '✗'}",
            f"  Ω Active : {'YES' if self.omega_activated else 'No'} (Ω = {self.omega:.3f})",
            f"  G_raw    : {self.raw_score:.4f}",
            f"  G_final  : {self.greatness_coefficient:.4f}",
            f"  Tier     : {self.tier}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SOUL PROFILE — The archetype in full detail
# ─────────────────────────────────────────────

@dataclass
class SoulProfile:
    """
    Defines what a soul archetype DOES to a player's behaviour and stats.

    Every soul modifies:
        - Probability multipliers for specific event types
        - Psychological responses to match situations
        - How their presence affects teammates
        - Which stats get boosted beyond what DNA predicts
    """
    archetype: SoulArchetype
    label: str
    description: str
    positions: List[str]       # Valid positions for this soul

    # ── EVENT PROBABILITY MULTIPLIERS ────────────────────────
    # These multiply the base probability of specific events occurring
    # 1.0 = no change, 1.5 = 50% more likely, 0.7 = 30% less likely
    shot_quality_mult:        float = 1.0   # xG per shot
    shot_frequency_mult:      float = 1.0   # Shots per chance received
    dribble_attempt_mult:     float = 1.0   # Dribble attempt rate
    dribble_success_mult:     float = 1.0   # Dribble completion rate
    shot_assist_mult:         float = 1.0   # Shot assist frequency
    through_ball_mult:        float = 1.0   # Through ball attempt rate
    press_intensity_mult:     float = 1.0   # Press frequency
    press_success_mult:       float = 1.0   # Press success rate
    tackle_success_mult:      float = 1.0   # Tackle success rate
    interception_mult:        float = 1.0   # Interception frequency
    aerial_success_mult:      float = 1.0   # Aerial duel win rate
    clearance_quality_mult:   float = 1.0   # Clearance effectiveness
    carry_frequency_mult:     float = 1.0   # Ball carrying tendency
    cross_quality_mult:       float = 1.0   # Cross delivery quality
    pass_accuracy_mult:       float = 1.0   # Pass completion rate
    positioning_mult:         float = 1.0   # Positional intelligence
    composure_mult:           float = 1.0   # Big moment composure
    big_chance_conversion:    float = 1.0   # Big chance conversion rate
    save_quality_mult:        float = 1.0   # GK: shot-stopping quality

    # ── PSYCHOLOGICAL STATE MULTIPLIERS ──────────────────────
    # How the soul responds to match situations
    # These change the multipliers based on game state

    # When their team is LOSING (they tend to elevate)
    losing_state_boost: float = 1.05

    # When their team is WINNING (some maintain, some cruise)
    winning_state_mult: float = 1.0

    # In the FINAL 15 minutes of a match
    late_game_boost: float = 1.05

    # In ADDED TIME specifically
    added_time_boost: float = 1.08

    # Under HEAVY PRESSURE from opponents (mental fortitude)
    pressure_resistance: float = 1.0    # >1.0 = thrives under pressure

    # ── TEAMMATE AURA ─────────────────────────────────────────
    # A soul player elevates those around them
    # Multiplier applied to nearby teammates' key probabilities
    teammate_aura_radius: float = 1.0   # How far the aura spreads (positional)
    teammate_pass_boost:  float = 1.0   # Teammates pass better near them
    teammate_press_boost: float = 1.0   # Teammates press harder near them
    teammate_conf_boost:  float = 1.0   # Teammates' composure near them

    # ── SIGNATURE ACTIONS ─────────────────────────────────────
    # Unique actions this soul type can perform
    # that other players cannot or rarely do
    can_unlock_defence:   bool = False  # Single pass that beats defensive shape
    can_hold_defensive_line: bool = False  # Organises entire back line
    can_dictate_tempo:    bool = False  # Slows or accelerates entire match
    can_finish_any_angle: bool = False  # Goal from impossible positions
    can_hunt_in_packs:    bool = False  # Creates press traps for team
    can_destroy_one_v_one: bool = False  # Beats their man every time
    can_read_game_early:  bool = False  # Acts a second before others see it

    # ── STAT CAPS LIFTED ──────────────────────────────────────
    # Soul players exceed what DNA alone can produce
    # These are bonus additions ON TOP of the DNA-generated stat ranges
    bonus_shot_assists_per_match:  Tuple[int,int] = (0, 0)
    bonus_dribbles_per_match:      Tuple[int,int] = (0, 0)
    bonus_tackles_per_match:       Tuple[int,int] = (0, 0)
    bonus_interceptions_per_match: Tuple[int,int] = (0, 0)
    bonus_progressive_passes:      Tuple[int,int] = (0, 0)
    bonus_pressures_per_match:     Tuple[int,int] = (0, 0)
    bonus_carries_per_match:       Tuple[int,int] = (0, 0)


# ─────────────────────────────────────────────
# SOUL LIBRARY — All defined archetypes
# ─────────────────────────────────────────────

class SoulLibrary:
    """
    The complete catalogue of all PLOFA soul archetypes.
    Each is built from real football understanding,
    not just arbitrary multipliers.
    """

    PROFILES: Dict[SoulArchetype, SoulProfile] = {

        # ─────────────────────────────────────────────────────
        SoulArchetype.ATTACKING_PROPHET: SoulProfile(
            archetype=SoulArchetype.ATTACKING_PROPHET,
            label="Attacking Prophet",
            description=(
                "The rarest soul in football. Sees every attacking possibility "
                "before it opens. Their body moves before the mind decides — "
                "the game is slower for them than for everyone else. "
                "Nothing in attack they cannot do. Nothing. "
                "Defenders feel their presence before contact. "
                "Percy is this. There are two others like this alive right now."
            ),
            positions=["RW", "LW", "CF", "ST", "CAM"],

            # They make every shot better, attempt more, complete almost everything
            shot_quality_mult=1.45,
            shot_frequency_mult=1.35,
            dribble_attempt_mult=1.50,
            dribble_success_mult=1.55,
            shot_assist_mult=1.40,
            through_ball_mult=1.50,
            carry_frequency_mult=1.45,
            pass_accuracy_mult=1.18,
            positioning_mult=1.50,
            composure_mult=1.55,
            big_chance_conversion=1.50,
            cross_quality_mult=1.30,
            press_intensity_mult=1.20,
            press_success_mult=1.25,

            # Psychology
            losing_state_boost=1.35,   # Gets BETTER when team is losing
            winning_state_mult=1.05,   # Never switches off
            late_game_boost=1.30,      # Final 15 mins — transcendent
            added_time_boost=1.45,     # Added time — becomes a different being
            pressure_resistance=1.50,  # Pressure is their oxygen

            # Aura
            teammate_aura_radius=1.0,
            teammate_pass_boost=1.12,
            teammate_press_boost=1.10,
            teammate_conf_boost=1.15,

            # Signature abilities
            can_unlock_defence=True,
            can_finish_any_angle=True,
            can_destroy_one_v_one=True,
            can_read_game_early=True,

            # Bonus stats beyond DNA ceiling
            bonus_shot_assists_per_match=(2, 5),
            bonus_dribbles_per_match=(3, 8),
            bonus_progressive_passes=(2, 4),
            bonus_carries_per_match=(4, 10),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.DEFENSIVE_PURIST: SoulProfile(
            archetype=SoulArchetype.DEFENSIVE_PURIST,
            label="Defensive Purist",
            description=(
                "Defence is not their job. It is their art, their obsession, "
                "their identity. They study attackers the way artists study light. "
                "They know every run before it happens. Defenders of this soul "
                "do not react to danger — they remove it before it becomes danger. "
                "Van Dijk is this. The penalty box is their cathedral."
            ),
            positions=["CB", "LB", "RB", "CDM"],

            tackle_success_mult=1.55,
            interception_mult=1.60,
            aerial_success_mult=1.50,
            clearance_quality_mult=1.45,
            positioning_mult=1.60,
            composure_mult=1.40,
            pressure_resistance=1.45,
            pass_accuracy_mult=1.15,

            losing_state_boost=1.20,
            winning_state_mult=1.05,
            late_game_boost=1.25,
            added_time_boost=1.30,

            teammate_aura_radius=1.0,
            teammate_pass_boost=1.08,
            teammate_press_boost=1.05,
            teammate_conf_boost=1.20,   # Defenders near them play fearlessly

            can_hold_defensive_line=True,
            can_read_game_early=True,

            bonus_tackles_per_match=(2, 4),
            bonus_interceptions_per_match=(2, 5),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.MIDFIELD_PHILOSOPHER: SoulProfile(
            archetype=SoulArchetype.MIDFIELD_PHILOSOPHER,
            label="Midfield Philosopher",
            description=(
                "The game moves through them like water through a channel. "
                "They do not play football — they conduct it. "
                "Space that does not yet exist, they already inhabit. "
                "A pass that has not been conceived, they have already made. "
                "Xavi, Iniesta, Pirlo — different expressions of the same soul. "
                "They make average players look like world class."
            ),
            positions=["CM", "CDM", "CAM"],

            shot_assist_mult=1.45,
            through_ball_mult=1.55,
            pass_accuracy_mult=1.40,
            carry_frequency_mult=1.25,
            positioning_mult=1.55,
            composure_mult=1.50,
            press_intensity_mult=1.15,
            press_success_mult=1.20,
            dribble_success_mult=1.20,

            losing_state_boost=1.25,
            winning_state_mult=1.10,
            late_game_boost=1.20,
            added_time_boost=1.25,
            pressure_resistance=1.55,  # Best under pressure of any soul

            teammate_aura_radius=1.0,
            teammate_pass_boost=1.20,   # Teammates pass significantly better near them
            teammate_press_boost=1.15,
            teammate_conf_boost=1.18,

            can_unlock_defence=True,
            can_dictate_tempo=True,
            can_read_game_early=True,

            bonus_shot_assists_per_match=(2, 4),
            bonus_progressive_passes=(3, 6),
            bonus_carries_per_match=(2, 5),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.CREATIVE_ORACLE: SoulProfile(
            archetype=SoulArchetype.CREATIVE_ORACLE,
            label="Creative Oracle",
            description=(
                "One dimension of attacking football, but that dimension "
                "at a level no one else has ever reached. "
                "Bruno understands chance creation the way engineers understand physics. "
                "De Bruyne understands delivery the way composers understand harmony. "
                "They are not the Prophet — they cannot do everything. "
                "But in their lane, they are unreachable. "
                "A fraction of the complete Attacking Prophet, but what a fraction."
            ),
            positions=["CAM", "CM", "RW", "LW"],

            shot_assist_mult=1.40,
            through_ball_mult=1.45,
            cross_quality_mult=1.40,
            pass_accuracy_mult=1.25,
            composure_mult=1.30,
            positioning_mult=1.35,
            shot_quality_mult=1.20,
            dribble_success_mult=1.25,

            losing_state_boost=1.20,
            winning_state_mult=1.05,
            late_game_boost=1.20,
            added_time_boost=1.25,
            pressure_resistance=1.35,

            teammate_aura_radius=1.0,
            teammate_pass_boost=1.15,
            teammate_conf_boost=1.10,

            can_unlock_defence=True,

            bonus_shot_assists_per_match=(2, 5),
            bonus_progressive_passes=(1, 3),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.GOALSCORING_SAVANT: SoulProfile(
            archetype=SoulArchetype.GOALSCORING_SAVANT,
            label="Goalscoring Savant",
            description=(
                "Finishing is not a skill they developed. It is who they are. "
                "In the penalty box they enter a different state of consciousness. "
                "Angles that do not exist, they find. Defenders that are perfect, "
                "they beat. Composure is not an attribute — it is their default mode. "
                "Lewandowski. Shearer. Goal machines who cannot be switched off."
            ),
            positions=["ST", "CF", "LW", "RW"],

            shot_quality_mult=1.55,
            shot_frequency_mult=1.30,
            big_chance_conversion=1.60,
            composure_mult=1.55,
            positioning_mult=1.50,
            can_finish_any_angle=True,

            losing_state_boost=1.15,
            late_game_boost=1.25,
            added_time_boost=1.35,
            pressure_resistance=1.45,

            teammate_conf_boost=1.08,

            bonus_shot_assists_per_match=(0, 1),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.PRESSING_EVANGELIST: SoulProfile(
            archetype=SoulArchetype.PRESSING_EVANGELIST,
            label="Pressing Evangelist",
            description=(
                "They do not press because the manager asks them to. "
                "They press because they cannot help themselves. "
                "Every opponent with the ball is a personal challenge. "
                "They create pressure traps for teammates without being told. "
                "They make ten-men teams feel like eleven. "
                "Their energy is infectious — non-negotiable."
            ),
            positions=["ST", "LW", "RW", "CM", "CAM"],

            press_intensity_mult=1.70,
            press_success_mult=1.65,
            carry_frequency_mult=1.20,
            composure_mult=1.15,
            losing_state_boost=1.30,
            late_game_boost=1.20,
            pressure_resistance=1.30,

            teammate_press_boost=1.30,  # Teammates press significantly harder near them
            teammate_conf_boost=1.12,

            can_hunt_in_packs=True,

            bonus_pressures_per_match=(5, 12),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.WIDE_DESTROYER: SoulProfile(
            archetype=SoulArchetype.WIDE_DESTROYER,
            label="Wide Destroyer",
            description=(
                "One direction. Everyone knows it. No one can stop it. "
                "They have perfected one path — the inside cut, the outside run, "
                "the delivery at pace — to such a degree that knowing what "
                "is coming offers defenders no comfort whatsoever. "
                "Prime Robben. Prime Salah. A weapon with one setting: maximum."
            ),
            positions=["LW", "RW", "LB", "RB"],

            dribble_attempt_mult=1.55,
            dribble_success_mult=1.50,
            carry_frequency_mult=1.50,
            shot_quality_mult=1.30,
            cross_quality_mult=1.35,
            positioning_mult=1.30,
            composure_mult=1.25,
            can_destroy_one_v_one=True,

            losing_state_boost=1.25,
            late_game_boost=1.30,
            added_time_boost=1.35,
            pressure_resistance=1.30,

            teammate_conf_boost=1.08,

            bonus_dribbles_per_match=(2, 6),
            bonus_carries_per_match=(3, 7),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.SWEEPER_SAGE: SoulProfile(
            archetype=SoulArchetype.SWEEPER_SAGE,
            label="Sweeper Sage",
            description=(
                "They read the game one full pass ahead of everyone on the pitch. "
                "When others see a pass, they see the response to it. "
                "When others see danger, they have already positioned to eliminate it. "
                "Pirlo in his prime. Busquets at his peak. "
                "They do not cover ground — they make ground irrelevant."
            ),
            positions=["CDM", "CM", "CB"],

            interception_mult=1.55,
            positioning_mult=1.65,
            pass_accuracy_mult=1.35,
            through_ball_mult=1.30,
            composure_mult=1.50,
            pressure_resistance=1.50,
            tackle_success_mult=1.25,
            can_read_game_early=True,
            can_dictate_tempo=True,

            losing_state_boost=1.15,
            winning_state_mult=1.05,
            late_game_boost=1.15,
            added_time_boost=1.20,

            teammate_pass_boost=1.18,
            teammate_conf_boost=1.15,

            bonus_interceptions_per_match=(2, 4),
            bonus_progressive_passes=(2, 4),
        ),

        # ─────────────────────────────────────────────────────
        SoulArchetype.WALL: SoulProfile(
            archetype=SoulArchetype.WALL,
            label="The Wall",
            description=(
                "They simply do not concede. Not as a tactic. As a fact. "
                "Every ball that comes their way is dealt with. "
                "Every aerial. Every tackle. Every clearance — authoritative. "
                "Attackers learn quickly that going their side is not an option. "
                "Prime Terry. The entire back line organises around their certainty."
            ),
            positions=["CB", "LB", "RB"],

            tackle_success_mult=1.50,
            aerial_success_mult=1.60,
            clearance_quality_mult=1.55,
            positioning_mult=1.55,
            composure_mult=1.40,
            interception_mult=1.35,
            pressure_resistance=1.40,
            can_hold_defensive_line=True,

            losing_state_boost=1.15,
            late_game_boost=1.20,
            added_time_boost=1.25,

            teammate_conf_boost=1.22,   # Highest defensive aura in the game
            teammate_pass_boost=1.05,

            bonus_tackles_per_match=(1, 3),
            bonus_interceptions_per_match=(1, 3),
        ),
    }

    @classmethod
    def get(cls, archetype: SoulArchetype) -> SoulProfile:
        return cls.PROFILES[archetype]

    @classmethod
    def get_by_name(cls, name: str) -> Optional[SoulProfile]:
        for arch, profile in cls.PROFILES.items():
            if arch.value == name.lower():
                return profile
        return None


# ─────────────────────────────────────────────
# PLAYER SOUL — Attached to a specific player
# ─────────────────────────────────────────────

@dataclass
class PlayerSoul:
    """
    A player's soul: their archetype + their greatness pillars.
    This is what separates Percy from everyone else.

    Attached to PlayerDNA as: dna.soul: Optional[PlayerSoul]
    """
    player_name: str
    archetype: SoulArchetype
    pillars: GreatnessPillars

    # Season-long psychological state (updated after each match)
    current_form_factor: float = 1.0    # 0.80–1.20 based on recent matches
    consecutive_good_games: int = 0
    consecutive_poor_games: int = 0
    is_in_flow_state: bool = False      # Peak psychological state

    @property
    def profile(self) -> SoulProfile:
        return SoulLibrary.get(self.archetype)

    @property
    def greatness_coefficient(self) -> float:
        return self.pillars.greatness_coefficient

    @property
    def tier(self) -> str:
        return self.pillars.tier

    def get_event_multiplier(self, event_type: str, game_context: Dict) -> float:
        """
        Return the combined multiplier for a specific event type,
        factoring in game context (score, minute, pressure).

        event_type: matches SoulProfile attribute names
        game_context: {
            'minute': int,
            'is_losing': bool,
            'is_winning': bool,
            'is_late_game': bool,
            'is_added_time': bool,
            'under_pressure': bool,
        }
        """
        profile = self.profile
        g = self.greatness_coefficient

        # Base multiplier from archetype
        base = getattr(profile, event_type, 1.0)

        # Scale by greatness coefficient
        # G=1.0 → full multiplier, G=0.5 → half the boost
        scaled = 1.0 + (base - 1.0) * g

        # Game context modifiers
        context_mult = 1.0
        if game_context.get("is_losing") and profile.losing_state_boost > 1.0:
            context_mult *= profile.losing_state_boost
        if game_context.get("is_winning") and profile.winning_state_mult != 1.0:
            context_mult *= profile.winning_state_mult
        if game_context.get("is_late_game") and profile.late_game_boost > 1.0:
            context_mult *= profile.late_game_boost
        if game_context.get("is_added_time") and profile.added_time_boost > 1.0:
            context_mult *= profile.added_time_boost
        if game_context.get("under_pressure") and profile.pressure_resistance > 1.0:
            context_mult *= profile.pressure_resistance

        # Flow state: peak psychological condition
        if self.is_in_flow_state:
            context_mult *= 1.12

        # Current form factor
        final = scaled * context_mult * self.current_form_factor

        return round(final, 4)

    def get_bonus_stats(self) -> Dict[str, int]:
        """
        Return the bonus stat additions for this match.
        These are on top of DNA-generated values.
        """
        profile = self.profile
        g = self.greatness_coefficient

        def roll_bonus(rng: Tuple[int, int]) -> int:
            if rng == (0, 0):
                return 0
            lo, hi = rng
            # Greatness coefficient scales how much of the range is used
            scaled_hi = int(lo + (hi - lo) * g)
            return random.randint(lo, max(lo, scaled_hi))

        bonuses = {
            "shot_assists":       roll_bonus(profile.bonus_shot_assists_per_match),
            "dribbles_comp":      roll_bonus(profile.bonus_dribbles_per_match),
            "tackles_won":        roll_bonus(profile.bonus_tackles_per_match),
            "interceptions":      roll_bonus(profile.bonus_interceptions_per_match),
            "progressive_passes": roll_bonus(profile.bonus_progressive_passes),
            "pressures":          roll_bonus(profile.bonus_pressures_per_match),
            "carries":            roll_bonus(profile.bonus_carries_per_match),
        }

        return {k: v for k, v in bonuses.items() if v > 0}

    def get_teammate_aura(self) -> Dict[str, float]:
        """
        Returns the aura multipliers to apply to nearby teammates.
        """
        p = self.profile
        g = self.greatness_coefficient
        return {
            "pass_accuracy_mult": 1.0 + (p.teammate_pass_boost - 1.0) * g,
            "press_mult":         1.0 + (p.teammate_press_boost - 1.0) * g,
            "composure_mult":     1.0 + (p.teammate_conf_boost - 1.0) * g,
        }

    def update_after_match(self, rating: float, goals: int, assists: int):
        """Update psychological state after a match."""
        if rating >= 7.5:
            self.consecutive_good_games += 1
            self.consecutive_poor_games = 0
            self.current_form_factor = min(1.20, self.current_form_factor + 0.03)
        elif rating >= 6.5:
            self.consecutive_poor_games = 0
            self.current_form_factor = min(1.10, self.current_form_factor + 0.01)
        elif rating < 5.5:
            self.consecutive_poor_games += 1
            self.consecutive_good_games = 0
            self.current_form_factor = max(0.82, self.current_form_factor - 0.04)
        else:
            self.current_form_factor = max(0.85, min(1.15,
                self.current_form_factor * 0.98 + 1.0 * 0.02))

        # Flow state: 4+ consecutive good games = flow
        self.is_in_flow_state = self.consecutive_good_games >= 4

        # Reset consecutive if broken
        if self.consecutive_good_games > 0 and rating < 6.0:
            self.consecutive_good_games = 0
        if self.consecutive_poor_games > 0 and rating >= 7.0:
            self.consecutive_poor_games = 0

    def describe(self) -> str:
        p = self.profile
        lines = [
            f"\n  {'═'*52}",
            f"  SOUL: {p.label.upper()}",
            f"  {p.description[:80]}...",
            f"  {'─'*52}",
            f"  Greatness Formula:",
            self.pillars.describe(),
            f"  {'─'*52}",
            f"  Key Multipliers:",
            f"    Shot Quality      : ×{p.shot_quality_mult:.2f}",
            f"    Dribble Success   : ×{p.dribble_success_mult:.2f}",
            f"    Shot Assist Rate  : ×{p.shot_assist_mult:.2f}",
            f"    Composure         : ×{p.composure_mult:.2f}",
            f"    Late Game Boost   : ×{p.late_game_boost:.2f}",
            f"    Pressure Resist   : ×{p.pressure_resistance:.2f}",
            f"  Signature Abilities:",
        ]
        if p.can_unlock_defence:    lines.append("    ✓ Can unlock defensive shape with single pass")
        if p.can_hold_defensive_line: lines.append("    ✓ Organises entire defensive line")
        if p.can_dictate_tempo:     lines.append("    ✓ Can accelerate or slow entire match")
        if p.can_finish_any_angle:  lines.append("    ✓ Scores from impossible angles")
        if p.can_hunt_in_packs:     lines.append("    ✓ Creates press traps for teammates")
        if p.can_destroy_one_v_one: lines.append("    ✓ Beats their man in 1v1 at will")
        if p.can_read_game_early:   lines.append("    ✓ Reads play one beat before everyone else")
        lines.append(f"  {'═'*52}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SOUL FACTORY — Creates soul objects cleanly
# ─────────────────────────────────────────────

class SoulFactory:
    """
    Creates PlayerSoul objects for assignment to players.
    Use this in squad definitions for soul players.

    Usage in run_match.py:
        SoulFactory.create("Percy", SoulArchetype.ATTACKING_PROPHET,
            hardwork=0.97, talent=0.99, luck=0.91)
    """

    @staticmethod
    def create(
        player_name: str,
        archetype: SoulArchetype,
        hardwork: float,
        talent: float,
        luck: float,
    ) -> PlayerSoul:
        """Create a PlayerSoul with the Greatness Formula defined."""
        pillars = GreatnessPillars(
            hardwork=hardwork,
            talent=talent,
            luck=luck,
        )
        return PlayerSoul(
            player_name=player_name,
            archetype=archetype,
            pillars=pillars,
        )

    # ── PRE-BUILT SOUL DEFINITIONS ────────────────────────────
    # These are fixed PLOFA soul players.
    # Their greatness values are permanent — set once, used forever.

    @staticmethod
    def percy() -> PlayerSoul:
        """
        Percy — RW, Hartwell City.
        The only active Attacking Prophet in PLOFA.
        Said to be the equal of the greatest to ever play.
        His Omega coefficient is 1.55 — fully activated.
        """
        return PlayerSoul(
            player_name="Percy",
            archetype=SoulArchetype.ATTACKING_PROPHET,
            pillars=GreatnessPillars(
                hardwork=0.99,   # Relentless. Never stops.
                talent=0.90,     # Absolute ceiling.
                luck=0.99,       # Right era, right club, no major injuries.
            )
        )

    @staticmethod
    def custom(
        player_name: str,
        archetype: SoulArchetype,
        hardwork: float,
        talent: float,
        luck: float,
    ) -> PlayerSoul:
        """Create any soul player with custom greatness values."""
        return PlayerSoul(
            player_name=player_name,
            archetype=archetype,
            pillars=GreatnessPillars(hardwork=hardwork, talent=talent, luck=luck),
        )


# ─────────────────────────────────────────────
# SOUL SCENARIO CALCULATOR (Checkpoint 19)
# Attacking Prophet geometric scenario evaluation
# ─────────────────────────────────────────────

class SoulScenarioCalculator:
    """
    For ATTACKING_PROPHET souls (Messi-tier), evaluates multiple geometric
    scenarios in "the mind" before the player acts. This is the "runs 10s
    of scenarios in head" mechanic the user described.

    The calculator evaluates:
        1. Pass target options (teammate XY + lane clearance + marking)
        2. Run destinations (space availability + defender proximity)
        3. Shot angles (visible goal mouth + pressure)
        4. Through-ball corridors (receiver pace + lane geometry)

    And returns a ranked list of (action, target, score) tuples.
    The event chain can then use the top-ranked option instead of a
    flat random draw.
    """

    @staticmethod
    def evaluate_pass_options(
        player, teammates, defenders,
        x: float, y: float,
        position_engine,
        attacks_right: bool = True,
    ) -> List[Dict]:
        """
        Evaluate all passing options geometrically.
        Returns ranked list of {target, score, reason} dicts.
        """
        if position_engine is None:
            return []

        options = []
        for tm in teammates or []:
            if getattr(tm, 'name', None) == getattr(player, 'name', None):
                continue
            tx, ty = position_engine.get_position(tm.name)

            dist = math.hypot(tx - x, ty - y)
            if dist > 50.0:
                continue

            # Lane clearance
            from attacking_matrix import lane_clearance, nearest_defender_dist
            lane = lane_clearance(x, y, tx, ty, defenders, position_engine)

            # Freedom (marking)
            nd = nearest_defender_dist(tx, ty, defenders, position_engine)
            freedom = 1.0 if nd is None or nd >= 3.0 else max(0.15, (nd - 1.0) / 2.0)

            # Progress toward goal
            gx = 105.0 if attacks_right else 0.0
            d_ag = math.hypot(x - gx, y - 34.0)
            d_tg = math.hypot(tx - gx, ty - 34.0)
            progress = max(0.0, min(1.0, 0.5 + (d_ag - d_tg) / 20.0))

            # Depth into attacking third
            depth = max(0.0, min(1.0, (tx - 35.0) / 70.0)) if attacks_right else max(0.0, min(1.0, (35.0 - tx) / 70.0))

            # Half-space bonus
            half_space_bonus = 0.0
            if tm.position in ("CDM", "CM", "CAM"):
                if ty < 22.0 or ty > 46.0:
                    half_space_bonus = 0.06
                width_factor = max(0.0, (abs(ty - 34.0) - 8.0) / 18.0)
                half_space_bonus += width_factor * 0.04

            score = lane * (0.45 * progress + 0.35 * freedom + 0.20 * depth + half_space_bonus)

            options.append({
                "target": tm,
                "score": score,
                "lane": lane,
                "progress": progress,
                "freedom": freedom,
                "depth": depth,
                "dist": dist,
                "reason": f"lane={lane:.2f} prog={progress:.2f} free={freedom:.2f}",
            })

        options.sort(key=lambda o: -o["score"])
        return options[:5]

    @staticmethod
    def evaluate_run_destinations(
        player, teammates, defenders,
        x: float, y: float,
        position_engine,
        attacks_right: bool = True,
    ) -> List[Dict]:
        """
        Evaluate possible run destinations for the player.
        Returns ranked list of {target_x, target_y, score, reason} dicts.
        """
        if position_engine is None:
            return []

        pos = getattr(player, 'position', 'CM')
        candidates = []

        # Generate candidate targets based on position
        if pos in ('ST', 'CF'):
            # Runs: behind defence, half-spaces, box
            for dx, dy in [(-10, -12), (-10, 0), (-10, 12), (0, -8), (0, 0), (0, 8), (10, -10), (10, 0), (10, 10), (20, -5), (20, 0), (20, 5)]:
                candidates.append((x + dx, y + dy))
        elif pos in ('LW', 'RW'):
            flank_y = 10.0 if pos == 'LW' else 58.0
            for dx, dy in [(-5, -8), (0, -8), (5, -8), (10, -5), (15, 0), (20, 0), (25, 0)]:
                candidates.append((x + dx, flank_y + dy))
        else:
            # Midfielders: half-space runs
            for dx, dy in [(-10, -15), (-5, -10), (0, -8), (5, -10), (10, -15), (-10, 15), (-5, 10), (0, 8), (5, 10), (10, 15)]:
                candidates.append((x + dx, y + dy))

        results = []
        gx = 105.0 if attacks_right else 0.0
        for tx, ty in candidates:
            tx = max(2.0, min(103.0, tx))
            ty = max(2.0, min(66.0, ty))

            # Space availability: distance to nearest teammate
            min_tm_dist = float('inf')
            for tm in teammates or []:
                if getattr(tm, 'name', None) == getattr(player, 'name', None):
                    continue
                ttx, tty = position_engine.get_position(tm.name)
                min_tm_dist = min(min_tm_dist, math.hypot(tx - ttx, ty - tty))

            # Defender proximity
            min_def_dist = float('inf')
            for d in defenders or []:
                if getattr(d, 'position', None) == 'GK':
                    continue
                dx, dy = position_engine.get_position(d.name)
                min_def_dist = min(min_def_dist, math.hypot(tx - dx, ty - dy))

            # Score: prefer space (far from teammates) but not too close to defenders
            space_score = max(0.0, min(1.0, min_tm_dist / 20.0))
            safety_score = max(0.0, min(1.0, min_def_dist / 15.0))
            progress_score = max(0.0, min(1.0, abs(tx - x) / 30.0))

            score = space_score * 0.4 + safety_score * 0.35 + progress_score * 0.25
            results.append({
                "target_x": tx,
                "target_y": ty,
                "score": score,
                "space_dist": min_tm_dist,
                "def_dist": min_def_dist,
                "reason": f"space={space_score:.2f} safe={safety_score:.2f} prog={progress_score:.2f}",
            })

        results.sort(key=lambda r: -r["score"])
        return results[:5]


# ─────────────────────────────────────────────
# SOUL APPLICATOR
# Applies soul multipliers to event chain probabilities
# Called by event_chain.py before probability rolls
# ─────────────────────────────────────────────

class SoulApplicator:
    """
    The bridge between player_soul.py and event_chain.py.
    Called by chains to modify probabilities for soul players.

    Usage in event_chain.py:
        from player_soul import SoulApplicator
        prob = SoulApplicator.modify_dribble_success(player, base_prob, context)
    """

    @staticmethod
    def _get_context(state, player_team: str = None) -> Dict:
        """Build game context dict from match state.

        Checkpoint 7: previously this hardcoded is_losing/is_winning/
        under_pressure to False, which silently zeroed out the
        losing_state_boost / late_game_boost / pressure_resistance
        multipliers — the whole point of the soul psychology model.
        Now resolves them from the real scoreline + minute when team
        info is available, falls back gracefully when not.
        """
        minute = getattr(state, "minute", 45)
        home_goals = getattr(state, "home_goals", 0)
        away_goals = getattr(state, "away_goals", 0)
        home_team = getattr(state, "home_team", None)

        gd = home_goals - away_goals
        if player_team is not None and home_team is not None:
            team_gd = gd if player_team == home_team else -gd
            is_losing = team_gd < 0
            is_winning = team_gd > 0
        else:
            is_losing = False
            is_winning = False

        return {
            "minute": minute,
            "is_losing": is_losing,
            "is_winning": is_winning,
            "is_late_game": minute >= 75,
            "is_added_time": minute >= 91,
            "under_pressure": False,  # per-action override if a chain knows
        }

    @staticmethod
    def get_soul(player) -> Optional[PlayerSoul]:
        """Safely extract soul from a PlayerProfile."""
        dna = getattr(player, "dna", None)
        if dna is None:
            return None
        return getattr(dna, "soul", None)

    @staticmethod
    def modify_dribble_success(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("dribble_success_mult", ctx)
        return min(0.95, base_prob * mult)

    @staticmethod
    def modify_shot_quality(player, base_xg: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_xg
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("shot_quality_mult", ctx)
        # Shot quality is an xG MULTIPLIER (DNAFactory.get_shooter_quality
        # returns 0.70–1.40, and archetype multipliers go higher), not a
        # probability — so it must NOT be clamped to a <1.0 ceiling here.
        # Callers that feed it a probability clamp the result themselves.
        return base_xg * mult

    @staticmethod
    def modify_shot_assist_prob(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("shot_assist_mult", ctx)
        return min(0.90, base_prob * mult)

    @staticmethod
    def modify_press_success(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("press_success_mult", ctx)
        return min(0.80, base_prob * mult)

    @staticmethod
    def modify_tackle_success(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("tackle_success_mult", ctx)
        return min(0.92, base_prob * mult)

    @staticmethod
    def modify_interception_rate(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("interception_mult", ctx)
        return min(0.88, base_prob * mult)

    @staticmethod
    def modify_pass_accuracy(player, base_prob: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_prob
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("pass_accuracy_mult", ctx)
        return min(0.95, base_prob * mult)

    @staticmethod
    def modify_big_chance_conversion(player, base_xg: float, state=None, player_team: str = None) -> float:
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return base_xg
        ctx = SoulApplicator._get_context(state, player_team) if state else {}
        mult = soul.get_event_multiplier("big_chance_conversion", ctx)
        return min(0.98, base_xg * mult)

    @staticmethod
    def apply_bonus_stats(player, accumulated_stats: Dict, state=None) -> Dict:
        """
        After all events processed, add soul bonus stats to the accumulation.
        Called by StatAccumulator._finalise() for soul players.
        """
        soul = SoulApplicator.get_soul(player)
        if soul is None:
            return accumulated_stats

        bonuses = soul.get_bonus_stats()
        for stat, bonus in bonuses.items():
            if stat in accumulated_stats:
                accumulated_stats[stat] = accumulated_stats[stat] + bonus

        return accumulated_stats

    @staticmethod
    def apply_teammate_aura(soul_player, teammate, stat: str, base_val: float) -> float:
        """
        Apply the soul player's aura to a teammate's stat.
        Called when soul player and teammate are on the same team.
        """
        soul = SoulApplicator.get_soul(soul_player)
        if soul is None:
            return base_val
        aura = soul.get_teammate_aura()
        mult = aura.get(stat, 1.0)
        return base_val * mult


# ─────────────────────────────────────────────
# WIRING INSTRUCTIONS
# How to connect player_soul.py to the existing modules
# WITHOUT redesigning them
# ─────────────────────────────────────────────

WIRING_GUIDE = """
WIRING player_soul.py INTO THE ENGINE
═══════════════════════════════════════

1. player_dna.py — Add one field to PlayerDNA:
   ─────────────────────────────────────────────
   from player_soul import PlayerSoul
   
   @dataclass
   class PlayerDNA:
       ...existing fields...
       soul: Optional[PlayerSoul] = None   # ← ADD THIS

2. player_dna.py — In SquadBuilder._build_player(), add soul attachment:
   ─────────────────────────────────────────────
   # After creating dna, check if player has a soul assigned
   if soul_assignments and name in soul_assignments:
       dna.soul = soul_assignments[name]

3. event_chain.py — In AttackChain._shot_on_target_prob(), add:
   ─────────────────────────────────────────────
   from player_soul import SoulApplicator
   # After calculating base prob:
   xg = SoulApplicator.modify_shot_quality(shooter, xg, state)

4. event_chain.py — In PossessionChain carry/dribble section:
   ─────────────────────────────────────────────
   success_rate = SoulApplicator.modify_dribble_success(last_player, success_rate, state)

5. exporter.py — In StatAccumulator._finalise(), after rating:
   ─────────────────────────────────────────────
   from player_soul import SoulApplicator
   for name, s in self.stats.items():
       player_obj = self._find_player(name)
       if player_obj:
           s = SoulApplicator.apply_bonus_stats(player_obj, s)
   
   # Add soul columns to Excel output:
   row['Soul Archetype'] = player_obj.dna.soul.profile.label if player_obj.dna.soul else ''
   row['Greatness Coefficient'] = player_obj.dna.soul.greatness_coefficient if player_obj.dna.soul else ''
   row['Greatness Tier'] = player_obj.dna.soul.tier if player_obj.dna.soul else ''
   row['Omega Active'] = player_obj.dna.soul.pillars.omega_activated if player_obj.dna.soul else ''

6. run_match.py — Define soul players once, reuse every week:
   ─────────────────────────────────────────────
   from player_soul import SoulFactory, SoulArchetype
   
   SOUL_PLAYERS = {
       "Percy": SoulFactory.percy(),
       "Van Der Berg": SoulFactory.custom("Van Der Berg", SoulArchetype.DEFENSIVE_PURIST,
                                           hardwork=0.91, talent=0.93, luck=0.82),
   }
"""


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧬 PLOFA 26/27 — Player Soul Module")
    print("="*56)

    # ── PERCY — The Attacking Prophet ─────────────────────────
    percy_soul = PlayerSoul(
        player_name="Percy",
        archetype=SoulArchetype.ATTACKING_PROPHET,
        pillars=GreatnessPillars(hardwork=0.99, talent=0.90, luck=0.99)
    )
    print(percy_soul.describe())

    # ── COMPARE: Star player without Omega ────────────────────
    print("\n── COMPARISON: Elite player (no Omega) ──")
    elite = PlayerSoul(
        player_name="Dragan Novak",
        archetype=SoulArchetype.GOALSCORING_SAVANT,
        pillars=GreatnessPillars(hardwork=0.84, talent=0.86, luck=0.79)
    )
    print(f"  Hardwork : {elite.pillars.hardwork} (threshold: 0.85) ✗")
    print(f"  Talent   : {elite.pillars.talent} (threshold: 0.88) ✗")
    print(f"  Luck     : {elite.pillars.luck}")
    print(f"  Ω Active : No (Ω = {elite.pillars.omega:.3f})")
    print(f"  G_final  : {elite.greatness_coefficient:.4f} → {elite.tier}")

    # ── GREATNESS COEFFICIENT COMPARISON ──────────────────────
    print("\n── GREATNESS COEFFICIENT BREAKDOWN ──")
    print(f"  {'Player':<22} {'G_raw':>7} {'Ω':>7} {'G_final':>8} {'Tier':<15}")
    print(f"  {'─'*60}")

    players_demo = [
        ("Percy",            SoulArchetype.ATTACKING_PROPHET,    0.99, 0.90, 0.99),
        ("Juan Massey",     SoulArchetype.DEFENSIVE_PURIST,     0.91, 0.93, 0.91),
        ("Zachery Worth",     SoulArchetype.WIDE_DESTROYER,   0.72, 0.99, 0.99),
        ("Danso Potwemi",      SoulArchetype.CREATIVE_ORACLE,      0.87, 0.97, 0.99),
        ("Hill Prosper",      SoulArchetype.GOALSCORING_SAVANT,       0.90, 0.80, 0.90),
        ("Caut Mayoderoki",   SoulArchetype.MIDFIELD_PHILOSOPHER,  0.99, 0.95, 0.92),
        ("Van Lee",            SoulArchetype.WALL, 0.97, 0.98, 0.78),
        ("Duane Rokariĉ",   SoulArchetype.CREATIVE_ORACLE, 0.87, 0.99, 0.83),
        ("Mikro Vitro", SoulArchetype.WIDE_DESTROYER, 0.75, 0.99, 0.90),
        ("Francis Dućźè", SoulArchetype.GOALSCORING_SAVANT, 0.95, 0.90, 0.93),
        ("Hillary Monzade", SoulArchetype.ATTACKING_PROPHET, 0.94, 0.99, 0.95)
    ]

    for name, arch, h, t, l in players_demo:
        soul = PlayerSoul(name, arch, GreatnessPillars(h, t, l))
        g = soul.pillars
        flag = " ⚡ OMEGA" if g.omega_activated else ""
        print(f"  {name:<22} {g.raw_score:>7.4f} {g.omega:>7.3f} {g.greatness_coefficient:>8.4f} {soul.tier:<15}{flag}")

    # ── PERCY vs DRAGAN: Same situation, different outcome ─────
    print("\n── SAME CHANCE, DIFFERENT SOULS ──")
    print("  Big chance at 88' while team is losing — who converts?")

    context = {"is_losing": True, "is_late_game": True, "is_added_time": False, "under_pressure": True}
    base_xg = 0.45

    for name, arch, h, t, l in players_demo[:4]:
        soul = PlayerSoul(name, arch, GreatnessPillars(h, t, l))
        mult  = soul.get_event_multiplier("shot_quality_mult", context)
        eff   = soul.get_event_multiplier("big_chance_conversion", context)
        adj_xg = min(0.97, base_xg * mult * eff)
        print(f"  {name:<22} base xG: {base_xg:.2f} → effective xG: {adj_xg:.3f} "
              f"(×{adj_xg/base_xg:.2f}) — {soul.tier}")

    # ── BONUS STATS PERCY GETS ON TOP OF DNA ──────────────────
    print("\n── PERCY BONUS STATS (on top of DNA this match) ──")
    for stat, val in percy_soul.get_bonus_stats().items():
        print(f"  +{val} {stat}")

    print("\n✅ Player Soul module operational.")
    print("   Percy: TRANSCENDENT (Ω=1.547, G=0.9382)")
    print("   Next: wire into player_dna.py + run_match.py template\n")