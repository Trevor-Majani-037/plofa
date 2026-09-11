"""
PLOFA 26/27 — PLAYER PERSONALITY MODULE
==========================================
player_personality.py

Closes gap #1 from the analyst feedback: "No mental traits beyond attributes."

Philosophy:
    Attributes (player_dna.py) say what a player CAN do.
    Soul (player_soul.py) says how TRANSCENDENT they are.
    Personality says how RELIABLY they show up and how they behave
    under the specific pressures of a season — training, big games,
    contract years, dressing-room politics.

    None of this is cosmetic. Every trait below is wired to a concrete
    numeric effect on match day or on availability between matches.

Traits (0-100, 50 = league-average):
    professionalism     — training application, injury/illness avoidance,
                           consistency of starting stamina between matches
    ambition             — drive in must-win games, likelihood of agitating
                           for a move when unhappy (season_manager.py hook)
    loyalty               — inverse of transfer unrest, dressing room stability
    temperament           — composure under provocation; low temperament
                           raises foul/card probability multiplicatively
    consistency            — variance dial on match rating; low consistency
                           = higher rating variance week to week
    big_game_mentality     — performance multiplier specifically in
                           "big games" (derby, rival, cup final flag)
    adaptability            — how much penalty a player takes when used
                           out of position or in an unfamiliar formation
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# PERSONALITY TRAITS
# ─────────────────────────────────────────────

@dataclass
class PersonalityTraits:
    """Psychological makeup beyond attributes and soul. 0-100 scale, 50 = average."""
    professionalism: float     = 50.0   # Turns up, maintains fitness, avoids scandals
    ambition: float            = 50.0   # Wants to win, pushes for moves, leadership drive
    loyalty: float             = 50.0   # Stays with club, doesn't agitate
    temperament: float         = 50.0   # Stays cool vs loses head (affects cards)
    consistency: float         = 50.0   # Performs every week vs inconsistent
    big_game_mentality: float  = 50.0   # Rises in finals vs disappears
    adaptability: float        = 50.0   # Fits different formations/positions

    def __post_init__(self):
        for f in ("professionalism", "ambition", "loyalty", "temperament",
                  "consistency", "big_game_mentality", "adaptability"):
            setattr(self, f, max(0.0, min(100.0, getattr(self, f))))

    # ── LIVE MATCH-DAY EFFECTS ─────────────────────────────────

    @property
    def card_risk_mult(self) -> float:
        """Low temperament raises foul-committed/card probability. Range ~0.75x-1.35x."""
        return round(1.35 - (self.temperament / 100.0) * 0.60, 3)

    @property
    def rating_variance_mult(self) -> float:
        """
        Low consistency widens the spread applied to a player's match rating.
        1.0 = average spread, up to 1.8x for a genuinely mercurial player,
        down to 0.55x for a metronome.
        """
        return round(1.80 - (self.consistency / 100.0) * 1.25, 3)

    def big_game_multiplier(self, is_big_game: bool) -> float:
        """
        Applied to the player's soul/DNA event multipliers when is_big_game=True
        (derby, rival fixture, cup final — set via MatchConfig.is_derby or a
        season_manager fixture flag).
        A player with big_game_mentality=85 gets a real lift; one at 20 fades.
        """
        if not is_big_game:
            return 1.0
        return round(0.78 + (self.big_game_mentality / 100.0) * 0.44, 3)  # 0.78x - 1.22x

    def adaptability_penalty(self, is_out_of_position: bool) -> float:
        """Multiplicative penalty to overall_rating-derived probabilities when
        played out of natural position. High adaptability nearly erases it."""
        if not is_out_of_position:
            return 1.0
        return round(0.75 + (self.adaptability / 100.0) * 0.24, 3)  # 0.75x - 0.99x

    # ── BETWEEN-MATCH / SEASON EFFECTS (used by season_manager.py) ──

    @property
    def training_fitness_mult(self) -> float:
        """
        Professionalism affects how reliably starting stamina recovers
        between matches. Low professionalism = occasionally undercooked.
        """
        return round(0.88 + (self.professionalism / 100.0) * 0.14, 3)  # 0.88x-1.02x

    @property
    def unfit_risk(self) -> float:
        """Chance a player turns up short of full training sharpness for
        a given match — used by season_manager availability rolls."""
        return round(max(0.01, 0.12 - (self.professionalism / 100.0) * 0.11), 3)

    @property
    def unrest_pressure(self) -> float:
        """
        Combines low loyalty + high ambition into a season-long 'wants out'
        pressure score (0-1). Season_manager can use this to flag transfer
        request narratives without needing a full transfer market module.
        """
        want_more = self.ambition / 100.0
        stays_put = self.loyalty / 100.0
        return round(max(0.0, want_more - stays_put) * 0.8 + want_more * 0.2, 3)

    def describe(self) -> str:
        def bar(v):
            filled = int(v / 10)
            return "█" * filled + "░" * (10 - filled)
        lines = [
            f"  Professionalism   {bar(self.professionalism)} {self.professionalism:.0f}",
            f"  Ambition          {bar(self.ambition)} {self.ambition:.0f}",
            f"  Loyalty           {bar(self.loyalty)} {self.loyalty:.0f}",
            f"  Temperament       {bar(self.temperament)} {self.temperament:.0f}  (card risk ×{self.card_risk_mult})",
            f"  Consistency       {bar(self.consistency)} {self.consistency:.0f}  (rating variance ×{self.rating_variance_mult})",
            f"  Big Game Mindset  {bar(self.big_game_mentality)} {self.big_game_mentality:.0f}",
            f"  Adaptability      {bar(self.adaptability)} {self.adaptability:.0f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# ARCHETYPE-INFORMED GENERATION
# Specialties/archetype nudge personality distributions so results
# feel earned rather than pure noise (a "captain" tag skews leadership
# traits up; "dirty_player" skews temperament down, etc.)
# ─────────────────────────────────────────────

_SPECIALTY_NUDGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # specialty -> {trait: (mean_shift, extra_std)}
    "captain":            {"ambition": (12, 0), "loyalty": (10, 0), "temperament": (10, 0)},
    "leadership":         {"ambition": (10, 0), "loyalty": (6, 0)},
    "dirty_player":       {"temperament": (-18, 4)},
    "big_game_player":    {"big_game_mentality": (20, 4)},
    "clutch":             {"big_game_mentality": (15, 3), "temperament": (8, 0)},
    "press_resistant":    {"consistency": (8, 0)},
    "workhorse":          {"professionalism": (10, 0)},
    "engine":             {"professionalism": (6, 0)},
    "two_footed":         {"adaptability": (10, 0)},
    "super_sub":          {"adaptability": (12, 0), "big_game_mentality": (6, 0)},
}


class PersonalityFactory:
    """Generates a PersonalityTraits instance with realistic natural variation."""

    @staticmethod
    def create(specialties: Optional[List[str]] = None, seed_name: str = "") -> PersonalityTraits:
        specs = specialties or []
        base = {
            "professionalism": random.gauss(58, 15),
            "ambition":        random.gauss(55, 16),
            "loyalty":         random.gauss(52, 16),
            "temperament":     random.gauss(55, 16),
            "consistency":     random.gauss(55, 15),
            "big_game_mentality": random.gauss(50, 16),
            "adaptability":    random.gauss(50, 14),
        }
        for spec in specs:
            nudges = _SPECIALTY_NUDGES.get(spec, {})
            for trait, (shift, extra_std) in nudges.items():
                noise = random.gauss(shift, extra_std) if extra_std else shift
                base[trait] += noise

        clipped = {k: max(5.0, min(97.0, round(v, 1))) for k, v in base.items()}
        return PersonalityTraits(**clipped)

    @staticmethod
    def custom(**kwargs) -> PersonalityTraits:
        """Explicit hand-authored personality (for named/story players)."""
        return PersonalityTraits(**kwargs)


# ─────────────────────────────────────────────
# WIRING GUIDE
# ─────────────────────────────────────────────

WIRING_GUIDE = """
WIRING player_personality.py INTO THE ENGINE
═════════════════════════════════════════════

1. player_dna.py — already patched: PlayerDNA.personality: Optional[PersonalityTraits]

2. player_dna.py — SquadBuilder._build_player(): assign a default personality
   if none is provided:
       from player_personality import PersonalityFactory
       dna.personality = PersonalityFactory.create(specialties, name)

3. event_chain.py — DisciplineChain._pick_fouler() / card_prob calculation:
       if fouler.dna.personality:
           card_prob *= fouler.dna.personality.card_risk_mult

4. exporter.py — StatAccumulator._calculate_rating():
       if player_obj.dna.personality:
           variance = player_obj.dna.personality.rating_variance_mult
           r += random.gauss(0, 0.35 * variance)   # replaces implicit fixed noise

5. run_match.py — set IS_DERBY / a "big_game" flag, then in event_chain shot
   quality / soul multiplier stacking:
       big_game_mult = shooter.dna.personality.big_game_multiplier(config.is_derby)
       xg *= big_game_mult

6. season_manager.py — availability rolls use personality.unfit_risk and
   personality.training_fitness_mult to decide starting_stamina each week,
   and personality.unrest_pressure to flag "wants a move" narrative players.
"""
