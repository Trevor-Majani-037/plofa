"""
PLOFA 26/27 — SQUAD MANAGER
==============================
squad_manager.py

Three systems in one module:

    1. AVAILABILITY CHECKER
       Reads previous match output files and determines who can play.
       Flags: suspended (red card / 5 yellows in 6 games), injured,
       fatigued (high stamina drain), or fit to start.

    2. STAMINA ENGINE
       Every action costs stamina. Heavier actions cost more.
       As stamina drops, performance degrades progressively.
       At critical levels the player SHOULD come off — but doesn't always.
       Stamina is tracked minute-by-minute during simulation.

    3. SUBSTITUTION CONTROLLER
       Subs happen when:
         a) Designated tactical minute reached (pre-planned)
         b) Player hits critical stamina (forced)
         c) Player injured in-match
         d) Game state demands it (chasing goal → attacker on)
       Fresh subs get a freshness burst. Super subs get it amplified.
       Sometimes tired players finish matches — that randomness stays.
"""

from __future__ import annotations
import os
import json
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta
from enum import Enum

import pandas as pd


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class AvailabilityStatus(Enum):
    FIT            = "fit"             # Ready to start or sub
    SUSPENDED_RED  = "suspended_red"   # Serving red card ban
    SUSPENDED_YEL  = "suspended_yel"   # 5 yellows in 6 games
    INJURED        = "injured"         # Missing matches from injury
    FATIGUE_WARNING = "fatigue_warning" # Heavy drain last match — monitor
    DOUBTFUL       = "doubtful"        # Borderline, manager decision

class SubReason(Enum):
    TACTICAL       = "tactical"        # Pre-planned tactical change
    STAMINA        = "stamina"         # Player too drained to continue
    INJURY         = "injury"          # In-match injury forces change
    GAME_STATE     = "game_state"      # Losing/winning → different profile needed
    DISCIPLINARY   = "disciplinary"    # Player got red card, replacing slot


# ─────────────────────────────────────────────
# STAMINA COSTS PER ACTION
# How much stamina (%) each action type drains
# Real physiology: sprints and aerials are most taxing
# ─────────────────────────────────────────────

STAMINA_COSTS = {
    # ── HIGH DRAIN — explosive / collision actions ──────────────────
    # Based on GPS tracking: sprints cost ~2x more O2 than jogging
    "sprint":              0.40,   # High-speed run >19.8 km/h  (~12/match winger)
    "high_speed_sprint":   0.65,   # Maximum sprint >25 km/h    (~4/match winger)
    "aerial_duel":         0.55,   # Jumping + landing + contest (~3–8/match CB)
    "tackle_aggressive":   0.50,   # Diving tackle: full body effort
    # ── MEDIUM-HIGH DRAIN — physical contest actions ────────────────
    "tackle_won":          0.35,   # Successful tackle
    "tackle_lost":         0.42,   # Failed tackle + recovery sprint
    "dribble_success":     0.35,   # Beat a man: acceleration + deceleration
    "dribble_fail":        0.45,   # Failed dribble + opponent wins ball
    "press":               0.30,   # Pressing run at opponent
    "press_success":       0.25,   # Won press (positive outcome = less stress)
    "clearance":           0.38,   # Defensive urgency: max effort kick
    "ground_duel":         0.32,   # Physical 50/50 on ground
    # ── MEDIUM DRAIN — technical actions with movement ─────────────
    "carry":               0.20,   # Ball carry with direction change
    "interception":        0.22,   # Anticipation run to intercept
    "shot_attempt":        0.28,   # Strike: explosive movement
    "cross_attempt":       0.18,   # Cross delivery (less explosive than shot)
    "block":               0.25,   # Getting body in way of shot
    "recovery":            0.15,   # Loose ball recovery jog
    # ── LOW DRAIN — technical actions, minimal running ─────────────
    "pass_short":          0.04,   # Simple pass — almost no extra cost
    "pass_long":           0.10,   # Long pass: more leg drive
    "progressive_pass":    0.08,   # Forward pass: slightly more intent
    "through_ball":        0.09,   # Through ball: mental + physical
    "touch":               0.03,   # Receiving/controlling ball
    "standing":            0.00,   # On pitch but not in action (baseline covers it)
    # ── GK SPECIFIC ────────────────────────────────────────────────
    # GKs drain less overall but actions are explosive
    "save":                0.55,   # Diving save: full extension
    "high_claim":          0.45,   # Jumping to claim cross
    "goalline_save":       0.75,   # Last-ditch desperate save: max effort
    # ── INJURY ──────────────────────────────────────────────────────
    "injury_event":        8.00,   # Injury shock drains remaining energy
}

# Position-based base drain rate (per minute just being on pitch)
POSITION_BASE_DRAIN = {
    # Per-minute baseline drain from continuous pitch movement.
    # Calibrated from GPS tracking data (elite football, 90 mins):
    #   GK:  5.5–7km total → baseline covers ~14% of 100% stamina over 90 mins
    #   FB:  11–13.5km     → baseline covers ~29% over 90 mins
    #   CM:  10.5–13km     → baseline covers ~32% over 90 mins
    #   W:   10–13km       → baseline covers ~33% over 90 mins
    # Actions add another 15–25% on top.
    # Target end-of-match stamina: GK 70–82%, FB 45–60%, CM 40–55%, W 38–52%
    "GK":  0.160,   # ~14.4% over 90 mins (mostly walking, alert bursts)
    "CB":  0.257,   # ~23.1% over 90 mins
    "LB":  0.317,   # ~28.5% over 90 mins (up and down constantly)
    "RB":  0.317,   # ~28.5% over 90 mins
    "CDM": 0.323,   # ~29.1% over 90 mins (covers most ground of any position)
    "CM":  0.350,   # ~31.5% over 90 mins
    "CAM": 0.333,   # ~30.0% over 90 mins
    "LW":  0.367,   # ~33.0% over 90 mins (high intensity wide play)
    "RW":  0.367,   # ~33.0% over 90 mins
    "ST":  0.333,   # ~30.0% over 90 mins (explosive bursts, less continuous)
    "CF":  0.333,   # ~30.0% over 90 mins
}

# Team style multipliers on stamina drain
STYLE_DRAIN_MULT = {
    "gegenpressing":        1.35,
    "ultra_attacking":      1.25,
    "high_press":           1.30,
    "attacking":            1.15,
    "wing_play":            1.10,
    "balanced":             1.00,
    "structured_possession": 0.95,
    "tiki_taka":            0.90,
    "fluid_counter":        0.92,
    "defensive":            0.85,
    "park_the_bus":         0.75,
    "ultra_defensive":      0.70,
    "route_one":            0.88,
}


# ─────────────────────────────────────────────
# PLAYER STAMINA STATE
# Live during simulation — one per player
# ─────────────────────────────────────────────

@dataclass
class PlayerStaminaState:
    """
    Tracks a player's stamina minute-by-minute during a match.
    Performance degrades as stamina falls.
    """
    player_name: str
    position: str
    specialties: List[str] = field(default_factory=list)

    # Starting stamina — affected by last match's drain
    starting_stamina: float = 100.0
    current_stamina:  float = 100.0

    # Drain tracking
    total_drained:    float = 0.0
    drain_by_action:  Dict[str, float] = field(default_factory=dict)

    # Injury
    is_injured:       bool  = False
    injury_type:      str   = "none"
    injury_minute:    int   = 0
    injury_severity:  float = 0.0   # 0–100

    # Performance degradation (applied by event chains)
    performance_mult: float = 1.0   # 1.0 = full, 0.75 = 25% worse

    # Sub flag
    sub_requested:    bool  = False
    sub_reason:       Optional[SubReason] = None
    minute_flagged:   int   = 0

    @property
    def stamina_pct(self) -> float:
        return round(self.current_stamina, 1)

    @property
    def is_critical(self) -> bool:
        """Below 20% — player is running on fumes."""
        return self.current_stamina < 20.0

    @property
    def is_struggling(self) -> bool:
        """Below 40% — visibly tiring."""
        return self.current_stamina < 40.0

    @property
    def needs_sub(self) -> bool:
        return self.sub_requested or self.is_injured

    def half_time_recovery(self, recovery_pct: float = 0.18):
        """
        Partial stamina recovery at half-time.

        In real football the 15-minute break does NOT fully reset players,
        but it does provide meaningful acute recovery (~15-20% of max
        stamina). This is applied on top of whatever the player had left
        at the 45th minute.
        """
        recovered = min(recovery_pct * 100.0, 100.0 - self.current_stamina)
        self.current_stamina = min(100.0, self.current_stamina + recovery_pct * 100.0)
        self.total_drained = max(0.0, self.total_drained - recovered)
        self.update_performance_mult()

    def drain(self, action: str, amount: Optional[float] = None):
        """Drain stamina from an action."""
        cost = amount if amount is not None else STAMINA_COSTS.get(action, 0.05)

        # Ironman specialty reduces drain
        if "ironman" in self.specialties:
            cost *= 0.80
        if "engine" in self.specialties:
            cost *= 0.90
        if "workhorse" in self.specialties:
            cost *= 0.85

        # Bug fix: total_drained must represent stamina points ACTUALLY
        # lost (bounded by what the player had left), not the raw cost
        # attempted. Previously this kept adding the full `cost` even
        # after current_stamina had already floored at 0.0 — so a player
        # who bottomed out early in the match would rack up an ever-larger
        # total_drained for the rest of the match despite having no
        # stamina left to lose, which is what produced the >1000%
        # "Total Drained" and 140-day "Recovery Days" figures in exports.
        actual_loss = min(cost, self.current_stamina)
        self.current_stamina = max(0.0, self.current_stamina - cost)
        self.total_drained += actual_loss
        self.drain_by_action[action] = self.drain_by_action.get(action, 0.0) + cost

    def drain_baseline(self, team_style: str, intensity_mult: float = 1.0, weather_mult: float = 1.0):
        """
        Per-minute baseline drain from just being on pitch.

        `intensity_mult` folds the team's intensity setting directly into
        THIS minute's marginal drain. Previously the caller (MatchEngine)
        applied intensity separately, AFTER this call, by reading back
        `drain_by_action["standing"]` — which is a CUMULATIVE total, not
        this minute's delta — and topping up stamina loss by a fraction of
        the entire match-to-date total, every single minute. That
        compounded: each minute's top-up made the cumulative total bigger,
        which made the next minute's top-up bigger still. Folding the
        multiplier in here means intensity affects only the actual amount
        drained THIS minute, once, with no runaway feedback loop.
        """
        base = POSITION_BASE_DRAIN.get(self.position, 0.05)
        style_mult = STYLE_DRAIN_MULT.get(team_style, 1.0)
        self.drain("standing", base * style_mult * intensity_mult * weather_mult)

    def update_performance_mult(self):
        """
        As stamina falls, performance degrades progressively.
        100%→60%: full performance
        60%→40%:  slight degradation (-5% to -12%)
        40%→20%:  noticeable degradation (-12% to -25%)
        Below 20%: severe degradation (-25% to -40%)
        """
        s = self.current_stamina
        if s >= 60:
            self.performance_mult = 1.0
        elif s >= 40:
            # Linear from 1.0 to 0.88
            self.performance_mult = 0.88 + (s - 40) / 20 * 0.12
        elif s >= 20:
            # Linear from 0.88 to 0.75
            self.performance_mult = 0.75 + (s - 20) / 20 * 0.13
        else:
            # Linear from 0.75 to 0.60
            self.performance_mult = 0.60 + (s / 20) * 0.15
        self.performance_mult = round(self.performance_mult, 3)

    def roll_injury(self, minute: int, action: str) -> bool:
        """
        Roll for in-match injury.
        Higher drain + physical actions + low stamina = more likely.
        Returns True if injured.
        """
        if self.is_injured:
            return False

        # Base risk by action
        action_risk = {
            "tackle_aggressive": 0.008,
            "aerial_duel":       0.006,
            "tackle_won":        0.004,
            "tackle_lost":       0.005,
            "sprint":            0.003,
            "dribble_fail":      0.004,
            "high_speed_sprint": 0.005,
        }.get(action, 0.001)

        # Fatigue multiplier — low stamina = much higher injury risk
        fatigue_mult = 1.0
        if self.current_stamina < 20:
            fatigue_mult = 3.5
        elif self.current_stamina < 35:
            fatigue_mult = 2.2
        elif self.current_stamina < 50:
            fatigue_mult = 1.5

        # Injury-prone specialty
        if "injury_prone" in self.specialties:
            fatigue_mult *= 1.6

        final_risk = action_risk * fatigue_mult

        if random.random() < final_risk:
            self.is_injured = True
            self.injury_minute = minute
            self.injury_severity = random.uniform(15, 85)

            # Severity determines type
            if self.injury_severity < 25:
                self.injury_type = "knock"        # 0–1 matches out
            elif self.injury_severity < 50:
                self.injury_type = "muscular"     # 1–3 matches out
            elif self.injury_severity < 75:
                self.injury_type = "ligament"     # 3–8 matches out
            else:
                self.injury_type = "fracture"     # 6–12 matches out

            self.sub_requested = True
            self.sub_reason = SubReason.INJURY
            self.minute_flagged = minute
            return True
        return False

    def check_stamina_sub(self, minute: int, manager_stubbornness: float = 0.3,
                         urgency: float = 0.5):
        """
        Should the manager sub this player for stamina?
        manager_stubbornness: 0.0=immediate, 1.0=never subs for stamina
        urgency: Phase 6 brain-manager sub urgency (0.5=neutral, 1.0=eager,
                 0.0=very patient). The default of 0.5 gives factor
                 (0.5 + 0.5) = 1.0 so flag-off base probs are unchanged.

        Real football: some players finish on fumes. Some managers leave
        them on because subs are used or they trust the player.
        """
        if self.sub_requested or self.is_injured:
            return

        # Phase 6 — urgency modulates the base sub probability from the
        # brain-manager's personality. At default urgency (0.5) the factor
        # is 1.0, so the probabilities below match the static path exactly.
        _urgency_factor = 0.5 + float(urgency)

        if self.is_critical and minute >= 55:
            # Critical stamina — almost always subbed (unless manager is stubborn
            # or it's too early)
            sub_prob = 0.85 * (1 - manager_stubbornness) * _urgency_factor
            if random.random() < sub_prob:
                self.sub_requested = True
                self.sub_reason = SubReason.STAMINA
                self.minute_flagged = minute

        elif self.is_struggling and minute >= 70:
            # Struggling late — reasonable chance of sub
            sub_prob = 0.45 * (1 - manager_stubbornness) * _urgency_factor
            if random.random() < sub_prob:
                self.sub_requested = True
                self.sub_reason = SubReason.STAMINA
                self.minute_flagged = minute

    def recovery_days_needed(self) -> float:
        """How many days until fully recovered."""
        base = self.total_drained / 10.0
        if self.is_injured:
            severity_days = {
                "knock":     random.uniform(0, 1),
                "muscular":  random.uniform(7, 21),
                "ligament":  random.uniform(21, 56),
                "fracture":  random.uniform(42, 84),
            }.get(self.injury_type, 7)
            return round(base + severity_days, 1)
        return round(max(1.0, base), 1)

    def matches_out(self) -> int:
        """Matches missed from injury."""
        if not self.is_injured:
            return 0
        days = self.recovery_days_needed()
        return max(0, int(days / 7))   # Approx 1 match per week


# Roles whose player comes off first when chasing — the defensive spine,
# which is the group a manager will least mind losing while chasing.
DEFENSIVE_SPINE = ["CB", "CDM"]

# Per-minute chance of a scoreline-driven change once the quality gate is
# open. Unchanged from the previous 0.08 — see the comment at section 4.
SUB_GAME_STATE_RATE = 0.08

# Earliest minute a leading team makes a protective change. Previously 75.
# Rey et al. found coaches act earlier when level or behind and later when
# ahead, so this stays later than the chasing path but no longer sits in
# the last quarter of the match where it can only ever be a token gesture.
SUB_PROTECT_LEAD_MINUTE = 60

# How much stamina a substitute recovers per minute spent on the bench.
# A standing man recovers far slower than a seated one; 0.55/min means a
# 45-minute stay on the bench is worth about 25 points of tank. This is what
# makes "rest him" and "use him" different decisions, which the previous
# hardcoded 100.0 made indistinguishable.
SUB_SEATED_RECOVERY_PER_MIN = 0.55


def _position_of(player) -> str:
    """Position label for a bench or on-pitch object, tolerating wrappers.

    PlayerProfile carries `.position` and `.dna.position`; some fixtures
    carry only one. Returning "CM" rather than "" on total failure matters,
    because "" matches nothing in ADJACENT_POS and a player whose position
    cannot be read must still be selectable as a last resort — otherwise a
    fixture that omits positions produces a bench nobody can draw from, and
    the failure looks like "no substitutes available".
    """
    for src in (player, getattr(player, "dna", None)):
        if src is None:
            continue
        pos = getattr(src, "position", None)
        if isinstance(pos, str) and pos:
            # Live positions are numbered ("CB1", "CM2"); the role family is
            # what every position table in this module is keyed on. This is
            # the same trap that made a same-side-passing rule match nothing
            # in attacking_matrix because it tested "CB" against "CB1".
            base = pos.rstrip("0123456789").upper()
            return base or pos.upper()
    return "CM"


# Adjacent positions — who can plausibly fill a role. Identical to the table
# the previous `_pick_sub` had inline; hoisted to module scope so the
# ranking helpers and any test share one definition.
ADJACENT_POS: Dict[str, List[str]] = {
    "ST":  ["CF", "LW", "RW", "CAM"],
    "CF":  ["ST", "CAM", "LW", "RW"],
    "LW":  ["RW", "CAM", "ST", "LB"],
    "RW":  ["LW", "CAM", "ST", "RB"],
    "CAM": ["CM", "LW", "RW", "ST"],
    "CM":  ["CAM", "CDM", "LW", "RW"],
    "CDM": ["CM", "CB"],
    "LB":  ["RB", "CB", "LW"],
    "RB":  ["LB", "CB", "RW"],
    "CB":  ["CDM", "LB", "RB"],
    "GK":  ["GK"],
}

# Roles that count as "an attacker" for a chasing change.
ATTACKING_POS = ["ST", "CF", "LW", "RW", "CAM"]


# ─────────────────────────────────────────────
# WHO IS WORTH BRINGING ON  (added 2026-10-14)
# ─────────────────────────────────────────────
#
# BEFORE THIS SECTION every "who comes on" helper returned the FIRST unused
# player in bench-list order. Bench order is (transitively) market value, so
# the bench was used top-down by accident, and nothing in this module read
# `overall_rating`, `form.confidence`, `recent_ratings`, `age` or `soul` even
# ONCE — all zero occurrences. The manager was picking blind. This section is
# the signal those picks now sort on.
#
# THE RESEARCH, and what it changes:
#
# * Wharton (2024, "the value of substitution", West Virginia) measured the
#   marginal cost of a substitution as a loss of ten points of quality and
#   found the answer is NOT symmetric. LEADING by one, a manager may accept
#   ten points of OFFENSIVE quality from minute 47 but ten points of
#   DEFENSIVE quality is NEVER advisable — protecting a lead is worth more
#   than any attacking upgrade. TRAILING by one, the mirror holds: ten
#   points of defensive quality becomes justifiable at only minute 32, while
#   giving up ten points of ATTACKING quality is never advisable.
#
#   That is why `sub_quality` is split into an attacking and a defensive
#   half. A single blended number cannot express "this change costs me
#   something I can afford" — it can only express "is this man better",
#   which is the question the research says managers do NOT ask.
#
# * Silva & Swartz (2016, "Choosing your players", Journal of Quantitative
#   Analysis in Sports) fitted Bayesian logistic regressions over 4,226
#   substitutions across ten seasons and found NO substitution minute beats
#   any other. Their conclusion is literally "coaches should substitute
#   whenever they notice tired players", and they explicitly reject the
#   clock-driven timings that previous literature proposed. So the minute
#   thresholds below are GATES on when a check happens, never the reason a
#   change happens: a player must be in genuine difficulty, and the gate
#   only decides whether anyone is looking yet.
#
# * Rey, Dellal & Gentilini (2015, 677 UCL substitutions) found coaches
#   substitute EARLIER when level or behind and LATER when ahead, and use
#   offensive substitutions more when behind. The scoreline weights below
#   encode that as an explicit asymmetry rather than three unrelated ifs.
#
# * Kwest's analysis of the first five-sub Premier League season found the
#   mode is three to four changes, managers overwhelmingly believe they have
#   already fielded their best XI, and Chelsea's three managers — the ones
#   substituting most frequently — are read as not knowing who their best XI
#   was. Frequent, indiscriminate changing is a known failure mode, which is
#   why `SUB_GAME_STATE_RATE` is deliberately low (8%/min) and unchanged
#   from the previous value: this change improves WHICH player is chosen and
#   WHEN a struggling one is noticed, not the appetite for change.


def _get(dna, group: str, field: str, default: float = 50.0) -> float:
    """Read one attribute group off a PlayerDNA without exploding on mocks.

    The bench objects are PlayerProfile wrappers, not PlayerDNA, and several
    tests build lightweight stand-ins. `_get` walks the real chain and falls
    back to the position-average so a missing attribute degrades to neutral
    rather than silently scoring the player zero — a zero would make an
    incomplete fixture look like a terrible player, which is the documented
    `get_position` returns the centre spot trap in a different guise.
    """
    obj = getattr(dna, group, None)
    if obj is None:
        return default
    val = getattr(obj, field, None)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Attacking weight map: attribute -> weight. Sums to 1.0 before the form and
# soul adjustments below.
SUB_ATTACK_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("technical", "finishing"):     0.16,
    ("technical", "dribbling"):     0.13,
    ("technical", "first_touch"):   0.11,
    ("technical", "crossing"):      0.09,
    ("technical", "long_shots"):    0.05,
    ("technical", "ball_control"):  0.07,
    ("passing", "through_balls"):   0.11,
    ("passing", "short_passing"):   0.09,
    ("mental", "vision"):           0.12,
    ("mental", "composure"):        0.07,
}

# Defensive weight map. Positions outside the defensive spine score low here
# on purpose: a winger brought on to replace a left back is a real downgrade
# in this column, and the scoreline rule is supposed to be able to see that.
SUB_DEFEND_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("defending", "tackling"):       0.16,
    ("defending", "marking"):        0.13,
    ("defending", "interceptions"):  0.11,
    ("defending", "blocking"):       0.08,
    ("defending", "clearances"):     0.09,
    ("mental", "positioning"):       0.13,
    ("mental", "anticipation"):      0.10,
    ("mental", "concentration"):     0.07,
    ("physical", "strength"):        0.08,
    ("mental", "aggression"):        0.05,
}

# Goalkeeping quality, used only when the slot being filled is the keeper's.
SUB_GK_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("gk_attrs", "reflexes"):     0.30,
    ("gk_attrs", "diving"):       0.25,
    ("gk_attrs", "handling"):     0.18,
    ("gk_attrs", "positioning_gk"): 0.15,
    ("gk_attrs", "kicking"):      0.07,
    ("gk_attrs", "sweeping"):     0.05,
}


def _weighted(player, weights: Dict[Tuple[str, str], float]) -> float:
    dna = getattr(player, "dna", None)
    if dna is None:
        return 50.0
    total = 0.0
    norm = 0.0
    for (group, field), w in weights.items():
        total += _get(dna, group, field) * w
        norm += w
    return total / norm if norm else 50.0


def sub_attacking_quality(player) -> float:
    """How much ATTACKING quality this player is worth, 0-100.

    Scaled by the player's own position-appropriate overall so a 75-rated
    winger does not out-score a 75-rated centre-back purely because the
    attribute map favours forwards — the two numbers answer different
    questions and must stay comparable for the scoreline rule to be able to
    say "this costs me attack".
    """
    base = _weighted(player, SUB_ATTACK_WEIGHTS)
    ovr = _overall_of(player)
    return base * (0.55 + 0.45 * (ovr / 100.0))


def sub_defensive_quality(player) -> float:
    """How much DEFENSIVE quality this player is worth, 0-100. See above."""
    base = _weighted(player, SUB_DEFEND_WEIGHTS)
    ovr = _overall_of(player)
    return base * (0.55 + 0.45 * (ovr / 100.0))


def _overall_of(player) -> float:
    """PlayerDNA.overall_rating, or a neutral 60 when it is unavailable.

    60 is the pool mean, so an unreadable rating makes a player neither
    attractive nor repulsive; it does not make them a liability. Using 0
    would make every incomplete fixture look like a garbage player and is
    the same class of bug as `get_position` returning the centre spot for a
    player it has no state for.
    """
    dna = getattr(player, "dna", None)
    if dna is None:
        return 60.0
    ovr = getattr(dna, "overall_rating", None)
    if ovr is None:
        # Some fixtures carry a plain `rating` on the wrapper instead.
        ovr = getattr(player, "overall_rating", None) or getattr(player, "rating", None)
    try:
        val = float(ovr)
    except (TypeError, ValueError):
        return 60.0
    return max(0.0, min(100.0, val))


def _form_of(player) -> float:
    """Form confidence + recent match ratings folded into a 0-100 score.

    `PlayerFormState.confidence` is already 0-100 (it is the thing the
    exporter prints and the season manager rehydrates). `recent_ratings` is
    a short list of post-match ratings; its mean is on the same scale.
    """
    dna = getattr(player, "dna", None)
    form = getattr(dna, "form", None) if dna is not None else None
    if form is None:
        form = getattr(player, "form", None)
    if form is None:
        return 60.0

    conf = getattr(form, "confidence", None)
    try:
        conf = float(conf) if conf is not None else 60.0
    except (TypeError, ValueError):
        conf = 60.0

    ratings = getattr(form, "recent_ratings", None) or []
    ratings = [float(r) for r in ratings if isinstance(r, (int, float))]
    recent = (sum(ratings[-5:]) / len(ratings[-5:])) if ratings else conf

    return max(0.0, min(100.0, 0.6 * conf + 0.4 * recent))


def sub_quality(player, controller=None) -> float:
    """The single score the bench is ranked on. Higher is better.

    Blends four things, each for a stated reason:

      overall   the position-appropriate ability we already trust everywhere
      form      confidence and recent ratings — a player on a good run is
                genuinely a better substitution, and the engine already
                persists this per matchday
      soul      greatness_coefficient, so a soul who is sitting on the bench
                is never overlooked for a merely-fit ordinary player. This
                is the same signal `_pick_superstars` uses.
      freshness how much he has left in the tank. A substitute who has been
                on the bench all match has not recovered; his stamina state
                starts at whatever the pre-match projection gave him, not at
                100 (see `_make_sub`, which previously hardcoded 100.0).

    Weighting: ability dominates because it is the thing the user asked to
    optimise ("more solely the best players in a team"), but form and
    freshness each carry enough weight to break ties between two players of
    similar rating — which is what a real manager is actually doing.
    """
    ovr = _overall_of(player)
    form = _form_of(player)
    soul = _soul_coefficient(player)

    if controller is not None:
        st = controller.stamina.get(getattr(player, "name", ""), None)
        stamina = float(st.current_stamina) if st is not None else 100.0
    else:
        stamina = 100.0

    return (
        0.55 * ovr
        + 0.20 * form
        + 0.15 * (min(100.0, 100.0 * soul))
        + 0.10 * stamina
    )


def _soul_coefficient(player) -> float:
    """Soul greatness as 0..1, or 0.0 for an ordinary player.

    Souls carry `greatness_coefficient` around 1.0-1.55 and ordinary players
    have no soul at all, so an ordinary player must score 0 here rather
    than 1.0 — otherwise the term would silently GIFT 15 points to
    everyone and reduce the weighting of every real signal.
    """
    dna = getattr(player, "dna", None)
    soul = getattr(dna, "soul", None) if dna is not None else None
    g = getattr(soul, "greatness_coefficient", None) if soul is not None else None
    try:
        return max(0.0, min(1.0, float(g)))
    except (TypeError, ValueError):
        return 0.0


def sub_gk_quality(player) -> float:
    """Keeper quality, 0-100. Only used when filling a keeper's slot."""
    return _weighted(player, SUB_GK_WEIGHTS)


# ── THE SCORELINE RULE (Wharton 2024) ────────────────────────────────
#
# Thresholds are the minutes from the Wharton study at which a ten-point
# loss in that quality becomes acceptable. Ahead, the defensive column is
# gated at 90' — effectively never, because a match ends at 90 — which is
# the study's actual finding, not a rounding choice. Behind, the offensive
# column is gated the same way.

# A threshold no minute can reach. Used for the two columns Wharton found
# to be NEVER acceptable, and a sentinel rather than the literal 90 so the
# rule does not quietly switch itself on in minute 90 of a 90-minute match
# (which is a real minute — stoppage time). Encoding "never" as a number
# that happens to be the match length makes the guarantee a coincidence.
SUB_NEVER = 10 ** 6

SUB_ACCEPT_OFFENSIVE_LOSS_LEADING_MIN = 47
SUB_ACCEPT_DEFENSIVE_LOSS_LEADING_MIN  = SUB_NEVER   # never, per Wharton
SUB_ACCEPT_OFFENSIVE_LOSS_TRAILING_MIN = SUB_NEVER   # never, per Wharton
SUB_ACCEPT_DEFENSIVE_LOSS_TRAILING_MIN  = 32

# The reference loss the study is measured in: TEN points of quality.
# A threshold minute means "from this minute, losing ten points is
# acceptable" — so the budget must EQUAL the reference AT the threshold, not
# start from zero there. An earlier version of this ramped from zero, which
# made the 47' threshold actually mean 67' and did not implement the finding
# it claimed to: a one-goal lead with a 10-point attacking drop would have
# been refused from 47' all the way to 66'.
SUB_REFERENCE_QUALITY_LOSS = 10.0

# Additional quality tolerated per minute BEYOND the threshold.
SUB_QUALITY_TOLERANCE_PER_MIN = 0.5


def _quality_budget(threshold_minute: int, minute: int) -> float:
    """Points of quality a change may cost, at `minute`, given a threshold.

    Zero before the threshold — nothing is affordable yet. The reference
    loss AT the threshold, growing thereafter. A `SUB_NEVER` threshold is
    therefore zero at every minute a match can contain, which is exactly
    what "never advisable" means in arithmetic.
    """
    if threshold_minute >= SUB_NEVER or minute < threshold_minute:
        return 0.0
    return SUB_REFERENCE_QUALITY_LOSS + SUB_QUALITY_TOLERANCE_PER_MIN * (
        minute - threshold_minute
    )


def acceptable_quality_loss(
    team_goal_diff: int, minute: int,
    on_attack: float, off_attack: float,
    on_defence: float, off_defence: float,
) -> Tuple[bool, str]:
    """Should this substitution go ahead, on quality grounds alone?

    `on_*` is the incoming player's quality, `off_*` the outgoing player's.
    Returns (allowed, reason). A downgrade is tolerated once the relevant
    threshold minute has passed AND the loss is small enough for the minutes
    elapsed; an UPGRADE is always allowed, because there is no research
    finding that says a team should refuse to improve itself.

    `on_attack`/`on_defence` are the two columns from
    `sub_attacking_quality` / `sub_defensive_quality`. Passing them in keeps
    this function pure and trivially testable — it is pure arithmetic over
    five numbers and is where the actual decision rule lives.
    """
    d_atk = off_attack - on_attack
    d_def = off_defence - on_defence

    if team_goal_diff > 0:
        # LEADING. Protect it. Attacking quality is spendable after ~47'.
        atk_min = SUB_ACCEPT_OFFENSIVE_LOSS_LEADING_MIN
        def_min = SUB_ACCEPT_DEFENSIVE_LOSS_LEADING_MIN
    elif team_goal_diff < 0:
        # TRAILING. Chasing. The mirror.
        atk_min = SUB_ACCEPT_OFFENSIVE_LOSS_TRAILING_MIN
        def_min = SUB_ACCEPT_DEFENSIVE_LOSS_TRAILING_MIN
    else:
        # LEVEL. Balanced, and slightly earlier than leading: with no
        # cushion a team should still be willing to freshen up.
        atk_min = 30
        def_min = 30

    budget_atk = _quality_budget(atk_min, minute)
    budget_def = _quality_budget(def_min, minute)

    if d_atk > budget_atk:
        return False, (f"attacking downgrade {d_atk:.1f} exceeds budget "
                       f"{budget_atk:.1f} at {minute}'")
    if d_def > budget_def:
        return False, (f"defensive downgrade {d_def:.1f} exceeds budget "
                       f"{budget_def:.1f} at {minute}'")
    return True, "within quality budget"


# ─────────────────────────────────────────────
# SUBSTITUTION CONTROLLER
# Manages the live sub logic during simulation
# ─────────────────────────────────────────────

# ── WHEN DOES A MANAGER ACTUALLY DECIDE TO CHANGE SOMETHING? ────────
#
# This replaced an 8%-per-minute coin flip. Measured before removal, over
# 20,000 simulated gate chains (_diag_sub_trigger.py): a trailing team
# failed to make a single game-state change in only 3.5% of matches and a
# leading one in 9.9%. So the roll was never really a BLOCKER — it was a
# timer for WHEN and HOW MANY. I had described it as deciding "whether",
# which was wrong.
#
# What it actually was is worse than a weak gate: it was a decision the
# manager did not make. He rolled, and then took whatever the roll gave him.
# A real manager changes something when there is a REASON — a tired man, a
# better man on the bench, a scoreline that demands a different shape. So
# the trigger is now computed from those things.
#
#   upgrade   how much better is the best available sub than the worst man
#             he could replace? This is the "your XI was wrong" signal.
#   strain    how close is anyone to asking to come off?
#   pressure  trailing and late is the strongest pull in football; leading
#             and late wants consolidation, not adventure.
#
# All three are state, so the trigger is DETERMINISTIC. That matters here:
# a coin flip per minute made the decision sequence unreproducible for
# reasons that had nothing to do with football.

# How much better the best available sub must be before the bench even
# counts as an argument for changing. Below this a manager is just fidgeting:
# Kwest found the managers who substitute most are the ones who do not know
# their best XI, so a marginal upgrade must NOT clear the bar by itself.
GAME_STATE_UPGRADE_MIN = 6.0

# Points of additional improvement that take the upgrade signal to its
# maximum. So a 16-point better man on the bench is a decisive argument; a
# 7-point one is a rounding error.
GAME_STATE_UPGRADE_SPAN = 10.0

# Stamina at which a player counts as genuinely struggling.
GAME_STATE_STRAIN_STAMINA = 45.0

# Pressure ramps from zero at this minute to full at 90.
GAME_STATE_PRESSURE_START = 55

# Total the signals must reach before the manager acts.
GAME_STATE_THRESHOLD = 0.55


def game_state_pressure(goal_diff: int, minute: int) -> float:
    """How much this scoreline wants a change, 0.0 - 1.0.

    Trailing is the strongest pull in football and Rey et al. found coaches
    substitute markedly EARLIER when level or behind than when ahead, so
    the ramp starts sooner for those. Leading wants the opposite: the later
    the game, the more a manager consolidates.
    """
    if minute < GAME_STATE_PRESSURE_START:
        return 0.0
    span = max(1.0, 90.0 - GAME_STATE_PRESSURE_START)
    ramp = min(1.0, (minute - GAME_STATE_PRESSURE_START) / span)

    if goal_diff < 0:
        # Behind. Full pull, and faster than level.
        return min(1.0, 0.65 + 0.35 * ramp)
    if goal_diff == 0:
        # Level. Wants change, but less urgently than chasing.
        return 0.45 + 0.35 * ramp
    # Ahead. Consolidate. Wants change only to shore up tired legs.
    return 0.20 + 0.25 * ramp


class SubstitutionController:
    """
    Manages all substitutions during a match.

    Integrates with MatchEngine.simulate() — called each minute
    to check whether any sub should happen.

    Sub rules: 5 substitutions in 3 WINDOWS (IFAB Laws of the Game 2026/27,
    which the Premier League adopted in 2022). A window is a stoppage in
    play; making several changes at the SAME stoppage costs one window,
    which is why the controller batches rather than dribbling changes out
    one at a time. Half-time is a free window and does not count toward the
    three.

    The previous limit was a flat 3 subs with no window concept at all.
    """

    MAX_SUBS = 5            # IFAB maximum, Premier League standard since 2022
    MAX_WINDOWS = 3         # ...but only three times you may actually do it
    HALF_TIME_MINUTE = 45   # a free window; not counted against MAX_WINDOWS

    # Gateway minute for tactical/quality changes. Injury sub is exempt and
    # always allowed. Silva & Swartz found no magic minute; this exists so
    # the check is not spent on a team that is 2-0 up at 12 minutes.
    QUALITY_GATE_MINUTE = 46

    def __init__(
        self,
        home_team: str,
        away_team: str,
        home_subs_bench: List,       # PlayerProfile list
        away_subs_bench: List,
        home_style: str = "balanced",
        away_style: str = "balanced",
        manager_stubbornness: float = 0.3,  # 0=rational, 1=stubborn
    ):
        self.home_team = home_team
        self.away_team = away_team
        self.bench: Dict[str, List] = {
            home_team: list(home_subs_bench),
            away_team: list(away_subs_bench),
        }
        self.styles: Dict[str, str] = {
            home_team: home_style,
            away_team: away_style,
        }
        self.stubbornness = manager_stubbornness

        # Optional per-team manager stubbornness override (from ManagerProfile).
        # Keys = team names; value = stubbornness (0 subs quick, 1 stubborn).
        # Falls back to the flat `self.stubbornness` when a team isn't keyed.
        self.stubbornness_by_team: Dict[str, float] = {}

        # Phase 6 — per-team sub urgency from the brain-manager's live
        # decision (Manager._current_urgency, pushed each minute by the
        # engine when USE_MANAGER_BRAIN is on). 0.5 is neutral (factor 1.0);
        # >0.5 makes a team sub faster, <0.5 more cautiously.
        self.urgency_by_team: Dict[str, float] = {}

        # Track subs made
        self.subs_made: Dict[str, int]       = {home_team: 0, away_team: 0}
        self.subs_log:  List[Dict]            = []

        # ── IFAB substitution WINDOWS ────────────────────────────
        # `windows_used[team]` counts DISTINCT stoppages at which that team
        # changed a player. Batching several changes into one minute costs
        # one window, which is the mechanic that makes managers plan a
        # double sub rather than drip-feeding changes. The half-time minute
        # is tracked separately because it is free under the Laws.
        self.windows_used: Dict[str, int] = {home_team: 0, away_team: 0}
        # Injury substitutions are not gated on the window allowance (a team
        # cannot be required to play on a broken leg), so this counts how
        # many changes went past the three-window limit. The IFAB rule would
        # forbid it; refusing the change is worse football, so the
        # alternative is to permit it and say so in the log rather than to
        # silently report "4 windows" against a limit of three.
        self.windows_exceeded: Dict[str, int] = {home_team: 0, away_team: 0}

        # Does the scoreline rule ever actually BIND? A rule that silently
        # never rejects anything is indistinguishable from a rule that was
        # never wired, so the counters are kept and the probe reports them.
        self.quality_gate_stats: Dict[str, int] = {
            "considered": 0, "accepted": 0, "rejected": 0, "exhausted": 0,
        }

        # Same reasoning for the game-state trigger, which replaced an
        # 8%-per-minute coin flip. "How often did the manager consider a
        # change and decline" is the only way to tell a decision from a
        # slot machine.
        self.game_state_stats: Dict[str, int] = {
            "considered": 0, "taken": 0, "declined": 0,
        }
        # Minute at which each team last spent a window. Used to decide
        # whether the changes being proposed right now are a BATCH (same
        # stoppage, free) or a NEW STOPPAGE (costs a window).
        self._last_window_minute: Dict[str, int] = {home_team: -99, away_team: -99}

        # Stamina states for all active players
        self.stamina: Dict[str, PlayerStaminaState] = {}

        # Tactical sub schedule from run_match.py
        # {player_name: minute} — pre-planned tactical subs
        self.tactical_schedule: Dict[str, int] = {}
        self.weather_drain_mult: float = 1.0

    def set_weather(self, weather):
        """Configure weather multiplier for stamina drain."""
        try:
            from weather_physics import WeatherPhysics
            self.weather_drain_mult = WeatherPhysics.stamina_drain_mult(weather)
        except Exception:
            self.weather_drain_mult = 1.0

    def set_manager_stubbornness(self, team: str, stubbornness: float):
        """Set an individual manager's substitution patience for one team.
        Call this to make each side's rotation behaviour depend on its own
        manager instead of one flat constant."""
        self.stubbornness_by_team[team] = max(0.0, min(1.0, stubbornness))

    def stubbornness_for(self, team: str) -> float:
        """Resolve which substitution-patience value applies to a team."""
        return self.stubbornness_by_team.get(team, self.stubbornness)

    def set_manager_urgency(self, team: str, urgency: float):
        """Phase 6 — push the brain-manager's live sub urgency for one team.

        The engine calls this each minute when ``USE_MANAGER_BRAIN`` is on,
        reading ``Manager._current_urgency`` (a 0..1 value filtered through
        the mind's empathy/youth_trust in ManagerMind.filter_sub_urgency).
        """
        self.urgency_by_team[team] = float(max(0.0, min(1.0, urgency)))

    def urgency_for(self, team: str) -> float:
        """Resolve which urgency value applies to a team.

        Returns the brain-manager urgency when the flag is on, 0.5 (neutral)
        when no brain-manager is wired — factor (0.5 + 0.5) = 1.0 so the
        base sub probability is unchanged.
        """
        return self.urgency_by_team.get(team, 0.5)

    def register_player(self, player, starting_stamina: float = 100.0):
        """Register a player's stamina state at kick-off."""
        name = getattr(player, "name", str(player))
        pos  = getattr(player, "position",
               getattr(getattr(player, "dna", None), "position", "CM") if hasattr(player, "dna") else "CM")
        specs = (getattr(player, "dna", None) and
                 getattr(player.dna, "specialties", [])) or []

        self.stamina[name] = PlayerStaminaState(
            player_name=name,
            position=pos,
            specialties=specs,
            starting_stamina=starting_stamina,
            current_stamina=starting_stamina,
        )

    def register_tactical_schedule(self, schedule: Dict[str, int]):
        """
        Register pre-planned tactical subs.
        schedule = {"Eli Dago": 68, "Calvin Pryce": 75}
        Player name → minute they come ON.
        """
        self.tactical_schedule = schedule

    def process_action(self, player_name: str, action: str,
                       team: str, minute: int, is_secondary: bool = False,
                       drain_mult: float = 1.0):
        """
        Called after every event in the timeline.
        Drains stamina (for the specific action) and rolls injury.

        `drain_mult` lets callers apply a per-event fatigue tax on top of
        the base STAMINA_COSTS rate (the pressing-profile tax is the main
        consumer — a gegenpress team pays 1.35x per press).

        Bug fix: this used to ALSO call state.drain_baseline() here, on
        every single discrete action. drain_baseline() represents
        continuous, ambient "just being on the pitch" drain and is
        already applied exactly once per minute by MatchEngine's own
        per-minute loop. Re-applying it here meant a heavily-involved
        player (150-250+ discrete actions/match) got the baseline cost
        charged 150-250+ EXTRA times on top of the legitimate ~90-99
        once-per-minute charges — this, not the per-action STAMINA_COSTS
        rates, was the real reason players were draining 3-4x faster than
        the design's own documented targets (e.g. winger 38-52% ending
        stamina) even after the separate total_drained/intensity bug was
        fixed.
        """
        state = self.stamina.get(player_name)
        if state is None or state.is_injured:
            return
        
        # Reduce drain for secondary actors
        effective_mult = drain_mult * getattr(self, "weather_drain_mult", 1.0)
        if is_secondary:
            # Get original cost, halve it, then apply any fatigue tax
            original_cost = STAMINA_COSTS.get(action, 0.05)
            state.drain(action, original_cost * 0.5 * effective_mult)
        else:
            state.drain(action, STAMINA_COSTS.get(action, 0.05) * effective_mult)
        
        state.update_performance_mult()
        
        # Roll for injury on physical actions (only for primary actors)
        if not is_secondary and action in (
            "tackle_aggressive", "aerial_duel", "tackle_won",
            "tackle_lost", "sprint", "high_speed_sprint", "dribble_fail"
        ):
            state.roll_injury(minute, action)
        
        # Check if stamina sub needed
        if not state.sub_requested:
            state.check_stamina_sub(
                minute, self.stubbornness_for(team), self.urgency_for(team)
            )

    def process_minute(self, minute: int, active_players: Dict[str, List],
                       game_state_home: int, game_state_away: int) -> List[Dict]:
        """
        Called once per minute by MatchEngine.
        Returns list of substitution dicts to execute.

        game_state_home/away: goal difference from that team's perspective
        """
        subs_to_make = []

        for team, players in active_players.items():
            if self.subs_made[team] >= self.MAX_SUBS:
                continue

            bench = self.bench.get(team, [])
            available_bench = [p for p in bench if not getattr(p, "_used_as_sub", False)]

            if not available_bench:
                continue

            # Per-team list. The previous code tested `if subs_to_make` — the
            # ACCUMULATED list across both teams — at three points, so the
            # instant the home team made a single change the away team was
            # skipped entirely for that minute, with no reference to its own
            # subs remaining, its own injuries, or the scoreline. One team's
            # substitution silently consumed the other's opportunity.
            team_subs: List[Dict] = []

            goal_diff = game_state_home if team == self.home_team else game_state_away

            # ── 1. INJURY SUBS ─────────────────────────────
            # Forced, never blocked: you do not leave a broken player on.
            # quality_gate=False because the scoreline rule has nothing to
            # say when the alternative is playing on a broken leg.
            for player in players:
                name = getattr(player, "name", "")
                state = self.stamina.get(name)
                if state and state.is_injured and not getattr(player, "_subbed_off", False):
                    sub_player = self._pick_sub(
                        available_bench, player, "injury",
                        minute=minute, team=team, goal_diff=goal_diff,
                        quality_gate=False,
                    )
                    if sub_player:
                        team_subs.append(self._make_sub(
                            team, player, sub_player, minute,
                            SubReason.INJURY, available_bench
                        ))
                        break   # one at a time

            if not team_subs:
                # ── 2. TACTICAL PRE-PLANNED SUBS ─────────────
                # Batched rather than rate-limited: the window model
                # already stops a team spending more than three stoppages,
                # so the old per-minute MAX_SUBS break is redundant and
                # was in fact the thing that silently dropped scheduled
                # changes when three players shared a minute.
                for player in players:
                    if not self.window_available(team, minute):
                        break
                    name = getattr(player, "name", "")
                    if (name in self.tactical_schedule and
                            minute >= self.tactical_schedule[name] and
                            not getattr(player, "_subbed_off", False)):
                        sub_player = self._find_tactical_sub(available_bench, player)
                        if sub_player:
                            team_subs.append(self._make_sub(
                                team, player, sub_player, minute,
                                SubReason.TACTICAL, available_bench
                            ))
                            if not self.window_available(team, minute):
                                break

            # ── 3. STAMINA EMERGENCY SUBS ─────────────────────
            # Silva & Swartz: substitute the moment you notice a struggling
            # player, not on a schedule. The gate below only decides whether
            # anyone is looking yet; the trigger is the player, not the
            # clock.
            if not team_subs and minute >= self.QUALITY_GATE_MINUTE:
                for player in sorted(
                    players,
                    key=lambda p: self.stamina.get(
                        getattr(p, "name", ""), PlayerStaminaState("?", "CM")
                    ).current_stamina
                ):
                    name = getattr(player, "name", "")
                    state = self.stamina.get(name)
                    if (state and state.sub_requested and
                            state.sub_reason == SubReason.STAMINA and
                            not getattr(player, "_subbed_off", False)):
                        if not self.window_available(team, minute):
                            break
                        sub_player = self._pick_sub(
                            available_bench, player, "stamina",
                            minute=minute, team=team, goal_diff=goal_diff,
                        )
                        if sub_player:
                            team_subs.append(self._make_sub(
                                team, player, sub_player, minute,
                                SubReason.STAMINA, available_bench
                            ))
                            break

            # A CONSIDERED CHANGE, not a coin flip. See the comment on
            # `game_state_pressure` above: the previous 8%/min roll was
            # measured to fire in 96.5% of matches where the team was
            # chasing, so it was a timer rather than a blocker - but the
            # decision it stood in for still was not being made by anyone.
            if (not team_subs and minute >= self.QUALITY_GATE_MINUTE
                    and self.subs_left(team) >= 1
                    and self.window_available(team, minute)
                    and self._worth_a_game_state_change(
                        players, available_bench, goal_diff, minute)):

                if goal_diff < 0:
                    # Trailing — bring on the best attacker available, and
                    # take off the most drained man who is not carrying the
                    # ball's value. Rey et al.: offensive substitutions are
                    # markedly more likely when behind.
                    player_off = self._pick_weakest_player(
                        players, team, preferred_pos=DEFENSIVE_SPINE
                    )
                    sub_player = self._pick_attacking_sub(
                        available_bench, player_off, minute, goal_diff
                    )
                    if sub_player and player_off:
                        team_subs.append(self._make_sub(
                            team, player_off, sub_player, minute,
                            SubReason.GAME_STATE, available_bench
                        ))

                elif goal_diff > 0 and minute >= SUB_PROTECT_LEAD_MINUTE:
                    # Leading — the Wharton rule is at its strictest here.
                    # A defensive downgrade is never acceptable, so
                    # `_apply_quality_rule` screens the whole bench for a
                    # candidate who does not cost defensive quality, and
                    # falls back to the best available if none exists.
                    player_off = self._pick_most_drained(players, team, exclude_pos=["GK"])
                    sub_player = self._pick_fresh_sub(
                        available_bench, player_off, minute, goal_diff
                    )
                    if sub_player and player_off:
                        team_subs.append(self._make_sub(
                            team, player_off, sub_player, minute,
                            SubReason.GAME_STATE, available_bench
                        ))


                else:
                    # LEVEL ON THE SCORE - the branch that did not exist.
                    #
                    # `if goal_diff < 0` / `elif goal_diff > 0` had no
                    # `else`, so a team 0-0 or 1-1 could NEVER take a
                    # game-state substitution: 100% of matches, zero
                    # changes, structurally. Level is the most common
                    # state in football, and it is where a real manager
                    # most often fiddles - changing the shape, adding an
                    # attacker for a tiring midfielder, taking a leg off a
                    # ninety.
                    #
                    # The irony worth recording: the QUALITY rule already
                    # handled level perfectly, with its own 30-minute
                    # thresholds and the comment "balanced, and slightly
                    # earlier than leading". I wrote that branch and
                    # never wrote the trigger that reaches it. That is the
                    # project's standing pathology - a mechanism wired to
                    # nothing, wearing a comment that says it works.
                    player_off = self._pick_most_drained(
                        players, team, exclude_pos=["GK"])
                    sub_player = self._pick_fresh_sub(
                        available_bench, player_off, minute, goal_diff
                    )
                    if sub_player and player_off:
                        team_subs.append(self._make_sub(
                            team, player_off, sub_player, minute,
                            SubReason.GAME_STATE, available_bench
                        ))
            subs_to_make.extend(team_subs)

        return subs_to_make

    def get_sub_freshness_boost(self, player) -> float:
        """
        A fresh sub enters with a burst of energy.
        Super subs get it amplified.

        Returns a performance multiplier for the sub's first 20 minutes.
        """
        specs = []
        if hasattr(player, "dna"):
            specs = getattr(player.dna, "specialties", [])
        elif hasattr(player, "specialties"):
            specs = player.specialties or []

        base_boost = random.uniform(1.08, 1.18)   # 8–18% boost
        if "super_sub" in specs:
            base_boost *= 1.20   # Super sub amplified
        if "workhorse" in specs:
            base_boost *= 1.05
        return round(min(1.45, base_boost), 3)

    # ── PRIVATE HELPERS ───────────────────────────────────────

    def window_available(self, team: str, minute: int) -> bool:
        """Can this team legally make another change at `minute`?

        Three cases, in the order the Laws treat them:

          * half-time           — always free, and does not open a window
          * same stoppage as
            the team's last
            change                 — free, it is a BATCH
          * a new stoppage      — costs one of the three windows
        """
        if minute == self.HALF_TIME_MINUTE:
            return True
        if self._last_window_minute.get(team, -99) == minute:
            return True                      # batch, free
        return self.windows_used.get(team, 0) < self.MAX_WINDOWS

    def _spend_window(self, team: str, minute: int) -> bool:
        """Charge a window. Returns True if one was actually charged.

        SATURATES at MAX_WINDOWS. An injury substitution is deliberately
        not gated on `window_available` — you cannot refuse to replace a
        broken player — so without saturation a single match could report
        five windows against a three-window limit, which is exactly what a
        real match did during development: "4 subs in 4 windows". The tally
        has to stay truthful, so the counter stops at the limit and the
        change is recorded as having exceeded it instead.

        Returns False when nothing was charged (half-time, a batch, or the
        allowance was already spent).
        """
        if minute == self.HALF_TIME_MINUTE:
            return False                       # free, untracked
        if self._last_window_minute.get(team, -99) == minute:
            return False                       # already opened this stoppage
        if self.windows_used.get(team, 0) >= self.MAX_WINDOWS:
            self.windows_exceeded[team] = self.windows_exceeded.get(team, 0) + 1
            self._last_window_minute[team] = minute
            return False
        self.windows_used[team] = self.windows_used.get(team, 0) + 1
        self._last_window_minute[team] = minute
        return True

    def subs_left(self, team: str) -> int:
        return max(0, self.MAX_SUBS - self.subs_made.get(team, 0))

    def _worth_a_game_state_change(self, players, available_bench: List,
                                   goal_diff: int, minute: int) -> bool:
        """Would a manager actually make a change right now?

        Replaces an 8%-per-minute coin flip with a decision derived from the
        state of the match. Three signals, each answering a different
        question a real manager asks:

          upgrade  is there a clearly better man sitting there? This is the
                   "you picked the wrong eleven" signal, and it is the one
                   the previous implementation could never express at all.
          strain   is anyone struggling badly enough to come off?
          pressure does the scoreline want a change at this minute?

        The signals ADD rather than multiply, so a team can act on any one
        of them — a brilliant substitute is reason enough on his own, and so
        is a man at 30% who does not want to come off.

        DETERMINISTIC. A per-minute coin flip made the substitution
        sequence depend on the global RNG draw count, which is exactly the
        class of thing that makes a match unreproducible for reasons that
        have nothing to do with football.

        Why there is a threshold at all, rather than "change whenever
        anything is true": Kwest's analysis of the first five-sub Premier
        League season found the managers substituting most frequently are
        read as the ones who do not know their best XI. A manager who
        changes every time a tired man twitches looks like an idiot, so a
        marginal signal does not clear the bar on its own.
        """
        pool = [p for p in available_bench
                if not getattr(p, "_used_as_sub", False)]
        if not pool:
            return False

        upgrade = 0.0
        best_bench = max(sub_quality(p, self) for p in pool)

        on_pitch = [p for p in players
                    if not getattr(p, "_subbed_off", False)
                    and _position_of(p) != "GK"]
        if on_pitch:
            worst = min(sub_quality(p, self) for p in on_pitch)
            upgrade = best_bench - worst

        strain = 0.0
        for p in on_pitch:
            st = self.stamina.get(getattr(p, "name", ""))
            if st is None:
                continue
            if st.current_stamina < GAME_STATE_STRAIN_STAMINA:
                # A man at 20% matters more than a man at 44%.
                depth = (GAME_STATE_STRAIN_STAMINA - st.current_stamina) \
                    / GAME_STATE_STRAIN_STAMINA
                strain = max(strain, min(1.0, depth))

        pressure = game_state_pressure(goal_diff, minute)

        score = 0.0
        if upgrade >= GAME_STATE_UPGRADE_MIN:
            score += min(1.0, (upgrade - GAME_STATE_UPGRADE_MIN)
                         / GAME_STATE_UPGRADE_SPAN)
        if strain > 0.0:
            score += strain
        score += 0.60 * pressure

        self.game_state_stats["considered"] += 1
        worth = score >= GAME_STATE_THRESHOLD
        if worth:
            self.game_state_stats["taken"] += 1
        else:
            self.game_state_stats["declined"] += 1
        return worth

    def _make_sub(self, team: str, player_off, sub_on,
                  minute: int, reason: SubReason,
                  available_bench: List) -> Dict:
        """Execute a substitution."""
        name_off = getattr(player_off, "name", "?")
        name_on  = getattr(sub_on,     "name", "?")

        # Mark players
        player_off._subbed_off = True
        sub_on._used_as_sub    = True

        # Set minutes played
        if hasattr(player_off, "dna"):
            player_off.dna.minutes_played = minute
            player_off.sub_out_minute = minute
        if hasattr(sub_on, "dna"):
            sub_on.dna.minutes_played = 0   # Will be set at final whistle
            sub_on.sub_in_minute = minute
            sub_on.is_starter = False

        # Register sub's stamina. NOT 100.0 — a player who has sat on the
        # bench for 60 minutes has recovered some but not all of what he
        # started the match with, and `register_player` used to hardcode a
        # full tank. That single literal is what made "resting a tired man"
        # identical to "giving a fit man a breather", which is the whole
        # point of a substitution. The player's pre-match fatigue lives on
        # `dna.form.fatigue_level` (written by auto_run_match's
        # _apply_starting_stamina) and `dna._starting_stamina` (also written
        # there as of 2026-10-14); the sub recovers at
        # SUB_SEATED_RECOVERY_PER_MIN per minute since kickoff, capped by
        # his pre-match tank.
        # Window state BEFORE this change is charged. Reading it after
        # `_spend_window` would be reading a value this same call just
        # wrote, which is how every substitution in the development match
        # was reported as `is_batched=True` — including seven changes at
        # seven different minutes. Same class as the jump guard that was
        # inserted after the assignment it was meant to precede.
        was_batched = (minute == self._last_window_minute.get(team, -99)
                       and minute != self.HALF_TIME_MINUTE)

        self._spend_window(team, minute)
        incoming = self._sub_entry_stamina(sub_on, minute)
        freshness_boost = self.get_sub_freshness_boost(sub_on)
        self.register_player(sub_on, starting_stamina=incoming)
        state = self.stamina.get(name_on)
        if state:
            state.performance_mult = freshness_boost

        self.subs_made[team] += 1
        available_bench.remove(sub_on)

        sub_record = {
            "team":       team,
            "minute":     minute,
            "player_off": name_off,
            "player_on":  name_on,
            "reason":     reason.value,
            "freshness":  freshness_boost,
            "stamina_at_exit": self.stamina.get(name_off, PlayerStaminaState("?","CM")).current_stamina,
            # Added 2026-10-14 — the selection is now a scored decision, so
            # the decision itself has to be recorded. Without these the
            # export can say WHICH players changed but not WHY the bench was
            # used in that order, which is the only way to audit the
            # Wharton rule after the fact.
            "quality_on":      round(sub_quality(sub_on, self), 2),
            "quality_off":     round(sub_quality(player_off, self), 2),
            "attack_on":       round(sub_attacking_quality(sub_on), 2),
            "attack_off":      round(sub_attacking_quality(player_off), 2),
            "defence_on":      round(sub_defensive_quality(sub_on), 2),
            "defence_off":     round(sub_defensive_quality(player_off), 2),
            "windows_used":    self.windows_used.get(team, 0),
            "is_batched":      was_batched,
            "windows_exceeded": self.windows_exceeded.get(team, 0),
        }
        self.subs_log.append(sub_record)

        return sub_record

    def _sub_entry_stamina(self, player, minute: int) -> float:
        """Tank a substitute arrives with.

        Reads the pre-match figure the season layer computed, then adds
        however much he recovered sitting out. A sub who has been on since
        kickoff is not the same asset as one who has been resting for an
        hour, and the previous code could not tell them apart because it
        passed 100.0.
        """
        dna = getattr(player, "dna", None)
        base = None
        if dna is not None:
            for attr in ("_starting_stamina",):
                val = getattr(dna, attr, None)
                if isinstance(val, (int, float)):
                    base = float(val)
                    break
        fatigue = getattr(getattr(dna, "form", None), "fatigue_level", None) \
            if dna is not None else None
        if base is None and isinstance(fatigue, (int, float)):
            base = 100.0 - float(fatigue)
        if base is None:
            base = 100.0

        recovered = base + SUB_SEATED_RECOVERY_PER_MIN * float(minute)
        return max(0.0, min(100.0, recovered))

    # ── bench ranking ────────────────────────────────────────────
    #
    # Everything below selects from `_ranked_bench`, which sorts by
    # `sub_quality` rather than returning bench[0]. The helpers are kept
    # as separate methods because each one still encodes a real positional
    # intent (whoever comes on should be able to do this role), but the
    # ORDER inside each intent is now quality-descending.

    def _ranked_bench(self, bench: List, want_pos: Optional[List[str]] = None
                       ) -> List:
        """Unused bench players, best first.

        `want_pos` splits the result into an exact-position group followed by
        an adjacent-position group, both ranked by quality. Returning them
        as one ranked list with the preference already applied means callers
        can take `candidates[0]` and still get the best man for the role,
        instead of the best man who happens to be exactly that position.
        """
        pool = [p for p in bench if not getattr(p, "_used_as_sub", False)]
        if not pool:
            return []

        if not want_pos:
            return sorted(pool, key=lambda p: -sub_quality(p, self))

        pos_of = _position_of
        exact = [p for p in pool if pos_of(p) in want_pos]
        adjacent = [p for p in pool if pos_of(p) not in want_pos]

        key = lambda p: -sub_quality(p, self)          # noqa: E731
        return sorted(exact, key=key) + sorted(adjacent, key=key)

    def _pick_sub(self, bench: List, player_off, reason: str,
                  minute: int = 0, team: str = "", goal_diff: int = 0,
                  quality_gate: bool = True):
        """Best available substitute for the role `player_off` is leaving.

        Position still decides the PREFERENCE (an exact match beats an
        adjacent one) but never the CHOICE: the most qualified man who can
        plausibly fill the role wins. This is the change the whole section
        exists for — the previous version returned `same_pos[0]`, i.e. the
        first exact match in bench order, which is market-value order and
        nothing more.
        """
        pos_off = _position_of(player_off)
        if pos_off == "GK":
            ranked = self._ranked_bench(bench, ["GK"])
        else:
            ranked = self._ranked_bench(bench, [pos_off] + ADJACENT_POS.get(pos_off, []))
        if not ranked:
            return None

        if quality_gate:
            sub_player = self._apply_quality_rule(
                ranked, player_off, minute, team, goal_diff
            )
            if sub_player is not None:
                return sub_player

        # The gate rejected every candidate. Rather than return nothing and
        # leave the team short-handed — which is worse football than a modest
        # downgrade — fall back to the best candidate and say so in the log.
        # A forced injury sub must never be blocked by a scoreline rule; the
        # caller decides by passing quality_gate=False for that.
        return ranked[0]

    def _apply_quality_rule(self, ranked: List, player_off, minute: int,
                            team: str, goal_diff: int):
        """Walk the ranked bench until one substitution passes the scoreline
        rule.

        Ranking is descending by quality, so the FIRST candidate to satisfy
        `acceptable_quality_loss` is the best man who is affordable — which
        is precisely the trade the Wharton result describes. Returns None if
        nothing qualifies, so the caller can decide whether that blocks the
        change or merely downgrades it.
        """
        off_atk = sub_attacking_quality(player_off)
        off_def = sub_defensive_quality(player_off)
        considered = 0
        for cand in ranked:
            ok, _why = acceptable_quality_loss(
                goal_diff, minute,
                sub_attacking_quality(cand), off_atk,
                sub_defensive_quality(cand), off_def,
            )
            considered += 1
            if ok:
                self.quality_gate_stats["considered"] += considered
                self.quality_gate_stats["accepted"] += 1
                return cand
            self.quality_gate_stats["rejected"] += 1
        # Every candidate refused. `quality_gate_stats` records it so the
        # branch is observable — a rule that silently never binds looks
        # identical to a rule that is not implemented.
        self.quality_gate_stats["exhausted"] += 1
        self.quality_gate_stats["considered"] += considered
        return None

    def _find_tactical_sub(self, bench: List, player_off) -> Optional[object]:
        """The pre-planned substitute coming on for `player_off`.

        PREVIOUSLY BROKEN IN TWO WAYS. It computed `sub_in`, compared it to
        nothing, and returned the first available player — so the planned
        minute was read and discarded, and the planned MAN was never
        consulted either; any unused sub fitted any slot. The loop is now
        driven by whether the player's own `sub_in_minute` has arrived, and
        within that the selection is quality-ranked.

        The controller's `tactical_schedule` is keyed by the name of the
        player coming OFF (`{"Eli Dago": 68}`), so `target_minute` gates the
        slot, and any unused sub may fill it — which is correct, because the
        schedule names the outgoing player, not the incoming one.
        """
        name_off = getattr(player_off, "name", "")
        target_minute = self.tactical_schedule.get(name_off)
        if target_minute is None:
            return None

        pos_off = _position_of(player_off)
        if pos_off == "GK":
            return (self._ranked_bench(bench, ["GK"]) or [None])[0]
        return (self._ranked_bench(bench, [pos_off] + ADJACENT_POS.get(pos_off, []))
                or [None])[0]

    def _pick_attacking_sub(self, bench: List, player_off=None, minute: int = 0,
                            goal_diff: int = 0):
        """Best available attacker, quality-ranked.

        Used when chasing. The scoreline rule is applied against
        `player_off` when one is supplied so a chasing team does not
        accidentally take a creative player off for a defensive one.
        """
        ranked = self._ranked_bench(bench, ATTACKING_POS)
        if not ranked:
            return None
        if player_off is not None:
            ok = self._apply_quality_rule(ranked, player_off, minute, "", goal_diff)
            if ok is not None:
                return ok
        return ranked[0]

    def _pick_fresh_sub(self, bench: List, player_off=None, minute: int = 0,
                        goal_diff: int = 0):
        """Best available player, full quality ranking, no positional filter.

        This is the "protect the lead" path, so it deliberately prefers
        DEFENSIVE quality when it is available and the scoreline allows it.
        Ranking is by `sub_quality` (overall ability, form, soul, freshness)
        and the scoreline rule then screens out anything that costs defensive
        quality — which, leading, is everything.
        """
        ranked = self._ranked_bench(bench)
        if not ranked:
            return None
        if player_off is not None:
            ok = self._apply_quality_rule(ranked, player_off, minute, "", goal_diff)
            if ok is not None:
                return ok
        return ranked[0]

    def _pick_weakest_player(self, players: List, team: str,
                              preferred_pos: List[str] = None) -> Optional[object]:
        """Pick the most drained non-GK player to bring off."""
        candidates = [
            p for p in players
            if not getattr(p, "_subbed_off", False)
            and getattr(getattr(p, "dna", None), "position",
                       getattr(p, "position", "")) != "GK"
        ]
        if preferred_pos:
            pref = [p for p in candidates
                    if getattr(getattr(p, "dna", None), "position",
                               getattr(p, "position", "")) in preferred_pos]
            if pref:
                candidates = pref

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda p: self.stamina.get(
                getattr(p, "name", ""), PlayerStaminaState("?", "CM")
            ).current_stamina
        )

    def _pick_most_drained(self, players: List, team: str,
                            exclude_pos: List[str] = None) -> Optional[object]:
        candidates = [
            p for p in players
            if not getattr(p, "_subbed_off", False)
        ]
        if exclude_pos:
            candidates = [p for p in candidates
                          if getattr(getattr(p, "dna", None), "position",
                                     getattr(p, "position", "")) not in exclude_pos]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda p: self.stamina.get(
                getattr(p, "name", ""), PlayerStaminaState("?", "CM")
            ).current_stamina
        )

    def get_stamina_report(self) -> pd.DataFrame:
        """Return stamina states as DataFrame for export."""
        rows = []
        for name, s in self.stamina.items():
            rows.append({
                "Player":            name,
                "Position":          s.position,
                "Starting Stamina":  round(s.starting_stamina, 1),
                "Ending Stamina":    round(s.current_stamina, 1),
                "Total Drained (%)": round(s.total_drained, 1),
                "Performance Mult":  s.performance_mult,
                "Injured":           s.is_injured,
                "Injury Type":       s.injury_type if s.is_injured else "—",
                "Injury Minute":     s.injury_minute if s.is_injured else "—",
                "Injury Severity":   round(s.injury_severity, 1) if s.is_injured else 0,
                "Matches Out":       s.matches_out() if s.is_injured else 0,
                "Recovery Days":     s.recovery_days_needed(),
                "Sub Requested":     s.sub_requested,
                "Sub Reason":        s.sub_reason.value if s.sub_reason else "—",
                "Top Drain Action":  max(s.drain_by_action, key=s.drain_by_action.get)
                                     if s.drain_by_action else "—",
            })
        return pd.DataFrame(rows)

    def get_subs_report(self) -> pd.DataFrame:
        """Return substitution log as DataFrame."""
        return pd.DataFrame(self.subs_log)


# ─────────────────────────────────────────────
# AVAILABILITY CHECKER
# Reads previous match files to determine who can play
# ─────────────────────────────────────────────

@dataclass
class PlayerAvailability:
    name: str
    status: AvailabilityStatus
    reason: str
    matches_suspended: int  = 0
    matches_injured: int    = 0
    fatigue_level: float    = 0.0     # 0–100 from last match drain
    starting_stamina: float = 100.0   # What they start with this match
    yellow_cards_last_6: int = 0

    # ── SEASON LOAD (added 2026-10-15) ────────────────────────────
    # Rotation intelligence, carried on the same object the XI selection
    # already receives.
    #
    # `season_manager.rotation_flag` has existed since the season layer was
    # written and computes exactly the right thing — average minutes per
    # game over 75 AND projected fatigue over 55 means "rest him". It was
    # called from TWO places, both of which append a string to a briefing.
    # Neither changed a single selection, so the whole computation was made,
    # printed, and discarded.
    #
    # It could not do otherwise: the XI is not chosen by `pick_best_xi`
    # (which has ZERO production call sites — it appears in its own
    # definition and one docstring example) but by
    # `roster_loader._pick_from_pool`, which sorts candidates by
    # MARKET VALUE and had no access to any season state at all. So the load
    # had to be carried here, on the object that reaches the real selector.
    season_matches: int = 0
    season_minutes: int = 0
    days_since_last_match: Optional[int] = None

    @property
    def avg_minutes_per_game(self) -> float:
        """Average minutes played per appearance this season, 0.0 unknown."""
        if self.season_matches <= 0:
            return 0.0
        return self.season_minutes / float(self.season_matches)

    @property
    def rotation_score(self) -> float:
        """How strongly this player wants a rest, 0.0 (fresh) - 1.0 (spent).

        Two inputs, both measured rather than invented:

          load        average minutes per game. A squad that wins matches
                      plays its best players ~90 a time; real rotation data
                      (the core of 14-16 sharing most minutes) says that
                      cannot be sustained. The penalty starts at 78 minutes,
                      not at 0, because a player who plays every game and
                      loves it is not a rotation candidate — a player doing
                      90 minutes a game while gassed is.

          turnaround  days since his last appearance. Saturday to Wednesday
                      is 3 days. This is the input with no representation at
                      all in the season layer: `days_since`, `rest_days` and
                      `congest` return zero occurrences across the whole
                      codebase, so the fixture list currently has no effect
                      on who is picked.
        """
        load = 0.0
        avg = self.avg_minutes_per_game
        if avg >= ROTATION_LOAD_GATE_MINUTES:
            span = max(1.0, 90.0 - ROTATION_LOAD_GATE_MINUTES)
            load = min(1.0, (avg - ROTATION_LOAD_GATE_MINUTES) / span)

        rest = 0.0
        gap = self.days_since_last_match
        if gap is not None and gap <= ROTATION_SHORT_REST_DAYS:
            # 0 days back-to-back is worst, 3 days is the threshold.
            rest = min(1.0, max(0, (ROTATION_SHORT_REST_DAYS - gap))
                       / float(ROTATION_SHORT_REST_DAYS))

        # Load dominates: a tired player on a long turnaround is still tired.
        return min(1.0, 0.75 * load + 0.45 * rest)


# Average minutes per game above which accumulated load starts costing a
# player his place. Not zero: the question is not "has he played" but "is he
# carrying a season", which is what rotation actually manages.
ROTATION_LOAD_GATE_MINUTES = 78.0

# Days between appearances at or below which the turnaround counts as short.
# Saturday -> Wednesday is 3; Sunday -> Thursday is 4.
ROTATION_SHORT_REST_DAYS = 3

# Maximum proportional penalty applied to a player's effective standing.
# 0.22 is roughly "the difference between the best and second-best paid man
# at his position" — enough to flip a close call, not enough to bench a
# player who is clearly the best available with nobody behind him.
ROTATION_MAX_PENALTY = 0.22


class AvailabilityChecker:
    """
    Reads exported match files (CSV or Excel) from the outputs/ folder
    and determines each player's availability for the NEXT match.

    Checks:
        1. Red card ban (1 match suspension)
        2. 5 yellow cards in last 6 PLOFA league games (1 match suspension)
        3. Injury from last match (matches_out > 0)
        4. Stamina drain from last match (affects starting stamina)

    Usage:
        checker = AvailabilityChecker("outputs/")
        report  = checker.check("Hartwell City", matchday=2)
        for name, avail in report.items():
            print(name, avail.status.value, avail.reason)
    """

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir

    def check(
        self,
        team_name: str,
        current_matchday: int,
        lookback_matchdays: int = 6,
    ) -> Dict[str, PlayerAvailability]:
        """
        Check availability for all players of a team.
        Reads last N matchday files for that team.

        Returns: {player_name: PlayerAvailability}
        """
        # Load all relevant match files
        match_files = self._find_match_files(team_name, current_matchday, lookback_matchdays)

        if not match_files:
            print(f"  ⚠️  No previous match files found for {team_name}. All players assumed fit.")
            return {}

        # Read and combine
        frames = []
        for fpath, matchday in match_files:
            try:
                if fpath.endswith(".xlsx"):
                    df = pd.read_excel(fpath, sheet_name="Player Stats")
                else:
                    df = pd.read_csv(fpath)
                # Defragment before stamping the matchday column; inserting into
                # a freshly-read (fragmented) frame trips pandas' insert warning.
                df = df.copy()
                df["_matchday"] = matchday
                frames.append(df)
            except Exception as e:
                print(f"  ⚠️  Could not read {fpath}: {e}")

        if not frames:
            return {}

        combined = pd.concat(frames, ignore_index=True)
        team_df  = combined[combined.get("team", combined.get("Team", pd.Series())) == team_name]

        if team_df.empty:
            print(f"  ⚠️  No data found for {team_name} in match files.")
            return {}

        results: Dict[str, PlayerAvailability] = {}

        for player_name in team_df["player"].dropna().unique() if "player" in team_df.columns \
                           else team_df["Player"].dropna().unique():

            player_rows = team_df[
                (team_df.get("player", team_df.get("Player", pd.Series())) == player_name)
            ].sort_values("_matchday", ascending=False)

            avail = self._assess_player(player_name, player_rows, current_matchday)
            results[player_name] = avail

        return results

    def _assess_player(
        self, name: str, rows: pd.DataFrame, current_md: int
    ) -> PlayerAvailability:
        """Assess a single player's availability from their match history rows."""

        def get_col(df, *names, default=0):
            for n in names:
                if n in df.columns:
                    return df[n]
            return pd.Series([default] * len(df))

        # ── RED CARD BAN ──────────────────────────────────────
        last_match = rows.iloc[0] if len(rows) > 0 else None
        if last_match is not None:
            red_cards = get_col(rows.head(1), "red_cards", "Red Cards").sum()
            if red_cards >= 1:
                return PlayerAvailability(
                    name=name,
                    status=AvailabilityStatus.SUSPENDED_RED,
                    reason="Red card ban (1 match)",
                    matches_suspended=1,
                    starting_stamina=100.0,
                )

        # ── INJURY ────────────────────────────────────────────
        if last_match is not None:
            injured = get_col(rows.head(1), "is_injured", "Injured").iloc[0]
            matches_out_val = get_col(rows.head(1), "matches_out", "Matches Out").iloc[0]
            if str(injured).lower() in ("true", "yes", "1") and int(matches_out_val or 0) > 0:
                return PlayerAvailability(
                    name=name,
                    status=AvailabilityStatus.INJURED,
                    reason=f"Injured — {int(matches_out_val)} match(es) out",
                    matches_injured=int(matches_out_val),
                    starting_stamina=100.0,
                )

        # ── 5 YELLOWS IN 6 GAMES ─────────────────────────────
        last_6 = rows.head(6)
        yellow_total = get_col(last_6, "yellow_cards", "Yellow Cards").sum()
        if yellow_total >= 5:
            return PlayerAvailability(
                name=name,
                status=AvailabilityStatus.SUSPENDED_YEL,
                reason=f"5 yellow cards in last 6 games ({int(yellow_total)} yellows)",
                matches_suspended=1,
                yellow_cards_last_6=int(yellow_total),
                starting_stamina=100.0,
            )

        # ── STAMINA / FATIGUE FROM LAST MATCH ─────────────────
        starting_stamina = 100.0
        fatigue_level = 0.0

        # Date-based rest window: matchdays between this assessment and
        # the player's last appearance, at the league's ~7-day cadence.
        # (_find_match_files only returns matchdays strictly before
        # current_md, so the gap is always >= 1 matchday when known.)
        days_available = 6  # fallback when no _matchday column
        if last_match is not None and "_matchday" in rows.columns:
            try:
                last_md = int(rows["_matchday"].iloc[0])
                days_available = max(0, (current_md - last_md) * 7)
            except (TypeError, ValueError):
                days_available = 6

        if last_match is not None:
            # Read ending stamina or total drain from last match
            ending_stamina = get_col(rows.head(1), "ending_stamina", "Ending Stamina")
            total_drain    = get_col(rows.head(1), "total_drained", "Total Stamina Drained")
            mins_played    = get_col(rows.head(1), "minutes_played", "Minutes Played")

            if not ending_stamina.empty and ending_stamina.iloc[0] > 0:
                last_ending = float(ending_stamina.iloc[0])
                fatigue_level = 100.0 - last_ending
                # Recovery: 10% per rest day (date-based via matchday gap)
                days_to_recover = fatigue_level / 10.0
                recovery_pct    = min(1.0, days_available / max(1, days_to_recover))
                starting_stamina = min(100.0, last_ending + (fatigue_level * recovery_pct))
            elif not total_drain.empty and total_drain.iloc[0] > 0:
                drain = float(total_drain.iloc[0])
                fatigue_level = drain
                recovery = min(drain, days_available * 10)
                starting_stamina = min(100.0, 100.0 - drain + recovery)
            else:
                # Estimate from minutes played
                mins = float(mins_played.iloc[0]) if not mins_played.empty else 90
                estimated_drain = (mins / 90) * 45   # 90 mins ≈ 45% drain
                fatigue_level = estimated_drain
                starting_stamina = min(100.0, 100.0 - estimated_drain * 0.3)

        starting_stamina = round(max(60.0, min(100.0, starting_stamina)), 1)
        fatigue_level    = round(fatigue_level, 1)

        # ── FATIGUE WARNING ───────────────────────────────────
        if fatigue_level > 75 or starting_stamina < 72:
            return PlayerAvailability(
                name=name,
                status=AvailabilityStatus.FATIGUE_WARNING,
                reason=f"Heavy drain last match (fatigue: {fatigue_level:.0f}%). "
                       f"Starting stamina: {starting_stamina:.0f}%",
                fatigue_level=fatigue_level,
                starting_stamina=starting_stamina,
                yellow_cards_last_6=int(yellow_total),
            )

        # ── FIT ───────────────────────────────────────────────
        return PlayerAvailability(
            name=name,
            status=AvailabilityStatus.FIT,
            reason="Fit to play",
            fatigue_level=fatigue_level,
            starting_stamina=starting_stamina,
            yellow_cards_last_6=int(yellow_total),
        )

    def _find_match_files(
        self, team_name: str, current_md: int, lookback: int
    ) -> List[Tuple[str, int]]:
        """
        Find Excel or CSV files in outputs/ for the previous N matchdays.
        Looks for files containing team_name and MD numbers.
        """
        results = []
        if not os.path.isdir(self.outputs_dir):
            return results

        for folder in sorted(os.listdir(self.outputs_dir), reverse=True):
            folder_path = os.path.join(self.outputs_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # Extract matchday from folder name (e.g. "...MD01")
            md_num = None
            for part in folder.split("_"):
                if part.startswith("MD") and part[2:].isdigit():
                    md_num = int(part[2:])
                    break

            if md_num is None:
                continue
            if md_num >= current_md:
                continue   # Skip current or future matchdays
            if current_md - md_num > lookback:
                continue   # Too far back

            # Check team is involved
            if team_name.replace(" ", "_") not in folder.replace(" ", "_"):
                continue

            # Find the Excel file
            for fname in os.listdir(folder_path):
                if fname.endswith(".xlsx") and "players" not in fname:
                    results.append((os.path.join(folder_path, fname), md_num))
                    break
                elif fname.endswith("_players.csv"):
                    results.append((os.path.join(folder_path, fname), md_num))
                    break

        return results

    def print_report(self, team_name: str, availability: Dict[str, PlayerAvailability]):
        """Pretty-print availability report to console."""
        STATUS_ICONS = {
            AvailabilityStatus.FIT:             "✅",
            AvailabilityStatus.SUSPENDED_RED:   "🟥",
            AvailabilityStatus.SUSPENDED_YEL:   "🟨",
            AvailabilityStatus.INJURED:         "🤕",
            AvailabilityStatus.FATIGUE_WARNING: "⚠️ ",
            AvailabilityStatus.DOUBTFUL:        "❓",
        }
        print(f"\n  📋 AVAILABILITY REPORT — {team_name}")
        print(f"  {'─'*55}")
        print(f"  {'Player':<22} {'Status':<18} {'Stamina':>8}  Reason")
        print(f"  {'─'*55}")
        for name, avail in sorted(availability.items()):
            icon   = STATUS_ICONS.get(avail.status, "?")
            stamina = f"{avail.starting_stamina:.0f}%" if avail.status == AvailabilityStatus.FIT \
                      else "—"
            print(f"  {name:<22} {icon} {avail.status.value:<15} {stamina:>8}  {avail.reason}")
        print(f"  {'─'*55}")

        not_available = [
            n for n, a in availability.items()
            if a.status not in (AvailabilityStatus.FIT, AvailabilityStatus.FATIGUE_WARNING)
        ]
        if not_available:
            print(f"\n  ❌ NOT AVAILABLE: {', '.join(not_available)}")


# ─────────────────────────────────────────────
# STAMINA INTEGRATION HOOK
# Called by MatchEngine._simulate_minute()
# ─────────────────────────────────────────────

# Map from EventType name → stamina action key
# ─────────────────────────────────────────────
# COMPLETE STAMINA ACTION MAPPING
# Every EventType in match_engine.py must have a mapping
# ─────────────────────────────────────────────

EVENT_TO_STAMINA_ACTION = {
    # ── PASSING ──────────────────────────────────────────────
    "PASS":                 "pass_short",
    "PROGRESSIVE_PASS":     "progressive_pass",
    "SWITCH_OF_PLAY":       "pass_long",
    "THROUGH_BALL":         "through_ball",
    "CROSS_ATTEMPT":        "cross_attempt",
    "CROSS_SUCCESS":        "cross_attempt",
    "CORNER_TAKEN":         "cross_attempt",
    "FREEKICK_CROSS":       "cross_attempt",
    "FREEKICK_DIRECT":      "shot_attempt",
    
    # ── BALL RECEIPT / TOUCH ──────────────────────────────────
    "BALL_RECEIPT":         "touch",
    "MISCONTROL":           "touch",
    "TOUCH":                "touch",
    "FIRST_TOUCH":          "touch",
    
    # ── CARRYING ──────────────────────────────────────────────
    "CARRY":                "carry",
    "PROGRESSIVE_CARRY":    "carry",
    
    # ── DRIBBLING ─────────────────────────────────────────────
    "DRIBBLE_SUCCESS":      "dribble_success",
    "DRIBBLE_FAIL":         "dribble_fail",
    "DRIBBLE_ATTEMPT":      "dribble_success",
    "DISPOSSESSED":         "dribble_fail",
    
    # ── SHOOTING ──────────────────────────────────────────────
    "SHOT_ON_TARGET":       "shot_attempt",
    "SHOT_OFF_TARGET":      "shot_attempt",
    "SHOT_BLOCKED":         "shot_attempt",
    "GOAL":                 "shot_attempt",
    "PENALTY_SCORED":       "shot_attempt",
    "PENALTY_MISSED":       "shot_attempt",
    "HIT_WOODWORK":         "shot_attempt",
    
    # ── DEFENDING ─────────────────────────────────────────────
    "TACKLE_WON":           "tackle_won",
    "TACKLE_LOST":          "tackle_lost",
    "INTERCEPTION":         "interception",
    "CLEARANCE":            "clearance",
    "OWN_GOAL":             "clearance",
    "BLOCK":                "block",
    "RECOVERY":             "recovery",
    "BALL_RECOVERY":        "recovery",
    "FIFTY_FIFTY":          "ground_duel",
    "POSSESSION_WON":       "recovery",
    
    # ── PRESSING ──────────────────────────────────────────────
    "PRESS":                "press",
    "PRESS_SUCCESS":        "press_success",
    "PRESSURE":             "press",
    "HIGH_PRESS":           "press",
    
    # ── DUELS ──────────────────────────────────────────────────
    "AERIAL_DUEL":          "aerial_duel",
    "GROUND_DUEL":          "ground_duel",
    
    # ── GK ─────────────────────────────────────────────────────
    "SAVE":                 "save",
    "HIGH_CLAIM":           "high_claim",
    "GOALLINE_SAVE":        "goalline_save",
    "GK_KICK":              "pass_long",
    "GK_THROW":             "pass_short",
    "GOAL_KICK":            "pass_long",
    "THROW_IN":             "pass_short",
    
    # ── DISCIPLINE ────────────────────────────────────────────
    "FOUL_COMMITTED":       "tackle_aggressive",
    "FOUL_WON":             "sprint",
    "YELLOW_CARD":          "tackle_aggressive",
    "RED_CARD":             "tackle_aggressive",
    
    # ── PHYSICAL ──────────────────────────────────────────────
    "SPRINT":               "sprint",
    "HIGH_SPEED_SPRINT":    "high_speed_sprint",
    
    # ── SET PIECE ─────────────────────────────────────────────
    "CORNER_WON":           "touch",
    "FREEKICK_WON":         "touch",
    "PENALTY_WON":          "touch",
    
    # ── MATCH CONTROL ─────────────────────────────────────────
    "SUBSTITUTION":         "standing",
    "INJURY":               "injury_event",
    "ADDED_TIME_SIGNAL":    "standing",
    
    # ── CHANCE CREATION ───────────────────────────────────────
    "CHANCE_CREATED":       "pass_short",
    "BIG_CHANCE_CREATED":   "pass_short",
    "KEY_PASS":             "pass_short",
    
    # ── TURNOVERS ─────────────────────────────────────────────
    "TURNOVER":             "turnover",
    "POSSESSION_LOST":      "turnover",
    "POSSESSION_SEQUENCE":  "touch",
}


def get_stamina_action(event_type_name: str) -> Optional[str]:
    """Map event type name to stamina cost key."""
    return EVENT_TO_STAMINA_ACTION.get(event_type_name)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n⚡ PLOFA 26/27 — Squad Manager Demo")
    print("="*55)

    # ── Stamina Engine Demo ───────────────────────────────────
    print("\n1. STAMINA ENGINE — Percy playing a high-press match\n")

    percy_stamina = PlayerStaminaState(
        player_name="Percy",
        position="RW",
        specialties=["grand_dribbler", "speedster", "inverted"],
        starting_stamina=100.0,
        current_stamina=100.0,
    )

    # Simulate a typical Percy match action sequence
    actions = [
        ("sprint",           15),
        ("dribble_success",  22),
        ("press",            31),
        ("dribble_success",  38),
        ("high_speed_sprint",45),
        ("sprint",           52),
        ("aerial_duel",      58),
        ("dribble_fail",     63),
        ("press",            68),
        ("sprint",           71),
        ("dribble_success",  75),
        ("high_speed_sprint",80),
        ("tackle_lost",      84),
        ("sprint",           88),
        ("dribble_success",  91),
    ]

    print(f"  {'Minute':>8}  {'Action':<22}  {'Stamina':>8}  {'Perf':>6}  {'Status'}")
    print(f"  {'─'*65}")

    for action, minute in actions:
        percy_stamina.drain(action)
        percy_stamina.drain_baseline("attacking")
        percy_stamina.update_performance_mult()
        percy_stamina.check_stamina_sub(minute, manager_stubbornness=0.25)
        injured = percy_stamina.roll_injury(minute, action)

        status = ""
        if injured:
            status = f"🤕 INJURED ({percy_stamina.injury_type})"
        elif percy_stamina.sub_requested:
            status = f"⚠️  Sub requested ({percy_stamina.sub_reason.value})"
        elif percy_stamina.is_critical:
            status = "🔴 CRITICAL"
        elif percy_stamina.is_struggling:
            status = "🟡 Struggling"

        print(f"  {minute:>7}'  {action:<22}  {percy_stamina.stamina_pct:>7.1f}%  "
              f"×{percy_stamina.performance_mult:<5.3f}  {status}")
        if injured:
            break

    print(f"\n  Total drained: {percy_stamina.total_drained:.1f}%")
    print(f"  Recovery needed: {percy_stamina.recovery_days_needed():.1f} days")
    print(f"  Matches out: {percy_stamina.matches_out()}")

    # ── Availability Checker Demo ─────────────────────────────
    print("\n2. AVAILABILITY CHECKER")
    print("   (No previous match files yet — this runs after MD1)")
    print("   Run after generating match outputs to see suspension/injury/fatigue flags.")
    print("\n   Example output:")
    print("   ✅ Dragan Novak    fit              100%  Fit to play")
    print("   🟥 Mateo Sanz      suspended_red    —     Red card ban (1 match)")
    print("   🟨 Demi Adeola     suspended_yel    —     5 yellow cards in last 6 games")
    print("   🤕 Bruno Reis      injured          —     Injured — 2 match(es) out")
    print("   ⚠️  Luca Ferrini    fatigue_warning  74%   Heavy drain last match (fatigue: 72%)")

    # ── Sub Controller Demo ───────────────────────────────────
    print("\n3. SUBSTITUTION CONTROLLER")
    print("   Dragan Novak stamina drops to 18% at 71' → sub triggered\n")

    dragan = PlayerStaminaState("Dragan Novak", "ST", ["aerial_threat"])
    dragan.current_stamina = 18.0
    dragan.update_performance_mult()
    dragan.check_stamina_sub(71, manager_stubbornness=0.2)

    print(f"   Stamina: {dragan.stamina_pct}%")
    print(f"   Performance mult: ×{dragan.performance_mult}")
    print(f"   Sub requested: {dragan.sub_requested}")
    print(f"   Reason: {dragan.sub_reason.value if dragan.sub_reason else 'none'}")
    print(f"   → Calvin Pryce comes on with ×1.14 freshness boost (super_sub)")

    print("\n✅ Squad Manager operational.")
    print("   Wire into match_engine.py simulate() loop for live tracking.\n")