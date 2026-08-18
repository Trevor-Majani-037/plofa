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

    def drain_baseline(self, team_style: str, intensity_mult: float = 1.0):
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
        self.drain("standing", base * style_mult * intensity_mult)

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

    def check_stamina_sub(self, minute: int, manager_stubbornness: float = 0.3):
        """
        Should the manager sub this player for stamina?
        manager_stubbornness: 0.0=immediate, 1.0=never subs for stamina

        Real football: some players finish on fumes. Some managers leave
        them on because subs are used or they trust the player.
        """
        if self.sub_requested or self.is_injured:
            return

        if self.is_critical and minute >= 55:
            # Critical stamina — almost always subbed (unless manager is stubborn
            # or it's too early)
            sub_prob = 0.85 * (1 - manager_stubbornness)
            if random.random() < sub_prob:
                self.sub_requested = True
                self.sub_reason = SubReason.STAMINA
                self.minute_flagged = minute

        elif self.is_struggling and minute >= 70:
            # Struggling late — reasonable chance of sub
            sub_prob = 0.45 * (1 - manager_stubbornness)
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


# ─────────────────────────────────────────────
# SUBSTITUTION CONTROLLER
# Manages the live sub logic during simulation
# ─────────────────────────────────────────────

class SubstitutionController:
    """
    Manages all substitutions during a match.

    Integrates with MatchEngine.simulate() — called each minute
    to check whether any sub should happen.

    Sub slots: 3 per team (PLOFA rules, 5 in some competitions)
    """

    MAX_SUBS = 3   # PLOFA standard

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

        # Track subs made
        self.subs_made: Dict[str, int]       = {home_team: 0, away_team: 0}
        self.subs_log:  List[Dict]            = []

        # Stamina states for all active players
        self.stamina: Dict[str, PlayerStaminaState] = {}

        # Tactical sub schedule from run_match.py
        # {player_name: minute} — pre-planned tactical subs
        self.tactical_schedule: Dict[str, int] = {}

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
        if is_secondary:
            # Get original cost, halve it, then apply any fatigue tax
            original_cost = STAMINA_COSTS.get(action, 0.05)
            state.drain(action, original_cost * 0.5 * drain_mult)
        else:
            state.drain(action, STAMINA_COSTS.get(action, 0.05) * drain_mult)
        
        state.update_performance_mult()
        
        # Roll for injury on physical actions (only for primary actors)
        if not is_secondary and action in (
            "tackle_aggressive", "aerial_duel", "tackle_won",
            "tackle_lost", "sprint", "high_speed_sprint", "dribble_fail"
        ):
            state.roll_injury(minute, action)
        
        # Check if stamina sub needed
        if not state.sub_requested:
            state.check_stamina_sub(minute, self.stubbornness)

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

            goal_diff = game_state_home if team == self.home_team else game_state_away
            subs_remaining = self.MAX_SUBS - self.subs_made[team]

            # ── 1. INJURY SUBS (highest priority) ─────────────
            for player in players:
                name = getattr(player, "name", "")
                state = self.stamina.get(name)
                if state and state.is_injured and not getattr(player, "_subbed_off", False):
                    sub_player = self._pick_sub(available_bench, player, "injury")
                    if sub_player:
                        subs_to_make.append(self._make_sub(
                            team, player, sub_player, minute,
                            SubReason.INJURY, available_bench
                        ))
                        break   # One sub at a time

            if subs_to_make:
                continue

            # ── 2. TACTICAL PRE-PLANNED SUBS ──────────────────
            for player in players:
                name = getattr(player, "name", "")
                if (name in self.tactical_schedule and
                        minute >= self.tactical_schedule[name] and
                        not getattr(player, "_subbed_off", False)):
                    # Find the pre-planned sub coming on
                    sub_player = self._find_tactical_sub(available_bench, player)
                    if sub_player:
                        subs_to_make.append(self._make_sub(
                            team, player, sub_player, minute,
                            SubReason.TACTICAL, available_bench
                        ))

            # ── 3. STAMINA EMERGENCY SUBS ─────────────────────
            if not subs_to_make and minute >= 55:
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
                        sub_player = self._pick_sub(available_bench, player, "stamina")
                        if sub_player:
                            subs_to_make.append(self._make_sub(
                                team, player, sub_player, minute,
                                SubReason.STAMINA, available_bench
                            ))
                            break

            # ── 4. GAME STATE SUBS (attacking/defensive) ──────
            if (not subs_to_make and minute >= 60
                    and subs_remaining >= 1
                    and random.random() < 0.08):   # ~8% chance per minute

                if goal_diff < 0 and minute >= 65:
                    # Losing — bring on attacker
                    sub_player = self._pick_attacking_sub(available_bench)
                    player_off = self._pick_weakest_player(players, team, preferred_pos=["CB", "CDM"])
                    if sub_player and player_off:
                        subs_to_make.append(self._make_sub(
                            team, player_off, sub_player, minute,
                            SubReason.GAME_STATE, available_bench
                        ))

                elif goal_diff > 0 and minute >= 75:
                    # Winning — bring on defender/fresh body
                    sub_player = self._pick_fresh_sub(available_bench)
                    player_off = self._pick_most_drained(players, team, exclude_pos=["GK"])
                    if sub_player and player_off:
                        subs_to_make.append(self._make_sub(
                            team, player_off, sub_player, minute,
                            SubReason.GAME_STATE, available_bench
                        ))

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

        # Register sub's stamina (starts fresh)
        freshness_boost = self.get_sub_freshness_boost(sub_on)
        self.register_player(sub_on, starting_stamina=100.0)
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
        }
        self.subs_log.append(sub_record)

        return sub_record

    def _pick_sub(self, bench: List, player_off, reason: str):
        """Pick best available sub for a position."""
        pos_off = getattr(player_off, "position",
                  getattr(getattr(player_off, "dna", None), "position", "CM"))

        # Try same position first
        same_pos = [p for p in bench if not getattr(p, "_used_as_sub", False)
                    and getattr(getattr(p, "dna", None), "position",
                               getattr(p, "position", "CM")) == pos_off]
        if same_pos:
            return same_pos[0]

        # Adjacent positions
        adjacent = {
            "ST": ["CF", "LW", "RW", "CAM"],
            "CF": ["ST", "CAM", "LW", "RW"],
            "LW": ["RW", "CAM", "ST", "LB"],
            "RW": ["LW", "CAM", "ST", "RB"],
            "CAM": ["CM", "LW", "RW", "ST"],
            "CM": ["CAM", "CDM", "LW", "RW"],
            "CDM": ["CM", "CB"],
            "LB": ["RB", "CB", "LW"],
            "RB": ["LB", "CB", "RW"],
            "CB": ["CDM", "LB", "RB"],
            "GK": ["GK"],
        }.get(pos_off, [])

        adj_subs = [p for p in bench if not getattr(p, "_used_as_sub", False)
                    and getattr(getattr(p, "dna", None), "position",
                               getattr(p, "position", "?")) in adjacent]
        if adj_subs:
            return adj_subs[0]

        # Any available
        available = [p for p in bench if not getattr(p, "_used_as_sub", False)]
        return available[0] if available else None

    def _find_tactical_sub(self, bench: List, player_off) -> Optional[object]:
        """
        Find the pre-planned sub coming on for this player.
        Looks for a sub whose sub_in_minute matches the schedule.
        """
        name_off = getattr(player_off, "name", "")
        target_minute = self.tactical_schedule.get(name_off)
        if target_minute is None:
            return None

        # Find bench player who has this as their sub_in_minute
        for p in bench:
            if not getattr(p, "_used_as_sub", False):
                sub_in = getattr(p, "sub_in_minute", None)
                if sub_in is None and hasattr(p, "dna"):
                    sub_in = getattr(p.dna, "sub_in_minute", None)
                # If no specific minute set, just take positional match
                return p   # Take first available for tactical slot
        return None

    def _pick_attacking_sub(self, bench: List):
        att_pos = ["ST", "CF", "LW", "RW", "CAM"]
        for p in bench:
            if not getattr(p, "_used_as_sub", False):
                pos = getattr(getattr(p, "dna", None), "position",
                             getattr(p, "position", ""))
                if pos in att_pos:
                    return p
        return bench[0] if bench else None

    def _pick_fresh_sub(self, bench: List):
        available = [p for p in bench if not getattr(p, "_used_as_sub", False)]
        return available[0] if available else None

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

        if last_match is not None:
            # Read ending stamina or total drain from last match
            ending_stamina = get_col(rows.head(1), "ending_stamina", "Ending Stamina")
            total_drain    = get_col(rows.head(1), "total_drained", "Total Stamina Drained")
            mins_played    = get_col(rows.head(1), "minutes_played", "Minutes Played")

            if not ending_stamina.empty and ending_stamina.iloc[0] > 0:
                last_ending = float(ending_stamina.iloc[0])
                fatigue_level = 100.0 - last_ending
                # Recovery: 1 day per 10% drain, matches typically 5–7 days apart
                days_to_recover = fatigue_level / 10.0
                days_available  = 6   # Typical week-to-week gap
                recovery_pct    = min(1.0, days_available / max(1, days_to_recover))
                starting_stamina = min(100.0, last_ending + (fatigue_level * recovery_pct))
            elif not total_drain.empty and total_drain.iloc[0] > 0:
                drain = float(total_drain.iloc[0])
                fatigue_level = drain
                days_available = 6
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