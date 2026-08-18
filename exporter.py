"""
PLOFA 26/27 — EXPORTER MODULE
================================
exporter.py

Exports match data in multiple formats AND generates
professional visualizations using matplotlib + mplsoccer.

Outputs:
    Data Exports:
        → Multi-sheet .xlsx  (primary output, full StatsBomb-level stats)
        → .csv               (flat player stats)
        → .json              (structured match data)

    Visualizations (saved as .png):
        → Shot Map           (mplsoccer pitch, all shots colored by outcome + xG bubble)
        → Pass Network       (mplsoccer pitch, player nodes + edge thickness by volume)
        → xG Timeline        (cumulative xG race chart across 90 mins)
        → Player Heatmap     (touch/action density per player)
        → Match Summary Card (scoreline, key stats, goal timeline — clean infographic)
        → Pressure Map       (team pressing zones heatmap)

Usage:
    from exporter import PLOFAExporter
    exporter = PLOFAExporter(result, player_stats)
    exporter.export_all("outputs/matchday_1_HartCity_vs_Thornfield")
"""

from __future__ import annotations
import os
import tempfile
import json
import random
from datetime import date
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from scipy.stats import poisson as scipy_poisson
import pandas as pd
import matplotlib

# Advanced valuation metrics
from advanced_valuation import (
    get_valuation_engine,
    create_action_from_event,
    ActionSnapshot,
)
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from mplsoccer import Pitch, VerticalPitch, FontManager

from match_engine import MatchEvent, EventType, SituationType, MatchResult, MatchConfig
from player_dna import PlayerProfile
from player_maps import plot_player_dashboard
from pass_network import PassMatrix, ChanceMatrix


# ─────────────────────────────────────────────
# PLOFA VISUAL IDENTITY
# ─────────────────────────────────────────────

class PLOFAStyle:
    """Visual constants for PLOFA brand identity."""

    # Primary palette
    BG_DARK      = "#0D0D0D"
    BG_CARD      = "#1A1A2E"
    BG_PANEL     = "#16213E"
    ACCENT_GOLD  = "#F5C518"
    ACCENT_TEAL  = "#00B4D8"
    ACCENT_RED   = "#E63946"
    ACCENT_GREEN = "#2DC653"
    TEXT_PRIMARY = "#F0F0F0"
    TEXT_MUTED   = "#8A8A8A"
    PITCH_GREEN  = "#2D5016"
    PITCH_LINE   = "#FFFFFF"

    # Shot outcome colors
    GOAL_COLOR   = "#FFD700"
    SOT_COLOR    = "#00B4D8"
    MISS_COLOR   = "#E63946"
    BLOCK_COLOR  = "#8A8A8A"

    # Team color defaults (overridable)
    HOME_COLOR   = "#003087"
    AWAY_COLOR   = "#C8102E"

    @staticmethod
    def apply_dark_style():
        """Apply PLOFA dark theme to matplotlib."""
        plt.rcParams.update({
            "figure.facecolor":  PLOFAStyle.BG_DARK,
            "axes.facecolor":    PLOFAStyle.BG_PANEL,
            "axes.edgecolor":    PLOFAStyle.TEXT_MUTED,
            "axes.labelcolor":   PLOFAStyle.TEXT_PRIMARY,
            "text.color":        PLOFAStyle.TEXT_PRIMARY,
            "xtick.color":       PLOFAStyle.TEXT_MUTED,
            "ytick.color":       PLOFAStyle.TEXT_MUTED,
            "grid.color":        "#2A2A2A",
            "grid.alpha":        0.5,
            "font.family":       "DejaVu Sans",
            "axes.titlecolor":   PLOFAStyle.TEXT_PRIMARY,
        })


# ─────────────────────────────────────────────
# FINANCIAL ENGINE
# Computes match attendance, ticket prices & revenue
# ─────────────────────────────────────────────

class MatchFinancials:
    """
    Computes realistic match attendance, ticket pricing, and revenue
    based on:
      - Stadium capacity
      - Is it a derby? (near sell-out)
      - Team "bigness" (average squad overall rating — proxy for club stature)
      - Big 6 teams (clubs with highest squad values) draw bigger crowds
      - Ticket prices inspired by real-world Premier League averages
    """

    # Real-world inspired base ticket prices by match tier (in local currency)
    TIER_TICKET_PRICES = {
        "derby":         65.00,   # Local rivalry — premium pricing
        "big6_vs_big6":  70.00,   # Two top clubs — highest demand
        "big6_home":     55.00,   # Top club at home vs mid-table
        "big6_away":     45.00,   # Mid-table hosting a top club
        "standard":      38.00,   # Regular mid-table fixture
        "low_interest":  28.00,   # Bottom-half vs bottom-half
    }

    # Minimum and maximum attendance fill rates
    MIN_FILL_RATE = 0.55   # Even the worst match has some fans
    MAX_FILL_RATE = 0.985  # Sell-out

    @classmethod
    def compute(
        cls,
        stadium_capacity: int,
        is_derby: bool,
        home_team: str,
        away_team: str,
        home_avg_rating: float,
        away_avg_rating: float,
        big6_teams: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Compute attendance, ticket price, and revenue for a match.

        Parameters
        ----------
        stadium_capacity : int
            How many seats the stadium has.
        is_derby : bool
            Local rivalry match.
        home_team, away_team : str
            Team names.
        home_avg_rating, away_avg_rating : float
            Average overall rating of each team's squad (proxy for "bigness").
        big6_teams : set, optional
            Names of the "Big 6" clubs. Auto-detected if None.

        Returns
        -------
        dict with keys:
            attendance       : int  — how many people showed up
            fill_rate        : float — % of capacity filled
            ticket_price     : float — average ticket price
            money_gained_home: float — revenue for the home team
            match_tier       : str  — tier label for reference
        """
        if big6_teams is None:
            # Auto-detect: top 6 teams by squad rating
            # (fallback — caller should provide real list)
            all_ratings = [(home_team, home_avg_rating), (away_team, away_avg_rating)]
            all_ratings.sort(key=lambda x: x[1], reverse=True)
            big6_teams = set()

        home_is_big6 = home_team in big6_teams
        away_is_big6 = away_team in big6_teams

        # ── Determine match tier ─────────────────────────────────
        if is_derby:
            match_tier = "derby"
        elif home_is_big6 and away_is_big6:
            match_tier = "big6_vs_big6"
        elif home_is_big6:
            match_tier = "big6_home"
        elif away_is_big6:
            match_tier = "big6_away"
        else:
            # Use squad quality to differentiate standard vs low interest
            avg_both = (home_avg_rating + away_avg_rating) / 2.0
            if avg_both >= 80.0:
                match_tier = "standard"      # Two good teams
            elif avg_both >= 72.0:
                match_tier = "standard"      # Decent matchup
            else:
                match_tier = "low_interest"  # Two weaker teams

        # ── Base ticket price ────────────────────────────────────
        base_ticket = cls.TIER_TICKET_PRICES.get(match_tier, 38.00)

        # Add small random variation per match (realism)
        ticket_variation = random.uniform(0.90, 1.10)
        ticket_price = round(base_ticket * ticket_variation, 2)

        # ── Attendance fill rate ─────────────────────────────────
        # Starting base depends on tier
        fill_rate_map = {
            "derby":         0.95,
            "big6_vs_big6":  0.97,
            "big6_home":     0.90,
            "big6_away":     0.85,
            "standard":      0.78,
            "low_interest":  0.62,
        }
        base_fill = fill_rate_map.get(match_tier, 0.75)

        # Squad quality modifier: better teams = more fans
        avg_rating = (home_avg_rating + away_avg_rating) / 2.0
        quality_modifier = (avg_rating - 65.0) / 100.0  # -0.15 to +0.35
        base_fill += quality_modifier

        # Random match-day factor (weather, day of week, form, etc.)
        random_factor = random.uniform(-0.04, 0.06)

        fill_rate = max(cls.MIN_FILL_RATE, min(cls.MAX_FILL_RATE, base_fill + random_factor))
        attendance = int(round(stadium_capacity * fill_rate))

        # ── Revenue ─────────────────────────────────────────────
        # Home team keeps all gate receipts (standard in most leagues)
        money_gained_home = round(attendance * ticket_price, 2)

        return {
            "attendance":        attendance,
            "fill_rate":         round(fill_rate * 100, 1),
            "ticket_price":      ticket_price,
            "money_gained_home": money_gained_home,
            "match_tier":        match_tier,
            "stadium_capacity":  stadium_capacity,
        }


# ─────────────────────────────────────────────
# STAT ACCUMULATOR
# Converts raw event timeline → per-player stat dicts
# ─────────────────────────────────────────────

class StatAccumulator:
    """
    Reads the MatchResult event timeline and accumulates
    per-player statistics. This is the bridge between
    the simulation and the export layer.
    """

    def __init__(self, result: MatchResult, all_players: Dict[str, List[PlayerProfile]],
                 big6_teams: Optional[set] = None):
        self.result  = result
        self.config  = getattr(result, 'config', None)
        self.players = all_players   # {team_name: [PlayerProfile]}
        self.stats: Dict[str, Dict] = {}
        self.big6_teams = big6_teams or set()
        # Financial data (computed in _finalise)
        self.match_financials: Dict[str, Any] = {}
        self._build()

    def _build(self):
        """Initialise every player then walk the timeline."""
        config = self.result.config

        # Initialise stat dicts for all players
        for team, squad in self.players.items():
            for p in squad["starters"] + squad.get("substitutes", []):
                self.stats[p.name] = self._blank_stat(p, team)

        # Walk every event in the timeline
        for event in self.result.timeline:
            self._process_event(event)

        # Enrich every shot-map entry with a shot trajectory (destination) so
        # the PNG shot map can draw where each shot went and the Excel "Shot
        # Map" sheet carries the same coordinates. The engine logs the shot's
        # origin + outcome but not an explicit target, so the destination is
        # reconstructed geometrically from origin and outcome (on-target shots
        # finish in the goal mouth, misses go wide/high, blocks are cut short,
        # woodwork strikes end at a post).
        for name, s in self.stats.items():
            team = s.get("team", "")
            for sh in s.get("shot_map", []):
                sh["end_x"], sh["end_y"] = self._shot_trajectory(
                    sh["x"], sh["y"], sh.get("outcome", "miss"), team
                )

        # ── GK xGOT FACED & GOALS PREVENTED ────────────────────
        # Opta/StatsBomb definition: Goals Prevented = xGOT_faced - Goals Conceded
        # xGOT = sum of xG for all shots ON TARGET the GK faces (saves + goals
        # conceded + woodwork hits + penalties faced). These events are credited
        # to the attacking team; we map them to the defending team.
        self.team_xgot: Dict[str, float] = defaultdict(float)
        _XGOT_TYPES = (
            EventType.SHOT_ON_TARGET, EventType.HIT_WOODWORK,
            EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
        )
        for e in self.result.timeline:
            if e.event_type in _XGOT_TYPES:
                defending = config.away_team if e.team == config.home_team else config.home_team
                self.team_xgot[defending] += e.xg or 0.0

        # Opta telemetry analytics — movement/activity, line-breaking &
        # packing passes, errors→shot/goal chains, dribblers tackled and
        # game-state minutes. Derived values OVERRIDE the legacy random
        # physical estimates and the crude packing heuristics below, so
        # every number reflects what the simulation actually produced.
        from opta_analytics import OptaAnalytics
        self.opta = OptaAnalytics(self.result, self.players)
        self.opta.compute()
        for name, s in self.stats.items():
            ad = self.opta.player_data.get(name)
            if ad:
                s.update(ad)
                
        # Use ChanceCreationLedger as the single source of truth for creation stats
        from chance_creation import ChanceCreationLedger
        cc_ledger = ChanceCreationLedger(self.result.timeline).compute()
        for name, s in self.stats.items():
            cc_stats = cc_ledger.per_player.get(name)
            if cc_stats:
                s.update(cc_stats)

        # Count off-ball runs from position_log
        self._count_off_ball_runs()

        # Post-process derived stats
        self._finalise()

    def _shot_trajectory(self, x: float, y: float, outcome: str,
                         team: str) -> Tuple[float, float]:
        """
        Reconstruct where a shot went as a (end_x, end_y) pitch-coordinate
        destination, derived deterministically from the shot's origin and
        outcome so every shot on the map has a stable, realistic trajectory:

          goal/saved   → finishes inside the goal mouth
          woodwork     → ends at a post
          blocked      → travel cut short (stopped before the goal)
          miss         → ends wide of the goal or over the bar
        """
        attacks_right = team == self.result.config.home_team
        goal_x = 105.0 if attacks_right else 0.0
        sign = 1.0 if attacks_right else -1.0

        # Deterministic pseudo-random offset from the origin coordinates so
        # shots from similar positions don't all share one identical line.
        jitter = ((int(round(x * 10)) * 7919 + int(round(y * 10)) * 104729)
                  % 1000) / 1000.0 - 0.5   # in [-0.5, 0.5]

        if outcome in ("goal", "saved"):
            # Aim into the goal mouth, slightly biased away from the origin
            # side so on-target trajectories fan out naturally.
            aim = 34.0 + (34.0 - y) * 0.20 + jitter * 2.5
            end_y = max(30.6, min(37.4, aim))
            end_x = goal_x
        elif outcome == "woodwork":
            # Off the woodwork: finish at a post (top or bottom of the mouth).
            end_x = goal_x
            end_y = 37.4 if jitter < 0 else 30.6
        elif outcome == "blocked":
            # Blocked shot: ball is stopped before it reaches the goal.
            frac = 0.55 + abs(jitter) * 0.35
            end_x = x + (goal_x - x) * frac
            end_y = y + (34.0 - y) * frac
        else:
            # Off target: wide of a post or over the bar.
            wide = 3.5 + abs(jitter) * 12.0
            end_y = (37.4 + wide) if jitter >= 0 else (30.6 - wide)
            end_y = max(0.0, min(68.0, end_y))
            end_x = goal_x

        return round(end_x, 1), round(end_y, 1)

    def _blank_stat(self, p: PlayerProfile, team: str) -> Dict:
        return {
            # Identity
            "player":     p.name,
            "team":       team,
            "position":   p.position,
            "archetype":  p.dna.archetype,
            "age":        p.dna.age,
            "nationality": p.dna.nationality,
            "is_starter": p.is_starter,
            "minutes_played": p.dna.minutes_played,
            "specialties": ", ".join(p.dna.specialties),
            "preferred_foot": p.dna.preferred_foot,
            "is_set_piece_taker": p.dna.is_set_piece_taker,
            "sub_in":  p.sub_in_minute,
            "sub_out": p.sub_out_minute,
            "soul_archetype": p.dna.soul.profile.label if p.dna.soul else "",
            "soul_tier": p.dna.soul.tier if p.dna.soul else "",
            "soul_greatness": round(p.dna.soul.greatness_coefficient, 4) if p.dna.soul else 0.0,
            "soul_omega": p.dna.soul.pillars.omega_activated if p.dna.soul else False,

            "home_or_away": "home" if team == self.result.config.home_team else "away",
            "venue": self.result.config.venue,

            # Goals & assists
            "goals": 0, "assists": 0, "own_goals": 0,
            "open_play_goals": 0, "headed_goals": 0, "left_foot_goals": 0, "right_foot_goals": 0, 
            "open_play_assists": 0, "setpiece_assists": 0,
            "pen_goals": 0, "pen_missed": 0,

            # Shooting
            "shots_on_target": 0, "shots_off_target": 0,
            "shots_blocked_att": 0,
            "hit_woodwork": 0,   # Checkpoint 6: post/bar strikes
            "shots_inside_box": 0, "shots_outside_box": 0,
            "big_chances_scored": 0, "big_chances_missed": 0,
            "big_chances_received": 0,

            # xG / xA
            "xg": 0.0, "xa": 0.0,
            "xg_open_play": 0.0, "xg_setpiece": 0.0, "xg_penalty": 0.0,
            "xa_open_play": 0.0, "xa_setpiece": 0.0,
            "npxg": 0.0,

            # Shot map (list of dicts for visualization)
            "shot_map": [],

            # Passing
            "passes_attempted": 0, "passes_completed": 0,
            "short_passes_att": 0, "short_passes_comp": 0,
            "long_passes_att": 0, "long_passes_comp": 0,
            "progressive_passes": 0, "passes_own_third": 0,
            "passes_mid_third": 0, "passes_final_third": 0,
            "passes_opp_box": 0, "shot_assists": 0,
            "through_balls_att": 0, "through_balls_comp": 0,
            "switches_of_play": 0, "passes_under_pressure": 0,
            "forward_passes": 0, "backward_passes": 0, "sideways_passes": 0,
            "line_breaking_passes": 0,
            "passes_right_foot": 0, "passes_left_foot": 0, "passes_head": 0,
            "chipped_passes": 0, "headed_passes": 0,

            # Crossing
            "crosses_att": 0, "crosses_comp": 0,
            "crosses_open_play_att": 0, "crosses_open_play_comp": 0,
            "crosses_corners_att": 0, "crosses_corners_comp": 0,
            "crosses_box_att": 0, "crosses_box_comp": 0,

            # Carrying
            "carries": 0, "progressive_carries": 0,
            "carries_own_half": 0, "carries_opp_half": 0,
            "carry_distance": 0.0,
            "progressive_carry_distance": 0.0,
            "longest_progressive_carry": 0.0,
            "final_third_carries": 0,
            "carries_opp_box": 0, "carries_own_box": 0,
            "runs_without_ball": 0,

            # Dribbling
            "dribbles_att": 0, "dribbles_comp": 0,
            "dribbles_own_half": 0, "dribbles_mid_third": 0, "dribbles_final_third": 0,
            "dribbles_to_box": 0, "dribble_distance": 0.0,

            # Chance creation
            "chances_created": 0, "big_chances_created": 0,
            "open_play_cc": 0, "setpiece_cc": 0,

            # Defending
            "tackles_att": 0, "tackles_won": 0,
            "interceptions": 0, "clearances": 0, "blocks": 0,
            "recoveries": 0, "ball_recoveries": 0,
            "pressures": 0, "press_success": 0,
            "aerial_duels_att": 0, "aerial_duels_won": 0,
            "ground_duels_att": 0, "ground_duels_won": 0,
            "last_man_tackles": 0, "dribbled_past": 0,
            "interceptions_def_third": 0, "interceptions_mid_third": 0, "interceptions_att_third": 0,
            "recoveries_def_third": 0, "recoveries_mid_third": 0, "recoveries_att_third": 0,

            # GK
            "saves": 0, "goals_conceded": 0, "high_claims": 0,
            "punches": 0, "sweeper_actions": 0,
            "saves_inside_box": 0, "saves_outside_box": 0,
            "goalline_saves": 0, "clean_sheet": False,
            "xgot_faced": 0.0, "goals_prevented": 0.0,

            # Discipline
            "fouls_committed": 0, "fouls_won": 0,
            "yellow_cards": 0, "red_cards": 0,
            "offsides": 0,

            # Physical
            "sprints": 0, "high_speed_sprints": 0,
            "distance_covered": 0.0, "top_speed": 0.0,
            "touches": 0, "touches_own_third": 0,
            "touches_mid_third": 0, "touches_final_third": 0,
            "touches_opp_box": 0,
            "standing_seconds": 0.0, "walking_seconds": 0.0,
            "jogging_seconds": 0.0, "running_seconds": 0.0,
            "sprinting_seconds": 0.0, "runs": 0,

            # Turnovers / possession
            "turnovers": 0, "bad_touches": 0, "dispossessed": 0,
            "possession_won": 0, "possession_lost": 0,

            # Advanced
            "sca": 0, "gca": 0,
            "packing_passes": 0, "packing_dribbles": 0, "total_packing": 0,
            "zone14_entries": 0,
            "deep_completions": 0, "progressive_pass_distance": 0.0,
            "xT": 0.0, "gpa": 0.0, "pva": 0.0, "epa": 0.0,

            # Errors → shot/goal chains (opta_analytics)
            "errors": 0, "errors_leading_to_shot": 0, "errors_leading_to_goal": 0,

            # Dribbler tackles (opta_analytics)
            "dribblers_tackled": 0, "dribbles_against": 0,

            # Game-state minutes (opta_analytics)
            "minutes_level": 0, "minutes_ahead": 0, "minutes_behind": 0,

            # Match result
            "match_result": "no_decision",
            "clean_sheet": False,
            "is_mvp": False,
            "rating": 6.0,

            # DNA attributes (for reference)
            "dna_overall": round(p.dna.overall_rating, 1),
            "dna_pace":    round(p.dna.physical.pace, 1),
            "dna_finishing": round(p.dna.technical.finishing, 1),
            "dna_passing": round(p.dna.passing.short_passing, 1),
            "dna_defending": round(p.dna.defending.tackling, 1),
            "dna_vision":  round(p.dna.mental.vision, 1),
            "dna_composure": round(p.dna.mental.composure, 1),
        }

    def _add_xg(self, actor: Dict, xg: float, situation: Optional[SituationType]):
        """
        Route a shot's xG into the total AND the breakdown buckets so:
            xg_penalty + xg_open_play + xg_setpiece == xg
            xg_open_play + xg_setpiece == npxg == xg - xg_penalty
        always hold. Previously the breakdown fields only accumulated on
        GOAL/PENALTY_SCORED events while "xg" accumulated on every shot
        outcome (saved, blocked, off target, woodwork, missed penalty) —
        summed over different event populations, so they could never match.
        """
        actor["xg"] += xg
        if situation == SituationType.PENALTY:
            actor["xg_penalty"] += xg
        else:
            actor["npxg"] += xg
            if situation in (SituationType.CORNER, SituationType.DIRECT_FREEKICK,
                             SituationType.CROSSED_FREEKICK):
                actor["xg_setpiece"] += xg
            else:
                actor["xg_open_play"] += xg

    def _process_event(self, e: MatchEvent):
        """Route each event to its stat update."""
        actor = self.stats.get(e.player)
        if not actor:
            return

        # ── GOALS & ASSISTS ───────────────────────────────────
        if e.event_type == EventType.GOAL:
            actor["goals"] += 1
            actor["shots_on_target"] += 1
            self._add_xg(actor, e.xg, e.situation)
            is_pen = e.situation == SituationType.PENALTY
            is_header = e.body_part == "head"
            is_left_foot = e.body_part == "left"
            is_right_foot = e.body_part == "right"
            if is_pen:
                actor["pen_goals"] += 1
                actor["xg_penalty"] += e.xg
            else:
                actor["open_play_goals"] += 1
                actor["npxg"] += e.xg
                if e.situation == SituationType.CORNER:
                    actor["xg_setpiece"] += e.xg
                else:
                    actor["xg_open_play"] += e.xg
            if is_header:
                actor["headed_goals"] += 1
            if is_left_foot:
                actor["left_foot_goals"] += 1
            if is_right_foot:
                actor["right_foot_goals"] += 1
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.home_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.away_team)
            )

            if is_box:
                actor["shots_inside_box"] += 1
            else:
                actor["shots_outside_box"] += 1
            actor["shot_map"].append({
                "x": e.location_x or 90, "y": e.location_y or 34,
                "outcome": "goal", "xg": e.xg,
                "body_part": e.body_part, "situation": e.situation.value if e.situation else "open_play"
            })
            actor["big_chances_scored"] += 1 if e.metadata.get("is_big_chance") else 0

            # Assist
            if e.secondary_player:
                assistant = self.stats.get(e.secondary_player)
                if assistant:
                    assistant["assists"] += 1
                    assistant["gca"] += 1
                    if e.situation == SituationType.PENALTY:
                        pass  # No assist for penalties typically
                    elif e.situation in (SituationType.CORNER, SituationType.CROSSED_FREEKICK,
                                         SituationType.DIRECT_FREEKICK):
                        assistant["setpiece_assists"] += 1
                        # NOTE: A goal assist is NOT also a key pass — the
                        # ChanceCreationLedger (single source of truth) uses
                        # the Opta strict convention where Assists + Key
                        # Passes = Chance Created (mutually exclusive). The
                        # ledger overwrites these values anyway; this line is
                        # kept consistent with that convention.
                        assistant["xa"] += e.xg
                        assistant["xa_setpiece"] += e.xg
                    else:
                        assistant["open_play_assists"] += 1

        elif e.event_type == EventType.PENALTY_SCORED:
            actor["goals"] += 1
            actor["pen_goals"] += 1
            self._add_xg(actor, 0.79, SituationType.PENALTY)
            actor["shots_on_target"] += 1
            actor["shot_map"].append({"x": 94, "y": 34, "outcome": "goal",
                                       "xg": 0.79, "body_part": "foot", "situation": "penalty"})

        elif e.event_type == EventType.PENALTY_MISSED:
            actor["pen_missed"] += 1
            actor["shots_on_target"] += 1
            self._add_xg(actor, 0.79, SituationType.PENALTY)
            actor["shot_map"].append({"x": 94, "y": 34, "outcome": "saved",
                                       "xg": 0.79, "body_part": "foot", "situation": "penalty"})

        # ── SHOTS ────────────────────────────────────────────
        elif e.event_type == EventType.SHOT_ON_TARGET:
            actor["shots_on_target"] += 1
            self._add_xg(actor, e.xg, e.situation)
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.home_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.away_team)
            )
            if is_box:
                actor["shots_inside_box"] += 1
            else:
                actor["shots_outside_box"] += 1
            actor["shot_map"].append({
                "x": e.location_x or 90, "y": e.location_y or 34,
                "outcome": "saved", "xg": e.xg,
                "body_part": e.body_part, "situation": e.situation.value if e.situation else "open_play"
            })
            actor["big_chances_received"] += 1 if e.metadata.get("is_big_chance") else 0

        elif e.event_type == EventType.SHOT_OFF_TARGET:
            actor["shots_off_target"] += 1
            self._add_xg(actor, e.xg, e.situation)
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.home_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.away_team)
            )
            if is_box:
                actor["shots_inside_box"] += 1
            else:
                actor["shots_outside_box"] += 1
            actor["shot_map"].append({
                "x": e.location_x or 88, "y": e.location_y or 30,
                "outcome": "miss", "xg": e.xg,
                "body_part": e.body_part or "foot", "situation": e.situation.value if e.situation else "open_play"
            })

        elif e.event_type == EventType.SHOT_BLOCKED:
            actor["shots_blocked_att"] += 1
            self._add_xg(actor, e.xg, e.situation)
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.home_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.away_team)
            )
            if is_box:
                actor["shots_inside_box"] += 1
            else:
                actor["shots_outside_box"] += 1
            actor["shot_map"].append({
                "x": e.location_x or 87, "y": e.location_y or 32,
                "outcome": "blocked", "xg": e.xg,
                "body_part": e.body_part or "foot", "situation": e.situation.value if e.situation else "open_play"
            })
            # Blocker
            if e.secondary_player:
                blocker = self.stats.get(e.secondary_player)
                if blocker:
                    blocker["blocks"] += 1

        elif e.event_type == EventType.HIT_WOODWORK:
            actor["hit_woodwork"] += 1
            self._add_xg(actor, e.xg, e.situation)
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.home_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.away_team)
            )
            if is_box:
                actor["shots_inside_box"] += 1
            else:
                actor["shots_outside_box"] += 1
            actor["shot_map"].append({
                "x": e.location_x or 100, "y": e.location_y or 34,
                "outcome": "woodwork", "xg": e.xg,
                "body_part": e.body_part or "foot",
                "situation": e.situation.value if e.situation else "open_play"
            })

        # ── SAVES ─────────────────────────────────────────────
        elif e.event_type == EventType.SAVE:
            actor["saves"] += 1
            if e.metadata.get("goalline_save", False):
                actor["goalline_saves"] += 1
            is_box = e.location_x is not None and (
                (e.location_x >= 83.0 and actor.get("team") == self.result.config.away_team) or
                (e.location_x <= 22.0 and actor.get("team") == self.result.config.home_team)
            )
            if is_box:
                actor["saves_inside_box"] += 1
            else:
                actor["saves_outside_box"] += 1
            if e.metadata.get("type") == "high_claim":
                actor["high_claims"] += 1
            if e.metadata.get("penalty_save"):
                actor["saves"] += 0  # Already counted

        # ── GOAL KICKS (Counted as Passes) ────────────────────
        elif e.event_type == EventType.GOAL_KICK:
            actor["passes_attempted"] += 1
            success = e.outcome
            is_long = e.metadata.get("long_launch", False)
            if is_long:
                actor["long_passes_att"] += 1
                if success:
                    actor["long_passes_comp"] += 1
                    actor["passes_completed"] += 1
            else:
                actor["short_passes_att"] += 1
                if success:
                    actor["short_passes_comp"] += 1
                    actor["passes_completed"] += 1
                    
            if success:
                actor["passes_own_third"] += 1
                actor["forward_passes"] += 1

        # ── PASSES ────────────────────────────────────────────
        elif e.event_type == EventType.PASS:
            actor["passes_attempted"] += 1
            is_long = e.metadata.get("is_long", False)
            is_cross = e.metadata.get("cross", False)
            success = e.outcome
            # Checkpoint 12 — Opta exclusion: a delivery reclassified as a
            # Cross (geometric verdict) is NEVER a long pass even if it
            # covers >= 35 yd. It stays in the short-pass bucket so pass
            # totals are preserved; crossing buckets own the cross credit.
            if is_long and not is_cross:
                actor["long_passes_att"] += 1
                if success:
                    actor["long_passes_comp"] += 1
                    actor["passes_completed"] += 1
            else:
                actor["short_passes_att"] += 1
                if success:
                    actor["short_passes_comp"] += 1
                    actor["passes_completed"] += 1
            if not success:
                actor["turnovers"] += 1
                actor["possession_lost"] += 1

            # Pass under pressure (StatsBomb definition)
            if e.metadata.get("under_pressure", False):
                actor["passes_under_pressure"] += 1

            # Zone (by start location) — direction-aware for home/away
            x = e.location_x or 50
            is_home = self._is_home_player(actor, e)
            if self._own_third(x, is_home):
                if success: actor["passes_own_third"] += 1
            elif self._mid_third(x, is_home):
                if success: actor["passes_mid_third"] += 1
            else:
                if success: actor["passes_final_third"] += 1
                if self._opp_box(x, is_home) and success:
                    actor["passes_opp_box"] += 1

            # Pass direction — use metadata pass_advance (signed: + = forward) if available
            if "pass_advance" in (e.metadata or {}):
                pa = e.metadata["pass_advance"]
                if pa > 1:
                    actor["forward_passes"] += 1
                elif pa < -1:
                    actor["backward_passes"] += 1
                else:
                    actor["sideways_passes"] += 1
            elif e.location_x is not None and e.end_x is not None:
                dx = e.end_x - e.location_x
                if dx > 1:
                    actor["forward_passes"] += 1
                elif dx < -1:
                    actor["backward_passes"] += 1
                else:
                    actor["sideways_passes"] += 1

            # Pass body part
            bp = (e.metadata or {}).get("body_part", "")
            if bp == "right_foot":
                actor["passes_right_foot"] += 1
            elif bp == "left_foot":
                actor["passes_left_foot"] += 1
            elif bp == "head":
                actor["passes_head"] += 1

            # Deep completions: completed passes into the box
            if success and e.end_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._opp_box(e.end_x, is_home):
                    actor["deep_completions"] += 1
            # Zone 14 entries: passes that enter the zone between x=70-83, y=central
            # Zone 14 = central area just outside the box (the "hole")
            if success and e.end_x is not None:
                is_home = self._is_home_player(actor, e)
                x14_lo, x14_hi = (70.0, 83.0) if is_home else (22.0, 35.0)
                if x14_lo <= e.end_x <= x14_hi and 20 <= (e.end_y or 34) <= 48:
                    actor["zone14_entries"] += 1
            # Packing passes: passes that beat opponent lines
            # Simplified heuristic: a completed forward pass past the halfway line
            # that progresses >10m beats at least one opponent line
            if success and e.location_x is not None and e.end_x is not None:
                dx = e.end_x - e.location_x
                if dx > 10:
                    actor["packing_passes"] += 1

        elif e.event_type == EventType.PROGRESSIVE_PASS:
            actor["passes_attempted"] += 1
            actor["passes_completed"] += 1
            actor["progressive_passes"] += 1
            actor["forward_passes"] += 1  # Always forward by definition
            # Checkpoint 12 — Opta metric rule applies to EVERY pass type: a
            # progressive pass that covers >= 35 yd is a long pass too.
            if e.metadata.get("is_long", False) and not e.metadata.get("cross", False):
                actor["long_passes_att"] += 1
                actor["long_passes_comp"] += 1
            else:
                actor["short_passes_att"] += 1
                actor["short_passes_comp"] += 1
            actor["line_breaking_passes"] += 1
            dist = (e.end_x or 0) - (e.location_x or 0)
            actor["progressive_pass_distance"] += max(0, dist)

            # Pass body part
            bp = (e.metadata or {}).get("body_part", "")
            if bp == "right_foot":
                actor["passes_right_foot"] += 1
            elif bp == "left_foot":
                actor["passes_left_foot"] += 1
            elif bp == "head":
                actor["passes_head"] += 1

            # Deep completions for progressive passes too
            if e.end_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._opp_box(e.end_x, is_home):
                    actor["deep_completions"] += 1
            # Zone 14 entries
            if e.end_x is not None:
                is_home = self._is_home_player(actor, e)
                x14_lo, x14_hi = (70.0, 83.0) if is_home else (22.0, 35.0)
                if x14_lo <= e.end_x <= x14_hi and 20 <= (e.end_y or 34) <= 48:
                    actor["zone14_entries"] += 1
            # Packing passes
            if e.location_x is not None and e.end_x is not None:
                if e.end_x - e.location_x > 10:
                    actor["packing_passes"] += 1

            # Progressive passes under pressure
            if e.metadata.get("under_pressure", False):
                actor["passes_under_pressure"] += 1

        elif e.event_type == EventType.SWITCH_OF_PLAY:
            actor["passes_attempted"] += 1
            actor["passes_completed"] += 1
            actor["switches_of_play"] += 1
            # Checkpoint 12 — switches are long-range by definition, but the
            # geometric stamp is authoritative (a short switch is rare).
            if e.metadata.get("is_long", True):
                actor["long_passes_att"] += 1
                actor["long_passes_comp"] += 1
            else:
                actor["short_passes_att"] += 1
                actor["short_passes_comp"] += 1
            # Switch of play is a sideways/lateral pass by definition
            actor["sideways_passes"] += 1

            # Deep completions for switches too
            if e.end_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._opp_box(e.end_x, is_home):
                    actor["deep_completions"] += 1

        elif e.event_type == EventType.THROUGH_BALL:
            actor["through_balls_att"] += 1
            actor["forward_passes"] += 1  # Always forward by definition
            if e.outcome:
                actor["through_balls_comp"] += 1
                actor["passes_completed"] += 1
                actor["shot_assists"] += 1
                actor["sca"] += 1
                # Through balls into box = deep completions
                if e.end_x is not None:
                    is_home = self._is_home_player(actor, e)
                    if self._opp_box(e.end_x, is_home):
                        actor["deep_completions"] += 1
                # Packing passes: through balls beat lines by definition
                actor["packing_passes"] += 1
            actor["passes_attempted"] += 1

        # ── CARRIES ────────────────────────────────────────────
        elif e.event_type == EventType.CARRY:
            actor["carries"] += 1
            actor["touches"] += 1
            dist = 0.0
            if e.location_x and e.end_x:
                dist = abs(e.end_x - e.location_x)
            actor["carry_distance"] += dist
            is_prog = e.metadata.get("progressive", False) or dist > 10
            if is_prog and e.outcome:
                actor["progressive_carries"] += 1
                actor["progressive_carry_distance"] += dist
                if dist > actor["longest_progressive_carry"]:
                    actor["longest_progressive_carry"] = dist
            x = e.location_x or 50
            is_home = self._is_home_player(actor, e)
            if x < 52:
                actor["carries_own_half"] += 1
            else:
                actor["carries_opp_half"] += 1
            if self._final_third(x, is_home):
                actor["final_third_carries"] += 1
            end_x = e.end_x or x
            if self._opp_box(end_x, is_home):
                actor["carries_opp_box"] += 1
            elif self._opp_box(end_x, not is_home):
                actor["carries_own_box"] += 1

        # ── DRIBBLES ───────────────────────────────────────────
        elif e.event_type in (EventType.DRIBBLE_ATTEMPT, EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL):
            actor["dribbles_att"] += 1
            actor["touches"] += 1
            if e.outcome:
                actor["dribbles_comp"] += 1
            x = e.location_x or 50
            is_home = self._is_home_player(actor, e)
            if self._own_third(x, is_home):
                actor["dribbles_own_half"] += 1
            elif self._mid_third(x, is_home):
                actor["dribbles_mid_third"] += 1
            else:
                actor["dribbles_final_third"] += 1
            if self._opp_box(x, is_home) and e.outcome:
                actor["dribbles_to_box"] += 1
            dist = 0.0
            if e.location_x and e.end_x:
                dist = abs(e.end_x - e.location_x)
            actor["dribble_distance"] += dist
            if not e.outcome:
                actor["possession_lost"] += 1
                actor["dribbled_past"] += 1

            # Every dribble attempt also counts as a ground duel
            actor["ground_duels_att"] += 1
            if e.outcome:
                actor["ground_duels_won"] += 1

        # ── CROSSES ────────────────────────────────────────────
        elif e.event_type in (EventType.CROSS_ATTEMPT, EventType.CORNER_TAKEN):
            actor["crosses_att"] += 1
            is_corner = e.situation == SituationType.CORNER
            if is_corner:
                actor["crosses_corners_att"] += 1
            else:
                actor["crosses_open_play_att"] += 1
            x = e.location_x or 50
            is_home = self._is_home_player(actor, e)
            if self._opp_box(x, is_home):
                actor["crosses_box_att"] += 1
            if e.outcome:
                actor["crosses_comp"] += 1
                if is_corner:
                    actor["crosses_corners_comp"] += 1
                else:
                    actor["crosses_open_play_comp"] += 1
                if self._opp_box(x, is_home):
                    actor["crosses_box_comp"] += 1

        # ── DEFENSIVE EVENTS ───────────────────────────────────
        elif e.event_type == EventType.TACKLE_WON:
            actor["tackles_att"] += 1
            actor["tackles_won"] += 1
            actor["possession_won"] += 1
            # Tackle zone tracking (for last man tackles)
            if e.location_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._own_third(e.location_x, is_home):
                    actor["interceptions_def_third"] += 1
                elif self._mid_third(e.location_x, is_home):
                    actor["interceptions_mid_third"] += 1
                else:
                    actor["interceptions_att_third"] += 1

        elif e.event_type == EventType.TACKLE_LOST:
            actor["tackles_att"] += 1
            actor["dribbled_past"] += 1

        elif e.event_type == EventType.INTERCEPTION:
            actor["interceptions"] += 1
            actor["possession_won"] += 1
            if e.location_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._own_third(e.location_x, is_home):
                    actor["interceptions_def_third"] += 1
                elif self._mid_third(e.location_x, is_home):
                    actor["interceptions_mid_third"] += 1
                else:
                    actor["interceptions_att_third"] += 1

        elif e.event_type == EventType.CLEARANCE:
            actor["clearances"] += 1
            if e.metadata.get("headed", False):
                actor["aerial_duels_att"] += 1
                actor["aerial_duels_won"] += 1

        elif e.event_type == EventType.BLOCK:
            actor["blocks"] += 1

        elif e.event_type == EventType.RECOVERY:
            actor["recoveries"] += 1
            actor["possession_won"] += 1
            if e.location_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._own_third(e.location_x, is_home):
                    actor["recoveries_def_third"] += 1
                elif self._mid_third(e.location_x, is_home):
                    actor["recoveries_mid_third"] += 1
                else:
                    actor["recoveries_att_third"] += 1

        elif e.event_type == EventType.BALL_RECOVERY:
            actor["ball_recoveries"] += 1
            actor["recoveries"] += 1
            actor["possession_won"] += 1
            if e.location_x is not None:
                is_home = self._is_home_player(actor, e)
                if self._own_third(e.location_x, is_home):
                    actor["recoveries_def_third"] += 1
                elif self._mid_third(e.location_x, is_home):
                    actor["recoveries_mid_third"] += 1
                else:
                    actor["recoveries_att_third"] += 1

        # ── PRESSING ───────────────────────────────────────────
        elif e.event_type == EventType.PRESS:
            actor["pressures"] += 1
        elif e.event_type == EventType.PRESS_SUCCESS:
            actor["pressures"] += 1
            actor["press_success"] += 1

        # ── AERIAL DUELS ───────────────────────────────────────
        elif e.event_type == EventType.AERIAL_DUEL:
            actor["aerial_duels_att"] += 1
            if e.outcome:
                actor["aerial_duels_won"] += 1
            else:
                actor["dribbled_past"] += 1

        # ── GROUND DUELS ───────────────────────────────────────
        elif e.event_type == EventType.GROUND_DUEL:
            actor["ground_duels_att"] += 1
            if e.outcome:
                actor["ground_duels_won"] += 1

        # ── DISCIPLINE ─────────────────────────────────────────
        elif e.event_type == EventType.FOUL_COMMITTED:
            actor["fouls_committed"] += 1

        elif e.event_type == EventType.FOUL_WON:
            actor["fouls_won"] += 1

        elif e.event_type == EventType.YELLOW_CARD:
            # A card always follows its own FOUL_COMMITTED event, so the
            # foul is already counted — only the booking is added here.
            actor["yellow_cards"] += 1

        elif e.event_type == EventType.RED_CARD:
            actor["red_cards"] += 1

        # ── OFFSIDE ────────────────────────────────────────────
        elif e.event_type == EventType.OFFSIDE:
            actor["offsides"] += 1
            actor["possession_lost"] += 1

        # ── PHYSICAL ───────────────────────────────────────────
        elif e.event_type == EventType.SPRINT:
            actor["sprints"] += 1
            if e.metadata.get("high_intensity"):
                actor["high_speed_sprints"] += 1

        # ── TOUCHES ────────────────────────────────────────────
        elif e.event_type == EventType.BALL_RECEIPT:
            actor["touches"] += 1
            x = e.location_x or 50
            is_home = self._is_home_player(actor, e)
            if self._own_third(x, is_home):
                actor["touches_own_third"] += 1
            elif self._mid_third(x, is_home):
                actor["touches_mid_third"] += 1
            else:
                actor["touches_final_third"] += 1
                if self._opp_box(x, is_home):
                    actor["touches_opp_box"] += 1

        elif e.event_type == EventType.MISCONTROL:
            actor["bad_touches"] += 1
            actor["possession_lost"] += 1

        elif e.event_type == EventType.DISPOSSESSED:
            actor["dispossessed"] += 1
            actor["possession_lost"] += 1

        # ── MISC ───────────────────────────────────────────────
        elif e.event_type == EventType.OWN_GOAL:
            actor["own_goals"] += 1

    def _compute_match_financials(self):
        """
        Compute attendance, ticket price, and home revenue for this match.
        Uses MatchFinancials engine with team quality as bigness proxy.
        """
        config = self.result.config

        # Compute average squad rating for each team
        home_ratings = []
        away_ratings = []
        for name, s in self.stats.items():
            if s["team"] == config.home_team and s["dna_overall"] > 0:
                home_ratings.append(s["dna_overall"])
            elif s["team"] == config.away_team and s["dna_overall"] > 0:
                away_ratings.append(s["dna_overall"])

        home_avg = np.mean(home_ratings) if home_ratings else 75.0
        away_avg = np.mean(away_ratings) if away_ratings else 75.0

        self.match_financials = MatchFinancials.compute(
            stadium_capacity=config.stadium_capacity,
            is_derby=config.is_derby,
            home_team=config.home_team,
            away_team=config.away_team,
            home_avg_rating=home_avg,
            away_avg_rating=away_avg,
            big6_teams=self.big6_teams,
        )

    def _count_off_ball_runs(self):
        """Count significant off-ball movements from position_log."""
        position_log = getattr(self.result, "position_log", [])
        if not position_log:
            return

        RUN_THRESHOLD = 15.0
        prev_positions: Dict[str, Tuple[float, float]] = {}
        for frame in position_log:
            for side in ("home", "away"):
                for p in frame.get(side, []):
                    name = p["player"]
                    x, y = p.get("x", 50.0), p.get("y", 34.0)
                    if name in prev_positions:
                        px, py = prev_positions[name]
                        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                        # Telemetry covers BOTH teams; an accumulator built
                        # for a single squad (or any partial player set)
                        # simply skips the names it isn't tracking.
                        if dist >= RUN_THRESHOLD and name in self.stats:
                            self.stats[name]["runs_without_ball"] += 1
                    prev_positions[name] = (x, y)

    def _finalise(self):
        """Post-process derived stats after all events are accumulated."""
        config = self.result.config

        # ── Compute match financials ────────────────────────────
        self._compute_match_financials()

        # ── PRESS SUCCESS INFERENCE ──
        # In StatsBomb, PRESS_SUCCESS is not a separate event. Instead,
        # pressure success is inferred from what happens next: if a PRESS
        # event from player X is followed by that same player winning a
        # TACKLE_WON, INTERCEPTION, RECOVERY, or BALL_RECOVERY within
        # the same sequence, the press is considered successful.
        # We compute this by scanning the timeline for PRESS events and
        # checking if the same player wins the ball shortly after.
        self._compute_press_success()

        # ── ADVANCED METRICS: xT, PVA & EPA ─────────────────────
        # Calculate Expected Threat and Possession Value Added for all players
        # These metrics measure ball progression value beyond goals/assists
        self._compute_advanced_metrics()

        # Checkpoint 7: soul bonus stats disabled — player stats reflect
        # only what actually happened in the simulation, no artificial padding.

        # Determine result
        hg = self.result.state.home_goals
        ag = self.result.state.away_goals
        home_result = "win" if hg > ag else ("draw" if hg == ag else "loss")
        away_result = "loss" if hg > ag else ("draw" if hg == ag else "win")

        # Determine MVP: highest combined xg + xa + goals*2 + assists
        mvp_score = {}
        for name, s in self.stats.items():
            mvp_score[name] = s["goals"] * 2.0 + s["assists"] * 1.0 + s["xg"] * 0.5 + s["xa"] * 0.3

        mvp_name = max(mvp_score, key=mvp_score.get) if mvp_score else None

        for name, s in self.stats.items():
            team = s["team"]

            # Match result
            if team == config.home_team:
                s["match_result"] = home_result
            else:
                s["match_result"] = away_result

            # Clean sheet
            goals_conceded = ag if team == config.home_team else hg
            s["clean_sheet"] = (goals_conceded == 0) and (s["minutes_played"] >= 60)

            # GK goals conceded
            if s["position"] == "GK":
                s["goals_conceded"] = goals_conceded
                shots_faced = s["saves"] + goals_conceded
                s["save_pct"] = round(s["saves"] / shots_faced * 100, 1) if shots_faced > 0 else 0.0
                xgot = round(self.team_xgot.get(team, 0.0), 3)
                s["xgot_faced"] = xgot
                prevented = round(xgot - goals_conceded, 3)
                s["goals_prevented"] = max(0.0, prevented)

            # Pass accuracy — DERIVED from the simulated events only.
            att = s["passes_attempted"]
            comp = s["passes_completed"]

            if att > 0:
                s["pass_accuracy"] = round(comp / att * 100, 1)
            else:
                s["pass_accuracy"] = 0.0
                
            s["short_pass_acc"] = round(
                s["short_passes_comp"] / s["short_passes_att"] * 100, 1
            ) if s["short_passes_att"] > 0 else 0.0
            s["long_pass_acc"] = round(
                s["long_passes_comp"] / s["long_passes_att"] * 100, 1
            ) if s["long_passes_att"] > 0 else 0.0

            # Dribble success %
            s["dribble_success_pct"] = round(
                s["dribbles_comp"] / s["dribbles_att"] * 100, 1
            ) if s["dribbles_att"] > 0 else 0.0

            # Tackle success %
            s["tackle_success_pct"] = round(
                s["tackles_won"] / s["tackles_att"] * 100, 1
            ) if s["tackles_att"] > 0 else 0.0

            # Aerial success %
            s["aerial_success_pct"] = round(
                s["aerial_duels_won"] / s["aerial_duels_att"] * 100, 1
            ) if s["aerial_duels_att"] > 0 else 0.0

            # Cross accuracy
            s["cross_acc"] = round(
                s["crosses_comp"] / s["crosses_att"] * 100, 1
            ) if s["crosses_att"] > 0 else 0.0

            # Shot conversion
            total_shots = s["shots_on_target"] + s["shots_off_target"] + s["shots_blocked_att"]
            s["shot_conversion"] = round(
                s["goals"] / total_shots * 100, 1
            ) if total_shots > 0 else 0.0

            # Last man tackles — heuristic: count tackles won in defensive third
            # where the defender had no other defenders behind them.
            # Simplified: count tackles_won that occur when x < 35 (own third)
            # as potential last-man situations. A more sophisticated version
            # would check PositionEngine for no teammates behind.
            s["last_man_tackles"] = 0  # Will be overridden if position engine data available

            # Physical estimates — distance_covered, sprints, high_speed_sprints
            # and top_speed are now derived from Opta telemetry (opta_analytics:
            # activity-time model over per-minute position logs + DNA). The
            # legacy random estimates were removed so the numbers are modelled,
            # not drawn from a hat.

            player_obj = self._find_player(name)
            if player_obj:

                # Touches estimate — total_t may be higher than the sum of
                # zone-tracked touches because POSSESSION_SEQUENCE events
                # increment touches but don't carry a location_x.
                s["touches"] = max(
                    s["touches"],
                    int(s["passes_attempted"] * 1.3 + s["carries"] + s["dribbles_att"])
                )
                total_t = s["touches"]

                # Touch zone distribution is now tracked from REAL coordinates
                # on every BALL_RECEIPT, CARRY, DRIBBLE_SUCCESS, and DRIBBLE_FAIL
                # event (see _process_event). We trust this real distribution
                # and just rescale proportionally to match the estimated total_t,
                # since total_t includes POSSESSION_SEQUENCE touches that aren't
                # individually zone-tagged.
                real_zone_sum = (
                    s["touches_own_third"] + s["touches_mid_third"] + s["touches_final_third"]
                )

                if real_zone_sum > 0:
                    scale = total_t / real_zone_sum
                    s["touches_own_third"] = int(s["touches_own_third"] * scale)
                    s["touches_mid_third"] = int(s["touches_mid_third"] * scale)
                    s["touches_final_third"] = max(
                        0, total_t - s["touches_own_third"] - s["touches_mid_third"]
                    )

                # touches_opp_box is NOT scaled — it reflects actual
                # BALL_RECEIPT events in the box, not an estimate.

                # xT and PVA are now calculated properly in _compute_advanced_metrics()
                # These placeholders ensure the keys exist before that method runs
                s.setdefault("xT", 0.0)
                s.setdefault("pva", 0.0)
                s.setdefault("gpa", 0.0)
                s.setdefault("epa", 0.0)

                # Rating (fixed weighted-sum system)
                s["rating"] = self._calculate_rating(s, goals_conceded)

            # MVP
            s["is_mvp"] = (name == mvp_name)

        # Big chances missed
        for name, s in self.stats.items():
            s["big_chances_missed"] = max(0, s["big_chances_received"] - s["big_chances_scored"])

        # ── LAST MAN TACKLES ──
        # Last man tackles are tackles won by a defender when they are the
        # last line of defense before the goal. For CBs, most tackles are
        # last-man situations since they're the deepest outfield line.
        # For FBs/CDM, defensive third tackles count as last-man.
        # Scan the timeline for TACKLE_WON events with valid coordinates.
        for name, s in self.stats.items():
            if s["position"] in ("CB", "LB", "RB", "CDM", "GK"):
                last_man_count = 0
                for e in self.result.timeline:
                    if (e.event_type == EventType.TACKLE_WON 
                            and e.player == name):
                        loc_x = e.location_x
                        if loc_x is not None:
                            is_home = s["team"] == self.result.config.home_team
                            # CBs and GK: own half = last-man
                            if s["position"] in ("CB", "GK"):
                                own_half = loc_x < 52.5 if is_home else loc_x > 52.5
                                if own_half:
                                    last_man_count += 1
                            # FBs/CDM: defensive third = last-man
                            elif s["position"] in ("LB", "RB", "CDM"):
                                def_third = loc_x < 35.0 if is_home else loc_x > 70.0
                                if def_third:
                                    last_man_count += 1
                        else:
                            # No location data: use a heuristic
                            if s["position"] in ("CB", "GK"):
                                # Most CB tackles are last-man situations
                                last_man_count = max(1, int(s["tackles_won"] * 0.4))
                            elif s["position"] in ("LB", "RB", "CDM"):
                                # Some FB/CDM tackles are last-man
                                if s["tackles_won"] >= 3:
                                    last_man_count = max(1, int(s["tackles_won"] * 0.2))
                if last_man_count > 0:
                    s["last_man_tackles"] = last_man_count

    def _compute_press_success(self):
        """
        StatsBomb-style press success calculation.
        
        In StatsBomb, there is no PRESS_SUCCESS event type. Instead, pressure
        success is inferred from the event chain: if a player applies pressure
        (PRESS event) and then wins the ball (TACKLE_WON, INTERCEPTION, 
        RECOVERY, BALL_RECOVERY) within the next few events, that press is
        considered successful.
        
        This method scans the timeline for PRESS events and checks if the
        pressing player wins the ball within the next 5 events.
        """
        timeline = self.result.timeline
        for i, e in enumerate(timeline):
            if e.event_type == EventType.PRESS:
                presser = e.player
                # Look ahead up to 5 events for a ball-winning action by the same player
                for j in range(i + 1, min(i + 6, len(timeline))):
                    next_e = timeline[j]
                    if next_e.player == presser and next_e.event_type in (
                        EventType.TACKLE_WON,
                        EventType.INTERCEPTION,
                        EventType.RECOVERY,
                        EventType.BALL_RECOVERY,
                    ):
                        # Successful press! Credit the presser
                        actor = self.stats.get(presser)
                        if actor:
                            actor["press_success"] += 1
                        break
                    # If another team's event happens first, the press failed
                    if next_e.team != e.team:
                        break

    def _compute_advanced_metrics(self):
        """
        Calculate Expected Threat (xT), Possession Value Added (PVA),
        and Expected Points Added (EPA) for all players.
        
        xT measures ball progression value based on grid zones.
        PVA extends xT by considering defensive context (opponents bypassed).
        EPA measures how much an action shifts the team's expected points
        in the match based on the win probability model.
        
        These metrics identify:
        - Progressive defenders who skip midfield with long balls
        - Box-to-box carriers who drive possession forward
        - Creative playmakers who consistently find dangerous zones
        - Side-pass merchants who recycle possession without progressing
        - High-impact actions that materially change match outcome probability
        """
        
        valuation_engine = get_valuation_engine()
        timeline = self.result.timeline
        
        # Track actions per player for xT/PVA/EPA calculation
        player_actions: Dict[str, List[ActionSnapshot]] = defaultdict(list)
        player_opponents: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
        
        # Pre-compute cumulative xG for home/away at each event index,
        # plus the live score, so EPA can read the actual match state
        # at the moment each action happens.
        cum_xg_home: List[float] = [0.0]
        cum_xg_away: List[float] = [0.0]
        home_goals_live: List[int] = [0]
        away_goals_live: List[int] = [0]
        
        for ev in timeline:
            hx = cum_xg_home[-1]
            ax = cum_xg_away[-1]
            hg = home_goals_live[-1]
            ag = away_goals_live[-1]
            if ev.team == self.config.home_team and getattr(ev, "xg", 0.0):
                hx += ev.xg
                if ev.event_type == EventType.GOAL:
                    hg += 1
            elif ev.team == self.config.away_team and getattr(ev, "xg", 0.0):
                ax += ev.xg
                if ev.event_type == EventType.GOAL:
                    ag += 1
            cum_xg_home.append(hx)
            cum_xg_away.append(ax)
            home_goals_live.append(hg)
            away_goals_live.append(ag)
        
        for i, event in enumerate(timeline):
            if not hasattr(event, 'player') or not event.player:
                continue
            
            player_name = event.player
            
            # Convert event to ActionSnapshot for passes, carries, dribbles
            action = None
            action_type = None
            
            if event.event_type == EventType.PASS:
                action_type = "pass"
                action = create_action_from_event(event, action_type)
            elif event.event_type == EventType.CARRY:
                action_type = "carry"
                action = create_action_from_event(event, action_type)
            elif event.event_type in (EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL):
                action_type = "dribble"
                action = create_action_from_event(event, action_type)
            elif event.event_type == EventType.PROGRESSIVE_PASS:
                action_type = "pass"
                action = create_action_from_event(event, action_type)
            
            if action is not None:
                player_actions[player_name].append(action)
                
                # Calculate xT for this action
                xt_added = valuation_engine.calculate_xt_added(action)
                
                # Calculate PVA using actual opponent positions from position_log
                # when available, falling back to a heuristic otherwise.
                opponent_positions = self._estimate_opponent_positions(
                    event, action.start_x, action.start_y
                )
                player_opponents[player_name].append(opponent_positions)
                # Defensive line = deepest (furthest-behind) defender of the
                # front-most unit; passing beyond it credits a line break.
                def_line_x = min(p[0] for p in opponent_positions)
                pva_added = valuation_engine.calculate_pva_added(
                    action, opponent_positions, defensive_line_x=def_line_x
                )
                
                # Calculate EPA (Expected Points Added)
                is_home = event.team == self.config.home_team
                team_score_diff = (
                    home_goals_live[i] - away_goals_live[i]
                    if is_home
                    else away_goals_live[i] - home_goals_live[i]
                )
                cumulative_xg_for = cum_xg_home[i] if is_home else cum_xg_away[i]
                cumulative_xg_against = cum_xg_away[i] if is_home else cum_xg_home[i]
                epa_added = valuation_engine.calculate_epa_added(
                    action=action,
                    minute=event.minute,
                    team_score_diff=team_score_diff,
                    cumulative_xg_for=cumulative_xg_for,
                    cumulative_xg_against=cumulative_xg_against,
                    is_home=is_home,
                )
                
                # Accumulate to player stats
                player_stats = self.stats.get(player_name)
                if player_stats:
                    player_stats["xT"] = round(player_stats.get("xT", 0.0) + xt_added, 3)
                    player_stats["pva"] = round(player_stats.get("pva", 0.0) + pva_added, 3)
                    player_stats["epa"] = round(player_stats.get("epa", 0.0) + epa_added, 4)
        
        # Calculate GPA (Goal Probability Added) - aggregate PVA
        # GPA = sum of PVA across all of a player's successful actions.
        for name, stats in self.stats.items():
            stats["gpa"] = valuation_engine.calculate_gpa(
                player_actions.get(name, []),
                player_opponents.get(name, []),
            )
    
    def _estimate_opponent_positions(
        self, event, ball_x: float, ball_y: float
    ) -> List[Tuple[float, float]]:
        """
        Estimate opponent positions around the ball for PVA calculation.
        
        Prefers actual PositionEngine coordinates from position_log when
        available, falling back to a simplified defensive-formation heuristic
        only when position data is missing.
        """
        attacking_team = event.team if hasattr(event, 'team') else None
        
        # Try actual position engine data first
        if attacking_team and hasattr(self.result, 'position_log') and self.result.position_log:
            minute = getattr(event, 'minute', None)
            if minute is not None:
                opponent_team = (
                    self.config.away_team
                    if attacking_team == self.config.home_team
                    else self.config.home_team
                )
                opp_side = (
                    "home" if opponent_team == self.config.home_team else "away"
                )
                defenders = {"CB", "LB", "RB", "CDM", "CM"}
                
                # Find the closest frame at or before this minute
                frame = None
                for f in reversed(self.result.position_log):
                    if f["minute"] <= minute:
                        frame = f
                        break
                
                if frame is None and self.result.position_log:
                    frame = self.result.position_log[0]
                
                if frame:
                    opponent_positions = [
                        (p["x"], p["y"])
                        for p in frame.get(opp_side, [])
                        if p.get("position") in defenders
                    ]
                    if opponent_positions:
                        return opponent_positions
        
        # Fallback: simplified heuristic using formation estimates
        opponent_positions = []
        
        if ball_x > 70:
            def_line_x = ball_x - random.uniform(5, 12)
            lateral_spread = 15
        elif ball_x > 40:
            def_line_x = ball_x - random.uniform(8, 18)
            lateral_spread = 25
        else:
            def_line_x = ball_x + random.uniform(5, 15)
            lateral_spread = 30
        
        center_y = 34.0
        
        opponent_positions.append((def_line_x, center_y - 4))
        opponent_positions.append((def_line_x, center_y + 4))
        opponent_positions.append((def_line_x - 3, center_y - 12))
        opponent_positions.append((def_line_x - 3, center_y + 12))
        opponent_positions.append((def_line_x + 10, center_y))
        opponent_positions.append((def_line_x + 8, center_y - 8))
        opponent_positions.append((def_line_x + 8, center_y + 8))
        
        return opponent_positions

    def _find_player(self, name: str) -> Optional[PlayerProfile]:
        for squad in self.players.values():
            for p in squad.get("starters", []) + squad.get("substitutes", []):
                if p.name == name:
                    return p
        return None

    def _calculate_rating(self, s: Dict, goals_conceded: int) -> float:
        r = 6.0
        pos = s["position"]
        minutes = s.get("minutes_played", 90)
        minutes_factor = max(0.35, min(1.0, minutes / 90.0))

        # Goals & assists
        r += s["goals"] * 1.0
        if s["goals"] >= 3: r += 0.5
        r += s["assists"] * 0.5
        r -= s["own_goals"] * 1.2

        # Shooting
        r += min(s["xg"] * 0.15, 0.3)
        r += min(s["shots_on_target"] * 0.04, 0.15)

        # Creativity
        r += min(s["shot_assists"] * 0.08, 0.5)
        r += min(s["big_chances_created"] * 0.25, 0.8)
        r += min(s["xa"] * 0.12, 0.3)

        # Passing
        r += min(s["passes_completed"] * 0.006, 0.35)
        r += min(s["progressive_passes"] * 0.04, 0.2)

        # Carrying
        r += min(s["progressive_carries"] * 0.03, 0.2)

        # Defending
        r += min(s["tackles_won"] * 0.06, 0.35)
        r += min(s["interceptions"] * 0.06, 0.3)
        r += min(s["clearances"] * 0.04, 0.25)
        r += min(s["blocks"] * 0.04, 0.15)
        r += min(s["recoveries"] * 0.02, 0.15)
        r += min(s["press_success"] * 0.02, 0.15)

        # Negatives
        r -= s["yellow_cards"] * 0.25
        if s["red_cards"] > 0:
            r -= 1.8
        r -= min(s["turnovers"] * 0.04, 0.35)
        r -= min(s["fouls_committed"] * 0.06, 0.35)
        r -= min(s["big_chances_missed"] * 0.1, 0.4)

        # GK special
        if pos == "GK":
            r += min(s["saves"] * 0.12, 1.2)
            if s.get("save_pct", 0) > 0:
                r += min(s["save_pct"] / 100 * 0.6, 0.6)
            r -= min(goals_conceded * 0.45, 1.2)
            gp = s.get("goals_prevented", 0)
            r += min(gp * 0.25, 0.75)
            if s["saves"] < 2: r -= 0.4
            if s["clean_sheet"] and s["saves"] >= 3: r += 0.5

        # Defenders: clean sheet
        elif pos in ("CB", "LB", "RB", "CDM"):
            if s["clean_sheet"]: r += 0.7
            if goals_conceded >= 4:
                r -= min((goals_conceded - 3) * 0.4, 1.2)

        # Result context
        res = s["match_result"]
        if res == "win":   r += 0.08
        elif res == "draw": r += 0.03
        elif res == "loss": r -= 0.08

        if s["is_mvp"]: r += 0.15

        # Minutes played adjustment — a 10-minute sub should not be rated
        # equally to a 90-minute starter, but their base must not collapse
        # to sub-3.0 either. Scale only the deviation from the 6.0 baseline.
        r = 6.0 + (r - 6.0) * minutes_factor

        # Personality variance — inconsistent players have wider swings.
        player_obj = self._find_player(s.get("player", ""))
        if player_obj is not None and getattr(player_obj, "dna", None) is not None:
            personality = getattr(player_obj.dna, "personality", None)
            if personality is not None:
                variance = personality.rating_variance_mult
                import random
                r += random.gauss(0, 0.35 * variance)

        return round(max(1.0, min(10.0, r)), 2)

    # ── DIRECTION-AWARE ZONE HELPERS ─────────────────────────────────────
    def _is_home_player(self, actor, event):
        team = actor.get("team") or getattr(event, "team", None)
        return team == self.result.config.home_team

    def _own_third(self, x, is_home):
        return x < 35.0 if is_home else x > 70.0

    def _mid_third(self, x, is_home):
        if is_home:
            return 35.0 <= x < 70.0
        return 35.0 < x <= 70.0

    def _final_third(self, x, is_home):
        if is_home:
            return x >= 70.0
        return x <= 35.0

    def _opp_box(self, x, is_home):
        return x >= 83.0 if is_home else x <= 22.0

    def to_dataframe(self) -> pd.DataFrame:
        """Return all player stats as a DataFrame."""
        rows = list(self.stats.values())
        df = pd.DataFrame(rows)
        # Round floats
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].round(3)
        return df


# ─────────────────────────────────────────────
# PLOFA EXPORTER
# ─────────────────────────────────────────────

class PLOFAExporter:
    """
    Master exporter. Call export_all() for everything.
    """

    def __init__(
        self,
        result: MatchResult,
        all_players: Dict[str, List[PlayerProfile]],
        home_color: str = None,
        away_color: str = None,
        sub_controller=None,   # SubstitutionController from squad_manager
        big6_teams: Optional[set] = None,
    ):
        self.result  = result
        self.players = all_players
        self.config  = result.config
        self.state   = result.state
        self.accumulator = StatAccumulator(result, all_players, big6_teams=big6_teams)
        self.df = self.accumulator.to_dataframe()
        self.sub_controller = sub_controller

        self.home_color = home_color or PLOFAStyle.HOME_COLOR
        self.away_color = away_color or PLOFAStyle.AWAY_COLOR

        self._seq_tracker = None   # lazily built in _get_sequence_tracker

    # ── DIRECTION-AWARE ZONE HELPERS ─────────────────────────────────────
    # Home attacks right (x=0 → x=105); away attacks left (x=105 → x=0).
    # Zone labels are from the player's OWN team's perspective so that
    # "own third / mid third / final third" are meaningful for both sides.
    def _is_home_player(self, actor, event):
        team = actor.get("team") or getattr(event, "team", None)
        return team == self.config.home_team

    def _own_third(self, x, is_home):
        return x < 35.0 if is_home else x > 70.0

    def _mid_third(self, x, is_home):
        if is_home:
            return 35.0 <= x < 70.0
        return 35.0 < x <= 70.0

    def _final_third(self, x, is_home):
        if is_home:
            return x >= 70.0
        return x <= 35.0

    def _opp_box(self, x, is_home):
        return x >= 83.0 if is_home else x <= 22.0

        PLOFAStyle.apply_dark_style()

    def _get_sequence_tracker(self):
        """Lazily build (once) the Opta-style SequenceTracker for this match."""
        if self._seq_tracker is None:
            from sequence_engine import SequenceTracker
            self._seq_tracker = SequenceTracker(
                self.config.home_team, self.config.away_team, self.result.timeline
            )
            self._seq_tracker.compute_metrics()
        return self._seq_tracker

    def _build_sequence_sheets(self):
        """
        Opta/StatsBomb-style possession-sequence analytics derived from the
        live timeline:
            attacking_styles_df : per-team macro attacking styles
                                  (Build-Up Attacks, Direct Attacks,
                                   Shot-Ending High Turnovers)
            shot_ending_df      : every shot-ending sequence with its build-up
                                  metrics (passes, time, progress, speed, width)
        """
        st = self._get_sequence_tracker()
        attacking_styles_df = pd.DataFrame(st.macro_rows())
        shot_ending_df = pd.DataFrame(st.shot_ending_rows())
        if not shot_ending_df.empty:
            shot_ending_df = shot_ending_df.sort_values(
                ["Team", "Minute"], ascending=[True, True]
            ).reset_index(drop=True)
        return attacking_styles_df, shot_ending_df

    def export_all(self, base_path: str):
        """
        Export everything to base_path.
        Creates directory if needed. Adds appropriate extensions.
        """
        os.makedirs(base_path, exist_ok=True)
        name = f"{self.config.home_team.replace(' ','_')}_vs_{self.config.away_team.replace(' ','_')}_MD{self.config.matchday}"

        print(f"\n📦 PLOFA Exporter — {self.config.home_team} vs {self.config.away_team}")
        print(f"   Season {self.config.season} | Matchday {self.config.matchday}")
        print(f"   Output: {base_path}/\n")

        # Print match financials
        fin = self.accumulator.match_financials
        if fin:
            print(f"   🏟️  Attendance: {fin['attendance']:,} / {fin['stadium_capacity']:,} "
                  f"({fin['fill_rate']}%) | "
                  f"Ticket: ${fin['ticket_price']:.2f} | "
                  f"Revenue: ${fin['money_gained_home']:,.2f}")
            print()

        # ── DATA EXPORTS ──────────────────────────────────────
        self.export_excel(f"{base_path}/{name}.xlsx")
        self.export_csv(f"{base_path}/{name}_players.csv")
        self.export_json(f"{base_path}/{name}.json")

        # ── VISUALIZATIONS ────────────────────────────────────
        self.plot_shot_map(f"{base_path}/{name}_shot_map.png")
        self.plot_pass_network(f"{base_path}/{name}_pass_network.png")
        self.plot_xg_timeline(f"{base_path}/{name}_xg_timeline.png")
        self.plot_momentum(f"{base_path}/{name}_momentum.png")
        self.plot_match_summary(f"{base_path}/{name}_summary.png")
        self.plot_pressure_map(f"{base_path}/{name}_pressure_map.png")
        self.plot_player_heatmap(f"{base_path}/{name}_player_heatmap.png")
        self.plot_soul_dashboards(base_path)

        print(f"\n✅ Export complete → {base_path}/")

    # ─────────────────────────────────────────
    # DATA EXPORTS
    # ─────────────────────────────────────────
    def _build_passing_sheets(self):
        """
        Build the three Opta/StatsBomb-style passing DataFrames, all
        derived from real timeline events (pass_network.py), not estimates:
            - Pass Combinations : every passer->receiver pair, attempted/
              completed/accuracy/avg distance/progressive count
            - Chance Combinations : every creator->shooter pair, chances
              created/big chances/xA/goals from that combo
            - Pass Profile : per-player summary incl. real average pass
              distance (not in the main Player Stats sheet)
        """
        home, away = self.config.home_team, self.config.away_team
        pm_home = PassMatrix.build(home, self.result.timeline)
        pm_away = PassMatrix.build(away, self.result.timeline)
        cm_home = ChanceMatrix.build(home, self.result.timeline)
        cm_away = ChanceMatrix.build(away, self.result.timeline)

        # ── Pass Combinations ──────────────────────────────────
        combo_rows = []
        for team, pm in [(home, pm_home), (away, pm_away)]:
            for row in pm.combo_rows():
                row["Team"] = team
                combo_rows.append(row)
        combo_df = pd.DataFrame(combo_rows)
        if not combo_df.empty:
            combo_df = combo_df[["Team", "Passer", "Receiver", "Attempted", "Completed",
                                  "Accuracy %", "Avg Distance (m)", "Progressive Passes"]]
            combo_df = combo_df.sort_values("Completed", ascending=False).reset_index(drop=True)

        # ── Chance Combinations ─────────────────────────────────
        chance_rows = []
        for team, cm in [(home, cm_home), (away, cm_away)]:
            for row in cm.rows():
                row["Team"] = team
                chance_rows.append(row)
        chance_df = pd.DataFrame(chance_rows)
        if not chance_df.empty:
            chance_df = chance_df[["Team", "Creator", "Shooter", "Chances Created",
                                    "Big Chances", "xA Generated", "Goals From Combo"]]
            chance_df = chance_df.sort_values("Chances Created", ascending=False).reset_index(drop=True)

        # ── Pass Profile (per player) ────────────────────────────
        profile_rows = []
        tax = self._build_pass_taxonomy()
        for name, s in self.accumulator.stats.items():
            t = tax.get(name, {})
            pm = pm_home if s["team"] == home else pm_away
            profile_rows.append({
                "Player": name,
                "Team": s["team"],
                "Position": s["position"],
                "Passes Attempted": s["passes_attempted"],
                "Passes Completed": s["passes_completed"],
                "Pass Accuracy %": s["pass_accuracy"],
                "Avg Pass Distance (m)": pm.player_avg_pass_distance(name),
                "Short Passes (2-bucket)": s["short_passes_att"],
                "Long Passes (2-bucket)": s["long_passes_att"],
                "Short Passes (3-bucket)": t.get("cn_short", 0),
                "Medium Passes (3-bucket)": t.get("cn_medium", 0),
                "Long Passes (3-bucket)": t.get("cn_long", 0),
                "Forward Passes": s["forward_passes"],
                "Backward Passes": s["backward_passes"],
                "Sideways Passes": s["sideways_passes"],
                "Chipped Passes": t.get("type_Chipped Pass") or t.get("type_chipped pass", 0),
                "Headed Passes": t.get("type_Headed Pass") or t.get("type_headed pass", 0),
                "Launches": sum(v for k, v in t.items() if k.startswith("type_") and "aunch" in k),
                "Flick-ons": t.get("type_Flick-on") or t.get("type_flick-on", 0),
                "Pull-backs": t.get("type_Pull-back") or t.get("type_pull-back", 0),
                "Lay-offs": t.get("type_Lay-off") or t.get("type_lay-off", 0),
                "Through Balls Att": s["through_balls_att"],
                "Through Balls Comp": s["through_balls_comp"],
                "Passes From Own Half": t.get("half_own", 0),
                "Passes From Opp Half": t.get("half_opposition", 0),
                "Passes From Def 3rd": t.get("third_defensive_third", 0),
                "Passes From Mid 3rd": t.get("third_middle_third", 0),
                "Passes From Final 3rd": t.get("third_final_third", 0),
                "Channel: Left": t.get("chan_left", 0),
                "Channel: Centre": t.get("chan_centre", 0),
                "Channel: Right": t.get("chan_right", 0),
                "Switches of Play": s["switches_of_play"],
                "Crosses Att": s["crosses_att"],
                "Crosses Comp": s["crosses_comp"],
                "Cross Acc %": s["cross_acc"],
                "Passes Under Pressure": s["passes_under_pressure"],
                "Shot Assists": s["shot_assists"],
                "xA": s["xa"],
                "Passes Right Foot": s["passes_right_foot"],
                "Passes Left Foot": s["passes_left_foot"],
                "Passes Head": s["passes_head"],
                "Runs to Opp Box (with ball)": s["carries_opp_box"],
                "Runs to Own Box (with ball)": s["carries_own_box"],
                "Runs Without Ball": s["runs_without_ball"],
                "Progressive Carry Distance": s["progressive_carry_distance"],
                "Longest Progressive Carry": s["longest_progressive_carry"],
            })
        profile_df = pd.DataFrame(profile_rows)
        if not profile_df.empty:
            profile_df = profile_df.sort_values("Passes Completed", ascending=False).reset_index(drop=True)

        # ── Team pass-type breakdown sheet ───────────────────────
        agg_home = self._base_type_taxonomy(home)
        agg_away = self._base_type_taxonomy(away)
        team_rows = [
            {"Pass Breakdown": label, home: agg_home.get(label, 0), away: agg_away.get(label, 0)}
            for label in sorted(set(agg_home) | set(agg_away))
        ]
        if team_rows:
            team_df = pd.DataFrame(team_rows)
        else:
            team_df = pd.DataFrame()

        return combo_df, chance_df, profile_df, team_df

    def _build_pass_taxonomy(self) -> Dict:
        """Per-player breakdown of the Opta-style pass classification stamps
        (Checkpoint 13) that event_chain wrote onto each pass event.

        Post-processing passes fix two known gaps in the upstream classifier:

        • HEADED PASS — the chain's foot-detector (_foot_for_pass) always
          returns right/left foot for normal passes, so the is_headed flag
          passed to classify_pass is never True and headed passes always
          fall through to "ground pass". We recover them by looking back
          up to 6 prior events for an AERIAL_DUEL or a preceding header-
          type action; if found and the current pass is forward, it is
          re-classified as a headed pass.

        • CHIPPED PASS — the cross detector only sets is_airborne for
          qualifying crosses. Non-cross lofted passes (a player chips the
          ball over a defender) therefore arrive at classify_pass with
          is_airborne=False and are stamped "ground pass". We detect them
          from pass_height in metadata: a pass with height "high" or
          "lofted" AND forward pass_advance > 0 is a chipped pass.
        """
        tax: Dict[str, Dict] = {}
        PASS_NAMES = {"PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
                      "GOAL_KICK", "CORNER_TAKEN", "FREEKICK_CROSS"}
        _AERIAL_TYPES = {
            EventType.AERIAL_DUEL,
            EventType.CORNER_TAKEN,
            EventType.FREEKICK_CROSS,
        }
        _HEADED_BODY = {"head"}

        # Build a quick lookup: event index → player name for the timeline
        # so we can efficiently check "did player X have an aerial action
        # immediately before this pass?"
        tl = self.result.timeline

        for idx, e in enumerate(tl):
            if e.event_type.name not in PASS_NAMES or not e.player:
                continue
            m = e.metadata or {}
            d = tax.setdefault(e.player, {})
            # length class (3-bucket)
            lc = m.get("length_class") or "medium"
            d["cn_" + lc] = d.get("cn_" + lc, 0) + 1
            # pass type — with post-hoc correction for headed / chipped
            pt = m.get("pass_type", "")
            # Look back for aerial / headed context for this passer
            is_headed_pass = False
            if pt and pt != "cross":
                look_start = max(0, idx - 6)
                for j in range(idx - 1, look_start - 1, -1):
                    prev = tl[j]
                    if prev.team != e.team:
                        break
                    if prev.player != e.player:
                        continue
                    if prev.event_type in _AERIAL_TYPES:
                        is_headed_pass = True
                        break
                    if prev.event_type == EventType.SHOT_ON_TARGET:
                        # If the player just headed a shot on target,
                        # any immediately following pass is a headed pass
                        if getattr(prev, "body_part", "") in _HEADED_BODY:
                            is_headed_pass = True
                            break
                    if getattr(prev, "body_part", "") in _HEADED_BODY:
                        is_headed_pass = True
                        break
                    # Stop looking back if we hit a non-aerial discrete event
                    if prev.event_type not in {
                        EventType.PASS, EventType.PROGRESSIVE_PASS,
                        EventType.SWITCH_OF_PLAY, EventType.THROUGH_BALL,
                        EventType.CARRY, EventType.DRIBBLE_SUCCESS,
                        EventType.DRIBBLE_ATTEMPT,
                    }:
                        break

            pass_height = (m.get("pass_height") or "").lower()
            pass_advance = m.get("pass_advance", 0)
            is_lofted_forward = pass_height in ("high", "lofted") and pass_advance > 0

            if is_headed_pass:
                key = "type_headed pass"
                actor = self.accumulator.stats.get(e.player)
                if actor is not None:
                    actor["headed_passes"] = actor.get("headed_passes", 0) + 1
            elif pt == "ground pass" and is_lofted_forward:
                key = "type_chipped pass"
                actor = self.accumulator.stats.get(e.player)
                if actor is not None:
                    actor["chipped_passes"] = actor.get("chipped_passes", 0) + 1
            elif pt:
                key = "type_" + pt
            else:
                key = None

            if key:
                d[key] = d.get(key, 0) + 1
            # origin half / third / channel
            for axis, prefix in (("start_half", "half_"), ("start_third", "third_"),
                                 ("pass_channel", "chan_")):
                val = m.get(axis)
                if val:
                    k = prefix + val
                    d[k] = d.get(k, 0) + 1
        return tax

    def _base_type_taxonomy(self, team: str) -> Dict:
        """Team-aggregated taxonomy (mirrors _build_pass_taxonomy at team level)."""
        agg: Dict[str, int] = {}
        PASS_NAMES = ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
                      "GOAL_KICK", "CORNER_TAKEN", "FREEKICK_CROSS")
        for e in self.result.timeline:
            if e.team != team or e.event_type.name not in PASS_NAMES:
                continue
            m = e.metadata or {}
            agg["count: " + (m.get("length_class") or "?")] = agg.get("count: " + (m.get("length_class") or "?"), 0) + 1
            pt = m.get("pass_type")
            if pt:
                agg["type: " + pt] = agg.get("type: " + pt, 0) + 1
            for axis, label in (("start_half", "from "), ("start_third", "third "),
                                ("pass_channel", "channel ")):
                val = m.get(axis)
                if val:
                    key = label + val
                    agg[key] = agg.get(key, 0) + 1
        return agg

    # ─────────────────────────────────────────
    # THIRD PROGRESSION + FOOTED EVENT LOG
    # ─────────────────────────────────────────
    # Zone-entry analytics: every pass is placed into a pitch third relative
    # to the team's own attacking direction. The HOME team attacks right
    # (toward x=105, its attacking goal), the AWAY team attacks left (toward
    # x=0). So for the home team the Final third is x≥70 / Mid is 35–70 /
    # Def is x<35, while for the away team it is the exact mirror.
    THIRD_NAMES = ["Defensive", "Mid", "Final"]
    _THIRD_PASS_TYPES = (
        EventType.PASS,
        EventType.PROGRESSIVE_PASS,
        EventType.SWITCH_OF_PLAY,
        EventType.THROUGH_BALL,
        EventType.CROSS_SUCCESS,
    )
    _FOOT_LABEL = {"right_foot": "Right", "left_foot": "Left",
                   "head": "Head", "other": "Other"}

    def _third_for_x(self, x: Optional[float], attacks_right: bool) -> str:
        """Return the third a coordinate sits in, relative to an attacking
        direction (home=right toward 105, away=left toward 0)."""
        if x is None:
            return "Mid"
        if attacks_right:
            if x < 35:
                return "Defensive"
            if x < 70:
                return "Mid"
            return "Final"
        # attacks left — the mirror image (own goal at x=105)
        if x > 70:
            return "Defensive"
        if x > 35:
            return "Mid"
        return "Final"

    def _build_third_progression(self):
        """
        Build Two DataFrames from the real pass events, both framed per-team
        using the team's OWN attacking direction (home attacks right, away
        attacks left):

            third_progression_df : per team × third —
                                    Entries (passes that end in the third)
                                    Completed Entries
                                    Exits (passes that start in the third)
            footed_pass_df       : every passing event with the foot used,
                                   from/to third, and what it entered.

        The foot itself is stamped on each event by event_chain as
        `body_part` ∈ {right_foot, left_foot, head}.
        """
        home = self.config.home_team
        away = self.config.away_team

        entries = defaultdict(lambda: defaultdict(int))    # team -> third : passes landing there
        comp_entries = defaultdict(lambda: defaultdict(int))
        exits = defaultdict(lambda: defaultdict(int))      # team -> third : passes starting there
        player_into = defaultdict(lambda: defaultdict(int))  # player -> third : passes entering it

        footed_rows = []
        for e in self.result.timeline:
            if e.event_type not in self._THIRD_PASS_TYPES:
                continue
            if e.location_x is None or e.end_x is None:
                continue

            attacks_right = e.team == home
            origin = self._third_for_x(e.location_x, attacks_right)
            dest = self._third_for_x(e.end_x, attacks_right)

            exits[e.team][origin] += 1
            entries[e.team][dest] += 1
            if e.outcome:
                comp_entries[e.team][dest] += 1
            if e.player:
                player_into[e.player][dest] += 1

            foot = (e.body_part or "").lower()
            foot_label = self._FOOT_LABEL.get(foot, foot or "n/a")
            footed_rows.append({
                "Minute": e.minute,
                "Team": e.team,
                "Passer": e.player or "",
                "Receiver": e.secondary_player or "",
                "Foot": foot_label,
                "Event": e.event_type.name,
                "From Third": origin,
                "To Third": dest,
                "Entered Final": dest == "Final",
                "Completed": bool(e.outcome),
                "From X": round(e.location_x, 1),
                "From Y": round(e.location_y or 0, 1),
                "To X": round(e.end_x, 1),
                "To Y": round(e.end_y or 0, 1),
            })

        # Team × Third summary
        prog_rows = []
        for team in (home, away):
            for third in self.THIRD_NAMES:
                prog_rows.append({
                    "Team": team,
                    "Attacks": "Right" if team == home else "Left",
                    "Third": third,
                    "Entries (Passes In)": entries[team][third],
                    "Completed Entries": comp_entries[team][third],
                    "Exits (Passes Out)": exits[team][third],
                })
        third_progression_df = pd.DataFrame(prog_rows)

        # Mid 3rd, final 3rd def 3rd entries and exits(passes) for home team (attacks right)
        # and away team (attacks left)
        team_third_summaries = {}
        for team in (home, away):
            for third in self.THIRD_NAMES:
                key = f"{team}_{third}"
                team_third_summaries[key] = {
                    "entries": entries[team][third],
                    "exits": exits[team][third],
                    "completed_entries": comp_entries[team][third],
                    "team": team,
                    "third": third,
                    "attacks": "Right" if team == home else "Left"
                }

        footed_pass_df = pd.DataFrame(footed_rows)
        if not footed_pass_df.empty:
            footed_pass_df = footed_pass_df.sort_values(
                ["Team", "Minute"], ascending=[True, True]
            ).reset_index(drop=True)

        # Per-player passes into the final third (the standout stat), framed
        # by each player's own attacking direction.
        player_third_rows = []
        for name, thirds in player_into.items():
            player_third_rows.append({
                "Player": name,
                "Team": self.accumulator.stats.get(name, {}).get("team", ""),
                "Into Final Third": thirds.get("Final", 0),
                "Into Mid Third": thirds.get("Mid", 0),
                "Into Defensive Third": thirds.get("Defensive", 0),
            })
        player_third_df = pd.DataFrame(player_third_rows)
        if not player_third_df.empty:
            player_third_df = player_third_df.sort_values(
                "Into Final Third", ascending=False
            ).reset_index(drop=True)

        return third_progression_df, footed_pass_df, player_third_df

    def export_third_progression_summary(self) -> pd.DataFrame:
        """
        Export detailed third progression analytics: Mid 3rd, final 3rd, def 3rd
        entries and exits for both home team (attacks right) and away team (attacks left),
        plus footed events.
        
        Returns:
            DataFrame with columns:
            - Team: Team name
            - Third: Defensive/Mid/Final third
            - Entries (Passes In): Number of passes entering this third
            - Completed Entries: Number of completed passes entering this third
            - Exits (Passes Out): Number of passes exiting this third
            - Footed Events: Breakdown by foot (Right, Left, Head, Other)
            - Key Passes: Number of key passes ending in final third
            - Pass Types: Breakdown by pass type (Regular, Progressive, Switch, Through, Cross)
        """
        home = self.config.home_team
        away = self.config.away_team
        
        # Get third progression data
        third_progression_df, footed_pass_df, player_third_df = self._build_third_progression()
        
        # Initialize result DataFrame
        result_rows = []
        
        # Add team third progression summary
        for team in (home, away):
            for third in self.THIRD_NAMES:
                entries = 0
                exits = 0
                completed_entries = 0
                
                for e in self.result.timeline:
                    if e.event_type not in self._THIRD_PASS_TYPES:
                        continue
                    if e.location_x is None or e.end_x is None:
                        continue
                    
                    attacks_right = e.team == home
                    origin = self._third_for_x(e.location_x, attacks_right)
                    dest = self._third_for_x(e.end_x, attacks_right)
                    
                    if e.team == team:
                        if dest == third:
                            entries += 1
                            if e.outcome:
                                completed_entries += 1
                        if origin == third:
                            exits += 1
                
                result_rows.append({
                    "Team": team,
                    "Attacks": "Right" if team == home else "Left",
                    "Third": third,
                    "Entries (Passes In)": entries,
                    "Completed Entries": completed_entries,
                    "Exits (Passes Out)": exits,
                })
        
        # Add footed events summary
        footed_summary = {}
        for _, row in footed_pass_df.iterrows():
            team = row["Team"]
            foot = row["Foot"]
            key = f"{team}_{foot}"
            if key not in footed_summary:
                footed_summary[key] = {
                    "Team": team,
                    "Foot": foot,
                    "Total Passes": 0,
                    "Completed Passes": 0,
                    "Key Passes": 0,
                    "Event Types": defaultdict(int)
                }
            footed_summary[key]["Total Passes"] += 1
            if row["Completed"]:
                footed_summary[key]["Completed Passes"] += 1
            if row["Entered Final"]:
                footed_summary[key]["Key Passes"] += 1
            footed_summary[key]["Event Types"][row["Event"]] += 1
        
        # Add footed events summary rows
        for key, summary in footed_summary.items():
            result_rows.append({
                "Team": summary["Team"],
                "Third": "Footed Events",
                "Foot": summary["Foot"],
                "Entries (Passes In)": summary["Total Passes"],
                "Completed Entries": summary["Completed Passes"],
                "Exits (Passes Out)": 0,  # Not tracked for footed events
                "Key Passes": summary["Key Passes"],
                "Event Types": str(dict(summary["Event Types"])),
                "Is Footed Event": True
            })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result_rows)
        
        # Sort by Team, Third, Foot
        result_df = result_df.sort_values(["Team", "Third", "Foot" if "Foot" in result_df.columns else ""], ascending=[True, True, True])
        
        return result_df

    def _process_third_progression_data(self, third_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process third progression data to show detailed statistics:
        - Mid 3rd entries and exits for home team (attacks right)
        - Final 3rd entries and exits for home team (attacks right)
        - Defensive 3rd entries and exits for home team (attacks right)
        - Mid 3rd entries and exits for away team (attacks left)
        - Final 3rd entries and exits for away team (attacks left)
        - Defensive 3rd entries and exits for away team (attacks left)
        
        Returns:
            DataFrame with processed third progression statistics
        """
        if third_df.empty:
            return pd.DataFrame()
        
        # Initialize result DataFrame with required columns
        result_rows = []
        
        # Add statistics for home team (attacks right)
        home = self.config.home_team
        
        # For each row in the third_df, extract relevant stats
        for _, row in third_df.iterrows():
            team = row["Team"]
            if team == home:
                attacks = "Right"
            else:
                attacks = "Left"
            
            # Get entries and exits for each third
            for third in self.THIRD_NAMES:
                entries = row.get(f"Entries ({third})", 0) if third in row else 0
                exits = row.get(f"Exits ({third})", 0) if third in row else 0
                
                result_rows.append({
                    "Team": team,
                    "Attacks": attacks,
                    "Third": third,
                    "Entries (Passes In)": entries,
                    "Exits (Passes Out)": exits,
                })
        
        result_df = pd.DataFrame(result_rows)
        
        # Sort by Team, Third
        result_df = result_df.sort_values(["Team", "Third"], ascending=[True, True])
        
        return result_df

    def _add_attacks_column(self, footed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the Attacks (Right/Left) column to footed events DataFrame
        based on the team's attacking direction (home attacks right, away attacks left).
        
        Returns:
            DataFrame with Attacks column added
        """
        if footed_df.empty:
            return footed_df
        
        home = self.config.home_team
        
        # Add Attacks column
        footed_df["Attacks"] = footed_df["Team"].apply(
            lambda team: "Right" if team == home else "Left"
        )
        
        return footed_df

    def plot_soul_dashboards(self, base_path: str):
        from player_maps import plot_player_dashboard
        from chance_creation import ChanceCreationLedger

        # Real key passes (the setup pass before each shot) for the gold
        # arrows on the pass maps — computed once, shared by every player.
        ledger = ChanceCreationLedger(self.result.timeline).compute()

        folder = os.path.join(base_path, "player_dashboards")
        os.makedirs(folder, exist_ok=True)
        
        for name, s in self.accumulator.stats.items():
            player_obj = self.accumulator._find_player(name)
            # Generate dashboard for soul players or top performers (e.g. rating > 7.5 or goals > 0)
            if player_obj and getattr(player_obj.dna, "soul", None) is not None:
                color = self.home_color if s["team"] == self.config.home_team else self.away_color
                safe_name = name.replace(" ", "_")
                plot_player_dashboard(
                    self.result.timeline, name, s["team"], s["position"],
                    os.path.join(folder, f"{safe_name}_dashboard.png"),
                    team_color=color, ledger=ledger
                )

    def export_excel(self, filepath: str, include_third_progression: bool = True, include_footed_events: bool = True):
        """Multi-sheet Excel export."""
        df = self.df.copy()

        # Drop shot_map column from main sheet (it's a list — not Excel-friendly)
        df_main = df.drop(columns=["shot_map"], errors="ignore")

        # Goal data
        goal_rows = []
        for i, g in enumerate(self.result.goals):
            goal_rows.append({
                "Minute": g.minute,
                "Team": g.team,
                "Scorer": g.player,
                "Assist": g.secondary_player or "",
                "Situation": g.situation.value if g.situation else "open_play",
                "xG": round(g.xg, 3),
                "Body Part": g.body_part or "",
                "Phase": g.phase.value if g.phase else "",
                "Game State": g.game_state.name if g.game_state else "",
                "Location X": round(g.location_x or 0, 1),
                "Location Y": round(g.location_y or 0, 1),
                "Is Big Chance": g.metadata.get("is_big_chance", False),
            })

        # Card data
        card_rows = []
        for c in self.result.cards:
            card_rows.append({
                "Minute": c.minute,
                "Team": c.team,
                "Player": c.player,
                "Card Type": "Red" if c.event_type == EventType.RED_CARD else "Yellow",
                "Reason": c.metadata.get("reason", "foul"),
                "Phase": c.phase.value if c.phase else "",
            })

        # Shot map
        shot_rows = []
        for name, s in self.accumulator.stats.items():
            for sh in s.get("shot_map", []):
                shot_rows.append({
                    "Player": name, "Team": s["team"], "Position": s["position"],
                    "x": sh["x"], "y": sh["y"],
                    "End X": sh.get("end_x", sh["x"]),
                    "End Y": sh.get("end_y", sh["y"]),
                    "Outcome": sh["outcome"], "xG": sh["xg"],
                    "Body Part": sh.get("body_part", ""),
                    "Situation": sh.get("situation", ""),
                })

        # Team summary — now with financial data
        home = self.config.home_team
        away = self.config.away_team
        home_df = df_main[df_main["team"] == home]
        away_df = df_main[df_main["team"] == away]
        fin = self.accumulator.match_financials
        press_metrics = self._team_pressing_metrics()
        team_rows = []
        for team, tdf in [(home, home_df), (away, away_df)]:
            pm = press_metrics.get(team, {})
            row = {
                "Team": team,
                "Goals": tdf["goals"].sum(),
                "xG": round(tdf["xg"].sum(), 2),
                "Shots on Target": tdf["shots_on_target"].sum(),
                "Total Shots": (tdf["shots_on_target"] + tdf["shots_off_target"] + tdf["shots_blocked_att"]).sum(),
                "Passes Completed": tdf["passes_completed"].sum(),
                "Pass Accuracy %": round(
                    tdf["passes_completed"].sum() / max(1, tdf["passes_attempted"].sum()) * 100, 1
                ),
                "Possession %": round(
                    tdf["passes_completed"].sum() /
                    max(1, df_main["passes_completed"].sum()) * 100, 1
                ),
                "Tackles Won": tdf["tackles_won"].sum(),
                "Interceptions": tdf["interceptions"].sum(),
                "Pressures": tdf["pressures"].sum(),
                "Opponent Press Triggers": pm.get("OPPONENT_PRESS_TRIGGERED", 0),
                "Phase Regressions": pm.get("PHASE_REGRESSION", 0),
                "Yellow Cards": tdf["yellow_cards"].sum(),
                "Red Cards": tdf["red_cards"].sum(),
                "Distance Covered (km)": round(tdf["distance_covered"].sum(), 1),
            }
            # Add financial data (only for home team — home team gets the revenue)
            if fin:
                if team == home:
                    row["Attendance"] = fin["attendance"]
                    row["Fill Rate %"] = fin["fill_rate"]
                    row["Avg Ticket Price"] = fin["ticket_price"]
                    row["Match Revenue"] = fin["money_gained_home"]
                else:
                    row["Attendance"] = fin["attendance"]
                    row["Fill Rate %"] = fin["fill_rate"]
                    row["Avg Ticket Price"] = fin["ticket_price"]
                    row["Match Revenue"] = ""  # Away team doesn't get home gate receipts
            team_rows.append(row)

        # Event timeline (condensed)
        timeline_rows = []
        for e in self.result.timeline:
            if e.event_type in (
                EventType.GOAL, EventType.YELLOW_CARD, EventType.RED_CARD,
                EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED,
                EventType.SAVE, EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
                EventType.SUBSTITUTION, EventType.CORNER_WON,
            ):
                timeline_rows.append({
                    "Minute": e.minute,
                    "Event": e.event_type.name,
                    "Team": e.team,
                    "Player": e.player,
                    "Secondary": e.secondary_player or "",
                    "xG": round(e.xg, 3) if e.xg else "",
                    "Phase": e.phase.value if e.phase else "",
                    "Location X": round(e.location_x or 0, 1),
                    "Location Y": round(e.location_y or 0, 1),
                })

        # Pattern of Play breakdown (Opta-style).
        # The internal SituationType maps onto the 7 canonical patterns of play:
        #   Regular, Set-piece (indirect free-kick), Throw-in, Direct free-kick,
        #   Direct corner, Fast break, Penalty.
        pattern_order = [
            ("Regular",              ("open_play",)),
            ("Set-piece",            ("crossed_freekick",)),
            ("Throw-in",             ("throw_in",)),
            ("Direct free-kick",     ("direct_freekick",)),
            ("Direct corner",        ("corner",)),
            ("Fast break",           ("fast_break",)),
            ("Penalty",              ("penalty",)),
        ]

        def _shot_situations() -> dict:
            counts = {}
            for name, s in self.accumulator.stats.items():
                for sh in s.get("shot_map", []):
                    sit = sh.get("situation", "open_play") or "open_play"
                    key = (s["team"], sit)
                    counts[key] = counts.get(key, 0) + 1
            return counts

        goal_situations = {}
        for g in self.result.goals:
            sit = g.situation.value if g.situation else "open_play"
            key = (g.team, sit)
            goal_situations[key] = goal_situations.get(key, 0) + 1
        shot_situations = _shot_situations()

        pattern_rows = []
        covered_sits = set()
        for label, sits in pattern_order:
            covered_sits.update(sits)
            pattern_rows.append({
                "Pattern of Play": label,
                f"{home} Goals": sum(v for (t, s), v in goal_situations.items() if s in sits and t == home),
                f"{away} Goals": sum(v for (t, s), v in goal_situations.items() if s in sits and t == away),
                f"{home} Shots": sum(v for (t, s), v in shot_situations.items() if s in sits and t == home),
                f"{away} Shots": sum(v for (t, s), v in shot_situations.items() if s in sits and t == away),
            })
        # Any leftover situation not in the 7 (e.g. own_goal) gets its own roll-up.
        known_sits = {s for _, s in goal_situations} | {s for _, s in shot_situations}
        for sit in sorted(known_sits - covered_sits):
            pattern_rows.append({
                "Pattern of Play": sit,
                f"{home} Goals": sum(v for (t, s), v in goal_situations.items() if s == sit and t == home),
                f"{away} Goals": sum(v for (t, s), v in goal_situations.items() if s == sit and t == away),
                f"{home} Shots": sum(v for (t, s), v in shot_situations.items() if s == sit and t == home),
                f"{away} Shots": sum(v for (t, s), v in shot_situations.items() if s == sit and t == away),
            })

        tmp_path = filepath.replace(".xlsx", ".tmp.xlsx")
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                df_main.to_excel(writer, sheet_name="Player Stats", index=False)
                pd.DataFrame(team_rows).to_excel(writer, sheet_name="Team Summary", index=False)
                pd.DataFrame(goal_rows).to_excel(writer, sheet_name="Goals", index=False)
                pd.DataFrame(pattern_rows).to_excel(writer, sheet_name="Pattern of Play", index=False)
                pd.DataFrame(shot_rows).to_excel(writer, sheet_name="Shot Map", index=False)
                pd.DataFrame(card_rows).to_excel(writer, sheet_name="Cards", index=False)
                pd.DataFrame(timeline_rows).to_excel(writer, sheet_name="Key Events Timeline", index=False)

                # Per-team sheets
                home_df.drop(columns=["shot_map"], errors="ignore").to_excel(
                    writer, sheet_name=f"{home[:20]} Stats", index=False)
                away_df.drop(columns=["shot_map"], errors="ignore").to_excel(
                    writer, sheet_name=f"{away[:20]} Stats", index=False)

                # Stamina & Substitution sheets
                if self.sub_controller is not None:
                    stamina_df = self.sub_controller.get_stamina_report()
                    if not stamina_df.empty:
                        stamina_df.to_excel(writer, sheet_name="Stamina & Recovery", index=False)

                    subs_df = self.sub_controller.get_subs_report()
                    if not subs_df.empty:
                        subs_df.to_excel(writer, sheet_name="Substitutions", index=False)

                    # Injury report sheet
                    injured = stamina_df[stamina_df["Injured"] == True] if not stamina_df.empty else pd.DataFrame()
                    if not injured.empty:
                        injured.to_excel(writer, sheet_name="Injuries", index=False)
                        
                # Passing sheets — Opta/StatsBomb-style pass combinations,
                # chance-creation combinations, and per-player pass profile,
                # all built from real timeline coordinates/outcomes.
                combo_df, chance_df, profile_df, team_df = self._build_passing_sheets()
                if not combo_df.empty:
                    combo_df.to_excel(writer, sheet_name="Pass Combinations", index=False)
                if not chance_df.empty:
                    chance_df.to_excel(writer, sheet_name="Chance Combinations", index=False)
                if not profile_df.empty:
                    profile_df.to_excel(writer, sheet_name="Pass Profile", index=False)
                if not team_df.empty:
                    team_df.to_excel(writer, sheet_name="Pass Breakdown", index=False)

                # Sequence analytics — Opta-style shot-ending build-up metrics
                styles_df, shot_df = self._build_sequence_sheets()
                if not styles_df.empty:
                    styles_df.to_excel(writer, sheet_name="Attacking Styles", index=False)
                if not shot_df.empty:
                    shot_df.to_excel(writer, sheet_name="Shot-Ending Sequences", index=False)

                # ── THIRD PROGRESSION SHEETS ──────────────────────────────
                # Zone-based progression analytics: Mid 3rd, final 3rd, def 3rd entries/exits
                # for home team (attacks right) and away team (attacks left)
                # Also footed events showing foot used per pass type
                if include_third_progression or include_footed_events:
                    # Third progression data
                    third_df, footed_df, player_third_df = self._build_third_progression()
                
                    # Process third progression to show mid, final, def 3rd entries/exits for both teams
                    third_progression_processed = self._process_third_progression_data(third_df)
                    if not third_progression_processed.empty:
                        third_progression_processed.to_excel(writer, sheet_name="Third Progression Summary", index=False)
                
                    if not footed_df.empty:
                        # Add team attacks (Right/Left) column to footed events
                        footed_df = self._add_attacks_column(footed_df)
                        footed_df.to_excel(writer, sheet_name="Footed Events", index=False)
                    if not player_third_df.empty:
                        player_third_df.to_excel(writer, sheet_name="Player Third Entries", index=False)

                # ── OPTA TELEMETRY SHEETS ──────────────────────────────
                # Movement/activity, momentum, game-state minutes, error→shot
                # chains, dribbler tackles and line-packing passes — all
                # derived from the position logs + timeline by opta_analytics.
                opta = getattr(self.accumulator, "opta", None)
                if opta is not None:
                    # Movement & activity (per player per minute)
                    if opta.activity_table:
                        pd.DataFrame(opta.activity_table).to_excel(
                            writer, sheet_name="Movement & Activity", index=False)

                    # Momentum series
                    if opta.momentum_series:
                        pd.DataFrame(opta.momentum_series).to_excel(
                            writer, sheet_name="Momentum", index=False)

                    # Game-state minutes (per-minute match state + per-player mins)
                    if opta.game_state_table:
                        pd.DataFrame(opta.game_state_table).to_excel(
                            writer, sheet_name="Game State", index=False)
                        gs_rows = []
                        for name, s in self.accumulator.stats.items():
                            gs_rows.append({
                                "Player": name,
                                "Team": s["team"],
                                "Minutes": s["minutes_played"],
                                "Level": s["minutes_level"],
                                "Ahead": s["minutes_ahead"],
                                "Behind": s["minutes_behind"],
                            })
                        if gs_rows:
                            pd.DataFrame(gs_rows).to_excel(
                                writer, sheet_name="Game State Minutes", index=False)

                        # Errors → shot/goal chains
                        if opta.error_chains:
                            pd.DataFrame(opta.error_chains).to_excel(
                                writer, sheet_name="Errors to Shots", index=False)

                        # Dribbler tackles (defender vs dribbler duel stats)
                        drb_rows = []
                        for name, s in self.accumulator.stats.items():
                            against = s["dribbles_against"]
                            drb_rows.append({
                                "Player": name,
                                "Team": s["team"],
                                "Dribblers Tackled": s["dribblers_tackled"],
                                "Dribbles Against": against,
                                "Duel Win %": round(s["dribblers_tackled"] / against * 100, 1) if against else 0.0,
                                "Dribbled Past": s["dribbled_past"],
                            })
                        if drb_rows:
                            pd.DataFrame(drb_rows).to_excel(
                                writer, sheet_name="Dribbler Tackles", index=False)

                        # Line heights + packing/line-breaking passes
                        line_rows = []
                        teams = (self.config.home_team, self.config.away_team)
                        for m in sorted({mi for team_lines in opta.lines_by_minute.values() for mi in team_lines}):
                            row = {"Minute": m}
                            for team in teams:
                                per_min = opta.lines_by_minute.get(team, {}).get(m, {})
                                tag = "Home" if team == self.config.home_team else "Away"
                                row[f"{tag} Def Line x"] = round(per_min.get("def", 0.0), 1)
                                row[f"{tag} Mid Line x"] = round(per_min.get("mid", 0.0), 1)
                            line_rows.append(row)
                        if line_rows:
                            pd.DataFrame(line_rows).to_excel(
                                writer, sheet_name="Line Heights", index=False)

                        pk_rows = []
                        for name, s in self.accumulator.stats.items():
                            pk_rows.append({
                                "Player": name,
                                "Team": s["team"],
                                "Position": s["position"],
                                "Packing Passes": s["packing_passes"],
                                "Packing Dribbles": s["packing_dribbles"],
                                "Total Packing": s["total_packing"],
                                "Line-Breaking Passes": s["line_breaking_passes"],
                            })
                        if pk_rows:
                            pd.DataFrame(pk_rows).to_excel(
                                writer, sheet_name="Line & Packing Passes", index=False)

                        # Per-event line-packing ledger (every pass/dribble that packed
                        # or line-broke, with the exact geometry that produced it).
                        if opta.packing_log:
                            pd.DataFrame(opta.packing_log).to_excel(
                                writer, sheet_name="Line & Packing Log", index=False)

                    # ── CHANCE CREATION LEDGER ───────────────────────────
                    # Every shot with its full creation chain, derived from the REAL
                    # timeline (chance_creation.py): key pass / assist, xA, second
                    # assist, shot-creating actions, fantasy assist, big chance.
                    from chance_creation import ChanceCreationLedger
                    cc = ChanceCreationLedger(self.result.timeline).compute()
                    if cc.records:
                        cc_rows = [r.as_dict() for r in cc.records]
                        for row in cc_rows:
                            row["goal_assist"] = row.pop("is_goal_assist")
                        pd.DataFrame(cc_rows).to_excel(
                            writer, sheet_name="Chance Creation", index=False)

                    # ── CROSSES ──────────────────────────────────────────
                    # Every delivery the geometric CrossDetector stamped `cross: true`
                    # (wide origin → into/through the opponent box, kicked — the
                    # Opta/StatsBomb rule), plus the engine's raw CROSS_ATTEMPT /
                    # CROSS_SUCCESS events (marked `Geometric Cross` True/False so a
                    # central, non-cross "cross" is honestly flagged). Includes the
                    # reclassified generic passes that are genuinely crosses.
                    CROSS_TYPES = (EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS)
                    cross_rows = []
                    for e in self.result.timeline:
                        meta = getattr(e, "metadata", None) or {}
                        geom = bool(meta.get("cross"))
                        if not (geom or e.event_type in CROSS_TYPES):
                            continue
                        cross_rows.append({
                            "Minute": e.minute,
                            "Team": e.team,
                            "Player": e.player,
                            "Event": e.event_type.name,
                            "Geometric Cross": geom,
                            "Airborne": bool(meta.get("is_airborne")),
                            "Origin Zone": meta.get("cross_origin", ""),
                            "Destination Zone": meta.get("cross_dest", ""),
                            "From X": round(e.location_x or 0, 1),
                            "From Y": round(e.location_y or 0, 1),
                            "To X": round(e.end_x or 0, 1),
                            "To Y": round(e.end_y or 0, 1),
                            "Completed": bool(e.outcome),
                        })
                    if cross_rows:
                        pd.DataFrame(cross_rows).to_excel(
                            writer, sheet_name="Crosses", index=False)

                        # Per-team geometric cross summary.
                        team_names = [self.config.home_team, self.config.away_team]
                        team_totals = {}
                        for r in cross_rows:
                            t = r["Team"]
                            if not r["Geometric Cross"]:
                                continue
                            tt = team_totals.setdefault(t, {
                                "Team": t, "Geometric Crosses": 0, "Airborne": 0,
                                "Landed in Box": 0, "Flashed Through Box": 0,
                                "Completed": 0,
                            })
                            tt["Geometric Crosses"] += 1
                            tt["Airborne"] += 1 if r["Airborne"] else 0
                            tt["Completed"] += 1 if r["Completed"] else 0
                            if r["Destination Zone"] == "penalty_box":
                                tt["Landed in Box"] += 1
                            elif r["Destination Zone"] == "through_box":
                                tt["Flashed Through Box"] += 1
                        if team_totals:
                            order = [team_totals[t] for t in team_names if t in team_totals]
                            pd.DataFrame(order).to_excel(
                                writer, sheet_name="Crosses Summary", index=False)

                    # ── THIRD PROGRESSION + FOOTED LOG ───────────────────
                    # Zone entries/exits for each pitch third (relative to each team's
                    # own attacking direction) plus a footed per-pass event log.
                    third_df, footed_df, player_third_df = self._build_third_progression()
                    if not third_df.empty:
                        third_df.to_excel(writer, sheet_name="Third Progression", index=False)
                    if not footed_df.empty:
                        footed_df.to_excel(writer, sheet_name="Footed Passes", index=False)
                    if not player_third_df.empty:
                        player_third_df.to_excel(writer, sheet_name="Third Entries (Player)", index=False)

            os.replace(tmp_path, filepath)
            print(f"   📊 Excel  → {filepath}")
        except PermissionError:
            print(f"   ⚠️  Excel locked — saved to {tmp_path} instead. Close the xlsx and rename manually.")

    def export_csv(self, filepath: str):
        self.df.drop(columns=["shot_map"], errors="ignore").to_csv(filepath, index=False)
        print(f"   📄 CSV    → {filepath}")

    def _defensive_awareness_json(self) -> dict:
        """
        Checkpoint 9 — the threat engine's view of both teams' defensive
        awareness across the match: minute-by-minute danger (with a goal
        peak), clearance effectiveness (headed vs foot), and how often the
        team got the ball away cleanly vs let it end in a shot on their goal.
        """
        threat = getattr(self.result, "threat", None)
        if threat is None:
            return {}
        return threat.report()

    def _timeline_json(self) -> list:
        """Serialize the FULL event timeline with real coordinates.

        Every MatchEvent in result.timeline carries exact location_x/y and
        end_x/end_y (passes, carries, dribbles, shots...). This is the raw
        feed the web app's player event maps read from — no synthesis.
        """
        PASS_KINDS = {
            EventType.PASS: "pass",
            EventType.PROGRESSIVE_PASS: "pass",
            EventType.SWITCH_OF_PLAY: "pass",
            EventType.THROUGH_BALL: "through",
            EventType.CROSS_ATTEMPT: "cross",
            EventType.CROSS_SUCCESS: "cross",
            EventType.CHANCE_CREATED: "chance",
            EventType.BIG_CHANCE_CREATED: "chance",
        }
        CARRY_KINDS = {EventType.CARRY}
        DRIBBLE_KINDS = {EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL}
        rows = []
        for e in self.result.timeline:
            t = e.event_type.name
            kind = None
            if e.event_type in PASS_KINDS:
                kind = PASS_KINDS[e.event_type]
            elif e.event_type in CARRY_KINDS:
                kind = "carry"
            elif e.event_type in DRIBBLE_KINDS:
                kind = "dribble"
            rows.append({
                "minute": e.minute,
                "second": e.second,
                "type": t,
                "kind": kind,
                "team": e.team,
                "player": e.player,
                "secondary_player": e.secondary_player or "",
                "x": None if e.location_x is None else round(e.location_x, 2),
                "y": None if e.location_y is None else round(e.location_y, 2),
                "end_x": None if e.end_x is None else round(e.end_x, 2),
                "end_y": None if e.end_y is None else round(e.end_y, 2),
                "outcome": bool(e.outcome),
                "xg": round(e.xg, 3) if e.xg else 0,
                "xa": round(e.xa, 3) if e.xa else 0,
                "situation": e.situation.value if e.situation else "open_play",
                "phase": e.phase.value if e.phase else "",
            })
        return rows

    def export_json(self, filepath: str):
        fin = self.accumulator.match_financials
        third_df, footed_df, _ = self._build_third_progression()
        payload = {
            "match": {
                "home_team": self.config.home_team,
                "away_team": self.config.away_team,
                "score": f"{self.state.home_goals}–{self.state.away_goals}",
                "home_xg": round(self.state.home_xg, 2),
                "away_xg": round(self.state.away_xg, 2),
                "matchday": self.config.matchday,
                "season": self.config.season,
                "competition": self.config.competition,
                "venue": self.config.venue,
                "date": str(self.config.match_date),
                "added_time": self.state.added_time,
                "is_derby": self.config.is_derby,
            },
            "timeline": self._timeline_json(),
            "financials": {
                "stadium_capacity": fin["stadium_capacity"],
                "attendance": fin["attendance"],
                "fill_rate_pct": fin["fill_rate"],
                "avg_ticket_price": fin["ticket_price"],
                "home_revenue": fin["money_gained_home"],
                "match_tier": fin["match_tier"],
            } if fin else {},
            "goals": [
                {
                    "minute": g.minute, "team": g.team,
                    "scorer": g.player, "assist": g.secondary_player or "",
                    "situation": g.situation.value if g.situation else "open_play",
                    "xg": round(g.xg, 3),
                }
                for g in self.result.goals
            ],
            "sequences": {
                "attacking_styles": self._get_sequence_tracker().macro_rows(),
                "shot_ending": self._get_sequence_tracker().shot_ending_rows(),
            },
            "players": {
                name: {k: v for k, v in s.items() if k != "shot_map"}
                for name, s in self.accumulator.stats.items()
            },
            "opta": self._opta_analytics_json(),
            "pressing": {
                "teams": [
                    {"team": team, **metrics}
                    for team, metrics in self._team_pressing_metrics().items()
                ],
            },
            "defensive_awareness": self._defensive_awareness_json(),
            "third_progression": {
                "teams": [
                    {
                        "team": team,
                        "attacks": "Right" if team == self.config.home_team else "Left",
                        "thirds": [
                            {
                                "third": r["Third"],
                                "entries_passes_in": int(r["Entries (Passes In)"]),
                                "completed_entries": int(r["Completed Entries"]),
                                "exits_passes_out": int(r["Exits (Passes Out)"]),
                            }
                            for _, r in third_df.iterrows()
                            if r["Team"] == team
                        ],
                    }
                    for team in (self.config.home_team, self.config.away_team)
                ],
                "footed_passes": footed_df.to_dict(orient="records") if not footed_df.empty else [],
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"   🗂️  JSON   → {filepath}")

    def _team_pressing_metrics(self) -> dict:
        """
        Per-team pressing export keys (Checkpoint 15):
            PASS_COMPLETED            — completed passes by the team
            OPPONENT_PRESS_TRIGGERED  — presses the OPPONENT triggered while
                                        this team was building
            PHASE_REGRESSION          — deliberate backward phase resets
                                        (recycle / release-to-GK /
                                        emergency drop-to-GK)
            PRESSURES / PRESS_SUCCESS — the team's own pressing volume
            PRESS_PROFILE             — the defending profile stamped on the
                                        team's own PRESS events
            PRESS_TAX                 — the stamina fatigue tax applied
        """
        home, away = self.config.home_team, self.config.away_team
        out = {
            t: {
                "PASS_COMPLETED": 0,
                "OPPONENT_PRESS_TRIGGERED": 0,
                "PHASE_REGRESSION": 0,
                "PRESSURES": 0,
                "PRESS_SUCCESS": 0,
                "PRESS_PROFILE": "",
                "PRESS_TAX": 1.0,
            }
            for t in (home, away)
        }
        regress = {"recycle_backward", "release_to_gk", "emergency_drop_to_gk"}
        pass_like = {
            EventType.PASS, EventType.PROGRESSIVE_PASS,
            EventType.SWITCH_OF_PLAY, EventType.THROUGH_BALL,
        }
        for ev in self.result.timeline:
            md = ev.metadata or {}
            if ev.event_type in pass_like:
                if ev.outcome and ev.team in out:
                    out[ev.team]["PASS_COMPLETED"] += 1
                if md.get("phase_directive") in regress and ev.team in out:
                    out[ev.team]["PHASE_REGRESSION"] += 1
                if md.get("under_pressure") and ev.team in out:
                    opp = away if ev.team == home else home
                    out[opp]["OPPONENT_PRESS_TRIGGERED"] += 1
            elif ev.event_type == EventType.PRESS and ev.team in out:
                out[ev.team]["PRESSURES"] += 1
                if md.get("press_profile"):
                    out[ev.team]["PRESS_PROFILE"] = md["press_profile"]
                out[ev.team]["PRESS_TAX"] = float(md.get("press_tax", out[ev.team]["PRESS_TAX"]))
            elif ev.event_type == EventType.PRESS_SUCCESS and ev.team in out:
                out[ev.team]["PRESS_SUCCESS"] += 1
        # Press success is inferred by the accumulator (a PRESS followed by the
        # same player winning the ball) — mirror its per-player verdict.
        for name, s in self.accumulator.stats.items():
            team = s.get("team")
            if team in out:
                out[team]["PRESS_SUCCESS"] = (
                    out[team]["PRESS_SUCCESS"] + int(s.get("press_success", 0))
                )
        return out

    def _opta_analytics_json(self) -> dict:
        opta = getattr(self.accumulator, "opta", None)
        if opta is None:
            return {}
        return {
            "momentum": opta.momentum_series,
            "game_state_minutes": opta.game_state_table,
            "errors_to_shot_goal": opta.error_chains,
            "activity": opta.activity_table,
            "lines": {
                team: {
                    str(m): v for m, v in team_lines.items()
                }
                for team, team_lines in opta.lines_by_minute.items()
            },
        }

    # ─────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────

    def plot_shot_map(self, filepath: str):
        """
        Shot map for both teams on a vertical pitch.
        StatsBomb/Opta-style: ALL shots are circular markers sized by xG.
        High-xG shots (≥0.5) get large, prominent circles.
        Low-xG shots (≤0.05) get tiny circles.
        Goals are distinguished with a gold edge and slightly larger size.
        Opacity scales with xG so low-value shots fade into the background.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                                  facecolor=PLOFAStyle.BG_DARK)
        fig.suptitle(
            f"SHOT MAP\n{self.config.home_team} {self.state.home_goals}–{self.state.away_goals} {self.config.away_team}",
            color=PLOFAStyle.TEXT_PRIMARY, fontsize=15, fontweight="bold", y=0.98
        )

        teams = [self.config.home_team, self.config.away_team]
        colors = [self.home_color, self.away_color]

        for ax, team, color in zip(axes, teams, colors):
            pitch = VerticalPitch(
                pitch_type="statsbomb",
                pitch_color=PLOFAStyle.PITCH_GREEN,
                line_color=PLOFAStyle.PITCH_LINE,
                half=True,
                line_zorder=2,
            )
            pitch.draw(ax=ax)
            ax.set_facecolor(PLOFAStyle.BG_DARK)

            # Collect shots for this team
            team_stats = {
                n: s for n, s in self.accumulator.stats.items()
                if s["team"] == team
            }

            # StatsBomb/Opta-style: all shots are circles sized by xG.
            # xG ~ 0.80+ → larger circle (high-quality chance)
            # xG ~ 0.15   → medium circle
            # xG ~ 0.02   → small circle (fades into the background)
            # Size range: ~50 (0 xG) to ~500 (0.99 xG) — deliberately
            # compact, matching the modest dot sizes used in real
            # Opta/StatsBomb shot maps.
            # Opacity: 0.25 (low xG) → 0.95 (high xG)
            home_panel = (team == self.config.home_team)
            for name, s in team_stats.items():
                for shot in s.get("shot_map", []):
                    sx = shot["x"]
                    sy = shot["y"]
                    # Each panel shows THAT team attacking the goal at the
                    # top. Home attacks right (x→105), away attacks left
                    # (x→0). Away x is mirrored so shots in the away half
                    # land in the same visual panel half as home shots;
                    # y is kept identical (no flip) so the goal mouth
                    # and shot width align with the home panel.
                    if home_panel:
                        sb_x = (sx / 105) * 120
                        sb_y = (sy / 68)  * 80
                    else:
                        sb_x = ((105 - sx) / 105) * 120
                        sb_y = (sy / 68)   * 80

                    # Trajectory destination (where the shot went), same
                    # mirror on x; y is kept identical.
                    ex = shot.get("end_x", sx)
                    ey = shot.get("end_y", sy)
                    if home_panel:
                        eb_x = (ex / 105) * 120
                        eb_y = (ey / 68) * 80
                    else:
                        # Mirror start position so away shots land in the same
                        # visual half as home shots, but do NOT mirror the endpoint
                        # because goal_x=0 for away team already points toward the
                        # away goal (visual left). Mirroring would send it to x=120
                        # creating a horizontal line instead of pointing at the goal.
                        eb_x = (ex / 105) * 120
                        eb_y = (ey / 68)  * 80

                        # Fix away-team trajectory corruption: when a shot
                        # originates in the defensive half (x > 52.5), the
                        # mirrored start lands at the bottom of the panel
                        # while the goal sits at the top, producing a long
                        # horizontal line across the whole pitch. Cap the
                        # trajectory so it always points toward the goal
                        # without spanning the entire panel.
                        max_traj = 72.0
                        if eb_x - sb_x > max_traj:
                            eb_x = sb_x + max_traj

                    outcome = shot.get("outcome", "miss")
                    xg_val = shot.get("xg", 0.05)

                    # Size scales with xG: exponential-style so high-xG shots
                    # stand out. xG=0 → 25, xG=0.99 → ~500
                    size = 5 + (xg_val ** 0.6) * 480

                    # Opacity: low-xG shots are faded, high-xG are solid
                    alpha = max(0.20, min(0.95, 0.25 + xg_val * 0.85))

                    # Color by outcome (Opta/StatsBomb style)
                    outcome_colors = {
                        "goal":     PLOFAStyle.GOAL_COLOR,
                        "saved":    PLOFAStyle.SOT_COLOR,
                        "miss":     PLOFAStyle.MISS_COLOR,
                        "blocked":  PLOFAStyle.BLOCK_COLOR,
                        "woodwork": PLOFAStyle.ACCENT_GOLD,
                    }
                    c = outcome_colors.get(outcome, PLOFAStyle.MISS_COLOR)

                    # Edge: goals get a prominent white border
                    edge_color = "white" if outcome == "goal" else "none"
                    edge_width = 2.5 if outcome == "goal" else 0

                    # Goals get a small size bump for prominence
                    if outcome == "goal":
                        size *= 1.15

                    # Dotted trajectory showing where each shot went
                    pitch.lines(
                        sb_x, sb_y, eb_x, eb_y, ax=ax,
                        ls=":", lw=1.0, color="white", alpha=0.35, zorder=3,
                    )

                    pitch.scatter(
                        sb_x, sb_y, ax=ax,
                        s=size, c=c, marker="o",
                        alpha=alpha, zorder=5,
                        edgecolors=edge_color,
                        linewidths=edge_width,
                    )

            # xG total
            team_xg = self.state.home_xg if team == self.config.home_team else self.state.away_xg
            total_goals = self.state.home_goals if team == self.config.home_team else self.state.away_goals

            ax.set_title(
                f"{team}\nGoals: {total_goals}  |  xG: {team_xg:.2f}",
                color=color, fontsize=12, fontweight="bold", pad=10
            )

        # Legend with size reference
        legend_elements = [
            mpatches.Patch(color=PLOFAStyle.GOAL_COLOR, label="Goal"),
            mpatches.Patch(color=PLOFAStyle.SOT_COLOR,  label="On Target"),
            mpatches.Patch(color=PLOFAStyle.MISS_COLOR,  label="Off Target"),
            mpatches.Patch(color=PLOFAStyle.BLOCK_COLOR, label="Blocked"),
            mpatches.Patch(color=PLOFAStyle.ACCENT_GOLD, label="Woodwork"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=5, facecolor=PLOFAStyle.BG_CARD,
                   labelcolor=PLOFAStyle.TEXT_PRIMARY, fontsize=9,
                   framealpha=0.8)

        # Add xG size reference annotation
        fig.text(0.5, 0.01, "Circle size ∝ xG (bigger = higher quality chance) · dotted lines = shot trajectory",
                 ha="center", fontsize=8, color=PLOFAStyle.TEXT_MUTED,
                 fontstyle="italic")

        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   🗺️  Shot Map       → {filepath}")

    def plot_pass_network(self, filepath: str):
        """
        Pass network for both teams.
        Player nodes positioned by average touch location.
        Edge thickness = pass volume between players.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 10),
                                  facecolor=PLOFAStyle.BG_DARK)
        fig.suptitle(
            f"PASS NETWORK\n{self.config.home_team} vs {self.config.away_team}",
            color=PLOFAStyle.TEXT_PRIMARY, fontsize=14, fontweight="bold"
        )

        teams     = [self.config.home_team, self.config.away_team]
        t_colors  = [self.home_color, self.away_color]

        for ax, team, tcolor in zip(axes, teams, t_colors):
            pitch = Pitch(
                pitch_type="statsbomb",
                pitch_color=PLOFAStyle.PITCH_GREEN,
                line_color=PLOFAStyle.PITCH_LINE,
            )
            pitch.draw(ax=ax)

            team_stats = {
                n: s for n, s in self.accumulator.stats.items()
                if s["team"] == team and s["minutes_played"] > 0
            }

            if not team_stats:
                ax.set_title(team, color=tcolor)
                continue

            # Use PassMatrix for real pass data
            pm = PassMatrix.build(team, self.result.timeline)
            positions: dict[str, tuple[float, float]] = {}
            for name in team_stats:
                avg = pm.average_position(name)
                if avg:
                    positions[name] = avg

            # Assign positions if PassMatrix has them, else fallback to position-based
            if positions:
                for name, (avg_x, avg_y) in positions.items():
                    sb_x = avg_x / 105 * 120
                    sb_y = avg_y / 68 * 80
                    pitch.scatter(sb_x, sb_y, ax=ax, s=200, c=tcolor,
                                 edgecolors="white", linewidth=1.5, zorder=5)
                    pitch.annotate(name, (sb_x, sb_y), ax=ax,
                                  fontsize=7, color="white",
                                  ha="center", va="bottom", zorder=6)
            else:
                # Fallback: position-based coordinates
                position_coords = {
                    "GK":  (8,  40), "CB":  (28, 40), "LB":  (28, 15),
                    "RB":  (28, 65), "CDM": (45, 40), "CM":  (55, 30),
                    "CAM": (70, 40), "LW":  (80, 15), "RW":  (80, 65),
                    "ST":  (95, 40), "CF":  (95, 40),
                }
                pos_seen = defaultdict(int)
                for name, s in team_stats.items():
                    base_pos = s["position"]
                    coords = position_coords.get(base_pos, (50, 40))
                    # Offset for duplicates
                    offset = pos_seen[base_pos] * 3
                    if offset > 5:
                        coords = (coords[0] + offset - 6, coords[1] + offset - 6)
                    pos_seen[base_pos] += 1
                    sb_x = coords[0] / 105 * 120
                    sb_y = coords[1] / 68 * 80
                    pitch.scatter(sb_x, sb_y, ax=ax, s=200, c=tcolor,
                                 edgecolors="white", linewidth=1.5, zorder=5)
                    pitch.annotate(name, (sb_x, sb_y), ax=ax,
                                  fontsize=7, color="white",
                                  ha="center", va="bottom", zorder=6)

            # Draw pass connections
            for combo in pm.combo_rows():
                passer = combo["Passer"]
                receiver = combo["Receiver"]
                if passer in team_stats and receiver in team_stats:
                    completed = combo["Completed"]
                    if completed > 0:
                        # Get positions for passer and receiver
                        ppos = None
                        rpos = None
                        if positions:
                            if passer in positions:
                                ppos = positions[passer]
                            if receiver in positions:
                                rpos = positions[receiver]
                        if ppos and rpos:
                            sx = ppos[0] / 105 * 120
                            sy = ppos[1] / 68 * 80
                            ex = rpos[0] / 105 * 120
                            ey = rpos[1] / 68 * 80
                        else:
                            # Fallback to position coords
                            ppos2 = position_coords.get(team_stats[passer]["position"], (50, 40))
                            rpos2 = position_coords.get(team_stats[receiver]["position"], (50, 40))
                            sx = ppos2[0] / 105 * 120
                            sy = ppos2[1] / 68 * 80
                            ex = rpos2[0] / 105 * 120
                            ey = rpos2[1] / 68 * 80
                        lw = max(0.5, min(6.0, completed / 3.0))
                        alpha = min(0.8, 0.2 + completed * 0.02)
                        pitch.lines(sx, sy, ex, ey, ax=ax,
                                   lw=lw, color="white", alpha=alpha, zorder=2)

            ax.set_title(team, color=tcolor)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   🗺️  Pass Network   → {filepath}")

    def plot_xg_timeline(self, filepath: str):
        """Cumulative xG timeline chart with goal markers at correct xG."""
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOFAStyle.BG_DARK)
        ax.set_facecolor(PLOFAStyle.BG_PANEL)

        home = self.config.home_team
        away = self.config.away_team

        # Collect all shot events and sort by minute
        shots = []
        for e in self.result.timeline:
            if e.event_type in (EventType.GOAL, EventType.SHOT_ON_TARGET,
                                EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED,
                                EventType.HIT_WOODWORK, EventType.PENALTY_SCORED,
                                EventType.PENALTY_MISSED):
                shots.append(e)

        shots.sort(key=lambda e: e.minute * 60 + e.second)

        # Build cumulative xG arrays
        home_mins, home_xg = [0], [0.0]
        away_mins, away_xg = [0], [0.0]
        # Track xG at each goal event for correct marker placement
        goal_xg_tracker = []  # (minute, cumulative_xg, team)
        for e in shots:
            minute = e.minute + e.second / 60.0
            if e.team == home:
                new_val = home_xg[-1] + e.xg
                home_mins.append(minute)
                home_xg.append(new_val)
                away_mins.append(minute)
                away_xg.append(away_xg[-1])
                if e.event_type in (EventType.GOAL, EventType.PENALTY_SCORED):
                    goal_xg_tracker.append((minute, new_val, home))
            else:
                new_val = away_xg[-1] + e.xg
                away_mins.append(minute)
                away_xg.append(new_val)
                home_mins.append(minute)
                home_xg.append(home_xg[-1])
                if e.event_type in (EventType.GOAL, EventType.PENALTY_SCORED):
                    goal_xg_tracker.append((minute, new_val, away))

        # Add end point
        total_mins = 90 + self.state.added_time
        home_mins.append(total_mins)
        home_xg.append(home_xg[-1])
        away_mins.append(total_mins)
        away_xg.append(away_xg[-1])

        # Plot
        ax.step(home_mins, home_xg, where="post", color=self.home_color,
                linewidth=2.5, label=f"{home} ({self.state.home_goals})")
        ax.step(away_mins, away_xg, where="post", color=self.away_color,
                linewidth=2.5, label=f"{away} ({self.state.away_goals})")

        # Goal markers at correct cumulative xG
        for g_minute, g_xg, g_team in goal_xg_tracker:
            team_color = self.home_color if g_team == home else self.away_color
            ax.scatter(g_minute, g_xg,
                      color=team_color, s=140, zorder=6,
                      edgecolors="white", linewidth=1.5)

        ax.set_title(f"xG TIMELINE — {home} vs {away}",
                     color=PLOFAStyle.TEXT_PRIMARY, fontsize=14, fontweight="bold")
        ax.set_xlabel("Minute", color=PLOFAStyle.TEXT_MUTED)
        ax.set_ylabel("Cumulative xG", color=PLOFAStyle.TEXT_MUTED)
        ax.legend(facecolor=PLOFAStyle.BG_CARD, labelcolor=PLOFAStyle.TEXT_PRIMARY,
                  fontsize=10, framealpha=0.8)
        ax.set_xlim(0, total_mins)
        ax.set_ylim(0, max(max(home_xg), max(away_xg), 0.5) * 1.15)

        # Grid for readability
        ax.grid(True, alpha=0.15, color=PLOFAStyle.TEXT_MUTED)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   xG Timeline    -> {filepath}")

    def plot_momentum(self, filepath: str):
        """Per-minute momentum chart with scoreline goal markers.

        Momentum is +100..−100 from the home team's perspective
        (positive = home dominant). The away side mirrors it.
        """
        home = self.config.home_team
        away = self.config.away_team
        series = getattr(self.accumulator, "opta", None)
        series = series.momentum_series if series is not None else []

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOFAStyle.BG_DARK)
        ax.set_facecolor(PLOFAStyle.BG_PANEL)

        if series:
            mins = [r["minute"] for r in series]
            vals = [r["momentum"] for r in series]
            ax.fill_between(mins, vals, 0, where=[v >= 0 for v in vals],
                            color=self.home_color, alpha=0.35)
            ax.fill_between(mins, vals, 0, where=[v < 0 for v in vals],
                            color=self.away_color, alpha=0.35)
            ax.plot(mins, vals, color="#FFFFFF", linewidth=2.0,
                    label="Momentum (+home / −away)")

        ax.axhline(0, color=PLOFAStyle.TEXT_MUTED, linewidth=1.0, alpha=0.6)

        # Goal markers push momentum on the scoring team's side
        for g in self.result.goals:
            gm = g.minute
            base = 100 if g.team == home else -100
            ax.scatter(gm, base * 0.95, s=120, zorder=6,
                       color=self.home_color if g.team == home else self.away_color,
                       edgecolors="white", linewidth=1.2)
        if self.result.goals:
            ax.text(0.985, 0.96,
                    "● goals", transform=ax.transAxes, ha="right",
                    color=PLOFAStyle.TEXT_MUTED, fontsize=9)

        ax.set_title(f"MOMENTUM — {home} vs {away}",
                     color=PLOFAStyle.TEXT_PRIMARY, fontsize=14, fontweight="bold")
        ax.set_xlabel("Minute", color=PLOFAStyle.TEXT_MUTED)
        ax.set_ylabel("Momentum", color=PLOFAStyle.TEXT_MUTED)
        ax.legend(facecolor=PLOFAStyle.BG_CARD, labelcolor=PLOFAStyle.TEXT_PRIMARY,
                  fontsize=10, framealpha=0.8)
        total_mins = 90 + self.state.added_time
        ax.set_xlim(0, total_mins)
        ax.set_ylim(-110, 110)
        ax.grid(True, alpha=0.15, color=PLOFAStyle.TEXT_MUTED)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   Momentum       -> {filepath}")

    def plot_match_summary(self, filepath: str):
        """Match summary infographic card — now includes financial data."""
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOFAStyle.BG_DARK)
        ax.set_facecolor(PLOFAStyle.BG_CARD)
        ax.axis("off")

        home = self.config.home_team
        away = self.config.away_team
        fin = self.accumulator.match_financials

        # Scoreline
        ax.text(0.5, 0.88, f"{home}  {self.state.home_goals}–{self.state.away_goals}  {away}",
                ha="center", va="center", fontsize=28, fontweight="bold",
                color=PLOFAStyle.TEXT_PRIMARY)

        # xG
        ax.text(0.5, 0.80, f"xG: {self.state.home_xg:.2f} – {self.state.away_xg:.2f}",
                ha="center", va="center", fontsize=14, color=PLOFAStyle.TEXT_MUTED)

        # Financial data
        if fin:
            ax.text(0.5, 0.755,
                    f"🏟️ {fin['attendance']:,} / {fin['stadium_capacity']:,}  ({fin['fill_rate']}%)  |  "
                    f"🎫 ${fin['ticket_price']:.2f}  |  💰 ${fin['money_gained_home']:,.2f}",
                    ha="center", va="center", fontsize=9, color=PLOFAStyle.ACCENT_GOLD)

        # Goals timeline
        y_start = 0.72
        ax.text(0.5, y_start, "GOALS", ha="center", va="center",
                fontsize=12, fontweight="bold", color=PLOFAStyle.ACCENT_GOLD)
        for i, g in enumerate(self.result.goals):
            y = y_start - 0.04 - (i * 0.035)
            color = self.home_color if g.team == home else self.away_color
            assist = f" (assist: {g.secondary_player})" if g.secondary_player else ""
            ax.text(0.5, y, f"{g.minute}'  {g.player}{assist}",
                    ha="center", va="center", fontsize=10, color=color)

        # ── Key stats ─────────────────────────────────────────────
        df_h = self.df[self.df["team"] == home]
        df_a = self.df[self.df["team"] == away]

        home_passes = int(df_h["passes_completed"].sum())
        away_passes = int(df_a["passes_completed"].sum())
        total_passes = home_passes + away_passes
        poss_h = round(100 * home_passes / total_passes, 1) if total_passes > 0 else 50
        poss_a = round(100 * away_passes / total_passes, 1) if total_passes > 0 else 50

        home_shots = int(df_h["shots_on_target"].sum() + df_h["shots_off_target"].sum() + df_h["shots_blocked_att"].sum())
        away_shots = int(df_a["shots_on_target"].sum() + df_a["shots_off_target"].sum() + df_a["shots_blocked_att"].sum())
        home_sot = int(df_h["shots_on_target"].sum())
        away_sot = int(df_a["shots_on_target"].sum())

        # Count corners from timeline
        home_corners = sum(
            1 for e in self.result.timeline
            if e.event_type == EventType.CORNER_TAKEN and e.team == home
        )
        away_corners = sum(
            1 for e in self.result.timeline
            if e.event_type == EventType.CORNER_TAKEN and e.team == away
        )

        home_yellow = int(df_h["yellow_cards"].sum())
        away_yellow = int(df_a["yellow_cards"].sum())

        stats_y = 0.52
        stats = [
            ("Passes", f"{home_passes} vs {away_passes}"),
            ("Possession", f"{poss_h}% vs {poss_a}%"),
            ("Shots", f"{home_shots} vs {away_shots}"),

        ]
        stats_right = [
            ("Shots on Target", f"{home_sot} vs {away_sot}"),
            ("Corners", f"{home_corners} vs {away_corners}"),
            ("Yellow Cards", f"{home_yellow} vs {away_yellow}"),
        ]

        for i, ((lbl1, val1), (lbl2, val2)) in enumerate(zip(stats, stats_right)):
            y = stats_y - (i * 0.04)
            ax.text(0.18, y, lbl1, ha="right", va="center", fontsize=9,
                    color=PLOFAStyle.TEXT_MUTED)
            ax.text(0.30, y, val1, ha="center", va="center", fontsize=9,
                    color=PLOFAStyle.TEXT_PRIMARY)
            ax.text(0.62, y, lbl2, ha="right", va="center", fontsize=9,
                    color=PLOFAStyle.TEXT_MUTED)
            ax.text(0.74, y, val2, ha="center", va="center", fontsize=9,
                    color=PLOFAStyle.TEXT_PRIMARY)

        # Match info
        ax.text(0.5, 0.08,
                f"{self.config.competition}  |  Matchday {self.config.matchday}  |  "
                f"{self.config.match_date}\n{self.config.venue}  |  Ref: {self.config.referee}",
                ha="center", va="center", fontsize=9, color=PLOFAStyle.TEXT_MUTED)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   🃏 Match Summary  → {filepath}")

    def plot_player_heatmap(self, filepath: str):
        """Touch-density heatmaps for key players from both teams.
        Shows the top 4 most-involved players per team on a grid.
        """
        home = self.config.home_team
        away = self.config.away_team

        # Count touches (any event with location data) per player
        touch_counts: dict[str, int] = defaultdict(int)
        player_locs: dict[str, list] = defaultdict(list)
        for e in self.result.timeline:
            if e.location_x is not None and e.location_y is not None and e.player:
                touch_counts[e.player] += 1
                player_locs[e.player].append((e.location_x, e.location_y))

        # Separate by team, sort by touches descending
        home_players = sorted(
            [(n, c) for n, c in touch_counts.items()
             if n in self.accumulator.stats and self.accumulator.stats[n]["team"] == home],
            key=lambda x: -x[1]
        )[:4]
        away_players = sorted(
            [(n, c) for n, c in touch_counts.items()
             if n in self.accumulator.stats and self.accumulator.stats[n]["team"] == away],
            key=lambda x: -x[1]
        )[:4]

        n_home = len(home_players)
        n_away = len(away_players)
        n_rows = max(n_home, n_away, 1)

        fig, axes = plt.subplots(
            n_rows, 2, figsize=(12, 4 * n_rows),
            facecolor=PLOFAStyle.BG_DARK
        )
        if n_rows == 1:
            axes = np.array([axes])

        for row_idx in range(n_rows):
            for col_idx, (team, players, team_color) in enumerate([
                (home, home_players, self.home_color),
                (away, away_players, self.away_color)
            ]):
                ax = axes[row_idx][col_idx]
                pitch = Pitch(
                    pitch_type="statsbomb",
                    pitch_color=PLOFAStyle.PITCH_GREEN,
                    line_color=PLOFAStyle.PITCH_LINE,
                )
                pitch.draw(ax=ax)

                if row_idx < len(players):
                    pname, _ = players[row_idx]
                    locs = player_locs.get(pname, [])
                    if locs:
                        xs = [p[0] for p in locs]
                        ys = [p[1] for p in locs]
                        pitch.kdeplot(xs, ys, ax=ax, cmap="Reds", fill=True, alpha=0.7)
                        ax.scatter(xs, ys, s=8, color="white", alpha=0.3, zorder=5)
                    ax.set_title(pname, color=team_color, fontsize=10, fontweight="bold")
                else:
                    ax.set_title("—", color=PLOFAStyle.TEXT_MUTED, fontsize=10)

                if row_idx == 0:
                    ax.text(
                        0.5, 1.08, team,
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color=team_color
                    )

        fig.suptitle(
            f"PLAYER TOUCH MAPS — {home} vs {away}",
            color=PLOFAStyle.TEXT_PRIMARY, fontsize=14, fontweight="bold", y=1.02
        )

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   👟 Player Heatmap → {filepath}")

    def plot_pressure_map(self, filepath: str):
        """Pressure heatmap for both teams."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                                  facecolor=PLOFAStyle.BG_DARK)
        fig.suptitle(
            f"PRESSURE MAP — {self.config.home_team} vs {self.config.away_team}",
            color=PLOFAStyle.TEXT_PRIMARY, fontsize=14, fontweight="bold"
        )

        teams = [self.config.home_team, self.config.away_team]

        for ax, team in zip(axes, teams):
            pitch = Pitch(pitch_type="statsbomb",
                          pitch_color=PLOFAStyle.PITCH_GREEN,
                          line_color=PLOFAStyle.PITCH_LINE)
            pitch.draw(ax=ax)

            # Collect press locations for this team
            press_locs = []
            for e in self.result.timeline:
                if e.event_type in (EventType.PRESS, EventType.PRESS_SUCCESS) and e.team == team:
                    if e.location_x is not None and e.location_y is not None:
                        press_locs.append((e.location_x, e.location_y))

            if press_locs:
                xs = [p[0] for p in press_locs]
                ys = [p[1] for p in press_locs]
                # Use kdeplot instead of heatmap to avoid mplsoccer API issues
                pitch.kdeplot(xs, ys, ax=ax, cmap="Reds", fill=True, alpha=0.6)

            ax.set_title(f"{team}\n{len(press_locs)} Pressures",
                         color=PLOFAStyle.TEXT_PRIMARY, fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=PLOFAStyle.BG_DARK)
        plt.close()
        print(f"   🔥 Pressure Map   → {filepath}")