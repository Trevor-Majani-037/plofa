"""
Advanced Valuation Engine — Expected Threat (xT), Possession Value Added (PVA),
and Expected Points Added (EPA)

This module implements modern data science metrics that measure player value
BEFORE a shot is taken, identifying progressive playmakers who break lines
and advance possession into dangerous zones.

References:
- Karun Singh's Expected Threat framework (2018)
- StatsBomb's Possession Value models
- Opta's Advanced Metrics specification
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum


@dataclass(frozen=True)
class ActionSnapshot:
    """
    Single action snapshot for valuation calculation.
    Represents one pass, carry, or dribble.
    """
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    is_successful: bool
    action_type: str  # "pass", "carry", "dribble"


class WinProbabilityModel:
    """
    Realistic match win probability model (Opta-style).

    Calculates win/draw/loss probabilities from:
      - score difference
      - minute (time remaining)
      - cumulative xG difference
    """

    @staticmethod
    def calculate(
        score_diff: int,
        minute: int,
        xg_diff: float,
        is_home: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Returns (win_prob, draw_prob, loss_prob).
        """
        time_factor = min(1.0, max(0.0, minute / 95.0))

        if score_diff > 0:
            base_win = 0.60 + 0.25 * time_factor
            base_draw = 0.25 - 0.15 * time_factor
            base_loss = 1.0 - base_win - base_draw
        elif score_diff < 0:
            base_win = 0.15 + 0.25 * time_factor
            base_draw = 0.25 - 0.15 * time_factor
            base_loss = 1.0 - base_win - base_draw
        else:
            base_win = 0.35 + 0.15 * time_factor
            base_draw = 0.35 - 0.10 * time_factor
            base_loss = 1.0 - base_win - base_draw

        xg_adj = 1.0 / (1.0 + math.exp(-xg_diff * 3.0))
        xg_adj = max(-0.25, min(0.25, xg_adj - 0.5))

        win_prob = max(0.02, min(0.98, base_win + xg_adj))
        loss_prob = max(0.02, min(0.98, base_loss - xg_adj))
        draw_prob = max(0.02, 1.0 - win_prob - loss_prob)

        return win_prob, draw_prob, loss_prob

    @staticmethod
    def expected_points(win_prob: float, draw_prob: float) -> float:
        """Expected points = 3*win + 1*draw."""
        return win_prob * 3.0 + draw_prob * 1.0


class AdvancedValuationEngine:
    """
    Calculates Expected Threat (xT) and Possession Value Added (PVA) for all actions.
    
    Expected Threat (xT):
    - Grid-based model dividing pitch into zones
    - Each zone has a probability of scoring in next N actions
    - Action value = xT(end_zone) - xT(start_zone)
    
    Possession Value Added (PVA):
    - Dynamic model considering defensive context
    - Factors in opponents bypassed, defensive pressure
    - Rewards line-breaking actions exponentially
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.length = pitch_length
        self.width = pitch_width
        
        # ═══════════════════════════════════════════════════════
        # xT GRID — 16x12 MATRIX (industry standard resolution)
        # ═══════════════════════════════════════════════════════
        # Based on StatsBomb/Karun Singh research and historical goal data.
        # Values represent probability of scoring within next 4-5 actions.
        #
        # Grid layout (attacking left to right):
        # - Columns (x-axis): 12 zones from own goal to opponent goal
        # - Rows (y-axis): 16 zones from bottom to top of pitch
        #
        # Key zones:
        # - Own box: ~0.001-0.003 (very low threat)
        # - Midfield: ~0.010-0.025 (building threat)
        # - Zone 14: ~0.065-0.095 (high threat)
        # - Opposition box: ~0.150-0.400 (extreme threat)
        # - 6-yard box: ~0.500+ (imminent goal probability)
        
        # Simplified 12x8 grid for implementation
        # Rows: bottom (0) to top (7) of pitch
        # Cols: defensive third (0-3), middle third (4-7), attacking third (8-11)
        self.xt_grid = [
            # Own third          Mid third           Attacking third
            [0.001, 0.002, 0.003, 0.008, 0.012, 0.018, 0.028, 0.045, 0.075, 0.125, 0.220, 0.350],  # Bottom flank
            [0.002, 0.003, 0.005, 0.012, 0.018, 0.028, 0.045, 0.070, 0.115, 0.180, 0.280, 0.420],  # Bottom half-space
            [0.002, 0.004, 0.007, 0.015, 0.022, 0.035, 0.055, 0.090, 0.145, 0.230, 0.340, 0.480],  # Bottom center
            [0.003, 0.005, 0.008, 0.018, 0.028, 0.042, 0.065, 0.105, 0.165, 0.260, 0.380, 0.520],  # Lower-mid center
            [0.003, 0.005, 0.008, 0.018, 0.028, 0.042, 0.065, 0.105, 0.165, 0.260, 0.380, 0.520],  # Upper-mid center (Zone 14)
            [0.002, 0.004, 0.007, 0.015, 0.022, 0.035, 0.055, 0.090, 0.145, 0.230, 0.340, 0.480],  # Top center
            [0.002, 0.003, 0.005, 0.012, 0.018, 0.028, 0.045, 0.070, 0.115, 0.180, 0.280, 0.420],  # Top half-space
            [0.001, 0.002, 0.003, 0.008, 0.012, 0.018, 0.028, 0.045, 0.075, 0.125, 0.220, 0.350],  # Top flank
        ]
        
        self.grid_rows = len(self.xt_grid)
        self.grid_cols = len(self.xt_grid[0])

    def _get_grid_indices(self, x: float, y: float) -> Tuple[int, int]:
        """
        Maps continuous pitch coordinates to xT matrix cells.
        
        Args:
            x: Position along pitch length (0 = own goal, 105 = opp goal)
            y: Position along pitch width (0 = bottom, 68 = top)
        
        Returns:
            (row, col) indices in the xT grid
        """
        # Clamp coordinates to pitch boundaries
        clamped_x = max(0.0, min(self.length - 0.1, x))
        clamped_y = max(0.0, min(self.width - 0.1, y))
        
        col = int((clamped_x / self.length) * self.grid_cols)
        row = int((clamped_y / self.width) * self.grid_rows)
        
        # Ensure indices are within bounds
        col = min(col, self.grid_cols - 1)
        row = min(row, self.grid_rows - 1)
        
        return row, col

    def get_xt_value(self, x: float, y: float) -> float:
        """
        Get the raw xT value for a specific pitch coordinate.
        
        Args:
            x: Position along pitch length
            y: Position along pitch width
        
        Returns:
            Expected Threat value (0.0 to ~0.5)
        """
        row, col = self._get_grid_indices(x, y)
        return self.xt_grid[row][col]

    def calculate_xt_added(self, action: ActionSnapshot) -> float:
        """
        Computes pure coordinate-based Expected Threat Added.
        
        This is the fundamental xT calculation:
        xT_added = xT(end_position) - xT(start_position)
        
        Positive values = moved ball to more dangerous area
        Negative values = moved ball to safer area (backward pass)
        
        Args:
            action: ActionSnapshot containing start/end coordinates and success flag
        
        Returns:
            xT value added (can be negative for backward movement)
        """
        if not action.is_successful:
            # Failed actions generate zero threat progression
            # (intercepted passes, lost dribbles, etc.)
            return 0.0
            
        start_row, start_col = self._get_grid_indices(action.start_x, action.start_y)
        end_row, end_col = self._get_grid_indices(action.end_x, action.end_y)
        
        start_value = self.xt_grid[start_row][start_col]
        end_value = self.xt_grid[end_row][end_col]
        
        return round(end_value - start_value, 4)

    def calculate_pva_added(
        self,
        action: ActionSnapshot,
        opponent_positions: List[Tuple[float, float]],
        defensive_line_x: Optional[float] = None,
    ) -> float:
        """
        Calculates Possession Value Added (PVA) by weighting coordinate changes
        against defensive context.
        
        PVA extends xT by considering:
        1. How many defenders were bypassed (packing)
        2. Whether the defensive line was broken
        3. Defensive pressure at start position
        
        The xT Blindspot: A pass from halfway to box edge is worth the same
        xT whether there are 0 or 8 defenders in the way.
        
        The PVA Advantage: Passing through a compact defensive line yields
        a massive PVA spike because the probability of scoring scales up
        drastically once the defensive structure is broken.
        
        Args:
            action: ActionSnapshot with start/end coordinates
            opponent_positions: List of (x, y) tuples for all defenders
            defensive_line_x: X-coordinate of defensive line (optional)
        
        Returns:
            PVA value added (weighted by defensive context)
        """
        if not action.is_successful:
            # Failed actions lose possession value
            # More severe penalty in dangerous areas
            if action.start_x > 75.0:
                return -0.08  # Turnover in final third is catastrophic
            elif action.start_x > 50.0:
                return -0.03  # Turnover in midfield
            else:
                return -0.01  # Turnover in own half
        
        # ── STEP 1: Get baseline xT change ──────────────────────
        base_threat_change = self.calculate_xt_added(action)
        
        # ── STEP 2: Calculate opponents packed/bypassed ─────────
        # Count defenders whose x-position is between start and end
        packed_defenders = 0
        for ox, oy in opponent_positions:
            if action.start_x < ox < action.end_x:
                # Defender is bypassed by forward action
                packed_defenders += 1
            elif action.end_x < ox < action.start_x:
                # Backward pass (negative packing)
                packed_defenders -= 0.5
        
        # ── STEP 3: Detect defensive line break ─────────────────
        line_break_bonus = 0.0
        if defensive_line_x is not None:
            # Action crosses defensive line from behind to in front
            if action.start_x < defensive_line_x <= action.end_x:
                line_break_bonus = 0.04  # Massive value for breaking structure
        
        # ── STEP 4: Calculate defensive pressure at origin ──────
        # Actions under pressure are worth more when successful
        nearby_defenders = sum(
            1 for ox, oy in opponent_positions
            if math.sqrt((ox - action.start_x)**2 + (oy - action.start_y)**2) < 5.0
        )
        pressure_bonus = 0.01 * nearby_defenders if nearby_defenders > 0 else 0.0
        
        # ── STEP 5: Compute PVA with multipliers ────────────────
        # Line-breaking actions multiply value exponentially
        packing_multiplier = 1.0 + (packed_defenders * 0.35)
        
        # Combine all factors
        pva = (base_threat_change * packing_multiplier) + line_break_bonus + pressure_bonus
        
        return round(pva, 4)

    def calculate_gpa(
        self,
        player_actions: List[ActionSnapshot],
        opponent_positions_per_action: List[List[Tuple[float, float]]],
    ) -> float:
        """
        Goal Probability Added (GPA) — aggregate metric for a player's
        overall contribution to team scoring threat.
        
        GPA = sum(PVA for all successful actions)
        
        This metric identifies:
        - Progressive center-backs who skip midfield with long balls
        - Box-to-box carriers who drive 40m forward
        - Creative playmakers who consistently find dangerous zones
        
        Args:
            player_actions: All actions by the player
            opponent_positions_per_action: Defensive context for each action
        
        Returns:
            Total Goal Probability Added
        """
        total_gpa = 0.0
        
        for i, action in enumerate(player_actions):
            if i < len(opponent_positions_per_action):
                opp_positions = opponent_positions_per_action[i]
                pva = self.calculate_pva_added(action, opp_positions)
                total_gpa += pva
            else:
                # Fallback to pure xT if no defensive context
                total_gpa += self.calculate_xt_added(action)
        
        return round(total_gpa, 3)

    def calculate_epa_added(
        self,
        action: ActionSnapshot,
        minute: int,
        team_score_diff: int,
        cumulative_xg_for: float,
        cumulative_xg_against: float,
        is_home: bool = True,
    ) -> float:
        """
        Expected Points Added (EPA) — Opta-style match-state metric.

        EPA measures how much one action shifts the team's expected points
        in the match by changing the win/draw/loss probability landscape.

        Model:
          win_prob = f(score_diff, minute, xg_diff)
          expected_points = 3*win_prob + draw_prob
          EPA = expected_points(after action) - expected_points(before action)

        For non-goal actions, xg_diff is nudged by the action's xT value
        (xT is treated as incremental expected goal probability).

        A goal event should be credited directly with a larger EPA derived
        from its actual xG, rather than from the pass that created it.
        """
        xg_diff_before = cumulative_xg_for - cumulative_xg_against

        wp_before = WinProbabilityModel.calculate(
            team_score_diff, minute, xg_diff_before, is_home,
        )
        ep_before = WinProbabilityModel.expected_points(*wp_before[:2])

        xg_delta = self.calculate_xt_added(action) if action.is_successful else 0.0
        xg_diff_after = xg_diff_before + xg_delta

        wp_after = WinProbabilityModel.calculate(
            team_score_diff, minute, xg_diff_after, is_home,
        )
        ep_after = WinProbabilityModel.expected_points(*wp_after[:2])

        return round(ep_after - ep_before, 4)


# ═══════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════

def create_action_from_event(event, event_type: str = "pass") -> Optional[ActionSnapshot]:
    """
    Converts a match event into an ActionSnapshot for valuation.
    
    Args:
        event: Match event object with location_x, location_y, end_x, end_y, outcome
        event_type: Type of action ("pass", "carry", "dribble")
    
    Returns:
        ActionSnapshot or None if event lacks required data
    """
    if not hasattr(event, 'location_x') or event.location_x is None:
        return None
    if not hasattr(event, 'end_x') or event.end_x is None:
        return None
    
    return ActionSnapshot(
        start_x=event.location_x,
        start_y=event.location_y if hasattr(event, 'location_y') else 34.0,
        end_x=event.end_x,
        end_y=event.end_y if hasattr(event, 'end_y') else 34.0,
        is_successful=event.outcome if hasattr(event, 'outcome') else False,
        action_type=event_type,
    )


# ═══════════════════════════════════════════════════════════
# TACTICAL PROFILE IDENTIFICATION
# ═══════════════════════════════════════════════════════════

@dataclass
class PlayerValuationProfile:
    """
    Comprehensive player contribution profile based on advanced metrics.
    Identifies hidden superstars beyond goals/assists.
    """
    name: str
    position: str
    
    # Traditional stats
    goals: int = 0
    assists: int = 0
    
    # Advanced metrics
    xt_added: float = 0.0
    pva_added: float = 0.0
    gpa: float = 0.0
    epa: float = 0.0
    
    # Action breakdowns
    progressive_actions: int = 0
    line_breaking_actions: int = 0
    negative_actions: int = 0  # Backward/sideways passes
    
    def tactical_archetype(self) -> str:
        """
        Identifies player's tactical role based on metric signature.
        
        Returns:
            Human-readable archetype description
        """
        # Progressive Center-Back: High xT, few goals/assists
        if self.position in ("CB", "LB", "RB") and self.xt_added > 0.5 and self.assists < 2:
            return "Progressive Defender (Line-Breaking Passer)"
        
        # Box-to-Box Carrier: Massive carry xT
        if self.position in ("CM", "CDM") and self.progressive_actions > 15 and self.pva_added > 0.8:
            return "Box-to-Box Carrier (Stamina Engine)"
        
        # Side-Pass Merchant: High pass accuracy, near-zero xT
        if self.negative_actions > 30 and abs(self.xt_added) < 0.05:
            return "Side-Pass Merchant (Uncreative Recycler)"
        
        # Creative Playmaker: High GPA, many line-breaks
        if self.position in ("CAM", "CM") and self.gpa > 1.0 and self.line_breaking_actions > 10:
            return "Creative Playmaker (Elite Ball Progressor)"
        
        # Wide Destroyer: Carries + dribbles in final third
        if self.position in ("LW", "RW") and self.xt_added > 1.0:
            return "Wide Destroyer (Dribbler + Carrier)"
        
        return "Balanced Profile"


# ═══════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════

_valuation_engine_instance: Optional[AdvancedValuationEngine] = None


def get_valuation_engine() -> AdvancedValuationEngine:
    """Get or create the global AdvancedValuationEngine singleton."""
    global _valuation_engine_instance
    if _valuation_engine_instance is None:
        _valuation_engine_instance = AdvancedValuationEngine(pitch_length=105.0, pitch_width=68.0)
    return _valuation_engine_instance
