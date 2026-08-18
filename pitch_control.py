"""
PLOFA 26/27 — PITCH CONTROL FIELD  (Checkpoint 22)
====================================================
pitch_control.py

Philosophy:
    Real space isn't about your distance to the ball — it's about who
    controls that patch of grass relative to everyone else. This module
    computes a simplified Fernandez/Bornn-style influence field over the
    pitch and derives:
        - per-cell team ownership (who controls this patch)
        - passing-lane danger (does the line cross opponent-controlled cells)
        - space-creation targets (uncontrolled cells with high xT value)
        - defensive compactness (area controlled in defensive third)

    Pure math. No physics engine. Cheap enough to recompute every minute.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Reuse the project's actual xT grid instead of reinventing one.
from advanced_valuation import AdvancedValuationEngine


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PITCH_X_MAX: float = 105.0
PITCH_Y_MAX: float = 68.0

# Grid resolution: ~5m cells gives 21 x 14 = 294 cells.
# Cheap to recompute every minute; fine enough for tactical shape.
CELL_SIZE: float = 5.0
N_COLS: int = int(math.ceil(PITCH_X_MAX / CELL_SIZE))   # 21
N_ROWS: int = int(math.ceil(PITCH_Y_MAX / CELL_SIZE))   # 14

# Influence falloff: sigma = pace_scaled_radius.
# Base sigma = 8m; pace 90+ stretches to ~11m, pace <40 shrinks to ~5m.
BASE_SIGMA: float = 8.0
SIGMA_PACE_SCALE: float = 0.035   # sigma += pace * SIGMA_PACE_SCALE

# Passing lane sampling: number of points to test along the straight line
# ball -> receiver. 8 samples is enough for a 5m grid.
LANE_SAMPLE_STEPS: int = 8

# xT lookup stub — replaced by ThreatEngine's spatial xT grid when wired in.
# Values are per-cell expected threat (0..~0.15). Neutral mid-cell ~0.02.
DEFAULT_XT_GRID: Dict[Tuple[int, int], float] = {}


# xT lookup: use the project's real 12x8 AdvancedValuationEngine grid,
# mapped onto our 5m cells. This avoids a second hand-authored xT table.
# We create one shared engine instance; get_xt_value() is pure lookup.
_XT_ENGINE = AdvancedValuationEngine()


def _cell_xt(col: int, row: int, cell_size: float, n_cols: int, n_rows: int) -> float:
    cx = (col + 0.5) * cell_size
    cy = (row + 0.5) * cell_size
    return _XT_ENGINE.get_xt_value(cx, cy)


def _build_default_xt_grid() -> Dict[Tuple[int, int], float]:
    g: Dict[Tuple[int, int], float] = {}
    for col in range(N_COLS):
        for row in range(N_ROWS):
            g[(col, row)] = round(_cell_xt(col, row, CELL_SIZE, N_COLS, N_ROWS), 4)
    return g


DEFAULT_XT_GRID: Dict[Tuple[int, int], float] = _build_default_xt_grid()


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PlayerInfluenceInput:
    """One player's spatial data for pitch-control computation."""
    name: str
    team: str
    position: str
    x: float
    y: float
    pace: float = 60.0
    is_goalkeeper: bool = False


@dataclass
class PitchControlResult:
    """Full pitch-control snapshot at one moment in time."""
    # Per-cell ownership: col,row -> "home" | "away" | "neutral"
    ownership: Dict[Tuple[int, int], str] = field(default_factory=dict)
    # Per-cell net influence: positive = home优势, negative = away优势
    net_influence: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # Aggregate stats
    home_controlled_cells: int = 0
    away_controlled_cells: int = 0
    neutral_cells: int = 0
    home_controlled_area_pct: float = 0.0
    away_controlled_area_pct: float = 0.0
    # Defensive third compactness (0-100, 100 = perfectly packed)
    home_defensive_compactness: float = 0.0
    away_defensive_compactness: float = 0.0
    # Minute this snapshot was taken
    minute: int = 0

    def as_dict(self) -> Dict:
        return {
            "minute": self.minute,
            "home_controlled_pct": round(self.home_controlled_area_pct, 1),
            "away_controlled_pct": round(self.away_controlled_area_pct, 1),
            "neutral_pct": round(self.neutral_cells / (N_COLS * N_ROWS) * 100, 1),
            "home_defensive_compactness": round(self.home_defensive_compactness, 1),
            "away_defensive_compactness": round(self.away_defensive_compactness, 1),
        }


@dataclass
class PassingLaneRisk:
    """Risk assessment for one passing lane."""
    lane_name: str          # e.g. "CB->RW"
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    risk_level: float = 0.0  # 0..1
    controlled_crossed: int = 0
    total_sampled: int = 0
    # Which cells along the lane are opponent-controlled
    danger_cells: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.risk_level < 0.25:
            return "safe"
        elif self.risk_level < 0.50:
            return "moderate"
        elif self.risk_level < 0.75:
            return "dangerous"
        return "critical"


@dataclass
class SpaceTarget:
    """A recommended off-ball run target for an attacker."""
    col: int
    row: int
    x: float
    y: float
    score: float = 0.0
    controlled_by: str = "neutral"
    xt_value: float = 0.0


# ─────────────────────────────────────────────
# PITCH CONTROL FIELD
# ─────────────────────────────────────────────

class PitchControlField:
    """
    Computes and caches a per-cell influence/ownership grid.

    Recompute every minute from live player positions. The result drives
    passing-lane danger, space-creation targets, and defensive compactness.
    """

    def __init__(
        self,
        n_cols: int = N_COLS,
        n_rows: int = N_ROWS,
        cell_size: float = CELL_SIZE,
        base_sigma: float = BASE_SIGMA,
        sigma_pace_scale: float = SIGMA_PACE_SCALE,
    ):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.cell_size = cell_size
        self.base_sigma = base_sigma
        self.sigma_pace_scale = sigma_pace_scale

        self._last_result: Optional[PitchControlResult] = None
        self._last_minute: int = -1
        self._last_key: str = ""

    # ── PUBLIC API ────────────────────────────────────────────

    def compute(
        self,
        home_players: List[PlayerInfluenceInput],
        away_players: List[PlayerInfluenceInput],
        minute: int = 0,
        xt_grid: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> PitchControlResult:
        """
        Recompute the full pitch-control grid.

        Args:
            home_players: all 11 home players with live positions
            away_players: all 11 away players with live positions
            minute: current match minute (for result labelling)
            xt_grid: optional per-cell xT values (uses default if None)

        Returns:
            PitchControlResult with ownership, net influence, compactness.
        """
        key = self._snapshot_key(home_players, away_players)
        if minute == self._last_minute and key == self._last_key:
            return self._last_result

        xt = xt_grid if xt_grid is not None else DEFAULT_XT_GRID
        ownership: Dict[Tuple[int, int], str] = {}
        net: Dict[Tuple[int, int], float] = {}

        home_controlled = 0
        away_controlled = 0
        neutral = 0

        for col in range(self.n_cols):
            for row in range(self.n_rows):
                cx = (col + 0.5) * self.cell_size
                cy = (row + 0.5) * self.cell_size

                h_inf = self._team_influence(home_players, cx, cy)
                a_inf = self._team_influence(away_players, cx, cy)
                net_val = h_inf - a_inf
                net[(col, row)] = round(net_val, 4)

                if net_val > 0.15:
                    ownership[(col, row)] = "home"
                    home_controlled += 1
                elif net_val < -0.15:
                    ownership[(col, row)] = "away"
                    away_controlled += 1
                else:
                    ownership[(col, row)] = "neutral"
                    neutral += 1

        total = self.n_cols * self.n_rows
        home_def_compact = self._defensive_compactness(
            home_players, ownership, "home"
        )
        away_def_compact = self._defensive_compactness(
            away_players, ownership, "away"
        )

        result = PitchControlResult(
            ownership=ownership,
            net_influence=net,
            home_controlled_cells=home_controlled,
            away_controlled_cells=away_controlled,
            neutral_cells=neutral,
            home_controlled_area_pct=home_controlled / total * 100.0,
            away_controlled_area_pct=away_controlled / total * 100.0,
            home_defensive_compactness=home_def_compact,
            away_defensive_compactness=away_def_compact,
            minute=minute,
        )
        self._last_result = result
        self._last_minute = minute
        self._last_key = key
        return result

    def passing_lane_risk(
        self,
        result: PitchControlResult,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        attacking_team: str,
        lane_name: str = "",
    ) -> PassingLaneRisk:
        """
        Sample cells along the straight line start->end and count how many
        are opponent-controlled. Returns a PassingLaneRisk.

        Args:
            attacking_team: "home" or "away" — the team making the pass.
            Only cells controlled by the OPPONENT count as danger.
        """
        opponent = "away" if attacking_team == "home" else "home"
        dx = end_x - start_x
        dy = end_y - start_y
        total = 0
        crossed = 0
        danger_cells: List[Tuple[int, int]] = []

        for i in range(LANE_SAMPLE_STEPS + 1):
            t = i / LANE_SAMPLE_STEPS
            px = start_x + dx * t
            py = start_y + dy * t
            col = min(self.n_cols - 1, max(0, int(px // self.cell_size)))
            row = min(self.n_rows - 1, max(0, int(py // self.cell_size)))
            cell = (col, row)
            total += 1
            owner = result.ownership.get(cell, "neutral")
            if owner == opponent:
                crossed += 1
                danger_cells.append(cell)

        risk = crossed / total if total > 0 else 0.0
        return PassingLaneRisk(
            lane_name=lane_name,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            risk_level=round(risk, 3),
            controlled_crossed=crossed,
            total_sampled=total,
            danger_cells=danger_cells,
        )

    def space_creation_targets(
        self,
        result: PitchControlResult,
        team: str,
        attacker_x: float,
        attacker_y: float,
        reachable_radius: float = 28.0,
        max_targets: int = 5,
        xt_grid: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> List[SpaceTarget]:
        """
        Find the best off-ball run targets for an attacker: cells within
        reachable_radius that are NOT controlled by the opponent and have
        high xT value, with a small repulsion penalty from teammates.

        `team` is the attacker's team; opponent cells are avoided.
        """
        xt = xt_grid if xt_grid is not None else DEFAULT_XT_GRID
        opponent = "away" if team == "home" else "home"
        targets: List[SpaceTarget] = []

        a_col = min(self.n_cols - 1, max(0, int(attacker_x // self.cell_size)))
        a_row = min(self.n_rows - 1, max(0, int(attacker_y // self.cell_size)))

        for col in range(self.n_cols):
            for row in range(self.n_rows):
                cx = (col + 0.5) * self.cell_size
                cy = (row + 0.5) * self.cell_size
                dist = math.hypot(cx - attacker_x, cy - attacker_y)
                if dist > reachable_radius:
                    continue

                owner = result.ownership.get((col, row), "neutral")
                if owner == opponent:
                    continue

                xt_val = xt.get((col, row), 0.0)
                if xt_val <= 0.0:
                    continue

                # Simple repulsion: if many teammates already own this cell,
                # downgrade it (so two forwards don't converge).
                # We approximate teammate density from net influence sign.
                net = result.net_influence.get((col, row), 0.0)
                teammate_density = max(0.0, net) if team == "home" else max(0.0, -net)
                repulsion = 1.0 / (1.0 + teammate_density * 0.5)

                score = xt_val * repulsion * max(0.2, 1.0 - dist / (reachable_radius + 1.0))
                targets.append(SpaceTarget(
                    col=col, row=row, x=cx, y=cy,
                    score=round(score, 4),
                    controlled_by=owner,
                    xt_value=round(xt_val, 4),
                ))

        targets.sort(key=lambda t: -t.score)
        return targets[:max_targets]

    def cell_ownership(self, result: PitchControlResult, x: float, y: float) -> str:
        """Who controls the cell containing (x, y)?"""
        col = min(self.n_cols - 1, max(0, int(x // self.cell_size)))
        row = min(self.n_rows - 1, max(0, int(y // self.cell_size)))
        return result.ownership.get((col, row), "neutral")

    # ── INTERNALS ─────────────────────────────────────────────

    @staticmethod
    def _influence(player: PlayerInfluenceInput, x: float, y: float) -> float:
        """
        Fernandez/Bornn-style influence:

            influence = 1 / (1 + dist^2 / (2 * sigma^2))

        sigma scales with pace: faster players control a wider radius.
        Goalkeepers have a smaller sigma (they don't dominate wide areas).
        """
        dx = x - player.x
        dy = y - player.y
        dist_sq = dx * dx + dy * dy
        sigma = max(3.0, BASE_SIGMA + player.pace * SIGMA_PACE_SCALE)
        if player.is_goalkeeper:
            sigma *= 0.6
        return 1.0 / (1.0 + dist_sq / (2.0 * sigma * sigma))

    def _team_influence(
        self,
        players: List[PlayerInfluenceInput],
        x: float,
        y: float,
    ) -> float:
        return sum(self._influence(p, x, y) for p in players)

    def _defensive_compactness(
        self,
        players: List[PlayerInfluenceInput],
        ownership: Dict[Tuple[int, int], str],
        team: str,
    ) -> float:
        """
        What fraction of cells the team controls in their own defensive third.
        Normalised to 0..100 where 100 = every cell in the defensive third
        is controlled (a classic parked bus).
        """
        opponent_third_col_max = 2 if team == "home" else self.n_cols - 3
        if team == "home":
            controlled = sum(
                1 for (col, row), owner in ownership.items()
                if col <= 2 and owner == team
            )
            total_third = (3) * self.n_rows
        else:
            controlled = sum(
                1 for (col, row), owner in ownership.items()
                if col >= self.n_cols - 3 and owner == team
            )
            total_third = (3) * self.n_rows

        if total_third <= 0:
            return 0.0
        return round(controlled / total_third * 100.0, 1)

    def _snapshot_key(
        self,
        home: List[PlayerInfluenceInput],
        away: List[PlayerInfluenceInput],
    ) -> str:
        def _key(players: List[PlayerInfluenceInput]) -> str:
            return "|".join(
                f"{p.team}:{p.name}:{p.x:.1f}:{p.y:.1f}:{p.pace:.0f}"
                for p in sorted(players, key=lambda p: p.name)
            )
        return _key(home) + "//" + _key(away)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python pitch_control.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏟️  PLOFA 26/27 — Pitch Control Field (Checkpoint 22) Standalone Demo")
    print("=" * 70)

    field_ = PitchControlField()

    home = [
        PlayerInfluenceInput("GK", "home", "GK", 8.0, 34.0, pace=60.0, is_goalkeeper=True),
        PlayerInfluenceInput("CB1", "home", "CB", 22.0, 24.0, pace=70.0),
        PlayerInfluenceInput("CB2", "home", "CB", 22.0, 44.0, pace=72.0),
        PlayerInfluenceInput("LB", "home", "LB", 26.0, 8.0, pace=78.0),
        PlayerInfluenceInput("RB", "home", "RB", 26.0, 60.0, pace=80.0),
        PlayerInfluenceInput("CDM", "home", "CDM", 40.0, 34.0, pace=68.0),
        PlayerInfluenceInput("CM1", "home", "CM", 52.0, 24.0, pace=74.0),
        PlayerInfluenceInput("CM2", "home", "CM", 52.0, 44.0, pace=72.0),
        PlayerInfluenceInput("CAM", "home", "CAM", 66.0, 34.0, pace=71.0),
        PlayerInfluenceInput("LW", "home", "LW", 82.0, 8.0, pace=88.0),
        PlayerInfluenceInput("RW", "home", "RW", 82.0, 60.0, pace=86.0),
        PlayerInfluenceInput("ST", "home", "ST", 88.0, 34.0, pace=82.0),
    ]
    away = [
        PlayerInfluenceInput("aGK", "away", "GK", 97.0, 34.0, pace=62.0, is_goalkeeper=True),
        PlayerInfluenceInput("aCB1", "away", "CB", 83.0, 24.0, pace=69.0),
        PlayerInfluenceInput("aCB2", "away", "CB", 83.0, 44.0, pace=71.0),
        PlayerInfluenceInput("aLB", "away", "LB", 79.0, 8.0, pace=77.0),
        PlayerInfluenceInput("aRB", "away", "RB", 79.0, 60.0, pace=79.0),
        PlayerInfluenceInput("aCDM", "away", "CDM", 65.0, 34.0, pace=67.0),
        PlayerInfluenceInput("aCM1", "away", "CM", 53.0, 24.0, pace=73.0),
        PlayerInfluenceInput("aCM2", "away", "CM", 53.0, 44.0, pace=71.0),
        PlayerInfluenceInput("aCAM", "away", "CAM", 39.0, 34.0, pace=70.0),
        PlayerInfluenceInput("aLW", "away", "LW", 23.0, 8.0, pace=87.0),
        PlayerInfluenceInput("aRW", "away", "RW", 23.0, 60.0, pace=85.0),
        PlayerInfluenceInput("aST", "away", "ST", 17.0, 34.0, pace=81.0),
    ]

    result = field_.compute(home, away, minute=30)
    d = result.as_dict()
    print("\n1. PITCH OWNERSHIP (minute 30, home attacks right)\n")
    print(f"   Home controlled: {d['home_controlled_pct']:.1f}%")
    print(f"   Away controlled: {d['away_controlled_pct']:.1f}%")
    print(f"   Neutral:         {d['neutral_pct']:.1f}%")
    print(f"   Home defensive compactness: {d['home_defensive_compactness']:.1f}/100")
    print(f"   Away defensive compactness: {d['away_defensive_compactness']:.1f}/100")

    print("\n2. PASSING LANE RISK (home GK -> ST through defensive third)\n")
    lane = field_.passing_lane_risk(result, 8.0, 34.0, 88.0, 34.0, attacking_team="home", lane_name="GK->ST")
    print(f"   Lane: {lane.lane_name}  band={lane.band}  risk={lane.risk_level:.2f}")
    print(f"   Opponent-controlled cells crossed: {lane.controlled_crossed}/{lane.total_sampled}")

    safe_lane = field_.passing_lane_risk(result, 52.0, 24.0, 82.0, 8.0, attacking_team="home", lane_name="CM->LW")
    print(f"   Lane: {safe_lane.lane_name}  band={safe_lane.band}  risk={safe_lane.risk_level:.2f}")

    print("\n3. SPACE CREATION TARGETS (home ST looking for runs)\n")
    targets = field_.space_creation_targets(result, "home", 88.0, 34.0, reachable_radius=30.0)
    for t in targets[:5]:
        print(f"   cell({t.col:>2},{t.row:>2}) = ({t.x:.1f},{t.y:.1f})  "
              f"score={t.score:.4f}  owner={t.controlled_by}  xT={t.xt_value:.4f}")

    print("\n4. CELL OWNERSHIP AT KEY POINTS\n")
    for label, x, y in [
        ("home box centre", 12.0, 34.0),
        ("halfway line", 52.5, 34.0),
        ("away box centre", 93.0, 34.0),
        ("right channel", 80.0, 58.0),
    ]:
        owner = field_.cell_ownership(result, x, y)
        print(f"   {label:<22} ({x:.0f},{y:.0f}) -> {owner}")

    print("\n✅ Pitch Control Field operational — pure math, no physics engine.")
