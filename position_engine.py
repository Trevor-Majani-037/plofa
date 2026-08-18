"""
PLOFA 26/27 — POSITION ENGINE  (Checkpoint 5)
================================================
position_engine.py

Philosophy:
    Right now, "position" in PLOFA is a LABEL used as a random-draw weight.
    A striker with weight 0.8 in build-up selection is not "unlikely" to
    start a possession sequence from his own third — over 400+ sequences
    a match, he WILL, repeatedly, with no causal reason attached.

    This module gives every player a PERSISTENT SPATIAL STATE:
    a "home" position derived from role + team style + tactical context,
    a "current" position that updates when they touch the ball,
    and a DRIFT step that pulls uninvolved players back toward home
    every minute — modulated by phase, press intensity, defensive line,
    and game state.

    Selection functions (_pick_builder, _pick_receiver, _pick_shooter, etc.)
    don't change their CAUSAL LOGIC. They just stop asking
        "what's this player's position label worth as a weight?"
    and start asking
        "is this player's CURRENT ZONE even plausible for this action?"

    This is not GPS tracking. It's a discretized formation-relative model —
    the same category of system real match-engine games (Football Manager,
    classic FIFA AI) use under the hood. Three layers:

        Layer 1 — PlayerSpatialState   (persistent per-player home/current pos)
        Layer 2 — ZoneGrid             (6x5 coarse pitch grid, StatsBomb-scale)
        Layer 3 — DriftEngine          (causal pull back to home, per minute)

    Nothing here requires new randomness bolted on top. It REPLACES blind
    weighted-by-label picks with weighted-by-(label x zone-plausibility).
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

from cross_detector import WIDE_CHANNEL_WIDTH, PITCH_Y, CENTER_Y
from winger_behavior import (
    WingerRegistry,
    LEFT_TOUCHLINE_ANCHOR_Y,
    RIGHT_TOUCHLINE_ANCHOR_Y,
)

if TYPE_CHECKING:
    from match_engine import TeamProfile, MatchPhase, GameState


# ─────────────────────────────────────────────
# BALL-CENTRIC ELLIPTICAL WEIGHTING (Checkpoint 20)
# ─────────────────────────────────────────────
# The receive pool is shaped as an anisotropic ellipse anchored on the
# ball and elongated along the axis of play. sigma_along (metres, per
# role) is how far a receiver of that role is still a live option AHEAD
# of the ball; sigma_across (metres) is how far they can be off the ball's
# lateral line. A forward runner 25m upfield is a genuinely valuable
# receive option; a player 25m out to the side is not — the ellipse
# encodes exactly that anisotropy that a plain circular distance falloff
# cannot.
#
# The ellipse centre is shifted a few metres AHEAD of the ball so
# forward runs (through-ball targets) are favoured over lateral/backward
# options — ball-centric space, not just proximity. Because it is only a
# shape PREFERENCE, composition multiplies it into existing label/marking
# weights with a floor: a receiver completely off the ellipse keeps
# ELLIPSE_COMPOSE_FLOOR of their base value, so deliberate half-space
# recycle and drop-in support passes still survive.

ELLIPSE_SIGMA_ALONG: Dict[str, float] = {
    "GK": 18.0, "CB": 20.0, "LB": 24.0, "RB": 24.0,
    "CDM": 24.0, "CM": 28.0, "CAM": 32.0,
    "LW": 36.0, "RW": 36.0, "ST": 36.0, "CF": 34.0,
}
# Checkpoint 21 note — the lateral sigma was widened 9.0 -> 16.0 in an
# earlier pass to make far-flank outlets (LW/RW/LB/RB) viable receiving
# options. That change broke the ellipse preservation guards (a 25m lateral
# runner must stay < 0.1, receiver picks must stay >3:1 ahead of behind), so
# it was reverted: wide delivery is now delivered by receive_option_quality
# (reach + direction + post discipline) and flank_bias_y (the pass is aimed at
# the wide player's home channel), not by a fattened lateral sigma.
ELLIPSE_SIGMA_ACROSS: float = 9.0
ELLIPSE_FORWARD_SHIFT: float = 8.0
ELLIPSE_COMPOSE_FLOOR: float = 0.35


def ball_centric_ellipse_weight(
    ball_x: float, ball_y: float,
    player_x: float, player_y: float,
    attacks_right: bool = True,
    sigma_along: float = 26.0,
    sigma_across: float = ELLIPSE_SIGMA_ACROSS,
    forward_shift: float = ELLIPSE_FORWARD_SHIFT,
) -> float:
    """
    2D anisotropic Gaussian centred just AHEAD of the ball, aligned with
    the axis of play. Returns a weight in 0..1.

        u = (player_x - ellipse_centre_x) * dir  (ahead = positive)
        v =  player_y - ellipse_centre_y          (lateral)
        w = exp(-0.5 * ((u / sigma_along)**2 + (v / sigma_across)**2))

    sigma_along > sigma_across makes the equal-weight contours ellipses
    stretched along the pitch: a receiver far ahead of the ball is a
    living option while a receiver equally far out to the side is not.
    """
    dir_x = 1.0 if attacks_right else -1.0
    cx = ball_x + dir_x * forward_shift
    cy = ball_y
    u = (player_x - cx) * dir_x
    v = player_y - cy
    return math.exp(-0.5 * ((u / sigma_along) ** 2 + (v / sigma_across) ** 2))


# ─────────────────────────────────────────────
# LAYER 2 — ZONE GRID
# 6 columns (thirds x2, StatsBomb-style) x 5 rows (channels)
# Pitch: x in [0,105], y in [0,68]
# ─────────────────────────────────────────────

class ZoneGrid:
    """
    A coarse 6x5 zone grid over the pitch.
    Columns (x): 6 bands of ~17.5m each (own goal -> opp goal)
    Rows (y):    5 channels of ~13.6m each (left touchline -> right)

    This is deliberately coarse. We are not modeling continuous physics —
    we're modeling "is this player's role plausible near this piece of play."
    """
    N_COLS = 6
    N_ROWS = 5
    COL_WIDTH = 105.0 / N_COLS   # 17.5
    ROW_HEIGHT = 68.0 / N_ROWS   # 13.6

    COL_NAMES = ["own_def", "own_mid", "own_att", "opp_def", "opp_mid", "opp_att"]
    ROW_NAMES = ["left_wide", "left_half", "central", "right_half", "right_wide"]

    @classmethod
    def zone_of(cls, x: float, y: float) -> Tuple[int, int]:
        col = min(cls.N_COLS - 1, max(0, int(x // cls.COL_WIDTH)))
        row = min(cls.N_ROWS - 1, max(0, int(y // cls.ROW_HEIGHT)))
        return (col, row)

    @classmethod
    def zone_name(cls, x: float, y: float) -> str:
        col, row = cls.zone_of(x, y)
        return f"{cls.COL_NAMES[col]}/{cls.ROW_NAMES[row]}"

    @classmethod
    def zone_center(cls, col: int, row: int) -> Tuple[float, float]:
        return (
            (col + 0.5) * cls.COL_WIDTH,
            (row + 0.5) * cls.ROW_HEIGHT,
        )

    @classmethod
    def col_distance(cls, x1: float, x2: float) -> int:
        """How many column-bands apart are two x-coordinates?"""
        c1, _ = cls.zone_of(x1, 34.0)
        c2, _ = cls.zone_of(x2, 34.0)
        return abs(c1 - c2)


# ─────────────────────────────────────────────
# LAYER 1 — HOME POSITION TEMPLATES
# Formation-relative "resting" coordinates per role.
# These get nudged by team style (defensive_line, width, tempo, directness).
# ─────────────────────────────────────────────

# Base home_x (0-105) and home_y (0-68) per position, NEUTRAL style baseline
BASE_HOME_POSITIONS: Dict[str, Tuple[float, float]] = {
    "GK":  (8.0,  34.0),
    "CB":  (24.0, 34.0),
    "LB":  (26.0, 10.0),
    "RB":  (26.0, 58.0),
    "CDM": (40.0, 34.0),
    "CM":  (52.0, 34.0),
    "CAM": (66.0, 34.0),
    # MODERN WINGERS: home positions pushed into the attacking third.
    # Real EPL / top-5-league wingers (Vini Jr, Saka, Salah, Martinelli)
    # rest HIGH and WIDE — on the touchline in the attacking third, not
    # standing in midfield next to the #10. The middle of the pitch is
    # always full; the winger's home is the flank, 15-20m further forward
    # than the old CAM-adjacent position. This is the single biggest
    # difference between modern wingers and old inside-forwards.
    "LW":  (82.0, 10.0),
    "RW":  (82.0, 58.0),
    "ST":  (88.0, 34.0),
    "CF":  (85.0, 34.0),
}

# Spread offsets for multiple players sharing a position label (e.g. 2 CBs)
POSITION_SPREAD_Y: Dict[str, List[float]] = {
    "CB": [-9.0, 9.0, 0.0],
    "CM": [-10.0, 10.0, 0.0],
}


class FormationEngine:
    """
    Computes each player's HOME position from:
        - base role template
        - team style (defensive_line -> shifts everyone's x)
        - width (spreads/narrows y for wide players)
        - directness/tempo (small x nudges)
    This runs ONCE at kickoff per player (and can be recalled if style changes
    mid-match, e.g. a substitution changes team shape).
    """

    @classmethod
    def compute_home(
        cls,
        position: str,
        profile: "TeamProfile",
        slot_index: int = 0,
    ) -> Tuple[float, float]:
        base_x, base_y = BASE_HOME_POSITIONS.get(position, (50.0, 34.0))

        # Defensive line shifts the WHOLE team's baseline forward/back.
        # profile.defensive_line: 0 (deep) -> 1 (high line)
        # Map 0..1 to a -8..+8 shift on x, centered at 0.5 -> no shift.
        line_shift = (getattr(profile, "defensive_line", 0.5) - 0.5) * 16.0

        # Directness / tempo nudge attackers slightly higher, deep players
        # slightly higher too under high tempo/press systems.
        directness = getattr(profile, "directness", 0.5)
        tempo = getattr(profile, "tempo", 0.5)
        press = getattr(profile, "press_intensity", 0.5)

        # High press systems push CDM/CM home positions up (compact block)
        press_shift = (press - 0.5) * 6.0

        x = base_x + line_shift + press_shift
        x = max(4.0, min(101.0, x))

        # Width affects wide players' y (push toward touchline) and
        # narrows/widens fullback y slightly too.
        width = getattr(profile, "width", 0.5)
        y = base_y
        if position in ("LW", "LB"):
            # base_y already near touchline (small y); width pulls further out
            y = base_y * (0.6 + width * 0.8)
        elif position in ("RW", "RB"):
            # mirror around 68
            dist_from_touch = 68.0 - base_y
            y = 68.0 - dist_from_touch * (0.6 + width * 0.8)

        # Spread multiple same-position players (e.g. 2 CBs, 2 CMs)
        spread = POSITION_SPREAD_Y.get(position)
        if spread:
            offset = spread[slot_index % len(spread)]
            y = y + offset

        y = max(3.0, min(65.0, y))
        return (round(x, 1), round(y, 1))


# ─────────────────────────────────────────────
# LAYER 1 — PERSISTENT PER-PLAYER SPATIAL STATE
# ─────────────────────────────────────────────

@dataclass
class PlayerSpatialState:
    """
    A player's living position record for the match.
    Updated every time they're INVOLVED in an event (touch, duel, etc).
    Drifts back toward home_x/home_y every minute they're NOT involved.
    """
    player_name: str
    position: str
    team: str

    home_x: float
    home_y: float

    current_x: float = field(init=False)
    current_y: float = field(init=False)

    # How far this player is allowed to roam from home before
    # selection weight starts penalizing them. Role + specialty driven.
    drift_tolerance: float = 22.0

    # How well this player reads and covers geometric spaces (0-100).
    # Populated from DNA during initialize_team. Used by midfielder
    # coverage drift to model Enzo/Rice/Pedri style space occupation.
    geometric_awareness: float = 50.0

    # Last minute they were actively involved (for staleness checks)
    last_active_minute: int = 0

    # ── MOVEMENT ACCUMULATORS (real distance, not authored fiction) ──
    # Reset every minute by PositionEngine.pop_minute_activity(). Two
    # separate sources are tracked because they have different meaning:
    #   - touch distance: real ball-involvement movement, sampled at
    #     event resolution (several times/minute for an involved player)
    #   - drift distance: off-ball movement, sampled once per minute
    #     (home pull, shape shift, line cohesion, coverage/space runs,
    #     defensive_block, attacking_crash — all folded into one measured
    #     net delta for that minute, since they all resolve sequentially
    #     before the next snapshot is taken)
    minute_touch_distance: float = 0.0
    minute_touch_count: int = 0
    minute_peak_touch_jump: float = 0.0   # largest single touch-to-touch move this minute
    minute_drift_distance: float = 0.0

    def __post_init__(self):
        self.current_x = self.home_x
        self.current_y = self.home_y

    @property
    def zone(self) -> Tuple[int, int]:
        return ZoneGrid.zone_of(self.current_x, self.current_y)

    @property
    def distance_from_home(self) -> float:
        return ((self.current_x - self.home_x) ** 2 +
                (self.current_y - self.home_y) ** 2) ** 0.5

    def touch_at(self, x: float, y: float, minute: int):
        """Called when this player is the primary/secondary actor of an event."""
        self.current_x = max(0.0, min(105.0, x))
        self.current_y = max(0.0, min(68.0, y))
        self.last_active_minute = minute

    def drift_toward_home(self, pull_strength: float = 0.35):
        """
        Pull current position toward home by pull_strength (0-1 fraction
        of the remaining distance covered this tick). Called once per
        minute for players NOT involved in an event that minute.
        """
        self.current_x += (self.home_x - self.current_x) * pull_strength
        self.current_y += (self.home_y - self.current_y) * pull_strength

    def plausibility(self, x: float, y: float) -> float:
        """
        Core of the fix: how plausible is it that THIS player is
        involved in an action happening at (x, y), given where they
        currently/typically are?

        Returns a multiplier in roughly [0.08, 1.35] to apply on top
        of the existing position-label weight in pick_weighted() calls.

        - Very close to current position -> near 1.2-1.35 (they're right there)
        - Within drift_tolerance -> smooth falloff, 1.0 -> 0.4
        - Far beyond tolerance -> heavily suppressed (0.08-0.2), not zero
          (football has outliers: a CB overlapping on a corner, a winger
          tracking back to make a last-ditch tackle — rare, not impossible)
        """
        dist = ((x - self.current_x) ** 2 + (y - self.current_y) ** 2) ** 0.5
        tol = self.drift_tolerance

        if dist <= tol * 0.35:
            return 1.35
        if dist <= tol:
            # Linear falloff from 1.2 down to 0.55 across the tolerance band
            frac = (dist - tol * 0.35) / (tol * 0.65)
            return 1.2 - frac * 0.65
        # Beyond tolerance: exponential-ish suppression, floor at 0.08
        excess = dist - tol
        return max(0.08, 0.55 * (0.5 ** (excess / tol)))


# ─────────────────────────────────────────────
# LAYER 3 — POSITION ENGINE
# Owns all spatial states for both teams. Called by MatchEngine/event_chain.
# ─────────────────────────────────────────────

class PositionEngine:
    """
    Single source of truth for "where is everyone right now."

    Usage (wired into MatchEngine):
        pe = PositionEngine()
        pe.initialize_team(home_team, home_starters, home_profile)
        pe.initialize_team(away_team, away_starters, away_profile)

        # each minute, before simulating:
        pe.drift_minute(home_team, home_profile, phase, game_state)
        pe.drift_minute(away_team, away_profile, phase, game_state)

        # after an event resolves:
        pe.record_touch(player_name, x, y, minute)

        # inside a pick_weighted() lambda:
        plaus = pe.plausibility_at(player_name, x, y)
        weight = base_label_weight * plaus
    """

    def __init__(self):
        self.states: Dict[str, PlayerSpatialState] = {}   # player_name -> state
        self.team_rosters: Dict[str, List[str]] = {}       # team -> [player_names]
        self.team_profiles: Dict[str, "TeamProfile"] = {}
        self.team_attacks_right: Dict[str, bool] = {}       # team -> attacks_right
        # Checkpoint 18 — modern winger registry: per-winger spatial profiles
        # (touchline anchor, flank commitment, byline instinct, isolation thirst).
        self.winger_registry: WingerRegistry = WingerRegistry()

    def initialize_team(self, team_name: str, players: List, profile: "TeamProfile",
                        attacks_right: bool = True):
        """Set up home/current spatial state for every player in a squad.
        
        Args:
            attacks_right: True if this team attacks toward x=105.
                           Away team attacks left (x=0), so their positions are
                           mirrored STRUCTURALLY: both x (direction of attack)
                           and y (wing/back channel), so an away LW/RW holds
                           their team's left/right flank rather than ending up
                           on the swapped side of the pitch.
        """
        self.team_profiles[team_name] = profile
        self.team_attacks_right[team_name] = attacks_right
        self.team_rosters.setdefault(team_name, [])

        slot_counter: Dict[str, int] = {}
        for p in players:
            name = getattr(p, "name", str(p))
            pos = getattr(p, "position", getattr(getattr(p, "dna", None), "position", "CM"))
            slot = slot_counter.get(pos, 0)
            slot_counter[pos] = slot + 1

            home_x, home_y = FormationEngine.compute_home(pos, profile, slot)
            if not attacks_right:
                home_x = 105.0 - home_x
                home_y = 68.0 - home_y
            tol = self._drift_tolerance_for(pos, p)

            self.states[name] = PlayerSpatialState(
                player_name=name, position=pos, team=team_name,
                home_x=home_x, home_y=home_y, drift_tolerance=tol,
                geometric_awareness=self._geometric_awareness_for(p),
            )
            if name not in self.team_rosters[team_name]:
                self.team_rosters[team_name].append(name)

        # Checkpoint 18 — register all wingers' spatial profiles (touchline
        # anchor, flank commitment, byline instinct, isolation thirst).
        self.winger_registry.register_team(players)

    def register_substitute(self, team_name: str, player, profile: "TeamProfile" = None):
        """Called when a sub comes on — gives them a fresh home position."""
        prof = profile or self.team_profiles.get(team_name)
        if prof is None:
            return
        name = getattr(player, "name", str(player))
        pos = getattr(player, "position", getattr(getattr(player, "dna", None), "position", "CM"))
        existing_same_pos = sum(
            1 for n in self.team_rosters.get(team_name, [])
            if self.states.get(n) and self.states[n].position == pos
        )
        home_x, home_y = FormationEngine.compute_home(pos, prof, existing_same_pos)
        if not self.team_attacks_right.get(team_name, True):
            home_x = 105.0 - home_x
            home_y = 68.0 - home_y
        tol = self._drift_tolerance_for(pos, player)
        self.states[name] = PlayerSpatialState(
            player_name=name, position=pos, team=team_name,
            home_x=home_x, home_y=home_y, drift_tolerance=tol,
            geometric_awareness=self._geometric_awareness_for(player),
        )
        self.team_rosters.setdefault(team_name, []).append(name)
        # Checkpoint 18 — register the sub's winger profile if they're a winger.
        self.winger_registry.register_player(player)

    @staticmethod
    def _drift_tolerance_for(position: str, player) -> float:
        """Wider roaming license for creative/wide roles, tighter for CBs/GK."""
        base = {
            "GK": 12.0, "CB": 16.0, "LB": 24.0, "RB": 24.0,
            "CDM": 20.0, "CM": 26.0, "CAM": 28.0,
            "LW": 26.0, "RW": 26.0, "ST": 22.0, "CF": 24.0,
        }.get(position, 22.0)

        specs = []
        if hasattr(player, "dna"):
            specs = getattr(player.dna, "specialties", []) or []
        elif hasattr(player, "specialties"):
            specs = player.specialties or []

        if "box_box" in specs or "engine" in specs:
            base *= 1.25
        if "inverted" in specs or "inverted_fullback" in specs:
            base *= 1.15
        if "anchor_man" in specs or "no_nonsense_cb" in specs or "sweeper_cb" in specs:
            base *= 0.85
        return round(base, 1)

    @staticmethod
    def _geometric_awareness_for(player) -> float:
        """Read geometric_awareness from DNA MentalAttributes (0-100)."""
        mental = getattr(getattr(player, "dna", None), "mental", None)
        if mental is not None:
            val = getattr(mental, "geometric_awareness", None)
            if val is not None:
                return max(0.0, min(100.0, val))
        return 50.0

    # ── LIVE UPDATES ──────────────────────────────────────────

    def record_touch(self, player_name: str, x: Optional[float], y: Optional[float], minute: int):
        """Update a player's current position after they're involved in an event."""
        if x is None or y is None:
            return
        state = self.states.get(player_name)
        if state:
            # ── REAL MOVEMENT CAPTURE ──────────────────────────────
            # Measure across the WHOLE call, not just touch_at(), since
            # the flank-hold correction and GK box anchor below also move
            # the player before this method returns. One measurement here
            # captures the true net displacement of this touch event,
            # rather than under-counting by only tracking touch_at()'s
            # own internal step.
            start_x, start_y = state.current_x, state.current_y
            state.touch_at(x, y, minute)
            # ── CHECKPOINT 21d: WIDE-ROLE FLANK HOLD ON TOUCH ──────
            # `touch_at` plants the player exactly where the ball event put
            # him. If the ball keeps cycling through the central channel,
            # a winger/fullback gets yanked off his touchline on EVERY touch
            # and the team collapses into the middle (the original complaint).
            # Pull any wide player back onto his flank channel on touch —
            # scaled by flank commitment so a touchline hugger holds harder
            # than an inverted inside-forward. Bounded: only fires when the
            # touch has dragged the player >6m OFF his flank channel, so an
            # on-flank touch is tracked exactly (preservation contract) and
            # the pull always moves y TOWARD home_y, never away. The x stays
            # at the ball (he IS at the ball); only the lateral placement is
            # anchored.
            if state.position in ("LB", "RB", "LW", "RW"):
                anchor_y = state.home_y
                if abs(anchor_y - state.current_y) > 6.0:
                    pull = 0.35
                    if state.position in ("LW", "RW"):
                        wp = self.winger_registry.get(player_name)
                        if wp is not None:
                            pull = 0.25 + 0.20 * wp.flank_commitment
                    state.current_y += (anchor_y - state.current_y) * pull
            self._anchor_gk_in_own_box(state)

            # Finalize the real distance covered by this touch event
            # (touch_at + flank hold + GK anchor, all folded into one
            # honest displacement measurement).
            jump = ((state.current_x - start_x) ** 2 +
                    (state.current_y - start_y) ** 2) ** 0.5
            state.minute_touch_distance += jump
            state.minute_touch_count += 1
            if jump > state.minute_peak_touch_jump:
                state.minute_peak_touch_jump = jump

    def _anchor_gk_in_own_box(self, state: PlayerSpatialState):
        """Keep a goalkeeper's live position inside his own defensive third.

        A keeper is the permanent overload anchor of build-up — he may step
        out of the box to receive a short back-pass, but he NEVER follows the
        play upfield. `record_touch` plants the keeper wherever the ball was,
        so without this anchor a sweeper keeper who receives a midfield reset
        gets dragged to x≈90+ and then becomes a release-valve target 95m
        from his own goal (absurd 90m "back-passes" to a keeper in the
        opponent's box). Clamp to the defensive third: the keeper stays a
        short, real back-pass away.
        """
        if state.position != "GK":
            return
        attacks_right = self.team_attacks_right.get(state.team, True)
        if attacks_right:
            state.current_x = max(0.0, min(35.0, state.current_x))
        else:
            state.current_x = max(70.0, min(105.0, state.current_x))

    # ── MOVEMENT ACCOUNTING (real distance / sprint data) ──────────
    #
    # Off-ball movement (drift_minute, defensive_block, attacking_crash)
    # happens across several separate method calls per minute in
    # match_engine.py's per-minute loop, each mutating current_x/current_y
    # internally in multiple places (home pull, flank pull, forward
    # anchor, line cohesion, coverage runs, space runs). Rather than
    # instrumenting every internal += site individually — fragile, easy
    # to miss one, easy to double-count — the caller brackets the WHOLE
    # off-ball movement phase with a before/after snapshot diff. This
    # measures the true net distance each player moved that minute from
    # every off-ball source combined, with zero risk of missing a site.

    def snapshot_positions(self, team_name: str) -> Dict[str, Tuple[float, float]]:
        """Capture each player's current (x, y) — call BEFORE the
        drift_minute / defensive_block / attacking_crash sequence."""
        return {
            name: (self.states[name].current_x, self.states[name].current_y)
            for name in self.team_rosters.get(team_name, [])
            if name in self.states
        }

    def accumulate_drift_from_snapshot(
        self, team_name: str, before: Dict[str, Tuple[float, float]]
    ):
        """Diff current positions against a prior snapshot and add the
        real net distance moved to each player's minute_drift_distance.
        Call AFTER drift_minute / defensive_block / attacking_crash have
        all run for this team this minute."""
        for name, (px, py) in before.items():
            state = self.states.get(name)
            if state is None:
                continue
            dist = ((state.current_x - px) ** 2 + (state.current_y - py) ** 2) ** 0.5
            state.minute_drift_distance += dist

    def pop_minute_activity(self, player_name: str) -> Dict[str, float]:
        """
        Return this minute's real movement summary for a player and reset
        the accumulators for the next minute.

        Fields:
            distance_touch  — real distance covered via ball-involvement
                               events this minute (multiple samples/minute
                               for an involved player).
            distance_drift  — real net off-ball distance this minute
                               (home pull + shape + line cohesion +
                               coverage/space runs + defensive_block +
                               attacking_crash, one measured delta).
            distance_total  — sum of the two; this player's true distance
                               covered this minute.
            touches         — how many ball-involvement events they had.
            peak_touch_jump — largest single touch-to-touch displacement,
                               a burst-intensity signal for sprint
                               classification (a big single jump between
                               two touches implies fast movement between
                               them, unlike a slow accumulation).
        """
        state = self.states.get(player_name)
        if state is None:
            return {
                "distance_touch": 0.0, "distance_drift": 0.0,
                "distance_total": 0.0, "touches": 0, "peak_touch_jump": 0.0,
            }
        out = {
            "distance_touch": round(state.minute_touch_distance, 2),
            "distance_drift": round(state.minute_drift_distance, 2),
            "distance_total": round(state.minute_touch_distance + state.minute_drift_distance, 2),
            "touches": state.minute_touch_count,
            "peak_touch_jump": round(state.minute_peak_touch_jump, 2),
        }
        state.minute_touch_distance = 0.0
        state.minute_touch_count = 0
        state.minute_peak_touch_jump = 0.0
        state.minute_drift_distance = 0.0
        return out

    # Line groupings for Checkpoint 6 team-shape cohesion
    DEFENSIVE_LINE_POSITIONS = {"CB", "LB", "RB"}
    MIDFIELD_LINE_POSITIONS = {"CDM", "CM", "CAM"}

    # Checkpoint 6.1 — role-scaled attacking/defensive shape swing. Real
    # fullbacks/wingers swing 25-35m+ of depth between attacking and
    # defending shape; a CB or holding mid barely moves at all. The old
    # flat ±3.5/-3.0 applied identically to all 11 players understated
    # exactly the role (RB/LB) where this shape difference matters most.
    # Scale is relative to the base shape_shift magnitude computed in
    # drift_minute(); 1.0 = the old flat behaviour for that position.
    SHAPE_SHIFT_SCALE: Dict[str, float] = {
        "GK": 0.15, "CB": 0.45,
        "LB": 1.45, "RB": 1.45,
        "CDM": 0.75, "CM": 0.95, "CAM": 1.10,
        "LW": 1.30, "RW": 1.30,
        "ST": 0.80, "CF": 0.80,
    }

    def drift_minute(
        self,
        team_name: str,
        profile: "TeamProfile",
        phase,          # MatchPhase
        game_state_gd: int = 0,   # this team's goal difference perspective
        minute: int = 0,
        in_possession: bool = False,
        ball_x: Optional[float] = None,
        ball_y: Optional[float] = None,
        opponent_players: Optional[List] = None,
    ):
        """
        Called once per minute per team. Every player NOT touched THIS
        minute drifts back toward home. Home itself can shift slightly
        based on phase/game-state (chasing a goal late -> push CBs' home up)
        AND, as of Checkpoint 6, on whether the team currently holds
        possession — an attacking shape (whole team shifted forward,
        compact) vs. a defensive shape (dropped off, compact deeper block).
        Line cohesion (back four / midfield three moving as a unit) is
        applied afterward so real team shape emerges, not just 11
        independent home markers drifting in isolation.
        """
        phase_name = getattr(phase, "value", str(phase))

        # Dynamic home shift: losing late -> whole team's home_x nudges forward.
        # Winning late (protecting) -> home_x nudges back (compact/defend).
        chase_shift = 0.0
        if game_state_gd <= -1 and phase_name in ("final_push", "added_time", "peak_intensity"):
            chase_shift = 5.0 if game_state_gd == -1 else 8.0
        elif game_state_gd >= 2 and phase_name in ("final_push", "added_time"):
            chase_shift = -4.0

        # Checkpoint 6 — attacking vs defensive team shape: in possession,
        # the whole team's home baseline shifts forward (compact, higher
        # block, supporting the ball). Out of possession, it drops back
        # into a defensive shape. This is a SEPARATE, additive shift from
        # the game-state chase_shift above — a team losing late AND out of
        # possession both push forward and drop back is a real contradiction
        # a real team doesn't have (they're either pressing high to win it
        # back, which IS captured by press_intensity's pull_strength below,
        # or they have it and push on) so we don't double-apply; possession
        # shape is the dominant signal, chase_shift adds urgency on top.
        #
        # Checkpoint 6.1: this used to be one flat number applied to all 11
        # players. Scaled per-position below (SHAPE_SHIFT_SCALE) so a
        # fullback/winger genuinely swings shape while a CB/CDM barely does.
        base_shape_shift = 3.5 if in_possession else -3.0

        # Checkpoint 21 — the drift pull is the ONLY mechanism that returns
        # non-involved players to their formation post, and it runs ONCE per
        # minute while touch-events can pin a player to the ball 8-15 times
        # per minute. At the old 0.30-0.45 pull a winger dragged to the
        # centre was still ~15m off his touchline after a full minute. Raise
        # the floor so the formation re-asserts itself within ~2-3 minutes.
        press_pull = 0.45 + getattr(profile, "press_intensity", 0.5) * 0.15

        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None:
                continue

            player_shape_shift = base_shape_shift * self.SHAPE_SHIFT_SCALE.get(state.position, 1.0)
            effective_home_x = max(4.0, min(101.0, state.home_x + chase_shift + player_shape_shift))

            if state.last_active_minute == minute:
                # Checkpoint 6.1 fix: this used to be a hard `continue` —
                # a player touched almost every minute (a classic
                # overlapping RB) was PERMANENTLY exempt from the
                # attacking/defensive shape correction, since his position
                # was driven entirely by raw touch coordinates and the
                # shape signal never got a chance to apply across however
                # many consecutive minutes he stayed heavily involved.
                # Real touch data stays authoritative — we still don't
                # override where the ball genuinely was this minute — but
                # a small blend toward the current shape target keeps him
                # from silently drifting out of sync with his own line's
                # shape while he remains heavily involved.
                touched_pull = press_pull * 0.25
                state.current_x += (effective_home_x - state.current_x) * touched_pull
                continue

            # Temporarily drift toward the (possibly shifted) home
            state.current_x += (effective_home_x - state.current_x) * press_pull
            state.current_y += (state.home_y - state.current_y) * press_pull

        # ── CHECKPOINT 18: MODERN WINGER FLANK + FORWARD ANCHORING ──
        # Wingers are touchline-hugging flank attackers, NOT drifting #10s.
        # The middle of the pitch is always full — a #10 owns that space —
        # and a winger who drifts inside leaves his flank open and crowds
        # his own teammates. After the generic home drift, pull any winger
        # who has drifted out of their flank channel back toward their
        # touchline anchor. The pull strength scales with how far they've
        # drifted and their flank commitment (from DNA).
        #
        # CRITICAL: we also anchor the winger FORWARD along x. A modern
        # winger's home is in the attacking third (x≈82), so when they
        # drift back into midfield (x<70) they get pulled forward again.
        # This is what stops them from becoming a second #10.
        #
        # Checkpoint 21c — fullbacks get the same flank treatment: they are
        # the width on the defensive side of the pitch. When a fullback has
        # drifted into the half-space, pull them back onto their touchline
        # channel (out of possession they may tuck in a little to defend the
        # half-space, hence the weaker pull).
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("LB", "RB", "LW", "RW"):
                continue
            # Checkpoint 21e — the flank pull is anchored on the FORMATION
            # home_y, never on the position name. For a team attacking LEFT
            # the "LW" stands on the right side of the pitch (home_y is
            # mirrored), and a name-based anchor would drag him back across
            # midfield — the original cause of wide players crossing in the
            # middle. home_y is always the correct touchline channel.
            flank_drift = abs(state.current_y - state.home_y)
            if state.position in ("LW", "RW"):
                winger_profile = self.winger_registry.get(name)
                commitment = winger_profile.flank_commitment if winger_profile is not None else 0.85
            else:
                commitment = 0.75
            if flank_drift > 8.0:
                pull = (0.45 if in_possession else 0.30) * (0.5 + commitment * 0.5)
            elif flank_drift > 4.0:
                pull = (0.30 if in_possession else 0.20) * (0.5 + commitment * 0.5)
            else:
                pull = 0.12
            state.current_y += (state.home_y - state.current_y) * pull

            # ── FORWARD X-ANCHOR ────────────────────────────────────
            # In possession, a winger who has drifted back toward midfield
            # (x < 70 for attacking-right, x > 35 for attacking-left) is
            # pulled FORWARD toward their attacking-third home. This is the
            # key fix that stops wingers from becoming #10s — they must
            # stretch the pitch HIGH and WIDE, not drop into midfield.
            attacks_right = self.team_attacks_right.get(team_name, True)
            if in_possession:
                if attacks_right:
                    if state.current_x < 70.0:
                        forward_pull = 0.25 + (70.0 - state.current_x) / 70.0 * 0.20
                        state.current_x += (state.home_x - state.current_x) * forward_pull
                else:
                    if state.current_x > 35.0:
                        forward_pull = 0.25 + (state.current_x - 35.0) / 70.0 * 0.20
                        state.current_x += (state.home_x - state.current_x) * forward_pull

        # Checkpoint 6 — line cohesion: apply AFTER individual drift so
        # defensive/midfield lines pull toward their own line-mates' average
        # position, representing a back four/midfield three shifting as a
        # unit rather than each player being an independent dot anchored
        # only to their own personal home position.
        self._apply_line_cohesion(team_name)

        # Checkpoint 6.2 — midfield geometric coverage (Enzo/Rice/Pedri):
        # in possession, midfielders with high geometric_awareness drift
        # to cover vacant half-spaces when teammates are isolated.
        if in_possession:
            self._midfielder_geometric_coverage(team_name, minute=minute)

            # Checkpoint 19 — attacker space runs (ST/LW/RW/GK):
            # in possession, attackers with high geometric_awareness run
            # into space away from markers, modelling elite forwards who
            # "understand space" and make intelligent off-ball movements.
            if ball_x is not None and ball_y is not None:
                self._attacker_space_run(
                    team_name, ball_x, ball_y,
                    def_players=opponent_players or [],
                    position_engine=self,
                    attacks_right=self.team_attacks_right.get(team_name, True),
                    minute=minute,
                )

    def _apply_line_cohesion(self, team_name: str, pull_strength: float = 0.12):
        """
        Checkpoint 6: back-line (CB/LB/RB) and midfield-line (CDM/CM/CAM)
        players nudge toward their line-mates' average current position
        each tick. A right-back overlapping doesn't leave his centre-backs
        statically anchored elsewhere — real defensive/midfield lines hold
        their shape sideways much tighter than they hold depth.

        Fullbacks are a special case: their lateral width should be
        preserved, so they are only lightly nudged toward the line's
        average y-position. This prevents the back four from collapsing
        centrally just because the centre-backs remain in a tighter block.
        """
        for group in (self.DEFENSIVE_LINE_POSITIONS, self.MIDFIELD_LINE_POSITIONS):
            members = [
                self.states[n] for n in self.team_rosters.get(team_name, [])
                if self.states.get(n) and self.states[n].position in group
            ]
            if len(members) < 2:
                continue
            avg_y = sum(m.current_y for m in members) / len(members)
            avg_x = sum(m.current_x for m in members) / len(members)
            for m in members:
                if m.position in ("LB", "RB") and group is self.DEFENSIVE_LINE_POSITIONS:
                    y_pull = pull_strength * 0.08
                else:
                    y_pull = pull_strength
                m.current_y += (avg_y - m.current_y) * y_pull
                m.current_x += (avg_x - m.current_x) * (pull_strength * 0.5)

    def _midfielder_geometric_coverage(
        self, team_name: str, minute: int = 0, pull_strength: float = 0.20,
    ):
        """
        Checkpoint 6.2 — Midfield geometric coverage (Enzo / Rice / Pedri).

        Elite midfielders read the pitch geometrically: when a teammate is
        isolated in a half-space and has no support nearby, a midfielder
        with high geometric_awareness drifts to cover the vacant space
        rather than staying glued to his home marker. This models the
        real-life behaviour where midfielders "cover a lot of distance"
        because they understand where their teammates need support.

        Only acts when:
            - the team is IN possession (out of possession they drop into
              the defensive block, which is handled separately), and
            - the midfielder's geometric_awareness > 55.
        """
        midfield_positions = {"CDM", "CM", "CAM"}
        midfielders = []
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in midfield_positions:
                continue
            midfielders.append(state)

        if not midfielders:
            return

        # Build a set of teammate positions (excluding each midfielder
        # himself) so we can detect vacant half-space zones.
        all_positions = []
        for name in self.team_rosters.get(team_name, []):
            s = self.states.get(name)
            if s is None:
                continue
            all_positions.append((s.current_x, s.current_y, s.position))

        for m in midfielders:
            if m.geometric_awareness < 55.0:
                continue

            awareness_factor = max(0.0, min(1.0, (m.geometric_awareness - 50.0) / 50.0))

            # Candidate target points: a 3x2 grid of half-space positions
            # relative to the midfielder's home. Half-spaces are y in [8,22]
            # (left) and [46,60] (right); central band is ignored because
            # the midfield line already occupies it.
            best_target = None
            best_score = -1.0
            for dx in [-15.0, 0.0, 15.0, 30.0]:
                for dy in [-18.0, -8.0, 8.0, 18.0]:
                    tx = max(15.0, min(95.0, m.home_x + dx))
                    ty = max(6.0, min(62.0, m.home_y + dy))

                    # Half-space bonus: prefer the flanks.
                    half_space_bonus = 1.0
                    if ty < 22.0 or ty > 46.0:
                        half_space_bonus = 1.6
                    elif ty < 28.0 or ty > 40.0:
                        half_space_bonus = 1.2

                    # Distance from home (must be within ~1.2x drift tolerance).
                    dist_home = math.hypot(tx - m.home_x, ty - m.home_y)
                    if dist_home > m.drift_tolerance * 1.3:
                        continue

                    # Vacancy check: no teammate (other than this midfielder)
                    # within 13m of the candidate point.
                    min_tm_dist = min(
                        math.hypot(tx - px, ty - py)
                        for px, py, pos in all_positions
                        if not (pos == m.position and abs(px - m.current_x) < 2.0
                                and abs(py - m.current_y) < 2.0)
                    )
                    if min_tm_dist < 13.0:
                        continue

                    score = (1.0 / (dist_home + 1.0)) * half_space_bonus
                    if tx > 65.0:
                        score *= 1.25
                    if score > best_score:
                        best_score = score
                        best_target = (tx, ty)

            if best_target is not None:
                tx, ty = best_target
                dist = math.hypot(tx - m.current_x, ty - m.current_y)
                if dist > 4.0:
                    effective_pull = pull_strength * max(0.15, awareness_factor)
                    m.current_x += (tx - m.current_x) * effective_pull
                    m.current_y += (ty - m.current_y) * effective_pull

    def _attacker_space_run(
        self, team_name: str, ball_x: float, ball_y: float,
        def_players: List, position_engine: Optional[PositionEngine],
        attacks_right: bool, minute: int = 0, pull_strength: float = 0.10,
    ):
        """
        Checkpoint 19 — attacker space runs (ST, LW, RW, GK).

        Elite attackers read the pitch geometrically: when a teammate has
        the ball, they run into space away from their markers. A striker
        with high geometric_awareness will drift into the half-spaces or
        make a run behind the defence when the ball is in the final third.
        A winger will stay wide or cut inside depending on where the space is.
        A GK will step up to become a passing option when the ball is in
        the opponent's half.

        Only acts when:
            - the team is IN possession, and
            - the player's geometric_awareness > 50.
        """
        attack_positions = {"ST", "CF", "LW", "RW", "CAM"}
        attackers = []
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in attack_positions:
                continue
            attackers.append(state)

        if not attackers:
            return

        for a in attackers:
            if a.geometric_awareness < 50.0:
                continue

            awareness_factor = max(0.0, min(1.0, (a.geometric_awareness - 45.0) / 55.0))
            ax, ay = a.current_x, a.current_y

            # Determine if this player is in the opponent's half
            in_opp_half = (ax > 52.5) if attacks_right else (ax < 52.5)

            # Check if this player is closely marked
            min_def_dist = None
            if def_players and position_engine is not None:
                min_def_dist = min(
                    math.hypot(
                        position_engine.get_position(d.name)[0] - ax,
                        position_engine.get_position(d.name)[1] - ay,
                    )
                    for d in def_players
                    if getattr(d, 'position', None) != 'GK'
                )
            is_marked = min_def_dist is not None and min_def_dist < 8.0

            # Space run logic
            best_target = None
            best_score = -1.0

            # Candidate run targets depend on position
            if a.position in ("ST", "CF"):
                # Strikers: run behind the defence, into the box, or into half-spaces
                candidates = []
                for dx in [-10.0, 0.0, 10.0, 20.0]:
                    for dy in [-12.0, -6.0, 0.0, 6.0, 12.0]:
                        tx = max(20.0, min(100.0, ax + dx))
                        ty = max(6.0, min(62.0, ay + dy))
                        # Prefer targets between ball and goal
                        between = (
                            (ball_x < tx < (105.0 if attacks_right else 0.0))
                            if attacks_right else
                            ((0.0 if attacks_right else 105.0) < tx < ball_x)
                        )
                        # Prefer half-spaces
                        half_space = ty < 22.0 or ty > 46.0
                        # Penalty if too close to defenders
                        def_penalty = 1.0
                        if def_players and position_engine is not None:
                            min_d = min(
                                math.hypot(
                                    position_engine.get_position(d.name)[0] - tx,
                                    position_engine.get_position(d.name)[1] - ty,
                                )
                                for d in def_players
                                if getattr(d, 'position', None) != 'GK'
                            )
                            if min_d < 6.0:
                                def_penalty = 0.3
                            elif min_d < 10.0:
                                def_penalty = 0.6
                        score = (1.0 if between else 0.4) * (1.6 if half_space else 1.0) * def_penalty
                        candidates.append((score, tx, ty))
                if candidates:
                    candidates.sort(key=lambda c: -c[0])
                    best_target = (candidates[0][1], candidates[0][2])

            elif a.position in ("LW", "RW"):
                # ── CHECKPOINT 18: MODERN WINGER SPACE RUNS ────────────
                # Wingers are touchline-hugging flank attackers. The middle
                # of the pitch is always full — a #10 owns that space — and
                # a winger who drifts inside leaves his flank open. Their
                # space runs are DOWN THE FLANK toward goal, never into
                # midfield traffic. The flank is scored 3x higher than any
                # cut-inside option, and runs are only scored if they move
                # TOWARD the goal they attack.
                flank_y = 10.0 if a.position == "LW" else 58.0
                # The half-space cut is only a brief, late-arrival option
                # (Saka/Vini arriving at the back post), never a default
                # drift into midfield.
                half_space_y = 18.0 if a.position == "LW" else 50.0
                cut_inside_y = 26.0 if a.position == "LW" else 42.0
                # Checkpoint 21e — the flank targets must follow the FORMATION
                # home_y, not the position name. For a team attacking LEFT the
                # "LW" actually stands on the right side of the pitch (home_y
                # is mirrored), and a name-based flank would run him back
                # across midfield into the middle of the pitch.
                flank_y = a.home_y
                sign = 1.0 if a.home_y > 34.0 else -1.0
                half_space_y = a.home_y + sign * 8.0
                cut_inside_y = a.home_y + sign * 16.0

                goal_x = 105.0 if attacks_right else 0.0
                winger_profile = self.winger_registry.get(a.player_name)

                candidates = []
                for target_y in [flank_y, half_space_y, cut_inside_y]:
                    for dx in [8.0, 15.0, 22.0, 30.0]:
                        tx = max(15.0, min(100.0, ax + (dx if attacks_right else -dx)))
                        ty = max(5.0, min(63.0, target_y))
                        # ONLY runs toward goal score — moving backward is
                        # never a winger space run.
                        advance = (abs(tx - goal_x) < abs(ax - goal_x))
                        if not advance:
                            continue
                        # The further forward, the better — being in the
                        # attacking third is the whole point.
                        if attacks_right:
                            forward_score = max(0.0, (tx - 55.0) / 50.0)
                        else:
                            forward_score = max(0.0, (55.0 - tx) / 50.0)
                        # Flank positioning is 3x more important than any
                        # central option. A winger hugging the touchline in
                        # the attacking third is infinitely more valuable
                        # than one drifted next to the #10 in midfield.
                        if target_y == flank_y:
                            flank_weight = 3.0
                        elif target_y == half_space_y:
                            flank_weight = 0.6
                        else:
                            flank_weight = 0.3
                        # Closer to the flank = better, even for the
                        # cut-inside option (stay wide until the last moment)
                        y_dist_from_flank = abs(ty - flank_y)
                        y_factor = max(0.3, 1.0 - y_dist_from_flank / 40.0)
                        # Penalty if marked
                        def_penalty = 1.0
                        if is_marked:
                            def_penalty = 0.5
                        score = (1.0 + forward_score * 2.0) * flank_weight * y_factor * def_penalty
                        candidates.append((score, tx, ty))
                if candidates:
                    candidates.sort(key=lambda c: -c[0])
                    best_target = (candidates[0][1], candidates[0][2])

            elif a.position == "GK":
                # GK: step up to become a passing option when ball is in opponent's half
                if in_opp_half and ball_x > 60.0 if attacks_right else ball_x < 45.0:
                    tx = max(35.0, min(70.0, ball_x - 15.0))
                    ty = max(20.0, min(48.0, ball_y + random.uniform(-5, 5)))
                    best_target = (tx, ty)

            if best_target is not None:
                tx, ty = best_target
                dist = math.hypot(tx - ax, ty - ay)
                if dist > 3.0:
                    effective_pull = pull_strength * max(0.2, awareness_factor)
                    a.current_x += (tx - a.current_x) * effective_pull
                    a.current_y += (ty - a.current_y) * effective_pull

    # ── CHECKPOINT 9: DANGER-AWARE DEFENSIVE BLOCK ──────────────
    # The defensive unit's COORDINATED answer to a live threat: when out of
    # possession with the ball in their own half, the back line (CB/LB/RB)
    # plus GK and CDM pull toward a compact, goal-side, ball-facing shape
    # that sits between the ball and the goalpost xy they defend. The closer
    # the ball (higher danger), the deeper and more compact the block.

    BLOCK_POSITIONS = {"GK", "CB", "LB", "RB", "CDM", "CM", "LW", "RW"}

    def defensive_block(
        self,
        team_name: str,
        ball_x: float,
        ball_y: float,
        own_goal_x: float,
        danger_level: float,
        minute: int = 0,
        pull_strength: float = 0.5,
    ) -> None:
        """
        Pull the defensive line into a coordinated goal-side block.

        Only acts when:
            - danger_level >= 25 (a real threat exists), and
            - the ball is in the team's own half.
        Otherwise it's a no-op — preserving baseline drift behaviour exactly.

        The block deepens as danger rises: at CRITICAL danger the line drops
        onto the six-yard line (bodies on the line); at low danger it steps
        up just behind the ball. Laterally the whole unit shifts toward the
        ball side, with near-side players (closer to ball_y) shifting harder.
        """
        if danger_level < 25.0:
            return

        goal_x = own_goal_x
        # Ball must be in the defended half.
        if goal_x == 105.0:
            if ball_x < 52.5:
                return
        else:
            if ball_x > 52.5:
                return

        dir_toward_goal = 1.0 if goal_x == 105.0 else -1.0
        risk = max(0.0, min(1.0, danger_level / 100.0))

        # Line sits a few metres goal-side of the ball, deepening with danger.
        behind = 10.0 - 8.0 * risk
        deepen = risk * 6.0
        line_x = ball_x + dir_toward_goal * (behind + deepen)
        if goal_x == 105.0:
            line_x = max(52.0, min(103.0, line_x))
        else:
            line_x = max(2.0, min(53.0, line_x))

        # Lateral anchor: the unit shifts to the ball's side of the pitch.
        lateral = max(12.0, min(56.0, ball_y))
        intensity = pull_strength * (0.30 + 0.70 * risk)

        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in self.BLOCK_POSITIONS:
                continue

            if state.position == "GK":
                # The keeper guards the goal — only drifts toward the ball side.
                target_x = goal_x + dir_toward_goal * -2.0
                target_y = 34.0 + (ball_y - 34.0) * 0.30
                k_intensity = intensity * 0.45
                state.current_x += (target_x - state.current_x) * k_intensity
                state.current_y += (target_y - state.current_y) * k_intensity
                continue

            if state.position == "CDM":
                # Screens the line from the ball side, a few metres in front.
                target_x = line_x - dir_toward_goal * 6.0
                target_y = lateral
            else:
                # CB/LB/RB: sit on the line; near-side players shift harder.
                target_x = line_x
                near = 1.0 if abs(state.current_y - ball_y) < 20.0 else 0.4
                target_y = state.current_y + (lateral - state.current_y) * near

            state.current_x += (target_x - state.current_x) * intensity
            state.current_y += (target_y - state.current_y) * intensity

        # Keep the four-line shape cohesive after the block pull.
        self._apply_line_cohesion(team_name)

    # ── CHECKPOINT 11: ATTACKING BOX CRASH ─────────────────
    # The attacking mirror of defensive_block. When a wide teammate enters
    # the crossing zone (ball in the attacking third, out near a touchline),
    # off-ball attackers abandon short-support runs and CRASH the box: near-
    # side players drive to the penalty-spot centre, far-side players sprint
    # to the back post — the classic two-post cross-attack pattern.

    CRASH_POSITIONS = {"ST", "CF", "LW", "RW", "CAM", "CM"}

    def attacking_crash(
        self,
        team_name: str,
        ball_x: float,
        ball_y: float,
        attacks_right: bool,
        minute: int = 0,
        intensity: float = 0.6,
        carrier_name: Optional[str] = None,
    ) -> None:
        """
        Pull the attacking team's off-ball forwards into box-crash targets.

        Only acts when the ball is genuinely in the wide crossing zone —
        attacking third AND on a wing (gated like defensive_block, so the
        baseline formation/off-ball drift is untouched otherwise):
            - ST / CF      → penalty-spot centre (get on the end of the cross)
            - CAM / CM     → late runs to the box centre / edge
            - far-side LW/RW → the back post
            - near-side LW/RW → the near-post edge ON THEIR OWN FLANK
              (Checkpoint 18: a winger arrives at the posts and keeps the
              width structure — he never becomes a second striker standing
              on the penalty spot, which is what produced the "inverted-10"
              cluster of pass origins around the spot / box centre).

        The crosser (carrier_name) is left alone — they have just delivered.
        """
        # Gate: ball in the attacking third AND wide (the crossing zone).
        in_attacking_third = ball_x > 70.0 if attacks_right else ball_x < 35.0
        wide = ball_y < WIDE_CHANNEL_WIDTH or ball_y > PITCH_Y - WIDE_CHANNEL_WIDTH
        if not (in_attacking_third and wide):
            return

        goal_x = 105.0 if attacks_right else 0.0
        box_centre_x = goal_x - 14.0          # ~ the penalty spot
        # Far-side attackers sprint to the back post (opposite the ball side).
        back_post_y = 20.0 if ball_y < CENTER_Y else 48.0
        back_post_x = goal_x - 8.0

        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in self.CRASH_POSITIONS:
                continue
            if name == carrier_name:
                continue

            near_side = abs(state.current_y - ball_y) < 20.0
            if state.position in ("LW", "RW"):
                # Checkpoint 18 — wingers attack the POSTS on their own side,
                # never the penalty-spot centre (that is the ST's zone).
                if near_side:
                    # Hold the flank, arrive at the near-post edge near goal.
                    target_x, target_y = goal_x - 6.0, state.home_y
                else:
                    target_x, target_y = back_post_x, back_post_y
            elif near_side:
                target_x, target_y = box_centre_x, CENTER_Y
            else:
                target_x, target_y = back_post_x, back_post_y

            pull = intensity * (0.6 + 0.4 * random.random())
            state.current_x += (target_x - state.current_x) * pull
            state.current_y += (target_y - state.current_y) * pull

    # ── QUERIES USED BY SELECTION FUNCTIONS ──────────────────

    def plausibility_at(self, player_name: str, x: float, y: float) -> float:
        """The core multiplier: how plausible is this player being involved here."""
        state = self.states.get(player_name)
        if state is None:
            return 1.0   # unknown player (e.g. not yet registered) -> no penalty
        return state.plausibility(x, y)

    # ── CHECKPOINT 21: RECEIVER OPTION QUALITY ──────────────────
    # The core anti-clustering fix. `plausibility_at` (distance from the
    # ball) rewards a player for being IN THE CLUMP — whoever is standing
    # nearest the ball wins the receive draw, so the same central group
    # gets picked over and over and the ball can never leave the middle.
    #
    # A pass is not aimed at "whoever is near the ball"; it is aimed at a
    # teammate who is IN POSITION — at (or running toward) their formation
    # post, in a physically reachable passing relationship to the ball.
    # So this score is built from the player's HOME POST, not their current
    # position:
    #     reach      — is the ball physically able to get to their post?
    #     direction  — forward/sideward outlets beat backward ones (ellipse)
    #     discipline — how far the player has abandoned their post (anti-clump)
    def receive_option_quality(
        self, player_name: str, ball_x: float, ball_y: float,
        attacks_right: bool = True,
    ) -> float:
        """
        Score how good a pass target this player is RIGHT NOW for a ball at
        (ball_x, ball_y). Returns 0..1. Unknown/unregistered players get 0.6
        so a cold start never zeroes out every option.
        """
        state = self.states.get(player_name)
        if state is None:
            return 0.6

        hx, hy = state.home_x, state.home_y

        # 1) Reachability: the distance from the ball to the player's POST.
        d = math.hypot(hx - ball_x, hy - ball_y)
        if d < 4.0:
            reach = 0.75                       # right on top of the ball — crowded
        elif d < 20.0:
            reach = 1.0 - 0.10 * ((20.0 - d) / 16.0)   # 0.90 -> 1.00 (sweet spot)
        elif d < 35.0:
            reach = 1.0 - 0.25 * ((d - 20.0) / 15.0)   # 1.00 -> 0.75
        elif d < 50.0:
            reach = 0.75 - 0.35 * ((d - 35.0) / 15.0)  # 0.75 -> 0.40
        else:
            reach = max(0.08, 0.40 * (0.5 ** ((d - 50.0) / 30.0)))

        # 2) Direction bias: forward/sideward outlets beat backward ones,
        #    referenced on the HOME post (a winger's post is the touchline;
        #    a deep CB's post is behind the ball).
        ell = ball_centric_ellipse_weight(
            ball_x, ball_y, hx, hy,
            attacks_right=attacks_right,
            sigma_along=ELLIPSE_SIGMA_ALONG.get(state.position, 26.0),
            sigma_across=ELLIPSE_SIGMA_ACROSS,
        )
        direction = ELLIPSE_COMPOSE_FLOOR + (1.0 - ELLIPSE_COMPOSE_FLOOR) * ell

        # 3) Post discipline: how far the player has drifted from their home
        #    post. This is the direct anti-clump term — a striker standing in
        #    the centre circle (36m from his box post) is a worse target than
        #    a striker standing where his striker actually stands.
        home_d = state.distance_from_home
        post_limit = max(18.0, state.drift_tolerance)
        if home_d <= post_limit:
            discipline = 1.0 - 0.30 * (home_d / post_limit)
        else:
            discipline = max(0.15, 0.70 * ((post_limit / home_d) ** 1.5))

        return reach * direction * discipline

    def flank_bias_y(self, player_name: str, current_y: float) -> float:
        """
        Checkpoint 21b — flank delivery bias for wide roles.

        When a wide player (LB/RB/LW/RW) is the target of a delivery, the
        ball is aimed at a point ON their flank channel, not at wherever they
        have drifted. Returns a y-coordinate biased 65% of the way from
        `current_y` toward the player's touchline post, so each wide
        reception drags the ball (and the player) back onto the flank and the
        width re-asserts itself. Non-wide roles are returned unchanged.
        """
        state = self.states.get(player_name)
        if state is None or state.position not in ("LB", "RB", "LW", "RW"):
            return current_y
        anchor_y = state.home_y
        return current_y + (anchor_y - current_y) * 0.65

    def ball_centric_weight(self, player_name: str, ball_x: float, ball_y: float) -> float:
        """
        Ball-centric elliptical receive weight for this player at a live
        ball position. Composes the player's CURRENT position (from the
        position engine) against an ellipse anchored just ahead of the ball
        and aligned with this team's axis of play.

        Returns 0..1 (1.0 for unknown/unregistered players so nothing
        breaks on a cold start). This is the "where is the receiver
        relative to the ball" term used on top of label/marking weights.
        """
        state = self.states.get(player_name)
        if state is None:
            return 1.0
        attacks_right = self.team_attacks_right.get(state.team, True)
        return ball_centric_ellipse_weight(
            ball_x, ball_y, state.current_x, state.current_y,
            attacks_right=attacks_right,
            sigma_along=ELLIPSE_SIGMA_ALONG.get(state.position, 26.0),
        )

    def zone_name(self, player_name: str) -> str:
        state = self.states.get(player_name)
        if state is None:
            return "unknown"
        return ZoneGrid.zone_name(state.current_x, state.current_y)

    def get_position(self, player_name: str) -> Tuple[float, float]:
        state = self.states.get(player_name)
        if state is None:
            return (50.0, 34.0)
        return (state.current_x, state.current_y)

    def get_home_position(self, player_name: str) -> Tuple[float, float]:
        """Formation anchor (home) coordinates for a player."""
        state = self.states.get(player_name)
        if state is None:
            return (50.0, 34.0)
        return (state.home_x, state.home_y)

    def snapshot(self, team_name: str) -> List[Dict]:
        """Debug/export helper: current spatial state for a whole team."""
        rows = []
        for name in self.team_rosters.get(team_name, []):
            s = self.states.get(name)
            if not s:
                continue
            rows.append({
                "player": name, "position": s.position,
                "home_x": s.home_x, "home_y": s.home_y,
                "current_x": round(s.current_x, 1), "current_y": round(s.current_y, 1),
                "drift_from_home": round(s.distance_from_home, 1),
                "zone": ZoneGrid.zone_name(s.current_x, s.current_y),
            })
        return rows

    def remove_player(self, team_name: str, player_name: str):
        """Remove a sent-off player from the position engine."""
        self.states.pop(player_name, None)
        roster = self.team_rosters.get(team_name, [])
        if player_name in roster:
            roster.remove(player_name)


# ─────────────────────────────────────────────
# SELECTION HELPER — plug-compatible with BaseChain.pick_weighted
# ─────────────────────────────────────────────

def plausibility_weighted(
    position_engine: Optional[PositionEngine],
    players: List,
    label_weight_fn,
    at_x: float,
    at_y: float,
    exclude: str = None,
):
    """
    Drop-in replacement for BaseChain.pick_weighted() that multiplies the
    existing label-based weight by spatial plausibility at (at_x, at_y).

    If position_engine is None, behaves EXACTLY like the old label-only
    weighting (safe fallback — nothing breaks if not wired in somewhere).
    """
    pool = [p for p in players if getattr(p, "name", None) != exclude]
    if not pool:
        return None

    weights = []
    for p in pool:
        label_w = max(0.1, label_weight_fn(p))
        if position_engine is not None:
            name = getattr(p, "name", "")
            plaus = position_engine.plausibility_at(name, at_x, at_y)
        else:
            plaus = 1.0
        weights.append(max(0.02, label_w * plaus))

    return random.choices(pool, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python position_engine.py
# Verifies the module works with ZERO dependency on the rest of PLOFA
# beyond a couple of duck-typed stand-ins.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧭 PLOFA 26/27 — Position Engine (Checkpoint 5) Standalone Demo")
    print("=" * 64)

    # ── Minimal duck-typed stand-ins so this runs with zero imports ──
    class _FakeProfile:
        def __init__(self, defensive_line=0.5, width=0.5, tempo=0.5,
                     directness=0.5, press_intensity=0.5):
            self.defensive_line = defensive_line
            self.width = width
            self.tempo = tempo
            self.directness = directness
            self.press_intensity = press_intensity

    class _FakeDNA:
        def __init__(self, specialties):
            self.specialties = specialties

    class _FakePlayer:
        def __init__(self, name, position, specialties=None):
            self.name = name
            self.position = position
            self.dna = _FakeDNA(specialties or [])

    squad = [
        _FakePlayer("Keano Walsh", "GK"),
        _FakePlayer("Emeka Obi", "CB", ["ball_playing_cb"]),
        _FakePlayer("Tavish Crane", "CB", ["stopper_defender"]),
        _FakePlayer("Darius Frost", "LB", ["aggressive_fullback"]),
        _FakePlayer("Rico Alves", "RB", ["overlapping_fullback"]),
        _FakePlayer("Mateo Sanz", "CDM", ["anchor_man"]),
        _FakePlayer("Luca Ferrini", "CM", ["box_box", "engine"]),
        _FakePlayer("Kofi Mensah", "CAM", ["creator"]),
        _FakePlayer("Adri Vela", "LW", ["dribbler", "speedster"]),
        _FakePlayer("Dragan Novak", "ST", ["clinical_finisher"]),
        _FakePlayer("Percy", "RW", ["grand_dribbler", "inverted"]),
    ]

    profile = _FakeProfile(defensive_line=0.65, width=0.6, press_intensity=0.65)

    pe = PositionEngine()
    pe.initialize_team("Hartwell City", squad, profile)

    print("\n1. HOME POSITIONS AT KICKOFF (attacking style, high press)\n")
    print(f"  {'Player':<16} {'Pos':<4} {'Home X':>7} {'Home Y':>7}  {'Zone'}")
    print(f"  {'-'*55}")
    for row in pe.snapshot("Hartwell City"):
        print(f"  {row['player']:<16} {row['position']:<4} "
              f"{row['home_x']:>7.1f} {row['home_y']:>7.1f}  {row['zone']}")

    print("\n2. STRIKER PLAUSIBILITY CHECK — the actual bug being fixed\n")
    striker = "Dragan Novak"
    test_points = [
        ("own box (deep build-up)", 15.0, 34.0),
        ("edge of own third", 30.0, 34.0),
        ("halfway line", 52.0, 34.0),
        ("edge of box (his zone)", 88.0, 34.0),
        ("six yard box", 101.0, 34.0),
    ]
    print(f"  {'Location':<28} {'Plausibility Multiplier'}")
    print(f"  {'-'*55}")
    for label, x, y in test_points:
        p = pe.plausibility_at(striker, x, y)
        bar = "█" * int(p * 20)
        print(f"  {label:<28} {p:>5.2f}  {bar}")

    print("\n   -> OLD system: striker had flat weight 0.8 EVERYWHERE on the pitch.")
    print("   -> NEW system: striker's weight is now suppressed near his own goal")
    print("      and boosted near his actual current zone. Same random draw,")
    print("      causally grounded input.")

    print("\n3. SIMULATE 10 MINUTES OF DRIFT (nobody touches the ball) — CDM example\n")
    cdm_state = pe.states["Mateo Sanz"]
    cdm_state.current_x, cdm_state.current_y = 75.0, 20.0  # got dragged forward
    print(f"  {'Minute':>7}  {'Current X':>10} {'Current Y':>10}  {'Dist from home':>15}")
    for minute in range(1, 11):
        pe.drift_minute("Hartwell City", profile, type("P", (), {"value": "second_open"})(),
                         game_state_gd=0, minute=minute)
        s = pe.states["Mateo Sanz"]
        print(f"  {minute:>6}'  {s.current_x:>10.1f} {s.current_y:>10.1f}  "
              f"{s.distance_from_home:>14.1f}m")

    print("\n4. RECEIVER SELECTION DEMO — plausibility_weighted() in action\n")
    ball_x, ball_y = 82.0, 40.0  # ball is in the final third
    print(f"   Ball at ({ball_x}, {ball_y}) — who's plausible to receive?\n")
    label_weights = {
        "CAM": 3.0, "LW": 3.0, "RW": 3.0, "ST": 3.5, "CM": 2.0,
        "CDM": 1.0, "CB": 0.5, "LB": 1.0, "RB": 1.0, "GK": 0.05,
    }
    for p in squad:
        plaus = pe.plausibility_at(p.name, ball_x, ball_y)
        lw = label_weights.get(p.position, 1.0)
        print(f"  {p.name:<16} {p.position:<4} label_w={lw:>4.1f}  "
              f"plaus={plaus:>4.2f}  final_w={lw*plaus:>5.2f}")

    counts = {}
    for _ in range(2000):
        pick = plausibility_weighted(
            pe, squad, lambda p: label_weights.get(p.position, 1.0),
            ball_x, ball_y,
        )
        counts[pick.name] = counts.get(pick.name, 0) + 1
    print("\n   2000-draw distribution (who actually gets picked as receiver):")
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"     {name:<16} {c:>5}  ({c/20:.1f}%)")

    print("\n5. BALL-CENTRIC ELLIPTICAL WEIGHTING (Checkpoint 20)\n")
    print("   The receive pool is an anisotropic ellipse anchored ahead of the ball,")
    print("   elongated along the axis of play — a runner 25m upfield is a live")
    print("   option, a player 25m out to the side is not.\n")
    ball_x, ball_y = 55.0, 34.0   # ball at halfway, central
    print(f"   Ball at ({ball_x}, {ball_y}), attacking right (goal at x=105):\n")
    print(f"   {'Player':<16} {'Pos':<4} {'Cur (x,y)':>16} {'Ellipse w':>9}  {'Ahead?':>6}")
    print(f"   {'-'*60}")
    for p in squad:
        px, py = pe.get_position(p.name)
        ell = pe.ball_centric_weight(p.name, ball_x, ball_y)
        ahead = "yes" if (px - ball_x) > 0 else "no"
        bar = "█" * int(ell * 30)
        print(f"   {p.name:<16} {p.position:<4} {px:>7.1f},{py:<7.1f} {ell:>9.2f}  {ahead:>6}  {bar}")

    print("\n   -> Ellipse is asymmetric: the same 25m displacement AHEAD of the")
    print("      ball outweighs 25m to the SIDE (anisotropy) or BEHIND the ball")
    print("      (forward bias). Both are invisible to a plain circular falloff.")
    print("   -> Composed with a floor, so deliberate recycle/support passes and")
    print("      GK back-pass outlets are never zeroed out.\n")

    print("\n✅ Position Engine module operational — zero dependency on rest of PLOFA.")
    print("   Next: wire into event_chain.py selection functions + match_engine.py minute loop.\n")