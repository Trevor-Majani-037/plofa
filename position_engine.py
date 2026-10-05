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
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

from cross_detector import WIDE_CHANNEL_WIDTH, PITCH_X, PITCH_Y, CENTER_Y
from tactical_shapes import (
    FormationStance,
    stance_delta_roles,
)
from attack_patterns import (
    AttackPattern,
    pattern_role_deltas,
)
from winger_behavior import (
    WingerRegistry,
    WingerBehaviorEngine,
    LEFT_TOUCHLINE_ANCHOR_Y,
    RIGHT_TOUCHLINE_ANCHOR_Y,
)
from fullback_behavior import FullbackRegistry, FullbackBehaviorEngine
from midfielder_behavior import MidfieldRegistry, MidfielderBehaviorEngine
from striker_behavior import StrikerRegistry, StrikerBehaviorEngine
from block_awareness import HalfSpaceMagnet
from marking import MarkingEngine, MarkAssignment

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
# PITCH WIDTH FIX: Increased from 9.0 to 13.0 to allow wider lateral drift
# for receiving, improving touchline usage and reducing center concentration.
ELLIPSE_SIGMA_ACROSS: float = 13.0
# VERTICALITY RE-BALANCE: forward_shift 8 -> 5m. The ellipse is still
# anchored AHEAD of the ball (so through-ball geometry survives and the
# P2 preservation guard holds), but a runner must sit closer to the ball
# line to earn the same receive weight — lateral and backward support
# outlets are no longer doubly-taxed by a long forward anchor.
ELLIPSE_FORWARD_SHIFT: float = 5.0
ELLIPSE_COMPOSE_FLOOR: float = 0.35

# Checkpoint 34 — WIDE FLANK OUTLET.
# A wide player standing within WIDE_OUTLET_CHANNEL_HALF metres of his flank
# anchor (home_y) right now is treated as a standing outlet in
# receive_option_quality() when the ball is central. Without this, the
# ellipse's lateral taper taxes a touchline target ~2.4x (direction ≈ 0.4)
# vs a central option (≈ 0.98), starving the wide channels at the SELECTION
# stage — the ball rarely gets switched wide because wide receivers score
# poorly the moment the ball is central. Lifting the direction floor for a
# genuinely-in-channel wide outlet lets the pass selection reach the touchline
# from a central ball (the delivery code already aims at the touchline band).
WIDE_OUTLET_CHANNEL_HALF: float = 10.0
WIDE_OUTLET_DIRECTION_FLOOR: float = 0.80

# Checkpoint 35 — PITCH-STRETCH RULE.
# Wide roles (LW/RW/LB/RB) are the team's WIDTH PROVIDERS. It is a
# structural duty — a RULE, not a preference — that they stretch the pitch
# when the middle is packed: the winger holds/pushes toward his touchline
# channel so the opposition block has to cover the full 68m, opening the
# central lanes the ball then exploits. The detector is the "spine weight"
# of the live ball: the deeper the ball sits in the central band (18-50m,
# peak 24-44m), the more convincingly the middle is crowded and the harder
# wide players pin the line. Every stretch movement clamps y INSIDE the
# pitch bounds (a stretch never leaves the field of play).
STRETCH_CENTER_LOW: float = 18.0
STRETCH_CENTER_HIGH: float = 50.0
STRETCH_SPINE_PEAK_LO: float = 24.0
STRETCH_SPINE_PEAK_HI: float = 44.0
# Strength additions, scaled by the spine weight (0..1):
STRETCH_DRIFF_BOOST: float = 0.5    # flank-anchor pull in drift_minute
STRETCH_TOUCH_BOOST: float = 0.30   # flank-hold pull in record_touch
STRETCH_TARGET_BOOST: float = 0.30  # off-ball target steer (match_engine)
#
# 2026-10-05 -- RAISED TO 0.55 AND 0.75 AS AN EXPERIMENT, THEN REVERTED.
#
# Measurement (`_diag_wide_target.py`, seed 777, ~140k in-possession samples
# per setting, same live loop):
#
#   boost   in-possession actual distance to the touchline (m)
#    0.30                        15.4
#    0.55                        14.6
#    0.75                        12.1   <- inside the real 3-12 m band
#
# So the knob works and the diagnosis under it is sound. It was reverted
# anyway, because it does not do the thing it was raised to do:
#
#   1. IT DOES NOT FIX THE COLLAPSE. Team width p05 stayed at 4.2-6.6 m and
#      p10 at 16-18 m across all three settings. The user reported a shape
#      that "sometimes collapses into a narrow block", and the collapse is
#      unchanged -- it is a tail event, and a rule that raises the AVERAGE
#      cannot touch a tail.
#      >>> 2026-10-05 CORRECTION: REASONS 1 AND 2 ARE VOID, BOTH NUMBERS. They
#      >>> come from the mispaired separation column of _diag_wide_target.py.
#      >>> _offball_tick_seq is a plain int ATTRIBUTE (match_engine.py:1990),
#      >>> not a method; the probe called it, TypeError was swallowed by the
#      >>> probe's own `except Exception: return -1`, and every wide player
#      >>> collapsed into ONE tick group. The touchline-distance column above
#      >>> is per-player and needs no pairing, so IT STANDS; only the side-
#      >>> to-side separation figures do not.
#      >>> The REASONING survives -- a rule that raises the mean cannot move a
#      >>> tail -- and is now independently supported by _diag_width_gate.py
#      >>> (p90 = 36.5 m: the 45-55 m band is geometrically unreachable, and
#      >>> lifting the median REDUCES the share inside the band). But the
#      >>> measurement offered for it here was not evidence. Do not quote it.
#   2. IT DOES NOT REACH 45-55 m. Median side-to-side separation moved 36.2
#      -> 38.9 m, nowhere near the band.   [VOID -- see correction above.
#      Trustworthy equivalent is the target-side median 30.6 m.]
#   3. IT COSTS REAL FOOTBALL. CK35 is applied AFTER the live run targets
#      (match_engine.py:2835-2843 then :2859), so a higher weight directly
#      undoes the winger's "cut inside" / box-entry run -- measured firing on
#      18% of samples at 24 m off the line. That run layer was wired
#      deliberately (2026-09-29) and is the thing that makes wingers look
#      like wingers.
#
# Trading a deliberate cut for 3 m of average width, while leaving both
# reported symptoms untouched, is a bad trade. The lever is right; the
# magnitude is not the problem. Record kept so the next attempt starts from
# 140k samples of evidence instead of a guess.

# Selection floor lift on top of WIDE_OUTLET_DIRECTION_FLOOR at full spine.
STRETCH_FLOOR_LIFT: float = 0.12
# Hard lateral clamp for wide roles under the rule — inside the pitch,
# a few metres off the ad boards. LB/LW anchor y=6, RB/RW y=62.
STRETCH_CLAMP_LOW: float = 3.0
STRETCH_CLAMP_HIGH: float = 65.0

# Checkpoint 36 — TRIANGLE SUPPORT RULE.
# Midfielders (CDM/CM/CAM) are the team's SHORT-PASSING LINKS: when the team
# holds the ball they must be ABLE to complete a passing triangle — the
# classic third-man shape — both on the wings (near-side CM joining the
# winger + full-back cluster) and in the centre (CM pair split around the
# ball with the CDM pivot behind). Before this checkpoint the live 10 Hz
# machine only compacted midfields 7% toward the ball's y, so the near-side
# CM physically could never reach a wide-support socket. The steer below is
# another TARGET steer (never a position jump), pace-preserving, bounded to
# half-space depth so a CM never pins the touchline like a winger does.
TRI_MIDFIELD_ROLES: tuple = ("CDM", "CM", "CAM")
TRI_WIDE_BAND: float = 18.0         # |ball_y - 34| above this = wide cluster
TRI_HALF_SPACE_MAX: float = 17.0    # near-side support lands <= 17m off centre
TRI_NEAR_ALPHA: float = 0.55        # near-side CM commits to the wing triangle
TRI_FAR_ALPHA: float = 0.15         # far-side CM balance shift (subtle)
TRI_PIVOT_ALPHA: float = 0.25       # CDM slides toward the ball side on wide play
TRI_CENTRAL_ALPHA: float = 0.22     # central-ball triangle spread strength
TRI_SUPPORT_BACK: float = 0.12      # support node sits 12% behind the play
TRI_CENTRAL_CM_OFFSET: float = 8.0  # CM pair split metres around the ball

# Checkpoint 37 — BACK-LINE BUILD-UP DROP (the "third-man" drop-in).
# In-possession between own goal and midfield, the ball-side CB sags toward
# the ball to offer the SHORT back option (a real team's build-out CB steps
# off the line to give the CM a 8-15m receive — otherwise the only reset is
# the 25-40m heave to the keeper). The socket is a TRI-style target steer:
# pace-capped movement, gated to possession + own/middle third, never into
# the six-yard wall.
BACKLINE_DROP_MAX_NX: float = 50.0   # drop engages only inside ~the halfway mark
BACKLINE_DROP_PEAK_NX: float = 30.0  # full strength through the own third
BACKLINE_DROP_MIN_NX: float = 18.0   # socket floor so a CB never hugs the goal line
BACKLINE_DROP_DEPTH: float = 0.30    # sag ~30% of the way back from ball to goal
BACKLINE_DROP_ALPHA: float = 0.55    # steer weight when the drop is fully engaged
BACKLINE_DROP_CENTER: float = 0.30   # socket pulls slightly toward the spine


def pitch_spine_weight(y: float) -> float:
    """Checkpoint 35 — how central/crowded a ball y-coordinate is: 0.0 on
    either touchline, 1.0 inside the 24-44m spine, linear taper through the
    outer 18m/50m edges of the central band. The wide-stretch duty grows
    with this value (0 = ball already wide, no need to stretch)."""
    if y < STRETCH_CENTER_LOW or y > STRETCH_CENTER_HIGH:
        return 0.0
    if y <= STRETCH_SPINE_PEAK_LO:
        return (y - STRETCH_CENTER_LOW) / (STRETCH_SPINE_PEAK_LO - STRETCH_CENTER_LOW)
    if y >= STRETCH_SPINE_PEAK_HI:
        return (STRETCH_CENTER_HIGH - y) / (STRETCH_CENTER_HIGH - STRETCH_SPINE_PEAK_HI)
    return 1.0


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
    # TOUCHLINE FIX: moved from y=10/58 to y=6/62 — 4m closer to each touchline.
    # Fullbacks need to be tight to the line in both build-up and overlap runs.
    "LB":  (26.0, 6.0),
    "RB":  (26.0, 62.0),
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
    # TOUCHLINE FIX: moved from y=10/58 to y=6/62 — 4m closer to each touchline.
    # Real EPL wingers operate 2-7m off the line. Old 10/58 kept them in the
    # wide-midfielder corridor. Everything that used these as anchors (carry
    # steering, winger_behavior, band-proximity) now naturally pulls them wider.
    "LW":  (82.0, 6.0),
    "RW":  (82.0, 62.0),
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
        # TOUCHLINE FIX: Old formula was base_y * (0.6 + width * 0.8) which
        # is INVERTED — at width=1.0 it gave LW home_y=14 (14m from touchline).
        # New formula: width REDUCES distance from touchline. At width=0.5
        # (default) player sits at base_y. At width=1.0 they're 3m closer to
        # touchline; at width=0.0 they're 3m further in. Keeps shape coherent.
        width = getattr(profile, "width", 0.5)
        y = base_y
        if position in ("LW", "LB"):
            # Low y = touchline side. Width pulls TOWARD y=0, not away.
            inward_offset = (0.5 - width) * 6.0   # +3m at width=0, -3m at width=1
            y = base_y + inward_offset
        elif position in ("RW", "RB"):
            # High y = touchline side. Width pulls TOWARD y=68, not away.
            inward_offset = (width - 0.5) * 6.0   # -3m at width=0, +3m at width=1
            y = base_y + inward_offset

        # Spread multiple same-position players (e.g. 2 CBs, 2 CMs)
        spread = POSITION_SPREAD_Y.get(position)
        if spread:
            offset = spread[slot_index % len(spread)]
            y = y + offset

        # TOUCHLINE FIX: widened from max(3.0, min(65.0)) to max(2.0, min(66.0))
        # so wide players can actually be homed near the touchline (y=2-6, y=62-66).
        y = max(2.0, min(66.0, y))
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

    # Movement vector (m/min), EMA-smoothed each minute from net drift deltas.
    # Feeds velocity-aware pitch-control influence (Checkpoint 26 wiring).
    velocity_x: float = 0.0
    velocity_y: float = 0.0

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

    # ── PHYSICS-DERIVED MOVEMENT (from PossessionEpisode trace) ──────
    # These are populated by record_physics_distance() and reset each minute.
    physics_distance_m: float = 0.0
    physics_walk_time_s: float = 0.0
    physics_jog_time_s: float = 0.0
    physics_sprint_time_s: float = 0.0
    physics_sprint_count: float = 0.0
    physics_high_speed_sprint_count: float = 0.0
    physics_top_speed_mps: float = 0.0

    # ── WIDE-RUN CONTINUITY (Checkpoint 32b) ──────────────────────
    # A cached run target the player travels toward at a pace-capped rate
    # each minute, so wide runs flow as trajectories instead of snapping.
    # run_mode is one of: None | "byline" | "cut" | "box" | "overlap" |
    # "underlap" | "tuck". top_speed_mpm bounds the per-minute step (m).
    run_target_x: float = 0.0
    run_target_y: float = 0.0
    run_mode: Optional[str] = None
    top_speed_mpm: float = 9.0

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


class _ShapeShim:
    """Minimal duck-type for the live off-ball loop.

    The behaviour engines (striker/winger/midfielder) were written against
    PlayerProfile objects and read only ``.name`` and ``.position``. The 10 Hz
    loop has spatial states instead, so it hands them this rather than either
    growing a real player reference into a hot function or teaching the
    engines about PlayerEngine. Same name, same slots, nothing else.
    """

    __slots__ = ("name", "position")

    def __init__(self, name: str, position: str) -> None:
        self.name = name
        self.position = position

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_ShapeShim({self.name!r}, {self.position!r})"


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
        # Checkpoint 32 — modern fullback registry: per-fullback spatial
        # profiles (flank commitment, advance/overlap/underlap/tuck instincts,
        # hold discipline, duel aggression, recovery urgency) built from the
        # fullback DNA archetypes in player_dna.
        self.fullback_registry: FullbackRegistry = FullbackRegistry()
        # Checkpoint 33 — midfielder (CM/CAM) and striker (ST/CF) registries:
        # per-player spatial profiles built from the DNA archetypes in
        # player_dna (box_to_box / deep_playmaker / progressive_midfielder /
        # classic_ten / shadow_striker, and poacher / target_man /
        # complete_striker / deep_lying_striker / speedster_striker).
        self.midfield_registry: MidfieldRegistry = MidfieldRegistry()
        self.striker_registry: StrikerRegistry = StrikerRegistry()
        # Checkpoint 29 — live opponent block shapes, refreshed per minute by
        # MatchEngine after _update_block_shapes(). Consumed by drift_minute's
        # HalfSpaceMagnet integration (in-possession half-space occupation).
        self._block_context: Tuple[Optional[object], Optional[object]] = (None, None)
        # Feature #1/#2 — in-match shape state. Base homes are captured at
        # kickoff; formation stances (score reaction) and attack patterns
        # (identity overloads) are applied as ADDITIVE role deltas on top, so
        # the XI actually takes the chasing/overload/box shape instead of just
        # switching dials. Applied from MatchEngine once per minute.
        self._formation_base_homes: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._stance_deltas: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._pattern_deltas: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._coach_deltas: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._applied_stance_key: Dict[str, tuple] = {}
        self._applied_pattern: Dict[str, Optional[AttackPattern]] = {}
        self._applied_coach_width: Dict[str, float] = {}
        # Pending set-piece slots; see open_setpiece_window.
        self._setpiece_targets: Dict[str, Tuple[float, float]] = {}
        self._setpiece_remaining_s: float = 0.0

    def set_block_context(self, home_block, away_block) -> None:
        """Refresh the per-minute block shapes used by drift_minute's magnet."""
        self._block_context = (home_block, away_block)

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
                top_speed_mpm=self._pace_mpm_for(p),
            )
            if name not in self.team_rosters[team_name]:
                self.team_rosters[team_name].append(name)

        # Feature #1/#2 — remember the authored shape as the STABLE base for
        # additive in-match stance / attack-pattern home deltas.
        self._capture_base_homes(team_name)
        self._applied_stance_key.pop(team_name, None)
        self._applied_pattern[team_name] = AttackPattern.NONE
        self._coach_deltas[team_name] = {}
        self._applied_coach_width[team_name] = 0.0

        # Checkpoint 18 — register all wingers' spatial profiles (touchline
        # anchor, flank commitment, byline instinct, isolation thirst).
        self.winger_registry.register_team(players)
        # Checkpoint 32 — register all fullbacks' spatial profiles
        # (flank commitment, advance/overlap/underlap/tuck instincts).
        self.fullback_registry.register_team(players)
        # Checkpoint 33 — register midfielders (CM/CAM) and strikers (ST/CF).
        self.midfield_registry.register_team(players)
        self.striker_registry.register_team(players)

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
            top_speed_mpm=self._pace_mpm_for(player),
        )
        self.team_rosters.setdefault(team_name, []).append(name)
        # Feature #1/#2 — the sub's home join the base-shape capture so the
        # in-match stance/pattern deltas apply to them like everyone else.
        self._capture_base_homes(team_name)
        self._applied_stance_key.pop(team_name, None)
        # Checkpoint 18 — register the sub's winger profile if they're a winger.
        self.winger_registry.register_player(player)
        # Checkpoint 32 — register the sub's fullback profile if they're a FB.
        self.fullback_registry.register_player(player)
        # Checkpoint 33 — register midfielder / striker profile if applicable.
        self.midfield_registry.register_player(player)
        self.striker_registry.register_player(player)

    # ── FEATURE #1/#2: IN-MATCH SHAPE STATE ────────────────────────────
    # Formation stances (chasing / seeing it out / man down ...) and attack
    # patterns (overloads, box midfield, wing isolation ...) are expressed as
    # ADDITIVE per-role home deltas on TOP of the authored shape. Home anchors
    # feed receiver quality (the pass network), restart snaps, width-stretch
    # steering and delivery aim — so a changed shape genuinely re-shapes the
    # live XI instead of only flipping a dial.

    def _capture_base_homes(self, team_name: str) -> None:
        """Record every current home anchor as the stable authored base."""
        self._formation_base_homes[team_name] = {
            name: (st.home_x, st.home_y)
            for name, st in self.states.items()
            if st.team == team_name
        }

    def apply_formation_stance(
        self,
        team_name: str,
        stance: FormationStance,
        own_red_cards: int = 0,
        avg_stamina: float = 100.0,
    ) -> bool:
        """Apply a formation stance's role deltas to a team's home anchors.

        Idempotent: returns True only when the composite shape actually
        changed (so the caller can start a reshape animation). Man-down and
        fatigue are stacked modifiers exactly as the dials are (which is why
        they are accepted here alongside the stance)."""
        key = (stance, own_red_cards > 0, avg_stamina < 75.0)
        if self._applied_stance_key.get(team_name) == key:
            return False
        self._stance_deltas[team_name] = stance_delta_roles(
            stance, own_red_cards=own_red_cards, avg_stamina=avg_stamina
        )
        self._applied_stance_key[team_name] = key
        self._recompute_homes(team_name)
        return True

    def apply_attack_pattern(
        self, team_name: str, pattern: AttackPattern
    ) -> bool:
        """Apply an attack pattern's role deltas (empty for NONE).

        Idempotent; returns True only when the pattern actually changed.
        A team OOP is given NONE so its shape is the pure defensive stance —
        patterns are possession shapes."""
        if self._applied_pattern.get(team_name) == pattern:
            return False
        self._pattern_deltas[team_name] = pattern_role_deltas(pattern)
        self._applied_pattern[team_name] = pattern
        self._recompute_homes(team_name)
        return True

    # ── COACH WIDTH INSTRUCTIONS (Phase 8 v1: "stay wide" / "tuck in") ──
    # Per-role home deltas in attacking-right normalised space, applied
    # BESIDE stance + pattern deltas. Idempotent on the command value.
    _COACH_WIDTH_DELTAS: Dict[float, Dict[str, Tuple[float, float]]] = {
        1.0: {"LW": (0.0, -2.5), "LB": (0.0, -2.5),
              "RW": (0.0, 2.5), "RB": (0.0, 2.5)},
        -1.0: {"LW": (0.0, 1.5), "LB": (0.0, 1.5),
               "RW": (0.0, -1.5), "RB": (0.0, -1.5)},
        0.0: {},
    }

    def apply_coach_width(self, team_name: str, width_cmd: float) -> bool:
        """Apply the coach's width instruction for a team.

        width_cmd: +1 stay wide, -1 tuck in, 0 none. Idempotent; returns
        True only when the command actually changed."""
        cmd = 1.0 if width_cmd > 0.5 else (-1.0 if width_cmd < -0.5 else 0.0)
        if self._applied_coach_width.get(team_name) == cmd:
            return False
        self._coach_deltas[team_name] = dict(self._COACH_WIDTH_DELTAS[cmd])
        self._applied_coach_width[team_name] = cmd
        self._recompute_homes(team_name)
        return True

    def _recompute_homes(self, team_name: str) -> None:
        """home = base + stance_delta + pattern_delta + coach_delta, mirrored
        for left-attacking teams, clamped to the pitch. All delta tables map
        by ROLE in attacking-right normalised space, so the away-mirror is
        just a sign flip (dx on the x-axis, dy on the y-axis)."""
        base = self._formation_base_homes.get(team_name, {})
        sgn = 1.0 if self.team_attacks_right.get(team_name, True) else -1.0
        sd = self._stance_deltas.get(team_name, {})
        pd = self._pattern_deltas.get(team_name, {})
        cd = self._coach_deltas.get(team_name, {})
        for name, st in self.states.items():
            if st.team != team_name:
                continue
            bx, by = base.get(name, (st.home_x, st.home_y))
            sdx = sd.get(st.position, (0.0, 0.0))
            pdx = pd.get(st.position, (0.0, 0.0))
            cdx = cd.get(st.position, (0.0, 0.0))
            dx = (sdx[0] + pdx[0] + cdx[0]) * sgn
            dy = (sdx[1] + pdx[1] + cdx[1]) * sgn
            st.home_x = max(4.0, min(101.0, bx + dx))
            st.home_y = max(2.0, min(66.0, by + dy))

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

    @staticmethod
    def _pace_mpm_for(player) -> float:
        """Top sprint speed in metres per simulation-minute (pace 50→~6.5, 95→~13).

        Bounds the per-minute run-step so wide runners travel at a
        realistic rate instead of teleporting (Checkpoint 32b continuity).
        """
        phys = getattr(getattr(player, "dna", None), "physical", None)
        pace = getattr(phys, "pace", 65.0) if phys is not None else 65.0
        return max(16.0, min(40.0, 14.0 + (pace - 50.0) / 45.0 * 26.0))

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
            # touch has dragged the player >FLANK_HOLD_TRIGGER_M OFF his
            # flank channel, so an on-flank touch is tracked exactly
            # (preservation contract) and the pull always moves y TOWARD
            # home_y, never away. The x stays at the ball (he IS at the
            # ball); only the lateral placement is anchored.
            # Checkpoint 34: trigger narrowed 6.0 -> 3.0. Wingers were
            # equilibrating ~8m off the line (peak receptions at y≈14 vs a
            # home anchor at y=6) because any reception landing 3-6m infield
            # was tracked exactly and pinned the half-space equilibrium in
            # place. Snapping at >3m instead drags the reception band back
            # toward the touchline.
            FLANK_HOLD_TRIGGER_M = 3.0
            if state.position in ("LB", "RB", "LW", "RW"):
                anchor_y = state.home_y
                if abs(anchor_y - state.current_y) > FLANK_HOLD_TRIGGER_M:
                    pull = 0.35
                    if state.position in ("LW", "RW"):
                        wp = self.winger_registry.get(player_name)
                        if wp is not None:
                            # Checkpoint 31: 0.25+0.20c left ~58% of every
                            # infield touch uncorrected, so the heat map kept
                            # the RM/LM half-space band (|y-c|≈15-17 vs an
                            # anchor at |y-c|=24). A real winger's first
                            # action after receiving infield is to rip back
                            # out to the touchline — pull hard.
                            pull = 0.40 + 0.35 * wp.flank_commitment
                    # Checkpoint 35 — PITCH-STRETCH RULE: a touch that
                    # pinned the wide player on the CENTRAL spine (packed
                    # middle) must snap back to the line harder than a
                    # half-space touch; the spine weight adds stretch power
                    # so the width duty survives a central reception. Clamped
                    # so the player can never be flung past the line.
                    pull = min(0.95, pull + STRETCH_TOUCH_BOOST * pitch_spine_weight(state.current_y))
                    state.current_y += (anchor_y - state.current_y) * pull
                    state.current_y = max(
                        STRETCH_CLAMP_LOW, min(STRETCH_CLAMP_HIGH, state.current_y)
                    )
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

    # ── PENDING SET-PIECE TARGET WINDOW (2026-10-02) ───────────────────────
    #
    # A set piece is the one situation where a player's target is NOT a function
    # of the ball and his shape anchor. During a corner the box arrangement IS
    # the position: a centre-back's job is to stand between his man and the
    # goal, which is 20 m from his shape anchor and 30 m from where the shape
    # layer would otherwise pull him.
    #
    # Without this, giving a corner any elapsed time makes things WORSE — the
    # integrator integrates toward `home_anchor + ball compaction`, which sits
    # OUTSIDE the box, so players jog away from their slot for the whole
    # jostling period. The window is opened by the set-piece chain and consumed
    # by the integrator itself (`tick_setpiece`), so no caller has to remember
    # to close it and an exception mid-chain cannot leave it stuck on.
    # How far off his assigned mark a player actually ends up after a
    # run into a crowded box. Real arrivals miss the chalk by a metre or
    # two because the man is jostling, being held and still adjusting.
    # Zero was measurably wrong: a player parked exactly on his slot is
    # standing on the ball's destination, so `resolve_aerial_delivery`
    # scores him at ~0 arrival and no marker can beat him. Corner aerial
    # duels resolved 8/8 and 15/15 to the attack before this.
    SET_PIECE_ARRIVAL_ERROR_M = 1.8
    # Ceiling for the same reason: `SET_PIECE_ARRIVED_M` is 3.0, and a
    # player who overshoots by more than that is NOT in the box.
    SET_PIECE_ARRIVAL_ERROR_MAX_M = 2.4
    SET_PIECE_ACTIVE: bool = True

    def advance_to_slots(self, targets: Dict[str, Tuple[float, float]],
                         seconds: float,
                         exclude: Optional[set] = None) -> float:
        """RUN the named players to their slots over `seconds`. Returns metres.

        Synchronous, because the delivery it precedes is resolved in the same
        call: a set-piece window opened for the post-chain integrator cannot
        influence the header it was opened for. See the module note on the
        ordering defect.

        Speed eases in and out -- a man does not accelerate from a standing
        start to a sprint inside one tick, and he decelerates into his slot
        rather than oscillating through it, because the target is re-read from
        the same table on every step.
        """
        import math as _m

        exclude = exclude or set()
        # While still running, settle onto the mark at this tolerance; the
        # arrival residual below decides where he finishes.
        SETTLE_GAP_M = 0.4
        budget = max(0.0, float(seconds))
        step_dt = 0.1
        steps = max(1, int(round(budget / step_dt)))
        total = 0.0
        for pname, (gx, gy) in (targets or {}).items():
            if pname in exclude:
                continue
            st = self.states.get(pname)
            if st is None or st.current_x is None or st.current_y is None:
                continue
            top = float(getattr(st, "physics_top_speed_mps", 0.0) or 0.0)
            if top <= 1.0:
                # Not yet measured by the physics layer this minute, so fall
                # back to the engine's own band: a corner run-in is a sprint,
                # not the 1.5-1.9 m/s shape-holding trot.
                top = 6.0
            top = min(max(top, 3.0), 7.5)
            cx, cy = float(st.current_x), float(st.current_y)
            travelled = 0.0
            sprint_s = 0.0
            for _ in range(steps):
                dx, dy = gx - cx, gy - cy
                dist = _m.hypot(dx, dy)
                # Stop short of the mark by a realistic residual rather than
                # driving to the coordinate exactly. See the class docstring:
                # parking a man exactly on the ball's destination makes every
                # corner aerial unwinnable for the defence, and the error is a
                # property of the man's arrival, not of his team.
                _ex = _m.hypot(cx - gx, cy - gy)
                if _ex <= SETTLE_GAP_M:
                    break
                # ease in over the first 40% of a second and out inside 2 m
                v = top
                if travelled < top * 0.4:
                    v = top * max(0.35, travelled / max(top * 0.4, 1e-6))
                if dist < 2.0:
                    v = min(v, dist / 0.4)
                adv = min(v * step_dt, dist)
                cx += dx / dist * adv
                cy += dy / dist * adv
                travelled += adv
                if v >= 5.5:
                    sprint_s += step_dt
            # The residual: where he actually finished relative to the mark.
            # One draw per player, so a full box does not draw 13 numbers and
            # silently shift the whole match's random stream twice.
            _gx_off = random.gauss(0.0, self.SET_PIECE_ARRIVAL_ERROR_M / 1.8)
            _gy_off = random.gauss(0.0, self.SET_PIECE_ARRIVAL_ERROR_M / 1.8)
            _off = _m.hypot(_gx_off, _gy_off)
            if _off > self.SET_PIECE_ARRIVAL_ERROR_MAX_M:
                _sc = self.SET_PIECE_ARRIVAL_ERROR_MAX_M / _off
                _gx_off *= _sc
                _gy_off *= _sc
            cx, cy = gx + _gx_off, gy + _gy_off
            st.current_x, st.current_y = cx, cy
            if travelled <= 0.0:
                continue
            total += travelled
            self.record_physics_distance(
                pname, distance_m=travelled, duration_s=budget,
                speed_mps=travelled / max(budget, 1e-6),
                sprint_time_s=sprint_s, sprint_count=1,
                high_speed_sprint_count=1 if sprint_s > 0.0 else 0,
                top_speed_mps=top)
        return total

    def open_setpiece_window(self, targets: Dict[str, Tuple[float, float]],
                             seconds: float) -> None:
        """Hold these targets for `seconds` of integrated off-ball time."""
        if not self.SET_PIECE_ACTIVE:
            return
        self._setpiece_targets = dict(targets or {})
        self._setpiece_remaining_s = max(0.0, float(seconds))

    def setpiece_target(self, player_name: str
                        ) -> Optional[Tuple[float, float]]:
        """The pending slot for this player, or None if he is free."""
        if self._setpiece_remaining_s <= 0.0:
            return None
        return self._setpiece_targets.get(player_name)

    def arrived_setpiece_players(self,
                              max_gap_m: float = 3.0) -> set:
        """Names of pending-slot holders who actually got there.

        Measured against the slot, not the goal: a player is 'in the box'
        only if he reached the position he was assigned. Measured in 12
        corners, 62% of assignments arrived and 26% were still >15 m
        away at the cross -- and those men are genuinely not in the duel.
        """
        if self._setpiece_remaining_s <= 0.0:
            return set()
        import math as _m
        got = set()
        for name, (gx, gy) in self._setpiece_targets.items():
            st = self.states.get(name)
            if st is None or st.current_x is None or st.current_y is None:
                continue
            if _m.hypot(gx - float(st.current_x),
                       gy - float(st.current_y)) <= max_gap_m:
                got.add(name)
        return got

    def tick_setpiece(self, seconds: float) -> None:
        """Consume the window. Called by the integrator with the time it ran."""
        if self._setpiece_remaining_s <= 0.0:
            return
        self._setpiece_remaining_s -= max(0.0, float(seconds))
        if self._setpiece_remaining_s <= 0.0:
            self._setpiece_targets = {}

    def setpiece_active(self) -> bool:
        return self._setpiece_remaining_s > 0.0

    def set_piece_place(self, player_name: str, x: float, y: float,
                        minute: int):
        """Place a player for a SET PIECE, honoured verbatim.

        `record_touch` is the right call for a ball touch: after planting the
        player it applies the wide-role flank hold (Checkpoint 21d) and the
        goalkeeper's own-box anchor, because in open play a winger who is
        dragged 20 m infield genuinely has drifted off his flank.

        Those corrections are WRONG for a wall. A wall is an ordered
        geometric arrangement, and measured on 3 real matches the corrections
        quietly dismantled it: 13 of 14 walls were wider than the men in them
        could possibly produce (expected 1.65 m of face for 4 men, measured up
        to 40.5 m), and individual men ended up 3.0 m and 4.0 m from the ball
        when the wall distance is 9.15 m. The men were being pulled to their
        shape anchors after being placed, so the blocker set handed to
        `resolve_shot` was a scatter of bodies rather than a screen.

        So this writes the position EXACTLY, and still books the displacement
        into `minute_touch_distance` so the distance-covered accounting stays
        honest — a man who jogs 20 m to the wall has covered 20 m, and hiding
        that would just move the lie to a different column.
        """
        if x is None or y is None:
            return
        state = self.states.get(player_name)
        if not state:
            return
        start_x, start_y = state.current_x, state.current_y
        state.touch_at(x, y, minute)   # exact write, clamped to the pitch
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
            state.current_x = max(0.0, min(28.0, state.current_x))
        else:
            state.current_x = max(77.0, min(105.0, state.current_x))

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

    def enforce_gk_anchor(self, team_name: str) -> int:
        """Re-apply the goalkeeper's own-box anchor to every keeper on a team.

        `_anchor_gk_in_own_box` only runs inside `record_touch`, so it covers
        the one path that plants a keeper at the ball. It does NOT cover the
        dozens of off-ball shape writes that mutate `current_x` directly
        (`drift_minute`, `defensive_block`, the press pulls, the run targets),
        and a keeper written upfield by any of those is never pulled back.

        Measured on a real match: 168 of 325 keeper position writes in a
        single match landed at x >= 40, peaking at 80-105, i.e. the keeper
        spent most of the match at the far end of the pitch. That is a
        PRE-EXISTING hole — the anchor was simply never applied on those
        paths — but the foul-awarded free kick (match_engine) gives it many
        more chances to bite, because every restart runs the off-ball phase
        again with the ball deep in the defending third.

        Enforcing it as a single invariant at the end of the off-ball phase is
        the right shape of fix: it is one choke point instead of ~30 write
        sites, and it cannot be bypassed by a shape rule added later. A
        keeper may step out to receive a back-pass (the anchor allows
        x <= 28) but never follows play upfield.

        Returns the number of keepers clamped, so a caller can assert on it.
        """
        n = 0
        for name in self.team_rosters.get(team_name, []):
            st = self.states.get(name)
            if st is None or getattr(st, "position", "") != "GK":
                continue
            before = st.current_x
            self._anchor_gk_in_own_box(st)
            if st.current_x != before:
                n += 1
        return n

    def snapshot_positions(self, team_name: str) -> Dict[str, Tuple[float, float]]:
        """Capture each player's current (x, y) — call BEFORE the
        drift_minute / defensive_block / attacking_crash sequence."""
        return {
            name: (self.states[name].current_x, self.states[name].current_y)
            for name in self.team_rosters.get(team_name, [])
            if name in self.states
        }

    # EMA smoothing for the velocity E-field feed (0.6 = strong current-minute weight)
    _VELOCITY_EMA_ALPHA: float = 0.6

    def accumulate_drift_from_snapshot(
        self, team_name: str, before: Dict[str, Tuple[float, float]]
    ):
        """Diff current positions against a prior snapshot and add the
        real net distance moved to each player's minute_drift_distance.
        Also folds the signed delta into the player's velocity EMA, so the
        motion-aware pitch control field sees stable movement vectors.
        Call AFTER drift_minute / defensive_block / attacking_crash have
        all run for this team this minute."""
        a = self._VELOCITY_EMA_ALPHA
        for name, (px, py) in before.items():
            state = self.states.get(name)
            if state is None:
                continue
            dx, dy = state.current_x - px, state.current_y - py
            dist = (dx * dx + dy * dy) ** 0.5
            state.minute_drift_distance += dist
            state.velocity_x = (1 - a) * state.velocity_x + a * dx
            state.velocity_y = (1 - a) * state.velocity_y + a * dy

    def record_physics_distance(
        self,
        player_name: str,
        distance_m: float = 0.0,
        duration_s: float = 0.0,
        speed_mps: float = 0.0,
        walk_time_s: float = 0.0,
        jog_time_s: float = 0.0,
        sprint_time_s: float = 0.0,
        sprint_count: float = 0.0,
        high_speed_sprint_count: float = 0.0,
        top_speed_mps: float = 0.0,
    ) -> None:
        """
        Accumulate physics-derived movement stats from PossessionEpisode trace.

        These are additive with the existing touch/drift accumulators and
        are reset each minute by pop_minute_activity().
        """
        state = self.states.get(player_name)
        if state is None:
            return
        state.physics_distance_m += float(distance_m)
        duration_s = max(0.0, float(duration_s))
        speed_mps = max(0.0, float(speed_mps))
        state.physics_walk_time_s += max(0.0, float(walk_time_s))
        state.physics_jog_time_s += max(0.0, float(jog_time_s))
        state.physics_sprint_time_s += max(0.0, float(sprint_time_s))
        if duration_s > 0.0:
            if speed_mps < 2.0:
                state.physics_walk_time_s += duration_s
            elif speed_mps < 7.0:
                state.physics_jog_time_s += duration_s
            else:
                state.physics_sprint_time_s += duration_s
        state.physics_sprint_count += float(sprint_count)
        state.physics_high_speed_sprint_count += float(high_speed_sprint_count)
        if float(top_speed_mps) > state.physics_top_speed_mps:
            state.physics_top_speed_mps = float(top_speed_mps)

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
            physics_distance_m            — true distance from physics trace.
            physics_sprint_count          — sprint segments from physics trace.
            physics_high_speed_sprint_count — high-speed sprint segments.
            physics_top_speed_mps         — max observed speed from physics trace.
        """
        state = self.states.get(player_name)
        if state is None:
            return {
                "distance_touch": 0.0, "distance_drift": 0.0,
                "distance_total": 0.0, "touches": 0, "peak_touch_jump": 0.0,
                "physics_distance_m": 0.0,
                "physics_walk_time_s": 0.0,
                "physics_jog_time_s": 0.0,
                "physics_sprint_time_s": 0.0,
                "physics_sprint_count": 0.0,
                "physics_high_speed_sprint_count": 0.0,
                "physics_top_speed_mps": 0.0,
            }
        out = {
            "distance_touch": round(state.minute_touch_distance, 2),
            "distance_drift": round(state.minute_drift_distance, 2),
            "distance_total": round(state.minute_touch_distance + state.minute_drift_distance, 2),
            "touches": state.minute_touch_count,
            "peak_touch_jump": round(state.minute_peak_touch_jump, 2),
            "physics_distance_m": round(state.physics_distance_m, 2),
            "physics_walk_time_s": round(state.physics_walk_time_s, 2),
            "physics_jog_time_s": round(state.physics_jog_time_s, 2),
            "physics_sprint_time_s": round(state.physics_sprint_time_s, 2),
            "physics_sprint_count": round(state.physics_sprint_count, 1),
            "physics_high_speed_sprint_count": round(state.physics_high_speed_sprint_count, 1),
            "physics_top_speed_mps": round(state.physics_top_speed_mps, 2),
        }
        state.minute_touch_distance = 0.0
        state.minute_touch_count = 0
        state.minute_peak_touch_jump = 0.0
        state.minute_drift_distance = 0.0
        state.physics_distance_m = 0.0
        state.physics_walk_time_s = 0.0
        state.physics_jog_time_s = 0.0
        state.physics_sprint_time_s = 0.0
        state.physics_sprint_count = 0.0
        state.physics_high_speed_sprint_count = 0.0
        state.physics_top_speed_mps = 0.0
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

    # Checkpoint 31 — deterministic role/third ball-proximity bands while
    # in possession. Defenders protect the line as the ball advances,
    # midfielders offer a passing tier, and the front line gets closer only
    # in the final third. Values are metres from the ball.
    BALL_PROXIMITY_BANDS_BY_THIRD: Dict[str, Dict[str, tuple]] = {
        "defensive": {
            "CB": (8.0, 12.0), "LB": (10.0, 15.0), "RB": (10.0, 15.0),
            "CDM": (9.0, 13.0), "CM": (11.0, 15.0),
            "LW": (22.0, 30.0), "RW": (22.0, 30.0),
            "ST": (28.0, 36.0), "CF": (26.0, 34.0),
        },
        "middle": {
            "CB": (16.0, 22.0), "LB": (16.0, 23.0), "RB": (16.0, 23.0),
            "CDM": (12.0, 16.0), "CM": (14.0, 18.0),
            "LW": (18.0, 25.0), "RW": (18.0, 25.0),
            "ST": (20.0, 28.0), "CF": (18.0, 26.0),
        },
        "attacking": {
            "CB": (30.0, 40.0), "LB": (25.0, 35.0), "RB": (25.0, 35.0),
            "CDM": (18.0, 24.0), "CM": (20.0, 27.0),
            "LW": (12.0, 18.0), "RW": (12.0, 18.0),
            "ST": (10.0, 16.0), "CF": (10.0, 16.0),
        },
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
        danger_level: float = 0.0,
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

        # Checkpoint 33b — CONTINUITY CLAMP snapshot. Capture each player's
        # position at the start of this minute so we can enforce a realistic
        # per-minute displacement cap at the end (no player may "teleport"
        # more than his top sprint speed allows in one tick), for EVERY
        # position and EVERY state — not just committed runs.
        _start_snap = {
            n: (s.current_x, s.current_y)
            for n, s in self.states.items() if s is not None
        }

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

        # ── CHECKPOINT 25: BALL-SIDE SQUEEZE WHEN DEFENDING ──────────
        # Below the defensive_block danger threshold the unit still shifts
        # laterally toward the ball every minute, graded by danger — real
        # defensive shape "slides across with the ball" continuously, not
        # only when a live threat forces it. Excludes wing anchors (LW/RW/
        # LB/RB) so the squeeze never fights the touchline anchoring.
        # Checkpoint 31b — CAM joined: the #10 must track the ball laterally
        # when defending too, not stand frozen on his home marker.
        BALL_SQUEEZE_POSITIONS = ("GK", "CB", "CDM", "CM", "CAM")
        squeeze_max = 0.30
        squeeze_shift = (
            squeeze_max * max(0.0, min(1.0, danger_level / 100.0))
            if (not in_possession and ball_y is not None) else 0.0
        )

        press_pull = 0.45 + getattr(profile, "press_intensity", 0.5) * 0.15
        attacks_right = self.team_attacks_right.get(team_name, True)

        # Checkpoint 29 — the OPPONENT's block this team orbits when in
        # possession. Home attacks right → opponent is the away team →
        # away_block (mirrored for the away side), same rule the pass
        # selection layer uses.
        opp_block = None
        if in_possession:
            ctx_home_block, ctx_away_block = self._block_context
            opp_block = ctx_away_block if attacks_right else ctx_home_block

        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None:
                continue

            player_shape_shift = base_shape_shift * self.SHAPE_SHIFT_SCALE.get(state.position, 1.0)
            effective_home_x = state.home_x + chase_shift + player_shape_shift

            # ── CHECKPOINT 30: FORWARD BUILD-UP DROP ───────────────────
            # A real forward is not a static tower. When the team builds from
            # the back he drops into the half-space to receive, creating a
            # passing lane and pulling a defender out of shape. The drop is
            # baked into effective_home_x so the normal drift pull naturally
            # moves him there — no extra post-pass override needed.
            if (
                in_possession
                and ball_x is not None
                and state.position in ("LW", "ST", "RW")
            ):
                if attacks_right:
                    if ball_x < 38.0:
                        effective_home_x = min(effective_home_x, 58.0 if state.position == "ST" else 62.0)
                    elif ball_x < 62.0:
                        effective_home_x = min(effective_home_x, 66.0 if state.position == "ST" else 70.0)
                else:
                    if ball_x > 67.0:
                        effective_home_x = max(effective_home_x, 47.0 if state.position == "ST" else 43.0)
                    elif ball_x > 43.0:
                        effective_home_x = max(effective_home_x, 39.0 if state.position == "ST" else 35.0)

            effective_home_x = max(4.0, min(101.0, effective_home_x))
            effective_home_y = state.home_y

            # ── CHECKPOINT 31: MEDIAN BALL-PROXIMITY BAND ───────────
            # In possession, radially adjust the anchor so its distance
            # from the ball sits inside the position's median band. The
            # drift then converges there naturally instead of fighting
            # a separate correction force.
            if (
                in_possession
                and ball_x is not None and ball_y is not None
                and state.position in self.BALL_PROXIMITY_BANDS_BY_THIRD["middle"]
            ):
                effective_home_x, effective_home_y = (
                    self._band_adjusted_anchor(
                        state, effective_home_x, effective_home_y,
                        ball_x, ball_y, attacks_right,
                    )
                )

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
            state.current_y += (effective_home_y - state.current_y) * press_pull

            # Checkpoint 25 — slide the block laterally toward the ball.
            if squeeze_shift > 0.0 and state.position in BALL_SQUEEZE_POSITIONS:
                state.current_y += (ball_y - state.current_y) * squeeze_shift

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
                # Checkpoint 32 — DNA-driven: a defensive_fullback snaps
                # back to his channel harder than an inverted_fullback,
                # who tolerates being infield during build-up.
                fb_profile = self.fullback_registry.get(name)
                commitment = fb_profile.flank_commitment if fb_profile is not None else 0.75
            if flank_drift > 8.0:
                pull = (0.45 if in_possession else 0.30) * (0.5 + commitment * 0.5)
            elif flank_drift > 4.0:
                pull = (0.30 if in_possession else 0.20) * (0.5 + commitment * 0.5)
            else:
                pull = 0.12
            # Checkpoint 35 — PITCH-STRETCH RULE: when the live ball sits on
            # the central spine the wide player's flank hold is a DUTY, not
            # a drift habit — boost the anchor pull by the spine weight so
            # he re-asserts the touchline while the middle is packed, and
            # clamp the result inside the pitch.
            pull *= 1.0 + STRETCH_DRIFF_BOOST * (
                pitch_spine_weight(ball_y) if ball_y is not None else 0.0
            )
            state.current_y += (state.home_y - state.current_y) * pull
            state.current_y = max(
                STRETCH_CLAMP_LOW, min(STRETCH_CLAMP_HIGH, state.current_y)
            )

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
                # Checkpoint 24 — byline retreat. Carries and runs can push a
                # winger onto the goal line (x≈100+), and since nothing ever
                # pulled him back, he CAMPED there — every disposal from the
                # corner flag is a cross/cutback by definition, which is what
                # inflated winger cross counts to 20-30/match. A real winger
                # operates from the cutback station (5-15m off the line):
                # beyond x≈94 he drifts back toward it.
                if state.position in ("LW", "RW"):
                    if attacks_right and state.current_x > 94.0:
                        state.current_x -= (state.current_x - 90.0) * 0.35
                    elif not attacks_right and state.current_x < 11.0:
                        state.current_x += (15.0 - state.current_x) * 0.35

        # ── FRONTLINE TRACKING-BACK OUT OF POSSESSION ─────────────
        # Wingers and strikers are not parked in the final third when the
        # team is defending. Out of possession they retreat toward the
        # midfield band so the frontline stays connected to the rest of
        # the shape instead of becoming isolated islands.
        if not in_possession:
            for name in self.team_rosters.get(team_name, []):
                state = self.states.get(name)
                if state is None or state.position not in ("LW", "ST", "RW"):
                    continue
                if attacks_right:
                    if state.current_x <= 65.0:
                        continue
                    target_x = 62.0 if state.position == "ST" else 65.0
                    pull = 0.20 if state.position == "ST" else 0.25
                else:
                    if state.current_x >= 40.0:
                        continue
                    target_x = 43.0 if state.position == "ST" else 40.0
                    pull = 0.20 if state.position == "ST" else 0.25
                state.current_x += (target_x - state.current_x) * pull

        # ── CHECKPOINT 32: FULLBACK ADVANCE / TUCK / RECOVER ──────
        # Consume the FullbackBehaviorEngine so overlaps, underlaps and
        # inverted tucks actually HAPPEN in the drift, instead of the
        # profiles sitting unused in the registry. In possession, when the
        # ball is live on (or near) his flank, the fullback may take an
        # outside overlap, an inside underlap, or — inverted FB in build-up
        # — step into the midfield pocket. Out of possession, if the ball is
        # turned over behind him on his flank, he sprints back (recovery).
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("LB", "RB"):
                continue
            fb_profile = self.fullback_registry.get(name)
            if fb_profile is None:
                continue
            # A fullback who was just involved this minute keeps his touch
            # coordinates authoritative (same rule as the winger forward
            # anchor above).
            if state.last_active_minute == minute:
                continue
            anchor_y = state.home_y

            if in_possession:
                # Continuous overlap/underlap/tuck runs are now handled by
                # _wide_run_step (pace-capped + shape-aware) at the end of
                # drift_minute, so they flow as trajectories instead of
                # snapping here.
                continue
            else:
                # Recovery: caught upfield, ball behind on his flank.
                if ball_x is None or ball_y is None:
                    continue
                if FullbackBehaviorEngine.should_recovery_sprint(
                        fb_profile, state.current_x, state.current_y,
                        ball_x, ball_y, attacks_right, anchor_y=anchor_y):
                    state.current_x += (state.home_x - state.current_x) * 0.30
                    state.current_y += (state.home_y - state.current_y) * 0.30

        # Checkpoint 31 — median ball-proximity bands: applied INSIDE the
        # drift below by radially adjusting each CDM/CM/LB/RB anchor so the
        # ordinary home drift itself converges inside the band (a post-hoc
        # nudge always lost to the much stronger home pull).

        # Checkpoint 6 — line cohesion: apply AFTER individual drift so
        # defensive/midfield lines pull toward their own line-mates' average
        # position, representing a back four/midfield three shifting as a
        # unit rather than each player being an independent dot anchored
        # only to their own personal home position.
        self._apply_line_cohesion(team_name)

        # Checkpoint 27 — formation graph physics: one spring iteration over
        # neighbor-pair edges (CB-CB, CDM-CM, flank pairs...) so pairs move
        # as coordinated units, plus min-separation repulsion so team-mates
        # don't cluster. Defending tightens springs (block as a unit);
        # attacking loosens them so runners can break shape.
        self._graph_relaxation(team_name, in_possession)

        # ── CHECKPOINT 29: HALF-SPACE MAGNET (post-cohesion) ─────────
        # Applied AFTER line cohesion and graph relaxation because both
        # average midfielders back toward their line's center — which
        # washed the magnet out when it was folded into the drift target
        # (A/B: half-space occupancy unchanged). Blending the live
        # position here lets the orbital web's NODES sit in the
        # half-space channels the pass-selection layer aims at.
        if in_possession and opp_block is not None:
            self._apply_half_space_magnet(team_name, opp_block)

        # Checkpoint 6.2 — midfield geometric coverage (Enzo/Rice/Pedri):
        # in possession, midfielders with high geometric_awareness drift
        # to cover vacant half-spaces when teammates are isolated.
        if in_possession:
            self._midfielder_geometric_coverage(team_name, minute=minute)

            # Checkpoint 31b — #10 pocket roaming: the CAM's home already
            # sits where the ball usually is, so the proximity band has
            # nothing to correct for him; give him an explicit roaming
            # behaviour one tier ahead of the carrier instead.
            if ball_x is not None and ball_y is not None:
                self._cam_pocket_roam(
                    team_name, ball_x, ball_y,
                    self.team_attacks_right.get(team_name, True), minute,
                )

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

            # Checkpoint 32b — continuous, shape-aware wide runs (wingers +
            # fullbacks) replace the old per-minute positional snap with a
            # cached run target travelled at a pace-capped rate, bending off
            # opponent pressure (counter-press awareness).
            self._wide_run_step(
                team_name, minute, ball_x, ball_y,
                self.team_attacks_right.get(team_name, True),
                opponent_players or [],
            )

            # Checkpoint 33 — continuous, shape-aware midfield + striker runs.
            self._midfield_run_step(
                team_name, minute, ball_x, ball_y,
                self.team_attacks_right.get(team_name, True),
                opponent_players or [],
                opp_block=opp_block,
            )
            self._striker_run_step(
                team_name, minute, ball_x, ball_y,
                self.team_attacks_right.get(team_name, True),
                opponent_players or [],
            )

            # Checkpoint 31c — the role/third proximity target is the final
            # in-possession positional authority. Role-specific run choices
            # may choose the route in a future layer, but they must not move
            # these roles outside their calibrated distance from the ball.
            self._enforce_ball_proximity_targets(
                team_name, ball_x, ball_y,
                self.team_attacks_right.get(team_name, True),
                start_positions=_start_snap,
                minute=minute,
            )

        else:
            # Checkpoint 33c — OUT-OF-POSSESSION midfield. The engine's
            # press/recovery instincts were built but never consumed; this
            # drives them: ball-winners step onto the carrier, holders drop
            # to screen the block, the rest hold shape. Pace-capped like the
            # possession run steps. Runs every OOP minute, but is itself
            # bounded by the continuity clamp below.
            self._midfield_defensive_step(
                team_name, minute, ball_x, ball_y,
                self.team_attacks_right.get(team_name, True),
                opponent_players or [],
            )

        # Checkpoint 33b — CONTINUITY CLAMP. Enforce a realistic per-minute
        # displacement cap for EVERY player in EVERY state (not just the
        # committed-run steps), so no one teleports/jumps more than his top
        # sprint speed allows in a single tick. Touched players (placed by
        # record_touch this minute) are skipped — their ball coordinates are
        # authoritative and must not be yanked off the ball they received.
        for name in self.team_rosters.get(team_name, []):
            st = self.states.get(name)
            if st is None or name not in _start_snap:
                continue
            if st.last_active_minute == minute:
                continue
            sx, sy = _start_snap[name]
            dx = st.current_x - sx
            dy = st.current_y - sy
            dist = math.hypot(dx, dy)
            cap = st.top_speed_mpm
            if dist > cap:
                f = cap / dist
                st.current_x = sx + dx * f
                st.current_y = sy + dy * f

    # ── LOW BLOCK ─────────────────────────────────────────────────────
    # PITCH bounds, named because the low-block geometry below is expressed
    # relative to them and bare 105.0/68.0/34.0 literals in a formula that
    # reads "half the pitch" are how a pitch-length change gets missed.
    PITCH_X_MAX = 105.0
    PITCH_Y_MAX = 68.0
    PITCH_Y_MID = 34.0

    # A low block is a SHAPE, not a press intensity. `pressing_profiles`
    # already models how hard such a team presses; these model where its
    # three lines stand and how narrow the unit gets.
    #
    # Every value is metres from own goal, or metres either side of the
    # touchline. They are tactical choices, and the reasoning is in
    # `low_block_target` — briefly:
    #
    #   back four  19 m   inside the own third, which is the definition of
    #                       the shape. PLOFA had them at 42.5 m, i.e. past
    #                       halfway, in the opponent's half.
    #   midfield   31 m   a genuinely separate line 12 m behind the front of
    #                       the defence. PLOFA had a 0.7 m gap, which is one
    #                       flat eight-man wall rather than three lines.
    #   front two  37 m   HIGH on purpose, ~6 m ahead of the midfield. A
    #                       deep regain needs an outlet immediately, and this
    #                       is the post's "platform to attack".
    LB_BACK_DEPTH = 19.0
    LB_MID_DEPTH = 31.0
    LB_FRONT_DEPTH = 37.0
    #: The block sinks a little further the wider the ball is, because a wide
    #: ball pulls the defensive line across and back. 4 m at the extreme flank.
    LB_FRONT_DROP = 4.0

    #: Half-widths. The midfield line is NARROWER than the back four, which is
    #: the real geometry of a low block: the wide midfielders tuck in to cut
    #: the half-space, and the fullbacks alone hold the width.
    LB_BACK_HALF_WIDTH = 20.0
    LB_MID_HALF_WIDTH = 16.0
    LB_FRONT_HALF_WIDTH = 11.0

    #: How far toward the ball the whole unit slides. Partial on purpose: a
    #: block that tracks the ball exactly has been stretched by it, and one
    #: that does not move is not compact. The generic shape target uses 0.07,
    #: which is far too little to be a block at all.
    LB_SHIFT = 0.42
    #: Half-width of the "spine" — within this of the centre line, the block
    #: starts squeezing.
    LB_CENTRAL_BAND_M = 12.0
    #: Fraction of the half-width lost at dead centre. 0.30 puts the back four
    #: at ~28 m of block width with the ball central, against ~40 m when it is
    #: wide: narrow enough to close the middle, wide enough that the fullbacks
    #: can still defend the touchline.
    LB_CENTRAL_SQUEEZE = 0.30
    #: Steer weight toward the block target. High, because the point is that a
    #: low block holds a shape rather than drifting between generic anchors —
    #: but still a steer, so the engine's own shaping layers keep priority.
    LB_ALPHA = 0.85

    def low_block_target(
        self, player_name: str, ball_x: float, ball_y: float,
        attacks_right: bool,
    ) -> Optional[Tuple[float, float, float]]:
        """The shape a LOW BLOCK holds: three lines, narrow, deep, front two high.

        Returns ``(alpha, tx, ty)`` to steer this player's off-ball target
        toward, or ``None`` if he is not part of the block.

        Why this exists
        ---------------
        Measured against the shape a real low block holds
        (``scripts/physics/probe_low_block.py``), PLOFA's ``park_the_bus`` was
        recognisably low but wrong in three specific ways:

        ==========================  ========  ==================
        metric (metres from own goal)  PLOFA   real low block
        ==========================  ========  ==================
        back four                       42.5    15-20
        back four -> midfield gap        0.7    10-15
        block width, ball central        50.7   narrows
        ==========================  ========  ==================

        The third row is the one that matters. A low block works by denying
        the middle: the block is compact, so there is nothing to play into
        centrally, and possession is forced wide where it is easier to defend.
        PLOFA's block sat 50.7 m wide on a 68 m pitch with the ball central —
        both half-spaces open. It was not compact; it was spread.

        The cause is visible in the generic shape target the off-ball machine
        builds for every team::

            tx = ax + (ball_x - ax) * 0.14
            ty = ay + (ball_y - ay) * 0.07

        A 7% lateral response means the block barely moves when the ball moves
        sideways, and the fullbacks hold their touchline channel regardless. That
        is the right default for most teams and the wrong one here.

        The three lines
        ---------------
        Depths are measured from own goal so they read the same whichever way
        the team attacks. The front pair is deliberately placed HIGH, ~16 m
        ahead of the midfield: the post's point is that the block is also a
        platform, and a deep regain needs an outlet immediately.

        Width narrows when the ball is central
        --------------------------------------
        ``NARROW_WHEN_CENTRAL_M`` is the half-width the block collapses to
        around the spine, and the reduction is applied to the *whole unit*
        rather than to individuals — which is what "all eleven defend as one
        unit" means geometrically.
        """
        state = self.states.get(player_name)
        if state is None:
            return None
        pos = state.position
        if pos == "GK":
            return None

        depth = self._low_block_depth(pos)
        if depth is None:
            return None
        base_depth, half_width = depth

        # Lateral centre: the unit shifts toward the ball side, but only
        # partially. A block that tracks the ball exactly has been stretched
        # by it; a block that does not move at all is not compact either.
        rel_ball_y = ball_y if attacks_right else (self.PITCH_Y_MAX - ball_y)
        centre = self.PITCH_Y_MID + (rel_ball_y - self.PITCH_Y_MID) * self.LB_SHIFT

        # ...and it narrows around the spine, which is the whole point.
        off_spine = abs(rel_ball_y - self.PITCH_Y_MID)
        if off_spine < self.LB_CENTRAL_BAND_M:
            squeeze = 1.0 - self.LB_CENTRAL_SQUEEZE * (
                1.0 - off_spine / self.LB_CENTRAL_BAND_M)
            half_width = half_width * squeeze

        # Keep the player on his own side of the block's centre; without this
        # the two fullbacks cross over and the block inverts.
        sign = 1.0 if state.current_y >= centre else -1.0
        ty = centre + sign * half_width
        ty = max(0.0, min(self.PITCH_Y_MAX, ty))

        depth_from_own = base_depth + self.LB_FRONT_DROP * (
            off_spine / (self.PITCH_Y_MAX * 0.5))
        tx = (self.PITCH_X_MAX - depth_from_own if attacks_right
              else depth_from_own)

        return (self.LB_ALPHA, tx, ty)

    def _low_block_depth(self, position: str) -> Optional[Tuple[float, float]]:
        """(depth from own goal, lateral half-width) for a role, or None."""
        table = {
            # back four — deepest line, widest, still inside own third
            "CB": (self.LB_BACK_DEPTH, self.LB_BACK_HALF_WIDTH),
            "LB": (self.LB_BACK_DEPTH, self.LB_BACK_HALF_WIDTH),
            "RB": (self.LB_BACK_DEPTH, self.LB_BACK_HALF_WIDTH),
            "LWB": (self.LB_BACK_DEPTH, self.LB_BACK_HALF_WIDTH),
            "RWB": (self.LB_BACK_DEPTH, self.LB_BACK_HALF_WIDTH),
            # midfield line — a genuine separate line, not level with the back four
            "CDM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "CM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "CAM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "LCM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "RCM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "LM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "RM": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            # wide forwards sit with the midfield line, not high and wide
            "LW": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            "RW": (self.LB_MID_DEPTH, self.LB_MID_HALF_WIDTH),
            # front pair — HIGH, and the outlet on a regain
            "ST": (self.LB_FRONT_DEPTH, self.LB_FRONT_HALF_WIDTH),
            "CF": (self.LB_FRONT_DEPTH, self.LB_FRONT_HALF_WIDTH),
        }
        return table.get(position)

    # ── DEFENSIVE BLOCKS ───────────────────────────────────────────
    # A defensive block is a SHAPE, not a press intensity.
    # `pressing_profiles` models how hard a team presses; these model where
    # its three lines stand, how narrow the unit gets, and how big the gaps
    # between the lines are.
    #
    # THREE PRESETS, ONE DISCIPLINE. The rule is the same at every height:
    #
    #     if you do not have the ball, do not leave a space you can be
    #     put into.
    #
    # What changes is only the AREA of the pitch the shape occupies, which is
    # the entire difference between a low block and a high one.
    #
    #            back   mid  front | backHW midHW frontHW | shift squeeze
    BLOCK_PRESETS: Dict[str, Tuple[float, ...]] = {
        "low":  (19.0, 31.0, 37.0,  20.0, 16.0, 11.0,  0.42, 0.30),
        "mid":  (30.0, 42.0, 48.0,  22.0, 18.0, 13.0,  0.45, 0.24),
        "high": (42.0, 54.0, 60.0,  24.0, 20.0, 15.0,  0.50, 0.18),
    }
    #: An unfamiliar block height degrades to the middle rather than crashing.
    BLOCK_DEFAULT = "mid"

    #: How much further the block sinks the wider the ball is: 4 m at the
    #: extreme flank, because a wide ball drags the line across and back.
    BLOCK_FRONT_DROP = 4.0
    #: Half-width of the "spine". Within this of the centre line the block
    #: starts squeezing — the mechanism by which the middle is denied.
    BLOCK_CENTRAL_BAND_M = 12.0
    #: Steer weight toward the block target. High, because the point is that a
    #: team HOLDS a shape rather than drifting between generic anchors — but
    #: still a steer, so the engine's own shaping layers keep priority.
    BLOCK_ALPHA = 0.85

    def defensive_block_target(
        self, player_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, preset: str = BLOCK_DEFAULT,
    ) -> Optional[Tuple[float, float, float]]:
        """The shape this team holds while defending.

        Returns ``(alpha, tx, ty)`` for this player's off-ball target, or
        ``None`` if he is not part of the block.

        Three lines, and the gaps between them are the whole point
        --------------------------------------------------------
        A block denies the middle because the unit is compact, so there is
        nothing to play into centrally and possession is forced wide. Measured
        against a real low block (``scripts/physics/probe_low_block.py``),
        PLOFA's ``park_the_bus`` sat 49.6 m wide on a 68 m pitch with the ball
        central — spread, not compact, with both half-spaces open.

        But compactness alone is not enough, and this is the part a naive
        "get closer together" fix gets wrong. Three lines held at ONE depth are
        not compact, they are a queue: a flat wall with a pocket on each
        shoulder and no pressure on the ball carrier. ``back_d``, ``mid_d`` and
        ``front_d`` are therefore separate depths, and the difference between
        them is deliberate:

          * back four behind midfield  — the pass has to travel *through* the
            midfield to reach a defender, so the receiver is never free;
          * midfield behind the front two — the front pair can screen the
            pass without dropping into their own defender's shadow.

        And the front two must be genuinely AHEAD. That is the counter-attack:
        a deep regain needs an outlet immediately, and a block whose front pair
        are level with its back four has none. The same three numbers serve
        both halves of the problem, which is why they cannot be tuned apart.
        """
        state = self.states.get(player_name)
        if state is None:
            return None
        pos = state.position
        if pos in ("GK", "GKC"):
            return None

        p = (self.BLOCK_PRESETS.get(preset)
             or self.BLOCK_PRESETS[self.BLOCK_DEFAULT])
        (back_d, mid_d, front_d, back_hw, mid_hw, front_hw,
         shift, squeeze) = p

        if pos in ("CB", "LB", "RB", "LWB", "RWB", "FB"):
            base_depth, half_width = back_d, back_hw
        elif pos in ("ST", "CF", "SS"):
            base_depth, half_width = front_d, front_hw
        else:
            base_depth, half_width = mid_d, mid_hw

        # Lateral centre: the unit shifts toward the ball side, but only
        # partially. A block that tracks the ball exactly has been stretched by
        # it; one that does not move at all is not compact.
        rel_ball_y = ball_y if attacks_right else (self.PITCH_Y_MAX - ball_y)
        centre = self.PITCH_Y_MID + (rel_ball_y - self.PITCH_Y_MID) * shift

        off_spine = abs(rel_ball_y - self.PITCH_Y_MID)
        if off_spine < self.BLOCK_CENTRAL_BAND_M:
            hw = half_width * (1.0 - squeeze * (
                1.0 - off_spine / self.BLOCK_CENTRAL_BAND_M))
        else:
            hw = half_width

        # Each player stays on his own side of the block's centre, or the two
        # fullbacks cross over and the block inverts.
        sign = 1.0 if state.current_y >= centre else -1.0
        ty = max(0.0, min(self.PITCH_Y_MAX, centre + sign * hw))

        depth = base_depth + self.BLOCK_FRONT_DROP * (
            off_spine / (self.PITCH_Y_MAX * 0.5))
        tx = (self.PITCH_X_MAX - depth if attacks_right else depth)

        return (self.BLOCK_ALPHA, tx, ty)

    def wide_stretch_blend(
        self, player_name: str, ball_y: float,
    ) -> float:
        """Checkpoint 35 — PITCH-STRETCH RULE: the blend weight a wide role's
        off-ball shape target is steered toward its touchline channel.

        Returns W (0..1) that the continuous off-ball machine
        (match_engine._offball_move_player) applies to the target y:
            `ty = ty + (home_y - ty) * W`,
        scaled by the live ball's spine weight so the stretch duty engages
        exactly when the middle is packed:
            - ball wide / already on a flank  -> 0  (no stretch needed)
            - ball on the 24-44m spine        -> ~0.30 toward the line/tick
        Steering the TARGET (not the position) keeps the actual movement
        pace-capped by the jog integrator, so the player physically re-asserts
        the line without teleporting. Non-wide roles return 0.
        """
        state = self.states.get(player_name)
        if state is None or state.position not in ("LB", "RB", "LW", "RW"):
            return 0.0
        return STRETCH_TARGET_BOOST * pitch_spine_weight(ball_y)

    def midfielder_triangle_support(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Checkpoint 36 — TRIANGLE SUPPORT RULE: the off-ball TARGET steer a
        midfielder should take to complete a passing triangle.

        Returns ``{name: (alpha, tx, ty)}`` for the team's CDM/CM/CAM — alpha
        in (0..1] applied by match_engine._offball_move_player to the shape
        target (`tx += (trix - tx)*alpha`, same pattern as the pitch-stretch),
        or an empty dict when the team is out of possession (their movement
        belongs to the defensive block / press, not triangles).

        Geometry:
          - WIDE ball (|ball_y-34| > TRI_WIDE_BAND) on a flank → the NEAR-SIDE
            CM commits toward a half-space support socket off the wide cluster
            (winger + full-back triangle); the far-side CM takes a subtle
            balance shift; the CDM pivot slides toward the ball side.
          - CENTRAL ball → the CM pair spreads either side of the ball with
            the CDM dropping as the pivot behind — the central triangle.

        Sides are formation-ordered (the 4-3-3's first CM is the left man,
        the second the right) so the rule is deterministic. The socket y is
        capped inside the pitch (half-space, never the touchline — CMs are
        links, not width providers). Steering the TARGET keeps the 10 Hz
        integrator's pace-capping intact.
        """
        if not has_ball or ball_x is None or ball_y is None:
            return {}
        roster = self.team_rosters.get(team_name, []) or []
        mid_names = [
            n for n in roster
            if self.states.get(n)
            and self.states[n].position in TRI_MIDFIELD_ROLES
        ]
        if not mid_names:
            return {}
        # Deterministic left/right split for repeated central roles, by lineup
        # order: the first CM/CAM is the left man, the second the right.
        sides = {}
        cm_seq = 0
        for n in mid_names:
            pos = self.states[n].position
            if pos == "CDM":
                sides[n] = 0
            else:
                sides[n] = -1 if cm_seq % 2 == 0 else 1
                cm_seq += 1
        ball_side = -1 if ball_y < 34.0 else (1 if ball_y > 34.0 else 0)
        wide = abs(ball_y - 34.0) > TRI_WIDE_BAND
        own_goal_x = 0.0 if attacks_right else 105.0
        out = {}
        for n in mid_names:
            st = self.states[n]
            s = sides.get(n, 0)
            if wide and ball_side != 0:
                if s == ball_side:
                    alpha = TRI_NEAR_ALPHA
                    sy = 34.0 + ball_side * min(
                        0.60 * abs(ball_y - 34.0), TRI_HALF_SPACE_MAX)
                elif s == 0:
                    alpha = TRI_PIVOT_ALPHA
                    sy = 34.0 + ball_side * min(
                        0.40 * abs(ball_y - 34.0), TRI_HALF_SPACE_MAX)
                else:
                    alpha = TRI_FAR_ALPHA
                    sy = st.home_y - ball_side * 3.0
                sx = ball_x + (own_goal_x - ball_x) * TRI_SUPPORT_BACK
            else:
                # Central ball -> central triangle: CM pair split around the
                # ball, CDM pivot drops behind it.
                alpha = TRI_CENTRAL_ALPHA
                sy = ball_y + s * TRI_CENTRAL_CM_OFFSET
                depth = 0.18 if st.position == "CDM" else 0.10
                sx = ball_x + (own_goal_x - ball_x) * depth
            out[n] = (alpha, sx, sy)
        return out

    def backline_build_up_support(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Checkpoint 37 — BACK-LINE BUILD-UP DROP: the off-ball TARGET steer
        that makes the back line offer a SHORT outlet under our own build-up.

        Returns ``{name: (alpha, tx, ty)}`` — same contract as
        ``midfielder_triangle_support`` (alpha in 0..1, applied by
        match_engine._offball_move_player as a target steer) — or empty when:
          - the team is out of possession (defensive block owns the back line),
          - the ball is past the halfway mark (progression owns the shape), or
          - no centre-back is alive in the spatial state.

        Geometry: the BALL-SIDE CB (nearest in y to the ball) drops toward a
        goal-side socket ~30% of the way back from ball to goal, nudged toward
        the spine — leaving one CB holding the line and the far side wide. The
        strength ramps up through the own third and fades to zero by midfield,
        and the socket is floored at BACKLINE_DROP_MIN_NX so the dropper never
        piles onto the goal line. Steering the TARGET (not the position) keeps
        the actual travel pace-capped by the jog integrator.
        """
        if not has_ball or ball_x is None or ball_y is None:
            return {}
        own_goal_x = 0.0 if attacks_right else 105.0
        nx = ball_x if attacks_right else 105.0 - ball_x
        if nx <= 1.0 or nx >= BACKLINE_DROP_MAX_NX:
            return {}
        strength = 1.0
        if nx > BACKLINE_DROP_PEAK_NX:
            denom = (BACKLINE_DROP_MAX_NX - BACKLINE_DROP_PEAK_NX) or 1.0
            strength = max(0.0, (BACKLINE_DROP_MAX_NX - nx) / denom)
        if strength <= 0.05:
            return {}
        cbs = [
            n for n in self.team_rosters.get(team_name, [])
            if (st := self.states.get(n)) is not None
            and st.position == "CB" and st.current_x is not None
        ]
        if not cbs:
            return {}
        ball_side = 1.0 if ball_y >= 34.0 else -1.0
        dropper, best = None, 1e9
        for n in cbs:
            sto = self.states[n]
            d = abs(sto.current_y - ball_y)
            if d < best:
                best, dropper = d, n
        if dropper is None:
            return {}
        sx = ball_x + (own_goal_x - ball_x) * BACKLINE_DROP_DEPTH
        sx_rel = sx if attacks_right else 105.0 - sx
        if sx_rel < BACKLINE_DROP_MIN_NX:
            sx = own_goal_x + BACKLINE_DROP_MIN_NX * (1.0 if attacks_right else -1.0)
            sx = max(0.0, min(105.0, sx))
        sy = ball_y + (34.0 - ball_y) * BACKLINE_DROP_CENTER
        alpha = BACKLINE_DROP_ALPHA * strength
        return {dropper: (alpha, sx, sy)}

    # Checkpoint 38 — PRESSURE-AWARE BACK-LINE SPREAD. When the team owns the
    # ball inside the build-up band AND the ball itself is being pressed, the
    # back four should not compact toward the ball (the default 10Hz steer) —
    # they should SPREAD: a counter-compaction push (each player's y away from
    # the ball's y) plus the far-side CB deepening to split the CH pair, so
    # the carrier always has a goal-side escape man that is ALSO a separate
    # passing lane. Counter to ball-compaction clustering shown by the spread
    # probe (15% of pressed build-up ticks had two back-line players <4m).
    BACKLINE_SPREAD_MAX_NX: float = 55.0   # engage only inside ~the halfway mark
    BACKLINE_SPREAD_MIN_NX: float = 18.0   # never spread into the six-yard wall
    BACKLINE_SPREAD_PRESS_M: float = 7.0   # nearest opponent to the ball <= this
    BACKLINE_SPREAD_ALPHA: float = 0.40    # steer weight at full pressure
    BACKLINE_SPREAD_Y_PUSH: float = 0.18   # counter-compaction: y away from ball
    BACKLINE_SPREAD_CB_FAR: float = 0.6    # far CB depth = n*0.6 (clamped 24-30)

    def backline_spread_pressure(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Checkpoint 38 — SPREAD THE PITCH UNDER PRESSURE.

        Returns ``{name: (alpha, tx, ty)}`` (same contract as CK37's
        backline_build_up_support) or {} when the situation is not on:
          - team out of possession,
          - ball beyond the build-up band (halfway mark),
          - nearest opponent to the BALL further than BACKLINE_SPREAD_PRESS_M
            (no live press ⇒ no forced spread).
        Targets: every back-line player's y is pushed AWAY from the ball's y
        (countering the default 10Hz ball-compaction steering that sucks the
        line into the press) but staying within reach (no touchline-pinning);
        the far-side CB additionally deepens to ~24-30m from his own goal so
        the two CHs split into a short (ball-side drop-in) + deep/wide pair —
        two separate escape lanes. TARGET steer only: the jog integrator keeps
        the actual travel pace-capped.
        """
        if not has_ball or ball_x is None or ball_y is None:
            return {}
        own_goal_x = 0.0 if attacks_right else 105.0
        nx = ball_x if attacks_right else 105.0 - ball_x
        if nx < self.BACKLINE_SPREAD_MIN_NX or nx > self.BACKLINE_SPREAD_MAX_NX:
            return {}
        other = next((t for t in self.team_rosters if t != team_name), None)
        if other is None:
            return {}
        press_d = None
        for n in self.team_rosters.get(other, []):
            st = self.states.get(n)
            if st is None or st.position == "GK":
                continue
            d = math.hypot(st.current_x - ball_x, st.current_y - ball_y)
            if press_d is None or d < press_d:
                press_d = d
        if press_d is None or press_d > self.BACKLINE_SPREAD_PRESS_M:
            return {}
        weight = (self.BACKLINE_SPREAD_PRESS_M - press_d) / self.BACKLINE_SPREAD_PRESS_M
        nx_str = 0.5 + 0.5 * max(0.0, 1.0 - (nx - 18.0) / (55.0 - 18.0))
        w = weight * nx_str
        if w <= 0.05:
            return {}
        alpha = self.BACKLINE_SPREAD_ALPHA * w
        sign = 1.0 if attacks_right else -1.0
        nx_far = min(30.0, max(24.0, nx * self.BACKLINE_SPREAD_CB_FAR))
        out = {}
        for n in self.team_rosters.get(team_name, []):
            st = self.states.get(n)
            if st is None or st.position not in ("CB", "LB", "RB"):
                continue
            if abs(st.home_y - ball_y) < 2.5:
                continue    # already on the ball's channel — nothing to pull him
            gry = st.home_y + (st.home_y - ball_y) * self.BACKLINE_SPREAD_Y_PUSH
            gry = max(1.0, min(67.0, gry))
            gx = st.home_x
            # far-side CB also deepens to ~24-30m so the CH pair SPLITS into a
            # short (ball-side drop-in) + deep/wide second outlet.
            if st.position == "CB" and (st.home_y - 34.0) * (ball_y - 34.0) < 0:
                gx = own_goal_x + nx_far * sign
                gx = max(0.0, min(105.0, gx))
            out[n] = (alpha, gx, gry)
        return out

    # LIVE-TICK SPACING GUARD (Checkpoint 38): the per-minute graph-relaxation
    # repel is too slow to stop two runners stacking at 10Hz inside one build-up
    # sequence. This is a TARGET redirect applied in match_engine._offball_move_
    # player right before the jog integrator: any same-team teammate currently
    # within LIVE_SEP_MIN redirects the runner's TARGET away from him (his own
    # target is untouched, so there is no oscillation).
    LIVE_SEP_MIN: float = 4.5
    LIVE_SEP_PUSH: float = 0.45

    def live_spacing_redirect(
        self, team_name: str, cx: float, cy: float, tx: float, ty: float,
    ) -> Tuple[float, float]:
        """Nudge a live off-ball target away from any teammate currently
        clustering on the runner."""
        best_d, p = self.LIVE_SEP_MIN, 1.0
        newx, newy = tx, ty
        for n in self.team_rosters.get(team_name, []):
            st = self.states.get(n)
            if st is None or (st.current_x is None or st.current_y is None):
                continue
            d = math.hypot(st.current_x - cx, st.current_y - cy)
            if d < best_d:
                if d < 1e-6:
                    best_d, p = d, 1.0
                    continue
                push = (best_d - d) / d * self.LIVE_SEP_PUSH
                newx += (cx - st.current_x) * push
                newy += (cy - st.current_y) * push
        return max(0.0, min(105.0, newx)), max(0.0, min(68.0, newy))

    # ── REST DEFENCE (positional play) ────────────────────────────────
    #
    # The last of the four superiority types with no representation here, and
    # the only one that is a CONSTRAINT rather than a preference.
    #
    # "Rest defence" is not a weight and must not be implemented as one. A
    # weight says "prefer holding a rest position"; a constraint says "you may
    # not leave the pitch entirely committed", which is a different claim, and
    # the same distinction the defensive-action brain drew when feasibility was
    # made a physics grip instead of a bonus (see AGENTS.md). Nothing in the
    # existing shape chain expresses it: every rule here is a *preference*
    # toward a socket, so a team whose preferences happen to all point upfield
    # legally ends up with ten men ahead of the ball and no way back.
    #
    # The invariant: WHILE IN POSSESSION, at least REST_DEFENCE_MIN_BEHIND
    # outfield players remain BEHIND the ball. Not a cap on how many may be
    # ahead — that would be wrong, because in the final third most of the team
    # SHOULD be ahead of the ball, and capping it would fight the box crash and
    # the striker. Only the degenerate case, the whole team beyond the ball, is
    # unreachable.
    #
    # Deliberately a BACKSTOP, not a constant force. In a normal shape the two
    # centre-backs and the pivot are behind the ball and this returns None, so
    # the rule costs nothing and perturbs nothing; it binds only when the shape
    # has genuinely broken. A rule that acted every tick would be a second
    # opinion fighting eleven tuned ones.
    #
    # Returns the name of the ONE player who must be held back, or None. The
    # shallowest player ahead of the ball is chosen because holding him costs
    # the attack the least — he is the closest to being a rest defender already.
    # The caller resolves this ONCE PER TICK (positions move as the tick
    # integrates, so resolving it per player could pick a different victim
    # mid-tick and pull back two men).
    REST_DEFENCE_ENABLED: bool = True
    REST_DEFENCE_MIN_BEHIND: int = 2
    REST_DEFENCE_TOUCH_M: float = 1.0   # level with the ball counts as behind

    def rest_defence_violator(
        self, team_name: str, ball_x: Optional[float], has_ball: bool,
        attacks_right: bool = True,
    ) -> Optional[str]:
        """Name of the outfield player who must be held behind the ball, or None.

        None means the invariant already holds (or does not apply). Kept
        separate from the clamp so the decision — which is a TEAM judgement —
        is made once per tick, and the per-player effect is a pure geometric
        clamp with no knowledge of teammates.
        """
        if not self.REST_DEFENCE_ENABLED or not has_ball:
            return None
        if ball_x is None:
            return None
        ball_nx = ball_x if attacks_right else PITCH_X - ball_x
        behind: List[str] = []
        ahead: List[Tuple[float, str]] = []
        for n in self.team_rosters.get(team_name, []):
            st = self.states.get(n)
            if st is None or getattr(st, "position", "") == "GK":
                continue
            if st.current_x is None:
                continue
            nx = st.current_x if attacks_right else PITCH_X - st.current_x
            if nx < ball_nx - self.REST_DEFENCE_TOUCH_M:
                behind.append(n)
            else:
                ahead.append((nx, n))
        if len(behind) >= self.REST_DEFENCE_MIN_BEHIND:
            return None
        if not ahead:
            return None
        ahead.sort()   # shallowest first: least advanced = cheapest to hold
        return ahead[0][1]

    REST_DEFENCE_GAP_M: float = 12.0   # metres behind the ball he is pinned

    def rest_defence_clamp(
        self, tx: float, ty: float, ball_x: float, attacks_right: bool,
    ) -> Tuple[float, float]:
        """Pin a target behind the ball. Pure depth clamp; y is left alone.

        Leaving y untouched is the point. The constraint is about DEPTH — "a
        body between the ball and our goal" — so re-stamping his lateral
        position would be a second, unrequested opinion about shape, and would
        collapse the width that CK35 spends its life re-asserting.
        """
        nx = tx if attacks_right else PITCH_X - tx
        ball_nx = ball_x if attacks_right else PITCH_X - ball_x
        rest_nx = ball_nx - self.REST_DEFENCE_GAP_M
        if nx <= rest_nx:
            return tx, ty
        clamped = rest_nx if attacks_right else PITCH_X - rest_nx
        return max(0.0, min(PITCH_X, clamped)), ty

    # ── STRIKER RUNS, WIRED LIVE ──────────────────────────────────────
    #
    # striker_behavior.py was fully built and never reachable from a match:
    # its only consumer, _striker_run_step, is called from drift_minute, and
    # MatchEngine never calls drift_minute. So the three striker principles the
    # positional-play book is most specific about — run in behind, drop to
    # link, attack a post channel — had no effect on a single match, and the
    # striker's only live contribution was the false-nine RECEIVE bonus in
    # attacking_matrix (a pass-value term, not movement).
    #
    # What it needs from the live loop that it did not have:
    #   * a TARGET, not a position write. _striker_run_step assigns
    #     state.current_x directly, which is a per-minute net delta and would
    #     fight the 10 Hz integrator if driven at tick rate. Here it is a
    #     target steer, exactly like CK36/CK37/CK38, and the integrator
    #     pace-caps the travel.
    #   * a decision cadence slower than the tick. decide_run draws from
    #     random; at 10 Hz that is a coin flip ten times a second, the run mode
    #     would flicker, and the RNG stream would be consumed ~10x per player
    #     per second. The caller caches per (team, minute) — the same
    #     granularity _striker_run_step used via last_active_minute.
    STRIKER_RUNS_LIVE: bool = True
    STRIKER_RUN_BLEND: float = 0.30   # target steer weight

    @staticmethod
    def _deterministic_rng(*parts) -> Callable[[], float]:
        """A counter-based [0,1) stream keyed on ``parts``.

        The run decision is made once a minute, but it still needs a coin flip,
        and ``random.random()`` is the wrong coin: every draw taken here shifts
        the football stream, and the project's own top open bug is that a match
        is not reproducible from ``random.seed`` (AGENTS.md). Adding a consumer
        makes that worse, so this layer takes none.

        crc32 rather than ``hash()`` because builtin string hashing is salted
        per process - the same rule the world layer follows for exactly this
        reason (``world/squads.py``, ``world/proof.py``). The counter keeps
        successive calls distinct, so a single decision can draw three times
        and still be reproducible.
        """
        key = "|".join(str(p) for p in parts)
        counter = [0]

        def _next() -> float:
            v = zlib.crc32(f"{key}#{counter[0]}".encode("utf-8"))
            counter[0] += 1
            return v / 4294967296.0

        return _next

    def _opponent_shape_shims(self, team_name: str) -> List:
        """Opponent players as the minimal duck-type StrikerBehaviorEngine reads.

        ``last_line_gap`` needs only ``.name`` and ``.position``; the live loop
        has spatial states, not PlayerProfile objects. Built here so the engine
        stays ignorant of which of the two it was handed.
        """
        other = next((t for t in self.team_rosters if t != team_name), None)
        if other is None:
            return []
        out = []
        for n in self.team_rosters.get(other, []):
            st = self.states.get(n)
            if st is None:
                continue
            out.append(_ShapeShim(n, getattr(st, "position", "")))
        return out

    def striker_run_targets(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
        stamina_pct: float = 100.0, minute: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float, float]]:
        """{name: (blend, tx, ty)} for each ST/CF committing to a run.

        Empty when out of possession (decide_run declines anyway) or when no
        ball, so a stale call cannot steer a striker at a remembered position.

        ``minute`` keys the decision's RNG so the flip is reproducible per
        (team, striker, minute) and consumes nothing from the global stream.

        Returns {name: (blend, tx, ty, mode)}. The MODE is the decision the
        engine actually made ("behind" / "hold" / "box") and used to be
        dropped here, leaving only a position delta downstream - which cannot
        tell an instructed in-behind run from a shape-compaction drift.
        """
        if not self.STRIKER_RUNS_LIVE or not has_ball:
            return {}
        if ball_x is None or ball_y is None:
            return {}
        defenders = self._opponent_shape_shims(team_name)
        out: Dict[str, Tuple[float, float, float, str]] = {}
        for name in self.team_rosters.get(team_name, []):
            st = self.states.get(name)
            if st is None or getattr(st, "position", "") not in ("ST", "CF"):
                continue
            prof = self.striker_registry.get(name)
            if prof is None:
                continue
            m = StrikerBehaviorEngine.decide_run(
                prof, st.current_x, st.current_y, attacks_right, ball_x, ball_y,
                in_possession=True, defenders=defenders, position_engine=self,
                anchor_y=st.home_y, stamina_pct=stamina_pct,
                rng=self._deterministic_rng(team_name, name, minute),
            )
            if m is None:
                continue
            if m == "behind":
                line = prof.offside_line_nx(attacks_right, defenders, self)
                target = prof.run_behind_target(attacks_right, st.home_y, line)
            elif m == "hold":
                target = prof.hold_up_target(ball_x, ball_y, attacks_right)
            else:   # "box"
                target = prof.box_arrival_target(ball_x, ball_y, attacks_right)
            tx, ty = target
            # Same pressure-bend the per-minute step used: a run into a
            # pressed pocket is abandoned for the safer anchor, so a striker
            # does not jog into three men every time the channel is nominally open.
            press = self._opponent_pressure_at(
                tx, ty, defenders, ball_x, ball_y, attacks_right)
            if press > 0.55:
                tx += (st.home_x - tx) * 0.35
                ty += (st.home_y - ty) * 0.30
            out[name] = (self.STRIKER_RUN_BLEND,
                         max(0.0, min(PITCH_X, tx)), max(0.0, min(PITCH_Y, ty)),
                         m)
        return out

    GK_BUILD_UP_LOW_NX: float = 22.0    # below this the ball is in the keeper's
                                        # dead-zone — hold near the line
    GK_BUILD_UP_ADVANCE_NX: float = 55.0  # beyond this the attacking-crash
                                          # GK step-up (opponent half) owns the shape
    GK_BUILD_UP_SOCKET: float = 24.0    # sweeper-keeper build-out post off his line
    GK_BUILD_UP_PULL: float = 0.25      # per-tick steer weight (pace-capped jog)

    def gk_build_up_advance(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Checkpoint 37 — SWEEPER-KEEPER BUILD-OUT ADVANCE.

        The keeper is the free-man release for the back line, but only when he
        OFFERS HIMSELF. The old keeper read was purely reactive: he sat on his
        line until a midfield reset planted him at x≈28 (the record_touch clamp),
        at which point the carrier was already 30-40m gone — the "back-pass"
        became a 35m+ diagonally-risky heave, so the 30m back-bump gate (new in
        Checkpoint 37) starved him of feeds entirely.

        Modern GK-as-sweeper build-out (Ederson/Neuer/ter Stegen): while the
        team builds from inside its own two-thirds, the keeper slides OUT to a
        ~24m socket so the CB/CDM/FB always has a genuine SHORT back bump (a
        10-30m ball to the keeper is a routine high-%, low-risk pass). He holds
        near the line when the ball is in his own six-yard dead zone, stands off
        when the opponent has the ball (defensive_block owns him there), and
        hands over to the existing opponent-half attack step-up beyond midfield.
        Returns (tx, ty) to steer toward, or None when no build-out is on.
        """
        if not has_ball or ball_x is None or ball_y is None:
            return None
        gk = next(
            (self.states[n] for n in self.team_rosters.get(team_name, [])
             if (st := self.states.get(n)) is not None
             and st.position == "GK"),
            None,
        )
        if gk is None:
            return None
        own_goal_x = 0.0 if attacks_right else 105.0
        sign = 1.0 if attacks_right else -1.0
        nx = ball_x if attacks_right else 105.0 - ball_x
        if nx < self.GK_BUILD_UP_LOW_NX:
            gx = own_goal_x + 12.0 * sign            # near the line, dead zone
        elif nx <= self.GK_BUILD_UP_ADVANCE_NX:
            gx = own_goal_x + self.GK_BUILD_UP_SOCKET * sign
        else:
            return None                              # opponent half — crash owns GK
        gy = 34.0 + (ball_y - 34.0) * 0.30           # slide light toward ball side
        gx = max(0.0, min(105.0, gx))
        gy = max(0.0, min(68.0, gy))
        return (gx, gy)

    def _opponent_pressure_at(
        self, x: float, y: float,
        opponents: Optional[List], ball_x: float, ball_y: float,
        attacks_right: bool,
    ) -> float:
        """
        Rough counter-press pressure at (x, y): inverse-distance sum of
        nearby opponents (and the loose ball). 0 (open lane) → ~1 (trap).
        Used by _wide_run_step to check a runner off a pressing trap.
        """
        if not opponents:
            return 0.0
        s = 0.0
        for o in opponents:
            oname = getattr(o, "name", None)
            if oname is None:
                continue
            ox, oy = self.get_position(oname)
            d = math.hypot(ox - x, oy - y)
            if d < 25.0:
                s += (25.0 - d) / 25.0
        if ball_x is not None and ball_y is not None:
            db = math.hypot(ball_x - x, ball_y - y)
            if db < 18.0:
                s += (18.0 - db) / 18.0 * 0.6
        return min(1.0, s / 2.5)

    # ── LIVE RUN TARGETS (wide / CM / CAM), and the striker layer above ──
    #
    # The three dormant run-steps decided real football — byline drive, cut
    # inside, box entry, overlap, underlap, tuck, drop to receive, carry, late
    # arrival, orbit, #10 pocket roam — and MatchEngine never called
    # drift_minute, so none of it happened in a match. What DID happen is the
    # passing half of positional play: attacking_matrix rewards a receiver in a
    # half-space and the triangle rule steers toward a passing socket. The ball
    # was offered the pocket and the pocket was empty. This is the movement half.
    OFFBALL_RUNS_LIVE: bool = True
    OFFBALL_RUN_BLEND_WIDE: float = 0.30
    OFFBALL_RUN_BLEND_MID: float = 0.28
    # The CAM's own dormant pull was 0.40 + an awareness bonus, and the pocket
    # roam is the least replaceable movement in the sim (he is the hub), so his
    # weight is used as-is rather than re-tuned down.

    def offball_run_targets(
        self, team_name: str, ball_x: Optional[float], ball_y: Optional[float],
        has_ball: bool, attacks_right: bool = True,
        opp_block=None, minute: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float, float, str]]:
        """{name: (blend, tx, ty, mode)} for every non-striker run, live.

        Merges the wide (incl. FULLBACK advance), CM and CAM decisions. The
        striker layer is a separate method so its Law-11 offside logic and its
        deterministic coin stay together; the caller merges the two.

        In-possession only: all three engines assume a team that has the ball,
        and steering a runner while defending is the shape layer's job.
        """
        if not self.OFFBALL_RUNS_LIVE or not has_ball:
            return {}
        if ball_x is None or ball_y is None:
            return {}
        out: Dict[str, Tuple[float, float, float, str]] = {}
        defenders = self._opponent_shape_shims(team_name)
        # Snapshot/restore so the engines' internal random.random() calls do
        # not advance the football stream. See the module note above.
        rng_state = random.getstate()
        try:
            for name, (mode, tx, ty) in self._wide_run_targets(
                    team_name, ball_x, ball_y, attacks_right, defenders).items():
                out[name] = (self.OFFBALL_RUN_BLEND_WIDE, tx, ty, mode)
            for name, (mode, tx, ty) in self._midfield_run_targets(
                    team_name, ball_x, ball_y, attacks_right, defenders,
                    opp_block).items():
                out[name] = (self.OFFBALL_RUN_BLEND_MID, tx, ty, mode)
            if minute is not None:
                for name, (weight, tx, ty) in self._cam_pocket_targets(
                        team_name, ball_x, ball_y, attacks_right, minute).items():
                    out[name] = (weight, tx, ty, "roam")
        finally:
            random.setstate(rng_state)
        # The MODE is carried through so the caller can record the DECISION,
        # not merely its positional effect. It used to be dropped here, which
        # is why the exporter could only guess at runs from geometry.
        return {
            name: (blend, max(0.0, min(PITCH_X, tx)), max(0.0, min(PITCH_Y, ty)), mode)
            for name, (blend, tx, ty, mode) in out.items()
        }

    def _wide_run_targets(
        self, team_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, opponent_players: Optional[List],
    ) -> Dict[str, Tuple[str, float, float]]:
        """{name: (mode, tx, ty)} for LW/RW/LB/RB runs - DECISION ONLY.

        Split out of _wide_run_step so the live 10 Hz loop can consume the same
        decisions as a TARGET STEER rather than as a position write. The step's
        write is a once-per-minute net delta; the live loop's own integrator
        does the travel and needs the decision available on its own cadence.

        This carries the FULLBACK advance modes as well as the winger's - they
        lived in the same method, so overlap / underlap / tuck were dead in a
        match for the same reason the winger's were.

        Deliberately has no `last_active_minute` guard: that guard exists to
        protect RECORDED TOUCH COORDINATES, which is the step's business. The
        live loop has a stronger equivalent already - it returns early for
        anyone in _on_ball_this_minute.
        """
        out: Dict[str, Tuple[str, float, float]] = {}
        if ball_x is None or ball_y is None:
            return out
        sign = 1.0 if attacks_right else -1.0
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("LW", "RW", "LB", "RB"):
                continue
            anchor_y = state.home_y
            pos = state.position
            target = None
            mode = None

            if pos in ("LW", "RW"):
                wp = self.winger_registry.get(name)
                if wp is not None:
                    x, y = state.current_x, state.current_y
                    if WingerBehaviorEngine.should_drive_byline(
                            wp, x, y, attacks_right,
                            defenders=opponent_players, position_engine=self,
                            anchor_y=anchor_y):
                        mode = "byline"
                        tx = min(max(x + sign * 14.0, 70.0), 98.0) if attacks_right \
                            else max(min(x + sign * 14.0, 35.0), 7.0)
                        ty = y + (anchor_y - y) * 0.40
                        target = (tx, ty)
                    elif WingerBehaviorEngine.should_cut_inside(
                            wp, x, y, attacks_right,
                            defenders=opponent_players, position_engine=self,
                            anchor_y=anchor_y):
                        mode = "cut"
                        tx = min(x + sign * 10.0, 92.0) if attacks_right \
                            else max(x + sign * 10.0, 13.0)
                        half_y = anchor_y + (CENTER_Y - anchor_y) * 0.55
                        target = (tx, half_y)
                    elif WingerBehaviorEngine.should_enter_box(
                            wp, x, y, attacks_right,
                            ball_on_opposite_flank=abs(ball_y - anchor_y) > 18.0):
                        mode = "box"
                        tx = min(max(x + sign * 6.0, 80.0), 96.0) if attacks_right \
                            else max(min(x + sign * 6.0, 25.0), 9.0)
                        ty = WingerBehaviorEngine.back_post_target_y(wp, ball_y)
                        target = (tx, ty)
            else:  # LB / RB
                fb = self.fullback_registry.get(name)
                if fb is not None:
                    tucking = (fb.tuck_instinct > 0.30
                               and fb.in_tuck_zone(ball_x, attacks_right))
                    may_advance = FullbackBehaviorEngine.should_advance(
                        fb, state.current_x, state.current_y, attacks_right,
                        ball_x, ball_y, in_possession=True, anchor_y=anchor_y,
                    )
                    if tucking or may_advance:
                        m = FullbackBehaviorEngine.choose_advance_mode(
                            fb, state.current_x, state.current_y, attacks_right,
                            ball_x, ball_y, defenders=opponent_players,
                            position_engine=self, anchor_y=anchor_y,
                        )
                        if m in ("overlap", "underlap", "tuck"):
                            mode = m
                            target = fb.advance_run_target(
                                state.current_x, state.current_y, attacks_right,
                                anchor_y=anchor_y, mode=m,
                            )

            if target is None:
                continue

            # ── SHAPE-AWARE: bend the run off a counter-press trap ──
            tx, ty = target
            press = self._opponent_pressure_at(
                tx, ty, opponent_players, ball_x, ball_y, attacks_right)
            if press > 0.55:
                tx = tx + (state.home_x - tx) * 0.35
                ty = ty + (anchor_y - ty) * 0.30
                target = (tx, ty)
            out[name] = (mode, tx, ty)
        return out

    def _wide_run_step(
        self,
        team_name: str, minute: int,
        ball_x: Optional[float], ball_y: Optional[float],
        attacks_right: bool, opponent_players: Optional[List],
    ) -> None:
        """
        Checkpoint 32b - CONTINUOUS, SHAPE-AWARE wide runs.

        For each winger/fullback NOT just involved this minute, decide a run
        target from the behavior engines (winger: byline drive / cut inside /
        box entry; fullback: overlap / underlap / tuck), then walk toward it at
        a pace-capped rate so a run unfolds as a trajectory rather than a jump.
        A target sitting under a counter-press trap is bent back toward
        home/width so the runner checks his run instead of sprinting in.

        Runs fire only on a genuine behavior-engine decision; with no run on,
        the cached target is cleared and the player settles via the normal
        pulls, so static circulation is untouched.

        The DECISION lives in _wide_run_targets; this method only walks.
        """
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("LW", "RW", "LB", "RB"):
                continue
            if state.last_active_minute == minute:
                continue  # touch coordinates are authoritative
            if ball_x is None or ball_y is None:
                state.run_mode = None
                continue
            entry = self._wide_run_targets(
                team_name, ball_x, ball_y, attacks_right, opponent_players
            ).get(name)
            if entry is None:
                state.run_mode = None
                continue
            mode, tx, ty = entry
            # - CONTINUITY: pace-capped travel toward cached target -
            state.run_target_x, state.run_target_y, state.run_mode = (
                tx, ty, mode)
            dx = tx - state.current_x
            dy = ty - state.current_y
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue
            step = min(dist, state.top_speed_mpm)
            state.current_x += dx / dist * step
            state.current_y += dy / dist * step

    # How far a midfielder may sit from the ball's side and still drop into the
    # central build-up pocket. The drop pocket is BETWEEN THE CENTRE-BACKS, so
    # dropping a far-side midfielder is a long lateral slide toward the ball -
    # it collapses exactly the width CK35 works to hold, and it broke
    # test_triangle_support ("Far CM should not drift toward the ball side").
    # Dropping is for the midfielder who is already on the ball's side, where
    # "come short and central" is a couple of metres, not fourteen.
    CM_DROP_SIDE_TOL_M: float = 18.0

    def _midfield_run_targets(
        self, team_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, opponent_players: Optional[List],
        opp_block=None, key: Optional[tuple] = None,
    ) -> Dict[str, Tuple[str, float, float]]:
        """{name: (mode, tx, ty)} for CM runs - DECISION ONLY.

        Split out of _midfield_run_step for the live 10 Hz loop, exactly as
        _wide_run_targets was. Without this the CM had no live run behaviour
        at all: drop-to-receive, carry forward, late box arrival and orbit all
        lived only under drift_minute, which MatchEngine never calls.

        `committed_carry` lives HERE, not in the write loop: it is a team-level
        rule (only one midfielder may vacate the pivot at a time) and the helper
        is what the live path calls exactly once per team per decision.
        """
        out: Dict[str, Tuple[str, float, float]] = {}
        if ball_x is None or ball_y is None:
            return out
        sign = 1.0 if attacks_right else -1.0
        committed_carry = False
        block_channels = getattr(opp_block, "channels", None) if opp_block else None
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position != "CM":
                continue
            prof = self.midfield_registry.get(name)
            if prof is None:
                continue
            anchor_y = state.home_y
            x, y = state.current_x, state.current_y
            target = None
            mode = None

            m = MidfielderBehaviorEngine.decide_run(
                prof, x, y, attacks_right, ball_x, ball_y,
                in_possession=True, defenders=opponent_players,
                position_engine=self, anchor_y=anchor_y,
                block_channels=block_channels,
            )
            # Coordination: only ONE midfielder commits a carry per tick so
            # the pivot isn't vacated by two runners at once.
            if m == "carry" and committed_carry:
                m = None
            # A midfielder more than CM_DROP_SIDE_TOL_M from the ball's side is
            # the FAR-SIDE player, and his only run is orbit. drop, carry and
            # late all steer toward the ball's side, so letting the far man take
            # any of them collapses the width CK35 exists to hold - which is what
            # the triangle suite caught (far CM 24.0 -> 26.3). A drop-only guard
            # did not fix it: `carry` was doing the pulling.
            _bside = 1.0 if ball_y >= CENTER_Y else -1.0
            _pside = 1.0 if y >= CENTER_Y else -1.0
            if _bside != _pside and abs(y - ball_y) > self.CM_DROP_SIDE_TOL_M:
                if m != "orbit":
                    m = None
            if m == "drop":
                target = prof.drop_target(attacks_right, anchor_y)
                mode = "drop"
            elif m == "carry":
                target = MidfielderBehaviorEngine.carry_target(
                    prof, x, y, attacks_right, anchor_y,
                    defenders=opponent_players, position_engine=self)
                mode = "carry"
                committed_carry = True
            elif m == "late":
                target = prof.box_arrival_target(ball_x, ball_y, attacks_right)
                mode = "late"
            elif m == "orbit":
                target = prof.orbit_target(
                    x, y, ball_x, ball_y, attacks_right, anchor_y,
                    block_channels=block_channels,
                    defenders=opponent_players, position_engine=self)
                mode = "orbit"

            if target is None:
                continue

            tx, ty = target
            press = self._opponent_pressure_at(
                tx, ty, opponent_players, ball_x, ball_y, attacks_right)
            if press > 0.55:
                tx = tx + (state.home_x - tx) * 0.35
                ty = ty + (anchor_y - ty) * 0.30
                target = (tx, ty)
            out[name] = (mode, tx, ty)
        return out

    def _midfield_run_step(
        self,
        team_name: str, minute: int,
        ball_x: Optional[float], ball_y: Optional[float],
        attacks_right: bool, opponent_players: Optional[List],
        opp_block=None,
    ) -> None:
        """
        Checkpoint 33 - CONTINUOUS, SHAPE-AWARE central-midfield runs (CM).

        CMs decide a run from the MidfielderBehaviorEngine: drop to receive
        (pivot), carry forward through the lines, a late box arrival, or drift
        to the far-side open channel (orbit). The target is cached and travelled
        at a pace-capped rate, bending off opponent pressure.

        CAMs are NOT handled here - their between-the-lines movement is owned by
        _cam_pocket_roam, to avoid double-moving them.

        The DECISION lives in _midfield_run_targets; this method only walks.
        """
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position != "CM":
                continue
            if state.last_active_minute == minute:
                continue
            if ball_x is None or ball_y is None:
                state.run_mode = None
                continue
            entry = self._midfield_run_targets(
                team_name, ball_x, ball_y, attacks_right, opponent_players,
                opp_block,
            ).get(name)
            if entry is None:
                state.run_mode = None
                continue
            mode, tx, ty = entry
            state.run_target_x, state.run_target_y, state.run_mode = (
                tx, ty, mode)
            dx = tx - state.current_x
            dy = ty - state.current_y
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue
            step = min(dist, state.top_speed_mpm)
            state.current_x += dx / dist * step
            state.current_y += dy / dist * step

    def _midfield_defensive_step(
        self,
        team_name: str, minute: int,
        ball_x: Optional[float], ball_y: Optional[float],
        attacks_right: bool, opponent_players: Optional[List],
    ) -> None:
        """
        Checkpoint 33c — CONTINUOUS, shape-aware OUT-OF-POSSESSION
        midfield movement (CM + CAM). Driven by the MidfielderBehaviorEngine's
        decide_defensive_role: a ball-winner steps onto the carrier
        (press_target), a holder drops to screen the block (recover_target),
        and everyone else holds the shape. Pace-capped by top_speed_mpm, and
        still bounded by the continuity clamp in the caller.
        """
        if ball_x is None or ball_y is None:
            return
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("CM", "CAM"):
                continue
            if state.last_active_minute == minute:
                continue
            prof = self.midfield_registry.get(name)
            if prof is None:
                continue
            x, y = state.current_x, state.current_y
            anchor_y = state.home_y
            role = MidfielderBehaviorEngine.decide_defensive_role(
                prof, x, y, attacks_right, ball_x, ball_y,
                opponent_players, self,
            )
            if role == "press":
                target = prof.press_target(x, y, opponent_players, self, attacks_right)
            elif role == "recover":
                target = prof.recover_target(attacks_right, anchor_y)
            else:
                target = None
            if target is None:
                continue
            tx, ty = target
            dx = tx - state.current_x
            dy = ty - state.current_y
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue
            step = min(dist, state.top_speed_mpm)
            state.current_x += dx / dist * step
            state.current_y += dy / dist * step

    def _striker_run_step(
        self,
        team_name: str, minute: int,
        ball_x: Optional[float], ball_y: Optional[float],
        attacks_right: bool, opponent_players: Optional[List],
    ) -> None:
        """
        Checkpoint 33 — CONTINUOUS, SHAPE-AWARE striker runs (ST/CF).

        Strikers decide a run from the StrikerBehaviorEngine: run in
        behind the last line, drop to link (hold-up), or attack a post
        channel in the box. Cached + pace-capped + pressure-bent, same as
        the other run steps. (Strikers are also nudged by _attacker_space_run;
        this adds the archetype-specific, decision-gated runs on top.)
        """
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in ("ST", "CF"):
                continue
            if state.last_active_minute == minute:
                continue
            if ball_x is None or ball_y is None:
                state.run_mode = None
                continue
            prof = self.striker_registry.get(name)
            if prof is None:
                continue
            anchor_y = state.home_y
            x, y = state.current_x, state.current_y
            m = StrikerBehaviorEngine.decide_run(
                prof, x, y, attacks_right, ball_x, ball_y,
                in_possession=True, defenders=opponent_players,
                position_engine=self, anchor_y=anchor_y,
            )
            target = None
            mode = None
            if m == "behind":
                target = prof.run_behind_target(attacks_right, anchor_y)
                mode = "behind"
            elif m == "hold":
                target = prof.hold_up_target(ball_x, ball_y, attacks_right)
                mode = "hold"
            elif m == "box":
                target = prof.box_arrival_target(ball_x, ball_y, attacks_right)
                mode = "box"

            if target is None:
                state.run_mode = None
                continue

            tx, ty = target
            press = self._opponent_pressure_at(
                tx, ty, opponent_players, ball_x, ball_y, attacks_right)
            if press > 0.55:
                tx = tx + (state.home_x - tx) * 0.35
                ty = ty + (anchor_y - ty) * 0.30
                target = (tx, ty)

            state.run_target_x, state.run_target_y, state.run_mode = (
                target[0], target[1], mode)
            dx = target[0] - state.current_x
            dy = target[1] - state.current_y
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue
            step = min(dist, state.top_speed_mpm)
            state.current_x += dx / dist * step
            state.current_y += dy / dist * step

    def _band_adjusted_anchor(
        self, state, hx: float, hy: float,
        ball_x: float, ball_y: float, attacks_right: bool,
    ):
        """
        Returns the anchor rescaled radially about the ball so its distance
        sits inside the role's calibrated band for the ball's attacking-
        relative third. Radial scaling preserves the existing line/channel
        direction; only ball proximity changes.
        """
        attacking_x = ball_x if attacks_right else PITCH_X - ball_x
        if attacking_x < 35.0:
            third = "defensive"
        elif attacking_x <= 70.0:
            third = "middle"
        else:
            third = "attacking"
        lo, hi = self.BALL_PROXIMITY_BANDS_BY_THIRD[third][state.position]

        # Only the near-side winger offers a direct support angle. The
        # opposite winger remains wide instead of collapsing onto the ball.
        if state.position in ("LW", "RW"):
            # Ball must be on same side of pitch as winger's home position
            ball_on_same_side = (
                (ball_y < 34.0 and state.home_y < 34.0) or  # Both on left
                (ball_y >= 34.0 and state.home_y >= 34.0)   # Both on right
            )
            if not ball_on_same_side:
                return hx, hy  # Don't adjust if ball is on opposite flank

        dx = hx - ball_x
        dy = hy - ball_y
        d = math.hypot(dx, dy)
        if lo <= d <= hi:
            return hx, hy
        if d < 1e-6:
            # Anchor coincides with the ball carrier: step goal-side of it
            # so the midfielder never collapses onto the ball.
            back = -1.0 if attacks_right else 1.0
            return max(4.0, min(101.0, ball_x + back * lo)), hy
        target_d = lo if d < lo else hi
        s = target_d / d
        ax = max(4.0, min(101.0, ball_x + dx * s))
        ay = max(2.0, min(66.0, ball_y + dy * s))
        return ax, ay

    def _enforce_ball_proximity_targets(
        self, team_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, start_positions: Dict[str, Tuple[float, float]],
        minute: int,
    ) -> None:
        """Steer toward the band without cancelling a physical run."""
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position not in self.BALL_PROXIMITY_BANDS_BY_THIRD["middle"]:
                continue
            if state.last_active_minute == minute:
                continue
            target_x, target_y = self._band_adjusted_anchor(
                state, state.home_x, state.home_y,
                ball_x, ball_y, attacks_right,
            )
            start = start_positions.get(name)
            if start is None:
                continue
            used = math.hypot(
                state.current_x - start[0], state.current_y - start[1]
            )
            remaining = max(0.0, state.top_speed_mpm - used)
            correction_x = target_x - state.current_x
            correction_y = target_y - state.current_y
            correction = math.hypot(correction_x, correction_y)
            if correction <= 1e-6 or remaining <= 1e-6:
                continue
            step = min(correction, remaining)
            state.current_x += correction_x / correction * step
            state.current_y += correction_y / correction * step

    # Checkpoint 31b — roaming patterns the #10 cycles through while his
    # team has the ball: (metres ahead of ball, lateral offset from ball).
    # Runs beyond the carrier, sits on his shoulder in the pocket, drops
    # deep to link play — the alternating excursions are what generate a
    # real #10's ground coverage, not a fixed radius around the ball.
    CAM_ROAM_PATTERNS = (
        (22.0, -18.0),   # run beyond, left half-space
        (13.0, 12.0),    # pocket on his shoulder, right
        (-6.0, -8.0),    # drop deep to link, central-left
        (18.0, 16.0),    # third-man run, right half-space
        (-10.0, 14.0),   # deep drop, wide right (rotate out)
    )

    def _cam_pocket_targets(
        self, team_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, minute: int,
    ) -> Dict[str, Tuple[float, float, float]]:
        """{name: (weight, tx, ty)} for CAM pocket roaming - DECISION ONLY.

        Split out of _cam_pocket_roam for the live 10 Hz loop, as with the wide
        and CM runs. The weight is returned rather than applied: the roam uses a
        fractional pull, not pace-capped travel, so the caller decides how hard
        to pull (and the dormant path still halves it on a touched minute).

        This is the piece that gives a #10 actual between-the-lines movement.
        Without it live, the CAM sat on his static home marker while
        attacking_matrix kept PREFERRING to pass to the half-space - the ball
        was offered a pocket nobody occupied.
        """
        out: Dict[str, Tuple[float, float, float]] = {}
        if ball_x is None or ball_y is None:
            return out
        dir_x = 1.0 if attacks_right else -1.0
        patterns = self.CAM_ROAM_PATTERNS
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position != "CAM":
                continue
            seed_v = sum((i + 1) * ord(c) for i, c in enumerate(name))
            dx_ahead, dy = patterns[(minute + seed_v) % len(patterns)]
            tx = max(25.0, min(94.0, ball_x + dir_x * dx_ahead))
            ty = max(6.0, min(62.0, ball_y + dy))
            # Checkpoint 33 — registry-aware CAM bias. The base roam target
            # is now blended with the engine's own pocket geometry
            # (cam_pocket_target), so the profile's pocket_target() is the
            # authoritative seam while the roaming offset keeps minute-to-
            # minute variation. A shadow striker is pulled further forward
            # into the goal-side seam; a classic ten holds the pocket.
            cam_prof = self.midfield_registry.get(name)
            if cam_prof is not None:
                ptx, pty = MidfielderBehaviorEngine.cam_pocket_target(
                    cam_prof, ball_x, ball_y, attacks_right)
                tx = tx * 0.45 + ptx * 0.55
                ty = ty * 0.45 + pty * 0.55
            awareness_bonus = max(0.0, min(                0.12, (state.geometric_awareness - 50.0) / 350.0))
            out[name] = (0.40 + awareness_bonus, tx, ty)
        return out

    def _cam_pocket_roam(
        self, team_name: str, ball_x: float, ball_y: float,
        attacks_right: bool, minute: int,
    ):
        """
        Checkpoint 31b - #10 pocket roaming in possession.

        A CAM doesn't hold a fixed post: he floats in the pockets between the
        opponent's midfield and defensive lines, one passing tier (~13m) ahead
        of the carrier, switching half-spaces as the ball moves.

        A #10 who just released the ball starts his next run IMMEDIATELY; the
        pull is halved rather than skipped on a touched minute, so recorded
        touch coordinates stay dominant while the after-release movement is
        still modelled.

        Targets are deterministic per (minute, player). The DECISION lives in
        _cam_pocket_targets; this method only applies it.
        """
        decided = self._cam_pocket_targets(
            team_name, ball_x, ball_y, attacks_right, minute)
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None or state.position != "CAM":
                continue
            entry = decided.get(name)
            if entry is None:
                continue
            weight, tx, ty = entry
            pull_mult = 0.5 if state.last_active_minute == minute else 1.0
            pull = weight * pull_mult
            state.current_x += (tx - state.current_x) * pull
            state.current_y += (ty - state.current_y) * pull

    def _apply_half_space_magnet(self, team_name: str, opp_block) -> None:
        """
        Checkpoint 29 — physical half-space occupation against a compact block.

        Runs AFTER line cohesion and graph relaxation (both average
        midfielders back toward their line's center, which washed the
        magnet out when folded into the drift target). HalfSpaceMagnet
        gates to CAM/CM/CF, compact blocks, accessible channels and the
        player's current side of the pitch; the pull is tempered so the
        channel is a gravitation, not a teleport.
        """
        for name in self.team_rosters.get(team_name, []):
            state = self.states.get(name)
            if state is None:
                continue
            tx, ty = HalfSpaceMagnet.drift_target(
                state,
                (state.current_x, state.current_y),
                opp_block,
                (state.current_x, state.current_y),
            )
            state.current_x += (tx - state.current_x) * 0.55
            state.current_y += (ty - state.current_y) * 0.55

    def _apply_line_cohesion(self, team_name: str, pull_strength: float = 0.12,
                             compactness: Optional[float] = None):
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

        P5 — `compactness` (0→1) strengthens the sideways nudge when the
        defending side is a packed/narrow block. Only callers that opt in
        (i.e. defensive_block with a non-zero dial) raise the pull; the
        default `None` keeps this method behaviourally identical.
        """
        y_boost = 1.0 + 0.5 * (compactness or 0.0)
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
                y_pull = min(1.0, y_pull * y_boost)
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
        # GK is included so the "step up as a back-pass outlet" branch below
        # actually runs — it was dead code while GK was excluded from this set.
        attack_positions = {"ST", "CF", "LW", "RW", "CAM", "GK"}
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

            # CK28 — cached pitch-control space targets for this attacker,
            # computed once per run; the candidate loops apply them.
            _st_targets = self._space_targets(team_name, ax, ay)

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
                                                    # Penalty if too close to defenders; flat bonus for
                            # distances beyond sprint reach — living space (CK26)
                        min_d = None
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
                        score = ((1.0 if between else 0.4) * (1.6 if half_space else 1.0)
                                 * def_penalty * self._living_space_bonus(min_d))
                        # CK28 — target quality from cached pitch-control field
                        score *= self._space_bonus(_st_targets, tx, ty)
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
                        # Penalty if marked; bonus when the mark is beyond
                        # sprint reach — living space (CK26)
                        def_penalty = 1.0
                        if is_marked:
                            def_penalty = 0.5
                        score = ((1.0 + forward_score * 2.0) * flank_weight * y_factor
                                 * def_penalty
                                 * self._living_space_bonus(min_def_dist)
                                 * self._space_bonus(_st_targets, tx, ty))   # CK28
                        candidates.append((score, tx, ty))
                if candidates:
                    candidates.sort(key=lambda c: -c[0])
                    best_target = (candidates[0][1], candidates[0][2])

            elif a.position == "GK":
                # GK: step up to become a passing option when ball is in opponent's half.
                # Old form `in_opp_half and ball_x > 60.0 if attacks_right else ball_x < 45.0`
                # parsed as a conditional expression and dropped the in_opp_half guard
                # whenever attacking left — evaluate direction first, then the gate.
                eligible = (
                    in_opp_half and (ball_x > 60.0 if attacks_right else ball_x < 45.0)
                )
                if eligible:
                    # GK stays goal-side of the play in both attack directions.
                    dx_back = -15.0 if attacks_right else 15.0
                    tx = max(35.0, min(70.0, ball_x + dx_back))
                    ty = max(20.0, min(48.0, ball_y + random.uniform(-5, 5)))
                    best_target = (tx, ty)

            if best_target is not None:
                tx, ty = best_target
                dist = math.hypot(tx - ax, ty - ay)
                if dist > 3.0:
                    effective_pull = pull_strength * max(0.2, awareness_factor)
                    a.current_x += (tx - a.current_x) * effective_pull
                    a.current_y += (ty - a.current_y) * effective_pull

        # ── CHECKPOINT 26: PHYSICS — LIVING SPACE ─────────────────
    # Defenders have a finite sprint reach (SPRINT_REACH ~ a defender
    # covering ~8m). Beyond that gap the target escapes the mark, so
    # candidate scores get a flat bonus — turning the old hard threshold
    # penalties into smooth "living space" physics.
    SPRINT_REACH: float = 8.0
    REACH_BONUS_SLOPE: float = 45.0   # bonus saturates at reach+45m (~1.4x)

    @staticmethod
    def _living_space_bonus(min_def_dist: Optional[float]) -> float:
        if min_def_dist is None:
            return 1.0
        return 1.0 + max(0.0, (min_def_dist - PositionEngine.SPRINT_REACH)
                         / PositionEngine.REACH_BONUS_SLOPE)

    # ── CHECKPOINT 27: FORMATION GRAPH PHYSICS ─────────────────────
    # Formation as a force graph: nodes = players, edges are the formation
    # neighbor pairs below. Each edge springs the pair toward the rest
    # length implied by their HOME positions, with role-pair stiffness
    # (CB-CB tight twin-springs, CAM-mid loose, wing-fullback flanks mid).
    # Out of possession springs tighten (block moves as one unit); in
    # possession they slacken so runners can break shape.
    GRAPH_EDGE_SPECS = (
        (("CB",), ("CB",), 0.14),
        (("LB", "RB"), ("CB",), 0.08),
        (("CDM", "CM"), ("CDM", "CM"), 0.10),
        (("CAM",), ("CDM", "CM"), 0.06),
        (("ST", "CF"), ("CAM", "CDM", "CM"), 0.05),
        (("LW", "RW"), ("LB", "RB"), 0.07),
    )
    MATCH_SIDE_PAIRS = {frozenset({"LW", "LB"}), frozenset({"RW", "RB"})}
    MIN_SEPARATION: float = 4.0
    MIN_SEPARATION_PUSH: float = 0.08

    # ── CHECKPOINT 28: SPACE-CREATION TARGET WIRING ──────────────────
    # The physics already hands a live pitch-control field to the engine
    # (CK26). Below, candidate run targets score an extra bonus when the
    # field's space_creation_targets nominate an uncontrolled high-xT cell
    # nearby — greedy run quality informed by actual open space, not just
    # hard-coded direction multipliers.
    SPACE_WINDOW: float = 20.0   # reach within which a candidate bonus decays

    def _space_targets(self, team_name: str, ax: float, ay: float):
        """Cached pitch-control space targets for one attacker, or []."""
        field = getattr(self, "pitch_control_field", None)
        result = getattr(self, "pitch_control_result", None)
        if field is None or result is None:
            return []
        team = "home" if self.team_attacks_right.get(team_name, True) else "away"
        try:
            targets = field.space_creation_targets(result, team, ax, ay)
        except Exception:
            return []
        return targets

    def _space_bonus(self, targets, tx: float, ty: float) -> float:
        """1.0 + normalized nearest-target score, decayed by distance."""
        if not targets:
            return 1.0
        best = None
        best_dist = float("inf")
        for t in targets:
            d = math.hypot(t.x - tx, t.y - ty)
            if d < best_dist:
                best_dist, best = d, t.score
        if best is None or best_dist > self.SPACE_WINDOW:
            return 1.0
        max_score = max(t.score for t in targets) or 1.0
        norm = best / max_score
        return 1.0 + norm * max(0.0, 1.0 - best_dist / self.SPACE_WINDOW)

    def _graph_relaxation(self, team_name: str, in_possession: bool):
        """One iteration of formation-graph spring physics per drift."""
        roster = self.team_rosters.get(team_name, [])
        if len(roster) < 2:
            return
        phase_mult = 0.6 if in_possession else 1.4

        # Pair resolution: derive SPRING edges from the specs (role pairs),
        # filtering wing-fullback pairs to the same side of the pitch.
        edges = []
        seen = set()
        for(grp_a, grp_b, stiffness) in self.GRAPH_EDGE_SPECS:
            for na in roster:
                sa = self.states.get(na)
                if sa is None or sa.position not in grp_a:
                    continue
                for nb in roster:
                    sb = self.states.get(nb)
                    if sb is None or nb <= na or sb.position not in grp_b:
                        continue
                    pair_key = frozenset({na, nb})
                    if pair_key in seen:
                        continue
                    # Same-side gate for flank pairs: LW-LB / RW-RB only
                    if frozenset({sa.position, sb.position}) in self.MATCH_SIDE_PAIRS:
                        if (sa.home_y - 34.0) * (sb.home_y - 34.0) < 0:
                            continue
                    seen.add(pair_key)
                    rest = math.hypot(sa.home_x - sb.home_x, sa.home_y - sb.home_y)
                    edges.append((na, nb, stiffness, rest))

        # Apply springs: symmetric forces on both endpoints.
        for na, nb, stiffness, rest in edges:
            sa, sb = self.states[na], self.states[nb]
            dx = sb.current_x - sa.current_x
            dy = sb.current_y - sa.current_y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            err = dist - rest
            k = stiffness * phase_mult
            fx = k * err * (dx / dist)
            fy = k * err * (dy / dist)
            sa.current_x += fx * 0.5
            sa.current_y += fy * 0.5
            sb.current_x -= fx * 0.5
            sb.current_y -= fy * 0.5

        # Global min-separation repulsion: team-mates resist clustering.
        for i, na in enumerate(roster):
            sa = self.states.get(na)
            if sa is None:
                continue
            for nb in roster[i + 1:]:
                sb = self.states.get(nb)
                if sb is None:
                    continue
                dx = sb.current_x - sa.current_x
                dy = sb.current_y - sa.current_y
                dist = math.hypot(dx, dy)
                if dist < self.MIN_SEPARATION and dist > 1e-6:
                    push = (self.MIN_SEPARATION - dist) / dist * 0.5
                    sa.current_x -= dx * push * self.MIN_SEPARATION_PUSH
                    sa.current_y -= dy * push * self.MIN_SEPARATION_PUSH
                    sb.current_x += dx * push * self.MIN_SEPARATION_PUSH
                    sb.current_y += dy * push * self.MIN_SEPARATION_PUSH

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
        defensive_line: float = 0.5,
        compactness: float = 0.0,
        attacking_team: Optional[str] = None,
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

        Style-aware offside line: `defensive_line` (0=deep, 1=high line)
        controls how far the block is allowed to STEP UP off its own goal.
        A deep low-block team (defensive_line ~0.1) sinks the whole unit to
        within ~18m of the goal even at moderate danger and compresses
        harder as danger spikes; a high-line team (defensive_line ~0.9)
        resists the pull and holds the block up around 40m out, only caving
        to the six-yard line at CRITICAL danger.

        CHECKPOINT P1 — MARKING LAYER:
        When `attacking_team` is supplied, the block is ALSO anchored to the
        OPPONENT'S attackers, not just the ball's coordinates. Each defender
        is assigned a man by MarkingEngine (an aerial threat draws the best
        CB, a breaking winger draws the near-side fullback); the defender is
        then pulled toward a GOAL-SIDE slot ON their man rather than the bare
        ball-geometry line. A defender whose man is out of reach ("beaten")
        recovers goal-ward instead of chasing — the classic "stop the ball,
        don't chase the man" rule. Without `attacking_team` the method is the
        pure ball-geometry block (backwards-compatible).
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
        line_style = max(0.0, min(1.0, defensive_line))

        # Style-aware ceiling on how far the block may step OFF its own goal:
        # deep blocks only allow ~15m out at LOW defensive_line; high lines
        # hold the shape up to ~45m out (i.e. near the halfway line).
        max_upfield_m = 15.0 + 30.0 * line_style

        # Line sits a few metres goal-side of the ball, deepening with danger.
        # Deep teams (low defensive_line) sink harder as danger spikes; high
        # lines hold their ground better.
        behind = 10.0 - 8.0 * risk
        deepen = risk * (6.0 + (1.0 - line_style) * 6.0)
        line_x = ball_x + dir_toward_goal * (behind + deepen)
        if goal_x == 105.0:
            # Stay at least `max_upfield_m` from the defended goal line.
            line_x = min(103.0, max((105.0 - max_upfield_m), line_x))
        else:
            line_x = max(2.0, min((0.0 + max_upfield_m), line_x))

        # Lateral anchor: the unit shifts to the ball's side of the pitch.
        lateral = max(12.0, min(56.0, ball_y))
        # Deep teams respond more decisively to danger; high lines hold.
        intensity = pull_strength * (0.30 + 0.70 * risk) \
            * (1.0 - 0.20 * line_style)

        # ── CHECKPOINT P1: MARKING ASSIGNMENTS (attacker-aware) ─────────────
        # Solve the defender->attacker assignment once per block. Attackers
        # come from the defending team's own roster list, so we use only the
        # OPPONENTS' spatial states (self.states[].team == attacking_team).
        assignments: Dict[str, MarkAssignment] = {}
        if attacking_team and attacking_team != team_name:
            defenders = [
                (n, s.position, s.current_x, s.current_y)
                for n, s in self.states.items()
                if s.team == team_name and s.position in self.BLOCK_POSITIONS
                and s.position != "GK"
            ]
            attackers = []
            for n, s in self.states.items():
                if s.team != attacking_team or s.position == "GK":
                    continue
                is_runner = s.position in ("ST", "CF", "LW", "RW")
                aerial = 50.0
                pace = 50.0
                # Prefer DNA-backed speed/aerial when available on the state.
                # (state does not carry DNA; fall back to role defaults.)
                attackers.append((n, s.position, s.current_x, s.current_y,
                                  (is_runner, aerial, pace)))
            if attackers:
                assignments, _ = MarkingEngine.assign(
                    defenders, attackers, own_goal_x, ball_x, ball_y,
                    danger_level=danger_level,
                )

        # ── P5 COMPACTNESS DIAL ───────────────────────────────────────
        # A narrow/packed team (compactness→1.0) keeps its back four hugged
        # together: every non-marking outfield target gets pulled toward the
        # line's own lateral centroid so the block leaves no pockets between
        # bodies. A spread team (compactness→0.0) is untouched (default), so
        # this is a pure no-op wherever the dial is not supplied.
        block_centroid_y = lateral
        if compactness > 0.0:
            bk = [
                s.current_y
                for n, s in self.states.items()
                if s.team == team_name and s.position in ("CB", "LB", "RB")
            ]
            if bk:
                block_centroid_y = sum(bk) / len(bk)

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

            mark = assignments.get(name)
            # A goal-side shot-block lane taker. Initialised here so the CDM
            # branch and the mark branch both leave it defined; only near-side,
            # UNMARKED CB/LB/RB defenders actually take the lane (below).
            shot_block = False
            if mark is not None and mark.covers_someone:
                # ── MAN-MARKING BRANCH ─────────────────────────────
                # The defender is responsible for a specific attacker.
                # Body-on-man: sit goal-side of the attacker, on the line
                # between the man and the goal, a touch closer to goal than
                # the attacker. If the mark is BEATEN, recover goal-ward
                # instead (don't chase a man who is already gone — take the
                # space between him and the goal).
                ax, ay = mark.attacker_x, mark.attacker_y
                if mark.marker_state == "beaten":
                    # Recover to a goal-side pocket of the beaten man.
                    target_x = ax + dir_toward_goal * 6.0
                    target_y = ay
                else:
                    # Goal-side stand-off, ~1.5m goal-side of the attacker.
                    target_x = ax + dir_toward_goal * 1.5
                    target_y = ay

                if goal_x == 105.0:
                    target_x = min(103.0, max((105.0 - max_upfield_m), target_x))
                else:
                    target_x = max(2.0, min((0.0 + max_upfield_m), target_x))

                # The defender commits to his man properly:
                #   • tight  — sticks to the man goal-side, heavy pull
                #   • free   — balances the man with a firm covering pull
                #   • beaten — he lost the race: recover HARD to the goal-side
                #              pocket (never chase the man who is already gone).
                if mark.marker_state == "beaten":
                    mark_intensity = intensity * 1.0
                else:
                    stick = 0.55 + 0.35 * mark.tightness
                    mark_intensity = intensity * stick
                state.current_x += (target_x - state.current_x) * mark_intensity
                state.current_y += (target_y - state.current_y) * mark_intensity
                continue

            if state.position == "CDM":
                # Screens the line from the ball side, a few metres in front.
                target_x = line_x - dir_toward_goal * 6.0
                target_y = lateral
                if compactness > 0.0:
                    # Pack the screen toward the line's own centre too.
                    target_y = block_centroid_y + (target_y - block_centroid_y) * (1.0 - 0.5 * compactness)
            else:
                # ── CHECKPOINT — SHOT-BLOCK ALIGNMENT (team blocks, defenders
                #    first) ───────────────────────────────────────────────
                # When the ball sits in the shooting box at high/CRITICAL
                # danger, the defence "predicts the shot": instead of only
                # holding the line, the nearest goal-side, UNMARKED defender
                # steps up onto the ball-to-goal shirt line — getting his body
                # in the way of where the shooter will aim before the trigger
                # is pulled. This is the "defenders predict and position for a
                # block" behaviour: CBs/LBs/RBs nearest the ball carrier slide
                # goal-side of him, the rest of the line stays compact so the
                # whole team funnels shots through bodies.
                #
                # This works with the geometry_engine blocker-closing fix:
                # `defensive_block` positions the defender on the shot lane
                # here, and resolve_shot's per-tick blocker advancement lunges
                # him into the ball when the shot actually comes.
                if risk >= 0.55 and mark is None:
                    # Ball deep in the box relative to the goal line.
                    shot_zone = (
                        (goal_x - ball_x) < 22.0 if goal_x == 105.0
                        else (ball_x - goal_x) < 22.0
                    )
                    # A near-side, free defender takes the shot lane.
                    near = abs(state.current_y - ball_y) < 18.0
                    if shot_zone and near:
                        # Goal-side of the ball, on the shirt line toward goal.
                        standoff = 3.0 + 3.0 * (1.0 - risk)
                        target_x = ball_x + dir_toward_goal * standoff
                        target_y = ball_y
                        shot_block = True

                if not shot_block:
                    # CB/LB/RB: sit on the line; near-side players shift harder.
                    target_x = line_x
                    near = 1.0 if abs(state.current_y - ball_y) < 20.0 else 0.4
                    target_y = state.current_y + (lateral - state.current_y) * near
                    if compactness > 0.0:
                        # Compact sides pinch the line laterally toward its centroid.
                        target_y = block_centroid_y + (target_y - block_centroid_y) * (1.0 - 0.5 * compactness)

            # The defender on the shot lane commits decisively (a stronger pull
            # than the general block drift) so his body is actually on the line
            # when the shot comes; everyone else takes the standard pull.
            mover_intensity = intensity * (1.6 if shot_block else 1.0)
            state.current_x += (target_x - state.current_x) * mover_intensity
            state.current_y += (target_y - state.current_y) * mover_intensity

        # Keep the four-line shape cohesive after the block pull.
        self._apply_line_cohesion(team_name, compactness=compactness)

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

        # ── CHECKPOINT 34: WIDE FLANK OUTLET BONUS ─────────────────
        # A wide player standing in his flank channel right now is a real
        # standing outlet (checkpoints 18 / 21d). When the ball is CENTRAL
        # the ellipse's lateral taper taxes a touchline target ~2.4x
        # (direction ≈ 0.4) vs a central option (≈ 0.98), starving the wide
        # channels at the SELECTION stage — before the flank-aiming delivery
        # code ever sees the ball. Lift the direction floor for an
        # in-channel wide outlet so the ball can be switched/flipped to the
        # line from a central ball. Non-wide roles, wide players out of their
        # channel, and balls already on a flank are untouched — central-option
        # preference and the anisotropy guards survive for everyone else.
        # Checkpoint 35 — PITCH-STRETCH RULE: the floor scales with the
        # spine weight, so a deep-pitch-middle ball (packed middle) makes the
        # wide outlet an almost-central option (floor 0.80 -> 0.92), while a
        # ball near the edge of the band only nudges it (+0.00 at the edges).
        if (
            state.position in ("LW", "RW", "LB", "RB")
            and abs(hy - state.current_y) <= WIDE_OUTLET_CHANNEL_HALF
            and 18.0 <= ball_y <= 50.0
        ):
            spine = pitch_spine_weight(ball_y)
            direction = max(
                direction,
                WIDE_OUTLET_DIRECTION_FLOOR + spine * STRETCH_FLOOR_LIFT,
            )

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
        have drifted. Returns a y-coordinate biased toward the player's
        touchline post from `current_y`, so each wide reception drags the
        ball (and the player) back onto the flank and the width re-asserts
        itself. Non-wide roles are returned unchanged.
        Checkpoint 35 — PITCH-STRETCH RULE: the delivery is now aimed 75% of
        the way to the channel (up from 65%) — the pass itself is a width
        statement; the ball must find the line a touch sooner than the man.
        """
        state = self.states.get(player_name)
        if state is None or state.position not in ("LB", "RB", "LW", "RW"):
            return current_y
        anchor_y = state.home_y
        return current_y + (anchor_y - current_y) * 0.75

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

    def tracked_position(self, player_name: str) -> Optional[Tuple[float, float]]:
        """The player's last TRACKED position, or None if never tracked.

        `get_position` answers (50.0, 34.0) for a player it has no state for —
        the centre spot, which reads as a real coordinate. Any caller that is
        about to WRITE a coordinate into an event must use this instead, so an
        untracked player yields an honest absence rather than a plausible lie.
        """
        state = self.states.get(player_name)
        if state is None:
            return None
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

    # ── CHECKPOINT 26: MOTION-AWARE PITCH CONTROL FEED ───────────────
    # Velocity-aware pitch control lives here: each minute, after drift
    # runs, match_engine calls update_pitch_control() and the cached
    # field+result is then consumed (e.g. by winger half-space openness).
    # PlayerInfluenceInput defaults (vx=vy=0) mean legacy callers of the
    # field are unaffected; this engine forwards live velocity vectors.

    def influence_inputs(self, team_name: str) -> List:
        """Build velocity-carrying PlayerInfluenceInput rows for a team."""
        from pitch_control import PlayerInfluenceInput
        rows = []
        for name in self.team_rosters.get(team_name, []):
            st = self.states.get(name)
            if st is None:
                continue
            rows.append(PlayerInfluenceInput(
                name=name,
                team=team_name,
                position=st.position,
                x=st.current_x,
                y=st.current_y,
                pace=60.0,
                vx=st.velocity_x,
                vy=st.velocity_y,
                is_goalkeeper=(st.position == "GK"),
            ))
        return rows

    def update_pitch_control(self, home_team: str, away_team: str, minute: int = 0):
        """Recompute and cache the motion-aware field from live positions."""
        from pitch_control import PitchControlField
        if self._pc_field is None:
            self._pc_field = PitchControlField()
        home_players = self.influence_inputs(home_team)
        away_players = self.influence_inputs(away_team)
        if not home_players or not away_players:
            return
        self.pitch_control_result = self._pc_field.compute(
            home_players, away_players, minute=minute,
        )
        self.pitch_control_field = self._pc_field

    pitch_control_result: Optional = None
    pitch_control_field: Optional = None
    _pc_field = None


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