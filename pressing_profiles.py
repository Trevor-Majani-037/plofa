"""
PLOFA 26/27 — PRESSING PROFILES (Geometric 30° Cover-Shadow Pressing)
======================================================================
pressing_profiles.py

Why this exists:
    Pressing in PLOFA was a flat probability table keyed off the ball's
    THIRD with a single press_intensity scalar. It could not express the
    three structurally distinct pressing identities a real league contains,
    and it had no GEOMETRY: a defender's cover shadow (the 30° cone a
    defender casts over the passing lanes behind him) never blocked a pass,
    so a high press never actually choked build-up routes into the keeper.

    This module is the pure decision/geometry layer for that system:

        Module 1 — Three pressing profiles, each with its own line of
                   engagement, engagement range, cover-shadow cone and
                   stamina fatigue tax:

                       ULTRA_HIGH_GEGENPRESS   — press the keeper & CBs in
                                                 the opponent's own third.
                       MID_BLOCK_TRAP          — sit in a mid-block, trap
                                                 wide/central passes.
                       LOW_BLOCK_CONTAIN       — hold the final third line,
                                                 only press inside 35m.

        Module 2 — The 30° cover-shadow predicate. A defender at (dx, dy)
                   casts a cone of half-angle 15° (30° total) BEHIND him
                   away from the carrier. A pass whose line runs through
                   that cone is geometrically choked — this is what drives
                   the possession phase engine's GK Emergency Phase
                   Regression: when every forward lane is inside a cone,
                   the machine regresses the phase to the goalkeeper.

        Module 3 — A stamina fatigue tax per profile. Pressing hard drains
                   more stamina per press, which lowers the team's average
                   stamina, which the TacticalAI turns into a lower effective
                   press intensity next minute — the "press yourself into
                   exhaustion" loop.

It has no dependency on the event chain, the RNG, or the position engine
(the position engine is passed in as an argument), mirroring
cross_detector.py / possession_phases.py, so it is trivially unit-testable.

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines).
nx = ball's distance from the ATTACKING team's own goal line.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from attacking_matrix import lane_clearance

PITCH_X: float = 105.0
PITCH_Y: float = 68.0

# A pass lane is treated as blocked when the geometric clearance (minimum of
# the perpendicular-distance model and the cover-shadow cone model) drops
# below this value. Shared with the phase engine's lane gates.
COVER_SHADOW_BLOCK_THRESHOLD: float = 0.30

# Total cover-shadow cone is 30° => half-angle 15°.
DEFAULT_CONE_HALF_ANGLE_DEG: float = 15.0

# How many points along the carrier→target segment are sampled for the
# cover-shadow cone test. 5 evenly-spaced samples is cheap and robust.
_SEGMENT_SAMPLES: int = 5


class PressingProfile(Enum):
    """The three structurally distinct pressing identities."""

    ULTRA_HIGH_GEGENPRESS = "ultra_high_gegenpress"
    MID_BLOCK_TRAP = "mid_block_trap"
    LOW_BLOCK_CONTAIN = "low_block_contain"


@dataclass(frozen=True)
class PressProfileConfig:
    """
    One profile's dials.

    engage_line_nx : the ball must be at least this far from the ATTACKING
                     team's own goal line (i.e. close enough to the
                     DEFENDING goal) for this team to engage. Low values =
                     the team presses high up the pitch.
    engagement_range_m : how close a defender must be to the carrier to
                     actually commit a press.
    cone_half_angle_deg : half-angle of the cover shadow (default 15° ⇒ 30°).
    shadow_length_m : how far behind a defender his cover shadow extends.
    stamina_tax : fatigue multiplier applied to every press the team makes.
    zone_probs : base press probability per third (before scaling by the
                 live, TacticalAI-adjusted press_intensity).
    """

    press_intensity_base: float
    engage_line_nx: float
    engagement_range_m: float
    cone_half_angle_deg: float
    shadow_length_m: float
    stamina_tax: float
    zone_probs: Dict[str, float]


PROFILES: Dict[PressingProfile, PressProfileConfig] = {
    PressingProfile.ULTRA_HIGH_GEGENPRESS: PressProfileConfig(
        press_intensity_base=0.95,
        engage_line_nx=27.0,       # presses the opponent's own third build-up
        engagement_range_m=14.0,   # a covering defender commits from distance
        cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
        shadow_length_m=14.0,      # long shadow = lanes close quickly
        stamina_tax=1.35,          # expensive: sprints back to goal side
        zone_probs={
            "own_third": 0.30, "mid_third": 0.45,
            "att_third": 0.55, "box": 0.62,
        },
    ),
    PressingProfile.MID_BLOCK_TRAP: PressProfileConfig(
        press_intensity_base=0.60,
        engage_line_nx=45.0,       # engages around the halfway line
        engagement_range_m=11.0,
        cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
        shadow_length_m=12.0,
        stamina_tax=1.15,          # moderate extra cost
        zone_probs={
            "own_third": 0.10, "mid_third": 0.28,
            "att_third": 0.44, "box": 0.55,
        },
    ),
    PressingProfile.LOW_BLOCK_CONTAIN: PressProfileConfig(
        press_intensity_base=0.25,
        engage_line_nx=65.0,       # only presses inside the final third
        engagement_range_m=8.0,
        cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
        shadow_length_m=10.0,
        stamina_tax=1.0,           # low block is energy-conserving
        zone_probs={
            "own_third": 0.03, "mid_third": 0.12,
            "att_third": 0.30, "box": 0.50,
        },
    ),
}


def profile_for_intensity(intensity: float) -> PressingProfile:
    """Pick a profile from the live (adjusted) press_intensity band."""
    if intensity >= 0.75:
        return PressingProfile.ULTRA_HIGH_GEGENPRESS
    if intensity >= 0.45:
        return PressingProfile.MID_BLOCK_TRAP
    return PressingProfile.LOW_BLOCK_CONTAIN


def profile_for_style(style_key: str) -> PressingProfile:
    """Pick a profile from the team's authored style."""
    key = (style_key or "").strip().lower()
    if key in ("gegenpressing", "ultra_attacking", "vertical_tiki_taka",
               "high_press", "gegenpress"):
        return PressingProfile.ULTRA_HIGH_GEGENPRESS
    if key in ("park_the_bus", "ultra_defensive", "defensive", "route_one",
               "low_block"):
        return PressingProfile.LOW_BLOCK_CONTAIN
    return PressingProfile.MID_BLOCK_TRAP


def resolve_profile(
    intensity: Optional[float] = None,
    style_key: Optional[str] = None,
) -> PressingProfile:
    """
    The team's pressing identity: authored style wins when it names one of
    the three structural archetypes outright; otherwise the live intensity
    band decides (which keeps TacticalAI's fatigue-driven intensity drops
    able to pull a team down a pressing tier).
    """
    styled = PressingProfile.MID_BLOCK_TRAP
    if style_key:
        styled = profile_for_style(style_key)
        # Only honour the style when it is an explicitly pressing archetype,
        # not when the style is e.g. balanced/tiki_taka (which sits in the
        # mid-block tier anyway).
        if style_key.strip().lower() in (
            "gegenpressing", "ultra_attacking", "vertical_tiki_taka",
            "high_press", "park_the_bus", "ultra_defensive",
            "defensive", "route_one",
        ):
            return styled
    if intensity is not None:
        return profile_for_intensity(intensity)
    return styled


def engagement_allows(nx: float, profile: PressingProfile) -> bool:
    """Is the ball far enough from the attacking goal for this profile to
    engage? nx = distance from the ATTACKING team's own goal line."""
    return nx >= PROFILES[profile].engage_line_nx


def in_cover_shadow(
    cx: float, cy: float,          # carrier
    dx: float, dy: float,          # defender (cone apex)
    px: float, py: float,          # test point
    half_angle_rad: float,
    shadow_len: float,
) -> bool:
    """
    Pure geometric cover-shadow predicate.

    The defender casts a cone of half-angle `half_angle_rad` BEHIND him,
    in the direction pointing away from the carrier. A test point is inside
    the shadow when it sits on the far side of the defender, within
    `shadow_len` metres, and within the cone half-angle.
    """
    ax, ay = dx - cx, dy - cy          # axis carrier -> defender
    norm = math.hypot(ax, ay)
    if norm < 1e-6:
        return False
    ax, ay = ax / norm, ay / norm

    vx, vy = px - dx, py - dy          # defender -> test point
    if ax * vx + ay * vy < 0.0:        # point is on the carrier's side
        return False
    dist = math.hypot(vx, vy)
    if dist > shadow_len or dist < 1e-6:
        return False
    cos_a = (ax * vx + ay * vy) / dist
    return cos_a >= math.cos(half_angle_rad)


def cover_shadow_clearance(
    x: float, y: float,
    tx: float, ty: float,
    defenders: Optional[List[Any]],
    position_engine: Any,
    profile: PressingProfile,
    engaged: bool = True,
) -> float:
    """
    Combined clearance for the carrier→target corridor.

    Returns the minimum of:
        • the perpendicular-distance lane model (defender ON the line), and
        • the cover-shadow cone model (a defender's 30° cone overlapping the
          line even when he is not on it).

    0.0 => blocked, 1.0 => completely clear. With no spatial info the
    corridor is treated as clear (1.0), preserving existing behaviour.
    """
    if position_engine is None or not defenders:
        return 1.0
    base = lane_clearance(x, y, tx, ty, defenders, position_engine)
    if base <= 0.0 or not engaged:
        return base

    cfg = PROFILES[profile]
    half_angle = math.radians(cfg.cone_half_angle_deg)
    seg_len = math.hypot(tx - x, ty - y)
    if seg_len < 1e-6:
        return base

    for d in defenders:
        if getattr(d, "position", None) == "GK":
            continue
        dx, dy = position_engine.get_position(d.name)
        for i in range(_SEGMENT_SAMPLES):
            t = i / (_SEGMENT_SAMPLES - 1)
            px = x + t * (tx - x)
            py = y + t * (ty - y)
            if in_cover_shadow(x, y, dx, dy, px, py, half_angle, cfg.shadow_length_m):
                return 0.0
    return base


def cover_shadow_blocked(
    x: float, y: float,
    tx: float, ty: float,
    defenders: Optional[List[Any]],
    position_engine: Any,
    profile: PressingProfile,
    engaged: bool = True,
) -> bool:
    """True when the corridor is choked below the shared block threshold."""
    return (
        cover_shadow_clearance(x, y, tx, ty, defenders, position_engine,
                               profile, engaged=engaged)
        < COVER_SHADOW_BLOCK_THRESHOLD
    )
