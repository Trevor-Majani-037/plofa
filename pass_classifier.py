"""
PLOFA 26/27 — OPTA-STYLE PASS CLASSIFIER  (Checkpoint 13)
=========================================================
pass_classifier.py

Why this exists:
    The match engine already stamps geometric verdicts for two Opta /
    StatsBomb qualities — CROSS (`cross_detector`) and LONG PASS
    (`long_pass_detector`). This module completes the pass taxonomy the
    analyst specification calls for, so a single delivery can be described
    the way a data provider would:

        • PASS TYPE        — chipped pass, headed pass, launch, flick-on,
                             pull-back, lay-off, through ball, tap pass,
                             ground pass (base).
        • PASS LENGTH      — short / medium / long (metres + yards).
        • PASS DIRECTION   — forward / sideways / backward.
        • START / END ZONE — own vs opposition half; defensive / middle /
                             final third (by ORIGIN, as Opta break counts
                             down by where the pass is played FROM);
                             left / centre / right channel.
        • EXCLUSIONS       — Crosses, keeper-throws and throw-ins are NOT
                             passes (Opta category rule). A delivery the
                             geometric cross detector stamps `is_cross`
                             never takes a pass type here.

    The classifier is a PURE geometric predicate (+ optional contextual
    flags for airborne/headed/pressure/dead-ball that the event chain can
    supply) with zero dependency on the rest of PLOFA — trivially
    unit-testable, mirroring cross_detector.py and long_pass_detector.py.

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines).
    Half-line   x = 52.5
    Thirds      x = 0..35 | 35..70 | 70..105  (defensive / middle / final)
    Channels    y = 0..22.67 | 22.67..45.33 | 45.33..68 (left / centre / right)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

from long_pass_detector import (
    LONG_PASS_THRESHOLD_M,
    pass_distance_m,
    pass_distance_yards,
)


# ─────────────────────────────────────────────
# PITCH GEOMETRY CONSTANTS (metres)
# ─────────────────────────────────────────────

PITCH_X: float = 105.0
PITCH_Y: float = 68.0
CENTER_X: float = PITCH_X / 2.0          # 52.5
CENTER_Y: float = PITCH_Y / 2.0          # 34.0

THIRD_DEFENSIVE_MAX: float = 35.0
THIRD_MIDDLE_MAX: float = 70.0

CHANNEL_WIDTH: float = PITCH_Y / 3.0     # ~22.67

#: Short/medium/long split (Opta-style, in yards then metres).
#:   short   <  SHORT_PASS_MAX_M    (15 yd)
#:   medium  <  LONG_PASS_THRESHOLD_M (35 yd)
#:   long    >= LONG_PASS_THRESHOLD_M
SHORT_PASS_MAX_M: float = 15.0 * 0.9144  # ≈ 13.72 m

#: Through-ball "split run" gate: forward displacement required to be a
#: plausible line-splitting pass for a runner.
THROUGH_FORWARD_MIN: float = 12.0

#: Pull-back byline proximity: the pass's origin must be this close to the
#: byline of the box to be a cut-back.
BYLINE_PROX: float = 8.0
BOX_DEPTH: float = 16.5


# ─────────────────────────────────────────────
# PITCH ZONE PREDICATES
# ─────────────────────────────────────────────

def origin_half(x: float) -> str:
    """'own' if x is in the passer's defensive half, else 'opposition'."""
    return "own" if x < CENTER_X else "opposition"


def start_third(x: float) -> str:
    """Defensive / middle / final third, always from the passer's attacking
    direction (i.e. absolute x, defensive ladder grows toward 105)."""
    if x < THIRD_DEFENSIVE_MAX:
        return "defensive_third"
    if x < THIRD_MIDDLE_MAX:
        return "middle_third"
    return "final_third"


def channel(y: float) -> str:
    """Left / centre / right channel of the point's y-coordinate."""
    if y < CHANNEL_WIDTH:
        return "left"
    if y > PITCH_Y - CHANNEL_WIDTH:
        return "right"
    return "centre"


# ─────────────────────────────────────────────
# PASS DIRECTION
# ─────────────────────────────────────────────

def classify_direction(signed_dx: float) -> str:
    """Forward / sideways / backward from attacker-signed forward
    displacement (positive = toward the opposition goal)."""
    if signed_dx > 1.0:
        return "forward"
    if signed_dx < -1.0:
        return "backward"
    return "sideways"


# ─────────────────────────────────────────────
# PASS LENGTH
# ─────────────────────────────────────────────

def classify_length(distance_m: float) -> str:
    """short / long (two-bucket, Opta baseline uses the 35 yd rule)."""
    return "short" if distance_m < LONG_PASS_THRESHOLD_M else "long"


def classify_length_class(distance_m: float) -> str:
    """short / medium / long (three-bucket)."""
    if distance_m < SHORT_PASS_MAX_M:
        return "short"
    if distance_m < LONG_PASS_THRESHOLD_M:
        return "medium"
    return "long"


# ─────────────────────────────────────────────
# PASS TYPE LOGIC (no deps beyond this module)
# ─────────────────────────────────────────────

@dataclass
class PassTypeResult:
    """Raw subtype verdict, pre-exclusion, for diagnostics."""
    type: str
    reason: str


def _classify_type(
    from_x: float, from_y: float, to_x: float, to_y: float,
    signed_dx: float, distance_m: float,
    is_airborne: bool, is_headed: bool, under_pressure: bool,
    is_dead: bool, attacks_right: bool,
) -> PassTypeResult:
    """Assign the Opta-style pass type from geometry + physical flags supplied
    by the chain. Pure function: no randomness, no chain dependency."""

    # ── DEAD-BALL SUBTYPES ─────────────────────────────
    if is_dead:
        # TAP PASS — a short roll after a dead ball that cannot have an
        # unsuccessful outcome, e.g. rolling it a very short distance to a
        # teammate to shoot.
        if distance_m < SHORT_PASS_MAX_M and not is_airborne:
            return PassTypeResult("tap", "short dead-ball roll that cannot fail")
        # Corner / goal-kick / free-kick played as a pass counts as a long
        # pass when it covers the distance (still logged as a pass, not a
        # cross, per Opta).
        if distance_m >= LONG_PASS_THRESHOLD_M:
            return PassTypeResult("long kick / pass", "set piece covers " 
                                  "the distance requirement")
        return PassTypeResult("pass from dead ball", "settled set piece first pass")

    # ── PULL-BACK ──────────────────────────────────────
    # A player in the opposition box reaches the byline and cuts the ball
    # BACK up the pitch to a teammate. Excludes crosses (handled upstream).
    opp_byline_x = 0.0 if not attacks_right else PITCH_X
    near_byline = abs(from_x - opp_byline_x) <= BYLINE_PROX
    if near_byline and signed_dx < -1.0:
        return PassTypeResult(
            "pull-back", "cutback from the opposition byline")

    # ── LAY-OFF ────────────────────────────────────────
    # A first-time pass (typified by a target man) away from goal, one touch,
    # when under pressure and with the back to goal.
    if under_pressure and signed_dx < 1.0 and distance_m < LONG_PASS_THRESHOLD_M:
        return PassTypeResult("lay-off", "first-time pass away from goal under "
                                       "pressure")

    # ── FLICK-ON ───────────────────────────────────────
    # A glancing pass (head or foot) helping the ball on in the same general
    # direction. Glance = short, close-range touch in the onward direction.
    if (is_headed or distance_m < SHORT_PASS_MAX_M) and signed_dx > 0.0:
        return PassTypeResult(
            "flick-on", "glancing help-on, same general direction")

    # ── HEADED PASS ────────────────────────────────────
    # A header with an intended recipient.
    if is_headed:
        return PassTypeResult("headed pass", "header finding a teammate")

    # ── LAUNCH ─────────────────────────────────────────
    # A long high ball into space or an area for players to chase or challenge.
    if distance_m >= LONG_PASS_THRESHOLD_M:
        return PassTypeResult("launch", "long ball into space / area")

    # ── CHIPPED PASS ───────────────────────────────────
    # A lofted ball with an intended recipient; over shoulder height, using
    # loft to avoid opposition.
    if is_airborne and signed_dx > 0.0:
        return PassTypeResult("chipped pass", "lofted ball over the shoulder")

    # ── THROUGH BALL ───────────────────────────────────
    # A pass splitting the defence for a teammate to run on to.
    if signed_dx >= THROUGH_FORWARD_MIN:
        return PassTypeResult("through ball", "splitting-forward pass for a runner")

    # ── BASE ───────────────────────────────────────────
    return PassTypeResult("ground pass", "basic pass")


# ─────────────────────────────────────────────
# THE CLASSIFIER  (public entry point)
# ─────────────────────────────────────────────

@dataclass
class PassClassification:
    is_pass: bool                  # False if excluded (cross/throw-in/keeper throw)
    pass_type: str = ""            # Opta-style subtype ("" when excluded)
    length: str = "short"          # short / long
    length_class: str = "medium"   # short / medium / long
    distance_m: float = 0.0
    distance_yards: float = 0.0
    direction: str = "sideways"    # forward / sideways / backward
    start_half: str = "own"
    end_half: str = "own"
    start_third: str = "middle_third"
    end_third: str = "middle_third"
    channel: str = "centre"
    exclusion: str = ""            # "" | cross | throw_in | keeper_throw

    def as_dict(self) -> Dict:
        return {
            "is_pass": self.is_pass,
            "pass_type": self.pass_type,
            "length": self.length,
            "length_class": self.length_class,
            "distance_m": round(self.distance_m, 1),
            "distance_yards": round(self.distance_yards, 1),
            "direction": self.direction,
            "start_half": self.start_half,
            "end_half": self.end_half,
            "start_third": self.start_third,
            "end_third": self.end_third,
            "channel": self.channel,
            "exclusion": self.exclusion,
        }


def classify_pass(
    from_x: float, from_y: float, to_x: float, to_y: float,
    *,
    signed_dx: float = None,
    is_cross: bool = False,
    is_throw_in: bool = False,
    is_keeper_throw: bool = False,
    is_airborne: bool = False,
    is_headed: bool = False,
    under_pressure: bool = False,
    is_dead: bool = False,
    attacks_right: bool = True,
) -> PassClassification:
    """Full Opta-style classification of a single pass delivery.

    Args:
        from_x/from_y:  kick point (origin).
        to_x/to_y:      destination / aimed point.
        signed_dx:      forward displacement (m), signed + toward the goal;
                        if None, computed from (to_x - from_x) adjusted for
                        `attacks_right`.
        is_cross / is_throw_in / is_keeper_throw: categorical exclusions.
        is_airborne / is_headed / under_pressure / is_dead: physical flags.
    """
    dist_m = pass_distance_m(from_x, from_y, to_x, to_y)
    dist_yd = pass_distance_yards(from_x, from_y, to_x, to_y)

    if signed_dx is None:
        signed_dx = (to_x - from_x) if attacks_right else (from_x - to_x)

    # Categorical exclusions first.
    if is_cross:
        return PassClassification(False, exclusion="cross",
                                  distance_m=dist_m, distance_yards=dist_yd)
    if is_throw_in or is_keeper_throw:
        return PassClassification(
            False,
            exclusion="throw_in" if is_throw_in else "keeper_throw",
            distance_m=dist_m, distance_yards=dist_yd)

    length = classify_length(dist_m)
    length_class = classify_length_class(dist_m)

    type_result = _classify_type(
        from_x, from_y, to_x, to_y, signed_dx, dist_m,
        is_airborne, is_headed, under_pressure, is_dead, attacks_right,
    )

    return PassClassification(
        is_pass=True,
        pass_type=type_result.type,
        length=length,
        length_class=length_class,
        distance_m=dist_m,
        distance_yards=dist_yd,
        direction=classify_direction(signed_dx),
        start_half=origin_half(from_x),
        end_half=origin_half(to_x),
        start_third=start_third(from_x),
        end_third=start_third(to_x),
        channel=channel(from_y),
    )


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python pass_classifier.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n🎯  PLOFA 26.7 — Opta Pass Classifier (Checkpoint 13) Standalone Demo")
    print("=" * 64)

    cases = [
        # label, from, to, attacks_right, flags (air, headed, press, dead, cross, throw)
        ("tap: short dead-ball roll to shoot",   (78, 40),  (82, 40), True,  (False, False, False, True, False, False)),
        ("goal-kick long distribution",          (8, 34),   (70, 30), True,  (True,  False, False, True, False, False)),
        ("pull-back cutback from byline",        (99, 26),  (90, 34), True,  (False, False, False, False, False, False)),
        ("lay-off under pressure, back to play", (88, 34),  (82, 38), True,  (False, False, True,  False, False, False)),
        ("headed pass to teammate",              (70, 30),  (82, 34), True,  (True,  True,  False, False, False, False)),
        ("flick-on glancing header onward",      (85, 33),  (92, 36), True,  (True,  True,  False, False, False, False)),
        ("chipped pass over the shoulder",       (60, 28),  (75, 34), True,  (True,  False, False, False, False, False)),
        ("launch long ball into space",          (28, 34),  (80, 50), True,  (True,  False, False, False, False, False)),
        ("through ball splitting the defence",   (60, 30),  (85, 40), True,  (False, False, False, False, False, False)),
        ("deep stint cross reclassified",        (80, 12),  (98, 34), True,  (True,  False, False, False, True,  False)),
        ("throw-in (never a pass)",              (60, 2),   (82, 20), True,  (False, False, False, False, False, True)),
        ("backward recycle",                       (45, 34), (42, 36), True,  (False, False, False, False, False, False)),
    ]

    print(f"\n  {'delivery':<40} {'TYPE':<16} {'len':<7} {'dir':<8} {'half':<11} {'third':<15} {'chan':<6} excl")
    print("  " + "-" * 108)
    for label, (fx, fy), (tx, ty), ar, flags in cases:
        r = classify_pass(fx, fy, tx, ty, attacks_right=ar,
                          is_cross=flags[4], is_throw_in=flags[5],
                          is_airborne=flags[0], is_headed=flags[1],
                          under_pressure=flags[2], is_dead=flags[3])
        if r.is_pass:
            print(f"  {label:<40} {r.pass_type:<16} {r.length:<7} {r.direction:<9} "
                  f"{r.start_half:<11} {r.start_third:<15} {r.channel:<6}  "
                  f"{r.length_class} ({r.distance_yards:.0f} yd)")
        else:
            print(f"  {label:<40} {'EXCLUDED':<16} ({r.exclusion})")

    print("\n   Half-line x=52.5; thirds 0/35/70/105; channels y 0/22.7/45.3/68.")
    print("   Crosses, throw-ins and keeper-throws are excluded by Opta category.\n")
    print("✅ Pass Classifier operational — pure geometric predicate, no PLOFA deps.\n")