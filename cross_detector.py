"""
PLOFA 26/27 — CROSS DETECTOR  (Checkpoint 11)
==============================================
cross_detector.py

Why this exists:
    Analytics providers (Opta, StatsBomb) do NOT classify a cross by the
    passer's intent. They stamp a pass as `cross: true` from a pure geometric
    and spatial rule set:

        Opta definition:       an airborne pass from a WIDE position into the
                               OPPONENT PENALTY AREA, intended to create a
                               scoring opportunity.
        StatsBomb definition:  a pass played from a wide position on either
                               flank into the penalty box, targeting a
                               teammate or an area of high attacking threat,
                               logged as a Pass event with a
                               `pass: { cross: true }` metadata qualifier.

    So a "generic pass" that starts in the wide third and lands inside (or
    flashes through) the opponent's box IS a cross — regardless of what the
    engine's decision loop thought it was doing. This module is that
    classifier: a pure geometric predicate with no dependency on the event
    chain, so it is trivially unit-testable, mirroring threat_engine.py.

    Three structural criteria:
        1. ORIGIN — the ball is kicked from the wide third of the pitch
                    (the wings / outer channels) in the attacking half.
        2. DESTINATION — it lands inside the opponent penalty box, OR the
                    delivery path flashes through the box.
        3. TRAJECTORY — it is a KICK (never a throw-in). Crosses typically
                    travel through the air (Z-axis lift), though low driven
                    crosses along the ground are included too.

    The output acts as a TRIGGER downstream:
        • threat_engine  — a detected cross into the box forces the localised
          danger level HIGH/CRITICAL (D >= 75).
        • aerial defence — an airborne cross routes in-box defenders to the
          headed-clearance logic (Z > 1.2m rule) instead of foot clearances.
        • attacking runs — off-ball attackers crash the box (penalty-spot
          centre / back post) when a teammate enters the wide crossing zone.

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines).
Penalty box: 16.5m deep, 40.3m wide (y = 34 ± 20.15).
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# PITCH GEOMETRY CONSTANTS (metres)
# ─────────────────────────────────────────────

PITCH_X: float = 105.0
PITCH_Y: float = 68.0

#: The "wide third" of the pitch — the outer channels. A ball is on a wing
#: when its y is inside the outer third on either touchline.
WIDE_CHANNEL_WIDTH: float = PITCH_Y / 3.0          # ~22.67m

#: Penalty area dimensions (FIFA laws): 16.5m deep, 40.3m wide.
BOX_DEPTH: float = 16.5
BOX_HALF_WIDTH: float = 20.15                       # half of 40.3m

CENTER_Y: float = PITCH_Y / 2.0                     # 34.0


def penalty_box_bounds(attacks_right: bool) -> Tuple[float, float, float, float]:
    """(min_x, max_x, min_y, max_y) of the OPPONENT penalty box.

    `attacks_right=True` means this team attacks the goal at x=105, so the
    opponent box occupies the last 16.5m of the pitch. `False` mirrors it
    to the goal at x=0.
    """
    min_y = CENTER_Y - BOX_HALF_WIDTH
    max_y = CENTER_Y + BOX_HALF_WIDTH
    if attacks_right:
        return PITCH_X - BOX_DEPTH, PITCH_X, min_y, max_y
    return 0.0, BOX_DEPTH, min_y, max_y


def point_in_box(x: float, y: float, attacks_right: bool) -> bool:
    min_x, max_x, min_y, max_y = penalty_box_bounds(attacks_right)
    return min_x <= x <= max_x and min_y <= y <= max_y


def origin_is_wide(x: float, y: float, attacks_right: bool) -> bool:
    """Criterion 1 — the ball is kicked from the wide third (a wing) of the
    attacking half. The attacking-half gate keeps a defensive "cross" from
    your own wing out of the count: a cross is a scoring-opportunity
    delivery aimed at the OPPONENT box, so it must originate advanced."""
    on_wing = y <= WIDE_CHANNEL_WIDTH or y >= PITCH_Y - WIDE_CHANNEL_WIDTH
    if not on_wing:
        return False
    if attacks_right:
        return x > PITCH_X / 2.0
    return x < PITCH_X / 2.0


# ─────────────────────────────────────────────
# SEGMENT  ↔  RECTANGLE INTERSECTION
# (handles the "flashes directly through the box" case — the landing point
# may be just short of the box or past it, but the ball's path crossed it)
# ─────────────────────────────────────────────

def _segment_intersects_rect(
    x1: float, y1: float, x2: float, y2: float,
    rx_min: float, rx_max: float, ry_min: float, ry_max: float,
) -> bool:
    """Liang–Barsky style slab test: does the segment from (x1,y1) to
    (x2,y2) intersect the axis-aligned rectangle? Zero-length or degenerate
    segments that already sit inside the rect also count."""
    if rx_min <= x1 <= rx_max and ry_min <= y1 <= ry_max:
        return True
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0

    def clip(p: float, q: float) -> bool:
        nonlocal t0, t1
        if abs(p) < 1e-9:
            return q >= 0.0          # parallel — inside only if q >= 0
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
        return True

    if not clip(-dx, x1 - rx_min): return False
    if not clip(dx,  rx_max - x1): return False
    if not clip(-dy, y1 - ry_min): return False
    if not clip(dy,  ry_max - y1): return False
    return True


def delivery_path_crosses_box(
    from_x: float, from_y: float, to_x: float, to_y: float,
    attacks_right: bool,
) -> bool:
    min_x, max_x, min_y, max_y = penalty_box_bounds(attacks_right)
    return _segment_intersects_rect(
        from_x, from_y, to_x, to_y, min_x, max_x, min_y, max_y
    )


# ─────────────────────────────────────────────
# THE CLASSIFIER
# ─────────────────────────────────────────────

@dataclass
class CrossResult:
    """One geometric verdict for a single delivery.

    `is_cross` is the final stamp (StatsBomb `pass: { cross: true }`).
    The component fields expose WHY so downstream code and the export layer
    can read the geometry that produced the verdict.
    """
    is_cross: bool
    origin_zone: str = "not_wide"     # "wide" | "not_wide"
    destination_zone: str = "outside_box"  # "penalty_box" | "through_box" | "outside_box"
    airborne: bool = False            # Z-axis lift (drives headed clearance)
    reason: str = ""

    def as_dict(self) -> Dict:
        return {
            "is_cross": self.is_cross,
            "origin_zone": self.origin_zone,
            "destination_zone": self.destination_zone,
            "airborne": self.airborne,
            "reason": self.reason,
        }


def detect_cross(
    from_x: float, from_y: float,
    to_x: float, to_y: float,
    attacks_right: bool,
    event_type: str = "PASS",
    is_throw_in: bool = False,
    driven_low: bool = False,
) -> CrossResult:
    """Classify a single delivery against the Opta/StatsBomb geometric rule.

    Args:
        from_x/from_y:  kick point (origin).
        to_x/to_y:      where the delivery lands / is aimed.
        attacks_right:  True if the delivery is aimed at the goal at x=105.
        event_type:     the raw engine event label (informational only).
        is_throw_in:    throw-ins are NEVER crosses (criterion 3).
        driven_low:     True for a low, driven cross along the ground
                        (still a cross — but not airborne, so defenders
                        meet it with a foot, not a header).

    Returns:
        CrossResult — is_cross=True only when ALL three structural criteria
        hold: wide origin, box destination (landing OR flash-through),
        and a kicked (non-throw-in) delivery.
    """
    # Criterion 3 — trajectory: a kick, never a throw-in.
    if is_throw_in:
        return CrossResult(False, reason=f"{event_type} is a throw-in — never a cross")

    # Criterion 1 — origin in the wide third of the attacking half.
    wide = origin_is_wide(from_x, from_y, attacks_right)
    origin_zone = "wide" if wide else "not_wide"
    if not wide:
        return CrossResult(False, origin_zone=origin_zone,
                           reason="origin not in the wide third of the attacking half")

    # Criterion 2 — destination in the box, or the path flashing through it.
    lands_in_box = point_in_box(to_x, to_y, attacks_right)
    if lands_in_box:
        destination_zone = "penalty_box"
    elif delivery_path_crosses_box(from_x, from_y, to_x, to_y, attacks_right):
        destination_zone = "through_box"
    else:
        return CrossResult(False, origin_zone=origin_zone,
                           destination_zone="outside_box",
                           reason="delivery neither lands in nor flashes through the box")

    return CrossResult(
        is_cross=True,
        origin_zone=origin_zone,
        destination_zone=destination_zone,
        airborne=not driven_low,
        reason=("kicked from wide into the box" if lands_in_box
                else "kicked from wide through the box"),
    )


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python cross_detector.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n📐  PLOFA 26/27 — Cross Detector (Checkpoint 11) Standalone Demo")
    print("=" * 64)

    cases = [
        # (label, from, to, attacks_right, event_type, is_throw_in, driven_low)
        ("LW whipped to far post (right→)",    (80, 12),  (98, 24), True,  "CROSS_ATTEMPT", False, False),
        ("RB low drive to near post (right→)", (75, 55),  (90, 33), True,  "CROSS_ATTEMPT", False, True),
        ("centrally-played through ball",      (80, 34),  (92, 34), True,  "THROUGH_BALL",  False, False),
        ("cutback from the byline (wide)",     (92, 14),  (82, 34), True,  "CROSS_ATTEMPT", False, False),
        ("deep pass from own half wing",       (40, 10),  (88, 30), True,  "PASS",          False, False),
        ("central midfield ping (not wide)",   (50, 34),  (90, 30), True,  "PASS",          False, False),
        ("long throw into the box",            (80, 2),   (95, 20), True,  "THROW_IN",      True,  False),
        ("back-post cross (left→ mirror)",     (25, 56),  (7, 48),  False, "CROSS_ATTEMPT", False, False),
        ("centralised engine 'cross' (wide x, central y)", (78, 34), (92, 34), True, "CROSS_ATTEMPT", False, False),
    ]
    print(f"\n  {'delivery':<44} {'verdict':<6} {'origin':<10} {'dest':<12} air  reason")
    print("  " + "-" * 100)
    for label, fr, to, ar, etype, thr, low in cases:
        r = detect_cross(*fr, *to, ar, etype, thr, low)
        print(f"  {label:<44} {'CROSS' if r.is_cross else 'pass':<6}"
              f" {r.origin_zone:<10} {r.destination_zone:<12}"
              f" {str(r.airborne):<5} {r.reason}")

    print("\n   Origin-wide gate: wing channel is y <= 22.7 or y >= 45.3")
    print("   (attacking half only); box = 16.5m deep, y = 34 +/- 20.15.\n")
    print("✅ Cross Detector module operational — zero dependency on rest of PLOFA.\n")
