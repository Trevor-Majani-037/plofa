"""
PLOFA 26/27 — LONG PASS DETECTOR  (Checkpoint 12)
==================================================
long_pass_detector.py

Why this exists:
    Analytics providers (Opta, StatsBomb) classify a LONG PASS by a strict
    metric threshold, not by the passer's intent:

        Opta baseline:       any ground or airborne pass that travels a
                             total distance of 35 yards (≈ 32.0 m) or more
                             over the pitch surface is a Long Pass. A pass
                             that meets this bar is stripped of its "short
                             pass" designation and automatically bucketed
                             as Long Pass.
        StatsBomb baseline:  a generic Pass event with qualifiers
                             pass.length (tracked strictly in meters) and
                             pass.height (Ground / Low Pass / High Pass).
                             Any pass where pass.length >= 32.0 functions
                             as a long pass within analytical models.

    So a "generic pass" that physically covers >= 32.0 m IS a long pass —
    regardless of what the engine's decision loop intended it to be. This
    module is that classifier: a pure geometric predicate with no
    dependency on the event chain, so it is trivially unit-testable,
    mirroring cross_detector.py and threat_engine.py.

    Two structural exclusions (Opta categorical logic):
        1. CROSSES — a wide player lofting a 40-yard ball from the wing into
           the opponent's box satisfies the distance requirement but is
           logged EXCLUSIVELY as a Cross, never a long pass.
        2. UNCONTROLLED CLEARANCES — a panicked boot upfield under defensive
           pressure is flagged as a Clearance. It only registers as a long
           pass if the player has controlled possession and intentionally
           targets a teammate or a specific structural space. Engine
           CLEARANCE events are excluded by default.

    The output acts as a TRIGGER downstream:
        • event_chain — every pass event is stamped with the geometric
          verdict (`is_long`, pass_length_m, pass_length_yards, pass_height).
        • exporter    — long pass counting uses the geometric stamp instead
          of a probabilistic flag, and cross/clearance deliveries are kept
          out of the long pass totals.

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines). The
"total distance over the pitch surface" is the Euclidean hypotenuse
between the kick point and the landing point — NOT just the forward x
displacement.
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Dict, Any


# ─────────────────────────────────────────────
# LONG PASS THRESHOLD (metres)
# ─────────────────────────────────────────────

#: Opta strict baseline: 35 yards. One yard = 0.9144 m.
LONG_PASS_YARDS: float = 35.0
LONG_PASS_THRESHOLD_M: float = LONG_PASS_YARDS * 0.9144   # ≈ 32.004 m

#: StatsBomb qualifier uses a rounded >= 32.0 m rule. Keep the exact yard
#: conversion (32.004) so a pass measured at exactly 35.0 yards qualifies,
#: matching both providers' strict threshold.
STATSBOMB_LONG_PASS_M: float = 32.0


def pass_distance_m(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Total distance (in metres) the ball travels over the pitch surface.

    Euclidean hypotenuse between origin and destination. The engine's
    coordinate space is already metric (pitch 105 x 68), so this is the
    real ground distance, including lateral movement — a 20 m forward
    switch can easily exceed the 32 m bar once its sideways component is
    added.
    """
    return math.hypot(to_x - from_x, to_y - from_y)


def pass_distance_yards(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Distance travelled, expressed in yards (Opta's native unit)."""
    return pass_distance_m(from_x, from_y, to_x, to_y) / 0.9144


# ─────────────────────────────────────────────
# PASS HEIGHT (StatsBomb qualifier)
# ─────────────────────────────────────────────

#: StatsBomb pass.height taxonomy.
HEIGHT_GROUND: str = "Ground"
HEIGHT_LOW: str = "Low Pass"
HEIGHT_HIGH: str = "High Pass"


def classify_pass_height(is_airborne: bool, driven_low: bool = False) -> str:
    """Map a trajectory to the StatsBomb pass.height qualifier.

    A low, driven delivery hugging the turf is `Ground` (or `Low Pass`);
    anything lifted off the deck is `High Pass`. The engine's crossing
    detector already computes `airborne` for wide deliveries; generic
    ground passes default to `Ground` unless a caller marks otherwise.
    """
    if is_airborne:
        return HEIGHT_HIGH
    if driven_low:
        return HEIGHT_LOW
    return HEIGHT_GROUND


# ─────────────────────────────────────────────
# THE CLASSIFIER
# ─────────────────────────────────────────────

@dataclass
class LongPassResult:
    """One geometric verdict for a single pass delivery.

    `is_long_pass` is the final Opta-style stamp. The component fields
    expose WHY so downstream code and the export layer can read the
    geometry that produced the verdict (and the exclusion that overrode
    an otherwise-qualifying distance).
    """
    is_long_pass: bool
    distance_m: float = 0.0
    distance_yards: float = 0.0
    height: str = HEIGHT_GROUND
    excluded: str = ""            # "" | "cross" | "clearance" | "throw_in"
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_long_pass": self.is_long_pass,
            "distance_m": round(self.distance_m, 1),
            "distance_yards": round(self.distance_yards, 1),
            "height": self.height,
            "excluded": self.excluded,
            "reason": self.reason,
        }


def detect_long_pass(
    from_x: float, from_y: float,
    to_x: float, to_y: float,
    event_type: str = "PASS",
    is_cross: bool = False,
    is_clearance: bool = False,
    is_throw_in: bool = False,
    is_airborne: bool = False,
    driven_low: bool = False,
) -> LongPassResult:
    """Classify a single delivery against the Opta/StatsBomb distance rule.

    Args:
        from_x/from_y:  kick point (origin).
        to_x/to_y:      where the delivery lands / is aimed.
        event_type:     the raw engine event label (informational only).
        is_cross:       True when the geometric cross detector already
                        stamped this delivery a Cross. Per Opta a qualifying
                        cross is logged exclusively as a Cross — NEVER a
                        long pass.
        is_clearance:   True for uncontrolled defensive boots. Engine
                        CLEARANCE events pass this; only controlled,
                        intentional long passes count.
        is_throw_in:    throw-ins are not passes (criterion exclusion).
        is_airborne:    trajectory lifted off the deck -> pass.height = High.
        driven_low:     a low, driven delivery -> pass.height = Low.

    Returns:
        LongPassResult — is_long_pass=True only when the raw Euclidean
        distance >= 35 yards AND the delivery is not excluded as a cross,
        clearance, or throw-in.
    """
    dist_m = pass_distance_m(from_x, from_y, to_x, to_y)
    dist_yd = pass_distance_yards(from_x, from_y, to_x, to_y)
    height = classify_pass_height(is_airborne, driven_low)

    # Exclusion 1 — throw-in: never a pass, never a long pass.
    if is_throw_in:
        return LongPassResult(
            False, dist_m, dist_yd, height,
            excluded="throw_in",
            reason=f"{event_type} is a throw-in — never a long pass",
        )

    # Exclusion 2 — crosses: satisfy the distance but are logged exclusively
    # as a Cross, never a long pass.
    if is_cross:
        return LongPassResult(
            False, dist_m, dist_yd, height,
            excluded="cross",
            reason=(
                f"{event_type} covers {dist_yd:.0f} yd ({dist_m:.1f} m) but is "
                f"a Cross — logged exclusively as a Cross, not a long pass"
            ),
        )

    # Exclusion 3 — uncontrolled clearances: only controlled, intentional
    # long distribution counts as a long pass.
    if is_clearance:
        return LongPassResult(
            False, dist_m, dist_yd, height,
            excluded="clearance",
            reason=f"{event_type} is an uncontrolled clearance — not a long pass",
        )

    # The metric threshold — the ONLY gate left.
    if dist_m >= LONG_PASS_THRESHOLD_M:
        return LongPassResult(
            True, dist_m, dist_yd, height,
            reason=f"covers {dist_yd:.0f} yd ({dist_m:.1f} m) >= 35 yd threshold",
        )

    return LongPassResult(
        False, dist_m, dist_yd, height,
        reason=f"covers {dist_yd:.0f} yd ({dist_m:.1f} m) < 35 yd threshold",
    )


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python long_pass_detector.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print("\n📐  PLOFA 26/27 — Long Pass Detector (Checkpoint 12) Standalone Demo")
    print("=" * 64)

    cases = [
        # (label, from, to, event_type, is_cross, is_clearance, is_throw_in)
        ("GK drop-kick to midfielder (right→)", (10, 34), (70, 34), "PASS", False, False, False),
        ("centre-back long diagonal switch",   (30, 10), (85, 55), "PASS", False, False, False),
        ("route-one pump to target man",       (15, 34), (65, 40), "PASS", False, False, False),
        ("40-yard cross into the box",         (75, 12), (99, 40), "CROSS_ATTEMPT", True, False, False),
        ("deep clearance (controlled?)",       (20, 34), (85, 45), "CLEARANCE", False, True, False),
        ("panicked defensive hoof",            (25, 30), (95, 25), "CLEARANCE", False, True, False),
        ("short sideways recycle",             (40, 34), (46, 36), "PASS", False, False, False),
        ("20 m forward ball (under bar)",      (40, 34), (59, 38), "PASS", False, False, False),
        ("long throw",                         (80, 2),  (95, 20), "THROW_IN", False, False, True),
        ("wing to far corner switch (left→)",  (60, 12), (15, 55), "PASS", False, False, False),
    ]
    print(f"\n  {'delivery':<44} {'LONG':<6} {'dist(yd)':<9} {'dist(m)':<8} {'excl':<9} reason")
    print("  " + "-" * 100)
    for label, fr, to, etype, xc, cl, thr in cases:
        r = detect_long_pass(*fr, *to, etype, xc, cl, thr)
        print(f"  {label:<44} {'LONG' if r.is_long_pass else 'short':<6}"
              f" {r.distance_yards:<9.0f} {r.distance_m:<8.1f}"
              f" {r.excluded or '-':<9} {r.reason}")

    print("\n   Threshold: >= 35 yd (%.1f m)" % LONG_PASS_THRESHOLD_M)
    print("   Crosses, clearances and throw-ins are excluded by Opta category.\n")
    print("✅ Long Pass Detector module operational — zero dependency on rest of PLOFA.\n")
