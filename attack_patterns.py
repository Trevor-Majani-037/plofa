"""
PLOFA 26/27 — ATTACK PATTERN LIBRARY
====================================
attack_patterns.py

Feature #2: team-specific attack patterns.

AttackingMatrix decides the *intent* of a sequence (shoot / key pass /
progressive pass / recycle) from the team's style, for a generic counter /
low-block / build-up / balanced scenario. What is missing is WHICH pattern of
play the team leans on to actually construct that intent — the recurring
attacking identities Opta writes about: Pep's box-midfield overloads,
Klopp's wing isolation, United's right-side overload then switch, a low-
block/spicy-counter outfit's striker-on-the-shoulder channels.

Each pattern carries:
  - role home-position deltas (stacked on the formation + stance shape so the
    OFF-BALL XI actually moves into the pattern — e.g. the box-midfield CM
    steps into the pocket behind the striker),
  - a favoured flank bias ("R" / "L" / None) consumed by the possession-phase
    engine (switch the play to the far overload when the near lane dies) and
    by receiver weighting (the ball gravitates to the pattern's side).

Selection is deterministic PER 5-MINUTE CHUNK (seeded by team + chunk) so a
team commits to a pattern for a stretch instead of flickering every sequence;
the map can also be overridden by the chasing/protecting shape (a chasing
team leans direct-channel, a leading team leans box-midfield).

PURE-DATA MODULE: engine imports are lazy (inside functions) to avoid import
cycles, identical to tactical_shapes.py.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, Optional, Tuple

AllRoles = ("GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF")


class AttackPattern(Enum):
    NONE = "none"
    # Central numerical superiority: a double-pivot CDM, a CM pushed into
    # the 10-pocket beside the CAM, wide players provide width. City's box.
    BOX_MIDFIELD = "box_midfield"
    # Deliberately crowd one flank (FB overlaps, CAM/CM lean over, the far
    # winger stands wide) to CREATE the 3v2 — then the far-side switch.
    OVERLOAD_RIGHT = "overload_right"
    OVERLOAD_LEFT = "overload_left"
    # Establish a 1v1 for one winger on the touchline, full-back underlaps /
    # supports, striker attacks the near post.
    WING_ISOLATION_RIGHT = "wing_isolation_right"
    WING_ISOLATION_LEFT = "wing_isolation_left"
    # Direct: striker hangs on the shoulder, wingers invert into the channels,
    # midfield hit it early. Counter/low-block outfit's out-ball identity.
    DIRECT_CHANNELS = "direct_channels"


# ─────────────────────────────────────────────
# PATTERN ROLE HOME DELTAS (dx forward m, dy flank m)
# ─────────────────────────────────────────────
# Same attacking-right normalised convention as tactical_shapes.py:
#   +x forward, low y = own-left channel, high y = own-right channel.
# Position deltas are additions on top of the formation + stance shape.

_PATTERN_ROLE_DELTAS: Dict[AttackPattern, Dict[str, Tuple[float, float]]] = {
    AttackPattern.NONE: {},
    AttackPattern.BOX_MIDFIELD: {
        "CDM": (-2.0,  0.0),   # the pivot sits
        "CM":  (6.0,  0.0),    # CM climbs into the 10-pocket
        "CAM": (2.0,  0.0),    # stays in the left pocket of the box
        "ST":  (0.0,  0.0),
        "CF":  (1.0,  0.0),
        "LW":  (3.0, -0.5),    # width
        "RW":  (3.0,  0.5),
        "LB":  (8.0,  0.5),    # full-backs advance for the width
        "RB":  (8.0, -0.5),
        "CB":  (1.0,  0.0),
    },
    # Overload the RIGHT: ball-side FB climbs, CAM/CM/CDM all lean over,
    # the far-side (left) winger inverts into the half-space, near RW anchors
    # the touchline as both the switch outlet and a box crasher.
    AttackPattern.OVERLOAD_RIGHT: {
        "CDM": (0.0,  3.0),
        "CM":  (0.0,  5.0),
        "CAM": (3.0,  6.0),
        "ST":  (2.0,  0.5),
        "CF":  (2.0,  0.5),
        "LW":  (0.0,  5.0),    # far winger inverts to form the overload-ish 3
        "RW":  (5.0,  0.5),    # near winger anchors the touchline
        "LB":  (-4.0, 8.0),    # far FB tucks in (counter cover + box support)
        "RB":  (16.0, 0.0),    # ball-side FB bombs up the outside
    },
    # Mirror image on the LEFT.
    AttackPattern.OVERLOAD_LEFT: {
        "CDM": (0.0, -3.0),
        "CM":  (0.0, -5.0),
        "CAM": (3.0, -6.0),
        "ST":  (2.0, -0.5),
        "CF":  (2.0, -0.5),
        "LW":  (5.0, -0.5),    # near winger anchors own touchline
        "RW":  (0.0, -5.0),    # far winger inverts
        "LB":  (16.0, 0.0),    # ball-side FB bombs up
        "RB":  (-4.0, -8.0),   # far FB tucks in
    },
    AttackPattern.WING_ISOLATION_RIGHT: {
        "CDM": (0.0,  0.0),
        "CM":  (0.0,  3.0),
        "CAM": (2.0,  4.0),
        "ST":  (2.0,  2.0),    # near-post run
        "CF":  (2.0,  2.0),
        "LW":  (2.0,  6.0),    # far winger comes narrow for the cut-back
        "RW":  (6.0,  0.5),    # isolated 1v1 anchor on the touchline
        "LB":  (-2.0, 8.0),    # far FB tucks
        "RB":  (14.0, 0.0),    # underlap/support runner
    },
    AttackPattern.WING_ISOLATION_LEFT: {
        "CDM": (0.0,  0.0),
        "CM":  (0.0, -3.0),
        "CAM": (2.0, -4.0),
        "ST":  (2.0, -2.0),
        "CF":  (2.0, -2.0),
        "LW":  (6.0, -0.5),
        "RW":  (2.0, -6.0),
        "LB":  (14.0, 0.0),
        "RB":  (-2.0, -8.0),
    },
    AttackPattern.DIRECT_CHANNELS: {
        "CDM": (0.0,  0.0),
        "CM":  (2.0,  0.0),
        "CAM": (5.0,  0.0),
        "ST":  (4.0,  0.0),    # hangs on the shoulder
        "CF":  (4.0,  0.0),
        "LW":  (1.0,  4.0),    # invert: wingers attack the channels
        "RW":  (1.0, -4.0),
        "LB":  (0.0,  0.0),    # full-backs hold
        "RB":  (0.0,  0.0),
    },
}


def pattern_role_deltas(pattern: AttackPattern) -> Dict[str, Tuple[float, float]]:
    """Per-role home deltas for a pattern (empty for NONE)."""
    return _PATTERN_ROLE_DELTAS.get(pattern, {})


def favored_flank(pattern: Optional[AttackPattern]) -> Optional[str]:
    """Which flank (in normalised terms, "R" = own-right, "L" = own-left)
    the pattern leans on for its overload / isolation, or None."""
    return {
        AttackPattern.OVERLOAD_RIGHT: "R",
        AttackPattern.OVERLOAD_LEFT: "L",
        AttackPattern.WING_ISOLATION_RIGHT: "R",
        AttackPattern.WING_ISOLATION_LEFT: "L",
    }.get(pattern)


# Style → candidate pattern pools. A team's identity determines WHICH
# patterns it favours; the chunked RNG picks between them.
_STYLE_PATTERN_POOLS: Dict[str, Tuple[AttackPattern, ...]] = {
    "tiki_taka": (AttackPattern.BOX_MIDFIELD, AttackPattern.OVERLOAD_RIGHT,
                  AttackPattern.OVERLOAD_LEFT),
    "structured_possession": (AttackPattern.BOX_MIDFIELD,
                              AttackPattern.OVERLOAD_RIGHT,
                              AttackPattern.OVERLOAD_LEFT),
    "vertical_tiki_taka": (AttackPattern.DIRECT_CHANNELS,
                           AttackPattern.BOX_MIDFIELD,
                           AttackPattern.OVERLOAD_RIGHT),
    "attacking": (AttackPattern.OVERLOAD_RIGHT, AttackPattern.OVERLOAD_LEFT,
                  AttackPattern.BOX_MIDFIELD, AttackPattern.WING_ISOLATION_RIGHT),
    "ultra_attacking": (AttackPattern.OVERLOAD_RIGHT, AttackPattern.OVERLOAD_LEFT,
                        AttackPattern.WING_ISOLATION_LEFT,
                        AttackPattern.DIRECT_CHANNELS),
    "wing_play": (AttackPattern.WING_ISOLATION_RIGHT, AttackPattern.WING_ISOLATION_LEFT,
                  AttackPattern.OVERLOAD_RIGHT, AttackPattern.OVERLOAD_LEFT),
    "balanced": (AttackPattern.BOX_MIDFIELD, AttackPattern.OVERLOAD_RIGHT,
                 AttackPattern.WING_ISOLATION_LEFT, AttackPattern.DIRECT_CHANNELS),
    "gegenpressing": (AttackPattern.OVERLOAD_RIGHT, AttackPattern.OVERLOAD_LEFT,
                      AttackPattern.WING_ISOLATION_RIGHT, AttackPattern.DIRECT_CHANNELS),
    "defensive": (AttackPattern.DIRECT_CHANNELS, AttackPattern.BOX_MIDFIELD),
    "ultra_defensive": (AttackPattern.DIRECT_CHANNELS,),
    "park_the_bus": (AttackPattern.DIRECT_CHANNELS,),
    "fluid_counter": (AttackPattern.DIRECT_CHANNELS, AttackPattern.WING_ISOLATION_LEFT),
    "route_one": (AttackPattern.DIRECT_CHANNELS,),
    "direct": (AttackPattern.DIRECT_CHANNELS, AttackPattern.BOX_MIDFIELD),
}


def pattern_for(
    style_name: str,
    state,
    team_name: str,
    home_team: str,
    minute: int,
    chasing: bool = False,
    protecting: bool = False,
) -> AttackPattern:
    """Deterministic, chunk-stable pattern selection for one team+minute.

    Seeds a throwaway RNG from (team, minute // CHUNK) so the chosen pattern
    holds for a ~5-minute block (a team commits to an overload for a real
    stretch) yet rotates across the half. Game-state overrides are COMMITS, in
    line with a real manager: a side chasing late abandons identity and
    blasts direct channels; a side protecting a lead keeps it on the box
    (low-risk, controlled central possession).
    `style_name` is the value of TeamStyle (e.g. "tiki_taka")."""

    if chasing:
        return AttackPattern.DIRECT_CHANNELS
    if protecting:
        return AttackPattern.BOX_MIDFIELD

    pool = _STYLE_PATTERN_POOLS.get(style_name, (AttackPattern.BOX_MIDFIELD,))
    chunk = minute // 5
    rng = random.Random(f"{team_name}|{home_team}|{chunk}")
    return pool[rng.randrange(len(pool))]


def flank_bias_multiplier(
    receiver_y: float,
    attacks_right: bool,
    favored_flank: Optional[str],
    near: float = 1.18,
    far: float = 0.88,
) -> float:
    """Receiver-weight multiplier from the pattern's favoured flank.

    Normalises `receiver_y` to attack direction (own-right = high normalised
    y), then: on the favoured flank a mild boost, on the far flank a mild
    penalty, neutral/unknown 1.0. Strength is modest so the existing
    orbital / channel / offside weighting keeps their authority — this only
    tips the tie when several outlets are comparable (the real feature of an
    overload: the ball LIVES on one side until the switch)."""
    if not favored_flank:
        return 1.0
    ny = receiver_y if attacks_right else 68.0 - receiver_y
    if favored_flank == "R":
        return near if ny >= 34.0 else far
    return near if ny < 34.0 else far