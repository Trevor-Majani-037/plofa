"""ENTITY TOKENS — the scene as a set of things to ATTEND to.

WHY THIS EXISTS
---------------
The 24-d shared block is a HAND-POOLED summary of the scene: "the nearest
defender is 6 m away", "the best forward teammate is 0.4 open", "four
defenders are within 10 m".  Every one of those is a number the code chose
to compute.  The network never sees WHO it is looking at.

That is the gap attention fills.  A player deciding between two passes
needs to know that one of the options is a striker in the box and the
other is a full-back in his own half — and the pooled scalars cannot say
that, because they were computed before anybody chose what to look at.

So this module turns the scene into a SET of entity tokens, one per
player, and ``situational_brain`` lets the network ATTEND to that set.

THE TWELVE FEATURES, AND WHY EACH ONE IS THERE
-----------------------------------------------
=====  ==================================================================
 0    signed FORWARD distance from the ball carrier (attacking direction)
 1    lateral offset from the centre line
 2    distance from the ball carrier
 3    is a TEAMMATE
 4    is an OPPONENT
 5    SPACE — for a teammate, the gap to his nearest opponent; for an
      opponent, the gap to his nearest team-mate (how tightly he is
      marked).  One number, two meanings, both about "can this man be
      used / can this man be passed".
 6    AHEAD of the carrier
 7    WIDE (more than 17 m off the centre line)
 8/9/10  coarse ROLE class: defender / midfielder / forward
 11   is the NEAREST OPPONENT — the man actually pressing, which is the
      single most important opponent to attend to
=====  ==================================================================

Roles are COARSE on purpose.  A fine role code (twelve positions) would
be a 12-slot one-hot per entity, which is 12 numbers describing a label
the team already knows; three numbers saying "defender / midfielder /
forward" carry the part that changes a decision.

WHICH ENTITIES ARE IN THE SET
-----------------------------
The nearest EIGHT teammates and the nearest EIGHT opponents — so both
kinds are always represented, and the network's choice of whom to attend
to is not pre-filtered by a distance rule that favours one side.  A
distance cap of 16 tokens keeps the attention O(n) per decision.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from situational_brain import N_MAX, TOKEN_D


#: How many of each kind make the set.
_PER_KIND = N_MAX // 2

#: The coarse role classes.
_ROLE_DEF = ("GK", "CB", "LB", "RB")
_ROLE_MID = ("CDM", "CM", "CAM")
_ROLE_FWD = ("ST", "CF", "LW", "RW")

#: The pitch, for normalisation.
_PITCH_X = 105.0
_PITCH_Y = 68.0
_CENTRE_Y = 34.0


def _position_of(p: Any) -> str:
    """The player's role, with the digits stripped (``CM1`` -> ``CM``).

    The digits matter: the live roster names its players ``CM1``/``CB2``
    and a guard written against ``"CM"`` would match nothing.  This is the
    same rule ``role_features._role_family`` follows.
    """
    raw = getattr(p, "position", "") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    return raw.rstrip("0123456789").upper()


def _role_class(pos: str) -> Tuple[float, float, float]:
    if pos in _ROLE_DEF:
        return 1.0, 0.0, 0.0
    if pos in _ROLE_MID:
        return 0.0, 1.0, 0.0
    if pos in _ROLE_FWD:
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


def _pos_of(p: Any, position_engine: Any, fallback: Tuple[float, float]
            ) -> Tuple[float, float]:
    """Where a player is, from the position engine when it knows."""
    name = getattr(p, "name", None)
    if name and position_engine is not None:
        try:
            tracked = position_engine.tracked_position(name)
            if tracked is not None:
                return float(tracked[0]), float(tracked[1])
        except Exception:
            pass
    try:
        px, py = position_engine.get_position(name) if name else (0.0, 0.0)
        return float(px), float(py)
    except Exception:
        return float(fallback[0]), float(fallback[1])


def _nearest_opponent_gap(px: float, py: float, opponents: List[Any],
                          position_engine: Any) -> float:
    """How much room this man has — the gap to his nearest marker."""
    best = None
    for o in opponents:
        ox, oy = _pos_of(o, position_engine, (px, py))
        d = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5
        if best is None or d < best:
            best = d
    return 15.0 if best is None else best


def entity_tokens(player: Any, x: float, y: float,
                  teammates: List[Any], defenders: List[Any],
                  position_engine: Any, attacks_right: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Build the attention's entity set.

    Returns ``(tokens, mask)`` with shapes (N_MAX, TOKEN_D) and (N_MAX,).
    The mask is 1.0 for a real entity and 0.0 for padding — and the
    attention gives a padded token NO mass, so padding cannot dilute the
    attention.
    """
    tokens = np.zeros((N_MAX, TOKEN_D), dtype=np.float64)
    mask = np.zeros(N_MAX, dtype=np.float64)

    def collect(players: List[Any], is_teammate: bool) -> List[Any]:
        rows = []
        for p in players or []:
            if p is player:
                continue
            px, py = _pos_of(p, position_engine, (x, y))
            d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            rows.append((d, p, px, py))
        rows.sort(key=lambda r: r[0])
        return rows[:_PER_KIND]

    mates = collect(teammates, True)
    foes = collect(defenders, False)

    # The nearest opponent is the presser — worth naming explicitly,
    # because "who is on me" is not the same question as "who is nearest".
    presser = foes[0][1] if foes else None

    row = 0
    for group, is_teammate in ((mates, True), (foes, False)):
        others = foes if is_teammate else mates
        for _d, p, px, py in group:
            if row >= N_MAX:
                break
            dx = (px - x) if attacks_right else (x - px)
            dy = py - y
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            gap = _nearest_opponent_gap(px, py, others, position_engine)
            pos = _position_of(p)
            rd, rm, rf = _role_class(pos)
            tokens[row] = [
                max(-1.5, min(1.5, dx / 60.0)),
                max(-1.5, min(1.5, dy / 34.0)),
                max(0.0, min(1.5, dist / 60.0)),
                1.0 if is_teammate else 0.0,
                0.0 if is_teammate else 1.0,
                max(0.0, min(1.5, gap / 15.0)),
                1.0 if dx > 0.0 else 0.0,
                1.0 if abs(py - _CENTRE_Y) > 17.0 else 0.0,
                rd, rm, rf,
                1.0 if (presser is not None and p is presser) else 0.0,
            ]
            mask[row] = 1.0
            row += 1

    return tokens, mask
