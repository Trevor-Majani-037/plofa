"""Ball-vision (2026-09-20): a real player's knowledge of the ball.

The engine feeds every off-ball decision the TRUE ball coordinates — as if
every player always knew exactly where the ball is.  A real player only
knows that if he can SEE it right now (his FOV cone + range, facing the
opponent goal like the actor gate), and otherwise plays off a "last known"
estimate whose uncertainty grows with every second it stays out of view.

This module replaces slots 0-1 of the off-ball vector with that perceived
ball.  Only the DECISION input becomes honest — the chase/close-down
geometry in the engine keeps the true ball, because a player pressing does
know the ball is in his zone.  The on-ball path is untouched (the ball is
at the carrier's feet; his own position IS the ball).

Deterministic per (seed, player, snapshot): the same situation draws the
same noise, exactly like the actor-position noise in perception.perceive.
Any missing geometry, malformed state or exception returns the TRUE ball —
ball-vision must NEVER kill a match.
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional, Tuple

import numpy as np

from perception import (get_perception_config, _bearing_diff,
                        _forward_angle, _mental_scale)


class _BallTrack:
    """Per-player last-known ball state."""

    __slots__ = ("x", "y", "t", "seen")

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.t = -1e9          # last second it was SEEN (match clock)
        self.seen = False


# engine -> {player_name: _BallTrack}.  WeakKey ensures per-match trackers
# are dropped once the engine is garbage collected (seasons run thousands).
_TRACK: Any = weakref.WeakKeyDictionary()


def clear_caches() -> None:
    """Drop all per-engine ball trackers (tests)."""
    _TRACK.clear()


def _tracks_for(engine: Any) -> Dict[str, _BallTrack]:
    try:
        return _TRACK.setdefault(engine, {})
    except TypeError:
        # Non-weakrefable engine (shouldn't happen) — degrade gracefully.
        return {}


def perceive_ball(
    engine: Any,
    pname: str,
    player: Any,
    attacks_right: bool,
    true_x: float,
    true_y: float,
    minute: float,
) -> Tuple[float, float, bool, float]:
    """The player's perceived ball ``(px, py, visible_now, sigma)``.

    When ball-vision is disabled (or geometry is missing / an exception
    occurs) returns the TRUE ball with ``visible_now=True`` — byte-identical
    to the pre-ball-vision engine.  ``sigma`` is the running uncertainty (m).
    """
    cfg = get_perception_config()
    if not cfg.ball_vision:
        return true_x, true_y, True, 0.0
    try:
        st = engine.position_engine.states.get(pname)
        if st is None:
            return true_x, true_y, True, 0.0
        rx, ry = st.current_x, st.current_y
    except Exception:
        return true_x, true_y, True, 0.0

    track = _tracks_for(engine).setdefault(pname, _BallTrack())

    # ── Visibility (same cone the actor gate uses) ───────────
    try:
        dist = float(np.hypot(true_x - rx, true_y - ry))
        fov_rad = np.deg2rad(float(cfg.ball_fov_deg))
        power, acc = _mental_scale(player)
        radius = float(cfg.ball_radius) * (0.5 + 0.5 * power)
        bearing = _bearing_diff(rx, ry, true_x, true_y,
                                _forward_angle(attacks_right))
        visible = (dist <= radius) and abs(bearing) <= fov_rad / 2.0
    except Exception:
        visible = True  # no geometry confidence -> optimistically seen

    if visible:
        # Distance-degraded read: further = fuzzier, elite mental = sharper.
        sigma = float(cfg.ball_noise) * (1.0 - acc) * (0.4 + 0.6 * dist / max(radius, 1.0))
        # Refresh the last-known state: while the ball is in sight the
        # tracker stays current, so staleness only accrues while blind.
        track.x, track.y, track.t, track.seen = true_x, true_y, minute, True
        return true_x, true_y, True, sigma

    # ── Out of sight: "last known" + growing uncertainty ────
    if not track.seen:
        # Nothing to remember yet — trust the ball until we've lost it once.
        track.x, track.y, track.t, track.seen = true_x, true_y, minute, True
        sigma = float(cfg.ball_noise) * (1.0 - acc)
        return true_x, true_y, True, sigma

    elapsed = max(0.0, minute - track.t)
    sigma = float(cfg.ball_noise) * (1.0 - acc) + (
        float(cfg.ball_stale_growth) * elapsed)
    # Deterministic draw per (player, snapshot, staleness bucket) so the
    # same recalled situation always says the same thing.
    ident = f"{pname}:{minute:.1f}:{elapsed:.1f}"
    rng = np.random.default_rng(
        (int(cfg.seed) * 2654435761 + hash(ident)) & 0xFFFFFFFF)
    px = track.x + float(rng.normal(0.0, sigma))
    py = track.y + float(rng.normal(0.0, sigma))
    return px, py, False, sigma


def movement_ball(
    engine: Any,
    pname: str,
    position: str,
    attacks_right: bool,
    true_x: float,
    true_y: float,
    minute: float,
) -> Tuple[float, float]:
    """The (px, py) an off-ball player's INDIVIDUAL physics acts on.

    This is the single seam the engine's movement integrator calls every
    tick (chase trigger/resustain + involvement gate).  It is deliberately
    NARROW: only the player's own physic-y read of the ball — team SHAPE
    compaction keeps the true ball (real teams shift shape by voice).

    Disabled (PLOFA_BALL_VISION=0 or PLOFA_BALL_VISION_MOVEMENT=0, or any
    geometry error) returns the TRUE ball so the movement feed is
    byte-identical to the pre-honesty engine.
    """
    cfg = get_perception_config()
    if not (cfg.ball_vision and cfg.ball_vision_movement):
        return true_x, true_y
    try:
        from types import SimpleNamespace
        view = SimpleNamespace(name=pname, position=position)
        px, py, _seen, _sigma = perceive_ball(
            engine, pname, view, attacks_right, true_x, true_y, minute)
        return float(px), float(py)
    except Exception:
        return true_x, true_y