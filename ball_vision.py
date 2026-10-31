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
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from perception import (get_perception_config, _bearing_diff,
                        _facing_angle,
                        _mental_scale, _tunnel_vision_fov,
                        _peripheral_glimpse)


class _BallTrack:
    """Per-player last-known ball state, with head-scan timing.

    A player does not stare rigidly forward.  At a cadence set by his
    mental attributes (vision + anticipation) he sweeps his head to
    re-check blind spots.  The scan widens his effective FOV for a brief
    window, and if the ball is found the ``last_seen`` clock resets so
    uncertainty stays low.
    """

    __slots__ = ("x", "y", "t", "seen", "last_scan_time", "scan_duration",
                 "in_scan", "scan_busy_until")

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.t = -1e9          # last second it was SEEN (match clock)
        self.seen = False
        # --- SCAN STATE (2026-10-27) ---
        self.last_scan_time: float = -1e9   # match-clock of the last head sweep
        self.scan_duration: float = 0.35    # seconds a scan lasts (quick shoulder-check)
        self.in_scan: bool = False          # currently mid-sweep
        self.scan_busy_until: float = -1e9  # match-clock until responsiveness returns


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

    # ── Visibility (head-turn + scan-aware + peripheral) (2026-10-27) ──
    # Previously this used the constant `_forward_angle`, so a defender
    # with the ball behind him was permanently blind and never checked his
    # shoulder.  Now the full human perception stack:
    #   1. Head turns toward the ball when it's behind (shoulder check).
    #   2. Periodic full scan sweeps widen effective FOV for ~0.35 s.
    #   3. Peripheral vision catches movement outside the central FOV.
    #   4. Fatigue degrades scan cadence, perception power, and accuracy.
    #   5. Sprinting narrows the FOV (tunnel vision).
    #   6. Scanning costs a brief window of reduced forward responsiveness.
    try:
        dist = float(np.hypot(true_x - rx, true_y - ry))
        base_fov_deg = float(cfg.ball_fov_deg)
        # Fatigue (0-1): derived from stamina attribute + match minute so a
        # player deep in the second half with poor stamina perceives worse.
        _dna = getattr(player, "dna", None)
        _phys = getattr(_dna, "physical", None)
        _stamina = float(getattr(_phys, "stamina", 65.0))
        _stamina_norm = max(0.0, min(1.0, _stamina / 100.0))
        _match_frac = max(0.0, min(1.15, minute / 90.0))
        fatigue = max(0.0, min(1.0, _match_frac * (1.35 - _stamina_norm)))
        # Approximate sprint speed from spatial state velocity if available.
        st_obj = engine.position_engine.states.get(pname)
        spd = float(np.hypot(
            getattr(st_obj, "velocity_x", 0.0),
            getattr(st_obj, "velocity_y", 0.0),
        )) if st_obj is not None else 0.0
        # Sprint + pressure tunnel vision
        fov_deg = _tunnel_vision_fov(base_fov_deg, spd)
        fov_rad = np.deg2rad(fov_deg)
        # Mental scale with fatigue degradation
        power, acc = _mental_scale(player, fatigue=fatigue)
        radius = float(cfg.ball_radius) * (0.5 + 0.5 * power)
        # Per-player head-turn toward the LAST-KNOWN ball position, not
        # the true position.  A player who has seen the ball before turns
        # his head toward where he REMEMBERS it being — trying to re-acquire
        # it.  A player who has never seen it faces forward (attacking
        # direction) by default.  This is critical: the true ball at 169°
        # behind is unreachable by neck turn; the last-known ball (which is
        # at a more recent, closer position) is a realistic look target.
        _look_x = track.x if track.seen else true_x
        _look_y = track.y if track.seen else true_y
        _head_facing = _facing_angle(player, rx, ry,
                                      _look_x, _look_y, attacks_right)
        bearing = _bearing_diff(rx, ry, true_x, true_y, _head_facing)
        visible = (dist <= radius) and abs(bearing) <= fov_rad / 2.0
        # ── Peripheral glimpse: even outside the central FOV, the
        # periphery catches movement.  Checked BEFORE the scan cadence
        # because it is passive (no head movement required).
        # Uses a deterministic RNG derived from the staleness draw seed.
        if not visible:
            _glimpse_rng = np.random.default_rng(
                (int(cfg.seed) * 2654435761
                 + zlib.crc32(f"{pname}:glimpse:{minute:.0f}".encode("utf-8")))
                & 0xFFFFFFFF)
            visible = _peripheral_glimpse(
                bearing, fov_rad, power, rng=_glimpse_rng)
        # ── Scan cadence: periodic head sweeps to check blind spots.
        # Fatigue slows the cadence: a gassed player scans less often.
        fatigue_scan_mult = 1.0 - 0.50 * fatigue  # 0.5x at max fatigue
        scan_interval = 1.0 / max(0.05, (0.2 + power * 0.6) * fatigue_scan_mult)
        now = minute
        if (not visible) and (now - track.last_scan_time >= scan_interval):
            # Start a scan: wider FOV sweep, costs forward responsiveness.
            track.last_scan_time = now
            track.in_scan = True
            track.scan_busy_until = now + track.scan_duration
            # During a scan the player sweeps his head in an arc, checking
            # both shoulders.  The scan checks three bearings: the current
            # head-facing direction, plus separate left and right extreme
            # sweeps (each capped by the max head-turn limit).  If the ball
            # is visible at ANY of these bearings, the scan finds it.
            scan_fov_rad = np.deg2rad(180.0)  # full peripheral sweep during scan
            scan_fov_half = scan_fov_rad / 2.0
            # Maximum head turn from the forward axis that the player can
            # achieve during a deliberate scan sweep.
            _max_turn = np.deg2rad(80.0)
            _forward = 0.0 if attacks_right else np.pi
            _look_toward = np.arctan2(_look_y - ry, _look_x - rx)
            _look_diff = (_look_toward - _forward + np.pi) % (2 * np.pi) - np.pi
            # Sweep: the player checks in three directions — a left extreme,
            # a right extreme, and the natural head-turn direction.
            _sweep_angles = []
            # Natural turn (toward last-known ball)
            _natural_turn = np.clip(_look_diff, -_max_turn, _max_turn)
            _sweep_angles.append(_forward + _natural_turn)
            # Left extreme (full left shoulder check)
            _sweep_angles.append(_forward - _max_turn)
            # Right extreme (full right shoulder check)
            _sweep_angles.append(_forward + _max_turn)
            # Also check the pure forward direction (the player might have
            # been looking away from the ball and swings through forward)
            if abs(_natural_turn) > 0.5:
                _sweep_angles.append(_forward)
            for _sw in _sweep_angles:
                _sw_bearing = _bearing_diff(rx, ry, true_x, true_y, _sw)
                if abs(_sw_bearing) <= scan_fov_half:
                    visible = True
                    break
        elif track.in_scan and (now - track.last_scan_time >= track.scan_duration):
            track.in_scan = False  # scan sweep complete
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
    #
    # ``zlib.crc32`` not ``hash()``: a ``hash()`` of a str is salted by
    # PYTHONHASHSEED, so the same fixture replayed under a different hash seed
    # recalled the ball differently and produced a different match. Measured:
    # identical seed + varied PYTHONHASHSEED gave three different matches;
    # pinning =0 gave three identical ones. crc32 is stable in every process,
    # forever -- the same discipline position_engine.py:2481, world/squads.py
    # and world/proof.py already follow.
    ident = f"{pname}:{minute:.1f}:{elapsed:.1f}"
    rng = np.random.default_rng(
        (int(cfg.seed) * 2654435761
         + zlib.crc32(ident.encode("utf-8"))) & 0xFFFFFFFF)
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