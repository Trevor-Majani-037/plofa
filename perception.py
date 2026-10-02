"""Perception layer — imperfect, attribute-dependent, role-specific world sensing.

Inserts BETWEEN ``extract_sensors`` (the exact world) and the brain, per
PLOFA V2 audit Step 1 (imperfect perception) and Step 5 (role-specific
blocks).  Produces a *perceived* 24-d vector from the exact one: range
limits, a forward-facing field of view, positional accuracy noise, and a
relevance top-k filter — all scaled by the player's DNA and role.

CRITICAL PROPERTIES
-------------------
- ``PerceptionConfig(enabled=False)`` returns the IDENTITY: ``perceive(...)``
  delegates straight to ``extract_sensors(...)`` and the array is
  byte-identical for v1 brains.  No behaviour change when off.
- DEFAULT (2026-09-13, user decision "real footballer"): the module default
  is now ``enabled=True, role_blocks=True`` — any runner inherits imperfect
  real-football perception unless a caller pins it.  Identity remains
  explicitly selectable via ``set_perception(PerceptionConfig(enabled=False))``
  or ``PLOFA_PERCEPTION=0``/``PLOFA_PERCEPTION_ROLES=0``.
- Degradation touches ONLY the other-actor geometry features (indices 4-9
  in the 24-d layout — the "omniscient" reads the audit flags).  Self
  (0-3), zone/match-state/DNA features (10-23) stay exact: those are the
  player's own knowledge, not perception of others.
- Role blocks (``role_blocks=True``): each position selects its own
  perception profile (range/FOV/noise scaling, relevance cap), so the
  same physical situation reads differently per role + DNA.  This is a
  *perceptual prior*, not an action rule — the brain chooses intent
  freely, nothing is hard-coded by role.
- Relevance top-k: after range/FOV gating, only the top-k most proximate
  actors (by distance) are kept.  k scales with role (strikers scan a
  smaller window; defenders/gk scan wide).  This implements audit H.2.1.
- Per-snapshot noise draw: positional offsets are drawn deterministically
  from a seed derived from (config.seed, snapshot identity).  Same
  snapshot + same seed → identical perceived vector; different situations
  → different noise (no fixed-bias degeneracy).
- No futurity: every value is derived from the CURRENT frame's geometry
  only.  An actor that is unseen sounds/reads exactly like an empty world
  (counts 0, nearest 1.0, best-forward 0) — perception never invents.
- Deterministic under a fixed seed: two perceive() calls with the same
  snapshot and same seed yield the same perceived vector, so gates and
  tests are reproducible.

Threaded through ``NeuralDecisionBrain.decide``; global switch via
``set_perception(...)`` / ``PLOFA_PERCEPTION`` env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from brain_sensors import extract_sensors, _get_attr


# ─────────────────────────────────────────────────────────────
# ROLE PROFILES (audit §H.2.8 / §I Step 5)
# ─────────────────────────────────────────────────────────────

# Each canonical position maps to a perception profile:
#   range_scale  — multiplier on the DNA-scaled perception radius
#   fov_deg      — horizontal field of view (degrees), centred forward
#   noise_scale  — multiplier on sigma (composure/anticipation shrinks
#                  sigma; role adjusts: wider-scan roles are more noise-
#                  tolerant, tight-scanning roles less so)
#   max_actors   — relevance top-k: after range/FOV filtering, keep at
#                  most this many combined defenders + teammates.  A
#                  striker sees fewer actors (sharper, narrower reads);
#                  a goalkeeper scans wide and deep.
#
# Positions not in this table get a neutral profile (1.0 / cfg.fov_deg /
# 1.0 / None = unlimited).  Canonicalization strips trailing digits:
# "CB1" → "CB", "CM2" → "CM".
ROLE_PROFILES = {
    "GK":  {"range_scale": 1.15, "fov_deg": 200.0, "noise_scale": 0.85, "max_actors": 12},
    "CB":  {"range_scale": 1.10, "fov_deg": 165.0, "noise_scale": 0.95, "max_actors": 10},
    "LB":  {"range_scale": 1.05, "fov_deg": 160.0, "noise_scale": 0.95, "max_actors": 9},
    "RB":  {"range_scale": 1.05, "fov_deg": 160.0, "noise_scale": 0.95, "max_actors": 9},
    "CDM": {"range_scale": 1.00, "fov_deg": 155.0, "noise_scale": 1.00, "max_actors": 9},
    "CM":  {"range_scale": 1.00, "fov_deg": 150.0, "noise_scale": 1.00, "max_actors": 8},
    "CAM": {"range_scale": 0.95, "fov_deg": 135.0, "noise_scale": 1.05, "max_actors": 7},
    "LW":  {"range_scale": 0.95, "fov_deg": 140.0, "noise_scale": 1.05, "max_actors": 7},
    "RW":  {"range_scale": 0.95, "fov_deg": 140.0, "noise_scale": 1.05, "max_actors": 7},
    "ST":  {"range_scale": 0.90, "fov_deg": 130.0, "noise_scale": 1.10, "max_actors": 6},
    "CF":  {"range_scale": 0.90, "fov_deg": 130.0, "noise_scale": 1.10, "max_actors": 6},
}


def _normalize_position(position: str) -> str:
    """Strip trailing digits from a player position: 'CB1' → 'CB'."""
    return position.strip().rstrip("0123456789")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class PerceptionConfig:
    """Tunables for the perception model.

    When ``enabled`` is False (default) :func:`perceive` returns the exact
    v1 ``extract_sensors`` array — byte-identical, no behaviour change.
    """
    enabled: bool = True
    # Range (metres) of reliable perception at the extremes of DNA vision.
    range_min: float = 8.0      # a very low-vision player barely perceives nearby actors
    range_max: float = 45.0     # elite vision stretches to midfield distances
    # Horizontal field of view (deg), centred on the forward direction
    # (toward the opponent goal).  Actor behind the player -> unseen.
    # Used as the default when role_blocks=False or position is unknown.
    fov_deg: float = 150.0
    # Accuracy: max positional sigma (metres) for a low mental-attribute
    # player; shrinks toward ~0 for elite vision+anticipation.
    noise_base: float = 3.0
    # Per-position role blocks: when True, each role selects its own
    # range/FOV/noise/top-k profile.  When False, the flat params above
    # apply (Step-1 behaviour).
    role_blocks: bool = True
    # Global top-k cap on total seen actors (defenders + teammates) after
    # range/FOV filtering.  Only used when role_blocks=False.  None means
    # unlimited (Step-1 behaviour).
    max_actors: Optional[int] = None
    # BALL-VISION (2026-09-20): off-ball players no longer get the TRUE
    # ball.  ``ball_vision=True`` gives them a PERCEIVED ball — exact when
    # inside their view cone / range (with distance noise), else a stale
    # "last known" estimate whose uncertainty grows with time out of sight.
    # The on-ball path is untouched (the ball is at the carrier's feet).
    ball_vision: bool = True
    # Maximum range (m) at which the ball is "seen" per snapshot (scaled by
    # the player's DNA vision/anticipation like actors).
    ball_radius: float = 60.0
    # Horizontal FOV (deg) centred on the forward direction — a ball behind
    # the player is NOT seen (same cone the actor gate uses).
    ball_fov_deg: float = 150.0
    # Per-snapshot positional sigma (m) at midfield distance for low mental
    # DNA; shrinks toward ~0 for elite vision/anticipation/composure.
    ball_noise: float = 2.5
    # Uncertainty growth while the ball is out of sight (m per simulated
    # second since it last entered the player's view).
    ball_stale_growth: float = 8.0
    # OFF-BALL ACTOR HONESTY (2026-09-20): the off-ball press brains used to
    # see all 21 other players at TRUE positions AND the true ball.  With
    # ``offball_actor_perception=True`` the actor lists fed to the off-ball
    # sensor vector are gated/degraded by the SAME cone/range/top-k/noise
    # rules the on-ball path uses (measured from the runner, facing the
    # opponent goal).  Disabled -> byte-identical to the pre-honesty feed.
    offball_actor_perception: bool = True
    # OFF-BALL MOVEMENT HONESTY (2026-09-20): the player's INDIVIDUAL
    # physic-y read of the ball (chase trigger/resustain + involvement
    # gate) runs off his PERSONAL perceived ball — a wrong belief means a
    # wasted sprint or a missed close-down.  The team's SHAPE compaction
    # stays on the true ball (real teams shift shape via communication).
    # Disabled -> byte-identical to the pre-honesty movement feed.
    ball_vision_movement: bool = True
    seed: int = 0

    @classmethod
    def from_env(cls) -> "PerceptionConfig":
        # 2026-09-13: production default is TRUE FOOTBALL — imperfect
        # role-specific perception like a real player sees.  Explicit env
        # values still work: PLOFA_PERCEPTION=0 / PLOFA_PERCEPTION_ROLES=0
        # restore the identity regime for harnesses that need it.
        def _flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return raw.lower() in ("1", "true", "yes", "on")
        enabled = _flag("PLOFA_PERCEPTION", True)
        role_blocks = _flag("PLOFA_PERCEPTION_ROLES", True)
        ball_vision = _flag("PLOFA_BALL_VISION", True)
        offball_actor_perception = _flag("PLOFA_OFFBALL_PERCEPTION", True)
        ball_vision_movement = _flag("PLOFA_BALL_VISION_MOVEMENT", True)
        return cls(enabled=enabled, role_blocks=role_blocks,
                   ball_vision=ball_vision,
                   offball_actor_perception=offball_actor_perception,
                   ball_vision_movement=ball_vision_movement)


_DEFAULT_CONFIG = PerceptionConfig.from_env()


def set_perception(config: PerceptionConfig) -> None:
    """Set the process-wide perception config (global brain switch)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config


def get_perception_config() -> PerceptionConfig:
    return _DEFAULT_CONFIG


def derive_match_seed(home: str, away: str, match_date, competition: str = "") -> int:
    """A stable per-match seed, derived from the match's identity.

    Why this exists
    ---------------
    ``PerceptionConfig.seed`` defaults to ``0`` and nothing ever changed it, so
    the noise a player gets when misreading the ball was byte-identical in every
    match of every season. The same striker, at the same minute, with the same
    staleness, produced the same misread against every opponent. That is not
    realistic, and it is not what the surrounding code intended — the draws are
    derived from a hash of the *situation*, so the design clearly expected the
    match to contribute a term.

    Seeding it from the fixture gives both properties at once:

    * the same fixture replayed produces the same noise (reproducible), and
    * different fixtures produce different noise (realistic variety).

    Why ``zlib.crc32`` and not ``hash()``
    -------------------------------------
    The existing draws use ``hash(ident)``, which for a ``str`` depends on
    ``PYTHONHASHSEED``. Set it to anything but 0 and the same match yields
    different noise — a silent reproducibility trap that only shows up on
    someone else's machine. ``crc32`` is stable everywhere, in every process,
    forever. New code should use it; the existing ``hash()`` sites are left
    alone deliberately, because changing them would alter 26/27 output.
    """
    import zlib
    ident = f"{competition}|{home}|{away}|{match_date}"
    return zlib.crc32(ident.encode("utf-8")) & 0xFFFFFFFF


def _resolve_profile(position: str, cfg: PerceptionConfig) -> dict:
    """Resolve per-role perception parameters.

    Returns a dict with keys: range_scale, fov_deg, noise_scale, max_actors.
    When role_blocks is False, returns the flat config defaults (no role
    override).  When role_blocks is True but the position is unknown, falls
    back to neutral (1.0 / cfg.fov_deg / 1.0 / cfg.max_actors).
    """
    if not cfg.role_blocks:
        return {
            "range_scale": 1.0,
            "fov_deg": cfg.fov_deg,
            "noise_scale": 1.0,
            "max_actors": cfg.max_actors,
        }
    canon = _normalize_position(position)
    profile = ROLE_PROFILES.get(canon, {})
    return {
        "range_scale": profile.get("range_scale", 1.0),
        "fov_deg": profile.get("fov_deg", cfg.fov_deg),
        "noise_scale": profile.get("noise_scale", 1.0),
        "max_actors": profile.get("max_actors", cfg.max_actors),
    }


# ─────────────────────────────────────────────────────────────
# PERCEIVED POSITION ENGINE (wraps the real engine)
# ─────────────────────────────────────────────────────────────

class _PerceivedEngine:
    """Delegates to the real position engine but returns noisy positions
    for known actors and (importantly) is PASSED through the SAME helper
    code paths brain_sensors uses, so the degradation is applied where
    distances/counts/openness are computed — feature-for-feature."""

    def __init__(self, engine: Any, offsets: dict[str, float],
                 seed: int):
        self._engine = engine
        self._offsets = offsets
        self._seed = seed

    def get_position(self, name: str):
        px, py = self._engine.get_position(name)
        off = self._offsets.get(name)
        if off is not None:
            px += off[0]
            py += off[1]
        return px, py


# ─────────────────────────────────────────────────────────────
# VISIBILITY GEOMETRY
# ─────────────────────────────────────────────────────────────

def _forward_angle(attacks_right: bool) -> float:
    """Bearing (radians) of the direction the player faces."""
    return 0.0 if attacks_right else np.pi


def _bearing_diff(from_x: float, from_y: float, to_x: float, to_y: float,
                  forward: float) -> float:
    """Signed angle between the player's forward direction and the vector
    from the player to ``(to_x, to_y)`` — used for the FOV cone."""
    ang = np.arctan2(to_y - from_y, to_x - from_x)
    diff = (ang - forward + np.pi) % (2 * np.pi) - np.pi
    return float(diff)


def _is_visible(from_x: float, from_y: float, to_x: float, to_y: float,
                attacks_right: bool, radius: float, fov_rad: float) -> bool:
    dist = np.hypot(to_x - from_x, to_y - from_y)
    if dist > radius:
        return False
    if abs(_bearing_diff(from_x, from_y, to_x, to_y,
                         _forward_angle(attacks_right))) > fov_rad / 2.0:
        return False
    return True


def _mental_scale(player: Any) -> tuple[float, float]:
    """Return (perception_power, accuracy_factor) from the player's DNA.

    ``perception_power`` in [0,1] scales the perception radius (vision +
    anticipation); ``accuracy_factor`` in [0,1] scales noise down (1 =
    elite accuracy ~ 0 noise)."""
    mental = getattr(getattr(player, "dna", None), "mental", None)
    vision = _clamp(_get_attr(mental, "vision", 60.0) / 100.0)
    anticipation = _clamp(_get_attr(mental, "anticipation", 60.0) / 100.0)
    composure = _clamp(_get_attr(mental, "composure", 60.0) / 100.0)
    power = 0.5 * vision + 0.5 * anticipation
    acc = 0.4 * vision + 0.3 * anticipation + 0.3 * composure
    return power, acc


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
# PERCEIVE
# ─────────────────────────────────────────────────────────────

def perceive(
    player: Any,
    x: float,
    y: float,
    teammates: List[Any],
    defenders: List[Any],
    position_engine: Any,
    under_pressure: bool,
    attacks_right: bool,
    game_state: Any,
    minute: float = 45.0,
    team_possession: bool = True,
    score_diff: int = 0,
    config: Optional[PerceptionConfig] = None,
    return_scene: bool = False,
):
    """24-d PERCEIVED sensor vector.  Disabled -> exact v1 identity.

    Same signature as ``brain_sensors.extract_sensors`` (plus ``config``)
    so it is a drop-in replacement at the decide() call site.

    When ``return_scene`` is True, returns ``(vector, scene)`` where
    ``scene`` = {"teammates", "defenders", "position_engine"} — the SAME
    perceived lists/engine the shared block was read through.  The role
    block (role_features.role_block) is then derived from that identical
    perceived world, so a v2 brain's shared 24-d and its role tail always
    agree about what the player can actually see.
    """
    cfg = config if config is not None else _DEFAULT_CONFIG
    if (not cfg.enabled) or position_engine is None:
        # IDENTITY MODE: byte-identical to v1 for backwards compatibility.
        # (Without a position engine there is no geometry to degrade and v1
        # already falls back to canned positions, so stay byte-identical.)
        vec = extract_sensors(
            player, x, y, teammates, defenders, position_engine,
            under_pressure, attacks_right, game_state, minute,
            team_possession=team_possession, score_diff=score_diff,
        )
        scene = {
            "teammates": teammates or [],
            "defenders": defenders or [],
            "position_engine": position_engine,
        }
        return (vec, scene) if return_scene else vec

    # ── Resolve role profile ─────────────────────────────────
    pos = getattr(player, "position", "")
    prof = _resolve_profile(pos, cfg)

    power, acc = _mental_scale(player)

    radius = (cfg.range_min + (cfg.range_max - cfg.range_min) * power) * prof["range_scale"]
    fov_rad = np.deg2rad(prof["fov_deg"])
    sigma = cfg.noise_base * (1.0 - acc) * prof["noise_scale"]

    def _truth(name):
        if position_engine is None:
            return None
        try:
            return position_engine.get_position(name)
        except Exception:
            return None

    def _visible_actor(a: Any, kind: str) -> bool:
        pos = _truth(a.name)
        if pos is None:
            return True  # no geometry available -> optimistically seen
        return _is_visible(x, y, pos[0], pos[1], attacks_right, radius, fov_rad)

    # Keep only actors the player can actually see right now.
    seen_defs = [d for d in (defenders or []) if _visible_actor(d, "def")]
    seen_tms = [t for t in (teammates or []) if _visible_actor(t, "tm")]

    # ── Relevance top-k (audit H.2.1) ──────────────────────
    k = prof["max_actors"]
    if k is not None and len(seen_defs) + len(seen_tms) > k:
        def _actor_dist(a):
            p = _truth(a.name)
            if p is None:
                return 9999.0
            return float(np.hypot(p[0] - x, p[1] - y))

        ranked = sorted(
            [(d, "def") for d in seen_defs] + [(t, "tm") for t in seen_tms],
            key=lambda pair: _actor_dist(pair[0]),
        )[:k]
        seen_defs = [a for a, kind in ranked if kind == "def"]
        seen_tms = [a for a, kind in ranked if kind == "tm"]

    # ── Per-snapshot noise draw ──────────────────────────────
    # Deterministic: same (seed, player, snapshot) → same offsets; different
    # situations → different draws.  Avoids the fixed-bias degeneracy of
    # using the same rng for every call.
    _ident = f"{getattr(player, 'name', '')}:{x:.3f}:{y:.3f}:{minute:.1f}:{len(seen_defs)}:{len(seen_tms)}"
    draw_seed = (cfg.seed * 2654435761 + hash(_ident)) & 0xFFFFFFFF
    rng = np.random.default_rng(draw_seed)

    offsets: dict[str, float] = {}
    for a in seen_defs + seen_tms:
        if sigma > 1e-9:
            offsets[a.name] = (float(rng.normal(0.0, sigma)),
                               float(rng.normal(0.0, sigma)))
        else:
            offsets[a.name] = (0.0, 0.0)
    noisy_engine = _PerceivedEngine(position_engine, offsets, cfg.seed)

    vec = extract_sensors(
        player, x, y, seen_tms, seen_defs, noisy_engine,
        under_pressure, attacks_right, game_state, minute,
        team_possession=team_possession, score_diff=score_diff,
    )
    scene = {
        "teammates": seen_tms,
        "defenders": seen_defs,
        "position_engine": noisy_engine,
    }
    return (vec, scene) if return_scene else vec


# ─────────────────────────────────────────────────────────────
# OFF-BALL ACTOR GATE (2026-09-20)
# ─────────────────────────────────────────────────────────────
# The off-ball press brains used to see all 21 other players' true
# positions.  This is the off-side counterpart of perceive(): the SAME
# cone/range/top-k/noise rules, measured from the runner (the player not on
# the ball) facing the opponent goal.  Disabled -> identity.

def perceive_offball_actors(
    player: Any,
    x: float,
    y: float,
    teammates: List[Any],
    defenders: List[Any],
    position_engine: Any,
    attacks_right: bool,
    config: Optional[PerceptionConfig] = None,
    return_meta: bool = False,
):
    """Gate + degrade the off-ball vision lists like the on-ball gate.

    Returns ``(seen_tms, seen_defs, engine_for_sensors)``.  When disabled
    (or no geometry) returns the ORIGINAL lists and engine — byte-identical
    to the pre-actor-honesty feed.  ``return_meta`` adds ``{"seen", "sigma",
    "radius"}`` for the diagnostic.
    """
    cfg = config if config is not None else _DEFAULT_CONFIG
    if (not cfg.enabled) or (not cfg.offball_actor_perception) \
            or position_engine is None:
        out = (teammates or [], defenders or [], position_engine)
        return (*out, {"seen": len(out[0]) + len(out[1]), "sigma": 0.0,
                       "radius": 0.0}) if return_meta else out

    pos = getattr(player, "position", "")
    prof = _resolve_profile(pos, cfg)
    power, acc = _mental_scale(player)
    radius = (cfg.range_min + (cfg.range_max - cfg.range_min) * power) * \
        prof["range_scale"]
    fov_rad = np.deg2rad(prof["fov_deg"])
    sigma = cfg.noise_base * (1.0 - acc) * prof["noise_scale"]

    def _truth(name):
        try:
            return position_engine.get_position(name)
        except Exception:
            return None

    def _visible(a):
        p = _truth(getattr(a, "name", ""))
        if p is None:
            return True  # no geometry -> optimistically seen
        return _is_visible(x, y, p[0], p[1], attacks_right, radius, fov_rad)

    seen_defs = [d for d in (defenders or []) if _visible(d)]
    seen_tms = [t for t in (teammates or []) if _visible(t)]

    k = prof["max_actors"]
    if k is not None and len(seen_defs) + len(seen_tms) > k:
        def _actor_dist(a):
            p = _truth(getattr(a, "name", ""))
            if p is None:
                return 9999.0
            return float(np.hypot(p[0] - x, p[1] - y))

        ranked = sorted(
            [(d, "def") for d in seen_defs] + [(t, "tm") for t in seen_tms],
            key=lambda pair: _actor_dist(pair[0]),
        )[:k]
        seen_defs = [a for a, kind in ranked if kind == "def"]
        seen_tms = [a for a, kind in ranked if kind == "tm"]

    ident = f"{getattr(player, 'name', '')}:{x:.3f}:{y:.3f}:{len(seen_defs)}:{len(seen_tms)}"
    draw_seed = (cfg.seed * 2654435761 + hash(ident)) & 0xFFFFFFFF
    rng = np.random.default_rng(draw_seed)

    offsets: dict[str, float] = {}
    for a in seen_defs + seen_tms:
        if sigma > 1e-9:
            offsets[a.name] = (float(rng.normal(0.0, sigma)),
                               float(rng.normal(0.0, sigma)))
        else:
            offsets[a.name] = (0.0, 0.0)
    noisy_engine = _PerceivedEngine(position_engine, offsets, cfg.seed) \
        if offsets else position_engine

    out = (seen_tms, seen_defs, noisy_engine)
    return (*out, {"seen": len(seen_tms) + len(seen_defs), "sigma": sigma,
                   "radius": radius}) if return_meta else out
