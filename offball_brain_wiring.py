"""Off-ball press-gate brain -> engine wiring (Phase 10, 2026-09-20).

The engine's per-player press Bernoulli (match_engine._offball_move_player)
currently uses a static role rate scaled by the shared TeamPressBrain
commitment:  prob = _PRESS_PROB[pos] * g.  This module slots the evolved,
per-position OFF-BALL brains (brains_offball/LW.json, RW.json, CM.json —
kind "offball_press_gate") into that Bernoulli: for a position WITH an
evolved brain the press probability is the brain's own p(press) read from
the same 24-d off-ball vector the brains were evolved/validated on, still
scaled by the team-level engagement brain so the unit keeps pressing
together:

    prob = brain_p * g          # position brain present
    prob = _PRESS_PROB[pos] * g # position brain absent (unchanged)

Positions without an evolved brain (ST/CB/LB/RB/CDM/GK) fall through to
today's behaviour, so the change is exact for the LW/RW/CM band only.

All reads are defensive: any missing file, malformed state or exception
returns None and the engine keeps the static Bernoulli — an off-ball
brain must NEVER kill a match.
"""

from __future__ import annotations

import os
import weakref
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAINS_DIR = os.path.join(_HERE, "brains_offball")
_BRAIN_CACHE: Dict[str, Optional[Any]] = {}
#: id(engine) -> (weakref to that engine, its view maps). The weakref is what
#: makes a recycled address safe; see _build_views for why.
_VIEW_CACHE: Dict[int, Any] = {}


class _View:
    """Minimal player-like object carrying the fields sensors read."""

    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position
        self.dna = None  # DNA slots default to 55/100 in the sensor layer


def clear_caches() -> None:
    """Drop the per-position brain + per-engine view caches (tests)."""
    _BRAIN_CACHE.clear()
    _VIEW_CACHE.clear()


def _load_position_brain(position: str) -> Optional[Any]:
    """Lazily load brains_offball/{position}.json (kind offball_press_gate)."""
    if position not in _BRAIN_CACHE:
        brain = None
        path = os.path.join(_BRAINS_DIR, f"{position}.json")
        try:
            if os.path.exists(path):
                from football_brain import OffBallBrain
                loaded = OffBallBrain.load(path)  # schema-validated internally
                if not isinstance(loaded, OffBallBrain):
                    raise TypeError("wrong brain class")
                brain = loaded
        except Exception:
            brain = None
        _BRAIN_CACHE[position] = brain
    return _BRAIN_CACHE.get(position)


def _build_views(engine: Any) -> Dict[str, Dict[str, Any]]:
    """One team -> name -> view map per engine instance (mirrors the
    offball_probe collector — prefers live players, falls back to a
    name/position view for roster-only names).

    Why this is keyed by weakref and not by ``id(engine)``
    -----------------------------------------------------
    ``id()`` is a memory ADDRESS, and CPython recycles addresses as soon as an
    object is freed. A season loop is precisely the pattern that triggers it:
    build engine -> simulate -> discard -> build engine. When the new engine
    lands on the dead one's address, an ``id``-keyed cache serves it the
    PREVIOUS match's player objects, positions and states.

    That is a cross-match contamination bug, not a theoretical one: the second
    match reads the first match's world. It showed up as a live match failing
    to replay from a fixed seed, with different players processed at the same
    point in the sequence.

    The fix keeps the cheap integer key but pairs every entry with a weak
    reference to the engine that created it. A recycled address now finds a
    dead weakref, counts as a miss, and rebuilds — so an entry can only ever be
    served to the exact live engine that made it.
    """
    key = id(engine)
    hit = _VIEW_CACHE.get(key)
    if hit is not None:
        engine_ref, maps = hit
        if engine_ref() is engine:        # same live engine
            return maps
        # address was recycled: drop the corpse and fall through to rebuild
        del _VIEW_CACHE[key]

    maps = {}
    for team, names in engine.position_engine.team_rosters.items():
        real = {}
        for p in engine.active_players.get(team, []):
            real[getattr(p, "name", None)] = p
        maps[team] = {}
        for n in names:
            st = engine.position_engine.states.get(n)
            pos = getattr(st, "position", "") if st is not None else ""
            maps[team][n] = real.get(n) or _View(n, pos)
    try:
        _VIEW_CACHE[key] = (weakref.ref(engine), maps)
    except TypeError:
        # engine is not weak-referenceable: serve the build without caching
        # rather than risk handing a recycled address someone else's maps.
        pass
    return maps


def offball_press_prob(engine: Any, pname: str, position: str, team: str,
                       ball_x: float, ball_y: float, danger_t: float,
                       minute: float, score_diff: int) -> Optional[float]:
    """p(press) from the evolved position brain, or None when unavailable.

    Returns the brain's raw sigmoid p(press) for this runner in this live
    state.  The caller multiplies it by the team commitment g and keeps
    the deterministic fallback otherwise.

    Since 2026-09-20 the slot-0/1 ball the brain reads is the runner's
    PERCEIVED ball (ball_vision): exact when inside his FOV/range, else his
    stale "last known" estimate.  True coords are only used for the chase
    geometry downstream, never for the decision input.
    """
    brain = _load_position_brain(position)
    if brain is None:
        return None
    try:
        st = engine.position_engine.states.get(pname)
        if st is None:
            return None
        rx, ry = st.current_x, st.current_y
        views = _build_views(engine)
        teammates: List[Any] = [
            views[team][n] for n in engine.position_engine.team_rosters.get(team, [])
            if n in views.get(team, {})
        ]
        defenders: List[Any] = []
        for ot, names in engine.position_engine.team_rosters.items():
            if ot == team:
                continue
            defenders.extend(views[ot][n] for n in names if n in views.get(ot, {}))
        attacks_right = engine.position_engine.team_attacks_right.get(team, True)
        from ball_vision import perceive_ball
        px, py, _seen, sigma = perceive_ball(
            engine, pname, views.get(team, {}).get(pname) or _View(pname, position),
            attacks_right, ball_x, ball_y, float(minute),
        )
        # OFF-BALL ACTOR HONESTY (2026-09-20): the runner's vision list is
        # gated/degraded by the SAME cone/range/top-k/noise rules as the
        # on-ball path, measured from the runner.  Disabled -> identity.
        from perception import perceive_offball_actors
        g_tms, g_defs, g_engine = perceive_offball_actors(
            _View(pname, position), rx, ry,
            teammates=teammates, defenders=defenders,
            position_engine=engine.position_engine,
            attacks_right=attacks_right,
        )
        from brain_sensors import extract_offball_sensors
        sensors = extract_offball_sensors(
            _View(pname, position), rx, ry, px, py,
            teammates=g_tms, defenders=g_defs,
            position_engine=g_engine,
            attacks_right=attacks_right,
            game_state=None,
            minute=float(minute),
            score_diff=score_diff,
            ball_sigma=sigma,
        )
        p = float(brain.forward(sensors))
        return min(1.0, max(0.0, p))
    except Exception:
        return None