"""THE SMALL GAME — a scenario harness for verifying that a principle appears.

Why this exists
---------------
The coaching material this was built from says it plainly: *"We cannot assume
the players will automatically see the connection… Then play and observe if it
appears without a cue."* and *"The small game simplifies the picture. The big
game reveals whether the players truly recognise it."*

PLOFA had that pathology three times over, and every instance was a mechanism
that existed, was wired to something, and never ran in a match:

  * `striker_behavior.py` was fully written; its only consumer
    `_striker_run_step` lived in `drift_minute`, which `MatchEngine` never
    calls. The striker's live contribution was a pass-value term, not movement.
  * the wide / full-back / midfielder / CAM run modes were dead in the same
    method, in the same way, for months.
  * `POLICY_INTENT_AUTHORITY` was gated at five receiver sites and not at the
    two that force the delivery class, so the "the brain decides alone"
    experiment had never actually been run.

And the project had no way to catch any of them: all 46 positional-play tests
were either unit tests of a shape function or a full 90-minute match. There
was nothing in between, so "does the striker get in behind?" could only be
answered by staring at a full match's aggregate — which is how four separate
false findings happened in one session.

What this is
------------
A controlled state, a few seconds of the REAL 10 Hz off-ball integrator, and
traces out. It drives `MatchEngine._offball_run` — the same entry point a live
match uses — so there is no separate simulation path that could drift from
production and pass here while failing there.

It asserts on **where players ended up**, never on which functions were called.
That distinction is the whole point: a test that asserts "the striker run layer
was invoked" would have passed against the dead `drift_minute` wiring.

Player DNA, souls and formation anchors are the real ones from the real
roster; only the initial coordinates are the scenario's. So the shape rules
under test are the shipping ones.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from match_engine import (
    Intensity, MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle,
)
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
PITCH_X, PITCH_Y = 105.0, 68.0
TICK = 0.1


@dataclass
class Scenario:
    """A controlled picture.

    home / away map a squad slot to (x, y). Positions are the INITIAL
    coordinates; the real formation anchors are left in place so the shipping
    shape rules still have a formation to work with.
    """
    name: str
    description: str
    home: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    away: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ball: Tuple[float, float] = (52.5, 34.0)
    possessing: str = "home"
    seconds: float = 6.0
    attacks_right: bool = True
    # Filled in by `play()` from the engine, so `attackers_right` never has to
    # guess at a team name.
    home_name: str = ""
    away_name: str = ""

    def attackers_right(self, team: str) -> bool:
        """Map our abstract 'home'/'away' onto the engine's real club names.

        Comparing the engine's team name against the literal string "home"
        silently matched nothing, so BOTH clubs were set to attack left and the
        striker layer reasoned from a mirrored pitch. The first run of fixture A
        failed because of this line, not because of the engine — which is the
        harness lying in exactly the way the harness exists to catch, so it is
        left here as a comment rather than quietly fixed.
        """
        if team == self.home_name:
            return self.attacks_right
        if team == self.away_name:
            return not self.attacks_right
        raise KeyError(
            f"unknown team {team!r}; expected {self.home_name!r} or "
            f"{self.away_name!r}")


def _engine():
    """A real engine on the real 26/27 roster, never simulated."""
    loader = get_loader(XLSX)
    home_raw = loader.build_matchday_squad("Oxton")
    away_raw = loader.build_matchday_squad("Natrican")
    home = SquadBuilder.build("Oxton", starters=home_raw["starters"],
                              substitutes=home_raw["substitutes"])
    away = SquadBuilder.build("Natrican", starters=away_raw["starters"],
                              substitutes=away_raw["substitutes"])
    cfg = MatchConfig(home_team="Oxton", away_team="Natrican",
                      matchday=1, stadium_capacity=45000)
    eng = MatchEngine(
        cfg,
        TeamProfile(name="Oxton", style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.POSSESSION,
                    intensity=Intensity.MEDIUM),
        TeamProfile(name="Natrican", style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.MIXED,
                    intensity=Intensity.MEDIUM))
    eng.set_squad("Oxton", home["starters"], home["substitutes"])
    eng.set_squad("Natrican", away["starters"], away["substitutes"])
    return eng


def _slot_engines() -> Dict[str, Tuple[object, str]]:
    """The real starting XI, in formation order, for either team."""
    eng = _engine()
    H, A = eng.config.home_team, eng.config.away_team
    out = {}
    for team in (H, A):
        for p in eng.active_players[team]:
            out[f"{team[0]}:{p.position}"] = (p, team)
    return out


def play(sc: Scenario, seed: int = 7) -> Dict[str, object]:
    """Set the picture, run the real integrator, return what happened.

    Returns:
        traces   {name: [(t, x, y), ...]} sampled at every tick
        final    {name: (x, y)}
        start    {name: (x, y)}
        ball     (x, y)
        engine   the engine, for anything the caller wants to inspect
    """
    eng = _engine()
    H, A = eng.config.home_team, eng.config.away_team
    sc.home_name, sc.away_name = H, A
    slots = {}
    for team, table in ((H, sc.home), (A, sc.away)):
        for p in eng.active_players[team]:
            slots.setdefault(p.position, []).append(p)

    # place everyone: scenario coordinate if given, else the real anchor.
    # A position key may hold ONE coordinate or a LIST of them, because two
    # centre-backs share a position and a scenario that cannot say "these two
    # specific defenders" cannot express an offside line — which is the whole
    # point of fixture A. Assignment is in squad order, so a list is stable.
    #
    # MIRRORING: if the scenario attacks left, the FORMATION ANCHORS must be
    # mirrored too, not just the initial coordinates. Without this the mirrored
    # fixture is not a mirror at all — the away side is placed at x~22 while its
    # anchors still say x~80, so the shape layer hauls it 55 m the wrong way
    # and the striker appears to "run in behind" purely by comparison. That
    # fixture XPASSed for entirely the wrong reason, which is the most
    # dangerous way a test can be green.
    if not sc.attacks_right:
        for team in (H, A):
            for p in eng.active_players[team]:
                s = eng.position_engine.states.get(p.name)
                if s is not None:
                    s.home_x = PITCH_X - s.home_x
    placed: Dict[str, Tuple[float, float]] = {}
    for team, table in ((H, sc.home), (A, sc.away)):
        counters: Dict[str, int] = {}
        for p in eng.active_players[team]:
            st = eng.position_engine.states.get(p.name)
            if st is None:
                continue
            spec = table.get(p.position)
            if spec is None:
                continue
            i = counters.get(p.position, 0)
            counters[p.position] = i + 1
            xy = spec[i] if isinstance(spec, (list, tuple)) and spec and \
                isinstance(spec[0], (list, tuple)) else spec
            st.current_x, st.current_y = float(xy[0]), float(xy[1])
            placed[p.name] = (st.current_x, st.current_y)

    # attack direction for the run/off-ball layers
    for team in (H, A):
        eng.position_engine.team_attacks_right[team] = sc.attackers_right(team)
    if hasattr(eng, "team_attacks_right"):
        eng.team_attacks_right = {
            H: sc.attackers_right(H), A: sc.attackers_right(A)}

    # the engine reads the ball from state, not from a parameter
    eng.state.last_ball_x, eng.state.last_ball_y = float(sc.ball[0]), float(sc.ball[1])
    eng.state.cross_team, eng.state.cross_active = "", False

    # the minute-snapshot and top-speed caches the integrator reads. Without
    # these the run silently degrades to defaults, which is exactly the class
    # of bug this harness exists to surface, so they are populated honestly
    # rather than left empty and hoped over.
    for pname, (x, y) in placed.items():
        eng._minute_start_snapshot[pname] = (x, y)
        p = next((q for q in eng.active_players[H] + eng.active_players[A]
                  if q.name == pname), None)
        pace = 70.0
        if p is not None:
            for attr in ("pace", "speed", "acceleration"):
                v = getattr(getattr(p, "dna", None), attr, None)
                if isinstance(v, (int, float)):
                    pace = float(v)
                    break
        eng._top_speed_cache[pname] = 5.0 + pace * 0.042

    eng.enable_virtual_gps(0.1)
    eng._striker_run_cache.clear()
    eng._chase_state.clear()
    eng._rest_defence_cache.clear() if hasattr(eng, "_rest_defence_cache") else None
    random.seed(seed)

    home_has_ball = (sc.possessing == "home")
    before = {k: (s.current_x, s.current_y)
              for k, s in eng.position_engine.states.items() if k in placed}
    traces: Dict[str, List[Tuple[float, float, float]]] = {k: [] for k in placed}
    t = 0.0
    steps = max(1, int(round(sc.seconds / TICK)))
    for _ in range(steps):
        eng.state.match_clock_s = t
        eng._offball_run(TICK, home_has_ball)
        t += TICK
        for k in placed:
            s = eng.position_engine.states.get(k)
            if s is not None:
                traces[k].append((t, s.current_x, s.current_y))

    final = {k: (s.current_x, s.current_y)
             for k, s in eng.position_engine.states.items() if k in placed}
    # Report the DECISION alongside the OUTCOME. A scenario that only shows
    # where people ended up cannot tell you whether a wrong outcome came from a
    # wrong decision or from correct-decided-but-poorly-executed, and those
    # need completely different fixes. "Design the game, then observe" means
    # observing the decision too.
    intended = {}
    for name, prof in (getattr(eng.intended_runs, "counts", {}) or {}).items():
        fired = {m: v for m, v in prof.items() if v}
        if fired:
            intended[name] = fired
    observed = {}
    for name, prof in (eng.run_tracker.counts if eng.run_tracker else {}).items():
        fired = {m: v for m, v in prof.items() if v}
        if fired:
            observed[name] = fired
    return {"traces": traces, "final": final, "start": before,
            "ball": sc.ball, "engine": eng, "slots": slots,
            "seconds": sc.seconds,
            "intended": intended, "observed": observed,
            "intended_diag": eng.intended_runs.diagnostics()}


# ── helpers for assertions ────────────────────────────────────────────────

def depth_series(trace: List[Tuple[float, float, float]],
                 attacks_right: bool = True) -> np.ndarray:
    """Depth in the ATTACKING frame. Must be normalised: the teams change ends
    at half time, and a plot that does not know which way a team is attacking
    cannot be read. (Learned the hard way — an un-normalised depth chart showed
    a 'shape oscillating end to end' that was a single half-time mirror.)"""
    return np.array([(x if attacks_right else PITCH_X - x) for _, x, _ in trace])


def width_series(trace: List[Tuple[float, float, float]]) -> np.ndarray:
    return np.array([y for _, _, y in trace])


def moved(trace: List[Tuple[float, float, float]]) -> float:
    return float(sum(math.hypot(b[1] - a[1], b[2] - a[2])
                     for a, b in zip(trace, trace[1:])))


def travelled(trace, attacks_right: bool = True) -> float:
    d = depth_series(trace, attacks_right)
    return float(abs(d[-1] - d[0])) if len(d) else 0.0
