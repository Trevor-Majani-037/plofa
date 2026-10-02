"""
MANAGER SENSORS (Phase 1) — the 12-float POV vector a manager sees.
Every sensor must be either an EXACT fact (scoreboard, clock) or a PERCEIVED
signal from manager_perception.  Internal engine floats (stamina, momentum
totals, threat levels, ...) must NEVER leak into the vector.
"""
from datetime import date

import numpy as np

from manager_perception import perceive_player_state
from manager_sensors import (
    MANAGER_SENSOR_LAYOUT,
    _worst_player_state,
    extract_manager_sensors,
)

HOME = "Hartwell City"
AWAY = "Thornfield United"


class _FakePE:
    def __init__(self):
        self.DEFENSIVE_LINE_POSITIONS = {"CB", "LB", "RB"}
        self.team_attacks_right = {HOME: True, AWAY: False}
        self.states = {}
        self.team_rosters = {"home": [], "away": []}


class _FakeConfig:
    home_team = HOME
    away_team = AWAY
    match_date = date(2026, 8, 16)
    matchday = 1
    season = "26/27"
    competition = "PLOFA"
    is_derby = False


class _FakeState:
    def __init__(self, minute=10, home=0, away=0, home_subs=0, away_subs=0):
        self.minute = minute
        self.home_goals = home
        self.away_goals = away
        self.match_clock_s = minute * 60.0
        self.home_subs_made = home_subs
        self.away_subs_made = away_subs
        self.match_ball_path = []


def _frame(minute, home_rows, away_rows):
    return {"minute": minute, "home": home_rows, "away": away_rows}


def _row(player, position, x, y, distance_total=500.0,
         physics_sprint_count=2.0, touches=4):
    return {
        "player": player, "position": position, "x": x, "y": y,
        "distance_total": distance_total,
        "physics_distance_m": distance_total,
        "physics_sprint_count": physics_sprint_count, "touches": touches,
    }


def _player(name, superstar=False):
    return type("P", (), {"name": name, "dna": type("D", (), {
        "is_superstar": superstar, "overall_rating": 80.0})})()


def _engine(position_log=None, timeline=None, minute=10, home=0, away=0,
            home_subs=0, away_subs=0, active=None):
    cfg = _FakeConfig()
    state = _FakeState(minute=minute, home=home, away=away,
                       home_subs=home_subs, away_subs=away_subs)
    pe = _FakePE()
    return type("E", (), {
        "config": cfg, "state": state, "position_engine": pe,
        "position_log": position_log or [], "timeline": timeline or [],
        "active_players": active or {HOME: [], AWAY: []},
        "squads": {},
    })()


# ── layout ──────────────────────────────────────────────────────────

def test_layout_exactly_12_named():
    assert isinstance(MANAGER_SENSOR_LAYOUT, dict)
    assert len(MANAGER_SENSOR_LAYOUT) == 12
    assert list(MANAGER_SENSOR_LAYOUT.keys()) == list(range(12))


def test_vector_shape_and_dtype():
    v = extract_manager_sensors(_engine(), HOME)
    assert isinstance(v, np.ndarray)
    assert v.shape == (12,)
    assert np.issubdtype(v.dtype, np.floating)
    assert np.all(np.isfinite(v))


# ── exact facts ──────────────────────────────────────────────────────

def test_exact_facts_mapped():
    # Home 2-1 up at minute 45, one sub used.
    eng = _engine(minute=45, home=2, away=1, home_subs=1)
    v = extract_manager_sensors(eng, HOME)
    assert v[0] == 1 / 5.0            # score_diff (2-1)/5 from home POV
    assert np.isclose(v[1], 45 / 90.0)  # minute_norm
    assert np.isclose(v[2], 2 / 3.0)    # subs_remaining = (3-1)/3
    assert 0.0 <= v[3] <= 1.0           # game_importance
    assert -1.0 <= v[4] <= 1.0          # recent_goal_impact
    for i in range(5, 12):
        assert v[i] in (0.0, 0.5, 1.0)  # perceived signals are coarse


def test_score_diff_sign_flips():
    # Away POV mirrors home POV on the scoreboard axis.
    home_v = extract_manager_sensors(_engine(minute=30, home=2, away=1), HOME)
    away_v = extract_manager_sensors(_engine(minute=30, home=2, away=1), AWAY)
    assert home_v[0] == 1 / 5.0
    assert away_v[0] == -1 / 5.0


# ── internal float leak guard ────────────────────────────────────────

def test_no_internal_floats_leak():
    """extract must NEVER touch the engine's internal float machinery."""

    class _GuardedEngine:
        FORBIDDEN = {
            "sub_controller", "momentum_engine", "threat_engine",
            "distance_engine", "state.momentum", "state.team_stances",
            "state.team_patterns",
        }

        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            key = name
            if key in self.FORBIDDEN:
                raise AssertionError(f"sensor leaked internal float: {key}")
            return getattr(self._base, name)

    eng = _GuardedEngine(
        _engine(position_log=[_frame(m, [_row("P1", "ST", 50, 34)],
                                     [_row("A1", "CB", 50, 34)])
                              for m in range(1, 6)], minute=10)
    )
    v = extract_manager_sensors(eng, HOME)
    assert v.shape == (12,)


# ── worst player state ──────────────────────────────────────────────

def test_worst_player_state():
    healthy = _player("P1")
    struggling = _player("P2")
    active = {HOME: [healthy, struggling], AWAY: [_player("A1")]}

    bl = [_frame(m, [_row("P1", "ST", 50, 34, distance_total=600.0,
                          physics_sprint_count=3.0),
                     _row("P2", "CM", 50, 34, distance_total=600.0,
                          physics_sprint_count=3.0)],
                   [_row("A1", "CB", 50, 34, distance_total=600.0,
                         physics_sprint_count=3.0)]) for m in range(1, 4)]
    plog = bl + [
        _frame(4, [_row("P1", "ST", 50, 34, distance_total=600.0,
                        physics_sprint_count=3.0),
                   _row("P2", "CM", 50, 34, distance_total=40.0,
                        physics_sprint_count=0.0)],
               [_row("A1", "CB", 50, 34, distance_total=600.0,
                     physics_sprint_count=3.0)]),
        _frame(5, [_row("P1", "ST", 50, 34, distance_total=600.0,
                        physics_sprint_count=3.0),
                   _row("P2", "CM", 50, 34, distance_total=40.0,
                        physics_sprint_count=0.0)],
               [_row("A1", "CB", 50, 34, distance_total=600.0,
                     physics_sprint_count=3.0)]),
    ]
    tl = [type("E", (), {"minute": m, "event_type": type(
        "T", (), {"name": "TURNOVER"})(), "team": HOME,
        "player": "P2", "outcome": False})() for m in (4, 5, 6)]

    eng = _engine(position_log=plog, timeline=tl, minute=8, active=active)
    assert perceive_player_state(eng, struggling) == 1.0   # P2 is BAD
    assert perceive_player_state(eng, healthy) == 0.0      # P1 is fine
    assert _worst_player_state(eng, HOME) == 1.0            # max across pitch