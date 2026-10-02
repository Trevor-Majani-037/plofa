"""
MANAGER PERCEPTION (Phase 1) — the manager's "eyes and ears".
Each perceived signal must be coarse (0.0 / 0.5 / 1.0), delayed by 2
minutes, and derived from observable match data (position_log, ball path,
timeline) — NOT from internal engine floats.
"""
from types import SimpleNamespace
from datetime import date

from manager_perception import (
    perceive_best_player_state,
    perceive_game_importance,
    perceive_key_player_available,
    perceive_momentum,
    perceive_opp_posture,
    perceive_player_state,
    perceive_recent_goal_impact,
    perceive_team_confidence,
    perceive_team_fatigue,
)

HOME = "Hartwell City"
AWAY = "Thornfield United"


class _FakePE:
    """PositionEngine duck — the perception layer only needs states/rosters."""

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
    def __init__(self):
        self.minute = 10
        self.home_goals = 0
        self.away_goals = 0
        self.match_clock_s = 600.0
        self.home_subs_made = 0
        self.away_subs_made = 0


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


def _engine(position_log=None, ball_path=None, timeline=None, minute=10,
            config=None, active_players=None, squads=None, clock=600.0):
    cfg = config or _FakeConfig()
    state = _FakeState()
    state.minute = minute
    state.match_clock_s = clock
    state.match_ball_path = ball_path or []
    pe = _FakePE()
    return SimpleNamespace(
        config=cfg, state=state,
        position_log=position_log or [],
        timeline=timeline or [],
        active_players=active_players or {HOME: [], AWAY: []},
        squads=squads or {},
        position_engine=pe,
    )


# ── perceive_team_fatigue ────────────────────────────────────────────

def test_fatigue_categorical():
    # A fresh start-of-match team with everyone moving at baseline reads FRESH.
    plog = [_frame(m, [_row("P1", "CM", 50, 34, distance_total=600.0,
                            physics_sprint_count=3.0)],
                   [_row("A1", "CM", 50, 34, distance_total=600.0,
                         physics_sprint_count=3.0)]) for m in range(1, 11)]
    eng = _engine(position_log=plog, minute=10)
    assert perceive_team_fatigue(eng, HOME) == 0.0

    # A sustained collapse (minutes 4..8 = 1/10th movement, no sprints):
    # by minute 8 the manager has fully absorbed it → SPENT.
    plog2 = [
        _frame(m, [_row("P1", "CM", 50, 34, distance_total=600.0,
                        physics_sprint_count=3.0)], [])
        for m in range(1, 4)
    ]
    plog2 += [
        _frame(m, [_row("P1", "CM", 50, 34, distance_total=60.0,
                        physics_sprint_count=0.0)], [])
        for m in range(4, 9)
    ]
    eng2 = _engine(position_log=plog2, minute=8)
    assert perceive_team_fatigue(eng2, HOME) == 1.0


def test_confidence_delay():
    # A burst of 10 turnovers all lands in minute 5.
    def _ev(m, etype, team, player="P1", outcome=True):
        return SimpleNamespace(minute=m, event_type=SimpleNamespace(name=etype),
                               team=team, player=player, outcome=outcome)
    tl = [_ev(5, "TURNOVER", HOME, outcome=False) for _ in range(10)]
    # At minute 6 the perception window ends at minute 4 → nothing seen.
    assert perceive_team_confidence(_engine(timeline=tl, minute=6), HOME) == 0.5
    # At minute 8 the window ends at minute 6 → burst is visible → DOWN.
    assert perceive_team_confidence(_engine(timeline=tl, minute=8), HOME) == 0.0


def test_confidence_from_events():
    def _ev(m, etype, team, player="P1", outcome=True):
        return SimpleNamespace(minute=m, event_type=SimpleNamespace(name=etype),
                               team=team, player=player, outcome=outcome)
    # 10 successful HOME actions vs 1 failure over the window → UP.
    timeline = [_ev(m, "PASS", HOME) for m in range(3, 9)]
    timeline += [_ev(4, "SHOT_ON_TARGET", HOME)]
    timeline += [_ev(5, "TACKLE_WON", HOME)]
    timeline += [_ev(6, "TURNOVER", HOME)]   # only 1 failure
    eng = _engine(timeline=timeline, minute=10)
    assert perceive_team_confidence(eng, HOME) == 1.0


# ── perceive_momentum ───────────────────────────────────────────────

def test_momentum_from_ball_path():
    # Ball in HOME's attacking third for 4 of the last 5 minutes → WITH_US.
    # Home attacks right (x=105 is their attacking goal) → att third x>70.
    ball_path = []
    clock = 600.0
    for i, x in enumerate((80.0, 85.0, 78.0, 90.0, 50.0)):   # 4 of 5 att third
        ball_path.append({"t": clock - (5 - i) * 55.0, "x": x, "y": 34,
                          "kind": "rest", "team": HOME})
    eng = _engine(ball_path=ball_path, minute=10)
    assert perceive_momentum(eng, HOME) == 1.0


# ── perceive_opp_posture ────────────────────────────────────────────

def test_opp_posture_from_positions():
    # Away (Thornfield) defends x=105. A back four parked at x~43 is HIGH
    # (pressed hard up the pitch, deep away from their goal).
    away_rows = [
        _row("A_CB1", "CB", 43, 20, distance_total=600.0),
        _row("A_CB2", "CB", 43, 48, distance_total=600.0),
        _row("A_LB", "LB", 43, 10, distance_total=600.0),
        _row("A_RB", "RB", 43, 60, distance_total=600.0),
    ]
    plog = [_frame(m, [_row("P1", "ST", 50, 34)], away_rows) for m in range(1, 6)]
    eng = _engine(position_log=plog, minute=6)
    # depth from own goal (105) = (105-43)/55 = 1.13 → HIGH.
    assert perceive_opp_posture(eng, AWAY) == 1.0

    # Same back four parked deep near their own goal → DEEP.
    deep_rows = [_row("A_CB1", "CB", 98, 20), _row("A_CB2", "CB", 98, 48),
                 _row("A_LB", "LB", 98, 10), _row("A_RB", "RB", 98, 60)]
    plog2 = [_frame(m, [_row("P1", "ST", 50, 34)], deep_rows) for m in range(1, 6)]
    eng2 = _engine(position_log=plog2, minute=6)
    assert perceive_opp_posture(eng2, AWAY) == 0.0


# ── perceive_player_state ───────────────────────────────────────────

def test_player_state_levels():
    # Baseline frames with full output; then P1's movement collapses.
    bl = [_frame(m, [_row("P1", "ST", 50, 34, distance_total=600.0,
                          physics_sprint_count=3.0)],
                 [_row("A1", "CB", 50, 34, distance_total=600.0,
                       physics_sprint_count=3.0)]) for m in range(1, 4)]
    # P1 collapses in minutes 4..5: near-zero distance and no sprints,
    # plus 3 turnovers logged in the timeline.
    plog = bl + [
        _frame(4, [_row("P1", "ST", 50, 34, distance_total=60.0,
                        physics_sprint_count=0.0)],
               [_row("A1", "CB", 50, 34, distance_total=600.0,
                     physics_sprint_count=3.0)]),
        _frame(5, [_row("P1", "ST", 50, 34, distance_total=60.0,
                        physics_sprint_count=0.0)],
               [_row("A1", "CB", 50, 34, distance_total=600.0,
                     physics_sprint_count=3.0)]),
    ]
    tl = [SimpleNamespace(minute=m, event_type=SimpleNamespace(name="TURNOVER"),
                          team=HOME, player="P1", outcome=False)
          for m in (4, 5, 6)]
    eng = _engine(position_log=plog, timeline=tl, minute=8)
    assert perceive_player_state(eng, "P1") == 1.0   # BAD


# ── perceive_best_player_state ──────────────────────────────────────

def test_best_player_attacker_fire():
    rows1 = [
        _row("PERCY", "RW", 80, 20, distance_total=600.0,
             physics_sprint_count=3.0, touches=8),
        _row("NOVAK", "ST", 75, 25, distance_total=600.0,
             physics_sprint_count=3.0, touches=2),
    ]
    plog = [
        _frame(m, rows1, [_row("A1", "CB", 50, 34)]) for m in range(1, 6)
    ]
    tl = [SimpleNamespace(minute=m, event_type=SimpleNamespace(name=e),
                          team=HOME, player="PERCY", outcome=True)
          for m in range(3, 7) for e in ("PASS", "PASS", "SHOT_ON_TARGET")]
    eng = _engine(position_log=plog, timeline=tl, minute=8)
    assert perceive_best_player_state(eng, HOME) == 1.0   # ON_FIRE


# ── perceive_key_player_available ───────────────────────────────────

def test_key_player_available():
    def _p(name, superstar=False, rating=75.0):
        return SimpleNamespace(name=name, dna=SimpleNamespace(
            is_superstar=superstar, overall_rating=rating))
    active = {"Hartwell City": [_p("Percy", superstar=True), _p("Novak")],
              "Thornfield United": [_p("Asante")]}
    eng = _engine(active_players=active)
    assert perceive_key_player_available(eng, HOME) == 1.0   # on pitch


# ── perceive_recent_goal_impact ─────────────────────────────────────

def test_recent_goal_decay():
    def _gol(m, team):
        return SimpleNamespace(minute=m, event_type=SimpleNamespace(name="GOAL"),
                               team=team, player="X", outcome=True)
    # A goal 1 minute ago has a higher impact than one 10 minutes ago.
    tl = [_gol(7, HOME), _gol(0, HOME)]    # minute 0 = 10 min before minute 10
    eng = _engine(timeline=tl, minute=10)
    # Both contribute +exp; recent dominates.
    assert perceive_recent_goal_impact(eng, HOME) > 0.5
    # Same goal conceded: recent negative outweighs the old one.
    tl2 = [_gol(9, AWAY), _gol(0, AWAY)]
    eng2 = _engine(timeline=tl2, minute=10)
    assert perceive_recent_goal_impact(eng2, HOME) < -0.5


# ── perceive_game_importance ────────────────────────────────────────

def test_game_importance_derby():
    cfg = _FakeConfig()
    cfg.is_derby = True
    eng = _engine(config=cfg)
    assert perceive_game_importance(eng) >= 0.7

    cfg2 = _FakeConfig()
    cfg2.is_derby = False
    cfg2.competition = "PLOFA Cup"
    eng2 = _engine(config=cfg2)
    assert perceive_game_importance(eng2) == 1.0   # cup final