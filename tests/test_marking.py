"""
PLOFA 26/27 — Marking Engine Tests (P1)
=========================================
test_marking.py

Validates the man/zonal marking layer end to end:

    1.  Priority        — attackers near the goal / central / aerial draw
                         higher cover priority than deep/wide ones.
    2.  Assignment      — the greedy solver pairs the best-suited defender
                         with each priority attacker (aerial ST draws the
                         best CB, a breaking winger draws the near FB).
    3.  marker_state    — tight / free / beaten transitions from geometry.
    4.  Block wiring    — PositionEngine.defensive_block with attacking_team
                         pulls a defender toward a MAN, not just the ball
                         line; a free box attacker attracts a coverer.
    5.  Corner grid     — SetPieceMarkingEngine assigns the best aerial CB to
                         the opponent's top threat, sets first man + GK side.
    6.  Full match      — a match with the marking layer still runs and the
                         corner metadata carries the set-piece grid.
"""

import random
from datetime import date

import pytest

from marking import (
    MarkingEngine, MarkAssignment, SetPieceMarkingEngine,
    BOX_MARK_RADIUS, TIGHT_MARK_DIST,
)
from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from position_engine import PositionEngine
from player_dna import SquadBuilder
from event_chain import ChainDispatcher


# ============================================================================
# Helpers
# ============================================================================

_ROLES = [
    ("GK", ["sweeper_keeper"]),
    ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]),
    ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]),
    ("CDM", ["anchor_man"]),
    ("CM", ["engine"]),
    ("CM", ["box_box"]),
    ("CAM", ["creator"]),
    ("LW", ["winger"]),
    ("ST", ["fox_in_box"]),
]


def _make_squad(team_name: str) -> list:
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specialties, 26)
        for i, (pos, specialties) in enumerate(_ROLES)
    ]
    return SquadBuilder.build(team_name, starters)["starters"]


def _defenders(pe: PositionEngine, team: str):
    """Build the defenders tuple list for MarkingEngine from live states."""
    out = []
    for n, s in pe.states.items():
        if s.team == team and s.position in ("CB", "LB", "RB", "CDM"):
            out.append((n, s.position, s.current_x, s.current_y))
    return out


def _attackers(pe: PositionEngine, team: str, runners=("ST", "CF", "LW", "RW")):
    out = []
    for n, s in pe.states.items():
        if s.team == team and s.position != "GK":
            out.append((n, s.position, s.current_x, s.current_y,
                        (s.position in runners, 60.0, 60.0)))
    return out


# ============================================================================
# 1. PRIORITY — geometry drives who must be covered
# ============================================================================

def test_attacker_priority_rises_toward_goal_and_centrality():
    # Away defends x=105; an ST at 90m drops toward the box.
    near = MarkingEngine.attacker_priority(92.0, 34.0, 105.0)
    deep = MarkingEngine.attacker_priority(60.0, 34.0, 105.0)
    wide_central = MarkingEngine.attacker_priority(92.0, 34.0, 105.0)
    wide_outside = MarkingEngine.attacker_priority(92.0, 64.0, 105.0)
    assert near > deep, "closer to goal must be a higher cover priority"
    assert wide_central > wide_outside, "central is more dangerous than wide"


def test_attacker_priority_boosts_aerial_and_runner():
    base = MarkingEngine.attacker_priority(92.0, 34.0, 105.0)
    with_aerial = MarkingEngine.attacker_priority(92.0, 34.0, 105.0, aerial=90.0)
    run = MarkingEngine.attacker_priority(92.0, 34.0, 105.0, pace=95.0, is_runner=True)
    assert with_aerial > base
    assert run > base


# ============================================================================
# 2. ASSIGNMENT — the right defender covers the right attacker
# ============================================================================

def test_aerial_st_draws_the_best_aerial_cb():
    """Two CBs defend a lone ST who is a big aerial target. The CB with the
    higher jumping/heading must be the assigned carrier."""
    # defenders: (name, position, x, y)
    cb_weak = ("Weak CB", "CB", 80.0, 34.0)
    cb_strong = ("Strong CB", "CB", 84.0, 34.0)
    defenders = [cb_weak, cb_strong]
    # attackers: (name, position, x, y, tuple)
    st = ("Tall ST", "ST", 88.0, 34.0, (False, 95.0, 95.0))
    assignments, _ = MarkingEngine.assign(
        defenders, [st], own_goal_x=105.0, ball_x=88.0, ball_y=34.0,
        danger_level=85.0,
    )
    # The strong CB must have picked up the ST (or at least some CB did).
    covered_by = [a for a in assignments.values() if a.attacker_name == "Tall ST"]
    assert covered_by, "the ST must be covered"
    assert covered_by[0].defender_name == "Strong CB", (
        f"expected Strong CB to carry the target, got {covered_by[0].defender_name}")


def test_breaking_winger_draws_near_side_fullback():
    """A RW attacking down the right touches the near-side LB first."""
    lb = ("My LB", "LB", 60.0, 10.0)
    rb = ("My RB", "RB", 60.0, 58.0)
    rw = ("Fast RW", "RW", 70.0, 62.0, (True, 50.0, 90.0))
    assignments, _ = MarkingEngine.assign(
        [lb, rb], [rw], own_goal_x=105.0, ball_x=72.0, ball_y=60.0,
        danger_level=60.0,
    )
    covering = [a for a in assignments.values() if a.attacker_name == "Fast RW"]
    assert covering, "the winger must be covered"
    assert covering[0].defender_name == "My RB", (
        f"near-side fullback should take the winger, got {covering[0].defender_name}")


def test_every_defender_gets_one_or_free():
    """With fewer attackers than defenders some defender ends up unassigned
    (space coverage) — that is fine and expected."""
    defenders = [("CB1", "CB", 75.0, 30.0), ("CB2", "CB", 78.0, 38.0),
                 ("LB", "LB", 72.0, 8.0), ("RB", "RB", 76.0, 60.0)]
    attackers = [("Only ST", "ST", 84.0, 34.0, (False, 70.0, 70.0))]
    assignments, left = MarkingEngine.assign(
        defenders, attackers, own_goal_x=105.0, ball_x=84.0, ball_y=34.0,
        danger_level=70.0,
    )
    assert len(assignments) >= 1, "at least one defender should be assigned"
    assert left, "leftovers listing is returned"


# ============================================================================
# 3. MARKER STATE — tight / free / beaten from geometry
# ============================================================================

def _assign_one(def_x, def_y, atk_x, atk_y, own_goal_x=105.0):
    defenders = [("D", "CB", def_x, def_y)]
    attackers = [("A", "ST", atk_x, atk_y, (False, 50.0, 50.0))]
    assignments, _ = MarkingEngine.assign(
        defenders, attackers, own_goal_x=own_goal_x,
        ball_x=atk_x, ball_y=atk_y, danger_level=80.0,
    )
    return assignments["D"]


def test_marker_state_tight_when_on_the_man_goal_side():
    # Defender sits between the man and the goal within 2.2m: tight cover.
    a = _assign_one(88.0, 34.0, 86.5, 34.0)   # defender 1.5m goalside of man
    assert a.marker_state == "tight"


def test_marker_state_beaten_when_attacker_has_shot_away():
    a = _assign_one(84.0, 34.0, 92.0, 34.0)   # attacker 8m toward goal
    assert a.marker_state == "beaten"


def test_marker_state_free_when_between():
    a = _assign_one(84.0, 34.0, 88.0, 34.0)   # ~4m gap
    assert a.marker_state == "free"


# ============================================================================
# 4. BLOCK WIRING — the block follows MAN, not just ball geometry
# ============================================================================

def _block_engine():
    profile = TeamProfile(name="Away FC", style=TeamStyle.BALANCED,
                          playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    pe = PositionEngine()
    away = _make_squad("Away FC")
    home = _make_squad("Home FC")
    # Away defends x=105; Home attacks right (x->105). State can be arbitrary —
    # we place players where we want them below.
    pe.initialize_team("Away FC", away, profile, attacks_right=False)
    pe.initialize_team("Home FC", home, profile, attacks_right=True)
    return pe, away, home


def test_block_covers_a_free_box_attacker():
    """A lone ST standing free in the box must attract a defender toward it —
    even a defender whose start position is on the opposite side of the line."""
    pe, away, home = _block_engine()
    # Pin the free ST into the box near away's goal (x=96, central).
    st = next(p for p in home if p.position == "ST")
    pe.states[st.name].current_x = 96.0
    pe.states[st.name].current_y = 34.0
    # Pin every away shield well upfield so the ball-geometry line alone would
    # leave the ST unmarked (the block must REACH for the man).
    for n, s in pe.states.items():
        if s.team == "Away FC" and s.position in ("CB", "LB", "RB", "CDM"):
            s.current_x = 70.0
            s.current_y = 30.0
    before = {n: pe.get_position(n) for n, s in pe.states.items()
              if s.team == "Away FC"}

    pe.defensive_block(
        "Away FC", 96.0, 34.0, own_goal_x=105.0, danger_level=92.0,
        pull_strength=1.0, defensive_line=0.5, attacking_team="Home FC",
    )

    # Every assigned defender must move GOAL-side (x increases toward 105) and
    # at least one must close on the ST's channel.
    squad_defenders = [p for p in away if p.position in ("CB", "LB", "RB", "CDM")]
    moved_goal_side = 0
    for n, s in pe.states.items():
        if s.team != "Away FC" or s.position == "GK":
            continue
        if s.position not in ("CB", "LB", "RB", "CDM"):
            continue
        bx = before[n][0]
        assert s.current_x > bx, f"{n} must be pulled goal-side (toward the box)"
        moved_goal_side += 1
    assert moved_goal_side == len(squad_defenders)
    # The defending shield must have closed on the ST: someone is now within
    # marking range of the box attacker.
    st_x, st_y = pe.get_position(st.name)
    defender_names = [p.name for p in next(iter([away]), [])
                      if p.position in ("CB", "LB", "RB", "CDM")]
    def _closest(team):
        d = []
        for n, s in pe.states.items():
            if s.team == team and s.position in ("CB", "LB", "RB", "CDM"):
                d.append(((s.current_x - st_x) ** 2 + (s.current_y - st_y) ** 2) ** 0.5)
        return min(d)
    dist_after = _closest("Away FC")
    dist_before = min(
        ((before[n][0] - st_x) ** 2 + (before[n][1] - st_y) ** 2) ** 0.5
        for n in defender_names if n in before
    )
    assert dist_after < 12.0, "a defender must actually reach the box attacker"
    assert dist_after < dist_before, "the closest defender must move closer to the attacker"


def test_block_backwards_compatible_without_attacking_team():
    """Without attacking_team the block is the pure ball-geometry version —
    the existing positional tests depend on that contract."""
    pe, away, home = _block_engine()
    cbs = [p for p in away if p.position == "CB"]
    before = {p.name: pe.get_position(p.name) for p in cbs}
    pe.defensive_block("Away FC", 88.0, 20.0, own_goal_x=105.0, danger_level=85.0,
                       pull_strength=1.0)
    for p in cbs:
        bx, by = before[p.name]
        ax, ay = pe.get_position(p.name)
        assert ax > bx, "backwards-compatible block still pulls goal-side"
        assert abs(ay - 20.0) < abs(by - 20.0)


# ============================================================================
# 5. SET-PIECE GRID — corner defence roles
# ============================================================================

def _corner_squads():
    defers = [
        ("Tall CB", "CB", 95.0, 90.0),
        ("Small LB", "LB", 50.0, 40.0),
        ("RB Out", "RB", 60.0, 45.0),
    ]
    attack = [
        ("Big Target", "ST", 92.0, 88.0),
        ("Lurking", "CB", 80.0, 70.0),
    ]
    return defers, attack


def test_set_piece_assigns_best_aerial_cb_to_top_threat():
    defers, attack = _corner_squads()
    sp = SetPieceMarkingEngine.build(defers, attack, own_goal_x=105.0,
                                     corner_side="right")
    assert sp.aerial_defender == "Tall CB", (
        f"dominant jumper must carry the target, got {sp.aerial_defender}")
    assert sp.assignments.get("Tall CB") == "Big Target"
    assert sp.first_man is not None and sp.first_man != "Tall CB"
    assert sp.gk_defended_side in ("left", "right")


def test_set_piece_gk_guards_defended_side():
    # Defending 105 (right goal), corner swung FROM the right side.
    defers, attack = _corner_squads()
    sp = SetPieceMarkingEngine.build(defers, attack, own_goal_x=105.0,
                                     corner_side="right")
    assert abs(sp.gk_x - 105.0) <= 3.0, "GK must start on the goal line"
    # In-swinger from the right -> GK defends the right side (lower y).
    assert sp.gk_y < 34.0


# ============================================================================
# 6. FULL MATCH — marking layer survives a real simulation
# ============================================================================

def test_full_match_runs_with_marking():
    random.seed(13)
    config = MatchConfig(home_team="Home FC", away_team="Away FC", match_date=date.today())
    home_profile = TeamProfile(name="Home FC", style=TeamStyle.ATTACKING,
                               playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    away_profile = TeamProfile(name="Away FC", style=TeamStyle.DEFENSIVE,
                               playing_style=PlayingStyle.LOW_BLOCK, intensity=Intensity.LOW)
    engine = MatchEngine(config, home_profile, away_profile)
    engine.set_squad("Home FC", _make_squad("Home FC"))
    engine.set_squad("Away FC", _make_squad("Away FC"))
    result = engine.simulate()

    assert result is not None
    assert len(result.goals) >= 0
    # Corner metadata carries the set-piece grid somewhere in the timeline.
    grid_seen = any(
        getattr(e, "metadata", {}).get("set_piece_marking") is not None
        for e in result.timeline
    )
    assert grid_seen, "at least one corner must record the marking grid"