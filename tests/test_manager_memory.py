"""
Manager Memory (Phase 4) — tests.
"""
from types import SimpleNamespace

import numpy as np

from manager_memory import ManagerMemory, OutcomeStats


def _ctx(score_diff=0, minute=20, posture="BALANCED"):
    return SimpleNamespace(score_diff=score_diff, minute=minute,
                           posture_taken=posture)


def _result(home_goals, away_goals, home_team="Home", home_xg=1.5, away_xg=1.2):
    return SimpleNamespace(
        config=SimpleNamespace(home_team=home_team),
        home_goals=home_goals, away_goals=away_goals,
        home_xg=home_xg, away_xg=away_xg,
    )


# ── Key construction ───────────────────────────────────────────────

def test_key_deterministic():
    m = ManagerMemory()
    k1 = m.key_for(_ctx(score_diff=1, minute=20, posture="ATTACK"))
    k2 = m.key_for(_ctx(score_diff=1, minute=25, posture="ATTACK"))
    assert k1 == k2, "same bucket/state/posture => same key"
    assert k1[0] == "leading"
    assert k1[1] == "15-30"

    # Boundary buckets
    assert ManagerMemory.key_for(_ctx(minute=15))[1] == "0-15"
    assert ManagerMemory.key_for(_ctx(minute=16))[1] == "15-30"
    assert ManagerMemory.key_for(_ctx(minute=45))[1] == "30-45"
    assert ManagerMemory.key_for(_ctx(minute=46))[1] == "45-60"
    assert ManagerMemory.key_for(_ctx(minute=90))[1] == "75-90"

    # Score states
    assert ManagerMemory.key_for(_ctx(score_diff=-2))[0] == "trailing"
    assert ManagerMemory.key_for(_ctx(score_diff=0))[0] == "level"


# ── Recording ──────────────────────────────────────────────────────

def test_record_updates_stats():
    m = ManagerMemory()
    m.record(_ctx(score_diff=1, minute=20, posture="ATTACK"), "ATTACK",
             _result(2, 0), "Home")
    key = ("leading", "15-30", "ATTACK")
    assert key in m.table
    s = m.table[key]
    assert s.wins == 1 and s.count >= 1
    assert s.xg_for == 1.5 and s.xg_against == 1.2

    # Another same-key record (draw this time)
    m.record(_ctx(score_diff=1, minute=22, posture="ATTACK"), "ATTACK",
             _result(1, 1), "Home")
    s = m.table[key]
    assert s.wins == 1 and s.draws == 1
    assert not s.losses


def test_away_team_result_flipped():
    m = ManagerMemory()
    # Home 2–1: from Away's POV that is a LOSS (they scored 1, conceded 2).
    m.record(_ctx(score_diff=-1, minute=60, posture="DEFEND"), "DEFEND",
             _result(2, 1, home_team="Home"), team="Away")
    key = ("trailing", "45-60", "DEFEND")
    s = m.table[key]
    assert s.wins == 0 and s.losses == 1
    assert s.xg_for == 1.2 and s.xg_against == 1.5


def test_decay_reduces_count():
    m = ManagerMemory(decay=0.5)
    m.record(_ctx(score_diff=0, minute=10, posture="DEFEND"), "DEFEND",
             _result(1, 0), "Home")
    key_a = ("level", "0-15", "DEFEND")
    assert m.table[key_a].count == 1

    # Recording a different key decays the first entry's count by 0.5.
    m.record(_ctx(score_diff=0, minute=40, posture="ATTACK"), "ATTACK",
             _result(1, 1), "Home")
    key_b = ("level", "30-45", "ATTACK")
    assert m.table[key_b].count == 1
    assert m.table[key_a].count == 0, "old entry decayed (int(1*0.5))"


def test_max_entries_bounded():
    m = ManagerMemory(max_entries=3, decay=0.99)
    for i in range(10):
        posture = ("DEFEND", "BALANCED", "ATTACK")[i % 3]
        m.record(_ctx(score_diff=0, minute=10 + i, posture=posture), posture,
                 _result(1, 0), "Home")
    assert len(m.table) <= 3


# ── Bias / strength ────────────────────────────────────────────────

def test_bias_for_returns_3_vector():
    m = ManagerMemory()
    v = m.bias_for(_ctx())
    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    assert np.all(np.abs(v) <= 1.0)


def test_bias_learns_from_outcomes():
    m = ManagerMemory(decay=1.0)   # no decay so counts are clean
    # 10 losses while trying ATTACK in these conditions
    for _ in range(10):
        m.record(_ctx(score_diff=1, minute=80, posture="ATTACK"), "ATTACK",
                 _result(0, 2), "Home")
    # A couple of wins while trying DEFEND in the same conditions
    for _ in range(2):
        m.record(_ctx(score_diff=1, minute=80, posture="DEFEND"), "DEFEND",
                 _result(2, 0), "Home")

    v = m.bias_for(_ctx(score_diff=1, minute=85))
    names = ["DEFEND", "BALANCED", "ATTACK"]
    as_dict = dict(zip(names, v))
    assert as_dict["ATTACK"] < as_dict["DEFEND"], \
        f"ATTACK should be punished: {as_dict}"


def test_strength_bounded():
    m = ManagerMemory()
    assert m.strength() == 0.0
    for i in range(60):
        posture = ("DEFEND", "BALANCED", "ATTACK")[i % 3]
        m.record(_ctx(score_diff=0, minute=30 + i % 60, posture=posture),
                 posture, _result(1, 0), "Home")
    assert 0.0 <= m.strength() <= 1.0


# ── Persistence ────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    m = ManagerMemory(max_entries=100, decay=0.97)
    m.record(_ctx(score_diff=2, minute=70, posture="ATTACK"), "ATTACK",
             _result(3, 1), "Home")
    m.record(_ctx(score_diff=2, minute=75, posture="ATTACK"), "ATTACK",
             _result(1, 1), "Home")
    p = str(tmp_path / "memory.json")
    m.save(p)
    m2 = ManagerMemory.load(p)
    assert m2.decay == m.decay
    assert m2.max_entries == m.max_entries
    assert m2.table == m.table
    # And behaviour matches
    v1 = m.bias_for(_ctx(score_diff=2, minute=70))
    v2 = m2.bias_for(_ctx(score_diff=2, minute=70))
    assert np.array_equal(v1, v2)