"""
Manager Composer (Phase 5) — tests.
"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from manager_brain import ManagerBrain
from manager_memory import ManagerMemory
from manager_mind import ManagerMind
from manager_profile import Manager, MANAGER_MIN_DWELL_S


# ── Mock engine ─────────────────────────────────────────────────

def _engine(minute=30, score=(1, 0), subs=(0, 0), last_goal_team=None):
    """Lightweight mock of MatchEngine for Manager.decide."""
    cfg = SimpleNamespace(home_team="Home", away_team="Away")
    state = SimpleNamespace(
        minute=float(minute),
        home_goals=score[0], away_goals=score[1],
        home_subs_made=subs[0], away_subs_made=subs[1],
        last_goal_team=last_goal_team,
    )
    return SimpleNamespace(config=cfg, state=state)


def _manager(seed=42):
    brain = ManagerBrain.random(seed=seed)
    mind = ManagerMind(dogma=0.5, eq=0.5, empathy=0.5)
    mem = ManagerMemory(decay=1.0)
    return Manager(name="Test", brain=brain, mind=mind, memory=mem,
                   memory_strength=0.0)


# ── Tests ───────────────────────────────────────────────────────

def test_decide_returns_all_keys(monkeypatch):
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    out = mgr.decide(_engine(), "Home")
    assert set(out.keys()) == {"posture", "posture_probs", "pressing", "sub_urgency"}
    assert out["posture"] in ("DEFEND", "BALANCED", "ATTACK")
    assert isinstance(out["pressing"], float)
    assert isinstance(out["sub_urgency"], float)


def test_posture_sums_to_one_after_memory(monkeypatch):
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    out = mgr.decide(_engine(), "Home")
    probs = out["posture_probs"]
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_dwell_respected(monkeypatch):
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    mgr.decide(_engine(minute=10), "Home")
    first_posture = mgr._current_posture
    first_change_time = mgr._last_posture_change_s
    mgr.decide(_engine(minute=11), "Home")
    assert mgr._current_posture == first_posture
    assert mgr._last_posture_change_s == first_change_time


def test_dwell_allows_change(monkeypatch):
    """After MANAGER_MIN_DWELL_S has elapsed, posture may change."""
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    mgr.decide(_engine(minute=10), "Home")
    out = mgr.decide(_engine(minute=10 + MANAGER_MIN_DWELL_S / 60.0 + 1), "Home")
    assert out["posture"] in ("DEFEND", "BALANCED", "ATTACK")


def test_on_event_goal_conceded_updates_mind():
    mgr = _manager()
    stress_before = mgr.mind.stress_accumulator
    engine = _engine(last_goal_team="Away")
    mgr.on_event("GOAL", engine, "Home")
    assert mgr.mind.stress_accumulator > stress_before


def test_on_event_goal_scored_updates_mind():
    mgr = _manager()
    mgr.mind.stress_accumulator = 0.8
    stress_before = mgr.mind.stress_accumulator
    engine = _engine(last_goal_team="Home")
    mgr.on_event("GOAL", engine, "Home")
    assert mgr.mind.stress_accumulator < stress_before


def test_end_of_match_writes_memory(monkeypatch):
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    for _ in range(3):
        mgr.decide(_engine(minute=30), "Home")
    result = SimpleNamespace(
        config=SimpleNamespace(home_team="Home"),
        home_goals=2, away_goals=1,
        home_xg=1.5, away_xg=0.9,
    )
    mgr.end_of_match(result, "Home")
    assert len(mgr.memory.table) > 0
    assert mgr.memory_strength == mgr.memory.strength()


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _manager()
    mgr.decide(_engine(minute=30), "Home")
    result = SimpleNamespace(
        config=SimpleNamespace(home_team="Home"),
        home_goals=1, away_goals=1,
        home_xg=1.0, away_xg=1.0,
    )
    mgr.end_of_match(result, "Home")
    d = str(tmp_path / "mgr_state")
    mgr.save(d)
    mgr2 = Manager.load(d)
    assert mgr2.name == mgr.name
    assert mgr2.memory_strength == mgr.memory_strength
    assert mgr2._current_posture == mgr._current_posture
    assert mgr2.memory.table == mgr.memory.table
    assert mgr2.mind.dogma == mgr.mind.dogma


def test_deterministic(monkeypatch):
    fixed = np.array([0.1, 0.2, 0.3, 0.4, 0.5,
                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6], dtype=np.float64)
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: fixed.copy(),
    )
    eng = _engine(minute=50)
    mgr = _manager()
    out1 = mgr.decide(eng, "Home")
    mgr2 = _manager()
    out2 = mgr2.decide(eng, "Home")
    assert out1["posture"] == out2["posture"]
    assert out1["pressing"] == out2["pressing"]
    assert out1["sub_urgency"] == out2["sub_urgency"]
    np.testing.assert_array_equal(out1["posture_probs"], out2["posture_probs"])