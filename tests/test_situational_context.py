"""The situational context blocks — with the plan-block regression pinned.

THE BUG THIS FILE EXISTS FOR (found and fixed 2026-11-08)
---------------------------------------------------------
``situation_blocks`` read the player's team with ``getattr(player, "team", "")``.
``PlayerProfile`` carries it as ``team_name`` and has no ``team`` attribute, so
the lookup returned ``""`` for every player in every match.  ``plan_tail`` then
bailed out on the empty team, and the six plan inputs the network reads were
constant zeros in every situation.  The network trained on a dead block and
correctly learned to ignore it — which is why the PLAN head could never have
worked.

These tests fail on the old code and pass on the fix.
"""
import numpy as np

import situational_context as sc
from player_state import NeuralStateStore, player_team
from situational_brain import MEMORY_D, PLAN_D, ROLE_D, STATE_D


class _Player:
    """Stands in for PlayerProfile: ``team_name``, never ``team``."""

    def __init__(self, name="Tester", position="CM", team_name="Oxton"):
        self.name = name
        self.position = position
        self.team_name = team_name


class _Engine:
    """Only the attribute the plan path touches."""

    def __init__(self):
        self.neural_states = NeuralStateStore()


def _adopt(store, team, plan_id=2, confidence=0.9):
    """Adopt a plan for real: ``propose`` needs ``min_hits=2`` identical calls."""
    tp = store.team(team)
    tp.propose(plan_id, confidence)
    tp.propose(plan_id, confidence)
    return tp


# ── plan_tail ────────────────────────────────────────────────────────────

def test_plan_tail_is_zero_without_a_team():
    assert np.all(sc.plan_tail(_Engine(), "") == 0.0)


def test_plan_tail_is_zero_without_an_engine():
    assert np.all(sc.plan_tail(None, "Oxton") == 0.0)


def test_plan_tail_is_zero_before_any_plan_is_adopted():
    """A fresh team reads as zeros — 'no plan' is a value, not an error."""
    assert np.all(sc.plan_tail(_Engine(), "Oxton") == 0.0)


def test_plan_tail_reports_an_adopted_plan():
    eng = _Engine()
    _adopt(eng.neural_states, "Oxton", plan_id=2)
    out = sc.plan_tail(eng, "Oxton")
    assert out.shape == (PLAN_D,)
    assert out[-1] == 1.0, "has_plan flag must be set"
    assert out[:PLAN_D - 2].sum() == 1.0, "exactly one phase must be hot"
    assert out[PLAN_D - 2] > 0.0, "confidence must be reported"


def test_plan_tail_width_is_pinned_even_if_the_summary_changes(monkeypatch):
    """A plan-head count change must not silently change the input width."""
    eng = _Engine()

    class _FatSummary:
        def summary(self):
            return np.ones(PLAN_D + 5, dtype=np.float64)

    class _Store:
        def team(self, team):
            return _FatSummary()

    eng.neural_states = _Store()
    out = sc.plan_tail(eng, "Oxton")
    assert out.shape == (PLAN_D,)


# ── the regression itself ────────────────────────────────────────────────

def test_player_profile_has_no_team_attribute():
    """If this ever fails, the old lookup would have started working by luck."""
    p = _Player()
    assert getattr(p, "team", "") == ""
    assert player_team(p) == "Oxton"


def test_situation_blocks_reads_team_name_not_team(monkeypatch):
    """The load-bearing test: the plan block must be live, not zeros."""
    monkeypatch.setattr(sc, "role_tail", lambda *a, **k: np.zeros(ROLE_D))
    monkeypatch.setattr(sc, "memory_tail", lambda *a, **k: np.zeros(MEMORY_D))

    eng = _Engine()
    _adopt(eng.neural_states, "Oxton", plan_id=2)
    p = _Player()

    role, mem, plan = sc.situation_blocks(
        p, 52.5, 34.0, [], [], eng, True, np.zeros(STATE_D))

    assert plan.shape == (PLAN_D,)
    assert plan[-1] == 1.0, (
        "plan block is zeros again — situation_blocks lost the team lookup")
    assert role.shape == (ROLE_D,)
    assert mem.shape == (MEMORY_D,)


def test_situation_blocks_gives_the_right_team_to_each_side(monkeypatch):
    """Two players on different teams must read different plans.

    This is the trap ``player_team`` documents: the fixture names both teams'
    players identically (CB1, CM1, ...), so any lookup keyed on NAME would hand
    half the squad the other team's plan.
    """
    monkeypatch.setattr(sc, "role_tail", lambda *a, **k: np.zeros(ROLE_D))
    monkeypatch.setattr(sc, "memory_tail", lambda *a, **k: np.zeros(MEMORY_D))

    eng = _Engine()
    _adopt(eng.neural_states, "Oxton", plan_id=1)
    _adopt(eng.neural_states, "Natrican", plan_id=3)

    home = _Player("CM1", "CM", "Oxton")
    away = _Player("CM1", "CM", "Natrican")

    _, _, p_home = sc.situation_blocks(
        home, 52.5, 34.0, [], [], eng, True, np.zeros(STATE_D))
    _, _, p_away = sc.situation_blocks(
        away, 52.5, 34.0, [], [], eng, False, np.zeros(STATE_D))

    assert int(np.argmax(p_home[:PLAN_D - 2])) == 1
    assert int(np.argmax(p_away[:PLAN_D - 2])) == 3
    assert not np.allclose(p_home, p_away), "both sides read the same plan"


def test_assemble_keeps_the_four_block_width():
    vec = sc.assemble(np.zeros(STATE_D), np.zeros(ROLE_D),
                      np.zeros(MEMORY_D), np.zeros(PLAN_D))
    assert vec.shape == (STATE_D + ROLE_D + MEMORY_D + PLAN_D,)
