"""
Manager Mind (Phase 3) — tests.
"""
from types import SimpleNamespace

import numpy as np

from manager_mind import ManagerMind


def _sensors():
    return np.array([0.2, 0.5, 0.66, 0.3, 0.1,
                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])


def _player(name, age=25, confidence=50.0, fatigue=0.0, rating=60.0):
    form = SimpleNamespace(confidence=confidence, fatigue_level=fatigue)
    dna = SimpleNamespace(
        age=age, form=form,
        overall_rating=rating, mental=SimpleNamespace(
            vision=55.0, composure=55.0, decisions=55.0),
    )
    return SimpleNamespace(name=name, dna=dna)


# ── Perception filters ─────────────────────────────────────────────

def test_identity_filter_when_neutral():
    # All traits 0.5 → filter_perception is a no-op (returns a copy).
    mind = ManagerMind()
    s = _sensors()
    out = mind.filter_perception(s, None)
    assert np.array_equal(out, s)
    assert out is not s, "must return a copy, not mutate the caller's array"


def test_paranoid_manager_exaggerates_fatigue():
    mind = ManagerMind(dogma=0.8, pressure_baseline=0.2)
    s = _sensors()
    out = mind.filter_perception(s, None)
    assert out[5] >= s[5]          # team_fatigue goes UP
    assert out[7] <= s[7]          # momentum read goes DOWN


def test_optimistic_manager_downplays_fatigue():
    mind = ManagerMind(eq=0.9, pressure_baseline=0.9)
    s = _sensors()
    out = mind.filter_perception(s, None)
    assert out[5] <= s[5]          # team_fatigue goes DOWN
    assert out[7] >= s[7]          # momentum read goes UP


def test_empathy_sways_confidence_with_mood():
    mind = ManagerMind(empathy=0.8, composure_current=0.8)
    s = _sensors()
    out = mind.filter_perception(s, None)
    # mood_bias = (0.8 - 0.5) * 0.2 = +0.06 → confidence read goes up
    assert out[6] > s[6]

    mind_down = ManagerMind(empathy=0.8, composure_current=0.2)
    out_down = mind_down.filter_perception(s, None)
    assert out_down[6] < s[6]


# ── Decision filters ───────────────────────────────────────────────

def test_filter_posture_with_low_dogma_is_identity():
    mind = ManagerMind(dogma=0.4)
    p = np.array([0.2, 0.5, 0.3])
    out = mind.filter_posture(p, None)
    assert np.array_equal(out, p)


def test_filter_posture_with_high_dogma_pulls_toward_preferred():
    mind = ManagerMind(dogma=0.9)
    mind.record_posture_taken("DEFEND")
    mind.record_posture_taken("DEFEND")
    p = np.array([0.2, 0.5, 0.3])   # BALANCED leads
    out = mind.filter_posture(p, None)
    # Dogma pushes probability toward the historical mode (DEFEND).
    assert out[0] >= p[0]
    assert abs(out.sum() - 1.0) < 1e-6


def test_filter_pressing_overcautious_clamp():
    paranoid = ManagerMind(dogma=0.8, pressure_baseline=0.2)
    assert paranoid.filter_pressing(0.9, None) == 0.7
    assert paranoid.filter_pressing(0.1, None) == 0.2


def test_filter_sub_urgency_empathy_bump():
    empathic = ManagerMind(empathy=0.8)
    base = empathic.filter_sub_urgency(0.0, None)
    assert base > 0.0              # +0.1 empathy
    youth = ManagerMind(empathy=0.8, youth_trust=0.8)
    assert youth.filter_sub_urgency(0.0, None) > base   # +0.05 extra


# ── Event callbacks ────────────────────────────────────────────────

def test_eq_restores_composure():
    low_eq = ManagerMind(eq=0.2)
    high_eq = ManagerMind(eq=0.8)
    for _ in range(3):
        low_eq.on_goal_conceded()
        high_eq.on_goal_conceded()
    assert high_eq.composure_current > low_eq.composure_current
    assert high_eq.stress_accumulator >= 0.0


def test_goal_scored_calms():
    mind = ManagerMind(eq=0.5)
    mind.stress_accumulator = 0.9
    mind.composure_current = 0.3
    mind.on_goal_scored()
    assert mind.composure_current > 0.3
    assert mind.stress_accumulator < 0.9


def test_end_of_match_win_boosts_loss_hurts():
    win_mind = ManagerMind(eq=0.5)
    win_mind.composure_current = 0.3
    win_mind.end_of_match(
        SimpleNamespace(config=SimpleNamespace(home_team="Home"),
                        home_goals=2, away_goals=0), "Home")
    assert win_mind.composure_current > 0.3

    loss_mind = ManagerMind(eq=0.5)
    loss_mind.composure_current = 0.7
    loss_mind.end_of_match(
        SimpleNamespace(config=SimpleNamespace(home_team="Home"),
                        home_goals=0, away_goals=2), "Home")
    assert loss_mind.composure_current < 0.7


def test_current_pressure_handling_bounded():
    mind = ManagerMind()
    for scenario in (0.9, 0.0, 1.0, 0.5):
        mind.composure_current = scenario
        mind.stress_accumulator = scenario
        v = mind.current_pressure_handling()
        assert 0.0 <= v <= 1.0


# ── Pedagogy ───────────────────────────────────────────────────────

def test_pedagogy_boosts_growth_for_young():
    mind = ManagerMind(pedagogy=0.9)
    young = _player("Y", age=19)
    assert mind.apply_pedagogy(young, 0.1) > 0.1


def test_pedagogy_ignores_veterans():
    mind = ManagerMind(pedagogy=0.9)
    veteran = _player("V", age=30)
    assert mind.apply_pedagogy(veteran, 0.1) == 0.1


# ── Selection ──────────────────────────────────────────────────────

def test_xi_selection_deterministic():
    mind = ManagerMind()
    players = [_player("B", confidence=80, age=28),
               _player("A", confidence=60, age=22),
               _player("C", confidence=70, age=25, fatigue=90)]
    first = [p.name for p in mind.select_starting_xi(players, 0.5)]
    second = [p.name for p in mind.select_starting_xi(players, 0.5)]
    assert first == second
    assert len(first) == 3


# ── Serialization ──────────────────────────────────────────────────

def test_serialize_roundtrip():
    mind = ManagerMind(dogma=0.8, eq=0.7, empathy=0.9, youth_trust=0.6,
                       rotation_discipline=0.4, pressure_baseline=0.3,
                       pedagogy=0.2)
    mind.record_posture_taken("ATTACK")
    mind.record_posture_taken("ATTACK")
    mind.stress_accumulator = 0.4
    data = mind.serialize()
    assert data["kind"] == "manager_mind"

    restored = ManagerMind.deserialize(data)
    assert restored.dogma == mind.dogma
    assert restored.eq == mind.eq
    assert restored.history_posture_counts == mind.history_posture_counts
    assert restored.stress_accumulator == mind.stress_accumulator
    assert restored.composure_current == mind.composure_current


def test_deserialize_rejects_wrong_kind():
    import pytest
    with pytest.raises(ValueError):
        ManagerMind.deserialize({"kind": "not_a_mind"})