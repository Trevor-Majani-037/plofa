#!/usr/bin/env python3
"""Tests for the outcome-driven full-match fitness generalisation.

Covers: extract_outcome_signals (side inference, xG, possession, turnovers,
shot counting), outcome_fitness (normalisation, weight blend, clamping), and
evaluate_full_match (multi-match averaging over a synthetic build_engine).
"""
from __future__ import annotations

from types import SimpleNamespace

from brain_evolution import (
    extract_outcome_signals,
    outcome_fitness,
    evaluate_full_match,
    OutcomeFitnessConfig,
    OUTCOME_DEFAULT_WEIGHTS,
)


# ── helpers ─────────────────────────────────────────────────

def _ev(event_type: str, team: str, player: str,
        metadata: dict | None = None, xg: float = 0.0):
    return SimpleNamespace(
        event_type=SimpleNamespace(name=event_type),
        team=team,
        player=player,
        metadata=metadata or {},
        xg=xg,
    )


class _Profile:
    """Cheap stand-in for PlayerProfile when player attr is an object."""
    def __init__(self, name: str):
        self.name = name


def _result(home_goals: int = 0, away_goals: int = 0,
            home_xg: float = 0.0, away_xg: float = 0.0,
            poss_home: float = 50.0, events: list | None = None):
    return SimpleNamespace(
        home_goals=home_goals,
        away_goals=away_goals,
        home_xg=home_xg,
        away_xg=away_xg,
        home_possession_pct=poss_home,
        timeline=events or [],
    )


# ── extract_outcome_signals ────────────────────────────────

def test_home_side_default_when_no_player():
    r = _result(home_xg=2.1, away_xg=0.6, poss_home=58.0, home_goals=3, away_goals=1)
    s = extract_outcome_signals(r, target_player=None)
    assert s["team"] == "Home"
    assert abs(s["xg_diff"] - 1.5) < 1e-9
    assert abs(s["goal_diff"] - 2.0) < 1e-9
    assert s["possession"] == 58.0
    assert s["turnover_rate"] == 0.0
    assert s["own_shots"] == 0
    print("  PASS test_home_side_default_when_no_player")


def test_away_side_inferred_from_player_rows():
    brain_meta = {"active_brain": {"intent": "PASS", "is_error": False}}
    events = [
        _ev("CARRY", "Away", "Cole", brain_meta),
        _ev("CARRY", "Away", "Cole", {"active_brain": {"intent": "SHOT", "is_error": True}}),
        _ev("SHOT_ON_TARGET", "Away", "Cole", xg=0.4),
    ]
    r = _result(home_xg=1.0, away_xg=2.5, poss_home=60.0, events=events)
    s = extract_outcome_signals(r, target_player="Cole")
    assert s["team"] == "Away"
    assert abs(s["xg_diff"] - 1.5) < 1e-9       # away - home
    assert abs(s["possession"] - 40.0) < 1e-9   # 100 - poss_home
    assert s["n_touches"] == 2
    assert s["n_errors"] == 1
    assert abs(s["turnover_rate"] - 0.5) < 1e-9
    assert s["own_shots"] == 1
    print("  PASS test_away_side_inferred_from_player_rows")


def test_player_profile_object_normalised():
    """player attr is a profile object, not a raw string — _name() must handle."""
    p = _Profile("Maxwell Reed")
    events = [
        _ev("CARRY", "Home", p, {"active_brain": {"intent": "DRIBBLE", "is_error": False}}),
    ]
    r = _result(events=events)
    s = extract_outcome_signals(r, target_player="Maxwell Reed")
    assert s["n_touches"] == 1
    assert s["n_errors"] == 0
    print("  PASS test_player_profile_object_normalised")


def test_shot_event_counting():
    """Only the candidate's own shot events count toward own_shots."""
    events = [
        _ev("SHOT_ON_TARGET", "Home", "Alice", xg=0.2),      # own
        _ev("SHOT_OFF_TARGET", "Home", "Alice", xg=0.1),     # own
        _ev("SHOT_BLOCKED", "Home", "Alice", xg=0.3),        # own
        _ev("GOAL", "Home", "Alice", xg=0.5),                # own
        _ev("HIT_WOODWORK", "Home", "Alice", xg=0.4),        # own
        _ev("SHOT_ON_TARGET", "Home", "Bob", xg=0.7),        # not own
        _ev("PASS", "Home", "Alice"),                         # not a shot
    ]
    r = _result(events=events)
    s = extract_outcome_signals(r, target_player="Alice")
    assert s["own_shots"] == 5
    print("  PASS test_shot_event_counting")


def test_team_override_beats_fragile_inference():
    """Explicit team param wins over timeline inference (duplicate names)."""
    # Both squads have a player literally named "ST"; first timeline hit is
    # the AWAY one — naive inference would mis-credit the wrong side.
    events = [
        _ev("SHOT_ON_TARGET", "Away", "ST", xg=0.5),
        _ev("CARRY", "Home", "ST", {"active_brain": {"intent": "SHOT", "is_error": False}}),
    ]
    r = _result(home_xg=2.0, away_xg=0.4, poss_home=58.0, events=events)

    wrong = extract_outcome_signals(r, target_player="ST", team=None)
    assert wrong["team"] == "Away"
    assert abs(wrong["xg_diff"] - (-1.6)) < 1e-9

    right = extract_outcome_signals(r, target_player="ST", team="home")
    assert right["team"] == "Home"
    assert abs(right["xg_diff"] - 1.6) < 1e-9
    assert abs(right["possession"] - 58.0) < 1e-9
    print("  PASS test_team_override_beats_fragile_inference")


# ── outcome_fitness ────────────────────────────────────────

def test_parity():
    """Diff = 0 and 50% possession is a neutral 0.5 on xg/goal axes, 0.5 poss."""
    signals = {"xg_diff": 0.0, "goal_diff": 0.0, "possession": 50.0,
               "turnover_rate": 0.0, "n_touches": 10, "n_errors": 0,
               "own_shots": 4}
    f = outcome_fitness(signals)
    # xg=0.5, goal=0.5, poss=0.5, retain=1.0, chance=0.5
    # weighted mean
    w = OUTCOME_DEFAULT_WEIGHTS
    total = sum(w.values())
    expected = (w["xg_diff"]*0.5 + w["goal_diff"]*0.5 + w["possession"]*0.5
                + w["turnover"]*1.0 + w["chances"]*0.5) / total
    assert abs(f - expected) < 1e-9
    assert 0.45 <= f <= 0.75
    print("  PASS test_parity")


def test_advantage_monotone():
    """Better xG diff → higher fitness, all else equal."""
    base = {"xg_diff": 0.0, "goal_diff": 0.0, "possession": 50.0,
            "turnover_rate": 0.0, "n_touches": 0, "n_errors": 0,
            "own_shots": 0}
    f_even = outcome_fitness(base)
    f_adv  = outcome_fitness({**base, "xg_diff": 1.5})
    f_big  = outcome_fitness({**base, "xg_diff": 3.0})
    assert f_adv > f_even, f"{f_adv} > {f_even}"
    assert f_big > f_adv,  f"{f_big} > {f_adv}"
    print("  PASS test_advantage_monotone")


def test_disadvantage_penalises():
    """Negative diff → lower fitness."""
    base = {"xg_diff": 0.0, "goal_diff": 0.0, "possession": 50.0,
            "turnover_rate": 0.0, "n_touches": 0, "n_errors": 0,
            "own_shots": 0}
    f_neg  = outcome_fitness({**base, "xg_diff": -2.0})
    f_even = outcome_fitness(base)
    assert f_neg < f_even
    print("  PASS test_disadvantage_penalises")


def test_high_turnover_reduces_fitness():
    f_clean = outcome_fitness({"xg_diff": 0.0, "goal_diff": 0.0,
                               "possession": 50.0, "turnover_rate": 0.0,
                               "n_touches": 20, "n_errors": 0,
                               "own_shots": 0})
    f_messy = outcome_fitness({"xg_diff": 0.0, "goal_diff": 0.0,
                               "possession": 50.0, "turnover_rate": 0.15,
                               "n_touches": 20, "n_errors": 3,
                               "own_shots": 0})
    f_catast = outcome_fitness({"xg_diff": 0.0, "goal_diff": 0.0,
                                "possession": 50.0, "turnover_rate": 0.30,
                                "n_touches": 20, "n_errors": 6,
                                "own_shots": 0})
    assert f_clean > f_messy > f_catast
    print("  PASS test_high_turnover_reduces_fitness")


def test_clamped_to_unit():
    """Final value always sits in [0, 1]."""
    extreme = {"xg_diff": 10.0, "goal_diff": 10.0, "possession": 100.0,
               "turnover_rate": 0.0, "n_touches": 50, "n_errors": 0,
               "own_shots": 100}
    f = outcome_fitness(extreme)
    assert 0.0 <= f <= 1.0, f"out of range: {f}"
    terrible = {"xg_diff": -10.0, "goal_diff": -10.0, "possession": 0.0,
                "turnover_rate": 1.0, "n_touches": 20, "n_errors": 20,
                "own_shots": 0}
    f2 = outcome_fitness(terrible)
    assert 0.0 <= f2 <= 1.0, f"out of range: {f2}"
    print("  PASS test_clamped_to_unit")


# ── evaluate_full_match ───────────────────────────────────

def test_evaluate_full_match_averages():
    """A canned build_engine that always returns the same result → fitness matches single-call."""
    from brain_evolution import outcome_fitness, extract_outcome_signals
    from football_brain import FootballBrain

    canned = _result(home_xg=1.8, away_xg=0.5, home_goals=2,
                     away_goals=0, poss_home=55.0)
    call_count = [0]
    def build_engine(brain):
        call_count[0] += 1
        return canned

    brain = FootballBrain.random(seed=42)
    f_multi = evaluate_full_match(brain, build_engine, n_matches=4)
    single = outcome_fitness(extract_outcome_signals(canned))
    assert abs(f_multi - single) < 1e-9, f"{f_multi} != {single}"
    assert call_count[0] == 4
    assert 0.0 <= f_multi <= 1.0
    print("  PASS test_evaluate_full_match_averages")


def test_evaluate_full_match_varied_build_engine():
    """Different build_engine calls produce different results → averaged."""
    from football_brain import FootballBrain

    results = [
        _result(home_xg=2.5, away_xg=0.1, home_goals=3, away_goals=0, poss_home=60.0),
        _result(home_xg=0.5, away_xg=1.8, home_goals=0, away_goals=2, poss_home=40.0),
    ]
    idx = [0]
    def build_engine(brain):
        r = results[idx[0] % len(results)]
        idx[0] += 1
        return r

    brain = FootballBrain.random(seed=7)
    f = evaluate_full_match(brain, build_engine, n_matches=2)
    # each result produces a different fitness; average should be between them
    from brain_evolution import outcome_fitness, extract_outcome_signals
    f0 = outcome_fitness(extract_outcome_signals(results[0]))
    f1 = outcome_fitness(extract_outcome_signals(results[1]))
    avg = (f0 + f1) / 2.0
    assert abs(f - avg) < 1e-9
    assert 0.0 <= f <= 1.0
    print("  PASS test_evaluate_full_match_varied_build_engine")


# ── run all ────────────────────────────────────────────────

if __name__ == "__main__":
    print("test_fullmatch_outcome")
    test_home_side_default_when_no_player()
    test_away_side_inferred_from_player_rows()
    test_player_profile_object_normalised()
    test_shot_event_counting()
    test_team_override_beats_fragile_inference()
    test_parity()
    test_advantage_monotone()
    test_disadvantage_penalises()
    test_high_turnover_reduces_fitness()
    test_clamped_to_unit()
    test_evaluate_full_match_averages()
    test_evaluate_full_match_varied_build_engine()
    print("ALL PASSED")
