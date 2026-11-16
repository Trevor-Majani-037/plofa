"""The exploration wiring — the trainer's collection can now cover rare actions.

WHY THIS FILE EXISTS (2026-11-08)
---------------------------------
``surrogate_rebuild._run_matches`` had no ``explore`` parameter, so
``surrogate_collect._EXPLORE_EPS`` kept its module default of ``0.0`` on the
trainer's path.  The ε-exploration machinery — which exists precisely so the
corpus contains actions the policy never chooses, and which weights
``goal_close → SHOOT`` at 8.0 — was unreachable from the trainer.  A brain that
never shoots therefore collected a corpus with no shooting in it.

These tests pin three things: the exploration is off at 0.0, it draws only from
the role's real vocabulary when on, it is situation-weighted, and — the subtle
one — the module globals are restored afterwards so an exploration rate cannot
leak into the gate arm that runs later in the same process.
"""
import random
from types import SimpleNamespace

import numpy as np
import pytest

import surrogate_collect as sc
import surrogate_rebuild as sr


class _Player:
    def __init__(self, position="CM"):
        self.position = position
        self.name = "Tester"


def _sensors(goal_close=True):
    """A 24-d sensor vector whose situation phase is goal_close or open."""
    s = np.zeros(24, dtype=np.float64)
    s[0], s[1] = 0.8, 0.5          # near the attacking goal
    s[12] = 1.0 if goal_close else 0.0   # is_final_third
    s[13] = 0.0                    # is_own_half
    s[14] = 0.2 if goal_close else 0.9   # goal_distance_norm
    s[15] = 1.0                    # central lane
    return s


# ── _explore_intent ──────────────────────────────────────────────────────

def test_exploration_off_returns_the_policy_choice(monkeypatch):
    monkeypatch.setattr(sc, "_EXPLORE_EPS", 0.0)
    monkeypatch.setattr(sc, "_EXPLORE_RNG", random.Random(0))
    for _ in range(50):
        assert sc._explore_intent(_Player(), sc.PlayerIntent.CARRY,
                                  _sensors()) is sc.PlayerIntent.CARRY


def test_exploration_on_draws_from_the_roles_own_vocabulary(monkeypatch):
    """A midfielder must never be asked to do something a midfielder cannot do."""
    from role_features import ROLE_INTENTS, _role_family

    monkeypatch.setattr(sc, "_EXPLORE_EPS", 1.0)
    monkeypatch.setattr(sc, "_EXPLORE_SITUATIONAL", False)
    monkeypatch.setattr(sc, "_EXPLORE_RNG", random.Random(1))

    fam = _role_family("CM")
    allowed = set(ROLE_INTENTS[fam])
    p = _Player("CM")
    drawn = {sc._explore_intent(p, sc.PlayerIntent.CARRY).value
             for _ in range(400)}
    assert drawn <= allowed, f"drew {drawn - allowed} outside the CM vocabulary"


def test_situational_exploration_favours_shooting_near_goal(monkeypatch):
    """goal_close weights SHOOT at 8.0; PROGRESSIVE_PASS at 2.0.

    NOTE ON THE ROLE.  This uses a STRIKER, not a midfielder.  The first
    version of this test used ``CM`` and failed with a SHOOT share of 0.00 —
    the test was wrong, not the code.  ``ROLE_INTENTS["CM"]`` deliberately
    holds no ``SHOOT`` ("a central midfielder: the through ball is his
    signature"), so the 8.0 weight is unreachable by design, and the
    situation weighting is filtered through the role's vocabulary before it
    is applied.  The striker's reachable set is SHOOT 8.0, THROUGH_BALL 3.0,
    PROGRESSIVE_PASS 2.0, SWITCH 1.5 (no CROSS in the ST vocabulary), so the
    expected share is 8/14.5 = 0.55.
    """
    monkeypatch.setattr(sc, "_EXPLORE_EPS", 1.0)
    monkeypatch.setattr(sc, "_EXPLORE_SITUATIONAL", True)
    monkeypatch.setattr(sc, "_EXPLORE_RNG", random.Random(7))

    p = _Player("ST")
    s = _sensors(goal_close=True)
    assert sc._situation_phase(s) == "goal_close"

    n = 2000
    shots = sum(1 for _ in range(n)
                if sc._explore_intent(p, sc.PlayerIntent.CARRY, s)
                is sc.PlayerIntent.SHOOT)
    share = shots / n
    assert share > 0.25, f"SHOOT share {share:.2f} too low for an 8.0 weight"


def test_situation_weighting_never_overrides_the_role(monkeypatch):
    """A CM in the box is still never explored as a shooter.

    This is the guard the failed test accidentally discovered: the phase
    weights are an ATTENTION budget over the role's vocabulary, not a
    licence to hand a role an action it does not own.  If this ever starts
    failing, the corpus has begun teaching midfielders to shoot — which is
    a change to the football model, not to the exploration.
    """
    from role_features import ROLE_INTENTS, _role_family

    monkeypatch.setattr(sc, "_EXPLORE_EPS", 1.0)
    monkeypatch.setattr(sc, "_EXPLORE_SITUATIONAL", True)
    monkeypatch.setattr(sc, "_EXPLORE_RNG", random.Random(11))

    assert "SHOOT" not in ROLE_INTENTS[_role_family("CM")]
    p = _Player("CM")
    s = _sensors(goal_close=True)
    assert sc._situation_phase(s) == "goal_close"

    drawn = {sc._explore_intent(p, sc.PlayerIntent.CARRY, s).value
             for _ in range(1000)}
    assert "SHOOT" not in drawn, "a CM was explored as a shooter"
    # and the weighting still bites inside the role's own vocabulary: the
    # through ball is weighted 3.0 there, the switch 1.5.
    assert "THROUGH_BALL" in drawn and "SWITCH" in drawn


def test_situational_exploration_does_not_shoot_from_own_half(monkeypatch):
    """own_half weights carry no SHOOT at all."""
    monkeypatch.setattr(sc, "_EXPLORE_EPS", 1.0)
    monkeypatch.setattr(sc, "_EXPLORE_SITUATIONAL", True)
    monkeypatch.setattr(sc, "_EXPLORE_RNG", random.Random(3))

    s = np.zeros(24, dtype=np.float64)
    s[0], s[1] = 0.2, 0.5
    s[12], s[13], s[14] = 0.0, 1.0, 0.95
    assert sc._situation_phase(s) == "own_half"

    p = _Player("CM")
    for _ in range(500):
        assert sc._explore_intent(p, sc.PlayerIntent.CARRY, s) \
            is not sc.PlayerIntent.SHOOT


# ── the wiring: set during the run, restored after it ────────────────────

def test_run_matches_sets_exploration_and_restores_it(monkeypatch):
    """The leak that would corrupt the gate: exploration must not survive."""
    seen = {}

    class _Eng:
        def simulate(self):
            seen["eps"] = sc._EXPLORE_EPS
            seen["situational"] = sc._EXPLORE_SITUATIONAL
            # A real MatchResult always carries a timeline, and _run_matches
            # REQUIRES one (see tests/test_shoot_credit.py: without it every
            # shoot decision is credited 0.0).  The fake must not be lazier
            # than the thing it stands in for.
            return SimpleNamespace(timeline=[])

    monkeypatch.setattr(sr, "_build_match_components",
                        lambda *a, **k: _Eng())
    monkeypatch.setattr(sr, "_correlate_outcomes_causal", lambda calls, **kw: [])

    sc._EXPLORE_EPS = 0.0
    sc._EXPLORE_SITUATIONAL = False

    out = sr._run_matches(1, 11, ["fluid_counter"], True, "neural",
                          explore=0.25, explore_situational=True)

    assert out == []
    assert seen["eps"] == pytest.approx(0.25), "exploration never reached the match"
    assert seen["situational"] is True
    assert sc._EXPLORE_EPS == 0.0, "exploration leaked past the run"
    assert sc._EXPLORE_SITUATIONAL is False


def test_run_matches_defaults_to_no_exploration(monkeypatch):
    """The historical behaviour must be the default, byte for byte."""
    seen = {}

    class _Eng:
        def simulate(self):
            seen["eps"] = sc._EXPLORE_EPS
            return SimpleNamespace(timeline=[])

    monkeypatch.setattr(sr, "_build_match_components", lambda *a, **k: _Eng())
    monkeypatch.setattr(sr, "_correlate_outcomes_causal", lambda calls, **kw: [])

    sr._run_matches(1, 11, ["fluid_counter"], True, "neural")
    assert seen["eps"] == 0.0
