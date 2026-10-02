"""Coach-to-player instructions (Phase 8 v1) — unit tests."""
import numpy as np
import pytest
from types import SimpleNamespace

import match_engine as _me
from coach_instructions import (
    CoachInstructions, instructions_for_manager, listener_scale,
)
from brain_integration import _apply_coach_bias, _flank_bonus
from football_brain import INTENT_LABELS


def _mgr(posture):
    return SimpleNamespace(decide=lambda e, t: {}, _current_posture=posture,
                           _current_pressing=0.8)


def _player(dec=70.0, com=70.0):
    mental = SimpleNamespace(decisions=dec, composure=com)
    return SimpleNamespace(name="P", dna=SimpleNamespace(mental=mental))


def test_derive_none_when_off_or_static_or_balanced():
    # The module default is ON since 2026-09-20; pin OFF here to prove the
    # instruction derivation is inert when the brain path is disabled.
    _old = _me.USE_MANAGER_BRAIN
    _me.USE_MANAGER_BRAIN = False
    try:
        assert instructions_for_manager(_mgr("ATTACK")) is None  # flag OFF
        assert instructions_for_manager(None) is None
        assert instructions_for_manager(SimpleNamespace()) is None  # no decide
    finally:
        _me.USE_MANAGER_BRAIN = _old


def test_derive_attack_defend(monkeypatch):
    import attack_patterns as ap
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    att = instructions_for_manager(_mgr("ATTACK"), ap.AttackPattern.OVERLOAD_RIGHT)
    assert att is not None and att.source == "ATTACK"
    assert att.intent_bias["SHOOT"] > 0 and att.width_cmd == 1.0
    assert att.favored_flank == "R"  # overload-right flank
    dfn = instructions_for_manager(_mgr("DEFEND"))
    assert dfn is not None and dfn.source == "DEFEND"
    assert dfn.intent_bias["SHOOT"] < 0 and dfn.width_cmd == -1.0
    assert dfn.favored_flank is None
    assert instructions_for_manager(_mgr("BALANCED")) is None


def test_bias_renorms_and_shifts_mass():
    probs = np.full(10, 0.1)
    i_shoot = INTENT_LABELS.index("SHOOT")
    out = _apply_coach_bias(probs, _player(), {"SHOOT": 0.25})
    assert out.sum() == pytest.approx(1.0)
    assert out[i_shoot] > 0.1  # shoot mass grew
    out2 = _apply_coach_bias(probs, _player(), {"SHOOT": -0.30})
    assert out2[i_shoot] < 0.1  # shoot-less bites
    assert _apply_coach_bias(probs, _player(), None) is probs
    assert _apply_coach_bias(probs, _player(), {}) is probs


def test_maverick_tax():
    probs = np.full(10, 0.1)
    i_shoot = INTENT_LABELS.index("SHOOT")
    pro = _apply_coach_bias(probs, _player(95.0, 95.0), {"SHOOT": 0.25})
    mav = _apply_coach_bias(probs, _player(30.0, 30.0), {"SHOOT": 0.25})
    assert pro[i_shoot] > mav[i_shoot] > 0.1
    assert listener_scale(_player(95.0, 95.0)) == pytest.approx(0.965)
    assert listener_scale(_player(100.0, 100.0)) == pytest.approx(1.0)
    assert listener_scale(_player(0.0, 0.0)) == pytest.approx(0.3)


def test_flank_bonus():
    assert _flank_bonus(60.0, "R", True) == pytest.approx(0.15)
    assert _flank_bonus(10.0, "R", True) == pytest.approx(0.0)
    assert _flank_bonus(10.0, "L", True) == pytest.approx(0.15)
    assert _flank_bonus(60.0, None, True) == pytest.approx(0.0)
    # mirrored side: actual low y == normalised high y when attacking left
    assert _flank_bonus(10.0, "R", False) == pytest.approx(0.15)
