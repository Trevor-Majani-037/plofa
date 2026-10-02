"""Phase 10 wiring tests (2026-09-20): the three wired-by-default seams.

  1. ManagerMind is the default cerebrum  (USE_MANAGER_BRAIN is True)
  2. The cognition layer is on by default (brain_integration._cognition_enabled)
     and consequence reasoning reaches the MIND path, not just the core.
  3. The evolved off-ball press brains are wired into the press Bernoulli,
     with None fallback for positions without a brain / any error.

Run:  python -m tests.test_full_wiring

Plain assert-based, matching the project's test style (no pytest).
"""
from __future__ import annotations

import random
import numpy as np

import brain_integration as bi
import consequence_decision as cd
import offball_brain_wiring as obw
import match_engine as _me


def _teardown():
    bi.clear_registry()
    from cognition.mind import clear_minds
    clear_minds()


# ─────────────────────────────────────────────────────────────
# 1. Default posture
# ─────────────────────────────────────────────────────────────

def test_phase10_defaults():
    assert _me.USE_MANAGER_BRAIN is True, (
        "manager brain must be the wired-by-default cerebrum")
    assert _me._OFFBALL_POS_BRAIN_WIRED is True, "off-ball brains wired"
    assert bi._cognition_enabled is True, "cognition layer on by default"
    assert cd.default_state() is True, "consequence reasoning on by default"


# ─────────────────────────────────────────────────────────────
# 2. Off-ball wiring: engine hook + None fallback
# ─────────────────────────────────────────────────────────────

def test_offball_hook_fires_during_a_real_match():
    """The press Bernoulli must call _offball_press_prob (and get a float)
    many times over a match, proving the wiring is live in play."""
    from pitch_replay import run_scratch_match

    calls = []
    original = _me.MatchEngine._offball_press_prob

    def counting(self, pname, position, team, ball_x, ball_y, danger_t):
        out = original(self, pname, position, team, ball_x, ball_y, danger_t)
        calls.append((pname, position, team, out))
        return out

    _me.MatchEngine._offball_press_prob = counting
    try:
        run_scratch_match(seed=7, verbose=False)
    finally:
        _me.MatchEngine._offball_press_prob = original

    assert len(calls) > 0, "off-ball press hook never ran"
    # A LW/RW/CM runner must have produced the brain's float at least once.
    brainy = [(p, pos, out) for p, pos, _, out in calls
              if pos in ("LW", "RW", "CM") and isinstance(out, float)]
    assert brainy, "no LW/RW/CM runner got a brain press probability"


def test_offball_brain_none_fallback():
    """Positions without an evolved brain (and any broken state) yield None
    so the engine keeps today's static Bernoulli."""
    obw.clear_caches()
    assert obw._load_position_brain("RB") is None, "no brain for RB"
    assert obw._load_position_brain("XX") is None, "missing file -> None"
    from football_brain import OffBallBrain
    loaded = obw._load_position_brain("LW")
    assert isinstance(loaded, OffBallBrain), "LW brain must load + validate"

    class _Pos:
        states = {}
        team_rosters = {}
        team_attacks_right = {}

    class _Stub:
        position_engine = _Pos()
        active_players = {}

    # states empty -> dead quick path -> None (never reaches the sensors).
    assert obw.offball_press_prob(_Stub(), "P", "LW", "H",
                                  40.0, 30.0, 1.0, 45, 0) is None


# ─────────────────────────────────────────────────────────────
# 3. Reasoning reaches the cognition (mind) path
# ─────────────────────────────────────────────────────────────

class _FakeStyle:
    def __init__(self, v):
        self.value = v


class _FakeTeamProfile:
    def __init__(self, style="balanced"):
        self.style = _FakeStyle(style)


class _FakeGameState:
    name = "LEVEL"


class _FakePositionEngine:
    def __init__(self, positions):
        self.positions = positions

    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def _make_player(name, position):
    from player_dna import (
        PlayerDNA, PlayerProfile, PhysicalAttributes, TechnicalAttributes,
        MentalAttributes, PassingAttributes, BehavioralTendencies,
        PlayerFormState,
    )
    mental = MentalAttributes(vision=55, composure=55, decisions=85,
                              anticipation=55)
    technical = TechnicalAttributes(dribbling=55, ball_control=55,
                                    crossing=50, finishing=50, long_shots=40)
    physical = PhysicalAttributes(pace=60, stamina=65, strength=55)
    passing = PassingAttributes(short_passing=60, long_passing=55,
                                through_balls=45, switch_play=50)
    tendencies = BehavioralTendencies(attempts_dribble=0.25,
                                      plays_through_ball=0.10,
                                      switches_play=0.08, plays_safe=0.50,
                                      crosses_from_wide=0.40,
                                      shoots_from_distance=0.15)
    form = PlayerFormState(confidence=50)
    dna = PlayerDNA(name=name, position=position, physical=physical,
                    technical=technical, mental=mental, passing=passing,
                    tendencies=tendencies, form=form)
    return PlayerProfile(dna=dna, team_name="Home")


def test_reasoning_reaches_cognition_path():
    from football_brain import FootballBrain
    from cognition import new_mind, register_mind
    from brain_integration import NeuralDecisionBrain

    _teardown()
    carrier = _make_player("Carrier", "CM")
    t1 = _make_player("T1", "CM")
    t2 = _make_player("T2", "CM")
    d1 = _make_player("D1", "CB")
    d2 = _make_player("D2", "CB")
    engine = _FakePositionEngine({
        "Carrier": (40.0, 30.0), "T1": (50.0, 30.0), "T2": (80.0, 30.0),
        "D1": (55.0, 32.0), "D2": (90.0, 30.0),
    })
    bi.register_brain("Carrier", FootballBrain.random(seed=7))
    register_mind("Carrier", new_mind(view_radius=25.0))  # sees everyone
    bi.set_cognition(True)
    cd.default_enabled(True)   # ensure reasoning is on for this process
    try:
        d = NeuralDecisionBrain.decide(
            carrier, 40, 30, [t1, t2], [d1, d2], engine,
            _FakeTeamProfile(), False, True, _FakeGameState(),
            minute=55, record_trace=True,
        )
        assert d.intent is not None
        assert d.trace is not None, "trace must be recorded"
        assert "consequence" in d.trace, (
            "the mind path must get the same critic read as the core")
        assert d.trace["consequence"] is not None
    finally:
        _teardown()
        bi.set_cognition(True)   # restore module default after toggling


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(123)
    tests = [
        test_phase10_defaults,
        test_offball_hook_fires_during_a_real_match,
        test_offball_brain_none_fallback,
        test_reasoning_reaches_cognition_path,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
    print(f"\n{passed}/{len(tests)} full-wiring tests passed.")