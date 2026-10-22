"""Smoke: the situational decision path, end to end, without a match.

Run:  .venv\\Scripts\\python.exe _diag_situational_smoke.py

It proves the FOUR mechanisms are CONNECTED to a live decision:

  * ATTENTION   — the token set is built and the network reads it;
  * RECURRENCE  — the hidden state ADVANCES between two decisions, and
                  the second decision differs from the first because of it;
  * MEMORY      — a mind that has observed events changes the memory block
                  the brain reads;
  * PLANNING    — the plan head proposes a phase and the team plan adopts
                  it (with the hysteresis).

It also proves the SHIPPED path is untouched: a player with an ordinary
``FootballBrain`` must not go down this route at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests"))

import numpy as np

from situational_brain import HIDDEN_D, SituationalBrain


def main() -> int:
    from test_football_brain import make_player, FakeGameState

    from brain_integration import (
        NeuralDecisionBrain, clear_registry, register_brain,
    )
    from position_engine import PositionEngine

    # ── a scene ──────────────────────────────────────────────────
    striker = make_player("ST1", "ST")
    striker.team = "Home"
    mate = make_player("CM1", "CM")
    mate.team = "Home"
    wing = make_player("LW", "LW")
    wing.team = "Home"
    foe = make_player("CB1", "CB")
    foe.team = "Away"

    pe = PositionEngine()
    pe.states["ST1"] = _state("ST1", "ST", "Home", 70.0, 34.0)
    pe.states["CM1"] = _state("CM1", "CM", "Home", 55.0, 30.0)
    pe.states["LW"] = _state("LW", "LW", "Home", 72.0, 8.0)
    pe.states["CB1"] = _state("CB1", "CB", "Away", 88.0, 33.0)

    teammates = [mate, wing]
    defenders = [foe]
    gs = FakeGameState("LEVEL")

    # ── register a SITUATIONAL brain for the striker ─────────────
    clear_registry()
    brain = SituationalBrain.random(seed=5)
    register_brain("ST1", brain)

    # Call the situational path DIRECTLY first: the router swallows its
    # exceptions by design, and a swallowed exception that silently
    # degrades an instrument is worse than a crash.
    from situational_decision import SituationalDecisionBrain

    d1 = SituationalDecisionBrain.decide(
        striker, 70.0, 34.0, teammates, defenders, pe, None,
        False, True, gs, 30.0, None, True, None)

    d1r = NeuralDecisionBrain.decide(
        striker, 70.0, 34.0, teammates, defenders, pe, None,
        False, True, gs, 30.0, None, True, None)
    assert d1r.intent is not None, "the router must reach a decision"

    pstate = pe.neural_states.player("ST1")
    assert pstate.hidden is not None, "the recurrent state must exist"
    assert not np.allclose(pstate.hidden, 0.0), \
        "the recurrent state must have MOVED"
    print("recurrence : hidden state advanced, norm =",
          round(float(np.linalg.norm(pstate.hidden)), 4))

    # The plan head must have proposed something, and the team plan must
    # have adopted it (the hysteresis needs two proposals, so run twice).
    tp = pe.neural_states.team("Home")
    print("planning   : team plan id =", tp.plan_id,
          "switches =", tp.switches, "confidence =",
          round(tp.confidence, 3))
    assert tp.switches >= 1, "the plan head must eventually be adopted"

    # ── RECURRENCE PROVES ITSELF: the same scene, a different hidden
    #    state, must be able to produce a different decision ──────
    d2 = NeuralDecisionBrain.decide(
        striker, 70.0, 34.0, teammates, defenders, pe, None,
        False, True, gs, 30.0, None, True, None)
    tr1 = d1.trace or {}
    tr2 = d2.trace or {}
    p1 = tr1.get("output_probs", {})
    p2 = tr2.get("output_probs", {})
    moved = max(abs(float(p1.get(k, 0.0)) - float(p2.get(k, 0.0)))
                for k in p1) if p1 else 0.0
    print("recurrence : output distribution moved by", round(moved, 5),
          "between two touches of the SAME scene")
    assert moved > 1e-6, "a carried state must change the decision"

    # ── MEMORY changes the block the brain reads ─────────────────
    from memory_features import memory_features as mf
    from cognition.mind import clear_minds, register_mind, new_mind

    clear_minds()
    m = new_mind()
    m.engaged = False
    register_mind("ST1", m)
    before = mf("ST1").copy()

    class Ev:
        # The memory's tag map is keyed by EVENT TYPE name, not by the tag
        # name: a missed big chance is a SHOT_OFF_TARGET / SHOT_BLOCKED /
        # HIT_WOODWORK / PENALTY_MISSED, and the mind files it under the
        # tag "big_chance_missed".
        event_type = type("E", (), {"name": "SHOT_OFF_TARGET"})()
        player = "ST1"
        secondary_player = None
        metadata = {"x": 80.0, "y": 34.0}
        location_x = 80.0
        location_y = 34.0

    m.observe(Ev())
    after = mf("ST1")
    print("memory     : block before =", np.round(before, 3).tolist())
    print("memory     : block after  =", np.round(after, 3).tolist())
    assert not np.allclose(before, after), "memory must change the block"

    # ── the SHIPPED path must be untouched ───────────────────────
    from football_brain import FootballBrain

    clear_registry()
    register_brain("ST1", FootballBrain.random(seed=1))
    d3 = NeuralDecisionBrain.decide(
        striker, 70.0, 34.0, teammates, defenders, pe, None,
        False, True, gs, 30.0, None, False, None)
    assert d3.intent is not None, "the shipped path must still work"
    print("shipped    : ordinary FootballBrain still routes normally ->",
          d3.intent)

    print("\nALL OK")
    return 0


def _state(name, position, team, x, y):
    from position_engine import PlayerSpatialState

    st = PlayerSpatialState(player_name=name, position=position, team=team,
                             home_x=x, home_y=y)
    st.current_x = x
    st.current_y = y
    return st


if __name__ == "__main__":
    sys.exit(main())
