"""The situational ROUTE is a router, not a second copy of the pipeline.

WHY THIS FILE EXISTS
--------------------
``brain_integration._maybe_situational`` used to contain a complete duplicate of
the situational pipeline instead of delegating to ``situational_decision``.  The
duplicate is what every match ran; ``SituationalDecisionBrain`` was imported by
nothing but its own tests.

That cost real work, and it cost it silently:

* the shot read-out was wired, tested and gated against a pipeline that never
  executed, and the gate returned bit-identical matches to four decimals;
* the temperament reconnection was written for the same reason and was equally
  dead;
* every measurement taken through ``SituationalDecisionBrain`` described a
  program that did not run.

The lesson is not "write fewer duplicates".  It is that a test which exercises
a module directly cannot tell whether the ENGINE calls it.  So the tests below
split into two kinds:

  * behaviour — the router delegates, degrades, and refuses correctly;
  * structure — the router stays a router.  A source-length guard is crude, but
    the failure mode it prevents is a 129-line body quietly growing back, and a
    crude guard that fires beats an elegant one that does not exist.
"""
import inspect
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brain_integration
import situational_decision
from brain_integration import (
    _maybe_situational, clear_registry, register_brain,
)
from football_brain import FootballBrain
from situational_brain import SituationalBrain

from test_cognition_brain import FakeGameState, FakeTeamProfile, _scene

BRAIN_PATH = os.path.join(_ROOT, "brains", "CAM.json")


@pytest.fixture
def scene():
    clear_registry()
    carrier, engine, team, defs = _scene()
    try:
        yield carrier, engine, team, defs
    finally:
        clear_registry()


def _route(carrier, engine, team, defs):
    return _maybe_situational(
        carrier, 40.0, 30.0, team, defs, engine,
        FakeTeamProfile("balanced"), False, True, FakeGameState("LEVEL"),
        45.0, None, False, None,
    )


def test_a_situational_brain_is_routed_through_the_shared_pipeline(scene, monkeypatch):
    """The one assertion that would have caught the fork in seconds."""
    carrier, engine, team, defs = scene
    register_brain(carrier.name, SituationalBrain.load(BRAIN_PATH))

    calls = []
    real = situational_decision.SituationalDecisionBrain.decide

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(
        situational_decision.SituationalDecisionBrain, "decide",
        staticmethod(spy))

    decision = _route(carrier, engine, team, defs)

    assert calls, ("the situational route served a decision WITHOUT entering "
                   "SituationalDecisionBrain.decide — the pipeline has been "
                   "forked again")
    assert decision is not None, "the route returned nothing for a situational brain"


def test_a_plain_football_brain_is_not_routed(scene, monkeypatch):
    """The route is chosen by brain TYPE, so a flat brain must fall through."""
    carrier, engine, team, defs = scene
    register_brain(carrier.name, FootballBrain.random())

    called = []
    monkeypatch.setattr(
        situational_decision.SituationalDecisionBrain, "decide",
        staticmethod(lambda *a, **kw: called.append(1)))

    assert _route(carrier, engine, team, defs) is None
    assert not called, "a FootballBrain was pushed down the situational pipeline"


def test_a_failing_pipeline_degrades_instead_of_killing_the_match(scene, monkeypatch):
    """A broken situational path must return None so the engine can fall back."""
    carrier, engine, team, defs = scene
    register_brain(carrier.name, SituationalBrain.load(BRAIN_PATH))

    def boom(*a, **kw):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(
        situational_decision.SituationalDecisionBrain, "decide",
        staticmethod(boom))

    assert _route(carrier, engine, team, defs) is None, (
        "an exception inside the situational pipeline must degrade to None, "
        "not propagate into the match loop")


def test_the_router_stays_a_router():
    """A structural guard: the body must not grow back into a pipeline.

    The removed copy was 129 lines of FOV gate, tokens, blocks, forward, plan,
    consequence, temperature and sampling.  A router is a type check, an import
    and a call.  The threshold is generous (90 lines) so that documentation can
    grow freely; it only fires if the IMPLEMENTATION comes back.
    """
    src = inspect.getsource(_maybe_situational)
    body = [ln for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    assert len(body) < 90, (
        f"_maybe_situational is {len(body)} lines of code. It is supposed to "
        "delegate to situational_decision, not reimplement it.")


def test_the_engine_module_references_the_shared_pipeline():
    """A module nobody imports cannot be on the live route.

    This is the exact condition that made every earlier measurement describe a
    program that did not run.
    """
    src = inspect.getsource(brain_integration)
    assert "situational_decision" in src, (
        "brain_integration no longer references situational_decision — the "
        "situational pipeline is unreachable again")


# ─────────────────────────────────────────────────────────────
# GATE 8 ON THE LIVE PATH
# ─────────────────────────────────────────────────────────────

def _score_diff_seen(monkeypatch, attacks_right, game_state_name):
    """Run the route and report the scoreline the SENSORS were given."""
    carrier, engine, team, defs = _scene()
    clear_registry()
    register_brain(carrier.name, SituationalBrain.load(BRAIN_PATH))
    seen = []
    # `perceive` is imported INSIDE decide(), so the patch has to go on the
    # module it is imported from, not on situational_decision.
    import perception
    real = perception.perceive

    def spy(*a, **kw):
        seen.append(kw.get("score_diff"))
        return real(*a, **kw)

    monkeypatch.setattr(perception, "perceive", spy)
    try:
        _maybe_situational(
            carrier, 40.0, 30.0, team, defs, engine,
            FakeTeamProfile("balanced"), False, attacks_right,
            FakeGameState(game_state_name), 45.0, None, False, None,
        )
    finally:
        clear_registry()
    assert seen, "the route never reached perceive — the wiring is broken"
    return seen[0]


def test_an_away_carrier_reads_his_own_scoreline_on_the_live_path(monkeypatch):
    """The engine's state names are from the HOME point of view.

    ``AWAY_CRUISE`` means the AWAY side is cruising — it leads by two — so the
    away carrier is two UP and must be told +2 (the scoreboard says -2).
    ``HOME_CRUISE`` is the case that matters most: the away carrier is two
    DOWN, and before gate 8 he was told +2, i.e. that he was winning.
    """
    assert _score_diff_seen(monkeypatch, False, "AWAY_CRUISE") == 2
    assert _score_diff_seen(monkeypatch, False, "AWAY_AHEAD") == 1
    assert _score_diff_seen(monkeypatch, False, "HOME_CRUISE") == -2


def test_a_home_carrier_reads_the_scoreboard_unchanged_on_the_live_path(monkeypatch):
    assert _score_diff_seen(monkeypatch, True, "HOME_CRUISE") == 2
    assert _score_diff_seen(monkeypatch, True, "HOME_CHASE") == -2
