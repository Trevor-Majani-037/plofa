"""The temperament seam, end to end, through the real production brain.

WHY THIS IS SEPARATE FROM ``test_temperament_seam.py``
------------------------------------------------------
That file tests ``_apply_temper`` in isolation — the arithmetic and the guards.
This one proves the wiring: a scarred player making a REAL decision with a REAL
shipped brain gets a measurably different distribution, and the direction is the
one football expects.

The distinction matters because the bug being repaired was never an arithmetic
bug.  ``mind.temper`` was correct and unreachable.  A unit test of a correct
unreachable function passes on the broken system, which is exactly what happened
for months.

MEASURED (brains/CAM.json, carrier at (40, 30), seed 99):

    risk 1.0  conf 1.0  -> SHOOT  p=0.8360  shift=0.0000   (untouched)
    risk 0.4  conf 1.0  -> SHOOT  p=0.8250  shift=0.0462   (fearful: down)
    risk 1.0  conf 2.0  -> SHOOT  p=0.9650  shift=0.1296   (confident: up)
    risk 1.0  conf 0.3  -> SHOOT  p=0.4350  shift=0.3990   (indecisive: down)

The assertions below are relative, not numeric, so a retrained brain does not
break the test while a broken seam still does.
"""
import os
import random
import sys

import pytest

# The fixture lives in a sibling test module, and the repo root must be on the
# path for the project imports.  pytest inserts the ROOTDIR, not ``tests/``, so
# both are added explicitly rather than relying on how pytest was invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from situational_brain import SituationalBrain
from brain_integration import register_brain, clear_registry
from cognition.mind import register_mind, clear_minds, new_mind
from situational_decision import SituationalDecisionBrain

from test_cognition_brain import _scene, FakeTeamProfile, FakeGameState

#: The shipped on-ball brain used as the carrier's policy.
BRAIN_PATH = os.path.join(_ROOT, "brains", "CAM.json")


@pytest.fixture(autouse=True)
def _arm_temper(monkeypatch):
    """Arm the temperament switch for this module only.

    Since 2026-11-16 the seam sits behind ``PLOFA_TEMPER``, because fixing the
    situational ROUTING made ``decide()`` the live path for the first time and
    the routing fix had to be provably behaviour-neutral.  See
    ``situational_decision._temper_armed`` for the full reason.

    This is not a weakening of the test: it proves the seam works when asked
    for.  ``test_the_switch_is_off_by_default`` below proves it is not applied
    when nobody asked.
    """
    monkeypatch.setenv("PLOFA_TEMPER", "1")


@pytest.fixture
def live():
    """A real carrier, a real shipped brain, a real mind — then clean up."""
    clear_registry()
    clear_minds()
    carrier, engine, team, defs = _scene()
    brain = SituationalBrain.load(BRAIN_PATH)
    register_brain(carrier.name, brain)
    mind = new_mind()
    register_mind(carrier.name, mind)
    try:
        yield carrier, engine, team, defs, mind
    finally:
        clear_registry()
        clear_minds()


def _decide(carrier, engine, team, defs, mind, risk=1.0, conf=1.0):
    mind.experience.bias["risk_in_box"] = risk
    mind.experience.bias["confidence"] = conf
    random.seed(99)
    return SituationalDecisionBrain.decide(
        carrier, 40.0, 30.0, team, defs, engine,
        FakeTeamProfile("balanced"), False, True, FakeGameState("LEVEL"),
        45.0, record_trace=True,
    )


def test_a_neutral_mind_changes_nothing(live):
    """The untouched path — this is what the shipped game has been running."""
    carrier, engine, team, defs, mind = live
    d = _decide(carrier, engine, team, defs, mind, risk=1.0, conf=1.0)
    assert d.trace["temperament"]["shift"] == pytest.approx(0.0, abs=1e-9)


def test_fear_makes_a_risky_intent_less_likely(live):
    carrier, engine, team, defs, mind = live
    neutral = _decide(carrier, engine, team, defs, mind, risk=1.0)
    fearful = _decide(carrier, engine, team, defs, mind, risk=0.4)

    assert fearful.trace["temperament"]["shift"] > 0.0, "the scar never landed"
    if neutral.intent is fearful.intent and neutral.intent.value in {
            "SHOOT", "DRIBBLE", "THROUGH_BALL", "CROSS", "SWITCH"}:
        assert fearful.confidence < neutral.confidence, (
            "a player burned in the box committed just as hard to a risky intent")


def test_confidence_sharpens_and_flattens_a_real_decision(live):
    carrier, engine, team, defs, mind = live
    neutral = _decide(carrier, engine, team, defs, mind, conf=1.0)
    firm = _decide(carrier, engine, team, defs, mind, conf=2.0)
    unsure = _decide(carrier, engine, team, defs, mind, conf=0.3)

    assert firm.confidence > neutral.confidence, "a confident player must commit"
    assert unsure.confidence < neutral.confidence, "an unsure player must waver"


def test_a_disengaged_mind_leaves_the_decision_identical(live):
    """The documented contract: no registered mind -> byte-identical path.

    COMPARED AGAINST NO MIND AT ALL, NOT AGAINST AN ENGAGED ONE.
    -----------------------------------------------------------
    The first version of this test compared an ENGAGED mind against a
    disengaged one and asserted the intents were equal.  That is not the
    contract: an engaged mind also gates the FIELD OF VIEW
    (``mind.perceive``), so the two decisions are taken on different scenes —
    the trace showed 4 tokens against 2.  It passed only because the two
    scenes happened to break the same way at seed 99.

    Fixing the entity tokens (the SPACE feature, 2026-11-16) changed the
    tie-break and the coincidence disappeared.  The honest comparison is
    disengaged-mind vs no-mind, because both skip the FOV gate and therefore
    see the SAME scene — so any difference could only come from temperament.
    """
    carrier, engine, team, defs, mind = live
    scarred = _decide(carrier, engine, team, defs, mind, risk=0.4)
    assert scarred.trace["temperament"] is not None
    assert scarred.trace["temperament"]["shift"] > 0.0

    mind.engaged = False
    disengaged = _decide(carrier, engine, team, defs, mind, risk=0.4)

    clear_minds()
    no_mind = _decide(carrier, engine, team, defs, mind, risk=0.4)

    assert disengaged.trace["temperament"] is None
    assert no_mind.trace["temperament"] is None
    assert disengaged.intent is no_mind.intent, (
        "a disengaged mind must leave the decision exactly as it would be "
        "with no mind at all")


def test_the_switch_is_off_by_default(live, monkeypatch):
    """Without ``PLOFA_TEMPER=1`` a scarred player is NOT distorted.

    This is the counterpart to the autouse fixture above, and it exists because
    of a specific near-miss.  ``auto_run_match`` registers a mind for every
    player; ``match_probe`` registers none.  So when the routing fix made
    ``decide()`` live, temperament would have switched itself on in every season
    run while every gated experiment showed no change.  A silent behaviour
    change hidden inside a routing fix is exactly the class of bug this whole
    session was about.
    """
    monkeypatch.delenv("PLOFA_TEMPER", raising=False)
    carrier, engine, team, defs, mind = live
    decision = _decide(carrier, engine, team, defs, mind, risk=0.4)
    assert decision.trace["temperament"] is None, (
        "temperament was applied without PLOFA_TEMPER=1")
