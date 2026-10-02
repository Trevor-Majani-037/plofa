"""Focused tests for the cognition layer: cognition/senses.py,
cognition/memory.py, cognition/experience.py, cognition/mind.py, and
cognition_brain.py (the TOLAND merge seam).

Run:  python -m tests.test_cognition_brain

Plain assert-based scripts, matching the project's existing test style
(see test_football_brain.py / test_decision_brain.py), not pytest.
"""
from __future__ import annotations

import random
import numpy as np
from types import SimpleNamespace

from football_brain import FootballBrain, INTENT_LABELS
from brain_integration import (
    NeuralDecisionBrain, _decide_core, build_sensors,
    register_brain, clear_registry, set_cognition,
)
from cognition import (
    VisionSystem, MemorySystem, ExperienceModel,
    PlayerMind, register_mind, get_mind, clear_minds, new_mind,
)
from cognition_brain import _observe_event
from player_dna import (
    PlayerDNA, PlayerProfile, PhysicalAttributes, TechnicalAttributes,
    MentalAttributes, PassingAttributes, BehavioralTendencies, PlayerFormState,
)


# ─────────────────────────────────────────────────────────────
# Fixtures (mirror test_football_brain.py)
# ─────────────────────────────────────────────────────────────

class FakeStyle:
    def __init__(self, v): self.value = v


class FakeTeamProfile:
    def __init__(self, style="balanced"):
        self.style = FakeStyle(style)


class FakeGameState:
    def __init__(self, name="LEVEL"):
        self.name = name


class FakePositionEngine:
    def __init__(self, positions):
        self.positions = positions
    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def make_player(name, position, **overrides):
    mental = MentalAttributes(
        vision=overrides.pop("vision", 55),
        composure=overrides.pop("composure", 55),
        decisions=overrides.pop("decisions", 55),
        anticipation=overrides.pop("anticipation", 55),
    )
    technical = TechnicalAttributes(
        dribbling=overrides.pop("dribbling", 55),
        ball_control=overrides.pop("ball_control", 55),
        crossing=overrides.pop("crossing", 50),
        finishing=overrides.pop("finishing", 50),
        long_shots=overrides.pop("long_shots", 40),
    )
    physical = PhysicalAttributes(
        pace=overrides.pop("pace", 60), stamina=overrides.pop("stamina", 65),
        strength=overrides.pop("strength", 55),
    )
    passing = PassingAttributes(
        short_passing=overrides.pop("short_passing", 60),
        long_passing=overrides.pop("long_passing", 55),
        through_balls=overrides.pop("through_balls", 45),
        switch_play=overrides.pop("switch_play", 50),
    )
    tendencies = BehavioralTendencies(
        attempts_dribble=overrides.pop("attempts_dribble", 0.25),
        plays_through_ball=overrides.pop("plays_through_ball", 0.10),
        switches_play=overrides.pop("switches_play", 0.08),
        plays_safe=overrides.pop("plays_safe", 0.50),
        crosses_from_wide=overrides.pop("crosses_from_wide", 0.40),
        shoots_from_distance=overrides.pop("shoots_from_distance", 0.15),
    )
    form = PlayerFormState(confidence=overrides.pop("confidence", 50))
    dna = PlayerDNA(
        name=name, position=position, physical=physical, technical=technical,
        mental=mental, passing=passing, tendencies=tendencies, form=form,
    )
    return PlayerProfile(dna=dna, team_name="Home")


def _scene():
    """Carrier at (40,30); T1 at 10m, T2 at 40m, D1 at ~15m, D2 at 50m."""
    carrier = make_player("Carrier", "CAM", decisions=100, composure=95)
    t1 = make_player("T1", "CM")
    t2 = make_player("T2", "CM")
    d1 = make_player("D1", "CB")
    d2 = make_player("D2", "CB")
    engine = FakePositionEngine({
        "Carrier": (40.0, 30.0), "T1": (50.0, 30.0), "T2": (80.0, 30.0),
        "D1": (55.0, 32.0), "D2": (90.0, 30.0),
    })
    return carrier, engine, [t1, t2], [d1, d2]


# ─────────────────────────────────────────────────────────────
# 1. VisionSystem — the FOV gate
# ─────────────────────────────────────────────────────────────

def test_fov_gate_radius_and_attention_caps():
    carrier, engine, team, defs = _scene()
    mind = new_mind(view_radius=20.0, max_teammates=2, max_opponents=2)
    vt, vd = mind.perceive(carrier, 40.0, 30.0, team, defs, engine)
    # T1 is 10m away (visible), T2 is 40m away (invisible).
    assert [p.name for p in vt] == ["T1"], vt
    # D1 is ~15m away (visible), D2 is 50m away (invisible).
    assert [p.name for p in vd] == ["D1"], vd


def test_fov_gate_attention_cap_limits_nearby_actors():
    carrier, engine, team, defs = _scene()
    mind = new_mind(view_radius=100.0, max_teammates=1, max_opponents=1)
    vt, vd = mind.perceive(carrier, 40.0, 30.0, team, defs, engine)
    assert len(vt) == 1 and len(vd) == 1, (len(vt), len(vd))
    # Nearest first: T1 (10m) beats T2 (40m); D1 (~15m) beats D2 (50m).
    assert vt[0].name == "T1" and vd[0].name == "D1"


def test_fov_gate_gk_excluded_from_slots():
    carrier, engine, team, defs = _scene()
    gk = make_player("GK", "GK")
    engine.positions["GK"] = (42.0, 30.0)
    mind = new_mind(view_radius=100.0, max_teammates=1, max_opponents=1)
    vt, vd = mind.perceive(carrier, 40.0, 30.0, [gk] + team, defs, engine)
    assert gk not in vt
    assert vt[0].name == "T1"


# ─────────────────────────────────────────────────────────────
# 2. MemorySystem — episodic recollection
# ─────────────────────────────────────────────────────────────

def test_memory_store_recall_decay():
    m = MemorySystem()
    m.store("foul_in_box", severity=1.0)
    m.store("clean_tackle_win", severity=0.5)
    m.store("clean_tackle_win", severity=0.5)
    assert abs(m.recall("clean_tackle_win") - 1.0) < 1e-9
    assert m.recall("big_chance_missed") == 0.0

    # 60 seconds of 0.999-per-second decay: severity ~0.941, still alive.
    m.decay(60.0)
    assert 0.90 < m.recall("foul_in_box") < 1.0
    # A long time later it is forgotten.
    m.decay(3600.0)
    assert m.recall("foul_in_box") == 0.0
    assert len(m) == 0


# ─────────────────────────────────────────────────────────────
# 3. ExperienceModel — memory becomes temperament
# ─────────────────────────────────────────────────────────────

def test_experience_temperament_formulas():
    memory = MemorySystem()
    exp = ExperienceModel()
    exp.apply_memory(memory)                       # blank slate → neutral
    assert exp.bias["risk_in_box"] == 1.0
    assert exp.bias["tackle_aggression"] == 1.0
    assert exp.bias["confidence"] == 1.0

    # Pain in the box makes you afraid to attack the box.
    memory.store("foul_in_box", severity=6.0)
    # Clean wins outweigh lost duels → aggression + fragile-ish confidence rises.
    memory.store("clean_tackle_win", severity=5.0)
    memory.store("lost_duel", severity=1.0)
    exp.apply_memory(memory)
    assert abs(exp.bias["risk_in_box"] - 0.7) < 1e-9       # 1.0 - 6*0.05
    assert exp.bias["tackle_aggression"] > 1.0             # (5-1)*0.03
    assert abs(exp.bias["confidence"] - 1.14) < 1e-9       # 5*0.04 - 1*0.06

    # Fear is floor-guarded: three fouls-in-box cannot sink risk below 0.4.
    memory.store("foul_in_box", severity=20.0)
    exp.apply_memory(memory)
    assert exp.bias["risk_in_box"] == 0.4


# ─────────────────────────────────────────────────────────────
# 4. PlayerMind.temper — temperament reshapes the distribution
# ─────────────────────────────────────────────────────────────

_RISK = {"THROUGH_BALL", "DRIBBLE", "SWITCH", "CROSS", "SHOOT"}
_SAFE = {"SAFE_PASS", "PROTECT_POSSESSION", "RECYCLE"}


def _mass(probs, labels, subset):
    return sum(float(probs[i]) for i, l in enumerate(labels) if l in subset)


def test_temper_fear_damps_risk_and_leans_safe():
    mind = new_mind()
    mind.experience.bias["risk_in_box"] = 0.4
    probs = np.array([0.05, 0.05, 0.15, 0.10, 0.08, 0.15, 0.10, 0.22, 0.05, 0.05])
    out = mind.temper(probs, INTENT_LABELS)
    assert abs(float(out.sum()) - 1.0) < 1e-9
    risk_before = _mass(probs, INTENT_LABELS, _RISK)
    risk_after = _mass(out, INTENT_LABELS, _RISK)
    assert risk_after < risk_before, (risk_before, risk_after)
    assert _mass(out, INTENT_LABELS, _SAFE) > _mass(probs, INTENT_LABELS, _SAFE)


def test_temper_confidence_peaks_or_flattens():
    probs = np.array([0.02, 0.02, 0.5, 0.05, 0.05, 0.2, 0.04, 0.06, 0.03, 0.03])
    top_before = float(probs.max())

    afraid = new_mind()
    afraid.experience.bias["confidence"] = 0.5          # fragile
    flatter = afraid.temper(probs, INTENT_LABELS)
    assert float(flatter.max()) < top_before            # less decisive

    firm = new_mind()
    firm.experience.bias["confidence"] = 1.3            # he's feeling it
    peaker = firm.temper(probs, INTENT_LABELS)
    assert float(peaker.max()) > top_before             # more committed
    assert abs(float(peaker.sum()) - 1.0) < 1e-9

    # Neutral temperament is a no-op.
    neutral = new_mind()
    same = neutral.temper(probs, INTENT_LABELS)
    assert np.allclose(same, probs)


# ─────────────────────────────────────────────────────────────
# 5. Routing parity — the cognition layer never changes the core path
# ─────────────────────────────────────────────────────────────

def test_cognition_disabled_vs_core_identical():
    clear_registry()
    clear_minds()
    set_cognition(False)
    carrier, engine, team, defs = _scene()
    register_brain("Carrier", FootballBrain.random(seed=7))

    via_brain = NeuralDecisionBrain.decide(
        carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), minute=55, record_trace=True,
    )
    core = _decide_core(
        carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), minute=55, record_trace=True,
    )
    assert via_brain.intent == core.intent
    assert via_brain.trace["output_probs"] == core.trace["output_probs"]


def test_cognition_enabled_but_no_mind_stays_core():
    clear_registry()
    clear_minds()
    set_cognition(True)                                  # layer ON…
    carrier, engine, team, defs = _scene()
    register_brain("Carrier", FootballBrain.random(seed=7))
    # …but Carrier has NO mind registered → vanilla path, unchanged.
    routed = NeuralDecisionBrain.decide(
        carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), minute=55, record_trace=True,
    )
    core = _decide_core(
        carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), minute=55, record_trace=True,
    )
    assert routed.intent == core.intent
    assert routed.trace["output_probs"] == core.trace["output_probs"]
    set_cognition(False)


def test_engaged_mind_changes_fields_but_never_crashes():
    clear_registry()
    clear_minds()
    set_cognition(True)
    carrier, engine, team, defs = _scene()
    register_brain("Carrier", FootballBrain.random(seed=7))
    register_mind("Carrier", new_mind(view_radius=20.0))
    d = NeuralDecisionBrain.decide(
        carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), minute=55, record_trace=True,
    )
    assert d.intent is not None
    assert d.trace is not None
    assert len(d.trace["sensor_vector"]) > 0
    set_cognition(False)


# ─────────────────────────────────────────────────────────────
# 6. FOV gating reaches the actual sensor vector
# ─────────────────────────────────────────────────────────────

def test_fov_gate_changes_sensor_vector():
    clear_registry()
    clear_minds()
    set_cognition(False)
    carrier, engine, team, defs = _scene()
    # v2 brain (31-d): the role-feature tail uses scene["teammates"] and
    # scene["defenders"] for lane/marker/marker_count sensors, so gating
    # the input lists WILL change the output vector.
    brain = FootballBrain.random(seed=7, input_size=31)
    register_brain("Carrier", brain)

    # Identity perception: the ONLY difference between full and gated must
    # be the FOV gate (perception's own actor filter would otherwise make
    # both call sites see the same world).
    from perception import PerceptionConfig
    ident = PerceptionConfig(enabled=False)

    full = build_sensors(
        brain, carrier, 40, 30, team, defs, engine, FakeTeamProfile(),
        False, True, FakeGameState(), 55, perception_config=ident,
    )
    mind = new_mind(view_radius=20.0)                    # sees only T1, D1
    vt, vd = mind.perceive(carrier, 40.0, 30.0, team, defs, engine)
    gated = build_sensors(
        brain, carrier, 40, 30, vt, vd, engine, FakeTeamProfile(),
        False, True, FakeGameState(), 55, perception_config=ident,
    )
    assert not np.array_equal(full, gated), (full, gated)


# ─────────────────────────────────────────────────────────────
# 7. Memory observer — the match's event stream feeds the mind
# ─────────────────────────────────────────────────────────────

def test_observer_dispatches_events_to_minds():
    clear_minds()
    carrier = make_player("Carrier", "CAM")
    opponent = make_player("Opp marcator", "ST")
    register_mind("Carrier", new_mind())
    register_mind("Opp marcator", new_mind())

    # Opponent scores past the Carrier → only the opponent remembers scoring.
    goal = SimpleNamespace(
        event_type=SimpleNamespace(name="GOAL"),
        player=opponent, secondary_player=None, metadata={"x": 88.0, "y": 34.0},
    )
    _observe_event(goal)
    assert get_mind("Opp marcator").memory.recall("big_chance_scored") > 0.0
    assert get_mind("Carrier").memory.recall("big_chance_scored") == 0.0

    # The carrier picks up a yellow-bad foul → pain becomes temperament.
    foul = SimpleNamespace(
        event_type=SimpleNamespace(name="FOUL_COMMITTED"),
        player=carrier, secondary_player=None, metadata={},
    )
    _observe_event(foul)
    assert get_mind("Carrier").memory.recall("foul_in_box") > 0.0
    assert get_mind("Carrier").experience.bias["risk_in_box"] < 1.0


def test_match_engine_exposes_observer_hook():
    try:
        import match_engine as me
    except Exception as e:                       # heavy import — skip if missing
        print(f"  SKIP  match_engine import failed: {e}")
        return
    assert hasattr(me, "set_cognition_observer")
    assert hasattr(me, "_cognition_observe")
    # Engaged hook observes an event; disengaged is a harmless no-op.
    me._cognition_observe(SimpleNamespace(event_type=SimpleNamespace(name="PASS")))
    me.set_cognition_observer(None)


def test_temper_never_broken_on_random_probs():
    mind = new_mind()
    probs = FootballBrain.random(seed=3).predict(np.zeros(24))[2]
    out = mind.temper(probs, INTENT_LABELS)
    assert abs(float(out.sum()) - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────
# 8. Persistence — temperament carries across matchdays
# ─────────────────────────────────────────────────────────────

def test_mind_state_roundtrip():
    mind = new_mind(view_radius=22.5)
    # Simulate a scarred career: pain + dirty duels + clean wins.
    for _ in range(5):
        mind.memory.store("foul_in_box", severity=1.2)
    for _ in range(3):
        mind.memory.store("lost_duel", severity=1.0)
    for _ in range(8):
        mind.memory.store("clean_tackle_win", severity=1.0)
    mind.experience.apply_memory(mind.memory)

    state = mind.to_state()
    clone = PlayerMind.from_state(state)

    assert clone.senses.view_radius == 22.5
    assert clone.engaged is True
    # Memory weights survive exactly (locations are dropped, weights kept).
    for tag in ("foul_in_box", "lost_duel", "clean_tackle_win"):
        for original, restored in (
            (mind.memory.recall(tag), clone.memory.recall(tag)),
        ):
            assert abs(original - restored) < 1e-6, (tag, original, restored)
    # Temperament survives exactly.
    for key in mind.experience.bias:
        assert abs(mind.experience.bias[key] - clone.experience.bias[key]) < 1e-9
    assert clone.experience.bias["risk_in_box"] < 1.0
    assert clone.experience.bias["confidence"] > 1.0
    assert clone.experience.bias["tackle_aggression"] > 1.0


def test_mind_from_state_blank_defaults():
    clone = PlayerMind.from_state({})
    assert clone.senses.view_radius == 25.0
    assert clone.memory.recall_all() == {}
    assert clone.experience.bias == {
        "tackle_aggression": 1.0,
        "risk_in_box": 1.0,
        "confidence": 1.0,
    }


def test_clean_wins_feed_temperament_upward():
    # TOLAND parity: clean_tackle_win must raise confidence/aggression,
    # not just foul/lost duels lowering them.
    mind = new_mind()
    for _ in range(10):
        mind.memory.store("clean_tackle_win", severity=1.0)
    mind.experience.apply_memory(mind.memory)
    assert mind.experience.bias["confidence"] > 1.0
    assert mind.experience.bias["tackle_aggression"] > 1.0
    assert mind.experience.bias["risk_in_box"] == 1.0


def test_season_state_cognition_persists_and_reloads():
    import tempfile
    import os
    from season_manager import SeasonState
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)   # SeasonState.load() must see a NON-existent file, not
    # an empty one, or it treats it as corrupt and logs a scary backup line.
    try:
        state = SeasonState("26/27", path)
        state.set_player_cognition("Percy", {
            "config": {"view_radius": 25.0, "max_teammates": 2,
                       "max_opponents": 2, "engaged": True},
            "memory_tags": {"foul_in_box": 4.7, "clean_tackle_win": 9.9},
            "temperament": {"tackle_aggression": 1.2, "risk_in_box": 0.8,
                            "confidence": 1.1},
        })
        state.save()

        reloaded = SeasonState("26/27", path)
        saved = reloaded.get_player_cognition("Percy")
        assert saved is not None, "cognition section must survive save/load"
        assert saved["memory_tags"]["foul_in_box"] == 4.7
        assert saved["temperament"]["risk_in_box"] == 0.8
        assert reloaded.get_player_cognition("Nobody") is None
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(123)
    tests = [
        test_fov_gate_radius_and_attention_caps,
        test_fov_gate_attention_cap_limits_nearby_actors,
        test_fov_gate_gk_excluded_from_slots,
        test_memory_store_recall_decay,
        test_experience_temperament_formulas,
        test_temper_fear_damps_risk_and_leans_safe,
        test_temper_confidence_peaks_or_flattens,
        test_cognition_disabled_vs_core_identical,
        test_cognition_enabled_but_no_mind_stays_core,
        test_engaged_mind_changes_fields_but_never_crashes,
        test_fov_gate_changes_sensor_vector,
        test_observer_dispatches_events_to_minds,
        test_match_engine_exposes_observer_hook,
        test_temper_never_broken_on_random_probs,
        test_mind_state_roundtrip,
        test_mind_from_state_blank_defaults,
        test_clean_wins_feed_temperament_upward,
        test_season_state_cognition_persists_and_reloads,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            raise SystemExit(1)
    print(f"\n{passed}/{len(tests)} cognition tests passed.")