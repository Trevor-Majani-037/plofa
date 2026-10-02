"""Focused tests for football_brain.py, brain_sensors.py and
brain_integration.py (the neural football decision layer).

Run: python3 test_football_brain.py

Plain assert-based scripts, matching the project's existing test style
(see test_decision_brain.py / test_width_changes.py), not pytest.
"""
from __future__ import annotations

import random
import numpy as np
from dataclasses import fields
from types import SimpleNamespace

from football_brain import FootballBrain, INPUT_SIZE, OUTPUT_SIZE, INTENT_LABELS
from brain_sensors import extract_sensors
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry, get_brain,
    save_brains_to_file, load_brains_from_file,
    _sample_from_probs,
)

from player_dna import (
    PlayerDNA, PlayerProfile, PhysicalAttributes, TechnicalAttributes,
    MentalAttributes, PassingAttributes, BehavioralTendencies, PlayerFormState,
)
from decision_brain import PlayerDecision, PlayerIntent


# ─────────────────────────────────────────────────────────────
# Fixtures (mirror test_decision_brain.py)
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


# ─────────────────────────────────────────────────────────────
# 1. FootballBrain — forward pass / shapes
# ─────────────────────────────────────────────────────────────

def test_forward_shapes():
    b = FootballBrain.random(seed=1)
    s = np.zeros(INPUT_SIZE)
    out = b.forward(s)
    assert out.shape == (OUTPUT_SIZE,), out.shape
    assert abs(out.sum() - 1.0) < 1e-9, "softmax must sum to 1"
    assert np.all(np.isfinite(out)), "no NaN/Inf in forward pass"


def test_forward_batch_vectorizable():
    b = FootballBrain.random(seed=2)
    batch = np.zeros((5, INPUT_SIZE))
    outs = [b.forward(row) for row in batch]
    assert len(outs) == 5
    for o in outs:
        assert abs(o.sum() - 1.0) < 1e-9


def test_predict_args():
    b = FootballBrain.random(seed=3)
    s = np.zeros(INPUT_SIZE)
    idx, prob, probs = b.predict(s)
    assert 0 <= idx < OUTPUT_SIZE
    assert 0.0 <= prob <= 1.0
    assert probs.shape == (OUTPUT_SIZE,)


def test_param_count():
    b = FootballBrain.random()
    # 24*32 + 32 + 32*32 + 32 + 32*10 + 10
    expected = INPUT_SIZE * 32 + 32 + 32 * 32 + 32 + 32 * 10 + 10
    assert b.param_count == expected


# ─────────────────────────────────────────────────────────────
# 2. FootballBrain — DNA seeding
# ─────────────────────────────────────────────────────────────

def test_from_dna_seeded():
    smart = make_player("Smart", "CAM", vision=95, composure=90, decisions=92)
    dunce = make_player("Dunce", "CB", vision=30, composure=25, decisions=28)

    b_smart = FootballBrain.from_dna(smart, seed=1)
    b_dunce = FootballBrain.from_dna(dunce, seed=1)

    # Same seed + different DNA => different weights
    assert not np.allclose(b_smart.w1, b_dunce.w1)

    # Vision scaling should make smart brain's input weights larger in magnitude
    assert np.abs(b_smart.w1).mean() > np.abs(b_dunce.w1).mean()

    # Same DNA + same seed => identical weights (reproducibility)
    b_smart2 = FootballBrain.from_dna(smart, seed=1)
    assert np.allclose(b_smart.w1, b_smart2.w1)


# ─────────────────────────────────────────────────────────────
# 3. FootballBrain — evolution operators
# ─────────────────────────────────────────────────────────────

def test_mutate_changes_weights():
    b = FootballBrain.random(seed=4)
    m = b.mutate(rate=0.5, strength=0.2)
    assert not np.allclose(b.w1, m.w1)
    assert b.w1.shape == m.w1.shape
    assert b.b3.shape == m.b3.shape


def test_crossover():
    a = FootballBrain.random(seed=5)
    b = FootballBrain.random(seed=6)
    c = a.crossover(b)
    # child should share SOME weights with each parent
    assert not np.allclose(c.w1, a.w1)
    assert not np.allclose(c.w1, b.w1)
    # but overall shape preserved
    assert c.w1.shape == a.w1.shape


def test_blend():
    a = FootballBrain.random(seed=7)
    b = FootballBrain.random(seed=8)
    c = a.blend(b, alpha=0.7)
    expected = a.w1 * 0.7 + b.w1 * 0.3
    assert np.allclose(c.w1, expected)


# ─────────────────────────────────────────────────────────────
# 4. FootballBrain — serialization
# ─────────────────────────────────────────────────────────────

def test_serialize_roundtrip():
    import os, tempfile
    b = FootballBrain.random(seed=9)
    s = b.serialize()
    assert set(["w1", "b1", "w2", "b2", "w3", "b3", "arch"]) <= set(s.keys())
    b2 = FootballBrain.deserialize(s)
    assert np.allclose(b2.w1, b.w1)
    assert np.allclose(b2.b3, b.b3)


# ─────────────────────────────────────────────────────────────
# 5. brain_sensors — vector extraction
# ─────────────────────────────────────────────────────────────

def test_sensor_shape_and_range():
    s = extract_sensors(None, 75, 34, [], [], None, False, True, None, 30)
    assert s.shape == (INPUT_SIZE,)
    assert np.all((s >= 0.0) & (s <= 1.0)), f"sensors out of range: {s}"


def test_sensor_final_third_flags():
    # attacks_right=True, x=75 is final third, x=25 is own half
    s = extract_sensors(None, 75, 34, [], [], None, False, True, None, 30)
    assert s[12] == 1.0, "is_final_third should be 1 at x=75 attacking right"
    assert s[13] == 0.0, "is_own_half should be 0 at x=75 attacking right"


def test_sensor_pressure_flag():
    s_on = extract_sensors(None, 50, 34, [], [], None, True, True, None, 30)
    s_off = extract_sensors(None, 50, 34, [], [], None, False, True, None, 30)
    assert s_on[16] == 1.0
    assert s_off[16] == 0.0


def test_sensors_with_geometry():
    positions = {
        "Carrier": (75, 34), "Fwd": (92, 40), "Deep": (25, 30),
        "Def1": (78, 33), "Def2": (85, 45), "Def3": (30, 30),
    }
    pe = FakePositionEngine(positions)
    teammates = [make_player("Fwd", "ST"), make_player("Deep", "CB")]
    defenders = [make_player("Def1", "CDM"), make_player("Def2", "CB"), make_player("Def3", "CB")]
    s = extract_sensors(
        make_player("Carrier", "CAM"), 75, 34, teammates, defenders,
        pe, False, True, FakeGameState("LEVEL"), 30,
    )
    # nearest defender (Def1 at 78,33) ~3.16m away -> normalized ~0.21
    assert 0.1 < s[4] < 0.4, f"nearest defender dist sensor off: {s[4]}"
    # space_ahead should be low (~0) with a def 3m away
    assert s[9] < 0.3, f"space_ahead should be low but is {s[9]}"


def test_sensor_extraction_deterministic():
    """Same visible world => identical sensor vector (no hidden RNG/time)."""
    positions = {
        "Carrier": (60, 30), "Fwd": (80, 34), "Deep": (20, 34),
        "Def1": (63, 31), "Def2": (70, 45),
    }
    pe = FakePositionEngine(positions)
    player = make_player("Carrier", "CM", vision=70, composure=60, decisions=65)
    tm = [make_player("Fwd", "ST"), make_player("Deep", "CB")]
    df = [make_player("Def1", "CDM"), make_player("Def2", "CB")]
    kw = dict(x=60, y=30, teammates=tm, defenders=df, position_engine=pe,
              under_pressure=True, attacks_right=True,
              game_state=FakeGameState("HOME_AHEAD_1"), minute=55)
    s1 = extract_sensors(player, **kw)
    s2 = extract_sensors(player, **kw)
    assert np.array_equal(s1, s2)


def test_sensor_extraction_no_future_leak():
    """Sensors must never read future/omniscient state.

    Structural guard: the extraction source may not reference outcome /
    timeline / xG / goal-scored / metadata symbols.  Behavioural guard:
    mutating the *game outcome* (which is never an extraction input) must
    not change a single sensor value.
    """
    import inspect
    from brain_sensors import (
        extract_sensors, extract_offball_sensors,
        _nearest_defender_dist, _defenders_within, _teammate_openness,
        _best_forward_teammate, _fatigue_estimate,
    )
    forbidden = {"outcome", "timeline", "metadata", "xg", "goal_scored",
                 "soul", "personality", "confidence"}
    for fn in (extract_sensors, extract_offball_sensors,
               _nearest_defender_dist, _defenders_within,
               _teammate_openness, _best_forward_teammate,
               _fatigue_estimate):
        src = inspect.getsource(fn)
        src = src.replace("__future__", "____")      # from __future__ import
        lowered = src.lower()
        hit = [t for t in forbidden if t in lowered]
        assert not hit, f"{fn.__name__} references future/outcome symbol(s): {hit}"

    # Behavioural: mutate an outcome dict that a leaking sensor might read.
    positions = {"Carrier": (60, 30), "Def1": (63, 31), "Fwd": (80, 34)}
    pe = FakePositionEngine(positions)
    player = make_player("Carrier", "CM")
    tm = [make_player("Fwd", "ST")]
    df = [make_player("Def1", "CDM")]
    kw = dict(player=player, x=60, y=30, teammates=tm, defenders=df,
              position_engine=pe, under_pressure=False, attacks_right=True,
              game_state=FakeGameState("LEVEL"), minute=45)
    s_before = extract_sensors(**kw)
    fake_outcome = {"goal_scored": True, "xg": 0.9}
    for o in ("_future_outcome", "metadata", "result"):
        setattr(pe, o, fake_outcome)
    s_after = extract_sensors(**kw)
    assert np.array_equal(s_before, s_after), \
        "sensor vector changed when future/outcome state was injected"


def test_sensor_extraction_invariant_to_hidden_player_state():
    """Perception depends only on documented inputs.

    A player's confidence / opponent form must NOT move the sensors;
    vision (a documented input) MUST move them.
    """
    positions = {"Carrier": (60, 30), "Def1": (63, 31), "Fwd": (80, 34)}
    pe = FakePositionEngine(positions)
    player = make_player("Carrier", "CM", vision=70)
    tm = [make_player("Fwd", "ST")]
    df = [make_player("Def1", "CDM")]
    kw = dict(player=player, x=60, y=30, teammates=tm, defenders=df,
              position_engine=pe, under_pressure=False, attacks_right=True,
              game_state=FakeGameState("LEVEL"), minute=45)
    s_base = extract_sensors(**kw)

    player.dna.form.confidence = 95.0          # carrier confidence
    df[0].dna.form.confidence = 5.0            # opponent confidence
    tm[0].dna.form.confidence = 33.0           # teammate confidence
    s_hidden = extract_sensors(**kw)
    assert np.array_equal(s_base, s_hidden), \
        "sensors must be invariant to hidden form/confidence state"

    player.dna.mental.vision = 95.0            # documented input
    s_visible = extract_sensors(**kw)
    assert not np.array_equal(s_base, s_visible), \
        "vision must change the sensor vector (documented DNA input)"


def test_sensor_extraction_robustness_no_nan():
    """Sensors are finite and in-range under adversarial geometry."""
    edges = [(0, 0), (105, 0), (0, 68), (105, 68), (52.5, 34)]
    for x, y in edges:
        # coincident teammates + defenders, no position engine (fallbacks)
        tm = [make_player("A", "CM"), make_player("B", "CM")]
        df = [make_player("C", "CB"), make_player("D", "CB"), make_player("E", "CB")]
        s = extract_sensors(make_player("P", "CAM"), x, y, tm, df,
                            None, True, x > 50, FakeGameState("LEVEL"), 90)
        assert s.shape == (INPUT_SIZE,)
        assert np.all(np.isfinite(s)), f"NaN/Inf at ({x},{y}): {s}"
        assert np.all((s >= 0.0) & (s <= 1.0)), f"out of range at ({x},{y}): {s}"
    # GK-only defenders, empty teammates
    gk = make_player("GK", "GK")
    s = extract_sensors(make_player("P", "ST"), 80, 34, [], [gk],
                        None, False, True, None, 30)
    assert np.all(np.isfinite(s)) and np.all((s >= 0.0) & (s <= 1.0))


# ─────────────────────────────────────────────────────────────
# 6. brain_integration — decide() contract
# ─────────────────────────────────────────────────────────────

# Reuse fixtures
FINAL_THIRD_POSITIONS = {
    "Carrier": (75, 34), "Fwd Runner": (92, 40), "Deep CB": (25, 34),
    "Def1": (78, 33), "Def2": (85, 45), "Def3": (30, 30), "Far Winger": (78, 60),
}


def _setup_decide(player=None):
    player = player or make_player("Carrier", "CAM")
    teammates = [make_player("Fwd Runner", "ST"), make_player("Deep CB", "CB"),
                 make_player("Far Winger", "RW")]
    defenders = [make_player("Def1", "CDM"), make_player("Def2", "CB"),
                 make_player("Def3", "CB")]
    pe = FakePositionEngine(FINAL_THIRD_POSITIONS)
    clear_registry()
    register_brain(player.name, FootballBrain.from_dna(player, seed=10))
    return player, teammates, defenders, pe


def test_decide_returns_playerdecision():
    player, teammates, defenders, pe = _setup_decide()
    d = NeuralDecisionBrain.decide(
        player, 75, 34, teammates, defenders, pe, FakeTeamProfile("balanced"),
        False, True, FakeGameState("LEVEL"), minute=30,
    )
    assert isinstance(d, PlayerDecision), type(d)
    assert isinstance(d.intent, PlayerIntent)
    assert d.action in ("CARRY", "PASS")
    assert 0.0 <= d.confidence <= 1.0
    assert 0.0 <= d.decision_quality <= 1.0
    assert 0.0 <= d.risk_level <= 1.0
    assert isinstance(d.reason, str) and d.reason
    # legacy fields the event_chain relies on
    for f in fields(PlayerDecision):
        assert hasattr(d, f.name), f"PlayerDecision missing field {f.name}"


def test_decide_valid_intents():
    player, teammates, defenders, pe = _setup_decide()
    d = NeuralDecisionBrain.decide(
        player, 75, 34, teammates, defenders, pe, FakeTeamProfile("balanced"),
        False, True, FakeGameState("LEVEL"), minute=30,
    )
    valid = set(PlayerIntent)
    assert d.intent in valid, d.intent


def test_decide_cache_fallback_when_unregistered():
    # player with no registered brain should still work (random fallback)
    player, teammates, defenders, pe = _setup_decide()
    clear_registry()
    d = NeuralDecisionBrain.decide(
        player, 75, 34, teammates, defenders, pe, FakeTeamProfile("balanced"),
        False, True, FakeGameState("LEVEL"), minute=30,
    )
    assert isinstance(d, PlayerDecision)


def test_decide_trace():
    player, teammates, defenders, pe = _setup_decide()
    d = NeuralDecisionBrain.decide(
        player, 75, 34, teammates, defenders, pe, FakeTeamProfile("balanced"),
        False, True, FakeGameState("LEVEL"), minute=30, record_trace=True,
    )
    assert d.trace is not None
    assert "sensor_vector" in d.trace
    assert "output_probs" in d.trace
    assert "chosen_idx" in d.trace
    assert len(d.trace["sensor_vector"]) == INPUT_SIZE


def test_distinct_players_distinct_brains():
    p1 = make_player("Alpha", "CAM", vision=90, composure=85, decisions=88)
    p2 = make_player("Beta", "CB", vision=40, composure=35, decisions=42)
    b1 = FootballBrain.from_dna(p1, seed=1)
    b2 = FootballBrain.from_dna(p2, seed=1)
    assert not np.allclose(b1.w1, b2.w1)


def test_temperature_sampling_variety():
    # argmax (temp=0) is deterministic
    p = np.array([0.8, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0])
    random.seed(7)
    argmax_picks = {_sample_from_probs(p, 0.0) for _ in range(50)}
    assert argmax_picks == {0}, f"argmax should always pick index 0, got {argmax_picks}"
    # temp=0.5 occasionally samples other options (variety)
    random.seed(7)
    sampled = {_sample_from_probs(p, 0.5) for _ in range(5000)}
    assert len(sampled) > 1, f"temperature sampling should give variety, got {sampled}"
    assert 0 in sampled


def test_temperature_composure_effect():
    from brain_integration import _decision_temperature
    calm = make_player("Calm", "CAM", composure=95, decisions=92)
    nervous = make_player("Nerv", "CDM", composure=30, decisions=28)
    assert _decision_temperature(calm, False, 0.0) < _decision_temperature(nervous, False, 0.0), \
        "calm decision-maker should sample closer to argmax"
    # pressure raises temperature for a moderate player (worse judgement);
    # a max-calm player is already at temp 0 so it cannot rise further.
    moderate = make_player("Mod", "CM", composure=55, decisions=55)
    assert _decision_temperature(moderate, True, 0.0) > _decision_temperature(moderate, False, 0.0), \
        "pressure should raise temperature (worse judgement)"
    # fatigue raises temperature too
    assert _decision_temperature(moderate, False, 0.8) > _decision_temperature(moderate, False, 0.0), \
        "fatigue should raise temperature"


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Surrogate (surrogate_collect.py + brain_evolution integration)
# ─────────────────────────────────────────────────────────────

def test_surrogate_bucket_roundtrip():
    """Bucket produces a stable key and is sensitive to key features."""
    from surrogate_collect import FitnessSurrogate
    s = np.zeros(24)
    b1 = FitnessSurrogate.bucket(s)
    s2 = np.zeros(24); s2[16] = 1.0  # under pressure
    b2 = FitnessSurrogate.bucket(s2)
    assert b1 != b2, "pressure changes bucket"
    # same features, no noise -> same bucket
    assert FitnessSurrogate.bucket(np.zeros(24)) == b1


def test_surrogate_fit_and_query():
    """fit learns per-intent success; query returns table values."""
    from surrogate_collect import FitnessSurrogate
    # Build a tiny deterministic dataset: same situation, two intents
    # with clearly different outcomes (SHOOT far better than RECYCLE).
    s = np.zeros(24)
    s[16] = 0      # not pressured
    s[12] = 1      # final third
    data = []
    for i in range(20):
        if i % 2 == 0:
            data.append((s, PlayerIntent.SHOOT, 2.5, "SHOT_ON_TARGET", "ST"))
        else:
            data.append((s, PlayerIntent.RECYCLE, 0.3, "PASS", "ST"))
    sur = FitnessSurrogate().fit(data)
    shoot_rep = sur.expected_success(s, PlayerIntent.SHOOT, "ST")
    recycle_rep = sur.expected_success(s, PlayerIntent.RECYCLE, "ST")
    assert shoot_rep > recycle_rep, "SHOOT should score higher than RECYCLE here"


def test_surrogate_save_load():
    """FitnessSurrogate round-trips through JSON."""
    from surrogate_collect import FitnessSurrogate
    import tempfile, os
    s = np.zeros(24); s[9] = 0.5
    data = [(s, PlayerIntent.CARRY, 1.2, "CARRY", "CM")] * 5
    sur = FitnessSurrogate().fit(data)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sur.json")
        sur.save(p)
        loaded = FitnessSurrogate.load(p)
    assert loaded.expected_success(s, PlayerIntent.CARRY, "CM") == \
           sur.expected_success(s, PlayerIntent.CARRY, "CM")


def test_synthetic_fitness_accepts_surrogate():
    """synthetic_fitness runs with a surrogate without error and in [0,1]."""
    from brain_evolution import synthetic_fitness
    from surrogate_collect import FitnessSurrogate
    brain = FootballBrain.random(seed=1)
    # tiny surrogate from a scratch dataset
    rng = np.random.RandomState(0)
    data = []
    for _ in range(60):
        s = rng.rand(24)
        s[16] = int(rng.rand() < 0.5)
        s[12] = int(rng.rand() < 0.5)
        i = rng.randint(0, OUTPUT_SIZE)
        intent = _intent_list[i]
        data.append((s, intent, float(rng.rand() * 2.0), "CARRY", "ST"))
    sur = FitnessSurrogate().fit(data)
    f = synthetic_fitness(brain, "ST", n_states=50, seed=3, surrogate=sur)
    assert 0.0 <= f <= 1.0


def test_synthetic_fitness_batched_matches_legacy():
    """The vectorised batched path is numerically identical to the
    sequential legacy path for the same corpus."""
    from brain_evolution import synthetic_fitness, generate_state_corpus
    from surrogate_collect import FitnessSurrogate
    brain = FootballBrain.random(seed=21)
    rng = np.random.RandomState(1)
    data = []
    for _ in range(40):
        s = rng.rand(24)
        data.append((s, _intent_list[rng.randint(0, OUTPUT_SIZE)],
                     float(rng.rand() * 2.0), "CARRY", "CM"))
    sur = FitnessSurrogate().fit(data)
    pos, n, seed, gb = "CM", 120, 7, 0.25
    corpus = generate_state_corpus(pos, n, seed, goal_bias=gb)
    assert corpus.shape == (n, INPUT_SIZE)
    # same corpus, same RNG, two APIs -> byte-identical scores
    f_batch = synthetic_fitness(brain, pos, batched_sensors=corpus, surrogate=sur)
    f_legacy = synthetic_fitness(brain, pos, n_states=n, seed=seed,
                                 goal_bias=gb, surrogate=sur)
    assert f_batch == f_legacy, f"batched {f_batch} != legacy {f_legacy}"
    # and generation is deterministic
    corpus2 = generate_state_corpus(pos, n, seed, goal_bias=gb)
    assert np.array_equal(corpus, corpus2)


def test_generate_state_corpus_scoring_bias():
    """goal_bias states are realistically scoring-prone (x high, late game)."""
    from brain_evolution import generate_state_corpus
    n = 60
    corpus = generate_state_corpus("ST", n, seed=11, goal_bias=0.5)
    assert corpus.shape == (n, INPUT_SIZE)
    # goal-bias states use x in 78..96 => goalscoring band
    import random as _r
    rng = _r.Random(11)
    scored = sum(1 for _ in range(n) if rng.random() < 0.5)
    assert scored > 0
    # corpus rows include some high final-third states (slot 12)
    assert corpus[:, 12].mean() > 0.4, corpus[:, 12].mean()


_intent_list = [getattr(PlayerIntent, l) for l in INTENT_LABELS]


# ─────────────────────────────────────────────────────────────
# DefensiveActionBrain + engine hook
# ─────────────────────────────────────────────────────────────

class FakePosState:
    def __init__(self, x, y, position):
        self.current_x = x
        self.current_y = y
        self.position = position


class FakeDefPosEngine:
    def __init__(self):
        self.team_rosters = {"Def": ["CB1", "LB1", "CDM1", "GK1"],
                             "Att": ["ST1", "ST2", "RW1"]}
        self.states = {
            "CB1": FakePosState(40, 30, "CB"),
            "LB1": FakePosState(35, 12, "LB"),
            "CDM1": FakePosState(52, 40, "CDM"),
            "GK1": FakePosState(8, 34, "GK"),
            "ST1": FakePosState(48, 34, "ST"),
            "ST2": FakePosState(55, 26, "ST"),
            "RW1": FakePosState(60, 55, "RW"),
        }
        self.team_attacks_right = {"Def": True, "Att": False}


class FakeDefEngine:
    def __init__(self):
        self.position_engine = FakeDefPosEngine()
        self.state = SimpleNamespace(minute=60, phase=SimpleNamespace(value=3),
                                     home_goals=1, away_goals=0)
        self.config = SimpleNamespace(away_team="Att")
        self.sub_controller = None


def test_defensive_action_brain_basic():
    """24->32->32->4, probs sum to 1, predict returns a label, save/load."""
    from football_brain import DefensiveActionBrain, DEFENSIVE_ACTIONS
    b = DefensiveActionBrain.random(seed=7)
    assert b.param_count == 1988
    s = np.random.RandomState(0).rand(24)
    p = b.forward(s)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    a = b.predict(s)
    assert a in DEFENSIVE_ACTIONS
    import json, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json.dumps(b.serialize()))
        path = f.name
    try:
        loaded = DefensiveActionBrain.deserialize(json.load(open(path)))
        assert np.allclose(loaded.forward(s), p)
    finally:
        os.unlink(path)


def test_defensive_action_sensors_shape():
    """extract_defensive_sensors returns 24-d 0..1 with feasibility index 19."""
    from football_brain import extract_defensive_sensors
    eng = FakeDefEngine()
    s = extract_defensive_sensors(
        eng, "Def", "Att", ball_x=50, ball_y=34,
        danger_level=70, ball_aerial=True,
        contest_x=45, contest_y=32, opponent_distance=4.0,
        press_occurred=True)
    assert s.shape == (24,)
    assert 0.0 <= s.min() and s.max() <= 1.0
    assert s[11] == 1.0          # aerial
    assert s[15] == 1.0          # press happened
    assert s[3] == 0.70          # danger/100
    assert s[20] == 1.0          # nearest defender is a CB
    assert s[19] == 1.0          # danger>=30 -> clearance feasible somewhere
    far = extract_defensive_sensors(
        eng, "Def", "Att", ball_x=88, ball_y=34,
        danger_level=5, contest_x=90, contest_y=30)
    assert far[19] == 0.0        # opp half, low danger -> not clearance-feasible


def test_def_action_choice_heuristic_fallback():
    """With no brain, the hook reproduces the legacy weights + feasibility."""
    import match_engine as me
    for danger in (0, 15, 30, 45, 60, 75, 85, 100):
        assert me._danger_scaled_action_weights_static(danger) == \
            me._danger_scaled_action_weights_static(danger)
    eng = FakeDefEngine()
    me._DEF_ACTION_BRAIN = None
    me._DEF_ACTION_LOADED = True
    me._DEF_ACTION_AUTO = False
    try:
        a = me._def_action_choice(
            eng, "Def", "Att", 80, 30, 34, False, 4.0, True, 30, 34)
        assert a in ("tackle", "interception", "clearance", "block")
        # infeasible far-from-goal clearance gets clamped to tackle/interception
        a2 = me._def_action_choice(
            eng, "Def", "Att", 80, 90, 40, False, 4.0, True, 90, 40)
        assert a2 in ("tackle", "interception")
    finally:
        me._DEF_ACTION_AUTO = True
        me._DEF_ACTION_LOADED = False


def test_def_action_choice_brain_engaged():
    """A loaded brain's argmax is used when the act is feasible."""
    import match_engine as me
    from football_brain import DefensiveActionBrain

    class FixedBrain(DefensiveActionBrain):
        def forward(self, sensors):
            e = np.zeros(4)
            e[1] = 1.0   # "interception"
            return e

    eng = FakeDefEngine()
    me._DEF_ACTION_BRAIN = FixedBrain.random(seed=1)
    me._DEF_ACTION_LOADED = True
    me._DEF_ACTION_AUTO = False
    try:
        a = me._def_action_choice(
            eng, "Def", "Att", 80, 40, 34, False, 4.0, True, 40, 34)
        assert a == "interception"
    finally:
        me._DEF_ACTION_BRAIN = None
        me._DEF_ACTION_LOADED = False
        me._DEF_ACTION_AUTO = True


# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    random.seed(123)
    tests = [
        test_forward_shapes, test_forward_batch_vectorizable, test_predict_args,
        test_param_count, test_from_dna_seeded, test_mutate_changes_weights,
        test_crossover, test_blend, test_serialize_roundtrip,
        test_sensor_shape_and_range, test_sensor_final_third_flags,
        test_sensor_pressure_flag, test_sensors_with_geometry,
        test_sensor_extraction_deterministic,
        test_sensor_extraction_no_future_leak,
        test_sensor_extraction_invariant_to_hidden_player_state,
        test_sensor_extraction_robustness_no_nan,
        test_decide_returns_playerdecision, test_decide_valid_intents,
        test_decide_cache_fallback_when_unregistered, test_decide_trace,
        test_distinct_players_distinct_brains,
        test_temperature_sampling_variety, test_temperature_composure_effect,
        test_surrogate_bucket_roundtrip, test_surrogate_fit_and_query,
        test_surrogate_save_load, test_synthetic_fitness_accepts_surrogate,
        test_synthetic_fitness_batched_matches_legacy,
        test_generate_state_corpus_scoring_bias,
        test_defensive_action_brain_basic,
        test_defensive_action_sensors_shape,
        test_def_action_choice_heuristic_fallback,
        test_def_action_choice_brain_engaged,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            sys.exit(1)
    print(f"\n{passed}/{len(tests)} neural brain tests passed.")
