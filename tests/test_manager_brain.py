"""
Manager Brain (Phase 2) — tests.
"""
from types import SimpleNamespace

import numpy as np

from manager_brain import (
    ARCH_VERSION,
    MANAGER_ARCH,
    MANAGER_POSTURE_LABELS,
    SENSOR_VERSION,
    ManagerBrain,
)


def _random_brain(seed=42):
    return ManagerBrain.random(seed)


def _sensors():
    return np.array([0.2, 0.5, 0.66, 0.3, 0.1,
                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])


# ── Forward output shapes ───────────────────────────────────────────

def test_forward_output_shapes():
    brain = _random_brain()
    out = brain.forward(_sensors())
    assert isinstance(out, dict)
    assert set(out.keys()) == {"posture_probs", "pressing", "sub_urgency"}
    assert out["posture_probs"].shape == (3,)
    assert out["pressing"].__class__ is float   # scalar, not array
    assert out["sub_urgency"].__class__ is float


def test_batched_forward_matches_loop():
    brain = _random_brain()
    batch_input = np.array([_sensors(), _sensors() * 0.8,
                            _sensors() * 1.1, np.zeros(12)])
    batch_out = brain.forward(batch_input)
    assert batch_out["posture_probs"].shape == (4, 3)

    for i in range(4):
        single_out = brain.forward(batch_input[i])
        assert np.allclose(batch_out["posture_probs"][i],
                           single_out["posture_probs"], atol=1e-6)
        assert abs(batch_out["pressing"][i] - single_out["pressing"]) < 1e-6
        assert abs(batch_out["sub_urgency"][i] - single_out["sub_urgency"]) < 1e-6


def test_posture_probs_sum_to_one():
    brain = _random_brain()
    for s in [_sensors(), np.zeros(12), _sensors() * 2]:
        p = brain.forward(s)["posture_probs"]
        assert abs(p.sum() - 1.0) < 1e-6, f"sum={p.sum()}"
        assert (p >= 0).all()


def test_pressing_in_unit_interval():
    brain = _random_brain()
    for seed in range(10):
        b = _random_brain(seed)
        v = b.forward(_sensors())["pressing"]
        assert 0.0 <= v <= 1.0, f"pressing={v}"


def test_sub_urgency_in_unit_interval():
    brain = _random_brain()
    for seed in range(10):
        b = _random_brain(seed)
        v = b.forward(_sensors())["sub_urgency"]
        assert 0.0 <= v <= 1.0, f"sub_urgency={v}"


# ── Serialization ───────────────────────────────────────────────────

def test_serialize_roundtrip():
    brain = _random_brain()
    data = brain.serialize()
    brain2 = ManagerBrain.deserialize(data)
    out1 = brain.forward(_sensors())
    out2 = brain2.forward(_sensors())
    assert np.allclose(out1["posture_probs"], out2["posture_probs"])
    assert abs(out1["pressing"] - out2["pressing"]) < 1e-10
    assert abs(out1["sub_urgency"] - out2["sub_urgency"]) < 1e-10


def test_version_mismatch_raises():
    brain = _random_brain()
    data = brain.serialize()
    # Wrong arch_version
    bad1 = {**data, "arch_version": ARCH_VERSION + 1}
    try:
        ManagerBrain.deserialize(bad1)
        assert False, "should have raised"
    except ValueError as e:
        assert "arch_version" in str(e)

    # Wrong sensor_version
    bad2 = {**data, "sensor_version": SENSOR_VERSION + 1}
    try:
        ManagerBrain.deserialize(bad2)
        assert False, "should have raised"
    except ValueError as e:
        assert "sensor_version" in str(e)

    # Wrong kind
    bad3 = {**data, "kind": "football_brain"}
    try:
        ManagerBrain.deserialize(bad3)
        assert False, "should have raised"
    except ValueError as e:
        assert "kind" in str(e)

    # Wrong arch shape
    bad4 = {**data, "arch": [24, 32, 32, 10]}
    try:
        ManagerBrain.deserialize(bad4)
        assert False, "should have raised"
    except ValueError as e:
        assert "arch" in str(e)


def test_random_param_count():
    brain = _random_brain()
    # 12*32 + 32 + 32*32 + 32 + 32*5 + 5 = 1637
    expected = 12 * 32 + 32 + 32 * 32 + 32 + 32 * 5 + 5
    assert brain.param_count == expected
    assert brain.param_count == 1637


def test_from_dna_accepts_player_dna():
    dna = SimpleNamespace(
        mental=SimpleNamespace(vision=80.0, composure=70.0, decisions=65.0),
    )
    brain = ManagerBrain.from_dna(dna, seed=99)
    out = brain.forward(_sensors())
    assert out["posture_probs"].shape == (3,)
    assert 0.0 <= out["pressing"] <= 1.0
    assert 0.0 <= out["sub_urgency"] <= 1.0

    # Also works when passed a player object wrapping dna
    player = SimpleNamespace(dna=dna)
    brain2 = ManagerBrain.from_dna(player, seed=99)
    out2 = brain2.forward(_sensors())
    assert np.allclose(out["posture_probs"], out2["posture_probs"])


# ── Sample forward pass (STOP report requirement) ───────────────────

def _sample_forward_print():
    """Not a real test — called manually for the STOP report."""
    brain = ManagerBrain.random(seed=42)
    sensors = np.array([0.2, 0.5, 0.66, 0.3, 0.1,
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    out = brain.forward(sensors)
    print(f"\n  Sample forward pass (seed=42):")
    print(f"  sensors    = {sensors.tolist()}")
    print(f"  posture    = {dict(zip(MANAGER_POSTURE_LABELS, out['posture_probs']))}")
    print(f"  pressing   = {out['pressing']:.4f}")
    print(f"  sub_urgency= {out['sub_urgency']:.4f}\n")


if __name__ == "__main__":
    _sample_forward_print()