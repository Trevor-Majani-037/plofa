"""The pre-decision micro-carry rate (2026-11-23).

A micro-carry is the 2-6 m touch emitted at `event_chain.py:1683` BEFORE the
on-ball decision. The attribution probe measured 55 of 57 CB carries in a
match as `micro_carry=True`, while the brain chose CARRY 0% of the time —
so this is a movement-layer rate, not a verb-layer change. The brain still
owns every action that follows.
"""
import event_chain as ec


def test_non_cb_rate_is_the_historical_flat_value():
    for position in ("CM", "ST", "LB", "RW", "GK", ""):
        assert ec.micro_carry_rate(position, 25.0, True) == ec.MICRO_CARRY_RATE
        assert ec.micro_carry_rate(position, 95.0, False) == ec.MICRO_CARRY_RATE


def test_cb_is_tapered_in_the_own_and_middle_third():
    assert ec.micro_carry_rate("CB", 10.0, True) == ec.CB_MICRO_CARRY_RATE_OWN
    assert ec.micro_carry_rate("CB", 55.0, True) == ec.CB_MICRO_CARRY_RATE_OWN
    assert ec.micro_carry_rate("CB", 69.9, True) == ec.CB_MICRO_CARRY_RATE_OWN


def test_cb_rate_steps_up_in_the_final_third():
    assert ec.micro_carry_rate("CB", 70.0, True) == ec.CB_MICRO_CARRY_RATE_ADV
    assert ec.micro_carry_rate("CB", 95.0, True) == ec.CB_MICRO_CARRY_RATE_ADV


def test_cb_rate_is_mirror_exact():
    for xa in (10.0, 40.0, 69.9, 70.0, 95.0):
        right = ec.micro_carry_rate("CB", xa, True)
        left = ec.micro_carry_rate("CB", 105.0 - xa, False)
        assert right == left, f"asymmetric at normalised depth {xa}"


def test_default_rates_are_the_measured_fix():
    assert ec.MICRO_CARRY_RATE == 0.55
    assert 0.0 < ec.CB_MICRO_CARRY_RATE_OWN < ec.CB_MICRO_CARRY_RATE_ADV
    assert ec.CB_MICRO_CARRY_RATE_ADV < ec.MICRO_CARRY_RATE


def test_legacy_env_restores_the_flat_rate(monkeypatch):
    monkeypatch.setenv("PLOFA_LEGACY_CB_MICRO", "1")
    assert ec.micro_carry_rate("CB", 10.0, True) == ec.MICRO_CARRY_RATE
    assert ec.micro_carry_rate("CB", 95.0, True) == ec.MICRO_CARRY_RATE
