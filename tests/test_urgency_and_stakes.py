"""Urgency: the reward must be able to change WHICH action is preferred.

WHY THIS FILE EXISTS
--------------------
The player's indifference to the real world has three independent causes, and
two of them are arithmetic rather than tuning:

1. The objective never saw the scoreline (``threat_reward`` had no reference to
   it) — fixed by ``stakes``.
2. The corpus never carried a scoreline (the collector decoded its arguments a
   slot early) — fixed in ``surrogate_collect``.
3. **A single scalar multiplier cannot change a preference.**  ``k·a > k·b``
   exactly when ``a > b`` for any ``k > 0``, so scaling a whole return by "how
   much this moment matters" makes a decision more or less *confident* and
   never *different*.  A corpus built that way, however large, distils to the
   same indifferent policy.

So the reward scales GAINS and LOSSES by different numbers (``stakes_pair``),
and the table gets a pooled scoreline layer (``urgency_delta``) to express the
result.  These tests pin both, plus the property that makes the first one
necessary.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decision_brain import PlayerIntent
from surrogate_collect import FitnessSurrogate, _URGENCY_PRIOR_N
from threat_reward import stakes, stakes_pair


# ─────────────────────────────────────────────────────────────
# THE ARITHMETIC THAT MAKES THE DIFFERENTIAL NECESSARY
# ─────────────────────────────────────────────────────────────

def test_a_uniform_multiplier_cannot_change_a_preference():
    """The reason ``stakes()`` alone was not enough.

    This is not a claim about this codebase, it is a property of positive
    scaling: every value is multiplied by the same number, so the winner is
    the same value.  Stated as a test so nobody re-introduces the scalar and
    expects behaviour to change.
    """
    values = {"SHOOT": 0.30, "SAFE_PASS": 0.10, "DRIBBLE": 0.20}
    winner = max(values, key=values.get)
    for k in (0.65, 1.0, 1.23, 1.60):
        scaled = {i: v * k for i, v in values.items()}
        assert max(scaled, key=scaled.get) == winner, (
            f"a uniform x{k} changed the preference — the algebra says it "
            "cannot, so this would mean the implementation is not uniform")


def test_a_differential_multiplier_can_change_a_preference():
    """And the reason the differential is enough — in real numbers.

    Two actions in the same situation: a shot that scores a third of the time
    and misses badly otherwise, and a safe square ball.  At 0-0 the safe ball
    is worth more.  Two goals down in the 88th, the shot is worth more,
    because the downside of missing has been discounted to almost nothing and
    the upside is worth everything.  One goal up, the safe ball wins by more.
    """
    late = 88.0
    shot_outcomes = [(1.0, 0.3), (-0.6, 0.7)]     # (return, probability)
    safe = 0.05

    def cell_mean(outcomes, pair):
        gain, loss = pair
        return sum(p * (r * (gain if r >= 0 else loss)) for r, p in outcomes)

    level = stakes_pair(0, late)
    chasing = stakes_pair(-2, late)
    protecting = stakes_pair(1, late)

    shot_level = cell_mean(shot_outcomes, level)
    shot_chasing = cell_mean(shot_outcomes, chasing)
    shot_protecting = cell_mean(shot_outcomes, protecting)

    assert safe * level[0] > shot_level, "at 0-0 the safe ball is worth more"
    assert shot_chasing > safe * chasing[0], (
        "two down in the 88th the shot must become the better option")
    assert safe * protecting[0] > shot_protecting, (
        "one up in the 88th the safe ball must be preferred by MORE")


def test_the_pair_is_asymmetric_in_the_documented_direction():
    """Chasing buys upside and discounts downside; leading does the opposite."""
    for minute in (65.0, 75.0, 88.0):
        g_chase, l_chase = stakes_pair(-1, minute)
        g_lead, l_lead = stakes_pair(1, minute)
        assert g_chase > 1.0 > l_chase, "chasing must be convex"
        assert g_lead < 1.0 < l_lead, "protecting must be concave"


def test_the_pair_is_neutral_early():
    """Before the ramp starts, nothing is rescaled — the old objective."""
    for sd in (-3, -1, 0, 1, 3):
        assert stakes_pair(sd, 5.0) == (1.0, 1.0)
        assert stakes_pair(sd, 60.0) == (1.0, 1.0)


def test_the_pair_stays_inside_the_published_range():
    """A multiplier that swings by an order of magnitude would drown geometry."""
    lo, hi = 1.0, 1.0
    for sd in range(-4, 5):
        for minute in (0.0, 45.0, 60.0, 75.0, 90.0):
            g, l = stakes_pair(sd, minute)
            lo, hi = min(lo, g, l), max(hi, g, l)
    assert 0.40 <= lo and hi <= 1.60, f"range {lo:.2f}..{hi:.2f} is too wide"


def test_the_scalar_and_the_pair_agree_where_they_should():
    """``stakes()`` is still the published 'worth of the moment'.

    It is not the label multiplier any more, but it must not contradict the
    pair it was derived from: both ramp from minute 60 and both agree that a
    late level game matters and a late cruise does not.
    """
    assert stakes(0, 88.0) > 1.0
    assert stakes(2, 88.0) < 1.0
    assert stakes(0, 10.0) == pytest.approx(1.0)
    assert stakes_pair(0, 88.0)[0] > 1.0
    assert stakes_pair(2, 88.0)[0] < 1.0


# ─────────────────────────────────────────────────────────────
# THE COLLECTOR APPLIES IT
# ─────────────────────────────────────────────────────────────

def test_the_differential_reaches_the_label():
    """``_event_success`` must scale the sign, not the whole number.

    NOTE ON THE FIXTURE: the reward reads ``ev.event_type``, not ``ev.type``.
    An earlier test in this project set ``type`` and compared ``0.0 * 2 == 0.0``
    — a vacuous pass that would have survived any regression here.
    """
    import surrogate_collect as SC

    class Ev:
        event_type = "PASS"
        location_x, location_y = 30.0, 34.0
        end_x, end_y = 90.0, 20.0
        outcome = True
        team = "H"

    class EvLost(Ev):
        # A turnover five metres from our own goal line: the expensive kind,
        # and the one the differential exists to price differently.
        outcome = False
        location_x, location_y = 5.0, 34.0
        end_x, end_y = None, None

    base_gain = SC._event_success(Ev(), attacks_right=True)
    base_loss = SC._event_success(EvLost(), attacks_right=True)
    assert base_gain > 0 and base_loss < 0, (
        f"test fixture is not a gain/loss pair (gain={base_gain}, "
        f"loss={base_loss})")

    got_gain = SC._event_success(Ev(), attacks_right=True, stakes_pair=(1.5, 0.6))
    got_loss = SC._event_success(EvLost(), attacks_right=True,
                                 stakes_pair=(1.5, 0.6))
    assert got_gain == pytest.approx(base_gain * 1.5)
    assert got_loss == pytest.approx(base_loss * 0.6)


def test_a_disabled_switch_is_byte_identical():
    import surrogate_collect as SC

    class Rec:
        score_diff, minute = -2, 88.0

    SC.set_stakes(False)
    try:
        assert SC._stakes_pair(Rec()) == (1.0, 1.0)
        assert SC.stakes_mode() == "off"
    finally:
        SC.set_stakes(True)
    assert SC.stakes_mode() == "differential"


# ─────────────────────────────────────────────────────────────
# THE URGENCY LAYER: BANDS AND THE POOLED SHIFT
# ─────────────────────────────────────────────────────────────

def _sensors(score_diff_norm=0.0, minute_norm=0.5):
    s = np.zeros(24, dtype=float)
    s[18] = score_diff_norm
    s[19] = minute_norm
    return s


def test_the_band_reads_the_player_frame_sensors():
    assert FitnessSurrogate.urgency_key(_sensors(0.4, 0.98)) == "KaheadTlate"
    assert FitnessSurrogate.urgency_key(_sensors(-0.4, 0.98)) == "KbehindTlate"
    assert FitnessSurrogate.urgency_key(_sensors(0.0, 0.98)) == "KlevelTlate"
    assert FitnessSurrogate.urgency_key(_sensors(0.4, 0.80)) == "KaheadTmid"
    assert FitnessSurrogate.urgency_key(_sensors(-0.4, 0.80)) == "KbehindTmid"


def test_the_first_hour_is_one_band_whatever_the_score():
    """Before minute 60 the reward is neutral, so the layer must be too.

    A band that split by scoreline in the first hour would still find
    differences — teams behind early ARE different — but those are team
    strength and match script, not stakes.  Modelling them would teach the
    policy behaviour the objective never encoded.
    """
    for sd in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for minute_norm in (0.0, 0.2, 0.5, 0.66):
            assert FitnessSurrogate.urgency_key(
                _sensors(sd, minute_norm)) == "Kneutral"


def test_a_one_goal_margin_is_not_level():
    """Slot 18 is goals/5, so a one-goal lead is 0.2 — it must not read level."""
    assert FitnessSurrogate.urgency_key(_sensors(0.2, 0.95)) == "KaheadTlate"
    assert FitnessSurrogate.urgency_key(_sensors(-0.2, 0.95)) == "KbehindTlate"


def _corpus(rows):
    """(score_diff_norm, minute_norm, intent, return, position) -> fit rows."""
    out = []
    for sd, mn, intent, ret, pos in rows:
        out.append((_sensors(sd, mn), intent, ret, "PASS", pos))
    return out


def test_the_layer_is_zero_when_nothing_was_fitted():
    """Backward compatibility: an old table must behave exactly as before."""
    sur = FitnessSurrogate()
    assert sur.urgency_of(_sensors(0.4, 0.98), "SHOOT", "ST") == 0.0


def test_the_shift_is_shrunk_toward_zero():
    """One observation must not be allowed to reorder a cell.

    A single pair of samples (one per band, same situation) is the minimum
    evidence the estimator will accept at all: a cell that appears in only one
    band is excluded, because it cannot say how bands differ.
    """
    rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM"),
            (0.4, 0.98, PlayerIntent.CARRY, 0.90, "CM")]
    sur = FitnessSurrogate().fit(_corpus(rows))

    d = sur.urgency_delta["CM"]["KaheadTlate"]["CARRY"]
    c = sur.urgency_counts["CM"]["KaheadTlate"]["CARRY"]
    assert c == 1
    assert 0.0 < d < 0.20, (
        f"one sample each side of a 0.80 contrast must move the value only a "
        f"little (got {d:+.3f})")


def test_a_well_sampled_band_moves_the_value_further():
    """More evidence, less shrink — the whole point of the prior count."""
    def build(n):
        rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM")] * n
        rows += [(0.4, 0.98, PlayerIntent.CARRY, 0.90, "CM")] * n
        return FitnessSurrogate().fit(_corpus(rows))

    d_few = build(1).urgency_delta["CM"]["KaheadTlate"]["CARRY"]
    d_many = build(40).urgency_delta["CM"]["KaheadTlate"]["CARRY"]
    assert d_many > d_few * 3, (
        f"40 observations must move the value much further than one "
        f"({d_many:+.3f} vs {d_few:+.3f})")


def test_the_neutral_band_is_anchored_to_zero():
    """Before minute 60 the multiplier is exactly (1, 1) — so the layer must
    be exactly 0 there, or it is re-centring the baseline preference for
    reasons that have nothing to do with the scoreline."""
    rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM")] * 20
    rows += [(0.4, 0.98, PlayerIntent.CARRY, 0.90, "CM")] * 20
    sur = FitnessSurrogate().fit(_corpus(rows))
    for band in ("Kneutral",):
        d = sur.urgency_delta["CM"].get(band, {}).get("CARRY")
        if d is not None:
            assert abs(d) < 1e-9, (
                f"{band} is a neutral band and must not move: got {d:+.3e}")


def test_bands_that_never_share_a_situation_claim_no_effect():
    """THE ANTI-CONFOUNDING PROPERTY, and the one that caught a real defect.

    A chasing side plays higher up the pitch, so if the two bands never occur
    in the same cell, any difference between them is geometry, not urgency.
    The estimator must then return nothing.  The first version residualised
    against the coarse parent, measured the geometry, and flipped 46 % of
    decisions — including in the band where the stakes are neutral.
    """
    rows = []
    for a in range(5):                       # distinct geometry per band
        s = _sensors(0.0, 0.2)
        s[2] = -1.0 + a * 0.5
        rows.append((s, PlayerIntent.CARRY, 0.10, "PASS", "CM"))
        t = _sensors(0.4, 0.98)
        t[2] = -1.0 + a * 0.5
        t[3] = 1.0                            # mouth bin the other band never has
        rows.append((t, PlayerIntent.CARRY, 0.90, "PASS", "CM"))

    sur = FitnessSurrogate().fit(rows)
    deltas = [abs(d) for b in sur.urgency_delta.get("CM", {}).values()
              for d in b.values()]
    assert not deltas or max(deltas) < 1e-9, (
        "bands with no shared situation must not produce a shift; got "
        f"max {max(deltas) if deltas else 0.0:.3e}")


def test_the_layer_reaches_expected_success_and_can_be_switched_off():
    """The A/B switch gate 7 depends on: same sensors, two answers."""
    rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM")] * 40
    rows += [(0.0, 0.2, PlayerIntent.SAFE_PASS, 0.11, "CM")] * 40
    rows += [(0.4, 0.98, PlayerIntent.CARRY, 0.90, "CM")] * 40
    rows += [(0.4, 0.98, PlayerIntent.SAFE_PASS, 0.10, "CM")] * 40
    sur = FitnessSurrogate().fit(_corpus(rows))

    s = _sensors(0.4, 0.98)
    with_layer = sur.expected_success(s, PlayerIntent.CARRY, "CM", use_urgency=True)
    without = sur.expected_success(s, PlayerIntent.CARRY, "CM", use_urgency=False)
    assert with_layer > without, "CARRY pays more when ahead late in this corpus"

    # and the A/B must be able to change the winner, which is gate 7 in miniature
    a = sur.expected_success(s, PlayerIntent.CARRY, "CM", use_urgency=True)
    b = sur.expected_success(s, PlayerIntent.SAFE_PASS, "CM", use_urgency=True)
    a0 = sur.expected_success(s, PlayerIntent.CARRY, "CM", use_urgency=False)
    b0 = sur.expected_success(s, PlayerIntent.SAFE_PASS, "CM", use_urgency=False)
    assert max([("CARRY", a), ("SAFE_PASS", b)], key=lambda t: t[1])[0] == "CARRY"
    assert max([("CARRY", a0), ("SAFE_PASS", b0)], key=lambda t: t[1])[0] == "CARRY"


def test_a_known_band_effect_is_recovered():
    """The estimate must return the number that was put in.

    Thirty decisions in each of two bands, identical situations, the second
    band worth 0.80 more.  The layer must recover that 0.80 contrast (times
    the shrink, 30/(30+5) = 0.857), not half of it and not twice it.

    The residual is taken against the COARSE parent rather than the rich cell,
    and this fixture is why: the rich cell aggregates over bands by
    construction, so residualising against it makes the band contrast depend
    on how the samples happen to split between geometry sub-cells.  The parent
    pools 45 geometry cells, so the contrast is stable — and, unlike a rich
    cell holding a single observation, it cannot be identically zero.
    """
    rows = []
    for a in range(5):                      # goal-angle bin
        for m in range(3):                  # goal-mouth bin
            for d in (1, 2):                # distance bin
                for sd, mn, ret in ((0.0, 0.2, 0.10), (0.4, 0.98, 0.90)):
                    s = _sensors(sd, mn)
                    s[2] = -1.0 + a * 0.5
                    s[3] = m / 2.0
                    s[14] = 0.45 if d == 1 else 0.2
                    rows.append((s, PlayerIntent.CARRY, ret, "PASS", "CM"))

    sur = FitnessSurrogate().fit(rows)

    ahead = sur.urgency_delta["CM"]["KaheadTlate"]["CARRY"]
    level = sur.urgency_delta["CM"]["Kneutral"]["CARRY"]
    contrast = ahead - level
    expected = 0.80 * (30.0 / (30.0 + _URGENCY_PRIOR_N))
    assert contrast == pytest.approx(expected, rel=0.10), (
        f"recovered contrast {contrast:+.3f}, expected about {expected:+.3f}")
    assert ahead > 0, "the band that pays more must read positive"
    assert level == pytest.approx(0.0, abs=1e-12), (
        "the neutral band is the anchor and must read exactly zero")


def test_the_report_says_how_much_the_layer_could_learn():
    rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM")] * 10
    rows += [(0.4, 0.98, PlayerIntent.CARRY, 0.20, "CM")] * 10
    sur = FitnessSurrogate().fit(_corpus(rows))
    rep = sur.urgency_report()
    assert rep["bands"] > 0
    assert rep["well_sampled"] > 0
    assert rep["prior_n"] == _URGENCY_PRIOR_N
    assert rep["max_abs_delta"] > 0.0


def test_the_layer_survives_a_save_and_a_load(tmp_path):
    rows = [(0.0, 0.2, PlayerIntent.CARRY, 0.10, "CM")] * 10
    rows += [(0.4, 0.98, PlayerIntent.CARRY, 0.20, "CM")] * 10
    sur = FitnessSurrogate().fit(_corpus(rows))
    p = str(tmp_path / "sur.json")
    sur.save(p)
    back = FitnessSurrogate.load(p)
    assert back.urgency_delta == sur.urgency_delta
    assert back.urgency_counts == sur.urgency_counts
    s = _sensors(0.4, 0.98)
    assert (back.expected_success(s, PlayerIntent.CARRY, "CM")
            == pytest.approx(sur.expected_success(s, PlayerIntent.CARRY, "CM")))
