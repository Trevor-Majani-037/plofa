"""Flank selection: foot is a preference, and exact ties are not row order.

Background
----------
`RosterLoader._pick_from_pool` used to sort flank candidates on
``(foot_rank, -standing)`` where ``foot_rank`` was a HARD gate: 0 both-footed,
1 preferred foot, 2 opposite foot.  Two consequences, both observed in the
2026-27 season:

1. An inverted winger (left foot on the right) could never outrank ANY
   same-foot winger, at any market value.  Perćy Luka -- RW, left foot, 120M,
   club captain -- sat behind Rayan Victor Jam¢s (RW, left foot, 120M) purely
   because both landed on ``(2, -120.0)`` and Python's stable sort kept the
   Excel row order.

2. Fullbacks and wingers were treated identically, even though a right-footed
   left-back is a liability while a left-footed right-winger is a modern
   standard.

These tests pin the corrected contract: a value tax for wings, a hard gate
only for fullbacks, and an authority tie-break that is a football decision
rather than a spreadsheet-order accident.
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from roster_loader import (
    INVERTED_WING_PENALTY,
    PlayerRecord,
    RosterLoader,
    _authority_tiebreak,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(name, pos, foot, value, *, club="Test", other=None, captain=False,
       age=25):
    dob = date(date.today().year - age, 1, 1)
    return PlayerRecord(
        club=club,
        leadership="Captain" if captain else None,
        role="First-Team",
        number=10,
        name=name,
        pos=pos,
        other_pos=other,
        dob=dob,
        market_value=value,
        preferred_foot=foot,
        specialties_raw="",
    )


def _pick(slot, players, load_map=None):
    """Run the real selector with a pool containing just these players."""
    loader = RosterLoader.__new__(RosterLoader)   # no workbook needed
    pool = {slot: list(players)}
    return loader._pick_from_pool(pool, slot, set(), load_map)


# ── wings: foot is a preference, not a gate ────────────────────────────────

def test_better_inverted_winger_beats_weaker_conventional_one():
    """The old rule benched a 120M left-footed RW for a 22M right-footed one."""
    inverted = _p("Inv", "RW", "Left", 120.0)
    conventional = _p("Conv", "RW", "Right", 22.0)
    assert _pick("RW", [inverted, conventional]) is inverted


def test_conventional_winger_wins_an_exact_value_tie():
    """The tax has to bite, otherwise the preference is meaningless."""
    conventional = _p("Conv", "RW", "Right", 100.0)
    inverted = _p("Inv", "RW", "Left", 100.0)
    assert _pick("RW", [inverted, conventional]) is conventional


def test_the_tax_is_a_fraction_not_a_disqualification():
    """An inverted winger wins if he is worth more than 1/(1-tax)."""
    inverted = _p("Inv", "RW", "Left", 100.0)
    conventional = _p("Conv", "RW", "Right", 100.0 * (1.0 - INVERTED_WING_PENALTY) - 1.0)
    assert _pick("RW", [conventional, inverted]) is inverted


def test_both_footed_counts_as_natural():
    both = _p("Both", "RW", "Both", 100.0)
    inverted = _p("Inv", "RW", "Left", 100.0)
    assert _pick("RW", [inverted, both]) is both


def test_left_footed_lw_is_not_penalised():
    natural = _p("Nat", "LW", "Left", 90.0)
    inverted = _p("Inv", "LW", "Right", 100.0)
    # 100 * 0.85 = 85 < 90, so the natural left-footer keeps the shirt
    assert _pick("LW", [inverted, natural]) is natural


# ── fullbacks: the hard gate survives ─────────────────────────────────────

def test_fullbacks_still_prefer_their_own_foot_over_value():
    """A right-footed LB is shown inside every time; value does not fix that."""
    natural = _p("Nat", "LB", "Left", 10.0)
    rich = _p("Rich", "LB", "Right", 200.0)
    assert _pick("LB", [rich, natural]) is natural


def test_right_back_still_prefers_a_right_foot():
    natural = _p("Nat", "RB", "Right", 10.0)
    rich = _p("Rich", "RB", "Left", 200.0)
    assert _pick("RB", [rich, natural]) is natural


# ── exact ties are broken by authority, not Excel row order ───────────────

def test_captain_wins_an_exact_tie_even_listed_second():
    captain = _p("Captain", "RW", "Left", 120.0, captain=True, age=34)
    other = _p("Other", "RW", "Left", 120.0, age=22)
    assert _pick("RW", [other, captain]) is captain
    assert _pick("RW", [captain, other]) is captain   # order-independent


def test_seniority_breaks_a_tie_between_two_non_captains():
    older = _p("Older", "RW", "Left", 50.0, age=33)
    younger = _p("Younger", "RW", "Left", 50.0, age=21)
    assert _pick("RW", [younger, older]) is older
    assert _pick("RW", [older, younger]) is older


def test_tiebreak_is_total_and_deterministic():
    a = _p("Aa", "RW", "Left", 50.0, age=25)
    b = _p("Bb", "RW", "Left", 50.0, age=25)
    assert _authority_tiebreak(a) != _authority_tiebreak(b)
    assert _pick("RW", [a, b]) is a
    assert _pick("RW", [b, a]) is a


# ── the real squad that exposed it ────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, "PLOFA-2026-2027.xlsx")),
    reason="roster workbook not present",
)
def test_percy_luka_starts_for_pearls():
    """Regression: the 120M, left-footed, club-captain RW was benched.

    He and Rayan Victor Jam¢s are identical on every selection input
    (RW / left foot / 120.0M), so the choice fell to Excel row order.
    """
    loader = RosterLoader()
    squad = loader.build_matchday_squad("Pearls", availability=None)
    starters = [row[0] for row in squad["starters"]]
    assert "Perćy Luka" in starters
