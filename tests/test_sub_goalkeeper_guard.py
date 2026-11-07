"""A goalkeeper must never be chosen to fill an outfield role.

Why this file exists
--------------------
`_ranked_bench(bench, want_pos)` returns `sorted(exact) + sorted(adjacent)`,
where `adjacent` is EVERYONE not in `want_pos`, ranked by `sub_quality`.
A backup keeper often wins that ranking on market value, and `_pick_sub`
returns `ranked[0]` even after the scoreline gate rejects the exact group.

Measured in Hartwell City 2-2 Thornfield (seed 42, 2026-11-07):
    Luca Ferrini (CM) came off at 80' for Joel Trent (GK).
    Joel Trent then covered 0.00 km in 19 minutes, top speed 0.00 m/s,
    x pinned to 11.8-11.9 m for the whole spell, while Keano Walsh (GK)
    played on.

The frozen body was the visible symptom. The cause was a second keeper:
every keeper code path scans the roster and `break`s on the FIRST keeper,
so the man registered behind him is never moved at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from squad_manager import SubstitutionController, ADJACENT_POS


class _Dna:
    def __init__(self, position, overall=70.0):
        self.position = position
        self.overall = overall
        self.minutes_played = 0


class _Player:
    def __init__(self, name, position, overall=70.0):
        self.name = name
        self.position = position
        self.dna = _Dna(position, overall)
        self.sub_in_minute = 0
        self.is_starter = True
        self._used_as_sub = False


def _controller():
    # The ranking helpers only need the controller's own scoring state; the
    # constructor wants both benches, so feed it empty ones.
    return SubstitutionController("Home", "Away", [], [])


def test_backup_keeper_cannot_fill_an_outfield_role():
    """The exact 2026-11-07 case: CM off, only a keeper and a defender spare."""
    c = _controller()
    keeper = _Player("Joel Trent", "GK", overall=95.0)   # best man on paper
    defender = _Player("Sam Boateng", "CB", overall=55.0)
    cm_off = _Player("Luca Ferrini", "CM")

    ranked = c._ranked_bench([keeper, defender], ["CM"] + ADJACENT_POS["CM"])
    assert ranked, "somebody must be selectable"
    assert ranked[0].name == "Sam Boateng", (
        "a keeper outranked an outfielder for a CM slot: "
        f"{[p.name for p in ranked]}")

    chosen = c._pick_sub([keeper, defender], cm_off, "test", minute=80,
                         team="Home", goal_diff=0)
    assert chosen is not None
    assert chosen.position.rstrip("0123456789").upper() != "GK", (
        f"an outfielder's replacement came on as a second keeper: {chosen.name}")


def test_a_keeper_is_still_selectable_for_a_keeper():
    """The gate must not remove the backup keeper from his own role."""
    c = _controller()
    keeper = _Player("Joel Trent", "GK", overall=60.0)
    striker = _Player("Zachery Worth", "ST", overall=99.0)
    gk_off = _Player("Keano Walsh", "GK")

    ranked = c._ranked_bench([keeper, striker], ["GK"])
    assert ranked and ranked[0].name == "Joel Trent", (
        "the backup keeper was excluded from a GK-for-GK change: "
        f"{[p.name for p in ranked]}")


def test_no_positional_filter_still_excludes_keepers():
    """`_pick_fresh_sub` ranks with no positional filter; a keeper is still
    not a legal answer to an outfield change."""
    c = _controller()
    keeper = _Player("Joel Trent", "GK", overall=99.0)
    mid = _Player("Tavish Crane", "CM", overall=50.0)

    ranked = c._ranked_bench([keeper, mid])
    assert [p.name for p in ranked] == ["Tavish Crane"], (
        f"a keeper survived the unfiltered ranking: {[p.name for p in ranked]}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
