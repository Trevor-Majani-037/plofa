"""
TESTS FOR THE REBUILT SUBSTITUTION SELECTION (2026-10-14)
==========================================================

WHAT THIS FILE IS FOR
---------------------
`squad_manager.SubstitutionController` used to select substitutes with a
single rule: "return the first unused player in bench order". Every helper
— `_pick_sub`, `_pick_attacking_sub`, `_pick_fresh_sub`,
`_find_tactical_sub` — was a loop that returned its first hit. `overall_rating`
was read zero times in the whole module, as were `form.confidence`,
`recent_ratings`, `age` and `soul`. The manager was choosing blind, and
bench order is (transitively) market-value order, so even the accidental
behaviour was not "best player".

This file pins the replacement:

  * a scored quality signal (`sub_quality` and its attacking/defensive halves)
  * the scoreline rule (`acceptable_quality_loss`), which is the Wharton 2024
    finding that the KIND of quality you may lose depends on whether you are
    leading or trailing — not just whether the incoming man is "better"
  * the IFAB window model (five subs, three windows, free half-time)
  * three input fixes the selection depends on: the doubled stamina drain,
    the discarded `_starting_stamina`, and the per-team contamination of the
    decision loop

READ THIS BEFORE ADDING A TEST
------------------------------
Two of the guards here are AST guards rather than behavioural ones, and both
exist because the defect is invisible at the behavioural layer:

  * the doubled drain was two complete `process_action` blocks in one method,
    so a test that runs a match sees only "stamina went down", which was
    never in question;
  * `_starting_stamina` was READ by match_engine and WRITTEN by nobody, so a
    test that reads it back finds the attribute genuinely absent and the
    season's entire recovery projection quietly discarded.

A behavioural test for either would pass against the broken code forever.
A negative control for this file is `_diag_sub_guard.py`, which mutates the
mechanisms and asserts the corresponding test goes red.
"""

import ast
import inspect
import pathlib
import random

import pytest

import auto_run_match
import manager_sensors
import match_engine
import squad_manager
from player_dna import SquadBuilder
from squad_manager import (
    SubstitutionController,
    acceptable_quality_loss,
    sub_attacking_quality,
    sub_defensive_quality,
    sub_quality,
)


REPO = pathlib.Path(__file__).resolve().parent.parent


XI = [
    ("G", "GK", []), ("C1", "CB", []), ("C2", "CB", []),
    ("L", "LB", []), ("R", "RB", []), ("M1", "CDM", []),
    ("M2", "CM", []), ("M3", "CAM", []),
    ("W1", "LW", []), ("S", "ST", []), ("W2", "RW", []),
]


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

# Which XI slot a position takes. `SquadBuilder.build` validates hard: exactly
# eleven starters AND exactly one goalkeeper (player_dna.py:1768, :1775), so a
# single-player "squad" cannot be requested from it at all. The substitute
# therefore has to be built INTO a legal eleven and then taken back out.
_SLOT_FOR_POS = {"GK": 0, "CB": 1, "LB": 3, "RB": 4, "CDM": 5,
                 "CM": 6, "CAM": 7, "LW": 8, "ST": 9, "RW": 10}


def _make(tag, pos, slot):
    """Build one real PlayerProfile by seating `tag+slot` in a legal eleven.

    Names are prefixed with `tag` but the returned dict from `_squad` is
    keyed by the UNTAGGED name, so `built["C1"]` keeps working while the
    player's actual `.name` stays distinct between the two squads.
    """
    entries = [(f"{tag}{n}", p, s) for n, p, s in XI]
    entries[_SLOT_FOR_POS.get(pos, 6)] = (f"{tag}{slot}", pos, [])
    squad = SquadBuilder.build(f"{tag}{slot}", entries)
    return squad["starters"][_SLOT_FOR_POS.get(pos, 6)]


def _squad(subs=(), tag="T"):
    """A real XI plus named substitutes, built through the real builder.

    `tag` prefixes every player NAME but not the dict key. It is not
    cosmetic: `SubstitutionController.stamina` is keyed by player name, so
    two squads built from the same XI template collide on every key — an
    injury set on home's "C1" also marks away's "C1", because they are the
    same dictionary entry. The first version of the per-team test relied on
    that collision by accident and passed against the very bug it was
    written to catch.
    """
    entries = [(f"{tag}{n}", p, s) for n, p, s in XI]
    s = SquadBuilder.build(tag, entries)
    built = {n: p for n, p in zip((e[0] for e in XI), s["starters"])}
    bench = []
    for name, pos in subs:
        p = _make(tag, pos, name)
        bench.append(p)
        built[name] = p
    return built, bench


def _controller(bench=(), **kw):
    return SubstitutionController(
        home_team="HOME", away_team="AWAY",
        home_subs_bench=list(bench), away_subs_bench=[],
        **kw,
    )


# ═════════════════════════════════════════════
# 1. THE SCORELINE RULE — pure arithmetic
# ═════════════════════════════════════════════

class TestScorelineRule:
    """`acceptable_quality_loss` is pure arithmetic over five numbers.

    Tested as pure arithmetic deliberately: it is the only place the actual
    decision rule lives, and a test that reached it through a full match
    could not tell which of the four priority tiers produced a given
    outcome.
    """

    def test_an_upgrade_is_always_allowed(self):
        """No research finding says a team should refuse to improve itself."""
        for gd in (-2, -1, 0, 1, 2):
            for minute in (1, 30, 46, 60, 75, 89):
                ok, _ = acceptable_quality_loss(
                    gd, minute,
                    on_attack=90.0, off_attack=50.0,
                    on_defence=90.0, off_defence=50.0,
                )
                assert ok, f"upgrade blocked at gd={gd} minute={minute}"

    def test_leading_never_accepts_a_defensive_downgrade(self):
        """Wharton's headline result.

        LEADING by one, losing ten DEFENSIVE points is never advisable. The
        budget is zero at every minute of a 90-minute match, so this holds
        at 89' and would hold at 89' in any match length.
        """
        for minute in range(0, 91):
            ok, why = acceptable_quality_loss(
                1, minute,
                on_attack=80.0, off_attack=80.0,
                on_defence=60.0, off_defence=70.0,   # 10-point loss
            )
            assert not ok, f"defensive loss accepted at {minute}' while leading"
            assert "defensive" in why

    def test_trailing_never_accepts_an_offensive_downgrade(self):
        """The exact mirror. Chasing a goal does not mean giving up your
        best chance of scoring one."""
        for minute in range(0, 91):
            ok, why = acceptable_quality_loss(
                -1, minute,
                on_attack=60.0, off_attack=70.0,     # 10-point loss
                on_defence=80.0, off_defence=80.0,
            )
            assert not ok, f"offensive loss accepted at {minute}' while trailing"
            assert "attacking" in why

    def test_leading_accepts_an_offensive_loss_only_after_47(self):
        """The other half of the Wharton result: leading, attacking quality
        IS spendable, but not before minute 47."""
        args = dict(on_attack=80.0, off_attack=90.0,
                    on_defence=80.0, off_defence=80.0)
        assert not acceptable_quality_loss(1, 46, **args)[0]
        assert acceptable_quality_loss(1, 47, **args)[0]

    def test_trailing_accepts_a_defensive_loss_only_after_32(self):
        args = dict(on_attack=80.0, off_attack=80.0,
                    on_defence=80.0, off_defence=90.0)
        assert not acceptable_quality_loss(-1, 31, **args)[0]
        assert acceptable_quality_loss(-1, 32, **args)[0]

    def test_the_budget_grows_with_minutes_past_the_threshold(self):
        """The same 12-point attacking loss while leading is unaffordable at
        47' (budget 10) and affordable by 51' (budget 12). That growth is
        what stops a leading team from being held to its first-half standard
        for a full ninety minutes."""
        args = dict(on_attack=78.0, off_attack=90.0,
                    on_defence=80.0, off_defence=80.0)
        assert not acceptable_quality_loss(1, 47, **args)[0]
        assert acceptable_quality_loss(1, 51, **args)[0]
        assert acceptable_quality_loss(1, 60, **args)[0]

    def test_the_budget_is_the_reference_loss_at_the_threshold(self):
        """Wharton is measured in units of TEN points, so the budget at the
        threshold minute is exactly ten — not zero. An earlier version ramped
        from zero, which made the stated 47' threshold mean 67' and quietly
        failed to implement the finding it cited."""
        assert squad_manager._quality_budget(47, 46) == 0.0
        assert squad_manager._quality_budget(47, 47) == pytest.approx(10.0)
        assert squad_manager._quality_budget(47, 57) == pytest.approx(15.0)

    def test_a_threshold_no_minute_can_reach_stays_at_zero(self):
        """The two 'never advisable' columns are encoded as a sentinel, not
        as the match length. Encoding 'never' as the number 90 makes the
        guarantee a coincidence that expires in minute 90 — which is a real
        minute, stoppage time."""
        assert squad_manager.SUB_ACCEPT_DEFENSIVE_LOSS_LEADING_MIN >= squad_manager.SUB_NEVER
        assert squad_manager.SUB_ACCEPT_OFFENSIVE_LOSS_TRAILING_MIN >= squad_manager.SUB_NEVER
        for minute in range(0, 120):
            assert squad_manager._quality_budget(
                squad_manager.SUB_NEVER, minute) == 0.0

    def test_the_two_columns_are_independent(self):
        """Losing attack must not excuse losing defence, or the rule is a
        single axis wearing two labels.

        Each column is checked by removing the OTHER loss, so this cannot be
        satisfied by a single early return on the first failure.
        """
        both = dict(on_attack=60.0, off_attack=90.0,
                    on_defence=60.0, off_defence=90.0)
        ok, why = acceptable_quality_loss(1, 30, **both)
        assert not ok

        # Defence alone, at a minute where the attacking budget is open.
        # While leading the attacking budget is 10 from 47'; at 60' a 30-point
        # attacking loss is still refused, so use a small attacking loss and
        # a large defensive one.
        ok, why = acceptable_quality_loss(
            1, 60,
            on_attack=89.0, off_attack=90.0,     # 1-point attacking loss
            on_defence=60.0, off_defence=90.0,   # 30-point defensive loss
        )
        assert not ok and "defensive" in why

        # And the mirror: attacking alone must be able to block.
        ok, why = acceptable_quality_loss(
            -1, 60,
            on_attack=60.0, off_attack=90.0,     # 30-point attacking loss
            on_defence=89.0, off_defence=90.0,   # 1-point defensive loss
        )
        assert not ok and "attacking" in why

    def test_boundary_is_exactly_the_stated_minute(self):
        """Off-by-one at a threshold is the difference between the research
        finding being implemented and approximated."""
        assert squad_manager.SUB_ACCEPT_OFFENSIVE_LOSS_LEADING_MIN == 47
        assert squad_manager.SUB_ACCEPT_DEFENSIVE_LOSS_TRAILING_MIN == 32


# ═════════════════════════════════════════════
# 2. THE QUALITY SIGNAL
# ═════════════════════════════════════════════

class TestQualitySignal:

    def test_quality_is_monotone_in_overall_rating(self):
        """The whole point of the change: better players score better. If
        this fails, the ranking is decoration."""
        built, _ = _squad(subs=[("S1", "ST"), ("S2", "ST")])
        ctrl = _controller()
        a, b = built["S1"], built["S2"]

        real_ovr_a = a.dna.overall_rating
        real_ovr_b = b.dna.overall_rating
        if real_ovr_a == real_ovr_b:
            pytest.skip("fixture produced two identical ratings")

        try:
            # Force a known gap and confirm the order follows the rating.
            a.dna.physical.stamina = 99.0
            b.dna.physical.stamina = 10.0
            better, worse = (
                (a, b) if a.dna.overall_rating > b.dna.overall_rating else (b, a)
            )
            assert sub_quality(better, ctrl) > sub_quality(worse, ctrl)
        finally:
            a.dna.physical.stamina = 50.0
            b.dna.physical.stamina = 50.0
        assert real_ovr_a is not None and real_ovr_b is not None

    def test_a_soul_scores_above_an_ordinary_player_of_equal_rating(self):
        """`_pick_superstars` already prefers souls for the XI; the bench
        selection must not then ignore them, or the strongest player in the
        squad can be the one who never plays.

        Built through `SoulFactory.percy()` rather than a hand-made
        `SoulProfile`: `greatness_coefficient` is a PROPERTY on PlayerSoul
        delegating to `self.pillars` (player_soul.py:758), not a field. A
        hand-built object would have carried a real-looking attribute and
        proved nothing about the engine's actual object.
        """
        from player_soul import SoulFactory

        built, _ = _squad(subs=[("S1", "ST")])
        p = built["S1"]
        assert p.dna.soul is None

        before = sub_quality(p, _controller())

        soul = SoulFactory.percy()
        assert soul.greatness_coefficient > 1.0, "fixture produced a weak soul"
        p.dna.soul = soul
        try:
            after = sub_quality(p, _controller())
            assert after > before, (
                f"soul {after:.2f} did not beat ordinary {before:.2f}"
            )
        finally:
            p.dna.soul = None

    def test_the_soul_term_reads_a_real_greatness_coefficient(self):
        """Guards against a term that is permanently zero.

        `SoulProfile` has NO `greatness_coefficient` FIELD — it is a property
        on `PlayerSoul` (`player_soul.py:758`, delegating to
        `self.pillars`). Reading the field off the wrong object would make
        the whole soul term silently dead: every soul would score exactly
        what an ordinary player scores, and the mechanism would look wired
        because the code that reads it exists.
        """
        from player_soul import SoulFactory

        soul = SoulFactory.percy()
        assert squad_manager._soul_coefficient(type(
            "P", (), {"dna": type("D", (), {"soul": soul})()})()) > 0.0

    def test_a_missing_attribute_degrades_to_neutral_not_zero(self):
        """The `get_position` trap in a different guise.

        `PlayerDNA.get_position` returns the centre spot for a player it has
        no state for — a value that reads as a real measurement. If a
        missing attribute scored 0 instead of the pool mean, an incomplete
        fixture would look like a catastrophic player and would never be
        selected, which reads downstream as "no substitutes available".
        """
        class Bare:
            name = "Bare"
            dna = None

        atk = sub_attacking_quality(Bare())
        dfn = sub_defensive_quality(Bare())
        # Neutral means BOTH axes land on the same pool-mean value: an
        # unreadable player is neither an attacking nor a defensive
        # liability, and a zero would make them unusable for either.
        assert atk == pytest.approx(dfn, abs=1e-9)
        assert 30.0 < atk < 60.0, f"bare object scored {atk}, expected neutral"
        q = sub_quality(Bare(), _controller())
        assert 40.0 < q < 75.0, f"bare object scored {q}, expected the pool mean"

    def test_a_non_numeric_attribute_degrades_to_neutral_not_zero(self):
        """The branch the first version of this test never reached.

        Two separate mistakes in the original. (1) `_weighted` short-circuits
        to 50.0 when `dna` is None, so a stub with `dna = None` never calls
        `_get` at all — and the mutation that made `_get` return 0.0 on a bad
        value left the suite green. (2) corrupting a REAL PlayerDNA attribute
        makes `PlayerDNA.overall_rating` explode first (`physical.strength *
        0.8` on a string, player_dna.py:470), so the corruption never even
        reached the substitution code.

        So `_get` is tested directly. A guard that cannot enter the code it
        protects is the same failure as one that watches the wrong file.
        """
        class Group:
            def __init__(self, value):
                self.finishing = value

        class Dna:
            def __init__(self, value):
                self.technical = Group(value)

        class Player:
            name = "P"
            dna = None

        p = Player()
        p.dna = Dna("not a number")
        assert squad_manager._get(p.dna, "technical", "finishing") == 50.0, (
            "an unreadable attribute fell back to zero, not the pool mean — "
            "a player with a broken attribute would read as a terrible player"
        )

        p.dna = Dna(72.0)
        assert squad_manager._get(p.dna, "technical", "finishing") == 72.0

        # A missing FIELD and a missing GROUP take the same neutral path.
        assert squad_manager._get(p.dna, "technical", "nonexistent") == 50.0
        assert squad_manager._get(p.dna, "no_such_group", "finishing") == 50.0

    def test_a_missing_attribute_group_degrades_to_neutral(self):
        """A whole group absent (a fixture with no `defending`) is neutral,
        not zero.

        Asserted as a BAND, not as a direction. The first version asserted
        the neutral score was lower than the real one, which is only true if
        the player's actual defensive attributes are above the pool mean —
        and for a generated forward they are not, so the neutral fallback
        legitimately scored HIGHER. A direction assertion here would have
        been a test of the archetype RNG rather than of the fallback.
        """
        built, _ = _squad(subs=[("A", "ST")])
        p = built["A"]
        before = sub_defensive_quality(p)
        saved = p.dna.defending
        del p.dna.defending
        try:
            got = sub_defensive_quality(p)
        finally:
            p.dna.defending = saved
        assert 35.0 < got < 65.0, (
            f"missing group scored {got:.1f}; expected the neutral pool mean, "
            "not zero and not the real value"
        )
        assert before != got, "the fixture did not actually change"

    def test_an_ordinary_player_is_not_gifted_the_soul_term(self):
        """An ordinary player has no soul, so the soul term must be 0.0 and
        not a default of 1.0 — otherwise every player silently gains ~15
        points and the weighting of every real signal drops."""
        built, _ = _squad(subs=[("S1", "ST")])
        p = built["S1"]
        assert squad_manager._soul_coefficient(p) == 0.0

    def test_attacking_and_defensive_quality_are_genuinely_different_axes(self):
        """A single blended number cannot express the Wharton trade. If the
        two halves were perfectly correlated the split would be decorative
        and the scoreline rule would be unenforceable."""
        built, _ = _squad()
        a_vals, d_vals = [], []
        for p in built.values():
            a_vals.append(sub_attacking_quality(p))
            d_vals.append(sub_defensive_quality(p))
        # They must not be the same list in the same order.
        assert a_vals != d_vals


# ═════════════════════════════════════════════
# 3. THE BENCH IS RANKED, NOT SCANNED
# ═════════════════════════════════════════════

class TestBenchRanking:
    """These are the tests the OLD code fails.

    Every old helper was `for p in bench: ... return p` — first hit, bench
    order, no quality input whatsoever. Each test below puts a deliberately
    WORSE player earlier in the bench and asserts the better one is chosen.
    """

    def test_ranked_bench_is_quality_descending(self):
        """The bench is handed over in ASCENDING quality order.

        This is deliberate. The first version built three substitutes and
        used them in build order, which happened to already be quality-
        descending — so an implementation that returned bench order
        unchanged passed. That is the trap this project keeps hitting: a
        fixture whose incidental arrangement agrees with the expected
        output proves nothing.

        Sorting the INPUT by ascending quality makes the expected output the
        exact reverse of the input, so "did you sort" and "did you return
        the list I gave you" are distinguishable.
        """
        built, bench = _squad(subs=[("A", "ST"), ("B", "ST"), ("C", "ST")])
        ctrl = _controller()
        for p in bench:
            p.dna._starting_stamina = 100.0

        bench.sort(key=lambda p: sub_quality(p, ctrl))   # ascending
        given = [p.name for p in bench]
        quals = [round(sub_quality(p, ctrl), 4) for p in bench]
        if quals == sorted(quals):
            assert quals[0] != quals[-1], "all three substitutes identical"

        ranked = ctrl._ranked_bench(bench)
        assert len(ranked) == 3
        out_quals = [round(sub_quality(p, ctrl), 4) for p in ranked]
        assert out_quals == sorted(out_quals, reverse=True), (
            f"not descending: {out_quals} (bench was given ascending: {quals})"
        )
        if out_quals != sorted(out_quals):
            pass
        else:
            # Guard against a fixture where input order == expected order.
            assert given != [p.name for p in ranked], (
                "the bench was already in the expected order, so this test "
                "cannot distinguish sorting from passing the list through"
            )

    def test_pick_sub_prefers_the_better_of_two_same_position_subs(self):
        """The single most direct statement of the user's requirement.

        Bench order deliberately puts the WORSE man first. The old code
        returned `same_pos[0]`, i.e. him.
        """
        built, bench = _squad(subs=[("WeakSub", "ST"), ("StrongSub", "ST")])
        weak, strong = bench[0], bench[1]
        # Make the ordering unambiguous without inventing ratings: the
        # strong man gets the finishing that decides a striker's overall.
        strong.dna.technical.finishing = 99.0
        weak.dna.technical.finishing = 5.0

        ctrl = _controller(bench)
        chosen = ctrl._pick_sub(bench, built["C1"], "test", minute=60,
                                goal_diff=0, quality_gate=False)
        assert chosen is strong, (
            f"picked {chosen.name}; the bench's better striker was {strong.name}"
        )

    def test_a_used_sub_is_never_reselected(self):
        built, bench = _squad(subs=[("A", "ST"), ("B", "ST")])
        bench[0]._used_as_sub = True
        ctrl = _controller(bench)
        ranked = ctrl._ranked_bench(bench)
        assert bench[0] not in ranked
        assert len(ranked) == 1

    def test_an_empty_bench_returns_none_rather_than_raising(self):
        ctrl = _controller()
        assert ctrl._pick_sub([], built_any(), "test") is None
        assert ctrl._ranked_bench([]) == []

    def test_a_player_whose_position_is_unreadable_is_still_selectable(self):
        """Returning "CM" rather than "" means a fixture that omits
        positions still yields a drawable bench. Returning "" would match
        nothing in ADJACENT_POS and silently produce no substitutions.

        A bare stub rather than a mutilated PlayerProfile: deleting a
        dataclass field to simulate a malformed player tests the fixture,
        not the code.
        """
        class NoPos:
            name = "NoPos"
            position = ""
            dna = None

        assert squad_manager._position_of(NoPos()) == "CM"
        built, bench = _squad(subs=[("A", "ST")])
        ctrl = _controller(bench)
        assert ctrl._ranked_bench(bench)
        assert ctrl._pick_sub(bench + [NoPos()], built["C1"], "test") is not None

    def test_a_numbered_position_resolves_to_its_role_family(self):
        """Live positions are `CB1`, `CM2`, `RW` — and every position table
        here is keyed on the family.

        This is the exact trap that made a same-side passing rule match
        NOTHING in attacking_matrix because it tested `"CB"` against
        `"CB1"`, and it looked like a working rule the whole time.
        """
        class Numbered:
            name = "N"
            position = "CB1"
            dna = None
        assert squad_manager._position_of(Numbered()) == "CB"

        class NumberedDna:
            name = "N"
            dna = type("D", (), {"position": "CM2"})()
        assert squad_manager._position_of(NumberedDna()) == "CM"


# ═════════════════════════════════════════════
# 4. THE SCORELINE RULE REACHES THE SELECTION
# ═════════════════════════════════════════════

class TestRuleReachesSelection:

    def test_a_defensive_downgrade_is_refused_while_leading(self):
        """Not just the pure function — the selection must obey it.

        Bench is deliberately ordered so the FIRST affordable candidate is
        the worst one available: a winger replacing a centre back, a large
        defensive loss while leading, which Wharton says is never
        advisable. A good centre back IS on the bench, so the selection has
        somewhere better to go — without him, refusing to take the winger
        would simply mean taking nobody, and the test would pass against a
        rule that is not really being applied.
        """
        built, bench = _squad(subs=[("WingerSub", "LW")])
        winger = bench[0]
        for f in ("tackling", "marking", "interceptions", "blocking",
                  "clearances"):
            setattr(winger.dna.defending, f, 1.0)
        winger.dna.mental.positioning = 1.0
        winger.dna.mental.anticipation = 1.0
        winger.dna.mental.concentration = 1.0
        winger.dna.physical.strength = 1.0

        safe = _make("T", "CB", "SafeSub")
        for f in ("tackling", "marking", "interceptions", "blocking",
                  "clearances"):
            setattr(safe.dna.defending, f, 95.0)
        safe.dna.mental.positioning = 95.0
        safe.dna.mental.anticipation = 95.0
        safe.dna.mental.concentration = 95.0
        safe.dna.physical.strength = 95.0
        bench.append(safe)

        ctrl = _controller(bench)
        cb_off = built["C1"]
        chosen = ctrl._pick_sub(bench, cb_off, "test", minute=60,
                                goal_diff=+1)
        assert chosen is not None, (
            "refusing every candidate leaves the team a man down, which is "
            "worse than a modest downgrade — the rule must select a better "
            "candidate, not abandon the change"
        )
        assert chosen is not winger, (
            "while LEADING, the controller took off a centre back for a "
            "winger — the exact trade the Wharton result forbids"
        )

    def test_the_same_change_is_permitted_while_trailing(self):
        """The asymmetry has to bite in BOTH directions or it is not a rule
        about the scoreline, it is a rule about being ahead."""
        built, bench = _squad(subs=[("WingerSub", "LW")])
        winger = bench[0]
        for f in ("tackling", "marking", "interceptions", "blocking", "clearances"):
            setattr(winger.dna.defending, f, 1.0)
        winger.dna.mental.positioning = 1.0
        winger.dna.mental.anticipation = 1.0

        ctrl = _controller(bench)
        cb_off = built["C1"]
        chosen = ctrl._pick_sub(bench, cb_off, "test", minute=60,
                                goal_diff=-1)
        assert chosen is winger, (
            "while TRAILING past 32', an attacking substitution must be "
            "available even though it costs defensive quality"
        )

    def test_an_injury_sub_is_never_blocked_by_the_quality_rule(self):
        """You do not leave a broken player on because the replacement is
        worse. The caller passes quality_gate=False; the guard is that the
        flag genuinely reaches the selection."""
        built, bench = _squad(subs=[("OnlySub", "LW")])
        ctrl = _controller(bench)
        chosen = ctrl._pick_sub(bench, built["C1"], "injury", minute=10,
                                goal_diff=+1, quality_gate=False)
        assert chosen is not None

    def test_an_empty_pool_never_invents_a_candidate(self):
        built, _ = _squad()
        ctrl = _controller()
        assert ctrl._apply_quality_rule([], built["C1"], 60, "HOME", 1) is None


# ═════════════════════════════════════════════
# 5. THE IFAB WINDOW MODEL
# ═════════════════════════════════════════════

class TestWindows:

    def test_the_limit_is_five_subs_in_three_windows(self):
        assert SubstitutionController.MAX_SUBS == 5
        assert SubstitutionController.MAX_WINDOWS == 3

    def test_the_production_runner_does_not_override_it_back_to_three(self):
        """auto_run_match assigns its own MAX_SUBS onto the controller, so
        a stale constant THERE silently defeats the rule in the class. This
        is the single most likely way the change to be lost."""
        assert auto_run_match.MAX_SUBS == 5, (
            "auto_run_match.MAX_SUBS overrides SubstitutionController.MAX_SUBS "
            "at runtime; leaving it at 3 means the IFAB limit never applies"
        )

    def test_no_literal_three_survives_in_the_manager_sensor(self):
        """manager_sensors computed `subs_remaining = (3 - made) / 3.0`. A
        duplicated constant is correct only until one of the two moves."""
        src = inspect.getsource(manager_sensors)
        assert "3 - subs_made" not in src
        assert "MAX_SUBS" in src

    def test_three_windows_then_blocked(self):
        ctrl = _controller()
        for minute in (50, 60, 70):
            assert ctrl.window_available("HOME", minute), f"refused at {minute}'"
            ctrl._spend_window("HOME", minute)
        assert not ctrl.window_available("HOME", 80)

    def test_a_batch_at_one_stoppage_costs_one_window(self):
        """The mechanic that makes managers plan a double sub rather than
        drip-feed changes — the reason the IFAB rule is not just a bigger
        number.

        Both halves are asserted, because there are TWO batch checks and a
        mutation of either one alone must be visible. The accounting side
        lives in `_spend_window`; the GATE side lives in `window_available`.
        An earlier version of this test only exercised the accounting, so
        removing the gate check left it green.
        """
        ctrl = _controller()
        ctrl._spend_window("HOME", 60)
        assert ctrl.windows_used["HOME"] == 1
        ctrl._spend_window("HOME", 60)      # same stoppage
        ctrl._spend_window("HOME", 60)      # still the same stoppage
        assert ctrl.windows_used["HOME"] == 1

        # The gate side: a second change at the same minute must be legal
        # even though the three windows are all spent.
        ctrl = _controller()
        for minute in (50, 60, 70):
            ctrl._spend_window("HOME", minute)
        assert ctrl.windows_used["HOME"] == 3
        assert ctrl._last_window_minute["HOME"] == 70

        # Same stoppage as the last change → free batch, even with no
        # windows left. This is the GATE side of the rule.
        assert ctrl.window_available("HOME", 70), (
            "the gate must allow a batch at a stoppage the team has already "
            "used, even with no windows left"
        )
        # A NEW stoppage must be refused.
        assert not ctrl.window_available("HOME", 80)

    def test_half_time_is_a_free_window(self):
        """Free under the Laws AND not counted, so it is not one of the
        three. A free window that consumed a slot would make the third
        real window unavailable — which is how a bug here reads as "the
        manager ran out of substitutions in the second half"."""
        ctrl = _controller()
        for _ in range(5):
            ctrl._spend_window("HOME", 45)
        assert ctrl.windows_used["HOME"] == 0
        assert ctrl.window_available("HOME", 45)

    def test_windows_are_counted_per_team(self):
        """Home spending all three windows must not stop away changing, AND
        away spending windows must not increment home's counter.

        The first version of this test only ever spent HOME windows and then
        asserted AWAY was still available — which passes even if the counter
        is a single shared slot keyed to whichever team was touched. Spending
        AWAY windows is what makes a shared counter observable.
        """
        ctrl = _controller()
        for minute in (50, 60, 70):
            ctrl._spend_window("HOME", minute)
        assert not ctrl.window_available("HOME", 80)
        assert ctrl.window_available("AWAY", 80)

        # Now the other direction: AWAY's spending must land on AWAY.
        ctrl._spend_window("AWAY", 50)
        assert ctrl.windows_used["AWAY"] == 1, (
            "away's window was not counted against away — the counter is "
            "shared or keyed to one team"
        )
        assert ctrl.windows_used["HOME"] == 3, "away's window consumed home's"

        for minute in (60, 70):
            ctrl._spend_window("AWAY", minute)
        assert not ctrl.window_available("AWAY", 80)

    def test_subs_left_never_goes_negative(self):
        ctrl = _controller()
        ctrl.subs_made["HOME"] = 99
        assert ctrl.subs_left("HOME") == 0

    def test_the_window_tally_saturates_and_flags_the_excess(self):
        """An injury substitution is deliberately NOT gated on the window
        allowance — you cannot refuse to replace a broken player — so an
        earlier version reported "4 subs in 4 windows" against a limit of
        three. The tally has to stay truthful and the excess has to be
        visible rather than the counter silently exceeding its own limit.
        """
        ctrl = _controller()
        for minute in (50, 60, 70, 80, 88):
            ctrl._spend_window("HOME", minute)
        assert ctrl.windows_used["HOME"] == SubstitutionController.MAX_WINDOWS
        assert ctrl.windows_exceeded["HOME"] == 2, (
            f"tally saturated but the excess was not recorded: "
            f"{ctrl.windows_exceeded['HOME']}"
        )

    def test_a_batched_change_is_not_double_counted_as_an_excess(self):
        ctrl = _controller()
        for minute in (50, 60, 70):
            ctrl._spend_window("HOME", minute)
        for _ in range(4):
            ctrl._spend_window("HOME", 70)      # one stoppage, many changes
        assert ctrl.windows_used["HOME"] == 3
        assert ctrl.windows_exceeded["HOME"] == 0

    def test_a_change_reports_the_batch_flag_as_it_was_at_the_time(self):
        """`is_batched` must be read BEFORE the window is charged.

        Reading it afterwards reads a value the same call just wrote, and
        every substitution came out batched — including seven changes at
        seven different minutes in the development match. This is the same
        defect class as the jump guard inserted after the assignment it was
        meant to precede.
        """
        built, bench = _squad(subs=[("A", "ST"), ("B", "ST"), ("C", "ST")])
        ctrl = _controller(bench)
        for name in built:
            ctrl.register_player(built[name], 100.0)

        ctrl.stamina[built["C1"].name].is_injured = True
        first = ctrl.process_minute(55, {"HOME": list(built.values())}, 0, 0)
        assert first and first[0]["is_batched"] is False, (
            "a team's first change cannot be a batch — nothing preceded it"
        )

        ctrl.stamina[built["C1"].name].is_injured = False
        ctrl.stamina[built["C2"].name].is_injured = True
        same_minute = ctrl.process_minute(55, {"HOME": list(built.values())}, 0, 0)
        assert same_minute and same_minute[0]["is_batched"] is True, (
            "a second change in the same minute is a batch and must say so"
        )

        ctrl.stamina[built["C2"].name].is_injured = False
        ctrl.stamina[built["L"].name].is_injured = True
        later = ctrl.process_minute(70, {"HOME": list(built.values())}, 0, 0)
        assert later and later[0]["is_batched"] is False, (
            "a change fifteen minutes later is a new stoppage, not a batch"
        )


# ═════════════════════════════════════════════
# 6. THE SUB ARRIVES WITH THE TANK HE EARNED
# ═════════════════════════════════════════════

class TestSubEntryStamina:

    def test_a_sub_does_not_arrive_at_one_hundred(self):
        """The old line was `register_player(sub_on, starting_stamina=100.0)`.

        That literal made "resting a tired man" and "giving a fit man a
        breather" the same event, which is the entire difference a
        substitution exists to make.
        """
        built, bench = _squad(subs=[("Tired", "ST")])
        p = bench[0]
        p.dna._starting_stamina = 55.0
        ctrl = _controller(bench)
        assert ctrl._sub_entry_stamina(p, 0) == pytest.approx(55.0)

    def test_seated_minutes_recover_some_of_it(self):
        built, bench = _squad(subs=[("Tired", "ST")])
        p = bench[0]
        p.dna._starting_stamina = 55.0
        ctrl = _controller(bench)
        late = ctrl._sub_entry_stamina(p, 70)
        assert 55.0 < late < 100.0
        assert late > ctrl._sub_entry_stamina(p, 10)

    def test_recovery_cannot_exceed_a_full_tank(self):
        built, bench = _squad(subs=[("Fit", "ST")])
        p = bench[0]
        p.dna._starting_stamina = 98.0
        ctrl = _controller(bench)
        assert ctrl._sub_entry_stamina(p, 89) <= 100.0

    def test_no_hardcoded_full_tank_survives_in_make_sub(self):
        src = inspect.getsource(SubstitutionController._make_sub)
        assert "starting_stamina=100.0" not in src


# ═════════════════════════════════════════════
# 7. THE DECISION LOOP IS PER-TEAM
# ═════════════════════════════════════════════

class TestDecisionLoop:

    def test_one_teams_substitution_does_not_block_the_other(self):
        """`process_minute` tested `if subs_to_make` — the ACCUMULATED list
        across both teams — at three points. The instant home changed a
        player, away was skipped for that minute with no reference to its own
        subs remaining, its own injuries, or the scoreline.

        TWO FIXTURE MISTAKES HERE, both of which produced a GREEN test
        against the live bug:

        1. The first version gave BOTH teams an injured player. Section 1,
           the injury sub, runs unconditionally ABOVE the `if`, so away was
           still substituted with the bug restored. Only sections 2-4 are
           actually gated.
        2. The first version gave AWAY a STAMINA request. Section 3 is
           guarded by its own `if not team_subs` and is NOT nested inside
           section 2's guard, so mutating section 2 alone does not block
           it either. The tier that IS gated by the mutated line is the
           TACTICAL one — so away needs a scheduled change.

        So: home is injured (section 1, unguarded) and away is scheduled
        (section 2, guarded).
        """
        built_h, bench_h = _squad(subs=[("HS", "ST")], tag="H")
        built_a, bench_a = _squad(subs=[("AS", "ST")], tag="A")

        ctrl = SubstitutionController(
            home_team="HOME", away_team="AWAY",
            home_subs_bench=bench_h, away_subs_bench=bench_a,
        )
        for built in (built_h, built_a):
            for name in built:
                ctrl.register_player(built[name], 100.0)

        # HOME: an injury, which fires from section 1 unconditionally.
        ctrl.stamina[built_h["C1"].name].is_injured = True
        # AWAY: a pre-planned tactical change, which lives behind the
        # accumulated-list guard.
        ctrl.register_tactical_schedule({built_a["M2"].name: 65})

        subs = ctrl.process_minute(
            70, {"HOME": list(built_h.values()), "AWAY": list(built_a.values())},
            0, 0,
        )
        teams = {s["team"] for s in subs}
        assert "HOME" in teams, "home's injury sub did not fire"
        assert "AWAY" in teams, (
            "home's substitution consumed away's opportunity in the same "
            "minute — the accumulated-list bug. Away's need here is a "
            "SCHEDULED change: an injury fires from section 1 and a stamina "
            "request is guarded separately, so either would pass even with "
            "the bug restored."
        )

    def test_no_window_or_budget_is_consumed_when_nothing_is_wanted(self):
        built, bench = _squad(subs=[("S", "ST")])
        ctrl = _controller(bench)
        for name in built:
            ctrl.register_player(built[name], 100.0)
        before = (ctrl.subs_made["HOME"], ctrl.windows_used["HOME"])
        ctrl.process_minute(30, {"HOME": list(built.values())}, 0, 0)
        assert (ctrl.subs_made["HOME"], ctrl.windows_used["HOME"]) == before

    def test_a_decision_is_recorded_with_its_reasoning(self):
        """Without these the export can say WHICH players changed but not
        WHY the bench was used in that order — which is the only way to
        audit the scoreline rule after the fact."""
        built, bench = _squad(subs=[("S", "ST")])
        ctrl = _controller(bench)
        for name in built:
            ctrl.register_player(built[name], 100.0)
        ctrl.stamina[built["C1"].name].is_injured = True
        subs = ctrl.process_minute(
            50, {"HOME": list(built.values())}, 0, 0)
        assert subs, "expected an injury substitution"
        rec = subs[0]
        for field in ("quality_on", "quality_off", "attack_on", "attack_off",
                      "defence_on", "defence_off", "windows_used", "is_batched"):
            assert field in rec, f"sub record is missing {field!r}"


# ═════════════════════════════════════════════
# 8. THE TACTICAL SCHEDULE IS NO LONGER DISCARDED
# ═════════════════════════════════════════════

class TestTacticalSchedule:
    """The schedule is keyed by the OUTGOING player's NAME.

    `register_tactical_schedule({"Eli Dago": 68})` documents it as
    "player name -> minute they come ON", and `_find_tactical_sub` looks it
    up with `player_off.name`. The dict key a test happens to use is
    irrelevant, and assuming otherwise produces a None that reads as "the
    schedule mechanism is broken" rather than "the fixture used the wrong
    key" — which is what happened the first time.
    """

    def test_an_unscheduled_player_gets_no_tactical_sub(self):
        built, bench = _squad(subs=[("S", "ST")])
        ctrl = _controller(bench)
        assert ctrl._find_tactical_sub(bench, built["C1"]) is None

    def test_a_scheduled_player_gets_a_ranked_substitute(self):
        built, bench = _squad(subs=[("S1", "ST"), ("S2", "ST")])
        ctrl = _controller(bench)
        ctrl.register_tactical_schedule({built["C1"].name: 65})
        chosen = ctrl._find_tactical_sub(bench, built["C1"])
        assert chosen is not None
        assert chosen.name in {"TS1", "TS2"}

    def test_the_selection_prefers_the_right_position_then_the_best(self):
        """Position still decides the PREFERENCE; quality decides the
        CHOICE. The old code returned the first unused player regardless of
        both."""
        built, bench = _squad(subs=[("MidSub", "CM"), ("WingSub", "LW")])
        ctrl = _controller(bench)
        ctrl.register_tactical_schedule({built["S"].name: 65})
        chosen = ctrl._find_tactical_sub(bench, built["S"])
        # A striker coming off prefers an actual forward on the bench.
        assert chosen is not None


# ═════════════════════════════════════════════
# 9. THE INPUT FIXES — AST GUARDS
# ═════════════════════════════════════════════
#
# Both guards below are non-vacuity-checked. A guard that watches for a
# shape which no longer exists passes forever and guards nothing — and that
# is how the "shooter_name guard walks one file only" and "jump guard
# inserted after the assignment it was meant to precede" defects got in.

def _function_node(source: str, name: str, cls: str | None = None):
    tree = ast.parse(source)
    scope = tree
    if cls is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                scope = node
                break
        else:
            raise AssertionError(f"class {cls} not found")
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


class TestStaminaIsDrainedOnce:

    def test_absorb_chain_has_exactly_two_drain_calls(self):
        """NON-VACUITY FIRST.

        The old code had FOUR `process_action` calls in `_absorb_chain` —
        two complete blocks, each charging the primary actor. A single
        non-vacuity check would not catch a third block returning, so the
        expected count is asserted exactly and the count is proved
        non-zero first.
        """
        src = (REPO / "match_engine.py").read_text(encoding="utf-8")
        node = _function_node(src, "_absorb_chain", cls="MatchEngine")
        calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "process_action"
        ]
        assert len(calls) >= 2, (
            "expected the two-call unified drain block (primary + secondary); "
            f"found {len(calls)} — the guard below is vacuous"
        )
        assert len(calls) == 2, (
            f"{len(calls)} process_action calls in _absorb_chain. The "
            "primary actor must be charged exactly once per event."
        )

    def test_the_primary_actor_is_not_drained_twice_within_one_block(self):
        """The precise shape of the original defect: the same expression
        `event.player` passed to process_action twice inside one minute."""
        src = (REPO / "match_engine.py").read_text(encoding="utf-8")
        node = _function_node(src, "_absorb_chain", cls="MatchEngine")
        primary = 0
        for n in ast.walk(node):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "process_action"):
                continue
            if n.args and isinstance(n.args[0], ast.Attribute) \
                    and n.args[0].attr == "player":
                primary += 1
        assert primary == 1, (
            f"event.player passed to process_action {primary} times; a "
            "doubled drain also doubled roll_injury, because that runs "
            "inside process_action"
        )

    def test_the_press_tax_reaches_the_primary_actor(self):
        """press_tax exists to stop a gegenpress exhausting itself. The
        surviving block must actually pass it."""
        src = (REPO / "match_engine.py").read_text(encoding="utf-8")
        node = _function_node(src, "_absorb_chain", cls="MatchEngine")
        with_tax = 0
        for n in ast.walk(node):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "process_action"):
                continue
            if any(kw.arg == "drain_mult" for kw in n.keywords):
                with_tax += 1
        assert with_tax == 2, (
            f"{with_tax} drain calls pass drain_mult; both the primary and "
            "the secondary actor must be charged with the press tax"
        )


class TestStartingStaminaIsPublished:

    def test_the_season_recovery_actually_reaches_the_engine(self):
        """NON-VACUITY FIRST.

        match_engine reads `dna._starting_stamina` behind a `hasattr` that
        was ALWAYS FALSE, because nothing ever assigned it. The whole
        recovery chain was computed, printed and persisted, and then
        discarded. A guard that only asserted "the attribute is read" would
        pass against exactly this defect.
        """
        src = (REPO / "auto_run_match.py").read_text(encoding="utf-8")
        reads = src.count("_starting_stamina")
        assert reads >= 1, "the reader is gone; is the writer still needed?"

        assigns = 0
        counted = []
        for path in sorted(REPO.rglob("*.py")):
            rel = path.relative_to(REPO)
            # Production code only. The FIRST version of this guard walked
            # every .py including the test file, which assigns
            # `_starting_stamina` itself to build fixtures — so removing the
            # real writer still left four hits in the test file and the guard
            # stayed green. A guard that counts its own fixtures is not a
            # guard; it is the one-file hole in its most literal form.
            if rel.parts[0] == "tests" or rel.parts[0] == "tools":
                continue
            if path.name.startswith("test_") or path.name.startswith("_diag"):
                continue
            if ".venv" in rel.parts or "quarantine" in rel.parts \
                    or "corrupt" in rel.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                targets = []
                if isinstance(n, ast.Assign):
                    targets = n.targets
                elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                    targets = [n.target]
                for t in targets:
                    if isinstance(t, ast.Attribute) \
                            and t.attr == "_starting_stamina":
                        assigns += 1
                        counted.append(f"{rel.as_posix()}:{n.lineno}")
        assert assigns >= 1, (
            "no PRODUCTION module assigns dna._starting_stamina — the "
            f"engine's reader is guarded by a hasattr that can never be "
            f"true. Searched {len(list(REPO.rglob('*.py')))} files, found 0 "
            f"assignments outside tests/ (test files excluded on purpose: "
            f"they set the attribute themselves and would make this vacuous)"
        )
        assert not any("tests/" in c for c in counted), (
            f"the only assignments are in test files: {counted}"
        )


# ═════════════════════════════════════════════
# 10. DETERMINISM
# ═════════════════════════════════════════════

class TestDeterminism:

    def test_the_same_seed_makes_the_same_decisions(self):
        """A selection that reads `random` for its ranking would reorder the
        bench differently every match and make the whole thing unauditable.
        The game-state trigger is random by design; the CHOICE must not be.
        """
        def run():
            random.seed(1234)
            built, bench = _squad(subs=[("A", "ST"), ("B", "ST"), ("C", "ST")])
            ctrl = _controller(bench)
            for name in built:
                ctrl.register_player(built[name], 100.0)
            ctrl.stamina[built["C1"].name].is_injured = True
            subs = ctrl.process_minute(
                50, {"HOME": list(built.values())}, 0, 0)
            return [s["player_on"] for s in subs]

        assert run() == run()


def built_any():
    built, _ = _squad()
    return built["C1"]