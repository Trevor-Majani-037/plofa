"""
Test PLOFA WORLD competition framework — world/competition.py
==============================================================
Phase 3 (PLOFA_WORLD_LAYER_AUDIT.md §16) shipped league + knockout
progression without a suite. This file is the gate it should have shipped with.

Proves the Phase 2–3 contracts that matter:

  * rules are DATA — frozen, JSON round-trippable, presets differ correctly
  * ``MatchContextFragment`` is strictly additive and never mutates its input;
    a fragment with every field neutral reproduces 26/27 ``MatchConfig`` exactly
  * league progression reuses the production ``LeagueTable``/``FixtureList``
    and a full double round-robin actually completes
  * knockout progression is deterministic under seeding, handles byes, and
    refuses to invent a winner from an unresolvable tie
  * NO COMPETITION EVER RUNS ITS OWN ENGINE — the only hand-off to the match
    engine is ``MatchConfig`` (plan §10 invariant)
"""
from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from world.competition import (
    CompetitionRules,
    CompetitionType,
    KnockoutCompetition,
    KnockoutTie,
    LeagueCompetition,
    MatchContextFragment,
    cup_rules,
    league_rules,
)

# ─────────────────────────────────────────────
# RULES ARE DATA
# ─────────────────────────────────────────────

def test_rules_are_frozen():
    r = league_rules()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.points_win = 2  # type: ignore[misc]


def test_league_and_cup_presets_differ_where_it_matters():
    lg, cp = league_rules(), cup_rules()
    assert lg.type == CompetitionType.LEAGUE
    assert cp.type == CompetitionType.CUP
    assert lg.extra_time is False and cp.extra_time is True
    assert cp.penalties is True
    assert lg.schedule_kind == "round_robin"
    assert cp.schedule_kind == "knockout"
    assert lg.double_round_robin is True
    assert cp.double_round_robin is False


def test_preset_overrides_win():
    r = cup_rules(legs=2, aggregate=True, penalties=False)
    assert (r.legs, r.aggregate, r.penalties) == (2, True, False)


def test_rules_json_round_trip_is_value_preserving():
    r = cup_rules(
        tie_breakers=("pts", "gd"),
        qualification_out=({"competition_id": "CMP-EUCL", "slots": (1, 2)},),
    )
    back = CompetitionRules.from_dict(r.to_dict())
    assert back == r


def test_rules_to_dict_survives_a_real_json_round_trip():
    """A world data file that reads back subtly different from what was written
    is silent corruption, so assert the exact contract: dump → load → identical.

    ``qualification_out`` holds dicts whose own values may be tuples, so a
    shallow list() coercion is not enough.
    """
    import json
    r = cup_rules(
        tie_breakers=("pts", "gd"),
        qualification_out=({"competition_id": "CMP-EUCL", "slots": (1, 2)},),
    )
    d = json.loads(json.dumps(r.to_dict()))
    assert d["tie_breakers"] == ["pts", "gd"]
    assert d["qualification_out"] == [
        {"competition_id": "CMP-EUCL", "slots": [1, 2]}
    ]
    # a genuine file round trip: JSON turns the lists back into tuples
    assert CompetitionRules.from_dict(d) == r


def test_rules_from_dict_ignores_unknown_keys():
    """Forward compatibility: a newer world file must still load."""
    r = CompetitionRules.from_dict({"points_win": 2, "some_future_rule": True})
    assert r.points_win == 2


def test_rules_carry_the_hard_coded_26_27_values_as_data():
    """The audit §4 rows 13–14: MAX_SUBS=3 and 5-yellows-in-6 are world data,
    not engine constants. They must be expressible without touching code."""
    r = league_rules()
    assert r.substitutions_max == 3
    assert r.max_bench == 7
    assert (r.suspension_type, r.suspension_threshold, r.suspension_window) == (
        "yellow_accum", 5, 6,
    )


# ─────────────────────────────────────────────
# MATCH CONTEXT FRAGMENT — THE ONLY ENGINE HAND-OFF
# ─────────────────────────────────────────────

def _plofa_config():
    from match_engine import MatchConfig
    return MatchConfig(home_team="Justice", away_team="Pearls",
                       match_date=date(2026, 8, 8), matchday=1, season="26/27")


def test_neutral_fragment_reproduces_26_27_config_exactly():
    """The no-regression guarantee of plan §10–11 / audit §17: with every new
    field neutral, a 26/27 fixture produces byte-identical MatchConfig."""
    base = _plofa_config()
    out = MatchContextFragment().apply(base)
    assert dataclasses.asdict(out) == dataclasses.asdict(base)


def test_fragment_never_mutates_its_input():
    base = _plofa_config()
    before = dataclasses.asdict(base)
    MatchContextFragment(competition_id="CMP-CUP", stage_name="Semi-final").apply(base)
    assert dataclasses.asdict(base) == before


def test_fragment_omits_none_so_unset_context_never_overrides():
    """``None`` means 'this competition has no opinion' — applying it must not
    clobber a value the caller already set."""
    base = _plofa_config()
    base.competition_id = "CMP-PRESET"
    out = MatchContextFragment(competition_id=None, stage_name="Final").apply(base)
    assert out.competition_id == "CMP-PRESET"
    assert out.stage_name == "Final"


def test_fragment_duck_types_against_a_foreign_config():
    """The world layer must not require the PLOFA MatchConfig class; a
    competition can drive any engine that exposes the same field names."""
    @dataclasses.dataclass
    class ForeignConfig:
        home_team: str
        away_team: str
        importance: float = 1.0

    out = MatchContextFragment(importance=2.5).apply(
        ForeignConfig(home_team="A", away_team="B")
    )
    assert out.importance == 2.5


def test_fragment_ignores_fields_the_engine_does_not_have():
    """A cup in a country whose engine has no penalty support must still
    schedule cleanly rather than crash the season."""
    @dataclasses.dataclass
    class LeanConfig:
        home_team: str
        away_team: str

    out = MatchContextFragment(penalties=True, stage_name="Final").apply(
        LeanConfig(home_team="A", away_team="B")
    )
    assert not hasattr(out, "penalties")


# ─────────────────────────────────────────────
# LEAGUE
# ─────────────────────────────────────────────

def _league(teams, **kw):
    return LeagueCompetition(
        competition_id="CMP-D1", name="Toland First Division",
        country_id="CTR-TOL", season_id="26/27",
        rules=league_rules(**kw), participant_ids=teams, seed=7,
    )


def test_league_rejects_cup_rules():
    with pytest.raises(ValueError):
        LeagueCompetition("CMP-D1", "X", "CTR-TOL", "26/27", cup_rules(), ["A", "B"])


def test_league_rejects_duplicate_participants():
    with pytest.raises(ValueError):
        _league(["A", "B", "A"])


def test_double_round_robin_is_complete_and_balanced():
    lg = _league(["A", "B", "C", "D"])
    fx = lg.make_fixtures(date(2026, 8, 8))
    # 4 teams, double round robin → each pair twice, 12 fixtures
    assert len(fx.fixtures) == 12
    assert lg.matchdays() == 6
    for a, b in (("A", "B"), ("A", "C"), ("B", "D")):
        pair = [f for f in fx.fixtures if {f.home_team, f.away_team} == {a, b}]
        assert len(pair) == 2
        # and the home/away legs are mirrored
        assert {f.home_team for f in pair} == {a, b}


def test_single_round_robin_halves_the_fixture_count():
    lg = _league(["A", "B", "C", "D"], double_round_robin=False)
    assert len(lg.make_fixtures(date(2026, 8, 8)).fixtures) == 6


def test_league_standings_follow_the_competition_points_system():
    lg = _league(["A", "B", "C", "D"], double_round_robin=False)
    lg.make_fixtures(date(2026, 8, 8))
    lg.apply_result("A", "B", 1, 2, 0)
    lg.apply_result("C", "D", 1, 1, 1)
    table = {t.team: t for t in lg.standings()}
    assert table["A"].points == 3
    assert table["C"].points == 1
    # A scored and kept a clean sheet, so it must sit above the draw
    assert table["A"].goal_diff == 2
    assert lg.standings()[0].team == "A"


def test_league_context_carries_its_own_identity_and_no_ko_rules():
    lg = _league(["A", "B"])
    ctx = lg.context_for(round_name="MD 12")
    assert ctx.competition_id == "CMP-D1"
    assert ctx.competition_type == CompetitionType.LEAGUE
    assert ctx.round_name == "MD 12"
    assert (ctx.extra_time, ctx.penalties) == (False, False)


def test_league_rejects_non_rules_object():
    with pytest.raises(TypeError):
        LeagueCompetition("CMP", "X", "CTR", "26/27", {"points_win": 3}, ["A"])


# ─────────────────────────────────────────────
# KNOCKOUT
# ─────────────────────────────────────────────

def _cup(teams, seed=7, **kw):
    return KnockoutCompetition(
        competition_id="CMP-CUP", name="Toland Cup",
        country_id="CTR-TOL", season_id="26/27",
        rules=cup_rules(**kw), participant_ids=teams, seed=seed,
    )


def test_knockout_rejects_league_rules():
    with pytest.raises(ValueError):
        KnockoutCompetition("CMP", "X", "CTR", "26/27", league_rules(), ["A", "B"])


def test_knockout_refuses_aggregate_until_it_is_implemented():
    """Fail loudly at construction rather than silently simulating a
    single-leg tie that the rules say must be played over two legs."""
    with pytest.raises(NotImplementedError):
        _cup(["A", "B"], legs=2)


def test_odd_field_is_padded_to_a_power_of_two():
    ko = _cup(["A", "B", "C"])
    assert ko.round_names() == ("Semi-final", "Final")


def test_round_names_scale_with_the_field():
    assert _cup([chr(65 + i) for i in range(16)]).round_names() == (
        "Round of 16", "Quarter-final", "Semi-final", "Final",
    )
    assert _cup(["A", "B"]).round_names() == ("Final",)


def test_draw_is_deterministic_for_a_seed():
    a = _cup(["A", "B", "C", "D", "E", "F", "G", "H"]).draw_round("Quarter-final")
    b = _cup(["A", "B", "C", "D", "E", "F", "G", "H"]).draw_round("Quarter-final")
    assert [(t.home_id, t.away_id) for t in a] == [(t.home_id, t.away_id) for t in b]


def test_different_seeds_can_produce_different_draws():
    """Not a randomness test — a guard that the seed is actually wired in."""
    draws = {
        tuple(sorted((t.home_id, t.away_id) for t in
                     _cup(list("ABCDEFGH"), seed=s).draw_round("Quarter-final")))
        for s in range(40)
    }
    assert len(draws) > 1


def test_a_round_cannot_be_drawn_twice():
    ko = _cup(list("ABCD"))
    ko.draw_round("Semi-final")
    with pytest.raises(ValueError):
        ko.draw_round("Semi-final")


def test_unknown_round_raises():
    with pytest.raises(ValueError):
        _cup(list("ABCD")).draw_round("Group Stage")


def test_bye_advances_straight_into_the_next_round():
    ko = _cup(["A", "B", "C"])          # 3 entrants → 4 slots → one bye
    ties = ko.draw_round("Semi-final")
    assert len(ties) == 1
    assert set(ko.entrants_in("Final")) == set("ABC") - {
        t.home_id for t in ties
    } - {t.away_id for t in ties}


def test_winner_advances_round_by_round_to_a_champion():
    ko = _cup(list("ABCDEFGH"))
    ko.draw_round("Quarter-final")
    for t in ko.ties_in("Quarter-final"):
        ko.apply_result(t, 2, 1)
    assert len(ko.entrants_in("Semi-final")) == 4
    ko.draw_round("Semi-final")
    for t in ko.ties_in("Semi-final"):
        ko.apply_result(t, 0, 1)
    assert len(ko.entrants_in("Final")) == 2
    ko.draw_round("Final")
    for t in ko.ties_in("Final"):
        ko.apply_result(t, 1, 0)
    assert ko.winner() in set("ABCDEFGH")
    assert ko.winner() is not None


def test_level_tie_is_resolved_by_the_penalty_shootout():
    ko = _cup(["A", "B"])
    tie = ko.draw_round("Final")[0]
    ko.apply_result(tie, 1, 1, home_penalties=4, away_penalties=3)
    assert tie.winner_id == tie.home_id
    assert ko.winner() == tie.home_id


def test_level_tie_without_a_shootout_raises_rather_than_inventing_a_winner():
    ko = _cup(["A", "B"])
    tie = ko.draw_round("Final")[0]
    with pytest.raises(ValueError):
        ko.apply_result(tie, 2, 2)


def test_level_tie_when_competition_forbids_penalties_raises():
    ko = _cup(["A", "B"], penalties=False)
    tie = ko.draw_round("Final")[0]
    with pytest.raises(ValueError):
        ko.apply_result(tie, 1, 1, home_penalties=5, away_penalties=4)


def test_shootout_cannot_itself_be_drawn():
    ko = _cup(["A", "B"])
    tie = ko.draw_round("Final")[0]
    with pytest.raises(ValueError):
        ko.apply_result(tie, 1, 1, home_penalties=3, away_penalties=3)


def test_a_resolved_tie_cannot_be_resolved_again():
    ko = _cup(["A", "B"])
    tie = ko.draw_round("Final")[0]
    ko.apply_result(tie, 3, 0)
    with pytest.raises(ValueError):
        ko.apply_result(tie, 0, 5)


def test_negative_goals_raise():
    ko = _cup(["A", "B"])
    tie = ko.draw_round("Final")[0]
    with pytest.raises(ValueError):
        ko.apply_result(tie, -1, 0)


def test_winner_is_none_until_the_final_is_decided():
    ko = _cup(list("ABCD"))
    assert ko.winner() is None
    ko.draw_round("Semi-final")
    for t in ko.ties_in("Semi-final"):
        ko.apply_result(t, 1, 0)
    ko.draw_round("Final")
    assert ko.winner() is None      # drawn, but not played


def test_knockout_context_advertises_extra_time_and_penalties():
    ko = _cup(["A", "B"])
    ctx = ko.context_for(stage_name="Final", round_name="Final")
    assert ctx.extra_time is True
    assert ctx.penalties is True
    assert ctx.competition_type == CompetitionType.CUP


def test_kickoff_substitution_rules_reach_the_context():
    ko = _cup(["A", "B"], substitutions_max=5, max_bench=9)
    ctx = ko.context_for()
    assert ctx.substitution_rules == {"max": 5, "bench": 9}


# ─────────────────────────────────────────────
# THE ARCHITECTURE INVARIANT
# ─────────────────────────────────────────────

def test_no_competition_class_owns_a_match_engine():
    """Plan §10: 'no competition runs its own engine'. A competition may only
    ever describe a match (a MatchContextFragment) and record its result —
    it must not import or construct the engine."""
    import world.competition as wc
    src = open(wc.__file__, encoding="utf-8").read()
    for forbidden in ("MatchEngine", "import match_engine", "from match_engine"):
        assert forbidden not in src, (
            f"world/competition.py references {forbidden!r} — competitions "
            f"must describe matches, never run them"
        )


def test_competition_modules_do_not_import_the_26_27_pipeline_at_load():
    """Scope guardrail from the module docstring: no import-time coupling to
    the live pipeline (heavy, and it would drag the season state in).

    Only *module-level* imports count — function-level ones are the whole point
    of the lazy-import discipline and are checked to exist by the reuse tests.
    """
    import world.competition as wc
    import world.ids as wi
    import world.model as wm
    for mod in (wc, wi, wm):
        for line in open(mod.__file__, encoding="utf-8"):
            # a module-level import starts in column 0; an indented one is lazy
            if not line.startswith(("import ", "from ")):
                continue
            assert "season_manager" not in line, (
                f"{mod.__name__} imports season_manager at module level; "
                f"such imports must be lazy"
            )
            assert "match_engine" not in line, (
                f"{mod.__name__} imports match_engine at module level; "
                f"the engine must only be reached at call time"
            )


def test_tie_serialises_for_the_world_ledger():
    tie = KnockoutTie(round_name="Final", home_id="A", away_id="B")
    d = tie.to_dict()
    assert d["round_name"] == "Final" and d["played"] is False
    assert KnockoutTie(**d) == tie
