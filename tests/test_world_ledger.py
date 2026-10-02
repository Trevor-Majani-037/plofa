"""
Test PLOFA WORLD state ledger — world/ledger.py
=================================================
Phase 6 (audit §16 row 6). The centrepiece is plan §18's "CRITICAL INTEGRATION
TEST", run here as an executable scenario:

    Saturday domestic league
    Wednesday Champions League
    Saturday domestic league
    Wednesday domestic cup

    "The same player must have one continuous state throughout."

The design distinction under test is the one plan §7 demands and that a naive
ledger gets wrong:

  * CARRY-ACROSS state (fatigue, injuries, cards, confidence, form, workload)
    is global to the person and must NOT reset at a competition boundary;
  * SCOPED state (goals, assists, xG, appearances) belongs to one competition,
    and league and continental rows must COEXIST rather than overwrite.

Plus the 26/27 dual-write guard (audit §14) and idempotent persistence.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from world.competition import cup_rules, league_rules
from world.ledger import (
    CARRY_ACROSS_FIELDS,
    SCOPED_STAT_FIELDS,
    LedgerWriteError,
    MatchReport,
    PlayerMatchLine,
    PlayerState,
    WorldLedger,
    import_plofa_season_state,
    read_plofa_season_state,
)

SEASON = "26/27"
START = date(2026, 8, 8)          # Saturday
LEAGUE = "CMP-D1"
CONTINENTAL = "CMP-EUCL"
CUP = "CMP-CUP"

ALEX = "PLY-ALEX"
BLRA = "PLY-BLRA"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ledger() -> WorldLedger:
    return WorldLedger(path="world_state.json", country_id="CTR-TOL", season_id=SEASON)


def _competition(competition_id: str, name: str, country_id: str, rules):
    """Build the right competition class for the rules given.

    The ledger cares about identity and rules, not about progression, so this
    only needs a valid, correctly-typed competition object.
    """
    from world.competition import (
        CompetitionType, KnockoutCompetition, LeagueCompetition,
    )
    if rules.type == CompetitionType.LEAGUE:
        return LeagueCompetition(competition_id, name, country_id, SEASON,
                                 rules, ["CLB-A", "CLB-B"], seed=1)
    return KnockoutCompetition(competition_id, name, country_id, SEASON,
                               rules, ["CLB-A", "CLB-B"], seed=1)


def _register(ledger: WorldLedger) -> None:
    ledger.register_competition(
        _competition(LEAGUE, "Toland D1", "CTR-TOL", league_rules()))
    ledger.register_competition(
        _competition(CONTINENTAL, "Continental", "CTR-EU", league_rules()))
    ledger.register_competition(
        _competition(CUP, "Toland Cup", "CTR-TOL", cup_rules()))


def _match(ledger: WorldLedger, comp: str, day: date, home: str, away: str,
           lines, hg: int = 0, ag: int = 0) -> MatchReport:
    return MatchReport(
        competition_id=comp, season_id=SEASON, match_date=day,
        home_id=home, away_id=away, home_goals=hg, away_goals=ag,
        players=list(lines),
    )


def _line(pid: str, club: str, **kw) -> PlayerMatchLine:
    return PlayerMatchLine(player_id=pid, club_id=club, **kw)


# ─────────────────────────────────────────────
# THE WRITE GUARD (audit §14 dual-write rule)
# ─────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "season_state.json", "season_stats.json", "manager_state.json",
    "referee_state.json", "alltime.db", "./season_state.json",
    "Season_State.JSON",
])
def test_ledger_refuses_to_write_a_protected_26_27_file(path):
    """The live season ledger is authoritative and a played fixture cannot be
    replayed, so this must be refused in code, not by convention."""
    with pytest.raises(LedgerWriteError):
        WorldLedger(path=path)


def test_ledger_refuses_to_save_over_a_protected_file(tmp_path):
    led = WorldLedger(path=str(tmp_path / "world_state.json"))
    with pytest.raises(LedgerWriteError):
        led.save(str(tmp_path / "season_state.json"))
    with pytest.raises(LedgerWriteError):
        WorldLedger.load(str(tmp_path / "season_state.json"))


def test_ledger_allows_its_own_competition_scoped_file(tmp_path):
    path = tmp_path / "world" / "cup_state.json"
    led = WorldLedger(path=str(path))
    led.save()
    assert path.exists()


# ─────────────────────────────────────────────
# THE CRITICAL INTEGRATION TEST (plan §18)
# ─────────────────────────────────────────────

def test_plan18_one_month_three_competitions_one_continuous_state():
    """Saturday league → Wednesday continental → Saturday league → Wednesday cup.

    Plan §18: "The same player must have one continuous state throughout."
    """
    led = _ledger()
    _register(led)

    # Sat 8 Aug — domestic league
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1, shots=3, xg=0.8,
              fatigue=0.55, fitness=0.80, confidence=0.62, form="W"),
    ], hg=1))
    alex = led.player(ALEX)
    assert (alex.minutes_played, alex.goals, alex.workload) == (90, 1, 90)
    assert alex.fatigue == 0.55
    # the league goal is attributed to the league only, not to the global total
    # of every competition at once
    assert led.season_line(ALEX, SEASON, LEAGUE)["goals"] == 1

    # Wed 12 Aug — Champions League. Fatigue must CARRY OVER, not reset.
    led.record_match(_match(led, CONTINENTAL, date(2026, 8, 12), "CLB-A", "CLB-C", [
        _line(ALEX, "CLB-A", minutes=75, goals=0, shots=1, xg=0.2,
              fatigue=0.70, fitness=0.74),
    ]))
    assert alex.minutes_played == 165, "minutes must accumulate across competitions"
    assert alex.workload == 165
    assert alex.fatigue == 0.70, "fatigue must carry across the competition boundary"
    assert alex.fitness == 0.74

    # Sat 15 Aug — domestic league again
    led.record_match(_match(led, LEAGUE, date(2026, 8, 15), "CLB-B", "CLB-A", [
        _line(ALEX, "CLB-A", minutes=90, goals=2, shots=4, xg=1.1,
              fatigue=0.40, fitness=0.86, confidence=0.70, form="W"),
    ], ag=2))
    assert alex.minutes_played == 255
    assert alex.goals == 3                       # 1 + 0 + 2, one continuous total
    assert alex.fatigue == 0.40                 # recovered, still continuous
    assert alex.appearances == 3

    # Wed 19 Aug — domestic cup
    led.record_match(_match(led, CUP, date(2026, 8, 19), "CLB-A", "CLB-D", [
        _line(ALEX, "CLB-A", minutes=60, goals=0, xg=0.1,
              injury="muscle_strain", injury_minute=58, fatigue=0.9),
    ]))
    assert alex.minutes_played == 315
    assert alex.goals == 3, "a cup match with no goal must not change the total"
    assert alex.injuries and alex.injuries[-1]["injury_type"] == "muscle_strain"


def test_scoped_lines_coexist_and_do_not_overwrite_each_other():
    """Plan §8: a player's league and continental rows must coexist."""
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1, assists=1, shots=3, xg=0.8),
    ], hg=1))
    led.record_match(_match(led, CONTINENTAL, date(2026, 8, 12), "CLB-A", "CLB-C", [
        _line(ALEX, "CLB-A", minutes=90, goals=3, shots=6, xg=2.4),
    ], hg=3))

    league = led.season_line(ALEX, SEASON, LEAGUE)
    euro = led.season_line(ALEX, SEASON, CONTINENTAL)
    assert league["goals"] == 1 and euro["goals"] == 3
    assert league["minutes"] == 90 and euro["minutes"] == 90
    assert sorted(led.player(ALEX).competition_ids(SEASON)) == sorted([LEAGUE, CONTINENTAL])


def test_career_totals_are_a_query_over_the_scoped_lines():
    led = _ledger()
    _register(led)
    for comp, day, goals in ((LEAGUE, date(2026, 8, 8), 1),
                             (CONTINENTAL, date(2026, 8, 12), 3),
                             (CUP, date(2026, 8, 19), 2)):
        led.record_match(_match(led, comp, day, "CLB-A", "CLB-B", [
            _line(ALEX, "CLB-A", minutes=90, goals=goals, xg=float(goals)),
        ], hg=goals))

    career = led.career(ALEX)
    assert career["overall"]["goals"] == 6
    assert career["per_season"][SEASON]["goals"] == 6
    assert sorted(career["competitions"]) == sorted([CONTINENTAL, CUP, LEAGUE])
    # and the global total agrees with the sum of the lines — one truth
    assert led.player(ALEX).goals == 6


def test_seasons_do_not_leak_into_each_other():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1)]))
    other = MatchReport(competition_id=LEAGUE, season_id="27/28",
                        match_date=date(2027, 8, 7), home_id="CLB-A",
                        away_id="CLB-B", players=[_line(ALEX, "CLB-A", minutes=90, goals=5)])
    led.record_match(other)

    assert led.season_line(ALEX, SEASON, LEAGUE)["goals"] == 1
    assert led.season_line(ALEX, "27/28", LEAGUE)["goals"] == 5
    # the continuous state still spans both seasons
    assert led.player(ALEX).goals == 6


def test_carry_across_fields_are_all_present_on_player_state():
    """Plan §7's list must actually exist on the state object."""
    fields = set(PlayerState(player_id="PLY-X").to_dict())
    missing = [f for f in CARRY_ACROSS_FIELDS if f not in fields]
    assert not missing, f"plan §7 carry-across fields missing: {missing}"


def test_carry_across_and_scoped_field_lists_are_consistent():
    """The two lists must not drift.

    Every scoped statistic must accumulate into exactly one carry-across field,
    and that field must actually exist. The names differ on purpose ("minutes"
    on a season line, "minutes_played" on the continuous state), so the
    relationship is declared in ``SCOPED_TO_CARRY_ACROSS`` and checked here.
    """
    from world.ledger import SCOPED_TO_CARRY_ACROSS

    assert set(SCOPED_STAT_FIELDS) == set(SCOPED_TO_CARRY_ACROSS), (
        "the scoped stat list and its carry-across mapping have drifted"
    )
    fields = set(PlayerState(player_id="PLY-X").to_dict())
    unmapped = [s for s in SCOPED_STAT_FIELDS
                if SCOPED_TO_CARRY_ACROSS[s] not in fields]
    assert not unmapped, f"scoped stats map to missing carry-across fields: {unmapped}"
    # and every mapped target is itself a declared carry-across field
    assert set(SCOPED_TO_CARRY_ACROSS.values()) <= set(CARRY_ACROSS_FIELDS)


def test_a_continuous_total_is_never_smaller_than_one_competition_line():
    """The global total must dominate every scoped line — if it does not, the
    two were updated inconsistently."""
    led = _ledger()
    _register(led)
    for comp, goals in ((LEAGUE, 1), (CONTINENTAL, 3), (CUP, 2)):
        led.record_match(_match(led, comp, date(2026, 8, 8), "CLB-A", "CLB-B", [
            _line(ALEX, "CLB-A", minutes=90, goals=goals, xg=float(goals))],
            hg=goals))
    alex = led.player(ALEX)
    for comp in (LEAGUE, CONTINENTAL, CUP):
        line = led.season_line(ALEX, SEASON, comp)
        assert line["goals"] <= alex.goals
        assert line["minutes"] <= alex.minutes_played


# ─────────────────────────────────────────────
# AVAILABILITY (plan §13)
# ─────────────────────────────────────────────

def test_competition_scoped_suspension_rule_is_used():
    """A five-yellow ban in the league need not apply in a cup whose
    accumulation window differs (audit §4 rows 13–14 — rules are data)."""
    led = _ledger()
    _register(led)
    for i in range(5):
        led.record_match(_match(led, LEAGUE, date(2026, 8, 8) + timedelta(days=i * 7),
                                "CLB-A", "CLB-B",
                                [_line(ALEX, "CLB-A", minutes=90, yellow_cards=1)]))

    on = date(2026, 9, 20)
    ok_league, why_league = led.availability(ALEX, on, LEAGUE)
    assert not ok_league, why_league
    assert "suspended" in why_league
    # the cards really were recorded, one event each — an accumulation rule
    # cannot count cards the ledger never stored
    assert len(led.player(ALEX).yellow_card_events) == 5


def test_yellows_outside_the_window_no_longer_suspend():
    """The window is counted in MATCHES, so five yellows spread over a long
    season must not ban a player forever."""
    led = _ledger()
    _register(led)
    for i in range(5):
        led.record_match(_match(led, LEAGUE, date(2026, 8, 8) + timedelta(days=i * 7),
                                "CLB-A", "CLB-B",
                                [_line(ALEX, "CLB-A", minutes=90, yellow_cards=1)]))
        # five clean matches in between push the cards out of a 6-match window
        for j in range(5):
            led.record_match(_match(
                led, LEAGUE,
                date(2026, 8, 8) + timedelta(days=i * 7 + 1 + j),
                "CLB-A", "CLB-B", [_line(ALEX, "CLB-A", minutes=90)]))
    ok, why = led.availability(ALEX, date(2026, 12, 1), LEAGUE)
    assert ok, f"cards have aged out of the window: {why}"


def test_cup_with_a_different_window_does_not_inherit_the_league_ban():
    led = _ledger()
    _register(led)
    for i in range(5):
        led.record_match(_match(led, LEAGUE, date(2026, 8, 8) + timedelta(days=i * 7),
                                "CLB-A", "CLB-B",
                                [_line(ALEX, "CLB-A", minutes=90, yellow_cards=1)]))
    # a cup that accumulates over 12 matches, not 6
    led.register_competition(_competition(
        CUP, "Cup", "CTR-TOL",
        cup_rules(suspension_window=12, suspension_threshold=8)))
    ok, why = led.availability(ALEX, date(2026, 9, 20), CUP)
    assert ok, f"cup window is 12 — five league yellows must not ban: {why}"


def test_injury_makes_a_player_unavailable_and_then_returns_them():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=60, injury="knock", injury_minute=58),
    ]))
    ok, why = led.availability(ALEX, date(2026, 8, 9))
    assert not ok and "injured" in why
    # the ledger settles on MATCHDAY boundaries, not on wall-clock time, so the
    # next matchday is what returns him
    led.advance_matchday()
    ok2, why2 = led.availability(ALEX, date(2026, 8, 16))
    assert ok2, f"a knock is one match and must expire: {why2}"


def test_injuries_expire_as_matchdays_are_advanced():
    """Without an advance step an injured player is unavailable FOREVER,
    silently ending their season. This is the mechanic that prevents it."""
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=30, injury="acl")]))
    assert not led.availability(ALEX, date(2026, 8, 16))[0]

    for _ in range(30):
        led.advance_matchday()
    assert led.availability(ALEX, date(2026, 12, 1))[0], (
        "an ACL must not keep a player out forever"
    )
    assert led.player(ALEX).injuries == []


def test_suspension_is_served_then_the_player_returns():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, red_card=True),
    ]))
    assert not led.availability(ALEX, date(2026, 8, 9))[0]
    led.advance_matchday()
    ok, why = led.availability(ALEX, date(2026, 8, 16))
    assert ok, f"a one-match suspension must be served: {why}"
    assert led.player(ALEX).suspensions == []


def test_record_matchday_records_then_advances():
    led = _ledger()
    _register(led)
    led.record_matchday([
        _match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
            _line(ALEX, "CLB-A", minutes=90, goals=1, red_card=True)]),
    ])
    assert led.player(ALEX).minutes_played == 90
    # the suspension was served by the same call
    assert led.availability(ALEX, date(2026, 8, 16), LEAGUE)[0]


def test_long_term_injury_keeps_a_player_out_for_longer():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=30, injury="acl"),
    ]))
    assert not led.availability(ALEX, date(2026, 8, 16))[0]
    assert not led.availability(ALEX, date(2026, 10, 1))[0]


def test_red_card_suspends_for_one_match():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, red_card=True),
    ]))
    assert not led.availability(ALEX, date(2026, 8, 9))[0]
    assert led.player(ALEX).red_cards == 1


def test_unknown_player_is_reported_not_silently_available():
    led = _ledger()
    ok, why = led.availability("PLY-NOBODY", date(2026, 8, 8))
    assert not ok and "unknown" in why


# ─────────────────────────────────────────────
# IDEMPOTENCE & INTEGRITY
# ─────────────────────────────────────────────

def test_an_unused_substitute_is_not_an_appearance():
    """The exporter emits a stat line for every NAMED player, including unused
    substitutes with zero minutes. Counting those inflates every appearance
    total in the warehouse.

    Found by the phase-10 integration test, which flagged "0 matches with
    minutes but 1 appearance" for eight players in its very first match.
    """
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1),
        _line(BLRA, "CLB-A", minutes=0),      # named, never came on
    ], hg=1))

    assert led.player(ALEX).appearances == 1
    assert led.player(BLRA).appearances == 0
    assert led.player(BLRA).minutes_played == 0
    # the line is still recorded — an unused sub is not erased
    assert led.season_line(BLRA, SEASON, LEAGUE)["appearances"] == 0
    assert len(led.player(BLRA).match_history) == 1


def test_appearances_always_equal_matches_with_minutes():
    """The invariant the integration test asserts after every matchday."""
    led = _ledger()
    _register(led)
    for i in range(3):
        led.record_match(_match(led, LEAGUE, date(2026, 8, 8) + timedelta(days=i * 7),
                                "CLB-A", "CLB-B", [
            _line(ALEX, "CLB-A", minutes=90 if i < 2 else 0),
        ]))
    state = led.player(ALEX)
    played = sum(1 for m in state.match_history if m["minutes"] > 0)
    assert state.appearances == played == 2
    assert state.career_totals()["overall"]["appearances"] == 2


def test_a_played_substitute_counts():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=12),      # came on, 12 minutes
    ]))
    assert led.player(ALEX).appearances == 1
    assert led.player(ALEX).minutes_played == 12


def test_the_write_guard_is_at_the_ledger_constructor():
    """Belt and braces: the refusal is the FIRST thing that happens, before any
    state is built, so a misconfigured path cannot half-initialise."""
    with pytest.raises(LedgerWriteError):
        WorldLedger(path="alltime.db")


def test_recording_the_same_match_twice_is_a_no_op():
    """Re-running a fixture must not double-count a player's minutes."""
    led = _ledger()
    _register(led)
    report = _match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1)])
    led.record_match(report)
    led.record_match(report)
    assert led.player(ALEX).minutes_played == 90
    assert led.player(ALEX).goals == 1
    assert len(led.match_history) == 1


def test_dedupe_survives_a_save_load_round_trip():
    led = _ledger()
    _register(led)
    report = _match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1)])
    led.record_match(report)
    led.save()
    back = WorldLedger.load(led.path)
    back.record_match(report)
    assert back.player(ALEX).minutes_played == 90


def test_player_state_mutation_cannot_leak_into_the_stored_line():
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1)]))
    line = led.season_line(ALEX, SEASON, LEAGUE)
    line["goals"] = 999
    assert led.season_line(ALEX, SEASON, LEAGUE)["goals"] == 1


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def test_ledger_round_trips_through_a_real_file(tmp_path):
    led = _ledger()
    _register(led)
    led.register_club("CLB-A", "Avada Zenith")
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1, assists=1, shots=3, xg=0.8,
              fatigue=0.55, injury="knock", injury_minute=70),
    ], hg=1))
    path = tmp_path / "world_state.json"
    led.save(str(path))

    back = WorldLedger.load(str(path))
    alex = back.player(ALEX)
    assert alex.minutes_played == 90
    assert alex.fatigue == 0.55
    assert alex.injuries[0]["injury_type"] == "knock"
    assert back.season_line(ALEX, SEASON, LEAGUE)["goals"] == 1
    assert back.club("CLB-A").name == "Avada Zenith"
    assert back.competition(LEAGUE).name == "Toland D1"
    assert back.to_dict() == led.to_dict()


def test_saved_file_is_key_sorted_and_versioned(tmp_path):
    led = _ledger()
    _register(led)
    path = tmp_path / "world_state.json"
    led.save(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] >= 1
    assert list(raw) == sorted(raw)


def test_match_report_round_trips():
    report = _match(None, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1, xg=0.8, fatigue=0.5),
    ], hg=1)
    back = MatchReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert back == report


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def test_cross_competition_line_shows_continuity_and_scope_together():
    """The plan §18 view: one continuous block plus one row per competition."""
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90, goals=1, fatigue=0.5)]))
    led.record_match(_match(led, CONTINENTAL, date(2026, 8, 12), "CLB-A", "CLB-C", [
        _line(ALEX, "CLB-A", minutes=45, goals=2, fatigue=0.8)]))

    view = led.cross_competition_line(ALEX, SEASON)
    assert view["continuous"]["minutes_played"] == 135
    assert view["continuous"]["fatigue"] == 0.8
    assert view["per_competition"][LEAGUE]["goals"] == 1
    assert view["per_competition"][CONTINENTAL]["goals"] == 2
    assert view["career"]["goals"] == 3


def test_player_match_history_records_every_competition():
    led = _ledger()
    _register(led)
    for comp, day in ((LEAGUE, date(2026, 8, 8)),
                      (CONTINENTAL, date(2026, 8, 12)),
                      (CUP, date(2026, 8, 19))):
        led.record_match(_match(led, comp, day, "CLB-A", "CLB-B", [
            _line(ALEX, "CLB-A", minutes=90)]))
    assert len(led.played_in(ALEX, LEAGUE)) == 1
    assert len(led.player(ALEX).match_history) == 3
    assert len(led.played_in(ALEX, "CMP-NONE")) == 0


def test_summary_counts_everything(tmp_path):
    led = _ledger()
    _register(led)
    led.record_match(_match(led, LEAGUE, date(2026, 8, 8), "CLB-A", "CLB-B", [
        _line(ALEX, "CLB-A", minutes=90)]))
    text = led.summary()
    assert "26/27" in text
    assert "1 competition" not in text      # three are registered
    assert "1 match(es)" in text


# ─────────────────────────────────────────────
# THE 26/27 READ-ONLY ADAPTER (audit §14)
# ─────────────────────────────────────────────

def test_read_plofa_season_state_is_read_only_and_copies(tmp_path):
    """The adapter must not be able to mutate live state through what it
    returns."""
    live = tmp_path / "season_state.json"
    live.write_text(json.dumps({
        "season": "26/27", "matchday": 7,
        "standings": {"Justice": {"points": 20}},
        "players": {"a": {}, "b": {}},
        "fixture_ledger": {"1": ["x"]},
    }), encoding="utf-8")

    view = read_plofa_season_state(str(live))
    assert view["season"] == "26/27"
    assert view["player_count"] == 2
    view["standings"]["Justice"]["points"] = 999
    view["standings"]["Injected"] = {}

    raw = json.loads(live.read_text(encoding="utf-8"))
    assert raw["standings"]["Justice"]["points"] == 20
    assert "Injected" not in raw["standings"]


def test_import_seeds_competition_and_clubs_only(tmp_path):
    """Per-player form must NOT be duplicated out of the live ledger — the live
    SeasonState stays the authority for the PLOFA league."""
    live = tmp_path / "season_state.json"
    live.write_text(json.dumps({
        "season": "26/27", "standings": {"Justice": {}, "Pearls": {}},
        "players": {"Percy": {"form": "W"}},
    }), encoding="utf-8")

    led = WorldLedger(path=str(tmp_path / "world_state.json"))
    import_plofa_season_state(led, str(live))
    assert led.season_id == "26/27"
    assert led.competition("CMP-PLOFA") is not None
    assert set(led.clubs) == {"justice", "pearls"}
    assert led.players == {}, "live player state must not be copied into the world ledger"


def test_reading_live_state_never_opens_it_for_writing(tmp_path, monkeypatch):
    """The read-only adapter must not write even by accident. Point the open
    call at a guard that raises on any write mode."""
    live = tmp_path / "season_state.json"
    live.write_text(json.dumps({"season": "26/27", "standings": {},
                                "players": {}}), encoding="utf-8")
    before = live.read_bytes()

    import builtins
    real_open = builtins.open

    def guarded(file, mode="r", *a, **kw):
        if "w" in mode or "a" in mode or "+" in mode:
            raise AssertionError(f"the read-only adapter opened {file} for writing")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    read_plofa_season_state(str(live))
    assert live.read_bytes() == before


# ─────────────────────────────────────────────
# ARCHITECTURE INVARIANT
# ─────────────────────────────────────────────

def test_ledger_does_not_import_the_26_27_pipeline_at_module_level():
    """Scope guardrail: the ledger reaches the live types only at call time, and
    only through its own read-only adapter."""
    import world.ledger as wl
    for line in open(wl.__file__, encoding="utf-8"):
        if line.startswith(("import ", "from ")):
            assert "season_manager" not in line
            assert "match_engine" not in line
            assert "auto_run_match" not in line


def test_ledger_does_not_import_the_match_engine_anywhere():
    """plan §3/§10: the ledger records football, it never runs or reads it.

    Checked with ``ast`` rather than a text grep, so a docstring that *mentions*
    ``MatchResult`` (which this module legitimately does, to explain what it is
    independent of) is not mistaken for a dependency.
    """
    import ast
    import world.ledger as wl

    tree = ast.parse(open(wl.__file__, encoding="utf-8").read())
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for forbidden in ("match_engine", "season_manager", "squad_manager",
                      "auto_run_match", "exporter", "alltime_db"):
        assert forbidden not in imported, (
            f"world/ledger.py imports {forbidden!r} at any level — the ledger "
            f"records results, it must not consume the live pipeline"
        )


def test_suspension_thresholds_come_from_the_competition_not_a_literal():
    """audit §4 rows 13–14: five-yellows-in-six is competition DATA.

    Two competitions with different declared rules must behave differently, and
    a competition with no declared rule must fall back without crashing.
    """
    led = _ledger()
    _register(led)
    led.register_competition(_competition(
        CUP, "Cup", "CTR-TOL", cup_rules(suspension_threshold=3)))
    for i in range(3):
        led.record_match(_match(led, CUP, date(2026, 8, 8) + timedelta(days=i * 7),
                                "CLB-A", "CLB-B",
                                [_line(BLRA, "CLB-A", minutes=90, yellow_cards=1)]))
    # three yellows bans under the cup's own rule of 3
    assert not led.availability(BLRA, date(2026, 9, 1), CUP)[0]
    # ...but the same player is fine in a competition with no rule declared
    assert led.availability(BLRA, date(2026, 9, 1), "CMP-UNKNOWN")[0]

    rule = led.suspension_rule(CUP)
    assert rule["threshold"] == 3
    assert led.suspension_rule("CMP-UNKNOWN") is None
