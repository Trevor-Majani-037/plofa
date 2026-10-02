"""
Test the engine → world crossing — world/ingest.py
=====================================================
Phase 6/7 integration. The world layer's two halves meet here, and the whole
point of this suite is a single negative claim:

    the ledger must NEVER re-derive anything the engine already decided.

Everything the engine owns — per-player lines, stamina, injuries — is taken
verbatim. The tests below assert that by planting recognisable sentinel values
in a fake engine result and checking they arrive in the ledger *unchanged*, and
by proving that when the engine says nothing, the adapter says nothing either
(rather than inventing a plausible number).

The final section runs a REAL match through the crossing, because a fake result
can only prove the translator's shape, not that the shapes actually line up.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from world.ingest import (
    SHOT_PARTS,
    apply_competition_context,
    apply_result,
    match_report_from_result,
    resolve_player_id,
)
from world.ledger import WorldLedger

SEASON = "26/27"
LEAGUE = "CMP-PLOFA"
HOME = "CLB-JUSTICE"
AWAY = "CLB-PEARLS"
WHEN = date(2026, 8, 8)


# ─────────────────────────────────────────────
# A FAKE ENGINE RESULT WITH RECOGNISABLE SENTINELS
# ─────────────────────────────────────────────

class _Cfg:
    match_date = WHEN
    home_team = "Justice"
    away_team = "Pearls"
    competition = "PLOFA"


class _State:
    home_goals = 0
    away_goals = 0
    sub_controller = None


class _Result:
    """Mimics the parts of ``MatchResult`` the adapter reads.

    ``home_goals``/``away_goals`` are PROPERTIES on the real MatchResult that
    read through to ``state`` — so the fake does the same, otherwise the adapter
    would read 0 and the test would prove nothing.
    """

    def __init__(self, stamina=None, home_goals=2, away_goals=1):
        self.config = _Cfg()
        self.state = _State()
        self.state.home_goals = home_goals
        self.state.away_goals = away_goals
        if stamina is not None:
            self.sub_controller = type("_C", (), {"stamina": stamina})()

    @property
    def home_goals(self) -> int:
        return self.state.home_goals

    @property
    def away_goals(self) -> int:
        return self.state.away_goals


class _Stam:
    def __init__(self, current, starting, injured=False,
                 injury_type="none", injury_minute=0):
        self.current_stamina = current
        self.starting_stamina = starting
        self.is_injured = injured
        self.injury_type = injury_type
        self.injury_minute = injury_minute


def _stats(**over):
    base = {
        "player": "Percy Osei", "team": "Justice", "minutes_played": 90,
        "goals": 1, "assists": 0, "shots_on_target": 2, "shots_off_target": 1,
        "shots_blocked_att": 1, "xg": 0.83, "yellow_cards": 0, "red_cards": 0,
    }
    base.update(over)
    return {"Percy Osei": base}


def _ledger():
    return WorldLedger(path="<memory>")


def _translate(result, stats=None, **kw):
    kw.setdefault("competition_id", LEAGUE)
    kw.setdefault("season_id", SEASON)
    kw.setdefault("home_id", HOME)
    kw.setdefault("away_id", AWAY)
    return match_report_from_result(result, stats, **kw)


# ─────────────────────────────────────────────
# VERBATIM PASS-THROUGH — THE CENTRAL CLAIM
# ─────────────────────────────────────────────

def test_engine_per_player_line_arrives_unchanged():
    """If the world and the published xlsx ever disagree about a match, the
    whole world layer is worthless. The line must be the exporter's, not a
    re-derivation."""
    stats = _stats(goals=2, assists=1, minutes_played=77, xg=1.234)
    report = _translate(_Result(), stats)
    line = report.players[0]
    assert line.goals == 2
    assert line.assists == 1
    assert line.minutes == 77
    assert line.xg == 1.234


def test_shots_are_the_sum_of_the_three_split_parts():
    """The exporter has no single "shots" key, so the ledger's one field is a
    definitional sum — pinned here so the choice is visible."""
    assert SHOT_PARTS == ("shots_on_target", "shots_off_target", "shots_blocked_att")
    stats = _stats(shots_on_target=3, shots_off_target=4, shots_blocked_att=2)
    assert _translate(_Result(), stats).players[0].shots == 9


def test_engine_stamina_arrives_as_a_lossless_unit_mapping():
    """0-100 stamina → 0-1 fatigue is a unit change, NOT a re-derivation."""
    result = _Result(stamina={"Percy Osei": _Stam(current=62.5, starting=95.0)})
    report = _translate(result, _stats())
    line = report.players[0]
    assert line.fatigue == pytest.approx(0.375, abs=1e-4)
    assert line.fitness == pytest.approx(0.95, abs=1e-4)


def test_engine_injury_arrives_verbatim_with_its_minute():
    result = _Result(stamina={"Percy Osei": _Stam(
        current=30.0, starting=90.0, injured=True,
        injury_type="muscle_strain", injury_minute=58)})
    line = _translate(result, _stats()).players[0]
    assert line.injury == "muscle_strain"
    assert line.injury_minute == 58


def test_the_controller_must_be_supplied_because_the_result_does_not_carry_it():
    """A real `MatchResult` has no `sub_controller` attribute — post-match
    stamina lives on the controller the caller handed to the engine. Pinned
    here because it is a genuine, surprising shape fact: an adapter that reads
    the result alone silently loses every fatigue and injury value.
    """
    from match_engine import MatchResult  # class only; no engine run needed
    import dataclasses
    assert "sub_controller" not in {f.name for f in dataclasses.fields(MatchResult)}

    # ...and the adapter is honest about losing them when it is not passed
    result = _Result()
    assert _translate(result, _stats()).players[0].fatigue is None
    # ...and recovers them when it is
    stamina = {"Percy Osei": _Stam(55.0, 88.0)}
    ctrl = type("_C", (), {"stamina": stamina})()
    line = _translate(result, _stats(), sub_controller=ctrl).players[0]
    assert line.fatigue == pytest.approx(0.45, abs=1e-4)


def test_cards_arrive_from_the_exporter_line():
    stats = _stats(yellow_cards=2, red_cards=1)
    line = _translate(_Result(), stats).players[0]
    assert line.yellow_cards == 2
    assert line.red_card is True


def test_absent_engine_values_stay_absent_rather_than_being_invented():
    """A match with no stamina controller must not produce plausible-looking
    fatigue. The ledger reads None as 'the engine did not say' and keeps what
    it had — inventing 0.0 or 1.0 here would silently corrupt a season."""
    report = _translate(_Result(), _stats())
    line = report.players[0]
    assert line.fatigue is None
    assert line.fitness is None
    assert line.injury is None
    assert line.confidence is None
    assert line.form is None


def test_absent_values_do_not_overwrite_what_the_ledger_already_knows():
    led = _ledger()
    pid = resolve_player_id("Percy Osei")
    led.record_match(_first_match(pid, fatigue=0.4, confidence=0.7))
    # a second match where the engine supplies nothing
    report = _translate(_Result(), _stats())
    led.record_match(report)

    state = led.player(pid)
    assert len(led.players) == 1, "the same person must be one identity"
    assert state.fatigue == pytest.approx(0.4), "an unstated value must not reset it"
    assert state.confidence == pytest.approx(0.7)
    # but the match's own line still landed
    assert state.minutes_played == 180
    assert state.appearances == 2


def _first_match(player_id, **kw):
    from world.ledger import MatchReport, PlayerMatchLine
    ln = PlayerMatchLine(player_id=player_id, club_id=HOME, minutes=90, **kw)
    return MatchReport(competition_id=LEAGUE, season_id=SEASON,
                       match_date=date(2026, 8, 1),
                       home_id=HOME, away_id=AWAY, home_goals=1, away_goals=0,
                       players=[ln])


def test_caller_supplied_confidence_and_form_are_passed_through():
    """confidence/form live in SeasonState, not MatchResult, so the caller
    supplies them — and they must arrive untouched."""
    result = _Result(stamina={"Percy Osei": _Stam(current=70.0, starting=90.0)})
    report = _translate(result, _stats(),
                        confidence={"Percy Osei": 0.83}, form={"Percy Osei": "WW"})
    line = report.players[0]
    assert line.confidence == 0.83
    assert line.form == "WW"


def test_only_players_filter_narrows_the_crossing():
    result = _Result()
    stats = {**_stats(), "Bench Player": {
        "player": "Bench Player", "team": "Pearls", "minutes_played": 0,
        "goals": 0, "assists": 0, "shots_on_target": 0, "shots_off_target": 0,
        "shots_blocked_att": 0, "xg": 0.0, "yellow_cards": 0, "red_cards": 0}}
    report = _translate(result, stats, only_players={"Percy Osei"})
    assert [ln.player_id for ln in report.players] == [
        resolve_player_id("Percy Osei")]


# ─────────────────────────────────────────────
# IDENTITY
# ─────────────────────────────────────────────

def test_player_names_resolve_to_canonical_ids_through_the_adapter():
    from world.ids import NameAdapter
    a = NameAdapter()
    pid = a.register("player", "Percy Osei")
    stats = _stats()
    report = _translate(_Result(), stats, adapter=a)
    assert report.players[0].player_id == pid


def test_an_unknown_player_still_gets_a_stable_id():
    """A cup guest or generated player must be recordable, and must resolve to
    the same ID every time."""
    first = resolve_player_id("Generated Cup Guest")
    second = resolve_player_id("Generated Cup Guest")
    assert first == second
    assert first.startswith("PLY-")
    assert resolve_player_id("generated  cup   guest") == first


def test_identity_survives_the_crossing_into_the_ledger():
    led = _ledger()
    apply_result(led, _Result(stamina={"Percy Osei": _Stam(60.0, 90.0)}),
                 _stats(goals=2), competition_id=LEAGUE, season_id=SEASON,
                 home_id=HOME, away_id=AWAY)
    pid = resolve_player_id("Percy Osei")
    assert led.player(pid).goals == 2
    assert led.season_line(pid, SEASON, LEAGUE)["goals"] == 2


# ─────────────────────────────────────────────
# RESULT-LEVEL FACTS
# ─────────────────────────────────────────────

def test_scoreline_and_competition_scope_come_across():
    led = _ledger()
    st = apply_result(led, _Result(home_goals=3, away_goals=0), _stats(),
                      competition_id=LEAGUE, season_id=SEASON,
                      home_id=HOME, away_id=AWAY, round_name="MD 1")
    assert st["score"] == "3-0"
    assert st["competition_id"] == LEAGUE
    assert st["players_recorded"] == 1
    assert st["players_with_lines"] == 1
    assert st["engine_values_carried"]["fatigue"] == 0


def test_apply_result_reports_how_many_engine_values_crossed():
    led = _ledger()
    st = apply_result(
        led, _Result(stamina={"Percy Osei": _Stam(50.0, 80.0, True, "knock", 70)}),
        _stats(), competition_id=LEAGUE, season_id=SEASON,
        home_id=HOME, away_id=AWAY)
    assert st["engine_values_carried"] == {"fatigue": 1, "fitness": 1,
                                            "injuries": 1}


def test_a_result_with_no_player_stats_still_records_the_match():
    led = _ledger()
    st = apply_result(led, _Result(), None, competition_id=LEAGUE,
                      season_id=SEASON, home_id=HOME, away_id=AWAY)
    assert st["players_recorded"] == 0
    assert len(led.match_history) == 1
    assert led.competition(LEAGUE) is not None


# ─────────────────────────────────────────────
# THE REVERSE DIRECTION
# ─────────────────────────────────────────────

def test_competition_context_folds_into_a_real_matchconfig_without_mutating_it():
    import dataclasses
    from match_engine import MatchConfig
    from world.competition import cup_rules, KnockoutCompetition

    ko = KnockoutCompetition("CMP-CUP", "Toland Cup", "CTR-TOL", "26/27",
                             cup_rules(), ["A", "B"], seed=1)
    base = MatchConfig(home_team="Justice", away_team="Pearls",
                       match_date=WHEN, matchday=1, season=SEASON)
    before = dataclasses.asdict(base)

    out = apply_competition_context(base, ko, round_name="Final")
    assert dataclasses.asdict(base) == before, "the input config was mutated"
    assert out.competition_id == "CMP-CUP"
    assert out.penalties is True
    assert out.extra_time is True
    assert out.round_name == "Final"


def test_a_league_competition_adds_no_ko_rules_to_a_real_config():
    """Every field the competition context does not own must survive untouched.
    The context fields themselves are expected to change — that is the whole
    point of the hand-off."""
    import dataclasses
    from match_engine import MatchConfig
    from world.competition import (
        CompetitionType, MatchContextFragment, league_rules, LeagueCompetition,
    )

    lg = LeagueCompetition("CMP-PLOFA", "PLOFA D1", "CTR-TOL", "26/27",
                           league_rules(), ["A", "B"], seed=1)
    base = MatchConfig(home_team="Justice", away_team="Pearls",
                       match_date=WHEN, matchday=1, season=SEASON)
    out = apply_competition_context(base, lg, round_name="MD 1")
    assert (out.extra_time, out.penalties) == (False, False)

    context_fields = {f.name for f in dataclasses.fields(MatchContextFragment)}
    original = dataclasses.asdict(base)
    result = dataclasses.asdict(out)
    for k, v in original.items():
        if k in context_fields:
            continue          # legitimately set by the competition
        assert result[k] == v, f"{k} changed but the competition does not own it"
    # and the ones it does own were actually applied
    assert out.competition_id == "CMP-PLOFA"
    assert out.competition_type == CompetitionType.LEAGUE
    assert out.round_name == "MD 1"


# ─────────────────────────────────────────────
# NON-PERSISTENCE
# ─────────────────────────────────────────────

def test_ingest_module_does_not_import_the_engine_at_module_level():
    import ast
    import world.ingest as wi
    tree = ast.parse(Path(wi.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("match_engine", "exporter", "squad_manager",
                      "season_manager", "alltime_db", "auto_run_match"):
        assert forbidden not in imported, (
            f"world/ingest.py imports {forbidden!r} at module level; the "
            f"engine must only be reached at call time"
        )


def test_ingest_does_not_write_to_any_26_27_file(tmp_path, monkeypatch):
    """The crossing must not write 26/27 state in either direction."""
    import builtins
    real_open = builtins.open
    forbidden = ("season_state.json", "season_stats.json", "manager_state.json",
                 "referee_state.json", "alltime.db")

    def guarded(file, mode="r", *a, **kw):
        name = Path(str(file)).name
        if "w" in mode and any(name == f for f in forbidden):
            raise AssertionError(f"ingest tried to write {name}")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    led = WorldLedger(path=str(tmp_path / "world_state.json"))
    apply_result(led, _Result(stamina={"Percy Osei": _Stam(60.0, 90.0)}),
                 _stats(), competition_id=LEAGUE, season_id=SEASON,
                 home_id=HOME, away_id=AWAY)
    led.save()
    assert (tmp_path / "world_state.json").exists()


# ─────────────────────────────────────────────
# A REAL MATCH THROUGH THE CROSSING
# ─────────────────────────────────────────────

def test_a_real_simulated_match_crosses_into_the_ledger():
    """A fake result can only prove the translator's shape. This runs an actual
    simulated match so the real ``MatchConfig`` / ``MatchResult`` / exporter-stats
    shapes are exercised — if any of them drift, this fails.

    Follows the documented non-persistent real path
    (``production_roster_sanity.py``): real Excel roster → SquadBuilder →
    MatchEngine. Nothing is written to any 26/27 ledger and
    ``auto_run_match.py`` is never executed.
    """
    import random

    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController
    # pure helper only — AGENTS.md is explicit that auto_run_match.py's helpers
    # may be imported but the module must never be EXECUTED
    from auto_run_match import _resolve_team_profile

    loader = get_loader()
    clubs = sorted(loader.get_all_clubs())
    home_name, away_name = clubs[0], clubs[1]

    home_raw = loader.build_matchday_squad(home_name)
    away_raw = loader.build_matchday_squad(away_name)
    home_squad = SquadBuilder.build(
        team_name=home_name, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"])
    away_squad = SquadBuilder.build(
        team_name=away_name, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"])

    home_profile = _resolve_team_profile(home_name, home_raw["formation"],
                                         is_home=True)
    away_profile = _resolve_team_profile(away_name, away_raw["formation"],
                                         is_home=False)

    config = MatchConfig(home_team=home_name, away_team=away_name,
                         match_date=WHEN, matchday=1, season="TEST",
                         venue=f"{home_name} Stadium", stadium_capacity=45000)
    sub_ctrl = SubstitutionController(
        home_team=home_name, away_team=away_name,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"])
    random.seed(7)
    engine = MatchEngine(config, home_profile, away_profile)
    engine.set_squad(home_name, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(away_name, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_ctrl)
    result = engine.simulate()

    # the exporter's own stat lines, the same dict the xlsx/csv/JSON use
    from exporter import PLOFAExporter
    all_players = {
        home_name: {"starters": home_squad["starters"],
                    "substitutes": home_squad["substitutes"]},
        away_name: {"starters": away_squad["starters"],
                    "substitutes": away_squad["substitutes"]},
    }
    exporter = PLOFAExporter(result, all_players, sub_controller=sub_ctrl)
    player_stats = exporter.accumulator.stats

    led = _ledger()
    st = apply_result(
        led, result, player_stats,
        competition_id="CMP-SCRATCH", season_id="TEST",
        home_id="CLB-H", away_id="CLB-A",
        sub_controller=sub_ctrl,
    )

    # 1. real players crossed, with real minutes
    assert st["players_recorded"] >= 20, "a full XI per side must cross"
    assert st["players_with_lines"] >= 20
    minutes = [p.minutes_played for p in led.players.values()]
    assert max(minutes) > 0

    # 2. the ledger agrees with the exporter EXACTLY — the whole point of
    #    reusing its lines rather than re-deriving them
    exported_minutes = sum(int(s.get("minutes_played", 0) or 0)
                           for s in player_stats.values())
    assert sum(minutes) == exported_minutes
    exported_goals = sum(int(s.get("goals", 0) or 0) for s in player_stats.values())
    assert sum(p.goals for p in led.players.values()) == exported_goals
    exported_xg = sum(float(s.get("xg", 0.0) or 0.0) for s in player_stats.values())
    assert sum(p.xg for p in led.players.values()) == pytest.approx(exported_xg, abs=1e-3)

    # 3. the engine's own stamina crossed as a real value, not a placeholder
    assert st["engine_values_carried"]["fatigue"] >= 20, (
        "the substitution controller's stamina must reach the ledger")
    fatigue_values = [p.fatigue for p in led.players.values()
                      if p.fatigue is not None]
    assert any(0.0 < f < 1.0 for f in fatigue_values), (
        "fatigue must be a real engine value in range, not a constant")

    # 4. the scoreline and the competition scope are right
    assert st["score"] == f"{result.home_goals}-{result.away_goals}"
    assert led.competition("CMP-SCRATCH") is not None

    # 5. the report serialises, so a world ledger can be written from a real match
    assert json.dumps(st)

    # 6. the whole ledger round-trips through a file
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "world_state.json"
        led.save(str(p))
        back = WorldLedger.load(str(p))
        assert sum(x.minutes_played for x in back.players.values()) == exported_minutes
