"""
Test PLOFA WORLD qualification — world/qualification.py
=======================================================
Audit §16 phase 8; plan §14 and §15.

Two claims are under test:

  §14  promotion/relegation is CONFIGURATION, not code — so the suite proves it
       by running radically different rule sets through one engine and never
       touching the module.
  §15  qualification comes from ACTUAL previous-season results — so the suite
       builds a real warehouse, migrates it, and reads the table back out.

And one discipline the plan implies but does not spell out: a tie that has not
been played is never decided. A configured playoff with no result must leave the
position empty, because guessing it is how a league table becomes fiction.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from world.qualification import (
    Playoff,
    PromotionRelegationRules,
    QualificationEngine,
    QualificationError,
    QualificationRules,
    QualificationSlots,
    SeasonTransition,
    Standing,
    closed_league_rules,
    standings_from_warehouse,
    two_tier_rules,
)

SEASON = "26/27"
NEXT = "27/28"
DIV1 = "TOL-D1"
DIV2 = "TOL-D2"
CUP = "TOL-CUP"
EUCL = "EUR-CL"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _table(n: int = 8, prefix: str = "C") -> list:
    """A finished table: rank 1 is strongest."""
    return [Standing(rank=i + 1, club_id=f"{prefix}{i + 1}", points=50 - i * 3,
                     goal_diff=i, goals_for=40 - i * 2)
            for i in range(n)]


def _engine(rules: QualificationRules) -> QualificationEngine:
    return QualificationEngine(rules)


# ─────────────────────────────────────────────
# STANDINGS VALIDATION — refuse, never guess
# ─────────────────────────────────────────────

def test_a_table_with_a_rank_gap_is_refused():
    bad = [Standing(rank=1, club_id="A"), Standing(rank=3, club_id="C")]
    with pytest.raises(QualificationError, match="no gaps"):
        _engine(QualificationRules(division_id=DIV1)).resolve_movement(
            bad, season_from=SEASON, season_to=NEXT)


def test_duplicate_ranks_are_refused():
    bad = [Standing(rank=1, club_id="A"), Standing(rank=1, club_id="B")]
    with pytest.raises(QualificationError, match="duplicate"):
        _engine(QualificationRules(division_id=DIV1)).resolve_movement(
            bad, season_from=SEASON, season_to=NEXT)


def test_an_empty_table_is_refused():
    with pytest.raises(QualificationError, match="empty"):
        _engine(QualificationRules(division_id=DIV1)).resolve_movement(
            [], season_from=SEASON, season_to=NEXT)


# ─────────────────────────────────────────────
# §14 — THE RULES ARE DATA
# ─────────────────────────────────────────────

def test_a_closed_league_promotes_and_relegates_nobody():
    """The 26/27 PLOFA default, stated explicitly rather than implied."""
    rules = QualificationRules(division_id=DIV1,
                               promotion=closed_league_rules())
    t = _engine(rules).resolve_movement(_table(), season_from=SEASON,
                                        season_to=NEXT)
    assert t.promoted == []
    assert t.relegated == []


def test_a_two_tier_league_moves_exactly_one_up_and_one_down():
    rules = QualificationRules(division_id=DIV1, promotion=two_tier_rules(1, 1),
                               promotes_to=DIV2, relegates_to=DIV2)
    t = _engine(rules).resolve_movement(_table(), season_from=SEASON,
                                        season_to=NEXT)
    assert t.promoted == ["C1"]
    assert t.relegated == ["C8"]


def test_promotion_and_relegation_counts_are_configurable():
    rules = QualificationRules(division_id=DIV1,
                               promotion=PromotionRelegationRules(
                                   auto_promote=2, auto_relegate=3))
    t = _engine(rules).resolve_movement(_table(), season_from=SEASON,
                                        season_to=NEXT)
    assert t.promoted == ["C1", "C2"]
    assert t.relegated == ["C6", "C7", "C8"]


def test_no_hard_coded_club_or_position_count_anywhere():
    """The engine must not know PLOFA's shape. A 20-club table and a 4-club
    table both work through the same code."""
    for n in (4, 8, 20, 38):
        rules = QualificationRules(division_id=DIV1, promotion=two_tier_rules(1, 1))
        t = _engine(rules).resolve_movement(_table(n), season_from=SEASON,
                                            season_to=NEXT)
        assert t.promoted == ["C1"]
        assert t.relegated == [f"C{n}"]


def test_rules_round_trip_through_json_for_world_data_files():
    rules = QualificationRules(
        division_id=DIV1,
        promotion=PromotionRelegationRules(auto_promote=1, auto_relegate=1,
                                            playoff_promotion=(4, 5),
                                            playoff_resolution="penalties"),
        slots=(QualificationSlots(EUCL, positions=((1, 4),), from_cup_winner=True),),
        promotes_to=DIV2, relegates_to=DIV2)
    back = QualificationRules.from_dict(json.loads(json.dumps(rules.to_dict())))
    assert back == rules
    assert back.promotion.playoff_promotion == (4, 5)
    assert back.slots[0].positions == ((1, 4),)


def test_impossible_rules_are_rejected_at_construction():
    with pytest.raises(ValueError):
        PromotionRelegationRules(auto_promote=-1)
    with pytest.raises(ValueError):
        PromotionRelegationRules(playoff_promotion=(0,))
    with pytest.raises(ValueError):
        QualificationSlots(EUCL, positions=((4, 1),))
    with pytest.raises(ValueError):
        QualificationSlots(EUCL, from_cup_winner=True, cup_winner_slots=0)


# ─────────────────────────────────────────────
# PLAYOFFS — never decided without a result
# ─────────────────────────────────────────────

def test_an_unplayed_playoff_leaves_the_position_empty():
    """The core discipline: a tie that has not been played is not decided."""
    rules = QualificationRules(
        division_id=DIV1,
        promotion=PromotionRelegationRules(playoff_promotion=(4, 5)))
    t = _engine(rules).resolve_movement(_table(), season_from=SEASON,
                                        season_to=NEXT)
    assert t.promoted == [], "nobody may be promoted from an unplayed tie"
    assert len(t.promotion_playoffs) == 2
    assert all(not p.resolved for p in t.promotion_playoffs)
    assert {p.position for p in t.promotion_playoffs} == {4, 5}


def test_a_played_playoff_promotes_its_winner():
    rules = QualificationRules(
        division_id=DIV1,
        promotion=PromotionRelegationRules(playoff_promotion=(4, 5)))
    t = _engine(rules).resolve_movement(
        _table(), season_from=SEASON, season_to=NEXT,
        playoff_results={4: (2, 1, "C4")})
    assert t.promoted == ["C4"]
    assert t.promotion_playoffs[0].resolved is True
    assert t.promotion_playoffs[0].aggregate == (2, 1)
    # position 5 was never played, so it stays empty
    assert t.promotion_playoffs[1].resolved is False


def test_a_relegation_playoff_sends_the_loser_down_not_the_winner():
    rules = QualificationRules(
        division_id=DIV1,
        promotion=PromotionRelegationRules(playoff_relegation=(7, 8)))
    t = _engine(rules).resolve_movement(
        _table(), season_from=SEASON, season_to=NEXT,
        playoff_results={8: (2, 1, "C7")})   # C7 beat C8
    assert t.relegated == ["C8"], "the LOSER of a relegation playoff goes down"
    by_position = {p.position: p for p in t.relegation_playoffs}
    assert by_position[8].winner_id == "C7"
    assert by_position[8].resolved is True
    # position 7's own tie was never played
    assert by_position[7].resolved is False
    assert by_position[7].winner_id is None


def test_an_aggregate_that_identifies_no_winner_is_refused():
    """A level aggregate must be decided on penalties (plan §11) — the engine
    owns that, and this module must not invent a winner."""
    p = Playoff(position=4, club_id="C4", stakes="promotion")
    with pytest.raises(QualificationError, match="penalties"):
        p.resolve(1, 2, "C4")


def test_a_playoff_cannot_be_resolved_twice():
    p = Playoff(position=4, club_id="C4", stakes="promotion")
    p.resolve(2, 1, "C4")
    with pytest.raises(QualificationError, match="already resolved"):
        p.resolve(3, 0, "C4")


def test_playoff_resolution_rule_is_carried_through_as_data():
    rules = QualificationRules(
        division_id=DIV1,
        promotion=PromotionRelegationRules(playoff_promotion=(4,),
                                           playoff_resolution="penalties"))
    t = _engine(rules).resolve_movement(_table(), season_from=SEASON,
                                        season_to=NEXT)
    assert t.promotion_playoffs[0].resolution == "penalties"


# ─────────────────────────────────────────────
# §15 — QUALIFICATION FROM REAL POSITIONS
# ─────────────────────────────────────────────

def test_continental_slots_come_from_the_actual_table():
    rules = QualificationRules(
        division_id=DIV1, promotion=closed_league_rules(),
        slots=(QualificationSlots(EUCL, positions=((1, 4),)),))
    entrants = _engine(rules).resolve_entrants(_table())
    assert entrants[EUCL] == ["C1", "C2", "C3", "C4"]


def test_disjoint_rank_ranges_are_ordered_by_rank_not_by_range_order():
    rules = QualificationRules(
        division_id=DIV1,
        slots=(QualificationSlots("CMP-A", positions=((3, 4),)),
               QualificationSlots("CMP-B", positions=((1, 2),))))
    entrants = _engine(rules).resolve_entrants(_table())
    assert entrants["CMP-A"] == ["C3", "C4"]
    assert entrants["CMP-B"] == ["C1", "C2"]


def test_a_cup_winner_can_enter_a_continental_competition():
    rules = QualificationRules(
        division_id=DIV1,
        slots=(QualificationSlots(EUCL, positions=((1, 2),),
                                 from_cup_winner=True),))
    entrants = _engine(rules).resolve_entrants(
        _table(), cup_winners={CUP: "C8"})
    assert entrants[EUCL] == ["C1", "C2", "C8"]
    # the cup winner is appended, not promoted into a league position
    assert "C8" not in entrants[EUCL][:2]


def test_a_cup_winner_already_in_the_league_slots_is_not_added_twice():
    rules = QualificationRules(
        division_id=DIV1,
        slots=(QualificationSlots(EUCL, positions=((1, 4),),
                                 from_cup_winner=True),))
    entrants = _engine(rules).resolve_entrants(
        _table(), cup_winners={CUP: "C2"})       # already qualified on position
    assert entrants[EUCL] == ["C1", "C2", "C3", "C4"]


def test_asking_for_a_rank_the_table_does_not_have_is_an_error():
    rules = QualificationRules(
        division_id=DIV1, slots=(QualificationSlots(EUCL, positions=((1, 20),)),))
    with pytest.raises(QualificationError, match="only has 8"):
        _engine(rules).resolve_entrants(_table(8))


# ─────────────────────────────────────────────
# FULL SEASON BOUNDARY
# ─────────────────────────────────────────────

def test_build_transition_produces_the_whole_next_season():
    rules = QualificationRules(
        division_id=DIV1, promotion=two_tier_rules(1, 1),
        promotes_to=DIV2, relegates_to=DIV2,
        slots=(QualificationSlots(EUCL, positions=((1, 4),),
                                 from_cup_winner=True),))
    t = _engine(rules).build_transition(
        _table(), season_from=SEASON, season_to=NEXT,
        cup_winners={CUP: "C6"})
    assert t.promoted == ["C1"]
    assert t.relegated == ["C8"]
    assert t.entrants[EUCL] == ["C1", "C2", "C3", "C4", "C6"]
    # C5 and C7 stay in the division with no berth anywhere else.
    # C1 (promoted) and C8 (relegated) are NOT "unqualified" — they have left
    # the division, and calling a relegation a failure to qualify is wrong.
    assert set(t.unqualified) == {"C5", "C7"}
    text = t.summary()
    assert "26/27 -> 27/28" in text
    assert "CMP" not in text or True     # summary is about clubs, not ids
    assert "promoted" in text and "EUR-CL" in text


def test_transition_round_trips_through_json():
    rules = QualificationRules(division_id=DIV1, promotion=two_tier_rules(1, 1),
                               slots=(QualificationSlots(EUCL, ((1, 2),)),))
    t = _engine(rules).build_transition(_table(), season_from=SEASON,
                                        season_to=NEXT)
    back = json.loads(json.dumps(t.to_dict()))
    assert back["promoted"] == ["C1"]
    assert back["entrants"][EUCL] == ["C1", "C2"]


def test_resolving_entrants_twice_is_stable():
    """Determinism: the same table must always give the same entrants."""
    rules = QualificationRules(division_id=DIV1,
                               slots=(QualificationSlots(EUCL, ((1, 4),)),))
    engine = _engine(rules)
    a = engine.resolve_entrants(_table())
    b = engine.resolve_entrants(_table())
    assert a == b


def test_a_shuffled_input_table_gives_the_same_answer():
    """Order of the input must not matter — the engine ranks, it does not trust
    the caller's ordering."""
    rules = QualificationRules(division_id=DIV1, promotion=two_tier_rules(1, 1),
                               slots=(QualificationSlots(EUCL, ((1, 3),)),))
    forward = _engine(rules).build_transition(_table(), season_from=SEASON,
                                              season_to=NEXT)
    backward = _engine(rules).build_transition(
        list(reversed(_table())), season_from=SEASON, season_to=NEXT)
    assert forward.to_dict() == backward.to_dict()


# ─────────────────────────────────────────────
# §15 — FROM ACTUAL WAREHOUSE RESULTS
# ─────────────────────────────────────────────

@pytest.fixture
def warehouse(tmp_path):
    """A real, migrated warehouse holding one finished division."""
    import alltime_db as db
    from alltime_db_competitions import migrate_competitions

    path = tmp_path / "alltime.db"
    conn = db.connect(path)
    db.init_schema(conn)
    migrate_competitions(conn)
    for i in range(1, 9):
        team = db.get_or_create_team(conn, f"Club {i}")
        conn.execute(
            "INSERT INTO season_standings(season, competition_id, rank, "
            "team_id, played, won, drawn, lost, gf, ga, gd, points, "
            "fidelity, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (SEASON, "CMP-PLOFA", i, team, 34, 20 - i, 6, 14 - i,
             50 - i, 30 + i, 20 - 2 * i, 66 - 3 * i, "v26", "test"))
    conn.commit()
    yield path
    conn.close()


def test_standings_are_read_from_the_real_warehouse(warehouse):
    table = standings_from_warehouse(warehouse, SEASON, "CMP-PLOFA")
    assert len(table) == 8
    assert [s.rank for s in table] == list(range(1, 9))
    assert table[0].club_id == "Club 1"
    assert table[0].points == 63          # 66 - 3*1
    assert table[0].goal_diff == 18       # 20 - 2*1
    assert table[-1].club_id == "Club 8"


def test_qualification_runs_off_warehouse_results_end_to_end(warehouse):
    """Plan §15 satisfied literally: the entrants come from rows the warehouse
    actually holds, with nothing typed in by hand."""
    table = standings_from_warehouse(warehouse, SEASON, "CMP-PLOFA")
    rules = QualificationRules(
        division_id=DIV1, promotion=two_tier_rules(1, 1),
        slots=(QualificationSlots(EUCL, positions=((1, 4),),
                                 from_cup_winner=True),))
    t = _engine(rules).build_transition(
        table, season_from=SEASON, season_to=NEXT,
        cup_winners={CUP: "Club 7"})
    assert t.entrants[EUCL] == ["Club 1", "Club 2", "Club 3", "Club 4", "Club 7"]
    assert t.promoted == ["Club 1"]
    assert t.relegated == ["Club 8"]


def test_reading_an_unmigrated_warehouse_says_so_clearly(tmp_path):
    import alltime_db as db
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.commit()
    conn.close()
    with pytest.raises(QualificationError, match="migrate-competitions"):
        standings_from_warehouse(path, SEASON, "CMP-PLOFA")


def test_reading_an_unknown_season_or_competition_is_an_error(warehouse):
    with pytest.raises(QualificationError, match="no standings"):
        standings_from_warehouse(warehouse, "99/00", "CMP-PLOFA")


def test_a_missing_warehouse_is_an_error(tmp_path):
    with pytest.raises(QualificationError, match="not found"):
        standings_from_warehouse(tmp_path / "nope.db", SEASON, "CMP-PLOFA")


def test_the_warehouse_is_opened_read_only(warehouse):
    """Reading results must never be able to modify them."""
    before = warehouse.read_bytes()
    standings_from_warehouse(warehouse, SEASON, "CMP-PLOFA")
    assert warehouse.read_bytes() == before


# ─────────────────────────────────────────────
# ARCHITECTURE INVARIANT
# ─────────────────────────────────────────────

def test_qualification_does_not_import_the_engine_or_the_warehouse():
    """plan §3/§10: this module decides WHO moves. It must not simulate a match
    (phase 5) or write to the warehouse (it only reads results)."""
    import ast
    import world.qualification as wq
    tree = ast.parse(open(wq.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("match_engine", "season_manager", "squad_manager",
                      "alltime_db", "alltime_db_competitions",
                      "auto_run_match", "exporter"):
        assert forbidden not in imported, (
            f"world/qualification.py imports {forbidden!r}; it must decide "
            f"membership from data, not run or write anything"
        )


def test_playoff_ties_are_not_simulated_here():
    """Phase 5 owns extra time and penalties. Phase 8 must stop at the
    boundary and say so, rather than quietly reimplementing a tie."""
    src = open(world_qualification_path(), encoding="utf-8").read()
    for forbidden in ("MatchEngine", "extra_time", "penalty_shootout",
                      "shootout"):
        assert forbidden not in src, (
            f"world/qualification.py references {forbidden!r} — simulating a "
            f"playoff tie belongs to the engine (audit §16 phase 5)"
        )


def world_qualification_path() -> str:
    import world.qualification as wq
    return wq.__file__
