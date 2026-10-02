"""Regression tests for alltime_db -- SQLite stats warehouse."""
import json
import sqlite3

import pytest

from alltime_db import (
    add_alias,
    alias_scan,
    apply_alias_fixup,
    connect,
    export_app_json,
    get_or_create_player,
    get_or_create_team,
    init_schema,
    ingest_match_package,
    import_legacy_xlsx,
)

AGENTS = "synthetic teams"


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture()
def match_json(tmp_path):
    p = tmp_path / "Test_vs_Tests_MD1.json"
    doc = {
        "match": {
            "home_team": "Alpha",
            "away_team": "Beta",
            "score": "2\u20131",
            "home_xg": 2.1,
            "away_xg": 0.9,
            "matchday": 1,
            "season": "26/27",
            "competition": "PLOFA",
            "venue": "Stadium",
            "date": "2026-09-01",
        },
        "financials": {"attendance": 50000, "stadium_capacity": 65000},
        "goals": [
            {"minute": 12, "team": "Alpha", "scorer": "Striker One",
             "assist": "Creator Two", "situation": "open_play", "xg": 0.12},
            {"minute": 40, "team": "Beta", "scorer": "Winger Three",
             "assist": "", "situation": "penalty", "xg": 0.79},
            {"minute": 78, "team": "Alpha", "scorer": "Mid Four",
             "assist": "Creator Two", "situation": "fast_break", "xg": 0.2},
        ],
        "timeline": [
            {"minute": 3, "type": "PASS", "team": "Alpha", "player": "Creator Two"},
            {"minute": 10, "type": "CORNER_TAKEN", "team": "Alpha"},
            {"minute": 20, "type": "CORNER_TAKEN", "team": "Alpha"},
            {"minute": 55, "type": "CORNER_TAKEN", "team": "Beta"},
        ],
        "players": {
            "Striker One": {
                "player": "Striker One", "team": "Alpha", "position": "ST",
                "age": 25, "goals": 2, "xg": 1.4, "shots_on_target": 3,
                "shots_off_target": 1, "shots_blocked_att": 0,
                "passes_attempted": 10, "passes_completed": 7,
                "fouls_committed": 1, "yellow_cards": 1, "minutes_played": 90,
                "is_starter": True, "home_or_away": "home",
            },
            "Creator Two": {
                "player": "Creator Two", "team": "Alpha", "position": "CAM",
                "age": 24, "goals": 0, "xg": 0.2, "shots_on_target": 1,
                "shots_off_target": 0, "shots_blocked_att": 1,
                "passes_attempted": 40, "passes_completed": 33,
                "assists": 2, "xa": 0.8, "fouls_committed": 2,
                "minutes_played": 90, "is_starter": True, "home_or_away": "home",
            },
            "Winger Three": {
                "player": "Winger Three", "team": "Beta", "position": "RW",
                "age": 26, "goals": 1, "xg": 0.79, "shots_on_target": 2,
                "shots_off_target": 2, "shots_blocked_att": 0,
                "passes_attempted": 25, "passes_completed": 20,
                "minutes_played": 90, "is_starter": True, "home_or_away": "away",
            },
            "Mid Four": {
                "player": "Mid Four", "team": "Beta", "position": "CM",
                "age": 23, "goals": 0, "xg": 0.1, "shots_on_target": 0,
                "shots_off_target": 1, "shots_blocked_att": 0,
                "passes_attempted": 30, "passes_completed": 27,
                "minutes_played": 90, "is_starter": True, "home_or_away": "away",
            },
        },
    }
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def test_ingest_match_package_roundtrip(db_path, match_json):
    conn = connect(db_path)
    init_schema(conn)
    res = ingest_match_package(conn, str(match_json))
    assert res["home"] == "Alpha"
    assert res["away"] == "Beta"

    mid = conn.execute(
        "SELECT match_id, home_goals, away_goals FROM matches"
    ).fetchone()
    assert mid["home_goals"] == 2
    assert mid["away_goals"] == 1

    n_players = conn.execute(
        "SELECT COUNT(*) FROM player_match_stats"
    ).fetchone()[0]
    assert n_players == 4

    goals = conn.execute(
        "SELECT COUNT(*) AS n, SUM(is_penalty) AS pens FROM goals"
    ).fetchone()
    assert goals["n"] == 3
    assert goals["pens"] == 1

    alpha_corners = conn.execute(
        """SELECT corners FROM team_match_stats t
           JOIN teams te ON te.team_id = t.team_id WHERE te.name='Alpha'"""
    ).fetchone()["corners"]
    assert alpha_corners == 2

    alpha_shots = conn.execute(
        """SELECT shots, shots_on_target FROM team_match_stats t
           JOIN teams te ON te.team_id = t.team_id WHERE te.name='Alpha'"""
    ).fetchone()
    assert alpha_shots["shots"] == 6
    assert alpha_shots["shots_on_target"] == 4

    scorer = conn.execute(
        "SELECT SUM(goals) AS g FROM player_match_stats WHERE player_name='Striker One'"
    ).fetchone()["g"]
    assert scorer == 2


def test_ingest_is_idempotent(db_path, match_json):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(match_json))
    ingest_match_package(conn, str(match_json))
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0] == 3


def test_fidelity_tags(db_path, match_json):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(match_json))
    assert conn.execute(
        "SELECT DISTINCT fidelity FROM matches").fetchall()[0]["fidelity"] == "v26"
    assert conn.execute(
        "SELECT DISTINCT fidelity FROM player_match_stats").fetchall()[0]["fidelity"] == "v26"


def _pkg(tmp_path, name, matchday, home="Alpha", away="Beta"):
    doc = {
        "match": {
            "home_team": home, "away_team": away, "score": "1\u20130",
            "home_xg": 1.0, "away_xg": 0.3, "matchday": matchday,
            "season": "26/27", "competition": "PLOFA",
            "venue": "Stadium", "date": f"2026-09-0{matchday}",
        },
        "timeline": [],
        "players": {
            name: {
                "player": name, "team": home, "position": "ST",
                "age": 25, "goals": 1, "xg": 0.8, "shots_on_target": 2,
                "passes_attempted": 10, "passes_completed": 8,
                "minutes_played": 90, "is_starter": True,
                "home_or_away": "home",
            },
        },
        "goals": [{
            "minute": 10, "team": home, "scorer": name,
            "assist": "", "situation": "open_play", "xg": 0.8,
        }],
    }
    p = tmp_path / f"match_{matchday}_{name.replace(' ', '_')}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def test_alias_add_forwards_new_ingests(db_path):
    conn = connect(db_path)
    init_schema(conn)
    add_alias(conn, "Victor James", "Rayan Victor James")
    old = get_or_create_player(conn, "Victor James")
    new = get_or_create_player(conn, "Rayan Victor James")
    assert old == new


def test_alias_add_apply_rewires_and_dedupes(db_path, tmp_path):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(_pkg(tmp_path, "Victor James", 1)))
    ingest_match_package(conn, str(_pkg(tmp_path, "Victor James", 2)))
    ingest_match_package(conn, str(_pkg(tmp_path, "Rayan Victor James", 3)))

    old_by_name = conn.execute(
        "SELECT p.player_id FROM players p WHERE p.name='Victor James'"
    ).fetchone()
    new_by_name = conn.execute(
        "SELECT p.player_id FROM players p WHERE p.name='Rayan Victor James'"
    ).fetchone()

    add_alias(conn, "Victor James", "Rayan Victor James")
    moved, deleted = apply_alias_fixup(conn)
    assert moved + deleted == 2  # the two 26/27 rows now live under canonical

    gone = conn.execute(
        "SELECT COUNT(*) FROM player_match_stats WHERE player_id=?",
        (old_by_name["player_id"],)).fetchone()[0]
    assert gone == 0
    canon_rows = conn.execute(
        "SELECT COUNT(*) FROM player_match_stats WHERE player_id=?",
        (new_by_name["player_id"],)).fetchone()[0]
    assert canon_rows == 3
    names = conn.execute(
        "SELECT DISTINCT player_name FROM player_match_stats WHERE player_id=?",
        (new_by_name["player_id"],)).fetchall()
    assert all(r["player_name"] == "Rayan Victor James" for r in names)


def test_alias_scan_proposes_same_person_pairs(db_path, tmp_path):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(_pkg(tmp_path, "Victor James", 1)))
    ingest_match_package(conn, str(_pkg(tmp_path, "Rayan Victor James", 2)))
    cands = alias_scan(conn)
    hits = [c for c in cands
            if "Victor James" in (c["alias"], c["canonical"])
            and "Rayan Victor James" in (c["alias"], c["canonical"])]
    assert hits, "alias_scan should propose the Victor James rename pair"


def test_idempotent_alias_add_same_name(db_path):
    conn = connect(db_path)
    init_schema(conn)
    add_alias(conn, "Rayan Victor James", "Rayan Victor James")
    assert conn.execute("SELECT COUNT(*) FROM player_aliases").fetchone()[0] == 0


def _pkg_season(tmp_path, name, season, matchday, goals, xg,
                team="Alpha", date=None):
    doc = {
        "match": {
            "home_team": team, "away_team": "Beta", "score": "1\u20130",
            "home_xg": 1.0, "away_xg": 0.3, "matchday": matchday,
            "season": season, "competition": "PLOFA",
            "venue": "Stadium",
            "date": date or f"2026-09-0{matchday}",
        },
        "timeline": [],
        "players": {
            name: {
                "player": name, "team": team, "position": "ST",
                "age": 25, "goals": goals, "xg": xg, "shots_on_target": 2,
                "passes_attempted": 10, "passes_completed": 8,
                "yellow_cards": 1,
                "minutes_played": 90, "is_starter": True,
                "home_or_away": "home",
            },
        },
        "goals": [{
            "minute": 10, "team": team, "scorer": name,
            "assist": "", "situation": "open_play", "xg": xg,
        }],
    }
    p = tmp_path / f"match_{season.replace('/', '_')}_{matchday}_{name.replace(' ', '_')}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def _export_payload(conn, tmp_path):
    out = tmp_path / "history"
    export_app_json(conn, str(out))
    return json.loads((out / "player_history.json").read_text(encoding="utf-8"))


def test_export_app_json_roundtrip(db_path, tmp_path):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(_pkg_season(
        tmp_path, "Striker One", "24/25", 1, goals=2, xg=1.4,
        date="2024-09-01")))
    ingest_match_package(conn, str(_pkg_season(
        tmp_path, "Striker One", "26/27", 1, goals=1, xg=0.8)))

    payload = _export_payload(conn, tmp_path)
    assert payload["schema"] == "plofa_alltime_player_history_v1"
    assert payload["seasons"] == ["24/25", "25/26", "26/27"]

    entry = payload["players"]["Striker One"]
    assert set(entry["seasons"]) == {"24/25", "26/27"}
    legacy = entry["seasons"]["24/25"][0]
    assert legacy["team"] == "Alpha"
    assert legacy["apps"] == 1
    assert legacy["goals"] == 2
    assert legacy["minutes_played"] == 90
    assert legacy["yellow_cards"] == 1

    career = entry["career"]
    assert career["apps"] == 2
    assert career["goals"] == 3
    assert career["minutes_played"] == 180
    assert career["xg"] == pytest.approx(2.2)
    assert entry["last_season"] == "26/27"
    assert entry["current_team"] == "Alpha"
    assert entry["position"] == "ST"


def test_export_app_json_unifies_aliases(db_path, tmp_path):
    conn = connect(db_path)
    init_schema(conn)
    ingest_match_package(conn, str(_pkg_season(
        tmp_path, "Victor James", "24/25", 1, goals=1, xg=0.6,
        date="2024-09-01")))
    ingest_match_package(conn, str(_pkg_season(
        tmp_path, "Rayan Victor James", "25/26", 1, goals=2, xg=1.1,
        date="2025-09-01")))
    add_alias(conn, "Victor James", "Rayan Victor James")
    moved, deleted = apply_alias_fixup(conn)
    assert moved + deleted == 1

    payload = _export_payload(conn, tmp_path)
    assert "Victor James" not in payload["players"]
    entry = payload["players"]["Rayan Victor James"]
    assert set(entry["seasons"]) == {"24/25", "25/26"}
    assert entry["aliases"] == ["Victor James"]
    assert entry["career"]["apps"] == 2
    assert entry["career"]["goals"] == 3


def test_export_app_json_skips_odd_season_rows(db_path, tmp_path):
    conn = connect(db_path)
    init_schema(conn)
    tid = get_or_create_team(conn, "Ghosts")
    pid = get_or_create_player(conn, "Ghost Player")
    conn.execute(
        "INSERT INTO player_season_stats(season, team_id, player_id, "
        "player_name, fidelity, source) VALUES ('ALL-TIME',?,?,?,'legacy','test')",
        (tid, pid, "Ghost Player"))
    conn.commit()

    payload = _export_payload(conn, tmp_path)
    assert payload["players"] == {}