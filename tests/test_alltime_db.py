"""Regression tests for alltime_db -- SQLite stats warehouse."""
import json
import sqlite3

import pytest

from alltime_db import (
    connect,
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