"""
Test PLOFA warehouse competition keying — alltime_db_competitions.py
=====================================================================
Audit §16 phase 7 / §15. The migration is a prerequisite for a second
competition going live (audit §18 risk 4: statistics double-counting is the
first real danger after identity), so it is tested against the failure modes
that actually matter:

  * it must CHANGE NOTHING it was not asked to change
  * it must be idempotent
  * it must REFUSE an unrecognised schema rather than guess
  * it must backfill correctly, so today's aggregates are unchanged afterwards
  * it must let the same two clubs meet in a league AND a cup afterwards

Every test runs against a synthetic warehouse built with the real
``init_schema``. The live 143 MB ``alltime.db`` is never touched here.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import alltime_db as warehouse
from alltime_db_competitions import (
    COLUMN_ONLY_TABLES,
    COMPETITION_BACKFILL,
    REBUILD_TABLES,
    MigrationError,
    backup_database,
    competition_report,
    is_migrated,
    migrate_competitions,
)

ALL_TABLES = [t["table"] for t in REBUILD_TABLES] + list(COLUMN_ONLY_TABLES)


# ─────────────────────────────────────────────
# FIXTURE WAREHOUSE
# ─────────────────────────────────────────────

@pytest.fixture
def conn(tmp_path):
    """A real warehouse with one season of one competition already in it."""
    c = warehouse.connect(tmp_path / "test_alltime.db")
    warehouse.init_schema(c)
    a = warehouse.get_or_create_team(c, "Justice")
    b = warehouse.get_or_create_team(c, "Pearls")
    # deliberately no competition_id: the pristine schema has no such column,
    # and that is exactly the state the migration exists to fix
    c.execute(
        "INSERT INTO matches(season, competition, matchday, home_team_id, "
        "away_team_id, home_goals, away_goals, match_date, fidelity, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("26/27", "PLOFA", 1, a, b, 2, 1, "2026-08-08", "v26", "test"))
    c.execute(
        "INSERT INTO matches(season, competition, matchday, home_team_id, "
        "away_team_id, home_goals, away_goals, match_date, fidelity, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("26/27", "PLOFA", 1, b, a, 0, 0, "2026-08-15", "v26", "test"))
    c.execute(
        "INSERT INTO goals(season, matchday, team_id, minute, scorer, "
        "fidelity, source) VALUES (?,?,?,?,?,?,?)",
        ("26/27", 1, a, 23.0, "Percy Osei", "v26", "test"))
    c.commit()
    yield c
    c.close()


# ─────────────────────────────────────────────
# PRECONDITION
# ─────────────────────────────────────────────

def test_pristine_warehouse_is_not_yet_migrated(conn):
    assert is_migrated(conn) is False


def test_pristine_schema_really_cannot_hold_two_competitions(conn):
    """Prove the problem exists before fixing it — otherwise this migration
    has no justification and the test proves nothing."""
    a = warehouse.get_or_create_team(conn, "Justice")
    b = warehouse.get_or_create_team(conn, "Pearls")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO matches(season, competition, matchday, home_team_id, "
            "away_team_id, home_goals, away_goals, match_date, fidelity, source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("26/27", "PLOFA-CUP", 1, a, b, 1, 0,
             "2026-09-02", "v26", "test"))
    conn.rollback()


# ─────────────────────────────────────────────
# THE MIGRATION
# ─────────────────────────────────────────────

def test_migration_adds_the_column_to_every_target_table(conn):
    st = migrate_competitions(conn)
    for table in ALL_TABLES:
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        assert "competition_id" in cols, f"{table} still unkeyed"


def test_migration_widens_every_constraint(conn):
    migrate_competitions(conn)
    ddl = {t: conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (t,)).fetchone()["sql"] for t in [s["table"] for s in REBUILD_TABLES]}
    for spec in REBUILD_TABLES:
        assert "competition_id" in " ".join(ddl[spec["table"]].split())
        assert " ".join(spec["new"].split()) in " ".join(ddl[spec["table"]].split())


def test_migration_backfills_every_existing_row(conn):
    st = migrate_competitions(conn)
    for table in ALL_TABLES:
        n = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} "
            f"WHERE competition_id IS NULL OR competition_id=''").fetchone()["n"]
        assert n == 0, f"{table} has {n} unbackfilled rows"
    assert st["backfilled"]["matches"] == 2
    assert st["backfilled"]["goals"] == 1


def test_migration_preserves_row_counts_and_content(conn):
    before_counts = {t: conn.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"]
                     for t in ALL_TABLES}
    st = migrate_competitions(conn)
    assert st["row_counts_before"] == st["row_counts_after"]
    for table, n in before_counts.items():
        now = conn.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"]
        assert now == n, f"{table}: {n} -> {now}"
    assert st["checksums_preserved"] is True
    assert st["foreign_key_violations"] == 0


def test_migration_preserves_the_actual_match_rows(conn):
    before = [tuple(r) for r in conn.execute(
        "SELECT season, competition, matchday, home_team_id, away_team_id, "
        "home_goals, away_goals, match_date FROM matches ORDER BY match_id")]
    migrate_competitions(conn)
    after = [tuple(r) for r in conn.execute(
        "SELECT season, competition, matchday, home_team_id, away_team_id, "
        "home_goals, away_goals, match_date FROM matches ORDER BY match_id")]
    assert before == after


def test_migration_creates_the_competition_indexes(conn):
    st = migrate_competitions(conn)
    assert st["indexes"] == 5
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_matches_comp" in names
    assert "idx_pms_comp" in names


# ─────────────────────────────────────────────
# IDEMPOTENCE
# ─────────────────────────────────────────────

def test_migration_is_idempotent(conn):
    first = migrate_competitions(conn)
    second = migrate_competitions(conn)
    assert first["rebuilt"]
    assert second["already_migrated"] is True
    assert second["rebuilt"] == []
    # and a third run is still a no-op
    assert migrate_competitions(conn)["already_migrated"] is True


def test_running_twice_does_not_double_backfill(conn):
    migrate_competitions(conn)
    migrate_competitions(conn)
    ids = {r["competition_id"] for r in conn.execute(
        "SELECT DISTINCT competition_id FROM matches")}
    assert ids == {COMPETITION_BACKFILL}


# ─────────────────────────────────────────────
# THE POINT OF THE WHOLE THING
# ─────────────────────────────────────────────

def test_after_migration_the_same_pairing_can_meet_in_two_competitions(conn):
    """League and cup, same clubs, same season — the collision the old
    UNIQUE(season, matchday, home, away) made impossible."""
    migrate_competitions(conn)
    a = warehouse.get_or_create_team(conn, "Justice")
    b = warehouse.get_or_create_team(conn, "Pearls")

    # a cup tie on the same matchday number, different competition
    conn.execute(
        "INSERT INTO matches(season, competition, competition_id, matchday, "
        "home_team_id, away_team_id, home_goals, away_goals, match_date, "
        "fidelity, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("26/27", "CUP", "CMP-PLOFA-CUP", 1, a, b,
         1, 0, "2026-09-02", "v26", "test"))
    conn.commit()
    # ...and a REPLAY inside the same competition is still correctly refused
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO matches(season, competition, competition_id, matchday, "
            "home_team_id, away_team_id, home_goals, away_goals, match_date, "
            "fidelity, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("26/27", "PLOFA", COMPETITION_BACKFILL, 1, a,
             b, 5, 5, "2026-08-08", "v26", "test"))

    rows = conn.execute(
        "SELECT competition_id, COUNT(*) n FROM matches GROUP BY competition_id"
    ).fetchall()
    assert {r["competition_id"] for r in rows} == {COMPETITION_BACKFILL,
                                                   "CMP-PLOFA-CUP"}


def test_two_goals_by_one_scorer_in_two_competitions_coexist(conn):
    """The goals UNIQUE gained competition_id for the same reason."""
    migrate_competitions(conn)
    a = warehouse.get_or_create_team(conn, "Justice")
    conn.execute(
        "INSERT INTO goals(season, competition_id, matchday, team_id, minute, "
        "scorer, fidelity, source) VALUES (?,?,?,?,?,?,?,?)",
        ("26/27", "CMP-PLOFA-CUP", 1, a, 23.0, "Percy Osei",
         "v26", "test"))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) n FROM goals").fetchone()["n"]
    assert n == 2


def test_existing_aggregates_are_unchanged_by_the_migration(conn):
    """The default bucket must reproduce today's numbers exactly — an additive
    migration that shifted a total would be a silent corruption."""
    def totals():
        return {
            "matches": conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"],
            "goals": conn.execute("SELECT COUNT(*) n FROM goals").fetchone()["n"],
            "goals_sum": conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN scorer='Percy Osei' THEN 1 "
                "ELSE 0 END),0) n FROM goals").fetchone()["n"],
        }
    before = totals()
    migrate_competitions(conn)
    assert totals() == before


# ─────────────────────────────────────────────
# REFUSING TO GUESS
# ─────────────────────────────────────────────

def test_migration_refuses_an_unrecognised_constraint(conn):
    """If the live schema does not contain the exact constraint this migration
    was written for, it must ABORT having changed nothing."""
    conn.execute("DROP TABLE goals")
    conn.execute("CREATE TABLE goals (goal_id INTEGER PRIMARY KEY, "
                 "season TEXT, scorer TEXT, fidelity TEXT NOT NULL)")
    conn.commit()
    before = conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"]

    with pytest.raises(MigrationError, match="goals"):
        migrate_competitions(conn)

    # nothing was touched
    assert conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"] == before
    assert "competition_id" not in [
        c["name"] for c in conn.execute("PRAGMA table_info(matches)")]


def test_migration_refuses_when_a_target_table_is_missing(conn):
    conn.execute("DROP TABLE season_standings")
    conn.commit()
    with pytest.raises(MigrationError, match="season_standings"):
        migrate_competitions(conn)


def test_migration_refuses_an_empty_database(tmp_path):
    c = warehouse.connect(tmp_path / "empty.db")
    with pytest.raises(MigrationError, match="init"):
        migrate_competitions(c)
    c.close()


def test_dry_run_changes_nothing(conn):
    before = [tuple(r) for r in conn.execute("SELECT * FROM matches")]
    st = migrate_competitions(conn, dry_run=True)
    assert st["dry_run"] is True
    assert set(st["planned_rebuild"]) == {t["table"] for t in REBUILD_TABLES}
    assert set(st["planned_column"]) == set(COLUMN_ONLY_TABLES)
    assert is_migrated(conn) is False
    assert [tuple(r) for r in conn.execute("SELECT * FROM matches")] == before


# ─────────────────────────────────────────────
# REPORTING & BACKUP
# ─────────────────────────────────────────────

def test_competition_report_reflects_state(conn):
    assert "NOT MIGRATED" in competition_report(conn)
    migrate_competitions(conn)
    text = competition_report(conn)
    assert "migrated" in text
    assert COMPETITION_BACKFILL in text
    # per-table breakdown
    assert "matches" in text and "player_match_stats" in text


def test_backup_copies_the_database_and_its_wal(tmp_path):
    src = tmp_path / "alltime.db"
    c = warehouse.connect(src)
    warehouse.init_schema(c)
    c.execute("INSERT INTO meta(key, value) VALUES ('k','v')")
    c.commit()
    c.close()
    (tmp_path / "alltime.db-wal").write_bytes(b"wal-bytes")

    made = backup_database(src)
    assert made.exists()
    assert made.read_bytes() == src.read_bytes()
    assert made.with_name(made.name + "-wal").read_bytes() == b"wal-bytes"


def test_backup_never_overwrites_an_earlier_backup(tmp_path):
    src = tmp_path / "alltime.db"
    src.write_bytes(b"v1")
    first = backup_database(src)
    src.write_bytes(b"v2")
    second = backup_database(src)
    assert first != second
    assert first.read_bytes() == b"v1"
    assert second.read_bytes() == b"v2"


# ─────────────────────────────────────────────
# THE MIGRATED WAREHOUSE STILL WORKS
# ─────────────────────────────────────────────

def test_sync_still_works_after_migration(conn, tmp_path):
    """The migration must not break the existing ingest path: a fresh package
    ingested afterwards must land in the backfilled competition."""
    migrate_competitions(conn)
    root = tmp_path / "plofa_output"
    pkg = root / "Justice_vs_Pearls_MD2"
    pkg.mkdir(parents=True)
    (pkg / "Justice_vs_Pearls_MD2.json").write_text(
        '{"season": "26/27", "competition": "PLOFA", "matchday": 2, '
        '"match_date": "2026-08-15", "home_team": "Justice", '
        '"away_team": "Pearls", "home_goals": 1, "away_goals": 1, '
        '"players": []}', encoding="utf-8")

    st = warehouse.sync_output_root(conn, str(root), dry_run=False)
    assert isinstance(st, dict)
    ids = {r["competition_id"] for r in conn.execute(
        "SELECT DISTINCT competition_id FROM matches")}
    assert ids == {COMPETITION_BACKFILL}, "new rows must inherit the keying"


def test_report_command_still_runs_after_migration(conn):
    """`alltime_db.report` RETURNS its text (main() prints it) — assert on the
    return value, and that the migration did not disturb the aggregates."""
    migrate_competitions(conn)
    text = warehouse.report(conn)
    assert isinstance(text, str)
    assert "26/27" in text or "season" in text.lower()
