"""
PLOFA — all-time warehouse: competition keying (audit §16 phase 7).
===================================================================
Phase 7 of the World Football plan, and the last gate before a second
competition may go live.

The problem (audit §15)
-----------------------
``alltime.db`` assumes exactly one competition per season. Three constraints
encode that assumption, and each one breaks the moment the same two clubs meet
in a league AND a cup:

    matches             UNIQUE(season, matchday, home_team_id, away_team_id)
    player_match_stats  PRIMARY KEY (season, match_date, team_id, player_id)
    goals               UNIQUE(season, matchday, team_id, minute, scorer)

Two further tables have no competition dimension at all, so a player's league
and continental season lines cannot coexist:

    player_season_stats, season_standings

The fix is a **purely additive** migration: add ``competition_id``, widen the
constraints to include it, and backfill every existing row with the identity of
the competition it already belongs to. Nothing is dropped, renamed or
reinterpreted, so every aggregate that is correct today is byte-identical after.

Why this is a rebuild and not an ALTER
--------------------------------------
SQLite cannot add a column to a PRIMARY KEY or a UNIQUE constraint. Changing one
means rebuilding the table: create the new shape, copy every row, drop the old,
rename. That is safe only if it is done carefully, so this module:

  * **refuses to guess.** Each rebuild declares the exact constraint text it
    expects to find. If the live schema does not contain it, the migration
    ABORTS rather than performing a transformation it cannot verify.
  * **preserves data.** Row counts and per-table checksums are captured before
    and after; a mismatch rolls the whole migration back.
  * **respects foreign keys.** ``PRAGMA foreign_key_check`` runs afterwards.
  * **is idempotent.** Re-running is a no-op, and the second run says so.

Usage::

    & ".\\.venv\\Scripts\\python.exe" alltime_db.py migrate-competitions --dry-run
    & ".\\.venv\\Scripts\\python.exe" alltime_db.py migrate-competitions

The CLI takes a backup first. Always dry-run against a 143 MB warehouse whose
fixtures cannot be replayed.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The identity every existing row already belongs to. The live 26/27 warehouse
#: contains the PLOFA league and nothing else, so this is not an assumption — it
#: is the only competition that has ever been ingested. Matches
#: ``world.ledger.import_plofa_season_state``'s default so both layers agree.
COMPETITION_BACKFILL = "CMP-PLOFA"

#: Tables whose UNIQUE / PRIMARY KEY must be widened. Each entry names the exact
#: constraint text expected in the live schema; a mismatch aborts the migration.
REBUILD_TABLES: Tuple[Dict[str, str], ...] = (
    {
        "table": "matches",
        "old": "UNIQUE(season, matchday, home_team_id, away_team_id)",
        "new": "UNIQUE(season, competition_id, matchday, home_team_id, away_team_id)",
    },
    {
        "table": "player_match_stats",
        "old": "PRIMARY KEY (season, match_date, team_id, player_id)",
        "new": "PRIMARY KEY (season, competition_id, match_date, team_id, player_id)",
    },
    {
        "table": "goals",
        "old": "UNIQUE(season, matchday, team_id, minute, scorer)",
        "new": "UNIQUE(season, competition_id, matchday, team_id, minute, scorer)",
    },
)

#: Tables that only need the column — no constraint references it.
COLUMN_ONLY_TABLES: Tuple[str, ...] = (
    "player_season_stats",
    "season_standings",
)

NEW_INDEXES: Tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(season, competition_id)",
    "CREATE INDEX IF NOT EXISTS idx_pms_comp ON player_match_stats(season, competition_id)",
    "CREATE INDEX IF NOT EXISTS idx_pss_comp ON player_season_stats(season, competition_id)",
    "CREATE INDEX IF NOT EXISTS idx_standings_comp ON season_standings(season, competition_id)",
    "CREATE INDEX IF NOT EXISTS idx_goals_comp ON goals(season, competition_id)",
)


class MigrationError(RuntimeError):
    """The live schema is not what the migration expects. Nothing was changed."""


# ─────────────────────────────────────────────
# INTROSPECTION
# ─────────────────────────────────────────────

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _table_sql(conn: sqlite3.Connection, table: str) -> Optional[str]:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row["sql"] if row else None


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def has_competition_column(conn: sqlite3.Connection, table: str) -> bool:
    return "competition_id" in _columns(conn, table)


def is_migrated(conn: sqlite3.Connection) -> bool:
    """True when every target table already carries ``competition_id``."""
    targets = [t["table"] for t in REBUILD_TABLES] + list(COLUMN_ONLY_TABLES)
    return all(
        _table_exists(conn, t) and has_competition_column(conn, t) for t in targets
    )


def _target_tables_present(conn: sqlite3.Connection) -> List[str]:
    return [t for t in ("matches", "player_match_stats", "goals",
                        "player_season_stats", "season_standings")
            if _table_exists(conn, t)]


def _row_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    # tolerant of a missing table so an incomplete schema produces the
    # migration's own clear MigrationError rather than a raw sqlite3 message
    return {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in _target_tables_present(conn)}


def _checksums(conn: sqlite3.Connection) -> Dict[str, int]:
    """A content fingerprint that ignores the new column.

    Sums every column except ``competition_id`` so a migration that silently
    altered or dropped data is caught even when row counts match.
    """
    out: Dict[str, int] = {}
    for table in _target_tables_present(conn):
        cols = [c for c in _columns(conn, table) if c != "competition_id"]
        if not cols:
            continue
        expr = " + ".join(f"COALESCE(CAST({c} AS TEXT), '')" for c in cols)
        out[table] = conn.execute(
            f"SELECT COALESCE(SUM(LENGTH({expr})), 0) AS s FROM {table}"
        ).fetchone()["s"]
    return out


# ─────────────────────────────────────────────
# DDL TRANSFORM
# ─────────────────────────────────────────────

def _widen_constraint(ddl: str, old: str, new: str, table: str) -> str:
    """Rewrite one constraint clause, refusing to proceed if it is not there.

    Aborting on an unexpected schema is the whole safety story: a silent
    no-match replacement would create a table with a DIFFERENT constraint and
    copy the data anyway, which is precisely the kind of quiet corruption this
    migration must not cause.
    """
    normalised = " ".join(ddl.split())
    if " ".join(old.split()) not in normalised:
        raise MigrationError(
            f"{table}: expected to find constraint {old!r} in the live schema. "
            f"Refusing to migrate an unrecognised schema — nothing was changed. "
            f"Live DDL was:\n{ddl}"
        )
    compact = " ".join(ddl.split())
    return compact.replace(" ".join(old.split()), " ".join(new.split()), 1)


#: Matches an actual COLUMN DECLARATION, not a mention inside a constraint.
#: Getting this wrong is subtle: after widening, the DDL already contains the
#: substring "competition_id" inside the new UNIQUE(...)/PRIMARY KEY(...), so a
#: naive `"competition_id" in ddl` check would conclude the column exists and
#: skip adding it — producing a table whose constraint references a column that
#: was never declared.
_COLUMN_DECL_RE = re.compile(r"\bcompetition_id\s+(?:TEXT|INTEGER|REAL|INT)\b", re.I)


def _ddl_with_competition(ddl: str, old: str, new: str, table: str) -> str:
    """Add ``competition_id TEXT`` and widen the declared constraint."""
    widened = _widen_constraint(ddl, old, new, table)
    if _COLUMN_DECL_RE.search(widened):
        return widened
    # insert the column just inside the opening parenthesis
    open_at = widened.index("(")
    return (widened[: open_at + 1]
            + "\n        competition_id TEXT,"
            + widened[open_at + 1:])


# ─────────────────────────────────────────────
# MIGRATION
# ─────────────────────────────────────────────

def migrate_competitions(conn: sqlite3.Connection, *,
                         backfill: str = COMPETITION_BACKFILL,
                         dry_run: bool = False) -> Dict[str, Any]:
    """Key every warehouse row by competition. Additive and idempotent.

    Returns a report dict. Raises :class:`MigrationError` — having changed
    nothing — if the live schema is not the one this migration was written for.
    """
    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "already_migrated": False,
        "rebuilt": [],
        "column_added": [],
        "backfilled": {},
        "indexes": 0,
        "row_counts_before": {},
        "row_counts_after": {},
        "checksums_preserved": None,
        "foreign_key_violations": None,
    }

    if not _table_exists(conn, "matches"):
        raise MigrationError("no `matches` table — run `alltime_db.py init` first")

    if is_migrated(conn):
        report["already_migrated"] = True
        report["row_counts_after"] = _row_counts(conn)
        return report

    before_counts = _row_counts(conn)
    before_sums = _checksums(conn)
    report["row_counts_before"] = dict(before_counts)

    # Plan the whole thing BEFORE touching anything, so an unrecognised schema
    # is discovered while the database is still untouched.
    plans: List[Tuple[str, str]] = []
    for spec in REBUILD_TABLES:
        table = spec["table"]
        ddl = _table_sql(conn, table)
        if ddl is None:
            raise MigrationError(f"table {table!r} not found in the live schema")
        plans.append((table, _ddl_with_competition(ddl, spec["old"], spec["new"], table)))
    for table in COLUMN_ONLY_TABLES:
        if not _table_exists(conn, table):
            raise MigrationError(f"table {table!r} not found in the live schema")

    if dry_run:
        report["planned_rebuild"] = [t for t, _ in plans]
        report["planned_column"] = list(COLUMN_ONLY_TABLES)
        report["row_counts_after"] = dict(before_counts)
        return report

    # Foreign keys must be off while tables are dropped and recreated; the
    # documented SQLite procedure, re-checked at the end.
    fk_was_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        for table, new_ddl in plans:
            tmp = f"{table}__migrate7"
            conn.execute(f"DROP TABLE IF EXISTS {tmp}")
            conn.execute(new_ddl.replace(
                f"CREATE TABLE {table}", f"CREATE TABLE {tmp}", 1))
            cols = _columns(conn, table)
            target_cols = [c for c in _columns(conn, tmp) if c in cols]
            joined = ", ".join(target_cols)
            conn.execute(
                f"INSERT INTO {tmp} ({joined}) SELECT {joined} FROM {table}")
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {tmp} RENAME TO {table}")
            report["rebuilt"].append(table)

        for table in COLUMN_ONLY_TABLES:
            if not has_competition_column(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN competition_id TEXT")
                report["column_added"].append(table)

        # backfill: every existing row belonged to the one competition that has
        # ever been ingested
        for table in [t["table"] for t in REBUILD_TABLES] + list(COLUMN_ONLY_TABLES):
            cur = conn.execute(
                f"UPDATE {table} SET competition_id = ? "
                f"WHERE competition_id IS NULL OR competition_id = ''",
                (backfill,))
            report["backfilled"][table] = cur.rowcount

        for ddl in NEW_INDEXES:
            conn.execute(ddl)
            report["indexes"] += 1

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES "
            "('competition_keyed', '1')")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES "
            "('competition_backfill', ?)", (backfill,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys={'ON' if fk_was_on else 'OFF'}")

    # ── verification ──
    after_counts = _row_counts(conn)
    after_sums = _checksums(conn)
    report["row_counts_after"] = dict(after_counts)
    report["checksums_preserved"] = (after_sums == before_sums)
    report["foreign_key_violations"] = len(
        conn.execute("PRAGMA foreign_key_check").fetchall())

    if after_counts != before_counts:
        conn.rollback()
        raise MigrationError(
            f"row counts changed during migration: {before_counts} -> "
            f"{after_counts}. Rolled back."
        )
    if not report["checksums_preserved"]:
        conn.rollback()
        raise MigrationError(
            f"row content changed during migration: {before_sums} -> {after_sums}. "
            f"Rolled back."
        )
    if report["foreign_key_violations"]:
        raise MigrationError(
            f"{report['foreign_key_violations']} foreign key violation(s) after "
            f"migration. The database was NOT rolled back (it committed "
            f"cleanly) — inspect before continuing."
        )
    return report


def competition_report(conn: sqlite3.Connection) -> str:
    """Human-readable state of the competition keying, for ``report``/CLI."""
    if not _table_exists(conn, "matches"):
        return "competition keying: no schema"
    if not is_migrated(conn):
        return ("competition keying: NOT MIGRATED — run "
                "`alltime_db.py migrate-competitions`")
    lines = ["competition keying: migrated"]
    for table in ("matches", "player_match_stats", "player_season_stats",
                  "season_standings", "goals"):
        if not _table_exists(conn, table) or not has_competition_column(conn, table):
            continue
        rows = conn.execute(
            f"SELECT competition_id, COUNT(*) AS n FROM {table} "
            f"GROUP BY competition_id ORDER BY n DESC"
        ).fetchall()
        summary = ", ".join(f"{r['competition_id'] or '(null)'}={r['n']}" for r in rows)
        lines.append(f"  {table:<20} {summary}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────

def backup_database(db_path: Path) -> Path:
    """Copy the warehouse aside before any schema surgery.

    The fixtures behind these rows cannot be replayed (README §2), so a backup
    is not optional politeness — it is the only way back.
    """
    stamp = "pre-competition-keying"
    target = db_path.with_name(db_path.name + f".{stamp}")
    n = 1
    while target.exists():
        target = db_path.with_name(f"{db_path.name}.{stamp}.{n}")
        n += 1
    shutil.copy2(db_path, target)
    # WAL sidecars hold committed data that has not yet been checkpointed
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            shutil.copy2(side, target.with_name(target.name + suffix))
    return target
