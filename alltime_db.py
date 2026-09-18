"""alltime_db -- SQLite stats warehouse for the whole PLOFA federation (like Opta).

Central, queryable, versioned store of every match, every player's per-match
line, every goal and every league table - kept forever, with a FIDELITY tag
on every fact row so we never confuse data sources of differing truthfulness.

Fidelity model
--------------
* ``legacy`` : imported from ``PLOFA-ALL-TIME.xlsx`` (seasons 24/25 & 25/26).
  Produced by an older exporter; volumetric categories are reliable but the
  granularity is coarser (e.g. the 24/25 player-match rows lost their
  matchday ids). Kept as-is, tagged, never silently mixed with the v26 data.
* ``v26``    : written live by the current exporter pipeline (season 26/27)
  from ``plofa_output/<Home_vs_Away_MD##>/<...>.json``. These rows are
  event-derived at per-match + per-player granularity.

Tables
------
* seasons / teams / players          - dimensions
* matches                            - one row per fixture (score, date, venue)
* team_match_stats                   - per team per match aggregates
* player_match_stats                 - per player per match line (~110 cols)
* player_season_stats                - all-time player aggregates (legacy)
* season_standings                   - league tables (legacy, raw preserved)
* goals                              - goal-level detail (v26; legacy raw in
                                       legacy_detailed_goals_raw)
* legacy_detailed_goals_raw          - raw preservation of the legacy goal log
* meta                               - schema version + import stamps

CLI
---
python alltime_db.py init [--db path]
python alltime_db.py import-legacy [--db path] [--xlsx path]
python alltime_db.py sync [--db path] [--root plofa_output] [--dry-run]
python alltime_db.py report [--db path]
python alltime_db.py alias add <alias> <canonical> [--apply] [--note text]
python alltime_db.py alias scan
python alltime_db.py alias list
python alltime_db.py alias apply --all

Run `sync` after every matchday you play to append new 26/27 matches to the
warehouse (idempotent). `alias` handles player renames / spelling variants:
the same person's different names share one canonical identity forever, and
reports aggregate across them.

The canonical DB lives at <repo>/alltime.db (gitignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "alltime.db"
LEGACY_XLSX = r"D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM\PLOFA-ALL-TIME.xlsx"
PLOFA_OUTPUT = Path(__file__).resolve().parent / "plofa_output"

SCHEMA_VERSION = 2

SEASONS = {
    "24/25": ("2024-08-01", "2025-06-30", "legacy",
              "imported 2026-09-18 from PLOFA-ALL-TIME.xlsx; UNREAL player-match "
              "rows lost most matchday ids, MATCH RESULTS authoritative for fixtures"),
    "25/26": ("2025-08-01", "2026-06-30", "legacy",
              "imported 2026-09-18 from PLOFA-ALL-TIME.xlsx; clean per-match data"),
    "26/27": ("2026-08-01", "2027-06-30", "v26",
              "live season; written by ingest_match_package from plofa_output exporter JSON"),
}

# Canonical per-player stat columns shared by player_match_stats and
# player_season_stats.  Names match the 26/27 exporter player-dict keys so the
# v26 ingester maps 1:1; the legacy importer maps its own labels onto these.
PLAYER_STAT_COLS = [
    ("goals", "INTEGER"), ("assists", "INTEGER"), ("own_goals", "INTEGER"),
    ("open_play_goals", "INTEGER"), ("headed_goals", "INTEGER"),
    ("pen_goals", "INTEGER"), ("pen_missed", "INTEGER"),
    ("shots_on_target", "INTEGER"), ("shots_off_target", "INTEGER"),
    ("shots_blocked_att", "INTEGER"), ("hit_woodwork", "INTEGER"),
    ("shots_inside_box", "INTEGER"), ("shots_outside_box", "INTEGER"),
    ("total_shots", "INTEGER"),
    ("big_chances_scored", "INTEGER"), ("big_chances_missed", "INTEGER"),
    ("big_chances_received", "INTEGER"), ("big_chances_created", "INTEGER"),
    ("xg", "REAL"), ("xa", "REAL"), ("npxg", "REAL"),
    ("passes_attempted", "INTEGER"), ("passes_completed", "INTEGER"),
    ("short_passes_att", "INTEGER"), ("short_passes_comp", "INTEGER"),
    ("long_passes_att", "INTEGER"), ("long_passes_comp", "INTEGER"),
    ("progressive_passes", "INTEGER"),
    ("passes_own_third", "INTEGER"), ("passes_mid_third", "INTEGER"),
    ("passes_final_third", "INTEGER"), ("passes_opp_box", "INTEGER"),
    ("shot_assists", "INTEGER"),
    ("through_balls_att", "INTEGER"), ("through_balls_comp", "INTEGER"),
    ("switches_of_play", "INTEGER"), ("passes_under_pressure", "INTEGER"),
    ("forward_passes", "INTEGER"), ("backward_passes", "INTEGER"),
    ("sideways_passes", "INTEGER"), ("line_breaking_passes", "INTEGER"),
    ("crosses_att", "INTEGER"), ("crosses_comp", "INTEGER"),
    ("crosses_open_play_att", "INTEGER"), ("crosses_open_play_comp", "INTEGER"),
    ("crosses_corners_att", "INTEGER"), ("crosses_corners_comp", "INTEGER"),
    ("carries", "INTEGER"), ("total_carries", "INTEGER"),
    ("progressive_carries", "INTEGER"), ("final_third_carries", "INTEGER"),
    ("carries_opp_box", "INTEGER"),
    ("carry_distance", "REAL"), ("progressive_carry_distance", "REAL"),
    ("longest_progressive_carry", "REAL"),
    ("runs_without_ball", "INTEGER"),
    ("dribbles_att", "INTEGER"), ("dribbles_comp", "INTEGER"),
    ("dribbles_to_box", "INTEGER"), ("dribble_distance", "REAL"),
    ("chances_created", "INTEGER"), ("open_play_cc", "INTEGER"),
    ("setpiece_cc", "INTEGER"),
    ("tackles_att", "INTEGER"), ("tackles_won", "INTEGER"),
    ("interceptions", "INTEGER"), ("clearances", "INTEGER"),
    ("blocks", "INTEGER"), ("recoveries", "INTEGER"),
    ("ball_recoveries", "INTEGER"),
    ("pressures", "INTEGER"), ("press_success", "INTEGER"),
    ("aerial_duels_att", "INTEGER"), ("aerial_duels_won", "INTEGER"),
    ("ground_duels_att", "INTEGER"), ("ground_duels_won", "INTEGER"),
    ("dribbled_past", "INTEGER"), ("last_man_tackles", "INTEGER"),
    ("saves", "INTEGER"), ("goals_conceded", "INTEGER"),
    ("high_claims", "INTEGER"), ("punches", "INTEGER"),
    ("goalline_saves", "INTEGER"), ("clean_sheet", "INTEGER"),
    ("save_pct", "REAL"),
    ("fouls_committed", "INTEGER"), ("fouls_won", "INTEGER"),
    ("yellow_cards", "INTEGER"), ("red_cards", "INTEGER"),
    ("offsides", "INTEGER"),
    ("sprints", "INTEGER"), ("high_speed_sprints", "INTEGER"),
    ("distance_covered", "REAL"), ("top_speed", "REAL"),
    ("touches", "INTEGER"), ("touches_own_third", "INTEGER"),
    ("touches_mid_third", "INTEGER"), ("touches_final_third", "INTEGER"),
    ("touches_opp_box", "INTEGER"),
    ("turnovers", "INTEGER"), ("bad_touches", "INTEGER"),
    ("dispossessed", "INTEGER"),
    ("possession_won", "INTEGER"), ("possession_lost", "INTEGER"),
    ("sca", "INTEGER"), ("gca", "INTEGER"),
    ("packing_passes", "INTEGER"), ("zone14_entries", "INTEGER"),
    ("deep_completions", "INTEGER"),
    ("xT", "REAL"), ("gpa", "REAL"), ("pva", "REAL"), ("epa", "REAL"),
    ("pass_accuracy", "REAL"), ("short_pass_acc", "REAL"),
    ("long_pass_acc", "REAL"),
    ("dribble_success_pct", "REAL"), ("tackle_success_pct", "REAL"),
    ("aerial_success_pct", "REAL"),
    ("cross_acc", "REAL"), ("shot_conversion", "REAL"),
    ("rating", "REAL"),
]

PLAYER_STAT_DDL = ", ".join(
    f'"{name}" {typ}' for name, typ in PLAYER_STAT_COLS
)

TEAM_STAT_COLS = [
    ("goals", "INTEGER"), ("xg", "REAL"), ("xa", "REAL"),
    ("shots", "INTEGER"), ("shots_on_target", "INTEGER"),
    ("shots_off_target", "INTEGER"), ("shots_blocked", "INTEGER"),
    ("shots_inside_box", "INTEGER"), ("shots_outside_box", "INTEGER"),
    ("corners", "INTEGER"), ("offsides", "INTEGER"),
    ("passes_attempted", "INTEGER"), ("passes_completed", "INTEGER"),
    ("pass_accuracy", "REAL"),
    ("tackles", "INTEGER"), ("interceptions", "INTEGER"),
    ("clearances", "INTEGER"), ("blocks", "INTEGER"),
    ("recoveries", "INTEGER"), ("pressures", "INTEGER"),
    ("fouls", "INTEGER"), ("yellow_cards", "INTEGER"),
    ("red_cards", "INTEGER"),
]
TEAM_STAT_DDL = ", ".join(f'"{name}" {typ}' for name, typ in TEAM_STAT_COLS)

# Legacy sheet labels (normalised: non-alphanumeric stripped, upper-cased) ->
# canonical stat column.
_LEGACY_LABEL_MAP = {
    "GOALS": "goals", "ASSISTS": "assists",
    "XG": "xg", "XA": "xa",
    "MINUTESPLAYED": "minutes_played",
    "SHOTSONTARGET": "shots_on_target", "SHOTSOFFTARGET": "shots_off_target",
    "SHOTSBLOCKED": "shots_blocked_att",
    "SHOTSINSIDEBOX": "shots_inside_box", "SHOTSOUTSIDEBOX": "shots_outside_box",
    "TOTALSHOTS": "total_shots",
    "BIGCHANCESSCORED": "big_chances_scored",
    "BIGCHANCESMISSED": "big_chances_missed",
    "BIGCHANCESRECEIVED": "big_chances_received",
    "BIGCC": "big_chances_created",
    "KEYPASSES": "shot_assists",
    "THROUGHBALLSATT": "through_balls_att",
    "THROUGHBALLSCOMP": "through_balls_comp",
    "SWITCHESOFPLAY": "switches_of_play",
    "TOTALPASSESATT": "passes_attempted", "TOTALPASSESCOMP": "passes_completed",
    "PASSACCURACY": "pass_accuracy", "PASSACCURACY": "pass_accuracy",
    "OPENPLAYCROSSESATT": "crosses_open_play_att",
    "OPENPLAYCROSSESCOMP": "crosses_open_play_comp",
    "TOTALCROSSESATT": "crosses_att", "TOTALCROSSESCOMP": "crosses_comp",
    "PROGRESSIVEPASSES": "progressive_passes",
    "DRIBBLESATTEMPTED": "dribbles_att",
    "DRIBBLESCOMPLETED": "dribbles_comp",
    "DRIBBLESUCCESS": "dribble_success_pct",
    "PROGRESSIVECARRIES": "progressive_carries",
    "TOTALCARRIES": "total_carries",
    "TACKLESATTEMPTED": "tackles_att", "TACKLESWON": "tackles_won",
    "TACKLESUCCESS": "tackle_success_pct",
    "INTERCEPTIONS": "interceptions",
    "PRESSURES": "pressures", "PRESSUREREGAINS": "press_success",
    "TOTALBALLRECOVERIES": "ball_recoveries",
    "POSSESSIONWON": "possession_won", "POSSESSIONLOST": "possession_lost",
    "TOUCHES": "touches",
    "FOULSCOMMITTED": "fouls_committed", "FOULSWON": "fouls_won",
    "YELLOWCARDS": "yellow_cards", "REDCARDS": "red_cards",
    "OFFSIDES": "offsides",
    "DISTANCECOVEREDKM": "distance_covered",
    "SAVES": "saves", "GOALSCONCEDED": "goals_conceded",
    "HIGHCLAIMS": "high_claims", "GOALLINESAVES": "goalline_saves",
    "RATING": "rating",
}

# Extra PLAYER STATS sheet emoji column -> canonical.
_EMOJI_LABEL_MAP = {
    "\u26bd": "goals",  # ball emoji column = goals
    "A": "assists",
}


def lnorm(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(label)).upper()


def _num(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _numi(v):
    f = _num(v)
    return None if f is None else int(f)


def _txt(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    return s


def _int_bool(v):
    if v is None:
        return None
    c = str(v).strip()
    if not c:
        return None
    return 1 if c.lower() in ("true", "1", "yes") else 0


def connect(db=None):
    path = Path(db) if db else DEFAULT_DB
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn):
    conn.executescript(f"""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    CREATE TABLE IF NOT EXISTS seasons (
        season TEXT PRIMARY KEY,
        start_date TEXT,
        end_date TEXT,
        fidelity TEXT NOT NULL,
        note TEXT
    );
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        team_id INTEGER REFERENCES teams(team_id),
        position TEXT,
        nationality TEXT,
        archetype TEXT,
        UNIQUE(name)
    );
    CREATE TABLE IF NOT EXISTS player_aliases (
        alias TEXT PRIMARY KEY,
        canonical_player_id INTEGER NOT NULL REFERENCES players(player_id),
        note TEXT,
        source TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_aliases_canon ON player_aliases
        (canonical_player_id);
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY,
        season TEXT,
        competition TEXT,
        matchday INTEGER,
        home_team_id INTEGER REFERENCES teams(team_id),
        away_team_id INTEGER REFERENCES teams(team_id),
        home_goals INTEGER,
        away_goals INTEGER,
        match_date TEXT,
        venue TEXT,
        attendance INTEGER,
        capacity INTEGER,
        ticket_price REAL,
        home_revenue REAL,
        added_time INTEGER,
        is_derby INTEGER,
        home_possession_pct REAL,
        away_possession_pct REAL,
        fidelity TEXT NOT NULL,
        source TEXT,
        raw_json TEXT,
        UNIQUE(season, matchday, home_team_id, away_team_id)
    );
    CREATE TABLE IF NOT EXISTS team_match_stats (
        match_id INTEGER REFERENCES matches(match_id),
        team_id INTEGER REFERENCES teams(team_id),
        side TEXT,
        {TEAM_STAT_DDL},
        fidelity TEXT NOT NULL,
        source TEXT,
        PRIMARY KEY (match_id, team_id)
    );
    CREATE TABLE IF NOT EXISTS player_match_stats (
        season TEXT,
        match_date TEXT,
        matchday INTEGER,
        match_id INTEGER REFERENCES matches(match_id),
        team_id INTEGER REFERENCES teams(team_id),
        player_id INTEGER REFERENCES players(player_id),
        player_name TEXT,
        position TEXT,
        archetype TEXT,
        age INTEGER,
        nationality TEXT,
        preferred_foot TEXT,
        minutes_played INTEGER,
        is_starter INTEGER,
        home_or_away TEXT,
        soul_archetype TEXT,
        {PLAYER_STAT_DDL},
        match_result TEXT,
        is_mvp INTEGER,
        stats_json TEXT,
        fidelity TEXT NOT NULL,
        source TEXT,
        PRIMARY KEY (season, match_date, team_id, player_id)
    );
    CREATE TABLE IF NOT EXISTS player_season_stats (
        season TEXT,
        team_id INTEGER REFERENCES teams(team_id),
        player_id INTEGER REFERENCES players(player_id),
        player_name TEXT,
        position TEXT,
        {PLAYER_STAT_DDL},
        stats_json TEXT,
        fidelity TEXT NOT NULL,
        source TEXT
    );
    CREATE TABLE IF NOT EXISTS season_standings (
        standings_id INTEGER PRIMARY KEY,
        season TEXT,
        scope TEXT,
        block INTEGER,
        rank INTEGER,
        team_id INTEGER REFERENCES teams(team_id),
        played INTEGER, won INTEGER, drawn INTEGER, lost INTEGER,
        gf INTEGER, ga INTEGER, gd INTEGER, points REAL,
        raw_json TEXT,
        fidelity TEXT NOT NULL,
        source TEXT
    );
    CREATE TABLE IF NOT EXISTS goals (
        goal_id INTEGER PRIMARY KEY,
        season TEXT,
        matchday INTEGER,
        match_id INTEGER REFERENCES matches(match_id),
        team_id INTEGER REFERENCES teams(team_id),
        opponent_id INTEGER REFERENCES teams(team_id),
        minute REAL,
        scorer TEXT,
        assist TEXT,
        situation TEXT,
        xg REAL,
        is_penalty INTEGER,
        is_own_goal INTEGER,
        fidelity TEXT NOT NULL,
        source TEXT,
        UNIQUE(season, matchday, team_id, minute, scorer)
    );
    CREATE TABLE IF NOT EXISTS legacy_detailed_goals_raw (
        row_id INTEGER PRIMARY KEY,
        season TEXT,
        raw_json TEXT,
        fidelity TEXT NOT NULL,
        source TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_pms_season ON player_match_stats(season);
    CREATE INDEX IF NOT EXISTS idx_pms_player ON player_match_stats(player_id);
    CREATE INDEX IF NOT EXISTS idx_goals_player ON goals(scorer);
    """)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    register_seasons(conn)
    conn.commit()


def register_seasons(conn):
    for season, (sd, ed, fid, note) in SEASONS.items():
        conn.execute(
            """INSERT OR REPLACE INTO seasons(season, start_date, end_date, fidelity, note)
               VALUES (?,?,?,?,?)""",
            (season, sd, ed, fid, note),
        )
    conn.commit()


def get_or_create_team(conn, name):
    name = _txt(name)
    if not name:
        return None
    row = conn.execute("SELECT team_id FROM teams WHERE name=?", (name,)).fetchone()
    if row:
        return row["team_id"]
    cur = conn.execute("INSERT INTO teams(name) VALUES (?)", (name,))
    return cur.lastrowid


def _resolve_alias(conn, name):
    """If `name` is registered as a player alias, return the canonical player_id,
    else None. Follows one level only (aliases always point at canonical ids)."""
    name = _txt(name)
    if not name:
        return None
    row = conn.execute(
        "SELECT canonical_player_id FROM player_aliases WHERE alias=?", (name,),
    ).fetchone()
    return row["canonical_player_id"] if row else None


def get_or_create_player(conn, name, team_id=None, position=None,
                         nationality=None, archetype=None):
    name = _txt(name)
    if not name:
        return None
    canon = _resolve_alias(conn, name)
    if canon is not None:
        conn.execute(
            """UPDATE players SET team_id=COALESCE(?, team_id),
               position=COALESCE(?, position), nationality=COALESCE(?, nationality),
               archetype=COALESCE(?, archetype) WHERE player_id=?""",
            (team_id, position, nationality, archetype, canon),
        )
        return canon
    row = conn.execute("SELECT player_id FROM players WHERE name=?", (name,)).fetchone()
    if row:
        pid = row["player_id"]
        conn.execute(
            """UPDATE players SET team_id=COALESCE(?, team_id),
               position=COALESCE(?, position), nationality=COALESCE(?, nationality),
               archetype=COALESCE(?, archetype) WHERE player_id=?""",
            (team_id, position, nationality, archetype, pid),
        )
        return pid
    cur = conn.execute(
        """INSERT INTO players(name, team_id, position, nationality, archetype)
           VALUES (?,?,?,?,?)""",
        (name, team_id, position, nationality, archetype),
    )
    return cur.lastrowid


def norm_name(name):
    """Lower-cased, accent-stripped, tokenised name for fuzzy identity matching."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", n)


def add_alias(conn, alias, canonical, note=None, source="manual"):
    """Register `alias` as another name of the same person (`canonical`).
    Returns the canonical player_id. Safe to call again (idempotent)."""
    alias = _txt(alias)
    canonical = _txt(canonical)
    if not alias or not canonical:
        raise ValueError("alias and canonical must both be non-empty")
    if alias.strip().lower() == canonical.strip().lower():
        # identical string spellings carry no alias information
        canon_id = get_or_create_player(conn, canonical)
        return canon_id
    canon_id = _resolve_alias(conn, canonical) or get_or_create_player(conn, canonical)
    conn.execute(
        """INSERT OR REPLACE INTO player_aliases(alias, canonical_player_id, note, source)
           VALUES (?,?,?,?)""",
        (alias, canon_id, note, source),
    )
    conn.commit()
    return canon_id


def apply_alias_fixup(conn):
    """Rewire historical stat rows from alias identities to their canonical
    player. Where both alias and canonical already have a row for the same
    match/aggregate key, keep the canonical row and drop the alias duplicate.
    Returns (moved, deleted). Idempotent."""
    moved = deleted = 0
    alias_rows = conn.execute(
        "SELECT alias, canonical_player_id FROM player_aliases",
    ).fetchall()
    for r in alias_rows:
        alias = r["alias"]
        canon_id = r["canonical_player_id"]
        prow = conn.execute(
            "SELECT player_id, name FROM players WHERE name=?", (alias,),
        ).fetchone()
        apid = prow["player_id"] if prow else None
        if apid is None or apid == canon_id:
            continue
        cname = conn.execute(
            "SELECT name FROM players WHERE player_id=?", (canon_id,),
        ).fetchone()["name"]
        for rw in conn.execute(
                """SELECT season, match_date, team_id FROM player_match_stats
                   WHERE player_id=?""", (apid,)).fetchall():
            if conn.execute(
                    """SELECT 1 FROM player_match_stats
                       WHERE season=? AND match_date=? AND team_id=? AND player_id=?""",
                    (rw["season"], rw["match_date"], rw["team_id"], canon_id)).fetchone():
                conn.execute(
                    """DELETE FROM player_match_stats
                       WHERE season=? AND match_date=? AND team_id=? AND player_id=?""",
                    (rw["season"], rw["match_date"], rw["team_id"], apid))
                deleted += 1
            else:
                conn.execute(
                    """UPDATE player_match_stats SET player_id=?, player_name=?
                       WHERE season=? AND match_date=? AND team_id=? AND player_id=?""",
                    (canon_id, cname, rw["season"], rw["match_date"],
                     rw["team_id"], apid))
                moved += 1
        for rw in conn.execute(
                "SELECT season, team_id FROM player_season_stats WHERE player_id=?",
                (apid,)).fetchall():
            if conn.execute(
                    """SELECT 1 FROM player_season_stats
                       WHERE season=? AND team_id=? AND player_id=?""",
                    (rw["season"], rw["team_id"], canon_id)).fetchone():
                conn.execute(
                    """DELETE FROM player_season_stats
                       WHERE season=? AND team_id=? AND player_id=?""",
                    (rw["season"], rw["team_id"], apid))
                deleted += 1
            else:
                conn.execute(
                    """UPDATE player_season_stats SET player_id=?, player_name=?
                       WHERE season=? AND team_id=? AND player_id=?""",
                    (canon_id, cname, rw["season"], rw["team_id"], apid))
                moved += 1
        # tidy: drop the now-redundant alias identity row in players
        rem = conn.execute(
            """SELECT 1 FROM player_match_stats WHERE player_id=?
               UNION ALL SELECT 1 FROM player_season_stats WHERE player_id=?""",
            (apid, apid)).fetchone()
        if not rem:
            conn.execute("DELETE FROM players WHERE player_id=?", (apid,))
        conn.execute(
            "UPDATE goals SET scorer=? WHERE scorer=?", (cname, alias))
    conn.commit()
    return moved, deleted


_career_facts_cache = None


def career_facts(conn, player_id, name):
    """Per-identity facts used by alias_scan: seasons, teams, positions and the
    number of matched games, drawn from both stat tables."""
    global _career_facts_cache
    if _career_facts_cache is None:
        cache = {}
        for r in conn.execute(
                """SELECT pms.player_id, pms.season, t.name AS team, pms.position
                   FROM player_match_stats pms
                   LEFT JOIN teams t ON t.team_id = pms.team_id"""):
            facts = cache.setdefault(
                r["player_id"],
                {"seasons": set(), "teams": set(), "positions": set(), "games": 0})
            team = _txt(r["team"])
            pos = _txt(r["position"])
            if r["season"]:
                facts["seasons"].add(r["season"])
            if team and team != "0":
                facts["teams"].add(team)
            if pos and pos != "0":
                facts["positions"].add(pos)
            facts["games"] += 1
        for r in conn.execute(
                """SELECT pss.player_id, pss.season, t.name AS team, pss.position
                   FROM player_season_stats pss
                   LEFT JOIN teams t ON t.team_id = pss.team_id"""):
            facts = cache.setdefault(
                r["player_id"],
                {"seasons": set(), "teams": set(), "positions": set(), "games": 0})
            team = _txt(r["team"])
            pos = _txt(r["position"])
            if r["season"]:
                facts["seasons"].add(r["season"])
            if team and team != "0":
                facts["teams"].add(team)
            if pos and pos != "0":
                facts["positions"].add(pos)
        _career_facts_cache = cache
    return _career_facts_cache.get(player_id, {"seasons": set(), "teams": set(),
                                              "positions": set(), "games": 0})


def alias_scan(conn, min_score=0):
    """Propose same-person name pairs. Heuristic: identical surname token with a
    token-subset or same-token-set name (accent/spelling variants and first-
    name extensions), sharing evidence (same club across seasons and/or same
    position). Returns candidates ranked by strength. NEVER auto-applies."""
    rows = conn.execute(
        """SELECT p.player_id, p.name FROM players p
           LEFT JOIN player_aliases a ON a.alias = p.name
           WHERE a.alias IS NULL""").fetchall()
    info = [(_r["player_id"], _r["name"]) for _r in rows]
    info = [(pid, name, career_facts(conn, pid, name)) for pid, name in info]
    candidates = []
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            pid_a, name_a, fa = info[i]
            pid_b, name_b, fb = info[j]
            ta, tb = norm_name(name_a), norm_name(name_b)
            if not ta or not tb or ta[-1] != tb[-1]:
                continue
            sa, sb = set(ta), set(tb)
            if sa == sb:
                kind = "spelling-variant"
            elif sa <= sb or sb <= sa:
                kind = "name-extension"
            else:
                continue
            shared_teams = fa["teams"] & fb["teams"]
            shared_pos = fa["positions"] & fb["positions"]
            overlap = fa["seasons"] & fb["seasons"]
            if kind == "name-extension" and overlap and not shared_teams:
                continue  # two live names in the same season at different clubs
            if not shared_teams and not shared_pos:
                continue
            score = (10 + 2 * len(shared_teams) if shared_teams else 2) \
                + (2 if shared_pos else 0)
            if score < min_score:
                continue
            candidates.append({
                "score": score, "kind": kind,
                "alias": name_a if sa <= sb else name_b,
                "canonical": name_b if sa <= sb else name_a,
                "seasons_a": sorted(fa["seasons"]), "seasons_b": sorted(fb["seasons"]),
                "teams_a": sorted(fa["teams"]), "teams_b": sorted(fb["teams"]),
                "shared_teams": sorted(shared_teams),
                "positions": sorted(shared_pos),
                "games_a": fa["games"], "games_b": fb["games"],
            })
    candidates.sort(key=lambda c: (-c["score"], c["alias"]))
    return candidates


def list_aliases(conn):
    return conn.execute(
        """SELECT a.alias, p.name AS canonical, a.note
           FROM player_aliases a JOIN players p ON p.player_id = a.canonical_player_id
           ORDER BY canonical, alias""").fetchall()


def _score_goals(score_str):
    m = re.search(r"(\d+)\s*[-:\u2013\u2014]\s*(\d+)", str(score_str or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _player_row_values(player_dict, season, match_date, matchday, match_id,
                       team_id, fidelity, source):
    """Map a 26/27 exporter player dict (canonical snake_case keys) onto the
    player_match_stats row."""
    vals = {
        "season": season, "match_date": match_date, "matchday": matchday,
        "match_id": match_id, "team_id": team_id,
        "player_id": player_dict.get("player"),
        "player_name": _txt(player_dict.get("player")),
        "position": _txt(player_dict.get("position")),
        "archetype": _txt(player_dict.get("archetype")),
        "age": _numi(player_dict.get("age")),
        "nationality": _txt(player_dict.get("nationality")),
        "preferred_foot": _txt(player_dict.get("preferred_foot")),
        "minutes_played": _numi(player_dict.get("minutes_played")),
        "is_starter": _int_bool(player_dict.get("is_starter")),
        "home_or_away": _txt(player_dict.get("home_or_away")),
        "soul_archetype": _txt(player_dict.get("soul_archetype")),
        "match_result": _txt(player_dict.get("match_result")),
        "is_mvp": _int_bool(player_dict.get("is_mvp")),
        "stats_json": json.dumps(player_dict, default=str),
        "fidelity": fidelity,
        "source": source,
    }
    for name, _typ in PLAYER_STAT_COLS:
        vals[name] = _num(player_dict.get(name))
    return vals


def _pms_order():
    return (["season", "match_date", "matchday", "match_id", "team_id",
             "player_id", "player_name", "position", "archetype", "age",
             "nationality", "preferred_foot", "minutes_played", "is_starter",
             "home_or_away", "soul_archetype"]
            + [c for c, _ in PLAYER_STAT_COLS]
            + ["match_result", "is_mvp", "stats_json", "fidelity", "source"])

_PMS_INSERT = (
    "INSERT OR REPLACE INTO player_match_stats("
    + ",".join(
        ["season", "match_date", "matchday", "match_id", "team_id", "player_id",
         "player_name", "position", "archetype", "age", "nationality",
         "preferred_foot", "minutes_played", "is_starter", "home_or_away",
         "soul_archetype"]
        + [c for c, _ in PLAYER_STAT_COLS]
        + ["match_result", "is_mvp", "stats_json", "fidelity", "source"]
    )
    + ") VALUES ("
    + ",".join(["?"] * len(_pms_order()))
    + ")"
)


def ingest_match_package(conn, match_json_path, source=None):
    """Ingest one 26/27 match package (exporter `<name>.json`) into the
    warehouse. Idempotent per (season, matchday, home, away)."""
    p = Path(match_json_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    match = doc.get("match", {}) or {}
    players = doc.get("players", {}) or {}
    goals = doc.get("goals", []) or []
    financials = doc.get("financials", {}) or {}
    timeline = doc.get("timeline", []) or []

    season = _txt(match.get("season")) or "26/27"
    matchday = _numi(match.get("matchday"))
    home = _txt(match.get("home_team"))
    away = _txt(match.get("away_team"))
    if not home or not away:
        raise ValueError(f"no home/away team in {p.name}")
    home_id = get_or_create_team(conn, home)
    away_id = get_or_create_team(conn, away)
    hg, ag = _score_goals(match.get("score"))
    match_date = _txt(match.get("date"))
    src = source or f"export:{p.name}"

    fid = "v26"
    row = conn.execute(
        """SELECT match_id FROM matches
           WHERE season=? AND matchday=? AND home_team_id=? AND away_team_id=?""",
        (season, matchday, home_id, away_id),
    ).fetchone()
    if row:
        mid = row["match_id"]
    else:
        cur = conn.execute(
            """INSERT INTO matches(season, competition, matchday, home_team_id,
                   away_team_id, home_goals, away_goals, match_date, venue,
                   attendance, capacity, ticket_price, home_revenue, added_time,
                   is_derby, home_possession_pct, away_possession_pct, fidelity,
                   source, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (season, _txt(match.get("competition")) or "PLOFA", matchday,
             home_id, away_id, hg, ag, match_date,
             _txt(match.get("venue")),
             _numi(financials.get("attendance")),
             _numi(financials.get("stadium_capacity")),
             _num(financials.get("avg_ticket_price")),
             _num(financials.get("home_revenue")),
             _numi(match.get("added_time")),
             _int_bool(match.get("is_derby")),
             _num(match.get("home_possession_pct")),
             _num(match.get("away_possession_pct")),
             fid, src, json.dumps(match, default=str)),
        )
        mid = cur.lastrowid

    order = _pms_order()
    for pname, pdict in players.items():
        pid = get_or_create_player(
            conn, pdict.get("player"), team_id=home_id if pdict.get("team") == home else away_id,
            position=_txt(pdict.get("position")), nationality=_txt(pdict.get("nationality")),
            archetype=_txt(pdict.get("archetype")),
        )
        vals = _player_row_values(pdict, season, match_date, matchday, mid,
                                  (home_id if pdict.get("team") == home else away_id),
                                  fid, src)
        vals["player_id"] = pid
        conn.execute(_PMS_INSERT, [vals[k] for k in order])

    corner_counts = {}
    for ev in timeline:
        if isinstance(ev, dict) and ev.get("type") in ("CORNER_TAKEN",):
            t = _txt(ev.get("team"))
            corner_counts[t] = corner_counts.get(t, 0) + 1

    player_to_team = {
        "goals": "goals", "xg": "xg", "xa": "xa",
        "shots_on_target": "shots_on_target", "shots_off_target": "shots_off_target",
        "shots_blocked_att": "shots_blocked",
        "shots_inside_box": "shots_inside_box", "shots_outside_box": "shots_outside_box",
        "offsides": "offsides",
        "passes_attempted": "passes_attempted", "passes_completed": "passes_completed",
        "tackles_won": "tackles",
        "interceptions": "interceptions", "clearances": "clearances",
        "blocks": "blocks", "ball_recoveries": "recoveries",
        "pressures": "pressures",
        "fouls_committed": "fouls", "yellow_cards": "yellow_cards",
        "red_cards": "red_cards",
    }
    float_keys = {"xg", "xa"}
    for side, team, op in (("home", home, away), ("away", away, home)):
        tid = home_id if side == "home" else away_id
        rows = [p for p in players.values() if p.get("team") == team]
        agg = {c: 0 for c, _ in TEAM_STAT_COLS}
        for p in rows:
            for pkey, tkey in player_to_team.items():
                agg[tkey] += (_num(p.get(pkey)) or 0.0) if pkey in float_keys \
                    else (_numi(p.get(pkey)) or 0)
        agg["shots"] = (agg["shots_on_target"] + agg["shots_off_target"]
                        + agg["shots_blocked"])
        agg["corners"] = corner_counts.get(team, 0) or 0
        if agg["passes_attempted"]:
            agg["pass_accuracy"] = round(
                100.0 * agg["passes_completed"] / agg["passes_attempted"], 2)
        cols = ["match_id", "team_id", "side"] + [c for c, _ in TEAM_STAT_COLS] \
            + ["fidelity", "source"]
        conn.execute(
            "INSERT OR REPLACE INTO team_match_stats(" + ",".join(cols) + ") VALUES ("
            + ",".join(["?"] * len(cols)) + ")",
            [mid, tid, side] + [agg[c] for c, _ in TEAM_STAT_COLS]
            + [fid, src],
        )

    for g in goals:
        gteam = _txt(g.get("team"))
        scorer = _txt(g.get("scorer"))
        if not scorer:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO goals(season, matchday, match_id, team_id, opponent_id,
                   minute, scorer, assist, situation, xg, is_penalty,
                   is_own_goal, fidelity, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (season, matchday, mid,
             home_id if gteam == home else away_id,
             away_id if gteam == home else home_id,
             _num(g.get("minute")), scorer, _txt(g.get("assist")),
             _txt(g.get("situation")), _num(g.get("xg")),
             1 if g.get("situation") == "penalty" else 0,
             1 if gteam and gteam != home and home not in (gteam,) and home == _txt(match.get("away_team")) else 0,
             fid, src),
        )
    conn.commit()
    return {"match_id": mid, "season": season, "matchday": matchday,
            "home": home, "away": away, "players": len(players),
            "goals": len(goals)}


def sync_output_root(conn, root=None, dry_run=False):
    root = Path(root) if root else PLOFA_OUTPUT
    seen = set()
    for row in conn.execute("SELECT source FROM matches"):
        s = row["source"]
        if s and s.startswith("export:"):
            seen.add(s[7:])
    to_ingest, ingested, errors = 0, 0, []
    for jp in sorted(root.rglob("*.json")):
        if jp.name in seen:
            continue
        to_ingest += 1
        if dry_run:
            continue
        try:
            ingest_match_package(conn, jp)
            ingested += 1
        except Exception as exc:  # keep going; report at end
            errors.append((str(jp), str(exc)))
    return {"scanned": to_ingest, "ingested": ingested, "errors": errors}


def import_legacy_xlsx(conn, path=None, dry_run=False):
    path = Path(path) if path else Path(LEGACY_XLSX)
    import pandas as pd

    xl = pd.ExcelFile(path)
    stats = {"match_rows": 0, "player_match_rows": 0, "standings_rows": 0,
             "player_season_rows": 0, "detailed_raw_rows": 0, "errors": []}
    src = f"legacy:{path.name}"

    # ---- MATCH RESULTS -> matches ----
    mr = xl.parse("MATCH RESULTS")
    cutoff = datetime(2025, 7, 1)
    for _, r in mr.iterrows():
        h, a = _txt(r.get("H")), _txt(r.get("A"))
        gh, ga = _num(r.get("G")), _num(r.get("G2"))
        if not h or not a or gh is None or ga is None:
            continue
        d = r.get("DATE2")
        try:
            dd = pd.to_datetime(d).date().isoformat()
            season = "24/25" if datetime.fromisoformat(dd) < cutoff else "25/26"
        except Exception:
            dd, season = None, "UNKNOWN"
        h_id, a_id = get_or_create_team(conn, h), get_or_create_team(conn, a)
        row = conn.execute(
            """SELECT match_id FROM matches
               WHERE season=? AND matchday=? AND home_team_id=? AND away_team_id=?""",
            (season, _num(r.get("MD")), h_id, a_id),
        ).fetchone()
        if row:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO matches(season, competition, matchday,
                   home_team_id, away_team_id, home_goals, away_goals,
                   match_date, venue, attendance, capacity, fidelity, source,
                   raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (season, _txt(r.get("MID")) or "LEAGUE", _num(r.get("MD")),
             h_id, a_id, _numi(gh), _numi(ga), dd, _txt(r.get("STADIUM")),
             _numi(r.get("ATTENDANCE")), _numi(r.get("CAPACITY")), "legacy",
             src, json.dumps({c: None if pd.isna(r[c]) else r[c] for c in r.index}, default=str)),
        )
        stats["match_rows"] += 1

    # ---- UNREAL -> player_match_stats (24/25 + 25/26) ----
    u = xl.parse("UNREAL")
    u = u[u["Season"].astype(str).isin(("24/25", "25/26"))]
    label2canon = {}
    for col in u.columns:
        canon = _LEGACY_LABEL_MAP.get(lnorm(col))
        if canon:
            label2canon[col] = canon
    order = _pms_order()
    for _, r in u.iterrows():
        season = _txt(r.get("Season"))
        if not season:
            continue
        team = _txt(r.get("Team"))
        pname = _txt(r.get("Player"))
        if not team or not pname:
            continue
        team_id = get_or_create_team(conn, team)
        md = r.get("Matchday")
        md = None if pd.isna(md) else _numi(md)
        d = r.get("Match Date")
        dd = None
        if not pd.isna(d):
            try:
                dd = pd.to_datetime(d).date().isoformat()
            except Exception:
                dd = None
        if not dd:
            continue
        pid = get_or_create_player(conn, pname, team_id=team_id,
                                   position=_txt(r.get("Pos")))
        vals = {
            "season": season, "match_date": dd, "matchday": md,
            "match_id": None, "team_id": team_id, "player_id": pid,
            "player_name": pname, "position": _txt(r.get("Pos")),
            "archetype": None, "age": None, "nationality": None,
            "preferred_foot": None, "minutes_played": _numi(r.get("Minutes\n Played")),
            "is_starter": None, "home_or_away": None, "soul_archetype": None,
            "match_result": _txt(r.get("Official Result")),
            "is_mvp": _int_bool(r.get("MVP")),
            "stats_json": json.dumps(
                {c: None if pd.isna(r[c]) else (str(r[c]) if not isinstance(r[c], (int, float)) else r[c])
                 for c in r.index}, default=str),
            "fidelity": "legacy", "source": src,
        }
        for name, _typ in PLAYER_STAT_COLS:
            vals[name] = None
        for col, canon in label2canon.items():
            if canon in vals:
                vals[canon] = _num(r[col])
        try:
            conn.execute(_PMS_INSERT, [vals[k] for k in order])
        except sqlite3.IntegrityError as exc:
            stats["errors"].append((team, pname, str(exc)))
            continue
        stats["player_match_rows"] += 1

    # ---- STANDINGS -> season_standings (blocks as stored) ----
    st = xl.parse("STANDINGS")
    st = st[~st.isna().all(axis=1)]
    block, prev_rank_null = 0, True
    for _, r in st.iterrows():
        if pd.isna(r.get("RANK")):
            block += 1
            prev_rank_null = True
            continue
        club = _txt(r.get("Club"))
        if not club:
            continue
        tid = get_or_create_team(conn, club)
        mp = _numi(r.get("MP"))
        scope = "alltime-cumulative" if (mp or 0) >= 50 else "unknown-single-season"
        conn.execute(
            """INSERT INTO season_standings(season, scope, block, rank, team_id,
                   played, won, drawn, lost, gf, ga, gd, points, raw_json,
                   fidelity, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("LEGACY", scope, block, _numi(r.get("RANK")), tid, mp,
             _numi(r.get("W")), _numi(r.get("D")), _numi(r.get("L")),
             _numi(r.get("GF")), _numi(r.get("GA")), _numi(r.get("GD")),
             _num(r.get("Points")),
             json.dumps({c: None if pd.isna(r[c]) else r[c] for c in r.index}, default=str),
             "legacy", src),
        )
        stats["standings_rows"] += 1
        prev_rank_null = False

    # ---- PLAYER STATS -> player_season_stats (all-time aggregate) ----
    pp = xl.parse("PLAYER STATS")
    label2canon = {}
    for col in pp.columns:
        c0 = _txt(col)
        canon = _EMOJI_LABEL_MAP.get(c0) or _LEGACY_LABEL_MAP.get(lnorm(col))
        if canon:
            label2canon[col] = canon
    for _, r in pp.iterrows():
        pname = _txt(r.get("\U0001f464"))
        team = _txt(r.get("Team"))
        if not pname:
            continue
        tid = get_or_create_team(conn, team) if team else None
        pid = get_or_create_player(conn, pname, team_id=tid,
                                   position=_txt(r.get("POS")))
        vals = {c: None for c, _ in PLAYER_STAT_COLS}
        for col, canon in label2canon.items():
            if canon in vals:
                vals[canon] = _num(r[col])
        conn.execute(
            "INSERT INTO player_season_stats(season, team_id, player_id, "
            "player_name, position, "
            + ",".join(f'"{c}"' for c, _ in PLAYER_STAT_COLS)
            + ", stats_json, fidelity, source) VALUES (?,?,?,?,?,"
            + ",".join(["?"] * len(PLAYER_STAT_COLS)) + ",?,?,?)",
            ["ALL-TIME", tid, pid, pname, _txt(r.get("POS"))]
            + [vals[c] for c, _ in PLAYER_STAT_COLS]
            + [json.dumps({c: None if pd.isna(r[c]) else (str(r[c]) if not isinstance(r[c], (int, float)) else r[c])
                           for c in r.index}, default=str),
               "legacy", src],
        )
        stats["player_season_rows"] += 1

    # ---- DETAILED MATCH RESULTS -> raw archive ----
    dd = xl.parse("DETAILED MATCH RESULTS")
    for _, r in dd.iterrows():
        conn.execute(
            "INSERT INTO legacy_detailed_goals_raw(season, raw_json, fidelity, source) "
            "VALUES (?,?,?,?)",
            ("LEGACY",
             json.dumps({c: None if pd.isna(r[c]) else r[c] for c in r.index}, default=str),
             "legacy", src),
        )
        stats["detailed_raw_rows"] += 1

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_import', ?)",
        (datetime.now().isoformat(timespec="seconds"),),
    )
    conn.commit()
    return stats


def report(conn):
    lines = []
    for season in SEASONS:
        m = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT matchday) FROM matches WHERE season=?",
            (season,),
        ).fetchone()
        pm = conn.execute(
            "SELECT COUNT(*) FROM player_match_stats WHERE season=?", (season,),
        ).fetchone()
        tms = conn.execute(
            "SELECT COUNT(*) FROM team_match_stats WHERE match_id IN "
            "(SELECT match_id FROM matches WHERE season=?)", (season,),
        ).fetchone()
        lines.append(f"{season}: matches={m[0]} (MDs={m[1]}), player-match rows={pm[0]}, team-match stats={tms[0]}")
    lines.append("")
    lines.append("Top 5 all-time scorers: " + str([
        dict(zip(("player", "goals", "games"),
                 row))
        for row in conn.execute(
            """SELECT p2.name, SUM(pms.goals) AS g, COUNT(*) AS games
               FROM player_match_stats pms
               JOIN players p ON p.player_id = pms.player_id
               LEFT JOIN player_aliases a ON a.alias = p.name
               JOIN players p2 ON p2.player_id = COALESCE(a.canonical_player_id, p.player_id)
               WHERE pms.season IN ('24/25','25/26','26/27')
               GROUP BY p2.player_id ORDER BY g DESC LIMIT 5""").fetchall()
    ]))
    lines.append("")
    lines.append("Rows per table: " + str({
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("teams", "players", "matches", "team_match_stats",
                  "player_match_stats", "player_season_stats",
                  "season_standings", "goals", "legacy_detailed_goals_raw")
    }))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PLOFA all-time stats warehouse")
    ap.add_argument("--db", default=None, help="sqlite path (default alltime.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create the schema")
    p_leg = sub.add_parser("import-legacy", help="import PLOFA-ALL-TIME.xlsx")
    p_leg.add_argument("--xlsx", default=None, help="legacy workbook path")
    p_sync = sub.add_parser("sync", help="ingest 26/27 plofa_output packages")
    p_sync.add_argument("--root", default=None, help="output root (default plofa_output)")
    p_sync.add_argument("--dry-run", action="store_true")
    sub.add_parser("report", help="print warehouse summary")
    p_alias = sub.add_parser("alias", help="manage player name identities")
    asub = p_alias.add_subparsers(dest="alias_cmd", required=True)
    p_aadd = asub.add_parser("add", help="register an alias name -> canonical name")
    p_aadd.add_argument("alias", help="old / variant player name")
    p_aadd.add_argument("canonical", help="current / true player name")
    p_aadd.add_argument("--note", default=None, help="why (renamed, spelling fix...)?")
    p_aadd.add_argument("--apply", action="store_true",
                        help="also rewire existing stat rows to the canonical identity")
    asub.add_parser("scan", help="propose same-person name pairs (never auto-applies)")
    asub.add_parser("list", help="show registered aliases")
    p_aapply = asub.add_parser("apply", help="rewire all registered aliases' stat rows")
    p_aapply.add_argument("--all", action="store_true", help="apply every registered alias")
    args = ap.parse_args(argv)

    conn = connect(args.db)
    if args.cmd == "init":
        init_schema(conn)
        print(f"schema ready: {args.db or DEFAULT_DB}")
    elif args.cmd == "import-legacy":
        init_schema(conn)
        st = import_legacy_xlsx(conn, args.xlsx)
        print("legacy import:", st)
    elif args.cmd == "sync":
        init_schema(conn)
        st = sync_output_root(conn, args.root, dry_run=args.dry_run)
        print("sync:", st)
    elif args.cmd == "report":
        init_schema(conn)
        print(report(conn))
    elif args.cmd == "alias":
        init_schema(conn)
        if args.alias_cmd == "add":
            canon = add_alias(conn, args.alias, args.canonical, note=args.note)
            cname = conn.execute(
                "SELECT name FROM players WHERE player_id=?", (canon,)).fetchone()["name"]
            if args.apply:
                moved, deleted = apply_alias_fixup(conn)
            else:
                moved = deleted = 0
            print(f"alias: {args.alias!r} -> {cname!r} (canonical player_id {canon}); "
                  f"applied moved={moved} deleted={deleted}")
        elif args.alias_cmd == "scan":
            cands = alias_scan(conn)
            if not cands:
                print("alias scan: no candidate name pairs found")
            for c in cands:
                print(f"[score {c['score']}] {c['kind']}: {c['alias']!r} -> {c['canonical']!r}")
                print(f"  {c['alias']!r}: seasons={c['seasons_a']} teams={c['teams_a']} games={c['games_a']}")
                print(f"  {c['canonical']!r}: seasons={c['seasons_b']} teams={c['teams_b']} games={c['games_b']}")
                print(f"  evidence: shared teams={c['shared_teams']} "
                      f"shared positions={c['positions']}")
                print(f"  apply: python alltime_db.py alias add {c['alias']!r} {c['canonical']!r} --apply")
        elif args.alias_cmd == "list":
            rows = list_aliases(conn)
            if not rows:
                print("alias list: none registered")
            for r in rows:
                print(f"{r['alias']!r} -> {r['canonical']!r}" + (f"  ({r['note']})" if r["note"] else ""))
        elif args.alias_cmd == "apply":
            if args.all:
                moved, deleted = apply_alias_fixup(conn)
                print(f"alias apply: moved={moved} rows, deleted={deleted} duplicates")
            else:
                print("alias apply: use --all to rewire every registered alias, or "
                      "re-run 'alias add ... --apply'")
    return 0


if __name__ == "__main__":
    sys.exit(main())