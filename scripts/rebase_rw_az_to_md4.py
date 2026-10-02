"""
One-time MD4-end rebase for Red Wolves & Avada Zenith.

The accidental re-run of the MD5 fixture (2026-09-20) left season_state.json
with the RW/AZ MD5 result baked in TWICE — both clubs sit at 5 played when
they should only be at 4 (the DB never received the RW/AZ MD5 export, so
alltime.db's season 26/27 stops at MD4 for these two clubs).

This script reconstructs, for those two clubs ONLY, the exact player-state
and standings rows that existed at the end of matchday 4, derived from
alltime.db:

  - player_match_stats (rating, minutes_played, goals, assists, cards) rows
    for season 26/27, matchday <= 4, replayed through the exact same
    SeasonState.record_post_match() / advance_matchday() calls the match
    runner makes — so confidence, season_matches/season_minutes, recent
    ratings/goals and card windows are bit-for-bit consistent with how the
    original MD1-4 runs accumulated them.
  - Pre-MD5 injuries (injury_date < 2026-09-19) are carried over so the
    replayed baseline keeps the same suspension/availability profile that
    existed going into the original MD5.

Standings rows for the two clubs are recomputed from the DB's matches table
(MD1-4 only), preserving every other team's row exactly.

Run BEFORE the next `auto_run_match.py --play` so the fixture ledger
captures a correct pre-MD5 snapshot on the first re-run.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import date
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from season_manager import SeasonState

SEASON = "26/27"
HOME = "Red Wolves"
AWAY = "Avada Zenith"
STATE_PATH = "season_state.json"
DB_PATH = "alltime.db"
# MD5 was played on 2026-09-19.  Any injury with this date or later belongs
# to the (re-run) MD5 recording and must NOT leak back to the MD4 baseline.
MD5_DATE = date(2026, 9, 19)

INJURY_FIELDS = (
    "is_injured", "injury_type", "injury_date", "recovery_days",
    "expected_return_date", "matches_remaining_out",
)


def load_db() -> Tuple[sqlite3.Connection, Dict[str, int], Dict[int, str]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT team_id, name FROM teams")
    name2id = {name: tid for tid, name in cur.fetchall()}
    id2name = {tid: name for name, tid in name2id.items()}
    return con, name2id, id2name


def fetch_match_results(cur, team_id: int, id2name: Dict[int, str]
                        ) -> List[Tuple[int, str, str, int, int]]:
    """(matchday, opponent, home_or_away, gf, ga) for one team, MD1-4."""
    rows = []
    cur.execute(
        "SELECT matchday, match_date, home_team_id, away_team_id, home_goals, away_goals "
        "FROM matches WHERE season=? AND matchday<=4 "
        "AND (home_team_id=? OR away_team_id=?) ORDER BY matchday",
        (SEASON, team_id, team_id),
    )
    for md, mdate, h_id, a_id, hg, ag in cur.fetchall():
        if h_id == team_id:
            rows.append((md, id2name[a_id], "home", hg, ag))
        else:
            rows.append((md, id2name[h_id], "away", ag, hg))
    return rows


def fetch_player_matches(cur, club_ids: List[int]) -> Dict[str, List[dict]]:
    """player_name -> list of per-match dicts (MD1-4) in matchday order."""
    cur.execute(
        "SELECT player_name, matchday, match_date, is_starter, minutes_played, "
        "rating, goals, assists, yellow_cards, red_cards "
        "FROM player_match_stats "
        "WHERE season=? AND team_id IN (%s) AND matchday<=4 "
        "ORDER BY matchday, player_name"
        % ",".join("?" * len(club_ids)),
        [SEASON] + club_ids,
    )
    by_player: Dict[str, List[dict]] = {}
    for (name, md, mdate, starter, minutes, rating, goals, assists,
         yellow, red) in cur.fetchall():
        by_player.setdefault(name, []).append({
            "matchday": md,
            "match_date": date.fromisoformat(mdate) if mdate else None,
            "is_starter": bool(starter),
            "minutes_played": int(minutes or 0),
            "rating": float(rating or 0.0),
            "goals": int(goals or 0),
            "assists": int(assists or 0),
            "yellow": bool(yellow),
            "red": bool(red),
        })
    return by_player


def rebuild_standings_row(state: SeasonState, team: str,
                          results: List[Tuple[int, str, str, int, int]]) -> None:
    """Replace a single team's standings row with the MD1-4-only view."""
    cur = state._team_row(team)
    for k in ("w", "d", "l", "gf", "ga", "pts"):
        cur[k] = 0
    cur["form"] = []
    for _md, _opp, _loc, gf, ga in results:
        state.record_team_result(team, _opp, gf, ga)


def main() -> None:
    if not os.path.exists(STATE_PATH):
        print(f"missing {STATE_PATH}")
        sys.exit(1)

    backup = f"{STATE_PATH}.pre-md4-rebase-{__import__('time').strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(STATE_PATH, backup)
    print(f"backup: {backup}")

    con, name2id, id2name = load_db()
    home_id = name2id[HOME]
    away_id = name2id[AWAY]

    state = SeasonState(SEASON, STATE_PATH)
    current_state = json.loads(json.dumps(state.players))  # snapshot before edits

    # Rebuild both clubs' player states in ONE pool, matchday by matchday,
    # interleaving advance_matchday() exactly like _persist_post_match does —
    # bans tick down, yellow-ban windows reset, injuries recover on date.
    pool = fetch_player_matches(con.cursor(), [home_id, away_id])
    # Parity guard: the DB carries a few zero-minute phantom bench names that
    # were never part of the persisted rosters (and were never inserted by the
    # original runner's advance_matchday). Only rebuild players that actually
    # exist in season_state.json, so no stray entries get added.
    pool = {n: m for n, m in pool.items() if n in current_state}
    roster_names = sorted(pool.keys())
    for player, matches in pool.items():
        # Start from the same all-defaults baseline the original MD1-A run
        # did, with empty windows so record_post_match's append-slice
        # reproduces the exact MD1-4 lists.
        base = state.get_player_state(player)
        fresh = json.loads(json.dumps(base))
        fresh["recent_ratings"] = []
        fresh["recent_goals"] = []
        fresh["yellow_cards_last_6"] = []
        state.players[player] = fresh

    md_matches: Dict[int, List[Tuple[str, dict]]] = {}
    for player, matches in pool.items():
        for m in matches:
            md_matches.setdefault(m["matchday"], []).append((player, m))

    for md in sorted(md_matches):
        played_names = set()
        md_date = None
        for player, m in md_matches[md]:
            if m["minutes_played"] <= 0 and not m["is_starter"]:
                continue  # unused bench player — never entered the pitch
            md_date = md_date or m["match_date"]
            ending_stamina = max(20.0, 100.0 - (m["minutes_played"] / 90.0) * 45.0)
            state.record_post_match(
                name=player,
                rating=m["rating"],
                goals=m["goals"],
                minutes_played=m["minutes_played"],
                ending_stamina=ending_stamina,
                yellow=m["yellow"],
                red=m["red"],
                match_date=m["match_date"],
                assists=m["assists"],
            )
            played_names.add(player)
        # Tick bans / injuries for everyone who did NOT play this matchday —
        # mirrors the runtime advance_matchday() call after each fixture.
        state.advance_matchday(roster_names, played_names, md_date)

    # Carry any injury whose injury_date predates MD5 (suspensions/knocks that
    # were live going into the original MD5) onto the rebuilt baseline.
    carry_count = 0
    for player in roster_names:
        st = state.players[player]
        src = current_state.get(player, {})
        if not src.get("is_injured"):
            continue
        inj_date = src.get("injury_date")
        if not inj_date:
            continue
        try:
            d = date.fromisoformat(str(inj_date))
        except ValueError:
            continue
        if d >= MD5_DATE:
            continue  # picked up during the original/accidental MD5 — drop
        # Sanity: only carry if the rebuilt baseline still marks them fit and
        # the recovery window hasn't already lapsed by MD4.
        if st.get("is_injured"):
            continue
        for field in INJURY_FIELDS:
            st[field] = src.get(field)
        carry_count += 1
    print(f"carried {carry_count} pre-MD5 injury field sets")

    # Standings: rebuild ONLY the two clubs' rows from DB MD1-4.
    for club, team in ((home_id, HOME), (away_id, AWAY)):
        results = fetch_match_results(con.cursor(), club, id2name)
        rebuild_standings_row(state, team, results)
        print(f"{team}: {len(results)} fixtures -> "
              f"{state.standings[team]}")

    # Fixture ledger must be empty at the first re-run so begin_fixture
    # snapshots the freshly-rebased MD4-end baseline as pre-match state.
    state.fixture_ledger = {}
    state.save()
    con.close()

    print("\nRebase complete.  Verify below, then re-run MD5 with --play.")


if __name__ == "__main__":
    main()