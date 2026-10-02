"""
Seed the fixture ledger for the original Red Wolves v Avada Zenith MD5.

Background: the original MD5 (RW 2-1 AZ) was played on 2026-09-19 and the
live `season_state.json` ALREADY reflects the correct post-MD5 world
(RW W3 D0 L2 PTS9, AZ W0 D1 L4 PTS1, both 5 played, injuries intact).
The only safety gap: the fixture ledger is empty, so a future re-run of MD5
via `auto_run_match.py --play` would snapshot post-MD5 as "pre-match" and
push both clubs to 6 played.

This script stores, under `fixture_ledger["5|Red Wolves|Avada Zenith"]`, a
faithful MD4-end snapshot of both clubs (players + standings rows) exactly
as `begin_fixture()` captures them. It is derived from alltime.db MD1-4 —
the same replay the match runner applies serial-wise — plus pre-MD5
injuries carried over from the live state. After seeding, any re-run of MD5
unwinds to MD4-end and re-records cleanly at 5 played.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import sys
import time as _time
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


def build_md4_snapshot(cur, live: SeasonState
                       ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Replay MD1-4 in a scratch SeasonState, mirroring the runner: returns
    ({player_name: state}, {team: standings_row}) at MD4-end."""
    scratch = SeasonState(SEASON, STATE_PATH)
    scratch.players = {}
    scratch.standings = {}
    scratch.fixture_ledger = {}

    _, name2id, id2name = load_db()
    home_id, away_id = name2id[HOME], name2id[AWAY]
    pool = fetch_player_matches(cur, [home_id, away_id])
    # Parity guard: only rebuild players that exist in the live rosters.
    live_names = set(live.players)
    pool = {n: m for n, m in pool.items() if n in live_names}
    roster_names = sorted(pool.keys())

    for player in roster_names:
        base = scratch.get_player_state(player)  # default 0-match template
        fresh = json.loads(json.dumps(base))
        fresh["recent_ratings"] = []
        fresh["recent_goals"] = []
        fresh["yellow_cards_last_6"] = []
        scratch.players[player] = fresh

    md_matches: Dict[int, List[Tuple[str, dict]]] = {}
    for player, matches in pool.items():
        for m in matches:
            md_matches.setdefault(m["matchday"], []).append((player, m))

    for md in sorted(md_matches):
        played = set()
        md_date = None
        for player, m in md_matches[md]:
            if m["minutes_played"] <= 0 and not m["is_starter"]:
                continue
            md_date = md_date or m["match_date"]
            ending = max(20.0, 100.0 - (m["minutes_played"] / 90.0) * 45.0)
            scratch.record_post_match(
                name=player, rating=m["rating"], goals=m["goals"],
                minutes_played=m["minutes_played"], ending_stamina=ending,
                yellow=m["yellow"], red=m["red"],
                match_date=m["match_date"], assists=m["assists"],
            )
            played.add(player)
        scratch.advance_matchday(roster_names, played, md_date)

    # Carry pre-MD5 injuries from the live (post-MD5) state onto the snapshot.
    for player in roster_names:
        st = scratch.players[player]
        src = live.players.get(player, {})
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
            continue  # acquired during MD5 — must not leak into the baseline
        for field in INJURY_FIELDS:
            if field in src:
                st[field] = copy.deepcopy(src[field])

    # Standings rows for the two clubs at MD4-end (from DB results).
    for club, team in ((home_id, HOME), (away_id, AWAY)):
        results = fetch_match_results(cur, club, id2name)
        scratch.record_team_result is not None
        for _md, opp, _loc, gf, ga in results:
            scratch.standings.setdefault(
                team, {"w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "form": []})
        # record fresh row
        scratch.standings[team] = {
            "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "form": []}
        for _md, opp, _loc, gf, ga in results:
            scratch.record_team_result(team, opp, gf, ga)

    players_out = {
        name: copy.deepcopy(scratch.players.get(name)) if name in scratch.players else None
        for name in roster_names
    }
    standings_out = {
        team: copy.deepcopy(scratch.standings.get(team)) for team in (HOME, AWAY)
    }
    return players_out, standings_out


def main() -> None:
    if not os.path.exists(STATE_PATH):
        print(f"missing {STATE_PATH}")
        sys.exit(1)

    backup = f"{STATE_PATH}.pre-ledger-seed-{_time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(STATE_PATH, backup)
    print(f"backup: {backup}")

    live = SeasonState(SEASON, STATE_PATH)
    print("live RW:", live.standings[HOME])
    print("live AZ:", live.standings[AWAY])

    con, _, _ = load_db()
    players_snap, standings_snap = build_md4_snapshot(con.cursor(), live)

    key = f"5|{HOME}|{AWAY}"
    live.fixture_ledger[key] = {
        "players": players_snap,
        "standings": standings_snap,
    }
    live.save()
    con.close()

    print(f"\nseeded fixture_ledger[{key!r}]")
    print("snapshot RW (MD4-end):", standings_snap[HOME])
    print("snapshot AZ (MD4-end):", standings_snap[AWAY])
    print("snapshot players:", len(players_snap))

    # ── Verify the invariant: a re-run of MD5 must land back at 5 played ──
    replay = SeasonState(SEASON, STATE_PATH)
    roster = list(players_snap.keys())
    was_replay = replay.begin_fixture(5, HOME, AWAY, roster)
    replay.record_team_result(HOME, AWAY, 2, 1)
    print("\nre-run detected:", was_replay)
    print("re-run RW:", replay.standings[HOME], "(expect 3-0-2 pts 9)")
    print("re-run AZ:", replay.standings[AWAY], "(expect 0-1-4 pts 1)")
    for hero in ("Nathan Opaz", "Viktor Brans", "Seane Wellington"):
        p = replay.players.get(hero, {})
        print(f"  {hero}: matches={p.get('season_matches')} "
              f"mins={p.get('season_minutes')} rr={p.get('recent_ratings')}")
    assert replay.standings[HOME]["pts"] == 9 and replay.standings[HOME]["l"] == 2
    assert replay.standings[AWAY]["pts"] == 1 and replay.standings[AWAY]["l"] == 4
    print("INVARIANT OK — MD5 re-run replaces instead of appending.")


if __name__ == "__main__":
    main()