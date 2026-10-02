"""
Regression tests for replace-on-rerun fixture recording.

A fixture keyed by (matchday, home, away) must only ever contribute ONE
matchday to player season totals and league standings, no matter how many
times it is re-run.  This guards the exact accident that corrupted the
Red Wolves vs Avada Zenith MD5 ledger: re-running a played fixture MUST
replace the previous recording, never append to it.
"""

from __future__ import annotations
import json

from season_manager import SeasonState


FIXTURE = (5, "Red Wolves", "Avada Zenith")

# Simulated two-club rosters (starters + bench) in one flat name list.
ROSTER = [
    "RW GK", "RW LB", "RW CB B", "RW CB A", "RW RB", "RW CM B", "RW CM A",
    "RW CAM", "RW ST B", "RW ST A", "RW W A", "RW W B", "RW SUB 1",
    "AZ GK", "AZ LB", "AZ CB B", "AZ CB A", "AZ RB", "AZ CM B", "AZ CM A",
    "AZ CAM", "AZ ST B", "AZ ST A", "AZ W A", "AZ W B", "AZ SUB 1",
]


def _record_once(state: SeasonState, matchday: int, home: str, away: str,
                 hg: int, ag: int) -> None:
    """Replicate what auto_run_match._persist_post_match does for one
    fixture: unwrap prior recording, snapshot pre-match, then apply."""
    state.begin_fixture(matchday, home, away, ROSTER)
    # officials / injury ticks are not needed; standings + player counters are
    state.record_team_result(home, away, hg, ag)
    for name in ROSTER:
        minutes = 90 if "ST" in name or "W " in name else (75 if "SUB" not in name else 12)
        state.record_post_match(
            name=name,
            rating=7.1 if hg > ag else 6.2,
            goals=1 if hg > ag and "ST" in name else 0,
            minutes_played=minutes,
            ending_stamina=85.0,
        )
    state.save()


def _snapshot(state: SeasonState) -> dict:
    return {
        "played": {p: state.players.get(p, {}).get("season_matches", 0)
                   for p in ROSTER},
        "minutes": {p: state.players.get(p, {}).get("season_minutes", 0)
                    for p in ROSTER},
        "standings": {**state.standings.get(FIXTURE[1], {}),
                      **state.standings.get(FIXTURE[2], {})},
    }


def test_replay_keeps_md_played_at_one(tmp_path):
    path = str(tmp_path / "season_state.json")
    state = SeasonState("26/27", path)
    md, home, away = FIXTURE

    for _ in range(3):
        was_replay = state.begin_fixture(md, home, away, ROSTER)
        state.record_team_result(home, away, 2, 1)
        for name in ROSTER:
            state.record_post_match(
                name=name, rating=7.1, goals=0,
                minutes_played=90 if "SUB" not in name else 12,
                ending_stamina=85.0,
            )
        state.save()

    # After 3 runs of the SAME MD5 fixture, everyone still has ONE
    # appearance and the standings show one result each.
    played = {p: state.players[p]["season_matches"] for p in ROSTER}
    assert set(played.values()) == {1}, played
    home_row = state.standings[home]
    away_row = state.standings[away]
    assert home_row["w"] + home_row["d"] + home_row["l"] == 1, home_row
    assert away_row["w"] + away_row["d"] + away_row["l"] == 1, away_row


def test_replay_uses_latest_score_only(tmp_path):
    path = str(tmp_path / "season_state.json")
    state = SeasonState("26/27", path)
    md, home, away = FIXTURE

    # First run: home wins.  Second run: away wins — a re-run must REPLACE.
    for hg, ag in ((2, 1), (0, 1)):
        state.begin_fixture(md, home, away, ROSTER)
        state.record_team_result(home, away, hg, ag)
        for name in ROSTER:
            state.record_post_match(
                name=name, rating=6.5, goals=0,
                minutes_played=90 if "SUB" not in name else 12,
                ending_stamina=85.0,
            )
        state.save()

    home_row = state.standings[home]
    away_row = state.standings[away]
    # Away victory is the canonical result now.
    assert home_row["l"] == 1 and home_row["w"] == 0, home_row
    assert away_row["w"] == 1 and away_row["l"] == 0, away_row
    assert state.standings[home]["pts"] + state.standings[away]["pts"] == 3


def test_replay_is_persistent_across_loads(tmp_path):
    path = str(tmp_path / "season_state.json")
    state = SeasonState("26/27", path)
    md, home, away = FIXTURE
    state.begin_fixture(md, home, away, ROSTER)
    state.record_team_result(home, away, 1, 1)
    for name in ROSTER:
        state.record_post_match(name=name, rating=6.0, goals=0,
                                minutes_played=20, ending_stamina=80.0)
    state.save()

    # Reload (new process) — the ledger snapshot must survive, so a re-run
    # after a reload still replaces instead of appending.
    reloaded = SeasonState("26/27", path)
    assert reloaded.fixture_was_played(md, home, away)
    was_replay = reloaded.begin_fixture(md, home, away, ROSTER)
    assert was_replay
    reloaded.record_team_result(home, away, 3, 0)
    for name in ROSTER:
        reloaded.record_post_match(name=name, rating=7.8, goals=0,
                                   minutes_played=20, ending_stamina=80.0)
    reloaded.save()

    final = SeasonState("26/27", path)
    assert final.standings[home]["w"] == 1
    assert final.standings[away]["l"] == 1
    assert final.players["RW GK"]["season_matches"] == 1


def test_ledger_does_not_leak_across_different_fixtures(tmp_path):
    path = str(tmp_path / "season_state.json")
    state = SeasonState("26/27", path)

    state.begin_fixture(5, "Red Wolves", "Avada Zenith", ROSTER)
    state.record_team_result("Red Wolves", "Avada Zenith", 2, 1)
    for name in ROSTER:
        state.record_post_match(name=name, rating=7.0, goals=0,
                                minutes_played=90, ending_stamina=80.0)
    state.save()

    # A DIFFERENT fixture (Play City home to RW on MD6) must still append.
    state.begin_fixture(6, "Play City", "Red Wolves", ROSTER)
    state.record_team_result("Play City", "Red Wolves", 0, 1)
    for name in ROSTER:
        state.record_post_match(name=name, rating=6.8, goals=0,
                                minutes_played=90, ending_stamina=80.0)
    state.save()

    assert state.standings["Red Wolves"]["w"] == 2
    assert state.standings["Red Wolves"]["l"] == 0
    assert state.standings["Play City"]["l"] == 1
    assert state.players["RW GK"]["season_matches"] == 2