"""
PLOFA 26/27 — Regression test for the runs/sprints corruption bug.

SYMPTOM
-------
After playing 90 minutes some players showed `sprints == 1` and `runs == 0/1`
while their `sprinting_seconds` / `running_seconds` / `distance_covered` were
perfectly normal. So the per-minute activity baseline was fine — the count-based
`runs` / `sprints` / `high_speed_sprints` fields were being clobbered.

ROOT CAUSE (opta_analytics.py)
------------------------------
1. `_compute_activity` ASSIGNED (overwrote) `self._physics_stats[name]` on every
   minute instead of accumulating it. Each position_log frame only carries ONE
   minute's physics trace, so the dict ended up holding only the LAST physics
   sample for a player (e.g. a single sprint seen in minute 90).
2. `_assemble_player_data` then REPLACED the calibrated full-match baseline
   `runs`/`sprints` with that single (sparse) physics count, collapsing
   90 minutes of running into 1 sprint / 1 run.

FIX
---
1. Accumulate `_physics_stats` across all minutes (sum counts, max top-speed).
2. Reconcile (max) physics with the baseline instead of replacing it: the trace
   is sparse and known to UNDER-count, so it may only raise counts above the
   calibrated baseline, never drag them below.

This test reproduces the exact scenario and asserts the fix holds. It does NOT
run a match, so season_stats / season_state are never touched.
"""
from types import SimpleNamespace

from opta_analytics import OptaAnalytics


def _make_player(name, position="CM", work_rate=50.0, pace=60.0):
    dna = SimpleNamespace(
        preferred_foot="right",
        physical=SimpleNamespace(pace=pace),
        mental=SimpleNamespace(work_rate=work_rate),
    )
    return SimpleNamespace(name=name, position=position, dna=dna)


def _player_row(name, minute, *, physics_dist=0.0, physics_sprint=0.0,
                physics_high=0.0, physics_top_speed=0.0, touches=0,
                distance_touch=1.0, distance_drift=0.5):
    return {
        "player": name,
        "position": "CM",
        "x": 52.5,
        "y": 34.0,
        "distance_touch": distance_touch,
        "distance_drift": distance_drift,
        "distance_total": distance_touch + distance_drift,
        "touches": touches,
        "peak_touch_jump": 0.0,
        "physics_distance_m": physics_dist,
        "physics_sprint_count": physics_sprint,
        "physics_high_speed_sprint_count": physics_high,
        "physics_top_speed_mps": physics_top_speed,
    }


def _build_result(frames):
    """Minimal MatchResult-like object with 90-minute telemetry."""
    config = SimpleNamespace(home_team="Home", away_team="Away")
    momentum_log = [
        {"minute": m, "momentum": 0.0, "home_goals": 0, "away_goals": 0}
        for m in range(1, 91)
    ]
    return SimpleNamespace(
        config=config, timeline=[], position_log=frames, momentum_log=momentum_log,
    )


def test_single_late_sprint_no_longer_collapses_to_one():
    """A player whose LAST physics sample held a single sprint must NOT end up
    with sprints==1 / runs==1 for the whole match."""
    frames = []
    for m in range(1, 91):
        home = [
            _player_row("Healthy", m),
            _player_row(
                "Victim", m,
                physics_dist=5.0 if m == 90 else 0.0,
                physics_sprint=1.0 if m == 90 else 0.0,
                physics_top_speed=8.5 if m == 90 else 0.0,
            ),
        ]
        frames.append({
            "minute": m, "home": home, "away": [],
            "home_goals": 0, "away_goals": 0,
            "possession_team": "Home", "phase": "open_play",
        })

    result = _build_result(frames)
    all_players = {"Home": {"starters": [_make_player("Victim"), _make_player("Healthy")],
                            "substitutes": []}}
    opta = OptaAnalytics(result, all_players).compute()

    out = opta.player_data["Victim"]
    assert out["sprints"] > 1, f"BUG: sprints={out['sprints']} (secs={out['sprinting_seconds']})"
    assert out["runs"] > 1, f"BUG: runs={out['runs']}"
    assert out["sprints"] >= 5, out
    assert out["runs"] >= 10, out

    h = opta.player_data["Healthy"]
    assert h["sprints"] >= 5, h
    assert h["runs"] >= 10, h


def test_physics_stats_accumulate_across_minutes():
    """BUG #1: _physics_stats must be SUMMED across minutes, not overwritten by
    the last frame."""
    frames = []
    physics_minutes = {10, 20, 30, 40, 50, 60, 70, 80, 90}  # 9 minutes
    for m in range(1, 91):
        home = [
            _player_row(
                "Victim", m,
                physics_dist=4.0 if m in physics_minutes else 0.0,
                physics_sprint=2.0 if m in physics_minutes else 0.0,
                physics_high=1.0 if m in physics_minutes else 0.0,
                physics_top_speed=9.0 if m in physics_minutes else 0.0,
            ),
        ]
        frames.append({
            "minute": m, "home": home, "away": [],
            "home_goals": 0, "away_goals": 0,
            "possession_team": "Home", "phase": "open_play",
        })

    result = _build_result(frames)
    all_players = {"Home": {"starters": [_make_player("Victim")], "substitutes": []}}
    opta = OptaAnalytics(result, all_players).compute()

    phys = opta._physics_stats["Victim"]
    assert phys["sprint_count"] == 18.0, f"accumulated sprint_count={phys['sprint_count']}"
    assert phys["high_speed_sprint_count"] == 9.0, phys
    assert phys["top_speed_mps"] == 9.0, phys  # max across minutes, not last

    out = opta.player_data["Victim"]
    # Physics accumulated to 18 (proven above); it may also have boosted the
    # per-minute sprinting-seconds baseline, so the reconciled value is the max
    # of both and can be >= 18. The critical regression check is that it is NOT
    # the collapsed last-minute-only value (2) the bug produced.
    assert out["sprints"] >= 18, f"sprints={out['sprints']} (must reflect accumulation, not last-minute=2)"
    assert out["runs"] >= 30, out


def test_runs_never_zero_for_full_match_participant():
    """A player in all 90 frames with baseline activity must never report
    0/1 runs or sprints."""
    frames = []
    for m in range(1, 91):
        home = [_player_row("Victim", m, touches=3, distance_touch=3.0, distance_drift=1.0)]
        frames.append({
            "minute": m, "home": home, "away": [],
            "home_goals": 0, "away_goals": 0,
            "possession_team": "Home", "phase": "open_play",
        })
    result = _build_result(frames)
    all_players = {"Home": {"starters": [_make_player("Victim")], "substitutes": []}}
    out = OptaAnalytics(result, all_players).compute().player_data["Victim"]
    assert out["sprints"] > 1, out
    assert out["runs"] > 1, out


if __name__ == "__main__":
    test_single_late_sprint_no_longer_collapses_to_one()
    test_physics_stats_accumulate_across_minutes()
    test_runs_never_zero_for_full_match_participant()
    print("ALL TESTS PASSED")
