"""Focused tests for pitch_replay.py — interpolation continuity, the
substitution band, ball sampling, the StatsBomb mapping, and the headless
step export.

Run: python3 test_pitch_replay.py     (from repo root, PYTHONPATH=.)

Plain assert-based scripts, matching the project's existing test style.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from pitch_replay import (
    ReplayData, to_statsbomb, export_interpolated, SB_LEN, SB_WID,
)


# ─────────────────────────────────────────────────────────────
# Synthetic replay fixture (no match engine needed)
# ─────────────────────────────────────────────────────────────

def _synthetic_replay() -> ReplayData:
    ball = [
        {"t": 0.0, "x": 5.0, "y": 34.0, "kind": "rest"},
        {"t": 30.0, "x": 30.0, "y": 36.0, "kind": "rest"},
        {"t": 60.0, "x": 52.5, "y": 34.0, "kind": "move"},
        {"t": 90.0, "x": 70.0, "y": 30.0, "kind": "move"},
        {"t": 120.0, "x": 90.0, "y": 32.0, "kind": "move"},
        {"t": 180.0, "x": 104.0, "y": 30.0, "kind": "move"},
    ]
    # 3 minute-frames.  Home: A,B,GK (full game), C subs on at minute 3.
    # Away: X,Y (full game), Z subbed off after minute 2.
    frames = [
        {
            "minute": 1,
            "home": [
                {"player": "Al", "position": "ST", "x": 70.0, "y": 30.0},
                {"player": "Bo", "position": "CM", "x": 45.0, "y": 34.0},
                {"player": "Keeper", "position": "GK", "x": 5.0, "y": 34.0},
            ],
            "away": [
                {"player": "Xen", "position": "ST", "x": 35.0, "y": 34.0},
                {"player": "Yan", "position": "CB", "x": 20.0, "y": 20.0},
                {"player": "Zed", "position": "MID", "x": 25.0, "y": 45.0},
            ],
        },
        {
            "minute": 2,
            "home": [
                {"player": "Al", "position": "ST", "x": 80.0, "y": 20.0},
                {"player": "Bo", "position": "CM", "x": 55.0, "y": 30.0},
                {"player": "Keeper", "position": "GK", "x": 6.0, "y": 34.0},
            ],
            "away": [
                {"player": "Xen", "position": "ST", "x": 40.0, "y": 30.0},
                {"player": "Yan", "position": "CB", "x": 18.0, "y": 22.0},
                {"player": "Zed", "position": "MID", "x": 22.0, "y": 40.0},
            ],
        },
        {
            "minute": 3,
            "home": [
                {"player": "Al", "position": "ST", "x": 90.0, "y": 28.0},
                {"player": "Bo", "position": "CM", "x": 60.0, "y": 34.0},
                {"player": "Keeper", "position": "GK", "x": 7.0, "y": 34.0},
                {"player": "Col", "position": "CM", "x": 50.0, "y": 40.0},
            ],
            "away": [
                {"player": "Xen", "position": "ST", "x": 45.0, "y": 25.0},
                {"player": "Yan", "position": "CB", "x": 16.0, "y": 24.0},
            ],
        },
    ]
    return ReplayData(
        home_team="Home FC", away_team="Away FC",
        home_goals=2, away_goals=1, home_possession=54.0,
        duration_s=180.0, ball=ball, frames=frames,
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_ball_interpolation_linear():
    r = _synthetic_replay()
    x, y = r.ball_at(45.0)   # halfway between 30s and 60s samples
    assert abs(x - 41.25) < 1e-6, x     # 30->52.5
    assert abs(y - 35.0) < 1e-6, y      # 36->34
    print(f"  PASS ball linear interp at t=45 -> ({x:.2f}, {y:.2f})")


def test_ball_clamped_to_endpoints():
    r = _synthetic_replay()
    assert r.ball_at(0.0) == (5.0, 34.0)
    assert r.ball_at(250.0) == (104.0, 30.0)
    print("  PASS ball clamps inside recorded path")


def test_player_interpolation_continuous():
    r = _synthetic_replay()
    # Al moves 70->80->90 along x over minutes 1-2-3 (t=60,120,180)
    xa_90, _ = r.side_at("home", 90.0)[0]["x"], r.side_at("home", 90.0)[0]["y"]
    mid = r.side_at("home", 150.0)  # between minute2 (120s) and minute3 (180s)
    al = next(p for p in mid if p["player"] == "Al")
    assert 80.0 < al["x"] < 90.0, al["x"]   # strictly between snapshots
    assert abs(al["x"] - 85.0) < 1e-3, al["x"]  # smoothstep midpoint ~ symmetric
    print(f"  PASS player smooth interp at t=150 -> Al x={al['x']:.2f}")


def test_sub_appears_only_after_first_sample():
    r = _synthetic_replay()
    t1 = r.side_at("home", 30.0)      # within minute 1
    t3 = r.side_at("home", 150.0)     # within minute 3 (col on at 3)
    assert all(p["player"] != "Col" for p in t1)
    assert any(p["player"] == "Col" for p in t3)
    print("  PASS sub Col absent at m1, present after minute 3 sample")


def test_sub_off_disappears_after_last_sample():
    r = _synthetic_replay()
    at_75 = r.side_at("away", 75.0)   # minute 2 frame (Zed still listed)
    at_160 = r.side_at("away", 160.0) # after minute 3 frame (Zed gone)
    assert any(p["player"] == "Zed" for p in at_75)
    assert all(p["player"] != "Zed" for p in at_160)
    print("  PASS sub-off Zed present at m2, gone after m3 sample")


def test_apply_subs_tightens_window():
    r = _synthetic_replay()
    # Engine says Col came on at exactly 130s (not the whole of minute 3).
    r.apply_subs({"home": [{"player": "Col", "on": 130.0}]})
    assert all(p["player"] != "Col" for p in r.side_at("home", 129.0))
    assert any(p["player"] == "Col" for p in r.side_at("home", 131.0))
    # Zed went off at exactly 115s even though his last frame is minute 2.
    r.apply_subs({"away": [{"player": "Zed", "off": 115.0}]})
    assert any(p["player"] == "Zed" for p in r.side_at("away", 114.0))
    assert all(p["player"] != "Zed" for p in r.side_at("away", 115.0))
    assert all(p["player"] != "Zed" for p in r.side_at("away", 116.0))
    # A swap at exactly 115s => 11 players, never 12.
    assert len(r.players_at(115.0)[1]) == 2  # Xen + Yan only
    print("  PASS apply_subs tightens on/off to exact instants")


def test_gk_and_roles_preserved():
    r = _synthetic_replay()
    home = r.side_at("home", 90.0)
    gk = next(p for p in home if p["position"] == "GK")
    assert gk["player"] == "Keeper"
    print("  PASS role labels preserved through interpolation")


def test_statsbomb_mapping():
    # Home centre spot -> 60,40 ; away mirrored x.
    sx, sy = to_statsbomb(52.5, 34.0, True)
    assert abs(sx - 60.0) < 1e-6 and abs(sy - 40.0) < 1e-6
    sx, sy = to_statsbomb(52.5, 34.0, False)
    assert abs(sx - 60.0) < 1e-6 and abs(sy - 40.0) < 1e-6
    sx, sy = to_statsbomb(90.0, 34.0, False)   # deep home side, away view
    assert abs(sx - (105.0 - 90.0) / 105.0 * SB_LEN) < 1e-6
    assert abs(sy - 34.0 / 68.0 * SB_WID) < 1e-6
    print("  PASS statsbomb mapping home/away mirror")


def test_json_roundtrip():
    r = _synthetic_replay()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "replay.json")
        r.save(path)
        r2 = ReplayData.load(path)
        assert r2.home_team == "Home FC"
        assert r2.away_goals == 1
        assert len(r2.ball) == len(r.ball)
        assert len(r2.frames) == len(r.frames)
        assert r2.subs == r.subs
    print("  PASS ReplayData JSON roundtrip")


def test_export_steps():
    r = _synthetic_replay()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "states.json")
        payload = export_interpolated(r, step=60.0, path=path, verbose=False)
        assert payload["n_states"] == 4   # t=0,60,120,180
        states = payload["states"]
        assert states[0]["score"] == [2, 1]
        assert states[0]["ball"]["x"] == 5.0
        assert states[3]["t"] == 180.0
        s2 = states[2]  # t=120 (minute 2 sample)
        assert any(p["player"] == "Col" for p in s2["home"])
    print("  PASS step export produces expected state list")


def main():
    tests = [
        test_ball_interpolation_linear,
        test_ball_clamped_to_endpoints,
        test_player_interpolation_continuous,
        test_sub_appears_only_after_first_sample,
        test_sub_off_disappears_after_last_sample,
        test_apply_subs_tightens_window,
        test_gk_and_roles_preserved,
        test_statsbomb_mapping,
        test_json_roundtrip,
        test_export_steps,
    ]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()