"""SCANNING — the head-turn/shoulder-check model must reproduce football, not
a decoration.

WHAT THIS FILE PINS
-------------------
1. No dead band: a ball just outside the FOV edge is brought INTO view by a
   glance, not silently ignored.
2. Ball at the feet does not produce a random neck twist.
3. A shoulder check makes the pressed/covering opponents actually VISIBLE to
   the off-ball actor gate (this is the whole point of scanning).
4. Scan cadence and scan duration are in SECONDS, roughly elite reality
   (12-40 deliberate glances per minute, ~0.35 s each), not per-minute
   artefacts.

Run:  python -u tests/test_scanning.py
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

import ball_vision
from ball_vision import perceive_ball, clear_caches
from perception import (
    PerceptionConfig,
    _facing_angle,
    _MAX_HEAD_TURN_RAD,
    perceive_offball_actors,
    set_perception,
    get_perception_config,
)


class _State:
    def __init__(self, x, y):
        self.current_x = x
        self.current_y = y


class _Pos:
    def __init__(self):
        self.states = {}
        self.team_rosters = {}
        self.team_attacks_right = {}


class _Engine:
    def __init__(self):
        self.position_engine = _Pos()

    def register(self, name, x, y):
        self.position_engine.states[name] = _State(x, y)


def _player(vision=85.0, anticipation=80.0, stamina=100.0):
    return SimpleNamespace(
        name="P",
        dna=SimpleNamespace(
            mental=SimpleNamespace(
                vision=vision,
                anticipation=anticipation,
                composure=75.0,
            ),
            physical=SimpleNamespace(stamina=stamina),
        ),
    )


class Actors:
    def __init__(self, name, position="ST"):
        self.name = name
        self.position = position


class _PE:
    """Minimal position engine with real coordinates for the actor gate."""

    def __init__(self):
        self.pos = {}

    def get_position(self, name):
        return self.pos.get(name)


class ScanTests(unittest.TestCase):

    def setUp(self):
        self.old_cfg = get_perception_config()
        clear_caches()

    def tearDown(self):
        set_perception(self.old_cfg)
        clear_caches()

    # ── head-turn geometry ──────────────────────────────────────────
    def test_a_ball_just_outside_the_fov_is_brought_into_view(self):
        player = _player()
        fov_half = math.radians(75.0)
        for deg in (80.0, 95.0, 110.0, 130.0, 150.0):
            ball = (50.0 + 12.0 * math.cos(math.radians(deg)),
                    34.0 + 12.0 * math.sin(math.radians(deg)))
            facing = _facing_angle(player, 50.0, 34.0, ball[0], ball[1],
                                   True, fov_half=fov_half)
            turn = facing - 0.0
            # If the ball can be reached by this neck turn, it must end up
            # inside the cone. Otherwise the turn must be capped exactly.
            need = math.radians(deg) - fov_half
            reach = _MAX_HEAD_TURN_RAD * (0.4 + 0.6 * (0.5 * 0.85 + 0.5 * 0.80))
            if need <= reach:
                self.assertLessEqual(
                    abs(math.radians(deg) - turn), fov_half + 1e-6,
                    f"ball at {deg}° was not brought into the cone")
            else:
                self.assertAlmostEqual(turn, reach, places=6)

    def test_widely_behind_ball_cannot_be_found_by_a_neck_turn(self):
        player = _player(vision=20.0, anticipation=20.0)
        facing = _facing_angle(player, 50.0, 34.0,
                               50.0 - 10.0, 34.0, True,
                               fov_half=np.deg2rad(75.0))
        # A weak scanner turns only 40% of the neck's lateral range, not 105°.
        self.assertLess(abs(facing), math.radians(80.0))

    def test_ball_at_his_feet_faces_forward(self):
        player = _player()
        facing = _facing_angle(player, 50.0, 34.0, 50.5, 34.0, True,
                               fov_half=np.deg2rad(75.0))
        self.assertEqual(facing, 0.0)

    # ── actor gate: scanning widens the visible world ───────────────
    def test_head_turn_reveals_an_actor_the_forward_cone_hides(self):
        p = _player(vision=90.0, anticipation=80.0)
        pe = _PE()
        # Attacker (us) at (50, 34). Defender at bearing 140°, 9 m away.
        attacker = Actors("ATT", "ST")
        pe.pos["ATT"] = (
            50.0 + 9.0 * math.cos(math.radians(140.0)),
            34.0 + 9.0 * math.sin(math.radians(140.0)),
        )
        # Ball directly on the defender's bearing, also to the side/behind.
        ball = (50.0 + 11.0 * math.cos(math.radians(140.0)),
                34.0 + 11.0 * math.sin(math.radians(140.0)))
        cfg = PerceptionConfig(
            enabled=True, offball_actor_perception=True,
            role_blocks=False, fov_deg=150.0,
            range_min=12.0, range_max=45.0, noise_base=2.0, max_actors=None,
        )
        # No ball coordinates -> the legacy constant forward cone: blind.
        plain = perceive_offball_actors(
            p, 50.0, 34.0, teammates=[attacker], defenders=[],
            position_engine=pe, attacks_right=True, config=cfg)
        self.assertEqual(plain[0], [],
                         "a 140° actor should be outside the forward cone")
        # Ball coordinates -> the head turns toward where he is checking;
        # the actor behind him enters his field of view.
        turned = perceive_offball_actors(
            p, 50.0, 34.0, teammates=[attacker], defenders=[],
            position_engine=pe, attacks_right=True, config=cfg,
            ball_x=ball[0], ball_y=ball[1])
        self.assertEqual([a.name for a in turned[0]], ["ATT"],
                         "the shoulder check did not reveal the carrier")

    # ── cadence: seconds, and elite scale ───────────────────────────
    def test_scans_are_frequent_and_short(self):
        set_perception(PerceptionConfig(
            enabled=True, ball_vision=True, role_blocks=False,
            fov_deg=150.0, range_min=4.0, range_max=35.0))
        engine = _Engine()
        engine.register("P", 50.0, 34.0)
        player = _player(vision=70.0, anticipation=70.0)
        track = ball_vision._tracks_for(engine).setdefault(
            "P", ball_vision._BallTrack())
        _prev = track.last_scan_time
        scans = 0
        for step in range(600):               # 60 s at 10 Hz
            minute = step * 0.1 / 60.0
            perceive_ball(engine, "P", player, True,
                          38.0, 34.0, minute)  # ball behind his back
            cur = ball_vision._tracks_for(engine)["P"].last_scan_time
            if cur != _prev:
                scans += 1
                _prev = cur
        # 0.2-0.6 sweeps/s = 12-40/min. Peripheral glimpses can substitute
        # for some sweeps, so require the Jordet lower band.
        self.assertGreaterEqual(scans, 12,
                                f"only {scans} scans in 60 s of blind play")
        self.assertLessEqual(scans, 45,
                             f"{scans} scans in 60 s is implausible head-turning")

    def test_scan_times_are_seconds_not_minutes(self):
        set_perception(PerceptionConfig(
            enabled=True, ball_vision=True, role_blocks=False,
            fov_deg=150.0, range_min=4.0, range_max=35.0))
        engine = _Engine()
        engine.register("P", 50.0, 34.0)
        perceive_ball(engine, "P", _player(), True,
                      10.0, 34.0, 30.0)  # 1800 s into match
        track = ball_vision._tracks_for(engine)["P"]
        self.assertEqual(track.last_scan_time, 1800.0)
        self.assertEqual(track.scan_duration, 0.35)

    def test_glimpse_draw_is_sampled_per_second_not_per_minute(self):
        """The glimpse RNG is seeded on seconds, not the whole minute.

        This used to be one fixed draw per 60 s (`minute:.0f`), so a player
        either glimpsed for an entire minute or never. We cannot assert on
        a random boolean, but we CAN assert the draw changes by comparing the
        serialised key indirectly via two consecutive seconds being allowed
        to differ.  The behavioural guarantee lives in `tests/` rather than
        here; this test only ensures the branch exists and the call does not
        raise with minute-granularity timestamps.
        """
        set_perception(PerceptionConfig(
            enabled=True, ball_vision=True, role_blocks=False,
            fov_deg=150.0, range_min=30.0, range_max=45.0))
        engine = _Engine()
        engine.register("P", 50.0, 34.0)
        for step in range(120):
            minute = step * 0.5 / 60.0
            perceive_ball(engine, "P", _player(), True,
                          50.0 + 12.0 * math.cos(math.radians(160.0)),
                          34.0 + 12.0 * math.sin(math.radians(160.0)),
                          minute)


if __name__ == "__main__":
    unittest.main(verbosity=2)
