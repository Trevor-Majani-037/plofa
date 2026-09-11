import unittest

from geometry_engine import (
    MovingPlayer, Vec2, Vec3, make_flight, make_ballistic_flight,
    resolve_aerial_delivery, resolve_aerial_pass,
    resolve_dribble, resolve_ground_pass, resolve_shot,
)


def player(name, x, y, pace=70, acceleration=5.5):
    return MovingPlayer(name, Vec2(x, y), pace, acceleration, reaction_time=0.18)


class GeometryEngineTests(unittest.TestCase):
    def test_open_pass_reaches_receiver(self):
        result = resolve_ground_pass(
            Vec2(30, 34), Vec2(48, 34), player("receiver", 46, 34), [], 15.0
        )
        self.assertEqual(result.outcome, "received")

    def test_defender_on_lane_intercepts_before_receiver(self):
        defender = player("defender", 38, 34, pace=82, acceleration=7.0)
        result = resolve_ground_pass(
            Vec2(30, 34), Vec2(52, 34), player("receiver", 52, 34), [defender], 12.0
        )
        self.assertEqual(result.outcome, "intercepted")
        self.assertIs(result.interceptor.player, defender.player)

    def test_dribble_is_tackled_when_defender_can_reach_carrier(self):
        # Defender sits on the escape lane well ahead of the ball, so he
        # cleanly beats the ball to the tackle envelope (big arrival margin)
        # — a pure reach win, deterministic. (A defender who only just catches
        # the carrier simultaneously now triggers the #4 body duel instead.)
        defender = player("defender", 47, 34, pace=80, acceleration=7.0)
        result = resolve_dribble(
            Vec2(40, 34), Vec2(50, 34), player("attacker", 40, 34), [defender]
        )
        self.assertEqual(result.outcome, "tackled")
        self.assertEqual(result.resolution_note, "reached_first")

    def test_aerial_delivery_awards_first_reachable_contact(self):
        flight = make_flight(Vec3(80, 6, 0), Vec3(96, 34, 1.4), 18.0, apex_z=5.5)
        defender = player("defender", 96, 34, pace=78, acceleration=6.5)
        result = resolve_aerial_delivery(flight, [], [defender])
        self.assertEqual(result.winner.player, defender.player)
        self.assertIn(result.outcome, {"controlled", "contested"})

    def test_shot_is_saved_when_keeper_reaches_goal_plane(self):
        keeper = player("keeper", 105, 34, pace=68, acceleration=5.5)
        flight = make_flight(Vec3(95, 34, 0.2), Vec3(105, 34, 1.1), 20.0, apex_z=1.2)
        result = resolve_shot(flight, keeper)
        self.assertEqual(result.outcome, "saved")

    def test_shot_striking_post_is_woodwork(self):
        flight = make_flight(Vec3(94, 34, 0.2), Vec3(105, 30.34, 1.0), 20.0, apex_z=1.1)
        result = resolve_shot(flight, None)
        self.assertEqual(result.outcome, "woodwork")


class AerialPassTests(unittest.TestCase):
    def test_open_lofted_pass_drops_to_receiver(self):
        flight = make_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 20.0, apex_z=4.0)
        result = resolve_aerial_pass(flight, player("receiver", 68, 34), [])
        self.assertEqual(result.outcome, "received")
        self.assertTrue(result.winner_is_receiver)

    def test_defender_wins_landing_duel_is_intercepted(self):
        # Receiver is far away; a defender sits right on the landing point and
        # wins the jump. Outcome must be "intercepted" with the defender as winner.
        flight = make_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 18.0, apex_z=4.0)
        result = resolve_aerial_pass(
            flight, player("receiver", 12, 34),
            [player("defender", 68, 34, pace=90, acceleration=7.0)],
        )
        self.assertEqual(result.outcome, "intercepted")
        self.assertFalse(result.winner_is_receiver)
        self.assertEqual(result.winner.player, "defender")

    def test_long_ball_out_of_reach_is_underhit(self):
        # Nobody is near the distant landing point -> the ball drops loose.
        flight = make_flight(Vec3(30, 34, 0.05), Vec3(95, 34, 1.0), 40.0, apex_z=6.0)
        result = resolve_aerial_pass(flight, player("receiver", 50, 34), [])
        self.assertEqual(result.outcome, "underhit")

    def test_aerial_pass_traces_3d_ball_path(self):
        flight = make_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 20.0, apex_z=4.0)
        result = resolve_aerial_pass(flight, player("receiver", 68, 34), [])
        self.assertTrue(result.ball_trajectory)
        # The arc must rise above both endpoints (a genuine 3D flight, not a
        # flat 2D line) — check every mid-flight sample is airborne.
        for t, v3 in result.ball_trajectory:
            if 0 < t < result.ball_travel_time:
                self.assertGreater(v3.z, 0.4)


class BallisticFlightTests(unittest.TestCase):
    def test_trajectory_lands_on_target(self):
        flight = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 22.0)
        self.assertTrue(flight.ballistic)
        end = flight.position_at(flight.duration)
        self.assertLess(abs(end.x - 70.0), 0.5)
        self.assertLess(end.z - 1.0, 0.5)

    def test_harder_strike_reaches_target_while_soft_hoof_falls_short(self):
        # A 14 m/s kick from 34->68 m needs loft to arrive; a 30 m/s driven
        # ball crosses comfortably and stays flatter. Both come out of
        # physics, not tuned apex numbers: the harder strike flies faster,
        # lower and arrives sooner.
        short = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(68, 40, 1.0), 14.0)
        long = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(68, 40, 1.0), 30.0)
        self.assertLess(long.duration, short.duration)
        self.assertLess(long.apex_z, short.apex_z)

    def test_gravity_shapes_peak_height(self):
        # Peak height is DERIVED from the launch velocity vector under g≈9.81,
        # not chosen. A launch whose vertical component maxes out higher must
        # produce a higher apex for the same g.
        g = 9.81
        flight = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 22.0, gravity=g)
        v0z = flight.launch_speed * __import__("math").sin(flight.launch_angle_rad)
        expected_apex = 0.05 + v0z * v0z / (2.0 * g)
        self.assertAlmostEqual(flight.apex_z, expected_apex, delta=0.3)

    def test_unreachable_kick_drops_short_as_underhit(self):
        # 12 m/s can't span 55 m of pitch — the ballistic flight must end well
        # short of the target instead of teleporting (the underhit fall, not a
        # telescoping hoof).
        flight = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(85, 40, 1.0), 12.0)
        end = flight.position_at(flight.duration)
        self.assertLess(end.x - 30.0, 48.0)
        self.assertGreater(flight.apex_z, 1.0)


class BallisticAerialPassTests(unittest.TestCase):
    def _moving(self, name, x, y, pace=70, accel=5.5, reaction=0.18, jumping=50):
        return MovingPlayer(
            name, Vec2(x, y), pace, accel, reaction_time=reaction,
            jump_height=0.35 + jumping * 0.0045,
            standing_reach=1.55 + jumping * 0.0025,
        )

    def test_ballistic_open_long_ball_reaches_receiver(self):
        flight = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.2), 22.0)
        receiver = self._moving("receiver", 68, 34, jumping=70)
        result = resolve_aerial_pass(flight, receiver, [])
        self.assertEqual(result.outcome, "received")
        self.assertTrue(result.winner_is_receiver)
        self.assertTrue(result.ball_trajectory)


if __name__ == "__main__":
    unittest.main()
