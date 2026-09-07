import unittest

from geometry_engine import (
    BallSpin, MovingPlayer, Vec2, Vec3, BALL_ROLLING_DECEL,
    make_flight, make_ballistic_flight, resolve_ground_pass,
    rolling_arrival_time, rolling_speed, rolling_position,
)
from possession_physics import delivery_spin, rolling_decel_for
from weather_physics import WeatherPhysics


def player(name, x, y, pace=70, acceleration=5.5):
    return MovingPlayer(name, Vec2(x, y), pace, acceleration, reaction_time=0.18)


class MagnusSpinTests(unittest.TestCase):
    """Ball #1 — spin / Magnus effect on the ballistic long ball."""

    def _flight(self, spin=None, rate=40.0):
        if spin is None:
            spin = BallSpin.side(rate)
        return make_ballistic_flight(
            Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 22.0, spin=spin,
        )

    def test_side_spin_bulges_mid_flight_and_lands_on_target(self):
        # A curled ball bends off the chord at mid-flight, yet the launcher
        # aimed for the curled landing point — so the deviation is zero at
        # both endpoints and (anti-symmetric) peaks in the middle.
        flight = self._flight()
        self.assertTrue(flight.spin.active)
        self.assertEqual(flight.spin.kind, "side")

        mid = flight.position_at(flight.duration / 2)
        self.assertGreater(abs(mid.y - 34.0), 0.4)  # visible banana bend

        end = flight.position_at(flight.duration)
        self.assertLess(abs(end.x - 70.0), 0.5)     # endpoint deviation is zero
        self.assertLess(abs(end.y - 34.0), 0.5)

    def test_topspin_sinks_apex_and_backspin_floats_apex(self):
        # Magnus vertical acceleration gives topspin an effective gravity
        # ABOVE 9.81 and backspin one BELOW it — so, for the same kick,
        # backspin must peak higher than a dead ball and topspin lower.
        plain = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 22.0)
        top = self._flight(spin=BallSpin.topspin(40.0))
        back = self._flight(spin=BallSpin.backspin(40.0))
        self.assertGreater(back.apex_z, plain.apex_z)
        self.assertLess(top.apex_z, plain.apex_z)
        for flight in (top, back):
            end = flight.position_at(flight.duration)
            self.assertLess(abs(end.x - 70.0), 0.5)
            self.assertLess(abs(end.y - 34.0), 0.5)

    def test_parametric_flight_accepts_spin_too(self):
        flight = make_flight(
            Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 20.0, apex_z=4.0,
            spin=BallSpin.side(46.0),
        )
        mid = flight.position_at(flight.duration / 2)
        self.assertGreater(abs(mid.y - 34.0), 0.3)
        end = flight.position_at(flight.duration)
        self.assertLess(abs(end.x - 70.0), 0.5)
        self.assertLess(abs(end.y - 34.0), 0.5)

    def test_dead_ball_stays_straight(self):
        flight = make_ballistic_flight(Vec3(30, 34, 0.05), Vec3(70, 34, 1.0), 22.0)
        self.assertIsNone(flight.spin)
        mid = flight.position_at(flight.duration / 2)
        self.assertAlmostEqual(mid.y, 34.0, places=2)


class RollingFrictionTests(unittest.TestCase):
    """Ball #2 — ground friction: passes decay over distance."""

    def test_rolling_arrival_matches_quadratic_solution(self):
        # d = v0*t - 0.5*k*t^2 -> the earlier root for a 40 m ball at 20 m/s
        # and the dry-turf baseline 1.2 m/s^2.
        expected = (20.0 - (20.0 ** 2 - 2 * BALL_ROLLING_DECEL * 40.0) ** 0.5) / BALL_ROLLING_DECEL
        self.assertAlmostEqual(rolling_arrival_time(40.0, 20.0, BALL_ROLLING_DECEL), expected, places=6)

    def test_unreachable_roll_returns_none(self):
        # 20 m/s stops after d = v0^2/(2k) ≈ 167 m — 200 m is undeliverable,
        # so the ball dies short rather than slide forever.
        self.assertIsNone(rolling_arrival_time(200.0, 20.0, BALL_ROLLING_DECEL))

    def test_rolling_speed_ever_decreases_never_below_zero(self):
        for t in (0.0, 1.0, 5.0, 30.0):
            self.assertGreaterEqual(rolling_speed(20.0, BALL_ROLLING_DECEL, t), 0.0)
        self.assertLess(rolling_speed(20.0, BALL_ROLLING_DECEL, 1.0), 20.0)

    def test_rolling_never_past_rest_point(self):
        rest = rolling_position(20.0, BALL_ROLLING_DECEL, 100.0)
        self.assertAlmostEqual(rest, 20.0 ** 2 / (2 * BALL_ROLLING_DECEL), places=3)

    def test_open_pass_arrives_slower_than_constant_speed_model(self):
        result = resolve_ground_pass(
            Vec2(30, 34), Vec2(70, 34),
            player("receiver", 68, 34), [], 20.0,
            rolling_decel=BALL_ROLLING_DECEL,
        )
        self.assertEqual(result.outcome, "received")
        self.assertFalse(result.ball_stopped_short)
        naive = 40.0 / 20.0
        self.assertGreater(result.ball_travel_time, naive)
        self.assertLess(result.ball_speed_at_arrival, 20.0)
        self.assertAlmostEqual(
            result.ball_speed_at_arrival,
            max(0.0, 20.0 - BALL_ROLLING_DECEL * result.ball_travel_time),
            places=1,
        )

    def test_under_struck_ball_dies_short_of_target(self):
        # 8 m/s with dry-turf friction rolls only ~27 m; hitting a 70 m
        # through ball must genuinely underhit and stop short.
        result = resolve_ground_pass(
            Vec2(30, 34), Vec2(100, 34),
            player("receiver", 98, 34), [], 8.0,
            rolling_decel=BALL_ROLLING_DECEL,
        )
        self.assertEqual(result.outcome, "underhit")
        self.assertTrue(result.ball_stopped_short)
        self.assertLess(result.contact_point.x, 60.0)


class WeatherRollingTests(unittest.TestCase):
    def setUp(self):
        WeatherPhysics.set_active_weather("clear", enabled=False)

    def tearDown(self):
        WeatherPhysics.set_active_weather("clear", enabled=False)

    def test_neutral_when_weather_disabled(self):
        self.assertEqual(WeatherPhysics.rolling_decel_mult("clear"), 1.0)
        self.assertEqual(rolling_decel_for("rain"), BALL_ROLLING_DECEL)

    def test_wet_turf_grips_more_never_more_than_45pct(self):
        WeatherPhysics.set_active_weather("rain", enabled=True)
        mult = WeatherPhysics.rolling_decel_mult("rain")
        self.assertGreater(mult, 1.0)
        self.assertLessEqual(mult, 1.45)
        self.assertAlmostEqual(rolling_decel_for("rain"), BALL_ROLLING_DECEL * mult, places=6)


class DeliverySpinTests(unittest.TestCase):
    def test_skill_sets_rotation_and_poor_technicians_hit_dry_ball(self):
        dead = delivery_spin(0.0)
        self.assertFalse(dead.active)
        elite = delivery_spin(90.0)
        self.assertTrue(elite.active)
        # rate = 12 + 90*0.58*uniform(0.7, 1.35) -> bounded by the extremes.
        self.assertGreater(elite.rate, 12.0 + 90.0 * 0.58 * 0.7 - 0.5)
        self.assertLessEqual(elite.rate, 12.0 + 90.0 * 0.58 * 1.35 + 0.5)

    def test_kinds_map_to_vertical_family(self):
        self.assertEqual(delivery_spin(70.0, kind="back").kind, "back")
        self.assertEqual(delivery_spin(70.0, kind="top").kind, "top")
        self.assertEqual(delivery_spin(70.0, kind="side").kind, "side")


if __name__ == "__main__":
    unittest.main()