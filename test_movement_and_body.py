import math
import random
import unittest

from geometry_engine import (
    MovingPlayer, Vec2, resolve_body_duel, resolve_dribble,
    _body_contest_win_probability, body_balance, body_mass,
    turn_speed_factor, _race_motion,
)
from possession_physics import PossessionEpisode


def player(name, x, y, pace=70, acceleration=5.5, **kw):
    kw.setdefault("reaction_time", 0.18)
    return MovingPlayer(name, Vec2(x, y), pace, acceleration, **kw)


class TimeToReachMovementCostsTests(unittest.TestCase):
    def setUp(self):
        # pace 70 -> top 5.0 + 70*0.042 = 7.94 m/s; accel 5.5; reaction 0.18
        self.p = player("runner", 0, 0, pace=70, acceleration=5.5)
        self.target = Vec2(40, 0)

    def _legacy_arrival(self):
        p, t = self.p, self.target
        remaining = t.x - p.position.x - p.control_radius
        accel = max(1.5, p.acceleration)
        top = 5.0 + p.pace * 0.042
        t_top = top / accel
        d_top = 0.5 * accel * t_top * t_top
        if remaining <= d_top:
            return p.reaction_time + math.sqrt(2.0 * remaining / accel)
        return p.reaction_time + t_top + (remaining - d_top) / top

    def test_flat_no_context_is_legacy_behavior(self):
        self.assertAlmostEqual(self.p.time_to_reach(self.target), self._legacy_arrival(), places=6)

    def test_race_motion_without_context_identical(self):
        self.assertEqual(_race_motion(self.p, self.target), self.p.time_to_reach(self.target))

    def test_aligned_heading_no_penalty(self):
        t0 = self.p.time_to_reach(self.target)
        th = self.p.time_to_reach(self.target, heading=Vec2(1, 0))
        self.assertAlmostEqual(th, t0, places=6)

    def test_wrong_way_turn_pays_penalty(self):
        aligned = self.p.time_to_reach(self.target, heading=Vec2(1, 0))
        wrong = self.p.time_to_reach(self.target, heading=Vec2(-1, 0))
        self.assertGreater(wrong, aligned)

    def test_ninety_degree_cut_cheaper_than_full_reverse(self):
        cut = self.p.time_to_reach(self.target, heading=Vec2(0, 1))
        reverse = self.p.time_to_reach(self.target, heading=Vec2(-1, 0))
        self.assertGreater(reverse, cut)
        # 90° factor ~0.70, 180° factor ~0.40 -> 70%-pace cutoff arrives first
        self.assertAlmostEqual(turn_speed_factor(math.pi / 2), 0.7, places=3)
        self.assertAlmostEqual(turn_speed_factor(math.pi), 0.4, places=3)

    def test_entry_speed_momentum_carry(self):
        from_rest = self.p.time_to_reach(self.target)
        sprinting = self.p.time_to_reach(self.target, entry_speed=self.p.top_speed)
        self.assertGreater(from_rest, sprinting)
        self.assertAlmostEqual(sprinting, self.p.reaction_time + (40 - self.p.control_radius) / self.p.top_speed,
                               places=6)

    def test_braking_when_entry_exceeds_turn_limit(self):
        # Full-reverse turn caps sustainable top at 0.40*top = 3.176; entering
        # at 8 m/s means braking before the new sprint. Must be SLOWER than a
        # wrong-way chase that started from rest.
        from_rest = self.p.time_to_reach(self.target, heading=Vec2(-1, 0))
        braked = self.p.time_to_reach(self.target, heading=Vec2(-1, 0), entry_speed=8.0)
        self.assertGreater(braked, from_rest)
        decel = min(6.0, self.p.acceleration * 1.8)
        top_cut = 0.40 * self.p.top_speed
        expected_brake = (8.0 - top_cut) / decel
        self.assertAlmostEqual(
            braked, self.p.reaction_time + expected_brake + (40 - self.p.control_radius) / top_cut,
            places=6,
        )

    def test_turn_speed_factor_monotone(self):
        f = [turn_speed_factor(t / 200.0 * math.pi) for t in range(10, 201, 10)]
        self.assertTrue(all(b <= a for a, b in zip(f, f[1:])))
        self.assertEqual(turn_speed_factor(0.0), 1.0)
        self.assertGreaterEqual(turn_speed_factor(0.2), 1.0)


class BodyPrimitiveTests(unittest.TestCase):
    def test_body_mass_range(self):
        self.assertEqual(body_mass(0), 55.0)
        self.assertEqual(body_mass(100), 95.0)
        self.assertAlmostEqual(body_mass(50), 75.0, places=6)

    def test_body_balance_range(self):
        self.assertAlmostEqual(body_balance(0, 0, 0), 0.30, places=6)
        self.assertAlmostEqual(body_balance(100, 100, 100), 0.95, places=6)

    def _mvp(self, name, mass, balance, pace=70, accel=5.5):
        return player(name, 0, 0, pace=pace, acceleration=accel,
                      body_mass_kg=mass, balance=balance)

    def test_tank_defender_bowls_featherweight(self):
        d = self._mvp("tank", 95, 0.95, pace=80, accel=7.0)
        c = self._mvp("feather", 56, 0.30, pace=50, accel=4.5)
        duel = resolve_body_duel(d, c, 5.0, 3.4)
        self.assertEqual(duel.winner, "defender")
        self.assertGreaterEqual(duel.impulse_ratio, 1.30)

    def test_strong_carrier_rides_challenge(self):
        d = self._mvp("feather", 56, 0.30, pace=80, accel=7.0)
        c = self._mvp("tank", 95, 0.95, pace=70, accel=5.5)
        duel = resolve_body_duel(d, c, 5.0, 5.4)
        self.assertEqual(duel.winner, "carrier")
        self.assertLessEqual(duel.impulse_ratio, 0.78)

    def test_equal_bodies_leave_contest(self):
        d = self._mvp("a", 76, 0.5)
        c = self._mvp("b", 76, 0.5)
        duel = resolve_body_duel(d, c, 5.0, 5.0)
        self.assertEqual(duel.winner, "contest")
        self.assertAlmostEqual(duel.impulse_ratio, 1.0, places=6)

    def test_contest_probability_monotone_bounded(self):
        self.assertAlmostEqual(_body_contest_win_probability(1.0), 0.5, places=6)
        self.assertEqual(_body_contest_win_probability(0.0), 0.15)
        self.assertEqual(_body_contest_win_probability(10.0), 0.80)
        self.assertLess(_body_contest_win_probability(0.9), 0.5)
        self.assertGreater(_body_contest_win_probability(1.1), 0.5)


class DribbleBodyTests(unittest.TestCase):
    def _mvp_xyz(self, name, x, body_mass_kg, balance, pace=70, accel=5.5):
        return player(name, x, 34, pace=pace, acceleration=accel,
                      body_mass_kg=body_mass_kg, balance=balance)

    def test_tank_defender_bowls_featherweight_over(self):
        # Near-simultaneous contact (defender sits close to the first steps):
        # a 95 kg, high-balance presser decides the challenge physically.
        defender = self._mvp_xyz("tank_d", 42, 95, 0.95, pace=90, accel=8.0)
        carrier = self._mvp_xyz("feather_c", 40, 56, 0.30, pace=40, accel=4.0)
        result = resolve_dribble(
            Vec2(40, 34), Vec2(50, 34), carrier, [defender],
            rng=random.Random(7),
        )
        self.assertEqual(result.outcome, "tackled")
        self.assertEqual(result.resolution_note, "bowled_over")

    def test_featherweight_defender_shrugged_off_by_strong_carrier(self):
        defender = self._mvp_xyz("feather_d", 42, 56, 0.30, pace=90, accel=7.0)
        carrier = self._mvp_xyz("tank_c", 40, 95, 0.95, pace=70, accel=5.5)
        result = resolve_dribble(
            Vec2(40, 34), Vec2(50, 34), carrier, [defender],
            rng=random.Random(7),
        )
        self.assertEqual(result.outcome, "retained")

    def test_even_physique_leaves_weighted_contest(self):
        defender = self._mvp_xyz("even_d", 42, 76, 0.5)
        carrier = self._mvp_xyz("even_c", 40, 76, 0.5)
        outcomes, notes = [], set()
        for seed in range(60):
            result = resolve_dribble(
                Vec2(40, 34), Vec2(50, 34), carrier, [defender],
                rng=random.Random(seed),
            )
            outcomes.append(result.outcome)
            notes.add(result.resolution_note)
        # Equal physique ties resolve in the impulse-weighted gray zone: the
        # roll splits, never a physique slam.
        self.assertIn("tackled", outcomes)
        self.assertIn("retained", outcomes)
        self.assertIn("50_50", notes)
        self.assertNotIn("bowled_over", notes)

    def test_clean_early_beat_still_reach_win(self):
        # A defender who cleanly beats the ball to the envelope (no simultaneous
        # contact) wins on reach regardless of physique.
        feather = self._mvp_xyz("early_d", 47, 56, 0.30, pace=80, accel=7.0)
        carrier = self._mvp_xyz("tank_c", 40, 95, 0.95)
        result = resolve_dribble(
            Vec2(40, 34), Vec2(50, 34), carrier, [feather],
            rng=random.Random(1),
        )
        self.assertEqual(result.outcome, "tackled")
        self.assertEqual(result.resolution_note, "reached_first")


class LooseBallBodyTests(unittest.TestCase):
    def _ep(self):
        return PossessionEpisode()

    def _adjacent(self, name, mass, balance, x=49.0):
        return player(name, x, 34, pace=75, acceleration=6.5,
                      body_mass_kg=mass, balance=balance)

    def test_weighted_loose_ball_tank_presser_wins_50_50(self):
        ep = self._ep()
        att = self._adjacent("light_att", 56, 0.30)
        dfd = self._adjacent("tank_def", 95, 0.95)
        # Both reach at the same instant -> body decides for the presser.
        self.assertEqual(ep.resolve_loose_ball(50, 34, [att], [dfd]), "defence")

    def test_weighted_loose_ball_strong_receiver_rides(self):
        ep = self._ep()
        att = self._adjacent("tank_att", 95, 0.95)
        dfd = self._adjacent("light_def", 56, 0.30)
        self.assertEqual(ep.resolve_loose_ball(50, 34, [att], [dfd]), "attack")

    def test_non_tie_loose_ball_is_pure_race(self):
        ep = self._ep()
        att = self._adjacent("far_att", 95, 0.95, x=40.0)
        dfd = self._adjacent("near_def", 56, 0.30, x=49.0)
        self.assertEqual(ep.resolve_loose_ball(50, 34, [att], [dfd]), "defence")


class DuelContestBodyTests(unittest.TestCase):
    def _mvp(self, name, mass, balance, x=60.0, y=35.0):
        return player(name, x, y, pace=70, acceleration=6.0,
                      body_mass_kg=mass, balance=balance)

    def test_tank_presser_bowls_carrier_off(self):
        ep = PossessionEpisode()
        attacker = self._mvp("small_c", 56, 0.30)
        challenger = self._mvp("tank_p", 95, 0.95, x=58.5, y=34.5)
        self.assertFalse(ep.resolve_duel_contest(attacker, challenger, 60.0, 35.0))

    def test_strong_carrier_holds_against_light_press(self):
        ep = PossessionEpisode()
        attacker = self._mvp("tank_c", 95, 0.95)
        challenger = self._mvp("light_p", 56, 0.30, x=58.5, y=34.5)
        self.assertTrue(ep.resolve_duel_contest(attacker, challenger, 60.0, 35.0))


class VelocityMemoryTests(unittest.TestCase):
    def test_tracked_motion_feeds_race_context(self):
        ep = PossessionEpisode()
        ep._track_motion("A", 0.0, 0.0, 0.0)
        ep._track_motion("A", 0.8, 0.0, 0.1)
        heading, speed = ep._velocity_of("A")
        self.assertAlmostEqual(speed, 8.0, places=6)
        self.assertAlmostEqual(heading.x, 1.0, places=6)
        self.assertAlmostEqual(heading.y, 0.0, places=6)

        mp = player("A", 0, 0, pace=70, acceleration=5.5)
        ctx = ep._build_race_context([mp])
        self.assertIsNotNone(ctx)
        h, s = ctx[id(mp)]
        self.assertAlmostEqual(s, 8.0, places=6)

    def test_rest_and_dead_band_yield_no_context(self):
        ep = PossessionEpisode()
        ep._track_motion("A", 0.0, 0.0, 0.0)
        ep._track_motion("A", 0.02, 0.0, 0.1)  # 0.2 m/s shuffle shot
        mp = player("A", 0, 0)
        self.assertIsNone(ep._build_race_context([mp]))

    def test_teleport_row_does_not_poison_velocity(self):
        ep = PossessionEpisode()
        ep._track_motion("A", 0.0, 0.0, 0.0)
        ep._track_motion("A", 50.0, 0.0, 0.1)  # 500 m/s isn't real motion
        ep._track_motion("A", 50.8, 0.0, 0.2)
        heading, speed = ep._velocity_of("A")
        self.assertAlmostEqual(speed, 8.0, places=6)


if __name__ == "__main__":
    unittest.main()