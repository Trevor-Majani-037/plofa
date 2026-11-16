"""THE STRIKER PRESS — the centre-back duel the league is built on.

THE ASK (2026-11-04)
--------------------
"Make strikers the most pressed.  The two CBs should have this at the back of
their heads 99% of the time: press the striker.  This league demands it, like
the EPL — yes strikers score, but you have to battle it out with your two
opponent CBs."

WHAT WAS THERE BEFORE
---------------------
Nothing.  The live shape chain gives every CB a depth slot from a preset
(``defensive_block_target``) and compacts it toward the ball.  The opposing
centre-forward is not in the target at all.  ``MarkingEngine.assign`` does
man-mark, but it runs inside ``PositionEngine.defensive_block``, which only
fires with the ball in the defending half AND danger >= 25, and which writes
into the same goal-side band the block already occupies.

WHAT THIS FILE PINS
-------------------
That the CB pair takes the opposing striker as its man, that the marker
stands GOAL-SIDE at a standoff that closes as the ball gets playable, that
the cover is the other CB and sits deeper on the same line, that the two are
never handed the same socket, and that the whole thing yields when the team
has the ball, when the ball cannot reach the striker, and when it is switched
off.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_engine import PositionEngine, PITCH_X  # noqa: E402

GOAL_Y = 34.0


def _pe():
    return PositionEngine()


def _add(pe, name, position, x, y, team="Home"):
    from position_engine import PlayerSpatialState
    pe.team_rosters.setdefault(team, []).append(name)
    st = PlayerSpatialState(player_name=name, position=position, team=team,
                            home_x=x, home_y=y)
    st.current_x = x
    st.current_y = y
    pe.states[name] = st
    return st


def _dist_to(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class StrikerPressTests(unittest.TestCase):
    """Our two CBs against their centre-forward."""

    def setUp(self):
        self.pe = _pe()
        # Home attacks right, so Home DEFENDS the goal at x=0.
        _add(self.pe, "CB1", "CB", 20.0, 30.0)
        _add(self.pe, "CB2", "CB", 24.0, 48.0)
        _add(self.pe, "ST9", "ST", 18.0, 28.0, team="Away")
        self.ball_at_feet = (18.0, 28.0)

    # ── the duel itself ────────────────────────────────────────────
    def test_the_nearest_cb_takes_the_striker(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        self.assertIn("CB1", out, "the man on the striker is not the near CB")
        self.assertIn("CB2", out, "the pair left the cover unstaffed")

    def test_the_marker_is_goal_side_and_touch_tight(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        tx, ty = out["CB1"]
        striker = self.ball_at_feet
        d = _dist_to((tx, ty), striker)
        self.assertAlmostEqual(
            d, self.pe.STRIKER_PRESS_TIGHT_M, places=1,
            msg=f"marker stands {d:.1f} m off the striker with the ball at his "
                f"feet; tight is {self.pe.STRIKER_PRESS_TIGHT_M} m")
        # Goal-side: nearer the defended goal (x=0) than the striker is.
        self.assertLess(tx, striker[0],
                        "the marker is not between the striker and the goal")

    def test_the_marker_stands_on_the_line_to_the_middle_of_the_goal(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        tx, ty = out["CB1"]
        sx, sy = self.ball_at_feet
        # The vector striker->marker must point at the goal's centre line,
        # i.e. its y must be moving toward 34, not away from it.
        self.assertGreater(ty, sy,
                           "the marker showed the striker toward the goal, "
                           "not away from it")
        # Perpendicular error from the striker->goal-centre line, in metres.
        vx, vy = 0.0 - sx, GOAL_Y - sy
        norm = math.hypot(vx, vy)
        ux, uy = vx / norm, vy / norm
        px, py = tx - sx, ty - sy
        perp = abs(px * uy - py * ux)
        self.assertLess(perp, 0.05, f"marker is {perp:.2f} m off the line")

    def test_the_standoff_closes_as_the_ball_gets_playable(self):
        near = self.pe.striker_press_target(
            "Home", 18.0, 28.0, has_ball=False, attacks_right=True)
        far = self.pe.striker_press_target(
            "Home", 18.0, 55.0, has_ball=False, attacks_right=True)
        near_d = _dist_to(near["CB1"], self.ball_at_feet)
        far_d = _dist_to(far["CB1"], self.ball_at_feet)
        self.assertLess(near_d, far_d,
                        "the CB stands as tight on a ball 27 m away as on one "
                        "at the striker's feet")

    def test_far_ball_still_earns_the_mark(self):
        """The '99% of the time' rule: no free striker just because the ball is
        far away."""
        out = self.pe.striker_press_target(
            "Home", 90.0, 10.0, has_ball=False, attacks_right=True)
        self.assertIn("CB1", out,
                      "a ball 75+ m away was allowed to free the striker")

    def test_the_cover_is_the_other_cb_and_sits_deeper(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        marker = _dist_to(out["CB1"], self.ball_at_feet)
        cover = _dist_to(out["CB2"], self.ball_at_feet)
        self.assertGreater(
            cover, marker,
            "the cover is not deeper than the marker; the pair is flat")
        self.assertAlmostEqual(
            cover - marker, self.pe.STRIKER_PRESS_COVER_GAP_M
            + self.pe.STRIKER_PRESS_COVER_SPINE_M * 0.27, delta=1.2,
            msg=f"cover {cover:.1f} m vs marker {marker:.1f} m")

    def test_the_two_cbs_never_get_the_same_socket(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        self.assertNotEqual(out["CB1"], out["CB2"],
                            "both CBs were handed the same target")

    # ── yields ────────────────────────────────────────────────────
    def test_no_press_in_possession(self):
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=True, attacks_right=True)
        self.assertEqual(out, {}, "the CB pair pressed while we had the ball")

    def test_a_single_cb_still_marks_him(self):
        del self.pe.team_rosters["Home"][1]
        del self.pe.states["CB2"]
        out = self.pe.striker_press_target(
            "Home", *self.ball_at_feet, has_ball=False, attacks_right=True)
        self.assertEqual(list(out), ["CB1"])

    def test_no_cbs_means_no_assignment(self):
        pe = _pe()
        _add(pe, "ST9", "ST", 18.0, 28.0, team="Away")
        self.assertEqual(pe.striker_press_target(
            "Home", 18.0, 28.0, has_ball=False, attacks_right=True), {})

    def test_a_false_nine_without_a_ST_still_gets_picked_up(self):
        pe = _pe()
        _add(pe, "CB1", "CB", 20.0, 30.0)
        _add(pe, "CB2", "CB", 24.0, 46.0)
        _add(pe, "CM8", "CM", 45.0, 34.0, team="Away")
        _add(pe, "LW11", "LW", 16.0, 12.0, team="Away")
        out = pe.striker_press_target("Home", 16.0, 12.0, has_ball=False,
                                      attacks_right=True)
        self.assertIn("CB1", out,
                      "with no ST/CF/SS on the pitch nobody picked up the most "
                      "advanced opponent")

    def test_it_mirrors_when_we_attack_left(self):
        pe = _pe()
        # Home defends the goal at x=105.
        _add(pe, "CB1", "CB", 85.0, 30.0)
        _add(pe, "CB2", "CB", 81.0, 48.0)
        _add(pe, "ST9", "ST", 87.0, 28.0, team="Away")
        out = pe.striker_press_target("Home", 87.0, 28.0, has_ball=False,
                                      attacks_right=False)
        tx, ty = out["CB1"]
        self.assertGreater(tx, 87.0,
                           "the marker is on the wrong side of the striker when "
                           "attacking left")
        self.assertAlmostEqual(_dist_to((tx, ty), (87.0, 28.0)),
                               pe.STRIKER_PRESS_TIGHT_M, places=1)

    def test_the_kill_switch_works(self):
        self.pe.STRIKER_PRESS_ENABLED = False
        try:
            self.assertEqual(self.pe.striker_press_target(
                "Home", *self.ball_at_feet, has_ball=False,
                attacks_right=True), {})
        finally:
            self.pe.STRIKER_PRESS_ENABLED = True

    def test_the_approach_is_a_sprint_not_a_jog(self):
        """The generic ladder trots; the duel has to be reached."""
        self.assertGreaterEqual(self.pe.STRIKER_PRESS_TOP_FRAC, 0.7)
        self.assertGreater(self.pe.STRIKER_PRESS_GAIN, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
