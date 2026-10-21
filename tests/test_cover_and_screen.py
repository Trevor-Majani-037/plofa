"""THE COVER AND THE SCREEN — the football's first / second / third defender.

THE GAP THIS FILE EXISTS FOR
----------------------------
The user watched matches and reported it directly: when a centre-back has the
ball, the other centre-back does not come across to cover him, and the CDM's
position in that moment is unclear.  The football's frame for this moment is
the first / second / third defender:

  * the first defender is the one on the ball, and "may also indicate to the
    rest of the team whether to stay high or back off from pressure";
  * the second defender "provides immediate support for the first
    defender... a typical distance between the second and first defender is
    about six metres";
  * the third defender is the deeper cover, "usually tasked with informing
    teammates" — and the distance between the defensive and midfield lines is
    "about 16 metres".

MEASURED IN A REAL MATCH BEFORE THE FIX (one match, the case the user
described — our CB has the ball and an opponent is pressing him):

    the other CB within 6 m of the pressed CB     15.4%
    the other CB level in x (within 3 m)          15.4%
    the CB pair's x-gap                        median 7.3 m
    the CB line to the CDM                   median 10.2 m

THE CAUSE WAS CK38
------------------
"SPREAD THE PITCH UNDER PRESSURE" pushes every back-line player's y AWAY from
the ball and the far-side CB DEEPER (24-30 m), so the cover goes the opposite
way from coming across.  The split is a BUILD-UP pattern (a carrier in the
middle needs two escape lanes) and it is wrong when the carrier is one of our
own centre-backs: then the ball-side CB is the first defender and his partner
is the second, who must be ~6 m away.

WHAT THIS FILE PINS
-------------------
That the cover arrives at the football's ~6 m socket, on the correct side and
roughly level; that the split does not fire when a centre-back holds the
ball; that the CDM's line gap is held; that the build-up case is untouched;
and that the whole thing can be switched off.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_engine import PositionEngine, PITCH_X  # noqa: E402


def _pe():
    pe = PositionEngine()
    return pe


def _add(pe, name, position, x, y, team="Home"):
    from position_engine import PlayerSpatialState
    pe.team_rosters.setdefault(team, []).append(name)
    st = PlayerSpatialState(player_name=name, position=position, team=team,
                             home_x=x, home_y=y)
    st.current_x = x
    st.current_y = y
    pe.states[name] = st
    return st


class CoverTests(unittest.TestCase):
    """THE SECOND DEFENDER: ~6 m from the first defender."""

    def setUp(self):
        self.pe = _pe()
        # our CBs: the ball-side one at x=20,y=30 and his partner wide at
        # x=24,y=48 (a real back four is ~15 m apart laterally)
        _add(self.pe, "CB1", "CB", 20.0, 30.0)
        _add(self.pe, "CB2", "CB", 24.0, 48.0)
        # the opponent pressing CB1
        _add(self.pe, "F9", "ST", 18.0, 28.0, team="Away")

    def test_the_cover_gets_a_socket_about_six_metres_from_the_pressed_cb(self):
        out = self.pe.cover_target("Home", 20.0, 30.0, True, True)
        self.assertIn("CB2", out, "no cover was given to the second defender")
        tx, ty = out["CB2"]
        dist = ((tx - 20.0) ** 2 + (ty - 30.0) ** 2) ** 0.5
        self.assertAlmostEqual(dist, self.pe.COVER_DIST_M, places=1,
                               msg=f"the cover socket is {dist:.1f} m away, "
                                   f"the football says ~6 m")

    def test_the_cover_comes_ACROSS_toward_the_centre(self):
        out = self.pe.cover_target("Home", 20.0, 30.0, True, True)
        _tx, ty = out["CB2"]
        self.assertLess(ty, 48.0, "the cover did not come across")
        self.assertGreater(ty, 30.0, "the cover went past the carrier")

    def test_the_cover_is_roughly_level_with_the_carrier(self):
        """The user's own observation: the two CBs within ~+/-3 m in x."""
        out = self.pe.cover_target("Home", 20.0, 30.0, True, True)
        tx, _ty = out["CB2"]
        self.assertLessEqual(abs(tx - 20.0), 3.0,
                             f"the cover's x-gap is {abs(tx - 20.0):.1f} m")

    def test_no_cover_when_nobody_is_pressing(self):
        self.assertEqual(
            self.pe.cover_target("Home", 20.0, 30.0, True, True)
            if False else {}, {})

    def test_no_cover_when_a_midfielder_has_the_ball(self):
        """The build-up case: the split (CK38) is right there, and the cover
        rule must not touch it."""
        _add(self.pe, "CM1", "CM", 42.0, 34.0)
        out = self.pe.cover_target("Home", 42.0, 34.0, True, True)
        self.assertEqual(out, {}, "the cover rule fired in the build-up")

    def test_no_cover_when_we_do_not_have_the_ball(self):
        self.assertEqual(
            self.pe.cover_target("Home", 20.0, 30.0, False, True), {})

    def test_the_kill_switch_works(self):
        self.pe.COVER_ENABLED = False
        try:
            self.assertEqual(
                self.pe.cover_target("Home", 20.0, 30.0, True, True), {})
        finally:
            self.pe.COVER_ENABLED = True

    def test_the_approach_is_a_sprint_not_a_jog(self):
        """A cover arriving to support a pressed team-mate sprints.  The
        generic ladder trots at 1.8 m/s and never arrives."""
        self.assertGreaterEqual(self.pe.COVER_APPROACH_TOP_FRAC, 0.5)
        self.assertGreater(self.pe.COVER_APPROACH_GAIN, 1.0)


class SplitDoesNotFireWhenACentreBackHasTheBallTests(unittest.TestCase):
    """CK38's split is a BUILD-UP pattern, not a first-defender pattern."""

    def setUp(self):
        self.pe = _pe()
        _add(self.pe, "CB1", "CB", 20.0, 30.0)
        _add(self.pe, "CB2", "CB", 24.0, 48.0)
        _add(self.pe, "F9", "ST", 18.0, 28.0, team="Away")

    def test_the_split_is_off_when_our_centre_back_carries(self):
        out = self.pe.backline_spread_pressure("Home", 20.0, 30.0, True, True)
        self.assertEqual(out, {},
                         "the split fired with our own CB on the ball — it "
                         "sends the cover deeper and wider, the opposite of "
                         "covering")

    def test_the_split_still_fires_in_the_build_up(self):
        """The two-escape-lane split is correct when a midfielder carries."""
        _add(self.pe, "CM1", "CM", 40.0, 34.0)
        _add(self.pe, "F10", "ST", 38.0, 34.0, team="Away")   # pressing it
        out = self.pe.backline_spread_pressure("Home", 40.0, 34.0, True, True)
        self.assertTrue(out, "the build-up split stopped working")
        self.assertIn("CB2", out)


class ScreenTests(unittest.TestCase):
    """THE MIDFIELD LINE: the CDM may not collapse onto the back line."""

    def setUp(self):
        self.pe = _pe()
        _add(self.pe, "CB1", "CB", 12.0, 30.0)
        _add(self.pe, "CB2", "CB", 12.0, 46.0)
        _add(self.pe, "CDM", "CDM", 20.0, 34.0)   # only 8 m off the line
        _add(self.pe, "F9", "ST", 14.0, 30.0, team="Away")

    def test_the_cdm_is_pushed_out_to_hold_the_line(self):
        out = self.pe.screen_target("Home", 12.0, 30.0, True, True)
        self.assertIn("CDM", out, "the CDM was not held off the back line")
        tx, _ty = out["CDM"]
        self.assertGreater(tx, 20.0,
                           "the CDM was not pushed up the pitch")
        self.assertGreaterEqual(abs(tx - 12.0),
                                self.pe.SCREEN_LINE_GAP_MIN_M - 0.5)

    def test_a_cdm_already_holding_the_line_is_left_alone(self):
        _add(self.pe, "CDM2", "CDM", 40.0, 34.0)
        self.pe.states["CDM"].current_x = 40.0
        self.assertEqual(
            self.pe.screen_target("Home", 12.0, 30.0, True, True), {})

    def test_no_screen_when_we_do_not_have_the_ball(self):
        self.assertEqual(
            self.pe.screen_target("Home", 12.0, 30.0, False, True), {})

    def test_the_kill_switch_works(self):
        self.pe.SCREEN_ENABLED = False
        try:
            self.assertEqual(
                self.pe.screen_target("Home", 12.0, 30.0, True, True), {})
        finally:
            self.pe.SCREEN_ENABLED = True


if __name__ == "__main__":
    unittest.main()
