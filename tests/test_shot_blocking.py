"""
Shot-blocking responsiveness.

A defender does not stand frozen where he happened to be when a shot is
struck. The instant a shot is launched the defence reacts — the nearest
players slide onto the shot's line before it reaches them. These tests pin
down that "predict / position for the block" behaviour at the physics layer:
a blocker a metre or two off the shot corridor must be able to close and
get his body in the way, while a properly-frozen blocker (zero reaction)
stays out and lets the shot through.

It also pins the TEAM-level shot-block alignment in the position engine:
when the ball sits in the box at high danger, a near-side unmarked defender
steps goal-side onto the ball-to-goal shirt line — the defence "predicts and
positions for a block" before the trigger is pulled.
"""
import unittest

from geometry_engine import (
    MovingPlayer, Vec2, Vec3, make_flight, resolve_shot,
)
from position_engine import PositionEngine


def blocker(name, x, y, pace=75, accel=5.5, balance=0.5, reaction=0.18):
    # A full-capacity upheaval of the default MovingPlayer kinematic state,
    # mirroring how event_chain._moving_player builds one from DNA.
    return MovingPlayer(
        name, Vec2(x, y), pace, accel, reaction_time=reaction,
        balance=balance, control_radius=1.05, tackle_radius=1.25,
        standing_reach=1.75,
    )


class ShotBlockingTests(unittest.TestCase):
    def test_defender_already_on_lane_blocks(self):
        # Central shot fired from ~14m. A defender sits goal-side on the
        # shirt line between shooter and goal (the position defensive_block
        # pulls him into BEFORE a shot). Because the ball then travels down
        # exactly his line, he gets the block. Under the old frozen-blocker
        # code this already worked; the point is that this is the baseline
        # shot-block the team positioning now sets up.
        flight = make_flight(Vec3(91, 34, 0.2), Vec3(105, 34, 1.0), 20.0, apex_z=1.2)
        keeper = blocker("keeper", 105, 34, pace=60, accel=4.0, reaction=0.20)
        defender = blocker("cb", 95, 34, pace=80, accel=6.0, balance=0.6, reaction=0.12)

        result = resolve_shot(flight, keeper, [defender], attacks_right=True)

        self.assertEqual(result.outcome, "blocked")
        self.assertIs(result.blocker.player, defender.player)

    def test_close_off_lane_defender_closes_onto_shot(self):
        # A defender JUST off the lane (0.8 m) closes the residual gap in the
        # short time the ball is near him, getting his body on the line. This
        # is the "closing" that a frozen blocker could not do — without the
        # responsiveness fix this test would sail through and be saved.
        flight = make_flight(Vec3(90, 34, 0.2), Vec3(105, 34, 1.2), 16.0, apex_z=1.4)
        keeper = blocker("keeper", 105, 34, pace=60, accel=4.0, reaction=0.25)
        defender = blocker("cb", 93, 33.2, pace=82, accel=6.5, balance=0.65, reaction=0.10)

        result = resolve_shot(flight, keeper, [defender], attacks_right=True)

        # Slower (16 m/s) shot that passes within reach of a reacting, close
        # defender -> the defence frames it.
        self.assertEqual(result.outcome, "blocked")
        self.assertIs(result.blocker.player, defender.player)

    def test_far_defender_cannot_teleport_into_block(self):
        # A defender 12 m away cannot materialise onto the ball in a ~0.5 s
        # low drive — his closing speed is physically bounded, so a drill
        # through the corridor stays unblocked (reaches the keeper).
        flight = make_flight(Vec3(90, 34, 0.2), Vec3(105, 34, 1.0), 26.0, apex_z=1.1)
        keeper = blocker("keeper", 105, 34, pace=60, accel=4.0, reaction=0.25)
        far = blocker("far", 90, 22, pace=75, accel=5.5, reaction=0.18)

        result = resolve_shot(flight, keeper, [far], attacks_right=True)

        self.assertEqual(result.outcome, "saved")
        self.assertIsNone(result.blocker)

    def test_zero_reaction_defender_does_not_close(self):
        # Sanity: if a blocker genuinely has no time to react (effectively a
        # frozen player), the responsiveness path cannot fire and the shot
        # passes to the keeper. This guards the teleport regression — a
        # defender must not cheat his reach forward.
        flight = make_flight(Vec3(90, 34, 0.2), Vec3(105, 34, 1.2), 16.0, apex_z=1.4)
        keeper = blocker("keeper", 105, 34, pace=60, accel=4.0, reaction=0.25)
        # reaction_time 1.0 s >> flight duration => never reacts.
        frozen = blocker("frozen", 93, 33.2, pace=82, accel=6.5, reaction=1.0)

        result = resolve_shot(flight, keeper, [frozen], attacks_right=True)

        self.assertEqual(result.outcome, "saved")
        self.assertIsNone(result.blocker)


class _FakeProfile:
    def __init__(self, defensive_line=0.5, width=0.5, tempo=0.5,
                 directness=0.5, press_intensity=0.5):
        self.defensive_line = defensive_line
        self.width = width
        self.tempo = tempo
        self.directness = directness
        self.press_intensity = press_intensity


class _FakeDNA:
    def __init__(self, specialties=(), geometric_awareness=50.0):
        self.specialties = list(specialties)

        class _Mental:
            pass

        mental = _Mental()
        mental.geometric_awareness = geometric_awareness
        self.mental = mental


class _FakePlayer:
    def __init__(self, name, position, specialties=(), geometric_awareness=50.0):
        self.name = name
        self.position = position
        self.dna = _FakeDNA(specialties, geometric_awareness)


def _block_engine_team():
    """Defensive unit spread across the back line, defending own 105.0 goal."""
    players = [
        _FakePlayer("LB", "LB"), _FakePlayer("CB1", "CB"),
        _FakePlayer("CB2", "CB"), _FakePlayer("RB", "RB"),
        _FakePlayer("CDM", "CDM"), _FakePlayer("GK", "GK"),
    ]
    pe = PositionEngine()
    pe.initialize_team("Test FC", players, _FakeProfile(), attacks_right=False)
    for n, x, y in (("LB", 92.0, 13.0), ("CB1", 91.0, 26.0),
                    ("CB2", 91.0, 44.0), ("RB", 92.0, 57.0),
                    ("CDM", 84.0, 34.0), ("GK", 103.0, 34.0)):
        pe.states[n].current_x = x
        pe.states[n].current_y = y
    return pe


class ShotBlockAlignmentTests(unittest.TestCase):
    def test_near_side_defender_steps_onto_shot_lane_in_box(self):
        # Ball deep in the box (86m vs the 105m goal line), at HIGH danger.
        # A near-side free CB must be pulled goal-side onto the ball-to-goal
        # shirt line — a committed shot-block stance that CLOSES the shot
        # (moves up toward the shooter/ball) rather than holding the deep
        # offside line. The result must put the defender between the ball and
        # goal, closer to the ball.
        for cb in ("CB1", "CB2"):
            pe = _block_engine_team()
            state = pe.states[cb]
            ball_x, ball_y = 86.0, state.current_y
            state.current_y = ball_y
            x0 = state.current_x
            pe.defensive_block(
                "Test FC", ball_x=ball_x, ball_y=ball_y,
                own_goal_x=105.0, danger_level=85.0, pull_strength=0.5,
                defensive_line=0.5,
            )
            new_x = state.current_x
            # Defenders block by stepping goal-side and CLOSING the shooter:
            # they move up (x toward the ball at 86.0, i.e. a lower x than the
            # deep line they were holding at ~91).
            self.assertLess(new_x, x0,
                            f"{cb} must close the shot lane toward the ball")
            # Still goal-side of the ball (between ball and own 105 goal).
            self.assertGreater(new_x, ball_x,
                               f"{cb} must stay goal-side of the shot")

    def test_low_danger_no_shot_block_alignment(self):
        # Below the shot-block danger threshold the near-side CB is NOT
        # yanked onto the shirt line; it reverts to the normal line-holding
        # block (a deep step toward the goal-side offside line).
        for cb in ("CB1", "CB2"):
            pe = _block_engine_team()
            state = pe.states[cb]
            ball_x, ball_y = 86.0, state.current_y
            state.current_y = ball_y
            x0 = state.current_x
            pe.defensive_block(
                "Test FC", ball_x=ball_x, ball_y=ball_y,
                own_goal_x=105.0, danger_level=30.0, pull_strength=0.5,
                defensive_line=0.5,
            )
            new_x = state.current_x
            # At moderate danger the block is a deep line, so x should not
            # close hard onto the 86m shooter lane.
            self.assertGreaterEqual(new_x + 1.0, x0,
                                    f"{cb} must not close the shot lane at low danger")


if __name__ == "__main__":
    unittest.main()

