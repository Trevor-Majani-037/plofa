"""
Through-ball opportunity gating.

A through ball is a pass DELIVERED THROUGH THE DEFENSIVE LINE (the seam
between CB/RB, CB-CB or LB-CB) into the SPACE BEHIND that line, ball reaching
a runner who is running onto it and bypassing everyone. It is an
OPPORTUNITY-DRIVEN action, not a flat coin-flip: it should only be attempted
when there is a genuine open channel to thread and a forward runner in space
past the line. These tests pin the `_through_ball_opportunity` scorer that
drives that gate.
"""
import unittest

from event_chain import PossessionChain


class _PE:
    """Minimal position-engine stand-in exposing get_position only."""

    def __init__(self, positions):
        self._pos = positions

    def get_position(self, name):
        return self._pos[name]


class _P:
    def __init__(self, name, position):
        self.name = name
        self.position = position


def scorer(passer, receiver, x, y, attacks_right, def_players, pe):
    return PossessionChain._through_ball_opportunity(
        passer, receiver, x, y, attacks_right, def_players, pe)


class ThroughBallOpportunityTests(unittest.TestCase):
    def _four_four_two(self, line_x=55.0):
        """Back four holding a line at line_x attacking RIGHT."""
        return [
            _P("rb", "RB"), _P("rcb", "CB"),
            _P("lcb", "CB"), _P("lb", "LB"),
        ]

    def test_open_channel_and_receiver_beyond_line_is_high(self):
        # Ball deep in the attacking half; the back four is spread wide with a
        # big central seam; the striker is already beyond the line in the gap.
        # -> a clear through-ball corridor.
        pe = _PE({
            "rb": (55.0, 6.0), "rcb": (55.0, 27.0),
            "lcb": (55.0, 41.0), "lb": (55.0, 62.0),
            "st": (68.0, 34.0),
        })
        passer, receiver = _P("cam", "CAM"), _P("st", "ST")
        opp = scorer(passer, receiver, 48.0, 34.0, True,
                     self._four_four_two(), pe)
        self.assertGreater(opp, 0.55)

    def test_tight_locked_block_suppresses_opportunity(self):
        # Back line packed tight (no real seam — gaps ~2-3m, below the
        # threadable threshold) and the striker camped on the closed line.
        # -> no channel, no space to run into.
        pe = _PE({
            "rb": (55.0, 27.0), "rcb": (55.0, 31.0),
            "lcb": (55.0, 35.0), "lb": (55.0, 39.0),
            "st": (55.0, 34.0),
        })
        passer, receiver = _P("cam", "CAM"), _P("st", "ST")
        opp = scorer(passer, receiver, 48.0, 34.0, True,
                     self._four_four_two(), pe)
        self.assertLess(opp, 0.45)

    def test_no_space_behind_line_is_low(self):
        # Wide seam in the line, but the striker is standing ON/behind the
        # offside line (no window to run into) -> low behind factor.
        pe = _PE({
            "rb": (55.0, 6.0), "rcb": (55.0, 27.0),
            "lcb": (55.0, 41.0), "lb": (55.0, 62.0),
            "st": (56.0, 34.0),
        })
        passer, receiver = _P("cam", "CAM"), _P("st", "ST")
        opp = scorer(passer, receiver, 48.0, 34.0, True,
                     self._four_four_two(), pe)
        # Seam is big but the receiver has ~1m behind the line; the combined
        # opportunity stays clearly below the open-channel case above.
        self.assertLess(opp, 0.6)
        self.assertGreater(opp, 0.0)

    def test_neutral_when_no_position_engine(self):
        # No spatial data -> the scorer stays neutral so the tendency/skill
        # weighting remains the arbiter downstream.
        passer, receiver = _P("cam", "CAM"), _P("st", "ST")
        opp = scorer(passer, receiver, 48.0, 34.0, True,
                     self._four_four_two(), None)
        self.assertEqual(opp, 0.5)

    def test_empty_defenders_neutral(self):
        pe = _PE({})
        passer, receiver = _P("cam", "CAM"), _P("st", "ST")
        opp = scorer(passer, receiver, 48.0, 34.0, True, [], pe)
        self.assertEqual(opp, 0.5)


if __name__ == "__main__":
    unittest.main()
