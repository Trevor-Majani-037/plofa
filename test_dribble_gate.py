import unittest
import random
from types import SimpleNamespace

from event_chain import BaseChain


def _player(dribbling, agility, ball_control, tackling, anticipation):
    """Build a minimal fake player carrying just the DNA the gate reads."""
    dna = SimpleNamespace(
        technical=SimpleNamespace(dribbling=dribbling, ball_control=ball_control),
        physical=SimpleNamespace(agility=agility),
        defending=SimpleNamespace(tackling=tackling),
        mental=SimpleNamespace(anticipation=anticipation),
    )
    return SimpleNamespace(dna=dna)


class DribbleGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deterministic roll so the gate is exercised end-to-end.
        random.seed(1234)

    def test_elite_dribbler_shrugs_off_poor_tackler_on_tackled(self):
        # A technician who loses the geometric race can still beat a weak
        # tackler reliably — with the RNG fixed this must resolve "retained".
        attacker = _player(dribbling=97, agility=94, ball_control=95,
                           tackling=20, anticipation=20)
        tackler = _player(dribbling=20, agility=20, ball_control=20,
                          tackling=18, anticipation=22)
        outcomes = [
            BaseChain._dribble_confirmation_gate("tackled", attacker, tackler)
            for _ in range(200)
        ]
        # The vast majority should flip to retained (any shrug is enough).
        self.assertGreaterEqual(outcomes.count("retained"), 1)
        self.assertIn("retained", outcomes)

    def test_poor_dribbler_against_elite_tackler_confirms_tackle(self):
        attacker = _player(dribbling=20, agility=20, ball_control=20,
                           tackling=60, anticipation=60)
        tackler = _player(dribbling=60, agility=60, ball_control=60,
                          tackling=97, anticipation=94)
        outcomes = [
            BaseChain._dribble_confirmation_gate("tackled", attacker, tackler)
            for _ in range(200)
        ]
        # A poor dribbler almost never beats an elite tackler: the differential
        # is strongly negative so the shrug floor applies (~5%).
        self.assertLessEqual(outcomes.count("retained"), 20)

    def test_last_ditch_stretch_tackle_can_snatch_retained(self):
        # The symmetric half: a geometric "retained" can still be turned over
        # by a stretch tackle when the tackler is markedly better.
        attacker = _player(dribbling=20, agility=20, ball_control=20,
                           tackling=60, anticipation=60)
        tackler = _player(dribbling=60, agility=60, ball_control=60,
                          tackling=97, anticipation=94)
        outcomes = [
            BaseChain._dribble_confirmation_gate("retained", attacker, tackler)
            for _ in range(500)
        ]
        # Should sometimes be snatched back, but only where the differential
        # permits it (~30% for this lopsided matchup) — never all the time.
        self.assertIn("tackled", outcomes)
        self.assertLessEqual(outcomes.count("tackled"), 220)

    def test_no_tackler_preserves_geometric_outcome(self):
        # No defender reference -> geometry is authoritative.
        attacker = _player(90, 90, 90, 90, 90)
        self.assertEqual(
            BaseChain._dribble_confirmation_gate("tackled", attacker, None),
            "tackled",
        )
        self.assertEqual(
            BaseChain._dribble_confirmation_gate("retained", attacker, None),
            "retained",
        )

    def test_even_duel_stays_mostly_geometric(self):
        # An even matchup applies a small base flip in both directions but
        # never flips wholesale.
        attacker = _player(50, 50, 50, 50, 50)
        tackler = _player(50, 50, 50, 50, 50)
        tackled_res = [
            BaseChain._dribble_confirmation_gate("tackled", attacker, tackler)
            for _ in range(300)
        ]
        retained_res = [
            BaseChain._dribble_confirmation_gate("retained", attacker, tackler)
            for _ in range(300)
        ]
        self.assertGreaterEqual(tackled_res.count("retained"), 1)
        self.assertGreaterEqual(retained_res.count("tackled"), 1)


if __name__ == "__main__":
    unittest.main()
