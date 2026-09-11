import unittest
import random
from types import SimpleNamespace

from event_chain import BaseChain


def _passer(long_passing):
    """Minimal fake passer carrying only the DNA the gate reads."""
    dna = SimpleNamespace(
        passing=SimpleNamespace(long_passing=long_passing),
    )
    return SimpleNamespace(dna=dna)


def _marker(tackling, anticipation):
    """Minimal fake marker carrying only the DNA the gate reads."""
    dna = SimpleNamespace(
        defending=SimpleNamespace(tackling=tackling),
        mental=SimpleNamespace(anticipation=anticipation),
    )
    return SimpleNamespace(dna=dna)


class AerialPassGateTests(unittest.TestCase):
    """DECISION regression tests: long balls DO carry the Checkpoint-28
    attribute gate (wall/geometry-first, attributes-second), matching shots,
    ground passes and dribbles. Geometry stays the primary authority; the
    gate only flips the contested ``intercepted`` case."""

    @classmethod
    def setUpClass(cls):
        random.seed(1234)

    def test_geometric_received_outcome_passes_through_untouched(self):
        result = BaseChain._aerial_pass_gate("received", None, _passer(80))
        self.assertEqual(result, "received")

    def test_underhit_is_never_revived_by_skill(self):
        # A ball nobody could physically reach is geometry-pure: no skill on
        # either side resurrects it into a completion.
        result = BaseChain._aerial_pass_gate("underhit", None, _passer(95))
        self.assertEqual(result, "underhit")

    def test_no_marker_keeps_geometric_interception(self):
        result = BaseChain._aerial_pass_gate("intercepted", None, _passer(80))
        self.assertEqual(result, "intercepted")

    def test_elite_marker_confirms_interception(self):
        marker = _marker(tackling=97, anticipation=94)
        outcomes = [
            BaseChain._aerial_pass_gate("intercepted", marker, _passer(45))
            for _ in range(200)
        ]
        # marker int_skill = (0.4*97 + 0.6*94)/100 = 0.952, passer 0.45 ->
        # intercept_prob = 0.15 + 0.502*0.35 ~= 0.33. A clearly superior
        # marker should convert a large share of his geometric arrivals.
        self.assertGreaterEqual(outcomes.count("intercepted"), 30)

    def test_world_class_long_passer_can_drop_it_onto_receiver(self):
        marker = _marker(tackling=50, anticipation=50)
        outcomes = [
            BaseChain._aerial_pass_gate("intercepted", marker, _passer(97))
            for _ in range(500)
        ]
        # marker int_skill = 0.50, passer 0.97 -> intercept_prob clamps to
        # the 0.08 floor, so ~92% of geometric interceptions are beaten by
        # the elite delivery.
        self.assertGreaterEqual(outcomes.count("received"), 350)


if __name__ == "__main__":
    unittest.main()