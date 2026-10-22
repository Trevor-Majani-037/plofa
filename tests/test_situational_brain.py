"""THE SITUATIONAL BRAIN — attention, recurrence, memory and a plan.

Run:  .venv\\Scripts\\python.exe -m pytest tests/test_situational_brain.py -q
   or: .venv\\Scripts\\python.exe tests/test_situational_brain.py

WHAT THIS FILE IS FOR
---------------------
The project has a recurring pathology: a mechanism that exists, is wired
to something, and never runs — or runs with a wrong gradient and still
looks like it is training.  Four capabilities were added to the on-ball
brain (attention, recurrence, memory, planning), and this file is the gate
that says each one is REAL:

  * the backward pass matches FINITE DIFFERENCES (a hand-written backward
    pass that is wrong does not crash — it just trains badly);
  * a padded token takes NO attention mass;
  * the recurrent state ADVANCES between touches, and the same scene with
    a carried state can decide differently;
  * the memory block changes when the player remembers something, and is
    zero when he does not;
  * the plan head proposes and the team plan adopts it WITH HYSTERESIS;
  * the SHIPPED path is untouched — an ordinary FootballBrain must not
    route into any of this.

And the file that matters most is ``TheBackwardPassIsCorrectTests``: the
three bugs that check caught (a sign in the GRU's update gate, the short
softmax Jacobian, and a transposed weight-gradient orientation that square
matrices hid) were all silent.  A gradient check that has never failed is
not a check.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from situational_brain import (
    EMBED_D, HIDDEN_D, MEMORY_D, N_MAX, OUT_D, PLAN_D, PLAN_OUT_D, ROLE_D,
    STATE_D, TOKEN_D, SituationalBrain,
)


def _softmax_rows(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fixture(seed=7, n_steps=4, n_tok=3):
    rng = np.random.default_rng(seed)
    brain = SituationalBrain.random(seed=seed)
    tokens = rng.normal(size=(n_steps, n_tok, TOKEN_D))
    mask = np.ones((n_steps, n_tok))
    if n_tok > 1:
        mask[-1, -1] = 0.0
    state = rng.normal(size=(n_steps, STATE_D))
    role = rng.normal(size=(n_steps, ROLE_D))
    memory = rng.normal(size=(n_steps, MEMORY_D))
    plan = rng.normal(size=(n_steps, PLAN_D))
    h0 = np.zeros(HIDDEN_D)
    y = rng.integers(0, OUT_D, size=n_steps)
    py = rng.integers(0, PLAN_OUT_D, size=n_steps)
    return brain, tokens, mask, state, role, memory, plan, h0, y, py


def _loss(brain, tokens, mask, state, role, memory, plan, h0, y, py):
    logits, plan_logits, _h, _c = brain.forward_sequence(
        tokens, mask, state, role, memory, plan, h0)
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    nll = -(z[np.arange(len(y)), y] - lse)
    zp = plan_logits - plan_logits.max(axis=1, keepdims=True)
    lpse = np.log(np.exp(zp).sum(axis=1))
    pnll = -(zp[np.arange(len(py)), py] - lpse)
    return float((nll + pnll).sum())


class TheBackwardPassIsCorrectTests(unittest.TestCase):
    """FINITE DIFFERENCES.  The load-bearing class in this file."""

    EPS = 1e-6

    def test_the_backward_pass_matches_finite_differences(self):
        brain, tokens, mask, state, role, memory, plan, h0, y, py = _fixture()
        logits, plan_logits, _h, caches = brain.forward_sequence(
            tokens, mask, state, role, memory, plan, h0)
        # The CE gradient w.r.t. the logits is (probs - onehot), NOT
        # (logits - onehot).
        probs = _softmax_rows(logits)
        dlogits = probs.copy()
        dlogits[np.arange(len(y)), y] -= 1.0
        pprobs = _softmax_rows(plan_logits)
        dplan = pprobs.copy()
        dplan[np.arange(len(py)), py] -= 1.0
        grads = brain.backward(dlogits, dplan, caches)

        rng = np.random.default_rng(99)
        for name in sorted(brain.w):
            flat = brain.w[name].reshape(-1)
            picks = rng.choice(len(flat), size=min(6, len(flat)), replace=False)
            for idx in picks:
                original = float(flat[idx])
                flat[idx] = original + self.EPS
                lp = _loss(brain, tokens, mask, state, role, memory, plan,
                            h0, y, py)
                flat[idx] = original - self.EPS
                lm = _loss(brain, tokens, mask, state, role, memory, plan,
                            h0, y, py)
                flat[idx] = original
                numeric = (lp - lm) / (2.0 * self.EPS)
                analytic = float(grads[name].reshape(-1)[idx])
                denom = max(1e-8, abs(numeric) + abs(analytic))
                rel = abs(numeric - analytic) / denom
                self.assertLess(
                    rel, 1e-4,
                    f"{name}[{idx}]: numeric {numeric:+.6e} vs analytic "
                    f"{analytic:+.6e} (relative {rel:.2e})")

    def test_a_masked_token_takes_no_attention_mass(self):
        brain, tokens, mask, state, role, memory, plan, h0, y, py = _fixture()
        _logits, _pl, _h, caches = brain.forward_sequence(
            tokens, mask, state, role, memory, plan, h0)
        for c in caches:
            self.assertGreaterEqual(float(c["alpha"].min()), -1e-12)
            self.assertAlmostEqual(float(c["alpha"].sum()), 1.0, places=9)
            for j, m in enumerate(c["mask"]):
                if m <= 0.0:
                    self.assertEqual(float(c["alpha"][j]), 0.0)

    def test_no_entities_at_all_gives_a_zero_context_and_no_nan(self):
        brain = SituationalBrain.random(seed=3)
        tokens = np.zeros((N_MAX, TOKEN_D))
        mask = np.zeros(N_MAX)
        logits, plan_logits, h, _c = brain.forward(
            tokens, mask, np.zeros(STATE_D), np.zeros(ROLE_D),
            np.zeros(MEMORY_D), np.zeros(PLAN_D), np.zeros(HIDDEN_D))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertTrue(np.all(np.isfinite(plan_logits)))
        self.assertTrue(np.all(np.isfinite(h)))


class TheRecurrenceIsRealTests(unittest.TestCase):
    """The hidden state must actually change the decision."""

    def test_the_state_advances_and_the_output_moves(self):
        brain = SituationalBrain.random(seed=11)
        tokens = np.random.default_rng(1).normal(size=(N_MAX, TOKEN_D))
        mask = np.ones(N_MAX)
        state = np.random.default_rng(2).normal(size=STATE_D)
        role = np.zeros(ROLE_D)
        mem = np.zeros(MEMORY_D)
        pl = np.zeros(PLAN_D)

        h0 = np.zeros(HIDDEN_D)
        l1, _p1, h1, _c = brain.forward(tokens, mask, state, role, mem, pl, h0)
        l2, _p2, h2, _c = brain.forward(tokens, mask, state, role, mem, pl, h1)

        self.assertFalse(np.allclose(h0, h1), "the state must move")
        self.assertFalse(np.allclose(h1, h2), "the state must keep moving")
        self.assertGreater(float(np.abs(l1 - l2).max()), 1e-6,
                           "the same scene with a carried state must be able "
                           "to decide differently")

    def test_the_same_state_gives_the_same_answer(self):
        """Determinism: the same inputs and the same state must agree."""
        brain = SituationalBrain.random(seed=11)
        tokens = np.random.default_rng(1).normal(size=(N_MAX, TOKEN_D))
        mask = np.ones(N_MAX)
        state = np.random.default_rng(2).normal(size=STATE_D)
        h0 = np.zeros(HIDDEN_D)
        a = brain.forward(tokens, mask, state, np.zeros(ROLE_D),
                           np.zeros(MEMORY_D), np.zeros(PLAN_D), h0)[0]
        b = brain.forward(tokens, mask, state, np.zeros(ROLE_D),
                           np.zeros(MEMORY_D), np.zeros(PLAN_D), h0)[0]
        self.assertTrue(np.allclose(a, b))


class ThePlanIsStickyTests(unittest.TestCase):
    def test_one_confident_touch_is_not_a_plan(self):
        from player_state import TeamPlan

        tp = TeamPlan(team="Home")
        self.assertFalse(tp.propose(1, 0.9))
        self.assertEqual(tp.plan_id, -1)
        self.assertTrue(tp.propose(1, 0.9))
        self.assertEqual(tp.plan_id, 1)

    def test_a_low_confidence_proposal_never_lands(self):
        from player_state import TeamPlan

        tp = TeamPlan(team="Home")
        for _ in range(20):
            tp.propose(2, 0.4)
        self.assertEqual(tp.plan_id, -1)

    def test_the_plan_does_not_survive_a_turnover(self):
        from player_state import TeamPlan

        tp = TeamPlan(team="Home")
        tp.propose(1, 0.9)
        tp.propose(1, 0.9)
        self.assertEqual(tp.plan_id, 1)
        self.assertTrue(tp.reset_possession(7))
        self.assertEqual(tp.plan_id, -1)


class TheMemoryIsAnInputTests(unittest.TestCase):
    def test_a_player_with_no_memory_reads_zeros(self):
        from cognition.mind import clear_minds
        from memory_features import memory_features

        clear_minds()
        block = memory_features("Nobody At All")
        self.assertEqual(block.shape, (MEMORY_D,))
        self.assertTrue(np.allclose(block, 0.0))

    def test_remembering_something_changes_the_block(self):
        from cognition.mind import clear_minds, register_mind, new_mind
        from memory_features import memory_features

        clear_minds()
        mind = new_mind()
        mind.engaged = False
        register_mind("ST1", mind)
        before = memory_features("ST1").copy()

        class Ev:
            # The memory's tag map is keyed by EVENT TYPE name: a missed
            # big chance is a SHOT_OFF_TARGET, filed under the tag
            # "big_chance_missed".
            event_type = type("E", (), {"name": "SHOT_OFF_TARGET"})()
            player = "ST1"
            secondary_player = None
            metadata = {"x": 80.0, "y": 34.0}
            location_x = 80.0
            location_y = 34.0

        mind.observe(Ev())
        after = memory_features("ST1")
        self.assertFalse(np.allclose(before, after))

    def test_the_provisioned_mind_is_UNENGAGED(self):
        """An engaged mind would change the decision path for every
        player — the whole point of the provisioning is that it does not."""
        from cognition.mind import clear_minds, get_mind
        from memory_features import ensure_mind

        clear_minds()
        mind = ensure_mind("Auto")
        self.assertIsNotNone(mind)
        self.assertFalse(mind.engaged)
        self.assertIs(get_mind("Auto"), mind)

    def test_an_existing_mind_is_left_alone(self):
        from cognition.mind import clear_minds, get_mind, register_mind, new_mind
        from memory_features import ensure_mind

        clear_minds()
        mine = new_mind()
        register_mind("Mine", mine)
        self.assertIs(ensure_mind("Mine"), mine)
        self.assertTrue(mine.engaged)


class TheShippedPathIsUntouchedTests(unittest.TestCase):
    def test_an_ordinary_brain_does_not_route_into_the_situational_path(self):
        from football_brain import FootballBrain
        from brain_integration import (
            NeuralDecisionBrain, clear_registry, register_brain,
        )

        clear_registry()
        register_brain("CM1", FootballBrain.random(seed=2))
        from tests.test_football_brain import (  # noqa: E402
            FakeGameState, make_player,
        )

        player = make_player("CM1", "CM")
        teammate = make_player("ST1", "ST")
        from position_engine import PlayerSpatialState, PositionEngine

        pe = PositionEngine()
        st = PlayerSpatialState(player_name="CM1", position="CM",
                                team="Home", home_x=55.0, home_y=30.0)
        st.current_x, st.current_y = 55.0, 30.0
        pe.states["CM1"] = st
        st2 = PlayerSpatialState(player_name="ST1", position="ST",
                                 team="Home", home_x=70.0, home_y=34.0)
        st2.current_x, st2.current_y = 70.0, 34.0
        pe.states["ST1"] = st2

        d = NeuralDecisionBrain.decide(
            player, 55.0, 30.0, [teammate], [], pe, None, False, True,
            FakeGameState("LEVEL"), 30.0, None, False, None)
        self.assertIsNotNone(d.intent)
        # The situational machinery must NOT have run for this player.
        self.assertIsNone(pe.neural_states.player("CM1").hidden)


class TheCorpusShapeTests(unittest.TestCase):
    def test_a_sequence_validates_its_widths(self):
        from situational_distill import Sequence

        seq = Sequence(
            tokens=np.zeros((3, N_MAX, TOKEN_D)),
            token_mask=np.ones((3, N_MAX)),
            state=np.zeros((3, STATE_D)),
            role=np.zeros((3, ROLE_D)),
            memory=np.zeros((3, MEMORY_D)),
            plan=np.zeros((3, PLAN_D)),
            intent=np.zeros(3, dtype=np.int64),
            phase=np.zeros(3, dtype=np.int64),
            position="ST",
        )
        seq.check()

    def test_a_mismatched_length_is_refused(self):
        from situational_distill import Sequence

        seq = Sequence(
            tokens=np.zeros((3, N_MAX, TOKEN_D)),
            token_mask=np.ones((2, N_MAX)),      # wrong length on purpose
            state=np.zeros((3, STATE_D)),
            role=np.zeros((3, ROLE_D)),
            memory=np.zeros((3, MEMORY_D)),
            plan=np.zeros((3, PLAN_D)),
            intent=np.zeros(3, dtype=np.int64),
            phase=np.zeros(3, dtype=np.int64),
            position="ST",
        )
        with self.assertRaises(ValueError):
            seq.check()


class TheTrainerLearnsTests(unittest.TestCase):
    """NON-VACUITY: the trainer must reduce the loss on data it can fit."""

    def test_training_reduces_the_loss(self):
        from situational_distill import Sequence, distill_situational

        rng = np.random.default_rng(5)
        seqs = []
        for _ in range(6):
            T = 5
            seqs.append(Sequence(
                tokens=rng.normal(size=(T, N_MAX, TOKEN_D)),
                token_mask=np.ones((T, N_MAX)),
                state=rng.normal(size=(T, STATE_D)),
                role=rng.normal(size=(T, ROLE_D)),
                memory=rng.normal(size=(T, MEMORY_D)),
                plan=rng.normal(size=(T, PLAN_D)),
                intent=np.zeros(T, dtype=np.int64),
                phase=np.zeros(T, dtype=np.int64),
                position="ST",
            ))

        class FakeSurrogate:
            """A surrogate whose table always prefers PROGRESSIVE_PASS."""

            def expected_success(self, sensors, intent, position):
                return 1.0 if intent.value == "PROGRESSIVE_PASS" else 0.0

        brain, report = distill_situational(
            "ST", FakeSurrogate(), seqs, epochs=40, learning_rate=0.02,
            seed=3)
        self.assertLess(report["loss_last"], report["loss_first"])
        # And the RESULT must be the target's intent, not some other one.
        self.assertIn("PROGRESSIVE_PASS", report["mix"])
        self.assertGreater(report["mix"]["PROGRESSIVE_PASS"], 0.5)


class TheCorpusAndCensusTests(unittest.TestCase):
    """The corpus format and the census rules — the two things that would
    silently train the wrong thing."""

    def _write_corpus(self, path, positions):
        import json

        seqs = []
        for pos, n in positions:
            for i in range(n):
                seqs.append({
                    "player": f"{pos}{i}",
                    "position": pos,
                    "steps": [{
                        "tokens": [[0.0] * TOKEN_D for _ in range(2)],
                        "token_mask": [1.0, 1.0],
                        "state": [0.0] * STATE_D,
                        "role": [0.0] * ROLE_D,
                        "memory": [0.0] * MEMORY_D,
                        "plan": [0.0] * PLAN_D,
                        "phase": 1,
                        "minute": 1.0,
                    } for _ in range(3)],
                })
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"sequences": seqs}, fh)
        return path

    def test_a_numbered_position_is_one_role_not_two(self):
        """A live roster names its players CM1 / CM2 / CB1.

        A per-LABEL grouping would split one role across two groups, so
        the trainer would see "CM" as empty and silently train nothing
        for the most important role on the pitch.
        """
        import tempfile
        from situational_distill import load_sequences

        tmp = tempfile.mkdtemp()
        path = self._write_corpus(
            os.path.join(tmp, "c.json"), [("CM1", 3), ("CM2", 3), ("CB1", 2)])
        seqs = load_sequences(path)
        fams = sorted({s.position for s in seqs})
        self.assertEqual(fams, ["CB", "CM"],
                         f"the roles were not normalised: {fams}")
        self.assertEqual(len(seqs), 8)

    def test_a_corpus_without_an_intent_still_loads(self):
        """The intent TARGET comes from the surrogate, so the corpus does
        not carry one — a missing key is normal, not malformed."""
        import tempfile
        from situational_distill import load_sequences

        tmp = tempfile.mkdtemp()
        path = self._write_corpus(os.path.join(tmp, "c.json"), [("ST", 2)])
        seqs = load_sequences(path)
        self.assertEqual(len(seqs), 2)
        self.assertEqual(len(seqs[0].intent), 3)

    def test_the_split_does_not_leak_and_is_reproducible(self):
        from situational_distill import Sequence, split_sequences

        def mk(i):
            return Sequence(
                tokens=np.zeros((2, 2, TOKEN_D)),
                token_mask=np.ones((2, 2)),
                state=np.zeros((2, STATE_D)),
                role=np.zeros((2, ROLE_D)),
                memory=np.zeros((2, MEMORY_D)),
                plan=np.zeros((2, PLAN_D)),
                intent=np.zeros(2, dtype=np.int64),
                phase=np.zeros(2, dtype=np.int64),
                position="ST",
            )

        seqs = [mk(i) for i in range(10)]
        train, ev = split_sequences(seqs, seed=123, holdout=0.2)
        self.assertEqual(len(train) + len(ev), 10)
        self.assertTrue(ev, "the hold-out must not be empty")
        self.assertFalse({id(s) for s in train} & {id(s) for s in ev},
                         "the split LEAKS")
        train2, ev2 = split_sequences(seqs, seed=123, holdout=0.2)
        self.assertEqual([id(s) for s in train], [id(s) for s in train2],
                         "the split must be reproducible — the census probe "
                         "depends on reproducing the trainer's split exactly")
        self.assertEqual([id(s) for s in ev], [id(s) for s in ev2])

    def test_the_census_flags_a_collapse(self):
        from situational_distill import census_flags

        collapsed = {"SAFE_PASS": 1.0}
        flags = census_flags("CM", collapsed)
        self.assertTrue(flags, "a one-intent mix is a collapse")

    def test_the_census_flags_a_role_inappropriate_action(self):
        from situational_distill import census_flags

        # The striker crossing: CROSS belongs to {WING, FB}.
        flags = census_flags("ST", {"CROSS": 0.5, "SAFE_PASS": 0.5})
        self.assertTrue(any("CROSS" in f for f in flags),
                        f"a striker crossing must be flagged: {flags}")

    def test_the_census_does_not_flag_the_cam_shooting(self):
        """The role-inappropriate rule was once a hand-written list of
        positions that "should not shoot", and it listed the CAM — but
        the CAM IS the attacking midfielder, and `ROLE_ONLY_INTENTS["SHOOT"]`
        is {ST, AM, WING}.  A guard that flags good football gets
        switched off, and then it flags nothing."""
        from situational_distill import census_flags

        flags = census_flags("CAM", {"SHOOT": 0.2, "SAFE_PASS": 0.8})
        self.assertFalse(any("SHOOT" in f for f in flags),
                         f"a CAM shooting is role-appropriate: {flags}")

    def test_the_census_passes_a_broad_mix(self):
        from situational_distill import census_flags

        broad = {"SAFE_PASS": 0.2, "CARRY": 0.2, "PROGRESSIVE_PASS": 0.15,
                 "PROTECT_POSSESSION": 0.15, "SWITCH": 0.1, "RECYCLE": 0.1,
                 "DRIBBLE": 0.1}
        self.assertEqual(census_flags("CM", broad), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
