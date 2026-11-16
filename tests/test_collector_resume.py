"""The collection can be stopped and resumed WITHOUT losing matches.

The collection is hours of CPU, so it has to be splittable: run three
matches, stop, run four more tomorrow.  That only works if a resumed
chunk ADDS to the corpus on disk.  It used to REPLACE it.

Two defects, both silent:

1. ``save_situational_corpus`` rebuilt the file from the records of the
   CURRENT process.  A resumed chunk holds only its own matches, so it
   overwrote the earlier ones while stamping the file with the TOTAL
   match count.  The next chunk then skipped the total, and the early
   matches were gone — with nothing on disk to say they ever existed.

2. The CLI wrote the corpus a SECOND time after collection, with no
   match count and no prior sequences, so ``matches_completed`` was
   reset to 0 on every exit and the next run replayed everything.

Both are tested here at the level they fail: the file.
"""

import inspect
import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import surrogate_collect as SC  # noqa: E402
from event_chain import PlayerIntent  # noqa: E402


def _recs(n, player="CM1", team="Home", order_base=0):
    """n decisions by one player, close enough to form ONE sequence."""
    out = []
    for i in range(n):
        out.append(SC.DecisionRecord(
            player, "CM", 10.0 + i, order_base + i,
            np.zeros(24), PlayerIntent.SAFE_PASS,
            team=team,
            tokens=np.zeros((2, 4)), token_mask=np.ones(2),
            role=np.zeros(3), memory=np.zeros(3),
            phase=1,
        ))
    return out


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="resume_")
        self.path = os.path.join(self.dir, "situational.json")

    def tearDown(self):
        for f in os.listdir(self.dir):
            os.unlink(os.path.join(self.dir, f))
        os.rmdir(self.dir)


class TheCorpusOnDisk(_Tmp):
    """What a resumed save actually writes."""

    def test_a_resumed_save_keeps_the_earlier_matches(self):
        # chunk 1: two matches, saved as two matches.
        SC.save_situational_corpus(_recs(3, "CM1"), self.path, quiet=True,
                                   matches_completed=2)
        first = SC.load_situational_sequences(self.path)
        self.assertEqual(len(first), 1, "chunk 1 wrote one sequence")

        # chunk 2: a DIFFERENT player, resumed from 2.
        prior = SC.load_situational_sequences(self.path)
        SC.save_situational_corpus(_recs(3, "ST1"), self.path, quiet=True,
                                   matches_completed=3,
                                   prior_sequences=prior)
        both = SC.load_situational_sequences(self.path)

        self.assertEqual(len(both), 2,
                         "the resumed chunk replaced the earlier match "
                         f"instead of adding to it: {len(both)} sequences")
        names = sorted(s["player"] for s in both)
        self.assertEqual(names, ["CM1", "ST1"],
                         "an earlier match is missing from the corpus")

    def test_the_earlier_sequences_are_written_back_byte_for_byte(self):
        SC.save_situational_corpus(_recs(3, "CM1"), self.path, quiet=True,
                                   matches_completed=1)
        before = SC.load_situational_sequences(self.path)

        SC.save_situational_corpus(_recs(3, "ST1"), self.path, quiet=True,
                                   matches_completed=2,
                                   prior_sequences=before)
        after = SC.load_situational_sequences(self.path)

        self.assertEqual(after[0], before[0],
                         "the earlier sequence was rewritten, not preserved")

    def test_the_match_count_is_the_total_not_the_chunk(self):
        SC.save_situational_corpus(_recs(3), self.path, quiet=True,
                                   matches_completed=4)
        prior = SC.load_situational_sequences(self.path)
        SC.save_situational_corpus(_recs(3, "ST1"), self.path, quiet=True,
                                   matches_completed=7,
                                   prior_sequences=prior)
        self.assertEqual(SC.completed_matches(self.path), 7)

    def test_two_chunks_hold_what_one_run_would_have(self):
        # one run: three matches, three players, in one go.
        whole = os.path.join(self.dir, "whole.json")
        recs = _recs(2, "CM1") + _recs(2, "ST1") + _recs(2, "CB1")
        SC.save_situational_corpus(recs, whole, quiet=True, matches_completed=3)

        # two chunks: 2 then 1, each saved through the resume path.
        SC.save_situational_corpus(_recs(2, "CM1") + _recs(2, "ST1"),
                                   self.path, quiet=True, matches_completed=2)
        prior = SC.load_situational_sequences(self.path)
        SC.save_situational_corpus(_recs(2, "CB1"), self.path, quiet=True,
                                   matches_completed=3, prior_sequences=prior)

        a = sorted(s["player"] for s in SC.load_situational_sequences(whole))
        b = sorted(s["player"] for s in SC.load_situational_sequences(self.path))
        self.assertEqual(a, b,
                         "a chunked collection is not the same corpus as the "
                         "same collection run in one go")


class ReadingBack(_Tmp):

    def test_a_corpus_that_does_not_exist_is_empty(self):
        self.assertEqual(SC.load_situational_sequences(self.path), [])
        self.assertEqual(SC.completed_matches(self.path), 0)

    def test_a_corrupt_corpus_is_empty_not_a_crash(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(SC.load_situational_sequences(self.path), [])
        self.assertEqual(SC.completed_matches(self.path), 0)

    def test_the_sequences_are_returned_as_saved(self):
        SC.save_situational_corpus(_recs(3), self.path, quiet=True,
                                   matches_completed=1)
        with open(self.path, encoding="utf-8") as fh:
            on_disk = json.load(fh)["sequences"]
        self.assertEqual(SC.load_situational_sequences(self.path), on_disk)


class NoCallSiteMayWriteTheCorpusBare(unittest.TestCase):
    """Every save must carry the count and the prior sequences.

    A bare ``save_situational_corpus(records, path)`` is the bug in
    defect 2: it defaults to ``matches_completed=0`` and no prior, which
    is indistinguishable from "delete everything and start again".
    """

    def test_no_save_omits_the_count_or_the_prior(self):
        src = inspect.getsource(SC)
        # CALL SITES only: the definition line has the same name.
        calls = re.findall(r"(?<!def )save_situational_corpus\((.*?)\)\s*$",
                           src, flags=re.M | re.S)
        self.assertTrue(calls, "the call was renamed; update this test")
        for call in calls:
            self.assertIn("matches_completed=", call,
                          "a save writes no match count, so the next run "
                          f"cannot resume: {call[:120]}")
            self.assertIn("prior_sequences=", call,
                          "a save drops the earlier matches: "
                          f"{call[:120]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
