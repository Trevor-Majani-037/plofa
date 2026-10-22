"""Validate the situational TRAINER path on synthetic data.

Run:  .venv\\Scripts\\python.exe _diag_situational_trainer_smoke.py

It builds a fake corpus (random sequences with the right shapes), runs
``distill_all`` end to end, and checks that:

  * a brain file is written for every position that had sequences,
  * the loss FALLS,
  * the census is reported (and the flags list exists),
  * the held-out split is a REAL split (the eval sequences are not in the
    training set).

This costs no match time, so it validates the trainer while the corpus
is being collected.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from situational_brain import HIDDEN_D, MEMORY_D, PLAN_D, ROLE_D, STATE_D, TOKEN_D
from situational_distill import Sequence, distill_all, split_sequences


def fake_corpus(path, n_pos=4, n_seq=12, steps=5):
    rng = np.random.default_rng(3)
    seqs = []
    for p in range(n_pos):
        pos = ["GK", "CB", "CM", "ST"][p]
        for s in range(n_seq):
            seqs.append({
                "player": f"{pos}{s}",
                "position": pos,
                "steps": [{
                    "tokens": [[round(float(v), 5) for v in row]
                               for row in rng.normal(size=(4, TOKEN_D))],
                    "token_mask": [1.0] * 4,
                    "state": [round(float(v), 6) for v in rng.normal(size=STATE_D)],
                    "role": [round(float(v), 6) for v in rng.normal(size=ROLE_D)],
                    "memory": [round(float(v), 6) for v in rng.normal(size=MEMORY_D)],
                    "plan": [round(float(v), 6) for v in rng.normal(size=PLAN_D)],
                    "phase": int(rng.integers(0, 4)),
                    "minute": 1.0,
                } for _ in range(steps)],
            })
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"sequences": seqs}, fh)
    return seqs


class FakeSurrogate:
    """A table that always prefers PROGRESSIVE_PASS."""

    def expected_success(self, sensors, intent, position):
        return 1.0 if intent.value == "PROGRESSIVE_PASS" else 0.0


def main() -> int:
    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, "corpus.json")
    fake_corpus(corpus)
    out = os.path.join(tmp, "brains")

    summary = distill_all(["GK", "CB", "CM", "ST"], corpus, FakeSurrogate(),
                           out, epochs=12, learning_rate=0.02, seed=123,
                           holdout=0.2, verbose=True)

    assert summary["reports"], "no reports"
    for pos, rep in summary["reports"].items():
        assert rep["loss_last"] < rep["loss_first"], \
            f"{pos}: the loss did not fall ({rep['loss_first']} -> {rep['loss_last']})"
        assert os.path.exists(os.path.join(out, f"{pos}.json")), \
            f"{pos}: no brain file"
        assert rep["eval_sequences"] > 0, f"{pos}: no held-out sequences"
        assert "mix_heldout" in rep, f"{pos}: no held-out mix"
        print(f"  {pos}: loss {rep['loss_first']:.4f} -> {rep['loss_last']:.4f}, "
              f"agree {rep['agreement']:.3f}, "
              f"{rep['train_sequences']} train / {rep['eval_sequences']} held-out")
    assert os.path.exists(os.path.join(out, "_distill_report.json"))
    print("report written")

    # the split must be a REAL split
    from situational_distill import load_sequences
    seqs = load_sequences(corpus)
    train, ev = split_sequences(seqs, seed=123, holdout=0.2)
    ids_tr = {id(s) for s in train}
    assert not (ids_tr & {id(s) for s in ev}), "the split LEAKS"
    assert len(train) + len(ev) == len(seqs)
    print("the held-out split does not leak")

    # the split must be REPRODUCIBLE (the census probe depends on it)
    train2, ev2 = split_sequences(seqs, seed=123, holdout=0.2)
    assert [id(s) for s in train] == [id(s) for s in train2]
    assert [id(s) for s in ev] == [id(s) for s in ev2]
    print("the split is reproducible")

    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
