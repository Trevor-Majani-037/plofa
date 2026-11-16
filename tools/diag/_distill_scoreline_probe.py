"""GATE 9 — does the DISTILLED net actually choose differently by scoreline?

    python tools/diag/_distill_scoreline_probe.py \
        --brains artifacts/pilot_brains --corpus artifacts/pilot_situational.json \
        --surrogate artifacts/pilot_surrogate.json --positions CM,ST

WHY THIS EXISTS
---------------
Gates 6 and 7 prove the TABLE's preference moves with the scoreline.  That is
not the same as the POLICY moving: ``situational_distill.intent_targets`` takes
the table's argmax and trains a net to reproduce it, so between "the target
changed" and "the net changed" there are three ways to lose the effect —

* the net cannot represent it (slot 18 is an input, but the weights ignore it);
* the target barely changed on the states the net was trained on;
* the training did not converge, so the net still predicts a near-uniform mix.

So this probe holds every input fixed except ``state[18]`` and asks whether the
net's argmax moves.  It reports the TARGET's flip rate next to the NET's, so a
failure is attributable: if the target flips and the net does not, the net
failed; if neither flips, the table has nothing to teach at this corpus size.

``state[18]`` is ``score_diff / 5``, so ±0.4 is a two-goal margin either way.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from decision_brain import PlayerIntent  # noqa: E402
from situational_brain import HIDDEN_D, SituationalBrain  # noqa: E402
from situational_distill import Sequence, _position_family, load_sequences  # noqa: E402

LABELS = [p.value for p in PlayerIntent]


def forward(brain, seq):
    logits, _pl, _h, _c = brain.forward_sequence(
        seq.tokens, seq.token_mask, seq.state, seq.role, seq.memory, seq.plan,
        np.zeros(HIDDEN_D))
    return logits


def with_scoreline(seq, value):
    """A copy of the sequence whose slot 18 is fixed to ``value``.

    Built through the constructor, not by copying ``__dict__``: ``Sequence``
    declares ``__slots__``, so there is no instance dictionary to copy and a
    ``__dict__``-based clone would either raise or silently produce an object
    whose arrays are shared with the original.
    """
    state = np.array(seq.state, dtype=float, copy=True)
    state[:, 18] = value
    return Sequence(
        tokens=seq.tokens, token_mask=seq.token_mask, state=state,
        role=seq.role, memory=seq.memory, plan=seq.plan, intent=seq.intent,
        phase=seq.phase, position=seq.position, player=seq.player,
        team=seq.team, gap=seq.gap)


def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def target_flips(surrogate, seqs, position, low, high):
    """How often the TABLE's preferred intent changes between the two states."""
    n = flips = 0
    moves = Counter()
    for seq in seqs:
        a = with_scoreline(seq, low)
        b = with_scoreline(seq, high)
        vocab = [p for p in PlayerIntent
                 if p.value in _vocab(surrogate, position)]
        if len(vocab) < 2:
            continue
        for i in range(len(a.state)):
            va = {p: surrogate.expected_success(a.state[i][:24], p, position)
                  for p in vocab}
            vb = {p: surrogate.expected_success(b.state[i][:24], p, position)
                  for p in vocab}
            pa = max(va, key=va.get)
            pb = max(vb, key=vb.get)
            n += 1
            if pa is not pb:
                flips += 1
                moves[f"{pa.value} -> {pb.value}"] += 1
    return n, flips, moves


def _vocab(surrogate, position):
    seen = set()
    for buckets in surrogate.table.get(position, {}).values():
        seen.update(buckets.keys())
    return seen


def net_flips(brain, seqs, low, high):
    """How often the NET's argmax changes between the two states."""
    n = flips = 0
    tv = []
    moves = Counter()
    for seq in seqs:
        la = forward(brain, with_scoreline(seq, low))
        lb = forward(brain, with_scoreline(seq, high))
        pa = softmax_rows(la)
        pb = softmax_rows(lb)
        ia = la.argmax(axis=1)
        ib = lb.argmax(axis=1)
        for i in range(len(ia)):
            n += 1
            if ia[i] != ib[i]:
                flips += 1
                moves[f"{LABELS[int(ia[i])]} -> {LABELS[int(ib[i])]}"] += 1
            tv.append(float(np.abs(pa[i] - pb[i]).sum() / 2.0))
    return n, flips, (float(np.mean(tv)) if tv else 0.0), moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brains", default="artifacts/pilot_brains")
    p.add_argument("--corpus", default="artifacts/pilot_situational.json")
    p.add_argument("--surrogate", default="artifacts/pilot_surrogate.json")
    p.add_argument("--positions", default="CM,ST")
    p.add_argument("--low", type=float, default=-0.4,
                   help="slot 18 for the 'behind' state (goals/5)")
    p.add_argument("--high", type=float, default=0.4)
    args = p.parse_args()

    from critic_surrogate import load_any_surrogate
    sur = load_any_surrogate(args.surrogate)
    all_seqs = load_sequences(args.corpus)
    by_family = defaultdict(list)
    for seq in all_seqs:
        by_family[_position_family(seq.position or "")].append(seq)

    print(f"brains    : {args.brains}")
    print(f"states    : slot 18 = {args.low:+.2f} (behind) vs {args.high:+.2f} "
          f"(ahead)\n")
    print(f"   {'pos':4s} {'steps':>6s} {'TARGET':>8s} {'NET':>8s} {'net TV':>8s}")

    ok = False
    any_net = 0
    for pos in [s.strip().upper() for s in args.positions.split(",")]:
        path = os.path.join(args.brains, f"{pos}.json")
        if not os.path.exists(path):
            print(f"   {pos:4s}  (no brain at {path})")
            continue
        brain = SituationalBrain.load(path)
        fam = _position_family(pos)
        seqs = by_family.get(fam, [])
        if not seqs:
            print(f"   {pos:4s}  (no sequences for family {fam})")
            continue

        n_t, f_t, _mt = target_flips(sur, seqs, pos, args.low, args.high)
        n_n, f_n, tv, moves = net_flips(brain, seqs, args.low, args.high)
        t_rate = f_t / n_t if n_t else 0.0
        n_rate = f_n / n_n if n_n else 0.0
        print(f"   {pos:4s} {n_n:6d} {t_rate:8.1%} {n_rate:8.1%} {tv:8.4f}")
        any_net = max(any_net, n_rate)
        if moves:
            top = ", ".join(f"{k} x{v}" for k, v in moves.most_common(3))
            print(f"        net moves: {top}")
        if t_rate > 0.02 and n_rate > 0.02:
            ok = True

    print("\n── VERDICT ──")
    print(f"   gate 9 (net carries the scoreline): "
          f"{'PASS' if ok else 'FAIL'}  (best net flip rate {any_net:.1%})")
    print("   A PASS needs the target to flip AND the net to follow it. "
          "Neither alone is evidence.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
