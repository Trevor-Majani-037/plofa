"""SITUATIONAL DISTILLATION — train the four-capability brain by BPTT.

WHAT THIS IS
------------
The same idea as ``brain_distill.py`` — fit the network to the SURROGATE'S
OWN POLICY rather than to a reward — extended to a network that has a
recurrent core, an attention block and a second head.

The target for every step is still "the intent the surrogate's table
prefers in this bucket, over the role's vocabulary".  That matters: the
CENSUS compares a trained set's intent mix against the table's mix, and
that comparison stays meaningful only if the objective is the same one.
What changes is the model, not the goal.

THE THREE ADDITIONS
-------------------
RECURRENCE
    The training signal now spans a SEQUENCE of touches, so the loss is
    back-propagated THROUGH TIME (BPTT) rather than through a single
    forward pass.  The hidden state is carried from step to step, exactly
    as it is at match time, and it is RESET at the start of every
    sequence — because a possession is not a continuation of the last one.

ATTENTION
    The attention reads a SET of entity tokens, so the sequence carries
    the token set and its mask, and the padded tokens take no attention
    mass (the mask is applied inside the softmax).

PLANNING
    A second head, supervised by the engine's OWN possession phase.  It
    is a SEPARATE loss term, weighted ``PLAN_LOSS_W``, because the plan
    is a different kind of answer (which phase are we in) from the
    intent (what do I do now).

WHY NOT TRAIN THE PLAN HEAD FROM THE TABLE
-------------------------------------------
Because the table says nothing about phases.  The engine's phase machine
(``possession_phases``) already names the phase, it is pure geometry, and
it is the thing the team is actually playing — so it is the honest
supervision signal for "which phase is this".

THE LOSS, EXACTLY
-----------------
::

    L = mean_t CE(intent_logits[t], intent_target[t])
      + PLAN_LOSS_W * mean_t CE(plan_logits[t], plan_target[t])

The plan term is a small addition, not a replacement: the intent is what
the player DOES, the plan is what he is TRYING to do.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from decision_brain import PlayerIntent
from role_features import ROLE_INTENTS, _role_family, ROLE_FAMILY, canonical_position
from situational_brain import (
    HIDDEN_D, MEMORY_D, OUT_D, PLAN_D, PLAN_OUT_D, ROLE_D, STATE_D,
    TOKEN_D, SituationalBrain,
)


#: How much the plan head counts for, relative to the intent head.
PLAN_LOSS_W = 0.35


def _vocabulary(position: str) -> List[PlayerIntent]:
    # Handle both position names (RW, LW, CB, etc.) and family names (WING, FB, etc.)
    # First try to get family from position name
    fam = _role_family(position) or position
    names = ROLE_INTENTS.get(fam, frozenset())
    return [i for i in PlayerIntent if i.value in names]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def intent_targets(surrogate: Any, state: np.ndarray, position: str) -> np.ndarray:
    """The table's preferred intent for each state, as an index.

    Identical in spirit to ``brain_distill._targets`` — the SAME objective,
    so the census comparison remains apples-to-apples.
    """
    vocab = _vocabulary(position)
    if not vocab:
        raise ValueError(f"no role vocabulary for position {position!r}")
    out = np.zeros(len(state), dtype=np.int64)
    for i, row in enumerate(state):
        best = None
        best_v = None
        for intent in vocab:
            v = float(surrogate.expected_success(row[:24], intent, position))
            if best_v is None or v > best_v:
                best, best_v = intent, v
        if best is None:
            continue
        out[i] = [p.value for p in PlayerIntent].index(best.value)
    return out


class Sequence:
    """One player's run of touches — the unit BPTT trains over."""

    __slots__ = ("tokens", "token_mask", "state", "role", "memory", "plan",
                 "intent", "phase", "position", "player", "team", "gap")

    def __init__(self, tokens, token_mask, state, role, memory, plan,
                 intent, phase, position, player="", team="", gap=None):
        self.tokens = np.asarray(tokens, dtype=np.float64)
        self.token_mask = np.asarray(token_mask, dtype=np.float64)
        self.state = np.asarray(state, dtype=np.float64)
        self.role = np.asarray(role, dtype=np.float64)
        self.memory = np.asarray(memory, dtype=np.float64)
        self.plan = np.asarray(plan, dtype=np.float64)
        self.intent = np.asarray(intent, dtype=np.int64)
        self.phase = np.asarray(phase, dtype=np.int64)
        self.position = position
        self.player = player
        self.team = team
        # The ORDER GAP: how many match decisions passed since this
        # player's previous touch.  It is the DISCRIMINATOR between an
        # interleave (a big spatial jump with a SMALL gap — two different
        # men) and a legitimate absence (a big jump with a LARGE gap — one
        # man who lost the ball and got it back).
        self.gap = (np.asarray(gap, dtype=np.int64) if gap is not None
                    else np.zeros(len(self.state), dtype=np.int64))

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def check(self) -> None:
        """Every array must agree on the length, or the BPTT is nonsense."""
        n = self.state.shape[0]
        for name in ("tokens", "token_mask", "role", "memory", "plan",
                     "intent", "phase"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"sequence length mismatch: {name} has {arr.shape[0]} "
                    f"steps, state has {n}")
        if self.state.shape[1] != STATE_D:
            raise ValueError(f"state width {self.state.shape[1]} != {STATE_D}")
        if self.role.shape[1] != ROLE_D:
            raise ValueError(f"role width {self.role.shape[1]} != {ROLE_D}")
        if self.memory.shape[1] != MEMORY_D:
            raise ValueError(f"memory width {self.memory.shape[1]} != {MEMORY_D}")
        if self.plan.shape[1] != PLAN_D:
            raise ValueError(f"plan width {self.plan.shape[1]} != {PLAN_D}")
        if self.tokens.shape[2] != TOKEN_D:
            raise ValueError(f"token width {self.tokens.shape[2]} != {TOKEN_D}")


def load_sequences(path: str) -> List[Sequence]:
    """Read a situational corpus (the format the collector writes)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: List[Sequence] = []
    for entry in raw.get("sequences", []):
        steps = entry["steps"]
        if not steps:
            continue
        # The corpus does NOT carry the intent: the intent TARGET comes
        # from the SURROGATE, so that the objective is identical to the
        # one the MLP brains were distilled against.  A missing key is
        # therefore normal, not a malformed corpus.
        #
        # THE POSITION IS NORMALISED TO ITS ROLE FAMILY.  A live roster
        # names its players "CM1", "CM2", "CB1" ... and a per-label
        # grouping would split ONE role across two groups — so the
        # trainer would see "CM" as empty and silently train nothing.
        # `role_features._role_family` is the canonical stripper (it
        # removes the digits), and it is the same one the rest of the
        # project uses, so there is one definition.
        raw_position = entry.get("position", "")
        family = _role_family(raw_position) or raw_position
        seq = Sequence(
            tokens=[s["tokens"] for s in steps],
            token_mask=[s["token_mask"] for s in steps],
            state=[s["state"] for s in steps],
            role=[s["role"] for s in steps],
            memory=[s["memory"] for s in steps],
            plan=[s["plan"] for s in steps],
            intent=[s.get("intent", 0) for s in steps],
            phase=[s.get("phase", -1) for s in steps],
            position=family,
            player=entry.get("player", ""),
            team=entry.get("team", ""),
            gap=[s.get("gap", 0) for s in steps],
        )
        seq.check()
        out.append(seq)
    return out


def distill_situational(
    position: str,
    surrogate: Any,
    sequences: List[Sequence],
    epochs: int = 60,
    learning_rate: float = 0.01,
    seed: int = 123,
    plan_loss_w: float = PLAN_LOSS_W,
    verbose: bool = False,
    brain: Optional[SituationalBrain] = None,
    hidden_size: int = 32,
    input_size: int = 51,
) -> Tuple[SituationalBrain, Dict[str, Any]]:
    """Fit one position's situational brain by BPTT over its sequences.

    Returns the brain and a report (the loss curve, the agreement with
    the table, and the intent mix of the result).
    """
    if not sequences:
        raise ValueError("no sequences to train on")
    rng = np.random.default_rng(seed)
    if brain is None:
        brain = SituationalBrain.random(seed=seed, hidden_size=hidden_size, input_size=input_size)

    params = [brain.w[k] for k in sorted(brain.w)]
    keys = sorted(brain.w)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    # The intent targets come from the SURROGATE, not from the corpus, so
    # the corpus does not have to carry them.
    for seq in sequences:
        seq.intent = intent_targets(surrogate, seq.state, position)
    n_steps = sum(len(s) for s in sequences)

    loss_curve = []
    agree_curve = []
    for epoch in range(epochs):
        grads = {k: np.zeros_like(brain.w[k]) for k in keys}
        total_loss = 0.0
        agree_hits = 0
        agree_n = 0
        for seq in sequences:
            T = len(seq)
            logits, plan_logits, _h, caches = brain.forward_sequence(
                seq.tokens, seq.token_mask, seq.state, seq.role,
                seq.memory, seq.plan, np.zeros(HIDDEN_D))
            logp = _log_softmax(logits)
            plan_logp = _log_softmax(plan_logits)
            nll = -(logp[np.arange(T), seq.intent])
            # THE PLAN TARGET MUST BE A REAL CLASS.  A step whose phase
            # is unknown (-1) is SKIPPED from the plan term, not folded
            # into the last class: `plan_logp[..., -1]` would be a
            # perfectly valid index and would teach the head a class
            # that does not exist.
            plan_mask = np.asarray(seq.phase) >= 0
            if plan_mask.any():
                pnll = -(plan_logp[np.arange(T)[plan_mask],
                                 np.asarray(seq.phase)[plan_mask]])
            else:
                pnll = np.zeros(T)
            total_loss += float((nll + plan_loss_w * pnll).sum())

            probs = np.exp(logp)
            agree_hits += int((probs.argmax(axis=1) == seq.intent).sum())
            agree_n += T

            dlogits = probs.copy()
            dlogits[np.arange(T), seq.intent] -= 1.0
            dlogits /= T
            dplan = np.zeros((T, PLAN_OUT_D), dtype=np.float64)
            if plan_mask.any():
                pprobs = np.exp(plan_logp)
                rows = np.arange(T)[plan_mask]
                cols = np.asarray(seq.phase)[plan_mask]
                dplan[rows, cols] = pprobs[rows, cols] - 1.0
                dplan *= plan_loss_w / max(1, int(plan_mask.sum()))

            g = brain.backward(dlogits, dplan, caches)
            for k in keys:
                grads[k] += g[k]

        loss_curve.append(total_loss / max(1, n_steps))
        agree_curve.append(agree_hits / max(1, agree_n))

        step = epoch + 1
        for i, k in enumerate(keys):
            gi = grads[k]
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * (gi * gi)
            mhat = m[i] / (1 - b1 ** step)
            vhat = v[i] / (1 - b2 ** step)
            brain.w[k] -= learning_rate * mhat / (np.sqrt(vhat) + eps)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:>4}  loss {loss_curve[-1]:.4f}  "
                  f"agree {agree_curve[-1]:.3f}")

    # The final mix, over the corpus.
    labels = [p.value for p in PlayerIntent]
    mix: Dict[str, int] = {}
    for seq in sequences:
        logits, _pl, _h, _c = brain.forward_sequence(
            seq.tokens, seq.token_mask, seq.state, seq.role, seq.memory,
            seq.plan, np.zeros(HIDDEN_D))
        for idx in logits.argmax(axis=1):
            mix[labels[int(idx)]] = mix.get(labels[int(idx)], 0) + 1
    total = max(1, sum(mix.values()))
    report = {
        "position": position,
        "sequences": len(sequences),
        "steps": n_steps,
        "epochs": epochs,
        "seed": seed,
        "loss_first": loss_curve[0] if loss_curve else None,
        "loss_last": loss_curve[-1] if loss_curve else None,
        "agreement_first": agree_curve[0] if agree_curve else None,
        "agreement": agree_curve[-1] if agree_curve else None,
        "mix": {k: round(v / total, 4) for k, v in sorted(mix.items())},
    }
    return brain, report


# ─────────────────────────────────────────────────────────────
# THE CENSUS RULES — the SAME ones the MLP census uses
# ─────────────────────────────────────────────────────────────
#
# Copied from `_diag_intent_mix_all.py` rather than re-invented, because
# the rules are the GATE and two gates that drift are one gate:
#   * fewer than five intents in the mix  -> collapsed
#   * any single intent over 40%           -> collapsed
#   * a role-specific action the role should not perform at all
#     (derived from `role_features.ROLE_ONLY_INTENTS`, NOT hand-written)
#
# The role-inappropriate rule was once a hand-written list of positions
# that "should not shoot" and it flagged the CAM — but the CAM IS the
# attacking midfielder and `ROLE_ONLY_INTENTS["SHOOT"]` is {ST, AM, WING}.
# A guard that flags good football gets switched off, and then it flags
# nothing.

CENSUS_MIN_INTENTS = 5
CENSUS_MAX_SHARE = 0.40
CENSUS_ROLE_ACTION_MAX = 0.01


def census_flags(position: str, mix: Dict[str, float]) -> List[str]:
    """The census verdict for one position's intent mix."""
    from role_features import ROLE_FAMILY, ROLE_ONLY_INTENTS

    flags: List[str] = []
    if not mix:
        return [f"{position}: empty mix"]
    total = max(1e-9, sum(mix.values()))
    top = max(mix, key=lambda k: mix[k])
    top_share = mix[top] / total
    if len(mix) < CENSUS_MIN_INTENTS:
        flags.append(f"{position}: only {len(mix)} intents")
    if top_share > CENSUS_MAX_SHARE:
        flags.append(f"{position}: {top} at {top_share:.0%}")
    fam = ROLE_FAMILY.get(position.rstrip("0123456789"))
    for intent, allowed in ROLE_ONLY_INTENTS.items():
        share = mix.get(intent, 0.0) / total
        if fam is not None and fam not in allowed and share > CENSUS_ROLE_ACTION_MAX:
            flags.append(f"{position}: {intent} at {share:.1%} "
                         f"(a role-inappropriate action)")
    return flags


def split_sequences(seqs: List[Sequence], seed: int = 123,
                    holdout: float = 0.2) -> Tuple[List[Sequence],
                                                    List[Sequence]]:
    """Split sequences into train / held-out, BY SEQUENCE.

    NEVER by step: a step's neighbours are the same possession, so a
    step-level split leaks the training data into the held-out set and
    the "generalisation" number becomes a training number with a
    different name.

    The split is DETERMINISTIC (seeded permutation of the position's own
    sequence list), so the census probe can reproduce EXACTLY the split
    the trainer used — and the census therefore reads data the brain
    has never seen.
    """
    if not seqs:
        return [], []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seqs))
    n_hold = max(1, int(len(seqs) * holdout)) if len(seqs) >= 5 else 0
    hold_idx = set(int(i) for i in order[:n_hold])
    train = [s for i, s in enumerate(seqs) if i not in hold_idx]
    eval_ = [s for i, s in enumerate(seqs) if i in hold_idx]
    if not train:
        train, eval_ = seqs, []
    return train, eval_


def _position_family(position: str) -> str:
    """The ROLE FAMILY of a position name.

    The corpus is grouped by FAMILY (the position label is normalised to
    it), because a live roster names its players "CM1" / "CB1".  The
    families are: GK, CB, FB (LB + RB), DM (CDM), CM, AM (CAM), WING
    (LW + RW), ST (ST + CF).

    The PROJECTION MATTERS: looping over the POSITION names ("LB", "LW",
    "CDM", ...) against a corpus keyed by FAMILY would find NOTHING for
    seven of the eleven positions, and the trainer would silently write
    no brain for them.  That is the same shape as the bug AGENTS.md
    records for `_diag_intent_mix_all.py`'s FAMILY map.
    """
    from role_features import ROLE_FAMILY

    return ROLE_FAMILY.get(position.rstrip("0123456789"), position)


def positions_for_family(family: str) -> List[str]:
    """Every POSITION name that belongs to a family."""
    from role_features import ROLE_FAMILY

    out = [p for p, fam in ROLE_FAMILY.items() if fam == family]
    return sorted(out)


def distill_all(
    positions: List[str],
    corpus_path: str,
    surrogate: Any,
    out_dir: str,
    epochs: int = 60,
    learning_rate: float = 0.01,
    seed: int = 123,
    holdout: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train every position from the situational corpus, and CENSUS it.

    THE HOLDOUT IS THE POINT.  The census asks "does this brain collapse?"
    and a collapse measured on the TRAINING set understates the collapse
    (a brain that memorised one action has memorised it on the training
    set too, so the two agree — but a brain that merely FAILED to learn
    looks different on data it has never seen).  So the sequences are
    split BY SEQUENCE — never by step, because a step's neighbours are
    the same possession and would leak — and the census is read on the
    held-out split.

    ONE BRAIN PER FAMILY, written to every position file in it.  The
    corpus cannot separate a left winger from a right winger (they share
    the WING family and the same role features), so training them
    separately would produce two IDENTICAL brains and a false impression
    of per-position specialisation.  One brain per family is the honest
    structure, and the file names stay the eleven position names so the
    registration works.
    """
    import os

    all_seqs = load_sequences(corpus_path)
    by_pos: Dict[str, List[Sequence]] = {}
    for seq in all_seqs:
        by_pos.setdefault(seq.position or "", []).append(seq)

    os.makedirs(out_dir, exist_ok=True)
    reports: Dict[str, Any] = {}
    flags: List[str] = []
    trained_families: Dict[str, Any] = {}

    for pos in positions:
        family = _position_family(pos)
        if family not in trained_families:
            seqs = by_pos.get(family, [])
            if not seqs:
                if verbose:
                    print(f"  {pos} (family {family}): no sequences in the "
                          f"corpus")
                continue
            train, eval_ = split_sequences(seqs, seed=seed, holdout=holdout)
            brain, report = distill_situational(
                canonical_position(family), surrogate, train, epochs=epochs,
                learning_rate=learning_rate, seed=seed, verbose=verbose)

            mix = _mix_over(brain, eval_ or train)
            report["mix_heldout"] = {k: round(v, 4)
                                     for k, v in sorted(mix.items())}
            report["train_sequences"] = len(train)
            report["eval_sequences"] = len(eval_)
            report["family"] = family
            # The census flags are computed from a POSITION name (the
            # role-inappropriate rule needs `ROLE_FAMILY.get(position)`,
            # and a FAMILY name is not a key of that map — passing "FB"
            # would silently SKIP the role rule).
            pos_flags = census_flags(pos, mix)
            report["census_flags"] = pos_flags
            flags.extend(f"{p}: {f}" for p in positions_for_family(family)
                         for f in pos_flags)
            trained_families[family] = (brain, report)
            if verbose:
                top = sorted(mix.items(), key=lambda kv: -kv[1])[:3]
                print(f"  {family}: {len(train)} train / {len(eval_)} held-out"
                      f"  loss {report['loss_last']:.4f}"
                      f"  agree {report['agreement']:.3f}"
                      f"  mix " + ", ".join(f"{k} {v:.0%}" for k, v in top))
                if pos_flags:
                    print(f"      CENSUS: {'; '.join(pos_flags)}")

        if family not in trained_families:
            continue
        brain, report = trained_families[family]
        brain.meta = dict(brain.meta or {})
        brain.meta.update({
            "position": pos,
            "family": family,
            "training_method": "bptt_situational",
            "corpus": os.path.basename(corpus_path),
            "epochs": epochs,
            "seed": seed,
        })
        brain.save(os.path.join(out_dir, f"{pos}.json"))
        reports[pos] = report

    summary = {"reports": reports, "flags": flags, "out_dir": out_dir}
    with open(os.path.join(out_dir, "_distill_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        print()
        if flags:
            print("CENSUS FLAGGED:")
            for f in flags:
                print(f"    {f}")
        else:
            print("CENSUS PASSES: every position uses >4 intents, no action "
                  "over 40%, and no role-specific action out of role.")
    return summary


def _mix_over(brain: Any, sequences: List[Sequence]) -> Dict[str, float]:
    """The intent mix over a set of sequences."""
    labels = [p.value for p in PlayerIntent]
    counts: Dict[str, int] = {}
    for seq in sequences:
        logits, _pl, _h, _c = brain.forward_sequence(
            seq.tokens, seq.token_mask, seq.state, seq.role, seq.memory,
            seq.plan, np.zeros(HIDDEN_D))
        for idx in logits.argmax(axis=1):
            name = labels[int(idx)]
            counts[name] = counts.get(name, 0) + 1
    total = max(1, sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train the situational brains by "
                                           "BPTT and census them.")
    p.add_argument("--corpus", required=True,
                   help="the situational corpus (--situational-out)")
    p.add_argument("--surrogate", required=True,
                   help="the FITTED surrogate the intent targets come from")
    p.add_argument("--out", default="brains_v8")
    p.add_argument("--positions", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--holdout", type=float, default=0.2)
    A = p.parse_args()

    from critic_surrogate import load_any_surrogate

    sur = load_any_surrogate(A.surrogate)
    positions = ([s.strip().upper() for s in A.positions.split(",")]
                 if A.positions else
                 ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW",
                  "ST", "CF"])
    distill_all(positions, A.corpus, sur, A.out, epochs=A.epochs,
                learning_rate=A.learning_rate, seed=A.seed,
                holdout=A.holdout, verbose=True)
