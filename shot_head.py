"""THE SHOT READ-OUT — a 33-parameter supervised head on the SHOOT logit.

WHY THIS EXISTS, AND WHY IT IS NOT §10.3 AS WRITTEN
---------------------------------------------------
`docs/RETRAIN_PLAN_2026-11-08.md` §10.3 proposed teaching the EXISTING output
layer to recognise a shot, by adding a supervised auxiliary loss to the brain's
fitness.  §10.5's feasibility test (`tools/diag/_shothead_probe.py`, measured
2026-11-16) showed why that fails, and what to do instead.

Measured on the shipped ST brain, 400 scoring states, held out:

  A logistic read-out on the FROZEN 32-d hidden state reaches AUC 0.954.
  The shipped output layer, reading the same hidden state, reaches 0.277 —
  worse than chance and anti-correlated with shot viability.

So the information is in the representation and the head reads it backwards.
Re-training that head fixes the ordering and destroys the policy (21 % of
held-out choices survive; the argmax mix collapses to SWITCH 88 %), and an
anchor strong enough to keep the policy gives the ordering back to chance.  The
two objectives fight over one bottleneck and nine shared logits.

This module is the mechanism that does not fight: a separate 33-parameter linear
map on the same hidden state, added to the SHOOT logit ALONE.

    logit_SHOOT += beta * (w_s . h + b_s)

No shared weight moves, so the other nine intents are untouched by
construction.  Measured at beta = 6 on ST: AUC 0.945, 92 % of states that should
not have changed left alone, and the brain takes the shot on 78 % of genuinely
shootable states where it previously took 0 %.

THE `beta` IS CHOSEN BY THE ARTIFACT, NOT BY HAND
-------------------------------------------------
`fit()` sweeps beta and picks the smallest value that reaches a usable shot rate
without collateral damage.  If no beta qualifies it returns ``None`` — a
position whose brain does not encode shot viability well enough is REFUSED
rather than shipped on a good result elsewhere.  CAM is exactly this case
(probe AUC 0.834, 64 % left alone at any usable beta) and the refusal is the
honest outcome, not a failure of the sweep.

WARM HIDDEN STATES
------------------
The probe is fitted on hidden states produced with the recurrent state CARRIED
along a short sequence, because that is what the hidden state looks like on the
pitch: `situational_decision` passes `h_prev` from the player's last touch.  A
probe fitted only on `h_prev = 0` would be fitted on a distribution that never
occurs in a match.  `fit()` reports both, so the gap is visible rather than
assumed away.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from attacking_matrix import shot_score
from brain_evolution import (
    _DummyPositionEngine, _OPP_POOL, _OUTFIELD_TM_POOL, _carrier,
    _role_coords_to_players, random_game_state, random_game_state_scoring,
)
from brain_sensors import extract_sensors
from football_brain import INTENT_LABELS
from situation_tokens import entity_tokens
from role_features import ROLE_ONLY_INTENTS, _role_family
from situational_context import situation_blocks

SHOOT_IDX = INTENT_LABELS.index("SHOOT")

# The viability threshold the label uses.  ``shot_score`` is NOT normalised to
# [0, 1] — it runs to ~3.0 at the six-yard box — so 0.20 corresponds to roughly
# 14 m central, which is a sane "this is a real chance" line and is the one the
# plan specifies.
SHOT_THRESHOLD = 0.20

DEFAULT_BETA_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

# THE BIAS IS CLAMPED, AND THIS IS NOT A DETAIL.
#
# Measured on the shipped ST brain over 400 real hidden states at beta = 6: the
# raw bias ran from -103.7 to +15.4.  A bias of -104 is not a prior, it is a
# VETO — p_SHOOT becomes exactly 0.0 and the brain can never shoot there, no
# matter what it sees.  That is the opposite of the design intent ("the brain
# still owns the verb: it may decline a shootable ball"), and it is brittle: one
# out-of-distribution hidden state produces an unbounded logit.
#
# Clamping to +-6 caps the odds the head can impose at e^6 ~ 400:1 — decisive
# enough to change a decision, finite enough that another intent can still win.
# The clamp lives inside `bias()` so the fitted numbers and the shipped artifact
# are the same function; clamping only at apply time would mean the reported
# shot rate was measured on a mechanism that never ships.
BIAS_CLAMP = 6.0

# Selection criteria.  These are ABSOLUTE, and they are the reason CAM is
# refused: a head that ranks well but never changes a decision is decoration,
# and a head that changes decisions it has no business changing is damage.
MIN_AUC = 0.85
MIN_SHOT_RATE = 0.60        # of genuinely shootable states
MIN_CLEAN = 0.90            # of states that should not change at all

# THE CONSTRAINT THAT WAS MISSING, AND WHAT IT COST
# ------------------------------------------------
# `MIN_CLEAN` measures COLLATERAL: states that are neither shootable nor already
# shooting must keep their argmax.  That set deliberately EXCLUDES every state
# where the brain was already shooting — so the head was never charged for
# silencing a shot the brain wanted.  Combined with `MIN_SHOT_RATE`, which only
# counts states the geometry calls shootable, the fitter was free to reward the
# head for overruling the brain wherever the two disagreed, and free of any
# penalty for doing so.
#
# It took the bribe.  The shipped ST head came out at beta 4.0 with **66 % of
# states pinned at the +-6 clamp wall**, and |bias| > |logit gap| on **76.5 %**
# of states — i.e. the head, not the brain, was choosing.  The real-match gate
# then showed what that means: shots 3 -> 0.  A 33-parameter linear read-out
# trained on synthetic geometry silenced a striker completely.
#
# A prior may BREAK A TIE.  It may not OVERTURN A DECISION.  These two numbers
# are that sentence, measured:
#   overrule — share of held-out states where arming changes the argmax at all;
#   silenced — share where the brain chose SHOOT and arming takes it away.
# Both are computed over EVERY state, so the head cannot hide its disagreements
# inside a set that excludes them.
MAX_OVERRULE = 0.15
MAX_SILENCED = 0.10

# PROVISIONAL tier.  Measured on ST at 400 / 1200 / 3000 states, the frontier is
# STABLE, so this is a property of the head and not of the sample size:
#
#   beta    AUC    shot rate   untouched states left alone
#    3.0    0.876      9 %            98 %
#    4.0    0.876     31 %            94 %
#    6.0    0.875     65 %            85 %
#    8.0    0.875     72 %            80 %
#
# There is no beta that reaches a 60 % shot rate while leaving 90 % of the other
# decisions alone, and more data does not move it (the probe's own AUC against
# the label is the binding constraint, ~0.88 under the carried recurrence).
# Pretending otherwise by lowering MIN_CLEAN would be the same vacuous pass that
# let a collapsed critic ship.  So a head that clears a usable bar but not the
# strict one is shipped as PROVISIONAL: it exists, it is measured, it is NOT
# eligible for promotion without a real-match gate.
PROVISIONAL_AUC = 0.80
PROVISIONAL_SHOT_RATE = 0.40


# ─────────────────────────────────────────────────────────────
# STATE BUILDING
# ─────────────────────────────────────────────────────────────

#: Share of the fitting corpus drawn from SCORING situations.  The rest comes
#: from the ordinary pitch.  This is not a detail — see `build_corpus`.
DEFAULT_SCORING_FRAC = 0.5


def _one_state(rng: random.Random, position: str,
               scoring_frac: float = 1.0):
    """One synthetic situation, assembled exactly as the pitch does.

    ``scoring_frac`` decides whether this state comes from the final-third
    generator or from the whole pitch.  The first version of this file used the
    scoring generator for EVERY state, and a real-match gate then showed why
    that was wrong: shots fell from 12 to 7 across two matches instead of
    rising, because a striker spends most of a match outside the final third,
    and the head had never been fitted on a single state there.  It vetoed
    shots in places it had no business having an opinion about.
    """
    if rng.random() < scoring_frac:
        st = random_game_state_scoring(rng, position)
    else:
        st = random_game_state(rng, position)
    t_coords, d_coords = st["teammates"], st["defenders"]
    names = {}
    for i, (tx, ty) in enumerate(t_coords):
        names[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(d_coords):
        names[f"d{i}"] = (dx, dy)
    pe = _DummyPositionEngine(names)

    teammates = _role_coords_to_players(t_coords, "t", _OUTFIELD_TM_POOL, rng)
    defenders = _role_coords_to_players(d_coords, "d", _OPP_POOL, rng)
    carrier = _carrier(position)

    # The label, from the coach's own authority — the matrix owns the noun.
    viability = float(shot_score(st["x"], st["y"], defenders, pe,
                                 st["attacks_right"], dna=None))

    base = extract_sensors(
        None, st["x"], st["y"], teammates, defenders, pe,
        st["under_pressure"], st["attacks_right"], st["game_state"],
        st["minute"])
    role, mem, plan = situation_blocks(
        carrier, st["x"], st["y"], teammates, defenders, pe,
        st["attacks_right"], base)
    tok, msk = entity_tokens(
        carrier, st["x"], st["y"], teammates, defenders, pe,
        st["attacks_right"])
    return tok, msk, base, role, mem, plan, viability


def build_corpus(position: str, n_states: int, seed: int,
                 seq_len: int = 5,
                 scoring_frac: float = DEFAULT_SCORING_FRAC) -> Dict[str, np.ndarray]:
    """Sequences of consecutive scoring states, plus their viability.

    Sequences exist so the hidden state can be collected with `h_prev` carried,
    which is the distribution the head will meet in a match.  The states inside
    a sequence are independent draws — there is no attempt to simulate a real
    possession — but the RECURRENCE is real, which is the only thing this needs
    from the sequence.
    """
    rng = random.Random(seed)
    n_seq = max(1, n_states // seq_len)
    rows = []
    for _ in range(n_seq):
        seq = [_one_state(rng, position, scoring_frac) for _ in range(seq_len)]
        rows.append(seq)

    out = {
        "tokens": np.asarray([[s[0] for s in seq] for seq in rows]),
        "mask": np.asarray([[s[1] for s in seq] for seq in rows]),
        "state": np.asarray([[s[2] for s in seq] for seq in rows]),
        "role": np.asarray([[s[3] for s in seq] for seq in rows]),
        "memory": np.asarray([[s[4] for s in seq] for seq in rows]),
        "plan": np.asarray([[s[5] for s in seq] for seq in rows]),
        "viability": np.asarray([[s[6] for s in seq] for seq in rows]),
    }
    out["label"] = (out["viability"] >= SHOT_THRESHOLD).astype(np.float64)
    return out


def collect_hidden_probs(brain, corpus: Dict[str, np.ndarray],
                         carried: bool = True):
    """Run the brain over the corpus; return (hidden, policy probs, viability).

    ``carried=True`` carries `h` across the sequence, ``False`` resets to zeros
    at every step.  BOTH are collected by ``fit`` and the head is fitted on the
    union, because the two answer different questions and the head has to work
    under both:

      reset   — the distribution the diagnostic probe measured, and the one a
                synthetic corpus can actually justify (states inside a
                synthetic "sequence" are independent draws, so a carried h
                encodes an unrelated situation: it is noise here).
      carried — the distribution the head meets in a MATCH, where `h_prev` is
                the same player's previous touch in the same possession and is
                therefore genuinely informative.

    Fitting on one and shipping the other is how a probe reports AUC 0.95
    offline and does nothing on the pitch.  Fitting on the union makes the head
    robust to where the recurrence happens to sit.
    """
    from situational_brain import _softmax

    n_seq, seq_len = corpus["state"].shape[:2]
    h_dim = brain.w_o.shape[0]
    H = np.zeros((n_seq * seq_len, h_dim), dtype=np.float64)
    P = np.zeros((n_seq * seq_len, len(INTENT_LABELS)), dtype=np.float64)
    V = np.zeros(n_seq * seq_len, dtype=np.float64)
    k = 0
    for i in range(n_seq):
        h = np.zeros(h_dim, dtype=np.float64)
        for t in range(seq_len):
            lg, _plg, h_new, _cache = brain.forward(
                corpus["tokens"][i, t], corpus["mask"][i, t],
                corpus["state"][i, t], corpus["role"][i, t],
                corpus["memory"][i, t], corpus["plan"][i, t], h)
            H[k] = h_new
            P[k] = _softmax(lg)
            V[k] = corpus["viability"][i, t]
            k += 1
            h = h_new if carried else np.zeros(h_dim, dtype=np.float64)
    return H, P, V


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    pos, neg = s[y > 0], s[y <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _fit_logistic(H: np.ndarray, y: np.ndarray,
                  epochs: int = 3000, lr: float = 0.35, l2: float = 1e-2):
    """Class-balanced logistic regression, returned in RAW feature space.

    Raw space matters: the head is applied to live hidden states, so it cannot
    depend on the mean and standard deviation of a training set it will never
    see again.
    """
    mu = H.mean(axis=0)
    sd = H.std(axis=0) + 1e-9
    A = (H - mu) / sd
    n, d = A.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(n - y.sum()))
    sw = np.where(y > 0, 0.5 / pos, 0.5 / neg) * n
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(A @ w + b)))
        g = (p - y) * sw
        w -= lr * (A.T @ g / n + l2 * w)
        b -= lr * float(g.mean())
    return w / sd, b - float((mu / sd) @ w)


# ─────────────────────────────────────────────────────────────
# THE ARTIFACT
# ─────────────────────────────────────────────────────────────

class ShotHead:
    """A per-position linear read-out of shot viability from the hidden state."""

    def __init__(self, position: str, w: np.ndarray, b: float, beta: float,
                 meta: Optional[Dict[str, Any]] = None):
        self.position = position
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.b = float(b)
        self.beta = float(beta)
        self.meta = dict(meta or {})

    # ── application ──────────────────────────────────────────
    def score(self, hidden: np.ndarray) -> np.ndarray:
        """The raw log-odds of "this is a shot", before ``beta``."""
        h = np.asarray(hidden, dtype=np.float64)
        return h @ self.w + self.b

    def bias(self, hidden: np.ndarray) -> np.ndarray:
        """The clamped additive bias for the SHOOT logit."""
        return np.clip(self.beta * self.score(hidden), -BIAS_CLAMP, BIAS_CLAMP)

    def apply_logits(self, logits: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """Add the bias to the SHOOT logit and to NOTHING else.

        This is the entire integration surface.  Because the other nine logits
        are untouched, no intent outside SHOOT can move relative to another —
        the only decision this can change is one where SHOOT becomes or stops
        being the argmax.
        """
        lg = np.array(logits, dtype=np.float64, copy=True)
        b = self.bias(hidden)
        if lg.ndim == 1:
            lg[SHOOT_IDX] += float(b)
        else:
            lg[..., SHOOT_IDX] += b
        return lg

    # ── persistence ──────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "beta": self.beta,
            "b": self.b,
            "w": [float(x) for x in self.w],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShotHead":
        return cls(d["position"], d["w"], d["b"], d["beta"], d.get("meta"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ShotHead":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ─────────────────────────────────────────────────────────────
# FITTING
# ─────────────────────────────────────────────────────────────

def fit(brain, position: str, n_states: int = 600, seed: int = 4242,
        seq_len: int = 5, holdout: float = 0.25,
        scoring_frac: float = DEFAULT_SCORING_FRAC,
        betas: Tuple[float, ...] = DEFAULT_BETA_GRID
        ) -> Tuple[Optional[ShotHead], Dict[str, Any]]:
    """Fit a ShotHead, or refuse and explain why.

    Returns ``(head, report)``.  ``head`` is ``None`` when no beta meets the
    selection criteria — the caller must treat that as a refusal, not as an
    error and not as a reason to lower the bar.
    """
    # A ROLE THAT CANNOT SHOOT DOES NOT GET A SHOT HEAD.
    #
    # `ROLE_INTENTS` is the project's existing, documented authority on what a
    # role may do: a keeper "does not cross or shoot", a centre-back "does not
    # cross", a full-back "does not shoot".  Only ST / AM / WING carry SHOOT.
    # Without this guard the fitter happily produced a PROVISIONAL head for the
    # GOALKEEPER (41 % shot rate on shootable states) — a number that is
    # arithmetically fine and football nonsense.  The vocabulary is the guard,
    # not a hand-written position list, so it cannot drift from the rest of the
    # engine.
    fam = _role_family(position)
    allowed = ROLE_ONLY_INTENTS["SHOOT"]
    if fam not in allowed:
        return None, {
            "position": position, "role_family": fam, "tier": "REFUSED",
            "verdict": (f"REFUSED — role family {fam} has no SHOOT in its "
                        f"vocabulary (allowed: {sorted(allowed)}). A shot head "
                        f"for this role would be teaching it an action it "
                        f"cannot perform."),
        }

    corpus = build_corpus(position, n_states, seed, seq_len=seq_len,
                          scoring_frac=scoring_frac)
    H_r, P_r, V_r = collect_hidden_probs(brain, corpus, carried=False)
    H_c, P_c, V_c = collect_hidden_probs(brain, corpus, carried=True)

    # Split by SEQUENCE, not by step: consecutive steps in one sequence share a
    # carried hidden state, so splitting mid-sequence would leak the test set
    # into the training set through the recurrence.
    n_seq = corpus["state"].shape[0]
    step = corpus["state"].shape[1]
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n_seq)
    n_te = max(1, int(n_seq * holdout))
    te_seq, tr_seq = order[:n_te], order[n_te:]
    tr = np.concatenate([np.arange(s * step, (s + 1) * step) for s in tr_seq])
    te = np.concatenate([np.arange(s * step, (s + 1) * step) for s in te_seq])

    # FIT ON THE UNION of the two recurrence regimes, so the head does not
    # depend on where the GRU happened to be.  Evaluate each regime separately
    # so a failure in one is visible instead of averaged away.
    H_tr = np.vstack([H_r[tr], H_c[tr]])
    y_tr = np.concatenate([(V_r[tr] >= SHOT_THRESHOLD).astype(np.float64),
                           (V_c[tr] >= SHOT_THRESHOLD).astype(np.float64)])
    w_s, b_s = _fit_logistic(H_tr, y_tr)

    y_r = (V_r[te] >= SHOT_THRESHOLD).astype(np.float64)
    y_c = (V_c[te] >= SHOT_THRESHOLD).astype(np.float64)
    auc_reset = _auc(H_r[te] @ w_s + b_s, y_r)
    auc_carried = _auc(H_c[te] @ w_s + b_s, y_c)

    # Pooled held-out rows: the trials are judged on both regimes together.
    H_te = np.vstack([H_r[te], H_c[te]])
    P_te = np.vstack([P_r[te], P_c[te]])
    V_te = np.concatenate([V_r[te], V_c[te]])
    y_te = np.concatenate([y_r, y_c])

    shootable = y_te > 0
    was_shooting = P_te.argmax(axis=1) == SHOOT_IDX
    collateral = (~shootable) & (~was_shooting)

    trials: List[Dict[str, float]] = []
    chosen = None
    tier = None
    for beta in betas:
        # The SAME clamp the artifact applies — otherwise the trial numbers
        # describe a mechanism that is not the one that ships.
        bias = np.clip(beta * (H_te @ w_s + b_s), -BIAS_CLAMP, BIAS_CLAMP)
        lg = np.log(np.clip(P_te, 1e-12, None))
        lg[:, SHOOT_IDX] += bias
        lg -= lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        pr = e / e.sum(axis=1, keepdims=True)
        ps = pr[:, SHOOT_IDX]
        clean = float((P_te[collateral].argmax(axis=1) ==
                       pr[collateral].argmax(axis=1)).mean()) \
            if collateral.any() else 1.0
        rate = float((pr[shootable].argmax(axis=1) == SHOOT_IDX).mean()) \
            if shootable.any() else 0.0
        auc_v = _auc(ps, y_te)
        # Measured over EVERY held-out state, not over a set that excludes the
        # head's disagreements.  See MAX_OVERRULE for why this exists.
        base_arg = P_te.argmax(axis=1)
        arm_arg = pr.argmax(axis=1)
        overrule = float((arm_arg != base_arg).mean())
        silenced = float(((base_arg == SHOOT_IDX) &
                          (arm_arg != SHOOT_IDX)).mean())
        row = {"beta": beta, "auc": round(auc_v, 4),
               "spearman": round(_spearman(ps, V_te), 4),
               "clean": round(clean, 4), "shot_rate": round(rate, 4),
               "overrule": round(overrule, 4), "silenced": round(silenced, 4)}
        trials.append(row)
        if (chosen is None and auc_v >= MIN_AUC
                and clean >= MIN_CLEAN and rate >= MIN_SHOT_RATE
                and overrule <= MAX_OVERRULE and silenced <= MAX_SILENCED):
            chosen = row
            tier = "ACCEPTED"

    base_rate = float((P_te[shootable].argmax(axis=1) == SHOOT_IDX).mean()) \
        if shootable.any() else 0.0

    report: Dict[str, Any] = {
        "position": position,
        "role_family": fam,
        "states": int(len(V_r)),
        "sequences": int(n_seq),
        "seed": seed,
        "scoring_frac": scoring_frac,
        "shootable_frac": round(float(y_te.mean()), 4),
        "probe_auc_reset": round(auc_reset, 4),
        "probe_auc_carried": round(auc_carried, 4),
        "baseline_shot_rate": round(base_rate, 4),
        "baseline_auc": round(_auc(P_te[:, SHOOT_IDX], y_te), 4),
        "trials": trials,
        "chosen": chosen,
        "criteria": {"min_auc": MIN_AUC, "min_clean": MIN_CLEAN,
                     "min_shot_rate": MIN_SHOT_RATE,
                     "max_overrule": MAX_OVERRULE,
                     "max_silenced": MAX_SILENCED,
                     "provisional_auc": PROVISIONAL_AUC,
                     "provisional_shot_rate": PROVISIONAL_SHOT_RATE},
        "tier": tier,
    }

    if chosen is None:
        # Second pass: the best usable trade-off, explicitly marked provisional.
        # The overrule/silence caps still apply.  They are not a quality bar that
        # a weak head gets to fail — they are the definition of "prior", and a
        # head that breaks them is a different mechanism, not a worse one.
        usable = [t for t in trials
                  if t["auc"] >= PROVISIONAL_AUC
                  and t["shot_rate"] >= PROVISIONAL_SHOT_RATE
                  and t["overrule"] <= MAX_OVERRULE
                  and t["silenced"] <= MAX_SILENCED]
        if usable:
            chosen = max(usable, key=lambda t: (t["clean"], t["auc"]))
            tier = "PROVISIONAL"

    if chosen is None:
        report["verdict"] = (
            "REFUSED — this brain does not encode shot viability well enough "
            "to act on. Nothing is shipped, and the bar is not lowered.")
        report["tier"] = "REFUSED"
        return None, report

    head = ShotHead(position, w_s, b_s, chosen["beta"], meta={
        "states": int(len(V_r)),
        "sequences": int(n_seq),
        "seed": seed,
        "shootable_frac": report["shootable_frac"],
        "probe_auc_reset": report["probe_auc_reset"],
        "probe_auc_carried": report["probe_auc_carried"],
        "holdout_auc": chosen["auc"],
        "holdout_spearman": chosen["spearman"],
        "holdout_clean": chosen["clean"],
        "holdout_shot_rate": chosen["shot_rate"],
        "baseline_shot_rate": report["baseline_shot_rate"],
        "threshold": SHOT_THRESHOLD,
        "tier": tier,
        "fit_at": "2026-11-16",
    })
    # The report dict is built BEFORE the provisional second pass reassigns
    # `chosen`, so these two fields must be written again here.  They were
    # stale (tier None, chosen None) until this line existed — the same
    # build-then-mutate mistake in two places is why it is called out.
    report["tier"] = tier
    report["chosen"] = chosen
    if tier == "ACCEPTED":
        report["verdict"] = (
            f"ACCEPTED at beta={chosen['beta']}: AUC {chosen['auc']:.3f}, "
            f"SHOOT on {chosen['shot_rate']:.0%} of shootable states "
            f"(was {base_rate:.0%}), {chosen['clean']:.0%} of untouched states "
            f"left alone.")
    else:
        report["verdict"] = (
            f"PROVISIONAL at beta={chosen['beta']}: AUC {chosen['auc']:.3f}, "
            f"SHOOT on {chosen['shot_rate']:.0%} of shootable states "
            f"(was {base_rate:.0%}), but only {chosen['clean']:.0%} of untouched "
            f"states are left alone. Not promotable without a real-match gate.")
    return head, report
