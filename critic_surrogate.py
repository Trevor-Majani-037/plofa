"""V3 episode-level critic surrogate: state-conditioned value objective.

V3 replaces V2's per-(position, bucket, intent) payoff TABLE with a
state-conditioned neural critic Q(s, pos)[intent] -> expected attack
return.  Instead of averaging all episodes inside a coarse bucket (which
washed scoring situations under a flat prior), the critic is a numpy MLP
over the full 24-d sensor vector + 11-d position one-hot (35 inputs,
He-init, GA-trained) whose per-intent outputs are the expected attack
payoff (3*goals + xg + 0.2*shots) of the possession that followed the
decision, rescaled into the [0.4, 1.5] reward band.

Situation weighting: training samples from scoring situations / episodes
that actually paid off are over-weighted so rare vertical intents (ST
SHOOT, CM THROUGH_BALL, winger CROSS) are represented in the critic's
argmax landscape — the "situation/episode weighting" fix V2 lacked.

The critic is a drop-in surrogate: it exposes the same
``expected_success(sensors, intent, position)`` interface the GA
(``brain_evolution.synthetic_fitness``) consumes, so evolution runs
unchanged.  Serialized with ``kind: "value_critic"`` and detected by
``load_any_surrogate`` (FitnessSurrogate-loaded table files still work).

Usage:
    python critic_surrogate.py --dataset brains_trainer/corpus_consequences.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from football_brain import INTENT_LABELS, _he_init
from surrogate_collect import FitnessSurrogate

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
CRITIC_INPUT = 24 + len(POSITIONS)         # sensors + position one-hot
H1 = H2 = 32
OUT = len(INTENT_LABELS)

_REWARD_LO, _REWARD_HI = 0.20, 1.5

# Empirical-Bayes shrink target: y_eff = prior + c/(c+k) * (y - prior).
# The count prior is per (position, intent), so rare vertical intents are
# pulled toward their (noisy) cell mean instead of being fitted as 0/1.
SHRINK_K = 5.0


def _attack_payoff(record: dict) -> float:
    return 3.0 * record["goals"] + record["xg"] + 0.2 * record["shots"]


def _reward(record: dict) -> float:
    """Ordinal reward ladder on the [0.20, 1.5] band.

    The v3 band [0.4, 1.5] was only START and SCORE — a turnover (the most
    common outcome, ~1 in 2 decisions) sat on the same 0.4 floor as a
    harmless pass, so the target carried almost no ranking information and
    the critic regressed to the mean (rho ~ 0).  v4 makes the ladder
    monotone in the episode return:

        turnover   ~0.20       (possession lost — WORSE than nothing)
        nothing    ~0.55       (possession continues)
        off-target  ~0.60      (shot, no goal / xG)
        goal       ~1.35       (2 goals clamped at 1.5)

    with xG riding in between.  The critic must therefore learn to order
    intents by their OWN outcome distribution, not by the floor.
    """
    v = (0.55
         + 0.8 * record["goals"]
         + 0.25 * record["xg"]
         + 0.05 * record["shots"]
         - 0.25 * max(0, int(record.get("turnovers", 0))))
    return float(np.clip(v, _REWARD_LO, _REWARD_HI))


def _weight(record: dict, sensors: np.ndarray) -> float:
    """Over-weight scoring situations and episodes that actually paid off.

    ``balance`` (inverse-frequency per position/intent, applied in
    ``_prepare`` after counts are known) additionally rescues rare
    vertical intents — ST SHOOT, CM THROUGH_BALL, winger CROSS — from
    being statistically indistinguishable under the pass/recycle mass.
    """
    w = 1.0
    if record["goals"] > 0 or record["shots"] > 0:
        w *= 4.0
    if record.get("turnovers", 0) > 0:
        w *= 2.0
    if sensors[12] > 0.5 or sensors[14] < 0.5:      # final third / goal close
        w *= 2.0
    else:
        w *= 1.5
    return w


def _intent_idx(intent: Any) -> int:
    ik = intent.value if hasattr(intent, "value") else str(intent)
    return INTENT_LABELS.index(ik)


def _onehot_position(position: str) -> np.ndarray:
    v = np.zeros(len(POSITIONS), dtype=np.float64)
    if position in POSITIONS:
        v[POSITIONS.index(position)] = 1.0
    return v


class ValueCritic:
    """35 -> 32 -> 32 -> 10 numpy MLP, GA-trained, linear output head."""

    __slots__ = ("w1", "b1", "w2", "b2", "w3", "b3",
                 "sensor_means", "sensor_stds", "meta")

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
        sensor_means: Optional[np.ndarray] = None,
        sensor_stds: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
        self.sensor_means = sensor_means if sensor_means is not None \
            else np.zeros(24, dtype=np.float64)
        self.sensor_stds = sensor_stds if sensor_stds is not None \
            else np.ones(24, dtype=np.float64)
        self.meta = meta

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "ValueCritic":
        rng = np.random.default_rng(seed)
        return cls(
            _he_init(CRITIC_INPUT, H1, rng),
            np.zeros(H1, dtype=np.float64),
            _he_init(H1, H2, rng),
            np.zeros(H2, dtype=np.float64),
            _he_init(H2, OUT, rng),
            np.zeros(OUT, dtype=np.float64),
        )

    def mutate(self, rate: float = 0.15, strength: float = 0.3) -> "ValueCritic":
        rng = np.random.default_rng()
        def _perturb(w: np.ndarray) -> np.ndarray:
            m = rng.random(w.shape) < rate
            noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
            return w + m * noise
        return ValueCritic(
            _perturb(self.w1), _perturb(self.b1),
            _perturb(self.w2), _perturb(self.b2),
            _perturb(self.w3), _perturb(self.b3),
            self.sensor_means, self.sensor_stds, self.meta,
        )

    def crossover(self, other: "ValueCritic") -> "ValueCritic":
        rng = np.random.default_rng()
        def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.where(rng.random(a.shape) < 0.5, a, b)
        return ValueCritic(
            _cross(self.w1, other.w1), _cross(self.b1, other.b1),
            _cross(self.w2, other.w2), _cross(self.b2, other.b2),
            _cross(self.w3, other.w3), _cross(self.b3, other.b3),
            self.sensor_means, self.sensor_stds, self.meta,
        )

    def build_input(self, sensors: np.ndarray, position: str) -> np.ndarray:
        z = (np.asarray(sensors, dtype=np.float64) - self.sensor_means) / self.sensor_stds
        return np.concatenate([z, _onehot_position(position)])

    def forward_x(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        return h2 @ self.w3 + self.b3

    def predict(self, sensors: np.ndarray, position: str) -> np.ndarray:
        return self.forward_x(self.build_input(sensors, position))

    def serialize(self) -> Dict[str, Any]:
        return {
            "kind": "value_critic",
            "arch": [CRITIC_INPUT, H1, H2, OUT],
            "sensor_means": self.sensor_means.tolist(),
            "sensor_stds": self.sensor_stds.tolist(),
            "meta": self.meta or {"training_method": "ga_critic", "sensor_schema": "v1_24d"},
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ValueCritic":
        return cls(
            np.array(data["w1"], dtype=np.float64),
            np.array(data["b1"], dtype=np.float64),
            np.array(data["w2"], dtype=np.float64),
            np.array(data["b2"], dtype=np.float64),
            np.array(data["w3"], dtype=np.float64),
            np.array(data["b3"], dtype=np.float64),
            np.array(data.get("sensor_means", [0.0] * 24), dtype=np.float64),
            np.array(data.get("sensor_stds", [1.0] * 24), dtype=np.float64),
            data.get("meta"),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ValueCritic":
        with open(path) as f:
            data = json.load(f)
        if data.get("kind") != "value_critic":
            raise ValueError(f"not a value_critic file: {path}")
        return cls.deserialize(data)


class CriticSurrogate:
    """Drop-in ``FitnessSurrogate``-compatible wrapper over a ValueCritic."""

    def __init__(self, critic: ValueCritic):
        self.critic = critic

    def expected_success(self, sensors: np.ndarray, intent: Any,
                         position: str = "") -> float:
        q = self.critic.predict(sensors, position)
        v = float(q[_intent_idx(intent)])
        return float(np.clip(v, _REWARD_LO, _REWARD_HI))

    @property
    def table(self) -> Dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        self.critic.save(path)

    @classmethod
    def load(cls, path: str) -> "CriticSurrogate":
        return cls(ValueCritic.load(path))

    def report(self) -> Dict[str, Any]:
        return {"kind": "value_critic", "arch": [CRITIC_INPUT, H1, H2, OUT]}


def load_any_surrogate(path: str) -> Any:
    """Load either a v1/v2 ``{prior, table, counts}`` surrogate or a V3 critic."""
    with open(path) as f:
        head = json.load(f)
    if isinstance(head, dict) and head.get("kind") == "value_critic":
        return CriticSurrogate.load(path)
    return FitnessSurrogate.load(path)


def _load_dataset(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("records", data)


def _prepare(records: list):
    """(X, I, y, w) arrays + sensor normalization statistics.

    Weights combine the per-record payoff/scoring multipliers with an
    inverse-frequency balance per (position, intent) so the weighted MSE
    fits every intent's conditional mean rather than the pass mass.  The
    target is empirically-Bayes shrunk toward each (position, intent)
    prior (``SHRINK_K`` strength) so rare vertical intents are stabilised
    instead of fitting their sparse, noisy realizations as-is.
    """
    n = len(records)
    raw = np.zeros((n, 24), dtype=np.float64)
    I = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float64)
    base = np.zeros(n, dtype=np.float64)
    pos = [""] * n
    counts: Dict[Tuple[str, int], int] = {}
    for i, r in enumerate(records):
        s = np.asarray(r["sensors"], dtype=np.float64)
        if len(s) < 20:
            continue
        raw[i] = s[:24]
        ik = _intent_idx(r["intent"])
        I[i] = ik
        pos[i] = r.get("position", "")
        y[i] = _reward(r)
        base[i] = _weight(r, s)
        counts[(pos[i], ik)] = counts.get((pos[i], ik), 0) + 1
    pos_max = {}
    for (p, _), c in counts.items():
        pos_max[p] = max(pos_max.get(p, 0), c)
    w = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(records):
        c = counts.get((pos[i], I[i]), 1)
        m = pos_max.get(pos[i], c)
        bal = float(np.clip(math.sqrt(max(1, m) / max(1, c)), 0.5, 8.0))
        w[i] = base[i] * bal
    prior_sum: Dict[Tuple[str, int], float] = {}
    for i in range(n):
        k = (pos[i], I[i])
        if counts.get(k, 0) > 0:
            prior_sum[k] = prior_sum.get(k, 0.0) + y[i]
    for i in range(n):
        k = (pos[i], I[i])
        c = counts.get(k, 0)
        if c <= 0:
            continue
        prior = prior_sum[k] / c
        shrink = c / (c + SHRINK_K)
        y[i] = prior + shrink * (y[i] - prior)
    means = raw.mean(axis=0)
    stds = raw.std(axis=0) + 1e-9
    z = (raw - means) / stds
    X = np.zeros((n, CRITIC_INPUT), dtype=np.float64)
    X[:, :24] = z
    for i, r in enumerate(records):
        X[i, 24:] = _onehot_position(r.get("position", ""))
    return X, I, y, w, means, stds


def _weighted_mse(critic: ValueCritic, X: np.ndarray, I: np.ndarray,
                  y: np.ndarray, w: np.ndarray) -> float:
    q = critic.forward_x(X)
    n = X.shape[0]
    return float(((q[np.arange(n), I] - y) ** 2 * w).sum() / w.sum())


def train_critic(
    X: np.ndarray, I: np.ndarray, y: np.ndarray, w: np.ndarray,
    means: np.ndarray, stds: np.ndarray,
    population_size: int = 32,
    generations: int = 100,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 123,
    verbose: bool = True,
) -> Tuple[ValueCritic, List[float], List[float]]:
    rng = np.random.default_rng(seed)
    pop = [ValueCritic.random(seed=seed + i) for i in range(population_size)]
    best_overall: Optional[ValueCritic] = None
    best_loss = math.inf
    gen_loss: List[float] = []
    gen_mean: List[float] = []
    mutate_rate = base_mutate_rate

    for gen in range(generations):
        losses = [_weighted_mse(c, X, I, y, w) for c in pop]
        order = sorted(range(population_size), key=lambda i: losses[i])
        bf = losses[order[0]]
        mf = float(np.mean(losses))
        gen_loss.append(bf)
        gen_mean.append(mf)
        best = pop[order[0]].mutate(rate=0.0)
        bc = ValueCritic(best.w1.copy(), best.b1.copy(), best.w2.copy(),
                         best.b2.copy(), best.w3.copy(), best.b3.copy(),
                         means, stds)
        if bf < best_loss:
            best_loss = bf
            best_overall = bc
        if verbose:
            print(f"  gen {gen+1:3d}  best={bf:.5f}  mean={mf:.5f}  "
                  f"mut_rate={mutate_rate:.3f}")

        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]
        n_parents = max(2, int(population_size * crossover_top))
        parents = [pop[i] for i in order[:n_parents]]
        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2)
            child = child.mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)
        pop = next_pop
        mutate_rate = max(0.05, mutate_rate * mutate_decay)

    if best_overall is None:
        best_overall = pop[0]
    best_overall.meta = {
        "training_method": "ga_critic",
        "sensor_schema": "v1_24d",
        "generations": generations,
        "population": population_size,
        "seed": seed,
        "best_loss": round(best_loss, 6),
    }
    return best_overall, gen_loss, gen_mean


def train_critic_backprop(
    X: np.ndarray, I: np.ndarray, y: np.ndarray, w: np.ndarray,
    Xv: np.ndarray, Iv: np.ndarray, yv: np.ndarray, wv: np.ndarray,
    means: np.ndarray, stds: np.ndarray,
    epochs: int = 200,
    batch: int = 512,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patience: int = 25,
    seed: int = 7,
    verbose: bool = True,
) -> Tuple[ValueCritic, List[float], List[float]]:
    """Numpy Adam backprop for the 35->32->32->10 critic.

    Weighted MSE on the CHOSEN-intent output only (each decision labels
    exactly one intent), L2 weight decay, holdout early-stopping.  This is
    the training regime the sparse/noisy episode targets need: GA fit the
    constant floor, backprop + early stopping can chase the same loss
    much faster and regularise properly.
    """
    rng = np.random.default_rng(seed)
    crit = ValueCritic.random(seed=seed)
    w1, b1 = crit.w1.copy(), crit.b1.copy()
    w2, b2 = crit.w2.copy(), crit.b2.copy()
    w3, b3 = crit.w3.copy(), crit.b3.copy()

    def _zeros_like(w):
        return np.zeros_like(w)

    def _adam_update(param, grad, ms, vs, step):
        ms = 0.9 * ms + 0.1 * grad
        vs = 0.999 * vs + 0.001 * grad * grad
        mhat = ms / (1.0 - 0.9 ** step)
        vhat = vs / (1.0 - 0.999 ** step)
        return param - (lr / (1.0 + 1e-3 * step)) * mhat / (np.sqrt(vhat) + 1e-8), ms, vs

    best_w = None
    best_loss = math.inf
    wait = 0
    gen_eval: List[float] = []
    gen_mean: List[float] = []
    n = X.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = X[idx]
            ib = I[idx]
            yb = y[idx]
            wb = w[idx]
            k = len(idx)

            h1 = np.maximum(0.0, xb @ w1 + b1)
            h2 = np.maximum(0.0, h1 @ w2 + b2)
            out = h2 @ w3 + b3

            sel = out[np.arange(k), ib]
            gsel = 2.0 * (sel - yb) * wb / wb.sum()

            g3 = np.zeros_like(out)
            g3[np.arange(k), ib] = gsel
            gW3 = h2.T @ g3
            gb3 = g3.sum(axis=0)
            gh2 = g3 @ w3.T * (h2 > 0.0)
            gW2 = h1.T @ gh2
            gb2 = gh2.sum(axis=0)
            gh1 = gh2 @ w2.T * (h1 > 0.0)
            gW1 = xb.T @ gh1
            gb1 = gh1.sum(axis=0)

            gW1 += weight_decay * w1
            gW2 += weight_decay * w2
            gW3 += weight_decay * w3

            if start == 0:
                sw1, sw2, sw3 = np.zeros_like(w1), np.zeros_like(w2), np.zeros_like(w3)
                sb1, sb2, sb3 = np.zeros_like(b1), np.zeros_like(b2), np.zeros_like(b3)
                vw1, vw2, vw3 = np.zeros_like(w1), np.zeros_like(w2), np.zeros_like(w3)
                vb1, vb2, vb3 = np.zeros_like(b1), np.zeros_like(b2), np.zeros_like(b3)
            step = ep * max(1, n // batch) + start // batch + 1
            w1, sw1, vw1 = _adam_update(w1, gW1, sw1, vw1, step)
            w2, sw2, vw2 = _adam_update(w2, gW2, sw2, vw2, step)
            w3, sw3, vw3 = _adam_update(w3, gW3, sw3, vw3, step)
            b1, sb1, vb1 = _adam_update(b1, gb1, sb1, vb1, step)
            b2, sb2, vb2 = _adam_update(b2, gb2, sb2, vb2, step)
            b3, sb3, vb3 = _adam_update(b3, gb3, sb3, vb3, step)

        tmp = ValueCritic(w1.copy(), b1.copy(), w2.copy(), b2.copy(),
                          w3.copy(), b3.copy(), means, stds)
        vloss = _weighted_mse(tmp, Xv, Iv, yv, wv)
        gen_eval.append(vloss)
        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_w = (w1.copy(), b1.copy(), w2.copy(), b2.copy(),
                      w3.copy(), b3.copy())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose:
            print(f"  epoch {ep+1:3d}  holdout={vloss:.5f}  "
                  f"(early stop in {patience - wait})")

    if best_w is None:
        best_w = (w1.copy(), b1.copy(), w2.copy(), b2.copy(),
                  w3.copy(), b3.copy())
    best = ValueCritic(best_w[0], best_w[1], best_w[2], best_w[3],
                       best_w[4], best_w[5], means, stds)
    return best, gen_eval, gen_mean


def _td_bootstrap(records: list, y_mc: np.ndarray, critic: ValueCritic,
                  gamma: float = 0.97, lo: float = _REWARD_LO,
                  hi: float = _REWARD_HI) -> np.ndarray:
    """TD(0) targets replacing single-roll Monte-Carlo returns.

    Only the LAST decision of a possession carries the full realized return
    (the anchor).  Every earlier decision in the same episode inherits
    ``gamma * Q(s_next, intent_next)`` — the bootstrap read of the state
    that actually followed — so the noisy final-outcome variance is
    attached to the final touch alone instead of washing over the whole
    build-up.  This is the variance lever: MC returns at ~63% turnover mass
    are near-noise for early touches; TD targets are smooth.

    ``y_mc`` is the post-shrink realized target from ``_prepare`` (kept for
    terminal records).  Episode linkage comes from the ``episode_idx`` /
    ``episode_team`` fields the collector now records.
    """
    groups: Dict[Tuple[Any, Any], List[int]] = {}
    for i, r in enumerate(records):
        key = (r.get("episode_idx"), r.get("episode_team"))
        groups.setdefault(key, []).append(i)
    y = y_mc.copy()
    for members in groups.values():
        if len(members) < 2:
            continue
        ms = sorted(members, key=lambda i: (float(records[i].get("minute", 0)), i))
        for pos in range(len(ms) - 1):
            cur = ms[pos]
            nxt = ms[pos + 1]
            s_next = np.asarray(records[nxt]["sensors"], dtype=np.float64)[:24]
            ik = _intent_idx(records[nxt]["intent"])
            q_next = float(critic.predict(
                s_next, records[nxt].get("position", ""))[ik])
            y[cur] = float(np.clip(gamma * q_next, lo, hi))
    return y


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _argmax_profile(critic: ValueCritic, position: str,
                    n_states: int = 300, goal_bias: float = 0.0,
                    seed: int = 21) -> Dict[str, Any]:
    """Argmax intent shares over synthetic sensor states — the exact
    landscape `brain_evolution.synthetic_fitness` evolves against."""
    from brain_evolution import generate_state_corpus
    corpus = generate_state_corpus(position, n_states, seed, goal_bias=goal_bias)
    counts = np.zeros(OUT, dtype=np.float64)
    values = np.zeros(OUT, dtype=np.float64)
    for s in corpus:
        q = critic.predict(s, position)
        idx = int(np.argmax(q))
        counts[idx] += 1.0
        values[idx] += float(np.clip(q[idx], _REWARD_LO, _REWARD_HI))
    shares = counts / counts.sum()
    top = []
    for ik in np.argsort(-shares)[:8]:
        if shares[ik] > 0.0:
            top.append((INTENT_LABELS[ik], round(float(shares[ik]), 3),
                        round(float(values[ik] / max(counts[ik], 1.0)), 3)))
    return {"n_states": n_states, "goal_bias": goal_bias, "top": top}


def _scoring_profile(critic: ValueCritic, records: list, pos: str) -> Dict[str, Any]:
    """Mean predicted per-intent value on real scoring-situation records."""
    acc = {}
    for r in records:
        if r.get("position") != pos:
            continue
        s = np.asarray(r["sensors"], dtype=np.float64)
        if s[12] <= 0.5 and s[14] >= 0.5:
            continue
        q = critic.predict(s, pos)
        for ik, v in zip(INTENT_LABELS, q):
            acc.setdefault(ik, []).append(float(np.clip(v, _REWARD_LO, _REWARD_HI)))
    if not acc:
        return {"samples": 0, "top": []}
    means = {ik: float(np.mean(vs)) for ik, vs in acc.items()}
    return {
        "samples": sum(len(v) for v in acc.values()),
        "top": sorted(means.items(), key=lambda x: -x[1])[:6],
    }


def build_critic(records: list, out_path: str, report_path: str,
                 generations: int = 100, population: int = 32,
                 seed: int = 123, method: str = "backprop",
                 epochs: int = 200, target_mode: str = "td") -> ValueCritic:
    X, I, y, w, means, stds = _prepare(records)
    n = len(records)
    hold = max(1, int(n * 0.2))
    rorder = np.arange(n)
    np.random.default_rng(seed + 999).shuffle(rorder)
    tr, te = rorder[hold:], rorder[:hold]

    if method == "ga":
        print(f"{n} records; GA-training critic (pop={population}, "
              f"gen={generations}, seed={seed}) ...")
        critic, gen_best, gen_mean = train_critic(
            X[tr], I[tr], y[tr], w[tr], means, stds,
            population_size=population, generations=generations, seed=seed,
        )
    else:
        if target_mode == "td":
            print(f"{n} records; TD(0) targets: bootstrap critic (mc) -> "
                  f"rewrite targets by episode -> backprop ...")
            boot, _, _ = train_critic_backprop(
                X[tr], I[tr], y[tr], w[tr], X[te], I[te], y[te], w[te],
                means, stds, epochs=max(25, epochs // 6), seed=seed,
                verbose=False)
            y = _td_bootstrap(records, y, boot, gamma=0.97,
                              lo=_REWARD_LO, hi=_REWARD_HI)
        print(f"{n} records; backprop-training critic on "
              f"{target_mode.upper()} targets (epochs={epochs}, "
              f"batch=512, seed={seed}) ...")
        critic, gen_best, gen_mean = train_critic_backprop(
            X[tr], I[tr], y[tr], w[tr], X[te], I[te], y[te], w[te],
            means, stds, epochs=epochs, seed=seed,
        )
    te_loss = _weighted_mse(critic, X[te], I[te], y[te], w[te])
    q = critic.forward_x(X)
    n_all = X.shape[0]
    rho_all = _spearman(q[np.arange(n_all), I], y)
    qte = critic.forward_x(X[te])
    rho_te = _spearman(qte[np.arange(len(te)), I[te]], y[te])

    # Outcome separation: does predicted q on holdout separate low- from
    # high-yield holdings at all (best-quartile minus worst-quartile mean y)?
    qt = qte[np.arange(len(te)), I[te]].argsort()
    nq = max(1, len(te) // 4)
    sep = float(y[te][qt[-nq:]].mean() - y[te][qt[:nq]].mean())

    # Honest transfer check against RAW episode outcomes (not the shrunk
    # training target): does holdout q rank real payoffs, and does the
    # top-decile q hand pick episodes that actually paid off?
    yraw = np.asarray([_reward(r) for r in records], dtype=np.float64)
    qte_sel = qte[np.arange(len(te)), I[te]]
    rho_raw_te = _spearman(qte_sel, yraw[te])
    n10 = max(1, len(te) // 10)
    top_recs = qt[-n10:]
    catch_top = float(yraw[te][top_recs].mean())
    base_mean = float(yraw[te].mean())

    report = {
        "kind": "value_critic",
        "n_records": n, "holdout": hold,
        "method": method, "target_mode": target_mode, "gamma_td": 0.97,
        "epochs": epochs if method == "backprop" else None,
        "train_loss": round(gen_best[-1], 6), "test_loss": round(te_loss, 6),
        "rho_all": round(rho_all, 4), "rho_holdout": round(rho_te, 4),
        "rho_raw_holdout": round(rho_raw_te, 4),
        "separation_holdout": round(sep, 4),
        "payoff_indicator_share": round(float(len(te)) / max(1, len(te)), 4),
        "raw_top_decile_catch": round(catch_top, 4),
        "raw_mean": round(base_mean, 4),
        "reward_band": [_REWARD_LO, _REWARD_HI], "shrink_k": SHRINK_K,
        "generations": generations, "population": population, "seed": seed,
        "scoring_profiles": {
            pos: _scoring_profile(critic, records, pos)
            for pos in ["ST", "CM", "CAM", "LW", "RW"]
        },
        "synthetic_argmax": {
            f"{pos}_scoring": _argmax_profile(critic, pos, goal_bias=1.0, n_states=300, seed=21)
            for pos in ["ST", "CM", "CAM", "LW", "RW"]
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    critic.meta = {
        "training_method": method,
        "target_mode": target_mode,
        "gamma_td": 0.97 if target_mode == "td" else None,
        "sensor_schema": "v1_24d",
        "epochs": epochs if method == "backprop" else None,
        "generations": generations if method == "ga" else None,
        "population": population if method == "ga" else None,
        "seed": seed,
        "best_loss": round(gen_best[-1], 6),
        "reward_band": [_REWARD_LO, _REWARD_HI],
        "shrink_k": SHRINK_K,
    }
    critic.save(out_path)
    print(f"Critic -> {out_path}   report -> {report_path}")
    print(f"train_loss={gen_best[-1]:.5f}  test_loss={te_loss:.5f}  "
          f"rho(train)={rho_all:.3f}  rho(holdout)={rho_te:.3f}  "
          f"rho_raw(holdout)={rho_raw_te:.3f}  sep(holdout)={sep:.4f}  "
          f"top-decile catch={catch_top:.3f} vs mean {base_mean:.3f}")
    print("\n=== V3 critic scoring-situation profiles (top-6 intents) ===")
    for pos, prof in report["scoring_profiles"].items():
        if prof["top"]:
            print(f"  {pos:4s} (n={prof['samples']:5d}): "
                  + "  ".join(f"{ik}={v:.3f}" for ik, v in prof["top"]))
        else:
            print(f"  {pos:4s}: no scoring-situation samples")
    print("\n=== V3 critic synthetic-scoring argmax (goal_bias=1.0, 300 states) ===")
    for key, prof in report["synthetic_argmax"].items():
        top = "  ".join(f"{ik}={sh:.3f}@{v:.3f}" for ik, sh, v in prof["top"][:5])
        print(f"  {key:12s}: {top}")
    return critic


def main() -> None:
    p = argparse.ArgumentParser(description="V3 episode-level critic surrogate builder.")
    p.add_argument("--dataset", default="brains_trainer/corpus_consequences.json")
    p.add_argument("--out", default="brains_trainer/critic_v3.json")
    p.add_argument("--report", default="value_experiments/v3_critic_report.json")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--method", choices=["backprop", "ga"], default="backprop")
    p.add_argument("--target", choices=["td", "mc"], default="td",
                   help="'td' bootstraps episode targets (gamma=0.97); "
                        "'mc' uses single-roll realized returns")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    records = _load_dataset(args.dataset)
    print(f"Loaded {len(records)} decision records from {args.dataset}")
    build_critic(records, args.out, args.report,
                 generations=args.generations, population=args.population,
                 seed=args.seed, method=args.method, epochs=args.epochs,
                 target_mode=args.target)


if __name__ == "__main__":
    main()