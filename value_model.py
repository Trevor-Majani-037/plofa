"""A consequence-learned value model for the on-ball decision system.

Prototype for PLOFA V2's value-based credit assignment.  The current
learning objective is hand-authored: per-intent reward tables, a
per-touch context-reward gate, and a decision-local surrogate table whose
labels come from the IMMEDIATE execution event of each touch.  No piece
of it is learned from what the possession actually became.

This module learns V(s) — the expected chain-level consequence of being
in vision state s — directly from real-match data with ordinary ridge
regression (pure numpy, no ML frameworks).  Targets are real consequences
the engine itself measured (goals, xG generated, shots manufactured,
possession lost), aggregated per possession episode, so credit assignment
is temporal: an early progressive pass shares credit for the goal it set
up, instead of being judged only by whether it connected.

API
---
    vm = ValueModel().fit(X, R)
    V = vm.predict(X)
    vm.score_cv(X, R, folds=5) -> dict

Boilerplate-free by design; keeps every coefficient inspectable
(no hidden layers, no black box), which is the whole point of testing
whether consequences are PAYABLE first.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# Sensor indices (see brain_sensors.py)
_PRESSURE = 16
_FINAL_THIRD = 12
_OWN_HALF = 13
_SPACE = 9
_CENTRAL = 15
_GOAL_DIST = 14
_NEAR_DEF = 4
# Duplicate / constant sensors: player_x==ball_x, player_y==ball_y,
# team_possession is always 1.0 for an on-ball carrier.
_DROP_COLS = (2, 3, 20)

_POSITIONS = [
    "GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF",
]
_POSITION_INDEX = {p: i for i, p in enumerate(_POSITIONS)}


def value_features(sensors: np.ndarray, position: str) -> np.ndarray:
    """Build the interpretable feature vector phi(s) for a vision state.

    Layout: surviving raw sensors (21) + physics interactions (5) +
    position one-hot (11) = 37 features.  The interactions capture the
    football shapes that matter for value: final-third + space, final
    third + central, pressure + space, final-third crowding, and
    goal-proximity + space.
    """
    s = np.asarray(sensors, dtype=np.float64)
    phi = [s[i] for i in range(24) if i not in _DROP_COLS]
    phi.append(s[_FINAL_THIRD] * s[_SPACE])
    phi.append(s[_FINAL_THIRD] * s[_CENTRAL])
    phi.append(s[_PRESSURE] * s[_SPACE])
    phi.append(s[_FINAL_THIRD] * s[_NEAR_DEF])
    phi.append(s[_GOAL_DIST] * s[_SPACE])
    one_hot = np.zeros(len(_POSITIONS))
    idx = _POSITION_INDEX.get(position)
    if idx is not None:
        one_hot[idx] = 1.0
    phi = np.concatenate([np.asarray(phi), one_hot])
    return phi


def value_feature_matrix(
    records: List[Tuple[np.ndarray, str]],
) -> np.ndarray:
    """(n, 37) feature matrix from a list of (sensors, position) records."""
    rows = [value_features(s, pos) for s, pos in records]
    return np.vstack(rows)


def _standardize(
    X: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


class ValueModel:
    """Linear value model V(s) = w . phi(s) via ridge regression.

    Closed-form ridge keeps this dependency-free and stops any black-box
    drift — the whole module exists to answer one empirical question with
    the fewest moving parts: is there PAYABLE signal in real match
    consequences?
    """

    def __init__(self, ridge_lambda: float = 1.0):
        self.ridge_lambda = ridge_lambda
        self.weights: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, R: np.ndarray) -> "ValueModel":
        """Fit on a feature matrix (n, d) and return targets (n,)."""
        Xs, mu, sd = _standardize(X)
        d = Xs.shape[1]
        eye = np.eye(d)
        self.weights = np.linalg.solve(
            Xs.T @ Xs + self.ridge_lambda * eye, Xs.T @ np.asarray(R, dtype=np.float64))
        self._mu, self._sd = mu, sd
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None, "fit() before predict()"
        Xs, _, _ = _standardize(X, self._mu, self._sd)
        return Xs @ self.weights

    def score_cv(self, X: np.ndarray, R: np.ndarray, folds: int = 5,
                 seed: int = 0) -> dict:
        """Out-of-sample evaluation: R^2 and Spearman over stratified-ish folds."""
        n = len(X)
        if n < folds:
            folds = max(1, n)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        splits = np.array_split(perm, folds)
        preds = np.empty(n)
        for fold in splits:
            mask = np.zeros(n, dtype=bool)
            mask[fold] = True
            m = ValueModel(self.ridge_lambda).fit(X[~mask], R[~mask])
            preds[fold] = m.predict(X[fold])
        return _regression_report(preds, np.asarray(R, dtype=np.float64))

    def top_weights(self, k: int = 12) -> List[Tuple[str, float]]:
        """Most influential features by |coefficient| (already standardized)."""
        assert self.weights is not None
        return sorted(zip(_FEATURE_LABELS, self.weights),
                      key=lambda t: -abs(t[1]))[:k]


_FEATURE_LABELS = (
    [f"s{i}" for i in range(24) if i not in _DROP_COLS]
    + ["FT*SPACE", "FT*CENTRAL", "PRESS*SPACE", "FT*NEARDEF", "GOALDIST*SPACE"]
    + [f"pos:{p}" for p in _POSITIONS]
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a_s, b_s = _rank(a), _rank(b)
    return float(np.corrcoef(a_s, b_s)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x)) + 1.0
    return ranks


def _regression_report(pred: np.ndarray, R: np.ndarray) -> dict:
    ss_res = float(np.sum((R - pred) ** 2))
    ss_tot = float(np.sum((R - R.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2_oos": round(r2, 4),
        "spearman": round(_spearman(pred, R), 4),
        "pearson": round(float(np.corrcoef(pred, R)[0, 1]), 4),
        "n": int(len(R)),
    }