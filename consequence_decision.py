"""Policy + Consequence Evaluation -> Decision (PLOFA reasoning seam).

The FootballBrain proposes intent probabilities (``probs``) from the
perceived sensors — "what could I do?"  The value critic predicts the
expected attack-payoff consequence of EACH intent in the SAME state —
"what is likely to happen if I do it?"  The final distribution samples
over a consequence-corrected version of the policy:

    q_i        = critic(s, intent_i, position)          # expected payoff
    q_ref      = sum_i p_i * q_i                        # my usual consequence here
    advantage_i = q_i - q_ref                           # does it beat my baseline?
    logit_i    = log(p_i + eps) + blend * advantage_i
    p'_i       = softmax(logit)

The critic corrects the policy's ranking by expected consequence; it
never overrides it.  ``blend == 0`` returns ``probs`` unchanged
(byte-identical to the vanilla neural path), so the seam is provably a
condition, not a replacement — the same discipline as perception and
cognition.

State is global and process-wide (like ``set_perception`` /
``set_cognition``): ``set_consequence(critic_path, blend)`` loads the
critic once; ``get_consequence()`` returns the active evaluator or None.
The match engine's ``_decide_core`` applies it in one call,
``apply_reasoning(probs, sensor_24, position)``, after the coach bias.

Since 2026-09-20 the reasoning seam is ON BY DEFAULT (the v5 TD critic at
blend 0.3 graduated the gate); disable per-process via
``PLOFA_CONSEQUENCE=0`` or ``default_enabled(False)``.

The v1 seam handles the vanilla neural path only.  Engaged minds route
through CognitionDecisionBrain (its own perception/sampling pipeline);
reasoning there is a future-merge seam, not this one.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from football_brain import INTENT_LABELS


# ─────────────────────────────────────────────────────────────
# CONSEQUENCE EVALUATOR
# ─────────────────────────────────────────────────────────────

class ConsequenceEvaluator:
    """Expected-payoff critic plus the blend knob for consequence-aware
    sampling.  Wraps anything exposing ``predict(sensors, position) ->
    (10,) payoff vector`` from the same 24-d base the brain read.

    The critic output band is [0.4, 1.5] (critic_surrogate reward band);
    advantage is computed relative to the policy's own expected payoff in
    this state, which makes the correction state- and player-relative
    rather than an absolute threshold.
    """

    # Reward band the critic was trained on.  Primarily read from the
    # critic's ``meta["reward_band"]`` (v4 emits it); legacy critics fall
    # back to the old [0.4, 1.5] band.  The critic fits an unconstrained
    # regression head whose raw output can explode on out-of-distribution
    # sensors; clamping back to the training band keeps the advantage
    # signal in calibrated payoff units and makes the correction scale
    # well-conditioned at every snapshot.
    _REWARD_LO = 0.4
    _REWARD_HI = 1.5

    def __init__(self, critic: Any, blend: float = 0.5):
        self.critic = critic
        self.blend = float(blend)
        meta = getattr(critic, "meta", None) or {}
        band = meta.get("reward_band")
        if isinstance(band, (list, tuple)) and len(band) == 2:
            try:
                self._REWARD_LO = float(band[0])
                self._REWARD_HI = float(band[1])
            except (TypeError, ValueError):
                pass

    def classify(self, sensor_24: np.ndarray, position: str,
                 probs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(q, advantage) for every intent in the current state.

        ``q`` is the critic's clipped expected-payoff prediction per intent;
        when ``probs`` is given, ``advantage`` is de-meaned against the
        policy-weighted reference.  Without ``probs`` the reference is
        the mean payoff (a neutral, non-policy baseline).
        """
        q = np.asarray(self.critic.predict(sensor_24, position),
                       dtype=np.float64)
        q = np.clip(q, self._REWARD_LO, self._REWARD_HI)
        if probs is None:
            ref = float(np.mean(q))
        else:
            p = np.asarray(probs, dtype=np.float64)
            ref = float(p @ q) if p.sum() > 0 else float(np.mean(q))
        return q, q - ref

    def correct_policy(self, probs: Any, sensor_24: np.ndarray,
                       position: str) -> np.ndarray:
        """Consequence-corrected probability vector (softmax normalised).

        ``blend == 0`` returns ``probs`` unchanged; ``blend > 0`` shifts
        mass toward intents whose predicted consequence beats this
        player's own expected consequence in the same state.
        """
        p = np.asarray(probs, dtype=np.float64)
        if p.shape[0] != len(INTENT_LABELS):
            raise ValueError(
                f"policy width {p.shape[0]} != critic width {len(INTENT_LABELS)}")
        if self.blend <= 1e-12:
            return p
        q, advantage = self.classify(sensor_24, position, probs=p)
        logits = np.log(np.clip(p, 1e-9, 1.0)) + self.blend * advantage
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    def payoff_traces(self, probs: Any, sensor_24: np.ndarray,
                      position: str) -> Dict[str, Any]:
        """Explainability dump: policy, critic q, advantage, corrected."""
        p = np.asarray(probs, dtype=np.float64)
        q, adv = self.classify(sensor_24, position, probs=p)
        corrected = self.correct_policy(p, sensor_24, position)
        return {
            "q": {label: round(float(q[i]), 4)
                  for i, label in enumerate(INTENT_LABELS)},
            "advantage": {label: round(float(adv[i]), 4)
                          for i, label in enumerate(INTENT_LABELS)},
            "corrected": {label: round(float(corrected[i]), 4)
                          for i, label in enumerate(INTENT_LABELS)},
        }


# ─────────────────────────────────────────────────────────────
# GLOBAL SWITCH (mirrors set_perception / set_cognition)
# ─────────────────────────────────────────────────────────────

_reasoner: Optional[ConsequenceEvaluator] = None

# Graduated default posture (validate_consequence_gate.txt, 2026-09-20):
# reasoning is ON by default unless disabled.  v5 = TD-episode critic;
# 0.3 is its gate-winning blend — 0.5 and higher over-correct.
_DEFAULT_CRITIC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "brains_trainer", "critic_v5.json")
_DEFAULT_BLEND = 0.3
_DEFAULT_ENABLED = True


def default_enabled(flag: bool) -> None:
    """Set the unset-env default posture for this process.

    ``False`` lets the collection/gate harnesses pin a clean (reasoning-
    off) corpus regardless of the graduated on-by-default standing;
    ``True`` restores it.  Only consulted when PLOFA_CONSEQUENCE is unset.
    """
    global _DEFAULT_ENABLED
    _DEFAULT_ENABLED = bool(flag)


def default_state() -> bool:
    """Current unset-env default posture (True = reasoning on)."""
    return _DEFAULT_ENABLED


def _load_critic(path: str) -> Any:
    from critic_surrogate import CriticSurrogate, load_any_surrogate
    obj = load_any_surrogate(path)
    # unwrap the FitnessSurrogate-compatible wrapper; the per-intent payoff
    # read we need lives on the underlying ValueCritic.predict()
    if isinstance(obj, CriticSurrogate):
        return obj.critic
    return obj


def set_evaluator(evaluator: Optional[ConsequenceEvaluator]) -> None:
    """Direct evaluator injection (tests / embedding); bypasses file load."""
    global _reasoner
    _reasoner = evaluator


def set_consequence(critic_path: Optional[str] = None, blend: float = 0.3,
                    enabled: bool = True) -> None:
    """Enable/configure the consequence reasoning seam.

    ``critic_path`` may be a V3 value_critic file or a legacy surrogate;
    ``None`` clears the seam (reasoning off).  ``blend`` is how strongly
    consequences correct the policy (0 = off/identity).
    """
    global _reasoner
    if not enabled or critic_path is None:
        _reasoner = None
        return
    critic = _load_critic(critic_path)
    _reasoner = ConsequenceEvaluator(critic, blend=blend)


def clear_consequence() -> None:
    global _reasoner
    _reasoner = None


def get_consequence() -> Optional[ConsequenceEvaluator]:
    """The active evaluator, or None when the seam is off."""
    return _reasoner


def consequence_enabled() -> bool:
    return _reasoner is not None


def apply_reasoning(probs: Any, sensor_24: np.ndarray,
                    position: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """One-call seam for ``_decide_core``.

    Returns ``(probs, None)`` when off (byte-identical); returns
    ``(corrected_probs, trace)`` when on.  ``trace`` carries the full q /
    advantage / corrected dump for the decision journal's ``why()``.
    """
    if _reasoner is None:
        return probs, None
    p = np.asarray(probs, dtype=np.float64)
    corrected = _reasoner.correct_policy(p, sensor_24, position)
    trace = _reasoner.payoff_traces(p, sensor_24, position)
    return corrected, trace


def init_from_env() -> Optional[ConsequenceEvaluator]:
    """Resolve the process-level reasoning posture and configure the seam.

    Called once per process by the match engine runner, on the first
    decision.

    POSTURE (2026-09-20): the consequence reasoning seam GRADUATED the
    gate (validate_consequence_gate.txt) and is now ON BY DEFAULT — a
    match reasons unless told otherwise:
      * env unset, or PLOFA_CONSEQUENCE in 1/true/yes/on  -> enabled.
      * PLOFA_CONSEQUENCE in 0/false/no/off                -> disabled.
      * module ``default_enabled(False)`` (used by the collection and
        gate harnesses) also disables the unset-env default, per-process.

    Enabled posture loads critic_v5.json (the TD-episode critic) at its
    gate-winning blend 0.3 — blend 0.5 demonstrably over-corrects and
    must not be used.  Override the critic via PLOFA_CONSEQUENCE_CRITIC
    and the strength via PLOFA_CONSEQUENCE_BLEND.

    The env can only ENABLE the seam; it never clears an evaluator that
    code (or a test) installed explicitly via ``set_consequence`` /
    ``set_evaluator``.
    """
    raw = os.environ.get("PLOFA_CONSEQUENCE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return _reasoner if _reasoner is not None else None
    on = raw in ("1", "true", "yes", "on") or (raw == "" and _DEFAULT_ENABLED)
    if on:
        critic = (os.environ.get("PLOFA_CONSEQUENCE_CRITIC", "").strip()
                  or _DEFAULT_CRITIC)
        if critic and os.path.exists(critic):
            blend = float(os.environ.get("PLOFA_CONSEQUENCE_BLEND", "")
                          or _DEFAULT_BLEND)
            set_consequence(critic, blend=blend)
            return _reasoner
    return _reasoner if _reasoner is not None else None