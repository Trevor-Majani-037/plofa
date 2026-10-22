"""MEMORY AS AN INPUT — what the player remembers, as numbers the brain reads.

THE GAP THIS FILE CLOSES
------------------------
``cognition/memory.py`` already stores episodic memory (tags, severity,
decay, locations) and ``cognition/experience.py`` already folds it into
temperament.  But the memory reaches the DECISION in exactly one way:
``PlayerMind.temper`` (``cognition/mind.py``) reshapes the sampled
distribution — confidence sharpens or flattens it, fear damps the risk
intents.  **The memory never reaches the brain's REPRESENTATION.**  The
brain reads a 24-d vector that contains nothing about what just happened
to this player, so "the striker who just missed a big chance" and "the
striker who just scored one" are the same input vector.

AND THE MEMORY BARELY EXISTED OUTSIDE PRODUCTION
--------------------------------------------------
``engage_mind_observer()`` is called from ``auto_run_match.py`` and
``season_manager.py`` — the PRODUCTION runner.  The scratch runner
(``run_match.py``) and the real-roster harnesses never call it, so no mind
ever observed an event and every memory block would have been zero.

THE FIX, IN THREE PARTS
-----------------------
1. A mind exists for EVERY player, in EVERY runner.
2. The memory observer is live in EVERY runner.
3. The memory becomes an INPUT BLOCK, so the brain can LEARN from it.

PART 1 IS THE DANGEROUS ONE, and it is worth being explicit about why:
``brain_integration._maybe_cognition`` routes to the mind path whenever a
mind EXISTS, and ``CognitionDecisionBrain.decide`` is a different
pipeline — FOV-gated scene, tempered sampling.  So auto-provisioning an
ENGAGED mind would silently change every decision in the match, which is
the opposite of what a memory feature is supposed to do.

The minds provisioned here are therefore created with ``engaged=False``:
they accumulate memory (the observer does not check ``engaged``) but they
do NOT change the decision path.  ``PlayerMind.observe`` ->
``ExperienceModel.apply_memory`` still runs, so the temperament is
available; it is simply not applied as a sampling bias on the legacy
path.  The production runner's own ENGAGED minds are left alone — this
function returns an existing mind untouched.

A MEASURED UNITS DISCREPANCY, RECORDED RATHER THAN WORKED AROUND
-----------------------------------------------------------------
``MemorySystem.DECAY_PER_SECOND`` is documented as a per-SECOND multiplier
and ``MemoryEvent.age`` is documented as "seconds since the event".  In
the live path neither is true: the observer calls ``mind.observe(event)``
with the default ``dt=1.0``, ONCE PER EVENT.  So in a live match:

    * ``age`` counts the player's INVOLVEMENTS since the memory, not seconds;
    * the decay is 0.999 per INVOLVEMENT, so within one match a memory
      essentially never fades (it would take ~2,300 involvements to fall
      below ``FORGET_BELOW``).

Two consequences, and the features below are built to be honest about both:

  * the raw ``recall()`` totals are effectively UNWEIGHTED within a match,
    so a feature that wants recency must compute it from
    ``memory.events`` directly, with an explicit recency weight — which is
    what ``memory_features`` does;
  * the "seconds" in the docstrings are aspirational.  They are NOT
    corrected here, because changing the decay constant changes every
    mind's behaviour and is a separate, deliberate calibration.  It is
    recorded so nobody reads ``age`` as a clock.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


#: How many feature slots the memory block occupies.
MEMORY_FEATURE_D = 8

#: The names, in order.  A schema that names its features is a schema that
#: can be checked; the role tail works that way and so does this.
MEMORY_FEATURE_NAMES: List[str] = [
    "duel_win_ratio",      # 0  recency-weighted duels won vs lost
    "chance_conversion",   # 1  recency-weighted big chances scored vs missed
    "foul_load",           # 2  how much trouble this player is in
    "emotional_load",      # 3  total surviving weight of everything
    "recency",             # 4  how recently the last significant event was
    "memory_count",        # 5  how much is remembered at all
    "confidence_bias",     # 6  the compressed confidence axis
    "risk_bias",           # 7  the compressed risk axis
]

#: The recency half-life, in INVOLVEMENTS (see the units note above).
#: 40 involvements is roughly a third of a busy player's match.
RECENCY_SCALE = 40.0

#: The count at which "how much is remembered" saturates.
COUNT_SCALE = 12.0

#: The tags each feature reads.
_TAG_DUEL_WON = "clean_tackle_win"
_TAG_DUEL_LOST = "lost_duel"
_TAG_CHANCE_SCORED = "big_chance_scored"
_TAG_CHANCE_MISSED = "big_chance_missed"
_TAG_FOUL = "foul_in_box"


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def _recency_weight(age: float) -> float:
    """How much an event this old still counts.

    ``age`` is in involvements, not seconds — see the module docstring.
    A linear decay is used rather than an exponential because the ages
    inside one match are small integers and the ordering is what matters.
    """
    a = float(age)
    if a < 0.0:
        a = 0.0
    return 1.0 / (1.0 + a / RECENCY_SCALE)


def ensure_mind(player_name: str) -> Any:
    """Return the player's mind, creating an UNENGAGED one if there is none.

    An unengaged mind is INERT for the decision path (``CognitionDecisionBrain``
    falls through to the vanilla core when ``engaged`` is False) but LIVE for
    memory: the observer writes to it and the temperament is updated.
    """
    from cognition.mind import PlayerMind, get_mind, register_mind

    mind = get_mind(player_name)
    if mind is not None:
        return mind
    mind = PlayerMind()
    mind.engaged = False
    register_mind(player_name, mind)
    return mind


def provision_minds(names: List[str]) -> int:
    """Make sure every named player has a mind.  Returns how many were made."""
    made = 0
    for n in names:
        if not n:
            continue
        from cognition.mind import get_mind

        if get_mind(n) is None:
            ensure_mind(n)
            made += 1
    return made


def memory_features(player_name: str, mind: Any = None) -> np.ndarray:
    """The player's memory, as ``MEMORY_FEATURE_D`` floats in [0, 1].

    Returns zeros for a player with no mind and no memory — an absent
    memory is a real value (the player has done nothing yet), not an error.
    """
    out = np.zeros(MEMORY_FEATURE_D, dtype=np.float64)
    if mind is None:
        mind = _lookup_mind(player_name)
    if mind is None:
        return out

    events = list(getattr(getattr(mind, "memory", None), "events", []) or [])
    if not events:
        # No events, but the temperament may still carry a bias (a mind
        # hydrated from a previous match).  Report the bias and leave the
        # episodic features at zero.
        _fill_bias(out, mind)
        return out

    duel_won = duel_lost = 0.0
    chance_ok = chance_bad = 0.0
    foul_load = 0.0
    emotional = 0.0
    newest_age = None
    for ev in events:
        w = _recency_weight(getattr(ev, "age", 0.0))
        sev = float(getattr(ev, "severity", 1.0))
        tag = getattr(ev, "tag", "")
        age = float(getattr(ev, "age", 0.0))
        if newest_age is None or age < newest_age:
            newest_age = age
        emotional += w * sev
        if tag == _TAG_DUEL_WON:
            duel_won += w
        elif tag == _TAG_DUEL_LOST:
            duel_lost += w
        elif tag == _TAG_CHANCE_SCORED:
            chance_ok += w
        elif tag == _TAG_CHANCE_MISSED:
            chance_bad += w
        elif tag == _TAG_FOUL:
            foul_load += w * sev

    total_duels = duel_won + duel_lost
    out[0] = _clamp01(0.5 + 0.5 * ((duel_won - duel_lost) / (total_duels + 1.0)))
    total_chances = chance_ok + chance_bad
    out[1] = _clamp01(0.5 + 0.5 * ((chance_ok - chance_bad) / (total_chances + 1.0)))
    out[2] = _clamp01(foul_load / (1.0 + foul_load))
    out[3] = _clamp01(emotional / (1.0 + emotional))
    out[4] = _clamp01(1.0 - (newest_age / RECENCY_SCALE)) if newest_age is not None else 0.0
    out[5] = _clamp01(len(events) / COUNT_SCALE)
    _fill_bias(out, mind)
    return out


def _fill_bias(out: np.ndarray, mind: Any) -> None:
    """Write the two temperament axes into slots 6 and 7."""
    bias = getattr(getattr(mind, "experience", None), "bias", None) or {}
    conf = float(bias.get("confidence", 1.0))
    risk = float(bias.get("risk_in_box", 1.0))
    # confidence lives in [0.5, ~1.5]; map the neutral 1.0 to the middle.
    out[6] = _clamp01((conf - 0.5) / 1.0)
    # risk_in_box lives in [0.4, 1.0]; a fearless player is at the top.
    out[7] = _clamp01((risk - 0.4) / 0.6)


def _lookup_mind(player_name: str) -> Optional[Any]:
    try:
        from cognition.mind import get_mind

        return get_mind(player_name)
    except Exception:
        return None
