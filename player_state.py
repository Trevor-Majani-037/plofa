"""PlayerState — the state a player carries BETWEEN touches.

WHY THIS FILE EXISTS
--------------------
The evolved brain is a PURE FUNCTION of its sensor vector: three matmuls,
two ReLUs, a softmax (``football_brain.py`` ``forward_logits``).  No
recurrence, no memory, no plan.  A centre-back who has just been
humiliated in a duel and one who has just won a tackle are BYTE-IDENTICAL
to it — and so are the first and the fiftieth touch of the match.

That cannot be fixed inside the brain file, for a structural reason worth
stating plainly, because it decides where this file has to live:

    ``FootballBrain`` instances are shared PER POSITION.

``brain_integration._resolve_brain`` looks a brain up by player NAME and
falls back to a per-POSITION cache (``_pos_brain_cache``) of the file in
``brains/``.  Every centre-back in a match therefore shares ONE
``FootballBrain`` object.  A recurrent state stored on the brain would be
shared by eleven players at once, and it would hold whichever player's
touch happened to be last.  **The state has to live with the PLAYER.**

So this module owns it, and ``PositionEngine`` holds the containers — it
already owns one per-player dict with exactly the right lifetime (created
by ``initialize_team``, dropped by ``remove_player``).

THE TWO PIECES
--------------
``PlayerNeuralState``
    Per player.  The recurrent core's hidden state (the GRU cell), the
    touch counter, and the last intent the player actually executed.

``TeamPlan``
    Per team.  The possession-level OBJECTIVE the eleven are currently
    executing — "get the ball out", "work it wide", "attack the box".
    This is the planning state, and it is deliberately TEAM state and not
    player state, because a plan belongs to the team and the players
    execute it.

THE PLAN VOCABULARY IS THE ENGINE'S OWN
----------------------------------------
``TeamPlan.plan_id`` indexes ``possession_phases.PossessionPhase``:
``REGROUP_BUILD_UP -> MIDFIELD_CIRCULATION -> WING_ISOLATION ->
BOX_PENETRATION``.  That phase machine already exists, already carries
football in it, and already names the receiver on 296 of 1474 passes.
Reusing it means the plan head's supervision signal is the engine's own
phase — not a new hand-written table — and it means a learned plan can be
checked against the phase the engine actually played.

THE SWITCH IS STICKY ON PURPOSE
-------------------------------
A plan that flickered every touch would be noise dressed as a plan.  The
adoption rule is therefore a HYSTERESIS: a proposed plan must be proposed
``min_hits`` times in a row, at or above ``threshold`` confidence, before
the team adopts it.  That is the same discipline the rest-defence
invariant uses — a preference yields, a plan has to be earned — and it
means one stray confident touch cannot rewrite the team's objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


#: The number of possession phases the plan head can name.  Derived from
#: the engine's own enum so the two cannot drift.
def _phase_count() -> int:
    try:
        from possession_phases import PossessionPhase

        return len(list(PossessionPhase))
    except Exception:                      # pragma: no cover - import guard
        return 4


PLAN_COUNT: int = _phase_count()


@dataclass
class PlayerNeuralState:
    """One player's recurrent state — the part of the brain that remembers.

    ``hidden`` is the recurrent core's state vector.  It is ``None`` until
    the first touch, and it is sized by the brain that owns it (a v1/v2/v3
    brain has no recurrent core and therefore never sets it).
    """

    player_name: str
    hidden: Optional[np.ndarray] = None
    touches: int = 0
    last_intent: int = -1
    last_touch_minute: float = -1.0
    last_touch_second: float = -1.0

    def ensure_hidden(self, size: int) -> np.ndarray:
        """Return the hidden state, sized to ``size``.

        A size change means a different brain now owns this player (a
        substitution, or a re-registration), so the old state is dropped
        rather than silently reinterpreted — a GRU's hidden vector is
        meaningless to a net it was not trained for.
        """
        if self.hidden is None or self.hidden.shape[0] != int(size):
            self.hidden = np.zeros(int(size), dtype=np.float64)
        return self.hidden

    def record_touch(self, intent_index: int, minute: float,
                     second: float = -1.0) -> None:
        """Note that this player has just had the ball."""
        self.touches += 1
        self.last_intent = int(intent_index)
        self.last_touch_minute = float(minute)
        self.last_touch_second = float(second)

    def reset(self) -> None:
        """Forget everything — used at a season's start, not mid-match."""
        self.hidden = None
        self.touches = 0
        self.last_intent = -1
        self.last_touch_minute = -1.0
        self.last_touch_second = -1.0


@dataclass
class TeamPlan:
    """The possession-level objective one team is executing.

    The plan is TEAM state because a plan belongs to the team: the eleven
    are executing one objective, and the individuals' decisions are the
    execution.  It resets when the possession changes.
    """

    team: str
    plan_id: int = -1                 # -1 = no plan adopted yet
    confidence: float = 0.0
    possession_id: int = 0
    switches: int = 0
    _candidate: int = -1
    _candidate_hits: int = 0
    _candidate_conf: float = 0.0

    # ── POSSESSION LIFECYCLE ─────────────────────────────────────────
    def reset_possession(self, possession_id: int) -> bool:
        """Drop the plan when the possession changes.

        Returns True if the plan was dropped.  The plan does NOT survive a
        turnover: a team that loses the ball mid-attack is not still
        "attacking the box", and carrying the old plan across the change
        of possession is how a stale objective starts steering the wrong
        team.
        """
        if int(possession_id) == int(self.possession_id):
            return False
        self.possession_id = int(possession_id)
        self.plan_id = -1
        self.confidence = 0.0
        self._candidate = -1
        self._candidate_hits = 0
        self._candidate_conf = 0.0
        return True

    # ── ADOPTION (the hysteresis) ───────────────────────────────────
    def propose(self, plan_id: int, confidence: float,
                threshold: float = 0.6, min_hits: int = 2) -> bool:
        """Offer a plan.  Returns True if the team ADOPTED it.

        The plan only changes when the SAME plan is proposed ``min_hits``
        times in a row at or above ``threshold`` confidence.  A single
        confident touch is not a plan; a repeated one is.
        """
        plan_id = int(plan_id)
        confidence = float(confidence)
        if plan_id < 0:
            return False
        if plan_id == self.plan_id:
            # Already executing it — just note the confidence.
            self.confidence = max(self.confidence, confidence)
            self._candidate = -1
            self._candidate_hits = 0
            self._candidate_conf = 0.0
            return False
        if plan_id == self._candidate:
            self._candidate_hits += 1
            self._candidate_conf = max(self._candidate_conf, confidence)
        else:
            self._candidate = plan_id
            self._candidate_hits = 1
            self._candidate_conf = confidence
        if self._candidate_hits >= int(min_hits) and \
                self._candidate_conf >= float(threshold):
            self.plan_id = plan_id
            self.confidence = self._candidate_conf
            self.switches += 1
            self._candidate = -1
            self._candidate_hits = 0
            self._candidate_conf = 0.0
            return True
        return False

    # ── READ ────────────────────────────────────────────────────────
    def one_hot(self, count: int = PLAN_COUNT) -> np.ndarray:
        """The plan as a one-hot vector (all zeros when there is no plan)."""
        out = np.zeros(int(count), dtype=np.float64)
        if 0 <= self.plan_id < int(count):
            out[self.plan_id] = 1.0
        return out

    def summary(self) -> np.ndarray:
        """The plan as [one_hot..., confidence, has_plan]."""
        out = np.zeros(PLAN_COUNT + 2, dtype=np.float64)
        out[:PLAN_COUNT] = self.one_hot()
        out[PLAN_COUNT] = float(np.clip(self.confidence, 0.0, 1.0))
        out[PLAN_COUNT + 1] = 1.0 if self.plan_id >= 0 else 0.0
        return out


def player_team(player: Any) -> str:
    """Which team this player is on.

    TWO TRAPS, AND BOTH ARE SILENT.

    1. ``PlayerProfile`` carries the team as ``team_name`` — there is NO
       ``team`` attribute.  Reading ``player.team`` returns the default
       ``""``, so the team's PLAN (and the corpus's team key) is
       silently lost and every player looks team-less.

    2. The fallback must NOT be the position engine.  The scratch
       fixture gives BOTH teams the same generic names (GK, CB1, CM1,
       ...), and ``PositionEngine.states`` is keyed by NAME — so the
       away team's state has overwritten the home team's, and a lookup by
       name returns the WRONG team for half the squad.  (Measured: 70 of
       79 sequences came back as the away team, because the away team was
       initialised last.)

    So the team is read from the PLAYER OBJECT, which is unambiguous:
    the two centre-backs called "CB1" are two different objects with two
    different ``team_name`` values.
    """
    for attr in ("team_name", "team"):
        value = getattr(player, attr, "") or ""
        if value:
            return str(value)
    return ""


@dataclass
class NeuralStateStore:
    """The two containers, in one place, with the lifecycle they need.

    The PositionEngine owns one of these per match.  It is deliberately a
    separate object from the spatial state: a player's position and a
    player's BRAIN STATE are different things, and merging them would make
    every spatial reset (a minute boundary, a substitution) silently wipe
    the player's memory.
    """

    players: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)

    def player(self, name: str) -> PlayerNeuralState:
        """The player's state, created on first use."""
        st = self.players.get(name)
        if st is None:
            st = PlayerNeuralState(player_name=name)
            self.players[name] = st
        return st

    def team(self, team: str) -> TeamPlan:
        """The team's plan, created on first use."""
        tp = self.teams.get(team)
        if tp is None:
            tp = TeamPlan(team=team)
            self.teams[team] = tp
        return tp

    def remove_player(self, name: str) -> None:
        """A sent-off or substituted player loses their state with it."""
        self.players.pop(name, None)

    def reset(self) -> None:
        """Forget every player and every plan (a new match, or a new season)."""
        self.players.clear()
        self.teams.clear()

    def names(self) -> List[str]:
        return list(self.players.keys())
