"""Collect (sensor, intent, outcome) training data from REAL matches.

The goal: build a fitness surrogate that predicts *expected success* of
an on-ball decision from the vision sensor state + chosen intent.  This
surrogate is learned from real MatchEngine outcomes, so evolution can
then optimize brains against it (fast) instead of against 20-second real
matches (slow).

Why instrument `decide` instead of reading the timeline?
---------------------------------------------------------
The event timeline does NOT contain the full spatial state (teammate and
defender positions relative to the ball) at the moment of each decision,
which the 24-float sensor vector needs.  But DecisionBrain.decide()
receives exactly that geometry as arguments.  So we patch the heuristic
brain to snapshot (player, minute, sensor_vector, intent) whenever it
decides, then JOIN each snapshot to the action event it caused.

How the join is made is NOT a detail — it is the thing that decides whether
the surrogate is football or noise.  The first implementation joined each
snapshot to "the first execution event by that player at or after this
minute", which is wrong, because a decision can emit MORE THAN ONE execution
event (a failed pass emits PASS *and* TURNOVER / DISPOSSESSED) and a
one-event-per-record cursor therefore mis-consumes the extra event against
the NEXT decision of that player.  Measured over 3 real matches, that join's
self-consistency was 12.5% for SHOOT, 0% for CROSS, 10% for THROUGH_BALL and
64.8% for SAFE_PASS — so a shot decision was charged for somebody else's
carry, and the surrogate fitted from it ranked SHOOT at -0.0133, which would
teach the GA never to shoot.

The join is now CAUSAL: `ChainDispatcher.possession` / `.attack` /
`.set_piece` are the units of work, the chain that makes the on-ball
decision also produces the event, and it emits the ACTION before its
consequences.  See `_correlate_outcomes_causal`.

Outcome definition (THREAT RETURN, roughly -0.6..+1.0, SIGNED):
    The reward is the change in threat the action caused, using the xT grid
    `momentum_model` derives from this project's own shot model:
    • ball movers (pass / carry / dribble / cross / through ball):
      completed -> PHI(end) - PHI(start), so a BACKWARD ball is negative;
      failed -> -0.6 * the OPPONENT's threat where the ball was lost.
    • SHOT: the absolute zone value (= the xG from that spot), half of it if
      off target.  A shot is the threat, so it is not scored as a difference.
    • GOAL: 1.0, deliberately above any achievable xG.
    • TURNOVER / MISCONTROL / DISPOSSESSED / INTERCEPTION: charged the same way.
    • default (no event found): 0.0 -- NEUTRAL, not 0.5, because with a signed
      reward a positive default would outrank every intent actually observed.

    The previous reward scored ENGINE SUCCESS (completed pass 2.0 + advance,
    retained carry 1.0, turnover 0.0).  That asked "which intent does this
    engine complete?" and the answer is always the square ball, which is how
    T2G160 reached the best fitness the project has ever recorded (0.6523,
    all eleven positions saturated) and pass maps indistinguishable from the
    heuristic.  See `threat_reward.py` for the full post-mortem.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from decision_brain import DecisionBrain, PlayerIntent
from decision_brain import CARRY_LIKE_INTENTS
from brain_sensors import extract_sensors
from brain_integration import (
    NeuralDecisionBrain, _INTENT_BY_INDEX, _score_diff_of,
)
from football_brain import INPUT_SIZE
from perception import get_perception_config, perceive
from role_features import (ROLE_FAMILY, ROLE_INTENTS, ROLE_ONLY_INTENTS,
                            _role_family)

#: Prefix used for the ROLE FAMILY rows inside the fit's grouping dict.  It
#: keeps the family tables OUT of `self.table` (which is keyed by position
#: and read by other tools) and in their own `family_table`.
_FAMILY_KEY_PREFIX = "~family:"

#: The value the surrogate returns for a ROLE-SPECIFIC action the role
#: cannot perform (a striker crossing, a keeper shooting).  It is far below
#: any real reward — the threat return's range is about [-0.6, +1.0] — so the
#: genetic algorithm treats the action as unavailable rather than merely
#: unattractive.
FORBIDDEN_VALUE = -1.0

#: Floor for the per-position reward scale (see `FitnessSurrogate.scales`).
#: A position whose fitted cells barely vary still has to be normalised to an
#: O(1) spread, or dividing by a near-zero std would blow a tiny difference up
#: into the only thing the GA can see — the same defect as the fixed guards,
#: aimed the other way.
_SCALE_FLOOR = 0.002

#: A position needs at least this many filled cells for its spread to mean
#: anything; below it the scale is 1.0 (no normalisation rather than a scale
#: computed from one or two numbers).
_SCALE_MIN_CELLS = 3


# ─────────────────────────────────────────────────────────────
# GLOBAL COLLECTOR (filled by the patched decide())
# ─────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    player: str
    position: str
    minute: float
    order: int              # global call order
    sensors: np.ndarray     # 24-float
    intent: PlayerIntent
    # ── THE TEAM, and it is NOT OPTIONAL ────────────────────────
    # The scratch collection fixture gives BOTH teams the same generic
    # names (GK, CB1, CM1, ...), so "CB1" is two different players.
    # A sequence keyed on the NAME ALONE would interleave the home
    # centre-back's touches with the away centre-back's, and the
    # recurrent core would be trained on a state that jumps halfway
    # across the pitch between two steps.  The team disambiguates them.
    team: str = ""
    # ── THE SITUATIONAL BLOCKS (2026-10-22) ─────────────────────
    # The four-capability brain reads four things the flat 24-d vector
    # cannot carry: the ENTITY SET it attends to, the player's own ROLE
    # block, what he REMEMBERS of his match so far, and the PLAN the team
    # is executing.  They are recorded here so the situational corpus can
    # be built WITHOUT re-running the matches.
    tokens: Optional[np.ndarray] = None       # (N_MAX, TOKEN_D)
    token_mask: Optional[np.ndarray] = None   # (N_MAX,)
    role: Optional[np.ndarray] = None         # (ROLE_D,)
    memory: Optional[np.ndarray] = None       # (MEMORY_D,)
    plan: Optional[np.ndarray] = None         # (PLAN_D,)
    phase: int = -1                           # the engine's possession phase


_COLLECTOR: List[DecisionRecord] = []
_CALL_ORDER = 0
_SAVED_DECIDE = None

#: The decision records ACCUMULATED ACROSS MATCHES, for the situational
#: corpus.  ``_COLLECTOR`` is cleared after every match (so the pairing
#: cursor does not walk into the next match); the situational corpus needs
#: the opposite, because a sequence can span a match boundary in the file
#: and the corpus is written once at the end.
_SITUATIONAL_RECORDS: List[DecisionRecord] = []

#: ε-exploration: the probability that a decision is REPLACED by a random
#: action from the player's role vocabulary.  0.0 = the policy's own choice
#: (the old behaviour).  See `_explore_intent`.
_EXPLORE_EPS = 0.0
_EXPLORE_RNG: Optional[random.Random] = None
_EXPLORED = 0

#: SITUATION-AWARE ε-exploration.  **OFF by default** — this is a proposal,
#: measured and written up in AGENTS.md (*THE CEILING*), not a decision.
#:
#: The uniform exploration samples an action the ROLE can perform, but not
#: one the SITUATION can be judged on, and the corpus does not contain the
#: decisions that matter.  Measured on the v5 corpus: the striker's clearest
#: chance (`P1FT1OH0S1C1G1` — final third, goal-close, central, with space)
#: held ~10 striker decisions across six matches and **not one shot**, so its
#: top value was PROTECT_POSSESSION +0.1068 — the largest positive value in
#: the striker's whole table.  The objective could not say "shoot" because
#: it had never been told.
#:
#: Two changes, both about the corpus's COVERAGE:
#:
#:   1. the explored action is WEIGHTED by the phase — the actions a
#:      situation can be judged on are sampled more often there;
#:   2. the exploration RATE is higher in the final third, because that is
#:      where the rare actions (shot, cross, through ball) are the only ones
#:      the situation can judge.
#:
#: The vocabulary stays the ROLE's: the striker is still never asked to
#: cross, and the keeper is still never asked to shoot.
_EXPLORE_SITUATIONAL = False

#: The exploration RATE in the final third, when the situational exploration
#: is on.  The rare actions live there, and the situations that can judge
#: them are rare, so the rate has to be high or the cell never fills.
_EXPLORE_EPS_FINAL_THIRD = 0.75

#: The actions a SITUATION can be judged on, by phase.  This is the football
#: truth about what a situation affords — the same kind of object as
#: `role_features.ROLE_INTENTS` ("what this role can do"), and NOT a table
#: of what happened to happen.  The weight is how much more often the
#: action is sampled in that phase.
#:
#: `goal_close` is its own phase because it is the only one where a shot is
#: GEOMETRICALLY possible: the engine's shot gate refuses the strike
#: otherwise, and a refused strike is recorded as a zero-length marker
#: carry, which is worth exactly 0.0 — so sampling shots where they cannot
#: be taken would teach the table that shooting is worth nothing.
_SITUATION_ACTIONS = {
    # close to goal: a shot is on, and so is a cross
    "goal_close": {"SHOOT": 8.0, "CROSS": 8.0, "THROUGH_BALL": 3.0,
                   "PROGRESSIVE_PASS": 2.0, "SWITCH": 1.5},
    # a final-third situation can judge a cross or a ball in behind; a shot
    # from here is usually refused by the geometry
    "final_third": {"CROSS": 5.0, "THROUGH_BALL": 4.0,
                    "PROGRESSIVE_PASS": 3.0, "SWITCH": 2.0, "SHOOT": 1.0,
                    "CARRY": 1.5},
    # an open situation can judge a switch, a ball in behind or a carry
    "open": {"SWITCH": 3.0, "THROUGH_BALL": 3.0, "PROGRESSIVE_PASS": 2.0,
             "CARRY": 2.0, "SAFE_PASS": 1.0},
    # in your own half the situation can judge a safe pass, a recycle or a
    # switch of play — and nothing else
    "own_half": {"SAFE_PASS": 3.0, "RECYCLE": 3.0, "PROTECT_POSSESSION": 2.0,
                 "SWITCH": 2.0, "PROGRESSIVE_PASS": 1.5},
}

#: The phases in which the exploration RATE is raised: the attacking ones,
#: where the rare actions are the only ones the situation can judge.
_EXPLORE_ATTACKING_PHASES = ("goal_close", "final_third")


def sc_explored() -> int:
    """How many decisions the ε-exploration replaced (for the report)."""
    return _EXPLORED


def _situation_phase(sensors) -> str:
    """Which phase the sensors describe.

    The phases are the surrogate's own bucket features, so the exploration
    and the surrogate agree on what a "situation" is:

        goal_close   final third AND close to goal (a shot is geometrically
                     possible here, and only here)
        final_third  in the attacking third
        own_half     in your own half
        open         neither
    """
    if sensors is None:
        return "open"
    try:
        final_third = float(sensors[12]) > 0.5
        own_half = float(sensors[13]) > 0.5
        goal_close = float(sensors[14]) < 0.5
    except (IndexError, TypeError, ValueError):
        return "open"
    if final_third and goal_close:
        return "goal_close"
    if final_third:
        return "final_third"
    if own_half:
        return "own_half"
    return "open"


def _explore_intent(player, intent, sensors=None):
    """Occasionally replace the decision with a random ACTION the role can
    actually perform.

    WHY THIS EXISTS.  The corpus is generated by a policy that never chooses
    the rare actions, so the corpus never CONTAINS them: the CM family
    performed no through-ball in 4,011 real decisions, and the striker
    performed exactly one shot.  A surrogate fitted on that corpus has no
    value for the actions that matter, and the genetic algorithm then learns
    the wrong thing (T2's documented collapse: "ST never shoots, CM abandons
    through-balls").

    The vocabulary is the ROLE's, from `role_features.ROLE_INTENTS` — the
    football truth about what the role can do, not a list of what happened
    to happen.  So the striker is never asked to cross, and the midfielder
    IS asked to play a through ball.

    The decision is replaced, not the outcome: the engine still executes it
    and the reward is still measured on what really happened.

    WITH `sensors` (and `_EXPLORE_SITUATIONAL` on), the exploration also
    looks at the SITUATION: the rate is higher in the final third and the
    action is drawn with the phase's weights, so the corpus ends up holding
    the decisions that matter instead of only the ones that happened.
    """
    global _EXPLORED
    if _EXPLORE_EPS <= 0.0:
        return intent
    fam = _role_family(getattr(player, "position", "") or "")
    vocab = ROLE_INTENTS.get(fam) if fam else None
    if not vocab:
        return intent
    phase = "open"
    eps = _EXPLORE_EPS
    if _EXPLORE_SITUATIONAL and sensors is not None:
        phase = _situation_phase(sensors)
        if phase in _EXPLORE_ATTACKING_PHASES:
            eps = max(eps, _EXPLORE_EPS_FINAL_THIRD)
    if _EXPLORE_RNG.random() >= eps:
        return intent
    options = sorted(vocab)
    if _EXPLORE_SITUATIONAL and sensors is not None:
        weights = _SITUATION_ACTIONS.get(phase, {})
        options = [o for o in options if weights.get(o, 0.0) > 0.0] or options
        w = [float(weights.get(o, 1.0)) for o in options]
        choice = _EXPLORE_RNG.choices(options, weights=w, k=1)[0]
    else:
        choice = _EXPLORE_RNG.choice(options)
    _EXPLORED += 1
    try:
        return PlayerIntent(choice)
    except ValueError:
        return intent


def _snapshot_decide(player, *args, **kwargs):
    """Patch that records (sensor, intent) then calls the real brain.

    Sensors are the PERCEIVED vector — the state the brain actually saw
    (perception layer applied, which is the production default).  The
    perception noise draw is deterministic per snapshot
    (perception.py ``_ident``), so recomputing ``perceive`` here with the
    same global config reproduces the exact vector ``_decide_core`` fed the
    network.  When perception is disabled (identity regime) ``perceive``
    delegates to ``extract_sensors`` and this is byte-identical to the
    old collector.
    """
    global _CALL_ORDER
    d = _SAVED_DECIDE(player, *args, **kwargs)
    # reconstruct sensors from the real args (same positional layout as
    # DecisionBrain.decide and NeuralDecisionBrain.decide)
    x = args[0]
    y = args[1]
    teammates = args[2]
    defenders = args[3]
    position_engine = args[4]
    # positional args after the player are: x, y, teammates, defenders,
    # position_engine, team_profile, under_pressure, attacks_right,
    # game_state, minute
    under_pressure = args[7] if len(args) > 7 else kwargs.get("under_pressure", False)
    attacks_right = args[8] if len(args) > 8 else kwargs.get("attacks_right", True)
    game_state = args[9] if len(args) > 9 else kwargs.get("game_state", None)
    minute = args[10] if len(args) > 10 else kwargs.get("minute", 45.0)
    team_possession = True  # on-ball carrier is by definition in possession

    # The sensors are computed BEFORE the exploration, because a
    # SITUATION-AWARE exploration needs to know where the player is: the
    # sensors are the same vector either way (the perception noise is
    # deterministic per snapshot, and `perceive` draws from a SEEDED
    # generator, so moving this line does not move the global RNG stream).
    #
    # THE SCENE IS REQUESTED TOO, because the situational blocks must be
    # built from the PERCEIVED scene — the same one the live path uses.
    # Reading the role block and the entity tokens from the FULL scene
    # would give the net more than the player can see, and the corpus
    # would then be in a DIFFERENT FRAME from the one it is played in:
    # a net trained on a leaked frame behaves differently on the pitch
    # than in training, and the difference is invisible.
    sensors, scene = perceive(
        player, x, y, teammates, defenders, position_engine,
        under_pressure, attacks_right, game_state, minute,
        team_possession=team_possession, score_diff=_score_diff_of(game_state),
        config=get_perception_config(), return_scene=True,
    )
    # ε-exploration: replace the decision with a random ACTION the role can
    # perform, so the corpus eventually covers the actions the policy never
    # chooses.  The replacement is returned to the engine, so the explored
    # action is REALLY executed and the reward is measured on what happened.
    explored = _explore_intent(player, d.intent, sensors=sensors)
    if explored is not d.intent:
        import dataclasses as _dc
        d = _dc.replace(
            d, intent=explored,
            action="CARRY" if explored in CARRY_LIKE_INTENTS else "PASS",
            target=None,          # the chain picks the receiver itself
            confidence=0.25,      # an exploratory pick is not a confident one
            reason=f"epsilon-exploration: {explored.value}",
            is_error=True,        # it IS an error against the policy's own choice
        )
    _COLLECTOR.append(DecisionRecord(
        player=getattr(player, "name", ""),
        position=getattr(player, "position", ""),
        minute=minute,
        order=_CALL_ORDER,
        sensors=sensors,
        intent=d.intent,
        team=_team_of(player, position_engine),
        **_situational_blocks(player, x, y,
                             scene.get("teammates", teammates),
                             scene.get("defenders", defenders),
                             scene.get("position_engine", position_engine),
                             attacks_right),
    ))
    _CALL_ORDER += 1
    return d


def _team_of(player: Any, position_engine: Any) -> str:
    """Which team this player is on.

    Read from the PLAYER OBJECT, which is the only unambiguous source:
    the scratch fixture gives both teams the same generic names, and the
    position engine's states are keyed by NAME, so a lookup there
    returns whichever team was initialised last.  (Measured: 70 of 79
    sequences came back as the away team for exactly that reason.)
    """
    from player_state import player_team

    return player_team(player)


def _situational_blocks(player, x, y, teammates, defenders,
                        position_engine, attacks_right) -> Dict[str, Any]:
    """The four blocks the situational brain reads, for the CORPUS.

    THE PLAN IS TEACHER-FORCED HERE, and it has to be said out loud: the
    plan INPUT recorded in the corpus is the engine's OWN possession phase
    (the ground truth), not a plan the brain proposed.  The shipped brains
    have no plan head, so during collection there IS no proposed plan.

    That makes the plan input at TRAINING time the true phase, while at
    INFERENCE it is the head's own adopted proposal — the standard
    teacher-forcing exposure gap.  It is measured rather than hidden: the
    corpus keeps the ground-truth phase, so the agreement between the
    head's proposal and the phase the engine is really in can be
    reported, and that agreement IS the number that says whether the
    learned plan is real.

    Every block is best-effort: a collector that cannot build one of them
    records None rather than raising, because a collection run is
    expensive and a missing block is a hole, not a failure.
    """
    out: Dict[str, Any] = {}
    try:
        from situation_tokens import entity_tokens

        tokens, mask = entity_tokens(player, x, y, teammates, defenders,
                                      position_engine, attacks_right)
        out["tokens"] = tokens
        out["token_mask"] = mask
    except Exception:
        pass
    try:
        from situational_context import role_tail

        out["role"] = role_tail(player, x, y, teammates, defenders,
                                position_engine, attacks_right)
    except Exception:
        pass
    try:
        from memory_features import memory_features as _mf

        out["memory"] = _mf(getattr(player, "name", "") or "")
    except Exception:
        pass
    try:
        from possession_phases import possession_phase_for

        phase = possession_phase_for(x, y, attacks_right)
        pid = _phase_index(phase)
        out["phase"] = int(pid)
        out["plan"] = _teacher_forced_plan(pid)
    except Exception:
        pass
    return out


def _phase_index(phase: Any) -> int:
    """The phase's index in the enum — the plan head's class order."""
    from possession_phases import PossessionPhase

    try:
        return list(PossessionPhase).index(phase)
    except ValueError:
        return -1


def _teacher_forced_plan(phase_id: int) -> np.ndarray:
    """The plan block for a KNOWN phase: one-hot + confidence + a flag."""
    from player_state import PLAN_COUNT

    out = np.zeros(PLAN_COUNT + 2, dtype=np.float64)
    if 0 <= phase_id < PLAN_COUNT:
        out[phase_id] = 1.0
    out[PLAN_COUNT] = 1.0
    out[PLAN_COUNT + 1] = 1.0 if phase_id >= 0 else 0.0
    return out


def _install_collector():
    """Routes the live decision entry point (NeuralDecisionBrain.decide,
    which event_chain.py calls for every touch) through the snapshot
    wrapper.  The real decision is still made by whatever is bound there
    (trained neural brain, or pin the heuristic first with
    ``_pin_on_ball_policy`` for a heuristic baseline).

    It ALSO captures the chain calls, which is what makes the causal join
    possible: the chain that decides is the chain that acts.
    """
    global _SAVED_DECIDE
    if _SAVED_DECIDE is None:
        _SAVED_DECIDE = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_snapshot_decide)
    _install_dispatcher_capture()


def _restore_collector():
    global _SAVED_DECIDE
    if _SAVED_DECIDE is not None:
        NeuralDecisionBrain.decide = _SAVED_DECIDE
        _SAVED_DECIDE = None
    _restore_dispatcher_capture()


# ─────────────────────────────────────────────────────────────
# CAUSAL PAIRING — the chain call is the unit of causality
# ─────────────────────────────────────────────────────────────
#
# `_correlate_outcomes` (below) joins a decision to "the first execution
# event by that player at or after this minute".  Measured over 3 real
# matches that join is wrong, and badly wrong for exactly the intents the
# new reward was built to favour:
#
#     SHOOT         1/8   (12.5% self-consistent)
#     CROSS         0/18  (0%)
#     THROUGH_BALL  1/10  (10%)
#     DRIBBLE       3/212 (1.4%)
#     SAFE_PASS     408/630 (64.8%)
#
# The mechanism: a decision can emit MORE THAN ONE execution event (a failed
# pass emits PASS *and* TURNOVER / DISPOSSESSED / MISCONTROL), the cursor
# consumes one event per record, and every extra event is therefore
# mis-consumed by the NEXT decision of that player.  The alignment then
# slides for the rest of the match.
#
# The fix is causal rather than clever.  `ChainDispatcher.possession` /
# `.attack` / `.set_piece` are the units of work: the chain that makes the
# on-ball decision also produces the event, and it emits the ACTION before
# its consequences.  So a decision joins to the first ACTION event of its
# own chain, by the same player, skipping the events that are not actions:
# a BALL_RECEIPT is the pass ARRIVING at the new carrier (emitted by the
# carrier, before the action he then takes) and a PRESS is not an on-ball
# action at all.
#
# Measured on the SAME 3 matches, same decisions, same rewards:
#
#     SAFE_PASS       BALL_RECEIPT:219 -> PASS:480
#     SWITCH          -> SWITCH_OF_PLAY:204
#     CARRY           -> CARRY:183
#     DRIBBLE         -> CARRY:198
#     PROGRESSIVE_PASS -> PASS:119, PROGRESSIVE_PASS:69
#     SHOOT           mean -0.0133 -> +0.0372  (highest of all intents)
#     drop rate       0.0%  (every decision finds its own chain's action)
#
# A SHOOT intent legitimately causes a CARRY into shooting position, not a
# shot: the shot is the deterministic AttackingMatrix gate's decision, and
# the policy never fires a shot in this engine.  So the reward the policy is
# credited with is the threat IT actually created, which is the honest
# number — and it is the highest of all the intents, which is the opposite
# of what the broken join produced.

_CHAIN_CALLS: List[Tuple[List["DecisionRecord"], List[Any]]] = []
_SAVED_DISPATCHER: Dict[str, Any] = {}

# Events that are not the on-ball action of the player they name.  A
# BALL_RECEIPT is the pass ARRIVING at the new carrier — it is emitted by the
# carrier, before the action he then takes.  A PRESS is not an on-ball action
# at all.
_NON_ACTION_EVENTS = frozenset({"BALL_RECEIPT", "PRESS", "PRESS_RESISTED"})

# …and the CONSEQUENCES of a failed action.  The failed action's OWN event
# already carries the loss charge — `threat_return` sees a PASS with
# outcome=False and charges the opponent's threat where the ball died — so
# claiming the consequence as well DOUBLE-CHARGES the same loss, against
# whichever decision happened to come next.  Measured over 3 real matches,
# excluding them moved 26 of 630 SAFE_PASS pairings off a GROUND_DUEL, and
# took SHOOT from +0.0201 to +0.0372 (the double-charge was penalising it).
_CONSEQUENCE_EVENTS = frozenset({
    "TURNOVER", "MISCONTROL", "DISPOSSESSED", "INTERCEPTION",
    "GROUND_DUEL", "AERIAL_DUEL",
})

#: Everything the causal join is allowed to claim as "the action".
_UNCLAIMABLE_EVENTS = _NON_ACTION_EVENTS | _CONSEQUENCE_EVENTS


def _make_chain_wrapper(func):
    """Build the observer wrapper for one chain entry point.

    It records the decisions made during the call and the events the call
    produced, and returns the chain result UNCHANGED — it is an observer,
    and it must not alter a single match.
    """

    def wrapper(*a, **kw):
        n0 = len(_COLLECTOR)
        res = func(*a, **kw)
        _CHAIN_CALLS.append((
            list(_COLLECTOR[n0:]),
            list(getattr(res, "events", None) or []),
        ))
        return res

    return wrapper


def _install_dispatcher_capture():
    """Wrap the chain entry points so each call records the decisions made
    during it together with the events it produced.

    The wrapper returns the chain result untouched — it is an observer, and
    it must not change a single match.
    """
    from event_chain import ChainDispatcher

    _CHAIN_CALLS.clear()
    _SAVED_DISPATCHER.clear()
    for name in ("possession", "attack", "set_piece"):
        raw = ChainDispatcher.__dict__.get(name)
        if raw is None:
            continue
        _SAVED_DISPATCHER[name] = raw
        func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        setattr(ChainDispatcher, name, staticmethod(_make_chain_wrapper(func)))


def _restore_dispatcher_capture():
    from event_chain import ChainDispatcher

    for name, raw in _SAVED_DISPATCHER.items():
        setattr(ChainDispatcher, name, raw)
    _SAVED_DISPATCHER.clear()
    _CHAIN_CALLS.clear()


def _correlate_outcomes_causal(
    calls: Sequence[Tuple[List["DecisionRecord"], List[Any]]],
    timeline: Optional[List[Any]] = None,
    attacks_right_by_team: Optional[Dict[str, bool]] = None,
    grid: Optional[Sequence[Sequence[float]]] = None,
) -> List[Tuple]:
    """Join each decision to the action event of its OWN chain call.

    Within a chain the order is causal: the decision is made, then the
    action event is created, then the consequences.  So the i-th decision
    joins to the first still-unclaimed ACTION event of that chain by the
    same player, and each event is claimed at most once.

    An ACTION event is one the player performed.  Two kinds are excluded:
    events that are not the player's action at all (a BALL_RECEIPT is the
    pass arriving at the new carrier; a PRESS is not an on-ball action), and
    the CONSEQUENCES of a failed action (a TURNOVER / DISPOSSESSED /
    GROUND_DUEL …), because the failed action's own event already carries
    the loss charge in ``threat_return`` — claiming the consequence as well
    would double-charge the same loss against the next decision.

    A decision with no action event in its own chain was never executed
    (the chain overrode it).  It is recorded NEUTRAL — 0.0 — rather than
    being charged with somebody else's event, which is what the proximity
    join did.

    **A SHOOT DECISION IS ANSWERED BY THE SHOT IT CAUSED.**  This is the
    one place the outcome lives in ANOTHER chain, and it is a structural
    fact of the engine, not a quirk: a policy SHOOT intent makes the
    possession chain create a ZERO-LENGTH marker carry
    (``location_x=x, end_x=x``), whose threat change is EXACTLY 0.0, and
    the real shot is fired by the ATTACK chain.  So the chain-local join
    credits the shoot decision with 0.0, always, while the shot's xG — the
    whole value of the chance — is credited to nobody.  The surrogate then
    cannot express "shooting is worth it", which is T2's collapse reached
    by a third route.

    So a SHOOT decision looks forward in the TIMELINE for the shot it
    caused: the first shot-ish event by the same player, at or after this
    chain's first event, in the SAME MINUTE (the possession chain and the
    attack chain run inside one minute of the match).  The minute bound is
    what keeps this from becoming the proximity join again.  Only one
    policy decision site exists in the engine (the possession chain), so
    the shot cannot be claimed twice.
    """
    pairs: List[Tuple] = []
    index_of: Dict[int, int] = {}
    if timeline is not None:
        index_of = {id(ev): i for i, ev in enumerate(timeline)}
    used_timeline: set = set()

    for decisions, events in calls:
        used = set()
        chain_start = len(timeline) if timeline is not None else 0
        if timeline is not None:
            _idxs = [index_of.get(id(e)) for e in events]
            _idxs = [i for i in _idxs if i is not None]
            if _idxs:
                chain_start = min(_idxs)
        for rec in decisions:
            ev = None
            # a SHOOT decision is answered by the shot it caused
            if (rec.intent is PlayerIntent.SHOOT and timeline is not None):
                for j in range(chain_start, len(timeline)):
                    if j in used_timeline:
                        continue
                    cand = timeline[j]
                    if getattr(cand, "player", None) != rec.player:
                        continue
                    if event_type_name(cand) not in _SHOT_TYPES:
                        continue
                    if abs(float(getattr(cand, "minute", 0.0))
                           - float(rec.minute)) > 1.0:
                        continue
                    ev = cand
                    used_timeline.add(j)
                    break
            if ev is None:
                for idx, e in enumerate(events):
                    if idx in used:
                        continue
                    if getattr(e, "player", None) != rec.player:
                        continue
                    if event_type_name(e) in _UNCLAIMABLE_EVENTS:
                        continue
                    ev = e
                    used.add(idx)
                    break
            if ev is None:
                pairs.append((rec.sensors, rec.intent, 0.0, "NONE", rec.position))
                continue
            ar = True if not attacks_right_by_team else bool(
                attacks_right_by_team.get(getattr(ev, "team", None), True))
            reward = _event_success(ev, attacks_right=ar, grid=grid)
            pairs.append((rec.sensors, rec.intent, reward,
                          event_type_name(ev), rec.position))
    return pairs


# ─────────────────────────────────────────────────────────────
# WHO DECIDES DURING COLLECTION
# ─────────────────────────────────────────────────────────────
#
# ``collect_from_matches`` has always documented "Run n real matches with the
# HEURISTIC brain".  Since the 2026-09-09 live swap it no longer did:
# ``event_chain`` calls ``NeuralDecisionBrain.decide`` directly, and that
# auto-loads ``BRAIN_DIR/<position>.json``.  Nothing pinned the heuristic, so
# collection silently ran on the trained T1 brains -- the very set the reward
# change exists to replace.
#
# That is not a cosmetic drift.  The surrogate is a table of
# ``E[threat return | position, bucket, intent]`` fitted over whatever intents
# the collecting policy happened to try, and every unsampled intent keeps the
# ``prior`` of 0.0.  Fit the table on a collapsed policy and the un-tried
# intents read as worthless while a completed safe pass reads positive --
# which is the documented T2 failure, reached here through coverage rather
# than through the objective.
#
# Keeping the collecting policy identical to T1's also leaves the REWARD as
# the single changed variable, which is what a controlled retrain needs.
# ``_diag_collect_who_decides.py`` measures which brains are actually bound.
_SAVED_ONBALL = None
_HEURISTIC_DECIDE = DecisionBrain.decide


def _heuristic_decide_compat(*a, **kw):
    """``DecisionBrain.decide`` with the post-swap kwargs it predates.

    ``event_chain.py`` calls ``NeuralDecisionBrain.decide(...)``, passing
    ``coach_instructions=``; the hand-calibrated heuristic predates that
    parameter and raises ``TypeError: got an unexpected keyword argument``
    on it.  ``tools/gates/validate_neural_xl.py::_run_heuristic`` uses the
    identical adapter — same reason, same kwarg — and that is the precedent
    this follows.  Only kwargs the heuristic does not know are dropped; the
    sensor snapshot in ``_snapshot_decide`` still sees the full call.
    """
    kw.pop("coach_instructions", None)
    return _HEURISTIC_DECIDE(*a, **kw)


COLLECT_POLICIES = ("heuristic", "neural")


def _pin_on_ball_policy(policy: str = "heuristic"):
    """Bind the on-ball policy the collector is documented to use.

    Returns a restore callable.  ``neural`` keeps whatever is bound (an
    explicit on-policy run); ``heuristic`` routes every touch through the
    genuine hand-calibrated ``DecisionBrain.decide``.

    Only the ON-BALL entry point is pinned.  The team press controller is left
    alone deliberately: it changes the state distribution the buckets are
    computed from, and collection should sample the phase the evolved brains
    will be evaluated in.
    """
    global _SAVED_ONBALL
    if policy not in COLLECT_POLICIES:
        raise ValueError(f"policy must be one of {COLLECT_POLICIES}, got {policy!r}")
    if policy == "neural":
        return lambda: None
    _SAVED_ONBALL = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_heuristic_decide_compat)

    def _restore():
        global _SAVED_ONBALL
        NeuralDecisionBrain.decide = _SAVED_ONBALL
        _SAVED_ONBALL = None

    return _restore


# ─────────────────────────────────────────────────────────────
# OUTCOME -> THREAT RETURN
# ─────────────────────────────────────────────────────────────
#
# The reward used to be "did this engine complete the action", which made a
# square ball (completed) beat a through-ball (usually not) and drove the GA to
# lateral spread.  It is now the CHANGE IN THREAT the action caused -- see
# `threat_reward.py` for the full argument.  The scaling is roughly
# [-0.6, +1.0] and SIGNED, which is why `FitnessSurrogate.prior` is 0.0 and
# `brain_evolution.synthetic_fitness` re-bases it before the confidence
# weighting (a signed mean times a confidence factor clamps to zero otherwise).

from threat_reward import (  # noqa: E402  (grouped for readability)
    threat_return, event_type_name, default_grid as _threat_grid,
    EXECUTION_TYPES as _EXECUTION_TYPES, describe as describe_reward,
    SHOT_TYPES as _SHOT_TYPES,
)


def _event_success(ev: Any, attacks_right: bool = True,
                   grid: Optional[Sequence[Sequence[float]]] = None) -> float:
    """Threat return of one execution event (legacy name, new meaning).

    Kept as a thin alias so callers and tests that imported the old name keep
    working; the substance lives in :mod:`threat_reward`.
    """
    return threat_return(ev, grid=grid, attacks_right=attacks_right)


def _correlate_outcomes(
    timeline: List[Any],
    records: List[DecisionRecord],
    attacks_right_by_team: Optional[Dict[str, bool]] = None,
    grid: Optional[Sequence[Sequence[float]]] = None,
) -> List[Tuple]:
    """Pair each decision record with its outcome by scanning events.

    ⚠️  THIS IS THE LEGACY PROXIMITY JOIN, AND IT IS KNOWN TO MIS-PAIR.
    It is kept because the tests exercise it and because
    ``surrogate_rebuild.py`` still calls it, but the PRODUCTION collection
    path uses :func:`_correlate_outcomes_causal` instead.

    The defect: "the first execution event by P at or after (M)" is not the
    event P's decision caused.  A decision can emit more than one execution
    event (a failed pass emits PASS *and* TURNOVER / DISPOSSESSED), and this
    function consumes one event per record, so the extra event is charged
    against the NEXT decision of that player.  Measured over 3 real matches,
    the self-consistency of the resulting pairs was 12.5% for SHOOT, 0% for
    CROSS, 10% for THROUGH_BALL and 64.8% for SAFE_PASS — and the surrogate
    fitted from it ranked SHOOT at -0.0133, which would teach the GA never
    to shoot.

    For each record (player P, minute M, order O), find the first
    execution event by P in the globally-ordered timeline that occurs
    at or after (M).  Because a player's on-ball decision immediately
    precedes their execution event, and no other execution by P can
    interleave, the first match in order is the outcome.

    ``attacks_right_by_team`` maps team name -> ``PositionEngine``'s own
    ``team_attacks_right``.  It matters: the reward is a threat CHANGE, and
    the xT grid is directional, so scoring a pass without knowing which way
    the team attacks scores half of them backwards.  Absent (the default) it
    falls back to ``True``, which is what the previous success metric
    effectively assumed.
    """
    # Build per-player event queue in timeline order
    by_player: Dict[str, List[Dict[str, Any]]] = {}
    for ev in timeline:
        p = getattr(ev, "player", None)
        if p is None or event_type_name(ev) not in _EXECUTION_TYPES:
            continue
        by_player.setdefault(p, []).append({
            "minute": getattr(ev, "minute", 0),
            "second": getattr(ev, "second", 0),
            "ev": ev,
        })

    # sort records by (player, order) so we consume the right events
    records_sorted = sorted(records, key=lambda r: (r.player, r.order))
    # per-player event cursor
    cursor: Dict[str, int] = {}

    paired = []
    for rec in records_sorted:
        events = by_player.get(rec.player, [])
        i = cursor.get(rec.player, 0)
        # advance past events before this decision's minute
        while i < len(events) and events[i]["minute"] < rec.minute:
            i += 1
        if i < len(events):
            ev = events[i]["ev"]
            ar = True if not attacks_right_by_team else bool(
                attacks_right_by_team.get(getattr(ev, "team", None), True))
            reward = _event_success(ev, attacks_right=ar, grid=grid)
            # record the resulting action type too (helps interpret)
            paired.append((rec.sensors, rec.intent, reward, event_type_name(ev),
                           rec.position))
            cursor[rec.player] = i + 1
        else:
            # No execution found: NEUTRAL 0.0.  A positive default here would
            # be a reward for an action that never happened, and with a signed
            # reward it would also make every unobserved intent look better
            # than everything the player actually did.
            paired.append((rec.sensors, rec.intent, 0.0, "NONE", rec.position))

    return paired


# ─────────────────────────────────────────────────────────────
# MATCH RUNNER + COLLECTION
# ─────────────────────────────────────────────────────────────

# Squad templates.  The base template mirrors the live harness (two CMs,
# no CAM/CF).  We rotate across templates during collection so every one
# of the 11 positions actually gets real-match touches (previously CAM
# and CF never appeared, so the surrogate had no data for them).
TEMPLATE_BASE = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
TEMPLATE_CAM = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CAM", "CAM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
TEMPLATE_CF = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("CF", "CF"), ("RW", "RW"),
]
# Rotation order so CAM and CF appear, and ST is on the pitch 2/3 of the time.
_COLLECTION_TEMPLATES = [TEMPLATE_BASE, TEMPLATE_CAM, TEMPLATE_CF]


def _template_name(template: List[tuple]) -> str:
    pos = [p for _, p in template]
    if "CAM" in pos:
        return "CAM"
    if "CF" in pos:
        return "CF"
    return "base"


def _build_match_components(home: str, away: str, home_style: str, away_style: str,
                            template: Optional[List[tuple]] = None,
                            home_template: Optional[List[tuple]] = None,
                            away_template: Optional[List[tuple]] = None):
    from match_engine import (
        MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
    )
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController

    if home_template is None or away_template is None:
        # legacy single-template call
        if template is None:
            template = TEMPLATE_BASE
        home_template = away_template = template
    subs = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]

    home_squad = SquadBuilder.build(home, list(home_template), list(subs))
    away_squad = SquadBuilder.build(away, list(away_template), list(subs))

    style_map = {
        "balanced": TeamStyle.BALANCED, "attacking": TeamStyle.ATTACKING,
        "defensive": TeamStyle.DEFENSIVE, "fluid_counter": TeamStyle.FLUID_COUNTER,
        "tiki_taka": TeamStyle.TIKI_TAKA, "wing_play": TeamStyle.WING_PLAY,
        "ultra_attacking": TeamStyle.ULTRA_ATTACKING,
    }

    def _tp(name, st):
        return TeamProfile(name=name, style=style_map.get(st, TeamStyle.BALANCED),
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)

    config = MatchConfig(home_team=home, away_team=away,
                         match_date=date(2026, 9, 6), matchday=3, season="26/27")
    hp = _tp(home, home_style)
    ap = _tp(away, away_style)
    sc = SubstitutionController(home_team=home, away_team=away,
                                home_subs_bench=home_squad["substitutes"],
                                away_subs_bench=away_squad["substitutes"])

    eng = MatchEngine(config, hp, ap)
    eng.set_squad(home, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(away, away_squad["starters"], away_squad["substitutes"])
    eng.set_stamina_controller(sc)
    return eng


def collect_from_matches(
    n_matches: int = 1,
    home: str = "Probe FC",
    away: str = "Rival FC",
    home_style: str = "balanced",
    away_style: str = "fluid_counter",
    seed: int = 0,
    verbose: bool = True,
    away_styles: Optional[List[str]] = None,
    policy: str = "heuristic",
    explore: float = 0.0,
    explore_situational: bool = False,
) -> List[Tuple[np.ndarray, PlayerIntent, float, str, str]]:
    """Run n real matches with the heuristic brain, collecting training data.

    If ``away_styles`` is given, opponent tactics are swept across it
    (rotating per match) so the surrogate isn't tuned to one opponent
    shape.  Returns list of (sensor_vector, intent, success, action_type,
    position).

    ``policy`` is what actually decides (see ``_pin_on_ball_policy``):
    ``heuristic`` (default, and what this function has always claimed to do)
    or ``neural`` for a deliberate on-policy collection.

    ``explore`` is the ε of ε-exploration: the probability that a decision is
    replaced by a random action the player's ROLE can perform, so the corpus
    covers the actions the policy never chooses.  0.0 = the policy's own
    choice (the historical behaviour).  See ``_explore_intent``.
    """
    global _COLLECTOR, _EXPLORE_EPS, _EXPLORE_RNG, _EXPLORED
    global _EXPLORE_SITUATIONAL, _SITUATIONAL_RECORDS
    # clear collector + install patch
    _COLLECTOR = []
    _SITUATIONAL_RECORDS = []
    _EXPLORE_EPS = float(explore)
    _EXPLORE_RNG = random.Random(seed + 977)
    _EXPLORED = 0
    _EXPLORE_SITUATIONAL = bool(explore_situational)
    restore_policy = _pin_on_ball_policy(policy)
    _install_collector()

    if away_styles is None:
        away_styles = [away_style]

    all_paired = []
    try:
        for m in range(n_matches):
            random.seed(seed + m * 1000)
            opp_style = away_styles[m % len(away_styles)]
            template = _COLLECTION_TEMPLATES[m % len(_COLLECTION_TEMPLATES)]
            # Each match is its own set of chain calls.  Without this clear,
            # match 2 would re-pair match 1's calls as well.
            _CHAIN_CALLS.clear()
            eng = _build_match_components(home, away, home_style, opp_style,
                                          template=template)
            result = eng.simulate()
            # The threat reward is directional, so it needs to know which end
            # each team attacks.  Read from the engine's own position layer
            # (set once per team at initialize_team) rather than assuming.
            _ar: Dict[str, bool] = {}
            try:
                _ar = dict(eng.position_engine.team_attacks_right)
            except Exception:
                _ar = {}
            paired = _correlate_outcomes_causal(
                _CHAIN_CALLS, timeline=result.timeline,
                attacks_right_by_team=_ar)
            if not paired and _COLLECTOR:
                # The dispatcher capture did not fire (a chain entered by a
                # path we do not wrap).  Fall back to the proximity join and
                # SAY SO — a silent fallback would reintroduce the defect
                # this function exists to remove, and a quietly wrong
                # training set is the worst possible outcome here.
                print("  WARNING: no chain calls captured — falling back to "
                      "the PROXIMITY join (known mis-paired; see "
                      "_correlate_outcomes_causal)")
                paired = _correlate_outcomes(
                    result.timeline, list(_COLLECTOR),
                    attacks_right_by_team=_ar)
            all_paired.extend(paired)
            _SITUATIONAL_RECORDS.extend(_COLLECTOR)
            _COLLECTOR.clear()
            if verbose:
                print(f"  match {m+1}: collected {len(paired)} pairs "
                      f"(opp={opp_style}, template={_template_name(template)}, "
                      f"policy={policy})")
    finally:
        _restore_collector()
        # Restore the TRUE original on-ball entry point.  _restore_collector
        # only puts back the PINNED decide, so without this the heuristic
        # would stay bound in the process after collection.
        restore_policy()
        # and the situational exploration off again: it is a collection-time
        # switch, and leaving it on would change every later collection in
        # the same process.
        _EXPLORE_SITUATIONAL = False

    return all_paired


# ─────────────────────────────────────────────────────────────
# SURROGATE: EMPIRICAL EXPECTED-SUCCESS TABLE
# ─────────────────────────────────────────────────────────────

# Coarse situational bins derived from key sensors.  This keeps the
# surrogate interpretable and robust to sparse data, while still
# capturing the dominant factors (pressure, space, depth, defense shape).
# Sensor indices (see brain_sensors):
#   4 nearest_defender_dist   9 space_ahead
#   12 is_final_third         13 is_own_half
#   14 goal_distance_norm     15 central_lane
#   16 under_pressure

@dataclass
class FitnessSurrogate:
    """Learned mapping position → (situation-bin, intent) → expected THREAT RETURN.

    Built from empirical real-match data, position-aware (the ST learns
    attacking values separately from the CB).  The table layout:
        self.table[position][bucket] = {intent: expected_threat_return}
        self.table[""][bucket]        = {intent: expected_threat_return}   # global

    ``prior`` is **0.0, not 0.5**.  The reward is SIGNED (a backward pass and
    a loss in a dangerous area are negative), so 0.0 is the true neutral point
    — "this intent gained no threat".  A 0.5 prior would tell evolution that
    every intent never sampled during collection is better than everything the
    player actually did, which is how a positive default becomes a hidden
    preference for the unmeasured.
    """
    table: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    prior: float = 0.0

    # ── ROLE-SCOPED POOLING (2026-10-20) ────────────────────────────────
    # The fallback chain used to be: position cell -> GLOBAL row cell.
    # The global row pools EVERY position, so a role-specific action's value
    # leaks across roles: the ST was reading the WINGERS' crossing value
    # (+0.0420 in the clear-chance bucket) against his own shooting value
    # (-0.0017), and the GA duly evolved a striker who CROSSED in 43% of
    # states and never shot.  Measured and reproduced; see AGENTS.md.
    #
    # The fallback chain is now: position cell -> ROLE FAMILY cell -> the
    # global row, and the global row may only lend an intent the ROLE FAMILY
    # actually performed.  `ROLE_FAMILY` is the project's own map
    # (role_features): ST/CF, LW/RW, LB/RB, CM, CAM, CDM, CB, GK.
    family_table: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    family_counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    #: family -> the set of intents that family actually performed.  This is
    #: the ROLE-APPROPRIATE intent vocabulary, derived from the DATA rather
    #: than invented: the ST family performed SHOOT but never CROSS, so the
    #: strikers can never borrow a crossing value.
    family_intents: Dict[str, Set[str]] = field(default_factory=dict)

    # ── PER-POSITION REWARD SCALE (2026-10-21) ─────────────────────────
    # The reward's spread is NOT comparable across positions: a centre-back's
    # actions swing the threat by ~0.09 while a midfielder's swing it by
    # ~0.002 (measured on the v5 corpus, fitted cells).  `synthetic_fitness`
    # then adds two guards with FIXED weights (0.05 each), so where the
    # reward was flat the guards were ~15x the football and the GA optimised
    # peakedness — which IS a collapsed intent mix.  That is the measured
    # cause of the v5 census (8 of 11 positions down to <=3 intents,
    # mean_conf 0.78-0.97 against the shipped 0.19-0.38).
    #
    # `scales[position]` is the count-weighted standard deviation of that
    # position's FITTED cell values (floor `_SCALE_FLOOR`, and 1.0 when the
    # position has fewer than `_SCALE_MIN_CELLS` cells).  `expected_success`
    # divides by it, so every position's reward has an O(1) spread and the
    # guards' fixed 0.05 is a genuine 5% tie-breaker everywhere.
    #
    # This is a monotone per-position transform, so the ranking of intents
    # WITHIN a position is unchanged — only the cross-position comparability
    # changes, which is the whole point.  The stored `table`/`counts` stay in
    # RAW units: the scale is applied at read time, so a fitted value remains
    # interpretable and the raw corpus can be refitted freely.
    #
    # THE FORBIDDEN VETO IS SCALED WITH IT.  `FORBIDDEN_VALUE = -1.0` is a
    # RAW-scale number, chosen because the raw threat return's range is about
    # [-0.6, +1.0].  Dividing it with the same scale as every other value
    # keeps the invariant STRUCTURAL: every raw reward is above -1.0, so
    # every scaled reward is above -1.0/scale, which is where the scaled veto
    # sits.  Leaving it undivided is NOT safe: on the real corpus the scaled
    # cells run as low as -7.59 and 59 cells fall below a fixed -1.0, so the
    # veto would sit ABOVE a legitimate action and hand the GA a reason to
    # perform the forbidden one.  Measured by
    # `tools/diag/_diag_forbidden_scale.py`.
    scales: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def bucket(sensors: np.ndarray) -> str:
        """Map a sensor vector to a coarse situation key."""
        pressure = int(sensors[16] > 0.5)
        final_third = int(sensors[12] > 0.5)
        own_half = int(sensors[13] > 0.5)
        space = int(sensors[9] > 0.4)   # has space ahead
        central = int(sensors[15] > 0.5)
        goal_close = int(sensors[14] < 0.5)  # within ~half of max goal distance
        return f"P{pressure}FT{final_third}OH{own_half}S{space}C{central}G{goal_close}"

    def fit(self, data: List[Tuple[np.ndarray, PlayerIntent, float, str, str]],
            min_samples: int = 3) -> "FitnessSurrogate":
        """Build per-position tables from (sensors, intent, success, action, position)."""
        # group samples by position, with "" as the global aggregate
        grouped: Dict[str, List[Tuple[np.ndarray, PlayerIntent, float]]] = {}
        for sensors, intent, success, _action, position in data:
            grouped.setdefault(position, []).append((sensors, intent, success))
            grouped.setdefault("", []).append((sensors, intent, success))  # global
            _fam = _role_family(position)
            if _fam:
                grouped.setdefault(_FAMILY_KEY_PREFIX + _fam, []).append(
                    (sensors, intent, success))

        for pos, samples in grouped.items():
            acc: Dict[str, Dict[str, float]] = {}
            cnt: Dict[str, Dict[str, int]] = {}
            for sensors, intent, success in samples:
                key = self.bucket(sensors)
                ik = intent.value if hasattr(intent, "value") else str(intent)
                acc.setdefault(key, {}).setdefault(ik, 0.0)
                cnt.setdefault(key, {}).setdefault(ik, 0)
                acc[key][ik] += success
                cnt[key][ik] += 1
            for key, intents in cnt.items():
                tot_s = sum(acc[key].get(ik, 0.0) for ik in intents)
                tot_c = sum(intents.values())
                overall = tot_s / tot_c if tot_c else self.prior
                for ik, c in intents.items():
                    if c < min_samples:
                        # shrink toward overall situation mean
                        est = (acc[key][ik] + overall * min_samples) / (c + min_samples)
                    else:
                        est = acc[key][ik] / c
                    if pos.startswith(_FAMILY_KEY_PREFIX):
                        _fam = pos[len(_FAMILY_KEY_PREFIX):]
                        self.family_table.setdefault(_fam, {}).setdefault(key, {})[ik] = est
                        self.family_counts.setdefault(_fam, {}).setdefault(key, {})[ik] = c
                    else:
                        self.table.setdefault(pos, {}).setdefault(key, {})[ik] = est
                        self.counts.setdefault(pos, {}).setdefault(key, {})[ik] = c

        # The role-appropriate intent vocabulary, derived from the DATA: the
        # intents the positions of each family actually performed.  The ST
        # family performed SHOOT but never CROSS, so the strikers can never
        # borrow a crossing value — which is the leak that produced a
        # striker who crossed 43% of the time and never shot.
        for _pos in list(self.table):
            if _pos == "":
                continue
            _fam = _role_family(_pos)
            if not _fam:
                continue
            _used = self.family_intents.setdefault(_fam, set())
            for _bucket, _intents in self.table.get(_pos, {}).items():
                _used.update(_intents.keys())

        # Per-position reward SCALE (see the `scales` field docstring).  The
        # count-weighted std of this position's FITTED cells — fitted, not
        # raw, because the shrinkage is what makes the spread vary 45x: the
        # raw per-position sd varies only ~4.8x, and it is the cell values
        # the GA actually reads.  The global row is skipped: it pools every
        # position, so no single scale is meaningful for it.
        self.scales = {}
        for _pos, _buckets in self.table.items():
            if _pos == "":
                continue
            _vals: List[float] = []
            _wts: List[float] = []
            for _bucket, _intents in _buckets.items():
                _crow = self.counts.get(_pos, {}).get(_bucket, {})
                for _ik, _v in _intents.items():
                    _vals.append(float(_v))
                    _wts.append(float(_crow.get(_ik, 1)))
            if len(_vals) < _SCALE_MIN_CELLS:
                self.scales[_pos] = 1.0
                continue
            _tw = sum(_wts)
            if _tw <= 0.0:
                self.scales[_pos] = 1.0
                continue
            _mean = sum(v * w for v, w in zip(_vals, _wts)) / _tw
            _var = sum(w * (v - _mean) ** 2 for v, w in zip(_vals, _wts)) / _tw
            self.scales[_pos] = max(_SCALE_FLOOR, _var ** 0.5)
        return self

    def _lookup(self, key: str, ik: str, pos: str) -> Optional[float]:
        # Position cell first.  Then the ROLE FAMILY's cell for the same
        # bucket and intent, then the global row for an intent the family
        # actually performed.  The global row is never allowed to lend a
        # role-specific action across roles: that is how the ST was handed
        # the wingers' crossing value, and the GA evolved a striker who
        # crossed 43% of the time and never shot.  We also NEVER return
        # the position row's bucket mean across *other* intents as if it
        # was this intent's value: that silently conflates e.g.
        # SWITCH/CARRY outcomes with SHOOT, which mis-educates evolution
        # about shooting value.
        # (the family map is role_features.ROLE_FAMILY)
        row = self.table.get(pos, {}).get(key, {})
        if ik in row:  # exact position cell first
            return row[ik]
        _fam = _role_family(pos)
        # A ROLE-SPECIFIC action the role cannot perform is FORBIDDEN, and
        # the check comes FIRST: it must not be readable from the position,
        # from the family, or from the global row.  The forbidden value sits
        # far below any real reward, because the neutral prior (0.0) would
        # otherwise be ATTRACTIVE next to a slightly-negative real cell --
        # "ignorance is attractive" -- and the striker took a 2.5% share of
        # crossing purely because nothing was known about it.  Ignorance
        # must not outrank evidence.
        _allowed = ROLE_ONLY_INTENTS.get(ik)
        if _allowed is not None and (_fam or "") not in _allowed:
            return FORBIDDEN_VALUE
        if _fam:
            _frow = self.family_table.get(_fam, {}).get(key, {})
            if ik in _frow:
                return _frow[ik]
            if ik in self.family_intents.get(_fam, ()):
                _grow = self.table.get("", {}).get(key, {})
                if ik in _grow:
                    return _grow[ik]
        # Neither this position, its role family, nor a role-appropriate
        # global pool has a REAL sample of this intent in this situation ->
        # admit ignorance (neutral prior) rather than fabricate from other
        # intents' outcomes or from another role's action.
        return None

    def expected_success(self, sensors: np.ndarray, intent: PlayerIntent,
                         position: str = "") -> float:
        """Expected THREAT RETURN for a sensor state + intent.

        Name kept for compatibility with every caller; the quantity is now the
        mean threat return, in roughly [-0.6, +1.0] for the position's OWN
        scale, and in the position's SCALED units here (see ``scales``).

        The division happens at READ time and only here, so the fitted
        ``table`` stays in raw units (interpretable, refittable) and every
        caller of this method — the GA at ``brain_evolution.py:553``, the
        probes, Percy's trainer — sees the same comparable quantity.

        THE FORBIDDEN VETO IS DIVIDED TOO, and that is a correction, not a
        detail.  ``FORBIDDEN_VALUE = -1.0`` was chosen because the RAW
        threat return's range is about [-0.6, +1.0] — on the real corpus the
        fitted cells run [-0.124, +0.137], so in RAW units the veto is below
        every legitimate reward.  After the per-position scale the scaled
        cells run as low as **-7.59** (RB), and 59 cells across the corpus
        fall below a FIXED -1.0.  A veto that sits ABOVE a legitimate action
        is not a veto: the GA is handed a reason to cross.  Dividing it with
        the same scale as everything else restores the invariant STRUCTURALLY
        — every raw reward is above -1.0, so every scaled reward is above
        -1.0/scale, which is exactly where the scaled veto sits.
        """
        key = self.bucket(sensors)
        ik = intent.value if hasattr(intent, "value") else str(intent)
        v = self._lookup(key, ik, position)
        if v is None:
            return self.prior
        _scale = self.scales.get(position, 1.0) or 1.0
        return v / _scale

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {"prior": self.prior, "table": self.table,
                "counts": self.counts,
                # The role-family pooling (see the class docstring): a
                # position may borrow its family's value for an intent, and
                # the global row only for an intent the family performed.
                "family_table": self.family_table,
                "family_counts": self.family_counts,
                "family_intents": {k: sorted(v)
                                   for k, v in self.family_intents.items()},
                # The per-position reward scale (see the `scales` field
                # docstring).  It travels WITH the table because a table read
                # without its scale is not the quantity the GA was trained on:
                # the same fitted value means a different amount of football
                # for a centre-back than for a midfielder.
                "scales": self.scales,
                # The reward definition travels WITH the table, because a
                # table of numbers is unreadable six months later and a
                # retrained surrogate must not be silently compared to one
                # fitted to a different objective.
                "reward": describe_reward()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FitnessSurrogate":
        with open(path, "r") as f:
            data = json.load(f)
        s = cls(prior=data.get("prior", 0.0))
        s.table = data.get("table", {})
        s.counts = data.get("counts", {})
        s.family_table = data.get("family_table", {})
        s.family_counts = data.get("family_counts", {})
        s.family_intents = {k: set(v) for k, v in
                            data.get("family_intents", {}).items()}
        # The per-position reward scale.  A surrogate saved WITHOUT it (the
        # pre-2026-10-21 tables) loads with an empty scale, which means
        # "no division" — i.e. exactly the old behaviour, not a crash and not
        # a silent zero.
        s.scales = {k: float(v) for k, v in data.get("scales", {}).items()}
        return s

    def report(self) -> Dict[str, Any]:
        positions = list(self.table.keys())
        n_bins = sum(len(v) for v in self.table.values())
        n_cells = sum(len(v) for v in self.table.values() for v2 in v.values())
        return {"positions": positions, "bins": n_bins,
                "filled_cells": n_cells, "reward": describe_reward(),
                # The per-position reward scale, so a report says what unit
                # the reward is in.  A reader who does not see this cannot
                # tell a CB's 0.09 from a CM's 0.002 apart.
                "scales": dict(self.scales)}


def build_surrogate(
    n_matches: int = 5,
    seed: int = 0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    away_styles: Optional[List[str]] = None,
    policy: str = "heuristic",
    explore: float = 0.0,
    explore_situational: bool = False,
) -> FitnessSurrogate:
    """Collect real-match data and fit a FitnessSurrogate.

    ``away_styles`` sweeps opponent tactics so the surrogate generalizes
    beyond a single opponent shape.  ``policy`` is what actually decides
    (``heuristic`` by default — see ``_pin_on_ball_policy``).
    ``explore_situational`` turns on the SITUATION-AWARE ε-exploration
    (default OFF — see `_explore_intent` and AGENTS.md *THE CEILING*).
    """
    if verbose:
        styles_txt = (f" (opponents: {away_styles})" if away_styles else "")
        print(f"Collecting {n_matches} real matches (on-ball policy: {policy})"
              f"{styles_txt}...")
        print("NOTE: each match takes ~90s on this machine; the old "
              "~15-20s figure is stale.")
    data = collect_from_matches(n_matches=n_matches, seed=seed, verbose=verbose,
                                away_styles=away_styles, policy=policy,
                                explore=explore,
                                explore_situational=explore_situational)
    if verbose:
        print(f"Total decision->outcome samples: {len(data)}")
        if explore > 0.0:
            print(f"Explored (replaced) decisions: {sc_explored()}")
    surrogate = FitnessSurrogate().fit(data)
    if verbose:
        print(f"Surrogate: {surrogate.report()}")
    if save_path:
        surrogate.save(save_path)
        if verbose:
            print(f"Saved surrogate to {save_path}")
    return surrogate


def save_raw_corpus(data: List[Tuple], path: str) -> None:
    """Save the RAW (sensors, intent, reward, event_type, position) pairs.

    A fitted table cannot be refitted: the shrinkage means the raw sums are
    not recoverable from it, so changing the FIT (the role-family pooling,
    the shrinkage, the bucket) used to mean re-running every match.  The raw
    corpus makes a refit free, and it is the same artefact
    `brain_self_trainer` already keeps.
    """
    out = []
    for sensors, intent, reward, ev_type, position in data:
        out.append({
            "sensors": [round(float(v), 6) for v in np.asarray(sensors).ravel()],
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "reward": float(reward),
            "event_type": ev_type,
            "position": position,
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pairs": out}, f)
    print(f"Saved raw corpus ({len(out)} pairs) to {path}")


def load_raw_corpus(path: str) -> List[Tuple]:
    """Read a raw corpus back into the fit's input shape."""
    from decision_brain import PlayerIntent
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data.get("pairs", []):
        out.append((np.asarray(row["sensors"], dtype=np.float64),
                    PlayerIntent(row["intent"]),
                    float(row["reward"]),
                    row.get("event_type", ""),
                    row.get("position", "")))
    return out


#: The longest run of touches one sequence may hold.  A possession is ~7
#: events, so 12 steps covers a long one without crossing into the next.
SEQUENCE_MAX_STEPS = 12

#: How many OTHER decisions may happen between two touches of the same
#: player before it counts as a new possession.  A possession is ~7 events,
#: so a gap of 20 means something else happened in between.
SEQUENCE_ORDER_GAP = 20

#: A RESTART CUT.  A goal, a kick-off or a goal kick RESETS the ball to a
#: goal line, so the carrier's x jumps 50-90 m with only a handful of
#: decisions in between — a speed no body has.  That is not two players
#: and it is not a player teleporting: it is a new context, and the
#: RECURRENT state must not be carried across it.  So a sequence is CUT
#: there.
#:
#: The test is the IMPLIED SPEED, not the distance: 85 m across 14
#: decisions is a 3 m/s JOG — perfectly possible for one man — while 85 m
#: across 2 decisions is 85 m/s, which is not football.
SECONDS_PER_DECISION = 0.5
MAX_IMPLIED_SPEED_MPS = 9.5


def build_situational_sequences(records: List[DecisionRecord]
                                ) -> List[Dict[str, Any]]:
    """Group the collected decisions into BPTT sequences.

    A sequence is a CONTIGUOUS run of touches by ONE player.  It is cut
    when:

      * another player has had more than ``SEQUENCE_ORDER_GAP`` decisions
        in between (so this is a different possession — and the
        recurrence and the plan both reset at a possession change), or
      * it reaches ``SEQUENCE_MAX_STEPS`` (so the BPTT does not run
        backwards over a whole half).

    Records without the situational blocks are SKIPPED, not faked: a
    sequence with a missing token set is not a sequence the network can be
    trained on.
    """
    by_player: Dict[str, List[DecisionRecord]] = {}
    for rec in records:
        if rec.tokens is None or rec.memory is None or rec.role is None:
            continue
        # KEYED BY (TEAM, PLAYER): the scratch fixture gives both teams
        # the same generic names, so the name alone interleaves two
        # players and the recurrent core would be trained on a state
        # that jumps halfway across the pitch between two steps.
        by_player.setdefault(f"{rec.team}|{rec.player}", []).append(rec)

    sequences: List[Dict[str, Any]] = []
    for key, recs in by_player.items():
        player = key.split("|", 1)[1]
        recs = sorted(recs, key=lambda r: r.order)
        chunk: List[DecisionRecord] = []
        prev_order = None
        prev_x = None
        for rec in recs:
            cut = False
            if prev_order is not None:
                gap = rec.order - prev_order
                if gap > SEQUENCE_ORDER_GAP or len(chunk) >= SEQUENCE_MAX_STEPS:
                    cut = True
                elif prev_x is not None and gap >= 1:
                    # A RESTART: the ball was reset to a goal line, so
                    # the carrier's x jumped at a speed no body has.  The
                    # recurrent context must not cross it.
                    #
                    # `sensors[2]` is the NORMALISED x (it is `x / 105.0`),
                    # so the jump in METRES is the difference of the raw
                    # values times 105.  (Multiplying an already-converted
                    # value by 105 a second time is a 105x error, and it
                    # cuts EVERY sequence — which is exactly what it did.)
                    jump = abs(float(rec.sensors[2]) - prev_x) * 105.0
                    if jump / (gap * SECONDS_PER_DECISION) > MAX_IMPLIED_SPEED_MPS:
                        cut = True
            if cut:
                _emit_sequence(sequences, chunk, player)
                chunk = []
            chunk.append(rec)
            prev_order = rec.order
            prev_x = float(rec.sensors[2])
        _emit_sequence(sequences, chunk, player)
    return sequences


def _emit_sequence(out: List[Dict[str, Any]], chunk: List[DecisionRecord],
                   player: str) -> None:
    if len(chunk) < 2:
        return
    steps = []
    prev_order = None
    for r in chunk:
        # THE ORDER GAP: how many decisions the whole match took since
        # this player's previous touch.  It is the DISCRIMINATOR between
        # an interleave and a legitimate long absence: a jump of 40 m
        # after 60 other decisions is a man who lost the ball and got it
        # back, and the SAME jump after 2 decisions is two different men.
        gap = 0 if prev_order is None else int(r.order - prev_order)
        prev_order = r.order
        steps.append({
            "tokens": [[round(float(v), 5) for v in row]
                       for row in np.asarray(r.tokens)],
            "token_mask": [float(v) for v in np.asarray(r.token_mask)],
            "state": [round(float(v), 6) for v in np.asarray(r.sensors)],
            "role": [round(float(v), 6) for v in np.asarray(r.role)],
            "memory": [round(float(v), 6) for v in np.asarray(r.memory)],
            "plan": [round(float(v), 6) for v in np.asarray(r.plan)]
            if r.plan is not None else [],
            "phase": int(r.phase),
            "minute": round(float(r.minute), 2),
            "gap": gap,
        })
    out.append({
        "player": player,
        "team": chunk[0].team,
        "position": chunk[0].position,
        "steps": steps,
    })


def save_situational_corpus(records: List[DecisionRecord], path: str) -> int:
    """Write the situational corpus (the sequences the BPTT trains on)."""
    sequences = build_situational_sequences(records)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sequences": sequences}, f)
    steps = sum(len(s["steps"]) for s in sequences)
    print(f"Saved situational corpus ({len(sequences)} sequences, "
          f"{steps} steps) to {path}")
    return len(sequences)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a fitness surrogate from real matches.")
    p.add_argument("--matches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="brains/surrogate.json")
    p.add_argument("--styles", type=str, default=None,
                   help="Comma-separated opponent styles to sweep "
                        "(e.g. fluid_counter,attacking,tiki_taka).")
    p.add_argument("--policy", type=str, default="heuristic",
                   choices=list(COLLECT_POLICIES),
                   help="On-ball policy that actually decides during collection. "
                        "'heuristic' is the documented default; 'neural' is an "
                        "on-policy run against whatever brains/ holds.")
    p.add_argument("--situational-out", type=str, default=None,
                   help="Also save the SITUATIONAL corpus: the entity tokens, "
                        "the role block, the memory block and the possession "
                        "phase for every decision, grouped into per-player "
                        "SEQUENCES for BPTT.  This is the corpus the "
                        "attention + recurrence + plan brain trains on.")
    p.add_argument("--raw-out", type=str, default=None,
                   help="Also save the RAW (sensors, intent, reward) pairs, so "
                        "the surrogate can be REFITTED without re-running the "
                        "matches.  The fitted table cannot be refitted: the "
                        "shrinkage hides the raw sums.")
    p.add_argument("--explore", type=float, default=0.0,
                   help="ε-exploration: the probability that a decision is "
                        "replaced by a random action the player's ROLE can "
                        "perform.  0.0 = the policy's own choice.  This is "
                        "how the corpus learns the actions the policy never "
                        "chooses (a centre midfielder's through ball, a "
                        "striker's shot).")
    p.add_argument("--explore-situational", action="store_true",
                   help="SITUATION-AWARE ε-exploration (default OFF).  The "
                        "uniform exploration samples an action the ROLE can "
                        "perform, but not one the SITUATION can be judged on: "
                        "the striker's clearest chance held ~10 decisions and "
                        "NOT ONE SHOT across six matches, so the surrogate "
                        "had no shot to learn from.  This raises the "
                        "exploration rate in the final third and weights the "
                        "explored action by the phase.")
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()] if args.styles else None
    if args.raw_out or args.situational_out:
        raw = collect_from_matches(n_matches=args.matches, seed=args.seed,
                                   verbose=True, away_styles=styles,
                                   policy=args.policy, explore=args.explore,
                                   explore_situational=args.explore_situational)
        print(f"Total decision->outcome samples: {len(raw)}")
        print(f"Explored (replaced) decisions: {sc_explored()}")
        surrogate = FitnessSurrogate().fit(raw)
        print(f"Surrogate: {surrogate.report()}")
        surrogate.save(args.out)
        print(f"Saved surrogate to {args.out}")
        if args.raw_out:
            save_raw_corpus(raw, args.raw_out)
        if args.situational_out:
            save_situational_corpus(list(_SITUATIONAL_RECORDS),
                                    args.situational_out)
    else:
        surrogate = build_surrogate(n_matches=args.matches, seed=args.seed,
                                    save_path=args.out, away_styles=styles,
                                    policy=args.policy, explore=args.explore,
                                    explore_situational=args.explore_situational)
    print(json.dumps(surrogate.report()))
