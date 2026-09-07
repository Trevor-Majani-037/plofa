"""
PLOFA — SET-PIECE ATTACKING ROUTINES (Feature #3)
==================================================
A pure-data library of committed set-piece ATTACKING schemes, the dead-ball
counterpart of `attack_patterns.py` / `tactical_shapes.py`.

The match PLAYS ITSELF: no score is fed in as an input. A routine is simply
the identity a side's trademark dead-ball play takes out of its DNA + the
live game state — chunk-stable (a team commits to one routine for a ~5-minute
block the same way they commit to an overload) and emergent from:
  - the manager's style (possession sides work the short corner; route-one
    sides pile the six-yard box; direct sides go for the posts),
  - the squad's aerial profile (no jumpers -> ground routines; an aerial
    monster -> post targets),
  - the game state (chasing late = numbers committed; protecting = the safe
    short corner to kill time).

Does NOT import event_chain / match_engine at module load (keeps the same
import-cycle hygiene as the other shape modules): engine concerns stay lazy.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SetPieceRoutine(Enum):
    """A committed dead-ball attacking scheme."""

    # Corner / crossed free-kick deliveries
    NEAR_POST_FLICKON  = "near_post_flickon"   # whipped at the front post, flick/glance on
    FAR_POST_OUTSWING  = "far_post_outswing"   # out-swinging ball to the long post zone
    SIX_YARD_PILE      = "six_yard_pile"       # crowd the six-yard box, crowd lethal
    PENALTY_SPOT_CROWD = "penalty_spot_crowd"  # attack the penalty spot with late runners
    SHORT_CORNER        = "short_corner"        # work it short, pull back to the edge / D

    # Free-kick schemes
    DIRECT_ATTEMPT     = "direct_attempt"      # strike the wall-face / goal
    TRAINED_CROSS      = "trained_cross"       # rehearsed delivery into the box

    @property
    def is_corner(self) -> bool:
        return self in (
            SetPieceRoutine.NEAR_POST_FLICKON,
            SetPieceRoutine.FAR_POST_OUTSWING,
            SetPieceRoutine.SIX_YARD_PILE,
            SetPieceRoutine.PENALTY_SPOT_CROWD,
            SetPieceRoutine.SHORT_CORNER,
        )

    @property
    def is_freekick(self) -> bool:
        return self in (SetPieceRoutine.DIRECT_ATTEMPT, SetPieceRoutine.TRAINED_CROSS)


# ── STYLE → CORNER ROUTINE POOLS ─────────────────────────────────────────
# Style-keyed (TeamStyle.value strings) committed pools. Order doesn't matter;
# the chunk RNG picks within the pool. Possession identities lean short/keras;
# direct identities lean the pile and the posts.
_STYLE_CORNER_POOLS: Dict[str, Tuple[SetPieceRoutine, ...]] = {
    "tiki_taka": (
        SetPieceRoutine.SHORT_CORNER,
        SetPieceRoutine.SHORT_CORNER,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.NEAR_POST_FLICKON,
    ),
    "structured_possession": (
        SetPieceRoutine.SHORT_CORNER,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.SIX_YARD_PILE,
    ),
    "vertical_tiki_taka": (
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.SHORT_CORNER,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.NEAR_POST_FLICKON,
    ),
    "balanced": (
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.SHORT_CORNER,
    ),
    "wing_play": (
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.SIX_YARD_PILE,
    ),
    "attacking": (
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
    ),
    "ultra_attacking": (
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
    ),
    "gegenpressing": (
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.FAR_POST_OUTSWING,
    ),
    "fluid_counter": (
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
        SetPieceRoutine.SHORT_CORNER,
    ),
    "route_one": (
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.NEAR_POST_FLICKON,
    ),
    "defensive": (
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.PENALTY_SPOT_CROWD,
    ),
    "ultra_defensive": (
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.FAR_POST_OUTSWING,
    ),
    "park_the_bus": (
        SetPieceRoutine.NEAR_POST_FLICKON,
        SetPieceRoutine.SIX_YARD_PILE,
        SetPieceRoutine.SHORT_CORNER,
        SetPieceRoutine.FAR_POST_OUTSWING,
    ),
}

_DEFAULT_CORNER_POOL: Tuple[SetPieceRoutine, ...] = (
    SetPieceRoutine.NEAR_POST_FLICKON,
    SetPieceRoutine.FAR_POST_OUTSWING,
    SetPieceRoutine.PENALTY_SPOT_CROWD,
    SetPieceRoutine.SIX_YARD_PILE,
)

# A squad with no aerial presence can't play the posts: ground routines only.
_GROUND_CORNER_POOL: Tuple[SetPieceRoutine, ...] = (
    SetPieceRoutine.SHORT_CORNER,
    SetPieceRoutine.SHORT_CORNER,
    SetPieceRoutine.PENALTY_SPOT_CROWD,
    SetPieceRoutine.PENALTY_SPOT_CROWD,
    SetPieceRoutine.FAR_POST_OUTSWING,
)


def corner_routine_for(
    style_name: str,
    state,
    team_name: str,
    home_team: str,
    minute: int,
    aerial_score: float = 0.5,
    chasing: bool = False,
    protecting: bool = False,
) -> SetPieceRoutine:
    """Deterministic, chunk-stable corner routine for one team+minute.

    Seeds a throwaway RNG from (team, minute // 5) so the routine holds for a
    ~5-minute block and rotates across the half. Game-state overrides are
    COMMITS, matching a manager's real dead-ball calls:
      - chasing   -> pile numbers on the six-yard box,
      - protecting-> the safe short corner that kills the clock.
    A squad without jumpers (``aerial_score`` well below 0.35) is steered to
    ground routines (short corner / penalty-spot). ``style_name`` is the
    value of TeamStyle (e.g. "route_one")."""

    if chasing:
        return SetPieceRoutine.SIX_YARD_PILE
    if protecting:
        return SetPieceRoutine.SHORT_CORNER

    pool = _STYLE_CORNER_POOLS.get(
        style_name, _DEFAULT_CORNER_POOL)

    if aerial_score < 0.35:
        pool = _GROUND_CORNER_POOL
    elif aerial_score >= 0.75:
        # An aerial monster keeps its post players on the field: drop the
        # short-corner safety valve out of the identity pool.
        pool = tuple(r for r in pool if r != SetPieceRoutine.SHORT_CORNER) or pool

    chunk = minute // 5
    rng = random.Random(f"spc|{team_name}|{home_team}|{chunk}")
    return pool[rng.randrange(len(pool))]


# ── DELIVERY PARAMETERS ──────────────────────────────────────────────────
# Each routine commits to a delivery character. ``target_zone`` is one of
# near/far/six/penalty/edge (the box zone the ball is aimed at), ``swing``
# bends the ball in toward the goal or out toward the touchline, ``height_bias``
# raises/flattens the delivery height from the taker's raw crossing quality,
# and ``crowd`` is the box-stacking commitment 0..1 (0 = nothing, up to the
# six-yard pile-in).
DeliveryParams = Dict[str, object]

_ZONE_DEFAULTS: DeliveryParams = {
    "target_zone": None,
    "swing": "in",
    "height_bias": 0.0,
    "crowd": 0.0,
}

_ROUTINE_DELIVERY: Dict[SetPieceRoutine, DeliveryParams] = {
    SetPieceRoutine.NEAR_POST_FLICKON: {
        "target_zone": "near",
        "swing": "in",
        "height_bias": -0.1,
        "crowd": 0.45,
    },
    SetPieceRoutine.FAR_POST_OUTSWING: {
        "target_zone": "far",
        "swing": "out",
        "height_bias": 0.15,
        "crowd": 0.55,
    },
    SetPieceRoutine.SIX_YARD_PILE: {
        "target_zone": "six",
        "swing": "in",
        "height_bias": 0.0,
        "crowd": 1.0,
    },
    SetPieceRoutine.PENALTY_SPOT_CROWD: {
        "target_zone": "penalty",
        "swing": "in",
        "height_bias": 0.1,
        "crowd": 0.7,
    },
    SetPieceRoutine.SHORT_CORNER: {
        "target_zone": "edge",
        "swing": "in",
        "height_bias": -0.3,
        "crowd": 0.2,
    },
}


def corner_delivery(routine: Optional[SetPieceRoutine]) -> DeliveryParams:
    """Delivery parameters for a routine; neutral defaults when None.

    The caller keeps perfect baseline behaviour by passing ``None``."""

    if routine is None:
        return dict(_ZONE_DEFAULTS)
    return dict(_ROUTINE_DELIVERY.get(routine, _ZONE_DEFAULTS))


# ── STYLE → FREE-KICK POOLS ──────────────────────────────────────────────
_STYLE_FK_POOLS: Dict[str, Tuple[SetPieceRoutine, ...]] = {
    # Direct identities put the ball on the goal-mouth.
    "route_one": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "attacking": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "ultra_attacking": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
    ),
    # Possession identities would rather work the ball into the box.
    "tiki_taka": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "structured_possession": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "vertical_tiki_taka": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "park_the_bus": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
    ),
    "defensive": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "ultra_defensive": (
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "fluid_counter": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "balanced": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
    ),
    "wing_play": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
    "gegenpressing": (
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
        SetPieceRoutine.DIRECT_ATTEMPT,
        SetPieceRoutine.TRAINED_CROSS,
    ),
}

_DEFAULT_FK_POOL: Tuple[SetPieceRoutine, ...] = (
    SetPieceRoutine.DIRECT_ATTEMPT,
    SetPieceRoutine.TRAINED_CROSS,
)

# A taker without a dead-ball strike in his locker hands it to the box.
_NO_STRIKE_FK_POOL: Tuple[SetPieceRoutine, ...] = (
    SetPieceRoutine.TRAINED_CROSS,
    SetPieceRoutine.TRAINED_CROSS,
    SetPieceRoutine.DIRECT_ATTEMPT,
    SetPieceRoutine.TRAINED_CROSS,
)


def freekick_routine_for(
    style_name: str,
    state,
    team_name: str,
    home_team: str,
    minute: int,
    direct_range: bool,
    taker_free_kick: float = 0.5,
    chasing: bool = False,
    protecting: bool = False,
) -> SetPieceRoutine:
    """Deterministic, chunk-stable free-kick routine.

    Outside the direct half-circle the ball is always worked into the box
    (``TRAINED_CROSS``). In ``direct_range`` the style identity decides
    whether to strike the wall or rehearse the cross, unless — as with the
    corner routines — the game state commits the call:
      - chasing    -> DIRECT_ATTEMPT (urgency: no time to work it),
      - protecting -> TRAINED_CROSS (the safe, low-risk dead ball).
    A taker well below 0.40 dead-ball skill forfeits the shot to the box."""

    if not direct_range:
        return SetPieceRoutine.TRAINED_CROSS
    if chasing:
        return SetPieceRoutine.DIRECT_ATTEMPT
    if protecting:
        return SetPieceRoutine.TRAINED_CROSS

    pool = _STYLE_FK_POOLS.get(style_name, _DEFAULT_FK_POOL)
    if taker_free_kick < 0.40:
        pool = _NO_STRIKE_FK_POOL

    chunk = minute // 5
    rng = random.Random(f"spf|{team_name}|{home_team}|{chunk}")
    return pool[rng.randrange(len(pool))]


# ── SQUAD AUXILIARY HELPERS ──────────────────────────────────────────────
def _attr(player, group: str, name: str, default: float) -> float:
    attrs = getattr(getattr(player, "dna", None), group, None)
    if attrs is None:
        return default
    val = getattr(attrs, name, None)
    return default if val is None else max(0.0, min(100.0, float(val)))


def aerial_presence(players: List) -> float:
    """0..1 aerial threat of an attacking squad (jumping + heading).

    Uses each player's DNA with defensive defaults so bare-bones fake players
    in tests resolve to ~0.33 without raising. Only CBs / centre-forwards are
    scored (the corner box bodies); with no such players it falls back to the
    whole outfield."""
    outfield = [p for p in players
                if getattr(p, "position", None) != "GK"]
    targets = [p for p in outfield
               if p.position in ("CB", "ST", "CF")] or outfield
    if not targets:
        return 0.5
    scores = [
        (_attr(p, "physical", "jumping", 60.0)
         + _attr(p, "technical", "heading", 60.0)) / 2.0
        for p in targets
    ]
    mean = sum(scores) / len(scores)
    return max(0.0, min(1.0, (mean - 40.0) / 60.0))


def receiver_weight(zone: Optional[str], player) -> float:
    """Selection weight for the corner-flight receiver under a routine zone.

    The counting stat stays 'corner won in the air', but WHO attacks the ball
    changes with the scheme: the posts get the big leapers, the penalty spot
    gets the late-arriving finishers, the edge/short-corner pull-back gets the
    feet of a quick press-resistant arrival. ``None`` zone (baseline) keeps
    the pure jumping+heading aerial weight."""

    j = _attr(player, "physical", "jumping", 60.0)
    h = _attr(player, "technical", "heading", 60.0)
    if zone is None:
        return j + h

    if zone == "penalty":
        comp = _attr(player, "mental", "composure", 60.0)
        fin = _attr(player, "technical", "finishing", 60.0)
        return (comp * 6.0) + (fin * 6.0)

    if zone == "edge":
        control = _attr(player, "technical", "ball_control", 60.0)
        pace = _attr(player, "physical", "pace", 60.0)
        return (control * 6.0) + (pace * 3.0)

    # near / far / six-yard: the aerial contest stays the core.
    return j + h