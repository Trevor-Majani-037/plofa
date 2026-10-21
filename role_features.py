"""Sensor schema v2 — role-specific structured feature blocks (audit §H.2.2).

Phase 3 of the PLOFA V2 vision ("do not force every player to perceive
football identically") lands here as a SENSOR BLOCK, not a new brain.

SCHEMA
------
The v1 24-d vector stays the SHARED block (audit §H.2.2: "keep the 24-d v1
vector loadable (versioned), add a role-specific perception block").  Each
canonical role then appends ITS OWN structured feature menu — concepts the
engine already knows (marking, defensive line, offside room, shooting
angle, passing lanes, overlap runs, box arrivals) expressed as plain
normalised numbers the brain can learn from:

    V2_INPUT_D[role] = V2_SHARED_D (24) + V2_ROLE_D[role]

Every role brain stays a separate network, so each XI just learns
(24 + role_D) inputs.  The network architecture never changes per role
family; new roles register a menu without rewriting the brain (audit goal).

DISCIPLINE
----------
- PURE GEOMETRY: every feature is derived only from the CURRENT frame's
  positions (the same actors the v1 sensors read).  No futurity, no
  velocity, no engine-internal state that the v1 sensor contract never
  exposed.
- DETERMINISTIC: same (player, geometry, attacks_right) -> same floats.
- NORMALISED: every feature in [0, 1] (clamped), consistent scale for the
  feed-forward net.
- SHARED with perception: when the perception layer is enabled it degrades
  the SHARED block exactly as today; the role block is derived from the
  SAME perceived geometry lists (a role feature is only as good as the
  actors the player actually sees).
- Perceptual, never prescriptive: a CB sees a high "box_threat" because
  attackers sit near the goal — the brain still chooses the intent.

Menus are keyed by the engine's canonical positions; trailing digits are
stripped ("CB1" -> "CB"), and CL/FB/DM are role FAMILIES shared by the
naturally symmetric in-game positions (LB/RB both get the FB block, etc.).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from brain_sensors import _pos, _clamp, _defenders_within

V2_SHARED_D = 24  # the v1 vector is the shared block


# ─────────────────────────────────────────────────────────────
# ROLE -> MENU (ordered feature names; the output order is fixed)
# ─────────────────────────────────────────────────────────────

MENU_NAMES: Dict[str, List[str]] = {
    "GK": [
        "attackers_ahead",      # opp forwards advancing beyond our midfield
        "attackers_close",      # nearest opp forward (high = on him)
        "defensive_shape",      # how far our defenders sit (1 - mean dist to own goal)
        "crossing_threat",      # opp wide attackers near OUR box
        "pass_open",            # max teammate lane openness (distribution options)
        "short_safety",         # release-valve teammate proximity
        "shot_threat",          # ball near our box + an opp forward on the ball
    ],
    "CB": [
        "own_line_height",      # where our deepest line sits on the pitch
        "space_behind",         # grass between our line and our own goal
        "marking_coverage",     # tightness of the nearest opp forward on the carrier
        "runner_danger",        # opp forwards moving toward OUR goal behind our shape
        "box_threat",           # opp attackers inside ~25 m of our goal
        "ball_own_third",       # ball in the defensive third
        "aerial_boundary",      # ball near goal while opp forwards camp the box
        # Build-up geometry (the two sensors missing for emergent corridor behavior):
        "same_side_lane_open",  # openness of the best fwd teammate on the CB's OWN pitch half
                                # (high = same-side forward option available; network learns
                                #  to prefer this lane when here and open, WITHOUT a rule)
        "far_side_gap",         # lateral width to the best far-side teammate
                                # (high = switch target wide open; network learns to switch
                                #  when this is high and same_side is congested, WITHOUT a rule)
    ],
    # ── 2026-10-20: `pass_lane_open` appended to FB and WING ────────────────
    # The plan's item 2 was "add the two missing forward-option inputs".  The
    # survey says the gap is narrower and more specific than that: `pass_lane_open`
    # (openness of the best forward teammate) has always been in the DM/CM/AM
    # menu, and `same_side_lane_open` covers the CB, so **FB and WING were the
    # only two families that could not see whether a forward pass option was
    # open at all.**  Measured on the menus: `overlap_option`, which reads as if
    # it fills that gap, actually computes
    # `clamp(distance_to_nearest_forward_teammate / 30)` -- a DISTANCE sensor,
    # not an openness one.  Worse, it returns 0.0 both when NOBODY is ahead and
    # when a teammate is standing on top of you: two opposite facts, one value.
    # So a full-back or winger in the attacking third had no input telling him a
    # forward option existed, which is precisely the blindness AGENTS.md item 5
    # describes.
    #
    # APPENDED, not inserted: the width-only legacy path takes "the first N", so
    # appending leaves every shipped brain's numbers untouched (it warns once
    # that the new input is inert for it -- see `align_role_tail`).  Inserting
    # would silently relabel every feature after the insertion point.
    #
    # ST is deliberately NOT given it: a striker's forward option is usually
    # nobody, so the sensor would be near-constant 0.0 -- an input that costs
    # training budget and carries no information.
    "FB": [
        "wide_marker_dist",     # nearest opp wide attacker ahead (the fullback's man)
        "one_v_one_space",      # openness vs that marker
        "inside_channel",       # lane openness of the diagonal-inside option
        "overlap_option",       # DISTANCE to the nearest forward teammate (misnamed; see above)
        "cross_zone",           # in the wide crossing corridor
        "own_line_height",      # our defensive line height (shared read)
        "short_cut",            # reset option out the back (nearest teammate)
        "pass_lane_open",       # openness of the BEST forward teammate (see _forward_options)
    ],
    "DM": [
        "pressure",             # 1 - nearest defender dist (high = pressed)
        "pass_lane_open",       # best forward teammate lane openness
        "progression",          # best forward option (progress x openness)
        "support_near",         # teammates inside 12 m
        "central_space",        # opp density in the central channel ahead
        "through_lane",         # depth spread of the opp line (loose = channel)
        "recycle_open",         # clean back option
    ],
    "CM": [
        "pressure",
        "pass_lane_open",
        "progression",
        "support_near",
        "forward_space",        # room to carry before the nearest defender ahead
        "scan_ahead",           # teammates ahead (plays/options upfield)
        "recycle_open",
    ],
    "AM": [
        "pressure",
        "pass_lane_open",
        "progression",
        "support_near",
        "space_central",        # lane openness of the central point ahead
        "shoot_opportunity",    # shooting angle vs a penalty-spot reference
        "scoring_zone",         # inside ~25 m of the opponent goal
        "recycle_open",
    ],
    "WING": [
        "marker_dist",          # nearest opp defender on the flank corridor ahead
        "one_v_one",            # openness vs that fullback
        "inside_channel",       # lane openness of the cut-inside diagonal
        "overlap_option",       # DISTANCE to the nearest forward teammate (misnamed; see FB note)
        "cross_zone",           # in the crossing corridor
        "cross_target_open",    # lane openness of the penalty-spot target zone
        "box_arrival",          # teammates arriving in the opponent box
        "cover_behind",         # 1 - proximity of cover behind (high = stranded)
        "pass_lane_open",       # openness of the BEST forward teammate (see _forward_options)
    ],
    "ST": [
        "cb_room",              # room to the nearest CB (high = unmarked)
        "offside_line_gap",     # grass between their line and their goal (in-behind channel)
        "def_gaps",             # depth spread of the opp line (loose = gaps)
        "run_support",          # teammates running beyond the line
        "shoot_angle",          # shooting angle vs penalty-spot reference
        "shoot_pressure",       # shooting angle devalued by a tight marker
        "rebound_proxy",        # close-range second-ball window
        "in_box",               # inside the penalty area
    ],
}

# Role family per in-game position (digits stripped by _role_family).
ROLE_FAMILY: Dict[str, str] = {
    "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "CDM": "DM",
    "CM": "CM", "CAM": "AM", "LW": "WING", "RW": "WING",
    "ST": "ST", "CF": "ST",
}

#: The ACTIONS each role performs — the on-ball vocabulary of a role.
#:
#: This is a hand-written FOOTBALL table, and it is hand-written on purpose.
#: It sits next to `MENU_NAMES` (the per-role SENSOR vocabulary), which is
#: the same kind of object: "what does this role perceive / do".
#:
#: WHY IT IS NEEDED, MEASURED (2026-10-20).  The surrogate's role-scoped
#: pooling derives each family's vocabulary from the DATA, and that is the
#: right default — but the data has a hole exactly where it matters:
#:
#:     * the CM family performed NO through-ball in 4,011 real decisions, and
#:       neither the heuristic CM nor the shipped T1 CM brain ever chooses
#:       THROUGH_BALL (0% over 2,000 synthetic states).  A data-derived
#:       vocabulary therefore says a midfielder cannot play a through-ball —
#:       which is T2's documented collapsed optimum ("CM abandons
#:       through-balls") reproduced by a different route.
#:     * the ST family performed SHOOT exactly ONCE, so its shooting value
#:       rests on a single sample shrunk to a bucket mean, and the striker's
#:       intent mix ends up decided by fourth-decimal differences.
#:
#: So the vocabulary has to say what a role CAN do, not only what it
#: happened to do.  Where the data is solid this table AGREES with it, and
#: where the data is silent it supplies the football truth the corpus is
#: missing.  The two are compared in AGENTS.md.
#:
#: The table is also what the collector explores: an ε-exploration policy
#: draws from it, so the corpus eventually covers every action the role can
#: actually perform.
ROLE_INTENTS: Dict[str, frozenset] = {
    # a keeper distributes and carries; he does not cross or shoot
    "GK": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "THROUGH_BALL"}),
    # a centre-back plays out, breaks a line and can carry; he does not cross
    "CB": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "THROUGH_BALL"}),
    # a full-back crosses and plays the through ball; he does not shoot
    "FB": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "CROSS", "THROUGH_BALL"}),
    # a holding midfielder recycles, carries and plays the long ball
    "DM": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "THROUGH_BALL"}),
    # a central midfielder: the same, and the through ball is his signature
    "CM": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "THROUGH_BALL"}),
    # an attacking midfielder: the same, plus he shoots when he gets there
    "AM": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "THROUGH_BALL", "SHOOT"}),
    # a winger crosses and cuts inside to shoot
    "WING": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                       "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                       "CROSS", "SHOOT"}),
    # a striker shoots, runs the channel and plays the through ball
    "ST": frozenset({"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                     "CARRY", "DRIBBLE", "SWITCH", "PROGRESSIVE_PASS",
                     "THROUGH_BALL", "SHOOT"}),
}

#: Actions that belong to particular ROLES.  Everything else — a safe pass, a
#: recycle, a carry, a switch, a through ball — is GENERIC, and any role may
#: perform it.  Only two actions are role-specific, and they are the two the
#: data showed being mis-credited:
#:
#:   * CROSS belongs to the wide roles.  The ST family performed it ZERO
#:     times in 4,011 real decisions, so every crossing value the striker
#:     could read was the WINGERS' — which is how the GA evolved a striker
#:     who crossed 43% of the time and never shot.
#:   * SHOOT belongs to the forward roles.  The CB and GK tables carry stray
#:     SHOOT samples (the heuristic's noise), not a role truth.
#:
#: It is deliberately NOT a full per-role table: a table that forbade the
#: keeper's progressive pass, for instance, would teach the keeper to hoard
#: the ball, and a wrong table is worse than no table.
ROLE_ONLY_INTENTS: Dict[str, frozenset] = {
    "CROSS": frozenset({"WING", "FB"}),
    "SHOOT": frozenset({"ST", "AM", "WING"}),
}

V2_ROLE_D: Dict[str, int] = {fam: len(names) for fam, names in MENU_NAMES.items()}
V2_INPUT_D: Dict[str, int] = {fam: V2_SHARED_D + V2_ROLE_D[fam] for fam in MENU_NAMES}


def _role_family(position: str) -> Optional[str]:
    return ROLE_FAMILY.get(position.strip().rstrip("0123456789"))


# ─────────────────────────────────────────────────────────────
# GEOMETRY CONTEXT
# ─────────────────────────────────────────────────────────────

_OPP_FWD = ("ST", "CF", "CAM", "LW", "RW")   # forward/wide line = "opp forwards"
_OUR_BACKLINE = ("CB", "LB", "RB", "CDM")    # our defensive shape for GK/CB reads
_OPP_WIDE = ("LW", "RW")


def _xs(attacks_right: bool) -> tuple[float, float]:
    """(opp_goal_x, own_goal_x) in the carrier frame."""
    return (105.0 if attacks_right else 0.0), (0.0 if attacks_right else 105.0)


def _d_goal(px: float, py: float, goal_x: float) -> float:
    return math.hypot(goal_x - px, 34.0 - py)


def _prog(px: float, x: float, attacks_right: bool) -> float:
    """Forward progress of px relative to the carrier (toward opp goal)."""
    return (px - x) if attacks_right else (x - px)


def _near_def(x: float, y: float, defenders: List[Any],
              position_engine: Any, radius: float = 999.0,
              include: Optional[set] = None) -> Optional[float]:
    best = None
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        if include is not None and getattr(d, "position", "") not in include:
            continue
        dx, dy = _pos(position_engine, d.name, (x + 10, y))
        dist = math.hypot(dx - x, dy - y)
        if dist <= radius and (best is None or dist < best):
            best = dist
    return best


def _openness(tx: float, ty: float, defenders: List[Any],
              position_engine: Any) -> float:
    """Lane openness at (tx,ty): same normalisation as v1 features 8/9."""
    nd = _near_def(tx, ty, defenders, position_engine)
    if nd is None:
        return 1.0
    return _clamp((nd - 1.5) / 8.5)


def _teammates(teammates: List[Any]) -> List[Any]:
    return [t for t in (teammates or []) if getattr(t, "position", "") != "GK"]


def _shoot_angle(goal_dist: float) -> float:
    """Shooting angle vs a penalty-spot reference; 1.0 at 11 m, clipped [0,1]."""
    if goal_dist <= 1e-6:
        return 1.0
    ref = 2.0 * math.atan(3.66 / 11.0)
    if goal_dist >= 3.66:
        ang = 2.0 * math.atan(3.66 / goal_dist)
    else:
        ang = math.pi
    return _clamp(ang / ref)


def _opp_line_x(defenders: List[Any], position_engine: Any,
                x: float, y: float) -> List[float]:
    """Sorted x-coords of opp outfield defenders (ascending pitch coords)."""
    return sorted(_pos(position_engine, d.name, (x, y))[0]
                  for d in defenders or []
                  if getattr(d, "position", "") != "GK")


def _forward_options(x: float, y: float, tms_n: List[Any], defs: List[Any],
                     position_engine: Any, attacks_right: bool):
    """Teammates ahead of the carrier, and the openness of the best one.

    ONE definition, because `pass_lane_open` is now a feature of four families
    (DM / CM / AM always had it; FB and WING gained it 2026-10-20) and a second
    copy would drift the moment either side is edited.  A duplicated feature
    definition is worse than a duplicated constant: the two copies look
    identical, the brains trained on the old one keep working, and the
    disagreement only shows up as an input nobody can explain.

    "Ahead" is > 4 m of forward progress.  `_openness` is the shared lane
    measure (distance to the nearest opponent, normalised the same way as the
    v1 features 8/9), so this answers exactly "is a forward pass on?".

    Returns ``(fwd_teammates, best_openness)``; ``0.0`` when nobody is ahead,
    which is a fact ("no forward option") rather than a neutral 0.5.
    """
    fwd = [t for t in tms_n
           if _prog(_pos(position_engine, t.name, (x, y))[0], x, attacks_right)
           > 4.0]
    if not fwd:
        return fwd, 0.0
    return fwd, max(_openness(*_pos(position_engine, t.name, (x, y)), defs,
                              position_engine) for t in fwd)


# ─────────────────────────────────────────────────────────────
# ROLE BLOCK EXTRACTORS (each returns the menu's floats in order)
# ─────────────────────────────────────────────────────────────

def _gk_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    _, own_gx = _xs(ar)
    tms_n = _teammates(tms)
    fwds = [d for d in defs if getattr(d, "position", "") in _OPP_FWD]
    wides = [d for d in defs if getattr(d, "position", "") in _OPP_WIDE]

    attackers_ahead = _clamp(sum(1 for d in fwds
                                 if _prog(_pos(pe, d.name, (x, y))[0], x, ar) > 10) / 3.0)
    near_fwd = _near_def(x, y, fwds, pe) if fwds else None
    attackers_close = 1.0 - _clamp(near_fwd / 20.0) if near_fwd is not None else 0.0

    shape_dists = [_d_goal(*_pos(pe, t.name, (x, y)), own_gx)
                   for t in tms_n if getattr(t, "position", "") in _OUR_BACKLINE]
    defensive_shape = 1.0 - _clamp((sum(shape_dists) / len(shape_dists)) / 70.0) \
        if shape_dists else 0.0

    crossing_threat = _clamp(sum(1 for d in wides
                                 if _d_goal(*_pos(pe, d.name, (x, y)), own_gx) < 30.0) / 2.0)

    roll_x = x + (18 if ar else -18)           # rolling-ground target ahead of GK
    pass_open = _openness(roll_x, 34.0, defs, pe)

    near_tm = _near_def(x, y, tms_n, pe)
    short_safety = 1.0 - _clamp(near_tm / 15.0) if near_tm is not None else 0.0

    ball_near_box = 1.0 - _clamp(_d_goal(x, y, own_gx) / 30.0)
    near_fwd_to_ball = _near_def(x, y, fwds, pe) if fwds else None
    ball_pressed = 1.0 - _clamp(near_fwd_to_ball / 10.0) if near_fwd_to_ball is not None else 0.0
    shot_threat = ball_near_box * ball_pressed

    return [attackers_ahead, attackers_close, defensive_shape, crossing_threat,
            pass_open, short_safety, shot_threat]


def _cb_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    _, own_gx = _xs(ar)
    tms_n = _teammates(tms)
    fwds = [d for d in defs if getattr(d, "position", "") in _OPP_FWD]
    back = [t for t in tms_n if getattr(t, "position", "") in _OUR_BACKLINE]

    back_xs = [_pos(pe, t.name, (x, y))[0] for t in back]
    deepest = max(back_xs) if back_xs else x  # line sits at the deepest teammate
    own_line_height = _clamp(deepest / 105.0)
    space_behind = _clamp(abs(deepest - own_gx) / 45.0)

    near_fwd = _near_def(x, y, fwds, pe) if fwds else None
    marking_coverage = 1.0 - _clamp(near_fwd / 15.0) if near_fwd is not None else 0.0

    ours = _d_goal(deepest, 34.0, own_gx)
    runners = sum(1 for d in fwds
                  if _d_goal(*_pos(pe, d.name, (x, y)), own_gx) < ours)
    runner_danger = _clamp(runners / 2.0)

    near_box = sum(1 for d in fwds
                   if _d_goal(*_pos(pe, d.name, (x, y)), own_gx) < 25.0)
    box_threat = _clamp(near_box / 3.0)

    ball_own_third = 1.0 if _d_goal(x, y, own_gx) < 35.0 else 0.0

    aerial_boundary = 0.0
    if _d_goal(x, y, own_gx) < 30.0:
        aerial_boundary = _clamp(sum(1 for d in fwds
                                     if _d_goal(*_pos(pe, d.name, (x, y)), own_gx) < 25.0) / 2.0)

    # ── Build-up geometry (emergent corridor + switch behavior) ──────────────
    # same_side_lane_open: openness of the best forward teammate on the SAME
    # half of the pitch width as this CB.  RCB at y<34 → looks at low-y fwd
    # options; LCB at y>34 → looks at high-y fwd options.  The network can
    # learn "when same_side_lane_open is high I'm in own half, pass forward
    # into my corridor" — zero hand-coded rule, pure geometry signal.
    PITCH_Y_MID = 34.0
    same_side = y < PITCH_Y_MID  # True = CB sits on right/low half
    same_side_open = 0.0
    far_best_gap = 0.0
    for t in tms_n:
        tx, ty = _pos(pe, t.name, (x, y))
        progress = _prog(tx, x, ar)
        if progress < 4.0:
            continue
        t_same_side = ty < PITCH_Y_MID
        openv = _openness(tx, ty, defs, pe)
        if t_same_side == same_side:
            same_side_open = max(same_side_open, openv)
        else:
            # far_side_gap: how far across the pitch this far-side teammate is
            gap = _clamp(abs(ty - y) / 55.0)  # 55 m ≈ full pitch width usable
            far_best_gap = max(far_best_gap, gap * openv)

    return [own_line_height, space_behind, marking_coverage, runner_danger,
            box_threat, ball_own_third, aerial_boundary,
            same_side_open, far_best_gap]


def _fb_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    _, own_gx = _xs(ar)
    tms_n = _teammates(tms)
    wides = [d for d in defs if getattr(d, "position", "") in _OPP_WIDE]

    markers = [d for d in wides if _prog(_pos(pe, d.name, (x, y))[0], x, ar) > -8.0]
    mdist = _near_def(x, y, markers, pe) if markers else None
    if mdist is not None:
        wide_marker_dist = _clamp(mdist / 15.0)
        one_v_one_space = _clamp((mdist - 1.5) / 8.5)
    else:
        wide_marker_dist, one_v_one_space = 1.0, 1.0

    in_px = x + (16 if ar else -16)
    in_py = y + (6 if y < 34 else -6)
    inside_channel = _openness(in_px, in_py, defs, pe)

    ahead = [t for t in tms_n
             if 5.0 < _prog(_pos(pe, t.name, (x, y))[0], x, ar) < 40.0]
    ov_min = _near_def(x, y, ahead, pe) if ahead else None
    overlap_option = _clamp(ov_min / 30.0) if ov_min is not None else 0.0

    cross_zone = 1.0 if ((x > 80.0) if ar else (x < 25.0)) else 0.0

    back_xs = [_pos(pe, t.name, (x, y))[0] for t in tms_n
               if getattr(t, "position", "") in _OUR_BACKLINE]
    deepest = max(back_xs) if back_xs else x
    own_line_height = _clamp(deepest / 105.0)

    near_tm = _near_def(x, y, tms_n, pe)
    short_cut = _clamp(near_tm / 40.0) if near_tm is not None else 0.0

    _, pass_lane_open = _forward_options(x, y, tms_n, defs, pe, ar)

    return [wide_marker_dist, one_v_one_space, inside_channel, overlap_option,
            cross_zone, own_line_height, short_cut, pass_lane_open]


def _dm_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    return _mid_block(p, x, y, tms, defs, pe, ar,
                      tail=("central_space", "through_lane"))


def _cm_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    return _mid_block(p, x, y, tms, defs, pe, ar,
                      tail=("forward_space", "scan_ahead"))


def _mid_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
               pe: Any, ar: bool, tail: tuple) -> List[float]:
    """Shared midfield options; `tail` picks the role-specific middle, then
    recycle_open closes every midfield menu."""
    tms_n = _teammates(tms)
    pressure = _clamp(_defenders_within(x, y, defs, pe, 6.0) / 3.0)

    fwd_tms, pass_lane_open = _forward_options(x, y, tms_n, defs, pe, ar)

    prog_max = 0.0
    for t in fwd_tms:
        tx, ty = _pos(pe, t.name, (x, y))
        progress = _clamp(_prog(tx, x, ar) / 35.0)
        openv = _openness(tx, ty, defs, pe)
        prog_max = max(prog_max, _clamp(progress * 0.55 + openv * 0.45))
    progression = _clamp(prog_max / 0.7)

    support = 0
    for t in tms_n:
        tx, ty = _pos(pe, t.name, (x, y))
        if math.hypot(tx - x, ty - y) < 12.0:
            support += 1
    support_near = _clamp(support / 4.0)

    mid: List[float] = []
    if "central_space" in tail:
        cent = [d for d in defs
                if _prog(_pos(pe, d.name, (x, y))[0], x, ar) > 10.0
                and abs(_pos(pe, d.name, (x, y))[1] - 34.0) < 12.0]
        mid.append(1.0 - _clamp(len(cent) / 2.0))
    if "through_lane" in tail:
        fx = _opp_line_x(defs, pe, x, y)
        spread = (fx[-1] - fx[-2]) if len(fx) >= 2 else 0.0
        mid.append(_clamp(spread / 25.0))
    if "forward_space" in tail:
        fs_min = None
        for d in defs:
            if getattr(d, "position", "") == "GK":
                continue
            dx, dy = _pos(pe, d.name, (x, y))
            if _prog(dx, x, ar) > 2.0:
                fs_min = min(fs_min, math.hypot(dx - x, dy - y)) \
                    if fs_min is not None else math.hypot(dx - x, dy - y)
        mid.append(_clamp(fs_min / 25.0) if fs_min is not None else 1.0)
    if "scan_ahead" in tail:
        mid.append(_clamp(len(fwd_tms) / 6.0))

    back_tms = [t for t in tms_n if _prog(_pos(pe, t.name, (x, y))[0], x, ar) < -4.0]
    if back_tms:
        def _bdist(t: Any) -> float:
            bx, by = _pos(pe, t.name, (x, y))
            return math.hypot(bx - x, by - y)
        nearest = min(back_tms, key=_bdist)
        recycle_open = _openness(*_pos(pe, nearest.name, (x, y)), defs, pe)
    else:
        recycle_open = 0.5  # no back option -> neutral, not inviting/blocking

    return [pressure, pass_lane_open, progression, support_near] + mid + [recycle_open]


def _am_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    base = _mid_block(p, x, y, tms, defs, pe, ar, tail=())
    opp_gx, _ = _xs(ar)
    sx, sy = x + (18 if ar else -18), 34.0
    space_central = _openness(sx, sy, defs, pe)
    gd = math.hypot(opp_gx - x, 34.0 - y)
    shoot_opportunity = _shoot_angle(gd)
    scoring_zone = 1.0 if gd < 25.0 else 0.0
    # base = [pressure, pass_lane_open, progression, support_near, recycle_open]
    return [base[0], base[1], base[2], base[3], space_central,
            shoot_opportunity, scoring_zone, base[-1]]


def _wing_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
                pe: Any, ar: bool) -> List[float]:
    opp_gx, _ = _xs(ar)
    tms_n = _teammates(tms)

    corridor = [d for d in defs
                if _prog(_pos(pe, d.name, (x, y))[0], x, ar) > -5.0
                and abs(_pos(pe, d.name, (x, y))[1] - y) < 22.0
                and getattr(d, "position", "") != "GK"]
    mdist = _near_def(x, y, corridor, pe) if corridor else None
    if mdist is not None:
        marker_dist = _clamp(mdist / 15.0)
        one_v_one = _clamp((mdist - 1.5) / 8.5)
    else:
        marker_dist, one_v_one = 1.0, 1.0

    ix, iy = x + (16 if ar else -16), y + (8 if y < 34 else -8)
    inside_channel = _openness(ix, iy, defs, pe)

    ahead = [t for t in tms_n
             if 5.0 < _prog(_pos(pe, t.name, (x, y))[0], x, ar) < 40.0]
    ov_min = _near_def(x, y, ahead, pe) if ahead else None
    overlap_option = _clamp(ov_min / 30.0) if ov_min is not None else 0.0

    cross_zone = 1.0 if ((x > 80.0) if ar else (x < 25.0)) else 0.0

    target_x = opp_gx + (-11.0 if ar else 11.0)
    cross_target_open = _openness(target_x, 34.0, defs, pe)

    in_box_tms = sum(1 for t in tms_n
                     if _d_goal(*_pos(pe, t.name, (x, y)), opp_gx) < 22.0)
    box_arrival = _clamp(in_box_tms / 2.0)

    behind = [t for t in tms_n if _prog(_pos(pe, t.name, (x, y))[0], x, ar) < 5.0]
    cover_min = _near_def(x, y, behind, pe) if behind else None
    cover_behind = 1.0 - _clamp(cover_min / 15.0) if cover_min is not None else 1.0

    _, pass_lane_open = _forward_options(x, y, tms_n, defs, pe, ar)

    return [marker_dist, one_v_one, inside_channel, overlap_option, cross_zone,
            cross_target_open, box_arrival, cover_behind, pass_lane_open]


def _st_block(p: Any, x: float, y: float, tms: List[Any], defs: List[Any],
              pe: Any, ar: bool) -> List[float]:
    opp_gx, _ = _xs(ar)
    tms_n = _teammates(tms)
    fx = _opp_line_x(defs, pe, x, y)

    cage = _clamp(_defenders_within(x, y, defs, pe, 8.0) / 4.0)
    cb_room = 1.0 - cage

    line_x = fx[-1] if fx else x
    offside_line_gap = _clamp(abs(opp_gx - line_x) / 30.0)
    if (ar and x > line_x) or (not ar and x < line_x):
        offside_line_gap = 0.0  # carrier already beyond their line -> no channel

    def_gaps = _clamp((fx[-1] - fx[-2]) / 25.0) if len(fx) >= 2 else 0.0

    run_support = _clamp(sum(1 for t in tms_n
                             if _prog(_pos(pe, t.name, (x, y))[0], x, ar) > 20.0) / 2.0)

    gd = math.hypot(opp_gx - x, 34.0 - y)
    shoot_angle = _shoot_angle(gd)
    shoot_pressure = shoot_angle * (1.0 - cage)

    rebound_chance = _clamp((25.0 - gd) / 25.0)
    rebound_proxy = rebound_chance * (1.0 - cage)
    in_box = 1.0 if gd < 22.0 else 0.0

    return [cb_room, offside_line_gap, def_gaps, run_support, shoot_angle,
            shoot_pressure, rebound_proxy, in_box]


_BLOCK_BY_FAMILY: Dict[str, Any] = {
    "GK": _gk_block, "CB": _cb_block, "FB": _fb_block, "DM": _dm_block,
    "CM": _cm_block, "AM": _am_block, "WING": _wing_block, "ST": _st_block,
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def role_block(
    player: Any,
    x: float,
    y: float,
    teammates: List[Any],
    defenders: List[Any],
    position_engine: Any,
    attacks_right: bool,
) -> Optional[List[float]]:
    """v2 role feature block for this player's position (order = MENU_NAMES).

    Returns None when the position has no role family (unknown roles keep
    the v1 24-d-only schema).  Pure geometry, deterministic, [0,1] floats.
    """
    fam = _role_family(getattr(player, "position", ""))
    if fam is None:
        return None
    block = _BLOCK_BY_FAMILY[fam](player, x, y, teammates or [], defenders or [],
                                  position_engine, attacks_right)
    assert len(block) == V2_ROLE_D[fam], f"{fam} block wrong cardinality"
    return block


def build_v2_vector(shared: Any, role: Optional[List[float]]) -> np.ndarray:
    """Concatenate the shared v1 vector with a role block (if any)."""
    base = np.asarray(shared, dtype=np.float64)
    if role is None:
        return base
    return np.concatenate([base, np.asarray(role, dtype=np.float64)])


# ─────────────────────────────────────────────────────────────
# SCHEMA-BY-NAME ALIGNMENT
# ─────────────────────────────────────────────────────────────

class RoleSchemaError(Exception):
    """A trained brain cannot be fed the role features it declares.

    Raised instead of silently truncating.  The failure mode this exists to
    prevent is real and was live in this repo: ``brains/CB.json`` was trained
    on 24 + 7 inputs while ``MENU_NAMES["CB"]`` had grown to 9 names.  The
    loader quietly dropped the two new sensors, so two trained inputs were
    inert and NOTHING anywhere reported it.  A schema change that invalidates
    every trained brain must be loud.
    """


_ALIGN_WARNED: set = set()


def reset_alignment_warnings() -> None:
    """Clear the once-per-(family,width) legacy warning memo (tests)."""
    _ALIGN_WARNED.clear()


def align_role_tail(
    family: Optional[str],
    role: Optional[List[float]],
    tail_width: int,
    want_names: Optional[Sequence[str]] = None,
    brain_label: str = "",
) -> np.ndarray:
    """Return the role tail laid out exactly as the BRAIN was trained on.

    The role block is produced in ``MENU_NAMES[family]`` order, which is the
    CURRENT schema.  A serialized brain was trained on whatever the schema was
    on its training date.  Selecting by NAME — rather than by width — is what
    makes adding a feature safe: a brain that never saw a feature simply does
    not ask for it, instead of receiving it by accident or losing one it did
    ask for.

    ``want_names``
        The names the brain declares (``meta.role_features``).  When given,
        every name must still be emitted by this build or
        :class:`RoleSchemaError` is raised; the values are returned in the
        DECLARED order.  A declared list whose length disagrees with the
        network's own input width is also an error.

    ``want_names is None`` (legacy brains, no declaration)
        Falls back to the historical width-only behaviour: the FIRST
        ``tail_width`` values, zero-padded or truncated to fit.  This
        reproduces the shipped ``brains/`` behaviour byte-for-byte, but it
        warns ONCE per (family, width) so a silent schema drift cannot happen
        again.  Re-save the brain with a name list to silence it.
    """
    if tail_width <= 0:
        return np.zeros(0, dtype=np.float64)
    cur = list(MENU_NAMES.get(family or "", ()))

    if want_names is not None:
        want = list(want_names)
        if family is None or not cur:
            raise RoleSchemaError(
                f"{brain_label or 'brain'} declares role features "
                f"{want[:3]}... but its position has no role family, so this "
                "build cannot produce them."
            )
        if len(want) != tail_width:
            raise RoleSchemaError(
                f"{brain_label or 'brain'} declares {len(want)} role features "
                f"but its architecture reserves {tail_width} inputs for them."
            )
        missing = [n for n in want if n not in cur]
        if missing:
            raise RoleSchemaError(
                f"{brain_label or 'brain'} was trained on role feature(s) "
                f"{missing} that {family} no longer emits. Known: {cur}. "
                "Removing or renaming a role feature is a breaking schema "
                "change; add the feature back, or retrain the brain."
            )
        idx = [cur.index(n) for n in want]
        src = np.zeros(len(cur), dtype=np.float64) if role is None \
            else np.asarray(role, dtype=np.float64)
        return src[idx]

    # Legacy: width-only.  Preserve today's numbers exactly, but say so.
    if role is None:
        return np.zeros(tail_width, dtype=np.float64)
    src = np.asarray(role, dtype=np.float64)
    if src.shape[0] == tail_width:
        return src
    key = (family, int(tail_width))
    if key not in _ALIGN_WARNED:
        _ALIGN_WARNED.add(key)
        warnings.warn(
            f"{brain_label or 'brain'} ({family}) has no declared role "
            f"features and expects {tail_width}; the current {family} schema "
            f"emits {src.shape[0]}. Falling back to the FIRST {tail_width} by "
            "width, which is the legacy behaviour. "
            + ("The extra features are INERT for this brain."
               if src.shape[0] > tail_width else
               "The missing inputs are zero-padded.")
            + " Retrain (or re-save with meta.role_features) to fix.",
            RuntimeWarning, stacklevel=2,
        )
    out = np.zeros(tail_width, dtype=np.float64)
    n = min(src.shape[0], tail_width)
    out[:n] = src[:n]
    return out