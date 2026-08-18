"""
PLOFA 26/27 — HIGH-DENSITY ATTACKING MATRIX (Checkpoint 10)
===========================================================
Each ball carrier evaluates the pitch as a dynamic spatial network:

    shooting window   -> shot_score  (distance decay × pressure tax × angle × finishing)
    passing corridors -> lane_clearance (defender blocking along the carrier→target segment)
    teammate value    -> strategic_value (lane × [0.45·progress + 0.35·freedom + 0.20·depth])

and resolves one of:
    SHOOT            — take the shot (hands off to AttackChain's existing pipeline)
    KEY_PASS         — play the killer ball (counter through-ball)
    PROGRESSIVE_PASS — advance the play cleanly (build-up / open progress)
    RECYCLE_PASS     — keep the ball (no good option)

The decision cascade is steered by the team's tactical scenario:

    counter   (fluid_counter, route_one)       -> hunt the far runner
    low_block (park_the_bus, ultra_defensive, defensive) -> panic shot from range
    build_up  (tiki_taka, structured_possession, vertical_tiki_taka, possession) -> clean progression
    balanced  (everything else)

Consumption rules
-----------------
- The matrix CONSUMES PositionEngine state; it never mutates it.
- No position engine wired in => the matrix returns a fallback decision
  (RECYCLE_PASS with target None) so existing receiver/destination logic
  runs byte-for-byte unchanged.
- All geometry is direction-aware: a team attacking left (away side) has
  its goal at x=0, so every distance/angle/depth term is mirrored.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Any

from position_engine import ELLIPSE_COMPOSE_FLOOR

# ─────────────────────────────────────────────
# PITCH GEOMETRY CONSTANTS
# ─────────────────────────────────────────────

GOAL_CENTER_Y = 34.0
POST_OFFSET_Y = 3.66          # 7.32m goal mouth -> posts at 34 ± 3.66
GOAL_X_ATTACKING = 105.0      # attacking right (home)
GOAL_X_DEFENDING = 0.0        # attacking left (away)

# Reference: a shooter standing dead central on the penalty spot line sees
# the goal mouth under 2·atan(3.66/11.0). This normalises the angle term.
REFERENCE_ANGLE = 2.0 * math.atan2(3.66, 11.0)

# Lane blocking ramp: a defender ≤1.2m off the passing lane blocks it,
# ≥3.0m clears it, linear between.
LANE_BLOCK_DIST = 1.2
LANE_CLEAR_DIST = 3.0

# Pressure: a defender within 1.5m of the carrier applies pressure.
PRESSURE_DIST = 1.5

# Network split: teammates < 15m are "close" (recycle/progress), ≥ 15m "far".
CLOSE_NETWORK_M = 15.0

# The keeper release valve is a SHORT SAFE RESET: it only exists for a keeper
# that is actually a short back-pass away. Mirror the phase engine's
# `_reachable` cap (45m) so a carrier in the opponent's third cannot ping a
# 90m diagonal all the way back to his own box — that is not a release, it is
# a surrendered turnover.
GK_RELEASE_MAX_DIST = 45.0


def _goal_x(attacks_right: bool) -> float:
    return GOAL_X_ATTACKING if attacks_right else GOAL_X_DEFENDING


def _attacking_third(x: float, attacks_right: bool) -> bool:
    return x > 70 if attacks_right else x < 35

# Reference to satisfy static analysis that this helper is intentionally available
# for external callers or reflective use elsewhere.
_ = _attacking_third


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _shooter_quality(dna: Optional[Any]) -> float:
    """0..1 — the better of the player's close and long shooting DNA."""
    technical = getattr(dna, "technical", None)
    if technical is None:
        return 0.5
    fin = getattr(technical, "finishing", 50.0)
    ls = getattr(technical, "long_shots", 45.0)
    if fin is None:
        fin = 50.0
    if ls is None:
        ls = 45.0
    return _clamp01(max(fin, ls) / 100.0)


# ─────────────────────────────────────────────
# GEOMETRY PRIMITIVES
# ─────────────────────────────────────────────

def nearest_defender_dist(
    x: float, y: float,
    defenders: Optional[List[Any]], position_engine: Any,
) -> Optional[float]:
    """
    Distance from (x, y) to the nearest outfield defender.
    Returns None when there is no spatial information (no engine / no outfield
    defenders) — callers treat None as "no pressure / no blocking".
    """
    if position_engine is None or not defenders:
        return None
    best = None
    for d in defenders:
        if getattr(d, "position", None) == "GK":
            continue
        dx, dy = position_engine.get_position(d.name)
        dist = math.hypot(dx - x, dy - y)
        if best is None or dist < best:
            best = dist
    return best


def shooting_angle_degrees(x: float, y: float, attacks_right: bool = True) -> float:
    """
    The shooter's angle off the goal-centre line, mirroring the pitch-level
    `_select_action_from_position` geometry: 0° = dead central, 90° = level
    with the goal line. Delegated to by AttackChain's selector.
    """
    gx = _goal_x(attacks_right)
    dist_from_line = abs(gx - x)
    if dist_from_line <= 0.01:
        return 90.0
    return math.degrees(math.atan2(abs(y - GOAL_CENTER_Y), dist_from_line))


def _angle_multiplier(x: float, y: float, attacks_right: bool) -> float:
    """
    Visible goal-mouth angular width as seen from (x, y), normalised by the
    penalty-spot reference. Central close-range positions dominate; acute
    wide positions are sharply penalised.
    """
    gx = _goal_x(attacks_right)
    d = abs(gx - x)
    if d <= 0.01:
        return 1.0
    top = GOAL_CENTER_Y + POST_OFFSET_Y
    bottom = GOAL_CENTER_Y - POST_OFFSET_Y
    a_top = math.atan2(top - y, d)
    a_bottom = math.atan2(bottom - y, d)
    visible = abs(a_top - a_bottom)
    return max(0.05, min(3.0, visible / REFERENCE_ANGLE))


def shot_score(
    x: float, y: float,
    defenders: Optional[List[Any]], position_engine: Any,
    attacks_right: bool = True,
    dna=None,
) -> float:
    """
    S_viability = distance_decay × pressure_tax × angle_multiplier × finishing.

    distance_decay : 1/(1+(dist/12)^1.8)  — dies fast past ~15m.
    pressure_tax   : 0.30 when a defender is within 1.5m, softened by
                     composure (100 composure ~ 0.965, 0 ~ 0.335); else 1.0.
    angle_multiplier : visible goal mouth / REFERENCE_ANGLE.
    finishing      : 0.60 + 0.55·max(finishing, long_shots)/100.
    """
    gx = _goal_x(attacks_right)
    dist = max(0.5, math.hypot(x - gx, y - GOAL_CENTER_Y))
    distance_decay = 1.0 / (1.0 + (dist / 12.0) ** 1.8)

    nd = nearest_defender_dist(x, y, defenders, position_engine)
    if nd is not None and nd < PRESSURE_DIST:
        comp = 0.5
        mental = getattr(dna, "mental", None)
        if mental is not None:
            comp = _clamp01(getattr(mental, "composure", 50.0) / 100.0)
        pressure_tax = 0.30 + 0.70 * comp
    else:
        pressure_tax = 1.0

    angle_mult = _angle_multiplier(x, y, attacks_right)

    finish_mult = 1.0
    if dna is not None:
        finish_mult = 0.60 + 0.55 * _shooter_quality(dna)

    return distance_decay * pressure_tax * angle_mult * finish_mult


def lane_clearance(
    x: float, y: float, tx: float, ty: float,
    defenders: Optional[List[Any]], position_engine,
) -> float:
    """
    How clear is the corridor from (x, y) to (tx, ty)?
    Uses the minimum perpendicular distance from any outfield defender to the
    carrier→target segment: ≤1.2m => 0 (blocked), ≥3m => 1, linear between.
    No spatial info => 1 (unblocked).
    """
    if position_engine is None or not defenders:
        return 1.0
    seg_len = math.hypot(tx - x, ty - y)
    best = None
    for d in defenders:
        if getattr(d, "position", None) == "GK":
            continue
        dx, dy = position_engine.get_position(d.name)
        if seg_len < 1e-6:
            dist = math.hypot(dx - x, dy - y)
        else:
            t = ((dx - x) * (tx - x) + (dy - y) * (ty - y)) / (seg_len * seg_len)
            t = max(0.0, min(1.0, t))
            px = x + t * (tx - x)
            py = y + t * (ty - y)
            dist = math.hypot(dx - px, dy - py)
        if best is None or dist < best:
            best = dist
    if best is None:
        return 1.0
    if best <= LANE_BLOCK_DIST:
        return 0.0
    if best >= LANE_CLEAR_DIST:
        return 1.0
    return (best - LANE_BLOCK_DIST) / (LANE_CLEAR_DIST - LANE_BLOCK_DIST)


def network_zone(dist: float) -> str:
    """'close' when the teammate is within 15m, 'far' beyond."""
    return "close" if dist < CLOSE_NETWORK_M else "far"


def strategic_value(
    x: float, y: float, tx: float, ty: float,
    defenders: Optional[List], position_engine,
    attacks_right: bool = True,
) -> float:
    """
    V_strategic = lane × (0.45·progress + 0.35·freedom + 0.20·depth)

    progress : how much closer to the goal this pass takes the ball.
    freedom  : 1.0 if no opponent within 3m of the target, else ramps down.
    depth    : how deep into the attacking third the target sits.
    """
    lane = lane_clearance(x, y, tx, ty, defenders, position_engine)
    if lane <= 0.0:
        return 0.0

    freedom = 1.0
    if position_engine is not None and defenders:
        nd = nearest_defender_dist(tx, ty, defenders, position_engine)
        if nd is not None and nd < 3.0:
            freedom = max(0.15, (nd - 1.0) / 2.0)
        else:
            freedom = 1.0

    gx = _goal_x(attacks_right)
    d_ag = math.hypot(x - gx, y - GOAL_CENTER_Y)
    d_tg = math.hypot(tx - gx, ty - GOAL_CENTER_Y)
    progress = _clamp01(0.5 + (d_ag - d_tg) / 20.0)

    depth = _clamp01((tx - 35.0) / 70.0) if attacks_right else _clamp01((35.0 - tx) / 70.0)

    return lane * (0.45 * progress + 0.35 * freedom + 0.20 * depth)


def shot_threshold(dna, scenario: str) -> float:
    """
    Base shooting threshold per scenario, nudged by DNA:
        base           : balanced 0.55, counter 0.50, low_block 0.52, build_up 0.62
        shooting_quality: −0.16·max(finishing,long_shots)/100
        decisions      : −0.06·decisions/100
        plays_safe     : +0.10·plays_safe
        confidence     : −0.05·min(1, confidence/100)
        base nudge     : +0.02

    FIX (scoreline realism): the previous thresholds (counter 0.42,
    low_block 0.45) were far too permissive — a carrier in the final
    third resolved to SHOOT on nearly every touch, inflating shot volume
    and feeding the 8-8 / 10-4 scorelines. The thresholds are raised so
    only genuinely high-viability windows trigger a shot; marginal
    windows recycle through the pass network instead.
    """
    base = {
        "counter": 0.50,
        "low_block": 0.52,
        "build_up": 0.62,
        "balanced": 0.55,
    }.get(scenario, 0.55)

    if dna is None:
        return base

    mental = getattr(dna, "mental", None)
    tendencies = getattr(dna, "tendencies", None)
    form = getattr(dna, "form", None)
    conf = _clamp01(getattr(form, "confidence", 50.0) / 100.0) if form is not None else 0.5
    decisions = _clamp01(getattr(mental, "decisions", 50.0) / 100.0) if mental is not None else 0.5
    plays_safe = getattr(tendencies, "plays_safe", 0.5) if tendencies is not None else 0.5
    plays_safe = max(0.0, min(1.0, plays_safe))

    return base - 0.16 * _shooter_quality(dna) - 0.06 * decisions + 0.10 * plays_safe - 0.05 * conf + 0.02


# ─────────────────────────────────────────────
# SCENARIO
# ─────────────────────────────────────────────

COUNTER_STYLES = {
    "fluid_counter", "route_one",
}
LOW_BLOCK_STYLES = {
    "park_the_bus", "ultra_defensive", "defensive",
}
BUILD_UP_STYLES = {
    "tiki_taka", "structured_possession", "possession", "vertical_tiki_taka",
}


def scenario_for(team_profile) -> str:
    """Map a team's tactical identity to a decision scenario."""
    style = getattr(team_profile, "style", None)
    value = getattr(style, "value", style)
    value = value or ""
    if value in COUNTER_STYLES:
        return "counter"
    if value in LOW_BLOCK_STYLES:
        return "low_block"
    if value in BUILD_UP_STYLES:
        return "build_up"
    return "balanced"


# ─────────────────────────────────────────────
# DECISION MODEL
# ─────────────────────────────────────────────

@dataclass
class AttackingDecision:
    """The carrier's resolved action for one touch."""
    action: str = "RECYCLE_PASS"          # SHOOT / KEY_PASS / PROGRESSIVE_PASS / RECYCLE_PASS
    target: Optional[object] = None       # PlayerProfile
    target_x: float = 0.0
    target_y: float = 0.0
    shot_score: float = 0.0
    scenario: str = "balanced"
    reason: str = ""
    fallback: bool = False
    x: float = 0.0
    y: float = 0.0

    @property
    def is_pass(self) -> bool:
        return self.action in ("KEY_PASS", "PROGRESSIVE_PASS", "RECYCLE_PASS")


@dataclass
class _Option:
    target: object
    lane: float
    progress: float
    freedom: float
    depth: float
    value: float
    dist: float
    zone: str


class AttackingMatrix:
    """
    Per-touch decision engine. Pure: reads PositionEngine state, writes nothing.
    """

    @staticmethod
    def _release_lane_clearance(
        x: float, y: float, tx: float, ty: float,
        defenders: Optional[List], position_engine,
    ) -> float:
        """
        Corridor check for a SHORT SAFE RESET (keeper release / recycle).

        Same perpendicular-distance model as lane_clearance(), except a
        defender whose projection onto the segment lands within 15% of the
        pass origin (i.e. a man PRESSING the carrier, t≈0) is ignored — he
        presses the ball, he does not screen the keeper. Only defenders
        actually between the carrier and the target (0.15 ≤ t ≤ 1.0) block.
        """
        if position_engine is None or not defenders:
            return 1.0
        seg_len = math.hypot(tx - x, ty - y)
        if seg_len < 1e-6:
            return 1.0
        best = None
        for d in defenders:
            if getattr(d, "position", None) == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            t = ((dx - x) * (tx - x) + (dy - y) * (ty - y)) / (seg_len * seg_len)
            if t < 0.15:
                continue
            t = max(0.0, min(1.0, t))
            px = x + t * (tx - x)
            py = y + t * (ty - y)
            dist = math.hypot(dx - px, dy - py)
            if best is None or dist < best:
                best = dist
        if best is None:
            return 1.0
        if best <= LANE_BLOCK_DIST:
            return 0.0
        if best >= LANE_CLEAR_DIST:
            return 1.0
        return (best - LANE_BLOCK_DIST) / (LANE_CLEAR_DIST - LANE_BLOCK_DIST)

    @staticmethod
    def _build_options(
        carrier, teammates, defenders,
        x: float, y: float, position_engine, attacks_right: bool,
        team_profile=None,
    ) -> List[_Option]:
        opts: List[_Option] = []
        for tm in teammates or []:
            if getattr(tm, "position", None) == "GK":
                continue
            if getattr(tm, "name", None) == getattr(carrier, "name", None):
                continue
            tx, ty = position_engine.get_position(tm.name)
            dist = math.hypot(tx - x, ty - y)
            lane = lane_clearance(x, y, tx, ty, defenders, position_engine)

            freedom = 1.0
            if defenders:
                nd = nearest_defender_dist(tx, ty, defenders, position_engine)
                if nd is not None and nd < 3.0:
                    freedom = max(0.15, (nd - 1.0) / 2.0)

            gx = _goal_x(attacks_right)
            d_ag = math.hypot(x - gx, y - GOAL_CENTER_Y)
            d_tg = math.hypot(tx - gx, ty - GOAL_CENTER_Y)
            progress = _clamp01(0.5 + (d_ag - d_tg) / 20.0)
            depth = _clamp01((tx - 35.0) / 70.0) if attacks_right else _clamp01((35.0 - tx) / 70.0)

            # Encourage wide fullback-to-wing combinations when the fullback
            # is genuinely on the flank. This keeps build-up flow on the same
            # side instead of overly centralising a safe/fullback recycle.
            same_flank_bonus = 0.0
            if carrier.position == "LB" and y < 34.0 and tm.position == "LW" and ty < 34.0:
                same_flank_bonus = 0.06
            elif carrier.position == "RB" and y > 34.0 and tm.position == "RW" and ty > 34.0:
                same_flank_bonus = 0.06

            # Checkpoint 18: modern winger flank geometry — modern wingers
            # (Vini Jr, Saka, Salah) are touchline-hugging flank attackers.
            # Their strategic value is highest when they're in the dangerous
            # wide zones — the touchline→byline corridor and the wide box-entry
            # channels. A winger standing on the touchline in the attacking
            # third is a MUCH more valuable receiver than one drifted into
            # midfield traffic.
            winger_flank_bonus = 0.0
            if tm.position in ("LW", "RW") and position_engine is not None:
                winger_profile = position_engine.winger_registry.get(tm.name)
                if winger_profile is not None:
                    danger = winger_profile.danger_zone_score(tx, ty, attacks_right)
                    winger_flank_bonus = danger * 0.12

            # Checkpoint 6.2 — midfield geometric coverage: reward midfielders
            # (CDM/CM/CAM) who occupy half-spaces, modelling Enzo/Rice/Pedri
            # style space occupation. A midfielder in a half-space is a more
            # valuable receiver than one stuck in central traffic.
            midfield_coverage_bonus = 0.0
            if tm.position in ("CDM", "CM", "CAM"):
                if ty < 22.0 or ty > 46.0:
                    midfield_coverage_bonus += 0.06
                between_ball_and_goal = (
                    (x < tx < gx) if attacks_right else (gx < tx < x)
                )
                if between_ball_and_goal:
                    midfield_coverage_bonus += 0.04
                width_factor = max(0.0, (abs(ty - 34.0) - 8.0) / 18.0)
                midfield_coverage_bonus += width_factor * 0.04

            # Checkpoint 18.1 — false nine triangle overload (Barcelona-style).
            # When a midfielder passes to a deep striker, it creates a numerical
            # overload in midfield — the classic false nine pattern. Only active
            # when the team profile explicitly opts in via uses_false_nine.
            false_nine_bonus = 0.0
            if getattr(team_profile, "uses_false_nine", False):
                if carrier.position in ("CM", "CAM", "CDM") and tm.position in ("ST", "CF"):
                    target_state = position_engine.states.get(tm.name) if position_engine is not None else None
                    if target_state is not None:
                        if attacks_right:
                            is_deep = target_state.current_x < 52.5
                            depth_ratio = max(0.0, min(1.0, (52.5 - target_state.current_x) / 25.0))
                        else:
                            is_deep = target_state.current_x > 52.5
                            depth_ratio = max(0.0, min(1.0, (target_state.current_x - 52.5) / 25.0))
                        if is_deep:
                            false_nine_bonus = depth_ratio * 0.12

            # Checkpoint 20 — ball-centric elliptical weighting: the receive
            # pool is an anisotropic ellipse anchored just ahead of the ball
            # along the axis of play. A runner 25m upfield is a live option;
            # a player 25m out to the side is not. Composed on the option's
            # value with a floor so deliberate lateral/backward recycle and
            # half-space support options keep a real chance (a team must be
            # able to go backwards under the press).
            ellipse = position_engine.ball_centric_weight(tm.name, x, y)

            value = lane * (0.45 * progress + 0.35 * freedom + 0.20 * depth + same_flank_bonus + winger_flank_bonus + midfield_coverage_bonus + false_nine_bonus)
            value *= (ELLIPSE_COMPOSE_FLOOR + (1.0 - ELLIPSE_COMPOSE_FLOOR) * ellipse)

            opts.append(_Option(
                target=tm, lane=lane, progress=progress, freedom=freedom,
                depth=depth, value=value, dist=dist,
                zone=network_zone(dist),
            ))
        return opts

    @staticmethod
    def _keeper_release_option(
        carrier, teammates, defenders,
        x: float, y: float, position_engine, attacks_right: bool,
    ) -> Optional[_Option]:
        """
        The press-escape release valve to the unmarked goalkeeper.

        Modern build-up (Guardiola/Arteta) treats the keeper as an unmarked
        eleventh outfield player — a permanent numerical overload against the
        press. The GK is deliberately excluded from `_build_options` (so it
        never competes as a routine receiver), and is surfaced HERE only as a
        deliberate safety valve for when the carrier is bottled up.

        The back-pass is the single safest outlet when forward corridors are
        shut: it resets the attack, drags the pressing front forward (and away
        from the CB/CDM build-up lanes), and buys the back line time to slide
        into space. Only offered when the corridor to the keeper is actually
        open — a keeper with a defender on his release line isn't an outlet.
        """
        gk = next(
            (tm for tm in (teammates or [])
             if getattr(tm, "position", None) == "GK"),
            None,
        )
        if gk is None:
            return None
        gx, gy = position_engine.get_position(gk.name)
        # Reachability gate: the release is only a safe reset when the keeper
        # is genuinely close. A keeper 60m+ away (carrier deep in the opponent
        # third) is NOT an outlet — force the carrier to resolve the touch
        # with his outfield options instead of dumping a doomed 80m lob back.
        if math.hypot(gx - x, gy - y) > GK_RELEASE_MAX_DIST:
            return None
        # Release-corridor check. The generic lane_clearance() would treat the
        # presser ON the carrier (t≈0) as blocking every lane out of the
        # carrier — including the 22m back-pass to an UNMARKED keeper behind
        # the ball — which made the intense-pressure release valve dead code
        # in exactly the scenario it exists for. A man screening the keeper
        # (between the ball and the keeper) still blocks; a man pressing the
        # carrier does not.
        lane = AttackingMatrix._release_lane_clearance(
            x, y, gx, gy, defenders, position_engine)
        if lane < 0.5:
            return None
        return _Option(
            target=gk, lane=lane, progress=0.0, freedom=1.0,
            depth=0.0, value=0.5 + 0.4 * lane,
            dist=math.hypot(gx - x, gy - y), zone="close",
        )

    @classmethod
    def decide(
        cls,
        carrier,
        teammates,
        defenders,
        x: float,
        y: float,
        position_engine=None,
        attacks_right: bool = True,
        scenario: Optional[str] = None,
        team_profile=None,
        under_pressure: bool = False,
        dna=None,
        danger_level: float = 0.0,
    ) -> AttackingDecision:
        """
        Resolve the carrier's action for this touch.

        Returns an AttackingDecision. With no position engine, returns a
        fallback RECYCLE_PASS (target None) so existing selection logic runs
        unchanged.

        Checkpoint 17 — GK/defender threat awareness: the attacking matrix
        now reads the same danger level defenders use. Under HIGH/CRITICAL
        danger the shoot threshold is raised because a defence that is
        already organised and under pressure is harder to break down — a
        marginal shot now recycles instead of forcing a low-percentage
        attempt that inflates scorelines.
        """
        if position_engine is None:
            return AttackingDecision(
                action="RECYCLE_PASS", scenario=scenario or "balanced",
                reason="no_position_engine", fallback=True,
                x=x, y=y,
            )

        if scenario is None:
            scenario = scenario_for(team_profile) if team_profile is not None else "balanced"

        dna = dna if dna is not None else getattr(carrier, "dna", None)
        shot = shot_score(x, y, defenders, position_engine,
                          attacks_right=attacks_right, dna=dna)
        threshold = shot_threshold(dna, scenario)

        # Threat-aware threshold: under HIGH/CRITICAL danger the defence is
        # organised and compact, so marginal shots should recycle.
        if danger_level >= 75.0:
            threshold += 0.10
        elif danger_level >= 55.0:
            threshold += 0.05
        threshold = min(0.85, threshold)
        gx = _goal_x(attacks_right)
        dist_to_goal = math.hypot(x - gx, y - GOAL_CENTER_Y)

        opts = cls._build_options(
            carrier, teammates, defenders, x, y, position_engine, attacks_right,
            team_profile=team_profile,
        )
        close_opts = [o for o in opts if o.zone == "close"]
        far_opts = [o for o in opts if o.zone == "far"]

        close_val = max((o.value for o in close_opts), default=0.0)
        close_prog = max((o.progress for o in close_opts), default=0.0)
        far_val = max((o.value for o in far_opts), default=0.0)
        far_lane = max((o.lane for o in far_opts), default=0.0)
        best_far = max(far_opts, key=lambda o: o.value) if far_opts else None
        best_close = max(close_opts, key=lambda o: o.value) if close_opts else None

        # Pressure / forward-corridor state for the GK back-pass release valve.
        # `intense_pressure` — a defender sits within PRESSURE_DIST (1.5m) of the
        # carrier (the spatial "man on" state), OR the chain already flagged the
        # touch as pressured. `forward_open` — at least one corridor genuinely
        # advances the ball: a clean through/counter lane or a clear progressive
        # close receiver. When intense pressure coincides with NO forward lane,
        # the unmarked keeper is the highest-value, safest outlet.
        intense_pressure = under_pressure
        _nd = nearest_defender_dist(x, y, defenders, position_engine)
        if _nd is not None:
            intense_pressure = intense_pressure or (_nd < PRESSURE_DIST)
        # A corridor is only "forward open" when its LANE is actually clear —
        # a progressive receiver smothered by a goalside defender (lane 0.0)
        # is not an option. Without this gate, a lane-blocked close option's
        # high progress kept the GK press-escape (rule 7) from ever firing.
        best_close_lane = best_close.lane if best_close is not None else 0.0
        forward_open = (far_lane >= 0.55) or (
            close_prog >= 0.60 and best_close_lane >= 0.40
        )

        def _pass(option: Optional[_Option], action: str, reason: str) -> AttackingDecision:
            if option is None:
                return AttackingDecision(
                    action="RECYCLE_PASS", scenario=scenario, reason=reason,
                    fallback=False, x=x, y=y, shot_score=shot,
                )
            tx, ty = position_engine.get_position(option.target.name)
            return AttackingDecision(
                action=action, target=option.target,
                target_x=tx, target_y=ty,
                shot_score=shot, scenario=scenario, reason=reason,
                fallback=False, x=x, y=y,
            )

        def _shoot(reason: str) -> AttackingDecision:
            return AttackingDecision(
                action="SHOOT", shot_score=shot, scenario=scenario,
                reason=reason, fallback=False, x=x, y=y,
            )

        # 1. ELITE SHOT — close range, high viability, no clearly better pass.
        if (dist_to_goal < 20.0 and shot >= max(0.50, threshold)
                and not (far_lane >= 0.55 and far_val >= shot)):
            return _shoot("elite_shot")

        # 2. COUNTER CHANCE — a clean far option into space.
        if scenario == "counter" and far_lane >= 0.55 and far_val >= 0.50 and not under_pressure:
            return _pass(best_far, "KEY_PASS", "counter_through_ball")

        # 3. LOW-BLOCK PANIC — parked-bus sides still have a go from range.
        if scenario == "low_block" and dist_to_goal < 35.0 and shot >= 0.42:
            return _shoot("low_block_panic_shot")

        # 4. CLEAN BUILD-UP — possession sides progress through a clean far lane.
        if scenario == "build_up" and far_lane >= 0.70 and far_val >= 0.55:
            return _pass(best_far, "PROGRESSIVE_PASS", "clean_build_up")

        # 5. PROGRESSIVE — a genuinely forward close option.
        if close_val >= 0.40 and close_prog >= 0.60:
            return _pass(best_close, "PROGRESSIVE_PASS", "progressive_pass")

        # 6. PRESSED PANIC — pressured low-block carrier lets fly from range.
        if under_pressure and scenario == "low_block" and dist_to_goal < 45.0:
            return _shoot("pressed_panic_shot")

        # 7. BACK-PASS RELEASE VALVE — press escape to the goalkeeper.
        # A carrier is "under intense pressure" when a defender sits within
        # PRESSURE_DIST (1.5m). Combined with every forward corridor blocked
        # (no clean progressive/through lane, no viable shot window), the
        # unmarked keeper becomes the highest-value, safest outlet on the
        # pitch instead of forcing an errant forward pass. This is exactly
        # the modern CB/CDM/LB "reset to the keeper" trigger: it lifts the
        # keeper's participation to a real outfield sweep-keeper's level.
        if intense_pressure and not forward_open:
            gk_opt = cls._keeper_release_option(
                carrier, teammates, defenders, x, y, position_engine, attacks_right,
            )
            if gk_opt is not None:
                return _pass(gk_opt, "RECYCLE_PASS", "back_to_keeper_under_pressure")

        # 8. RECYCLE — keep the ball through the best available option.
        target_opt = best_close if best_close is not None else best_far
        if target_opt is None:
            gk_opt = cls._keeper_release_option(
                carrier, teammates, defenders, x, y, position_engine, attacks_right,
            )
            if gk_opt is not None:
                return _pass(gk_opt, "RECYCLE_PASS", "back_to_keeper_recycle")
        return _pass(target_opt, "RECYCLE_PASS", "no_clear_option")
