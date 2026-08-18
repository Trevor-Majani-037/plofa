"""
PLOFA 26/27 — TACTICAL POSSESSION PHASES
=========================================
possession_phases.py

Why this exists:
    Modern positional play (Juego de Posición) treats a team's possession as a
    sequence of DELIBERATE, GEOMETRIC phases. The ball does not fly around
    randomly — it moves according to structural rules, each phase with its own
    spatial sub-goal:

        [Phase 1: Build-Up/Regroup]   ──►  [Phase 2: Midfield Circulation]
        (GK is the Overload Anchor)        (Baiting the Midblock)

        ──►  [Phase 3: Wing Isolation]  ──►  [Phase 4: Box Penetration]
             (Stretching the Backline)      (Through-balls & Crosses)

    Crucially, the machine is NOT a one-way street. When forward routes are
    congested the attacking team must REGRESS the phase — recycle the ball
    backward, ultimately to the goalkeeper — to preserve possession and drag
    the opponent's block out of shape. Without that regression valve a
    goalkeeper can finish a match with zero passes (the ball never returns to
    him, the engine never lets build-up be pressed, and the keeper is reduced
    to a static shot-stopper).

    This module is the pure decision layer for that system: it reads a pitch
    snapshot (ball position, carrier, teammate positions/marking, defender
    presence) and returns a tactical directive + target. It has no dependency
    on the event chain or RNG, mirroring cross_detector.py / threat_engine.py,
    so it is trivially unit-testable. The event chain just executes whatever
    directive comes back.

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines).
Attacking right (home): x grows toward the opponent goal (goal at 105).
Attacking left  (away): x shrinks toward the opponent goal (goal at 0).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# ENUMS — The language of the phase machine
# ─────────────────────────────────────────────

class PossessionPhase(Enum):
    """The four geometric phases of a possession sequence."""

    REGROUP_BUILD_UP        = "regroup_build_up"
    MIDFIELD_CIRCULATION    = "midfield_circulation"
    WING_ISOLATION          = "wing_isolation"
    BOX_PENETRATION         = "box_penetration"


class TacticalDirective(Enum):
    """What the phase engine orders the carrier to do with this touch."""

    PROGRESS               = "progress"                # Advance to the next phase
    SUSTAIN_CIRCULATION    = "sustain_circulation"     # Keep the ball, same phase
    RECYCLE_BACKWARD       = "recycle_backward"        # Tactical reset to a deep anchor
    RELEASE_TO_GK          = "release_to_gk"           # Build-up recycle through the keeper
    EMERGENCY_DROP_TO_GK   = "emergency_drop_to_gk"    # Danger valve — pass to keeper
    WING_SWITCH            = "wing_switch"             # Push wide to stretch the block


# ─────────────────────────────────────────────
# PITCH GEOMETRY CONSTANTS (metres)
# ─────────────────────────────────────────────

PITCH_X: float = 105.0
PITCH_Y: float = 68.0

THIRD_DEPTH: float = 35.0          # one third of the pitch (105/3)
WING_LINE: float = 20.0            # y < WING_LINE or y > PITCH_Y - WING_LINE = wide
FINAL_THIRD_X: float = 70.0        # attacking third starts 35m from opponent goal
BOX_LINE_X_ATTACKING: float = 88.5  # penalty area edge (105 - 16.5)
BOX_LINE_X_DEFENDING: float = 16.5

# Deep roles that are legitimate "reset anchors" — they sit behind the ball
# and can recycle possession or hand it to the keeper.
RESET_ANCHOR_ROLES: Tuple[str, ...] = ("CB", "LB", "RB", "CDM")

# Wide roles used to stretch the opponent backline in Phase 3.
WING_ROLES: Tuple[str, ...] = ("LW", "RW", "LB", "RB")

# Carrier roles allowed to execute an EMERGENCY_DROP_TO_GK. A striker never
# drops all the way to his own keeper with a free midfield lane available.
DEEP_CARRIER_ROLES: Tuple[str, ...] = ("CB", "LB", "RB", "CDM", "CM", "GK")

# Marking tightness beyond which a teammate is considered "shut out" as a
# passing outlet (0 = free, 1 = smothered).
BLOCKED_MARKING: float = 0.62

# Tempo circulation (Checkpoint 23): the support-pass geometry. A circulation
# target must be close enough that the pass is routine, but not standing on
# the carrier's toes.
CIRCULATION_RANGE_M: float = 26.0
CIRCULATION_MIN_RANGE_M: float = 3.0

# How far AHEAD of the ball a teammate may sit and still count as a
# circulation (tempo) target rather than a progressive option — a support
# runner half a step ahead is still a lateral pass.
CIRCULATION_AHEAD_TOLERANCE_M: float = 6.0

# Role appetite for receiving a circulation pass. Midfielders and the back
# line ARE the circulation network; wingers hold width (their ball is the
# WING_SWITCH) and strikers pinning the backline are not tempo outlets.
CIRCULATION_ROLE_WEIGHT: dict = {
    "CDM": 1.35, "CM": 1.35, "CB": 1.05, "LB": 1.00, "RB": 1.00,
    "CAM": 0.70, "LW": 0.35, "RW": 0.35, "ST": 0.15, "CF": 0.15,
}


# ─────────────────────────────────────────────
# SNAPSHOTS — what the engine reads
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class TeammateSnapshot:
    """A teammate the carrier could pass to, with live spatial state."""

    name: str
    role: str
    x: float
    y: float
    marking: float = 0.0     # 0 = completely free, 1 = smothered
    lane_blocked: bool = False  # cover-shadow geometry chokes this corridor


@dataclass(frozen=True)
class GKSnapshot:
    """The team's goalkeeper as a passing option."""

    name: str
    x: float
    y: float
    lane_open: bool = True   # is the corridor to the keeper actually clear?


# ─────────────────────────────────────────────
# PHASE CLASSIFICATION — pure geometry
# ─────────────────────────────────────────────

def possession_phase_for(x: float, y: float, attacks_right: bool = True) -> PossessionPhase:
    """
    Classify which tactical phase the ball is in from its (x, y).

    Normalised x is the ball's distance from the team's OWN goal line along
    its attacking direction, so the classification is direction-agnostic.
    """
    nx = x if attacks_right else PITCH_X - x

    if nx < THIRD_DEPTH:
        return PossessionPhase.REGROUP_BUILD_UP

    is_wide = (y < WING_LINE or y > PITCH_Y - WING_LINE)
    if nx < FINAL_THIRD_X:
        # Wide in the middle/attacking half = isolating a winger to stretch
        # the backline; central = circulating through the midfield.
        if is_wide and nx > THIRD_DEPTH + 15.0:
            return PossessionPhase.WING_ISOLATION
        return PossessionPhase.MIDFIELD_CIRCULATION

    # Final third: wide deliveries / box entries = wing isolation & penetration.
    if is_wide and nx >= FINAL_THIRD_X + 5.0:
        return PossessionPhase.WING_ISOLATION
    return PossessionPhase.BOX_PENETRATION


def normalize_x(x: float, attacks_right: bool) -> float:
    """Ball's distance (m) from the team's own goal line, direction-agnostic."""
    return x if attacks_right else PITCH_X - x


# ─────────────────────────────────────────────
# THE PHASE ENGINE — the tactical brain
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PossessionDecision:
    """The engine's verdict for one touch."""

    phase: PossessionPhase
    directive: TacticalDirective
    target: Optional[str] = None          # preferred pass target (name)
    regress_to_gk: bool = False           # True when the keeper is the outlet
    reason: str = ""


class PossessionPhaseEngine:
    """
    Stateless tactical decision engine for one attacking team.

    Reads a pitch snapshot, decides whether to progress to the next phase,
    sustain the current one, or REGRESS (recycle backward / emergency drop to
    the goalkeeper) when the forward routes are shut. This is the layer that
    turns the goalkeeper into a genuine eleventh outfield player in build-up:
    against an aggressive press the keeper becomes a permanent overload anchor,
    and when a winger's crossing lane is blocked the ball comes ALL THE WAY
    back to reset the phase — exactly the modern CB→LB→LW→(blocked)→LB→CB→GK
    structural reset sequence seen in Opta team profiles.
    """

    def __init__(
        self,
        gk: GKSnapshot,
        style_key: str = "balanced",
        carrier_iq: float = 0.70,
    ):
        self.gk = gk
        self.style_key = style_key
        # Carrier intelligence (vision/composite composure, 0-1). Elite
        # playmakers sustain a little less because they spot the vertical
        # ball earlier — circulation is patience, never a cap on ambition.
        self.carrier_iq = max(0.0, min(1.0, carrier_iq))

    # ── PUBLIC API ──────────────────────────────

    def decide(
        self,
        current_phase: PossessionPhase,
        ball_x: float,
        ball_y: float,
        carrier_role: str,
        teammates: List[TeammateSnapshot],
        under_pressure: bool = False,
        attacks_right: bool = True,
    ) -> PossessionDecision:
        """
        Analyse the possession state and return (phase, directive, target).

        forward_lanes_open is derived from the teammate snapshots: teammates
        AHEAD of the ball with marking below BLOCKED_MARKING count as an open
        lane; a teammate smothered by a goalside defender is not an option.
        """
        nx = normalize_x(ball_x, attacks_right)
        forward_options = self._forward_options(teammates, ball_x, attacks_right)
        lanes_open = len(forward_options)
        backward_options = self._backward_options(teammates, ball_x, attacks_right)

        # ── RULE 1: FORCED EMERGENCY REGRESSION (danger valve) ────────
        # A deep carrier bottled up with zero forward lanes MUST hand the
        # ball to the safety node. The keeper is the permanent overload
        # anchor of build-up — passing to him makes it a 3v2 (GK+2CB) against
        # the press rather than a hopeless 2v2.
        if (under_pressure and lanes_open == 0 and carrier_role in DEEP_CARRIER_ROLES
                and carrier_role != "GK"):
            if self.gk.lane_open and self._reachable(ball_x, ball_y, self.gk.x, self.gk.y):
                return PossessionDecision(
                    current_phase,
                    TacticalDirective.EMERGENCY_DROP_TO_GK,
                    target=self.gk.name, regress_to_gk=True,
                    reason="emergency_drop_to_gk",
                )
            reset = self._pick_reset_anchor(backward_options, ball_x, ball_y)
            if reset is not None:
                return PossessionDecision(
                    PossessionPhase.REGROUP_BUILD_UP,
                    TacticalDirective.RECYCLE_BACKWARD,
                    target=reset.name,
                    reason="emergency_recycle_backward",
                )

        # ── RULE 1b: BUILD-UP RELEASE TO GK (Juego de Posición) ───────
        # In the regroup/build-up phase a deep carrier routinely hands the
        # ball back to the keeper to invite the press and create a free man
        # — the modern GK-as-eleventh-outfielder. This is what produces the
        # CB→LB→LW→(blocked)→LB→CB→GK structural reset sequences and the
        # 15-30 touch keeper lines seen against pressing sides. Frequency
        # scales with the team's possession identity (tiki-taka sides lean
        # on the keeper constantly, route-one sides almost never).
        # The gate is GEOMETRIC (where the ball physically is), not the
        # carried narrative phase — a dropped-back recycle to the own third
        # must count as build-up even if the last directive was PROGRESS.
        geo_phase = possession_phase_for(ball_x, ball_y, attacks_right)
        if (geo_phase == PossessionPhase.REGROUP_BUILD_UP
                and carrier_role in DEEP_CARRIER_ROLES
                and carrier_role != "GK"):
            if (random.random() < self._gk_recycle_rate()
                    and self.gk.lane_open
                    and self._reachable(ball_x, ball_y, self.gk.x, self.gk.y)):
                return PossessionDecision(
                    geo_phase,
                    TacticalDirective.RELEASE_TO_GK,
                    target=self.gk.name, regress_to_gk=True,
                    reason="build_up_release_to_gk",
                )

        # ── RULE 2: TACTICAL PHASE REGRESSION (decompression) ────────
        # High up the pitch the box is congested (0 forward lanes open).
        # Do NOT force a low-probability final pass — reset the phase to
        # build-up via the deepest unmarked anchor, or a direct diagonal
        # back-pass to the keeper if there is no anchor behind the ball.
        if current_phase in (PossessionPhase.WING_ISOLATION, PossessionPhase.BOX_PENETRATION):
            if lanes_open == 0:
                gk_reset = (
                    self.gk.lane_open
                    and self._reachable(ball_x, ball_y, self.gk.x, self.gk.y)
                    and carrier_role != "GK"
                )
                reset = self._pick_reset_anchor(backward_options, ball_x, ball_y)
                # Structural reset through the keeper: when a WING attack is
                # shut down in the MIDDLE third, possession sides habitually
                # recycle the whole structure back to the keeper (rather than
                # stopping at the nearest anchor) so the build-up can re-form
                # with a free man. The diagonal is short enough to be safe
                # (GK ≤ ~45m) and this is the sequence that produces the
                # 15-30 touch keeper lines against pressing sides. Under
                # pressure the reset to the keeper is forced outright.
                #
                # Checkpoint 24 — wingers are exempt: a winger launching a
                # 40-60m diagonal back to his own keeper is not a pattern of
                # play, it's a giveaway. The shut-down winger's reset is the
                # SHORT ball to his overlapping fullback or the nearest
                # midfielder (the anchor pick below), or patience.
                if (gk_reset and carrier_role not in ("LW", "RW")
                        and (under_pressure or nx < 55.0)):
                    if under_pressure or random.random() < self._gk_recycle_rate():
                        return PossessionDecision(
                            PossessionPhase.REGROUP_BUILD_UP,
                            TacticalDirective.RELEASE_TO_GK,
                            target=self.gk.name, regress_to_gk=True,
                            reason="wing_blocked_structural_reset",
                        )
                if reset is not None:
                    # Checkpoint 24 — a winger's "nearest backward anchor"
                    # can be a CB 50-70m away when the whole structure has
                    # pushed up around him: launching that bomb back is not
                    # football. Beyond ~30m the shut-down winger recirculates
                    # instead (short support pass via the circulation web).
                    if (carrier_role in ("LW", "RW")
                            and math.hypot(reset.x - ball_x, reset.y - ball_y) > 30.0):
                        return PossessionDecision(
                            current_phase, TacticalDirective.SUSTAIN_CIRCULATION,
                            reason="wing_blocked_no_near_anchor",
                        )
                    return PossessionDecision(
                        PossessionPhase.REGROUP_BUILD_UP,
                        TacticalDirective.RECYCLE_BACKWARD,
                        target=reset.name,
                        reason="tactical_recycle_backward",
                    )
                if gk_reset and carrier_role not in ("LW", "RW"):
                    return PossessionDecision(
                        PossessionPhase.REGROUP_BUILD_UP,
                        TacticalDirective.EMERGENCY_DROP_TO_GK,
                        target=self.gk.name, regress_to_gk=True,
                        reason="direct_wing_to_gk_reset",
                    )
                return PossessionDecision(
                    current_phase, TacticalDirective.SUSTAIN_CIRCULATION,
                    reason="congested_no_backwards_option",
                )

        # ── RULE 3: PROGRESSIVE FLUID FLOW (default when lanes open) ─
        if lanes_open > 0:
            # TEMPO CIRCULATION (Checkpoint 23) — real possession is patient
            # by DEFAULT. An open forward lane does not mean the ball goes
            # forward: hub midfielders (Modric, Rodri, Tanaka) play lateral
            # and backward support passes on most touches, moving the block
            # and waiting for the lane that MATTERS. Without this roll the
            # phase machine is a one-way street — every touch progresses,
            # the ball leaves the middle third in 2-3 passes, and midfielder
            # pass maps come out as a sparse vertical fan instead of the
            # dense multidirectional web data providers publish.
            #
            # This is not a cap on ambition: the roll is probabilistic, it
            # shrinks for high-IQ carriers, it only governs the two patient
            # phases (final-third play stays aggressive), and a genuine
            # shooting window still overrides it downstream.
            if (carrier_role != "GK"
                    and current_phase in (PossessionPhase.REGROUP_BUILD_UP,
                                          PossessionPhase.MIDFIELD_CIRCULATION)):
                if random.random() < self._sustain_probability(
                        current_phase, under_pressure):
                    target = self._pick_circulation_target(
                        teammates, ball_x, ball_y, attacks_right,
                        carrier_role=carrier_role,
                    )
                    if target is not None:
                        return PossessionDecision(
                            current_phase,
                            TacticalDirective.SUSTAIN_CIRCULATION,
                            target=target.name,
                            reason="tempo_circulation",
                        )

            if current_phase == PossessionPhase.REGROUP_BUILD_UP:
                return PossessionDecision(
                    PossessionPhase.MIDFIELD_CIRCULATION,
                    TacticalDirective.PROGRESS,
                    reason="progress_to_midfield",
                )
            if current_phase == PossessionPhase.MIDFIELD_CIRCULATION:
                if self._wing_available(forward_options):
                    return PossessionDecision(
                        PossessionPhase.WING_ISOLATION,
                        TacticalDirective.WING_SWITCH,
                        target=self._pick_wing(forward_options),
                        reason="stretch_to_wings",
                    )
                return PossessionDecision(
                    PossessionPhase.BOX_PENETRATION,
                    TacticalDirective.PROGRESS,
                    reason="progress_to_final_third",
                )
            # Wing isolation or box penetration with a lane open — stay put
            # and try to penetrate.
            return PossessionDecision(
                PossessionPhase.BOX_PENETRATION,
                TacticalDirective.PROGRESS,
                reason="penetrate_box_for_shot",
            )

        # ── FALLBACK: keep the ball circulating ─────────────────────
        return PossessionDecision(
            current_phase, TacticalDirective.SUSTAIN_CIRCULATION,
            reason="sustain_circulation",
        )

    # ── INTERNAL HELPERS ───────────────────────

    def _sustain_probability(
        self,
        phase: PossessionPhase,
        under_pressure: bool = False,
    ) -> float:
        """
        Probability the carrier keeps circulating on THIS touch instead of
        progressing, even with a forward lane nominally open.

        Scaled by three things:
          - team identity   : tiki-taka sides live on the support pass,
                              route-one sides almost never circulate.
          - phase           : regroup is calmer than middle-third play
                              (the GK release rule already covers a chunk of
                              regroup patience, so it sits slightly lower).
          - carrier IQ      : elite vision trims the roll — a De Bruyne sees
                              the vertical ball a touch earlier, so he
                              sustains less. Nobody sustains MORE than their
                              style allows; circulation is a floor on
                              patience, never a ceiling on ambition.
          - pressure        : a pressed carrier plays the safe support ball
                              a little more often.
        """
        base = {
            "tiki_taka": 0.62,
            "structured_possession": 0.60,
            "possession": 0.60,
            "vertical_tiki_taka": 0.54,
            "wing_play": 0.50,
            "balanced": 0.52,
            "defensive": 0.48,
            "attacking": 0.46,
            "ultra_defensive": 0.42,
            "park_the_bus": 0.40,
            "gegenpressing": 0.38,
            "ultra_attacking": 0.38,
            "fluid_counter": 0.28,
            "counter": 0.28,
            "route_one": 0.18,
            "direct": 0.18,
        }.get(self.style_key, 0.50)

        if phase == PossessionPhase.REGROUP_BUILD_UP:
            base *= 0.70

        base *= 1.20 - 0.45 * self.carrier_iq
        if under_pressure:
            base *= 1.15
        return min(0.80, base)

    def _pick_circulation_target(
        self,
        teammates: List[TeammateSnapshot],
        ball_x: float,
        ball_y: float,
        attacks_right: bool,
        carrier_role: Optional[str] = None,
    ) -> Optional[TeammateSnapshot]:
        """
        Choose the support-angle receiver for a tempo-circulation pass.

        The target geometry mirrors how a real hub midfielder scans:
          - level with or BEHIND the ball (a support angle, not a progressive
            option) — a teammate a half-step ahead still counts as lateral;
          - near enough that the pass is routine (this is the safe,
            high-completion ball that keeps 87-96% accuracy alive);
          - free of marking and outside a cover-shadow corridor.

        Weighting favours the short, safe triangle (nearby midfielders and
        defenders), rewards the drop-back anchor slightly (the tempo setter),
        and gives the far-side diagonal switch a modest bonus so the lateral
        long balls on a Modric map appear at a realistic trickle. Weighted
        random selection keeps the web varied instead of deterministic.
        """
        sign = 1.0 if attacks_right else -1.0
        candidates: List[TeammateSnapshot] = []
        weights: List[float] = []
        lateral_idx: List[int] = []
        for t in teammates:
            # A tempo pass goes to a FREE man — stricter than the generic
            # outlet bar. This is what keeps circulation completion in the
            # 90%+ band the hub-midfielder reference maps show.
            if t.lane_blocked or t.marking >= 0.45:
                continue
            d = math.hypot(t.x - ball_x, t.y - ball_y)
            if d < CIRCULATION_MIN_RANGE_M or d > CIRCULATION_RANGE_M:
                continue
            ahead = (t.x - ball_x) * sign
            if ahead > CIRCULATION_AHEAD_TOLERANCE_M:
                continue
            # Checkpoint 24 — for WIDE carriers the box-bound "support" ball
            # is the cross mechanism's job; circulating into the box just
            # stamped Wingers' ordinary passes as crosses (20+/match).
            if carrier_role in ("LW", "RW", "LB", "RB"):
                box_line = BOX_LINE_X_ATTACKING if attacks_right else BOX_LINE_X_DEFENDING
                in_box = (t.x > box_line) if attacks_right else (t.x < box_line)
                if in_box:
                    continue
            w = CIRCULATION_ROLE_WEIGHT.get(t.role, 0.8)
            w *= 1.0 / (1.0 + d / 9.0)        # short passes dominate
            w *= 1.0 - 0.7 * t.marking         # free men get the ball
            dy_abs = abs(t.y - ball_y)
            if abs(ahead) <= 4.0 and dy_abs >= 3.0:
                w *= 1.60                      # the square ball is the heartbeat
            elif ahead < -2.0:
                w *= 0.95                      # drop-back available, not overused
            elif ahead > 1.0 and dy_abs < ahead:
                w *= 0.70                      # nudge forward: rare in circulation
            if dy_abs > 15.0 and d >= 18.0:
                w *= 1.35                      # far-side switch diagonal
            if abs(ahead) <= 4.0 and dy_abs >= 3.0:
                lateral_idx.append(len(candidates))
            candidates.append(t)
            weights.append(w)
        if not candidates:
            return None
        # The circulation web is built on the square ball: whenever a
        # level-with-the-ball option exists the carrier uses it, and the
        # drop-back to the defence stays the change of rhythm rather than
        # the default (deep anchors are abundant in any shape, so an
        # unweighted pool drifts backward-heavy and the pass map loses its
        # lateral thread).
        if lateral_idx:
            pool = [candidates[i] for i in lateral_idx]
            pool_w = [weights[i] for i in lateral_idx]
        else:
            pool, pool_w = candidates, weights
        return random.choices(pool, weights=pool_w, k=1)[0]

    def _gk_recycle_rate(self) -> float:
        """Probability a deep carrier releases the ball to the keeper on a
        given build-up touch, scaled by the team's possession identity."""
        if self.style_key in ("tiki_taka", "vertical_tiki_taka",
                               "structured_possession", "possession"):
            return 0.70 if self.style_key == "tiki_taka" else 0.62
        if self.style_key in ("balanced", "attacking", "wing_play"):
            return 0.40
        if self.style_key in ("fluid_counter", "counter", "defensive"):
            return 0.24
        # route_one, gegenpressing, park_the_bus, ultra_attacking, direct…
        return 0.12

    def _forward_options(
        self,
        teammates: List[TeammateSnapshot],
        ball_x: float,
        attacks_right: bool,
    ) -> List[TeammateSnapshot]:
        ahead = []
        for t in teammates:
            is_ahead = (t.x > ball_x) if attacks_right else (t.x < ball_x)
            if is_ahead and not t.lane_blocked and t.marking < BLOCKED_MARKING:
                ahead.append(t)
        return ahead

    def _backward_options(
        self,
        teammates: List[TeammateSnapshot],
        ball_x: float,
        attacks_right: bool,
    ) -> List[TeammateSnapshot]:
        """Reset anchors behind the ball line (excludes the keeper — he is
        handled as the dedicated GK outlet)."""
        anchors = []
        for t in teammates:
            if t.role not in RESET_ANCHOR_ROLES:
                continue
            behind = (t.x < ball_x) if attacks_right else (t.x > ball_x)
            if behind and not t.lane_blocked and t.marking < BLOCKED_MARKING:
                anchors.append(t)
        return anchors

    def _pick_reset_anchor(
        self, anchors: List[TeammateSnapshot], ball_x: float, ball_y: float
    ) -> Optional[TeammateSnapshot]:
        if not anchors:
            return None
        # The safest reset is the unmarked deep anchor CLOSEST to the ball
        # (a short recycle is far less risky than a raking diagonal).
        return min(anchors, key=lambda t: math.hypot(t.x - ball_x, t.y - ball_y))

    def _wing_available(self, forward_options: List[TeammateSnapshot]) -> bool:
        return any(t.role in WING_ROLES for t in forward_options)

    def _pick_wing(self, forward_options: List[TeammateSnapshot]) -> Optional[str]:
        wings = [t for t in forward_options if t.role in WING_ROLES]
        if not wings:
            return None
        return min(wings, key=lambda t: t.marking).name

    @staticmethod
    def _reachable(x0: float, y0: float, x1: float, y1: float) -> bool:
        # A keeper 70m upfield is not a passing option; within ~45m he is a
        # viable (and in build-up, routine) outlet.
        return math.hypot(x1 - x0, y1 - y0) <= 45.0
