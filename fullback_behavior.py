"""
PLOFA 26/27 — MODERN FULLBACK BEHAVIOR ENGINE (Checkpoint 32)
==============================================================
fullback_behavior.py

Philosophy:
    The modern fullback (LB/RB) is the most positionally schizophrenic
    player on the pitch. Over one possession he may be a touchline
    defender, an overlapping sprinter, an underlapping half-space runner,
    or — for inverted fullbacks (Zinchenko, Cancelo) — a de facto central
    midfielder. And 10 seconds later he must be all the way back holding
    a defensive line against a counter down his flank.

    Real fullback behavior is therefore not ONE position but a SET OF
    MODES with real triggers between them:

        IN POSSESSION:
        1. OVERLAP      — winger has the ball on HIS flank ahead → sprint
                          OUTSIDE him into the touchline→byline corridor
                          (Robertson/Trent), dragging the defending winger
                          and offering the wide release pass.
        2. UNDERLAP     — run INSIDE the winger into the half-space
                          channel between opponent fullback and centre-back
                          when that corridor is open.
        3. INVERT/TUCK  — inverted fullbacks tuck INTO midfield next to the
                          CDM during build-up, forming the box midfield,
                          leaving the winger as the sole width on the flank.
        4. HOLD         — ball on the FAR flank / defensive_fullback DNA:
                          stay home, hold the back line, do NOT join.

        OUT OF POSSESSION:
        5. ENGAGE/JOCKEY— facing the opposition winger 1v1: engage hard
                          only with cover behind; otherwise jockey/delay
                          and show him down the line.
        6. RECOVER      — turnover and the ball is running down his flank
                          behind him: emergency recovery sprint priority.

        ON THE BALL:
        7. DELIVER      — from the byline corridor the modern fullback's
                          highest-value ball is the CUTBACK (Trent/
                          Robertson); whipped crosses are secondary.

    This module gives every fullback a persistent spatial identity built
    once at kickoff from DNA archetype + tendencies + specialties:
        - advance_instinct     (does he join attacks at all)
        - overlap_instinct     (outside-the-winger run)
        - underlap_instinct    (inside-the-winger run)
        - tuck_instinct        (invert into midfield in build-up)
        - delivery_instinct    (cutbacks/crosses from wide)
        - hold_discipline      (resists advancing when he shouldn't)
        - duel_aggression      (1v1 defending engagement)
        - recovery_urgency     (counter-attack sprint priority)

    It is PURE — reads PositionEngine state, writes nothing. The event
    chain and position engine consume its geometry to steer fullback
    positioning, run timing, engagement decisions, and delivery choice.

    Pitch geometry (StatsBomb scale):
        x: 0 (own goal line) → 105 (opponent goal line)
        y: 0 (left touchline) → 68 (right touchline)
        Center: y = 34
        Left flank channel:  y < 20   (LB home_y = 10)
        Right flank channel: y > 48   (RB home_y = 58)

    Archetypes consumed (player_dna.ArchetypeLibrary):
        attacking_fullback  → high/overlapping (Alexander-Arnold, Robertson)
        inverted_fullback   → tucks inside (Zinchenko, Cancelo)
        defensive_fullback  → stay-back first (traditional FB)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# FLANK GEOMETRY CONSTANTS
# ─────────────────────────────────────────────

PITCH_Y = 68.0
CENTER_Y = 34.0
PITCH_X = 105.0

# Touchline anchors — mirror BASE_HOME_POSITIONS in position_engine
# (LB home_y=10, RB home_y=58 before width scaling). Formation-aware
# callers pass the player's actual formation home_y instead.
LEFT_TOUCHLINE_ANCHOR_Y = 10.0
RIGHT_TOUCHLINE_ANCHOR_Y = 58.0

# Byline / cross-zone geometry — deliberately identical to
# winger_behavior.py so both wide roles share ONE definition of the
# dangerous corridor (Checkpoint 24's tightened cross zone).
BYLINE_X_ATTACKING = 105.0
BYLINE_X_DEFENDING = 0.0
CROSS_ZONE_X_ATTACKING = 82.0
CROSS_ZONE_X_DEFENDING = 23.0

# Flank channel half-width measured from the FORMATION anchor — same
# banding as the wingers (FLANK_CHANNEL_HALF_WIDTH_M = 7.0).
FLANK_CHANNEL_HALF_WIDTH_M = 7.0

# Overlap corridor: the fullback's runway OUTSIDE the winger, from his
# anchor to the touchline, active once play is in the final two thirds.
OVERLAP_ZONE_X_ATTACKING = 62.0
OVERLAP_ZONE_X_DEFENDING = 43.0
OVERLAP_MAX_X_ATTACKING = 97.0     # don't camp on the goal line
OVERLAP_MAX_X_DEFENDING = 8.0

# Underlap channel: half-space band immediately infield of the anchor —
# same width as the winger's HALF_SPACE_WIDTH_M so "the gap between
# their fullback and centre-back" means the same thing everywhere.
UNDERLAP_HALFSPACE_WIDTH_M = 14.0
UNDERLAP_DEFENDER_RADIUS_M = 10.0

# Invert/tuck zone: build-up territory where an inverted fullback steps
# into midfield next to the CDM (own + middle third). Target y sits this
# far infield of the anchor (≈ the CDM pocket, y≈28-40).
TUCK_ZONE_X_MAX_ATTACKING = 62.0
TUCK_ZONE_X_MIN_ATTACKING = 20.0
TUCK_Y_INFIELD_M = 18.0

# Advance gating: how close to the halfway line play must be before an
# attacking fullback even CONSIDERS bombing on (attacking-right frame).
ADVANCE_GATE_X_ATTACKING = 45.0
ADVANCE_GATE_X_DEFENDING = 60.0

# 1v1 defending: the opposition winger is "on the fullback" when within
# ISOLATION_RANGE_M; ENGAGE_RANGE_M is when a decision must be made.
ISOLATION_RANGE_M = 8.0
ENGAGE_RANGE_M = 6.0

# Cover: a centre-back recovering goal-side within this distance means
# the fullback CAN commit to the tackle (cover is behind him).
COVER_BEHIND_RANGE_M = 12.0

# Recovery sprint: an opponent carrier beyond this far ahead of the
# fullback on his flank is a genuine emergency (fullback behind play).
RECOVERY_BEHIND_BALL_M = 12.0

# Fatigue banding — deliberately mirrors squad_manager.PlayerStaminaState
# bands AND winger_behavior._stamina_mult so BOTH wide roles degrade on
# the SAME curve the rest of the engine uses.
def _stamina_mult(stamina_pct: float) -> float:
    s = max(0.0, min(100.0, stamina_pct))
    if s >= 60:
        return 1.0
    elif s >= 40:
        return 0.88 + (s - 40) / 20 * 0.12
    elif s >= 20:
        return 0.75 + (s - 20) / 20 * 0.13
    else:
        return 0.60 + (s / 20) * 0.15


def _attacks_right_goal_x(attacks_right: bool) -> float:
    return BYLINE_X_ATTACKING if attacks_right else BYLINE_X_DEFENDING


def _attacks_right_cross_zone_x(attacks_right: bool) -> float:
    return CROSS_ZONE_X_ATTACKING if attacks_right else CROSS_ZONE_X_DEFENDING


def _attacks_right_overlap_zone_x(attacks_right: bool) -> float:
    return OVERLAP_ZONE_X_ATTACKING if attacks_right else OVERLAP_ZONE_X_DEFENDING


def _attacks_right_advance_gate_x(attacks_right: bool) -> float:
    return ADVANCE_GATE_X_ATTACKING if attacks_right else ADVANCE_GATE_X_DEFENDING


# ─────────────────────────────────────────────
# FULLBACK SPATIAL PROFILE
# ─────────────────────────────────────────────

@dataclass
class FullbackSpatialProfile:
    """
    Persistent per-fullback geometry identity. Built once at kickoff from
    the player's DNA archetype + tendencies + specialties, then read
    every touch / every drift minute.

    Attributes:
        flank: "left" | "right" — which touchline this fullback owns.
        touchline_anchor_y: the y-coordinate of their touchline channel.
        flank_commitment: 0-1 — how strongly they resist drifting off
            their touchline channel (consumed directly by the position
            engine's flank pull, exactly like the wingers').
        advance_instinct: 0-1 — base appetite to join the attack at all.
            0.9 = Robertson-type high fullback; 0.25 = stay-back FB.
        overlap_instinct: 0-1 — how often they take the OUTSIDE run
            around the winger when he receives on their flank.
        underlap_instinct: 0-1 — how often they take the INSIDE run into
            the half-space channel instead.
        tuck_instinct: 0-1 — how often they invert into midfield during
            build-up (Zinchenko/Cancelo mode). High ONLY for inverted FBs.
        delivery_instinct: 0-1 — willingness to deliver from the wide
            crossing zone vs recycle/carry on.
        hold_discipline: 0-1 — resistance to advancing when the trigger
            conditions aren't clean (ball far side, team out of shape).
        duel_aggression: 0-1 — how eagerly they engage the opposition
            winger 1v1 vs jockeying and delaying.
        recovery_urgency: 0-1 — how hard they sprint back when the ball
            turns over down their flank.
    """
    flank: str = "right"
    touchline_anchor_y: float = RIGHT_TOUCHLINE_ANCHOR_Y
    flank_commitment: float = 0.75
    advance_instinct: float = 0.50
    overlap_instinct: float = 0.50
    underlap_instinct: float = 0.35
    tuck_instinct: float = 0.10
    delivery_instinct: float = 0.35
    hold_discipline: float = 0.60
    duel_aggression: float = 0.50
    recovery_urgency: float = 0.65

    @property
    def is_left(self) -> bool:
        return self.flank == "left"

    @property
    def is_right(self) -> bool:
        return self.flank == "right"

    def flank_channel(self, y: float) -> bool:
        """Is this y-coordinate inside this fullback's flank channel? (name-based frame fallback)"""
        return abs(y - self.touchline_anchor_y) <= FLANK_CHANNEL_HALF_WIDTH_M + 6.0

    def in_flank_channel(self, y: float, anchor_y: Optional[float] = None) -> bool:
        """
        Formation-aware flank-channel membership. `anchor_y` (formation
        home_y) is authoritative — a mirrored team's LB stands on the
        right side of the pitch, so the name-based check is only a
        fallback when no anchor is supplied.
        """
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        return abs(y - a) <= FLANK_CHANNEL_HALF_WIDTH_M + 6.0

    def touchline_distance(self, y: float) -> float:
        """How far is this fullback from their touchline (metres)?"""
        return abs(y - self.touchline_anchor_y)

    def byline_distance(self, x: float, attacks_right: bool) -> float:
        """How far is this fullback from the byline they attack (metres)?"""
        return abs(_attacks_right_goal_x(attacks_right) - x)

    def in_cross_zone(self, x: float, y: float, attacks_right: bool,
                      anchor_y: Optional[float] = None) -> bool:
        """Is the fullback in the dangerous wide crossing zone?"""
        cross_x = _attacks_right_cross_zone_x(attacks_right)
        in_x = x > cross_x if attacks_right else x < cross_x
        return in_x and self.in_flank_channel(y, anchor_y)

    # ── RUN-CORRIDOR GEOMETRY ────────────────────────────────────

    def ball_on_my_flank(self, ball_y: float, anchor_y: Optional[float] = None,
                         tolerance_m: float = 16.0) -> bool:
        """Is the live ball on THIS fullback's flank?"""
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        return abs(ball_y - a) <= tolerance_m

    def ball_on_far_flank(self, ball_y: float, anchor_y: Optional[float] = None,
                          tolerance_m: float = 22.0) -> bool:
        """Is the live ball clearly across the pitch from this fullback?"""
        return not self.ball_on_my_flank(ball_y, anchor_y, tolerance_m)

    def in_overlap_corridor(self, x: float, y: float, attacks_right: bool,
                            anchor_y: Optional[float] = None) -> bool:
        """
        The outside-run runway: advanced enough (final-two-thirds gate),
        wide (within ~10m of the touchline side of the anchor), and not
        camped on the goal line.
        """
        gate = _attacks_right_overlap_zone_x(attacks_right)
        in_x = x > gate if attacks_right else x < gate
        max_x = OVERLAP_MAX_X_ATTACKING if attacks_right else OVERLAP_MAX_X_DEFENDING
        beyond_max = x > max_x if attacks_right else x < max_x
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        # Outside-run lane: ON THE TOUCHLINE SIDE of the anchor (allowing
        # ≤2m inside tolerance), within 10m of it.
        if a <= CENTER_Y:
            wide_enough = (a - y) <= 10.0 and y <= a + 2.0
        else:
            wide_enough = (y - a) <= 10.0 and y >= a - 2.0
        return in_x and not beyond_max and wide_enough

    def in_underlap_channel(self, x: float, y: float, attacks_right: bool,
                            anchor_y: Optional[float] = None) -> bool:
        """
        The inside-run channel: half-space band strictly INFIELD of the
        anchor (at least 3m inside — a ball at his own touchline post is
        not "between fullback and centre-back").
        """
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        lo, hi = (a, a + UNDERLAP_HALFSPACE_WIDTH_M) if a <= CENTER_Y \
            else (a - UNDERLAP_HALFSPACE_WIDTH_M, a)
        if a <= CENTER_Y:
            if not (lo + 3.0 <= y <= hi + 2.0):
                return False
        else:
            if not (lo - 2.0 <= y <= hi - 3.0):
                return False
        gate = _attacks_right_overlap_zone_x(attacks_right)
        return x > gate if attacks_right else x < gate

    def in_tuck_zone(self, x: float, attacks_right: bool) -> bool:
        """Build-up territory where an inverted fullback steps inside."""
        if attacks_right:
            return TUCK_ZONE_X_MIN_ATTACKING <= x <= TUCK_ZONE_X_MAX_ATTACKING
        return (PITCH_X - TUCK_ZONE_X_MAX_ATTACKING) <= x \
            <= (PITCH_X - TUCK_ZONE_X_MIN_ATTACKING)

    def tuck_target_y(self, anchor_y: Optional[float] = None) -> float:
        """
        Where does the tucked fullback stand? TUCK_Y_INFIELD_M infield of
        his anchor — the CDM pocket (y≈28-40), NOT the exact centre (he
        keeps a residual flank bias, like Zinchenko shading left-CM).
        """
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        direction = 1.0 if a <= CENTER_Y else -1.0
        return round(a + direction * TUCK_Y_INFIELD_M, 1)

    def advance_run_target(
        self,
        x: float, y: float,
        attacks_right: bool,
        anchor_y: Optional[float] = None,
        mode: str = "overlap",
    ) -> Tuple[float, float]:
        """
        Destination of an advancing run.

        overlap   → tight to the touchline, AHEAD of the ball toward the
                    byline corridor (the release-pass station).
        underlap  → half-space channel, slightly less deep than the
                    overlap (arriving between fullback and centre-back).
        Returns (target_x, target_y).
        """
        sign = 1.0 if attacks_right else -1.0
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        goal_x = _attacks_right_goal_x(attacks_right)

        if mode == "underlap":
            ty = a + (CENTER_Y - a) * 0.55
            tx = x + sign * 10.0
            tx = min(tx, goal_x - 14.0) if attacks_right else max(tx, goal_x + 14.0)
        elif mode == "tuck":
            ty = self.tuck_target_y(a)
            tx = min(max(x, 30.0), TUCK_ZONE_X_MAX_ATTACKING) if attacks_right \
                else max(min(x, PITCH_X - 30.0), PITCH_X - TUCK_ZONE_X_MAX_ATTACKING)
        else:  # overlap
            # Pull the y-target 70% of the way from current y toward the
            # touchline edge of his channel — the outside lane.
            touch_edge = max(4.0, a - 3.5) if a <= CENTER_Y else min(64.0, a + 3.5)
            ty = y + (touch_edge - y) * 0.70
            tx = min(x + sign * 14.0, goal_x - sign * 8.0)
            if attacks_right:
                tx = min(tx, OVERLAP_MAX_X_ATTACKING)
            else:
                tx = max(tx, OVERLAP_MAX_X_DEFENDING)

        tx = round(max(4.0, min(PITCH_X - 4.0, tx)), 1)
        ty = round(max(4.0, min(64.0, ty)), 1)
        return tx, ty

    # ── DANGER / THREAT GEOMETRY ─────────────────────────────────

    def danger_zone_score(self, x: float, y: float, attacks_right: bool) -> float:
        """
        How dangerous is this fullback's CURRENT attacking position? 0-1.
        Peaks in the advanced wide corridor — the areas overlapping
        fullbacks create overloads from.
        """
        dist_to_byline = self.byline_distance(x, attacks_right)
        byline_score = max(0.0, 1.0 - dist_to_byline / 40.0)
        touch_score = max(0.0, 1.0 - self.touchline_distance(y) / 18.0)
        return max(0.0, min(1.0, byline_score * 0.55 + touch_score * 0.45))

    def space_ahead(
        self,
        x: float, y: float,
        attacks_right: bool,
        defenders: Optional[List],
        position_engine,
    ) -> float:
        """
        How much room lies ahead of an advancing fullback? 1.0 (empty
        highway) → 0.0 (wall). Samples defender density along the
        corridor 12m ahead of the fullback, using the same
        position_engine.get_position() pattern as the winger module.
        """
        if position_engine is None or not defenders:
            return 0.6  # unknown — neutral
        sign = 1.0 if attacks_right else -1.0
        probe_x = x + sign * 12.0
        closest = float("inf")
        for d in defenders:
            if getattr(d, "position", None) == "GK":
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, dy = position_engine.get_position(dname)
            dist = math.hypot(dx - probe_x, dy - y)
            if dist < closest:
                closest = dist
        if closest == float("inf"):
            return 1.0
        return max(0.0, min(1.0, closest / UNDERLAP_DEFENDER_RADIUS_M))

    def half_space_openness(
        self,
        x: float, y: float,
        attacks_right: bool,
        defenders: Optional[List],
        position_engine,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        How open is the underlap channel (the corridor infield of the
        anchor)? 0 (packed) → 1 (wide open). Mirrors the winger module's
        half_space_openness with pure defender geometry.
        """
        if position_engine is None or not defenders:
            return 0.5  # unknown — neutral

        a = anchor_y if anchor_y is not None else self.touchline_anchor_y
        lo, hi = (a, a + UNDERLAP_HALFSPACE_WIDTH_M) if a <= CENTER_Y \
            else (a - UNDERLAP_HALFSPACE_WIDTH_M, a)

        sign = 1.0 if attacks_right else -1.0
        cx = x + sign * 10.0
        cy = (lo + hi) / 2.0

        closest = float("inf")
        occupants = 0
        for d in defenders:
            if getattr(d, "position", None) == "GK":
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, dy = position_engine.get_position(dname)
            if not (lo - 3.0 <= dy <= hi + 3.0):
                continue
            dist = math.hypot(dx - cx, dy - cy)
            if dist < UNDERLAP_DEFENDER_RADIUS_M:
                occupants += 1
            if dist < closest:
                closest = dist

        if closest == float("inf"):
            score = 1.0
        else:
            score = max(0.0, min(1.0, closest / UNDERLAP_DEFENDER_RADIUS_M))
        score -= 0.15 * max(0, occupants - 1)
        return max(0.0, min(1.0, score))

    # ── DEFENSIVE GEOMETRY ───────────────────────────────────────

    def winger_isolation_defense(
        self,
        x: float, y: float,
        attackers: Optional[List],
        position_engine,
        attacks_right: bool = True,
    ) -> Tuple[bool, Optional[object], float]:
        """
        Is THIS fullback isolated 1v1 against the opposition WINGER?

        Returns (isolated, winger, distance). Isolated when the nearest
        attacker is the opposition winger who runs at THIS fullback —
        mirroring winger_behavior's convention (`marking_fullback_pos =
        "RB" if is_left`): an opposing LW attacks our RB, an opposing RW
        attacks our LB — inside ISOLATION_RANGE_M.
        """
        if position_engine is None or not attackers:
            return False, None, float("inf")

        threat_positions = ("LW",) if self.is_right else ("RW",)

        best_w = None
        best_w_dist = float("inf")
        nearest_att = None
        nearest_dist = float("inf")

        for p in attackers:
            if getattr(p, "position", None) == "GK":
                continue
            pname = getattr(p, "name", None)
            if pname is None:
                continue
            px, py = position_engine.get_position(pname)
            dist = math.hypot(px - x, py - y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_att = p
            if getattr(p, "position", None) in threat_positions:
                if dist < best_w_dist:
                    best_w_dist = dist
                    best_w = p

        if best_w is None:
            return False, None, float("inf")

        isolated = (
            nearest_att is not None
            and best_w is not None
            and nearest_att.name == best_w.name
            and best_w_dist < ISOLATION_RANGE_M
        )
        return isolated, best_w, best_w_dist

    def cover_behind_available(
        self,
        x: float, y: float,
        teammates: Optional[List],
        position_engine,
        attacks_right: bool = True,
        exclude_name: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Is a centre-back (or CDM) goal-side of me, close enough to cover
        if I dive into the 1v1? Fullbacks engage MUCH harder with cover —
        Van Dijk's presence is why Trent can jump into presses.

        Returns (cover_available, distance_of_nearest_cover).
        """
        if position_engine is None or not teammates:
            return False, float("inf")

        sign = 1.0 if attacks_right else -1.0
        nearest_cover = float("inf")
        for m in teammates:
            name = getattr(m, "name", None)
            if name is None or name == exclude_name:
                continue
            if getattr(m, "position", None) not in ("CB", "CDM"):
                continue
            mx, my = position_engine.get_position(name)
            goal_side = (mx - x) * sign > -2.0   # level or goal-side
            if not goal_side:
                continue
            dist = math.hypot(mx - x, my - y)
            if dist < nearest_cover:
                nearest_cover = dist

        return nearest_cover < COVER_BEHIND_RANGE_M, nearest_cover

    def recovery_priority(
        self,
        fb_x: float, fb_y: float,
        ball_x: float, ball_y: float,
        attacks_right: bool,
        opponents: Optional[List] = None,
        position_engine=None,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        Emergency-recovery urgency 0-1 after a turnover. Peaks when the
        ball (or an opponent carrier) is running down HIS flank BEHIND
        him toward his own goal — the classic exposed-overlap scenario.
        """
        a = anchor_y if anchor_y is not None else self.touchline_anchor_y

        # Ball on his flank?
        flank_score = max(0.0, 1.0 - abs(ball_y - a) / 22.0)

        # Ball behind him (goal-side of the fullback)?
        if attacks_right:
            behind = (ball_x - fb_x) < -RECOVERY_BEHIND_BALL_M
            depth_score = max(0.0, min(1.0, (fb_x - ball_x) / 30.0)) \
                if ball_x < fb_x else 0.0
            own_half_threat = max(0.0, 1.0 - ball_x / 45.0)
        else:
            behind = (fb_x - ball_x) < -RECOVERY_BEHIND_BALL_M
            depth_score = max(0.0, min(1.0, (ball_x - fb_x) / 30.0)) \
                if ball_x > fb_x else 0.0
            own_half_threat = max(0.0, 1.0 - (PITCH_X - ball_x) / 45.0)

        urgency = flank_score * 0.40 + depth_score * 0.35 + own_half_threat * 0.25
        if behind:
            urgency = min(1.0, urgency + 0.20)
        return max(0.0, min(1.0, urgency))


# ─────────────────────────────────────────────
# FULLBACK BEHAVIOR ENGINE
# ─────────────────────────────────────────────

class FullbackBehaviorEngine:
    """
    Pure decision engine for modern fullback play. Reads PositionEngine
    state + the fullback's spatial profile, returns steering decisions
    that the event chain and position engine consume.

    All methods are stateless — the profile is passed in per call.
    """

    # ── POSITIONING STEERING ──────────────────────────────────

    @staticmethod
    def flank_pull_strength(
        profile: FullbackSpatialProfile,
        current_y: float,
        in_possession: bool = True,
        stamina_pct: float = 100.0,
    ) -> float:
        """
        How hard should the position engine pull this fullback back onto
        their touchline channel this minute?

        - An advancing fullback mid-overlap holds width more loosely than
          a defensive fullback; the pull scales with hold discipline.
        - Out of possession the pull strengthens (shape integrity).
        """
        stam_mult = 0.80 + 0.20 * _stamina_mult(stamina_pct)
        if profile.flank_channel(current_y):
            base = 0.15 if in_possession else 0.20
        else:
            drift = profile.touchline_distance(current_y)
            if drift > FLANK_CHANNEL_HALF_WIDTH_M + 6.0:
                base = 0.45 if in_possession else 0.50
            else:
                base = 0.30 if in_possession else 0.35
        # Hold discipline modulates: a stay-back FB snaps back harder;
        # an inverted FB tolerates being infield.
        discipline_mult = 0.70 + 0.45 * profile.hold_discipline
        return round(min(1.0, base * stam_mult * discipline_mult), 3)

    # ── ADVANCING DECISIONS ───────────────────────────────────

    @staticmethod
    def should_advance(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attacks_right: bool,
        ball_x: float, ball_y: float,
        in_possession: bool = True,
        under_pressure: bool = False,
        stamina_pct: float = 100.0,
        anchor_y: Optional[float] = None,
        game_state_urgent: bool = False,
    ) -> bool:
        """
        Base gate: does this fullback join the attack AT ALL right now?

        Triggers stacked:
            - team in possession
            - ball past the advance gate (middle third onwards)
            - NOT under immediate pressure in a risky area
            - advance instinct beats the roll, minus hold discipline
              when the ball is on the FAR flank (cross-field advances
              leave the back line exposed — real teams mostly advance
              the BALL-SIDE fullback)
        Losing leads / chasing games can force it (game_state_urgent).
        """
        if not in_possession:
            return False

        stam_mult = _stamina_mult(stamina_pct)
        gate = _attacks_right_advance_gate_x(attacks_right)
        ball_advanced = ball_x > gate if attacks_right else ball_x < (PITCH_X - gate)
        if not (ball_advanced or game_state_urgent):
            return False

        if under_pressure and profile.hold_discipline > 0.5:
            return False

        # Far-flank advances are heavily discounted unless urgent
        far_flank_mult = 0.45 if profile.ball_on_far_flank(ball_y, anchor_y) else 1.0

        prob = (
            profile.advance_instinct * 0.70
            + (1.0 - profile.hold_discipline) * 0.20
        ) * far_flank_mult * stam_mult
        return random.random() < prob

    @staticmethod
    def choose_advance_mode(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attacks_right: bool,
        ball_x: float, ball_y: float,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: Optional[float] = None,
        stamina_pct: float = 100.0,
    ) -> Optional[str]:
        """
        WHICH advancing run does he take? Returns one of:
            "overlap"   — outside the winger, touchline→byline lane
            "underlap"  — inside the winger into the half-space
            "tuck"      — invert into the midfield pocket (build-up only)
            None        — no run (hold)

        Priority logic mirrors real coaching:
            - Build-up territory (own/mid third) + inverted DNA → tuck.
            - Final two thirds: pick the run whose CHANNEL is openest —
              overlap needs the outside lane free, underlap needs the
              half-space open. Instinct weights break near-ties.
        """
        stam_mult = _stamina_mult(stamina_pct)

        # ── TUCK (inverted fullback, build-up territory only) ──
        if profile.tuck_instinct > 0.30 and profile.in_tuck_zone(ball_x, attacks_right):
            prob = profile.tuck_instinct * 0.85 * stam_mult
            if random.random() < prob:
                return "tuck"
            return None

        # ── FINAL TWO THIRDS: overlap vs underlap ──
        gate = _attacks_right_overlap_zone_x(attacks_right)
        ball_advanced = ball_x > gate if attacks_right else ball_x < (PITCH_X - gate)
        if not ball_advanced:
            return None

        openness_outside = profile.space_ahead(
            x, y, attacks_right, defenders, position_engine,
        ) if (defenders and position_engine is not None) else 0.7
        openness_inside = profile.half_space_openness(
            x, y, attacks_right, defenders, position_engine, anchor_y,
        ) if (defenders and position_engine is not None) else 0.5

        overlap_score = profile.overlap_instinct * (0.45 + 0.55 * openness_outside)
        underlap_score = profile.underlap_instinct * (0.45 + 0.55 * openness_inside)

        total = overlap_score + underlap_score
        if total <= 0.05:
            return None
        if random.random() < (overlap_score / total) * stam_mult:
            return "overlap"
        return "underlap"

    @staticmethod
    def should_hold_position(
        profile: FullbackSpatialProfile,
        ball_x: float, ball_y: float,
        attacks_right: bool,
        in_possession: bool,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        How strongly should this fullback resist advancing this minute?
        0 (free to go) → 1 (nailed to the back line).

        Defensive fullbacks and far-flank situations pin the fullback;
        ball-side possession releases him.
        """
        strength = profile.hold_discipline * 0.55
        if profile.ball_on_far_flank(ball_y, anchor_y):
            strength += 0.25
        if not in_possession:
            strength += 0.30
        return max(0.0, min(1.0, strength))

    # ── DELIVERY DECISIONS ────────────────────────────────────

    @staticmethod
    def should_deliver(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attacks_right: bool,
        under_pressure: bool = False,
        anchor_y: Optional[float] = None,
    ) -> bool:
        """
        Should this fullback deliver from the wide zone right now?

        Fullbacks deliver LESS than wingers (a real attacking fullback
        hits ~1-4 open-play crosses/90 vs a winger's 2-6) — the base
        probability is deliberately lower, and the byline bonus favours
        the CUTBACK station rather than the goal line.
        """
        if not profile.in_cross_zone(x, y, attacks_right, anchor_y):
            return False
        byline_dist = profile.byline_distance(x, attacks_right)
        cutback_station_bonus = max(0.0, 1.0 - abs(byline_dist - 11.0) / 20.0) * 0.25
        if under_pressure:
            prob = profile.delivery_instinct * 0.35
        else:
            prob = profile.delivery_instinct * 0.45 + cutback_station_bonus
        return random.random() < prob

    @staticmethod
    def decide_delivery_type(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attacks_right: bool,
        under_pressure: bool = False,
    ) -> str:
        """
        What KIND of delivery? Modern fullbacks are cutback specialists
        (Trent/Robertson): the byline-adjacent pull-back dominates.
        Same taxonomy as the winger module so downstream consumers can
        treat deliveries uniformly.
        Returns one of: "cutback", "driven_low", "whipped_early", "chip".
        """
        byline_dist = profile.byline_distance(x, attacks_right)

        if byline_dist < 13.0 and not under_pressure:
            # The cutback station — the modern fullback's signature ball.
            return "cutback"
        if byline_dist < 20.0:
            return "driven_low"
        if profile.delivery_instinct >= 0.55 and byline_dist < 32.0:
            return "whipped_early"
        return "chip"

    # ── CARRY STEERING ────────────────────────────────────────

    @staticmethod
    def carry_direction_bias(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attacks_right: bool,
        in_possession: bool = True,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        Lateral bias for a fullback's carry (positive = toward own
        touchline, negative = toward center, in metres).

        Standard fullbacks carry ALONG the touchline. Inverted fullbacks
        in build-up territory bias INWARD toward their tuck pocket.
        Opponent-avoidance probe eases the bias off when a body sits
        exactly where the bias steers (same pattern as the winger).
        """
        anchor = anchor_y if anchor_y is not None else profile.touchline_anchor_y

        if profile.in_flank_channel(y, anchor_y):
            base_bias = 0.0
        else:
            base_bias = (anchor - y) * 0.30

        # Inverted fullback inward drift in build-up zones
        if (in_possession and profile.tuck_instinct > 0.40
                and profile.in_tuck_zone(x, attacks_right)):
            target = profile.tuck_target_y(anchor)
            inward = (target - y) * 0.25
            base_bias = base_bias + inward if abs(base_bias) > 0.01 else inward

        if defenders and position_engine is not None and base_bias != 0.0:
            direction = 1.0 if base_bias > 0 else -1.0
            probe_y = y + direction * 4.0
            for d in defenders:
                if getattr(d, "position", None) == "GK":
                    continue
                dname = getattr(d, "name", None)
                if dname is None:
                    continue
                dx, dy = position_engine.get_position(dname)
                if math.hypot(dx - x, dy - probe_y) < 5.0:
                    base_bias *= 0.4
                    break

        return base_bias

    # ── DEFENSIVE DECISIONS ───────────────────────────────────

    @staticmethod
    def defend_engagement(
        profile: FullbackSpatialProfile,
        x: float, y: float,
        attackers: List,
        teammates: Optional[List],
        position_engine=None,
        attacks_right: bool = True,
        winger_pace_advantage: bool = False,
        stamina_pct: float = 100.0,
    ) -> str:
        """
        The 1v1 defending decision facing an oncoming winger. Returns:
            "engage"       — dive into the tackle/challenge now
            "jockey"       — delay, show down the line, wait for support
            "show_inside"  — angle the body to force him infield (toward
                             cover) instead of the byline

        Real logic: engage HARD only with cover behind; without cover,
        delay. Pace disadvantage pushes toward showing inside (concede
        the line only when he can't win the footrace). Scaled by
        duel_aggression and stamina.
        """
        isolated, winger, dist = profile.winger_isolation_defense(
            x, y, attackers, position_engine, attacks_right,
        )
        if not isolated or winger is None:
            return "jockey"

        stam_mult = _stamina_mult(stamina_pct)
        cover_available, _ = profile.cover_behind_available(
            x, y, teammates or [], position_engine, attacks_right,
            exclude_name=getattr(profile, "_owner_name", None),
        )

        if cover_available:
            engage_prob = (0.35 + profile.duel_aggression * 0.55) * stam_mult
            if winger_pace_advantage:
                engage_prob *= 0.75   # respect the footrace even with cover
        else:
            # No cover — engaging is a gamble; only hyper-aggressive
            # fullbacks gamble, everyone else delays.
            engage_prob = profile.duel_aggression * 0.30 * stam_mult

        if random.random() < engage_prob:
            return "engage"
        if winger_pace_advantage and not cover_available:
            return "show_inside"
        return "jockey"

    @staticmethod
    def should_recovery_sprint(
        profile: FullbackSpatialProfile,
        fb_x: float, fb_y: float,
        ball_x: float, ball_y: float,
        attacks_right: bool,
        stamina_pct: float = 100.0,
        anchor_y: Optional[float] = None,
    ) -> bool:
        """
        Turnover down his flank with the fullback caught upfield: the
        emergency recovery sprint. Urgency must clear a bar scaled DOWN
        by fatigue (gassed fullbacks jog back — the classic conceded-
        goal clip) but a true emergency still forces maximum effort.
        """
        urgency = profile.recovery_priority(
            fb_x, fb_y, ball_x, ball_y, attacks_right,
            anchor_y=anchor_y,
        )
        stam_mult = _stamina_mult(stamina_pct)
        threshold = 0.55 - (1.0 - stam_mult) * 0.15
        return urgency >= threshold

    # ── PROFILE BUILDER ───────────────────────────────────────

    @staticmethod
    def build_profile_from_dna(player) -> FullbackSpatialProfile:
        """
        Build a FullbackSpatialProfile from a player's DNA archetype +
        tendencies + specialties. Called once at kickoff.

        DNA signals used:
            tendencies.cuts_inside           → tuck_instinct (inversion!)
            tendencies.sprints_frequently    → overlap appetite
            tendencies.holds_position        → hold_discipline
            tendencies.tackles_aggressively  → duel_aggression
            tendencies.crosses_from_wide     → delivery_instinct
            tendencies.presses_high          → advance/recovery appetite
            archetype                        → base profile shape
            specialties                      → explicit overrides
        """
        dna = getattr(player, "dna", None)
        position = getattr(player, "position", "RB")
        flank = "left" if position == "LB" else "right"
        anchor_y = LEFT_TOUCHLINE_ANCHOR_Y if flank == "left" else RIGHT_TOUCHLINE_ANCHOR_Y

        # Defaults: balanced modern fullback
        profile = FullbackSpatialProfile(
            flank=flank,
            touchline_anchor_y=anchor_y,
            flank_commitment=0.75,
            advance_instinct=0.50,
            overlap_instinct=0.45,
            underlap_instinct=0.35,
            tuck_instinct=0.10,
            delivery_instinct=0.35,
            hold_discipline=0.60,
            duel_aggression=0.50,
            recovery_urgency=0.65,
        )

        if dna is None:
            return profile

        tendencies = getattr(dna, "tendencies", None)
        if tendencies is not None:
            cuts = getattr(tendencies, "cuts_inside", 0.30)
            profile.tuck_instinct = max(0.02, min(0.95, (cuts - 0.30) * 2.2))

            sprints = getattr(tendencies, "sprints_frequently", 0.50)
            profile.overlap_instinct = max(0.15, min(0.95, 0.20 + sprints * 0.85))

            holds = getattr(tendencies, "holds_position", 0.60)
            profile.hold_discipline = max(0.15, min(0.95, holds))

            tackles = getattr(tendencies, "tackles_aggressively", 0.40)
            aggression = getattr(getattr(dna, "mental", None), "aggression", 55.0)
            profile.duel_aggression = max(0.15, min(0.95,
                0.25 + tackles * 0.60 + (aggression - 55.0) / 100.0))

            crosses = getattr(tendencies, "crosses_from_wide", 0.45)
            profile.delivery_instinct = max(0.15, min(0.90, 0.15 + crosses * 0.80))

            presses = getattr(tendencies, "presses_high", 0.40)
            work_rate = getattr(getattr(dna, "mental", None), "work_rate", 65.0)
            profile.advance_instinct = max(0.15, min(0.95,
                0.30 + presses * 0.55 + (work_rate - 65.0) / 120.0))
            profile.recovery_urgency = max(0.30, min(0.98,
                0.45 + (work_rate - 65.0) / 100.0 + presses * 0.25))

        # ── ARCHETYPE OVERRIDES ──
        archetype = getattr(dna, "archetype", "")
        if archetype == "attacking_fullback":
            profile.flank_commitment = max(profile.flank_commitment, 0.72)
            profile.advance_instinct = max(profile.advance_instinct, 0.78)
            profile.overlap_instinct = max(profile.overlap_instinct, 0.82)
            profile.underlap_instinct = max(profile.underlap_instinct, 0.45)
            profile.delivery_instinct = max(profile.delivery_instinct, 0.60)
            profile.hold_discipline = min(profile.hold_discipline, 0.45)
            profile.recovery_urgency = max(profile.recovery_urgency, 0.75)
        elif archetype == "inverted_fullback":
            profile.flank_commitment = min(profile.flank_commitment, 0.50)
            profile.tuck_instinct = max(profile.tuck_instinct, 0.75)
            profile.advance_instinct = max(profile.advance_instinct, 0.60)
            profile.underlap_instinct = max(profile.underlap_instinct, 0.55)
            profile.overlap_instinct = min(profile.overlap_instinct, 0.35)
            profile.delivery_instinct = min(profile.delivery_instinct, 0.30)
            profile.hold_discipline = min(profile.hold_discipline, 0.55)
        elif archetype == "defensive_fullback":
            profile.flank_commitment = max(profile.flank_commitment, 0.88)
            profile.advance_instinct = min(profile.advance_instinct, 0.30)
            profile.overlap_instinct = min(profile.overlap_instinct, 0.25)
            profile.underlap_instinct = min(profile.underlap_instinct, 0.20)
            profile.tuck_instinct = min(profile.tuck_instinct, 0.05)
            profile.delivery_instinct = min(profile.delivery_instinct, 0.25)
            profile.hold_discipline = max(profile.hold_discipline, 0.85)
            profile.duel_aggression = max(profile.duel_aggression, 0.65)
            profile.recovery_urgency = max(profile.recovery_urgency, 0.80)

        # ── SPECIALTY OVERRIDES ──
        specialties = getattr(dna, "specialties", []) or []
        if "overlapping_fullback" in specialties or "aggressive_fullback" in specialties:
            profile.advance_instinct = max(profile.advance_instinct, 0.80)
            profile.overlap_instinct = max(profile.overlap_instinct, 0.85)
            profile.delivery_instinct = max(profile.delivery_instinct, 0.55)
            profile.hold_discipline = min(profile.hold_discipline, 0.42)
        if "underlapping_fullback" in specialties:
            profile.underlap_instinct = max(profile.underlap_instinct, 0.80)
            profile.advance_instinct = max(profile.advance_instinct, 0.65)
            profile.tuck_instinct = max(profile.tuck_instinct, 0.30)
        if "defensive_fullback" in specialties:
            profile.advance_instinct = min(profile.advance_instinct, 0.32)
            profile.overlap_instinct = min(profile.overlap_instinct, 0.28)
            profile.hold_discipline = max(profile.hold_discipline, 0.85)
            profile.duel_aggression = max(profile.duel_aggression, 0.70)

        return profile


# ─────────────────────────────────────────────
# FULLBACK REGISTRY — per-team per-player profiles
# ─────────────────────────────────────────────

class FullbackRegistry:
    """
    Holds FullbackSpatialProfile for every fullback in a match.
    Built once at kickoff, read every touch by the event chain and
    position engine.
    """

    def __init__(self):
        self.profiles: Dict[str, FullbackSpatialProfile] = {}

    def register_player(self, player) -> Optional[FullbackSpatialProfile]:
        """Build + store a profile for a fullback. Returns the profile."""
        position = getattr(player, "position", "")
        if position not in ("LB", "RB"):
            return None
        profile = FullbackBehaviorEngine.build_profile_from_dna(player)
        name = getattr(player, "name", str(player))
        try:
            profile._owner_name = name
        except Exception:
            pass
        self.profiles[name] = profile
        return profile

    def register_team(self, players: List) -> None:
        """Register all fullbacks in a squad."""
        for p in players or []:
            self.register_player(p)

    def get(self, player_name: str) -> Optional[FullbackSpatialProfile]:
        return self.profiles.get(player_name)

    def is_fullback(self, player_name: str) -> bool:
        return player_name in self.profiles

    def remove(self, player_name: str) -> None:
        self.profiles.pop(player_name, None)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n🛡️  PLOFA 26/27 — Modern Fullback Behavior Engine Demo")
    print("=" * 64)

    class _TendA:  # attacking fullback
        cuts_inside = 0.20
        sprints_frequently = 0.75
        holds_position = 0.40
        tackles_aggressively = 0.45
        crosses_from_wide = 0.55
        presses_high = 0.55

    class _MentalA:
        aggression = 70.0
        work_rate = 85.0

    class _DNAA:
        archetype = "attacking_fullback"
        specialties = ["overlapping_fullback"]
        tendencies = _TendA()
        mental = _MentalA()

    class _TendI:  # inverted fullback
        cuts_inside = 0.65
        sprints_frequently = 0.45
        holds_position = 0.55
        tackles_aggressively = 0.30
        crosses_from_wide = 0.25
        presses_high = 0.40

    class _MentalI:
        aggression = 52.0
        work_rate = 74.0

    class _DNAI:
        archetype = "inverted_fullback"
        specialties = ["underlapping_fullback"]
        tendencies = _TendI()
        mental = _MentalI()

    class _TendD:  # defensive fullback
        cuts_inside = 0.20
        sprints_frequently = 0.40
        holds_position = 0.78
        tackles_aggressively = 0.55
        crosses_from_wide = 0.30
        presses_high = 0.30

    class _MentalD:
        aggression = 66.0
        work_rate = 70.0

    class _DNAD:
        archetype = "defensive_fullback"
        specialties = []
        tendencies = _TendD()
        mental = _MentalD()

    class _Player:
        def __init__(self, name, position, dna):
            self.name = name
            self.position = position
            self.dna = dna

    class _FakePE:
        def get_position(self, name):
            return {
                "Opp LW": (78.0, 54.0),   # their winger on our RB's flank
                "Opp CB": (70.0, 34.0),
                "Our CB": (30.0, 44.0),
                "Our CDM": (42.0, 36.0),
            }.get(name, (50.0, 34.0))

    class _Pl:
        def __init__(self, name, pos):
            self.name = name
            self.position = pos

    pe = _FakePE()
    attackers = [_Pl("Opp LW", "LW"), _Pl("Opp CB", "CB")]
    mates = [_Pl("Our CB", "CB"), _Pl("Our CDM", "CDM")]

    players = [
        _Player("Dario Vela", "LB", _DNAI()),
        _Player("Rico Alves", "RB", _DNAA()),
        _Player("Tom Kade", "RB", _DNAD()),
    ]
    registry = FullbackRegistry()
    registry.register_team(players)

    for pl in players:
        p = registry.get(pl.name)
        print(f"\n1. PROFILE — {pl.name} ({pl.position}, {pl.dna.archetype}):")
        print(f"   Flank:             {p.flank}")
        print(f"   Touchline anchor:  y={p.touchline_anchor_y}")
        print(f"   Flank commitment:  {p.flank_commitment:.2f}")
        print(f"   Advance instinct:  {p.advance_instinct:.2f}")
        print(f"   Overlap instinct:  {p.overlap_instinct:.2f}")
        print(f"   Underlap instinct: {p.underlap_instinct:.2f}")
        print(f"   Tuck instinct:     {p.tuck_instinct:.2f}")
        print(f"   Delivery instinct: {p.delivery_instinct:.2f}")
        print(f"   Hold discipline:   {p.hold_discipline:.2f}")
        print(f"   Duel aggression:   {p.duel_aggression:.2f}")
        print(f"   Recovery urgency:  {p.recovery_urgency:.2f}")

    rb = registry.get("Rico Alves")     # attacking, overlapping
    lb = registry.get("Dario Vela")     # inverted, tucking
    db = registry.get("Tom Kade")       # defensive, staying

    print("\n2. GEOMETRY CHECKS (attacking right, RB):")
    pts = [
        ("deep, wide",        40.0, 56.0),
        ("build-up zone",     50.0, 56.0),
        ("overlap corridor",  84.0, 57.0),
        ("underlap channel",  84.0, 46.0),
        ("drifted central",   84.0, 34.0),
        ("cutback station",   93.0, 56.0),
    ]
    print(f"   {'Location':<20} {'Overlap':>8} {'Underlp':>8} {'Cross':>6} {'Danger':>7}")
    for label, x, y in pts:
        print(f"   {label:<20} {str(rb.in_overlap_corridor(x, y, True)):>8} "
              f"{str(rb.in_underlap_channel(x, y, True)):>8} "
              f"{str(rb.in_cross_zone(x, y, True)):>6} "
              f"{rb.danger_zone_score(x, y, True):>7.2f}")

    print("\n3. ADVANCE DECISIONS (ball on his flank at x=75):")
    rolls = sum(
        FullbackBehaviorEngine.should_advance(rb, 55, 56, True, 75, 56)
        for _ in range(200)
    )
    rolls_db = sum(
        FullbackBehaviorEngine.should_advance(db, 55, 56, True, 75, 56)
        for _ in range(200)
    )
    print(f"   Attacking FB advances: {rolls}/200")
    print(f"   Defensive FB advances: {rolls_db}/200")
    modes = {}
    for _ in range(300):
        m = FullbackBehaviorEngine.choose_advance_mode(
            rb, 60.0, 56.0, True, 78.0, 56.0, attackers, pe,
        )
        modes[m] = modes.get(m, 0) + 1
    print(f"   Attacking FB advance modes (final-third ball): {modes}")

    print("\n4. INVERTED LB TUCK (build-up, ball at x=45):")
    tuck_hits = sum(
        1 for _ in range(200)
        if FullbackBehaviorEngine.choose_advance_mode(
            lb, 38.0, 12.0, True, 45.0, 20.0, attackers, pe,
        ) == "tuck"
    )
    print(f"   Tuck rate: {tuck_hits}/200, tuck_target_y={lb.tuck_target_y():.1f}")

    print("\n5. 1v1 DEFENSE (opp LW on our RB, cover behind):")
    for label, prof in (("attacking", rb), ("defensive", db)):
        decisions = {}
        for _ in range(300):
            d = FullbackBehaviorEngine.defend_engagement(
                prof, 84.0, 55.0, attackers, mates, pe, True,
            )
            decisions[d] = decisions.get(d, 0) + 1
        print(f"   {label:>10} FB vs winger: {decisions}")

    print("\n6. RECOVERY (turnover, ball behind RB on his flank):")
    urg = rb.recovery_priority(70.0, 56.0, 45.0, 52.0, True)
    sprint = FullbackBehaviorEngine.should_recovery_sprint(rb, 70.0, 56.0, 45.0, 52.0, True)
    print(f"   Urgency: {urg:.2f} → sprint={sprint}")

    print("\n7. DELIVERY TYPE at the cutback station (93m):")
    dtype = FullbackBehaviorEngine.decide_delivery_type(rb, 93.0, 56.0, True)
    print(f"   Type: {dtype}")

    print("\n✅ Modern Fullback Behavior Engine operational — pure geometry, zero deps.")
