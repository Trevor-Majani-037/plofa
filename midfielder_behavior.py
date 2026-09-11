"""
PLOFA 26/27 — MODERN MIDFIELDER BEHAVIOR ENGINE (Checkpoint 33)
===============================================================
midfielder_behavior.py

Philosophy:
    Central midfield is the connective tissue of a team. Unlike wingers
    (touchline runners) and fullbacks (overlap/tuck specialists), a
    midfielder's "position" is really a SET OF DUTIES that shift with
    the phase of play and the archetype:

        IN POSSESSION — BUILD-UP:
        1. DROP TO RECEIVE   — the pivot (regista / deep playmaker) drops
                               into the half-space between the centre-backs
                               to offer a safe outlet and break the first
                               line of press (the "split the CBs" pattern).
        2. CARRY FORWARD     — a progressive midfielder / box-to-box drives
                               through the lines into the half-space ahead
                               when the lane is open.
        3. POCKET ROAM       — a classic #10 operates BETWEEN the lines,
                               not in a fixed spot, constantly finding the
                               half-space seam the block leaves open.

        IN POSSESSION — FINAL THIRD:
        4. LATE BOX ARRIVAL  — box-to-box / shadow striker time a run from
                               deep onto the edge of / into the box (the
                               "arrives late" runner who isn't picked up).

        OUT OF POSSESSION:
        5. RECOVER / PRESS    — ball-winning types step up to press; holders
                               sit and protect the block.

    This module gives every CM/CAM a persistent profile built from DNA
    archetype + tendencies + specialties, and exposes PURE decision
    functions (read PositionEngine state, return a target / decision).
    The position engine consumes them via continuous, pace-capped,
    shape-aware run steps — same machinery as the winger/fullback
    engines, so midfield movement flows as trajectories, not snaps.

    Archetypes consumed (player_dna.ArchetypeLibrary):
        CM:  box_to_box, deep_playmaker, progressive_midfielder
        CAM: classic_ten, shadow_striker
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Optional pitch-control awareness (same dormancy pattern as the winger
# module): if pitch_control.py is wired into the per-minute loop later, the
# openness/through-ball checks corroborate real territory; until then they
# degrade gracefully to defender-geometry only, so current behavior is
# unchanged and the bonus path simply stays asleep.
try:
    from pitch_control import PitchControlField, PitchControlResult
    _HAS_PITCH_CONTROL = True
except ImportError:
    _HAS_PITCH_CONTROL = False

# Reuse the shared pitch frame (StatsBomb scale).
PITCH_Y = 68.0
CENTER_Y = 34.0
PITCH_X = 105.0

# Third gates (attacking-right frame).
MID_THIRD_MIN_X_ATT = 35.0
FINAL_THIRD_X_ATT = 70.0
BUILD_ZONE_X_ATT = 45.0          # own + deep-middle third for drop-to-receive
HALF_SPACE_WIDTH_M = 14.0
HALF_SPACE_DEF_RADIUS_M = 11.0
BOX_ENTRY_X_ATT = 82.0
POST_BAND_Y = (30.34, 37.66)      # six-yard box edge (near/far post channels)

# Fatigue banding — identical curve to winger/fullback so every role in
# the engine degrades on the SAME stamina math.
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
    return PITCH_X if attacks_right else 0.0


# ─────────────────────────────────────────────
# MIDFIELDER SPATIAL PROFILE
# ─────────────────────────────────────────────

@dataclass
class MidfielderSpatialProfile:
    """
    Persistent per-midfielder geometry/role identity. Built once at
    kickoff from DNA archetype + tendencies + specialties.

    Attributes:
        role: "cm" | "cam"
        build_duty: 0-1 — drops deep between the CBs to receive.
        carry_forward_instinct: 0-1 — drives through the lines.
        late_box_instinct: 0-1 — times late runs into the box.
        pocket_instinct: 0-1 — operates between the lines (#10 seam).
        delivery_instinct: 0-1 — through balls / switches.
        press_instinct: 0-1 — steps up to press out of possession.
        recovery_instinct: 0-1 — drops to protect the block OOP.
    """
    role: str = "cm"
    build_duty: float = 0.45
    carry_forward_instinct: float = 0.45
    late_box_instinct: float = 0.35
    pocket_instinct: float = 0.40
    delivery_instinct: float = 0.45
    press_instinct: float = 0.45
    recovery_instinct: float = 0.55
    orbit_instinct: float = 0.50

    # ── ZONE HELPERS ────────────────────────────────────────

    def in_build_zone(self, x: float, attacks_right: bool) -> bool:
        return x < BUILD_ZONE_X_ATT if attacks_right else x > (PITCH_X - BUILD_ZONE_X_ATT)

    def in_final_third(self, x: float, attacks_right: bool) -> bool:
        return x > FINAL_THIRD_X_ATT if attacks_right else x < (PITCH_X - FINAL_THIRD_X_ATT)

    def local_openness(
        self, x: float, y: float,
        defenders: Optional[List], position_engine,
        pitch_control_result=None, pitch_control_field=None,
        attacks_right: bool = True,
    ) -> float:
        """Openness (1 open → 0 walled) around (x, y) from real defender
        DENSITY — counts every body inside the closure radius and compounds
        for a second occupant, mirroring the winger module's
        half_space_openness so the two engines agree. Optionally
        corroborated by pitch control when that layer is live."""
        if position_engine is None or not defenders:
            return 0.6
        closest = float("inf")
        occupants = 0
        for d in defenders:
            if getattr(d, "position", None) == "GK":
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, dy = position_engine.get_position(dname)
            dist = math.hypot(dx - x, dy - y)
            if dist < HALF_SPACE_DEF_RADIUS_M:
                occupants += 1
            if dist < closest:
                closest = dist
        if closest == float("inf"):
            base = 1.0
        else:
            base = max(0.0, min(1.0, closest / HALF_SPACE_DEF_RADIUS_M))
        # A second body in the corridor closes it far more than linearly.
        base -= 0.15 * max(0, occupants - 1)
        base = max(0.0, min(1.0, base))
        if _HAS_PITCH_CONTROL and pitch_control_result is not None and pitch_control_field is not None:
            owner = pitch_control_field.cell_ownership(pitch_control_result, x, y)
            attacking_owner = "home" if attacks_right else "away"
            pc = 0.85 if owner == attacking_owner else (0.55 if owner == "neutral" else 0.15)
            return round(base * 0.65 + pc * 0.35, 3)
        return round(base, 3)

    def space_ahead(
        self, x: float, y: float, attacks_right: bool,
        defenders: Optional[List], position_engine,
        pitch_control_result=None, pitch_control_field=None,
    ) -> float:
        """Room AHEAD of the midfielder for a carry (1 open → 0 walled)."""
        if position_engine is None or not defenders:
            return 0.6
        sign = 1.0 if attacks_right else -1.0
        probe_x = x + sign * 12.0
        return self.local_openness(
            probe_x, y, defenders, position_engine,
            pitch_control_result, pitch_control_field, attacks_right)

    def drop_openness(
        self, attacks_right: bool, anchor_y: float,
        defenders: Optional[List], position_engine,
        pitch_control_result=None, pitch_control_field=None,
    ) -> float:
        """Openness of the ACTUAL DROP pocket (between the CBs), not the
        space ahead of the current position. A pivot must read the pocket
        he's dropping INTO, not the lane he's leaving."""
        tx, ty = self.drop_target(attacks_right, anchor_y)
        return self.local_openness(
            tx, ty, defenders, position_engine,
            pitch_control_result, pitch_control_field, attacks_right)

    def forward_lane_openness(
        self, x: float, y: float, attacks_right: bool,
        defenders: Optional[List], position_engine,
        pitch_control_result=None, pitch_control_field=None,
    ) -> float:
        """Openness of the forward passing lane toward goal — used to judge
        whether a through ball / switch has a corridor to be played into."""
        sign = 1.0 if attacks_right else -1.0
        probe_x = x + sign * 18.0
        return self.local_openness(
            probe_x, y, defenders, position_engine,
            pitch_control_result, pitch_control_field, attacks_right)

    def press_target(
        self, x: float, y: float,
        attackers: Optional[List], position_engine,
        attacks_right: bool = True,
    ) -> Tuple[float, float]:
        """Where a pressing midfielder steps to: onto the nearest opponent
        carrier, arriving ~2m short (a real pressing distance, not a
        teleport onto his toes)."""
        best = None
        best_d = float("inf")
        for p in attackers or []:
            if getattr(p, "position", None) == "GK":
                continue
            pname = getattr(p, "name", None)
            if pname is None:
                continue
            px, py = position_engine.get_position(pname)
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d = d
                best = (px, py)
        if best is None:
            return (x, y)
        bx, by = best
        dx, dy = bx - x, by - y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return (bx, by)
        stop = max(0.0, dist - 2.0)
        return (x + dx / dist * stop, y + dy / dist * stop)

    def recover_target(
        self, attacks_right: bool, anchor_y: float,
    ) -> Tuple[float, float]:
        """Protective screen: drop into the block just in front of the
        centre-backs on the midfielder's own side of the pitch."""
        tx = 32.0 if attacks_right else PITCH_X - 32.0
        return (tx, anchor_y)

    def drop_target(self, attacks_right: bool, anchor_y: float) -> Tuple[float, float]:
        """Where the pivot drops to receive — own third, half-space seam."""
        sign = 1.0 if attacks_right else -1.0
        tx = 32.0 if attacks_right else PITCH_X - 32.0
        # Half-space lane next to his flank side of centre.
        direction = 1.0 if anchor_y <= CENTER_Y else -1.0
        ty = CENTER_Y + direction * (HALF_SPACE_WIDTH_M * 0.6)
        return tx, ty

    def pocket_target(self, ball_x: float, ball_y: float, attacks_right: bool) -> Tuple[float, float]:
        """The #10 seam: ahead of the ball, between the lines."""
        sign = 1.0 if attacks_right else -1.0
        tx = max(45.0, min(80.0, ball_x + sign * 16.0))
        ty = ball_y + (CENTER_Y - ball_y) * 0.25
        return tx, ty

    def box_arrival_target(self, ball_x: float, ball_y: float, attacks_right: bool) -> Tuple[float, float]:
        """Late run onto the edge/inside the box (edge of the seam)."""
        sign = 1.0 if attacks_right else -1.0
        tx = max(BOX_ENTRY_X_ATT, min(94.0, ball_x + sign * 6.0)) if attacks_right \
            else min(PITCH_X - BOX_ENTRY_X_ATT, max(11.0, ball_x + sign * 6.0))
        # Arrive at the far-side post channel from the ball.
        ty = POST_BAND_Y[1] if ball_y < CENTER_Y else POST_BAND_Y[0]
        return tx, ty

    def orbit_target(
        self, x: float, y: float, ball_x: float, ball_y: float,
        attacks_right: bool, anchor_y: float,
        block_channels: Optional[List] = None,
        defenders: Optional[List] = None,
        position_engine=None,
    ) -> Tuple[float, float]:
        """Far-side open channel relative to the ball — the orbital node.

        When the ball is on one side of the pitch, the CM/CDM who is NOT
        the carrier drifts to the open channel on the OPPOSITE side so
        the circulation web always has a receiver on the far edge of the
        block.  If block channels are available the most accessible one
        on the far side of the ball is used; otherwise a geometric
        fallback places the node in the far half-space."""
        sign = 1.0 if attacks_right else -1.0

        # Determine which side of the pitch the ball is on
        ball_side = 1.0 if ball_y >= CENTER_Y else -1.0

        # If we have block channels, pick the best far-side one
        best_ch = None
        best_score = -1.0
        if block_channels:
            for ch in block_channels:
                ch_cx, ch_cy = ch.center
                ch_side = 1.0 if ch_cy >= CENTER_Y else -1.0
                # Must be on the opposite side of the ball
                if ch_side == ball_side:
                    continue
                if ch.accessibility < 0.15:
                    continue
                # Score: accessibility weighted by distance from current pos
                # (farther = more "orbital" but not too far)
                dist = math.hypot(ch_cx - x, ch_cy - y)
                if dist < 5.0:
                    continue  # Too close — not really orbiting
                score = ch.accessibility * min(1.0, dist / 20.0)
                if score > best_score:
                    best_score = score
                    best_ch = ch

        if best_ch is not None:
            tx, ty = best_ch.center
            # Don't go all the way to the channel center — blend 70% toward it
            tx = x + (tx - x) * 0.70
            ty = y + (ty - y) * 0.70
        else:
            # Geometric fallback: mirror to the far half-space
            mirror_y = CENTER_Y + (CENTER_Y - ball_y) * 0.8
            mirror_y = max(6.0, min(62.0, mirror_y))
            # Stay at a reasonable depth — same x-band as current or slightly
            # ahead/behind depending on build zone
            tx = max(30.0, min(80.0, x + sign * 4.0))
            ty = mirror_y

        # Clamp to pitch
        tx = max(8.0, min(PITCH_X - 8.0, tx))
        ty = max(4.0, min(PITCH_Y - 4.0, ty))
        return (tx, ty)

    # ── COVER / PRESS GEOMETRY ──────────────────────────────

    def press_trigger(
        self, x: float, y: float,
        attackers: Optional[List], position_engine,
        attacks_right: bool = True,
    ) -> bool:
        """Is an opponent carrier within this midfielder's press radius?"""
        if position_engine is None or not attackers:
            return False
        for p in attackers:
            if getattr(p, "position", None) == "GK":
                continue
            pname = getattr(p, "name", None)
            if pname is None:
                continue
            px, py = position_engine.get_position(pname)
            if math.hypot(px - x, py - y) < 9.0:
                return True
        return False


# ─────────────────────────────────────────────
# MIDFIELDER BEHAVIOR ENGINE
# ─────────────────────────────────────────────

class MidfielderBehaviorEngine:
    """
    Pure decision engine for modern midfield play. Stateless — the
    profile is passed in per call.
    """

    @staticmethod
    def should_drop_to_receive(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        under_pressure: bool = False,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: float = CENTER_Y,
        stamina_pct: float = 100.0,
    ) -> bool:
        """Drop between the CBs to offer a build-up outlet."""
        if not profile.in_build_zone(x, attacks_right) and not profile.in_build_zone(ball_x, attacks_right):
            return False
        if under_pressure and profile.build_duty < 0.6:
            return False
        stam = _stamina_mult(stamina_pct)
        # Only drop if there's actually a pocket of space to drop into.
        if defenders is not None and position_engine is not None:
            open_space = profile.drop_openness(attacks_right, anchor_y, defenders, position_engine)
            if open_space < 0.25:
                return False
        # Under pressure even a high build-duty pivot hesitates (press-risk).
        pressure_mult = 0.6 if under_pressure else 1.0
        prob = (profile.build_duty * 0.7 + profile.delivery_instinct * 0.15) * stam * pressure_mult
        return random.random() < prob

    @staticmethod
    def should_carry_forward(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: float = CENTER_Y,
        stamina_pct: float = 100.0,
    ) -> bool:
        """Drive through the lines into the half-space ahead."""
        stam = _stamina_mult(stamina_pct)
        open_ahead = profile.space_ahead(x, y, attacks_right, defenders, position_engine) \
            if (defenders and position_engine) else 0.6
        prob = (profile.carry_forward_instinct * 0.6 + open_ahead * 0.3) * stam
        return random.random() < prob

    @staticmethod
    def should_arrive_late(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_in_final_third: bool = False,
        stamina_pct: float = 100.0,
    ) -> bool:
        """Time a late run into the box from deep."""
        if not ball_in_final_third:
            return False
        stam = _stamina_mult(stamina_pct)
        prob = profile.late_box_instinct * 0.7 * stam
        return random.random() < prob

    @staticmethod
    def should_orbit(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        defenders: Optional[List] = None,
        position_engine=None,
        block_channels: Optional[List] = None,
        stamina_pct: float = 100.0,
    ) -> bool:
        """Drift to the far-side open channel — the orbital node duty.

        When the ball is circulating on one side and this CM/CDM is idle
        (not carrying, not dropping, not arriving late), they reposition
        to the opposite side of the block so the circulation web always
        has a far-edge receiver.  This is what generates the sequential
        half-space → deep-recycle → half-space pass chain on Modric maps.

        Only fires when:
          - The ball is in the mid-third (circulation, not final-third)
          - The player is NOT already in an open channel
          - The player's position is on the SAME side as the ball
            (so moving to the far side is meaningful)
        """
        stam = _stamina_mult(stamina_pct)
        ball_mid = (MID_THIRD_MIN_X_ATT <= ball_x <= FINAL_THIRD_X_ATT) if attacks_right \
            else ((PITCH_X - FINAL_THIRD_X_ATT) <= ball_x <= (PITCH_X - MID_THIRD_MIN_X_ATT))
        if not ball_mid:
            return False

        # Check if the player is already in an open half-space channel
        if block_channels:
            for ch in block_channels:
                if "half_space" not in ch.name:
                    continue
                if ch.accessibility < 0.30:
                    continue
                cx, cy = ch.center
                if math.hypot(x - cx, y - cy) < ch.width * 0.8:
                    return False  # Already in a good orbital node — hold

        # Is the player on the same side as the ball?
        ball_side = 1.0 if ball_y >= CENTER_Y else -1.0
        player_side = 1.0 if y >= CENTER_Y else -1.0
        same_side = (ball_side == player_side)

        prob = profile.orbit_instinct * 0.55 * stam
        if same_side:
            prob *= 1.35  # Stronger pull when player mirrors the ball
        return random.random() < prob

    @staticmethod
    def decide_run(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        in_possession: bool = True,
        under_pressure: bool = False,
        defenders: Optional[List] = None,
        attackers: Optional[List] = None,
        position_engine=None,
        anchor_y: float = CENTER_Y,
        stamina_pct: float = 100.0,
        block_channels: Optional[List] = None,
    ) -> Optional[str]:
        """
        Which run does this midfielder make right now?
            "drop"   — build-up pivot drops to receive
            "carry"  — progressive carry through the lines
            "pocket" — #10 operates between the lines
            "late"   — late box arrival
            "orbit"  — drift to far-side open channel (orbital node)
            None     — hold (no committed run)
        """
        if not in_possession:
            return None
        stam = _stamina_mult(stamina_pct)
        ball_final = profile.in_final_third(ball_x, attacks_right)

        if profile.role == "cam":
            # Classic #10: pocket-first; shadow striker leans late box.
            if ball_final and profile.late_box_instinct > 0.45:
                if random.random() < profile.late_box_instinct * 0.7 * stam:
                    return "late"
            if profile.pocket_instinct > 0.30:
                if random.random() < (profile.pocket_instinct * 0.6 + 0.2) * stam:
                    return "pocket"
            return None

        # CM: duty-driven.
        if MidfielderBehaviorEngine.should_drop_to_receive(
                profile, x, y, attacks_right, ball_x, ball_y,
                under_pressure=under_pressure, defenders=defenders,
                position_engine=position_engine, anchor_y=anchor_y,
                stamina_pct=stamina_pct):
            return "drop"
        if ball_final and MidfielderBehaviorEngine.should_arrive_late(
                profile, x, y, attacks_right, ball_in_final_third=True,
                stamina_pct=stamina_pct):
            return "late"
        if MidfielderBehaviorEngine.should_carry_forward(
                profile, x, y, attacks_right, defenders=defenders,
                position_engine=position_engine, anchor_y=anchor_y,
                stamina_pct=stamina_pct):
            return "carry"
        if MidfielderBehaviorEngine.should_orbit(
                profile, x, y, attacks_right, ball_x, ball_y,
                defenders=defenders, position_engine=position_engine,
                block_channels=block_channels, stamina_pct=stamina_pct):
            return "orbit"
        return None

    @staticmethod
    def carry_direction_bias(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        in_possession: bool = True,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: float = CENTER_Y,
    ) -> float:
        """Lateral bias for a midfield carry (toward the half-space seam)."""
        if defenders and position_engine is not None:
            # Aim for the half-space lane beside centre.
            target_y = (CENTER_Y + HALF_SPACE_WIDTH_M * 0.5) if anchor_y <= CENTER_Y \
                else (CENTER_Y - HALF_SPACE_WIDTH_M * 0.5)
            return (target_y - y) * 0.25
        return 0.0

    @staticmethod
    def carry_target(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        anchor_y: float,
        defenders: Optional[List] = None,
        position_engine=None,
        pitch_control_result=None,
        pitch_control_field=None,
    ) -> Tuple[float, float]:
        """Forward carry destination: 12m up the pitch, bent toward the
        half-space seam on this midfielder's side via carry_direction_bias
        (so the previously-orphaned bias method now actually steers)."""
        sign = 1.0 if attacks_right else -1.0
        tx = max(30.0, min(82.0, x + sign * 12.0))
        bias = MidfielderBehaviorEngine.carry_direction_bias(
            profile, x, y, attacks_right, in_possession=True,
            defenders=defenders, position_engine=position_engine,
            anchor_y=anchor_y)
        ty = y + bias
        return (tx, ty)

    @staticmethod
    def decide_defensive_role(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        attackers: Optional[List], position_engine,
        stamina_pct: float = 100.0,
    ) -> str:
        """
        Out-of-possession assignment for a CM/CAM:
            "press"   — ball-winner steps onto the carrier (press_instinct).
            "recover" — holder drops to screen the block (recovery_instinct).
            "hold"    — stays in shape.
        This is the live consumer of the profile's press/recovery fields,
        which were previously built but never read.
        """
        stam = _stamina_mult(stamina_pct)
        triggered = profile.press_trigger(x, y, attackers, position_engine, attacks_right)
        if triggered and profile.press_instinct > 0.45:
            if random.random() < profile.press_instinct * 0.85 * stam:
                return "press"
        own_half = (ball_x < 52.5) if attacks_right else (ball_x > 52.5)
        if profile.recovery_instinct >= profile.press_instinct and own_half:
            if random.random() < profile.recovery_instinct * 0.70 * stam:
                return "recover"
        return "hold"

    @staticmethod
    def should_play_through_ball(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        defenders: Optional[List] = None,
        position_engine=None,
        stamina_pct: float = 100.0,
        pitch_control_result=None,
        pitch_control_field=None,
    ) -> bool:
        """Does this midfielder's delivery instinct + an open forward lane
        favour a through ball right now?"""
        stam = _stamina_mult(stamina_pct)
        lane = profile.forward_lane_openness(
            x, y, attacks_right, defenders, position_engine,
            pitch_control_result, pitch_control_field) \
            if (defenders and position_engine) else 0.6
        prob = (profile.delivery_instinct * 0.6 + lane * 0.3) * stam
        return random.random() < prob

    @staticmethod
    def should_switch(
        profile: MidfielderSpatialProfile,
        x: float, y: float, attacks_right: bool,
        defenders: Optional[List] = None,
        position_engine=None,
        stamina_pct: float = 100.0,
    ) -> bool:
        """A patient playmaker switching the point of attack."""
        stam = _stamina_mult(stamina_pct)
        prob = profile.delivery_instinct * 0.5 * stam
        return random.random() < prob

    @staticmethod
    def cam_pocket_target(
        profile: MidfielderSpatialProfile,
        ball_x: float, ball_y: float, attacks_right: bool,
    ) -> Tuple[float, float]:
        """The #10 seam from the profile's own pocket geometry, nudged
        goal-side for shadow strikers. Consumed by position_engine's
        _cam_pocket_roam so pocket_target() is no longer dead code."""
        tx, ty = profile.pocket_target(ball_x, ball_y, attacks_right)
        if profile.late_box_instinct > 0.50:
            sign = 1.0 if attacks_right else -1.0
            tx = max(45.0, min(96.0, tx + sign * 8.0))
        return (tx, ty)

    @staticmethod
    def build_profile_from_dna(player) -> MidfielderSpatialProfile:
        """Build a MidfielderSpatialProfile from DNA archetype + tendencies."""
        dna = getattr(player, "dna", None)
        position = getattr(player, "position", "CM")
        role = "cam" if position == "CAM" else "cm"

        profile = MidfielderSpatialProfile(role=role)

        if dna is None:
            return profile

        tendencies = getattr(dna, "tendencies", None)
        mental = getattr(dna, "mental", None)
        if tendencies is not None:
            arrives = getattr(tendencies, "arrives_late", 0.20)
            profile.late_box_instinct = max(0.05, min(0.95, arrives * 1.6))
            behind = getattr(tendencies, "makes_runs_behind", 0.35)
            profile.carry_forward_instinct = max(0.15, min(0.90, 0.25 + behind * 0.7))
            through = getattr(tendencies, "plays_through_ball", 0.10)
            profile.delivery_instinct = max(0.15, min(0.90, 0.20 + through * 1.4))
            holds = getattr(tendencies, "holds_position", 0.60)
            profile.recovery_instinct = max(0.20, min(0.95, holds + 0.1))
            presses = getattr(tendencies, "presses_high", 0.40)
            profile.press_instinct = max(0.15, min(0.95, presses))
            # build duty: deep playmakers are patient; progressive types less so
            profile.build_duty = max(0.15, min(0.90, 0.30 + holds * 0.4 - through * 0.2))
            # pocket for CAM types: vision-heavy
            vision = getattr(mental, "vision", 60.0) if mental else 60.0
            profile.pocket_instinct = max(0.20, min(0.95, 0.25 + (vision - 55.0) / 100.0))
            # orbit: patient recyclers + vision-heavy playmakers drift to
            # far-side channels; aggressive carriers stay low
            work_rate = getattr(mental, "work_rate", 60.0) if mental else 60.0
            profile.orbit_instinct = max(0.20, min(0.85,
                0.25 + holds * 0.30 + (vision - 55.0) / 200.0 + (work_rate - 55.0) / 200.0))

        archetype = getattr(dna, "archetype", "")
        if role == "cam":
            if archetype == "classic_ten":
                profile.pocket_instinct = max(profile.pocket_instinct, 0.80)
                profile.delivery_instinct = max(profile.delivery_instinct, 0.75)
                profile.late_box_instinct = max(profile.late_box_instinct, 0.45)
            elif archetype == "shadow_striker":
                profile.late_box_instinct = max(profile.late_box_instinct, 0.80)
                profile.pocket_instinct = min(profile.pocket_instinct, 0.45)
                profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.55)
        else:  # CM
            if archetype == "box_to_box":
                profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.70)
                profile.late_box_instinct = max(profile.late_box_instinct, 0.65)
                profile.press_instinct = max(profile.press_instinct, 0.65)
                profile.recovery_instinct = max(profile.recovery_instinct, 0.70)
                profile.orbit_instinct = max(profile.orbit_instinct, 0.55)
            elif archetype == "deep_playmaker":
                profile.build_duty = max(profile.build_duty, 0.85)
                profile.delivery_instinct = max(profile.delivery_instinct, 0.80)
                profile.pocket_instinct = max(profile.pocket_instinct, 0.55)
                profile.carry_forward_instinct = min(profile.carry_forward_instinct, 0.35)
                profile.orbit_instinct = max(profile.orbit_instinct, 0.75)
            elif archetype == "progressive_midfielder":
                profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.78)
                profile.build_duty = min(profile.build_duty, 0.45)
                profile.delivery_instinct = max(profile.delivery_instinct, 0.65)
                profile.orbit_instinct = min(profile.orbit_instinct, 0.40)

        # Specialty nudges
        specs = getattr(dna, "specialties", []) or []
        if "engine" in specs or "box_box" in specs:
            profile.late_box_instinct = max(profile.late_box_instinct, 0.65)
            profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.70)
        if "playmaker" in specs or "dl_playmaker" in specs:
            profile.build_duty = max(profile.build_duty, 0.82)
            profile.delivery_instinct = max(profile.delivery_instinct, 0.78)
            profile.orbit_instinct = max(profile.orbit_instinct, 0.70)
        if "ball_progressor" in specs:
            profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.80)
        if "creator" in specs or "grand_creator" in specs:
            profile.pocket_instinct = max(profile.pocket_instinct, 0.82)
            profile.delivery_instinct = max(profile.delivery_instinct, 0.80)
        if "deep_lying_forward" in specs or "late_runner" in specs:
            profile.late_box_instinct = max(profile.late_box_instinct, 0.78)

        return profile


# ─────────────────────────────────────────────
# MIDFIELDER REGISTRY
# ─────────────────────────────────────────────

class MidfieldRegistry:
    """Holds MidfielderSpatialProfile per CM/CAM in a match."""

    def __init__(self):
        self.profiles: Dict[str, MidfielderSpatialProfile] = {}

    def register_player(self, player) -> Optional[MidfielderSpatialProfile]:
        position = getattr(player, "position", "")
        if position not in ("CM", "CAM"):
            return None
        profile = MidfielderBehaviorEngine.build_profile_from_dna(player)
        name = getattr(player, "name", str(player))
        self.profiles[name] = profile
        return profile

    def register_team(self, players: List) -> None:
        for p in players or []:
            self.register_player(p)

    def get(self, player_name: str) -> Optional[MidfielderSpatialProfile]:
        return self.profiles.get(player_name)

    def is_midfielder(self, player_name: str) -> bool:
        return player_name in self.profiles

    def remove(self, player_name: str) -> None:
        self.profiles.pop(player_name, None)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n⚙️  PLOFA 26/27 — Modern Midfielder Behavior Engine Demo")
    print("=" * 64)

    class _Tend:
        arrives_late = 0.45
        makes_runs_behind = 0.40
        plays_through_ball = 0.20
        holds_position = 0.70
        presses_high = 0.55

    class _Mental:
        vision = 78.0
        work_rate = 80.0

    class _DNA:
        archetype = "box_to_box"
        specialties = ["engine"]
        tendencies = _Tend()
        mental = _Mental()

    class _TendP:
        arrives_late = 0.20
        makes_runs_behind = 0.30
        plays_through_ball = 0.30
        holds_position = 0.85
        presses_high = 0.30

    class _MentalP:
        vision = 84.0
        work_rate = 70.0

    class _DNAP:
        archetype = "deep_playmaker"
        specialties = ["playmaker"]
        tendencies = _TendP()
        mental = _MentalP()

    class _TendT:
        arrives_late = 0.60
        makes_runs_behind = 0.35
        plays_through_ball = 0.25
        holds_position = 0.55
        presses_high = 0.45

    class _MentalT:
        vision = 82.0
        work_rate = 75.0

    class _DNAT:
        archetype = "classic_ten"
        specialties = ["creator"]
        tendencies = _TendT()
        mental = _MentalT()

    class _Player:
        def __init__(self, name, position, dna):
            self.name = name
            self.position = position
            self.dna = dna

    players = [
        _Player("Rice", "CM", _DNA()),
        _Player("Modric", "CM", _DNAP()),
        _Player("Bruno", "CAM", _DNAT()),
    ]
    registry = MidfieldRegistry()
    registry.register_team(players)

    for pl in players:
        p = registry.get(pl.name)
        print(f"\n1. PROFILE — {pl.name} ({pl.position}, {pl.dna.archetype}):")
        print(f"   role:                 {p.role}")
        print(f"   build_duty:          {p.build_duty:.2f}")
        print(f"   carry_forward:       {p.carry_forward_instinct:.2f}")
        print(f"   late_box:            {p.late_box_instinct:.2f}")
        print(f"   pocket_instinct:     {p.pocket_instinct:.2f}")
        print(f"   delivery_instinct:   {p.delivery_instinct:.2f}")
        print(f"   press_instinct:      {p.press_instinct:.2f}")

    import types
    class _FakePE:
        def get_position(self, name):
            return {"Opp CB": (70.0, 34.0), "Opp CM": (60.0, 40.0)}.get(name, (50.0, 34.0))
    pe = _FakePE()
    attackers = [type("P", (), {"name": "Opp CM", "position": "CM"})()]
    defenders = [type("P", (), {"name": "Opp CB", "position": "CB"})(),
                 type("P", (), {"name": "Opp CM", "position": "CM"})()]

    print("\n2. RUN DECISIONS (attacking right):")
    rice = registry.get("Rice")
    modric = registry.get("Modric")
    bruno = registry.get("Bruno")
    for label, prof in (("box_to_box CM", rice), ("deep_playmaker CM", modric), ("classic_ten CAM", bruno)):
        drops = sum(1 for _ in range(200)
                    if MidfielderBehaviorEngine.decide_run(
                        prof, 40.0, 34.0, True, 40.0, 30.0, in_possession=True,
                        defenders=defenders, position_engine=pe, anchor_y=34.0))
        carries = sum(1 for _ in range(200)
                       if MidfielderBehaviorEngine.should_carry_forward(
                           prof, 55.0, 34.0, True, defenders=defenders, position_engine=pe))
        print(f"   {label:<20} drop-runs={drops}/200, carry-runs={carries}/200")

    print("\n3. CAM POCKET vs LATE (ball in final third at x=78):")
    late = sum(1 for _ in range(200)
               if MidfielderBehaviorEngine.decide_run(
                   bruno, 70.0, 34.0, True, 78.0, 30.0, in_possession=True,
                   defenders=defenders, position_engine=pe, anchor_y=34.0) == "late")
    print(f"   classic_ten late-box rate: {late}/200")

    print("\n4. OUT-OF-POSSESSION ROLES + DELIVERY:")
    # Press scenario: an opponent carrier sitting right on the midfielder.
    pe_press = _FakePE()
    pe_press.get_position = lambda n: (48.0, 34.0)
    attackers_near = [type("P", (), {"name": "Opp CM", "position": "CM"})()]
    b2b = registry.get("Rice")     # ball-winner
    dp = registry.get("Modric")    # holder
    roles_b2b = {"press": 0, "recover": 0, "hold": 0}
    roles_dp = {"press": 0, "recover": 0, "hold": 0}
    for _ in range(400):
        r = MidfielderBehaviorEngine.decide_defensive_role(
            b2b, 45.0, 34.0, True, 40.0, 34.0, attackers_near, pe_press)
        roles_b2b[r] += 1
        r = MidfielderBehaviorEngine.decide_defensive_role(
            dp, 45.0, 34.0, True, 40.0, 34.0, attackers_near, pe_press)
        roles_dp[r] += 1
    print(f"   box_to_box (presser):   {roles_b2b}")
    print(f"   deep_playmaker (holder): {roles_dp}")
    tb = sum(1 for _ in range(200)
             if MidfielderBehaviorEngine.should_play_through_ball(
                 dp, 55.0, 34.0, True, defenders=defenders, position_engine=pe))
    print(f"   deep_playmaker through-ball rate: {tb}/200")
    ct = MidfielderBehaviorEngine.carry_target(dp, 50.0, 34.0, True, 34.0,
                                               defenders=defenders, position_engine=pe)
    print(f"   deep_playmaker carry_target: {ct}")

    print("\n✅ Modern Midfielder Behavior Engine operational — pure geometry, zero deps.")
