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

    # ── ZONE HELPERS ────────────────────────────────────────

    def in_build_zone(self, x: float, attacks_right: bool) -> bool:
        return x < BUILD_ZONE_X_ATT if attacks_right else x > (PITCH_X - BUILD_ZONE_X_ATT)

    def in_final_third(self, x: float, attacks_right: bool) -> bool:
        return x > FINAL_THIRD_X_ATT if attacks_right else x < (PITCH_X - FINAL_THIRD_X_ATT)

    def space_ahead(
        self, x: float, y: float, attacks_right: bool,
        defenders: Optional[List], position_engine,
    ) -> float:
        """Room ahead of the midfielder for a carry (1 open → 0 walled)."""
        if position_engine is None or not defenders:
            return 0.6
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
        return max(0.0, min(1.0, closest / HALF_SPACE_DEF_RADIUS_M))

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
            open_space = profile.space_ahead(x, y, attacks_right, defenders, position_engine)
            if open_space < 0.25:
                return False
        prob = (profile.build_duty * 0.7 + profile.delivery_instinct * 0.15) * stam
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
    ) -> Optional[str]:
        """
        Which run does this midfielder make right now?
            "drop"   — build-up pivot drops to receive
            "carry"  — progressive carry through the lines
            "pocket" — #10 operates between the lines
            "late"   — late box arrival
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
            elif archetype == "deep_playmaker":
                profile.build_duty = max(profile.build_duty, 0.85)
                profile.delivery_instinct = max(profile.delivery_instinct, 0.80)
                profile.pocket_instinct = max(profile.pocket_instinct, 0.55)
                profile.carry_forward_instinct = min(profile.carry_forward_instinct, 0.35)
            elif archetype == "progressive_midfielder":
                profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.78)
                profile.build_duty = min(profile.build_duty, 0.45)
                profile.delivery_instinct = max(profile.delivery_instinct, 0.65)

        # Specialty nudges
        specs = getattr(dna, "specialties", []) or []
        if "engine" in specs or "box_box" in specs:
            profile.late_box_instinct = max(profile.late_box_instinct, 0.65)
            profile.carry_forward_instinct = max(profile.carry_forward_instinct, 0.70)
        if "playmaker" in specs or "dl_playmaker" in specs:
            profile.build_duty = max(profile.build_duty, 0.82)
            profile.delivery_instinct = max(profile.delivery_instinct, 0.78)
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

    print("\n✅ Modern Midfielder Behavior Engine operational — pure geometry, zero deps.")
