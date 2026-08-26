"""
PLOFA 26/27 — MODERN STRIKER BEHAVIOR ENGINE (Checkpoint 33)
=============================================================
striker_behavior.py

Philosophy:
    The striker is the tip of the spear, but "striker" is NOT one
    behaviour — it's a family of movements that depend on archetype:

        1. RUN IN BEHIND     — poacher / speedster time runs onto
                               through balls, stretching the last line
                               and forcing the offside trap to step.
        2. HOLD / LINK       — target man / deep-lying striker drop off
                               the front to receive, lay off, and bring
                               wide men + the #10 into play.
        3. PIN THE LAST LINE — poacher stays HIGH, never dropping, so the
                               defence can't step up and compress space.
        4. BOX ARRIVAL       — when the ball is wide / in the box, the
                               striker attacks the near or far post
                               channel (near-post flick / far-post tap-in).

    This module gives every ST/CF a persistent profile from DNA
    archetype + tendencies + specialties, and PURE decision functions
    consumed by the position engine's continuous, pace-capped,
    shape-aware run step.

    Archetypes consumed (player_dna.ArchetypeLibrary):
        poacher, target_man, complete_striker, deep_lying_striker,
        speedster_striker
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

PITCH_Y = 68.0
CENTER_Y = 34.0
PITCH_X = 105.0

FINAL_THIRD_X_ATT = 70.0
BOX_ENTRY_X_ATT = 82.0
POST_BAND_Y = (30.34, 37.66)      # six-yard box edge (near/far post channels)
LAST_LINE_PRESSURE_M = 12.0        # distance to the last defender for an in-behind gap

# Same stamina curve as the other wide/mid engines.
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
# STRIKER SPATIAL PROFILE
# ─────────────────────────────────────────────

@dataclass
class StrikerSpatialProfile:
    """
    Persistent per-striker geometry/role identity. Built once at kickoff.

    Attributes:
        run_behind_instinct: 0-1 — times runs onto through balls.
        hold_up_instinct: 0-1 — drops to link / lay off.
        pin_last_line_instinct: 0-1 — stays high stretching the defence.
        near_post_instinct: 0-1 — attacks the near post in the box.
        far_post_instinct: 0-1 — attacks the far post in the box.
        press_instinct: 0-1 — presses the centre-backs.
        aerial_instinct: 0-1 — targets / competes for aerial balls.
    """
    run_behind_instinct: float = 0.45
    hold_up_instinct: float = 0.40
    pin_last_line_instinct: float = 0.55
    near_post_instinct: float = 0.50
    far_post_instinct: float = 0.50
    press_instinct: float = 0.40
    aerial_instinct: float = 0.50

    # ── GEOMETRY ──────────────────────────────────────────

    def in_final_third(self, x: float, attacks_right: bool) -> bool:
        return x > FINAL_THIRD_X_ATT if attacks_right else x < (PITCH_X - FINAL_THIRD_X_ATT)

    def in_box(self, x: float, attacks_right: bool) -> bool:
        return x > BOX_ENTRY_X_ATT if attacks_right else x < (PITCH_X - BOX_ENTRY_X_ATT)

    def last_line_gap(
        self, x: float, y: float, attacks_right: bool,
        defenders: Optional[List], position_engine,
    ) -> float:
        """
        How much room is BEHIND the last defender for an in-behind run?
        1.0 = massive channel, 0.0 = no gap (offside trap tight).
        Uses the second-last-defender x (the offside line).
        """
        if position_engine is None or not defenders:
            return 0.5
        goal_x = _attacks_right_goal_x(attacks_right)
        # Find the deepest defender (closest to his own goal) apart from GK.
        deepest = float("inf")
        for d in defenders:
            if getattr(d, "position", None) in ("GK",):
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, _ = position_engine.get_position(dname)
            # "deepest" for the defending team = smallest distance to their goal
            dist_to_goal = abs(goal_x - dx)
            if dist_to_goal < deepest:
                deepest = dist_to_goal
        if deepest == float("inf"):
            return 0.5
        # The offside line sits at `deepest`; gap = how far ahead of it the
        # striker currently is (positive = onside channel to exploit).
        offside_line_x = goal_x - math.copysign(deepest, 1.0)  # mirror by side
        gap = abs(x - offside_line_x)
        # Normalise: a 15m+ onside channel is a clear run; <3m is nothing.
        return max(0.0, min(1.0, gap / 15.0))

    def run_behind_target(self, attacks_right: bool, anchor_y: float) -> Tuple[float, float]:
        """Destination of an in-behind run: high + wide of the last line."""
        sign = 1.0 if attacks_right else -1.0
        tx = 92.0 if attacks_right else PITCH_X - 92.0
        # Stay on the touchline side he's already on (split the CBs out wide).
        ty = anchor_y + ((CENTER_Y - anchor_y) * 0.4)
        return tx, ty

    def hold_up_target(self, ball_x: float, ball_y: float, attacks_right: bool) -> Tuple[float, float]:
        """Drop to link: come short, toward the ball, half-turned."""
        sign = 1.0 if attacks_right else -1.0
        tx = max(35.0, min(62.0, ball_x - sign * 6.0))
        ty = ball_y + (CENTER_Y - ball_y) * 0.2
        return tx, ty

    def box_arrival_target(self, ball_x: float, ball_y: float, attacks_right: bool) -> Tuple[float, float]:
        """Near/far post channel run when the ball is in the box / wide."""
        tx = max(BOX_ENTRY_X_ATT, min(96.0, ball_x)) if attacks_right \
            else min(PITCH_X - BOX_ENTRY_X_ATT, max(9.0, ball_x))
        ty = POST_BAND_Y[0] if ball_y < CENTER_Y else POST_BAND_Y[1]
        return tx, ty


# ─────────────────────────────────────────────
# STRIKER BEHAVIOR ENGINE
# ─────────────────────────────────────────────

class StrikerBehaviorEngine:
    """Pure decision engine for modern striker play. Stateless."""

    @staticmethod
    def should_run_behind(
        profile: StrikerSpatialProfile,
        x: float, y: float, attacks_right: bool,
        defenders: Optional[List] = None, position_engine=None,
        anchor_y: float = CENTER_Y, stamina_pct: float = 100.0,
    ) -> bool:
        """Time a run in behind the last line."""
        stam = _stamina_mult(stamina_pct)
        gap = profile.last_line_gap(x, y, attacks_right, defenders, position_engine) \
            if (defenders and position_engine) else 0.5
        # Needs a real gap to exploit; a tight offside trap kills the run.
        if gap < 0.30:
            return False
        prob = (profile.run_behind_instinct * 0.7 + gap * 0.25) * stam
        return random.random() < prob

    @staticmethod
    def should_hold_up(
        profile: StrikerSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        in_possession: bool = True,
    ) -> bool:
        """Drop off the front to link play (target man / DLS)."""
        if not in_possession:
            return False
        # Only drop when the ball is in the build/middle third (not already in the box).
        if profile.in_final_third(ball_x, attacks_right):
            return False
        prob = profile.hold_up_instinct * 0.6
        return random.random() and random.random() < prob

    @staticmethod
    def should_pin_last_line(
        profile: StrikerSpatialProfile,
        x: float, attacks_right: bool,
    ) -> bool:
        """Stay HIGH (pin the offside line) rather than dropping."""
        return profile.pin_last_line_instinct > 0.5

    @staticmethod
    def decide_run(
        profile: StrikerSpatialProfile,
        x: float, y: float, attacks_right: bool,
        ball_x: float, ball_y: float,
        in_possession: bool = True,
        defenders: Optional[List] = None, position_engine=None,
        anchor_y: float = CENTER_Y, stamina_pct: float = 100.0,
    ) -> Optional[str]:
        """
        Which run does this striker make?
            "behind"  — run in behind the last line
            "hold"    — drop to link
            "box"     — attack a post channel (ball wide/in box)
            None      — hold the line / no committed run
        """
        if not in_possession:
            return None
        stam = _stamina_mult(stamina_pct)
        if StrikerBehaviorEngine.should_run_behind(
                profile, x, y, attacks_right, defenders=defenders,
                position_engine=position_engine, anchor_y=anchor_y,
                stamina_pct=stamina_pct):
            return "behind"
        if profile.in_box(ball_x, attacks_right) and profile.in_final_third(ball_x, attacks_right):
            if random.random() < (profile.near_post_instinct * 0.5 + profile.far_post_instinct * 0.5) * stam:
                return "box"
        if StrikerBehaviorEngine.should_hold_up(profile, x, y, attacks_right, ball_x, ball_y, in_possession=True):
            return "hold"
        return None

    @staticmethod
    def build_profile_from_dna(player) -> StrikerSpatialProfile:
        """Build a StrikerSpatialProfile from DNA archetype + tendencies."""
        dna = getattr(player, "dna", None)
        profile = StrikerSpatialProfile()

        if dna is None:
            return profile

        tendencies = getattr(dna, "tendencies", None)
        if tendencies is not None:
            behind = getattr(tendencies, "makes_runs_behind", 0.35)
            profile.run_behind_instinct = max(0.10, min(0.95, behind * 1.2))
            shoots = getattr(tendencies, "shoots_from_distance", 0.15)
            profile.far_post_instinct = max(0.20, min(0.90, 0.40 + shoots * 0.8))
            near = getattr(tendencies, "shoots_from_distance", 0.15)  # same tendency drives both posts
            profile.near_post_instinct = max(0.20, min(0.90, 0.40 + near * 0.7))
            holds = getattr(tendencies, "holds_position", 0.60)
            profile.hold_up_instinct = max(0.15, min(0.90, 0.30 + (1.0 - holds) * 0.4))
            presses = getattr(tendencies, "presses_high", 0.40)
            profile.press_instinct = max(0.15, min(0.90, presses))
            profile.pin_last_line_instinct = max(0.20, min(0.95, 0.40 + behind * 0.5))

        archetype = getattr(dna, "archetype", "")
        if archetype == "poacher":
            profile.run_behind_instinct = max(profile.run_behind_instinct, 0.85)
            profile.pin_last_line_instinct = max(profile.pin_last_line_instinct, 0.85)
            profile.hold_up_instinct = min(profile.hold_up_instinct, 0.25)
            profile.near_post_instinct = max(profile.near_post_instinct, 0.65)
        elif archetype == "target_man":
            profile.hold_up_instinct = max(profile.hold_up_instinct, 0.85)
            profile.aerial_instinct = max(profile.aerial_instinct, 0.85)
            profile.run_behind_instinct = min(profile.run_behind_instinct, 0.30)
            profile.pin_last_line_instinct = min(profile.pin_last_line_instinct, 0.45)
        elif archetype == "complete_striker":
            profile.run_behind_instinct = max(profile.run_behind_instinct, 0.55)
            profile.hold_up_instinct = max(profile.hold_up_instinct, 0.55)
            profile.pin_last_line_instinct = max(profile.pin_last_line_instinct, 0.65)
            profile.press_instinct = max(profile.press_instinct, 0.55)
        elif archetype == "deep_lying_striker":
            profile.hold_up_instinct = max(profile.hold_up_instinct, 0.82)
            profile.run_behind_instinct = min(profile.run_behind_instinct, 0.45)
            profile.pin_last_line_instinct = min(profile.pin_last_line_instinct, 0.40)
        elif archetype == "speedster_striker":
            profile.run_behind_instinct = max(profile.run_behind_instinct, 0.90)
            profile.pin_last_line_instinct = max(profile.pin_last_line_instinct, 0.80)
            profile.hold_up_instinct = min(profile.hold_up_instinct, 0.25)

        specs = getattr(dna, "specialties", []) or []
        if "fox_in_box" in specs or "poacher" in specs:
            profile.run_behind_instinct = max(profile.run_behind_instinct, 0.85)
            profile.pin_last_line_instinct = max(profile.pin_last_line_instinct, 0.85)
        if "target_man" in specs or "aerial_threat" in specs:
            profile.hold_up_instinct = max(profile.hold_up_instinct, 0.85)
            profile.aerial_instinct = max(profile.aerial_instinct, 0.85)
        if "clinical_finisher" in specs or "cold_blooded" in specs:
            profile.near_post_instinct = max(profile.near_post_instinct, 0.70)
        if "speedster" in specs:
            profile.run_behind_instinct = max(profile.run_behind_instinct, 0.90)

        return profile


# ─────────────────────────────────────────────
# STRIKER REGISTRY
# ─────────────────────────────────────────────

class StrikerRegistry:
    """Holds StrikerSpatialProfile per ST/CF in a match."""

    def __init__(self):
        self.profiles: Dict[str, StrikerSpatialProfile] = {}

    def register_player(self, player) -> Optional[StrikerSpatialProfile]:
        position = getattr(player, "position", "")
        if position not in ("ST", "CF"):
            return None
        profile = StrikerBehaviorEngine.build_profile_from_dna(player)
        name = getattr(player, "name", str(player))
        self.profiles[name] = profile
        return profile

    def register_team(self, players: List) -> None:
        for p in players or []:
            self.register_player(p)

    def get(self, player_name: str) -> Optional[StrikerSpatialProfile]:
        return self.profiles.get(player_name)

    def is_striker(self, player_name: str) -> bool:
        return player_name in self.profiles

    def remove(self, player_name: str) -> None:
        self.profiles.pop(player_name, None)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n🥅  PLOFA 26/27 — Modern Striker Behavior Engine Demo")
    print("=" * 64)

    class _Tend:
        makes_runs_behind = 0.65
        shoots_from_distance = 0.10
        holds_position = 0.50
        presses_high = 0.50

    class _DNA:
        archetype = "poacher"
        specialties = ["fox_in_box"]
        tendencies = _Tend()

    class _TendT:
        makes_runs_behind = 0.30
        shoots_from_distance = 0.15
        holds_position = 0.85
        presses_high = 0.30

    class _DNAT:
        archetype = "target_man"
        specialties = ["aerial_threat"]
        tendencies = _TendT()

    class _TendC:
        makes_runs_behind = 0.45
        shoots_from_distance = 0.20
        holds_position = 0.60
        presses_high = 0.55

    class _DNAC:
        archetype = "complete_striker"
        specialties = []
        tendencies = _TendC()

    class _Player:
        def __init__(self, name, position, dna):
            self.name = name
            self.position = position
            self.dna = dna

    players = [
        _Player("Kane", "ST", _DNA()),
        _Player("Lukaku", "ST", _DNAT()),
        _Player("Kane2", "ST", _DNAC()),
    ]
    registry = StrikerRegistry()
    registry.register_team(players)

    for pl in players:
        p = registry.get(pl.name)
        print(f"\n1. PROFILE — {pl.name} ({pl.position}, {pl.dna.archetype}):")
        print(f"   run_behind:        {p.run_behind_instinct:.2f}")
        print(f"   hold_up:           {p.hold_up_instinct:.2f}")
        print(f"   pin_last_line:     {p.pin_last_line_instinct:.2f}")
        print(f"   near_post:         {p.near_post_instinct:.2f}")
        print(f"   far_post:          {p.far_post_instinct:.2f}")
        print(f"   aerial:            {p.aerial_instinct:.2f}")

    import types
    class _FakePE:
        def get_position(self, name):
            # Last defender (CB) sitting at x=78; GK at 4.
            return {"Opp CB": (78.0, 34.0), "Opp GK": (4.0, 34.0)}.get(name, (50.0, 34.0))
    pe = _FakePE()
    defenders = [type("P", (), {"name": "Opp CB", "position": "CB"})(),
                 type("P", (), {"name": "Opp GK", "position": "GK"})()]

    print("\n2. RUN DECISIONS (attacking right, ball progressing):")
    for label, name in (("poacher", "Kane"), ("target_man", "Lukaku"), ("complete", "Kane2")):
        prof = registry.get(name)
        behind = sum(1 for _ in range(200)
                     if StrikerBehaviorEngine.should_run_behind(
                         prof, 70.0, 34.0, True, defenders=defenders, position_engine=pe))
        print(f"   {label:<12} in-behind runs={behind}/200")

    print("\n✅ Modern Striker Behavior Engine operational — pure geometry, zero deps.")
