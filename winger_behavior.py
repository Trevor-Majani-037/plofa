"""
PLOFA 26/27 — MODERN WINGER BEHAVIOR ENGINE (Checkpoint 18)
============================================================
winger_behavior.py

Philosophy:
    Modern wingers in the EPL / Europe's top 5 leagues (Vini Jr, Saka,
    Salah, Doku, Rodrygo, Martinelli) are NOT drifting number-10s. They are
    told to stay by the touchline and flanks — the middle of the pitch is
    always full, a #10 already owns that space, and a winger who drifts
    inside leaves his flank open and crowds his own teammates.

    The modern winger's job is GEOMETRIC:
        1. HUG THE TOUCHLINE  — stretch the pitch, pin the fullback back,
           create width so the midfield can play through the half-spaces.
        2. ATTACK THE FLANK   — use the touchline→byline corridor as a
           runway: isolate the fullback 1v1, drive to the byline, and
           deliver from the dangerous wide zones.
        3. ENTER THE BOX FROM WIDE — when the ball is on the OPPOSITE
           flank, sprint to the back post (Saka/Vini arriving late).
        4. CUT INSIDE ONLY AT THE RIGHT MOMENT — inverted wingers cut
           onto their strong foot when the geometry says the half-space
           is open, NOT as a default drift into midfield traffic.

    This module gives every winger a persistent spatial identity:
        - a touchline anchor (their flank's y-channel)
        - a flank commitment score (how strongly they resist central drift)
        - a byline instinct (drive to the byline vs cut early)
        - an isolation thirst (attack the fullback 1v1)
        - a box-entry instinct (late runs to the back post)

    It is PURE — reads PositionEngine state, writes nothing. The event
    chain and position engine consume its geometry to steer winger
    positioning, dribble/carry decisions, and cross timing.

    Pitch geometry (StatsBomb scale):
        x: 0 (own goal line) → 105 (opponent goal line)
        y: 0 (left touchline) → 68 (right touchline)
        Center: y = 34
        Left flank channel:  y < 20
        Right flank channel: y > 48
        Byline: x = 105 (attacking right) / x = 0 (attacking left)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Optional pitch-control awareness. pitch_control.py exists as a sibling
# module but — as of this upgrade — is NOT yet called from
# match_engine.py's per-minute loop, so PitchControlResult objects will
# not actually arrive here until that separate wiring is done elsewhere.
# Every function below that accepts pitch_control_result/pitch_control_field
# treats them as optional and degrades gracefully to real defender-geometry
# checks (which ARE live today) when they're None — so this import failing
# or the params never being populated changes nothing about current
# behavior; it only means the pitch-control bonus path stays dormant
# until wired in.
try:
    from pitch_control import PitchControlField, PitchControlResult
    _HAS_PITCH_CONTROL = True
except ImportError:
    _HAS_PITCH_CONTROL = False

# ─────────────────────────────────────────────
# FLANK GEOMETRY CONSTANTS
# ─────────────────────────────────────────────

PITCH_Y = 68.0
CENTER_Y = 34.0
PITCH_X = 105.0

# Flank channels (StatsBomb-style wide channels)
LEFT_FLANK_MAX_Y = 48.0
RIGHT_FLANK_MIN_Y = 20.0

# Touchline anchors: how close to the touchline a winger's home should sit.
# These match the StatsBomb frame used everywhere in this codebase (y=10 is the
# LEFT touchline, y=58 the RIGHT — see BASE_HOME_POSITIONS in position_engine:
# LW home_y=10, RW home_y=58).
#
# NOTE (Checkpoint 21e): position NAME alone cannot decide a winger's flank —
# for a team attacking LEFT the "LW" actually stands on the right side of the
# pitch (home_y is mirrored by the formation). The anchors below are therefore
# NOT the source of truth for positioning; the position engine anchors wide
# players on their formation home_y (direction-aware) in drift_minute /
# record_touch / delivery bias. These constants only feed the name-based
# helpers (carry / flank pull / cross-zone checks), which defer to the profile's
# own touchline_anchor_y (built from these, and correct in the attacks-right
# frame). Fixed: they used to be swapped (LEFT=58/RIGHT=10), which would have
# pushed an inverted LW toward the WRONG touchline if ever wired into a carry.
LEFT_TOUCHLINE_ANCHOR_Y = 10.0
RIGHT_TOUCHLINE_ANCHOR_Y = 58.0

# The "danger corridor" — from the touchline to the byline, and the wide
# box-entry zones where modern wingers do their damage.
BYLINE_X_ATTACKING = 105.0
BYLINE_X_DEFENDING = 0.0

# Crossing zone: attacking third + wide
# Checkpoint 24 — the crossing zone is the byline corridor, not the entire
# final third. At 70.0 a winger who had barely entered the attacking third
# was "in the cross zone", which is what turned wingers into cross-machines
# (12-35 deliveries/match vs a real winger's 2-6).
CROSS_ZONE_X_ATTACKING = 82.0
CROSS_ZONE_X_DEFENDING = 23.0

# Box entry: the wide channels of the penalty area (x within ~16m of goal)
BOX_ENTRY_X_ATTACKING = 89.0
BOX_ENTRY_X_DEFENDING = 16.0

# Fullback isolation: a winger is "isolated" when the nearest defender is
# the fullback and within this distance (1v1 opportunity)
ISOLATION_RANGE_M = 8.0

# How far a winger can drift from their touchline anchor before the
# position engine starts pulling them back hard (metres)
FLANK_DRIFT_LIMIT_M = 4.0

# Half-width (metres) of the flank channel measured from the winger's
# FORMATION anchor (home_y). Checkpoint 21e: a team attacking left mirrors
# the shape, so "am I still on my flank" must be answered relative to the
# formation anchor, never the position name. The name-based
# flank_channel() (whole left/right half) is only a frame fallback.
FLANK_CHANNEL_HALF_WIDTH_M = 12.0

# ── HALF-SPACE GEOMETRY ──────────────────────────────────────────
# The half-space is the corridor between a winger's flank channel and
# the central channel — the gap between opposing fullback and centre-back
# that an inverted winger's cut-inside is actually trying to exploit.
# Defined as a width-band immediately infield of the winger's own
# touchline anchor, not the whole central third.
HALF_SPACE_WIDTH_M = 14.0        # width of the corridor, in from the flank
HALF_SPACE_DEFENDER_RADIUS_M = 10.0   # a defender this close "closes" the space

# A covering defender (second body) recovering within this distance of a
# byline-driving winger means the "isolation" is about to stop being 1v1 —
# real wingers hesitate here rather than committing blind.
COVER_RECOVERY_RANGE_M = 11.0

# Fatigue banding — deliberately mirrors squad_manager.PlayerStaminaState.
# update_performance_mult()'s bands (100-60 full, 60-40 slight, 40-20
# noticeable, <20 severe) so winger aggression degrades on the SAME curve
# the rest of the engine already uses for tired players, rather than a
# freshly invented one.
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


def _attacks_right_box_entry_x(attacks_right: bool) -> float:
    return BOX_ENTRY_X_ATTACKING if attacks_right else BOX_ENTRY_X_DEFENDING


# ─────────────────────────────────────────────
# WINGER SPATIAL PROFILE
# ─────────────────────────────────────────────

@dataclass
class WingerSpatialProfile:
    """
    Persistent per-winger geometry identity. Built once at kickoff from
    the player's DNA tendencies + archetype, then read every touch.

    Attributes:
        flank: "left" | "right" — which touchline this winger owns.
        touchline_anchor_y: the y-coordinate of their touchline channel.
        flank_commitment: 0-1 — how strongly they resist drifting central.
            1.0 = pure touchline hugger (never leaves the flank channel).
            0.0 = drifts inside freely (old-style inside forward).
        byline_instinct: 0-1 — drive to the byline vs cut inside early.
            1.0 = always attacks the byline (traditional winger).
            0.0 = always cuts inside early (pure inverted).
        isolation_thirst: 0-1 — how often they attack the fullback 1v1.
        box_entry_instinct: 0-1 — how often they make late back-post runs.
        cross_instinct: 0-1 — how often they deliver from wide vs carry on.
    """
    flank: str = "left"
    touchline_anchor_y: float = LEFT_TOUCHLINE_ANCHOR_Y
    flank_commitment: float = 0.85
    byline_instinct: float = 0.55
    isolation_thirst: float = 0.65
    box_entry_instinct: float = 0.50
    cross_instinct: float = 0.45

    @property
    def is_left(self) -> bool:
        return self.flank == "left"

    @property
    def is_right(self) -> bool:
        return self.flank == "right"

    def flank_channel(self, y: float) -> bool:
        """Is this y-coordinate inside this winger's flank channel? (name-based)"""
        if self.is_left:
            return y < LEFT_FLANK_MAX_Y
        return y > RIGHT_FLANK_MIN_Y

    def in_flank_channel(self, y: float, anchor_y: Optional[float] = None) -> bool:
        """
        Formation-aware flank-channel membership (Checkpoint 21e / 18 wiring).

        When `anchor_y` (the winger's formation home_y) is supplied, "on the
        flank" means within FLANK_CHANNEL_HALF_WIDTH_M of that anchor — this
        is the correct test for a mirrored (attacking-left) team, where a
        name-based 'LW' actually stands on the right side of the pitch.

        When `anchor_y` is None, falls back to the name-based frame check.
        """
        if anchor_y is not None:
            return abs(y - anchor_y) <= FLANK_CHANNEL_HALF_WIDTH_M
        return self.flank_channel(y)

    def touchline_distance(self, y: float) -> float:
        """How far is this winger from their touchline (metres)?"""
        return abs(y - self.touchline_anchor_y)

    def byline_distance(self, x: float, attacks_right: bool) -> float:
        """How far is this winger from the byline they attack (metres)?"""
        goal_x = _attacks_right_goal_x(attacks_right)
        return abs(goal_x - x)

    def in_cross_zone(self, x: float, y: float, attacks_right: bool) -> bool:
        """Is the winger in the dangerous wide crossing zone?"""
        cross_x = _attacks_right_cross_zone_x(attacks_right)
        if attacks_right:
            in_x = x > cross_x
        else:
            in_x = x < cross_x
        return in_x and self.flank_channel(y)

    def in_box_entry_zone(self, x: float, y: float, attacks_right: bool) -> bool:
        """Is the winger in the wide box-entry zone (near the byline)?"""
        box_x = _attacks_right_box_entry_x(attacks_right)
        if attacks_right:
            in_x = x > box_x
        else:
            in_x = x < box_x
        return in_x and self.flank_channel(y)

    def danger_zone_score(self, x: float, y: float, attacks_right: bool) -> float:
        """
        How dangerous is this winger's current position? 0-1.
        Peaks in the touchline→byline corridor and the wide box-entry zones
        — the exact areas Vini/Saka do their damage from.
        """
        goal_x = _attacks_right_goal_x(attacks_right)
        dist_to_byline = abs(goal_x - x)

        # Byline proximity: 1.0 at the byline, decays to 0 by ~35m out
        byline_score = max(0.0, 1.0 - dist_to_byline / 35.0)

        # Touchline proximity: 1.0 on the touchline, decays to 0 by ~20m in
        touch_score = max(0.0, 1.0 - self.touchline_distance(y) / 20.0)

        # Box entry bonus: inside the wide box channels, danger spikes
        box_bonus = 0.0
        if self.in_box_entry_zone(x, y, attacks_right):
            box_bonus = 0.30

        return max(0.0, min(1.0, byline_score * 0.55 + touch_score * 0.35 + box_bonus))

    def fullback_isolation(
        self,
        x: float,
        y: float,
        defenders: Optional[List],
        position_engine,
        attacks_right: bool = True,
    ) -> Tuple[bool, Optional[object], float]:
        """
        Is this winger isolated 1v1 against the opposing fullback?

        Returns (isolated, fullback, distance_to_fullback).
        A winger is "isolated" when:
            - they are in their flank channel (wide)
            - the nearest outfield defender is the OPPOSING fullback
              (LB for a right winger, RB for a left winger)
            - that fullback is within ISOLATION_RANGE_M
        """
        if position_engine is None or not defenders:
            return False, None, float("inf")

        # Which fullback marks this winger?
        marking_fullback_pos = "RB" if self.is_left else "LB"

        best_fb = None
        best_fb_dist = float("inf")
        nearest_def = None
        nearest_dist = float("inf")

        for d in defenders:
            if getattr(d, "position", None) == "GK":
                continue
            dx, dy = position_engine.get_position(d.name)
            dist = math.hypot(dx - x, dy - y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_def = d
            if getattr(d, "position", None) == marking_fullback_pos:
                if dist < best_fb_dist:
                    best_fb_dist = dist
                    best_fb = d

        if best_fb is None:
            return False, None, float("inf")

        # Isolated when the fullback is the nearest defender AND close enough
        isolated = (
            nearest_def is not None
            and best_fb is not None
            and nearest_def.name == best_fb.name
            and best_fb_dist < ISOLATION_RANGE_M
        )
        return isolated, best_fb, best_fb_dist

    def covering_defender_arriving(
        self,
        x: float,
        y: float,
        defenders: Optional[List],
        position_engine,
        exclude_name: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Is a SECOND defender (cover) recovering into range while the winger
        is nominally isolated 1v1?

        Real wingers don't blindly commit to a byline drive just because
        the strict nearest-defender isolation test passes this instant —
        if a covering centre-back or midfielder is closing fast, the 1v1
        is about to stop being a 1v1. This checks for any OTHER defender
        (not the marking fullback itself) within COVER_RECOVERY_RANGE_M.

        Returns (cover_arriving, distance_of_nearest_cover).
        """
        if position_engine is None or not defenders:
            return False, float("inf")

        nearest_cover = float("inf")
        for d in defenders:
            name = getattr(d, "name", None)
            if name is None or name == exclude_name:
                continue
            if getattr(d, "position", None) == "GK":
                continue
            dx, dy = position_engine.get_position(name)
            dist = math.hypot(dx - x, dy - y)
            if dist < nearest_cover:
                nearest_cover = dist

        return nearest_cover < COVER_RECOVERY_RANGE_M, nearest_cover

    def half_space_openness(
        self,
        x: float,
        y: float,
        attacks_right: bool,
        defenders: Optional[List],
        position_engine,
        pitch_control_result: Optional["PitchControlResult"] = None,
        pitch_control_field: Optional["PitchControlField"] = None,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        How open is the half-space this winger would cut into? 0 (packed)
        to 1 (wide open).

        PRIMARY signal (live today): real defender density in the
        corridor immediately infield of this winger's flank channel, in
        the final third — the actual gap between opposing fullback and
        centre-back an inverted winger's cut is trying to exploit. Uses
        the same position_engine.get_position() pattern already proven
        in fullback_isolation() above.

        `anchor_y` (formation home_y, Checkpoint 21e) makes the corridor
        formation-aware; when omitted, the name-based touchline_anchor_y
        is used (correct for the attacks-right frame).

        SECONDARY signal (dormant until pitch_control.py is wired into
        match_engine.py's per-minute loop): if a PitchControlResult is
        supplied, cells in the corridor that are opponent-controlled
        further suppress the score, cells that are neutral/attacking-team
        controlled corroborate openness. Safe no-op when not supplied.
        """
        if position_engine is None or not defenders:
            return 0.5   # unknown — neutral, neither open nor closed

        # Corridor: a band running from this winger's flank anchor
        # inward by HALF_SPACE_WIDTH_M, spanning the attacking third.
        # Anchor-relative (Checkpoint 21e): mirrored formations cut toward
        # the pitch centre from wherever their formation actually puts them.
        anchor = anchor_y if anchor_y is not None else self.touchline_anchor_y
        if anchor <= CENTER_Y:
            corridor_y_lo = anchor
            corridor_y_hi = anchor + HALF_SPACE_WIDTH_M
        else:
            corridor_y_lo = anchor - HALF_SPACE_WIDTH_M
            corridor_y_hi = anchor

        corridor_cx = x - (12.0 if attacks_right else -12.0)  # slightly infield/ahead
        corridor_cy = (corridor_y_lo + corridor_y_hi) / 2.0

        closest = float("inf")
        occupants = 0
        for d in defenders:
            if getattr(d, "position", None) in ("GK",):
                continue
            dname = getattr(d, "name", None)
            if dname is None:
                continue
            dx, dy = position_engine.get_position(dname)
            if not (corridor_y_lo - 3.0 <= dy <= corridor_y_hi + 3.0):
                continue
            dist = math.hypot(dx - corridor_cx, dy - corridor_cy)
            if dist < HALF_SPACE_DEFENDER_RADIUS_M:
                occupants += 1
            if dist < closest:
                closest = dist

        if closest == float("inf"):
            defender_score = 1.0   # nobody nearby at all — wide open
        else:
            defender_score = max(0.0, min(1.0, closest / HALF_SPACE_DEFENDER_RADIUS_M))
        defender_score -= 0.15 * max(0, occupants - 1)  # a second body compounds the closure
        defender_score = max(0.0, min(1.0, defender_score))

        if pitch_control_result is not None and pitch_control_field is not None:
            owner = pitch_control_field.cell_ownership(
                pitch_control_result, corridor_cx, corridor_cy
            )
            attacking_owner = "home" if attacks_right else "away"
            if owner == attacking_owner:
                pc_score = 0.85
            elif owner == "neutral":
                pc_score = 0.55
            else:
                pc_score = 0.15
            # Blend: defender geometry is the primary signal (already
            # correct on its own), pitch control corroborates/refines it.
            return round(defender_score * 0.65 + pc_score * 0.35, 3)

        return round(defender_score, 3)


# ─────────────────────────────────────────────
# WINGER BEHAVIOR ENGINE
# ─────────────────────────────────────────────

class WingerBehaviorEngine:
    """
    Pure decision engine for modern winger play. Reads PositionEngine
    state + the winger's spatial profile, returns steering decisions
    that the event chain and position engine consume.

    All methods are stateless — the profile is passed in per call.
    """

    # ── POSITIONING STEERING ──────────────────────────────────

    @staticmethod
    def flank_pull_strength(
        profile: WingerSpatialProfile,
        current_y: float,
        in_possession: bool = True,
        stamina_pct: float = 100.0,
    ) -> float:
        """
        How hard should the position engine pull this winger back toward
        their touchline anchor this minute?

        - In possession: strong pull when they've drifted out of the flank
          channel (they must stretch the pitch).
        - Out of possession: slightly weaker (they may tuck in to help
          defend the half-space, but still shouldn't fully abandon the flank).
        - A tired winger holds discipline a little less sharply — the pull
          is softened (not removed) by the same fatigue banding the rest
          of the engine uses, since a gassed player drifts more even when
          "supposed" to hold width.
        """
        stam_mult = 0.75 + 0.25 * _stamina_mult(stamina_pct)  # softer floor than combat actions

        if profile.flank_channel(current_y):
            # Already wide — light touch, let them roam within the channel
            base = 0.15 if in_possession else 0.10
        else:
            # Drifted central — pull hard back to the touchline
            drift = profile.touchline_distance(current_y)
            if drift > FLANK_DRIFT_LIMIT_M:
                base = 0.55 if in_possession else 0.35
            else:
                base = 0.35 if in_possession else 0.20

        return round(base * stam_mult, 3)

    @staticmethod
    def should_drive_byline(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        isolated: bool = False,
        under_pressure: bool = False,
        defenders: Optional[List] = None,
        position_engine=None,
        winger_name: Optional[str] = None,
        stamina_pct: float = 100.0,
        anchor_y: Optional[float] = None,
    ) -> bool:
        """
        Should this winger drive to the byline right now?

        Modern winger geometry: when in the final third, wide, and either
        isolated 1v1 or with space ahead, the byline is the highest-value
        destination — it pins the fullback, opens cut-back angles, and
        creates the touchline→byline danger corridor.

        `isolated=True` no longer guarantees an automatic commit. A real
        winger who is technically 1v1 THIS INSTANT still hesitates if a
        covering defender is visibly recovering into range — the isolation
        is about to stop being clean. When defenders + position_engine are
        supplied, this is checked for real via
        profile.covering_defender_arriving(); if no cover is detected (or
        the caller hasn't supplied defenders/position_engine — old
        behavior preserved), an isolated winger still drives with very
        high probability, same spirit as the original unconditional True.

        Returns True when:
            - in the attacking third
            - in their flank channel (formation-aware via `anchor_y`)
            - not under heavy pressure (or isolated with no cover arriving)
            - their byline instinct beats the random roll
            - scaled down as stamina drops
        """
        # Checkpoint 24 — the drive gate uses the attacking third (x>70);
        # the tightened cross zone (82) governs DELIVERY only, otherwise
        # the tight zone would starve the corridor of drives entirely.
        if attacks_right:
            in_att_third = x > 70.0
        else:
            in_att_third = x < 35.0
        if not in_att_third or not profile.in_flank_channel(y, anchor_y):
            return False

        stam_mult = _stamina_mult(stamina_pct)

        if isolated:
            if defenders is not None and position_engine is not None:
                cover_arriving, _ = profile.covering_defender_arriving(
                    x, y, defenders, position_engine, exclude_name=winger_name,
                )
                if not cover_arriving:
                    return True
                # Cover is recovering — still very likely to drive (real
                # wingers back themselves in this window) but no longer
                # a certainty, and it now respects fatigue.
                return random.random() < 0.80 * stam_mult
            return True

        if under_pressure:
            return False

        # Byline instinct + territory bonus (closer to byline = more likely)
        byline_dist = profile.byline_distance(x, attacks_right)
        territory_bonus = max(0.0, 1.0 - byline_dist / 40.0) * 0.25
        prob = (profile.byline_instinct * 0.55 + territory_bonus) * stam_mult
        return random.random() < prob

    @staticmethod
    def should_cut_inside(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        half_space_open: Optional[bool] = None,
        defenders: Optional[List] = None,
        position_engine=None,
        pitch_control_result: Optional["PitchControlResult"] = None,
        pitch_control_field: Optional["PitchControlField"] = None,
        stamina_pct: float = 100.0,
        anchor_y: Optional[float] = None,
    ) -> bool:
        """
        Should this winger cut inside onto their strong foot?

        The modern inverted winger (Salah, Saka, Rodrygo) cuts inside ONLY
        when the geometry says the half-space is open — not as a default
        drift into midfield traffic. The cut is a deliberate diagonal into
        the channel between fullback and centre-back, not a wander central.

        half_space_open resolution order (backward compatible):
            1. If explicitly passed True/False by the caller, respected
               as-is — old call sites keep their exact old behavior.
            2. Else, if defenders + position_engine are supplied, computed
               LIVE from real defender geometry via
               profile.half_space_openness() (optionally corroborated by
               pitch_control_result/field if those are also supplied).
            3. Else (nothing supplied), defaults to False — identical to
               the original function's old default.

        Returns True when:
            - in the attacking third
            - in their flank channel (formation-aware via `anchor_y`)
            - the half-space is open (real geometry or explicit override)
            - their byline instinct is LOW (inverted wingers cut more)
            - scaled down as stamina drops (a tired winger is less likely
              to commit to a sharp diagonal cut)
        """
        # Checkpoint 24 — same attacking-third gate as the drive decision
        # (the tightened cross zone governs delivery only).
        if attacks_right:
            in_att_third = x > 70.0
        else:
            in_att_third = x < 35.0
        if not in_att_third or not profile.in_flank_channel(y, anchor_y):
            return False

        if half_space_open is None:
            if defenders is not None and position_engine is not None:
                openness = profile.half_space_openness(
                    x, y, attacks_right, defenders, position_engine,
                    pitch_control_result=pitch_control_result,
                    pitch_control_field=pitch_control_field,
                    anchor_y=anchor_y,
                )
                half_space_open = openness > 0.55
            else:
                half_space_open = False

        # Inverted wingers cut; traditional wingers drive to the byline
        cut_prob = (1.0 - profile.byline_instinct) * 0.60
        if half_space_open:
            cut_prob += 0.25
        cut_prob *= _stamina_mult(stamina_pct)
        return random.random() < cut_prob

    @staticmethod
    def should_cross(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        under_pressure: bool = False,
    ) -> bool:
        """
        Should this winger deliver a cross right now?

        Modern wingers cross when they've reached the dangerous wide zone
        (touchline→byline corridor) and either:
            - they've driven to the byline and the cut-back is on, OR
            - they're in the crossing zone with space to whip it in

        Returns True when in the cross zone + flank channel + cross instinct
        beats the roll (boosted near the byline).
        """
        if not profile.in_cross_zone(x, y, attacks_right):
            return False
        if under_pressure:
            # Pressed wingers still cross — it's often the only outlet
            prob = profile.cross_instinct * 0.40
        else:
            byline_dist = profile.byline_distance(x, attacks_right)
            byline_bonus = max(0.0, 1.0 - byline_dist / 30.0) * 0.30
            prob = profile.cross_instinct * 0.50 + byline_bonus
        return random.random() < prob

    @staticmethod
    def decide_cross_type(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        under_pressure: bool = False,
    ) -> str:
        """
        What KIND of delivery is this, geometrically? A cutback from the
        byline and an early whipped ball from deep in the cross zone are
        tactically different deliveries with different target runs and
        success profiles — should_cross() only ever returned a bare bool,
        collapsing that distinction. This is additive: it doesn't change
        should_cross()'s existing contract, it gives callers who want it a
        real classification to drive delivery-specific logic downstream
        (target player selection, completion odds, xA weighting).

        Returns one of: "cutback", "driven_low", "whipped_early", "chip".
        """
        byline_dist = profile.byline_distance(x, attacks_right)

        if byline_dist < 8.0 and not under_pressure:
            # Right at the byline, time to pick a pass — the highest-value
            # modern delivery (Robertson/Trent-style pull-back).
            return "cutback"
        if byline_dist < 15.0:
            # Still close, but not quite at the line — a firm ball driven
            # low across the face of goal.
            return "driven_low"
        if profile.in_cross_zone(x, y, attacks_right) and byline_dist < 30.0:
            # Traditional out-swinging delivery from the wide channel.
            return "whipped_early"
        # Further back / under pressure and forced early — a deeper,
        # lower-percentage chip into the box.
        return "chip"

    @staticmethod
    def should_enter_box(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        ball_on_opposite_flank: bool = False,
        stamina_pct: float = 100.0,
    ) -> bool:
        """
        Should this winger make a late run into the box from wide?

        The modern winger's box entry (Saka at the back post, Vini arriving
        late) happens when:
            - the ball is on the OPPOSITE flank (they sprint to the back post)
            - OR they're already in the wide box-entry zone and the cut-back
              is on

        A back-post sprint is a genuine physical effort late in a phase of
        play — scaled down as stamina drops, same banding as the rest of
        the engine's fatigue model.

        Returns True when the box-entry instinct + situation beats the roll.
        """
        stam_mult = _stamina_mult(stamina_pct)
        if ball_on_opposite_flank:
            # Back-post sprint — high probability for box-entry instinct players
            prob = (0.35 + profile.box_entry_instinct * 0.45) * stam_mult
            return random.random() < prob

        if profile.in_box_entry_zone(x, y, attacks_right):
            prob = profile.box_entry_instinct * 0.55 * stam_mult
            return random.random() < prob

        return False

    @staticmethod
    def back_post_target_y(profile: WingerSpatialProfile, ball_y: float) -> float:
        """
        Where should this winger run when the ball is on the opposite flank?
        The far-side winger sprints to the BACK POST — the opposite side of
        the goal from the ball.

        Proportional, not a fixed binary side: a cross from the direct
        opposite touchline (ball_y far from centre) pulls the run all the
        way to the far six-yard-box edge; a cross from just past centre
        produces a much shallower, more central back-post run — which is
        what actually happens in real deliveries (the run's width scales
        with how far across the ball is, not a coin-flip "left post/right
        post"). Also weighted by box_entry_instinct: a player who commits
        harder to the box (higher instinct) runs tighter to the six-yard
        line rather than hanging at the far edge of the box.
        """
        # How far across the pitch is the ball, as a fraction of the half-width?
        cross_dist_frac = min(1.0, abs(ball_y - CENTER_Y) / (PITCH_Y / 2.0))

        # Far edge of the six-yard box on the opposite side from the ball.
        far_post_edge = 48.0 if ball_y < CENTER_Y else 20.0

        # Blend: at cross_dist_frac=0 (ball near centre), target stays near
        # centre (shallow back-post run); at 1.0 (ball on the far
        # touchline), target goes all the way to the far post edge.
        target = CENTER_Y + (far_post_edge - CENTER_Y) * cross_dist_frac

        # box_entry_instinct pulls the run tighter toward the six-yard
        # line (more central/aggressive) rather than hanging wide at the
        # box edge — a higher-instinct runner commits harder to goal.
        six_yard_edge = 37.66 if ball_y < CENTER_Y else 30.34
        target = target + (six_yard_edge - target) * (profile.box_entry_instinct * 0.35)

        return round(max(4.0, min(64.0, target)), 1)

    @staticmethod
    def carry_direction_bias(
        profile: WingerSpatialProfile,
        x: float,
        y: float,
        attacks_right: bool,
        defenders: Optional[List] = None,
        position_engine=None,
        anchor_y: Optional[float] = None,
    ) -> float:
        """
        Lateral bias for a winger's carry: how strongly should their carry
        stay in the flank channel (positive = toward touchline, negative =
        toward center)?

        Modern wingers carry ALONG the touchline (down the flank), not
        diagonally into midfield. Returns a y-bias in metres.

        `anchor_y` (formation home_y, Checkpoint 21e) makes the bias
        formation-aware — a mirrored (attacking-left) winger is steered to
        the touchline his FORMATION puts him on, never the position-name
        anchor. When omitted, the profile's name-based touchline_anchor_y
        is used (correct for the attacks-right frame).

        When defenders + position_engine are supplied, adds a real
        opponent-avoidance nudge: if the nearest defender is sitting
        directly in the touchline-anchor direction, the bias eases off
        rather than blindly running the carrier into a body. Backward
        compatible — omitting these params reproduces the old pure
        anchor-pull behavior exactly.
        """
        anchor = anchor_y if anchor_y is not None else profile.touchline_anchor_y
        if profile.in_flank_channel(y, anchor_y):
            base_bias = 0.0
        else:
            # Drifted off the flank — the carry pushes back onto the
            # touchline, using the formation-corrected anchor so a left
            # winger is always dragged to the correct touchline.
            base_bias = (anchor - y) * 0.30

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
                    # A body sits right where this bias is steering toward —
                    # ease off rather than run straight at them.
                    base_bias *= 0.4
                    break

        return base_bias

    @staticmethod
    def build_profile_from_dna(player) -> WingerSpatialProfile:
        """
        Build a WingerSpatialProfile from a player's DNA tendencies +
        archetype + specialties. Called once at kickoff.

        DNA signals used:
            - tendencies.cuts_inside        → inverse of byline_instinct
            - tendencies.attempts_dribble   → isolation_thirst
            - tendencies.makes_runs_behind  → box_entry_instinct
            - archetype                     → base profile shape
            - specialties                   → explicit overrides
        """
        dna = getattr(player, "dna", None)
        position = getattr(player, "position", "LW")
        flank = "left" if position == "LW" else "right"
        anchor_y = LEFT_TOUCHLINE_ANCHOR_Y if flank == "left" else RIGHT_TOUCHLINE_ANCHOR_Y

        # Defaults: modern touchline-hugging winger
        profile = WingerSpatialProfile(
            flank=flank,
            touchline_anchor_y=anchor_y,
            flank_commitment=0.85,
            byline_instinct=0.55,
            isolation_thirst=0.65,
            box_entry_instinct=0.50,
            cross_instinct=0.45,
        )

        if dna is None:
            return profile

        tendencies = getattr(dna, "tendencies", None)
        if tendencies is not None:
            # cuts_inside is the inverse of byline instinct:
            #   traditional winger (cuts_inside=0.20) → byline_instinct high
            #   inverted winger (cuts_inside=0.72)    → byline_instinct low
            cuts = getattr(tendencies, "cuts_inside", 0.30)
            profile.byline_instinct = max(0.05, min(0.95, 1.0 - cuts * 1.1))

            # attempts_dribble drives isolation thirst
            dribble = getattr(tendencies, "attempts_dribble", 0.25)
            profile.isolation_thirst = max(0.20, min(0.95, 0.35 + dribble * 0.9))

            # makes_runs_behind drives box entry
            runs = getattr(tendencies, "makes_runs_behind", 0.35)
            profile.box_entry_instinct = max(0.15, min(0.90, 0.25 + runs * 0.8))

        # Archetype overrides
        archetype = getattr(dna, "archetype", "")
        if archetype == "traditional_winger":
            profile.byline_instinct = max(profile.byline_instinct, 0.75)
            profile.cross_instinct = max(profile.cross_instinct, 0.60)
            profile.flank_commitment = max(profile.flank_commitment, 0.90)
        elif archetype == "inverted_winger":
            profile.byline_instinct = min(profile.byline_instinct, 0.35)
            profile.cross_instinct = min(profile.cross_instinct, 0.30)
            profile.isolation_thirst = max(profile.isolation_thirst, 0.70)
        elif archetype == "pressing_winger":
            profile.flank_commitment = max(profile.flank_commitment, 0.88)
            profile.box_entry_instinct = max(profile.box_entry_instinct, 0.55)
        elif archetype == "speedster_striker":
            # Speedster wingers (Martinelli-type) hug the line and run in behind
            profile.byline_instinct = max(profile.byline_instinct, 0.65)
            profile.box_entry_instinct = max(profile.box_entry_instinct, 0.70)
            profile.flank_commitment = max(profile.flank_commitment, 0.88)

        # Specialty overrides
        specialties = getattr(dna, "specialties", []) or []
        if "crosser" in specialties:
            profile.cross_instinct = max(profile.cross_instinct, 0.70)
            profile.byline_instinct = max(profile.byline_instinct, 0.80)
        if "inverted" in specialties:
            profile.byline_instinct = min(profile.byline_instinct, 0.30)
            profile.cross_instinct = min(profile.cross_instinct, 0.25)
        if "speedster" in specialties:
            profile.byline_instinct = max(profile.byline_instinct, 0.60)
            profile.box_entry_instinct = max(profile.box_entry_instinct, 0.65)
        if "grand_dribbler" in specialties or "dribbler" in specialties:
            profile.isolation_thirst = max(profile.isolation_thirst, 0.80)
        if "pressing_forward" in specialties:
            profile.flank_commitment = max(profile.flank_commitment, 0.90)

        return profile


# ─────────────────────────────────────────────
# WINGER REGISTRY — per-team per-player profiles
# ─────────────────────────────────────────────

class WingerRegistry:
    """
    Holds WingerSpatialProfile for every winger in a match.
    Built once at kickoff, read every touch by the event chain and
    position engine.
    """

    def __init__(self):
        self.profiles: Dict[str, WingerSpatialProfile] = {}

    def register_player(self, player) -> WingerSpatialProfile:
        """Build + store a profile for a winger. Returns the profile."""
        position = getattr(player, "position", "")
        if position not in ("LW", "RW"):
            return None
        profile = WingerBehaviorEngine.build_profile_from_dna(player)
        name = getattr(player, "name", str(player))
        self.profiles[name] = profile
        return profile

    def register_team(self, players: List) -> None:
        """Register all wingers in a squad."""
        for p in players or []:
            self.register_player(p)

    def get(self, player_name: str) -> Optional[WingerSpatialProfile]:
        return self.profiles.get(player_name)

    def is_winger(self, player_name: str) -> bool:
        return player_name in self.profiles

    def remove(self, player_name: str) -> None:
        self.profiles.pop(player_name, None)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n🌊 PLOFA 26/27 — Modern Winger Behavior Engine Demo")
    print("=" * 64)

    # ── Duck-typed stand-ins ──
    class _Tend:
        cuts_inside = 0.20
        attempts_dribble = 0.45
        makes_runs_behind = 0.35

    class _DNA:
        archetype = "traditional_winger"
        specialties = ["crosser", "speedster"]
        tendencies = _Tend()

    class _Player:
        name = "Adri Vela"
        position = "LW"
        dna = _DNA()

    class _FakePE:
        def get_position(self, name):
            return {"Opp LB": (60.0, 12.0), "Opp RB": (60.0, 56.0)}.get(name, (50.0, 34.0))

    class _Def:
        def __init__(self, name, pos):
            self.name = name
            self.position = pos

    # ── Build profile ──
    profile = WingerBehaviorEngine.build_profile_from_dna(_Player())
    print(f"\n1. PROFILE for {_Player.name} ({_Player.position}):")
    print(f"   Flank:            {profile.flank}")
    print(f"   Touchline anchor: y={profile.touchline_anchor_y}")
    print(f"   Flank commitment: {profile.flank_commitment:.2f}")
    print(f"   Byline instinct:  {profile.byline_instinct:.2f}")
    print(f"   Isolation thirst: {profile.isolation_thirst:.2f}")
    print(f"   Box entry:        {profile.box_entry_instinct:.2f}")
    print(f"   Cross instinct:   {profile.cross_instinct:.2f}")

    # ── Geometry checks ──
    print("\n2. GEOMETRY CHECKS (attacking right):")
    test_points = [
        ("own half, wide",        45.0, 12.0),
        ("mid third, wide",       60.0, 12.0),
        ("att third, wide",       80.0, 12.0),
        ("byline, wide",          100.0, 10.0),
        ("drifted central",       80.0, 34.0),
        ("box entry zone",        95.0, 14.0),
    ]
    print(f"   {'Location':<24} {'In flank':>9} {'In cross':>9} {'In box':>7} {'Danger':>7}")
    for label, x, y in test_points:
        print(f"   {label:<24} {str(profile.flank_channel(y)):>9} "
              f"{str(profile.in_cross_zone(x, y, True)):>9} "
              f"{str(profile.in_box_entry_zone(x, y, True)):>7} "
              f"{profile.danger_zone_score(x, y, True):>7.2f}")

    # ── Isolation check ──
    print("\n3. FULLBACK ISOLATION (1v1):")
    defenders = [_Def("Opp LB", "LB"), _Def("Opp CB", "CB")]
    pe = _FakePE()
    isolated, fb, dist = profile.fullback_isolation(
        80.0, 12.0, defenders, pe, attacks_right=True
    )
    print(f"   At (80, 12) vs LB: isolated={isolated}, fullback={fb.name if fb else None}, dist={dist:.1f}m")

    # ── Decision steering ──
    print("\n4. DECISION STEERING (attacking right, wide at x=80):")
    print(f"   Drive byline (isolated):     {WingerBehaviorEngine.should_drive_byline(profile, 80, 12, True, isolated=True)}")
    print(f"   Drive byline (open):         {WingerBehaviorEngine.should_drive_byline(profile, 80, 12, True)}")
    print(f"   Cut inside (half-space open):{WingerBehaviorEngine.should_cut_inside(profile, 80, 12, True, half_space_open=True)}")
    print(f"   Cross (in cross zone):       {WingerBehaviorEngine.should_cross(profile, 80, 12, True)}")
    print(f"   Enter box (opp flank):       {WingerBehaviorEngine.should_enter_box(profile, 80, 12, True, ball_on_opposite_flank=True)}")

    print("\n✅ Modern Winger Behavior Engine operational — pure geometry, zero deps.")