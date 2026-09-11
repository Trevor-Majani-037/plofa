"""
PLOFA 26/27 — THREAT / DANGER INTELLIGENCE  (Checkpoint 9)
===========================================================
threat_engine.py

Philosophy:
    A defender's only job is to keep the ball away from the goalpost xy of the
    goal they defend. Every threat, every clearance, every body-in-the-box is
    just an attempt to LOWER the danger level — and football being football,
    sometimes the opponent wins and the danger level PEAKS as a goal goes in.

    This module is the shared "awareness" every defender reads. It turns pure
    pitch geometry into a live, per-team DANGER LEVEL (0-100):

        Danger rises  as the ball approaches the defended goalpost xy.
        Danger falls  when the defence wins it back (tackle/interception/block)
                      or clears it away from the goal (foot OR headed).
        Danger peaks  the moment a goal is conceded, then resets to the low
                      kickoff baseline when play restarts.

    It is deliberately pure — no dependency on the event-chain or match-engine
    loops. It consumes (ball_x, ball_y, which goal, who's near) and returns a
    DangerAssessment. The match engine feeds it events; DefensiveChain and
    PositionEngine read its output to pick actions and set the block.

    Components:
        DangerAssessment — one assessment of the ball position vs a defended goal
        ThreatEngine     — live per-team danger, history, relief, reporting
        DANGER_BANDS     — LOW / MODERATE / HIGH / CRITICAL classification
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# DANGER BANDS — shared classification
# ─────────────────────────────────────────────

#: Upper bound (inclusive) of each danger band. Danger is 0-100.
DANGER_BANDS: Dict[str, float] = {
    "LOW":      30.0,   # ball comfortably away, defence reorganising
    "MODERATE": 60.0,   # ball in the defensive third, pressure building
    "HIGH":     85.0,   # ball in/near the box — clear the lines
    "CRITICAL": 100.0,  # six-yard scramble, bodies on the line
}


def band_of(danger: float) -> str:
    """Classify a 0-100 danger level into a named band."""
    for name, upper in DANGER_BANDS.items():
        if danger <= upper:
            return name
    return "CRITICAL"


# ─────────────────────────────────────────────
# ZONE GEOMETRY — measured from the DEFENDED goal
# Mirrors PitchZone.xg_zone but expressed from the goal being defended,
# so it works for both teams without direction arguments.
# ─────────────────────────────────────────────

def _distance_from_goal_line(ball_x: float, own_goal_x: float) -> float:
    return abs(ball_x - own_goal_x)


def danger_zone(ball_x: float, own_goal_x: float) -> str:
    """Zone of the ball relative to the goal being defended.

    Thresholds match PitchZone.xg_zone (six-yard = 6m, box = 22m,
    edge of box = 35m, outside box = 70m) so exports line up with the
    existing shot-map / xG zone vocabulary.
    """
    d = _distance_from_goal_line(ball_x, own_goal_x)
    if d <= 6.0:
        return "six_yard_box"
    if d <= 22.0:
        return "inside_box"
    if d <= 35.0:
        return "edge_of_box"
    if d <= 70.0:
        return "outside_box"
    return "deep"


# Zone multiplier — the closer the zone is to the goal, the more it amplifies
# the raw proximity danger. Non-decreasing as the ball approaches the goal,
# which (together with monotonic proximity) guarantees overall monotonicity.
ZONE_MULTIPLIERS: Dict[str, float] = {
    "six_yard_box": 1.30,
    "inside_box":   1.15,
    "edge_of_box":  1.05,
    "outside_box":  0.95,
    "deep":         0.50,
}

#: Distance beyond which the ball poses zero threat to the defended goal.
ZERO_THREAT_DISTANCE: float = 70.0


# ─────────────────────────────────────────────
# DANGER ASSESSMENT — one geometric snapshot
# ─────────────────────────────────────────────

@dataclass
class DangerAssessment:
    """A single read of "how dangerous is the ball for the team defending
    the goal at `own_goal_x` right now."

    `level` is the composite 0-100 danger. The other fields expose the
    factor decomposition so callers (and the export layer) can see WHY.
    """
    level: float = 0.0
    proximity: float = 0.0      # 0-1: how close to the goal line
    centrality: float = 0.0     # 0-1: how central vs wide (1 = dead central)
    zone: str = "deep"
    zone_mult: float = 0.5      # zone multiplier applied
    pressure: float = 0.0       # attacker-vs-defender density around the ball
    momentum: float = 0.0       # 0-1: is the ball advancing toward the goal
    shot_mult: float = 1.0      # boost when the ball is in a shooting position
    dist_to_goal: float = 0.0   # euclidean metres ball → goal centre
    ball_x: float = 0.0
    ball_y: float = 0.0
    attackers_near: int = 0
    defenders_near: int = 0

    @property
    def band(self) -> str:
        return band_of(self.level)

    def as_dict(self) -> Dict:
        return {
            "level": round(self.level, 1),
            "band": self.band,
            "proximity": round(self.proximity, 3),
            "centrality": round(self.centrality, 3),
            "zone": self.zone,
            "pressure": round(self.pressure, 3),
            "momentum": round(self.momentum, 3),
            "shot_mult": round(self.shot_mult, 3),
            "dist_to_goal": round(self.dist_to_goal, 1),
            "ball_x": round(self.ball_x, 1),
            "ball_y": round(self.ball_y, 1),
        }


def assess(
    ball_x: float,
    ball_y: float,
    own_goal_x: float,
    attackers_near: int = 0,
    defenders_near: int = 0,
    approach: float = 1.0,
) -> DangerAssessment:
    """Compute the danger level for a ball position vs a defended goal.

    Args:
        ball_x: ball x (0-105)
        ball_y: ball y (0-68)
        own_goal_x: x of the goal being defended (0 for home, 105 for away)
        attackers_near: attacking players within ~8m of the ball
        defenders_near: defending players within ~8m of the ball
        approach: 1.0 if the ball is advancing toward the defended goal,
                  0.0 if it is retreating (1.0 default = neutral threat)
    """
    dist = math.hypot(ball_x - own_goal_x, ball_y - 34.0)

    # Proximity — monotonic: closer ⇒ strictly higher.
    proximity = max(0.0, min(1.0, 1.0 - dist / ZERO_THREAT_DISTANCE))

    # Centrality — central balls are far more dangerous than wide ones.
    centrality = 1.0 - min(1.0, abs(ball_y - 34.0) / 28.0)

    zone = danger_zone(ball_x, own_goal_x)
    zone_mult = ZONE_MULTIPLIERS.get(zone, 0.5)

    # Pressure — an attacker outnumbering the defence at the ball is worse.
    # The factor sits at 0.875 for even numbers (ball in the box is ALREADY
    # dangerous on its own) and climbs toward 1.0 as attackers pile in, or
    # drops toward 0.75 when the defence outnumbers the ball.
    density = 0.5 + 0.15 * (attackers_near - defenders_near)
    pressure = 0.75 + 0.25 * max(0.0, min(1.0, density))

    # Momentum — a ball advancing toward the goal is a live threat.
    momentum = 0.65 + 0.35 * max(0.0, min(1.0, approach))

    # Shot position — within 30m of goal and with a workable angle.
    dist_from_line = abs(ball_x - own_goal_x)
    y_off = abs(ball_y - 34.0)
    angle = math.degrees(math.atan2(y_off, dist_from_line)) if dist_from_line > 0.01 else 90.0
    shot_pos = dist_from_line < 30.0 and angle < 60.0
    shot_mult = 1.15 if shot_pos else 1.0

    level = 100.0 * min(
        1.0,
        proximity ** 1.5
        * zone_mult
        * (0.55 + 0.45 * centrality)
        * pressure
        * momentum
        * shot_mult,
    )

    return DangerAssessment(
        level=round(level, 1),
        proximity=round(proximity, 3),
        centrality=round(centrality, 3),
        zone=zone,
        zone_mult=zone_mult,
        pressure=round(pressure, 3),
        momentum=round(momentum, 3),
        shot_mult=shot_mult,
        dist_to_goal=round(dist, 1),
        ball_x=round(ball_x, 1),
        ball_y=round(ball_y, 1),
        attackers_near=attackers_near,
        defenders_near=defenders_near,
    )


# ─────────────────────────────────────────────
# DANGER RELIEF — what a defensive win does to the danger level
# ─────────────────────────────────────────────

def danger_after_clearance(
    old_level: float,
    own_goal_x: float,
    from_x: float,
    from_y: float,
    to_x: float,
    to_y: float,
    success: bool,
) -> float:
    """Danger level after a clearance.

    A failed clearance barely moves the ball — the danger barely falls.
    An effective clearance moves the ball away from the goal by `moved_away`
    metres; each 45m of relief removes up to 75% of the remaining danger.
    """
    if not success:
        return round(old_level * 0.95, 1)
    dist_from = math.hypot(from_x - own_goal_x, from_y - 34.0)
    dist_to = math.hypot(to_x - own_goal_x, to_y - 34.0)
    moved_away = dist_to - dist_from
    relief = max(0.15, min(1.0, moved_away / 45.0))
    return round(old_level * (1.0 - 0.75 * relief), 1)


#: Flat relief applied when the defence WINS the ball without clearing it
#: (tackle / interception / block / recovery) — the danger drops but the ball
#: is still often in the defensive half.
WIN_RELIEF_FACTOR: float = 0.80


# ─────────────────────────────────────────────
# BODY ORIENTATION & BIOMECHANICS
# ─────────────────────────────────────────────
# The second pillar of spatial awareness: the defender's BODY ORIENTATION
# relative to the ball. Two vectors — the defender's facing vector (hips/chest
# to the point they're looking at) and the ball vector (hips/chest to the ball)
# — combine into a signed angle in [-180, +180]:
#
#       0°   the ball is dead in front  → OPTIMAL (full power, baseline risk)
#     ±90°   the ball is on a flank     → FLANK (reach/torso twist, slight risk)
#    ±180°   the ball is behind         → BLIND/PANIC (backward overhead kick,
#                                        sliced clearances, own goals)
#
# All of the following are PURE geometry — no dependency on PlayerProfile, the
# event chain or the match loop — so they are trivially unit-testable.

def calculate_relative_ball_angle(
    d_x: float, d_y: float,
    f_x: float, f_y: float,
    b_x: float, b_y: float,
) -> float:
    """Signed angle (degrees, -180..+180) of the ball relative to the
    defender's heading.

    0      = ball directly in front of the defender
    +90    = ball directly on the defender's left flank
    -90    = ball directly on the defender's right flank
    ±180   = ball directly behind the defender

    Uses atan2 so division-by-zero and wrap-around edge cases are handled
    automatically.
    """
    facing_angle = math.atan2(f_y - d_y, f_x - d_x)
    ball_angle = math.atan2(b_y - d_y, b_x - d_x)
    relative_angle = ball_angle - facing_angle
    relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(relative_angle)


def defender_facing_point(
    def_x: float, def_y: float,
    ball_x: float, ball_y: float,
    own_goal_x: float,
) -> Tuple[float, float]:
    """Derive which way a defender is facing from pitch geometry alone.

    A defender still goal-side of the ball faces the ball (Optimal zone —
    they can step in and clear cleanly). A defender who has been BEATEN —
    the ball is closer to their own goalpost xy than they are — is sprinting
    back toward their own goal, so they face their own goal (the Blind/Panic
    zone where sliced clearances and own goals live).
    """
    if abs(own_goal_x - ball_x) < abs(own_goal_x - def_x):
        return own_goal_x, def_y   # ball is in behind → facing own goal
    return ball_x, ball_y          # ball in front → facing the ball


#: Biomechanical zones and their P_fail multipliers. `max_abs_angle` is the
#: inclusive upper bound of |angle| for that zone.
ORIENTATION_ZONES: Dict[str, Dict[str, float]] = {
    "optimal": {"max_abs_angle": 30.0,  "failure_mult": 1.0,  "label": "Optimal Clearance Zone"},
    "flank":   {"max_abs_angle": 90.0,  "failure_mult": 1.15, "label": "Flank Adjustment Zone"},
    "blind":   {"max_abs_angle": 180.0, "failure_mult": 1.40, "label": "Blind / Panic Zone"},
}


def orientation_zone(angle_deg: float) -> str:
    """Classify a relative ball angle into its biomechanical zone."""
    abs_a = abs(angle_deg)
    if abs_a <= ORIENTATION_ZONES["optimal"]["max_abs_angle"]:
        return "optimal"
    if abs_a <= ORIENTATION_ZONES["flank"]["max_abs_angle"]:
        return "flank"
    return "blind"


def clearance_failure_multiplier(
    angle_deg: float = 0.0,
    contested_distance: Optional[float] = None,
    stamina: Optional[float] = None,
) -> float:
    """Combined P_fail amplifier for a clearance.

    Stacked, multiplicatively:
      • body orientation   — optimal 1.0 / flank 1.15 / blind 1.40
      • contested          — an attacker within ~2m (spec: 0.5m = fully
                             contested) amplifies failure up to +40%
      • fatigue (E)        — stamina below 50% adds up to +35% (89th-minute
                             mistimed jumps / sliced kicks)
    """
    zone = orientation_zone(angle_deg)
    mult = ORIENTATION_ZONES[zone]["failure_mult"]

    if contested_distance is not None:
        d = max(0.0, contested_distance)
        if d < 2.0:
            contested_factor = 1.0 + (1.0 - d / 2.0) * 0.40
            mult *= contested_factor

    if stamina is not None:
        s = max(0.0, min(100.0, stamina))
        if s < 50.0:
            mult *= 1.0 + (50.0 - s) / 50.0 * 0.35

    return round(mult, 3)


def clearance_foot_for_angle(angle_deg: float, preferred_foot: str = "right") -> str:
    """Which foot a defender uses for a foot clearance.

    Within the Optimal dead-zone (±30°) the defender uses their preferred
    foot. Outside it the sign of the angle dictates the foot — a ball on the
    left flank (+angle) is met with the left foot, a ball on the right flank
    with the right — mirroring the stance of a real centre-back.
    """
    if abs(angle_deg) <= 30.0:
        foot = preferred_foot
    elif angle_deg > 0.0:
        foot = "left"
    else:
        foot = "right"
    return "right_foot" if foot != "left" else "left_foot"


def clearance_target_vector(
    from_x: float, from_y: float,
    own_goal_x: float,
) -> Tuple[float, float]:
    """The pure Target Clearance Vector: V_clear = V_away (+ Y_bias applied
    downstream via apply_width_bias).

    V_away = B(x, y) - G(x, y) — points straight away from the defended
    goalpost xy. The width bias (apply_width_bias) then steers the landing
    point toward a touchline so the clearance never feeds Zone 14.
    """
    return round(from_x - own_goal_x, 3), round(from_y - 34.0, 3)


def apply_width_bias(
    end_x: float, end_y: float,
    own_goal_x: float,
    bias: float = 10.0,
) -> float:
    """Push a clearance landing point toward the touchlines.

    A ball that would come down in the central Zone-14 corridor (within
    22–40m of the defended goal line, y central) is the exact danger an
    intelligent defender avoids — it hands the second ball to an attacker
    in the shooting channel. Such balls are nudged to the nearer side.
    """
    d_line = abs(end_x - own_goal_x)
    central = abs(end_y - 34.0) < 8.0
    if 22.0 <= d_line <= 40.0 and central:
        # Nudge toward the NEARER touchline: push up toward y=68 if the ball
        # sits above the centre line, down toward y=0 if below it.
        toward_side = 1.0 if end_y >= 34.0 else -1.0
        return round(max(2.0, min(66.0, end_y + toward_side * bias)), 1)
    return round(end_y, 1)


def own_goal_probability(
    angle_deg: float = 0.0,
    contested_distance: Optional[float] = None,
    stamina: Optional[float] = None,
    danger_level: float = 0.0,
    base: float = 0.012,
) -> float:
    """Probability a FAILED clearance turns into a catastrophic own goal.

    The Critical-Failure chance. It explodes exactly where real defenders
    panic: ball behind them (blind zone), an attacker breathing down their
    neck (<1m), empty legs in the 89th minute, and CRITICAL danger — the
    desperate backwards header or the shinned swing that redirects the ball
    into the defender's own net.
    """
    p = base
    zone = orientation_zone(angle_deg)
    if zone == "blind":
        p *= 3.5
    elif zone == "flank":
        p *= 1.6

    if contested_distance is not None and contested_distance < 1.0:
        p *= 2.0
    if stamina is not None and stamina < 40.0:
        p *= 1.8

    p *= 0.5 + max(0.0, min(100.0, danger_level)) / 100.0
    return round(max(0.0, min(0.20, p)), 5)


# ─────────────────────────────────────────────
# THREAT ENGINE — live per-team danger
# ─────────────────────────────────────────────

# Event types that represent the defence winning the ball cleanly. On these,
# danger relief is applied to the winner's own danger level.
DEFENSIVE_WIN_TYPES = {
    "INTERCEPTION", "TACKLE_WON", "BLOCK", "RECOVERY", "BALL_RECOVERY",
    "PRESS_SUCCESS", "GOAL_KICK",
}

# Event types that are clearances (relief scaled by distance moved).
CLEARANCE_TYPES = {"CLEARANCE"}

# Event types that move the ball toward the danger but are not yet a shot.
THREAT_PROGRESSION_TYPES = {
    "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "CROSS_ATTEMPT",
    "CROSS_SUCCESS", "THROUGH_BALL", "CARRY", "DRIBBLE_SUCCESS",
    "BALL_RECEIPT", "DRIBBLE_ATTEMPT",
}

# Event types that ARE a cross delivery by construction (the engine's own
# crossing events). Any event carrying the geometric CrossDetector's
# `cross: true` / `is_airborne` metadata (including ordinary passes
# reclassified as crosses) also qualifies — see observe_event.
CROSS_TRIGGER_TYPES = {
    "CROSS_ATTEMPT", "CROSS_SUCCESS", "CORNER_TAKEN", "FREEKICK_CROSS",
}


class ThreatEngine:
    """
    The shared awareness a defensive unit reads all match.

    Home team defends the goal at x=0; away team defends the goal at x=105.
    Danger for each team is simply "how close is the ball to MY goalpost xy",
    updated live from every timeline event, with relief on defensive wins and
    a peak + reset when a goal goes in.
    """

    def __init__(self, home_team: str, away_team: str):
        self.home_team = home_team
        self.away_team = away_team
        self._danger: Dict[str, float] = {home_team: 0.0, away_team: 0.0}
        self._prev_ball: Optional[Tuple[float, float]] = None
        self._samples: List[Tuple[int, float, float]] = []      # (minute, home, away)
        self.clearances: Dict[str, Dict[str, int]] = {
            "headed": {home_team: 0, away_team: 0},
            "foot":   {home_team: 0, away_team: 0},
        }
        self.goals_conceded: Dict[str, int] = {home_team: 0, away_team: 0}

    # ── CONFIG ─────────────────────────────────────────

    def own_goal_x(self, team: str) -> float:
        return 0.0 if team == self.home_team else 105.0

    def is_home(self, team: str) -> bool:
        return team == self.home_team

    # ── LIVE DANGER QUERY ─────────────────────────────

    def danger_at(self, team: str) -> float:
        """The current live danger level for a team (0-100)."""
        return self._danger.get(team, 0.0)

    def set_danger(self, team: str, level: float):
        self._danger[team] = round(max(0.0, min(100.0, level)), 1)

    def assessment_at(
        self,
        team: str,
        ball_x: float,
        ball_y: float,
        attackers_near: int = 0,
        defenders_near: int = 0,
        approach: float = 1.0,
    ) -> DangerAssessment:
        """A fresh, on-demand assessment for a team at arbitrary ball coords
        (used when the threat is evaluated at a context position, not the
        last event's)."""
        return assess(ball_x, ball_y, self.own_goal_x(team),
                      attackers_near=attackers_near,
                      defenders_near=defenders_near,
                      approach=approach)

    # ── LIVE UPDATES FROM THE TIMELINE ─────────────────

    def observe_event(self, event, minute: int,
                      near_counts: Optional[Dict[str, int]] = None) -> None:
        """
        Consume one timeline event and keep both teams' danger live.

        The ball position is taken from the event (end_x/end_y preferred).
        Both teams' danger is recomputed from the pure geometry — the ball
        near MY goal is dangerous to ME regardless of who touched it last.
        `near_counts` maps team → number of that team's outfield players
        within ~8m of the ball (used for the pressure factor). Defensive wins
        and clearances then apply relief to the winning team.
        """
        bx = event.end_x if event.end_x is not None else event.location_x
        by = event.end_y if event.end_y is not None else event.location_y
        if bx is None or by is None:
            return

        prev = self._prev_ball
        self._prev_ball = (bx, by)

        for team in (self.home_team, self.away_team):
            goal_x = self.own_goal_x(team)
            approach = self._approach_direction(prev, bx, by, goal_x)
            if near_counts:
                opp = self.away_team if team == self.home_team else self.home_team
                att = near_counts.get(opp, 0)
                dfd = near_counts.get(team, 0)
            else:
                att = dfd = 0
            a = assess(bx, by, goal_x,
                       attackers_near=att,
                       defenders_near=dfd,
                       approach=approach)
            self._danger[team] = a.level

        # ── CHECKPOINT 11: CROSS TRIGGER ─────────────────────────
        # A detected cross into the box forces the localized danger level
        # HIGH/CRITICAL (D >= 75). The resting-position geometry above
        # understates a whipped ball that is still travelling into the box
        # among the bodies, so any cross — an engine CROSS_ATTEMPT/SUCCESS
        # event OR any delivery stamped `cross: true` / `is_airborne` by the
        # geometric CrossDetector (including reclassified generic passes) —
        # that finishes in a team's box (inside-box / six-yard, NOT merely
        # the edge) raises that team's floor to 75. This is what hands the
        # defence to the headed-clearance matrix.
        name = getattr(event, "event_type", None)
        etype = name.name if name is not None else ""
        meta = getattr(event, "metadata", None) or {}
        is_cross_meta = bool(meta.get("cross") or meta.get("is_airborne"))
        if etype in CROSS_TRIGGER_TYPES or is_cross_meta:
            for team in (self.home_team, self.away_team):
                if danger_zone(bx, self.own_goal_x(team)) in (
                    "inside_box", "six_yard_box"
                ):
                    if self._danger[team] < 75.0:
                        self._danger[team] = 75.0

        # Relief for the defending side that won the ball / cleared it.
        name = getattr(event, "event_type", None)
        etype = name.name if name is not None else ""
        team = getattr(event, "team", "")

        if etype in CLEARANCE_TYPES and team in self._danger:
            body = getattr(event, "body_part", "") or "right_foot"
            kind = "headed" if body == "head" else "foot"
            self.clearances[kind][team] = self.clearances[kind].get(team, 0) + 1
            old = self._danger[team]
            to_x = bx
            to_y = by
            from_x = getattr(event, "location_x", None) or to_x
            from_y = getattr(event, "location_y", None) or to_y
            success = getattr(event, "outcome", True)
            self.set_danger(team, danger_after_clearance(
                old, self.own_goal_x(team),
                from_x, from_y, to_x, to_y, success,
            ))

        elif etype in DEFENSIVE_WIN_TYPES and team in self._danger:
            old = self._danger[team]
            self.set_danger(team, old * WIN_RELIEF_FACTOR)

        self._record_sample(minute)

    def _approach_direction(self, prev, bx, by, goal_x) -> float:
        """1.0 if the ball moved toward the defended goal, 0.0 otherwise."""
        if prev is None:
            return 1.0
        return 1.0 if abs(bx - goal_x) < abs(prev[0] - goal_x) else 0.0

    def _record_sample(self, minute: int):
        self._samples.append((minute, self._danger[self.home_team],
                              self._danger[self.away_team]))

    # ── GOAL FLOW ──────────────────────────────────────

    def on_cross(self, team: str, minute: int) -> None:
        """A cross delivery has been detected into `team`'s box — the
        localized danger level is forced HIGH/CRITICAL (D >= 75)."""
        self.set_danger(team, max(self.danger_at(team), 75.0))
        self._record_sample(minute)

    def on_goal(self, conceding_team: str, minute: int) -> None:
        """The threat was realised — the danger peaks, then the kickoff resets
        it back to the low baseline."""
        self.goals_conceded[conceding_team] = self.goals_conceded.get(conceding_team, 0) + 1
        self.set_danger(conceding_team, 100.0)
        self._record_sample(minute)

    def on_kickoff(self, minute: int) -> None:
        """Ball back at the centre circle — both teams start clean."""
        self._danger = {self.home_team: 0.0, self.away_team: 0.0}
        self._prev_ball = None
        self._record_sample(minute)

    # ── REPORTING / EXPORT ─────────────────────────────

    def minute_averages(self) -> Dict[str, Dict[int, float]]:
        """Per-minute average danger for each team (keys 1..minute max)."""
        by_minute: Dict[str, Dict[int, List[float]]] = {
            self.home_team: {}, self.away_team: {},
        }
        for minute, h, a in self._samples:
            by_minute[self.home_team].setdefault(minute, []).append(h)
            by_minute[self.away_team].setdefault(minute, []).append(a)
        out = {}
        for team in (self.home_team, self.away_team):
            out[team] = {
                m: round(sum(v) / len(v), 1) for m, v in by_minute[team].items()
            }
        return out

    def report(self) -> Dict:
        """Full defensive-awareness summary, ready for the exporter."""
        minute_avg = self.minute_averages()
        report = {}
        for team in (self.home_team, self.away_team):
            mins = minute_avg.get(team, {})
            values = list(mins.values())
            peak = max(values) if values else 0.0
            avg = sum(values) / len(values) if values else 0.0
            counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
            for v in values:
                counts[band_of(v)] += 1
            report[team] = {
                "peak_danger": round(peak, 1),
                "avg_danger": round(avg, 1),
                "minutes_low": counts["LOW"],
                "minutes_moderate": counts["MODERATE"],
                "minutes_high": counts["HIGH"],
                "minutes_critical": counts["CRITICAL"],
                "clearances_headed": self.clearances["headed"].get(team, 0),
                "clearances_foot": self.clearances["foot"].get(team, 0),
                "goals_conceded": self.goals_conceded.get(team, 0),
                "danger_timeline": sorted(mins.items()),
            }
        return report


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python threat_engine.py
# Verifies the module works with ZERO dependency on the rest of PLOFA.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🛡️  PLOFA 26/27 — Threat Intelligence (Checkpoint 9) Standalone Demo")
    print("=" * 64)

    print("\n1. DANGER vs BALL PROXIMITY (away defends goal at x=105, ball central)\n")
    print(f"  {'Ball X':>7} {'Zone':<14} {'Danger':>7} {'Band':<10}")
    for bx in [102, 99, 95, 88, 78, 70, 60, 45, 35, 20, 10, 0]:
        a = assess(bx, 34.0, 105.0)
        print(f"  {bx:>7.1f} {a.zone:<14} {a.level:>6.1f} {a.band:<10}")

    print("\n2. CENTRAL vs WIDE (same x=95, away defends goal at x=105)\n")
    for by in [34.0, 44.0, 55.0, 63.0]:
        a = assess(95.0, by, 105.0)
        bar = "█" * int(a.level / 5)
        print(f"  y={by:>4.0f}  centrality={a.centrality:>4.2f}  "
              f"danger={a.level:>6.1f}  {bar}")

    print("\n3. ZONE ORDERING (central, advancing, even pressure)\n")
    for bx, zone in [(102, "six_yard"), (95, "inside_box"), (82, "edge_of_box"),
                     (60, "outside_box"), (20, "deep")]:
        a = assess(bx, 34.0, 105.0)
        print(f"  {zone:<12} x={bx:>5.1f}  danger={a.level:>6.1f}")

    print("\n4. CLEARANCE RELIEF (from the 6-yard box, away defends x=105)\n")
    old = 90.0
    for to_x, to_y, success, label in [
        (100.0, 40.0, True,  "headed to box edge"),
        (85.0,  60.0, True,  "headed to touchline"),
        (55.0,  34.0, True,  "foot hoof to midfield"),
        (100.0, 40.0, False, "scuffed, ball stays in box"),
    ]:
        after = danger_after_clearance(old, 105.0, 100.0, 40.0, to_x, to_y, success)
        print(f"  {label:<28} {old:>5.1f} → {after:>5.1f}")

    print("\n5. THREAT ENGINE LIVE TRACKING (mini sequence)\n")
    from types import SimpleNamespace

    te = ThreatEngine("Hartwell City", "Thornfield United")
    evs = [
        SimpleNamespace(event_type=SimpleNamespace(name="PASS"), team="Thornfield United",
                        end_x=60.0, end_y=34.0, location_x=70.0, location_y=34.0,
                        outcome=True),
        SimpleNamespace(event_type=SimpleNamespace(name="PASS"), team="Thornfield United",
                        end_x=80.0, end_y=34.0, location_x=60.0, location_y=34.0,
                        outcome=True),
        SimpleNamespace(event_type=SimpleNamespace(name="CARRY"), team="Thornfield United",
                        end_x=95.0, end_y=34.0, location_x=80.0, location_y=34.0,
                        outcome=True),
        SimpleNamespace(event_type=SimpleNamespace(name="CLEARANCE"), team="Hartwell City",
                        body_part="head", end_x=60.0, end_y=30.0,
                        location_x=100.0, location_y=38.0, outcome=True),
    ]
    for i, e in enumerate(evs):
        te.observe_event(e, minute=70 + i)
        print(f"  {i}: {e.event_type.name:<11} → "
              f"home={te.danger_at('Hartwell City'):>5.1f}  "
              f"away={te.danger_at('Thornfield United'):>5.1f}")

    print("\n6. BODY ORIENTATION — relative ball angle vs defender heading\n")
    for label, (d, b, f) in [
        ("ball in front",     ((90, 34), (92, 34), (93, 34))),
        ("left flank",        ((90, 34), (92, 40), (93, 34))),
        ("right flank",       ((90, 34), (92, 28), (93, 34))),
        ("ball behind",       ((90, 34), (88, 34), (93, 34))),
        ("over right shoulder", ((90, 34), (88, 40), (93, 34))),
    ]:
        ang = calculate_relative_ball_angle(*d, *f, *b)
        print(f"  {label:<22} angle={ang:>7.1f}°  zone={orientation_zone(ang):<8}")

    print("\n7. BIOMECHANICAL P_fail AMPLIFIERS (contested + fatigue)\n")
    print(f"  {'scenario':<32} {'P_fail mult':>10}")
    for label, ang, contested, stam in [
        ("clean, ball in front",          0.0, None, 100.0),
        ("flank, slight pressure",        60.0, 1.0, 80.0),
        ("blind, striker breathing",      170.0, 0.3, 72.0),
        ("blind, contested, 89th min",    175.0, 0.4, 22.0),
    ]:
        print(f"  {label:<32} {clearance_failure_multiplier(ang, contested, stam):>10.2f}")

    print("\n8. OWN-GOAL CRITICAL FAILURE PROBABILITY\n")
    for label, ang, contested, stam, danger in [
        ("calm 6-yard header",           10.0,  2.5, 95.0, 70.0),
        ("desperate blind slice",       175.0,  0.4, 60.0, 95.0),
        ("89th-min panic, smothered",   180.0,  0.2, 15.0, 100.0),
    ]:
        p = own_goal_probability(ang, contested, stam, danger)
        print(f"  {label:<30} own-goal P={p*100:>6.2f}%")

    print("\n9. TARGET CLEARANCE VECTOR + WIDTH BIAS (Zone-14 avoidance)\n")
    vx, vy = clearance_target_vector(96.0, 34.0, 105.0)
    print(f"  away defend x=105, ball (96,34): V_away=({vx},{vy})")
    for end_x, end_y in [(75.0, 40.0), (75.0, 55.0), (60.0, 40.0)]:
        y_biased = apply_width_bias(end_x, end_y, 105.0)
        print(f"  landing ({end_x}, {end_y}) → ({end_x}, {y_biased})"
              f"{'  [Zone-14 nudge]' if y_biased != end_y else ''}")

    print("\n✅ Threat Intelligence module operational — zero dependency on rest of PLOFA.")
    print("   Next: wire into event_chain.py (DefensiveChain) + match_engine.py minute loop.\n")
