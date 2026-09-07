"""
PLOFA 26/27 — MARKING ENGINE  (P1: man/zonal marking layer)
============================================================
marking.py

Why this exists:
    PLOFA's defensive system was GEOMETRIC but not PLAYER-AWARE. The
    coordinated block (PositionEngine.defensive_block) pulled bodies toward
    ball-relative coordinates with zero reference to where the OPPONENT'S
    attackers were. That is the single biggest difference between a
    "defensive shape" (a set of coordinates) and a real defensive TEAM
    (a set of defenders each responsible for an attacker).

    Real football defence is assignment-based:
        • the tall centre-back carries the opposition's aerial threat,
        • the near-side fullback picks up the opposition winger who is
          breaking into the channel,
        • a striker free in the box gets "picked up" by SOMEONE before
          he can finish,
        • a defender whose man has made a run "on the blindside" is
          beaten, and the danger that implies is real.

    This module is the pure decision layer for that assignment problem. It
    has no dependency on the event chain / RNG / position engine (positions
    are passed in), mirroring threat_engine.py, so it is trivially
    unit-testable.

Design:
    MarkingEngine.assign() solves a one-sided assignment: for a set of
    DEFENDERS it chooses, for each, the single ATTACKER to cover. The real
    assignment starts from the BALL and the danger: the closer/more
    dangerous an attacker is to the defended goal, the higher his
    "cover priority". We then assign defenders greedily in priority order
    (an aerial threat draws the best aerial CB; the nearest fullback takes
    the breaking winger), awarding a pair cost by defender relevance and
    distance.

    Output                     MarAssignment
        attacker_name          who they cover ("" = none free)
        attacker_x/y           attacker's current position
        priority               the attacker's cover priority
        tightness              0..1 how well they are on the man
        marker_state           "tight" | "free" | "beaten"
        goal_side_dist         how far goal-side of the attacker they are

    marker_state semantics (shared vocabulary for block + corner + export):
        tight   — defender is on the man's goal side, within marking range
        free    — defender assigned but goal-side/wide (risk of a gap)
        beaten  — the attacker has won the race / is on the blindside; the
                  defender must recover goal-ward, not chase the man

Pitch model: x in [0,105] (goal lines), y in [0,68] (touchlines).
own_goal_x is the goal THE DEFENDER IS DEFENDING (0 home, 105 away).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants of the marking model
# ---------------------------------------------------------------------------

#: An attacker inside this many metres of the defended goal is "in the box"
#: and eligible for a (near-ubiquitous) pick-up mark. Outside it, marking is
#: looser (space coverage) unless the attacker is a runner.
BOX_MARK_RADIUS: float = 22.0

#: Below this distance between a defender and his assigned attacker the mark
#: is "tight".
TIGHT_MARK_DIST: float = 2.2

#: Above this distance the defender has lost his man entirely ("beaten" —
#: the attacker is gone and the defender must recover to a goal-side slot).
BEATEN_DIST: float = 6.0

#: Priority weight for attackers who are a live aerial threat (headed
#: finish / cross target). Drives the "best CB carries the big target" rule.
AERIAL_THREAT_WEIGHT: float = 2.2

#: A forward runner breaking toward the box on the blindside of his defender
#: is the most dangerous situation; this boosts any attacker one goal-side
#: of the nearest defender.
BEHIND_LINE_BOOST: float = 1.8

#: How much defender role relevance dominates raw distance when choosing who
#: marks whom. A CB two metres further away from the ST still draws the ST.
RELEVANCE_WEIGHT: float = 1.4


@dataclass
class MarkAssignment:
    """One defender's marking assignment (or none)."""
    defender_name: str
    defender_position: str
    attacker_name: str = ""
    attacker_x: float = 52.5
    attacker_y: float = 34.0
    priority: float = 0.0
    tightness: float = 0.0
    marker_state: str = "free"
    goal_side_dist: float = 0.0

    @property
    def covers_someone(self) -> bool:
        return bool(self.attacker_name)


class MarkingEngine:
    """
    Pure assignment solver. Stateless: every call recomputes the marking
    from the current snapshot.
    """

    @staticmethod
    def attacker_priority(
        ax: float, ay: float,
        own_goal_x: float,
        aerial: float = 50.0,
        pace: float = 50.0,
        is_runner: bool = False,
    ) -> float:
        """
        How badly must this attacker be covered, independent of any specific
        defender. Grows as the attacker approaches the defended goal, is
        central, is a live aerial target (corner/cross), or is a runner on
        the break.
        """
        dist = abs(ax - own_goal_x)
        # Proximity: linear from 0 at ~45m out to 1.0 in the six-yard box.
        prox = max(0.0, min(1.0, (45.0 - dist) / 45.0))
        # Centrality: an attacker in the goalmouth channel is deadlier.
        centrality = 1.0 - min(1.0, abs(ay - 34.0) / 34.0)
        p = 0.6 * prox + 0.25 * centrality + 0.15 * (aerial / 100.0)
        if is_runner:
            p += 0.15 * (pace / 100.0)
        return max(0.0, min(2.0, p))

    @staticmethod
    def defender_relevance(position: str, attacker_position: str) -> float:
        """
        How appropriate is a defender of `position` to mark an attacker of
        `attacker_position` (0..~2). Anchor/big-CB types draw central targets;
        fullbacks draw wingers; a box ST is everyone's priority.
        """
        a = attacker_position
        if a in ("ST", "CF"):
            return {"CB": 2.0, "CDM": 1.6, "CM": 1.0}.get(position, 0.7)
        if a in ("LW", "RW"):
            return {"LB": 2.0 if a == "LW" else 1.4,
                    "RB": 2.0 if a == "RW" else 1.4,
                    "CM": 1.1}.get(position, 0.6)
        if a in ("CAM",):
            return {"CDM": 1.8, "CM": 1.6, "CB": 1.0}.get(position, 0.8)
        if a in ("CM", "CDM"):
            return {"CM": 1.5, "CDM": 1.7}.get(position, 0.9)
        return 0.8

    @staticmethod
    def aerial_defender_suitability(position: str, jumping: float,
                                    heading: float) -> float:
        """Who should carry the opposition's aerial threat at a corner /
        cross. CBs + the dominant jumper lead; wide roles are poor fits."""
        role = {"CB": 1.6, "CDM": 1.2, "CM": 1.0, "LB": 0.7, "RB": 0.7}.get(
            position, 0.5)
        return role * (0.6 * jumping + 0.4 * heading) / 100.0

    @staticmethod
    def assign(
        defenders: List[Tuple[str, str, float, float]],
        attackers: List[Tuple[str, str, float, float, float]],
        own_goal_x: float,
        ball_x: float,
        ball_y: float,
        danger_level: float = 50.0,
    ) -> Tuple[Dict[str, MarkAssignment], Dict[str, List[str]]]:
        """
        Greedy one-sided assignment.

        defenders: list of (name, position, x, y)
        attackers: list of (name, position, x, y, is_runner)

        Returns:
            (assignments_by_defender, attackers_left_free)
        """
        if not defenders or not attackers:
            return {}, {}

        danger_risk = max(0.0, min(1.0, danger_level / 100.0))

        # Rank attackers by cover priority. An aerial threat already inside
        # the mark radius is weighted high; a free box attacker even more so.
        ranked = []
        for (aname, apos, ax, ay, is_runner) in attackers:
            aerial = 50.0
            pace = 50.0
            # Caller may pack (is_runner, aerial, pace) into the is_runner slot.
            if isinstance(is_runner, (tuple, list)):
                raw = list(is_runner)
                if len(raw) >= 3:
                    is_runner, aerial, pace = raw[0], raw[1], raw[2]
            base = MarkingEngine.attacker_priority(
                ax, ay, own_goal_x, aerial=aerial, pace=pace, is_runner=is_runner)
            # Inside the box, the threat of an unmarked man is acute.
            if abs(ax - own_goal_x) <= BOX_MARK_RADIUS:
                base += 0.35 * danger_risk
            ranked.append((aname, apos, ax, ay, base))
        ranked.sort(key=lambda r: -r[4])

        # Defender relevance to a given attacker, blended with proximity.
        def pair_cost(def_name, def_pos, dx, dy, atk_ax, atk_ay, atk_pos):
            rel = MarkingEngine.defender_relevance(def_pos, atk_pos)
            dist = math.hypot(dx - atk_ax, dy - atk_ay)
            return rel * RELEVANCE_WEIGHT - dist * 0.15

        remaining_defs = {name: (pos, x, y) for name, pos, x, y in defenders}
        assignments: Dict[str, MarkAssignment] = {}

        for (aname, apos, ax, ay, prio) in ranked:
            if not remaining_defs:
                break
            # Pick the best remaining defender for THIS attacker.
            best = None
            best_score = float("-inf")
            for dname, (dpos, dx, dy) in remaining_defs.items():
                score = pair_cost(dname, dpos, dx, dy, ax, ay, apos)
                # Slight bias toward marking, since covering is itself an art.
                if score > best_score:
                    best_score = score
                    best = (dname, dpos, dx, dy)
            if best is None:
                continue
            dname, dpos, dx, dy = best
            del remaining_defs[dname]

            dist = math.hypot(dx - ax, dy - ay)
            # Goal-side gap: how far IS the defender goal-side of the man
            # (positive = between man and goal, best for covering).
            if abs(own_goal_x - ax) < 1e-6:
                goal_side_dist = 0.0
            else:
                dir_goal = 1.0 if own_goal_x > ax else -1.0
                goal_side_dist = (dx - ax) * dir_goal

            if dist <= TIGHT_MARK_DIST and goal_side_dist >= 0.0:
                marker_state = "tight"
            elif dist > BEATEN_DIST:
                marker_state = "beaten"
            else:
                marker_state = "free"

            tight = max(0.0, min(1.0, 1.0 - dist / TIGHT_MARK_DIST))
            assignments[dname] = MarkAssignment(
                defender_name=dname, defender_position=dpos,
                attacker_name=aname, attacker_x=ax, attacker_y=ay,
                priority=prio, tightness=tight,
                marker_state=marker_state, goal_side_dist=goal_side_dist,
            )

        # Leftover defenders cover space (no one to mark) — mark as free,
        # so the block keeps them on the ball geometry line.
        attackers_left_free: Dict[str, List[str]] = {"ranked": [r[0] for r in ranked]}
        return assignments, attackers_left_free


# ---------------------------------------------------------------------------
# Set-piece marking grid (P1 extension for _corner_chain)
# ---------------------------------------------------------------------------

@dataclass
class SetPieceMarking:
    """The defensive set-up for a corner / wide free kick."""
    aerial_defender: Optional[str] = None   # carries the opposition's top threat
    first_man: Optional[str] = None          # the man on the near post / line
    near_post: Optional[str] = None          # zonal near-post slot
    far_post: Optional[str] = None           # zonal far-post slot
    gk_defended_side: Optional[str] = None   # which post the GK starts at
    assignments: Dict[str, str] = field(default_factory=dict)  # defender -> attacker
    gk_x: float = 105.0
    gk_y: float = 34.0

    def as_dict(self) -> Dict:
        return {
            "aerial_defender": self.aerial_defender,
            "first_man": self.first_man,
            "near_post": self.near_post,
            "far_post": self.far_post,
            "gk_defended_side": self.gk_defended_side,
            "assignments": dict(self.assignments),
            "gk_x": round(self.gk_x, 1),
            "gk_y": round(self.gk_y, 1),
        }


class SetPieceMarkingEngine:
    """Builds a real corner-defence set-up from both squads' aerial make-up."""

    @staticmethod
    def build(
        defenders: List[Tuple[str, str, float, float]],
        attackers: List[Tuple[str, str, float, float]],
        own_goal_x: float,
        corner_side: str = "right",
    ) -> SetPieceMarking:
        """
        defenders: (name, position, jumping, heading)
        attackers: (name, position, jumping, heading)   [aerial threats in box]

        Roles follow modern corner defence:
            aerial_defender — the best-suited CB carries the opponent's
                              single biggest aerial threat (their "target").
            first_man       — a small/quick player stands on the near post
                              line to win the first header / block short.
            near/far_post   — zonal slots for the two-post delivery.
            GK              — starts slightly on the side the ball is swung
                              from, guarding the defended side.
        """
        if not defenders:
            return SetPieceMarking(own_goal_x=own_goal_x, gk_x=own_goal_x)

        # Rank the opponent's aerial threats to pick the man to carry.
        threats = sorted(
            attackers,
            key=lambda t: (0.6 * t[2] + 0.4 * t[3]),
            reverse=True,
        )
        target = threats[0] if threats else None

        # Aerial defender = best-suited CB / jumper for the target.
        def suit(d):
            return MarkingEngine.aerial_defender_suitability(d[1], d[2], d[3])

        eligible = [d for d in defenders if d[1] != "GK"]
        if not eligible:
            return SetPieceMarking(own_goal_x=own_goal_x, gk_x=own_goal_x)

        eligible_sort = sorted(eligible, key=suit, reverse=True)
        aerial_def = eligible_sort[0]
        # First man: pick someone NOT the aerial defender, ideally a smaller
        # / less-aerial player who can guard the near-post line.
        first_pool = [d for d in eligible if d[0] != aerial_def[0]]
        first_man = None
        if first_pool:
            # Prefer low aerial for the line — a natural "first post" pick
            # who wins the short/low first ball rather than a jump duel.
            first_man = min(
                first_pool,
                key=lambda d: abs(d[3] - 50.0),  # heading 50 is the least-aerial
            )

        assignments: Dict[str, str] = {}
        if target:
            assignments[aerial_def[0]] = target[0]
        if first_man and target:
            # first man covers the short option / near-post zonal pocket.
            assignments.setdefault(first_man[0], "near_post_zone")

        # GK: defends the side the out-swinging ball is coming from.
        gk_def = "right" if (own_goal_x == 105.0 and corner_side == "right") \
            or (own_goal_x == 0.0 and corner_side == "left") else "left"
        gk_x = own_goal_x - (2.0 if own_goal_x == 105.0 else +2.0)
        gk_y = 34.0 + (8.0 if gk_def == "left" else -8.0)

        # Zonal slots for the two posts (inherit the aerial defender's flank).
        return SetPieceMarking(
            aerial_defender=aerial_def[0],
            first_man=first_man[0] if first_man else None,
            near_post=aerial_def[0],
            far_post=first_man[0] if first_man else aerial_def[0],
            gk_defended_side=gk_def,
            assignments=assignments,
            gk_x=gk_x,
            gk_y=min(68.0, max(0.0, gk_y)),
        )
