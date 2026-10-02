"""
physics/world.py — the orchestrator.
=====================================

§17 sketches the integration as a pipeline::

    MatchEngine
      -> Player Brain decides action
      -> Physics World resolves physical execution
      -> Continuous simulation clock advances
      -> MatchEvent generated
      -> PositionEngine updated
      -> Stamina updated
      -> Next decision

This module is the middle of that. It owns a clock, a ball model and a
violation log, and it answers one question per call: given what the brain wants
and where everyone currently is, what physically happens and when?

The one architectural decision worth stating loudly
--------------------------------------------------
**This world holds no player positions.**

§16 is emphatic: "There must be ONE authoritative current position." So
positions are *passed in* from ``PositionEngine`` on every call and are never
cached here. A physics world that remembered where players were would be a
second source of truth, and the two would drift the first time a substitution
or a reset happened.

The consequence is that this module is cheap to call and impossible to
desynchronise: it holds no match state that could go stale, only the clock —
which *is* meant to be stateful, because time is the one thing a simulation has
to remember.

Failure mode
------------
A pass whose physics cannot be resolved — nobody can reach the ball, the
trajectory is impossible — returns a :class:`PassResolution` with
``resolved=False`` and a reason. It does not raise, because a football match
must be able to absorb an impossible pass and carry on. Callers that need
strictness (tests, calibration) check ``resolved`` or ask the violation log to
raise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from physics.ball_physics import BallPhysics, Trajectory, profile_for
from physics.calibration import Calibration, PhysicalProfile, calibrate_from_values
from physics.collision import (
    Candidate,
    ContestResult,
    arrival_contest,
    resolve_receiver,
)
from physics.continuous_time import ContinuousClock, seconds_to_match_time
from physics.player_motion import KinematicState, PlayerMotion, arrival_time
from physics.violations import PhysicsViolation, ViolationLog

#: Milestone labels the timeline uses. Strings rather than an enum because they
#: are written straight into reports and tests assert on them.
MILESTONE_PASS_START = "PASS START"
MILESTONE_BALL_ARRIVES = "BALL REACHES TARGET"
MILESTONE_REACTION = "REACTION COMPLETE"
MILESTONE_ARRIVAL = "ARRIVAL"
MILESTONE_CONTROL = "CONTROLS BALL"
MILESTONE_UNCLAIMED = "BALL UNCLAIMED"
MILESTONE_INTERCEPTED = "INTERCEPTED"


@dataclass(frozen=True)
class TimelineStep:
    """One entry in a physical timeline."""

    time: float
    label: str
    subject: str = ""
    detail: Dict[str, object] = field(default_factory=dict)

    @property
    def match_time(self) -> str:
        return str(seconds_to_match_time(self.time))

    def __str__(self) -> str:
        who = f"  {self.subject}" if self.subject else ""
        return f"{self.match_time}  {self.label}{who}"


@dataclass
class PassResolution:
    """Everything that happened when a pass was struck."""

    trajectory: Trajectory
    contest: ContestResult
    timeline: List[TimelineStep]
    receiver: str
    resolved: bool
    reason: str = ""
    kind: str = ""

    @property
    def launch_time(self) -> float:
        return self.trajectory.launch_time

    @property
    def ball_arrival(self) -> float:
        return self.trajectory.arrival_time

    @property
    def control_time(self) -> float:
        """When someone actually gets the ball — the last meaningful step."""
        steps = [s for s in self.timeline
                 if s.label in (MILESTONE_CONTROL, MILESTONE_UNCLAIMED,
                                MILESTONE_INTERCEPTED)]
        return steps[-1].time if steps else self.trajectory.arrival_time

    @property
    def intercepted(self) -> bool:
        return bool(self.receiver) and self.receiver not in self._intended

    _intended: str = ""

    def timeline_text(self) -> str:
        return "\n".join(f"  {s}" for s in self.timeline)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "resolved": self.resolved,
            "receiver": self.receiver,
            "intercepted": self.intercepted,
            "reason": self.reason,
            "trajectory": self.trajectory.to_dict(),
            "contest": self.contest.to_dict(),
            "control_time": round(self.control_time, 4),
            "timeline": [{"time": round(s.time, 4), "label": s.label,
                          "subject": s.subject} for s in self.timeline],
        }


class PhysicsWorld:
    """Continuous physical-time resolution for one match.

    Construct one per match, hand it calibrated players and positions, and ask
    it what happens. It is opt-in by construction: the engine does not create
    one unless physics is enabled, so the default path is unchanged.
    """

    def __init__(self, clock: Optional[ContinuousClock] = None,
                 ball: Optional[BallPhysics] = None,
                 calibration: Optional[Calibration] = None,
                 speed_ceiling: float = 12.0) -> None:
        self.clock = clock or ContinuousClock()
        self.ball = ball or BallPhysics()
        self.calibration = calibration or Calibration()
        self.violations = ViolationLog()
        self.speed_ceiling = speed_ceiling
        #: every resolution this world has produced, for reporting
        self.history: List[PassResolution] = []

    # -- players --------------------------------------------------------
    def profile(self, *, name: str = "", position: str = "",
                pace: float = 60.0, acceleration: float = 60.0,
                anticipation: float = 60.0, agility: float = 60.0,
                stamina: float = 100.0) -> PhysicalProfile:
        """Calibrate a player. Scores in, SI units out."""
        return calibrate_from_values(
            name=name, position=position, pace_score=pace,
            acceleration_score=acceleration, anticipation_score=anticipation,
            agility=agility, stamina=stamina, calibration=self.calibration)

    # -- the main event -------------------------------------------------
    def resolve_pass(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        candidates: Sequence[Candidate] = (),
        kind: str = "short",
        *,
        speed: Optional[float] = None,
        launch_height: float = 0.0,
        launch_time: Optional[float] = None,
        intended: str = "",
        advance_clock: bool = True,
    ) -> PassResolution:
        """Strike a ball and resolve who reaches it.

        ``start`` and ``end`` are pitch coordinates in metres. ``candidates``
        are players already reduced to a name and an arrival time — build them
        with :meth:`candidate_for`, which is where the kinematics live.

        The clock is advanced to the moment the ball is actually resolved, not
        to when it was struck, because that is when the world has changed.
        """
        t0 = self.clock.play_seconds if launch_time is None else float(launch_time)

        traj = self.ball.plan(start, end, kind=kind, launch_time=t0,
                             speed=speed, launch_height=launch_height)

        if traj.travel_time == math.inf:
            reason = ("the ball comes to rest before travelling that far; "
                      "the delivery is not physically possible")
            self.violations.record(_trajectory_violation(traj, start, end, t0))
            res = PassResolution(
                trajectory=traj,
                contest=arrival_contest(traj.arrival_time, ()),
                timeline=[TimelineStep(t0, MILESTONE_PASS_START,
                                       detail={"kind": kind})],
                receiver="", resolved=False, reason=reason, kind=kind,
                _intended=intended)
            self.history.append(res)
            return res

        # The contest compares RELATIVE times: "seconds since the ball was
        # struck". Passing traj.arrival_time here (an absolute play-clock value)
        # against candidates whose arrival is relative produced margins like
        # "arrives 2240.427s early", because an absolute time was being
        # subtracted from a duration.
        contest = arrival_contest(traj.travel_time, candidates)
        receiver = resolve_receiver(contest, intended)

        timeline = [TimelineStep(t0, MILESTONE_PASS_START,
                                 subject=_subject_of(candidates, intended),
                                 detail={"kind": kind,
                                         "distance": round(traj.distance, 2),
                                         "ball_speed": round(traj.launch_speed, 2)})]
        timeline.append(TimelineStep(
            traj.arrival_time, MILESTONE_BALL_ARRIVES,
            detail={"travel_time": round(traj.travel_time, 4),
                    "apex": round(traj.apex, 2),
                    "aerial": traj.is_aerial}))

        # Reaction completion is the moment the player starts moving toward the
        # ball, which is BEFORE their arrival. An earlier version emitted this
        # milestone at the full arrival time and labelled it "reaction 1.479s",
        # which reported the whole journey as if it were the reaction delay.
        lead = contest.leader
        if lead is not None and lead.reaction > 0.0:
            timeline.append(TimelineStep(
                t0 + lead.reaction, MILESTONE_REACTION, subject=lead.name,
                detail={"delay": round(lead.reaction, 4),
                        "travel": round(lead.travel, 4)}))

        for cand in contest.ranked:
            timeline.append(TimelineStep(
                t0 + cand.arrival, MILESTONE_ARRIVAL, subject=cand.name,
                detail={"team": cand.team, "distance": round(cand.distance, 2),
                        "early_by": round(cand.arrival - traj.travel_time, 4)}))

        if contest.unclaimed:
            timeline.append(TimelineStep(
                traj.arrival_time, MILESTONE_UNCLAIMED,
                detail={"reason": contest.reason}))
        else:
            label = (MILESTONE_INTERCEPTED
                     if receiver and receiver != intended and intended
                     else MILESTONE_CONTROL)
            # Control happens when the ball gets there AND the player is there
            # to meet it — whichever is later.
            #
            # This was a real bug caught by a test. A receiver whose arrival
            # time beat the ball's was emitting CONTROLS BALL *before* the ball
            # had arrived, which is not a football situation. In that case the
            # attacker gets there early, positions himself, and the ball
            # arrives to him — so control is the ball's arrival, and the
            # timeline reads forwards.
            #
            # The brief's illustrative sequence agrees: ATTACKER ARRIVAL and
            # CONTROLS BALL share a timestamp, because the two coincide at the
            # moment of contact.
            leader = contest.leader
            control_offset = traj.travel_time
            if leader is not None:
                control_offset = max(traj.travel_time, leader.arrival)
            timeline.append(TimelineStep(
                t0 + control_offset, label, subject=receiver,
                detail={"tie": contest.tied,
                        "margin": round(contest.margin, 4),
                        "ball_travel": round(traj.travel_time, 4),
                        "player_arrival": round(leader.arrival, 4)
                        if leader else None}))

        timeline.sort(key=lambda s: (s.time, s.label))

        if advance_clock and timeline:
            self.clock.advance_to(timeline[-1].time)

        res = PassResolution(
            trajectory=traj, contest=contest, timeline=timeline,
            receiver=receiver, resolved=not contest.unclaimed,
            reason=contest.reason, kind=kind, _intended=intended)
        self.history.append(res)
        return res

    # -- helpers --------------------------------------------------------
    def candidate_for(self, name: str, position: Tuple[float, float],
                      ball_target: Tuple[float, float],
                      state: KinematicState, profile: PhysicalProfile,
                      team: str = "", intended: bool = False,
                      reach: Optional[float] = None) -> Tuple[Candidate, float]:
        """Build a :class:`Candidate` and report the distance covered.

        Returns ``(candidate, distance)`` so a caller can log a race without
        recomputing geometry. The candidate carries the reaction/movement split
        explicitly, because the profile is the only thing that knows where that
        boundary is.
        """
        kw = {} if reach is None else {"reach": reach}
        t = arrival_time(state, ball_target, profile, **kw)
        dist = math.hypot(ball_target[0] - state.x, ball_target[1] - state.y)
        reaction = min(profile.reaction_time, t)
        return Candidate(
            name=name, arrival=t, team=team, position=profile.position,
            distance=dist, intended=intended, reaction=reaction), dist

    def run_player_to(self, player: PlayerMotion, target: Tuple[float, float],
                      reaction: Optional[float] = None) -> float:
        """Advance a player to a target, recording any violation. Returns seconds."""
        t = player.arrival(target, reaction=reaction)
        final = player.state.advanced(t)
        self.violations.check((player.state.x, player.state.y),
                              (final.x, final.y), t, subject=player.name)
        return t

    # -- reporting ------------------------------------------------------
    def stoppage(self, event: str) -> float:
        """Charge stoppage for a dead-ball event (play clock is untouched)."""
        from physics.continuous_time import stoppage_for
        return self.clock.add_stoppage(stoppage_for(event))

    def report(self) -> str:
        lines = [
            "PHYSICS WORLD",
            f"  clock        : {self.clock}",
            f"  ball         : {self.ball.describe()}",
            f"  calibration  : {self.calibration.describe()}",
            f"  resolutions  : {len(self.history)}",
            f"  violations   : {self.violations.summary()}",
        ]
        return "\n".join(lines)

    def timeline_text(self) -> str:
        if not self.history:
            return "no resolved actions"
        out = []
        for res in self.history:
            out.append(f"{res.kind.upper()} -> {res.receiver or '(nobody)'}")
            out.append(res.timeline_text())
        return "\n".join(out)

    def assert_clean(self) -> None:
        """Raise if anything impossible happened. For tests and calibration."""
        self.violations.raise_if_any()


# ─────────────────────────────────────────────
# module helpers
# ─────────────────────────────────────────────

def _subject_of(candidates: Sequence[Candidate], intended: str) -> str:
    if intended:
        return intended
    return candidates[0].name if candidates else ""


def _trajectory_violation(traj: Trajectory, start, end, t0) -> PhysicsViolation:
    from physics.violations import PhysicsViolation
    return PhysicsViolation(
        kind="impossible_trajectory",
        detail=(f"ball cannot cover {traj.distance:.1f} m at "
                f"{traj.launch_speed:.1f} m/s"),
        subject="ball", start=tuple(start), end=tuple(end), dt=traj.travel_time)
