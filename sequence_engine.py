"""
PLOFA 26/27 — SEQUENCE ENGINE
=================================
sequence_engine.py

Opta/StatsBomb-style possession-sequence analytics.

In Opta data, a *sequence* is a run of events controlled by one team that
ends in either:
    • a shot (shot-ending sequence), or
    • the loss of possession (turnover, defensive win, restart, out of play).

For every shot-ending sequence we derive the standard Opta build-up metrics:

    Passes        — number of successful (completed) passes before the shot
    Sequence Time — duration in seconds from first touch to the shot
                    (modelled as ~3s per on-ball action, the sim does not
                    track real elapsed time between atomic events)
    Progress      — net distance (m) the ball travelled toward the
                    opponent's goal line during the sequence
    Direct Speed  — Progress ÷ Sequence Time (m/s)
    Width         — horizontal span (m) between the leftmost and rightmost
                    points the ball reached in the sequence

And the three macro attacking styles Opta synthesises from sequences:

    Build-Up Attacks   — open-play sequences with ≥10 passes that end in a
                         shot or a touch inside the box  (possession teams)
    Direct Attacks     — open-play sequences that start in a team's own
                         half, have <10 passes, ≥50% forward movement, and
                         end in a shot  (counter-attacking teams)
    Shot-Ending High Turnovers — a defensive win within 40m of the
                         opponent's goal that leads directly to a shot
                         (high-press efficiency)

Call SequenceTracker.build(home_team, away_team, timeline) once per match.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from match_engine import EventType, MatchEvent, SituationType


# ─────────────────────────────────────────────
# POSSESSION BOUNDARY CLASSIFICATION
# ─────────────────────────────────────────────

# Events that START a new possession for the team named in the event.
POSSESSION_GAIN_EVENTS = {
    EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.CLEARANCE,
    EventType.BLOCK, EventType.RECOVERY, EventType.BALL_RECOVERY,
    EventType.SAVE, EventType.GOAL_KICK, EventType.THROW_IN,
    EventType.CORNER_TAKEN, EventType.FREEKICK_DIRECT, EventType.FREEKICK_CROSS,
    EventType.KICKOFF, EventType.PRESS_SUCCESS, EventType.PENALTY_WON,
    EventType.FOUL_WON,
}

# Events that END a possession (the named team no longer has the ball).
POSSESSION_LOSS_EVENTS = {
    EventType.TURNOVER, EventType.MISCONTROL, EventType.DISPOSSESSED,
    EventType.OFFSIDE,
}

# Events that count as one "on-ball action" for sequence-time modelling.
ON_BALL_ACTIONS = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS, EventType.THROUGH_BALL,
    EventType.CARRY, EventType.DRIBBLE_ATTEMPT, EventType.DRIBBLE_SUCCESS,
    EventType.DRIBBLE_FAIL, EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
    EventType.SHOT_BLOCKED, EventType.GOAL, EventType.HIT_WOODWORK,
    EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
}

# Completed-pass event types (successful passes count toward "Passes").
COMPLETED_PASS_TYPES = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.CROSS_SUCCESS, EventType.THROUGH_BALL,
}

# Defensive wins that constitute a "high turnover" when inside the
# opponent's final 40m.
HIGH_TURNOVER_WINS = {
    EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.CLEARANCE,
    EventType.BLOCK, EventType.RECOVERY, EventType.BALL_RECOVERY,
    EventType.PRESS_SUCCESS,
}

# Seconds of real time per on-ball action (sim models atomically, so we
# derive duration from event count — a 10-pass sequence ≈ 30s, matching Opta).
SECONDS_PER_ACTION = 3.0

# Penalty-area x boundary (16.5m from goal line) used for "touch in box".
BOX_EDGE = 88.0


# ─────────────────────────────────────────────
# SEQUENCE OBJECT
# ─────────────────────────────────────────────

@dataclass
class Sequence:
    """One possession segment from the timeline."""
    team: str
    opponent: str
    events: List[MatchEvent] = field(default_factory=list)
    started_by: Optional[EventType] = None   # how this possession began
    start_event: Optional[MatchEvent] = None

    # ── Shot-ending metrics ─────────────────────────────────
    ends_in_shot: bool = False
    shot: Optional[MatchEvent] = None
    passes: int = 0                  # successful passes before the shot
    actions: int = 0                 # total on-ball actions
    sequence_time: float = 0.0       # seconds
    progress: float = 0.0            # m toward opponent goal line
    direct_speed: float = 0.0        # m/s
    width: float = 0.0               # m horizontal span
    starts_own_half: bool = False
    touch_in_box: bool = False
    forward_events: int = 0          # on-ball events that moved forward
    forward_movement_ratio: float = 0.0
    high_turnover: bool = False      # won ball inside opp final 40m

    def __len__(self) -> int:
        return len(self.events)

    def summary(self) -> Dict:
        return {
            "Team": self.team,
            "Opponent": self.opponent,
            "Ends In Shot": self.ends_in_shot,
            "Shooter": self.shot.player if self.shot else "",
            "Shot Outcome": self.shot.event_type.name if self.shot else "",
            "Situation": self.shot.situation.value if self.shot and self.shot.situation else "",
            "Minute": self.shot.minute if self.shot else "",
            "Passes": self.passes,
            "Actions": self.actions,
            "Sequence Time (s)": round(self.sequence_time, 1),
            "Progress (m)": round(self.progress, 1),
            "Direct Speed (m/s)": round(self.direct_speed, 2),
            "Width (m)": round(self.width, 1),
            "Starts Own Half": self.starts_own_half,
            "Touch In Box": self.touch_in_box,
            "Forward Movement %": round(self.forward_movement_ratio * 100, 1),
            "High Turnover": self.high_turnover,
        }


# ─────────────────────────────────────────────
# SEQUENCE TRACKER
# ─────────────────────────────────────────────

class SequenceTracker:
    """Segments a match timeline into possession sequences and derives
    Opta-style shot-ending build-up metrics plus macro attacking styles."""

    def __init__(self, home_team: str, away_team: str, timeline: List[MatchEvent]):
        self.home_team = home_team
        self.away_team = away_team
        self.timeline = timeline
        self.sequences: List[Sequence] = []
        self._segment()

    # ── SEGMENTATION ─────────────────────────────────────

    def _segment(self):
        current: Optional[Sequence] = None
        for e in self.timeline:
            # A defensive win / restart hands possession to e.team's side.
            if e.event_type in POSSESSION_GAIN_EVENTS:
                current = self._start_sequence(e)
            # A shot ends the current sequence (shot belongs to it).
            elif e.is_shot:
                if current is None:
                    current = self._start_sequence(e)
                self._append(current, e)
                current.ends_in_shot = True
                current.shot = e
                self.sequences.append(current)
                current = None
            # A turnover / dispossession ends the sequence without a shot.
            elif e.event_type in POSSESSION_LOSS_EVENTS:
                if current is not None:
                    self._append(current, e)
                    self.sequences.append(current)
                    current = None
            # Ball carries on in the same possession.
            else:
                if current is None:
                    current = self._start_sequence(e)
                else:
                    self._append(current, e)

        if current is not None:
            self.sequences.append(current)

    def _start_sequence(self, e: MatchEvent) -> Sequence:
        opponent = self.away_team if e.team == self.home_team else self.home_team
        seq = Sequence(team=e.team, opponent=opponent,
                       started_by=e.event_type, start_event=e)
        self._append(seq, e)
        return seq

    def _append(self, seq: Sequence, e: MatchEvent):
        seq.events.append(e)
        if e.event_type in COMPLETED_PASS_TYPES and e.outcome:
            seq.passes += 1
        if e.event_type in ON_BALL_ACTIONS:
            seq.actions += 1

    # ── TEAM ATTACK DIRECTION ────────────────────────────

    def _attacks_right(self, team: str) -> bool:
        return team == self.home_team   # home attacks right (x→105)

    def _goal_line_x(self, team: str) -> float:
        return 105.0 if self._attacks_right(team) else 0.0

    def _fwd_advance(self, e: MatchEvent, team: str) -> float:
        """Net metres toward the opponent goal between two points."""
        sx = e.location_x if e.location_x is not None else 0.0
        ex = e.end_x if e.end_x is not None else sx
        return (ex - sx) if self._attacks_right(team) else (sx - ex)

    # ── FINALISE METRICS (post-segmentation) ─────────────

    def compute_metrics(self):
        for seq in self.sequences:
            seq.sequence_time = seq.actions * SECONDS_PER_ACTION

            xs: List[float] = []
            ys: List[float] = []
            forward = 0
            back_or_side = 0
            gx = self._goal_line_x(seq.team)

            for e in seq.events:
                if e.location_x is not None:
                    xs.append(e.location_x)
                    ys.append(e.location_y if e.location_y is not None else 34.0)
                if e.end_x is not None:
                    xs.append(e.end_x)
                    ys.append(e.end_y if e.end_y is not None else 34.0)

                if e.event_type in ON_BALL_ACTIONS and e.location_x is not None:
                    adv = self._fwd_advance(e, seq.team)
                    if adv > 0.5:
                        forward += 1
                    elif adv < -0.5:
                        back_or_side += 1

            if xs:
                seq.progress = max(0.0, gx - min(xs) if self._attacks_right(seq.team)
                                   else max(xs) - gx)
                seq.width = max(ys) - min(ys) if len(ys) > 1 else 0.0

                # Touch inside the opponent's box.
                box_min = BOX_EDGE if self._attacks_right(seq.team) else 0.0
                box_max = 105.0 if self._attacks_right(seq.team) else 105.0 - BOX_EDGE
                seq.touch_in_box = any(
                    (box_min <= x <= box_max) for x in xs
                )

            # Sequence starts in the team's own half.
            if seq.events:
                first_x = seq.events[0].location_x
                if first_x is not None:
                    own_half = first_x < 52.5 if self._attacks_right(seq.team) else first_x > 52.5
                    seq.starts_own_half = own_half

            total_movement = forward + back_or_side
            seq.forward_movement_ratio = (
                forward / total_movement if total_movement > 0 else 0.0
            )

            # High turnover: possession won inside opponent's final 40m.
            if seq.started_by in HIGH_TURNOVER_WINS and seq.start_event:
                sx = seq.start_event.location_x
                if sx is not None:
                    dist_from_opp_goal = abs(self._goal_line_x(seq.team) - sx)
                    seq.high_turnover = dist_from_opp_goal <= 40.0

            if seq.ends_in_shot and seq.sequence_time > 0:
                seq.direct_speed = seq.progress / seq.sequence_time

    # ── PUBLIC API ──────────────────────────────────────

    def shot_ending_sequences(self) -> List[Sequence]:
        return [s for s in self.sequences if s.ends_in_shot]

    def team_sequences(self, team: str) -> List[Sequence]:
        return [s for s in self.sequences if s.team == team]

    # ── MACRO STYLE METRICS ─────────────────────────────

    def build_up_attacks(self, team: str) -> int:
        """Open-play sequences with ≥10 passes ending in a shot or box touch."""
        count = 0
        for s in self.team_sequences(team):
            is_open = not (s.shot and s.shot.situation and s.shot.situation
                           in (SituationType.CORNER, SituationType.DIRECT_FREEKICK,
                               SituationType.CROSSED_FREEKICK, SituationType.PENALTY))
            if is_open and s.passes >= 10 and (s.ends_in_shot or s.touch_in_box):
                count += 1
        return count

    def direct_attacks(self, team: str) -> int:
        """Own-half start, <10 passes, ≥50% forward movement, ends in a shot."""
        count = 0
        for s in self.team_sequences(team):
            if not s.ends_in_shot:
                continue
            is_open = not (s.shot and s.shot.situation and s.shot.situation
                           in (SituationType.CORNER, SituationType.DIRECT_FREEKICK,
                               SituationType.CROSSED_FREEKICK, SituationType.PENALTY))
            if (is_open and s.starts_own_half and s.passes < 10
                    and s.forward_movement_ratio >= 0.5):
                count += 1
        return count

    def shot_ending_high_turnovers(self, team: str) -> int:
        """Defensive win within 40m of opp goal that ends in a shot."""
        return sum(1 for s in self.team_sequences(team)
                   if s.high_turnover and s.ends_in_shot)

    # ── EXPORT HELPERS ──────────────────────────────────

    def shot_ending_rows(self, team: str = None) -> List[Dict]:
        seqs = self.shot_ending_sequences()
        if team:
            seqs = [s for s in seqs if s.team == team]
        return [s.summary() for s in seqs]

    def macro_rows(self) -> List[Dict]:
        rows = []
        for team in (self.home_team, self.away_team):
            rows.append({
                "Team": team,
                "Build-Up Attacks": self.build_up_attacks(team),
                "Direct Attacks": self.direct_attacks(team),
                "Shot-Ending High Turnovers": self.shot_ending_high_turnovers(team),
                "Shot-Ending Sequences": len([s for s in self.team_sequences(team) if s.ends_in_shot]),
                "Total Sequences": len(self.team_sequences(team)),
            })
        return rows
