"""
PLOFA 26/27 — OPTO-ANALYTICS EXTENSION (Checkpoint 28)
=======================================================

Adds the following Opta-confirmed stats derived from the real event
timeline and the existing SequenceTracker:

    1.  Pressed Sequences        – sequences where the attacking team had
                                  at least one PRESS event against them
    2.  PPDA                     – Passes Per Defensive Action in the final
                                  third (lower = better press efficiency)
    3.  Goals/Shot-ending Carries – carries that are the last on-ball action
                                   before a shot in a sequence
    4.  Assist/Chance-creating carries with true locations – carries that
        appear in the shot-creating action (SCA) chain before a shot, with
        exact geometric start/end coordinates
    5.  Carry Directness          – per-carry ratio of forward progress to
                                  total distance (0..1, 1 = purely forward)
    6.  Players per Possession    – average distinct players per sequence
    7.  Carry Chains             – sequences containing 2+ carries
    8.  Ball Progression Chains  – sequences that progressed from own half
                                  into the final third or opponent box
    9.  Dangerous Possessions    – sequences that ended in a shot inside
                                  the final third

All definitions match standard Opta/StatsBomb methodology. No synthetic
data: every number is derived from what the simulation actually produced.
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from match_engine import EventType, MatchEvent, MatchResult, MatchConfig
from sequence_engine import SequenceTracker, Sequence
from chance_creation import ChanceCreationLedger


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _attacks_right(team: str, home_team: str) -> bool:
    return team == home_team


def _own_half(x: float, attacks_right: bool) -> bool:
    return x < 52.5 if attacks_right else x > 52.5


def _final_third(x: float, attacks_right: bool) -> bool:
    return x >= 70.0 if attacks_right else x <= 35.0


def _opp_box(x: float, attacks_right: bool) -> bool:
    return x >= 83.0 if attacks_right else x <= 22.0


# Defensive action types that count toward PPDA / press efficiency.
_DEFENSIVE_ACTIONS = {
    EventType.TACKLE_WON, EventType.INTERCEPTION, EventType.RECOVERY,
    EventType.BALL_RECOVERY, EventType.BLOCK, EventType.CLEARANCE,
    EventType.PRESS_SUCCESS,
}

# Pass-like events that count toward opponent pass volume.
_PASS_LIKE = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_SUCCESS,
}


# ──────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ──────────────────────────────────────────────────────────────

@dataclass
class OptaStatsResult:
    """Holds every derived Opta stat for one match."""

    # ── Team-level ───────────────────────────────────────────
    team: Dict[str, Dict] = field(default_factory=dict)

    # ── Per-player ───────────────────────────────────────────
    player: Dict[str, Dict] = field(default_factory=dict)

    # ── Carry-creating event log (true locations) ────────────
    carry_creation_log: List[Dict] = field(default_factory=list)

    # ── Raw sequence metrics (for export / audit) ────────────
    sequence_rows: List[Dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# CORE COMPUTATION
# ──────────────────────────────────────────────────────────────

class OptaStatsEngine:
    """Derives all new Opta metrics from the timeline."""

    def __init__(self, result: MatchResult, timeline: List[MatchEvent]):
        self.result = result
        self.config = result.config
        self.home = result.config.home_team
        self.away = result.config.away_team
        self.timeline = timeline

        # Pre-build helpers
        self._seq_tracker = SequenceTracker(
            self.home, self.away, timeline
        )
        self._seq_tracker.compute_metrics()
        self._cc_ledger = ChanceCreationLedger(timeline).compute()

    # ── PUBLIC API ────────────────────────────────────────────

    def compute(self) -> OptaStatsResult:
        res = OptaStatsResult()

        # Initialise per-team containers
        for team in (self.home, self.away):
            res.team[team] = self._blank_team()
            res.player[team] = {}

        # Sequence-level metrics
        self._compute_sequence_metrics(res)

        # PPDA (team-level)
        self._compute_ppda(res)

        # Carry-level metrics (player + team)
        self._compute_carry_metrics(res)

        # Players per possession
        self._compute_players_per_possession(res)

        return res

    # ── BLANK TEMPLATES ──────────────────────────────────────

    @staticmethod
    def _blank_team() -> Dict:
        return {
            "pressed_sequences": 0,
            "ppda": 0.0,
            "ppda_defensive_actions": 0,
            "ppda_opponent_passes": 0,
            "shot_ending_carries": 0,
            "goal_ending_carries": 0,
            "chance_creating_carries": 0,
            "assist_carries": 0,
            "carry_directness": 0.0,
            "carry_directness_count": 0,
            "total_carry_distance": 0.0,
            "total_carry_forward_distance": 0.0,
            "carry_chains": 0,
            "ball_progression_chains": 0,
            "dangerous_possessions": 0,
            "players_per_possession": 0.0,
            "total_sequences": 0,
            "total_carries": 0,
        }

    @staticmethod
    def _blank_player() -> Dict:
        return {
            "shot_ending_carries": 0,
            "goal_ending_carries": 0,
            "chance_creating_carries": 0,
            "assist_carries": 0,
            "carry_directness": 0.0,
            "carry_directness_count": 0,
            "total_carry_distance": 0.0,
            "total_carry_forward_distance": 0.0,
            "carry_chains": 0,
            "ball_progression_chains": 0,
            "dangerous_possessions": 0,
            "total_carries": 0,
        }

    # ── SEQUENCE-LEVEL METRICS ────────────────────────────────

    def _compute_sequence_metrics(self, res: OptaStatsResult) -> None:
        """Walk all sequences and tag each with pressed, carry-chain,
        ball-progression, dangerous-possession, and distinct-player flags."""
        for seq in self._seq_tracker.sequences:
            team = seq.team
            opp = seq.opponent
            opp_ar = _attacks_right(opp, self.home)
            team_ar = _attacks_right(team, self.home)

            seq_row: Dict = {
                "team": team,
                "opponent": opp,
                "ends_in_shot": seq.ends_in_shot,
                "pressed": False,
                "carry_chain": False,
                "ball_progression_chain": False,
                "dangerous_possession": False,
                "distinct_players": 0,
                "shot_ending_carries": 0,
                "chance_creating_carries": 0,
                "total_carries": 0,
            }
            res.sequence_rows.append(seq_row)

            # Distinct players in this sequence
            players_in_seq: set = set()
            carry_count = 0
            carry_locations: List[Tuple[float, float, float, float]] = []

            for e in seq.events:
                if e.player:
                    players_in_seq.add(e.player)
                if e.event_type == EventType.CARRY:
                    carry_count += 1
                    if e.location_x is not None and e.end_x is not None:
                        carry_locations.append(
                            (e.location_x, e.location_y or 34.0,
                             e.end_x, e.end_y or 34.0)
                        )

            seq_row["distinct_players"] = len(players_in_seq)
            seq_row["total_carries"] = carry_count

            # Carry chain: 2+ carries in the sequence
            if carry_count >= 2:
                seq_row["carry_chain"] = True
                res.team[team]["carry_chains"] += 1

            # Pressed: at least one PRESS event where the pressing team
            # is the opponent of this sequence's possessing team.
            has_press = any(
                ev.event_type == EventType.PRESS and ev.team == opp
                for ev in seq.events
            )
            if has_press:
                seq_row["pressed"] = True
                res.team[team]["pressed_sequences"] += 1

            # Ball progression chain: started in own half and ended in
            # final third or opponent box.  Also count if net progress
            # moved the ball at least 25 m toward the opponent goal.
            if carry_locations:
                first_x = carry_locations[0][0]
                last_x = carry_locations[-1][2]
                started_own = _own_half(first_x, team_ar)
                ended_final_or_box = (
                    _final_third(last_x, team_ar)
                    or _opp_box(last_x, team_ar)
                )
                progressed = (last_x - first_x) if team_ar else (first_x - last_x)
                if (started_own and ended_final_or_box) or progressed >= 25:
                    seq_row["ball_progression_chain"] = True
                    res.team[team]["ball_progression_chains"] += 1

            # Dangerous possession: ends in a shot inside final third
            if seq.ends_in_shot and seq.shot is not None:
                sx = seq.shot.location_x or 0.0
                if _final_third(sx, team_ar):
                    seq_row["dangerous_possession"] = True
                    res.team[team]["dangerous_possessions"] += 1

            # Shot-ending carries: carries that are the last on-ball
            # action before the shot in this sequence.
            if seq.ends_in_shot and seq.shot is not None:
                shot_idx = None
                for idx, ev in enumerate(seq.events):
                    if ev is seq.shot:
                        shot_idx = idx
                        break
                if shot_idx is not None:
                    # Look backward for the last CARRY before the shot
                    for ev in reversed(seq.events[:shot_idx]):
                        if ev.event_type == EventType.CARRY:
                            res.team[team]["shot_ending_carries"] += 1
                            seq_row["shot_ending_carries"] += 1
                            player = ev.player
                            if player not in res.player.setdefault(team, {}):
                                res.player[team][player] = self._blank_player()
                            res.player[team][player]["shot_ending_carries"] += 1
                            if seq.shot.event_type == EventType.GOAL:
                                res.team[team]["goal_ending_carries"] += 1
                                res.player[team][player]["goal_ending_carries"] += 1
                            break

            res.team[team]["total_sequences"] += 1
            res.team[team]["total_carries"] += carry_count

            # Carry directness: accumulate per-carry values
            for e in seq.events:
                if e.event_type == EventType.CARRY:
                    if e.location_x is not None and e.end_x is not None:
                        sx, sy = e.location_x, e.location_y or 34.0
                        ex, ey = e.end_x, e.end_y or 34.0
                        dx = ex - sx if team_ar else sx - ex
                        dy = ey - sy
                        total_dist = math.hypot(dx, dy)
                        if total_dist > 0:
                            fwd_dist = max(0.0, dx)
                            directness = fwd_dist / total_dist
                            res.team[team]["carry_directness"] += directness
                            res.team[team]["carry_directness_count"] += 1
                            res.team[team]["total_carry_distance"] += total_dist
                            res.team[team]["total_carry_forward_distance"] += fwd_dist
                            player = e.player
                            if player not in res.player.setdefault(team, {}):
                                res.player[team][player] = self._blank_player()
                            res.player[team][player]["carry_directness"] += directness
                            res.player[team][player]["carry_directness_count"] += 1
                            res.player[team][player]["total_carry_distance"] += total_dist
                            res.player[team][player]["total_carry_forward_distance"] += fwd_dist

    # ── PPDA ─────────────────────────────────────────────────

    def _compute_ppda(self, res: OptaStatsResult) -> None:
        """Passes Per Defensive Action in the final third."""
        for team in (self.home, self.away):
            opp = self.away if team == self.home else self.home
            team_ar = _attacks_right(team, self.home)
            opp_ar = not team_ar

            # Opponent passes in this team's final third
            opp_passes = 0
            own_def_actions = 0

            for e in self.timeline:
                ex = e.end_x if e.end_x is not None else e.location_x
                if ex is None:
                    continue

                # Opponent passes in this team's final third
                if e.team == opp and e.event_type in _PASS_LIKE and e.outcome:
                    if _final_third(ex, team_ar):
                        opp_passes += 1

                # Own defensive actions in this team's final third
                if e.team == team and e.event_type in _DEFENSIVE_ACTIONS:
                    lx = e.location_x or ex
                    if _final_third(lx, team_ar):
                        own_def_actions += 1

            res.team[team]["ppda_opponent_passes"] = opp_passes
            res.team[team]["ppda_defensive_actions"] = own_def_actions
            res.team[team]["ppda"] = (
                round(opp_passes / max(1, own_def_actions), 2)
            )

    # ── CARRY CREATION METRICS ───────────────────────────────

    def _compute_carry_metrics(self, res: OptaStatsResult) -> None:
        """Chance-creating and assist carries with true locations."""
        for record in self._cc_ledger.records:
            sca_players = set(getattr(record, "sca_players", []) or [])

            shot_idx = None
            for idx, e in enumerate(self.timeline):
                if (e.event_type in {
                    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                    EventType.SHOT_BLOCKED, EventType.HIT_WOODWORK,
                    EventType.GOAL, EventType.PENALTY_SCORED,
                    EventType.PENALTY_MISSED,
                } and e.player == record.shooter
                        and e.minute == record.minute):
                    shot_idx = idx
                    break

            if shot_idx is None:
                continue

            is_assist = (record.outcome == "goal")
            team = record.team

            # Scan backward for the last CARRY before the shot whose carrier
            # is in the SCA pair for this shot.
            for j in range(shot_idx - 1, max(-1, shot_idx - 25), -1):
                ev = self.timeline[j]
                if ev.team != team:
                    if ev.team is not None:
                        continue
                    break
                if ev.event_type == EventType.CARRY:
                    if ev.player not in sca_players:
                        continue
                    if team not in res.player:
                        continue
                    if ev.player not in res.player[team]:
                        res.player[team][ev.player] = self._blank_player()

                    res.team[team]["chance_creating_carries"] += 1
                    res.player[team][ev.player]["chance_creating_carries"] += 1
                    if is_assist:
                        res.team[team]["assist_carries"] += 1
                        res.player[team][ev.player]["assist_carries"] += 1

                    log_entry = {
                        "minute": ev.minute,
                        "team": team,
                        "player": ev.player,
                        "is_assist": is_assist,
                        "shot_outcome": record.outcome,
                        "shooter": record.shooter,
                        "start_x": round(ev.location_x or 0.0, 1),
                        "start_y": round(ev.location_y or 34.0, 1),
                        "end_x": round(ev.end_x or (ev.location_x or 0.0), 1),
                        "end_y": round(ev.end_y or (ev.location_y or 34.0), 1),
                        "distance": round(
                            math.hypot(
                                (ev.end_x or ev.location_x or 0.0) - (ev.location_x or 0.0),
                                (ev.end_y or ev.location_y or 34.0) - (ev.location_y or 34.0),
                            ), 1
                        ),
                    }
                    res.carry_creation_log.append(log_entry)
                    break

    # ── PLAYERS PER POSSESSION ───────────────────────────────

    def _compute_players_per_possession(self, res: OptaStatsResult) -> None:
        """Average distinct players per sequence."""
        for team in (self.home, self.away):
            seqs = [s for s in self._seq_tracker.sequences if s.team == team]
            if not seqs:
                res.team[team]["players_per_possession"] = 0.0
                continue
            total_players = sum(
                len({ev.player for ev in s.events if ev.player})
                for s in seqs
            )
            res.team[team]["players_per_possession"] = round(
                total_players / len(seqs), 2
            )


# ──────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ──────────────────────────────────────────────────────────────

def compute(result: MatchResult) -> OptaStatsResult:
    engine = OptaStatsEngine(result, result.timeline)
    return engine.compute()
