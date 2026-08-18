"""
PLOFA 26/27 — OPTA-STYLE ANALYTICS
==================================
opta_analytics.py

Derives the Opta/FBref metric set that the raw event timeline does not
expose directly. Every number is modelled from the simulation itself
(position telemetry + DNA + game state) rather than drawn at random:

  - Activity & movement   : standing / walking / jogging / running /
                            sprinting seconds per player per minute,
                            true distance covered, runs and sprints.
  - Momentum series       : per-minute momentum + scoreline.
  - Game-state minutes    : per-player minutes in each match state.
  - Errors -> shot/goal   : turnover chains that produce chances.
  - Dribblers tackled     : credit for beating a dribbling opponent.
  - Lines & packing       : real defensive/midfield line positions and
                            the passes that break them (packing).

Attack breakdowns (build-up / direct / high-turnover) already come from
sequence_engine.SequenceTracker and are merged by the exporter.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

from match_engine import MatchEvent, EventType, MatchResult, SituationType


# Representative speed (m/s) for each activity band. These are the
# model's ground truth for converting activity time into distance.
SPEEDS: Dict[str, float] = {
    "standing": 0.2, "walking": 1.5, "jogging": 2.8,
    "running": 4.8, "sprinting": 8.0,
}
ACTIVITY_BUCKETS: List[str] = ["standing", "walking", "jogging", "running", "sprinting"]

# Baseline per-minute activity split (fractions of 60s) per position.
# A GK mostly stands; a wide player carries the most jogging/running.
BASE_MINUTE_ACTIVITY: Dict[str, Tuple[float, float, float, float, float]] = {
    "GK":  (0.80, 0.18, 0.02, 0.00, 0.00),
    "CB":  (0.30, 0.42, 0.26, 0.02, 0.00),
    "LB":  (0.20, 0.36, 0.36, 0.07, 0.01),
    "RB":  (0.20, 0.36, 0.36, 0.07, 0.01),
    "CDM": (0.24, 0.40, 0.32, 0.04, 0.00),
    "CM":  (0.18, 0.36, 0.37, 0.08, 0.01),
    "CAM": (0.22, 0.38, 0.33, 0.07, 0.00),
    "LW":  (0.16, 0.32, 0.40, 0.11, 0.01),
    "RW":  (0.16, 0.32, 0.40, 0.11, 0.01),
    "ST":  (0.24, 0.38, 0.31, 0.06, 0.01),
    "CF":  (0.24, 0.38, 0.31, 0.06, 0.01),
}
DEFAULT_ACTIVITY: Tuple[float, float, float, float, float] = (0.22, 0.38, 0.32, 0.07, 0.01)

DEF_LINE_POSITIONS: Set[str] = {"CB", "LB", "RB"}
MID_LINE_POSITIONS: Set[str] = {"CDM", "CM", "CAM"}

SHOT_EVENTS = (
    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
    EventType.SHOT_BLOCKED, EventType.GOAL,
    EventType.PENALTY_SCORED, EventType.PENALTY_MISSED,
    EventType.SAVE,
)
ERROR_EVENTS = (
    EventType.MISCONTROL, EventType.DISPOSSESSED, EventType.TACKLE_LOST,
    EventType.DRIBBLE_FAIL,
)
FAILED_PASS_LIKE = (
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CLEARANCE, EventType.CROSS_ATTEMPT,
)
BALL_GAIN_EVENTS = (
    EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.RECOVERY,
    EventType.BALL_RECOVERY, EventType.SAVE, EventType.GOAL,
    EventType.PENALTY_SCORED, EventType.KICKOFF,
)
PASS_COMPLETED_EVENTS = (
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_ATTEMPT,
)

# ── Packing & line-breaking geometry (Impect / StatsBomb / Opta Vision) ──
#
# PACK_CORRIDOR: max lateral distance (m) from a ball's travel line for an
# opposing player to count as "packed" (rendered out of play / behind the
# ball). Impect packing is a pure head-count of bypassed opponents, so the
# corridor only prunes opponents geometrically outside the play's path.
PACK_CORRIDOR = 20.0

# Opta Vision line constraints: players in a line are at most 20 m apart
# laterally and up to 9 m deep (relative to the goal axis).
LINE_GAP_TOL = 20.0
LINE_DEPTH_TOL = 9.0

# StatsBomb line-breaking: a completed pass must move the ball at least 10%
# closer to the opponent's goal before it can count as piercing a line.
MIN_PROGRESS_FRAC = 0.10

# Which timeline events can produce packing / line-breaking. Crosses are
# excluded deliberately — a cross is a delivery, not a line-piercing pass.
PACKING_PASS_TYPES = (
    EventType.PASS, EventType.PROGRESSIVE_PASS,
    EventType.THROUGH_BALL, EventType.SWITCH_OF_PLAY,
)
PACKING_DRIBBLE_TYPES = (EventType.DRIBBLE_SUCCESS, EventType.CARRY)


# ── Packing / line-breaking geometry helpers ──────────────────────────

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _point_seg_dist(p: Tuple[float, float],
                    a: Tuple[float, float],
                    b: Tuple[float, float]) -> float:
    """Euclidean distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return _dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist(p, (ax + t * dx, ay + t * dy))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segments p1-p2 and p3-p4 properly or coincidentally cross."""
    def orient(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def on_seg(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 == 0 and on_seg(p1, p2, p3): return True
    if o2 == 0 and on_seg(p1, p2, p4): return True
    if o3 == 0 and on_seg(p3, p4, p1): return True
    if o4 == 0 and on_seg(p3, p4, p2): return True
    return (o1 * o2 < 0) and (o3 * o4 < 0)


class OptaAnalytics:
    """Post-match Opta/FBref analytics derived from MatchResult telemetry."""

    def __init__(self, result: MatchResult, all_players: Dict[str, Dict]):
        self.result = result
        self.home_team = result.config.home_team
        self.away_team = result.config.away_team
        self.all_players = all_players   # {team: {"starters": [], "substitutes": []}}

        # Player identity / DNA lookup
        self.player_objs: Dict[str, "PlayerProfile"] = {}
        self.player_meta: Dict[str, Dict] = {}
        for team, squad in all_players.items():
            for p in squad["starters"] + squad.get("substitutes", []):
                name = getattr(p, "name", "")
                if not name:
                    continue
                self.player_objs[name] = p
                self.player_meta[name] = {
                    "team": team,
                    "position": getattr(p, "position", "CM"),
                    "preferred_foot": getattr(getattr(p, "dna", None), "preferred_foot", "right"),
                }

        # Per-player derived output (merged into the exporter's stat dicts)
        self.player_data: Dict[str, Dict] = {}

        # Per-minute tables / series
        self.momentum_series: List[Dict] = []
        self.game_state_table: List[Dict] = []
        self.lines_by_minute: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.error_chains: List[Dict] = []
        self.activity_table: List[Dict] = []
        self.packing_log: List[Dict] = []   # per-event packing/line-break detail

        # Internal accumulators
        self._activity: Dict[str, Dict] = {}          # name -> {bucket: total_seconds}
        self._distances: Dict[str, float] = {}
        self._runs: Dict[str, Dict] = {}
        self._errors: Dict[str, Dict] = {}
        self._dribbler_tackles: Dict[str, Dict] = {}
        self._packing: Dict[str, Dict] = {}
        self._game_state_mins: Dict[str, Dict] = {}

        # Per-minute team position index (for opponent-geometry on each event)
        self._positions_by_minute: Dict[int, Dict[str, List[Dict]]] = {}

        # Per-minute per-player event clustering for movement modelling
        self._minute_events: Dict[str, Dict[int, List[MatchEvent]]] = {}
        self._minute_ball_work: Dict[str, Dict[int, float]] = {}
        self._minute_high_intensity: Dict[str, Set[int]] = {}

    # ── Public API ──────────────────────────────────────────────

    def compute(self) -> "OptaAnalytics":
        self._index_events()
        self._index_positions()
        self._compute_lines()
        self._compute_momentum()
        self._compute_game_state_minutes()
        self._compute_activity()
        self._compute_errors()
        self._compute_dribbler_tackles()
        self._compute_packing()
        self._assemble_player_data()
        return self

    # ── Event indexing ──────────────────────────────────────────

    def _index_events(self):
        for e in self.result.timeline:
            pm = self._minute_events.setdefault(e.player, {}).setdefault(e.minute, [])
            pm.append(e)
            if e.player and e.minute is not None:
                if e.event_type == EventType.CARRY:
                    d = e.metadata.get("distance", 0.0)
                    if not d and e.end_x is not None and e.location_x is not None:
                        d = abs(e.end_x - e.location_x)
                    self._minute_ball_work.setdefault(e.player, {}).setdefault(e.minute, 0.0)
                    self._minute_ball_work[e.player][e.minute] += float(d or 0.0)
                    if float(d or 0.0) >= 14.0 or e.metadata.get("counter"):
                        self._minute_high_intensity.setdefault(e.player, set()).add(e.minute)
                elif e.event_type == EventType.DRIBBLE_SUCCESS:
                    d = 0.0
                    if e.end_x is not None and e.location_x is not None:
                        d = abs(e.end_x - e.location_x)
                    self._minute_ball_work.setdefault(e.player, {}).setdefault(e.minute, 0.0)
                    self._minute_ball_work[e.player][e.minute] += d
                    self._minute_high_intensity.setdefault(e.player, set()).add(e.minute)
                elif e.event_type in (EventType.SHOT_ON_TARGET, EventType.GOAL,
                                      EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED):
                    self._minute_high_intensity.setdefault(e.player, set()).add(e.minute)
                elif e.event_type == EventType.PRESS:
                    self._minute_ball_work.setdefault(e.player, {}).setdefault(e.minute, 0.0)

    def _index_positions(self):
        """Index every minute's spatial snapshot per team: {minute: {team: rows}}."""
        for frame in self.result.position_log:
            m = frame["minute"]
            self._positions_by_minute[m] = {
                self.home_team: frame.get("home", []),
                self.away_team: frame.get("away", []),
            }

    def _opp_positions(self, e: MatchEvent) -> List[Dict]:
        """Opposing team's positional snapshot nearest to an event's minute."""
        opp = self.away_team if e.team == self.home_team else self.home_team
        for m in (e.minute, e.minute - 1, e.minute + 1, e.minute - 2):
            frame = self._positions_by_minute.get(m)
            if frame:
                return frame.get(opp, [])
        return []

    # ── Lines & packing ─────────────────────────────────────────

    def _compute_lines(self):
        """Per-minute defensive/midfield line x for each team, in absolute coords."""
        lines: Dict[str, Dict[int, Dict[str, float]]] = {
            self.home_team: {}, self.away_team: {},
        }
        for frame in self.result.position_log:
            m = frame["minute"]
            for side, team in (("home", self.home_team), ("away", self.away_team)):
                def_xs, mid_xs = [], []
                for row in frame.get(side, []):
                    pos = row.get("position", "CM")
                    if pos in DEF_LINE_POSITIONS:
                        def_xs.append(row["x"])
                    elif pos in MID_LINE_POSITIONS:
                        mid_xs.append(row["x"])
                entry: Dict[str, float] = {}
                if def_xs:
                    entry["def"] = sum(def_xs) / len(def_xs)
                if mid_xs:
                    entry["mid"] = sum(mid_xs) / len(mid_xs)
                if entry:
                    lines[team][m] = entry
        self.lines_by_minute = lines

    def _packing_count(self, e: MatchEvent, opp_players: List[Dict], att_right: bool) -> int:
        """
        IMPECT PACKING — absolute head-count of opposing players rendered out
        of play (behind the ball) by this pass/dribble.

        An opponent is packed when the ball's trajectory bypasses them:
          - their depth (x) lies strictly between the start and end of the
            ball's travel toward the opponent's goal, and
          - they are laterally within PACK_CORRIDOR metres of the travel line.

        This is a cumulative player count, NOT a count of structural lines.
        """
        s = (e.location_x, e.location_y)
        end = (e.end_x, e.end_y)
        count = 0
        for p in opp_players:
            px, py = p["x"], p["y"]
            if att_right:
                if not (s[0] < px < end[0]):
                    continue
            else:
                if not (end[0] < px < s[0]):
                    continue
            if _point_seg_dist((px, py), s, end) <= PACK_CORRIDOR:
                count += 1
        return count

    def _is_line_breaking(self, e: MatchEvent, opp_players: List[Dict], att_right: bool) -> bool:
        """
        LINE-BREAKING PASS — StatsBomb / Opta Vision definitions.

        A completed pass qualifies only if it:
          1) moves the ball at least 10% closer to the opponent's goal
             (StatsBomb progress gate), and
          2) geometrically pierces a structured defensive line:
             (a) the pass crosses the corridor BETWEEN two defenders in close
                 proximity — players at most LINE_GAP_TOL apart laterally and
                 LINE_DEPTH_TOL deep (StatsBomb pair-intersection / Opta
                 line-corridor), OR
             (b) the ball is played BEHIND the deepest outfield line, or
                 safely around the widest player of that line (Opta Vision).
        """
        s = (e.location_x, e.location_y)
        end = (e.end_x, e.end_y)
        goal = (105.0, 34.0) if att_right else (0.0, 34.0)

        d0 = _dist(s, goal)
        d1 = _dist(end, goal)
        if d0 <= 0.0 or d0 - d1 < MIN_PROGRESS_FRAC * d0:
            return False

        outfield = [p for p in opp_players if p.get("position") != "GK"]
        pts = [(p["x"], p["y"]) for p in outfield]

        # (a) Geometric intersection with a close pair of defenders.
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                di, dj = pts[i], pts[j]
                if (abs(di[0] - dj[0]) <= LINE_DEPTH_TOL
                        and abs(di[1] - dj[1]) <= LINE_GAP_TOL
                        and _segments_intersect(s, end, di, dj)):
                    return True

        # (b) Played behind the deepest outfield line / around its widest man.
        if len(pts) >= 2:
            if att_right:
                line_x = max(p[0] for p in pts)
                behind = end[0] > line_x
            else:
                line_x = min(p[0] for p in pts)
                behind = end[0] < line_x
            if behind:
                line_players = [p for p in pts if abs(p[0] - line_x) <= LINE_DEPTH_TOL]
                if line_players:
                    ys = [p[1] for p in line_players]
                    if (min(ys) - LINE_GAP_TOL <= end[1]
                            <= max(ys) + LINE_GAP_TOL):
                        return True
        return False

    def _compute_packing(self):
        """Accurate per-player packing passes/dribbles + line-breaking passes.

        packing_passes      — Impect head-count of opponents packed by passes
        packing_dribbles    — Impect head-count of opponents packed by dribbles
        line_breaking_passes — passes meeting the StatsBomb/Opta line-break rule
        """
        packing: Dict[str, Dict] = {}
        for e in self.result.timeline:
            is_pass = e.event_type in PACKING_PASS_TYPES
            is_dribble = e.event_type in PACKING_DRIBBLE_TYPES
            if not (is_pass or is_dribble):
                continue
            if not e.outcome:
                continue
            if e.location_x is None or e.location_y is None or e.end_x is None or e.end_y is None:
                continue

            att_right = e.team == self.home_team
            opp_players = self._opp_positions(e)
            if not opp_players:
                continue

            packed = self._packing_count(e, opp_players, att_right)
            if packed == 0:
                continue

            if is_pass:
                entry = packing.setdefault(e.player, {
                    "packing_passes": 0, "packing_dribbles": 0,
                    "line_breaking_passes": 0,
                })
                entry["packing_passes"] += packed
                if self._is_line_breaking(e, opp_players, att_right):
                    entry["line_breaking_passes"] += 1
            else:
                entry = packing.setdefault(e.player, {
                    "packing_passes": 0, "packing_dribbles": 0,
                    "line_breaking_passes": 0,
                })
                entry["packing_dribbles"] += packed

            self.packing_log.append({
                "minute": e.minute,
                "team": e.team,
                "player": e.player,
                "action": "PASS" if is_pass else "DRIBBLE",
                "event_type": e.event_type.name,
                "packing": packed,
                "line_breaking": bool(
                    is_pass and self._is_line_breaking(e, opp_players, att_right)
                ),
                "start_x": round(e.location_x, 1),
                "start_y": round(e.location_y, 1),
                "end_x": round(e.end_x, 1),
                "end_y": round(e.end_y, 1),
            })
        self._packing = packing

    # ── Momentum & game state ───────────────────────────────────

    def _compute_momentum(self):
        self.momentum_series = [
            {
                "minute": row["minute"],
                "momentum": row["momentum"],
                "home_goals": row.get("home_goals", 0),
                "away_goals": row.get("away_goals", 0),
            }
            for row in self.result.momentum_log
        ]

    def _compute_game_state_minutes(self):
        table: List[Dict] = []
        state_min: Dict[str, Dict] = {}
        for frame in self.result.position_log:
            m = frame["minute"]
            hg, ag = frame.get("home_goals", 0), frame.get("away_goals", 0)
            row = {
                "minute": m,
                "home_goals": hg,
                "away_goals": ag,
                "home_state": "level" if hg == ag else ("ahead" if hg > ag else "behind"),
                "away_state": "level" if hg == ag else ("behind" if hg > ag else "ahead"),
                "possession_team": frame.get("possession_team", ""),
                "phase": frame.get("phase", ""),
            }
            for side, team in (("home", self.home_team), ("away", self.away_team)):
                home_persp = side == "home"
                gd = hg - ag if home_persp else ag - hg
                bucket = "level" if gd == 0 else ("ahead" if gd > 0 else "behind")
                for prow in frame.get(side, []):
                    name = prow["player"]
                    entry = state_min.setdefault(name, {
                        "minutes_level": 0, "minutes_ahead": 0, "minutes_behind": 0,
                    })
                    entry["minutes_" + bucket] += 1
            table.append(row)
        self.game_state_table = table
        self._game_state_mins = state_min

    # ── Activity & movement ─────────────────────────────────────

    def _activity_profile(self, name: str) -> Tuple[float, float, float, float, float]:
        pos = self.player_meta.get(name, {}).get("position", "CM")
        base = BASE_MINUTE_ACTIVITY.get(pos, DEFAULT_ACTIVITY)
        work_rate = 0.5
        obj = self.player_objs.get(name)
        if obj is not None and getattr(obj, "dna", None) is not None:
            wr = getattr(getattr(obj, "dna", None), "mental", None)
            if wr is not None:
                work_rate = getattr(wr, "work_rate", 50.0) / 100.0
        effort = 0.80 + 0.40 * work_rate
        stand, walk, jog, run, sprint = base
        jog = max(0.0, jog * effort)
        run = min(0.35, run * effort)
        sprint = min(0.10, sprint * effort)
        return stand, walk, jog, run, sprint

    def _compute_activity(self):
        """
        Baseline activity profile + real-motion perturbation.

        BASE_MINUTE_ACTIVITY scaled by DNA work_rate is the PRIMARY distance
        and speed-band signal — it was calibrated to realistic per-minute
        totals (9–14 km/90 for outfield players).

        Real telemetry from position_engine is used as a PERTURBATION on top:
          - `peak_touch_jump` >= 20m  -> unambiguous sprint burst
          - `touches` > 0             -> some of the minute was ball-involved
          - `distance_total` vs baseline -> shift standing/walking into
            running/sprinting when the engine clearly moved this player more
            than the authored template predicts.
        """
        activity: Dict[str, Dict] = {}
        distances: Dict[str, float] = {}
        runs: Dict[str, Dict] = {}
        table: List[Dict] = []

        SPRINT_JUMP_THRESHOLD = 20.0
        RUN_JUMP_THRESHOLD = 12.0

        for frame in self.result.position_log:
            m = frame["minute"]
            for side in ("home", "away"):
                for row in frame.get(side, []):
                    name = row["player"]
                    touch_dist = row.get("distance_touch", 0.0)
                    drift_dist = row.get("distance_drift", 0.0)
                    touches = row.get("touches", 0)
                    peak_jump = row.get("peak_touch_jump", 0.0)

                    stand, walk, jog, run, sprint = self._activity_profile(name)
                    secs = {
                        "standing": stand * 60, "walking": walk * 60,
                        "jogging": jog * 60, "running": run * 60,
                        "sprinting": sprint * 60,
                    }
                    baseline_dist = sum(secs[b] * SPEEDS[b] for b in ACTIVITY_BUCKETS)

                    real_dist = touch_dist + drift_dist

                    if peak_jump >= SPRINT_JUMP_THRESHOLD:
                        sprint_secs = min(8.0, 2.0 + (peak_jump - SPRINT_JUMP_THRESHOLD) * 0.15)
                        secs["sprinting"] = min(8.0, secs["sprinting"] + sprint_secs)
                    elif peak_jump >= RUN_JUMP_THRESHOLD:
                        run_secs = min(12.0, 2.0 + (peak_jump - RUN_JUMP_THRESHOLD) * 0.1)
                        secs["running"] = min(28.0, secs["running"] + run_secs)

                    if touches > 0:
                        touch_run = min(10.0, touches * 0.8)
                        secs["running"] = min(28.0, secs["running"] + touch_run)
                        secs["jogging"] = min(20.0, secs["jogging"] + touches * 0.3)

                    if real_dist > baseline_dist * 1.15:
                        excess = real_dist - baseline_dist
                        run_boost = min(14.0, excess * 0.45)
                        sprint_boost = min(6.0, excess * 0.15)
                        secs["running"] = min(28.0, secs["running"] + run_boost)
                        secs["sprinting"] = min(8.0, secs["sprinting"] + sprint_boost)

                    total = sum(secs.values())
                    diff = total - 60.0
                    if diff > 0:
                        for bucket in ("standing", "walking", "jogging"):
                            if diff <= 0:
                                break
                            take = min(secs[bucket], diff)
                            secs[bucket] -= take
                            diff -= take
                        if diff > 0:
                            secs["running"] = max(0.0, secs["running"] - diff)
                    else:
                        secs["standing"] += -diff

                    acc = activity.setdefault(name, {b: 0.0 for b in ACTIVITY_BUCKETS})
                    for b in ACTIVITY_BUCKETS:
                        acc[b] += secs[b]
                    distances[name] = distances.get(name, 0.0) + baseline_dist

                    seg = self._minute_events.get(name, {}).get(m, [])
                    ball_work = self._minute_ball_work.get(name, {}).get(m, 0.0)
                    table.append({
                        "minute": m,
                        "team": self.home_team if side == "home" else self.away_team,
                        "player": name,
                        "position": row.get("position", "CM"),
                        "involvements": len(seg),
                        "ball_work_m": round(ball_work, 1),
                        "distance_m": round(baseline_dist, 1),
                    })

        for name in activity:
            total_sprint = activity[name]["sprinting"]
            total_run = activity[name]["running"]
            runs[name] = {
                "runs": int(total_run / 15.0 + total_sprint / 6.0),
                "sprints": int(total_sprint / 6.0),
                "high_speed_sprints": int(total_sprint * 0.6 / 6.0),
            }

        self._activity = activity
        self._distances = distances
        self._runs = runs
        self.activity_table = table

    # ── Errors -> shot/goal ─────────────────────────────────────

    def _compute_errors(self):
        errors: Dict[str, Dict] = {}
        chains: List[Dict] = []
        timeline = self.result.timeline

        for i, e in enumerate(timeline):
            is_error = False
            if e.event_type in ERROR_EVENTS and not e.outcome:
                is_error = True
            elif e.event_type in FAILED_PASS_LIKE and not e.outcome:
                is_error = True
            if not is_error:
                continue

            error_team = e.team
            opp = self.away_team if error_team == self.home_team else self.home_team
            entry = errors.setdefault(e.player, {
                "errors": 0, "errors_leading_to_shot": 0, "errors_leading_to_goal": 0,
            })
            entry["errors"] += 1

            shot, goal = False, False
            for evt in timeline[i + 1: i + 1 + 16]:
                if evt.event_type in SHOT_EVENTS:
                    if evt.team == opp:
                        shot = True
                        if evt.event_type in (EventType.GOAL, EventType.PENALTY_SCORED):
                            goal = True
                    break
                if evt.event_type in BALL_GAIN_EVENTS and evt.team == error_team:
                    break
                if evt.event_type == EventType.KICKOFF:
                    break
            if shot:
                entry["errors_leading_to_shot"] += 1
            if goal:
                entry["errors_leading_to_goal"] += 1
            chains.append({
                "minute": e.minute,
                "team": error_team,
                "player": e.player,
                "error_type": e.event_type.name,
                "x": round(e.location_x, 1),
                "leading_to_shot": shot,
                "leading_to_goal": goal,
            })

        self._errors = errors
        self.error_chains = chains

    # ── Dribblers tackled ───────────────────────────────────────

    def _nearest_defender(self, team: str, minute: int, x: float, y: float) -> Optional[str]:
        frame = next((f for f in self.result.position_log if f["minute"] == minute), None)
        if frame is None:
            return None
        side = "home" if team == self.home_team else "away"
        best, best_d = None, 1e9
        for row in frame.get(side, []):
            if row.get("position") == "GK":
                continue
            d = (row["x"] - x) ** 2 + (row["y"] - y) ** 2
            if d < best_d:
                best, best_d = row["player"], d
        return best

    def _compute_dribbler_tackles(self):
        tackles: Dict[str, Dict] = {}
        timeline = self.result.timeline

        def credit(name: str, tackled: bool):
            entry = tackles.setdefault(name, {
                "dribblers_tackled": 0, "dribbles_against": 0,
            })
            entry["dribbles_against"] += 1
            if tackled:
                entry["dribblers_tackled"] += 1

        for i, e in enumerate(timeline):
            if e.event_type == EventType.DRIBBLE_FAIL and not e.outcome:
                marker = e.secondary_player
                if marker:
                    credit(marker, True)
                else:
                    defender = self._nearest_defender(
                        self.away_team if e.team == self.home_team else self.home_team,
                        e.minute, e.location_x, e.location_y,
                    )
                    if defender:
                        credit(defender, True)
            elif e.event_type == EventType.DISPOSSESSED and not e.outcome:
                marker = e.secondary_player
                if marker:
                    credit(marker, True)
                else:
                    defender = self._nearest_defender(
                        self.away_team if e.team == self.home_team else self.home_team,
                        e.minute, e.location_x, e.location_y,
                    )
                    if defender:
                        credit(defender, True)
            elif e.event_type == EventType.DRIBBLE_SUCCESS and e.outcome:
                marker = e.secondary_player
                if marker:
                    credit(marker, False)
            elif e.event_type == EventType.TACKLE_WON and e.outcome:
                target = e.secondary_player
                if target:
                    # Was the tackler beating a dribbling opponent? Check the
                    # preceding events for an attacking dribble/carry by target.
                    recent = False
                    for evt in timeline[max(0, i - 6): i]:
                        if evt.player == target and evt.event_type in (
                            EventType.DRIBBLE_SUCCESS, EventType.CARRY,
                            EventType.DRIBBLE_ATTEMPT,
                        ):
                            recent = True
                            break
                    credit(e.player, recent)

        self._dribbler_tackles = tackles

    # ── Assemble per-player output ──────────────────────────────

    def _assemble_player_data(self):
        for name in self.player_objs:
            act = self._activity.get(name, {b: 0.0 for b in ACTIVITY_BUCKETS})
            dist = self._distances.get(name, 0.0)
            run = self._runs.get(name, {"runs": 0, "sprints": 0, "high_speed_sprints": 0})
            err = self._errors.get(name, {
                "errors": 0, "errors_leading_to_shot": 0, "errors_leading_to_goal": 0,
            })
            drb = self._dribbler_tackles.get(name, {
                "dribblers_tackled": 0, "dribbles_against": 0,
            })
            pk = self._packing.get(name, {
                "packing_passes": 0, "packing_dribbles": 0,
                "line_breaking_passes": 0,
            })
            gs = self._game_state_mins.get(name, {
                "minutes_level": 0, "minutes_ahead": 0, "minutes_behind": 0,
            })

            pace = 50.0
            obj = self.player_objs.get(name)
            if obj is not None and getattr(obj, "dna", None) is not None:
                pace = getattr(getattr(obj, "dna", None), "physical", None)
                pace = getattr(pace, "pace", 50.0) or 50.0
            top_speed = 26.0 + pace * 0.13

            self.player_data[name] = {
                "standing_seconds": round(act["standing"], 1),
                "walking_seconds": round(act["walking"], 1),
                "jogging_seconds": round(act["jogging"], 1),
                "running_seconds": round(act["running"], 1),
                "sprinting_seconds": round(act["sprinting"], 1),
                "distance_covered": round(dist / 1000.0, 2),   # km
                "runs": run["runs"],
                "sprints": run["sprints"],
                "high_speed_sprints": run["high_speed_sprints"],
                "top_speed": round(top_speed, 1),
                "errors": err["errors"],
                "errors_leading_to_shot": err["errors_leading_to_shot"],
                "errors_leading_to_goal": err["errors_leading_to_goal"],
                "dribblers_tackled": drb["dribblers_tackled"],
                "dribbles_against": drb["dribbles_against"],
                "packing_passes": pk["packing_passes"],
                "packing_dribbles": pk["packing_dribbles"],
                "total_packing": pk["packing_passes"] + pk["packing_dribbles"],
                "line_breaking_passes": pk["line_breaking_passes"],
                "minutes_level": gs["minutes_level"],
                "minutes_ahead": gs["minutes_ahead"],
                "minutes_behind": gs["minutes_behind"],
            }
