"""virtual_gps.py

A per-tick (10 Hz) "GPS" recorder for the PLOFA match simulator.

The match engine already integrates every player's position at 10 Hz
(0.1 s steps) via the continuous off-ball integrator
(``match_engine._offball_move_player``) plus event-time position snaps.
This module taps that stream and writes it out as a real GPS-style log:

  * every player's (x, y) at 10 Hz for the whole match
  * the ball position, sampled on the same clock
  * an INDEPENDENT pass over that log that recomputes physical stats
    (total distance, sprint count, high-speed count, top speed,
    standing/walking/jogging/running/sprinting seconds) directly from the
    raw position deltas — not from the engine's internal accumulation.

The point of recomputing independently is verification/visualisation:
if the numbers produced here match the engine's physics store closely,
the simulation's physical measurement is trustworthy. The raw log can
also be dumped to CSV/GPX for debugging or plotting.

The speed thresholds (m/s) mirror the engine's constants so results are
comparable:
    SPRINT_THRESHOLD        = 7.0
    HIGH_SPEED_THRESHOLD    = 8.5
(Speed-band split thresholds used only for the activity breakdown, and
are configurable via ``band_speeds``.)
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Tuple


class VirtualGPS:
    """Samples every player + the ball at a fixed tick rate and derives
    physical stats from the resulting position trace."""

    # Same thresholds as the engine for comparability.
    SPRINT_THRESHOLD = 7.0        # m/s
    HIGH_SPEED_THRESHOLD = 8.5    # m/s

    # Standard football GPS "high-speed running" threshold (19.8 km/h ≈ 5.5 m/s,
    # per Di Salvo/Carling et al.; many providers use 19.8 km/h).
    HSR_THRESHOLD = 5.5           # m/s

    # Default speed-band split (m/s) for the activity breakdown.
    # Represents the upper bound of each band. Mirrors the SPEEDS table
    # in opta_analytics (standing 0.2, walking 1.5, jogging 2.8,
    # running 4.8, sprinting 8.0).
    BANDS = [
        ("standing", 0.0, 0.6),
        ("walking", 0.6, 2.0),
        ("jogging", 2.0, 3.6),
        ("running", 3.6, 7.0),
        ("sprinting", 7.0, 100.0),
    ]

    # Beyond this, a per-tick displacement is a position discontinuity
    # (kickoff reset, halves reset, event snap), not real movement — same
    # as a real GPS receiver rejecting a satellite jump. Implied speed
    # above this is clamped out and does NOT contribute distance or count
    # toward a sprint band. (Human sprint cap is ~12.5 m/s.)
    MAX_PLAUSIBLE_SPEED = 10.5
    # Absolute top speed a tracked player may honestly achieve given the DNA
    # ceiling (pace 100 -> 9.2 m/s) plus a small margin; anything above is a
    # position-sync artifact and never sets the player's recorded top speed.
    MAX_TRACKED_SPEED = 9.8

    def __init__(
        self,
        tick_s: float = 0.1,
        sprint_cutoff: Optional[float] = None,
        high_speed_cutoff: Optional[float] = None,
    ):
        self.tick_s = tick_s
        self.sprint_cutoff = sprint_cutoff or self.SPRINT_THRESHOLD
        self.high_speed_cutoff = high_speed_cutoff or self.HIGH_SPEED_THRESHOLD

        # Raw log: ordered list of samples, each a dict:
        #   {"t": match_clock_s, "minute": int, "player": name,
        #    "team": team, "position": pos, "x": float, "y": float,
        #    "speed_mps": float}
        # Distances are accumulated geometrically between consecutive
        # samples of the same (player, team) in time order.
        self.samples: List[dict] = []
        self._last_pos: Dict[str, Tuple[float, float]] = {}
        self._current_t: float = 0.0
        self._sprint_state: Dict[str, Dict] = {}

        # Derived, accumulated per player -> stat dict.
        self.players: Dict[str, dict] = {}
        self._ensure_player = lambda name: self.players.setdefault(
            name,
            {
                "team": "",
                "position": "",
                "minutes": 0.0,
                "samples": 0,
                "distance_m": 0.0,
                "sprint_distance_m": 0.0,
                "high_speed_distance_m": 0.0,
                "hsr_distance_m": 0.0,
                "sprint_count": 0,
                "high_speed_count": 0,
                "top_speed_mps": 0.0,
                "band_seconds": {b[0]: 0.0 for b in self.BANDS},
                "band_counts": {b[0]: 0 for b in self.BANDS},
            },
        )

    # ── RECORDING API ────────────────────────────────────────────────

    def begin_minute(self, minute: int, match_clock_s: float) -> None:
        """Anchor the current clock at the start of a minute."""
        self._current_t = match_clock_s

    def record_tick(
        self,
        minute: int,
        match_clock_s: float,
        position_engine,
        home_team: str,
        away_team: str,
        ball_x: Optional[float] = None,
        ball_y: Optional[float] = None,
        skip_accumulate: Optional[set] = None,
    ) -> None:
        """Snapshot every player (and optionally the ball) at this tick.

        Reads current positions directly from ``PositionEngine.states``
        and records them onto the 10 Hz clock. ``ball_x/ball_y`` may be
        passed from ``state.last_ball_x/y``; a per-tick ball sample is
        appended with player == "__ball__". ``match_clock_s`` is the
        sample's timestamp (used for speed = delta / tick_s).

        ``skip_accumulate``: a set of player names whose movement this
        tick should NOT be accumulated from position gaps (they are moved
        by a possession episode and ingested separately via
        ``ingest_episode_stats`` — e.g. on-ball players during a live
        window). Their positions are still logged.
        """
        self._current_t = match_clock_s
        states = position_engine.states
        rosters = position_engine.team_rosters

        for team in (home_team, away_team):
            for name in rosters.get(team, []):
                st = states.get(name)
                if st is None:
                    continue
                self._push_sample(
                    minute,
                    match_clock_s,
                    name,
                    team,
                    getattr(st, "position", ""),
                    st.current_x,
                    st.current_y,
                    accumulate=not (skip_accumulate and name in skip_accumulate),
                )

        if ball_x is not None and ball_y is not None:
            self._push_sample(
                minute, match_clock_s, "__ball__", "", "", ball_x, ball_y
            )

    def ingest_episode_stats(
        self,
        player_stats: Dict[str, Dict[str, float]],
        minute: int,
    ) -> None:
        """Merge on-ball movement measured by a possession-episode physics
        trace into the recorder. The episode integrates per-tick movement of
        players involved in the chain (runs, carries, duels) that the off-ball
        sampler doesn't cover; feeding those totals here keeps the final
        physical stats complete without re-deriving them from position gaps.

        ``player_stats`` matches ``PossessionEpisode.calculate_distance_stats``:
        per player -> {distance_m, sprint_distance_m, high_speed_sprint_distance_m,
                       sprint_count, high_speed_sprint_count, top_speed_mps}.
        """
        for name, s in (player_stats or {}).items():
            if name == "__ball__":
                continue
            p = self._ensure_player(name)
            p["distance_m"] += s.get("distance_m", 0.0)
            p["sprint_distance_m"] += s.get("sprint_distance_m", 0.0)
            p["high_speed_distance_m"] += s.get("high_speed_sprint_distance_m", 0.0)
            p["hsr_distance_m"] += s.get("hsr_distance_m", 0.0)
            p["sprint_count"] += int(s.get("sprint_count", 0))
            p["high_speed_count"] += int(s.get("high_speed_sprint_count", 0))
            p["top_speed_mps"] = max(
                p["top_speed_mps"], s.get("top_speed_mps", 0.0)
            )
            p["samples"] += int(s.get("sprint_count", 0))  # bookkeeping only

    def _push_sample(self, minute, t, name, team, position, x, y,
                     accumulate=True) -> None:
        key = name
        prev = self._last_pos.get(key)
        speed = 0.0
        if prev is not None:
            d = math.hypot(x - prev[0], y - prev[1])
            speed = d / self.tick_s if self.tick_s > 0 else 0.0
            if name != "__ball__" and accumulate:
                # Reject position discontinuities (kickoff/halves resets,
                # event snaps) — a real GPS filters these spikes out.
                if speed <= self.MAX_PLAUSIBLE_SPEED:
                    self._accumulate(name, team, position, d, speed, t)
        self._last_pos[key] = (x, y)
        self.samples.append({
            "t": round(t, 4),
            "minute": minute,
            "player": name,
            "team": team,
            "position": position,
            "x": round(x, 3),
            "y": round(y, 3),
            "speed_mps": round(min(speed, self.MAX_PLAUSIBLE_SPEED), 3),
        })

    def _accumulate(self, name, team, position, d, speed, t) -> None:
        p = self._ensure_player(name)
        if not p["team"]:
            p["team"] = team
        if not p["position"]:
            p["position"] = position
        p["distance_m"] += d
        p["samples"] += 1

        # Sprint / high-speed segment counting, mirroring
        # match_engine._record_offball_distance (min consecutive ticks to
        # open a segment + cooldown), but computed purely from the log.
        sprint_inc = 0
        hi_inc = 0
        st = self._sprint_state.setdefault(
            name,
            {
                "in_s": False, "in_h": False,
                "s_run": 0, "h_run": 0,
                "s_cd": 0, "h_cd": 0,
            },
        )
        in_s = speed >= self.sprint_cutoff
        in_h = speed >= self.high_speed_cutoff
        if in_s:
            st["s_cd"] = 0
            st["s_run"] += 1
            if st["s_run"] >= 3 and not st["in_s"]:
                sprint_inc = 1
                st["in_s"] = True
        else:
            if st["s_run"] >= 3:
                sprint_inc = 1
            st["s_run"] = 0
            st["in_s"] = False
            st["s_cd"] = 5
        if in_h:
            st["h_cd"] = 0
            st["h_run"] += 1
            if st["h_run"] >= 2 and not st["in_h"]:
                hi_inc = 1
                st["in_h"] = True
        else:
            if st["h_run"] >= 2:
                hi_inc = 1
            st["h_run"] = 0
            st["in_h"] = False
            st["h_cd"] = 4

        # Only a SUSTAINED high-speed tick (>=2 consecutive) may set the
        # player's top speed. Isolated one-tick position spikes (event sync
        # snaps, loose-ball scrambles) are filtered, matching a real GPS
        # low-pass. Genuine chase bursts and races run many consecutive
        # high ticks, so their peaks still register.
        if (self.MAX_TRACKED_SPEED >= speed > p["top_speed_mps"]
                and (st["s_run"] >= 2 or st["h_run"] >= 2)):
            p["top_speed_mps"] = speed
        p["sprint_count"] += sprint_inc
        p["high_speed_count"] += hi_inc
        if speed >= self.HSR_THRESHOLD:
            p["hsr_distance_m"] += d
        if in_s:
            p["sprint_distance_m"] += d
        if in_h:
            p["high_speed_distance_m"] += d
        if speed > p["top_speed_mps"]:
            p["top_speed_mps"] = speed

        # Speed-band seconds.
        for label, lo, hi in self.BANDS:
            if lo <= speed < hi:
                p["band_seconds"][label] += self.tick_s
                p["band_counts"][label] += 1
                break

    # ── DERIVED OUTPUT ───────────────────────────────────────────────

    def player_summary(self, name: str) -> Optional[dict]:
        p = self.players.get(name)
        if p is None:
            return None
        dist_m = p["distance_m"]
        return {
            "player": name,
            "team": p["team"],
            "position": p["position"],
            "samples": p["samples"],
            "duration_min": p["samples"] * self.tick_s / 60.0,
            "distance_m": round(dist_m, 1),
            "distance_km": round(dist_m / 1000.0, 2),
            "sprint_distance_m": round(p["sprint_distance_m"], 1),
            "high_speed_distance_m": round(p["high_speed_distance_m"], 1),
            "hsr_distance_m": round(p["hsr_distance_m"], 1),
            "sprint_count": p["sprint_count"],
            "high_speed_count": p["high_speed_count"],
            "top_speed_mps": round(p["top_speed_mps"], 2),
            "top_speed_kmh": round(p["top_speed_mps"] * 3.6, 1),
            "activity_seconds": {
                k: round(v, 1) for k, v in p["band_seconds"].items()
            },
        }

    def all_summaries(self) -> List[dict]:
        return [
            self.player_summary(name)
            for name in sorted(self.players)
            if name != "__ball__"
        ]

    def max_speed_all(self) -> float:
        return max((p["top_speed_mps"] for p in self.players.values()
                    if p["team"]), default=0.0)

    # ── EXPORT ───────────────────────────────────────────────────────

    def to_csv(self, path: str) -> str:
        """Write the raw per-tick log to CSV. Returns the path written."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=[
                    "t", "minute", "player", "team", "position",
                    "x", "y", "speed_mps",
                ],
            )
            w.writeheader()
            for s in self.samples:
                w.writerow(s)
        return path

    def to_summary_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows = self.all_summaries()
        # Build dynamic fieldnames (flat + activity second columns).
        fieldnames = []
        if rows:
            base_fields = [k for k in rows[0] if k != "activity_seconds"]
            act_fields = [f"sec_{b[0]}" for b in self.BANDS]
            fieldnames = base_fields + act_fields
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                row = {k: v for k, v in r.items() if k != "activity_seconds"}
                act = r.get("activity_seconds", {})
                for k, v in act.items():
                    row[f"sec_{k}"] = v
                w.writerow(row)
        return path
