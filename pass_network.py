"""
PLOFA 26/27 — REAL PASS NETWORK MODULE
==========================================
pass_network.py

Closes the technical gap flagged directly in the analyst feedback:
    "You're not actually tracking WHICH players passed to WHICH other
    players - you're estimating from positions... To make this truly
    StatsBomb-level, you'd need to track secondary_player on every PASS
    event and build actual pass matrices."

Good news found while wiring this up: every PASS / PROGRESSIVE_PASS /
SWITCH_OF_PLAY / THROUGH_BALL / CROSS_SUCCESS event in event_chain.py
ALREADY carries `secondary_player` = the intended/actual receiver, and
`outcome` = whether it was completed. That data was just never being
read. This module reads it and builds a real weighted directed pass
matrix plus average touch locations, replacing exporter.py's
`_estimate_connection` heuristic entirely.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from match_engine import MatchEvent, EventType


PASS_EVENT_TYPES = (
    EventType.PASS,
    EventType.PROGRESSIVE_PASS,
    EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL,
)


@dataclass
class PassMatrix:
    """
    A real player x player directed pass matrix for one team, built
    directly from the event timeline (not estimated from adjacency).
    """
    team_name: str

    # matrix[passer][receiver] = completed pass count
    matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    attempted: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # Average touch location, built from every event this player is the
    # primary actor in (location_x/location_y) — this is what StatsBomb
    # uses to place nodes on a real pass map, rather than a fixed
    # position-template coordinate.
    _loc_sum: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0]))

    # Sum of real completed-pass distances per passer->receiver pair,
    # computed from each event's own location_x/y -> end_x/y — not an
    # estimate. Used for "Avg Distance" per combination and per player.
    _dist_sum: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    _player_dist_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _player_dist_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Progressive completed-pass count per passer->receiver pair (PASS
    # events flagged is_progressive in metadata, plus all PROGRESSIVE_PASS
    # events, which are progressive by definition).
    _progressive: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    @classmethod
    def build(cls, team_name: str, timeline: List[MatchEvent]) -> "PassMatrix":
        pm = cls(team_name=team_name)
        for e in timeline:
            if e.team != team_name:
                continue

            # Touch-location accumulation (any event with a primary actor + coords)
            if e.player and e.location_x is not None and e.location_y is not None:
                acc = pm._loc_sum[e.player]
                acc[0] += e.location_x
                acc[1] += e.location_y
                acc[2] += 1

            if e.event_type in PASS_EVENT_TYPES and e.secondary_player:
                pm.attempted[e.player][e.secondary_player] += 1
                if e.outcome:
                    pm.matrix[e.player][e.secondary_player] += 1

                    if (e.location_x is not None and e.location_y is not None
                            and e.end_x is not None and e.end_y is not None):
                        dist = ((e.end_x - e.location_x) ** 2 +
                                (e.end_y - e.location_y) ** 2) ** 0.5
                        pm._dist_sum[e.player][e.secondary_player] += dist
                        pm._player_dist_sum[e.player] += dist
                        pm._player_dist_count[e.player] += 1

                    is_prog = (e.event_type == EventType.PROGRESSIVE_PASS
                               or (e.metadata or {}).get("is_progressive", False))
                    if is_prog:
                        pm._progressive[e.player][e.secondary_player] += 1

            elif e.event_type == EventType.CROSS_SUCCESS and e.secondary_player:
                pm.matrix[e.player][e.secondary_player] += 1
                pm.attempted[e.player][e.secondary_player] += 1

        return pm

    def average_position(self, player_name: str) -> Optional[Tuple[float, float]]:
        acc = self._loc_sum.get(player_name)
        if not acc or acc[2] == 0:
            return None
        return round(acc[0] / acc[2], 2), round(acc[1] / acc[2], 2)

    def connection_strength(self, p1: str, p2: str) -> int:
        """Total COMPLETED passes between two players, either direction —
        this is the real number, not an estimate."""
        return self.matrix[p1].get(p2, 0) + self.matrix[p2].get(p1, 0)

    def top_combinations(self, n: int = 10) -> List[Tuple[str, str, int]]:
        pairs: Dict[Tuple[str, str], int] = {}
        for p1, receivers in self.matrix.items():
            for p2, count in receivers.items():
                key = tuple(sorted([p1, p2]))
                pairs[key] = pairs.get(key, 0) + count
        ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [(k[0], k[1], v) for k, v in ranked]

    def player_pass_accuracy_by_target(self, player_name: str) -> Dict[str, float]:
        """Real completion % from player_name to each teammate they targeted."""
        out = {}
        for receiver, att in self.attempted.get(player_name, {}).items():
            comp = self.matrix[player_name].get(receiver, 0)
            out[receiver] = round(comp / att * 100, 1) if att else 0.0
        return out

    def total_completed(self, player_name: str) -> int:
        return sum(self.matrix.get(player_name, {}).values())

    def total_attempted(self, player_name: str) -> int:
        return sum(self.attempted.get(player_name, {}).values())

    def avg_distance(self, p1: str, p2: str) -> float:
        """Average real distance (m) of completed passes p1 -> p2."""
        n = self.matrix[p1].get(p2, 0)
        if not n:
            return 0.0
        return round(self._dist_sum[p1].get(p2, 0.0) / n, 1)

    def player_avg_pass_distance(self, player_name: str) -> float:
        """Average real distance (m) of ALL this player's completed passes,
        across every receiver — not an estimate, computed from real
        location_x/y -> end_x/y on each completed pass event."""
        n = self._player_dist_count.get(player_name, 0)
        if not n:
            return 0.0
        return round(self._player_dist_sum.get(player_name, 0.0) / n, 1)

    def combo_rows(self) -> List[Dict]:
        """One row per passer->receiver pair with real attempted/completed/
        accuracy/distance/progressive counts — the data source for the
        'Pass Combinations' export sheet."""
        rows = []
        for p1, receivers in self.attempted.items():
            for p2, att in receivers.items():
                comp = self.matrix[p1].get(p2, 0)
                rows.append({
                    "Passer": p1,
                    "Receiver": p2,
                    "Attempted": att,
                    "Completed": comp,
                    "Accuracy %": round(comp / att * 100, 1) if att else 0.0,
                    "Avg Distance (m)": self.avg_distance(p1, p2),
                    "Progressive Passes": self._progressive[p1].get(p2, 0),
                })
        return rows

@dataclass
class ChanceMatrix:
    """
    Real creator -> shooter chance-creation matrix for one team, built
    from CHANCE_CREATED / BIG_CHANCE_CREATED events (secondary_player is
    the shooter on every such event — already in the timeline, just never
    aggregated into a pairwise table before) plus GOAL events (to show
    which combinations actually converted).
    """
    team_name: str
    counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    big_counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    xa_sum: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    goals: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    @classmethod
    def build(cls, team_name: str, timeline: List[MatchEvent]) -> "ChanceMatrix":
        cm = cls(team_name=team_name)
        for e in timeline:
            if e.team != team_name:
                continue
            if (e.event_type in (EventType.CHANCE_CREATED, EventType.BIG_CHANCE_CREATED)
                    and e.secondary_player):
                cm.counts[e.player][e.secondary_player] += 1
                cm.xa_sum[e.player][e.secondary_player] += e.xa
                if e.event_type == EventType.BIG_CHANCE_CREATED:
                    cm.big_counts[e.player][e.secondary_player] += 1
            elif e.event_type == EventType.GOAL and e.secondary_player:
                # secondary_player on GOAL is the assistant; e.player is the scorer
                cm.goals[e.secondary_player][e.player] += 1
        return cm

    def rows(self) -> List[Dict]:
        out = []
        for creator, receivers in self.counts.items():
            for shooter, n in receivers.items():
                out.append({
                    "Creator": creator,
                    "Shooter": shooter,
                    "Chances Created": n,
                    "Big Chances": self.big_counts[creator].get(shooter, 0),
                    "xA Generated": round(self.xa_sum[creator].get(shooter, 0.0), 3),
                    "Goals From Combo": self.goals.get(creator, {}).get(shooter, 0),
                })
        return out



def build_both_pass_matrices(home_team: str, away_team: str,
                              timeline: List[MatchEvent]) -> Dict[str, PassMatrix]:
    return {
        home_team: PassMatrix.build(home_team, timeline),
        away_team: PassMatrix.build(away_team, timeline),
    }


# ─────────────────────────────────────────────
# WIRING GUIDE
# ─────────────────────────────────────────────

WIRING_GUIDE = """
WIRING pass_network.py INTO exporter.py
══════════════════════════════════════════

Replace the estimated network in PLOFAExporter.plot_pass_network():

    from pass_network import PassMatrix

    pm = PassMatrix.build(team, self.result.timeline)

    # Node position: use REAL average touch location instead of the
    # fixed position_coords template:
    for name in team_stats:
        avg = pm.average_position(name)
        if avg:
            # avg is in [0,105]x[0,68] pitch coords -> convert to
            # StatsBomb [0,120]x[0,80] drawing coords used by the plot:
            node_positions[name] = (avg[0] / 105 * 120, avg[1] / 68 * 80)

    # Edge thickness: use the REAL connection strength instead of
    # _estimate_connection():
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            strength = pm.connection_strength(n1, n2)
            if strength > 0:
                lw = max(0.5, min(6.0, strength / 3.0))
                ax.plot([x1, x2], [y1, y2], color=tcolor, alpha=0.35, lw=lw)

This also unlocks two exports that weren't possible before:
    - "Top Combinations" table (pm.top_combinations()) for the Excel export
    - Per-player pass-accuracy-by-target for a real "who under-serves whom"
      diagnostic in Player Stats.
"""
