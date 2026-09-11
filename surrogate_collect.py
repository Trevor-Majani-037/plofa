"""Collect (sensor, intent, outcome) training data from REAL matches.

The goal: build a fitness surrogate that predicts *expected success* of
an on-ball decision from the vision sensor state + chosen intent.  This
surrogate is learned from real MatchEngine outcomes, so evolution can
then optimize brains against it (fast) instead of against 20-second real
matches (slow).

Why instrument `decide` instead of reading the timeline?
---------------------------------------------------------
The event timeline does NOT contain the full spatial state (teammate and
defender positions relative to the ball) at the moment of each decision,
which the 24-float sensor vector needs.  But DecisionBrain.decide()
receives exactly that geometry as arguments.  So we patch the heuristic
brain to snapshot (player, minute, sensor_vector, intent) whenever it
decides, then correlate each snapshot with the next on-ball execution
event (PASS/CARRY/SHOT) recorded for that same player in the timeline.

Outcome definition (0..~3, higher = better):
    • PASS:   completed (2.0) or incomplete (0.0), + pass_advance/30
              bonus for progressive advancement, + 1.0 if it directly
              led to a shot assist / shot on target.
    • CARRY:  retained (1.0) or lost (0.0), + distance/20 bonus, +1.0
              if the carry was progressive (broke a line).
    • SHOT:   xG added to the shot (scaled), capped.
    • TURNOVER / MISCONTROL / DISPOSSESSED: 0.0.
    • default (no event found): a neutral 0.5.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from decision_brain import DecisionBrain, PlayerIntent
from brain_sensors import extract_sensors
from brain_integration import NeuralDecisionBrain, _INTENT_BY_INDEX
from football_brain import INPUT_SIZE


# ─────────────────────────────────────────────────────────────
# GLOBAL COLLECTOR (filled by the patched decide())
# ─────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    player: str
    position: str
    minute: float
    order: int              # global call order
    sensors: np.ndarray     # 24-float
    intent: PlayerIntent


_COLLECTOR: List[DecisionRecord] = []
_CALL_ORDER = 0
_SAVED_DECIDE = None


def _snapshot_decide(player, *args, **kwargs):
    """Patch that records (sensor, intent) then calls the real brain."""
    global _CALL_ORDER
    d = _SAVED_DECIDE(player, *args, **kwargs)
    # reconstruct sensors from the real args (same positional layout as
    # DecisionBrain.decide and NeuralDecisionBrain.decide)
    x = args[0]
    y = args[1]
    teammates = args[2]
    defenders = args[3]
    position_engine = args[4]
    # positional args after the player are: x, y, teammates, defenders,
    # position_engine, team_profile, under_pressure, attacks_right,
    # game_state, minute
    under_pressure = args[7] if len(args) > 7 else kwargs.get("under_pressure", False)
    attacks_right = args[8] if len(args) > 8 else kwargs.get("attacks_right", True)
    game_state = args[9] if len(args) > 9 else kwargs.get("game_state", None)
    minute = args[10] if len(args) > 10 else kwargs.get("minute", 45.0)
    team_possession = True  # on-ball carrier is by definition in possession

    sensors = extract_sensors(
        player, x, y, teammates, defenders, position_engine,
        under_pressure, attacks_right, game_state, minute,
        team_possession=team_possession, score_diff=0,
    )
    _COLLECTOR.append(DecisionRecord(
        player=getattr(player, "name", ""),
        position=getattr(player, "position", ""),
        minute=minute,
        order=_CALL_ORDER,
        sensors=sensors,
        intent=d.intent,
    ))
    _CALL_ORDER += 1
    return d


def _install_collector():
    """Routes the live decision entry point (NeuralDecisionBrain.decide,
    which event_chain.py calls for every touch) through the snapshot
    wrapper.  The real decision is still made by whatever is bound there
    (trained neural brain, or pin the heuristic first with
    match_probe._pin_heuristic for a heuristic baseline).
    """
    global _SAVED_DECIDE
    if _SAVED_DECIDE is None:
        _SAVED_DECIDE = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_snapshot_decide)


def _restore_collector():
    global _SAVED_DECIDE
    if _SAVED_DECIDE is not None:
        NeuralDecisionBrain.decide = _SAVED_DECIDE
        _SAVED_DECIDE = None


# ─────────────────────────────────────────────────────────────
# OUTCOME -> SUCCESS SCORING
# ─────────────────────────────────────────────────────────────

def _event_success(ev: Any) -> float:
    """Return a 0..~4 success score for an execution event.

    Attacking output (shots on target, goals) is weighted more heavily
    than possession retention, since the whole point of the neural
    system is to win matches, not just keep the ball.
    """
    et = ev.event_type.name
    md = getattr(ev, "metadata", None) or {}
    outcome = getattr(ev, "outcome", None)

    if et == "PASS":
        score = 2.0 if outcome else 0.0
        advance = md.get("pass_advance", 0.0)
        score += min(advance / 25.0, 1.5)
        if md.get("is_progressive"):
            score += 0.5
        return score
    if et == "CARRY":
        score = 1.0 if outcome else 0.0
        dist = md.get("distance", 0.0)
        score += min(dist / 20.0, 1.0)
        if md.get("progressive"):
            score += 0.5
        return score
    if et in ("SHOT_OFF_TARGET",):
        xg = getattr(ev, "xg", 0.0) or 0.0
        return 0.5 + min(xg * 4.0, 1.0)  # attempted a chance, missed
    if et in ("SHOT_ON_TARGET", "SAVE", "SHOT_BLOCKED"):
        xg = getattr(ev, "xg", 0.0) or 0.0
        return 1.5 + min(xg * 4.0, 2.0)  # test the keeper, worth more
    if et == "GOAL":
        xg = getattr(ev, "xg", 0.0) or 0.0
        return 4.0 + min(xg * 4.0, 1.0)  # goals are the top reward
    if et in ("TURNOVER", "MISCONTROL", "DISPOSSESSED"):
        return 0.0
    return 0.5  # neutral default


_EXECUTION_TYPES = frozenset({
    "PASS", "CARRY", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "GOAL",
    "SAVE", "SHOT_BLOCKED", "TURNOVER", "MISCONTROL", "DISPOSSESSED",
})


def _correlate_outcomes(timeline: List[Any], records: List[DecisionRecord]) -> List[Tuple]:
    """Pair each decision record with its outcome by scanning events.

    For each record (player P, minute M, order O), find the first
    execution event by P in the globally-ordered timeline that occurs
    at or after (M).  Because a player's on-ball decision immediately
    precedes their execution event, and no other execution by P can
    interleave, the first match in order is the outcome.
    """
    # Build per-player event queue in timeline order
    by_player: Dict[str, List[Dict[str, Any]]] = {}
    for ev in timeline:
        p = getattr(ev, "player", None)
        if p is None or ev.event_type.name not in _EXECUTION_TYPES:
            continue
        by_player.setdefault(p, []).append({
            "minute": getattr(ev, "minute", 0),
            "second": getattr(ev, "second", 0),
            "ev": ev,
        })

    # sort records by (player, order) so we consume the right events
    records_sorted = sorted(records, key=lambda r: (r.player, r.order))
    # per-player event cursor
    cursor: Dict[str, int] = {}

    paired = []
    for rec in records_sorted:
        events = by_player.get(rec.player, [])
        i = cursor.get(rec.player, 0)
        # advance past events before this decision's minute
        while i < len(events) and events[i]["minute"] < rec.minute:
            i += 1
        if i < len(events):
            ev = events[i]["ev"]
            success = _event_success(ev)
            # record the resulting action type too (helps interpret)
            paired.append((rec.sensors, rec.intent, success, ev.event_type.name,
                           rec.position))
            cursor[rec.player] = i + 1
        else:
            paired.append((rec.sensors, rec.intent, 0.5, "NONE", rec.position))

    return paired


# ─────────────────────────────────────────────────────────────
# MATCH RUNNER + COLLECTION
# ─────────────────────────────────────────────────────────────

# Squad templates.  The base template mirrors the live harness (two CMs,
# no CAM/CF).  We rotate across templates during collection so every one
# of the 11 positions actually gets real-match touches (previously CAM
# and CF never appeared, so the surrogate had no data for them).
TEMPLATE_BASE = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
TEMPLATE_CAM = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CAM", "CAM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
TEMPLATE_CF = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("CF", "CF"), ("RW", "RW"),
]
# Rotation order so CAM and CF appear, and ST is on the pitch 2/3 of the time.
_COLLECTION_TEMPLATES = [TEMPLATE_BASE, TEMPLATE_CAM, TEMPLATE_CF]


def _template_name(template: List[tuple]) -> str:
    pos = [p for _, p in template]
    if "CAM" in pos:
        return "CAM"
    if "CF" in pos:
        return "CF"
    return "base"


def _build_match_components(home: str, away: str, home_style: str, away_style: str,
                            template: Optional[List[tuple]] = None,
                            home_template: Optional[List[tuple]] = None,
                            away_template: Optional[List[tuple]] = None):
    from match_engine import (
        MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
    )
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController

    if home_template is None or away_template is None:
        # legacy single-template call
        if template is None:
            template = TEMPLATE_BASE
        home_template = away_template = template
    subs = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]

    home_squad = SquadBuilder.build(home, list(home_template), list(subs))
    away_squad = SquadBuilder.build(away, list(away_template), list(subs))

    style_map = {
        "balanced": TeamStyle.BALANCED, "attacking": TeamStyle.ATTACKING,
        "defensive": TeamStyle.DEFENSIVE, "fluid_counter": TeamStyle.FLUID_COUNTER,
        "tiki_taka": TeamStyle.TIKI_TAKA, "wing_play": TeamStyle.WING_PLAY,
        "ultra_attacking": TeamStyle.ULTRA_ATTACKING,
    }

    def _tp(name, st):
        return TeamProfile(name=name, style=style_map.get(st, TeamStyle.BALANCED),
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)

    config = MatchConfig(home_team=home, away_team=away,
                         match_date=date(2026, 9, 6), matchday=3, season="26/27")
    hp = _tp(home, home_style)
    ap = _tp(away, away_style)
    sc = SubstitutionController(home_team=home, away_team=away,
                                home_subs_bench=home_squad["substitutes"],
                                away_subs_bench=away_squad["substitutes"])

    eng = MatchEngine(config, hp, ap)
    eng.set_squad(home, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(away, away_squad["starters"], away_squad["substitutes"])
    eng.set_stamina_controller(sc)
    return eng


def collect_from_matches(
    n_matches: int = 1,
    home: str = "Probe FC",
    away: str = "Rival FC",
    home_style: str = "balanced",
    away_style: str = "fluid_counter",
    seed: int = 0,
    verbose: bool = True,
    away_styles: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, PlayerIntent, float, str, str]]:
    """Run n real matches with the heuristic brain, collecting training data.

    If ``away_styles`` is given, opponent tactics are swept across it
    (rotating per match) so the surrogate isn't tuned to one opponent
    shape.  Returns list of (sensor_vector, intent, success, action_type,
    position).
    """
    global _COLLECTOR
    # clear collector + install patch
    _COLLECTOR = []
    _install_collector()

    if away_styles is None:
        away_styles = [away_style]

    all_paired = []
    try:
        for m in range(n_matches):
            random.seed(seed + m * 1000)
            opp_style = away_styles[m % len(away_styles)]
            template = _COLLECTION_TEMPLATES[m % len(_COLLECTION_TEMPLATES)]
            eng = _build_match_components(home, away, home_style, opp_style,
                                          template=template)
            result = eng.simulate()
            paired = _correlate_outcomes(result.timeline, list(_COLLECTOR))
            all_paired.extend(paired)
            _COLLECTOR.clear()
            if verbose:
                print(f"  match {m+1}: collected {len(paired)} pairs "
                      f"(opp={opp_style}, template={_template_name(template)})")
    finally:
        _restore_collector()

    return all_paired


# ─────────────────────────────────────────────────────────────
# SURROGATE: EMPIRICAL EXPECTED-SUCCESS TABLE
# ─────────────────────────────────────────────────────────────

# Coarse situational bins derived from key sensors.  This keeps the
# surrogate interpretable and robust to sparse data, while still
# capturing the dominant factors (pressure, space, depth, defense shape).
# Sensor indices (see brain_sensors):
#   4 nearest_defender_dist   9 space_ahead
#   12 is_final_third         13 is_own_half
#   14 goal_distance_norm     15 central_lane
#   16 under_pressure

@dataclass
class FitnessSurrogate:
    """Learned mapping position → (situation-bin, intent) → expected success.

    Built from empirical real-match data, now position-aware (the ST
    learns attacking values separately from the CB).  The table layout:
        self.table[position][bucket] = {intent: expected_success}
        self.table[""][bucket]        = {intent: expected_success}   # global fallback
    """
    table: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    prior: float = 0.5

    @staticmethod
    def bucket(sensors: np.ndarray) -> str:
        """Map a sensor vector to a coarse situation key."""
        pressure = int(sensors[16] > 0.5)
        final_third = int(sensors[12] > 0.5)
        own_half = int(sensors[13] > 0.5)
        space = int(sensors[9] > 0.4)   # has space ahead
        central = int(sensors[15] > 0.5)
        goal_close = int(sensors[14] < 0.5)  # within ~half of max goal distance
        return f"P{pressure}FT{final_third}OH{own_half}S{space}C{central}G{goal_close}"

    def fit(self, data: List[Tuple[np.ndarray, PlayerIntent, float, str, str]],
            min_samples: int = 3) -> "FitnessSurrogate":
        """Build per-position tables from (sensors, intent, success, action, position)."""
        # group samples by position, with "" as the global aggregate
        grouped: Dict[str, List[Tuple[np.ndarray, PlayerIntent, float]]] = {}
        for sensors, intent, success, _action, position in data:
            grouped.setdefault(position, []).append((sensors, intent, success))
            grouped.setdefault("", []).append((sensors, intent, success))  # global too

        for pos, samples in grouped.items():
            acc: Dict[str, Dict[str, float]] = {}
            cnt: Dict[str, Dict[str, int]] = {}
            for sensors, intent, success in samples:
                key = self.bucket(sensors)
                ik = intent.value if hasattr(intent, "value") else str(intent)
                acc.setdefault(key, {}).setdefault(ik, 0.0)
                cnt.setdefault(key, {}).setdefault(ik, 0)
                acc[key][ik] += success
                cnt[key][ik] += 1
            for key, intents in cnt.items():
                tot_s = sum(acc[key].get(ik, 0.0) for ik in intents)
                tot_c = sum(intents.values())
                overall = tot_s / tot_c if tot_c else self.prior
                for ik, c in intents.items():
                    if c < min_samples:
                        # shrink toward overall situation mean
                        est = (acc[key][ik] + overall * min_samples) / (c + min_samples)
                    else:
                        est = acc[key][ik] / c
                    self.table.setdefault(pos, {}).setdefault(key, {})[ik] = est
                    self.counts.setdefault(pos, {}).setdefault(key, {})[ik] = c
        return self

    def _lookup(self, key: str, ik: str, pos: str) -> Optional[float]:
        # Exact position cell first; if the specific intent has NO real
        # samples for this situation, fall through to the GLOBAL row's
        # real cell for the SAME intent (pooled across positions — real
        # data, not a fabrication).  We NEVER return the position row's
        # bucket mean across *other* intents as if it were this intent's
        # value: that silently conflates e.g. SWITCH/CARRY outcomes with
        # SHOOT, which mis-educates evolution about shooting value.
        row = self.table.get(pos, {}).get(key, {})
        if ik in row:
            return row[ik]
        grow = self.table.get("", {}).get(key, {})
        if ik in grow:
            return grow[ik]
        # Neither this position nor the global pool has a REAL sample of
        # this intent in this situation -> admit ignorance (neutral prior)
        # rather than fabricate from other intents' outcomes.
        return None

    def expected_success(self, sensors: np.ndarray, intent: PlayerIntent,
                         position: str = "") -> float:
        """Expected success for a sensor state + intent."""
        key = self.bucket(sensors)
        ik = intent.value if hasattr(intent, "value") else str(intent)
        v = self._lookup(key, ik, position)
        if v is not None:
            return v
        return self.prior

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {"prior": self.prior, "table": self.table,
                "counts": self.counts}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FitnessSurrogate":
        with open(path, "r") as f:
            data = json.load(f)
        s = cls(prior=data.get("prior", 0.5))
        s.table = data.get("table", {})
        s.counts = data.get("counts", {})
        return s

    def report(self) -> Dict[str, Any]:
        positions = list(self.table.keys())
        n_bins = sum(len(v) for v in self.table.values())
        n_cells = sum(len(v) for v in self.table.values() for v2 in v.values())
        return {"positions": positions, "bins": n_bins, "filled_cells": n_cells}


def build_surrogate(
    n_matches: int = 5,
    seed: int = 0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    away_styles: Optional[List[str]] = None,
) -> FitnessSurrogate:
    """Collect real-match data and fit a FitnessSurrogate.

    ``away_styles`` sweeps opponent tactics so the surrogate generalizes
    beyond a single opponent shape.
    """
    if verbose:
        styles_txt = (f" (opponents: {away_styles})" if away_styles else "")
        print(f"Collecting {n_matches} real matches (heuristic brain){styles_txt}...")
        print(f"NOTE: each match takes ~15-20s.")
    data = collect_from_matches(n_matches=n_matches, seed=seed, verbose=verbose,
                                away_styles=away_styles)
    if verbose:
        print(f"Total decision->outcome samples: {len(data)}")
    surrogate = FitnessSurrogate().fit(data)
    if verbose:
        print(f"Surrogate: {surrogate.report()}")
    if save_path:
        surrogate.save(save_path)
        if verbose:
            print(f"Saved surrogate to {save_path}")
    return surrogate


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a fitness surrogate from real matches.")
    p.add_argument("--matches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="brains/surrogate.json")
    p.add_argument("--styles", type=str, default=None,
                   help="Comma-separated opponent styles to sweep "
                        "(e.g. fluid_counter,attacking,tiki_taka).")
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()] if args.styles else None
    surrogate = build_surrogate(n_matches=args.matches, seed=args.seed,
                                save_path=args.out, away_styles=styles)
    print(json.dumps(surrogate.report()))
