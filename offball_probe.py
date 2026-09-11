"""Off-ball conscience — collection probe (NO engine writes).

Patches the LIVE off-ball decision surface: MatchEngine._offball_move_player
(match_engine.py:1904).  The discrete run modes (byline/cut/box,
drop/carry/orbit...) in position_engine are NOT wired into matches — they are
simulator/test-only.  The one stochastic off-ball decision a real match makes
is the CHASE/PRESS COMMIT:

    chasing = (not has_ball) and ball_dist < _CHASE_TRIGGER   (12.5 m)
    if cst['allow'] == 0.0 and cst['p'] <= 0.0:
        cst['allow'] = 1.0 if random.random() < _PRESS_PROB[pos] else -1.0
                                                      (match_engine.py:1991-96)

This probe wraps that method observation-only (it reads the allow transition
0 -> +/-1 after the fact; it consumes ZERO RNG and changes NOTHING).  For
every press decision it records (player, position, decision=[press|hold],
minute, geometry, 24-d off-ball sensors) from the PRE-tick spatial state.

Each decision is then correlated with the next DEFENSIVE outcome in the
timeline (first relevant event inside rec.minute+2):

    D team recovers (RECOVERY/TACKLE_WON/INTERCEPTION/CLEARANCE/BLOCK)  -> 1.0
    opponent TURNOVER/MISCONTROL/DISPOSSESSED                           -> 0.9
    opponent SHOT_OFF_TARGET                                            -> 0.6
    opponent SHOT_ON_TARGET/SAVE/BLOCKED                                -> 0.4
    opponent GOAL                                                       -> 0.0
    nothing in window                                                   -> 0.5 (neutral)

This is the defensive mirror of surrogate_collect._event_success: a press
"wins" by recovering the ball or denying a chance, and loses if the opponent
scores while the presser is committed.  reward = episode_score (no reception
term: a presser is off the ball by definition; recovery IS the payoff).

Run:  python offball_probe.py [--matches 1] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

import match_engine as match_engine_module
import match_probe
from brain_sensors import extract_offball_sensors


# ─────────────────────────────────────────────────────────────
# COLLECTOR STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class PressRecord:
    player: str
    position: str
    minute: float
    clock: float               # true match clock seconds at decision
    order: int
    team: str
    decision: str            # "press" | "hold"
    runner_x: float
    runner_y: float
    ball_x: float
    ball_y: float
    ball_dist: float
    attacked_ok: bool        # allow>0 => press burst committed
    sensors: np.ndarray = field(repr=False)


_COLLECTOR: List[PressRecord] = []
_ORDER = 0
_ORIG_OFFBALL = None
_VIS_CACHE: Dict[int, Dict[str, Dict[str, Any]]] = {}


# ─────────────────────────────────────────────────────────────
# LIGHTWEIGHT PLAYER VIEWS (sensors need .name/.position/.dna)
# ─────────────────────────────────────────────────────────────

class _View:
    """Minimal player-like object carrying the fields sensors read."""

    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position
        self.dna = None  # DNA slots default to 55/100 in the sensor layer


def _build_views(engine: Any) -> Dict[str, Dict[str, Any]]:
    """Cache one team->name->view map per engine instance.

    Prefers the real active player objects (which carry DNA); falls back to
    a name/position view if the roster lists a name with no live object.
    """
    key = id(engine)
    if key in _VIS_CACHE:
        return _VIS_CACHE[key]
    maps: Dict[str, Dict[str, Any]] = {}
    for team, names in engine.position_engine.team_rosters.items():
        real = {}
        for p in engine.active_players.get(team, []):
            real[getattr(p, "name", None)] = p
        maps[team] = {}
        for n in names:
            st = engine.position_engine.states.get(n)
            pos = getattr(st, "position", "") if st is not None else ""
            maps[team][n] = real.get(n) or _View(n, pos)
    _VIS_CACHE[key] = maps
    return maps


# ─────────────────────────────────────────────────────────────
# LIVE-SURFACE PATCH
# ─────────────────────────────────────────────────────────────

def _wrap_offball(self, pname: str, team: str, ball_x: float, ball_y: float,
                  has_ball: bool, danger_t: float, cross_team: str, DT: float):
    """Wrap _offball_move_player: detect a fresh chase-allow decision.

    The allow field transitions 0 -> +/-1 exactly when the press Bernoulli
    fires this tick (inside `if chasing: if cst['allow']==0.0 ...`).  We read
    it BEFORE and AFTER the original to detect that edge without touching the
    RNG stream.
    """
    st = self.position_engine.states.get(pname)
    rx, ry = (st.current_x, st.current_y) if st is not None else (ball_x, ball_y)

    cst_pre = self._chase_state.get(pname)
    allow_pre = cst_pre["allow"] if cst_pre else 0.0

    result = _ORIG_OFFBALL(self, pname, team, ball_x, ball_y, has_ball,
                           danger_t, cross_team, DT)

    cst_post = self._chase_state.get(pname)
    allow_post = cst_post["allow"] if cst_post else 0.0
    if allow_pre == 0.0 and allow_post != 0.0 and not has_ball:
        global _ORDER
        views = _build_views(self)
        teammates = [views[team][n] for n in
                     self.position_engine.team_rosters.get(team, [])
                     if n in views[team]]
        defenders = []
        for ot, names in self.position_engine.team_rosters.items():
            if ot == team:
                continue
            defenders.extend(views[ot][n] for n in names if n in views[ot])
        attacks_right = self.position_engine.team_attacks_right.get(team, True)
        runner_view = _View(pname, getattr(st, "position", "") if st else "")
        sensors = extract_offball_sensors(
            runner_view, rx, ry, ball_x, ball_y,
            teammates=teammates, defenders=defenders,
            position_engine=self.position_engine,
            attacks_right=attacks_right,
            game_state=None,
            minute=float(self.state.minute),
            score_diff=self.state.home_goals - self.state.away_goals,
        )
        dist = ((ball_x - rx) ** 2 + (ball_y - ry) ** 2) ** 0.5
        _clock = self.state.match_clock_s
        _COLLECTOR.append(PressRecord(
            player=pname,
            position=getattr(st, "position", "") if st else "",
            minute=float(self.state.minute) + float(_clock % 60.0) / 60.0,
            clock=_clock,
            order=_ORDER, team=team,
            decision="press" if allow_post > 0 else "hold",
            runner_x=rx, runner_y=ry, ball_x=ball_x, ball_y=ball_y,
            ball_dist=dist, attacked_ok=allow_post > 0,
            sensors=sensors,
        ))
        _ORDER += 1
    return result


def _install() -> None:
    global _ORIG_OFFBALL
    _ORIG_OFFBALL = match_engine_module.MatchEngine._offball_move_player
    match_engine_module.MatchEngine._offball_move_player = _wrap_offball


def _restore() -> None:
    global _ORIG_OFFBALL
    if _ORIG_OFFBALL is not None:
        match_engine_module.MatchEngine._offball_move_player = _ORIG_OFFBALL
        _ORIG_OFFBALL = None
    _VIS_CACHE.clear()


# ─────────────────────────────────────────────────────────────
# CORRELATION (defensive episode outcome)
# ─────────────────────────────────────────────────────────────

_RECOVERY_TYPES = frozenset({
    "RECOVERY", "TACKLE_WON", "INTERCEPTION", "CLEARANCE", "BLOCK",
})
_OPP_LOSS_TYPES = frozenset({
    "TURNOVER", "MISCONTROL", "DISPOSSESSED",
})
_OPP_SHOT_OFF = frozenset({"SHOT_OFF_TARGET"})
_OPP_SHOT_ON = frozenset({"SHOT_ON_TARGET", "SAVE", "SHOT_BLOCKED"})
_OPP_GOAL = frozenset({"GOAL", "PENALTY_SCORED"})


def _score(ev: Any) -> Optional[float]:
    name = getattr(ev, "event_type", None)
    name = name.name if hasattr(name, "name") else str(name)
    if name in _RECOVERY_TYPES:
        return 1.0
    if name in _OPP_LOSS_TYPES:
        return 0.9
    if name in _OPP_SHOT_OFF:
        return 0.6
    if name in _OPP_SHOT_ON:
        return 0.4
    if name in _OPP_GOAL:
        return 0.0
    return None


def _correlate(
    records: List[PressRecord],
    timeline: List[Any],
    chronology: Optional[Any] = None,
    window_s: float = 20.0,
    max_dist: float = 25.0,
) -> List[Dict[str, Any]]:
    """Pair each press decision with the next DECISION-LOCAL defensive outcome.

    Unlike the old 2-minute team-window sweep (which saturated at ~0.86 for
    every decision because 87% of windows contained a recovery), this uses the
    TRUE match-clock seconds from the chronograph (placeholder timeline
    seconds are unreliable) and only credits outcomes that happen quickly
    AND near the presser — the only signal a single player's press can
    realistically influence:

      * time window: [rec.clock, rec.clock + window_s] (default 20 s)
      * distance:   outcome event within max_dist m (default 25 m) of the
                    presser's position at decision time.

    Events far in space or time are ignored, so a goal forced on the other
    wing no longer credits a press that had nothing to do with it.  Rows with
    no nearby outcome keep the neutral 0.5.
    """
    clock_by_id: Dict[int, float] = {}
    if chronology is not None:
        t_idx = {id(e): i for i, e in enumerate(timeline)}
        for te in getattr(chronology, "events", ()):
            idx = getattr(te, "source_event_index", -1)
            if 0 <= idx < len(timeline):
                clock_by_id[id(timeline[idx])] = float(getattr(te, "match_clock_s", 0.0))

    def _event_clock(ev: Any) -> float:
        c = clock_by_id.get(id(ev))
        if c is not None:
            return c
        return float(getattr(ev, "minute", 99.0)) * 60.0 + \
            float(getattr(ev, "second", 0))

    by_team: Dict[str, List[Any]] = collections.defaultdict(list)
    for ev in timeline:
        by_team[getattr(ev, "team", "")].append(ev)
    for team in by_team:
        by_team[team].sort(key=_event_clock)

    rows = []
    for rec in records:
        outcome_score = 0.5
        seen_in_window = False
        for ev in by_team.get(rec.team, []):
            ev_clock = _event_clock(ev)
            if ev_clock < rec.clock:
                continue
            if ev_clock > rec.clock + window_s:
                break
            s = _score(ev)
            if s is None:
                continue
            ev_x = float(getattr(ev, "location_x", 50.0))
            ev_y = float(getattr(ev, "location_y", 34.0))
            d = ((ev_x - rec.runner_x) ** 2 + (ev_y - rec.runner_y) ** 2) ** 0.5
            if d > max_dist:
                continue
            outcome_score = s
            seen_in_window = True
            break
        rows.append({
            "player": rec.player,
            "position": rec.position,
            "minute": rec.minute,
            "decision": rec.decision,
            "ball_dist": rec.ball_dist,
            "episode_score": outcome_score,
            "seen": seen_in_window,
            "sensors": rec.sensors.tolist(),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# OFF-BALL SURROGATE (learnt expected-success table)
# ─────────────────────────────────────────────────────────────

@dataclass
class OffBallSurrogate:
    """position → (situation bucket) → {press, hold} → expected success.

    Same bucket scheme as the on-ball FitnessSurrogate (sensor indices 16,
    12, 13, 9, 15, 14 map to pressure / final third / own half / space /
    central / goal-close), but the per-bucket value is the expected
    defensive-episode score of COMMITTING press vs declining (hold).
    """
    table: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    prior: float = 0.5

    @staticmethod
    def bucket(sensors: np.ndarray) -> str:
        pressure = int(sensors[16] > 0.5)
        final_third = int(sensors[12] > 0.5)
        own_half = int(sensors[13] > 0.5)
        space = int(sensors[9] > 0.4)
        central = int(sensors[15] > 0.5)
        goal_close = int(sensors[14] < 0.5)
        return f"P{pressure}FT{final_third}OH{own_half}S{space}C{central}G{goal_close}"

    def fit(self, rows: List[Dict[str, Any]], min_samples: int = 3) -> "OffBallSurrogate":
        grouped: Dict[str, List[Tuple[np.ndarray, str, float]]] = {}
        for r in rows:
            sensors = np.asarray(r["sensors"], dtype=np.float64)
            pos = r.get("position", "")
            grouped.setdefault(pos, []).append((sensors, r["decision"], r["episode_score"]))
            grouped.setdefault("", []).append((sensors, r["decision"], r["episode_score"]))

        for pos, samples in grouped.items():
            acc: Dict[str, Dict[str, float]] = {}
            cnt: Dict[str, Dict[str, int]] = {}
            for sensors, decision, score in samples:
                key = self.bucket(sensors)
                ik = decision
                acc.setdefault(key, {}).setdefault(ik, 0.0)
                cnt.setdefault(key, {}).setdefault(ik, 0)
                acc[key][ik] += score
                cnt[key][ik] += 1
            for key, decs in cnt.items():
                tot_s = sum(acc[key].get(ik, 0.0) for ik in decs)
                tot_c = sum(decs.values())
                overall = tot_s / tot_c if tot_c else self.prior
                for ik, c in decs.items():
                    if c < min_samples:
                        est = (acc[key][ik] + overall * min_samples) / (c + min_samples)
                    else:
                        est = acc[key][ik] / c
                    self.table.setdefault(pos, {}).setdefault(key, {})[ik] = est
                    self.counts.setdefault(pos, {}).setdefault(key, {})[ik] = c
        return self

    def _lookup(self, key: str, decision: str, pos: str) -> Optional[float]:
        for p in (pos, ""):
            row = self.table.get(p, {}).get(key, {})
            if decision in row:
                return row[decision]
            if row:
                return sum(row.values()) / len(row)
        return None

    def expected_success(self, sensors: np.ndarray, decision: str,
                         position: str = "") -> float:
        key = self.bucket(sensors)
        v = self._lookup(key, decision, position)
        return v if v is not None else self.prior

    def support(self, sensors: np.ndarray, decision: str,
                position: str = "") -> int:
        """Number of real-match samples behind (position, bucket, decision).

        Evidence gate for the conscience: a cell only permits overriding the
        heuristic when BOTH decisions have >= min_samples support here, so the
        surrogate does not reward press simply because the heuristic pressed
        (off-policy bias).  Global '' fallback mirrors _lookup.
        """
        key = self.bucket(sensors)
        for p in (position, ""):
            row = self.counts.get(p, {}).get(key, {})
            if decision in row:
                return int(row[decision])
            return max(row.values(), default=0)
        return 0

    def is_evidence_cell(self, sensors: np.ndarray, position: str = "",
                         min_both: int = 5) -> bool:
        key = self.bucket(sensors)
        for p in (position, ""):
            row = self.counts.get(p, {}).get(key, {})
            if row and len(row) >= 2:
                return min(row.values()) >= min_both
        return False

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"prior": self.prior, "table": self.table,
                       "counts": self.counts}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OffBallSurrogate":
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


# ─────────────────────────────────────────────────────────────
# COLLECTOR (sweep + save samples / surrogate)
# ─────────────────────────────────────────────────────────────

def collect_press_samples(
    n_matches: int = 6,
    seed: int = 0,
    away_styles: Optional[List[str]] = None,
    home_style: str = "balanced",
    home_team: str = "Probe FC",
    away_team: str = "Rival FC",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run n real matches collecting press decisions (Probe FC team frame).

    Opponent style sweeps across ``away_styles`` (rotating per match) so the
    surrogate is not tuned to one opposition shape.  Returns correlated rows
    (sensors + decision + defensive episode score).  The probe is
    RNG-neutral and writes nothing to the engine.
    """
    global _ORDER
    _ORDER = 0
    if away_styles is None:
        away_styles = ["fluid_counter"]

    all_rows: List[Dict[str, Any]] = []
    for m in range(n_matches):
        seed_i = seed + m * 1000
        random.seed(seed_i)
        opp_style = away_styles[m % len(away_styles)]
        t0 = time.time()

        home_squad, away_squad = match_probe._build_squads(home_team, away_team)
        config = match_probe.MatchConfig(
            home_team=home_team, away_team=away_team,
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile(home_team, home_style)
        ap_ = match_probe._team_profile(away_team, opp_style)
        eng = match_probe.MatchEngine(config, hp, ap_)
        eng.set_squad(home_team, home_squad["starters"], home_squad["substitutes"])
        eng.set_squad(away_team, away_squad["starters"], away_squad["substitutes"])

        match_probe._pin_heuristic()
        try:
            result = eng.simulate()
        finally:
            match_probe._restore_neural()

        recs = [r for r in _COLLECTOR if r.team == home_team]
        _COLLECTOR.clear()
        rows = _correlate(recs, result.timeline, chronology=result.chronology)
        all_rows.extend(rows)
        if verbose:
            print(f"  match {m+1}: {result.score_str}  press decisions={len(rows)}  "
                  f"(opp={opp_style}, {time.time()-t0:.1f}s)")
    return all_rows


def main():
    ap = argparse.ArgumentParser(description="Off-ball press-decision collection probe.")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--away-styles", type=str, default=None,
                    help="Comma-separated opponent styles to sweep "
                         "(default: fluid_counter).")
    ap.add_argument("--out-samples", type=str, default=None,
                    help="JSON path to dump raw sampled rows.")
    ap.add_argument("--out-surrogate", type=str, default="brains_offball/surrogate.json",
                    help="JSON path to write the learnt OffBallSurrogate.")
    args = ap.parse_args()

    styles = [s.strip() for s in args.away_styles.split(",") if s.strip()] \
        if args.away_styles else None

    _install()
    try:
        print(f"Collecting {args.matches} real matches (heuristic brain pinned; "
              f"opponents: {styles or ['fluid_counter']})...")
        all_rows = collect_press_samples(
            n_matches=args.matches, seed=args.seed, away_styles=styles,
        )
    finally:
        _restore()

    print(f"\ntotal press decisions: {len(all_rows)}")
    if not all_rows:
        return

    print("\n--- decision distribution ---")
    for dec, n in collections.Counter(r["decision"] for r in all_rows).most_common():
        print(f"  {dec:<6} {n}")

    print("\n--- decision by position ---")
    by_pos_dec: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    for r in all_rows:
        by_pos_dec[(r["position"], r["decision"])] += 1
    for (pos, dec), n in sorted(by_pos_dec.items(), key=lambda kv: -kv[1]):
        print(f"  {pos:<4} {dec:<6} n={n}")

    print("\n--- defensive outcome by decision (0..1, higher = better) ---")
    by_dec: Dict[str, List[float]] = collections.defaultdict(list)
    for r in all_rows:
        by_dec[r["decision"]].append(r["episode_score"])
    for dec, vals in sorted(by_dec.items(), key=lambda kv: -len(kv[1])):
        print(f"  {dec:<6} n={len(vals):>4}  mean={statistics.mean(vals):.3f}  "
              f"recovery-rate={(sum(1 for v in vals if v >= 0.9) / len(vals)):.2f}")

    print("\n--- ball distance at decision by decision ---")
    dists: Dict[str, List[float]] = collections.defaultdict(list)
    for r in all_rows:
        dists[r["decision"]].append(r["ball_dist"])
    for dec, vals in sorted(dists.items(), key=lambda kv: -len(kv[1])):
        print(f"  {dec:<6} n={len(vals):>4}  mean={statistics.mean(vals):.1f}m")

    surrogate = OffBallSurrogate().fit(all_rows)
    print(f"\n--- off-ball surrogate: {surrogate.report()} ---")
    if args.out_surrogate:
        surrogate.save(args.out_surrogate)
        print(f"surrogate written to {args.out_surrogate}")
    if args.out_samples:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_samples)) or ".", exist_ok=True)
        with open(args.out_samples, "w") as f:
            json.dump(all_rows, f)
        print(f"samples written to {args.out_samples}")


if __name__ == "__main__":
    main()