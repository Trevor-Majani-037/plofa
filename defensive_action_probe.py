"""Defensive-action collection probe (NO engine writes beyond the hook).

Wraps the LIVE defensive-action selector ``match_engine._def_action_choice``
(the single decision point behind open-play contests, direct clearances and
deep recoveries).  Whenever the engine picks tackle / interception /
clearance / block, this wrapper records:

  * the 24-d DEFENSIVE-state vector (extract_defensive_sensors),
  * the action actually taken (the original selector — heuristic table by
    default, a loaded DefensiveActionBrain when engaged — still decides; this
    probe only OBSERVES),
  * and, after the match, the DECISION-LOCAL outcome via the off-ball scoring
    (recovery/interception/tackle = 1.0, opponent turnover 0.9, shot off 0.6,
    shot on/save/block 0.4, goal against 0.0, nothing 0.5).

The ``DefensiveActionSurrogate`` learns, per state bucket, the expected
success of each action type — the fitness signal for DefensiveActionBrain.

Run:  python defensive_action_probe.py [--matches 6] [--seed 21]
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
from football_brain import extract_defensive_sensors as _extract_def_sensors
from offball_probe import _score

DEFENSIVE_ACTIONS = ("tackle", "interception", "clearance", "block")


# ─────────────────────────────────────────────────────────────
# COLLECTOR STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class DefRecord:
    team: str
    clock: float
    minute: float
    action: str
    style: str
    ball_x: float
    ball_y: float
    ctx_x: float
    ctx_y: float
    site: str              # "contest" | "direct_clearance" | "recovery"
    sensors: np.ndarray = field(repr=False)


_COLLECTOR: List[DefRecord] = []
_ORIG_CHOICE = None
_STYLE = ""
_INTERVENE: Optional[str] = None      # force a single action per match (counterfactual)
_INTERVENE_KEEP: Optional[str] = None # the action kept for the RNG-consuming selection


def _wrap_def_action_choice(
    engine, defending_team, attacking_team,
    danger_level, ctx_x, ctx_y,
    ball_aerial, opponent_distance, attacks_right,
    ball_x, ball_y, press_occurred=False, fallback=None, site="contest",
):
    global _INTERVENE_KEEP
    # Sensors BEFORE the original decides (same formula as the live hook).
    try:
        sens = _extract_def_sensors(
            engine, defending_team, attacking_team,
            ball_x, ball_y,
            danger_level=danger_level, ball_aerial=ball_aerial,
            contest_x=ctx_x, contest_y=ctx_y,
            opponent_distance=opponent_distance or 0.0,
            press_occurred=press_occurred,
        )
        sens_ok = sens is not None and sens.shape == (24,)
    except Exception:
        sens_ok = False

    if _INTERVENE is not None:
        # Counterfactual: force the target action.  Keep the ORIGINAL's RNG
        # consumption identical by still calling it, then OVERRIDE the pick.
        forced = str(_INTERVENE)
        original_pick = _ORIG_CHOICE(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y,
            ball_aerial, opponent_distance, attacks_right,
            ball_x, ball_y, press_occurred, fallback, site,
        )
        action = forced
        _INTERVENE_KEEP = original_pick
    else:
        action = _ORIG_CHOICE(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y,
            ball_aerial, opponent_distance, attacks_right,
            ball_x, ball_y, press_occurred, fallback, site,
        )

    if sens_ok:
        _clock = engine.state.match_clock_s
        _COLLECTOR.append(DefRecord(
            team=defending_team,
            clock=_clock,
            minute=float(engine.state.minute) + float(_clock % 60.0) / 60.0,
            action=action,
            style=_STYLE,
            ball_x=ball_x, ball_y=ball_y,
            ctx_x=ctx_x if ctx_x is not None else ball_x,
            ctx_y=ctx_y if ctx_y is not None else ball_y,
            site=site,
            sensors=sens,
        ))
    return action


def _install(style: str = "", intervene: Optional[str] = None) -> None:
    global _ORIG_CHOICE, _STYLE, _INTERVENE, _INTERVENE_KEEP
    _STYLE = style
    _INTERVENE = intervene
    _INTERVENE_KEEP = None
    # make sure no defensive brain is engaged during collection so we observe
    # the HEURISTIC's choices (the baseline the surrogate must beat)
    match_engine_module.set_defensive_action_brain(None)
    match_engine_module._DEF_ACTION_LOADED = True   # don't auto-load a brain
    if _ORIG_CHOICE is None:
        _ORIG_CHOICE = match_engine_module._def_action_choice
        match_engine_module._def_action_choice = _wrap_def_action_choice


def _restore() -> None:
    global _ORIG_CHOICE, _STYLE, _INTERVENE, _INTERVENE_KEEP
    if _ORIG_CHOICE is not None:
        match_engine_module._def_action_choice = _ORIG_CHOICE
        _ORIG_CHOICE = None
    _STYLE = ""
    _INTERVENE = None
    _INTERVENE_KEEP = None
    match_engine_module._DEF_ACTION_LOADED = False


# ─────────────────────────────────────────────────────────────
# DECISION-LOCAL OUTCOME
# ─────────────────────────────────────────────────────────────

def _correlate_def(records: List[DefRecord], timeline: List[Any],
                   chronology: Optional[Any] = None,
                   window_s: float = 12.0, max_dist: float = 20.0) -> List[Dict[str, Any]]:
    """Pair each defensive action with its own defensive outcome.

    The chosen action IS the next event, so the window can be short (12 s)
    and the proximity centred on the contest point.  Scoring mirrors the
    off-ball probe: winning the ball / blocking is positive; conceding a shot
    or goal is negative; nothing nearby is a neutral 0.5.
    """
    clock_by_id: Dict[int, float] = {}
    if chronology is not None:
        for te in getattr(chronology, "events", ()):
            idx = getattr(te, "source_event_index", -1)
            if 0 <= idx < len(timeline):
                clock_by_id[id(timeline[idx])] = float(getattr(te, "match_clock_s", 0.0))

    def _ev_clock(ev: Any) -> float:
        c = clock_by_id.get(id(ev))
        if c is not None:
            return c
        return float(getattr(ev, "minute", 99.0)) * 60.0 + float(getattr(ev, "second", 0))

    by_team: Dict[str, List[Any]] = collections.defaultdict(list)
    for ev in timeline:
        by_team[getattr(ev, "team", "")].append(ev)
    for t in by_team:
        by_team[t].sort(key=_ev_clock)

    rows = []
    for rec in records:
        outcome = 0.5
        seen = False
        for ev in by_team.get(rec.team, []):
            t = _ev_clock(ev)
            if t < rec.clock - 0.01:
                continue
            if t > rec.clock + window_s:
                break
            s = _score(ev)
            if s is None:
                continue
            dx = float(getattr(ev, "location_x", rec.ctx_x)) - rec.ctx_x
            dy = float(getattr(ev, "location_y", rec.ctx_y)) - rec.ctx_y
            if (dx * dx + dy * dy) ** 0.5 > max_dist:
                continue
            outcome = s
            seen = True
            break
        rows.append({
            "team": rec.team, "clock": rec.clock, "minute": rec.minute,
            "action": rec.action, "style": rec.style,
            "ball_x": rec.ball_x, "ball_y": rec.ball_y,
            "ctx_x": rec.ctx_x, "ctx_y": rec.ctx_y, "site": rec.site,
            "episode_score": outcome, "seen": seen,
            "sensors": rec.sensors.tolist(),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# DEFENSIVE-ACTION SURROGATE
# ─────────────────────────────────────────────────────────────

class DefensiveActionSurrogate:
    """style -> defensive bucket -> {tackle, interception, clearance, block}
    -> expected success.

    Bucket uses the booleans that dominate a contest choice: ball zone vs our
    goal (own-third danger), danger level, aerial/ground, our vs opp density
    at the ball, clearance feasibility, opp bodies near our goal-mouth.
    Shrinkage (min_samples) keeps sparse cells statistically honest.
    """
    def __init__(self):
        self.table: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.prior = 0.5

    @staticmethod
    def bucket(sensors: np.ndarray) -> str:
        proj = float(sensors[2])                      # 0 our goal .. 1 opp goal
        zone = 2 if proj < 0.33 else (1 if proj < 0.66 else 0)  # 2=own third
        danger = int(sensors[3] > 0.5)
        aerial = int(sensors[11] > 0.5)
        ours = int(sensors[8] > 0.25)
        opp = int(sensors[9] > 0.25)
        feas = int(sensors[19] > 0.5)
        opmen = int(sensors[16] > 0.25)
        return f"Z{zone}D{danger}A{aerial}O{ours}E{opp}F{feas}M{opmen}"

    def fit(self, rows: List[Dict[str, Any]], min_samples: int = 3) -> "DefensiveActionSurrogate":
        grouped: Dict[str, List[Tuple[np.ndarray, str, float]]] = {}
        for r in rows:
            sensors = np.asarray(r["sensors"], dtype=np.float64)
            st = r.get("style", "")
            grouped.setdefault(st, []).append((sensors, r["action"], r["episode_score"]))
            grouped.setdefault("", []).append((sensors, r["action"], r["episode_score"]))

        for style, samples in grouped.items():
            acc: Dict[str, Dict[str, float]] = {}
            cnt: Dict[str, Dict[str, int]] = {}
            for sensors, action, score in samples:
                key = self.bucket(sensors)
                acc.setdefault(key, {}).setdefault(action, 0.0)
                cnt.setdefault(key, {}).setdefault(action, 0)
                acc[key][action] += score
                cnt[key][action] += 1
            for key, acts in cnt.items():
                tot_s = sum(acc[key].get(a, 0.0) for a in acts)
                tot_c = sum(acts.values())
                overall = tot_s / tot_c if tot_c else self.prior
                for a, c in acts.items():
                    if c < min_samples:
                        est = (acc[key][a] + overall * min_samples) / (c + min_samples)
                    else:
                        est = acc[key][a] / c
                    self.table.setdefault(style, {}).setdefault(key, {})[a] = est
                    self.counts.setdefault(style, {}).setdefault(key, {})[a] = c
        return self

    def _lookup(self, key: str, action: str, style: str) -> Optional[float]:
        for s in (style, ""):
            row = self.table.get(s, {}).get(key, {})
            if action in row:
                return row[action]
            if row:
                return sum(row.values()) / len(row)
        return None

    def expected_success(self, sensors: np.ndarray, action: str,
                         style: str = "") -> float:
        key = self.bucket(sensors)
        v = self._lookup(key, action, style)
        return v if v is not None else self.prior

    def support(self, sensors: np.ndarray, action: str, style: str = "") -> int:
        key = self.bucket(sensors)
        for s in (style, ""):
            row = self.counts.get(s, {}).get(key, {})
            if action in row:
                return int(row[action])
            return max(row.values(), default=0)
        return 0

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"prior": self.prior, "table": self.table,
                       "counts": self.counts}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DefensiveActionSurrogate":
        with open(path, "r") as f:
            data = json.load(f)
        s = cls()
        s.prior = data.get("prior", 0.5)
        s.table = data.get("table", {})
        s.counts = data.get("counts", {})
        return s

    def report(self) -> Dict[str, Any]:
        styles = list(self.table.keys())
        n_bins = sum(len(v) for v in self.table.values())
        n_cells = sum(len(v) for v in self.table.values() for v2 in v.values())
        return {"styles": styles, "bins": n_bins, "filled_cells": n_cells}


# ─────────────────────────────────────────────────────────────
# COLLECTOR
# ─────────────────────────────────────────────────────────────

def collect_defensive_samples(
    n_matches: int = 6,
    seed: int = 0,
    away_styles: Optional[List[str]] = None,
    home_style: str = "balanced",
    home_team: str = "Probe FC",
    away_team: str = "Rival FC",
    verbose: bool = True,
    intervene_actions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if away_styles is None:
        away_styles = ["fluid_counter"]

    all_rows: List[Dict[str, Any]] = []
    for m in range(n_matches):
        seed_i = seed + m * 1000
        random.seed(seed_i)
        opp_style = away_styles[m % len(away_styles)]
        target = None
        if intervene_actions:
            target = intervene_actions[m % len(intervene_actions)]
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
        from squad_manager import SubstitutionController
        eng.set_stamina_controller(SubstitutionController(
            home_team=home_team, away_team=away_team,
            home_subs_bench=home_squad["substitutes"],
            away_subs_bench=away_squad["substitutes"]))

        _install(style=opp_style, intervene=target)
        try:
            match_probe._pin_heuristic()
            try:
                result = eng.simulate()
            finally:
                match_probe._restore_neural()
        finally:
            _restore()

        recs = list(_COLLECTOR)
        _COLLECTOR.clear()
        rows = _correlate_def(recs, result.timeline, chronology=result.chronology)
        all_rows.extend(rows)
        if verbose:
            tag = f"force={target}" if target is not None else "heuristic"
            print(f"  match {m+1}: {result.score_str}  def-actions={len(rows)}  "
                  f"(opp={opp_style}, {tag}, {time.time()-t0:.1f}s)")
    return all_rows


def main():
    ap = argparse.ArgumentParser(description="Defensive-action collector.")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--away-styles", type=str, default=None,
                    help="Comma-separated opponent styles to sweep.")
    ap.add_argument("--intervene", type=str, default=None,
                    help="Comma-separated ACTIONS to force per match "
                         "(tackle,interception,clearance,block). Counterfactual "
                         "experiments generate the low-observation actions the "
                         "heuristic under-uses. Omit for plain observation.")
    ap.add_argument("--out-samples", type=str, default="brains_def/samples.json")
    ap.add_argument("--out-surrogate", type=str, default="brains_def/surrogate.json")
    args = ap.parse_args()

    styles = [s.strip() for s in args.away_styles.split(",") if s.strip()] \
        if args.away_styles else None
    targets = None
    if args.intervene:
        targets = [s.strip() for s in args.intervene.split(",") if s.strip()] or None

    print(f"Collecting {args.matches} real matches (heuristic on-ball; "
          f"opponents: {styles or ['fluid_counter']}); "
          f"actions: {targets or 'observed heuristic'}")
    all_rows = collect_defensive_samples(
        n_matches=args.matches, seed=args.seed, away_styles=styles,
        intervene_actions=targets,
    )

    print(f"\ntotal defensive-action samples: {len(all_rows)}")
    if not all_rows:
        return

    print("\n--- action distribution ---")
    for a, n in collections.Counter(r["action"] for r in all_rows).most_common():
        print(f"  {a:<14} n={n}")

    print("\n--- outcome by action (0..1, higher = better) ---")
    by_a: Dict[str, List[float]] = collections.defaultdict(list)
    for r in all_rows:
        by_a[r["action"]].append(r["episode_score"])
    for a, vals in sorted(by_a.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a:<14} n={len(vals):>4}  mean={statistics.mean(vals):.3f}")

    print("\n--- outcome by site ---")
    by_s: Dict[str, List[float]] = collections.defaultdict(list)
    for r in all_rows:
        by_s[r["site"]].append(r["episode_score"])
    for s, vals in sorted(by_s.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s:<14} n={len(vals):>4}  mean={statistics.mean(vals):.3f}")

    surrogate = DefensiveActionSurrogate().fit(all_rows)
    print(f"\n--- defensive surrogate: {surrogate.report()} ---")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_samples)) or ".", exist_ok=True)
    with open(args.out_samples, "w") as f:
        json.dump(all_rows, f)
    surrogate.save(args.out_surrogate)
    print(f"samples: {args.out_samples}\nsurrogate: {args.out_surrogate}")


if __name__ == "__main__":
    main()