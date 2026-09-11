"""Team-level off-ball press engagement — collection probe (NO engine writes).

The individual-conscience probe proved pressing is a UNIT act: one player's
press/hold barely moved the team's defensive outcome (|delta| ~ 0.02).  The
team-level controller replaces the whole XI's independent Bernoulli flips
with ONE shared engagement scalar g in [0,1]:

    g ~ 1  -> every near-ball defender presses at his role rate simultaneously
              (coordinated collective press)
    g ~ 0  -> nobody leaves the shape; compact block (SIT)

This probe samples the LIVE surface (MatchEngine._offball_move_player) at the
moment a player is about to decide a chase commit (allow==0, ball within
_CHASE_TRIGGER, defending).  It records:

  * 24-d TEAM sensors (ball zone relative to our goal, danger, score state,
    our / opp unit density around the ball, our shape compactness, block
    depth, formations, etc.),
  * the unit's CURRENT achieved engagement (%) = fraction of our outfielders
    already in an active chase burst at that instant,
  * and, after the match, the DEFENSIVE outcome of that moment via the
    decision-local correlation (true chronograph clock, 20 s window, 25 m
    proxmity to the ball point).

Rows are deduplicated into 0.5 s buckets per team so 11 chasers do not
swamp one unit-decision.

The surrogate (TeamPressSurrogate) then learns, per team-state bucket, the
expected success of engaging at SIT / MED / PRESS intensity — the fitness
signal for TeamPressBrain (team_offball_evolution.py).

Run:  python team_offball_probe.py [--matches 6] [--seed 21]
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
from football_brain import extract_team_sensors as _extract_team_sensors
from offball_probe import _score


# ─────────────────────────────────────────────────────────────
# COLLECTOR STATE
# ─────────────────────────────────────────────────────────────

_ENGAGEMENT_BANDS = ("press", "med", "sit")  # engagement >=0.6 / >=0.2 / else


def _band(effort: float) -> str:
    if effort >= 0.60:
        return "press"
    if effort >= 0.20:
        return "med"
    return "sit"


@dataclass
class TeamPressRecord:
    team: str
    clock: float
    minute: float
    trigger_player: str
    trigger_pos: str
    ball_x: float
    ball_y: float
    effort: float            # expected unit engagement around the ball (0..1)
    band: str                # sit | med | press
    style: str               # opponent style tag for the surrogate
    sensors: np.ndarray = field(repr=False)


_COLLECTOR: List[TeamPressRecord] = []
_ORIG_OFFBALL = None
_LAST_SAMPLE: Dict[int, float] = {}   # engine id -> last sampled pseudo-second
_STYLE = ""                           # current opponent style (per match)
_INTERVENE: Optional[float] = None    # forced engagement g for experimental matches
_CONTROLLER = None                     # TeamPressBrain live-drive mode
_G_CACHE: Dict[tuple, float] = {}      # (engine id, tick key, team) -> g this tick


def _unit_commitment(engine: Any, team: str, ball_x: float,
                     ball_y: float) -> float:
    """Collective engagement: fraction of the near-ball unit mid chase-burst.

    A press is a UNIT act, so the label is how much of the unit around the
    play is ACTUALLY committed (allow>0, ball within 30 m).  The current
    heuristic commits players independently and (as collection showed) almost
    never together — the whole point the team controller fixes by scaling
    every player's Bernoulli with g (prob = _PRESS_PROB[pos] * g).  This
    metric is exactly what a higher g drives up monotonically, so the
    surrogate can learn which collective-engagement band pays off per state.
    """
    pe = engine.position_engine
    near = []
    for nm in pe.team_rosters.get(team, []):
        st = pe.states.get(nm)
        if st is None:
            continue
        if getattr(st, "position", "") == "GK":
            continue
        d = ((st.current_x - ball_x) ** 2 + (st.current_y - ball_y) ** 2) ** 0.5
        if d <= 30.0:
            near.append((nm, getattr(st, "position", ""), d))
    if not near:
        return 0.0
    committed = 0
    for nm, pos, d in near:
        cst = engine._chase_state.get(nm)
        if cst is not None and cst.get("allow", 0.0) > 0:
            committed += 1
    return committed / len(near)


def _wrap_offball(self, pname: str, team: str, ball_x: float, ball_y: float,
                  has_ball: bool, danger_t: float, cross_team: str, DT: float):
    """Sample a TEAM press decision when any player is about to commit.

    When an experimental engagement ``g`` is set (_INTERVENE), the wrapper
    PRE-SETS the player's chase ``allow`` before the original runs, so the
    original's Bernoulli is skipped and the WHOLE unit plays at that fixed
    engagement (prob = _PRESS_PROB[pos] * g).  RNG consumption is identical
    to a plain match (one random() per commit), so runs stay deterministic
    per seed; no engine code is modified and _restore() fully reverts.
    """
    st = self.position_engine.states.get(pname)
    if st is None or getattr(st, "position", "") == "GK":
        return _ORIG_OFFBALL(self, pname, team, ball_x, ball_y, has_ball,
                             danger_t, cross_team, DT)

    cst = self._chase_state.get(pname)
    allow_pre = cst.get("allow", 0.0) if cst else -1.0
    ball_dist = ((ball_x - getattr(st, "current_x", ball_x)) ** 2 +
                 (ball_y - getattr(st, "current_y", ball_y)) ** 2) ** 0.5

    _clock = self.state.match_clock_s
    _key = round(_clock * 2.0)
    if team == self.config.away_team:
        _key += 1000000

    # Effective engagement this tick: live controller g (computed ONCE per
    # tick/team), else the forced experimental g, else None (observation).
    g_eff = None
    if _CONTROLLER is not None:
        ck = (id(self), _key, team)
        if ck in _G_CACHE:
            g_eff = _G_CACHE[ck]
        else:
            sens = _extract_team_sensors(self, team, ball_x, ball_y, danger_t)
            g_eff = float(_CONTROLLER.forward(sens))
            _G_CACHE[ck] = g_eff

    if g_eff is not None and cst is not None:
        # A pending commit moment: inject the engagement before original.
        if (cst.get("allow", 0.0) == 0.0 and cst.get("p", 0.0) <= 0.0
                and (not has_ball) and ball_dist <= self._CHASE_TRIGGER):
            prob = self._PRESS_PROB.get(getattr(st, "position", ""), 0.60) \
                * g_eff
            cst["allow"] = 1.0 if random.random() < prob else -1.0

    # Dedup into 0.5 s pseudo-seconds so 11 chasers sample one unit decision.
    _clock = self.state.match_clock_s
    _key = round(_clock * 2.0)
    if team == self.config.away_team:
        _key += 1000000

    result = _ORIG_OFFBALL(self, pname, team, ball_x, ball_y, has_ball,
                           danger_t, cross_team, DT)

    # Observed effort: live controller g, else forced experimental g, else
    # realised unit commitment (how much of the near-ball unit is in a burst).
    if _CONTROLLER is not None:
        effort = g_eff if g_eff is not None else _INTERVENE or 1.0
    elif _INTERVENE is not None:
        effort = _INTERVENE
    else:
        effort = _unit_commitment(self, team, ball_x, ball_y)

    # Opportunity: player was about to decide (allow==0 at entry) AND defending
    # AND ball near. allow_pre was captured BEFORE any intervention injection.
    ball_dist = ((ball_x - getattr(st, "current_x", ball_x)) ** 2 +
                 (ball_y - getattr(st, "current_y", ball_y)) ** 2) ** 0.5
    if allow_pre == 0.0 and (not has_ball) and ball_dist <= 12.5:
        if _LAST_SAMPLE.get(id(self), None) == _key:
            return result
        _LAST_SAMPLE[id(self)] = _key
        rec = TeamPressRecord(
            team=team,
            clock=_clock,
            minute=float(self.state.minute) + float(_clock % 60.0) / 60.0,
            trigger_player=pname,
            trigger_pos=getattr(st, "position", ""),
            ball_x=ball_x, ball_y=ball_y,
            effort=effort, band=_band(effort),
            style="", sensors=np.zeros(24, dtype=np.float64),
        )
        rec.sensors = _extract_team_sensors(self, team, ball_x, ball_y, danger_t)
        rec.style = _STYLE
        _COLLECTOR.append(rec)
    return result


def _install(style: str = "", intervene: Optional[float] = None,
             controller=None) -> None:
    global _ORIG_OFFBALL, _STYLE, _INTERVENE, _CONTROLLER
    _STYLE = style
    _INTERVENE = intervene
    _CONTROLLER = controller
    _LAST_SAMPLE.clear()   # engine ids are reused after GC — reset per match
    _G_CACHE.clear()
    if _ORIG_OFFBALL is None:
        _ORIG_OFFBALL = match_engine_module.MatchEngine._offball_move_player
        match_engine_module.MatchEngine._offball_move_player = _wrap_offball


def _restore() -> None:
    global _ORIG_OFFBALL, _STYLE, _INTERVENE, _CONTROLLER
    if _ORIG_OFFBALL is not None:
        match_engine_module.MatchEngine._offball_move_player = _ORIG_OFFBALL
        _ORIG_OFFBALL = None
    _STYLE = ""
    _INTERVENE = None
    _CONTROLLER = None
    _G_CACHE.clear()


# ─────────────────────────────────────────────────────────────
# DECISION-LOCAL OUTCOME (per team moment)
# ─────────────────────────────────────────────────────────────

def _correlate_team(records: List[TeamPressRecord], timeline: List[Any],
                    chronology: Optional[Any] = None,
                    window_s: float = 20.0, max_dist: float = 25.0) -> List[Dict[str, Any]]:
    """Pair each team engagement moment with its own defensive outcome.

    Same true-clock + proximity attribution as the individual probe, but the
    focal point is the BALL at decision time (the play is happening there) and
    the decision is the whole unit's engagement.  Nearest scored team event in
    [clock, clock+window_s] within max_dist of (ball_x, ball_y).
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
            if t < rec.clock:
                continue
            if t > rec.clock + window_s:
                break
            s = _score(ev)
            if s is None:
                continue
            dx = float(getattr(ev, "location_x", 50.0)) - rec.ball_x
            dy = float(getattr(ev, "location_y", 34.0)) - rec.ball_y
            if (dx * dx + dy * dy) ** 0.5 > max_dist:
                continue
            outcome = s
            seen = True
            break
        rows.append({
            "team": rec.team, "clock": rec.clock, "minute": rec.minute,
            "style": rec.style, "effort": rec.effort, "band": rec.band,
            "ball_x": rec.ball_x, "ball_y": rec.ball_y,
            "trigger_pos": rec.trigger_pos,
            "episode_score": outcome, "seen": seen,
            "sensors": rec.sensors.tolist(),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# TEAM PRESS SURROGATE
# ─────────────────────────────────────────────────────────────

class TeamPressSurrogate:
    """style -> team-state bucket -> {sit, med, press} -> expected success.

    Bucket uses a modest boolean subset of the team sensors so cells keep
    enough samples: ball zone (own/mid/final), danger high, our numbers near
    the ball, opp numbers near the ball, our-block compact, opp pressure on
    our goal-mouth.  Same shrinkage/min_samples scheme as OffBallSurrogate.
    """
    def __init__(self):
        self.table: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.prior = 0.5

    @staticmethod
    def bucket(sensors: np.ndarray) -> str:
        zone = 2 if sensors[14] > 0.5 else (1 if sensors[13] > 0.5 else 0)
        danger = int(sensors[3] > 0.5)
        ours = int(sensors[8] > 0.25)      # our unit around the ball
        opp = int(sensors[9] > 0.25)       # opp unit around the ball
        comp = int(sensors[7] < 0.5)       # our block compact
        gm = int(sensors[19] > 0.25)       # opp men near our goal-mouth
        return f"Z{zone}D{danger}O{ours}E{opp}C{comp}G{gm}"

    def fit(self, rows: List[Dict[str, Any]], min_samples: int = 3) -> "TeamPressSurrogate":
        grouped: Dict[str, List[Tuple[np.ndarray, str, float]]] = {}
        for r in rows:
            sensors = np.asarray(r["sensors"], dtype=np.float64)
            st = r.get("style", "")
            grouped.setdefault(st, []).append((sensors, r["band"], r["episode_score"]))
            grouped.setdefault("", []).append((sensors, r["band"], r["episode_score"]))

        for style, samples in grouped.items():
            acc: Dict[str, Dict[str, float]] = {}
            cnt: Dict[str, Dict[str, int]] = {}
            for sensors, band, score in samples:
                key = self.bucket(sensors)
                acc.setdefault(key, {}).setdefault(band, 0.0)
                cnt.setdefault(key, {}).setdefault(band, 0)
                acc[key][band] += score
                cnt[key][band] += 1
            for key, bands in cnt.items():
                tot_s = sum(acc[key].get(b, 0.0) for b in bands)
                tot_c = sum(bands.values())
                overall = tot_s / tot_c if tot_c else self.prior
                for band, c in bands.items():
                    if c < min_samples:
                        est = (acc[key][band] + overall * min_samples) / (c + min_samples)
                    else:
                        est = acc[key][band] / c
                    self.table.setdefault(style, {}).setdefault(key, {})[band] = est
                    self.counts.setdefault(style, {}).setdefault(key, {})[band] = c
        return self

    def _lookup(self, key: str, band: str, style: str) -> Optional[float]:
        for s in (style, ""):
            row = self.table.get(s, {}).get(key, {})
            if band in row:
                return row[band]
            if row:
                return sum(row.values()) / len(row)
        return None

    def expected_success(self, sensors: np.ndarray, band: str,
                         style: str = "") -> float:
        key = self.bucket(sensors)
        v = self._lookup(key, band, style)
        return v if v is not None else self.prior

    def support(self, sensors: np.ndarray, band: str, style: str = "") -> int:
        key = self.bucket(sensors)
        for s in (style, ""):
            row = self.counts.get(s, {}).get(key, {})
            if band in row:
                return int(row[band])
            return max(row.values(), default=0)
        return 0

    def is_evidence_cell(self, sensors: np.ndarray, style: str = "",
                         min_both: int = 5) -> bool:
        key = self.bucket(sensors)
        for s in (style, ""):
            row = self.counts.get(s, {}).get(key, {})
            best = max(row.values(), default=0)
            others = sum(v for b, v in row.items()
                         if b != max(row, key=row.get, default=""))
            if best >= min_both and others >= min_both:
                return True
        return False

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"prior": self.prior, "table": self.table,
                       "counts": self.counts}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TeamPressSurrogate":
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

def collect_team_samples(
    n_matches: int = 6,
    seed: int = 0,
    away_styles: Optional[List[str]] = None,
    home_style: str = "balanced",
    home_team: str = "Probe FC",
    away_team: str = "Rival FC",
    verbose: bool = True,
    intervene_levels: Optional[List[float]] = None,
    controller_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if away_styles is None:
        away_styles = ["fluid_counter"]

    controller = None
    if controller_path:
        from football_brain import TeamPressBrain
        controller = TeamPressBrain.load(controller_path)

    all_rows: List[Dict[str, Any]] = []
    for m in range(n_matches):
        seed_i = seed + m * 1000
        random.seed(seed_i)
        opp_style = away_styles[m % len(away_styles)]
        g_force = None
        if intervene_levels:
            g_force = intervene_levels[m % len(intervene_levels)]
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

        _install(style=opp_style, intervene=g_force, controller=controller)
        try:
            match_probe._pin_heuristic()
            try:
                result = eng.simulate()
            finally:
                match_probe._restore_neural()
        finally:
            _restore()

        recs = [r for r in _COLLECTOR if r.team == home_team]
        _COLLECTOR.clear()
        rows = _correlate_team(recs, result.timeline, chronology=result.chronology)
        all_rows.extend(rows)
        if verbose:
            if controller is not None:
                tag = "controller"
            elif g_force is not None:
                tag = f"g={g_force:.2f}"
            else:
                tag = "heuristic"
            print(f"  match {m+1}: {result.score_str}  team moments={len(rows)}  "
                  f"(opp={opp_style}, {tag}, {time.time()-t0:.1f}s)")
    return all_rows


def main():
    ap = argparse.ArgumentParser(description="Team-level press engagement collector.")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--away-styles", type=str, default=None,
                    help="Comma-separated opponent styles to sweep.")
    ap.add_argument("--intervene", type=str, default=None,
                    help="Comma-separated engagement g levels to FORCE per match "
                         "(0.15 sit / 0.5 med / 0.9 press). Experiments generate "
                         "the high-engagement counterfactuals the heuristic never "
                         "produces. Omit for plain observation.")
    ap.add_argument("--controller", type=str, default=None,
                    help="Path to a brains/XI.json TeamPressBrain to LIVE-DRIVE "
                         "off-ball engagement during the match (real-effort "
                         "validation; no engine edits).")
    ap.add_argument("--out-samples", type=str, default="brains_team/samples.json")
    ap.add_argument("--out-surrogate", type=str, default="brains_team/surrogate.json")
    args = ap.parse_args()

    styles = [s.strip() for s in args.away_styles.split(",") if s.strip()] \
        if args.away_styles else None
    levels = None
    if args.intervene:
        levels = [float(v.strip()) for v in args.intervene.split(",") if v.strip()]

    print(f"Collecting {args.matches} real matches (heuristic pinned; "
          f"opponents: {styles or ['fluid_counter']}); "
          f"engagement: {args.controller or (f'g levels {levels}' if levels else 'observed')}")
    all_rows = collect_team_samples(
        n_matches=args.matches, seed=args.seed, away_styles=styles,
        intervene_levels=levels, controller_path=args.controller,
    )

    print(f"\ntotal team press moments: {len(all_rows)}")
    if not all_rows:
        return

    if levels:
        print("\n--- engagement level -> defensive outcome (experimental data) ---")
        by_g: Dict[float, List[float]] = collections.defaultdict(list)
        for r in all_rows:
            by_g[round(r["effort"], 2)].append(r["episode_score"])
        for g, vals in sorted(by_g.items()):
            print(f"  g={g:.2f}  n={len(vals):>4}  mean={statistics.mean(vals):.3f}")

    print("\n--- engagement band distribution ---")
    for band, n in collections.Counter(r["band"] for r in all_rows).most_common():
        print(f"  {band:<6} n={n}")

    print("\n--- defensive outcome by engagement band (0..1, higher = better) ---")
    by_band: Dict[str, List[float]] = collections.defaultdict(list)
    for r in all_rows:
        by_band[r["band"]].append(r["episode_score"])
    for band, vals in sorted(by_band.items(), key=lambda kv: -len(kv[1])):
        print(f"  {band:<6} n={len(vals):>4}  mean={statistics.mean(vals):.3f}")

    print("\n--- mean effort by outcome-window-seen/not ---")
    seen = [r["effort"] for r in all_rows if r["seen"]]
    unseen = [r["effort"] for r in all_rows if not r["seen"]]
    print(f"  seen   n={len(seen):>4} effort={statistics.mean(seen) if seen else 0:.3f}")
    print(f"  unseen n={len(unseen):>4} effort={statistics.mean(unseen) if unseen else 0:.3f}")

    surrogate = TeamPressSurrogate().fit(all_rows)
    print(f"\n--- team press surrogate: {surrogate.report()} ---")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_samples)) or ".", exist_ok=True)
    with open(args.out_samples, "w") as f:
        json.dump(all_rows, f)
    surrogate.save(args.out_surrogate)
    print(f"samples: {args.out_samples}\nsurrogate: {args.out_surrogate}")


if __name__ == "__main__":
    main()