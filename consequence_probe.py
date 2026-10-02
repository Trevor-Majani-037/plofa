"""Offline consequence-collection probe + decisive experiment driver.

Credit-assignment prototype for PLOFA V2.  For every on-ball decision of
the live neural XI it records the vision state the brain actually saw,
then attributes to that decision the *possession-episode outcome* that
followed it in the SAME match (goal, xG generated, shots manufactured,
possession lost).  That turns the learning signal from decision-local
("was this one touch completed?") into temporal credit assignment
("did this possession end in a goal we can back-propagate to every
touch that built it?").

ZERO engine edits.  Two observation-only wrappers are installed around
live call sites and fully reverted before exit:
  * NeuralDecisionBrain.decide      (decision snapshot, from surrogate_collect)
  * MatchEngine._absorb_chain       (chain snapshot: exact per-chain outcome)
Nothing writes to season state; experiment artifacts go to a scratch dir.

The decisive comparison (main/run_experiment):
  - learned value model V(s)  (value_model.ValueModel, ridge, pure numpy)
  - production v1 surrogate expected_success(s, intent, pos)
  - intent mean baseline (does knowing the intent alone predict anything?)
evaluated on the SAME real decisions against the REAL chain outcome R.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain_sensors import extract_sensors
from perception import PerceptionConfig, get_perception_config, set_perception
from surrogate_collect import (
    DecisionRecord, _build_match_components, _COLLECTION_TEMPLATES,
    _install_collector, _restore_collector, _template_name,
    _COLLECTOR, _EXECUTION_TYPES,
)
from value_model import ValueModel, value_feature_matrix, _spearman


# ─────────────────────────────────────────────────────────────
# EPISODE SEGMENTATION (read: possession structure from the timeline)
# ─────────────────────────────────────────────────────────────

# Restart / intermission events close the previous episode but do NOT
# belong to it (kickoffs, throw-ins, goal kicks, corners, free kicks,
# penalties, offside flags, celebrations ...).
_RESTART_BOUNDARIES = {
    "KICKOFF", "THROW_IN", "GOAL_KICK", "CORNER_TAKEN", "CORNER_WON",
    "FREEKICK_WON", "FREEKICK_DIRECT", "FREEKICK_CROSS",
    "OWN_GOAL", "GOAL_CELEBRATION", "OFFSIDE",
}
# Chance RESOLUTION events: they belong to the episode whose shot climax
# closed immediately before (SHOT -> SAVE -> GOAL is the same attempt).
_RESOLUTION_EVENTS = {
    "SAVE", "GOAL", "HIT_WOODWORK", "PENALTY_SCORED", "PENALTY_MISSED",
    "VAR_DISALLOWED_GOAL",
}
# Shot/chance + possession-turnover events are the CLIMAX of the
# possession that produced them: they belong to that episode, then close
# it.  Defensive takeovers (interception, tackle won, clearance ...) end
# the attacking run.
_CLIMAX_BOUNDARIES = {
    "SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
    "HIT_WOODWORK", "SAVE", "GOAL",
    "TURNOVER", "MISCONTROL", "DISPOSSESSED",
    "INTERCEPTION", "TACKLE_WON", "CLEARANCE", "RECOVERY", "BLOCK",
}
_ON_BALL_ACTION = {
    "PASS", "PROGRESSIVE_PASS", "THROUGH_BALL", "SWITCH_OF_PLAY",
    "CROSS_ATTEMPT", "CROSS_SUCCESS", "CARRY", "DRIBBLE_ATTEMPT",
    "DRIBBLE_SUCCESS", "DRIBBLE_FAIL",
}
# The source events that CARRY the engine's xG for a manufactured chance.
_SHOT_SOURCE = {
    "SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
    "HIT_WOODWORK",
}
_SHOT_TYPES = _SHOT_SOURCE | {"SAVE", "GOAL"}


@dataclass
class Episode:
    """A possession run: consecutive touches by one team until resolution."""
    idx: int
    team: str = ""
    events: List[Any] = field(default_factory=list)
    minutes: Tuple[float, float] = (0.0, 0.0)

    @property
    def goals(self) -> int:
        return sum(1 for e in self.events
                   if e.event_type.name in ("GOAL", "PENALTY_SCORED")
                   and e.team == self.team)

    @property
    def xg(self) -> float:
        return float(sum(getattr(e, "xg", 0.0) or 0.0 for e in self.events
                         if e.event_type.name in _SHOT_SOURCE
                         and e.team == self.team))

    @property
    def shots(self) -> int:
        return sum(1 for e in self.events
                   if e.event_type.name in _SHOT_TYPES and e.team == self.team)

    @property
    def turnovers(self) -> int:
        return sum(1 for e in self.events
                   if e.event_type.name in ("TURNOVER", "MISCONTROL", "DISPOSSESSED")
                   and e.team == self.team)

    @property
    def return_value(self) -> float:
        """Chain consequence scalar (documented, engine-measured outputs).

        Weights simply rank the engine's OWN outcome measurements, they do
        not re-define success: a goal is worth most, engine xG is worth its
        measured size, shots make small progress, losing possession is a
        cost.  Every component is counted for the EPISODE team's own events.
        """
        return (3.0 * self.goals + self.xg + 0.2 * self.shots
                - 0.6 * self.turnovers)


def segment_episodes(timeline: List[Any]) -> List[Episode]:
    """Split the match timeline into possession episodes.

    Shots/goals/turnovers are the CLIMAX of the possession that built
    them (they belong to it and close it); restarts are intermissions
    that close the previous episode only.  Returns the episode list; the
    event->episode map is built by the caller from the episode event
    lists.
    """
    episodes: List[Episode] = []
    cur: Optional[Episode] = None
    last_closed: Optional[Episode] = None

    for ev in timeline:
        et = getattr(ev.event_type, "name", "")
        team = getattr(ev, "team", "")

        if et in _RESOLUTION_EVENTS:
            # SAVE/GOAL complete the attempt whose SHOT climax just closed.
            if last_closed is not None:
                last_closed.events.append(ev)
                last_closed = None
            continue

        if et in _RESTART_BOUNDARIES:
            if cur is not None and cur.events:
                episodes.append(cur)
            cur = None
            last_closed = None
            continue

        if et in _CLIMAX_BOUNDARIES:
            if cur is None:
                cur = Episode(idx=len(episodes), team=team)
            cur.idx = len(episodes)
            cur.events.append(ev)
            episodes.append(cur)
            if et in _SHOT_TYPES:
                last_closed = cur
            else:
                last_closed = None
            cur = None
            continue

        if cur is None:
            cur = Episode(idx=len(episodes), team=team)
        elif et in _ON_BALL_ACTION and cur.team and team and team != cur.team:
            # possession switched on the ball -> previous run is complete
            episodes.append(cur)
            cur = Episode(idx=len(episodes), team=team)

        if not cur.team and team:
            cur.team = team
        cur.events.append(ev)

    if cur is not None and cur.events:
        episodes.append(cur)

    for ep in episodes:
        if ep.events:
            ep.minutes = (min(e.minute for e in ep.events),
                          max(e.minute for e in ep.events))
    return episodes


# ─────────────────────────────────────────────────────────────
# DECISION -> EPISODE CORRELATION
# ─────────────────────────────────────────────────────────────

def _pair_decisions_to_events(timeline: List[Any],
                              records: List[DecisionRecord],
                              ) -> List[Tuple[DecisionRecord, Any]]:
    """Pair each decision snapshot to its own on-ball execution event.

    Mirrors surrogate_collect._correlate_outcomes but keeps the EVENT
    object so we can look up which possession episode it landed in.

    FIX (2026-09-12): the old pairing walked each player's execution
    events with a strict "minute >= decision.minute" cursor.  Shot events
    carry the timeline clock, which can sit ~1 minute AROUND the decision's
    snapshot minute (chance chains span several play minutes), so SUCCEEDing
    shots were skipped and SHOOT decisions paired to a later unrelated
    PASS/MISCONTROL — the v2 shooting signal came out all zeros.  Now the
    match is windowed (decision.minute +/- 1.6) and intent-aware:
    SHOOT decisions PREFER shot-type events (SHOT_*/GOAL/SAVE) within the
    window, everything else ties to the time-nearest execution event.
    """
    from surrogate_collect import _EXECUTION_TYPES
    _SHOT_EVENTS = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
                    "GOAL", "SAVE"}
    _WINDOW = 1.6

    by_player: Dict[str, List[Dict[str, Any]]] = {}
    for ev in timeline:
        p = getattr(ev, "player", None)
        if p is None or getattr(ev.event_type, "name", "") not in _EXECUTION_TYPES:
            continue
        by_player.setdefault(p, []).append({
            "minute": getattr(ev, "minute", 0),
            "ev": ev,
        })

    records_sorted = sorted(records, key=lambda r: (r.player, r.order))
    cursor: Dict[str, int] = {}
    paired: List[Tuple[DecisionRecord, Any]] = []
    for rec in records_sorted:
        events = by_player.get(rec.player, [])
        if not events:
            continue
        m = float(rec.minute)
        ik = rec.intent.value if hasattr(rec.intent, "value") else str(rec.intent)
        preferred = _SHOT_EVENTS if ik == "SHOOT" else frozenset()

        i = cursor.get(rec.player, 0)
        while i < len(events) and events[i]["minute"] < m - _WINDOW:
            i += 1
        best = -1
        best_rank = 1
        best_dist = float("inf")
        j = i
        while j < len(events) and events[j]["minute"] <= m + _WINDOW:
            nm = events[j]["ev"].event_type.name
            rank = 0 if nm in preferred else 1
            dist = abs(events[j]["minute"] - m)
            if rank < best_rank or (rank == best_rank and dist < best_dist):
                best, best_rank, best_dist = j, rank, dist
            j += 1
        if best < 0:
            # nothing inside the window: fall back to the strict-next event
            while i < len(events) and events[i]["minute"] < m:
                i += 1
            if i >= len(events):
                continue
            best = i
        paired.append((rec, events[best]["ev"]))
        cursor[rec.player] = best + 1
    return paired


# ─────────────────────────────────────────────────────────────
# COLLECTION
# ─────────────────────────────────────────────────────────────

@dataclass
class Consequence:
    """One decision with its full possession outcome."""
    sensors: np.ndarray
    intent: str
    position: str
    player: str
    minute: float
    episode_team: str
    R: float
    goals: int
    xg: float
    shots: int
    turnovers: int
    episode_n_events: int
    episode_idx: int


def collect_consequences(n_matches: int = 6, seed: int = 21,
                         home: str = "Probe FC", away: str = "Rival FC",
                         home_style: str = "balanced",
                         away_style: str = "fluid_counter",
                         away_styles: Optional[List[str]] = None,
                         verbose: bool = True,
                         perception: bool = True,
                         ) -> Tuple[List[Consequence], List[Episode]]:
    """Run real neural-XI matches and collect decision->episode outcomes.

    Returns (consequences, all_episodes) so the caller can inspect the
    raw possession structure behind the targets.

    ``perception=True`` (default) pins the production perception world
    (enabled + role blocks, the gate-winning regime) for the whole run,
    so the corpus records states the players ACTUALLY saw, regardless of
    the calling process's env flags.  ``perception=False`` restores the
    identity regime (the pre-2026-09-13 corpus).

    The collection must record the RAW base-policy distribution, so
    consequence reasoning is pinned OFF for the run (the process default
    is ON since the 2026-09-20 graduation) and the previous posture
    restored on exit.
    """
    if away_styles is None:
        away_styles = [away_style]

    _COLLECTOR.clear()
    _install_collector()

    saved_cfg = get_perception_config()
    set_perception(PerceptionConfig(enabled=perception, role_blocks=True, seed=0))

    from brain_integration import clear_consequence as _clear_cons
    from consequence_decision import (
        default_enabled as _set_default, default_state as _get_default,
    )
    _saved_default = _get_default()
    _clear_cons()
    _set_default(False)

    all_conseq: List[Consequence] = []
    all_episodes: List[Episode] = []
    try:
        for m in range(n_matches):
            random.seed(seed + m * 1000)
            opp = away_styles[m % len(away_styles)]
            template = _COLLECTION_TEMPLATES[m % len(_COLLECTION_TEMPLATES)]
            eng = _build_match_components(home, away, home_style, opp,
                                          template=template)
            result = eng.simulate()

            events = list(result.timeline)
            episodes = segment_episodes(events)
            all_episodes.extend(episodes)
            event_to_episode = {}
            for ep in episodes:
                for ev in ep.events:
                    event_to_episode[id(ev)] = ep

            paired = _pair_decisions_to_events(events, list(_COLLECTOR))
            for rec, ev in paired:
                ep = event_to_episode.get(id(ev))
                if ep is None:
                    continue
                all_conseq.append(Consequence(
                    sensors=rec.sensors,
                    intent=rec.intent.value if hasattr(rec.intent, "value") else str(rec.intent),
                    position=rec.position,
                    player=rec.player,
                    minute=rec.minute,
                    episode_team=ep.team,
                    R=ep.return_value,
                    goals=ep.goals,
                    xg=ep.xg,
                    shots=ep.shots,
                    turnovers=ep.turnovers,
                    episode_n_events=len(ep.events),
                    episode_idx=ep.idx,
                ))
            _COLLECTOR.clear()
            if verbose:
                print(f"  match {m+1}: {len(paired)} decisions, "
                      f"{sum(1 for c in all_conseq if c.episode_idx >= 0)} matched, "
                      f"{len(episodes)} episodes "
                      f"(opp={opp}, template={_template_name(template)})")
    finally:
        _restore_collector()
        set_perception(saved_cfg)
        _set_default(_saved_default)
    return all_conseq, all_episodes


# ─────────────────────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────────────────────

def _surrogate_predictions(sur, conseq: List[Consequence]) -> np.ndarray:
    preds = []
    for c in conseq:
        intent = c.intent
        preds.append(sur.expected_success(c.sensors, intent, c.position))
    return np.asarray(preds)


def run_experiment(n_matches: int, seed: int, surrogate_path: str,
                   out_dir: str, verbose: bool = True,
                   perception: bool = True,
                   away_styles: Optional[List[str]] = None) -> Dict[str, Any]:
    print(f"Collecting {n_matches} real matches (live neural XI) ...")
    conseq, episodes = collect_consequences(n_matches=n_matches, seed=seed,
                                            perception=perception,
                                            away_styles=away_styles)
    n = len(conseq)
    print(f"Total decision->episode samples: {n}")

    # Episode anatomy: how do possessions end, and where do decisions live?
    n_ep = len(episodes)
    payoff_eps = [ep for ep in episodes if ep.goals > 0 or ep.shots > 0]
    turnover_eps = [ep for ep in episodes if ep.turnovers > 0]
    episode_anatomy = {
        "episodes": n_ep,
        "mean_touches_per_episode": round(float(
            np.mean([len(ep.events) for ep in episodes])), 2),
        "payoff_episode_share": round(len(payoff_eps) / n_ep, 4),
        "turnover_episode_share": round(len(turnover_eps) / n_ep, 4),
        "goal_episode_share": round(float(
            sum(1 for ep in episodes if ep.goals > 0)) / n_ep, 4),
    }
    if n == 0:
        raise SystemExit("no decision->episode samples collected")

    # Dataset
    records = [(c.sensors, c.position) for c in conseq]
    X = value_feature_matrix(records)
    R = np.asarray([c.R for c in conseq], dtype=np.float64)

    # 1. Learned value model
    vm = ValueModel(ridge_lambda=1.0)
    report_model = vm.score_cv(X, R, folds=5, seed=3)

    # 2. Production v1 surrogate
    from surrogate_collect import FitnessSurrogate
    sur = FitnessSurrogate.load(surrogate_path)
    sur_pred = _surrogate_predictions(sur, conseq)
    report_sur = {"spearman": _spearman(sur_pred, R),
                  "pearson": round(float(np.corrcoef(sur_pred, R)[0, 1]), 4),
                  "n": n}

    # 3. Intent-mean baseline (intent alone carries no credit)
    intent_mean = {}
    for c in conseq:
        intent_mean.setdefault(c.intent, []).append(c.R)
    baseline_pred = np.asarray(
        [float(np.mean(intent_mean.get(c.intent, [0.0]))) for c in conseq])
    report_intent = {"spearman": _spearman(baseline_pred, R),
                     "pearson": round(float(np.corrcoef(baseline_pred, R)[0, 1]), 4),
                     "n": n}

    # 3b. Binary payoff view: can any predictor rank which possessions
    # will END in a manufactured shot or goal (vs die harmlessly)?  The
    # regression target above is dominated by turnover mass; the payoff
    # indicator is the decision-relevant signal.
    payoff = (np.asarray([(c.goals > 0) or (c.shots > 0) for c in conseq],
                          dtype=np.float64))
    payoff_base = float(payoff.mean())
    vm_pay = ValueModel(ridge_lambda=1.0)
    pay_pred = np.empty(n)
    rng = np.random.RandomState(3)
    for fold in np.array_split(rng.permutation(n), 5):
        mask = np.zeros(n, dtype=bool)
        mask[fold] = True
        m = ValueModel(1.0).fit(X[~mask], payoff[~mask])
        pay_pred[fold] = m.predict(X[fold])
    report_payoff = {
        "payoff_share": round(payoff_base, 4),
        "value_model_spearman": round(_spearman(pay_pred, payoff), 4),
        "value_model_accuracy": round(float(
            np.mean(((pay_pred >= payoff_base) > 0.5) == (payoff == 1.0))), 4),
        "catch_rate_at_top_decile": round(float(
            payoff[pay_pred >= np.percentile(pay_pred, 90)].mean()), 4),
    }

    # 4. Per-position table
    per_pos = {}
    for c in conseq:
        per_pos.setdefault(c.position, []).append(c)
    pos_table = []
    for pos, cs in sorted(per_pos.items()):
        Xp = value_feature_matrix([(c.sensors, c.position) for c in cs])
        Rp = np.asarray([c.R for c in cs], dtype=np.float64)
        vm_p = ValueModel(ridge_lambda=1.0).fit(Xp, Rp)
        model_rho = _spearman(vm_p.predict(Xp), Rp)
        sur_rho = _spearman(_surrogate_predictions(sur, cs), Rp)
        pos_table.append({
            "position": pos, "n": len(cs),
            "goals": sum(c.goals for c in cs),
            "mean_R": round(float(Rp.mean()), 3),
            "model_infit_rho": round(model_rho, 3),
            "surrogate_rho": round(sur_rho, 3),
        })

    # 5. Physics interpretation of the learned value
    s_all = np.asarray([c.sensors for c in conseq])
    vm_full = ValueModel(ridge_lambda=1.0).fit(X, R)
    V = vm_full.predict(X)
    fwd = s_all[:, 0]                    # normalized x from own goal
    goal_dist = s_all[:, 14]             # normalized distance to enemy goal
    rho_fwd = round(_spearman(V, fwd), 4)         # value rises toward enemy goal?
    rho_gd = round(_spearman(V, -goal_dist), 4)   # value rises as goal nears?

    # 6. V-delta direction demo: within each episode, does model value of
    #    later touches exceed earlier ones when the episode PAYS OFF?
    by_ep = {}
    for c in conseq:
        by_ep.setdefault((c.episode_idx, c.episode_team), []).append(c)
    rises_payoff = rises_no = falls_payoff = falls_no = 0
    for (eidx, team), cs in by_ep.items():
        if len(cs) < 2:
            continue
        order = sorted(cs, key=lambda c: c.minute)
        v_first = vm_full.predict(value_feature_matrix(
            [(order[0].sensors, order[0].position)]))[0]
        v_last = vm_full.predict(value_feature_matrix(
            [(order[-1].sensors, order[-1].position)]))[0]
        payoff = (order[-1].goals > 0) or (order[-1].shots > 0)
        if v_last > v_first:
            rises_payoff += int(payoff)
            rises_no += int(not payoff)
        else:
            falls_payoff += int(payoff)
            falls_no += int(not payoff)
    rise_total = rises_payoff + rises_no
    vdelta = {
        "episodes_with_2plus_decisions": rise_total + falls_payoff + falls_no,
        "value_rose": rise_total,
        "of_rose_payoff": rises_payoff,
        "of_rose_no_payoff": rises_no,
        "rose_payoff_share": round(rises_payoff / (rise_total or 1), 3),
        "fall_payoff_share": round(falls_payoff / ((falls_payoff + falls_no) or 1), 3),
    }

    report = {
        "matches": n_matches, "seed": seed, "samples": n,
        "perception": "role_blocks" if perception else "identity",
        "episodes_covered": len(by_ep),
        "episode_anatomy": episode_anatomy,
        "goal_decisions_share": round(float(sum(1 for c in conseq if c.goals)) / n, 4),
        "mean_R": round(float(R.mean()), 3),
        "value_model": report_model,
        "surrogate_v1": report_sur,
        "intent_baseline": report_intent,
        "payoff_view": report_payoff,
        "position_table": pos_table,
        "physics": {"value_vs_forward": rho_fwd, "value_vs_goal_proximity": rho_gd},
        "v_delta_direction": vdelta,
        "top_value_weights": vm_full.top_weights(k=12),
    }

    os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(out_dir, "consequence_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump([
            {"sensors": [round(float(v), 6) for v in c.sensors],
             "intent": c.intent, "position": c.position, "player": c.player,
             "minute": c.minute, "R": c.R, "goals": c.goals, "xg": c.xg,
             "shots": c.shots, "turnovers": c.turnovers,
             "episode_idx": c.episode_idx, "episode_team": c.episode_team}
            for c in conseq
        ], f, indent=1)
    report_path = os.path.join(out_dir, "value_experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print("DECISIVE EXPERIMENT — consequence-learned value vs production predictors")
    print("=" * 64)
    print(f"  {n} decisions, {n_ep} episodes "
          f"({episode_anatomy['mean_touches_per_episode']} touches/ep), "
          f"mean episode R = {episode_anatomy.get('mean_R', R.mean()):.3f}")
    a = episode_anatomy
    print(f"  Episode anatomy: payoff {a['payoff_episode_share']:.1%} | "
          f"goal {a['goal_episode_share']:.1%} | "
          f"turnover {a['turnover_episode_share']:.1%}")
    print("\n  Predictor        | R2_oos | Spearman | Pearson")
    print("  " + "-" * 52)
    print(f"  value model V(s) | {report_model['r2_oos']:>6} | {report_model['spearman']:>8} | {report_model['pearson']:>7}")
    print(f"  surrogate v1     |   n/a  | {report_sur['spearman']:>8} | {report_sur['pearson']:>7}")
    print(f"  intent baseline  |   n/a  | {report_intent['spearman']:>8} | {report_intent['pearson']:>7}")
    print("\n  Physics of the learned value:")
    print(f"    value vs forward-progress     rho = {rho_fwd}")
    print(f"    value vs goal-proximity       rho = {rho_gd}")
    print(f"\n  Payoff view (possession ends with a shot/goal, {report_payoff['payoff_share']:.1%} of decisions):")
    print(f"    value model ranks payoff        rho = {report_payoff['value_model_spearman']}")
    print(f"    catch-rate in model's top decile = {report_payoff['catch_rate_at_top_decile']:.3f} "
          f"(base rate {report_payoff['payoff_share']:.3f})")
    if rise_total:
        print(f"\n  V-delta direction (episodes with 2+ decisions, {rise_total + falls_payoff + falls_no}):")
        print(f"    V rose in {rise_total}: payoff share {vdelta['rose_payoff_share']} "
              f"(vs {vdelta['fall_payoff_share']} when V fell)")
    print("\n  Top value coefficients (standardized):")
    for label, w in report["top_value_weights"]:
        print(f"    {w:+.3f}  {label}")
    print("\n  Per-position:")
    hdr = f"    {'pos':<4} {'n':>4} {'goals':>5} {'meanR':>6} {'model-rho':>9} {'sur-rho':>8}"
    print(hdr)
    for row in pos_table:
        print(f"    {row['position']:<4} {row['n']:>4} {row['goals']:>5} {row['mean_R']:>6} "
              f"{row['model_infit_rho']:>9} {row['surrogate_rho']:>8}")
    print("\n  Artifacts: " + dataset_path)
    print("  Report:     " + report_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Consequence-learning value-model experiment (offline, read-only).")
    p.add_argument("--matches", type=int, default=6)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--surrogate", type=str, default="brains/surrogate_pos.json")
    p.add_argument("--out", type=str, default="value_experiments")
    p.add_argument("--away-styles", type=str, default="fluid_counter",
                   help="comma-separated opposition styles rotated across matches")
    p.add_argument("--perception", dest="perception", action="store_true",
                   default=True, help="collect under the gate-winning role-block perception")
    p.add_argument("--no-perception", dest="perception", action="store_false",
                   help="identity regime (pre-2026-09-13 corpus)")
    args = p.parse_args()
    styles = [s.strip() for s in args.away_styles.split(",") if s.strip()]
    run_experiment(args.matches, args.seed, args.surrogate, args.out,
                   perception=args.perception, away_styles=styles)


if __name__ == "__main__":
    main()