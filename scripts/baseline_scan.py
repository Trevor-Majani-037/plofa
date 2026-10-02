"""
PLOFA Phase 0 — MANAGERLESS BASELINE SCAN
==========================================
Runs 20 managerless scratch matches (same fixture as run_match.py) at fixed
seeds, captures score / xG / possession plus the env-gated MANAGER_TRACE
posture log, and checks determinism by running seed 42 twice.

DEVIATION from pitch_replay.run_scratch_match (documented in README.md):
run_scratch_match wires a random ManagerPool manager per seed
(pitch_replay.py:404-427), so a raw 20-match loop would mix manager variance
into seed variance. This runner builds the identical fixture but calls
engine.set_managers(None, None) — a pure squad/style baseline. The manager
bias layer (chase_shift/protect_shift thresholds) is therefore at its
neutral defaults inside TacticalAI.adjust.

Usage:  python scripts/baseline_scan.py
Output: manager_brains/v1/baseline_manager/{manager_trace.jsonl,
        match_results.csv, summary.txt}
"""
from __future__ import annotations
import json
import os
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_match as RM
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from squad_manager import SubstitutionController

BASE = os.path.join("manager_brains", "v1", "baseline_manager")
TRACE_FILE = os.path.join(BASE, "manager_trace.jsonl")
CSV_FILE = os.path.join(BASE, "match_results.csv")
SUMMARY_FILE = os.path.join(BASE, "summary.txt")
HOME = RM.HOME_TEAM
AWAY = RM.AWAY_TEAM

N_SEEDS = 20
DET_A = ("detA", 42)
DET_B = ("detB", 42)


@dataclass
class Row:
    label: str
    seed: int
    score: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    home_possession_pct: float
    match_clock_s: float


def managerless_scratch(seed):
    """Replicate pitch_replay.run_scratch_match but WIRE NO MANAGERS."""
    random.seed(seed)
    os.environ["PLOFA_TRACE_SEED"] = str(seed)

    home_squad = SquadBuilder.build(
        team_name=HOME,
        starters=RM.HOME_STARTERS,
        substitutes=RM.HOME_SUBS,
        team_superstars=RM.HOME_SUPERSTARS,
        set_piece_takers=RM.HOME_SP_TAKERS,
    )
    away_squad = SquadBuilder.build(
        team_name=AWAY,
        starters=RM.AWAY_STARTERS,
        substitutes=RM.AWAY_SUBS,
        team_superstars=RM.AWAY_SUPERSTARS,
        set_piece_takers=RM.AWAY_SP_TAKERS,
    )

    all_players_flat = (
        home_squad["starters"] + home_squad["substitutes"] +
        away_squad["starters"] + away_squad["substitutes"]
    )
    for player in all_players_flat:
        if player.name in RM.SOUL_PLAYERS:
            player.dna.soul = RM.SOUL_PLAYERS[player.name]

    config = MatchConfig(
        home_team=HOME,
        away_team=AWAY,
        match_date=RM.MATCH_DATE,
        matchday=RM.MATCHDAY,
        season=RM.SEASON,
        competition=RM.COMPETITION,
        venue=RM.VENUE,
        stadium_capacity=RM.CAPACITY,
        referee=RM.REFEREE,
        referee_strictness=RM.STRICTNESS,
        is_derby=RM.IS_DERBY,
    )

    sub_controller = SubstitutionController(
        home_team=HOME,
        away_team=AWAY,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=RM.HOME_STYLE.style.value,
        away_style=RM.AWAY_STYLE.style.value,
        manager_stubbornness=RM.MANAGER_STUBBORNNESS,
    )
    sub_controller.MAX_SUBS = RM.MAX_SUBS

    engine = MatchEngine(config, RM.HOME_STYLE, RM.AWAY_STYLE)
    engine.set_squad(HOME, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(AWAY, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_controller)
    engine.set_managers(None, None)  # managerless baseline
    return engine.simulate()


def to_row(label, seed, result) -> Row:
    return Row(
        label=label,
        seed=seed,
        score=result.score_str,
        home_goals=result.home_goals,
        away_goals=result.away_goals,
        home_xg=result.home_xg,
        away_xg=result.away_xg,
        home_possession_pct=result.home_possession_pct,
        match_clock_s=result.match_clock_s,
    )


def write_outputs(rows) -> None:
    """CSV + summary.txt. May be called from raw results (simulation) or
    re-run from an existing CSV (--postprocess)."""
    import csv

    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "seed", "score", "home_goals", "away_goals",
                    "home_xg", "away_xg", "home_possession_pct", "match_clock_s"])
        for r in rows:
            w.writerow([r.label, r.seed, r.score, r.home_goals, r.away_goals,
                        r.home_xg, r.away_xg, r.home_possession_pct, r.match_clock_s])

    base_rows = [r for r in rows if r.label.startswith("seed")]
    detA = next(r for r in rows if r.label == "detA")
    detB = next(r for r in rows if r.label == "detB")

    def avg(xs):
        return round(statistics.fmean(xs), 3)

    det_ok = (detA.score == detB.score and abs(detA.home_xg - detB.home_xg) < 1e-9
              and abs(detA.home_possession_pct - detB.home_possession_pct) < 1e-9)

    # ── posture distribution from the trace (20 baseline seeds only).
    # Trace seeds are the raw integers 0..19; detA/detB both log seed 42 and
    # must be excluded (their "42" is >= N_SEEDS).
    posture_count = Counter()
    minute_count = Counter()
    trace_rows = 0
    seen = set()
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not (rec["seed"].isdigit() and int(rec["seed"]) < N_SEEDS):
                continue
            key = (rec["seed"], rec["team"], rec["minute"])
            if key in seen:
                continue
            seen.add(key)
            trace_rows += 1
            posture_count[(rec["team"], rec["posture"])] += 1
            minute_count[rec["team"]] += 1

    det_line = (
        "IDENTICAL  (score=%s xG=%.2f poss=%.1f%%)" % (detA.score, detA.home_xg, detA.home_possession_pct)
        if det_ok else
        "MISMATCH  (detA %s %.2f %.1f%% vs detB %s %.2f %.1f%%)" % (
            detA.score, detA.home_xg, detA.home_possession_pct,
            detB.score, detB.home_xg, detB.home_possession_pct)
    )

    # ── summary.txt ────────────────────────────────────────────────────
    with open(SUMMARY_FILE, "w", encoding="utf-8") as s:
        s.write("PLOFA Phase 0 — MANAGERLESS BASELINE SNAPSHOT\n")
        s.write("=" * 72 + "\n\n")
        s.write("Fixture        : %s vs %s\n" % (HOME, AWAY))
        s.write("Home style     : %s / %s / %s\n" % (
            RM.HOME_STYLE.style.value, RM.HOME_STYLE.playing_style.value,
            RM.HOME_STYLE.intensity.value))
        s.write("Away style     : %s / %s / %s\n" % (
            RM.AWAY_STYLE.style.value, RM.AWAY_STYLE.playing_style.value,
            RM.AWAY_STYLE.intensity.value))
        s.write("Managers       : NONE (set_managers(None, None))\n")
        s.write("Seeds          : 0..%d + determinism pair (42 x2)\n\n" % (N_SEEDS - 1))

        s.write("%-7s %-10s %6s %7s %8s\n" % ("label", "score", "homeXG", "awayXG", "poss%"))
        s.write("-" * 44 + "\n")
        for r in base_rows:
            s.write("%-7s %-10s %6.2f %7.2f %7.1f\n" % (
                r.label, r.score, r.home_xg, r.away_xg, r.home_possession_pct))
        s.write("%-7s %-10s %6.2f %7.2f %7.1f\n" % (
            "detA", detA.score, detA.home_xg, detA.away_xg, detA.home_possession_pct))
        s.write("%-7s %-10s %6.2f %7.2f %7.1f\n" % (
            "detB", detB.score, detB.home_xg, detB.away_xg, detB.home_possession_pct))

        s.write("\nAGGREGATES (seeds 0..%d, n=%d)\n" % (N_SEEDS - 1, len(base_rows)))
        s.write("  home goals   avg %.2f  max %d  min %d\n" % (
            avg([r.home_goals for r in base_rows]),
            max(r.home_goals for r in base_rows),
            min(r.home_goals for r in base_rows)))
        s.write("  away goals   avg %.2f  max %d  min %d\n" % (
            avg([r.away_goals for r in base_rows]),
            max(r.away_goals for r in base_rows),
            min(r.away_goals for r in base_rows)))
        s.write("  total goals  avg %.2f\n" % avg(
            [r.home_goals + r.away_goals for r in base_rows]))
        s.write("  home xG      avg %.3f\n" % avg([r.home_xg for r in base_rows]))
        s.write("  away xG      avg %.3f\n" % avg([r.away_xg for r in base_rows]))
        s.write("  home poss %%  avg %.1f\n" % avg(
            [r.home_possession_pct for r in base_rows]))
        hw = sum(1 for r in base_rows if r.home_goals > r.away_goals)
        dr = sum(1 for r in base_rows if r.home_goals == r.away_goals)
        aw = len(base_rows) - hw - dr
        s.write("  results      %d home wins / %d draws / %d away wins\n\n" % (hw, dr, aw))

        s.write("POSTURE DISTRIBUTION (MANAGER_TRACE, %d adjust-call rows)\n" % trace_rows)
        s.write("  rows are one (team, minute) sample; att/def calls deduped by seed+minute+team\n")
        for team in (HOME, AWAY):
            s.write("  %s (n=%d sample-minutes)\n" % (team, minute_count[team]))
            for posture, cnt in posture_count.most_common():
                if posture[0] != team:
                    continue
                pct = 100.0 * cnt / minute_count[team] if minute_count[team] else 0.0
                s.write("    %-18s %5d  %5.1f%%\n" % (posture[1], cnt, pct))

        s.write("\nDETERMINISM (seed 42 twice): %s\n" % det_line)
        s.write("\nNotes\n")
        s.write("  - Managerless by design; run_scratch_match() wires ManagerPool\n")
        s.write("    managers, so this runner deliberately bypasses it.\n")
        s.write("  - MANAGER_TRACE is env-gated (PLOFA_MANAGER_TRACE); normal play\n")
        s.write("    is unchanged when unset.\n")

    print("\n[baseline_scan] wrote %s, %s, %s" % (CSV_FILE, TRACE_FILE, SUMMARY_FILE))
    print("[baseline_scan] determinism: %s" % (
        "OK (seed 42 reproducible)" if det_ok else "FAILED"))
    return 0 if det_ok else 1


def _load_rows_from_csv():
    import csv

    rows = []
    with open(CSV_FILE, newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(Row(
                label=rec["label"], seed=int(rec["seed"]), score=rec["score"],
                home_goals=int(rec["home_goals"]), away_goals=int(rec["away_goals"]),
                home_xg=float(rec["home_xg"]), away_xg=float(rec["away_xg"]),
                home_possession_pct=float(rec["home_possession_pct"]),
                match_clock_s=float(rec["match_clock_s"]),
            ))
    return rows


def main() -> int:
    if "--postprocess" in sys.argv:
        print("[baseline_scan] postprocess: re-aggregating %s + %s (no simulation)"
              % (CSV_FILE, TRACE_FILE))
        return write_outputs(_load_rows_from_csv())

    os.makedirs(BASE, exist_ok=True)
    open(TRACE_FILE, "w").close()
    os.environ["PLOFA_MANAGER_TRACE"] = TRACE_FILE

    jobs = [("seed%d" % s, s) for s in range(N_SEEDS)] + [DET_A, DET_B]
    rows = []

    print("[baseline_scan] managerless baseline (%d seeds + determinism pair)" % N_SEEDS)
    for label, seed in jobs:
        result = managerless_scratch(seed)
        r = to_row(label, seed, result)
        rows.append(r)
        print("  %-6s %s  xG %s-%s  poss %s%%" % (
            r.label, r.score, r.home_xg, r.away_xg, r.home_possession_pct))
    return write_outputs(rows)


if __name__ == "__main__":
    raise SystemExit(main())