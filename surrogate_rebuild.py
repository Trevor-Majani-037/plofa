"""Rebuild the fitness surrogate with MORE real data, covering all 11
positions (including CAM and CF, which the base template never fields).

Collects real matches, then audits per-position bucket counts and
signature-situation coverage.

Two collection modes:
  --decision heuristic  (default): pin the heuristic baseline so the
      sample distribution matches the original surrogate protocol.
  --decision neural:     run with the TRAINED neural XI (auto-load), which
      actually shoots from scoring positions.  This is required to seed
      real SHOOT outcomes in the "clear scoring chance" bucket: the
      heuristic brain almost never shoots there (ST clear-chance SHOOT
      was 1/736), so its own distribution can never fill that cell.
  --targeted: home plays the CAM template and away the CF template for
      every match, so CAM/CF are on the pitch 100% (rotation gave them
      1/3) and ST gets full coverage too.

Usage:
    python surrogate_rebuild.py --matches 20 --decision heuristic \
        --out brains/surrogate_pos_v2.json
    python surrogate_rebuild.py --matches 20 --decision neural --targeted \
        --out brains/surrogate_pos_neural.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from match_probe import _pin_heuristic, _restore_neural
from surrogate_collect import (
    TEMPLATE_BASE, TEMPLATE_CAM, TEMPLATE_CF,
    FitnessSurrogate, _build_match_components,
)
from surrogate_collect import _COLLECTOR, _correlate_outcomes

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

SIGNATURE = {
    "ST": ["SHOOT"], "CF": ["SHOOT"], "CAM": ["THROUGH_BALL", "PROGRESSIVE_PASS"],
    "CM": ["THROUGH_BALL", "PROGRESSIVE_PASS"], "CB": ["RECYCLE", "SAFE_PASS"],
    "CDM": ["RECYCLE", "SAFE_PASS"], "GK": ["RECYCLE", "SAFE_PASS"],
    "LB": ["CROSS"], "RB": ["CROSS"], "LW": ["CROSS"], "RW": ["CROSS"],
}

# bucket: P{pressure} FT{final_third} OH{own_half} S{space} C{central} G{goal_close}
CLEAR_CHANCE = "P0FT1OH0S1C1G1"          # final third, central, space, goal-close


def _clear_chance_buckets():
    # "clear scoring chance" = final_third(1), central(1), not own_half(0),
    # goal_close(1).  Two families:
    #   strict:  S=1 (space ahead) — analyst's literal definition, nearly
    #            unreachable in the engine (attacker at goal-close central
    #            almost never has open space ahead by sensor 9's definition)
    #   reachable: S free — the real scoring-position family where SHOOT
    #            samples actually accumulate (P?FT1OH0S?C1G1, 8 buckets)
    strict = {"P0FT1OH0S1C1G1", "P1FT1OH0S1C1G1"}
    reachable = {
        f"P{p}FT1OH0S{s}C1G1"
        for p in (0, 1) for s in (0, 1)
    }
    return strict, reachable


def _run_matches(n_matches, seed, away_styles, targeted, decision,
                 home_template=None, away_template=None):
    """Run real matches and return (sensor, intent, success, action, pos) pairs.

    Collects through the live NeuralDecisionBrain entry point (which the
    collector patches).  If ``decision == 'heuristic'`` the heuristic is
    pinned first; if ``'neural'`` the trained XI auto-loads and actually
    shoots from scoring positions (real outcomes).
    """
    import random

    from surrogate_collect import _install_collector, _restore_collector, _template_name

    if decision == "heuristic":
        _pin_heuristic()
    # neural: leave NeuralDecisionBrain auto-loading the trained XI

    _install_collector()
    all_paired = []
    try:
        for m in range(n_matches):
            random.seed(seed + m * 1000)
            opp_style = away_styles[m % len(away_styles)]
            if targeted:
                # home = CAM template (ST + CAM), away = CF template (CF):
                # CAM and CF at 100% presence instead of the 1/3 rotation.
                ht, at = TEMPLATE_CAM, TEMPLATE_CF
                tpl_tag = "CAM_home/CF_away"
            else:
                from surrogate_collect import _COLLECTION_TEMPLATES
                ht = at = _COLLECTION_TEMPLATES[m % len(_COLLECTION_TEMPLATES)]
                tpl_tag = _template_name(ht)
            if home_template is not None:
                ht = home_template
                tpl_tag = f"home={_template_name(ht)}"
            if away_template is not None:
                at = away_template
                tpl_tag += f"/away={_template_name(at)}"
            eng = _build_match_components(
                "Probe FC", "Rival FC", "attacking", opp_style,
                home_template=ht, away_template=at)
            result = eng.simulate()
            paired = _correlate_outcomes(result.timeline, list(_COLLECTOR))
            all_paired.extend(paired)
            _COLLECTOR.clear()
            print(f"  match {m+1}: collected {len(paired)} pairs "
                  f"(opp={opp_style}, template={tpl_tag}, decision={decision})")
    finally:
        _restore_collector()
        if decision == "heuristic":
            _restore_neural()
    return all_paired


def audit_position(sur, pos: str) -> dict:
    counts = sur.counts.get(pos, {})
    total = sum(c for k, v in counts.items() for c in v.values())
    # bucket totals (across intents)
    bucket_totals = {k: sum(v.values()) for k, v in counts.items()}
    fewest = min(bucket_totals.items(), key=lambda kv: kv[1]) if bucket_totals else (None, 0)
    strict, reachable = _clear_chance_buckets()
    sig_cells = {}
    sig_cells_strict = {}
    for intent in SIGNATURE.get(pos, []):
        for bucket in reachable:
            key = (bucket, intent)
            sig_cells[key] = counts.get(bucket, {}).get(intent, 0)
        for bucket in strict:
            key = (bucket, intent)
            sig_cells_strict[key] = counts.get(bucket, {}).get(intent, 0)
    sig_total = sum(sig_cells.values())
    sig_strict = sum(sig_cells_strict.values())
    return {
        "position": pos, "total_samples": total, "n_buckets": len(bucket_totals),
        "fewest": {"bucket": fewest[0], "n": fewest[1]},
        "clear_chance_samples": sig_total,
        "clear_chance_strict": sig_strict,
        "clear_chance_by_cell": {f"{b}|{i}": n for (b, i), n in sig_cells.items()},
    }


def _data_path(out: str) -> str:
    return out.replace(".json", ".raw.json")


def _load_raw(path: str):
    if not os.path.exists(path):
        return []
    from decision_brain import PlayerIntent
    with open(path) as f:
        obj = json.load(f)
    out = []
    for o in obj:
        sensors, ik, success, action, pos = o[0], o[1], o[2], o[3], o[4]
        out.append((np.array(sensors), PlayerIntent(ik), success, action, pos))
    return out


def _save_raw(path: str, data) -> None:
    obj = [[s.tolist(), intent.value, suc, act, pos]
           for (s, intent, suc, act, pos) in data]
    with open(path, "w") as f:
        json.dump(obj, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=20)
    p.add_argument("--styles", type=str,
                   default="fluid_counter,tiki_taka,attacking")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="brains/surrogate_pos_v2.json")
    p.add_argument("--decision", choices=("heuristic", "neural"),
                   default="heuristic")
    p.add_argument("--targeted", action="store_true",
                   help="CAM home / CF away every match (all 11 present 100 pct)")
    p.add_argument("--merge", action="store_true",
                   help="append to the raw data file from previous runs, then refit")
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    raw_path = _data_path(args.out)

    data = _run_matches(args.matches, args.seed, styles, args.targeted,
                        args.decision)
    print(f"\nTotal NEW decision->outcome samples: {len(data)}")

    if args.merge:
        prior = _load_raw(raw_path)
        data = prior + data
        print(f"Merged with {len(prior)} prior raw samples -> {len(data)} total")
    _save_raw(raw_path, data)

    sur = FitnessSurrogate().fit(data)
    sur.save(args.out)
    print(f"Surrogate saved -> {args.out}  (raw -> {raw_path})")
    print(f"Surrogate report: {json.dumps(sur.report())}\n")

    # Audit every position
    print("=== PER-POSITION SAMPLE AUDIT ===")
    print("(clear-chance = REACHABLE family P?FT1OH0S?C1G1: final third, "
          "central, not own-half, goal-close)")
    print("(strict      = analyst's literal 4-condition bucket w/ space ahead)")
    rows = []
    for pos in ALL_POSITIONS:
        a = audit_position(sur, pos)
        rows.append(a)
        print(f"{pos:>3}: samples={a['total_samples']:5d}  buckets={a['n_buckets']:2d}  "
              f"fewest={a['fewest']['bucket']}(n={a['fewest']['n']})  "
              f"clear-chance-{','.join(SIGNATURE[pos])}={a['clear_chance_samples']} "
              f"(strict-space={a['clear_chance_strict']})")
        for k, v in a["clear_chance_by_cell"].items():
            if v > 0:
                print(f"       clear-chance cell {k}: n={v}")
    with open("surrogate_audit.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nAudit JSON -> surrogate_audit.json")

    # Signature-situation <8 real samples check (reachable family)
    print("\n=== SIGNATURE SITUATION >=8 SAMPLE CHECK (reachable family) ===")
    thin = [a for a in rows if a["clear_chance_samples"] < 8]
    for a in rows:
        note = ("  (strict-space only: "
                f"{a['clear_chance_strict']})" if a["clear_chance_strict"] > 0 else "")
        mark = "PASS" if a["clear_chance_samples"] >= 8 else "THIN"
        print(f"{a['position']:>3}: {mark}  samples={a['clear_chance_samples']}{note}")
    if thin:
        print("\n! Positions with <8 clear-chance samples:", [a['position'] for a in thin])
    else:
        print("\nAll signature situations cleared 8 real samples.")


if __name__ == "__main__":
    main()