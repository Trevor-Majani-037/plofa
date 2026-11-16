"""The real-match gate for the shot read-out.

Compares the SAME brains, same seed, same styles, with the read-out off and on.
The only difference between the two arms is `PLOFA_SHOT_HEAD`, which is read at
decision time, so this is a paired comparison and not two independent samples.

What it measures, and why each one is here:

  goals / xG     — the thing football is scored by.  A gate that ignores these
                   is a gate that can promote a team that shoots beautifully and
                   loses 0-3.
  shots          — the behaviour the read-out exists to change.  If this does not
                   move, nothing happened and the rest is noise.
  shots on the   — a rise in shots with a flat xG per shot means the player has
  same xG        — started shooting from bad positions: the failure mode the
                   "untouched states left alone" number was trying to predict.
  intent mix     — the argmax distribution.  The offline measurement promised the
                   other nine intents would not move; this is where that promise
                   is checked against the engine instead of the model.

Usage:
    python tools/ops/gate_shot_head.py --seeds 123,456
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from validate_neural_xl import _run_neural  # noqa: E402


def _shots(result):
    """Shot intents and goal types from the timeline."""
    n_shot = 0
    n_goal = 0
    intents = {}
    for ev in getattr(result, "timeline", []) or []:
        md = getattr(ev, "metadata", None) or {}
        brain = md.get("active_brain") or {}
        intent = brain.get("intent")
        if intent:
            intents[intent] = intents.get(intent, 0) + 1
        if intent == "SHOOT":
            n_shot += 1
        etype = str(getattr(ev, "event_type", "") or getattr(ev, "type", "") or "")
        if "GOAL" in etype.upper():
            n_goal += 1
    return n_shot, n_goal, intents


def _arm(brains, seed, home, away, armed):
    """One match with the read-out explicitly on or off."""
    if armed:
        os.environ["PLOFA_SHOT_HEAD"] = "1"
    else:
        os.environ.pop("PLOFA_SHOT_HEAD", None)

    random.seed(seed)
    result, fit = _run_neural(brains, seed, home, away)
    shots, goals_ev, intents = _shots(result)
    return {
        "armed": armed,
        "seed": seed,
        "score": result.score_str,
        "goals": fit.get("goals"),
        "conceded": result.away_goals,
        "xg": round(float(fit.get("xg", 0.0) or 0.0), 3),
        "possession": fit.get("possession_pct"),
        "shots": shots,
        "goal_events": goals_ev,
        "fitness": round(float(fit.get("fitness", 0.0) or 0.0), 4),
        "intents": intents,
    }


def _load_rows(path: str) -> list:
    """Previously completed arms, so a crash costs one match and not the run.

    This exists because it already happened twice: a 16-minute four-match run
    was lost when the machine restarted, and an earlier corpus collection died
    the same way because results were only written at the very end.  A match is
    four to five minutes of work; it should never be thrown away.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("rows", [])
    except Exception:
        return []


def _save(path: str, rows: list) -> None:
    """Write what we have so far.  Called after EVERY arm."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "complete": False}, fh, indent=2)
    os.replace(tmp, path)          # atomic: a crash mid-write cannot corrupt it


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-match gate for the shot head.")
    ap.add_argument("--seeds", default="123,456")
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--home-style", default="fluid_counter")
    ap.add_argument("--away-style", default="low_block")
    ap.add_argument("--out", default=os.path.join(
        "artifacts", "shot_head_gate_20261116.json"))
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore and overwrite any checkpointed rows.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = [] if args.fresh else _load_rows(args.out)
    done = {(r["seed"], bool(r["armed"])) for r in rows}
    if done:
        print(f"resuming: {len(done)} arm(s) already complete in {args.out}",
              flush=True)
    t0 = time.time()

    for seed in seeds:
        for armed in (False, True):
            if (seed, armed) in done:
                print(f"[{'ON ' if armed else 'OFF'}] seed {seed}: already done",
                      flush=True)
                continue
            tag = "ON " if armed else "OFF"
            print(f"[{tag}] seed {seed} ...", flush=True)
            t1 = time.time()
            row = _arm(args.brains, seed, args.home_style, args.away_style, armed)
            row["elapsed_s"] = round(time.time() - t1, 1)
            rows.append(row)
            _save(args.out, rows)
            print(f"[{tag}] seed {seed}: {row['score']}  goals {row['goals']}  "
                  f"xg {row['xg']}  shots {row['shots']}  "
                  f"poss {row['possession']}  fit {row['fitness']}  "
                  f"({row['elapsed_s']}s)", flush=True)

    os.environ.pop("PLOFA_SHOT_HEAD", None)

    off = [r for r in rows if not r["armed"]]
    on = [r for r in rows if r["armed"]]

    def tot(rs, k):
        return sum(r[k] for r in rs)

    def avg(rs, k):
        return round(sum(r[k] for r in rs) / len(rs), 4) if rs else 0.0

    summary = {
        "matches_per_arm": min(len(off), len(on)),
        "seeds": sorted({r["seed"] for r in rows}),
        "goals": {"off": tot(off, "goals"), "on": tot(on, "goals")},
        "conceded": {"off": tot(off, "conceded"), "on": tot(on, "conceded")},
        "xg": {"off": avg(off, "xg"), "on": avg(on, "xg")},
        "shots": {"off": tot(off, "shots"), "on": tot(on, "shots")},
        "fitness": {"off": avg(off, "fitness"), "on": avg(on, "fitness")},
        "possession": {"off": avg(off, "possession"), "on": avg(on, "possession")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary["xg_per_shot"] = {
        "off": round(summary["xg"]["off"] * max(1, len(off)) /
                     max(1, summary["shots"]["off"]), 4),
        "on": round(summary["xg"]["on"] * max(1, len(on)) /
                    max(1, summary["shots"]["on"]), 4),
    }
    paired = min(len(off), len(on))

    print(f"\n=== SHOT HEAD GATE ({paired} paired match(es)) ===")
    if paired == 0:
        print("  no complete pairs yet — run again to add the other arm")
    print(f"  goals      OFF {summary['goals']['off']}   ON {summary['goals']['on']}")
    print(f"  conceded   OFF {summary['conceded']['off']}   ON {summary['conceded']['on']}")
    print(f"  xG / match OFF {summary['xg']['off']:.3f}  ON {summary['xg']['on']:.3f}")
    print(f"  shots      OFF {summary['shots']['off']}   ON {summary['shots']['on']}")
    print(f"  xG / shot  OFF {summary['xg_per_shot']['off']:.4f}  "
          f"ON {summary['xg_per_shot']['on']:.4f}")
    print(f"  fitness    OFF {summary['fitness']['off']:.4f}  "
          f"ON {summary['fitness']['on']:.4f}")
    print(f"  possession OFF {summary['possession']['off']}  "
          f"ON {summary['possession']['on']}")
    if paired < 3:
        print(f"\n  !! {paired} pair(s) cannot decide this. Add seeds before "
              f"reading anything into the direction.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows, "complete": True}, fh,
                  indent=2)
    print(f"\n  written -> {args.out}")


if __name__ == "__main__":
    main()
