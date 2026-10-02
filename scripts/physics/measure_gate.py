"""
MEASURE: what do the physics gates actually do to a match?
============================================================

The honest answer to "is it fixable, and will the results be good?" cannot be
asserted. It has to be measured.

Why the measurement is split in two
-----------------------------------
PLOFA's engine is **not seed-reproducible**: the same seed gives different
matches (a player-iteration ordering divergence, diagnosed and not yet fixed).
So comparing one match with physics on against one with it off confounds the
gates with that noise, and a single fixture tells you nothing.

  (A) THE GATES' OWN COUNTERS — how often each fired, how hard, and how many
      blocks they removed. This isolates the mechanism and is nearly noise-free,
      because it counts decisions rather than outcomes.

  (B) THE AGGREGATE — pass completion, interceptions, shots and goals over
      several fixtures, with the run-to-run spread shown, so the effect can be
      compared against the noise it has to beat.

Read (A) first. If (A) shows the gates firing on a sensible fraction of
decisions and (B) shows a change smaller than the spread, the honest conclusion
is "realistic improvement, no measurable disruption" — not "better results".

Run:  python physics/measure_gate.py     (GATE_FIXTURES=n to change the count)
"""
from __future__ import annotations

import collections
import math
import os
import random
import statistics
import sys
import time as _wallclock

# Reported, not set. os.environ["PYTHONHASHSEED"] has no effect on a running
# interpreter — string hashing is fixed at startup — so setting it here would
# only look like control. It has to come from the shell.
HASH_SEED = os.environ.get("PYTHONHASHSEED")
if HASH_SEED is None:
    print("WARNING: PYTHONHASHSEED is not set. The engine's perception seeds")
    print("         are derived from hash() on player names, so the two")
    print("         conditions compared below would NOT be like for like.")
    print('         Set it in the shell:  $env:PYTHONHASHSEED="0"')
    print()

sys.path.insert(0, r"D:\PLOFA\plofa")

FIXTURES = int(os.environ.get("GATE_FIXTURES", "4"))


def _events(result) -> list:
    """The match's events.

    ``MatchResult`` exposes them as ``timeline`` — a list of ``MatchEvent``.
    An earlier version of this script read ``result.events``, which does not
    exist, so every count came back zero and section (B) printed a table of
    0.0 ± 0.0 that looked like a result rather than a bug. Worth recording:
    a measurement that reports all zeros is almost always measuring nothing.
    """
    return list(getattr(result, "timeline", None) or [])


def _event_counts(result) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for ev in _events(result):
        counts[getattr(ev.event_type, "name", "?")] += 1
    return counts


def _stats(result, counts) -> dict:
    passes = (counts["PASS"] + counts["PROGRESSIVE_PASS"])
    completed = sum(1 for ev in _events(result)
                    if getattr(ev.event_type, "name", "")
                    in ("PASS", "PROGRESSIVE_PASS")
                    and getattr(ev, "outcome", False))
    return {
        "score": f"{getattr(result, 'home_goals', '?')}-"
                 f"{getattr(result, 'away_goals', '?')}",
        "passes": passes,
        "completed": completed,
        "completion": (completed / passes * 100.0) if passes else 0.0,
        "interceptions": counts["INTERCEPTION"],
        "shots_on": counts["SHOT_ON_TARGET"],
        "shots_off": counts["SHOT_OFF_TARGET"],
        "goals": counts["GOAL"],
    }


def _gate_counters() -> collections.Counter:
    from physics import adapter as A
    return A._GATE_COUNTERS


def run_condition(enabled: bool, label: str, fixtures, base_seed: int = 20260927):
    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController
    from datetime import date

    from physics import adapter as A
    A._GATE_COUNTERS.clear()

    loader = get_loader()
    rows = []
    for n, (home, away) in enumerate(fixtures):
        raw = {c: loader.build_matchday_squad(c) for c in (home, away)}
        squads = {c: SquadBuilder.build(
            team_name=c, starters=raw[c]["starters"],
            substitutes=raw[c]["substitutes"],
            team_superstars=raw[c]["superstars"],
            set_piece_takers=raw[c]["sp_takers"]) for c in (home, away)}
        profs = {c: _resolve_team_profile(c, raw[c]["formation"],
                                           is_home=(c == home))
                 for c in (home, away)}

        # SEED THE SAME WAY IN BOTH CONDITIONS, AND RESET IT PER FIXTURE.
        #
        # An earlier version of this harness never called random.seed() at all,
        # so the baseline ran and the physics run then continued from wherever
        # the global stream happened to be. The deltas it reported — passes -169
        # while the gate had only evaluated 69 block opportunities — were read
        # as "engine noise" and blamed on a determinism bug in the engine.
        #
        # The engine was fine. `physics/probe_determinism.py` shows three runs
        # of the same seed in one process, and three fresh processes, all
        # producing a byte-identical timeline (3,283 events, sha e59c7889d1dbad5a).
        # The noise was entirely self-inflicted, by a harness that was not
        # controlling its own experiment.
        random.seed(base_seed + n)

        cfg = MatchConfig(home_team=home, away_team=away,
                          match_date=date(2026, 9, 8), physics_enabled=enabled)
        eng = MatchEngine(cfg, profs[home], profs[away])
        for c in (home, away):
            eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
        subs = [x for c in (home, away) for x in squads[c]["substitutes"]]
        eng.set_stamina_controller(SubstitutionController(
            home_team=home, away_team=away,
            home_subs_bench=subs, away_subs_bench=subs))
        result = eng.simulate()
        counts = _event_counts(result)
        rows.append(_stats(result, counts))
    return rows


def main() -> int:
    from roster_loader import get_loader
    loader = get_loader()
    clubs = sorted(loader.get_all_clubs())
    fixtures = [(clubs[i], clubs[i + 1]) for i in range(0, FIXTURES * 2, 2)]

    print("=" * 78)
    print("MEASURING THE PHYSICS GATES")
    print("=" * 78)
    print(f"  fixtures : {len(fixtures)}")
    print(f"  clubs    : {', '.join(c for p in fixtures for c in p)}")
    print()

    results = {}
    timings = {}
    for enabled, label in ((False, "BASELINE (physics off)"),
                           (True, "PHYSICS GATES on")):
        t0 = _wallclock.perf_counter()
        rows = run_condition(enabled, label, fixtures)
        elapsed = _wallclock.perf_counter() - t0
        results[enabled] = rows
        timings[enabled] = elapsed
        print(f"  {label:<26} {elapsed:6.1f}s")
        for (h, a), r in zip(fixtures, rows):
            print(f"    {h[:18]:<19} v {a[:18]:<19} {r['score']:>5}   "
                  f"pass {r['completed']:>3}/{r['passes']:<3} "
                  f"({r['completion']:5.1f}%)  int {r['interceptions']:>2}  "
                  f"shots {r['shots_on']:>2}/{r['shots_off']:<2}  "
                  f"goals {r['goals']}")
        print()

    if timings[True] and timings[False]:
        print(f"  cost of the physics: {timings[True] / timings[False]:.2f}x "
              f"wall clock ({timings[False]:.0f}s -> {timings[True]:.0f}s) for "
              f"{len(fixtures)} matches")
        print()

    # ── THE CONTROL ────────────────────────────────────────────────
    # Run the BASELINE twice, with physics off both times. If the engine and
    # this harness are both under control, the two runs are identical. If they
    # are not, then any difference attributed to the gates in section B is
    # really the harness.
    #
    # This is one extra match rather than a full extra condition, and it is the
    # check that would have caught the unseeded-`random` bug immediately
    # instead of two measurement runs later.
    print("=" * 78)
    print("(C) CONTROL — the baseline run twice, physics off both times")
    print("=" * 78)
    control_a = run_condition(False, "control A", fixtures[:1])
    control_b = run_condition(False, "control B", fixtures[:1])
    same = control_a == control_b
    for label, rows in (("control A", control_a), ("control B", control_b)):
        r = rows[0]
        print(f"  {label}: {r['score']:>5}  pass {r['completed']}/{r['passes']} "
              f"({r['completion']:.1f}%)  int {r['interceptions']}  "
              f"shots {r['shots_on']}/{r['shots_off']}  goals {r['goals']}")
    print()
    if same:
        print("  Identical. The engine and this harness are both under control,")
        print("  so section B's deltas are attributable to the gates.")
    else:
        print("  !! DIFFERENT. The comparison in section B is not trustworthy —")
        print("     the difference is the harness, not the physics.")
    print()

    counters = _gate_counters()
    print("=" * 78)
    print("(A) THE GATES' OWN COUNTERS — the mechanism, isolated")
    print("=" * 78)
    total = sum(counters.values())
    if not total:
        print("  the gates never fired")
    else:
        for key, n in sorted(counters.items()):
            share = n / total * 100.0
            print(f"  {key:<34} {n:>6}  {share:5.1f}%")
    # Sanity: a gate that never fires is a gate that is wired up but doing
    # nothing, and a gate that fires on everything is a gate that is about to
    # ruin a season. Both are worth seeing stated rather than inferred.
    #
    # The two gates' keys are kept strictly apart. An earlier version summed
    # every "changed" key regardless of which gate it belonged to, and
    # cheerfully reported "keeper saves: 2 considered, 48 changed (2400%)" —
    # which was pass-block arithmetic added to a keeper total. A summary that
    # cannot be a real number should be read as a bug, so it is now impossible
    # to produce one by accident.
    GATE_KEYS = {
        "pass blocks": ("considered",
                        ("attenuated", "impossible", "driven_to_floor")),
        "keeper saves": ("gk_considered",
                         ("gk_attenuated", "gk_unreachable", "gk_no_time")),
    }
    for label, (considered_key, changed_keys) in GATE_KEYS.items():
        considered = counters.get(considered_key, 0)
        if not considered:
            print(f"  -> {label}: never evaluated — the gate did not fire at all")
            continue
        changed = sum(counters.get(k, 0) for k in changed_keys)
        pct = changed / considered * 100.0
        print(f"  -> {label}: {considered} evaluated, {changed} changed "
              f"({pct:.1f}%)")
    print()

    print("=" * 78)
    print("(B) AGGREGATE, against the spread the effect has to beat")
    print("=" * 78)

    def agg(rows, key):
        vals = [r[key] for r in rows]
        mean = sum(vals) / len(vals) if vals else 0.0
        return mean, (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    # A sanity check on the measurement itself, before any of it is believed.
    # Every metric reading zero means the script found no events, not that the
    # gates had no effect — and those look identical in a table.
    total_events = sum(r["passes"] + r["goals"] + r["interceptions"]
                       for r in results[False] + results[True])
    if total_events == 0:
        print("  !! NO EVENTS FOUND — section B is measuring nothing.")
        print("     This is a bug in this script, not a finding about the gates.")
        return 1

    print(f"  events counted across both conditions: {total_events}")
    print()
    print(f"  {'metric':<16} {'off (mean±sd)':>22} {'on (mean±sd)':>22} {'delta':>10}")
    for key in ("passes", "completed", "interceptions", "shots_on",
                "goals", "completion"):
        off_v, off_sd = agg(results[False], key)
        on_v, on_sd = agg(results[True], key)
        d = on_v - off_v
        print(f"  {key:<16} {off_v:>12.1f} ± {off_sd:<7.1f} "
              f"{on_v:>12.1f} ± {on_sd:<7.1f} {d:>+10.1f}")
    print()
    print("  Read this honestly: these are per-match means. The control in")
    print("  section C is what licenses taking the deltas seriously — without a")
    print("  repeatable baseline, a delta is indistinguishable from a difference")
    print("  in circumstances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
