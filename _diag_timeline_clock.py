"""THE TIMELINE'S TIME AXIS: the placeholder second vs the TRUE chronology.

The question the user asked: in the export, is the timeline in seconds or
milliseconds (StatsBomb style), and why does the first ten seconds not flow?

THE ANSWER, from the code:
  * `MatchEvent.second` is a PLACEHOLDER.  `event_chain.make_event` stamps
    `second=kwargs.pop("second", random.randint(0, 59))` — a random draw
    unless the caller passes one (only two call sites do).  The engine's own
    docstring on `MatchEngine.chronograph` says it: "The timeline's
    MatchEvent.second is only a bucket (mostly a placeholder random value)".
  * The TRUE clock exists: `MatchEngine.chronograph()` rebuilds every event's
    real time from `state.match_clock_s` and the `_chain_clock_marks` (the
    window each chain actually occupied), and it is carried on
    `MatchResult.chronology` as `TimedEvent`s with a FRACTIONAL second, a
    true `match_clock_s`, and a `duration`.
  * The EXPORTER does not use it.  `plofa_export.py` reads `e.minute` and
    `e.second` — the placeholder.

This probe runs a real match and prints both axes side by side, so the
difference is visible rather than asserted.

Usage:
    python _diag_timeline_clock.py [--seed 777]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    import random
    import numpy as np
    import match_engine as ME
    import run_match as R
    from player_dna import SquadBuilder

    random.seed(args.seed)
    np.random.seed(args.seed)
    home = SquadBuilder.build(R.HOME_TEAM, list(R.HOME_STARTERS),
                              list(R.HOME_SUBS))
    away = SquadBuilder.build(R.AWAY_TEAM, list(R.AWAY_STARTERS),
                              list(R.AWAY_SUBS))
    cfg = ME.MatchConfig(home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM,
                         match_date=R.MATCH_DATE, matchday=R.MATCHDAY,
                         season=R.SEASON, competition=R.COMPETITION,
                         venue=R.VENUE, stadium_capacity=R.CAPACITY,
                         referee=R.REFEREE, referee_strictness=R.STRICTNESS,
                         is_derby=R.IS_DERBY)
    eng = ME.MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home["starters"], home["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away["starters"], away["substitutes"])
    result = eng.simulate()

    tl = list(getattr(result, "timeline", []) or [])
    chron = getattr(result, "chronology", None)
    print(f"timeline events: {len(tl)}   chronology events: "
          f"{len(getattr(chron, 'events', []) or [])}")
    print(f"match_duration_s: {getattr(chron, 'match_duration_s', None)}")
    print()
    print("--- the first 18 events: PLACEHOLDER second vs the TRUE clock ---")
    print(f"{'#':>4} {'type':<22} {'minute':>7} {'2nd(EXPORTER)':>14} "
          f"{'TRUE 2nd':>10} {'clock_s':>10} {'dur':>7}")
    tes = list(getattr(chron, "events", []) or [])
    for i, e in enumerate(tl[:18]):
        t = getattr(getattr(e, "event_type", None), "name", str(getattr(e, "event_type", "")))
        te = tes[i] if i < len(tes) else None
        print(f"{i:>4} {t:<22} {getattr(e,'minute',0):>7} "
              f"{getattr(e,'second',0):>14} "
              f"{(getattr(te,'second',0) if te else float('nan')):>10.1f} "
              f"{(getattr(te,'match_clock_s',0) if te else float('nan')):>10.2f} "
              f"{(getattr(te,'duration',0) if te else float('nan')):>7.2f}")

    # how far apart are the two axes?
    if tes:
        ex = [getattr(e, "second", 0) for e in tl]
        tr = [t.second for t in tes]
        diffs = [abs(a - b) for a, b in zip(ex, tr)]
        diffs.sort()
        print()
        print("--- the placeholder vs the true second, over the whole match ---")
        print(f"  median |difference| {diffs[len(diffs)//2]:.1f} s   "
              f"max {diffs[-1]:.1f} s")
        print(f"  events whose placeholder is within 1 s of the truth: "
              f"{sum(1 for d in diffs if d <= 1.0)}/{len(diffs)} "
              f"({100.0*sum(1 for d in diffs if d <= 1.0)/len(diffs):.1f}%)")
        # does the TRUE clock flow?
        clock = [t.match_clock_s for t in tes]
        deltas = [b - a for a, b in zip(clock, clock[1:])]
        neg = sum(1 for d in deltas if d < -1e-9)
        dl = sorted(deltas)
        print()
        print("--- the TRUE clock's flow ---")
        print(f"  first {clock[0]:.2f}   last {clock[-1]:.2f}   "
              f"span {clock[-1]-clock[0]:.1f} s")
        print(f"  out-of-order steps: {neg}")
        print(f"  delta min {dl[0]:.2f}  median {dl[len(dl)//2]:.2f}  max {dl[-1]:.2f}")
        print(f"  the first 12 deltas: {[round(d,2) for d in deltas[:12]]}")
        print(f"  as StatsBomb-style milliseconds: {int(round(clock[0]*1000))} .. "
              f"{int(round(clock[-1]*1000))}")


if __name__ == "__main__":
    main()
