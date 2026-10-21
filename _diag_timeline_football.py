"""THE TIMELINE'S FOOTBALL REALISM — is the within-possession timing plausible?

The export now flows (monotonic, real clock).  But the clock's sub-second
resolution is a MODEL: `chronograph()` spreads each chain's window evenly
across the chain's events.  So the question is football-wise:

  * how many events does a real match have, and how far apart?
  * how much of PLOFA's timeline is inside a chain (where the times are
    spread) versus between chains (where they are the clock's own)?

THE BENCHMARK, football-wise:
  * a Premier League match has ~1,050-1,150 PASSES in total (both teams),
    over ~96 minutes of match time — so **one pass every ~5 seconds**;
  * a possession's actions are NOT simultaneous: a pass's flight, then a
    reception, then the next action.  A player takes 0.5-1.5 s to decide.

This probe measures PLOFA's timeline against those facts.

Run:  .venv\\Scripts\\python.exe -u _diag_timeline_football.py
"""
from __future__ import annotations

import io
import os
import random
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    from tools.diag._diag_watch import build
    eng, _, _ = build()
    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    tl = [e for e in res.timeline
          if getattr(e, "match_clock_s", None) is not None]
    tl.sort(key=lambda e: e.match_clock_s)

    print(f"events with a clock: {len(tl)}")
    span = tl[-1].match_clock_s - tl[0].match_clock_s
    print(f"span: {span:.1f} s = {span/60:.1f} min")

    # ── how many passes, and how far apart? ────────────────────────────
    passes = [e for e in tl
              if getattr(e.event_type, "name", "") in
              ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
               "CROSS_ATTEMPT", "CROSS_SUCCESS")]
    print(f"\n=== THE PASS COUNT — the football-meaningful one ===")
    print(f"pass-like events: {len(passes)}")
    print(f"real PL: ~1,050-1,150 per match (both teams)")
    if len(passes) > 2:
        gaps = [b.match_clock_s - a.match_clock_s
                for a, b in zip(passes, passes[1:])]
        gaps_sorted = sorted(gaps)
        print(f"interval between consecutive passes (s): "
              f"min {gaps_sorted[0]:.2f} / p10 {gaps_sorted[len(gaps_sorted)//10]:.2f}"
              f" / median {statistics.median(gaps):.2f}"
              f" / p90 {gaps_sorted[int(len(gaps_sorted)*0.9)]:.2f}"
              f" / max {gaps_sorted[-1]:.2f}")
        print(f"real football: one pass every ~5 s")
        print(f"mean pass interval: {span/max(len(passes),1):.2f} s")

    # ── the chain structure: how many events share a timestamp window? ──
    print(f"\n=== THE CHAIN STRUCTURE ===")
    # events inside the same 0.05 s are, to within the spread, the same
    # instant; count the runs of events closer than 0.05 s
    runs = []
    cur = 1
    for a, b in zip(tl, tl[1:]):
        if (b.match_clock_s - a.match_clock_s) < 0.05:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    print(f"runs of events at the SAME instant: {len(runs)}")
    print(f"  events in them: {sum(runs)} of {len(tl)} "
          f"({100*sum(runs)/len(tl):.1f}%)")
    print(f"  mean run length: {statistics.mean(runs):.2f}, "
          f"max {max(runs)}")
    hist = Counter(runs)
    print("  run length histogram:",
          ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())[:8]))

    # ── the gaps between the runs ──────────────────────────────────────
    gap_of_run = []
    i = 0
    while i < len(tl):
        j = i
        while (j + 1 < len(tl)
               and tl[j + 1].match_clock_s - tl[j].match_clock_s < 0.05):
            j += 1
        if j + 1 < len(tl):
            gap_of_run.append(tl[j + 1].match_clock_s - tl[j].match_clock_s)
        i = j + 1
    if gap_of_run:
        gs = sorted(gap_of_run)
        print(f"\n  gap between instants (s): median {statistics.median(gs):.2f}"
              f" / p90 {gs[int(len(gs)*0.9)]:.2f} / max {gs[-1]:.2f}")

    # ── what the export actually shows for the first 30 events ─────────
    print(f"\n=== THE FIRST 30 EXPORTED EVENTS ===")
    print(f"{'clock':>10} {'type':<22}{'player':<22}{'team':<18}")
    for e in tl[:30]:
        cs = e.match_clock_s
        print(f"{int(cs)//60:02d}:{cs%60:06.3f}  "
              f"{getattr(e.event_type,'name',''):<22}{e.player:<22}{e.team:<18}")


if __name__ == "__main__":
    main()
