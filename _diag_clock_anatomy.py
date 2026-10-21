"""HOW THE MATCH'S 96 MINUTES ARE SPENT — the clock's anatomy.

The fix for the compressed timeline needs the split between:
  * LIVE time (the chains' windows — the ball's motion today), and
  * DEAD time (the off-ball integrator phase between possessions),

because the fix REDISTRIBUTES the clock rather than adding to it: real
football's possession is flight + decision + execution, and PLOFA's total
is already football-plausible (96.4 min).  The time exists; it is in the
wrong places.

It also measures the chains themselves: how many events each one carries,
how long each window is, and how much of a window each event gets.

Run:  .venv\\Scripts\\python.exe -u _diag_clock_anatomy.py
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

    ch = res.chronology
    print(f"=== THE CLOCK'S ANATOMY (one match) ===")
    print(f"total clock:        {ch.match_duration_s:,.1f} s "
          f"({ch.match_duration_s/60:.1f} min)")
    print(f"measured PLAY time: {ch.measured_play_s:,.1f} s "
          f"({ch.measured_play_s/60:.1f} min)")
    print(f"DEAD time:          {ch.dead_time_s:,.1f} s "
          f"({ch.dead_time_s/60:.1f} min)")
    print(f"chains:             {ch.n_chains}")
    print(f"unmarked events:    {ch.n_unmarked_events}")
    share = ch.measured_play_s / max(ch.match_duration_s, 1e-9)
    print(f"live share:         {share*100:.1f}% of the match")
    print(f"real football: ~55-60 min of play in a ~96 min match "
          f"(~60% live)")

    marks = eng._chain_clock_marks
    print(f"\n=== THE CHAINS ===")
    sizes = [len(m.events) for m in marks if m.events]
    spans = [m.end_clock - m.start_clock for m in marks if m.events]
    print(f"chains with events: {len(sizes)}")
    if sizes:
        print(f"events per chain: mean {statistics.mean(sizes):.2f} "
              f"median {statistics.median(sizes):.0f} max {max(sizes)}")
        hist = Counter(sizes)
        print("  size histogram:",
              ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())[:10]))
    if spans:
        ss = sorted(spans)
        print(f"chain window (s): min {ss[0]:.2f} / median "
              f"{statistics.median(ss):.2f} / p90 {ss[int(len(ss)*0.9)]:.2f} "
              f"/ max {ss[-1]:.2f}")
        print(f"  total window: {sum(spans):,.1f} s")

    # the per-event share of a chain's window
    per_event = []
    for m in marks:
        n = len(m.events)
        if n >= 2 and m.end_clock > m.start_clock:
            per_event.append((m.end_clock - m.start_clock) / (n - 1))
    if per_event:
        ps = sorted(per_event)
        print(f"\nseconds per event inside a chain: median "
              f"{statistics.median(ps):.3f} / p90 {ps[int(len(ps)*0.9)]:.3f} "
              f"/ max {ps[-1]:.3f}")
        print(f"  (real football: a possession's actions are 2-4 s apart)")

    # how much dead time is there between the chains?
    print(f"\n=== THE GAPS BETWEEN CHAINS ===")
    gaps = []
    for a, b in zip(marks, marks[1:]):
        g = b.start_clock - a.end_clock
        if g > 0:
            gaps.append(g)
    if gaps:
        gs = sorted(gaps)
        print(f"gap between chains (s): n={len(gaps)} / min {gs[0]:.2f} "
              f"/ median {statistics.median(gs):.2f} / p90 "
              f"{gs[int(len(gs)*0.9)]:.2f} / max {gs[-1]:.2f}")
        print(f"  total gap time: {sum(gaps):,.1f} s "
              f"({sum(gaps)/max(ch.match_duration_s,1e-9)*100:.1f}% of the match)")
        print(f"  (real football: the ball is out of play ~10-20 s at a time)")

    # the event type mix, so the duration table is designed against the
    # events that actually occur
    print(f"\n=== THE EVENT MIX (what a duration table must cover) ===")
    mix = Counter(getattr(e.event_type, "name", "?") for e in res.timeline)
    for name, n in mix.most_common(18):
        print(f"  {name:<24}{n:>6}")


if __name__ == "__main__":
    main()
