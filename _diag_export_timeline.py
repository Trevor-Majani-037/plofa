"""WHY DOES THE EXPORTED TIMELINE NOT FLOW?

The export fix stamps the TRUE clock on every event.  Three tests then failed:
  * the first event is not at kickoff,
  * the timeline goes BACKWARDS somewhere,
  * one event's exported second (59.999) disagreed with the chronology's
    (0.0), which smells like a rounding difference between the two.

This probe runs the match ONCE and dumps the whole clock sequence so the
answer can be READ rather than guessed at.

Run:  .venv\\Scripts\\python.exe -u _diag_export_timeline.py
"""
from __future__ import annotations

import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "_diag_export_timeline.json"


def _run():
    from tools.diag._diag_watch import build
    eng, _, _ = build()
    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real
    return eng, res


def main():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            doc = json.load(f)
    else:
        eng, res = _run()
        tl = []
        for i, e in enumerate(res.timeline):
            tl.append({
                "i": i,
                "type": getattr(getattr(e, "event_type", None), "name", "?"),
                "team": e.team,
                "player": e.player,
                "minute": e.minute,
                "second": e.second,
                "clock_s": getattr(e, "match_clock_s", None),
            })
        chron = []
        for t in (getattr(res, "chronology", None).events
                  if getattr(res, "chronology", None) else []):
            chron.append({
                "source_event_index": getattr(t, "source_event_index", -1),
                "minute": t.minute,
                "second": t.second,
                "match_clock_s": t.match_clock_s,
                "type": getattr(t, "event_type", ""),
            })
        doc = {"timeline": tl, "chronology": chron,
               "events": len(res.timeline)}
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)

    tl = doc["timeline"]
    print(f"timeline events: {len(tl)}   chronology events: {len(doc['chronology'])}")

    # ── 1. the first event ───────────────────────────────────────────────
    print("\n=== 1. THE FIRST EVENTS ===")
    for r in tl[:6]:
        print(f"  #{r['i']:<5} {r['type']:<18} {r['team']:<10} "
              f"min {r['minute']:<3} sec {r['second']:<3} clock_s {r['clock_s']}")

    # ── 2. the backwards steps ───────────────────────────────────────────
    print("\n=== 2. THE TIMELINE'S ORDER vs ITS CLOCK ===")
    back = []
    for i in range(1, len(tl)):
        a, b = tl[i - 1], tl[i]
        ca = a["clock_s"]
        cb = b["clock_s"]
        if ca is None or cb is None:
            continue
        if cb < ca:
            back.append((i, ca, cb, a, b))
    print(f"backwards steps: {len(back)}")
    for i, ca, cb, a, b in back[:12]:
        print(f"  at #{i}: {ca:.3f} -> {cb:.3f}  (back {ca-cb:.3f} s)")
        print(f"      before: #{a['i']} {a['type']:<18} {a['team']:<10} "
              f"min {a['minute']} sec {a['second']} clock {a['clock_s']}")
        print(f"      after : #{b['i']} {b['type']:<18} {b['team']:<10} "
              f"min {b['minute']} sec {b['second']} clock {b['clock_s']}")

    # ── 3. the rounding disagreement ─────────────────────────────────────
    print("\n=== 3. STAMP vs CHRONOLOGY (same event, two numbers) ===")
    by_index = {t["source_event_index"]: t for t in doc["chronology"]
                if t["source_event_index"] >= 0}
    diffs = []
    for r in tl:
        t = by_index.get(r["i"])
        if t is None or r["clock_s"] is None:
            continue
        if abs(t["match_clock_s"] - r["clock_s"]) > 0.0005:
            diffs.append((r, t))
    print(f"events where the stamp and the chronology disagree: {len(diffs)}")
    for r, t in diffs[:10]:
        print(f"  #{r['i']} {r['type']:<16} stamped {r['clock_s']:<12} "
              f"chronology {t['match_clock_s']}")

    # ── 4. how many events have NO clock at all ──────────────────────────
    print("\n=== 4. COVERAGE ===")
    none_clock = [r for r in tl if r["clock_s"] is None]
    print(f"events with no stamped clock: {len(none_clock)}")
    for r in none_clock[:12]:
        print(f"  #{r['i']} {r['type']:<18} {r['team']:<10} min {r['minute']}")

    # ── 5. the chronology's own order ────────────────────────────────────
    print("\n=== 5. IS THE CHRONOLOGY ORDERED? ===")
    cs = [t["match_clock_s"] for t in doc["chronology"]]
    cback = [(i, a, b) for i, (a, b) in enumerate(zip(cs, cs[1:])) if b < a]
    print(f"chronology backwards steps: {len(cback)}")
    for i, a, b in cback[:8]:
        print(f"  at {i}: {a:.3f} -> {b:.3f}")

    # ── 6. the timeline index vs the chronology index ────────────────────
    print("\n=== 6. INDEX ALIGNMENT ===")
    ti = [r["i"] for r in tl]
    ci = [t["source_event_index"] for t in doc["chronology"]]
    print(f"timeline indices covered by the chronology: "
          f"{len(set(ti) & set(ci))} of {len(ti)}")
    missing = sorted(set(ti) - set(ci))
    print(f"timeline events NOT in the chronology: {len(missing)}")
    for i in missing[:12]:
        r = tl[i]
        print(f"  #{i} {r['type']:<18} {r['team']:<10} min {r['minute']} "
              f"clock {r['clock_s']}")

    # ── 7. the exported second, as the exporter computes it ─────────────
    print("\n=== 7. THE EXPORTED SECOND ===")
    for r in tl:
        c = r["clock_s"]
        if c is None:
            continue
        exported = round(float(c) % 60.0, 3)
        t = by_index.get(r["i"])
        if t is not None and abs(exported - (t["match_clock_s"] % 60.0)) > 0.01:
            print(f"  #{r['i']} {r['type']:<16} stamped {c:<12} exported second "
                  f"{exported:<9} chronology says {t['match_clock_s'] % 60.0}")
            break


if __name__ == "__main__":
    main()
