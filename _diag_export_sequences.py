"""Does the export's `sequences` block carry a real time axis?

The `timeline` uses the random placeholder second (measured: median 16.6 s
from the truth).  The `sequences` block is the export's other time-bearing
structure, so it is worth knowing whether IT flows.

Usage:
    python _diag_export_sequences.py <path.json>
"""
import json
import sys


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    seqs = data.get("sequences") or []
    print(f"sequences: {len(seqs)}")
    if not seqs:
        return
    print("keys:", sorted(seqs[0].keys()))
    print()
    for s in seqs[:6]:
        print({k: s.get(k) for k in sorted(s.keys())
               if k not in ("events", "passes")})
    # any time-ish fields at all?
    time_keys = set()
    for s in seqs:
        for k in s:
            if any(t in k.lower() for t in ("time", "second", "minute",
                                            "clock", "stamp", "dur")):
                time_keys.add(k)
    print()
    print("time-ish keys on a sequence:", sorted(time_keys))
    # and inside a sequence's events?
    ev_keys = set()
    for s in seqs:
        for e in (s.get("events") or []):
            if isinstance(e, dict):
                ev_keys.update(e.keys())
    print("event keys inside a sequence:",
          sorted(k for k in ev_keys
                 if any(t in k.lower() for t in ("time", "second", "minute",
                                                "clock", "stamp", "dur"))))


if __name__ == "__main__":
    main()
