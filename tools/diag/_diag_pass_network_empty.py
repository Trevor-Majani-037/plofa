"""WHY IS THE HOME TEAM'S PASS NETWORK EMPTY? (from the export, no match)

The export's pass-network plot draws nothing for the home team while the
away team draws fine — in the SAME export, from the SAME match.  The plot
filters the accumulator's stats by `minutes_played > 0` and then takes each
player's average position from `PassMatrix.build(team, timeline)`.

This reads the exported JSON (so it costs no match) and asks, in order:

  * are there PASS events for the home team at all?
  * does `PassMatrix.build` see them?
  * does `average_position` return anything for the home players?
  * what are the home players' `minutes_played`?

    .venv\\Scripts\\python.exe tools\\diag\\_diag_pass_network_empty.py \\
        plofa_output/_v7_gate/Probe_FC_vs_Rival_FC_MD3.json
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pass_network import PassMatrix  # noqa: E402


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events") or data.get("timeline") or []
    print(f"{path}: {len(events)} events")
    if events:
        print(f"  the first event's keys: {sorted(events[0].keys())}")
        print(f"  the first event: {json.dumps(events[0])[:300]}")
    teams = collections.Counter(e.get("team") for e in events)
    print(f"  teams: {dict(teams)}")
    per_team_passes = collections.Counter()
    per_team_events = collections.Counter()
    for e in events:
        t = e.get("team")
        per_team_events[t] += 1
        etype = str(e.get("event_type", e.get("type", ""))).upper()
        if etype in ("PASS", "THROUGH_BALL", "CROSS"):
            per_team_passes[t] += 1
    print(f"  pass-like events per team: {dict(per_team_passes)}")
    print(f"  all events per team:       {dict(per_team_events)}")

    for team in teams:
        pm = PassMatrix.build(team, events)
        names = sorted({e.get("player") for e in events
                        if e.get("team") == team and e.get("player")})
        got = {n: pm.average_position(n) for n in names}
        with_pos = {n: v for n, v in got.items() if v}
        print(f"\n  {team}: {len(names)} players, "
              f"{len(with_pos)} with an average position")
        for n in names:
            print(f"      {n:<22} {got[n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
