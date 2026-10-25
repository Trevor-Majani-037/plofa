"""FAST STRIKER-PASS + FORWARD PASS GATE (2026-10-25)

One match, two metrics the project cares about:
  1. striker pass involvement (received / made) -- the ball-magnet check
  2. forward-pass % and median pass length -- the realism gate (AGENTS.md
     wants forward 35-40%, median pass length 15-18m).

Forward is defined as dx toward the attacking goal (home attacks +x, away
attacks -x), the same direction the matrix's `progress` uses.

Run: .venv\\Scripts\\python.exe tools\\diag\\_diag_striker_forward_gate.py [seed]
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from _diag_chance_coords import build_pair  # noqa: E402

PASS_EVENTS = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS",
               "PROGRESSIVE_PASS", "CROSS_ATTEMPT", "CROSS_SUCCESS",
               "CORNER_TAKEN", "FREEKICK_CROSS"}


def _positions(res):
    pos = defaultdict(dict)
    for team, data in (getattr(res, "squads", None) or {}).items():
        for key in ("starters", "substitutes"):
            for row in (data.get(key, []) or []):
                n, p = getattr(row, "name", None), getattr(row, "position", None)
                if n and p:
                    pos[team][n] = p
    return pos


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    random.seed(seed)
    res = build_pair("Oxton", "Natrican").simulate()
    home = res.config.home_team
    pos = _positions(res)

    recv = Counter()       # (team, name)
    made = Counter()       # (team, name)
    recv_by_role = Counter()
    lengths = defaultdict(list)      # team -> pass lengths
    fwd = Counter()                  # team -> forward count
    tot = Counter()                  # team -> total count

    for e in res.timeline:
        if e.event_type.name not in PASS_EVENTS:
            continue
        t = e.team
        if e.player:
            made[(t, e.player)] += 1
        if e.secondary_player:
            recv[(t, e.secondary_player)] += 1
            recv_by_role[(t, pos.get(t, {}).get(e.secondary_player, "?"))] += 1
        if e.location_x is not None and e.end_x is not None:
            dx = e.end_x - e.location_x
            attacks_right = (t == home)
            forward = (dx > 0) if attacks_right else (dx < 0)
            if forward:
                fwd[t] += 1
            tot[t] += 1
            lengths[t].append(((e.end_x - e.location_x) ** 2 +
                               ((e.end_y or 34) - (e.location_y or 34)) ** 2) ** 0.5)

    print(f"\nscore {res.home_goals}-{res.away_goals}  (seed {seed})")
    for t in (home, res.config.away_team):
        p = pos.get(t, {})
        print(f"\n  === {t} ===")
        print("  --- STRIKER ---")
        for n, rp in p.items():
            if rp in ("ST", "CF"):
                m = made.get((t, n), 0)
                r = recv.get((t, n), 0)
                print(f"      {rp} {n:<20} made={m:>4}  recv={r:>4}  "
                      f"involvement={m + r:>4}")
        print("  --- passes RECEIVED by position ---")
        for (tt, role), c in sorted(recv_by_role.items(), key=lambda x: -x[1]):
            if tt == t:
                print(f"      {role:<4} {c}")
        L = sorted(lengths[t])
        med = L[len(L) // 2] if L else 0.0
        fp = 100.0 * fwd[t] / max(tot[t], 1)
        print(f"  forward-pass % = {fp:.1f}%   median pass length = {med:.1f} m")


if __name__ == "__main__":
    main()
