"""WHO ACCUMULATES THE PASSES? (2026-10-25)

User reports: "strikers have the top 2 most passes in a game". Before
reasoning about the fake-nine or any other tactical cause, measure it:
for each team in a match, count
  * PASSES MADE   (events where the player is the passer)
  * PASSES RECEIVED (events where the player is the receiver)

Then aggregate BY POSITION and list the top players. This tells us whether
the ball genuinely funnels through the strikers, and whether it is because
they receive a lot (the ball flows to them) or are responsible for a lot of
the next pass (they recycle it).

Observation-only; patches nothing. Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_passes_by_role.py [n_matches]
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict

# Make the project root importable regardless of cwd (the diag harness runs
# from D:\\PLOFA\\plofa\\tools\\diag, where match_engine is one level up).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from _diag_chance_coords import build_pair   # noqa: E402

PASS_EVENTS = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS",
               "PROGRESSIVE_PASS", "CROSS_ATTEMPT", "CROSS_SUCCESS",
               "CORNER_TAKEN", "FREEKICK_CROSS"}


def _positions_for(res):
    """Return {team: {player_name: position}} from the result squads."""
    pos = defaultdict(dict)
    squads = getattr(res, "squads", None) or {}
    for team, data in squads.items():
        if not isinstance(data, dict):
            continue
        for key in ("starters", "substitutes"):
            for row in data.get(key, []) or []:
                name = getattr(row, "name", None)
                p = getattr(row, "position", None)
                if name and p:
                    pos[team][name] = p
    return pos


def analyse(res, lines):
    def p(m=""):
        print(m)
        lines.append(m)

    pos = _positions_for(res)
    made = Counter()        # (team, name) -> passes made
    recv = Counter()        # (team, name) -> passes received
    made_by_role = Counter()
    recv_by_role = Counter()

    for e in res.timeline:
        if e.event_type.name not in PASS_EVENTS:
            continue
        t = e.team
        p_name = e.player
        if not p_name:
            continue
        pr = pos.get(t, {}).get(p_name, "?")
        made[(t, p_name)] += 1
        made_by_role[(t, pr)] += 1
        if e.secondary_player:
            rr = pos.get(t, {}).get(e.secondary_player, "?")
            recv[(t, e.secondary_player)] += 1
            recv_by_role[(t, rr)] += 1

    for t in (res.config.home_team, res.config.away_team):
        p(f"\n  === {t} ===")
        team_pos = pos.get(t, {})

        def _rows(counter):
            return [(n, counter.get((t, n), 0)) for n in team_pos]

        made_rows = sorted(_rows(made), key=lambda r: -r[1])
        recv_rows = sorted(_rows(recv), key=lambda r: -r[1])
        p("  --- passes MADE by player ---")
        for n, c in made_rows[:11]:
            p(f"      {team_pos.get(n,'?'):<4} {str(n):<22} {c}")
        p("  --- passes RECEIVED by player ---")
        for n, c in recv_rows[:11]:
            p(f"      {team_pos.get(n,'?'):<4} {str(n):<22} {c}")
        p("  --- passes MADE by POSITION (made / received) ---")
        for (role, rc) in sorted(made_by_role.items(), key=lambda r: -r[1]):
            if role[0] == t:
                p(f"      {role[1]:<4} made={rc}  recv={recv_by_role.get((role[0], role[1]), 0)}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    lines = []

    def p(m=""):
        print(m)
        lines.append(m)

    for i in range(n):
        h, a = pairs[i % len(pairs)]
        random.seed(3000 + i)
        p(f"\n########## [{i+1}/{n}] {h} v {a}  ##########")
        res = build_pair(h, a).simulate()
        p(f"score {res.home_goals}-{res.away_goals}")
        analyse(res, lines)

    with open("_diag_passes_by_role.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_passes_by_role.txt")


if __name__ == "__main__":
    main()
