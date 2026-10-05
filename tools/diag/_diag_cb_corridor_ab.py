"""DID THE CB CORRIDOR CHANGE THE PASS MAP? (the limit I said was unverified)

The change under test: `_cb_same_side_corridor` in attacking_matrix.py grants a
CB the same-side corridor bonus when passing to a pivot on his own side and
AHEAD of the ball. I shipped it having verified only that the predicate is
correct (8/8 unit cases) and that it FIRES (155 times in one match). A fire
count is not an outcome: 155 firings could all be losers.

This measures the thing that was actually unverified — the CB pass map.

  --arm before : `_cb_same_side_corridor` forced to False (pre-change behaviour)
  --arm after  : the shipped rule

Reported per arm, over CB-originated passes only:
  * same-side share  — the headline; this is the corridor
  * forward share    — does the corridor cost progression?
  * median advance   — did the ball actually go further forward
  * completion rate  — the cost of asking for more ambitious passes

CAVEAT, stated because it limits the comparison and is NOT hidden: the arms
play DIFFERENT matches from the first divergence onward (forcing a receiver
changes the whole downstream state). So these are aggregate rates, not paired
per-state comparisons, and the denominators differ. That is the same
limitation the TeamPressBrain counterfactual sweep accepted.

Run (one arm per process):
    .venv\\Scripts\\python.exe _diag_cb_corridor_ab.py --arm before
    .venv\\Scripts\\python.exe _diag_cb_corridor_ab.py --arm after
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

import attacking_matrix
from _diag_chance_coords import build_pair

PASS_TYPES = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS"}


def force_off():
    """Neutralise the new rule WITHOUT editing the file (byte-exact restore)."""
    attacking_matrix._cb_same_side_corridor = lambda *a, **kw: False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("before", "after"), required=True)
    ap.add_argument("--matches", type=int, default=2)
    args = ap.parse_args()
    if args.arm == "before":
        force_off()

    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
    lines = []

    def p(m=""):
        print(m, flush=True)
        lines.append(m)

    p(f"  ARM = {args.arm}   ({args.matches} matches)")
    agg = Counter()
    adv = []
    pos_of_seen = {}

    for i in range(args.matches):
        h, a = pairs[i % len(pairs)]
        random.seed(4000 + i)
        res = build_pair(h, a).simulate()
        home = res.config.home_team
        tl = res.timeline

        # `event.player` is the player's NAME ("Ednar Pekisc"), NOT their
        # position. A first draft filtered `player.startswith("CB")` and
        # captured ZERO CB passes while reporting it as a result. Build the
        # name->position map from the squads instead -- same discipline as
        # `tracked_position` vs `get_position` in AGENTS.md.
        pos_of = {}
        for _team, squad in (res.squads or {}).items():
            # `squads[team]` is {"starters": [...], "substitutes": [...]},
            # NOT a flat list of players -- assumed once, captured zero passes.
            pool = []
            if isinstance(squad, dict):
                for key in ("starters", "substitutes"):
                    pool.extend(squad.get(key) or [])
            elif isinstance(squad, (list, tuple)):
                pool.extend(squad)
            for pl in pool:
                nm = getattr(pl, "name", None)
                if nm:
                    pos_of[nm] = str(getattr(pl, "position", ""))
        pos_of_seen.update({k: v for k, v in list(pos_of.items())[:400]})

        # Carrier-side reference comes from the PASS location itself; the
        # receiver's side is judged from the pass END, which is where the ball
        # was actually aimed (Checkpoint 21 aims at the receiver's live spot).
        for e in tl:
            if e.event_type.name not in PASS_TYPES:
                continue
            if not pos_of.get(e.player, "").startswith("CB"):
                continue
            if e.end_x is None or e.location_x is None or e.end_y is None:
                continue
            ly = e.location_y if e.location_y is not None else 34.0
            ry = e.end_y
            same = (ry < 34.0) == (ly < 34.0)
            agg["cb_passes"] += 1
            agg["same_side"] += 1 if same else 0
            if e.outcome:
                agg["completed"] += 1
            adv.append((e.end_x - e.location_x) if e.team == home
                       else (e.location_x - e.end_x))
        p(f"    [{i+1}/{args.matches}] {h} v {a}: {res.home_goals}-{res.away_goals}, "
          f"CB passes so far {agg['cb_passes']}")

    n = max(agg["cb_passes"], 1)
    p()
    p(f"  ARM = {args.arm}")
    p(f"    CB-originated passes   : {agg['cb_passes']}")
    if not agg["cb_passes"]:
        p("    NO CB PASSES CAPTURED -- inconclusive, not a negative.")
        p(f"    (names resolved to a position: {len(pos_of_seen)}; "
          f"sample {list(pos_of_seen)[:5]})")
        _dump(lines, args.arm)
        return
    p(f"    SAME-SIDE share        : {100.0*agg['same_side']/n:5.1f}%   <- headline")
    p(f"    completion rate        : {100.0*agg['completed']/n:5.1f}%")
    if adv:
        s = sorted(adv)
        fwd = sum(1 for d in adv if d > 0)
        p(f"    forward share          : {100.0*fwd/len(adv):5.1f}%")
        p(f"    advance median {s[len(s)//2]:+.1f} m   mean {sum(adv)/len(adv):+.1f} m")
    p()
    p("  Read: SAME-SIDE up = the corridor is forming. If it rose while")
    p("  completion or the forward share FELL, the extra ambition is being")
    p("  paid for in turnovers -- which would make this a worse engine, not a")
    p("  more realistic one.")

    _dump(lines, args.arm)


def _dump(lines, arm):
    out = f"_diag_cb_corridor_{arm}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()