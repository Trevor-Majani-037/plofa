"""
HOW MUCH OF THE MATCH IS THE FABRICATED SHOT PIPELINE? (2026-10-02)
==================================================================
`event_chain` has a CAUSAL shot pipeline: PossessionChain knows which man
decided to shoot (`shoot_player`) and which man passed to him
(`shoot_assister`), and AttackChain resolves the geometry from those.

`match_engine._simulate_shot_sequence` is a SECOND, non-causal one:

    shooter = self._pick_player(team, preferred_positions=['ST','CF','LW','RW','CAM'])
    creator = self._pick_player(team, preferred_positions=['CAM','CM','LW','RW','CDM'], ...)
    loc     = self._shot_location(zone, attacks_right)

Every one of those is an independent random draw. The shooter is not the man
who had the ball, the creator never touched it, and the location is a point
sampled from a ZONE NAME with no reference to any player or the ball. It then
emits CHANCE_CREATED, SHOT_ON_TARGET, SAVE and GOAL at that invented point.

It fires from the `shot_prob` funnel at `match_engine.py:5238`, gated only on
`not poss_result.shot_taken and in_final_third` — so it is a parallel goal
route, not a fallback.

This counts how many shots and goals each pipeline produces, and whether the
fabricated pipeline's coordinate agrees with the man it names.

    .venv\\Scripts\\python.exe _diag_shot_pipeline_split.py [n]
"""

import math
import random
import sys

from _diag_chance_coords import build_pair

PAIRS = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
         ("Justice", "Triumpher")]

SHOT_TYPES = {
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SHOT_SAVED",
    "SHOT_POST", "SHOT_WOODWORK", "SHOT_MISS", "GOAL",
}

# event object id -> provenance
TAGS = {}
_lock = __import__("threading").Lock()


def _install():
    from match_engine import MatchEngine
    from event_chain import AttackChain

    # ── the fabricated pipeline ──
    # `_simulate_shot_sequence` emits through `self._emit_event`, so swap that
    # for a spy for the duration of the call and tag whatever comes out. This
    # attributes by ACTUAL EMISSION rather than by guessing which events a
    # caller went on to absorb.
    fake = MatchEngine._simulate_shot_sequence

    def instrumented(self, minute, attacking_team, *a, **kw):
        made = []
        real_emit = self._emit_event

        def spy(**kwargs):
            ev = real_emit(**kwargs)
            if ev is not None:
                made.append(ev)
            return ev

        self._emit_event = spy
        try:
            return fake(self, minute, attacking_team, *a, **kw)
        finally:
            self._emit_event = real_emit
        for ev in made:
            if ev.event_type.name in SHOT_TYPES:
                TAGS[id(ev)] = "FABRICATED"

    MatchEngine._simulate_shot_sequence = instrumented

    # ── the causal pipeline: tag its events ──
    causal = AttackChain.generate.__func__

    def causal_wrapper(cls, minute, attacking_team, *a, **kw):
        res = causal(cls, minute, attacking_team, *a, **kw)
        for ev in getattr(res, "events", []) or []:
            if ev.event_type.name in SHOT_TYPES:
                TAGS[id(ev)] = "CAUSAL"
        return res

    AttackChain.generate = classmethod(causal_wrapper)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    _install()
    counts = {"CAUSAL": 0, "FABRICATED": 0, "untagged": 0}
    goals = {"CAUSAL": 0, "FABRICATED": 0, "untagged": 0}
    rows = []

    for i in range(n):
        h, a = PAIRS[i % len(PAIRS)]
        random.seed(3000 + i)
        print(f"  [{i+1}] {h} v {a} ...", flush=True)
        eng = build_pair(h, a)
        res = eng.simulate()

        for e in (getattr(res, "timeline", []) or []):
            if e.event_type.name not in SHOT_TYPES:
                continue
            tag = TAGS.get(id(e), "untagged")
            counts[tag] = counts.get(tag, 0) + 1
            if e.event_type.name == "GOAL":
                goals[tag] = goals.get(tag, 0) + 1
                rows.append((tag, e.team, e.player, e.secondary_player,
                             round(e.location_x, 1), round(e.location_y, 1)))
        print(f"      score {res.home_goals}-{res.away_goals}", flush=True)

    tot = sum(counts.values())
    gtot = sum(goals.values())
    print("\n" + "=" * 74)
    print(f"=== SHOT PIPELINE SPLIT over {n} match(es) ===\n")
    print(f"  shots+goals:  causal {counts['CAUSAL']:3d}   "
          f"FABRICATED {counts['FABRICATED']:3d}   untagged {counts['untagged']:3d}"
          f"   (n={tot})")
    print(f"  GOALS ONLY:   causal {goals['CAUSAL']:3d}   "
          f"FABRICATED {goals['FABRICATED']:3d}   untagged {goals['untagged']:3d}"
          f"   (n={gtot})")
    if gtot:
        pf = 100.0 * goals["FABRICATED"] / gtot
        print(f"\n  >>> {pf:.0f}% OF ALL GOALS COME FROM THE FABRICATED PIPELINE")

    print("\n  EVERY GOAL, BY SOURCE (assist is what ships in the Goals sheet):")
    for tag, team, scorer, assist, x, y in rows:
        print(f"    {tag:<12} {team:<13} scorer {str(scorer):<18} "
              f"assist {str(assist):<18} at ({x}, {y})")
    return counts, goals, rows


if __name__ == "__main__":
    main()