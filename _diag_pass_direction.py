"""IS `pass_direction` ACTUALLY WRONG? (2026-10-02) — verify before "fixing"

AGENTS.md carries a standing finding: "`pass_direction` metadata is
uncorrelated with geometry ... the exporter stamps it wrong ~2/3 of the time."
That was measured, but the code that produces it looks CORRECT: `event_chain.py`
computes

    pass_advance = end_px - x
    if not attacks_right:
        pass_advance = -pass_advance

and hands that to `classify_pass(..., signed_dx=pass_advance,
attacks_right=attacks_right)`. The direction is therefore a pure function of the
delivery geometry IN THE PASSING TEAM'S FRAME.

The signature that produced the finding — marginals match, assignment looks
random, "sideways" unaffected — is exactly what you get if the CHECK flips the
sign for only one team. "Sideways" is a |dx| band and cannot be affected by a
sign error at all, which is why it was the only category that agreed.

So this probe measures the same thing the right way round: derive the true
forward displacement using each team's OWN attack direction (home right, away
left, and whatever the half-time flip does), and compare.

Run: .venv\\Scripts\\python.exe _diag_pass_direction.py [n]
"""
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from match_engine import EventType
from pass_classifier import classify_direction

from _diag_chance_coords import build_pair

PASS_TYPES = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
    EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS,
}


def analyse(res, eng=None):
    tl = res.timeline
    home = res.config.home_team
    away = res.config.away_team
    pe = getattr(eng, "position_engine", None)
    tar = dict(getattr(pe, "team_attacks_right", {}) or {})

    # What attack direction does the engine itself believe, per (team, minute)?
    dirs = Counter()
    for e in tl:
        if e.event_type not in PASS_TYPES or e.minute is None:
            continue
        if e.end_x is None:
            continue
        ar = tar.get(e.team)
        dirs[(e.team, "H1" if e.minute < 45 else "H2", ar)] += 1

    agree = disagree = 0
    naive_agree = naive_disagree = 0
    confusion = Counter()
    per_team = Counter()
    for e in tl:
        if e.event_type not in PASS_TYPES:
            continue
        md = e.metadata or {}
        label = md.get("pass_direction")
        if not label or e.location_x is None or e.end_x is None:
            continue
        raw_dx = e.end_x - e.location_x
        ar = tar.get(e.team, e.team == home)
        true_dx = raw_dx if ar else -raw_dx
        truth = classify_direction(true_dx)
        per_team[(e.team, "home" if e.team == home else "away", ar)] += 1
        if truth == label:
            agree += 1
        else:
            disagree += 1
            confusion[(label, truth)] += 1
        # What the ORIGINAL measurement must have done: no flip at all.
        naive = classify_direction(raw_dx)
        if naive == label:
            naive_agree += 1
        else:
            naive_disagree += 1

    return {
        "passes": agree + disagree,
        "agree": agree, "disagree": disagree,
        "naive_agree": naive_agree, "naive_disagree": naive_disagree,
        "confusion": confusion, "dirs": dirs, "per_team": per_team,
        "tar": tar, "home": home, "away": away,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    tot = Counter()
    conf, dirs, ptot = Counter(), Counter(), Counter()
    for i in range(n):
        h, a = pairs[i % len(pairs)]
        random.seed(3000 + i)
        print(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
        eng = build_pair(h, a)
        res = eng.simulate()
        r = analyse(res, eng)
        for k in ("passes", "agree", "disagree", "naive_agree", "naive_disagree"):
            tot[k] += r[k]
        conf += r["confusion"]
        dirs += r["dirs"]
        ptot += r["per_team"]
        print(f"      {res.home_goals}-{res.away_goals}  "
              f"passes {r['passes']}  agree {r['agree']}  disagree {r['disagree']}",
              flush=True)

    print("\n" + "=" * 74)
    print("  passes examined (with a label and both endpoints):", tot["passes"])
    print(f"  CORRECT check (flip for the away side):     "
          f"agree {tot['agree']}  DISAGREE {tot['disagree']}"
          f"   ({100.0*tot['disagree']/max(tot['passes'],1):.1f}% wrong)")
    print(f"  NAIVE check (no flip, as first measured):   "
          f"agree {tot['naive_agree']}  disagree {tot['naive_disagree']}"
          f"   ({100.0*tot['naive_disagree']/max(tot['passes'],1):.1f}% 'wrong')")
    print("\n  direction of play the engine believes, by (team, half):")
    for k, v in sorted(dirs.items(), key=lambda kv: str(kv[0])):
        print(f"    {k[0]:<18} {k[1]}  attacks_right={k[2]!s:<6} {v:>5}")
    print("\n  passes by (team, side, attacks_right):")
    for k, v in sorted(ptot.items(), key=lambda kv: str(kv[0])):
        print(f"    {k[0]:<18} {k[1]:<5} attacks_right={k[2]!s:<6} {v:>5}")
    if conf:
        print("\n  confusion (label on the event -> label from true geometry):")
        for (lab, truth), v in conf.most_common(20):
            print(f"    {str(lab):<10} -> {str(truth):<10} {v:>5}")


if __name__ == "__main__":
    main()
