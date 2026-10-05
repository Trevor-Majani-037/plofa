"""Does `pass_advance` agree with the geometry, in the export frame?

This matters more than a format question. `surrogate_collect.py:154` scores a
completed pass as `2.0 + min(pass_advance / 25, 1.5)` — the evolved brains have
been maximising that term since 2026-09-11. If `pass_advance`'s SIGN disagrees
with which way the ball actually travelled for one of the two teams, then half
the corpus taught the policy that going forward was going backwards.

`pass_advance` is computed in `event_chain.py` as
    pass_advance = end_px - x
    if not attacks_right: pass_advance = -pass_advance
so it is meant to be positive when the ball moves towards the acting team's
target. In the export frame the target is x=105 for the home side and x=0 for
the away side. So the expected sign of (to.x - at.x) is + for home, - for away.

Checked per team and per half, because both matter.
"""
import collections
import io
import json
import random
import sys

import plofa_export as px


def main():
    sys.path.insert(0, ".")
    from _diag_watch import build
    from _diag_plofa_export import roster_from

    eng, hr, ar = build()
    eng.enable_virtual_gps(0.1)
    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    doc = px.export_match(res, engine=eng, roster=roster_from(eng))
    H = doc["match"]["home"]
    A = doc["match"]["away"]

    buckets = collections.defaultdict(lambda: [0, 0])   # -> [agree, total]
    print(f"{'team':<11}{'half':>5}{'n':>7}{'agree':>8}{'rate':>8}   meaning")
    for e in doc["events"]:
        if e["type"] not in ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY",
                             "THROUGH_BALL", "CROSS_ATTEMPT",
                             "CROSS_SUCCESS"):
            continue
        p = e.get("pass") or {}
        adv = p.get("pass_advance")
        at, to = e.get("at"), e.get("to")
        if adv is None or not at or not to:
            continue
        if p.get("is_airborne") or p.get("is_long"):
            continue
        team = e.get("team")
        if team not in (H, A):
            continue
        actual = to[0] - at[0]
        expected_sign = 1.0 if team == H else -1.0
        agree = (actual * expected_sign) > 0 or abs(actual) < 1.0
        k = (team, e["half"])
        buckets[k][0] += int(agree)
        buckets[k][1] += 1

    for (team, half), (ok, n) in sorted(buckets.items()):
        rate = 100.0 * ok / n if n else 0.0
        mean = ""
        print(f"{team:<11}{half:>5}{n:>7}{ok:>8}{rate:>7.1f}%   {mean}")

    print("\nA rate near 50% means the sign is uncorrelated with reality.")
    print("A rate near 100% means pass_advance and the geometry agree, and")
    print("the surrogate has been rewarding real forward progress all along.")

    # Also: does is_progressive agree with the geometry's own 25% rule?
    print("\nis_progressive vs Opta's own 25%-towards-goal rule, by team:")
    for team in (H, A):
        n = ok = 0
        for e in doc["events"]:
            if e["type"] not in ("PASS", "PROGRESSIVE_PASS") or \
                    e.get("team") != team:
                continue
            at, to = e.get("at"), e.get("to")
            if not at or not to:
                continue
            g = 105.0 if team == H else 0.0
            d0, d1 = abs(at[0] - g), abs(to[0] - g)
            n += 1
            if d0 > 0 and (d0 - d1) >= 0.25 * d0 == \
                    bool((e.get("pass") or {}).get("is_progressive")):
                ok += 1
        if n:
            print(f"   {team:<11} n={n:<6} agreement {100.0*ok/n:5.1f}%")


if __name__ == "__main__":
    main()
