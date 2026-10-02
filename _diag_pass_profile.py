"""WHERE DO THE 675 NON-PROGRESSIVE PASSES GO? (metadata edition)

First attempt failed and the traceback was informative: the field names are
`location_x/location_y/end_x/end_y`, and the engine ALREADY computes everything
I was about to recompute badly — `pass_direction`, `length_class`,
`pass_advance`, `start_third`/`end_third`, `pass_channel`, `possession_phase`,
`pass_type`, and `attacking_matrix.action` with a reason. So this reads those,
and separately re-derives the direction from raw geometry to check the engine's
own labels against the coordinates. A classifier that is only ever compared
with itself tells you nothing.

Classified in the PASSING TEAM's attacking frame, so the half-time ends change
is normalised out. Reference bands are Premier League / StatsBomb for a
top-flight team and are the target, not this engine's history.
"""
import math
import random
from collections import Counter

import numpy as np

from _diag_watch import build

PASS_TYPES = ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY")


def main():
    eng, hr, ar = build()
    eng.enable_virtual_gps(0.1)
    random.seed(31)
    res = eng.simulate()
    print("SCORE", res.score_str)

    rows = []
    for e in res.timeline:
        t = getattr(e, "event_type", None)
        if t is None or t.name not in PASS_TYPES:
            continue
        md = getattr(e, "metadata", None) or {}
        x0 = getattr(e, "location_x", None)
        y0 = getattr(e, "location_y", None)
        x1 = getattr(e, "end_x", None)
        y1 = getattr(e, "end_y", None)
        if None in (x0, y0, x1, y1):
            continue
        team = getattr(e, "team", "")
        minute = getattr(e, "minute", 0) or 0
        right = True
        if team == eng.config.away_team:
            right = False
        if minute >= 45:
            right = not right
        dx = (x1 - x0) * (1.0 if right else -1.0)
        rows.append(dict(
            t=t.name, team=team, minute=minute,
            dx=dx, dy=(y1 - y0),
            dist=float(md.get("pass_length_m")
                       or math.hypot(x1 - x0, y1 - y0)),
            label_dir=md.get("pass_direction", "?"),
            length_class=md.get("length_class", "?"),
            is_prog=bool(md.get("is_progressive")),
            start_third=md.get("start_third", "?"),
            end_third=md.get("end_third", "?"),
            channel=md.get("pass_channel", "?"),
            ptype=md.get("pass_type", "?"),
            phase=md.get("possession_phase", "?"),
            matrix=(md.get("attacking_matrix") or {}).get("action", "?"),
            matrix_reason=(md.get("attacking_matrix") or {}).get("reason", ""),
            intent=((md.get("active_brain") or {}).get("intent", "?")),
            # attacking-frame absolute depth of start and end
            ax0=(x0 if right else 105.0 - x0), ax1=(x1 if right else 105.0 - x1),
        ))

    n = len(rows)
    print(f"\npasses with usable geometry: {n}")
    if not n:
        return

    # ── 1. does the engine's own direction label match the coordinates? ──
    def geo(r):
        if r["dx"] > 5.0:
            return "forward"
        if r["dx"] < -5.0:
            return "backward"
        return "square"

    agree = Counter()
    for r in rows:
        agree[(r["label_dir"], geo(r))] += 1
    print("\n1. ENGINE LABEL vs RAW GEOMETRY (sanity - if these disagree,")
    print("   nothing below can be trusted)")
    print("   label".ljust(11), "forward".rjust(9), "square".rjust(8),
          "backward".rjust(9))
    for lab in ("forward", "sideways", "backward"):
        if lab not in {a for a, _ in agree}:
            continue
        print(f"   {lab:<11}{agree[(lab,'forward')]:>9}"
              f"{agree[(lab,'square')]:>8}{agree[(lab,'backward')]:>9}")

    # ── 2. the headline direction split ────────────────────────────────
    c = Counter(geo(r) for r in rows)
    print("\n2. DIRECTION (geometry, attacking frame, 5 m deadband)")
    print("   direction".ljust(13), "count".rjust(7), "share".rjust(8),
          "real PL".rjust(11))
    for k, rl in (("forward", "35-40%"), ("square", "45-50%"),
                  ("backward", "10-15%")):
        print(f"   {k:<13}{c[k]:>7}{100*c[k]/n:>7.1f}%{rl:>12}")

    # ── 3. joint direction x length: which of the three problems is it? ──
    print("\n3. JOINT - backward? square-and-long? or short-and-square?")
    print("   band".ljust(12), "n".rjust(6), "fwd".rjust(7), "sq".rjust(7),
          "back".rjust(7), "median dx".rjust(11))
    for lo, hi, lab in ((0, 10, "<10 m"), (10, 20, "10-20 m"),
                        (20, 30, "20-30 m"), (30, 45, "30-45 m"),
                        (45, 1e9, "45 m+")):
        sub = [r for r in rows if lo <= r["dist"] < hi]
        if not sub:
            continue
        cc = Counter(geo(r) for r in sub)
        mdx = np.median([r["dx"] for r in sub])
        print(f"   {lab:<12}{len(sub):>6}{100*cc['forward']/len(sub):>6.0f}%"
              f"{100*cc['square']/len(sub):>6.0f}%"
              f"{100*cc['backward']/len(sub):>6.0f}%{mdx:>+11.1f}")

    # ── 4. length classes the engine itself assigned ───────────────────
    print("\n4. LENGTH CLASS (engine's own label)")
    lc = Counter(r["length_class"] for r in rows)
    for k, v in lc.most_common():
        print(f"   {str(k):<12}{v:>6}{100*v/n:>7.1f}%")
    d = np.array([r["dist"] for r in rows])
    print(f"   length m: median {np.median(d):.1f}, mean {d.mean():.1f}, "
          f"p90 {np.percentile(d,90):.1f}   (real PL median 15-18)")

    # ── 5. is the progressive FLAG the problem, or the passes? ──────────
    print("\n5. PROGRESSIVE FLAG vs GEOMETRY")
    print(f"   flagged {sum(1 for r in rows if r['is_prog'])} / {n} "
          f"({100*sum(1 for r in rows if r['is_prog'])/n:.1f}%)   real 25-35%")
    # Opta's own definition, x-only: cuts distance to the goal centre by >=25%
    def opta(r):
        d0, d1 = abs(r["ax0"] - 52.5), abs(r["ax1"] - 52.5)
        return d0 > 0 and (d0 - d1) >= 0.25 * d0
    flagged = [r for r in rows if r["is_prog"]]
    unflag = [r for r in rows if not r["is_prog"]]
    print(f"   of the {len(unflag)} unflagged, {sum(1 for r in unflag if opta(r))}"
          f" ({100*sum(1 for r in unflag if opta(r))/max(len(unflag),1):.0f}%)"
          f" meet Opta's own x-only definition")
    print(f"   of the {len(flagged)} flagged, {sum(1 for r in flagged if opta(r))}"
          f" ({100*sum(1 for r in flagged if opta(r))/max(len(flagged),1):.0f}%)"
          f" do")
    print("   -> the flag is "
          f"{'UNDER-counting real progression' if sum(1 for r in unflag if opta(r)) > 0.15*len(unflag) else 'not the problem: the other passes genuinely do not go forward'}")

    # ── 6. what the engine says it was trying to do ────────────────────
    print("\n6. ATTACKING MATRIX ACTION (what the team was told to do)")
    for k, v in Counter(r["matrix"] for r in rows).most_common():
        print(f"   {str(k):<22}{v:>6}{100*v/n:>7.1f}%")
    print("\n   BRAIN INTENT")
    for k, v in Counter(r["intent"] for r in rows).most_common(10):
        print(f"   {str(k):<22}{v:>6}{100*v/n:>7.1f}%")
    print("\n   POSSESSION PHASE")
    for k, v in Counter(r["phase"] for r in rows).most_common():
        print(f"   {str(k):<28}{v:>6}{100*v/n:>7.1f}%")

    # ── 7. third-to-third: does the ball ever get up the pitch? ────────
    print("\n7. WHERE PASSES END (attacking frame)")
    for k, v in Counter(r["end_third"] for r in rows).most_common():
        print(f"   {str(k):<20}{v:>6}{100*v/n:>7.1f}%")
    ft = sum(1 for r in rows if r["ax1"] > 70)
    print(f"   final third (x>70): {ft} ({100*ft/n:.1f}%)   real PL 15-20%")
    atk_third = sum(1 for r in rows if r["ax0"] > 70)
    print(f"   passes STARTING in the final third: {atk_third} "
          f"({100*atk_third/n:.1f}%)")
    print("   channel:")
    for k, v in Counter(r["channel"] for r in rows).most_common():
        print(f"     {str(k):<18}{v:>6}{100*v/n:>7.1f}%")

    # ── 8. per team, and does anyone differ? ──────────────────────────
    print("\n8. PER TEAM")
    for tm in (eng.config.home_team, eng.config.away_team):
        sub = [r for r in rows if r["team"] == tm]
        if not sub:
            continue
        cc = Counter(geo(r) for r in sub)
        m = len(sub)
        print(f"   {tm:<12} n={m:<5} fwd {100*cc['forward']/m:>4.0f}%"
              f"  sq {100*cc['square']/m:>4.0f}%"
              f"  back {100*cc['backward']/m:>4.0f}%"
              f"  prog {100*sum(1 for r in sub if r['is_prog'])/m:>4.0f}%"
              f"  median dx {np.median([r['dx'] for r in sub]):>+5.1f}")


if __name__ == "__main__":
    main()
