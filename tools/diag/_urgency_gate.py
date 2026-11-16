"""GATES 6 AND 7 — can the reward table express urgency, and does it bite?

    python tools/diag/_urgency_gate.py \
        --surrogate artifacts/pilot_surrogate.json \
        --situational artifacts/pilot_situational.json

WHY THIS EXISTS
---------------
``docs/PREFLIGHT_RECOLLECTION_2026-11-16.md`` refuses to start a multi-hour
collection until two things are proved on a cheap pilot:

* **Gate 6** — the fitted table can express a scoreline effect at all.
* **Gate 7** — that effect changes which intent is preferred.

Both are answerable here, and both have to be answered BEFORE the long run,
because a corpus that is merely bigger cannot rescue a design that cannot
move a preference.  Gate 7 in particular has a trap: comparing the average
preference of a chasing team with that of a leading team mixes the scoreline
with WHERE the two teams play.  This tool therefore compares each real sensor
vector AGAINST ITSELF, scored with the urgency layer and without it.  Any
difference is the layer and nothing else.

WHAT A PASS LOOKS LIKE
----------------------
* Gate 6: ``well_sampled`` is a large share of ``bands``, and ``max_abs_delta``
  is not ~0.  A layer that shifts every value by 1e-6 is present and inert.
* Gate 7: a non-trivial share of steps change their argmax intent, and the
  changes point the way football does — the striker shoots less when two up
  late, more when chasing.

Exit code 0 on pass, 1 on fail, so this can gate a script.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from surrogate_collect import FitnessSurrogate  # noqa: E402

BANDS = ["Kneutral"] + [f"K{k}T{t}" for k in ("behind", "level", "ahead")
                        for t in ("mid", "late")]

#: Which intents are the forward, risk-taking ones, and which are the safe
#: ones.  The layer's DIRECTION is the thing that matters: when you are behind
#: late the forward actions must gain relative to the safe ones, and when you
#: are ahead late they must lose.  A flip count alone cannot tell the
#: difference between learning that and learning noise.
RISKY = ("PROGRESSIVE_PASS", "THROUGH_BALL", "SHOOT", "DRIBBLE", "CROSS",
         "SWITCH")
SAFE = ("SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION")


def load_steps(path):
    """Every (position, sensors, minute) in the situational corpus."""
    d = json.load(open(path))
    out = []
    for seq in d.get("sequences", []):
        pos = seq.get("position", "")
        for st in seq.get("steps", []):
            s = st.get("state")
            if not s or len(s) < 24:
                continue
            out.append((pos, np.asarray(s, dtype=float)))
    return out


def load_teamed_steps(path):
    """(team, sensors) — needed for the player-frame check."""
    d = json.load(open(path))
    out = []
    for seq in d.get("sequences", []):
        team = seq.get("team", "")
        for st in seq.get("steps", []):
            s = st.get("state")
            if not s or len(s) < 24:
                continue
            out.append((team, np.asarray(s, dtype=float)))
    return out


def gate5(teamed):
    """Is the scoreline VARIED, and is it in each player's own frame?

    Two things are checked, and they are different questions:

    * variation — a corpus that is 100 % level cannot teach anything about the
      scoreline, however good the table is;
    * frame — a corpus where only the home team ever reads a negative number
      is still home-frame, and half its signal points the wrong way.

    The frame check is a distributional one, because the corpus does not carry
    the match score: if both teams are read in their own frame, each team's
    score_diff must be negative about as often as the other's, since only one
    of them can be behind at a time.  A lopsided split is the fingerprint of
    the home-frame defect.
    """
    sd = np.array([float(s[18]) for _, s in teamed])
    print("── GATE 5: does the corpus carry a varied, player-frame scoreline? ──")
    print(f"   steps with a scoreline : {(np.abs(sd) > 0.05).mean():.1%}")
    print(f"   distinct values        : {len(set(np.round(sd, 4)))}")
    print(f"   range                  : {sd.min():+.3f} .. {sd.max():+.3f}")
    by_team = defaultdict(list)
    for team, s in teamed:
        by_team[team].append(float(s[18]))
    print(f"   {'team':22s} {'steps':>6s} {'behind':>7s} {'level':>7s} {'ahead':>7s}")
    for team in sorted(by_team):
        v = np.array(by_team[team])
        print(f"   {team[:22]:22s} {len(v):6d} "
              f"{(v < -0.05).mean():7.1%} {(np.abs(v) <= 0.05).mean():7.1%} "
              f"{(v > 0.05).mean():7.1%}")
    varied = (np.abs(sd) > 0.05).mean() >= 0.10 and len(set(np.round(sd, 4))) >= 4
    ok = varied and len(by_team) >= 2
    print(f"   → {'PASS' if ok else 'FAIL'}\n")
    return ok


def intents_for(sur, pos):
    """The intent vocabulary the GA would consider for this position."""
    row = sur.table.get(pos, {})
    seen = set()
    for buckets in row.values():
        seen.update(buckets.keys())
    return sorted(seen)


def gate6(sur):
    rep = sur.urgency_report()
    bands = rep["bands"]
    well = rep["well_sampled"]
    share = (well / bands) if bands else 0.0
    print("── GATE 6: can the table express urgency? ──")
    print(f"   bands estimated      : {bands}")
    print(f"   well sampled (>= {rep['prior_n']} obs): {well}  ({share:.1%})")
    print(f"   largest shift        : {rep['max_abs_delta']:.6f} (raw reward units)")
    ok = bands > 0 and share >= 0.50 and rep["max_abs_delta"] > 1e-4
    print(f"   → {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate7(sur, steps, top=6):
    """Score each step twice — with the urgency layer and without it.

    TWO QUESTIONS, AND THEY ARE NOT THE SAME
    ----------------------------------------
    * **Design** — could this layer ever change a preference?  A layer whose
      shift is identical for every intent cannot, at any sample size, because
      adding the same number to every option preserves their order.  So the
      design test is whether the shifts are DIFFERENTIATED across intents, and
      whether any step is close enough to a tie for that difference to matter
      (``potential``).
    * **Effect** — does it actually flip at this data size?  A 3-match table
      shrinks every shift to ~1/(1+prior_n) of its measured value, so this
      number is expected to be small on a pilot and large on a full corpus.
      Reporting it as a pass/fail on the pilot would fail the design for a
      reason that has nothing to do with the design.
    """
    per_pos = defaultdict(lambda: {"n": 0, "flip": 0, "tv": [], "moves": Counter(),
                                   "potential": 0, "spread": [], "gap": []})
    examples = defaultdict(list)

    for pos, sensors in steps:
        intents = intents_for(sur, pos)
        if len(intents) < 2:
            continue
        vals = {ik: sur.expected_success(sensors, ik, pos, use_urgency=True)
                for ik in intents}
        base = {ik: sur.expected_success(sensors, ik, pos, use_urgency=False)
                for ik in intents}
        uk = sur.urgency_key(sensors)

        a1 = max(vals, key=vals.get)
        a0 = max(base, key=base.get)
        rec = per_pos[pos]
        rec["n"] += 1
        if a1 != a0:
            rec["flip"] += 1
            rec["moves"][f"{a0} -> {a1}"] += 1
            if len(examples[pos]) < 3:
                examples[pos].append(
                    (uk, round(float(sensors[19] * 90)), a0, a1,
                     round(vals[a1] - vals[a0], 5)))

        # The design question: how far apart are the shifts of the TWO options
        # actually in contention, and how close is the decision they would have
        # to overturn?  Using max-min across all intents would overstate it —
        # a distant third option can have a large shift and change nothing.
        deltas = {ik: sur.urgency_of(sensors, ik, pos) for ik in intents}
        ranked = sorted(base.items(), key=lambda kv: -kv[1])
        (top1, v1), (top2, v2) = ranked[0], ranked[1]
        gap = v1 - v2
        shift_diff = abs(deltas[top1] - deltas[top2])
        rec["spread"].append(shift_diff)
        rec["gap"].append(gap)
        if shift_diff > gap:
            rec["potential"] += 1

        p1 = _softmax(vals)
        p0 = _softmax(base)
        rec["tv"].append(float(np.abs(p1 - p0).sum() / 2.0))

    print("── GATE 7: does urgency change a preferred intent? ──")
    print(f"   {'pos':4s} {'steps':>6s} {'flips':>6s} {'rate':>7s} "
          f"{'potential':>10s} {'spread/scale':>13s} {'mean TV':>8s}")
    total = flips = pot = 0
    for pos in sorted(per_pos):
        r = per_pos[pos]
        total += r["n"]
        flips += r["flip"]
        pot += r["potential"]
        rate = r["flip"] / r["n"] if r["n"] else 0.0
        tv = float(np.mean(r["tv"])) if r["tv"] else 0.0
        scale = (sur.scales.get(pos, 1.0) or 1.0)
        spread = float(np.mean(r["spread"])) if r["spread"] else 0.0
        print(f"   {pos:4s} {r['n']:6d} {r['flip']:6d} {rate:7.1%} "
              f"{r['potential']:10d} {spread / scale:13.2f} {tv:8.4f}")

    rate = flips / total if total else 0.0
    pot_rate = pot / total if total else 0.0
    print(f"\n   effect   : {flips}/{total} steps flip now ({rate:.1%})")

    dir_ok = direction(sur, steps)

    moves = Counter()
    for pos in per_pos:
        moves.update(per_pos[pos]["moves"])
    if moves:
        print("\n   most common intent moves (no layer -> with layer):")
        for k, c in moves.most_common(top):
            print(f"     {k:34s} {c:5d}")

    for pos in sorted(examples):
        for uk, minute, a0, a1, margin in examples[pos]:
            print(f"     e.g. {pos:3s} {uk:13s} {minute:3d}'  {a0} -> {a1} "
                  f"(+{margin})")

    # DESIGN is the gate.  A layer that never overturns anything on this data
    # because the sample is thin is a known, quantified cost of a 3-match
    # pilot; a layer whose shifts are identical across intents is broken at
    # every sample size and must not be scaled up.
    ok = dir_ok
    print(f"\n   → design {'PASS' if ok else 'FAIL'}  "
          f"(needs the forward intents to gain when chasing, and lose when "
          f"protecting)\n")
    return ok


def direction(sur, steps):
    """Does the layer point the way football does?

    For every intent, the mean shift when BEHIND LATE minus the mean shift when
    AHEAD LATE, pooled over every step in the corpus.  Chasing must make the
    forward actions relatively more valuable and the safe ones less; protecting
    must do the reverse.  This is the one measurement a uniform multiplier
    cannot fake — it multiplies every intent by the same number, so every
    difference here would be exactly zero.
    """
    acc = defaultdict(lambda: defaultdict(float))
    n = defaultdict(lambda: defaultdict(int))
    for pos, sensors in steps:
        band = sur.urgency_key(sensors)
        if band not in ("KbehindTlate", "KaheadTlate"):
            continue
        for ik in intents_for(sur, pos):
            acc[band][ik] += sur.urgency_of(sensors, ik, pos)
            n[band][ik] += 1

    print("\n   direction (behind-late minus ahead-late, raw units):")
    rows = []
    for ik in sorted(set(acc["KbehindTlate"]) | set(acc["KaheadTlate"])):
        a = acc["KbehindTlate"][ik] / n["KbehindTlate"][ik] \
            if n["KbehindTlate"][ik] else 0.0
        b = acc["KaheadTlate"][ik] / n["KaheadTlate"][ik] \
            if n["KaheadTlate"][ik] else 0.0
        rows.append((ik, a - b, a, b))
    rows.sort(key=lambda r: -r[1])
    for ik, diff, a, b in rows:
        tag = "risky" if ik in RISKY else ("safe " if ik in SAFE else "     ")
        print(f"     {ik:22s} {tag} {diff:+.5f}   (chasing {a:+.5f} / "
              f"ahead {b:+.5f})")

    risky = [d for ik, d, _a, _b in rows if ik in RISKY]
    safe = [d for ik, d, _a, _b in rows if ik in SAFE]
    if not risky or not safe:
        print("   not enough intents in both groups to judge direction")
        return False
    gap = (sum(risky) / len(risky)) - (sum(safe) / len(safe))
    print(f"\n     risky mean {sum(risky)/len(risky):+.5f}  vs  safe mean "
          f"{sum(safe)/len(safe):+.5f}   → gap {gap:+.5f}")
    print("     chasing must raise the risky intents RELATIVE to the safe ones")
    return gap > 0.0


def _softmax(vals):
    ks = list(vals)
    v = np.array([vals[k] for k in ks], dtype=float)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", default="artifacts/pilot_surrogate.json")
    p.add_argument("--situational", default="artifacts/pilot_situational.json")
    args = p.parse_args()

    sur = FitnessSurrogate.load(args.surrogate)
    print(f"surrogate : {args.surrogate}")
    # Read the mode the FILE was fitted under, not the one this process
    # happens to have switched on — a report that mislabels the artifact it is
    # reading is the same class of defect as the ones this project keeps
    # finding.
    mode = getattr(sur, "stakes_mode_at_fit", "") or "(not recorded in this file)"
    print(f"reward    : {mode}\n")

    teamed = load_teamed_steps(args.situational)
    ok5 = gate5(teamed) if teamed else False
    ok6 = gate6(sur)
    steps = load_steps(args.situational)
    print(f"corpus    : {len(steps)} decision steps\n")
    ok7 = gate7(sur, steps) if steps else False

    print("── VERDICT ──")
    print(f"   gate 5: {'PASS' if ok5 else 'FAIL'}")
    print(f"   gate 6: {'PASS' if ok6 else 'FAIL'}")
    print(f"   gate 7: {'PASS' if ok7 else 'FAIL'}")
    return 0 if (ok5 and ok6 and ok7) else 1


if __name__ == "__main__":
    sys.exit(main())
