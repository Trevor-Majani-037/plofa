"""P1.3 — audit the reward table BEFORE trusting the GA.

WHY THIS IS THE GATE THAT MATTERS
---------------------------------
The match gate is downstream of this.  If the table says a shot from 18 m dead
centre with the mouth open is worth less than dribbling, then the genetic
algorithm will faithfully learn not to shoot, and a 6-match gate will happily
promote the result.  The GA is not the thing to check; the objective is.

The measured failure this exists to catch (2026-11-08): in the striker's
clearest chance the table returned DRIBBLE 1.163, SWITCH 1.027, SHOOT 0.957 —
shooting priced 18% BELOW dribbling in the one bucket where shooting is
obviously right.

WHAT IT ASSERTS
---------------
  1. the reward ladder is populated enough to be used, not starved to `prior`;
  2. among real striker samples in a clear chance, a central look with an open
     mouth prices SHOOT above a byline look with the mouth shut;
  3. SHOOT beats CROSS and DRIBBLE in the best available chance.

Run:  python tools/diag/_audit_reward_table.py [surrogate.json] [corpus.json]
Exit code 0 = the ordering is football; 1 = the GA must not run yet.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from surrogate_collect import FitnessSurrogate, _role_family  # noqa: E402
from decision_brain import PlayerIntent  # noqa: E402


def load_corpus(path):
    """(sensors, intent, success, action, position) — the shape ``fit`` wants."""
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    out = []
    for r in raw:
        if isinstance(r, (list, tuple)) and len(r) >= 5:
            s, it, succ, act, pos = r[:5]
            try:
                intent = PlayerIntent(it)
            except ValueError:
                continue
            out.append((np.asarray(s, dtype=np.float64), intent,
                        float(succ), str(act), str(pos)))
        elif isinstance(r, dict):
            s = r.get("sensors") or r.get("state")
            if s is None:
                continue
            try:
                intent = PlayerIntent(r.get("intent"))
            except (ValueError, TypeError):
                continue
            out.append((np.asarray(s, dtype=np.float64), intent,
                        float(r.get("success", r.get("reward", 0.0))),
                        str(r.get("action", r.get("intent", ""))),
                        str(r.get("position", ""))))
    return out


def ladder_report(sur, data):
    rich = sum(1 for pos, b in sur.counts.items() if pos
               for k in b if FitnessSurrogate.is_rich_key(k))
    coarse = sum(1 for pos, b in sur.counts.items() if pos
                 for k in b if not FitnessSurrogate.is_rich_key(k))
    hit_r = hit_c = miss = 0
    for sensors, intent, _s, _act, pos in data:
        keys = FitnessSurrogate.bucket_keys(sensors)
        ik = intent.value
        if sur._lookup_cell(keys[0], ik, pos) is not None:
            hit_r += 1
        elif sur._lookup_cell(keys[1], ik, pos) is not None:
            hit_c += 1
        else:
            miss += 1
    n = max(1, len(data))
    print(f"ladder: rich cells {rich}, coarse cells {coarse}")
    print(f"lookups: rich {hit_r / n:.1%} | coarse {hit_c / n:.1%} | "
          f"prior {miss / n:.1%}")
    return hit_r / n


def shot_by_geometry(sur, data, families=("ST",)):
    """Mean SHOOT reward per geometry bin, over real striker chances."""
    groups = defaultdict(list)
    for sensors, intent, _succ, _act, pos in data:
        if intent is not PlayerIntent.SHOOT:
            continue
        if _role_family(pos) not in families:
            continue
        # a clear chance: final third, goal-close, central
        if sensors[12] <= 0.5 or sensors[14] >= 0.5 or sensors[15] <= 0.5:
            continue
        geom = FitnessSurrogate.goal_geometry(sensors)
        groups[geom].append(sur.expected_success(sensors, intent, pos))
    return groups


def main():
    sur_path = sys.argv[1] if len(sys.argv) > 1 else "brains_trainer/surrogate_refit.json"
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else "brains_trainer/corpus.raw.json"
    for p in (sur_path, corpus_path):
        if not os.path.exists(p):
            raise SystemExit(f"! missing {p}")

    sur = FitnessSurrogate.load(sur_path)
    data = load_corpus(corpus_path)
    print(f"surrogate: {sur_path}")
    print(f"corpus   : {corpus_path} ({len(data)} samples)\n")

    coverage = ladder_report(sur, data)

    print("\nSHOOT reward by goal geometry — striker clear chances")
    groups = shot_by_geometry(sur, data)
    if not groups:
        print("  (no striker clear-chance SHOOT samples — cannot audit)")
        raise SystemExit(1)
    for k in sorted(groups):
        v = groups[k]
        print(f"  {k}: n={len(v):3d} mean={np.mean(v):+.4f}")

    # The mouth bin is the middle character pair; angle bin is the first.
    def bin_of(k, letter):
        i = k.index(letter)
        return int(k[i + 1])

    # A GEOMETRY COMPARISON NEEDS SUPPORT.  The first version of this check
    # compared the best open-central cell against the best shut/wide cell with
    # no minimum n, and duly "failed" on cells holding 2 samples each — noise
    # dressed as a verdict.  A cell has to hold enough shots before its mean
    # means anything.
    MIN_N = 8

    # POOLING IS REQUIRED, AND IT IS A MEASURED FINDING RATHER THAN A RELAXATION.
    # The second version demanded a single shut/wide CELL with n >= MIN_N.  No
    # such cell exists, and none can: a shut mouth (M0) is a shot almost nobody
    # takes, so its samples scatter across angle and distance instead of
    # concentrating.  Measured on 15,426 samples (115 striker clear-chance
    # shots): the shut-mouth population is 8 shots spread over 4 cells, and the
    # extreme-angle population is 8 over 3 cells.  Demanding that one cell hold
    # 8 of them demands a sample size the simulation cannot produce at any
    # affordable cost, so the gate would block evolution forever while
    # appearing to wait for more data.  That is a deadlock, not a gate.
    #
    # The claim was always about populations — "an open central look is worth
    # more than a shut or a wide look" — so the comparison is between pooled
    # populations, with the same MIN_N applied to each group.
    def _pool(pred):
        vals = [x for k, v in groups.items() if pred(k) for x in v]
        return len(vals), (float(np.mean(vals)) if vals else 0.0)

    n_open, m_open = _pool(lambda k: bin_of(k, "M") == 2 and bin_of(k, "A") == 2)
    n_shut, m_shut = _pool(lambda k: bin_of(k, "M") == 0 or bin_of(k, "A") in (0, 4))
    n_shutmouth, m_shutmouth = _pool(lambda k: bin_of(k, "M") == 0)
    total_shots = sum(len(v) for v in groups.values())

    print(f"\nordering checks (a pooled group needs n >= {MIN_N})")
    print(f"  striker clear-chance SHOOT samples: {total_shots}")
    ok = True
    sufficient = True

    if n_open >= MIN_N and n_shut >= MIN_N:
        passed = m_open > m_shut
        print(f"  [{'PASS' if passed else 'FAIL'}] open central look "
              f"({m_open:+.4f}, n={n_open}) > shut/wide look "
              f"({m_shut:+.4f}, n={n_shut})")
        ok &= passed
    elif total_shots < 100:
        print("  [INSUFFICIENT] too few striker shots to judge the geometry "
              "ordering — collect more matches before evolving")
        sufficient = False
    else:
        print(f"  [INSUFFICIENT] pooled support: open central n={n_open}, "
              f"shut/wide n={n_shut}, both need {MIN_N}")
        sufficient = False

    # The mouth dimension on its own, reported because it is the one the
    # earlier gate was silently unable to measure.
    if n_shutmouth >= MIN_N and n_open >= MIN_N:
        passed = m_open > m_shutmouth
        print(f"  [{'PASS' if passed else 'FAIL'}] mouth open ({m_open:+.4f}, "
              f"n={n_open}) > mouth shut ({m_shutmouth:+.4f}, n={n_shutmouth})")
        ok &= passed
    else:
        print(f"  [n/a] mouth-shut population n={n_shutmouth} — too thin to "
              f"compare on its own (pooled into the shut/wide check above)")

    # SHOOT vs the alternatives, in the best chance the corpus holds.
    # "Best" must mean best-SUPPORTED, not highest mean: with thin cells the
    # highest mean is usually a two-sample fluke, and probing it would test the
    # table on exactly the cell it knows least about.
    supported = {k: v for k, v in groups.items() if len(v) >= MIN_N}
    best_geom = (max(supported, key=lambda k: np.mean(supported[k]))
                 if supported else max(groups, key=lambda k: len(groups[k])))
    probe = None
    for sensors, intent, _s, _act, pos in data:
        if (intent is PlayerIntent.SHOOT and _role_family(pos) == "ST"
                and sensors[12] > 0.5 and sensors[14] < 0.5 and sensors[15] > 0.5
                and FitnessSurrogate.goal_geometry(sensors) == best_geom):
            probe = (sensors, pos)
            break
    if probe is not None:
        sensors, pos = probe
        # Support per rival IN THIS CELL.  The previous version compared SHOOT
        # against DRIBBLE and CROSS unconditionally, and flagged a rival sitting
        # on the prior as vacuous evidence.  That flag was right about the
        # principle ("ignorance is attractive") but wrong about the remedy:
        # measured on 15,426 samples, a striker in a clear central chance takes
        # ZERO dribbles — the action simply does not occur there, so the table
        # can never hold a dribble cell to compare against, and requiring one
        # makes the gate permanently unpassable for a reason that has nothing to
        # do with the table.  Absence of an action is a fact about the action
        # distribution; it is reported, not counted against the table.  A rival
        # that DOES have samples but still sits exactly on the prior is the real
        # vacuity, and still blocks.
        cell_counts = defaultdict(int)
        for s2, i2, *_rest, p2 in data:
            # Support must be counted IN THE ROW THE LOOKUP WILL READ.  The
            # table is per position, and ST and CF are one role family but two
            # rows: a cell can hold four through-balls played by centre-forwards
            # and still return the prior for a striker, because the striker's
            # row has no entry.  Counting family-wide support and then comparing
            # against one position's row would report that prior as "ignorant"
            # when it is simply another player's row.
            if p2 != pos:
                continue
            if s2[12] <= 0.5 or s2[14] >= 0.5 or s2[15] <= 0.5:
                continue
            if FitnessSurrogate.goal_geometry(s2) == best_geom:
                cell_counts[i2.value] += 1

        rivals = ("PROGRESSIVE_PASS", "CROSS", "SWITCH", "THROUGH_BALL", "DRIBBLE")
        vals = {i.value: sur.expected_success(sensors, i, pos)
                for i in (PlayerIntent.SHOOT,)
                + tuple(PlayerIntent[r] for r in rivals)}
        print(f"  best chance {best_geom}: " + ", ".join(
            f"{k} {v:+.4f}" for k, v in sorted(vals.items())))
        print(f"  support in that cell: " + ", ".join(
            f"{r}={cell_counts.get(r, 0)}" for r in rivals))
        for rival in rivals:
            n_rival = cell_counts.get(rival, 0)
            if n_rival == 0:
                print(f"  [n/a] SHOOT vs {rival} — no {rival} samples in this "
                      f"situation; not evidence either way, not required")
                continue
            passed = vals["SHOOT"] > vals[rival]
            # The old "vacuous" test inferred ignorance from the VALUE sitting
            # on the prior.  That inference is unsound: a through-ball played in
            # a clear chance measures EXACTLY 0.0 in the ladder (possession kept,
            # threat unchanged), so a genuinely measured cell is
            # indistinguishable from a missing one by value alone.  Support is
            # the honest test, and it is already available — a rival with zero
            # samples is the n/a branch above, and a rival with samples has been
            # measured, whatever number it landed on.
            print(f"  [{'PASS' if passed else 'FAIL'}] SHOOT > {rival} "
                  f"(n={n_rival}, {vals[rival]:+.4f})")
            ok &= passed

    print(f"\ncoverage: {coverage:.1%} of lookups resolved at some level")
    if coverage < 0.5:
        print("  !! most lookups fall to the neutral prior — the table cannot "
              "teach anything. Recollect before evolving.")
        ok = False
        sufficient = False

    if not ok:
        verdict = "TABLE ORDERING WRONG — do NOT evolve"
    elif not sufficient:
        verdict = ("ORDERING HOLDS WHERE MEASURED, BUT THE CORPUS IS TOO THIN "
                   "TO CONFIRM IT — collect more before evolving")
    else:
        verdict = "TABLE ORDERING OK — safe to evolve"
    print("\n" + verdict)
    raise SystemExit(0 if (ok and sufficient) else 1)


if __name__ == "__main__":
    main()
