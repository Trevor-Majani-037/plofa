"""THE BALANCE — does the reward now outrank the GA's guards, per position?

WHY
---
The v5 retrain collapsed every position to <=3 intents while every fitness
number stayed unremarkable (0.60-0.67).  The mechanism was in
`brain_evolution.synthetic_fitness`:

    base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)      # the football
    fitness = base
              + CONFIDENCE_GUARD_W * (mean_conf - 0.5)   # 0.05
              + DIVERSITY_GUARD_W * bdiv                 # 0.05
              - indecision_penalty / n_states            # up to 0.10

Every one of those guards is a FIXED-WEIGHT term.  The reward is not: its
scale depends on the position's own table.  So wherever a position's reward
was flat, the guards WERE the objective and the GA optimised peakedness
instead of football.

THE MEASUREMENT
---------------
For each position, over the PRODUCTION state corpus (300 states, seed 123),
the reward's total swing is

    0.5 * ( mean over states of max_intent reward
          - mean over states of min_intent reward )

which is exactly the range `base` can take between a brain that always picks
the best available intent and one that always picks the worst.  That is the
gradient the GA is given.

The guard budget is a fixed, LABELLED number, not a guess dressed as a
measurement:

    indecision penalty   0.00 - 0.10   (a near-tie in the top two logits)
    confidence guard     0.00 - 0.04   (mean_conf 0.1 -> 0.9, weight 0.05)
    diversity guard      0.00 - 0.05   (bdiv 0 -> 1, weight 0.05)

The guards are ANTI-CORRELATED in practice (a peaked brain has a high
confidence and a low diversity and few near-ties), so the honest comparison
is the reward's swing against the LARGEST SINGLE guard term, and against the
whole budget.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_scale_balance.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from brain_evolution import (  # noqa: E402
    CONFIDENCE_GUARD_W, DIVERSITY_GUARD_W, generate_state_corpus,
)
from decision_brain import PlayerIntent  # noqa: E402
from surrogate_collect import (  # noqa: E402
    FORBIDDEN_VALUE, FitnessSurrogate, load_raw_corpus,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json"))

N_STATES = 300
SEED = 123

#: the guard budget, from the constants in brain_evolution (see the docstring)
INDECISION_MAX = 0.10
CONF_RANGE = CONFIDENCE_GUARD_W * (0.9 - 0.1)
DIV_RANGE = DIVERSITY_GUARD_W * 1.0
GUARD_BUDGET = INDECISION_MAX + CONF_RANGE + DIV_RANGE
LARGEST_GUARD = INDECISION_MAX


def _swing(sur, sensors, position):
    """The reward's swing over the corpus, in `base` units."""
    from role_features import ROLE_INTENTS, _role_family
    fam = _role_family(position) or ""
    allowed = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, ())]
    best_sum = 0.0
    worst_sum = 0.0
    n = 0
    for row in sensors:
        vals = []
        for intent in allowed:
            v = sur.expected_success(row[:24], intent, position)
            if v is None:
                continue
            vals.append(float(v))
        if not vals:
            continue
        best_sum += max(vals)
        worst_sum += min(vals)
        n += 1
    if n == 0:
        return 0.0
    return 0.5 * ((best_sum - worst_sum) / n)


def _scaled_mean(sur, position):
    """The count-weighted mean of the position's SCALED cells.

    `base = 0.5 + 0.5 * clip(mean_reward, -1, 1)` clips the MEAN reward, so a
    position whose cells all sit far from zero would saturate the clip for
    every brain and lose its ranking entirely.
    """
    vals, wts = [], []
    for bucket, intents in sur.table.get(position, {}).items():
        for ik, v in intents.items():
            vals.append(float(v))
            wts.append(float(sur.counts.get(position, {}).get(bucket, {}).get(ik, 1)))
    if not vals:
        return 0.0
    tw = sum(wts)
    m = sum(v * w for v, w in zip(vals, wts)) / tw
    return m / (sur.scales.get(position, 1.0) or 1.0)


def main():
    if RAW.endswith(".raw.json"):
        pairs = load_raw_corpus(RAW)
        sur = FitnessSurrogate().fit(pairs)
        print(f"corpus: {RAW}  ({len(pairs)} pairs, fitted in memory)")
    else:
        sur = FitnessSurrogate.load(RAW)
        print(f"surrogate: {RAW}  (loaded from the fitted file)")
    scales = dict(sur.scales)

    print(f"corpus states: {N_STATES}, seed {SEED}  (production config)")
    print()
    print(f"guard budget: indecision {INDECISION_MAX:.3f} + confidence "
          f"{CONF_RANGE:.3f} + diversity {DIV_RANGE:.3f} = {GUARD_BUDGET:.3f}")
    print(f"largest single guard: {LARGEST_GUARD:.3f}")
    print()
    print(f"{'pos':<6}{'raw swing':>12}{'scaled swing':>14}"
          f"{'raw/guard':>12}{'scaled/guard':>15}{'scaled mean':>13}"
          f"{'verdict':>12}")
    bad = []
    for pos in sorted(p for p in sur.table if p):
        corpus = generate_state_corpus(pos, N_STATES, SEED, 0.0)
        sur.scales = {}                       # the BEFORE arm: no division
        raw = _swing(sur, corpus, pos)
        sur.scales = dict(scales)             # the AFTER arm: scaled
        scaled = _swing(sur, corpus, pos)
        ratio_raw = raw / LARGEST_GUARD
        ratio_scaled = scaled / LARGEST_GUARD
        verdict = "REWARD" if ratio_scaled > 1.0 else "GUARD"
        if ratio_scaled <= 1.0:
            bad.append(pos)
        print(f"{pos:<6}{raw:>12.5f}{scaled:>14.5f}"
              f"{ratio_raw:>12.2f}{ratio_scaled:>15.2f}"
              f"{_scaled_mean(sur, pos):>13.5f}{verdict:>12}")
    sur.scales = dict(scales)

    print()
    print("The 'scaled mean' column is the count-weighted mean of the "
          "position's SCALED cells")
    print("- i.e. where a random choice among that position's own actions "
          "sits.")
    print("If it is far from 0.0, `clip(mean_reward, -1, 1)` can SATURATE "
          "for every brain")
    print("and the reward's ranking is flattened.  A position whose scaled "
          "mean is")
    print("beyond +-1.0 has no gradient left at all.")

    print()
    if bad:
        print(f"VERDICT: the guards still decide for {bad}. "
              f"The per-position scale does NOT fix the balance — "
              f"do not spend 30 minutes of evolution on it.")
    else:
        print("VERDICT: for every position the reward's swing now exceeds "
              "the largest single guard term, so the guards are a "
              "tie-breaker rather than the objective.")
    print()
    print("NOTE: the 'raw swing' column is the reward's reach WITHOUT the "
          "per-position scale (the v5 situation); the 'scaled swing' column "
          "is the same corpus with the scale applied.")


if __name__ == "__main__":
    sys.exit(main())
