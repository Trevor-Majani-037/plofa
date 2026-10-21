"""WHAT IS THE SURROGATE TELLING THE GA TO DO — ON THE GA'S OWN CORPORA.

The census is the gate on a retrain, but a census failure has TWO possible
causes and they need different fixes:

  (a) the TABLE is wrong — the surrogate says one action wins in every
      situation, so the GA is right to converge on it and the OBJECTIVE is
      the problem; or
  (b) the TABLE is fine — it prefers different actions in different
      situations — and the GA failed to find them, which is a search
      problem.

This probe separates them.  For every position it walks a corpus and reads
the surrogate the way the GA reads it — through `expected_success`, with the
role-family and global fallbacks — and reports, per intent:

    mean     the average reward the GA would collect by always choosing it
    argmax   the share of states in which it IS the best available action

and then THE HEADROOM: the per-state oracle minus the best single-intent
constant.  That gap is the only thing the GA can win by being
state-dependent.

⚠️  THE CORPUS MUST BE THE GA'S OWN.  `brain_evolution.evolve` re-generates
the corpus EVERY generation, with `seed + gen * 1000 + i` for population
member i — so a brain is scored on a DIFFERENT corpus every generation, and
brain i is always compared against brain j on corpus j's states.  A corpus
built with the default arguments (24-d, no perception) is a DIFFERENT set of
situations, and the constants measured on it are the wrong ones: the
surrogate's bucket composition, and therefore the best constant, moves with
the corpus.

This probe therefore uses the production configuration exactly —
role width, perception on, and the per-generation seeds — and reports the
headroom ACROSS generations, because that is the spread the GA lives with.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_intent_value.py \\
        brains/surrogate_pos_v6.json [--ga]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from brain_evolution import generate_state_corpus  # noqa: E402
from decision_brain import PlayerIntent  # noqa: E402
from role_features import ROLE_INTENTS, V2_INPUT_D, _role_family  # noqa: E402
from surrogate_collect import FitnessSurrogate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT = os.path.join(ROOT, "brains", "surrogate_pos_v6.json")

N_STATES = 300
SEED = 123
POP = 32

#: the generations whose corpora are sampled (the GA re-samples every one)
GENERATIONS = [0, 1, 10, 40, 79]


def _corpus(pos, gen, member):
    """The EXACT corpus the GA scored population member `member` in `gen`."""
    from perception import PerceptionConfig
    fam = _role_family(pos) or ""
    width = V2_INPUT_D.get(fam, 24)
    cfg = PerceptionConfig(enabled=True, role_blocks=True, seed=SEED)
    return generate_state_corpus(pos, N_STATES, SEED + gen * 1000 + member,
                                 0.0, input_size=width,
                                 perception_config=cfg)


def _stats(sur, corpus, pos, allowed):
    sums = {i.value: 0.0 for i in allowed}
    wins = {i.value: 0 for i in allowed}
    per_state_max = []
    for row in corpus:
        vals = []
        for intent in allowed:
            v = float(sur.expected_success(row[:24], intent, pos))
            sums[intent.value] += v
            vals.append((v, intent.value))
        if vals:
            per_state_max.append([v for v, _ in vals])
            wins[max(vals)[1]] += 1
    n = len(corpus)
    best_const = max(sums[i.value] for i in allowed) / n
    oracle = sum(max(v) for v in per_state_max) / n
    top = max(wins, key=lambda k: wins[k])
    return {
        "sums": sums, "wins": wins, "n": n,
        "best_const": best_const, "oracle": oracle,
        "headroom": oracle - best_const,
        "top": top, "top_share": wins[top] / n,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    ga_mode = "--ga" in sys.argv
    sur = FitnessSurrogate.load(path)
    print(f"surrogate: {path}")
    if ga_mode:
        print("corpus: the GA's own (role width, perception on, "
              "per-generation seeds)")
    else:
        print("corpus: 300 states, seed 123, plain (NO role width, NO "
              "perception)")
    print()

    for pos in sorted(p for p in sur.table if p):
        fam = _role_family(pos) or ""
        allowed = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, ())]
        corpora = []
        if ga_mode:
            for gen in GENERATIONS:
                corpora.append((f"gen{gen}/m0", _corpus(pos, gen, 0)))
                corpora.append((f"gen{gen}/m{POP - 1}",
                                _corpus(pos, gen, POP - 1)))
        else:
            from brain_evolution import INPUT_SIZE
            corpora.append(("plain",
                            generate_state_corpus(pos, N_STATES, SEED, 0.0,
                                                  input_size=INPUT_SIZE)))
        print(f"=== {pos} (family {fam}) ===")
        for label, corpus in corpora:
            st = _stats(sur, corpus, pos, allowed)
            print(f"    {label:<14} best constant {st['best_const']:+.4f} | "
                  f"oracle {st['oracle']:+.4f} | "
                  f"HEADROOM {st['headroom']:+.4f} | "
                  f"top {st['top']} {100.0 * st['top_share']:.1f}%")
        st = _stats(sur, corpora[0][1], pos, allowed)
        print("    per-intent on the first corpus:")
        for intent in sorted(allowed, key=lambda i: -st["wins"][i.value]):
            print(f"      {intent.value:<20} argmax "
                  f"{100.0 * st['wins'][intent.value] / st['n']:>5.1f}%   "
                  f"mean {st['sums'][intent.value] / st['n']:+.4f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
