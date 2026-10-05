"""WHICH module-level state breaks seed reproducibility? (2026-10-04)

The project has carried a standing claim for weeks: a match is not
reproducible from `random.seed`, the cause is module-level brain/mind caches,
and "the fix is three lines at the top of `_initialize_simulation()`". The
cause was never measured, and AGENTS.md's own note flags that the fix needs a
judgement call because the state might be MEANT to persist.

This probe measures instead of assuming.

METHOD. One process. Run 1 is the reference — it is first, so nothing can have
contaminated it. Then for each arm: reset the seed to the same value, clear the
arm's candidates, simulate again, and diff the timeline signature against run 1.
Any arm whose signature matches run 1 has identified a necessary clearing; an
arm that still diverges has not.

The signature is per-event (type, player, secondary, x, y, minute) so a
divergence report can name the actual event where the two matches parted
company, which is worth far more than a boolean.

NEGATIVE CONTROL. The `--arm none` arm clears nothing and is EXPECTED to
diverge. If it matches, then the non-reproducibility claim is wrong and every
multi-match sweep in this project was fine after all. A probe whose control
passes is a probe whose measurement means nothing.

NUMPY. AGENTS.md notes the live path was suspected of `np.random.default_rng()`,
which is entropy-seeded and ignores `np.random.seed()`. Rather than assume,
`--arm numpy` counts how many times `np.random.default_rng` is called during a
match. Zero calls exonerates numpy outright.

WHAT THIS DOES NOT DO. It does not decide the persistence question; it
identifies the state. What to DO about it depends on intent, and the intent is
readable in the source: `auto_run_match.py:1157` hydrates minds from the season
store "carrying last matchday's temperament" and clears `_minds` before and
after, so minds are meant to persist THROUGH `season_state`, not through the
module dict.
"""
import argparse
import hashlib
import random
import sys

sys.path.insert(0, ".")

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--arm", default="all",
                help="comma list of: none minds registry pos_brain "
                     "all  (default: run every arm)")
A = ap.parse_args()


def sig(res):
    rows = []
    for e in res.timeline:
        rows.append("|".join([
            e.event_type.name,
            getattr(e, "player", "") or "",
            getattr(e, "secondary_player", "") or "",
            f"{getattr(e, 'location_x', 0.0):.1f}",
            f"{getattr(e, 'location_y', 0.0):.1f}",
            f"{getattr(e, 'minute', 0.0):.2f}",
        ]))
    return rows


def digest(rows):
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]


def clear_registry():
    import brain_integration as bi
    bi._brain_registry.clear()
    bi._pos_brain_cache.clear()


def clear_minds():
    from cognition.mind import clear_minds as _cm
    _cm()


ARMS = {
    "none": (),
    "minds": (clear_minds,),
    "registry": (clear_registry,),
    "pos_brain": (lambda: __import__("brain_integration")._pos_brain_cache.clear(),),
    "all": (clear_minds, clear_registry),
}

from _diag_chance_coords import build_pair  # noqa: E402

want = (list(ARMS) if A.arm == "all" else A.arm.split(","))

# ── numpy: is the live path even reaching default_rng? ────────────
_DRG = [0]
_orig_drg = np.random.default_rng


def counted_drg(*a, **kw):
    _DRG[0] += 1
    return _orig_drg(*a, **kw)


np.random.default_rng = counted_drg

# ── reference run, first in the process ───────────────────────────
random.seed(A.seed)
np.random.seed(A.seed)
_DRG[0] = 0
ref = sig(build_pair("Oxton", "Natrican").simulate())
print(f"reference  seed={A.seed}  events={len(ref)}  "
      f"digest={digest(ref)}  np.default_rng calls={_DRG[0]}", flush=True)

results = {}
for arm in want:
    fns = ARMS[arm]
    random.seed(A.seed)
    np.random.seed(A.seed)
    for fn in fns:
        fn()
    got = sig(build_pair("Oxton", "Natrican").simulate())
    same = got == ref
    results[arm] = same
    tag = "MATCHES reference" if same else "DIVERGES"
    print(f"  arm {arm:<10} events={len(got):<5} digest={digest(got)}  {tag}",
          flush=True)
    if not same:
        n = min(len(ref), len(got))
        first = next((i for i in range(n) if ref[i] != got[i]), n)
        print(f"      first divergence at event {first} of {n} common")
        if first < n:
            print(f"        ref: {ref[first]}")
            print(f"        got: {got[first]}")
        print(f"      length delta {len(got) - len(ref):+d} events")

print()
print("=" * 66)
print("negative control: 'none' MUST diverge for the claim to hold")
print("  none diverges :", "PASS" if results.get("none") is not True
      else "FAIL — the premise is wrong")
for arm, same in results.items():
    if arm == "none":
        continue
    # A run that MATCHES the reference proves the clearing was unnecessary.
    # This label was inverted on the first run and printed "YES: necessary"
    # for every arm that matched -- i.e. it contradicted its own raw verdicts
    # one line above. An inverted summary is the same failure as a probe whose
    # control passes silently: the output reads clean and says the opposite
    # of the truth.
    print(f"  {arm:<10} clearing was {'NOT ' if same else ''}necessary: "
          f"{'arm matched reference' if same else 'arm diverged'}")
print("=" * 66)