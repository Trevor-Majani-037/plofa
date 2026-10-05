"""WHAT is the PYTHONHASHSEED dependency? (2026-10-04)

Established by measurement already (`_diag_repro_process.py`):

  in-process, 2nd simulate(), nothing cleared ....... byte-identical
  cross-process, PYTHONHASHSEED=0, three runs ....... byte-identical x3
  cross-process, PYTHONHASHSEED 0 / 1 / 12345 ....... three different matches

So the hash seed is the ENTIRE cross-process reproducibility story, and the
question is what on the live path actually reads it.

Mechanism, stated first so the measurement can be checked against it.
PYTHONHASHSEED salts `str.__hash__`. Two things and only two are affected:

  * iteration order of a `set`/`frozenset` of strings
  * anything computed from `hash(some_string)` directly

`dict` is NOT affected -- it has been insertion-ordered since 3.7, so
`for k in d` and `list(d.keys())` are stable regardless of the seed. That is
why a codebase with no `for x in {...}` in it can still be
hash-order-dependent: the set is usually *behind* a dict, and the dict's
INSERTION order is what got poisoned. A dict populated by iterating an
unordered collection inherits that collection's order and is itself
deterministic-but-arbitrary.

`min()`/`max()` over a set, `set.pop()`, and `random.shuffle(list(s))` are the
same defect wearing different clothes.

METHOD. Dump the event signature to a file under one hash seed, then diff two
dumps and read the FIRST divergent index. That names an event, and an event
names a code path. No guessing at candidates, which is what the greps for
`for x in {...}` just did -- they found nothing, because the route is indirect.

  --mode run  --out F            one match, dump to F
  --mode diff --a F --b F        first divergence + context around it

Diff context is +/-8 events because the interesting signal is not the
divergent event itself but WHAT it was: if the first divergence is a player
choice, the cause is an ordered collection of players; if it is an event
COUNT with identical leading events, the cause is a branch that consumed a
different number of RNG draws.
"""
import argparse
import hashlib
import json
import random
import sys

sys.path.insert(0, ".")

# cp1252 cannot encode real roster names -- 'Perćy Luka', 'Mirev Jr' with a
# hook above the k -- and this crashed the diff AFTER it had already printed
# the first divergence, losing the tail analysis. The same trap is already
# fixed once in this repo (`alltime_db._make_console_utf8_safe`) for exactly
# this reason. A probe that dies on the last thing it was asked to report is a
# probe that reports nothing next time.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("run", "diff"), default="run")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--out", default="_diag_hash_run.json")
ap.add_argument("--a", default=None)
ap.add_argument("--b", default=None)
ap.add_argument("--ctx", type=int, default=8)
A = ap.parse_args()


def sig_of(res):
    rows = []
    for e in res.timeline:
        md = e.metadata or {}
        rows.append({
            "t": e.event_type.name,
            "p": getattr(e, "player", "") or "",
            "s": getattr(e, "secondary_player", "") or "",
            "x": round(float(getattr(e, "location_x", 0.0) or 0.0), 2),
            "y": round(float(getattr(e, "location_y", 0.0) or 0.0), 2),
            "m": round(float(getattr(e, "minute", 0.0) or 0.0), 3),
            # metadata keys only -- VALUES may contain player names picked by
            # an unordered collection, which would make every row differ and
            # hide the first real divergence under a cloud of noise.
            "k": sorted(md)[:12],
        })
    return rows


def key(r):
    return (r["t"], r["p"], r["s"], r["x"], r["y"], r["m"], tuple(r["k"]))


if A.mode == "run":
    import os
    random.seed(A.seed)
    try:
        import numpy as np
        np.random.seed(A.seed)
    except Exception:
        pass
    from _diag_chance_coords import build_pair
    res = build_pair("Oxton", "Natrican").simulate()
    rows = sig_of(res)
    payload = {
        "seed": A.seed,
        "hashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "n": len(rows),
        "digest": hashlib.sha1(
            "|".join("/".join(map(str, key(r))) for r in rows).encode()
        ).hexdigest()[:12],
        "rows": rows,
    }
    with open(A.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"hashseed={payload['hashseed']} events={payload['n']} "
          f"digest={payload['digest']} -> {A.out}", flush=True)
    sys.exit(0)

# ── diff mode ─────────────────────────────────────────────────────
A_ = json.load(open(A.a, encoding="utf-8"))
B_ = json.load(open(A.b, encoding="utf-8"))
ra, rb = A_["rows"], B_["rows"]
print(f"A: hashseed={A_['hashseed']} n={A_['n']} digest={A_['digest']}")
print(f"B: hashseed={B_['hashseed']} n={B_['n']} digest={B_['digest']}")
print()

n = min(len(ra), len(rb))
first = next((i for i in range(n) if key(ra[i]) != key(rb[i])), None)

if first is None and len(ra) == len(rb):
    print("IDENTICAL over the common prefix and the same length.")
    sys.exit(0)

if first is None:
    print(f"No divergence in the first {n} events; B ran "
          f"{len(rb) - len(ra):+d} events longer.")
    first = n

lo = max(0, first - A.ctx)
hi = min(n, first + A.ctx + 1)
print(f"FIRST DIVERGENCE at event {first} of {n} common "
      f"(showing {lo}..{hi - 1}, '>>>' marks the break)")
print()
for i in range(lo, hi):
    ka, kb = key(ra[i]), key(rb[i])
    mark = ">>>" if ka != kb else "   "
    print(f"{mark} [{i:4}] A {ka}")
    print(f"    [{i:4}] B {kb}")
    if ka[0] != kb[0]:
        print(f"         ^^ EVENT TYPE differs -- {ka[0]} vs {kb[0]}")
    elif ka[1] != kb[1]:
        print(f"         ^^ PLAYER differs -- {ka[1]!r} vs {kb[1]!r}")

# how many agree on the prefix, and how the type mix diverges after
agree = sum(1 for i in range(first) if key(ra[i]) == key(rb[i]))
print()
print(f"prefix identical for {agree} events before the break")
tail = slice(first, first + 400)
ta = {}
tb = {}
for r in ra[tail]:
    ta[r["t"]] = ta.get(r["t"], 0) + 1
for r in rb[tail]:
    tb[r["t"]] = tb.get(r["t"], 0) + 1
moved = sorted(set(ta) | set(tb),
               key=lambda k: -abs(ta.get(k, 0) - tb.get(k, 0)))[:10]
print(f"event-type mix in the {len(ra[tail])}/{len(rb[tail])} events "
      f"after the break:")
for k in moved:
    print(f"    {k:<22} A {ta.get(k, 0):>5}   B {tb.get(k, 0):>5}")
