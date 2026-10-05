"""What does the SHIPPED brains/CB.json actually see after the 2 new CB sensors?

Production brains/CB.json is arch [31,32,32,10] = 24 shared + 7 CB features
trained by GA.  _cb_block now returns 9.  brain_integration decides which
schema to use by `n_in` and then pads/truncates the built vector to n_in
(brain_integration.py:524-527).

This probe answers three questions, no guessing:
  A. What input width does the loader feed the shipped CB brain?
  B. Are the first 31 components identical to the pre-change 7-feature block?
  C. Do the new features survive at all?

Also checks whether the schema validator still accepts the on-disk file.
"""
from __future__ import annotations

import json

import numpy as np

import brain_schema
import role_features as rf

OUT = "_diag_cb_sensor_dim.txt"
lines: list[str] = []


def p(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


# ── on-disk arch ───────────────────────────────────────────────────────────
data = json.load(open("brains/CB.json"))
arch = tuple(data["arch"])
w1 = np.array(data["w1"])
p(f"brains/CB.json arch          : {arch}   w1 {w1.shape}")
p(f"V2_INPUT_D['CB'] now         : {rf.V2_INPUT_D['CB']}")
p(f"V2_ROLE_D['CB']  now         : {rf.V2_ROLE_D['CB']}  (was 7)")
p(f"allowed input widths v2      : {sorted(set(rf.V2_INPUT_D.values()))}")

# ── A/B the block itself, same synthetic scene ────────────────────────────
OLD_NAMES = rf.MENU_NAMES["CB"][:7]
p()
p(f"new CB menu ({len(rf.MENU_NAMES['CB'])}) : {rf.MENU_NAMES['CB']}")
p(f"first 7 (unchanged, in order): {OLD_NAMES}")


class P:
    def __init__(self, pos, x, y):
        self.name, self.position, self.x, self.y = "X", pos, x, y


class PE:
    """Minimal position engine stand-in: _pos() only needs (x, y)."""

    def get_position(self, name):
        return (50.0, 34.0)


def block_now():
    return rf._cb_block(P("CB1", 15.0, 15.0), 15.0, 15.0, [], [], PE(), True)


def block_old():
    """Recompute with the 9-feature block truncated to the original 7."""
    return rf._cb_block(P("CB1", 15.0, 15.0), 15.0, 15.0, [], [], PE(), True)[:7]


full = block_now()
old = block_old()
p()
p(f"block now  ({len(full)}): {[round(v, 4) for v in full]}")
p(f"block old  ({len(old)}): {[round(v, 4) for v in old]}")

# ── A: what width does the loader actually feed? ───────────────────────────
n_in = w1.shape[0]
vec_full = rf.build_v2_vector(np.zeros(24), full)
loader_sees = np.zeros(n_in)
k = min(vec_full.shape[0], n_in)
loader_sees[:k] = vec_full[:n_in]
p()
p(f"A. loader feeds n_in={n_in} to the shipped brain")
p(f"   built vector width        : {vec_full.shape[0]}")
p(f"   after pad/truncate        : {loader_sees.shape[0]}")
dropped = vec_full[n_in:]
p(f"   components DROPPED        : {len(dropped)}  "
  f"{rf.MENU_NAMES['CB'][len(old):len(old) + len(dropped)]}")
p(f"   dropped values            : {[round(float(v), 4) for v in dropped]}")
p(f"   A -> new features reach the shipped brain? "
  f"{'NO - truncated away' if len(dropped) else 'yes'}")

# ── B: is the shipped brain's input bit-identical to before the change? ───
shared = np.linspace(0.1, 0.9, 24)
pre_change_vec = rf.build_v2_vector(shared, old)
post_change_vec = rf.build_v2_vector(shared, full)
pad = lambda v: np.concatenate([v, np.zeros(max(0, n_in - v.shape[0]))])[:n_in]
identical = np.array_equal(pad(pre_change_vec), pad(post_change_vec))
p()
p(f"B. pre-change padded vector == post-change padded vector? "
  f"{'IDENTICAL' if identical else 'CHANGED'}")
if not identical:
    d = np.abs(pad(pre_change_vec) - pad(post_change_vec))
    p(f"   max abs delta: {d.max():.6g} at index {int(d.argmax())}")

# ── C: does the argmax intent change on a randomised sweep? ───────────────
rng = np.random.default_rng(7)
diffs = 0
trials = 4000
for _ in range(trials):
    s = rng.random(24)
    r = rng.random(9)
    a = w1.T
    pre = pad(rf.build_v2_vector(s, r[:7]))
    post = pad(rf.build_v2_vector(s, r))
    pa = np.argmax(pre)
    pb = np.argmax(post)
    if pa != pb:
        diffs += 1
p()
p(f"C. argmax differs on {diffs}/{trials} random inputs")
p(f"   -> intent distribution {'MOVES' if diffs else 'UNCHANGED'} "
  f"with the shipped brain")

# ── D: schema validator on the untouched on-disk file ─────────────────────
try:
    brain_schema.validate(json.load(open("brains/CB.json")))
    p()
    p("D. brains/CB.json still validates: YES")
except Exception as exc:  # noqa: BLE001
    p()
    p(f"D. brains/CB.json validation FAILED: {type(exc).__name__}: {exc}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")