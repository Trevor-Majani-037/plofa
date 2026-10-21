"""Smoke: the threat reward, the schema guard, and the re-based fitness.

Zero matches. Fast. Every check prints its value so a wrong table cannot hide
above a right one.
"""
from __future__ import annotations

import random
from types import SimpleNamespace as NS

import numpy as np

import threat_reward as TR
from threat_reward import threat_return, default_grid, describe
from decision_brain import PlayerIntent
from football_brain import FootballBrain


def ev(et, **kw):
    d = dict(event_type=NS(name=et), player="A", team="Home", minute=10,
             second=0, outcome=None, location_x=50.0, location_y=34.0,
             end_x=None, end_y=None, xg=None, metadata={})
    d.update(kw)
    return NS(**d)


print("=" * 62)
print("A. REWARD — SIGN, ORDERING, SCALE")
print("=" * 62)
grid = default_grid()
print("grid rows/cols:", len(grid), len(grid[0]))

cases = [
    ("goal", ev("GOAL", location_x=95, location_y=34, xg=0.4)),
    ("shot on target (xg .35)", ev("SHOT_ON_TARGET", location_x=95, location_y=34, xg=0.35)),
    ("shot off target (xg .35)", ev("SHOT_OFF_TARGET", location_x=95, location_y=34, xg=0.35)),
    ("shot blocked (xg .35)", ev("SHOT_BLOCKED", location_x=95, location_y=34, xg=0.35)),
    ("shot, no xg (zone)", ev("SHOT_ON_TARGET", location_x=95, location_y=34)),
    ("fwd pass 40->75", ev("PASS", outcome=True, location_x=40, location_y=34, end_x=75, end_y=34)),
    ("fwd pass 40->52", ev("PASS", outcome=True, location_x=40, location_y=34, end_x=52, end_y=34)),
    ("square pass", ev("PASS", outcome=True, location_x=60, location_y=34, end_x=60, end_y=42)),
    ("back pass 70->45", ev("PASS", outcome=True, location_x=70, location_y=34, end_x=45, end_y=34)),
    ("through ball 55->88", ev("THROUGH_BALL", outcome=True, location_x=55, location_y=34, end_x=88, end_y=34)),
    ("failed through (their 3rd)", ev("THROUGH_BALL", outcome=False, location_x=70, location_y=34, end_x=88, end_y=34)),
    ("failed pass (our 3rd)", ev("PASS", outcome=False, location_x=20, location_y=34, end_x=30, end_y=34)),
    ("turnover (our box)", ev("TURNOVER", location_x=8, location_y=34)),
    ("turnover (their half)", ev("TURNOVER", location_x=80, location_y=34)),
    ("carry 40->50", ev("CARRY", outcome=True, location_x=40, location_y=34, end_x=50, end_y=34)),
    ("cross attempt ok", ev("CROSS_ATTEMPT", outcome=True, location_x=90, location_y=6, end_x=98, end_y=38)),
    ("foul (neutral)", ev("FOUL_COMMITTED", location_x=50, location_y=34)),
]
vals = {}
for name, e in cases:
    v = threat_return(e, grid=grid, attacks_right=True)
    vals[name] = v
    print(f"  {name:32s} {v:+.4f}")

print()
checks = []
checks.append(("goal beats best shot", vals["goal"] > vals["shot on target (xg .35)"]))
checks.append(("on target > off target", vals["shot on target (xg .35)"] > vals["shot off target (xg .35)"]))
checks.append(("fwd 40->75 > fwd 40->52", vals["fwd pass 40->75"] > vals["fwd pass 40->52"]))
checks.append(("forward > square > backward",
               vals["fwd pass 40->52"] > vals["square pass"] > vals["back pass 70->45"]))
checks.append(("back pass negative", vals["back pass 70->45"] < 0))
checks.append(("loss in own box < loss in their half",
               vals["turnover (our box)"] < vals["turnover (their half)"]))
checks.append(("turnover is not free (negative)", vals["turnover (our box)"] < 0))
checks.append(("failed through <= 0", vals["failed through (their 3rd)"] <= 0))
checks.append(("neutral event exactly 0", vals["foul (neutral)"] == 0.0))
checks.append(("reward within [-1, 1]", all(-1.0 <= v <= 1.0 for v in vals.values())))
for n, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'} {n}")
assert all(ok for _, ok in checks), "reward ordering broken"

# telescoping: A->B->C should equal A->C
a = ev("PASS", outcome=True, location_x=30, location_y=34, end_x=55, end_y=34)
b = ev("PASS", outcome=True, location_x=55, location_y=34, end_x=85, end_y=30)
ab = threat_return(a, grid) + threat_return(b, grid)
ac = threat_return(ev("PASS", outcome=True, location_x=30, location_y=34,
                      end_x=85, end_y=30), grid)
print(f"  telescoping: {ab:+.6f} vs {ac:+.6f}  diff {abs(ab-ac):.2e}")
assert abs(ab - ac) < 1e-12, "potential difference does not telescope"

# mirrored: away team attacking left, same geometry, same value
away_a = ev("PASS", outcome=True, location_x=75, location_y=34, end_x=105 - 40, end_y=34)
away_a.location_x = 105 - 40
away_a.end_x = 105 - 75
print(f"  mirror symmetry: {threat_return(away_a, grid, False):+.5f} vs "
      f"{vals['fwd pass 40->75']:+.5f}")
assert abs(threat_return(away_a, grid, False) - vals["fwd pass 40->75"]) < 1e-12
print("  OK  mirror symmetry")

print()
print("reward definition:", describe()["form"])


print()
print("=" * 62)
print("B. SCHEMA ALIGNMENT — the silent truncation is now a hard error")
print("=" * 62)
import role_features as RF
from brain_schema import BrainMeta

RF.reset_alignment_warnings()
cb_names = list(RF.MENU_NAMES["CB"])
print("CB schema names:", cb_names, "-> width", len(cb_names))
full = [float(i) + 1.0 for i in range(len(cb_names))]

legacy = RF.align_role_tail("CB", full, 7)
byname = RF.align_role_tail("CB", full, 7, want_names=cb_names[:7])
print("legacy 7  :", legacy)
print("by-name 7 :", byname)
assert np.allclose(legacy, byname), "legacy and by-name alignment disagree for the shipped CB brain"
print("  OK  legacy == by-name for the shipped CB brain")

# warn exactly once for (family, width), no names declared
RF.reset_alignment_warnings()
import warnings as _w
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    RF.align_role_tail("CB", full, 7)
    RF.align_role_tail("CB", full, 7)
print(f"  warnings raised for two legacy calls: {len(caught)} (expect 1)")
assert len(caught) == 1

for bad, why in [
    (dict(want_names=["nope_a"] * 7), "unknown name"),
    (dict(want_names=cb_names), "declared count != width"),
]:
    try:
        RF.align_role_tail("CB", full, 7, **bad)
    except RF.RoleSchemaError as ex:
        print(f"  OK  raises on {why}: {str(ex)[:64]}")
    else:
        raise AssertionError(f"did not raise on {why}")

exact = RF.align_role_tail("CB", full, 9, want_names=cb_names)
print("  exact-width 9:", exact)
assert np.allclose(exact, full)

try:
    RF.align_role_tail("CB", full, 9, want_names=cb_names[:5])
except RF.RoleSchemaError as ex:
    print(f"  OK  raises when a declaration is narrower than the arch: {str(ex)[:56]}")
else:
    raise AssertionError("a 5-name declaration in a 9-wide tail should be refused")

# role=None with a declaration is legitimate: an untracked player.
zeros = RF.align_role_tail("CB", None, 9, want_names=cb_names)
print("  declared but no role values ->", zeros)
assert np.allclose(zeros, 0.0)
print("  OK  a declared tail with no values is zeros, not a crash")


print()
print("=" * 62)
print("C. SURROGATE + RE-BASED FITNESS")
print("=" * 62)
from surrogate_collect import FitnessSurrogate, _event_success, _EXECUTION_TYPES
from brain_evolution import synthetic_fitness

print("execution types:", len(_EXECUTION_TYPES))
for t in ("PASS", "PROGRESSIVE_PASS", "THROUGH_BALL", "SWITCH_OF_PLAY",
          "CARRY", "CROSS_ATTEMPT", "DRIBBLE_FAIL", "GOAL"):
    assert t in _EXECUTION_TYPES, t
print("  OK  progressive / through / switch / cross are counted as executions")
assert "CROSS_SUCCESS" not in _EXECUTION_TYPES, "CROSS_SUCCESS would double-count a cross"
assert "SHOT_ATTEMPT" not in _EXECUTION_TYPES, "SHOT_ATTEMPT would mask the resolution"

back = ev("PASS", outcome=True, location_x=70, location_y=34, end_x=45, end_y=34)
fwdbe = ev("PASS", outcome=True, location_x=40, location_y=34, end_x=75, end_y=34)
assert _event_success(back) < 0 < _event_success(fwdbe)
print(f"  OK  legacy alias is signed: back {_event_success(back):+.3f} < 0 < fwd {_event_success(fwdbe):+.3f}")

sur = FitnessSurrogate(prior=0.0)
s = np.zeros(24)
sur.table["ST"] = {"p1":{"high": {PlayerIntent.SHOOT.value: 0.30,
                                  PlayerIntent.RECYCLE.value: -0.05}}}
sur.counts["ST"] = {"p1":{"high": {PlayerIntent.SHOOT.value: 50,
                                   PlayerIntent.RECYCLE.value: 50}}}
print("  prior:", sur.prior)

random.seed(3)
brain = FootballBrain.random(3)
sur_flat = FitnessSurrogate(prior=0.0)
sur_flat.table["ST"] = {"z": {"z": {PlayerIntent.CARRY.value: 0.0}}}
f = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=sur_flat)
print(f"  synthetic_fitness (all-zero table): {f:.4f}")
assert 0.0 <= f <= 1.0

# a NEGATIVE mean must not clamp to zero -> the gradient survives
sur_neg = FitnessSurrogate(prior=-0.5)
sur_neg.table["ST"] = {"z": {"z": {i.value: -0.5 for i in PlayerIntent}}}
f_neg = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=sur_neg)
sur_pos = FitnessSurrogate(prior=0.5)
sur_pos.table["ST"] = {"z": {"z": {i.value: 0.5 for i in PlayerIntent}}}
f_pos = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=sur_pos)
print(f"  mean -0.5 -> {f_neg:.4f}    mean +0.5 -> {f_pos:.4f}")
assert f_neg > 0.0, "a negative mean clamped to zero: the gradient is gone"
assert f_pos > f_neg

# collapse penalties off by default, and they cost something when on
f_off = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=sur_pos)
f_on = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=sur_pos,
                         collapse_penalties=True)
print(f"  collapse penalties OFF {f_off:.4f}  ON {f_on:.4f}")
assert f_on < f_off, "the penalty flag has no effect"

print()
print("ALL CHECKS PASSED")
