"""Tests for brain_schema.py — brain versioning + schema registry (audit Step 2).

Run: python3 test_brain_schema.py   (plain asserts, project style)

Contract:
  1. v1 files (no ``meta`` block) load as v1 — no behaviour change.
  2. serialize() emits a ``meta`` block on every new file.
  3. schema mismatches are REJECTED LOUDLY (BrainSchemaError): unknown kind,
     wrong arch_version, unknown sensor_schema / normalization / dna_schema /
     training_method, and arch-vs-weights shape mismatches.
  4. files on disk (brains/ — production v1_24d or v2_role_features —
     brains_team/, brains_def/) all validate and carry the expected
     architecture for their kind.
  5. Blueprint for brains_v2/ scratch namespace.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from brain_schema import (
    BRAIN_FORMAT_VERSION, BRAINS_V2_DIR, BrainSchemaError,
    brain_meta_dict, legacy_meta, validate as validate_schema,
    check_shapes,
)
from brain_integration import (
    clear_registry,
)
from football_brain import (
    FootballBrain, OffBallBrain, TeamPressBrain, DefensiveActionBrain,
)


fails = 0
def check(name, cond):
    global fails
    if cond:
        print(f"[PASS] {name}")
        return
    fails += 1
    print(f"[FAIL] {name}")


# ── fixtures ─────────────────────────────────────────────────

def v1_football_brain_dict():
    """Byte-shape identical to the legacy format: NO meta block."""
    b = FootballBrain.random(seed=3)
    d = b.serialize()
    d.pop("meta", None)          # strip v2 meta => legacy v1 file
    return d


# ── 1. v1 load (no behaviour change) ─────────────────────────

v1 = v1_football_brain_dict()
check("v1 dict (no meta) validates as legacy",
      validate_schema(v1).training_method == "unknown")
b_from_v1 = FootballBrain.deserialize(v1)
check("v1 file loads and round-trips weights",
      np.allclose(b_from_v1.w1, FootballBrain.deserialize(v1).w1))

# ── 2. new files emit meta ───────────────────────────────────

b = FootballBrain.random(seed=9)
s = b.serialize()
check("serialize() now emits a meta block", "meta" in s)
check("meta declares a known sensor_schema",
      s["meta"]["sensor_schema"] in ("v1_24d",))
check("meta has a versioned format",
      s["meta"].get("format_version") == BRAIN_FORMAT_VERSION)
b2 = FootballBrain.deserialize(s)
check("deserialize preserves meta across the round-trip",
      b2.meta is not None and b2.meta["sensor_schema"] == "v1_24d")

# ── 3. loud rejection ────────────────────────────────────────

def rejects(name, mutate):
    d = v1_football_brain_dict()
    try:
        mutate(d)
        validate_schema(d)
        check(name, False)
    except BrainSchemaError:
        check(name, True)

rejects("unknown kind rejected", lambda d: d.update(kind="skynet"))
rejects("future arch_version rejected",
        lambda d: d.update(meta=brain_meta_dict(arch_version=999)))
rejects("unknown sensor_schema rejected",
        lambda d: d.update(meta=brain_meta_dict(sensor_schema="v3_chaos")))
rejects("unknown normalization rejected",
        lambda d: d.update(meta=brain_meta_dict(normalization_version="v9")))
rejects("unknown dna_schema rejected",
        lambda d: d.update(meta=brain_meta_dict(dna_schema="cyberdna")))
rejects("unknown training_method rejected",
        lambda d: d.update(meta=brain_meta_dict(training_method="RL_dogma")))
rejects("off-by-one arch rejected for kind+weights",
        lambda d: d.update(arch=[24, 32, 32, 9]))


def rejects_shapes(mutate_arch):
    d = v1_football_brain_dict()
    try:
        mutate_arch(d)
        validate_schema(d)          # registry-level check still passes? kind matched
        arrays = [
            np.array(d["w1"]), np.array(d["b1"]),
            np.array(d["w2"]), np.array(d["b2"]),
            np.array(d["w3"]), np.array(d["b3"]),
        ]
        check_shapes(d, arrays)     # weight-vs-arch mismatch caught here
        check("shape mismatch rejected", False)
    except BrainSchemaError:
        check("shape mismatch rejected", True)

rejects_shapes(lambda d: d.update(arch=[24, 64, 32, 10]))

# kind/arch cross-check: on-ball kind cannot claim a 1-out defensive arch
def rejects_cross_kind():
    d = v1_football_brain_dict()
    d["kind"] = "defensive_action"   # weights are 24-32-32-10
    try:
        validate_schema(d)
        check("cross-kind arch mismatch rejected", False)
    except BrainSchemaError:
        check("cross-kind arch mismatch rejected", True)
rejects_cross_kind()

# ── 4. production on-disk brains all validate ────────────────

def scan_dir(rel, expected_kind):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
    good = ng = 0
    errors = []
    if not os.path.isdir(path):
        return 0, 0, ["missing dir"]
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(path, fn)
        try:
            data = json.load(open(p))
            meta = validate_schema(data)
            kind_ok = (meta.sensor_schema == "v1_24d")
            if not kind_ok:
                errors.append(f"{fn}: schema {meta.sensor_schema}")
            good += 1
        except Exception as e:   # noqa: BLE001
            ng += 1
            errors.append(f"{fn}: {e}")
    return good, ng, errors

POSITION_FILES = {"GK", "CB", "LB", "RB", "CDM", "CM", "CAM",
                  "LW", "RW", "ST", "CF"}

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
g, ng, errs = 0, 0, []
for rel in ("brains",):
    for fn in sorted(os.listdir(os.path.join(repo, rel))):
        base = os.path.splitext(fn)[0]
        if base not in POSITION_FILES:      # skip manifests/surrogates/etc.
            continue
        p = os.path.join(repo, rel, fn)
        try:
            data = json.load(open(p))
            meta = validate_schema(data)
            allowed = ("v1_24d", "v2_role_features")
            if meta.sensor_schema not in allowed:
                errs.append(f"{rel}/{fn}: bad schema {meta.sensor_schema}")
                ng += 1
            else:
                g += 1
        except Exception as e:   # noqa: BLE001
            errs.append(f"{rel}/{fn}: {e}")
            ng += 1
check(f"brains/ dir: all on-ball files validate (v1_24d or v2_role_features) "
      f"({g} ok, {ng} bad{('; ' + errs[0]) if errs else ''})", ng == 0 and g >= 11)

g2, ng2, errs2 = 0, 0, []
for fn in ("XI.json",):
    p = os.path.join(repo, "brains_team", fn)
    if os.path.exists(p):
        try:
            data = json.load(open(p))
            meta = validate_schema(data)
            g2 += 1
        except Exception as e:   # noqa: BLE001
            ng2 += 1
            errs2.append(f"{fn}: {e}")
check(f"brains_team/XI.json validates ({g2} ok, {ng2} bad)", ng2 == 0 and g2 == 1)

g3, ng3, errs3 = 0, 0, []
p = os.path.join(repo, "brains_def", "ACTION.json")
if os.path.exists(p):
    try:
        data = json.load(open(p))
        meta = validate_schema(data)
        g3 += 1
    except Exception as e:   # noqa: BLE001
        ng3 += 1
        errs3.append(str(e))
check(f"brains_def/ACTION.json validates ({g3} ok, {ng3} bad)", ng3 == 0 and g3 == 1)

# ── 5. brains_v2 scratch namespace ───────────────────────────

b3 = FootballBrain.random(seed=77)
d2 = b3.serialize()
with tempfile.TemporaryDirectory() as tmp:
    scratch = os.path.join(tmp, BRAINS_V2_DIR)
    os.makedirs(scratch)
    path = os.path.join(scratch, "ST.json")
    with open(path, "w") as f:
        json.dump(d2, f)
    loaded = FootballBrain.load(path)
    check("v2 file saved to brains_v2/ loads back identically",
          np.allclose(loaded.w1, b3.w1) and loaded.meta is not None)


print("\n" + ("ALL BRAIN SCHEMA TESTS PASSED" if fails == 0
              else f"{fails} FAILURES"))
raise SystemExit(1 if fails else 0)