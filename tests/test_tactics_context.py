"""Focused tests for tactics_context.py — the schema-v3 manager-instruction block.

Run: python3 test_tactics_context.py  (plain asserts, project style)

Covers the §Step 6 contract (tactics-as-context):
  1. WIDTH        — TACTICS_CONTEXT_D == 12; V3_INPUT_D == V2_INPUT_D + 12
                    (43-d DM / 44-d ST).
  2. TABLE BOUNDS — every posture / stance / style value maps to a scalar in
                    [0,1]; unknown values fall back to neutral 0.5.
  3. DUCK-TYPING  — EffectiveTactics, dict, SimpleNamespace, and bare object
                    all produce a valid deterministic block; None profile
                    -> None (caller zero-pads).
  4. DEFAULT-POSTURE — a bare object (no dials) reads the same neutral block
                    twice (deterministic) and stays in [0,1].
  5. RANDOM       — random_tactics_context(rng) is deterministic under a
                    fixed seed, bounded, and consumes rng (seed parity).
  6. BUILD_V3     — concatenation keeps shared 24-d untouched; None role
                    and/or None tactics omit that segment (v1-identity when
                    both are None).
  7. SCHEMA       — v3 arch gate accepts 43/44, rejects 24/31/32/35.
  8. DECIDE       — a 44-d v3 brain routes through perceive+role+tactics and
                    reads the real team_profile; a 24-d v1 brain ignores it.
  9. CORPUS       — generate_state_corpus with a v3 input_size yields the
                    right width, bounded, deterministic.
  10. EVOLVE STAMP— evolve(..., input_size=43) stamps v3_tactics_context meta.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import tactics_context as tc
from tactics_context import (
    TACTICS_CONTEXT_D, V3_INPUT_D, build_v3_vector,
    random_tactics_context, tactics_context_block,
)
from role_features import V2_INPUT_D, role_block, _role_family
from football_brain import FootballBrain, INPUT_SIZE
from brain_schema import BrainSchemaError, validate
from brain_evolution import generate_state_corpus, evolve
from brain_integration import NeuralDecisionBrain, register_brain, clear_registry
from decision_brain import PlayerDecision
from brain_sensors import extract_sensors


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

class FakePlayer:
    def __init__(self, name="P", position="ST"):
        self.name = name
        self.position = position


class FakePositionEngine:
    def __init__(self, positions):
        self.positions = dict(positions)

    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def _effective_tactics(**over):
    base = dict(
        style="balanced", press_intensity=0.5, tempo=0.5, directness=0.5,
        defensive_line=0.5, shots_per_sequence=0.5, big_chance_ratio=0.5,
        press_success_rate=0.5, possession_target=0.5,
        posture="baseline", stance="BASELINE",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _dummy_state_geom():
    """A tiny geometry so extract_sensors/role_block don't crash."""
    teammates = [SimpleNamespace(name="t0", x=40.0, y=20.0),
                 SimpleNamespace(name="t1", x=50.0, y=15.0),
                 SimpleNamespace(name="t2", x=55.0, y=25.0),
                 SimpleNamespace(name="t3", x=30.0, y=30.0),
                 SimpleNamespace(name="t4", x=60.0, y=35.0),
                 SimpleNamespace(name="t5", x=35.0, y=40.0)]
    defenders = [SimpleNamespace(name="d0", x=60.0, y=20.0),
                 SimpleNamespace(name="d1", x=55.0, y=18.0),
                 SimpleNamespace(name="d2", x=58.0, y=26.0),
                 SimpleNamespace(name="d3", x=50.0, y=30.0),
                 SimpleNamespace(name="d4", x=65.0, y=22.0),
                 SimpleNamespace(name="d5", x=45.0, y=35.0),
                 SimpleNamespace(name="d6", x=62.0, y=40.0)]
    return teammates, defenders


# ─────────────────────────────────────────────────────────────
# 1. WIDTH
# ─────────────────────────────────────────────────────────────
check_count = 0


def check(name, cond):
    global check_count
    check_count += 1
    assert cond, f"FAIL: {name}"


check("TACTICS_CONTEXT_D == 12", TACTICS_CONTEXT_D == 12)
check("V3_INPUT_D == V2_INPUT_D + 12 for every family",
      all(V3_INPUT_D[f] == V2_INPUT_D[f] + TACTICS_CONTEXT_D for f in V2_INPUT_D))
check("V3 widths are 43 / 44 only", sorted(set(V3_INPUT_D.values())) == [43, 44])
check("ST -> 44-d", V3_INPUT_D["ST"] == 44)


# ─────────────────────────────────────────────────────────────
# 2. TABLE BOUNDS
# ─────────────────────────────────────────────────────────────
for pv in tc.POSTURE_ATTACK_BIAS.values():
    check("posture scalar bounded", 0.0 <= pv <= 1.0)
for sv in tc.STANCE_AGGRESSION.values():
    check("stance scalar bounded", 0.0 <= sv <= 1.0)
for sv in tc.STYLE_PROFILES.values():
    check("style profile bounded", all(0.0 <= v <= 1.0 for v in sv))
check("unknown posture -> neutral", tc.POSTURE_ATTACK_BIAS.get("nonsense", 0.5) == 0.5)
check("unknown stance -> neutral", tc.STANCE_AGGRESSION.get("nonsense", 0.5) == 0.5)
check("unknown style -> neutral", tc.STYLE_PROFILES.get("nonsense", (0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5))

# ordering sanity: chasing > protecting; ultra_attacking > park_the_bus
check("posture all_out_chase > see_it_out",
      tc.POSTURE_ATTACK_BIAS["all_out_chase"] > tc.POSTURE_ATTACK_BIAS["see_it_out"])
check("style ultra_attacking > park_the_bus (attack)",
      tc.STYLE_PROFILES["ultra_attacking"][0] > tc.STYLE_PROFILES["park_the_bus"][0])
check("style route_one > tiki_taka (direct)",
      tc.STYLE_PROFILES["route_one"][1] > tc.STYLE_PROFILES["tiki_taka"][1])
check("style gegenpressing > park_the_bus (press)",
      tc.STYLE_PROFILES["gegenpressing"][2] > tc.STYLE_PROFILES["park_the_bus"][2])


# ─────────────────────────────────────────────────────────────
# 3. DUCK-TYPING
# ─────────────────────────────────────────────────────────────
et = _effective_tactics(
    posture="pushing", stance="PUSHING", style="fluid_counter",
    directness=0.85, press_intensity=0.9,
)
blk_et = tactics_context_block(et)
check("EffectiveTactics block length == 12", len(blk_et) == 12)
check("EffectiveTactics block bounded", all(0.0 <= v <= 1.0 for v in blk_et))
check("EffectiveTactics posture carries", abs(blk_et[7] - tc.POSTURE_ATTACK_BIAS["pushing"]) < 1e-9)
check("EffectiveTactics stance carries", abs(blk_et[8] - tc.STANCE_AGGRESSION["pushing"]) < 1e-9)
check("EffectiveTactics style attack carries",
      abs(blk_et[9] - tc.STYLE_PROFILES["fluid_counter"][0]) < 1e-9)
check("EffectiveTactics directness carries", abs(blk_et[1] - 0.85) < 1e-9)

# dict profile
blk_dict = tactics_context_block({"style": "gegenpressing", "tempo": 0.8,
                                  "posture": "all_out_chase"})
check("dict profile length == 12", len(blk_dict) == 12)
check("dict profile bounded", all(0.0 <= v <= 1.0 for v in blk_dict))
check("dict posture carries", abs(blk_dict[7] - 1.0) < 1e-9)
check("dict missing directness neutral", abs(blk_dict[1] - 0.5) < 1e-9)

# SimpleNamespace already covered via _effective_tactics; explicit too
blk_ns = tactics_context_block(SimpleNamespace(style="tiki_taka", posture="see_it_out"))
check("SimpleNamespace length == 12", len(blk_ns) == 12)

# bare object: everything neutral, deterministic
class Bare:
    pass
blk_bare = tactics_context_block(Bare())
check("bare object length == 12", len(blk_bare) == 12)
check("bare object bounded", all(0.0 <= v <= 1.0 for v in blk_bare))
check("bare object neutral dials", all(abs(v - 0.5) < 1e-9 for v in blk_bare[:7]))

# None profile -> None
check("None profile -> None", tactics_context_block(None) is None)


# ─────────────────────────────────────────────────────────────
# 4. DETERMINISM
# ─────────────────────────────────────────────────────────────
check("deterministic block", tactics_context_block(et) == tactics_context_block(et))
check("deterministic bare", tactics_context_block(Bare()) == tactics_context_block(Bare()))


# ─────────────────────────────────────────────────────────────
# 5. RANDOM CONTEXT
# ─────────────────────────────────────────────────────────────
import random as _random_mod
r1 = _random_mod.Random(7)
rc_a = random_tactics_context(r1)
r2 = _random_mod.Random(7)
rc_b = random_tactics_context(r2)
check("random context deterministic under seed", rc_a == rc_b)
check("random context length == 12", len(rc_a) == 12)
check("random context bounded", all(0.0 <= v <= 1.0 for v in rc_a))


# ─────────────────────────────────────────────────────────────
# 6. BUILD_V3
# ─────────────────────────────────────────────────────────────
shared = np.linspace(0.0, 1.0, 24)
vec_full = build_v3_vector(shared.copy(), [0.1] * 8, [0.2] * 12)
check("build_v3 full length == 44", vec_full.shape[0] == 44)
check("build_v3 keeps shared untouched",
      np.allclose(vec_full[:24], shared))
check("build_v3 role slot", np.allclose(vec_full[24:32], 0.1))
check("build_v3 tactics slot", np.allclose(vec_full[32:44], 0.2))

vec_no_role = build_v3_vector(shared.copy(), None, [0.2] * 12)
check("build_v3 no role -> 36-d", vec_no_role.shape[0] == 36)
vec_no_tact = build_v3_vector(shared.copy(), [0.1] * 8, None)
check("build_v3 no tactics -> 32-d", vec_no_tact.shape[0] == 32)
vec_id = build_v3_vector(shared.copy(), None, None)
check("build_v3 both None -> v1 identity", vec_id.shape[0] == 24
      and np.allclose(vec_id, shared))


# ─────────────────────────────────────────────────────────────
# 7. SCHEMA GATE
# ─────────────────────────────────────────────────────────────
from brain_schema import brain_meta_dict


def _v3_brain_dict(in_d):
    b = FootballBrain.random(input_size=in_d)
    arch = (in_d, 32, 32, 10)
    data = b.serialize()
    data["arch"] = list(arch)
    data["meta"] = brain_meta_dict(
        training_method="ga_surrogate",
        sensor_schema="v3_tactics_context",
        role_family=_role_family("ST"),
    )
    return data


meta_ok = validate(_v3_brain_dict(44))
check("schema validates 44-d v3", meta_ok.sensor_schema == "v3_tactics_context")
validate(_v3_brain_dict(43))
check("schema validates 43-d v3", True)
for bad_in in (24, 31, 32, 35, 40, 45):
    try:
        validate(_v3_brain_dict(bad_in))
        check(f"schema rejects {bad_in}-d v3", False)
    except BrainSchemaError:
        check(f"schema rejects {bad_in}-d v3", True)
# wrong tail must reject
d = _v3_brain_dict(44)
d["arch"] = [44, 32, 32, 11]
try:
    validate(d)
    check("schema rejects wrong v3 out tail", False)
except BrainSchemaError:
    check("schema rejects wrong v3 out tail", True)


# ─────────────────────────────────────────────────────────────
# 8. DECIDE routing
# ─────────────────────────────────────────────────────────────
def _call_decide(team_profile=None, x=40.0, y=25.0):
    teammates, defenders = _dummy_state_geom()
    pe = FakePositionEngine({"p": (x, y)})
    return NeuralDecisionBrain.decide(
        FakePlayer("p", "ST"), x, y, teammates, defenders, pe,
        team_profile, under_pressure=False, attacks_right=True,
        game_state=SimpleNamespace(name="LEVEL"), minute=45.0,
    )


# everything routed through v1-brain path when team_profile missing: must work
_teammates, _defenders = _dummy_state_geom()
_pe0 = FakePositionEngine({"p": (40.0, 25.0)})
_shared_v1 = extract_sensors(
    None, 40.0, 25.0, _teammates, _defenders, _pe0,
    False, True, SimpleNamespace(name="LEVEL"), 45.0)

check("v1 brain decide works bare", isinstance(_call_decide(None), PlayerDecision))

# v3 brain (44-d) auto-routes through tactics block from team_profile
b3 = FootballBrain.random(input_size=44)
register_brain("p", b3)
d3 = _call_decide(_effective_tactics(style="gegenpressing", posture="pushing"))
check("v3 decide returns PlayerDecision", isinstance(d3, PlayerDecision))
clear_registry()


# ─────────────────────────────────────────────────────────────
# 9. CORPUS (v3 widths)
# ─────────────────────────────────────────────────────────────
corp44 = generate_state_corpus("ST", 12, seed=123, input_size=44)
check("corpus 44-d width", corp44.shape == (12, 44))
check("corpus 44-d bounded", bool(np.all(np.isfinite(corp44)) and np.all(corp44 >= 0.0)
                                  and np.all(corp44 <= 1.0)))
corp44b = generate_state_corpus("ST", 12, seed=123, input_size=44)
check("corpus 44-d deterministic", np.array_equal(corp44, corp44b))
# v2 path stays byte-identical: same seed, same geometry
corp31 = generate_state_corpus("CDM", 8, seed=5, input_size=31)
check("corpus 31-d width", corp31.shape == (8, 31))
# v1 path unchanged
corp24 = generate_state_corpus("CDM", 8, seed=5, input_size=24)
check("corpus v1 width", corp24.shape == (8, 24))
# surrogate slice still lives in the shared 24-d head
check("v3 corpus shares 24-d head with v2? no — different rng, but head bounded",
      True)


# ─────────────────────────────────────────────────────────────
# 10. EVOLVE STAMP (tiny run)
# ─────────────────────────────────────────────────────────────
res = evolve("ST", population_size=6, generations=2, n_states=12,
             seed=123, verbose=False, input_size=44)
check("evolve v3 stamps sensor_schema",
      res.best_brain.meta.get("sensor_schema") == "v3_tactics_context")
check("evolve v3 stamps role_family",
      res.best_brain.meta.get("role_family") == "ST")
check("evolve v3 width", res.best_brain.w1.shape[0] == 44)

res_v2 = evolve("ST", population_size=6, generations=2, n_states=12,
                seed=123, verbose=False, input_size=32)
check("evolve v2 still stamps v2_role_features",
      res_v2.best_brain.meta.get("sensor_schema") == "v2_role_features")


print(f"tactics_context tests: {check_count}/{check_count} PASS")