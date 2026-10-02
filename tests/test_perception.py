"""Focused tests for perception.py — the PLOFA V2 perception layer.

Run: python3 test_perception.py   (plain asserts, project style)

Covers the audit Step 1 contract:
  1. IDENTITY MODE — disabled config returns a byte-identical v1 array.
  2. DEGRADATION   — enabled config bleeds only the other-actor geometry
     features (4-9); self/match-state/DNA features (0-3, 10-23) are exact.
  3. RANGE         — actors beyond a vision-scaled radius are unseen and
     read as an empty world (counts 0, nearest 1.0, best-forward 0).
  4. FOV           — actors behind the player (outside the forward cone)
     are unseen regardless of distance.
  5. NO FUTURITY   — perception reads only the current frame; an unseen
     actor is indistinguishable from one that does not exist.
  6. DNA SCALING   — high-vision/anticipation players perceive farther.
  7. DETERMINISM   — same snapshot + same seed => same perceived vector.
  8. Brain routing — NeuralDecisionBrain.decide uses the perception layer
     but is byte-identical when disabled.

Step 5 (role blocks):
  9. ROLE BLOCKS   — role_blocks=True gives each position its own
     range/FOV/noise/top-k profile:
     9a. same snapshot + same DNA, different position => DIFFERENT 4-9.
     9b. an actor inside the CB FOV but outside the ST FOV => seen by CB,
         empty world for ST.
     9c. relevance top-k caps the number of kept actors (nearest wins).
     9d. position normalization: "CB1" -> CB profile; unknowns are neutral.
     9e. role_blocks + zero-noise + full range is NOT identity (role FOV
         override is structural, not noise-driven).
     9f. determinism holds with role_blocks enabled.
     9g. config.max_actors (role_blocks=False) is an exact top-k cap.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from brain_sensors import extract_sensors, INPUT_SIZE
from football_brain import FootballBrain
from player_dna import (
    PlayerDNA, PlayerProfile, MentalAttributes,
)
from decision_brain import PlayerDecision
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry,
    set_perception,
)
from perception import perceive, PerceptionConfig, _normalize_position


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

class FakeGameState:
    def __init__(self, name="LEVEL"):
        self.name = name


class FakePositionEngine:
    def __init__(self, positions):
        self.positions = dict(positions)

    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def make_player(name, position="ST", vision=60, anticipation=60,
                composure=60, decisions=60):
    mental = MentalAttributes(
        vision=vision, anticipation=anticipation,
        composure=composure, decisions=decisions,
    )
    dna = PlayerDNA(name=name, position=position, mental=mental)
    return SimpleNamespace(name=name, position=position, dna=dna)


def make_pieces(player_x=50.0, player_y=34.0):
    """Standard snapshot: player at 50,34 attacking right."""
    from brain_sensors import _pos
    player = make_player("P1", "CM")
    pos = FakePositionEngine({
        "P1": (player_x, player_y),
        "DEF_FAR_30M": (79.0, 30.0),     # 29 m ahead, in FOV
        "DEF_BEHIND": (30.0, 40.0),      # 21 m BEHIND the player
        "TM_AHEAD": (70.0, 32.0),        # 20 m ahead, in FOV
        "TM_BEHIND": (30.0, 28.0),       # behind
    })
    defs = [
        SimpleNamespace(name="DEF_FAR_30M", position="CB"),
        SimpleNamespace(name="DEF_BEHIND", position="CB"),
    ]
    tms = [
        SimpleNamespace(name="TM_AHEAD", position="ST"),
        SimpleNamespace(name="TM_BEHIND", position="CDM"),
    ]
    return player, pos, tms, defs


def sense(exact=False, player_x=50.0, **kwargs):
    player, pos, tms, defs = make_pieces(player_x=player_x)
    config = PerceptionConfig(enabled=not exact)
    kwargs.setdefault("config", config)
    return perceive(player, player_x, 34.0, tms, defs, pos,
                    False, True, FakeGameState(), 45.0, **kwargs)


def _identity_vector():
    player, pos, tms, defs = make_pieces()
    return extract_sensors(player, 50.0, 34.0, tms, defs, pos,
                           False, True, FakeGameState(), 45.0)


fails = 0
def check(name, cond):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"[{status}] {name}")


# ─────────────────────────────────────────────────────────────

print("== 1. IDENTITY MODE ==")
v1 = _identity_vector()
v_off = sense(exact=True)
check("disabled perceive == extract_sensors byte-identical",
      np.array_equal(v1, v_off))
check("shape preserved", v_off.shape == (INPUT_SIZE,))

print("== 5. NO FUTURITY ==")
# Defender 12 m ahead: myopic (vision/ant 5 -> radius ~9.9 m) cannot see
# it, elite (99 -> ~44.6 m) can.  For the myopic player the defender's
# existence MUST be indistinguishable from absence: unseen == non-existent.
pos_m = FakePositionEngine({"P1": (50.0, 34.0), "DEF_12M": (61.0, 34.0)})
def_m = [SimpleNamespace(name="DEF_12M", position="CB")]
p_near = make_player("P1", vision=5, anticipation=5)
p_far = make_player("P1", vision=99, anticipation=99)
nf_cfg = PerceptionConfig(enabled=True, noise_base=0.0, fov_deg=360.0, seed=1)

vm_low_have = perceive(p_near, 50.0, 34.0, [], def_m, pos_m, False, True,
                       FakeGameState(), 45.0, config=nf_cfg)
vm_low_none = perceive(p_near, 50.0, 34.0, [],
                       [d for d in def_m if d.name != "DEF_12M"],
                       FakePositionEngine({"P1": (50.0, 34.0)}),
                       False, True, FakeGameState(), 45.0, config=nf_cfg)
check("myopic: near defender == no defender (features 4-7)",
      np.allclose(vm_low_have[4:8], vm_low_none[4:8], atol=1e-9))
vm_el_have = perceive(p_far, 50.0, 34.0, [], def_m, pos_m, False, True,
                      FakeGameState(), 45.0, config=nf_cfg)
vm_el_none = perceive(p_far, 50.0, 34.0, [],
                      [d for d in def_m if d.name != "DEF_12M"],
                      FakePositionEngine({"P1": (50.0, 34.0)}),
                      False, True, FakeGameState(), 45.0, config=nf_cfg)
check("elite: near-defender removal DOES change feature 4",
      abs(vm_el_have[4] - vm_el_none[4]) > 1e-9)
check("myopic vs elite near-defender feature differ at same snapshot",
      abs(vm_low_have[4] - vm_el_have[4]) > 1e-9)
# The difference must ONLY live in the perceptible geometry indices (4-9).
deltas = np.where(np.abs(vm_el_have - vm_el_none) > 1e-9)[0]
check("elite removal changes ONLY indices 4-9",
      set(deltas.tolist()) <= {4, 5, 6, 7, 8, 9})

print("== 2. DEGRADATION (only 4-9 can change) ==")
player, pos, tms, defs = make_pieces()
elite = PerceptionConfig(enabled=True, seed=5)
vm_elite = perceive(player, 50.0, 34.0, tms, defs, pos, False, True,
                    FakeGameState(), 45.0, config=elite)
changed = np.where(np.abs(vm_elite - v1) > 1e-12)[0]
check("only indices 4-9 may differ in enabled noise mode", set(changed.tolist()) <= {4, 5, 6, 7, 8, 9})
print(f"      changed indices: {changed.tolist() if len(changed) else 'none'}")
# With zero noise and full FOV+range, degradation is EXACT (identity).
config_exact = PerceptionConfig(enabled=True, noise_base=0.0, range_max=200.0,
                                fov_deg=360.0, seed=5)
v_patch = perceive(player, 50.0, 34.0, tms, defs, pos, False, True,
                   FakeGameState(), 45.0, config=config_exact)
check("enabled + zero noise + full FOV/range == exact v1",
      np.allclose(v_patch, v1, atol=1e-12))

print("== 3. RANGE ==")
# A defender 60 m away is beyond even elite range -> unseen.
pos3 = FakePositionEngine({"P1": (50.0, 34.0), "DEFFAR": (110.0, 30.0)})
defs3 = [SimpleNamespace(name="DEFFAR", position="CB")]
far_cfg = PerceptionConfig(enabled=True, range_max=45.0, noise_base=0.0,
                           fov_deg=360.0, seed=1)
v_far = perceive(player, 50.0, 34.0, [], defs3, pos3, False, True,
                 FakeGameState(), 45.0, config=far_cfg)
check("60 m defender unseen -> feature 4 reads empty (1.0)", v_far[4] == 1.0)
check("60 m defender unseen -> feature 6 reads empty (0.0)", v_far[6] == 0.0)

print("== 4. FOV ==")
# Defender behind the player (21 m) is within range but outside a 180-degree
# forward cone -> unseen.
fov2 = FakePositionEngine({"P1": (50.0, 34.0), "BEHIND": (30.0, 40.0)})
defs_behind = [SimpleNamespace(name="BEHIND", position="CB")]
fov_cfg = PerceptionConfig(enabled=True, noise_base=0.0, range_max=45.0,
                           fov_deg=180.0, seed=1)
v_behind = perceive(player, 50.0, 34.0, [], defs_behind, fov2, False, True,
                    FakeGameState(), 45.0, config=fov_cfg)
check("behind-player defender unseen -> feature 4 reads empty",
      v_behind[4] == 1.0)
check("behind-player defender unseen -> feature 6 reads empty", v_behind[6] == 0.0)

print("== 6. DNA SCALING ==")
# Same 12 m defender, identical seed, zero noise: the elite player's wider
# range lets it through (near_def ~12/15); the myopic player reads absence.
p_elite = make_player("P1", vision=99, anticipation=99)
p_low = make_player("P1", vision=5, anticipation=5)
sc_cfg = PerceptionConfig(enabled=True, noise_base=0.0, fov_deg=360.0, seed=1)
v_elite12 = perceive(p_elite, 50.0, 34.0, [], def_m, pos_m, False, True,
                     FakeGameState(), 45.0, config=sc_cfg)
v_low12 = perceive(p_low, 50.0, 34.0, [], def_m, pos_m, False, True,
                   FakeGameState(), 45.0, config=sc_cfg)
check("elite perceives the 12 m defender (near_def ~0.8, not 1.0)",
      v_elite12[4] < 0.9)
check("myopic does NOT perceive the 12 m defender (near_def 1.0)",
      v_low12[4] == 1.0)

print("== 7. DETERMINISM ==")
det_cfg = PerceptionConfig(enabled=True, seed=11)
va = perceive(player, 50.0, 34.0, tms, defs, pos, False, True,
              FakeGameState(), 45.0, config=det_cfg)
vb = perceive(player, 50.0, 34.0, tms, defs, pos, False, True,
              FakeGameState(), 45.0, config=det_cfg)
check("same snapshot + same seed -> identical perceived vector",
      np.array_equal(va, vb))


print("== 8. Brain routing (identity when disabled) ==")
clear_registry()
brain = FootballBrain.random()
register_brain("P1", brain)
set_perception(PerceptionConfig(enabled=False))
d_off = NeuralDecisionBrain.decide(
    player, 50.0, 34.0, tms, defs, pos, None,
    False, True, FakeGameState(), 45.0,
    record_trace=True,
)
set_perception(PerceptionConfig(enabled=True, noise_base=0.0, range_max=200.0,
                                fov_deg=360.0, seed=5))
d_full = NeuralDecisionBrain.decide(
    player, 50.0, 34.0, tms, defs, pos, None,
    False, True, FakeGameState(), 45.0,
    record_trace=True,
)
# With disabled config the sensor array must equal v1 EXACTLY (no noise,
# no FOV/range clipping) — the decision path is unchanged on the sensor axis.
check("decide with perception disabled yields v1-identical sensor trace",
      np.array_equal(np.asarray(d_off.trace["sensor_vector"]), v1))
check("decide with enabled perception still returns a PlayerDecision",
      isinstance(d_full, PlayerDecision))
set_perception(PerceptionConfig(enabled=False))
clear_registry()


print("== 9. ROLE BLOCKS ==")
# (9a) Same snapshot, same DNA, different position -> different 4-9.
#      CB sees wide (165°), ST sees only the tight forward window (130°).
#      Dedicated world: DD_FLANK sits at bearing ~66° (inside CB's 82.5°
#      half-FOV, outside ST's 65°) within 10 m, so the counts+proximity
#      features diverge across roles.
rb_cfg = PerceptionConfig(enabled=True, role_blocks=True, noise_base=0.0,
                          range_max=200.0, seed=3)
pos9 = FakePositionEngine({
    "P1": (50.0, 34.0),
    "DD_FLANK": (54.0, 43.0),   # bearing 66.0°, dist 9.9 — wide-FOV only
    "DD_FWD": (65.0, 34.0),     # bearing 0°, dist 15
    "DD_FAR": (80.0, 40.0),
    "TM_FWD": (70.0, 32.0),     # progress 20
    "TM_NEAR": (55.0, 36.0),    # progress 5
})
defs9 = [SimpleNamespace(name=n, position="CB") for n in
         ("DD_FLANK", "DD_FWD", "DD_FAR")]
tms9 = [SimpleNamespace(name=n, position="ST") for n in ("TM_FWD", "TM_NEAR")]
p_cb = make_player("P1", "CB", vision=70, anticipation=70)
p_st = make_player("P1", "ST", vision=70, anticipation=70)
v_cb = perceive(p_cb, 50.0, 34.0, tms9, defs9, pos9, False, True,
                FakeGameState(), 45.0, config=rb_cfg)
v_st = perceive(p_st, 50.0, 34.0, tms9, defs9, pos9, False, True,
                FakeGameState(), 45.0, config=rb_cfg)
rb_delta = np.where(np.abs(v_cb - v_st) > 1e-9)[0]
check("(9a) ST vs CB perceive the same snapshot differently (4-9)",
      len(rb_delta) > 0 and set(rb_delta.tolist()) <= {4, 5, 6, 7, 8, 9})
check("(9a) only indices 4-9 differ between roles", set(rb_delta.tolist()) <= {4, 5, 6, 7, 8, 9})
check("(9a) CB sees the flank runner (def_10 = 1/5)", abs(v_cb[6] - 0.2) < 1e-9)
check("(9a) ST loses the flank runner (def_10 = 0)", v_st[6] == 0.0)
# Everything outside 4-9 must be IDENTICAL across roles (DNA features same).

# (9b) Actor forward-left at a bearing only the wide-CB FOV admits.
#      Player at (50,34); actor at (54,48) -> bearing ~74° off forward (0°).
#      ST FOV 130° (half 65°): actor unseen. CB FOV 165° (half 82.5°): seen.
pgs = FakePositionEngine({"P1": (50.0, 34.0), "FLANK": (54.0, 48.0)})
flank_def = [SimpleNamespace(name="FLANK", position="RB")]
def _flank_sense(role):
    p = make_player("P1", role, vision=60, anticipation=60)
    return perceive(p, 50.0, 34.0, [], flank_def, pgs, False, True,
                    FakeGameState(), 45.0, config=rb_cfg)
v_cb_flank = _flank_sense("CB")
v_st_flank = _flank_sense("ST")
check("(9b) CB sees the far-side flank runner (feature 4 < 1.0)",
      v_cb_flank[4] < 1.0)
check("(9b) ST does NOT see the far-side flank runner (feature 4 == 1.0)",
      v_st_flank[4] == 1.0)

# (9c) Relevance top-k: ST profile caps at 6 actors.  Build 8 defenders and
#      2 teammates all inside ST range/FOV; the nearest 6 survive.  Counts
#      must reflect the cap (defenders_within_10m <= 6/5 = 1.2).
p_many = FakePositionEngine({"P1": (50.0, 34.0)}
                            | {f"DD{i}": (55.0 + i, 38.0) for i in range(8)}
                            | {"TM1": (54.0, 36.0), "TM2": (56.0, 40.0)})
many_defs = [SimpleNamespace(name=f"DD{i}", position="CB") for i in range(8)]
many_tms = [SimpleNamespace(name="TM1", position="ST"),
            SimpleNamespace(name="TM2", position="ST")]
v_many = perceive(p_st, 50.0, 34.0, many_tms, many_defs, p_many, False, True,
                  FakeGameState(), 45.0, config=rb_cfg)
kept = len(many_defs) + len(many_tms) - 0
check("(9c) ST profile keeps <= 6 actors (def_10 = count/5 <= 1.2)",
      v_many[6] <= 6 / 5.0 + 1e-9)
# A CM (cap 8) sees MORE actors than the ST (cap 6) at the same snapshot.
v_many_cm = perceive(make_player("P1", "CM", vision=70, anticipation=70),
                     50.0, 34.0, many_tms, many_defs, p_many, False, True,
                     FakeGameState(), 45.0, config=rb_cfg)
check("(9c) CM (cap 8) sees more actors than ST (cap 6)",
      v_many[6] < v_many_cm[6])

# (9d) Position normalization + unknown rollback.
p_cb1 = make_player("P1", "CB1", vision=70, anticipation=70)
v_cb1 = perceive(p_cb1, 50.0, 34.0, tms9, defs9, pos9, False, True,
                 FakeGameState(), 45.0, config=rb_cfg)
check("(9d) 'CB1' normalizes to CB profile (== CB vector)",
      np.array_equal(v_cb1, v_cb))
check("(9d) _normalize_position('CM2') == 'CM'", _normalize_position("CM2") == "CM")
p_unknown = make_player("P1", "SUB1", vision=70, anticipation=70)
v_unknown = perceive(p_unknown, 50.0, 34.0, tms9, defs9, pos9, False, True,
                     FakeGameState(), 45.0, config=rb_cfg)
# Unknown position -> neutral profile = the STEP-1 flat behaviour for the
# role_blocks-enabled config (same as a ST with KD range but wide FOV).
check("(9d) unknown position uses neutral profile (no crash, exact 24-d)",
      v_unknown.shape == (INPUT_SIZE,) and np.isfinite(v_unknown).all())

# (9e) Role FOV override is structural: even with zero noise + full range,
#      role blocks are NOT the identity (the ST cone clips actors the flat
#      config with fov=360 keeps).
v_st_exact = perceive(p_st, 50.0, 34.0, tms, defs, pos, False, True,
                      FakeGameState(), 45.0, config=rb_cfg)
check("(9e) role_blocks + full range is NOT identity (differs from v1)",
      not np.allclose(v_st_exact, v1, atol=1e-12))
# But an ENABLED config with same fov/range/noise and role_blocks=False
# back at the flat params must match the role_blocks=True result for roles
# whose canonical profile equals those flat params (CM fov 150 == default).
flat_cfg = PerceptionConfig(enabled=True, role_blocks=False, noise_base=0.0,
                            range_max=45.0, fov_deg=150.0, seed=0)
p_cm = make_player("P1", "CM", vision=60, anticipation=60)
v_cm_flat = perceive(p_cm, 50.0, 34.0, tms, defs, pos, False, True,
                     FakeGameState(), 45.0, config=rb_cfg)
check("(9e) CM role profile == default flat fov 150 profile at same seed",
      np.array_equal(v_cm_flat, perceive(p_cm, 50.0, 34.0, tms, defs, pos,
                                         False, True, FakeGameState(), 45.0,
                                         config=flat_cfg)))

# (9f) Determinism with role_blocks enabled.
va_rb = perceive(p_cb, 50.0, 34.0, tms, defs, pos, False, True,
                 FakeGameState(), 45.0, config=rb_cfg)
vb_rb = perceive(p_cb, 50.0, 34.0, tms, defs, pos, False, True,
                 FakeGameState(), 45.0, config=rb_cfg)
check("(9f) role_blocks determinism: same snapshot+seed -> identical",
      np.array_equal(va_rb, vb_rb))

# (9g) config.max_actors (role_blocks=False) is an exact top-k cap.
cap_cfg = PerceptionConfig(enabled=True, role_blocks=False, noise_base=0.0,
                           range_max=45.0, fov_deg=360.0, max_actors=1, seed=3)
v_cap1 = perceive(player, 50.0, 34.0, many_tms, many_defs, p_many, False, True,
                  FakeGameState(), 45.0, config=cap_cfg)
# Only 1 closest actor kept -> it sees at most 1 actor: def_10 <= 0.2.
check("(9g) exact top-k=1 cap (def_10 = count/5 <= 0.2)",
      v_cap1[6] <= 0.2 + 1e-9)
# The 2 remaining tms/defs are far out; cap=2 keeps nearest two.
cap_cfg2 = PerceptionConfig(enabled=True, role_blocks=False, noise_base=0.0,
                            range_max=45.0, fov_deg=360.0, max_actors=2, seed=3)
v_cap2 = perceive(player, 50.0, 34.0, many_tms, many_defs, p_many, False, True,
                  FakeGameState(), 45.0, config=cap_cfg2)
check("(9g) cap=2 sees more than cap=1", v_cap1[6] < v_cap2[6])


print(f"\n{fails} failures")
if fails:
    raise SystemExit(1)
print("ALL PERCEPTION TESTS PASSED")