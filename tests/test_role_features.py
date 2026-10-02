"""Focused tests for role_features.py — the schema-v2 role-specific blocks.

Run: python3 test_role_features.py  (plain asserts, project style)

Covers the §H.2.2 contract:
  1. FAMILY MAP   — every canonical position maps to its role family;
     trailing digits strip ("CB1" -> CB); unknown roles -> None.
  2. SHAPE        — block length == menu length; V2_INPUT_D == 24 + role_b.
  3. BOUNDS       — every feature in [0,1] for both attack directions.
  4. DETERMINISM  — same arguments => identical float list every call.
  5. GEOMETRY     — the features respond to the geometry they claim:
      5a. ST shoot_angle higher when closer to goal.
      5b. CB box_threat rises when opp forwards camp our goal.
      5c. GK shot_threat only real when ball is near our goal AND the
          opposing forward is on the ball.
      5d. CM forward_space is room before the nearest defender ahead.
      5e. WING cross_target_open falls when a defender covers the spot.
      5f. GK defensive_shape tracks our backline dropping deep.
      5g. CB space_behind grows as our line pushes up (grass IS there).
      5h. ST cb_room shrinks as a CB closes in.
      5i. CF shares the ST menu.
  6. BUILD_V2     — concatenation keeps the shared 24-d block untouched;
     a None role block leaves the v1 vector identity.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import role_features as rf
from role_features import (
    MENU_NAMES, ROLE_FAMILY, V2_INPUT_D, V2_ROLE_D, V2_SHARED_D,
    _role_family, build_v2_vector, role_block,
)
from football_brain import FootballBrain
from brain_schema import brain_meta_dict
from brain_evolution import generate_state_corpus
from brain_integration import NeuralDecisionBrain, register_brain, clear_registry
from decision_brain import PlayerDecision


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

class FakePositionEngine:
    def __init__(self, positions):
        self.positions = dict(positions)

    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def P(name, position, x, y):
    return SimpleNamespace(name=name, position=position, x=x, y=y)


def world():
    """A 23-player world (attacking-right home XI + an 11-defender away XI).

    Returns (home, away, engine).  Home includes a GK so _teammates() has
    a keeper to skip; every family's canonical position is represented.
    """
    home = [
        P("GK_h", "GK", 5.0, 34.0),
        P("CB1", "CB", 30.0, 20.0),
        P("CB2", "CB", 30.0, 48.0),
        P("LB", "LB", 22.0, 6.0),
        P("RB", "RB", 22.0, 62.0),
        P("CDM", "CDM", 48.0, 30.0),
        P("CM1", "CM", 52.0, 36.0),
        P("CM2", "CM", 55.0, 30.0),
        P("CAM", "CAM", 66.0, 34.0),
        P("LW", "LW", 62.0, 10.0),
        P("ST", "ST", 72.0, 34.0),
        P("RW", "RW", 62.0, 58.0),
        P("CF", "CF", 76.0, 40.0),
    ]
    away = [
        P("AGK", "GK", 100.0, 34.0),
        P("ACB1", "CB", 80.0, 22.0),
        P("ACB2", "CB", 80.0, 46.0),
        P("ALB", "LB", 82.0, 10.0),
        P("ARB", "RB", 82.0, 58.0),
        P("ADM", "CDM", 72.0, 34.0),
        P("ACM1", "CM", 68.0, 28.0),
        P("ACM2", "CM", 68.0, 40.0),
        P("ACAM", "CAM", 60.0, 34.0),
        P("AW", "LW", 58.0, 8.0),
        P("AST", "ST", 58.0, 34.0),
    ]
    pos = {p.name: (p.x, p.y) for p in home + away}
    return home, away, FakePositionEngine(pos)


def block_for(position, x=50.0, y=34.0, attackers_right=True,
              tighten=None):
    """Compute role_block for a picked position with optional perturbations.

    `tighten`: optional {actor_name: (new_x, new_y)} moved BEFORE the
    position engine is rebuilt, so perturbations truly change lookups.
    """
    home, away, _ = world()
    own = [p for p in home if p.position == position]
    if not own:
        return None
    p = own[0]
    if tighten:
        by_name = {q.name: q for q in home + away}
        for name, (nx, ny) in tighten.items():
            if name in by_name:
                by_name[name].x, by_name[name].y = nx, ny
    pe = FakePositionEngine({q.name: (q.x, q.y) for q in home + away})
    return role_block(p, x, y, home, away, pe, attackers_right)


fails = 0


def check(name, cond):
    global fails
    if not cond:
        fails += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# ─────────────────────────────────────────────────────────────
# 1. FAMILY MAP
# ─────────────────────────────────────────────────────────────
print("== 1. FAMILY MAP ==")
expect = {"GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "CDM": "DM",
          "CM": "CM", "CAM": "AM", "LW": "WING", "RW": "WING",
          "ST": "ST", "CF": "ST"}
check("canonical position->family map complete",
      ROLE_FAMILY == expect)
check("digit stripping: 'CB1' -> CB", _role_family("CB1") == "CB")
check("unknown position -> None", _role_family("KEEPER") is None)
check("role_block(unknown) -> None",
      role_block(P("PX", "KEEPER", 0, 0), 0, 0, [], [], FakePositionEngine({}), True) is None)

# ─────────────────────────────────────────────────────────────
# 2. SHAPE
# ─────────────────────────────────────────────────────────────
print("== 2. SHAPE ==")
for fam, names in MENU_NAMES.items():
    blk = block_for({"GK": "GK", "FB": "LB", "ST": "ST",
                     "WING": "LW", "AM": "CAM", "CM": "CM",
                     "DM": "CDM", "CB": "CB"}[fam])
    check(f"{fam}: block length == menu length ({len(names)})",
          blk is not None and len(blk) == len(names) == V2_ROLE_D[fam])
    check(f"{fam}: V2_INPUT_D == 24 + role_D",
          V2_INPUT_D[fam] == V2_SHARED_D + V2_ROLE_D[fam])
check("all menu names unique per family",
      all(len(set(n)) == len(n) for n in MENU_NAMES.values()))

# ─────────────────────────────────────────────────────────────
# 3. BOUNDS (both attack directions)
# ─────────────────────────────────────────────────────────────
print("== 3. BOUNDS ==")
sampled_positions = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM",
                     "LW", "RW", "ST", "CF"]
ok = True
for ar in (True, False):
    for pos in sampled_positions:
        blk = block_for(pos, x=50.0, y=34.0, attackers_right=ar)
        if blk is None or not all(isinstance(v, float) and 0.0 <= v <= 1.0
                                  for v in blk):
            ok = False
            print(f"      OUT OF BOUNDS: {pos} ar={ar} {blk}")
check("all features in [0,1] for both directions", ok)

# ─────────────────────────────────────────────────────────────
# 4. DETERMINISM
# ─────────────────────────────────────────────────────────────
print("== 4. DETERMINISM ==")
home, away, pe = world()
st = next(p for p in home if p.position == "ST")
a = role_block(st, 50.0, 34.0, home, away, pe, True)
b = role_block(st, 50.0, 34.0, home, away, pe, True)
check("same args -> identical block", a == b)

# ─────────────────────────────────────────────────────────────
# 5. GEOMETRY
# ─────────────────────────────────────────────────────────────
print("== 5. GEOMETRY ==")
# 5a. ST shoot_angle rises toward goal (attacking right -> opp goal x=105).
near_g = block_for("ST", x=90.0, y=34.0)
far_g  = block_for("ST", x=55.0, y=34.0)
i_sa = rf.MENU_NAMES["ST"].index("shoot_angle")
check("ST shoot_angle higher at 15 m than 50 m", near_g[i_sa] > far_g[i_sa])

# 5b. CB box_threat rises when an opp forward camps our goal (own goal x=0).
i_bt = rf.MENU_NAMES["CB"].index("box_threat")
quiet = block_for("CB", x=30.0, y=34.0)
under = block_for("CB", x=30.0, y=34.0, tighten={"AST": (12.0, 34.0)})
check("CB box_threat higher with an opp ST at our six-yard line",
      under[i_bt] > quiet[i_bt])

# 5c. GK shot_threat needs BOTH proximity and a forward on the ball.
i_st_sh = rf.MENU_NAMES["GK"].index("shot_threat")
dangerous = block_for("GK", x=8.0, y=34.0,
                       tighten={"AST": (12.0, 34.0), "ADM": (50.0, 50.0)})
ball_far  = block_for("GK", x=60.0, y=34.0,
                       tighten={"AST": (12.0, 34.0), "ADM": (50.0, 50.0)})
fwd_away  = block_for("GK", x=8.0, y=34.0,
                       tighten={"AST": (70.0, 34.0), "ADM": (50.0, 50.0)})
check("GK shot_threat real only near own goal with fwd on ball",
      dangerous[i_st_sh] > 0.2 and ball_far[i_st_sh] == 0.0 and fwd_away[i_st_sh] == 0.0)

# 5d. CM forward_space = room before nearest defender ahead.
i_fs = rf.MENU_NAMES["CM"].index("forward_space")
open_cm  = block_for("CM", x=50.0, y=34.0)
cramped  = block_for("CM", x=50.0, y=34.0, tighten={"ADM": (57.0, 34.0)})
check("CM forward_space lower with a defender 7 m ahead",
      open_cm[i_fs] > cramped[i_fs])

# 5e. WING cross_target_open falls when a defender covers the spot.
# (attacking right, target_x = 105 - 11 = 94)
i_ct = rf.MENU_NAMES["WING"].index("cross_target_open")
open_w    = block_for("LW", x=88.0, y=8.0)
covered_w = block_for("LW", x=88.0, y=8.0, tighten={"ACB1": (94.0, 30.0)})
check("WING cross_target_open lower with a CB on the spot",
      open_w[i_ct] > covered_w[i_ct])

# 5f. GK defensive_shape — higher = defenders dropped deep (compact shape).
i_ds = rf.MENU_NAMES["GK"].index("defensive_shape")
high_line = block_for("GK", x=5.0, y=34.0,
                       tighten={"CB1": (44.0, 20.0), "CB2": (44.0, 48.0),
                                "CDM": (48.0, 30.0)})
deep_line = block_for("GK", x=5.0, y=34.0,
                       tighten={"CB1": (12.0, 20.0), "CB2": (12.0, 48.0),
                                "CDM": (14.0, 30.0)})
check("GK defensive_shape higher when backline sits deeper",
      deep_line[i_ds] > high_line[i_ds])

# 5g. CB space_behind — more grass when our line pushes forward.
i_sb = rf.MENU_NAMES["CB"].index("space_behind")
line_high = block_for("CB", x=40.0, y=34.0,
                       tighten={"CB2": (50.0, 48.0), "CDM": (54.0, 30.0)})
line_low  = block_for("CB", x=40.0, y=34.0,
                       tighten={"CB2": (10.0, 48.0), "CDM": (14.0, 30.0)})
check("CB space_behind larger when the line pushes up",
      line_high[i_sb] > line_low[i_sb])

# 5h. ST cb_room shrinks as the nearest CB closes in.
i_cr = rf.MENU_NAMES["ST"].index("cb_room")
clear_st  = block_for("ST", x=72.0, y=34.0,
                       tighten={"ADM": (50.0, 40.0), "ACB1": (64.0, 22.0)})
tight_st  = block_for("ST", x=72.0, y=34.0,
                       tighten={"ADM": (50.0, 40.0), "ACB1": (69.0, 33.0)})
check("ST cb_room lower with a CB within 3 m", clear_st[i_cr] > tight_st[i_cr])

# 5i. CF shares the ST menu.
cf = block_for("CF", x=50.0, y=34.0)
check("CF -> ST block (8 features)", len(cf) == len(MENU_NAMES["ST"]))

# ─────────────────────────────────────────────────────────────
# 6. BUILD_V2
# ─────────────────────────────────────────────────────────────
print("== 6. BUILD_V2 ==")
shared = np.linspace(0.0, 1.0, 24)
no_blk = build_v2_vector(shared, None)
check("None role block -> v1 identity", np.array_equal(no_blk, shared))
with_blk = build_v2_vector(shared, [0.1] * V2_ROLE_D["ST"])
check("with role block -> 24 + 8", with_blk.shape == (32,))
check("shared 24-d preserved verbatim", np.array_equal(with_blk[:24], shared))
check("role tail appended", np.all(with_blk[24:] == 0.1))

# ─────────────────────────────────────────────────────────────
# 7. WIRING (schema v2 round-trip, synthetic corpus, decide routing)
# ─────────────────────────────────────────────────────────────
print("== 7. WIRING ==")

# 7a. v1 brain is untouched — arch stays [24, 32, 32, 10].
b24 = FootballBrain.random(seed=1)
check("v1 brain arch stays [24,32,32,10]",
      b24.serialize()["arch"] == [24, 32, 32, 10])
check("v1 round-trip keeps 24-d input",
      FootballBrain.deserialize(b24.serialize()).w1.shape[0] == 24)

# 7b. v2 role-features brain round-trips through the schema validator.
b32 = FootballBrain.random(seed=1, input_size=32)
d32 = b32.serialize()
check("v2 brain arch input is 32 (tail 32/32/10 exact)",
      d32["arch"][0] == 32 and d32["arch"][1:] == [32, 32, 10])
d32["meta"] = brain_meta_dict(training_method="ga_surrogate",
                              sensor_schema="v2_role_features",
                              role_family="ST",
                              generation=80, fitness=0.9, seed=42)
load32 = FootballBrain.deserialize(d32)
check("v2 round-trip through schema validator", load32.w1.shape[0] == 32)

# 7c. a v2 brain with a non-role input width is REJECTED loudly.
bad = dict(d32)
bad["arch"] = [30, 32, 32, 10]
try:
    FootballBrain.deserialize(bad)
    rej = False
except Exception:  # noqa: BLE001  (BrainSchemaError)
    rej = True
check("v2 arch with input 30 is REJECTED (BrainSchemaError)", rej)

# 7d. synthetic corpus: role width, determinism, bounds; OFF stays 24-d.
c1 = generate_state_corpus("ST", 5, seed=7, input_size=32)
c2 = generate_state_corpus("ST", 5, seed=7, input_size=32)
check("role corpus width matches role input (ST -> 32)", c1.shape == (5, 32))
check("role corpus deterministic per seed", np.array_equal(c1, c2))
check("role corpus values in [0,1]", bool(np.all((c1 >= 0.0) & (c1 <= 1.0))))
c_off = generate_state_corpus("ST", 5, seed=7)
check("role-OFF corpus stays 24-d (v1 byte-compatible)", c_off.shape == (5, 24))

# 7e. decide() integration: a v2 ST brain consumes 32 sensors (role tail
# appended from the perceived scene); a v1 brain stays 24.
home, away, pe = world()
stg = SimpleNamespace(name="LEVEL")

st = next(p for p in home if p.position == "ST")
st.name = "ST_v2test"
clear_registry()
register_brain(st.name, FootballBrain.random(seed=3, input_size=32))
dec = NeuralDecisionBrain.decide(
    st, 70.0, 34.0, home, away, pe, None, False, True, stg,
    record_trace=True)
check("decide with v2 ST brain returns PlayerDecision",
      isinstance(dec, PlayerDecision))
check("decide v2 trace sensor length == 32",
      len(dec.trace["sensor_vector"]) == 32)
clear_registry()

cm = next(p for p in home if p.position == "CM")
cm.name = "CM_v1test"
clear_registry()
register_brain(cm.name, FootballBrain.random(seed=3))
dec1 = NeuralDecisionBrain.decide(
    cm, 50.0, 34.0, home, away, pe, None, False, True, stg,
    record_trace=True)
check("decide with v1 CM brain -> 24-d trace",
      len(dec1.trace["sensor_vector"]) == 24)
clear_registry()


print("=" * 30)
print(f"role_features tests: {'ALL PASS' if fails == 0 else str(fails) + ' FAILURES'}")
raise SystemExit(1 if fails else 0)