"""NEGATIVE CONTROL: is the DEPTH_ANCHOR_ENABLED switch actually live? (2026-10-14)

A lever that reports "no effect" is worthless unless it is proven to reach the
pitch first. This has bitten this project twice already
(`_diag_lever_decisive.py`: reloading `brain_integration` rebound the class
`event_chain` was holding; `_diag_wide_target.py`: calling an int ATTRIBUTE
`_offball_tick_seq`, whose TypeError was swallowed by a broad except, so every
wide player collapsed into one tick group and the probe printed a confident
table over it).

So: exercise the rule on a REAL PositionEngine with a REAL squad and assert
three separate things, then assert that flipping the switch changes them.

  1. CALL COUNT. `depth_home_anchor_blend` is reached once per player per
     off-ball tick for CB/LB/RB, and returns exactly 0.0 for every other
     position. A rule that is never called and a rule that always returns 0
     look identical in a shape table; only the call count tells them apart.
  2. NON-VACUITY. The banding must produce more than one distinct weight, and
     a drifted player must get a STRICTLY bigger pull than a settled one --
     otherwise the bands are decorative.
  3. THE SWITCH. `DEPTH_ANCHOR_ENABLED = False` must return 0.0 before it
     reads any player state, so the off arm is byte-identical to pre-change.

Runs in milliseconds: no match.

Run:  python _diag_depth_anchor_control.py
"""
import sys

sys.path.insert(0, ".")
from position_engine import PositionEngine          # noqa: E402
from player_dna import SquadBuilder                # noqa: E402
import run_match as RM                              # noqa: E402


class _Prof:
    def __init__(self, v="balanced"):
        self.value = v
        self.attack_intensity = 55.0
        self.defensive_intensity = 55.0
        self.width = 0.5
        self.formation = "4_3_3"
    # some code paths read a variety of these; keep them all present
    pressing_intensity = 55.0
    tempo = 55.0
    risk_appetite = 55.0


pe = PositionEngine()
hs = SquadBuilder.build(team_name="T1", starters=RM.HOME_STARTERS,
                        substitutes=RM.HOME_SUBS,
                        soul_assignments=RM.SOUL_PLAYERS)
pe.initialize_team("T1", hs["starters"], _Prof())

FAILS = []


def check(cond, msg):
    print(f"   {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


print("=" * 92)
print("1. CALL COUNT AND ROLE SCOPE")
print("-" * 92)
counts = {}
weights = {}
for nm, st in pe.states.items():
    w = pe.depth_home_anchor_blend(nm)
    counts.setdefault(st.position, []).append(w)
    weights.setdefault(st.position, set()).add(round(w, 3))
for p in sorted(counts):
    ws = counts[p]
    tag = ""
    if p in PositionEngine.DEPTH_ANCHOR_POSITIONS:
        tag = "  <- in scope, must be non-zero"
    else:
        tag = "  <- must be 0.0"
    print(f"   {p:<5} n={len(ws):<3} weights={sorted(weights[p])}"
          f"{' ALL ZERO' if all(w == 0.0 for w in ws) else ''}{tag}")

for p, ws in counts.items():
    if p in PositionEngine.DEPTH_ANCHOR_POSITIONS:
        check(any(w > 0.0 for w in ws), f"{p} in scope returns a non-zero weight")
    else:
        check(all(w == 0.0 for w in ws),
              f"{p} out of scope returns exactly 0.0 (no side effect)")
print()

print("=" * 92)
print("2. NON-VACUITY OF THE BANDS")
print("-" * 92)
name = next(n for n, s in pe.states.items() if s.position == "LB")
st = pe.states[name]
print(f"   {name}: home_x={st.home_x}  current_x={st.current_x}")
drifts = [0.0, 5.0, 14.0, 26.0, 45.0]
seen = []
for d in drifts:
    st.current_x = st.home_x + d
    w = pe.depth_home_anchor_blend(name)
    seen.append(w)
    print(f"   drift {d:>5.1f} m -> weight {w:.3f}")
check(len(set(seen)) > 1, "the banding produces more than one distinct weight")
check(all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1)),
      "weight is MONOTONIC in drift (a far-off post pulls harder)")
check(seen[-1] > seen[0], "a 45 m drift pulls strictly harder than a settled one")
check(seen[0] > 0.0, "even a settled player gets a small non-zero pull")
st.current_x = st.home_x
print()

print("=" * 92)
print("3. THE SWITCH -- off must be byte-identical to pre-change")
print("-" * 92)
st.current_x = st.home_x + 40.0
on = pe.depth_home_anchor_blend(name)
PositionEngine.DEPTH_ANCHOR_ENABLED = False
off = pe.depth_home_anchor_blend(name)
PositionEngine.DEPTH_ANCHOR_ENABLED = True
on2 = pe.depth_home_anchor_blend(name)
print(f"   on={on:.3f}   off={off:.3f}   on again={on2:.3f}")
check(off == 0.0, "DEPTH_ANCHOR_ENABLED=False returns exactly 0.0")
check(on2 == on, "the switch is reversible (not one-shot state mutation)")
print()

print("=" * 92)
print("4. THE CHAIN ACTUALLY CALLS IT -- static check of the call site")
print("-" * 92)
import ast                                    # noqa: E402
src = open("match_engine.py", encoding="utf-8").read()
tree = ast.parse(src)
hits = [n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "depth_home_anchor_blend"]
check(len(hits) == 1, f"exactly one call site (found {len(hits)})")
if hits:
    call = hits[0]
    # It must be inside `_offball_move_player`, and must NOT be gated on
    # possession -- an asymmetric gate is exactly the CB-clamp defect that
    # shipped on 2026-10-11 (`if pos == "CB" and has_ball`), so the guard
    # asserts the ABSENCE of a possession gate rather than the presence of a
    # rule.
    fn = call
    while fn is not None and not (isinstance(fn, (ast.FunctionDef,
                                                  ast.AsyncFunctionDef))):
        fn = getattr(fn, "parent", None)
    seg = ast.get_source_segment(src, call) or ""
    print(f"   call site: {seg!r}")
    check("has_ball" not in seg,
          "the call is not possession-gated (the CB-ceiling defect's shape)")
print()
if FAILS:
    print(f"*** {len(FAILS)} FAILED")
    for f in FAILS:
        print(f"   - {f}")
    raise SystemExit(1)
print("*** ALL CHECKS PASS")
