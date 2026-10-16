"""WHAT TACTICAL FRAME DID EACH ASANTE NUMBER COME FROM? (2026-10-15)

The claim under test, from the user: "Asante's pass map should be 100% affected
by team style... you can't say his pass map is worse without actively checking
his team to see why."

I reported Asante's median depth as 59.3 -> 46.2 and called it a regression
against a fixed "real CAM 58-70" band. That comparison is only valid if both
numbers were produced under the SAME tactical frame. They were not:

  * 59.3 came from `_diag_wide_depth_live.py`, which builds Thornfield with
    TeamStyle.BALANCED / PlayingStyle.MIXED.
  * 46.2 came from `_diag_depth_anchor_ab.py`, which builds Thornfield with
    TeamStyle.BALANCED / PlayingStyle.MIXED.

...so the two probe arms DO match. The frame that does NOT match is the one the
user was actually looking at: `plofa_output/Hartwell_City_vs_Thornfield_United_
MD01`, whose styles are whatever the season/scratch runner set. If that match
ran a more defensive frame, then a deeper Asante is the CORRECT output of that
frame and my "regression" is a category error.

This script reads the export's own match block and prints it, then checks
whether the engine actually has a style -> line-depth mechanism (it must, or
the user's claim is false and mine stands).

Run:  python _diag_asante_frame.py
"""
import json
import os
import sys

sys.path.insert(0, ".")

EXP = os.path.join("plofa_output", "Hartwell_City_vs_Thornfield_United_MD01",
                   "Hartwell_City_vs_Thornfield_United_MD1.json")
doc = json.load(open(EXP, encoding="utf-8"))
m = doc["match"]

print("=" * 92)
print("1. THE MATCH THE USER WAS LOOKING AT -- its own tactical frame")
print("-" * 92)
for k in sorted(m):
    v = m[k]
    if isinstance(v, (str, int, float, bool)) or v is None:
        print(f"   {k:<28} {v}")
print()
for k in ("home_team", "away_team"):
    if k in m:
        print(f"   {k}: {m[k]}")
print()

print("=" * 92)
print("2. THE FRAME MY TWO PROBE ARMS USED")
print("-" * 92)
print("   _diag_wide_depth_live.py   : Thornfield BALANCED / MIXED")
print("   _diag_depth_anchor_ab.py   : Thornfield BALANCED / MIXED")
print("   -> the two arms match each other.")
print()

print("=" * 92)
print("3. DOES THE ENGINE ACTUALLY HAVE A STYLE -> LINE-DEPTH MECHANISM?")
print("   If not, the user's claim is false and my comparison stands.")
print("-" * 92)
import position_engine as PE                       # noqa: E402
import match_engine as ME                         # noqa: E402

# The mechanism, read off the source rather than assumed:
#   _recompute_homes: home = base + stance_delta + pattern_delta + coach_delta
#   apply_formation_stance / apply_attack_pattern set those deltas.
# So the question is whether TeamStyle/PlayingStyle reach them.
src = open("position_engine.py", encoding="utf-8").read()
for token in ("_stance_deltas", "_pattern_deltas", "_coach_deltas",
              "def apply_formation_stance", "def apply_attack_pattern",
              "def _recompute_homes"):
    print(f"   position_engine.py contains {token!r}: {token in src}")
src2 = open("match_engine.py", encoding="utf-8").read()
for token in ("apply_formation_stance", "apply_attack_pattern",
              "team_stances", "team_patterns"):
    print(f"   match_engine.py   contains {token!r}: {token in src2}")
print()

print("=" * 92)
print("4. WHAT THE DELTA TABLES ACTUALLY CONTAIN (depth terms only)")
print("-" * 92)
import inspect                                     # noqa: E402
for name in ("apply_formation_stance", "apply_attack_pattern"):
    fn = getattr(PE.PositionEngine, name, None)
    if fn is None:
        print(f"   {name}: NOT FOUND")
        continue
    seg = inspect.getsource(fn)
    print(f"   --- {name} ---")
    for line in seg.splitlines():
        s = line.strip()
        if any(t in s for t in ("delta", "stance", "pattern", "width",
                                "depth", "return", "def ")):
            print(f"      {s[:96]}")
print()

print("=" * 92)
print("5. THE VERDICT THIS PROBE CAN AND CANNOT SUPPORT")
print("-" * 92)
print("   It can say whether a style -> line-depth mechanism EXISTS.")
print("   It cannot say whether Asante's 46.2 is right for Thornfield's frame,")
print("   because that needs his pass origins measured UNDER THAT FRAME. The")
print("   export in section 1 is the only record of it, and it predates the")
print("   fixes -- so there is no post-fix pass map for that frame at all.")
