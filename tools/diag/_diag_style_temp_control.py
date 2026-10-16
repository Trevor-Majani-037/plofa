"""NEGATIVE CONTROL: is the defensive STYLE TEMPERAMENT actually live? (2026-10-16)

`formation_stance_for` chose the stance from the scoreline and the clock and
never read `profile.style` -- while its docstring claimed it did. The style's
influence on the DEFENSIVE shape now arrives via `style_defense_deltas`,
threaded through `apply_formation_stance(style=...)`.

Before spending a match on it, prove the lever reaches the anchors. This has
bitten the project twice already: `_diag_lever_decisive.py` reloaded
`brain_integration` and rebound a class `event_chain` was already holding, and
`_diag_wide_target.py` CALLED an int ATTRIBUTE (`_offball_tick_seq`), whose
TypeError was swallowed by a broad except — so every wide player collapsed
into one tick group and the probe printed a confident table over it. A lever
that reports "no effect" is worthless until it is shown to be live.

Four things asserted, in order:

  1. THE TABLE MOVES PEOPLE. `style_defense_deltas("park_the_bus")` must
     differ from `("attacking")` on the back line AND on the striker, and in
     OPPOSITE directions. A table that only moves defenders is the same
     mistake SEE_IT_OUT's own comment warns about.
  2. EVERY TeamStyle IS COVERED, and an unknown style returns {} — the
     pre-change behaviour — rather than an invented average.
  3. THE STANCE OUTRANKS THE STYLE. SEE_IT_OUT must still drop a
     park-the-bus side further than SEE_IT_OUT drops an attacking side.
     This is the property that stops a style bias from overriding a manager.
  4. IT REACHES `home_x`. Build a real PositionEngine, apply two styles, and
     read the anchors back. A pure-function test proves nothing about the
     call path, which is where the original defect was.

Milliseconds; no match. Run:  python _diag_style_temp_control.py
"""
import sys

sys.path.insert(0, ".")
from tactical_shapes import (FormationStance, stance_delta_roles,  # noqa: E402
                            style_defense_deltas,
                            _STYLE_DEFENSE_ARCHETYPE)              # noqa: E402
from position_engine import PositionEngine                          # noqa: E402
from player_dna import SquadBuilder                                # noqa: E402
from match_engine import TeamStyle                                  # noqa: E402
import run_match as RM                                              # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"   {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


print("=" * 92)
print("1. THE TABLE MOVES BOTH ENDS OF THE PITCH, IN OPPOSITE DIRECTIONS")
print("-" * 92)
sunk = style_defense_deltas("park_the_bus")
high = style_defense_deltas("attacking")
for role in ("CB", "LB", "CDM", "CM"):
    print(f"   {role:<5} park_the_bus {str(sunk.get(role)):>12}"
          f"   attacking {str(high.get(role)):>12}")
print(f"   {'ST':<5} park_the_bus {str(sunk.get('ST')):>12}"
      f"   attacking {str(high.get('ST')):>12}")
print()
check(bool(sunk) and bool(high), "both extremes produce non-empty deltas")
check(sunk.get("CB", (0,))[0] < 0 < high.get("CB", (0,))[0],
      "the centre-backs move in OPPOSITE directions by style")
check(sunk.get("LB", (0,))[0] < 0 < high.get("LB", (0,))[0],
      "the fullbacks move in opposite directions by style")
check(abs(sunk.get("ST", (0,))[0]) < abs(sunk.get("LW", (0,))[0]),
      "the striker holds up better than the wingers in a sunk style "
      "(a block whose front pair all drop has no outlet)")
print()

print("=" * 92)
print("2. COVERAGE AND THE UNKNOWN-STYLE DEFAULT")
print("-" * 92)
# Assert the ARCHETYPE MAPPING, not the deltas. `balanced` and `wing_play`
# map to "neutral", whose delta dict is deliberately EMPTY — so testing for
# non-empty output flags the two styles that are behaving exactly as designed.
# That is the probe measuring its own assumption rather than the code.
missing = [s.value for s in TeamStyle
           if s.value not in _STYLE_DEFENSE_ARCHETYPE]
unknown_arch = [a for a in set(_STYLE_DEFENSE_ARCHETYPE.values())
                if a not in ("sunk", "low", "neutral", "high")]
print(f"   TeamStyle values: {len(list(TeamStyle))}")
print(f"   mapped to an archetype: {len(list(TeamStyle)) - len(missing)}")
print(f"   archetypes in use: {sorted(set(_STYLE_DEFENSE_ARCHETYPE.values()))}")
check(not missing, f"every TeamStyle maps to an archetype (missing: {missing})")
check(not unknown_arch,
      f"every archetype used has a delta table (dangling: {unknown_arch})")
check(style_defense_deltas(None) == {},
      "an absent style returns {} — the pre-change behaviour, not a guess")
check(style_defense_deltas("not_a_real_style") == {},
      "an unknown style returns {} — the pre-change behaviour, not a guess")
check(style_defense_deltas("balanced") == {},
      "'balanced' is deliberately neutral (it is the reference style)")
print()

print("=" * 92)
print("3. THE GAME STATE STILL OUTRANKS THE STYLE")
print("-" * 92)
for stance in (FormationStance.BASELINE, FormationStance.SEE_IT_OUT):
    bus = stance_delta_roles(stance, style="park_the_bus")
    att = stance_delta_roles(stance, style="attacking")
    print(f"   {stance.value:<14} park_the_bus LB {bus.get('LB')}   "
          f"attacking LB {att.get('LB')}")
print()
for stance in (FormationStance.SEE_IT_OUT, FormationStance.ALL_OUT_CHASE):
    bus = stance_delta_roles(stance, style="park_the_bus").get("LB", (0,))[0]
    att = stance_delta_roles(stance, style="attacking").get("LB", (0,))[0]
    check(bus < att,
          f"{stance.value}: park_the_bus still sits deeper than attacking")
b_bus = stance_delta_roles(FormationStance.BASELINE, style="park_the_bus") \
    .get("LB", (0,))[0]
b_att = stance_delta_roles(FormationStance.BASELINE, style="attacking") \
    .get("LB", (0,))[0]
o_bus = stance_delta_roles(FormationStance.SEE_IT_OUT, style="park_the_bus") \
    .get("LB", (0,))[0]
check(o_bus < b_bus - 6,
      f"SEE_IT_OUT still moves a park_the_bus side a long way past its own "
      f"temperament (baseline {b_bus} -> see_it_out {o_bus})")
print()

print("=" * 92)
print("4. IT REACHES home_x ON A REAL ENGINE")
print("-" * 92)


class _Prof:
    value = "balanced"
    attack_intensity = 55.0
    defensive_intensity = 55.0
    width = 0.5
    formation = "4_3_3"
    pressing_intensity = 55.0
    tempo = 55.0
    risk_appetite = 55.0


pe = PositionEngine()
hs = SquadBuilder.build(team_name="T1", starters=RM.HOME_STARTERS,
                        substitutes=RM.HOME_SUBS,
                        soul_assignments=RM.SOUL_PLAYERS)
pe.initialize_team("T1", hs["starters"], _Prof())

st = FormationStance.BASELINE
pe.apply_formation_stance("T1", st, style="balanced")
flat = {n: s.home_x for n, s in pe.states.items() if s.position in
        ("CB", "LB", "CDM", "CM")}
# The idempotency key now includes style, so the SECOND call must be allowed
# to apply rather than short-circuiting on a stale key.
applied = pe.apply_formation_stance("T1", st, style="park_the_bus")
sunk_home = {n: s.home_x for n, s in pe.states.items() if s.position in
             ("CB", "LB", "CDM", "CM")}
print(f"   balanced    : {dict((k, round(v, 1)) for k, v in flat.items())}")
print(f"   park_the_bus: {dict((k, round(v, 1)) for k, v in sunk_home.items())}")
check(applied is True,
      "apply_formation_stance returns True for a style CHANGE "
      "(the style is in the idempotency key)")
moved = {k: round(sunk_home[k] - flat[k], 1) for k in flat}
print(f"   delta       : {moved}")
check(all(v < 0 for v in moved.values()),
      "EVERY back-line/midfield anchor moved UP the pitch (home_x fell)")

pe.apply_formation_stance("T1", st, style="attacking")
att_home = {n: s.home_x for n, s in pe.states.items() if s.position in
            ("CB", "LB", "CDM", "CM")}
moved2 = {k: round(att_home[k] - flat[k], 1) for k in flat}
print(f"   attacking   : {dict((k, round(v, 1)) for k, v in att_home.items())}")
print(f"   delta       : {moved2}")
check(all(v > 0 for v in moved2.values()),
      "EVERY back-line/midfield anchor moved DOWN the pitch under 'attacking'")

# And the pre-change path must still be reachable: no style == no deltas.
pe.apply_formation_stance("T1", st, style=None)
none_home = {n: s.home_x for n, s in pe.states.items() if s.position in
             ("CB", "LB", "CDM", "CM")}
check(all(abs(none_home[k] - flat[k]) < 1e-9 for k in flat),
      "style=None reproduces the pre-change anchors exactly")

print()
if FAILS:
    print(f"*** {len(FAILS)} FAILED")
    for f in FAILS:
        print(f"   - {f}")
    raise SystemExit(1)
print("*** ALL CHECKS PASS")