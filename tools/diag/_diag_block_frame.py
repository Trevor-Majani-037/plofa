"""UNIT CHECK: is `defensive_block_target`'s depth axis mirrored? (2026-10-15)

`position_engine.defensive_block_target` computes:

    depth = base_depth + BLOCK_FRONT_DROP * (off_spine / 34)
    tx = (PITCH_X_MAX - depth if attacks_right else depth)

`depth` is measured FROM THE PLAYER'S OWN GOAL -- the whole preset table is
documented that way ("back four behind midfield", LB_BACK_DEPTH = 19.0, and
`_low_block_depth` returns "depth from own goal"). A defender 19 m from his own
goal therefore sits at x = 19 when his team attacks RIGHT (own goal at x = 0),
and at x = 105 - 19 = 86 when his team attacks LEFT (own goal at x = 105).

The code writes the opposite in both branches.

Why this is the same bug three times over: the project's own history records
an `attacks_right` mirror error in the corner chain (every away-team corner
resolved at its own end), in the PLOFA export frame, in the `pass_direction`
probe, and in `StrikerSpatialProfile.offside_line_nx`. It is the single most
recurring defect in this codebase and it is invisible in a source read --
`PITCH_X_MAX - depth` looks like "distance from the far goal", which is a
reasonable thing to want until you remember `depth` is measured from the other
end.

This script does not simulate a match. It calls the function and checks the
number against the documented meaning of its own constant.

Run:  python _diag_block_frame.py
"""
import sys

sys.path.insert(0, ".")
from position_engine import PositionEngine      # noqa: E402
from player_dna import SquadBuilder            # noqa: E402
import run_match as RM                          # noqa: E402


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
pe.team_attacks_right["T1"] = True

FAILS = []


def check(cond, msg):
    print(f"   {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


print("=" * 92)
print("WHAT THE CONSTANT ITSELF SAYS")
print("-" * 92)
print(f"   LB_BACK_DEPTH (back four, low block) = {PositionEngine.LB_BACK_DEPTH}")
print("   BLOCK_PRESETS:")
for k, v in PositionEngine.BLOCK_PRESETS.items():
    print(f"      {k:<5} back={v[0]:<5} mid={v[1]:<5} front={v[2]}")
print("   The docstring: \"Returns (depth from own goal, lateral half-width)\".")
print()

print("=" * 92)
print("1. A LOW BLOCK'S BACK FOUR, TEAM ATTACKING RIGHT (own goal at x=0)")
print("   A 19 m block line must sit at x ~ 19. Anything near 105-19 = 86 is")
print("   the defender standing in the OPPONENT'S penalty area.")
print("-" * 92)
print(f"{'player':<18}{'preset':<8}{'tx':>8}{'ty':>8}{'expected ~':>13}")
for preset in ("low", "mid", "high"):
    for nm in ("Darius Frost", "Rico Alves", "Emeka Obi"):
        r = pe.defensive_block_target(nm, ball_x=50.0, ball_y=34.0,
                                      attacks_right=True, preset=preset)
        if r is None:
            continue
        _, tx, ty = r
        back = PositionEngine.BLOCK_PRESETS[preset][0]
        print(f"{nm:<18}{preset:<8}{tx:>8.1f}{ty:>8.1f}{back:>13.1f}")
print()
for preset in ("low", "mid", "high"):
    r = pe.defensive_block_target("Emeka Obi", ball_x=50.0, ball_y=34.0,
                                  attacks_right=True, preset=preset)
    back = PositionEngine.BLOCK_PRESETS[preset][0]
    check(r is not None and abs(r[1] - back) < 8.0,
          f"'{preset}' block back four sits near nx {back:.0f} (got "
          f"{r[1]:.1f}) when attacking right")

print()
print("=" * 92)
print("2. THE MIRROR, ATTACKING LEFT (own goal at x=105)")
print("   Same team, mirrored: a 19 m block line must sit at x ~ 86.")
print("-" * 92)
pe.team_attacks_right["T1"] = False
for preset in ("low", "mid", "high"):
    r = pe.defensive_block_target("Emeka Obi", ball_x=50.0, ball_y=34.0,
                                  attacks_right=False, preset=preset)
    back = PositionEngine.BLOCK_PRESETS[preset][0]
    exp = 105.0 - back
    print(f"   {preset:<6} tx={r[1]:>6.1f}   expected ~{exp:>6.1f}")
for preset in ("low", "mid", "high"):
    r = pe.defensive_block_target("Emeka Obi", ball_x=50.0, ball_y=34.0,
                                  attacks_right=False, preset=preset)
    back = PositionEngine.BLOCK_PRESETS[preset][0]
    check(r is not None and abs(r[1] - (105.0 - back)) < 8.0,
          f"'{preset}' block back four mirrors correctly attacking left (got "
          f"{r[1]:.1f}, expected ~{105.0 - back:.1f})")
pe.team_attacks_right["T1"] = True

print()
print("=" * 92)
print("3. THE SIGN OF THE ERROR, STATED PLAINLY")
print("-" * 92)
r_right = pe.defensive_block_target("Emeka Obi", 50.0, 34.0, True, "low")[1]
print(f"   attacking right: a block 19 m from goal returns tx = {r_right:.1f}")
if r_right > 50:
    print(f"   -> the block line is placed {r_right - 19:.0f} m FURTHER UP THE")
    print(f"      PITCH than the shape it is supposed to describe. The centre-")
    print(f"      backs defend from inside the opponent's penalty area.")
print()
print("   The engine's own comment on the low block says: \"The first attempt")
print("   steered the finished target by 0.85 at the end of the chain ... the")
print("   shape reached 47.5 m from its own goal when the block slot was 19 m.\"")
print("   That is THIS bug, already observed once and attributed to a blend")
print("   weight instead. The blend was raised; the mirror was never fixed.")

if FAILS:
    print(f"\n*** {len(FAILS)} FAILED -- the block depth axis is mirrored")
    raise SystemExit(1)
print("\n*** ALL CHECKS PASS")
