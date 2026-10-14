"""Can any soul, in any game state, produce an impossible probability?

The wiring is only safe if every modifier lands inside [0, ceiling] for
every soul at every context, including the most extreme one a match can
reach (losing, late, added time, under pressure, all at once).
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from player_soul import SoulApplicator, PlayerSoul, GreatnessPillars, SoulArchetype, SoulLibrary


class _Holder:
    def __init__(self, soul):
        self.dna = type("D", (), {"soul": soul})()


class _State:
    """A real state-shaped object, so _get_context resolves everything."""
    def __init__(self, minute=88, hg=0, ag=2, home="Us"):
        self.minute = minute
        self.home_goals = hg
        self.away_goals = ag
        self.home_team = home
        self.under_pressure = True


CONTEXTS = {
    "neutral 20'": _State(20, 0, 0),
    "winning 55'": _State(55, 2, 0),
    "LOSING 88'": _State(88, 0, 2),
    "LOSING 91' added": _State(91, 0, 3),
}
CASES = [
    ("PRESSING_EVANGELIST", "press_success_mult", "modify_press_success", 0.30),
    ("WALL", "tackle_success_mult", "modify_tackle_success", 0.65),
    ("DEFENSIVE_PURIST", "interception_mult", "modify_interception_rate", 0.55),
    ("SWEEPER_SAGE", "interception_mult", "modify_interception_rate", 0.55),
    ("ATTACKING_PROPHET", "shot_assist_mult", "modify_shot_assist_prob", 0.70),
    ("ATTACKING_PROPHET", "pass_accuracy_mult", "modify_pass_accuracy", 0.85),
    ("CREATIVE_ORACLE", "dribble_success_mult", "modify_dribble_success", 0.80),
    ("ATTACKING_PROPHET", "shot_quality_mult", "modify_shot_quality", 0.30),
]

print("=" * 100)
print("EVERY soul x EVERY wired modifier, in every game state")
print("=" * 100)
violations = 0
rows = []
for arch, key, fn, base in CASES:
    for pillars, tag in (((0.99, 0.99, 0.99), "max"), ((0.70, 0.70, 0.70), "min")):
        soul = PlayerSoul("p", archetype=getattr(SoulArchetype, arch),
                          pillars=GreatnessPillars(*pillars))
        for cname, st in CONTEXTS.items():
            out = getattr(SoulApplicator, fn)(_Holder(soul), base, st, "Us")
            if key == "shot_quality_mult":
                # xG multiplier: not a probability, so no ceiling. Only flag
                # something absurd.
                flag = "" if out < 5.0 else "  <-- ABSURD"
                if flag:
                    violations += 1
            else:
                ceil = SoulApplicator.CEILINGS[key]
                flag = "" if out <= ceil + 1e-9 else f"  <-- OVER {ceil}"
                if flag:
                    violations += 1
            rows.append((arch, key, tag, cname, base, out, flag))

print(f"{'soul':<21}{'multiplier':<24}{'pil':<5}{'context':<20}{'base':>6}{'result':>9}  ")
for a, k, t, c, b, o, f in rows:
    print(f"{a:<21}{k:<24}{t:<5}{c:<20}{b:>6.2f}{o:>9.4f}{f}")

print()
print(f"CEILING VIOLATIONS: {violations}")

print()
print("=" * 100)
print("NO SOUL = UNTOUCHED (the regression guard)")
print("=" * 100)
plain = _Holder(None)
for arch, key, fn, base in CASES:
    out = getattr(SoulApplicator, fn)(plain, base, _State(88, 0, 2), "Us")
    assert out == base, f"{fn} changed a non-soul player: {base} -> {out}"
print("  all 8 modifiers leave a non-soul player bit-identical: OK")

print()
print("=" * 100)
print("GREATNESS ORDERING: a stronger soul must never be WORSE")
print("=" * 100)
bad_order = 0
for arch, key, fn, base in CASES:
    if key == "shot_quality_mult":
        continue
    vals = []
    for p in ((0.70, 0.70, 0.70), (0.85, 0.85, 0.85), (0.99, 0.95, 0.95), (0.99, 0.99, 0.99)):
        s = PlayerSoul("p", archetype=getattr(SoulArchetype, arch),
                       pillars=GreatnessPillars(*p))
        vals.append(getattr(SoulApplicator, fn)(_Holder(s), base, _State(88, 0, 2), "Us"))
    mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    if not mono:
        bad_order += 1
    print(f"  {arch:<21}{key:<24} " + " -> ".join(f"{v:.4f}" for v in vals)
          + ("  monotone" if mono else "  <-- NOT MONOTONE"))
print(f"\nnon-monotone: {bad_order}")

print()
print("=" * 100)
print("THE OLD BEHAVIOUR, for the record")
print("=" * 100)
soul = PlayerSoul("p", archetype=SoulArchetype.PRESSING_EVANGELIST,
                  pillars=GreatnessPillars(0.99, 0.95, 0.92))
ctx = {"is_losing": True, "is_late_game": True, "under_pressure": True}
mult = soul.get_event_multiplier("press_success_mult", ctx)
print(f"  press_success_mult at 70' losing late under pressure = {mult}")
print(f"  0.30 * {mult} = {0.30*mult:.4f}   <- the +290% that got the site commented out")
print(f"  now: _apply_to_probability(0.30, {mult}, 0.80) = "
      f"{SoulApplicator._apply_to_probability(0.30, mult, 0.80):.4f}")
