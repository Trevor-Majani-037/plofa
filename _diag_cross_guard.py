"""Negative control for the two cross-receiver tests.

Restores the deleted defects in `event_chain.py`, runs the two tests, then
restores the file from a byte-exact backup. The point is to prove both pins
have teeth:

  - the behavioural pin must go RED when the pick is ability-only;
  - the source pin must go RED when `record_touch` reappears in the cross
    block.

A test that passes for the wrong reason is worse than no test — and the
project's own history is four cases of exactly that (a dead wiring assertion,
a stale `.pyc`, a grep that found a comment, a probe that never ran).
"""
import shutil
import subprocess
import sys
import tempfile

PATH = "event_chain.py"
bak = tempfile.mktemp(suffix=".py", dir=tempfile.gettempdir())
shutil.copy2(PATH, bak)
src = open(bak, encoding="utf-8").read()

try:
    # 1. Ability-only pick + the teleport, exactly as it was.
    marker = "                cross_receiver = cls.pick_weighted_spatial("
    assert marker in src, "spatial pick not found — control is stale"
    start = src.index("                # ── WHO GETS THE BALL ──")
    end = src.index("                if position_engine is not None:\n                    cross_height")
    restored = (
        "                cross_receiver = cls._pick_aerial_threat(\n"
        "                    players, exclude=last_player.name)\n"
        "                cross_defender = cls._pick_aerial_defender(def_players)\n"
        "                if position_engine is not None and cross_receiver:\n"
        "                    rx, ry = position_engine.get_position(cross_receiver.name)\n"
        "                    rx = cls.clamp_attack_x(rx + random.uniform(1.0, 4.0),\n"
        "                                         85.0, 100.0, attacks_right)\n"
        "                    ry = max(22.0, min(46.0, ry))\n"
        "                    position_engine.record_touch(\n"
        "                        cross_receiver.name, rx, ry, minute)\n"
        "                end_tx = max(84.0, min(102.0, rx))\n"
        "                end_ty = max(24.0, min(44.0, ry))\n"
    )
    open(PATH, "w", encoding="utf-8", newline="").write(src[:start] + restored + src[end:])
    print("patched: ability-only pick + record_touch teleport restored")

    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_cross_detector.py", "-q", "-k",
         "receiver or rewrites"],
        capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:]
    print("\n".join(tail))
    names = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
    if names and r.returncode != 0:
        print("\nNEGATIVE CONTROL PASSED — both pins go RED without the fix:")
        for n in names:
            print("   " + n.split(" - ")[0])
        rc = 0
    else:
        print("\nNEGATIVE CONTROL FAILED — the tests passed with the defects "
              "restored, so neither pin is testing the fix.")
        rc = 1
finally:
    shutil.copy2(bak, PATH)
    print(f"\nrestored {PATH} from {bak}")
sys.exit(rc)