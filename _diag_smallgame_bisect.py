"""BISECT: are those two `test_small_game` failures mine?

AGENTS.md records two failures in that file as pre-existing:
  * `test_striker_gets_in_behind_not_offside` — an XPASS(strict), i.e. the
    xfail marker is STALE (the documented 2026-09-30 open finding now
    passes);
  * `test_fullback_can_tuck_inside` — "identical number at HEAD".

"Documented pre-existing is not evidence" is this project's own standing
rule, so this BISECTS: it puts the engine back into the PRE-FIX state
(the decision time switched off, which is what widened the chain windows)
and runs the same two tests.  If they fail identically in both arms, they
are independent of this work.

Run:  .venv\\Scripts\\python.exe -u _diag_smallgame_bisect.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS = ("tests/test_small_game.py::test_striker_gets_in_behind_not_offside",
         "tests/test_small_game.py::test_fullback_can_tuck_inside")

PATCH = """
import match_timeline
match_timeline.DECISION_S = 0.0
match_timeline.CONTROL_S = 0.0
"""


def run(tag, patch):
    env = dict(os.environ)
    if patch:
        # a conftest that patches the module BEFORE the tests import it
        path = os.path.join(ROOT, "_diag_bisect_conftest.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(patch)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        # pytest picks up a root conftest.py only; use -p to load the patch
        cmd = [sys.executable, "-m", "pytest", "-q", "-p",
               "no:randomly", "-p", "_diag_bisect_conftest", *TESTS]
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly",
               *TESTS]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(f"\n=== {tag} ===")
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-12:]
    print("\n".join(tail))
    return r.stdout


def main():
    run("ARM 1: AS SHIPPED (the decision time ON)", None)
    out = run("ARM 2: THE PRE-FIX STATE (the decision time OFF)", PATCH)
    # clean up the patch module
    try:
        os.remove(os.path.join(ROOT, "_diag_bisect_conftest.py"))
    except OSError:
        pass
    print("\nThe two arms must show the SAME failures for the failures to be "
          "independent of this work.")


if __name__ == "__main__":
    main()
