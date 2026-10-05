r"""
_diag_drift_bisect.py — is `test_position_engine_drift_toward_home` a
regression from the 2026-10-05 geometric_awareness fix, or pre-existing?

The previous probe (_diag_drift_regression.py) answered nothing: it built a
one-player team and the player never moved at all (start == final, 4.27 -> 4.27
in BOTH arms), so it measured a fixture that is not the test. A probe that
reproduces nothing is not evidence in either direction. Discarded.

This asks the question directly instead. It applies the EXACT pre-fix state —
`geometric_awareness = 50.0`, the dataclass default the builder never used to
overwrite — to the real file, runs the real pytest node, and restores the file
byte-exactly. Because the property test is Hypothesis-driven its examples are
draw-dependent, so each arm is run THREE times and the whole run is also
reported. Same discipline as the multi-seed corner A/B that was wrongly quoted
from a single seed: one sample of a property test is not a result.

Run: .venv\Scripts\python.exe _diag_drift_bisect.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
DNA = os.path.join(HERE, "player_dna.py")

# Default node, overridable on argv:  _diag_drift_bisect.py <node> [repeats]
NODE = (sys.argv[1] if len(sys.argv) > 1
        else "tests/test_preservation_properties.py::test_position_engine_drift_toward_home")
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

FIND = ('            geometric_awareness = cls._attr(arch, '
        '"mental.geometric_awareness", (50, 70), mental_age),')
REPL = "            geometric_awareness = 50.0,  # PRE-FIX STATE (bisect)"

_LINES: list[str] = []


def p(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def rule(ch: str = "=") -> None:
    p(ch * 78)


def run_node() -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", NODE, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=HERE, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def summarise(out: str) -> str:
    keep = []
    for ln in out.splitlines():
        if ("passed" in ln or "failed" in ln or "error" in ln.lower()) and len(ln) < 200:
            keep.append(ln.strip())
    return keep[-1] if keep else "(no summary line)"


def main() -> int:
    rule()
    p("A.  IS THE DRIFT FAILURE CAUSED BY THE FIX?  (real test, real file, 3 runs/arm)")
    rule()
    p("")
    p(f"  node: {NODE}")
    p("  The property test is Hypothesis-driven, so which examples run depends on")
    p("  the Hypothesis database and on what else is in the session. A single")
    p("  result — in either direction — is a sample, not a verdict.")
    p("")

    original = open(DNA, "rb").read()
    text = original.decode("utf-8")
    if FIND not in text:
        p("  !! ANCHOR NOT FOUND in player_dna.py — the file moved on.")
        p("     Not mutating: an anchor that does not match would mutate nothing")
        p("     and then report a clean PASS, which is the worst possible outcome.")
        return 2

    results: dict[str, list[tuple[int, str]]] = {}
    try:
        for arm, src in (("post-fix (shipped)", text),
                         ("pre-fix  (ga = 50.0 constant)", text.replace(FIND, REPL, 1))):
            open(DNA, "w", encoding="utf-8").write(src)
            runs = []
            for i in range(REPEATS):
                rc, out = run_node()
                runs.append((rc, summarise(out)))
            results[arm] = runs
            print(f"  {arm}")
            for i, (rc, s) in enumerate(runs, 1):
                print(f"    run {i}: exit {rc}   {s}")
            print()
    finally:
        open(DNA, "wb").write(original)

    after = open(DNA, "rb").read()
    rule()
    p("B.  RESTORE")
    rule()
    print(f"  player_dna.py byte-identical: {after == original}  ({len(after)} bytes)")
    if after != original:
        return 2

    rule()
    p("C.  VERDICT")
    rule()
    post = [rc for rc, _ in results["post-fix (shipped)"]]
    pre = [rc for rc, _ in results["pre-fix  (ga = 50.0 constant)"]]
    print(f"  post-fix exit codes: {post}")
    print(f"  pre-fix  exit codes: {pre}")
    print()

    if all(rc != 0 for rc in post) and all(rc == 0 for rc in pre):
        p("  >>> REGRESSION. The test passes only with the fix reverted, so the")
        p("      geometric_awareness change is the cause. Do NOT lower the band to")
        p("      make it pass — that is tuning real behaviour to satisfy an")
        p("      assertion whose premise (a wide player should converge on the")
        p("      defensive shape target) is now contested by a second, live rule.")
        p("      Diagnose which rule and whether it SHOULD apply to a wide player.")
        return 1
    if all(rc != 0 for rc in pre):
        p("  >>> PRE-EXISTING. The test fails with the fix reverted too, so the")
        p("      fix is not the cause and this is one more entry for the standing")
        p("      pre-existing-failure list in AGENTS.md.")
        return 0
    if any(rc != 0 for rc in post):
        p("  >>> INCONCLUSIVE / FLAKY. It does not fail in every arm, so this is")
        p("      example-dependence, not a deterministic effect. Recorded as such.")
        return 0
    p("  >>> GREEN in both arms. The earlier failure was example-dependence in a")
    p("      Hypothesis property test running inside a larger session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())