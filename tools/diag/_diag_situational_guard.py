"""NEGATIVE CONTROL for tests/test_situational_exploration.py.

A guard that has never been seen to fail is not a guard.  This reintroduces
the situation-aware exploration -- and the pieces that stop it -- one at a
time, and requires the RIGHT test class to go red, not merely "something
failed".

Five mutations, all in surrogate_collect.py:

  M1  the phase weights are empty (the weighting does nothing)
  M2  the exploration RATE is not raised in the attacking phases
  M3  there is no `goal_close` phase (it is folded into the final third, so
      shots are sampled where the geometry refuses them -- and the refused
      strike is a zero-length marker carry worth exactly 0.0)
  M4  the role lookup is broken (every player is treated as a striker, so
      the keeper is asked to shoot)
  M5  the switch is on by default

Every mutant must still COMPILE.  A syntax error counted as a caught
mutation proves nothing about which assertion pins the defect.  Anchors are
escaped literals joined with an explicit ``\\r?\\n``, because this file is
CRLF and an LF anchor matches NOTHING -- a mutation that silently does
nothing is the failure this control exists to catch.

Each file is restored from an in-memory snapshot taken immediately before its
mutation and verified by sha256.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_situational_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = ["tests/test_situational_exploration.py"]


def _pat(*lines: str) -> str:
    """An exact multi-line anchor, tolerant of the file's own line endings."""
    return r"\r?\n".join(re.escape(l) for l in lines)


MUTATIONS = [
    ("surrogate_collect.py",
     "M1 the phase weights are empty (the weighting does nothing)",
     _pat("        weights = _SITUATION_ACTIONS.get(phase, {})"),
     "        weights = {}",
     "TheSituationAwareExplorationTests"),

    ("surrogate_collect.py",
     "M2 the exploration RATE is not raised in the attacking phases",
     _pat("        if phase in _EXPLORE_ATTACKING_PHASES:",
          "            eps = max(eps, _EXPLORE_EPS_FINAL_THIRD)"),
     "        if False:\n"
     "            eps = max(eps, _EXPLORE_EPS_FINAL_THIRD)",
     "TheSituationAwareExplorationTests"),

    ("surrogate_collect.py",
     "M3 there is no goal_close phase (shots are sampled where the geometry "
     "refuses them)",
     _pat('    if final_third and goal_close:',
          '        return "goal_close"'),
     '    if False:\n'
     '        return "goal_close"',
     "ThePhaseIsTheBucketPhaseTests"),

    ("surrogate_collect.py",
     "M4 the role lookup is broken (every player is treated as a striker)",
     _pat('    fam = _role_family(getattr(player, "position", "") or "")'),
     '    fam = "ST"',
     "TheSituationAwareExplorationTests"),

    ("surrogate_collect.py",
     "M5 the switch is on by default",
     _pat("_EXPLORE_SITUATIONAL = False",
          "",
          "#: The exploration RATE in the final third, when the situational "
          "exploration"),
     "_EXPLORE_SITUATIONAL = True\n"
     "\n"
     "#: The exploration RATE in the final third, when the situational "
     "exploration",
     "TheSwitchIsOffByDefaultTests"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_tests() -> tuple:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True,
        env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO),
             "PATH": os.environ.get("PATH", ""),
             "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    originals = {}
    for rel in {m[0] for m in MUTATIONS}:
        originals[rel] = (REPO / rel).read_bytes()
    hashes = {k: sha(v) for k, v in originals.items()}

    rc, out = run_tests()
    if rc != 0:
        print("BASELINE IS NOT GREEN -- aborting, nothing measured.")
        print(out[-2500:])
        return 1
    print("baseline: green")
    print()

    failures = 0
    try:
        for rel, label, pattern, repl, expected in MUTATIONS:
            target = REPO / rel
            snapshot = target.read_bytes()
            body = snapshot.decode("utf-8")
            hits = len(re.findall(pattern, body))
            if hits != 1:
                print(f"  [{rel}] {label}\n    ANCHOR MATCHED {hits} TIMES -- "
                      f"the mutation is a no-op, aborting this arm.")
                failures += 1
                target.write_bytes(snapshot)
                continue

            mutant = re.sub(pattern, repl, body, count=1)
            try:
                compile(mutant, str(target), "exec")
            except SyntaxError as exc:
                print(f"  [{rel}] {label}\n    MUTANT DOES NOT COMPILE "
                      f"({exc}) -- counting as a MISS, not a catch.")
                failures += 1
                target.write_bytes(snapshot)
                continue

            target.write_bytes(mutant.encode("utf-8"))
            rc, out = run_tests()
            caught = rc != 0 and expected in out
            if caught:
                names = re.findall(r"FAILED (\S+?)(?:\s+-|$)", out, re.M)
                short = [n.split("::", 1)[1] if "::" in n else n for n in names]
                print(f"  [{rel}] {label}\n    CAUGHT: "
                      f"{', '.join(dict.fromkeys(short)) or '?'}")
            else:
                print(f"  [{rel}] {label}\n    MISS (rc={rc}); expected "
                      f"{expected} to fail.")
                print("    " + (out.strip().splitlines() or [""])[-1])
                failures += 1

            target.write_bytes(snapshot)
            assert sha(target.read_bytes()) == sha(snapshot), "RESTORE FAILED"
    finally:
        for rel, blob in originals.items():
            (REPO / rel).write_bytes(blob)

    for rel, blob in originals.items():
        assert sha((REPO / rel).read_bytes()) == hashes[rel], \
            f"FINAL RESTORE FAILED for {rel}"
    rc, out = run_tests()
    print()
    for rel, h in hashes.items():
        print(f"restored byte-identically: {rel} (sha256 {h[:16]})")
    print(f"baseline after restore: {'green' if rc == 0 else 'RED'}")
    print()
    print("ALL CONTROLS PASSED" if not failures else f"{failures} MISS(ES)")
    return 0 if (failures == 0 and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
