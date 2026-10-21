"""NEGATIVE CONTROL for tests/test_cover_and_screen.py.

A guard that has never been seen to fail is not a guard.  This reintroduces
the defects the cover and the screen exist to prevent, one at a time, and
requires the RIGHT test class to go red -- not merely "something failed".

Six mutations:

  M1  the cover's socket is 20 m away instead of the football's ~6 m
  M2  the cover does not come ACROSS
  M3  the cover is not level with the carrier (the user's +/-3 m)
  M4  CK38's split fires again when our own centre-back has the ball
  M5  the CDM's line gap is not held
  M6  the cover approaches at a jog instead of a sprint

Every mutant must still COMPILE.  A syntax error counted as a caught mutation
proves nothing about which assertion pins the defect.  Anchors are escaped
literals joined with an explicit ``\\r?\\n``, because these files are CRLF and
an LF anchor matches NOTHING -- a mutation that silently does nothing is the
failure this control exists to catch.

Each file is restored from an in-memory snapshot taken immediately before its
mutation and verified by sha256.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_cover_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = ["tests/test_cover_and_screen.py"]


def _pat(*lines: str) -> str:
    """An exact multi-line anchor, tolerant of the file's own line endings."""
    return r"\r?\n".join(re.escape(l) for l in lines)


MUTATIONS = [
    ("position_engine.py",
     "M1 the cover's socket is 20 m away instead of ~6 m",
     _pat("    COVER_DIST_M: float = 6.0"),
     "    COVER_DIST_M: float = 20.0",
     "CoverTests"),

    ("position_engine.py",
     "M2 the cover does not come ACROSS",
     _pat("    COVER_ACROSS_M: float = 5.5"),
     "    COVER_ACROSS_M: float = 0.0",
     "CoverTests"),

    ("position_engine.py",
     "M3 the cover is not level with the carrier",
     _pat("    COVER_GOALSIDE_M: float = 2.5"),
     "    COVER_GOALSIDE_M: float = 12.0",
     "CoverTests"),

    ("position_engine.py",
     "M4 CK38's split fires again when our own centre-back has the ball",
     _pat("            if _cst is not None and getattr(_cst, \"position\", "
          "\"\") == \"CB\":",
          "                return {}"),
     "            if False:\n                return {}",
     "SplitDoesNotFireWhenACentreBackHasTheBallTests"),

    ("position_engine.py",
     "M5 the CDM's line gap is not held",
     _pat("    SCREEN_LINE_GAP_MIN_M: float = 12.0"),
     "    SCREEN_LINE_GAP_MIN_M: float = 0.0",
     "ScreenTests"),

    ("position_engine.py",
     "M6 the cover approaches at a jog instead of a sprint",
     _pat("    COVER_APPROACH_GAIN: float = 1.1"),
     "    COVER_APPROACH_GAIN: float = 0.1",
     "CoverTests"),
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
