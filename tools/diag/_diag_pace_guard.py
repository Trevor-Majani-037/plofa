"""NEGATIVE CONTROL for tests/test_role_pace_calibration.py.

A guard that has never been seen to fail is not a guard. This reintroduces the
defects the test file exists for, one at a time, and requires the RIGHT test to
go red -- not merely "something failed".

Three mutations:
  M1  restore the centre-back row to the pre-fix 1.5  -> the outlier assertions
  M2  leave the centre-back correct but put `DMC` back at 1.5 -> the floor
      assertion (a dead alias inheriting a walking pace for a midfielder)
  M3  leave `CB` correct but desynchronise the `LCB` alias -> the alias test

Every mutant must still COMPILE and still be IMPORTABLE. A syntax error counted
as a caught mutation proves nothing about which assertion pins the defect --
that mistake was made once already in `_diag_distance_guard.py` (M2 there
"passed" as a collection error).

`match_engine.py` is restored from an in-memory snapshot taken immediately
before each mutation, and the restore is verified by sha256. A backup is only
valid for the state it was taken from.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_pace_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# tools/diag/<this file> -> repo root
REPO = Path(__file__).resolve().parent.parent.parent
TARGET = REPO / "match_engine.py"
TEST = "tests/test_role_pace_calibration.py"

MUTATIONS = [
    ("M1 centre-back row reverted to the pre-fix 1.5",
     r'"CB": 1\.9, "DC": 1\.9, "LCB": 1\.9, "RCB": 1\.9',
     '"CB": 1.5, "DC": 1.5, "LCB": 1.5, "RCB": 1.5',
     "RolePaceOutlierTests"),
    ("M2 dead `DMC` alias back at the walking pace",
     r'"DMC": 2\.0',
     '"DMC": 1.5',
     "RolePaceOutlierTests"),
    ("M3 `LCB` alias desynchronised from `CB`",
     r'"LCB": 1\.9',
     '"LCB": 1.7',
     "RolePaceOutlierTests"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_tests() -> tuple:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True,
        env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO),
             "PATH": os.environ.get("PATH", ""),
             "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    original = TARGET.read_bytes()
    original_hash = sha(original)
    # Match on the file's OWN line endings. Anchors written with \n against a
    # CRLF file match nothing and every mutation silently becomes a no-op --
    # that exact failure happened in `_diag_deference_guard.py`.
    text = original.decode("utf-8")

    rc, out = run_tests()
    if rc != 0:
        print("BASELINE IS NOT GREEN -- aborting, nothing measured.")
        print(out[-2000:])
        return 1
    print("baseline: green")
    print()

    failures = 0
    try:
        for label, pattern, repl, expected_class in MUTATIONS:
            snapshot = TARGET.read_bytes()
            body = snapshot.decode("utf-8")
            hits = len(re.findall(pattern, body))
            if hits != 1:
                print(f"  {label}\n    ANCHOR MATCHED {hits} TIMES -- the "
                      f"mutation is a no-op, aborting this arm.")
                failures += 1
                TARGET.write_bytes(snapshot)
                continue

            mutant = re.sub(pattern, repl, body, count=1)
            try:
                compile(mutant, str(TARGET), "exec")
            except SyntaxError as exc:
                print(f"  {label}\n    MUTANT DOES NOT COMPILE ({exc}) -- "
                      f"counting as a MISS, not a catch.")
                failures += 1
                TARGET.write_bytes(snapshot)
                continue

            TARGET.write_bytes(mutant.encode("utf-8"))
            rc, out = run_tests()
            caught = rc != 0 and expected_class in out
            if caught:
                # Name the assertion that actually fired, not just the class.
                names = re.findall(r"FAILED (\S+?)(?:\s+-|$)", out, re.M)
                short = [n.split("::", 1)[1] if "::" in n else n for n in names]
                print(f"  {label}\n    CAUGHT: "
                      f"{', '.join(dict.fromkeys(short)) or '?'}")
            else:
                print(f"  {label}\n    MISS (rc={rc}); expected "
                      f"{expected_class} to fail.")
                print("    " + (out.strip().splitlines() or [""])[-1])
                failures += 1

            TARGET.write_bytes(snapshot)
            assert sha(TARGET.read_bytes()) == sha(snapshot), "RESTORE FAILED"
    finally:
        TARGET.write_bytes(original)

    assert sha(TARGET.read_bytes()) == original_hash, "FINAL RESTORE FAILED"
    rc, out = run_tests()
    print()
    print(f"restored byte-identically (sha256 {original_hash[:16]})")
    print(f"baseline after restore: {'green' if rc == 0 else 'RED'}")
    print()
    print("ALL CONTROLS PASSED" if not failures else f"{failures} MISS(ES)")
    return 0 if (failures == 0 and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
