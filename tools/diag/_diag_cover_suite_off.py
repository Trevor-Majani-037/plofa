"""Run the whole shape suite with the cover/screen feature OFF.

The bisect showed `test_fullback_can_tuck_inside` and
`test_striker_gets_in_behind_not_offside` fail with the feature off as well,
so they are independent of this work.  The touchline failure did not reproduce
in isolation (it needs the wider suite), so it is checked HERE, in the same
suite order, with the feature off.

Usage:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_cover_suite_off.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PE = REPO / "position_engine.py"
TESTS = ["tests/test_positional_play.py", "tests/test_small_game.py",
         "tests/test_touchline_wide.py"]

EDITS = [
    ("    COVER_ENABLED: bool = True", "    COVER_ENABLED: bool = False"),
    ("    SCREEN_ENABLED: bool = True", "    SCREEN_ENABLED: bool = False"),
    ("    BACKLINE_SPLIT_SKIPS_CB_CARRIER: bool = True",
     "    BACKLINE_SPLIT_SKIPS_CB_CARRIER: bool = False"),
]


def main() -> int:
    snap = PE.read_bytes()
    body = snap.decode("utf-8")
    try:
        for old, new in EDITS:
            if body.count(old) != 1:
                print(f"ANCHOR MISSING: {old!r}")
                return 1
            body = body.replace(old, new, 1)
        PE.write_bytes(body.encode("utf-8"))
        p = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=str(REPO), capture_output=True, text=True,
            env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO),
                 "PATH": os.environ.get("PATH", ""),
                 "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
        )
        out = (p.stdout or "") + (p.stderr or "")
        print("\n".join(out.strip().splitlines()[-8:]))
        print("\nFAILING TESTS:")
        for f in sorted(set(re.findall(r"FAILED (\S+?)(?:\s+-|$)", out))):
            print(f"    {f}")
    finally:
        PE.write_bytes(snap)
        assert hashlib.sha256(PE.read_bytes()).hexdigest() == \
            hashlib.sha256(snap).hexdigest(), "RESTORE FAILED"
        print("position_engine.py restored byte-identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
