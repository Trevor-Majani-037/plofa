"""Bisect the two tests/test_attacking_matrix.py failures against HEAD.

Swaps event_chain.py + attacking_matrix.py to their HEAD bytes, runs the two
failing nodes, restores the working copies BYTE-IDENTICALLY (hash asserted),
and prints both verdicts. Written because "pre-existing" must be measured, not
assumed -- and PowerShell `>` emits UTF-16, which GitHub's own rulebook says
kills the suite.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["event_chain.py", "attacking_matrix.py"]
NODES = [
    "tests/test_attacking_matrix.py::test_possession_chain_policy_target_beats_matrix_target",
    "tests/test_attacking_matrix.py::test_shot_origin_binds_to_shooter_real_position",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    saved = {f: (ROOT / f).read_bytes() for f in FILES}
    before = {f: hashlib.sha256(b).hexdigest() for f, b in saved.items()}
    print("working-copy digests:", before)
    try:
        for f in FILES:
            head = subprocess.run(
                ["git", "show", f"HEAD:{f}"],
                cwd=ROOT, capture_output=True, check=True,
            ).stdout
            (ROOT / f).write_bytes(head)
            print(f"  swapped {f} -> HEAD ({len(head)} bytes)")
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", *NODES, "-q", "--no-header"],
            cwd=ROOT,
        ).returncode
        print(f"\nHEAD verdict: rc={rc}  (0 = both PASS at HEAD, so the failure is NEW)")
    finally:
        for f, data in saved.items():
            (ROOT / f).write_bytes(data)
        after = {f: digest(ROOT / f) for f in FILES}
        print("\nrestored digests:", after)
        if after != before:
            print("!! RESTORE MISMATCH -- inspect manually")
            return 2
        print("restore byte-identical OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
