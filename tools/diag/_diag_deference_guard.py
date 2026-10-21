"""Negative control for tests/test_policy_deference.py (2026-10-19).

A test that has never been seen to fail is not a test -- the project's own
rule, learned five times over. This mutates `event_chain.py` back into each
defect the suite claims to pin, runs the suite, and requires it to go RED for
the RIGHT reason (the mutant must still IMPORT, and the specific node must be
among the failures).

    M1  the provenance guard is dropped  -> a phase order / role rule becomes
                                           outvotable by a preference
    M2  the keeper-release guard is dropped
    M3  the reference is the global max instead of the chosen option's own score
    M4  the rule consumes the global RNG
    M5  `target_authority` is not stamped on the pass

Files are restored BYTE-IDENTICALLY and the restore is hash-asserted.

Run:  .venv\\Scripts\\python.exe tools\\diag\\_diag_deference_guard.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "event_chain.py"
SUITE = "tests/test_policy_deference.py"

MUTATIONS = [
    (
        "M1 provenance guard dropped",
        '        if forced_source != "matrix" or matrix_decision is None:\n'
        '            return False, None\n',
        '        if matrix_decision is None:\n'
        '            return False, None\n',
        "test_only_the_matrix_is_deferred_to",
    ),
    (
        "M2 keeper-release guard dropped",
        '        if str(getattr(matrix_decision, "reason", "")).startswith("back_to_keeper"):\n'
        '            return False, None\n',
        '        if False:\n'
        '            return False, None\n',
        "test_keeper_release_is_safety_not_preference",
    ),
    (
        "M3 reference is the global max",
        '        mtx_val = float(getattr(matrix_decision, "value", 0.0) or 0.0)\n',
        '        mtx_val = float(max(scores.values()) or 0.0)\n',
        "test_the_margin_is_relative_to_the_chosen_option",
    ),
    (
        "M4 the rule draws from the global RNG",
        '        margin = float(cls.POLICY_TARGET_DEFERENCE_MARGIN)\n',
        '        _ = random.random()\n'
        '        margin = float(cls.POLICY_TARGET_DEFERENCE_MARGIN)\n',
        "test_the_rule_consumes_no_random_state",
    ),
    (
        "M5 target_authority not stamped",
        '                        "target_authority": receiver_authority or "unknown",\n',
        '',
        "test_deference_off_keeps_the_matrix_and_still_reports_authority",
    ),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_suite() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def imports_clean() -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", "import event_chain"],
        cwd=ROOT, capture_output=True,
    )
    return proc.returncode == 0


def _apply(text: str, old: str, new: str):
    """Replace `old` with `new`, tolerating the file's line endings.

    This file is CRLF on this checkout and the anchors above are written with
    LF, so a naive `str.replace` matches NOTHING and every mutation silently
    becomes a no-op -- which is the exact failure this control exists to catch,
    reproduced in the control itself. Try the file's own convention first.
    """
    for conv in (lambda s: s.replace("\n", "\r\n"), lambda s: s):
        o, n = conv(old), conv(new)
        if text.count(o) == 1:
            return text.replace(o, n, 1), 1
    return text, 0


def main() -> int:
    original = TARGET.read_bytes()
    before = digest(original)
    print(f"baseline event_chain.py sha256={before} ({len(original)} bytes)")

    rc, out = run_suite()
    print(f"baseline suite: rc={rc}")
    if rc != 0:
        print(out[-3000:])
        print("BASELINE IS NOT GREEN -- refusing to run the control")
        return 2

    failures = 0
    try:
        for name, old, new, node in MUTATIONS:
            text = original.decode("utf-8")
            mutated_text, occurrences = _apply(text, old, new)
            if occurrences != 1:
                print(f"\n{name}: ANCHOR MATCHED {occurrences} TIMES -- mutation "
                      f"would be a no-op or ambiguous. FAIL.")
                failures += 1
                continue
            mutated = mutated_text.encode("utf-8")
            TARGET.write_bytes(mutated)

            if not imports_clean():
                print(f"\n{name}: mutant does not import -- a syntax error is "
                      f"not a caught defect. FAIL.")
                failures += 1
                continue

            rc, out = run_suite()
            caught = rc != 0 and node in out
            print(f"\n{name}: rc={rc}  target-node-failed={node in out}  "
                  f"-> {'CAUGHT' if caught else 'MISS'}")
            if not caught:
                failures += 1
                tail = "\n".join(out.strip().splitlines()[-12:])
                print(tail)
            TARGET.write_bytes(original)
            if digest(TARGET.read_bytes()) != before:
                print("!! RESTORE MISMATCH -- aborting immediately")
                return 3
    finally:
        TARGET.write_bytes(original)

    restored = TARGET.read_bytes()
    print(f"\nrestored sha256={digest(restored)} ({len(restored)} bytes)")
    if digest(restored) != before:
        print("!! FINAL RESTORE MISMATCH")
        return 3

    rc, _ = run_suite()
    print(f"restored suite: rc={rc}")
    if rc != 0:
        print("!! SUITE NOT GREEN AFTER RESTORE")
        return 3

    print(f"\n{len(MUTATIONS) - failures}/{len(MUTATIONS)} mutations CAUGHT")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
