"""NEGATIVE CONTROL for tests/test_surrogate_role_pooling.py AND
tests/test_reward_scale.py.

A guard that has never been seen to fail is not a guard.  This reintroduces
the cross-role leak -- and the pieces that stop it -- one at a time, and
requires the RIGHT test class to go red, not merely "something failed".

Twelve mutations, all in surrogate_collect.py:

  M1  the global row is lent unconditionally (the original leak)
  M2  the role-family vocabulary guard removed
  M3  the family row ignored
  M4  the family tables are never fitted
  M5  the vocabulary built from the global row (so every role "performs"
      every intent, and the leak returns through the back door)
  M6  the family table is not saved
  M7  the family table is not loaded
  M8  the family lookup stops stripping the digits from a numbered
      position (CM1 / CB2 are the live roster's names)
  M9  the role-specific FORBIDDEN rule is removed (the striker is handed
      the neutral prior for crossing, and ignorance outranks evidence)
  M10 the per-position reward scale is never applied (the v5 situation:
      the guards are 5x-20x the objective)
  M11 the scale is not saved
  M12 the scale is not loaded

Every mutant must still COMPILE.  A syntax error counted as a caught
mutation proves nothing about which assertion pins the defect.  Anchors are
escaped literals joined with an explicit ``\\r?\\n``, because this file is
CRLF and an LF anchor matches NOTHING -- a mutation that silently does
nothing is the failure this control exists to catch.

Each file is restored from an in-memory snapshot taken immediately before its
mutation and verified by sha256.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_role_pooling_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = ["tests/test_surrogate_role_pooling.py", "tests/test_reward_scale.py"]


def _pat(*lines: str) -> str:
    """An exact multi-line anchor, tolerant of the file's own line endings."""
    return r"\r?\n".join(re.escape(l) for l in lines)


MUTATIONS = [
    ("surrogate_collect.py",
     "M1 the global row is lent unconditionally (the original leak)",
     _pat("            if ik in self.family_intents.get(_fam, ()):",
          '                _grow = self.table.get("", {}).get(key, {})',
          "                if ik in _grow:",
          "                    return _grow[ik]"),
     '            _grow = self.table.get("", {}).get(key, {})\n'
     "            if ik in _grow:\n"
     "                return _grow[ik]",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M2 the role-family vocabulary guard removed",
     _pat("            if ik in self.family_intents.get(_fam, ()):"),
     "            if True:",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M3 the family row is ignored",
     _pat("            _frow = self.family_table.get(_fam, {}).get(key, {})",
          "            if ik in _frow:"),
     "            _frow = self.family_table.get(_fam, {}).get(key, {})\n"
     "            if False:",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M4 the family tables are never fitted",
     _pat("                    if pos.startswith(_FAMILY_KEY_PREFIX):"),
     "                    if False:",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M5 the vocabulary is built from the global row (leak returns by the "
     "back door)",
     _pat("            for _bucket, _intents in self.table.get(_pos, {}).items():",
          "                _used.update(_intents.keys())"),
     '            for _bucket, _intents in self.table.get("", {}).items():\n'
     "                _used.update(_intents.keys())",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M6 the family table is not saved",
     _pat('                "family_table": self.family_table,'),
     '                "family_table": {},',
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M7 the family table is not loaded",
     _pat('        s.family_table = data.get("family_table", {})'),
     '        s.family_table = {}',
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M8 the family lookup stops stripping the digits from a numbered "
     "position (CM1 / CB2 are the live roster's names)",
     _pat("        _fam = _role_family(pos)"),
     "        _fam = ROLE_FAMILY.get(pos)",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M9 the role-specific FORBIDDEN rule is removed (the striker is handed "
     "the neutral prior for crossing, and ignorance outranks evidence)",
     _pat("        _allowed = ROLE_ONLY_INTENTS.get(ik)",
          "        if _allowed is not None and (_fam or \"\") not in _allowed:",
          "            return FORBIDDEN_VALUE"),
     "        _allowed = ROLE_ONLY_INTENTS.get(ik)\n"
     "        if False:\n"
     "            return FORBIDDEN_VALUE",
     "RoleScopedLookupTests"),

    ("surrogate_collect.py",
     "M10 the per-position reward scale is never applied (the v5 situation: "
     "the guards are 5x-20x the objective)",
     _pat("        _scale = self.scales.get(position, 1.0) or 1.0",
          "        return v / _scale"),
     "        _scale = self.scales.get(position, 1.0) or 1.0\\n"
     "        return v",
     "ScaleApplicationTests"),

    ("surrogate_collect.py",
     "M11 the scale is not saved",
     _pat('                "scales": self.scales,'),
     '                "scales": {},',
     "PersistenceTests"),

    ("surrogate_collect.py",
     "M12 the scale is not loaded",
     _pat('        s.scales = {k: float(v) for k, v in '
          'data.get("scales", {}).items()}'),
     "        s.scales = {}",
     "PersistenceTests"),
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
