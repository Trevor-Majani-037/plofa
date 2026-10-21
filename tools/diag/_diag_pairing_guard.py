"""NEGATIVE CONTROL for tests/test_surrogate_pairing.py.

A guard that has never been seen to fail is not a guard.  This reintroduces
the defects that file exists for, one at a time, and requires the RIGHT test
class to go red -- not merely "something failed".

Eight mutations, all in surrogate_collect.py:

  M1  consequences become claimable again -> the double-charge test
  M2  the BALL_RECEIPT becomes claimable  -> the receipt test
  M3  an event may be claimed twice        -> the claim-once test
  M4  the chain buffer is not cleared per match -> the wiring test
  M5  the dispatcher is not restored        -> the wiring test
  M6  an unmatched decision gets a fabricated 0.5 instead of NEUTRAL
  M7  the team's own attacking direction is ignored
  M8  the join claims another player's event

Every mutant must still COMPILE.  A syntax error counted as a caught mutation
proves nothing about which assertion pins the defect -- `_diag_distance_guard.py`
M2 and `_diag_deference_guard.py` both made that mistake, the latter by writing
LF anchors against a CRLF file so every mutation silently became a no-op.  So
anchors here are escaped literals joined with an explicit ``\\r?\\n``.

Each file is restored from an in-memory snapshot taken immediately before its
mutation and verified by sha256.  A backup is only valid for the state it was
taken from.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_pairing_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = ["tests/test_surrogate_pairing.py"]
TARGET = "surrogate_collect.py"


def _pat(*lines: str) -> str:
    """An exact multi-line anchor, tolerant of the file's own line endings."""
    return r"\r?\n".join(re.escape(l) for l in lines)


# (file, label, pattern, replacement, expected failing test CLASS)
MUTATIONS = [
    ("surrogate_collect.py",
     "M1 consequences become claimable again (the double-charge)",
     _pat("                    if event_type_name(e) in _UNCLAIMABLE_EVENTS:",
          "                        continue"),
     "                    if event_type_name(e) in _NON_ACTION_EVENTS:\n"
     "                        continue",
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M2 the BALL_RECEIPT becomes claimable",
     _pat('_NON_ACTION_EVENTS = frozenset({"BALL_RECEIPT", "PRESS", '
          '"PRESS_RESISTED"})'),
     '_NON_ACTION_EVENTS = frozenset({"PRESS", "PRESS_RESISTED"})',
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M3 an event may be claimed twice",
     _pat("                    if idx in used:",
          "                        continue"),
     "                    if False:\n                        continue",
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M4 the chain-call buffer is not cleared between matches",
     _pat("            # Each match is its own set of chain calls.  Without this "
          "clear,",
          "            # match 2 would re-pair match 1's calls as well.",
          "            _CHAIN_CALLS.clear()"),
     "            # Each match is its own set of chain calls.  Without this "
     "clear,\n"
     "            # match 2 would re-pair match 1's calls as well.",
     "ProductionWiringTests"),

    ("surrogate_collect.py",
     "M5 the chain entry points are not restored",
     _pat("    for name, raw in _SAVED_DISPATCHER.items():",
          "        setattr(ChainDispatcher, name, raw)"),
     "    for name, raw in {}:\n        setattr(ChainDispatcher, name, raw)",
     "ProductionWiringTests"),

    ("surrogate_collect.py",
     "M6 an unmatched decision gets a fabricated 0.5 instead of NEUTRAL",
     _pat('                pairs.append((rec.sensors, rec.intent, 0.0, '
          '"NONE", rec.position))'),
     '                pairs.append((rec.sensors, rec.intent, 0.5, '
     '"NONE", rec.position))',
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M7 the team's own attacking direction is ignored",
     _pat('                pairs.append((rec.sensors, rec.intent, 0.0, '
          '"NONE", rec.position))',
          "                continue",
          "            ar = True if not attacks_right_by_team else bool(",
          '                attacks_right_by_team.get(getattr(ev, "team", None), '
          "True))"),
     '                pairs.append((rec.sensors, rec.intent, 0.0, '
     '"NONE", rec.position))\n'
     "                continue\n"
     "            ar = True if True else bool(\n"
     '                attacks_right_by_team.get(getattr(ev, "team", None), '
     "True))",
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M8 the join claims another player's event",
     _pat('                    if getattr(e, "player", None) != rec.player:',
          "                        continue"),
     "                    if False:\n                        continue",
     "CausalJoinTests"),

    ("surrogate_collect.py",
     "M9 a SHOOT decision is no longer answered by the shot it caused (the "
     "policy is credited with a ZERO-LENGTH marker carry instead, so every "
     "shoot decision is worth exactly 0.0)",
     _pat("            if (rec.intent is PlayerIntent.SHOOT and timeline is "
          "not None):"),
     "            if False:",
     "ShootDecisionIsAnsweredByItsShotTests"),
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
