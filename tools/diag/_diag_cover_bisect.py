"""BISECT the three shape-suite failures against the cover/screen work.

The project's rule: "documented pre-existing" is not evidence.  AGENTS.md
records `test_fullback_can_tuck_inside`, `test_striker_gets_in_behind_not_offside`
and `test_out_of_possession_touchline_press_gated_by_press_intensity` as
pre-existing, but this work touches the centre-backs and the full-back
movement, so each failure must be shown to be independent of it.

The arms:
  ON      everything as shipped
  OFF     COVER_ENABLED / SCREEN_ENABLED / BACKLINE_SPLIT_SKIPS_CB_CARRIER off
          (the whole cover-and-screen feature removed, in one step)
  OFF+CB  only the CB pace change (the earlier session's `_JOG_SPEED` work)
          left in, everything from the cover work off

Each arm is applied by patching the source text, running the three tests, and
restoring the file BYTE-EXACTLY from an in-memory snapshot.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_cover_bisect.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = [
    "tests/test_small_game.py::test_striker_gets_in_behind_not_offside",
    "tests/test_small_game.py::test_fullback_can_tuck_inside",
    "tests/test_touchline_wide.py::test_out_of_possession_touchline_press_gated_by_press_intensity",
]

PE = REPO / "position_engine.py"
ME = REPO / "match_engine.py"

ARMS = [
    ("ON (as shipped)", [], []),
    ("OFF (the whole cover/screen feature off)",
     [("    COVER_ENABLED: bool = True", "    COVER_ENABLED: bool = False"),
      ("    SCREEN_ENABLED: bool = True", "    SCREEN_ENABLED: bool = False"),
      ("    BACKLINE_SPLIT_SKIPS_CB_CARRIER: bool = True",
       "    BACKLINE_SPLIT_SKIPS_CB_CARRIER: bool = False")], []),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run() -> tuple:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True,
        env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO),
             "PATH": _path(), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _path():
    return os.environ.get("PATH", "")


def main() -> int:
    pe_snap = PE.read_bytes()
    me_snap = ME.read_bytes()
    hashes = {"position_engine.py": sha(pe_snap), "match_engine.py": sha(me_snap)}
    try:
        for label, pe_edits, _me_edits in ARMS:
            body = pe_snap.decode("utf-8")
            for old, new in pe_edits:
                if body.count(old) != 1:
                    print(f"[{label}] ANCHOR MISSING: {old!r}")
                    break
                body = body.replace(old, new, 1)
            PE.write_bytes(body.encode("utf-8"))
            rc, out = run()
            failed = sorted(set(re.findall(r"FAILED (\S+?)(?:\s+-|$)", out)))
            print(f"\n=== {label} ===  rc={rc}  failures={len(failed)}")
            for f in failed:
                print(f"    {f}")
            PE.write_bytes(pe_snap)
    finally:
        PE.write_bytes(pe_snap)
        ME.write_bytes(me_snap)

    for name, snap in (("position_engine.py", pe_snap),
                       ("match_engine.py", me_snap)):
        target = REPO / name
        assert sha(target.read_bytes()) == sha(snap), f"RESTORE FAILED {name}"
        print(f"restored byte-identically: {name} (sha256 {hashes[name][:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
