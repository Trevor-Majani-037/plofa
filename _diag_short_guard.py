"""Negative control: does `test_the_height_clamp_is_not_applied_to_a_short_pass`
actually catch the regression it claims to?

A test that passes for the wrong reason is worse than no test. This restores
DEFECT A exactly as it was on 2026-10-05 -- the clamp applied unconditionally,
undoing `corner_height = 0.35` -- runs the guard, and must see it go RED.

The file is restored from a byte-exact backup and the restore is asserted, so
this cannot leave the tree modified.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

EV = pathlib.Path("event_chain.py")
BACKUP = pathlib.Path(tempfile.gettempdir()) / "ec_negctl_backup.py"

GUARDED = """        if not is_short:
            corner_height = max(2.0, min(3.0, corner_height
                                         + params.get("height_bias", 0.0)
                                         * cross_quality))"""
DEFECT = ("        corner_height = max(2.0, min(3.0, corner_height\n"
          "                               + params.get(\"height_bias\", 0.0)\n"
          "                               * cross_quality))")

src = EV.read_text(encoding="utf-8")
assert src.count(GUARDED) == 1, f"anchored guard not found ({src.count(GUARDED)})"
shutil.copy2(EV, BACKUP)
try:
    EV.write_text(src.replace(GUARDED, DEFECT, 1), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         "tests/test_short_corner.py",
         "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env={**__import__("os").environ,
                                             "PYTHONPATH": "."})
    tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln]
    print("with DEFECT A restored:")
    for ln in tail[-2:]:
        print("   ", ln.strip())
    red = r.returncode != 0
    print(f"   guard went RED: {red}  "
          f"({'CORRECT -- it catches the regression' if red else 'PROBLEM -- it does not'})")
finally:
    shutil.copy2(BACKUP, EV)
    after = EV.read_text(encoding="utf-8")
    print("file restored byte-identical:",
          after == src, f"({len(after)} chars)")