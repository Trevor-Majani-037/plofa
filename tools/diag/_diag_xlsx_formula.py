"""Is PLAYERS!R207 ("Months Left") a formula or a literal, and how often is the
roster workbook edited in git?

`_diag_xlsx_diff.py` found one real content change: PLAYERS!R207 44 -> 43, and
`_diag_xlsx_cell.py` showed column R is "Months Left" for Mose Gredon (CB,
contract end 2030-06-03). Nothing in the repo writes this workbook -- every
reference uses `load_workbook(read_only=True)` or `roster_loader`'s read.

Two things decide whether this matters:
  1. Is R207 a FORMULA? If so, a cached-value difference is not a data edit.
  2. What is 44 vs 43 as of today? Months from 2026-10-04 to 2030-06-03 is 44,
     so the HEAD value is the one consistent with today's date.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import openpyxl

TARGET = "PLOFA-2026-2027.xlsx"


def main():
    root = Path(__file__).resolve().parent

    # 1. formula or literal, at HEAD and in the working copy
    for label, path in (("HEAD", None), ("WORK", root / TARGET)):
        if path is None:
            blob = subprocess.run(["git", "show", f"HEAD:{TARGET}"],
                                 cwd=root, capture_output=True)
            if blob.returncode != 0:
                print(f"cannot read HEAD blob: {blob.stderr.decode()}")
                return 2
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "h.xlsx"
                p.write_bytes(blob.stdout)
                wb = openpyxl.load_workbook(p, data_only=False)
                raw = wb["PLAYERS"]["R207"].value
                wb.close()
        else:
            wb = openpyxl.load_workbook(path, data_only=False)
            raw = wb["PLAYERS"]["R207"].value
            wb.close()
        print(f"{label}: PLAYERS!R207 raw (formulas shown) = {raw!r}")

    # 2. months-left arithmetic
    end = date(2030, 6, 3)
    today = date(2026, 10, 4)
    months = (end.year - today.year) * 12 + (end.month - today.month)
    print(f"\ncontract end {end}, today {today} -> {months} whole months left")
    print("so the HEAD value (44) matches today's date; the working copy (43) "
          "is one month FURTHER along.")

    # 3. edit history of the workbook
    log = subprocess.run(
        ["git", "log", "--oneline", "--format=%h %ad %s", "--date=short",
         "-8", "--", TARGET],
        cwd=root, capture_output=True, text=True)
    print(f"\nlast 8 commits touching {TARGET}:\n{log.stdout or '  (none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
