"""Is the modified roster workbook a CONTENT change or just a zip repack?

Git reports `Bin 128571 -> 128591 bytes` for PLOFA-2026-2027.xlsx and nothing in
the repo writes that file (`roster_loader.py` only reads it; the exporter writes
its own workbook). A 20-byte delta on an .xlsx is usually nothing more than the
archive being rewritten with different compression, but "usually" is not a
measurement, and this is the real roster -- so compare the CELLS.

Reads the HEAD blob via `git show` into a temp file and both workbooks with
openpyxl, then reports any differing cell. Exits non-zero on a real difference.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import openpyxl

TARGET = "PLOFA-2026-2027.xlsx"


def cells(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    out = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.value is not None:
                    out[(ws.title, c.coordinate)] = c.value
    wb.close()
    return out


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        head_path = Path(td) / "head.xlsx"
        blob = subprocess.run(
            ["git", "show", f"HEAD:{TARGET}"],
            cwd=root, capture_output=True,
        )
        if blob.returncode != 0:
            print(f"could not read HEAD:{TARGET}: {blob.stderr.decode()}")
            return 2
        head_path.write_bytes(blob.stdout)
        head, work = cells(head_path), cells(root / TARGET)

    keys = set(head) | set(work)
    diffs = [(k, head.get(k, "<absent>"), work.get(k, "<absent>"))
             for k in sorted(keys) if head.get(k, "<absent>") != work.get(k, "<absent>")]

    print(f"{TARGET}: {len(head)} non-empty cells at HEAD, {len(work)} in the "
          f"working copy, {len(diffs)} differing")
    for (sheet, coord), a, b in diffs[:40]:
        print(f"  {sheet}!{coord}: {a!r} -> {b!r}")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")
    if not diffs:
        print("\nCONTENT IDENTICAL -> the size delta is an archive repack, "
              "not a data change. Safe to leave alone.")
        return 0
    print("\nREAL CONTENT DIFF -> do not discard; find what wrote it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
