"""Put the flattened plofa_output PNGs back where they belong.

What happened: a cleanup command used
    Get-ChildItem -File -Include "*.txt","*.png" -Recurse -Depth 0
and PowerShell's `-Include` combined with `-Recurse` ignored `-Depth 0`, so it
matched RECURSIVELY. `Move-Item ... "artifacts/$($f.Name)"` then flattened every
path into one directory. `plofa_output/` lost all 258+ of its PNGs (the
`player_dashboards/` subfolders included); nothing was deleted.

258 PNGs were restored exactly from HEAD~1. This puts back the rest -- the
NEWER on-disk versions, which git never had.

Matching is done on the normalised match name, because the folder and the file
disagree: folder `Avada_Zenith_vs_Claw_MD03`, file `Avada_Zenith_vs_Claw_MD3_*`
(MD03 -> MD3). A plain prefix match therefore matches NOTHING, which is why the
first attempt returned 0.

Dashboard PNGs carry no match prefix (`Kwame_Asante_dashboard.png`), so they
are located from the git tree first, then by the player names in each match's
`*_players.csv`. Anything still unresolved is LEFT IN artifacts/ and reported,
never dumped somewhere plausible-looking.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

# This script lives in tools/diag/, so its own directory is NOT the repo root.
# Two parents up is the repo root; using __file__'s directory silently pointed
# every path at tools/diag/ and failed with FileNotFoundError on the first read.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "plofa_output"
SRC = ROOT / "artifacts"

SUFFIXES = (
    "_ball_motion", "_momentum", "_pass_network", "_player_heatmap",
    "_pressure_map", "_shot_map", "_xg_timeline", "_summary", "_shot_map",
)


def norm(s: str) -> str:
    """MD03 -> MD3, so folder and filename agree."""
    return re.sub(r"MD0(\d)", r"MD\1", s)


def main() -> None:
    # ---- folder -> normalised name -------------------------------------
    folders = {}
    for d in OUT.iterdir():
        if d.is_dir():
            folders.setdefault(norm(d.name), d)

    # ---- dashboard owner, from the git tree ---------------------------
    dash_owner: dict[str, str] = {}
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD~1", "--", "plofa_output"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.splitlines()
    for line in tree:
        if line.endswith(".png") and "player_dashboards" in line:
            parent = line.split("/")[-2]
            dash_owner.setdefault(Path(line).name, parent)

    # ---- dashboard owner, from each match's players.csv ----------------
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        for csvf in d.glob("*_players.csv"):
            try:
                with open(csvf, encoding="utf-8", errors="replace") as fh:
                    for row in csv.DictReader(fh):
                        for key in ("Player", "player", "Name", "name"):
                            nm = row.get(key)
                            if nm:
                                dash_owner.setdefault(f"{nm.strip()}_dashboard.png",
                                                      d.name)
                                dash_owner.setdefault(
                                    f"{nm.strip().split()[0]}_dashboard.png",
                                                    d.name)
            except Exception:
                continue

    moved = skipped = 0
    unresolved: list[str] = []
    for png in sorted(SRC.glob("*.png")):
        stem = png.stem
        dest_dir = None
        for suf in SUFFIXES:
            if stem.endswith(suf):
                dest_dir = folders.get(norm(stem[: -len(suf)]))
                break
        if dest_dir is None and stem.endswith("_dashboard"):
            owner = dash_owner.get(png.name)
            if owner:
                cand = OUT / owner / "player_dashboards"
                dest_dir = cand if cand.is_dir() else (OUT / owner)
        if dest_dir is None:
            unresolved.append(png.name)
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / png.name
        try:
            os.replace(png, target)          # newer version wins
            moved += 1
        except OSError:
            skipped += 1

    print(f"  restored into plofa_output : {moved}")
    print(f"  could not write            : {skipped}")
    print(f"  LEFT IN artifacts/          : {len(unresolved)}")
    for n in unresolved[:10]:
        print(f"      {n}")
    total = len(list(OUT.rglob("*.png")))
    print(f"  PNGs now under plofa_output: {total}")
    print(f"  PNGs still in artifacts/    : {len(list(SRC.glob('*.png')))}")


if __name__ == "__main__":
    main()