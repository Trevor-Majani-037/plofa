"""
PLOFA TEAM SHEETS — 2026/27
===========================
Rebuilds the "SQUAD DEPTH" team sheets (previously a 24/25 deck of JPGs) for the
live 2026/27 season, straight from the production roster workbook.

Data source of truth
--------------------
`PLOFA-2026-2027.xlsx` via :class:`roster_loader.RosterLoader` — the same object
the match engine uses. Formation comes from ``build_matchday_squad()``, so a
sheet shows the shape the club actually plays, not a guess.

Placement rule
--------------
Every squad player is placed exactly once:

  1. natural position  (engine_pos == slot)
  2. secondary position (engine_other_pos == slot)
  3. positional cover   (COVER: CB→CDM, CDM→CM, CAM→ST, ...)
  4. any non-GK slot with spare capacity

Within a slot, players are ordered First-Team → Second-Team → Third-Team and then
by market value, so the top name on a plate is the starter. A plate holds at most
three names — the same depth the 24/25 sheets showed.

Output
------
A native PowerPoint deck (editable: every crest, plate and name is a real shape)
plus PNG exports of every slide.

Usage
-----
    python tools/team_sheets_2627.py --out "D:\\TOLAND FOOTBALL FEDERATION\\Team Sheets 26-27 ⚽"

Add ``--no-export`` to skip the PowerPoint COM step (writes .pptx only).
Add ``--json PATH`` to dump the resolved sheet data for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roster_loader import FORMATION_SLOTS, PlayerRecord, RosterLoader  # noqa: E402

# ─────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────

LOGO_DIR = next(Path("D:/TOLAND FOOTBALL FEDERATION").glob("Logos*"))
LEAGUE_CREST = LOGO_DIR / "plofaleaguenewlogo.jpg"

#: club → crest file. Chosen from the federation's crest pack; the 24/25 deck used
#: the same art. Avada Zenith is filed under its short name "Adveece" in the pack
#: (`adveece.jpeg`, the gold "A" shield).
CRESTS: Dict[str, str] = {
    "Uditon": "uditon1.jpeg",
    "Claw": "claw1.jpeg",
    "Pearls": "pearls2.jpeg",
    "Natrican": "natrican.jpeg",
    "Lige-8": "ligue8 1.jpeg",
    "Triumpher": "triumpherfc.jpeg",
    "Play City": "playcity.jpeg",
    "Red Wolves": "rewu.jpeg",
    "Telbey": "telbey1.jpeg",
    "Justice": "justice.jpeg",
    "Tryox City": "tryox.jpeg",
    "Oxton": "oxton.jpeg",
    "Trendboys": "trendboys.jpeg",
    "Club Chovers": "clubchovers.jpeg",
    "Seafcea": "seafcea.jpeg",
    "Ganester": "ganester.jpeg",
    "Rodice": "rodice.jpeg",
    "Avada Zenith": "adveece.jpeg",
}

#: Accent per club. Home colour from auto_run_match.TEAM_CATALOG where it has
#: enough contrast on white; otherwise a crest-sampled darker variant. The two
#: malformed catalog values (Lige-8 "#061FBEE6") are corrected here.
ACCENTS: Dict[str, str] = {
    "Uditon": "01C271",
    "Claw": "E90000",
    "Pearls": "0E8C8C",
    "Natrican": "E85D04",
    "Lige-8": "1E88C7",
    "Triumpher": "14532D",
    "Play City": "E08A00",
    "Red Wolves": "CC0000",
    "Telbey": "0E7A3C",
    "Justice": "A8862B",
    "Tryox City": "0E7C6B",
    "Oxton": "A63A1E",
    "Trendboys": "1B2A4A",
    "Club Chovers": "16294A",
    "Seafcea": "0097A7",
    # home kit is white in TEAM_CATALOG, so the accent is sampled from the crest
    # (the gold "A" shield): a bronze that stays legible under white text.
    "Avada Zenith": "7A5C1C",
    "Ganester": "C8102E",
    "Rodice": "0B3E0C",
}

# ─────────────────────────────────────────────
# PITCH GEOMETRY
# ─────────────────────────────────────────────

#: formation → [(x, y)] in 0..1 pitch coordinates, in FORMATION_SLOTS order.
#: y = 0 is the opponent goal (attacking up), y = 1 is own goal.
FORMATION_XY: Dict[str, List[Tuple[float, float]]] = {
    "4-3-3": [
        (0.50, 0.84),                                   # GK
        (0.37, 0.74), (0.63, 0.74),                     # CB CB
        (0.11, 0.72), (0.89, 0.72),                     # LB RB
        (0.50, 0.57),                                   # CDM
        (0.30, 0.49), (0.70, 0.49),                     # CM CM
        (0.13, 0.24), (0.87, 0.24), (0.50, 0.15),       # LW RW ST
    ],
    "4-2-3-1": [
        (0.50, 0.84),
        (0.37, 0.74), (0.63, 0.74),
        (0.11, 0.72), (0.89, 0.72),
        (0.36, 0.55), (0.64, 0.55),                     # CDM CM
        (0.50, 0.42),                                   # CAM
        (0.13, 0.22), (0.87, 0.22), (0.50, 0.14),       # LW RW ST
    ],
    "4-4-2": [
        (0.50, 0.84),
        (0.37, 0.74), (0.63, 0.74),
        (0.11, 0.72), (0.89, 0.72),
        (0.38, 0.52), (0.62, 0.52),                     # CM CM
        (0.11, 0.47), (0.89, 0.47),                     # LW RW
        (0.38, 0.17), (0.62, 0.17),                     # ST ST
    ],
    "3-4-3": [
        (0.50, 0.84),
        (0.29, 0.70), (0.50, 0.72), (0.71, 0.70),       # CB CB CB
        (0.10, 0.68), (0.90, 0.68),                     # LB RB
        (0.37, 0.50), (0.63, 0.50),                     # CM CM
        (0.14, 0.26), (0.86, 0.26), (0.50, 0.15),       # LW RW ST
    ],
    "3-4-2-1": [
        (0.50, 0.84),
        (0.29, 0.70), (0.50, 0.72), (0.71, 0.70),
        (0.10, 0.66), (0.90, 0.66),
        (0.38, 0.57), (0.62, 0.57),
        (0.18, 0.32), (0.82, 0.32), (0.50, 0.14),
    ],
    "3-5-2": [
        (0.50, 0.84),
        (0.29, 0.70), (0.50, 0.72), (0.71, 0.70),
        (0.10, 0.66), (0.90, 0.66),
        (0.50, 0.60),                                   # CDM
        (0.29, 0.50), (0.71, 0.50),                     # CM CM
        (0.38, 0.17), (0.62, 0.17),                     # ST ST
    ],
    "5-3-2": [
        (0.50, 0.84),
        (0.29, 0.72), (0.50, 0.74), (0.71, 0.72),
        (0.10, 0.70), (0.90, 0.70),
        (0.50, 0.56),                                   # CDM
        (0.30, 0.47), (0.70, 0.47),
        (0.38, 0.17), (0.62, 0.17),
    ],
    "5-4-1": [
        (0.50, 0.84),
        (0.29, 0.72), (0.50, 0.74), (0.71, 0.72),
        (0.10, 0.70), (0.90, 0.70),
        (0.37, 0.54), (0.63, 0.54),
        (0.12, 0.47), (0.88, 0.47),
        (0.50, 0.15),
    ],
    "4-5-1": [
        (0.50, 0.84),
        (0.37, 0.74), (0.63, 0.74),
        (0.11, 0.72), (0.89, 0.72),
        (0.22, 0.54), (0.50, 0.58), (0.78, 0.54), (0.50, 0.44),
        (0.50, 0.30),                                   # CAM
        (0.50, 0.14),                                   # ST
    ],
}

#: A player at home in one position can cover these, in order of preference.
COVER: Dict[str, Tuple[str, ...]] = {
    "GK": (),
    "CB": ("CDM",),
    "LB": ("CB", "LW", "CDM"),
    "RB": ("CB", "RW", "CDM"),
    "CDM": ("CM", "CB"),
    "CM": ("CDM", "CAM"),
    "CAM": ("CM", "ST", "CF", "LW", "RW"),
    "LW": ("ST", "CAM", "CM"),
    "RW": ("ST", "CAM", "CM"),
    "ST": ("CF", "LW", "RW", "CAM"),
    "CF": ("ST", "CAM", "LW", "RW"),
}

ROLE_RANK = {"First-Team": 0, "Second-Team": 1, "Third-Team": 2}
MAX_PER_SLOT = 3

# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────


@dataclass(eq=False)
class Slot:
    pos: str
    x: float
    y: float
    players: List[PlayerRecord] = field(default_factory=list)


@dataclass
class TeamSheet:
    club: str
    formation: str
    accent: str
    crest: Optional[Path]
    slots: List[Slot]
    squad_size: int
    unplaced: List[str] = field(default_factory=list)


def _role_rank(p: PlayerRecord) -> int:
    return ROLE_RANK.get(p.role or "", 3)


def _value(p: PlayerRecord) -> float:
    return float(p.market_value or 0.0)


def _assign_players(players: Sequence[PlayerRecord], slots: List[Slot]) -> List[str]:
    """Place every player on exactly one slot. Returns names that did not fit."""
    order = sorted(players, key=lambda p: (_role_rank(p), -_value(p), p.name))

    def free(slot: Slot) -> bool:
        return len(slot.players) < MAX_PER_SLOT

    def take(slot: Slot, p: PlayerRecord) -> None:
        slot.players.append(p)

    unplaced: List[str] = []
    for p in order:
        ep = p.engine_pos
        eop = p.engine_other_pos

        # 1. natural position — spread across duplicate slots, then 2. secondary
        for target in (ep, eop):
            if not target:
                continue
            cands = [s for s in slots if s.pos == target and free(s)]
            if cands:
                take(min(cands, key=lambda s: (len(s.players), slots.index(s))), p)
                break
        else:
            # 3. positional cover
            placed = False
            for target in COVER.get(ep, ()) + (COVER.get(eop, ()) if eop else ()):
                cands = [s for s in slots if s.pos == target and free(s)]
                if cands:
                    take(min(cands, key=lambda s: (len(s.players), slots.index(s))), p)
                    placed = True
                    break
            if placed:
                continue

            # 4. any non-GK slot with room
            cands = [s for s in slots if s.pos != "GK" and free(s)]
            if cands:
                take(min(cands, key=lambda s: (len(s.players), slots.index(s))), p)
            else:
                unplaced.append(p.name)

    for s in slots:
        s.players.sort(key=lambda p: (_role_rank(p), -_value(p), p.name))
    return unplaced


def build_sheet(loader: RosterLoader, club: str) -> TeamSheet:
    squad = loader.build_matchday_squad(club)
    formation = squad["formation"]
    slots_def = FORMATION_SLOTS[formation]
    coords = FORMATION_XY[formation]

    slots = [Slot(pos=pos, x=x, y=y) for pos, (x, y) in zip(slots_def, coords)]
    players = loader.get_club_players(club)
    unplaced = _assign_players(players, slots)

    crest_name = CRESTS.get(club)
    crest = (LOGO_DIR / crest_name) if crest_name else None
    accent = ACCENTS.get(club, "444444")
    if crest is None or not crest.exists():
        crest = _placeholder_crest(club, accent)

    return TeamSheet(
        club=club,
        formation=formation,
        accent=accent,
        crest=crest,
        slots=slots,
        squad_size=len(players),
        unplaced=unplaced,
    )


def _placeholder_crest(club: str, accent: str) -> Path:
    """Generated badge for a club with no art in the federation crest pack."""
    from PIL import Image, ImageDraw, ImageFont

    out_dir = ROOT / "artifacts" / "crests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{club.replace(' ', '_').lower()}_placeholder.png"

    size = 512
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    rgb = tuple(int(accent[i:i + 2], 16) for i in (0, 2, 4))

    d.ellipse((14, 14, size - 14, size - 14), fill=(255, 255, 255, 255),
              outline=rgb + (255,), width=18)
    d.ellipse((54, 54, size - 54, size - 54), outline=rgb + (255,), width=6)

    initials = "".join(w[0] for w in club.split()[:2]).upper()
    font = None
    for name in ("arialbd.ttf", "seguisb.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, 190)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), initials, font=font)
    d.text(((size - (box[2] - box[0])) / 2 - box[0],
            (size - (box[3] - box[1])) / 2 - box[1]),
           initials, font=font, fill=rgb + (255,))

    img.save(out)
    return out



def build_all(loader: Optional[RosterLoader] = None) -> List[TeamSheet]:
    loader = loader or RosterLoader()
    clubs = sorted(loader.get_all_clubs(), key=lambda c: c.lower())
    return [build_sheet(loader, c) for c in clubs]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def surname(name: str) -> str:
    """Last token, skipping generational suffixes; falls back to the full name
    when the surname is only an initial ("Pablo V.") so nothing reads as "V."."""
    parts = name.split()
    if len(parts) == 1:
        return name
    last = parts[-1]
    if last.lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv"} and len(parts) > 2:
        return parts[-2]
    if len(last) <= 2:
        return name
    return last


def display_names(players: Sequence[PlayerRecord]) -> List[str]:
    """Surname only — unless two players in the same plate share one."""
    sns = [surname(p.name) for p in players]
    out = []
    for p, sn in zip(players, sns):
        out.append(sn if sns.count(sn) == 1 else p.name)
    return out


def to_json(sheets: Sequence[TeamSheet]) -> dict:
    return {
        "season": "2026/27",
        "clubs": [
            {
                "club": s.club,
                "formation": s.formation,
                "squad_size": s.squad_size,
                "accent": s.accent,
                "crest": str(s.crest) if s.crest else None,
                "unplaced": s.unplaced,
                "slots": [
                    {
                        "pos": sl.pos,
                        "x": sl.x,
                        "y": sl.y,
                        "players": [
                            {"no": p.number, "name": p.name, "pos": p.engine_pos,
                             "role": p.role, "age": p.age}
                            for p in sl.players
                        ],
                    }
                    for sl in s.slots
                ],
            }
            for s in sheets
        ],
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

#: 24/25 → 26/27 membership change, taken from alltime.db (24/25 matches) versus
#: the 26/27 workbook. Three clubs left the division and three came up; the
#: 24/25 deck's Buelton / FCTP2 / GPC slides therefore have no 26/27 successor.
OUT_OF = ("Buelton", "FCTP2", "GPC")
IN_TO = ("Avada Zenith", "Rodice", "Seafcea")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="output folder for the deck (and PNGs)")
    ap.add_argument("--json", type=Path, default=None,
                    help="dump the resolved sheet data to this JSON file")
    ap.add_argument("--no-export", action="store_true",
                    help="write the .pptx only, skip the PNG export")
    args = ap.parse_args(argv)

    from team_sheets_render import export_pngs, render_deck

    sheets = build_all()
    print(f"built {len(sheets)} club sheets "
          f"({sum(s.squad_size for s in sheets)} players)")

    problems = [s for s in sheets if s.unplaced]
    for s in problems:
        print(f"  !! {s.club}: {len(s.unplaced)} unplaced -> {s.unplaced}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(to_json(sheets), indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"data -> {args.json}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pptx = render_deck(
        sheets,
        out_dir / "PLOFA-SQUAD-DEPTH-2026-2027.pptx",
        league_crest=LEAGUE_CREST,
        out_of=OUT_OF,
        in_to=IN_TO,
    )
    print(f"deck -> {pptx}")

    if not args.no_export:
        pngs = export_pngs(pptx, out_dir / "PNG")
        print(f"pngs -> {pngs}")

    for s in sheets:
        top = " | ".join(
            f"{sl.pos}: " + ", ".join(f"{p.number} {surname(p.name)}" for p in sl.players)
            for sl in s.slots
        )
        print(f"  {s.club:<15} {s.formation:<8} {top}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
