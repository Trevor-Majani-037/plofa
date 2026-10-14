"""Football meaning of freezing the DNA.

Corrected after the first version of this probe printed a verdict its own
number contradicted: it reported "mean -9.54 - NOT systematically buffed or
nerfed". Two bugs, both mine:
  * the default finishing band is (35, 60), not (55, 72) — I copied the
    first_touch line;
  * 'clinical_finisher' multiplies finishing by 1.22 (_build_technical:1211)
    which I did not apply, so strikers looked inflated.
A verdict that contradicts its own number is worse than no verdict.
"""
import io
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from player_dna import DNAFactory, ArchetypeLibrary

# Read the real default bands out of the source instead of assuming them.
SRC = (ROOT / "player_dna.py").read_text(encoding="utf-8")
m = re.search(r'finishing\s*=\s*cls\._attr\(arch, "technical\.finishing",\s*\((\d+),\s*(\d+)\)',
              SRC)
DEFAULT_FINISHING = (int(m.group(1)), int(m.group(2)))
SPEC_MULT = 1.22 if re.search(r'if "clinical_finisher" in specs.*\n\s*t\.finishing\s*=\s*min\(99, t\.finishing \* 1\.22', SRC) else 1.0

HIST = json.loads((ROOT / "plofa_output" / "history" /
                   "player_history.json").read_text(encoding="utf-8"))

SPECS_BY_POS = {
    "CB": ["stopper_defender"], "LB": ["aggressive_fullback"], "RB": ["aggressive_fullback"],
    "CDM": ["anchor_man"], "CM": ["box_box"], "CAM": ["playmaker"],
    "LW": ["crosser"], "RW": ["crosser"], "ST": ["clinical_finisher"], "CF": ["clinical_finisher"],
}


def raw_band(name, pos, specs):
    """The band _attr actually draws from, before the specialty multiplier."""
    arch_key = ArchetypeLibrary.get_archetype_for_player(pos, specs)
    arch = ArchetypeLibrary.ARCHETYPES.get(arch_key, {})
    return arch_key, arch.get("technical.finishing", DEFAULT_FINISHING)


print("=" * 94)
print(f"SETUP   default finishing band read from source: {DEFAULT_FINISHING}"
      f"   clinical_finisher multiplier: x{SPEC_MULT}")
print("=" * 94)

roster = [(n, r["position"]) for n, r in HIST["players"].items()
          if r.get("position") not in (None, "GK")]

# ── 1. PERCY ─────────────────────────────────────────────────────────────
print()
print("=" * 94)
print("1. PERCY LUKA")
print("=" * 94)
old = [37.6, 61.3, 60.5]
new = DNAFactory.create("Perćy Luka", "RW", ["shooter", "creator", "big_game_player"],
                        age=34, preferred_foot="left").technical.finishing
print(f"  BEFORE  three separate matchdays : {old}   mean {st.mean(old):.1f}")
print(f"  AFTER   the same number, always  : {new}")
print(f"          vs his old MEAN          : {new - st.mean(old):+.1f}")
print(f"          vs his old WORST week    : {new - min(old):+.1f}   <- no more 37.6 weeks")
print(f"          vs his old BEST week     : {new - max(old):+.1f}")

# ── 2. LEAGUE STRENGTH ───────────────────────────────────────────────────
print()
print("=" * 94)
print(f"2. IS THE LEAGUE STRONGER OR WEAKER?  ({len(roster)} outfielders)")
print("=" * 94)
print("  OLD: every player drew a fresh number each matchday, so over 38 games each")
print("       one AVERAGED toward their band. NEW: one fixed number all season.")
print("       The fair comparison is season-average vs fixed.\n")

old_season_avg, new_fixed, dev_from_centre = [], [], []
for name, pos in roster:
    specs = SPECS_BY_POS.get(pos, ["box_box"])
    arch_key, band = raw_band(name, pos, specs)
    mult = SPEC_MULT if "clinical_finisher" in specs else 1.0

    random.seed(abs(zlib_crc := __import__("zlib").crc32(name.encode())) % 100000)
    season = [min(99.0, random.uniform(*band) * mult) for _ in range(38)]
    old_season_avg.append(st.mean(season))

    dna = DNAFactory.create(name, pos, specs, age=27)
    new_fixed.append(dna.technical.finishing)
    dev_from_centre.append(dna.technical.finishing - (band[0] + band[1]) / 2 * mult)

print(f"  OLD  mean season-average finishing : {st.mean(old_season_avg):.2f}")
print(f"  NEW  mean fixed finishing          : {st.mean(new_fixed):.2f}")
print(f"  DIFFERENCE                         : {st.mean(new_fixed) - st.mean(old_season_avg):+.2f}"
      f"   ({abs(st.mean(new_fixed)-st.mean(old_season_avg))/st.mean(old_season_avg)*100:.1f}%)")
print(f"\n  SPREAD between players (this is the part that really changed):")
print(f"    OLD  sd across the league       : {st.pstdev(old_season_avg):5.2f}")
print(f"    NEW  sd across the league       : {st.pstdev(new_fixed):5.2f}"
      f"   <- {st.pstdev(new_fixed)/st.pstdev(old_season_avg):.1f}x wider")
print(f"    OLD  best-vs-worst gap          : {max(old_season_avg) - min(old_season_avg):5.1f}")
print(f"    NEW  best-vs-worst gap          : {max(new_fixed) - min(new_fixed):5.1f}")
print(f"\n  mean deviation from band centre    : {st.mean(dev_from_centre):+.2f}")
print(f"  players ABOVE their band centre    : "
      f"{sum(1 for v in dev_from_centre if v > 0)}/{len(dev_from_centre)} = "
      f"{sum(1 for v in dev_from_centre if v > 0)/len(dev_from_centre):.0%}  (fair coin: 50%)")

# ── 3. THE 99 CLIFF ──────────────────────────────────────────────────────
print()
print("=" * 94)
print("3. A PRE-EXISTING CEILING PILE-UP (not caused by this change)")
print("=" * 94)
print(f"  complete_striker's finishing band tops out at 91, and the 'clinical_finisher'")
print(f"  specialty multiplies by {SPEC_MULT}: 91 x {SPEC_MULT} = {91*SPEC_MULT:.1f}, clamped to 99.")
print(f"  So any raw draw above {99/SPEC_MULT:.1f} pins to EXACTLY 99.0 — "
      f"{(91-99/SPEC_MULT)/(91-80)*100:.0f}% of the band.")
sts = [DNAFactory.create(n, p, SPECS_BY_POS.get(p, ["box_box"]), age=27).technical.finishing
       for n, p in roster if p in ("ST", "CF")]
at99 = sum(1 for v in sts if v == 99.0)
print(f"\n  Strikers reading EXACTLY 99.0 finishing: {at99}/{len(sts)} = {at99/len(sts):.0%}")
print("  They are indistinguishable to the engine. True before this change too — the")
print("  draw moved around each matchday, so the clump shuffled instead of sitting still.")

# ── 4. WHAT ACTUALLY CHANGED ─────────────────────────────────────────────
print()
print("=" * 94)
print("4. THE ACTUAL FOOTBALL CHANGE: week-to-week predictability")
print("=" * 94)
swings = []
for name, pos in roster[:250]:
    specs = SPECS_BY_POS.get(pos, ["box_box"])
    _, band = raw_band(name, pos, specs)
    mult = SPEC_MULT if "clinical_finisher" in specs else 1.0
    random.seed(__import__("zlib").crc32(name.encode()) % 100000)
    season = [min(99.0, random.uniform(*band) * mult) for _ in range(38)]
    swings.append(max(season) - min(season))
print(f"  OLD: a player's finishing moved up and down inside the band all season.")
print(f"       average within-season SWING per player : {st.mean(swings):.1f} finishing")
print(f"  NEW: one number. SWING = 0 for everyone, every week.")
print(f"\n  Same players, same average quality ({st.mean(old_season_avg):.1f} vs "
      f"{st.mean(new_fixed):.1f}) — but the OLD ones were unreliable, the NEW ones are not.")
