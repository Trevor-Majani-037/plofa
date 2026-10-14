"""Soul players: does greatness actually emerge from soulness?

Answers three questions with numbers, no vibes:
  1. is a soul reliably better than an ordinary player of the same position?
  2. do the PILLARS change the outcome (the thing that was provably dead)?
  3. did the double-charge fix clear the 99.0 striker pile-up?
"""
import io
import statistics as st
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from player_soul import PlayerSoul, GreatnessPillars, SoulArchetype
from player_dna import DNAFactory

SPECS = {
    "ST": ["shooter", "creator"], "RW": ["shooter", "creator", "big_game_player"],
    "LW": ["shooter", "creator", "big_game_player"], "CM": ["box_box"],
    "CDM": ["anchor_man"], "CB": ["stopper_defender"],
}
ROSTER = [
    ("Haaland",    "GOALSCORING_SAVANT",  "ST",  (0.99, 0.97, 0.90)),
    ("Yamal",      "CREATIVE_ORACLE",     "RW",  (0.94, 0.99, 0.93)),
    ("Mbappe",     "ATTACKING_PROPHET",   "LW",  (0.97, 0.98, 0.92)),
    ("Bellingham", "MIDFIELD_PHILOSOPHER","CM",  (0.99, 0.95, 0.90)),
    ("Rice",       "SWEEPER_SAGE",        "CDM", (0.99, 0.90, 0.92)),
    ("Saliba",     "DEFENSIVE_PURIST",    "CB",  (0.98, 0.93, 0.91)),
    ("Perćy Luka", "ATTACKING_PROPHET",   "RW",  (0.99, 0.90, 0.99)),
]

FIELDS = [("finishing", "technical"), ("dribbling", "technical"), ("crossing", "technical"),
          ("composure", "mental"), ("vision", "mental"), ("positioning", "mental"),
          ("anticipation", "mental"), ("work_rate", "mental"),
          ("pace", "physical"), ("stamina", "physical"),
          ("tackling", "defending"), ("interceptions", "defending")]


def soul(name, arch, pos, pillars, age=26):
    hw, ta, lk = pillars
    s = PlayerSoul(name, archetype=getattr(SoulArchetype, arch),
                   pillars=GreatnessPillars(hardwork=hw, talent=ta, luck=lk))
    return DNAFactory.create(name, pos, SPECS[pos], age=age,
                             preferred_foot="left", soul=s), s


def make_soul(name, arch, pillars=(0.99, 0.95, 0.92)):
    hw, ta, lk = pillars
    return PlayerSoul(name, archetype=getattr(SoulArchetype, arch),
                      pillars=GreatnessPillars(hardwork=hw, talent=ta, luck=lk))


def line(label, dna, extra=""):
    cells = "".join(f"{getattr(getattr(dna, dom), f):>9.1f}" for f, dom in FIELDS)
    return f"{label:<14}{extra:<34}{cells}"


hdr = f"{'player':<14}{'soul / G / tier':<34}" + "".join(f"{f[:8]:>9}" for f, _ in FIELDS)
print("=" * len(hdr))
print("1. SOULS vs THE ORDINARY PLAYER THEY REPLACE")
print("=" * len(hdr))
print(hdr)
print("-" * len(hdr))
for name, arch, pos, pil in ROSTER:
    d, s = soul(name, arch, pos, pil)
    print(line(name, d, f"{arch[:14]} G={s.greatness_coefficient:.3f} {s.tier[:11]}"))
print()
for pos, lbl in (("ST", "Striker"), ("RW", "Right winger"), ("CM", "Midfielder"),
                 ("CDM", "Defensive mid"), ("CB", "Centre-back")):
    d = DNAFactory.create(f"Ordinary {pos}", pos, SPECS[pos], age=26)
    print(line(lbl, d, "(no soul)"))

# ── 2. DO THE PILLARS MATTER? ────────────────────────────────────────────
print()
print("=" * 78)
print("2. PILLARS: does the SAME soul with different pillars differ?")
print("=" * 78)
print("   Before 2026-10-14 this was provably dead: G=0.70 QUALITY and")
print("   G=1.5543 TRANSCENDENT produced IDENTICAL gameplay modifiers.\n")
print(f"   {'pillars (H/T/L)':<26}{'G':>8}{'tier':>15}{'finishing':>11}{'composure':>11}{'vision':>9}")
for label, p in [("weakest  0.70/0.70/0.70", (0.70, 0.70, 0.70)),
                 ("middling 0.85/0.80/0.80", (0.85, 0.80, 0.80)),
                 ("Percy    0.99/0.90/0.99", (0.99, 0.90, 0.99)),
                 ("strongest .99/.99/.99",    (0.99, 0.99, 0.99))]:
    d, s = soul(f"Probe {label[:6]}", "ATTACKING_PROPHET", "RW", p)
    print(f"   {label:<26}{s.greatness_coefficient:>8.3f}{s.tier:>15}"
          f"{d.technical.finishing:>11.1f}{d.mental.composure:>11.1f}{d.mental.vision:>9.1f}")

# ── 3. THE 99 CLIFF ──────────────────────────────────────────────────────
print()
print("=" * 78)
print("3. THE 99.0 CEILING PILE-UP")
print("=" * 78)
sts = []
for i in range(400):
    d = DNAFactory.create(f"Striker {i:03d}", "ST", ["shooter", "creator"], age=26)
    sts.append(d.technical.finishing)
at99 = sum(1 for v in sts if v == 99.0)
print(f"   400 strikers reading EXACTLY 99.0 : {at99}  ({at99/len(sts):.0%})")
print(f"   was 86% before the double-charge fix")
print(f"   range now {min(sts):.1f} - {max(sts):.1f}, sd {st.pstdev(sts):.2f}")

# ── 4. THE SOUL REACHES THE PITCH SURFACES ──────────────────────────────
print()
print("=" * 78)
print("4. CONTROL: the soul must be VISIBLE to the systems that read it")
print("=" * 78)
# exporter.py:573-576 reads these four off dna.soul. If create() only used the
# soul for the band lift without attaching it, the lift would be real but
# unattributable in every export.
s = make_soul("Export Probe", "GOALSCORING_SAVANT")
d = DNAFactory.create("Export Probe", "ST", SPECS["ST"], age=26, soul=s)
print(f"   dna.soul attached          : {d.soul is s}")
print(f"   exporter soul_archetype    : {d.soul.profile.label}")
print(f"   exporter soul_tier         : {d.soul.tier}")
print(f"   exporter soul_greatness    : {round(d.soul.greatness_coefficient, 4)}")
print(f"   exporter soul_omega        : {d.soul.pillars.omega_activated}")
print()
plain = DNAFactory.create("Export Probe", "ST", SPECS["ST"], age=26)
print(f"   same player, no soul       : dna.soul={plain.soul}  "
      f"finishing={plain.technical.finishing}")
print(f"   with soul                  : finishing={d.technical.finishing}")
print()
print("   NOTE: an ordinary player IS unchanged by the soul machinery itself,")
print("   but NOT by the separate double-charge fix — that one intentionally")
print("   changes every player whose specialty selected its archetype, which is")
print("   what cleared the 99.0 pile-up in section 3.")
