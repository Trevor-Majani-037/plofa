"""
_diag_awareness_gates.py — why do 100% of midfielders clear the >= 55 gate,
when the default band's floor is 50?

Written because the new test failed. Two candidate explanations:
  (a) my (50, 70) default still produces a distribution too high, or
  (b) the DEFAULT BAND BARELY REACHES MIDFIELDERS, because the eight named
      archetype templates already cover the mid positions and every one of
      their bands is 60-93 — i.e. the 100% clear rate is a property of the
      TEMPLATES, not of my choice, and I was arguing with the wrong lever.

These are very different findings with different fixes, so measure before
touching the band again. No match required.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import player_dna as PD  # noqa: E402

_LINES: list[str] = []


def p(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def rule(ch: str = "=") -> None:
    p(ch * 78)


TMPL = PD.ArchetypeLibrary.ARCHETYPES
SPEC_MAP = PD.ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP
POS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

# Which specialties reach a template that NAMES geometric_awareness?
NAMED_KEY = "mental.geometric_awareness"
named_arch = {a for a, t in TMPL.items() if NAMED_KEY in t}
named_specs = {s for s, a in SPEC_MAP.items() if a in named_arch}

rule()
p("A.  WHICH ARCHETYPES NAME THE FIELD, AND HOW REACHABLE ARE THEY?")
rule()
p("")
p(f"  archetypes total                     : {len(TMPL)}")
p(f"  archetypes naming geometric_awareness: {len(named_arch)}")
for a in sorted(named_arch):
    lo, hi = TMPL[a][NAMED_KEY]
    p(f"      {a:<24} ({lo}, {hi})")
p("")
p(f"  specialties in SPECIALTY_ARCHETYPE_MAP: {len(SPEC_MAP)}")
p(f"  specialties reaching a NAMED archetype: {len(named_specs)} ({100.0*len(named_specs)/len(SPEC_MAP):.0f}%)")
p("")
p("  -> a player gets a template band only if his SPECIALTY maps to one of the")
p("     eight. Everything else falls back to the default band. So the realised")
p("     distribution is a MIXTURE, and the default band's reach is the question.")
p("")

# ── Now measure which archetype each midfielder actually got ──
rule()
p("B.  THE REALISED MIXTURE PER POSITION (2400 players, 3 seeds each)")
rule()
p("")
rows = []
for seed in (1234, 4242, 777):
    random.seed(seed)
    np.random.seed(seed)
    for i in range(800):
        pos = POS[i % len(POS)]
        age = random.randint(18, 37)
        spec = random.choice(list(SPEC_MAP.keys()))
        dna = PD.DNAFactory.create(f"M{i}_{seed}", pos, [spec], age=age)
        arch = SPEC_MAP[spec]
        rows.append((pos, arch, arch in named_arch, dna.mental.geometric_awareness, age))

p("  B1. share of players whose archetype NAMES the field, by position")
p(f"      {'pos':<6} {'n':>5} {'named':>8} {'share':>8}")
for pos in POS:
    sub = [r for r in rows if r[0] == pos]
    named = sum(1 for r in sub if r[2])
    p(f"      {pos:<6} {len(sub):>5} {named:>8} {100.0*named/max(1,len(sub)):>7.1f}%")
p("")
p("  B2. the two populations separated, per position — this is the whole answer")
p(f"      {'pos':<6} {'named n':>8} {'named med':>10} {'named <55':>10} "
  f"{'deflt n':>8} {'deflt med':>10} {'deflt <55':>10}")
p(f"      {'':<6} {'':>8} {'':>10} {'(gate)':>10} "
  f"{'':>8} {'':>10} {'(gate)':>10}")
for pos in POS:
    nsub = [r[3] for r in rows if r[0] == pos and r[2]]
    dsub = [r[3] for r in rows if r[0] == pos and not r[2]]
    def f(v):
        return f"{np.median(v):>10.1f}" if v else f"{'-':>10}"
    def s(v):
        return (f"{100.0*sum(1 for x in v if x < 55)/len(v):>9.0f}%" if v else f"{'-':>10}")
    p(f"      {pos:<6} {len(nsub):>8} {f(nsub)} {s(nsub)} "
      f"{len(dsub):>8} {f(dsub)} {s(dsub)}")
p("")
p("  B3. gate outcomes, WITH the named and default populations recombined")
p(f"      {'pos':<6} {'n':>5} {'>=55 (mid gate)':>15} {'>=50 (att gate)':>15}")
for pos in POS:
    sub = [r[3] for r in rows if r[0] == pos]
    if not sub:
        continue
    p(f"      {pos:<6} {len(sub):>5} "
      f"{100.0*sum(1 for x in sub if x >= 55)/len(sub):>14.1f}% "
      f"{100.0*sum(1 for x in sub if x >= 50)/len(sub):>14.1f}%")
p("")
p("  B4. the same, split by age — the age multiplier is a real lever here")
p(f"      {'age band':<10} {'n':>5} {'median ga':>11} {'<55':>8}")
for lo, hi, label in ((18, 21, "18-21"), (22, 25, "22-25"), (26, 30, "26-30"), (31, 37, "31-37")):
    sub = [r[3] for r in rows if lo <= r[4] <= hi]
    if not sub:
        continue
    p(f"      {label:<10} {len(sub):>5} {np.median(sub):>11.1f} "
      f"{100.0*sum(1 for x in sub if x < 55)/len(sub):>7.1f}%")
p("")

# ── the decisive test of explanation (a) vs (b) ──
rule()
p("C.  IS THE 100% CLEAR RATE MY DEFAULT BAND, OR THE TEMPLATES?")
rule()
p("")
p("  C1. recompute the midfielder gate rate with THREE candidate default bands,")
p("      holding the templates fixed. If the rate barely moves, the default band")
p("      is NOT the lever and (b) is the answer.")
p("")
mid_named = np.array([r[3] for r in rows if r[0] in ("CDM", "CM", "CAM") and r[2]], dtype=float)
mid_deflt_n = sum(1 for r in rows if r[0] in ("CDM", "CM", "CAM") and not r[2])
p(f"      midfielder sample: {len(mid_named)} template-named, {mid_deflt_n} default-band")
p("")
p(f"      {'default band':<16} {'mid <55 (defaults)':>20} {'recombined clear':>18}")
for lo, hi in ((40, 60), (45, 65), (50, 70), (55, 74), (50, 60)):
    random.seed(31337)
    np.random.seed(31337)
    draw = np.array([round(min(99.0, max(30.0, random.uniform(lo, hi) * 0.98)), 1)
                     for _ in range(max(1, mid_deflt_n))], dtype=float)
    if len(draw):
        below = 100.0 * (draw < 55).mean()
    else:
        below = float("nan")
    combined = np.concatenate([mid_named, draw]) if len(draw) else mid_named
    clear = 100.0 * (combined >= 55).mean()
    p(f"      {f'({lo}, {hi})':<16} {below:>19.1f}% {clear:>17.1f}%")
p("")
p("      If these five rows are within a few points of each other, no default")
p("      band rescues the gate, because almost no midfielder USES the default.")
p("")

p("  C2. what share of MIDFIELDERS use the default band at all?")
for pos in ("CDM", "CM", "CAM", "LW", "RW", "ST", "CF", "CB", "LB", "RB", "GK"):
    sub = [r for r in rows if r[0] == pos]
    if not sub:
        continue
    share = 100.0 * sum(1 for r in sub if not r[2]) / len(sub)
    p(f"      {pos:<5} default-band share {share:>5.1f}%")
p("")

# ── so what IS the truth? ──
rule()
p("D.  THE ARITHMETIC, stated plainly")
rule()
p("")
p("  The :3431 gate is 55. The template bands are:")
bands = sorted(TMPL[a][NAMED_KEY][0] for a in named_arch)
p(f"      template floors: {bands}")
p(f"      lowest template floor {min(bands)} vs the gate 55 -> gap {min(bands)-55:+d}")
p("")
p("  Every named template's LOWEST possible value is above the gate, so a")
p("  midfielder who lands on ANY of the eight templates clears the gate on")
p("  every single draw. The gate can therefore only ever be meaningful for the")
p("  default-band population, and section C2 measures how small that is.")
p("")
p("  CONSEQUENCE: this is NOT primarily a consequence of the one-line fix. It")
p("  is a PRE-EXISTING property of the gate-vs-template relationship that only")
p("  became VISIBLE because the attribute stopped being a constant. Before the")
p("  fix the gate rejected 100% (constant 50.0) and the templates were dead")
p("  weight; now the templates work and the gate is close to vacuous for them.")
p("")
p("  Two honest options, and picking one is a design decision, not a tuning")
p("  decision, so it is NOT taken here:")
p("")
p("  (i)  Accept it. The rules were written as 'awareness-gated drift' and a")
p("       midfield squad of spatially-aware players drifting into half-spaces")
p("       IS positional play. The gate then documents an intent that in a")
p("       normal squad is simply always true, which is not harmful.")
p("  (ii) Raise the gate above the template band. e.g. :3431 at 76 would keep")
p("       roughly the top third of the template population, which is what a")
p("       gate implies. But that is changing a TUNED threshold to suit a")
p("       distribution I just created, and no measurement says the resulting")
p("       shape is better. Per CORNER STEP 6 discipline — do not tune real")
p("       behaviour to make an assertion pass, and do not tune it to make a")
p("       test pass either.")
p("")
p("  What the TEST should therefore assert is NOT 'the gate rejects some")
p("  players' — that is a claim about a distribution nobody specified. It")
p("  should assert what is actually true and would be a real regression if it")
p("  stopped: the attribute VARIES, it reaches the gates, and it produces a")
p("  continuous per-player factor rather than a constant.")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_awareness_gates.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_LINES) + "\n")
print(f"\n[written] {out}")
