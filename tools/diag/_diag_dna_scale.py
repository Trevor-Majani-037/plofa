"""
_diag_dna_scale.py — WHY PLOFA DNA ≠ FIFA/EAFC ATTRIBUTES

Read-only. No match simulation, no engine writes. Answers five questions
with numbers rather than argument:

  A. How many attributes exist, and how many of them are ever generated?
  B. How much of the archetype library actually reaches a player, and how
     much falls through to a DEGENERATE default range?
  C. Does `overall_rating` share a common scale across positions, and can
     it exceed 99 while every component is capped at 99?
  D. What is one attribute point WORTH on the pitch? (per-attribute output
     sensitivity, read off the real consuming formulas)
  E. How far can a soul player exceed a non-soul with identical attributes?

Writes _diag_dna_scale.txt.
"""

from __future__ import annotations
import ast
import dataclasses
import inspect
import io
import textwrap
import os
import random
import sys
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _src(obj):
    return textwrap.dedent(inspect.getsource(obj))


OUT = io.StringIO()
L = []


def p(s: str = "") -> None:
    print(s)
    L.append(s)


def rule(ch="=", n=78):
    p(ch * n)


# ── import quietly ────────────────────────────────────────────────────────
buf = io.StringIO()
with redirect_stdout(buf):
    import player_dna as PD
    import player_soul as PS

rule()
p("A. THE ATTRIBUTE SURFACE")
rule()

domains = {
    "physical": PD.PhysicalAttributes,
    "technical": PD.TechnicalAttributes,
    "mental": PD.MentalAttributes,
    "passing": PD.PassingAttributes,
    "defending": PD.DefendingAttributes,
    "gk": PD.GoalkeeperAttributes,
}
FIELDS = {d: [f.name for f in dataclasses.fields(c)] for d, c in domains.items()}
N_ALL = sum(len(v) for v in FIELDS.values())
p(f"attributes declared            : {N_ALL}  (FIFA/EAFC rates 30)")
p(f"domains                        : {len(domains)}")
for d, names in FIELDS.items():
    p(f"   {d:<11} {len(names):>2}  {', '.join(names)}")

# gk_attrs is only built for GK (player_dna.py:970)
src = _src(PD.DNAFactory.create)
gk_only = "if position == \"GK\"" in src or "if position == 'GK'" in src
p("")
p(f"gk_attrs built for outfielders?  : {'YES' if not gk_only else 'NO  -> 8 attrs are dead 60.0 defaults for every outfielder'}")

# ── which fields does each builder actually ASSIGN? ───────────────────────
rule()
p("B. WHICH ATTRIBUTES ARE EVER GENERATED")
rule()
builders = {
    "physical": PD.DNAFactory._build_physical,
    "technical": PD.DNAFactory._build_technical,
    "mental": PD.DNAFactory._build_mental,
    "passing": PD.DNAFactory._build_passing,
    "defending": PD.DNAFactory._build_defending,
}
generated = set()
for dom, fn in builders.items():
    tree = ast.parse(_src(fn))
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for kw in node.keywords:
                if kw.arg:
                    assigned.add(kw.arg)
    generated |= assigned
    missing = [n for n in FIELDS[dom] if n not in assigned]
    flag = "  <-- NEVER ASSIGNED" if missing else ""
    p(f"{dom:<11} assigned {len(assigned):>2}/{len(FIELDS[dom])}"
      + (f"   missing: {', '.join(missing)}{flag}" if missing else ""))

DEAD = sorted({n for d, names in FIELDS.items() if d != "gk" for n in names} - generated)
p("")
if DEAD:
    p(f"DEAD ATTRIBUTES (declared, never written by any builder): {len(DEAD)}")
    p(f"   -> always the dataclass default: "
      f"{', '.join(f'{n}=' + str(getattr(PD.MentalAttributes(), n)) for n in DEAD)}")
else:
    p("no dead attributes")

# ── archetype template coverage ───────────────────────────────────────────
rule()
p("C. ARCHETYPE TEMPLATE COVERAGE (does the library reach the player?)")
rule()
ARCH = PD.ArchetypeLibrary.ARCHETYPES
p(f"archetypes defined             : {len(ARCH)}")

# the per-attribute default bands, read from the builders' own call sites
DEFAULTS = {}
for dom, fn in builders.items():
    tree = ast.parse(_src(fn))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
           and node.func.attr == "_attr" and len(node.args) >= 3:
            key = node.args[1]
            rng = node.args[2]
            if isinstance(key, ast.Constant) and isinstance(rng, ast.Tuple) \
               and len(rng.elts) == 2 and all(isinstance(e, ast.Constant) for e in rng.elts):
                DEFAULTS[key.value] = (rng.elts[0].value, rng.elts[1].value)

degenerate = {k: v for k, v in DEFAULTS.items() if v[0] == v[1]}
p(f"attributes with a default band  : {len(DEFAULTS)}")
p(f"   of which DEGENERATE (lo==hi, zero variance, every player identical): {len(degenerate)}")
for k, v in sorted(degenerate.items()):
    p(f"      {k:<34} = {v[0]:.0f}  (constant)")

# per-attribute: how many archetypes name it
allkeys = set()
for a in ARCH.values():
    allkeys |= {k for k in a if k.startswith(("physical.", "technical.", "mental.",
                                              "passing.", "defending.", "gk_attrs."))}
p("")
p(f"{'attribute':<34} {'named by':>9}  {'of':>3}  fallback band")
rows = []
for k in sorted(allkeys):
    named = sum(1 for a in ARCH.values() if k in a)
    fb = DEFAULTS.get(k)
    rows.append((named, k, fb))
for named, k, fb in sorted(rows):
    if named == 0:
        p(f"{k:<34} {named:>9}  {len(ARCH):>3}  {fb if fb else '(dataclass default)'}"
          + ("   <-- FALLS THROUGH" if named == 0 else ""))
cov = [n for n, _, _ in rows]
p("")
p(f"template keys named by at least one archetype : {sum(1 for n in cov if n)} / {len(cov)}")
p(f"mean archetypes naming each key              : {np.mean([n for n,_,_ in rows]):.1f} / {len(ARCH)}")

# ── overall_rating ────────────────────────────────────────────────────────
rule()
p("D. overall_rating — is it one scale or several?")
rule()
tree = ast.parse(_src(PD.PlayerDNA.overall_rating.fget))
# Walk the if/elif chain MANUALLY. ast.walk(node) descends into nested
# branches, so an earlier version of this probe attributed every inner
# np.mean to the outermost `if` and labelled every row "pos == 'GK'".
def _branch_div(stmts):
    """The explicit `/ value` applied to THIS branch's own return."""
    for st in stmts:
        if isinstance(st, ast.Return) and isinstance(st.value, ast.BinOp) \
           and isinstance(st.value.op, ast.Div) and isinstance(st.value.right, ast.Constant):
            return st.value.right.value
    return None

def _branch_mean(stmts):
    for st in stmts:
        node = st.value if isinstance(st, ast.Return) else st
        if isinstance(node, ast.BinOp):
            node = node.left
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
           and node.func.attr == "mean" and node.args and isinstance(node.args[0], ast.List):
            return node.args[0].elts
    return []

POS_SCALE = []
func = tree.body[0]                       # the overall_rating property getter
# body[0] is the docstring, NOT the first branch -- look for the If itself.
cur = next(s for s in func.body if isinstance(s, ast.If))
while isinstance(cur, ast.If):
    cond = ast.unparse(cur.test)
    elts = _branch_mean(cur.body)
    outer_div = _branch_div(cur.body)
    weights, names = [], []
    ok = bool(elts)
    for e in elts:
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult) and isinstance(e.right, ast.Constant):
            weights.append(e.right.value)
            names.append(ast.unparse(e.left))
        elif isinstance(e, ast.Attribute):
            weights.append(1.0)
            names.append(ast.unparse(e))
        else:
            ok = False
            break
    if ok and weights:
        n = len(weights)
        wsum = sum(weights)
        # np.mean -> /n, then this branch divides again by outer_div
        eff = n * outer_div if outer_div else n
        infl = eff / wsum
        POS_SCALE.append((cond, n, wsum, outer_div, eff, infl, names))
        p(f"  {cond}")
        p(f"     n={n}   sum(weights)={wsum:.1f}   explicit '/ 1.07' -> {outer_div}")
        p(f"     effective divisor = n * {outer_div} = {eff:.3f}"
          f"   vs a plain mean's {wsum:.3f}   inflation x{infl:.4f} ({infl*100-100:+.2f}%)")
        p(f"     terms: {', '.join(f'{nm.split('.')[-1]}x{w}' for nm, w in zip(names, weights))}")
    cur = cur.orelse[0] if len(cur.orelse) == 1 else None

# does it clamp?
fget = _src(PD.PlayerDNA.overall_rating.fget)
clamped = "min(" in fget or "clip(" in fget
p("")
p(f"overall_rating clamped to 99?     : {'yes' if clamped else 'NO  -> weighted sum can exceed the 99 cap on every component'}")
if POS_SCALE:
    infl = [s[5] for s in POS_SCALE]
    p(f"per-position inflation spread    : {min(infl):.4f}x .. {max(infl):.4f}x"
      f"  (a {(max(infl)-min(infl))*100:.2f} point band)")
    p("  -> one hard-coded divisor (1.07) serves every position, but no")
    p("     position's weights actually sum to it, so each position's Overall")
    p("     sits on a slightly different scale from the others.")

# ── age curve ─────────────────────────────────────────────────────────────
rule()
p("E. THE AGE CURVE IS BAKED INTO THE ATTRIBUTE (player_dna.py:_attr)")
rule()
for r, m in sorted(PD.DNAFactory.AGE_CURVE.items(), key=lambda kv: kv[0].start):
    p(f"   age {r.start}-{r.stop-1:<3} x{m}")
p("")
p("   _attr() = clamp(uniform(lo,hi) * age_mult, 30, 99)")
p("   -> age multiplies EVERY attribute, and is stored permanently.")
ex_hi = 95.0
for age, m in ((18, 0.82), (25, 0.97), (28, 1.00), (33, 0.91), (40, 0.82)):
    p(f"      a 95-attribute player at age {age}: {ex_hi*m:.1f}")
p("   -> the SAME ability reads ~78 at 40 and 95 at 28.")

# ── attribute -> output sensitivity ───────────────────────────────────────
rule()
p("F. WHAT IS ONE ATTRIBUTE POINT WORTH? (read off the real consumers)")
rule()
p("")
p("  finishing -> on-target prob   (event_chain.py:6697)")
for f in (35, 60, 90):
    base = 0.20 + 0.10 * 0.65 + ((60 + f) / 100.0) * 0.05
    p(f"      finishing {f:>3}: base on-target ~{base:.4f}")
d = (0.20 + 0.10 * 0.65 + ((60 + 90) / 100.0) * 0.05) - (0.20 + 0.10 * 0.65 + ((60 + 35) / 100.0) * 0.05)
p(f"      -> 55 attribute points (35->90) = {d*100:.2f} percentage points of on-target")
p("")
p("  pace -> press effectiveness    (event_chain.py:8364)")
for pa in (50, 70, 95):
    p(f"      pace {pa:>3}: {pa/100.0*0.3+0.7:.4f}")
p(f"      -> 45 attribute points (50->95) = {((95/100*0.3+0.7)-(50/100*0.3+0.7))*100:.1f}% relative")
p("")
p("  short_passing -> pass accuracy (player_dna.py:1308)")
for s in (50, 65, 90):
    p(f"      short_passing {s:>3}: {0.55+(s/100.0)*0.45:.3f}")
p(f"      -> 40 attribute points = {((0.55+0.9*0.45)-(0.55+0.5*0.45))*100:.1f}% relative")
p("")
p("  finishing -> xG multiplier    (player_dna.py:1285)   [uses effective_*]")
for f in (35, 60, 90):
    p(f"      finishing {f:>3}: {0.70+(f/100.0)*0.70:.4f}")
p("")
p("  >>> the SAME 0-100 scale is consumed at 0.02pp, 14%, 18% and 49% weight.")
p("      There is no single conversion from an attribute point to football.")

# ── the live shot path ────────────────────────────────────────────────────
rule()
p("G. DOES THE LIVE SHOT PATH USE form/fatigue/stamina?")
rule()
live = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_chain.py"),
            encoding="utf-8").read()
body = live[live.index("def _shot_on_target_prob"):]
body = body[:body.index("\n    @", 200) if "\n    @" in body[200:] else len(body)]
uses_raw = "dna.technical.finishing" in body and "effective_finishing" not in body
uses_eff = "effective_finishing" in body
p(f"event_chain._shot_on_target_prob uses effective_finishing : {uses_eff}")
p(f"event_chain._shot_on_target_prob uses RAW finishing       : {uses_raw}")
p("")
p("  -> form_multiplier / fatigue_multiplier / live_performance_mult are")
p("     NOT applied to the live shot. get_shooter_quality() (which does use")
p("     them) is not on this path. So the state layer that makes the numbers")
p("     feel alive is absent from the most important action in football.")

# ── soul ──────────────────────────────────────────────────────────────────
rule()
p("H. SOUL: how far can a player exceed an identical-attribute non-soul?")
rule()
ap = PS.SoulLibrary.PROFILES[PS.SoulArchetype.ATTACKING_PROPHET]
p(f"ATTACKING_PROPHET shot_quality_mult = {ap.shot_quality_mult}")
p("")
p("  get_event_multiplier: scaled = 1.0 + (base - 1.0) * G     (player_soul.py:706)")
p("  then x context_mult (losing 1.05, late 1.05, added 1.08, flow 1.12)")
p("")
hdr = f"  {'G':>6}  {'scaled':>7}  {'x all ctx':>10}"
p(hdr)
for G in (0.62, 0.82, 0.88, 0.95, 1.55):
    scaled = 1.0 + (ap.shot_quality_mult - 1.0) * G
    ctx = ap.losing_state_boost * ap.late_game_boost * 1.12
    p(f"  {G:>6.2f}  {scaled:>7.4f}  {scaled*ctx:>10.4f}")
p("")
p(f"  a non-soul with identical attributes: 1.0000")
p(f"  a GENERATIONAL soul, losing late in flow: up to "
  f"{(1.0+(ap.shot_quality_mult-1.0)*1.55)*ap.losing_state_boost*ap.late_game_boost*1.12:.2f}x")
p("")
p("  >>> FIFA has no multiplicative activation term. To express a 2x output")
p("      difference, that player would need roughly +20 overall points.")

rule()
p("I. GATE: the soul's activation is a HARD AND on all three pillars")
rule()
GT = PS.GreatnessPillars
p(f"  hardwork > {GT.HARDWORK_THRESHOLD}  AND talent > {GT.TALENT_THRESHOLD}"
  f"  AND luck > {GT.LUCK_THRESHOLD}")
p("  omega is 1.0 unless ALL THREE clear. Luck carries weight "
  f"{GT.GAMMA} (the smallest) but is a gate.")
t, h, l = 0.99, 0.99, 0.70
p(f"  talent={t} hardwork={h} luck={l} (perfect two, unlucky) -> "
  f"omega={PS.GreatnessPillars(h, t, l).omega}")
p(f"  talent={t} hardwork={h} luck=0.76                      -> "
  f"omega={PS.GreatnessPillars(h, t, 0.76).omega}")

rule()
p("J. ARE THE SOUL BONUS STATS APPLIED IN PRODUCTION?")
rule()
p("  get_bonus_stats() returns HARD ADDITIONS on top of simulated stats,")
p("  e.g. ATTACKING_PROPHET: "
  f"shot_assists +{ap.bonus_shot_assists_per_match[0]}-{ap.bonus_shot_assists_per_match[1]}, "
  f"carries +{ap.bonus_carries_per_match[0]}-{ap.bonus_carries_per_match[1]}")
p("")
for fn in ("auto_run_match.py", "run_match.py", "exporter.py", "world/ingest.py"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
    if not os.path.exists(path):
        continue
    txt = open(path, encoding="utf-8").read()
    n = txt.count("get_bonus_stats")
    p(f"   {fn:<20} get_bonus_stats call sites: {n}"
      + ("   <-- PADDS STATS" if n else ""))
p("")
p("  exporter.py:1682 — 'Checkpoint 7: soul bonus stats disabled ...")
p("  no artificial padding.' The production path honours that; the scratch")
p("  runner does not. Same player, different numbers, per runner.")

# ── sample league ─────────────────────────────────────────────────────────
rule()
p("K. A SAMPLE GENERATED LEAGUE (what the numbers actually look like)")
rule()
random.seed(1234)
np.random.seed(1234)
POS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
SPEC = list(PD.ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP.keys())
rows_ = []
for i in range(400):
    pos = POS[i % len(POS)]
    age = random.randint(18, 37)
    dna = PD.DNAFactory.create(f"P{i}", pos, [random.choice(SPEC)], age=age)
    rows_.append((dna, age))

def stats(fn):
    v = np.array([fn(d) for d, _ in rows_], dtype=float)
    return v.min(), np.percentile(v, 50), v.max()

for label, fn in (
    ("overall_rating", lambda d: d.overall_rating),
    ("finishing", lambda d: d.technical.finishing),
    ("pace", lambda d: d.physical.pace),
    ("tackling", lambda d: d.defending.tackling),
    ("composure", lambda d: d.mental.composure),
    ("short_passing", lambda d: d.passing.short_passing),
    ("geometric_awareness", lambda d: d.mental.geometric_awareness),
):
    lo, md, hi = stats(fn)
    p(f"  {label:<22} min {lo:>6.1f}   median {md:>6.1f}   max {hi:>6.1f}"
      + ("   <-- CONSTANT" if lo == hi else ""))
p("")
ga = {d.mental.geometric_awareness for d, _ in rows_}
p(f"distinct geometric_awareness values across 400 players: {sorted(ga)}")
p("")
# per-position overall spread — is one position systematically higher?
p(f"  {'pos':<5} {'n':>3}  {'overall median':>14}  {'min':>6}  {'max':>6}")
for pos in POS:
    v = np.array([d.overall_rating for d, a in rows_ if d.position == pos], dtype=float)
    if len(v):
        p(f"  {pos:<5} {len(v):>3}  {np.median(v):>14.1f}  {v.min():>6.1f}  {v.max():>6.1f}")
p("")
p("  >>> identical raw attributes do NOT give an identical overall across")
p("      positions: each position has its own weight set AND its own divisor.")

rule("=")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_dna_scale.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print(f"\n[written] {out}")

