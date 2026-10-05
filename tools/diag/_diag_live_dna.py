"""
_diag_live_dna.py — measure the two 2026-10-05 DNA fixes BEFORE claiming anything.

FIX 1  geometric_awareness is now generated in `_build_mental`.
        Three LIVE position_engine.py shape rules consumed it and were inert:
          :3292  awareness_bonus   (CAM pocket blend)  was exactly 0.0
          :3431  midfielder half-space coverage        needed >= 55, never fired
          :3519  attacker drift                        ran at flat 0.091
FIX 2  `_shot_on_target_prob` now reads `effective_finishing` /
        `effective_composure`, so form + fatigue reach shooting.

NO MATCH IS RUN HERE. Everything below is microseconds, and the whole point is
to answer the two questions that decide whether a match-level gate is even worth
paying for:

  Q1  does generating geometric_awareness actually reach the three gates, and
      how much does each rule's steering weight move?
  Q2  is the shot fix a real behaviour change, or a no-op with better code?

Q2 has a sharp answer that a single number would hide, so it is split by PATH:
the scratch roster and the production season path hydrate form differently, and
only one of them is affected.

Writes _diag_live_dna.txt next to itself.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import player_dna as PD  # noqa: E402

_LINES: list[str] = []


def p(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def rule(ch: str = "=") -> None:
    p(ch * 78)


# ── the three live gates, transcribed from position_engine.py ──────────────
# Reimplemented here DELIBERATELY, from the source text, and the values are
# printed next to the line numbers so a reader can check the transcription.
# A probe that silently reuses the engine's own function cannot detect that the
# gate moved; a probe that hard-codes the gate can. This is the same discipline
# as the _diag_matrix_sensitivity oracle lesson.
GATE_MID_MIN = 55.0   # position_engine.py:3431  `if m.geometric_awareness < 55.0: continue`
GATE_ATT_MIN = 50.0   # position_engine.py:3519  `if a.geometric_awareness < 50.0: continue`
def mid_factor(ga: float) -> float:   # :3434  min(1, (ga - 50) / 50)
    return max(0.0, min(1.0, (ga - 50.0) / 50.0))
def att_factor(ga: float) -> float:   # :3522  min(1, (ga - 45) / 55)
    return max(0.0, min(1.0, (ga - 45.0) / 55.0))
def cam_bonus(ga: float) -> float:    # :3292  min(0.12, (ga - 50) / 350)
    return max(0.0, min(0.12, (ga - 50.0) / 350.0))

# Before the fix, EVERY player read 50.0 (the dataclass default, never assigned).
# These are the two steering values the engine used for all of them.
BEFORE_MID_FACTOR = mid_factor(50.0)   # 0.0, and the gate rejects them first
BEFORE_ATT_FACTOR = att_factor(50.0)   # 0.0909...  <- the flat number AGENTS.md quotes
BEFORE_CAM_BONUS = cam_bonus(50.0)     # 0.0, exactly

# Which archetype templates have always specified a band for this field, and
# what they say. Read from the source, not from a hard-coded copy.
def _template_coverage() -> tuple[dict, int]:
    tmpl = PD.ArchetypeLibrary.ARCHETYPES
    named = {k: v["mental.geometric_awareness"]
             for k, v in tmpl.items()
             if "mental.geometric_awareness" in v}
    return named, len(tmpl)


# ─────────────────────────────────────────────────────────────────────────
rule()
p("A.  ARCHETYPE COVERAGE — who has always specified geometric_awareness")
rule()
named, total = _template_coverage()
p(f"  archetypes defined                : {total}")
p(f"  archetypes WITH the band          : {len(named)}   ({100.0*len(named)/total:.0f}%)")
p(f"  archetypes falling back to default: {total - len(named)}   ({100.0*(total-len(named))/total:.0f}%)")
p("")
p("  Every template that bothered to specify it treats it as ABOVE AVERAGE.")
p("  The lowest band in the whole library is 60:")
p("")
p(f"  {'archetype':<24} {'band':>12}")
for k, (lo, hi) in sorted(named.items(), key=lambda kv: kv[1][0]):
    p(f"  {k:<24} {f'({lo}, {hi})':>12}")
p("")
p("  Read that as a design statement: the templates assumed a real, mostly")
p("  above-50 trait. A 50.0 default was not a neutral placeholder, it was a")
p("  contradiction of every band above.")
p("")
# Which POSITIONS do the named archetypes actually reach? An archetype is
# reached from a specialty, not from a position, so this is about how much of a
# real XI the named bands can even touch.
spec_map = PD.ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP
reach: dict[str, int] = {}
for spec, arch in spec_map.items():
    reach[arch] = reach.get(arch, 0) + 1
p("  specialities mapping to a NAMED archetype (i.e. how often a real player")
p("  can reach a specified band at all):")
for arch in sorted(named, key=lambda a: -reach.get(a, 0)):
    p(f"    {arch:<24} {reach.get(arch, 0):>2} specialities")
unreached = [a for a in named if reach.get(a, 0) == 0]
p("")
p(f"  named archetypes with ZERO specialties pointing at them: "
  f"{len(unreached)} {unreached if unreached else ''}")
p("")
p("  >>> so the DEFAULT BAND carries most of the population. The choice of")
p("      default band is therefore the real decision in fix 1, and section C")
p("      measures what it does to the gates.")


# ─────────────────────────────────────────────────────────────────────────
rule()
p("B.  THE GENERATED DISTRIBUTION — 400 players, real builder, real archetypes")
rule()
random.seed(1234)
np.random.seed(1234)
POS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
SPEC = list(PD.ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP.keys())
rows = []
for i in range(400):
    pos = POS[i % len(POS)]
    age = random.randint(18, 37)
    dna = PD.DNAFactory.create(f"P{i}", pos, [random.choice(SPEC)], age=age)
    rows.append((dna, age))

ga = np.array([d.mental.geometric_awareness for d, _ in rows], dtype=float)
p(f"  geometric_awareness   min {ga.min():.1f}   p25 {np.percentile(ga,25):.1f}"
  f"   median {np.median(ga):.1f}   p75 {np.percentile(ga,75):.1f}   max {ga.max():.1f}")
p(f"  distinct values       {len(set(ga.round(3)))}   (before the fix: 1, the constant 50.0)")
p("")
p("  for scale, the other mental attributes on the same 400 players:")
for label, fn in (("vision", lambda d: d.mental.vision),
                  ("positioning", lambda d: d.mental.positioning),
                  ("anticipation", lambda d: d.mental.anticipation),
                  ("work_rate", lambda d: d.mental.work_rate),
                  ("composure", lambda d: d.mental.composure)):
    v = np.array([fn(d) for d, _ in rows], dtype=float)
    p(f"    {label:<16} min {v.min():>5.1f}  median {np.median(v):>5.1f}  max {v.max():>5.1f}")
p("")
p("  geometric_awareness is NOT an outlier: it lands inside the same band the")
p("  rest of the mental block occupies, which is what a missing field should")
p("  look like once filled, not a new axis nothing else moves on.")


# ─────────────────────────────────────────────────────────────────────────
rule()
p("C.  DOES IT REACH THE THREE GATES?  (the actual claim)")
rule()
p("")
p("  C1. midfielder half-space coverage, position_engine.py:3431  gate >= 55")
p(f"      {'pos':<6} {'n':>4} {'clear gate':>11} {'factor before':>14} {'factor after':>13}")
for pos in ("CDM", "CM", "CAM"):
    v = np.array([d.mental.geometric_awareness for d, _ in rows if d.position == pos], dtype=float)
    if not len(v):
        continue
    clear = float((v >= GATE_MID_MIN).mean()) * 100.0
    fa = np.array([mid_factor(x) for x in v], dtype=float)
    p(f"      {pos:<6} {len(v):>4} {clear:>10.0f}% {BEFORE_MID_FACTOR:>14.3f} {np.median(fa):>13.3f}")
p("")
p("      'factor before' is 0.000 for all of them because the gate rejects a")
p("      50.0 player outright — this rule has never moved a single midfielder")
p("      in this codebase's history.")
p("")
p("  C2. attacker drift, position_engine.py:3519  gate >= 50")
p(f"      {'pos':<6} {'n':>4} {'clear gate':>11} {'factor before':>14} {'factor after':>13}")
for pos in ("LW", "RW", "ST", "CF", "CAM"):
    v = np.array([d.mental.geometric_awareness for d, _ in rows if d.position == pos], dtype=float)
    if not len(v):
        continue
    clear = float((v >= GATE_ATT_MIN).mean()) * 100.0
    fa = np.array([att_factor(x) for x in v], dtype=float)
    p(f"      {pos:<6} {len(v):>4} {clear:>10.0f}% {BEFORE_ATT_FACTOR:>14.3f} {np.median(fa):>13.3f}")
p("")
p(f"      the flat {BEFORE_ATT_FACTOR:.3f} is the number AGENTS.md already records")
p("      ('awareness_factor = 0.091 for all'). It is now a real per-player")
p("      spread rather than a constant — which is the point of the fix, and")
p("      also the reason the behaviour will change.")
p("")
p("  C3. CAM pocket blend weight, position_engine.py:3292  (0.40 + bonus)")
p(f"      {'pos':<6} {'n':>4} {'bonus before':>14} {'bonus after':>13} {'blend after':>13}")
v = np.array([d.mental.geometric_awareness for d, _ in rows if d.position == "CAM"], dtype=float)
cb = np.array([cam_bonus(x) for x in v], dtype=float)
p(f"      {'CAM':<6} {len(v):>4} {BEFORE_CAM_BONUS:>14.3f} {np.median(cb):>13.3f}"
  f" {0.40 + np.median(cb):>13.3f}")
p("")
p(f"      cap is 0.12, so the blend ranges 0.40 -> 0.52. Before it was")
p(f"      exactly {0.40 + BEFORE_CAM_BONUS:.2f} for every player ever created.")
p("")
p("  C4. HOW MANY players clear NEITHER gate (GK, CB, LB, RB were never")
p("      consumers — the three rules are midfielder/attacker/CAM rules):")
for pos in ("GK", "CB", "LB", "RB"):
    v = np.array([d.mental.geometric_awareness for d, _ in rows if d.position == pos], dtype=float)
    if len(v):
        p(f"      {pos:<5} median {np.median(v):>5.1f}  — now varies, but nothing reads it here yet")


# ─────────────────────────────────────────────────────────────────────────
rule()
p("D.  THE SHOT FIX — is it a behaviour change, and on WHICH path?")
rule()
p("")
p("  multiplier product = form.form_multiplier * form.fatigue_multiplier")
p("                      * live_performance_mult")
p("")
p("  D1. live_performance_mult has NO WRITER in the repository.")
p("      Grep finds: the dataclass default (1.0) and tests/test_defensive_")
p("      awareness.py:672. It is inert for every consumer of every effective_*")
p("      property, including get_shooter_quality. So the user's 'live stamina'")
p("      contributes exactly 1.0 — before and after this fix.")
p("")
p("  D2. the two paths that DO hydrate form differ, so the fix is not uniform:")
p("")
p(f"      {'path':<38} {'confidence':>11} {'fatigue':>9} {'product':>9} {'shot input':>11}")
cases = [
    ("scratch roster (run_match/validate_*)", 50.0, 0.0),
    ("season, neutral (confidence 50)", 50.0, 15.0),
    ("season, hot (confidence 85)", 85.0, 15.0),
    ("season, cold (confidence 25)", 25.0, 15.0),
    ("season, exhausted (start stamina 70)", 50.0, 30.0),
    ("season, best case (conf 100, fresh)", 100.0, 0.0),
    ("season, worst case (conf 0, fatigued)", 0.0, 30.0),
]
scratch_product = None
for label, conf, fat in cases:
    form_m = 0.80 + (conf / 100.0) * 0.40
    fat_m = max(0.85, 1.0 - (fat / 100.0) * 0.15)
    prod = form_m * fat_m
    if scratch_product is None:
        scratch_product = prod
    p(f"      {label:<38} {conf:>11.1f} {fat:>9.1f} {prod:>9.4f} {prod:>10.4f}x")
p("")
p(f"      scratch product {scratch_product:.4f} == 1.0 exactly, so the scratch")
p("      path is BYTE-IDENTICAL before and after. Every A/B harness in this")
p("      project (validate_neural_xl, compare_striker, all _diag_* probes)")
p("      runs the scratch roster, so none of their historical numbers move.")
p("")
p("      This is PROVEN, not assumed. `confidence` is written in exactly one")
p("      live place outside tests — season_manager.py:590, which hydrates it")
p("      from persisted season state. The method that would move it DURING a")
p("      match, PlayerFormState.update_after_match (player_dna.py:212), has")
p("      ZERO call sites; season_manager.py:632 reimplements the same rating")
p("      -> confidence mapping inline instead. So on a scratch match confidence")
p("      is 50.0 from player creation to final whistle, and the multiplier")
p("      product is exactly 1.0 for every shot. That is another instance of the")
p("      standing pathology — a documented method that looks like the mechanism")
p("      and is never called — recorded, not fixed here.")
p("")
p("      Worth stating plainly rather than celebrating: it also means NO A/B in")
p("      this project can demonstrate this fix. The gate has to be a unit test")
p("      (done, tests/test_dna_awareness_shots.py, negative-controlled by")
p("      _diag_dna_guard.py) plus this static call-site argument.")
p("")
p("  D3. magnitude on the production path. auto_run_match.py:618 clamps")
p("      starting stamina to [70, 100] and :622 maps it to fatigue_level, so")
p("      fatigue can only ever reach 30 -> fatigue_multiplier floor 0.955.")
p("      Form is the real lever at +/-20%.")
p("")
p("  D4. what that does to the shot probability. A realistic profile:")
p("      xg 0.12, composure 70, finishing 75 -> base 0.278 before the attribute")
p("      term, +0.0725 from (comp+fin)*0.05.")
xg, comp0, fin0 = 0.12, 70.0, 75.0
b = 0.20 + xg * 0.65 + (comp0 + fin0) / 100.0 * 0.05
p(f"      base = 0.20 + {xg}*0.65 + ({comp0}+{fin0})/100*0.05 = {b:.4f}")
p("")
p(f"      {'state':<38} {'comp':>6} {'fin':>6} {'base':>8} {'delta':>8}")
for label, conf, fat in cases:
    comp = comp0 * (0.80 + conf/100.0*0.40) * max(0.85, 1.0 - fat/100.0*0.15)
    fin = fin0 * (0.80 + conf/100.0*0.40) * max(0.85, 1.0 - fat/100.0*0.15)
    bb = 0.20 + xg * 0.65 + (comp + fin) / 100.0 * 0.05
    p(f"      {label:<38} {comp:>6.1f} {fin:>6.1f} {bb:>8.4f} {bb-b:>+8.4f}")
p("")
p("      The attribute term is 0.05 weight on the SUM of two 0-100 values, so")
p("      a +/-20% form swing moves the term by +/-0.0072 and `base` by ~2.6%")
p("      relative — a real but deliberately small effect, which is the right")
p("      size for a state modifier and NOT a reason to say the fix is cosmetic.")
p("      It is also the same order as the soul multiplier's own contribution.")


# ─────────────────────────────────────────────────────────────────────────
rule()
p("E.  WHAT IS STILL NOT FIXED, printed here rather than left implicit")
rule()
p("")
p("  1. THREE writers of form state are missing, and all three are the reason")
p("     this fix cannot be seen on a pitch:")
p("     (a) live_performance_mult has no writer at all — so 'live stamina' is")
p("         a constant 1.0 for EVERY effective_* consumer, including the")
p("         ones that shipped before this pass. Not invented here: writing it")
p("         means giving SubstitutionController.stamina a per-player,")
p("         per-minute hook into the DNA, which is a real subsystem change.")
p("     (b) PlayerFormState.update_after_match has zero call sites, so form")
p("         cannot move WITHIN a match anywhere.")
p("     (c) only the production season path writes fatigue before kickoff")
p("         (auto_run_match.py:622), and it clamps starting stamina to")
p("         [70, 100], so fatigue is capped at 30 -> multiplier floor 0.955.")
p("     Together these mean the scratch roster sees this fix as identity and")
p("     the production path sees at most +/-20% from hydrated form.")
p("")
p("  2. geometric_awareness now varies, so the three rules will move real")
p("     players for the first time. Two of them (midfielder half-space at")
p("     :3431, attacker drift at :3519) steer the SHAPE, and the unresolved")
p("     width-collapse tail lives in the same shape family. A match-level")
p("     before/after on team width is the honest next gate, and it costs two")
p("     matches. NOT RUN WITHOUT ASKING.")
p("")
p("  3. THE DEFAULT BAND, and the one place I was wrong first.")
p("     First attempt: (55, 74), on the argument that the eight named template")
p("     bands are 60-93 so the author must have meant an above-average trait,")
p("     and that work_rate (55, 74) was the closest precedent. Measurement")
p("     killed it: the floor sat exactly ON the :3431 gate floor, so 100% of")
p("     400 players cleared BOTH gates. The rules stopped selecting anyone and")
p("     became uniform rules — the same inertness as the 50.0 constant, aimed")
p("     the other way. Shipped band is (50, 70), the house default five of this")
p("     builder's other ten mental fields already use, which puts the floor")
p("     below both gates so they scale continuously instead of all-or-nothing.")
p("     The lesson worth keeping: a rule's GATE is a statement about the")
p("     distribution it was written against, and generating the distribution is")
p("     therefore a coupled change. You cannot fill in the attribute and leave")
p("     the thresholds as they were without re-deriving what they mean.")


out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_live_dna.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_LINES) + "\n")
print(f"\n[written] {out}")
