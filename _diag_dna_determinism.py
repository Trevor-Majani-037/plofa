"""Negative control for tests/test_dna_determinism.py.

Restores each defect one at a time, runs the real suite, and asserts it
goes RED. Then restores player_dna.py byte-exactly.

Without this, "11 passed" is indistinguishable from "the tests assert
nothing". The project's own record says a silent mutation is worse than no
guard (_diag_dna_guard.py, mutation M2).

Gates, in order — a control that does not check its own lever is worse than
no control:
  1. the anchor must exist (a non-matching anchor mutates nothing and then
     reports a clean PASS)
  2. the mutant must compile
  3. the mutant must not be a no-op (source bytes must differ)
  4. the file must be restored byte-identically
"""
import io
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent          # probe lives in the repo root
SRC = ROOT / "player_dna.py"
BACKUP = ROOT / "_diag_dna_determinism_backup.py"
TESTS = [ROOT / "tests" / "test_dna_determinism.py",
         ROOT / "tests" / "test_soul_dna.py",
         ROOT / "tests" / "test_soul_wiring.py"]

MUTATIONS = [
    ("M1  _attr ignores the crc32 (back to the global stream)",
     "        if unit is None:\n            base = random.uniform(lo, hi)\n        else:\n            if lifted:",
     "        unit = None\n        if unit is None:\n            base = random.uniform(lo, hi)\n        else:\n            if lifted:"),

    ("M2  _attr drops the parity draw (stream shift = option (a))",
     "        _stream_parity_uniform(lo, hi)          # burn the global draw, ignore it\n",
     ""),

    ("M3  footedness reverts to a global coin flip",
     "            flip = _stream_parity_random() < 0.20 if _unit is None else _unit < 0.20",
     "            flip = random.random() < 0.20"),

    ("M4  identity never populated -> every player shares one DNA",
     '        _DNA_IDENTITY["name"] = name or ""',
     '        _DNA_IDENTITY["name"] = ""'),

    ("M5  seed keyed on the band instead of the player (twins everywhere)",
     '    z = (zlib.crc32(f"{name}|{key}".encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF',
     '    z = (zlib.crc32(f"{key}".encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF'),

    ("M6  crc32 replaced by hash() (PYTHONHASHSEED-salted again)",
     '    z = (zlib.crc32(f"{name}|{key}".encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF',
     '    z = (hash(f"{name}|{key}") % 1000003) + 0x9E3779B97F4A7C15'),

    # The clustering defect this control was extended for: dropping the
    # avalanche step is NOT caught by "is it stable" or "is it wide" — only
    # by a uniformity check. It shipped once already and was found by an
    # UNRELATED suite (test_dna_awareness_shots), not by this file.
    ("M7  avalanche step dropped (crc32 clusters on similar names)",
     '    z = (zlib.crc32(f"{name}|{key}".encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF\n'
     '    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF\n'
     '    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF\n'
     '    return (z ^ (z >> 31)) / 18446744073709551616.0',
     '    return zlib.crc32(f"{name}|{key}".encode("utf-8")) / 4294967296.0'),

    ("M8  name dropped from the seed (every player identical)",
     '    z = (zlib.crc32(f"{name}|{key}".encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF',
     '    z = (zlib.crc32(key.encode("utf-8")) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF'),

    # The soul wiring. All three were real defects during development:
    # M9 was a live bug (the talent bias applied even at zero lift, so a
    # QUALITY-tier soul was not identical to an ordinary player), and M11 was
    # found by a test rather than by reading.
    ("M9  talent bias applies even when lift is zero (QUALITY soul not ordinary)",
     "            if lifted:\n",
     "            if soul is not None:\n"),

    ("M10 soul never attached to dna (invisible to the exporter)",
     "            soul=soul,\n        )",
     "        )"),

    ("M11 the double-charge guard removed (99.0 pile-up returns)",
     "and _spec_applies(\"clinical_finisher\")",
     "and True"),

    # The one-file guard hole. tests/test_soul_dna.py's AST guard originally
    # walked auto_run_match.py only and would have stayed GREEN with
    # run_match.py unthreaded — the omission class this project has now hit
    # four times. A guard that protects one file cannot catch the class.
    ("M12 scratch runner stops threading souls (one-file guard hole)",
     "        soul_assignments=SOUL_PLAYERS,\n    )\n\n    away_squad = SquadBuilder.build(\n"
     "        team_name=AWAY_TEAM,\n        starters=AWAY_STARTERS,",
     "    )\n\n    away_squad = SquadBuilder.build(\n"
     "        team_name=AWAY_TEAM,\n        starters=AWAY_STARTERS,"),

    # ── The soul-wiring pass. Each of these was a real defect at some point
    #    during it, or is the exact thing a future edit could silently undo. ──
    ("M13 state modifiers stack multiplicatively again (press 1.17)",
     "        context_mult = min(self.CONTEXT_MAX, 1.0 + bonus * g)\n"
     "\n"
     "        # Flow state: peak psychological condition\n"
     "        if self.is_in_flow_state:\n"
     "            context_mult = min(self.CONTEXT_MAX, context_mult * 1.12)",
     "        context_mult = 1.0\n"
     "        for _f in (profile.losing_state_boost if game_context.get(\"is_losing\") else 1.0,\n"
     "                   profile.late_game_boost if game_context.get(\"is_late_game\") else 1.0,\n"
     "                   profile.pressure_resistance if game_context.get(\"under_pressure\") else 1.0):\n"
     "            context_mult *= _f\n"
     "        if self.is_in_flow_state:\n"
     "            context_mult *= 1.12"),

    ("M14 probability modifiers scale past their ceiling",
     "        lifted = base_prob + (ceiling - base_prob) * (mult - 1.0)\n"
     "        return max(0.0, min(ceiling, lifted))",
     "        return base_prob * mult"),

    ("M15 press modifier is commented out again",
     "        press_success_rate = _get_soul_applicator().modify_press_success(\n"
     "            presser, press_success_rate, state, pressing_team)",
     "        # press_success_rate = _get_soul_applicator().modify_press_success(\n"
     "        #     presser, press_success_rate, state, pressing_team)"),

    ("M16 bonus stat fabrication returns",
     "    # ── PROBABILITY-SAFE APPLICATION (added 2026-10-14) ──",
     "    @staticmethod\n"
     "    def apply_bonus_stats(player, stats, state=None):\n"
     "        stats['shot_assists'] = stats.get('shot_assists', 0) + 3\n"
     "        return stats\n\n"
     "    # ── PROBABILITY-SAFE APPLICATION (added 2026-10-14) ──"),

    ("M17 superstar selection goes back to market value only",
     "        picked = [p.name for p in sorted_p if p.name in soul_names]",
     "        picked = []"),

    ("M18 a redundant multiplier is reintroduced as decoration",
     "    attribute_ceilings: Dict[str, float] = field(default_factory=dict)",
     "    composure_mult: float = 1.55\n"
     "    attribute_ceilings: Dict[str, float] = field(default_factory=dict)"),
]


MUTATION_FILES = {
    "M12": ROOT / "run_match.py",
    "M15": ROOT / "event_chain.py",
    "M17": ROOT / "roster_loader.py",
    # The soul-wiring mutations all live in player_soul.py. Leaving them on the
    # default (player_dna.py) is why GATE 1 rejected M13/M14/M16/M18 four
    # times in a row while their anchors were provably present in the file I
    # thought I was editing. A wrong default target fails exactly like a stale
    # anchor, so GATE 1 cannot tell them apart — which is why the fix is to
    # name the file, not to loosen the gate.
    "M13": ROOT / "player_soul.py",
    "M14": ROOT / "player_soul.py",
    "M16": ROOT / "player_soul.py",
    "M18": ROOT / "player_soul.py",
}
def run_tests():
    r = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "pytest",
         *[str(x) for x in TESTS], "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    return r.returncode, (r.stdout or "")[-400:]


def dna_probe_sig():
    """A behavioural fingerprint of the DNA layer, from a fresh subprocess.

    Reports TWO axes, because a mutation can hit either one:
      * the DNA values themselves (M1, M3, M4, M5, M6)
      * where the GLOBAL random stream ends up (M2 — option (a)'s stream
        shift, which changes no rating at all)

    GATE 3b exists because bytes differing is NOT enough. The first version
    of M3 replaced the footedness expression with one that reduces to the
    original under Python's operator precedence, so the source changed while
    the behaviour did not, and the mutation reported GREEN. The first
    version of this fingerprint then rejected M2 for the opposite reason —
    M2 legitimately changes no rating. A control that measures one axis
    rejects the other axis's real defect.
    """
    code = (
        "import sys, random; sys.path.insert(0, %r)\n"
        "from player_dna import DNAFactory, _dna_unit, _DNA_IDENTITY\n"
        "from player_soul import PlayerSoul, GreatnessPillars, SoulArchetype\n"
        "def _soul(n,a,p):\n"
        "    return PlayerSoul(n, archetype=getattr(SoulArchetype,a),"
        " pillars=GreatnessPillars(hardwork=p[0],talent=p[1],luck=p[2]))\n"
        "random.seed(777)\n"
        "out=[]\n"
        "for i in range(12):\n"
        "    d=DNAFactory.create('P%%03d'%%i,'RW',['shooter','creator','big_game_player'],age=34,preferred_foot='left',is_superstar=(i%%3==0))\n"
        "    out.append((round(float(d.technical.finishing),4),round(float(d.mental.vision),4),"
        "round(float(d.overall_rating),4),d.footedness))\n"
        # soul axis: pillars must separate, a QUALITY soul must equal an
        # ordinary player, and the soul must be ATTACHED to the DNA
        "souls=[]\n"
        "for lbl,arch,pil in (('weak','ATTACKING_PROPHET',(0.70,0.70,0.70)),\n"
        "                      ('perc','ATTACKING_PROPHET',(0.99,0.90,0.99)),\n"
        "                      ('max','ATTACKING_PROPHET',(0.99,0.99,0.99)),\n"
        "                      ('sav','GOALSCORING_SAVANT',(0.99,0.97,0.90))):\n"
        "    so=_soul('S%%s'%%lbl,arch,pil)\n"
        "    for i in range(8):\n"
        "        d=DNAFactory.create('S%%s%%02d'%%(lbl,i),'RW',['shooter','creator'],age=28,soul=so)\n"
        "        souls.append((lbl,round(float(d.technical.finishing),3),"
        "round(float(d.mental.vision),3),d.soul is not None))\n"
        # and the double-charge axis\n"
        "dc=[]\n"
        "for i in range(120):\n"
        "    d=DNAFactory.create('DC%%03d'%%i,'ST',['clinical_finisher'],age=27)\n"
        "    dc.append(round(float(d.technical.finishing),3))\n"
        "# uniformity of the seed stream, on the cohort shape that clusters\n"
        "saved=_DNA_IDENTITY['name']\n"
        "counts=[0]*10\n"
        "try:\n"
        "    for i in range(300):\n"
        "        _DNA_IDENTITY['name']='D%%d'%%i\n"
        "        counts[min(9,int(_dna_unit('mental.geometric_awareness')*10))]+=1\n"
        "finally:\n"
        "    _DNA_IDENTITY['name']=saved\n"
        "print(repr((out, round(random.random(),12), tuple(counts),"
        " tuple(souls), tuple(dc))))\n" % str(ROOT)
    )
    r = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-c", code],
                       cwd=ROOT, capture_output=True, text=True)
    return (r.stdout or "").strip()


def main():
    original = SRC.read_bytes()
    BACKUP.write_bytes(original)
    print(f"backup: {BACKUP.name}  ({len(original)} bytes)\n")

    base_rc, _ = run_tests()
    base_sig = dna_probe_sig()
    print(f"BASELINE (unmutated) exit={base_rc}  "
          f"{'OK' if base_rc == 0 else 'ALREADY RED — control is useless'}")
    print(f"BASELINE behaviour fingerprint: {base_sig[:110]}...\n")
    if base_rc != 0:
        print("ABORT: the suite is red before any mutation.")
        return 1

    failures = []
    for label, old, new in MUTATIONS:
        # M12 lives in run_match.py, not player_dna.py
        target = MUTATION_FILES.get(label.split()[0], SRC)
        key = label.split()[0]
        # Read with newline="" so the file's own line endings survive, then
        # normalise to \n for MATCHING only. Without this, a source file saved
        # with CRLF never matches a hand-written \n anchor and GATE 1 rejects
        # every multi-line mutation — which is exactly what happened to M13,
        # M14, M16 and M18 on their first run.
        raw = target.read_bytes().decode("utf-8")
        src = raw.replace("\r\n", "\n")
        was_src = target == SRC

        if old not in src:
            print(f"  {label}\n      GATE 1 FAIL: anchor not found -> mutation would be a NO-OP")
            hint = old.splitlines()[0][:70] if old else ""
            print(f"                wanted to find: {hint!r}")
            failures.append(label)
            continue

        mutated = src.replace(old, new, 1)
        # Write back in the file's ORIGINAL line-ending style, so a mutation
        # never silently converts a CRLF file to LF (which is what made an
        # earlier run report a false "not restored" for run_match.py).
        eol = "\r\n" if "\r\n" in raw else "\n"
        target.write_bytes(mutated.replace("\n", eol).encode("utf-8"))

        # GATE 2 - must compile
        c = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m",
                            "py_compile", str(target)], cwd=ROOT, capture_output=True, text=True)
        if c.returncode != 0:
            print(f"  {label}\n      GATE 2 FAIL: does not compile")
            print("      " + (c.stderr or "").strip().splitlines()[-1])
            target.write_text(src, encoding="utf-8")
            failures.append(label)
            continue

        # GATE 3 - must not be a source no-op
        if mutated.encode("utf-8") == src.encode("utf-8"):
            print(f"  {label}\n      GATE 3 FAIL: source unchanged")
            target.write_text(src, encoding="utf-8")
            failures.append(label)
            continue

        # GATE 3b - must not be a BEHAVIOURAL no-op, measured where the
        # mutation lives. M12/M15/M17 are runner/wiring/selection changes that
        # the DNA-layer fingerprint is blind to by construction; they are
        # validated by the AST and structural guards instead, and said so.
        if was_src and key in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
                               "M9", "M10", "M11"):
            mut_sig = dna_probe_sig()
            if mut_sig == base_sig:
                print(f"  {label}\n      GATE 3b FAIL: source changed but behaviour is identical")
                print("                -> this is a rewritten no-op, not a real mutation")
                target.write_text(src, encoding="utf-8")
                failures.append(label)
                continue
            changed = "behaviour changed OK"
        else:
            changed = "runner wiring changed (AST guard is the instrument)"

        rc, tail = run_tests()
        caught = rc != 0
        print(f"  {label}\n      {changed}; suite "
              f"{'RED (caught)' if caught else 'GREEN (MISSED!)'}")
        if not caught:
            failures.append(label)
            tail_lines = tail.strip().splitlines()
            if tail_lines:
                print("      " + tail_lines[-1])

        target.write_bytes(raw.encode("utf-8"))

    # GATE 4 - byte-exact restore
    SRC.write_bytes(original)
    restored = SRC.read_bytes()
    ok = restored == original
    print(f"\nRESTORE: byte-identical = {ok}  ({len(restored)} bytes)")
    if not ok:
        failures.append("restore")

    rc, _ = run_tests()
    post_sig = dna_probe_sig()
    print(f"POST-RESTORE suite exit={rc}  {'OK' if rc == 0 else 'RED'}")
    print(f"POST-RESTORE behaviour restored = {post_sig == base_sig}")
    if post_sig != base_sig:
        failures.append("behaviour restore")
    BACKUP.unlink(missing_ok=True)

    print()
    if failures:
        print(f"RESULT: {len(failures)} PROBLEM(S): {failures}")
        return 1
    print(f"RESULT: all {len(MUTATIONS)} mutations caught, file restored. CONTROL PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
