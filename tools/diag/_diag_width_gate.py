"""DID THE 2026-10-05 `geometric_awareness` FIX WIDEN THE COLLAPSE TAIL?

Context. AGENTS.md's WIDTH section records the user's actual complaint — the
shape collapses into a narrow block and should sit in a 45-55 m band — as
UNRESOLVED, with a tail (team-width p05 4.2-6.6 m, p10 16-18 m) that was never
isolated. Fix 1 then made `position_engine.py:3519` (attacker drift) a real
per-player factor instead of a flat 0.091, and that rule steers the SAME axis
as `winger_behavior.should_cut_inside`, which already pulls a wide player
inward. Inward is the direction of the collapse. So the fix may have made the
user's original complaint WORSE, and nothing in the unit suites can see that.

Two traps this probe is built to avoid, both of which have produced false
findings in this project before.

  1. **THE MUTATION DESYNCS THE RNG STREAM.** `PlayerDNA._attr` (player_dna.py:
     1063) calls `random.uniform(lo, hi)` — it DRAWS from the global football
     stream. The pre-fix builder never called `_attr` for
     `geometric_awareness`, so naively replacing the line with `= 50.0` removes
     one draw per player per match and every later number in the match shifts.
     An A/B built that way measures STREAM SHIFT, not awareness — the same
     class as `_STREAM_PARITY_DRAW`, and the reason it exists at all. So the
     pre-fix arm here still makes the identical `_attr` call and DISCARDS the
     result. The two arms therefore differ in exactly one thing: whether the
     drawn value is used. `parity_ok` below proves both arms took that draw.

  2. **A TAIL NEEDS SAMPLES, NOT MATCHES.** `_diag_wide_channel.py` samples
     `_offball_run` — ~409 times a match, and AGENTS.md already records its
     absolutes as disagreeing with the dense instrument by 2-3 m. A p05 over
     409 frames is not a p05. This probe uses the DENSE instrument
     (`PositionEngine.live_spacing_redirect`, ~300k calls a match, carrying the
     finished target of the whole chain) that AGENTS.md names as the one to
     quote medians from, with tick-identity pairing (never list index — the
     documented mispairing trap).

`lever_ok` is the liveness proof, and it is checked BEFORE any verdict is
printed: if the two arms produce the same frame digest the mutation never
reached the pitch and every number below is an artefact.

Run:  .venv\\Scripts\\python.exe _diag_width_gate.py [seed]
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import zlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
DNA = os.path.join(HERE, "player_dna.py")

FIND = ('            geometric_awareness = cls._attr(arch, '
        '"mental.geometric_awareness", (50, 70), mental_age),')

# The pre-fix state, WITH STREAM PARITY. `PlayerDNA._attr` draws from the global
# football stream, and the pre-fix builder never called it for this field, so a
# naive `= 50.0` removes one draw per player and shifts every later number in the
# match -- an A/B built that way measures stream shift, not awareness.
#
# The draw must therefore still HAPPEN. `_build_mental` builds `MentalAttributes`
# as a constructor ARGUMENT LIST, so a bare statement cannot be inserted here --
# the first version of this replacement was exactly that and did not compile. The
# lambda evaluates the call (consuming the draw), receives the value, and returns
# the constant: identical stream position, identical pre-fix behaviour.
REPL = ('            geometric_awareness = (lambda _v: 50.0)('
        'cls._attr(arch, "mental.geometric_awareness", (50, 70), mental_age)),'
        '  # PRE-FIX ARM: draw, discard, stream stays aligned')

WIDE = ("LW", "RW", "LB", "RB")

# The child dispatch MUST happen before argparse runs. The first version built
# the parent parser at module level, so a child invocation
# (`--child pre 777 out.json`) was parsed by the PARENT's grammar and died on
# `argument seed: invalid int value: 'pre'` -- the arm never ran and no match was
# simulated. It aborted loudly rather than reporting a wrong number, which is the
# behaviour you want, but it is still a bug: a probe that cannot reach its own
# measurement has measured nothing.
#
# `A` is only parsed when this file is the __main__ script, so the module is
# importable. That is not cosmetic: `repo_digest` was smoke-tested by importing
# it, and a module-level parse_args makes every import a parse_args against the
# importer's argv.
_IS_CHILD = len(sys.argv) > 1 and sys.argv[1] == "--child"
_IS_MAIN = __name__ == "__main__"

ap = argparse.ArgumentParser()
ap.add_argument("seed", nargs="?", type=int, default=777)
# `_IS_MAIN and not _IS_CHILD`, not `_IS_MAIN` alone: a child invocation is also
# __main__ (it runs as a script), so the parent grammar fired on the child's
# argv again and died on `invalid int value: 'pre'`. Guarding only on
# _IS_MAIN fixed the import and re-broke the dispatch, in two separate edits.
A = ap.parse_args() if (_IS_MAIN and not _IS_CHILD) else None

_LINES: list[str] = []


def p(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def rule(ch: str = "=") -> None:
    p(ch * 96)


# ─────────────────────────────────────────────────────────────────────────
# CHILD: one match, instrumented, metrics -> JSON
# ─────────────────────────────────────────────────────────────────────────
def child(arm: str, seed: int, out_path: str) -> None:
    sys.path.insert(0, HERE)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    import match_engine as ME
    from position_engine import PositionEngine
    from _diag_chance_coords import build_pair

    # tick -> {pos: (ty_target, cy_now, has_ball, in_block)}
    ticks: dict[int, dict] = defaultdict(dict)
    stretch_w: list[float] = []
    run_hit: list[float] = []
    ga: dict[str, float] = {}
    calls = [0]
    wide_calls = [0]

    _red = PositionEngine.live_spacing_redirect
    _str_raw = PositionEngine.wide_stretch_blend
    _runs_raw = ME.MatchEngine._striker_runs

    def _red_p(self, team, cx, cy, tx, ty):
        out = _red(self, team, cx, cy, tx, ty)
        calls[0] += 1
        f = sys._getframe(1).f_locals
        pos = f.get("pos")
        if f.get("pname") and pos in WIDE:
            wide_calls[0] += 1
            # `_offball_tick_seq` is a plain int ATTRIBUTE (match_engine.py:1990),
            # NOT a method. `_diag_wide_target.py:_tickid` CALLS it, so `int()`
            # raises TypeError, its own `except Exception: return -1` swallows it,
            # and every wide player in the match collapses into ONE tick group --
            # which is how the first run of this probe reported `wide-ticks=1` and
            # a p05 over a single overwriting dict. A swallowed exception that
            # silently degrades an instrument is worse than a crash.
            tid = getattr(f.get("self"), "_offball_tick_seq", None)
            tid = int(tid) if isinstance(tid, int) else -1
            ticks[tid][pos] = (float(out[1]), float(cy),
                               bool(f.get("has_ball")),
                               bool(f.get("_in_block")))
        return out

    def _str_p(self, player_name, ball_y):
        w = _str_raw(self, player_name, ball_y)
        f = sys._getframe(1).f_locals
        if f.get("pname") and f.get("pos") in WIDE:
            stretch_w.append(float(w))
        return w

    def _runs_p(self, team, ball_x, ball_y, has_ball):
        r = _runs_raw(self, team, ball_x, ball_y, has_ball)
        f = sys._getframe(1).f_locals
        if f.get("pname") and f.get("pos") in WIDE and f.get("pname") in r:
            run_hit.append(1.0)
        return r

    PositionEngine.live_spacing_redirect = _red_p
    PositionEngine.wide_stretch_blend = _str_p
    ME.MatchEngine._striker_runs = _runs_p

    eng = build_pair("Oxton", "Natrican")
    res = eng.simulate()
    pe = eng.position_engine

    pos_of: dict[str, str] = {}
    for team in (res.config.home_team, res.config.away_team):
        for nm in pe.team_rosters.get(team, []):
            st = pe.states.get(nm)
            if st is not None:
                pos_of[nm] = st.position
                try:
                    p = (res.squads.get(team, {}).get("starters") or [])
                    pa = next((q for q in p if getattr(q, "name", "") == nm), None)
                    ga[nm] = float(pa.dna.mental.geometric_awareness)
                except Exception:
                    pass

    # frame digest — liveness proof. Both flanks of both wide pairs, every tick.
    _buf = []
    for _tid, _row in sorted(ticks.items()):
        _cells = "|".join("{0}={1:.3f},{2:.3f}".format(k, v[1], v[0])
                          for k, v in sorted(_row.items()))
        _buf.append("{}:{};".format(_tid, _cells).encode())
    h = zlib.crc32(b"".join(_buf))

    def pct(v, q):
        if not v:
            return 0.0
        s = sorted(v)
        return s[min(len(s) - 1, int(q / 100.0 * len(s)))]

    sep = {k: [] for k in ("tgt_w", "now_w", "tgt_fb", "now_fb")}
    sep_phase: dict[tuple, list] = defaultdict(list)
    per_pos: dict[str, list] = defaultdict(list)
    per_pos_tgt: dict[str, list] = defaultdict(list)
    for tid, row in ticks.items():
        for grp, tkey, nkey in ((("LW", "RW"), "tgt_w", "now_w"),
                                (("LB", "RB"), "tgt_fb", "now_fb")):
            m = {k: row[k] for k in grp if k in row}
            if len(m) != 2:
                continue
            t = abs(m[grp[0]][0] - m[grp[1]][0])
            n = abs(m[grp[0]][1] - m[grp[1]][1])
            sep[tkey].append(t)
            sep[nkey].append(n)
            hb = m[grp[0]][2]
            blk = m[grp[0]][3]
            key = ("IN POSS" if hb else ("OUT blk" if blk else "OUT open"),
                   "wingers" if grp[0] == "LW" else "fullbacks")
            sep_phase[key].append(n)
        for posn, (ty, cy, _hb, _blk) in row.items():
            per_pos[posn].append(min(cy, 68.0 - cy))
            per_pos_tgt[posn].append(min(ty, 68.0 - ty))

    payload = {
        "arm": arm,
        "seed": seed,
        "events": len(res.timeline),
        "digest": f"{h:08x}",
        "frame_hash": f"{h:08x}",
        "ticks_with_wide": len(ticks),
        "calls": calls[0],
        "wide_calls": wide_calls[0],
        "sep_n": {k: len(v) for k, v in sep.items()},
        "sep": {k: {"n": len(v), "p05": pct(v, 5), "p10": pct(v, 10),
                    "p50": pct(v, 50), "p90": pct(v, 90),
                    "mean": (sum(v) / len(v)) if v else 0.0,
                    "band": (100.0 * sum(1 for x in v if 45.0 <= x <= 55.0) / len(v))
                    if v else 0.0} for k, v in sep.items()},
        "sep_phase": {f"{k[0]}|{k[1]}": {"n": len(v), "p05": pct(v, 5),
                                         "p10": pct(v, 10), "p50": pct(v, 50),
                                         "band": (100.0 * sum(1 for x in v
                                                             if 45.0 <= x <= 55.0) / len(v))
                                         if v else 0.0}
                      for k, v in sorted(sep_phase.items())},
        "per_pos_actual": {k: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90)}
                           for k, v in sorted(per_pos.items())},
        "per_pos_target": {k: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90)}
                           for k, v in sorted(per_pos_tgt.items())},
        "ck35_active": (100.0 * sum(1 for w in stretch_w if w > 0.0)
                        / max(len(stretch_w), 1)),
        "ck35_samples": len(stretch_w),
        "run_target_pct": 100.0 * len(run_hit) / max(len(stretch_w), 1),
        "ga": {k: round(v, 1) for k, v in sorted(ga.items())},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


# ─────────────────────────────────────────────────────────────────────────
# PARENT: patch -> child -> restore -> compare
# ─────────────────────────────────────────────────────────────────────────
_IMPORT_CLOSURE: list[str] | None = None


def _import_closure() -> list[str]:
    """Repo-root module files the match path can reach, found STATICALLY.

    Two earlier attempts, both wrong in instructive ways:

      * hashing every `.py` in the repo root -- aborted because I edited a
        PROBE the match never imports. Too broad. A guard that fires on
        irrelevant edits gets switched off, and then it fires on nothing.

      * `import match_engine` + `sys.modules` -- returns 23 files and MISSES
        `event_chain.py`, `player_dna.py` and `attacking_matrix.py`, because
        those are imported lazily inside functions. That is the same "one-file
        hole" this project has now hit three times: the guard would have missed
        precisely the edits it exists to catch. A runtime closure is only
        truthful if taken from inside a real match, and paying a match to learn
        it would defeat the purpose.

    So: walk `import` / `from X import` statements transitively from
    `match_engine.py` over repo-root files. Static is an OVER-approximation of
    the runtime closure (it includes imports behind `if` branches), which is
    the safe direction for a guard. Costs microseconds, not a match.
    """
    global _IMPORT_CLOSURE
    if _IMPORT_CLOSURE is not None:
        return _IMPORT_CLOSURE
    root = ast.parse(open(os.path.join(HERE, "match_engine.py"),
                          encoding="utf-8").read())
    seen: set[str] = set()
    todo = ["match_engine.py"]
    while todo:
        fn = todo.pop()
        if fn in seen or not os.path.exists(os.path.join(HERE, fn)):
            continue
        seen.add(fn)
        try:
            tree = ast.parse(open(os.path.join(HERE, fn), encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                cand = m.split(".")[0] + ".py"
                if cand not in seen and os.path.exists(os.path.join(HERE, cand)):
                    todo.append(cand)
    del root
    _IMPORT_CLOSURE = sorted(seen)
    return _IMPORT_CLOSURE


def repo_digest(exclude=("player_dna.py",)):
    """A single number over every .py the MATCH PATH imports.

    The FIRST version of this probe ran both arms against the live working tree
    while a parallel session was editing `attacking_matrix.py`. The pre arm ran
    against one version and the post arm crashed against another, so the A/B
    was measuring somebody else's edit. "Documented" concurrency is not
    protection: the only thing that distinguishes an A/B from two unrelated
    matches is that the arms differ in the intended way and nothing else, and
    that has to be CHECKED rather than assumed.

    Scope is the engine's import closure, not the whole repo root -- see
    `_import_closure` for why that distinction is load-bearing.

    `exclude` is the set of files an arm is *supposed* to change.
    """
    # hashlib, NOT zlib.crc32: crc32 returns an int, so there is nothing to
    # `.update()` and it cannot be accumulated over a directory. Two probe
    # cycles went into that (zlib.crc32() with no seed is a TypeError; then
    # int.update is an AttributeError). Both cost zero matches -- the abort
    # happened before run_child -- but a probe that cannot reach its own
    # measurement has measured nothing.
    h = hashlib.sha256()
    for fn in sorted(_import_closure()):
        if fn in exclude:
            continue
        with open(os.path.join(HERE, fn), "rb") as fh:
            h.update(fn.encode())
            h.update(fh.read())
    return h.hexdigest()[:16]


def run_child(arm: str, seed: int) -> dict:
    out = os.path.join(HERE, f"_diag_width_gate_{arm}_{seed}.json")
    if os.path.exists(out):
        os.remove(out)
    cmd = [PY, os.path.abspath(__file__), "--child", arm, str(seed), out]
    pr = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if not os.path.exists(out):
        print(f"  !! child {arm} produced no output. stderr tail:\n"
              f"{pr.stdout[-1500:]}\n{pr.stderr[-1500:]}")
        raise SystemExit(2)
    return json.load(open(out, encoding="utf-8"))


def main() -> int:
    rule()
    p("A.  DID FIX 1 WIDEN THE NARROW-BLOCK TAIL?   (stream-parity A/B)")
    rule()
    p(f"  seed {A.seed} x 2 arms = 2 matches.  Real target band: 45-55 m.")
    p("  Pre-fix arm still draws from the RNG via _attr and DISCARDS the value,")
    p("  so the two arms differ ONLY in whether awareness is used.")
    p("")

    original = open(DNA, "rb").read()
    text = original.decode("utf-8")
    if FIND not in text:
        p("  !! ANCHOR NOT FOUND in player_dna.py — the file moved on.")
        p("     Not mutating: a non-matching anchor would mutate nothing and")
        p("     then report a clean PASS, which is the worst possible outcome.")
        return 2

    # PRE-FLIGHT: the mutation must COMPILE, and must actually leave the drawn
    # value discarded. Both of this probe's first two runs died here in one way
    # or another — a nested f-string would not parse, and then the replacement
    # was not valid inside a constructor argument list. Each failure cost a
    # cycle and, worse, would have looked like a missing JSON file rather than
    # a broken probe. Cheap check, run before any match is simulated.
    mutant = text.replace(FIND, REPL, 1)
    if mutant == text:
        p("  !! MUTATION WAS A NO-OP — REPL does not differ from what it replaced.")
        return 2
    try:
        compile(mutant, "player_dna.py", "exec")
    except SyntaxError as exc:
        p(f"  !! PRE-FIX ARM DOES NOT COMPILE: {exc}")
        p("     Refusing to simulate two matches against an arm that cannot load.")
        return 2
    p(f"  pre-flight: pre-fix arm compiles, mutation is a no-op: False")
    p("")

    data: dict[str, dict] = {}
    digests: dict[str, str] = {}
    try:
        for arm, src in (("pre", mutant), ("post", text)):
            open(DNA, "w", encoding="utf-8").write(src)
            digests[arm] = repo_digest()
            data[arm] = run_child(arm, A.seed)
            digests[arm + "_after"] = repo_digest()
            d = data[arm]
            print(f"  {arm:<5} events={d['events']:<5} "
                  f"live_spacing_redirect calls={d['calls']:<8} "
                  f"wide={d['wide_calls']:<8} paired ticks={d['ticks_with_wide']:<7}"
                  f" sep_n={d['sep_n']}  digest={d['digest']}")
    finally:
        open(DNA, "wb").write(original)

    # THE GUARD. Every .py except player_dna.py must be byte-identical across
    # the whole run, and player_dna.py must differ between the arms by exactly
    # the mutation. If a parallel session touched anything on the match path
    # while these matches were running, the arms are two different programs and
    # there is no verdict.
    all_d = sorted(set(digests.values()))
    stable = len(all_d) == 1
    p("")
    p(f"  guard scope: {len(_import_closure())} module files in the engine's "
      f"import closure (probes excluded)")
    p(f"  match-path digest: {', '.join(all_d)}  -> "
      f"{'STABLE' if stable else 'CHANGED MID-RUN'}")
    if not stable:
        p("")
        p("  >>> A PARALLEL SESSION EDITED A FILE ON THE MATCH PATH WHILE THESE")
        p("      MATCHES RAN. The two arms are different programs, so every")
        p("      difference below is partly that edit. NO VERDICT -- do not read")
        p("      the tables. Re-run when the tree is quiet.")
        return 2

    after = open(DNA, "rb").read()
    rule()
    p("B.  RESTORE + LEVER LIVENESS")
    rule()
    print(f"  player_dna.py byte-identical: {after == original}  "
          f"({len(after)} bytes)")
    if after != original:
        return 2
    ga_pre = sorted(set(data["pre"]["ga"].values()))
    ga_post = sorted(set(data["post"]["ga"].values()))
    p(f"  geometric_awareness  pre-arm : {ga_pre}")
    p(f"  geometric_awareness  post-arm: {ga_post}")
    lever = data["pre"]["digest"] != data["post"]["digest"]
    # The pre-fix arm must be the EXACT pre-fix state -- a flat 50.0 for every
    # player. Anything else means the mutation is not the thing being compared.
    flat = ga_pre == [50.0]
    parity = ga_pre != ga_post
    p(f"  pre-arm is exactly the pre-fix constant 50.0 : {flat}")
    p(f"  lever reached the pitch (digests differ)    : {lever}")
    p(f"  awareness value actually differs             : {parity}")
    if not flat:
        p("")
        p("  >>> THE PRE-FIX ARM IS NOT THE PRE-FIX STATE. Read its values above;")
        p("      the mutation did not land and no comparison here is valid.")
        return 2
    if not lever:
        p("")
        p("  >>> THE MUTATION NEVER REACHED THE PITCH. Every width number below")
        p("      would be an artefact. A probe that proves nothing is not evidence")
        p("      in either direction, so there is no verdict.")
        return 2
    if not parity:
        p("")
        p("  >>> Both arms produced the SAME awareness values: the anchor matched")
        p("      but the replacement did not take effect.")
        return 2

    # NON-VACUITY. A percentile over a collapsed tick grouping is not a
    # measurement, it is a number about one dictionary. This is exactly how the
    # first run reported `wide-ticks=1` and still had a table to print. If the
    # pairing produced too few groups to support a p05, say so and stop.
    _n = data["post"]["sep_n"]["now_w"]
    _t = data["post"]["ticks_with_wide"]
    p(f"  sample size: {data['post']['wide_calls']} wide-player calls across "
      f"{_t} paired ticks -> {_n} usable winger-pair observations per arm")
    if _t < 200 or _n < 200:
        p("")
        p("  >>> TOO FEW PAIRED TICKS FOR A PERCENTILE. A p05 needs a real")
        p("      distribution; this grouping is degenerate. No verdict.")
        return 2
    p("")

    rule()
    p("C.  SIDE-TO-SIDE SEPARATION (the 'spread play' claim)")
    rule()
    p(f"{'metric':<26} {'arm':<5} {'n':>7} {'p05':>7} {'p10':>7} {'p50':>7} "
      f"{'p90':>7} {'mean':>7} {'in45-55':>9}")
    p("-" * 96)
    for key, label in (("tgt_w", "wingers  TARGET"),
                       ("now_w", "wingers  ACTUAL"),
                       ("tgt_fb", "fullbacks TARGET"),
                       ("now_fb", "fullbacks ACTUAL")):
        for arm in ("pre", "post"):
            s = data[arm]["sep"][key]
            p(f"{label:<26} {arm:<5} {s['n']:>7} {s['p05']:>7.1f} "
              f"{s['p10']:>7.1f} {s['p50']:>7.1f} {s['p90']:>7.1f} "
              f"{s['mean']:>7.1f} {s['band']:>8.1f}%")
        dp5 = data["post"]["sep"][key]["p05"] - data["pre"]["sep"][key]["p05"]
        d50 = data["post"]["sep"][key]["p50"] - data["pre"]["sep"][key]["p50"]
        dband = (data["post"]["sep"][key]["band"]
                 - data["pre"]["sep"][key]["band"])
        p(f"{'':<26} {'DELTA':<5} {'':>7} {dp5:>+7.1f} {'':>7} {d50:>+7.1f} "
          f"{'':>7} {'':>7} {dband:>+8.1f}pp")
        p("-" * 96)

    rule()
    p("D.  BY PHASE — a low block legitimately narrows, so they are separate")
    rule()
    keys = sorted(set(data["pre"]["sep_phase"]) | set(data["post"]["sep_phase"]))
    p(f"{'phase|role':<24} {'arm':<5} {'n':>7} {'p05':>7} {'p10':>7} {'p50':>7} "
      f"{'in45-55':>9}")
    p("-" * 72)
    for k in keys:
        for arm in ("pre", "post"):
            s = data[arm]["sep_phase"].get(k)
            if not s:
                continue
            p(f"{k:<24} {arm:<5} {s['n']:>7} {s['p05']:>7.1f} {s['p10']:>7.1f} "
              f"{s['p50']:>7.1f} {s['band']:>8.1f}%")
        a = data["pre"]["sep_phase"].get(k) or {}
        b = data["post"]["sep_phase"].get(k) or {}
        if a and b:
            p(f"{'':<24} {'DELTA':<5} {'':>7} {b['p05']-a['p05']:>+7.1f} "
              f"{b['p10']-a['p10']:>+7.1f} {b['p50']-a['p50']:>+7.1f} "
              f"{b['band']-a['band']:>+8.1f}pp")
        p("-" * 72)

    rule()
    p("E.  DISTANCE TO NEAREST TOUCHLINE (real wide players sit 3-12 m off)")
    rule()
    p(f"{'pos':<5} {'arm':<5} {'n':>8} {'actual p50':>11} {'p90':>7} "
      f"{'TARGET p50':>12} {'p90':>7}")
    p("-" * 72)
    for posn in ("LW", "RW", "LB", "RB"):
        for arm in ("pre", "post"):
            a = data[arm]["per_pos_actual"].get(posn) or {}
            t = data[arm]["per_pos_target"].get(posn) or {}
            if not a:
                continue
            p(f"{posn:<5} {arm:<5} {a['n']:>8} {a['p50']:>11.1f} {a['p90']:>7.1f} "
              f"{t.get('p50', 0.0):>12.1f} {t.get('p90', 0.0):>7.1f}")
        p("-" * 72)

    rule()
    p("F.  THE TWO STRETCH/RUN TERMS (unchanged by fix 1 — printed as controls)")
    rule()
    for arm in ("pre", "post"):
        d = data[arm]
        p(f"  {arm:<5} CK35 stretch active {d['ck35_active']:>5.1f}% of "
          f"{d['ck35_samples']} samples   live run target present "
          f"{d['run_target_pct']:>5.1f}%")
    p("")
    p("  These two are the ONLY chain terms that can touch a wide role (CK36/37/38")
    p("  are midfielder/backline terms keyed by player name). If they are")
    p("  unchanged and the separation still moved, the movement came from")
    p("  somewhere else — and the place to look first is :3519 itself.")
    p("=" * 96)

    with open(os.path.join(HERE, "_diag_width_gate.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        sys.exit(main())
