"""STEP 3: the 7-match physics gate, one subprocess per match.

Why a subprocess per match
--------------------------
Measured with `probe_determinism` and `probe_leak` (2026-11-01/02):

* a FRESH interpreter reproduces a match exactly -- run 0 gave
  `sha 744a7a6f0141517c` in two independent processes, twice each;
* a SECOND match in the same process does not -- run 1 and run 2 diverge from
  run 0 *and from each other*, even with the global RNG reseeded and the brain
  registry, physics gate counters, numpy global state and match_engine's
  module-level brain singletons all reset.

So the engine is deterministic per process and something is leaking between
matches inside one process. Resetting every candidate we could name did not
stop it. Until the leak is found, the only honest way to compare two conditions
is to give every match a virgin interpreter -- which is what this does.

The control
-----------
The first fixture is run TWICE per condition. If the two runs disagree the
script refuses to print a result, because that would be the harness drifting
rather than the physics. (`measure_gate.py` learned this the hard way: it once
reported "passes -169" against 69 opportunities.)

Run:  $env:PYTHONHASHSEED="0"; python -m scripts.physics.gate_ab
      python -m scripts.physics.gate_ab --fixtures 7 --seed 123
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, r"D:\PLOFA\plofa")

#: Outfield of the 18 clubs the roster loader actually serves. Paired so that
#: no club meets itself, and strong/weak are spread across both conditions.
FIXTURES = [
    ("Uditon", "Claw"),
    ("Avada Zenith", "Trendboys"),
    ("Natrican", "Lige-8"),
    ("Pearls", "Oxton"),
    ("Play City", "Rodice"),
    ("Red Wolves", "Triumpher"),
    ("Seafcea", "Telbey"),
]


# ─────────────────────────────────────────────────────────────────────
# CHILD: one match, one condition, JSON out
# ─────────────────────────────────────────────────────────────────────
def run_one(home: str, away: str, on: bool, seed: int, out_path: str) -> int:
    import contextlib
    import io
    import random
    from datetime import date

    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    random.seed(seed)
    loader = get_loader()
    raw = {c: loader.build_matchday_squad(c) for c in (home, away)}
    squads = {c: SquadBuilder.build(
        team_name=c, starters=raw[c]["starters"],
        substitutes=raw[c]["substitutes"],
        team_superstars=raw[c]["superstars"],
        set_piece_takers=raw[c]["sp_takers"]) for c in (home, away)}
    profs = {c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == home))
             for c in (home, away)}

    cfg = MatchConfig(home_team=home, away_team=away,
                      match_date=date(2026, 9, 8), physics_enabled=on)
    eng = MatchEngine(cfg, profs[home], profs[away])
    for c in (home, away):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (home, away) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=subs, away_subs_bench=subs))

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        res = eng.simulate()

    totals = res.physics_totals()
    team_of = {}
    for c in (home, away):
        for r in eng.position_engine.snapshot(c):
            team_of[r["player"]] = c

    def _agg(club):
        km = sum(a["distance_m"] for n, a in totals.items()
                 if team_of.get(n) == club) / 1000.0
        sp = sum(a["sprint_count"] for n, a in totals.items()
                 if team_of.get(n) == club)
        return round(km, 2), round(sp, 1)

    payload = {
        "home": home, "away": away, "physics": bool(on), "seed": seed,
        "home_goals": int(res.home_goals), "away_goals": int(res.away_goals),
        "home_km": _agg(home)[0], "away_km": _agg(away)[0],
        "home_sprints": _agg(home)[1], "away_sprints": _agg(away)[1],
        "events": len(res.timeline),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return 0


# ─────────────────────────────────────────────────────────────────────
# PARENT: orchestrate, verify the control, print the gate
# ─────────────────────────────────────────────────────────────────────
def _spawn(home, away, on, seed):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="gate_")
    os.close(fd)
    try:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED", "0")
        cmd = [sys.executable, "-m", "scripts.physics.gate_ab",
               "--one", home, away, "on" if on else "off", str(seed), path]
        proc = subprocess.run(cmd, cwd=r"D:\PLOFA\plofa", env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"child failed for {home} v {away} physics={on}\n"
                f"{proc.stderr[-2000:]}")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main(argv) -> int:
    if "--one" in argv:
        i = argv.index("--one")
        home, away, cond, seed, path = argv[i + 1:i + 6]
        return run_one(home, away, cond == "on", int(seed), path)

    n = 7
    seed = 123
    if "--fixtures" in argv:
        n = int(argv[argv.index("--fixtures") + 1])
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])
    fixtures = FIXTURES[:max(1, min(n, len(FIXTURES)))]

    print(f"physics gate   fixtures={len(fixtures)}   seed={seed}")
    print("one subprocess per match (in-process reuse is not reproducible)")
    print()

    # ── control: fixture 0, both conditions, twice ──────────────────
    home, away = fixtures[0]
    ctl = {}
    for on in (False, True):
        a = _spawn(home, away, on, seed)
        b = _spawn(home, away, on, seed)
        if a != b:
            print("CONTROL FAILED -- the same match twice, same seed, different"
                  " result:")
            print("  ", a)
            print("  ", b)
            print("Refusing to present a gate on a harness that is drifting.")
            return 2
        ctl[on] = a
    print(f"control OK: {home} v {away} reproduces in both conditions")
    print()

    rows = []
    for idx, (home, away) in enumerate(fixtures):
        off = ctl[False] if idx == 0 else _spawn(home, away, False, seed)
        on = ctl[True] if idx == 0 else _spawn(home, away, True, seed)
        rows.append((home, away, off, on))
        print(f"  {home} v {away}:  off {off['home_goals']}-{off['away_goals']}"
              f"   on {on['home_goals']}-{on['away_goals']}")

    print()
    hdr = f"{'':<26}{'OFF':>8}{'ON':>8}{'delta':>8}"
    print(hdr)
    print("-" * len(hdr))

    def _sum(key):
        return 0

    off_g = sum(r[2]["home_goals"] + r[2]["away_goals"] for r in rows)
    on_g = sum(r[3]["home_goals"] + r[3]["away_goals"] for r in rows)
    off_km = sum(r[2]["home_km"] + r[2]["away_km"] for r in rows)
    on_km = sum(r[3]["home_km"] + r[3]["away_km"] for r in rows)
    off_sp = sum(r[2]["home_sprints"] + r[2]["away_sprints"] for r in rows)
    on_sp = sum(r[3]["home_sprints"] + r[3]["away_sprints"] for r in rows)

    print(f"{'goals':<26}{off_g:>8}{on_g:>8}{on_g - off_g:>+8}")
    print(f"{'distance (km)':<26}{off_km:>8.1f}{on_km:>8.1f}"
          f"{on_km - off_km:>+8.1f}")
    print(f"{'sprints':<26}{off_sp:>8.0f}{on_sp:>8.0f}{on_sp - off_sp:>+8.0f}")
    print()
    print(f"per-match distance: off {off_km / len(rows):.1f} km   "
          f"on {on_km / len(rows):.1f} km")
    print(f"per-match sprints : off {off_sp / len(rows):.0f}   "
          f"on {on_sp / len(rows):.0f}")
    print(f"per-match goals   : off {off_g / len(rows):.2f}   "
          f"on {on_g / len(rows):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
