"""
Where exactly does PLOFA stop being reproducible?
===============================================

**Result so far: the engine DOES replay. Same seed and same
``PYTHONHASHSEED`` gives a byte-identical timeline, across separate processes.
What it does not survive is a different ``PYTHONHASHSEED``** — 3,283 events
under seed 0, 3,024 under seed 1, an 8% difference in how much football gets
played. So the open "determinism bug" recorded in the physics report is
misdiagnosed: it is not a same-seed divergence, it is an *environment*
dependency.

That is a better problem than the one it was filed under, because it is
specific and findable — code that iterates a ``set`` of strings, or calls
``hash()`` on one. Both orders are stable within a process and change between
processes with a different hash seed.

This probe answers three questions, and the answers point at different bugs:

  * Diverges *within* a process across two runs with the seed reset
      -> state is leaking between runs. A module-level cache, a global counter,
         a set that was mutated. Something is not being reset.
  * Agrees *within* a process, diverges *across* fresh processes with the same
    ``PYTHONHASHSEED``
      -> ``id()`` or object-identity hashing. The allocator decides.
  * Agrees across processes at one ``PYTHONHASHSEED``, diverges at another
      -> ``hash()`` on strings, or ``set``/``frozenset`` iteration order.

Note on setting the seed: ``os.environ["PYTHONHASHSEED"] = "0"`` inside a
running interpreter has **no effect** — string hashing is fixed at startup. An
earlier version of this file did exactly that, which made it look like it was
controlling the hash seed when it was not. The value has to be set in the shell
before Python starts:

    $env:PYTHONHASHSEED="0"; python -m physics.probe_determinism

Run:  python physics/probe_determinism.py [runs] [seed]
"""
from __future__ import annotations

import hashlib
import os
import random
import sys

# Reported, not set. Setting it here would be a no-op and would read as though
# this script were controlling it.
HASH_SEED = os.environ.get("PYTHONHASHSEED")
if HASH_SEED is None:
    print("WARNING: PYTHONHASHSEED is not set in the environment. String")
    print("         hashing will be random per process, so any comparison")
    print("         across processes is meaningless until you fix it:")
    print('             $env:PYTHONHASHSEED="0"')
    print()

sys.path.insert(0, r"D:\PLOFA\plofa")

# Hardcoded squads, lifted from _repro.py. Using SquadBuilder directly rather
# than the roster loader keeps the loader's own caches out of the question —
# this is a test of the match engine, not of how a squad gets picked.
from _repro import (AWAY_STARTERS, AWAY_SUBS, AWAY_TEAM,
                    HOME_STARTERS, HOME_SUBS, HOME_TEAM)

sys.path.insert(0, r"D:\PLOFA\plofa")


def _squads():
    from player_dna import SquadBuilder
    return {
        HOME_TEAM: SquadBuilder.build(
            team_name=HOME_TEAM, starters=HOME_STARTERS,
            substitutes=HOME_SUBS, team_superstars=["Percy", "Dragan Novak"],
            set_piece_takers=["Percy", "Kofi Mensah"]),
        AWAY_TEAM: SquadBuilder.build(
            team_name=AWAY_TEAM, starters=AWAY_STARTERS,
            substitutes=AWAY_SUBS, team_superstars=["Kwame Asante"],
            set_piece_takers=["Kwame Asante", "Bruno Reis"]),
    }


def _profiles():
    from match_engine import (Intensity, MatchConfig, MatchEngine,  # noqa: F401
                              PlayingStyle, TeamProfile, TeamStyle)
    hpro = TeamProfile(name=HOME_TEAM, style=TeamStyle.ATTACKING,
                       playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    apro = TeamProfile(name=AWAY_TEAM, style=TeamStyle.FLUID_COUNTER,
                       playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM)
    return hpro, apro


def fingerprint(seed: int) -> list:
    """One short, comparable string per timeline event."""
    from datetime import date

    from match_engine import MatchConfig, MatchEngine

    random.seed(seed)
    squads = _squads()
    hpro, apro = _profiles()
    cfg = MatchConfig(home_team=HOME_TEAM, away_team=AWAY_TEAM,
                      match_date=date(2026, 9, 8))
    eng = MatchEngine(cfg, hpro, apro)
    eng.quiet = True
    for team in (HOME_TEAM, AWAY_TEAM):
        eng.set_squad(team, squads[team]["starters"], squads[team]["substitutes"])
    res = eng.simulate()
    return [
        "|".join(str(x) for x in (
            getattr(e, "minute", "?"),
            getattr(e, "second", "?"),
            getattr(getattr(e, "event_type", None), "name", "?"),
            getattr(e, "player", "") or "",
            getattr(e, "secondary_player", "") or "",
            round(float(getattr(e, "xg", 0.0) or 0.0), 4),
            getattr(e, "outcome", None),
            round(float(getattr(e, "location_x", 0.0) or 0.0), 2),
            round(float(getattr(e, "location_y", 0.0) or 0.0), 2),
        ))
        for e in (getattr(res, "timeline", None) or [])
    ]


def fingerprint_loader(seed: int) -> list:
    """Same match, but assembled the way production assembles one.

    ``physics/measure_gate.py`` reported a failed control: two baseline runs,
    physics off both times, same seed, different results. This function
    replicates that setup exactly — real loader, real ``SquadBuilder``,
    ``_resolve_team_profile``, ``SubstitutionController`` — so the difference
    between the two harnesses can be attributed.

    The one thing that stays identical to ``fingerprint`` is the seed reset
    *after* setup, because ``SquadBuilder.build`` and ``_resolve_team_profile``
    both draw from ``random`` and would otherwise shift the match's own stream
    depending on how many times they had run.
    """
    from datetime import date

    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    loader = get_loader()
    home, away = sorted(loader.get_all_clubs())[:2]
    raw = {c: loader.build_matchday_squad(c) for c in (home, away)}
    squads = {c: SquadBuilder.build(
        team_name=c, starters=raw[c]["starters"],
        substitutes=raw[c]["substitutes"],
        team_superstars=raw[c]["superstars"],
        set_piece_takers=raw[c]["sp_takers"]) for c in (home, away)}
    profs = {c: _resolve_team_profile(c, raw[c]["formation"],
                                      is_home=(c == home))
             for c in (home, away)}

    random.seed(seed)

    cfg = MatchConfig(home_team=home, away_team=away, match_date=date(2026, 9, 8))
    eng = MatchEngine(cfg, profs[home], profs[away])
    eng.quiet = True
    for c in (home, away):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (home, away) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=home, away_team=away, home_subs_bench=subs, away_subs_bench=subs))
    res = eng.simulate()
    return [
        "|".join(str(x) for x in (
            getattr(e, "minute", "?"), getattr(e, "second", "?"),
            getattr(getattr(e, "event_type", None), "name", "?"),
            getattr(e, "player", "") or "", getattr(e, "secondary_player", "") or "",
            round(float(getattr(e, "xg", 0.0) or 0.0), 4), getattr(e, "outcome", None),
            round(float(getattr(e, "location_x", 0.0) or 0.0), 2),
            round(float(getattr(e, "location_y", 0.0) or 0.0), 2)))
        for e in (getattr(res, "timeline", None) or [])
    ]


def _hash(rows: list) -> str:
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def _first_difference(a: list, b: list):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), None, None
    return None, None, None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = int(args[0]) if len(args) > 0 else 3
    seed = int(args[1]) if len(args) > 1 else 1000
    use_loader = "--loader" in sys.argv

    print("=" * 78)
    print("DETERMINISM PROBE" + ("  (production squad path)" if use_loader else ""))
    print("=" * 78)
    print(f"  seed {seed}   PYTHONHASHSEED={HASH_SEED!r}   "
          f"{runs} runs in one process")
    print()
    make = fingerprint_loader if use_loader else fingerprint
    prints = []
    for i in range(runs):
        fp = make(seed)
        prints.append(fp)
        print(f"  run {i}: {len(fp):>4} events   sha {_hash(fp)}"
              + ("   <-- DIVERGED" if fp != prints[0] else ""))
    print()

    base = prints[0]
    diverged = [i for i, fp in enumerate(prints) if fp != base]
    if not diverged:
        print("  All runs identical WITHIN this process.")
        print("  So state is not leaking between runs. If the engine still fails to")
        print("  replay across processes, the cause is process-level: id(), object")
        print("  hashing, or a hash() that depends on the environment.")
        print()
        print(f"  Timeline sha for this configuration: {_hash(base)}")
        print("  Run this again from the shell with the SAME PYTHONHASHSEED and")
        print("  compare. Then run it with a DIFFERENT one — that is the test")
        print("  that separates a cross-process id() bug from a hash-seed one.")
        return 0

    print(f"  !! Diverges WITHIN this process (runs {diverged} differ from run 0).")
    print("     State is leaking between runs, or something unseeded varies.")
    print()
    for i in diverged:
        idx, x, y = _first_difference(base, prints[i])
        print(f"  run 0 vs run {i}: first difference at event {idx}")
        if x is None:
            print(f"     one timeline simply ended early: "
                  f"{len(base)} vs {len(prints[i])} events")
        else:
            for label, row in (("run 0", x), (f"run {i}", y)):
                fields = row.split("|")
                print(f"     {label:<6} min {fields[0]:>3}:{fields[1]:<3} "
                      f"{fields[2]:<20} {fields[3]:<18} xg={fields[5]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
