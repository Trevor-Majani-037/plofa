"""
Print the opening of a match, every field, for side-by-side diffing.

``probe_determinism.py`` found that the production squad path diverges across
processes: three fresh interpreters, same seed, same PYTHONHASHSEED, gave 3,057
/ 3,060 / 3,095 events. The hardcoded-squad path is stable across processes.

The first divergence is around event 5 (minute 1:40, a DISPOSSESSED), but the
probe's fingerprint view prints only four of the nine fields it compares, so it
cannot show *what* differs. This prints all of them, for the opening of the
match, so the differing field is visible.

Run:  python physics/dump_opening.py > a.txt   (in two shells, then diff)
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\PLOFA\plofa")

from scripts.physics.probe_determinism import (  # noqa: E402
    HASH_SEED, fingerprint_loader)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260927

FIELDS = ("minute", "second", "event_type", "team", "player", "secondary_player",
          "situation", "location_x", "location_y", "end_x", "end_y", "xg", "xa",
          "outcome", "body_part", "phase", "game_state")


def main() -> int:
    print(f"# PYTHONHASHSEED={HASH_SEED}  seed={SEED}")
    from datetime import date

    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    import random
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

    print(f"# home={home}  away={away}")
    print(f"# home formation={raw[home]['formation']}  "
          f"away formation={raw[away]['formation']}")
    print(f"# home XI={[getattr(p, 'name', p) for p in squads[home]['starters']]}")
    print(f"# away XI={[getattr(p, 'name', p) for p in squads[away]['starters']]}")
    print(f"# home XI roles={[(getattr(p, 'position', '?')) for p in squads[home]['starters']]}")

    random.seed(SEED)
    cfg = MatchConfig(home_team=home, away_team=away, match_date=date(2026, 9, 8))
    eng = MatchEngine(cfg, profs[home], profs[away])
    eng.quiet = True
    for c in (home, away):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (home, away) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=home, away_team=away, home_subs_bench=subs, away_subs_bench=subs))
    res = eng.simulate()

    print(f"# final {res.home_goals}-{res.away_goals}  "
          f"events={len(getattr(res, 'timeline', None) or [])}")
    for i, e in enumerate((getattr(res, "timeline", None) or [])[:N]):
        parts = []
        for f in FIELDS:
            v = getattr(e, f, "<none>")
            if f == "event_type":
                v = getattr(v, "name", v)
            elif isinstance(v, float):
                v = f"{v:.6f}"
            parts.append(f"{f}={v}")
        md = getattr(e, "metadata", None)
        print(f"[{i:>3}] " + " ".join(parts))
        if md:
            keys = sorted(md.keys()) if isinstance(md, dict) else []
            print(f"      metadata({len(keys)}): " +
                  ", ".join(f"{k}={md[k]!r}" for k in keys[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
