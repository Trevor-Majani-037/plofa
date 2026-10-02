"""
Run a seeded PLOFA mini-league in-memory and reduce every match to the same
per-team-per-match feature rows used by the StatsBomb extractor.

NO season_state.json, NO exporter, NO persistence of the league ledger — each
match is built, simulated, reduced, and discarded. Match variety comes from
the 18-club roster and a rotating style palette.

Usage
-----
    python -m battlefield.run_plofa --matches 60 --seed 2026 \
            --out battlefield/data/plofa_feats.json

Feature row keys (mirror battlefield.fetch_statsbomb output):
    {team, is_home, goals, xg, shots, shots_on_target, shots_inside_box_pct,
     passes, pass_accuracy_pct, possession_pct,
     corners, fouls, yellow_cards, red_cards, offsides, ball_recoveries}
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ensure repo root is importable even when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_engine import (                                   # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder                          # noqa: E402
from squad_manager import SubstitutionController             # noqa: E402
from manager_profile import ManagerPool                      # noqa: E402
from roster_loader import get_loader                         # noqa: E402

_BOX_FRACTION = 16.5 / 105.0   # same real 18-yard depth as StatsBomb side


def ensure_deterministic_process() -> None:
    """Warn if PYTHONHASHSEED is not pinned. Without it, dict/set iteration
    order varies per process (randomized hash()) and two identical runs of a
    seeded probe produce different matches. Always run battlefield scripts with
    PYTHONHASHSEED=0."""
    if os.environ.get("PYTHONHASHSEED", "") != "0":
        sys.stderr.write(
            "[determinism] PYTHONHASHSEED is not set to '0'; probe/run results "
            "will NOT be reproducible across processes.\n"
            "    PowerShell: $env:PYTHONHASHSEED='0'; python -m ...\n"
        )

# Rotating style palette (home gets palette[i % n], away the half-turn).
_STYLE_PALETTE: List[Tuple[TeamStyle, PlayingStyle, Intensity]] = [
    (TeamStyle.TIKI_TAKA,            PlayingStyle.POSSESSION,        Intensity.HIGH),
    (TeamStyle.ATTACKING,            PlayingStyle.HIGH_PRESS,        Intensity.HIGH),
    (TeamStyle.FLUID_COUNTER,        PlayingStyle.COUNTER,           Intensity.MEDIUM),
    (TeamStyle.GEGENPRESSING,        PlayingStyle.HIGH_PRESS,        Intensity.HIGH),
    (TeamStyle.BALANCED,             PlayingStyle.MIXED,             Intensity.MEDIUM),
    (TeamStyle.WING_PLAY,            PlayingStyle.DIRECT,            Intensity.MEDIUM),
    (TeamStyle.STRUCTURED_POSSESSION, PlayingStyle.PATIENT_BUILD_UP, Intensity.MEDIUM),
    (TeamStyle.DEFENSIVE,            PlayingStyle.LOW_BLOCK,         Intensity.LOW),
    (TeamStyle.ROUTE_ONE,            PlayingStyle.DIRECT,            Intensity.HIGH),
    (TeamStyle.PARK_THE_BUS,         PlayingStyle.LOW_BLOCK,         Intensity.LOW),
]


def _team_profile(club: str, idx: int, is_home: bool) -> TeamProfile:
    n = len(_STYLE_PALETTE)
    slot = idx % n if is_home else (idx + n // 2) % n
    style, playing, intensity = _STYLE_PALETTE[slot]
    return TeamProfile(name=club, style=style, playing_style=playing, intensity=intensity)


def _pick_clubs(clubs: List[str], n_matches: int) -> List[Tuple[str, str]]:
    """Home/away pairs; never a club vs itself."""
    k = len(clubs)
    pairs = []
    for i in range(n_matches):
        home = clubs[(2 * i) % k]
        away = clubs[(2 * i + 1) % k]
        if home == away:
            away = clubs[(2 * i + 2) % k]
        pairs.append((home, away))
    return pairs


def _register_sub_schedule(sub_controller: Any, squads: Dict[str, Dict]) -> None:
    tactical: Dict[str, int] = {}
    for team, squad in squads.items():
        for p in squad["starters"]:
            if getattr(p, "sub_out_minute", None):
                tactical[p.name] = p.sub_out_minute
        for sp in squad["substitutes"]:
            sm = getattr(sp, "sub_in_minute", None)
            if sm is None and hasattr(sp, "dna"):
                sm = getattr(sp.dna, "sub_in_minute", None)
            if sm:
                pos = getattr(sp, "position", "CM")
                adj = {
                    "ST": ["CF", "LW", "RW", "CAM"], "CF": ["ST", "CAM", "LW", "RW"],
                    "LW": ["RW", "CAM", "ST", "LB"], "RW": ["LW", "CAM", "ST", "RB"],
                    "CAM": ["CM", "LW", "RW", "ST"], "CM": ["CAM", "CDM", "LW", "RW"],
                    "CDM": ["CM", "CB"], "LB": ["RB", "CB", "LW"],
                    "RB": ["LB", "CB", "RW"], "CB": ["CDM", "LB", "RB"], "GK": ["GK"],
                }.get(pos, [])
                cand = [
                    s for s in squad["starters"]
                    if getattr(s, "position", "CM") in ([pos] + adj)
                    and s.name not in tactical
                ]
                if cand:
                    tactical[cand[0].name] = sm
    sub_controller.register_tactical_schedule(tactical)


# ---------- feature extraction ------------------------------------------------

def _empty_row(team: str, is_home: bool) -> Dict:
    return {
        "team": team, "is_home": is_home,
        "goals": 0, "xg": 0.0, "shots": 0, "shots_on_target": 0,
        "shots_inside_box_pct": 0.0, "_ib": 0,
        "passes": 0, "passes_completed": 0,
        "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0,
        "offsides": 0, "ball_recoveries": 0,
    }


_ATTEMPTS = frozenset({
    "PASS", "PROGRESSIVE_PASS", "THROUGH_BALL", "SWITCH_OF_PLAY",
    "CROSS_ATTEMPT", "CORNER_TAKEN",
})
_SOT = frozenset({"SHOT_ON_TARGET", "HIT_WOODWORK"})
_SHOT_OFF = frozenset({"SHOT_OFF_TARGET", "SHOT_BLOCKED"})
_PEN = frozenset({"PENALTY_SCORED", "PENALTY_MISSED"})


def _reduce_result(result: Any) -> Dict:
    cfg = result.config
    home, away = cfg.home_team, cfg.away_team
    rows = {home: _empty_row(home, True), away: _empty_row(away, False)}

    for e in result.timeline:
        et = getattr(e, "event_type", None)
        name = getattr(et, "name", "")
        team = getattr(e, "team", "")
        if team not in rows:
            continue
        r = rows[team]
        if name in _ATTEMPTS:
            r["passes"] += 1
            if getattr(e, "outcome", False):
                r["passes_completed"] += 1
            if name == "CORNER_TAKEN":
                r["corners"] += 1
        elif name in _SOT or name in _SHOT_OFF or name in _PEN:
            r["shots"] += 1
            xg = getattr(e, "xg", 0.0) or 0.0
            if name in _PEN:
                xg = 0.79
            r["xg"] += xg
            if name in _SOT or name == "PENALTY_SCORED":
                r["shots_on_target"] += 1
            if _shot_in_box(e, home):
                r["_ib"] += 1
        elif name == "CROSS_SUCCESS":
            pass  # duplicate marker of CROSS_ATTEMPT (counted above)
        elif name == "FOUL_COMMITTED":
            r["fouls"] += 1
        elif name == "YELLOW_CARD":
            r["yellow_cards"] += 1
        elif name == "RED_CARD":
            r["red_cards"] += 1
        elif name == "OFFSIDE":
            r["offsides"] += 1
        elif name == "BALL_RECOVERY":
            r["ball_recoveries"] += 1

    rows[home]["goals"] = result.state.home_goals
    rows[away]["goals"] = result.state.away_goals
    rows[home]["possession_pct"] = result.home_possession_pct
    rows[away]["possession_pct"] = round(100.0 - result.home_possession_pct, 1)

    out = []
    for team, r in rows.items():
        r["pass_accuracy_pct"] = round(100.0 * r["passes_completed"] / max(1, r["passes"]), 1)
        r["shots_inside_box_pct"] = round(100.0 * r["_ib"] / max(1, r["shots"]), 1)
        del r["_ib"]
        del r["passes_completed"]
        out.append(r)
    return out


def _shot_in_box(e: Any, home_name: str) -> bool:
    x = getattr(e, "location_x", None)
    if x is None:
        return False
    attacks_right = getattr(e, "team", "") == home_name
    depth = (105.0 - x) if attacks_right else x
    return depth / 105.0 <= _BOX_FRACTION


# ---------- match runner ------------------------------------------------------

def run_matches(clubs: List[str], n_matches: int, seed: int,
                log: Any = sys.stdout) -> List[Dict]:
    loader = get_loader()
    pairs = _pick_clubs(clubs, n_matches)
    all_rows: List[Dict] = []
    t0 = time.time()

    for i, (home_club, away_club) in enumerate(pairs, 1):
        random.seed(seed + i * 1000)

        home_raw = loader.build_matchday_squad(home_club)
        away_raw = loader.build_matchday_squad(away_club)

        home_style = _team_profile(home_club, i, True)
        away_style = _team_profile(away_club, i, False)

        home_squad = SquadBuilder.build(
            team_name=home_club, starters=home_raw["starters"],
            substitutes=home_raw["substitutes"],
            team_superstars=home_raw["superstars"],
            set_piece_takers=home_raw["sp_takers"],
        )
        away_squad = SquadBuilder.build(
            team_name=away_club, starters=away_raw["starters"],
            substitutes=away_raw["substitutes"],
            team_superstars=away_raw["superstars"],
            set_piece_takers=away_raw["sp_takers"],
        )

        config = MatchConfig(
            home_team=home_club, away_team=away_club,
            season="26/27", competition="PLOFA-BATTLEFIELD",
            venue="BF Arena", stadium_capacity=40000,
            referee="Battle Ref", referee_strictness=0.5,
        )

        mgr_pool = ManagerPool(clubs=[home_club, away_club],
                               style_lookup={home_club: home_style.style.value,
                                             away_club: away_style.style.value})
        sub_controller = SubstitutionController(
            home_team=home_club, away_team=away_club,
            home_subs_bench=home_squad["substitutes"],
            away_subs_bench=away_squad["substitutes"],
            home_style=home_style.style.value,
            away_style=away_style.style.value,
            manager_stubbornness=0.35,
        )
        sub_controller.MAX_SUBS = 3
        _register_sub_schedule(sub_controller, {
            home_club: home_squad, away_club: away_squad,
        })

        engine = MatchEngine(config, home_style, away_style)
        engine.set_squad(home_club, home_squad["starters"], home_squad["substitutes"])
        engine.set_squad(away_club, away_squad["starters"], away_squad["substitutes"])
        engine.set_stamina_controller(sub_controller)
        engine.set_managers(home_manager=mgr_pool.manager_for(home_club),
                            away_manager=mgr_pool.manager_for(away_club))
        engine.quiet = True

        # Silent run — suppress per-chain DEBUG/GOAL narrative noise.
        with contextlib.redirect_stdout(io.StringIO()):
            result = engine.simulate()
        rows = _reduce_result(result)
        for r in rows:
            r["seed"] = seed + i * 1000
        all_rows.extend(rows)

        if i % 5 == 0 or i == n_matches:
            elapsed = time.time() - t0
            per = elapsed / i
            eta = per * (n_matches - i)
            print(f"  [{i}/{n_matches}] {home_club} {result.state.home_goals}-"
                  f"{result.state.away_goals} {away_club}  "
                  f"({elapsed:.0f}s, ~{per:.1f}s/match, ETA {eta:.0f}s)", file=log)
            log.flush() if hasattr(log, "flush") else None

    return all_rows


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run a seeded PLOFA mini-league in-memory.")
    p.add_argument("--matches", type=int, default=48)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=str, default="battlefield/data/plofa_feats.json")
    args = p.parse_args(argv)

    loader = get_loader()
    clubs = loader.get_all_clubs()
    if len(clubs) < 4:
        raise SystemExit(f"Need at least 4 clubs, found {len(clubs)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"PLOFA battlefield — {args.matches} in-memory matches, seed {args.seed}, {len(clubs)} clubs")
    rows = run_matches(clubs, args.matches, args.seed)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {len(rows)} team-match rows → {out_path}")


if __name__ == "__main__":
    main()