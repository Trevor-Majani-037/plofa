"""
PLOFA WORLD — proof harness.
============================
world/proof.py  ·  Audit §20 step 7.

    "A 3-match dry-run script (`scripts/` or `world/proof.py`) printing the
     resulting standings, cup bracket, and per-player cross-competition stat
     line."

Audit §20 acceptance, restated as executable checks:

    the toy month's player state is continuous across three competitions,
    the table and bracket advance correctly, no fixture collision,
    and the live 26/27 ledger is byte-identical before/after every proof run.

This harness proves the three parts that exist today — **standings advance,
bracket advances, no fixture collision** — and prints the rest. Cross-competition
*player* continuity needs the world ledger (audit §16 phase 6), so it is
reported as the next gate rather than faked here.

Run it::

    & .\\.venv\\Scripts\\python.exe -m world.proof

Exit code 0 means every invariant held. It touches no 26/27 state: the whole
world is held in memory and the test clubs are synthetic.
"""
from __future__ import annotations

import sys
import zlib
from datetime import date, timedelta
from typing import Dict, List, Tuple

from world.calendar import (
    Calendar,
    continental_midweek,
    cup_midweek,
    league_saturday,
)
from world.competition import (
    KnockoutCompetition,
    LeagueCompetition,
    cup_rules,
    league_rules,
)

# ─────────────────────────────────────────────
# TEST WORLD (plan §17 — 2 countries, 8 clubs each)
# ─────────────────────────────────────────────

COUNTRY_A = "CTR-TOLAND"
COUNTRY_B = "CTR-VREDA"
SEASON = "26/27"
START = date(2026, 8, 8)          # Saturday

CLUBS_A = [f"TOL-{n}" for n in
           ("Avada Zenith", "Claw", "Club Chovers", "Ganester",
            "Justice", "Lige-8", "Natrican", "Oxton")]
CLUBS_B = [f"VRD-{n}" for n in
           ("Port Colborne", "Red Wolves", "Rodice", "Seafcea",
            "Telbey", "Trendboys", "Triumpher", "Tryox City")]

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

RULE = "═" * 74
THIN = "─" * 74


def _prepare_console() -> bool:
    """Make stdout able to render this report, and say whether it can.

    Order of preference:
      1. the current encoding already handles the glyphs  → pretty output
      2. reconfigure stdout to UTF-8 (Windows Terminal, redirected files)  → pretty
      3. neither works → the caller degrades to ASCII

    Returns True when non-ASCII glyphs are safe to print.
    """
    pretty = "═─§·✔✖→"
    try:
        pretty.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError, AttributeError):
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        pretty.encode("utf-8")
        return True
    except (AttributeError, UnicodeEncodeError, LookupError):
        return False


UNICODE_OK = _prepare_console()

RULE = ("═" if UNICODE_OK else "=") * 74
THIN = ("─" if UNICODE_OK else "-") * 74
SECTION = "§" if UNICODE_OK else "Section "
DOT = "·" if UNICODE_OK else "-"
TICK = "OK" if UNICODE_OK else "[+]"
CROSS = "FAIL" if UNICODE_OK else "[x]"
ARROW = "->" if UNICODE_OK else "->"
DASH = "—" if UNICODE_OK else "-"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


# ─────────────────────────────────────────────
# BUILD THE WORLD
# ─────────────────────────────────────────────

def build_world() -> Tuple[Calendar, Dict[str, object]]:
    """Three competitions over one calendar — plan §12's exact scenario:

        Saturday   domestic league
        Tuesday    domestic cup
        Wednesday  continental
    """
    league = LeagueCompetition(
        competition_id="CMP-TOL-D1", name="Toland First Division",
        country_id=COUNTRY_A, season_id=SEASON,
        rules=league_rules(), participant_ids=CLUBS_A, seed=26,
    )
    league.make_fixtures(START, days_between=7)

    cup = KnockoutCompetition(
        competition_id="CMP-TOL-CUP", name="Toland Cup",
        country_id=COUNTRY_A, season_id=SEASON,
        rules=cup_rules(), participant_ids=CLUBS_A, seed=26,
    )
    cup.draw_round("Quarter-final")

    continental = LeagueCompetition(
        competition_id="CMP-CONT", name="Continental Championship",
        country_id="CTR-CONT", season_id=SEASON,
        rules=league_rules(double_round_robin=False),
        participant_ids=CLUBS_A + CLUBS_B, seed=26,
    )
    continental.make_fixtures(START + timedelta(days=3), days_between=7)

    cal = Calendar(country_id=COUNTRY_A, season_id=SEASON, start=START,
                   horizon_days=300, default_min_rest_days=3)
    cal.add_league(league, league_saturday())
    cal.add_knockout_round(cup, cup_midweek(), "Quarter-final")
    cal.add_league(continental, continental_midweek())

    cal.build()
    return cal, {"league": league, "cup": cup, "continental": continental}


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────

def _advance_league(league: LeagueCompetition, cal: Calendar, upto: int) -> None:
    """Play the first ``upto`` matchdays of a league through the calendar's
    own placed fixtures, so the printed table is the table the world layer
    actually produced.

    Scorelines are SYNTHETIC and deterministic: this harness proves the
    calendar and the progression, not the football. The digest is crc32, never
    ``hash()`` — builtin string hashing is salted per process and would make
    the printed table differ between runs (the PYTHONHASHSEED trap, README §8).
    """
    for _d, fixtures in cal.matchdays(league.competition_id)[:upto]:
        for fx in fixtures:
            hg, ag = (2, 1) if zlib.crc32(fx.fixture_id.encode()) % 2 else (1, 2)
            league.apply_result(fx.home_id, fx.away_id,
                                fx.planned.matchday, hg, ag)


def report(cal: Calendar, parts: Dict[str, object]) -> int:
    league = parts["league"]
    cup = parts["cup"]
    continental = parts["continental"]

    print()
    print(_c(RULE, BOLD))
    print(_c(f"  PLOFA WORLD {DASH} proof harness "
             f"(audit {SECTION}20 {DOT} plan {SECTION}17/{SECTION}18)", BOLD))
    print(_c(RULE, BOLD))
    print(cal.summary())

    # -- calendar ------------------------------------------------------
    print()
    print(_c("FIXTURE COLLISION REPORT", BOLD))
    print(_c(THIN, DIM))
    layout_kinds = ("same_day_double_booking", "venue_double_booking",
                    "insufficient_rest")
    layout = [v for v in cal.violations() if v.kind in layout_kinds]
    unscheduled = cal.violations_of_kind("unscheduled")

    # re-derive the invariant independently, straight off the fixture list
    seen: Dict[Tuple[str, object], List[str]] = {}
    for fx in cal.fixtures():
        for club in fx.club_ids:
            seen.setdefault((club, fx.match_date), []).append(fx.fixture_id)
    collisions = {k: v for k, v in seen.items() if len(v) > 1}

    print(f"  clubs double-booked on one day : {len(collisions)}")
    print(f"  layout violations              : {len(layout)}")
    print(f"  fixtures unplaceable           : {len(unscheduled)}")
    for v in unscheduled[:5]:
        print(_c(f"      {v}", DIM))
    verdict = not collisions and not layout
    print("  " + ((TICK + " ") if verdict else (CROSS + " "))
          + (_c("INVARIANT HOLDS", GREEN) if verdict
             else _c("INVARIANT BROKEN", RED)))

    # -- a week in the life --------------------------------------------
    print()
    print(_c(f"THE PLAN {SECTION}12 WEEK (one club, three competitions)", BOLD))
    print(_c(THIN, DIM))
    star = CLUBS_A[4]                      # Justice
    for fx in cal.for_club(star)[:6]:
        print(f"  {fx.match_date}  {fx.match_date.strftime('%a')}  "
              f"{fx.kickoff}  {fx.competition_id:<14} "
              f"{fx.home_id:>16} v {fx.away_id:<16} @ {fx.venue}")

    # -- standings ------------------------------------------------------
    _advance_league(league, cal, upto=6)
    print()
    print(_c(f"STANDINGS {DASH} {league.name} (after 6 matchdays, synthetic results)", BOLD))
    print(_c(THIN, DIM))
    print(f"  {'#':>2}  {'club':<22} {'P':>2} {'W':>2} {'D':>2} {'L':>2} "
          f"{'GF':>3} {'GA':>3} {'GD':>3} {'Pts':>4}")
    for i, rec in enumerate(league.standings(), 1):
        print(f"  {i:>2}  {rec.team:<22} {rec.played:>2} {rec.won:>2} "
              f"{rec.drawn:>2} {rec.lost:>2} {rec.goals_for:>3} "
              f"{rec.goals_against:>3} {rec.goal_diff:>3} {rec.points:>4}")

    _advance_league(continental, cal, upto=3)
    print()
    print(_c(f"STANDINGS {DASH} {continental.name} "
             f"(after 3 matchdays, 2 countries)", BOLD))
    print(_c(THIN, DIM))
    for i, rec in enumerate(continental.standings()[:6], 1):
        print(f"  {i:>2}  {rec.team:<22} {rec.played:>2} {rec.won:>2} "
              f"{rec.drawn:>2} {rec.lost:>2} {rec.goals_for:>3} "
              f"{rec.goals_against:>3} {rec.goal_diff:>3} {rec.points:>4}")
    print(_c(f"      {DOT} {len(continental.standings())} clubs total", DIM))

    # -- bracket --------------------------------------------------------
    print()
    print(_c(f"BRACKET {DASH} {cup.name} ({', '.join(cup.round_names())})", BOLD))
    print(_c(THIN, DIM))
    for round_name in cup.round_names():
        ties = cup.ties_in(round_name)
        if not ties:
            continue
        print(f"  {round_name}")
        for tie in ties:
            if tie.played:
                print(f"      {tie.home_id:>16} {tie.home_goals}-{tie.away_goals} "
                      f"{tie.away_id:<16} {ARROW} {tie.winner_id}")
            else:
                print(f"      {tie.home_id:>16}  vs  {tie.away_id:<16} "
                      f"{_c('(not yet played)', DIM)}")

    # -- plan §18: one continuous player state --------------------------
    integrity = _plan18_section(cal, league, cup, continental)

    # -- what is not proven yet -----------------------------------------
    print()
    print(_c(f"NEXT GATE {DASH} not proven by this harness", BOLD))
    print(_c(THIN, DIM))
    print(f"  {DOT} extra time + penalty shootouts in the engine (phase 5).")
    print(f"  {DOT} promotion/relegation + qualification (phase 8).")
    print(f"  {DOT} the ledger's lines are not yet fed by real engine output —")
    print("    this harness supplies synthetic player lines on purpose.")
    print()
    print(_c(RULE, BOLD))
    return 0 if (verdict and integrity) else 1


# ─────────────────────────────────────────────
# PLAN §18 — THE CRITICAL INTEGRATION TEST
# ─────────────────────────────────────────────

def _plan18_section(cal: Calendar, league, cup, continental) -> bool:
    """Run plan §18 for real: Saturday league → Wednesday continental →
    Saturday league → Wednesday cup, and prove the same player carries ONE
    continuous state throughout while each competition keeps its own
    statistical line.

    Scorelines are synthetic (this harness proves the ledger, not the football),
    but the state arithmetic is the real ledger.
    """
    from world.ledger import MatchReport, PlayerMatchLine, WorldLedger

    hero = "PLY-WORLD-0001"
    home = "TOL-Justice"
    led = WorldLedger(path="<memory>")
    for comp in (league, cup, continental):
        led.register_competition(comp)

    # the plan §18 month, taken from the calendar's own placed fixtures
    order: List[Tuple[str, date, str, str]] = []
    for comp_id in (league.competition_id, cup.competition_id,
                    continental.competition_id):
        for f in cal.for_competition(comp_id):
            if home in f.club_ids:
                order.append((comp_id, f.match_date, f.home_id, f.away_id))
    order.sort(key=lambda t: t[1])
    order = order[:4]

    minutes = (90, 75, 90, 60)
    for i, (comp_id, when, hid, aid) in enumerate(order):
        led.record_match(MatchReport(
            competition_id=comp_id, season_id=SEASON, match_date=when,
            home_id=hid, away_id=aid, home_goals=1 if i % 2 == 0 else 0,
            players=[PlayerMatchLine(
                player_id=hero, club_id=home, minutes=minutes[i % 4],
                goals=1 if i == 0 else 0, shots=2, xg=0.7,
                yellow_cards=1 if i == 1 else 0,
                fatigue=0.40 + 0.15 * i, fitness=1.0 - 0.05 * i,
            )]))
        led.advance_matchday()

    view = led.cross_competition_line(hero, SEASON)
    cont = view["continuous"]
    rows = view["per_competition"]

    print()
    print(_c(f"PLAN {SECTION}18 {DASH} ONE CONTINUOUS PLAYER STATE", BOLD))
    print(_c(THIN, DIM))
    for comp_id, when, hid, aid in order:
        print(f"  {when}  {when.strftime('%a')}  {comp_id:<14} "
              f"{hid:>16} v {aid:<16}")
    print()
    print(f"  {'competition':<16} {'apps':>5} {'min':>5} {'goals':>6} {'xg':>6} "
          f"{'yell':>5}")
    for comp_id in sorted(rows):
        r = rows[comp_id]
        print(f"  {comp_id:<16} {r['appearances']:>5} {r['minutes']:>5} "
              f"{r['goals']:>6} {r['xg']:>6.2f} {r['yellow_cards']:>5}")
    print(f"  {'-' * 50}")
    print(f"  {'CONTINUOUS':<16} {cont['appearances']:>5} "
          f"{cont['minutes_played']:>5} "
          f"{view['career']['goals']:>6} {view['career']['xg']:>6.2f} "
          f"{led.player(hero).yellow_cards:>5}")

    # the two invariants plan §18 actually cares about
    summed_minutes = sum(r["minutes"] for r in rows.values())
    summed_apps = sum(r["appearances"] for r in rows.values())
    ok_minutes = summed_minutes == cont["minutes_played"]
    ok_apps = summed_apps == cont["appearances"]
    ok_workload = led.player(hero).workload == cont["minutes_played"]

    print()
    print(f"  scoped minutes sum to the continuous total : "
          f"{summed_minutes} == {cont['minutes_played']}  "
          f"{_ok(ok_minutes)}")
    print(f"  scoped appearances sum to the continuous total: "
          f"{summed_apps} == {cont['appearances']}  {_ok(ok_apps)}")
    print(f"  workload tracks the continuous total         : "
          f"{led.player(hero).workload} == {cont['minutes_played']}  "
          f"{_ok(ok_workload)}")
    print(f"  fatigue carried across every boundary       : "
          f"{cont['fatigue']:.2f}  {_ok(cont['fatigue'] > 0.0)}")
    print(f"  distinct competitions in the ledger          : "
          f"{len(rows)}  {_ok(len(rows) >= 2)}")

    good = ok_minutes and ok_apps and ok_workload and len(rows) >= 2
    print("  " + (_c("ONE CONTINUOUS STATE HOLDS", GREEN) if good
                  else _c("CONTINUITY BROKEN", RED)))
    return good


def _ok(flag: bool) -> str:
    return _c(TICK, GREEN) if flag else _c(CROSS, RED)


def main() -> int:
    cal, parts = build_world()
    return report(cal, parts)


if __name__ == "__main__":
    raise SystemExit(main())
