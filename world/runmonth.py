"""
Run the controlled one-month integration test (audit §16 phase 10 / plan §18).

    python -m world.runmonth

Plays 24 real matches — Saturday league, Wednesday continental, Tuesday cup,
Saturday league — through the same MatchEngine, into the world ledger, checking
plan §18's continuity promise after every single matchday.

Writes nothing to 26/27.
"""
from __future__ import annotations

import sys
import time
from datetime import date

from world import realworld as rw
from world.ledger import WorldLedger
from world.month import WorldMonth
from world.testworld import COUNTRY, build_test_world, month_fixtures

SEASON = rw.SEASON


def main(argv=None) -> int:
    t0 = time.time()
    print("Building the test world (plan 17)...", flush=True)
    world = build_test_world()
    print(world.summary(), flush=True)
    print("\n" + rw.season_summary(), flush=True)

    fixtures = month_fixtures(world, days=31)
    comps = {f.competition_id for f in fixtures}
    print(f"\nMonth: {len(fixtures)} real fixtures across "
          f"{len(comps)} competitions", flush=True)
    days = sorted({f.match_date for f in fixtures})
    for d in days:
        n = sum(1 for f in fixtures if f.match_date == d)
        print(f"  {d} {d.strftime('%a')}  {n} match(es)", flush=True)

    ledger = WorldLedger(path="<memory>", country_id=COUNTRY,
                         season_id=SEASON)
    month = WorldMonth(world, ledger=ledger, seed=17)

    print("\nSimulating...", flush=True)
    report = month.play(fixtures)

    # ── the §18 report ──
    print("\n" + "=" * 74)
    print("  PLAN 18 - ONE-MONTH INTEGRATION TEST")
    print("=" * 74)
    print(f"  matches played : {report.matches_played}")
    print(f"  total goals    : {report.total_goals}")
    print(f"  players seen   : {report.players_seen}")
    print(f"  continuity     : {'HOLDS' if report.ok else 'BROKEN'}")

    print("\n  per matchday:")
    for d in report.matchdays:
        flag = "ok" if d.ok else f"{len(d.issues)} ISSUE(S)"
        print(f"    {d.match_date} {d.match_date.strftime('%a')}  "
              f"{d.played} match(es), {d.goals} goal(s)   {flag}")
        for issue in d.issues[:4]:
            print(f"        {issue}")

    # ── a player who lived the whole §18 pattern ──
    print("\n  a player carried across competitions:")
    best = None
    for pid, state in ledger.players.items():
        comps_seen = state.competition_ids(SEASON)
        if len(comps_seen) >= 3 and state.minutes_played > 0:
            score = (len(comps_seen), state.minutes_played)
            if best is None or score > best[0]:
                best = (score, pid, state)
    if best:
        _score, pid, state = best
        view = ledger.cross_competition_line(pid, SEASON)
        print(f"    {pid}  ({state.minutes_played} min over "
              f"{state.appearances} matches)")
        print(f"      continuous : fatigue={view['continuous']['fatigue']} "
              f"fitness={view['continuous']['fitness']} "
              f"workload={view['continuous']['workload']} "
              f"injuries={view['continuous']['injuries']}")
        for cid, line in sorted(view["per_competition"].items()):
            print(f"      {cid:<20} apps={line['appearances']:>2} "
                  f"min={line['minutes']:>3} goals={line['goals']:>2} "
                  f"xg={line['xg']:.2f}")
        print(f"      {'CAREER':<20} apps={view['career']['appearances']:>2} "
              f"min={view['career']['minutes']:>3} "
              f"goals={view['career']['goals']:>2} "
              f"xg={view['career']['xg']:.2f}")

    # ── competitions advanced? ──
    print("\n  competitions advanced:")
    for comp in (world.league, world.continental, world.cup):
        if comp is None:
            continue
        played = ledger.competition(comp.competition_id).results
        print(f"    {comp.competition_id:<22} {len(played)} result(s) recorded")

    print(f"\nelapsed: {time.time() - t0:.0f}s")
    print("=" * 74)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
