"""DO KEY PASSES AND ASSISTS HAVE TRUE TRACKED COORDINATES? (2026-10-02)

The claim under test: "key passes / chance-creation events had end points but
not start points — only where the shot happens."

Reading suggests three separate things are tangled together, so this measures
each one rather than agreeing with any of them:

  A. THE LEDGER (`chance_creation.py::ChanceRecord`) — the real chain. It has
     pass_x/pass_y (start) and pass_end_x/pass_end_y (end). Is the start ever
     populated, and is the end genuinely the PASS destination rather than the
     shot origin?

  B. THE ENGINE'S CHANCE_CREATED EVENTS (`event_chain.py:5649`) — the
     supposedly-superseded fabrication. Its start is
     `x - random.uniform(5, 20)`, i.e. a random offset off the shot, and its
     end is `x, y` = the shot location. If that is what ships in the timeline,
     every chance in the match has an invented start and a forced end.

  C. THE ASSIST ON THE GOAL — `result.goal_assistant` comes from
     `_pick_creator`, a position-WEIGHTED, distance-weighted random pick, not
     from "who actually passed to the scorer". `exporter.py` writes
     `g.secondary_player` into the Goals sheet. So the two sources can disagree,
     and the Goals sheet is not the ledger.

Observation-only; patches nothing. Run:
    .venv\\Scripts\\python.exe _diag_chance_coords.py [n]
"""
import math
import random
import sys
from collections import Counter
from datetime import date

from match_engine import (
    EventType, MatchConfig, MatchEngine, PlayingStyle,
    TeamProfile, TeamStyle, Intensity,
)
from player_dna import SquadBuilder
from roster_loader import get_loader
from chance_creation import ChanceCreationLedger

XLSX = "PLOFA-2026-2027.xlsx"

RESULTS = []


def build_pair(home, away):
    loader = get_loader(XLSX)
    hr = loader.build_matchday_squad(home)
    ar = loader.build_matchday_squad(away)
    hs = SquadBuilder.build(home, starters=hr["starters"],
                            substitutes=hr["substitutes"])
    aw = SquadBuilder.build(away, starters=ar["starters"],
                             substitutes=ar["substitutes"])
    cfg = MatchConfig(home_team=home, away_team=away,
                      match_date=date(2026, 8, 16), matchday=1,
                      venue=f"{home} Stadium", stadium_capacity=45000)
    eng = MatchEngine(
        cfg,
        TeamProfile(name=home, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.POSSESSION,
                    intensity=Intensity.MEDIUM),
        TeamProfile(name=away, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.MIXED,
                    intensity=Intensity.MEDIUM))
    eng.set_squad(home, hs["starters"], hs["substitutes"])
    eng.set_squad(away, aw["starters"], aw["substitutes"])
    return eng


def _is_off_pitch(x, y):
    return x is None or y is None or not (-5 <= x <= 110 and -5 <= y <= 73)


def analyse(res, n):
    tl = res.timeline
    led = ChanceCreationLedger(tl).compute()

    # ── A. THE LEDGER ────────────────────────────────────────────────
    with_creator = [r for r in led.records if r.creator]
    print(f"\n  A. LEDGER — {len(led.records)} shots, "
          f"{len(with_creator)} with a creator")
    start_missing = sum(
        1 for r in with_creator if r.pass_x is None or r.pass_y is None)
    end_missing = sum(
        1 for r in with_creator
        if r.pass_end_x is None or r.pass_end_y is None)
    at_origin = sum(
        1 for r in with_creator
        if r.pass_x is not None and r.pass_end_x is not None
        and math.hypot(r.pass_end_x - r.pass_x, r.pass_end_y - r.pass_y) < 0.5)
    print(f"    creator start missing        {start_missing}/{len(with_creator)}")
    print(f"    creator end missing          {end_missing}/{len(with_creator)}")
    print(f"    zero-length pass (<0.5 m)    {at_origin}/{len(with_creator)}")

    # pass end vs shot origin — the carry question
    carries, exact = 0, 0
    gaps = []
    for r in with_creator:
        g = math.hypot(r.pass_end_x - r.shot_x, r.pass_end_y - r.shot_y)
        gaps.append(g)
        if g < 1.0:
            exact += 1
        else:
            carries += 1
    if gaps:
        gaps.sort()
        print(f"    pass-end -> shot-origin gap  median "
              f"{gaps[len(gaps)//2]:.1f} m   max {gaps[-1]:.1f} m")
        print(f"      ends AT the shot  {exact}/{len(gaps)}"
              f"   |   receiver carried on  {carries}/{len(gaps)}")

    # WHICH event is the ledger calling the key pass, and is its endpoint real?
    kinds = Counter()
    for r in with_creator:
        pe = tl[r.pass_event_index]
        kinds[pe.event_type.name] += 1
    print(f"    key-pass event types  {dict(kinds)}")

    # how many pass-like events in the whole match carry a real destination?
    pass_like = [e for e in tl if e.event_type in (
        EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
        EventType.THROUGH_BALL, EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
        EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS)]
    have_end = sum(1 for e in pass_like
                   if e.end_x is not None and e.end_y is not None)
    have_recv = sum(1 for e in pass_like if e.secondary_player)
    print(f"    ALL deliveries {len(pass_like)}: "
          f"with a real destination {have_end} ({100.0*have_end/max(len(pass_like),1):.0f}%)"
          f"   with a receiver named {have_recv}")
    byk = Counter()
    byk_end = Counter()
    for e in pass_like:
        byk[e.event_type.name] += 1
        if e.end_x is not None and e.end_y is not None:
            byk_end[e.event_type.name] += 1
    for k, v in byk.most_common():
        print(f"      {k:<18} {v:>4}  with a destination {byk_end[k]:>4}")

    # ── B. THE ENGINE'S CHANCE_CREATED EVENTS ────────────────────────
    cc = [e for e in tl if e.event_type in (EventType.CHANCE_CREATED,
                                            EventType.BIG_CHANCE_CREATED)]
    print(f"\n  B. ENGINE CHANCE_CREATED EVENTS — {len(cc)} in the timeline"
          f" = {len(cc)/max(n,1):.1f}/match")
    if cc:
        lens, ends_match_shot = [], 0
        real_start = 0
        pass_origin_lookup = set()
        for e in tl:
            if e.event_type in (EventType.PASS, EventType.PROGRESSIVE_PASS,
                                EventType.THROUGH_BALL, EventType.SWITCH_OF_PLAY,
                                EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
                                EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS):
                if e.location_x is not None:
                    pass_origin_lookup.add(
                        (e.player, round(e.location_x, 1), round(e.location_y, 1)))
        for e in cc:
            lx, ly, ex, ey = e.location_x, e.location_y, e.end_x, e.end_y
            if lx is not None and ex is not None:
                lens.append(math.hypot(ex - lx, ey - ly))
                # does the following shot event sit at the same place?
                for nx in tl[0:0]:
                    pass
            if (e.player, round(lx or 0, 1), round(ly or 0, 1)) in pass_origin_lookup:
                real_start += 1
        if lens:
            lens.sort()
            print(f"    start->end length  min {lens[0]:.2f}  "
                  f"median {lens[len(lens)//2]:.2f}  max {lens[-1]:.2f}  "
                  f"mean {sum(lens)/len(lens):.2f}")
            in_band = sum(1 for d in lens if 5.0 <= d <= 20.0)
            print(f"    length inside [5,20] m   {in_band}/{len(lens)}"
                  f"   <- the `random.uniform(5, 20)` signature")
        print(f"    start matches a REAL pass origin by that player  "
              f"{real_start}/{len(cc)}")
        print(f"    (a fabricated origin matches none; that is the test)")

    # does the end equal the shot that follows it?
    for i, e in enumerate(tl):
        if e.event_type not in (EventType.CHANCE_CREATED,
                                EventType.BIG_CHANCE_CREATED):
            continue
        for nx in tl[i + 1:i + 6]:
            if nx.event_type in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                                 EventType.SHOT_BLOCKED, EventType.HIT_WOODWORK,
                                 EventType.GOAL, EventType.PENALTY_SCORED,
                                 EventType.PENALTY_MISSED):
                # The shot's TAKEN location, not its endpoint: the shot
                # event's end_x is the flight terminus (the goal), which is a
                # different question entirely.
                if nx.location_x is not None and e.end_x is not None:
                    if math.hypot(nx.location_x - e.end_x,
                                  nx.location_y - e.end_y) < 1.0:
                        ends_match_shot += 1
                break
    if cc:
        print(f"    end == the FOLLOWING SHOT's TAKEN location  "
              f"{ends_match_shot}/{len(cc)}")

    # ── C. ASSIST ON THE GOAL ────────────────────────────────────────
    goals = [g for g in res.goals if g.event_type != EventType.OWN_GOAL]
    print(f"\n  C. ASSISTS — {len(goals)} goals")
    agree = disagree = noreal = 0
    detail = []
    for g in goals:
        cc_rec = next(
            (r for r in led.records
             if r.outcome == "goal" and r.shooter == g.player
             and abs(r.minute - (g.minute or 0)) <= 1), None)
        engine_assist = g.secondary_player or ""
        if cc_rec is None or not cc_rec.creator:
            noreal += 1
            detail.append((g.minute, g.player, engine_assist, "(no ledger key pass)"))
            continue
        if cc_rec.creator == engine_assist:
            agree += 1
        else:
            disagree += 1
            detail.append((g.minute, g.player, engine_assist, cc_rec.creator))
    print(f"    engine goal_assistant == ledger creator   {agree}/{len(goals)}")
    print(f"    DISAGREE                                  {disagree}/{len(goals)}")
    print(f"    no ledger key pass for the goal           {noreal}/{len(goals)}")
    for m, s, e, l in detail[:14]:
        print(f"      min {m:>3}  {s:<22} engine={e or '-':<22} ledger={l}")

    # ── D. WHAT IS ACTUALLY IN THE TIMELINE BEFORE EACH GOAL? ──────
    # Before "fixing" the ledger for failing to find an assist, establish
    # whether an assist EXISTS in the data. Print the run-up verbatim.
    print(f"\n  D. RUN-UP TO EACH GOAL (the events a ledger must read)")
    for gi, g in enumerate(res.goals, 1):
        try:
            gidx = next(i for i, e in enumerate(tl) if e is g)
        except StopIteration:
            continue
        cc_rec = next(
            (r for r in led.records
             if r.outcome == "goal" and r.shooter == g.player
             and abs(r.minute - (g.minute or 0)) <= 1), None)
        tag = "no key pass" if (cc_rec is None or not cc_rec.creator) \
            else f"key pass by {cc_rec.creator}"
        print(f"    GOAL {gi} min {g.minute} {g.player} "
              f"(engine assist: {g.secondary_player or '-'})  -> {tag}")
        for j in range(max(0, gidx - 12), gidx + 1):
            e = tl[j]
            if e.team != g.team:
                continue
            mark = ">>" if j == gidx else "  "
            print(f"      {mark} {e.event_type.name:<18} "
                  f"{e.player:<20} -> {e.secondary_player or '-':<20} "
                  f"({e.location_x:5.1f},{e.location_y:4.1f})"
                  f"->({(e.end_x if e.end_x is not None else float('nan')):5.1f},"
                  f"{(e.end_y if e.end_y is not None else float('nan')):4.1f})"
                  f" {'OK' if e.outcome else 'X'}")

    # ── E. ARE SHOTS TAKEN FROM WHERE FOOTBALL ALLOWS? ──────────────
    # A shot's own end point says which goal it was aimed at, so it names the
    # attack direction WITHOUT needing to know the half. Compare the shot's
    # taken location against that goal: the distance from the shot to the goal
    # line it attacked must be a plausible shooting distance. Anything inside
    # a few metres of the line is either a tap-in or a broken coordinate, and
    # the two are indistinguishable from the event alone — so measure and
    # report, do not interpret.
    print(f"\n  E. SHOT GEOMETRY — distance from the shot to the goal it attacked")
    home = res.config.home_team
    # Step 1: DERIVE the attack direction from the data. Only shots that
    # already carry an endpoint can tell us which goal line they attacked.
    # Assume nothing about the half-time ends change — measure it.
    xtab = Counter()
    for e in tl:
        if not e.is_shot or e.end_x is None:
            continue
        side = "home" if e.team == home else "away"
        half = "H1" if (e.minute or 0) < 45 else "H2"
        gl = "x0" if abs(e.end_x) < 52.5 else "x105"
        xtab[(side, half, gl)] += 1
    print(f"    attack direction, from shots that DO carry an endpoint:")
    for k in sorted(xtab):
        print(f"      {k[0]:<5} {k[1]}  ->  {k[2]:<5}  {xtab[k]:>3}")
    # The rule, whatever the table says it is.
    def _goal_line(e):
        side = "home" if e.team == home else "away"
        half = "H1" if (e.minute or 0) < 45 else "H2"
        # whichever line this (side, half) cell predominantly aimed at
        cand = {gl: xtab[(side, half, gl)] for gl in ("x0", "x105")}
        if cand["x0"] == cand["x105"]:
            return None
        return 0.0 if max(cand, key=cand.get) == "x0" else 105.0

    shots = [e for e in tl if e.is_shot]
    with_ep = sum(1 for e in shots if e.end_x is not None)
    print(f"    shots {len(shots)}, carrying an endpoint {with_ep} "
          f"({100.0*with_ep/max(len(shots),1):.0f}%)   <- coverage, not quality")
    dists, weird, unknown = [], [], 0
    for e in shots:
        gl = _goal_line(e)
        if gl is None:
            unknown += 1
            continue
        d = abs(gl - e.location_x)
        dists.append(d)
        if d < 4.0:
            weird.append((e.minute, e.event_type.name, e.player,
                          e.team, e.location_x, e.location_y, e.end_x, e.end_y))
    if dists:
        dists.sort()
        n = len(dists)
        print(f"    distance from the shot to the goal line it attacked "
              f"({n} shots, {unknown} unattributable):")
        print(f"      min {dists[0]:.1f}  p10 {dists[n//10]:.1f}  "
              f"median {dists[n//2]:.1f}  p90 {dists[9*n//10]:.1f}  max {dists[-1]:.1f}")
        for lo, hi, lab in ((0, 4, "inside 4 m (tap-in or broken coordinate)"),
                            (4, 8, "4-8 m"),
                            (8, 16, "8-16 m"),
                            (16, 25, "16-25 m"),
                            (25, 99, "beyond 25 m")):
            c = sum(1 for d in dists if lo <= d < hi)
            print(f"      {lab:<42} {c:>4}  ({100.0*c/n:4.1f}%)")
        print(f"    real PL: essentially 0% under 4 m, ~5-8% 4-8 m, peak 10-18 m")
    if weird:
        print(f"    the sub-4 m shots ({len(weird)}):")
        for m, t, p, tm, lx, ly, ex, ey in weird[:12]:
            print(f"      min {m:>3} {t:<18} {p:<20} {tm:<10} "
                  f"({lx:5.1f},{ly:4.1f})->({ex:5.1f},{ey:4.1f})")

    # ── G. WRONG-END SHOTS — where they still come from ─────────────
    # A team's shots must sit on ITS attacking half. Anything past halfway
    # for the away team (or short of it for the home team) is a shot resolved
    # against the wrong end. List them with the situation that produced them:
    # that names the code path without guessing at it.
    print(f"\n  G. WRONG-END SHOTS (taken on the team's OWN half)")
    away = res.config.away_team
    wrong = []
    for e in shots:
        if e.team == home and e.location_x < 52.5:
            wrong.append((e, "home", "own half"))
        elif e.team == away and e.location_x > 52.5:
            wrong.append((e, "away", "own half"))
    tot_wrong = len(wrong)
    print(f"    {tot_wrong} of {len(shots)} shots "
          f"({100.0*tot_wrong/max(len(shots),1):.0f}%)")
    bysit = Counter()
    for e, side, _ in wrong:
        sit = e.situation.name if e.situation else "?"
        bysit[(e.event_type.name, sit)] += 1
    for k, v in bysit.most_common():
        print(f"      {k[0]:<18} situation={k[1]:<18} {v:>3}")
    for e, side, _ in wrong[:10]:
        sit = e.situation.name if e.situation else "?"
        print(f"      min {e.minute:>3} {side:<5} {e.event_type.name:<16} "
              f"sit={sit:<16} {e.player:<20} ({e.location_x:5.1f},{e.location_y:4.1f})")

    # ── F. PER TEAM — the decisive split ────────────────────────────
    # Pooled, a shot taken from the wrong end and a shot aimed at the wrong
    # goal look identical. Split them: where each team TAKES its shots, versus
    # which goal line its shots are RESOLVED against.
    print(f"\n  F. PER TEAM — where shots are taken vs which goal they are aimed at")
    for tm in (home, res.config.away_team):
        s = [e for e in shots if e.team == tm]
        if not s:
            continue
        xs = sorted(e.location_x for e in s)
        ep = [e for e in s if e.end_x is not None]
        gls = Counter("x0" if abs(e.end_x) < 52.5 else "x105" for e in ep)
        n = len(xs)
        print(f"    {tm} ({'home' if tm == home else 'away'}): "
              f"{n} shots, {len(ep)} with an endpoint")
        print(f"      taken at x:  min {xs[0]:.1f}  p25 {xs[n//4]:.1f}  "
              f"median {xs[n//2]:.1f}  p75 {xs[3*n//4]:.1f}  max {xs[-1]:.1f}")
        if gls:
            print(f"      aimed at:    {dict(gls)}")
        # If the team shoots from the x105 end but its endpoints claim x0,
        # the resolution is aimed at the wrong goal. Compare the medians.
        if ep:
            d_claim = sorted(abs((0.0 if abs(e.end_x) < 52.5 else 105.0)
                                 - e.location_x) for e in ep)
            d_flip = sorted(abs(105.0 - e.location_x) if abs(e.end_x) < 52.5
                            else abs(e.location_x) for e in ep)
            print(f"      distance to the goal its endpoint claims: "
                  f"median {d_claim[len(d_claim)//2]:.1f} m")
            print(f"      distance to the OPPOSITE goal:             "
                  f"median {d_flip[len(d_flip)//2]:.1f} m")
        print(f"      (real PL median shot distance ~17 m, p90 < 30 m)")
    return {
        "shots": len(led.records), "creators": len(with_creator),
        "start_missing": start_missing, "end_missing": end_missing,
        "carry": carries, "exact": exact, "gaps": gaps,
        "cc": cc, "lens": lens if cc else [], "real_start": real_start,
        "ends_match_shot": ends_match_shot,
        "goals": len(goals), "agree": agree, "disagree": disagree,
        "noreal": noreal,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    for i in range(n):
        h, a = pairs[i % len(pairs)]
        random.seed(3000 + i)
        print(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
        res = build_pair(h, a).simulate()
        r = analyse(res, n)
        RESULTS.append(r)
        print(f"      score {res.home_goals}-{res.away_goals}", flush=True)

    print("\n" + "=" * 74)
    nm = max(n, 1)
    S = sum(r["shots"] for r in RESULTS)
    C = sum(r["creators"] for r in RESULTS)
    SM = sum(r["start_missing"] for r in RESULTS)
    EM = sum(r["end_missing"] for r in RESULTS)
    CC = sum(len(r["cc"]) for r in RESULTS)
    G = sum(r["goals"] for r in RESULTS)
    print(f"VERDICT over {n} matches")
    print(f"  ledger shots {S}, with a creator {C}")
    print(f"    creator START missing {SM}/{C}    END missing {EM}/{C}")
    allg = sorted(g for r in RESULTS for g in r["gaps"])
    if allg:
        print(f"    pass-end -> shot gap: median {allg[len(allg)//2]:.1f} m, "
              f"max {allg[-1]:.1f} m")
    print(f"  engine CHANCE_CREATED events {CC} = {CC/nm:.1f}/match")
    alll = sorted(d for r in RESULTS for d in r["lens"])
    if alll:
        inb = sum(1 for d in alll if 5.0 <= d <= 20.0)
        print(f"    start->end in [5,20] m  {inb}/{len(alll)}")
    print(f"  assists: agree {sum(r['agree'] for r in RESULTS)}, "
          f"DISAGREE {sum(r['disagree'] for r in RESULTS)}, "
          f"no ledger key pass {sum(r['noreal'] for r in RESULTS)}  (of {G} goals)")


if __name__ == "__main__":
    main()