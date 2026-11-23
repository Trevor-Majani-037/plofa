#!/usr/bin/env python
"""Duel ledger audit: volume + zero-sum (won == opp lost) + source census.

Opta semantics being checked:
  * every dribble attempt is a ground duel attempt (won if completed);
  * every tackle is a ground duel (won if the tackle won);
  * 50-50 ground duels that are not dribbles/tackles exist too;
  * a dispossession is a duel lost ONLY when the winner is named;
  * only a CONTESTED header is an aerial duel;
  * for every duel won by one side there is exactly one duel lost by the
    other, so team A won == team B lost, per duel type.

Runs real matches, reads the SHIPPED exporter ledger (StatAccumulator),
then replays the exporter's duel rules per event source to show which
stream contributes what, and attributes any zero-sum break to events
that carry no opponent (secondary_player) or no paired counterpart.
"""
import os
import sys
import random
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import auto_run_match
from player_dna import SquadBuilder
from match_engine import (MatchEngine, MatchState, MatchPhase, MatchConfig,
                          EventType)
from exporter import StatAccumulator

HOME, AWAY = "Oxton", "Natrican"

GROUND_SRC = (EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL,
              EventType.TACKLE_WON, EventType.TACKLE_LOST,
              EventType.GROUND_DUEL, EventType.FOUL_COMMITTED,
              EventType.FOUL_WON, EventType.DISPOSSESSED)
AERIAL_SRC = (EventType.AERIAL_DUEL, EventType.CLEARANCE)
AERIAL_KINDS = ("aerial duel", "aerial duel(mirror)",
                "contested header", "contested header(mirror)")


def build_match(seed):
    random.seed(seed)
    np.random.seed(seed)
    loader = auto_run_match.get_loader()
    home_raw = loader.build_matchday_squad(HOME, availability=None)
    away_raw = loader.build_matchday_squad(AWAY, availability=None)
    home_squad = SquadBuilder.build(
        team_name=HOME, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"],
        soul_assignments=auto_run_match.SOUL_PLAYERS)
    away_squad = SquadBuilder.build(
        team_name=AWAY, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"],
        soul_assignments=auto_run_match.SOUL_PLAYERS)
    home_style = auto_run_match._resolve_team_profile(
        HOME, home_raw["formation"], True, None, "fluid_counter")
    away_style = auto_run_match._resolve_team_profile(
        AWAY, away_raw["formation"], False, None, "attacking")
    config = MatchConfig(home_team=HOME, away_team=AWAY,
                         match_date=date(2026, 9, 6), matchday=3,
                         season="26/27")
    state = MatchState()
    state.phase = MatchPhase.OPENING
    state.minute = 0
    state.match_clock_s = 0.0
    engine = MatchEngine(config, home_style, away_style)
    engine.state = state
    engine.set_squad(HOME, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(AWAY, away_squad["starters"], away_squad["substitutes"])
    return engine, {HOME: home_squad, AWAY: away_squad}


def wiped_tackles(tl):
    """Same scan the exporter does: a foul wipes its failed tackle."""
    wiped = set()
    for idx, ev in enumerate(tl):
        if (ev.event_type == EventType.FOUL_COMMITTED
                and (ev.metadata or {}).get("from_failed_tackle", False)):
            for j in range(max(0, idx - 5), idx):
                cand = tl[j]
                if (cand.event_type == EventType.TACKLE_LOST
                        and cand.player == ev.player
                        and cand.minute == ev.minute):
                    wiped.add(id(cand))
                    break
    return wiped


def team_of(acc, name):
    s = acc.stats.get(name)
    return s.get("team") if s else None


def duel_opp(e):
    """Passive ledger: winner named in metadata, else the event's opponent."""
    return (e.metadata or {}).get("duel_winner") or e.secondary_player


def audit(seed):
    engine, squads = build_match(seed)
    result = engine.simulate()
    acc = StatAccumulator(result, squads)
    tl = list(result.timeline)
    wiped = wiped_tackles(tl)

    print(f"\n=== seed {seed}: {HOME} {result.home_goals}-{result.away_goals} {AWAY} "
          f"| events {len(tl)} ===", flush=True)

    # ── 1. shipped ledger: team totals + zero-sum ──
    tot = {}
    for team in (HOME, AWAY):
        g_att = g_won = a_att = a_won = 0
        for s in acc.stats.values():
            if s.get("team") != team:
                continue
            g_att += s["ground_duels_att"]
            g_won += s["ground_duels_won"]
            a_att += s["aerial_duels_att"]
            a_won += s["aerial_duels_won"]
        tot[team] = (g_att, g_won, a_att, a_won)
    for team in (HOME, AWAY):
        g_att, g_won, a_att, a_won = tot[team]
        print(f"  {team:9s} ground {g_won:4d}/{g_att:4d} won/att "
              f"(lost {g_att - g_won:4d}) | aerial {a_won:3d}/{a_att:3d} "
              f"(lost {a_att - a_won:3d})")
    hg, hw, ha, haw = tot[HOME]
    ag, aw, aa, aaw = tot[AWAY]
    print(f"  ZERO-SUM ground: {HOME} won {hw} vs {AWAY} lost {ag - aw} "
          f"-> delta {hw - (ag - aw):+d}")
    print(f"  ZERO-SUM ground: {AWAY} won {aw} vs {HOME} lost {hg - hw} "
          f"-> delta {aw - (hg - hw):+d}")
    print(f"  ZERO-SUM aerial: {HOME} won {haw} vs {AWAY} lost {aa - aaw} "
          f"-> delta {haw - (aa - aaw):+d}")
    print(f"  ZERO-SUM aerial: {AWAY} won {aaw} vs {HOME} lost {ha - haw} "
          f"-> delta {aaw - (ha - haw):+d}")

    # ── 2. event census: who carries an opponent? ──
    print("  event census (count / named opponent / paired_with_tackle):")
    census = defaultdict(lambda: [0, 0, 0])
    g50_wins = g50_n = 0
    contested_headers = 0
    for e in tl:
        if e.event_type in GROUND_SRC or e.event_type in AERIAL_SRC:
            c = census[e.event_type.name]
            c[0] += 1
            if duel_opp(e):
                c[1] += 1
            if (e.metadata or {}).get("paired_with_tackle"):
                c[2] += 1
        if e.event_type == EventType.GROUND_DUEL:
            g50_n += 1
            g50_wins += 1 if e.outcome else 0
        if (e.metadata or {}).get("contested_header"):
            contested_headers += 1
    for name in sorted(census):
        n, sec, paired = census[name]
        print(f"    {name:16s} {n:4d} / {sec:4d} / {paired:4d}")
    if g50_n:
        print(f"    branch-4 50-50 carrier win share: {g50_wins}/{g50_n} "
              f"= {g50_wins / g50_n:.1%}")
    print(f"    contested headers (aerial duels): {contested_headers}")

    # ── 3. replay exporter rules per source, per team ──
    src = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # src->team->[att,won]
    not_a_duel = defaultdict(int)  # events correctly excluded from the duel ledger

    def credit(team, kind, att, won):
        if team is None:
            return
        src[kind][team][0] += att
        src[kind][team][1] += won

    for e in tl:
        t = team_of(acc, e.player)
        st = team_of(acc, duel_opp(e)) if duel_opp(e) else None
        et = e.event_type
        md = e.metadata or {}
        if et in (EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL):
            paired = md.get("paired_with_tackle", False)
            if not (duel_opp(e) or paired):
                not_a_duel["dribble, no opponent"] += 1
                continue
            credit(t, "dribble", 1, 1 if e.outcome else 0)
            if not paired:
                if e.outcome:
                    credit(st, "dribble(mirror)", 1, 0)
                elif st:
                    credit(st, "dribble(mirror)", 1, 1)
        elif et == EventType.TACKLE_WON:
            if not duel_opp(e):
                not_a_duel["tackle won, no opponent"] += 1
                continue
            credit(t, "tackle won", 1, 1)
            credit(st, "tackle won(mirror)", 1, 0)
        elif et == EventType.TACKLE_LOST:
            if id(e) in wiped:
                continue
            if not duel_opp(e):
                not_a_duel["tackle lost, no opponent"] += 1
                continue
            credit(t, "tackle lost", 1, 0)
        elif et == EventType.GROUND_DUEL:
            if not duel_opp(e):
                not_a_duel["ground duel, no opponent"] += 1
                continue
            _pd = md.get("presser_distance")
            if _pd is not None and _pd > 3.5:
                not_a_duel["press too far (not a contest)"] += 1
                continue
            credit(t, "50-50 ground", 1, 1 if e.outcome else 0)
            credit(st, "50-50 ground(mirror)", 1, 0 if e.outcome else 1)
        elif et == EventType.FOUL_COMMITTED:
            if duel_opp(e):
                credit(t, "foul committed", 1, 0)
            else:
                not_a_duel["foul committed, no victim"] += 1
        elif et == EventType.FOUL_WON:
            if duel_opp(e):
                credit(t, "foul won", 1, 1)
            else:
                not_a_duel["foul won, no fouler"] += 1
        elif et == EventType.DISPOSSESSED:
            if md.get("micro_carry"):
                not_a_duel["micro-carry loss (not a duel)"] += 1
                continue
            if not duel_opp(e):
                not_a_duel["dispossessed, no tackler (turnover)"] += 1
                continue
            credit(t, "dispossessed", 1, 0)
            credit(st, "dispossessed(mirror)", 1, 1)
        elif et == EventType.AERIAL_DUEL:
            if not duel_opp(e):
                not_a_duel["aerial duel, no opponent"] += 1
                continue
            _pd = md.get("presser_distance")
            if _pd is not None and _pd > 3.5:
                not_a_duel["aerial press too far (not a contest)"] += 1
                continue
            credit(t, "aerial duel", 1, 1 if e.outcome else 0)
            credit(st, "aerial duel(mirror)", 1, 0 if e.outcome else 1)
        elif et == EventType.CLEARANCE:
            if md.get("contested_header"):
                credit(t, "contested header", 1, 1 if e.outcome else 0)
                credit(st, "contested header(mirror)", 1, 0 if e.outcome else 1)
            elif md.get("headed"):
                not_a_duel["unopposed header (not a duel)"] += 1

    print("  ground duel att/won by source (replay of exporter rules):")
    for kind in sorted(src):
        if kind in AERIAL_KINDS:
            continue
        h = src[kind][HOME]
        a = src[kind][AWAY]
        print(f"    {kind:26s} {HOME} {h[1]:4d}/{h[0]:4d}   {AWAY} {a[1]:4d}/{a[0]:4d}")
    print("  aerial duel att/won by source:")
    for kind in sorted(src):
        if kind not in AERIAL_KINDS:
            continue
        h = src[kind][HOME]
        a = src[kind][AWAY]
        print(f"    {kind:26s} {HOME} {h[1]:4d}/{h[0]:4d}   {AWAY} {a[1]:4d}/{a[0]:4d}")

    # replay sanity vs shipped ledger
    for team in (HOME, AWAY):
        r_att = sum(v[team][0] for k, v in src.items() if k not in AERIAL_KINDS)
        r_won = sum(v[team][1] for k, v in src.items() if k not in AERIAL_KINDS)
        g_att, g_won, _, _ = tot[team]
        assert (r_att, r_won) == (g_att, g_won), (team, r_att, r_won, g_att, g_won)
        ra_att = sum(v[team][0] for k, v in src.items() if k in AERIAL_KINDS)
        ra_won = sum(v[team][1] for k, v in src.items() if k in AERIAL_KINDS)
        _, _, a_att, a_won = tot[team]
        assert (ra_att, ra_won) == (a_att, a_won), (team, ra_att, ra_won, a_att, a_won)
    print("  replay matches shipped ledger (ground AND aerial): OK")

    print("  events correctly NOT counted as duels:")
    for reason, n in sorted(not_a_duel.items()):
        print(f"    {reason:42s} {n:4d}")

    # ── 4. same contest logged twice? adjacent events, same player pair ──
    duel_types = set(GROUND_SRC) | {EventType.AERIAL_DUEL}
    overlaps = 0
    examples = []
    for i, e in enumerate(tl):
        if e.event_type not in (EventType.TACKLE_WON, EventType.TACKLE_LOST):
            continue
        if not duel_opp(e):
            continue
        pair = {e.player, duel_opp(e)}
        for j in range(max(0, i - 6), i):
            o = tl[j]
            if o.event_type not in duel_types:
                continue
            if o.event_type in (EventType.TACKLE_WON, EventType.TACKLE_LOST):
                continue
            opair = {o.player, o.secondary_player} if o.secondary_player else {o.player}
            if len(opair) == 2 and pair <= opair:
                overlaps += 1
                if len(examples) < 5:
                    examples.append((o.minute, o.event_type.name, o.player,
                                     e.event_type.name, e.player))
                break
    print(f"  tackle events whose player pair already had a duel/dribble/"
          f"dispossession in the prior 6 events: {overlaps}")
    for ex in examples:
        print(f"    min {ex[0]}: {ex[1]}({ex[2]}) then {ex[3]}({ex[4]})")

    # ── 5. per-player volume sanity ──
    top = sorted(acc.stats.values(), key=lambda s: -s["ground_duels_att"])[:5]
    print("  top ground-duel players (att/won):")
    for s in top:
        print(f"    {s['player']:22s} {s.get('position','?'):4s} "
              f"{s['ground_duels_won']:3d}/{s['ground_duels_att']:3d} "
              f"team {s.get('team')}")
    return tot, not_a_duel


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [123, 223, 323]
    agg = defaultdict(int)
    for sd in seeds:
        _, nd = audit(sd)
        for k, v in nd.items():
            agg[k] += v
    print("\n=== aggregate non-duel events over seeds ===")
    for k, v in sorted(agg.items()):
        print(f"  {k:42s} {v:4d}")