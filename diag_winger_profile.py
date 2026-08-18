"""
diag_winger_profile.py — Checkpoint 24 regression tool.

Measures per-winger attacking profiles from simulated matches and compares
them against a real-world touchline-winger reference (Doku-style card):
    passes ~30-45 @ 88-95% | crosses 2-6/match (low accuracy is normal)
    long balls ~1-2/match | avg pass length ~10-16m
    carries heavy with ~45-65% success | dribble attempts 4-10, ~50% success
    dispossessed 2-5/match

Archetype diversity must survive: traditional wingers/crossers deliver more
crosses than inverted wingers; dribblers attempt more take-ons. What must
NOT come back: 12+ crosses/match, 6+ long balls/match, 95%+ carry success.

Usage:  python diag_winger_profile.py [n_matches]
"""

import math
import sys
from collections import defaultdict

from diag_midfielder_passmap import run_one_match
from match_engine import EventType

PASS_TYPES = (EventType.PASS, EventType.PROGRESSIVE_PASS,
              EventType.SWITCH_OF_PLAY, EventType.THROUGH_BALL)


def measure(seeds):
    agg = defaultdict(lambda: defaultdict(float))
    arche = {}
    matches = 0
    for seed in seeds:
        res = run_one_match(seed)
        matches += 1
        pos_of = {p.name: p.position for t, s in res.squads.items()
                  for p in s["starters"]}
        for ev in res.timeline:
            if pos_of.get(ev.player) not in ("LW", "RW"):
                continue
            st = agg[ev.player]
            ar = ev.team == res.config.home_team
            ox = ev.location_x if ar else 105 - ev.location_x
            et = ev.event_type
            if et in PASS_TYPES:
                st["passes"] += 1
                st["pass_cmp"] += ev.outcome
                if ox >= 52.5:
                    st["p_opp"] += 1
                else:
                    st["p_own"] += 1
                if ev.metadata.get("is_long"):
                    st["long"] += 1
                if ev.metadata.get("cross"):
                    st["cross"] += 1
                    st["cross_cmp"] += ev.outcome
                ex = ev.end_x if ev.end_x is not None else ev.location_x
                ey = ev.end_y if ev.end_y is not None else ev.location_y
                st["len_sum"] += math.hypot(ex - ev.location_x,
                                            ey - ev.location_y)
            elif et in (EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS):
                st["cross"] += 1
                st["cross_cmp"] += ev.outcome
            elif et == EventType.CARRY:
                st["carries"] += 1
                st["carry_ok"] += ev.outcome
            elif et in (EventType.DRIBBLE_ATTEMPT, EventType.DRIBBLE_SUCCESS,
                        EventType.DRIBBLE_FAIL):
                st["drb"] += 1
                st["drb_ok"] += (et == EventType.DRIBBLE_SUCCESS)
            elif et == EventType.DISPOSSESSED:
                st["disp"] += 1
            elif et in (EventType.SHOT_ATTEMPT, EventType.SHOT_ON_TARGET,
                        EventType.SHOT_OFF_TARGET, EventType.GOAL):
                st["shots"] += 1
    for team, squad in res.squads.items():
        for p in squad["starters"]:
            if p.position in ("LW", "RW"):
                arche[p.name] = getattr(p.dna, "archetype", "") or "-"
    return agg, arche, matches


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    agg, arche, n = measure(range(300, 300 + n))
    hdr = (f"{'winger':<16}{'arch':<18}{'pass':>5}{'acc%':>6}{'cross':>6}"
           f"{'long':>5}{'avgLen':>7}{'carry':>6}{'cOK%':>6}{'drb':>5}"
           f"{'dOK%':>6}{'disp':>5}{'shots':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, st in sorted(agg.items(), key=lambda kv: -kv[1]["passes"]):
        p = max(st["passes"], 1)
        c = max(st["carries"], 1)
        d = max(st["drb"], 1)
        print(f"{name:<16}{arche.get(name, '-'):<18}{st['passes']/n:>5.1f}"
              f"{100*st['pass_cmp']/p:>6.1f}{st['cross']/n:>6.1f}"
              f"{st['long']/n:>5.1f}{st['len_sum']/p:>7.1f}"
              f"{st['carries']/n:>6.1f}{100*st['carry_ok']/c:>6.1f}"
              f"{st['drb']/n:>5.1f}{100*st['drb_ok']/d:>6.1f}"
              f"{st['disp']/n:>5.1f}{st['shots']/n:>6.1f}")
    print("\nREAL-WINGER REFERENCE: 30-45 passes @ 80-92% | 2-6 crosses | "
          "1-2 long | ~10-18m avg | 4-10 take-ons @ 45-70% | 2-5 dispossessed"
          "\n(note: Opta 'carries' complete ~90% — the coin-flip stat is "
          "take-ons/dribbles, not plain carries)")


if __name__ == "__main__":
    main()
