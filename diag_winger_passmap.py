"""Diagnostic: why does the winger make so many passes from outside the box /
central areas instead of hugging the touchline?

Measures, for every winger in a set of seeded matches:
  1. Pass-origin lateral deviation from the winger's own touchline anchor
     (|origin_y - home_y|) — the "how far inside is he passing from" metric.
  2. Carry y-step magnitudes for the winger (random-walk check).
  3. The same pass-origin metric under a HIGH-press vs a PARK-THE-BUS opponent,
     to test whether "defenders not pressing" is the driver.
"""
import random
import sys
from collections import defaultdict
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tests import HOME_STARTERS, build_match
from match_engine import TeamStyle, PlayingStyle, Intensity

ANCHOR_LEFT = 10.0   # left touchline reference (StatsBomb frame)
ANCHOR_RIGHT = 58.0  # right touchline reference


def run_match(seed, away_style, away_press):
    engine, home, away = build_match(
        home_style=TeamStyle.ATTACKING, away_style=away_style, seed=seed,
    )
    result = engine.simulate()
    pe = engine.position_engine
    stats = {}
    for wname in ("Adri Vela", "Percy"):
        if wname not in (pe.states or {}):
            continue
        state = pe.states[wname]
        anchor = state.home_y
        side = "left" if anchor < 34.0 else "right"
        origin_dev = []
        origin_x = []
        carry_dy = []
        all_touch_dev = []
        carry_chain_dev = []   # pass-origin lateral dev when a carry preceded
        direct_receive_dev = []  # pass-origin lateral dev straight from a receipt
        pass_types = defaultdict(int)
        prev = None
        for e in result.timeline:
            if getattr(e, "player", None) != wname:
                continue
            etn = getattr(getattr(e, "event_type", None), "name", "") or ""
            ox, oy = e.location_x, e.location_y
            all_touch_dev.append(abs(oy - anchor))
            if etn in ("PASS", "THROUGH_BALL", "CROSS_ATTEMPT", "SHOT_ON_TARGET",
                       "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL"):
                pass_types[etn] += 1
                d_dev = abs(oy - anchor)
                origin_dev.append(d_dev)
                origin_x.append(ox)
                if prev is not None and prev == "CARRY":
                    carry_chain_dev.append(d_dev)
                elif prev is not None and prev in ("BALL_RECEIPT", "PASS"):
                    direct_receive_dev.append(d_dev)
            elif etn == "CARRY":
                cy0, cy1 = e.location_y, e.end_y
                if cy0 is not None and cy1 is not None:
                    carry_dy.append(cy1 - cy0)
            prev = etn
        stats[wname] = dict(
            side=side, anchor=round(anchor, 1),
            n_passes=sum(pass_types.values()),
            pass_types=dict(pass_types),
            mean_dev=round(mean(origin_dev), 1) if origin_dev else None,
            dev_bins=hist(origin_dev),
            mean_touch_dev=round(mean(all_touch_dev), 1) if all_touch_dev else None,
            mean_origin_x=round(mean(origin_x), 1) if origin_x else None,
            n_carries=len(carry_dy),
            carry_dy_mean=round(mean(carry_dy), 2) if carry_dy else None,
            carry_dy_absmean=round(mean([abs(d) for d in carry_dy]), 2) if carry_dy else None,
            carry_dy_pos=sum(1 for d in carry_dy if d > 0),
            carry_dy_neg=sum(1 for d in carry_dy if d < 0),
            post_carry_dev=round(mean(carry_chain_dev), 1) if carry_chain_dev else None,
            post_carry_n=len(carry_chain_dev),
            direct_receive_dev=round(mean(direct_receive_dev), 1) if direct_receive_dev else None,
            direct_receive_n=len(direct_receive_dev),
        )
    return result, stats


def hist(devs, edges=(0, 6, 12, 20, 30, 99)):
    out = {}
    labels = ("0-6m on flank", "6-12m half-space edge", "12-20m inside",
              "20-30m central", "30m+ far central")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        out[lab] = sum(1 for d in devs if lo <= d < hi)
    return out


def main():
    combos = [
        ("AWAY_HIGH_PRESS", TeamStyle.GEGENPRESSING, PlayingStyle.HIGH_PRESS),
        ("AWAY_PARK_BUS",   TeamStyle.PARK_THE_BUS,  PlayingStyle.LOW_BLOCK),
    ]
    for label, away_style, away_press in combos:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        agg = defaultdict(list)
        for seed in range(1, 3):
            res, stats = run_match(seed, away_style, away_press)
            for wname, s in stats.items():
                print(f"  seed={seed} {wname:<12} anchor_y={s['anchor']:<6} "
                      f"passes={s['n_passes']:<3} mean|lateral|={s['mean_dev']} "
                      f"postCARRY_passing_from={s.get('post_carry_dev')}(n={s.get('post_carry_n')}) "
                      f"directReceive_pass_from={s.get('direct_receive_dev')}(n={s.get('direct_receive_n')}) "
                      f"carries={s['n_carries']} dy_absmean={s['carry_dy_absmean']}")
                for k, v in s.items():
                    agg[(wname, k)].append(v)
        for wname in ("Adri Vela", "Percy"):
            key = (wname, "mean_dev")
            vals = [v for v in agg.get(key, []) if v is not None]
            if vals:
                print(f"\n  AVERAGE {wname}: mean lateral dev of pass origin = {mean(vals):.1f} m")
            key = (wname, "dev_bins")
            bins = agg.get(key, [])
            if bins:
                tot = defaultdict(int)
                for b in bins:
                    for k, v in b.items():
                        tot[k] += v
                print(f"  {wname} pass-origin distribution: {dict(tot)}")


if __name__ == "__main__":
    main()
