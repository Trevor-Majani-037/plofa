"""
PLOFA 26/27 — MIDFIELDER PASS-MAP DIAGNOSTIC
=============================================
Runs seeded matches and measures, for every midfielder (CDM/CM/CAM):

    volume      — total passes per midfielder per match
    accuracy    — completion %
    direction   — forward / lateral / backward split (by angle vs. attack dir)
    origin      — which third the pass STARTS in (def / mid / final)
    length      — pass distance profile (short <15m / medium 15-25 / long >25)

Compares against the real-world reference profile of elite hub midfielders
(Modric 79 passes @87%, Tanaka 75 @96%, Odegaard 58 @95% — pass maps show
dense multidirectional webs in the middle third, NOT a vertical fan).

Usage:
    python diag_midfielder_passmap.py [n_matches]
"""
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date

from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
    EventType,
)
from player_dna import SquadBuilder

PASS_TYPES = (
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_SUCCESS, EventType.CROSS_ATTEMPT,
    EventType.FREEKICK_CROSS, EventType.FREEKICK_DIRECT, EventType.CORNER_TAKEN,
)
OPEN_PLAY_PASS_TYPES = (
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL,
)
MID_POSITIONS = ("CDM", "CM", "CAM")

HOME_STARTERS = [
    ("Keano Walsh",   "GK",  ["sweeper_keeper"],                    29),
    ("Darius Frost",  "LB",  ["aggressive_fullback", "engine"],     24),
    ("Emeka Obi",     "CB",  ["ball_playing_cb"],                   27),
    ("Tavish Crane",  "CB",  ["stopper_defender", "strong"],        30),
    ("Rico Alves",    "RB",  ["overlapping_fullback"],              25),
    ("Mateo Sanz",    "CDM", ["anchor_man", "interceptor"],         28),
    ("Luca Ferrini",  "CM",  ["box_box", "engine"],                 26),
    ("Kofi Mensah",   "CAM", ["creator", "sup_vision"],             24),
    ("Adri Vela",     "LW",  ["dribbler", "speedster"],             22),
    ("Dragan Novak",  "ST",  ["clinical_finisher", "aerial_threat"],29),
    ("Percy",         "RW",  ["grand_dribbler", "inverted",
                               "clinical_finisher", "speedster"],   24),
]
AWAY_STARTERS = [
    ("Pavel Renko",   "GK",  ["sweeper_keeper"],                    31),
    ("Jide Afolabi",  "LB",  [],                                    26),
    ("Bart Kuipers",  "CB",  ["stopper_defender"],                  28),
    ("Ciro Mancini",  "CB",  ["ball_playing_cb"],                   26),
    ("Lee Sung-jin",  "RB",  ["overlapping_fullback"],              28),
    ("Demi Adeola",   "CDM", ["ball_winner", "regista"],            27),
    ("Finn Larsson",  "CM",  ["press_resistant", "engine"],         25),
    ("Kwame Asante",  "CAM", ["playmaker", "creator"],              23),
    ("Bruno Reis",    "LW",  ["speedster", "counter_attacker"],     24),
    ("Nico Strauss",  "ST",  ["fox_in_box", "cold_blooded"],        27),
    ("Tariq El-Amin", "RW",  ["dribbler", "inverted"],              22),
]


def classify_direction(dx_attack: float, dy: float) -> str:
    """Forward/backward/lateral by angle vs attacking direction.
    +-45 deg cone = forward/backward, everything else = lateral."""
    dist = math.hypot(dx_attack, dy)
    if dist < 0.5:
        return "lateral"
    cos_angle = dx_attack / dist
    if cos_angle >= 0.5:
        return "forward"
    if cos_angle <= -0.5:
        return "backward"
    return "lateral"


def run_one_match(seed: int):
    random.seed(seed)
    config = MatchConfig(
        home_team="Hartwell City", away_team="Thornfield United",
        match_date=date(2026, 8, 16), matchday=1,
        referee="Marcus Osei", referee_strictness=0.55,
    )
    home_profile = TeamProfile(
        name="Hartwell City", style=TeamStyle.ATTACKING,
        playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH,
    )
    away_profile = TeamProfile(
        name="Thornfield United", style=TeamStyle.FLUID_COUNTER,
        playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM,
    )
    home_squad = SquadBuilder.build(
        team_name="Hartwell City", starters=HOME_STARTERS, substitutes=[],
        team_superstars=["Percy", "Dragan Novak"], set_piece_takers=["Kofi Mensah"],
    )
    away_squad = SquadBuilder.build(
        team_name="Thornfield United", starters=AWAY_STARTERS, substitutes=[],
        team_superstars=["Kwame Asante"], set_piece_takers=["Kwame Asante"],
    )
    engine = MatchEngine(config, home_profile, away_profile)
    engine.quiet = True
    engine.set_squad("Hartwell City", home_squad["starters"], [])
    engine.set_squad("Thornfield United", away_squad["starters"], [])
    return engine.simulate()


def analyze(result):
    pos_of = {}
    for team, squad in result.squads.items():
        for p in squad["starters"]:
            pos_of[p.name] = p.position

    per_player = defaultdict(lambda: {
        "passes": 0, "completed": 0,
        "dir": defaultdict(int), "origin_third": defaultdict(int),
        "lengths": [], "progressive": 0,
    })

    for ev in result.timeline:
        if ev.event_type not in OPEN_PLAY_PASS_TYPES:
            continue
        pos = pos_of.get(ev.player)
        if pos not in MID_POSITIONS:
            continue
        attacks_right = (ev.team == result.config.home_team)
        ex = ev.end_x if ev.end_x is not None else ev.location_x
        ey = ev.end_y if ev.end_y is not None else ev.location_y
        dx = (ex - ev.location_x) * (1 if attacks_right else -1)
        dy = ey - ev.location_y
        dist = math.hypot(ex - ev.location_x, ey - ev.location_y)

        st = per_player[ev.player]
        st["passes"] += 1
        st["completed"] += 1 if ev.outcome else 0
        st["dir"][classify_direction(dx, dy)] += 1
        ox = ev.location_x if attacks_right else 105 - ev.location_x
        third = "def" if ox < 35 else ("mid" if ox < 70 else "final")
        st["origin_third"][third] += 1
        st["lengths"].append(dist)
        if dx >= 10:
            st["progressive"] += 1
    return per_player


def main(n_matches: int = 4):
    agg = defaultdict(list)
    for seed in range(100, 100 + n_matches):
        result = run_one_match(seed)
        for name, st in analyze(result).items():
            agg[name].append(st)

    print(f"\n{'='*78}")
    print(f"  MIDFIELDER PASS PROFILE — {n_matches} matches per player (open play)")
    print(f"{'='*78}")
    print(f"{'player':<16}{'pos':<5}{'passes':>7}{'acc%':>6}{'fwd%':>6}{'lat%':>6}"
          f"{'bwd%':>6}{'avgLen':>7}{'prog':>5}  origins def/mid/fin")
    all_dir = defaultdict(int)
    tot_p = tot_c = 0
    for name, runs in sorted(agg.items()):
        pos = None
        p = sum(r["passes"] for r in runs) / len(runs)
        c = sum(r["completed"] for r in runs) / len(runs)
        dirs = defaultdict(int)
        thirds = defaultdict(int)
        lens, prog = [], 0.0
        for r in runs:
            for k, v in r["dir"].items():
                dirs[k] += v
                all_dir[k] += v
            for k, v in r["origin_third"].items():
                thirds[k] += v
            lens += r["lengths"]
            prog += r["progressive"]
        tot_p += sum(r["passes"] for r in runs)
        tot_c += sum(r["completed"] for r in runs)
        n = sum(dirs.values()) or 1
        nt = sum(thirds.values()) or 1
        print(f"{name:<16}{'?':<5}{p:>7.1f}{100*c/max(p,1):>6.1f}"
              f"{100*dirs['forward']/n:>6.1f}{100*dirs['lateral']/n:>6.1f}"
              f"{100*dirs['backward']/n:>6.1f}{statistics.mean(lens):>7.1f}"
              f"{prog/len(runs):>5.1f}  "
              f"{100*thirds['def']/nt:>4.0f}/{100*thirds['mid']/nt:>4.0f}/{100*thirds['final']/nt:>4.0f}")

    n = sum(all_dir.values()) or 1
    print(f"\n  ALL MIDFIELDERS: {tot_p/n_matches:.0f} passes/match, "
          f"{100*tot_c/max(tot_p,1):.1f}% completed")
    print(f"  Direction mix: forward {100*all_dir['forward']/n:.1f}% | "
          f"lateral {100*all_dir['lateral']/n:.1f}% | "
          f"backward {100*all_dir['backward']/n:.1f}%")
    print(f"\n  REAL-WORLD REFERENCE (hub midfielders):")
    print(f"    volume 55-80/match | accuracy 87-96% | avg length ~14-17m")
    print(f"    direction ~30-40% fwd, ~30-35% lateral, ~25-30% backward")
    print(f"    origins dominated by MIDDLE third (55-65%)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
