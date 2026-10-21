"""WHERE DOES A CENTRE-BACK'S DISTANCE GO?  (CB pace gap, 2026-10-20)

The 2026-10-19 locomotion work left one named residual: centre-backs cover
8.41 km against a real ~10.5-11.0, while every other role is in or near band.
The Next Move note listed `_JOG_SPEED["CB"]` first and COMPACTION second, and
insisted on MEASURING before changing either.

This probe attributes every metre the 10 Hz integrator books for a centre-back
to the BRANCH of `_offball_move_player` that produced it, by reading the
caller's frame locals at the single writer (`_diag_touch_sites.py`'s join).
That separates the two competing diagnoses:

  * "he is too slow while moving"  -> to_target / orbit mean speed is low
  * "he is not moving at all"      -> the arrived branches dominate the seconds

Then it recomputes the published per-position distance table from
`opta_analytics` so the two instruments can be compared on one screen.

One real match, seed 777, the same fixture as
`_diag_measured_activity_verify.py`. Run from the repo root:

    .venv\\Scripts\\python.exe tools\\diag\\_diag_cb_pace.py
"""
import random
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, ".")

from match_engine import (  # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder  # noqa: E402
import opta_analytics  # noqa: E402

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Jonas Keller", "CM", ["engine"], 25),
    ("Adriela Vela", "LW", ["dribbler"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher"], 29),
    ("Percy", "RW", ["grand_dribbler"], 24),
]
AWAY_STARTERS = [
    ("Bram Osei", "GK", ["shot_stopper"], 28),
    ("Yuri Tanaka", "LB", ["aggressive_fullback"], 23),
    ("Marek Dubois", "CB", ["ball_playing_cb"], 26),
    ("Stefan Berg", "CB", ["stopper_defender"], 29),
    ("Owen Castillo", "RB", ["overlapping_fullback"], 24),
    ("Tariq Aziz", "CDM", ["anchor_man"], 27),
    ("Nils Werner", "CM", ["box_box"], 25),
    ("Diego Ramos", "CAM", ["creator"], 24),
    ("Leo Mbeki", "LW", ["speedster"], 21),
    ("Ivan Petrov", "ST", ["target_man"], 28),
    ("Samir Haddad", "RW", ["dribbler"], 23),
]

# ── attribution: branch of _offball_move_player that produced each step ──────
KEEP = ("pname", "pos", "dist", "arrive", "_in_block", "jog", "_sp_slot",
        "chasing", "involved", "has_ball", "_slot_dist")

_stats = defaultdict(lambda: {"s": 0.0, "m": 0.0, "n": 0})
_pos_km = defaultdict(float)          # position -> metres booked

_orig_rec = MatchEngine._record_offball_distance


def _patched(self, name, moved, speed, top, duration_s):
    try:
        fr = sys._getframe(1).f_locals
        loc = {k: fr.get(k) for k in KEEP}
    except Exception:
        loc = {}
    pos = loc.get("pos") or "?"
    _pos_km[pos] += max(0.0, moved)
    dist = loc.get("dist") or 0.0
    arrive = loc.get("arrive") or 2.0
    if loc.get("_sp_slot") is not None and dist > arrive:
        branch = "set_piece"
    elif loc.get("_in_block") and dist > arrive:
        branch = "block_recover"
    elif dist > arrive:
        branch = "to_target"
    elif loc.get("_in_block"):
        branch = "block_shuffle"
    else:
        branch = "orbit"
    b = _stats[(pos, branch)]
    b["s"] += duration_s
    b["m"] += max(0.0, moved)
    b["n"] += 1
    return _orig_rec(self, name, moved, speed, top, duration_s)


MatchEngine._record_offball_distance = _patched

try:
    random.seed(777)
    home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
    away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
    config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                         match_date=date(2026, 8, 22), matchday=2)
    hp = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA,
                     PlayingStyle.POSSESSION, Intensity.MEDIUM)
    ap = TeamProfile("Thornfield United", TeamStyle.BALANCED,
                     PlayingStyle.MIXED, Intensity.MEDIUM)
    engine = MatchEngine(config, hp, ap)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
    result = engine.simulate()
finally:
    MatchEngine._record_offball_distance = _orig_rec

all_players = {"Hartwell City": {"starters": home["starters"],
                                 "substitutes": home["substitutes"]},
               "Thornfield United": {"starters": away["starters"],
                                     "substitutes": away["substitutes"]}}
analytics = opta_analytics.OptaAnalytics(result, all_players)
analytics.compute()

frames = len(result.position_log)
clock = frames * 60.0
print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  frames {frames} -> {clock:.0f} s/player")

# Position label lives on the position log, not on the published stat dict.
POS = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            POS[row["player"]] = row.get("position", "?")

print()
print("=" * 88)
print("1. METRES BY BRANCH OF _offball_move_player, PER ROLE")
print("=" * 88)
ROLES = ("CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST")
_counts = defaultdict(int)
for p_ in POS.values():
    _counts[p_] += 1
BR = ("to_target", "block_recover", "block_shuffle", "orbit", "set_piece")
print(f"{'pos':<5}{'n':>3}  " + "".join(f"{b[:11]:>13}" for b in BR)
      + f"{'TOTAL':>10}{'s/pl':>8}{'m/s':>7}")
for p_ in ROLES:
    n = _counts.get(p_, 0)
    if not n:
        continue
    cells = []
    tot_m = tot_s = 0.0
    for b in BR:
        d = _stats.get((p_, b))
        if d:
            cells.append(f"{d['m'] / n:>13.0f}")
            tot_m += d["m"]
            tot_s += d["s"]
        else:
            cells.append(f"{'--':>13}")
    print(f"{p_:<5}{n:>3}  " + "".join(cells)
          + f"{tot_m / n:>10.0f}{tot_s / n:>8.0f}"
          f"{tot_m / max(tot_s, 1e-9):>7.2f}")

print()
print("  mean speed (m/s) per branch, so a slow branch is distinguishable from")
print("  a branch that is never entered:")
print(f"{'pos':<5}  " + "".join(f"{b[:11]:>13}" for b in BR))
for p_ in ROLES:
    n = _counts.get(p_, 0)
    if not n:
        continue
    cells = []
    for b in BR:
        d = _stats.get((p_, b))
        cells.append(f"{(d['m'] / d['s']):>13.2f}" if d and d["s"] else f"{'--':>13}")
    print(f"{p_:<5}  " + "".join(cells))

print()
print("  share of STEPPED SECONDS per branch:")
print(f"{'pos':<5}  " + "".join(f"{b[:11]:>13}" for b in BR))
for p_ in ROLES:
    n = _counts.get(p_, 0)
    if not n:
        continue
    tot_s = sum(_stats[(p_, b)]["s"] for b in BR if (p_, b) in _stats)
    cells = []
    for b in BR:
        d = _stats.get((p_, b))
        cells.append(f"{(d['s'] / max(tot_s, 1e-9) * 100):>12.0f}%" if d else f"{'--':>13}")
    print(f"{p_:<5}  " + "".join(cells))

# Position label lives on the position log, not on the published stat dict.
POS = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            POS[row["player"]] = row.get("position", "?")

# The integrator pool above covers every CB in the match (both teams).
n_cb = _counts.get("CB", 0)
cb_m = sum(_stats[("CB", b)]["m"] for b in BR if ("CB", b) in _stats)
cb_s = sum(_stats[("CB", b)]["s"] for b in BR if ("CB", b) in _stats)
if n_cb:
    print(f"\n  per-CB shares (n={n_cb}): "
          f"{cb_m / n_cb / 1000.0:.2f} km booked by the integrator")

print()
print("=" * 88)
print("1b. ONE LINE PER CENTRE-BACK  (integrator vs published)")
print("=" * 88)
_frame_dist = defaultdict(float)
_frame_secs = defaultdict(float)
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            n = row["player"]
            _frame_dist[n] += float(row.get("physics_distance_m", 0.0) or 0.0)
            _frame_secs[n] += sum(float(row.get(k, 0.0) or 0.0) for k in (
                "physics_standing_time_s", "physics_walk_time_s",
                "physics_jog_time_s", "physics_running_time_s",
                "physics_sprint_time_s"))
print(f"{'player':<20}{'pos':<5}{'integrator m':>14}{'log physics m':>15}"
      f"{'published km':>14}{'log s':>8}")
for n in sorted((k for k in POS if str(POS[k]).startswith("CB")),
                key=lambda k: -_frame_dist[k]):
    print(f"{n:<20}{POS[n]:<5}{_pos_km.get('CB', 0) / max(n_cb, 1):>14.0f}"
          f"{_frame_dist[n]:>15.0f}"
          f"{analytics.player_data[n]['distance_covered']:>14.2f}"
          f"{_frame_secs[n]:>8.0f}")

print()
print("=" * 88)
print("2. WHAT THE PUBLISHER SAYS  (distance_covered, per position)")
print("=" * 88)
per_pos = defaultdict(list)
stand = defaultdict(list)
for n, d in analytics.player_data.items():
    per_pos[POS.get(n, "?")].append(d["distance_covered"])
    stand[POS.get(n, "?")].append(d.get("standing_pct", 0.0))
print(f"{'pos':<6}{'n':>4}{'km mean':>10}{'km min':>9}{'km max':>9}{'real band':>16}")
BAND = {"CB": "10.5-11.0", "LB": "11.0-12.0", "RB": "11.0-12.0",
        "CDM": "10.5-11.0", "CM": "11.5-12.5", "CAM": "11.5-12.0",
        "LW": "11.5-12.5", "RW": "11.5-12.5", "ST": "10.5-11.0", "GK": "4.6-5.8"}
for p_ in ("GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST"):
    v = per_pos.get(p_)
    if not v:
        continue
    print(f"{p_:<6}{len(v):>4}{sum(v) / len(v):>10.2f}{min(v):>9.2f}"
          f"{max(v):>9.2f}{BAND.get(p_, ''):>16}")

print()
print("=" * 88)
print("3. THE ARITHMETIC OF THE GAP")
print("=" * 88)
cb_km = per_pos.get("CB")
if cb_km and n_cb:
    mean_km = sum(cb_km) / len(cb_km)
    moving_s = cb_s / n_cb
    avg_moving = (cb_m / max(cb_s, 1e-9))
    print(f"  CB published mean      : {mean_km:.2f} km")
    print(f"  integrator mean speed  : {avg_moving:.2f} m/s while stepped")
    print(f"  stepped seconds / CB   : {moving_s:.0f} s of {clock:.0f}")
    print(f"  => needed for 10.75 km : "
          f"{10750.0 / max(moving_s, 1e-9):.2f} m/s at the same stepped seconds")
    print(f"  => or {10750.0 / max(avg_moving, 1e-9):.0f} stepped s at the "
          f"same pace (there are only {clock:.0f})")
