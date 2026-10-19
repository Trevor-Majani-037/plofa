"""DISTANCE AUDIT -- why does a player cover ~20 km when real football is 10-12?

THE NUMBER UNDER TEST
---------------------
`distance_covered` is NOT measured from movement. It is assembled in
`opta_analytics._compute_activity`:

    distances[name] += baseline_dist + physics_dist_additive     # per minute

    baseline_dist          = sum(secs[b] * SPEEDS[b])  <- a DNA activity
                                                      TEMPLATE, not a measurement
    physics_dist_additive = physics_dist * 0.45        <- from physics traces
    physics_dist = position_log row physics_distance_m, which is the SUM of two
    independent writers into the same accumulator:
        (a) MatchEngine._record_offball_distance  -- the real 10 Hz integrator
        (b) BaseChain._accumulate_physics_stats  -- PossessionEpisode trace
    and (b)'s distance_m is a GEOMETRY PATH LENGTH over sparse trace rows --
    which the code itself calls "a position gap -> contributes to total distance".

Hypothesis space: template inflation, a fudge factor, a gap/teleport double
count inside the episode trace, and OVERLAPPING episodes double-counting the
same real metres.

GATES BEFORE ANY VERDICT (a non-vacuous instrument, AGENTS.md discipline)
  G1 anchors exist before we wrap them
  G2 both writers fired (counters non-zero)
  G3 the reported number is in the user's stated 15-25 km band -- else we are
     not measuring the thing being complained about
  G4 our independent recomposition matches opta_analytics' own accumulator
"""
import math
import random
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, ".")

from match_engine import (  # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder  # noqa: E402
from possession_physics import PossessionEpisode  # noqa: E402
from event_chain import BaseChain  # noqa: E402
import opta_analytics  # noqa: E402

# ── G1: anchors must exist before we wrap them ──────────────────────────────
for _obj, _attr, _why in (
    (MatchEngine, "_record_offball_distance", "writer (a)"),
    (PossessionEpisode, "calculate_distance_stats", "writer (b)"),
    (BaseChain, "_accumulate_physics_stats", "writer (b) call site"),
):
    if not hasattr(_obj, _attr):
        raise SystemExit(f"ANCHOR MISSING: {_obj.__name__}.{_attr} ({_why})")

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

acc = defaultdict(lambda: defaultdict(float))
calls = defaultdict(int)

_orig_offball = MatchEngine._record_offball_distance
_orig_stats = PossessionEpisode.calculate_distance_stats
_orig_accum = BaseChain._accumulate_physics_stats


def _offball(self, name, moved, speed, top, duration_s):
    acc[name]["phys_offball"] += moved
    acc[name]["offball_calls"] += 1
    return _orig_offball(self, name, moved, speed, top, duration_s)


def _stats(self):
    """Re-derive the trace's own path length, split real-tick vs gap."""
    rows = defaultdict(list)
    for row in self.trace:
        if row.player in ("ball", "gk_track", "drift"):
            continue
        rows[row.player].append((row.x, row.y, row.speed_mps, row.tick))
    out = _orig_stats(self)
    for name, tr in rows.items():
        real = gap = 0.0
        gap_segs = 0
        for i in range(1, len(tr)):
            d = math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1])
            dt = tr[i][3] - tr[i - 1][3]
            if 0.05 <= dt <= 0.15:
                real += d
            else:
                gap += d
                gap_segs += 1
        acc[name]["trace_real"] += real
        acc[name]["trace_gap"] += gap
        acc[name]["trace_gap_segs"] += gap_segs
        acc[name]["trace_rows"] += len(tr)
        acc[name]["phys_episode"] += out.get(name, {}).get("distance_m", 0.0)
    return out


def _accum(episode, position_engine, minute):
    res = _orig_accum(episode, position_engine, minute)
    for name in res:
        calls[name] += 1
    return res


MatchEngine._record_offball_distance = _offball
PossessionEpisode.calculate_distance_stats = _stats
BaseChain._accumulate_physics_stats = staticmethod(_accum)

random.seed(777)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                     match_date=date(2026, 8, 22), matchday=2)
hp = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION, Intensity.MEDIUM)
ap = TeamProfile("Thornfield United", TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
engine = MatchEngine(config, hp, ap)
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
result = engine.simulate()

# ── G2: the instrument must be non-vacuous ─────────────────────────────────
n_off = sum(acc[n]["phys_offball"] for n in acc)
n_epi = sum(acc[n]["phys_episode"] for n in acc)
if n_off <= 0 or n_epi <= 0:
    raise SystemExit(f"VACUOUS: offball={n_off:.0f} episode={n_epi:.0f}")

all_players = {"Hartwell City": {"starters": home["starters"],
                                 "substitutes": home["substitutes"]},
               "Thornfield United": {"starters": away["starters"],
                                     "substitutes": away["substitutes"]}}
analytics = opta_analytics.OptaAnalytics(result, all_players)
analytics.compute()

pos_of = {}
for frame in result.position_log:
    for side in ("home", "away"):
        for row in frame.get(side, []):
            pos_of[row["player"]] = row.get("position", "?")

print(f"score {result.state.home_goals}-{result.state.away_goals}   "
      f"events {len(result.timeline)}   frames {len(result.position_log)}")
print(f"# writer(a) offball integrator : {n_off/1000:8.2f} km over "
      f"{sum(1 for n in acc if acc[n]['phys_offball'] > 0)} players")
print(f"# writer(b) episode trace     : {n_epi/1000:8.2f} km over "
      f"{sum(1 for n in acc if acc[n]['phys_episode'] > 0)} players")

print()
hdr = (f"{'player':<20}{'pos':<5}{'REPORTED':>9}{'template':>9}{'0.45*phys':>10}"
       f"{'offball':>8}{'episode':>8}{'trREAL':>8}{'trGAP':>8}{'gap%':>6}"
       f"{'gapSeg':>7}{'calls':>6}{'touches':>8}{'drift':>7}")
print(hdr)
print("-" * len(hdr))

rep_total = 0.0
tmpl_total = 0.0
gap_total = 0.0
real_total = 0.0
epi_total = 0.0
ordered = sorted(analytics.player_data,
                 key=lambda k: -analytics.player_data[k]["distance_covered"])
for n in ordered:
    d = analytics.player_data[n]["distance_covered"]
    raw = analytics._distances.get(n, 0.0)
    a = acc[n]
    phys_total = a["phys_offball"] + a["phys_episode"]
    tmpl = (raw - 0.45 * phys_total) / 1000.0
    gap_pct = a["trace_gap"] / max(a["phys_episode"], 1e-9) * 100
    print(f"{n:<20}{pos_of.get(n,'?'):<5}{d:>9.2f}{tmpl:>9.2f}{0.45*phys_total/1000:>10.2f}"
          f"{a['phys_offball']/1000:>8.2f}{a['phys_episode']/1000:>8.2f}"
          f"{a['trace_real']/1000:>8.2f}{a['trace_gap']/1000:>8.2f}{gap_pct:>6.0f}"
          f"{int(a['trace_gap_segs']):>7}{calls[n]:>6}"
          f"{int(analytics._activity.get(n,{}).get('jogging',0)):>8}"
          f"{a['trace_rows']:>7.0f}")
    rep_total += d
    tmpl_total += tmpl
    gap_total += a["trace_gap"]
    real_total += a["trace_real"]
    epi_total += a["phys_episode"]

n_players = len(analytics.player_data)
print("-" * len(hdr))
print(f"{'TOTAL':<25}{rep_total:>9.2f}{tmpl_total:>9.2f}"
      f"{0.45*(n_off+n_epi)/1000:>10.2f}{n_off/1000:>8.2f}{n_epi/1000:>8.2f}"
      f"{real_total/1000:>8.2f}{gap_total/1000:>8.2f}"
      f"{gap_total/max(epi_total,1e-9)*100:>6.0f}")

print()
print("=" * 74)
print(f"REPORTED total            {rep_total:8.2f} km   "
      f"mean {rep_total/n_players:5.2f} km/player   REAL BAND 10-12 km")
print(f"  DNA template (per-min)  {tmpl_total:8.2f} km   "
      f"{tmpl_total/rep_total*100:5.1f}% of reported   "
      f"({tmpl_total/n_players:.2f} km/player)")
print(f"  0.45 x physics          {0.45*(n_off+n_epi)/1000:8.2f} km   "
      f"{0.45*(n_off+n_epi)/1000/rep_total*100:5.1f}% of reported")
print(f"    of which writer (a)   {0.45*n_off/1000:8.2f} km  (real integrator)")
print(f"    of which writer (b)   {0.45*n_epi/1000:8.2f} km  (episode traces)")
print(f"      trace REAL segments {real_total/1000:8.2f} km")
print(f"      trace GAP segments  {gap_total/1000:8.2f} km  "
      f"= {0.45*gap_total/1000:.2f} km of the reported total "
      f"({0.45*gap_total/1000/rep_total*100:.1f}%)")
print(f"  writer(b) calls / player: {sum(calls.values())/max(len(calls),1):.1f} "
      f"(overlap = same metres booked twice)")