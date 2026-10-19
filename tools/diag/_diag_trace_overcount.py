"""DISTANCE AUDIT PART 2 -- why does the PossessionEpisode trace book 405 km
when the 10 Hz integrator that actually watched those players walk books 74 km?

PART 1 established:
    distance_covered = DNA template (41%) + 0.45 x physics (59%)
    physics = writer(a) offball integrator 74 km  + writer(b) episode traces 405 km
    trace GAP segments are 1% of writer(b) -- the "sparse row teleport" theory
    is DEAD. 99% of the trace distance sits in consecutive-tick segments.

Two candidates remain for why the trace is 5.5x the integrator:
  (A) PER-TICK INTERPOLATION OVERSHOOT. possession_physics.py:1406 says it in
      the source: "the coarse per-tick interpolation makes receivers/defenders
      jump 1-2+ m per 0.1 s tick (an implied 10-20 m/s), which a real GPS would
      never see". Speed is then CLASSIFIED from a label while distance is booked
      from that geometry. Measure: implied_speed = seg_dist/dt vs the label.
  (B) OVERLAPPING EPISODES. 187 accumulate calls per player. A parent chain's
      episode covers a whole possession; the sub-chain it delegates to builds a
      SECOND episode over the same wall-clock seconds. Measure: sum(dt) per
      player against the 5400 s a match actually contains.

Both are measured here, plus which `_snap` note books the distance.
GATES: anchors exist; trace-seconds > 0; the per-tick segment count is non-zero.
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

if not hasattr(PossessionEpisode, "calculate_distance_stats"):
    raise SystemExit("ANCHOR MISSING: PossessionEpisode.calculate_distance_stats")

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

# per-player accumulators
trace_s = defaultdict(float)          # sum of dt  (trace-seconds)
trace_d = defaultdict(float)          # sum of seg_dist
label_d = defaultdict(float)          # sum of dt * speed_label  (what it SHOULD be)
distinct_ticks = defaultdict(set)     # player -> set of rounded tick values
implied_hist = defaultdict(float)     # bucket -> metres
implied_hist_n = defaultdict(int)
by_note = defaultdict(lambda: defaultdict(float))
episodes = [0]
rows_total = [0]

_orig_stats = PossessionEpisode.calculate_distance_stats
_orig_init = PossessionEpisode.__init__


def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    episodes[0] += 1
    self._audit_rows = defaultdict(list)


def _stats(self):
    by_player = defaultdict(list)
    for row in self.trace:
        if row.player in ("ball", "gk_track", "drift"):
            continue
        by_player[row.player].append(row)
    for name, tr in by_player.items():
        for i in range(1, len(tr)):
            a, b = tr[i - 1], tr[i]
            d = math.hypot(b.x - a.x, b.y - a.y)
            dt = b.tick - a.tick
            rows_total[0] += 1
            if not (0.05 <= dt <= 0.15):
                continue
            trace_s[name] += dt
            trace_d[name] += d
            label_d[name] += dt * b.speed_mps
            distinct_ticks[name].add(round(b.tick, 1))
            v = d / dt
            # bands chosen to bracket real football: walk<2, jog 2-4.5,
            # run 4.5-7, sprint 7-8.5, and the impossible bands above 8.5.
            if v < 2.0:
                k = "<2 walk"
            elif v < 4.5:
                k = "2-4.5 jog"
            elif v < 7.0:
                k = "4.5-7 run"
            elif v < 8.5:
                k = "7-8.5 sprint"
            elif v < 12.0:
                k = "8.5-12 IMPOSSIBLE"
            else:
                k = "12+ IMPOSSIBLE"
            implied_hist[k] += d
            implied_hist_n[k] += 1
            if getattr(b, "note", ""):
                by_note[b.note][name] += d
    return _orig_stats(self)


PossessionEpisode.__init__ = _init
PossessionEpisode.calculate_distance_stats = _stats

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

if episodes[0] == 0 or rows_total[0] == 0:
    raise SystemExit(f"VACUOUS: episodes={episodes[0]} rows={rows_total[0]}")

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}")
print(f"episodes created: {episodes[0]}   consecutive-tick segments: {rows_total[0]}")

print()
print("=" * 78)
print("(B) OVERLAP -- trace-seconds per player vs the 5400 s a match contains")
print("=" * 78)
print(f"{'player':<20}{'trace_s':>10}{'distinct_t':>12}{'dup_factor':>12}"
      f"{'trace_km':>10}{'m/s':>8}")
tot_s = tot_d = tot_lab = 0.0
for n in sorted(trace_d, key=lambda k: -trace_d[k])[:10]:
    ts, td = trace_s[n], trace_d[n]
    distinct = len(distinct_ticks[n])
    dup = ts / max(distinct * 0.1, 1e-9)
    print(f"{n:<20}{ts:>10.0f}{distinct:>12}{dup:>12.2f}"
          f"{td/1000:>10.2f}{td/max(ts,1e-9):>8.2f}")
    tot_s += ts
    tot_d += td
print(f"{'ALL PLAYERS':<20}{tot_s:>10.0f}{'':>12}{'':>12}{tot_d/1000:>10.2f}"
      f"{tot_d/max(tot_s,1e-9):>8.2f}")
print(f"  -> 20 outfielders x 5400 s = 108,000 trace-seconds available; "
      f"booked {tot_s:.0f}")

print()
print("=" * 78)
print("(A) INTERPOLATION OVERSHOOT -- geometry vs the engine's own speed label")
print("=" * 78)
print(f"{'band':<22}{'metres':>12}{'share':>8}{'segments':>11}")
tot_i = sum(implied_hist.values())
for k in ("<2 walk", "2-4.5 jog", "4.5-7 run", "7-8.5 sprint",
          "8.5-12 IMPOSSIBLE", "12+ IMPOSSIBLE"):
    m = implied_hist[k]
    if m <= 0:
        continue
    print(f"{k:<22}{m/1000:>12.2f}{m/tot_i*100:>7.1f}%{implied_hist_n[k]:>11}")
lab_km = sum(label_d.values()) / 1000
geo_km = tot_i / 1000
print(f"\n  geometry booked as distance : {geo_km:8.2f} km  "
      f"(mean implied {geo_km*1000/max(tot_s,1e-9):5.2f} m/s)")
print(f"  same segments x speed LABEL : {lab_km:8.2f} km  "
      f"(mean label {lab_km*1000/max(tot_s,1e-9):5.2f} m/s)")
print(f"  GEOMETRY / LABEL            : {geo_km/max(lab_km,1e-9):8.2f}x")
print(f"  implied speed a human cannot exceed 8.5 m/s (30.6 km/h): "
      f"{(implied_hist['8.5-12 IMPOSSIBLE']+implied_hist['12+ IMPOSSIBLE'])/tot_i*100:.1f}% of metres")

print()
print("=" * 78)
print("WHO BOOKS IT -- distance by _snap note (top notes, all players)")
print("=" * 78)
agg = defaultdict(float)
for note, per in by_note.items():
    agg[note] = sum(per.values())
tn = sum(agg.values())
for note, m in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {note:<26}{m/1000:>9.2f} km{m/tn*100:>8.1f}%")