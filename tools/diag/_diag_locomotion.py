"""WHERE DOES EVERY PUBLISHED ACTIVITY SECOND COME FROM?

The user asked two questions about the exported standing/walking/jogging/
running/sprinting table:

  1. is it TRUE DATA or a HEURISTIC?
  2. why do players stand so much (the locomotion gap)?

Both are answered by attributing every second to the LINE OF CODE that
produced it (`sys._getframe` join, the `_diag_touch_sites.py` method), not by
inferring intent from a context label.

Writers of measured seconds:
  A. MatchEngine._record_offball_distance       -> the 10 Hz shape integrator
  B. BaseChain._accumulate_physics_stats        -> the PossessionEpisode trace
  C. (everything else)                          -> BASE_MINUTE_ACTIVITY table
                                                   in opta_analytics

Also counts the ticks the integrator NEVER SAW:
  * on-ball players skipped for the whole minute (`_on_ball_this_minute`)
  * keepers (early return before the integrator body)
  * `_offball_move_player` early returns

and the integrator's own speed distribution, so "arrive and stop" can be
separated from "was never stepped".

One real match. Run from the repo root:

    .venv\\Scripts\\python.exe tools\\diag\\_diag_locomotion.py
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
import position_engine as PE  # noqa: E402
from event_chain import BaseChain  # noqa: E402

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

BANDS = ["standing", "walking", "jogging", "running", "sprinting"]

# ── instrumentation ─────────────────────────────────────────────────────────
orig_record = PE.PositionEngine.record_physics_distance
orig_move = MatchEngine._offball_move_player

by_source = defaultdict(lambda: defaultdict(float))   # name -> source -> s
dist_by_source = defaultdict(float)
move_stats = defaultdict(lambda: {"ticks": 0, "moving": 0, "still": 0,
                                  "speed_sum": 0.0, "moved_sum": 0.0,
                                  "speed_moving_sum": 0.0})
phase_stats = defaultdict(lambda: {"ticks": 0, "still": 0,
                                   "moved_sum": 0.0, "speed_sum": 0.0})
_phase = {"name": "?"}
orig_offball_run = MatchEngine._offball_run
orig_continuous = MatchEngine._continuous_offball_phase


def offball_run(self, *a, **kw):
    _phase["name"] = "live"
    return orig_offball_run(self, *a, **kw)


def continuous_offball_phase(self, *a, **kw):
    _phase["name"] = "dead"
    return orig_continuous(self, *a, **kw)
skipped = defaultdict(lambda: {"on_ball": 0, "gk": 0,
                                "on_ball_live": 0, "on_ball_dead": 0})
cur = {}   # player -> (phase, in_block) for THIS tick


def record_physics_distance(self, player_name, distance_m=0.0, duration_s=0.0,
                            speed_mps=0.0, walk_time_s=0.0, jog_time_s=0.0,
                            sprint_time_s=0.0, standing_time_s=0.0,
                            running_time_s=0.0, **kw):
    f = sys._getframe(1)
    src = f.f_code.co_name
    # A: the shape integrator writes five bands through `duration_s`
    # B: the episode trace writes three bands explicitly (walk/jog/sprint)
    st = self.states.get(player_name)
    if src == "_accumulate_physics_stats":
        # classify the trace's own THREE bands into the same five the
        # module claims to publish.  <2.0 m/s currently books as walking
        # even at 0.0 m/s, and 2.0-7.0 books as jogging even at 6.9.
        by_source[player_name]["trace.walk"] += walk_time_s
        by_source[player_name]["trace.jog"] += jog_time_s
        by_source[player_name]["trace.sprint"] += sprint_time_s
        by_source[player_name]["trace.SUM"] += (walk_time_s + jog_time_s
                                                + sprint_time_s)
    else:
        band = PE.ACTIVITY_BAND_NAMES[PE.activity_band_index(speed_mps)]
        by_source[player_name]["integrator." + band] += duration_s
        by_source[player_name]["integrator.SUM"] += duration_s
        mv = move_stats[player_name]
        mv["ticks"] += 1
        mv["speed_sum"] += speed_mps
        mv["moved_sum"] += distance_m
        if distance_m > 0.0:
            mv["moving"] += 1
            mv["speed_moving_sum"] += speed_mps
        else:
            mv["still"] += 1
        ph, in_block = cur.get(player_name, ("?", False))
        key = f"{ph}/{'block' if in_block else 'open'}"
        ps = phase_stats[key]
        ps["ticks"] += 1
        ps["speed_sum"] += speed_mps
        ps["moved_sum"] += distance_m
        if distance_m <= 0.0:
            ps["still"] += 1
    dist_by_source[src] += distance_m
    return orig_record(self, player_name, distance_m=distance_m,
                       duration_s=duration_s, speed_mps=speed_mps,
                       walk_time_s=walk_time_s, jog_time_s=jog_time_s,
                       sprint_time_s=sprint_time_s,
                       standing_time_s=standing_time_s,
                       running_time_s=running_time_s, **kw)


def offball_move_player(self, pname, *a, **kw):
    st = self.position_engine.states.get(pname) if self.position_engine else None
    team = a[0] if a else ""
    has_ball = a[3] if len(a) > 3 else False
    in_block = bool((not has_ball) and self._defensive_block.get(team))
    ph = _phase["name"]
    cur[pname] = (ph, in_block)
    if st is None or getattr(st, "position", "") == "GK":
        skipped[pname]["gk"] += 1
    elif pname in self._on_ball_this_minute:
        skipped[pname]["on_ball"] += 1
        skipped[pname]["on_ball_" + ph] += 1
    return orig_move(self, pname, *a, **kw)


PE.PositionEngine.record_physics_distance = record_physics_distance
MatchEngine._offball_move_player = offball_move_player
MatchEngine._offball_run = offball_run
MatchEngine._continuous_offball_phase = continuous_offball_phase

try:
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
finally:
    PE.PositionEngine.record_physics_distance = orig_record
    MatchEngine._offball_move_player = orig_move
    MatchEngine._offball_run = orig_offball_run
    MatchEngine._continuous_offball_phase = orig_continuous

all_players = {"Hartwell City": {"starters": home["starters"],
                                 "substitutes": home["substitutes"]},
               "Thornfield United": {"starters": away["starters"],
                                     "substitutes": away["substitutes"]}}
analytics = opta_analytics.OptaAnalytics(result, all_players)
analytics.compute()

pos = {}
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            pos[row["player"]] = row.get("position", "?")
frames = len(result.position_log)
available = frames * 60.0

print(f"score {result.state.home_goals}-{result.state.away_goals}  "
      f"events {len(result.timeline)}  frames {frames}")

print()
print("=" * 106)
print("A0. RAW ROWS vs PUBLISHED, five bands (seconds)")
print("=" * 106)
_KEYS = [("standing", "physics_standing_time_s"),
         ("walking", "physics_walk_time_s"),
         ("jogging", "physics_jog_time_s"),
         ("running", "physics_running_time_s"),
         ("sprinting", "physics_sprint_time_s")]
raw = defaultdict(lambda: defaultdict(float))
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            for band, key in _KEYS:
                raw[row["player"]][band] += row.get(key, 0.0)
print(f"{'player':<19}{'pos':<4}" + "".join(f"{b:>10}" for b, _ in _KEYS)
      + f"{'RAW tot':>9} |" + "".join(f"{b[:5]:>10}" for b, _ in _KEYS)
      + f"{'PUB tot':>9}")
for n in sorted(raw, key=lambda k: -sum(analytics._activity.get(k, {}).values()))[:12]:
    pub = analytics._activity.get(n, {})
    print(f"{n:<19}{pos.get(n,'?'):<4}"
          + "".join(f"{raw[n][b]:>10.0f}" for b, _ in _KEYS)
          + f"{sum(raw[n].values()):>9.0f} |"
          + "".join(f"{pub.get(b,0):>10.0f}" for b, _ in _KEYS)
          + f"{sum(pub.values()):>9.0f}")

print()
print("=" * 106)
print("A. WHERE THE PUBLISHED SECONDS COME FROM (whole match, seconds)")
print("=" * 106)
print(f"{'player':<19}{'pos':<4}{'integr':>9}{'trace':>9}{'MEASURED':>10}"
      f"{'TEMPLATE':>10}{'published':>11}{'meas%':>8}{'stand%':>9}")
rows = []
for n in analytics._activity:
    p = pos.get(n, "?")
    pub = analytics._activity[n]
    integ = by_source[n]["integrator.SUM"]
    trace = by_source[n]["trace.SUM"]
    meas = integ + trace
    pub_total = sum(pub.values())
    tpl = pub_total - meas
    rows.append((p, n, integ, trace, meas, tpl, pub_total))
for p, n, integ, trace, meas, tpl, pub_total in sorted(
        rows, key=lambda r: -r[6]):
    print(f"{n:<19}{p:<4}{integ:>9.0f}{trace:>9.0f}{meas:>10.0f}{tpl:>10.0f}"
          f"{pub_total:>11.0f}{meas/max(pub_total,1e-9)*100:>7.0f}%"
          f"{analytics._activity[n]['standing']/max(pub_total,1e-9)*100:>7.0f}%")

print()
print("=" * 106)
print("B. INTEGRATOR SPEED DISTRIBUTION (only the ticks it actually saw)")
print("=" * 106)
print(f"{'player':<19}{'pos':<4}{'ticks':>8}{'moving':>8}{'still':>8}"
      f"{'mean all':>10}{'mean moving':>13}{'m/tick':>9}")
for n, mv in sorted(move_stats.items(), key=lambda kv: -kv[1]["ticks"]):
    if pos.get(n) == "GK":
        continue
    t = mv["ticks"]
    print(f"{n:<19}{pos.get(n,'?'):<4}{t:>8.0f}{mv['moving']:>8.0f}"
          f"{mv['still']:>8.0f}{mv['speed_sum']/max(t,1):>10.2f}"
          f"{mv['speed_moving_sum']/max(mv['moving'],1):>13.2f}"
          f"{mv['moved_sum']/max(t,1):>9.3f}")

print()
print("=" * 106)
print("B2. WHERE THE INTEGRATOR'S TIME GOES (all players pooled)")
print("=" * 106)
print(f"{'phase/block':<18}{'ticks':>9}{'sec':>9}{'still':>9}"
      f"{'still%':>8}{'mean m/s':>10}{'moving m/s':>12}")
for key, ps in sorted(phase_stats.items(), key=lambda kv: -kv[1]["ticks"]):
    t = max(ps["ticks"], 1)
    moving = t - ps["still"]
    print(f"{key:<18}{ps['ticks']:>9.0f}{ps['ticks']*0.1:>9.0f}"
          f"{ps['still']:>9.0f}{ps['still']/t*100:>7.0f}%"
          f"{ps['speed_sum']/t:>10.2f}"
          f"{ps['moved_sum']/max(moving,1)/0.1:>12.2f}")

print()
print("=" * 106)
print("C. TICKS THE INTEGRATOR NEVER SAW")
print("=" * 106)
print(f"{'player':<19}{'pos':<4}{'on-ball':>10}{'gk':>8}"
      f"{'ob live':>9}{'ob dead':>9}")
for n, s in sorted(skipped.items(), key=lambda kv: -kv[1]["on_ball"]):
    print(f"{n:<19}{pos.get(n,'?'):<4}{s['on_ball']:>10.0f}{s['gk']:>8.0f}"
          f"{s['on_ball_live']:>9.0f}{s['on_ball_dead']:>9.0f}")

print()
print("=" * 106)
print("D. DISTANCE BY WRITER (metres, whole match)")
print("=" * 106)
for src, d in sorted(dist_by_source.items(), key=lambda kv: -kv[1]):
    print(f"  {src:<32}{d:>12.0f} m")

print()
print("=" * 106)
print("E. THE BAND-EDGE MISMATCH THE MODULE CLAIMS TO HAVE FIXED")
print("=" * 106)
print("  position_engine.ACTIVITY_BAND_EDGES =", PE.ACTIVITY_BAND_EDGES)
print("  possession_physics three-band cut  = <2.0 walk, <7.0 jog, else sprint")
print("  -> trace seconds at 0.0 m/s are booked as WALKING, and 6.9 m/s as")
print("     JOGGING, so the trace half of the measurement does not use the")
print("     bands the published table claims.")
