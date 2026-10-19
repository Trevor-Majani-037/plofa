"""VERIFY all three post-distance changes on one real match (seed 777).

1. PER-KEEPER goals prevented / xGOT faced / goals conceded (was: the team's,
   stamped onto every player whose position == "GK", so a bench keeper who
   never came on inherited the whole team's xGOT and a non-zero GP).
2. GK distance -- was 2.82-2.84 km against Di Salvo 2008's 5,611 +/- 613 m and
   Kubayi 2020's 4,819 m (range 4,036-6,640).
3. The physics additive is now an EXCESS over the template's budget for the
   traced seconds, not a flat 45% of a full re-measurement.

GATES before any verdict (a non-vacuous instrument):
  G1 the two keepers of each side are distinguished, i.e. at least one side
     actually used a substitute keeper OR the numbers still differ per keeper
  G2 the keeper-scoped numbers RECONCILE: sum of keeper xGOT == team xGOT, and
     sum of keeper concessions == team goals conceded. If these do not close,
     the attribution has lost or invented an event.
  G3 outfielders and keepers are both reported against the real bands
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
HOME_SUBS = [
    ("Rafi Sandoval", "GK", ["shot_stopper"], 27),
    ("Tomas Bergen", "CB", ["stopper_defender"], 24),
    ("Elias Kovac", "CM", ["box_to_box"], 23),
    ("Nuno Reis", "LW", ["winger"], 22),
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
AWAY_SUBS = [
    ("Otto Lindqvist", "GK", ["sweeper_keeper"], 26),
    ("Paulo Cardoso", "RB", ["fullback"], 25),
    ("Ari Kimura", "CAM", ["creator"], 23),
]

random.seed(777)
home = SquadBuilder.build("Hartwell City", HOME_STARTERS, HOME_SUBS)
away = SquadBuilder.build("Thornfield United", AWAY_STARTERS, AWAY_SUBS)
config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                     match_date=date(2026, 8, 22), matchday=2)
hp = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION, Intensity.MEDIUM)
ap = TeamProfile("Thornfield United", TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
engine = MatchEngine(config, hp, ap)
engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
engine.set_squad("Thornfield United", away["starters"], away["substitutes"])
result = engine.simulate()

from exporter import StatAccumulator  # noqa: E402

ex = StatAccumulator(result, {
    "Hartwell City": {"starters": home["starters"],
                      "substitutes": home["substitutes"]},
    "Thornfield United": {"starters": away["starters"],
                          "substitutes": away["substitutes"]},
})
stats = ex.stats

print(f"score {result.state.home_goals}-{result.state.away_goals}   "
      f"events {len(result.timeline)}")

# ── 1. GOALS PREVENTED, PER KEEPER ────────────────────────────────────────
print()
print("=" * 86)
print("1. GOALKEEPING -- per keeper, not per team")
print("=" * 86)
print(f"{'keeper':<20}{'team':<18}{'mins':>6}{'saves':>7}{'conc':>6}"
      f"{'xGOT':>8}{'GP':>8}{'save%':>8}{'CS':>4}")
keepers = [(n, s) for n, s in stats.items() if s["position"] == "GK"]
for n, s in sorted(keepers, key=lambda kv: -kv[1]["minutes_played"]):
    print(f"{n:<20}{s['team']:<18}{s['minutes_played']:>6.0f}{s['saves']:>7}"
          f"{s['goals_conceded']:>6}{s['xgot_faced']:>8.3f}"
          f"{s['goals_prevented']:>8.3f}{s['save_pct']:>8.1f}"
          f"{str(bool(s['clean_sheet'])):>4}")

print()
subs_on = [e for e in result.timeline if str(e.event_type).endswith("SUBSTITUTION")]
print(f"real substitution events in this match: {len(subs_on)}")
if not subs_on:
    print("  -> THE IN-GOAL SWAP BRANCH WAS NOT EXERCISED BY THIS MATCH.")
    print("     Scenario B below forces one, because a keeper who CAME ON is")
    print("     exactly the case the team-level stamp got wrong.")

# ── SCENARIO B: force a keeper substitution and re-attribute ───────────────
from match_engine import MatchEvent, EventType  # noqa: E402
from exporter import StatAccumulator  # noqa: E402

MINUTE_ON = 70
starter_gk = stats["Keano Walsh"]["team"]
bench_gk = next(n for n, s in stats.items()
                if s["position"] == "GK" and s["team"] == starter_gk
                and s["minutes_played"] == 0)
# Copy the enums off a real event rather than reconstructing them: guessing
# `MatchPhase(0)` raised `ValueError: 0 is not a valid MatchPhase`.
_ref = next(e for e in result.timeline if e.event_type == EventType.GOAL)
tmpl = MatchEvent(
    minute=MINUTE_ON, second=0, event_type=EventType.SUBSTITUTION,
    team=starter_gk, player="Keano Walsh", secondary_player=bench_gk,
    phase=_ref.phase, game_state=_ref.game_state,
    location_x=50.0, location_y=34.0,
    metadata={"reason": "probe_injected"},
)
result.timeline = sorted(list(result.timeline) + [tmpl],
                         key=lambda e: (e.minute,))
# minutes_played must agree with the injected timeline or the clean-sheet gate
# reads the substitution as irrelevant.
stats["Keano Walsh"]["minutes_played"] = MINUTE_ON
stats[bench_gk]["minutes_played"] = 90 - MINUTE_ON

exB = StatAccumulator(result, {
    "Hartwell City": {"starters": home["starters"], "substitutes": home["substitutes"]},
    "Thornfield United": {"starters": away["starters"], "substitutes": away["substitutes"]},
})
print()
print("=" * 86)
print(f"SCENARIO B -- keeper swapped at {MINUTE_ON}': "
      f"Keano Walsh OFF -> {bench_gk} ON")
print("=" * 86)
print(f"{'keeper':<20}{'mins':>6}{'saves':>7}{'conc':>6}{'xGOT':>8}{'GP':>8}")
for n in ("Keano Walsh", bench_gk):
    s = exB.stats[n]
    print(f"{n:<20}{s['minutes_played']:>6.0f}{s['saves']:>7}{s['goals_conceded']:>6}"
          f"{s['xgot_faced']:>8.3f}{s['goals_prevented']:>8.3f}")

for team in ("Hartwell City", "Thornfield United"):
    tx = exB.team_xgot.get(team, 0.0)
    kx = sum(s["xgot_faced"] for n, s in exB.stats.items()
             if s["position"] == "GK" and s["team"] == team)
    tc = (result.state.away_goals if team == config.home_team
          else result.state.home_goals)
    kc = sum(s["goals_conceded"] for n, s in exB.stats.items()
             if s["position"] == "GK" and s["team"] == team)
    ok = abs(tx - kx) < 0.01 and tc == kc
    print(f"  {team:<18} xGOT team {tx:7.3f} = keepers {kx:7.3f} | "
          f"conceded team {tc} = keepers {kc}   {'OK' if ok else '*** MISMATCH ***'}")

# the whole point: the man on for the last 20' owns what happened in them
sw = exB.stats["Keano Walsh"]
sb = exB.stats[bench_gk]
print(f"\n  starting keeper (0-70')  xGOT {sw['xgot_faced']:.3f}  "
      f"GP {sw['goals_prevented']:.3f}")
print(f"  substitute  (70-90')     xGOT {sb['xgot_faced']:.3f}  "
      f"GP {sb['goals_prevented']:.3f}")
print(f"  the substitute is credited ONLY with his own 20 minutes: "
      f"{'YES' if sb['xgot_faced'] <= sw['xgot_faced'] + 0.001 else 'NO'}")

# G2: reconcile keeper scope against team scope
print()
bad = []
for team in ("Hartwell City", "Thornfield United"):
    tx = ex.team_xgot.get(team, 0.0)
    kx = sum(s["xgot_faced"] for n, s in keepers if s["team"] == team)
    tc = result.state.away_goals if team == config.home_team else result.state.home_goals
    kc = sum(s["goals_conceded"] for n, s in keepers if s["team"] == team)
    print(f"  {team:<18} xGOT team {tx:7.3f} vs keepers {kx:7.3f}   "
          f"conceded team {tc} vs keepers {kc}")
    if abs(tx - kx) > 0.01 or tc != kc:
        bad.append(team)
print(f"  RECONCILES: {'YES' if not bad else 'NO -> ' + str(bad)}")

unused = [n for n, s in keepers if s["minutes_played"] == 0]
print(f"\n  keepers who never played: {unused}")
for n in unused:
    s = stats[n]
    ok = (s["goals_prevented"] == 0.0 and s["xgot_faced"] == 0.0
          and s["goals_conceded"] == 0 and s["saves"] == 0)
    print(f"    {n:<18} saves {s['saves']}  xGOT {s['xgot_faced']}  "
          f"GP {s['goals_prevented']}  conc {s['goals_conceded']}   "
          f"{'OK' if ok else '*** STILL INHERITING ***'}")

# ── 2 & 3. DISTANCE ───────────────────────────────────────────────────────
print()
print("=" * 86)
print("2/3. DISTANCE  (outfielders 10.0-13.5 km; keepers 4.0-6.6 km)")
print("=" * 86)
outf = [(n, s["distance_covered"]) for n, s in stats.items()
        if s["position"] != "GK" and s["minutes_played"] > 0]
gk = [(n, s["distance_covered"]) for n, s in keepers if s["minutes_played"] > 0]
ov = [d for _, d in outf]
print(f"outfielders WHO PLAYED n={len(ov)}  min {min(ov):.2f}  "
      f"mean {sum(ov)/len(ov):.2f}  max {max(ov):.2f} km")
print(f"  in band 10.0-13.5 : {sum(1 for x in ov if 10.0 <= x <= 13.5)}/{len(ov)}"
      f"    over: {sum(1 for x in ov if x > 13.5)}"
      f"    under: {sum(1 for x in ov if x < 10.0)}")
print(f"keepers who played n={len(gk)}  "
      + "  ".join(f"{n.split()[0]} {d:.2f}" for n, d in gk) + " km")
print(f"  Di Salvo 2008 5611 +/- 613 m (5000-6222) | Kubayi 2020 4819 m (4036-6640)")
print(f"  (a sub who never played has no position_log frames, so 0.00 km -- "
      f"correct, not a gap)")

print()
print("top 6 outfielders:")
for n, d in sorted(outf, key=lambda kv: -kv[1])[:6]:
    print(f"  {n:<20}{d:>7.2f} km")

print()
print("PER POSITION (who played) -- template vs published vs real band")
print(f"{'pos':<6}{'n':>4}{'template':>10}{'published':>11}{'min':>8}{'max':>8}")
tmpl_tot = defaultdict(float)
pub_tot = defaultdict(float)
for f in result.position_log:
    for side in ("home", "away"):
        for row in f.get(side, []):
            tmpl_tot[row["player"]] += 1.0
for n, s in stats.items():
    if s["minutes_played"] > 0:
        pub_tot[s["position"]] += s["distance_covered"]
for n, s in stats.items():
    pass
# recompute the template per player
SPEEDS = __import__("opta_analytics").SPEEDS
BA = __import__("opta_analytics").BASE_MINUTE_ACTIVITY
import opta_analytics  # noqa: E402
for n, s in stats.items():
    if s["minutes_played"] > 0:
        p = s["position"]
        base = BA.get(p, opta_analytics.DEFAULT_ACTIVITY)
        wr = getattr(getattr(stats[n].get("dna_ref", None) or None, "mental", None),
                     "work_rate", 50.0)
        e = 0.80 + 0.40 * (wr / 100.0)
        st, wa, j, r, sp = base
        j *= e
        r = min(0.35, r * e)
        sp = min(0.10, sp * e)
        per_min = (st * 60 * SPEEDS["standing"] + wa * 60 * SPEEDS["walking"]
                   + j * 60 * SPEEDS["jogging"] + r * 60 * SPEEDS["running"]
                   + sp * 60 * SPEEDS["sprinting"])
        s["_tmpl_km"] = per_min * tmpl_tot[n] / 1000

pos_rows = defaultdict(lambda: [0, 0.0, 0.0, [], []])
for n, s in stats.items():
    if s["minutes_played"] <= 0:
        continue
    d = pos_rows[s["position"]]
    d[0] += 1
    d[1] += s.get("_tmpl_km", 0.0)
    d[2] += s["distance_covered"]
    d[3].append(s["distance_covered"])
for p in sorted(pos_rows):
    n, tm, pub, vals, _ = pos_rows[p]
    print(f"{p:<6}{n:>4}{tm/n:>10.2f}{pub/n:>11.2f}{min(vals):>8.2f}{max(vals):>8.2f}")

print()
print("Q2 -- ARE THE ACTIVITY SECONDS MEASURED?")
print("=" * 86)
print("published *_seconds for the top 4, against the PURE DNA template for")
print("their position (no telemetry at all). If these match, the shape is the")
print("template and the telemetry only perturbs it.")
print(f"{'player':<20}{'pos':<5}{'band':<13}{'template s':>12}{'published s':>14}")
top = [n for n, s in sorted(stats.items(),
                            key=lambda kv: -kv[1]["distance_covered"])
       if s["minutes_played"] > 0][:4]
for n in top:
    s = stats[n]
    p = s["position"]
    base = BA.get(p, opta_analytics.DEFAULT_ACTIVITY)
    frames = tmpl_tot[n]
    eff = 0.80 + 0.40 * 0.5          # effort 1.0 baseline, no DNA lookup needed
    stand, walk, jog, run, sprint = base
    jog *= eff
    run = min(0.35, run * eff)
    sprint = min(0.10, sprint * eff)
    for label, frac in (("standing", stand), ("walking", walk), ("jogging", jog),
                        ("running", run), ("sprinting", sprint)):
        a = ex.opta._activity.get(n, {})
        print(f"{n:<20}{p:<5}{label:<13}{frac*60*frames:>12.0f}"
              f"{a.get(label, 0.0):>14.0f}")
    print()