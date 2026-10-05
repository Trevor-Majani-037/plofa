"""Do the wingers and full-backs actually HOLD THE TOUCHLINE? (2026-10-04)

The user asked, watching a match: "do wingers/fullbacks stick to the touchline
to spread play like modern wingers? cause i see sometimes drifts inside,
sometimes its well, maybe they follow managers instructions i dont know."

Three questions hide in that, and each needs a different instrument:

  1. IS THERE A MODEL?   -> read the source. There is, and it is substantial:
                           CHECKPOINT 18 / 21c / 35 in position_engine.py,
                           plus a live "cut inside" mode in
                           winger_behavior.should_cut_inside.
  2. IS THERE A MANAGER CHANNEL? -> the user's own guess. There is, and it is
                           the twist: TWO manager-side width channels reach
                           the wide players' anchors.
  3. DOES IT REACH THE PITCH? -> needs measurement, because "a shape function
                           that looks right" and "a winger who actually stays
                           wide" are different claims, and this project has
                           four instances of a mechanism existing and never
                           running.

INSTRUMENT. `MatchEngine._offball_run` is wrapped and every wide player is
sampled on every call, reading `current_y` AND `home_y` AT THE SAME INSTANT.
That last detail is deliberate and is the seventh instance of the audit's
standing trap: `home_y` is MUTATED during a match (`_recompute_homes`, called
by `apply_coach_width` / `apply_attack_pattern` / stance changes), so reading
`home_y` after `simulate()` returns attributes final-whistle state to a
mid-match moment -- and it would have silently mislabelled every coach-width
excursion.

MEASURED, not asserted:
  drift_m    abs(current_y - home_y), against the FORMATION anchor. A
             name-based reference would be wrong for a team attacking left,
             whose "LW" stands on the right of the pitch.
  in_channel drift <= FLANK_CHANNEL_HALF_WIDTH_M (10 m). That is the engine's
             OWN definition (winger_behavior.in_flank_channel), so the verdict
             is the model's, not mine.
  RESIDENT vs TRANSIENT is reported as run lengths, because a median cannot
             distinguish a player who alternates every few seconds from one
             who parks in the half-space for ten minutes. Those are opposite
             findings and they need opposite fixes.

Run:  python _diag_wide_channel.py [--seed N]
"""
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from winger_behavior import FLANK_CHANNEL_HALF_WIDTH_M  # noqa: E402

WIDE = ("LW", "RW", "LB", "RB")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
A = ap.parse_args()

import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(A.seed)
np.random.seed(A.seed)

import match_engine as ME  # noqa: E402
from _diag_chance_coords import build_pair  # noqa: E402

SAMPLES = defaultdict(list)          # name -> [(drift, cy, hy, width_cmd)]
FRAMES = []                          # per call: {name: cy} for BOTH teams
PHASE = []                           # per call: (home_has_ball, block_home, block_away)
CLOCKS = []                          # match_clock_s at each sample
TICKS = [0]

_raw = ME.MatchEngine._offball_run


def _patched(self, duration_s, home_has_ball):
    pe = self.position_engine
    if pe is not None:
        cmds = pe._applied_coach_width
        frame = {}
        for team, names in pe.team_rosters.items():
            cmd = cmds.get(team, 0.0)
            for nm in names:
                st = pe.states.get(nm)
                if st is None or st.position not in WIDE:
                    continue
                if st.current_y is None or st.home_y is None:
                    continue
                SAMPLES[nm].append((abs(st.current_y - st.home_y),
                                    st.current_y, st.home_y, cmd))
                frame[nm] = st.current_y
        FRAMES.append(frame)
        # `_in_block` in the live loop is `self._defensive_block.get(team) is
        # not None` (match_engine.py:2789-2790). Read the same dict rather than
        # re-deriving it, so "in a block" means what the engine means.
        blk = getattr(self, "_defensive_block", {}) or {}
        PHASE.append((bool(home_has_ball),
                      blk.get(self.config.home_team) is not None,
                      blk.get(self.config.away_team) is not None))
        CLOCKS.append(float(getattr(self.state, "match_clock_s", 0.0)))
    TICKS[0] += 1
    return _raw(self, duration_s, home_has_ball)


ME.MatchEngine._offball_run = _patched

pair = build_pair("Oxton", "Natrican")
res = pair.simulate()
pe = pair.position_engine

print(f"seed={A.seed}  events={len(res.timeline)}  "
      f"frames={len(res.position_log)}  ticks_sampled={TICKS[0]}")
print(f"FLANK_CHANNEL_HALF_WIDTH_M = {FLANK_CHANNEL_HALF_WIDTH_M}")
print(f"USE_MANAGER_BRAIN = {getattr(ME, 'USE_MANAGER_BRAIN', None)}")
print()

print("=" * 104)
print("THE TWO MANAGER-SIDE WIDTH CHANNELS THAT REACH THE ANCHOR")
print("-" * 104)
for side, prof in (("home", pair.home_profile), ("away", pair.away_profile)):
    print(f"{side:<6} tactical philosophy={getattr(prof,'philosophy','')!r:<22} "
          f"style={getattr(prof,'style','')!r:<22} "
          f"width={getattr(prof,'width',None)}")
print("  channel 1  FormationEngine.compute_home(pos, profile, slot) reads "
          "profile.width:")
print("             +3 m outward at width=1.0, -3 m inward at width=0.0 "
              "(position_engine.py:362-371)")
print("  channel 2  coach_instructions -> PositionEngine.apply_coach_width:")
print("             posture ATTACK -> width_cmd +1 -> -2.5 m on wide roles")
print("             posture DEFEND -> width_cmd -1 -> +1.5 m inward "
              "(position_engine.py:753-759)")
print("             i.e. the coach can move a winger 2.5 m, and the whole")
print("             channel is inert unless USE_MANAGER_BRAIN is on and the")
print("             manager has decided a live ATTACK/DEFEND posture.")
print()

# coach command actually applied, over the match
cmd_hist = defaultdict(int)
for nm, rows in SAMPLES.items():
    for _, _, _, c in rows:
        cmd_hist[c] += 1
print("coach width_cmd distribution across ALL wide-player samples:",
      dict(sorted(cmd_hist.items())), " (+1 stay wide / -1 tuck in / 0 none)")
print()


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def med(vals):
    return pct(vals, 50)


def mean(vals):
    return (sum(vals) / len(vals)) if vals else 0.0


# MEASURE the sample cadence, do not assume it. The first version of this
# probe hard-coded TICK_S = 0.1 on the belief that `_offball_run` is a 10 Hz
# tick. It is called ~409 times over a 90-minute match, i.e. roughly every
# 13 s -- so every duration it printed was wrong by two orders of magnitude.
# That is the `top_speed_mpm` trap (AGENTS.md, MEASUREMENTS) again: read the
# units off the instrument, never off its name.
_deltas = [CLOCKS[i + 1] - CLOCKS[i] for i in range(len(CLOCKS) - 1)]
_deltas = [d for d in _deltas if d > 0]
SAMPLE_S = med(_deltas) if _deltas else float("nan")
print(f"sampled {len(CLOCKS)} times; median gap between samples = "
      f"{SAMPLE_S:.1f} s  (min {min(_deltas) if _deltas else 0:.1f} / "
      f"max {max(_deltas) if _deltas else 0:.1f})")
print()


def runs(rows):
    """(median OFF-flank excursion, median on-flank spell), in SAMPLES then
    seconds. Returned in samples first so the raw count is always visible and
    the seconds figure can be re-derived if the cadence assumption changes."""
    ins, outs = [], []
    prev, n = None, 0
    for d, _, _, _ in rows:
        cur = d <= FLANK_CHANNEL_HALF_WIDTH_M
        if prev is None:
            prev, n = cur, 1
            continue
        if cur == prev:
            n += 1
        else:
            (ins if prev is False else outs).append(n)
            prev, n = cur, 1
    if prev is not None:
        (ins if prev is False else outs).append(n)
    return med(ins), med(outs)


print("=" * 112)
print(f"{'player':<20} {'pos':<4} {'n':>6} {'med':>6} {'p90':>6} {'p99':>6} "
      f"{'in_ch%':>7} {'max':>6} {'nOut':>5} {'OUT samp':>9} {'OUT s':>7} "
      f"{'stayWide':>9} {'tuck':>7}")
print("-" * 112)

pos_of, team_of, PLAYER = {}, {}, {}
for team in (res.config.home_team, res.config.away_team):
    for nm in pe.team_rosters.get(team, []):
        st = pe.states.get(nm)
        if st is not None:
            pos_of[nm] = st.position
            team_of[nm] = team
    for p in (res.squads.get(team, {}).get("starters") or []):
        PLAYER[getattr(p, "name", "")] = p

summary = {}
for nm in sorted(SAMPLES, key=lambda n: (team_of.get(n, ""), pos_of.get(n, ""), n)):
    rows = SAMPLES[nm]
    drifts = [d for d, _, _, _ in rows]
    in_n, out_n = runs(rows)
    w = [d for d, _, _, c in rows if c > 0]
    t = [d for d, _, _, c in rows if c < 0]
    n_out = 0
    prev = None
    for d, _, _, _ in rows:
        cur = d <= FLANK_CHANNEL_HALF_WIDTH_M
        if prev is not None and prev is True and cur is False:
            n_out += 1
        prev = cur
    summary[nm] = (med(drifts), 100.0 * sum(1 for d in drifts
                                           if d <= FLANK_CHANNEL_HALF_WIDTH_M)
                   / len(drifts), out_n * SAMPLE_S)
    print(f"{nm:<20} {pos_of.get(nm,'?'):<4} {len(rows):>6} {med(drifts):>6.1f} "
          f"{pct(drifts,90):>6.1f} {pct(drifts,99):>6.1f} "
          f"{100.0*sum(1 for d in drifts if d <= FLANK_CHANNEL_HALF_WIDTH_M)/len(drifts):>7.1f} "
          f"{max(drifts):>6.1f} {n_out:>5} {out_n:>9.0f} "
          f"{out_n*SAMPLE_S:>7.0f} "
          f"{med(w) if w else float('nan'):>9.1f} "
          f"{med(t) if t else float('nan'):>7.1f}")

print("-" * 112)
print(f"nOut       = how many separate excursions OFF the flank channel he "
      f"made all match")
print(f"OUT samp/s = length of a typical ON-flank spell, in samples and in "
      f"seconds ({SAMPLE_S:.0f} s/sample)")
print(f"stayWide/tuck = median drift while the coach command is +1 vs -1. The "
      f"whole documented")
print(f"             size of that command is 2.5 m, so the two columns should "
      f"differ by about that")
print(f"             and no more. A bigger gap means something ELSE is moving "
      f"him.")
print()

print("=" * 104)
print("PER-PLAYER STYLE — read from the registries the shape layer itself uses")
print("-" * 104)
print(f"{'player':<20} {'team':<12} {'archetype':<22} {'flankCommit':>11} "
      f"{'byline':>7} {'cutBase':>8} {'homeY':>6}")
print("-" * 104)
for nm in sorted(SAMPLES, key=lambda n: (team_of.get(n, ""), pos_of.get(n, ""), n)):
    p = pe.winger_registry.get(nm) or pe.fullback_registry.get(nm)
    st = pe.states.get(nm)
    if p is None or st is None:
        print(f"{nm:<20} {team_of.get(nm,'?'):<12} {'(no profile)':<22}")
        continue
    # NOTE: WingerSpatialProfile / FullbackSpatialProfile do NOT store the DNA
    # they were built from -- the first version of this table read
    # `p.dna.archetype`, got None, and printed "?" for all eight players,
    # which reads as "no archetype" rather than "wrong attribute".
    arch = getattr(getattr(PLAYER.get(nm), "dna"), "archetype", "") or "?"
    by = getattr(p, "byline_instinct", float("nan"))
    cut = (1.0 - by) * 0.60 if by == by else float("nan")
    print(f"{nm:<20} {team_of.get(nm,'?'):<12} {arch:<22} "
          f"{p.flank_commitment:>11.2f} {by:>7.2f} {cut:>8.2f} {st.home_y:>6.1f}")
print("-" * 104)
print("cutBase = should_cut_inside()'s baseline cut probability BEFORE the "
      "+0.25 half-space")
print("           bonus, i.e. (1 - byline_instinct) * 0.60. An inverted winger")
print("           has a high cutBase BY DESIGN; that is the drift the user is "
      "seeing.")
print()

print("=" * 104)
print("TEAM WIDTH — the 'spread play' claim, measured")
print("-" * 104)
# FRAMES is one dict per _offball_run CALL, so both flanks are read at the
# same instant by construction. The first version reconstructed the pairing
# from per-player list INDICES, which silently misaligns whenever one player
# is skipped (a None coordinate shifts everyone after him in roster order) --
# and it reported "never both flanks occupied", which is a probe artefact, not
# a finding.
wid_w, wid_fb = defaultdict(list), defaultdict(list)
for frame in FRAMES:
    for team in (res.config.home_team, res.config.away_team):
        w = {pos_of.get(nm): y for nm, y in frame.items()
             if team_of.get(nm) == team}
        if "LW" in w and "RW" in w:
            wid_w[team].append(abs(w["LW"] - w["RW"]))
        if "LB" in w and "RB" in w:
            wid_fb[team].append(abs(w["LB"] - w["RB"]))
for label, bucket in (("winger-to-winger (LW-RW)", wid_w),
                      ("fullback-to-fullback (LB-RB)", wid_fb)):
    for team, vals in sorted(bucket.items()):
        if vals:
            print(f"{label:<30} {team:<12} n={len(vals):>4}  "
                  f"mean {mean(vals):>5.1f} m  median {med(vals):>5.1f} m"
                  f"  p10 {pct(vals,10):>5.1f}  p90 {pct(vals,90):>5.1f}")
    if not bucket:
        print(f"{label:<30} (never both flanks occupied in the same call)")
print()
print("reference: the pitch is 68 m wide. Real Premier League winger")
print("separation in possession runs ~45-55 m. Touchlines are y=0 and y=68.")
print()

# ── THE PHRASE SPLIT, which is the question the user actually asked ────────
# The single 38 m mean above mixes three different football situations, and
# only ONE of them is supposed to be narrow. `_offball_run` hands the wrapper
# `home_has_ball` directly and the first version of this probe threw it away,
# so the headline number could not be compared against a reference band that
# is itself a POSSESSION figure. Split by possession and by block state, and
# report distance-to-nearest-TOUCHLINE as well as drift-from-anchor: the first
# is "is he actually near the line", the second mixes in where the line is.
print("=" * 108)
print("WIDTH BY PHASE — possession matters, and the reference band is a "
      "possession figure")
print("-" * 108)
PH = {  # name -> (winger_sep, fb_sep) tuples of (home_sep, away_sep)
    "IN POSSESSION": ([], [], [], []),
    "OUT, no block": ([], [], [], []),
    "OUT, in block": ([], [], [], []),
}
LINED = defaultdict(list)   # (phase, role) -> distance to nearest touchline
for frame, (hb, bh, ba) in zip(FRAMES, PHASE):
    for team in (res.config.home_team, res.config.away_team):
        inblock = bh if team == res.config.home_team else ba
        key = ("IN POSSESSION" if (hb == (team == res.config.home_team))
               else ("OUT, in block" if inblock else "OUT, no block"))
        w = {pos_of.get(nm): y for nm, y in frame.items()
             if team_of.get(nm) == team}
        slot = PH[key]
        if "LW" in w and "RW" in w:
            slot[0 if team == res.config.home_team else 1].append(
                abs(w["LW"] - w["RW"]))
        if "LB" in w and "RB" in w:
            slot[2 if team == res.config.home_team else 3].append(
                abs(w["LB"] - w["RB"]))
        for nm, y in frame.items():
            if team_of.get(nm) != team:
                continue
            LINED[(key, pos_of.get(nm, "?"))].append(min(y, 68.0 - y))

print(f"{'phase':<18} {'grp':<6} {'n':>5} {'mean':>7} {'median':>7} "
      f"{'p10':>7} {'p90':>7} {'p05':>7} {'in 45-55':>9}")
print("-" * 108)
for key in ("IN POSSESSION", "OUT, no block", "OUT, in block"):
    hw, aw, hf, af = PH[key]
    for lbl, vals in ((f"{key} wingers", hw + aw), (f"{key} fullbacks", hf + af)):
        if not vals:
            continue
        band = 100.0 * sum(1 for v in vals if 45.0 <= v <= 55.0) / len(vals)
        print(f"{lbl:<18} {'-':<6} {len(vals):>5} {mean(vals):>7.1f} "
              f"{med(vals):>7.1f} {pct(vals,10):>7.1f} {pct(vals,90):>7.1f} "
              f"{pct(vals,5):>7.1f} {band:>8.1f}%")
print("-" * 108)
print()
print("DISTANCE TO NEAREST TOUCHLINE (m). Real wide players sit 3-12 m off the")
print("line; 'collapsing' means this climbing into the 20s.")
print("-" * 108)
print(f"{'phase':<18} {'LW':>16} {'RW':>16} {'LB':>16} {'RB':>16}")
for key in ("IN POSSESSION", "OUT, no block", "OUT, in block"):
    cells = []
    for posn in ("LW", "RW", "LB", "RB"):
        v = LINED.get((key, posn)) or []
        cells.append(f"{med(v):>6.1f} (p90 {pct(v,90):>4.1f})" if v
                     else f"{'-':>16}")
    print(f"{key:<18} " + " ".join(cells))
print("-" * 108)