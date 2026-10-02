"""WATCH A MATCH: real roster, real brains, then read the movement honestly.

Uses the REAL 26/27 clubs from the Excel workbook (the two "Hartwell City /
Thornfield United" sides are test teams hand-entered in run_match.py; the real
world is the 18 clubs in the workbook - see AGENTS.md).

REVISION 2 (2026-09-30). The first version of this figure LIED, and it lied
because of how matplotlib draws a dense series: ~55 000 GPS samples per player
compressed into ~400 px means each pixel column spans ~135 samples, and
matplotlib plots the EXTREME of each column. One genuine 82 m single-tick
on-ball reposition therefore paints a full-pitch line, and ~800 of them
dominate the picture while representing 0.1% of the data. The first version's
"long straight lines crossing the pitch / players never settle" reading was an
artefact of that envelope, not a property of the movement. Measured properly
the motion is p50 = 0.00 m/s (standing), p95 = 1.8-3.4 m/s (walking/jogging).

Fixes applied here:
  * panel 2 drops segments above `JUMP_CUT` m/s, and says how many it dropped;
  * panel 3 resamples to one point per 30 s and draws a MEDIAN, plus a
    10-90 percentile band, so a single jump cannot paint the chart;
  * panel 6 replaces the old run-type heatmap position with the honest speed
    CDF per role, which is where the sprint story actually lives.
"""
import math
import random
from collections import Counter, defaultdict
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from match_engine import (
    MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle, Intensity,
)
from player_dna import SquadBuilder
from roster_loader import get_loader
from run_tracking import RUN_TYPES

OUT = "_diag_watch"
XLSX = "PLOFA-2026-2027.xlsx"
JUMP_CUT = 15.0        # m/s - above this a segment is a reposition, not a run
RESAMPLE_S = 30.0      # one depth sample per this many seconds


def build():
    loader = get_loader(XLSX)
    HOME, AWAY = "Oxton", "Natrican"
    home_raw = loader.build_matchday_squad(HOME)
    away_raw = loader.build_matchday_squad(AWAY)
    home = SquadBuilder.build(
        HOME, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"])
    away = SquadBuilder.build(
        AWAY, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"])
    cfg = MatchConfig(home_team=HOME, away_team=AWAY,
                      match_date=date(2026, 8, 16), matchday=1,
                      venue=f"{HOME} Stadium", stadium_capacity=45000)
    hp = TeamProfile(name=HOME, style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.POSSESSION,
                     intensity=Intensity.MEDIUM)
    ap = TeamProfile(name=AWAY, style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.MIXED,
                     intensity=Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad(HOME, home["starters"], home["substitutes"])
    eng.set_squad(AWAY, away["starters"], away["substitutes"])
    return eng, home_raw, away_raw


def main():
    import pathlib
    pathlib.Path(OUT).mkdir(exist_ok=True)
    eng, hr, ar = build()
    print(f"home formation {hr['formation']}, away {ar['formation']}")
    eng.enable_virtual_gps(0.1)
    random.seed(31)
    res = eng.simulate()
    print(f"\nSCORE {res.score_str}")
    print(f"possession {res.home_possession_pct:.1f}%  xG "
          f"{res.home_xg:.2f} - {res.away_xg:.2f}")

    home = eng.config.home_team
    pos_of = {n: st.position for n, st in eng.position_engine.states.items()}
    anchors = {n: st.home_x for n, st in eng.position_engine.states.items()}

    # ---- per-player series, with the jump segments identified -------------
    ser = defaultdict(list)        # name -> [(t, x, y)]
    spd = defaultdict(list)        # name -> [m/s] frame-to-frame, jumps kept
    for s in eng.gps.samples:
        if s["player"] == "__ball__" or s["team"] != home:
            continue
        n = s["player"]
        ser[n].append((s["t"], s["x"], s["y"]))
        if s.get("speed_mps"):
            spd[n].append(s["speed_mps"])
    ball = [(s["x"], s["y"]) for s in eng.gps.samples
            if s["player"] == "__ball__"]

    n_jump = 0
    n_all = 0
    for n, pts in ser.items():
        for a, b in zip(pts, pts[1:]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            n_all += 1
            d = math.hypot(b[1] - a[1], b[2] - a[2]) / dt
            if d > JUMP_CUT:
                n_jump += 1
    print(f"\nsegments: {n_all} total, {n_jump} above {JUMP_CUT} m/s "
          f"({100*n_jump/max(n_all,1):.2f}%) - the reposition artefacts")

    GROUP = {"ST": "ST", "CF": "ST", "LW": "W", "RW": "W", "LB": "FB",
             "RB": "FB", "CB": "DEF", "CDM": "MID", "CM": "MID",
             "CAM": "CAM", "GK": "GK"}
    COLOR = {"ST": "#d62728", "W": "#ff7f0e", "FB": "#8c564b",
             "DEF": "#2ca02c", "MID": "#1f77b4", "CAM": "#e377c2",
             "GK": "#7f7f7f"}
    RC = {"ST": "#d62728", "CAM": "#e377c2", "CM": "#1f77b4",
          "CDM": "#17becf", "CB": "#2ca02c", "RW": "#ff7f0e",
          "LW": "#f1c40f", "LB": "#8c564b", "RB": "#9467bd"}

    fig = plt.figure(figsize=(21, 15), facecolor="#101010")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.30,
                          wspace=0.16)

    def pitch(ax, title):
        ax.set_facecolor("#0b3d0b")
        ax.set_xlim(0, 105); ax.set_ylim(0, 68)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("white")
        ax.axhline(34, color="white", lw=0.8)
        ax.add_patch(plt.Rectangle((17.5, 0), 30, 16.5, fill=False, ec="w"))
        ax.add_patch(plt.Rectangle((57.5, 0), 30, 16.5, fill=False, ec="w"))
        ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, ec="w", lw=0.8))
        ax.set_title(title, color="white", fontsize=10)

    # 1. ball motion
    ax = fig.add_subplot(gs[:, 0])
    pitch(ax, "1. BALL MOTION - whole match")
    ax.scatter([b[0] for b in ball], [b[1] for b in ball], s=0.8,
               c="yellow", alpha=0.22, linewidths=0)
    ax.set_aspect("equal")

    # 2. tracks, jump segments REMOVED
    ax = fig.add_subplot(gs[0, 0])
    pitch(ax, f"2. OUTFIELD TRACKS, teleport segments removed "
              f"(> {JUMP_CUT:.0f} m/s)")
    for n, pts in ser.items():
        g = GROUP.get(pos_of.get(n, ""), "MID")
        if g == "GK":
            continue
        run = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            dt = b[0] - a[0]
            d = math.hypot(b[1] - a[1], b[2] - a[2])
            if dt <= 0 or d / dt > JUMP_CUT:
                if len(run) > 1:
                    ax.plot([q[1] for q in run], [q[2] for q in run], lw=0.45,
                            color=COLOR.get(g, "w"), alpha=0.5)
                run = [b]
            else:
                run.append(b)
        if len(run) > 1:
            ax.plot([q[1] for q in run], [q[2] for q in run], lw=0.45,
                    color=COLOR.get(g, "w"), alpha=0.5)
    ax.set_aspect("equal")

    # 3. depth over time - MEDIAN per 30 s, plus a 10-90 band.
    #    DIRECTION-NORMALISED, and this is not cosmetic. The first version of
    #    this panel plotted raw x, so at half time - when the teams change ends
    #    - every role's depth mirrors around 52.5. I read that mirror, plus the
    #    legitimate 35-55 m surges a striker or an overlapping full-back really
    #    does make, as "the shape oscillates end to end every few minutes".
    #    Normalised, the median depth range inside a 30 s window is 2.8-5.2 m.
    #    A plot that does not know which way a team is attacking cannot be read.
    HALF = 2700.0
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#0b1a0b")
    ax.axhspan(35, 70, color="white", alpha=0.05)
    picked = {}
    for n in ser:
        p = pos_of.get(n, "")
        if p in RC and p not in picked:
            picked[p] = n
    for p in ("ST", "CAM", "CM", "CDM", "CB", "RW", "LW"):
        n = picked.get(p)
        if not n:
            continue
        pts = ser[n]
        bins = defaultdict(list)
        for t, x, _ in pts:
            bins[int(t / RESAMPLE_S)].append(
                x if t < HALF else 105.0 - x)
        ks = sorted(bins)
        med = [np.median(bins[k]) for k in ks]
        lo = [np.percentile(bins[k], 10) for k in ks]
        hi = [np.percentile(bins[k], 90) for k in ks]
        xs = [k * RESAMPLE_S / 60.0 for k in ks]
        ax.fill_between(xs, lo, hi, color=RC[p], alpha=0.13, lw=0)
        ax.plot(xs, med, lw=0.9, color=RC[p], label=f"{p}")
        if anchors.get(n):
            ax.axhline(anchors[n] if anchors[n] < 52.5 else 105 - anchors[n],
                        color=RC[p], ls="--", lw=0.5, alpha=0.45)
    ax.axvline(45, color="w", ls=":", lw=0.8, alpha=0.6)
    ax.text(45.6, 100, "ends change", color="w", fontsize=6, va="top")
    ax.set_ylim(0, 105); ax.set_xlim(0, 95)
    ax.set_xlabel("match clock (min)", color="w", fontsize=8)
    ax.set_ylabel("depth x (m)   0 = own goal  (ends normalised)", color="w",
                  fontsize=8)
    ax.tick_params(colors="w", labelsize=7)
    ax.grid(alpha=0.13, color="w")
    ax.set_title("3. DEPTH OVER TIME - median per 30 s, band = 10-90 pct,\n"
                 "DIRECTION-NORMALISED for the half-time ends change",
                 color="white", fontsize=9)
    ax.legend(fontsize=7, ncol=7, facecolor="black", labelcolor="w",
              loc="upper center")

    # 4. density
    ax = fig.add_subplot(gs[2, 0])
    pitch(ax, "4. HOME XI POSITION DENSITY")
    xs = [q[1] for p in ser.values() for q in p]
    ys = [q[2] for p in ser.values() for q in p]
    ax.hist2d(xs, ys, bins=[28, 18], range=[[0, 105], [0, 68]], cmap="magma")
    ax.set_aspect("equal")

    # 5. speed CDF per role - the honest view of the sprint question
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#101010")
    grid = np.linspace(0, 8, 200)
    order = ["ST", "LW", "RW", "LB", "RB", "CAM", "CM", "CDM", "CB"]
    for p in order:
        v = [s for n in ser if pos_of.get(n) == p for s in spd.get(n, [])]
        if not v:
            continue
        v = np.sort(np.array(v))
        cdf = np.searchsorted(v, grid) / len(v)
        ax.plot(grid, cdf, lw=1.3, color=RC[p], label=p)
    ax.axvline(7.0, color="w", ls="--", lw=0.9)
    ax.text(7.05, 0.5, " sprint\n threshold", color="w", fontsize=7)
    ax.set_xlabel("speed (m/s)", color="w", fontsize=8)
    ax.set_ylabel("fraction of samples at or below", color="w", fontsize=8)
    ax.tick_params(colors="w", labelsize=7)
    ax.grid(alpha=0.13, color="w")
    ax.set_title("5. SPEED CDF BY ROLE - engine-reported speed\n"
                 "a curve that reaches the dashed line CAN sprint",
                 color="white", fontsize=9)
    ax.legend(fontsize=7, ncol=3, facecolor="black", labelcolor="w")

    # 6. run types
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#101010")
    prof = res.run_profile or {}
    names = [n for n in ser if n in prof]
    names.sort(key=lambda n: -sum(prof[n].values()))
    M = np.array([[prof[n].get(k, 0) for k in RUN_TYPES] for n in names])
    ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(RUN_TYPES)))
    ax.set_xticklabels(RUN_TYPES, rotation=40, color="w", fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} {pos_of.get(n,'')}" for n in names], color="w",
                       fontsize=6)
    ax.set_title("6. OBSERVED RUN TYPES (geometric, jump-filtered)",
                 color="white", fontsize=9)
    for i in range(len(names)):
        for j in range(len(RUN_TYPES)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="w", fontsize=5)

    # 7. speed bands, the numbers
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor("#101010")
    ax.axis("off")
    # NOTE: the last band must start at 7.0, not 6.0. SPRINT_THRESHOLD is 7.0
    # and a band labelled "SPRINT" that starts at 6.0 reports ~9% for a
    # full-back who is recorded with ZERO sprints - the label and the counter
    # then disagree and the figure quietly becomes a lie. I made exactly that
    # error on the first pass of this script.
    bands = [(0, .5), (.5, 1.5), (1.5, 2.8), (2.8, 4.5), (4.5, 6.0),
             (6.0, 7.0), (7.0, 99)]
    rows = ["| role | standing | walk | jog | run | fast | 6-7 | SPRINT 7+ |",
            "|---|---|---|---|---|---|---|---|"]
    for p in ("ST", "CAM", "CM", "CDM", "CB", "LW", "RW", "LB", "RB"):
        v = np.array([s for n in ser if pos_of.get(n) == p
                      for s in spd.get(n, [])])
        if not len(v):
            continue
        c = [100 * ((v >= lo) & (v < hi)).mean() for lo, hi in bands]
        rows.append(f"| {p} | " + " | ".join(f"{x:.1f}%" for x in c) + " |")
    tot = Counter()
    for n in prof:
        for k, v_ in prof[n].items():
            tot[k] += v_
    ip = getattr(res, "intended_run_profile", {}) or {}
    it = Counter()
    for n in ip:
        for k, v_ in ip[n].items():
            it[k] += v_
    txt = ("7. SPEED BANDS (% of samples)\n" + "\n".join(rows)
           + "\n\nobserved run types: " + ", ".join(
               f"{k} {v_}" for k, v_ in tot.most_common())
           + "\nintended runs:      " + ", ".join(
               f"{k} {v_}" for k, v_ in it.most_common()))
    ax.text(0, 1, txt, va="top", color="w", fontsize=8, family="monospace")

    fig.suptitle(f"PLOFA - {eng.config.home_team} v {eng.config.away_team}, "
                 f"one full match (real 26/27 roster)", color="white",
                 fontsize=14)
    fig.savefig(f"{OUT}/watch.png", dpi=100, facecolor="#101010")
    print(f"wrote {OUT}/watch.png")

    print("\n--- per-player movement ---")
    print(f"{'player':<18}{'pos':<5}{'dist_km':>9}{'sprints':>8}{'runs':>6}")
    gp = eng.gps.players
    for n in sorted(ser, key=lambda k: -gp.get(k, {}).get("distance_m", 0)):
        g = gp.get(n, {})
        print(f"{n:<18}{pos_of.get(n,''):<5}"
              f"{g.get('distance_m',0)/1000:>9.2f}"
              f"{g.get('sprint_count',0):>8d}"
              f"{sum(prof.get(n,{}).values()):>6d}")


if __name__ == "__main__":
    main()
