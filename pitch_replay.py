"""pitch_replay.py — watch a PLOFA match on a 2D pitch, at any clock speed.

The engine simulates a full 90-minute match in ~20-30 seconds of wall time,
but it records everything against its true match clock:

  - per-minute, per-player position snapshots   (MatchResult.position_log)
  - a full per-0.1 s ball trajectory            (MatchResult.full_match_ball_path)

This tool bridges those two resolutions so a match can be WATCHED on a pitch
instead of read off a spreadsheet:

  - the ball uses the recorded 10 Hz trajectory directly;
  - every player is interpolated (smoothstep-eased) between their consecutive
    per-minute snapshots, so movement is continuous rather than steppy;
  - playback speed is a pure multiplier on the recorded clock: 1x is a real
    ~90-minute match, 64x fits the whole game into ~90 seconds of your time.

Keyboard controls (live window):
  space       pause / play
  + / -       faster / slower (multiplier)
  left/right  seek back / forward 30 match-seconds
  r           restart at kickoff
  q           quit

Headless export (no window needed):

    python pitch_replay.py --export-json replay.json --step 0.5

writes one interpolated world-state per ``--step`` match-seconds (default
1.0 s) for the whole match: ball + every player, including substitution
appearances and departures.  That JSON is the ready-to-animate dataset a
separate renderer (web, three.js, video pipeline, ...) can play 1:1.

Replay determinism: matches are seeded (default 42), and a finished match can
be saved to JSON (`--save replay.json` / `--load replay.json`) and replayed or
re-exported without re-simulating.

This module only READS the recorded chronology.  It modifies no match-engine
code and persists no season data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# PITCH MAPPING — same convention as exporter.plot_ball_motion:
# sim 105x68 -> mplsoccer StatsBomb 120x80, away attacks mirrored on x.
# ─────────────────────────────────────────────────────────────────────────

PITCH_LEN = 105.0
PITCH_WID = 68.0
SB_LEN = 120.0
SB_WID = 80.0


def to_statsbomb(x: float, y: float, home: bool) -> Tuple[float, float]:
    sx = (x / PITCH_LEN) * SB_LEN
    sy = (y / PITCH_WID) * SB_WID
    if not home:
        sx = ((PITCH_LEN - x) / PITCH_LEN) * SB_LEN
    return sx, sy


# ─────────────────────────────────────────────────────────────────────────
# ReplayData — the recorded payload (ball path + per-minute frames)
# ─────────────────────────────────────────────────────────────────────────


class ReplayData:
    """Serialisable snapshot of a match's recorded chronology."""

    def __init__(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_possession: float,
        duration_s: float,
        ball: List[Dict[str, Any]],
        frames: List[Dict[str, Any]],
        subs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        player_path: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    ):
        self.home_team = home_team
        self.away_team = away_team
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.home_possession = home_possession
        self.duration_s = duration_s
        self.ball = ball                    # [{"t","x","y",...}] sorted by t
        self.frames = frames                # per-minute {"minute","home","away"}
        self.subs = subs or {}
        self.player_path = player_path or {}  # {name: [(t, x, y), ...]} on clock

        self.ball_t = np.asarray([p["t"] for p in ball], dtype=float)
        self.ball_x = np.asarray([p["x"] for p in ball], dtype=float)
        self.ball_y = np.asarray([p["y"] for p in ball], dtype=float)

        self._idx = {"home": {}, "away": {}}
        for i, f in enumerate(self.frames):
            minute = float(f["minute"])
            for side in ("home", "away"):
                for r in f.get(side, []):
                    rec = self._idx[side].setdefault(
                        r["player"],
                        {"position": r["position"], "ts": [], "xs": [], "ys": []},
                    )
                    rec["ts"].append(minute * 60.0)
                    rec["xs"].append(r["x"])
                    rec["ys"].append(r["y"])
        for side in self._idx:
            for name, rec in self._idx[side].items():
                rec["ts"] = np.asarray(rec["ts"], dtype=float)
                rec["xs"] = np.asarray(rec["xs"], dtype=float)
                rec["ys"] = np.asarray(rec["ys"], dtype=float)
                # Replay-quality motion: the dense recorder path is the ground
                # truth when available; the per-minute frames stay as identity
                # source + fallback.
                hp = self.player_path.get(name)
                if hp is not None and len(hp) >= 2:
                    rec["hp_ts"] = np.asarray([s[0] for s in hp], dtype=float)
                    rec["hp_xs"] = np.asarray([s[1] for s in hp], dtype=float)
                    rec["hp_ys"] = np.asarray([s[2] for s in hp], dtype=float)
                    rec["hp"] = True
                    band_lo = rec["hp_ts"][0]
                    band_hi = rec["hp_ts"][-1]
                else:
                    rec["hp"] = False
                    band_lo = rec["ts"][0] - 60.0
                    band_hi = rec["ts"][-1]
                # A player listed in frame-minute m is on the pitch for the
                # whole of that minute.  First sample minute m1 => present
                # from (m1-1)*60; last sample minute m2 => present until m2*60.
                # apply_subs() may tighten these to exact substitution instants.
                rec["t_min"] = max(rec["ts"][0] - 60.0, band_lo)
                rec["t_max"] = min(rec["ts"][-1], band_hi)
                rec["t_excl_end"] = False
        if self.subs:
            self.apply_subs(self.subs)

    def apply_subs(self, subs: Dict[str, List[Dict[str, Any]]]) -> None:
        """Tighten on/off windows from exact sub instants.

        ``subs[side]`` is a list of {"player","side","on","off"} with
        substitution times (sim seconds).  "off" is the outgoing player who
        disappears at that second; "on" is the incoming one who appears.
        """
        self.subs = subs
        for side, lst in subs.items():
            for sub in lst:
                rec = self._idx.get(side, {}).get(sub["player"])
                if rec is None:
                    continue
                if sub.get("on") is not None:
                    rec["t_min"] = max(rec["t_min"], float(sub["on"]))
                if sub.get("off") is not None:
                    rec["t_max"] = min(rec["t_max"], float(sub["off"]))
                    rec["t_excl_end"] = True

    # ── ball ────────────────────────────────────────────────────────────

    def ball_at(self, t: float) -> Tuple[float, float]:
        """Interpolated ball position (sim metres) at match-time t."""
        t = float(np.clip(t, self.ball_t[0], self.ball_t[-1]))
        return float(np.interp(t, self.ball_t, self.ball_x)), float(
            np.interp(t, self.ball_t, self.ball_y)
        )

    # ── players ─────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(x: float) -> float:
        return x * x * (3.0 - 2.0 * x)

    def _pos_at(self, rec: Dict[str, Any], t: float) -> Optional[Tuple[float, float]]:
        if t < rec["t_min"]:
            return None  # not on the pitch during this window (a sub band)
        if t > rec["t_max"] or (rec.get("t_excl_end") and t >= rec["t_max"]):
            return None  # exact sub-off instant is exclusive (swap is atomic)
        if rec.get("hp"):
            # Dense ~5 Hz recorder path: plain linear interpolation between
            # genuine consecutive samples (no easing needed).
            ts, xs, ys = rec["hp_ts"], rec["hp_xs"], rec["hp_ys"]
            if t <= ts[0]:
                return float(xs[0]), float(ys[0])
            if t >= ts[-1]:
                return float(xs[-1]), float(ys[-1])
            i = int(np.searchsorted(ts, t, side="right")) - 1
            w = (t - ts[i]) / (ts[i + 1] - ts[i])
            x = xs[i] + (xs[i + 1] - xs[i]) * w
            y = ys[i] + (ys[i + 1] - ys[i]) * w
            return float(x), float(y)
        ts = rec["ts"]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        if i < 0:
            return float(rec["xs"][0]), float(rec["ys"][0])
        if i >= len(ts) - 1:
            return float(rec["xs"][-1]), float(rec["ys"][-1])
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        f = self._smooth(max(0.0, min(1.0, w)))
        x = rec["xs"][i] + (rec["xs"][i + 1] - rec["xs"][i]) * f
        y = rec["ys"][i] + (rec["ys"][i + 1] - rec["ys"][i]) * f
        return float(x), float(y)

    def side_at(self, side: str, t: float) -> List[Dict[str, Any]]:
        """Players on the pitch at match-time t for one team.

        Each row: {"player", "position", "x", "y"} in sim metres.  Players in
        a substitution window (off before their first sample / after their
        last sample) are omitted.
        """
        rows = []
        for name, rec in self._idx[side].items():
            pos = self._pos_at(rec, t)
            if pos is not None:
                rows.append({
                    "player": name,
                    "position": rec["position"],
                    "x": pos[0],
                    "y": pos[1],
                })
        return rows

    def players_at(self, t: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self.side_at("home", t), self.side_at("away", t)

    def frame_at(self, minute: int) -> Dict[str, Any]:
        """Nearest recorded per-minute frame (for phase/stance/score labels)."""
        best = None
        for f in self.frames:
            if best is None or abs(f["minute"] - minute) <= abs(best["minute"] - minute):
                best = f
        return best or {}

    # ── serialisation ───────────────────────────────────────────────────

    def to_json(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_possession": self.home_possession,
            "duration_s": round(self.duration_s, 3),
            "ball": self.ball,
            "frames": self.frames,
            "subs": self.subs,
            "player_path": {k: _decimate_path(v) for k, v in self.player_path.items()},
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ReplayData":
        return cls(
            home_team=data["home_team"],
            away_team=data["away_team"],
            home_goals=data["home_goals"],
            away_goals=data["away_goals"],
            home_possession=data["home_possession"],
            duration_s=data["duration_s"],
            ball=data["ball"],
            frames=data["frames"],
            subs=data.get("subs") or {},
            player_path=data.get("player_path") or {},
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str) -> "ReplayData":
        with open(path, "r") as f:
            return cls.from_json(json.load(f))


def _extract_subs(result: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Exact substitution instants from the match timeline.

    The engine records one SUBSTITUTION event per swap: ``player`` = player
    OFF, ``secondary_player`` = player ON, at ``minute`` granularity.  These
    tight windows remove the pre/post sub ghosting that the raw per-minute
    ``position_log`` cannot (the log dumps every registered spatial state).
    """
    out: Dict[str, List[Dict[str, Any]]] = {"home": [], "away": []}
    home_team = result.config.home_team
    away_team = result.config.away_team
    for e in getattr(result, "timeline", []):
        t = getattr(e, "event_type", None)
        if t is None or getattr(t, "name", "") != "SUBSTITUTION":
            continue
        side = "home" if e.team == home_team else ("away" if e.team == away_team else None)
        if side is None:
            continue
        t_s = float(e.minute) * 60.0
        if e.player:
            out[side].append({"player": e.player, "off": t_s})
        if e.secondary_player:
            out[side].append({"player": e.secondary_player, "on": t_s})
    return out


def _decimate_path(path, step: float = 0.5):
    """Keep one sample per ``step`` seconds so saved replays stay compact
    (5 Hz in-memory -> ~2 Hz on disk; the viewer re-interpolates losslessly)."""
    out = []
    last_bucket = -1.0
    for samp in path:
        bucket = float(samp[0]) // step
        if bucket != last_bucket:
            last_bucket = bucket
            out.append(samp)
    return out


def build_replay(result: Any) -> ReplayData:
    """Build ReplayData from a real MatchResult."""
    ball = list(result.full_match_ball_path)
    frames = list(result.position_log)
    duration_s = float(ball[-1]["t"]) if ball else 0.0
    if frames:
        duration_s = max(duration_s, float(frames[-1]["minute"]) * 60.0)
    player_path = {}
    hp = getattr(result, "full_match_player_path", None)
    if hp:
        player_path = hp.get("path") or {}
    return ReplayData(
        home_team=result.config.home_team,
        away_team=result.config.away_team,
        home_goals=result.home_goals,
        away_goals=result.away_goals,
        home_possession=result.home_possession_pct,
        duration_s=duration_s,
        ball=ball,
        frames=frames,
        subs=_extract_subs(result),
        player_path=player_path,
    )


# ─────────────────────────────────────────────────────────────────────────
# MATCH RUNNER — one real match via the scratch (non-persisting) runner
# ─────────────────────────────────────────────────────────────────────────


def run_scratch_match(seed: int = 42, verbose: bool = True) -> Any:
    """Simulate a real match using run_match.py's fixture.  Nothing is
    exported, no season file is written — the result object is returned."""
    import run_match as RM
    from match_engine import MatchEngine, MatchConfig
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController
    from manager_profile import ManagerPool

    random.seed(seed)

    home_squad = SquadBuilder.build(
        team_name=RM.HOME_TEAM,
        starters=RM.HOME_STARTERS,
        substitutes=RM.HOME_SUBS,
        team_superstars=RM.HOME_SUPERSTARS,
        set_piece_takers=RM.HOME_SP_TAKERS,
    )
    away_squad = SquadBuilder.build(
        team_name=RM.AWAY_TEAM,
        starters=RM.AWAY_STARTERS,
        substitutes=RM.AWAY_SUBS,
        team_superstars=RM.AWAY_SUPERSTARS,
        set_piece_takers=RM.AWAY_SP_TAKERS,
    )

    all_players_flat = (
        home_squad["starters"] + home_squad["substitutes"] +
        away_squad["starters"] + away_squad["substitutes"]
    )
    for player in all_players_flat:
        if player.name in RM.SOUL_PLAYERS:
            player.dna.soul = RM.SOUL_PLAYERS[player.name]

    config = MatchConfig(
        home_team=RM.HOME_TEAM,
        away_team=RM.AWAY_TEAM,
        match_date=RM.MATCH_DATE,
        matchday=RM.MATCHDAY,
        season=RM.SEASON,
        competition=RM.COMPETITION,
        venue=RM.VENUE,
        stadium_capacity=RM.CAPACITY,
        referee=RM.REFEREE,
        referee_strictness=RM.STRICTNESS,
        is_derby=RM.IS_DERBY,
    )

    mgr_pool = ManagerPool(
        clubs=[RM.HOME_TEAM, RM.AWAY_TEAM],
        style_lookup={RM.HOME_TEAM: RM.HOME_STYLE.style.value,
                      RM.AWAY_TEAM: RM.AWAY_STYLE.style.value},
    )
    home_mgr = mgr_pool.manager_for(RM.HOME_TEAM)
    away_mgr = mgr_pool.manager_for(RM.AWAY_TEAM)

    sub_controller = SubstitutionController(
        home_team=RM.HOME_TEAM,
        away_team=RM.AWAY_TEAM,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=RM.HOME_STYLE.style.value,
        away_style=RM.AWAY_STYLE.style.value,
        manager_stubbornness=RM.MANAGER_STUBBORNNESS,
    )
    sub_controller.MAX_SUBS = RM.MAX_SUBS

    engine = MatchEngine(config, RM.HOME_STYLE, RM.AWAY_STYLE)
    engine.set_squad(RM.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(RM.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_controller)
    engine.set_managers(home_manager=home_mgr, away_manager=away_mgr)

    t0 = time.time()
    result = engine.simulate()
    if verbose:
        print(f"[sim] {result.summary()}  ({time.time() - t0:.1f}s wall, "
              f"{result.match_clock_s:.0f}s match clock)")
    return result


# ─────────────────────────────────────────────────────────────────────────
# HEADLESS EXPORT — interpolated world-states on a fixed step
# ─────────────────────────────────────────────────────────────────────────


def export_interpolated(
    replay: ReplayData,
    step: float = 1.0,
    path: str = "replay.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Write one interpolated world-state per ``step`` match-seconds."""
    states = []
    t = 0.0
    while t <= replay.duration_s + 1e-9:
        minute = t / 60.0
        bx, by = replay.ball_at(t)
        home, away = replay.players_at(t)
        frame = replay.frame_at(int(minute))
        states.append({
            "t": round(t, 3),
            "minute": round(minute, 2),
            "score": [frame.get("home_goals", replay.home_goals),
                      frame.get("away_goals", replay.away_goals)],
            "possession_team": frame.get("possession_team", ""),
            "phase": frame.get("phase", ""),
            "home_stance": frame.get("home_stance", ""),
            "away_stance": frame.get("away_stance", ""),
            "ball": {"x": round(bx, 2), "y": round(by, 2)},
            "home": [
                {"player": r["player"], "position": r["position"],
                 "x": round(r["x"], 2), "y": round(r["y"], 2)}
                for r in home
            ],
            "away": [
                {"player": r["player"], "position": r["position"],
                 "x": round(r["x"], 2), "y": round(r["y"], 2)}
                for r in away
            ],
        })
        t += step

    payload = {
        "home_team": replay.home_team,
        "away_team": replay.away_team,
        "duration_s": replay.duration_s,
        "step": step,
        "n_states": len(states),
        "subs": replay.subs,
        "states": states,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    if verbose:
        print(f"[export] {len(states)} states @ {step}s -> {path}")
    return payload


# ─────────────────────────────────────────────────────────────────────────
# LIVE 2D VIEWER (matplotlib + mplsoccer)
# ─────────────────────────────────────────────────────────────────────────


class ReplayViewer:
    """Animated statsbomb pitch that plays the recorded match clock."""

    HOME_COLOR = "#003087"
    AWAY_COLOR = "#C8102E"
    BALL_COLOR = "#111111"
    PITCH_GREEN = "#2e7d32"
    PITCH_LINE = "#c8e6c9"

    def __init__(
        self,
        replay: ReplayData,
        speed: float = 1.0,
        fps: int = 30,
        show_labels: bool = False,
    ):
        import matplotlib.pyplot as plt
        from mplsoccer import Pitch

        self.replay = replay
        self.speed = speed
        self.fps = fps
        self.show_labels = show_labels
        self.t = 0.0
        self.paused = True  # start paused so the user can press space

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.06, 0.02, 0.88, 0.96])
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color=self.PITCH_GREEN,
            line_color=self.PITCH_LINE,
            line_alpha=1.0,
            line_zorder=2,
        )
        pitch.draw(ax=self.ax)
        self.ax.set_xlim(0, SB_LEN)
        self.ax.set_ylim(0, SB_WID)
        self.ax.set_facecolor(self.PITCH_GREEN)

        self.home_out = self.ax.scatter([], [], s=210, c=self.HOME_COLOR,
                                        edgecolors="white", linewidths=0.7, zorder=4)
        self.away_out = self.ax.scatter([], [], s=210, c=self.AWAY_COLOR,
                                        edgecolors="white", linewidths=0.7, zorder=4)
        self.home_gk = self.ax.scatter([], [], s=250, c=self.HOME_COLOR, marker="D",
                                       edgecolors="yellow", linewidths=0.9, zorder=5)
        self.away_gk = self.ax.scatter([], [], s=250, c=self.AWAY_COLOR, marker="D",
                                       edgecolors="yellow", linewidths=0.9, zorder=5)
        self.ball = self.ax.scatter([], [], s=120, c=self.BALL_COLOR,
                                    edgecolors="white", linewidths=0.8, zorder=6)

        self.score_text = self.ax.text(
            0.5, 1.03, "", transform=self.ax.transAxes, ha="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(facecolor="#111111", alpha=0.75, pad=6),
        )

        self.labels_home = []
        self.labels_away = []
        if self.show_labels:
            for i in range(11):
                th = self.ax.text(0, 0, "", fontsize=8, color="white",
                                  ha="center", va="center", zorder=6)
                ta = self.ax.text(0, 0, "", fontsize=8, color="white",
                                  ha="center", va="center", zorder=6)
                self.labels_home.append(th)
                self.labels_away.append(ta)

        self._last_wall = time.perf_counter()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._end_reached = False

    # ── state ──────────────────────────────────────────────────────────

    def _render(self) -> List[Any]:
        home, away = self.replay.players_at(self.t)
        bx, by = self.replay.ball_at(self.t)

        def _offsets(rows, is_home, gk_only):
            pts = []
            for r in rows:
                if (r["position"] == "GK") != gk_only:
                    continue
                pts.append(to_statsbomb(r["x"], r["y"], is_home))
            n = len(pts)
            return (np.asarray(pts, dtype=float).reshape(n, 2)
                    if n else np.empty((0, 2)))

        self.home_out.set_offsets(_offsets(home, True, False))
        self.away_out.set_offsets(_offsets(away, False, False))
        self.home_gk.set_offsets(_offsets(home, True, True))
        self.away_gk.set_offsets(_offsets(away, False, True))

        sxb, syb = to_statsbomb(bx, by, True)
        self.ball.set_offsets([(sxb, syb)])

        mm = int(self.t // 60)
        ss = int(self.t % 60)
        minute_str = f"{mm:02d}:{ss:02d}"
        frame = self.replay.frame_at(mm)
        phase = frame.get("phase", "")
        hg = frame.get("home_goals", self.replay.home_goals)
        ag = frame.get("away_goals", self.replay.away_goals)
        self.score_text.set_text(
            f"{self.replay.home_team} {hg}–"
            f"{ag} {self.replay.away_team}  |  "
            f"{minute_str}  |  possession {self.replay.home_possession}%  |  "
            f"{phase}  |  x{self.speed:.1f}"
        )

        if self.show_labels:
            for i, r in enumerate(home):
                if i < len(self.labels_home):
                    sx, sy = to_statsbomb(r["x"], r["y"], True)
                    self.labels_home[i].set_position((sx, sy + 3.5))
                    self.labels_home[i].set_text(r["player"].split()[-1])
            for i, r in enumerate(away):
                if i < len(self.labels_away):
                    sx, sy = to_statsbomb(r["x"], r["y"], False)
                    self.labels_away[i].set_position((sx, sy + 3.5))
                    self.labels_away[i].set_text(r["player"].split()[-1])

        artists = [self.home_out, self.away_out, self.home_gk, self.away_gk,
                   self.ball, self.score_text] + self.labels_home + self.labels_away
        return artists

    # ── animation ──────────────────────────────────────────────────────

    def _init(self) -> List[Any]:
        self._last_wall = time.perf_counter()
        return self._render()

    def _update(self, _frame: int) -> List[Any]:
        now = time.perf_counter()
        dt = now - self._last_wall
        self._last_wall = now
        if not self.paused and not self._end_reached:
            self.t += dt * self.speed
            if self.t >= self.replay.duration_s:
                self.t = float(self.replay.duration_s)
                self.paused = True
                self._end_reached = True
        return self._render()

    def _on_key(self, event: Any) -> None:
        key = event.key
        if key == " ":
            self.paused = not self.paused
            self._end_reached = False
        elif key == "+":
            self.speed = min(256.0, self.speed * 1.5)
        elif key == "-":
            self.speed = max(0.02, self.speed / 1.5)
        elif key in ("left", "right"):
            delta = 30.0 if key == "right" else -30.0
            self.t = max(0.0, min(self.replay.duration_s, self.t + delta))
        elif key == "r":
            self.t = 0.0
            self.paused = True
            self._end_reached = False
        elif key in ("q", "escape"):
            import matplotlib.pyplot as pltmod
            pltmod.close(self.fig)
            return
        self.fig.canvas.draw_idle()

    def play(self) -> None:
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        self._anim = FuncAnimation(
            self.fig, self._update, init_func=self._init,
            interval=int(1000.0 / self.fps), blit=True, cache_frame_data=False,
        )
        plt.show()


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a PLOFA match on a 2D pitch.")
    p.add_argument("--seed", type=int, default=42, help="Match seed (default 42).")
    p.add_argument("--load", type=str, default=None,
                   help="Replay a saved replay.json instead of simulating.")
    p.add_argument("--save", type=str, default=None,
                   help="Save the (freshly run) replay to replay.json.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Initial playback speed multiplier (1.0 = real 90 min).")
    p.add_argument("--fps", type=int, default=30, help="Playback frames per second.")
    p.add_argument("--labels", action="store_true",
                   help="Draw (short) player names next to markers.")
    p.add_argument("--export", type=str, default=None,
                   help="Headless: write interpolated states to this JSON file.")
    p.add_argument("--step", type=float, default=1.0,
                   help="Export step in match-seconds (used with --export).")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="Cap replay at this many match minutes (0 = full match).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.load:
        replay = ReplayData.load(args.load)
        print(f"[load] {replay.home_team} vs {replay.away_team} "
              f"{replay.home_goals}–{replay.away_goals} "
              f"({replay.duration_s:.0f}s match clock)")
    else:
        result = run_scratch_match(seed=args.seed, verbose=True)
        replay = build_replay(result)
        if args.save:
            replay.save(args.save)
            print(f"[save] -> {args.save}")

    if args.max_minutes > 0:
        replay.duration_s = min(replay.duration_s, args.max_minutes * 60.0)

    if args.export:
        export_interpolated(replay, step=args.step, path=args.export)
        return 0

    viewer = ReplayViewer(replay, speed=args.speed, fps=args.fps,
                          show_labels=args.labels)
    print("[controls] space=pause  +/-=speed  left/right=seek  r=restart  q=quit")
    viewer.play()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())