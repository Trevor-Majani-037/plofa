"""broadcast.py — PLOFA LIVE broadcast builder + server.

Turns a simulated match (fresh scratch match, a saved replay.json, or a
finished plofa_output fixture folder) into an EPL-style broadcast viewer
(``viewer.html`` + ``broadcast_data.json``) with score bug, live stats,
commentary feed, lower-thirds and camera modes.

Usage (PowerShell, venv Python):
    # fresh scratch match, serve on :8765 and open the browser
    & ".\\.venv\\Scripts\\python.exe" plofa_broadcast\\broadcast.py --seed 42

    # watch a match you already exported (must have replay.json)
    & ".\\.venv\\Scripts\\python.exe" plofa_broadcast\\broadcast.py --replay plofa_output\\Justice_vs_Rodice_MD09\\replay.json

    # attach events from a saved match.json so banners/stats exist
    & ".\\.venv\\Scripts\\python.exe" plofa_broadcast\\broadcast.py --replay plofa_output\\Justice_vs_Rodice_MD09\\replay.json --match-json plofa_output\\Justice_vs_Rodice_MD09\\Justice_vs_Rodice_MD9.json

    # build only (no server) + standalone single-file for preview
    & ".\\.venv\\Scripts\\python.exe" plofa_broadcast\\broadcast.py --seed 42 --no-server --standalone

This module only READS recorded data or simulates a non-persisting
scratch match. It never writes to season_state.json or plofa_output/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _event_time(e: Any) -> float:
    """True match-clock seconds for a MatchEvent (never the random
    placeholder ``second`` field — see match_engine.MatchEvent)."""
    t = getattr(e, "match_clock_s", None)
    if t is not None:
        return float(t)
    return float(getattr(e, "minute", 0)) * 60.0 + float(getattr(e, "second", 0) or 0)


def _event_text(e: Any) -> str:
    name = getattr(getattr(e, "event_type", None), "name", str(getattr(e, "event_type", "")))
    player = getattr(e, "player", "") or ""
    sec = getattr(e, "secondary_player", "") or ""
    if name == "GOAL":
        return f"GOAL {player}" + (f" (assist {sec})" if sec else "")
    if name == "OWN_GOAL":
        return f"Own goal {player}"
    if name == "PENALTY_SCORED":
        return f"Penalty scored {player}"
    if name == "PENALTY_MISSED":
        return f"Penalty missed {player}"
    if name == "SUBSTITUTION":
        return f"{player} → {sec}" if sec else f"{player} off"
    if name in ("YELLOW_CARD", "RED_CARD"):
        return f"{name.replace('_', ' ').title()} {player}"
    if name == "HIT_WOODWORK":
        return f"{player} hits the woodwork"
    if name == "VAR_DISALLOWED_GOAL":
        return f"{player} — goal disallowed (VAR)"
    if name == "INJURY":
        return f"{player} — injury"
    if name == "BIG_CHANCE_CREATED":
        return f"Big chance {player}"
    if name in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SAVE"):
        return f"{player} — {name.replace('_', ' ').lower()}"
    return player


def build_payload(replay: Any, events: List[Dict[str, Any]], meta_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Interpolated broadcast states (1 s step) + normalised events."""
    from pitch_replay import build_replay  # noqa: F401  (kept for parity)
    states: List[Dict[str, Any]] = []
    step = float(meta_overrides.pop("step", 1.0))
    t = 0.0
    while t <= replay.duration_s + 1e-9:
        minute = t / 60.0
        bx, by = replay.ball_at(t)
        home, away = replay.players_at(t)
        frame = replay.frame_at(int(minute))
        states.append({
            "t": round(t, 2),
            "minute": round(minute, 2),
            "score": [frame.get("home_goals", replay.home_goals),
                      frame.get("away_goals", replay.away_goals)],
            "possession_team": frame.get("possession_team", ""),
            "phase": frame.get("phase", ""),
            "ball": {"x": round(bx, 1), "y": round(by, 1)},
            "home": [{"player": r["player"], "position": r["position"],
                      "x": round(r["x"], 1), "y": round(r["y"], 1)} for r in home],
            "away": [{"player": r["player"], "position": r["position"],
                      "x": round(r["x"], 1), "y": round(r["y"], 1)} for r in away],
        })
        t += step
    meta = {
        "home": replay.home_team,
        "away": replay.away_team,
        "home_goals": replay.home_goals,
        "away_goals": replay.away_goals,
        "home_possession": replay.home_possession,
        "duration_s": round(replay.duration_s, 1),
        "competition": "PLOFA",
        "season": "26/27",
        "matchday": "",
    }
    meta.update(meta_overrides)
    return {"meta": meta, "duration": round(replay.duration_s, 1),
            "states": states, "events": events}


def events_from_result(result: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in getattr(result, "timeline", []) or []:
        name = getattr(getattr(e, "event_type", None), "name", None)
        if not name:
            continue
        out.append({
            "t": round(_event_time(e), 1),
            "type": name,
            "team": getattr(e, "team", "") or "",
            "player": getattr(e, "player", "") or "",
            "secondary": getattr(e, "secondary_player", "") or "",
            "xg": float(getattr(e, "xg", 0) or 0),
            "text": _event_text(e),
        })
    out.sort(key=lambda d: d["t"])
    return out


def events_from_match_json(path: str) -> List[Dict[str, Any]]:
    """Parse an exporter ``*.json`` timeline into broadcast events."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    out: List[Dict[str, Any]] = []
    for e in doc.get("timeline", []):
        name = str(e.get("type", ""))
        if not name:
            continue
        t = e.get("clock_s")
        if t is None:
            t = float(e.get("minute", 0)) * 60.0
        player = e.get("player", "") or ""
        sec = e.get("secondary_player", "") or ""
        if name == "GOAL":
            text = f"GOAL {player}" + (f" (assist {sec})" if sec else "")
        elif name == "SUBSTITUTION":
            text = f"{player} → {sec}" if sec else f"{player} off"
        elif name in ("YELLOW_CARD", "RED_CARD"):
            text = f"{name.replace('_', ' ').title()} {player}"
        elif name == "PENALTY_SCORED":
            text = f"Penalty scored {player}"
        elif name == "OWN_GOAL":
            text = f"Own goal {player}"
        elif name == "HIT_WOODWORK":
            text = f"{player} hits the woodwork"
        elif name == "VAR_DISALLOWED_GOAL":
            text = f"{player} — goal disallowed (VAR)"
        else:
            text = player
        out.append({
            "t": round(float(t), 1), "type": name,
            "team": e.get("team", "") or "", "player": player,
            "secondary": sec, "xg": float(e.get("xg", 0) or 0), "text": text,
        })
    out.sort(key=lambda d: d["t"])
    return out


def write_standalone(data: Dict[str, Any], out_path: str) -> None:
    with open(os.path.join(HERE, "viewer.html"), encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("<!--BCAST_DATA-->",
                        f"<script>window.BCAST_DATA={payload};</script>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[standalone] -> {out_path} ({len(html)/1e6:.1f} MB)")


def serve(port: int) -> None:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    os.chdir(HERE)
    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass
    print(f"[serve] PLOFA LIVE -> http://localhost:{port}/  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="PLOFA LIVE broadcast builder.")
    p.add_argument("--seed", type=int, default=42, help="Seed for a fresh scratch match.")
    p.add_argument("--replay", type=str, default=None,
                   help="Replay a saved replay.json instead of simulating.")
    p.add_argument("--match-json", type=str, default=None,
                   help="Exporter match.json whose timeline supplies events.")
    p.add_argument("--step", type=float, default=1.0, help="Broadcast state step in match-seconds.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-server", action="store_true", help="Build files only, do not serve.")
    p.add_argument("--standalone", action="store_true",
                   help="Also write broadcast_standalone.html with data embedded.")
    p.add_argument("--no-standalone", action="store_true",
                   help="Do not write broadcast_standalone.html.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    events: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"step": args.step}

    if args.replay:
        from pitch_replay import ReplayData
        replay = ReplayData.load(args.replay)
        print(f"[load] {replay.home_team} vs {replay.away_team} "
              f"{replay.home_goals}–{replay.away_goals} ({replay.duration_s:.0f}s clock)")
        if args.match_json:
            events = events_from_match_json(args.match_json)
            print(f"[events] {len(events)} from {args.match_json}")
    else:
        from pitch_replay import run_scratch_match, build_replay
        result = run_scratch_match(seed=args.seed, verbose=True)
        replay = build_replay(result)
        events = events_from_result(result)
        print(f"[events] {len(events)} from live sim")
        try:
            import run_match as RM  # fixture metadata for the bug
            meta.update({"competition": RM.COMPETITION, "season": RM.SEASON,
                         "matchday": RM.MATCHDAY})
        except Exception:
            pass
        # also persist replay.json next to the viewer for re-watch
        replay.save(os.path.join(HERE, "replay.json"))
        print(f"[save] -> {os.path.join(HERE, 'replay.json')}")

    payload = build_payload(replay, events, meta)
    data_path = os.path.join(HERE, "broadcast_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[export] {len(payload['states'])} states @ {args.step}s -> {data_path} "
          f"({os.path.getsize(data_path)/1e6:.1f} MB)")
    print(f"[meta] {payload['meta']['home']} vs {payload['meta']['away']} "
          f"{payload['meta']['home_goals']}–{payload['meta']['away_goals']}")

    if args.standalone or not args.no_standalone:
        write_standalone(payload, os.path.join(HERE, "broadcast_standalone.html"))

    if not args.no_server:
        serve(args.port)
    else:
        print("[done] serve with: in plofa_broadcast run  "
              "python -m http.server 8765  then open http://localhost:8765/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
