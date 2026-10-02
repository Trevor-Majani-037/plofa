"""
Fetch a StatsBomb open-data competition season and reduce every match to a
compact per-team-per-match feature row.

Usage
-----
    python -m battlefield.fetch_statsbomb --competition 11 --season 90 \
            --max-matches 150 --out battlefield/data/statsbomb_feats.json

Produces a JSON list of dicts, one per team-match, keyed as:
    {comp_id, season_id, match_id, team, is_home,
     goals, xg, shots, shots_on_target, shots_inside_box_pct,
     passes, pass_accuracy_pct, possession_pct,
     corners, fouls, yellow_cards, red_cards, offsides, ball_recoveries}
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_TIMEOUT = 20  # seconds per request

# ---------- helpers -----------------------------------------------------------

def _fetch_json(url: str, retries: int = 3) -> Any:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "plofa-battlefield/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last}")


def _load_manifest(path: Path) -> set:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_manifest(path: Path, ids: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


def _save_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)


# ---------- reduction ---------------------------------------------------------

_PENALTY_XG = 0.79
_GOAL_KICK = "Goal Kick"
_KICK_OFF   = "Kick Off"
_THROW_IN   = "Throw-in"
_BOX_FRACTION = 16.5 / 105.0   # real 18-yard box depth / pitch length


def _reduce_match(match: Dict, events: List[Dict], comp_id: int, season_id: int) -> List[Dict]:
    """Return two feature dicts, one per team."""
    home_name: str = match["home_team"]["home_team_name"]
    away_name: str = match["away_team"]["away_team_name"]
    match_id = match["match_id"]

    feats: Dict[str, Dict] = {
        home_name: _empty_feats(comp_id, season_id, match_id, home_name, True),
        away_name: _empty_feats(comp_id, season_id, match_id, away_name, False),
    }

    # possession event counts (proxy for possession %)
    poss_counts: Dict[str, int] = {home_name: 0, away_name: 0}

    for ev in events:
        etype = (ev.get("type") or {}).get("name")
        team = (ev.get("team") or {}).get("name")
        poss_team = (ev.get("possession_team") or {}).get("name")
        if team not in feats:
            continue

        # possession share proxy (event-level tagging)
        if poss_team and poss_team in poss_counts:
            poss_counts[poss_team] += 1

        if etype == "Pass":
            ptype = ((ev.get("pass") or {}).get("type") or {}).get("name")
            if ptype in (_GOAL_KICK, _KICK_OFF, _THROW_IN):
                continue
            outcome = (ev.get("pass") or {}).get("outcome")
            f = feats[team]
            f["passes"] += 1
            if outcome is None:
                f["passes_completed"] += 1
            if ptype == "Corner":
                f["corners"] += 1

        elif etype == "Shot":
            shot = ev.get("shot") or {}
            sout = (shot.get("outcome") or {}).get("name")
            xg = shot.get("xg") if shot.get("xg") is not None else shot.get("statsbomb_xg", 0.0)
            f = feats[team]
            f["shots"] += 1
            f["xg"] += float(xg or 0.0)
            loc = ev.get("location") or [60.0, 40.0]
            # inside-box check using real 18-yard depth scaled to StatsBomb coords
            _in_box = _shot_in_box(loc, match, home_name, team)
            if _in_box:
                f["shots_inside_box"] += 1
            if sout in ("Goal", "Saved"):
                f["shots_on_target"] += 1

        elif etype == "Foul Committed":
            feats[team]["fouls"] += 1
            _card = (ev.get("foul_committed") or {}).get("card") or (ev.get("card"))
            if _card:
                cname = _card.get("name")
                if cname == "Yellow Card":
                    feats[team]["yellow_cards"] += 1
                elif cname in ("Red Card", "Second Yellow"):
                    feats[team]["red_cards"] += 1

        elif etype == "Card":
            cname = (ev.get("card") or {}).get("name")
            if cname == "Yellow Card":
                feats[team]["yellow_cards"] += 1
            elif cname in ("Red Card", "Second Yellow"):
                feats[team]["red_cards"] += 1

        elif etype == "Offside":
            feats[team]["offsides"] += 1

        elif etype == "Ball Recovery":
            feats[team]["ball_recoveries"] += 1

    # finalize derived fields
    total_poss = sum(poss_counts.values()) or 1
    for name, f in feats.items():
        f["possession_pct"] = round(100.0 * poss_counts.get(name, 0) / total_poss, 1)
        f["pass_accuracy_pct"] = round(100.0 * f["passes_completed"] / max(1, f["passes"]), 1)
        f["shots_inside_box_pct"] = round(100.0 * f["shots_inside_box"] / max(1, f["shots"]), 1)
        # use official match score for goals
        f["goals"] = match["home_score"] if f["is_home"] else match["away_score"]
        # clean transient keys
        del f["passes_completed"]
        del f["shots_inside_box"]

    return [feats[home_name], feats[away_name]]


def _empty_feats(comp_id, season_id, match_id, team, is_home) -> Dict:
    return {
        "comp_id": comp_id, "season_id": season_id, "match_id": match_id,
        "team": team, "is_home": is_home,
        "goals": 0, "xg": 0.0, "shots": 0, "shots_on_target": 0,
        "shots_inside_box": 0, "shots_inside_box_pct": 0.0,
        "passes": 0, "passes_completed": 0, "pass_accuracy_pct": 0.0,
        "possession_pct": 50.0,
        "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0,
        "offsides": 0, "ball_recoveries": 0,
    }


def _shot_in_box(loc: List[float], match: Dict, home_name: str, team: str) -> bool:
    """Is the shot inside the 16.5 m box?

    StatsBomb normalises EVERY attacking action to the 0→120 direction:
    the shot goal is always at x=120, whichever team shoots. So the depth
    from the goal line is always ``120 - x``.
    """
    x = loc[0]
    return (120.0 - x) / 120.0 <= _BOX_FRACTION


# ---------- CLI ---------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Fetch StatsBomb open data and reduce to features.")
    p.add_argument("--competition", type=int, default=11, help="Competition ID (default 11 = La Liga)")
    p.add_argument("--season", type=int, default=90, help="Single season ID (kept for compat)")
    p.add_argument("--seasons", type=str, default=None, help="Comma list of season IDs, e.g. 90,42,4")
    p.add_argument("--max-matches", type=int, default=200, help="Max matches to fetch")
    p.add_argument("--offset", type=int, default=0, help="Skip first N matches (resume support)")
    p.add_argument("--out", type=str, default="battlefield/data/statsbomb_feats.json")
    p.add_argument("--manifest", type=str, default="battlefield/data/sb_manifest.json")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load existing rows + manifest to enable resume
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            rows: List[Dict] = json.load(f)
    else:
        rows = []
    completed = _load_manifest(manifest_path)

    # fetch match list(s)
    seasons = [int(s.strip()) for s in (args.seasons or str(args.season)).split(",") if s.strip()]
    done_total = 0
    for sid in seasons:
        matches_url = f"{_REPO}/matches/{args.competition}/{sid}.json"
        print(f"Fetching match list: {matches_url}")
        matches = _fetch_json(matches_url)
        print(f"Season {sid}: {len(matches)} matches")

        targets = [m for m in matches if m["match_id"] not in completed][args.offset:]
        targets = targets[: args.max_matches]
        print(f"  New matches to fetch: {len(targets)}")

        for i, m in enumerate(targets, 1):
            mid = m["match_id"]
            ev_url = f"{_REPO}/events/{mid}.json"
            try:
                events = _fetch_json(ev_url)
            except Exception as exc:
                print(f"  WARN: match {mid} failed ({exc}), skipping.")
                continue
            pair = _reduce_match(m, events, args.competition, sid)
            rows.extend(pair)
            completed.add(mid)
            if i % 10 == 0 or i == len(targets):
                _save_rows(out_path, rows)
                _save_manifest(manifest_path, completed)
                print(f"  [{i}/{len(targets)}] saved {len(rows)} rows")
            else:
                _save_rows(out_path, rows)
                _save_manifest(manifest_path, completed)
            done_total += 1

    _save_rows(out_path, rows)
    _save_manifest(manifest_path, completed)
    print(f"\nDone. Total rows: {len(rows)}  |  file: {out_path}")


if __name__ == "__main__":
    main()
