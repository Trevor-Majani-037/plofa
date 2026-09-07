"""
Patch all match JSONs in plofa_output/ with corrected chance-creation stats
from the fixed ChanceCreationLedger.

This avoids re-running matches — it re-derives creation stats directly
from each match's timeline and writes them back into the players blocks.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional

from match_engine import MatchEvent, EventType, SituationType
from chance_creation import ChanceCreationLedger

OUTPUT_DIR = "plofa_output"
PLAYER_CREATION_KEYS = {
    "assists", "goal_assists", "shot_assists", "chances_created",
    "big_chances_created", "open_play_cc", "setpiece_cc",
    "open_play_assists", "setpiece_assists",
    "open_play_shot_assists", "setpiece_shot_assists",
    "xa", "xa_open_play", "xa_setpiece",
    "second_assists", "sca", "fantasy_assists",
}


def _load_match_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if "match" in data and "players" in data:
            return data
    except Exception:
        pass
    return None


def _build_timeline(data: Dict[str, Any]) -> list:
    timeline = []
    for e in data.get("timeline", []):
        situation = None
        sit_str = e.get("situation", "")
        if sit_str:
            try:
                situation = SituationType[sit_str.upper()]
            except Exception:
                situation = None

        evt = MatchEvent(
            event_type=EventType[e["type"]],
            team=e["team"],
            player=e["player"],
            secondary_player=e.get("secondary_player") or None,
            minute=e.get("minute", 0),
            second=e.get("second", 0),
            location_x=e.get("x", 50.0),
            location_y=e.get("y", 34.0),
            end_x=e.get("end_x"),
            end_y=e.get("end_y"),
            xg=e.get("xg", 0.0),
            xa=e.get("xa", 0.0),
            outcome=e.get("outcome", True),
            situation=situation,
            metadata=e.get("metadata", {}),
        )
        timeline.append(evt)
    return timeline


def _patch_match(path: str) -> Dict[str, Any]:
    data = _load_match_json(path)
    if data is None:
        return {"file": path, "status": "skipped", "reason": "not a match json"}

    timeline = _build_timeline(data)
    ledger = ChanceCreationLedger(timeline).compute()

    patched_players = []
    for name, s in data.get("players", {}).items():
        cc = ledger.per_player.get(name)
        if cc:
            for key in PLAYER_CREATION_KEYS:
                if key in cc:
                    s[key] = cc[key]
            patched_players.append(name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "file": os.path.relpath(path, OUTPUT_DIR),
        "status": "patched",
        "players_updated": len(patched_players),
        "sample": patched_players[:5],
    }


def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output directory not found: {OUTPUT_DIR}")
        return

    results = []
    for entry in sorted(os.listdir(OUTPUT_DIR)):
        full = os.path.join(OUTPUT_DIR, entry)
        if not os.path.isdir(full):
            continue
        json_path = None
        for fname in os.listdir(full):
            if fname.endswith(".json"):
                candidate = os.path.join(full, fname)
                if _load_match_json(candidate):
                    json_path = candidate
                    break
        if json_path:
            results.append(_patch_match(json_path))

    print(f"\nPatched {len(results)} match JSONs:\n")
    for r in results:
        if r["status"] == "patched":
            print(f"  ✅ {r['file']}  ({r['players_updated']} players updated)")
        else:
            print(f"  ⏭  {r['file']}  ({r['reason']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
