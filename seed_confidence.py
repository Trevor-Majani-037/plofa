"""
PLOFA 26/27 — CONFIDENCE SEEDING FROM LAST SEASON
===================================================
seed_confidence.py

Reads 25/26 PLAYER STATS from the previous season's Excel file and
seeds starting confidence values into season_state.json using a
regression-toward-the-mean formula.

Approach: Option C — Seeded with Regression
  • Players with last-season data get confidence derived from their
    average rating, goals, and assists, regressed 55% toward 50.
  • Players with no data (new/reserve) start at 50.
  • Result: elite performers start ~62-67, average ~50, poor ~35-45.

Usage:
    python seed_confidence.py [--dry-run]
"""

from __future__ import annotations
import json
import os
import re
import unicodedata
from typing import Dict, Optional, Tuple

import openpyxl

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREV_SEASON = os.path.join(
    r"D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM",
    "PLOFA 2025-2026 II.xlsx",
)
CURRENT_SEASON = os.path.join(BASE_DIR, "PLOFA-2026-2027.xlsx")
SEASON_STATE = os.path.join(BASE_DIR, "season_state.json")


# ── Helpers ────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Strip accents, lower-case, collapse whitespace, remove non-alnum."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _safe_rating(val) -> Optional[float]:
    """Safely convert a cell to float, returning None on failure."""
    if val is None:
        return None
    try:
        r = float(val)
        return r if 1.0 <= r <= 10.0 else None
    except (ValueError, TypeError):
        return None


def _safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ── Formula ────────────────────────────────────────────────────

def compute_confidence(
    rating: Optional[float],
    goals: int,
    assists: int,
    starts: int,
) -> float:
    """
    Compute starting confidence for a player from last-season stats.

    Design targets:
      - Elite (8.0 rating, 25+ goals)   -> ~65
      - Strong (7.0-7.5, decent G+A)    -> ~55-62
      - Average (6.5, no goals)          -> ~50
      - Below avg (6.0)                  -> ~43-45
      - Poor (5.5)                       -> ~38-40
      - No data / new player             -> 50 (handled externally)

    Formula:
      1. Rating maps to a base score (0-100 linear from 5.0-8.0)
      2. Goal/assist bonus is added (capped)
      3. 50% regression toward 50 prevents over-amplification
    """
    # -- Rating component (primary, ~80% weight) --
    if rating is not None:
        # Linear: 5.0 -> 0, 6.5 -> 50, 8.0 -> 100
        base = max(0.0, min(100.0, (rating - 5.0) / 3.0 * 100.0))
    else:
        base = 50.0

    # -- Goal/assist bonus (secondary, ~20% weight, capped) --
    if starts > 0:
        g_per_start = goals / starts
        a_per_start = assists / starts
    else:
        g_per_start = 0.0
        a_per_start = 0.0

    # Each goal per start adds up to 15 pts, each assist up to 8 pts
    goal_bonus = min(15.0, g_per_start * 20.0)
    assist_bonus = min(8.0, a_per_start * 12.0)
    bonus = goal_bonus + assist_bonus

    # Combined: 80% rating + 20% bonus (so bonus can't dominate)
    raw = base * 0.80 + bonus * 0.20

    # -- Regression toward 50 --
    confidence = 50.0 + (raw - 50.0) * 0.50

    return round(max(30.0, min(70.0, confidence)), 1)


# ── Excel readers ──────────────────────────────────────────────

def read_prev_season_stats(path: str) -> Dict[str, dict]:
    """
    Read PLAYER STATS sheet from the previous season's Excel.
    Returns dict keyed by normalised player name -> {rating, goals, assists, starts}.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["PLAYER STATS"]

    # Known column positions for 25/26 PLAYER STATS sheet (1-indexed):
    # Col 4 = Player Name (header is emoji 👤)
    # Col 7 = Started
    # Col 8 = Goals (header is emoji ⚽)
    # Col 9 = Assists
    # Col 247 = Rating
    #
    # NOTE: Emoji headers normalise to empty string "", so header-based
    # detection fails. We use the known positions directly.
    name_col = 4
    starts_col = 7
    goals_col = 8
    assists_col = 9
    rating_col = 247

    players = {}
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
        # Skip rows with no player name
        raw_name = row[name_col - 1] if name_col - 1 < len(row) else None
        if not raw_name or not str(raw_name).strip():
            continue
        name = str(raw_name).strip()
        norm = _normalise(name)

        rating = _safe_rating(row[rating_col - 1] if rating_col - 1 < len(row) else None)
        goals = _safe_int(row[goals_col - 1] if goals_col - 1 < len(row) else None)
        assists = _safe_int(row[assists_col - 1] if assists_col - 1 < len(row) else None)
        starts = _safe_int(row[starts_col - 1] if starts_col - 1 < len(row) else None, 1)

        # Skip summary/totals rows
        if goals > 100:
            continue

        players[norm] = {
            "name": name,
            "rating": rating,
            "goals": goals,
            "assists": assists,
            "starts": starts,
        }

    wb.close()
    return players


def read_current_roster(path: str) -> list:
    """
    Read PLAYERS sheet from current season's Excel.
    Returns list of (raw_name, normalised_name) tuples.
    The RAW name is what the match engine uses as PlayerProfile.name and
    therefore what season_state.json keys MUST match exactly.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["PLAYERS"]

    # Find player name column
    headers = {}
    for col_idx, cell in enumerate(next(ws.iter_rows(min_row=1, max_row=1, values_only=False)), 1):
        val = cell.value
        if val is not None:
            headers[_normalise(str(val))] = col_idx

    name_col = None
    for key, idx in headers.items():
        if "player" in key and "name" in key:
            name_col = idx
            break
        elif key == "player name":
            name_col = idx
            break

    if name_col is None:
        # Fallback: column 5
        name_col = 5

    names = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        # Skip aggregate/summary rows (club == "Total" or empty), matching
        # roster_loader.py behaviour
        club_val = row[0] if row else None
        if club_val is not None:
            club_str = str(club_val).strip()
            if club_str.lower() in ("total", "") or club_str.lower() in ("", "all"):
                continue
        raw_name = row[name_col - 1] if name_col - 1 < len(row) else None
        if raw_name and str(raw_name).strip():
            raw = str(raw_name).strip()
            names.append((raw, _normalise(raw)))

    wb.close()
    return names


# ── Main seeding logic ────────────────────────────────────────

def seed_confidence(dry_run: bool = False):
    print("=" * 60)
    print("  PLOFA 26/27 — CONFIDENCE SEEDING")
    print("  Option C: Seeded with Regression toward Mean")
    print("=" * 60)
    print()

    # Load last season stats
    print(f"Reading 25/26 stats from: {PREV_SEASON}")
    prev_stats = read_prev_season_stats(PREV_SEASON)
    print(f"  -> {len(prev_stats)} players with stats from last season")
    print()

    # Load current roster
    print(f"Reading 26/27 roster from: {CURRENT_SEASON}")
    current_roster = read_current_roster(CURRENT_SEASON)
    print(f"  -> {len(current_roster)} players in current roster")
    print()

    # Load existing season state
    with open(SEASON_STATE, "r", encoding="utf-8-sig") as f:
        state = json.load(f)

    # Fresh-season seed: reset players to an empty dict so we write exactly
    # one entry per current-roster player keyed by their exact raw name.
    state["players"] = {}

    seeded = 0
    missed = 0
    new_players = 0
    results = []

    # Keep a lookup of prev-season stats by normalised name
    prev_by_norm = {}
    for norm, data in prev_stats.items():
        # On collision keep the first (by rating desc preferred)
        prev_by_norm.setdefault(norm, data)

    for raw_name, norm_name in current_roster:
        if norm_name in prev_by_norm:
            s = prev_by_norm[norm_name]
            conf = compute_confidence(
                rating=s["rating"],
                goals=s["goals"],
                assists=s["assists"],
                starts=s["starts"],
            )
            results.append((raw_name, s["rating"], s["goals"], s["assists"], conf))
            seeded += 1
        else:
            conf = 50.0
            new_players += 1

        state["players"][raw_name] = {
            "confidence": conf,
            "fatigue_level": 0.0,
            "starting_stamina": 100.0,
            "is_injured": False,
            "injury_type": "none",
            "matches_remaining_out": 0,
            "yellow_cards_last_6": [],
            "red_card_ban": 0,
            "season_minutes": 0,
            "season_matches": 0,
            "recent_ratings": [6.0] * 5,
            "recent_goals": [0] * 5,
            "injury_history": [],
            "injury_date": None,
            "recovery_days": 0,
            "expected_return_date": None,
        }

    # Show seeding results
    results.sort(key=lambda x: -x[4])
    print("─" * 75)
    print(f"{'Player':<28} {'Rating':>7} {'G':>4} {'A':>4} {'Conf':>7}")
    print("─" * 75)
    for name, rating, goals, assists, conf in results[:25]:
        r_str = f"{rating:.2f}" if rating else "N/A"
        print(f"{name:<28} {r_str:>7} {goals:>4} {assists:>4} {conf:>7.1f}")
    print("─" * 75)
    print(f"  Seeded: {seeded} players | New/unmatched: {new_players} (all at 50.0)")
    print()

    # Stats
    confs = [r[4] for r in results]
    print(f"  Confidence distribution among seeded players:")
    print(f"    Min:    {min(confs):.1f}")
    print(f"    Max:    {max(confs):.1f}")
    print(f"    Mean:   {sum(confs)/len(confs):.1f}")
    print(f"    Median: {sorted(confs)[len(confs)//2]:.1f}")
    print()

    if dry_run:
        print("  [DRY RUN] No changes written.")
    else:
        # Atomic save
        tmp = f"{SEASON_STATE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, SEASON_STATE)
        print(f"  [OK] Written to {SEASON_STATE}")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    seed_confidence(dry_run=dry_run)
