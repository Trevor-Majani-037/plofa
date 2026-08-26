"""
PLOFA Stats Hub â€” Data Builder
================================
Reads all match JSONs from plofa_output/ and season_stats.json,
computes league table from scratch, and outputs clean JSON
to the adventurous-mendel/data/ directory.

Usage: python build_app_data.py
"""

import os
import sys
import json
import glob
import math
from pathlib import Path
from collections import defaultdict

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PLOFA_DIR = Path(__file__).parent
OUTPUT_DIR = Path(r"C:\Users\Trevor Majani\Documents\antigravity\adventurous-mendel\data")
SEASON_STATS_FILE = PLOFA_DIR / "season_stats.json"
SEASON_STATE_FILE = PLOFA_DIR / "season_state.json"
PLOFA_OUTPUT_DIR = PLOFA_DIR / "plofa_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filename):
    out = OUTPUT_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] {filename} ({out.stat().st_size // 1024}KB)")

def safe_round(val, ndigits=2):
    """Round a value, returning 0.0 if val is None or not numeric."""
    try:
        return round(float(val or 0), ndigits)
    except (TypeError, ValueError):
        return 0.0


# â”€â”€â”€ Discover all match JSON files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def discover_matches():
    matches = []
    for match_dir in sorted(PLOFA_OUTPUT_DIR.iterdir()):
        if not match_dir.is_dir():
            continue
        for jf in match_dir.glob("*.json"):
            # Skip player CSV's json twin; only take the main match json
            if "players" not in jf.name.lower():
                matches.append(jf)
    return matches

# â”€â”€â”€ Parse one match JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def parse_match(path: Path):
    data = load_json(path)
    info = data.get("match", {})
    goals = data.get("goals", [])
    players = data.get("players", {})

    home = info.get("home_team", "")
    away = info.get("away_team", "")
    score_str = info.get("score", "0â€“0")
    
    # Parse score â€” handles unicode dash variants
    score_str_clean = score_str.replace("â€“", "-").replace("â€”", "-").replace("\u2013", "-")
    parts = score_str_clean.split("-")
    home_goals_count = int(parts[0].strip()) if len(parts) == 2 else 0
    away_goals_count = int(parts[1].strip()) if len(parts) == 2 else 0

    # Aggregate team stats from player data
    team_stats = {}
    for pname, pdata in players.items():
        team = pdata.get("team", "")
        if team not in team_stats:
            team_stats[team] = {
                "shots_on_target": 0, "shots_off_target": 0, "shots_blocked_att": 0,
                "passes_attempted": 0, "passes_completed": 0,
                "xg": 0.0, "xa": 0.0,
                "tackles_won": 0, "tackles_att": 0,
                "interceptions": 0, "clearances": 0,
                "fouls_committed": 0, "yellow_cards": 0, "red_cards": 0,
                "corners_att": 0, "offsides": 0,
                "big_chances_scored": 0, "big_chances_missed": 0,
                "progressive_passes": 0, "key_passes": 0,
                "crosses_att": 0, "crosses_comp": 0,
                "aerial_duels_att": 0, "aerial_duels_won": 0,
                "ground_duels_att": 0, "ground_duels_won": 0,
                "dribbles_att": 0, "dribbles_comp": 0,
                "pressures": 0, "press_success": 0,
                "distance_covered": 0.0, "sprints": 0,
                "touches": 0, "recoveries": 0,
                "saves": 0, "goals_conceded": 0,
                "chances_created": 0, "big_chances_created": 0,
            }
        ts = team_stats[team]
        for field in ts:
            val = pdata.get(field, 0)
            if isinstance(val, (int, float)):
                ts[field] += val

    # Compute possession from passes
    home_passes = team_stats.get(home, {}).get("passes_attempted", 0)
    away_passes = team_stats.get(away, {}).get("passes_attempted", 0)
    total_passes = home_passes + away_passes
    home_poss = round(home_passes / total_passes * 100, 1) if total_passes > 0 else 50.0
    away_poss = round(100 - home_poss, 1)

    # Player ratings per team
    home_players_list = []
    away_players_list = []
    for pname, pdata in players.items():
        team = pdata.get("team", "")
        entry = {
            "name": pname,
            "position": pdata.get("position", ""),
            "archetype": pdata.get("archetype", ""),
            "is_starter": pdata.get("is_starter", False),
            "minutes_played": pdata.get("minutes_played", 0),
            "sub_in": pdata.get("sub_in"),
            "sub_out": pdata.get("sub_out"),
            "rating": pdata.get("rating", 6.0),
            "is_mvp": pdata.get("is_mvp", False),
            "goals": pdata.get("goals", 0),
            "assists": pdata.get("assists", 0),
            "yellow_cards": pdata.get("yellow_cards", 0),
            "red_cards": pdata.get("red_cards", 0),
            "xg": round(pdata.get("xg", 0.0), 3),
            "xa": round(pdata.get("xa", 0.0), 3),
            "passes_attempted": pdata.get("passes_attempted", 0),
            "pass_accuracy": round(pdata.get("pass_accuracy", 0.0), 1),
            "shots_on_target": pdata.get("shots_on_target", 0),
            "tackles_won": pdata.get("tackles_won", 0),
            "interceptions": pdata.get("interceptions", 0),
            "key_passes": pdata.get("key_passes", 0),
            "dribbles_comp": pdata.get("dribbles_comp", 0),
            "distance_covered": pdata.get("distance_covered", 0.0),
            "saves": pdata.get("saves"),
            "goals_conceded": pdata.get("goals_conceded"),
            "clean_sheet": pdata.get("clean_sheet", False),
        }
        if team == home:
            home_players_list.append(entry)
        elif team == away:
            away_players_list.append(entry)

    # Sort: starters first, by position order
    pos_order = {"GK": 0, "CB": 1, "LB": 2, "RB": 3, "CDM": 4, "CM": 5, "CAM": 6,
                 "LM": 6, "RM": 6, "LW": 7, "RW": 7, "CF": 8, "ST": 9}
    for lst in [home_players_list, away_players_list]:
        lst.sort(key=lambda p: (not p["is_starter"], pos_order.get(p["position"], 10)))

    # Determine result
    if home_goals_count > away_goals_count:
        home_result, away_result = "W", "L"
    elif home_goals_count < away_goals_count:
        home_result, away_result = "L", "W"
    else:
        home_result, away_result = "D", "D"

    # Find PNG assets relative to plofa_output
    assets = {}
    parent_dir = path.parent
    for png in parent_dir.glob("*.png"):
        name = png.name
        # Make path relative-ish for app
        rel = str(png).replace(str(PLOFA_DIR), "").replace("\\", "/").lstrip("/")
        if "shot_map" in name:
            assets["shot_map"] = rel
        elif "pass_network" in name:
            assets["pass_network"] = rel
        elif "pressure_map" in name:
            assets["pressure_map"] = rel
        elif "xg_timeline" in name:
            assets["xg_timeline"] = rel
        elif "summary" in name:
            assets["summary"] = rel

    return {
        "id": f"{home.replace(' ', '_')}_{away.replace(' ', '_')}_MD{info.get('matchday', 0)}",
        "matchday": info.get("matchday", 0),
        "home_team": home,
        "away_team": away,
        "score": score_str,
        "home_goals": home_goals_count,
        "away_goals": away_goals_count,
        "home_result": home_result,
        "away_result": away_result,
        "home_xg": round(info.get("home_xg", 0.0), 2),
        "away_xg": round(info.get("away_xg", 0.0), 2),
        "date": info.get("date", ""),
        "venue": info.get("venue", ""),
        "season": info.get("season", "26/27"),
        "competition": info.get("competition", "PLOFA"),
        "is_derby": info.get("is_derby", False),
        "added_time": info.get("added_time", 0),
        "home_possession": home_poss,
        "away_possession": away_poss,
        "goals": goals,
        "home_stats": team_stats.get(home, {}),
        "away_stats": team_stats.get(away, {}),
        "home_players": home_players_list,
        "away_players": away_players_list,
        "assets": assets,
    }

# â”€â”€â”€ Build League Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_league_table(all_matches):
    table = {}

    def add_team(name):
        if name not in table:
            table[name] = {
                "team": name, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                "gf": 0, "ga": 0, "gd": 0, "points": 0,
                "xgf": 0.0, "xga": 0.0,
                "cs": 0,  # clean sheets
                "form": [],  # last 5
            }

    for m in all_matches:
        home, away = m["home_team"], m["away_team"]
        add_team(home)
        add_team(away)

        hg, ag = m["home_goals"], m["away_goals"]

        # Home
        table[home]["played"] += 1
        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[home]["xgf"] += m["home_xg"]
        table[home]["xga"] += m["away_xg"]
        if ag == 0:
            table[home]["cs"] += 1
        if m["home_result"] == "W":
            table[home]["won"] += 1
            table[home]["points"] += 3
            table[home]["form"].append("W")
        elif m["home_result"] == "D":
            table[home]["drawn"] += 1
            table[home]["points"] += 1
            table[home]["form"].append("D")
        else:
            table[home]["lost"] += 1
            table[home]["form"].append("L")

        # Away
        table[away]["played"] += 1
        table[away]["gf"] += ag
        table[away]["ga"] += hg
        table[away]["xgf"] += m["away_xg"]
        table[away]["xga"] += m["home_xg"]
        if hg == 0:
            table[away]["cs"] += 1
        if m["away_result"] == "W":
            table[away]["won"] += 1
            table[away]["points"] += 3
            table[away]["form"].append("W")
        elif m["away_result"] == "D":
            table[away]["drawn"] += 1
            table[away]["points"] += 1
            table[away]["form"].append("D")
        else:
            table[away]["lost"] += 1
            table[away]["form"].append("L")

    # Sort: pts â†’ gd â†’ gf â†’ name
    sorted_table = sorted(
        table.values(),
        key=lambda t: (-t["gd"], -t["gf"], -t["points"],  t["team"])
    )

    # Add position + computed fields
    for i, row in enumerate(sorted_table):
        row["pos"] = i + 1
        row["gd"] = row["gf"] - row["ga"]
        row["xgf"] = round(row["xgf"], 2)
        row["xga"] = round(row["xga"], 2)
        row["xgd"] = round(row["xgf"] - row["xga"], 2)
        row["form"] = row["form"][-5:]  # last 5

    return sorted_table

# â”€â”€â”€ Build Players Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_players(season_stats, season_state):
    players_out = {}

    for player_name, pdata in season_stats.get("players", {}).items():
        info = pdata.get("info", {})
        totals = pdata.get("totals", {})
        per_matchday = pdata.get("per_matchday", {})
        
        state = season_state.get("players", {}).get(player_name, {})

        # Match log â€” sorted by matchday
        match_log = []
        for md_key, md_data in per_matchday.items():
            match_log.append({
                "matchday": int(md_key),
                "rating": md_data.get("rating", 6.0),
                "goals": md_data.get("goals", 0),
                "assists": md_data.get("assists", 0),
                "minutes_played": md_data.get("minutes_played", 0),
                "is_mvp": md_data.get("is_mvp", False),
                "match_result": md_data.get("match_result", ""),
                "xg": round(md_data.get("xg", 0.0), 3),
                "xa": round(md_data.get("xa", 0.0), 3),
            })
        match_log.sort(key=lambda x: x["matchday"])

        # DNA attributes
        dna = info.get("dna", {})

        players_out[player_name] = {
            "name": player_name,
            "team": info.get("team", ""),
            "position": info.get("position", ""),
            "archetype": info.get("archetype", ""),
            "age": info.get("age", 0),
            "nationality": info.get("nationality", ""),
            "preferred_foot": info.get("preferred_foot", ""),
            "specialties": info.get("specialties", ""),
            "dna": {
                "pace": round(dna.get("dna_pace", 50), 1),
                "finishing": round(dna.get("dna_finishing", 50), 1),
                "passing": round(dna.get("dna_passing", 50), 1),
                "defending": round(dna.get("dna_defending", 50), 1),
                "vision": round(dna.get("dna_vision", 50), 1),
                "composure": round(dna.get("dna_composure", 50), 1),
                "overall": round(dna.get("dna_overall", 50), 1),
            },
            "season": {
                "matches_played": totals.get("matches_played", 0),
                "minutes": totals.get("minutes", 0),
                "starts": totals.get("starts", 0),
                "goals": totals.get("goals", 0),
                "assists": totals.get("assists", 0),
                "xg": safe_round(totals.get("xg", 0.0), 3),
                "xa": safe_round(totals.get("xa", 0.0), 3),
                "npxg": safe_round(totals.get("npxg", 0.0), 3),
                "shots_on_target": totals.get("shots_on_target", 0),
                "shots": totals.get("shots", 0),
                "passes_attempted": totals.get("passes_attempted", 0),
                "passes_completed": totals.get("passes_completed", 0),
                "pass_accuracy": safe_round(totals.get("pass_accuracy", 0.0), 1),
                "short_pass_acc": safe_round(totals.get("short_pass_acc", 0.0), 1),
                "long_pass_acc": safe_round(totals.get("long_pass_acc", 0.0), 1),
                "key_passes": totals.get("key_passes", 0),
                "chances_created": totals.get("chances_created", 0),
                "big_chances_created": totals.get("big_chances_created", 0),
                "progressive_passes": totals.get("progressive_passes", 0),
                "through_balls_comp": totals.get("through_balls_comp", 0),
                "crosses_att": totals.get("crosses_att", 0),
                "crosses_comp": totals.get("crosses_comp", 0),
                "dribbles_att": totals.get("dribbles_att", 0),
                "dribbles_comp": totals.get("dribbles_comp", 0),
                "carries": totals.get("carries", 0),
                "progressive_carries": totals.get("progressive_carries", 0),
                "tackles_att": totals.get("tackles_att", 0),
                "tackles_won": totals.get("tackles_won", 0),
                "interceptions": totals.get("interceptions", 0),
                "clearances": totals.get("clearances", 0),
                "blocks": totals.get("blocks", 0),
                "aerial_duels_att": totals.get("aerial_duels_att", 0),
                "aerial_duels_won": totals.get("aerial_duels_won", 0),
                "pressures": totals.get("pressures", 0),
                "press_success": totals.get("press_success", 0),
                "fouls_committed": totals.get("fouls_committed", 0),
                "fouls_won": totals.get("fouls_won", 0),
                "yellow_cards": totals.get("yellow_cards", 0),
                "red_cards": totals.get("red_cards", 0),
                "offsides": totals.get("offsides", 0),
                "distance_covered": safe_round(totals.get("distance_covered", 0.0), 2),
                "sprints": totals.get("sprints", 0),
                "xT": safe_round(totals.get("xT", 0.0), 3),
                "gpa": safe_round(totals.get("gpa", 0.0), 3),
                "pva": safe_round(totals.get("pva", 0.0), 3),
                "saves": totals.get("saves", 0),
                "goals_conceded": totals.get("goals_conceded", 0),
                "save_pct": round(float(totals.get("save_pct") or 0.0), 1),
                "clean_sheet_pct": round(float(totals.get("clean_sheet_pct") or 0.0), 1),
                "sca": totals.get("sca", 0),
                "gca": totals.get("gca", 0),
                "touches": totals.get("touches", 0),
                "wins": totals.get("wins", 0),
                "draws": totals.get("draws", 0),
                "losses": totals.get("losses", 0),
                "avg_rating": safe_round(totals.get("avg_rating", 6.0), 2),
                # Per 90s
                "goals_per90": safe_round(totals.get("goals_per90", 0.0), 2),
                "assists_per90": safe_round(totals.get("assists_per90", 0.0), 2),
                "xg_per90": safe_round(totals.get("xg_per90", 0.0), 3),
                "xa_per90": safe_round(totals.get("xa_per90", 0.0), 3),
                "shots_on_target_per90": safe_round(totals.get("shots_on_target_per90", 0.0), 2),
                "chances_created_per90": safe_round(totals.get("chances_created_per90", 0.0), 2),
                "big_chances_created_per90": safe_round(totals.get("big_chances_created_per90", 0.0), 2),
                "tackles_won_per90": safe_round(totals.get("tackles_won_per90", 0.0), 2),
                "interceptions_per90": safe_round(totals.get("interceptions_per90", 0.0), 2),
                "clearances_per90": safe_round(totals.get("clearances_per90", 0.0), 2),
                "pressures_per90": safe_round(totals.get("pressures_per90", 0.0), 2),
                "blocks_per90": safe_round(totals.get("blocks_per90", 0.0), 2),
                "touches_per90": safe_round(totals.get("touches_per90", 0.0), 2),
                "carries_per90": safe_round(totals.get("carries_per90", 0.0), 2),
                "dribbles_comp_per90": safe_round(totals.get("dribbles_comp_per90", 0.0), 2),
                "sca_per90": safe_round(totals.get("sca_per90", 0.0), 2),
                "gca_per90": safe_round(totals.get("gca_per90", 0.0), 2),
                "turnovers_per90": safe_round(totals.get("turnovers_per90", 0.0), 2),
                "fouls_committed_per90": safe_round(totals.get("fouls_committed_per90", 0.0), 2),
                "recoveries_per90": safe_round(totals.get("recoveries_per90", 0.0), 2),
            },
            "state": {
                "is_injured": state.get("is_injured", False),
                "injury_type": state.get("injury_type", "none"),
                "matches_remaining_out": state.get("matches_remaining_out", 0),
                "yellow_cards_last_6": state.get("yellow_cards_last_6", []),
                "red_card_ban": state.get("red_card_ban", 0),
                "season_minutes": state.get("season_minutes", 0),
                "recent_ratings": state.get("recent_ratings", []),
                "injury_history": state.get("injury_history", []),
                "confidence": state.get("confidence", 50),
                "fatigue_level": state.get("fatigue_level", 0),
            },
            "match_log": match_log,
            "ratings_list": totals.get("ratings", []),
        }

    return players_out

# â”€â”€â”€ Build Teams Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_teams(all_matches, players_data):
    teams = {}

    for m in all_matches:
        for side in ["home", "away"]:
            team = m[f"{side}_team"]
            if team not in teams:
                teams[team] = {
                    "name": team,
                    "matches": [],
                }
            opp_side = "away" if side == "home" else "home"
            teams[team]["matches"].append({
                "matchday": m["matchday"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "result": m[f"{side}_result"],
                "goals_for": m[f"{side}_goals"],
                "goals_against": m[f"{opp_side}_goals"],
                "xgf": m[f"{side}_xg"],
                "xga": m[f"{opp_side}_xg"],
                "date": m["date"],
                "score": m["score"],
                "match_id": m["id"],
            })

    # Attach players to teams
    for team_name, team_data in teams.items():
        team_data["players"] = [
            {k: v for k, v in p.items() if k != "match_log"}
            for p in players_data.values()
            if p["team"] == team_name
        ]
        # Sort team matches by matchday
        team_data["matches"].sort(key=lambda x: x["matchday"])

    return teams

# â”€â”€â”€ Build Matches Index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_matches_index(all_matches):
    # Strip heavy player data for index; keep full data for detail
    index = []
    for m in all_matches:
        index.append({
            "id": m["id"],
            "matchday": m["matchday"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "score": m["score"],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "home_result": m["home_result"],
            "away_result": m["away_result"],
            "home_xg": m["home_xg"],
            "away_xg": m["away_xg"],
            "date": m["date"],
            "venue": m["venue"],
            "is_derby": m["is_derby"],
            "assets": m["assets"],
        })
    index.sort(key=lambda x: (x["matchday"], x["date"]))
    return index

# â”€â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    print("PLOFA Stats Hub â€” Data Builder")
    print("=" * 40)

    # Load source files
    print("\n[1/5] Loading season_stats.json ...")
    season_stats = load_json(SEASON_STATS_FILE)
    print(f"  Players in season_stats: {len(season_stats.get('players', {}))}")
    print(f"  Matches processed: {len(season_stats.get('matches_processed', []))}")

    print("\n[2/5] Loading season_state.json ...")
    season_state = load_json(SEASON_STATE_FILE)
    print(f"  Players in season_state: {len(season_state.get('players', {}))}")

    print("\n[3/5] Discovering and parsing match JSONs ...")
    match_files = discover_matches()
    print(f"  Found {len(match_files)} match files")

    all_matches = []
    for mf in match_files:
        try:
            parsed = parse_match(mf)
            all_matches.append(parsed)
            print(f"  [OK] {parsed['home_team']} vs {parsed['away_team']} MD{parsed['matchday']} ({parsed['score']})")
        except Exception as e:
            print(f"  [ERR] Error parsing {mf.name}: {e}")

    all_matches.sort(key=lambda m: (m["matchday"], m["date"]))
    print(f"\n  Total matches parsed: {len(all_matches)}")

    print("\n[4/5] Building output data ...")

    # League table
    league_table = build_league_table(all_matches)
    save_json({"season": "26/27", "table": league_table}, "league_table.json")  # type: ignore

    # Matches index
    matches_index = build_matches_index(all_matches)
    save_json(matches_index, "matches_index.json")

    # Full match details (one per match)
    matches_dir = OUTPUT_DIR / "matches"
    matches_dir.mkdir(exist_ok=True)
    for m in all_matches:
        save_json(m, f"matches/{m['id']}.json")

    # Players
    players_data = build_players(season_stats, season_state)
    save_json(players_data, "players.json")
    print(f"  âœ“ players.json ({len(players_data)} players)")

    # Teams
    teams_data = build_teams(all_matches, players_data)
    save_json(teams_data, "teams.json")
    print(f"  âœ“ teams.json ({len(teams_data)} teams)")

    print("\n[5/5] Done! Data written to:")
    print(f"  {OUTPUT_DIR}")
    print("\nTo serve the app:")
    print(f"  cd \"{OUTPUT_DIR.parent}\"")
    print(f"  python -m http.server 8080")
    print(f"  Open: http://localhost:8080")

if __name__ == "__main__":
    main()
