"""
PLOFA 26/27 — SEASON STATS ACCUMULATOR
========================================
season_stats.py

Scans all match output directories, reads every match.json, and accumulates
per-player stats across the entire season. Produces season_stats.json with:

    per-player cumulative totals, per-matchday breakdowns, recalculated
    percentages, per-90 rates, rating averages, and auto-generated
    leaderboards for every measurable category.

Usage:
    python season_stats.py

Or programmatically:
    from season_stats import SeasonStatsAccumulator
    acc = SeasonStatsAccumulator()
    acc.scan_outputs()
    leader = acc.leaderboard("goals", top_n=10)
    acc.save("season_stats.json")
"""
#
from __future__ import annotations
import os
import sys
import json
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, datetime


ADDITIVE_STATS = {
    "goals", "assists", "own_goals", "open_play_goals", "headed_goals",
    "open_play_assists", "setpiece_assists", "pen_goals", "pen_missed",
    "shots_on_target", "shots_off_target", "shots_blocked_att",
    "hit_woodwork", "shots_inside_box", "shots_outside_box",
    "big_chances_scored", "big_chances_missed", "big_chances_received",
    "passes_attempted", "passes_completed",
    "short_passes_att", "short_passes_comp",
    "long_passes_att", "long_passes_comp",
    "progressive_passes", "passes_own_third", "passes_mid_third",
    "passes_final_third", "passes_opp_box", "shot_assists",
    "through_balls_att", "through_balls_comp", "switches_of_play",
    "passes_under_pressure", "forward_passes", "backward_passes",
    "sideways_passes", "line_breaking_passes",
    "crosses_att", "crosses_comp",
    "crosses_open_play_att", "crosses_open_play_comp",
    "crosses_corners_att", "crosses_corners_comp",
    "crosses_box_att", "crosses_box_comp",
    "carries", "progressive_carries", "carries_own_half", "carries_opp_half",
    "final_third_carries",
    "progressive_carry_distance", "longest_progressive_carry",
    "dribbles_att", "dribbles_comp",
    "dribbles_own_half", "dribbles_mid_third", "dribbles_final_third",
    "dribbles_to_box",
    "chances_created", "big_chances_created", "open_play_cc", "setpiece_cc",
    "tackles_att", "tackles_won", "interceptions", "clearances", "blocks",
    "recoveries", "ball_recoveries", "pressures", "press_success",
    "aerial_duels_att", "aerial_duels_won",
    "ground_duels_att", "ground_duels_won",
    "last_man_tackles", "dribbled_past",
    "interceptions_def_third", "interceptions_mid_third", "interceptions_att_third",
    "recoveries_def_third", "recoveries_mid_third", "recoveries_att_third",
    "saves", "goals_conceded", "high_claims", "punches", "sweeper_actions",
    "saves_inside_box", "saves_outside_box", "goalline_saves",
    "xgot_faced", "goals_prevented",
    "fouls_committed", "fouls_won",
    "yellow_cards", "red_cards", "offsides",
    "sprints", "high_speed_sprints",
    "touches", "touches_own_third", "touches_mid_third",
    "touches_final_third", "touches_opp_box",
    "turnovers", "bad_touches", "dispossessed",
    "possession_won", "possession_lost",
    "sca", "gca", "packing_passes", "zone14_entries", "deep_completions",
    "chipped_passes", "headed_passes",
    "minutes_played",
}

SUM_ONLY_FLOAT_STATS = {
    "xg", "xa", "xg_open_play", "xg_setpiece", "xg_penalty",
    "xa_open_play", "xa_setpiece", "npxg",
    "carry_distance", "progressive_carry_distance", "dribble_distance",
    "distance_covered",
    "progressive_pass_distance", "xT", "gpa", "pva", "epa",
}

# Peak values — season aggregate is the single best (max) across matches,
# never a sum or average. e.g. top_speed = fastest km/h reached all season.
MAX_STATS = {
    "top_speed",
    "longest_progressive_carry",
}

NON_ADDITIVE_FIELDS = {
    "player", "team", "position", "archetype", "age", "nationality",
    "specialties", "preferred_foot", "is_set_piece_taker",
    "sub_in", "sub_out", "is_starter",
    "match_result", "is_mvp", "clean_sheet",
}

DNA_FIELDS = {
    "dna_overall", "dna_pace", "dna_finishing", "dna_passing",
    "dna_defending", "dna_vision", "dna_composure",
}

PERCENTAGE_DEFS = {
    "pass_accuracy":          ("passes_completed", "passes_attempted"),
    "short_pass_acc":         ("short_passes_comp", "short_passes_att"),
    "long_pass_acc":          ("long_passes_comp", "long_passes_att"),
    "dribble_success_pct":    ("dribbles_comp", "dribbles_att"),
    "tackle_success_pct":     ("tackles_won", "tackles_att"),
    "aerial_success_pct":     ("aerial_duels_won", "aerial_duels_att"),
    "ground_duels_pct":       ("ground_duels_won", "ground_duels_att"),
    "cross_acc":              ("crosses_comp", "crosses_att"),
    "press_success_pct":      ("press_success", "pressures"),
    "through_ball_acc":       ("through_balls_comp", "through_balls_att"),
}

MINIMUM_DENOMS = {
    "pass_accuracy": 10,
    "dribble_success_pct": 3,
    "tackle_success_pct": 3,
    "aerial_success_pct": 3,
    "ground_duels_pct": 3,
    "cross_acc": 3,
    "press_success_pct": 5,
    "short_pass_acc": 10,
    "long_pass_acc": 5,
    "through_ball_acc": 2,
}

PER90_STATS = {
    "goals", "assists", "shots", "shots_on_target", "shot_assists",
    "chances_created", "big_chances_created", "open_play_cc", "setpiece_cc",
    "tackles_won", "interceptions", "clearances", "blocks",
    "recoveries", "pressures",
    "carries", "progressive_carries", "progressive_carry_distance", "dribbles_comp",
    "crosses_att", "fouls_committed", "fouls_won",
    "sprints", "touches", "turnovers",
    "sca", "gca", "xg", "xa",
    "through_balls_att", "switches_of_play", "passes_completed",  "short_passes_att", "short_passes_comp",
        "long_passes_att", "long_passes_comp", "touches_opp_box", "possession_won",
    "chipped_passes", "headed_passes",
}

BOOLEAN_COUNT_FIELDS = {
    "clean_sheet": "clean_sheets",
    "is_mvp": "mvp_awards",
    "is_starter": "starts",
}

MATCH_RESULT_MAP = {
    "win": "wins",
    "loss": "losses",
    "draw": "draws",
}

ALL_LEADERBOARD_CATS = [
    ("top_scorers", "goals", "goals"),
    ("top_assisters", "assists", "assists"),
    ("top_xG", "xg", "Total xG"),
    ("top_xA", "xa", "Total xA"),
    ("most_minutes", "minutes_played", "Minutes Played"),
    ("most_chances_created", "chances_created", "Chances Created"),
    ("most_big_chances_created", "big_chances_created", "Big Chances Created"),
    ("most_shots", "shots", "Total Shots"),
    ("most_shots_on_target", "shots_on_target", "Shots on Target"),
    ("best_shot_conversion", "shot_conversion", "Shot Conversion %"),
    ("most_shot_assists", "shot_assists", "Shot Assists"),
    ("most_through_balls", "through_balls_att", "Through Balls"),
    ("most_crosses", "crosses_att", "Crosses"),
    ("best_pass_accuracy", "pass_accuracy", "Pass Accuracy %"),
    ("most_progressive_passes", "progressive_passes", "Progressive Passes"),
    ("most_switches", "switches_of_play", "Switches of Play"),
    ("most_carries", "carries", "Carries"),
    ("most_progressive_carries", "progressive_carries", "Progressive Carries"),
    ("most_carry_distance", "carry_distance", "Carry Distance"),
    ("most_dribbles", "dribbles_att", "Dribbles Attempted"),
    ("best_dribble_success", "dribble_success_pct", "Dribble Success %"),
    ("most_tackles", "tackles_won", "Tackles Won"),
    ("best_tackle_success", "tackle_success_pct", "Tackle Success %"),
    ("most_interceptions", "interceptions", "Interceptions"),
    ("most_clearances", "clearances", "Clearances"),
    ("most_blocks", "blocks", "Blocks"),
    ("most_recoveries", "recoveries", "Ball Recoveries"),
    ("most_pressures", "pressures", "Pressures"),
    ("best_press_success", "press_success_pct", "Press Success %"),
    ("most_aerial_duels", "aerial_duels_att", "Aerial Duels"),
    ("best_aerial_success", "aerial_success_pct", "Aerial Success %"),
    ("most_ground_duels", "ground_duels_att", "Ground Duels"),
    ("most_fouls_won", "fouls_won", "Fouls Won"),
    ("most_fouls_committed", "fouls_committed", "Fouls Committed"),
    ("most_yellow_cards", "yellow_cards", "Yellow Cards"),
    ("most_red_cards", "red_cards", "Red Cards"),
    ("most_offsides", "offsides", "Offsides"),
    ("most_turnovers", "turnovers", "Turnovers"),
    ("most_bad_touches", "bad_touches", "Bad Touches"),
    ("most_sprints", "sprints", "Sprints"),
    ("most_distance_covered", "distance_covered", "Distance Covered (km)"),
    ("most_progressive_carry_distance", "progressive_carry_distance", "Progressive Carry Distance"),
    ("longest_progressive_carry", "longest_progressive_carry", "Longest Progressive Carry"),
    ("highest_avg_rating", "avg_rating", "Average Rating"),
    ("most_saves", "saves", "Saves"),
    ("most_goals_conceded", "goals_conceded", "Goals Conceded"),
    ("most_goals_prevented", "goals_prevented", "Goals Prevented"),
    ("most_clean_sheets", "clean_sheets", "Clean Sheets"),
    ("best_save_pct", "save_pct", "Save %"),
    ("most_high_claims", "high_claims", "High Claims"),
    ("most_punches", "punches", "Punches"),
    ("most_xgot_faced", "xgot_faced", "xGOT Faced"),
    ("most_mvp_awards", "mvp_awards", "MVP Awards"),
    ("most_touches", "touches", "Touches"),
    ("most_sca", "sca", "Shot-Creating Actions"),
    ("most_gca", "gca", "Goal-Creating Actions"),
    ("most_deep_completions", "deep_completions", "Deep Completions"),
    ("most_zone14_entries", "zone14_entries", "Zone 14 Entries"),
    ("most_packing_passes", "packing_passes", "Packing Passes"),
    ("most_line_breaking_passes", "line_breaking_passes", "Line-Breaking Passes"),
    ("most_chipped_passes", "chipped_passes", "Chipped Passes"),
    ("most_headed_passes", "headed_passes", "Headed Passes"),
    ("most_possession_won", "possession_won", "Possession Won"),
    ("most_possession_lost", "possession_lost", "Possession Lost"),
    ("most_dribbled_past", "dribbled_past", "Dribbled Past"),
    ("most_last_man_tackles", "last_man_tackles", "Last-Man Tackles"),
    ("most_head_goals", "headed_goals", "Headed Goals"),
    ("most_big_chances_missed", "big_chances_missed", "Big Chances Missed"),
    ("best_xG_per_shot", "xg_per_shot", "xG per Shot"),
    ("most_big_chances_received", "big_chances_received", "Big Chances Received"),
    ("most_woodwork", "hit_woodwork", "Hit Woodwork"),
    ("most_pen_goals", "pen_goals", "Penalty Goals"),
    ("most_pen_missed", "pen_missed", "Penalties Missed"),
]


class SeasonStatsAccumulator:

    def __init__(self, output_dirs: Optional[List[str]] = None):
        self.output_dirs = output_dirs or ["plofa_output", "outputs"]
        self.players: Dict[str, Dict[str, Any]] = {}
        self.matches_processed: List[Dict[str, Any]] = []
        self.season = "26/27"
        self._match_keys_seen: set = set()

    # ── PUBLIC API ────────────────────────────────

    def scan_outputs(self):
        for d in self.output_dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                full = os.path.join(d, entry)
                if not os.path.isdir(full):
                    continue
                json_path = self._find_match_json(full)
                if json_path:
                    self._process_match_json(json_path)

    def leaderboard(self, stat_key: str, top_n: int = 20) -> List[Dict[str, Any]]:
        board = []
        for name, pdata in self.players.items():
            val = self._resolve_stat(pdata, stat_key)
            if val is not None:
                board.append({
                    "player": name,
                    "team": pdata["info"].get("team", "?"),
                    "position": pdata["info"].get("position", "?"),
                    "matches": pdata["totals"].get("matches_played", 0),
                    "minutes": pdata["totals"].get("minutes", 0),
                    "value": round(val, 2) if isinstance(val, float) else val,
                })
        board.sort(key=lambda r: r["value"], reverse=True)
        return board[:top_n]

    def get_player(self, name: str) -> Optional[Dict[str, Any]]:
        return self.players.get(name)

    def to_dict(self) -> Dict[str, Any]:
        players_clean = {}
        for name, pdata in self.players.items():
            totals = dict(pdata["totals"])
            totals.pop("ratings_list", None)
            if "ratings" in totals:
                totals["ratings"] = list(totals["ratings"])
            players_clean[name] = {
                "info": pdata["info"],
                "totals": totals,
                "per_matchday": pdata["per_matchday"],
            }
        return {
            "season": self.season,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "matches_processed": self.matches_processed,
            "players": players_clean,
            "leaderboards": self._build_all_leaderboards(),
        }

    def save(self, path: str = "season_stats.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str = "season_stats.json") -> "SeasonStatsAccumulator":
        acc = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        acc.season = data.get("season", "26/27")
        acc.players = data.get("players", {})
        acc.matches_processed = data.get("matches_processed", [])
        return acc

    # ── INTERNALS ─────────────────────────────────

    @staticmethod
    def _find_match_json(dir_path: str) -> Optional[str]:
        for fname in os.listdir(dir_path):
            if fname.endswith(".json"):
                full = os.path.join(dir_path, fname)
                try:
                    with open(full, encoding="utf-8") as f:
                        data = json.load(f)
                    if "match" in data and "players" in data:
                        return full
                except (json.JSONDecodeError, IOError):
                    continue
        return None

    def _process_match_json(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        match_info = data.get("match", {})
        md = match_info.get("matchday", 0)
        home = match_info.get("home_team", "?")
        away = match_info.get("away_team", "?")
        score = match_info.get("score", "?-?")
        season = match_info.get("season", self.season)

        key = (home, away, md)
        if key in self._match_keys_seen:
            return
        self._match_keys_seen.add(key)

        self.season = season

        match_record = {
            "matchday": md,
            "home": home,
            "away": away,
            "score": score,
            "file": os.path.relpath(path),
            "date": match_info.get("date", ""),
        }
        self.matches_processed.append(match_record)

        players_data = data.get("players", {})
        for pname, pstats in players_data.items():
            self._accumulate_player(pname, pstats, md)

    def _accumulate_player(self, name: str, stats: dict, md: int):
        if name not in self.players:
            self.players[name] = self._fresh_player_entry(stats)

        pdata = self.players[name]

        md_str = str(md)
        pdata["per_matchday"][md_str] = dict(stats)

        total = pdata["totals"]
        for field in ADDITIVE_STATS:
            val = stats.get(field, 0)
            if isinstance(val, (int, float)):
                total[field] = total.get(field, 0) + val

        for field in SUM_ONLY_FLOAT_STATS:
            val = stats.get(field, 0.0)
            if isinstance(val, (int, float)):
                total[field] = total.get(field, 0.0) + val

        for field in MAX_STATS:
            val = stats.get(field)
            if isinstance(val, (int, float)):
                total[field] = max(total.get(field, 0.0), val)

        for field, target_key in BOOLEAN_COUNT_FIELDS.items():
            if stats.get(field, False):
                total[target_key] = total.get(target_key, 0) + 1

        mr = stats.get("match_result", "")
        if mr in MATCH_RESULT_MAP:
            total[MATCH_RESULT_MAP[mr]] = total.get(MATCH_RESULT_MAP[mr], 0) + 1

        rating = stats.get("rating")
        if rating is not None:
            total["ratings_list"].append(rating)

        dna = pdata["info"]["dna"]
        for field in DNA_FIELDS:
            val = stats.get(field)
            if val is not None:
                dna[field] = val

        pdata["info"]["team"] = stats.get("team", pdata["info"]["team"])
        pdata["info"]["position"] = stats.get("position", pdata["info"]["position"])

        total["matches_played"] = len(pdata["per_matchday"])
        total["minutes"] = total.get("minutes_played", 0)
        total["sub_appearances"] = sum(
            1 for pm in pdata["per_matchday"].values()
            if pm.get("is_starter") is False
        )
        total["starts"] = total.get("starts", total["matches_played"] - total["sub_appearances"])

        self._recalc_derived(pdata)

    def _recalc_derived(self, pdata: dict):
        total = pdata["totals"]
        mins = total.get("minutes", 0) or 1

        for pct_name, (num_field, den_field) in PERCENTAGE_DEFS.items():
            num = total.get(num_field, 0)
            den = total.get(den_field, 0)
            min_den = MINIMUM_DENOMS.get(pct_name, 1)
            if den >= min_den:
                total[pct_name] = round((num / den) * 100, 1)
            else:
                total[pct_name] = None

        total_shots = (total.get("shots_on_target", 0)
                       + total.get("shots_off_target", 0)
                       + total.get("shots_blocked_att", 0))
        total["shots"] = total_shots
        if total_shots > 0:
            total["shot_conversion"] = round((total.get("goals", 0) / total_shots) * 100, 1)
            total["xg_per_shot"] = round(total.get("xg", 0.0) / total_shots, 3)
        else:
            total["shot_conversion"] = None
            total["xg_per_shot"] = None

        sv = total.get("saves", 0)
        gc = total.get("goals_conceded", 0)
        if sv + gc > 0:
            total["save_pct"] = round((sv / (sv + gc)) * 100, 1)
        else:
            total["save_pct"] = None

        total["clean_sheet_pct"] = (
            round((total.get("clean_sheets", 0) / max(total["matches_played"], 1)) * 100, 1)
        )

        for_p90 = PER90_STATS & set(total.keys())
        for stat in for_p90:
            val = total.get(stat, 0)
            if isinstance(val, (int, float)):
                total[f"{stat}_per90"] = round((val / mins) * 90, 2)

        ratings = total.get("ratings_list", [])
        if ratings:
            total["avg_rating"] = round(sum(ratings) / len(ratings), 2)
            total["ratings"] = ratings
        else:
            total["avg_rating"] = None
            total["ratings"] = []

    def _resolve_stat(self, pdata: dict, stat_key: str):
        total = pdata["totals"]
        if stat_key in total:
            v = total[stat_key]
            if v is None:
                return None
            return v
        if stat_key in pdata["info"]:
            return pdata["info"][stat_key]
        return None

    def _build_all_leaderboards(self) -> Dict[str, list]:
        boards = {}
        for board_key, stat_key, _ in ALL_LEADERBOARD_CATS:
            boards[board_key] = self.leaderboard(stat_key, top_n=20)
        return boards

    @staticmethod
    def _fresh_player_entry(stats: dict) -> dict:
        return {
            "info": {
                "team": stats.get("team", "?"),
                "position": stats.get("position", "?"),
                "archetype": stats.get("archetype", "?"),
                "age": stats.get("age"),
                "nationality": stats.get("nationality", "?"),
                "preferred_foot": stats.get("preferred_foot", "?"),
                "specialties": stats.get("specialties", ""),
                "dna": {},
            },
            "totals": {
                "matches_played": 0,
                "minutes": 0,
                "starts": 0,
                "sub_appearances": 0,
                "ratings_list": [],
            },
            "per_matchday": {},
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    acc = SeasonStatsAccumulator()
    acc.scan_outputs()
    acc.save()
    mds = sorted(set(m["matchday"] for m in acc.matches_processed))
    print(f"Processed {len(acc.matches_processed)} matches (MDs {mds})")
    print(f"Players tracked: {len(acc.players)}")
    top5 = acc.leaderboard("goals", top_n=5)
    print("\nTop Scorers:")
    for r in top5:
        print(f"  {r['player']:25s}  {r['value']:>2d} goals  ({r['team']})")
    top5a = acc.leaderboard("assists", top_n=5)
    print("\nTop Assisters:")
    for r in top5a:
        print(f"  {r['player']:25s}  {r['value']:>2d} assists  ({r['team']})")
    top5c = acc.leaderboard("chances_created", top_n=5)
    print("\nMost Chances Created:")
    for r in top5c:
        print(f"  {r['player']:25s}  {r['value']:>2d} chances  ({r['team']})")
    top5r = acc.leaderboard("avg_rating", top_n=5)
    print("\nHighest Avg Rating:")
    for r in top5r:
        print(f"  {r['player']:25s}  {r['value']:.2f} avg  ({r['team']}, {r['matches']} apps)")
    print(f"\nFull data written to season_stats.json")


if __name__ == "__main__":
    main()
