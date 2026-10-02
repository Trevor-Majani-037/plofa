"""Reconstruct the missing raw RW vs AZ MD05 engine JSON from canonical sources.

Sources:
  season_stats.json per_matchday['5']  -> full stat payloads for all participants
  app export (adventurous-mendel)      -> score, goals, xG, bench identity+rating
NOT re-simulated. Deterministic. Unrecoverable fields (financials, timeline,
possession clock seconds) are left empty / set to reproduce stored pct.

See MD5_RECOVERY_RUNBOOK.md for the full incident record and procedure.
"""
import json, sys, io, os, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = r"C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa"
APP = r"C:\Users\Trevor Majani\Documents\antigravity\adventurous-mendel\data\matches\Red_Wolves_Avada_Zenith_MD5.json"
OUTDIR = os.path.join(REPO, "plofa_output", "Red_Wolves_vs_Avada_Zenith_MD05")
OUT = os.path.join(OUTDIR, "Red_Wolves_vs_Avada_Zenith_MD5.json")
TEMPLATE = os.path.join(REPO, "plofa_output", "Ganester_vs_Pearls_MD05", "Ganester_vs_Pearls_MD5.json")

HOME, AWAY = "Red Wolves", "Avada Zenith"
DROP_KEYS = {"fantasy_assists", "goal_assists", "open_play_shot_assists",
             "second_assists", "setpiece_shot_assists", "shots_faced_by_creation"}

app = json.load(open(APP, encoding="utf-8"))
ss = json.load(open(os.path.join(REPO, "season_stats.json"), encoding="utf-8"))
tpl = json.load(open(TEMPLATE, encoding="utf-8"))
raw_player_keys = list(tpl["players"][next(iter(tpl["players"]))].keys())


def empty_like(v):
    if isinstance(v, dict):
        return {}
    if isinstance(v, list):
        return []
    return None


# Engine (app export) wins over the season_stats accumulator for these fields.
# Proven divergence: MD5 corner assist credited to Julian Lingl by the engine,
# but the accumulator stored it on Benardo Zico (assists 0/1 vs 1/0).
OVERRIDE_FIELDS = ("assists", "open_play_assists", "setpiece_assists",
                   "open_play_cc", "setpiece_cc")
exp_by_name = {}
for lst in (app["home_players"], app["away_players"]):
    for e in lst:
        exp_by_name[e["name"]] = e

players = {}
n_pm, n_synth, n_fix = 0, 0, 0
for side, lst, team, hoa, result in (
        ("home", app["home_players"], HOME, "home", "win"),
        ("away", app["away_players"], AWAY, "away", "loss")):
    for e in lst:
        name = e["name"]
        pm = (ss["players"].get(name, {}).get("per_matchday", {}) or {}).get("5")
        if pm:
            p = {k: copy.deepcopy(v) for k, v in pm.items() if k not in DROP_KEYS}
            p.setdefault("save_pct", None)
            for f in OVERRIDE_FIELDS:
                if f in p and e.get(f) is not None and p[f] != e[f]:
                    n_fix += 1
                    p[f] = e[f]
            # keep raw key order, append extras at end
            ordered = {k: p.get(k) for k in raw_player_keys}
            extra = {k: v for k, v in p.items() if k not in ordered}
            ordered.update(extra)
            players[name] = ordered
            n_pm += 1
        else:
            info = (ss["players"].get(name, {}).get("info")) or {}
            dna = info.get("dna") or {}
            p = {k: None for k in raw_player_keys}
            for k in raw_player_keys:
                if isinstance(tpl["players"][next(iter(tpl["players"]))][k], bool):
                    p[k] = False
                elif isinstance(tpl["players"][next(iter(tpl["players"]))][k], (int, float)):
                    p[k] = 0
                elif isinstance(tpl["players"][next(iter(tpl["players"]))][k], list):
                    p[k] = []
                elif isinstance(tpl["players"][next(iter(tpl["players"]))][k], dict):
                    p[k] = {}
                else:
                    p[k] = None
            p.update({
                "player": name, "team": team, "position": e.get("position") or info.get("position"),
                "archetype": e.get("archetype") or info.get("archetype"),
                "age": info.get("age"), "nationality": info.get("nationality"),
                "preferred_foot": info.get("preferred_foot"),
                "specialties": info.get("specialties"),
                "is_starter": False, "minutes_played": 0, "sub_in": None, "sub_out": None,
                "is_set_piece_taker": False, "soul_archetype": "", "soul_tier": "",
                "soul_greatness": 0.0, "soul_omega": False,
                "home_or_away": hoa, "venue": "Unity Stadium",
                "match_result": result, "is_mvp": False,
                "rating": e.get("rating", 6.0),
                "goals": 0, "assists": 0, "yellow_cards": 0, "red_cards": 0,
                "saves": 0, "goals_conceded": 0, "clean_sheet": False, "save_pct": 0.0 if (e.get("position") == "GK") else None,
                "dna_pace": (dna.get("pace") if isinstance(dna, dict) else None),
                "dna_passing": (dna.get("passing") if isinstance(dna, dict) else None),
                "dna_defending": (dna.get("defending") if isinstance(dna, dict) else None),
                "dna_finishing": (dna.get("finishing") if isinstance(dna, dict) else None),
                "dna_vision": (dna.get("vision") if isinstance(dna, dict) else None),
                "dna_composure": (dna.get("composure") if isinstance(dna, dict) else None),
                "dna_overall": (dna.get("overall") if isinstance(dna, dict) else None),
            })
            players[name] = p
            n_synth += 1

doc = {
    "match": {
        "home_team": HOME, "away_team": AWAY, "score": "2\u20131",
        "home_xg": app["home_xg"], "away_xg": app["away_xg"],
        "matchday": 5, "season": "26/27", "competition": "PLOFA",
        "venue": "Unity Stadium", "date": "2026-09-19",
        "added_time": app["added_time"], "is_derby": False,
        "home_possession_pct": app["home_possession"], "away_possession_pct": app["away_possession"],
        "home_possession_s": 518.0, "away_possession_s": 482.0,
    },
    "goals": app["goals"],
    "players": players,
    "financials": {},
    "timeline": [],
    "sequences": empty_like(tpl.get("sequences")),
    "opta": empty_like(tpl.get("opta")),
    "opta_stats": empty_like(tpl.get("opta_stats")),
    "pressing": empty_like(tpl.get("pressing")),
    "defensive_awareness": empty_like(tpl.get("defensive_awareness")),
    "third_progression": empty_like(tpl.get("third_progression")),
}

os.makedirs(OUTDIR, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print("wrote:", OUT)
print("players:", len(players), "from per_matchday:", n_pm, "synthesized bench:", n_synth)
print("assist overrides applied:", n_fix)
print("goals:", len(doc["goals"]), "score:", doc["match"]["score"])
# sanity: key parity vs template for a participant
sample = players.get("Nathan Opaz", {})
missing = [k for k in raw_player_keys if k not in sample]
print("participant key gaps vs raw template:", missing if missing else "none")
