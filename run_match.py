"""
╔══════════════════════════════════════════════════════════════════════╗
║           PLOFA 26/27 — MATCH RUNNER                                ║
║           run_match.py                                               ║
║                                                                      ║
║  THIS IS THE ONLY FILE YOU EDIT WEEK TO WEEK.                       ║
║                                                                      ║
║  To run a match:                                                     ║
║    1. Fill in MATCH INFO (date, matchday, teams, venue)              ║
║    2. Fill in HOME TEAM squad (starters + subs)                      ║
║    3. Fill in AWAY TEAM squad (starters + subs)                      ║
║    4. Set team styles                                                ║
║    5. Mark any soul players (Percy etc.)                             ║
║    6. Run: python run_match.py                                       ║
║                                                                      ║
║  Output goes to: outputs/<HomeTeam_vs_AwayTeam_MD##>/               ║
║    → match.xlsx  (8 sheets, full stats)                              ║
║    → players.csv                                                     ║
║    → match.json                                                      ║
║    → shot_map.png                                                    ║
║    → pass_network.png                                                ║
║    → xg_timeline.png                                                 ║
║    → match_summary.png                                               ║
║    → pressure_map.png                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""
#USE auto_run_match.py for running matches automatically with default settings. This file is for manual match runs with custom squads and settings.
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from player_soul import PlayerSoul, SoulArchetype, GreatnessPillars, SoulApplicator
from exporter import PLOFAExporter
from squad_manager import SubstitutionController, AvailabilityChecker
from roster_loader import get_loader
from season_manager import SeasonState


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — SOUL PLAYERS
# Define once per season. Only edit when a new soul player is added.
# These are applied automatically when their name appears in a squad.
# ══════════════════════════════════════════════════════════════════════

SOUL_PLAYERS = {

    # ── Percy — Hartwell City RW — ATTACKING PROPHET ──────────
    "Percy": PlayerSoul(
        player_name="Percy",
        archetype=SoulArchetype.ATTACKING_PROPHET,
        pillars=GreatnessPillars(
            hardwork=0.97,   # Relentless. Never stops.
            talent=0.99,     # Absolute ceiling.
            luck=0.91,       # Right era, right club, no major injuries.
        )
    ),


    # ── Add other soul players below as the season progresses ──
    "Luca Ferrini": PlayerSoul(
        player_name="Luca Ferrini",
        archetype=SoulArchetype.MIDFIELD_PHILOSOPHER,
        pillars=GreatnessPillars(hardwork=0.91, talent=0.93, luck=0.82)
    ),
    #
    "Kwame Asante": PlayerSoul(
        player_name="Kwame Asante",
        archetype=SoulArchetype.CREATIVE_ORACLE,
        pillars=GreatnessPillars(hardwork=0.88, talent=0.90, luck=0.78)
    ),

}


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — MATCH INFO
# Edit these every matchday
# ══════════════════════════════════════════════════════════════════════

MATCH_DATE   = date(2026, 8, 16)    # Year, Month, Day
MATCHDAY     = 1                    # 1–34
SEASON       = "26/27"
COMPETITION  = "PLOFA"

HOME_TEAM    = "Hartwell City"
AWAY_TEAM    = "Thornfield United"

VENUE        = "Hartwell Arena"
CAPACITY     = 42_000

REFEREE      = "Marcus Osei"
STRICTNESS   = 0.55    # 0.0 = lenient, 1.0 = very strict

IS_DERBY     = False   # True if local rivalry match

# Team colors for visualizations (hex)
HOME_COLOR   = "#003087"
AWAY_COLOR   = "#C8102E"


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — TEAM STYLES
# Set once per match. Change if team's approach changes.
#
# STYLES:
#   ultra_attacking, attacking, balanced, defensive, ultra_defensive
#   gegenpressing, tiki_taka, park_the_bus, route_one, wing_play
#   vertical_tiki_taka, fluid_counter, structured_possession
#
# PLAYING STYLES:
#   possession, counter, mixed, direct, patient_build_up
#   high_press, low_block, transition_focused
#
# INTENSITY:
#   LOW, MEDIUM, HIGH, VERY_HIGH
# ══════════════════════════════════════════════════════════════════════

HOME_STYLE = TeamProfile(
    name=HOME_TEAM,
    style=TeamStyle.ATTACKING,
    playing_style=PlayingStyle.HIGH_PRESS,
    intensity=Intensity.HIGH,
)

AWAY_STYLE = TeamProfile(
    name=AWAY_TEAM,
    style=TeamStyle.FLUID_COUNTER,
    playing_style=PlayingStyle.COUNTER,
    intensity=Intensity.MEDIUM,
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — SQUADS
#
# FORMAT (starters):
#   ("Player Name", "POSITION", ["specialty1", "specialty2"], AGE)
#
# FORMAT (substitutes):
#   ("Player Name", "POSITION", ["specialty1"], AGE, SUB_IN_MINUTE)
#   (SUB_IN_MINUTE is optional — leave out if unknown before the match)
#
# POSITIONS: GK, CB, LB, RB, CDM, CM, CAM, LW, RW, ST, CF
#
# KEY SPECIALTIES (from player_dna.py — use as many as fit):
#   GK:  sweeper_keeper, shot_stopper, distribution_gk
#   CB:  ball_playing_cb, stopper_defender, no_nonsense_cb, sweeper_cb
#   FB:  aggressive_fullback, overlapping_fullback, defensive_fullback
#   CDM: anchor_man, ball_winner, regista, interceptor
#   CM:  box_box, engine, playmaker, deep_playmaker, progressive_midfielder
#   CAM: creator, grand_creator, sup_vision, shadow_striker
#   W:   dribbler, grand_dribbler, speedster, inverted, crosser, pressing_forward
#   ST:  clinical_finisher, fox_in_box, target_man, aerial_threat, poacher
#   ALL: strong, two_footed, captain, set_piece_specialist, dirty_player,
#        press_resistant, workhorse, cold_blooded, clutch, big_game_player
#
# SUPERSTARS: Add name to team_superstars list (10-20% attribute boost)
# SET PIECE TAKERS: Add name to set_piece_takers list
# ══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# HOME TEAM: Hartwell City
# ─────────────────────────────────────────────

HOME_STARTERS = [
    ("Keano Walsh",   "GK",  ["sweeper_keeper"],                    29),
    ("Darius Frost",  "LB",  ["aggressive_fullback", "engine"],     24),
    ("Emeka Obi",     "CB",  ["ball_playing_cb"],                   27),
    ("Tavish Crane",  "CB",  ["stopper_defender", "strong"],        30),
    ("Rico Alves",    "RB",  ["overlapping_fullback"],              25),
    ("Mateo Sanz",    "CDM", ["anchor_man", "interceptor"],         28),
    ("Luca Ferrini",  "CM",  ["box_box", "engine"],                 26),
    ("Kofi Mensah",   "CAM", ["creator", "sup_vision"],             24),
    ("Adri Vela",     "LW",  ["dribbler", "speedster"],             22),
    ("Dragan Novak",  "ST",  ["clinical_finisher", "aerial_threat"],29),
    ("Percy",         "RW",  ["grand_dribbler", "inverted",
                               "clinical_finisher", "speedster"],   24),  # ← SOUL PLAYER
]

HOME_SUBS = [
    ("Joel Trent",    "GK",  [],                                    26),
    ("Sam Boateng",   "CB",  ["stopper_defender"],                  28),
    ("Oscar Muñoz",   "CM",  ["playmaker", "press_resistant"],      25),
    ("Eli Dago",      "LW",  ["speedster", "dribbler"],             21, 68),  # sub at 68'
    ("Calvin Pryce",  "ST",  ["poacher", "fox_in_box"],             27, 75),  # sub at 75'
]

HOME_SUPERSTARS    = ["Percy", "Dragan Novak"]
HOME_SP_TAKERS     = ["Percy", "Kofi Mensah"]   # set piece takers


# ─────────────────────────────────────────────
# AWAY TEAM: Thornfield United
# ─────────────────────────────────────────────

AWAY_STARTERS = [
    ("Pavel Renko",   "GK",  ["sweeper_keeper"],                    31),
    ("Jide Afolabi",  "LB",  [],                                    26),
    ("Bart Kuipers",  "CB",  ["stopper_defender"],                  28),
    ("Ciro Mancini",  "CB",  ["ball_playing_cb"],                   26),
    ("Lee Sung-jin",  "RB",  ["overlapping_fullback"],              28),
    ("Demi Adeola",   "CDM", ["ball_winner", "regista"],            27),
    ("Finn Larsson",  "CM",  ["press_resistant", "engine"],         25),
    ("Kwame Asante",  "CAM", ["playmaker", "creator"],              23),
    ("Bruno Reis",    "LW",  ["speedster", "counter_attacker"],     24),
    ("Nico Strauss",  "ST",  ["fox_in_box", "cold_blooded"],        27),
    ("Tariq El-Amin", "RW",  ["dribbler", "inverted"],              22),
]

AWAY_SUBS = [
    ("Martin Horak",  "GK",  [],                                    29),
    ("Danny Cross",   "CB",  [],                                    24),
    ("Isaac Bello",   "CM",  ["box_box"],                           23, 72),
    ("Rafiq Nasser",  "ST",  ["poacher", "super_sub"],              26, 80),
    ("Olu Adeyemi",   "RW",  ["speedster"],                         21, 85),
]

AWAY_SUPERSTARS    = ["Kwame Asante"]
AWAY_SP_TAKERS     = ["Kwame Asante", "Bruno Reis"]


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — SQUAD MANAGER SETTINGS
# Controls substitution behaviour and stamina tracking
# ══════════════════════════════════════════════════════════════════════

# How stubborn is the manager about keeping tired players on?
# 0.0 = subs immediately when player is drained
# 0.5 = balanced (recommended)
# 1.0 = never subs for stamina (old school)
MANAGER_STUBBORNNESS = 0.35

# Max substitutions allowed (PLOFA standard = 3)
MAX_SUBS = 3

# Check availability from previous matches?
# Set to True after Matchday 1 when output files exist
CHECK_AVAILABILITY = False
OUTPUTS_DIR = "plofa_output"   # Where your match output folders live

# ── Season state file (updated after every match) ──────────────
SEASON_STATE_FILE = "season_state.json"


# ══════════════════════════════════════════════════════════════════════
# ENGINE — Do not edit below this line
# Everything below runs automatically from your inputs above
# ══════════════════════════════════════════════════════════════════════

def run():
    import os

    print(f"\n{'═'*64}")
    print(f"  PLOFA {SEASON} — Matchday {MATCHDAY}")
    print(f"  {HOME_TEAM} vs {AWAY_TEAM}")
    print(f"  {MATCH_DATE.strftime('%A %d %B %Y')} | {VENUE}")
    print(f"  Referee: {REFEREE} (strictness: {STRICTNESS})")
    print(f"{'═'*64}\n")

    # ── Build squads ─────────────────────────────────────────
    home_squad = SquadBuilder.build(
        team_name=HOME_TEAM,
        starters=HOME_STARTERS,
        substitutes=HOME_SUBS,
        team_superstars=HOME_SUPERSTARS,
        set_piece_takers=HOME_SP_TAKERS,
    )

    away_squad = SquadBuilder.build(
        team_name=AWAY_TEAM,
        starters=AWAY_STARTERS,
        substitutes=AWAY_SUBS,
        team_superstars=AWAY_SUPERSTARS,
        set_piece_takers=AWAY_SP_TAKERS,
    )

    # ── Attach souls to players ──────────────────────────────
    all_players_flat = (
        home_squad["starters"] + home_squad["substitutes"] +
        away_squad["starters"] + away_squad["substitutes"]
    )
    souls_attached = []
    for player in all_players_flat:
        if player.name in SOUL_PLAYERS:
            player.dna.soul = SOUL_PLAYERS[player.name]
            souls_attached.append(player.name)

    if souls_attached:
        print(f"  🔮 Soul players active: {', '.join(souls_attached)}")
        for name in souls_attached:
            soul = SOUL_PLAYERS[name]
            print(f"     {name}: {soul.profile.label} | "
                  f"G={soul.greatness_coefficient:.4f} | "
                  f"Tier={soul.tier} | "
                  f"{'⚡ OMEGA ACTIVE' if soul.pillars.omega_activated else 'No Omega'}")
        print()

    # ── Match config ──────────────────────────────────────────
    config = MatchConfig(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        match_date=MATCH_DATE,
        matchday=MATCHDAY,
        season=SEASON,
        competition=COMPETITION,
        venue=VENUE,
        stadium_capacity=CAPACITY,
        referee=REFEREE,
        referee_strictness=STRICTNESS,
        is_derby=IS_DERBY,
    )

    # ── Availability check ────────────────────────────────────
    if CHECK_AVAILABILITY:
        print("  📋 Checking player availability from previous matches...")
        checker = AvailabilityChecker(OUTPUTS_DIR)
        for team_name in [HOME_TEAM, AWAY_TEAM]:
            availability = checker.check(team_name, MATCHDAY)
            if availability:
                checker.print_report(team_name, availability)
        print()

    # ── Build substitution controller ─────────────────────────
    # Build tactical schedule from pre-planned sub minutes
    tactical_schedule = {}
    for p in home_squad["starters"] + away_squad["starters"]:
        # If a starter has a sub_out_minute defined, schedule it
        if getattr(p, "sub_out_minute", None):
            tactical_schedule[p.name] = p.sub_out_minute

    # Also capture from subs' sub_in_minute
    for p in home_squad["substitutes"] + away_squad["substitutes"]:
        sm = getattr(p, "sub_in_minute", None)
        if sm is None and hasattr(p, "dna"):
            sm = getattr(p.dna, "sub_in_minute", None)
        # Find the player they're replacing (same position group)
        # This is approximate — the controller handles exact matching
        if sm:
            pass  # Controller will match by position at the designated minute

    sub_controller = SubstitutionController(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=HOME_STYLE.style.value,
        away_style=AWAY_STYLE.style.value,
        manager_stubbornness=MANAGER_STUBBORNNESS,
    )
    sub_controller.MAX_SUBS = MAX_SUBS

    # Register pre-planned tactical sub minutes.
    # Format: {player_OFF_name: minute_they_come_off}
    # Built from starters' sub_out_minute and from bench sub_in_minute
    # (inferring the player going off by finding a starter in the same
    # position group who doesn't already have a sub scheduled).
    tactical_schedule = {}
    for p in home_squad["starters"] + away_squad["starters"]:
        if getattr(p, "sub_out_minute", None):
            tactical_schedule[p.name] = p.sub_out_minute
    for bench, starters in [
        (home_squad["substitutes"], home_squad["starters"]),
        (away_squad["substitutes"], away_squad["starters"]),
    ]:
        for sub_p in bench:
            sm = getattr(sub_p, "sub_in_minute", None)
            if sm is None and hasattr(sub_p, "dna"):
                sm = getattr(sub_p.dna, "sub_in_minute", None)
            if not sm:
                continue
            sub_pos = getattr(sub_p, "position",
                              getattr(getattr(sub_p, "dna", None), "position", "CM"))
            adj = {
                "ST": ["CF", "LW", "RW", "CAM"], "CF": ["ST", "CAM", "LW", "RW"],
                "LW": ["RW", "CAM", "ST", "LB"], "RW": ["LW", "CAM", "ST", "RB"],
                "CAM": ["CM", "LW", "RW", "ST"], "CM": ["CAM", "CDM", "LW", "RW"],
                "CDM": ["CM", "CB"], "LB": ["RB", "CB", "LW"],
                "RB": ["LB", "CB", "RW"], "CB": ["CDM", "LB", "RB"], "GK": ["GK"],
            }.get(sub_pos, [])
            candidates = [
                s for s in starters
                if getattr(s, "position",
                           getattr(getattr(s, "dna", None), "position", "CM")) in ([sub_pos] + adj)
                and s.name not in tactical_schedule
            ]
            if candidates:
                tactical_schedule[candidates[0].name] = sm
    sub_controller.register_tactical_schedule(tactical_schedule)

    # ── Run simulation ────────────────────────────────────────
    engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
    engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_controller)

    result = engine.simulate()
    print(result.summary())

    # ── Export ────────────────────────────────────────────────
    all_players = {HOME_TEAM: home_squad, AWAY_TEAM: away_squad}

    folder_name = (
        f"{HOME_TEAM.replace(' ', '_')}_vs_"
        f"{AWAY_TEAM.replace(' ', '_')}_MD{MATCHDAY:02d}"
    )
    output_path = os.path.join("plofa_output", folder_name)

    # ── Resolve colors (ensure home != away) ────────────────────
    home_color = HOME_COLOR
    away_color = AWAY_COLOR

    if away_color == home_color:
        fallbacks = [
            "#C8102E", "#FFFFFF", "#00B4D8", "#F5C518",
            "#2DC653", "#E63946", "#B388FF", "#FF6B6B",
        ]
        for fb in fallbacks:
            if fb != home_color:
                away_color = fb
                break

    exporter = PLOFAExporter(
        result=result,
        all_players=all_players,
        home_color=home_color,
        away_color=away_color,
        sub_controller=sub_controller,
    )
    exporter.export_all(output_path)

    # ── Soul player match report ──────────────────────────────
    if souls_attached:
        print(f"\n  🔮 SOUL PLAYER MATCH REPORT")
        print(f"  {'─'*40}")
        acc = exporter.accumulator
        for name in souls_attached:
            s = acc.stats.get(name)
            if s:
                soul = SOUL_PLAYERS[name]
                bonuses = soul.get_bonus_stats()
                print(f"  {name} ({soul.profile.label})")
                print(f"    Goals: {s['goals']}  Assists: {s['assists']}  "
                      f"Rating: {s['rating']}")
                print(f"    xG: {s['xg']:.3f}  xA: {s['xa']:.3f}")
                print(f"    Dribbles: {s['dribbles_comp']}/{s['dribbles_att']}  "
                      f"Shot Assists: {s['shot_assists']}  "
                      f"Carries: {s['carries']}")
                if bonuses:
                    print(f"    Soul bonuses applied: "
                          f"{', '.join(f'+{v} {k}' for k,v in bonuses.items())}")

    # ── Persist season state ──────────────────────────────────
    try:
        loader = get_loader()
        season_state = SeasonState(SEASON, SEASON_STATE_FILE)

        played_names: set[str] = set()
        acc = exporter.accumulator
        for player in all_players_flat:
            s = acc.stats.get(player.name)
            if not s:
                continue
            stamina_state = sub_controller.stamina.get(player.name)
            ending_stamina = stamina_state.current_stamina if stamina_state else 100.0
            season_state.record_post_match(
                name=player.name,
                rating=s.get("rating", 6.0),
                goals=s.get("goals", 0),
                minutes_played=s.get("minutes_played", 0),
                ending_stamina=ending_stamina,
                yellow=s.get("yellow_cards", 0) > 0,
                red=s.get("red_cards", 0) > 0,
                injured=stamina_state.is_injured if stamina_state else False,
                injury_type=stamina_state.injury_type if stamina_state else "none",
                matches_out=stamina_state.matches_out() if stamina_state else 0,
                match_date=MATCH_DATE,
            )
            played_names.add(player.name)

        # Tick down bans/injuries for non-players in both clubs
        all_club_names: list[str] = []
        for club in [HOME_TEAM, AWAY_TEAM]:
            for rec in loader.get_club_players(club):
                all_club_names.append(rec.name)

        season_state.advance_matchday(all_club_names, played_names, MATCH_DATE)
        season_state.save()
        print(f"\n  💾 Season state saved → {SEASON_STATE_FILE}")
    except Exception as e:
        print(f"\n  ⚠️  Season state save failed: {e}")

    print(f"\n  ✅ Done. Output → {output_path}/\n")


if __name__ == "__main__":
    run()
