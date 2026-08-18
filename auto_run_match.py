"""
╔══════════════════════════════════════════════════════════════════════╗
║           PLOFA 26/27 — AUTOMATED MATCH RUNNER                       ║
║           auto_run_match.py                                          ║
║                                                                      ║
║  Only edit the USER CONFIG section below.                            ║
║  Everything else — squads, bench, subs, availability — is           ║
║  handled automatically from PLOFA-2026-2027.xlsx + match history.   ║
║                                                                      ║
║  What you set per match:                                             ║
║    • Which teams are playing (HOME_TEAM / AWAY_TEAM)                 ║
║    • The date and matchday number                                     ║
║    • Referee name and strictness (0.0 lenient → 1.0 very strict)    ║
║    • Weather (clear / rain / wind / fog)                             ║
║    • Is it a derby? (True / False)                                   ║
║                                                                      ║
║  What the system does automatically:                                 ║
║    • Reads all 300+ players from the Excel file                      ║
║    • Checks who is injured / suspended / fatigued from prior games   ║
║    • Picks the best available Starting XI per formation              ║
║    • Builds the bench (max 7, always keeps a backup GK)              ║
║    • Falls back to 2nd-team players if 1st-team can't fill a slot    ║
║    • Applies soul player buffs if a soul player is in the squad      ║
║    • Saves form/fatigue/injury/suspension state for next matchday    ║
║    • Updates the league table and exports all match files            ║
║                                                                      ║
║  Run: python auto_run_match.py                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
#C:\Users\Trevor Majani\AppData\Local\Python\bin>python "C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa\auto_run_match.py"
#THIS IS THE DEFAULT FOR RUNNING MATCHES NOT run_match.py because it automatically handles squads, bench, subs, and availability from the Excel file and match history. Use run_match.py for manual match runs with custom squads and settings. 

from __future__ import annotations
import os
import sys
from datetime import date
from typing import TypedDict, cast

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from player_soul import PlayerSoul, SoulArchetype, GreatnessPillars
from exporter import PLOFAExporter
from squad_manager import SubstitutionController, AvailabilityChecker, AvailabilityStatus
from roster_loader import RosterLoader, get_loader, auto_team_style
from season_manager import SeasonState


class _TeamEntry(TypedDict):
    home_color: str
    away_color: str
    style: TeamStyle | None
    playing_style: PlayingStyle | None
    intensity: Intensity | None


# ══════════════════════════════════════════════════════════════════════
# ▌ USER CONFIG — EDIT THIS BLOCK EVERY MATCHDAY
# ══════════════════════════════════════════════════════════════════════

# ── Match basics ───────────────────────────────────────────────────────
MATCH_DATE   = date(2026, 12, 29) # Year, Month, Day
MATCHDAY     = 20 # League matchday number (1–34)
SEASON       = "26/27"
COMPETITION  = "PLOFA"

# ── Teams — use exact names from the Excel (see TEAM_CATALOG below) ───
HOME_TEAM  = "Triumpher"
AWAY_TEAM  = "Uditon"

# ── Venue — leave "" to auto-fill "<HomeTeam> Stadium" ────────────────
VENUE     = ""
CAPACITY  = 100_000

# ── Referee ────────────────────────────────────────────────────────────
REFEREE    = "Marcus Osei"
STRICTNESS = 0.0 # 0.0 = very lenient  |  1.0 = very strict

# ── Conditions ─────────────────────────────────────────────────────────
WEATHER    = "clear"   # clear | rain | wind | fog
IS_DERBY   = False     # True for a local rivalry

# ── Substitution behaviour ─────────────────────────────────────────────
MANAGER_STUBBORNNESS = 0.35   # 0 = subs quickly  |  1 = never subs for stamina
MAX_SUBS  = 3

# ── Persistence ───────────────────────────────────────────────────────
#   After Matchday 1 this file accumulates injuries, suspensions and
#   fatigue.  It is read at the start of every match and updated after.
SEASON_STATE_FILE = "season_state.json"

# ── Outputs ────────────────────────────────────────────────────────────
OUTPUTS_DIR = "plofa_output"


# ══════════════════════════════════════════════════════════════════════
# ▌ SOUL PLAYERS
# Add / remove as the season progresses.
# The engine attaches souls automatically when the player's name appears
# in either squad — you don't need to touch run_match.py at all.
# ══════════════════════════════════════════════════════════════════════

SOUL_PLAYERS: dict[str, PlayerSoul] = {
    "Perćy Luka": PlayerSoul(
        "Perćy Luka",
        archetype=SoulArchetype.ATTACKING_PROPHET,
        pillars=GreatnessPillars(hardwork=0.99, talent=0.90, luck=0.99,),
    ),
    # Template — uncomment and fill in when a new soul emerges:
    "Juan Massey": PlayerSoul(
         "Juan Massey",
         archetype=SoulArchetype.DEFENSIVE_PURIST,
         pillars=GreatnessPillars(hardwork=0.99, talent=0.87, luck=0.96),
     ),
    "Zachery Worth": PlayerSoul(
         "Zachery Worth",
         archetype=SoulArchetype.WIDE_DESTROYER,
         pillars=GreatnessPillars(hardwork=0.72, talent=0.99, luck=0.99)
     ),
    "Danso Potwemi": PlayerSoul(
         "Danso Potwemi",
         archetype=SoulArchetype.CREATIVE_ORACLE,
         pillars=GreatnessPillars(hardwork=0.87, talent=0.97, luck=0.99)
     ),
    "Hill Prosper": PlayerSoul(
         "Hill Prosper",
         archetype=SoulArchetype.GOALSCORING_SAVANT,
         pillars=GreatnessPillars(hardwork=0.90, talent=0.80, luck=0.93)
     ),
    "Caut Mayoderoki": PlayerSoul(
         "Caut Mayoderoki",
         archetype=SoulArchetype.MIDFIELD_PHILOSOPHER,
         pillars=GreatnessPillars(hardwork=0.99, talent=0.95, luck=0.94)
     ),
    "Van Lee": PlayerSoul(
         "Van Lee",
         archetype=SoulArchetype.WALL,
         pillars=GreatnessPillars(hardwork=0.97, talent=0.98, luck=0.83)
     ),
    "Duane Rokariĉ": PlayerSoul(
         "Duane Rokariĉ",
         archetype=SoulArchetype.CREATIVE_ORACLE,
         pillars=GreatnessPillars(hardwork=0.96, talent=0.90, luck=0.83)
     ),
    "Mikro Vitro": PlayerSoul(
         "Mikro Vitro",
         archetype=SoulArchetype.WIDE_DESTROYER,
         pillars=GreatnessPillars(hardwork=0.82, talent=0.90, luck=0.90)
     ),
    "Carl Tœvoda": PlayerSoul(
        "Carl Tœvoda",
        archetype=SoulArchetype.DEFENSIVE_PURIST,
        pillars=GreatnessPillars(hardwork=0.88, talent=0.87, luck=0.86)
    ),
    "Francis Dućźè": PlayerSoul(
        "Francis Dućźè",
        archetype=SoulArchetype.GOALSCORING_SAVANT,
        pillars=GreatnessPillars(hardwork=0.95, talent=0.90, luck=0.93)
    ),
    "Hillary Monzade": PlayerSoul(
        "Hillary Monzade",
        archetype=SoulArchetype.ATTACKING_PROPHET,
        pillars=GreatnessPillars(hardwork=0.83, talent=0.96, luck=0.95)
    ),

}


# ══════════════════════════════════════════════════════════════════════
# ▌ TEAM CATALOG
# Maps every club name to its kit colors and preferred style overrides.
# Formation-appropriate styles are auto-picked when no override exists.
# Add overrides below if you want a club to always play a certain style.
# ══════════════════════════════════════════════════════════════════════

# fmt: off
TEAM_CATALOG: dict[str, _TeamEntry] = {
    # ── key: exact club name from Excel ────────────────────────────────
    # Required: "home_color", "away_color"
    # Optional: "style", "playing_style", "intensity"  (override auto-pick)
    "Hartwell City": {
        "home_color": "#003087",
        "away_color": "#C8102E",
        "style":         TeamStyle.ATTACKING,
        "playing_style": PlayingStyle.HIGH_PRESS,
        "intensity":     Intensity.HIGH,
    },
    "Thornfield United": {
        "home_color": "#C8102E",
        "away_color": "#FFFFFF",
        "style":         TeamStyle.FLUID_COUNTER,
        "playing_style": PlayingStyle.COUNTER,
        "intensity":     Intensity.MEDIUM,
    },
    "Uditon": {
        "home_color": "#01C271",
        "away_color": "#F8C300",
        "style":         TeamStyle.ATTACKING,
        "playing_style": PlayingStyle.POSSESSION,
        "intensity":     Intensity.HIGH,
    },
    "Claw": {
        "home_color": "#E90000",
        "away_color": "#E0E0E0",
        "style":         TeamStyle.GEGENPRESSING,
        "playing_style": PlayingStyle.HIGH_PRESS,
        "intensity":     Intensity.VERY_HIGH,
    },
    "Pearls": {
        "home_color": "#C0C0C0",
        "away_color": "#8B008B",
        "style":         TeamStyle.ULTRA_ATTACKING,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.HIGH,
    },
    "Natrican": {
        "home_color": "#FA6807",
        "away_color": "#002244",
        "style":         TeamStyle.ULTRA_ATTACKING,
        "playing_style": PlayingStyle.DIRECT,
        "intensity":     Intensity.MEDIUM,
    },
    "Lige-8": {
        "home_color": "#061FBEE6",
        "away_color": "#000000EF",
        "style":         TeamStyle.TIKI_TAKA,
        "playing_style": PlayingStyle.POSSESSION,
        "intensity":     Intensity.HIGH,
    },
    "Triumpher": {
        "home_color": "#FFFFFF",
        "away_color": "#E23C8F",
        "style":         TeamStyle.ULTRA_ATTACKING,
        "playing_style": PlayingStyle.POSSESSION,
        "intensity":     Intensity.VERY_HIGH,
    },
    "Play City": {
        "home_color": "#FDBA6C",
        "away_color": "#FFFFFF",
        "style":         TeamStyle.PARK_THE_BUS,
        "playing_style": PlayingStyle.COUNTER,
        "intensity":     Intensity.MEDIUM,
    },
    "Red Wolves": {
        "home_color": "#CC0000",
        "away_color": "#1C1C1C",
        "style":         TeamStyle.ULTRA_ATTACKING,
        "playing_style": PlayingStyle.TRANSITION_FOCUSED,
        "intensity":     Intensity.VERY_HIGH,
    },
    "Telbey": {
        "home_color": "#005B8E",
        "away_color": "#F5A623",
        "style":         TeamStyle.BALANCED,
        "playing_style": PlayingStyle.LOW_BLOCK,
        "intensity":     Intensity.LOW,
    },
    "Justice": {
        "home_color": "#2C3E50",
        "away_color": "#E74C3C",
        "style":         TeamStyle.BALANCED,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.LOW,
    },
    "Tryox City": {
        "home_color": "#1ABC9C",
        "away_color": "#2C3E50",
        "style":         TeamStyle.FLUID_COUNTER,
        "playing_style": PlayingStyle.COUNTER,
        "intensity":     Intensity.HIGH,
    },
    "Oxton": {
        "home_color": "#8E44AD",
        "away_color": "#ECF0F1",
        "style":         TeamStyle.DEFENSIVE,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.VERY_HIGH,
    },
    "Trendboys": {
        "home_color": "#F39C12",
        "away_color": "#2E4053",
        "style":         TeamStyle.ROUTE_ONE,
        "playing_style": PlayingStyle.DIRECT,
        "intensity":     Intensity.MEDIUM,
    },
    "Club Chovers": {
        "home_color": "#27AE60",
        "away_color": "#1A252F",
        "style":         TeamStyle.FLUID_COUNTER,
        "playing_style": PlayingStyle.TRANSITION_FOCUSED,
        "intensity":     Intensity.HIGH,
    },
    "Seafcea": {
        "home_color": "#0097A7",
        "away_color": "#FFFFFF",
        "style":         TeamStyle.WING_PLAY,
        "playing_style": PlayingStyle.COUNTER,
        "intensity":     Intensity.LOW,
    },
    "Avada Zenith":{
        "home_color": "#FFFFFF",
        "away_color": "#EA09AA",
        "style":         TeamStyle.ATTACKING,
        "playing_style": PlayingStyle.COUNTER,
        "intensity":     Intensity.LOW,
    },
    "Ganester":{
        "home_color": "#EAF207",
        "away_color": "#0F22CD",
        "style":         TeamStyle.ROUTE_ONE,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.LOW,
    },
    "Rodice": {
        "home_color": "#0B3E0C",
        "away_color": "#090202",
        "style":         TeamStyle.GEGENPRESSING,
        "playing_style": PlayingStyle.MIXED,
        "intensity":     Intensity.MEDIUM,
    }
}
# fmt: on

# ══════════════════════════════════════════════════════════════════════
# ▌ ENGINE — do not edit below this line
# ══════════════════════════════════════════════════════════════════════

def _resolve_team_profile(club: str, formation: str, is_home: bool) -> TeamProfile:
    """
    Build a TeamProfile for `club`.
    Uses manual overrides from TEAM_CATALOG when present,
    otherwise calls auto_team_style() which maps formation → sensible defaults.
    """
    catalog = TEAM_CATALOG.get(club, {})
    if "style" in catalog and "playing_style" in catalog and "intensity" in catalog:
        return TeamProfile(
            name=club,
            style=catalog["style"],
            playing_style=catalog["playing_style"],
            intensity=catalog["intensity"],
        )
    return auto_team_style(club, formation, is_home=is_home)


def _resolve_color(club: str, is_home: bool) -> str:
    catalog = TEAM_CATALOG.get(club, {})
    key = "home_color" if is_home else "away_color"
    return catalog.get(key, "#003087" if is_home else "#C8102E")


def _build_availability(
    team: str,
    matchday: int,
    season_state: SeasonState,
    outputs_dir: str,
    match_date=None,
) -> dict:
    """
    Merge availability from two sources:
      1. SeasonState (JSON — persistent across matchdays, most authoritative)
      2. AvailabilityChecker (reads prior match CSV/XLSX files as fallback)

    SeasonState wins when it says a player is unavailable.
    Returns {player_name: PlayerAvailability-like object} understood by
    RosterLoader._filter_eligible().
    """
    from squad_manager import PlayerAvailability

    merged: dict[str, PlayerAvailability] = {}

    # ── Source 1: AvailabilityChecker (file-based) ──────────────────
    if matchday > 1 and os.path.isdir(outputs_dir):
        checker = AvailabilityChecker(outputs_dir)
        file_avail = checker.check(team, matchday)
        merged.update(file_avail)

    # ── Source 2: SeasonState (JSON — cross-matchday persistence) ───
    for name, state in season_state.players.items():
        # Only care about players on this team — we can't filter by team
        # in SeasonState directly (it's flat), so we apply all and let
        # RosterLoader ignore unknowns naturally.
        ok, reason = season_state.is_available(name, match_date)
        if not ok:
            status_map = {
                "suspended (red card)":  AvailabilityStatus.SUSPENDED_RED,
                "suspended (5 yellows)": AvailabilityStatus.SUSPENDED_YEL,
            }
            status = status_map.get(reason, AvailabilityStatus.INJURED)
            merged[name] = PlayerAvailability(
                name=name,
                status=status,
                reason=reason,
                starting_stamina=float(state.get("starting_stamina", 100.0)),
            )
        elif state.get("fatigue_level", 0) > 65 or state.get("starting_stamina", 100) < 75:
            existing = merged.get(name)
            # Only downgrade to FATIGUE_WARNING if not already harder status
            if existing is None or existing.status == AvailabilityStatus.FIT:
                merged[name] = PlayerAvailability(
                    name=name,
                    status=AvailabilityStatus.FATIGUE_WARNING,
                    reason=(
                        f"Fatigue carryover: {state.get('fatigue_level', 0):.0f}% drain, "
                        f"starting stamina ~{state.get('starting_stamina', 100):.0f}%"
                    ),
                    fatigue_level=float(state.get("fatigue_level", 0)),
                    starting_stamina=float(state.get("starting_stamina", 100.0)),
                )

    return merged


def _print_availability_report(
    team: str,
    availability: dict,
    squad_result: dict,
) -> None:
    """
    Print a clear pre-match availability summary for a team:
      - Who was excluded and why
      - Who started despite a fatigue flag
      - Notes from the roster loader (2nd-team call-ups etc.)
    """
    print(f"\n  📋 {team} — Availability Report")
    print(f"  {'─' * 55}")

    blocked = {
        n: a for n, a in availability.items()
        if a.status.value in ("suspended_red", "suspended_yel", "injured")
    }
    fatigued = {
        n: a for n, a in availability.items()
        if a.status == AvailabilityStatus.FATIGUE_WARNING
    }

    if blocked:
        for name, avail in blocked.items():
            icon = "🚫" if "suspended" in avail.status.value else "🤕"
            print(f"  {icon}  {name:<22} OUT — {avail.reason}")
    else:
        print("  ✅  No suspensions or injuries on record.")

    if fatigued:
        starter_names = {s[0] for s in squad_result.get("starters", [])}
        for name, avail in fatigued.items():
            tag = " (started anyway)" if name in starter_names else " (benched/rested)"
            print(f"  ⚠️   {name:<22} FATIGUE WARNING{tag}")

    for note in squad_result.get("notes", []):
        if any(tag in note for tag in ("⛔", "⚠️", "❗", "Formation")):
            print(f"  ℹ️   {note}")


def _attach_souls(all_players: list) -> list[str]:
    """Attach soul profiles to any player whose name is in SOUL_PLAYERS."""
    attached = []
    for player in all_players:
        if player.name in SOUL_PLAYERS:
            player.dna.soul = SOUL_PLAYERS[player.name]
            attached.append(player.name)
    return attached


def _apply_starting_stamina(
    all_players: list,
    availability: dict,
    season_state: SeasonState,
    match_date=None,
) -> None:
    """
    Hydrate each player's starting stamina from persisted season state
    or their availability record, whichever is more precise.
    """
    for player in all_players:
        # SeasonState is most authoritative
        p_state = season_state.get_player_state(player.name)
        starting = float(p_state.get("starting_stamina", 100.0))

        # Fall back to availability checker's stamina reading
        if starting == 100.0 and player.name in availability:
            starting = float(availability[player.name].starting_stamina or 100.0)

        # Clamp so no one starts below 70%
        starting = max(70.0, min(100.0, starting))

        # Apply season state hydrations FIRST (confidence, recent form, injuries)
        # Also auto-clears injuries if return date has passed
        if hasattr(player, "dna"):
            season_state.apply_pre_match(player.dna, player.name, match_date)

        # THEN override stamina/fatigue with the clamped starting values
        if hasattr(player, "dna") and hasattr(player.dna, "form"):
            player.dna.form.fatigue_level = max(0.0, 100.0 - starting)


def _persist_post_match(
    result,
    exporter: PLOFAExporter,
    sub_controller: SubstitutionController,
    home_squad: dict,
    away_squad: dict,
    season_state: SeasonState,
    loader: RosterLoader,
    match_date=None,
) -> None:
    """
    After the final whistle: update SeasonState for every player in
    both full rosters (including those who didn't play today).
    """
    acc = exporter.accumulator
    all_players = (
        home_squad["starters"] + home_squad["substitutes"] +
        away_squad["starters"] + away_squad["substitutes"]
    )

    played_names: set[str] = set()
    try:
        for player in all_players:
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
                match_date=match_date,
            )
            played_names.add(player.name)

        # Tick down bans/injuries for everyone in both clubs who DIDN'T play
        all_club_names: list[str] = []
        for club in [result.config.home_team, result.config.away_team]:
            for rec in loader.get_club_players(club):
                all_club_names.append(rec.name)

        season_state.advance_matchday(all_club_names, played_names, match_date)
    finally:
        season_state.save()


# ── MAIN ─────────────────────────────────────────────────────────────

def run():
    sys.stdout.reconfigure(encoding="utf-8")

    venue = VENUE if VENUE else f"{HOME_TEAM} Stadium"

    print(f"\n{'═' * 64}")
    print(f"  PLOFA {SEASON} — Matchday {MATCHDAY}")
    print(f"  {HOME_TEAM}  vs  {AWAY_TEAM}")
    print(f"  {MATCH_DATE.strftime('%A %d %B %Y')}  |  {venue}")
    print(f"  Referee: {REFEREE}  (strictness: {STRICTNESS})")
    print(f"  Weather: {WEATHER}{'  |  DERBY' if IS_DERBY else ''}")
    print(f"{'═' * 64}")

    # ── Load Excel roster ─────────────────────────────────────
    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in [HOME_TEAM, AWAY_TEAM]:
        if team not in clubs:
            print(f"\n  ❌ Team '{team}' not found in Excel.")
            print(f"     Available clubs: {', '.join(clubs)}")
            sys.exit(1)

    # ── Season state (cross-matchday persistence) ─────────────
    season_state = SeasonState(SEASON, SEASON_STATE_FILE)

    # ── Build availability for both teams ─────────────────────
    home_avail = _build_availability(HOME_TEAM, MATCHDAY, season_state, OUTPUTS_DIR, MATCH_DATE)
    away_avail = _build_availability(AWAY_TEAM, MATCHDAY, season_state, OUTPUTS_DIR, MATCH_DATE)

    # ── Auto-select squads from Excel ─────────────────────────
    print(f"\n  🔍 Auto-selecting squads from Excel...")

    home_raw = loader.build_matchday_squad(HOME_TEAM, availability=home_avail)
    away_raw = loader.build_matchday_squad(AWAY_TEAM, availability=away_avail)

    home_formation = home_raw["formation"]
    away_formation = away_raw["formation"]

    # ── Availability reports ──────────────────────────────────
    _print_availability_report(HOME_TEAM, home_avail, home_raw)
    _print_availability_report(AWAY_TEAM, away_avail, away_raw)

    # ── Team profiles (styles) ────────────────────────────────
    HOME_STYLE = _resolve_team_profile(HOME_TEAM, home_formation, is_home=True)
    AWAY_STYLE = _resolve_team_profile(AWAY_TEAM, away_formation, is_home=False)

    print(f"\n  🏟️  {HOME_TEAM} [{home_formation}] — {HOME_STYLE.style.value} / {HOME_STYLE.playing_style.value}")
    print(f"  ✈️  {AWAY_TEAM} [{away_formation}] — {AWAY_STYLE.style.value} / {AWAY_STYLE.playing_style.value}")

    # ── Build PlayerProfile squads via SquadBuilder ───────────
    home_squad = SquadBuilder.build(
        team_name=HOME_TEAM,
        starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"],
    )
    away_squad = SquadBuilder.build(
        team_name=AWAY_TEAM,
        starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"],
    )

    # ── Print selected lineups ────────────────────────────────
    print(f"\n  📋 {HOME_TEAM} Starting XI ({home_formation}):")
    for p in home_squad["starters"]:
        print(f"       {p.position:<4}  {p.name}")
    print(f"  📋 {HOME_TEAM} Bench:")
    for p in home_squad["substitutes"]:
        sub_min = getattr(p, "sub_in_minute", None) or getattr(p.dna, "sub_in_minute", None)
        min_tag = f"  (sub ~{sub_min}')" if sub_min else ""
        print(f"       {p.position:<4}  {p.name}{min_tag}")

    print(f"\n  📋 {AWAY_TEAM} Starting XI ({away_formation}):")
    for p in away_squad["starters"]:
        print(f"       {p.position:<4}  {p.name}")
    print(f"  📋 {AWAY_TEAM} Bench:")
    for p in away_squad["substitutes"]:
        sub_min = getattr(p, "sub_in_minute", None) or getattr(p.dna, "sub_in_minute", None)
        min_tag = f"  (sub ~{sub_min}')" if sub_min else ""
        print(f"       {p.position:<4}  {p.name}{min_tag}")

    # ── Average age of starting XI ────────────────────────────
    home_avg_age = sum(p.dna.age for p in home_squad["starters"]) / len(home_squad["starters"])
    away_avg_age = sum(p.dna.age for p in away_squad["starters"]) / len(away_squad["starters"])
    print(f"\n  📊 Average age of starting XI:")
    print(f"       {HOME_TEAM}: {home_avg_age:.1f} years")
    print(f"       {AWAY_TEAM}: {away_avg_age:.1f} years")

    # ── Apply starting stamina from season state ──────────────
    all_players_flat = (
        home_squad["starters"] + home_squad["substitutes"] +
        away_squad["starters"] + away_squad["substitutes"]
    )
    _apply_starting_stamina(all_players_flat, {**home_avail, **away_avail}, season_state, MATCH_DATE)

    # ── Attach soul players ───────────────────────────────────
    souls_attached = _attach_souls(all_players_flat)
    if souls_attached:
        print(f"\n  🔮 Soul players active: {', '.join(souls_attached)}")
        for name in souls_attached:
            soul = SOUL_PLAYERS[name]
            print(f"     {name}: {soul.profile.label} | "
                  f"G={soul.greatness_coefficient:.4f} | "
                  f"Tier={soul.tier} | "
                  f"{'⚡ OMEGA ACTIVE' if soul.pillars.omega_activated else 'No Omega'}")

    # ── Match config ──────────────────────────────────────────
    config = MatchConfig(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        match_date=MATCH_DATE,
        matchday=MATCHDAY,
        season=SEASON,
        competition=COMPETITION,
        venue=venue,
        stadium_capacity=CAPACITY,
        referee=REFEREE,
        referee_strictness=STRICTNESS,
        is_derby=IS_DERBY,
        weather=WEATHER,
    )

    # ── Substitution controller ───────────────────────────────
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

    # ── Simulate ──────────────────────────────────────────────
    print(f"\n  ⚽ Simulating...\n")
    engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
    engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_controller)

    result = engine.simulate()
    print(result.summary())

    # ── Big 6 teams (clubs with highest market values) ─────────
    # These draw bigger crowds and command higher ticket prices.
    # Auto-detected from actual squad market values in the DB.
    BIG6_TEAMS: set[str] = set()
    try:
        all_club_values = []
        for club in loader.get_all_clubs():
            players = loader.get_club_players(club)
            if players:
                values = [
                    p.market_value for p in players
                    if (p.is_first_team or p.is_second_team) and p.market_value > 0
                ]
                if values:
                    all_club_values.append((club, sum(values)))
        all_club_values.sort(key=lambda x: x[1], reverse=True)
        BIG6_TEAMS = {c[0] for c in all_club_values[:6]}
        print(f"\n  🏆 Big 6 teams (by market value): {', '.join(sorted(BIG6_TEAMS))}")
    except Exception:
        BIG6_TEAMS = {"Pearls", "Claw", "Uditon", "Lige-8", "Triumpher", "Natrican"}
        pass

    # ── Export ────────────────────────────────────────────────
    folder_name = (
        f"{HOME_TEAM.replace(' ', '_')}_vs_"
        f"{AWAY_TEAM.replace(' ', '_')}_MD{MATCHDAY:02d}"
    )
    output_path = os.path.join(OUTPUTS_DIR, folder_name)
    os.makedirs(output_path, exist_ok=True)

    # ── Resolve colors (ensure home != away) ────────────────────
    home_color = _resolve_color(HOME_TEAM, is_home=True)
    away_color = _resolve_color(AWAY_TEAM, is_home=False)

    if away_color == home_color:
        alt = _resolve_color(AWAY_TEAM, is_home=True)
        if alt != home_color:
            away_color = alt
        else:
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
        all_players={HOME_TEAM: home_squad, AWAY_TEAM: away_squad},
        home_color=home_color,
        away_color=away_color,
        sub_controller=sub_controller,
        big6_teams=BIG6_TEAMS,
    )

    # ── Persist season state BEFORE export ────────────────────
    # This ensures state is saved even if export crashes mid-way.
    print(f"\n  💾 Saving season state → {SEASON_STATE_FILE}")
    _persist_post_match(
        result=result,
        exporter=exporter,
        sub_controller=sub_controller,
        home_squad=home_squad,
        away_squad=away_squad,
        season_state=season_state,
        loader=loader,
        match_date=MATCH_DATE,
    )

    try:
        exporter.export_all(output_path)
    except Exception as e:
        print(f"\n  ⚠️  Export failed: {e}")
        print(f"  Season state was already saved — you can re-run without losing progress.")
        raise

    # ── Soul player report ────────────────────────────────────
    if souls_attached:
        print(f"\n  🔮 SOUL PLAYER MATCH REPORT")
        print(f"  {'─' * 40}")
        acc = exporter.accumulator
        for name in souls_attached:
            s = acc.stats.get(name)
            if s:
                soul = SOUL_PLAYERS[name]
                print(f"  {name} ({soul.profile.label})")
                print(f"    Goals: {s['goals']}  Assists: {s['assists']}  "
                      f"Rating: {s['rating']}")
                print(f"    xG: {s['xg']:.3f}  xA: {s['xa']:.3f}")
                print(f"    Dribbles: {s['dribbles_comp']}/{s['dribbles_att']}  "
                      f"Shot Assists: {s['shot_assists']}  "
                      f"Carries: {s['carries']}")

    print(f"\n  ✅ Done. Output → {output_path}/")
    print(f"  📊 Season state updated. Injuries/suspensions carry to next matchday.\n")


if __name__ == "__main__":
    run()
