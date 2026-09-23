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
#
# ⛔ OFFICIAL RUNNER — TREVOR ONLY. DO NOT RUN VIA AI/AGENT.
# This is the official, canonical match runner. It overwrites the authoritative
# season_state.json (form, fatigue, injuries, suspensions, season minutes) on
# every run. A fixture may ONLY be run once, by the human runner (Trevor), who
# supplies the real date / home team / weather / referee inputs per match.
# AI assistants or automated agents must NEVER execute or re-run this file:
# an accidental rerun of a played fixture corrupts the season ledger. A fixture's
# results are authoritative once recorded and must never be re-run.
# For agent-controlled test simulations (scratch, non-canonical) use run_match.py. 

from __future__ import annotations
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, TypedDict, cast

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from player_soul import PlayerSoul, SoulArchetype, GreatnessPillars
from exporter import PLOFAExporter
from squad_manager import (
    SubstitutionController,
    AvailabilityChecker,
    AvailabilityStatus,
    PlayerAvailability,
)
from roster_loader import RosterLoader, get_loader, auto_team_style
from season_manager import SeasonState
from referee_pool import RefereeManager


class _TeamEntry(TypedDict):
    home_color: str
    away_color: str
    style: TeamStyle | None
    playing_style: PlayingStyle | None
    intensity: Intensity | None


class _SquadResult(TypedDict, total=False):
    starters: list[Any]
    notes: list[str]


class _BuiltSquad(TypedDict):
    starters: list[Any]
    substitutes: list[Any]


# ══════════════════════════════════════════════════════════════════════
# ▌ USER CONFIG — EDIT THIS BLOCK EVERY MATCHDAY
# ══════════════════════════════════════════════════════════════════════

# ── Match basics ───────────────────────────────────────────────────────
MATCH_DATE   = date(2026, 9, 20) # Year, Month, Day
MATCHDAY     = 5 # League matchday number (1–34)
SEASON       = "26/27"
COMPETITION  = "PLOFA"

# ── Teams — use exact names from the Excel (see TEAM_CATALOG below) ───
HOME_TEAM  = "  "
AWAY_TEAM  = "  "

# ── Venue — leave "" to auto-fill "<HomeTeam> Stadium" ────────────────
VENUE     = "   "
CAPACITY  = 87_000

# ── Referee (auto-assigned by default) ──────────────────────────────
# Set FORCE_REF to a referee name to override rotation (e.g. for a derby).
# Leave as None to let the system auto-pick based on EPL rotation rules.
FORCE_REF = None  # e.g. "Marcus Osei" or None for auto-assign
REFEREE    = None  # auto-filled below
STRICTNESS = None  # auto-filled below


# ── Conditions ─────────────────────────────────────────────────────────
# 'WEATHER' accepts a fixed condition (clear | rain | wind | fog) OR 'auto'
# to derive real seasonal climate (weather + temperature) from venue & date.
WEATHER        = "auto"
IS_DERBY   = False     # True for a local rivalry
# Toggle weather/fixture-time physics ON (True) or OFF (False). OFF keeps
# Checkpoint-6 behaviour exactly; ON activates pass/shot/stamina/GK
# multipliers and kickoff-time attendance with real per-match climate.
WEATHER_ENABLED = True

# ── Substitution behaviour ─────────────────────────────────────────────
MANAGER_STUBBORNNESS = 0.35   # 0 = subs quickly  |  1 = never subs for stamina
MAX_SUBS  = 3

# ── Persistence ───────────────────────────────────────────────────────
#   After Matchday 1 this file accumulates injuries, suspensions and
#   fatigue.  It is read at the start of every match and updated after.
SEASON_STATE_FILE = "season_state.json"

# ── Cognition (TOLAND mind layer) ─────────────────────────────────────
#   When ON, every player is given a PlayerMind hydrated from the season
#   store (so temperament/memory carry across matchdays), the engine
#   decision path routes through the mind (FOV gate + temperament sampling
#   + consequence reasoning merge since 2026-09-20), and the match event
#   stream feeds episodic memory live.  When OFF, the engine is the plain
#   reasoned-neural path (no minds).
#   ON BY DEFAULT since 2026-09-20. Override: PLOFA_COGNITION=0 python ...
COGNITION = os.environ.get("PLOFA_COGNITION", "1") == "1"

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
    "Francis Bonadi": PlayerSoul(
        "Francis Bonadi",
         archetype=SoulArchetype.WALL,
         pillars=GreatnessPillars(hardwork=0.90, talent=0.90, luck=0.89),
     ),
    "Zachery Worth": PlayerSoul(
         "Zachery Worth",
         archetype=SoulArchetype.WIDE_DESTROYER,
         pillars=GreatnessPillars(hardwork=0.83, talent=0.99, luck=0.99)
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
        "intensity":     Intensity.HIGH,
    },
    "Pearls": {
        "home_color": "#C0C0C0",
        "away_color": "#8B008B",
        "style":         TeamStyle.ATTACKING,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.HIGH,
    },
    "Natrican": {
        "home_color": "#FA6807",
        "away_color": "#002244",
        "style":         TeamStyle.ATTACKING,
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
        "style":         TeamStyle.ATTACKING,
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
        "style":         TeamStyle.ATTACKING,
        "playing_style": PlayingStyle.TRANSITION_FOCUSED,
        "intensity":     Intensity.HIGH,
    },
    "Telbey": {
        "home_color": "#005B8E",
        "away_color": "#F5A623",
        "style":         TeamStyle.BALANCED,
        "playing_style": PlayingStyle.LOW_BLOCK,
        "intensity":     Intensity.HIGH,
    },
    "Justice": {
        "home_color": "#2C3E50",
        "away_color": "#E74C3C",
        "style":         TeamStyle.BALANCED,
        "playing_style": PlayingStyle.PATIENT_BUILD_UP,
        "intensity":     Intensity.HIGH,
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
        "playing_style": PlayingStyle.COUNTER,
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
        "intensity":     Intensity.LOW,
    }
}
# fmt: on

# ══════════════════════════════════════════════════════════════════════
# ▌ ENGINE — do not edit below this line
# ══════════════════════════════════════════════════════════════════════

def _resolve_team_profile(club: str, formation: str, is_home: bool,
                         club_override: dict | None = None,
                         manager_label: str | None = None) -> TeamProfile:
    """
    Build a TeamProfile for `club`.
    Uses manual overrides from TEAM_CATALOG when present,
    otherwise calls auto_team_style() which maps formation → sensible defaults.
    Checkpoint 30 — resolves the ClubPhilosophy from (1) per-club catalog
    override, (2) persisted season_state override, (3) manager label string.
    """
    catalog = TEAM_CATALOG.get(club)
    if catalog is not None:
        style = catalog.get("style")
        if style is None:
            return auto_team_style(club, formation, is_home=is_home)

        playing_style = catalog.get("playing_style")
        if playing_style is None:
            playing_style = PlayingStyle.MIXED

        intensity = catalog.get("intensity")
        if intensity is None:
            intensity = Intensity.MEDIUM

        profile = TeamProfile(
            name=club,
            style=style,
            playing_style=playing_style,
            intensity=intensity,
        )
        # ── Philosophy resolution (Checkpoint 30) ──
        # Priority: catalog explicit "philosophy" key > club_override >
        #           manager_label (fallback to balanced)
        cat_phi = catalog.get("philosophy")
        if cat_phi is None and club_override is not None:
            cat_phi = club_override
        if cat_phi is None and manager_label is not None:
            from philosophy import resolve_philosophy
            resolved = resolve_philosophy(club, manager_label=manager_label)
            if resolved is not None:
                profile.philosophy = resolved
                profile._apply_philosophy()
        elif isinstance(cat_phi, dict):
            from philosophy import resolve_philosophy
            resolved = resolve_philosophy(club, club_override=cat_phi)
            if resolved is not None:
                profile.philosophy = resolved
                profile._apply_philosophy()
        elif cat_phi is not None and isinstance(cat_phi, str):
            from philosophy import resolve_philosophy
            resolved = resolve_philosophy(club, club_override={"archetype": cat_phi})
            if resolved is not None:
                profile.philosophy = resolved
                profile._apply_philosophy()
        return profile
    profile = auto_team_style(club, formation, is_home=is_home)
    if manager_label is not None:
        from philosophy import resolve_philosophy
        resolved = resolve_philosophy(club, manager_label=manager_label)
        if resolved is not None:
            profile.philosophy = resolved
            profile._apply_philosophy()
    return profile


def _resolve_color(club: str, is_home: bool) -> str:
    catalog = TEAM_CATALOG.get(club)
    if catalog is None:
        return "#003087" if is_home else "#C8102E"
    return catalog["home_color"] if is_home else catalog["away_color"]


def _build_availability(
    team: str,
    matchday: int,
    season_state: SeasonState,
    outputs_dir: str,
    match_date: date | None = None,
) -> dict[str, PlayerAvailability]:
    """
    Merge availability from two sources:
      1. SeasonState (JSON — persistent across matchdays, most authoritative)
      2. AvailabilityChecker (reads prior match CSV/XLSX files as fallback)

    SeasonState wins when it says a player is unavailable.
    Returns {player_name: PlayerAvailability-like object} understood by
    RosterLoader._filter_eligible().
    """
    merged: dict[str, PlayerAvailability] = {}

    # ── Source 1: AvailabilityChecker (file-based) ──────────────────
    if matchday > 1 and os.path.isdir(outputs_dir):
        checker = AvailabilityChecker(outputs_dir)
        file_avail = checker.check(team, matchday)
        merged.update(file_avail)

    # ── Source 2: SeasonState (JSON — cross-matchday persistence) ───
    for name, state in season_state.players.items():
        # Only care about players on this team — we can't filter by team
        # in SeasonState directly, so we apply all and let
        # RosterLoader ignore unknowns naturally.
        ok, reason = season_state.is_available(name, match_date)
        # Date-based recovery projected onto this fixture's date
        st, fat = season_state.project_stamina(state, match_date)
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
                starting_stamina=st,
            )
            continue

        existing = merged.get(name)
        if existing is not None and existing.status.value in (
            "suspended_red", "suspended_yel", "injured"
        ):
            continue  # keep a harder status from the file checker

        if fat > 65 or st < 75:
            merged[name] = PlayerAvailability(
                name=name,
                status=AvailabilityStatus.FATIGUE_WARNING,
                reason=(
                    f"Fatigue carryover: {fat:.0f}% fatigue, "
                    f"starting stamina ~{st:.0f}%"
                ),
                fatigue_level=fat,
                starting_stamina=st,
            )
        else:
            # SeasonState is authoritative for stamina once it has data
            merged[name] = PlayerAvailability(
                name=name,
                status=AvailabilityStatus.FIT,
                reason="Fit to play",
                fatigue_level=fat,
                starting_stamina=st,
            )

    return merged



def _print_availability_report(
    team: str,
    availability: dict[str, PlayerAvailability],
    squad_result: _SquadResult,
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


def _attach_souls(all_players: list[Any]) -> list[str]:
    """Attach soul profiles to any player whose name is in SOUL_PLAYERS."""
    attached: list[str] = []
    for player in all_players:
        if player.name in SOUL_PLAYERS:
            player.dna.soul = SOUL_PLAYERS[player.name]
            attached.append(player.name)
    return attached


def _apply_starting_stamina(
    all_players: list[Any],
    availability: dict[str, PlayerAvailability],
    season_state: SeasonState,
    match_date: date | None = None,
) -> None:
    """
    Hydrate each player's starting stamina from persisted season state
    or their availability record, whichever is more precise.

    SeasonState projects date-based recovery (days since last match × 10%)
    inside apply_pre_match, so the returned value is kickoff-day stamina.
    """
    for player in all_players:
        # SeasonState hydrations FIRST (confidence, recent form, injuries,
        # date-based stamina projection). Also auto-clears injuries if the
        # return date has passed.
        projected = 100.0
        if hasattr(player, "dna"):
            apply_pre_match = cast(
                Callable[[Any, str, date | None], Any],
                getattr(season_state, "apply_pre_match"),
            )
            projected = float(apply_pre_match(player.dna, player.name, match_date) or 100.0)
        else:
            st_fat = season_state.project_stamina(
                season_state.get_player_state(player.name), match_date
            )
            projected = float(st_fat[0])

        starting = projected

        # Fall back to availability checker's stamina reading when the
        # season state has nothing (player never recorded a match)
        if starting == 100.0 and player.name in availability:
            starting = float(availability[player.name].starting_stamina or 100.0)

        # Clamp so no one starts below 70%
        starting = max(70.0, min(100.0, starting))

        # THEN override stamina/fatigue with the clamped starting values
        if hasattr(player, "dna") and hasattr(player.dna, "form"):
            player.dna.form.fatigue_level = max(0.0, 100.0 - starting)


def _persist_post_match(
    result: Any,
    exporter: PLOFAExporter,
    sub_controller: SubstitutionController,
    home_squad: _BuiltSquad,
    away_squad: _BuiltSquad,
    season_state: SeasonState,
    loader: RosterLoader,
    match_date: date | None = None,
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
    starter_names = {
        player.name
        for squad in (home_squad, away_squad)
        for player in squad["starters"]
    }

    # Record the fixture into the league standings so attendance reflects
    # each team's season performance (position + recent form).
    matchday = getattr(result.config, "matchday", 0) or MATCHDAY
    # Full roster of BOTH clubs (players + bench) — the ledger snapshot
    # must capture every name the post-match loop can touch so a re-run
    # restores a clean pre-match baseline.
    ledger_names: list[str] = []
    for club in [result.config.home_team, result.config.away_team]:
        for rec in loader.get_club_players(club):
            ledger_names.append(rec.name)
    if season_state.begin_fixture(
        matchday, result.config.home_team, result.config.away_team,
        sorted(set(ledger_names)),
    ):
        print(f"\n  🔁 Re-run detected for MD{matchday} — previous recording "
              f"rolled back; new result replaces it (roles stay at "
              f"{matchday} played).")
    season_state.record_team_result(
        result.config.home_team,
        result.config.away_team,
        result.home_goals,
        result.away_goals,
    )

    played_names: set[str] = set()
    try:
        for player in all_players:
            # ``acc.stats`` is exposed with loose, unparameterized dictionary
            # types at runtime; coerce it to a plain mapping before reading a
            # player's stat payload so static analysis stops flagging the
            # partially-unknown dict entries.
            stats_map = cast(dict[str, Any], getattr(acc, "stats", {}))
            raw_stats = stats_map.get(player.name)
            if not isinstance(raw_stats, dict):
                continue
            s: dict[str, Any] = {str(k): v for k, v in raw_stats.items()}
            minutes_played = int(s.get("minutes_played", 0) or 0)
            actually_entered = (
                player.name in starter_names
                or bool(getattr(player, "_entered_pitch", False))
                or minutes_played > 0
            )
            if not actually_entered:
                # A named substitute who never entered is not a match
                # participant: do not update form, fatigue, cards, or
                # season_matches for an unused bench player.
                continue
            stamina_state = sub_controller.stamina.get(player.name)
            ending_stamina = stamina_state.current_stamina if stamina_state else 100.0
            season_state.record_post_match(
                name=player.name,
                rating=s.get("rating", 6.0),
                goals=s.get("goals", 0),
                minutes_played=minutes_played,
                ending_stamina=ending_stamina,
                yellow=s.get("yellow_cards", 0) > 0,
                red=s.get("red_cards", 0) > 0,
                injured=stamina_state.is_injured if stamina_state else False,
                injury_type=stamina_state.injury_type if stamina_state else "none",
                matches_out=stamina_state.matches_out() if stamina_state else 0,
                match_date=match_date,
                assists=s.get("assists", 0),
            )
            played_names.add(player.name)

        # Tick down bans/injuries for everyone in both clubs who DIDN'T play
        all_club_names: list[str] = []
        for club in [result.config.home_team, result.config.away_team]:
            for rec in loader.get_club_players(club):
                all_club_names.append(rec.name)

        # `SeasonState.advance_matchday()` is typed loosely in the runtime module,
        # so force the call through `Any` to keep static analysis happy while
        # preserving the real runtime behavior.
        cast(Any, season_state).advance_matchday(
            all_club_names,
            played_names,
            match_date,
        )
    finally:
        season_state.save()


def _resolve_fixture_info(loader: RosterLoader) -> dict[str, Any]:
    """
    Pull fixture metadata (Start Time, Venue, Capacity) for this matchday's
    fixture from the Excel FIXTURES sheet. Returns a dict with keys
    start_time / venue / capacity (None when not found / not playable).
    """
    try:
        from weather_physics import load_fixture_info as _load_fixture_info
        excel_path = getattr(loader, "excel_path", None)
        if not excel_path:
            return {"start_time": None, "venue": None, "capacity": None}
        info = _load_fixture_info(excel_path, MATCHDAY, HOME_TEAM, AWAY_TEAM)
        return {
            "start_time": info.get("start_time"),
            "venue": info.get("venue"),
            "capacity": info.get("capacity"),
        }
    except Exception:
        return {"start_time": None, "venue": None, "capacity": None}


def _activate_weather(
    fixture_info: dict[str, Any], weather_str: str, enabled: bool
) -> dict[str, Any]:
    """
    Resolve the WeatherCondition to use for this match.
      - disabled: return {"condition": None, "enabled": False} (zero regression).
      - enabled:  derive the condition from the fixture's real climate when no
                  explicit weather override is set; otherwise map the string.
    Activates the WeatherPhysics engine and returns the resolved metadata.
    """
    start_time = fixture_info["start_time"]
    if not enabled:
        return {"condition": None, "enabled": False, "start_time": start_time}

    from weather_physics import (
        WeatherPhysics, WeatherCondition, resolve_real_weather,
    )

    if weather_str and str(weather_str).strip().lower() not in ("", "auto", "real"):
        condition = WeatherCondition.from_string(weather_str, temperature_c=15.0)
    else:
        # Real weather first (Open-Meteo, no key), seasonal climate as fallback.
        # The club name (HOME_TEAM) is the location key — each club owns a ground.
        condition = resolve_real_weather(
            venue=HOME_TEAM or (fixture_info["venue"] or f"{HOME_TEAM} Stadium"),
            match_date=MATCH_DATE,
            start_time=start_time,
        )

    WeatherPhysics.set_active_weather(condition, enabled=True)
    return {"condition": condition, "enabled": True, "start_time": start_time}


# ── MAIN ─────────────────────────────────────────────────────────────

def _fixture_already_played() -> list[str]:
    """Return evidence that the configured fixture was already played.

    Checks four independent sources (any single hit blocks the run):
      1. fixture_ledger key in season_state.json (replace-semantics record)
      2. raw match package in plofa_output (engine export)
      3. match row in alltime.db (warehouse)
      4. matches_processed entry in season_stats.json (accumulator)
    Read-only: never mutates state. A missing/unreadable source abstains
    (counts as "not played") — only positive evidence blocks.
    """
    hits: list[str] = []
    pair = {HOME_TEAM, AWAY_TEAM}

    # 1. fixture ledger
    try:
        with open(SEASON_STATE_FILE, encoding="utf-8") as f:
            ledger = json.load(f).get("fixture_ledger", {}) or {}
        key = f"{MATCHDAY}|{HOME_TEAM}|{AWAY_TEAM}"
        if key in ledger:
            hits.append(f"fixture_ledger key {key!r} in {SEASON_STATE_FILE}")
    except Exception:
        pass

    # 2. raw match packages
    try:
        root = Path(OUTPUTS_DIR)
        if root.is_dir():
            for jp in sorted(root.rglob("*.json")):
                try:
                    doc = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                m = doc.get("match", {}) or {}
                try:
                    md = int(m.get("matchday"))
                except (TypeError, ValueError):
                    continue
                if (str(m.get("season")) == SEASON and md == MATCHDAY
                        and {str(m.get("home_team")), str(m.get("away_team"))} == pair):
                    hits.append(f"match package {jp.name}")
                    break
    except Exception:
        pass

    # 3. warehouse DB row
    try:
        import sqlite3
        conn = sqlite3.connect("alltime.db")
        try:
            row = conn.execute(
                """SELECT m.match_id FROM matches m
                   JOIN teams h ON h.team_id = m.home_team_id
                   JOIN teams a ON a.team_id = m.away_team_id
                   WHERE m.season = ? AND m.matchday = ?
                     AND ((h.name = ? AND a.name = ?)
                       OR (h.name = ? AND a.name = ?))""",
                (SEASON, MATCHDAY, HOME_TEAM, AWAY_TEAM, AWAY_TEAM, HOME_TEAM),
            ).fetchone()
        finally:
            conn.close()
        if row:
            hits.append(f"alltime.db match row (id {row[0]})")
    except Exception:
        pass

    # 4. season_stats accumulator record
    try:
        with open("season_stats.json", encoding="utf-8") as f:
            processed = json.load(f).get("matches_processed", []) or []
        for e in processed:
            if (isinstance(e, dict) and e.get("matchday") == MATCHDAY
                    and {e.get("home"), e.get("away")} == pair):
                hits.append(
                    f"season_stats.json matches_processed MD{MATCHDAY} "
                    f"{e.get('home')} vs {e.get('away')} ({e.get('score')})"
                )
                break
    except Exception:
        pass

    return hits


def _enforce_single_play(force_replay: bool) -> None:
    """Refuse to re-simulate an already-played fixture.

    Must be called first in run(), before any state is read for mutation.
    Exits with code 3 unless force_replay is set (deliberate replay keeps
    the ledger's replace-instead-of-append semantics).
    """
    hits = _fixture_already_played()
    if not hits:
        return
    print(f"\n  ⛔ Fixture already played: {HOME_TEAM} vs {AWAY_TEAM} "
          f"(MD{MATCHDAY}, {SEASON}). Refusing to re-simulate.")
    for h in hits:
        print(f"     • {h}")
    if force_replay:
        print("     ⚠️  --force-replay given: proceeding; "
              "ledger replace-semantics apply.")
        return
    print("     Pass --play --force-replay to deliberately replay it.")
    sys.exit(3)


def run(force_replay: bool = False):
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    # ── Already-played guard: runs before ANY state is touched ──
    _enforce_single_play(force_replay)

    # ── Fixture metadata (Start Time / Venue / Capacity) from Excel ──
    _fixture_info = _resolve_fixture_info(get_loader())
    _weather_meta = _activate_weather(_fixture_info, WEATHER, WEATHER_ENABLED)
    START_TIME = _weather_meta["start_time"]

    venue = VENUE if VENUE else (_fixture_info["venue"] or f"{HOME_TEAM} Stadium")
    # Prefer the user's explicit CAPACITY; fall back to the Excel fixture
    # capacity only when the venue itself is auto-resolved from Excel.
    _capacity = CAPACITY
    if not VENUE and _fixture_info["capacity"]:
        _capacity = _fixture_info["capacity"]

    # ── Auto-assign referee ──────────────────────────────────
    ref_mgr = RefereeManager()
    chosen_ref = ref_mgr.assign(MATCHDAY, HOME_TEAM, AWAY_TEAM, force_ref=FORCE_REF)
    REFEREE = chosen_ref.name
    STRICTNESS = chosen_ref.strictness

    print(f"\n{'═' * 64}")
    print(f"  PLOFA {SEASON} — Matchday {MATCHDAY}")
    print(f"  {HOME_TEAM}  vs  {AWAY_TEAM}")
    print(f"  {MATCH_DATE.strftime('%A %d %B %Y')}  |  {venue}  |  KO {START_TIME if START_TIME else 'TBC'}")
    print(f"  Referee: {REFEREE}  (strictness: {STRICTNESS})")
    _weather_label = WEATHER
    if _weather_meta["enabled"] and _weather_meta["condition"] is not None:
        _weather_label = _weather_meta["condition"].summary()
    print(f"  Weather: {_weather_label}{'  |  PHYSICS ON' if _weather_meta['enabled'] else ''}{'  |  DERBY' if IS_DERBY else ''}")
    print(f"{'═' * 64}")

    # ── Load Excel roster ─────────────────────────────────────
    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in [HOME_TEAM, AWAY_TEAM]:
        if team not in clubs:
            print(f"\n  ❌ Team '{team}' not found in Excel.")
            print(f"     Available clubs: {', '.join(clubs)}")
            sys.exit(1)

    # ── Managers (assigned from pool; auto-generated for promoted) ──
    from manager_profile import ManagerPool
    style_lookup: dict[str, str] = {}
    for name, entry in TEAM_CATALOG.items():
        style_obj = entry.get("style")
        if style_obj is not None and hasattr(style_obj, "value"):
            style_lookup[name] = style_obj.value
        elif isinstance(style_obj, str):
            style_lookup[name] = style_obj
        else:
            style_lookup[name] = ""
    mgr_pool = ManagerPool(clubs=clubs, style_lookup=style_lookup)
    home_mgr = mgr_pool.manager_for(HOME_TEAM)
    away_mgr = mgr_pool.manager_for(AWAY_TEAM)
    if home_mgr is None or away_mgr is None:
        missing_managers = [
            team for team, manager in ((HOME_TEAM, home_mgr), (AWAY_TEAM, away_mgr))
            if manager is None
        ]
        raise RuntimeError(
            f"No manager assigned for: {', '.join(missing_managers)}"
        )
    print(f"  🧑‍💼 {HOME_TEAM}: {home_mgr.name}  ({home_mgr.tactical_philosophy})  "
          f"[{home_mgr.job_status()}]")
    print(f"  🧑‍💼 {AWAY_TEAM}: {away_mgr.name}  ({away_mgr.tactical_philosophy})  "
          f"[{away_mgr.job_status()}]")

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
    _print_availability_report(HOME_TEAM, home_avail, cast(_SquadResult, home_raw))
    _print_availability_report(AWAY_TEAM, away_avail, cast(_SquadResult, away_raw))

    # ── Team profiles (styles) ────────────────────────────────
    HOME_STYLE = _resolve_team_profile(
        HOME_TEAM, home_formation, is_home=True,
        club_override=season_state.philosophies.get(HOME_TEAM),
        manager_label=home_mgr.tactical_philosophy,
    )
    AWAY_STYLE = _resolve_team_profile(
        AWAY_TEAM, away_formation, is_home=False,
        club_override=season_state.philosophies.get(AWAY_TEAM),
        manager_label=away_mgr.tactical_philosophy,
    )

    print(f"\n  🏟️  {HOME_TEAM} [{home_formation}] — {HOME_STYLE.style.value} / {HOME_STYLE.playing_style.value}")
    print(f"  ✈️  {AWAY_TEAM} [{away_formation}] — {AWAY_STYLE.style.value} / {AWAY_STYLE.playing_style.value}")

    # ── Build PlayerProfile squads via SquadBuilder ───────────
    # The imported builder exposes a partially-typed signature in the IDE, so
    # we normalize the call to avoid false-positive unknown-member diagnostics.
    home_squad = cast(
        _BuiltSquad,
        cast(Any, SquadBuilder).build(
            team_name=HOME_TEAM,
            starters=cast(list[Any], home_raw["starters"]),
            substitutes=cast(list[Any], home_raw["substitutes"]),
            team_superstars=cast(list[str], home_raw["superstars"]),
            set_piece_takers=cast(list[str], home_raw["sp_takers"]),
        ),
    )
    away_squad = cast(
        _BuiltSquad,
        cast(Any, SquadBuilder).build(
            team_name=AWAY_TEAM,
            starters=cast(list[Any], away_raw["starters"]),
            substitutes=cast(list[Any], away_raw["substitutes"]),
            team_superstars=cast(list[str], away_raw["superstars"]),
            set_piece_takers=cast(list[str], away_raw["sp_takers"]),
        ),
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

    # ── Pre-match fatigue briefing: let the manager READ their own
    #    fatigue before kickoff — true projected start %, carryover drain,
    #    seasonal load, rotation calls, and bench readiness. Shows the real
    #    number, flagging [floored→70%] where the engine's clamp would
    #    otherwise hide genuine exhaustion. ───────────────────────
    season_state.print_fatigue_briefing(HOME_TEAM, home_squad["starters"],
                                        bench=home_squad["substitutes"], match_date=MATCH_DATE)
    season_state.print_fatigue_briefing(AWAY_TEAM, away_squad["starters"],
                                        bench=away_squad["substitutes"], match_date=MATCH_DATE)

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
        stadium_capacity=_capacity,
        referee=REFEREE,
        referee_strictness=STRICTNESS,
        is_derby=IS_DERBY,
        weather=(_weather_meta["condition"] if _weather_meta["enabled"] else WEATHER),
        start_time=START_TIME,
        weather_enabled=_weather_meta["enabled"],
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
    # Per-manager substitution patience (overrides the flat constant).
    sub_controller.set_manager_stubbornness(HOME_TEAM, home_mgr.stubbornness())
    sub_controller.set_manager_stubbornness(AWAY_TEAM, away_mgr.stubbornness())

    # Register pre-planned tactical sub minutes.
    # Format: {player_OFF_name: minute_they_come_off}
    # Built from starters' sub_out_minute and from bench sub_in_minute
    # (inferring the player going off by finding a starter in the same
    # position group who doesn't already have a sub scheduled).
    tactical_schedule: Dict[str, int] = {}
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
            # Find a starter in the same position group who isn't already
            # scheduled to come off, and pair them with this bench player.
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

    # ── Simulate ──────────────────────────────────────────────
    print(f"\n  ⚽ Simulating...\n")
    # ── Cognition (TOLAND mind layer) ─────────────────────────
    # Hydrate every player's mind from the season store (carrying last
    # matchday's temperament), register them, and point the event stream
    # at the observer live, before kickoff.
    if COGNITION:
        from brain_integration import set_cognition
        from cognition_brain import engage_mind_observer
        from cognition.mind import clear_minds, register_mind, PlayerMind
        clear_minds()
        for p in all_players_flat:
            saved = season_state.get_player_cognition(p.name)
            mind = (PlayerMind.from_state(saved) if saved else PlayerMind())
            register_mind(p.name, mind)
        set_cognition(True)
        engage_mind_observer()
    engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
    # Live brain-manager (since 2026-09-20 wired by default): the club's
    # persisted mind/memory drives posture/pressing/urgency via the engine's
    # USE_MANAGER_BRAIN hooks.  ManagerProfile pool managers (home_mgr /
    # away_mgr) keep handling club lifecycle (records/sack risk) separately.
    from manager_profile import brain_manager_for, save_live_manager
    home_brain_mgr = brain_manager_for(HOME_TEAM)
    away_brain_mgr = brain_manager_for(AWAY_TEAM)
    # MatchEngine's external type hints leave the player list element type
    # unresolved; the squads have already been normalized to PlayerProfiles.
    cast(Any, engine).set_squad(
        HOME_TEAM,
        home_squad["starters"],
        home_squad["substitutes"],
    )
    cast(Any, engine).set_squad(
        AWAY_TEAM,
        away_squad["starters"],
        away_squad["substitutes"],
    )
    cast(Any, engine).set_stamina_controller(sub_controller)
    cast(Any, engine).set_managers(home_manager=home_brain_mgr,
                               away_manager=away_brain_mgr)

    result = engine.simulate()
    print(result.summary())

    # ── Harvest cognition back into the season store ──────────
    # Persist each mind's accumulated memory weights + temperament so the
    # next matchday can re-hydrate them.  Runs BEFORE _persist_post_match
    # so its final save() also writes the cognition section.
    if COGNITION:
        from brain_integration import set_cognition
        from cognition_brain import disengage_mind_observer
        from cognition.mind import get_mind, clear_minds
        for p in all_players_flat:
            mind = get_mind(p.name)
            if mind is not None:
                season_state.set_player_cognition(p.name, mind.to_state())
        disengage_mind_observer()
        clear_minds()
        set_cognition(False)

    # Persist each club's live brain-manager mind/memory for the next
    # matchday (standalone match or season — the state save below also
    # writes the manager dir via save_live_manager).
    save_live_manager(home_brain_mgr, HOME_TEAM)
    save_live_manager(away_brain_mgr, AWAY_TEAM)

    # ── Big 6 teams (clubs with highest market values) ─────────
    # These draw bigger crowds and command higher ticket prices.
    # Auto-detected from actual squad market values in the DB.
    big6_teams: set[str] = set()
    try:
        all_club_values: list[tuple[str, float]] = []
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
        big6_teams = {c[0] for c in all_club_values[:6]}
        print(f"\n  🏆 Big 6 teams (by market value): {', '.join(sorted(big6_teams))}")
    except Exception:
        big6_teams = {"Pearls", "Claw", "Uditon", "Lige-8", "Triumpher", "Natrican"}
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
        # Exporter expects each team value to be a player list; the squad
        # builder's mapping also contains starters/substitutes metadata.
        all_players=cast(Any, {HOME_TEAM: home_squad, AWAY_TEAM: away_squad}),
        home_color=home_color,
        away_color=away_color,
        sub_controller=sub_controller,
        big6_teams=big6_teams,
        standings=season_state.standings_info(),
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
        stats = cast(Dict[str, Dict[str, Any]], cast(Any, acc).stats)
        for name in souls_attached:
            s = stats.get(name)
            if s:
                soul = SOUL_PLAYERS[name]
                print(f"  {name} ({soul.profile.label})")
                print(f"    Goals: {s['goals']}  Assists: {s['assists']}  "
                      f"Rating: {s['rating']}")
                print(f"    xG: {s['xg']:.3f}  xA: {s['xa']:.3f}")
                print(f"    Dribbles: {s['dribbles_comp']}/{s['dribbles_att']}  "
                      f"Shot Assists: {s['shot_assists']}  "
                      f"Carries: {s['carries']}")

    # ── Record referee assignment for rotation tracking ──────
    ref_mgr.record_assignment(chosen_ref, MATCHDAY, HOME_TEAM, AWAY_TEAM)
    ref_mgr.save()

    # ── Record manager results for job security ──────────────
    home_pts, away_pts = 3, 0
    if result.home_goals == result.away_goals:
        home_pts = away_pts = 1
    elif result.home_goals < result.away_goals:
        home_pts, away_pts = 0, 3
    # Expected-points proxy: a neutral expectation is 1.0 for a tie, ~1.5
    # if a team overperformed this matchday, ~0.5 if underperformed. We
    # anchor xP to the actual xG differential so it's emergent, not random.
    home_xg, away_xg = result.home_xg, result.away_xg
    home_xp = 1.0 + (home_xg - away_xg) * 0.6
    away_xp = 1.0 + (away_xg - home_xg) * 0.6
    home_mgr.record_result(MATCHDAY, home_pts, max(0.0, home_xp))
    away_mgr.record_result(MATCHDAY, away_pts, max(0.0, away_xp))
    mgr_pool.save()

    print(f"\n  ✅ Done. Output → {output_path}/")
    print(f"  📊 Season state updated. Injuries/suspensions carry to next matchday.\n")
    print(f"  🟨 {ref_mgr.summary()}")
    print(f"  🧑‍💼 {HOME_TEAM} manager risk: {home_mgr.sack_risk():.0%}  "
          f"({home_mgr.job_status()})")
    print(f"  🧑‍💼 {AWAY_TEAM} manager risk: {away_mgr.sack_risk():.0%}  "
          f"({away_mgr.job_status()})")


# ── Entry guard ─────────────────────────────────────────────────────
# This runner OVERWRITES season_state.json / referee_state.json /
# manager_state.json — a stray or unintended invocation corrupts the
# season ledger (a fixture may only be played once). By design it does
# nothing with bare args, `-h`, or `--help`; ONLY `--play` executes a
# fixture. No agent/assistant should ever pass `--play`.
_USAGE = """
  PLOFA AUTOMATED MATCH RUNNER — Trevor only.

  Usage:
    python auto_run_match.py            print this help (no match is run)
    python auto_run_match.py --help     print this help (no match is run)
    python auto_run_match.py --play     RUN THE CONFIGURED FIXTURE

  A fixture may only be played once. This script overwrites
  season_state.json, referee_state.json and manager_state.json.
  If any of (fixture_ledger, plofa_output package, alltime.db row,
  season_stats record) shows the fixture as played, the run is REFUSED
  unless --force-replay is also passed:
    python auto_run_match.py --play --force-replay
  A forced replay replaces (never appends) via the fixture ledger.
  Use run_match.py for non-canonical/test simulations.
"""


def _main() -> None:
    args = [a for a in sys.argv[1:] if a]
    wants_help = (not args) or any(a in ("-h", "--help") for a in args)
    if wants_help:
        print(_USAGE)
        sys.exit(0)
    if args == ["--play"] or args == ["--play", "--force-replay"]:
        run(force_replay="--force-replay" in args)
        return
    print("✋ Unexpected argument(s):", " ".join(args), file=sys.stderr)
    print("   To actually run the configured fixture, use exactly: --play", file=sys.stderr)
    print("   To deliberately replay a played fixture: --play --force-replay", file=sys.stderr)
    print("   See --help for details.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    _main()
