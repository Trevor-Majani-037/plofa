"""
PLOFA 26/27 — SEASON MANAGER
================================
season_manager.py

Closes the two biggest structural gaps from the analyst feedback:

    "League Management: ❌ 10% — Table, fixtures, standings, automatic
     accumulation"
    "AI Management: ❌ 0% — Squad selection, tactical decisions, transfers"

and directly answers the "Fixtures Problem" quoted in the feedback doc:
    "I already have fixtures and dates and times, so run the code at the
     exact time it was predestined to end, then accumulate the stats
     myself... This is a reasonable approach for now, but you're right to
     want automation."

What this module gives you that manual matchday-by-matchday running
didn't:
    1. A real fixture list + league table that updates itself.
    2. A SeasonState JSON file that persists player form, fatigue,
       injuries, cards, chemistry and season-minutes BETWEEN matchdays —
       so run_match.py stops being the only source of truth and
       CHECK_AVAILABILITY stops being a manual toggle.
    3. A lightweight AI squad selector: given availability + form +
       overall rating, pick the strongest available XI automatically
       (used for squads you haven't hand-written a lineup for, or as a
       sanity check against your own selections).
    4. A rotation policy: flags players who are being overplayed
       (season_minutes vs. games remaining) so fatigue-driven squad
       rotation isn't purely manual either.

This module is deliberately independent of run_match.py's structure —
you keep writing your weekly lineups exactly as you do now. This just
gives you a second mode: `run_matchday()` / `run_full_season()`, which
drives run_match.py's underlying engine off a fixture list instead of
you re-running the script by hand every week.
"""

from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any

from player_dna import PlayerProfile, SquadBuilder
from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from squad_manager import SubstitutionController, AvailabilityChecker, AvailabilityStatus
from exporter import PLOFAExporter


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@dataclass
class Fixture:
    matchday: int
    home_team: str
    away_team: str
    match_date: date
    venue: str = "TBC"
    played: bool = False
    home_goals: int = 0
    away_goals: int = 0
    is_derby: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["match_date"] = self.match_date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Fixture":
        d = dict(d)
        d["match_date"] = date.fromisoformat(d["match_date"])
        return cls(**d)


class FixtureList:
    """A season's full fixture list. Supports round-robin auto-generation
    or hand-authored fixtures (what most people already have, per the
    feedback doc — this just gives that list a home to live in)."""

    def __init__(self, fixtures: Optional[List[Fixture]] = None):
        self.fixtures: List[Fixture] = fixtures or []

    @classmethod
    def round_robin(cls, teams: List[str], start_date: date,
                     days_between: int = 7, double_round: bool = True) -> "FixtureList":
        """Classic circle-method round robin — home/away balanced."""
        teams = list(teams)
        if len(teams) % 2 == 1:
            teams.append("BYE")
        n = len(teams)
        rounds = []
        fixed = teams[0]
        rest = teams[1:]
        for r in range(n - 1):
            pairs = list(zip([fixed] + rest[:n // 2 - 1], reversed(rest[n // 2 - 1:])))
            rounds.append(pairs)
            rest = [rest[-1]] + rest[:-1]

        fixtures = []
        md = 1
        current_date = start_date
        for rnd in rounds:
            for home, away in rnd:
                if "BYE" in (home, away):
                    continue
                fixtures.append(Fixture(matchday=md, home_team=home, away_team=away,
                                         match_date=current_date))
            md += 1
            current_date += timedelta(days=days_between)

        if double_round:
            second_leg_start_md = md
            for i, rnd in enumerate(rounds):
                for home, away in rnd:
                    if "BYE" in (home, away):
                        continue
                    fixtures.append(Fixture(matchday=second_leg_start_md + i,
                                             home_team=away, away_team=home,
                                             match_date=current_date))
                current_date += timedelta(days=days_between)

        return cls(fixtures)

    def matchday(self, md: int) -> List[Fixture]:
        return [f for f in self.fixtures if f.matchday == md]

    def unplayed(self) -> List[Fixture]:
        return [f for f in self.fixtures if not f.played]

    def next_matchday(self) -> Optional[int]:
        unplayed = self.unplayed()
        return min((f.matchday for f in unplayed), default=None)

    def mark_played(self, home_team: str, away_team: str, matchday: int,
                     home_goals: int, away_goals: int):
        for f in self.fixtures:
            if (f.matchday == matchday and f.home_team == home_team
                    and f.away_team == away_team):
                f.played = True
                f.home_goals = home_goals
                f.away_goals = away_goals
                return

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump([fx.to_dict() for fx in self.fixtures], f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FixtureList":
        with open(path) as f:
            data = json.load(f)
        return cls([Fixture.from_dict(d) for d in data])


# ─────────────────────────────────────────────
# LEAGUE TABLE
# ─────────────────────────────────────────────

@dataclass
class TeamRecord:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    form: List[str] = field(default_factory=list)   # last 5: "W"/"D"/"L"

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


class LeagueTable:
    def __init__(self, teams: List[str]):
        self.records: Dict[str, TeamRecord] = {t: TeamRecord(team=t) for t in teams}

    def add_result(self, home: str, away: str, hg: int, ag: int):
        h, a = self.records[home], self.records[away]
        h.played += 1; a.played += 1
        h.goals_for += hg; h.goals_against += ag
        a.goals_for += ag; a.goals_against += hg
        if hg > ag:
            h.won += 1; h.points += 3; a.lost += 1
            h.form.append("W"); a.form.append("L")
        elif hg < ag:
            a.won += 1; a.points += 3; h.lost += 1
            a.form.append("W"); h.form.append("L")
        else:
            h.drawn += 1; a.drawn += 1; h.points += 1; a.points += 1
            h.form.append("D"); a.form.append("D")
        h.form = h.form[-5:]
        a.form = a.form[-5:]

    def standings(self) -> List[TeamRecord]:
        return sorted(
            self.records.values(),
            key=lambda r: (r.points, r.goal_diff, r.goals_for),
            reverse=True,
        )

    def print_table(self):
        print(f"\n  {'#':<3}{'Team':<22}{'P':>3}{'W':>3}{'D':>3}{'L':>3}{'GF':>4}{'GA':>4}{'GD':>5}{'Pts':>5}  Form")
        print("  " + "─" * 65)
        for i, r in enumerate(self.standings(), 1):
            print(f"  {i:<3}{r.team:<22}{r.played:>3}{r.won:>3}{r.drawn:>3}{r.lost:>3}"
                  f"{r.goals_for:>4}{r.goals_against:>4}{r.goal_diff:>+5}{r.points:>5}   "
                  f"{''.join(r.form)}")

    def export_csv(self, path: str):
        import pandas as pd
        rows = []
        for i, r in enumerate(self.standings(), 1):
            rows.append({
                "Pos": i, "Team": r.team, "P": r.played, "W": r.won, "D": r.drawn,
                "L": r.lost, "GF": r.goals_for, "GA": r.goals_against,
                "GD": r.goal_diff, "Pts": r.points, "Form": "".join(r.form),
            })
        pd.DataFrame(rows).to_csv(path, index=False)


# ─────────────────────────────────────────────
# Realistic recovery days by injury type (based on medical data)
# Each range is (min_days, max_days) — actual value is randomized
# within the range for realistic variation.
# ─────────────────────────────────────────────

INJURY_RECOVERY_DAYS: Dict[str, Tuple[int, int]] = {
    "knock":         (1, 3),
    "cramp":         (0, 1),
    "muscular":      (7, 21),
    "hamstring":     (14, 28),
    "calf":          (10, 21),
    "groin":         (7, 21),
    "ankle_sprain":  (7, 42),
    "knee_sprain":   (14, 56),
    "acl":           (180, 365),
    "fracture":      (30, 90),
    "concussion":    (7, 14),
    "shoulder":      (14, 42),
    "hip":           (7, 28),
    "thigh":         (10, 28),
    "back":          (7, 28),
    "none":          (0, 0),
}

DEFAULT_RECOVERY_RANGE = (7, 21)


# ─────────────────────────────────────────────
# PERSISTED SEASON STATE
# The piece that actually closes the "manual accumulation" gap: player
# form/fatigue/injuries/cards/chemistry now live in a file that every
# matchday reads and updates, instead of resetting to defaults each time
# run_match.py starts a fresh Python process.
# ─────────────────────────────────────────────

class SeasonState:
    """
    JSON-backed persistence for everything that should carry over between
    matchdays: per-player form/fatigue/injury/cards, per-team chemistry,
    and season-cumulative minutes for rotation/fatigue-management decisions.
    """

    def __init__(self, season: str, path: str = "season_state.json"):
        self.season = season
        self.path = path
        self.players: Dict[str, Dict[str, Any]] = {}   # name -> state dict
        self.chemistry: Dict[str, Dict[str, Any]] = {}  # team -> chemistry dict
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.players = data.get("players", {})
            self.chemistry = data.get("chemistry", {})
        except (json.JSONDecodeError, OSError) as e:
            # Corrupt/truncated state file (e.g. from a crash before the
            # atomic-save fix). Back it up so nothing is silently lost, then
            # start fresh instead of crashing the whole match run.
            import shutil
            backup = f"{self.path}.corrupt-{int(__import__('time').time())}"
            try:
                shutil.copy2(self.path, backup)
            except OSError:
                pass
            print(f"  ⚠️  Season state file unreadable ({e}). "
                  f"Backed up to {backup}; starting fresh.")
            self.players = {}
            self.chemistry = {}

    def save(self):
        """Atomically persist season state.

        Writes to a temp file in the same directory then renames over the
        real path, so a crash mid-write can never leave season_state.json
        truncated or corrupt (the old in-place 'w' mode could and did).
        """
        data = {"season": self.season, "players": self.players,
                "chemistry": self.chemistry}
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # os.replace is atomic on both Windows and POSIX.
        os.replace(tmp_path, self.path)

    @staticmethod
    def _recovery_days(injury_type: str) -> int:
        """Return random recovery days within the realistic range for this injury type."""
        lo, hi = INJURY_RECOVERY_DAYS.get(injury_type.lower(), DEFAULT_RECOVERY_RANGE)
        return random.randint(lo, hi)

    # ── PLAYER STATE ──────────────────────────────────────────

    def get_player_state(self, name: str) -> Dict[str, Any]:
        return self.players.get(name, {
            "confidence": 50.0, "fatigue_level": 0.0, "starting_stamina": 100.0,
            "is_injured": False, "injury_type": "none", "matches_remaining_out": 0,
            "yellow_cards_last_6": [], "red_card_ban": 0,
            "season_minutes": 0, "season_matches": 0, "recent_ratings": [6.0] * 5,
            "recent_goals": [0] * 5,
            "injury_history": [],
            "injury_date": None,
            "recovery_days": 0,
            "expected_return_date": None,
        })

    def apply_pre_match(self, dna, name: str, match_date: Optional[date] = None):
        """Called before kickoff: hydrate a fresh PlayerDNA with this
        player's persisted season state instead of factory defaults.
        If match_date is provided, checks if the player has recovered."""
        s = self.get_player_state(name)

        # Auto-clear injury if expected return date has passed
        if match_date and s.get("expected_return_date"):
            ret = s["expected_return_date"]
            if isinstance(ret, str):
                ret = date.fromisoformat(ret)
            if match_date > ret:
                s["is_injured"] = False
                s["injury_type"] = "none"
                s["injury_date"] = None
                s["recovery_days"] = 0
                s["expected_return_date"] = None
                self.players[name] = s

        dna.form.confidence = s["confidence"]
        dna.form.fatigue_level = s["fatigue_level"]
        dna.form.recent_ratings = list(s["recent_ratings"])
        dna.form.recent_goals = list(s["recent_goals"])
        dna.form.is_injured = s["is_injured"]
        dna.form.injury_type = s["injury_type"]
        dna.form.matches_remaining_out = s.get("matches_remaining_out", 0)
        dna.season_minutes = s["season_minutes"]
        dna.season_matches = s["season_matches"]
        return s.get("starting_stamina", 100.0)

    def record_post_match(self, name: str, rating: float, goals: int,
                           minutes_played: int, ending_stamina: float,
                           yellow: bool = False, red: bool = False,
                           injured: bool = False, injury_type: str = "none",
                           matches_out: int = 0,
                           match_date: Optional[date] = None):
        """
        Records what happened FOR A PLAYER WHO ACTUALLY PLAYED this match:
        form, fatigue, season load, and any NEW card/injury picked up.

        When an injury is recorded with a match_date, it calculates realistic
        recovery days and an expected return date. The player is unavailable
        until current_match_date > expected_return_date (checked in
        is_available()). Every injury is appended to injury_history.

        Deliberately does NOT tick down existing suspensions/injuries —
        see advance_matchday() for why. A suspended player doesn't play,
        so if ban-serving lived here, it would never fire and the ban
        would never end. (Found by tests.py::test_season_state_
        suspension_logic — the yellow-card version of this bug cleared
        itself in the same call that triggered it, silently never
        blocking anyone; the red-card/injury version was subtler: it
        just never ran at all for the suspended player it was supposed
        to apply to.)
        """
        s = self.get_player_state(name)

        s["recent_ratings"] = (s["recent_ratings"] + [rating])[-5:]
        s["recent_goals"] = (s["recent_goals"] + [goals])[-5:]

        if rating >= 7.5:
            s["confidence"] = min(100, s["confidence"] + 8)
        elif rating >= 7.0:
            s["confidence"] = min(100, s["confidence"] + 4)
        elif rating < 5.5:
            s["confidence"] = max(0, s["confidence"] - 8)

        # Recovery between matches (assume ~6 days til next match)
        fatigue_now = 100.0 - ending_stamina
        recovered = min(fatigue_now, 6 * 10.0)
        s["fatigue_level"] = round(max(0.0, fatigue_now - recovered), 1)
        s["starting_stamina"] = round(min(100.0, ending_stamina + recovered), 1)

        s["season_minutes"] = s["season_minutes"] + minutes_played
        s["season_matches"] = s["season_matches"] + 1

        # NEW cards picked up THIS match (existing suspensions are ticked
        # down separately in advance_matchday, not here)
        s["yellow_cards_last_6"] = (s["yellow_cards_last_6"] + [1 if yellow else 0])[-6:]
        if red:
            s["red_card_ban"] = 1

        # NEW injury picked up this match — date-based recovery
        if injured:
            recovery_days = self._recovery_days(injury_type)
            s["is_injured"] = True
            s["injury_type"] = injury_type
            s["recovery_days"] = recovery_days
            s["matches_remaining_out"] = matches_out
            if match_date:
                s["injury_date"] = match_date.isoformat()
                ret = match_date + timedelta(days=recovery_days)
                s["expected_return_date"] = ret.isoformat()
                # Append to injury history
                history = s.get("injury_history", [])
                history.append({
                    "date": match_date.isoformat(),
                    "injury_type": injury_type,
                    "recovery_days": recovery_days,
                    "expected_return": ret.isoformat(),
                })
                s["injury_history"] = history

        self.players[name] = s

    def advance_matchday(self, squad_pool_names: List[str], played_names: set,
                          match_date: Optional[date] = None):
        """
        Call ONCE PER MATCHDAY for every player in both squads' full
        rosters (starters + substitutes), regardless of whether they
        actually played — this correctly serves suspensions and
        advances calendar time for injury recovery. A player left out
        because they're suspended is exactly the player this needs to
        run for.

        Injury recovery is now date-based: if match_date is provided,
        any injured player whose expected_return_date has passed is
        automatically cleared to fit.

        `played_names`: names already updated via record_post_match this
        matchday — skipped here to avoid double-processing.
        """
        for name in squad_pool_names:
            if name in played_names:
                continue
            s = self.get_player_state(name)

            if s.get("red_card_ban", 0) > 0:
                s["red_card_ban"] = max(0, s["red_card_ban"] - 1)

            if sum(s.get("yellow_cards_last_6", [])) >= 5:
                s["yellow_cards_last_6"] = []   # ban served this matchday

            # Date-based injury recovery
            if match_date and s.get("expected_return_date"):
                ret = s["expected_return_date"]
                if isinstance(ret, str):
                    ret = date.fromisoformat(ret)
                if match_date > ret:
                    s["is_injured"] = False
                    s["injury_type"] = "none"
                    s["injury_date"] = None
                    s["recovery_days"] = 0
                    s["expected_return_date"] = None

            self.players[name] = s

    def is_available(self, name: str, match_date: Optional[date] = None) -> Tuple[bool, str]:
        s = self.get_player_state(name)
        if s.get("red_card_ban", 0) > 0:
            return False, "suspended (red card)"
        if sum(s.get("yellow_cards_last_6", [])) >= 5:
            return False, "suspended (5 yellows)"
        if s.get("is_injured", False):
            if match_date and s.get("expected_return_date"):
                ret = s["expected_return_date"]
                if isinstance(ret, str):
                    ret = date.fromisoformat(ret)
                if match_date > ret:
                    return True, "fit (recovered)"
            ret_str = s.get("expected_return_date", "unknown")
            return False, (f"injured ({s.get('injury_type','injury')}, "
                          f"recovery until {ret_str})")
        return True, "fit"

    def rotation_flag(self, name: str, games_remaining_in_block: int = 4,
                       minutes_threshold_per_game: float = 75.0) -> bool:
        """True if a player's season load suggests they should be rested soon."""
        s = self.get_player_state(name)
        if s["season_matches"] == 0:
            return False
        avg = s["season_minutes"] / s["season_matches"]
        return avg >= minutes_threshold_per_game and s["fatigue_level"] > 55.0


# ─────────────────────────────────────────────
# AI SQUAD SELECTION
# ─────────────────────────────────────────────

class SquadSelector:
    """
    Automatic best-XI picker: availability × form × overall rating.
    Use it to auto-fill a lineup, or as a second opinion against a
    hand-written one from run_match.py.
    """

    FORMATION_442 = ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "LW", "RW", "ST", "ST"]
    FORMATION_433 = ["GK", "LB", "CB", "CB", "RB", "CDM", "CM", "CM", "LW", "RW", "ST"]
    FORMATION_4231 = ["GK", "LB", "CB", "CB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST"]

    @classmethod
    def pick_best_xi(
        cls,
        all_players: List[PlayerProfile],
        season_state: SeasonState,
        formation: Optional[List[str]] = None,
    ) -> Tuple[List[PlayerProfile], List[str]]:
        """
        Returns (starting_xi, notes). Greedy positional fill: for each slot
        in the formation, pick the best available (fit, not suspended)
        player for that position by a blended score of overall rating,
        current form (confidence), and freshness (inverse fatigue).
        """
        formation = formation or cls.FORMATION_433
        notes: List[str] = []
        pool = list(all_players)
        chosen: List[PlayerProfile] = []
        used_names = set()

        for slot_pos in formation:
            candidates = [p for p in pool if p.position == slot_pos and p.name not in used_names]
            if not candidates:
                # Fall back to adjacent-position candidates so a thin squad
                # still fields 11 rather than crashing the selector
                candidates = [p for p in pool if p.name not in used_names]
            scored = []
            for p in candidates:
                available, reason = season_state.is_available(p.name)
                if not available:
                    notes.append(f"{p.name} unavailable ({reason}) — excluded")
                    continue
                s = season_state.get_player_state(p.name)
                form_score = s["confidence"] / 100.0
                freshness = max(0.3, 1.0 - s["fatigue_level"] / 100.0)
                overall = p.dna.overall_rating / 100.0
                score = overall * 0.55 + form_score * 0.25 + freshness * 0.20
                scored.append((score, p))
            if not scored:
                notes.append(f"No fit candidate for {slot_pos} — squad may be thin")
                continue
            scored.sort(key=lambda t: t[0], reverse=True)
            best = scored[0][1]
            chosen.append(best)
            used_names.add(best.name)

        return chosen, notes

    @classmethod
    def rotation_suggestions(cls, all_players: List[PlayerProfile],
                              season_state: SeasonState) -> List[str]:
        flags = []
        for p in all_players:
            if season_state.rotation_flag(p.name):
                flags.append(f"{p.name} ({p.position}) — heavy season load, consider resting")
        return flags


# ─────────────────────────────────────────────
# MATCHDAY RUNNER
# The direct answer to the "Fixtures Problem" section of the feedback doc.
# ─────────────────────────────────────────────

class LeagueRunner:
    """
    Orchestrates fixtures + league table + season state together.
    Individual matches are still simulated by the exact same MatchEngine
    used in run_match.py — this just removes the manual "run the script,
    copy the score into a spreadsheet" step.
    """

    def __init__(self, teams: List[str], season: str = "26/27",
                 state_path: str = "season_state.json",
                 fixtures_path: str = "fixtures.json",
                 outputs_dir: str = "outputs"):
        self.season = season
        self.teams = teams
        self.table = LeagueTable(teams)
        self.state = SeasonState(season, state_path)
        self.outputs_dir = outputs_dir
        self.fixtures_path = fixtures_path

        if os.path.exists(fixtures_path):
            self.fixtures = FixtureList.load(fixtures_path)
        else:
            self.fixtures = FixtureList()

        # Rebuild table from any already-played fixtures on disk, so the
        # table is always a pure function of the fixture list — no
        # separate "manually accumulated" spreadsheet needed.
        for fx in self.fixtures.fixtures:
            if fx.played:
                self.table.add_result(fx.home_team, fx.away_team, fx.home_goals, fx.away_goals)

    def run_matchday(
        self,
        matchday: int,
        squads: Dict[str, Dict[str, List]],           # team -> {starters, substitutes}
        team_profiles: Dict[str, TeamProfile],
        referee: str = "League Referee",
        referee_strictness: float = 0.5,
        home_colors: Dict[str, str] = None,
        away_colors: Dict[str, str] = None,
        full_roster: Dict[str, List[str]] = None,
    ) -> List[Dict]:
        """
        Run every fixture scheduled for `matchday`. `squads` must already
        contain a PlayerProfile list per team (built via SquadBuilder, as
        in run_match.py) — this function hydrates season state into each
        player's DNA, simulates, exports, updates the table, and persists
        state, all in one call.

        `full_roster` (optional): {team_name: [all contracted player
        names]}, including anyone left OUT of this week's `squads` due to
        suspension/injury. Without this, a suspended player's ban never
        gets ticked down — record_post_match only fires for players who
        actually played, and a suspended player by definition didn't
        (found by tests.py::test_season_state_suspension_logic). If you
        pass the full club roster here, everyone left out gets their
        suspension/injury countdown advanced by exactly one matchday via
        SeasonState.advance_matchday(), same as real calendar time passing.
        """
        results = []
        fixtures_today = self.fixtures.matchday(matchday)
        if not fixtures_today:
            print(f"  ⚠️  No fixtures found for matchday {matchday}.")
            return results

        for fx in fixtures_today:
            if fx.played:
                continue

            home, away = fx.home_team, fx.away_team
            home_squad = squads[home]
            away_squad = squads[away]

            all_players = (home_squad["starters"] + home_squad["substitutes"] +
                           away_squad["starters"] + away_squad["substitutes"])
            starting_stamina: Dict[str, float] = {}
            for p in all_players:
                starting_stamina[p.name] = self.state.apply_pre_match(p.dna, p.name)

            config = MatchConfig(
                home_team=home, away_team=away, match_date=fx.match_date,
                matchday=matchday, season=self.season, competition="PLOFA",
                venue=fx.venue, referee=referee, referee_strictness=referee_strictness,
                is_derby=fx.is_derby,
            )

            engine = MatchEngine(config, team_profiles[home], team_profiles[away])
            engine.set_squad(home, home_squad["starters"], home_squad["substitutes"])
            engine.set_squad(away, away_squad["starters"], away_squad["substitutes"])

            sub_controller = SubstitutionController(
                home_team=home, away_team=away,
                home_subs_bench=home_squad["substitutes"],
                away_subs_bench=away_squad["substitutes"],
                home_style=team_profiles[home].style.value,
                away_style=team_profiles[away].style.value,
                manager_stubbornness=0.35,
            )
            engine.set_stamina_controller(sub_controller)

            result = engine.simulate()
            print(result.summary())

            folder = f"{home.replace(' ','_')}_vs_{away.replace(' ','_')}_MD{matchday:02d}"
            out_path = os.path.join(self.outputs_dir, folder)

            home_color = (home_colors or {}).get(home, "#003087")
            away_color = (away_colors or {}).get(away, "#C8102E")

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
                result=result, all_players={home: home_squad, away: away_squad},
                home_color=home_color,
                away_color=away_color,
                sub_controller=sub_controller,
            )
            exporter.export_all(out_path)

            # ── persist post-match state for every player ──────────
            played_names = set()
            for p in all_players:
                s = exporter.accumulator.stats.get(p.name)
                if not s:
                    continue
                stamina_state = sub_controller.stamina.get(p.name)
                ending_stamina = stamina_state.current_stamina if stamina_state else 100.0
                self.state.record_post_match(
                    p.name,
                    rating=s["rating"], goals=s["goals"],
                    minutes_played=s["minutes_played"],
                    ending_stamina=ending_stamina,
                    yellow=s["yellow_cards"] > 0, red=s["red_cards"] > 0,
                    injured=stamina_state.is_injured if stamina_state else False,
                    injury_type=stamina_state.injury_type if stamina_state else "none",
                    matches_out=stamina_state.matches_out() if stamina_state else 0,
                )
                played_names.add(p.name)

            # Anyone left out of this week's squad (suspended, injured,
            # rotated) still needs their ban/injury countdown to advance
            # by one matchday — see docstring above.
            if full_roster:
                roster_names = set(full_roster.get(home, [])) | set(full_roster.get(away, []))
                self.state.advance_matchday(list(roster_names), played_names)

            self.table.add_result(home, away, result.home_goals, result.away_goals)
            self.fixtures.mark_played(home, away, matchday, result.home_goals, result.away_goals)

            results.append({
                "home": home, "away": away,
                "score": f"{result.home_goals}-{result.away_goals}",
                "output": out_path,
            })

            # Per-match persistence: if the process crashes during a later
            # fixture, everything up to and including this one is already on
            # disk (atomic write), instead of losing the entire matchday.
            self.state.save()
            self.fixtures.save(self.fixtures_path)

        self.state.save()
        self.fixtures.save(self.fixtures_path)
        self.table.print_table()
        return results

    def availability_report(self, team: str, squad_players: List[PlayerProfile]) -> List[str]:
        lines = []
        for p in squad_players:
            ok, reason = self.state.is_available(p.name)
            if not ok:
                lines.append(f"  ❌ {p.name} ({p.position}) — {reason}")
        return lines or ["  ✅ Full squad available"]


# ─────────────────────────────────────────────
# WIRING GUIDE
# ─────────────────────────────────────────────

WIRING_GUIDE = """
USING season_manager.py ALONGSIDE run_match.py
═════════════════════════════════════════════════

You don't have to abandon your weekly run_match.py workflow — this slots
in underneath it:

    from season_manager import LeagueRunner, FixtureList
    from datetime import date

    teams = ["Hartwell City", "Thornfield United", ...]
    runner = LeagueRunner(teams, season="26/27")

    # First time only: generate or load your real fixture list
    if not runner.fixtures.fixtures:
        runner.fixtures = FixtureList.round_robin(teams, start_date=date(2026,8,16))
        runner.fixtures.save("fixtures.json")

    # Each week: build squads exactly as in run_match.py (SquadBuilder.build),
    # put them in a dict keyed by team name, then:
    squads = {"Hartwell City": hartwell, "Thornfield United": thornfield, ...}
    profiles = {"Hartwell City": HOME_STYLE, "Thornfield United": AWAY_STYLE, ...}
    runner.run_matchday(matchday=1, squads=squads, team_profiles=profiles)

Everything — form, fatigue, injuries, suspensions, season minutes — now
carries automatically into matchday 2 without you touching a spreadsheet.
`runner.availability_report(team, squad)` replaces the manual
CHECK_AVAILABILITY toggle in run_match.py.

For AI-picked lineups (e.g. an eleven you haven't hand-authored):
    from season_manager import SquadSelector
    xi, notes = SquadSelector.pick_best_xi(full_squad_pool, runner.state)
"""
