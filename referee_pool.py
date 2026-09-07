"""
PLOFA 26/27 — REFEREE POOL & AUTOMATIC ASSIGNMENT
==================================================
referee_pool.py

Automatic referee rotation system modelled on EPL Select Group 1 rules:

  - ~20 referees appointed per season
  - Each match gets exactly one referee
  - No referee officiates the same team more than ~3× per season
  - Referees get roughly equal game assignments across the season
  - A referee does not officiate back-to-back matchdays (rest rule)
  - Referees can be rested / rotated at the manager's discretion

Usage:
    from referee_pool import RefereeManager

    mgr = RefereeManager()                          # loads default pool
    ref = mgr.assign(matchday=5, home="Natrican", away="Avada Zenith")
    # ref.name, ref.strictness — plug straight into MatchConfig

    # After the match:
    mgr.record_assignment(ref, matchday=5, home="Natrican", away="Avada Zenith")
    mgr.save()   # persists to referee_state.json
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# REFEREE DATA
# ─────────────────────────────────────────────

@dataclass
class Referee:
    """A single referee with their attributes and tracking state."""
    name: str
    strictness: float                       # 0.0 = lenient, 1.0 = strict
    tier: int = 1                           # 1 = Select Group 1, 2 = Select Group 2
    preferred_foot: str = "right"           # cosmetic
    experience_years: int = 8               # cosmetic flavour
    bias_home: float = 0.0                  # slight home-foul-calling tendency (-0.1 to 0.1)
    bias_away: float = 0.0                  # slight away-foul-calling tendency (-0.1 to 0.1)

    # ── tracking state (updated across the season) ──
    matches_officiated: int = 0
    matchdays_refd: List[int] = field(default_factory=list)
    teams_seen: Dict[str, int] = field(default_factory=dict)       # team -> times ref'd them
    last_matchday: int = -99                # last matchday this ref worked

    def games_this_season(self) -> int:
        return self.matches_officiated

    def team_count(self, team: str) -> int:
        return self.teams_seen.get(team, 0)

    def rested_since(self, matchday: int) -> bool:
        """True if the ref's last matchday was before `matchday` (rest rule)."""
        return self.last_matchday < matchday - 1

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Referee":
        d = dict(d)
        return cls(**d)


# ─────────────────────────────────────────────
# DEFAULT PLOFA REFEREE POOL
# ─────────────────────────────────────────────

def _default_pool() -> List[Referee]:
    """20 generic PLOFA referees for the 26/27 season."""
    return [
        # ── Original 9 ──
        Referee("Marcus Osei",      0.00, experience_years=12, bias_home= 0.02, bias_away=-0.01),
        Referee("Juri Yuresh",      0.00, experience_years=10, bias_home=-0.01, bias_away= 0.02),
        Referee("Eric Vanam",       0.20, experience_years=14, bias_home= 0.00, bias_away= 0.00),
        Referee("Daniel Liu",       0.00, experience_years= 9, bias_home= 0.01, bias_away=-0.02),
        Referee("Ashley Mantu",     0.10, experience_years=11, bias_home=-0.02, bias_away= 0.01),
        Referee("Feriza Maria",     0.00, experience_years= 8, bias_home= 0.01, bias_away= 0.00),
        Referee("William Hurte",    0.00, experience_years=13, bias_home= 0.00, bias_away= 0.01),
        Referee("Jofart Eli",       0.10, experience_years= 7, bias_home=-0.01, bias_away=-0.01),
        Referee("Carlos Caper",     0.10, experience_years=15, bias_home= 0.02, bias_away=-0.02),

        # ── New refs to fill the pool to 20 ──
        Referee("Aron Deuztch",     0.00, experience_years=10, bias_home= 0.01, bias_away=-0.01),
        Referee("Semi Okarfor",     0.15, experience_years= 9, bias_home=-0.01, bias_away= 0.02),
        Referee("Tomasz Bier",      0.05, experience_years=11, bias_home= 0.00, bias_away= 0.01),
        Referee("Liam Hargreaves",  0.08, experience_years= 8, bias_home= 0.02, bias_away=-0.01),
        Referee("Daisuke Tanaka",   0.12, experience_years=10, bias_home=-0.02, bias_away= 0.00),
        Referee("Raphel Conti",     0.03, experience_years=14, bias_home= 0.01, bias_away= 0.01),
        Referee("Nabil Zarouk",     0.07, experience_years= 7, bias_home= 0.00, bias_away=-0.02),
        Referee("Eoin Gallagher",   0.10, experience_years=12, bias_home=-0.01, bias_away= 0.01),
        Referee("Kofi Asante",      0.05, experience_years= 9, bias_home= 0.01, bias_away= 0.00),
        Referee("Mikhail Sorokin",  0.18, experience_years=13, bias_home=-0.02, bias_away= 0.02),
        Referee("Javier Rollan",    0.02, experience_years=11, bias_home= 0.00, bias_away=-0.01),
    ]


# ─────────────────────────────────────────────
# REFEREE MANAGER
# ─────────────────────────────────────────────

class RefereeManager:
    """
    Manages automatic referee assignment for an entire PLOFA season.

    Assignment priority (EPL-style):
      1. Ref must not have ref'd back-to-back matchdays (rest rule).
      2. Ref must not have officiated either team more than MAX_TEAM_GAMES times.
      3. Among eligible refs, prefer the one with fewest total games (equity).
      4. Tiebreak: random.
    """

    MAX_TEAM_GAMES = 3          # max times a ref can officiate the same team per season
    MAX_MATCHDAY_GAP = 1        # rest: must sit out at least 1 matchday between games

    def __init__(self, pool: Optional[List[Referee]] = None,
                 state_path: str = "referee_state.json"):
        self.pool: List[Referee] = pool or _default_pool()
        self.state_path = state_path
        self._load()

    # ── persistence ──────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            self.pool = [Referee.from_dict(r) for r in data.get("pool", [])]

    def save(self):
        data = {"pool": [r.to_dict() for r in self.pool]}
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── assignment ───────────────────────────────────────────

    def _eligible(self, matchday: int, home: str, away: str) -> List[Referee]:
        """Return refs who satisfy rest rule and team-cap rule."""
        eligible = []
        for ref in self.pool:
            # Rest rule: must not have ref'd the previous matchday
            if not ref.rested_since(matchday):
                continue
            # Team cap: no ref can do the same team too many times
            if ref.team_count(home) >= self.MAX_TEAM_GAMES:
                continue
            if ref.team_count(away) >= self.MAX_TEAM_GAMES:
                continue
            eligible.append(ref)
        return eligible

    def assign(self, matchday: int, home: str, away: str,
               force_ref: Optional[str] = None) -> Referee:
        """
        Pick the best available referee for this fixture.

        If `force_ref` is given (a name string), use that ref directly
        regardless of rotation rules (useful for marquee / derby matches).

        Returns a Referee object whose name/strictness can be passed
        straight into MatchConfig.
        """
        # Forced assignment (e.g. for a specific match you want a particular ref)
        if force_ref:
            match = [r for r in self.pool if r.name.lower() == force_ref.lower()]
            if match:
                return match[0]
            raise ValueError(f"Referee '{force_ref}' not found in pool.")

        eligible = self._eligible(matchday, home, away)

        if not eligible:
            # Fallback: relax rest rule but still respect team caps
            eligible = [
                r for r in self.pool
                if r.team_count(home) < self.MAX_TEAM_GAMES
                and r.team_count(away) < self.MAX_TEAM_GAMES
            ]

        if not eligible:
            # Extreme fallback: pick least-used ref overall
            eligible = sorted(self.pool, key=lambda r: r.matches_officiated)

        # Equity: prefer fewest total games, then random tiebreak
        min_games = min(r.matches_officiated for r in eligible)
        best = [r for r in eligible if r.matches_officiated == min_games]
        return random.choice(best)

    def record_assignment(self, ref: Referee, matchday: int,
                          home: str, away: str):
        """Call this AFTER the match is played to update tracking state."""
        ref.matches_officiated += 1
        ref.matchdays_refd.append(matchday)
        ref.teams_seen[home] = ref.teams_seen.get(home, 0) + 1
        ref.teams_seen[away] = ref.teams_seen.get(away, 0) + 1
        ref.last_matchday = matchday

    def assign_all_matchday(self, matchday: int,
                            fixtures: List[Tuple[str, str]],
                            force_ref: Optional[str] = None
                            ) -> List[Tuple[Referee, str, str]]:
        """
        Assign referees to every fixture on a matchday.
        `fixtures` is a list of (home, away) tuples.
        Returns [(Referee, home, away), ...] in fixture order.
        """
        assignments = []
        for home, away in fixtures:
            ref = self.assign(matchday, home, away, force_ref=force_ref)
            self.record_assignment(ref, matchday, home, away)
            assignments.append((ref, home, away))
        return assignments

    # ── reporting ────────────────────────────────────────────

    def standings(self) -> str:
        """Print a table of referee usage this season."""
        lines = ["  Referee                  | Games | Teams Ref'd"]
        lines.append("  " + "─" * 47)
        for ref in sorted(self.pool, key=lambda r: -r.matches_officiated):
            teams = ", ".join(f"{t}({c})" for t, c in
                             sorted(ref.teams_seen.items(), key=lambda x: -x[1]))
            lines.append(f"  {ref.name:<24} | {ref.matches_officiated:>5} | "
                         f"{teams or '—'}")
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary of pool state."""
        total = sum(r.matches_officiated for r in self.pool)
        active = sum(1 for r in self.pool if r.matches_officiated > 0)
        return (f"Ref Pool: {len(self.pool)} refs | "
                f"{active} active | {total} total assignments")

    def get_ref_by_name(self, name: str) -> Optional[Referee]:
        for r in self.pool:
            if r.name.lower() == name.lower():
                return r
        return None
