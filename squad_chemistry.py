"""
PLOFA 26/27 — SQUAD CHEMISTRY MODULE
=======================================
squad_chemistry.py

Closes gap #2 from analyst feedback: "No chemistry between players,
no leadership effects on teammates" + "No knock-on effects."

Philosophy:
    A team is not eleven independent probability generators. Two fullbacks
    who've started 40 games together complete more give-and-gos than two
    who met in pre-season. A captain's composure props up a shaky teenager.
    An injury to a soul-tier player doesn't just remove his stats — it
    dents everyone else's for the next 20 minutes.

    This module is intentionally decoupled from match_engine's per-event
    loop: it produces MULTIPLIERS that event_chain.py applies alongside
    the existing soul/anti-soul multiplier stack, so nothing upstream
    needs to be rewritten — only extended (see WIRING_GUIDE at bottom).
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# SQUAD CHEMISTRY
# ─────────────────────────────────────────────

@dataclass
class SquadChemistry:
    """How well a squad works together. One instance per team, persisted
    across the season by season_manager.py (chemistry builds/decays over time)."""

    team_name: str

    # {player_name: {teammate_name: chemistry_score 0-100}}
    pair_chemistry: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Leadership hierarchy
    captain: Optional[str]        = None
    vice_captain: Optional[str]   = None
    senior_players: List[str]     = field(default_factory=list)  # 3+ full seasons at club

    # Cliques — players who train/room together, small chemistry bonus among themselves
    cliques: List[List[str]]      = field(default_factory=list)

    # Games started together, tracked so chemistry can grow organically
    _appearances_together: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # ── QUERIES ─────────────────────────────────────────────

    def get_pair(self, p1: str, p2: str) -> float:
        """Symmetric lookup, defaults to a neutral 50 for two strangers."""
        if p1 in self.pair_chemistry and p2 in self.pair_chemistry[p1]:
            return self.pair_chemistry[p1][p2]
        if p2 in self.pair_chemistry and p1 in self.pair_chemistry[p2]:
            return self.pair_chemistry[p2][p1]
        # Clique members who've never played together still get a small bump
        for clique in self.cliques:
            if p1 in clique and p2 in clique:
                return 58.0
        return 50.0

    def pass_chemistry_mult(self, p1: str, p2: str) -> float:
        """0.90x (ice cold) to 1.14x (telepathic). Applied to pass accuracy
        between two specific players when a pass is attempted p1 -> p2."""
        c = self.get_pair(p1, p2)
        return round(0.90 + (c / 100.0) * 0.24, 4)

    def positional_understanding_mult(self, p1: str, p2: str) -> float:
        """Affects defensive shape / off-ball coordination (e.g. a CB pairing's
        joint tackle_success / press-trap coordination)."""
        c = self.get_pair(p1, p2)
        return round(0.92 + (c / 100.0) * 0.18, 4)

    def leadership_composure_mult(self, player_name: str) -> float:
        """
        Composure boost for anyone playing alongside the captain / senior
        players. Strongest for the captain themself (steadies the ship),
        smaller ripple for the rest of the XI.
        """
        if player_name == self.captain:
            return 1.10
        if player_name == self.vice_captain:
            return 1.06
        if player_name in self.senior_players:
            return 1.03
        return 1.0

    def team_leadership_aura(self, captain_on_pitch: bool, is_derby: bool = False) -> float:
        """
        Applied to the WHOLE team's composure_mult when the captain is on
        the pitch — bigger effect in big/derby games, gone entirely if the
        captain has been subbed off or sent off.
        """
        if not captain_on_pitch:
            return 0.94   # visible dip without on-pitch leadership
        return 1.06 if is_derby else 1.03

    # ── UPDATES (called by season_manager.py after each match) ──

    def register_appearance_together(self, lineup: List[str]):
        """Call once per match with the full starting XI (or full match squad)
        of players who shared the pitch; grows chemistry organically."""
        for i, p1 in enumerate(lineup):
            for p2 in lineup[i + 1:]:
                key = tuple(sorted([p1, p2]))
                self._appearances_together[key] = self._appearances_together.get(key, 0) + 1
                games = self._appearances_together[key]
                # Diminishing-returns growth curve: fast early gains, plateaus ~90
                current = self.get_pair(p1, p2)
                target = min(92.0, 50.0 + games * 3.2)
                new_val = current + (target - current) * 0.35
                self.pair_chemistry.setdefault(p1, {})[p2] = round(new_val, 1)

    def decay_unused_pairs(self, active_names: List[str], decay_rate: float = 0.5):
        """Chemistry between players who DIDN'T play together this week fades
        slightly (injuries, rotation) — call once per matchday for realism."""
        for p1, teammates in self.pair_chemistry.items():
            for p2 in list(teammates.keys()):
                if not (p1 in active_names and p2 in active_names):
                    teammates[p2] = max(40.0, teammates[p2] - decay_rate)

    def set_leadership(self, captain: str, vice_captain: str = None,
                        senior_players: List[str] = None):
        self.captain = captain
        self.vice_captain = vice_captain
        self.senior_players = senior_players or []


# ─────────────────────────────────────────────
# KNOCK-ON EFFECTS
# What happens to the OTHER 21 players when something dramatic
# happens to one of them.
# ─────────────────────────────────────────────

class KnockOnEffects:
    """
    Stateless calculators for ripple effects. Called by match_engine.py /
    squad_manager.py at the moment of the triggering event; the returned
    multiplier is applied to teammates' composure_mult / decisions for a
    bounded window (handled by the caller, e.g. "next 10 minutes").
    """

    @staticmethod
    def key_player_injured(injured_player_dna, teammates_dna: List) -> float:
        """
        Morale drop for teammates when a key player goes down.
        Bigger dip if the injured player is a soul player, captain,
        or superstar. Returns a composure multiplier (<1.0 = worse).
        """
        severity = 0.94
        if getattr(injured_player_dna, "soul", None) is not None:
            severity -= 0.06     # losing a soul player really hurts
        if getattr(injured_player_dna, "is_superstar", False):
            severity -= 0.03
        return round(max(0.80, severity), 3)

    @staticmethod
    def key_player_sent_off(carded_team_composure: float = 0.90) -> float:
        """Red card shock — applied to the reduced team's remaining players."""
        return carded_team_composure

    @staticmethod
    def captain_removed(chemistry: SquadChemistry) -> float:
        """Captain subbed/injured/sent off mid-match: team loses its
        leadership aura for the rest of the match."""
        return chemistry.team_leadership_aura(captain_on_pitch=False)

    @staticmethod
    def prima_donna_drag(anti_soul_players: List) -> float:
        """
        Cumulative dressing-room drag from any PRIMA_DONNA-type players
        in the XI. Small on its own, compounds if a team has more than one.
        """
        from player_soul import AntiSoulApplicator
        drag = 1.0
        for p in anti_soul_players:
            drag *= AntiSoulApplicator.teammate_drag(p)
        return round(max(0.85, drag), 3)

    @staticmethod
    def comeback_lift(goal_difference: int, minute: int) -> float:
        """Not a personality effect, but a genuine 'knock-on' of the
        scoreline itself — momentum already models the macro version of
        this in match_engine.MomentumEngine; this is the composure-specific
        micro version used when picking individual event outcomes."""
        if goal_difference <= -2 and minute >= 70:
            return 0.93   # capitulation risk
        if goal_difference == -1 and minute >= 80:
            return 1.04   # backs-to-the-wall lift for the chasing team
        return 1.0


# ─────────────────────────────────────────────
# WIRING GUIDE
# ─────────────────────────────────────────────

WIRING_GUIDE = """
WIRING squad_chemistry.py INTO THE ENGINE
═══════════════════════════════════════════

1. run_match.py — build once per team, persisted via season_manager.py:
       chem = SquadChemistry(team_name=HOME_TEAM)
       chem.set_leadership(captain="Mateo Sanz", vice_captain="Emeka Obi",
                            senior_players=["Keano Walsh", "Tavish Crane"])

2. event_chain.py — PossessionChain._pass_success(): multiply the DNA-based
   accuracy by chem.pass_chemistry_mult(passer.name, receiver.name).

3. event_chain.py — AttackChain / composure rolls: multiply composure by
       chem.leadership_composure_mult(player.name)
       * chem.team_leadership_aura(captain_on_pitch, is_derby)

4. squad_manager.py — PlayerStaminaState.roll_injury(): on a True return for
   a key player, call KnockOnEffects.key_player_injured(...) and apply the
   result as a temporary composure multiplier to that team's other active
   PlayerStaminaState.performance_mult for ~10 in-match minutes.

5. season_manager.py — after each match:
       chem.register_appearance_together([p.name for p in starters])
       chem.decay_unused_pairs([p.name for p in this_week_squad])
   then persist chem to the season state file (see season_manager.py).
"""
