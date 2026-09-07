"""
PLOFA 26/27 — MANAGER PROFILES
================================
manager_profile.py

Gives every club a real ManagerProfile instead of a bare name. Per the
analyst guidance, a manager is NOT a new decision-maker in the engine —
it's a small set of tendencies that RETUNE dials already wired into
TacticalAI, SubstitutionController and SquadChemistry. Same bias-layer
pattern as souls and chemistry.

What a ManagerProfile holds (all 0.0–1.0 tendencies, no arbitrary
"tactics: 85" attributes):

    risk_tolerance        How aggressively they chase/protect leads.
                          Shifts WHEN TacticalAI triggers all_out_chase
                          (0 = early, 1 = late) and WHEN it parks the bus
                          to protect a lead.
    substitution_patience How patient they are leaving tired players on.
                          0 = subs quickly, 1 = stubborn (maps straight
                          onto SubstitutionController.manager_stubbornness).
    style_bias            Nudges the team's chosen TeamStyle / PlayingStyle.
                          +1 stays rigidly on identity, 0 lets results push
                          them off it.
    rotation_tendency     Favours a set XI (0) vs. active squad rotation (1).
    man_management        Scales SquadChemistry's leadership aura — "gets
                          more out of a group than the raw numbers".
    prefer_possession     Consistent stylistic leaning (0=direct, 1=possession)
                          used as a soft override on posture outcomes.

Usage:
    from manager_profile import ManagerPool

    pool = ManagerPool()                      # loads overrides + auto-generates
    mgr  = pool.manager_for("Natrican")       # ManagerProfile object
    mgr.stubbornness()                        # -> substitution patience value
    mgr.risk_tolerance                        # -> tactical posture shifting
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class ManagerProfile:
    name: str
    club: str = ""

    # ── philosophical tendencies (0.0–1.0) ──
    risk_tolerance: float = 0.5           # 0 = cautious, 1 = ALL-OUT attacking
    substitution_patience: float = 0.35   # 0 = subs early, 1 = stubborn
    style_bias: float = 0.7               # 0 = results drive style, 1 = rigid identity
    rotation_tendency: float = 0.3        # 0 = set XI, 1 = constant rotation
    man_management: float = 0.5           # 0 = tactical robot, 1 = player's man-manager
    prefer_possession: float = 0.5        # 0 = direct, 1 = possession

    # ── flavour (narrative only, no engine effect) ──
    nationality: str = ""
    tactical_philosophy: str = ""         # e.g. "Gegenpress", "Low block"
    face_pressure: float = 0.5            # tendency to wilt/rise under pressure

    # ── job-security tracking (derived, not authored) ──
    matches_under_contract_at: int = 0    # for resilience nudges if desired
    result_history: list = field(default_factory=list)   # (matchday, actual_pts, xp_pts)

    # ── derived helpers ─────────────────────────────────────────

    def stubbornness(self) -> float:
        """0 = subs quickly, 1 = never subs for stamina. Used as
        SubstitutionController.manager_stubbornness."""
        return round(self.substitution_patience, 4)

    def chase_shift(self) -> float:
        """How many minutes EARLIER an attacker chases the game.
        Positive = aggressive (pushes early). Range ~[-8, +8]."""
        return round((self.risk_tolerance - 0.5) * 16.0, 2)

    def protect_shift(self) -> float:
        """How many minutes LATER a cautious manager parks the bus.
        Negative risk = protects a lead earlier. Range ~[-8, +8].
        (High risk -> protects later; low risk -> protects earlier.)"""
        return round((self.risk_tolerance - 0.5) * -16.0, 2)

    def leadership_aura_scale(self) -> float:
        """Multiplier applied to SquadChemistry team-leadership aura.
        A great man-manager gives the aura an extra boost (up to ~1.06)."""
        return round(0.975 + 0.05 * self.man_management, 4)

    def composure_scale(self) -> float:
        """Flat composure boost for the whole side from a stable, respected
        manager. Ranges ~0.995 (inoffensive) to ~1.04 (transformative)."""
        return round(0.99 + 0.045 * self.man_management, 4)

    # ── job security (emergent) ─────────────────────────────────

    def record_result(self, matchday: int, actual_pts: float, xp_pts: float):
        """Append a played result; keep only the last RECENT_WINDOW games."""
        self.result_history.append((matchday, actual_pts, xp_pts))
        self.result_history = self.result_history[-8:]

    def sack_risk(self, window: int = 8) -> float:
        """
        Emergent job-security signal. Reads two numbers the engine ALREADY
        generates: actual points vs. expected points (xP) over the rolling
        window. If a manager is 6+ combined points *behind* xP, that's a
        genuine sack-risk trigger — no new randomness, the article writes
        itself off data we already produce.

        Returns 0.0 (safe) to ~1.0 (on the brink).
        """
        if not self.result_history:
            return 0.0
        recent = self.result_history[-window:]
        actual = sum(r[1] for r in recent)
        xp = sum(r[2] for r in recent)
        under_performance = xp - actual
        if under_performance <= 0:
            return 0.0
        return round(min(1.0, under_performance / 10.0), 3)

    def job_status(self) -> str:
        risk = self.sack_risk()
        if risk >= 0.7:
            return "CRITICAL — job on the line"
        if risk >= 0.4:
            return "UNDER PRESSURE"
        if risk >= 0.2:
            return "questionable"
        return "secure"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ManagerProfile":
        d = dict(d)
        d["result_history"] = [tuple(r) for r in d.get("result_history", [])]
        return cls(**d)


# ─────────────────────────────────────────────
# MANAGER NAMES (REAL 2025/26 COACHES + GENERICS)
# ─────────────────────────────────────────────
#
# Loaded from "PLOFA 2025-2026 II.xlsx" → sheet "TEAMS PROFILES"
# (columns: HEAD-COACH, Manager's Famous Tactic, 25/26 POSITION,
#  MANAGER LIKELY HOOD OF COACHING NEXT SEASON).
#
# Each entry: profile = (tendency-template-name, real_coach_name).
# The tendency template turns the coach's famous tactic + league position
# into the philosophical dials (risk, possession, rotation, man-management)
# that retune TacticalAI / SubstitutionController / SquadChemistry.
#
# Promoted sides (Avada Zenith, Ganester, Rodice) and any club whose coach
# likelihood == NO get a fresh auto-generated manager — those are marked
# explicitly below, so edit this table instead of the picker logic.
MANAGER_OVERRIDES: Dict[str, tuple] = {
    # ── 2025/26 clubs that KEPT their coach (approx. from II.xlsx) ──
    "Claw":          ("gegenpress", "Roger Cyzzo"),
    "Club Chovers":  ("counter",    "Chilan Strain"),
    "Justice":       ("defensive",  "Patrick Quintin"),
    "Lige-8":        ("possession", "Don Goyle"),
    "Oxton":         ("patient",    "Jack Patro"),
    "Pearls":        ("attacking",  "Nurem Sopan"),
    "Seafcea":       ("balanced",   "Picaster Levi"),
    "Telbey":        ("defensive",  "Meso Jones"),
    "Trendboys":     ("direct",     "Lucas Roe"),
    "Tryox City":    ("counter",    "Wehl McKinn"),

    # ── Coaches whose "LIKELY HOOD" == NO → replaced next season ──
    # Natrican (Fabisch Diaz), Play City (Michael Borin),
    # Red Wolves (Zack Loriente), Triumpher (Garley Smiddos) get new/auto.
    "Uditon":        ("possession", "Jovey Leinard"),   # Micha Sunod out → Leinard in

    # ── Promoted sides (no 2025/26 coach data) ──
    # Avada Zenith, Ganester, Rodice → auto-generated below.
}

# Tendency templates — turn a coach's famous tactic into philosophical dials.
# Fields: (risk, possession, pressure, rotation, man_management)
MANAGER_TYPES: Dict[str, tuple] = {
    "gegenpress":  (0.78, 0.45, 0.34, 0.55, 0.40),
    "counter":     (0.58, 0.32, 0.42, 0.48, 0.52),
    "defensive":   (0.30, 0.40, 0.55, 0.25, 0.60),
    "possession":  (0.45, 0.80, 0.28, 0.45, 0.55),
    "patient":     (0.42, 0.72, 0.42, 0.35, 0.62),
    "attacking":   (0.85, 0.55, 0.22, 0.52, 0.48),
    "direct":      (0.55, 0.28, 0.45, 0.40, 0.45),
    "balanced":    (0.50, 0.50, 0.35, 0.40, 0.50),
}

# Generic names available for clubs without a known manager (auto-assigned).
GENERIC_MANAGER_NAMES: list = [
    "Tomas Okafor", "Petr Novak", "Emil Sato", "Rico Santana",
    "Aldous Mbeki", "Yanis Ferhat", "Kasper Lind", "Marco Tullio",
    "Dmitri Volkov", "Owen Carrick", "Sergio Maldonado", "Tariq Nassar",
    "Lars Eriksen", "Boubakary Diop", "Ilan Peretz", "Rui Castelo",
    "Kael Morrow", "Sipho Nkosi", "Viktor Halvorsen", "Rafael Duenas",
    "Callum Byrne", "Mateusz Zieliński", "Omar Haddad", "Alexis Fontaine",
]


# ─────────────────────────────────────────────
# MANAGER POOL
# ─────────────────────────────────────────────

class ManagerPool:
    """
    Resolves a ManagerProfile for every club. Loads real overrides where
    provided and auto-generates a plausible generic manager for clubs that
    don't have one (promoted teams), biased toward that club's playing
    style so a high-press club tends to get a gegenpress-minded boss.
    """

    def __init__(self, clubs: Optional[list] = None,
                 style_lookup: Optional[Dict[str, str]] = None,
                 state_path: str = "manager_state.json"):
        self.state_path = state_path
        self.overrides = dict(MANAGER_OVERRIDES)
        self._managers: Dict[str, ManagerProfile] = {}
        self._style_lookup = style_lookup or {}
        self._load()
        self._ensure_all(clubs or [])

    # ── persistence ──────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                for club, mdict in data.get("managers", {}).items():
                    self._managers[club] = ManagerProfile.from_dict(mdict)
            except Exception:
                self._managers = {}

    def save(self):
        data = {"managers": {c: m.to_dict() for c, m in self._managers.items()}}
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── construction ─────────────────────────────────────────

    def _manager_from_template(self, club: str, template: str,
                               name: str, used_names: set) -> ManagerProfile:
        """Build a ManagerProfile from one of the MANAGER_TYPES templates
        (a coach's famous tactic → philosophical dials)."""
        risk, possessed, pressure, rotation, man_mgmt = MANAGER_TYPES.get(
            template, MANAGER_TYPES["balanced"]
        )
        used_names.add(name)
        return ManagerProfile(
            name=name,
            club=club,
            nationality=random.choice(["English", "Spanish", "Italian", "German",
                                       "Portuguese", "Dutch", "French", "Danish"]),
            risk_tolerance=round(random.uniform(-0.06, 0.06) + risk, 3),
            substitution_patience=round(random.uniform(0.15, 0.45) + pressure * 0.3, 3),
            style_bias=round(random.uniform(0.55, 0.85), 3),
            rotation_tendency=round(rotation, 3),
            man_management=round(man_mgmt, 3),
            prefer_possession=round(possessed, 3),
            face_pressure=round(0.4 + pressure * 0.5, 3),
            tactical_philosophy=template.title().replace("_", " "),
        )

    def _random_manager(self, club: str, used_names: set) -> ManagerProfile:
        style = (self._style_lookup or {}).get(club, "").lower()
        name = next((n for n in GENERIC_MANAGER_NAMES if n not in used_names),
                    f"Generic Manager {len(used_names)+1}")
        used_names.add(name)

        # Bias tendencies from the club's playing style (TeamStyle .value).
        if style in ("gegenpressing",):
            template = "gegenpress"
        elif style in ("tiki_taka", "vertical_tiki_taka", "structured_possession"):
            template = "possession"
        elif style in ("fluid_counter", "wing_play"):
            template = "counter"
        elif style in ("route_one",):
            template = "direct"
        elif style in ("park_the_bus", "defensive", "ultra_defensive"):
            template = "defensive"
        elif style in ("ultra_attacking", "attacking"):
            template = "attacking"
        else:
            template = "balanced"

        risk, possessed, pressure, rotation, man_mgmt = MANAGER_TYPES.get(
            template, MANAGER_TYPES["balanced"]
        )
        return ManagerProfile(
            name=name,
            club=club,
            nationality=random.choice(["English", "Spanish", "Italian", "German",
                                       "Portuguese", "Dutch", "French", "Danish"]),
            risk_tolerance=round(random.uniform(0.30, 0.85) * 0.5 + risk * 0.5, 3),
            substitution_patience=round(random.uniform(0.1, 0.6), 3),
            style_bias=round(random.uniform(0.5, 0.9), 3),
            rotation_tendency=round(rotation + random.uniform(-0.1, 0.1), 3),
            man_management=round(random.uniform(0.3, 0.85), 3),
            prefer_possession=round(possessed, 3),
            face_pressure=round(0.4 + pressure * 0.5, 3),
            tactical_philosophy=template.title().replace("_", " "),
        )

    def _ensure_all(self, clubs: list):
        """Guarantee every club has a manager. Real coaches from MANAGER_OVERRIDES
        take priority (name + tendency template from their 2025/26 famous tactic).
        Any club NOT overridden — e.g. promoted sides (Avada Zenith, Ganester,
        Rodice) or clubs whose coach was sacked — gets an auto-generated manager
        biased toward that club's current playing style."""
        assigned = set(self._managers.keys())
        # Real coach overrides take priority.
        for club, (template, real_name) in self.overrides.items():
            if club in self._managers:
                existing = self._managers[club]
                # Reuse tracked profile but re-apply the coach's template dials
                # (so job-security history isn't lost across manager changes).
                tpl = MANAGER_TYPES.get(template, MANAGER_TYPES["balanced"])
                existing.name = real_name
                existing.club = club
                existing.risk_tolerance = tpl[0]
                existing.prefer_possession = tpl[1]
                existing.face_pressure = round(0.4 + tpl[2] * 0.5, 3)
                existing.rotation_tendency = tpl[3]
                existing.man_management = tpl[4]
                existing.tactical_philosophy = template.title().replace("_", " ")
            else:
                self._managers[club] = self._manager_from_template(
                    club, template, real_name, assigned)
            assigned.add(club)

        # Auto-generate for any club still missing one.
        for club in clubs:
            if club not in self._managers:
                self._managers[club] = self._random_manager(club, assigned)
                assigned.add(club)

    @staticmethod
    def known_clubs() -> list:
        return list(MANAGER_OVERRIDES.keys())

    # ── access ───────────────────────────────────────────────

    def manager_for(self, club: str) -> Optional[ManagerProfile]:
        """Get (and lazily create) the profile for a given club."""
        if club not in self._managers:
            self._managers[club] = self._random_manager(club, set(self._managers.keys()))
        return self._managers[club]

    def all(self) -> Dict[str, ManagerProfile]:
        return self._managers
