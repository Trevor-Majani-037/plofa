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
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from manager_brain import ManagerBrain
from manager_memory import ManagerMemory
from manager_mind import ManagerMind
from manager_sensors import extract_manager_sensors


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
        """Record a played result; keep only the last RECENT_WINDOW games.

        Re-running the same matchday REPLACES the prior entry (a fixture may
        only contribute one matchday to a manager's history)."""
        self.result_history = [
            r for r in self.result_history if r[0] != matchday
        ] + [(matchday, actual_pts, xp_pts)]
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


# ─────────────────────────────────────────
# MANAGER (BRAIN + MIND + MEMORY COMPOSER)
# ─────────────────────────────────────────

MANAGER_MIN_DWELL_S = 180          # minimum seconds between posture changes

_MANAGER_POSTURE_LABELS = ("DEFEND", "BALANCED", "ATTACK")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _opponent_scored(event_type: str, engine, team: str) -> bool:
    """True if the engine's most recent goal belongs to the opponent.

    The engine (Phase 6 wiring) stamps ``state.last_goal_team`` after a
    goal chain; tests can set it directly. Falls back to False if the
    engine exposes no scorer info yet.
    """
    last_team = None
    state = getattr(engine, "state", None)
    if state is not None:
        last_team = getattr(state, "last_goal_team", None)
    if last_team is None:
        last_team = getattr(engine, "last_goal_team", None)
    if last_team is None:
        return False
    return str(last_team) != str(team)


@dataclass
class Manager:
    """A full manager: the brain (network), mind (personality) and
    episodic memory, composed into one decision-maker.

    Decides at engine trigger events (Phase 6), caches the current
    posture/pressing/urgency, and records every conviction into memory
    at the final whistle.
    """

    name: str
    brain: ManagerBrain
    mind: ManagerMind
    memory: ManagerMemory
    memory_strength: float = 0.3

    # ── Cooldown tracking (decision state) ──
    _last_posture_change_s: float = -999.0
    _current_posture: str = "BALANCED"
    _current_pressing: float = 0.5
    _current_urgency: float = 0.5

    # ── Decision pipeline (D5 order of operations) ─────────────

    def decide(self, engine, team: str) -> dict:
        """Produce a full decision dict for a trigger event.

        Order of operations (D5):
          1. raw = extract_manager_sensors(engine, team)
          2. perceived = mind.filter_perception(raw, engine)
          3. out = brain.forward(perceived)
          4. posture-probs = mind.filter_posture(out["posture_probs"])
          5. pressing = mind.filter_pressing(out["pressing"])
          6. urgency = mind.filter_sub_urgency(out["sub_urgency"])
          7. memory bias applied to posture (in log-space)
          8. MANAGER_MIN_DWELL_S respected for posture changes
          9. current values stored on self
         10. dict returned

        Returns keys: posture, posture_probs, pressing, sub_urgency.
        """
        raw = extract_manager_sensors(engine, team)
        perceived = self.mind.filter_perception(raw, engine)
        out = self.brain.forward(perceived)

        posture_probs = self.mind.filter_posture(out["posture_probs"], engine)
        pressing = self.mind.filter_pressing(out["pressing"], engine)
        urgency = self.mind.filter_sub_urgency(out["sub_urgency"], engine)

        # ── Memory bias (log-space) ─────────────────────────────
        ctx = self._context_for(engine, team, posture=self._current_posture)
        bias = self.memory.bias_for(ctx)
        if np.any(bias != 0.0):
            logits = np.log(np.clip(posture_probs, 1e-9, 1.0)) \
                + self.memory_strength * bias
            posture_probs = _softmax(logits)

        # ── Dwell-time guard on posture changes ─────────────────
        now_s = float(getattr(engine.state, "minute", 0)) * 60.0
        chosen_idx = int(np.argmax(posture_probs))
        proposed = _MANAGER_POSTURE_LABELS[chosen_idx]

        if now_s - self._last_posture_change_s >= MANAGER_MIN_DWELL_S:
            if proposed != self._current_posture:
                self._current_posture = proposed
                self._last_posture_change_s = now_s
        # (else: keep _current_posture; dwell not elapsed)

        # Every trigger decision is a conviction, even a held posture.
        self.mind.record_posture_taken(self._current_posture)

        self._current_pressing = pressing
        self._current_urgency = urgency

        return {
            "posture": self._current_posture,
            "posture_probs": posture_probs,
            "pressing": pressing,
            "sub_urgency": urgency,
        }

    # ── Event callbacks ────────────────────────────────────────

    def on_event(self, event_type: str, engine, team: str):
        if event_type in ("GOAL",) and _opponent_scored(event_type, engine, team):
            self.mind.on_goal_conceded({})
        elif event_type == "GOAL":
            self.mind.on_goal_scored({})
        # every other event type is currently a no-op

    def end_of_match(self, result, team: str):
        self.mind.end_of_match(result, team)

        # Record each posture conviction of this match into memory.
        for posture, count in self.mind.history_posture_counts.items():
            if count > 0:
                ctx = self._context_for(
                    SimpleNamespace(
                        state=SimpleNamespace(minute=90),
                        config=result.config if hasattr(result, "config") else None,
                    ),
                    team, posture=posture,
                )
                self.memory.record(ctx, posture, result, team)

        self.memory_strength = self.memory.strength()

    # ── Context builder for memory keys ────────────────────────

    def _context_for(self, engine_or_state, team: str, posture: str):
        """Build the memory key context (score_diff, minute, posture)."""
        state = getattr(engine_or_state, "state", engine_or_state)
        home_goals = getattr(state, "home_goals", 0) or 0
        away_goals = getattr(state, "away_goals", 0) or 0
        is_home = True
        cfg = getattr(engine_or_state, "config", None)
        if cfg is not None:
            is_home = (getattr(cfg, "home_team", None) == team)
        score_diff = (home_goals - away_goals) if is_home \
            else (away_goals - home_goals)
        minute = int(getattr(state, "minute", 0) or 0)
        return SimpleNamespace(
            score_diff=score_diff, minute=minute, posture_taken=str(posture))

    # ── Persistence (3 files: manager-bundle, mind, memory) ────

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        data = {
            "kind": "manager",
            "version": 1,
            "name": self.name,
            "memory_strength": self.memory_strength,
            "last_posture_change_s": self._last_posture_change_s,
            "current_posture": self._current_posture,
            "current_pressing": self._current_pressing,
            "current_urgency": self._current_urgency,
            "brain": self.brain.serialize(),
        }
        with open(os.path.join(dir_path, "manager.json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.mind.save(os.path.join(dir_path, "mind.json"))
        self.memory.save(os.path.join(dir_path, "memory.json"))

    @classmethod
    def load(cls, dir_path: str) -> "Manager":
        with open(os.path.join(dir_path, "manager.json"), "r",
                  encoding="utf-8") as f:
            data = json.load(f)
        if data.get("kind") != "manager":
            raise ValueError(data.get("kind"))
        brain = ManagerBrain.deserialize(data["brain"])
        mind = ManagerMind.load(os.path.join(dir_path, "mind.json"))
        memory = ManagerMemory.load(os.path.join(dir_path, "memory.json"))
        return cls(
            name=data.get("name", "Manager"),
            brain=brain,
            mind=mind,
            memory=memory,
            memory_strength=float(data.get("memory_strength", 0.3)),
            _last_posture_change_s=float(data.get("last_posture_change_s", -999.0)),
            _current_posture=data.get("current_posture", "BALANCED"),
            _current_pressing=float(data.get("current_pressing", 0.5)),
            _current_urgency=float(data.get("current_urgency", 0.5)),
        )

    def __repr__(self) -> str:
        return (f"Manager(name={self.name!r}, posture={self._current_posture}, "
                f"strength={self.memory_strength:.2f})")


# ─────────────────────────────────────────
# LIVE BRAIN-MANAGER BUILDER (Phase 8: manager goes live)
# ─────────────────────────────────────────
# The engine's brain path (match_engine.USE_MANAGER_BRAIN + TacticalAI)
# only engages for real ``Manager`` objects (brain + mind + memory), but
# ``ManagerPool`` resolves static ``ManagerProfile`` objects. These helpers
# bridge the gap:
#
#   * the brain is shared — loaded once from the evolved
#     ``manager_brains/v1/test_manager.json`` (deterministic), with a
#     club-stable random fallback if the file is missing;
#   * the mind starts neutral; the memory is per-club and PERSISTED under
#     ``manager_brains/live/<club>/`` so past matches carry over.
#
# The module default stays flag-OFF (baseline + tests assert False); live
# is explicit opt-in via ``pitch_replay.run_scratch_match(use_manager_brain=True)``.

LIVE_BRAIN_PATH = os.path.join("manager_brains", "v1", "test_manager.json")
LIVE_MEMORY_DIR = os.path.join("manager_brains", "live")


def _stable_seed(text: str) -> int:
    """Club-stable 32-bit seed without touching the global RNG stream."""
    import zlib
    return zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF


def _live_club_dir(club: str, memory_dir: str = LIVE_MEMORY_DIR) -> str:
    safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in str(club))
    return os.path.join(memory_dir, safe)


def brain_manager_for(club: str,
                      brain_path: Optional[str] = None,
                      memory_dir: str = LIVE_MEMORY_DIR,
                      mind: Optional[ManagerMind] = None) -> Manager:
    """Build (or reload) the live brain-manager for *club*.

    Brain: ``brain_path`` or ``LIVE_BRAIN_PATH`` if it exists, else a
    club-stable random brain. Mind: neutral unless given. Memory: loaded
    from the club's live dir if present, else fresh.
    """
    path = brain_path or LIVE_BRAIN_PATH
    brain = None
    if path and os.path.exists(path):
        try:
            brain = ManagerBrain.deserialize(
                json.load(open(path, "r", encoding="utf-8")))
        except Exception:
            brain = None
    if brain is None:
        brain = ManagerBrain.random(seed=_stable_seed(f"brain|{club}"))

    club_dir = _live_club_dir(club, memory_dir)
    mem_path = os.path.join(club_dir, "memory.json")
    if os.path.exists(mem_path):
        try:
            memory = ManagerMemory.load(mem_path)
        except Exception:
            memory = ManagerMemory()
    else:
        memory = ManagerMemory()

    live_mind = mind if mind is not None else ManagerMind()
    mind_path = os.path.join(club_dir, "mind.json")
    if mind is None and os.path.exists(mind_path):
        try:
            live_mind = ManagerMind.load(mind_path)
        except Exception:
            live_mind = ManagerMind()

    mgr = Manager(name=f"{club} Brain", brain=brain, mind=live_mind,
                  memory=memory, memory_strength=memory.strength())
    return mgr


def save_live_manager(manager: Manager, club: str,
                      memory_dir: str = LIVE_MEMORY_DIR) -> None:
    """Persist a live manager's mind + memory (the brain file is shared)."""
    club_dir = _live_club_dir(club, memory_dir)
    os.makedirs(club_dir, exist_ok=True)
    manager.mind.save(os.path.join(club_dir, "mind.json"))
    manager.memory.save(os.path.join(club_dir, "memory.json"))
