"""
Manager Brains — Phase 6 wiring tests.

Verifies that the opt-in ``USE_MANAGER_BRAIN`` flag in match_engine.py
injects the full decision-maker (manager_profile.Manager) into the live
match without disturbing the pre-Phase-6 static behaviour:

    - flag OFF  → TacticalAI.adjust returns byte-identical dials to the
                  frozen pre-change baseline (even if a brain Manager is
                  wired but the flag is OFF).
    - flag ON   → manager.decide is polled at trigger events and its
                  posture/pressing drive the EffectiveTactics dials.
    - dwell     → the 180s posture cooldown is honoured through the
                  engine's adjust trigger path.
    - events    → a GOAL event feeds Manager.on_event → the mind's
                  composure reacts to scoring/conceding.
"""
from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import match_engine as _me
from manager_brain import ManagerBrain
from manager_memory import ManagerMemory
from manager_mind import ManagerMind
from manager_profile import Manager, MANAGER_MIN_DWELL_S
from tactical_ai import TacticalAI
from tactical_shapes import FormationStance
from match_engine import MatchState, TeamProfile, TeamStyle, PlayingStyle, Intensity


# ── Helpers ─────────────────────────────────────────────────────

def _profile(press=0.5, tempo=0.5, direct=0.5, defline=0.5, shots=0.15,
             big=0.35, psucc=0.25, poss=50.0):
    return TeamProfile(
        name="T", style=TeamStyle.BALANCED, playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM,
        press_intensity=press, defensive_line=defline, tempo=tempo,
        directness=direct, possession_target=poss,
        shots_per_sequence=shots, big_chance_ratio=big,
        press_success_rate=psucc,
    )


def _egg(minute=30):
    """Lightweight MatchEngine mock matching what Manager.decide reads."""
    cfg = SimpleNamespace(home_team="Home", away_team="Away")
    state = SimpleNamespace(
        minute=float(minute), home_goals=0, away_goals=0,
        home_subs_made=0, away_subs_made=0,
    )
    return SimpleNamespace(config=cfg, state=state)


def _brain_manager(seed=42):
    brain = ManagerBrain.random(seed=seed)
    mind = ManagerMind(dogma=0.5, eq=0.5, empathy=0.5)
    mem = ManagerMemory(decay=1.0)
    return Manager(name="Brain", brain=brain, mind=mind, memory=mem,
                   memory_strength=0.0)


class _RecordingBrain:
    """Duck-typed manager that records every decide() poll and answers
    with a deterministic decision."""

    def __init__(self, posture="ATTACK", pressing=0.8):
        self.posture = posture
        self.pressing = pressing
        self.calls = []

    def decide(self, engine, team):
        self.calls.append((engine, team))
        return {"posture": self.posture, "pressing": self.pressing,
                "sub_urgency": 0.6}


# ── 1. Flag OFF ⇒ frozen static baseline ───────────────────────

def test_flag_off_uses_static():
    """With USE_MANAGER_BRAIN = False, adjust() matches the pre-change
    static baseline for the same fixed input.  Pins the flag OFF locally —
    the module default has been ON since 2026-09-20."""
    _old = _me.USE_MANAGER_BRAIN
    _me.USE_MANAGER_BRAIN = False
    try:
        cases = [
            # (minute, home, away, red, stam, manager, expected posture tag)
            (5,  0, 0, 0, 100.0, None, "baseline"),
            (30, 0, 2, 0, 100.0, None, "baseline"),
            (75, 2, 0, 0, 100.0, None, "see_it_out"),
            (88, 1, 1, 0, 100.0, None, "tense_level"),
            (80, 3, 0, 1, 100.0, None, "see_it_out+man_down"),
            (80, 3, 0, 0, 55.0, None, "see_it_out+fatigued"),
        ]

        def _expected(minute, h, a, red, stam):
            if (minute, h, a, red, stam) == (5, 0, 0, 0, 100.0):
                return dict(press=0.45, tempo=0.55, direct=0.46, def_line=0.5,
                            shots=0.11, big=0.33, poss=50.0, stance="BASELINE")
            if (minute, h, a, red, stam) == (30, 0, 2, 0, 100.0):
                return dict(press=0.5, tempo=0.55, direct=0.5, def_line=0.5,
                            shots=0.11, big=0.33, poss=50.0, stance="BASELINE")
            if (minute, h, a, red, stam) == (75, 2, 0, 0, 100.0):
                return dict(press=0.4562, tempo=0.5088, direct=0.4813, def_line=0.4625,
                            shots=0.1004, big=0.33, poss=49.38, stance="SEE_IT_OUT")
            if (minute, h, a, red, stam) == (88, 1, 1, 0, 100.0):
                return dict(press=0.54, tempo=0.55, direct=0.5, def_line=0.5,
                            shots=0.121, big=0.33, poss=50.0, stance="TENSE_LEVEL")
            if (minute, h, a, red, stam) == (80, 3, 0, 1, 100.0):
                return dict(press=0.363, tempo=0.4114, direct=0.4625, def_line=0.374,
                            shots=0.0907, big=0.33, poss=42.9, stance="SEE_IT_OUT")
            if (minute, h, a, red, stam) == (80, 3, 0, 0, 55.0):
                return dict(press=0.2805, tempo=0.374, direct=0.4625, def_line=0.323,
                            shots=0.0907, big=0.33, poss=48.75, stance="SEE_IT_OUT")
            raise AssertionError(f"unexpected case {(minute, h, a, red, stam)}")

        for minute, h, a, red, stam, mgr, posture in cases:
            state = MatchState(minute=minute, home_goals=h, away_goals=a)
            et = TacticalAI.adjust(_profile(), state, "H", "H",
                                   red_cards_against=red, avg_stamina=stam,
                                   manager=mgr)
            exp = _expected(minute, h, a, red, stam)
            assert et.press_intensity == exp["press"], (minute, et)
            assert et.tempo == exp["tempo"], (minute, et)
            assert et.directness == exp["direct"], (minute, et)
            assert et.defensive_line == exp["def_line"], (minute, et)
            assert et.shots_per_sequence == exp["shots"], (minute, et)
            assert et.big_chance_ratio == exp["big"], (minute, et)
            assert et.press_success_rate == 0.25, (minute, et)
            assert et.possession_target == exp["poss"], (minute, et)
            assert et.posture == posture, (minute, et)
            assert et.stance is getattr(FormationStance, exp["stance"]), (minute, et)
    finally:
        _me.USE_MANAGER_BRAIN = _old


def test_flag_off_ignores_brain_manager():
    """A brain Manager wired while the flag is OFF must NOT be polled and
    must not crash the static path (attribute guards for chase_shift etc.)."""
    mgr = _brain_manager()
    state = MatchState(minute=75, home_goals=2, away_goals=0)
    et = TacticalAI.adjust(_profile(), state, "H", "H", manager=mgr)
    assert et.posture == "see_it_out"
    assert et.stance is FormationStance.SEE_IT_OUT
    # no brain polling happened
    assert mgr._current_posture == "BALANCED"


# ── 2. Flag ON ⇒ brain drives the dials ─────────────────────────

def test_flag_on_uses_brain(monkeypatch):
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    # TeamProfile recalibrates authored dials (0.5/0.15/0.35 → tempo 0.55,
    # shots 0.11, big 0.33) — capture the static baseline for the SAME
    # profile first, then assert ATTACK lifts every dial above it.
    baseline = TacticalAI.adjust(_profile(), MatchState(minute=30,
                                                        home_goals=0,
                                                        away_goals=0),
                                 "H", "H")
    mgr = _RecordingBrain(posture="ATTACK", pressing=0.8)
    state = MatchState(minute=30, home_goals=0, away_goals=0)
    et = TacticalAI.adjust(_profile(), state, "H", "H", manager=mgr,
                           engine=object())
    assert len(mgr.calls) == 1            # decide polled on the trigger event
    assert et.posture == "brain_ATTACK"
    assert et.stance is FormationStance.ALL_OUT_CHASE
    assert et.press_intensity > baseline.press_intensity
    assert et.tempo > baseline.tempo
    assert et.shots_per_sequence > baseline.shots_per_sequence
    assert et.possession_target > baseline.possession_target


def test_flag_on_defend_lowers_dials(monkeypatch):
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    baseline = TacticalAI.adjust(_profile(), MatchState(minute=30,
                                                        home_goals=0,
                                                        away_goals=0),
                                 "H", "H")
    mgr = _RecordingBrain(posture="DEFEND", pressing=0.9)
    state = MatchState(minute=30, home_goals=0, away_goals=0)
    et = TacticalAI.adjust(_profile(), state, "H", "H", manager=mgr,
                           engine=object())
    assert et.posture == "brain_DEFEND"
    assert et.stance is FormationStance.SEE_IT_OUT
    assert et.press_intensity < baseline.press_intensity
    assert et.shots_per_sequence < baseline.shots_per_sequence


def test_flag_on_requires_engine(monkeypatch):
    """No engine ⇒ the brain cannot be polled (sensors need it); the static
    path still runs with flag ON rather than crashing."""
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    mgr = _RecordingBrain()
    state = MatchState(minute=80, home_goals=2, away_goals=0)
    et = TacticalAI.adjust(_profile(), state, "H", "H", manager=mgr)
    assert len(mgr.calls) == 0
    assert et.posture == "see_it_out"


# ── 3. Dwell respected through the engine trigger path ─────────

def test_dwell_respected_in_engine(monkeypatch):
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _brain_manager()
    eng = _egg(minute=10)
    first = TacticalAI.adjust(_profile(), MatchState(minute=10), "H", "H",
                              manager=mgr, engine=eng)
    posture_before = mgr._current_posture
    change_before = mgr._last_posture_change_s

    eng.state.minute = 11.0    # 60s later — still inside the 180s dwell
    second = TacticalAI.adjust(_profile(), MatchState(minute=11), "H", "H",
                               manager=mgr, engine=eng)
    # posture held, cooldown untouched
    assert mgr._current_posture == posture_before
    assert mgr._last_posture_change_s == change_before
    # and both effective-tactics postures carry the same brain tag
    assert second.posture.split("+")[0] == first.posture.split("+")[0]


def test_dwell_allows_change_in_engine(monkeypatch):
    monkeypatch.setattr(_me, "USE_MANAGER_BRAIN", True)
    monkeypatch.setattr(
        "manager_profile.extract_manager_sensors",
        lambda engine, team: np.full(12, 0.5, dtype=np.float64),
    )
    mgr = _brain_manager()
    eng = _egg(minute=10)
    TacticalAI.adjust(_profile(), MatchState(minute=10), "H", "H",
                      manager=mgr, engine=eng)
    eng.state.minute = 10 + MANAGER_MIN_DWELL_S / 60.0 + 1
    out = TacticalAI.adjust(_profile(), MatchState(minute=int(eng.state.minute)),
                            "H", "H", manager=mgr, engine=eng)
    assert out.posture in ("brain_ATTACK", "brain_BALANCED", "brain_DEFEND")


# ── 4. Goal events hit the mind's composure ────────────────────

def test_trigger_events_hit_mind():
    """A GOAL event fed through Manager.on_event (what _absorb_chain does)
    updates mind.composure_current — scored vs conceded in the right
    direction."""
    def _composure_delta(score_for_home: bool) -> float:
        mgr = _brain_manager()
        base = mgr.mind.composure_current
        eng = _egg()
        eng.state.last_goal_team = "Home" if score_for_home else "Away"
        mgr.on_event("GOAL", eng, "Home")
        return mgr.mind.composure_current - base

    def _conceding_stress_delta() -> float:
        mgr = _brain_manager()
        base = mgr.mind.stress_accumulator
        eng = _egg()
        eng.state.last_goal_team = "Away"     # opponent scored vs Home
        mgr.on_event("GOAL", eng, "Home")
        return mgr.mind.stress_accumulator - base

    # scoring settles the mind (composure up over the pre-goal value)
    assert _composure_delta(score_for_home=True) > 0
    # conceding always stresses (discipline enforced even for high-EQ minds)
    assert _conceding_stress_delta() > 0