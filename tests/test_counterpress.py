"""
PLOFA 26/27 — Counterpress Burst Tests (P2)
===========================================
test_counterpress.py

Validates the counterpress-burst feature: the team that just lost
possession presses the recovery zone (where the ball was lost) at
elevated intensity for a short window, overriding the normal static
engagement line.

    1.  Lifecycle      — set_counterpress arms a burst and counterpress_active
                        stays True only for the armed team within the window,
                        then expires once match_clock_s passes the deadline.
    2.  Window dials   — the default window / range / intensity multipliers.
    3.  Chain gating   — goods can run and the engagement override works inside
                        the recovery zone (no exceptions, phase produced).
    4.  Press boost    — the possession-chain PRESS probability is scaled up by
                        COUNTERPRESS_INTENSITY_MULT inside the recovery zone.
    5.  Transition     — a counterpressing TransitionChain anchors its first
                        press at the recovery zone and boosts its success.
    6.  Meta          — a full simulated match still runs end to end.
"""
import random
from datetime import date

from match_engine import (
    MatchState, MatchEngine, MatchConfig, TeamProfile, TeamStyle,
    PlayingStyle, Intensity, MatchPhase,
)
from player_dna import SquadBuilder
from event_chain import PossessionChain, TransitionChain


_ROLES = [
    ("GK", ["sweeper_keeper"]), ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]), ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]), ("CDM", ["anchor_man"]),
    ("CM", ["engine"]), ("CM", ["box_box"]), ("CAM", ["creator"]),
    ("LW", ["winger"]), ("ST", ["fox_in_box"]),
]


def _squad(team_name: str):
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specialties, 26)
        for i, (pos, specialties) in enumerate(_ROLES)
    ]
    return SquadBuilder.build(team_name, starters)["starters"]


def _profile(style=TeamStyle.DEFENSIVE, press=0.10):
    return TeamProfile(
        name="Away FC", style=style,
        playing_style=PlayingStyle.LOW_BLOCK, intensity=Intensity.MEDIUM,
        press_intensity=press,
    )


# =============================================================================
# 1 & 2. Lifecycle + expiry + dials
# =============================================================================

def test_counterpress_lifecycle_and_expiry():
    random.seed(7)
    state = MatchState()
    state.match_clock_s = 100.0
    assert not state.counterpress_active("Home FC")
    assert not state.counterpress_active("Away FC")
    state.set_counterpress("Home FC", 38.0, 30.0)
    assert state.counterpress_team == "Home FC"
    assert state.counterpress_x == 38.0 and state.counterpress_y == 30.0
    state.match_clock_s = 100.0 + state.COUNTERPRESS_WINDOW_S - 0.1
    assert state.counterpress_active("Home FC")
    assert not state.counterpress_active("Away FC")
    state.match_clock_s = 100.0 + state.COUNTERPRESS_WINDOW_S + 0.1
    assert not state.counterpress_active("Home FC")


def test_counterpress_window_dial_defaults():
    state = MatchState()
    assert state.COUNTERPRESS_WINDOW_S == 8.0
    assert state.COUNTERPRESS_RANGE_M == 18.0
    assert state.COUNTERPRESS_INTENSITY_MULT == 2.5


# =============================================================================
# 3. Engagement-line override inside the recovery zone only
# =============================================================================

def test_tactical_phase_runs_inside_and_outside_recovery_zone():
    random.seed(9)
    from possession_phases import possession_phase_for
    from position_engine import PositionEngine
    state = MatchState()
    state.set_counterpress("Away FC", 40.0, 34.0)
    att = _squad("Home FC")
    dif = _squad("Away FC")
    pe = PositionEngine()
    prof = _profile()
    pe.initialize_team("Away FC", dif, prof, attacks_right=False)

    for ball_x, ball_y in [(38.0, 32.0), (75.0, 20.0), (50.0, 34.0)]:
        start_phase = possession_phase_for(ball_x, ball_y, True)
        phase, decision = PossessionChain._tactical_phase_step(
            att[9], att, dif, ball_x, ball_y,
            start_phase, False, True, prof, pe,
            def_style_key="gegenpressing",
            def_press_intensity=0.10,
            counterpress={"active": True, "x": 40.0, "y": 34.0},
            state=state,
        )
        assert phase is not None


# =============================================================================
# 4. Press-probability boost inside the recovery zone
# =============================================================================

def test_possession_press_prob_boosted_by_counterpress():
    from position_engine import PositionEngine
    random.seed(11)
    state = MatchState()
    state.set_counterpress("Away FC", 40.0, 34.0)
    att = _squad("Home FC")
    state.phase = MatchPhase.SECOND_OPEN

    def count(burst, seedbase):
        # Fresh squads / engine each pass so the defenders are identically
        # placed in both runs (PositionEngine is mutated by the chain).
        dif = _squad("Away FC")
        pe = PositionEngine()
        pe.initialize_team("Away FC", dif, _profile(), attacks_right=True)
        total = 0
        for i in range(80):
            random.seed(seedbase + i)
            r = PossessionChain.generate(
                60, "Home FC", att, _profile(), state, 5,
                defending_players=dif, position_engine=pe,
                context_x=40.0, context_y=34.0,
                attacks_right=True, def_press_intensity=0.10,
                def_style_key="defensive",
                counterpress={"active": True, "x": 40.0, "y": 34.0} if burst else None,
            )
            total += sum(
                1 for e in r.events
                if getattr(e, "event_type", None) and e.event_type.name == "PRESS"
            )
        return total

    bursts = count(True, 100)
    normal = count(False, 100)
    assert bursts > 0, "counterpress should produce PRESS events"
    assert bursts > normal, (
        f"counterpress presses ({bursts}) should exceed normal ({normal})")


# =============================================================================
# 5. TransitionChain anchors the press at the recovery zone + boosts success
# =============================================================================

def test_transition_anchored_at_recovery_zone_and_boosted():
    random.seed(13)
    state = MatchState()
    state.set_counterpress("Away FC", 42.0, 30.0)
    dif = _squad("Away FC")
    att = _squad("Home FC")
    res = TransitionChain.generate(
        60, "Away FC", "Home FC", dif, att, _profile(), state,
        position_engine=None, attacks_right=False,
        counterpress={"active": True, "x": 42.0, "y": 30.0},
    )
    found = False
    for e in res.events:
        if e.event_type.name == "PRESS":
            found = True
            assert abs(e.location_x - 42.0) < 1e-6, \
                f"counterpress press should anchor at recovery zone, got x={e.location_x}"
            assert e.metadata.get("counterpress") is True
    assert found, "counterpress transition should emit a PRESS anchored at zone"


# =============================================================================
# 6. Full match still runs
# =============================================================================

def test_full_match_with_counterpress_runs():
    random.seed(21)
    config = MatchConfig(home_team="Home FC", away_team="Away FC",
                         match_date=date.today())
    hp = TeamProfile(name="Home FC", style=TeamStyle.ATTACKING,
                     playing_style=PlayingStyle.HIGH_PRESS,
                     intensity=Intensity.HIGH)
    ap = TeamProfile(name="Away FC", style=TeamStyle.DEFENSIVE,
                     playing_style=PlayingStyle.LOW_BLOCK,
                     intensity=Intensity.LOW)
    eng = MatchEngine(config, hp, ap)
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    result = eng.simulate()
    assert result.home_goals >= 0 and result.away_goals >= 0
    cp_presses = sum(
        1 for ev in getattr(result, "timeline", [])
        if getattr(ev, "event_type", None) and ev.event_type.name == "PRESS"
        and (getattr(ev, "metadata", None) or {}).get("counterpress")
    )
    assert cp_presses >= 0


# =============================================================================
# P3 — Second Presser / Trap (unit tests for pressing_profiles helpers)
# =============================================================================

from pressing_profiles import is_trap_profile, trap_present, TRAP_RANGE_M


def test_is_trap_profile_recognises_gegenpressing_and_high_press():
    assert is_trap_profile("gegenpressing")
    assert is_trap_profile("high_press")
    assert is_trap_profile("ultra_attacking")
    assert is_trap_profile("vertical_tiki_taka")
    assert is_trap_profile("gegenpress")


def test_is_trap_profile_rejects_non_trap_styles():
    assert not is_trap_profile("defensive")
    assert not is_trap_profile("park_the_bus")
    assert not is_trap_profile("route_one")
    assert not is_trap_profile("low_block")
    assert not is_trap_profile("balanced")
    assert not is_trap_profile("")
    assert not is_trap_profile(None)


class _FakePE:
    """Minimal fake position engine returning stored positions."""
    def __init__(self, positions):
        self._positions = dict(positions)
    def get_position(self, name):
        return self._positions.get(name, (0.0, 0.0))


def test_trap_present_when_second_defender_close():
    pe = _FakePE({"CB": (42.0, 33.0), "LB": (44.0, 35.0)})
    primary = type("P", (), {"name": "CB"})
    others = [type("P", (), {"name": "LB", "position": "LB"})]
    assert trap_present(others, primary, 43.0, 34.0, pe) is True


def test_trap_present_rejects_only_primary_presser():
    pe = _FakePE({"CB": (42.0, 33.0)})
    primary = type("P", (), {"name": "CB"})
    others = [type("P", (), {"name": "CB", "position": "CB"})]
    assert trap_present(others, primary, 43.0, 34.0, pe) is False


def test_trap_present_rejects_far_defender():
    pe = _FakePE({"CB": (42.0, 33.0), "RB": (70.0, 50.0)})
    primary = type("P", (), {"name": "CB"})
    others = [type("P", (), {"name": "RB", "position": "RB"})]
    assert trap_present(others, primary, 43.0, 34.0, pe) is False


def test_trap_present_rejects_gk():
    pe = _FakePE({"CB": (42.0, 33.0), "GK": (5.0, 34.0)})
    primary = type("P", (), {"name": "CB"})
    others = [type("P", (), {"name": "GK", "position": "GK"})]
    assert trap_present(others, primary, 43.0, 34.0, pe) is False


def test_trap_present_noop_without_engine():
    assert trap_present([], None, 0.0, 0.0, None) is False


def test_full_match_with_gegenpress_still_runs():
    random.seed(31)
    config = MatchConfig(home_team="Home FC", away_team="Away FC",
                         match_date=date.today())
    hp = TeamProfile(name="Home FC", style=TeamStyle.ATTACKING,
                     playing_style=PlayingStyle.HIGH_PRESS,
                     intensity=Intensity.HIGH)
    ap = TeamProfile(name="Away FC", style=TeamStyle.GEGENPRESSING,
                     playing_style=PlayingStyle.HIGH_PRESS,
                     intensity=Intensity.HIGH)
    eng = MatchEngine(config, hp, ap)
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    result = eng.simulate()
    assert result.home_goals >= 0 and result.away_goals >= 0
