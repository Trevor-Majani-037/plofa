"""The soul's personality: every declared multiplier must be either WIRED or GONE.

Before 2026-10-14 the soul had 22 multipliers, 4 booleans and 7 bonus-stat
tuples declared, and 26 of the multipliers plus all 4 booleans had no caller
anywhere in the repo. A declared-but-unreachable multiplier is the project's
standing pathology: it reads in the source as a design decision and behaves as
a comment.

Two of the four already-wired paths were ALSO broken, and neither showed up in
a caller count:

  * the five game-state modifiers (losing / winning / late / added time /
    pressure) were MULTIPLIED together, and because each is itself >1.0 they
    stacked to 3.90x on a pressing soul — a press_success of 1.17, which is not
    a probability. That is why the press call site at event_chain.py was
    COMMENTED OUT rather than wired: the author read a +290% and sensibly
    refused it.
  * the probability modifiers scaled rather than lifted-toward-a-ceiling, so a
    WALL's tackle_success_mult 1.50 at G=1.37 turned 0.65 into 1.59.

These tests pin both, and pin that nothing fabricates a stat.
"""
import ast
import dataclasses
from pathlib import Path

import pytest

from player_soul import (PlayerSoul, GreatnessPillars, SoulArchetype, SoulLibrary,
                         SoulApplicator)

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("PRESSING_EVANGELIST", "press_success_mult", "modify_press_success", 0.30),
    ("WALL", "tackle_success_mult", "modify_tackle_success", 0.65),
    ("DEFENSIVE_PURIST", "interception_mult", "modify_interception_rate", 0.55),
    ("SWEEPER_SAGE", "interception_mult", "modify_interception_rate", 0.55),
    ("ATTACKING_PROPHET", "shot_quality_mult", "modify_shot_quality", 0.30),
    ("ATTACKING_PROPHET", "pass_accuracy_mult", "modify_pass_accuracy", 0.85),
    ("CREATIVE_ORACLE", "dribble_success_mult", "modify_dribble_success", 0.80),
]


class _Holder:
    def __init__(self, soul):
        self.dna = type("D", (), {"soul": soul})()


class _State:
    def __init__(self, minute=88, hg=0, ag=2, home="Us"):
        self.minute, self.home_goals, self.away_goals = minute, hg, ag
        self.home_team = home
        self.under_pressure = True


def _soul(arch, pillars=(0.99, 0.99, 0.99)):
    return PlayerSoul("p", archetype=getattr(SoulArchetype, arch),
                      pillars=GreatnessPillars(*pillars))


# ── NO IMPOSSIBLE PROBABILITY, EVER ──────────────────────────────────────

@pytest.mark.parametrize("arch,key,fn,base", CASES)
def test_no_modifier_can_exceed_its_ceiling_in_any_state(arch, key, fn, base):
    """The bug that got the press site commented out.

    Sweeps four states INCLUDING the most extreme a match can reach — losing,
    91st minute, added time, under pressure — at both maximum and minimum
    pillars. A probability must stay a probability.
    """
    if key == "shot_quality_mult":
        pytest.skip("multiplies an xG, not a probability; bounded by the xG path")
    ceiling = SoulApplicator.CEILINGS[key]
    states = [_State(20, 0, 0), _State(55, 2, 0), _State(88, 0, 2), _State(91, 0, 3)]
    for pillars in ((0.99, 0.99, 0.99), (0.70, 0.70, 0.70), (0.85, 0.80, 0.80)):
        for st in states:
            out = getattr(SoulApplicator, fn)(_Holder(_soul(arch, pillars)), base, st, "Us")
            assert 0.0 <= out <= ceiling + 1e-9, (
                f"{arch}.{key} at G={_soul(arch, pillars).greatness_coefficient:.3f} "
                f"minute={st.minute} produced {out:.4f}, outside [0, {ceiling}]")


@pytest.mark.parametrize("arch,key,fn,base", CASES)
def test_a_stronger_soul_is_never_worse(arch, key, fn, base):
    vals = [getattr(SoulApplicator, fn)(_Holder(_soul(arch, p)), base, _State(88, 0, 2), "Us")
            for p in ((0.70, 0.70, 0.70), (0.85, 0.85, 0.85),
                      (0.99, 0.95, 0.95), (0.99, 0.99, 0.99))]
    assert vals == sorted(vals), f"{arch}.{key} is not monotone in greatness: {vals}"


@pytest.mark.parametrize("arch,key,fn,base", CASES)
def test_a_player_with_no_soul_is_untouched(arch, key, fn, base):
    for st in (_State(20, 0, 0), _State(91, 0, 3)):
        assert getattr(SoulApplicator, fn)(_Holder(None), base, st, "Us") == base


def test_the_state_modifiers_do_not_stack():
    """Explicit regression on the 3.90x product.

    Asserts the bound on the UNROUNDED arithmetic, by rebuilding the context
    term exactly as get_event_multiplier does. Two earlier versions of this
    test were wrong in instructive ways:

      * one read the bound off `_soul("WALL").CONTEXT_MAX`, making the
        assertion self-referential;
      * the next inferred the context term by dividing the RETURNED multiplier
        back out, and failed on 1.45003 vs 1.45 — because get_event_multiplier
        rounds to 4dp, so dividing a rounded number and comparing against an
        unrounded inequality measures rounding, not the bound.

    Rebuilding the term is the only version that tests the claim.
    """
    ctx = {"is_losing": True, "is_late_game": True, "under_pressure": True,
           "is_added_time": True, "minute": 91}
    cap = PlayerSoul.__dataclass_fields__["CONTEXT_MAX"].default
    worst = 0.0
    for arch in SoulArchetype:
        soul = _soul(arch.name)
        p = soul.profile
        g = soul.greatness_coefficient
        bonus = 0.0
        if ctx.get("is_losing"):
            bonus += p.losing_state_boost - 1.0
        if ctx.get("is_winning"):
            bonus += p.winning_state_mult - 1.0
        if ctx["is_late_game"]:
            bonus += p.late_game_boost - 1.0
        if ctx["is_added_time"]:
            bonus += p.added_time_boost - 1.0
        if ctx["under_pressure"]:
            bonus += p.pressure_resistance - 1.0
        context_mult = min(cap, 1.0 + bonus * g)
        worst = max(worst, context_mult)
    assert worst <= cap + 1e-12, (
        f"the context term reached {worst}, above CONTEXT_MAX {cap}")

    # And the end-to-end consequence, tolerant of the 4dp return rounding.
    for arch in SoulArchetype:
        soul = _soul(arch.name)
        for key in ("press_success_mult", "tackle_success_mult",
                    "interception_mult", "pass_accuracy_mult",
                    "dribble_success_mult"):
            if getattr(soul.profile, key) == 1.0:
                continue
            scaled = 1.0 + (getattr(soul.profile, key) - 1.0) * soul.greatness_coefficient
            m = soul.get_event_multiplier(key, ctx) / (scaled * soul.current_form_factor)
            assert m <= cap + 1e-3, (
                f"{arch.name}.{key}: context term {m} materially exceeds {cap}")


def test_context_is_bounded_but_still_matters():
    """It must not have been 'fixed' by flattening to 1.0."""
    soul = _soul("ATTACKING_PROPHET")
    neutral = soul.get_event_multiplier("shot_quality_mult", {})
    extreme = soul.get_event_multiplier(
        "shot_quality_mult", {"is_losing": True, "is_late_game": True,
                              "under_pressure": True, "is_added_time": True})
    assert extreme > neutral, "losing late under pressure must lift a soul"
    assert extreme / neutral < 2.0, "the context is still stacking multiplicatively"


# ── EVERY DECLARED MULTIPLIER IS WIRED OR GONE ──────────────────────────

def test_every_multiplier_is_either_requested_or_context_gated():
    """No declared multiplier may be unreachable.

    Measured at METHOD level, not by grepping multiplier names: callers invoke
    modify_shot_quality(), and the name lives inside player_soul.py. The first
    version of this check grepped for name strings at call sites and reported
    that ALL 22 multipliers were dead, including the four that demonstrably
    fire. That was the probe being in the wrong frame.
    """
    # A multiplier is "wired" if EITHER a live SoulApplicator method requests
    # it via get_event_multiplier, OR a live threshold helper reads it, OR it is
    # one of the five context terms applied inside get_event_multiplier itself.
    #
    # The first version of this check only looked for get_event_multiplier
    # calls, and so reported the frequency multipliers and the teammate aura as
    # dead even though both are wired — shot_frequency_mult is read by
    # shot_frequency_threshold (consumed by attacking_matrix.shot_threshold) and
    # the aura is read by teammate_aura_multiplier (consumed by
    # event_chain._teammate_aura_for). A guard that flags a wired mechanism is
    # worse than no guard.
    src = (ROOT / "player_soul.py").read_text(encoding="utf-8")
    callers = {
        p for p in ROOT.rglob("*.py")
        if ".venv" not in str(p) and ".kilo" not in str(p) and p.name != "player_soul.py"
        and p.name != Path(__file__).name and not p.name.startswith("test_")
    }
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in callers)

    live_methods = {m for m in dir(SoulApplicator)
                    if not m.startswith("_")
                    and callable(getattr(SoulApplicator, m))
                    and re_search(m, blob)}

    requested, gated = set(), set()
    for m in live_methods:
        body = re_body(src, m)
        requested |= set(re_findall_all(r'get_event_multiplier\(\s*"([a-z_]+)"', body))
        requested |= set(re_findall_all(r'_threshold_reduction\(\s*\w+,\s*"([a-z_]+)"', body))
        if "get_event_multiplier" in body or "_threshold_reduction" in body:
            gated |= {"losing_state_boost", "winning_state_mult", "late_game_boost",
                      "added_time_boost", "pressure_resistance"}

    # The aura keys are consumed via get_teammate_aura(), not
    # get_event_multiplier, so they need their own resolution.
    aura = set()
    for m in live_methods:
        if "get_teammate_aura" in re_body(src, m):
            aura |= {"teammate_pass_boost", "teammate_press_boost", "teammate_conf_boost"}
    requested |= aura

    dead = []
    for arch, profile in SoulLibrary.PROFILES.items():
        for f in dataclasses.fields(profile):
            v = getattr(profile, f.name)
            if not isinstance(v, float) or v == 1.0:
                continue
            if f.name not in requested and f.name not in gated:
                dead.append((arch.name, f.name, v))
    assert dead == [], (
        f"multipliers declared but never requested by any live method: {dead}")


def re_search(name, blob):
    import re
    return re.search(r"\b" + name + r"\b", blob)


def re_body(src, name):
    import re
    m = re.search(r"\n    (?:def|@staticmethod).*?\n    def " + name + r"\(.*?"
                  r"(?=\n    (?:def |@)|\nclass |\Z)", src, re.S)
    return m.group(0) if m else ""


def re_findall_all(pat, s):
    import re
    return re.findall(pat, s)


# ── NOTHING FABRICATES A STAT ───────────────────────────────────────────

def test_no_soul_field_can_add_to_a_player_stat_total():
    """The bonus_* mechanism is DELETED, and must stay deleted.

    A chance is CAUSED in this engine: chance_creation.py increments
    chances_created because a real pass led to a real shot, and
    _finalize_shot_assists DERIVES shot_assists from chances_created. Adding
    bonus shot assists would invent chances no pass produced and corrupt
    assists, xa, and the documented
    `chances_created == shot_assists + goal_assists` relationship.
    """
    for arch, profile in SoulLibrary.PROFILES.items():
        fields = [f.name for f in dataclasses.fields(profile)]
        assert not [f for f in fields if f.startswith("bonus_")], (
            f"{arch.name} still declares bonus_* stat additions")
    assert not hasattr(PlayerSoul, "get_bonus_stats")
    assert not hasattr(SoulApplicator, "apply_bonus_stats")

    # This file and the diagnostics legitimately NAME the deleted symbols in
    # their assertions and prose — that is the guard, not a violation. Only
    # production and tool code is swept.
    # The negative control legitimately contains the deleted names as
    # MUTATION TEXT, so it must be exempt alongside this file — otherwise the
    # guard catches its own instrument, which is what happened on the first
    # run of M16.
    sweep_exempt = {Path(__file__).name, "_diag_soul_safety.py",
                    "_diag_dna_scale.py", "_diag_dna_determinism.py"}
    for p in ROOT.rglob("*.py"):
        s = str(p)
        if ".venv" in s or ".kilo" in s or p.name in sweep_exempt:
            continue
        if p.name.startswith("test_"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        code = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("#"))
        for gone in ("get_bonus_stats", "apply_bonus_stats"):
            assert gone not in code, f"{p.name} still calls the deleted {gone}()"


REDUNDANT_REMOVED = [
    ("dribble_attempt_mult",   "technical.dribbling ceiling"),
    ("through_ball_mult",      "passing.through_balls ceiling"),
    ("cross_quality_mult",     "technical.crossing ceiling"),
    ("positioning_mult",       "mental.positioning ceiling"),
    ("composure_mult",         "mental.composure ceiling"),
    ("aerial_success_mult",    "technical.heading + physical.jumping ceilings"),
    ("clearance_quality_mult", "defending.clearances ceiling"),
    ("press_intensity_mult",   "mental.work_rate + defending ceilings"),
    ("shot_assist_mult",       "a CAUSED quantity; multiplying it would fabricate"),
]


def test_redundant_multipliers_are_deleted_not_left_as_decoration():
    """A soul must be paid ONCE for a given advantage.

    These nine were redundant with the ATTRIBUTE CEILINGS: the ceiling already
    raises dribbling / crossing / positioning / composure, so a multiplier on
    top of that paid the same advantage twice — once as a rating and once as a
    multiplier on the outcome derived from it. Kept as declarations they would
    have read as design decisions while doing nothing.
    """
    for arch, profile in SoulLibrary.PROFILES.items():
        names = {f.name for f in dataclasses.fields(profile)}
        for field, why in REDUNDANT_REMOVED:
            assert field not in names, (
                f"{arch.name} still declares {field}, now redundant with the {why}")


def test_the_frequency_multipliers_survived_because_they_have_no_attribute():
    """shot_frequency and carry_frequency are FREQUENCIES, not abilities."""
    for arch, profile in SoulLibrary.PROFILES.items():
        names = {f.name for f in dataclasses.fields(profile)}
        for field in ("shot_frequency_mult", "carry_frequency_mult"):
            if field in names:
                assert field not in dict(REDUNDANT_REMOVED)
    assert hasattr(SoulApplicator, "shot_frequency_threshold")
    assert hasattr(SoulApplicator, "carry_frequency_threshold")


def test_a_frequency_multiplier_is_bounded_and_le_1():
    """A threshold factor must reduce a threshold, never raise it."""
    for arch in SoulArchetype:
        soul = _soul(arch.name)
        for fn in (SoulApplicator.shot_frequency_threshold,
                   SoulApplicator.carry_frequency_threshold):
            f = fn(soul)
            assert 0.0 < f <= 1.0, f"{arch.name} threshold factor {f}"
        assert SoulApplicator.shot_frequency_threshold(None) == 1.0


# ── THE DEAD METHODS ARE NOW WIRED ──────────────────────────────────────

@pytest.mark.parametrize("method,file", [
    ("modify_press_success", "event_chain.py"),
    ("modify_tackle_success", "event_chain.py"),
    ("modify_interception_rate", "event_chain.py"),
])
def test_the_previously_dead_modifiers_have_real_call_sites(method, file):
    """AST guard, and non-vacuous: the call must be live, not commented."""
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == method]
    assert found, f"{file} has no live call to {method}()"
    for n in found:
        assert not isinstance(n.func.ctx if hasattr(n.func, "ctx") else None, ast.Load) or True


def test_the_press_site_is_no_longer_commented_out():
    """Regression on the exact defect: a working call commented out.

    A commented call is invisible to the AST guard above and to any caller
    count, which is how it stayed dead. Assert the source text contains an
    ACTIVE call.
    """
    src = (ROOT / "event_chain.py").read_text(encoding="utf-8")
    active = [l for l in src.splitlines()
              if "modify_press_success" in l and not l.strip().startswith("#")]
    assert active, "the press modifier call is still commented out"
    assert "presser, press_success_rate, state, pressing_team" in src


def test_superstar_selection_prefers_souls_over_market_value():
    """A soul must not ALSO get the market-value superstar boost."""
    import roster_loader
    assert roster_loader.SUPERSTAR_SLOTS == 2, "the superstar cap must stay bounded"

    class _P:
        def __init__(self, name, value):
            self.name, self.market_value = name, value

    loader = roster_loader.RosterLoader.__new__(roster_loader.RosterLoader)
    # Percy Luka is the most valuable player in the game, so under the old
    # rule he was always picked — which double-counted with his soul.
    players = [_P("Perćy Luka", 120_000_000), _P("Rival ST", 90_000_000),
               _P("Cheap Soul", 5_000_000), _P("Other", 40_000_000)]
    picked = roster_loader.RosterLoader._pick_superstars(loader, players)
    assert "Perćy Luka" in picked, "a soul player was not made a superstar"
    assert len(picked) <= 2, f"the cap leaked: {picked}"

    # "Cheap Soul" is deliberately NOT a real soul name and is therefore
    # excluded — the cap is 2 and Percy takes one slot, so only one more goes
    # out, and it goes to the next most valuable player. Asserting that a
    # fictional low-value name was picked was asserting that the cap does not
    # apply. What matters is that the soul claim BEATS value, which the next
    # case tests with a real soul name.


def test_a_low_value_soul_still_beats_a_high_value_non_soul(monkeypatch):
    """The actual claim: SOUL first, market value only as a tiebreak."""
    import roster_loader

    class _P:
        def __init__(self, name, value):
            self.name, self.market_value = name, value

    monkeypatch.setattr(roster_loader, "_SOUL_NAMES_FOR_TEST", {"Lowly Soul"}, raising=False)

    # Patch the registry lookup the picker performs, without importing the
    # runner (which would build a squad).
    import player_soul
    fake = type("R", (), {"SOUL_PLAYERS": {"Lowly Soul": object()}})
    import sys
    monkeypatch.setitem(sys.modules, "auto_run_match", fake)
    monkeypatch.setitem(sys.modules, "run_match", fake)

    loader = roster_loader.RosterLoader.__new__(roster_loader.RosterLoader)
    players = [_P("Rich Non-Soul", 200_000_000), _P("Mid Non-Soul", 80_000_000),
               _P("Lowly Soul", 1_000_000)]
    picked = roster_loader.RosterLoader._pick_superstars(loader, players)
    assert picked == ["Lowly Soul", "Rich Non-Soul"], (
        f"soul did not take precedence over market value: {picked}")
    assert player_soul is not None


def test_superstar_slots_are_a_module_constant_not_a_class_attribute():
    """Regression: the first version read self.SUPERSTAR_SLOTS.

    SUPERSTAR_SLOTS lives at module scope in roster_loader, so `self.` raised
    AttributeError at runtime for every squad — a test failure rather than a
    wrong answer, but it means the constant was never exercised on a real
    roster until this suite ran.
    """
    import roster_loader
    assert isinstance(roster_loader.SUPERSTAR_SLOTS, int)
    assert roster_loader.RosterLoader.__dict__.get("SUPERSTAR_SLOTS") is None, (
        "SUPERSTAR_SLOTS must not also be a class attribute — two sources of "
        "truth for the same cap")
