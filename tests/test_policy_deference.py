"""PLOFA 26/27 — POLICY TARGET DEFERENCE (2026-10-19)

The on-ball policy names the ACTION; the structure names the RECEIVER on
ordinary circulation. That division is deliberate and measured (matrix 1178 /
phase 296 / wide-combo 225 / heuristic 13 / policy 1 on a live match), but it
has one blind spot: the policy's own receiver is computed every touch by
`_find_target` and then discarded, so nobody has ever known whether it is any
good.

`PossessionChain.POLICY_TARGET_DEFERENCE` is the cheap test of that question:
where the matrix is nearly indifferent between its own choice and the policy's
(in the matrix's OWN units), prefer the policy's.

This file pins four things, because each of them is a way the experiment could
report a result that is not about the question:

    P1  the matrix now exposes its own scores (`value` / `scores`), without
        which "nearly indifferent" cannot be evaluated at all
    P2  the rule is BOUNDED and every exclusion holds -- a phase directive, the
        wide combination, a keeper release and an already-applied deference are
        non-preferences and must never be outvoted
    P3  the rule draws NO random number, so an off arm and an on arm differ
        only by the branch taken
    P4  the pass event says who named the receiver (`target_authority`), with
        the flag OFF, so the question is answerable from the export instead of
        from a probe wrapping internals

Run with:  .venv\\Scripts\\python.exe -m pytest tests/test_policy_deference.py -q
"""

from __future__ import annotations

import random

import pytest

from match_engine import (
    MatchState, TeamProfile, TeamStyle, PlayingStyle, Intensity, EventType,
)
from event_chain import ChainDispatcher, PossessionChain, SituationType
from brain_integration import NeuralDecisionBrain
from decision_brain import PlayerDecision, PlayerIntent
from position_engine import PositionEngine
from player_dna import SquadBuilder
from attacking_matrix import AttackingDecision, AttackingMatrix

_ROLES = [
    ("GK", ["sweeper_keeper"]),
    ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]),
    ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]),
    ("CDM", ["anchor_man"]),
    ("CM", ["engine"]),
    ("CM", ["box_box"]),
    ("CAM", ["creator"]),
    ("LW", ["winger"]),
    ("ST", ["fox_in_box"]),
]


# ── fixtures ──────────────────────────────────────────────────────────────

def _make_squad(team_name: str):
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specs, 26)
        for i, (pos, specs) in enumerate(_ROLES)
    ]
    return SquadBuilder.build(team_name, starters)["starters"]


def _make_engine(att_squad, def_squad, att_team="Att FC", def_team="Def FC"):
    pe = PositionEngine()
    prof = TeamProfile(att_team, TeamStyle.BALANCED,
                       PlayingStyle.MIXED, Intensity.MEDIUM)
    dprof = TeamProfile(def_team, TeamStyle.BALANCED,
                        PlayingStyle.MIXED, Intensity.MEDIUM)
    pe.initialize_team(att_team, att_squad, prof, attacks_right=True)
    pe.initialize_team(def_team, def_squad, dprof, attacks_right=False)
    return pe, prof


class _P:
    """Minimal player stub -- the rule only ever reads `.name`."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_P({self.name!r})"


def _decision(target_name, scores, value, reason="clean_build_up"):
    return AttackingDecision(
        action="PROGRESSIVE_PASS",
        target=_P(target_name) if target_name else None,
        scores=dict(scores),
        value=float(value),
        reason=reason,
    )


def _active(target_name, intent=PlayerIntent.PROGRESSIVE_PASS):
    return PlayerDecision(
        intent=intent, action="PASS", confidence=0.9, reason="policy outlet",
        risk_level=0.1, decision_quality=0.9, evaluation_error=0.0,
        is_error=False, target=_P(target_name) if target_name else None,
    )


def _rule(monkeypatch, *, on=True, source="matrix", receiver="M",
          matrix=None, active=None, players=None):
    monkeypatch.setattr(PossessionChain, "POLICY_TARGET_DEFERENCE", on)
    players = players if players is not None else [_P("M"), _P("P"), _P("Q")]
    return PossessionChain._policy_target_deference(
        source,
        _P(receiver) if receiver else None,
        matrix,
        active,
        players,
    )


# ── P1: the matrix exposes the scores the rule needs ─────────────────────

def test_matrix_decision_carries_its_own_scores():
    """Without `value`/`scores`, 'nearly indifferent' is unevaluable.

    This is the load-bearing payload change: the rule compares the policy's
    option to the matrix's choice IN THE MATRIX'S UNITS, so the matrix must
    publish those units. A regression here silently turns the whole experiment
    into a no-op (the rule bails on an empty `scores`), which is exactly the
    class of defect this project keeps finding.
    """
    random.seed(11)
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = next(p for p in att if p.position == "CM")
    teammates = [p for p in att if p.name != carrier.name]
    defenders = [p for p in d if p.position != "GK"]

    dec = AttackingMatrix.decide(
        carrier, teammates, defenders, 55.0, 34.0,
        position_engine=pe, attacks_right=True, team_profile=prof,
    )

    assert dec.is_pass, "fixture must resolve a pass, or the payload is untested"
    assert dec.scores, "candidate scores must be published"
    assert dec.target is not None, "a pass decision names a target"
    assert dec.target.name in dec.scores, (
        "the chosen target must appear in the scores it is compared against"
    )
    assert abs(dec.scores[dec.target.name] - dec.value) < 1e-9, (
        "`value` must be the chosen option's own score, not the global max: "
        "the cascade deliberately consults subsets, so the decision actually "
        "made is the comparison that matters"
    )
    # Recorded deliberately, because it is NOT a fixture defect: at this
    # position every candidate the matrix weighed scored 0.0 (the
    # "no clear option" branch) while an UNCHOSEN far option scored 0.199.
    # That is the cascade working as designed, and it is exactly why `value`
    # -- not `max(scores.values())` -- is the reference the rule uses.
    assert dec.value == 0.0
    assert max(dec.scores.values()) > dec.value, (
        "this state must keep the chosen-value-vs-global-max distinction "
        "visible, or the relative-margin test below proves nothing"
    )


# ── P2: bounded, and every exclusion holds ───────────────────────────────

def test_flag_off_returns_no_verdict(monkeypatch):
    m = _decision("M", {"M": 0.50, "P": 0.50}, 0.50)
    applied, meta = _rule(monkeypatch, on=False, matrix=m,
                          active=_active("P"))
    assert applied is False
    assert meta is None, "an off flag must do no work and record nothing"


def test_only_the_matrix_is_deferred_to(monkeypatch):
    """A phase order, a role rule and an applied deference are not preferences.

    Driven through the real provenance label rather than a boolean, because
    the failure mode is a *new* source appearing and not being excluded.
    """
    m = _decision("M", {"M": 0.50, "P": 0.50}, 0.50)
    act = _active("P")
    for source in ("phase", "wide_combo", "policy_deference", ""):
        applied, meta = _rule(monkeypatch, source=source, matrix=m,
                              active=act)
        assert applied is False, f"source={source!r} must never be outvoted"
        assert meta is None


def test_keeper_release_is_safety_not_preference(monkeypatch):
    """A release valve exists because the corridors are closed; it may not be
    overruled even when the numbers happen to be close."""
    m = _decision("GK", {"GK": 0.40, "P": 0.40}, 0.40,
                  reason="back_to_keeper_under_pressure")
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"),
                          receiver="GK")
    assert applied is False
    assert meta is None


def test_within_margin_defers_to_the_policy(monkeypatch):
    m = _decision("M", {"M": 0.50, "P": 0.48}, 0.50)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"))
    assert applied is True
    assert meta["fired"] is True and meta["applied"] is True
    assert meta["matrix_target"] == "M" and meta["policy_target"] == "P"
    assert meta["gap_frac"] == pytest.approx(0.04, abs=1e-6)


def test_outside_margin_the_structure_keeps_it(monkeypatch):
    """0.40 against 0.50 is a 20% gap -- the matrix is NOT indifferent, so the
    structural claim stands and the policy's option is recorded, not applied."""
    m = _decision("M", {"M": 0.50, "P": 0.40}, 0.50)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"))
    assert applied is False
    assert meta is not None, "the gap must still be recorded for calibration"
    assert meta["fired"] is False
    assert meta["gap_frac"] == pytest.approx(0.20, abs=1e-6)


def test_the_margin_is_relative_to_the_chosen_option(monkeypatch):
    """The comparison is against the matrix's choice, never the global max.

    The cascade selects from subsets on purpose; a policy target that beats
    the chosen option's *score* is not a reason to overturn a decision the
    cascade made for a different reason. Here the chosen value is 0.20 (a
    recycle under congestion) while an unchosen far option scores 0.90, and a
    policy value of 0.20 must still be admissible -- proving the reference is
    `value`, not `max(scores.values())`.
    """
    m = _decision("M", {"M": 0.20, "P": 0.20, "Q": 0.90}, 0.20,
                  reason="no_clear_option")
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"))
    assert applied is True
    assert meta["matrix_value"] == 0.20
    assert meta["matrix_silent"] is False, "0.20 is a preference, not silence"
    assert meta["matrix_reason"] == "no_clear_option"


def test_silence_is_a_separate_mode_from_a_near_tie(monkeypatch):
    """`gap_frac == 0.0` means two different things and must be separable.

    MEASURED 2026-10-19 on the first live run: the `fired` set was DOMINATED by
    `gap_frac == 0.000`, i.e. by the `matrix_value <= 0` branch -- where the
    margin is not applied at all. So the single number conflated a comparison
    with its absence, and the margin looked like it was doing work it was not.
    These two passes both carry `gap_frac = 0.0` and must NOT be
    indistinguishable in the payload:
      * `matrix_silent` -- the chosen option scored nothing, so there was no
        relative quantity to be indifferent about; the rule fires outright.
      * a genuine near-tie -- the matrix scored its choice and the policy's
        option is within the margin.
    """
    silent = _decision("M", {"M": 0.0, "P": 0.0}, 0.0, reason="no_clear_option")
    applied, meta_s = _rule(monkeypatch, matrix=silent, active=_active("P"))
    assert applied is True
    assert meta_s["matrix_silent"] is True
    assert meta_s["gap_frac"] == 0.0
    assert meta_s["matrix_reason"] == "no_clear_option"

    tie = _decision("M", {"M": 0.400, "P": 0.400}, 0.400)
    applied, meta_t = _rule(monkeypatch, matrix=tie, active=_active("P"))
    assert applied is True
    assert meta_t["matrix_silent"] is False
    assert meta_t["gap_frac"] == 0.0
    # Same `gap_frac`, different mode -- which is the whole point.
    assert meta_s["gap_frac"] == meta_t["gap_frac"]
    assert meta_s["matrix_silent"] != meta_t["matrix_silent"]
    assert meta_s["matrix_value"] != meta_t["matrix_value"]


def test_agreement_fires_but_applies_nothing(monkeypatch):
    """The most common real case: both layers pick the same man."""
    m = _decision("P", {"M": 0.50, "P": 0.51}, 0.50)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"),
                          receiver="P")
    assert applied is False
    assert meta["fired"] is True
    assert meta["matrix_target"] == meta["policy_target"] == "P"


def test_an_unknown_policy_target_is_refused(monkeypatch):
    """`_find_target` resolving to someone outside the squad must not reach
    the pitch as a ghost receiver -- the same rule `_named_shooter` follows."""
    m = _decision("M", {"M": 0.50, "P": 0.50}, 0.50)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"),
                          players=[_P("M"), _P("Q")])
    assert applied is False
    assert meta["fired"] is True, "the score still compared -- only the squad did not"
    assert meta["applied"] is False


def test_no_matrix_scores_means_no_verdict(monkeypatch):
    """A shot decision, or a fallback with no target, carries no scores."""
    m = AttackingDecision(action="SHOOT", reason="elite_shot")
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"))
    assert applied is False and meta is None


def test_no_policy_target_means_no_verdict(monkeypatch):
    m = _decision("M", {"M": 0.50, "P": 0.50}, 0.50)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active(None))
    assert applied is False and meta is None


def test_zero_valued_options_mean_the_matrix_has_no_preference(monkeypatch):
    """When every candidate scored 0.0 the matrix is not indifferent, it is
    SILENT -- and silence is where a preference may speak.

    This case is load-bearing. The first version of the rule returned early on
    `matrix_value <= 0.0` (to protect `gap_frac` from dividing by zero), which
    made it INERT on the "no clear option" branch -- a large share of real
    passes, and precisely the ones where the structure has nothing to say.
    A gate that cannot fire is the project's standing pathology, so it is
    pinned here rather than left to a comment.
    """
    m = _decision("M", {"M": 0.0, "P": 0.0}, 0.0)
    applied, meta = _rule(monkeypatch, matrix=m, active=_active("P"))
    assert applied is True
    assert meta["fired"] is True
    assert meta["gap_frac"] == 0.0, "no division by zero is performed"
    assert meta["matrix_value"] == 0.0
    assert meta["matrix_silent"] is True, (
        "the export must be able to separate 'the margin fired' from 'there "
        "was no relative quantity to compare'")


# ── P3: the rule is a comparison, not a draw ─────────────────────────────

def test_the_rule_consumes_no_random_state(monkeypatch):
    """An off arm and an on arm must differ by ONE BRANCH, not by the whole
    downstream stream. This project has already paid for this lesson twice
    (the crc32 hash seed, the striker's deterministic coin)."""
    m = _decision("M", {"M": 0.50, "P": 0.48}, 0.50)
    monkeypatch.setattr(PossessionChain, "POLICY_TARGET_DEFERENCE", True)
    before = random.getstate()
    PossessionChain._policy_target_deference(
        "matrix", _P("M"), m, _active("P"), [_P("M"), _P("P")],
    )
    assert random.getstate() == before


# ── P4: wired end to end, and the export says who chose ──────────────────

def _run_possession(monkeypatch, *, deference, matrix_values,
                    policy_name, matrix_name):
    """Drive one real possession with a pinned policy and a pinned matrix."""
    random.seed(17)
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = next(p for p in att if p.position == "CM")
    policy_target = next(p for p in att if p.name != carrier.name
                         and p.position == "CB")
    matrix_target = next(p for p in att if p.name != carrier.name
                         and p.position == "ST")

    def policy(player, x, y, teammates, *a, **kw):
        return PlayerDecision(
            intent=PlayerIntent.SAFE_PASS, action="PASS", confidence=0.95,
            reason="policy safe outlet", risk_level=0.1, decision_quality=0.9,
            evaluation_error=0.0, is_error=False, target=policy_target,
        )

    def matrix(player, teammates, *a, **kw):
        tx, ty = pe.get_position(matrix_target.name)
        return AttackingDecision(
            action="PROGRESSIVE_PASS", target=matrix_target,
            target_x=tx, target_y=ty,
            scores={policy_target.name: matrix_values[0],
                    matrix_target.name: matrix_values[1]},
            value=matrix_values[1],
            reason="clean_build_up",
        )

    monkeypatch.setattr(NeuralDecisionBrain, "decide", staticmethod(policy))
    monkeypatch.setattr(AttackingMatrix, "decide", staticmethod(matrix))
    monkeypatch.setattr(PossessionChain, "POLICY_TARGET_DEFERENCE", deference)

    res = ChainDispatcher.possession(
        30, "Att FC", att, prof, MatchState(), 1,
        defending_players=d, position_engine=pe,
        context_x=55.0, context_y=34.0, attacks_right=True,
    )
    passes = [e for e in res.events if e.event_type == EventType.PASS]
    assert passes, "the fixture must produce a pass or nothing is measured"
    ev = passes[0]
    return ev, policy_target.name, matrix_target.name


def test_deference_off_keeps_the_matrix_and_still_reports_authority(monkeypatch):
    ev, policy_name, matrix_name = _run_possession(
        monkeypatch, deference=False, matrix_values=(0.48, 0.50),
        policy_name="P", matrix_name="M",
    )
    assert ev.secondary_player == matrix_name
    assert ev.metadata["target_authority"] == "matrix", (
        "the shipped path must SAY the structure named the receiver -- this is "
        "the honesty property, and it holds with the experiment off"
    )
    assert "target_deference" not in ev.metadata, (
        "an off flag must not change the event payload"
    )


def test_deference_on_within_margin_delivers_to_the_policy(monkeypatch):
    ev, policy_name, matrix_name = _run_possession(
        monkeypatch, deference=True, matrix_values=(0.48, 0.50),
        policy_name="P", matrix_name="M",
    )
    assert ev.secondary_player == policy_name, (
        "inside the margin the policy's receiver must actually receive"
    )
    assert ev.metadata["target_authority"] == "policy_deference"
    assert ev.metadata["target_deference"]["applied"] is True


def test_deference_on_outside_margin_is_recorded_not_applied(monkeypatch):
    ev, policy_name, matrix_name = _run_possession(
        monkeypatch, deference=True, matrix_values=(0.30, 0.50),
        policy_name="P", matrix_name="M",
    )
    assert ev.secondary_player == matrix_name
    assert ev.metadata["target_authority"] == "matrix"
    assert ev.metadata["target_deference"]["fired"] is False


def test_the_decision_authority_field_is_about_the_action(monkeypatch):
    """`decision_authority` and `target_authority` answer different questions
    and must not be conflated: the policy chose the ACTION here while the
    structure chose the RECEIVER. That distinction is the whole point of the
    new field, and it is the one the previous schema could not express."""
    ev, _, _ = _run_possession(
        monkeypatch, deference=False, matrix_values=(0.48, 0.50),
        policy_name="P", matrix_name="M",
    )
    assert ev.metadata["decision_authority"] == "player_policy"
    assert ev.metadata["target_authority"] == "matrix"


# ── P5: with the flag OFF the call is inert (metadata-only difference) ───

_NEW_KEYS = ("target_authority", "target_deference")


def _decision_digest(events):
    """Everything that decides football, and nothing that describes it."""
    out = []
    for e in events:
        out.append((
            e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
            e.player, e.secondary_player,
            round(float(e.location_x), 6), round(float(e.location_y), 6),
            None if e.end_x is None else round(float(e.end_x), 6),
            None if e.end_y is None else round(float(e.end_y), 6),
            bool(e.outcome),
            tuple(sorted((k, str(v)) for k, v in (e.metadata or {}).items()
                         if k not in _NEW_KEYS)),
        ))
    return out


def _off_arm(seed0: int, count: int, neutralize: bool):
    """One arm of the inertness check, on a freshly-built world.

    A fresh world per arm matters: sharing a PositionEngine would let the
    first arm's touches change the second arm's geometry, which is how a
    "no difference" result gets manufactured.
    """
    random.seed(seed0)
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    if neutralize:
        orig = PossessionChain._policy_target_deference.__func__

        def _noop(cls, *a, **kw):
            return False, None

        PossessionChain._policy_target_deference = classmethod(_noop)
    try:
        events = []
        for i in range(count):
            random.seed(seed0 * 1000 + i)
            res = ChainDispatcher.possession(
                10 + (i % 70), "Att FC", att, prof, MatchState(), 1,
                defending_players=d, position_engine=pe,
                context_x=40.0 + (i % 25), context_y=10.0 + (i % 48),
                attacks_right=True,
            )
            events.extend(res.events)
        return events
    finally:
        if neutralize:
            PossessionChain._policy_target_deference = classmethod(orig)


def test_the_off_path_changes_no_decision_only_the_payload(monkeypatch):
    """The shipped default must be decision-identical to not calling the rule.

    `POLICY_TARGET_DEFERENCE` is False on the shipped path, so the experiment
    is only safe if turning the CODE on without the FLAG is inert. Proved by
    running the same 60 seeded possessions twice -- once with the rule present,
    once with it replaced by a no-op -- on a freshly-built world each time, and
    comparing every field that decides football. The only permitted difference
    is the new descriptive metadata, which is stripped before comparing and
    separately asserted to be the ONLY addition.
    """
    monkeypatch.setattr(PossessionChain, "POLICY_TARGET_DEFERENCE", False)
    count = 60
    with_rule = _off_arm(11, count, neutralize=False)
    without_rule = _off_arm(11, count, neutralize=True)

    assert len(with_rule) == len(without_rule) > 0, (
        "the fixture must produce events, or the comparison is vacuous"
    )
    assert _decision_digest(with_rule) == _decision_digest(without_rule), (
        "with the flag off the rule must not move a single decision"
    )

    # And the payload difference is exactly the two new keys, nothing else.
    passes_seen = 0
    for a, b in zip(with_rule, without_rule):
        extra = set((a.metadata or {})) - set((b.metadata or {}))
        assert extra <= set(_NEW_KEYS), (
            f"unexpected payload change on {a.event_type}: {sorted(extra)}"
        )
        if a.event_type == EventType.PASS:
            passes_seen += 1
            assert "target_authority" in (a.metadata or {}), (
                "every PASS must now say who named the receiver"
            )
            assert a.metadata["target_authority"] in (
                "matrix", "phase", "wide_combo", "gk_distribution",
                "policy", "policy_deference", "heuristic", "soul", "unknown",
            ), f"undocumented authority {a.metadata['target_authority']!r}"
    assert passes_seen > 0, "the arm produced no passes, so nothing was checked"
