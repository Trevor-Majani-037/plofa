"""Style-conditioned deep circulation (2026-11-26).

Flag-gated and DEFAULT OFF: possession TeamStyles (tiki_taka,
structured_possession, vertical_tiki_taka) let CB/CDM/CM own the support web,
the way real possession sides do. This is a structure knob — WHOM the support
pass looks for — never a verb or weight edit. Non-possession styles, and every
run with PLOFA_DEEP_CIRC unset, must stay byte-identical.
"""
import possession_phases as pp


def _engine(style="tiki_taka"):
    return pp.PossessionPhaseEngine(
        pp.GKSnapshot("GK", 5.0, 34.0, lane_open=True),
        style_key=style,
    )


def test_default_off_returns_the_shared_table_itself():
    assert pp.deep_circulation_weights("tiki_taka", False) is pp.CIRCULATION_ROLE_WEIGHT
    assert pp.deep_circulation_weights("route_one", True) is pp.CIRCULATION_ROLE_WEIGHT
    assert pp.deep_circulation_weights("unknown_style", True) is pp.CIRCULATION_ROLE_WEIGHT


def test_enabled_possession_style_boosts_only_deep_roles():
    out = pp.deep_circulation_weights("tiki_taka", True)
    assert out is not pp.CIRCULATION_ROLE_WEIGHT
    for role in ("CB", "CDM", "CM"):
        assert out[role] > pp.CIRCULATION_ROLE_WEIGHT[role]
    for role in ("LB", "RB", "CAM", "LW", "RW", "ST", "CF"):
        assert out[role] == pp.CIRCULATION_ROLE_WEIGHT[role]
    assert pp.CIRCULATION_ROLE_WEIGHT["CB"] == 1.05, "module table must not be mutated"


def test_engine_defaults_to_the_shared_table(monkeypatch):
    monkeypatch.delenv("PLOFA_DEEP_CIRC", raising=False)
    eng = _engine("tiki_taka")
    assert eng._role_weights is pp.CIRCULATION_ROLE_WEIGHT


def test_engine_flag_is_style_conditioned(monkeypatch):
    monkeypatch.setenv("PLOFA_DEEP_CIRC", "1")
    pos = _engine("structured_possession")
    direct = _engine("fluid_counter")
    assert pos._role_weights["CB"] == pp.DEEP_CIRC_ROLE_OVERRIDES["CB"]
    assert direct._role_weights is pp.CIRCULATION_ROLE_WEIGHT


def test_sustain_probability_only_moves_when_active(monkeypatch):
    monkeypatch.delenv("PLOFA_DEEP_CIRC", raising=False)
    off = _engine("tiki_taka")
    monkeypatch.setenv("PLOFA_DEEP_CIRC", "1")
    on = _engine("tiki_taka")
    phase = pp.PossessionPhase.MIDFIELD_CIRCULATION
    assert on._sustain_probability(phase) == off._sustain_probability(phase) + pp.DEEP_CIRC_SUSTAIN_BUMP
    assert on._sustain_probability(phase) <= 0.80


def test_direct_style_ignores_the_flag_for_sustain(monkeypatch):
    monkeypatch.delenv("PLOFA_DEEP_CIRC", raising=False)
    off = _engine("route_one")
    monkeypatch.setenv("PLOFA_DEEP_CIRC", "1")
    on = _engine("route_one")
    phase = pp.PossessionPhase.MIDFIELD_CIRCULATION
    assert on._sustain_probability(phase) == off._sustain_probability(phase)


def test_gk_recycle_bump_only_for_active_possession_style(monkeypatch):
    monkeypatch.delenv("PLOFA_DEEP_CIRC", raising=False)
    off = _engine("tiki_taka")
    monkeypatch.setenv("PLOFA_DEEP_CIRC", "1")
    on = _engine("tiki_taka")
    direct = _engine("route_one")
    assert on._gk_recycle_rate() == min(0.85, off._gk_recycle_rate() + pp.DEEP_CIRC_GK_BUMP)
    assert direct._gk_recycle_rate() == 0.12


def test_no_silent_role_weight_mutation_across_engines(monkeypatch):
    monkeypatch.setenv("PLOFA_DEEP_CIRC", "1")
    a = _engine("tiki_taka")
    b = _engine("balanced")
    assert a._role_weights["CB"] == pp.DEEP_CIRC_ROLE_OVERRIDES["CB"]
    assert b._role_weights is pp.CIRCULATION_ROLE_WEIGHT
