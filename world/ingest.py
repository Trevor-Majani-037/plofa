"""
PLOFA WORLD — engine → world ingest.
=====================================
world/ingest.py  ·  Phase 6/7 integration.

The two halves of the world layer meet here. Until this module existed, the
ledger's player lines were synthetic (see ``world/proof.py``), which proved the
ledger's arithmetic but not that real engine output survives the crossing.

The one rule that governs this file
----------------------------------
**The ledger never re-derives anything the engine already decided.**

Plan §13 is explicit that existing squad/fitness logic is reused rather than
duplicated, so this adapter is a *translator*, not a second source of truth:

  * per-player match lines come from ``exporter.PLOFAExporter.stats`` — the same
    dict the xlsx/csv/JSON exports use, so the world and the published reports
    can never disagree about what a player did;
  * fatigue, fitness and injuries come from
    ``squad_manager.SubstitutionController.stamina`` verbatim;
  * ``confidence`` and ``form`` live in ``SeasonState``, not in a
    ``MatchResult``, so they are passed in by the caller and default to ``None``
    — meaning "the engine did not say", which the ledger reads as "keep what you
    had". Nothing is invented.

Identity is the other half of the job. The ledger is keyed by canonical ID, the
engine speaks names, so every name is resolved through a
:class:`~world.ids.NameAdapter` and only falls back to a derived ID when the
roster has never heard of that player.

Non-persistence: this module writes only to the :class:`WorldLedger` it is
handed. It never touches ``season_state.json``, ``season_stats.json`` or
``plofa_output/`` — the 26/27 dual-write rule (audit §14) holds in the crossing
direction too.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence

from world.ids import NameAdapter, canonical_key, mint_id
from world.ledger import MatchReport, PlayerMatchLine, WorldLedger

#: The exporter splits shots three ways and has no single "shots" key, so the
#: ledger's one field is their sum. Stated here rather than buried in an
#: expression, because this is a definitional choice someone will question.
SHOT_PARTS = ("shots_on_target", "shots_off_target", "shots_blocked_att")


# ─────────────────────────────────────────────
# IDENTITY
# ─────────────────────────────────────────────

def resolve_player_id(name: str, adapter: Optional[NameAdapter] = None) -> str:
    """Canonical world ID for an engine player name.

    Resolved through the roster adapter when one is supplied, so the same person
    is the same identity across every competition. A name the roster has never
    seen (a cup guest, a generated player) still gets a stable derived ID rather
    than being dropped — a player who appears in a match must be recordable.
    """
    if adapter is not None:
        pid = adapter.to_id("player", name)
        if pid is not None:
            return pid
    return mint_id("player", canonical_key(name))


# ─────────────────────────────────────────────
# ENGINE VALUE EXTRACTION (verbatim, never re-derived)
# ─────────────────────────────────────────────

def _stamina_lookup(result: Any,
                    sub_controller: Any = None) -> Dict[str, Any]:
    """The engine's authoritative per-player stamina/injury state.

    **The engine does not attach the substitution controller to
    ``MatchResult``** — ``MatchResult`` carries config/state/timeline/squads and
    nothing else, so post-match stamina lives on the ``SubstitutionController``
    the caller created and handed to ``set_stamina_controller``. The controller
    must therefore be passed in explicitly; the result is searched as a
    fallback for harnesses that do attach it.

    Returns an empty mapping when neither is available, so the caller degrades to
    "the engine said nothing" instead of crashing.
    """
    for candidate in (sub_controller,
                      getattr(result, "sub_controller", None),
                      getattr(getattr(result, "state", None),
                              "sub_controller", None)):
        stamina = getattr(candidate, "stamina", None)
        if isinstance(stamina, dict):
            return stamina
    return {}


def _shots(stat: Dict[str, Any]) -> int:
    return int(sum(int(stat.get(k, 0) or 0) for k in SHOT_PARTS))


def _minutes(stat: Dict[str, Any]) -> int:
    return int(stat.get("minutes_played", 0) or 0)


# ─────────────────────────────────────────────
# THE TRANSLATOR
# ─────────────────────────────────────────────

def match_report_from_result(
    result: Any,
    player_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    competition_id: str,
    season_id: str,
    home_id: str,
    away_id: str,
    match_date: Optional[date] = None,
    round_name: str = "",
    stage_name: str = "",
    is_neutral: bool = False,
    adapter: Optional[NameAdapter] = None,
    sub_controller: Any = None,
    confidence: Optional[Dict[str, float]] = None,
    form: Optional[Dict[str, str]] = None,
    only_players: Optional[Iterable[str]] = None,
) -> MatchReport:
    """Translate a real :class:`~match_engine.MatchResult` into a
    :class:`~world.ledger.MatchReport`.

    ``player_stats`` is the exporter's ``stats`` dict. It is optional because a
    caller may not have run the exporter; without it the report still carries
    the result, cards and engine stamina, just no per-player line, and the
    ledger will record nothing for individuals.

    ``sub_controller`` is REQUIRED for fatigue/fitness/injuries to cross — the
    engine keeps post-match stamina there and never puts it on ``MatchResult``.
    Omit it and those fields simply stay absent, which the ledger reads as "the
    engine did not say".
    """
    config = getattr(result, "config", None)
    stats = player_stats or {}
    stamina = _stamina_lookup(result, sub_controller)
    keep = set(only_players) if only_players is not None else None

    when = match_date or getattr(config, "match_date", None) or date.today()

    lines: List[PlayerMatchLine] = []
    for name, stat in stats.items():
        if keep is not None and name not in keep:
            continue
        st = stamina.get(name)

        # engine values, passed straight through
        fatigue = fitness = None
        injury = injury_minute = None
        if st is not None:
            # a lossless unit mapping, not a re-derivation: the engine works in
            # 0-100 stamina, the ledger in 0-1 fatigue
            fatigue = round(1.0 - (float(st.current_stamina) / 100.0), 4)
            # the engine's own view of a player's condition going in
            fitness = round(float(st.starting_stamina) / 100.0, 4)
            if getattr(st, "is_injured", False):
                injury = st.injury_type
                injury_minute = st.injury_minute

        lines.append(PlayerMatchLine(
            player_id=resolve_player_id(name, adapter),
            club_id=stat.get("team", "") or home_id,
            minutes=_minutes(stat),
            goals=int(stat.get("goals", 0) or 0),
            assists=int(stat.get("assists", 0) or 0),
            shots=_shots(stat),
            xg=round(float(stat.get("xg", 0.0) or 0.0), 4),
            yellow_cards=int(stat.get("yellow_cards", 0) or 0),
            red_card=bool(stat.get("red_cards", 0) or 0),
            fatigue=fatigue,
            fitness=fitness,
            # confidence/form live in SeasonState, not in a MatchResult — if the
            # caller did not supply them the ledger keeps what it already had
            confidence=(confidence or {}).get(name),
            form=(form or {}).get(name),
            injury=injury,
            injury_minute=injury_minute,
        ))

    return MatchReport(
        competition_id=competition_id,
        season_id=season_id,
        match_date=when,
        home_id=home_id,
        away_id=away_id,
        home_goals=int(getattr(result, "home_goals", 0) or 0),
        away_goals=int(getattr(result, "away_goals", 0) or 0),
        round_name=round_name,
        stage_name=stage_name,
        is_neutral=is_neutral,
        players=lines,
    )


def apply_result(
    ledger: WorldLedger,
    result: Any,
    player_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Translate a real result and fold it into the world ledger.

    Returns the ledger's own record of the match so a caller can see exactly
    what crossed the boundary. Purely in-memory with respect to 26/27: the only
    file this can ever write is ``ledger.path``.
    """
    report = match_report_from_result(result, player_stats, **kwargs)
    touched = ledger.record_match(report)
    return {
        "competition_id": report.competition_id,
        "season_id": report.season_id,
        "match_date": report.match_date.isoformat(),
        "score": f"{report.home_goals}-{report.away_goals}",
        "players_recorded": len(touched),
        "players_with_lines": sum(1 for ln in report.players if ln.minutes > 0),
        "engine_values_carried": {
            "fatigue": sum(1 for ln in report.players if ln.fatigue is not None),
            "fitness": sum(1 for ln in report.players if ln.fitness is not None),
            "injuries": sum(1 for ln in report.players if ln.injury),
        },
    }


# ─────────────────────────────────────────────
# REVERSE DIRECTION (audit §10: production.py's job)
# ─────────────────────────────────────────────

def apply_competition_context(config: Any, competition: Any, *,
                              round_name: str = "", leg: Optional[str] = None,
                              aggregate: Optional[Sequence[int]] = None) -> Any:
    """Fold a competition's rules into a real ``MatchConfig``.

    The one hand-off from the world layer to the match engine. It returns a NEW
    config and never mutates the input, and it is deliberately tiny: the engine
    must consume a field only when it is set, or 26/27 regresses.
    """
    ctx = competition.context_for(round_name=round_name or None, leg=leg,
                                  aggregate=tuple(aggregate) if aggregate else None)
    return ctx.apply(config)
