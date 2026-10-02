"""
PLOFA 26/27 — MANAGER SAMPLE COLLECTION (Phase 7)
====================================================
manager_collection.py

Builds the training curriculum for ManagerBrain evolution by recording the
STATIC ManagerProfile's in-match decisions (flag OFF — the surrogate is a
curriculum, not the final signal).

How it works:
  - Install two opt-in hooks:
      * tactical_ai._COLLECTION_HOOK          → every decision moment
                                              (team, sensors, posture)
      * match_engine._collection_checkpoint_hook → every 5' even when no
        tactical decision fired (sensors + the posture currently held)
  - Run `n_matches` scratch matches (own engine, static ManagerPool
    managers, opposition style cycled through `away_styles`).
  - After EACH match, correlate every recorded decision on a side with
    that side's final outcome (points 3/1/0) AND xG delta:
        outcome_points = 0.6 * points_norm + 0.4 * xg_norm
        points_norm    = 1.0 / 0.5 / 0.0 for W/D/L
        xg_norm        = 0.5 + 0.5 * tanh(team_xg - opp_xg)   -> [0, 1]
  - Return rows [(sensors, posture, outcome_points, match_outcome)].

Hooks are ALWAYS restored to None afterwards, so normal engine behaviour
is untouched when collection is not active.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import match_engine
import tactical_ai
from manager_sensors import extract_manager_sensors

Row = Tuple[np.ndarray, str, float, str]


def _static_posture_to_class(posture: str) -> str:
    """Map the static manager's raw posture strings onto the brain's
    3-class vocabulary (DEFEND / BALANCED / ATTACK) so the surrogate
    rewards the same postures a ManagerBrain can actually choose."""
    p = posture.lower()
    if any(tok in p for tok in ("chase", "push", "tense")):
        return "ATTACK"
    if any(tok in p for tok in ("protect", "see_it_out")):
        return "DEFEND"
    return "BALANCED"


def _points_norm(points: int) -> float:
    return {3: 1.0, 1: 0.5, 0: 0.0}[points]


def _outcome_points(points: int, xg_delta: float) -> float:
    xg_norm = 0.5 + 0.5 * float(np.tanh(xg_delta))
    return max(0.0, min(1.0, 0.6 * _points_norm(points) + 0.4 * xg_norm))


def _build_engine(seed: int, away_style_spec):
    """Replicate pitch_replay.run_scratch_match's engine with an optional
    away TeamProfile override and an engine.quiet=True to silence live
    prints (they do not affect RNG — stdout only)."""
    import run_match as RM
    from match_engine import MatchEngine, MatchConfig, TeamProfile
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController
    from manager_profile import ManagerPool

    random.seed(seed)

    home_squad = SquadBuilder.build(
        team_name=RM.HOME_TEAM, starters=RM.HOME_STARTERS,
        substitutes=RM.HOME_SUBS, team_superstars=RM.HOME_SUPERSTARS,
        set_piece_takers=RM.HOME_SP_TAKERS)
    away_squad = SquadBuilder.build(
        team_name=RM.AWAY_TEAM, starters=RM.AWAY_STARTERS,
        substitutes=RM.AWAY_SUBS, team_superstars=RM.AWAY_SUPERSTARS,
        set_piece_takers=RM.AWAY_SP_TAKERS)

    all_players = (home_squad["starters"] + home_squad["substitutes"] +
                   away_squad["starters"] + away_squad["substitutes"])
    for player in all_players:
        if player.name in RM.SOUL_PLAYERS:
            player.dna.soul = RM.SOUL_PLAYERS[player.name]

    config = MatchConfig(
        home_team=RM.HOME_TEAM, away_team=RM.AWAY_TEAM,
        match_date=RM.MATCH_DATE, matchday=RM.MATCHDAY, season=RM.SEASON,
        competition=RM.COMPETITION, venue=RM.VENUE,
        stadium_capacity=RM.CAPACITY, referee=RM.REFEREE,
        referee_strictness=RM.STRICTNESS, is_derby=RM.IS_DERBY)

    if away_style_spec is None:
        away_profile = RM.AWAY_STYLE
    else:
        style, playing_style, intensity = away_style_spec
        away_profile = TeamProfile(
            name=RM.AWAY_TEAM, style=style, playing_style=playing_style,
            intensity=intensity)

    splash = RM.HOME_STYLE

    engine = MatchEngine(config, splash, away_profile)
    engine.quiet = True
    engine.set_squad(RM.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(RM.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(
        SubstitutionController(
            home_team=RM.HOME_TEAM, away_team=RM.AWAY_TEAM,
            home_subs_bench=home_squad["substitutes"],
            away_subs_bench=away_squad["substitutes"],
            home_style=RM.HOME_STYLE.style.value,
            away_style=away_profile.style.value,
            manager_stubbornness=RM.MANAGER_STUBBORNNESS))
    engine.sub_controller.MAX_SUBS = RM.MAX_SUBS
    pool = ManagerPool(
        clubs=[RM.HOME_TEAM, RM.AWAY_TEAM],
        style_lookup={RM.HOME_TEAM: RM.HOME_STYLE.style.value,
                      RM.AWAY_TEAM: away_profile.style.value})
    engine.set_managers(home_manager=pool.manager_for(RM.HOME_TEAM),
                        away_manager=pool.manager_for(RM.AWAY_TEAM))
    return engine


def collect_manager_samples(n_matches: int = 6, seed: int = 42,
                            away_styles: Optional[Sequence[Any]] = None) -> List[Row]:
    """Collect (sensors, posture, outcome_points, match_outcome) rows from
    STATIC-manager matches (flag OFF)."""
    if away_styles is None:
        away_styles = [None]

    collected: List[Row] = []

    match_engine.USE_MANAGER_BRAIN = False

    for match_idx in range(n_matches):
        match_seed = seed + match_idx
        spec = away_styles[match_idx % len(away_styles)]
        engine = _build_engine(match_seed, spec)

        # per-match recording state
        records: List[Tuple[str, int, np.ndarray, str]] = []
        last_posture: dict = {}
        last_record_minute: dict = {}

        def _decide_hook(team_name, state, posture, eng):
            if eng is None:
                return
            sensors = extract_manager_sensors(eng, team_name)
            posture = _static_posture_to_class(posture)
            records.append((team_name, int(state.minute), sensors, posture))
            last_posture[team_name] = posture
            last_record_minute[team_name] = int(state.minute)

        def _checkpoint_hook(eng, minute):
            for team in (eng.config.home_team, eng.config.away_team):
                if last_record_minute.get(team, -1) == minute:
                    continue
                post = last_posture.get(team)
                if post is None:
                    continue
                sensors = extract_manager_sensors(eng, team)
                records.append((team, int(minute), sensors, post))
                last_record_minute[team] = minute

        tactical_ai._COLLECTION_HOOK = _decide_hook
        match_engine._collection_checkpoint_hook = _checkpoint_hook
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = engine.simulate()
        finally:
            tactical_ai._COLLECTION_HOOK = None
            match_engine._collection_checkpoint_hook = None

        hg, ag = int(result.home_goals), int(result.away_goals)
        for team, minute, sensors, posture in records:
            is_home = (team == engine.config.home_team)
            if is_home:
                points = 3 if hg > ag else (1 if hg == ag else 0)
                xg_delta = float(result.home_xg) - float(result.away_xg)
            else:
                points = 3 if ag > hg else (1 if hg == ag else 0)
                xg_delta = float(result.away_xg) - float(result.home_xg)
            side = "H" if is_home else "A"
            collected.append((
                sensors,
                posture,
                _outcome_points(points, xg_delta),
                f"{side}:{hg}-{ag}",
            ))

    return collected