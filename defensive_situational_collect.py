"""Defensive Action Situational Corpus Collector

Collects per-decision defensive action data with full situational context.
Outputs sequences grouped by (team, player) for BPTT training.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match_engine import MatchEngine, MatchState, MatchPhase, GameState, MatchConfig
from player_dna import SquadBuilder
from player_state import NeuralStateStore
from role_features import ROLE_FAMILY, ROLE_INTENTS, _role_family
from decision_brain import PlayerIntent
from event_chain import EventType
from situation_tokens import entity_tokens
from situational_context import assemble
from memory_features import memory_features
from player_state import NeuralStateStore
from datetime import date


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

class DefensiveSequence:
    """One possession-spanning sequence for a single player."""
    def __init__(self, team: str, player: str, position: str, family: str):
        self.team = team
        self.player = player
        self.position = position
        self.family = family
        self.steps: List[Dict[str, Any]] = []
        self._prev_h = None
        self.role_block = None

    def add_step(self, tokens, token_mask, state, role, memory, plan, h_prev,
                 action: str, sensors: np.ndarray, engine_snapshot: Dict):
        step = {
            "tokens": tokens.copy() if isinstance(tokens, np.ndarray) else tokens,
            "token_mask": token_mask.copy() if isinstance(token_mask, np.ndarray) else token_mask,
            "state": state.copy() if isinstance(state, np.ndarray) else state,
            "role": role.copy() if isinstance(role, np.ndarray) else role,
            "memory": memory.copy() if isinstance(memory, np.ndarray) else memory,
            "plan": plan.copy() if isinstance(plan, np.ndarray) else plan,
            "h_prev": h_prev.copy() if h_prev is not None else None,
            "action": action,
            "sensors": sensors.copy() if isinstance(sensors, np.ndarray) else sensors,
            "engine_snapshot": engine_snapshot,
        }
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        return {
            "team": self.team,
            "player": self.player,
            "position": self.position,
            "family": self.family,
            "steps": len(self.steps),
            "data": convert(self.steps),
        }


# ═════════════════════════════════════════════════════════════════════════════
# COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DefensiveSituationalCollector:
    def __init__(self, engine):
        self.engine = engine
        self.sequences: List[DefensiveSequence] = []
        self._current_seq: Dict[Tuple[str, str], DefensiveSequence] = {}
        self._matches_completed = 0

    def _get_player_position(self, player_name: str, team: str) -> Tuple[str, str]:
        for p in self.engine.squads[team].get("starters", []):
            if p.name == player_name:
                pos = getattr(p, "position", "CM")
                fam = ROLE_FAMILY.get(pos, pos)
                if fam == "DM":
                    fam = "CDM"
                return pos, fam
        return "CM", "CM"

    def _should_cut_sequence(self, ball_x: float, ball_y: float) -> bool:
        if ball_x <= 5.0 or ball_x >= 100.0:
            return True
        return False

    def _get_role_block(self, position: str) -> np.ndarray:
        fam = _role_family(position) or position
        vocab = ROLE_INTENTS.get(fam, frozenset())
        from decision_brain import PlayerIntent
        block = np.zeros(10, dtype=np.float64)
        for i, intent in enumerate(PlayerIntent):
            if intent.value in vocab:
                block[i] = 1.0
        return block

    def _get_entity_tokens(self, team: str, player_name: str, ball_x: float, ball_y: float):
        from situation_tokens import entity_tokens
        return entity_tokens(self.engine, team, player_name, ball_x, ball_y)

    def _build_sensors(self, team: str, player_name: str, position: str,
                       engine, ball_x: float, ball_y: float, danger_t: float):
        from brain_sensors import extract_sensors
        # Find player object
        player_obj = None
        for p in engine.squads[team].get("starters", []):
            if p.name == player_name:
                player_obj = p
                break
        if player_obj is None:
            for p in engine.squads[team].get("substitutes", []):
                if p.name == player_name:
                    player_obj = p
                    break
        
        sensors = extract_sensors(
            player=player_obj,
            x=ball_x,
            y=ball_y,
            teammates=[p for p in engine.squads[team].get("starters", []) if p.name != player_name],
            defenders=[],
            position_engine=engine.position_engine,
            under_pressure=danger_t > 0.5,
            attacks_right=True,  # Will be overridden by caller if needed
            game_state=None,
            minute=45.0,
        )
        return np.array(sensors, dtype=np.float64)

    def record_decision(self, defender_name: str, team: str, position: str,
                        action: str, sensors: np.ndarray,
                        ball_x: float, ball_y: float, danger_t: float,
                        ball_aerial: bool, ctx_x: float, ctx_y: float,
                        opponent_distance: float, press_occurred: bool,
                        minute: float, clock: float, attacks_right: bool):
        team_key = f"{team}|{defender_name}"

        if self._should_cut_sequence(ball_x, ball_y):
            if team_key in self._current_seq:
                self.sequences.append(self._current_seq[team_key])
                del self._current_seq[team_key]

        if team_key not in self._current_seq:
            pos, fam = self._get_player_position(defender_name, team)
            self._current_seq[f"{team}|{defender_name}"] = DefensiveSequence(team, defender_name, pos, fam)

        seq = self._current_seq[f"{team}|{defender_name}"]

        pstate = self.engine.position_engine.neural_states.player(defender_name)

        from situation_tokens import entity_tokens
        # Find player object for entity_tokens
        player_obj = None
        for p in self.engine.squads[team].get("starters", []):
            if p.name == defender_name:
                player_obj = p
                break
        if player_obj is None:
            for p in self.engine.squads[team].get("substitutes", []):
                if p.name == defender_name:
                    player_obj = p
                    break
        tokens, token_mask = entity_tokens(
            player=player_obj,
            x=ball_x,
            y=ball_y,
            teammates=[p for p in self.engine.squads[team].get("starters", []) if p.name != defender_name],
            defenders=[],
            position_engine=self.engine.position_engine,
            attacks_right=attacks_right,
        )

        memory_block = memory_features(defender_name, pstate)

        plan_block = self.engine.position_engine.neural_states.team(team).summary()

        role_block = self._get_role_block(position)

        shared_sensors = self._build_sensors(team, defender_name, position, self.engine, ball_x, ball_y, danger_t)

        from situational_context import assemble
        situation = assemble(
            state=sensors[:24] if len(sensors) >= 24 else np.zeros(24),
            role=seq.role_block if hasattr(seq, 'role_block') else np.zeros(9),
            memory=memory_block,
            plan=plan_block,
        )

        from situation_tokens import entity_tokens
        # Find player object for entity_tokens
        player_obj = None
        for p in self.engine.squads[team].get("starters", []):
            if p.name == defender_name:
                player_obj = p
                break
        if player_obj is None:
            for p in self.engine.squads[team].get("substitutes", []):
                if p.name == defender_name:
                    player_obj = p
                    break
        tokens, token_mask = entity_tokens(
            player=player_obj,
            x=ball_x,
            y=ball_y,
            teammates=[p for p in self.engine.squads[team].get("starters", []) if p.name != defender_name],
            defenders=[],
            position_engine=self.engine.position_engine,
            attacks_right=attacks_right,
        )

        h_prev = seq._prev_h if seq.steps else np.zeros(32, dtype=np.float64)

        engine_snapshot = {
            "minute": minute,
            "clock": clock,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "possession": getattr(self.engine.state, 'possession_winner', '?'),
        }

        seq.add_step(
            tokens=tokens, token_mask=token_mask,
            state=situation[:24], role=situation[24:33],
            memory=situation[33:41], plan=situation[41:47],
            h_prev=h_prev, action=action, sensors=sensors,
            engine_snapshot=engine_snapshot,
        )
        seq._prev_h = h_prev


# ═══════════════════════════════════════════════════════════════════════════════
# PATCHER FOR MATCH ENGINE
# ════════════════════════════════════════════════════════════════════════════════

_collector_instance = None

def install_defensive_collector(engine):
    global _collector_instance
    collector = DefensiveSituationalCollector(engine)
    _collector_instance = collector

    import match_engine
    import event_chain

    # Patch _def_action_choice to store the action type
    original_def_action = match_engine._def_action_choice
    def def_action_wrapper(engine, defending_team, attacking_team,
                         danger_level, ctx_x, ctx_y, ball_aerial,
                         opponent_distance, attacks_right, ball_x, ball_y,
                         press_occurred=False, fallback=None, site="contest"):
        action = original_def_action(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y, ball_aerial, opponent_distance,
            attacks_right, ball_x, ball_y, press_occurred, None, "contest"
        )
        # Store action for this team
        if not hasattr(engine, '_pending_def_action'):
            engine._pending_def_action = {}
        engine._pending_def_action[defending_team] = action
        return action
    match_engine._def_action_choice = def_action_wrapper

    # Patch DefensiveChain.generate to record decisions
    import event_chain
    original_cm = event_chain.DefensiveChain.__dict__['generate']
    original_func = original_cm.__func__

    def generate_wrapper(cls, minute, defending_team, attacking_team,
                         def_players, att_players, state, action_type="tackle",
                         context_x=None, context_y=None, attacks_right=True,
                         referee_strictness=0.5, danger_level=0.0, ball_aerial=False,
                         own_goal_x=105.0, position_engine=None, ball_z=None,
                         defender_facing_x=None, defender_facing_y=None,
                         opponent_distance=None, stamina=None):
        
        # Get action type from pending actions
        action_type = "tackle"
        engine = engine if hasattr(state, 'engine') else None
        if engine and hasattr(engine, '_pending_def_action'):
            action_type = cls._defensive_collector_engine._pending_def_action.get(defending_team, "tackle")
            if defending_team in cls._defensive_collector_engine._pending_def_action:
                del cls._defensive_collector_engine._pending_def_action[defending_team]
        
        # Call original (already bound classmethod)
        result = original_func(
            cls, minute, defending_team, attacking_team,
            def_players, att_players, state, action_type,
            context_x, context_y, attacks_right, referee_strictness,
            danger_level, ball_aerial, own_goal_x, position_engine,
            opponent_distance, stamina
        )

        # Record the decision if we have a collector
        global _collector_instance
        if _collector_instance is not None:
            from event_chain import EventType
            from decision_brain import PlayerIntent
            from brain_sensors import extract_sensors
            from situational_context import assemble
            from situation_tokens import entity_tokens
            from memory_features import memory_features

            # Extract defender from result events
            defender_name = None
            for event in result.events:
                if event.event_type in (EventType.TACKLE_WON, EventType.INTERCEPTION, EventType.CLEARANCE, EventType.BLOCK):
                    defender_name = event.player
                    break

            if defender_name:
                # Find the defender's profile for position
                defender_profile = None
                for p in def_players:
                    if p.name == defender_name:
                        defender_profile = p
                        break

                if defender_profile:
                    position = defender_profile.position
                    team = defending_team
                    player_name = defender_name
                else:
                    # Fallback
                    if def_players:
                        defender_profile = def_players[0]
                        position = defender_profile.position
                        team = defending_team
                        player_name = defender_profile.name
                    else:
                        return result

                # Find player object
                player_obj = None
                for p in def_players:
                    if p.name == player_name:
                        player_obj = p
                        break
                if player_obj is None:
                    for p in att_players:
                        if p.name == player_name:
                            player_obj = p
                            break

                # Build sensors
                sensors = extract_sensors(
                    player=player_obj,
                    x=context_x if context_x is not None else 0.0,
                    y=context_y if context_y is not None else 0.0,
                    teammates=def_players,
                    defenders=att_players,
                    position_engine=cls._defensive_collector_engine.position_engine,
                    under_pressure=danger_level > 0.5,
                    attacks_right=attacks_right,
                    game_state=state,
                    minute=minute,
                )

                from player_state import NeuralStateStore
                pstate = cls._defensive_collector_engine.position_engine.neural_states.player(defender_name)
                memory_block = memory_features(defender_name, pstate)
                plan_block = cls._defensive_collector_engine.position_engine.neural_states.team(defending_team).summary()
                role_block = _collector_instance._get_role_block(position)

                shared_sensors = extract_sensors(
                    player=player_obj,
                    x=context_x if context_x is not None else 0.0,
                    y=context_y if context_y is not None else 0.0,
                    teammates=def_players,
                    defenders=att_players,
                    position_engine=cls._defensive_collector_engine.position_engine,
                    under_pressure=danger_level > 0.5,
                    attacks_right=attacks_right,
                    game_state=state,
                    minute=minute,
                )

                from situational_context import assemble
                situation = assemble(
                    state=sensors[:24] if len(sensors) >= 24 else np.zeros(24),
                    role=role_block,
                    memory=memory_block,
                    plan=plan_block,
                )

                from situation_tokens import entity_tokens
                tokens, token_mask = entity_tokens(
                    player=player_obj,
                    x=context_x if context_x is not None else 0.0,
                    y=context_y if context_y is not None else 0.0,
                    teammates=def_players,
                    defenders=att_players,
                    position_engine=cls._defensive_collector_engine.position_engine,
                    attacks_right=attacks_right,
                )

                from player_state import NeuralStateStore
                pstate = cls._defensive_collector_engine.position_engine.neural_states.player(defender_name)
                h_prev = np.zeros(32, dtype=np.float64)

                engine_snapshot = {
                    "minute": minute,
                    "clock": state.match_clock_s if hasattr(state, 'match_clock_s') else 0.0,
                    "ball_x": context_x if context_x is not None else 0.0,
                    "ball_y": context_y if context_y is not None else 0.0,
                    "possession": getattr(state, 'possession_winner', '?'),
                }

                # Record the decision
                _collector_instance.record_decision(
                    defender_name=defender_name,
                    team=defending_team,
                    position=position,
                    action=action_type,
                    sensors=sensors,
                    ball_x=context_x if context_x is not None else 0.0,
                    ball_y=context_y if context_y is not None else 0.0,
                    danger_t=danger_level,
                    ball_aerial=ball_aerial,
                    ctx_x=context_x if context_x is not None else 0.0,
                    ctx_y=context_y if context_y is not None else 0.0,
                    opponent_distance=opponent_distance if opponent_distance is not None else 0.0,
                    press_occurred=False,
                    minute=minute,
                    clock=state.match_clock_s if hasattr(state, 'match_clock_s') else 0.0,
                    attacks_right=attacks_right
                )
            else:
                return result
                from situational_context import assemble
                situation = assemble(
                    state=sensors[:24] if len(sensors) >= 24 else np.zeros(24),
                    role=np.zeros(9),
                    memory=memory_block,
                    plan=cls._defensive_collector_engine.position_engine.neural_states.team(defending_team).summary(),
                )

                from situation_tokens import entity_tokens
                tokens, token_mask = entity_tokens(
                    player=player_obj,
                    x=context_x if context_x is not None else 0.0,
                    y=context_y if context_y is not None else 0.0,
                    teammates=def_players,
                    defenders=att_players,
                    position_engine=cls._defensive_collector_engine.position_engine,
                    attacks_right=attacks_right,
                )

                from player_state import NeuralStateStore
                pstate = cls._defensive_collector_engine.position_engine.neural_states.player(defender_name)
                h_prev = np.zeros(32, dtype=np.float64)  # Simplified

                engine_snapshot = {
                    "minute": minute,
                    "clock": state.match_clock_s if hasattr(state, 'match_clock_s') else 0.0,
                    "ball_x": context_x if context_x is not None else 0.0,
                    "ball_y": context_y if context_y is not None else 0.0,
                    "possession": getattr(state, 'possession_winner', '?'),
                }

                # Record the decision
                from defensive_situational_collect import _collector_instance
                if _collector_instance is not None:
                    _collector_instance.record_decision(
                        defender_name=defender_name,
                        team=defending_team,
                        position=position,
                        action=action_type,
                        sensors=np.array([0.0]*24),  # placeholder
                        ball_x=context_x if context_x is not None else 0.0,
                        ball_y=context_y if context_y is not None else 0.0,
                        danger_t=danger_level,
                        ball_aerial=ball_aerial,
                        ctx_x=context_x if context_x is not None else 0.0,
                        ctx_y=context_y if context_y is not None else 0.0,
                        opponent_distance=opponent_distance if opponent_distance is not None else 0.0,
                        press_occurred=False,
                        minute=minute,
                        clock=state.match_clock_s if hasattr(state, 'match_clock_s') else 0.0,
                        attacks_right=attacks_right
                    )

        return result

    original_cm = event_chain.DefensiveChain.__dict__['generate']
    original_func = original_cm.__func__
    event_chain.DefensiveChain._original_generate = original_cm
    event_chain.DefensiveChain._defensive_collector_engine = engine
    event_chain.DefensiveChain.generate = classmethod(generate_wrapper)

    return collector


def save_sequences(sequences: List[DefensiveSequence], path: str):
    data = {"sequences": [seq.to_dict() for seq in sequences]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(sequences)} sequences to {path}")


def run_collection(matches: int = 12, seed: int = 123, styles: str = "fluid_counter,attacking,tiki_taka",
                   out_path: str = "brains_def/situational_sequences.json"):
    import random
    import auto_run_match
    from datetime import date

    style_list = styles.split(",")

    for m in range(matches):
        match_seed = seed + m * 100
        random.seed(match_seed)
        np.random.seed(match_seed)

        loader = auto_run_match.get_loader()
        home_raw = auto_run_match.get_loader().build_matchday_squad("Oxton", availability=None)
        away_raw = auto_run_match.get_loader().build_matchday_squad("Natrican", availability=None)

        from player_dna import SquadBuilder
        home_squad = SquadBuilder.build(
            team_name="Oxton",
            starters=cast(list, home_raw["starters"]),
            substitutes=cast(list, home_raw["substitutes"]),
            team_superstars=cast(list, home_raw["superstars"]),
            set_piece_takers=cast(list, home_raw["sp_takers"]),
            soul_assignments=auto_run_match.SOUL_PLAYERS,
        )
        away_squad = SquadBuilder.build(
            team_name="Natrican",
            starters=cast(list, away_raw["starters"]),
            substitutes=cast(list, away_raw["substitutes"]),
            team_superstars=cast(list, away_raw["superstars"]),
            set_piece_takers=cast(list, away_raw["sp_takers"]),
            soul_assignments=auto_run_match.SOUL_PLAYERS,
        )

        from auto_run_match import _resolve_team_profile
        home_style = _resolve_team_profile("Oxton", home_raw["formation"], True, None, "fluid_counter")
        away_style = _resolve_team_profile("Natrican", away_raw["formation"], False, None, "attacking")

        from match_engine import MatchEngine, MatchState, MatchPhase, GameState, MatchConfig
        from datetime import date
        config = MatchConfig(
            home_team="Oxton",
            away_team="Natrican",
            match_date=date(2026, 9, 6),
            matchday=3,
            season="26/27",
        )
        state = MatchState()
        state.phase = MatchPhase.OPENING
        state.minute = 0
        state.match_clock_s = 0.0

        engine = MatchEngine(config, home_style, away_style)
        engine.state = state

        collector = install_defensive_collector(engine)

        engine.set_squad("Oxton", home_squad["starters"], home_squad["substitutes"])
        engine.set_squad("Natrican", away_squad["starters"], away_squad["substitutes"])

        result = engine.simulate()

        print(f"Match {m+1}/{matches} done.")
        # Save sequences from this match
        if collector.sequences:
            save_sequences(collector.sequences, out_path)

    print("Collection complete.")
    return collector


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--styles", type=str, default="fluid_counter,attacking,tiki_taka")
    parser.add_argument("--out", type=str, default="brains_def/situational_sequences.json")
    args = parser.parse_args()

    print(f"Collecting defensive situational corpus: {args.matches} matches")
    print("This will run real matches and collect per-decision data...")

    # Run the collection
    collector = run_collection(matches=args.matches, seed=args.seed, styles=args.styles, out_path=args.out)