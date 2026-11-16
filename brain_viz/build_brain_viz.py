#!/usr/bin/env python
"""Build a self-contained HTML observatory for the shipped v5 situational brains.

WHAT THIS DOES
--------------
For every position brain in ``brains/<POS>.json`` (v5_situational, 47-d input,
attention over 16 entity tokens + a 32-unit GRU + memory block + plan block),
this script reconstructs a set of authored match situations and runs them
through the REAL production pipeline:

    perceive()            -> the 24-d shared state block (+ what is visible)
    entity_tokens()       -> the 16 x 12 attention token set (from the mind frame)
    role_tail()           -> the 9-d role block (real geometry, real names)
    memory_features()     -> the 8-d memory block (real mind, real events)
    plan_tail()           -> the 6-d plan block (real TeamPlan.summary())
    SituationalBrain.forward() -> alpha, GRU gates, h_new, intents, plan head

Then it captures the introspection tensors (attention, gates, neuron
activations, per-input ablations) and writes ONE offline HTML file that lets
the owner click through every brain, every situation, every neuron.

Only the SCENES and the memory events are authored. Every number downstream of
them is produced by the shipped code. Output:

    brain_viz/brain_thinking.html

Usage:
    python brain_viz/build_brain_viz.py
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import (                                     # noqa: E402
    PerceptionConfig, _is_visible, _mental_scale, _resolve_profile, perceive,
)
from situation_tokens import entity_tokens, N_MAX, TOKEN_D  # noqa: E402
from situational_brain import (                              # noqa: E402
    HIDDEN_D, INTENT_LABELS, SituationalBrain, _softmax, _sigmoid,
)
from situational_context import role_tail, plan_tail         # noqa: E402
from memory_features import memory_features                  # noqa: E402
from role_features import MENU_NAMES, _role_family           # noqa: E402
from player_state import TeamPlan, player_team               # noqa: E402
from brain_integration import (                              # noqa: E402
    _apply_coach_bias, _decision_temperature,
)
from coach_instructions import (                             # noqa: E402
    CoachInstructions, _ATTACK_BIAS, _DEFEND_BIAS, listener_scale,
)
from consequence_decision import (                           # noqa: E402
    ConsequenceEvaluator, _load_critic,
)
from cognition.memory import MemoryEvent                     # noqa: E402
from cognition.mind import PlayerMind, register_mind          # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS / METADATA
# ──────────────────────────────────────────────────────────────────────

PERCEPTION = PerceptionConfig(enabled=True, seed=7)

STATE_META: List[Tuple[str, str, str]] = [
    ("ball_x", "BALL X", "Carrier/ball x position, normalised by 105 m"),
    ("ball_y", "BALL Y", "Carrier/ball y position, normalised by 68 m"),
    ("goal_angle", "GOAL BEARING", "Signed bearing to the goal centre, /90 deg. 0 = dead ahead, +1 = goal to the right"),
    ("goal_mouth", "GOAL MOUTH", "Angle the two posts subtend, / the penalty-spot reference. 1.0 = a penalty-spot sight"),
    ("near_def", "NEAREST OPP", "Distance to the nearest outfield opponent, /15 m"),
    ("def_5", "OPP <=5m", "Opponents inside 5 m, count /5"),
    ("def_10", "OPP <=10m", "Opponents inside 10 m, count /5"),
    ("fwd_dist", "FWD OPTION", "Distance ahead of the best forward teammate, /40 m"),
    ("fwd_open", "FWD OPEN", "Openness of that forward teammate's lane, 0..1"),
    ("space", "SPACE", "Space ahead of the carrier, 0..1"),
    ("cross_zone", "CROSS ZONE", "Carrier is in a wide crossing corridor (wide roles only)"),
    ("near_goal", "NEAR GOAL", "Within 25 m of the opponent goal"),
    ("final_third", "FINAL THIRD", "Carrier is in the attacking third"),
    ("own_half", "OWN HALF", "Carrier is in his own half"),
    ("goal_dist", "GOAL DIST", "Distance to the goal centre, /70 m"),
    ("central", "CENTRAL", "Within 18 m of the centre line"),
    ("pressure", "PRESSURE", "Under pressure flag"),
    ("fatigue", "FATIGUE", "Match fatigue estimate, 0..1"),
    ("score_diff", "SCORE DIFF", "Goal difference /5, clamped -1..1"),
    ("minute", "MINUTE", "Match minute /90"),
    ("possession", "POSSESSION", "Team has possession"),
    ("vision", "VISION", "Player DNA vision /100"),
    ("composure", "COMPOSURE", "Player DNA composure /100"),
    ("decisions", "DECISIONS", "Player DNA decisions /100"),
]

TOKEN_META = [
    "forward_m", "lateral_m", "distance_m", "is_teammate", "is_opponent",
    "space_m", "ahead", "wide", "role_def", "role_mid", "role_fwd",
    "is_presser",
]

MEMORY_META = [
    ("duels", "DUELS", "recency-weighted duels won vs lost"),
    ("chances", "CHANCES", "recency-weighted big chances scored vs missed"),
    ("fouls", "FOULS", "how much trouble he is in"),
    ("emotion", "EMOTION", "total surviving emotional weight"),
    ("recency", "RECENCY", "how recently the last significant event was"),
    ("mem_n", "MEMORY N", "how much is remembered at all"),
    ("conf_bias", "CONF BIAS", "compressed confidence axis"),
    ("risk_bias", "RISK BIAS", "compressed risk axis"),
]

PLAN_META = [
    ("REGROUP / BUILD-UP", "Keep the ball, draw the press, build from the back"),
    ("MIDFIELD CIRCULATION", "Move the block, switch the point of attack"),
    ("WING ISOLATION", "Feed the wide 1v1 and attack the full-back"),
    ("BOX PENETRATION", "Break the last line and attack the box"),
    ("CONFIDENCE", "plan confidence"),
    ("HAS PLAN", "a plan has been adopted"),
]

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

#: Position -> (role family the brain's role block uses, carrier key in the base XI)
POS_INFO: Dict[str, Tuple[str, str]] = {
    "GK": ("GK", "GK"), "CB": ("CB", "CB1"),
    "LB": ("FB", "LB"), "RB": ("FB", "RB"),
    "CDM": ("DM", "CDM"), "CM": ("CM", "CM"), "CAM": ("AM", "CAM"),
    "LW": ("WING", "LW"), "RW": ("WING", "RW"),
    "ST": ("ST", "ST"), "CF": ("ST", "ST"),
}

MIRRORED = {"RB", "RW"}

#: Coach posture per (role family, chain id).  "BALANCED" is the no-bias path
#: (the real seam returns probs untouched when there is no instruction).
COACH_BY_CHAIN: Dict[Tuple[str, str], str] = {
    ("GK", "build"): "DEFEND", ("GK", "chase"): "ATTACK",
    ("CB", "step"): "ATTACK", ("CB", "counter"): "DEFEND",
    ("FB", "overlap"): "ATTACK", ("FB", "recover"): "DEFEND",
    ("DM", "half_turn"): "ATTACK", ("DM", "screen"): "DEFEND",
    ("CM", "third_man"): "ATTACK", ("CM", "legs"): "BALANCED",
    ("AM", "pocket"): "ATTACK", ("AM", "crowded"): "DEFEND",
    ("WING", "isolation"): "ATTACK", ("WING", "chase_wing"): "ATTACK",
    ("ST", "channel"): "ATTACK", ("ST", "aerial"): "ATTACK",
}

#: The shipped reasoning critic and its gate-winning blend.
CRITIC_PATH = ROOT / "brains_trainer" / "critic_v5.json"
CRITIC_BLEND = 0.3


def _coach_instructions(source: str) -> CoachInstructions | None:
    """Rebuild the exact instruction the manager layer would emit."""
    if source == "ATTACK":
        return CoachInstructions(intent_bias=dict(_ATTACK_BIAS),
                                 favored_flank="L", width_cmd=1.0,
                                 source="ATTACK")
    if source == "DEFEND":
        return CoachInstructions(intent_bias=dict(_DEFEND_BIAS),
                                 favored_flank=None, width_cmd=-1.0,
                                 source="DEFEND")
    return None


def _load_evaluator() -> ConsequenceEvaluator | None:
    try:
        return ConsequenceEvaluator(_load_critic(str(CRITIC_PATH)),
                                    blend=CRITIC_BLEND)
    except Exception as exc:                 # pragma: no cover - optional seam
        print(f"  [warn] critic unavailable: {exc}")
        return None


CRITIC = _load_evaluator()

#: DNA per role family — a plausible senior professional, not a superhero.
DNA: Dict[str, Dict[str, float]] = {
    "GK":  dict(vision=62, anticipation=60, composure=78, decisions=72, stamina=58),
    "CB":  dict(vision=58, anticipation=66, composure=74, decisions=70, stamina=70),
    "FB":  dict(vision=62, anticipation=64, composure=66, decisions=64, stamina=82),
    "DM":  dict(vision=68, anticipation=72, composure=72, decisions=76, stamina=78),
    "CM":  dict(vision=72, anticipation=74, composure=70, decisions=74, stamina=76),
    "AM":  dict(vision=78, anticipation=76, composure=62, decisions=72, stamina=72),
    "WING": dict(vision=74, anticipation=70, composure=58, decisions=66, stamina=84),
    "ST":  dict(vision=66, anticipation=72, composure=74, decisions=68, stamina=80),
}

NAMES = {
    "GK": "A. Mireles", "LB": "T. Okonkwo", "CB1": "J. Vandermeer",
    "CB2": "S. Almeida", "RB": "D. Whitlock", "CDM": "M. Okafor",
    "CM": "L. Bennett", "LW": "R. Sallow", "CAM": "N. Petrov",
    "RW": "K. Adeyemi", "ST": "V. Marchetti",
    "OGK": "P. Novak", "ORB": "E. Rossi", "OCB1": "G. Mensah",
    "OCB2": "H. Lindqvist", "OLB": "F. Duarte", "ORM": "C. Ibarra",
    "OCM1": "T. Sorenson", "OCM2": "A. Kovac", "OLM": "Y. Haddad",
    "OST1": "B. Ferreira", "OST2": "O. Diallo",
}

#: 4-2-3-1 in possession, attacking right.
BASE_MATES = {
    "GK": (8, 34),
    "LB": (24, 12), "CB1": (22, 26), "CB2": (22, 42), "RB": (24, 56),
    "CDM": (42, 32), "CM": (48, 40),
    "LW": (64, 10), "CAM": (60, 34), "RW": (64, 58),
    "ST": (74, 34),
}
#: 4-4-2 defending.
BASE_OPPS = {
    "OGK": (100, 34),
    "ORB": (86, 10), "OCB1": (84, 26), "OCB2": (84, 42), "OLB": (86, 58),
    "ORM": (78, 14), "OCM1": (76, 30), "OCM2": (76, 44), "OLM": (78, 54),
    "OST1": (62, 32), "OST2": (60, 44),
}

MEMORIES: Dict[str, List[Tuple[str, float, float]]] = {
    "clean": [],
    "missed": [("big_chance_missed", 2.0, 1.4), ("lost_duel", 9.0, 0.8)],
    "scored": [("big_chance_scored", 3.0, 1.3), ("clean_tackle_win", 7.0, 0.7)],
    "dirty": [("foul_in_box", 4.0, 1.2), ("lost_duel", 5.0, 0.9),
              ("big_chance_missed", 12.0, 0.7)],
    "battle": [("clean_tackle_win", 3.0, 1.0), ("lost_duel", 8.0, 0.8),
               ("clean_tackle_win", 15.0, 0.6)],
}


def S(xy, pressure, minute, score, plan, mem, label, note="", over=None):
    """One authored touch."""
    return {
        "xy": xy, "pressure": pressure, "minute": minute, "score": score,
        "plan": plan, "mem": mem, "label": label, "note": note,
        "over": over or {},
    }


# ──────────────────────────────────────────────────────────────────────
# AUTHORED SITUATIONS — two chains per role family, three touches each
# ──────────────────────────────────────────────────────────────────────

CHAINS: Dict[str, List[Dict[str, Any]]] = {
    "GK": [
        {"id": "build", "label": "Playing out from the back",
         "sub": "The keeper is the first playmaker: can he beat the first press?",
         "carrier": "GK", "steps": [
             S((8, 34), True, 9, 0, (0, 0.72), "clean",
               "First line, pressed",
               "Two strikers jump the keeper; the centre-backs are split wide.",
               {"OST1": (15, 32), "OST2": (17, 41), "OCM1": (26, 30), "ORM": (24, 16)}),
             S((16, 30), True, 10, 0, (0, 0.68), "clean",
               "Carried out, still pressed",
               "He has carried past the first line; the press follows.",
               {"OST1": (23, 29), "OST2": (25, 40), "OCM2": (32, 44)}),
             S((28, 36), False, 11, 0, (1, 0.61), "clean",
               "Released into midfield",
               "The press is broken; the team is circulating.",
               {"OST1": (38, 33), "OST2": (40, 44), "OCM1": (46, 30)}),
         ]},
        {"id": "chase", "label": "Chasing it late",
         "sub": "One goal down, 78th minute. The same geometry, a different mind.",
         "carrier": "GK", "steps": [
             S((10, 34), False, 78, -1, (0, 0.55), "battle",
               "Restart, trailing",
               "Tired legs, a negative memory block, a team that needs a goal.",
               {"OST1": (22, 33)}),
             S((20, 40), True, 79, -1, (1, 0.58), "battle",
               "Press arrives",
               "He steps inside and the striker closes him down.",
               {"OST1": (27, 37), "OCM1": (36, 44)}),
             S((34, 30), True, 80, -1, (3, 0.64), "battle",
               "Forced forward",
               "The plan has turned to box penetration; the safe option is gone.",
               {"OST1": (40, 28), "OCM1": (46, 34)}),
         ]},
    ],
    "CB": [
        {"id": "step", "label": "Stepping out of the first line",
         "sub": "The centre-back carries into midfield and must choose a lane.",
         "carrier": "CB1", "steps": [
             S((24, 26), True, 18, 0, (0, 0.70), "clean",
               "First line, pressed",
               "The striker presses his outside shoulder.",
               {"OST1": (31, 24), "OST2": (34, 38), "OCM1": (44, 30)}),
             S((38, 30), False, 19, 0, (1, 0.63), "clean",
               "Into the gap",
               "He has stepped past the striker; midfield is rotating.",
               {"OST1": (46, 31), "OCM2": (52, 40)}),
             S((54, 32), True, 20, 0, (1, 0.66), "clean",
               "Second line, pressed",
               "The next line engages; a forward pass is on if he sees it.",
               {"OST1": (60, 30), "OCM1": (64, 38), "OCM2": (66, 26)}),
         ]},
        {"id": "counter", "label": "Defending the counter",
         "sub": "Late, losing, and the ball is in his zone with runners beyond.",
         "carrier": "CB1", "steps": [
             S((30, 30), True, 76, -1, (0, 0.40), "dirty",
               "Turnover, immediate press",
               "He has just lost a duel and the ball is live beside him.",
               {"OST1": (37, 29), "OST2": (38, 38)}),
             S((44, 26), True, 77, -1, (1, 0.50), "dirty",
               "Carrying out of trouble",
               "Two forwards are ahead of the ball; the safe pass is backwards.",
               {"OST1": (52, 27), "OCM1": (56, 33)}),
             S((58, 22), True, 78, -1, (1, 0.55), "dirty",
               "Halfway, still pressed",
               "The team is committed forward; a turnover here is fatal.",
               {"OCM1": (64, 26), "ORM": (68, 16)}),
         ]},
    ],
    "FB": [
        {"id": "overlap", "label": "Overlap on the touchline",
         "sub": "The full-back owns the whole flank: underlap, overlap or cross?",
         "carrier": "LB", "steps": [
             S((30, 12), False, 22, 0, (0, 0.70), "clean",
               "Free on the touchline",
               "The winger is inside; the flank is his to run.",
               {"ORM": (38, 14), "OCM1": (46, 26)}),
             S((52, 8), True, 24, 0, (1, 0.64), "clean",
               "Pressed wide",
               "The wide midfielder engages; the overlap is arriving.",
               {"ORM": (59, 9), "OCM2": (62, 30)}),
             S((76, 12), True, 26, 0, (2, 0.70), "clean",
               "Final third, 1v1",
               "The right-back is isolated; the box is filling.",
               {"ORB": (84, 11), "OCB2": (86, 26), "OCM2": (80, 34)}),
         ]},
        {"id": "recover", "label": "Recovery run, legs gone",
         "sub": "74th minute, a goal down, and he is the one asked to go again.",
         "carrier": "LB", "steps": [
             S((36, 16), False, 74, -1, (1, 0.50), "battle",
               "Out of the block",
               "The team is stretched; he receives with grass ahead.",
               {"ORM": (44, 18)}),
             S((54, 10), True, 76, -1, (2, 0.60), "battle",
               "Carrying up the line",
               "The wide midfielder is on his shoulder.",
               {"ORM": (61, 11)}),
             S((70, 8), True, 77, -1, (2, 0.66), "battle",
               "Final third, tired",
               "The cross is on; the legs are not.",
               {"ORB": (78, 9), "OCB2": (84, 24)}),
         ]},
    ],
    "DM": [
        {"id": "half_turn", "label": "Half-turn release",
         "sub": "The holding midfielder receives with his back to the press.",
         "carrier": "CDM", "steps": [
             S((42, 32), True, 30, 0, (0, 0.68), "clean",
               "Back to goal, pressed",
               "The striker is on his back; both centre-backs are options.",
               {"OST1": (48, 34), "OST2": (50, 42), "OCM1": (56, 30)}),
             S((56, 30), False, 31, 0, (1, 0.63), "clean",
               "Turned out",
               "He has escaped the press and faces the field.",
               {"OST1": (62, 32), "OCM2": (66, 40)}),
             S((68, 34), True, 33, 0, (3, 0.60), "clean",
               "At the edge of the box",
               "The last line is set; the through-ball lane is tight.",
               {"OCM1": (72, 36), "OCB1": (78, 28), "OST1": (60, 36)}),
         ]},
        {"id": "screen", "label": "Screening a counter",
         "sub": "80th minute, a goal down: the most dangerous man is the one who must be safe.",
         "carrier": "CDM", "steps": [
             S((48, 34), False, 80, -1, (1, 0.50), "dirty",
               "In front of the back four",
               "The team is committed; he is the insurance policy.",
               {"OST1": (56, 35)}),
             S((58, 30), True, 81, -1, (1, 0.52), "dirty",
               "Press on the half-turn",
               "A forward jumps him from the blind side.",
               {"OST1": (64, 31), "OCM1": (68, 38)}),
             S((70, 36), True, 82, -1, (3, 0.60), "dirty",
               "Forced to progress",
               "The safe recycle is gone; the plan says attack the box.",
               {"OCM2": (74, 38), "OCB1": (80, 30)}),
         ]},
    ],
    "CM": [
        {"id": "third_man", "label": "Third-man progression",
         "sub": "The connector: receive, rotate, and find the man beyond the line.",
         "carrier": "CM", "steps": [
             S((48, 40), False, 35, 0, (0, 0.66), "clean",
               "In the pocket",
               "The ball arrives with space to turn.",
               {"OCM2": (56, 42), "OST2": (58, 44)}),
             S((60, 36), True, 37, 0, (1, 0.62), "clean",
               "Between the lines",
               "The opposition block is compact; a third-man run is starting.",
               {"OCM2": (66, 37), "OST1": (64, 32), "OCB1": (74, 30)}),
             S((74, 30), True, 39, 0, (3, 0.67), "clean",
               "At the last line",
               "He is the free man in the pocket; the box is loaded.",
               {"OCM1": (78, 32), "OCB1": (82, 28), "OCB2": (84, 42)}),
         ]},
        {"id": "legs", "label": "Legs have gone",
         "sub": "78 minutes in, one goal down, and the engine room is empty.",
         "carrier": "CM", "steps": [
             S((52, 40), True, 78, -1, (1, 0.45), "battle",
               "Receive under pressure",
               "The press is quicker than his legs.",
               {"OST2": (58, 41)}),
             S((64, 34), True, 80, -1, (3, 0.55), "battle",
               "Forced forward",
               "The team is chasing; safe possession no longer helps.",
               {"OCM2": (70, 35), "OCB1": (76, 30)}),
             S((78, 36), True, 82, -1, (3, 0.62), "battle",
               "On the edge of the box",
               "He arrives late into the area.",
               {"OCB1": (84, 34), "OCB2": (86, 46)}),
         ]},
    ],
    "AM": [
        {"id": "pocket", "label": "Between the lines",
         "sub": "The 10 receives in the pocket: shoot, slip the striker, or switch?",
         "carrier": "CAM", "steps": [
             S((58, 34), False, 40, 0, (1, 0.70), "clean",
               "Free in the pocket",
               "Both midfielders have jumped; the last line is 20 m away.",
               {"OCM1": (64, 32), "OCM2": (64, 40)}),
             S((70, 30), True, 42, 0, (3, 0.64), "clean",
               "Pocket, pressed",
               "The holding midfielder arrives late and from behind.",
               {"OCB1": (76, 28), "OCM1": (72, 36), "OCB2": (82, 40)}),
             S((82, 26), True, 44, 0, (3, 0.72), "clean",
               "Edge of the box",
               "The shooting angle opens; a centre-back steps out.",
               {"OCB1": (88, 26), "OCB2": (86, 40), "ORB": (90, 18)}),
         ]},
        {"id": "crowded", "label": "Crowded pocket",
         "sub": "79th minute, a missed big chance behind him, and no room to breathe.",
         "carrier": "CAM", "steps": [
             S((62, 34), True, 79, -1, (3, 0.50), "missed",
               "Receive in traffic",
               "Two midfielders close; the memory block is heavy.",
               {"OCM1": (67, 33), "OCM2": (67, 39)}),
             S((74, 32), True, 80, -1, (3, 0.58), "missed",
               "Tight pocket",
               "The last line has stepped up; the crowd is denser.",
               {"OCB1": (79, 30), "OCM2": (78, 40)}),
             S((84, 30), True, 81, -1, (3, 0.64), "missed",
               "Final window",
               "Half a yard to shoot, or one more pass.",
               {"OCB1": (89, 29), "OCB2": (88, 42)}),
         ]},
    ],
    "WING": [
        {"id": "isolation", "label": "1v1 isolation",
         "sub": "The winger is fed on the touchline and the full-back is alone.",
         "carrier": "LW", "steps": [
             S((64, 10), False, 28, 0, (1, 0.66), "clean",
               "Wide, free",
               "The full-back is 12 m away; the box is empty.",
               {"ORB": (72, 10), "ORM": (76, 14)}),
             S((76, 8), True, 30, 0, (2, 0.70), "clean",
               "1v1, engaged",
               "The full-back shows him inside; the overlap is behind.",
               {"ORB": (82, 8), "OCM2": (80, 22), "OCB2": (86, 40)}),
             S((90, 16), True, 32, 0, (3, 0.68), "clean",
               "Byline, angle closing",
               "The goal mouth is collapsing; the cut-back is on.",
               {"ORB": (96, 15), "OCB2": (90, 26), "OGK": (102, 34)}),
         ]},
        {"id": "chase_wing", "label": "Chasing it late",
         "sub": "76th minute, a goal down, and every touch has to be a threat.",
         "carrier": "LW", "steps": [
             S((66, 8), False, 76, -1, (2, 0.55), "missed",
               "Wide restart",
               "He is the out-ball; the memory of the miss is still there.",
               {"ORB": (74, 8)}),
             S((78, 10), True, 78, -1, (3, 0.60), "missed",
               "1v1 again",
               "The full-back has help inside this time.",
               {"ORB": (84, 10)}),
             S((92, 14), True, 79, -1, (3, 0.66), "missed",
               "Byline, tired",
               "The legs are heavy at the exact moment the angle disappears.",
               {"ORB": (98, 14), "OCB2": (92, 24)}),
         ]},
    ],
    "ST": [
        {"id": "channel", "label": "Channel to the box",
         "sub": "The striker drops, spins, and attacks the last line.",
         "carrier": "ST", "steps": [
             S((74, 34), False, 45, 0, (1, 0.60), "clean",
               "Between the centre-backs",
               "He has half a yard on the shoulder.",
               {"OCB1": (82, 30), "OCB2": (82, 40)}),
             S((86, 30), True, 46, 0, (3, 0.70), "clean",
               "Into the channel",
               "The near centre-back is beaten; the keeper is exposed.",
               {"OCB1": (90, 29), "OCB2": (90, 40), "OCM1": (82, 34)}),
             S((96, 34), True, 47, 0, (3, 0.76), "clean",
               "Six-yard box",
               "The angle is closing but the mouth is still there.",
               {"OGK": (102, 34), "OCB2": (94, 42), "OCB1": (94, 28)}),
         ]},
        {"id": "aerial", "label": "Aerial target, chasing",
         "sub": "80th minute, one goal down: the team goes long and he must hold it up.",
         "carrier": "ST", "steps": [
             S((70, 34), True, 80, -1, (1, 0.50), "missed",
               "Back to goal, pressed",
               "The centre-back is tight; the memory of the miss is fresh.",
               {"OCB1": (76, 32)}),
             S((84, 28), True, 81, -1, (3, 0.58), "missed",
               "Channel, isolated",
               "He is ahead of the ball with no support.",
               {"OCB1": (89, 27), "OCB2": (88, 40)}),
             S((95, 32), True, 82, -1, (3, 0.66), "missed",
               "Six-yard box, tired",
               "The last chance of the match.",
               {"OGK": (101, 34), "OCB2": (93, 42)}),
         ]},
    ],
}


# ──────────────────────────────────────────────────────────────────────
# PLUMBING
# ──────────────────────────────────────────────────────────────────────

class Engine:
    """Minimal position engine: name -> (x, y), plus the neural store."""

    def __init__(self, positions: Dict[str, Tuple[float, float]]):
        self.positions = dict(positions)
        self.neural_states: Any = None

    def get_position(self, name: str) -> Tuple[float, float]:
        return self.positions.get(name, (52.5, 34.0))

    def tracked_position(self, name: str):
        return self.positions.get(name)


class Store:
    def __init__(self, plan: TeamPlan):
        self._plan = plan

    def team(self, team: str) -> TeamPlan:
        return self._plan


class WrappedEngine(Engine):
    """Delegates positions to a (noisy) engine but exposes the neural store."""

    def __init__(self, base: Any, plan: TeamPlan):
        self._base = base
        self.neural_states = Store(plan)

    def get_position(self, name: str):
        return self._base.get_position(name)

    def tracked_position(self, name: str):
        try:
            return self._base.tracked_position(name)
        except Exception:
            return self._base.get_position(name)


def make_player(key: str, position: str, name: str | None = None) -> Any:
    fam = _role_family(position) or "CM"
    d = DNA.get(fam, DNA["CM"])
    return SimpleNamespace(
        name=name or NAMES.get(key, key),
        position=position,
        team_name="PLOFA XI",
        dna=SimpleNamespace(
            mental=SimpleNamespace(
                vision=d["vision"], anticipation=d["anticipation"],
                composure=d["composure"], decisions=d["decisions"],
            ),
            physical=SimpleNamespace(stamina=d["stamina"]),
            form=SimpleNamespace(fatigue_level=0.0),
        ),
    )


def mirror_xy(xy: Tuple[float, float]) -> Tuple[float, float]:
    return (float(xy[0]), float(68.0 - xy[1]))


def world_for(mirror: bool) -> Dict[str, Tuple[float, float]]:
    w: Dict[str, Tuple[float, float]] = {}
    for k, xy in {**BASE_MATES, **BASE_OPPS}.items():
        w[k] = mirror_xy(xy) if mirror else (float(xy[0]), float(xy[1]))
    return w


def tokens_with_names(player, x, y, mates, opps, engine):
    """Replicate entity_tokens' ordering so each row can be NAMED."""
    def _pos(p):
        n = getattr(p, "name", None)
        try:
            t = engine.tracked_position(n)
            if t is not None:
                return float(t[0]), float(t[1])
        except Exception:
            pass
        try:
            px, py = engine.get_position(n)
            return float(px), float(py)
        except Exception:
            return x, y

    def collect(players):
        rows = []
        for p in players or []:
            if p is player:
                continue
            px, py = _pos(p)
            rows.append((math.hypot(px - x, py - y), p, px, py))
        rows.sort(key=lambda r: r[0])
        return rows[: N_MAX // 2]

    ordered = collect(mates) + collect(opps)
    out = []
    presser = collect(opps)[0][1] if collect(opps) else None
    for dist, p, px, py in ordered:
        out.append({
            "name": getattr(p, "name", "?"),
            "role": getattr(p, "position", "?"),
            "team": "mate" if p in (mates or []) else "opp",
            "pos": [round(px, 2), round(py, 2)],
            "dist": round(float(dist), 2),
            "presser": p is presser,
        })
    return out


def softmax_rows(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / e.sum()


def sample_with_temperature(probs: np.ndarray, temp: float, seed: int) -> int:
    """Mirror brain_integration._sample_from_probs, with a local RNG."""
    p = np.asarray(probs, dtype=np.float64)
    if temp <= 1e-6:
        return int(np.argmax(p))
    t = np.exp(np.log(p + 1e-12) / temp)
    t = t / t.sum()
    r = np.random.default_rng(seed).random()
    cum = 0.0
    for i, v in enumerate(t):
        cum += v
        if r <= cum:
            return int(i)
    return int(np.argmax(t))


def r4(x: float) -> float:
    return round(float(x), 4)


# ──────────────────────────────────────────────────────────────────────
# THE REST OF THE MECHANISM STACK — each block calls the shipped code
# ──────────────────────────────────────────────────────────────────────

def perception_block(carrier: Any, position: str, x: float, y: float,
                     positions: Dict[str, Tuple[float, float]],
                     vis_names: set, scene: Any,
                     mind_names: Optional[set] = None) -> Dict[str, Any]:
    """The two gates, in the order production applies them.

    ``mind_names`` is what the mind's frame let through (25 m, 2 + 2, no
    cone); ``vis_names`` is what the cone then kept of that frame.  They are
    different worlds and the reason an actor is withheld depends on which
    gate dropped him, so each hidden actor carries the gate that cut him.
    """
    prof = _resolve_profile(position, PERCEPTION)
    power, acc = _mental_scale(carrier)
    radius = (PERCEPTION.range_min +
              (PERCEPTION.range_max - PERCEPTION.range_min) * power) \
        * prof["range_scale"]
    fov_rad = math.radians(prof["fov_deg"])
    sigma = PERCEPTION.noise_base * (1.0 - acc) * prof["noise_scale"]
    cap = prof.get("max_actors")
    frame = set(mind_names) if mind_names is not None else None

    hidden: List[Dict[str, Any]] = []
    visible = 0        # inside the cone AND inside the mind's frame
    cone_all = 0       # inside the cone, frame ignored
    frame_n = 0        # inside the mind's frame, cone ignored
    total = 0
    for key, (px, py) in positions.items():
        nm = NAMES.get(key, key)
        if nm == getattr(carrier, "name", None):
            continue
        total += 1
        dist = math.hypot(px - x, py - y)
        in_range = dist <= radius
        in_cone = _is_visible(x, y, px, py, True, radius, fov_rad)
        in_frame = frame is None or nm in frame
        if in_cone:
            cone_all += 1
        if in_frame:
            frame_n += 1
            if in_cone:
                visible += 1
        if nm not in vis_names:
            if not in_frame and not in_cone:
                why = "outside his frame and behind the cone"
            elif not in_frame:
                why = "in the cone, but outside the mind's 25 m / 2 + 2 frame"
            elif not in_range:
                why = "out of range"
            elif not in_cone:
                why = "behind / outside the cone"
            else:
                why = "cut by the relevance top-k cap"
            hidden.append({"name": nm, "dist": r4(dist), "why": why})

    capped = cap is not None and visible > cap
    return {
        "fov": r4(prof["fov_deg"]),
        "radius": r4(radius),
        "sigma": r4(sigma),
        "power": r4(power),
        "accuracy": r4(acc),
        "cap": cap,
        "seen": len(vis_names),
        "total": total,
        "visibleBeforeCap": visible,
        "capped": bool(capped),
        "hidden": sorted(hidden, key=lambda h: h["dist"]),
        "ballExact": True,
        "rangeScale": r4(prof["range_scale"]),
        "noiseScale": r4(prof["noise_scale"]),
        # ── the mind gate that runs BEFORE the cone (2026-11-16) ──
        "mindSeen": frame_n,
        "mindRadius": 25.0,
        "mindCapT": 2,
        "mindCapO": 2,
        "coneAll": cone_all,
        "tokenSlots": int(N_MAX),
    }


def memory_block(mind: PlayerMind, mem: np.ndarray) -> Dict[str, Any]:
    """Episodes, the weight the brain actually reads, and the temperament."""
    from memory_features import RECENCY_SCALE, _recency_weight
    mind.experience.apply_memory(mind.memory)
    b = mind.experience.bias
    events = [{
        "tag": e.tag,
        "age": r4(e.age),
        "severity": r4(e.severity),
        # memory_features() does NOT use the episodic exp decay: it reads a
        # linear recency weight 1/(1+age/40) and multiplies the stored
        # severity by it.  Show the product that actually enters the block.
        "recency": r4(_recency_weight(e.age)),
        "weight": r4(_recency_weight(e.age) * e.severity),
    } for e in mind.memory.events]
    return {
        "events": events,
        "confidence": r4(b["confidence"]),
        "risk": r4(b["risk_in_box"]),
        "aggression": r4(b["tackle_aggression"]),
        "features": [r4(v) for v in mem],
        "recencyScale": r4(RECENCY_SCALE),
        "decay": mind.memory.DECAY_PER_SECOND,
        "forgetBelow": mind.memory.FORGET_BELOW,
    }


def coach_block(carrier: Any, source: str, probs: np.ndarray):
    """The manager's on-ball say, applied by the shipped bias function."""
    ci = _coach_instructions(source)
    listener = listener_scale(carrier)
    bias = []
    if ci is not None:
        for label, bonus in ci.intent_bias.items():
            bias.append({"intent": label, "logit": r4(bonus),
                         "applied": r4(bonus * listener)})
    post = _apply_coach_bias(probs, carrier, ci.intent_bias if ci else None)
    post = np.asarray(post, dtype=np.float64)
    delta = (post - probs) * 100.0
    mover = int(np.argmax(np.abs(delta)))
    return {
        "source": source,
        "listener": r4(listener),
        "bias": bias,
        "flank": ci.favored_flank if ci else None,
        "width": r4(ci.width_cmd) if ci else 0.0,
        "probs": [r4(v) for v in post],
        "delta": [r4(v) for v in delta],
        "mover": mover,
        "active": bool(bias),
    }, post


def consequence_block(probs: np.ndarray, base_vec: np.ndarray,
                      position: str):
    """The TD critic's payoff correction, from the shipped evaluator."""
    if CRITIC is None:
        return None, probs
    try:
        tr = CRITIC.payoff_traces(probs, base_vec, position)
        raw = np.asarray(CRITIC.critic.predict(
            np.asarray(base_vec, dtype=np.float64), position), dtype=np.float64)
    except Exception:
        return None, probs
    q = np.array([tr["q"][l] for l in INTENT_LABELS], dtype=np.float64)
    adv = np.array([tr["advantage"][l] for l in INTENT_LABELS],
                   dtype=np.float64)
    post = np.array([tr["corrected"][l] for l in INTENT_LABELS],
                    dtype=np.float64)
    delta = (post - probs) * 100.0
    lo, hi = CRITIC._REWARD_LO, CRITIC._REWARD_HI
    # A TD critic trained on a narrow reward band can extrapolate wildly on a
    # state it has never seen; the band clip then flattens every intent to the
    # same value and the advantage collapses.  Report it instead of hiding it.
    at_hi = int(np.sum(raw >= hi - 1e-9))
    at_lo = int(np.sum(raw <= lo + 1e-9))
    return {
        "blend": r4(CRITIC.blend),
        "critic": "brains_trainer/critic_v5.json",
        "q": [r4(v) for v in q],
        "qRaw": [r4(v) for v in raw],
        "band": [r4(lo), r4(hi)],
        "atHi": at_hi,
        "atLo": at_lo,
        "saturated": bool(at_hi >= 5),
        "advantage": [r4(v) for v in adv],
        "probs": [r4(v) for v in post],
        "delta": [r4(v) for v in delta],
        "mover": int(np.argmax(np.abs(delta))),
    }, post


def sampling_block(probs: np.ndarray, temperature: float,
                   seed: int) -> Dict[str, Any]:
    """Sharpened distribution + the actual uniform draw that settled it."""
    p = np.asarray(probs, dtype=np.float64)
    if temperature <= 1e-6:
        sharp = p.copy()
        draw = None
    else:
        t = np.exp(np.log(p + 1e-12) / temperature)
        sharp = t / t.sum()
        draw = float(np.random.default_rng(seed).random())
    idx = sample_with_temperature(p, temperature, seed)

    def _entropy(v):
        w = v[v > 0]
        return float(-(w * np.log2(w)).sum()) if w.size else 0.0

    return {
        "temperature": r4(temperature),
        "sharpened": [r4(v) for v in sharp],
        "draw": r4(draw) if draw is not None else None,
        "sampled": int(idx),
        "pSampled": r4(sharp[idx]),
        "entropyBefore": r4(_entropy(p)),
        "entropyAfter": r4(_entropy(sharp)),
        "deterministic": bool(temperature <= 1e-6),
    }


def temper_block(mind: PlayerMind, probs: np.ndarray, pressure: bool,
                 fatigue: float) -> Dict[str, Any]:
    """The cognition route's temperament reshape (counterfactual here)."""
    tempered = np.asarray(mind.temper(probs, INTENT_LABELS, pressure, fatigue),
                          dtype=np.float64)
    delta = (tempered - probs) * 100.0
    return {
        "probs": [r4(v) for v in tempered],
        "delta": [r4(v) for v in delta],
        "mover": int(np.argmax(np.abs(delta))),
    }


def build_step(brain: SituationalBrain, position: str, family: str,
               carrier_key: str, chain: Dict[str, Any], spec: Dict[str, Any],
               h_prev: np.ndarray, mirror: bool, idx: int,
               coach_source: str = "BALANCED",
               plan_tracker: TeamPlan | None = None,
               prev_choice: str | None = None,
               ) -> Tuple[Dict[str, Any], np.ndarray]:
    # ── world ────────────────────────────────────────────────────────
    positions = world_for(mirror)
    positions[carrier_key] = mirror_xy(spec["xy"]) if mirror else spec["xy"]
    for k, xy in spec["over"].items():
        positions[k] = mirror_xy(xy) if mirror else xy
    # The engine is looked up by PLAYER NAME in production (PositionEngine is
    # name-keyed), so translate the keyed world here — otherwise every actor
    # collapses onto the default (52.5, 34) and perception sees an empty pitch.
    engine = Engine({NAMES.get(k, k): xy for k, xy in positions.items()})

    carrier = make_player(carrier_key, position)
    x, y = positions[carrier_key]
    mates = [make_player(k, k) for k in BASE_MATES if k != carrier_key]
    opps = [make_player(k, k) for k in BASE_OPPS]

    # ── memory (real PlayerMind + real memory_features) ─────────────
    mind = PlayerMind()
    for tag, age, sev in MEMORIES[spec["mem"]]:
        mind.memory.store(tag, severity=sev)
        mind.memory.events[-1].age = float(age)
    # REGISTERED — exactly what auto_run_match.py does for every player
    # before kickoff (PLOFA_COGNITION defaults on, and a fresh PlayerMind
    # is engaged).  Without this the token path takes the no-mind branch
    # that _maybe_situational only reaches with cognition switched off.
    register_mind(carrier.name, mind)

    # ── the mind's own frame (THIS is what attention is built from) ──
    # VisionSystem.filter_frame: 25 m, nearest 2 mates + 2 opponents, GKs
    # skipped, NO cone.  Production feeds these lists to entity_tokens, so
    # the attention set is at most four men — the 16 slots are capacity.
    frame_mates, frame_opps = mind.perceive(
        carrier, x, y, mates, opps, engine)
    frame_names = {getattr(p, "name", "") for p in frame_mates} | \
                  {getattr(p, "name", "") for p in frame_opps}
    mem = np.asarray(memory_features(carrier.name, mind), dtype=np.float64)
    memdeep = memory_block(mind, mem)

    # ── plan (real TeamPlan + real plan_tail) ───────────────────────
    # The chain's TeamPlan is a LIVE object across the three touches, exactly
    # as in a match: the block the brain reads on touch N is the plan the
    # team adopted after touch N-1.  The authored plan initialises it.
    if plan_tracker is not None:
        team_plan = plan_tracker
    else:
        plan_id0, plan_conf0 = spec["plan"]
        team_plan = TeamPlan(team="PLOFA XI", plan_id=int(plan_id0),
                             confidence=float(plan_conf0), possession_id=1)
    viz_engine = WrappedEngine(engine, team_plan)

    # ── perception → 24-d state + visible scene ─────────────────────
    base_vec, scene = perceive(
        carrier, x, y, frame_mates, frame_opps, engine,
        spec["pressure"], True, None,
        minute=spec["minute"], team_possession=True,
        score_diff=spec["score"], config=PERCEPTION, return_scene=True,
    )
    vis_mates, vis_opps = scene["teammates"], scene["defenders"]
    seen_engine = WrappedEngine(scene["position_engine"], team_plan)
    vis_names = {getattr(p, "name", "") for p in vis_mates} | \
                {getattr(p, "name", "") for p in vis_opps}
    percep = perception_block(carrier, position, x, y, positions, vis_names,
                              scene, mind_names=frame_names)

    # ── tokens / role / plan (the real extractors) ──────────────────
    # PRODUCTION FIDELITY (brain_integration._maybe_situational, in order):
    #   1. mind.perceive()  → the 25 m / 2 + 2 frame        (no cone)
    #   2. entity_tokens()  → the attention set, from THAT frame
    #   3. perceive()       → the cone, which gates the 24-d state + role
    # So the tokens and the state are read from different worlds: a token
    # can be a man the cone threw out.  Steps 1-3 are the shipped order,
    # not a reconstruction.
    tokens, mask = entity_tokens(carrier, x, y, frame_mates, frame_opps,
                                 engine, True)
    role_full = np.asarray(
        role_tail(carrier, x, y, vis_mates, vis_opps, seen_engine, True),
        dtype=np.float64).reshape(-1)
    # The brain's own width decides how much of the role block it eats: the
    # shipped v5 files are 47-d, i.e. 24 + 9 + 8 + 6, so the 10 extra role
    # slots role_tail() now emits are computed and then ignored.
    role_width = int(brain.w_q.shape[0]) - 24 - 8 - 6
    if role_full.size < role_width:
        role = np.pad(role_full, (0, role_width - role_full.size))
    else:
        role = role_full[:role_width]
    plan = plan_tail(seen_engine, "PLOFA XI")
    fed_plan = int(team_plan.plan_id)
    fed_conf = float(team_plan.confidence)

    # ── forward ─────────────────────────────────────────────────────
    logits, plan_logits, h_new, cache = brain.forward(
        tokens, mask, base_vec, role, mem, plan, h_prev)
    probs = _softmax(logits)
    plan_probs = _softmax(plan_logits)
    k = int(np.argmax(probs))
    base_p = float(probs[k])

    # ── the seams between the net and the pitch ──────────────────────
    coach, after_coach = coach_block(carrier, coach_source, probs)
    cons, after_cons = consequence_block(after_coach, base_vec, position)
    fatigue_now = float(base_vec[17])
    temper = temper_block(mind, after_cons, bool(spec["pressure"]),
                          fatigue_now)

    # ── ablations: block-level and per-input ────────────────────────
    def fwd(state=None, role_=None, mem_=None, plan_=None, mask_=None):
        lg, _, _, _ = brain.forward(
            tokens, mask if mask_ is None else mask_,
            base_vec if state is None else state,
            role if role_ is None else role_,
            mem if mem_ is None else mem_,
            plan if plan_ is None else plan_,
            h_prev)
        return _softmax(lg)

    abl = {}
    abl["state"] = r4((fwd(state=np.zeros(24))[k] - base_p) * 100.0)
    abl["role"] = r4((fwd(role_=np.zeros_like(role))[k] - base_p) * 100.0)
    abl["memory"] = r4((fwd(mem_=np.zeros_like(mem))[k] - base_p) * 100.0)
    abl["plan"] = r4((fwd(plan_=np.zeros_like(plan))[k] - base_p) * 100.0)
    abl["attention"] = r4((fwd(mask_=np.zeros_like(mask))[k] - base_p) * 100.0)

    u_labels: List[str] = [m[1] for m in STATE_META]
    fam_names = MENU_NAMES[family]
    role_names = [fam_names[i] if i < len(fam_names) else "(unused pad)"
                  for i in range(9)]
    u_labels += [f"role:{n}" for n in role_names]
    u_labels += [f"mem:{m[0]}" for m in MEMORY_META]
    u_labels += [f"plan:{i}" for i in range(6)]

    sensors = []
    for i in range(24):
        z = base_vec.copy(); z[i] = 0.0
        sensors.append({"u": i, "block": "state", "label": STATE_META[i][1],
                        "dp": r4((fwd(state=z)[k] - base_p) * 100.0)})
    for i in range(9):
        z = np.asarray(role).copy(); z[i] = 0.0
        sensors.append({"u": 24 + i, "block": "role", "label": role_names[i],
                        "dp": r4((fwd(role_=z)[k] - base_p) * 100.0)})
    for i in range(8):
        z = mem.copy(); z[i] = 0.0
        sensors.append({"u": 33 + i, "block": "memory", "label": MEMORY_META[i][1],
                        "dp": r4((fwd(mem_=z)[k] - base_p) * 100.0)})
    for i in range(6):
        z = plan.copy(); z[i] = 0.0
        sensors.append({"u": 41 + i, "block": "plan", "label": PLAN_META[i][0],
                        "dp": r4((fwd(plan_=z)[k] - base_p) * 100.0)})

    # ── attention context attribution (which token feeds each ctx dim) ──
    alpha = np.asarray(cache["alpha"], dtype=np.float64)
    e = np.asarray(cache["e"], dtype=np.float64)
    v = np.asarray(cache["v"], dtype=np.float64)
    token_meta = tokens_with_names(carrier, x, y, frame_mates, frame_opps,
                                   engine)
    ctx_top = []
    for d in range(32):
        contrib = alpha * v[:, d]
        if np.any(np.isfinite(contrib)):
            j = int(np.argmax(np.abs(contrib)))
            nm = token_meta[j]["name"] if j < len(token_meta) else "-"
        else:
            nm = "-"
        ctx_top.append(nm)
    u = np.asarray(cache["u"], dtype=np.float64)
    w_n = brain.w["w_n"]; w_z = brain.w["w_z"]; w_r = brain.w["w_r"]

    units = []
    for i in range(HIDDEN_D):
        contrib_by_gate = {
            "n": u * w_n[:, i],
            "z": u * w_z[:, i],
            "r": u * w_r[:, i],
        }
        # pick, per input dim, the gate it drives hardest
        best = np.zeros(79)
        gate = np.full(79, "n", dtype=object)
        for g, arr in contrib_by_gate.items():
            mask_better = np.abs(arr) > np.abs(best)
            best[mask_better] = arr[mask_better]
            gate[mask_better] = g
        order = np.argsort(-np.abs(best))[:4]
        drivers = []
        for j in order:
            j = int(j)
            if j < len(u_labels):
                lab = u_labels[j]
            else:
                lab = f"ctx#{j - 47} -> {ctx_top[j - 47]}"
            drivers.append({
                "label": lab, "gate": str(gate[j]),
                "value": r4(u[j]), "w": r4(best[j]),
            })
        units.append({
            "i": i,
            "h": r4(cache["h_new"][i]),
            "z": r4(cache["z"][i]),
            "r": r4(cache["r"][i]),
            "n": r4(cache["n"][i]),
            "prev": r4(cache["h"][i]),
            "contrib": [r4(cache["h_new"][i] * brain.w["w_o"][i, kk])
                        for kk in range(10)],
            "drivers": drivers,
        })

    # ── facts + narrative ───────────────────────────────────────────
    nearest = float(base_vec[4])
    def5 = int(round(float(base_vec[5]) * 5.0))
    def10 = int(round(float(base_vec[6]) * 5.0))
    goal_deg = float(base_vec[2]) * 90.0
    mouth_deg = float(base_vec[3]) * 36.81
    goal_m = float(base_vec[14]) * 70.0
    fwd_m = float(base_vec[7]) * 40.0
    fatigue = float(base_vec[17])

    vis_names = {getattr(p, "name", "") for p in vis_mates} | \
                {getattr(p, "name", "") for p in vis_opps}
    token_rank = {t["name"]: n for n, t in enumerate(token_meta)}
    actors = []
    for key, (px, py) in positions.items():
        if key == carrier_key:
            continue
        nm = NAMES.get(key, key)
        is_mate = key in BASE_MATES
        perceived = None
        if nm in vis_names:
            try:
                sx, sy = scene["position_engine"].get_position(nm)
                perceived = [r4(sx), r4(sy)]
            except Exception:
                perceived = None
        actors.append({
            "name": nm, "role": key, "team": "mate" if is_mate else "opp",
            "pos": [r4(px), r4(py)], "seenPos": perceived,
            "seen": nm in vis_names,
            "inFrame": nm in frame_names,
            "rank": token_rank.get(nm),
        })
    actors.sort(key=lambda a: (a["rank"] is None, a["rank"] or 0))

    # plan head proposal / adoption status
    pid = int(np.argmax(plan_probs))
    pconf = float(plan_probs[pid])
    tracker = plan_tracker
    adopted = False
    hits_before = int(tracker._candidate_hits) if tracker else 0
    cand_before = int(tracker._candidate) if tracker else -1
    conf_before = float(tracker._candidate_conf) if tracker else 0.0
    if tracker is not None:
        adopted = bool(tracker.propose(pid, pconf))
    cand_after = int(tracker._candidate) if tracker else -1
    hits_after = int(tracker._candidate_hits) if tracker else 0
    conf_after = float(tracker._candidate_conf) if tracker else 0.0
    # The gates describe the candidate that is on the table AFTER this touch.
    # propose() resets the candidate when it adopts, so an adoption is read
    # from its own fields; a fresh candidate always starts at one hit.
    if adopted:
        shown_hits = hits_before + 1
        shown_conf = max(conf_before, pconf) if cand_before == pid else pconf
    elif cand_after == pid:
        shown_hits = hits_after
        shown_conf = conf_after
    else:
        shown_hits = 0
        shown_conf = 0.0
    if adopted:
        plan_status = (f"{PLAN_META[pid][0]} adopted on this touch — the team "
                       f"plan switches (switch #{tracker.switches})")
    elif pid == fed_plan:
        plan_status = f"the head agrees with {PLAN_META[pid][0]}"
    elif pconf < 0.60:
        plan_status = (f"proposal {PLAN_META[pid][0]} at {pconf:.0%} sits "
                       f"below the 0.60 adoption threshold")
    elif shown_hits >= 2 and shown_conf < 0.60:
        plan_status = (f"two proposals of {PLAN_META[pid][0]}, but its best "
                       f"confidence {shown_conf:.0%} is under 0.60 — not adopted")
    else:
        plan_status = (f"proposal {PLAN_META[pid][0]} at {pconf:.0%} — needs "
                       f"two consecutive touches at >= 0.60 to switch")
    plan_state = {
        "proposed": pid,
        "prob": r4(pconf),
        "fedPlan": fed_plan,
        "fedConf": r4(fed_conf),
        "candidate": cand_before,
        "candidateAfter": cand_after,
        "hits": shown_hits,
        "hitsBefore": hits_before,
        "hitsAfter": hits_after,
        "candidateConf": r4(shown_conf),
        "needed": 2,
        "threshold": 0.6,
        "adopted": adopted,
        "adoptedPlan": int(tracker.plan_id) if tracker else -1,
        "switches": int(tracker.switches) if tracker else 0,
        "sticky": plan_tracker is not None,
        "statusText": plan_status,
    }

    temp = float(_decision_temperature(carrier, spec["pressure"], fatigue))
    seed = zlib.crc32(f"{position}:{chain['id']}:{idx}".encode()) & 0xFFFFFFFF
    samp = sampling_block(after_cons, temp, seed)
    sampled = int(samp["sampled"])

    att_sorted = sorted(
        [{"name": t["name"], "alpha": float(alpha[j])} for j, t in enumerate(token_meta)],
        key=lambda z: -z["alpha"])
    top_att = [a for a in att_sorted if a["alpha"] > 1e-6]
    att_entropy = 0.0
    for a in top_att:
        if a["alpha"] > 0:
            att_entropy -= a["alpha"] * math.log2(a["alpha"])
    att_norm = att_entropy / math.log2(len(top_att)) if len(top_att) > 1 else 0.0
    top3_share = sum(a["alpha"] for a in top_att[:3])

    active = sum(1 for un in units if abs(un["h"]) >= 0.05)
    loudest = max(units, key=lambda un: abs(un["h"]))
    mean_z = float(np.mean(cache["z"]))
    mean_r = float(np.mean(cache["r"]))
    second = int(np.argsort(-probs)[1])
    margin = float(probs[k] - probs[second])

    h_in = np.asarray(h_prev, dtype=np.float64)
    h_out = np.asarray(cache["h_new"], dtype=np.float64)
    n_in, n_out = float(np.linalg.norm(h_in)), float(np.linalg.norm(h_out))
    cosine = float(h_in @ h_out / (n_in * n_out)) if n_in > 1e-9 and n_out > 1e-9 else 0.0
    store = {
        "touch": idx + 1,
        "hInNorm": r4(n_in),
        "hOutNorm": r4(n_out),
        "cosine": r4(cosine),
        "carry": r4(1.0 - float(np.mean(cache["z"]))),
        "prevIntent": prev_choice,
        "resetRule": ("The hidden vector starts at zero on the first touch "
                      "of a possession and is dropped when the player is "
                      "substituted or sent off."),
        "sizedBy": f"{HIDDEN_D}-d, owned by the player, not the brain",
        "hiddenD": int(HIDDEN_D),
    }

    seen_count = len(vis_names)
    frame_count = len(frame_names)
    if nearest >= 0.999:
        near_txt = "no opponent inside 15 m"
    else:
        near_txt = f"nearest opponent {nearest * 15.0:.1f} m"
    side = "right" if goal_deg > 1.0 else ("left" if goal_deg < -1.0 else "dead ahead")
    mem_active = [(MEMORY_META[i][1], float(mem[i])) for i in range(8)
                  if mem[i] > 0.02]

    strongest = max(sensors, key=lambda s: abs(s["dp"]))
    narrative = [
        {"tag": "situation",
         "title": "The moment",
         "text": (f"{spec['minute']:.0f}' · "
                  f"{'under pressure' if spec['pressure'] else 'in space'} · "
                  f"{PLAN_META[fed_plan][0].title() if fed_plan >= 0 else 'no plan'}"
                  f" at {fed_conf:.0%} · score {spec['score']:+d}")},
        {"tag": "sees",
         "title": "What he sees",
         "text": (f"His mind puts {frame_count} of 21 in frame (25 m, nearest "
                  f"two of each); the cone then keeps {seen_count}. "
                  f"{near_txt[0].upper() + near_txt[1:]}, {def5} inside 5 m and "
                  f"{def10} inside 10 m. The goal is {goal_m:.0f} m away, "
                  f"{abs(goal_deg):.0f}° to his {side} with the mouth subtending "
                  f"{mouth_deg:.1f}° ({base_vec[3]:.0%} of a penalty-spot sight)."
                  if frame_count or seen_count else
                  "Nobody is in his field of view on this frame.")},
        {"tag": "attention",
         "title": "What he looks at",
         "text": ((f"He puts {top_att[0]['alpha']:.0%} of his attention on "
                   f"{top_att[0]['name']}, then {top_att[1]['alpha']:.0%} on "
                   f"{top_att[1]['name']}. The top three hold "
                   f"{top3_share:.0%} of the mass (attention entropy "
                   f"{att_entropy:.2f} bits, {att_norm:.0%} of maximum). "
                   f"Attention is not the cone: it splits mass over the "
                   f"{len(top_att)} men in his mind's frame, and that frame is "
                   f"not the cone either \u2014 25 m and the nearest two of each, "
                   f"with no angle test at all.")
                  if len(top_att) >= 2 else
                  (f"Only {len(top_att)} token(s) are in frame; attention is "
                   f"forced onto {top_att[0]['name'] if top_att else 'nobody'}."))},
        {"tag": "memory",
         "title": "What he remembers",
         "text": ("The memory block is empty — nothing significant has happened "
                  "to him yet."
                  if not mem_active else
                  "The memory block carries " +
                  ", ".join(f"{n.lower()} {v:.2f}" for n, v in mem_active) +
                  "; it is not a mood, it is eight numbers the GRU reads.")},
        {"tag": "plan",
         "title": "What the team is doing",
         "text": (f"The team plan is "
                  f"{PLAN_META[fed_plan][0].title() if fed_plan >= 0 else 'not adopted'}"
                  f" at {fed_conf:.0%}. The plan head proposes "
                  f"{PLAN_META[pid][0].title()} at {pconf:.0%} — {plan_status}.")},
        {"tag": "recurrence",
         "title": "How it carries",
         "text": (f"The GRU keeps {mean_z:.0%} of the previous hidden state "
                  f"(reset gate {mean_r:.0%}); {active}/32 units fire. The "
                  f"loudest is unit {loudest['i']} at h={loudest['h']:+.2f}.")},
        {"tag": "choice",
         "title": "What he chooses",
         "text": (f"{INTENT_LABELS[k]} at {probs[k]:.1%}, {margin * 100:.0f} "
                  f"points clear of {INTENT_LABELS[second]}. The coach "
                  f"({coach['source'].lower()}) moves "
                  f"{INTENT_LABELS[coach['mover']]} by "
                  f"{coach['delta'][coach['mover']]:+.1f} pts and the "
                  f"consequence critic moves "
                  f"{INTENT_LABELS[cons['mover']] if cons else 'nothing'} by "
                  f"{cons['delta'][cons['mover']] if cons else 0.0:+.1f} pts. "
                  f"Temperature {temp:.2f} sharpens the policy, so the "
                  f"executed draw is {INTENT_LABELS[sampled]}.")},
        {"tag": "evidence",
         "title": "What made the difference",
         "text": (f"Zeroing the STATE block moves this choice by {abl['state']:+.1f} "
                  f"pts, the ROLE block {abl['role']:+.1f}, MEMORY "
                  f"{abl['memory']:+.1f}, PLAN {abl['plan']:+.1f} and ATTENTION "
                  f"{abl['attention']:+.1f}. The single biggest input is "
                  f"{strongest['label']} ({strongest['dp']:+.1f} pts).")},
    ]

    step = {
        "id": f"{position}-{chain['id']}-{idx}",
        "label": spec["label"],
        "note": spec["note"],
        "minute": spec["minute"],
        "pressure": bool(spec["pressure"]),
        "score": int(spec["score"]),
        "world": {"attacksRight": True, "mirror": mirror},
        "carrier": {"name": carrier.name, "position": position,
                    "pos": [r4(x), r4(y)]},
        "goal": {"pos": [105.0, 34.0]},
        "actors": actors,
        "facts": {
            "seen": seen_count, "total": len(actors),
            "mind": frame_count, "mindCap": 4,
            "nearestM": None if nearest >= 0.999 else r4(nearest * 15.0),
            "def5": def5, "def10": def10, "goalDeg": r4(goal_deg),
            "mouthDeg": r4(mouth_deg), "mouthRatio": r4(base_vec[3]),
            "goalM": r4(goal_m), "fwdM": r4(fwd_m), "fatigue": r4(fatigue),
            "entropy": r4(att_entropy), "entropyNorm": r4(att_norm),
            "top3": r4(top3_share), "meanZ": r4(mean_z), "meanR": r4(mean_r),
            "active": active, "temperature": r4(temp),
        },
        "tokens": [
            {"name": t["name"], "role": t["role"], "team": t["team"],
             "dist": t["dist"], "presser": t["presser"],
             "mask": r4(mask[j]), "alpha": r4(alpha[j]),
             "inView": t["name"] in vis_names,
             "pos": t["pos"], "feat": [r4(v) for v in tokens[j]]}
            for j, t in enumerate(token_meta)
        ],
        "state": [r4(v) for v in base_vec],
        "role": [r4(v) for v in role],
        "memory": [r4(v) for v in mem],
        "plan": [r4(v) for v in plan],
        "out": {
            "logits": [r4(v) for v in logits],
            "probs": [r4(v) for v in probs],
            "planLogits": [r4(v) for v in plan_logits],
            "planProbs": [r4(v) for v in plan_probs],
            "argmax": k, "chosen": k, "chosenProb": r4(base_p),
            "sampled": sampled, "second": second, "margin": r4(margin),
        },
        "abl": abl,
        "sensors": sensors,
        "gates": {
            "z": [r4(v) for v in cache["z"]],
            "r": [r4(v) for v in cache["r"]],
            "n": [r4(v) for v in cache["n"]],
            "prev": [r4(v) for v in cache["h"]],
            "h": [r4(v) for v in cache["h_new"]],
            "ctxNorm": r4(float(np.linalg.norm(cache["ctx"]))),
        },
        "units": units,
        "planHead": {"proposed": pid, "prob": r4(pconf),
                     "status": plan_status, "planId": fed_plan,
                     "planConf": r4(fed_conf)},
        "perception": percep,
        "memoryDeep": memdeep,
        "store": store,
        "planState": plan_state,
        "coach": coach,
        "consequence": cons,
        "temper": temper,
        "sampling": samp,
        "narrative": narrative,
        "roleNote": (
            f"role_tail() emits {role_full.size} slots ({len(fam_names)} named "
            f"{family} sensors); this brain's {int(brain.w_q.shape[0])}-d input "
            f"has room for {role_width} role slots "
            f"(24 + {role_width} + 8 + 6). "
            + (f"The first {role_width} are read and the other "
               f"{role_full.size - role_width} are computed, then dropped."
               if role_full.size > role_width else
               f"{role_width - role_full.size} slot(s) are zero padding.")
        ),
    }
    return step, h_new


def build_brain(position: str) -> Dict[str, Any]:
    family, carrier_key = POS_INFO[position]
    path = ROOT / "brains" / f"{position}.json"
    brain = SituationalBrain.load(str(path))
    params = int(sum(np.asarray(v).size for v in brain.w.values()))
    mirror = position in MIRRORED

    chains = []
    for chain in CHAINS[family]:
        h = np.zeros(HIDDEN_D, dtype=np.float64)
        ck = carrier_key
        steps = []
        p0 = chain["steps"][0]["plan"]
        tracker = TeamPlan(team="PLOFA XI", plan_id=int(p0[0]),
                           confidence=float(p0[1]), possession_id=1)
        prev_choice = None
        coach_source = COACH_BY_CHAIN.get((family, chain["id"]), "BALANCED")
        for idx, spec in enumerate(chain["steps"]):
            step, h = build_step(brain, position, family, ck, chain, spec,
                                 h, mirror, idx, coach_source=coach_source,
                                 plan_tracker=tracker,
                                 prev_choice=prev_choice)
            step["touch"] = idx + 1
            prev_choice = INTENT_LABELS[step["out"]["sampled"]]
            steps.append(step)
        chains.append({"id": chain["id"], "label": chain["label"],
                       "sub": chain["sub"], "coach": coach_source,
                       "steps": steps})
        print(f"  {position} · {chain['label']}: {len(steps)} touches")

    meta = brain.meta or {}
    return {
        "id": position,
        "family": family,
        "file": f"brains/{position}.json",
        "params": params,
        "hidden": HIDDEN_D,
        "meta": {
            "sensor_schema": meta.get("sensor_schema", "v5_situational"),
            "training_method": meta.get("training_method", "bptt_situational"),
            "arch_version": meta.get("arch_version", 3),
        },
        "dna": DNA[family],
        "roleNote": chains[0]["steps"][0]["roleNote"],
        "chains": chains,
    }


# ──────────────────────────────────────────────────────────────────────
# DATA DOCUMENT
# ──────────────────────────────────────────────────────────────────────

def build_document() -> Dict[str, Any]:
    print("Building v5 situational brain observatory ...")
    brains = []
    for pos in POSITIONS:
        print(f"[{pos}]")
        brains.append(build_brain(pos))

    state_labels = [{"name": n, "short": s, "desc": d}
                    for n, s, d in STATE_META]
    role_labels = {}
    for fam, names in MENU_NAMES.items():
        role_labels[fam] = [
            (names[i] if i < len(names) else "(unused pad)")
            for i in range(9)
        ]

    return {
        "version": "1.0",
        "title": "Situational Brain Observatory",
        "schema": {
            "name": "v5_situational",
            "arch_version": 3,
            "input": 47,
            "blocks": {"state": 24, "role": 9, "memory": 8, "plan": 6},
            "tokens": N_MAX,
            "token_dim": TOKEN_D,
            "hidden": HIDDEN_D,
            "intents": 10,
            "plans": 4,
            "params_per_position": int(np.sum([b["params"] for b in brains]) / len(brains)),
        },
        "intents": [
            {"name": n, "short": n.replace("_", " ").title()}
            for n in INTENT_LABELS
        ],
        "stateLabels": state_labels,
        "roleLabels": role_labels,
        "memoryLabels": [{"name": n, "short": s, "desc": d}
                         for n, s, d in MEMORY_META],
        "planLabels": [{"name": n, "desc": d} for n, d in PLAN_META],
        "tokenLabels": TOKEN_META,
        "provenance": {
            "pipeline": [
                "mind.perceive() → the vision gate: ≤25 m, nearest 2 + 2, no cone",
                "entity_tokens() → 16 × 12 token set, filled from that frame",
                "perceive() → 24-d state + cone, range and top-k capped scene",
                "role_tail() → 9-d role geometry (named sensors)",
                "memory_features() → 8-d episodic memory block",
                "plan_tail() → 6-d team-plan block",
                "SituationalBrain.forward() → alpha, GRU z/r/n, h, 10 intents + 4 plan logits",
                "_apply_coach_bias() → the manager's log-space intent bonus",
                "ConsequenceEvaluator.correct_policy() → TD-critic payoff correction",
                "_decision_temperature() + _sample_from_probs() → the executed draw",
            ],
            "note": ("Scenes, DNA, memory events and the coach posture are authored. "
                     "Every tensor shown is produced by the shipped pipeline and the "
                     "shipped weights in brains/<POS>.json — nothing is re-implemented "
                     "in JavaScript. Every player here carries a REGISTERED, ENGAGED "
                     "PlayerMind, exactly as auto_run_match.py provisions one before "
                     "kickoff, so the vision gate runs in the production order: the "
                     "mind puts at most 2 teammates and 2 opponents inside 25 m in "
                     "frame (no angle test), the attention set is built from that "
                     "frame, and the cone is applied afterwards to the 24-d state and "
                     "the role tail. The two gates therefore disagree by design — a "
                     "token can be a man the cone discarded. The coach bias, the "
                     "consequence critic and the "
                     "temperature draw use the shipped functions; the cognition route "
                     "(temperament reshaping the sample) is shown as a labelled "
                     "counterfactual because this scene runs the neural core. One "
                     "thing the page reports rather than fixes: the TD critic's "
                     "output explodes on the score-difference input when a team is "
                     "losing, so its reward-band clip flattens the correction. That "
                     "is the shipped behaviour, not an artefact of this page."),
        },
        "brains": brains,
    }


def main() -> int:
    doc = build_document()
    template = (Path(__file__).parent / "app_template.html").read_text(encoding="utf-8")
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    html = template.replace("/*__BRAIN_DATA__*/null", payload)
    out = Path(__file__).parent / "brain_thinking.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nwritten -> {out}  ({size_mb:.2f} MB)")
    print(f"brains: {len(doc['brains'])} | states: "
          f"{sum(len(c['steps']) for b in doc['brains'] for c in b['chains'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
