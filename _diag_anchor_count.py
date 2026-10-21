"""Count the anchor occurrences in surrogate_collect.py (one per line)."""
import sys

PATS = [
    "if event_type_name(e) in _UNCLAIMABLE_EVENTS:",
    '_NON_ACTION_EVENTS = frozenset({"BALL_RECEIPT", "PRESS", "PRESS_RESISTED"})',
    "if idx in used:",
    "_CHAIN_CALLS.clear()",
    "def _restore_dispatcher_capture():",
    'pairs.append((rec.sensors, rec.intent, 0.0, "NONE", rec.position))',
    "ar = True if not attacks_right_by_team else bool(",
    'if getattr(e, "player", None) != rec.player:',
    "for name, raw in _SAVED_DISPATCHER.items():",
    "used = set()",
]

path = sys.argv[1] if len(sys.argv) > 1 else "surrogate_collect.py"
with open(path, encoding="utf-8", newline="") as f:
    s = f.read()
for p in PATS:
    print(f"{s.count(p)}  {p}")
