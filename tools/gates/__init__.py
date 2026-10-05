"""Recurring behavioural gates — see `tools/__init__.py`.

Repeatable checks on the shipped engine (`validate_neural_xl.py --matches 7`,
`validate_team_press.py`, `compare_striker.py`). Unlike `tools/diag/`, a good
result here is a real gate: a challenger that fails one of these has not
shipped. These are the candidates for CI.

Each runs real matches, so budget CPU deliberately before running several.
"""