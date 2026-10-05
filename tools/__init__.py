"""Developer tooling — NOT engine code.

Nothing in this package is imported by the live match engine. The engine must
never depend on anything under `tools/`.

Two sub-packages, deliberately split by PURPOSE rather than by name prefix:

  tools/diag/   98 measurement instruments. One question, one script. Most are
                one-off; some back a finding that is still open. They measure,
                they do not assert.
  tools/gates/  11 recurring behavioural gates -- `validate_neural_xl.py
                --matches 7`, `validate_team_press.py`, `compare_striker.py`.
                These are repeatable checks on the shipped engine and are the
                ones worth wiring into CI.

`tests/` is a third thing and must stay separate: it asserts. A directory of
probes placed there would dilute "run the suite and see if it still works".

THE PATH SHIM, and why it is not optional. These scripts import each other by
BARE module name (`from _diag_watch import build` — 57 such sites). They were
written when everything sat in the repo root, where a bare name just resolved.
Importing this package therefore puts the sub-directories back on `sys.path`,
so those bare imports keep working unchanged. Without it, moving the files
breaks every one of them.

The sub-directories are ALSO importable as packages, which is how
`tests/test_plofa_export.py` reaches two of them (`tools.diag._diag_watch`).
Those scripts do real work at import time, so that is a live cost the test
already paid when they lived at the root.

Run from the repo root:

    $env:PYTHONPATH="."
    .venv\\Scripts\\python.exe tools\\diag\\_diag_something.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "diag", "gates"):
    _p = os.path.join(_HERE, _sub) if _sub else _HERE
    if _p not in sys.path:
        # Inserted, not appended, to match the behaviour these scripts had
        # when they lived in the repo root.
        sys.path.insert(0, _p)