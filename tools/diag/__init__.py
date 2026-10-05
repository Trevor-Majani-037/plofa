"""Measurement instruments — see `tools/__init__.py`.

Importable both as a package (`tools.diag._diag_x`) and by bare name
(`_diag_x`, via the path shim in the parent package), because these scripts
were written to import each other by bare name when they lived in the repo
root. Do not add assertions here; this is where things get MEASURED.
"""