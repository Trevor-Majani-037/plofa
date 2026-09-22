"""Rebalance the on-ball fitness surrogate so SWITCH is no longer over-priced.

The trained per-position brains (brains/*.json) were evolved against
brains/surrogate_pos.json.  A handful of sparse SWITCH cells carry
lucky, goal-weighted payoff samples that outprice the safe options
(e.g. CM midfield-pressure bucket SWITCH=2.31 vs SAFE_PASS=1.78, and
own-half buckets at 1.6-2.2).  n a state, evolution picks the ARGMAX of
``expected_success``, so the converged nets learned "SWITCH is the best
midfield play" and emit SWITCH on ~1 in 4 touches -> the match engine's
forced-switch delivery truncates possessions (events collapse 3000+ ->
~1900).

This transform keeps genuine final-third switches untouched (they are a
killer cross-field pass) but caps SWITCH expected-success in every
OWN-HALF / MIDDLE bucket strictly BELOW the best safe option
(SAFE_PASS / RECYCLE / PROTECT_POSSESSION / PROGRESSIVE_PASS), so
retraining stops rewarding midfield/own-half switches.  The neural
policy still decides everything; only its learned value signal changes.
"""
import copy
import json
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "brains", "surrogate_pos.json")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "surrogate_switchfix.json")

SAFE_INTENTS = ("SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION",
                "PROGRESSIVE_PASS")
CAP_RATIO = 0.85
MISSING_SAFE_CAP = 0.8   # no safe reference available -> hard ceiling


def _cap_switch_cell(bucket: str, cell: dict) -> float:
    if "FT1" in bucket:
        return float(cell.get("SWITCH", MISSING_SAFE_CAP))
    cell_sw = float(cell.get("SWITCH", MISSING_SAFE_CAP))
    safe = [float(cell[k]) for k in SAFE_INTENTS if k in cell and cell[k] is not None]
    if safe:
        return min(cell_sw, CAP_RATIO * max(safe))
    return min(cell_sw, MISSING_SAFE_CAP)


def main() -> None:
    with open(SRC, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if "table" not in data:
        raise SystemExit("surrogate_pos.json has no 'table'")

    total = 0
    lifted = 0
    for pos, buckets in data["table"].items():
        for bucket, cell in buckets.items():
            if not isinstance(cell, dict) or "SWITCH" not in cell:
                continue
            old = float(cell["SWITCH"])
            new = _cap_switch_cell(bucket, cell)
            if new < old - 1e-9:
                cell["SWITCH"] = new
                total += 1
                if new > old:
                    lifted += 1
    data["prior"] = 0.5
    data["meta"] = {
        "source": os.path.relpath(SRC, os.path.dirname(DST)),
        "transform": "switch_overprice_cap",
        "rule": ("SWITCH capped at %.2f x max(safe) outside final-third; "
                 "final-third SWITCH untouched") % CAP_RATIO,
        "cells_capped": total,
        "regenerated": True,
    }
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"capped {total} SWITCH cells; wrote {DST}")


if __name__ == "__main__":
    main()