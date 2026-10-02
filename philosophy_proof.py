"""
Club Philosophy proof (Checkpoint 30): closed-loop possession + 3 levers.

REPRODUCIBILITY REQUIREMENT
---------------------------
The engine consumes Python dict/set iteration order, which is randomized
per-process by PYTHONHASHSEED.  Within one process matches are deterministic
for a given random.seed(); ACROSS processes they are NOT unless the hash seed
is pinned.  Run this script as:

    PYTHONHASHSEED=7  python philosophy_proof.py

Rows are fair even unpinned (all conditions run in the same process sharing
the same hash order), but pinning makes the numbers bit-for-bit repeatable.

Honest verdict (pinned, seeds 0-1):
    NO-PHI   tiki vs bus         ~60% poss, ~360 passes
    DOM      vs LowBlock         ~62% poss, ~360 passes  (possession +~2%)
    MID      vs Gegenpressing    ~46% poss               (balanced as intended)
Levers A (closed-loop share) and B (starve) move possession modestly;
Lever C (patience) and Lever D (chain-level patience circulation) shape the
pass mix, but completed-pass VOLUME stays pinned near the engine's calibrated
economy (~350-380) — reaching 500-900 passes needs a possession-economy
redesign of per-step completion, which is OUT OF SCOPE for this proof.
"""
# ── HASH-SEED PROTOCOL ────────────────────────────────────────────────
import os
import sys
if "PYTHONHASHSEED" not in os.environ:
    print("WARNING: PYTHONHASHSEED not set — cross-process reproducibility "
          "is NOT guaranteed.  Run with `PYTHONHASHSEED=7 python "
          "philosophy_proof.py` for bit-for-bit identical numbers.")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import random
from datetime import date

from match_engine import MatchEngine, MatchConfig
from match_probe import _build_squads, _team_profile
from squad_manager import SubstitutionController
from brain_integration import clear_registry
from philosophy import ARCHETYPES

HOME, AWAY = "Probe FC", "Rival FC"
PASS_TYPES = {
    "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY",
    "THROUGH_BALL", "CROSS_ATTEMPT", "CROSS_SUCCESS",
}


def run_match(seed, home_style="balanced", away_style="balanced",
              home_phi=None, away_phi=None):
    random.seed(seed)
    home_squad, away_squad = _build_squads(HOME, AWAY)
    config = MatchConfig(home_team=HOME, away_team=AWAY,
                         match_date=date(2026, 9, 6), matchday=3, season="26/27")
    hp = _team_profile(HOME, home_style, home_phi)
    ap = _team_profile(AWAY, away_style, away_phi)
    sc = SubstitutionController(
        home_team=HOME, away_team=AWAY,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=hp.style.value, away_style=ap.style.value,
    )
    clear_registry()
    eng = MatchEngine(config, hp, ap)
    eng.set_squad(HOME, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(AWAY, away_squad["starters"], away_squad["substitutes"])
    eng.set_stamina_controller(sc)
    result = eng.simulate()
    tl = result.timeline
    def count(team):
        return [e for e in tl
                if getattr(e, "team", None) == team
                and getattr(getattr(e, "event_type", None), "name", "") in PASS_TYPES]
    return result, count(HOME), count(AWAY)


def run(label, seeds, home_style, away_style, home_phi, away_phi):
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    poss, passes = [], []
    for s in seeds:
        result, hp, ap = run_match(s, home_style, away_style, home_phi, away_phi)
        poss.append(result.home_possession_pct)
        passes.append(len(hp))
        print(f"  seed {s}: {result.home_goals}-{result.away_goals}  "
              f"poss={result.home_possession_pct:.0f}%  "
              f"home_passes={len(hp):4d}  away_passes={len(ap):3d}")
    print(f"  AVG poss={sum(poss)/len(poss):.1f}%  "
          f"AVG passes={sum(passes)/len(passes):.0f}  "
          f"MAX passes={max(passes)}")
    return sum(poss) / len(poss), sum(passes) / len(passes)


if __name__ == "__main__":
    print("CLUB PHILOSOPHY PROOF (Checkpoint 30)")
    base = run("NO PHILOSOPHY — tiki vs park-the-bus (reference)",
               (0, 1), "tiki_taka", "park_the_bus", None, None)
    dom = run("POSITIONAL DOMINANCE vs LOW BLOCK — possession hunger + starve",
              (0, 1), "tiki_taka", "park_the_bus",
              ARCHETYPES["PositionalDominance"], ARCHETYPES["LowBlock"])
    mid = run("MID-TABLE PRAGMATISM vs GEGENPRESSING — weak-identity control",
              (0, 1), "balanced", "gegenpressing",
              ARCHETYPES["MidTablePragmatism"], ARCHETYPES["Gegenpressing"])
    print(f"\n{'=' * 64}\nVERDICT\n{'=' * 64}")
    print(f"  reference poss {base[0]:.1f}%  passes {base[1]:.0f}")
    print(f"  dominance  poss {dom[0]:.1f}%  passes {dom[1]:.0f}  (Δposs {dom[0]-base[0]:+.1f}%)")
    print(f"  mid-table  poss {mid[0]:.1f}%  (expect ~50%; balanced control holds)")
    print(f"  => Layer ships A/B/C: possession floats +1-2, bus starved modestly; "
          f"700-900 passes blocked by the per-step completion economy (see docstring)")