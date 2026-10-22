"""NEUTRALITY — does provisioning a mind change the match? (2026-10-22)

Run ONE arm per PROCESS (the module-state rule):
    .venv\\Scripts\\python.exe tools\\diag\\_diag_memory_neutrality.py on
    .venv\\Scripts\\python.exe tools\\diag\\_diag_memory_neutrality.py off

WHY THIS PROBE EXISTS
----------------------
The memory fix (2026-10-22) makes three changes to the live path:

  1. every player gets a MIND (``MatchEngine.set_squad`` provisions an
     UNENGAGED one), so the episodic memory has somewhere to live in every
     runner and not only in production;
  2. the memory tap is the match engine's DEFAULT observer, so every
     absorbed event reaches the players' minds;
  3. ``memory_features`` turns that memory into an 8-d INPUT block.

Change 1 is the dangerous one.  ``brain_integration._maybe_cognition``
routes to the mind path whenever a mind EXISTS, and
``CognitionDecisionBrain.decide`` is a DIFFERENT pipeline from the vanilla
core — FOV-gated scene, tempered sampling.  If an ENGAGED mind were
provisioned, every decision in the match would change.

The provisioned minds are UNENGAGED, so the routing falls straight through
to the vanilla core — but that is an argument, not a measurement.  This
probe IS the measurement: the same seed, the same fixture, the same
shipped brains, and the two arms must produce a BYTE-IDENTICAL timeline.

It also proves the other half of the claim: in the ``on`` arm the memory
must actually FILL (a mind that observes nothing is not a memory).

The fixture is the REAL roster (``_diag_v7_export_real.run_real_match``
builds it), because the scratch fixture's two teams share generic player
names and the exporter keys its per-player stats by name.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))
sys.path.insert(0, os.path.join(ROOT, "tools", "gates"))

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["on", "off"])
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--home", default="Natrican")
ap.add_argument("--away", default="Tryox City")
A = ap.parse_args()

random.seed(A.seed)
try:
    import numpy as np

    np.random.seed(A.seed)
except Exception:
    pass

if A.arm == "off":
    # Disable BOTH halves of the memory fix: no mind is provisioned, and
    # the memory tap is switched off entirely (set_cognition_observer(None)
    # now disables the default as well as the explicit hook).
    import memory_features

    memory_features.provision_minds = lambda names: 0
    from match_engine import set_cognition_observer

    set_cognition_observer(None)

from brain_integration import set_brain_dir

set_brain_dir("brains")

from football_brain import FootballBrain
from brain_integration import clear_registry, register_brain
from match_engine import MatchConfig, MatchEngine
from player_dna import SquadBuilder
from roster_loader import get_loader
from squad_manager import SubstitutionController
from validate_neural_xl import POSITION_BRAIN_MAP
import production_roster_sanity as prs

match_date = date(2026, 9, 6)
loader = get_loader()
clubs = loader.get_all_clubs()
for team in (A.home, A.away):
    if team not in clubs:
        raise SystemExit(f"team {team!r} not in the roster")

from season_manager import SeasonState

season_state = SeasonState("26/27", prs.SEASON_STATE_FILE)
home_avail = prs._build_availability(A.home, 3, season_state,
                                     prs.OUTPUTS_DIR, match_date)
away_avail = prs._build_availability(A.away, 3, season_state,
                                     prs.OUTPUTS_DIR, match_date)
home_raw = loader.build_matchday_squad(A.home, availability=home_avail)
away_raw = loader.build_matchday_squad(A.away, availability=away_avail)
home_style = prs._resolve_team_profile(A.home, home_raw["formation"],
                                       is_home=True)
away_style = prs._resolve_team_profile(A.away, away_raw["formation"],
                                       is_home=False)
home_squad = SquadBuilder.build(
    team_name=A.home, starters=home_raw["starters"],
    substitutes=home_raw["substitutes"],
    team_superstars=home_raw["superstars"],
    set_piece_takers=home_raw["sp_takers"])
away_squad = SquadBuilder.build(
    team_name=A.away, starters=away_raw["starters"],
    substitutes=away_raw["substitutes"],
    team_superstars=away_raw["superstars"],
    set_piece_takers=away_raw["sp_takers"])
all_players = (home_squad["starters"] + home_squad["substitutes"]
               + away_squad["starters"] + away_squad["substitutes"])
prs._apply_starting_stamina(all_players, {**home_avail, **away_avail},
                            season_state, match_date)
prs._attach_souls(all_players)

# The shipped brains for the HOME XI only — the away side auto-loads the
# same set by position, so the match is set-vs-set (the structure every
# harness in this project uses).
clear_registry()
bound = 0
for p in home_squad["starters"]:
    stem = POSITION_BRAIN_MAP.get(p.position)
    if not stem:
        continue
    path = os.path.join(ROOT, "brains", f"{stem}.json")
    if not os.path.exists(path):
        continue
    register_brain(p.name, FootballBrain.load(path))
    bound += 1

config = MatchConfig(home_team=A.home, away_team=A.away,
                      match_date=match_date, matchday=3, season="26/27")
sc = SubstitutionController(
    home_team=A.home, away_team=A.away,
    home_subs_bench=home_squad["substitutes"],
    away_subs_bench=away_squad["substitutes"],
    home_style=home_style.style.value, away_style=away_style.style.value)

engine = MatchEngine(config, home_style, away_style)
engine.set_squad(A.home, home_squad["starters"], home_squad["substitutes"])
engine.set_squad(A.away, away_squad["starters"], away_squad["substitutes"])
engine.set_stamina_controller(sc)

result = engine.simulate()

rows = ["|".join([
    e.event_type.name,
    getattr(e, "player", "") or "",
    getattr(e, "secondary_player", "") or "",
    f"{getattr(e, 'location_x', 0.0):.1f}",
    f"{getattr(e, 'location_y', 0.0):.1f}",
    f"{getattr(e, 'minute', 0.0):.2f}",
]) for e in result.timeline]
digest = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]
goals = len([e for e in result.timeline if e.event_type.name == "GOAL"])

# ── the memory side of the claim ────────────────────────────────────
from cognition.mind import get_mind

minds = 0
events_seen = 0
nonzero_blocks = 0
sample = []
for p in home_squad["starters"] + away_squad["starters"]:
    m = get_mind(p.name)
    if m is None:
        continue
    minds += 1
    n = len(m.memory.events)
    events_seen += n
    try:
        from memory_features import memory_features as _mf

        block = _mf(p.name, m)
    except Exception:
        block = None
    if block is not None and float(block[:6].max()) > 0.0:
        nonzero_blocks += 1
    if len(sample) < 3:
        sample.append({
            "player": p.name,
            "memories": n,
            "block": [round(float(v), 3) for v in block] if block is not None else None,
        })

payload = {
    "arm": A.arm, "seed": A.seed, "bound": bound,
    "events": len(rows), "goals": goals, "digest": digest,
    "minds": minds, "memories_observed": events_seen,
    "players_with_memory": nonzero_blocks,
    "sample": sample,
}
line = json.dumps(payload, ensure_ascii=False)
# ALSO write it to a file, so the result survives a shell pipeline that
# truncates stdout (a truncated pipeline can leave the upstream command's
# exit code looking like a failure, which is exactly the kind of thing
# that turns a good result into a suspicious one).
out_dir = os.path.join(ROOT, "artifacts")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir,
                        f"_diag_memory_neutrality_{A.arm}.json")
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(line + "\n")
print(line)
print(f"WROTE {out_path}")
