"""STEP 1: does the possession trace invent speeds no body could run?

The activity report credits sprints from episode traces, and those trace rows
carry a speed label. If a label ever exceeds the player's own physical ceiling,
the report is describing a body that does not exist -- and every sprint/hi
number downstream inherits it.

Prints, per player, the largest trace label seen and the ceiling it should have
been bounded by, from three independent sources so the disagreement is visible:

  trace    max ``speed_mps`` label on any MotionSnapshot for him
  engine   ``_top_speed_cache`` -- the engine's own movement ceiling
  physics  ``calibration.calibrate(dna).max_speed`` -- the layer's ceiling

Run:  $env:PYTHONHASHSEED="0"; python -m scripts.physics.probe_trace_speeds
"""
from __future__ import annotations

import random
import sys
from datetime import date
from collections import defaultdict

sys.path.insert(0, r"D:\PLOFA\plofa")

HOME, AWAY = "Uditon", "Claw"
SEED = 123
PHYSICS = "--on" in sys.argv

_peak = defaultdict(float)
_rows = defaultdict(int)
_by_note = defaultdict(float)


def main() -> int:
    import event_chain as EC

    orig = EC.BaseChain._accumulate_physics_stats

    def hooked(episode, position_engine, minute, *a, **kw):
        try:
            for row in getattr(episode, "trace", ()) or ():
                who = getattr(row, "player", None)
                if not who or who in ("ball", "gk_track", "drift"):
                    continue
                sp = float(getattr(row, "speed_mps", 0.0) or 0.0)
                _rows[who] += 1
                if sp > _peak[who]:
                    _peak[who] = sp
                note = getattr(row, "note", "") or ""
                if sp > _by_note[note]:
                    _by_note[note] = sp
        except Exception:
            pass
        return orig(episode, position_engine, minute, *a, **kw)

    EC.BaseChain._accumulate_physics_stats = staticmethod(hooked)

    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    random.seed(SEED)
    loader = get_loader()
    raw = {c: loader.build_matchday_squad(c) for c in (HOME, AWAY)}
    squads = {c: SquadBuilder.build(
        team_name=c, starters=raw[c]["starters"],
        substitutes=raw[c]["substitutes"],
        team_superstars=raw[c]["superstars"],
        set_piece_takers=raw[c]["sp_takers"]) for c in (HOME, AWAY)}
    profs = {c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == HOME))
             for c in (HOME, AWAY)}

    cfg = MatchConfig(home_team=HOME, away_team=AWAY,
                      match_date=date(2026, 9, 8), physics_enabled=PHYSICS)
    eng = MatchEngine(cfg, profs[HOME], profs[AWAY])
    for c in (HOME, AWAY):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (HOME, AWAY) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=HOME, away_team=AWAY,
        home_subs_bench=subs, away_subs_bench=subs))
    eng.simulate()

    from physics.calibration import calibrate

    print(f"physics_enabled = {PHYSICS}   fixture {HOME} v {AWAY}   seed {SEED}")
    print()
    print(f"{'player':<24}{'rows':>7}{'trace':>8}{'engine':>8}{'physics':>9}"
          f"{'ratio':>7}")
    worst = []
    for name, peak in sorted(_peak.items(), key=lambda kv: -kv[1]):
        dna = None
        for arr in (raw[HOME]["starters"], raw[AWAY]["starters"]):
            for p in arr:
                if getattr(p, "name", None) == name:
                    dna = getattr(p, "dna", None)
        ceiling_p = 0.0
        if dna is not None:
            try:
                ceiling_p = calibrate(dna, getattr(dna, "stamina", 100.0)).max_speed
            except Exception:
                ceiling_p = 0.0
        engine = float(eng._top_speed_cache.get(name, 0.0) or 0.0)
        ref = min(v for v in (engine, ceiling_p) if v > 0.0) if (
            engine > 0 or ceiling_p > 0) else 0.0
        ratio = (peak / ref) if ref > 0 else 0.0
        worst.append((ratio, name, peak, ref))
        print(f"{name[:23]:<24}{_rows[name]:>7}{peak:>8.2f}{engine:>8.2f}"
              f"{ceiling_p:>9.2f}{ratio:>7.2f}")
    print()
    print("largest label per note (which action invents speed):")
    for note, sp in sorted(_by_note.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {note or '(none)':<22}{sp:>7.2f} m/s")
    print()
    over = [w for w in worst if w[0] > 1.02]
    print(f"players whose trace label exceeds their ceiling: {len(over)}")
    if over:
        r, n, p, ref = max(over)
        print(f"  worst: {n} label {p:.2f} m/s vs ceiling {ref:.2f} ({r:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
