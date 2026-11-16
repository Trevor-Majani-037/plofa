"""BEFORE vs AFTER: what does "all brains at 10 Hz" actually cost?

This is a TEST script. It does NOT touch season state, does NOT call
auto_run_match, and does NOT modify any project file. Everything is
monkeypatched at runtime and restored afterwards.

Arms
----
  count    : run as-is, but count and time every brain call (the truth
             about what the current hybrid actually does)
  all10hz  : force the expensive brain work on EVERY tick for EVERY player
             (caches bypassed) -- the literal "all brains at 10 Hz" proposal
  realistic: consult each brain at a human decision cadence instead of
             10 Hz -- a footballer does not re-decide 10x/second

Usage
-----
  py -3 _brain_arm_test.py count
  py -3 _brain_arm_test.py all10hz
  py -3 _brain_arm_test.py realistic --tick-hz 3
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import match_engine as me
import football_brain as fb

# ── counters ──────────────────────────────────────────────────────────
CALLS: dict[str, int] = defaultdict(int)
SECS: dict[str, float] = defaultdict(float)


def wrap(obj, attr, label):
    orig = getattr(obj, attr)
    if getattr(orig, "_instrumented", False):
        return orig

    def inner(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            CALLS[label] += 1
            SECS[label] += time.perf_counter() - t0

    inner._instrumented = True
    inner._orig = orig
    setattr(obj, attr, inner)
    return inner


def instrument():
    wrap(fb.OffBallBrain, "forward", "football_brain.OffBallBrain.forward")
    wrap(fb.TeamPressBrain, "forward", "football_brain.TeamPressBrain.forward")
    wrap(fb.DefensiveActionBrain, "forward",
         "football_brain.DefensiveActionBrain.forward")
    wrap(fb.DefensiveActionBrain, "predict",
         "football_brain.DefensiveActionBrain.predict")
    wrap(me, "_team_press_g", "match_engine._team_press_g")
    wrap(me.MatchEngine, "_offball_press_prob",
         "match_engine._offball_press_prob")
    wrap(me, "_def_action_choice", "match_engine._def_action_choice")


# ── arms ──────────────────────────────────────────────────────────────
def arm_count():
    """Do nothing: measure the current hybrid."""
    return "count"


def arm_all10hz():
    """Force a fresh brain decision on EVERY tick for EVERY player.

    Two throttles have to be defeated, and clearing the caches only
    defeats one of them:

      1. memo caches -- `_team_press_g_cache` (per team-tick) and
         `_offball_press_cache` (per player+ball+danger)
      2. the PRESS-WINDOW GATE in `_offball_move_player`:
             if cst["p"] <= 0.0 and cst["allow"] == 0.0:
         which consults the brain only when a press window OPENS. Clearing
         caches alone leaves this guard in place, so the brain is still
         called ~0.25 times per second rather than 10.

    Resetting `cst["p"]` and `cst["allow"]` every tick re-opens the gate,
    which is what "all brains at 10 Hz" actually means.
    """
    orig_move = me.MatchEngine._offball_move_player

    def move(self, pname, *a, **k):
        cst = self._chase_state.get(pname)
        if cst is not None:
            cst["p"] = 0.0
            cst["allow"] = 0.0
        self._team_press_g_cache.clear()
        self._offball_press_cache.clear()
        return orig_move(self, pname, *a, **k)

    me.MatchEngine._offball_move_player = move
    return "all10hz"


def arm_realistic(tick_hz: float = 3.0):
    """Consult each player's brain on a human cadence, not every tick.

    A footballer's *reaction* is 0.2-0.5 s, and a tactical re-read is far
    slower than that. We throttle per player: the owning brain may only
    be re-consulted once every 1/tick_hz seconds.
    """
    period = 1.0 / max(0.1, tick_hz)
    last: dict[str, float] = {}
    orig_obw = __import__("offball_brain_wiring").offball_press_prob

    def throttled(engine, pname, position, team, bx, by, danger_t, **kw):
        now = engine.state.match_clock_s
        prev = last.get(pname)
        if prev is not None and (now - prev) < period:
            return None          # hold the previous decision, no brain call
        last[pname] = now
        return orig_obw(engine, pname, position, team, bx, by, danger_t, **kw)

    import offball_brain_wiring as obw
    obw.offball_press_prob = throttled
    return f"realistic@{tick_hz}Hz"


# ── the match (built exactly like run_match, but writes nothing) ───────
def build_engine():
    from datetime import date
    from match_engine import (MatchEngine, MatchConfig, TeamProfile,
                              TeamStyle, PlayingStyle, Intensity)
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController
    from manager_profile import ManagerProfile
    from run_match import (HOME_STARTERS, HOME_SUBS, AWAY_STARTERS,
                           AWAY_SUBS, HOME_SUPERSTARS, AWAY_SUPERSTARS)

    HOME, AWAY = "Hartwell City", "Thornfield"
    cfg = MatchConfig(home_team=HOME, away_team=AWAY,
                      match_date=date(2026, 10, 27), matchday=1)
    home_style = TeamProfile(name=HOME, style=TeamStyle.ATTACKING,
                             playing_style=PlayingStyle.HIGH_PRESS,
                             intensity=Intensity.HIGH)
    away_style = TeamProfile(name=AWAY, style=TeamStyle.FLUID_COUNTER,
                             playing_style=PlayingStyle.COUNTER,
                             intensity=Intensity.MEDIUM)
    home = SquadBuilder.build(HOME, HOME_STARTERS, HOME_SUBS,
                              team_superstars=HOME_SUPERSTARS)
    away = SquadBuilder.build(AWAY, AWAY_STARTERS, AWAY_SUBS,
                              team_superstars=AWAY_SUPERSTARS)
    sub = SubstitutionController(
        home_team=HOME, away_team=AWAY,
        home_subs_bench=home["substitutes"],
        away_subs_bench=away["substitutes"],
        home_style=home_style.style.value,
        away_style=away_style.style.value)

    eng = MatchEngine(cfg, home_style, away_style)
    eng.set_squad(HOME, home["starters"], home["substitutes"])
    eng.set_squad(AWAY, away["starters"], away["substitutes"])
    eng.set_stamina_controller(sub)
    eng.set_managers(home_manager=ManagerProfile(name="H", club=HOME),
                     away_manager=ManagerProfile(name="A", club=AWAY))
    eng.quiet = True
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["count", "all10hz", "realistic"])
    ap.add_argument("--tick-hz", type=float, default=3.0)
    args = ap.parse_args()

    instrument()
    label = {"count": arm_count,
             "all10hz": arm_all10hz,
             "realistic": lambda: arm_realistic(args.tick_hz)}[args.arm]()

    eng = build_engine()
    t0 = time.perf_counter()
    result = eng.simulate()
    wall = time.perf_counter() - t0

    print("\n" + "=" * 62)
    print(f"ARM: {label}")
    print(f"WALL: {wall:.2f}s")
    print(f"RESULT: {result.summary().splitlines()[1].strip()}")
    print("-" * 62)
    print(f"{'brain hook':<48} {'calls':>10} {'secs':>9}")
    tot_c, tot_s = 0, 0.0
    for k in sorted(CALLS, key=lambda x: -SECS[x]):
        print(f"{k:<48} {CALLS[k]:>10,} {SECS[k]:>8.2f}s")
        tot_c += CALLS[k]
        tot_s += SECS[k]
    print("-" * 62)
    print(f"{'TOTAL':<48} {tot_c:>10,} {tot_s:>8.2f}s")
    print(f"{'share of wall time':<48} {'':>10} {100*tot_s/max(wall,1e-9):>8.1f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
