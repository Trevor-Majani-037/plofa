"""FOV audit — did the cognition merge NERF the previous perception FOV?

Per live decision of ONE seeded match, we compute BOTH views of the SAME
scene side by side:

    OLD EYES      = perception.perceive(PerceptionConfig()) — the DEFAULT
                    DNA-scaled radius, role FOV cone, role top-k, noise
                    (this is exactly what ran before the merge)
    COGNITION EYES= VisionSystem gate (fixed 25 m, at most 2 teammates +
                    2 opponents, 360 degrees, no noise)

Then we count how often cognition kept FEWER actors than the old eyes, and
how often the single BEST forward option (as the old sensors would have
scored it) was dropped by the cognition gate.

Run:  .\\.venv\\Scripts\\python.exe fov_audit.py
"""
import random
import statistics
from collections import defaultdict

import run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from brain_integration import set_cognition
from cognition_brain import engage_mind_observer, disengage_mind_observer, \
    CognitionDecisionBrain
from cognition.mind import clear_minds, register_mind, new_mind
from perception import perceive, PerceptionConfig


# ── scene-level replication of the engine's "best forward option" scorer ──
def _best_forward(position_engine, x, y, teammates, old_defs, attacks_right):
    """Name of the best forward-running teammate, scored like the engine's
    brain_sensors._best_forward_teammate (progress*0.55 + openness*0.45),
    evaluated on the OLD layer's seen defenders (the honest prior view)."""
    best_val, best_name = -1.0, None
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        try:
            tx, ty = position_engine.get_position(t.name)
        except Exception:
            continue
        progress = (tx - x) if attacks_right else (x - tx)
        if progress < 4.0:
            continue
        # openness at (tx, ty): nearest defender distance (engine formula)
        bd = None
        for d in old_defs or []:
            if getattr(d, "position", "") == "GK":
                continue
            try:
                dx, dy = position_engine.get_position(d.name)
            except Exception:
                continue
            dist = ((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5
            if bd is None or dist < bd:
                bd = dist
        openv = clamp((bd - 1.5) / 8.5, 0.0, 1.0) if bd is not None else 1.0
        value = min(progress / 35.0, 1.0) * 0.55 + openv * 0.45
        if value > best_val:
            best_val, best_name = value, t.name
    return best_name


STATS = defaultdict(lambda: {"calls": 0, "old_seen": [], "cog_seen": [],
                             "shrunk": 0, "old_radius": []})
DROPPED_BEST = 0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    global DROPPED_BEST
    home = SquadBuilder.build(
        team_name=R.HOME_TEAM, starters=R.HOME_STARTERS, substitutes=R.HOME_SUBS,
        team_superstars=R.HOME_SUPERSTARS, set_piece_takers=R.HOME_SP_TAKERS,
    )
    away = SquadBuilder.build(
        team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS, substitutes=R.AWAY_SUBS,
        team_superstars=R.AWAY_SUPERSTARS, set_piece_takers=R.AWAY_SP_TAKERS,
    )
    all_players = (home["starters"] + home["substitutes"]
                   + away["starters"] + away["substitutes"])
    for p in all_players:
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]

    def wrapped(player, x, y, teammates, defenders, position_engine,
                team_profile, under_pressure, attacks_right, game_state,
                minute=45.0, soul=None, record_trace=False):
        # OLD EYES — the default (DNA-scaled) perception layer of the
        # pre-merge engine, on the SAME scene.
        old_vec, scene = perceive(
            player, x, y, teammates, defenders, position_engine,
            under_pressure, attacks_right, game_state, minute,
            team_possession=True, config=PerceptionConfig(), return_scene=True,
        )
        old_tms, old_defs = scene["teammates"], scene["defenders"]

        # COGNITION EYES — the attention gate.
        mind = None
        from cognition.mind import get_mind
        mind = get_mind(player.name)
        gated_tms, gated_defs = ([], [])
        if mind is not None:
            gated_tms, gated_defs = mind.perceive(
                player, x, y, teammates, defenders, position_engine)

        role = (getattr(player, "position", "?") or "?").rstrip("0123456789")
        s = STATS[role]
        s["calls"] += 1
        s["old_seen"].append(len(old_tms) + len(old_defs))
        s["cog_seen"].append(len(gated_tms) + len(gated_defs))
        if len(old_tms) + len(old_defs) > len(gated_tms) + len(gated_defs):
            s["shrunk"] += 1

        # Did the OLD eyes have a forward option beyond what cognition saw?
        best = _best_forward(position_engine, x, y, old_tms, old_defs,
                             attacks_right)
        if best is not None and best not in {t.name for t in gated_tms}:
            DROPPED_BEST += 1
        return CognitionDecisionBrain.decide(
            player, x, y, teammates, defenders, position_engine, team_profile,
            under_pressure, attacks_right, game_state, minute, soul,
            record_trace)

    CognitionDecisionBrain.decide = staticmethod(wrapped)

    clear_minds()
    for p in all_players:
        register_mind(p.name, new_mind(view_radius=25.0))
    set_cognition(True)
    engage_mind_observer()

    cfg = MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, match_date=R.MATCH_DATE,
        matchday=R.MATCHDAY, season=R.SEASON, competition=R.COMPETITION,
        venue=R.VENUE, stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )
    random.seed(424242)
    eng = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home["starters"], home["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away["starters"], away["substitutes"])
    res = eng.simulate()
    print(f"\n  match: {R.HOME_TEAM} {eng.state.home_goals} - {res.away_goals} {R.AWAY_TEAM}")

    set_cognition(False)
    disengage_mind_observer()
    clear_minds()

    print("\n  Per-role FOV audit (decisions on the ball):")
    print(f"  {'role':<5} {'calls':>6} {'old_seen':>10} {'cog_max':>7} "
          f"{'cog_mean':>8} {'%shrunk':>8}")
    total_calls = total_old = total_cog = total_shrunk = 0
    for role in sorted(STATS):
        s = STATS[role]
        if s["calls"] == 0:
            continue
        old_avg = statistics.mean(s["old_seen"])
        cog_avg = statistics.mean(s["cog_seen"])
        shr_pct = 100.0 * s["shrunk"] / s["calls"]
        total_calls += s["calls"]; total_old += sum(s["old_seen"])
        total_cog += sum(s["cog_seen"]); total_shrunk += s["shrunk"]
        print(f"  {role:<5} {s['calls']:>6} {old_avg:>7.2f} "
              f"{max(s['cog_seen']):>7} {cog_avg:>8.2f} {shr_pct:>7.1f}%")
    print(f"  {'ALL':<5} {total_calls:>6} {total_old/total_calls:>7.2f} "
          f"{'-':>7} {total_cog/total_calls:>8.2f} "
          f"{100.0*total_shrunk/total_calls:>7.1f}%")
    print(f"\n  Decisions where the OLD eyes had the single best forward "
          f"option and COGNITION dropped it: {DROPPED_BEST} "
          f"({100.0*DROPPED_BEST/total_calls:.1f}% of on-ball decisions)")


if __name__ == "__main__":
    main()