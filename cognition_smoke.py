"""Live-match smoke test for the cognition layer (the TOLAND merge).

Runs a REAL full match twice with the SAME seed, same squads, same brains:

    Run A — vanilla:        set_cognition(False)   (the pre-merge engine)
    Run B — cognition on:   set_cognition(True)    + PlayerMinds for every
                              registered player + the memory observer fed by
                              match events.

Because the brains are identical and only the cognition toggle differs, any
divergence in the match sequence is the merge working: FOV gating removes
options no human would take, memory fills from real match events, and
temperament reshapes which intent gets sampled.  The script also prints the
episodic-memory state accumulated by each team's minds.

Run:  .\\.venv\\Scripts\\python.exe cognition_smoke.py
"""

import random

import run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from brain_integration import set_cognition
from cognition_brain import engage_mind_observer, disengage_mind_observer
from cognition.mind import register_mind, get_mind, new_mind, clear_minds


def _build_squad(team, starters, subs, superstars, sp_takers):
    squad = SquadBuilder.build(
        team_name=team, starters=starters, substitutes=subs,
        team_superstars=superstars, set_piece_takers=sp_takers,
    )
    for p in squad["starters"] + squad["substitutes"]:
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]
    return squad


def _config():
    return MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, match_date=R.MATCH_DATE,
        matchday=R.MATCHDAY, season=R.SEASON, competition=R.COMPETITION,
        venue=R.VENUE, stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )


def _new_engine(home_squad, away_squad):
    eng = MatchEngine(_config(), R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    return eng


def _register_minds(home_squad, away_squad, radius=25.0):
    players = (home_squad["starters"] + home_squad["substitutes"]
               + away_squad["starters"] + away_squad["substitutes"])
    for p in players:
        register_mind(p.name, new_mind(view_radius=radius))


def _memory_report():
    lines = []
    for name, mind in sorted(get_mind_names()):
        tags = mind.memory.recall_all()
        if tags:
            summary = ", ".join(f"{t}={w:.2f}" for t, w in sorted(tags.items()))
            lines.append(f"    {name}: {summary}")
    return lines


def get_mind_names():
    import cognition.mind as cm
    return list(cm._minds.items())


def _run(seed, cognition_on, home_squad, away_squad, log=print):
    random.seed(seed)
    set_cognition(cognition_on)
    if cognition_on:
        engage_mind_observer()
    else:
        disengage_mind_observer()
    eng = _new_engine(home_squad, away_squad)
    res = eng.simulate()
    log(
        f"  [{cognition_on and 'COGNITION' or 'VANILLA '}] "
        f"{R.HOME_TEAM} {eng.state.home_goals} - {res.away_goals} {R.AWAY_TEAM}"
        f"  | possession {res.home_possession_pct:.0f}/{100 - res.home_possession_pct:.0f}"
        f"  | events {len(eng.timeline)}"
    )
    return eng


def main():
    home_squad = _build_squad(
        R.HOME_TEAM, R.HOME_STARTERS, R.HOME_SUBS,
        R.HOME_SUPERSTARS, R.HOME_SP_TAKERS,
    )
    away_squad = _build_squad(
        R.AWAY_TEAM, R.AWAY_STARTERS, R.AWAY_SUBS,
        R.AWAY_SUPERSTARS, R.AWAY_SP_TAKERS,
    )

    print("=" * 72)
    print("Run A — vanilla (cognition off):")
    clear_minds()
    set_cognition(False)
    eng_a = _run(seed=20260915, cognition_on=False,
                 home_squad=home_squad, away_squad=away_squad)

    print("=" * 72)
    print("Run B — cognition on (FOV gate + temperament + memory):")
    clear_minds()
    _register_minds(home_squad, away_squad, radius=25.0)
    eng_b = _run(seed=20260915, cognition_on=True,
                 home_squad=home_squad, away_squad=away_squad)

    print("=" * 72)
    print("Cognition memory after the match (player: tag=emotional weight):")
    lines = _memory_report()
    if lines:
        print("\n".join(lines[:24]))
        if len(lines) > 24:
            print(f"    ... and {len(lines) - 24} more players remembering something.")
    else:
        print("    (no players accumulated episodic memory this match)")

    print("=" * 72)
    if eng_a.state.home_goals != eng_b.state.home_goals or \
       eng_a.state.away_goals != eng_b.state.away_goals:
        print("DIVERGED:  same seed, same brains, different scoreline -> the minds changed the match.")
    else:
        print("Same scoreline this time — but the memory stream above still proves "
              "the minds observed the match.  Re-run or tweak radius to see divergence.")
    set_cognition(False)
    disengage_mind_observer()


if __name__ == "__main__":
    main()