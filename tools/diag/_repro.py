"""Quick repro harness for red-card + away-win investigations."""
import random
from match_engine import (MatchEngine, MatchConfig, TeamProfile, TeamStyle,
                           PlayingStyle, Intensity)
from player_dna import SquadBuilder

HOME_TEAM = "Hartwell City"
AWAY_TEAM = "Thornfield United"

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback", "engine"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender", "strong"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man", "interceptor"], 28),
    ("Luca Ferrini", "CM", ["box_box", "engine"], 26),
    ("Kofi Mensah", "CAM", ["creator", "sup_vision"], 24),
    ("Adri Vela", "LW", ["dribbler", "speedster"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher", "aerial_threat"], 29),
    ("Percy", "RW", ["grand_dribbler", "inverted", "clinical_finisher", "speedster"], 24),
]
HOME_SUBS = [
    ("Joel Trent", "GK", [], 26),
    ("Sam Boateng", "CB", ["stopper_defender"], 28),
    ("Oscar Muñoz", "CM", ["playmaker", "press_resistant"], 25),
    ("Eli Dago", "LW", ["speedster", "dribbler"], 21, 68),
    ("Calvin Pryce", "ST", ["poacher", "fox_in_box"], 27, 75),
]
HOME_SUPERSTARS = ["Percy", "Dragan Novak"]
HOME_SP_TAKERS = ["Percy", "Kofi Mensah"]

AWAY_STARTERS = [
    ("Pavel Renko", "GK", ["sweeper_keeper"], 31),
    ("Jide Afolabi", "LB", [], 26),
    ("Bart Kuipers", "CB", ["stopper_defender"], 28),
    ("Ciro Mancini", "CB", ["ball_playing_cb"], 26),
    ("Lee Sung-jin", "RB", ["overlapping_fullback"], 28),
    ("Demi Adeola", "CDM", ["ball_winner", "regista"], 27),
    ("Finn Larsson", "CM", ["press_resistant", "engine"], 25),
    ("Kwame Asante", "CAM", ["playmaker", "creator"], 23),
    ("Bruno Reis", "LW", ["speedster", "counter_attacker"], 24),
    ("Nico Strauss", "ST", ["fox_in_box", "cold_blooded"], 27),
    ("Tariq El-Amin", "RW", ["dribbler", "inverted"], 22),
]
AWAY_SUBS = [
    ("Martin Horak", "GK", [], 29),
    ("Danny Cross", "CB", [], 24),
    ("Isaac Bello", "CM", ["box_box"], 23, 72),
    ("Rafiq Nasser", "ST", ["poacher", "super_sub"], 26, 80),
    ("Olu Adeyemi", "RW", ["speedster"], 21, 85),
]
AWAY_SUPERSTARS = ["Kwame Asante"]
AWAY_SP_TAKERS = ["Kwame Asante", "Bruno Reis"]


def run_one(strictness, seed, swap=False, balanced=False):
    random.seed(seed)
    home_squad = SquadBuilder.build(team_name=HOME_TEAM, starters=HOME_STARTERS,
                                    substitutes=HOME_SUBS, team_superstars=HOME_SUPERSTARS,
                                    set_piece_takers=HOME_SP_TAKERS)
    away_squad = SquadBuilder.build(team_name=AWAY_TEAM, starters=AWAY_STARTERS,
                                    substitutes=AWAY_SUBS, team_superstars=AWAY_SUPERSTARS,
                                    set_piece_takers=AWAY_SP_TAKERS)
    hpro = TeamProfile(name=HOME_TEAM, style=TeamStyle.ATTACKING,
                       playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    apro = TeamProfile(name=AWAY_TEAM, style=TeamStyle.FLUID_COUNTER,
                       playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM)
    if balanced:
        hpro = TeamProfile(name=HOME_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
        apro = TeamProfile(name=AWAY_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    home_name, away_name = (AWAY_TEAM, HOME_TEAM) if swap else (HOME_TEAM, AWAY_TEAM)
    home_sq, away_sq = (away_squad, home_squad) if swap else (home_squad, away_squad)
    home_pr, away_pr = (apro, hpro) if swap else (hpro, apro)
    config = MatchConfig(home_team=home_name, away_team=away_name, matchday=1,
                         referee="Test", referee_strictness=strictness)
    eng = MatchEngine(config, home_pr, away_pr)
    eng.quiet = True
    eng.set_squad(home_name, home_sq["starters"], home_sq["substitutes"])
    eng.set_squad(away_name, away_sq["starters"], away_sq["substitutes"])
    res = eng.simulate()
    st = res.state if hasattr(res, "state") else res
    # Return from perspective of the nominal HOME_TEAM/AWAY_TEAM for consistency
    if swap:
        return (st.away_goals, st.home_goals, st.away_red_cards, st.home_red_cards)
    return (st.home_goals, st.away_goals, st.home_red_cards, st.away_red_cards)


def run_one_instr(strictness, seed, swap=False, balanced=False):
    """Like run_one but also returns foul/yellow/red event counts."""
    import event_chain as EC
    hg, ag, hrc, arc = run_one(strictness, seed, swap=swap, balanced=balanced)
    # re-run to capture timeline (cheap relative to already-slow sim)
    random.seed(seed)
    home_squad = SquadBuilder.build(team_name=HOME_TEAM, starters=HOME_STARTERS,
                                    substitutes=HOME_SUBS, team_superstars=HOME_SUPERSTARS,
                                    set_piece_takers=HOME_SP_TAKERS)
    away_squad = SquadBuilder.build(team_name=AWAY_TEAM, starters=AWAY_STARTERS,
                                    substitutes=AWAY_SUBS, team_superstars=AWAY_SUPERSTARS,
                                    set_piece_takers=AWAY_SP_TAKERS)
    hpro = TeamProfile(name=HOME_TEAM, style=TeamStyle.ATTACKING,
                       playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    apro = TeamProfile(name=AWAY_TEAM, style=TeamStyle.FLUID_COUNTER,
                       playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM)
    if balanced:
        hpro = TeamProfile(name=HOME_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
        apro = TeamProfile(name=AWAY_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    home_name, away_name = (AWAY_TEAM, HOME_TEAM) if swap else (HOME_TEAM, AWAY_TEAM)
    home_sq, away_sq = (away_squad, home_squad) if swap else (home_squad, away_squad)
    home_pr, away_pr = (apro, hpro) if swap else (hpro, apro)
    config = MatchConfig(home_team=home_name, away_team=away_name, matchday=1,
                         referee="Test", referee_strictness=strictness)
    eng = MatchEngine(config, home_pr, away_pr)
    eng.quiet = True
    eng.set_squad(home_name, home_sq["starters"], home_sq["substitutes"])
    eng.set_squad(away_name, away_sq["starters"], away_sq["substitutes"])
    res = eng.simulate()
    fouls = sum(1 for e in res.timeline if e.event_type == EC.EventType.FOUL_COMMITTED)
    yellows = sum(1 for e in res.timeline if e.event_type == EC.EventType.YELLOW_CARD)
    reds = sum(1 for e in res.timeline if e.event_type == EC.EventType.RED_CARD)
    return (hg, ag, hrc, arc, fouls, yellows, reds)


def main():
    import sys
    strict = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hwin = awin = draw = 0
    reds = 0
    hr = ar = 0
    for i in range(n):
        hg, ag, hrc, arc = run_one(strict, 1000 + i)
        if hg > ag: hwin += 1
        elif ag > hg: awin += 1
        else: draw += 1
        if hrc or arc: reds += 1
        hr += hrc; ar += arc
        print(f"  match {i}: {hg}-{ag} red h={hrc} a={arc}")
    print(f"strictness={strict}: home={hwin} away={awin} draw={draw} "
          f"matches_with_red={reds}/{n} total_reds={hr+ar} (home {hr}, away {ar})")


if __name__ == "__main__":
    import sys
    strict = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    swap = "--swap" in sys.argv
    balanced = "--balanced" in sys.argv
    tag = ("SWAP " if swap else "") + ("BALANCED " if balanced else "")
    print(f"=== {tag}===")
    hwin = awin = draw = 0
    reds = 0
    hr = ar = 0
    for i in range(n):
        hg, ag, hrc, arc = run_one(strict, 1000 + i, swap=swap, balanced=balanced)
        if hg > ag: hwin += 1
        elif ag > hg: awin += 1
        else: draw += 1
        if hrc or arc: reds += 1
        hr += hrc; ar += arc
        print(f"  match {i}: {hg}-{ag} red h={hrc} a={arc}")
    print(f"strictness={strict}: home={hwin} away={awin} draw={draw} "
          f"matches_with_red={reds}/{n} total_reds={hr+ar} (home {hr}, away {ar})")
