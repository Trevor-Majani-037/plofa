"""THE THREE FIXTURES â€” does each principle actually appear?

Each test is a small game: a controlled picture, a few seconds of the real
10 Hz integrator, and an assertion about where players ENDED UP. Nothing here
asserts that a function was called, because that is the mistake that let the
striker layer sit dead in `drift_minute` for months while its own unit tests
passed.

Every test drives `MatchEngine._offball_run`, the same entry point a live match
uses, so there is no second simulation path that could pass here and fail
there.

Fixture A is aimed at the two offside-line bugs found on 2026-09-29 (the line
was the DEEPEST defender, not the second-deepest; and the depth axis saturated
for a team attacking left) plus the dead-layer bug. One picture, three defects.
Fixture B is aimed at `Tuck In = 0`. Fixture C is aimed at the question the
session started with: does a striker settle into a position, or is he
permanently in transit?
"""
import pytest

from small_game import (
    PITCH_X, Scenario, depth_series, moved, play, width_series,
)

# A 4-2-3-1 shell for the team in possession, attacking RIGHT by default.
HOME = {
    "GK": (12.0, 34.0),
    "CB": [(38.0, 24.0), (38.0, 44.0)],
    "LB": (42.0, 8.0),
    "RB": (42.0, 60.0),
    "CDM": (46.0, 34.0),
    "CM": [(50.0, 22.0), (50.0, 46.0)],
    "CAM": (56.0, 40.0),
    "LW": (52.0, 10.0),
    "RW": (52.0, 58.0),
    "ST": (66.0, 34.0),
}


def _defenders(last=(85.0, 30.0), second=(79.0, 46.0), cover=(70.0, 34.0)):
    """A genuine two-man line plus a covering mid â€” so 'second deepest' and
    'deepest' are DIFFERENT defenders, which is the whole point of fixture A.

    STABILITY NOTE, learned by getting this wrong first: the defensive line has
    to be far enough from the ball that shape compaction does not dissolve it
    before the run resolves. The first version of this fixture placed the line
    at 70-80 with the ball at 45, i.e. 25 m away, and the backline travelled
    35 m up the pitch in six seconds â€” entirely correct engine behaviour, and
    it left the striker nothing to run behind. A scenario has to be a picture
    that can actually occur, or it tests nothing.
    """
    return {
        "GK": (97.0, 34.0),
        "CB": [second, last],
        "LB": (79.0, 62.0),
        "RB": (79.0, 6.0),
        "CDM": cover,
        "CM": [(68.0, 20.0), (68.0, 48.0)],
        "CAM": (60.0, 34.0),
        "LW": (58.0, 62.0),
        "RW": (58.0, 6.0),
        "ST": (50.0, 34.0),
    }


def _by_position(eng, team, position):
    return [p.name for p in eng.active_players[team] if p.position == position]


def _mirror(table):
    return {k: ([(PITCH_X - x, y) for x, y in v] if isinstance(v, list)
                else (PITCH_X - v[0], v[1]))
            for k, v in table.items()}


# â”€â”€ FIXTURE A â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The striker must get in BEHIND: beyond the second-deepest defender (the
# offside line, Law 11) and NOT beyond the deepest one (offside).

@pytest.mark.xfail(strict=True, reason=(
    "OPEN FINDING (2026-09-30), found by this harness. In a stable picture - "
    "ball in midfield 24 m from a two-man line at 79/85, striker at 66 - the "
    "striker moves BACKWARDS (66.0 -> 62.3) instead of getting in behind. Not "
    "a scenario artefact: an earlier version of this fixture DID dissolve the "
    "defensive line 35 m in 6 s because the line sat 25 m from the ball, and "
    "fixing that only changed the failure from -4.7 m to -3.7 m. Narrowed to "
    "either the 'behind' target being barely ahead of where he already stands, "
    "or STRIKER_RUN_BLEND (0.30) losing to the ball-side shape compaction, "
    "which sits the whole team at ~62 when the ball is at 55. NOT YET "
    "DIAGNOSED. strict=True so that fixing it turns the suite RED and forces "
    "this note to be deleted rather than left to rot."))
def test_striker_gets_in_behind_not_offside():
    sc = Scenario("run_in_behind",
                  "ball in midfield, striker 13 m behind the offside line",
                  home=HOME, away=_defenders(), ball=(55.0, 34.0),
                  possessing="home", seconds=6.0, attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    st_name = _by_position(eng, "Oxton", "ST")[0]
    d = depth_series(r["traces"][st_name], True)
    OFFSIDE_LINE = 79.0        # the second-deepest defender
    LAST_MAN = 85.0            # the deepest
    start = float(r["start"][st_name][0])

    assert d[-1] > start + 3.0, (
        f"striker did not move forward: {start:.1f} -> {d[-1]:.1f}")
    assert d.max() > OFFSIDE_LINE - 2.0, (
        f"never came near the offside line ({OFFSIDE_LINE}): max {d.max():.1f}")
    assert d.max() <= LAST_MAN + 1.5, (
        f"ran offside: reached {d.max():.1f}, last man at {LAST_MAN}")


def test_striker_runs_in_behind_attacking_LEFT():
    """The mirror image. The 2026-09-29 bug was `abs(x + 78)` saturating
    `last_line_gap` at 1.0 for a whole half, so in-behind was ALWAYS allowed
    when attacking left. The assertion that catches it is the mirror of the
    one above: the striker must NOT sail past the last man."""
    sc = Scenario("run_in_behind_mirrored",
                  "same picture, attacking LEFT",
                  home=_mirror(HOME), away=_mirror(_defenders()),
                  ball=(PITCH_X - 55.0, 34.0), possessing="home",
                  seconds=6.0, attacks_right=False)
    r = play(sc)
    eng = r["engine"]
    st_name = _by_position(eng, "Oxton", "ST")[0]
    d = depth_series(r["traces"][st_name], False)   # normalised to attack
    start, offside, last = float(r["start"][st_name][0]), 79.0, 85.0
    assert d[-1] > start + 3.0, f"no forward movement: {start:.1f} -> {d[-1]:.1f}"
    assert d.max() <= last + 1.5, (
        f"the left-attack saturation bug: reached {d.max():.1f}, "
        f"last man at {last}")


# â”€â”€ FIXTURE B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# An overlapping full-back must be able to TUCK â€” invert and come inside. The
# export reported `Runs Intended: Tuck In (FB) = 0`, which was a false finding
# caused by a synthetic fixture with no tuck instinct in the DNA. This one
# uses the real roster, so if tuck is genuinely unreachable it will show here.

def test_fullback_can_tuck_inside():
    home = dict(HOME)
    home["RB"] = (56.0, 62.0)          # high and wide, available to overlap
    home["RW"] = (60.0, 58.0)          # the winger he would underlap for
    sc = Scenario("tuck", "overlapping right-back, ball on the flank",
                  home=home, away=_defenders(), ball=(60.0, 58.0),
                  possessing="home", seconds=6.0, attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    rb = _by_position(eng, "Oxton", "RB")[0]
    y = width_series(r["traces"][rb])
    tuck = y.max() - y.min()
    assert tuck >= 2.0, (
        f"full-back never changed his width at all (range {tuck:.2f} m) â€” "
        f"the tuck/overlap/underlap layer is not reaching this picture")


# â”€â”€ FIXTURE C â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Does a striker SETTLE, or is he permanently in transit? The ball is held deep
# by a centre-back, so the striker has no reason to move. His depth should be
# quiet. The session's first reading of a full-match chart said he oscillated
# the length of the pitch â€” which turned out to be an un-normalised half-time
# mirror. This asserts the quiet version, so the claim is pinned either way.

def test_striker_settles_when_there_is_nothing_to_run_onto():
    home = dict(HOME)
    home["ST"] = (62.0, 34.0)
    home["CB"] = [(26.0, 24.0), (26.0, 44.0)]
    sc = Scenario("settle", "ball held deep by the centre-backs",
                  home=home, away=_defenders(last=(84.0, 30.0),
                                             second=(76.0, 46.0),
                                             cover=(70.0, 34.0)),
                  ball=(26.0, 34.0), possessing="home", seconds=15.0,
                  attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    st_name = _by_position(eng, "Oxton", "ST")[0]
    d = depth_series(r["traces"][st_name], True)
    span = float(d.max() - d.min())
    assert span < 25.0, (
        f"the striker moved {span:.1f} m of depth in 15 s with the ball 36 m "
        f"behind him â€” that is transit, not a settled line")


# â”€â”€ THE HARNESS ITSELF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# A scenario runner that cannot move anyone would pass every "he did not do the
# wrong thing" assertion above, so its own liveness is asserted.

def test_harness_moves_real_players():
    sc = Scenario("liveness", "everyone in a shell, ball on the flank",
                  home=HOME, away=_defenders(), ball=(52.0, 12.0),
                  possessing="home", seconds=6.0, attacks_right=True)
    r = play(sc)
    total = sum(moved(t) for t in r["traces"].values() if t)
    assert total > 100.0, f"only {total:.0f} m covered by the whole off-ball XI"
    assert len(r["start"]) >= 18, f"only {len(r['start'])} players placed"


def test_harness_places_the_picture_it_was_given():
    sc = Scenario("placement", "ST placed at 66,34",
                  home=HOME, away=_defenders(), ball=(55.0, 34.0),
                  possessing="home", seconds=0.1, attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    st = _by_position(eng, "Oxton", "ST")[0]
    x0, y0 = r["start"][st]
    assert abs(x0 - 66.0) < 0.01 and abs(y0 - 34.0) < 0.01, (
        f"scenario placement ignored: ST started at ({x0:.1f},{y0:.1f})")


def test_two_players_can_share_a_position_slot():
    """Fixture A is meaningless without this: the offside line is defined by
    the SECOND deepest of several defenders."""
    sc = Scenario("two_cbs", "two distinct centre-back positions",
                  home=HOME, away=_defenders(), ball=(45.0, 34.0),
                  possessing="home", seconds=0.1, attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    cbs = _by_position(eng, "Natrican", "CB")
    assert len(cbs) == 2
    xs = sorted(r["start"][c][0] for c in cbs)
    assert xs[1] - xs[0] > 5.0, (
        f"both centre-backs were placed at the same spot: {xs}")

