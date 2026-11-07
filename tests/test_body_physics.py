"""Body physics gates (Fixes 1-8) + block shape gates (Fix 5).

Fix 1: record_touch sets a sprint target, never teleports the body.
Fix 2: every instant write goes through pace_limited_move with telemetry.
Fix 3: kickoff reset migrates as pending sprints (celebration jog).
Fix 4: home anchors chase authored stance targets, they do not teleport.
Fix 5: low/mid/high block presets produce their documented line depths.
Fix 6: a resting ball always has a man on it (see the match-level diag).
Fix 7: two bodies never share one coordinate.
Fix 8: an on-ball man with a pending target still moves during live play.

These are fast unit tests — they arrange position_engine state directly,
never simulate a match.
"""
import math

from match_engine import Intensity, MatchState, TeamProfile, TeamStyle
from player_dna import SquadBuilder
from position_engine import PositionEngine

# Named exactly like tests/_make_squad in test_defensive_awareness.py:
# f"{team_name[:3]} {pos} {i}" with _ROLES order:
# 0 GK, 1 CB, 2 CB, 3 LB, 4 RB, 5 CDM, 6 CM, 7 CM, 8 CAM, 9 LW, 10 ST
ROLES = [
    ("GK", ["sweeper_keeper"]),
    ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]),
    ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]),
    ("CDM", ["anchor_man"]),
    ("CM", ["engine"]),
    ("CM", ["box_box"]),
    ("CAM", ["creator"]),
    ("LW", ["winger"]),
    ("ST", ["fox_in_box"]),
]


def _squad(team="T FC"):
    players = [(f"T FC {pos} {i}", pos, spec, 26) for i, (pos, spec) in enumerate(ROLES)]
    return SquadBuilder.build(team_name=team, starters=players)["starters"]


def _engine(team="T FC"):
    prof = TeamProfile(name=team, style=TeamStyle.BALANCED,
                       playing_style=None, intensity=Intensity.MEDIUM)
    pe = PositionEngine()
    pe.initialize_team(team, _squad(team), prof, attacks_right=True)
    return pe


def _state(pe, name):
    return pe.states[name]


# ── Fix 1 ───────────────────────────────────────────────────────────────
def test_record_touch_does_not_move_body():
    pe = _engine()
    st = _state(pe, "T FC CDM 5")
    x0, y0 = st.current_x, st.current_y
    pe.record_touch("T FC CDM 5", x0 + 40.0, y0, minute=5)
    assert abs(st.current_x - x0) < 1e-9 and abs(st.current_y - y0) < 1e-9, \
        "touch must not teleport the body"
    assert st.pending_x is not None and abs(st.pending_x - (x0 + 40.0)) < 1e-6
    assert st.pending_ticks == 0


def test_wall_place_is_the_documented_exception():
    pe = _engine()
    st = _state(pe, "T FC CDM 5")
    x0 = st.current_x
    pe.set_piece_place("T FC CDM 5", x0 + 10.0, st.current_y, minute=5)
    assert abs(st.current_x - (x0 + 10.0)) < 1e-6, "wall is placed exactly"


# ── Fix 2 ───────────────────────────────────────────────────────────────
def test_pace_limited_move_caps_speed_and_counts():
    pe = _engine()
    st = _state(pe, "T FC CDM 5")
    x0 = st.current_x
    moved = pe.pace_limited_move(st, x0 + 100.0, st.current_y,
                                 max_mps=10.0, dt=0.1, site="unit")
    assert moved <= 1.0 + 1e-9, "0.1 s at 10 m/s is 1 m"
    rep = pe.teleport_report()
    assert rep["events"] >= 1 and rep["suppressed_m"] > 90


def test_teleport_to_is_counted_test_only():
    pe = _engine()
    st = _state(pe, "T FC CDM 5")
    st.teleport_to(st.current_x + 5.0, st.current_y, minute=1)
    assert st.pending_x is None


# ── Fix 4 ───────────────────────────────────────────────────────────────
def test_home_anchor_chases_stance_target():
    pe = _engine()
    st = _state(pe, "T FC CB 1")
    h0 = st.home_x
    pe._stance_deltas["T FC"] = {"CB": (14.0, 0.0)}
    pe._recompute_homes("T FC")
    assert h0 < st.home_x <= h0 + 8.0, "home must approach, not jump"
    assert st.home_target_x is not None and st.home_target_x >= st.home_x
    # Repeated calls keep converging without teleporting.
    pe._recompute_homes("T FC")
    assert st.home_x <= h0 + 16.0


# ── Fix 7: no two bodies on one coordinate ──────────────────────────────
def test_two_bodies_are_never_aimed_at_one_coordinate():
    pe = _engine()
    a, b = pe.states["T FC CB 1"], pe.states["T FC CB 2"]
    a.current_x, a.current_y = 50.5, 34.0
    b.current_x, b.current_y = 50.0, 34.0
    # Two men handed the identical target — what a duel does, because the
    # event carries one location for the winner and the loser.
    ta = pe.separated_target("T FC", a.player_name, 50.25, 34.0)
    tb = pe.separated_target("T FC", b.player_name, 50.25, 34.0)
    assert math.hypot(ta[0] - b.current_x, ta[1] - b.current_y) >= \
        pe.MIN_BODY_GAP - 1e-6
    assert math.hypot(tb[0] - a.current_x, tb[1] - a.current_y) >= \
        pe.MIN_BODY_GAP - 1e-6
    assert ta[0] > tb[0], "the pair must split apart, not stack"
    # The correction is bounded, so it can never become a relocation.
    assert math.hypot(ta[0] - 50.25, ta[1] - 34.0) <= pe.BODY_PUSH_MAX + 1e-9
    assert math.hypot(tb[0] - 50.25, tb[1] - 34.0) <= pe.BODY_PUSH_MAX + 1e-9


def test_separation_is_deterministic_and_clears_the_body():
    pe = _engine()
    mate = pe.states["T FC CB 2"]
    mate.current_x, mate.current_y = 50.0, 34.0
    t1 = pe.separated_target("T FC", "T FC CB 1", 50.0, 34.0)
    t2 = pe.separated_target("T FC", "T FC CB 1", 50.0, 34.0)
    assert t1 == t2, "a coincident target must not flip between ticks"
    assert math.hypot(t1[0] - mate.current_x, t1[1] - mate.current_y) >= \
        pe.MIN_BODY_GAP - 1e-6


def test_position_invariant_never_leaves_teammates_on_one_coordinate():
    pe = _engine()
    a, b = pe.states["T FC CB 1"], pe.states["T FC CB 2"]
    a.current_x, a.current_y = 50.0, 34.0
    b.current_x, b.current_y = 50.0, 34.0
    # A target-level correction is not enough: later constraints can hand two
    # men the same post, so the invariant is enforced on the coordinates.
    first = pe.separate_bodies("T FC")
    assert first > 0.0
    gap = math.hypot(a.current_x - b.current_x, a.current_y - b.current_y)
    assert gap > 0.0, "exactly coincident teammates must split"
    # The correction is a sidestep, never a relocation.
    assert first <= 2.0 * pe.BODY_SEP_STEP + 1e-9
    # Repeated ticks converge on the invariant instead of oscillating.
    for _ in range(40):
        pe.separate_bodies("T FC")
    gap = math.hypot(a.current_x - b.current_x, a.current_y - b.current_y)
    assert gap >= pe.MIN_BODY_GAP - 1e-6
    # Settled pair: no further correction is booked.
    assert pe.separate_bodies("T FC") == 0.0


def test_position_invariant_leaves_opponents_alone():
    pe = _engine()
    a = pe.states["T FC CB 1"]
    b = pe.states["T FC CB 2"]
    a.current_x, a.current_y = 50.0, 34.0
    b.current_x, b.current_y = 50.0, 34.0
    # Opponents contesting the same blade of grass is football (a duel, a
    # marker). The invariant is about teammates only.
    pe.separate_bodies("Other FC")
    assert (a.current_x, a.current_y) == (50.0, 34.0)
    assert (b.current_x, b.current_y) == (50.0, 34.0)


# ── Fix 5 ───────────────────────────────────────────────────────────────
def test_mid_block_presets_match_documented_depths():
    pe = _engine()
    # Home attacks right -> own goal at x=0, depth = x.
    checks = {
        "low": (19.0, 31.0, 37.0),
        "mid": (30.0, 42.0, 48.0),
        "high": (42.0, 54.0, 60.0),
    }
    ball_x, ball_y = 30.0, 34.0  # wide enough to avoid the central squeeze
    res = {}
    for preset, _ in checks.items():
        # CBs / mids / front sample via the public target API
        for pos_player in ("T FC CB 1", "T FC CDM 5", "T FC ST 10"):
            out = pe.defensive_block_target(
                pos_player, ball_x, ball_y, attacks_right=True, preset=preset)
            assert out is not None
            alpha, tx, ty = out
            res.setdefault(preset, {})[pos_player] = tx
    # back four ~19/30/42, midfield ~31/42/54, front ~37/48/60 (ball shifted
    # sideways by BLOCK_SHIFT, depth is unshifted).
    assert abs(res["low"]["T FC CB 1"] - 19.0) < 3.0
    assert abs(res["mid"]["T FC CB 1"] - 30.0) < 3.0
    assert abs(res["high"]["T FC CB 1"] - 42.0) < 3.0
    assert abs(res["low"]["T FC ST 10"] - 37.0) < 3.0
    assert abs(res["mid"]["T FC ST 10"] - 48.0) < 3.0
    assert abs(res["high"]["T FC ST 10"] - 60.0) < 3.0


# ── Fix 11: a synchronous run-in writes its own trajectory ──────────────
def test_setpiece_run_in_is_recorded_as_a_path_not_a_snap():
    import random
    random.seed(7)
    pe = _engine()
    st = pe.states["T FC ST 10"]
    st.current_x, st.current_y = 40.0, 34.0
    got = []
    pe.tick_recorder = lambda n, t, x, y: got.append((n, t, x, y))
    pe.advance_to_slots({"T FC ST 10": (60.0, 34.0)}, 2.0, clock_start=100.0)
    assert len(got) >= 15, f"expected a sampled run-in, got {len(got)} samples"
    times = [g[1] for g in got]
    assert times == sorted(times), "run-in samples must be time ordered"
    assert times[0] > 100.0
    steps = [math.hypot(b[2] - a[2], b[3] - a[3])
             for a, b in zip(got, got[1:])]
    # No single recorded step may exceed a sprint of 7.5 m/s at 0.1 s.
    assert max(steps) <= 0.75 + 1e-9, f"max step {max(steps):.2f} m"
    # Enough of the 20 m travelled is drawn that it cannot read as a teleport.
    drawn = sum(steps)
    # He eases in and settles short of the mark, so the drawn path is a large
    # fraction of the ~11 m a 2 s sprint covers -- not a single 20 m snap.
    assert drawn >= 7.0, f"only {drawn:.1f} m of the run-in was drawn"


# ── Fix 9: the live window reads the ball where it actually is ──────────
def test_track_interpolation_follows_the_ball_between_samples():
    from match_engine import MatchEngine
    track = [
        (100.0, 40.0, 34.0, "carry"),
        (101.0, 42.0, 34.0, "carry"),
        (102.0, 42.0, 36.0, "pass"),
    ]
    # Exactly on a sample.
    assert MatchEngine._track_at(track, 101.0)[:2] == (42.0, 34.0)
    # Half way along the first leg -- the old code reported the END (42, 34)
    # for the whole window, which is what drew the carrier trailing 2 m.
    x, y, kind = MatchEngine._track_at(track, 100.5)
    assert abs(x - 41.0) < 1e-9 and abs(y - 34.0) < 1e-9
    assert kind == "carry"
    # Second leg is a pass.
    x, y, kind = MatchEngine._track_at(track, 101.75)
    assert abs(x - 42.0) < 1e-9 and abs(y - 35.5) < 1e-9
    assert kind == "pass"
    # Clamped at both ends so a rounded sample can never read off the path.
    assert MatchEngine._track_at(track, 0.0)[:2] == (40.0, 34.0)
    assert MatchEngine._track_at(track, 1e6)[:2] == (42.0, 36.0)
    assert MatchEngine._track_at([], 5.0) is None
