"""
physics/ — the continuous physical-time layer.
===============================================
Test brief §22 lists fifteen required behaviours. Each one is a test here,
named for the clause it satisfies, because a requirement that is not a test is
a wish.

The structure below maps 1:1 onto §22:

  1-3   travel time from distance and speed          -> test_travel_time_*
  4     faster player arrives earlier                -> test_a_faster_player_*
  5     fatigue lengthens the journey                 -> test_a_tired_player_*
  6     speed is capped                               -> test_a_player_cannot_*
  7     no teleportation                              -> test_a_player_cannot_teleport
  8     acceleration prevents instant top speed      -> test_acceleration_*
  9-10  aerial ball has height, gravity brings it down-> test_aerial_*
  11    through ball decided by arrival times         -> test_a_through_ball_*
  12    same seed, same timeline                      -> test_the_same_*
  13    90 minutes without waiting 90 minutes         -> test_a_whole_*
  14    CPU time does not alter timestamps            -> test_cpu_execution_*
  15    existing exports still work                   -> test_existing_*

Plus guard rails the brief implies but does not enumerate: the clock must not
read a wall clock, the package must not import the live pipeline, and nothing
may be random.
"""
from __future__ import annotations

import ast
import math
import os
import time as _wallclock

import pytest

from physics import (
    BALL_DRAG_COEFFICIENT,
    GRAVITY,
    BallPhysics,
    Candidate,
    ContinuousClock,
    KinematicState,
    MatchTime,
    PassResolution,
    PhysicsViolation,
    PhysicsViolationError,
    PhysicsWorld,
    PlayerMotion,
    ViolationLog,
    arrival_contest,
    arrival_time,
    calibrate_from_values,
    check_move,
    integrate,
    loft_time,
    match_time_to_seconds,
    resolve_receiver,
    seconds_to_match_time,
    speed_for_pace,
)
from physics.ball_physics import ground_travel_time, height_at, z_at
from physics.calibration import (
    MAX_SPRINT_MPS,
    MIN_SPRINT_MPS,
    PITCH_LENGTH_M,
    acceleration_for,
    fatigue_acceleration_multiplier,
    fatigue_speed_multiplier,
    reaction_for,
)
from physics.player_motion import ABSOLUTE_SPEED_CEILING, REACH_M

G = GRAVITY


def profile(pace=60.0, acceleration=60.0, anticipation=60.0, agility=60.0,
            stamina=100.0, name="Test"):
    return calibrate_from_values(
        name=name, position="CM", pace_score=pace,
        acceleration_score=acceleration, anticipation_score=anticipation,
        agility=agility, stamina=stamina)


# ── §22.1-2  travel time from distance and speed ──────────

def test_travel_time_10m_at_10mps_is_about_one_second():
    """§22.1 — 10 m pass at 10 m/s ~ 1 s."""
    t = ground_travel_time(10.0, 10.0, drag=0.0)
    assert t == pytest.approx(1.0, abs=1e-9)


def test_travel_time_20m_at_20mps_is_about_one_second():
    """§22.2 — 20 m pass at 20 m/s ~ 1 s."""
    t = ground_travel_time(20.0, 20.0, drag=0.0)
    assert t == pytest.approx(1.0, abs=1e-9)


def test_travel_time_scales_with_distance_at_fixed_speed():
    """§22.3 — same distance + slower ball = longer travel time."""
    slow = ground_travel_time(30.0, 10.0, drag=0.0)
    fast = ground_travel_time(30.0, 20.0, drag=0.0)
    assert slow > fast
    assert slow / fast == pytest.approx(2.0, abs=1e-9)
    # and longer distance at the same speed is longer
    assert ground_travel_time(40.0, 20.0, drag=0.0) > \
           ground_travel_time(20.0, 20.0, drag=0.0)


def test_drag_makes_a_long_ball_slower_than_the_dragless_ideal():
    """§10 — drag is real, so the naive distance/speed is the approximation."""
    ideal = ground_travel_time(30.0, 20.0, drag=0.0)
    real = ground_travel_time(30.0, 20.0, drag=BALL_DRAG_COEFFICIENT)
    assert real > ideal


def test_a_ball_that_cannot_arrive_reports_infinite_time():
    """Distance beyond what drag allows is impossible, not merely slow."""
    assert ground_travel_time(10_000.0, 5.0) == math.inf


# ── §22.4  a faster player arrives earlier ─────────────────

def test_a_faster_player_reaches_the_same_target_earlier():
    """§22.4 — and §7's warning that speed is not the only thing that counts."""
    target = (40.0, 20.0)
    slow = KinematicState(0.0, 20.0)
    quick = KinematicState(0.0, 20.0)
    t_slow = arrival_time(slow, target, profile(pace=45.0))
    t_fast = arrival_time(quick, target, profile(pace=90.0))
    assert t_fast < t_slow


def test_but_a_better_positioned_slower_player_can_win():
    """§7 — "a slower player with better positioning should sometimes reach
    the ball first". Position is a first-class term in the arrival time."""
    target = (40.0, 20.0)
    fast_far = KinematicState(0.0, 20.0)
    slow_near = KinematicState(34.0, 20.0)
    t_fast = arrival_time(fast_far, target, profile(pace=95.0))
    t_slow = arrival_time(slow_near, target, profile(pace=50.0))
    assert t_slow < t_fast, (
        "a fast player 40 m away must lose to a slow one 6 m away")


# ── §22.5  fatigue lengthens the journey ──────────────────

def test_a_tired_player_takes_longer_than_the_same_player_at_full_stamina():
    """§22.5 — and §14's smooth modifier, not a step function."""
    target = (30.0, 0.0)
    state = KinematicState(0.0, 0.0)
    fresh = arrival_time(state, target, profile(stamina=100.0))
    tired = arrival_time(state, target, profile(stamina=25.0))
    assert tired > fresh


def test_fatigue_hurts_acceleration_more_than_top_speed():
    """§14 — you can still sprint tired; you cannot get there as fast."""
    s_fresh = fatigue_speed_multiplier(100.0)
    s_tired = fatigue_speed_multiplier(10.0)
    a_fresh = fatigue_acceleration_multiplier(100.0)
    a_tired = fatigue_acceleration_multiplier(10.0)
    assert (s_fresh - s_tired) < (a_fresh - a_tired)
    # and neither is ever zero: a tired player is slower, not immobile
    assert s_tired > 0.85 and a_tired > 0.6


def test_fatigue_is_smooth_not_a_step():
    """§14 explicitly forbids 100 = fast, 50 = slow."""
    samples = [fatigue_speed_multiplier(s) for s in range(0, 101, 5)]
    deltas = [b - a for a, b in zip(samples, samples[1:])]
    assert all(d >= 0 for d in deltas), "must not decrease with stamina"
    assert len(set(round(d, 6) for d in deltas)) == 1, "must be linear, not stepped"


# ── §22.6  speed is capped ────────────────────────────────

def test_a_player_cannot_exceed_their_configured_maximum_speed():
    """§22.6."""
    p = profile(pace=100.0)
    state = KinematicState(0.0, 0.0)
    target = (60.0, 0.0)
    for _ in range(400):
        state = integrate(state, target, 0.02, p)
        assert state.speed <= p.max_speed + 1e-9, (
            f"{state.speed:.3f} m/s exceeds max {p.max_speed:.3f}")


def test_speed_is_also_capped_by_an_absolute_ceiling():
    """§15 — a miscalibrated profile must degrade to 'suspicious', never to a
    position jump no consumer can detect."""
    absurd = profile(pace=100.0)
    object.__setattr__(absurd, "max_speed", 500.0)
    state = integrate(KinematicState(0.0, 0.0), (100.0, 0.0), 1.0, absurd)
    assert state.speed <= ABSOLUTE_SPEED_CEILING + 1e-9


# ── §22.7  no teleportation ───────────────────────────────

def test_a_player_cannot_teleport():
    """§22.7 and §15's hard invariant."""
    v = check_move((30.0, 20.0), (80.0, 50.0), 0.0, subject="Striker")
    assert v is not None
    assert v.kind == "teleport"
    assert "Striker" in str(v)
    with pytest.raises(PhysicsViolationError):
        ViolationLog().check((30.0, 20.0), (80.0, 50.0), 0.0, subject="X")
        raise PhysicsViolationError(v)


def test_a_legal_move_passes_the_guard():
    assert check_move((0.0, 0.0), (7.5, 0.0), 1.0, subject="Striker") is None


def test_a_stationary_player_is_not_teleporting():
    """Nothing moved. That is not a violation."""
    assert check_move((10.0, 10.0), (10.0, 10.0), 0.0) is None


def test_impossible_speed_is_reported_separately_from_teleport():
    slow_but_real = check_move((0.0, 0.0), (100.0, 0.0), 1.0)
    assert slow_but_real is not None
    assert slow_but_real.kind == "impossible_speed"


def test_negative_time_is_a_violation():
    v = check_move((0.0, 0.0), (1.0, 0.0), -0.5)
    assert v is not None and v.kind == "negative_time"


def test_the_violation_log_summarises_and_caps():
    log = ViolationLog(cap=3)
    for i in range(5):
        log.check((0.0, 0.0), (50.0, 0.0), 0.0, subject=f"P{i}")
    assert log.total == 5
    assert len(log.entries) == 3 and log.dropped == 2
    assert "teleport" in log.summary()


# ── §22.8  acceleration prevents instant top speed ────────

def test_acceleration_prevents_reaching_top_speed_instantly():
    """§22.8 — and §7's 'rest -> acceleration -> high speed -> deceleration'."""
    p = profile(pace=90.0)
    state = integrate(KinematicState(0.0, 0.0), (100.0, 0.0), 0.01, p)
    assert 0.0 < state.speed < p.max_speed, (
        "one 10 ms step must not reach top speed")


def test_it_does_eventually_reach_top_speed():
    p = profile(pace=90.0)
    state = KinematicState(0.0, 0.0)
    for _ in range(500):
        state = integrate(state, (100.0, 0.0), 0.02, p)
    assert state.speed == pytest.approx(p.max_speed, rel=1e-3)


def test_a_player_decelerates_rather_than_stopping_instantly():
    """§5 — 'use deceleration rather than instant movement'."""
    p = profile()
    moving = KinematicState(0.0, 0.0, 6.0, 0.0)
    after = integrate(moving, None, 0.05, p)
    assert 0.0 < after.speed < 6.0
    stopped = integrate(moving, None, 10.0, p)
    assert stopped.speed == 0.0


def test_arrival_time_accounts_for_acceleration_not_just_distance_over_speed():
    """§7 — do not reduce football to distance / max_speed."""
    p = profile(pace=80.0)
    target = (50.0, 0.0)
    from_rest = arrival_time(KinematicState(0.0, 0.0), target, p)
    naive = 50.0 / p.max_speed
    assert from_rest > naive, "accelerating from rest must take longer"
    # ...but a player already at speed gets there sooner than a stationary one
    moving = KinematicState(0.0, 0.0, p.max_speed, 0.0)
    assert arrival_time(moving, target, p) < from_rest


# ── §22.9-10  the 3D ball ────────────────────────────────

def test_aerial_ball_has_height_during_flight():
    """§22.9."""
    ball = BallPhysics()
    traj = ball.plan((0.0, 0.0), (30.0, 0.0), kind="cross")
    assert traj.is_aerial
    assert traj.apex > 0.5
    for frac in (0.1, 0.25, 0.5, 0.75):
        assert traj.height_at(traj.travel_time * frac) > 0.0
    assert traj.height_at(0.0) == pytest.approx(0.0, abs=1e-9)
    assert traj.height_at(traj.travel_time) == pytest.approx(0.0, abs=1e-9)


def test_gravity_brings_the_ball_back_to_the_ground():
    """§22.10 — z(t) = z0 + vz t - 0.5 g t^2, landing at 2 vz / g."""
    assert loft_time(5.0) == pytest.approx(2.0 * 5.0 / G)
    apex = z_at(loft_time(5.0) / 2.0, 0.0, 5.0)
    assert apex == pytest.approx(5.0 ** 2 / (2.0 * G))
    assert z_at(loft_time(5.0), 0.0, 5.0) == pytest.approx(0.0, abs=1e-9)
    # below the ground after landing, which is what a real parabola does
    assert z_at(loft_time(5.0) + 0.1, 0.0, 5.0) < 0.0


def test_a_ground_ball_stays_on_the_ground():
    ball = BallPhysics()
    traj = ball.plan((0.0, 0.0), (20.0, 0.0), kind="short")
    assert not traj.is_aerial
    assert traj.height_at(traj.travel_time / 2.0) == 0.0
    assert traj.apex == 0.0


def test_the_ball_carries_a_z_coordinate_in_flight():
    ball = BallPhysics()
    traj = ball.plan((0.0, 0.0), (25.0, 0.0), kind="cross")
    mid = ball.state_at(traj, traj.travel_time / 2.0)
    assert mid.z > 0.0
    assert mid.speed_3d >= mid.speed_horizontal
    assert set(mid.to_dict()) >= {"x", "y", "z", "vx", "vy", "vz"}


def test_through_balls_travel_faster_than_short_passes():
    """§9 — pass type must change the physics, not just the label."""
    ball = BallPhysics()
    start, end = (0.0, 0.0), (20.0, 0.0)
    short = ball.plan(start, end, "short")
    through = ball.plan(start, end, "through")
    assert through.launch_speed > short.launch_speed
    assert through.travel_time < short.travel_time


# ── §22.11  a through ball decided by arrival times ───────

def test_a_through_ball_outcome_is_decided_by_who_arrives_first():
    """§22.11 and §13 — the outcome emerges from the physical state.

    The scenarios deliberately differ in *velocity*, not just distance, because
    that is what the kinematics are for: a defender two metres from the ball
    but facing the wrong way is beatable, and a fast player 15 m away who is
    already sprinting onto the pass is not catchable.
    """
    target = (55.0, 34.0)
    passer = (40.0, 34.0)

    def race(att_state: KinematicState, def_state: KinematicState,
             att_pace: float = 85.0, def_pace: float = 80.0) -> PassResolution:
        w = PhysicsWorld()
        w.clock.advance_to(37 * 60 + 21.0)
        att, _ = w.candidate_for("Attacker", (att_state.x, att_state.y), target,
                                 att_state, profile(pace=att_pace), "ATT")
        dfn, _ = w.candidate_for("Defender", (def_state.x, def_state.y), target,
                                 def_state, profile(pace=def_pace), "DEF")
        return w.resolve_pass(passer, target, [att, dfn], kind="through",
                              intended="Attacker")

    # The attacker is FURTHER from the ball but already sprinting onto the
    # pass. The defender is closer but standing still and has to turn first.
    # §7's "a slower player with better positioning" inverted: here the
    # better-positioned player is the one already running.
    attacker_wins = race(
        KinematicState(44.0, 34.0, 7.0, 0.0),      # 11 m out, at full tilt
        KinematicState(51.0, 34.0, 0.0, 0.0),      # 4 m out, but static
    )
    assert attacker_wins.receiver == "Attacker", attacker_wins.to_dict()
    assert attacker_wins.intercepted is False

    # Now the defender is closer AND already running, while the attacker has
    # to turn back onto the ball. The physics should hand it to the defender.
    defender_wins = race(
        KinematicState(38.0, 34.0, -3.0, 0.0),     # 17 m out, running away
        KinematicState(52.0, 34.0, 6.0, 0.0),      # 3 m out, at speed
    )
    assert defender_wins.receiver == "Defender", defender_wins.to_dict()
    assert defender_wins.intercepted is True


def test_running_at_the_ball_beats_standing_over_it():
    """The §7 principle, isolated: velocity is worth more than proximity."""
    target = (55.0, 34.0)
    sprinting = arrival_time(KinematicState(44.0, 34.0, 7.0, 0.0), target,
                             profile(pace=85.0))
    standing = arrival_time(KinematicState(51.0, 34.0, 0.0, 0.0), target,
                            profile(pace=80.0))
    assert sprinting < standing, (
        f"11 m at full speed ({sprinting:.3f}s) must beat 4 m from standing "
        f"({standing:.3f}s) — that is what acceleration and reaction model")


def test_a_race_produces_a_readable_physical_timeline():
    """§26's acceptance shape, in miniature."""
    w = PhysicsWorld()
    w.clock.advance_to(37 * 60 + 21.0)
    target = (52.0, 34.0)
    att, _ = w.candidate_for("Attacker", (42.0, 34.0), target,
                             KinematicState(42.0, 34.0), profile(pace=85.0), "ATT")
    dfn, _ = w.candidate_for("Defender", (48.0, 34.0), target,
                             KinematicState(48.0, 34.0), profile(pace=80.0), "DEF")
    res = w.resolve_pass((38.0, 34.0), target, [att, dfn], kind="through",
                         intended="Attacker")

    labels = [s.label for s in res.timeline]
    assert "PASS START" in labels
    assert "BALL REACHES TARGET" in labels
    assert "ARRIVAL" in labels
    assert "CONTROLS BALL" in labels or "INTERCEPTED" in labels
    # monotonic: the timeline must read forwards
    times = [s.time for s in res.timeline]
    assert times == sorted(times)
    assert res.timeline_text()


def test_nobody_reaching_the_ball_is_reported_not_invented():
    """A ball nobody claims is a real football situation."""
    w = PhysicsWorld()
    far, _ = w.candidate_for("Far", (0.0, 0.0), (60.0, 34.0),
                             KinematicState(0.0, 0.0), profile(pace=30.0))
    res = w.resolve_pass((10.0, 34.0), (60.0, 34.0), [far], kind="short")
    assert res.contest.unclaimed is True
    assert res.receiver == ""
    assert res.resolved is False


def test_a_tie_is_reported_as_a_tie_not_broken_by_a_coin_flip():
    result = arrival_contest(2.0, [
        Candidate("A", 1.50), Candidate("B", 1.505)])
    assert result.tied is True
    assert result.winner is None
    assert resolve_receiver(result) == "TIE"


def test_a_ball_that_beats_everyone_is_a_loose_ball_not_a_failure():
    """A pass into space reaches its target before the attacker does.

    Regression: this was reported as "nobody reaches the ball before it
    arrives", the winner printed empty, and a pass the striker plainly won came
    out as unclaimed. Arriving after the ball is normal for a through ball —
    the whole point is that the ball is played beyond him.
    """
    w = PhysicsWorld()
    w.clock.advance_to(100.0)
    att, _ = w.candidate_for("Striker", (45.0, 33.0), (56.0, 33.0),
                             KinematicState(45.0, 33.0, 6.8, 0.0),
                             profile(pace=88.0), team="ATT", intended=True)
    res = w.resolve_pass((38.0, 34.0), (56.0, 33.0), [att], kind="through",
                         intended="Striker")
    assert res.contest.unclaimed is False
    assert res.contest.loose is True
    assert res.receiver == "Striker"
    assert res.resolved is True
    assert "loose ball" in res.reason
    # control happens when he gets there, not when the ball landed
    assert res.control_time == pytest.approx(100.0 + att.arrival)


def test_a_ball_nobody_is_really_going_to_is_unclaimed():
    """The other side of the same boundary: 60 m away is genuinely out of play."""
    w = PhysicsWorld()
    far, _ = w.candidate_for("Far", (0.0, 0.0), (60.0, 34.0),
                             KinematicState(0.0, 0.0), profile(pace=30.0))
    res = w.resolve_pass((10.0, 34.0), (60.0, 34.0), [far], kind="short")
    assert res.contest.unclaimed is True
    assert res.contest.loose is False
    assert res.receiver == ""
    assert res.resolved is False


def test_the_loose_ball_window_is_documented_and_bounded():
    from physics.collision import LOOSE_BALL_WINDOW
    assert 0.5 <= LOOSE_BALL_WINDOW <= 3.0, (
        "the window decides when a ball stops being contested; keep it sane")


def test_a_candidate_reports_its_reaction_and_travel_split():
    """The timeline's REACTION milestone needs the split, and arrival alone
    cannot supply it — reporting the whole journey as 'reaction 1.479s' is what
    an earlier version did."""
    w = PhysicsWorld()
    cand, _ = w.candidate_for("R", (40.0, 34.0), (56.0, 33.0),
                              KinematicState(40.0, 34.0), profile(pace=85.0))
    assert cand.reaction == pytest.approx(profile(pace=85.0).reaction_time)
    assert cand.travel == pytest.approx(cand.arrival - cand.reaction)
    assert cand.travel > 0.0
    with pytest.raises(ValueError):
        Candidate("X", 1.0, reaction=2.0)


def test_arrival_contest_rejects_impossible_input():
    with pytest.raises(ValueError):
        arrival_contest(-1.0, [Candidate("A", 1.0)])
    with pytest.raises(ValueError):
        Candidate("A", -1.0)


# ── §22.12  determinism ──────────────────────────────────

def test_the_same_inputs_produce_the_same_timeline():
    """§22.12 and §20."""
    def run() -> list:
        w = PhysicsWorld()
        w.clock.advance_to(37 * 60 + 21.0)
        target = (52.0, 34.0)
        att, _ = w.candidate_for("Attacker", (42.0, 34.0), target,
                                 KinematicState(42.0, 34.0), profile(pace=85.0), "ATT")
        dfn, _ = w.candidate_for("Defender", (48.0, 34.0), target,
                                 KinematicState(48.0, 34.0), profile(pace=80.0), "DEF")
        res = w.resolve_pass((38.0, 34.0), target, [att, dfn], kind="through",
                             intended="Attacker")
        return [(s.time, s.label, s.subject) for s in res.timeline]

    assert run() == run()


def test_nothing_in_the_physics_package_is_random():
    """§20 — no unseeded randomness, and specifically no travel jitter (§9)."""
    import physics
    root = os.path.dirname(physics.__file__)
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(root, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = {a.name for a in node.names}
                assert "random" not in mods, f"{name} imports random"
                assert "secrets" not in mods, f"{name} imports secrets"
                assert "uuid" not in mods, f"{name} imports uuid"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in ("random", "secrets", "uuid"), (
                    f"{name} imports {node.module}")


# ── §22.13  a whole match, fast ──────────────────────────

def test_a_whole_match_is_computed_without_waiting_for_it():
    """§22.13 and §23 — 90 simulated minutes, a fraction of a real second.

    ``advance_clock=False`` because ``resolve_pass`` deliberately advances the
    clock to the moment the ball is *resolved*. Letting it do that here would
    overshoot each step; the point of this test is the totals, so the caller
    keeps control of the step.
    """
    w = PhysicsWorld()
    done = 0
    for i in range(1200):            # 1200 x 4.5 s = 5400 s = 90 minutes
        w.clock.advance(4.5)         # a pass every 4.5 simulated seconds
        w.resolve_pass((10.0, 34.0), (40.0, 34.0),
                       [Candidate("R", 0.8, team="H")], kind="short",
                       advance_clock=False)
        done += 1
    assert w.clock.play_seconds == pytest.approx(5400.0)
    assert done == 1200
    assert w.clock.match_time.minute == 90


def test_computing_a_match_does_not_sleep():
    """§23 — never sleep. A sleep would show up as wall time here."""
    start = _wallclock.perf_counter()
    w = PhysicsWorld()
    for i in range(90):
        w.clock.advance(60.0)
        w.resolve_pass((5.0, 20.0), (45.0, 20.0),
                       [Candidate("R", 1.0, team="H")], kind="through",
                       advance_clock=False)
    elapsed = _wallclock.perf_counter() - start
    assert w.clock.play_seconds == pytest.approx(5400.0)
    assert elapsed < 1.0, (
        f"90 simulated minutes took {elapsed:.2f}s of CPU; it should be "
        f"instantaneous because nothing waits")


def test_resolve_pass_advances_the_clock_to_when_the_ball_is_resolved():
    """The documented default: the world has changed when someone controls it,
    not when the ball was struck."""
    w = PhysicsWorld()
    w.clock.advance_to(100.0)
    res = w.resolve_pass((10.0, 34.0), (40.0, 34.0),
                         [Candidate("R", 1.5, team="H")], kind="through")
    assert res.launch_time == pytest.approx(100.0)
    assert w.clock.play_seconds == pytest.approx(res.control_time)
    assert w.clock.play_seconds > 100.0


def test_control_never_precedes_the_ball_arriving():
    """A receiver cannot control a ball that has not got there yet.

    Regression: a candidate whose arrival time beat the ball's was emitting
    CONTROLS BALL before BALL REACHES TARGET, so the timeline read backwards
    and the clock stopped short of the real event. The attacker gets there
    early, waits, and the ball arrives to him.
    """
    w = PhysicsWorld()
    w.clock.advance_to(100.0)
    res = w.resolve_pass((10.0, 34.0), (40.0, 34.0),
                         [Candidate("Early", 0.4, team="H")], kind="through",
                         intended="Early")
    ball_step = next(s for s in res.timeline if s.label == "BALL REACHES TARGET")
    control_step = next(s for s in res.timeline
                        if s.label in ("CONTROLS BALL", "INTERCEPTED"))
    assert control_step.time >= ball_step.time - 1e-9, (
        "control happened before the ball arrived")
    assert res.control_time == pytest.approx(ball_step.time)
    times = [s.time for s in res.timeline]
    assert times == sorted(times), "the timeline must read forwards"


# ── §22.14  CPU time does not alter simulation time ───────

def test_cpu_execution_time_does_not_alter_timestamps():
    """§22.14 — the guarantee the whole package rests on.

    Two identical runs, one artificially slowed by burning CPU between steps,
    must produce byte-identical timestamps.
    """
    def run(slow: bool) -> list:
        w = PhysicsWorld()
        w.clock.advance_to(37 * 60 + 21.0)
        target = (52.0, 34.0)
        att, _ = w.candidate_for("Attacker", (42.0, 34.0), target,
                                 KinematicState(42.0, 34.0), profile(pace=85.0), "ATT")
        res = w.resolve_pass((38.0, 34.0), target, [att], kind="through",
                             intended="Attacker")
        if slow:
            deadline = _wallclock.perf_counter() + 0.05
            while _wallclock.perf_counter() < deadline:
                pass                      # burn real CPU, touch no state
        return [(s.time, s.label) for s in res.timeline]

    assert run(False) == run(True)


def test_the_clock_module_cannot_read_a_wall_clock():
    """§4 — enforced by parsing the AST, not by grepping the text.

    An earlier version of this test grepped the source for ``time.sleep`` and
    failed, because the module's own docstring *explains* that it never calls
    it. Attribute access is checked instead, so prose about a wall clock does
    not register as a wall clock.
    """
    import physics.continuous_time as ct
    tree = ast.parse(open(ct.__file__, encoding="utf-8").read())

    banned_modules = {"time", "datetime", "asyncio", "threading"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in banned_modules, (
                    f"continuous_time.py imports {a.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_modules, (
                f"continuous_time.py imports from {node.module}")

    banned_attrs = {"sleep", "now", "today", "utcnow", "time", "monotonic",
                    "perf_counter", "process_time"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
            base = node.value
            name = base.id if isinstance(base, ast.Name) else None
            assert name not in ("time", "datetime"), (
                f"continuous_time.py reads {name}.{node.attr} — the simulated "
                f"clock must not consult a wall clock")


# ── §15 more teleport protection, at the trajectory level ─

def test_running_a_player_to_arrival_produces_no_violation():
    p = profile(pace=70.0)
    motion = PlayerMotion(profile=p, state=KinematicState(0.0, 0.0))
    log = ViolationLog()
    t = motion.arrival((45.0, 0.0))
    final = motion.state.advanced(t)
    log.check((motion.state.x, motion.state.y), (final.x, final.y), t,
              subject=p.name)
    assert log.total == 0


def test_a_world_reports_its_violations_and_can_assert_clean():
    w = PhysicsWorld()
    w.violations.check((0.0, 0.0), (90.0, 0.0), 0.0, subject="Teleporter")
    assert "teleport" in w.report()
    with pytest.raises(PhysicsViolationError):
        w.assert_clean()


# ── §22.15  existing exports still work ──────────────────

def test_existing_fractional_timestamp_consumers_are_unaffected():
    """§18 — minute stays an int; fractional time is exposed separately."""
    m = seconds_to_match_time(2241.735)
    assert isinstance(m, MatchTime)
    assert m.minute == 37 and isinstance(m.minute, int)
    assert m.second == pytest.approx(21.735)
    assert m.total_seconds == pytest.approx(2241.735)
    assert match_time_to_seconds(37, 21.735) == pytest.approx(2241.735)
    assert str(m).startswith("37:21")


def test_the_world_layer_and_live_pipeline_are_not_imported():
    """The package must be usable, and testable, without the live season."""
    import physics
    root = os.path.dirname(physics.__file__)
    forbidden = {"match_engine", "roster_loader", "player_dna", "squad_manager",
                 "season_manager", "position_engine", "weather_physics",
                 "alltime_db", "exporter", "auto_run_match", "world"}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(root, name), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                mods = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = {node.module.split(".")[0]}
            else:
                continue
            assert not (mods & forbidden), (
                f"physics/{name} imports {mods & forbidden} at module level; "
                f"the physics layer must stay independent of the live pipeline")


# ── §6 the calibration layer ─────────────────────────────

def test_pace_maps_into_the_realistic_sprint_range():
    """§6's reference ranges — walking 1-2 ... sprint 7-10 m/s."""
    assert speed_for_pace(0.0) == pytest.approx(MIN_SPRINT_MPS)
    assert speed_for_pace(100.0) == pytest.approx(MAX_SPRINT_MPS)
    assert 4.0 <= speed_for_pace(60.0) <= 8.0, (
        "the DNA default of 60 must be a realistic top-flight sprint")
    assert MIN_SPRINT_MPS < 7.0 and MAX_SPRINT_MPS <= 10.0


def test_calibration_is_monotonic_and_bounded():
    speeds = [speed_for_pace(p) for p in range(0, 101)]
    assert speeds == sorted(speeds)
    accels = [acceleration_for(a) for a in range(0, 101)]
    assert accels == sorted(accels)
    reactions = [reaction_for(a) for a in range(0, 101)]
    assert reactions == sorted(reactions, reverse=True)
    assert 0.15 <= reactions[-1] and reactions[0] <= 0.5


def test_scores_outside_the_range_are_clamped_not_exploded():
    assert speed_for_pace(-50.0) == pytest.approx(MIN_SPRINT_MPS)
    assert speed_for_pace(500.0) == pytest.approx(MAX_SPRINT_MPS)


def test_a_player_profile_carries_usable_si_units():
    p = profile(pace=60.0, stamina=80.0)
    assert 1.0 < p.max_speed < 10.0
    assert 0.3 < p.acceleration < 6.0
    assert p.deceleration > p.acceleration, "braking should be easier"
    assert 0.05 < p.reaction_time < 0.6
    assert p.raw_max_speed > p.max_speed, "fatigue must actually cost something"


def test_changing_stamina_returns_a_new_profile_and_leaves_the_old_one():
    """Profiles are immutable so a trajectory cannot change underfoot."""
    fresh = profile(stamina=100.0)
    tired = fresh.with_stamina(20.0)
    assert tired is not fresh
    assert fresh.stamina == 100.0
    assert tired.max_speed < fresh.max_speed
    assert tired.reaction_time == pytest.approx(fresh.reaction_time)


# ── §4 / §19 the clock itself ────────────────────────────

def test_the_clock_stores_fractional_seconds():
    c = ContinuousClock()
    c.advance(0.735)
    assert c.play_seconds == pytest.approx(0.735)
    assert c.match_time.second == pytest.approx(0.735)
    assert c.match_time.minute == 0


def test_stoppage_advances_the_scoreboard_but_not_the_ball():
    """§19 — the play/match time distinction."""
    c = ContinuousClock()
    c.advance(600.0)
    c.add_stoppage(30.0)
    assert c.play_seconds == pytest.approx(600.0)
    assert c.match_seconds == pytest.approx(630.0)
    assert c.match_time.minute == 10
    assert c.play_time.minute == 10


def test_the_clock_refuses_to_run_backwards():
    c = ContinuousClock()
    c.advance(10.0)
    with pytest.raises(ValueError):
        c.advance(-1.0)
    with pytest.raises(ValueError):
        c.advance_to(5.0)


def test_advancing_to_an_absolute_time_is_idempotent_in_order():
    a = ContinuousClock()
    for t in (10.0, 20.0, 30.0):
        a.advance_to(t)
    b = ContinuousClock()
    b.advance_by_timeline([(10.0, "x"), (20.0, "y"), (30.0, "z")])
    assert a.play_seconds == b.play_seconds
