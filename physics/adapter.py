"""
physics/adapter.py — the only place PLOFA and the physics layer meet.
=====================================================================

``physics/`` is pure: it knows about metres, seconds and 0-100 opinion scores,
and it imports nothing from the live pipeline. This module is the seam. It reads
the engine's state and hands the physics layer SI-unit numbers, and it is the
*only* file permitted to know about both.

That boundary is not tidiness, it is the mechanism that let the physics be
built, tested and calibrated while 26/27 was running. It is also enforced:
``tests/test_physics.py`` walks the AST of every module in the package and fails
if any of them imports ``match_engine``, ``position_engine``, ``squad_manager``,
``weather_physics`` or the rest at module level. The imports below are all
inside functions, for the same reason.

Three problems this solves, none of them obvious
-----------------------------------------------
**1. The engine has no velocity.** ``PlayerSpatialState`` stores
``current_x``/``current_y`` and nothing else. But the whole arrival-time model
turns on velocity — 11 m at full sprint beats 4 m from standing, and that fact
is invisible without it. So the adapter estimates it by finite difference over
simulated time, and clamps the result to a physical maximum. A naive difference
over a 20 ms engine tick would be mostly quantisation noise; a difference over a
long window would lag a player who has just turned. The window is a parameter
and the default is documented below.

**2. Stamina changes every tick, calibration does not.** Calibrating a
``PhysicalProfile`` is arithmetic worth caching; deriving it at 45% stamina is
not, and must not be cached or the physics would use a stale body. So the
stamina-free profile is cached per player and re-fatigued per call via
``PhysicalProfile.with_stamina``.

**3. Weather already exists and should not be reinvented.** §21 says use
PLOFA's ``weather_physics``. It exposes ``rolling_decel_mult`` (>1 = a wet pitch
grips the ball and it dies sooner) and ``pitch_grip`` (<1 = slick studs). Both
map onto physics parameters, in opposite directions, and getting either backwards
would be invisible until a rainy match played wrong.

Positions are never cached
--------------------------
§16: "There must be ONE authoritative current position." ``PositionEngine``
owns where players are. This module reads them on demand and holds nothing, so a
substitution or a reset cannot leave the physics working from a stale copy.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from physics.ball_physics import BALL_DRAG_COEFFICIENT, BallPhysics
from physics.calibration import Calibration, PhysicalProfile, calibrate
from physics.collision import Candidate
from physics.continuous_time import ContinuousClock
from physics.movement import MovementMemory
from physics.player_motion import ABSOLUTE_SPEED_CEILING, KinematicState
from physics.world import PassResolution, PhysicsWorld

#: Simulated seconds between velocity samples. A long enough window that engine
#: tick quantisation does not dominate the difference, short enough that a
#: player who has just turned is not still reported as running the old way.
VELOCITY_WINDOW_S = 0.35

#: Two samples closer together than this are treated as "no movement" rather
#: than as an implausibly large speed from a tiny denominator.
MIN_VELOCITY_SAMPLE_S = 1e-3

#: How many samples to keep per player. Three is enough for a window and a
#: little slack; unbounded would be a slow leak in a long match.
VELOCITY_HISTORY = 3


@dataclass
class _Sample:
    t: float
    x: float
    y: float


# ─────────────────────────────────────────────
# THE ACTIVE ADAPTER — a single slot, not a registry
# ─────────────────────────────────────────────
# ``PossessionChain.generate`` is a classmethod with no engine reference: it
# receives ``state`` and ``position_engine`` and nothing that identifies the
# match. Threading an adapter through that signature would mean touching every
# call site in the engine's hot path.
#
# A single module-level slot is the alternative, and it is the pattern this
# codebase already uses for exactly this problem — ``WeatherPhysics`` is set
# from the engine constructor and read from deep inside the chains.
#
# **One slot, deliberately, not a dict.** A registry keyed by engine id is the
# ``id()``-reuse bug that cross-match contamination is made of: entries outlive
# the match that created them and a recycled address hands one match another
# match's adapter. A single slot cannot accumulate, and ``_init_physics`` always
# writes it — including writing ``None`` when physics is off.
_ACTIVE_ADAPTER: Optional["EngineAdapter"] = None


def set_active_adapter(adapter: Optional["EngineAdapter"]) -> None:
    """Point the module at the current match's adapter, or ``None`` to clear."""
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter


def active_adapter() -> Optional["EngineAdapter"]:
    """The current match's adapter, or ``None`` when physics is off."""
    return _ACTIVE_ADAPTER


#: How late a defender may be before a block stops being physically possible.
#: A player arriving a fraction of a second after the ball can still get a toe
#: on it; one arriving much later cannot, whatever his lane distance suggests.
BLOCK_LATE_TOLERANCE_S = 0.50

#: The existing block probability's own floor (``_pass_block_probability`` clamps
#: to 0.02). The gate is applied *under* that floor rather than replacing it, so
#: a physically impossible block becomes merely very unlikely instead of
#: impossible — which is both more honest and less disruptive to a calibration
#: that was tuned around a 2% floor.
BLOCK_PROBABILITY_FLOOR = 0.02

#: What the gate did, counted. Read by ``physics/measure_gate.py`` so the
#: mechanism can be measured without confounding it against the engine's own
#: run-to-run non-determinism.
_GATE_COUNTERS: Dict[str, int] = {}

# ── goalkeeper dive constants ────────────────────────────────────────────
#: A keeper can extend roughly this far beyond his standing reach at full
#: stretch. Real maximum is around 2.5 m of lateral reach from a set position;
#: the model's ``base_reach`` is 2.5, so this is the *extra* a dive buys.
GK_MAX_DIVE_EXTENSION_M = 1.4

#: Metres of extra reach per second of available dive time. Deliberately
#: modest: a keeper's dive is explosive but short, and overstating this would
#: credit him for shots he cannot actually reach.
GK_DIVE_RATE_MPS = 2.5

#: How far beyond the available reach a shot may be before the keeper is given
#: no credit at all. Matches the shape of the engine's own falloff
#: (``max(0.2, 1.0 - overshoot * 0.4)``) over a comparable distance.
GK_DIVE_SLACK_M = 1.2


def _name_of(player) -> str:
    """A player's name, whether the engine holds an object or a raw tuple.

    ``set_squad`` stores its argument verbatim, and production supplies
    ``PlayerProfile`` objects while a test or a tool may supply
    ``(name, position, specialties)`` tuples. Both are legitimate engine
    states; neither should make the physics layer quietly find nobody.
    """
    name = getattr(player, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(player, (tuple, list)) and player:
        first = player[0]
        if isinstance(first, str) and first:
            return first
    return ""


class EngineAdapter:
    """Reads a live ``MatchEngine`` and answers physics questions about it."""

    def __init__(self, engine, *, clock: Optional[ContinuousClock] = None,
                 world: Optional[PhysicsWorld] = None,
                 weather=None, speed_ceiling: float = 12.0) -> None:
        self.engine = engine
        self.clock = clock or ContinuousClock()
        self.world = world or PhysicsWorld(clock=self.clock)
        self.weather = weather
        self.speed_ceiling = speed_ceiling
        self._profiles: Dict[str, PhysicalProfile] = {}
        self._velocity: Dict[str, Deque[_Sample]] = {}
        #: Per-player momentum + held target, for the physical off-ball step.
        #: Dropped alongside ``_velocity`` on substitution, because a carried
        #: velocity across a substitution is the same teleport problem.
        self._movement: Dict[str, MovementMemory] = {}
        #: every resolution produced, for reporting and for the §27 audit
        self.history: List[PassResolution] = []

    # -- weather (§21) --------------------------------------------------
    def calibration(self) -> Calibration:
        """Translate PLOFA's weather into physics multipliers.

        Direction matters and is easy to get backwards:

        * ``rolling_decel_mult`` is **>1 on a wet pitch** — the ball bites and
          dies sooner — so it multiplies drag directly.
        * ``pitch_grip`` is **<1 on a slick pitch** — less traction, so a player
          turns and accelerates worse. Grip is mapped into a narrow band around
          1.0 so slick conditions cost a few percent rather than destroying
          movement.

        One trap worth naming: every one of these multipliers is gated behind
        ``WeatherPhysics.is_active()``. Pass a rainy condition with weather
        switched off and PLOFA itself returns 1.0 for all of them — so the
        physics would quietly simulate a dry match. That is PLOFA's rule, not
        something to override, so this records it in ``notes`` rather than
        forcing the flag on behind the engine's back.
        """
        if self.weather is None:
            return Calibration(notes=("no weather supplied; neutral",))
        from weather_physics import WeatherPhysics

        if not WeatherPhysics.is_active(self.weather):
            return Calibration(
                notes=("weather supplied but WeatherPhysics is inactive, so "
                       "PLOFA returns neutral multipliers; the physics is "
                       "deliberately neutral too rather than overriding it",))

        rolling = float(WeatherPhysics.rolling_decel_mult(self.weather))
        grip = float(WeatherPhysics.pitch_grip(self.weather))
        # grip 0.65 (mud) -> 0.93; grip 1.0 (dry) -> 1.0
        grip_factor = 0.85 + 0.15 * grip
        return Calibration(
            speed_multiplier=grip_factor,
            acceleration_multiplier=grip_factor,
            reaction_multiplier=1.0,
            notes=(f"rolling_decel_mult={rolling:.2f} (drag x{rolling:.2f}), "
                   f"pitch_grip={grip:.2f} (movement x{grip_factor:.3f})",),
        )

    def ball(self) -> BallPhysics:
        """A ball model whose drag reflects the conditions."""
        drag = BALL_DRAG_COEFFICIENT
        if self.weather is not None:
            from weather_physics import WeatherPhysics
            if WeatherPhysics.is_active(self.weather):
                drag *= float(WeatherPhysics.rolling_decel_mult(self.weather))
        return BallPhysics(drag=drag)

    # -- players (§6, §14) ---------------------------------------------
    def _player_objects(self, team: Optional[str] = None) -> Dict[str, object]:
        """Map player name -> the engine's player object, for DNA lookup.

        Handles both shapes the engine can hold. Production calls
        ``set_squad`` with the ``PlayerProfile`` objects from
        ``SquadBuilder``, which carry ``.name`` and ``.dna``. But
        ``set_squad`` stores whatever it is given verbatim, so a caller that
        passes the raw ``(name, pos, specialties)`` tuples leaves tuples in
        ``active_players`` — and an adapter that only understood objects would
        silently return zero candidates, which is the worst possible failure
        mode for a physics layer: no error, just a match where nobody ever
        contests a ball.
        """
        active = getattr(self.engine, "active_players", {}) or {}
        out: Dict[str, object] = {}
        for team_name, players in active.items():
            if team is not None and team_name != team:
                continue
            for p in players or ():
                name = _name_of(p)
                if name:
                    out[name] = p
        return out

    def stamina_of(self, name: str, default: float = 100.0) -> float:
        """Current stamina, 0-100, from the existing system (§14).

        Reads ``squad_manager`` rather than keeping a second copy, because §14
        says not to build a second stamina engine.

        The engine stores the controller as ``sub_controller`` (set by
        ``set_stamina_controller``). ``stamina_controller`` is accepted as a
        fallback so this keeps working if that is ever renamed.
        """
        controller = getattr(self.engine, "sub_controller", None)
        if controller is None:
            controller = getattr(self.engine, "stamina_controller", None)
        table = getattr(controller, "stamina", None) if controller else None
        if not table:
            return default
        state = table.get(name)
        if state is None:
            return default
        value = getattr(state, "current_stamina", None)
        return default if value is None else float(value)

    def base_profile(self, name: str) -> PhysicalProfile:
        """The stamina-free profile, cached per player.

        Calibration reads six DNA scores and does four multiplications; doing it
        for 22 players on every pass would be waste. The cache holds the
        *stamina-free* profile only.
        """
        cached = self._profiles.get(name)
        if cached is not None:
            return cached

        player = self._player_objects().get(name)
        dna = getattr(player, "dna", None)
        if dna is not None:
            profile = calibrate(dna, stamina=100.0,
                                calibration=self.calibration())
        else:
            profile = self.world.profile(name=name)
        profile = PhysicalProfile(
            name=profile.name or name,
            position=profile.position,
            max_speed=profile.raw_max_speed,
            acceleration=profile.raw_acceleration,
            deceleration=profile.raw_acceleration * 1.4,
            reaction_time=profile.reaction_time,
            stamina=100.0,
            raw_max_speed=profile.raw_max_speed,
            raw_acceleration=profile.raw_acceleration,
            agility=profile.agility,
        )
        self._profiles[name] = profile
        return profile

    def profile_of(self, name: str) -> PhysicalProfile:
        """The player's profile at their CURRENT stamina. Never cached."""
        return self.base_profile(name).with_stamina(self.stamina_of(name))

    # -- positions (§16) -----------------------------------------------
    def position_of(self, name: str) -> Optional[Tuple[float, float]]:
        pe = getattr(self.engine, "position_engine", None)
        states = getattr(pe, "states", None) if pe else None
        if not states:
            return None
        st = states.get(name)
        if st is None:
            return None
        x, y = getattr(st, "current_x", None), getattr(st, "current_y", None)
        if x is None or y is None:
            return None
        return (float(x), float(y))

    def velocity_of(self, name: str) -> Tuple[float, float]:
        """Estimated velocity in m/s, from finite differences.

        The engine stores no velocity, so this samples position against the
        simulated clock. Clamped to :data:`ABSOLUTE_SPEED_CEILING`, because a
        finite difference across a substitution or a reset would otherwise
        report a teleport as a velocity and the physics would believe it.
        """
        pos = self.position_of(name)
        if pos is None:
            return (0.0, 0.0)

        t = self.clock.play_seconds
        buf = self._velocity.setdefault(name, deque(maxlen=VELOCITY_HISTORY))
        buf.append(_Sample(t, pos[0], pos[1]))

        if len(buf) < 2:
            return (0.0, 0.0)
        newest, oldest = buf[-1], buf[0]
        dt = newest.t - oldest.t
        if dt < MIN_VELOCITY_SAMPLE_S:
            return (0.0, 0.0)

        vx = (newest.x - oldest.x) / dt
        vy = (newest.y - oldest.y) / dt
        speed = math.hypot(vx, vy)
        if speed > ABSOLUTE_SPEED_CEILING:
            scale = ABSOLUTE_SPEED_CEILING / speed
            vx, vy = vx * scale, vy * scale
        return (vx, vy)

    def state_of(self, name: str) -> Optional[KinematicState]:
        pos = self.position_of(name)
        if pos is None:
            return None
        vx, vy = self.velocity_of(name)
        return KinematicState(pos[0], pos[1], vx, vy)

    def team_of(self, name: str) -> str:
        for team_name, players in (getattr(self.engine, "active_players", {})
                                    or {}).items():
            for p in players or ():
                if _name_of(p) == name:
                    return team_name
        return ""

    # -- races (§12, §13) ----------------------------------------------
    def candidates_for(self, target: Tuple[float, float],
                       team: Optional[str] = None,
                       names: Optional[Sequence[str]] = None,
                       reach: Optional[float] = None) -> List[Candidate]:
        """Build arrival-time candidates for a ball arriving at ``target``.

        Both teams by default: a through ball is contested by the attacker's
        run *and* the defender's recovery, which is the whole point of §13.
        """
        pool: Iterable[str]
        if names is not None:
            pool = names
        else:
            players = self._player_objects(team)
            pool = sorted(players)

        out: List[Candidate] = []
        for name in pool:
            state = self.state_of(name)
            if state is None:
                continue
            profile = self.profile_of(name)
            cand, _dist = self.world.candidate_for(
                name, (state.x, state.y), target, state, profile,
                team=self.team_of(name), reach=reach)
            out.append(cand)
        return out

    def resolve_pass(self, start: Tuple[float, float], end: Tuple[float, float],
                     kind: str = "short", *, intended: str = "",
                     names: Optional[Sequence[str]] = None,
                     reach: Optional[float] = None) -> PassResolution:
        """Plan a pass in engine coordinates and resolve it physically."""
        self.world.ball = self.ball()
        candidates = self.candidates_for(end, names=names, reach=reach)
        res = self.world.resolve_pass(start, end, candidates, kind=kind,
                                       intended=intended)
        self.history.append(res)
        return res

    # -- housekeeping ---------------------------------------------------
    def sync_clock(self, match_clock_s: Optional[float] = None) -> ContinuousClock:
        """Align the physics clock with the engine's own continuous clock.

        The engine already has a fractional ``match_clock_s`` (§4 was already
        satisfied), so the physics clock follows it rather than inventing a
        second timeline. Only ever moves forward.
        """
        if match_clock_s is None:
            state = getattr(self.engine, "state", None)
            match_clock_s = getattr(state, "match_clock_s", None) if state else None
        if match_clock_s is None:
            return self.clock
        target = float(match_clock_s)
        if target > self.clock.play_seconds:
            self.clock.advance_to(target)
        return self.clock

    # -- pass blocking: physics against the engine's existing model ------
    def block_feasibility(self, defender_name: str,
                          start: Tuple[float, float],
                          end: Tuple[float, float],
                          ball_speed_mps: float,
                          lane_dist: float = 0.0,
                          ) -> Tuple[float, str]:
        """Can this defender physically reach the pass lane before the ball?

        Returns ``(factor, reason)`` with ``factor`` in ``[0, 1]``. It
        **attenuates, never replaces**.

        Why not just swap in an arrival-time contest? Because the engine already
        has a geometric block model and it is *tuned*:
        ``_pick_pass_blocker`` finds the nearest defender within
        ``BLOCK_CORRIDOR_M`` of the lane, and ``_pass_block_probability`` turns
        lane distance, ball speed and pass length into a probability clamped to
        0.02-0.45. Replacing that with a hard contest would turn a probability
        into a guaranteed winner, make interceptions far more decisive, and move
        every scoreline in a season calibrated around it — with physics that has
        never been calibrated against PLOFA's own output.

        What the existing model genuinely cannot do is ask whether the defender
        can *arrive in time*. It is a spatial test — he is near the lane — never
        a temporal one. A defender two metres off the line but twenty metres back
        along it, facing the wrong way, against a 25 m/s pass, is not blocking
        anything, and the current model still gives him a floor chance.

        So this returns a multiplier. It can only *reduce* a block, only for a
        defender the physics says cannot get there, and only when physics is
        enabled. A defender who arrives comfortably is untouched.
        """
        state = self.state_of(defender_name)
        if state is None:
            return 1.0, "no state for the blocker; leaving the block alone"

        profile = self.profile_of(defender_name)

        # Where on the segment the defender would meet the ball: the projection
        # of his position onto the pass line, clamped to the segment.
        vx, vy = end[0] - start[0], end[1] - start[1]
        seg2 = vx * vx + vy * vy
        if seg2 < 1e-6:
            return 1.0, "degenerate pass; leaving the block alone"
        t = ((state.x - start[0]) * vx + (state.y - start[1]) * vy) / seg2
        t = max(0.0, min(1.0, t))
        meet = (start[0] + vx * t, start[1] + vy * t)

        from physics.ball_physics import ground_travel_time
        from physics.player_motion import arrival_time

        along = math.hypot(meet[0] - start[0], meet[1] - start[1])
        t_ball = ground_travel_time(along, max(1.0, ball_speed_mps))
        if t_ball == math.inf:
            return 1.0, "ball cannot reach that point; leaving the block alone"
        t_def = arrival_time(state, meet, profile)

        slack = t_ball - t_def
        if slack >= 0.0:
            return 1.0, f"arrives {slack:+.3f}s (in time; block unchanged)"
        late = -slack
        if late >= BLOCK_LATE_TOLERANCE_S:
            return 0.0, f"arrives {late:.3f}s late — physically cannot block"
        factor = 1.0 - (late / BLOCK_LATE_TOLERANCE_S)
        return factor, (f"arrives {late:.3f}s late — block scaled to "
                        f"{factor:.2f}")

    # -- ball flight time (§ the sequence of play) -----------------------
    def pass_flight_time(self, distance_m: float,
                         speed_mps: float) -> Tuple[float, str]:
        """How long the ball is actually in the air for this pass.

        Returns ``(seconds, reason)``. Zero means "do not shift the clock" —
        the caller treats that as physics declining rather than as an instant
        pass.

        Why this is worth having at all
        --------------------------------
        PLOFA already computes a flight time and stores it as
        ``geometry_meta["ball_travel_s"]``. It then uses it for exactly one
        thing: recovering pass *distance* for the miscontrol model
        (``ball_travel_s * ball_speed``). The timeline itself ignores it — the
        pass and the receiver's touch are stamped at the same instant, so a
        40 m ball in behind is logged as arriving at the moment it was struck.

        That is the difference between an event log and a recording of a match.
        It is also why a through ball cannot be chased: the defender's chance to
        arrive is decided by a race model, but the *timeline* still claims
        nothing happened in between.
        """
        from physics.ball_physics import ground_travel_time
        try:
            distance_m = float(distance_m)
            speed_mps = float(speed_mps)
        except Exception:
            return 0.0, "unusable pass geometry; leaving the timeline alone"
        if distance_m <= 0.0 or speed_mps <= 0.0:
            return 0.0, "no usable distance or speed; leaving the timeline alone"
        t = ground_travel_time(distance_m, speed_mps)
        if t == math.inf or t != t:            # inf or NaN
            return 0.0, "ball never arrives; leaving the timeline alone"
        _GATE_COUNTERS["flight_considered"] = (
            _GATE_COUNTERS.get("flight_considered", 0) + 1)
        return t, f"{distance_m:.1f} m at {speed_mps:.1f} m/s -> {t:.3f} s of flight"

    # -- goalkeeper reach: the same missing time dimension ----------------
    def keeper_feasibility(self, shot_x: float, shot_y: float,
                           ball_x: float, ball_y: float,
                           gk_x: float, gk_y: float,
                           base_reach: float,
                           reaction_time: float,
                           shot_speed_mps: Optional[float] = None,
                           ) -> Tuple[float, str]:
        """Given the ball's flight time, can this keeper physically get there?

        Returns ``(factor, reason)`` with ``factor`` in ``[0, 1]``, to
        **attenuate** the engine's save multiplier. Like the block gate, it can
        only remove credit the physics cannot justify — never add any.

        The gap this fills
        ------------------
        ``GoalkeeperEngine._get_gk_positioning`` is a good model: it bisects the
        angle, sets depth by distance to goal, and derives reaction time from
        reflexes and composure. ``_is_shot_savable`` then works out an
        ``effective_reach = reach * (0.8 + reaction_time * 0.4)`` and asks
        whether the shot is inside it.

        What it cannot express is the **flight time**. That formula gives a
        keeper identical reach against a shot from 30 metres as against one from
        8 — even though the first arrives in about 1.2 s and the second in about
        0.35 s, which is a factor of three in the time he actually has. A keeper
        who is beaten by a driven shot from 25 yards has usually not been beaten
        on reach; he has been beaten on time.

        So the ball's travel time is computed here and turned into dive capacity:
        the keeper has ``flight time - reaction`` seconds to extend himself, at
        a finite dive rate, up to a hard cap. That is the one dimension the
        existing model is missing, and it is added rather than substituted.

        Why a keeper's movement is modelled as a dive, not a run
        ----------------------------------------------------------
        A keeper does not sprint to a shot in the penalty area; he extends. So
        his locomotion model is reach (arms, body, momentum) rather than
        top-speed acceleration, and the relevant question is how far he can
        stretch in the time available — not whether he can run there. Using the
        outfield kinematics here would be wrong in a way that looked plausible.
        """
        from physics.ball_physics import PASS_PROFILES, ground_travel_time

        if shot_speed_mps is None:
            shot_speed_mps = PASS_PROFILES["shot"].nominal

        along = ((shot_x - ball_x) ** 2 + (shot_y - ball_y) ** 2) ** 0.5
        t_ball = ground_travel_time(along, max(1.0, shot_speed_mps))
        if t_ball == math.inf:
            return 0.0, "the ball cannot reach that point; keeper gets no credit"

        # Time the keeper has to move after he has reacted at all.
        available = t_ball - max(0.0, reaction_time)
        _GATE_COUNTERS["gk_considered"] = _GATE_COUNTERS.get("gk_considered", 0) + 1
        if available <= 0.0:
            _GATE_COUNTERS["gk_no_time"] = _GATE_COUNTERS.get("gk_no_time", 0) + 1
            return 0.0, (f"only {t_ball:.3f}s of flight against "
                          f"{reaction_time:.3f}s of reaction — no time to dive")

        # Dive extension is finite in both rate and reach.
        extension = min(GK_MAX_DIVE_EXTENSION_M,
                        GK_DIVE_RATE_MPS * available)
        effective_reach = base_reach + extension

        dy = abs(shot_y - gk_y)
        dx = abs(shot_x - gk_x)
        dist = math.hypot(dx, dy)

        _GATE_COUNTERS["gk_total"] = _GATE_COUNTERS.get("gk_total", 0) + 1
        if dist <= effective_reach:
            _GATE_COUNTERS["gk_reachable"] = (
                _GATE_COUNTERS.get("gk_reachable", 0) + 1)
            return 1.0, (f"within {effective_reach:.2f}m of a "
                         f"{dist:.2f}m shot with {available:.2f}s to dive "
                         f"(credit unchanged)")

        # Beyond reach: fall off over GK_DIVE_SLACK_M, the same shape as the
        # engine's own `max(0.2, 1.0 - overshoot * 0.4)`, but anchored on the
        # reach the physics allows rather than the reach the formula assumed.
        overshoot = dist - effective_reach
        if overshoot >= GK_DIVE_SLACK_M:
            _GATE_COUNTERS["gk_unreachable"] = (
                _GATE_COUNTERS.get("gk_unreachable", 0) + 1)
            return 0.0, (f"{dist:.2f}m away, only {effective_reach:.2f}m of "
                          f"reach available in {available:.2f}s — beaten")
        factor = 1.0 - (overshoot / GK_DIVE_SLACK_M)
        _GATE_COUNTERS["gk_attenuated"] = _GATE_COUNTERS.get("gk_attenuated", 0) + 1
        return factor, (f"{dist:.2f}m away vs {effective_reach:.2f}m reach in "
                        f"{available:.2f}s — save credit scaled to {factor:.2f}")

    # -- physical movement: momentum, fatigue, reaction --------------------
    def offball_step(self, player_name: str,
                     target: Tuple[float, float],
                     dt: float,
                     ) -> Optional[Tuple[float, float, float, float]]:
        """Move one outfield player one tick, physically. Or decline.

        Returns ``(x, y, moved, speed)``, or ``None`` when the physics cannot
        own this step — no state, no profile, or a player it cannot see. The
        caller then uses the engine's own movement unchanged, which is the
        whole safety property: a physics problem costs realism, never a match.

        What this adds over the engine's own step
        -------------------------------------------
        The engine moves a player by assigning him a scalar speed and
        displacing him along the straight line to his shape target. That is
        time-stepped and pace-capped, but it means:

          * **no momentum** — he can reverse at full pace, because direction is
            recomputed from scratch every tick;
          * **no acceleration limit on turning** — the chase ramp limits speed,
            not heading;
          * **no fatigue in the legs** — the engine's top speed is
            ``5.0 + pace * 0.042`` from DNA alone, so a player on 20% stamina
            runs as fast as one on 100%;
          * **no reaction delay** — a changed target is acted on instantly.

        :meth:`profile_of` supplies a stamina-scaled max speed, so fatigue
        reaches the legs here for the first time; ``movement.perceive`` holds
        the previous target while a reaction is pending, so the player runs on
        toward where he *thought* the ball was rather than freezing.
        """
        pos = self.position_of(player_name)
        if pos is None:
            return None
        profile = self.profile_of(player_name)
        if profile is None:
            return None

        memory = self._movement.get(player_name)
        if memory is None:
            memory = self._movement[player_name] = MovementMemory()

        from physics import movement as M
        believed = M.perceive(memory, target, dt)
        if believed is None:
            return None
        x, y, moved = M.step(memory, pos, believed, profile, dt)
        return x, y, moved, (moved / dt if dt > 0 else 0.0)

    def keeper_save_multiplier(self, save_mult: float, shot_x: float,
                               shot_y: float, ball_x: float, ball_y: float,
                               positioning: dict) -> Tuple[float, str]:
        """Attenuate the engine's save multiplier by physical reachability.

        ``save_mult`` is the engine's own number and is left alone unless the
        keeper physically cannot get there. Because the caller already floors it
        at 1.0 (``adjusted_xg = base_prob / max(1.0, save_mult)``), pushing it
        toward 1.0 converges on the raw xG — which is the engine's own documented
        safe fallback. The gate can therefore only ever remove save credit, which
        is why it cannot inflate scorelines the way an inverted multiplier did.
        """
        factor, reason = self.keeper_feasibility(
            shot_x, shot_y, ball_x, ball_y,
            positioning.get("gk_x", 102.0), positioning.get("start_y", 34.0),
            float(positioning.get("reach", 2.5)),
            float(positioning.get("reaction_time", 0.7)),
        )
        if factor >= 1.0:
            return save_mult, reason
        return max(1.0, save_mult * factor), (
            f"{reason} (save_mult {save_mult:.3f} -> "
            f"{max(1.0, save_mult * factor):.3f})")

    def block_probability(self, base_probability: float, defender_name: str,
                          start: Tuple[float, float], end: Tuple[float, float],
                          ball_speed_mps: float,
                          lane_dist: float = 0.0) -> Tuple[float, str]:
        """Attenuate the engine's own block probability by feasibility.

        The base probability is the engine's, unchanged. The factor is applied
        to it and the result floored at the engine's own 0.02, so the gate can
        remove a physically impossible block but can never push a block
        probability below the value the simulation was calibrated around.
        """
        factor, reason = self.block_feasibility(
            defender_name, start, end, ball_speed_mps, lane_dist)
        _GATE_COUNTERS["considered"] = _GATE_COUNTERS.get("considered", 0) + 1
        if factor >= 1.0:
            _GATE_COUNTERS["feasible_unchanged"] = (
                _GATE_COUNTERS.get("feasible_unchanged", 0) + 1)
            return base_probability, reason
        adjusted = max(BLOCK_PROBABILITY_FLOOR, base_probability * factor)
        _GATE_COUNTERS["attenuated"] = _GATE_COUNTERS.get("attenuated", 0) + 1
        if factor <= 0.0:
            _GATE_COUNTERS["impossible"] = _GATE_COUNTERS.get("impossible", 0) + 1
        elif adjusted <= BLOCK_PROBABILITY_FLOOR + 1e-9:
            _GATE_COUNTERS["driven_to_floor"] = (
                _GATE_COUNTERS.get("driven_to_floor", 0) + 1)
        return adjusted, (f"{reason} (p {base_probability:.3f} -> "
                          f"{adjusted:.3f})")

    def invalidate(self, name: Optional[str] = None) -> None:
        """Forget cached profiles/velocity/momentum — call on substitution or reset.

        Without this a player who came off the bench keeps a velocity estimate
        from a name collision, which is precisely the class of stale-state bug
        this module exists to avoid.

        ``_movement`` goes with the rest for the same reason and one more: it
        holds *carried momentum* and a *held target*. A substituted player who
        inherited either would arrive at his new position already travelling at
        the pace and in the direction of a man who is no longer on the pitch.
        """
        if name is None:
            self._profiles.clear()
            self._velocity.clear()
            self._movement.clear()
        else:
            self._profiles.pop(name, None)
            self._velocity.pop(name, None)
            self._movement.pop(name, None)

    # -- reporting ------------------------------------------------------
    def report(self) -> str:
        return (f"{self.world.report()}\n"
                f"  adapter      : {len(self.history)} resolution(s), "
                f"{len(self._profiles)} profile(s) cached, "
                f"{self.speed_ceiling} m/s ceiling")

    def timeline_text(self) -> str:
        if not self.history:
            return "no passes resolved through the physics layer"
        return "\n".join(
            f"{r.kind.upper()} -> {r.receiver or '(nobody)'}\n{r.timeline_text()}"
            for r in self.history)
