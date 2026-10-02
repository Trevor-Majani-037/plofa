"""Checkpoint 39 — OFF-BALL RUN TRACKING (Gradient-style six run types).

Tracks live off-ball movement of the IN-POSSESSION team at the engine's 10 Hz
integrator tick and classifies sustained movement segments into six run types
(the taxonomy from the Gradient "Off-Ball Runs Overview" card):

    advance    In front of the ball — already ahead of the ball, driving on
    overlap    Around the outside — same flank as the ball, wider than the
               carrier, driving forward (the classic FB overlap)
    underlap   Cuts inside the carrier — was outside the ball's channel
               earlier in the segment, has crossed to the inside while advancing
    far_side   Opposite the ball — forward movement on the flank the ball is
               NOT on (ball clearly committed to one flank)
    forward    Behind to ahead — was goal-side of the ball earlier in the
               segment and is now ahead of it with enough forward gain
    support    Behind, in support — drops off behind the ball to offer a
               recycling/passing option

Transition types (underlap/forward) track segment-scoped flags
(``ever_wider``/``ever_behind``) rather than single-tick comparisons — at
10 Hz a player cannot cross a 4 m band in one tick, so tick-deltas would
only fire on ball teleports between chains.

Counting rules: a type fires only when its geometric predicate holds AND the
player has accumulated enough segment gain (forward metres for forward runs,
backward metres for support) since the last counted run of ANY type, and the
per-type cooldown (COOLDOWN_S) has elapsed. Gaps between samples larger than
GAP_S (dead chains, possession flips) reset the segment so no phantom jump
counts as a run. Non-possession samples clear the player's segment state.
Only outfield players are sampled — keepers never count.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

RUN_TYPES = (
    "advance",   # in front of the ball
    "overlap",   # around the outside
    "underlap",  # cuts inside the carrier
    "far_side",  # opposite the ball
    "forward",   # behind -> ahead
    "support",   # behind, in support
)

# The vocabulary the behaviour ENGINES actually decide in. This is deliberately
# a different namespace from RUN_TYPES above, and keeping them apart is the
# whole point of this module's second half: RunTracker observes movement
# geometry AFTER the fact, while these record the decision that CAUSED it.
# Comparing the two is how you find out whether a striker's committed in-behind
# run actually happened, or whether he set off and arrived nowhere.
#
# NOTE the deliberate word collision: "overlap" and "underlap" name BOTH a
# geometric observation and an engine decision, and both are correct - the
# fullback engine really does choose an overlap run, and the tracker really
# does observe a player overlapping. What disambiguates them is the export KEY
# namespace (`run_overlap` vs `run_intended_overlap`), not the word, so do not
# "fix" the collision by renaming one of them.
INTENDED_RUN_MODES = (
    # striker
    "behind", "hold", "box",
    # winger
    "byline", "cut",
    # fullback
    "overlap", "underlap", "tuck",
    # central midfielder
    "drop", "carry", "late", "orbit",
    # attacking midfielder
    "roam",
)


class IntendedRunRecorder:
    """Counts the run a player was told to make, at the moment he was told.

    Why this exists
    ---------------
    The live run layers decide a mode ("behind", "cut", "orbit") and then hand
    the caller a target. The mode used to be discarded at that boundary, which
    meant the only surviving evidence was a position delta - and a position
    delta cannot distinguish "ran in behind as instructed" from "drifted upfield
    because the shape compaction pulled him there". The exporter then guessed
    from geometry, which is the opposite of the information the engine already
    had.

    So: one count per DECISION, not per tick. Decisions are cached per
    (team, minute, ball-zone) by the caller, so recording inside the cache miss
    is exactly one entry per real decision. Counted runs are intentions, not
    completions - pair with RunTracker's observed types.

    THE COMMON BAR (added after the export showed the counts were not
    comparable across modes)
    ------------------------------------------------------------------
    A cache miss is a *decision*, not a *run*, and the difference is not
    academic. The CAM pocket roam is UNCONDITIONAL - it returns a target on
    every single call - so it was recorded on every (minute, zone) the CAM was
    on the pitch with the ball, 300+ times a match, four times the next
    largest mode. It was counting positionings. Worse, the modes were not on a
    common scale at all: a mode whose target happens to sit where the player
    already stands is recorded with exactly the same weight as one that sends
    him 30 m, so no mode-weighted total meant anything.

    So a decision only counts as a run when it asks the player to go
    MATERIALLY somewhere he is not: `MOVE_MIN_M`. That is one bar applied to
    every mode, and it is deliberately LOOSER than the observed tracker's own
    thresholds (`FWD_MIN` 8 m, `BACK_MIN` 6 m) because an intention should be
    easier to record than a completion. The observed taxonomy measures what
    happened; this measures what was ordered, and the two are only comparable
    because the recorder no longer inflates the easy modes.
    """

    MOVE_MIN_M: float = 6.0
    ENABLED: bool = True

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {}
        self._position: Dict[str, str] = {}
        # diagnostics - a bar nobody can inspect is a bar nobody should trust
        self.considered: int = 0
        self.skipped: int = 0
        self.unmeasured: int = 0
        self.displacements: List[float] = []

    def record(self, player: str, position: str, mode: str,
               tx: Optional[float] = None, ty: Optional[float] = None,
               from_x: Optional[float] = None,
               from_y: Optional[float] = None) -> bool:
        """Count one intended run. Returns True if it was counted.

        `tx/ty` are where the decision is sending the player and
        `from_x/from_y` where he is now. A decision with no geometry supplied
        is counted and flagged in `unmeasured` rather than dropped: silently
        refusing to count would disable the whole export for any caller that
        forgets the arguments, which is a far worse failure than an inflated
        count that is visible in the diagnostics.
        """
        if not mode:
            return False
        d: Optional[float] = None
        if None not in (tx, ty, from_x, from_y):
            d = math.hypot(tx - from_x, ty - from_y)
        self.considered += 1
        if self.ENABLED:
            if d is None:
                self.unmeasured += 1
            elif d < self.MOVE_MIN_M:
                self.skipped += 1
                return False
        if d is not None:
            self.displacements.append(d)
        slot = self.counts.setdefault(
            player, {m: 0 for m in INTENDED_RUN_MODES})
        if mode not in slot:
            # A mode the engines added later must not KeyError the export.
            slot[mode] = 0
        slot[mode] += 1
        self._position[player] = position
        return True

    def diagnostics(self) -> Dict[str, float]:
        d = sorted(self.displacements)
        return {
            "considered": self.considered,
            "counted": self.considered - self.skipped,
            "skipped_below_bar": self.skipped,
            "unmeasured": self.unmeasured,
            "median_displacement_m": (d[len(d) // 2] if d else 0.0),
            "min_displacement_m": (d[0] if d else 0.0),
        }

    def profile(self) -> Dict[str, Dict[str, int]]:
        return {n: dict(c) for n, c in self.counts.items()}

    def position_of(self, player: str) -> str:
        return self._position.get(player, "")

    def total(self, player: str) -> int:
        return sum(self.counts.get(player, {}).values())



class RunTracker:
    COOLDOWN_S: float = 6.0     # per-player per-type double-count guard
    FWD_MIN: float = 8.0        # forward gain (m) for overlap/advance/far_side
    FWD_CROSS: float = 6.0      # forward gain (m) for the behind->ahead cross
    FWD_UNDER: float = 5.0      # forward gain (m) for an underlap cut
    AHEAD_M: float = 2.0        # rel_x beyond this counts as "ahead of ball"
    WIDER_M: float = 2.0        # player outside the ball's channel by this
    BALL_SIDE_M: float = 8.0    # ball must be this far off-centre for
                                # overlap/underlap (a central ball has no
                                # "outside" to overlap)
    FLANK_M: float = 12.0       # ball must be this far off-centre for far_side
    BACK_MIN: float = 6.0       # backward gain (m) for a support drop
    GAP_S: float = 0.5          # sample gap > this resets the segment

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {}
        self._prev: Dict[str, Optional[tuple]] = {}
        self._gain: Dict[str, float] = {}
        self._back: Dict[str, float] = {}
        self._ever_behind: Dict[str, bool] = {}
        self._ever_ahead: Dict[str, bool] = {}
        self._ever_wider: Dict[str, bool] = {}
        self._last_t: Dict[str, float] = {}
        self._cd: Dict[str, Dict[str, float]] = {}
        self._possessing: str = ""
        # Jump filtering: see `set_top_speeds` and the guard in `sample`.
        self._top_speeds: Dict[str, float] = {}
        self.jumps_filtered = 0

    # ── jump-aware filtering ──────────────────────────────────────────
    #
    # On-ball players are repositioned by possession-episode traces, and keepers
    # are written directly by the build-out and block layers, so a player's
    # position legitimately changes discontinuously — up to ~99 m in one 10 Hz
    # tick. `RunTracker` accumulates forward gain between samples, so a jump is
    # indistinguishable from a very fast run and was firing the predicates:
    # measured at 12.6% of all observed runs.
    #
    # The guard refuses to OBSERVE a discontinuity. The segment is cleared and
    # the post-jump position adopted as the new reference without accumulating
    # any gain, so a run can never be counted across a teleport and the teleport
    # itself is never counted as movement. The alternative — counting it and
    # letting a cooldown absorb it — leaves a segment whose geometry is
    # meaningless.
    #
    # Speeds come from the engine (`5.0 + pace*0.042` m/s) and are injected, not
    # recomputed here: the units of `top_speed_mpm` are a per-minute STEP
    # distance, not a speed, and guessing that produced two completely wrong
    # contamination figures before this was wired properly.
    JUMP_FILTER_ENABLED: bool = True
    JUMP_TOL: float = 1.15   # slack for the integrator's straight-line steps

    def set_top_speeds(self, speeds: Dict[str, float]) -> None:
        """Inject each player's top speed in m/s, from MatchEngine."""
        self._top_speeds = dict(speeds or {})

    # ── public API ────────────────────────────────────────────────
    def profile(self) -> Dict[str, Dict[str, int]]:
        return {n: dict(c) for n, c in self.counts.items()}

    def _clear_segment(self, name: str) -> None:
        self._prev[name] = None
        self._gain[name] = 0.0
        self._back[name] = 0.0
        self._ever_behind[name] = False
        self._ever_ahead[name] = False
        self._ever_wider[name] = False

    def sample(
        self, t: float, name: str, px: float, py: float,
        bx: float, by: float, attacks_right: bool,
        team: str, has_ball: bool,
    ) -> None:
        if not has_ball:
            self._clear_segment(name)
            return
        if self._possessing != team:
            self._possessing = team
            for other in list(self._prev.keys()):
                self._clear_segment(other)
        self.counts.setdefault(name, {k: 0 for k in RUN_TYPES})
        prev_t = self._last_t.get(name)
        if prev_t is None or (t - prev_t) > self.GAP_S:
            self._clear_segment(name)
        # Capture the elapsed time BEFORE overwriting _last_t. The jump guard
        # below needs the interval since the PREVIOUS sample; reading it after
        # the assignment gives t - t == 0, so `0 < dt` is never true and the
        # guard is inert. That is not hypothetical: a controlled replay of a
        # recorded feed showed jumps_filtered == 0 with 200+ non-physical
        # samples present, because of exactly this.
        dt_prev = (t - prev_t) if prev_t is not None else 0.0
        self._last_t[name] = t

        s = 1.0 if attacks_right else -1.0
        rel_x = (px - bx) * s
        prev = self._prev.get(name)
        if prev is None:
            self._prev[name] = (rel_x, px, py, by)
            if rel_x < -self.AHEAD_M:
                self._ever_behind[name] = True
            if rel_x > self.AHEAD_M:
                self._ever_ahead[name] = True
            if abs(py - 34.0) > abs(by - 34.0) + self.WIDER_M:
                self._ever_wider[name] = True
            return
        prev_rel, ppx, ppy, pby = prev
        # ── jump guard: never observe a discontinuity as movement ──
        if self.JUMP_FILTER_ENABLED and self._top_speeds:
            _vmax = self._top_speeds.get(name, 0.0)
            if 0.0 < dt_prev <= 0.35 and _vmax > 0.0:
                _step = math.hypot(px - ppx, py - ppy)
                if _step > _vmax * dt_prev * self.JUMP_TOL:
                    self.jumps_filtered += 1
                    # Segment geometry is void across a teleport: drop the
                    # accumulators, then adopt the new position as the
                    # reference WITHOUT banking any gain from the jump.
                    self._clear_segment(name)
                    self._prev[name] = (rel_x, px, py, by)
                    self._last_t[name] = t
                    return
        dfwd = (px - ppx) * s
        if dfwd > 0.0:
            self._gain[name] = self._gain.get(name, 0.0) + dfwd
        elif dfwd < 0.0:
            self._back[name] = self._back.get(name, 0.0) - dfwd
        gain = self._gain.get(name, 0.0)
        back = self._back.get(name, 0.0)

        side = 1.0 if py > 34.0 else -1.0
        bside = 1.0 if by > 34.0 else -1.0
        ball_off = abs(by - 34.0)
        w_now = abs(py - 34.0) > abs(by - 34.0) + self.WIDER_M
        ahead_now = rel_x > self.AHEAD_M
        behind_now = rel_x < -self.AHEAD_M
        if behind_now:
            self._ever_behind[name] = True
        if ahead_now:
            self._ever_ahead[name] = True
        if w_now:
            self._ever_wider[name] = True
        ever_behind = self._ever_behind.get(name, False)
        ever_ahead = self._ever_ahead.get(name, False)
        ever_wider = self._ever_wider.get(name, False)
        cd = self._cd.setdefault(name, {})

        def ready(kind: str) -> bool:
            stamp = cd.get(kind)
            return stamp is None or (t - stamp) >= self.COOLDOWN_S

        hit: Optional[str] = None
        # NET DIRECTION is what separates the two GENERAL cases.
        #
        # Before, `support` sat last in the chain with the loosest predicate in
        # the file (`behind_now and back >= 6m`) while the five forward types
        # above it all required ball-side or width conditions that rarely hold.
        # Being last, it became a catch-all: measured on a real match, support
        # was 1421 of 1497 observed runs (95%) and `overlap` fired once.
        #
        # The cause is that `gain` and `back` are independent accumulators that
        # both survive until a run fires, so a player oscillating around the
        # ball accumulates backward metres while also accumulating forward
        # metres, and whichever accumulator crossed its line first won. A
        # segment that is net FORWARD could therefore be filed as a support
        # drop, which is simply the wrong answer.
        #
        # So the two general cases now require the segment's NET direction to
        # match the label. The four specific cases (underlap / overlap /
        # forward / far_side) already imply their own direction and are
        # untouched.
        net = gain - back
        if (side == bside and ball_off >= self.BALL_SIDE_M
                and ever_wider and not w_now
                and gain >= self.FWD_UNDER and ready("underlap")):
            hit = "underlap"                     # cut inside the carrier
        elif (side == bside and ball_off >= self.BALL_SIDE_M
              and w_now and gain >= self.FWD_MIN and ready("overlap")):
            hit = "overlap"                      # around the outside
        elif (ever_behind and ahead_now and gain >= self.FWD_CROSS
              and ready("forward")):
            hit = "forward"                      # behind -> ahead
        elif (side != bside and ball_off >= self.FLANK_M
              and gain >= self.FWD_MIN and ready("far_side")):
            hit = "far_side"                     # run opposite the ball
        elif ahead_now and gain >= self.FWD_MIN and net > 0.0 \
                and ready("advance"):
            hit = "advance"                      # in front of the ball
        elif (ever_ahead and behind_now and back >= self.BACK_MIN and net < 0.0
              and ready("support")):
            hit = "support"                      # drop behind in support

        if hit is not None:
            self.counts[name][hit] += 1
            cd[hit] = t
            self._clear_segment(name)
            self._prev[name] = (rel_x, px, py, by)
            return
        self._prev[name] = (rel_x, px, py, by)
