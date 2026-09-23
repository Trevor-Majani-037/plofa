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

from typing import Dict, Optional

RUN_TYPES = (
    "advance",   # in front of the ball
    "overlap",   # around the outside
    "underlap",  # cuts inside the carrier
    "far_side",  # opposite the ball
    "forward",   # behind -> ahead
    "support",   # behind, in support
)


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
        self._ever_wider: Dict[str, bool] = {}
        self._last_t: Dict[str, float] = {}
        self._cd: Dict[str, Dict[str, float]] = {}
        self._possessing: str = ""

    # ── public API ────────────────────────────────────────────────
    def profile(self) -> Dict[str, Dict[str, int]]:
        return {n: dict(c) for n, c in self.counts.items()}

    def _clear_segment(self, name: str) -> None:
        self._prev[name] = None
        self._gain[name] = 0.0
        self._back[name] = 0.0
        self._ever_behind[name] = False
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
        self._last_t[name] = t

        s = 1.0 if attacks_right else -1.0
        rel_x = (px - bx) * s
        prev = self._prev.get(name)
        if prev is None:
            self._prev[name] = (rel_x, px, py, by)
            if rel_x < -self.AHEAD_M:
                self._ever_behind[name] = True
            if abs(py - 34.0) > abs(by - 34.0) + self.WIDER_M:
                self._ever_wider[name] = True
            return
        prev_rel, ppx, ppy, pby = prev
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
        if w_now:
            self._ever_wider[name] = True
        ever_behind = self._ever_behind.get(name, False)
        ever_wider = self._ever_wider.get(name, False)
        cd = self._cd.setdefault(name, {})

        def ready(kind: str) -> bool:
            stamp = cd.get(kind)
            return stamp is None or (t - stamp) >= self.COOLDOWN_S

        hit: Optional[str] = None
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
        elif ahead_now and gain >= self.FWD_MIN and ready("advance"):
            hit = "advance"                      # in front of the ball
        elif behind_now and back >= self.BACK_MIN and ready("support"):
            hit = "support"                      # drop behind in support

        if hit is not None:
            self.counts[name][hit] += 1
            cd[hit] = t
            self._clear_segment(name)
            self._prev[name] = (rel_x, px, py, by)
            return
        self._prev[name] = (rel_x, px, py, by)
