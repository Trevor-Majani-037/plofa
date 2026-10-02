"""
physics/collision.py — who gets there first.
=============================================

This is the module the whole package exists to enable, and it is the smallest.

§12 lays it out as nine steps: plan the ball, work out where it will be and
when, identify the candidates, work out their reaction times, work out their
movement, work out when each can physically arrive, compare, resolve, emit an
event. Steps 1, 3 and 4 belong to the caller; this module owns 6, 7 and 8.

The comparison itself is two lines. The value is that steps 6 and 7 are
*physical* rather than probabilistic, which is what §13 asks for: the outcome
of a through ball should emerge from the state of the world, not from
``random.random() < success_probability``.

Ties
----
A dead heat is possible and is not resolved by a coin flip. When two players
are within :data:`TIE_WINDOW` seconds the result is reported as a tie, because
racing a duel to a probabilistic winner is exactly the habit §13 tells us to
leave behind. A caller who wants to break the tie should do it with a football
reason — a deflection, a header, the keeper claiming — not with a die.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Two arrivals within this many seconds are a tie. Roughly 30 ms: well inside
#: the noise of the model, and narrow enough that genuine races are decided.
TIE_WINDOW = 0.03

#: How long a ball may lie with nobody near it before it counts as unclaimed.
#:
#: This exists because "nobody arrived BEFORE the ball" is not the same thing as
#: "nobody got there at all". A through ball played into space typically reaches
#: its target before the attacker does — he is being asked to catch up with it —
#: and that is a loose ball he collects, not a failure. Calling it unclaimed
#: reported "nobody reaches the ball before it arrives" for a pass the striker
#: plainly won, and printed an empty winner.
#:
#: 1.5 s is roughly how long a player will chase a ball he is not close to
#: before it is honestly out of play.
LOOSE_BALL_WINDOW = 1.5


@dataclass(frozen=True)
class Candidate:
    """One player's claim on a ball, already reduced to a time."""

    name: str
    #: seconds between the ball being struck and this player reaching it
    arrival: float
    team: str = ""
    position: str = ""
    #: how far they had to travel, for reporting
    distance: float = 0.0
    #: True when the player is the intended receiver
    intended: bool = False
    #: the reaction component of ``arrival``, in seconds. Carried explicitly
    #: because the reaction/movement split is the interesting part of the
    #: timeline and cannot be recovered from ``arrival`` alone.
    reaction: float = 0.0

    def __post_init__(self) -> None:
        if self.arrival < 0:
            raise ValueError(
                f"{self.name}: arrival time cannot be negative, got "
                f"{self.arrival}")
        if self.reaction < 0:
            raise ValueError(
                f"{self.name}: reaction cannot be negative, got {self.reaction}")
        if self.reaction > self.arrival + 1e-9:
            raise ValueError(
                f"{self.name}: reaction {self.reaction}s exceeds arrival "
                f"{self.arrival}s")

    @property
    def travel(self) -> float:
        """Seconds spent actually moving, i.e. arrival minus reaction."""
        return max(0.0, self.arrival - self.reaction)

    def to_dict(self) -> dict:
        return {"name": self.name, "arrival": round(self.arrival, 4),
                "team": self.team, "position": self.position,
                "distance": round(self.distance, 2), "intended": self.intended,
                "reaction": round(self.reaction, 4),
                "travel": round(self.travel, 4)}


@dataclass(frozen=True)
class ContestResult:
    """The outcome of a race for a ball."""

    winner: Optional[Candidate]
    #: every candidate, fastest first
    ranked: Tuple[Candidate, ...]
    ball_arrival: float
    #: seconds between the winner and the ball arriving
    margin: float = 0.0
    tied: bool = False
    #: set when nobody can reach the ball before it lands
    unclaimed: bool = False
    reason: str = ""

    @property
    def contested(self) -> bool:
        return len(self.ranked) > 1

    @property
    def loose(self) -> bool:
        """True when the ball reached its target before anyone got there.

        A distinct and very common football situation: the pass beats the
        defence to the space, and the attacker collects it having arrived after
        the ball. Not an interception, not a failure — a loose ball.
        """
        if self.unclaimed or not self.ranked:
            return False
        return self.ranked[0].arrival > self.ball_arrival

    @property
    def leader(self) -> Optional[Candidate]:
        """Fastest candidate, even when the race is tied or unclaimed.

        Exposed so reporting can say "X was closest" without a caller having to
        treat a tie as an error.
        """
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict:
        return {
            "winner": self.winner.to_dict() if self.winner else None,
            "ranked": [c.to_dict() for c in self.ranked],
            "ball_arrival": round(self.ball_arrival, 4),
            "margin": round(self.margin, 4),
            "tied": self.tied,
            "unclaimed": self.unclaimed,
            "reason": self.reason,
        }


def arrival_contest(ball_arrival: float,
                    candidates: Sequence[Candidate]) -> ContestResult:
    """Resolve who physically reaches the ball first.

    ``ball_arrival`` is measured on the same clock as each candidate's
    ``arrival``, so both are "seconds since the ball was struck". Comparing
    them directly answers the only question that matters: does anyone get there
    before it does, and if so who.

    A candidate arriving after the ball has landed is still ranked — they may
    be the next to play it, and the margin is informative — but the result
    reports ``unclaimed`` when nobody beats the ball, because "the ball
    arrived and sat there" is a real and distinct football situation.
    """
    if ball_arrival < 0:
        raise ValueError(f"ball_arrival cannot be negative, got {ball_arrival}")

    ranked = tuple(sorted(candidates, key=lambda c: (c.arrival, c.name)))
    if not ranked:
        return ContestResult(
            winner=None, ranked=(), ball_arrival=ball_arrival, unclaimed=True,
            reason="no candidates")

    first = ranked[0]
    tied = len(ranked) > 1 and abs(ranked[1].arrival - first.arrival) <= TIE_WINDOW

    # Unclaimed means nobody is realistically getting there — not merely that
    # the ball beat them. See LOOSE_BALL_WINDOW.
    if first.arrival - ball_arrival > LOOSE_BALL_WINDOW:
        return ContestResult(
            winner=None, ranked=ranked, ball_arrival=ball_arrival,
            margin=ball_arrival - first.arrival, unclaimed=True,
            reason=(f"nearest player is {first.arrival - ball_arrival:.2f}s "
                    f"away; the ball is not being contested"))

    winner = None if tied else first
    if first.arrival > ball_arrival:
        reason = (f"{first.name} arrives {first.arrival - ball_arrival:.3f}s "
                  f"after the ball — a loose ball he collects")
    else:
        reason = (f"{first.name} arrives "
                  f"{ball_arrival - first.arrival:.3f}s early")
    return ContestResult(
        winner=winner,
        ranked=ranked,
        ball_arrival=ball_arrival,
        margin=ball_arrival - first.arrival,
        tied=tied,
        unclaimed=False,
        reason="tie" if tied else reason,
    )


def resolve_receiver(result: ContestResult,
                    intended_name: str = "") -> str:
    """What the caller should emit: a name, ``""`` for nobody, ``"TIE"``.

    Kept separate from the contest so the *reasoning* (physics) stays distinct
    from the *event wording* (the engine's job, per the brain/physics split).

    ``intended_name`` is advisory: it lets a caller label the result as the
    intended receiver completing the pass, but it never changes who won. A
    defender who genuinely gets there first has won, whatever the passer
    intended.
    """
    if result.unclaimed:
        return ""
    if result.tied:
        return "TIE"
    if result.winner is None:
        return ""
    return result.winner.name

def margin_table(result: ContestResult) -> Dict[str, float]:
    """Arrival time of every candidate relative to the ball, for reporting."""
    return {c.name: round(c.arrival - result.ball_arrival, 4)
            for c in result.ranked}


def race_preview(candidates: Sequence[Candidate]) -> str:
    """A readable preview of a race. Used by tests and by the demo."""
    if not candidates:
        return "no candidates"
    rows = sorted(candidates, key=lambda c: c.arrival)
    width = max(len(c.name) for c in rows)
    lines = [f"  {'player'.ljust(width)}  {'team':<6} {'arrive':>8}  {'dist':>7}"]
    for c in rows:
        lines.append(f"  {c.name.ljust(width)}  {c.team:<6} {c.arrival:>8.3f}  "
                     f"{c.distance:>7.1f}")
    return "\n".join(lines)
