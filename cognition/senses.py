"""FOV gating — which actors a player can actually SEE at the decision instant.

Ported from TOLAND's ``VisionSystem`` (D:\\TOLAND FOOTBALL FEDERATION\\
football_sim\\cognition\\senses.py) and adapted to PLOFA's geometry: the
original worked on live world entities; here player coordinates come from
the ``position_engine`` (the engine's single source of positional truth),
so the gate can be applied BEFORE the neural sensor block is built.

Correctness argument: PLOFA composes each sensor from the teammate/defender
lists handed to ``decide()``.  If an actor is outside the player's view
radius (or beyond the teammate/opponent slots available), it simply is NOT
in the filtered list — so the role block / marker lanes read "nothing
there".  That is a genuine FOV: a player cannot pick a teammate he cannot
see, and an unseen defender puts no pressure on the option.  This composes
with the existing perception layer (pressure / fatigue degradation), which
continues to operate on whatever scene survives the gate.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


class VisionSystem:
    """The player's subjective field of view.

    ``view_radius`` is in pitch units (PLOFA uses ~100 x 60, so 25.0 is a
    quarter of the pitch — the same feel as TOLAND's 25 m).  Only the
    nearest ``max_teammates`` teammates and ``max_opponents`` opponents
    inside the radius are perceived, nearest first (attention is limited).
    """

    def __init__(
        self,
        view_radius: float = 25.0,
        max_teammates: int = 2,
        max_opponents: int = 2,
    ):
        self.view_radius = view_radius
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents

    def _pos(
        self,
        actor: Any,
        position_engine: Any,
        fallback: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Resolve an actor's live coordinates, tolerating dead/absent
        position engines (fallback keeps the gate permissive, never
        crashing)."""
        if position_engine is None:
            return fallback
        try:
            return position_engine.get_position(getattr(actor, "name", ""))
        except Exception:
            return fallback

    def filter_frame(
        self,
        player: Any,
        x: float,
        y: float,
        teammates: List[Any],
        defenders: List[Any],
        position_engine: Any,
    ) -> Tuple[List[Any], List[Any]]:
        """Return ``(visible_teammates, visible_defenders)`` — the actors
        inside view radius, nearest-first, capped by the attention slots.

        Goalkeepers are excluded from both lists: the role blocks and lane
        estimators already skip GK markers, and burning a limited slot on
        one would push a real outfield option out of view.
        """
        name = getattr(player, "name", "?")

        def _visible(actors: List[Any], cap: int, skip_gk: bool) -> List[Any]:
            seen: List[Tuple[float, Any]] = []
            for a in actors or []:
                aname = getattr(a, "name", "")
                if aname == name:
                    continue
                if skip_gk and getattr(a, "position", "") == "GK":
                    continue
                ax, ay = self._pos(a, position_engine, (x + 999.0, y))
                d = math.hypot(ax - x, ay - y)
                if d <= self.view_radius:
                    seen.append((d, a))
            seen.sort(key=lambda pair: (pair[0], getattr(pair[1], "name", "")))
            return [a for _, a in seen[:cap]]

        visible_team = _visible(teammates, self.max_teammates, True)
        visible_def = _visible(defenders, self.max_opponents, True)
        return visible_team, visible_def