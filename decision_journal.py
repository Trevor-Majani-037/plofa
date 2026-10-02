"""Decision journal + ``why(player, minute)`` query (audit Step 7).

Generalises the on-ball "record_trace" into an append-only decision journal:

  world state, perceived state, sensor values, DNA, tactics, probs,
  alternatives, chosen intent, result, immediate value, future value,
  final consequence.

``what the player saw and picked, and how it turned out`` is a **read-only,
opt-in** sidecar: nothing in the production match path calls it unless a
harness passes an explicit journal. Default (no journal enabled) the on-ball
decision returns byte-identical output -- the journal adds zero behaviour.

Design intent (mirrors the audit's observability item): the on-ball brain's
``record_trace=True`` trace (candidates + perceived, `decision_brain.decide`)
is the *input seam*; ``journalize_trace`` lifts that existing shape into a
journal entry without re-running perception. ``why()`` reconstructs one
decision at (player, minute) into a human-readable + structured reply using
the *same* fields the brain already journaled -- no template, no behaviour.

Determinism: journal entries carry no RNG; ``why()`` is a pure function of
recorded fields. Equality of two journals implies equality of their ``why()``
output, so a byte-identical-on-ball regression is trivially assertable.
"""

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple


def _round3(x: float) -> float:
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return 0.0


def _field(entry: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Tolerant accessor: accepts both dotted and flat journal keys."""
    if key in entry:
        return entry[key]
    dotted = key.replace("_", ".")
    if dotted in entry:
        return entry[dotted]
    return default


class DecisionJournal:
    """Append-only, deterministic decision journal.

    Entries are added via :meth:`record` (or the :func:`journalize_trace`
    adapter over an existing on-ball trace). -- is queried with
    :meth:`why` / :meth:`since`.
    """

    __slots__ = ("_entries", "_by_player")

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._by_player: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def record(self, entry: Mapping[str, Any]) -> "DecisionJournal":
        """Append a validated journal entry. Never mutates match behaviour."""
        field_order = (
            "minute",
            "player",
            "world_state",
            "perceived_state",
            "sensor_values",
            "dna",
            "tactics",
            "probs",
            "alternatives",
            "chosen_intent",
            "result",
            "immediate_value",
            "future_value",
            "final_consequence",
        )
        norm = {k: entry[k] for k in field_order if k in entry}
        # tolerate a couple of common aliases (so journalize_trace maps cleanly)
        for alias, canonical in (("player_id", "player"), ("minute_rev", "minute")):
            if canonical not in norm and alias in entry:
                norm[canonical] = entry[alias]
        if "minute" in norm:
            try:
                norm["minute"] = float(norm["minute"])
            except (TypeError, ValueError):
                norm["minute"] = 0.0
        idx = len(self._entries)
        self._entries.append(norm)
        player = str(norm.get("player", "?"))
        self._by_player.setdefault(player, []).append(idx)
        return self

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DecisionJournal):
            return NotImplemented
        return self._entries == other._entries

    def __iter__(self):
        return iter(self._entries)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def justify(self, player: str, minute: float, *,
                tolerance: float = 0.0) -> Optional[Dict[str, Any]]:
        """Most recent journal entry for ``player`` at (or just before) ``minute``.

        Returns the structured entry dict, or ``None`` if nothing matching.
        """
        minute = float(minute)
        idxs = self._by_player.get(str(player), [])
        best: Optional[Tuple[int, Dict[str, Any]]] = None
        for idx in idxs:
            e = self._entries[idx]
            m = float(_field(e, "minute", -1.0))
            if m <= minute + tolerance and (best is None or m >= best[0]):
                best = (m, e)
        return best[1] if best else None

    def why(self, player: str, minute: float, *,
            tolerance: float = 0.0) -> Dict[str, Any]:
        """``why(player, minute)`` -- explain one recorded decision.

        Returns ``{"answered": True, "explanation": str, "entry": {...}}`` or
        ``{"answered": False, "explanation": "...", "entry": None}``.
        """
        e = self.justify(player, minute, tolerance=tolerance)
        if e is None:
            return {
                "answered": False,
                "explanation": (
                    "no journal entry for {0!r} at minute {1}".format(player, minute)
                ),
                "entry": None,
            }
        explanation = self._explain(e)
        return {"answered": True, "explanation": explanation, "entry": e}

    def since(self, minute: float) -> List[Dict[str, Any]]:
        """All entries with ``minute >= given`` (useful for realism sweeps)."""
        minute = float(minute)
        return [e for e in self._entries if float(_field(e, "minute", -1.0)) >= minute]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _explain(e: Mapping[str, Any]) -> str:
        player = str(_field(e, "player", "?"))
        minute = float(_field(e, "minute", 0.0))
        chosen = str(_field(e, "chosen_intent", "RECYCLE"))
        immedi = _round3(float(_field(e, "immediate_value", 0.0) or 0.0))
        future = _round3(float(_field(e, "future_value", 0.0) or 0.0))
        consequence = str(_field(e, "final_consequence", "n/a"))
        result = str(_field(e, "result", "n/a"))

        alts = _field(e, "alternatives", [])
        alt_txt = ""
        if isinstance(alts, Sequence) and not isinstance(alts, (str, bytes)):
            names = []
            for a in alts:
                if isinstance(a, Mapping):
                    n = str(_field(a, "intent", _field(a, "name", "?")))
                    names.append(n)
                else:
                    names.append(str(a))
            if names:
                alt_txt = " (alternatives: {0})".format(", ".join(names[:8]))

        return (
            "minute {minute:g} · {player} chose {chosen}{alt}; "
            "immediate value {immediate}, future value {future}; "
            "result: {result}; final consequence: {consequence}".format(
                minute=minute, player=player, chosen=chosen, alt=alt_txt,
                immediate=immedi, future=future, result=result,
                consequence=consequence,
            )
        )

    def to_records(self) -> List[Dict[str, Any]]:
        """Stable dict dump (regression-friendly)."""
        return [dict(e) for e in self._entries]


def journalize_trace(
    trace: Mapping[str, Any],
    *,
    world_state: Optional[Mapping[str, Any]] = None,
    perceived_state: Optional[Mapping[str, Any]] = None,
    dna: Optional[Mapping[str, Any]] = None,
    tactics: Optional[Mapping[str, Any]] = None,
    future_value: Optional[float] = None,
    final_consequence: Optional[str] = None,
) -> Dict[str, Any]:
    """Lift an existing on-ball ``record_trace=True`` trace into a journal entry.

    Reuses the *same* candidate/perceived fields the brain already produces
    (no re-perception, no behaviour). Fields not present in ``trace`` fall
    back to the (optional) overrides, so a caller can enrich the journal with
    DNA / tactics / value / consequence without touching the decide path.
    """
    player = str(trace.get("player", "?"))
    minute = float(trace.get("minute", 0.0) or 0.0)

    # candidates -> alternatives (intent + objective value + risk)
    alternatives = _field(trace, "candidates", [])
    alts: List[Dict[str, Any]] = []
    if isinstance(alternatives, Sequence) and not isinstance(alternatives, (str, bytes)):
        for c in alternatives:
            if isinstance(c, Mapping):
                alts.append(
                    {
                        "intent": str(c.get("intent", c.get("name", "?"))),
                        "objective_value": _round3(float(c.get("objective_value", 0.0) or 0.0)),
                        "risk": _round3(float(c.get("risk", 0.0) or 0.0)),
                    }
                )
            else:
                alts.append({"intent": str(c)})

    # perceived -> probs/alternatives detail already carries probability
    perceived = _field(trace, "perceived", [])
    probs: Dict[str, float] = {}
    if isinstance(perceived, Sequence) and not isinstance(perceived, (str, bytes)):
        for p in perceived:
            if isinstance(p, Mapping):
                intent = str(p.get("intent", "?"))
                val = p.get("perceived_value", p.get("probability", 0.0))
                probs[intent] = _round3(float(val if val is not None else 0.0))

    chosen_idx = trace.get("chosen_idx", 0)
    chosen_intent = str(trace.get("chosen_intent", ""))
    if not chosen_intent and alts:
        try:
            chosen_intent = alts[int(chosen_idx)]["intent"]
        except (TypeError, ValueError, IndexError):
            chosen_intent = alts[0]["intent"] if alts else "RECYCLE"

    return {
        "minute": _round3(minute),
        "player": player,
        "world_state": dict(world_state) if world_state else {},
        "perceived_state": dict(perceived_state) if perceived_state else {},
        "sensor_values": list(trace.get("sensor_vector", [])),
        "dna": dict(dna) if dna else {},
        "tactics": dict(tactics) if tactics else {},
        "probs": probs,
        "alternatives": alts,
        "chosen_intent": chosen_intent,
        "result": str(trace.get("result", "n/a")),
        "immediate_value": _round3(
            float(trace.get("immediate_value", p.get("perceived_value", 0.0) if False else 0.0) or 0.0)
        ),
        "future_value": _round3(float(future_value if future_value is not None else 0.0)),
        "final_consequence": str(final_consequence or "n/a"),
    }


def new_journal() -> DecisionJournal:
    """Convenience factory (keeps production import paths short)."""
    return DecisionJournal()
