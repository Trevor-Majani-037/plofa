"""Close the last two holes the gate found.

GATE SAID (5 cases, 2 failures):
  P1 argument order : FAIL on the identical-spot tie
  P2 resolution     : FAIL on "defender out on the flight line"

CAUSE 1 — I moved the bug instead of fixing it. The per-player update was
`if prev is None or entry[:3] < prev[:3]`, so on an exact tie the FIRST player
iterated won — and iteration is `attacking + defending`, i.e. the attacker
again. The `id(player)` I put at position 4 was never reached, because a strict
`<` on an exact tie never replaces. Fixed by including a stable, attacker-blind
identity in the compared key. `id()` is not good enough: it is not reproducible
across processes, so it would make a match non-reproducible for a reason the
project cannot explain. The player's NAME is used instead — deterministic,
symmetric between sides, and stable across runs.

CAUSE 2 — the feasibility test `arrival > time_s` is evaluated against the
sampled times, so a contestant's score still depends on how finely the flight
was sliced. That is why "defender out on the flight line" returned ATTACKER at
0.1 s and DEFENDER at 0.01 s. The scoring grid is now FIXED and independent of
the caller's `sample_step`, which is retained only as a floor. This function is
pure geometry called a handful of times per match; 240 samples x ~12 players is
a few thousand trapezoid evaluations and costs nothing next to a match tick.
The point is that the ANSWER must be a function of the players, not of the
arithmetic the caller happened to ask for.
"""
import pathlib

p = pathlib.Path("geometry_engine.py")
src = p.read_text(encoding="utf-8")

# ── 1. fixed scoring grid ──
OLD_STEPS = ("    steps = max(1, int(math.ceil(flight.duration / "
             "sample_step)))\n"
             "\n"
             "    # ── PER-CONTESTANT SCORE, NOT A LIST-ORDER TIE")
NEW_STEPS = ("    # A FIXED scoring grid. It used to be derived from `sample_step`,\n"
             "    # which made the winner a function of the caller's arithmetic: the\n"
             "    # same duel returned ATTACKER at 0.1 s and DEFENDER at 0.01 s,\n"
             "    # because the feasibility test `arrival > time_s` is evaluated\n"
             "    # against the sampled times. `sample_step` is still honoured, but\n"
             "    # only as a FLOOR -- it can ask for more resolution, never less.\n"
             "    steps = max(AERIAL_SCORE_STEPS,\n"
             "                int(math.ceil(flight.duration / sample_step)))\n"
             "\n"
             "    # ── PER-CONTESTANT SCORE, NOT A LIST-ORDER TIE")
assert src.count(OLD_STEPS) == 1, f"steps anchor {src.count(OLD_STEPS)}"
src = src.replace(OLD_STEPS, NEW_STEPS, 1)

# ── 2. the tie must be broken by something, not by iteration order ──
OLD_ENTRY = """            entry = (arrival, -max_reach, -ahead, id(player), player,
                     player in attacking_players, time_s, point,
                     jump_start, max_reach)
            prev = per_player.get(id(player))
            if prev is None or entry[:3] < prev[:3]:
                per_player[id(player)] = entry

    candidates = sorted(per_player.values(), key=lambda item: item[:3])"""
NEW_ENTRY = """            # The 4th term is the player's NAME, and it is load-bearing:
            # an exact tie must not be resolved by which man happened to be
            # iterated first, because iteration is `attacking + defending`.
            # Without it the per-player update below never REPLACES on a tie
            # (a strict `<`), so the attacker silently kept every dead heat
            # even after the sort key was fixed. The name is used rather than
            # `id()` because `id()` is not reproducible across processes, and
            # a match that varied run to run for a tie-break would be worse
            # than the bias it removed.
            entry = (arrival, -max_reach, -ahead, _pname(player), player,
                     player in attacking_players, time_s, point,
                     jump_start, max_reach)
            prev = per_player.get(_pname(player))
            if prev is None or entry[:4] < prev[:4]:
                per_player[_pname(player)] = entry

    candidates = sorted(per_player.values(), key=lambda item: item[:4])"""
assert src.count(OLD_ENTRY) == 1, f"entry anchor {src.count(OLD_ENTRY)}"
src = src.replace(OLD_ENTRY, NEW_ENTRY, 1)

# ── the two names the new code needs ──
OLD_HELP = "def resolve_aerial_delivery(\n"
NEW_HELP = (
    "# A corner or a cross is resolved in a few milliseconds and called a"
    " handful of\n"
    "# times a match, so the flight is scored on a fixed fine grid. This is"
    " the\n"
    "# floor on `sample_step`, not a replacement for it -- see the scoring"
    " loop.\n"
    "AERIAL_SCORE_STEPS = 240\n\n\n"
    "def _pname(player: \"MovingPlayer\") -> str:\n"
    "    \"\"\"A stable, side-agnostic identity for tie-breaking.\"\"\"\n"
    "    inner = getattr(player, \"player\", None)\n"
    "    return str(getattr(inner, \"name\", id(player)))\n\n\n"
    "def resolve_aerial_delivery(\n")
assert src.count(OLD_HELP) == 1, f"helper anchor {src.count(OLD_HELP)}"
src = src.replace(OLD_HELP, NEW_HELP, 1)

p.write_text(src, encoding="utf-8")
print("geometry_engine.py:")
print("  fixed scoring grid   :", "AERIAL_SCORE_STEPS = 240" in src)
print("  sample_step is floor :", "int(math.ceil(flight.duration / "
                                 "sample_step)))" in src)
print("  name tiebreak        :", "_pname(player)" in src)
print("  compared on 4 terms  :", "entry[:4] < prev[:4]" in src)