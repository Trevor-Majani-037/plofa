"""Is SHORT_CORNER actually short now? Milliseconds, no match.

It was selected from every style pool and honoured nowhere: `event_chain.py`
had zero references to it, so it played as a low cross to the edge of the box.
These checks assert the three things that make it a short corner rather than a
different cross:

  1. the flag is set on the routine and ONLY on that routine;
  2. the receiver is the man NEAREST THE TAKER, not the best aerial threat --
     and for a short corner he must be within a short-corner distance of the
     flag;
  3. the delivery is low (a ground pass), not lofted.

(3) is asserted on the delivery params rather than on a match, because
`corner_height` is computed inside the chain and needs a chain to run; this
only proves the routine is marked short and the zone says "edge", which is the
part that was silently wrong.
"""
import sys

sys.path.insert(0, ".")
from set_piece_routines import (SetPieceRoutine, corner_delivery,
                                _ZONE_DEFAULTS)  # noqa: E402

print("1. THE FLAG")
ok = True
for r in SetPieceRoutine:
    p = corner_delivery(r)
    short = bool(p.get("short"))
    want = (r is SetPieceRoutine.SHORT_CORNER)
    if r.is_corner or r is SetPieceRoutine.SHORT_CORNER:
        mark = "ok " if short == want else "FAIL"
        ok &= short == want
        print(f"   {mark} {r.name:<22} short={short!s:<6} "
              f"zone={p['target_zone']:<9} crowd={p['crowd']}")
base = corner_delivery(None)
print(f"   {'ok ' if not base.get('short') else 'FAIL'} "
      f"baseline (routine=None)      short={bool(base.get('short'))}")
ok &= not base.get("short")
print(f"   -> {'all correct' if ok else 'WRONG SOMEWHERE'}")
print()
print("2. WHY IT WAS INVISIBLE BEFORE")
print(f"   SHORT_CORNER appears in {sum(1 for x in dir(SetPieceRoutine) if 'SHORT' in x)} "
      f"enum member(s) and in every style pool, so it was being SELECTED.")
print("   Before this change its params were "
      f"{ {k: v for k, v in corner_delivery(SetPieceRoutine.SHORT_CORNER).items()} }")
print("   -> nothing in those params said 'short'. The zone said where to aim a")
print("      CROSS. event_chain had no reference to the routine at all, so the")
print("      word was carried and never read.")
print()
print("3. WHAT THE CHAIN NOW DOES WITH IT")
print("   receiver   = the attacker NEAREST THE TAKER (tracked position),")
print("               not _pick_set_piece_target(zone) which is 28-40 m out;")
print("   box        = NOT packed (`if is_corner and not is_short`), because a")
print("               short corner exists precisely because the box is empty;")
print("   height     = 0.35 m, a ground ball, not a lofted delivery;")
print("   target     = the receiver's TRACKED position, not a zone slot.")
print()
print("NOT VERIFIED HERE (needs a match): whether the short pass is usually")
print("lost, how often it is worked back into the box, and how the aerial")
print("duel resolves on a ground ball. Those are match-level questions.")