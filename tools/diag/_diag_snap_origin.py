"""Is the on-ball snap a CORRECTION or a FABRICATION? (open item #1)

The snap at `match_engine.py:5764/5772` writes the coordinate of the event the
player just performed onto that player's tracked position. 91% of the >=12 m
`record_touch` jumps in a match are these three lines. When one moves a man 29 m
in a tick there are two readings this project has never been able to separate:

  (a) CORRECTION - the chain's coordinate is right and the position engine was
      simply stale, or
  (b) FABRICATION - the coordinate was invented, and the snap promotes the
      invention to a tracked fact that the exporter, the shot map and the next
      aerial all then read as measured.

The earlier attempt failed because it inferred INTENT from context labels. This
one does not infer anything: it joins the snap to the event object in the
caller's frame locals (an exact identity join, not a coordinate match), then
joins THAT event to the `make_event` call site that created it. So every large
snap is attributed to the line of code that produced its coordinate, and the
answer is read off the source rather than argued.

Outputs `_diag_snap_origin.txt`.

One match, one process. `_diag_chance_coords.build_pair` supplies the engine.
"""
import collections
import random
import sys

sys.path.insert(0, ".")

from event_chain import BaseChain  # noqa: E402
from position_engine import PositionEngine  # noqa: E402

BIG = 12.0

# id(event) -> "file:lineno" of the make_event call that created it
EVENT_SITE = {}
_orig_make = BaseChain.__dict__["make_event"]


def _make_event_traced(*a, **kw):
    ev = _orig_make(*a, **kw)
    fr = sys._getframe(1)
    fn = fr.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
    EVENT_SITE[id(ev)] = f"{fn}:{fr.f_lineno}"
    return ev


# `make_event` is a staticmethod, so wrap it as one or the descriptor protocol
# hands the wrapper `minute` as `self`.
BaseChain.make_event = staticmethod(_make_event_traced)

_orig_touch = PositionEngine.record_touch
ROWS = collections.defaultdict(lambda: {
    "n": 0, "metres": 0.0, "types": collections.Counter(),
    "at_ball": 0, "samples": [],
})


def traced(self, name, x, y, minute=0, **kw):
    st = self.states.get(name)
    if st is not None:
        jump = ((x - st.current_x) ** 2 + (y - st.current_y) ** 2) ** 0.5
        if jump >= BIG:
            f1 = sys._getframe(1)
            site = f"{f1.f_code.co_filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{f1.f_lineno}"
            ev = f1.f_locals.get("event")
            made = EVENT_SITE.get(id(ev)) if ev is not None else None
            # the ball's own tracked coordinate, if the frame can see it
            ball_x = None
            eng = f1.f_locals.get("self")
            if eng is not None and getattr(eng, "state", None) is not None:
                ball_x = getattr(eng.state, "last_ball_x", None)
            key = f"{site}  <-  {made or 'NOT AN EVENT SNAP'}"
            b = ROWS[key]
            b["n"] += 1
            b["metres"] += jump
            b["types"][ev.event_type.name if ev is not None else "?"] += 1
            if ball_x is not None and abs(ball_x - x) < 3.0:
                b["at_ball"] += 1
            if len(b["samples"]) < 2:
                b["samples"].append(
                    f"min{minute:>3} {name:<18} "
                    f"({st.current_x:5.1f},{st.current_y:5.1f}) -> "
                    f"({x:5.1f},{y:5.1f}) {jump:5.1f} m  ball_x="
                    f"{'%.1f' % ball_x if ball_x is not None else 'n/a'}")
    return _orig_touch(self, name, x, y, minute, **kw)


PositionEngine.record_touch = traced

from _diag_chance_coords import build_pair  # noqa: E402

LINES = ["running 1 real match...\n"]
random.seed(4242)
build_pair("Oxton", "Natrican").simulate()
LINES.append("done\n\n")

rows = sorted(ROWS.items(), key=lambda kv: -kv[1]["metres"])
tot_n = sum(b["n"] for b in ROWS.values())
tot_m = sum(b["metres"] for b in ROWS.values())
LINES.append(f"{'SNAP SITE  <-  make_event SITE':<52}{'N':>6}{'METRES':>9}"
             f"{'ATBALL':>8}  EVENT TYPES")
LINES.append("-" * 118)
for key, b in rows:
    left, _, right = key.partition("  <-  ")
    atb = f"{b['at_ball']}/{b['n']}"
    LINES.append(f"{left + ' <- ' + right:<52}{b['n']:>6}{b['metres']:>9.0f}"
                 f"{atb:>8}  {dict(b['types'])}")
LINES.append("-" * 118)
LINES.append(f"{'TOTAL':<52}{tot_n:>6}{tot_m:>9.0f}")
LINES.append("")
LINES.append("ATBALL = the snapped x was within 3 m of the engine's own "
             "tracked ball coordinate at that instant.")
LINES.append("")
LINES.append("samples:")
for key, b in rows:
    LINES.append(f"  {key}")
    for s in b["samples"]:
        LINES.append(f"      {s}")

text = "\n".join(LINES)
with open("_diag_snap_origin.txt", "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
print(text)