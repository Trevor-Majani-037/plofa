"""How often does the centre-spot DEFAULT become a player's tracked position?

Open item #1, second half. `_diag_snap_origin.py` attributed every large
`record_touch` jump to the `make_event` call site that produced its coordinate.
Reading those sites turned up `event_chain.py:8167`, a `PASS` with no
`location_x`/`location_y`/`end_x`/`end_y` at all -- yet 20 snaps and 509 m came
from it.

The reason is `match_engine.py:592`:

    location_x: float = 50.0     # 0-105 (meters from home goal line)
    location_y: float = 34.0     # 0-68  (meters from left touchline)

so an event that omits its coordinates gets THE CENTRE SPOT, and
`_absorb_chain:5764`'s `if event.location_x is not None` guard cannot see it --
50.0 is not None. This is the same trap AGENTS.md already records for
`PositionEngine.get_position` (which returns `(50.0, 34.0)` for an untracked
player): a value that reads as a real measurement and is not one.

This probe does not infer. `make_event` receives `**kwargs`, so the PRESENCE of
`location_x` is knowable exactly at the call, before the default is ever
applied. Events lacking it are then joined to their snaps through the `event`
object in `_absorb_chain`'s frame locals -- an identity join, not a coordinate
match.

One match, one process. Outputs `_diag_default_loc.txt`.
"""
import collections
import random
import sys

sys.path.insert(0, ".")

from event_chain import BaseChain  # noqa: E402
from position_engine import PositionEngine  # noqa: E402

BIG = 12.0
DEFAULT_X, DEFAULT_Y = 50.0, 34.0

# id(event) -> (call site, had_explicit_location, location_x, location_y)
EVENT_INFO = {}
_orig_make = BaseChain.__dict__["make_event"]


def _make_event_traced(*a, **kw):
    explicit = "location_x" in kw or "location_y" in kw
    ev = _orig_make(*a, **kw)
    fr = sys._getframe(1)
    fn = fr.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
    site = f"{fn}:{fr.f_lineno}"
    EVENT_INFO[id(ev)] = (site, explicit, ev.location_x, ev.location_y)
    # Count HERE, not in __post_init__: the presence of the kwarg is only known
    # in this frame, and __post_init__ runs *inside* _orig_make, i.e. before
    # this wrapper has registered the event -- an earlier version looked the
    # site up from __post_init__ and therefore reported 1 site instead of all.
    TOTAL_EVENTS[0] += 1
    if not explicit:
        NO_LOCATION[0] += 1
        NO_LOCATION_TYPES[ev.event_type.name] += 1
        MISSING_SITES[site][ev.event_type.name] += 1
    return ev


BaseChain.make_event = staticmethod(_make_event_traced)

_orig_touch = PositionEngine.record_touch

TOTAL_EVENTS = [0]
NO_LOCATION = [0]
NO_LOCATION_TYPES = collections.Counter()
MISSING_SITES = collections.defaultdict(lambda: collections.Counter())
SNAPS = collections.defaultdict(lambda: {"n": 0, "metres": 0.0})
SNAP_EVENTS = collections.defaultdict(lambda: collections.Counter())
AT_CENTRE = [0]


def traced(self, name, x, y, minute=0, **kw):
    st = self.states.get(name)
    if st is not None:
        jump = ((x - st.current_x) ** 2 + (y - st.current_y) ** 2) ** 0.5
        if jump >= BIG:
            f1 = sys._getframe(1)
            ev = f1.f_locals.get("event")
            info = EVENT_INFO.get(id(ev)) if ev is not None else None
            if info is not None:
                site, explicit, lx, ly = info
                if not explicit:
                    AT_CENTRE[0] += 1
                    b = SNAPS[site]
                    b["n"] += 1
                    b["metres"] += jump
                    SNAP_EVENTS[site][ev.event_type.name] += 1
    return _orig_touch(self, name, x, y, minute, **kw)


PositionEngine.record_touch = traced

# (the __post_init__ patch is gone -- counting happens in make_event, where the
# kwarg presence is actually observable. See the note there.)
from _diag_chance_coords import build_pair  # noqa: E402

LINES = ["running 1 real match...\n"]
random.seed(4242)
build_pair("Oxton", "Natrican").simulate()
LINES.append("done\n\n")

ev_total, ev_missing = TOTAL_EVENTS[0], NO_LOCATION[0]
snap_total = sum(b["n"] for b in SNAPS.values())
snap_m = sum(b["metres"] for b in SNAPS.values())

LINES.append("EVENTS with no explicit location")
LINES.append("=" * 70)
LINES.append(f"  events created                      {ev_total}")
LINES.append(f"  sitting on the centre-spot default  {ev_missing} "
             f"({100.0 * ev_missing / max(ev_total, 1):.1f}%)")
LINES.append("")
LINES.append("  by event type:")
for t, c in NO_LOCATION_TYPES.most_common(14):
    LINES.append(f"    {t:<22} {c}")
LINES.append("")
LINES.append("ALL sites that omit a location (every event type, not just "
             "snapping ones)")
LINES.append("=" * 70)
for site, types in sorted(MISSING_SITES.items(),
                          key=lambda kv: -sum(kv[1].values())):
    LINES.append(f"  {site:<26} {sum(types.values()):>4}  {dict(types)}")
LINES.append("")
LINES.append("SNAPS (>=12 m) from those events")
LINES.append("=" * 70)
LINES.append(f"{'make_event SITE':<26}{'N':>6}{'METRES':>9}  EVENT TYPES")
LINES.append("-" * 70)
for site, b in sorted(SNAPS.items(), key=lambda kv: -kv[1]["metres"]):
    LINES.append(f"{site:<26}{b['n']:>6}{b['metres']:>9.0f}  "
                 f"{dict(SNAP_EVENTS[site])}")
LINES.append("-" * 70)
LINES.append(f"{'TOTAL':<26}{snap_total:>6}{snap_m:>9.0f}")
LINES.append("")
LINES.append(f"Of {snap_total} large snaps joined to an event, {AT_CENTRE[0]} "
             f"came from an event with NO explicit location.")

text = "\n".join(LINES)
with open("_diag_default_loc.txt", "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
print(text)