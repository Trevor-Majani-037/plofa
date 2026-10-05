"""What the aerial fix did to real corners. ONE match; run in its own process.

Reports the aerial win split, which is the number that was broken: attackers
won 93-100% before because dead heats went to whoever was appended first.
Run several seeds in SEPARATE processes -- an in-process sweep inherits the
module-level brain caches and is meaningless.

Usage:  python _diag_corner_split.py <seed>
"""
import io
import random
import sys

sys.path.insert(0, ".")
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402
from event_chain import EventType  # noqa: E402
import importlib  # noqa: E402

ph = importlib.import_module("possession_physics")
PE = ph.PossessionEpisode
_ra = PE.resolve_aerial
_current = [None]
rows = []


def resolve_aerial(self, flight, attackers, defenders):
    res = _ra(self, flight, attackers, defenders)
    pe = _current[0]
    if pe is not None and pe.setpiece_active():
        w = getattr(getattr(res, "winner", None), "player", None)
        rows.append((getattr(w, "name", "") or "", res.outcome,
                     len([a for a in attackers if a is not None]),
                     len([d for d in defenders if d is not None])))
    return res


PE.resolve_aerial = resolve_aerial

_gen = event_chain.SetPieceChain.generate.__func__


def generate(cls, *a, **kw):
    prev = _current[0]
    _current[0] = kw.get("position_engine")
    try:
        return _gen(cls, *a, **kw)
    finally:
        _current[0] = prev


event_chain.SetPieceChain.generate = classmethod(generate)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
eng, _, _ = build()
random.seed(seed)
real = sys.stdout
sys.stdout = io.StringIO()
try:
    res = eng.simulate()
finally:
    sys.stdout = real

corners = [e for e in res.timeline if e.event_type == EventType.CORNER_TAKEN]
att = sum(1 for e in corners if e.outcome)
gk = sum(1 for r in rows if r[0] and r[0].endswith("GK")) if rows else 0
split = {}
for r in rows:
    # NB: this column CANNOT tell which side won -- it only knows whether the
    # winner is the keeper. Labelling it "attacker" was actively misleading
    # once the defence started winning: a clean-looking column that is not
    # measuring what it says.
    who = "keeper" if r[0].endswith("GK") else ("outfield" if r[0] else "loose")
    split[who] = split.get(who, 0) + 1
n = max(1, len(rows))
print(f"seed {seed:>3} | corners {len(corners):>2} | attacker-won {att:>2} "
      f"({100 * att / max(1, len(corners)):.0f}%) | duels {len(rows):>2} "
      f"| split {split} | GK wins {gk}")