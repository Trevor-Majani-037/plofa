"""Apportion the `record_touch` "inside-the-box jump" by CALL SITE.

`_diag_cross_clamp.py` signature-tests a SYMPTOM: a position written inside an
attacking box band from more than ~5 m outside it. It reports ~79 per two
matches. That number barely moved when the cross-receiver clamp was deleted
(77 -> 79, across different matches), which proves the cross-receiver clamp was
a real defect but a SMALL contributor, and that the bulk is something else.

The earlier attempt to apportion this could not separate a clamp from a
deliberate placement, because it inferred INTENT from context labels. That was
the wrong question. The right question is "WHICH LINE OF CODE WROTE THIS", and
that is answered EXACTLY by the caller's stack frame -- no inference, no
signature, no ambiguity. So this probe records filename+lineno of the immediate
caller (and of ITS caller, which names the owning chain) and buckets by it.

Two honest caveats recorded here rather than discovered later:
  * The first version of this probe printed a heading saying "by signature
    (new_x in band, y in [22,46], ...)" and then summed EVERY site -- it never
    implemented the filter it advertised, so the number under that heading was
    the >=12 m total wearing a false label. Fixed: `y` is now captured and the
    filter is actually applied.
  * The signature is direction-free (`x` near EITHER goal line), because
    `record_touch` has no idea which way the team attacks. `_diag_cross_clamp`
    used the home-attacks-right band only, so the two counts are comparable in
    spirit but not in exact definition.

Output goes to `_diag_touch_sites.txt`, not to stdout: piping a long table
through `Select-Object -Last N` truncates the TOP, which is precisely the part
worth reading, and it did so on the first run.
"""
import collections
import sys

sys.path.insert(0, ".")

from position_engine import PositionEngine  # noqa: E402

_orig = PositionEngine.record_touch
SITES = collections.defaultdict(
    lambda: {"n": 0, "metres": 0.0, "sample": None,
             "sig_n": 0, "sig_m": 0.0})
BIG = 12.0
LINES = []


def _site(frame):
    fn = frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
    return f"{fn}:{frame.f_lineno}"


def _into_a_box(new_x, new_y, old_x):
    """Direction-free version of the old clamp signature."""
    in_band = (85.0 <= new_x <= 100.0) or (5.0 <= new_x <= 20.0)
    return in_band and 22.0 <= new_y <= 46.0 and abs(new_x - old_x) > 5.0


def traced(self, name, x, y, minute=0, **kw):
    old_x = None
    try:
        prev = self.states.get(name)
        old_x = prev.current_x if prev is not None else None
    except Exception:
        old_x = None
    jump = abs(x - old_x) if old_x is not None else 0.0
    if jump >= BIG:
        here = _site(sys._getframe(1))
        owner = _site(sys._getframe(2))
        b = SITES[here]
        b["n"] += 1
        b["metres"] += jump
        if _into_a_box(x, y, old_x):
            b["sig_n"] += 1
            b["sig_m"] += jump
        if b["sample"] is None:
            b["sample"] = (f"min {minute:>3}  {name:<18} {old_x:6.1f} -> "
                           f"{x:6.1f} (y {y:5.1f})  {jump:5.1f} m   "
                           f"owner {owner}")
    return _orig(self, name, x, y, minute, **kw)


PositionEngine.record_touch = traced

from _diag_chance_coords import build_pair  # noqa: E402

import random as _r  # noqa: E402

LINES.append("running 2 real matches...\n")
PAIRS = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
for i, (h, a) in enumerate(PAIRS):
    _r.seed(3000 + i)
    build_pair(h, a).simulate()
    LINES.append(f"  [{i + 1}/{len(PAIRS)}] {h} v {a} done\n")

rows = sorted(SITES.items(), key=lambda kv: -kv[1]["metres"])
gn = gm = gs_n = gs_m = 0
LINES.append(f"{'SITE':<26}{'JUMPS':>7}{'METRES':>9}{'PITCH':>7}"
             f"{'BOXJMP':>8}{'BOX m':>8}")
LINES.append("-" * 65)
for site, b in rows:
    gn += b["n"]
    gm += b["metres"]
    gs_n += b["sig_n"]
    gs_m += b["sig_m"]
    LINES.append(f"{site:<26}{b['n']:>7}{b['metres']:>9.0f}"
                 f"{b['metres'] / 105:>7.1f}{b['sig_n']:>8}{b['sig_m']:>8.0f}")
LINES.append("-" * 65)
LINES.append(f"{'TOTAL':<26}{gn:>7}{gm:>9.0f}{gm / 105:>7.1f}"
             f"{gs_n:>8}{gs_m:>8.0f}")
LINES.append("")
LINES.append("BOXJMP = the old clamp signature: new x within 5-15 m of EITHER")
LINES.append("goal line, y in [22,46], and moved >5 m to get there. Compare")
LINES.append("against the 77 and 79 figures in AGENTS.md, remembering that")
LINES.append("those used the home-attacks-right band ONLY, so they are a")
LINES.append("subset of this count, not the same count.")
LINES.append("")
LINES.append("worked samples (one per site):")
for site, b in rows:
    LINES.append(f"  {site}\n      {b['sample']}")

text = "\n".join(LINES)
with open("_diag_touch_sites.txt", "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
print(text)
print("\nwrote _diag_touch_sites.txt")
