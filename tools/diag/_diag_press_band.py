"""Did the BEFORE arm's press-band restore actually fire? (2026-10-04)

Settles a falsification. `_diag_football_sites.py --arm before` reported
TransitionChain presses (`event_chain.py:8130`) spanning x = 0.0 to 105.0 with
22 of 44 below x=55 -- in the arm that was supposed to restore the old
`random.uniform(55, 85)` draw.

Two explanations, and they mean opposite things:

  (a) LEGITIMATE. The restore deliberately exempts counterpress presses
      (`if not md.get("counterpress")`), and counterpress coordinates come
      from the real zone. If roughly half of TransitionChain presses are
      counterpresses, then the band holds exactly where it was supposed to and
      the out-of-band values are real.
  (b) THE CONTROL FAILED. The wrapper never fired, the "before" arm was
      secretly running fixed code, and the band claim has no evidence behind it
      at all.

A silently-no-op control is worse than no control, because it reads as
support. So this probe does not infer the answer from the distribution -- it
counts how many presses the restore actually rewrote, and reports the band
only for the group the restore was supposed to govern.

EXITS NONZERO if the band is violated by a non-counterpress press, because a
claim that needs a caveat is not a claim.
"""
import collections
import json
import random
import sys

sys.path.insert(0, ".")

from event_chain import BaseChain  # noqa: E402

_RAW = BaseChain.__dict__["make_event"]
REWRITTEN = []
NONCP_X = []
CP_X = []
SITE = {}
# OWNER is recorded at EMISSION, not recovered later. A previous version of
# this file tried to identify the governed site inside the verdict loop, where
# the event object is out of scope and the only handle is a dict that was
# never defined -- so the loop raised NameError rather than measuring. Guessing
# which site "governs" the band is the same inference this probe exists to
# avoid, in the other direction.
OWNER = {}


def _make(*a, **kw):
    ev = _RAW(*a, **kw)
    fr = sys._getframe(1)
    f = fr
    site = f"{f.f_code.co_filename.split(chr(92))[-1]}:{f.f_lineno}"
    SITE[id(ev)] = site
    md = ev.metadata or {}
    # KEY ON THE CHAIN, NOT THE EVENT TYPE. An earlier version fired on every
    # non-counterpress PRESS and so rewrote PossessionChain's press too, which
    # is `ball +/- 3 m` (`event_chain.py:1560`) and was never fabricated. It
    # reported 84 such rewrites and let the band claim look like it covered
    # every press in the match. The `cls` local names the emitting chain.
    cls = fr.f_locals.get("cls")
    owner = getattr(cls, "__name__", "")
    OWNER[id(ev)] = owner
    if (ev.event_type.name == "PRESS"
            and owner == "TransitionChain"
            and not md.get("counterpress")):
        ev.location_x = random.uniform(55, 85)
        ev.location_y = random.uniform(10, 58)
        REWRITTEN.append(site)
    return ev


BaseChain.make_event = staticmethod(_make)

from _diag_chance_coords import build_pair  # noqa: E402

random.seed(9100)
res = build_pair("Oxton", "Natrican").simulate()

by_site = collections.defaultdict(lambda: {"cp": [], "ncp": []})
for e in res.timeline:
    if e.event_type.name != "PRESS":
        continue
    site = SITE.get(id(e), "?")
    md = e.metadata or {}
    key = "cp" if md.get("counterpress") else "ncp"
    by_site[site][key].append((e.location_x, e.location_y,
                               OWNER.get(id(e), "?")))

print(f"rewrites performed by the restore: {len(REWRITTEN)}")
by_rewritten_site = collections.Counter(REWRITTEN)
for s, n in by_rewritten_site.most_common():
    print(f"    {s}: {n}")
if not REWRITTEN:
    print("    *** NONE -- THE CONTROL DID NOT FIRE ***")

print()
print(f"{'site':<22} {'grp':<4} {'n':>4} {'x_min':>7} {'x_med':>7} "
      f"{'x_max':>7}  owner")
violations = 0
governed = 0
for site, groups in sorted(by_site.items()):
    for key in ("ncp", "cp"):
        pts = groups[key]
        if not pts:
            continue
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        owners = sorted({p[2] for p in pts})
        xmed = xs[len(xs) // 2]
        print(f"{site:<22} {key:<4} {len(pts):>4} {xs[0]:>7.1f} "
              f"{xmed:>7.1f} {xs[-1]:>7.1f}  {','.join(owners)}")
        if key != "ncp":
            continue
        # The band was NEVER PossessionChain's rule -- that press is
        # `ball +/- 3 m` (`event_chain.py:1560`) and legitimately spans the
        # whole pitch. This check fired on every non-counterpress press and
        # reported 49 "violations" that were all PossessionChain presses
        # measured against a rule that was never theirs. A check that does not
        # measure the thing under test is the same defect as a mutation with
        # the wrong site condition: both look clean and are wrong.
        if "TransitionChain" not in owners:
            print(f"{'':<22}     not the governed site -- band does not "
                  f"apply; this press is ball +/- 3 m")
            continue
        governed += len(pts)
        bad = [v for v in xs if v < 55.0 or v > 85.0]
        bady = [v for v in ys if v < 10.0 or v > 58.0]
        if bad or bady:
            violations += len(bad) + len(bady)
            print(f"{'':<22}     BAND VIOLATION: x outside [55,85] "
                  f"{len(bad)} times, y outside [10,58] {len(bady)} times")

print()
print("=" * 68)
if not REWRITTEN:
    print("VERDICT: control did not fire. The band claim has NO evidence.")
elif not governed:
    print("VERDICT: no governed presses found -- the restore did not target "
          "the site under test.")
elif violations:
    print(f"VERDICT: {violations} of {governed} governed presses fall outside "
          "the band. The claim as stated is FALSE.")
else:
    print(f"VERDICT: all {governed} non-counterpress TransitionChain presses "
          "sit inside [55,85]x[10,58].")
    print("The band claim holds for TransitionChain non-counterpress presses")
    print("only. Counterpress presses were always real, and PossessionChain's")
    print("press is ball +/- 3 m and was never in a band at all.")
print("=" * 68)

with open("_diag_press_band.json", "w", encoding="utf-8") as fh:
    json.dump({"rewrites": len(REWRITTEN),
               "violations": violations,
               "sites": {s: {k: [list(p) for p in v]
                             for k, v in g.items() if v}
                         for s, g in by_site.items()}}, fh, indent=2)

sys.exit(1 if (violations or not REWRITTEN) else 0)