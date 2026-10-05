"""Why does the attacker win every real corner when the isolated probe says the
defender wins that geometry?

The two disagree, so the probe is not modelling the live case. This logs the
race inside a REAL match instead of reasoning about it.

It wraps `geometry_engine._race_motion` -- the function that produces every
`arrival` figure the resolver scores on -- and records, for corner duels only:

  * the distance from each man to the ball point he was chasing
  * the arrival time the race gave him

then reports whether the winner won on ARRIVAL (genuinely first to it) or
downstream of a tie. `resolve_aerial_delivery` is imported BY NAME into
`possession_physics`, so patching the geometry_engine attribute intercepts
nothing -- the entry point has to be `PossessionEpisode.resolve_aerial`, which
is what `_diag_corner_split.py` already proved works.

And the corner has to be identified INSIDE `resolve_aerial`, because the
set-piece window is opened partway through the chain: testing it when
`generate` is entered finds it closed and measures nothing. That mistake
produced "corners measured 0" twice before this line.

Usage: python _diag_race_in_match.py <seed>
"""
import importlib
import io
import math
import random
import sys

sys.path.insert(0, ".")
import geometry_engine as G  # noqa: E402
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402

_current = [None]
_in_duel = [False]
duels = []
_log = []

_orig_race = G._race_motion


def race(player, target, radius=None, player_context=None):
    r = _orig_race(player, target, radius, player_context)
    if _in_duel[0]:
        nm = getattr(getattr(player, "player", None), "name", "?")
        d = math.hypot(player.position.x - target.x,
                       player.position.y - target.y)
        _log.append((nm, round(d, 2), round(r, 3)))
    return r


G._race_motion = race

ph = importlib.import_module("possession_physics")
PE = ph.PossessionEpisode
_ra = PE.resolve_aerial


def resolve_aerial(self, flight, attackers, defenders):
    pe = _current[0]
    if pe is None or not pe.setpiece_active():
        return _ra(self, flight, attackers, defenders)
    att_names = {getattr(getattr(p, "player", None), "name", "")
                 for p in attackers if p is not None}
    _log.clear()
    _in_duel[0] = True
    try:
        res = _ra(self, flight, attackers, defenders)
    finally:
        _in_duel[0] = False
    w = getattr(getattr(res, "winner", None), "player", None)
    winner = getattr(w, "name", "") or ""
    per = {}
    for nm, d, arr in _log:
        e = per.setdefault(nm, {"dist": 9e9, "arrival": 9e9,
                                "att": nm in att_names})
        e["dist"] = min(e["dist"], d)
        e["arrival"] = min(e["arrival"], arr)
    duels.append({"winner": winner, "att": winner in att_names,
                  "per": per, "outcome": getattr(res, "outcome", "")})
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

print(f"seed {seed} | corners measured {len(duels)}")
won = sum(1 for d in duels if d["att"])
print(f"attacker won the aerial: {won}/{len(duels)}")
print()
for i, d in enumerate(duels[:5]):
    print(f"duel {i}: winner {d['winner']!r} "
          f"({'attacker' if d['att'] else 'defender'}) {d['outcome']}")
    for nm, e in sorted(d["per"].items(), key=lambda kv: kv[1]["arrival"]):
        mark = "  <-- WON" if nm == d["winner"] else ""
        print(f"    {nm:<26} {'ATT' if e['att'] else 'DEF':>3}  "
              f"nearest {e['dist']:6.2f} m   best arrival {e['arrival']:6.3f} s"
              f"{mark}")
    print()

uniq = gaps = 0
gaps_l = []
for d in duels:
    if not d["per"] or not d["winner"]:
        continue
    w = d["per"].get(d["winner"], {}).get("arrival")
    if w is None:
        continue
    others = sorted(v["arrival"] for nm, v in d["per"].items()
                    if nm != d["winner"])
    n_tied = 1 + sum(1 for a in others if a <= w + 1e-9)
    if n_tied == 1:
        uniq += 1
        if others:
            gaps_l.append(others[0] - w)
print(f"duels where the winner was UNIQUELY fastest: {uniq}/{len(duels)}")
if gaps_l:
    s = sorted(gaps_l)
    print(f"  margin over the next best: median {s[len(s) // 2]:.3f} s, "
          f"min {s[0]:.3f} s")