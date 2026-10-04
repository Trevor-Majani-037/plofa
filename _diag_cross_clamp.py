"""THE CROSS-RECEIVER CLAMP — isolate it, then measure its removal (2026-10-02)

`_diag_teleport_moves.py` measured the POPULATION of single-call `record_touch`
jumps (1,040 per match, 27,333 m) but could not separate a deliberate
placement from a clamp, because it attributed by wrapping each chain's
`generate` and pushing a context label. AGENTS.md records that as the fourth
rejected probe pairing of the session: context cannot distinguish them.

So do not attribute by context. SIGNATURE-TEST the mechanism instead, exactly
as AGENTS.md prescribes for this case:

    new_x inside the attacking [85, 100] band (or [5, 20] when attacking left)
    AND new_y inside [22, 46]
    AND old_x more than ~5 m outside that band

The clamp is the only site in the codebase that reads a real position, clamps
it into that band, and writes the RESULT back. `clamp_attack_x` is called from
five sites but four clamp a live-frame x for a CONTACT POINT and never persist
it; this one persists.

Two numbers matter and they are separate questions:

  1. How much FABRICATED MOVEMENT does the clamp bank? (distance covered that
     the player never covered)
  2. What does it do to the FOOTBALL? It writes the receiver into the box, and
     `_moving_player` reads that back before `resolve_aerial_delivery` scores
     the duel — so he arrives already there and is charged no movement cost. A
     cross from 45 m should mostly be cleared.

Run: .venv\\Scripts\\python.exe _diag_cross_clamp.py [n]
"""
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from match_engine import EventType
from position_engine import PositionEngine
from _diag_chance_coords import build_pair

_orig_record_touch = PositionEngine.record_touch

CROSS_EVENTS = {EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS}


class ClampAudit:
    """Wrap record_touch and signature-test the cross-receiver clamp."""

    def __init__(self):
        self.hits = []
        self.total_calls = 0
        self.jumps = 0

    def install(self):
        audit = self

        def _team_of(pe, name):
            for team, roster in (getattr(pe, "team_rosters", {}) or {}).items():
                for p in roster:
                    if getattr(p, "name", "") == name:
                        return team
            return None

        def wrapped(self, name, new_x, new_y, minute=0, *a, **kw):
            audit.total_calls += 1
            trk = self.tracked_position(name)
            if trk is not None:
                ox, oy = trk
                d = ((new_x - ox) ** 2 + (new_y - oy) ** 2) ** 0.5
                if d >= 12.0:
                    audit.jumps += 1
                # attacks_right for this player's team, per the engine's own map
                ar = self.team_attacks_right.get(_team_of(self, name), True)
                lo, hi = (85.0, 100.0) if ar else (5.0, 20.0)
                if (lo <= new_x <= hi and 22.0 <= new_y <= 46.0
                        and not (lo - 5.0 <= ox <= hi + 5.0)):
                    audit.hits.append((minute, name, round(ox, 1), round(oy, 1),
                                       round(new_x, 1), round(new_y, 1), round(d, 1)))
            return _orig_record_touch(self, name, new_x, new_y, minute, *a, **kw)

        PositionEngine.record_touch = wrapped

    def restore(self):
        PositionEngine.record_touch = _orig_record_touch

    def report(self):
        n = len(self.hits)
        if not n:
            return "0 clamps"
        dist = sum(h[6] for h in self.hits)
        return (f"{n} clamps, {dist:.0f} m of movement never covered "
                f"({dist / 105.0:.1f} pitches)")


def analyse(res, audit):
    tl = res.timeline
    crosses = [e for e in tl if e.event_type in CROSS_EVENTS]
    won = [e for e in tl if e.event_type == EventType.WON_AERIAL] \
        if hasattr(EventType, "WON_AERIAL") else []
    goals = [g for g in res.goals if g.event_type != EventType.OWN_GOAL]
    return {
        "crosses": len(crosses),
        "cross_success": sum(1 for e in crosses if e.event_type == EventType.CROSS_SUCCESS),
        "won_aerial": len(won),
        "goals": len(goals),
        "record_touch_calls": audit.total_calls,
        "jumps12": audit.jumps,
        "clamps": len(audit.hits),
        "clamp_m": sum(h[6] for h in audit.hits),
        "clamp_sample": audit.hits[:12],
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    tot = Counter()
    samples = []
    for i in range(n):
        h, a = pairs[i % len(pairs)]
        random.seed(3000 + i)
        print(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
        audit = ClampAudit()
        audit.install()
        try:
            res = build_pair(h, a).simulate()
        finally:
            audit.restore()
        r = analyse(res, audit)
        for k in ("crosses", "cross_success", "won_aerial", "goals",
                  "record_touch_calls", "jumps12", "clamps", "clamp_m"):
            tot[k] += r[k]
        samples += r["clamp_sample"]
        print(f"      {res.home_goals}-{res.away_goals}  crosses {r['crosses']} "
              f"(ok {r['cross_success']})  clamps {r['clamps']} "
              f"{r['clamp_m']:.0f}m  jumps>=12m {r['jumps12']}", flush=True)

    print("\n" + "=" * 74)
    print(f"  record_touch calls            {tot['record_touch_calls']}")
    print(f"  of those, jumps >= 12 m       {tot['jumps12']}")
    print(f"  MATCHING THE CLAMP SIGNATURE  {tot['clamps']}  "
          f"({tot['clamp_m']:.0f} m never covered = "
          f"{tot['clamp_m']/105.0:.1f} pitches)")
    print(f"  crosses {tot['crosses']} (completed {tot['cross_success']}), "
          f"aerials won {tot['won_aerial']}, goals {tot['goals']}")
    if samples:
        print("\n  sample (minute, player, OLD -> NEW, metres):")
        for m, nm, ox, oy, nx, ny, d in samples:
            print(f"    {m:>3}  {nm:<22} ({ox},{oy}) -> ({nx},{ny})  {d} m")


if __name__ == "__main__":
    main()
