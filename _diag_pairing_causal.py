"""Pairing experiment: proximity join vs CAUSAL in-chain join.

WHAT IS UNDER TEST
------------------
`surrogate_collect._correlate_outcomes` attaches each decision record to "the
first execution event by that player at or after this record's minute",
consuming ONE event per record.  Measured on 3 real matches, its
self-consistency (does the paired event actually correspond to the intent?):

    SHOOT         1/8   (12.5%)
    CROSS         0/18  (0%)
    THROUGH_BALL  1/10  (10%)
    DRIBBLE       3/212 (1.4%)
    SAFE_PASS     408/630 (64.8%)

The v4 surrogate therefore ranks SHOOT at −0.0133, which would teach the GA
never to shoot — T2's documented second collapsed optimum.

MECHANISM (measured, not assumed)
----------------------------------
A decision can emit MORE THAN ONE execution event (a failed pass emits PASS
*and* TURNOVER / DISPOSSESSED / MISCONTROL), and the proximity cursor
consumes one event per record, so every extra event is mis-consumed by the
NEXT decision of that player and the alignment slides.

THE CAUSAL ALTERNATIVE
----------------------
`ChainDispatcher.possession` / `.attack` / `.set_piece` are the units of work:
the chain that makes the on-ball decision also produces the event, and it
emits the ACTION before its consequences.  So a decision can be joined to an
event of that same chain call, with no proximity and no cursor.

BUT — the first version of this probe got the causal rule wrong, and the
measurement is what said so.  It took "the first event by that player in that
chain", and for SHOOT that was a BALL_RECEIPT 5 times out of 8: the pass
ARRIVING at the new carrier is emitted by the carrier, BEFORE the action he
then takes.  The action is the first event by that player that is not a
receipt and not a press.

This probe runs every joining over the SAME matches and prints them side by
side, so the comparison is a controlled pair rather than two separate runs.

Usage:
    python _diag_pairing_causal.py [--matches 3] [--seed 123]
"""
import argparse
import collections
import statistics as st
import sys

SHOT_TYPES = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
              "HIT_WOODWORK", "SAVE", "GOAL", "PENALTY_SCORED",
              "PENALTY_MISSED"}
# Events that are not the on-ball action of the player they name.
NON_ACTION = {"BALL_RECEIPT", "PRESS", "PRESS_RESISTED"}
# …and the CONSEQUENCES of a failed action.  The failed action's OWN event
# (a PASS with outcome=False) already carries the loss charge in
# `threat_return`, so claiming the consequence as well DOUBLE-CHARGES the
# same loss — against the next decision, which did not cause it.
CONSEQUENCES = {"TURNOVER", "MISCONTROL", "DISPOSSESSED", "INTERCEPTION",
                "GROUND_DUEL", "AERIAL_DUEL"}


def _intent_value(intent):
    return getattr(intent, "value", str(intent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--styles", type=str, default="fluid_counter,attacking,tiki_taka")
    args = ap.parse_args()

    import surrogate_collect as sc
    from event_chain import ChainDispatcher
    from threat_reward import threat_return, event_type_name

    calls = []

    def wrap(name, raw):
        func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

        def wrapper(*a, **kw):
            n0 = len(sc._COLLECTOR)
            res = func(*a, **kw)
            decisions = list(sc._COLLECTOR[n0:])
            events = list(getattr(res, "events", None) or [])
            calls.append((decisions, events))
            return res
        setattr(ChainDispatcher, name, staticmethod(wrapper))

    saved = {}
    for name in ("possession", "attack", "set_piece"):
        raw = ChainDispatcher.__dict__.get(name)
        if raw is None:
            continue
        saved[name] = raw
        wrap(name, raw)

    away_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    try:
        sc.collect_from_matches(
            n_matches=args.matches, seed=args.seed,
            away_styles=away_styles, policy="heuristic", verbose=False,
        )
    finally:
        for name, raw in saved.items():
            setattr(ChainDispatcher, name, raw)

    out = []
    def w(line=""):
        out.append(line)
        print(line)

    # ── join B: causal, in-chain, skipping non-action events ────────────
    causal = []            # (intent, reward, event_type)
    refined = []           # same, but consequences are not claimable either
    dropped = collections.Counter()
    dropped_refined = collections.Counter()
    decided = collections.Counter()
    decisions_per_chain = collections.Counter()

    for decisions, events in calls:
        decisions_per_chain[len(decisions)] += 1
        used = set()
        used_ref = set()
        for d in decisions:
            i = _intent_value(d.intent)
            decided[i] += 1
            ev = None
            ev_ref = None
            for idx, e in enumerate(events):
                et = event_type_name(e)
                if (idx not in used and ev is None
                        and getattr(e, "player", None) == d.player
                        and et not in NON_ACTION):
                    ev = e
                    used.add(idx)
                if (idx not in used_ref and ev_ref is None
                        and getattr(e, "player", None) == d.player
                        and et not in NON_ACTION and et not in CONSEQUENCES):
                    ev_ref = e
                    used_ref.add(idx)
                if ev is not None and ev_ref is not None:
                    break
            if ev is None:
                dropped[i] += 1
                causal.append((i, 0.0, "NONE"))
            else:
                causal.append((i, threat_return(ev), event_type_name(ev)))
            if ev_ref is None:
                dropped_refined[i] += 1
                refined.append((i, 0.0, "NONE"))
            else:
                refined.append((i, threat_return(ev_ref),
                                event_type_name(ev_ref)))

    # ── join A: proximity, re-implemented on the same data ──────────────
    class _Rec:
        __slots__ = ("player", "minute", "order", "intent")

    recs = []
    for decisions, _events in calls:
        for d in decisions:
            r = _Rec()
            r.player = d.player
            r.minute = d.minute
            r.order = d.order
            r.intent = d.intent
            recs.append(r)

    timeline_events = []
    for _decisions, events in calls:
        timeline_events.extend(events)
    by_player = collections.defaultdict(list)
    for ev in timeline_events:
        p = getattr(ev, "player", None)
        if p is None:
            continue
        by_player[p].append(ev)

    prox = []
    for r in sorted(recs, key=lambda r: (r.player, r.order)):
        i = _intent_value(r.intent)
        ev = None
        for e in by_player.get(r.player, []):
            if getattr(e, "minute", 0) >= r.minute:
                ev = e
                break
        if ev is None:
            prox.append((i, 0.0, "NONE"))
            continue
        prox.append((i, threat_return(ev), event_type_name(ev)))

    def summarise(rows, label, drop_counter=None):
        rewards = collections.defaultdict(list)
        mix = collections.defaultdict(collections.Counter)
        for i, r, t in rows:
            rewards[i].append(r)
            mix[i][t] += 1
        w(f"=== {label} ===")
        w("  intent                 n    mean      min      max  drop%  paired-with")
        for intent in sorted(rewards, key=lambda i: -len(rewards[i])):
            rs = rewards[intent]
            d = (drop_counter or {}).get(intent, 0)
            tot = decided.get(intent, len(rs))
            drop_pct = 100.0 * d / tot if tot else 0.0
            top = ", ".join(f"{t}:{n}" for t, n in mix[intent].most_common(3))
            w(f"  {intent:<20} {len(rs):<5} {st.mean(rs):+.4f} {min(rs):+.4f} "
              f"{max(rs):+.4f} {drop_pct:>5.1f}%  {top}")
        w("")

    w(f"matches {args.matches}, seed {args.seed}")
    w(f"chain calls captured: {len(calls)}   decisions: {len(recs)}")
    w(f"decisions per chain call: {dict(sorted(decisions_per_chain.items()))}")
    w("")

    summarise(prox, "A. PROXIMITY join (what the collector does today)")
    summarise(causal, "B. CAUSAL in-chain join (skip non-action events)",
              drop_counter=dropped)
    summarise(refined, "C. CAUSAL + consequences excluded (no double-charge)",
              drop_counter=dropped_refined)

    w("=== the rare intents, all three ways ===")
    for label, rows in (("proximity", prox), ("causal", causal),
                        ("refined", refined)):
        for intent in ("SHOOT", "CROSS", "THROUGH_BALL"):
            rs = [(r, t) for i, r, t in rows if i == intent]
            if not rs:
                w(f"  {label:<10} {intent:<13} (none)")
                continue
            shotish = sum(1 for _r, t in rs if t in SHOT_TYPES)
            vals = [r for r, _t in rs]
            w(f"  {label:<10} {intent:<13} n={len(rs):<4} shot-ish: {shotish:<4} "
              f"mean {st.mean(vals):+.4f}  mix "
              + ", ".join(f"{t}:{n}" for t, n in collections.Counter(t for _r, t in rs).most_common(4)))
        w("")

    w("=== decisions with no action event in their own chain (drop rate) ===")
    for intent in sorted(decided, key=lambda i: -decided[i]):
        d = decided[intent]
        dr = dropped.get(intent, 0)
        drr = dropped_refined.get(intent, 0)
        w(f"  {intent:<20} decided {d:<5} drop B {dr:<5} ({100.0*dr/d:>5.1f}%)"
          f"   drop C {drr:<5} ({100.0*drr/d:>5.1f}%)")

    with open("_diag_pairing_causal.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("[written to _diag_pairing_causal.txt]")


if __name__ == "__main__":
    sys.exit(main())
