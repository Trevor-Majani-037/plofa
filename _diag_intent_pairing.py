"""Which event did each INTENT actually get paired with?

Why this probe exists
---------------------
The 6-match v4 surrogate's sample-weighted per-intent reward says:

    THROUGH_BALL  +0.0106 (34 samples)
    ...
    CROSS         -0.0032 (70)
    SHOOT         -0.0133 (38)     ← NEGATIVE

But `threat_reward.threat_return` scores EVERY shot outcome positively:
SHOT_ON_TARGET / SAVE / SHOT_BLOCKED / HIT_WOODWORK → +xg,
SHOT_OFF_TARGET → +0.5·xg, GOAL → +1.0, PENALTY_MISSED → −0.5·xg.
A penalty miss is the only negative shot outcome, and penalties are rare.

So a negative SHOOT mean cannot come from the reward function.  It has to
come from the PAIRING: `surrogate_collect._correlate_outcomes` attaches each
decision record to "the first execution event by that player at or after this
record's minute", consuming one event per record.  The cursor advances on
MINUTE granularity only — the record's `second` is not used — so when a player
makes several decisions inside one minute the attachment slides.

This probe reads the cross-tab directly: intent × the event type it was
paired with, plus the reward distribution for each pairing.  It does NOT
re-implement the reward and it does NOT re-implement the pairing — it calls
the collector's own `collect_from_matches` and reads the 4th element of the
tuple the collector already produces.

Output goes to `_diag_intent_pairing.txt` as well as stdout, because a wide
cross-tab wraps in the terminal and a wrapped table is how a wrong column
becomes a right-looking one.

Usage:
    python _diag_intent_pairing.py [--matches 3] [--seed 123]
"""
import argparse
import collections
import statistics as st
import sys

SHOT_TYPES = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
              "HIT_WOODWORK", "SAVE", "GOAL", "PENALTY_SCORED",
              "PENALTY_MISSED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--styles", type=str, default="fluid_counter,attacking,tiki_taka")
    args = ap.parse_args()

    import surrogate_collect as sc

    away_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    pairs = sc.collect_from_matches(
        n_matches=args.matches,
        seed=args.seed,
        away_styles=away_styles,
        policy="heuristic",
        verbose=False,
    )

    out = []
    def w(line=""):
        out.append(line)
        print(line)

    w(f"collected {len(pairs)} pairs over {args.matches} matches "
      f"(seed {args.seed}, styles {away_styles})")

    cross = collections.defaultdict(collections.Counter)
    rewards = collections.defaultdict(list)
    by_intent = collections.defaultdict(list)

    for sensors, intent, reward, ev_type, position in pairs:
        i = getattr(intent, "value", str(intent))
        cross[i][ev_type] += 1
        rewards[(i, ev_type)].append(reward)
        by_intent[i].append(reward)

    w("")
    w("=== intent x PAIRED EVENT TYPE (count) ===")
    all_types = sorted({t for c in cross.values() for t in c})
    w("  " + "intent".ljust(20) + "".join(f"{t[:16]:>17}" for t in all_types))
    for intent in sorted(cross, key=lambda i: -sum(cross[i].values())):
        row = "  " + intent.ljust(20)
        for t in all_types:
            n = cross[intent].get(t, 0)
            row += f"{n:>17}" if n else " " * 17
        w(row)

    w("")
    w("=== per intent: reward, and the full paired-event mix ===")
    for intent in sorted(by_intent, key=lambda i: -len(by_intent[i])):
        rs = by_intent[intent]
        w(f"  {intent:<20} n={len(rs):<5} mean={st.mean(rs):+.4f}  "
          f"min={min(rs):+.4f} max={max(rs):+.4f}")
        for t, n in cross[intent].most_common():
            tr = rewards[(intent, t)]
            w(f"      {t:<20} n={n:<5} mean={st.mean(tr):+.4f}  "
              f"min={min(tr):+.4f} max={max(tr):+.4f}")

    w("")
    w("=== the rare intents, pairing by pairing ===")
    for intent in ("SHOOT", "CROSS", "THROUGH_BALL"):
        if intent not in cross:
            w(f"  {intent}: NO RECORDS AT ALL")
            continue
        total = sum(cross[intent].values())
        real = sum(n for t, n in cross[intent].items() if t in SHOT_TYPES)
        w(f"  {intent:<14} n={total}  paired with a shot-ish event: {real}/{total}")
        for t, n in cross[intent].most_common():
            tr = rewards[(intent, t)]
            w(f"      {t:<20} n={n:<4} mean={st.mean(tr):+.4f}  "
              f"values={[round(x, 4) for x in tr[:14]]}")

    w("")
    w("=== summary: does the reward of an intent look like its own event? ===")
    for intent in sorted(by_intent):
        total = sum(cross[intent].values())
        if total == 0:
            continue
        # what fraction of this intent's samples paired with an event whose
        # type would plausibly BE that intent
        self_ev = {
            "SAFE_PASS": {"PASS"}, "RECYCLE": {"PASS"}, "PROGRESSIVE_PASS": {"PASS"},
            "SWITCH": {"SWITCH_OF_PLAY", "PASS"}, "CARRY": {"CARRY"},
            "DRIBBLE": {"DRIBBLE_SUCCESS", "DRIBBLE_FAIL"},
            "PROTECT_POSSESSION": {"PASS", "CARRY"},
            "CROSS": {"CROSS_ATTEMPT", "CROSS_SUCCESS"},
            "THROUGH_BALL": {"PASS", "THROUGH_BALL"},
            "SHOOT": SHOT_TYPES,
        }.get(intent, set())
        hit = sum(n for t, n in cross[intent].items() if t in self_ev)
        w(f"  {intent:<20} self-consistent pairings: {hit}/{total} "
          f"({100.0*hit/total:.1f}%)")

    with open("_diag_intent_pairing.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[written to _diag_intent_pairing.txt]")


if __name__ == "__main__":
    sys.exit(main())
