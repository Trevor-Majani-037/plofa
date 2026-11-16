"""Is the consequence critic blind to the goal?

WHY THIS EXISTS
---------------
`docs/FORESIGHT_PLAN_2026-11-08.md` §1 claims the shipped critic is "stale and
half-blind": `brains_trainer/critic_v5.json` was trained 2026-09-20, six weeks
before the on-ball sensor vector gained goal geometry on 2026-10-31.  Back then
slots 2-3 were `player_x/player_y` — exact duplicates of slots 0-1 for a
carrier — so they carried no information at all.

A claim about blindness has to be measured, not asserted.  This probe makes two
independent measurements against whatever critic you point it at, so the same
command produces the before/after pair for a retrain:

  A. RESPONSE  — hold one clear-chance state fixed and sweep slot 2 (goal
                 bearing) and slot 3 (mouth width) across their whole range.
                 If Q(SHOOT) barely moves while Q responds normally to slot 14
                 (goal distance, which the old critic *was* trained on), the
                 critic is blind to the angle and not to the distance.  The
                 distance sweep is the control: it proves the probe can detect
                 a response at all, so a flat bearing curve means blindness
                 rather than a broken harness.

  B. RANKING   — on real clear-chance states taken from the CURRENT corpus, ask
                 the football question directly: does the critic put SHOOT
                 above SAFE_PASS and CARRY?  The foresight plan's Level 0 gate
                 is exactly this, on held-out states.

Nothing here trains or writes anything.  Read-only by construction.

Usage:
    python tools/diag/_critic_goalvision_probe.py
    python tools/diag/_critic_goalvision_probe.py --critic brains_trainer/critic_v6_goalvision.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from football_brain import INTENT_LABELS  # noqa: E402
from critic_surrogate import (  # noqa: E402
    load_any_surrogate, critic_health, _prepare, _load_dataset, _print_health,
)

# The attacker roles that can plausibly be handed a clear chance.  A centre-back
# standing in the six-yard box is a set-piece accident, not a shooting decision.
ATTACK_ROLES = ("ST", "CF", "CAM", "LW", "RW")

SHOOT = INTENT_LABELS.index("SHOOT")
SAFE_PASS = INTENT_LABELS.index("SAFE_PASS")
CARRY = INTENT_LABELS.index("CARRY")


def _load_states(path):
    with open(path) as f:
        data = json.load(f)
    states = []
    for r in data:
        if not isinstance(r, list) or len(r) < 5:
            continue
        s = np.asarray(r[0], dtype=np.float64)
        if s.shape[0] < 24:
            continue
        states.append((s[:24], r[4]))
    return states


def _clear_chances(states):
    """Unpressured, central, near-goal states held by an attacker.

    Deliberately narrow: this is the situation where every coach in football
    agrees the shot is the right answer, so a critic that does not rank it
    first is wrong for a reason that needs no interpretation.
    """
    out = []
    for s, pos in states:
        if pos not in ATTACK_ROLES:
            continue
        if s[15] < 0.5:            # central lane
            continue
        if s[16] > 0.5:            # under pressure
            continue
        if s[11] < 0.5 and s[14] > 0.30:   # neither near goal nor close by distance
            continue
        out.append((s, pos))
    return out


def _q(critic, s, pos):
    # `load_any_surrogate` returns a CriticSurrogate wrapper for value critics
    # and a bare FitnessSurrogate otherwise; both expose `predict` one level
    # down.  Resolve once rather than at every call site.
    fn = getattr(critic, "predict", None)
    if fn is None:
        fn = critic.critic.predict
    return np.asarray(fn(s, pos), dtype=np.float64)


def _sweep(critic, base, pos, slot, values, label, fmt):
    rows = []
    for v in values:
        s = base.copy()
        s[slot] = v
        q = _q(critic, s, pos)
        rows.append({
            "value": v,
            "q_shoot": float(q[SHOOT]),
            "q_safe_pass": float(q[SAFE_PASS]),
            "q_carry": float(q[CARRY]),
            "argmax": INTENT_LABELS[int(np.argmax(q))],
            "q_mean": float(q.mean()),
        })
    qs = [r["q_shoot"] for r in rows]
    means = [r["q_mean"] for r in rows]
    print(f"\n  {label}  (slot [{slot}], {fmt})")
    print(f"    {'value':>6}  {'Q(SHOOT)':>9}  {'Q(SAFE)':>9}  {'Q(CARRY)':>9}  argmax")
    for r in rows:
        print(f"    {r['value']:>6.2f}  {r['q_shoot']:>+9.4f}  "
              f"{r['q_safe_pass']:>+9.4f}  {r['q_carry']:>+9.4f}  {r['argmax']}")
    return {
        "slot": slot,
        "label": label,
        "q_shoot_range": round(max(qs) - min(qs), 5),
        "q_mean_range": round(max(means) - min(means), 5),
        "argmax_changed": len({r["argmax"] for r in rows}) > 1,
    }


def main():
    p = argparse.ArgumentParser(description="Measure a critic's goal vision.")
    p.add_argument("--critic", default="brains_trainer/critic_v5.json")
    p.add_argument("--corpus", default="brains_trainer/corpus.raw.json")
    p.add_argument("--consequences",
                   default="brains_trainer/corpus_consequences.json",
                   help="episode-outcome dataset; enables the health verdict")
    p.add_argument("--max-states", type=int, default=400)
    args = p.parse_args()

    critic = load_any_surrogate(args.critic)
    meta = getattr(getattr(critic, "critic", None), "meta", None) or {}
    states = _load_states(args.corpus)
    chances = _clear_chances(states)
    if len(chances) > args.max_states:
        step = len(chances) / args.max_states
        chances = [chances[int(i * step)] for i in range(args.max_states)]

    print("=" * 70)
    print(f"critic : {args.critic}")
    print(f"schema : {meta.get('sensor_schema', '?')}   "
          f"trained: {meta.get('training_method', '?')} / {meta.get('target_mode', '?')}")
    print(f"corpus : {args.corpus} ({len(states)} decisions)")
    print(f"clear chances (central, unpressured, attacker, near goal): {len(chances)}")
    print("=" * 70)

    if not chances:
        print("\nNo clear-chance states in this corpus — nothing to measure.")
        return

    # ── B. RANKING on real clear chances ────────────────────────────────
    qs = np.stack([_q(critic, s, pos) for s, pos in chances])
    mean_q = qs.mean(axis=0)
    argmaxes = qs.argmax(axis=1)
    n = len(chances)

    print("\nB. RANKING — real clear chances")
    print(f"  {'intent':<18} {'mean Q':>9}   share of states where it is argmax")
    for ik in np.argsort(-mean_q):
        if mean_q[ik] < mean_q.max() - 0.0 and ik != SHOOT and ik != SAFE_PASS \
                and ik != CARRY and mean_q[ik] < mean_q.max() - 0.25:
            continue
        print(f"  {INTENT_LABELS[ik]:<18} {mean_q[ik]:>+9.4f}   "
              f"{float((argmaxes == ik).mean()):>6.1%}")

    shoot_first = float((argmaxes == SHOOT).mean())
    beat_safe = float((qs[:, SHOOT] > qs[:, SAFE_PASS]).mean())
    beat_carry = float((qs[:, SHOOT] > qs[:, CARRY]).mean())
    others = [i for i in range(len(INTENT_LABELS)) if i != SHOOT]
    beat_best_other = float((qs[:, SHOOT] > qs[:, others].max(axis=1)).mean())

    print(f"\n  SHOOT is the critic's argmax          : {shoot_first:>6.1%}")
    print(f"  Q(SHOOT) > Q(SAFE_PASS)               : {beat_safe:>6.1%}")
    print(f"  Q(SHOOT) > Q(CARRY)                   : {beat_carry:>6.1%}")
    print(f"  Q(SHOOT) > every other intent         : {beat_best_other:>6.1%}")

    # ── A. RESPONSE to goal geometry ────────────────────────────────────
    # One representative state: the median-bearing clear chance, so the sweep
    # starts from something real rather than from a hand-built vector.
    bearings = np.array([s[2] for s, _ in chances])
    med = float(np.median(bearings))
    base, pos = min(chances, key=lambda sp: abs(sp[0][2] - med))
    print(f"\nA. RESPONSE — sweep from one real clear chance ({pos}, "
          f"bearing {base[2]:+.3f}, mouth {base[3]:.3f}, "
          f"dist {base[14]:.3f})")

    resp = [
        _sweep(critic, base, pos, 2, [-1.0, -0.5, 0.0, 0.5, 1.0],
               "goal BEARING (slot 2) — 0 = dead ahead, +/-1 = goal line",
               "{:+.2f}"),
        _sweep(critic, base, pos, 3, [0.0, 0.25, 0.5, 0.75, 1.0],
               "goal MOUTH WIDTH (slot 3) — 1 = penalty-spot sight",
               "{:.2f}"),
        _sweep(critic, base, pos, 14, [0.10, 0.20, 0.30, 0.40, 0.50],
               "goal DISTANCE (slot 14) — the control",
               "{:.2f}"),
    ]

    # ── C. HEALTH, on the episode-outcome dataset it was trained for ─────
    # The sweeps above ask "does it see the goal?".  This asks the prior
    # question: "is anything alive in there?"  A flat network answers both
    # the same way, so the two measurements must be read together.
    health = None
    if os.path.exists(args.consequences):
        recs = _load_dataset(args.consequences)
        if recs:
            X, I, y, w, means, stds = _prepare(recs)
            n = len(recs)
            hold = max(1, int(n * 0.2))
            order = np.arange(n)
            np.random.default_rng(123 + 999).shuffle(order)
            health = critic_health(
                getattr(critic, "critic", critic), X, I, y,
                order[hold:], order[:hold])
            print(f"\nC. HEALTH — on {n} episode-outcome records "
                  f"({args.consequences})")
            _print_health(health)

            # How much of Q is just the intent?  A critic that is mostly a
            # lookup table cannot be conditioning on the goal, however wide
            # its output range.
            qall = getattr(critic, "critic", critic).forward_x(X)
            between = qall.mean(axis=0)
            share = float(between.var() / qall.var()) if qall.var() > 0 else 1.0
            print(f"  intent alone explains {share:.1%} of Q variance")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    bearing = resp[0]
    mouth = resp[1]
    dist = resp[2]
    control = dist["q_shoot_range"]
    print(f"  Q(SHOOT) moves {bearing['q_shoot_range']:+.4f} across the whole "
          f"bearing range")
    print(f"  Q(SHOOT) moves {mouth['q_shoot_range']:+.4f} across the whole "
          f"mouth range")
    print(f"  Q(SHOOT) moves {control:+.4f} across the distance control")
    blind_bearing = bearing["q_shoot_range"] < 0.05
    blind_mouth = mouth["q_shoot_range"] < 0.05
    control_works = control > 0.05
    if blind_bearing and blind_mouth:
        if control_works:
            print("\n  BLIND TO THE GOAL'S SHAPE, NOT ITS DISTANCE — it sees how")
            print("  far away the goal is and cannot see where it is or how wide.")
        else:
            print("\n  FLAT ON EVERYTHING — the sweep found no response at all,")
            print("  not even to distance. That is not a broken harness: a dead")
            print("  network answers every question with the same number.")
    elif blind_bearing or blind_mouth:
        which = "bearing" if blind_bearing else "mouth width"
        print(f"\n  PARTIALLY BLIND — flat on {which}.")
    else:
        print("\n  SEES THE GOAL — both geometry slots move Q(SHOOT) materially.")
    print(f"  SHOOT argmax on {shoot_first:.1%} of clear chances "
          f"(football says it should be most of them).")
    if shoot_first >= 0.95 and (blind_bearing or blind_mouth):
        print("\n  THAT 100% IS NOT VISION. The critic ranks SHOOT first everywhere,")
        print("  including where the goal is invisible, because SHOOT carries the")
        print("  highest intent prior in the corpus. A ranking that never changes")
        print("  is a constant, and a constant is not a judgement — so the Level 0")
        print("  gate as originally written would pass a dead critic. Use C. HEALTH.")
    if health is not None and health["verdict"] == "FAIL":
        print(f"\n  HEALTH: FAIL — {len(health['failures'])} functional check(s).")
        print("  This critic is not fit to steer a decision at any blend.")


if __name__ == "__main__":
    main()
