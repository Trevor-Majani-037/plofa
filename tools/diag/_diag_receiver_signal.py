"""IS THERE ANY LEARNABLE SIGNAL IN *WHICH TEAMMATE* YOU PASS TO?

The premise test for "option 1" (a receiver-scoring / pointer head). The
question this answers, in one number:

    can a geometry model pick the RIGHT RECEIVER better than chance,
    and can it predict WHICH PASSES SUCCEED?

If yes -> there is real football to learn, a target head is worth building.
If no  -> no target head can ever help, and we skip the whole build.

WHY THIS PROBE INSTRUMENTS `_find_target` INSTEAD OF RECONSTRUCTING FROM THE
TIMELINE.  `brain_integration._find_target` receives the live `teammates` list
AND the live `position_engine`, so at the instant of the decision we read
exactly the candidate set the engine actually saw.  Reading positions after
`simulate()` returns attributes FINAL-WHISTLE state to a mid-match moment --
the documented trap in this project -- so the timeline cannot supply the
candidate set at all.  Pass events carry only the receiver that WAS chosen,
which is the label, not the alternatives.

OBSERVATION-ONLY.  The wrapper calls the original unchanged, consumes NO RNG,
writes nothing.  Matches replay byte-identically, so the arms are not needed:
there is only one arm.

Run:
    .venv\\Scripts\\python.exe _diag_receiver_signal.py [n_matches]
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter

import numpy as np

import brain_integration
from _diag_chance_coords import build_pair

OUT = "_diag_receiver_signal.txt"
SNAPSHOTS: list[dict] = []
_lines: list[str] = []

PASS_TYPES = {"PASS", "THROUGH_BALL", "CROSS", "SWITCH_PASS", "LONG_PASS",
              "FREEKICK_CROSS", "CORNER_TAKEN"}
FORWARD_ROLES = {"ST", "CF", "LW", "RW", "CAM"}


def p(msg: str = "", flush: bool = False) -> None:
    print(msg, flush=flush)
    _lines.append(msg)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION
# ─────────────────────────────────────────────────────────────
def _openness(tx, ty, defenders, pe):
    """Replicates brain_integration._best_forward's openness term exactly.

    Verified below by checking chosen == argmax(our value); if that agreement
    is high our replication is faithful, if it is low the features are not the
    engine's own and the whole probe would be measuring the wrong function.
    """
    open_val = 0.5
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = pe.get_position(d.name) if pe else (tx + 10, ty)
        dist = math.hypot(dx - tx, dy - ty)
        open_val = min(open_val, _clamp((dist - 1.5) / 8.5))
    return open_val


def install():
    original = brain_integration._find_target

    def wrapper(intent, player, x, y, teammates, defenders,
                position_engine, attacks_right, favored_flank=None):
        chosen = original(intent, player, x, y, teammates, defenders,
                          position_engine, attacks_right, favored_flank)
        try:
            pe = position_engine
            untracked = 0
            cands = []
            for t in teammates or []:
                pos = getattr(t, "position", "")
                if pos == "GK":
                    continue
                pxy = pe.tracked_position(t.name) if hasattr(pe, "tracked_position") else None
                if pxy is None:
                    tx, ty = pe.get_position(t.name) if pe else (x, y)
                    untracked += 1
                else:
                    tx, ty = pxy
                progress = (tx - x) if attacks_right else (x - tx)
                dist = math.hypot(tx - x, ty - y)
                open_val = _openness(tx, ty, defenders, pe)
                # the engine's own composite score
                val = _clamp(progress / 35.0) * 0.55 + open_val * 0.45
                eligible = progress >= 4.0
                cands.append({
                    "name": t.name, "pos": pos, "x": tx, "y": ty,
                    "progress": progress, "open": open_val, "dist": dist,
                    "dy": abs(ty - y), "val": val if eligible else -1.0,
                    "same_side": (ty < 34.0) == (y < 34.0),
                    "forward": pos in FORWARD_ROLES,
                })
            SNAPSHOTS.append({
                "carrier": getattr(player, "name", "?"),
                "cpos": getattr(player, "position", ""),
                "x": x, "y": y, "ar": bool(attacks_right),
                "intent": getattr(intent, "name", str(intent)),
                "chosen": getattr(chosen, "name", None) if chosen else None,
                "cands": cands, "untracked": untracked,
            })
        except Exception as exc:  # noqa: BLE001 - probe must never break a match
            SNAPSHOTS.append({"error": f"{type(exc).__name__}: {exc}"})
        return chosen

    brain_integration._find_target = wrapper
    return original


def restore(original):
    brain_integration._find_target = original


# ─────────────────────────────────────────────────────────────
# MODELS (numpy only; conditional logit = exactly the pointer
# head's own training objective)
# ─────────────────────────────────────────────────────────────
FEATS = ["progress", "open", "dist", "same_side", "dy", "forward", "prog_x_open"]


def featurize(c: dict, x: float, y: float) -> np.ndarray:
    prog = max(0.0, min(1.0, c["progress"] / 35.0))
    return np.array([
        prog,
        c["open"],
        min(1.0, c["dist"] / 40.0),
        1.0 if c["same_side"] else 0.0,
        min(1.0, c["dy"] / 55.0),
        1.0 if c["forward"] else 0.0,
        prog * c["open"],
    ], dtype=np.float64)


def fit_conditional_logit(groups, labels, l2=1e-3, iters=400, lr=0.5):
    """groups: list of (n_i, 7) arrays. labels: list of chosen index per group.

    Maximises sum_i log softmax(score)[chosen_i] -- a Plackett-Luce model with
    one positive per group.  This is precisely what a pointer head would be
    trained on, so the accuracy it reaches here is an honest ceiling preview.
    """
    w = np.zeros(FEATS_LEN)
    n = len(groups)
    for _ in range(iters):
        grad = np.zeros(FEATS_LEN)
        for X, y in zip(groups, labels):
            s = X @ w
            s -= s.max()
            e = np.exp(s)
            p = e / e.sum()
            grad += X[y] - (p[:, None] * X).sum(0)
        grad = grad / max(n, 1) - l2 * w
        w += lr * grad
    return w


FEATS_LEN = 7


def top1_accuracy(groups, labels, w):
    hit = 0
    ranks = []
    for X, y in zip(groups, labels):
        s = X @ w
        order = np.argsort(-s)
        rank = int(np.where(order == y)[0][0])
        ranks.append(rank)
        if rank == 0:
            hit += 1
    return hit / max(len(groups), 1), float(np.mean(ranks)), ranks


def auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if l in (0, 1)]
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    tot = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return tot / (len(pos) * len(neg))


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
    original = install()
    results = []
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(4000 + i)
            p(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
            SNAPSHOTS.clear()
            res = build_pair(h, a).simulate()
            results.append((res, list(SNAPSHOTS)))
            p(f"      score {res.home_goals}-{res.away_goals}, "
              f"{len(SNAPSHOTS)} target decisions", flush=True)
    finally:
        restore(original)

    snaps = [s for _, ss in results for s in ss if "error" not in s]
    errs = [s for _, ss in results for s in ss if "error" in s]
    p()
    p(f"  target decisions captured : {len(snaps)}   (wrapper errors {len(errs)})")
    if errs:
        p(f"    first error: {errs[0]['error']}")
    if not snaps:
        p("  NO SNAPSHOTS - the wrapper never fired. Not a result.")
        _dump()
        return

    # ── 0. FIDELITY: is our replication the engine's own scoring? ──
    # PER-INTENT.  `_find_target` dispatches to FOUR different lookups
    # (_nearest / _deepest / _best_forward / _wide), so comparing every intent
    # against _best_forward's composite measures the dispatch, not our
    # replication.  Comparing only on the intents that actually use
    # _best_forward is the control.
    agree = elig = 0
    by_intent = Counter()
    agree_by = Counter()
    for s in snaps:
        if not s["chosen"] or len(s["cands"]) < 2:
            continue
        if s["intent"] not in ("PROGRESSIVE_PASS", "THROUGH_BALL"):
            continue
        elig_c = [c for c in s["cands"] if c["val"] >= 0.0]
        if not elig_c:
            continue
        elig += 1
        by_intent[s["intent"]] += 1
        best = max(elig_c, key=lambda c: c["val"])
        if best["name"] == s["chosen"]:
            agree += 1
            agree_by[s["intent"]] += 1
    p(f"  0. FIDELITY on _best_forward intents only : {agree}/{elig} "
      f"({100.0*agree/max(elig,1):.1f}%)")
    for k in sorted(by_intent):
        p(f"       {k:<20} {agree_by[k]}/{by_intent[k]}")
    p("     (if this is high, our features ARE the engine's scoring inputs)")
    p(f"     intent mix over all decisions : "
      f"{dict(Counter(s['intent'] for s in snaps if s['chosen']))}")

    # ── JOIN snapshots -> PASS events for the outcome label ──────────
    joined = 0
    unmatched = 0
    outcome_of = {}
    for res, ss in results:
        tl = res.timeline
        passes = [e for e in tl if e.event_type.name in PASS_TYPES]
        used = [False] * len(passes)
        for s in ss:
            if "error" in s or not s["chosen"]:
                continue
            hit = None
            for j, e in enumerate(passes):
                if used[j]:
                    continue
                if e.player == s["carrier"] and (e.secondary_player or "") == s["chosen"]:
                    hit = (j, e)
                    break
            if hit is None:
                unmatched += 1
                continue
            j, e = hit
            used[j] = True
            joined += 1
            outcome_of[id(s)] = bool(e.outcome)
    total_with_choice = sum(1 for s in snaps if s["chosen"])
    p(f"  JOIN  snapshots with a target : {total_with_choice}")
    p(f"        matched to a PASS event : {joined}  ({100.0*joined/max(total_with_choice,1):.0f}%)")
    p(f"        unmatched               : {unmatched}")
    p("     (a low match rate would mean the label is unreliable - treat the")
    p("      learnability numbers below as unproven, not as a negative)")

    # ── 1. LEARNABILITY: pick the receiver (the pointer-head question) ──
    groups, labels, chances = [], [], []
    for s in snaps:
        if "error" in s or not s["chosen"]:
            continue
        cands = [c for c in s["cands"] if c["val"] >= 0.0]
        if len(cands) < 2:
            continue
        names = [c["name"] for c in cands]
        if s["chosen"] not in names:
            continue
        groups.append(np.stack([featurize(c, s["x"], s["y"]) for c in cands]))
        labels.append(names.index(s["chosen"]))
        chances.append(1.0 / len(cands))
    p()
    p(f"  1. RECEIVER CHOICE  (the pointer-head question)")
    p(f"     usable decisions (>=2 eligible candidates) : {len(groups)}")
    if len(groups) >= 40:
        w = fit_conditional_logit(groups, labels)
        acc, mean_rank, ranks = top1_accuracy(groups, labels, w)
        chance = float(np.mean(chances))
        p(f"     random baseline top-1        : {100.0*chance:5.1f}%")
        p(f"     GEOMETRY MODEL top-1          : {100.0*acc:5.1f}%")
        p(f"     mean rank of the true receiver: {mean_rank+1:.2f} of "
          f"{np.mean([len(g) for g in groups]):.2f} candidates")
        lift = acc / chance if chance else float("nan")
        p(f"     LIFT OVER CHANCE              : {lift:.2f}x")
        p(f"     learned weights ({', '.join(FEATS)}):")
        for f, v in zip(FEATS, w):
            p(f"        {f:<12} {v:+.3f}")

        # ── THE CIRCULARITY CONTROL (this is the whole section) ──────────
        # The chosen receiver was produced by `_best_forward`, which is a
        # DETERMINISTIC function of (progress, openness) -- the very features
        # above.  So a high top-1 is EXPECTED and PROVES NOTHING: the model
        # is reverse-engineering a hand-coded lookup, not discovering football.
        # The control: score the candidates with THE ENGINE'S OWN composite and
        # compare.  If the fitted model merely imitates the rule, the two
        # accuracies coincide.  If the model beat the rule, there would be
        # something beyond the lookup -- and there is not, because the rule
        # DEFINES the label.
        rule_hits = tot = 0
        for (X, y) in zip(groups, labels):
            comp = np.array([X[:, 0] * 0.55 + X[:, 1] * 0.45])  # progress*.55 + open*.45
            tot += 1
            if int(np.argmax(comp)) == y:
                rule_hits += 1
        rule_acc = rule_hits / max(tot, 1)
        p()
        p(f"     CONTROL — the ENGINE'S OWN composite top-1 : {100.0*rule_acc:5.1f}%")
        p(f"              the FITTED model top-1             : {100.0*acc:5.1f}%")
        p(f"              difference                          : "
          f"{100.0*(acc-rule_acc):+.1f} pts")
        p()
        p("     >>> SECTION 1 IS CIRCULAR BY CONSTRUCTION. The label IS the")
        p("         hand-coded lookup, and these features ARE its inputs, so")
        p("         lift over chance only measures how well the model imitates")
        p("         `_find_target`. It is NOT evidence of learnable football.")
        p("         The real question is section 2 (outcome) and 3 (coverage).")
    else:
        p("     too few usable decisions - INCONCLUSIVE, not a negative")

    # ── 2. OUTCOME: can geometry predict WHICH PASS SUCCEEDS? ─────────
    gs, ls, gsucc = [], [], []
    for s in snaps:
        if "error" in s or not s["chosen"] or id(s) not in outcome_of:
            continue
        cands = [c for c in s["cands"] if c["val"] >= 0.0]
        if len(cands) < 2:
            continue
        names = [c["name"] for c in cands]
        if s["chosen"] not in names:
            continue
        gs.append(np.stack([featurize(c, s["x"], s["y"]) for c in cands]))
        gsucc.append(outcome_of[id(s)])
    p()
    p(f"  2. PASS OUTCOME  (can geometry predict success OUT OF SAMPLE?)")
    p(f"     joined decisions : {len(gs)}")
    if len(gs) >= 60:
        # Honest protocol: fit on TRAIN, score on TEST. The previous version of
        # this probe scored training data and returned AUC 1.000, which is
        # 7 free parameters memorising the labels -- a perfect number that
        # means nothing.
        idx = list(range(len(gs)))
        random.Random(99).shuffle(idx)
        cut = int(len(idx) * 0.7)
        tr, te = idx[:cut], idx[cut:]
        Xtr = np.concatenate([gs[i] for i in tr])
        ytr = np.concatenate([[float(gsucc[i])] * gs[i].shape[0] for i in tr])
        Xte = np.concatenate([gs[i] for i in te])
        yte = np.concatenate([[float(gsucc[i])] * gs[i].shape[0] for i in te])
        wv = np.zeros(Xtr.shape[1])
        for _ in range(600):
            pr = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ wv, -30, 30)))
            wv -= 0.4 * ((Xtr * (pr - ytr)[:, None]).sum(0) / len(Xtr) + 1e-3 * wv)
        ptr = 1.0 / (1.0 + np.exp(-np.clip(Xte @ wv, -30, 30)))
        a_te = auc(ptr.tolist(), yte.tolist())
        base = float(ytr.mean())
        p(f"     train/test decisions      : {len(tr)}/{len(te)}")
        p(f"     train success rate         : {base:.3f}")
        p(f"     OUT-OF-SAMPLE outcome AUC  : {a_te:.3f}   (0.50 = no signal)")
        p(f"     >>> {'outcome is weakly predictable' if a_te >= 0.56 else 'outcome is NOT predictable out of sample'}"
          f"  (threshold 0.56)")
    else:
        p("     too few joined decisions - INCONCLUSIVE, not a negative")

    # ── 2b. OFF-POLICY COVERAGE (the decisive structural question) ────
    p()
    p(f"  2b. OFF-POLICY COVERAGE  (can a pointer head be TRAINED at all?)")
    p("     A value model needs, for a given state, to know which receiver")
    p("     SUCCEEDS. Observation gives us the outcome of the receiver the")
    p("     hand-coded rule ALREADY picked, and nothing about the others.")
    p("     If almost no candidate is ever picked, there is no training data.")
    seen = Counter()
    for s in snaps:
        if "error" in s or not s["chosen"]:
            continue
        cb = (round(s["x"] / 15.0), round(s["y"] / 17.0), s["cpos"][:2])
        ch = next((c for c in s["cands"] if c["name"] == s["chosen"]), None)
        if ch is None:
            continue
        key = (cb, round(ch["x"] / 15.0), round(ch["y"] / 17.0))
        seen[key] += 1
    total_cells = 0
    for s in snaps:
        if "error" in s or not s["chosen"]:
            continue
        cb = (round(s["x"] / 15.0), round(s["y"] / 17.0), s["cpos"][:2])
        for c in s["cands"]:
            if c["val"] >= 0.0:
                total_cells += 1
    covered = 0
    for s in snaps:
        if "error" in s or not s["chosen"]:
            continue
        cb = (round(s["x"] / 15.0), round(s["y"] / 17.0), s["cpos"][:2])
        for c in s["cands"]:
            if c["val"] >= 0.0 and seen.get((cb, round(c["x"] / 15.0),
                                             round(c["y"] / 17.0)), 0) > 0:
                covered += 1
    cov = covered / max(total_cells, 1)
    p(f"     (state, receiver) cell pairs offered : {total_cells}")
    p(f"     pairs ever actually chosen at least once: {covered} "
      f"({100.0*cov:.1f}%)")
    p(f"     distinct observed receiver cells     : {len(seen)}")
    p(f"     >>> {'SPARSE - most candidates have NO outcome label'
                if cov < 0.25 else 'reasonably dense'}")
    p("         Observation alone cannot train this head. The project already")
    p("         solved this exact problem for TeamPressBrain with forced")
    p("         counterfactuals (`team_offball_probe.py --intervene`), which")
    p("         set the decision BEFORE the rule sees it. The same lever")
    p("         would be required here.")

    # ── 3. SHAPE: for CBs, is a same-side option even AVAILABLE? ─────
    p()
    p(f"  3. SHAPE AVAILABILITY  (the CB corridor question)")
    p("     If the same-side man is not on the pitch, no target head can pass")
    p("     to him. This measures whether he is THERE.")
    cb = [s for s in snaps if "error" not in s and s["cpos"].startswith("CB")
          and s["chosen"] and s["x"] < 45.0]
    p(f"     CB decisions in own half with a target : {len(cb)}")
    if cb:
        avail, gaps, chosen_same, n_fwd_same = 0, [], 0, []
        for s in cb:
            cands = [c for c in s["cands"] if c["val"] >= 0.0]
            if not cands:
                continue
            same = [c for c in cands if c["same_side"] and c["forward"]]
            if same:
                avail += 1
                n_fwd_same.append(len(same))
            gaps.append(min(c["dy"] for c in cands))
            ch = next((c for c in cands if c["name"] == s["chosen"]), None)
            if ch is not None and ch["same_side"]:
                chosen_same += 1
        gaps.sort()
        p(f"     has a same-side FORWARD option : {avail}/{len(cb)} "
          f"({100.0*avail/len(cb):.0f}%)")
        p(f"     chose the same side            : {chosen_same}/{len(cb)} "
          f"({100.0*chosen_same/len(cb):.0f}%)")
        p(f"     nearest candidate lateral gap : median {gaps[len(gaps)//2]:.1f} m, "
          f"min {gaps[0]:.1f} m")
        p(f"     (real: a CB's same-side CDM is typically 15-30 m away laterally)")
    else:
        p("     no CB-in-own-half decisions captured - INCONCLUSIVE")

    _dump()


def _dump():
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()