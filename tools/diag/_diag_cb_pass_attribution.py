"""WHY DO CBs CARRY INSTEAD OF PASS?  — decision attribution probe (2026-11-23)

The user's report: after MD11 the engine's best CB makes 43.5 completed passes
per game (Joe Berv), against Sofascore's Dias 87 / Van Dijk 81 / Guehi 67 /
Bassey 64 / Dunk 64. Top CBs also make 18-22 CARRY events per match, where a
real ball-playing CB makes ~4-6. A CARRY and a PASS are mutually exclusive at
one decision (`event_chain.py:2244` chooses, `:2375` is the pass branch), so
this is a VERB question and therefore a brain question.

This probe does NOT change any weights and writes nothing. It runs the SAME
matches through five arms and measures how much of the CB carry bias survives
each intervention:

  baseline    - shipped path, nothing patched.
  plan_teacher- feed the plan block the distribution the v8 brains were
                distilled on: one_hot(possession_phase_for(x,y)) + conf 1.0
                + has_plan 1 (`surrogate_collect._teacher_forced_plan`).
                Production feeds the hysteretic TeamPlan summary, which is
                ALL ZEROS on the first touch of every possession.
  critic_off  - bypass `_apply_consequence` (the consequence critic predates
                the 2026-10-31 sensor change and saturates when losing).
  both        - plan_teacher + critic_off.
  cb_prior    - the candidate FIX, not just a diagnostic: at the ONE seam
                every path shares (`brain_integration._sample_from_probs`,
                immediately before the intent is sampled), multiply CARRY /
                DRIBBLE probability by `CB_PRIOR_FACTOR` for a CB in his own
                or middle third. The net is untouched; this is the same kind
                of documented seam as PLOFA_SHOT_HEAD.

For each arm it reports, for CBs in the own/mid third:
  decisions, CARRY share, mean P(CARRY), plan-block-zero rate, and the
  resulting timeline counts (CB passes, CB carries, team passes, and passes
  per possession).

Run:
    python -X utf8 tools\\diag\\_diag_cb_pass_attribution.py
    python -X utf8 tools\\diag\\_diag_cb_pass_attribution.py Lige-8 Justice 123 223
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as _np

from _diag_chance_coords import build_pair
from brain_integration import NeuralDecisionBrain, _INTENT_BY_INDEX
from decision_brain import CARRY_LIKE_INTENTS

# ─────────────────────────────────────────────────────────────
# TUNABLES
# ─────────────────────────────────────────────────────────────
HOME = "Lige-8"
AWAY = "Justice"
SEEDS = [123, 223, 323]
CB_PRIOR_FACTOR = 0.45     # candidate seam strength for the cb_prior arm
PASS_TYPES = {
    "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
    "GOAL_KICK", "CORNER_TAKEN", "FREEKICK_CROSS",
}
CARRY_IDX = [i for i, intent in enumerate(_INTENT_BY_INDEX)
             if intent in CARRY_LIKE_INTENTS]

# ─────────────────────────────────────────────────────────────
# CAPTURE
# ─────────────────────────────────────────────────────────────
_ORIG_DECIDE = NeuralDecisionBrain.decide
_ORIG_SAMPLE = None
_ORIG_BLOCKS = None
_ORIG_CONSEQ = None
_ORIG_MAKE_EVENT = None
_LOG: List[Dict[str, Any]] = []
_CTX: Dict[str, Dict[str, Any]] = {}
_CARRY_PROV: Counter = Counter()
_CB_NAMES: set = set()


def _make_event_probe(minute, event_type, team, player, phase, game_state,
                      **kwargs):
    """Count every emitted CARRY by provenance. A `micro_carry` is emitted
    at event_chain.py:1683 BEFORE the decision is even taken; a `main` carry
    comes from the CARRY verb at :2326. Exported JSON strips metadata, which
    is why this has to be observed live."""
    if getattr(event_type, "name", str(event_type)) == "CARRY" \
            and player in _CB_NAMES:
        md = kwargs.get("metadata") or {}
        _CARRY_PROV["micro" if md.get("micro_carry") else "main"] += 1
    return _ORIG_MAKE_EVENT(minute, event_type, team, player, phase,
                            game_state, **kwargs)


def _decide_capture(
    player: Any, x: float, y: float, teammates, defenders,
    position_engine, team_profile, under_pressure: bool,
    attacks_right: bool, game_state, minute: float = 45.0,
    soul: Any = None, record_trace: bool = False,
    coach_instructions: Any = None,
):
    """Force a trace and log CB decisions. Behaviour otherwise unchanged."""
    pos = str(getattr(player, "position", "") or "")
    ctx_key = None
    if pos == "CB":
        nx = float(x) if attacks_right else (105.0 - float(x))
        ctx_key = f"{id(player)}:{minute}"
        _CTX[ctx_key] = {"nx": nx}
    try:
        dec = _ORIG_DECIDE(
            player, x, y, teammates, defenders, position_engine,
            team_profile, under_pressure, attacks_right, game_state,
            minute, soul, True, coach_instructions,
        )
    finally:
        if ctx_key is not None:
            _CTX.pop(ctx_key, None)

    if pos == "CB":
        nx = float(x) if attacks_right else (105.0 - float(x))
        if nx < 70.0:                      # own / middle third only
            tr = getattr(dec, "trace", None) or {}
            sit = tr.get("situation") or {}
            plan = [float(v) for v in (sit.get("plan") or [])]
            probs = tr.get("output_probs") or {}
            chosen = tr.get("chosen_idx")
            intent = getattr(dec, "intent", None)
            action = getattr(dec, "action", "")
            _LOG.append({
                "player": getattr(player, "name", "?"),
                "team": str(getattr(player, "team_name", "") or ""),
                "minute": float(minute),
                "nx": nx,
                "under_pressure": bool(under_pressure),
                "chosen_idx": chosen,
                "intent": str(getattr(intent, "value", intent)),
                "action": action,
                "p_carry": sum(float(probs.get(_INTENT_BY_INDEX[i].value, 0.0))
                               for i in CARRY_IDX),
                "plan": plan,
                "plan_zero": bool(plan) and all(abs(v) < 1e-9 for v in plan),
                "consequence": tr.get("consequence"),
                "plan_adopted": (tr.get("plan") or {}).get("adopted"),
            })
    return dec


# ─────────────────────────────────────────────────────────────
# PATCHES (installed per arm, always restored)
# ─────────────────────────────────────────────────────────────
def _teacher_blocks(player, x, y, teammates, defenders,
                    position_engine, attacks_right, state):
    """The corpus' plan block, computed from the SAME live (x, y) the brain
    sees, instead of the team's hysteretic plan summary."""
    from possession_phases import possession_phase_for
    from surrogate_collect import _phase_index, _teacher_forced_plan

    role, mem, _ = _ORIG_BLOCKS(
        player, x, y, teammates, defenders, position_engine,
        attacks_right, state)
    phase = possession_phase_for(float(x), float(y), bool(attacks_right))
    plan = _teacher_forced_plan(_phase_index(phase))
    return role, mem, _np.asarray(plan, dtype=_np.float64)[:6]


def _no_consequence(probs, sensor_24, position, record_trace=False):
    return probs, None


def _tilted_sample(probs, temperature):
    """Thin, explicit prior at the last common seam before the verb is sampled."""
    ctx = None
    for v in _CTX.values():
        ctx = v
        break
    if ctx is not None and ctx.get("nx", 100.0) < 70.0:
        p = _np.asarray(probs, dtype=_np.float64).copy()
        for i in CARRY_IDX:
            p[i] *= CB_PRIOR_FACTOR
        s = float(p.sum())
        if s > 0:
            p = p / s
        return _ORIG_SAMPLE(p, temperature)
    return _ORIG_SAMPLE(probs, temperature)


def install(arm: str) -> None:
    global _ORIG_SAMPLE, _ORIG_BLOCKS, _ORIG_CONSEQ, _ORIG_MAKE_EVENT

    import brain_integration as bi
    import situational_context as sc
    from event_chain import BaseChain

    NeuralDecisionBrain.decide = staticmethod(_decide_capture)
    if _ORIG_MAKE_EVENT is None:
        _ORIG_MAKE_EVENT = BaseChain.make_event
    BaseChain.make_event = staticmethod(_make_event_probe)

    if arm in ("plan_teacher", "both"):
        _ORIG_BLOCKS = sc.situation_blocks
        sc.situation_blocks = _teacher_blocks
    if arm in ("critic_off", "both"):
        _ORIG_CONSEQ = bi._apply_consequence
        bi._apply_consequence = _no_consequence
    if arm == "cb_prior":
        _ORIG_SAMPLE = bi._sample_from_probs
        bi._sample_from_probs = _tilted_sample


def restore() -> None:
    import brain_integration as bi
    import situational_context as sc
    from event_chain import BaseChain

    NeuralDecisionBrain.decide = staticmethod(_ORIG_DECIDE)
    if _ORIG_MAKE_EVENT is not None:
        BaseChain.make_event = staticmethod(_ORIG_MAKE_EVENT)
    if _ORIG_BLOCKS is not None:
        sc.situation_blocks = _ORIG_BLOCKS
    if _ORIG_CONSEQ is not None:
        bi._apply_consequence = _ORIG_CONSEQ
    if _ORIG_SAMPLE is not None:
        bi._sample_from_probs = _ORIG_SAMPLE


# ─────────────────────────────────────────────────────────────
# MEASUREMENT
# ─────────────────────────────────────────────────────────────
def timeline_counts(res, cb_names: set, team: str) -> Dict[str, float]:
    cb_pass = cb_carry = 0
    team_pass = 0
    runs, cur_team, cur = [], None, []
    for e in res.timeline:
        et = getattr(e.event_type, "name", str(e.event_type))
        t = getattr(e, "team", "")
        if t != cur_team:
            if cur:
                runs.append((cur_team, cur))
            cur_team, cur = t, []
        if et in PASS_TYPES:
            cur.append(1)
            if t == team:
                team_pass += 1
            if getattr(e, "player", "") in cb_names:
                cb_pass += 1
        elif et == "CARRY" and getattr(e, "player", "") in cb_names:
            cb_carry += 1
    if cur:
        runs.append((cur_team, cur))
    return {
        "cb_pass": cb_pass,
        "cb_carry": cb_carry,
        "team_pass": team_pass,
        "pass_per_run": sum(len(r[1]) for r in runs) / max(1, len(runs)),
        "runs": len(runs),
    }


def run_arm(arm: str, home: str, away: str, seeds: List[int]) -> Dict[str, Any]:
    _LOG.clear()
    _CARRY_PROV.clear()
    tile: Dict[str, List[Dict[str, Any]]] = {}
    for seed in seeds:
        random.seed(seed)
        try:
            import numpy as _np2
            _np2.random.seed(seed % (2 ** 32))
        except Exception:
            pass
        eng = build_pair(home, away)
        _CB_NAMES.clear()
        for roster in (getattr(eng, "squads", {}) or {}).values():
            for p in (roster.get("starters") or []) + (roster.get("substitutes") or []):
                if str(getattr(p, "position", "")) == "CB":
                    _CB_NAMES.add(str(getattr(p, "name", "")))
        res = eng.simulate()
        tile[f"seed{seed}"] = timeline_counts(
            res,
            {d["player"] for d in _LOG},
            next((d["team"] for d in _LOG if d["team"]), home),
        )

    n = len(_LOG)
    carry = sum(1 for d in _LOG if d["action"] == "CARRY")
    zero = sum(1 for d in _LOG if d["plan_zero"])
    cons = sum(1 for d in _LOG if d["consequence"])
    intents = Counter(d["intent"] for d in _LOG)
    return {
        "arm": arm,
        "decisions": n,
        "carry_share": 100.0 * carry / max(1, n),
        "mean_p_carry": 100.0 * sum(d["p_carry"] for d in _LOG) / max(1, n),
        "plan_zero_pct": 100.0 * zero / max(1, n),
        "critic_trace_pct": 100.0 * cons / max(1, n),
        "intents": dict(intents.most_common(6)),
        "carry_micro": _CARRY_PROV.get("micro", 0),
        "carry_main": _CARRY_PROV.get("main", 0),
        "timeline": tile,
    }


def show(rows: List[Dict[str, Any]]) -> None:
    print(f"\n{'arm':<13} {'CB dec':>7} {'CARRY%':>7} {'P(CARRY)':>9} "
          f"{'plan0%':>7} {'critic%':>8} {'CB pass':>8} {'CARRY micro':>11} "
          f"{'CARRY main':>11} {'team pass':>10} {'pass/run':>9}")
    print("-" * 112)
    for r in rows:
        tl = r["timeline"]
        cb_p = sum(x["cb_pass"] for x in tl.values())
        tm_p = sum(x["team_pass"] for x in tl.values())
        ppr = sum(x["pass_per_run"] * x["runs"] for x in tl.values()) / max(
            1, sum(x["runs"] for x in tl.values()))
        print(f"{r['arm']:<13} {r['decisions']:>7} {r['carry_share']:>7.1f} "
              f"{r['mean_p_carry']:>9.1f} {r['plan_zero_pct']:>7.1f} "
              f"{r['critic_trace_pct']:>8.1f} {cb_p:>8} "
              f"{r['carry_micro']:>11} {r['carry_main']:>11} "
              f"{tm_p:>10} {ppr:>9.2f}")

    base = rows[0]
    print("\nATTRIBUTION vs baseline (percentage points of CARRY share):")
    for r in rows[1:]:
        print(f"  {r['arm']:<13} {r['carry_share'] - base['carry_share']:+6.1f} pp"
              f"   | intents {r['intents']}")
    print("\nREADING: a negative number means that intervention removed carries.")
    print("If plan_teacher / critic_off recover most of the gap, the bias is a")
    print("MIS-FED INPUT, not a learned preference — fix the feed, keep the net.")
    print("If only cb_prior moves it, the net genuinely prefers CARRY and a")
    print("documented seam (the same shape as PLOFA_SHOT_HEAD) is the honest fix.")


def main() -> None:
    args = sys.argv[1:]
    home = args[0] if len(args) > 0 else HOME
    away = args[1] if len(args) > 1 else AWAY
    seeds = [int(s) for s in args[2:]] or SEEDS
    arms_env = os.environ.get("PLOFA_PROBE_ARMS", "").strip()
    arms = ([a.strip() for a in arms_env.split(",") if a.strip()]
            or ["baseline", "plan_teacher", "critic_off", "both", "cb_prior"])

    print(f"PLOFA CB carry->pass attribution: {home} v {away}, seeds {seeds}")
    print(f"CARRY-like intent indices: {CARRY_IDX}  prior factor {CB_PRIOR_FACTOR}")
    rows: List[Dict[str, Any]] = []
    for arm in arms:
        try:
            install(arm)
            rows.append(run_arm(arm, home, away, seeds))
        finally:
            restore()
    show(rows)


if __name__ == "__main__":
    main()
