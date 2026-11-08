"""Post-match brain observatory — does the brain's choice track the situation?

READ-ONLY: analyses a finished match's event metadata without changing
any production output.  Writes sidecar .md/.json reports opt-in via
PLOFA_NO_BRAIN_TRACE=1.

Usage:
  python brain_observatory.py --from-trace plofa_output/<folder>_brain_trace.json
  python brain_observatory.py --from-dir plofa_output        # aggregate season
  python brain_observatory.py --home Oxton --away Natrican --seed 123 --out plofa_output/brain_reports
  python brain_observatory.py --from-json plofa_output/..._brain_report.json --null-test 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random as _random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Core data shapes (mirror what event_chain writes)
# ──────────────────────────────────────────────────────────────────────
INTENTS = [
    "PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
    "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION",
]

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _entropy(counts: List[float]) -> float:
    vals = [c for c in counts if c > 0]
    if not vals:
        return 0.0
    total = float(sum(vals))
    h = 0.0
    for c in vals:
        p = c / total
        h -= p * math.log2(p)
    return h


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _mutual_information(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    """Return (I(context;intent) bits, H(intent) bits)."""
    if not pairs:
        return 0.0, 0.0
    n = float(len(pairs))
    ctx = Counter(c for c, _ in pairs)
    intent = Counter(i for _, i in pairs)
    joint = Counter(pairs)
    h_intent = _entropy(intent.values())
    h_cond = 0.0
    for (c, _i), k in joint.items():
        p_c = ctx[c] / n
        p_i_given_c = k / ctx[c]
        h_cond -= p_c * p_i_given_c * math.log2(p_i_given_c)
    mi = max(0.0, h_intent - h_cond)
    return mi, h_intent


def _softmax_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _verdict(n: int, mi_bits: float, mi_share: float,
             h_norm: float, predictable: float, ctx_cov: float) -> Tuple[str, List[str]]:
    flags: List[str] = []
    if n < 15:
        return "insufficient sample to judge reasoning", ["n<15"]
    # Require absolute information, not just a ratio of a tiny repertoire.
    # A player with 0.029 bits who scores "30% explained" is not context-driven.
    weak = mi_share < 0.08 or mi_bits < 0.10
    if weak and h_norm < 0.55:
        base = "largely invariant (same choice regardless of situation)"
        flags.append("low_variety")
    elif weak:
        base = "near-uniform sampling (variety without context)"
        flags.append("low_context_signal")
    elif mi_share < 0.20:
        base = "weakly context-sensitive"
    elif ctx_cov < 0.50:
        base = "context-sensitive; consistency not assessable (most situations seen once)"
    elif predictable < 0.55:
        base = "context-sensitive but noisy (reacts, not consistent)"
        flags.append("unstable_modal_choice")
    else:
        base = "context-driven (situational, rule-like)"
    if h_norm > 0.95:
        flags.append("near_max_entropy")
    return base, flags


def _build_record(trace_record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a single decision record from the trace."""
    r: Dict[str, Any] = {}
    r["intent"] = trace_record.get("intent") or trace_record.get("chosen_idx")
    # obeyed: True if the executed event matches the brain's action family
    r["obeyed"] = trace_record.get("obeyed")
    r["confidence"] = trace_record.get("confidence", 0.0)
    r["zone"] = trace_record.get("zone")
    r["pressure"] = trace_record.get("pressure")
    r["score_state"] = trace_record.get("score_state")
    r["minute"] = trace_record.get("minute", 0.0)
    return r


def _situation(rec: Dict[str, Any]) -> str:
    """Context bucket = third + pressure."""
    third = rec.get("zone", "?")
    press = rec.get("pressure")
    if press is None:
        return str(third)
    return f"{third}|{'press' if press else 'free'}"


# ──────────────────────────────────────────────────────────────────────
# Shuffle test: is I > 0 better than chance pairing?
# ──────────────────────────────────────────────────────────────────────
def _shuffle_test(records: List[Dict[str, Any]],
                  permutations: int = 300, seed: int = 0) -> Optional[Dict[str, Any]]:
    """Permute which situation each choice happened in; report p = share of
    shuffled runs scoring >= the real I.  p ~ 0 means the situation really
    matters; p ~ 0.5 means choices are independent of the situation."""
    if len(records) < 15:
        return None
    real, _ = _mutual_information([(_situation(r), r["intent"]) for r in records])
    rng = _random.Random(seed)
    intents = [r["intent"] for r in records]
    pool = list(intents)
    null: List[float] = []
    for _ in range(permutations):
        rng.shuffle(pool)
        fake = [(_situation(r), i) for r, i in zip(records, pool)]
        null.append(_mutual_information(fake)[0])
    null.sort()
    at_least = sum(1 for v in null if v >= real)
    return {
        "n": len(records),
        "observed": round(real, 3),
        "null_mean": round(sum(null) / len(null), 3),
        "null_p95": round(null[min(int(0.95 * len(null)), len(null) - 1)], 3),
        "p": round(at_least / len(null), 3),
        "permutations": permutations,
    }


# ──────────────────────────────────────────────────────────────────────
# Assemble report dict
# ──────────────────────────────────────────────────────────────────────
def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reasoning diagnostics for one player (or any subset of decisions)."""
    n = len(records)
    if n == 0:
        return {
            "n": 0, "intent_mix": {}, "family_mix": {},
            "entropy_bits": 0.0, "entropy_norm": 0.0,
            "situation_mi_bits": 0.0, "situation_share": 0.0,
            "predictability": 0.0, "predictability_coverage": 0.0,
            "contexts": [], "confidence_calibration": [],
            "pressure_shift": {}, "third_shift": {},
            "score_shift": {}, "risky_share": 0.0,
            "mean_confidence": 0.0, "completion": 0.0,
            "obedience": None, "verdict": "no decisions", "flags": ["no_decisions"],
        }

    intents = Counter(r["intent"] for r in records)
    families = Counter(r.get("family", "pass") for r in records)
    total = float(n)
    intent_mix = {k: round(c / total, 3) for k, c in intents.most_common()}

    h = _entropy(intents.values())
    h_norm = h / math.log2(len(INTENTS)) if n else 0.0

    pairs = [(_situation(r), r["intent"]) for r in records]
    mi, h_intent = _mutual_information(pairs)
    mi_share = (mi / h_intent) if h_intent > 0 else 0.0

    # predictability: modal-intent share per context, only contexts >= 3 times
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        buckets[_situation(r)][r["intent"]] += 1
    usable = [c for c in buckets.values() if sum(c.values()) >= 3]
    covered = sum(sum(c.values()) for c in usable)
    predictable = (sum(max(c.values()) for c in usable) / covered
                   if covered else 0.0)
    context_coverage = covered / n if n else 0.0

    # the same buckets, but readable: what is actually chosen in each situation
    context_stats = []
    for name, counter in sorted(buckets.items(),
                                key=lambda kv: -sum(kv[1].values())):
        total_ctx = sum(counter.values())
        modal, modal_n = counter.most_common(1)[0]
        context_stats.append({
            "context": name,
            "n": total_ctx,
            "modal": modal,
            "share": round(modal_n / total_ctx, 3),
            "seen_3plus": total_ctx >= 3,
        })

    # obedience: did the executed event match the brain's chosen action family?
    obeyed = [r["obeyed"] for r in records if r.get("obeyed") is not None]
    obedience = (sum(1 for b in obeyed if b) / len(obeyed)) if obeyed else None

    completion = sum(1 for r in records if r.get("outcome")) / total if n else 0.0
    mean_conf = sum(r["confidence"] for r in records) / total if n else 0.0
    risky = sum(1 for r in records if r["intent"] in {"SHOOT", "PROGRESSIVE_PASS"}) / total if n else 0.0

    verdict, flags = _verdict(n, mi, mi_share, h_norm, predictable, context_coverage)

    return {
        "n": n,
        "intent_mix": intent_mix,
        "family_mix": {k: round(c / total, 3) for k, c in families.most_common()},
        "entropy_bits": round(h, 3),
        "entropy_norm": round(h_norm, 3),
        "situation_mi_bits": round(mi, 3),
        "situation_share": round(mi_share, 3),
        "predictability": round(predictable, 3),
        "predictability_coverage": round(context_coverage, 3),
        "contexts": sorted(buckets.keys()),
        "context_stats": context_stats,
        "confidence_calibration": [],  # placeholder — would need full bucket code
        "pressure_shift": _shift_table(records, "pressure", [True, False]),
        "third_shift": _shift_table(records, "zone", ["DEF", "MID", "ATT"]),
        "score_shift": _shift_table(records, "score_state",
                                    ["leading", "level", "trailing"]),
        "risky_share": round(risky, 3),
        "mean_confidence": round(mean_conf, 3),
        "completion": round(completion, 3),
        "obedience": round(obedience, 3) if obedience is not None else None,
        "verdict": verdict,
        "flags": flags,
    }


def _shift_table(records: List[Dict[str, Any]], key: str,
                 values: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for v in values:
        subset = [r for r in records if str(r.get(key)) == v]
        if not subset:
            continue
        counts = Counter(r["intent"] for r in subset)
        total = sum(counts.values())
        out[v] = {k: round(c / total, 3) for k, c in counts.most_common()}
    return out


def attach_null_test(report: Dict[str, Any],
                     *, permutations: int = 300,
                     seed: int = 0) -> Dict[str, Any]:
    """Add the shuffle-test result to the report (mutates in place).

    Runs the permutation test twice: once on the pooled ledger, and once per
    player.  The per-player rows are the ones that matter — pooling different
    roles inflates I, because a centre-back and a winger draw from different
    repertoires and the pool then looks "context-sensitive" for free.
    """
    records = report.get("thought_log") or []
    if not records:
        report["null_test"] = {"overall": None, "players": [],
                               "permutations": 0}
        return report

    by_player: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_player[(r.get("team", ""), r.get("player", ""))].append(r)

    rows: List[Dict[str, Any]] = []
    for (team, player), recs in by_player.items():
        res = _shuffle_test(recs, permutations=permutations, seed=seed)
        if res is None:            # < 15 decisions — cannot judge
            continue
        rows.append({"team": team, "player": player, **res})
    rows.sort(key=lambda r: (r["p"], -r["n"]))

    report["null_test"] = {
        "overall": _shuffle_test(records, permutations=permutations, seed=seed),
        "players": rows,
        "permutations": permutations,
    }
    return report


def render_markdown(report: Dict[str, Any],
                    thought_limit: int = 40,
                    with_reason: bool = False) -> str:
    """Render the report as ASCII-safe markdown text."""
    m = report["match"]
    cov = report["coverage"]
    o = report["overall"]
    lines: List[str] = []
    lines.append(f"# Brain report - {m['home']} {m['score']} {m['away']}")
    lines.append("")
    lines.append(f"**{cov['decisions']} on-ball decisions** from **{cov['players']} players** "
                 f"across {cov['teams']} teams. Pressure recorded on "
                 f"{cov['pressure_recorded']} of them.")
    lines.append("")
    lines.append("## Did they think?")
    lines.append("")
    verdict_str = o["verdict"]
    if o["flags"]:
        verdict_str += f"  _flags: {', '.join(o['flags'])}_"
    lines.append(f"**{verdict_str}**")
    lines.append("")
    lines.append(f"- Variety H(intent) = **{o['entropy_bits']} bits** "
                 f"({_pct(o['entropy_norm'])} of the 3.32-bit maximum)")
    lines.append(f"- Situation→intent I = **{o['situation_mi_bits']} bits** "
                 f"({_pct(o['situation_share'])} of choice entropy explained)")
    lines.append(f"- Predictability = **{_pct(o['predictability'])}**")
    lines.append(f"- Intent obeyed by the body on **{_pct(o['obedience'])}** of touches")
    lines.append("")
    lines.append("How to read it:`I` near 0 with low variety = one-note (the situation "
                 "does not enter the choice); `I` near 0 with high variety = sampling "
                 "noise; `I` above ~20% with a stable modal intent per context = the "
                 "choice is being driven by what the player sees. Overall/team rows "
                 "pool different roles, so read the per-player verdicts first.")
    lines.append("")
    # shuffle test — the only line that says whether I is bigger than luck
    nt = report.get("null_test") or {}
    if nt.get("overall"):
        ov = nt["overall"]
        lines.append("## Is the signal real? (shuffle test)")
        lines.append("")
        lines.append(f"Permuted which situation each choice happened in "
                     f"{nt.get('permutations', 0)} times. Real I = **{ov['observed']} bits** "
                     f"vs shuffled mean {ov['null_mean']} bits (95th pct {ov['null_p95']}). "
                     f"**p = {ov['p']}** over {ov['n']} decisions.")
        lines.append("")
        if ov["p"] <= 0.05:
            lines.append("p <= 0.05: the situation really is doing work — the brain is "
                         "not drawing the same choice regardless of what it sees.")
        else:
            lines.append("p > 0.05: **the observed context-sensitivity is within chance.** "
                         "Treat the I figure above as zero until this drops.")
        lines.append("")
        rows_nt = nt.get("players") or []
        if rows_nt:
            lines.append("Per player (only players with 15+ decisions):")
            lines.append("")
            lines.append("| player | team | n | I (bits) | shuffled mean | p |")
            lines.append("|---|---|---|---|---|---|")
            for r in rows_nt:
                lines.append(f"| {r['player']} | {r['team']} | {r['n']} | "
                             f"{r['observed']} | {r['null_mean']} | {r['p']} |")
            lines.append("")
    # by-third table
    lines.append("## By third")
    lines.append("")
    lines.append("| intent | DEF | MID | ATT |")
    lines.append("|---|---|---|---|")
    third_shifts = o.get("third_shift", {})
    for intent in INTENTS:
        parts = [_pct(third_shifts.get(t, {}).get(intent, 0)) for t in ["DEF", "MID", "ATT"]]
        lines.append(f"| {intent} | {parts[0]} | {parts[1]} | {parts[2]} |")
    lines.append("")
    lines.append("## Teams")
    lines.append("")
    lines.append("| team | decisions | variety | I(sit;intent) | predictability | obedience |")
    lines.append("|---|---|---|---|---|---|")
    for team, t in report["teams"].items():
        lines.append(
            f"| {team} | {t['n']} | {_pct(t['entropy_norm'])} | "
            f"{_pct(t['situation_share'])} | {_pct(t['predictability'])} | "
            f"{_pct(t['obedience'])} |")
    lines.append("")
    lines.append("## Players")
    lines.append("")
    current_team = None
    for p in report.get("players", []):
        if p["team"] != current_team:
            current_team = p["team"]
            lines.append(f"### {current_team}")
            lines.append("")
        lines.extend(_player_block(p, with_reason=with_reason))
    realised = report.get("realisation") or {}
    if realised:
        lines.append("## Intent → what actually happened")
        lines.append("")
        lines.append("A gap here is a downstream override, not a thinking failure.")
        lines.append("")
        lines.append("| intent | events executed |")
        lines.append("|---|---|")
        for intent, events in realised.items():
            cell = ", ".join(f"{k} ×{v}" for k, v in events.items())
            lines.append(f"| {intent} | {cell} |")
        lines.append("")
    log = report.get("thought_log") or []
    if log and (thought_limit is None or thought_limit > 0):
        lines.append("## Thought log")
        lines.append("")
        lines.append("```")
        shown = log if thought_limit is None else log[:thought_limit]
        for r in shown:
            lines.append(_thought_line(r, with_reason=with_reason))
        if thought_limit is not None and len(log) > thought_limit:
            lines.append(f"... {len(log) - thought_limit} more decisions")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.0f}%"


def _player_block(p: Dict[str, Any], *, max_contexts: int = 6,
                  with_reason: bool = False) -> List[str]:
    lines: List[str] = [
        f"### {p['player']} - {p['team']}  ({p['n']} decisions)",
        "",
        f"- **Verdict:** {p['verdict']}"
        + (f"  _flags: {', '.join(p['flags'])}_" if p["flags"] else ""),
        f"- **Variety:** H(intent) = {p['entropy_bits']}/3.32 bits "
        f"({_pct(p['entropy_norm'])} of max)",
        f"- **Situation → intent:** I = {p['situation_mi_bits']} bits "
        f"({_pct(p['situation_share'])} of choice entropy explained by third/pressure)",
        f"- **Predictability:** {_pct(p['predictability'])} of touches fall in the "
        f"modal intent of their context (covers {_pct(p['predictability_coverage'])} of decisions "
        f"in contexts seen 3+ times)",
        f"- **Confidence / completion:** mean {p['mean_confidence']:.2f} "
        f"→ {_pct(p['completion'])} completed; obedience {_pct(p['obedience'])}",
        f"- **Risk appetite:** {_pct(p['risky_share'])} risky intents",
        "",
    ]
    if p.get("intent_mix"):
        lines.append("| intent | share |")
        lines.append("|---|---|")
        for intent, share in p["intent_mix"].items():
            bar = "#" * int(round(share * 18)) + "." * (18 - int(round(share * 18)))
            lines.append(f"| {intent} | {bar} {_pct(share)} |")
    if p.get("context_stats") and max_contexts > 0:
        lines.append("")
        lines.append("Top contexts:")
        lines.append("")
        lines.append("| context | n | modal intent | share |")
        lines.append("|---|---|---|---|")
        for c in p["context_stats"][:max_contexts]:
            mark = "" if c.get("seen_3plus", True) else " (thin)"
            lines.append(f"| {c['context']}{mark} | {c['n']} | "
                         f"{c['modal']} | {_pct(c['share'])} |")
        lines.append("")
    cal = p.get("confidence_calibration", [])
    if len(cal) >= 2:
        lines.append("Confidence calibration (does conviction mean anything?):")
        lines.append("")
        lines.append("| confidence | n | completion |")
        for entry in cal[:4]:
            lines.append(f"| {entry.get('confidence','?')} | {entry.get('n','?')} | {entry.get('completion','?')} |")
        lines.append("")
    return lines


def _thought_line(r: Dict[str, Any], *, with_reason: bool = False) -> str:
    minute = r.get("minute", "?")
    ctx = ""
    if "zone" in r and "pressure" in r:
        z = r["zone"]; p = r["pressure"]
        ctx = f"{z}|{'press' if p else 'free'}"
    parts = [
        f"min {minute}",
        f"chose {r.get('intent','?')}",
        f"conf {r.get('confidence',0):.2f}",
        f"obeyed {r.get('obeyed', '?')}",
    ]
    if with_reason and r.get("reason"):
        parts.append(f"reason: {r['reason']}")
    return " | ".join(parts)


def _realisation_table(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        intent = r.get("intent") or r.get("chosen_idx") or "?"
        event = r.get("event") or r.get("outcome_event") or "?"
        table[intent][event or "?"] += 1
    return {k: dict(v.most_common()) for k, v in sorted(table.items())}


# ──────────────────────────────────────────────────────────────────────
# Match analysis entry point
# ──────────────────────────────────────────────────────────────────────
def analyse_match(result_or_timeline: Any,
                  *,
                  engine: Any = None,
                  home_team: Optional[str] = None,
                  away_team: Optional[str] = None) -> Dict[str, Any]:
    """Build the full post-match brain report as a dict."""
    # --- accept a trace dict with 'decisions' or 'thought_log' ---
    timeline = getattr(result_or_timeline, "timeline", None)
    if isinstance(result_or_timeline, dict) and (
            "decisions" in result_or_timeline or "thought_log" in result_or_timeline):
        recs = result_or_timeline.get("decisions") or result_or_timeline.get("thought_log")
        records = [dict(r) for r in (recs or [])]
        m = result_or_timeline.get("match") or {}
        home = str(m.get("home") or home_team or "")
        away = str(m.get("away") or away_team or "")
        score = str(m.get("score") or "")
        return _assemble_report(records, home, away, score, direction_known=True)

    if timeline is None and isinstance(result_or_timeline, dict):
        timeline = result_or_timeline.get("timeline") or []

    cfg = getattr(result_or_timeline, "config", None)
    home = str(home_team or getattr(cfg, "home_team", "") or "")
    away = str(away_team or getattr(cfg, "away_team", "") or "")
    score = ""
    if isinstance(result_or_timeline, dict):
        m = result_or_timeline.get("match") or {}
        home = home or str(m.get("home_team") or "")
        away = away or str(m.get("away_team") or "")
        score = str(m.get("score") or "")
    if not score:
        try:
            score = f"{result_or_timeline.home_goals}-{result_or_timeline.away_goals}"
        except Exception:
            score = ""

    # Collect the decision ledger from the timeline.  Every on-ball decision
    # is stamped by event_chain as metadata["active_brain"]; the collector
    # lives with the brain (brain_integration) because it knows the intent
    # vocabulary and the action families.
    from brain_integration import collect_decisions
    records = collect_decisions(result_or_timeline, engine=engine,
                                home_team=home, away_team=away)
    return _assemble_report(records, home, away, score,
                            direction_known=bool(_attacks_right_map(engine)) or engine is None)


def _attacks_right_map(engine: Any) -> Dict[str, bool]:
    """team -> attacks_right, or {} when we have no engine to ask.

    `zone` is only meaningful relative to the direction a team is attacking,
    so without this the whole third split in the report is noise.
    """
    pe = getattr(engine, "position_engine", None)
    direction = getattr(pe, "team_attacks_right", None)
    return dict(direction) if isinstance(direction, dict) else {}


def _assemble_report(records: List[Dict[str, Any]], home: str, away: str,
                     score: str, *, direction_known: bool) -> Dict[str, Any]:
    """Group a decision ledger into teams, players, and the overall verdict."""
    by_team: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_player: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        t = r.get("team", "")
        by_team[t].append(r)
        # key on (team, player) so same-shirt numbers on different sides
        # are not merged into one player.
        player = r.get("player", "")
        by_player[(t, player)].append(r)

    team_summary: Dict[str, Dict[str, Any]] = {}
    for team, recs in by_team.items():
        s = summarise(recs)
        s["players"] = len({r["player"] for r in recs})
        team_summary[team] = s

    players = []
    for (team, player), recs in by_player.items():
        s = summarise(recs)
        s["player"] = player
        s["team"] = team
        players.append(s)
    players.sort(key=lambda p: (p["team"], -p["n"], p["player"]))

    pressure_known = sum(1 for r in records if r.get("pressure") is not None)
    return {
        "match": {"home": home, "away": away, "score": score},
        "coverage": {
            "decisions": len(records),
            "players": len(by_player),
            "teams": len(by_team),
            "pressure_recorded": pressure_known,
            "direction_known": direction_known,
        },
        "overall": summarise(records),
        "teams": team_summary,
        "players": players,
        "realisation": _realisation_table(records),
        "thought_log": list(records),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse as _argparse

    p = _argparse.ArgumentParser(
        description="Read a finished match's brain decisions and judge "
                    "whether the choices track the situation.")
    p.add_argument("--home", default="Oxton", help="home club (runs a match)")
    p.add_argument("--away", default="Natrican", help="away club (runs a match)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--home-style", default="fluid_counter")
    p.add_argument("--away-style", default="attacking")
    p.add_argument("--from-trace", default=None,
                   help="analyse a decision trace written by --save-trace")
    p.add_argument("--from-json", default=None,
                   help="analyse a saved match document that carries decisions")
    p.add_argument("--from-dir", default=None,
                   help="aggregate every *_brain_trace.json under this directory "
                        "(a season, not one match)")
    p.add_argument("--save-trace", default=None,
                   help="write the decision ledger for this run to this path")
    p.add_argument("--out", default=None,
                   help="directory for the .md/.json report")
    p.add_argument("--thought-limit", type=int, default=40,
                   help="decisions shown in the printed report (-1 = all)")
    p.add_argument("--reasons", action="store_true",
                   help="show the brain's own reason string in the thought log")
    p.add_argument("--null-test", type=int, default=0, metavar="N",
                   help="add the shuffle test (N permutations): keep every decision "
                        "and randomly reassign which situation each choice happened in. "
                        "report `p` = share of shuffles scoring at least as high as the "
                        "real data.  p small = the situation really is doing work.")

    args = p.parse_args(argv)

    engine = None
    report: Dict[str, Any] = {}

    if args.from_trace:
        with open(args.from_trace, encoding="utf-8") as fh:
            doc = json.load(fh)
        report = analyse_match(doc)

    elif args.from_json:
        with open(args.from_json, encoding="utf-8") as fh:
            doc = json.load(fh)
        report = analyse_match(doc if isinstance(doc, dict) else doc)

    elif args.from_dir:
        import glob as _glob
        files = sorted(_glob.glob(
            os.path.join(args.from_dir, "**", "*_brain_trace.json"),
            recursive=True))
        if not files:
            print(f"No *_brain_trace.json under {args.from_dir!r}.")
            return 2
        all_recs: List[Dict[str, Any]] = []
        for path in files:
            dc = json.load(open(path, encoding="utf-8"))
            all_recs.extend(dc.get("decisions") or [])
        print(f"Aggregating {len(files)} match traces ({len(all_recs)} decisions) ...")
        report = analyse_match({
            "match": {"home": f"{len(files)}-match sample", "away": "",
                      "score": ""},
            "decisions": all_recs,
        })

    else:
        # run a fresh match
        import auto_run_match
        from datetime import date
        import random as _rand
        _rand.seed(args.seed)

        loader = auto_run_match.get_loader()
        home_raw = loader.build_matchday_squad(args.home, availability=None)
        away_raw = loader.build_matchday_squad(args.away, availability=None)

        def _build(team, raw, formation):
            from player_dna import SquadBuilder
            return SquadBuilder.build(
                team_name=team,
                starters=raw["starters"],
                substitutes=raw["substitutes"],
                team_superstars=raw.get("superstars"),
                set_piece_takers=raw.get("sp_takers"),
                soul_assignments=auto_run_match.SOUL_PLAYERS,
            )

        home_squad = _build(args.home, home_raw, home_raw["formation"])
        away_squad = _build(args.away, away_raw, away_raw["formation"])

        hp = auto_run_match._resolve_team_profile(
            args.home, home_raw["formation"], True, None, args.home_style)
        ap = auto_run_match._resolve_team_profile(
            args.away, away_raw["formation"], False, None, args.away_style)

        config = auto_run_match.MatchConfig(
            home_team=args.home, away_team=args.away,
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        from match_engine import MatchEngine, MatchConfig, MatchPhase, MatchState
        state = MatchState()
        state.phase = MatchPhase.OPENING
        state.minute = 0
        state.match_clock_s = 0.0
        engine = MatchEngine(config, hp, ap)
        engine.set_squad(args.home, home_squad["starters"], home_squad["substitutes"])
        engine.set_squad(args.away, away_squad["starters"], away_squad["substitutes"])
        result = engine.simulate()

        if args.save_trace:
            write_trace(result, args.save_trace, engine=engine)
            print(f"trace -> {args.save_trace}")

        report = analyse_match(result, engine=engine)

    limit = None if args.thought_limit < 0 else args.thought_limit

    # render + print
    if args.null_test > 0:
        attach_null_test(report, permutations=args.null_test)

    text = render_markdown(report, thought_limit=limit, with_reason=args.reasons)
    print()
    print(text)

    if args.out:
        stem = "brain_report"
        if args.home and args.away:
            stem = f"{args.home}_vs_{args.away}".replace(" ", "_")
        out_dir = args.out if args.out else "plofa_output/brain_reports"
        # derive stem from the loaded match if possible
        m = report.get("match") or {}
        home = (m.get("home") or args.home or "").strip()
        away = (m.get("away") or args.away or "").strip()
        if home and away:
            stem = f"{home}_vs_{away}".replace(" ", "_")
        paths = write_report(report, out_dir, stem=stem,
                             thought_limit=limit, with_reason=args.reasons)
        print()
        print(f"written -> {paths['markdown']}")
        print(f"written -> {paths['json']}")

    return 0


def write_report(report: Dict[str, Any], out_dir: str,
                 *,
                 stem: str = "brain_report",
                 thought_limit: Optional[int] = None,
                 with_reason: bool = True,
                 include_log: bool = True) -> Dict[str, str]:
    """Write .md and .json sidecar files.

    The JSON never carries `thought_log` — the trace sidecar already holds the
    ledger, and duplicating it doubles the season's disk cost for no gain.
    `include_log=False` drops the log from the markdown too, for callers that
    write both files beside each other.
    """
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{stem}.md")
    js_path = os.path.join(out_dir, f"{stem}.json")

    # markdown
    md_text = render_markdown(report,
                              thought_limit=thought_limit if include_log else 0,
                              with_reason=with_reason)
    # strip non-ASCII so Windows cp1252 consoles don't crash
    md_text_enc = md_text.encode("cp1252", errors="xmlcharrefreplace").decode("cp1252")
    with open(md_path, "w", encoding="cp1252") as fh:
        fh.write(md_text_enc)

    # json — omit the full thought_log to keep the file small and avoid
    # duplicating the trace that the sidecar .json already holds.
    doc = {k: v for k, v in report.items() if k != "thought_log"}
    with open(js_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    return {"markdown": md_path, "json": js_path}


# ──────────────────────────────────────────────────────────────────────
# Trace sidecar
# ──────────────────────────────────────────────────────────────────────
def write_trace(result_or_timeline: Any, path: str,
                *, engine: Any = None,
                home_team: Optional[str] = None,
                away_team: Optional[str] = None) -> str:
    """Write the decision ledger for one match to `path` as JSON.

    This is the compact sidecar `--from-trace` and `--from-dir` read.  It
    holds one record per on-ball decision — not the full event stream — so a
    whole season of traces stays small enough to keep.
    """
    from brain_integration import collect_decisions

    cfg = getattr(result_or_timeline, "config", None)
    home = str(home_team or getattr(cfg, "home_team", "") or "")
    away = str(away_team or getattr(cfg, "away_team", "") or "")
    score = ""
    try:
        score = f"{result_or_timeline.home_goals}-{result_or_timeline.away_goals}"
    except Exception:
        score = ""

    records = collect_decisions(result_or_timeline, engine=engine,
                                home_team=home, away_team=away, compact=True)
    doc = {
        "match": {"home": home, "away": away, "score": score},
        "decisions": records,
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    return path


# ──────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raise SystemExit(_main())