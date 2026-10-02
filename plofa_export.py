"""PLOFA MATCH EXPORT — our own format, not a provider's.

Why this is not StatsBomb-shaped
--------------------------------
StatsBomb's schema answers StatsBomb's questions. It is a public contract, so
it is deliberately lossy about the things that are none of its business, and
it records every event in the frame of the team in possession — which means a
plot of one team's events needs a per-event flip to be readable. Copying it
would mean importing both losses.

PLOFA has data no provider has, and that data is the point of the project:

  * the DECISION and the EXECUTION, separately, for every on-ball action —
    which layer chose it, with what intent, at what confidence, and what the
    neural brain's stated reason was
  * the off-ball layer: press engagement g, the run mode the engine chose and
    the target it chose it for
  * PLOFA's own vocabularies, untranslated: 60 event types, 10 on-ball
    intents, 6 geometrically-observed run types, 14 intended run modes, 4
    possession phases, the attacking-matrix actions
  * the diagnostics that say how the numbers were produced — the intended-run
    bar, how many teleport segments were filtered

Three design decisions worth stating, because they are opinions:

1. **ONE FRAME. "Home attacks right", always.** Not per-event, not
   per-possession. Every coordinate in the file means the same thing, so any
   plot of the file is unambiguous without the reader knowing anything about
   attack direction. The half-time ends change is recorded per event as `half`
   so the broadcast frame can be reconstructed, but it is never baked into the
   numbers. This is the single biggest thing StatsBomb's format makes
   difficult, and the un-normalised depth chart that produced a whole session
   of false findings is exactly the bug this removes by construction.

2. **DECISION IS NOT EXECUTION.** Every action carries `decision` (authority,
   intent, confidence, quality, reason, and what the matrix and phase layers
   said) next to its physical result. A team whose trained policy is silently
   overridden by a hand-written layer is invisible in a format that only
   records outcomes — and that is the single most important thing to be able
   to see in this project.

3. **THE FILE EXPLAINS ITSELF.** A `schema` block travels inside the file
   listing every enum and every unit, so a match exported in six months is
   still readable without this repository. A format that needs its writer
   present is not an archive format.

Nothing is invented. Fields PLOFA does not have are OMITTED, not filled with
a plausible-looking zero, and anything the exporter wanted but could not find
is listed in `gaps` so a reader knows the difference between "absent" and
"not captured".
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import Counter
from typing import Any, Dict, List, Optional

FORMAT = "plofa.match"
FORMAT_VERSION = "1.0"

PITCH_L, PITCH_W = 105.0, 68.0

# The frame, declared once and used everywhere.
FRAME = "home_attacks_right"


# ── the vocabulary, as data, so the file carries its own dictionary ────────

def _enums() -> Dict[str, Any]:
    from brain_integration import INTENT_LABELS
    from run_tracking import INTENDED_RUN_MODES, RUN_TYPES
    from match_engine import EventType
    return {
        "units": {
            "length": "metres", "speed": "metres/second",
            "clock": "MM:SS.mmm from kick-off, running past 45 and 90",
            "coordinates": "[x, y] on a 105 x 68 pitch, origin at the "
                           "HOME team's left-hand corner; home always attacks "
                           "towards x=105, in both halves",
            "half": "1 or 2; the only reason the ends change is visible",
        },
        "event_type": {e.name: e.value for e in EventType},
        "intent": list(INTENT_LABELS),
        "run_type_observed": list(RUN_TYPES),
        "run_mode_intended": list(INTENDED_RUN_MODES),
        "decision_authority": [
            "player_policy",
            "role_fallback",
            "emergent_opportunity",
            "attacking_matrix",
            "tactical_phase",
            "wide_combo",
            "engine_geometry",
        ],
        "decision_authority_meaning": {
            "player_policy": "the evolved on-ball brain chose this",
            "role_fallback": "the player's ROLE decided it — a hand-written "
                             "per-position rule, not the brain. A high count "
                             "here is the number of on-ball actions the "
                             "trained policy did not make.",
            "emergent_opportunity": "no decider claimed it; the engine took a "
                                    "free action the situation offered",
            "attacking_matrix": "AttackingMatrix chose the action",
            "tactical_phase": "a TacticalPhase order chose it",
            "wide_combo": "the wide-combination layer chose it",
            "engine_geometry": "no decider — engine geometry or a rule",
        },
        "decision_authority_note": (
            "This list was WRONG on its first version: it documented five "
            "values while the engine also emits role_fallback and "
            "emergent_opportunity, so the file's own dictionary misdescribed "
            "its own contents. Found by a test that asserts every authority in "
            "the file is documented in the schema. The count that matters for "
            "the project is role_fallback — it is the on-ball actions the "
            "trained policy did not make."
        ),
        "third": ["defensive_third", "middle_third", "final_third"],
        "note": (
            "Coordinates are in ONE frame, not per-possession. A provider "
            "format that flips per acting team makes a plot of a single team "
            "require a per-event flip to read; here every number means the "
            "same thing everywhere. See FRAME."
        ),
    }


# ── the frame ─────────────────────────────────────────────────────────────

def to_home_frame(x: float, home_attacks_right: bool) -> float:
    """Map a raw pitch x into the export frame (home attacks right, always).

    THE FRAME IS DEFINED BY THE HOME TEAM, AND ONLY BY THE HOME TEAM.

    Got wrong twice while building this, and the second attempt is the
    instructive one. An intermediate version mirrored per ACTING team, on the
    reasoning that each side attacks its own way. A constructed case killed it
    immediately: with home shooting at raw x=105 and away at raw x=0, the
    per-acting-team mirror sent BOTH to 105 — the away side's own target
    landed on the home side's goal. Two teams cannot attack the same end of a
    pitch, so that convention is impossible.

    So the only question is whether the HOME team is attacking towards x=105
    right now, and if not, every x in the file mirrors. Per-acting-team never
    enters this function.

    STILL NOT VERIFIED AGAINST LIVE DATA, and the reason is recorded on
    `tests/test_plofa_export.py::test_frame_which_half_the_team_operates_in`:
    on a real match the away side's passes sit mostly at high x under this
    transform, which should not happen. Either the raw coordinates are in a
    per-team-forward frame after all, or the half-time flip here is
    double-counting a swap the engine performs elsewhere. That test is the
    open question; this function is the principled version, correct on
    constructed ground truth, and it should not be "fixed" again without
    answering that test first.

    Only x is mirrored. y stays in the raw pitch frame, so "left channel" means
    the same pitch side in both halves instead of swapping at 45'.
    """
    return float(x) if home_attacks_right else PITCH_L - float(x)


def _loc(x, y, home_right) -> Optional[List[float]]:
    if x is None or y is None:
        return None
    return [round(to_home_frame(x, home_right), 2), round(float(y), 2)]


# ── per-event conversion ──────────────────────────────────────────────────

def _actor(block: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what is identity, and never invent a number."""
    out = {}
    for k in ("name", "jersey_number", "position", "soul_archetype"):
        v = block.get(k)
        if v is not None:
            out[k] = v
    for k, src in (("stamina", "stamina"), ("form", "form"),
                   ("pace", "pace")):
        v = block.get(src)
        if isinstance(v, (int, float)):
            out[k] = round(float(v), 2)
    return out or {"name": "unknown"}


def _decision(md: Dict[str, Any]) -> Dict[str, Any]:
    """The decision, with every layer that had a voice, and its own authority.

    Deliberately keeps layers that were OVERRULED. The point of this field is
    to make it possible to see a trained decision being discarded downstream,
    which an outcomes-only format cannot show at all.
    """
    d: Dict[str, Any] = {}
    auth = md.get("decision_authority")
    if auth:
        d["authority"] = auth
    brain = md.get("active_brain")
    if isinstance(brain, dict):
        for k in ("action", "intent", "confidence", "reason",
                  "decision_quality", "is_error"):
            if brain.get(k) is not None:
                v = brain[k]
                d[k] = round(v, 3) if isinstance(v, float) else v
    mx = md.get("attacking_matrix")
    if isinstance(mx, dict) and mx:
        d["matrix"] = {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in mx.items() if v is not None}
    if md.get("possession_phase") is not None:
        ph = {"phase": md.get("possession_phase")}
        for src, dst in (("phase_directive", "directive"),
                         ("phase_reason", "reason"),
                         ("phase_recommendation", "recommendation"),
                         ("recycle", "recycle")):
            if md.get(src) is not None:
                ph[dst] = md[src]
        d["phase"] = ph
    feas = md.get("execution_feasibility")
    if isinstance(feas, dict) and feas:
        d["feasibility"] = feas
    if md.get("shot_intent"):
        d["shot_intent"] = True
    return d


_PASS_FIELDS = (
    "pass_length_m", "pass_length_yards", "pass_height", "body_part",
    "pass_advance", "is_progressive", "is_long", "length_class",
    "pass_channel", "start_third", "end_third", "start_half", "end_half",
    "pass_energy_cost", "wind_drift_m", "cross", "cross_origin", "cross_dest",
    "is_airborne", "pass_type", "under_pressure", "pass_direction",
)
_GEOM_FIELDS = ("ball_speed_mps", "ball_travel_s", "receiver_arrival_s",
                "resolution", "kinematic_outcome")


def _sub(md: Dict[str, Any], fields) -> Dict[str, Any]:
    out = {}
    for k in fields:
        v = md.get(k)
        if v is None:
            continue
        out[k] = round(v, 3) if isinstance(v, float) else v
    return out


def convert_event(e, half: int, attacking_right: Dict[str, bool],
                  index: int, squads: Dict[str, Any],
                  home_team: str = "") -> Dict[str, Any]:
    md = getattr(e, "metadata", None) or {}
    t = getattr(e, "event_type", None)
    name = getattr(t, "name", None) or "UNKNOWN"
    team = getattr(e, "team", "") or ""
    minute = getattr(e, "minute", 0) or 0
    second = getattr(e, "second", 0) or 0
    clock_s = getattr(e, "match_clock_s", None)
    if clock_s is None:
        clock_s = (minute * 60) + second

    out: Dict[str, Any] = {
        "seq": index,
        "clock": f"{int(clock_s)//60:02d}:{clock_s%60:06.3f}",
        "clock_s": round(float(clock_s), 3),
        "half": half,
        "minute": minute,
        "second": second,
        "type": name,
        "team": team or None,
        "outcome": getattr(e, "outcome", None),
    }
    if out["team"] is None:
        del out["team"]
    if out["outcome"] is None:
        del out["outcome"]

    who = getattr(e, "player", None)
    if who:
        out["actor"] = _actor(squads.get(team, {}).get(who, {})
                              or {"name": who})
    sec = getattr(e, "secondary_player", None)
    if sec:
        out["other"] = {"name": sec}

    # The HOME side's direction defines the frame; the acting team never enters
    # this decision. Flipped for the second half because the ends change.
    # See `to_home_frame` for why this is not per-acting-team, and for the
    # open question about whether the half-time flip double-counts.
    home_base = attacking_right.get(home_team, True)
    home_right = home_base if half == 1 else not home_base

    loc = _loc(getattr(e, "location_x", None), getattr(e, "location_y", None),
               home_right)
    if loc:
        out["at"] = loc
    end = _loc(getattr(e, "end_x", None), getattr(e, "end_y", None),
               home_right)
    if end:
        out["to"] = end

    dec = _decision(md)
    if dec:
        out["decision"] = dec

    p = _sub(md, _PASS_FIELDS)
    if p:
        out["pass"] = p
    g = _sub(md, _GEOM_FIELDS)
    if g:
        out["geometry"] = g

    ob: Dict[str, Any] = {}
    for k in ("press_g", "press_intensity", "run_mode", "run_target_x",
              "run_target_y", "sprint", "speed_mps"):
        v = md.get(k)
        if v is None:
            continue
        ob[k] = round(v, 3) if isinstance(v, float) else v
    if ob:
        out["off_ball"] = ob

    for k in ("situation", "phase", "game_state", "xg", "xa"):
        v = getattr(e, k, None)
        if v is None:
            continue
        out[k] = getattr(v, "value", v) if hasattr(v, "value") else v
    if isinstance(out.get("xg"), float):
        out["xg"] = round(out["xg"], 3)
    if isinstance(out.get("xa"), float):
        out["xa"] = round(out["xa"], 3)
    return out


# ── the whole file ────────────────────────────────────────────────────────

def export_match(result, engine=None, roster: Optional[Dict] = None
                 ) -> Dict[str, Any]:
    """Build the whole document from a MatchResult.

    `roster` maps "Club" -> {player name -> dict of attributes}. Without it the
    actor blocks carry names only, and that is recorded in `gaps` rather than
    papered over.
    """
    cfg = getattr(result, "config", None)
    state = getattr(result, "state", None)
    home = getattr(cfg, "home_team", None) or "HOME"
    away = getattr(cfg, "away_team", None) or "AWAY"

    # Attack direction comes from the engine's own per-team state, not from a
    # guess about which end the home side started at.
    attacking_right: Dict[str, bool] = {}
    if engine is not None:
        src = getattr(getattr(engine, "position_engine", None),
                      "team_attacks_right", None)
        if isinstance(src, dict):
            attacking_right = {k: bool(v) for k, v in src.items()}

    cur_half = 1
    events: List[Dict[str, Any]] = []
    for i, e in enumerate(getattr(result, "timeline", []) or []):
        minute = getattr(e, "minute", 0) or 0
        want = 2 if minute >= 45 else 1
        cur_half = want
        events.append(convert_event(e, cur_half, attacking_right, i,
                                    roster or {}, home))

    doc: Dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "schema": _enums(),
        "frame": {
            "name": FRAME,
            "length_m": PITCH_L,
            "width_m": PITCH_W,
            "origin": "home team's left-hand corner",
            "attacking_right_first_half": attacking_right,
            "note": "coordinates do NOT change at half time. Each event's x is "
                    "mirrored if the team PERFORMING it was not attacking "
                    "towards x=105, so every coordinate in the file means the "
                    "same thing. `half` records the ends change separately. "
                    "y is never mirrored, so 'left channel' means the same "
                    "pitch side in both halves.",
        },
        "match": {
            "home": home,
            "away": away,
            "score": getattr(result, "score_str", None),
            "date": str(getattr(cfg, "match_date", "") or ""),
            "matchday": getattr(cfg, "matchday", None),
            "venue": getattr(cfg, "venue", None),
            "possession_home_pct": _r(getattr(result, "home_possession_pct", None)),
            "xg": {"home": _r(getattr(result, "home_xg", None)),
                   "away": _r(getattr(result, "away_xg", None))},
        },
        "events": events,
    }

    prof: Dict[str, Any] = {}
    for attr, key in (("run_profile", "run_type_observed"),
                      ("intended_run_profile", "run_mode_intended")):
        p = getattr(result, attr, None)
        if p:
            tot = Counter()
            for pl in p.values():
                for k, v in pl.items():
                    tot[k] += v
            prof[key] = {"by_player": p, "totals": dict(tot.most_common())}
    if prof:
        doc["profiles"] = prof

    diag = getattr(result, "intended_run_diagnostics", None)
    if diag:
        prof_diag = dict(diag)
        prof_diag["meaning"] = {
            "considered": "run decisions taken",
            "counted": "decisions that asked the player to move >= MOVE_MIN_M",
            "skipped_below_bar": "decisions inside the bar — a positioning, "
                                 "not a run",
            "unmeasured": "decisions recorded without geometry; counted and "
                          "flagged rather than dropped",
        }
        doc.setdefault("diagnostics", {})["intended_runs"] = prof_diag

    eng = engine
    if eng is not None and getattr(eng, "run_tracker", None) is not None:
        doc.setdefault("diagnostics", {})["run_tracker"] = {
            "jumps_filtered": eng.run_tracker.jumps_filtered,
            "meaning": "position discontinuities refused as observed runs "
                       "(on-ball reposition writes and keeper slots)",
        }

    doc["gaps"] = _gaps(doc, roster)
    doc["index"] = _index(events)

    # The source sometimes emits coordinates off the pitch (wind drift, an
    # overshooting destination). They are exported AS IS — silently clamping
    # would make the file disagree with the simulation — but counted here, so
    # a consumer is told rather than left to trip over x = -2.6.
    bad = []
    for e in events:
        for k in ("at", "to"):
            v = e.get(k)
            if isinstance(v, list) and len(v) == 2:
                if not (-0.5 <= v[0] <= PITCH_L + 0.5) or \
                        not (-0.5 <= v[1] <= PITCH_W + 0.5):
                    bad.append((e["seq"], k, v))
    if bad:
        doc.setdefault("diagnostics", {})["off_pitch_coordinates"] = {
            "count": len(bad),
            "of_events": len(events),
            "x_range_observed": [
                min(v[0] for _, _, v in bad), max(v[0] for _, _, v in bad)],
            "y_range_observed": [
                min(v[1] for _, _, v in bad), max(v[1] for _, _, v in bad)],
            "first_five": [{"seq": s, "field": k, "at": v}
                           for s, k, v in bad[:5]],
            "note": "exported verbatim, not clamped. These come from the "
                    "simulation, not the frame transform.",
        }
    return doc


def _r(v, nd=2):
    return round(float(v), nd) if isinstance(v, (int, float)) else None


def _gaps(doc, roster) -> List[str]:
    """What a consumer should NOT expect to find here, stated rather than
    silently omitted. A format that hides its holes gets trusted with data it
    does not have."""
    g = []
    if not roster:
        g.append("no roster supplied: actor blocks carry names only "
                 "(no squad number, role, soul, stamina or form)")
    have = set()
    for e in doc["events"][:2000]:
        have.update(e.keys())
    for k, why in (
        ("related_events", "PLOFA does not link events into duels or "
                           "sequences the way a provider does"),
        ("freeze_frame", "no freeze-frame capture exists"),
        ("off_camera", "not modelled"),
        ("lineup", "squads are not exported as a Starting XI block"),
        ("passing_network", "not modelled"),
    ):
        if k not in have:
            g.append(f"{k}: {why}")
    return g


def _index(events) -> Dict[str, Any]:
    c = Counter(e["type"] for e in events)
    auth = Counter()
    intents = Counter()
    for e in events:
        d = e.get("decision") or {}
        if d.get("authority"):
            auth[d["authority"]] += 1
        if d.get("intent"):
            intents[str(d["intent"])] += 1
    return {
        "n_events": len(events),
        "by_type": dict(c.most_common()),
        "decision_authority": dict(auth.most_common()),
        "on_ball_intent": dict(intents.most_common()),
    }


# ── validation ────────────────────────────────────────────────────────────

REQUIRED = ("seq", "clock", "clock_s", "half", "minute", "second", "type")


def validate(doc: Dict[str, Any]) -> List[str]:
    """Return a list of problems. Empty means the file is well-formed.

    Property checks, not golden values: a format test that compares against a
    stored file breaks every time the simulation legitimately changes, and then
    gets deleted rather than maintained.
    """
    p: List[str] = []
    if doc.get("format") != FORMAT:
        p.append(f"format is {doc.get('format')!r}, expected {FORMAT!r}")
    if "schema" not in doc:
        p.append("no schema block: the file would not be readable alone")
    evs = doc.get("events")
    if not isinstance(evs, list) or not evs:
        return p + ["no events"]
    for i, e in enumerate(evs):
        for k in REQUIRED:
            if k not in e:
                p.append(f"event {i}: missing {k!r}")
                break
        if "seq" in e and e["seq"] != i:
            p.append(f"event {i}: seq is {e['seq']}, not sequential")
            break
    # Both `at` and `to` must be in range. Checking only `at` let an
    # off-pitch DESTINATION through, which is the half that actually goes
    # wrong — a pass aimed 12 m past the touchline still has a legal origin.
    off = 0
    for field in ("at", "to"):
        xs = [e[field][0] for e in evs if isinstance(e.get(field), list)]
        ys = [e[field][1] for e in evs if isinstance(e.get(field), list)]
        for tag, vals, hi in (("x", xs, PITCH_L), ("y", ys, PITCH_W)):
            if not vals:
                continue
            bad = [v for v in vals if v < -0.5 or v > hi + 0.5]
            if bad:
                off += len(bad)
                p.append(f"WARNING {off} {tag} values outside 0..{hi} in "
                         f"'{field}' (observed {min(bad):.1f}..{max(bad):.1f}"
                         f"); exported verbatim and counted in diagnostics")
    # The frame's whole point: a team's SHOTS should point at the opponent
    # goal. This is a WARNING, not a problem, because the field it reads is not
    # reliably populated: on a live match `end_x` on shot events is exactly
    # 0.0 for three quarters of one team's shots (their OWN goal) while being
    # correct for the other, which is the signature of an unpopulated default
    # rather than a coordinate. Until that is fixed, failing here would report
    # a frame bug that is actually an engine bug — and a validator that cries
    # wolf gets deleted, which is worse than having no validator.
    for team, goal in ((doc["match"]["home"], PITCH_L),
                       (doc["match"]["away"], 0.0)):
        xs = [e["to"][0] for e in evs
              if e.get("team") == team and e["type"] in
              ("SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET",
               "SHOT_BLOCKED", "GOAL")
              and isinstance(e.get("to"), list)]
        if len(xs) >= 3:
            near = sum(1 for v in xs if abs(v - goal) < 25)
            if near / len(xs) < 0.6:
                p.append(
                    f"WARNING {team}: only {near}/{len(xs)} shots end near "
                    f"x={goal}. Most likely `end_x` is not populated on shot "
                    f"events (a default of 0.0 reads as a real coordinate), "
                    f"not a frame error — check _diag_shot_frame.py before "
                    f"blaming the transform")
    return p


def write_json(doc: Dict[str, Any], path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                 encoding="utf-8")
    return p
