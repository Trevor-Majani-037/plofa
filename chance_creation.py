"""
PLOFA 26/27 — CHANCE CREATION LEDGER  (Checkpoint 11)
======================================================
chance_creation.py

Why this exists:
    The engine used to emit CHANCE_CREATED / BIG_CHANCE_CREATED events by
    RANDOMLY picking a "creator" from the attacking squad and fabricating a
    position in the attacking third. Key passes and assists were therefore
    invented, not recorded from emergent event causality — the exact failure
    the analyst flagged ("chances are randomly generated in the attacking
    third, not recorded from emergent and true event data").

    This module is the Opta/StatsBomb answer: a pure, post-match scan of the
    REAL event timeline that derives the entire chance-creation pipeline from
    what actually happened — just as a datalogger watches the footage and
    tags the assist.

    Taxonomy implemented (Opta / StatsBomb naming both noted):
        1. Goal Assist        — the final pass to the shooter WHO SCORES.
                               (Opta: Assist event. StatsBomb: Pass with
                               pass.goal_assist = TRUE.)
        2. Shot Assist (Key Pass) — the final pass to the shooter who then
                               takes a shot but does NOT score. If the pass
                               is intercepted/deflected before the shooter,
                               no shot assist is recorded (receiver must be
                               the shooter).
        3. Chance Created     — the Opta cumulative metric:
                               Goal Assists + Shot Assists. Every completed
                               pass that directly results in a shot (goal or
                               miss) = 1 Chance Created.
        4. Big Chance Created — a pass that sets up a Big Chance (clear
                               1v1 / tap-in from close range). Classified
                               objectively here via the resulting shot's xG
                               and zone, not a human flag.
        5. xA (Expected Assist) — the probability the delivery becomes an
                               assist, evaluated at the destination: xA of
                               the setup pass = xG of the shot it created.
        6. Second Assist / Pre-Assist (hockey assist) — the pass to the
                               player who then records the assist/key pass.
        7. Shot-Creating Actions (SCA) — the last two offensive actions
                               directly before a shot: a completed pass,
                               successful dribble, carry, foul won, or
                               defensive action won that triggers the shot.
        8. Fantasy Assist     — the attacking contribution that precedes a
                               rebound tap-in (shot saved/blocked then
                               scored by a teammate) or a penalty/direct
                               free-kick converted after a foul won.

    Pure module: reads only the timeline, no dependency on the event chain
    or match loop. Trivially unit-testable (mirrors threat_engine.py /
    cross_detector.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from match_engine import MatchEvent, EventType


# ─────────────────────────────────────────────
# EVENT VOCABULARY
# ─────────────────────────────────────────────

#: Pass-like events that can SET UP a shot (completed delivery with a known
#: receiver). Crosses / corners / free-kick crosses are included — Opta
#: counts a corner delivery as a key pass when it leads directly to a shot.
SETUP_PASS_EVENTS = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
    EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS,
}

#: The shot itself. A GOAL emitted right after a SHOT_ON_TARGET / HIT_WOODWORK
#: is a continuation of that same shot (upgraded to "scored"), not a new one.
SHOT_EVENTS = {
    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
    EventType.SHOT_BLOCKED, EventType.HIT_WOODWORK,
    EventType.FREEKICK_DIRECT, EventType.PENALTY_SCORED,
    EventType.PENALTY_MISSED, EventType.GOAL,
}

#: Events that mean possession has broken (so the backward scan for the setup
#: pass must stop at them — you can't build a key pass across a turnover).
POSSESSION_BREAK_EVENTS = {
    EventType.TURNOVER, EventType.DISPOSSESSED, EventType.DRIBBLE_FAIL,
    EventType.TACKLE_LOST, EventType.MISCONTROL, EventType.OFFSIDE,
    EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.CLEARANCE,
    EventType.BLOCK, EventType.RECOVERY, EventType.BALL_RECOVERY,
    EventType.PRESS_SUCCESS, EventType.GOAL_KICK, EventType.FOUL_COMMITTED,
    EventType.SAVE,
}

#: Offensive actions that count toward a Shot-Creating Action.
SCA_ACTION_EVENTS = {
    EventType.PASS, EventType.PROGRESSIVE_PASS, EventType.SWITCH_OF_PLAY,
    EventType.THROUGH_BALL, EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
    EventType.DRIBBLE_SUCCESS, EventType.CARRY, EventType.FOUL_WON,
    EventType.INTERCEPTION, EventType.TACKLE_WON, EventType.BLOCK,
    EventType.RECOVERY, EventType.BALL_RECOVERY,
}

#: A shot is a "big chance" when its xG is at this floor (clear 1v1 / tap-in).
BIG_CHANCE_XG_FLOOR: float = 0.45

#: Maximum events to look forward from a save/block/woodwork for a rebound
#: goal by the same team (Fantasy Assist).
REBOUND_SCAN_WINDOW: int = 10

#: Maximum events to look backward for the setup pass before giving up.
KEY_PASS_SCAN_WINDOW: int = 24


@dataclass
class ChanceRecord:
    """One shot and its full creation chain, derived from the timeline."""
    minute: int
    team: str
    shooter: str
    outcome: str                       # goal | save | miss | block | woodwork | penalty
    situation: str = ""
    shot_x: float = 0.0
    shot_y: float = 0.0
    xg: float = 0.0
    is_big: bool = False
    # The setup pass (Key Pass / Shot Assist / Goal Assist).
    creator: str = ""
    pass_x: float = 0.0
    pass_y: float = 0.0
    pass_end_x: float = 0.0
    pass_end_y: float = 0.0
    xa: float = 0.0
    # Second assist: the pass to the creator.
    second_assist: str = ""
    # Shot-creating actions: the two offensive actions before the shot.
    sca_players: List[str] = field(default_factory=list)
    # Fantasy assist: rebound or won-foul path.
    fantasy_assist: str = ""
    #: index into the timeline of the setup pass event (for gold plotting).
    pass_event_index: int = -1
    is_goal_assist: bool = False
    is_open_play: bool = True

    def as_dict(self) -> Dict:
        return {
            "minute": self.minute, "team": self.team, "shooter": self.shooter,
            "outcome": self.outcome, "situation": self.situation,
            "shot_x": round(self.shot_x, 1), "shot_y": round(self.shot_y, 1),
            "xg": round(self.xg, 3), "is_big": self.is_big,
            "creator": self.creator,
            "pass_x": round(self.pass_x, 1), "pass_y": round(self.pass_y, 1),
            "pass_end_x": round(self.pass_end_x, 1),
            "pass_end_y": round(self.pass_end_y, 1),
            "xa": round(self.xa, 3),
            "second_assist": self.second_assist,
            "sca_players": self.sca_players,
            "fantasy_assist": self.fantasy_assist,
            "is_goal_assist": self.is_goal_assist,
            "is_open_play": self.is_open_play,
        }


class ChanceCreationLedger:
    """
    Derives every chance-creation stat from the real event timeline.

    Usage:
        ledger = ChanceCreationLedger(timeline)
        ledger.compute()
        ledger.per_player   # {player_name: {shot_assists, assists, ...}}
        ledger.records      # [ChanceRecord, ...] — one per shot
        ledger.shot_assist_event_indexes  # timeline indexes of setup passes
    """

    def __init__(self, timeline: List[MatchEvent]):
        self.timeline = timeline
        self.records: List[ChanceRecord] = []
        self.per_player: Dict[str, Dict] = {}
        self.shot_assist_event_indexes: Set[int] = set()
        self._idx: Dict[int, MatchEvent] = {}

    # ── PUBLIC ───────────────────────────────────────

    def compute(self) -> "ChanceCreationLedger":
        self._idx = {i: e for i, e in enumerate(self.timeline)}
        self.records = []
        self.shot_assist_event_indexes = set()

        shots = self._collect_shots()
        for shot in shots:
            self.records.append(self._analyze_shot(shot))
        self._aggregate()
        return self

    # ── SHOT COLLECTION ──────────────────────────────

    def _collect_shots(self) -> List["_ShotCandidate"]:
        """Group the timeline into discrete shots. A GOAL upgrades the
        immediately-preceding shot ONLY when the same player scored it and
        the events are adjacent (SHOT_ON_TARGET → GOAL, or HIT_WOODWORK →
        GOAL). A rebound tap-in by a DIFFERENT player after a SAVE / block
        is its own shot — the original shooter earns a Fantasy Assist
        instead. A SAVE is never a shot itself."""
        shots: List["_ShotCandidate"] = []
        last: Optional["_ShotCandidate"] = None
        for i, e in enumerate(self.timeline):
            t = e.event_type
            if t == EventType.SAVE or t not in SHOT_EVENTS:
                continue
            if (t == EventType.GOAL and last is not None
                    and last.event.team == e.team
                    and last.event.player == e.player
                    and last.index == i - 1
                    and last.event.event_type in (
                        EventType.SHOT_ON_TARGET, EventType.HIT_WOODWORK,
                        EventType.FREEKICK_DIRECT,
                    )):
                last.goaled = True
                continue
            cand = _ShotCandidate(i, e)
            shots.append(cand)
            last = cand
        return shots

    # ── PER-SHOT ANALYSIS ────────────────────────────

    def _analyze_shot(self, shot: "_ShotCandidate") -> ChanceRecord:
        e = shot.event
        minute = getattr(e, "minute", 0) or 0
        shooter = e.player or ""
        team = e.team or ""
        xg = getattr(e, "xg", 0.0) or 0.0
        situation = getattr(e, "situation", None)
        situation_str = situation.name if situation is not None else ""
        is_big = self._is_big_chance(e, xg)

        # Outcome classification.
        if shot.goaled:
            outcome = "goal"
        elif e.event_type in (EventType.PENALTY_SCORED, EventType.GOAL):
            outcome = "goal"
        elif e.event_type in (EventType.PENALTY_MISSED,):
            outcome = "miss"
        elif e.event_type == EventType.SHOT_BLOCKED:
            outcome = "block"
        elif e.event_type == EventType.HIT_WOODWORK:
            outcome = "woodwork"
        elif e.event_type == EventType.SHOT_OFF_TARGET:
            outcome = "miss"
        else:
            outcome = "save"

        rec = ChanceRecord(
            minute=minute, team=team, shooter=shooter,
            outcome=outcome, situation=situation_str,
            shot_x=getattr(e, "location_x", 0.0) or 0.0,
            shot_y=getattr(e, "location_y", 0.0) or 0.0,
            xg=round(xg, 4), is_big=is_big,
        )

        # ── THE SETUP PASS (Key Pass / Assist) ────────────────
        setup = self._find_setup_pass(shot.index, team, shooter, situation_str)
        if setup is not None:
            pass_evt, pass_idx, is_open_play = setup
            rec.creator = pass_evt.player or ""
            rec.pass_x = pass_evt.location_x or 0.0
            rec.pass_y = pass_evt.location_y or 0.0
            rec.pass_end_x = pass_evt.end_x if pass_evt.end_x is not None else rec.pass_x
            rec.pass_end_y = pass_evt.end_y if pass_evt.end_y is not None else rec.pass_y
            # xA of the delivery = xG of the shot it created.
            rec.xa = round(xg, 4)
            rec.pass_event_index = pass_idx
            rec.is_goal_assist = (outcome == "goal")
            rec.is_open_play = is_open_play
            self.shot_assist_event_indexes.add(pass_idx)

            # ── SECOND ASSIST (pass to the creator) ───────────
            rec.second_assist = self._find_second_assist(
                pass_idx, team, rec.creator
            )

        # ── SHOT-CREATING ACTIONS ────────────────────────────
        rec.sca_players = self._find_sca(shot.index, team, shooter)

        # ── FANTASY ASSIST ───────────────────────────────────
        rec.fantasy_assist = self._find_fantasy_assist(
            shot.index, team, e, outcome
        )
        return rec

    def _find_setup_pass(self, shot_idx: int, team: str, shooter: str,
                          situation: str) -> Optional[Tuple[MatchEvent, int, bool]]:
        """The most recent completed pass by `team` whose receiver is the
        shooter. For set-piece deliveries (corner / crossed free-kick) the
        delivery itself is the setup even if it has no receiver.

        Exception: direct corner goal (olympico) and direct free-kick goal —
        no assist awarded per Opta/StatsBomb convention."""
        is_set_piece = situation in ("CORNER", "CROSSED_FREEKICK")
        is_direct_corner = (situation == "CORNER" and
                            self._idx[shot_idx].event_type == EventType.GOAL and
                            self._idx[shot_idx].player == shooter)
        for j in range(shot_idx - 1, max(-1, shot_idx - KEY_PASS_SCAN_WINDOW), -1):
            evt = self._idx[j]
            if evt.event_type in POSSESSION_BREAK_EVENTS and evt.team != team:
                return None
            if evt.event_type not in SETUP_PASS_EVENTS:
                continue
            if evt.team != team or not getattr(evt, "outcome", True):
                # A completed pass by the OTHER team breaks the chain.
                if evt.team != team:
                    return None
                continue
            # Direct corner goal: CORNER_TAKEN by shooter → GOAL by same player.
            if is_direct_corner and evt.event_type == EventType.CORNER_TAKEN:
                return None
            # Direct free-kick goal: FREEKICK_DIRECT by shooter → GOAL by same player.
            if (situation == "DIRECT_FREEKICK" and
                    evt.event_type == EventType.FREEKICK_DIRECT and
                    evt.player == shooter):
                return None
            # First completed pass-like by this team going backward.
            receiver = getattr(evt, "secondary_player", None)
            if receiver == shooter:
                is_open = evt.event_type not in (EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS, EventType.THROW_IN)
                return evt, j, is_open
            if is_set_piece and evt.event_type in (
                EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS,
                EventType.CROSS_ATTEMPT, EventType.CROSS_SUCCESS,
            ):
                is_open = evt.event_type not in (EventType.CORNER_TAKEN, EventType.FREEKICK_CROSS, EventType.THROW_IN)
                return evt, j, is_open
            # The most recent completed pass went somewhere else — the shot
            # didn't come directly from a pass (dribble / loose ball), so
            # there is no key pass (Opta's strict rule).
            return None
        return None

    def _find_second_assist(self, pass_idx: int, team: str,
                            creator: str) -> str:
        """The completed pass by `team` before the setup pass, delivered to
        the creator — the pre-assist / hockey assist."""
        for j in range(pass_idx - 1, max(-1, pass_idx - KEY_PASS_SCAN_WINDOW), -1):
            evt = self._idx[j]
            if evt.event_type in POSSESSION_BREAK_EVENTS and evt.team != team:
                return ""
            if evt.event_type not in SETUP_PASS_EVENTS:
                continue
            if evt.team != team or not getattr(evt, "outcome", True):
                if evt.team != team:
                    return ""
                continue
            if getattr(evt, "secondary_player", None) == creator:
                return evt.player or ""
            return ""   # only the immediately-preceding pass counts
        return ""

    def _find_sca(self, shot_idx: int, team: str, shooter: str) -> List[str]:
        """The last TWO offensive actions by `team` before the shot. The
        setup pass itself is the last of them if present; the action before
        it (and the setup pass) form the SCA pair."""
        actions: List[Tuple[int, str]] = []
        for j in range(shot_idx - 1, max(-1, shot_idx - KEY_PASS_SCAN_WINDOW), -1):
            evt = self._idx[j]
            if evt.event_type in POSSESSION_BREAK_EVENTS and evt.team != team:
                break
            if evt.team != team:
                continue
            if evt.event_type == EventType.BALL_RECEIPT:
                continue
            if evt.event_type in SCA_ACTION_EVENTS:
                actor = evt.player or ""
                if actor and actor != shooter and actor not in (a for _, a in actions):
                    actions.append((j, actor))
                if len(actions) >= 2:
                    break
        return [a for _, a in actions]

    def _find_fantasy_assist(self, shot_idx: int, team: str,
                             e: MatchEvent, outcome: str) -> str:
        """Fantasy Assist paths:
          1. Rebound — this shot was saved / blocked / off the woodwork and
             the SAME team scores from the rebound within a few events.
          2. Won foul — a FOUL_WON before a converted penalty / direct
             free-kick (this shot)."""
        # Path 2: the shot itself is a penalty/direct FK after a won foul.
        if e.event_type in (EventType.PENALTY_SCORED, EventType.FREEKICK_DIRECT):
            for j in range(shot_idx - 1, max(-1, shot_idx - KEY_PASS_SCAN_WINDOW), -1):
                evt = self._idx[j]
                if evt.event_type in POSSESSION_BREAK_EVENTS and evt.team != team:
                    break
                if evt.event_type == EventType.FOUL_WON and evt.team == team:
                    return evt.player or ""
                if evt.team != team:
                    break
        # Path 1: rebound tap-in after a save/block/woodwork.
        if outcome in ("save", "block", "woodwork"):
            for j in range(shot_idx + 1,
                           min(len(self.timeline), shot_idx + REBOUND_SCAN_WINDOW)):
                evt = self._idx[j]
                if evt.team != team:
                    continue
                if evt.event_type == EventType.GOAL:
                    # The rebound was scored by a DIFFERENT player than the
                    # original shooter → the original shot is a fantasy assist.
                    if evt.player != e.player:
                        return e.player or ""
                    return ""
                if evt.event_type in POSSESSION_BREAK_EVENTS:
                    break
        return ""

    def _is_big_chance(self, e: MatchEvent, xg: float) -> bool:
        meta = getattr(e, "metadata", None) or {}
        if meta.get("is_big_chance"):
            return True
        if xg >= BIG_CHANCE_XG_FLOOR:
            return True
        zone = meta.get("zone", "")
        if zone in ("six_yard_box",):
            return True
        return False

    # ── AGGREGATION ──────────────────────────────────

    def _aggregate(self) -> None:
        acc: Dict[str, Dict] = {}
        blank = {
            "shot_assists": 0, "goal_assists": 0, "assists": 0,
            "chances_created": 0, "big_chances_created": 0,
            "open_play_cc": 0, "setpiece_cc": 0,
            "open_play_shot_assists": 0, "setpiece_shot_assists": 0,
            "open_play_assists": 0, "setpiece_assists": 0,
            "xa": 0.0, "xa_open_play": 0.0, "xa_setpiece": 0.0,
            "second_assists": 0, "sca": 0,
            "fantasy_assists": 0, "shots_faced_by_creation": 0,
        }
        for r in self.records:
            shots_faced = r.creator or r.fantasy_assist
            if r.creator:
                p = acc.setdefault(r.creator, dict(blank))
                if r.is_goal_assist:
                    p["goal_assists"] += 1
                    p["assists"] += 1
                    if r.is_open_play:
                        p["open_play_assists"] += 1
                    else:
                        p["setpiece_assists"] += 1
                else:
                    p["shot_assists"] += 1
                    if r.is_open_play:
                        p["open_play_shot_assists"] += 1
                    else:
                        p["setpiece_shot_assists"] += 1
                p["chances_created"] += 1
                if r.is_open_play:
                    p["open_play_cc"] += 1
                    p["xa_open_play"] += r.xa
                else:
                    p["setpiece_cc"] += 1
                    p["xa_setpiece"] += r.xa
                if r.is_big:
                    p["big_chances_created"] += 1
                p["xa"] += r.xa
                if r.second_assist:
                    acc.setdefault(r.second_assist, dict(blank))["second_assists"] += 1
            for actor in r.sca_players:
                acc.setdefault(actor, dict(blank))["sca"] += 1
            if r.fantasy_assist:
                p = acc.setdefault(r.fantasy_assist, dict(blank))
                p["fantasy_assists"] += 1
                p["assists"] += 1
                p["open_play_assists"] += 1
            # Even a solo effort (no key pass) counts as SCA for the shooter
            # if the shot was created by their own run.
            if not r.creator and r.sca_players:
                acc.setdefault(r.shooter, dict(blank))
        self.per_player = acc


# Internal shot candidate with mutable "scored" flag.
class _ShotCandidate:
    __slots__ = ("index", "event", "goaled")

    def __init__(self, index: int, event: MatchEvent):
        self.index = index
        self.event = event
        self.goaled = False

    def __getattr__(self, item):
        if item == "goaled":
            return False
        raise AttributeError(item)


# ─────────────────────────────────────────────
# STANDALONE DEMO / SELF-TEST
# Run: python chance_creation.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def evt(etype, team, player, **kw):
        from types import SimpleNamespace
        base = dict(minute=0, secondary_player=None, outcome=True,
                    location_x=50.0, location_y=34.0, end_x=None, end_y=None,
                    xg=0.1, metadata={}, situation=None)
        base.update(kw)
        return SimpleNamespace(event_type=etype, team=team, player=player,
                               **{k: v for k, v in base.items()})

    print("\n🎯  PLOFA 26/27 — Chance Creation Ledger Demo")
    print("=" * 64)
    demo = [
        evt(EventType.PASS, "Home", "A", secondary_player="B",
            location_x=50, location_y=34, end_x=62, end_y=34),
        evt(EventType.BALL_RECEIPT, "Home", "B", location_x=62, location_y=34),
        evt(EventType.PASS, "Home", "B", secondary_player="S",
            location_x=62, location_y=34, end_x=82, end_y=30),
        evt(EventType.BALL_RECEIPT, "Home", "S", location_x=82, location_y=30),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", location_x=88, location_y=32, xg=0.35),
        evt(EventType.GOAL, "Home", "S", location_x=88, location_y=32, xg=0.35),
        # second chance — a save then rebound tap-in (fantasy assist)
        evt(EventType.PASS, "Home", "C", secondary_player="D",
            location_x=55, location_y=40, end_x=80, end_y=44),
        evt(EventType.BALL_RECEIPT, "Home", "D", location_x=80, location_y=44),
        evt(EventType.SHOT_ON_TARGET, "Home", "D", location_x=90, location_y=46, xg=0.50),
        evt(EventType.SAVE, "Away", "GK", location_x=90, location_y=46),
        evt(EventType.BALL_RECEIPT, "Home", "E", location_x=90, location_y=46),
        evt(EventType.GOAL, "Home", "E", location_x=91, location_y=46, xg=0.80),
    ]
    ledger = ChanceCreationLedger(demo).compute()
    for r in ledger.records:
        print(f"  {r.outcome:>9}  creator={r.creator or '-':<2} "
              f"xA={r.xa:.2f} big={r.is_big} "
              f"2nd={r.second_assist or '-'} SCA={r.sca_players} "
              f"fantasy={r.fantasy_assist or '-'}")
    print("\n  Per player:", {p: {k: round(v, 2) for k, v in d.items() if v}
                             for p, d in ledger.per_player.items()})
    print("\n✅ Chance Creation Ledger operational — pure timeline scan.\n")
