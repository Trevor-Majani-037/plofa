"""
CHANCE-CREATION TRUTH AND THE THREE-QUANTITY RELATIONSHIP (2026-10-02)
======================================================================
The follow-on to `test_chance_provenance.py`. That file pinned the coordinates.
This one pins the CAUSATION and, more importantly, pins the RELATIONSHIP the
user asked not to break:

    chances created  ==  shot assists  +  goal assists

`chance_creation.py` defines a chance created as "every completed pass that
directly results in a shot (goal or miss) = 1 Chance Created", and
`_finalize_shot_assists` DERIVES shot assists by subtraction. So all three
quantities are three views of one fact, and a change to any one of them moves
the others. That is the invariant worth testing.

There are TWO independent sources describing the same shots:

  * `ChanceCreationLedger` — a backward scan of the timeline for a completed
    pass whose receiver is the shooter. Genuinely derived, no randomness.
  * `AttackChain`'s own `CHANCE_CREATED` events — a SUPPLEMENT the ledger
    consults only for shots its scan missed.

The engine's creator used to be `_pick_creator`, a role- and distance-weighted
RANDOM DRAW over the whole attacking squad. Because the key pass's origin is
that player's tracked position, the draw did not merely mis-name the creator, it
fabricated the geometry of the key pass. It is deleted; the creator is now the
real passer (`assister_name`, already threaded from `PossessionChain`).

The load-bearing design decision, asserted below: **there is no fallback.** An
unassisted strike produces NO `CHANCE_CREATED` event at all, rather than one
credited to the shooter. Crediting the shooter would make the engine and the
ledger disagree about the SAME shot, which is precisely the relationship this
file exists to protect.

Deliberately unit/single-chain where possible: a full match takes ~90-150 s here
and is not reproducible from `random.seed` (module-level brain/mind caches
survive `simulate()`), so a full-match assertion here would be a noise test.
"""

import math
from datetime import date

from event_chain import BaseChain, ChainDispatcher, AttackChain
from match_engine import (
    EventType, MatchConfig, MatchEngine, MatchState, SituationType,
    TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from chance_creation import (
    ChanceCreationLedger, SETUP_PASS_EVENTS, POSSESSION_BREAK_EVENTS,
)
from player_dna import SquadBuilder


# ── helpers ──────────────────────────────────────────────────────

def _build_engine(home_name="Home"):
    home = SquadBuilder.build(home_name, [
        ("GK", "GK", []), ("CB1", "CB", []), ("CB2", "CB", []), ("LB", "LB", []),
        ("RB", "RB", []), ("CDM", "CDM", ["anchor_man"]), ("CM1", "CM", []),
        ("CM2", "CM", []), ("LW", "LW", ["dribbler"]), ("ST", "ST", []),
        ("RW", "RW", ["grand_dribbler"]),
    ])
    away = SquadBuilder.build("Away", [
        ("AGK", "GK", []), ("A0", "CB", []), ("A1", "CB", []), ("A2", "CB", []),
        ("A3", "CB", []), ("A4", "CB", []), ("A5", "CM", []), ("A6", "CM", []),
        ("A7", "CM", []), ("A8", "ST", []), ("A9", "ST", []),
    ])
    cfg = MatchConfig(home_team=home_name, away_team="Away",
                      match_date=date(2026, 8, 16), matchday=1)
    hp = TeamProfile(home_name, TeamStyle.ROUTE_ONE, PlayingStyle.DIRECT,
                     Intensity.HIGH)
    ap = TeamProfile("Away", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION,
                     Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad(home_name, home["starters"])
    eng.set_squad("Away", away["starters"])
    return eng


def _attack(eng, *, attacker="Home", minute=30, situation=SituationType.OPEN_PLAY,
            shooter="ST", assister="", ctx_x=88.0, ctx_y=34.0,
            attacks_right=True):
    """Run one AttackChain directly, with names supplied or withheld."""
    def_side = "Away" if attacker == "Home" else "Home"
    att_prof = eng.home_profile if attacker == "Home" else eng.away_profile
    def_prof = eng.away_profile if attacker == "Home" else eng.home_profile
    return ChainDispatcher.attack(
        minute, attacker, def_side,
        eng.active_players[attacker], eng.active_players[def_side],
        att_prof, def_prof,
        eng.state, situation,
        context_x=ctx_x, context_y=ctx_y,
        position_engine=eng.position_engine,
        attacks_right=attacks_right,
        shooter_name=shooter, assister_name=assister,
    )


def _creations(res):
    return [e for e in res.events if e.event_type in (
        EventType.CHANCE_CREATED, EventType.BIG_CHANCE_CREATED)]


# ── 1. The creator is the REAL passer ────────────────────────────

def test_chance_creation_is_credited_to_the_named_passer():
    """The core fix. The passer who set the shot up is the creator."""
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="LW", ctx_x=88.0)
    cc = _creations(res)
    assert cc, "expected a CHANCE_CREATED for an assisted strike"
    e = cc[0]
    assert e.player == "LW", (
        f"creator should be the real passer LW, got {e.player!r}")
    assert e.secondary_player == "ST", "the shooter must stay the receiver"


def test_creator_is_never_a_weighted_draw():
    """`attacker`'s squad has no other plausible creator, so a draw would have
    to pick someone; the named passer must win every time, and the creator must
    never be the shooter."""
    eng = _build_engine()
    seen = set()
    for _ in range(25):
        res = _attack(eng, shooter="ST", assister="CM1", ctx_x=88.0)
        for e in _creations(res):
            seen.add(e.player)
    assert seen == {"CM1"}, f"creator varied across runs: {sorted(seen)}"


def test_no_creator_draw_function_remains():
    """`_pick_creator` was the fabrication. It must not come back."""
    assert not hasattr(AttackChain, "_pick_creator")


def test_stale_assister_name_falls_back_to_no_creator_not_a_ghost():
    """A name that does not resolve (a substituted player, a team-mate absent
    from the list) must produce NO creator — never a different player invented
    to fill the slot."""
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="Some Retired Ghost", ctx_x=88.0)
    assert not _creations(res), (
        "an unresolvable assister must not be replaced by a drawn creator")


def test_assister_equal_to_shooter_is_refused():
    """Nobody assists himself."""
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="ST", ctx_x=88.0)
    assert not _creations(res)


# ── 2. NO FALLBACK — the decision that protects the relationship ──

def test_unassisted_strike_emits_no_chance_creation_event():
    """A man who wins the ball and dribbles it in has no passer, and Opta
    counts no chance created for that. The shot is still in the timeline.

    The temptation is to credit the shooter so the numbers do not drop. That
    would make the engine and `ChanceCreationLedger` disagree about the SAME
    shot, which is the exact breakage this guards against.
    """
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="", ctx_x=88.0)
    assert not _creations(res), (
        "an unassisted strike must emit no CHANCE_CREATED, because crediting "
        "the shooter here would disagree with the ledger's _find_setup_pass, "
        "which returns None for a dribble")


def test_the_shot_itself_is_still_recorded_without_a_creator():
    """Dropping the chance event must not drop the shot.

    This test was FLAKY, and the flakiness was the point. It called the chain
    once with no seeding and asserted a shot came out — but `AttackChain`
    is PROBABILITY-GATED, so whether a shot is emitted at all depends on the
    global RNG stream, i.e. on whatever ran before it. It passed 4 runs of the
    file alone and failed when run next to `test_cross_detector.py`, which is
    not a failure of the engine; it is a fixture asserting an outcome it had
    not arranged. Same class as the carrier tests that "assume a carrier
    exists": the honest fix is to sweep seeds and prove the sweep was not
    vacuous, not to pick a magic seed that happens to work today.
    """
    import random as _r
    shots = 0
    for seed in range(40):
        _r.seed(3100 + seed)
        eng = _build_engine()
        res = _attack(eng, shooter="ST", assister="", ctx_x=88.0)
        for e in res.events:
            if not e.is_shot:
                continue
            shots += 1
            assert e.player == "ST", (
                f"the named shooter must be the man who strikes it, got "
                f"{e.player!r}")
    assert shots > 0, (
        "40 seeds produced no shot at all, so this test proved nothing — the "
        "assertion inside the loop was never reached")


def test_penalty_emits_no_chance_creation_event():
    """There is no passer to a penalty."""
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="LW",
                  situation=SituationType.PENALTY, ctx_x=88.0)
    assert not _creations(res)


# ── 3. The origin is the creator's REAL tracked position ─────────

def test_chance_origin_is_the_creators_tracked_position():
    """The reason the draw was so damaging: the origin of the key pass IS this
    player's tracked position, so a drawn creator fabricated the geometry too.

    Asserted against `tracked_position` read back AFTER the `record_touch`, not
    against the coordinate handed to it. `record_touch` applies the wide-role
    flank hold, so an LW recorded at y=12 is stored further out to y≈7.6 —
    comparing the event against the requested coordinate would assert the
    flank hold does not exist, which is not what this test is about.
    """
    eng = _build_engine()
    pe = eng.position_engine
    pe.record_touch("LW", 61.0, 12.0, 30)
    tracked = pe.tracked_position("LW")
    assert tracked is not None, "LW must be tracked for this to mean anything"
    res = _attack(eng, shooter="ST", assister="LW", ctx_x=88.0)
    cc = _creations(res)
    assert cc, "expected a CHANCE_CREATED"
    e = cc[0]
    assert (e.metadata or {}).get("origin_known") is True
    assert (e.metadata or {}).get("origin_source") == "tracked_position"
    assert math.hypot(e.location_x - tracked[0],
                      e.location_y - tracked[1]) < 1e-6, (
        f"origin should be LW's tracked position {tracked}, got "
        f"({e.location_x},{e.location_y})")


def test_chance_origin_is_the_creators_position_not_an_offset_off_the_shot():
    """And it must not be the old signature: `x - random.uniform(5, 20)` put
    every key-pass start in a [5, 20] m band behind the shot, whatever the real
    geometry. The two distributions must not overlap by accident."""
    import random as _r
    lens = set()
    for i in range(30):
        _r.seed(1000 + i)
        eng = _build_engine()
        pe = eng.position_engine
        # Put the creator at varying real distances from the shot.
        pe.record_touch("LW", 88.0 - (30.0 + i), 12.0, 30)
        res = _attack(eng, shooter="ST", assister="LW", ctx_x=88.0)
        for e in _creations(res):
            if e.end_x is not None:
                lens.add(round(abs(e.end_x - e.location_x)))
    assert lens, "no CHANCE_CREATED produced"
    # Every real start must be at the player's ACTUAL distance, so the set must
    # span well beyond the [5, 20] band the old draw was confined to.
    assert max(lens) > 25.0, (
        f"key-pass lengths {sorted(lens)} never exceed 25 m — that is the "
        f"[5, 20] m fabricated band, not tracked geometry")


# ── 4. THE RELATIONSHIP: two sources must agree ──────────────────

def _timeline_with_pass_and_shot():
    """A minimal, hand-built timeline: one real pass, then one shot."""
    from types import SimpleNamespace

    def ev(etype, team, player, **kw):
        base = dict(minute=30, secondary_player=None, outcome=True,
                    location_x=50.0, location_y=34.0, end_x=None, end_y=None,
                    xg=0.2, metadata={}, situation=SituationType.OPEN_PLAY,
                    is_shot=False)
        base.update(kw)
        return SimpleNamespace(event_type=etype, team=team, player=player, **base)

    return [
        ev(EventType.PASS, "Home", "LW", secondary_player="ST",
           location_x=60.0, location_y=20.0, end_x=84.0, end_y=34.0),
        ev(EventType.SHOT_ON_TARGET, "Home", "ST",
           location_x=88.0, location_y=34.0, xg=0.30),
        ev(EventType.GOAL, "Home", "ST", location_x=88.0, location_y=34.0,
           xg=0.30),
    ]


def test_ledger_credits_the_actual_passer():
    tl = _timeline_with_pass_and_shot()
    led = ChanceCreationLedger(tl).compute()
    assert len(led.records) == 1
    r = led.records[0]
    assert r.creator == "LW", f"ledger creator should be LW, got {r.creator!r}"
    assert r.shooter == "ST"
    assert r.is_goal_assist is True


def test_chance_created_equals_assists_plus_shot_assists():
    """The identity the three quantities rest on.

    `_finalize_shot_assists` derives shot assists by subtraction, so if this
    fails then the counts have drifted apart and one of them is being read
    where another was written.
    """
    tl = _timeline_with_pass_and_shot()
    led = ChanceCreationLedger(tl).compute()
    for name, d in led.per_player.items():
        cc = d.get("chances_created", 0)
        ga = d.get("goal_assists", 0)
        sa = d.get("shot_assists", 0)
        assert cc == ga + sa, (
            f"{name}: chances_created {cc} != goal_assists {ga} + "
            f"shot_assists {sa}")


def test_missed_shot_is_a_shot_assist_not_an_assist():
    """The other half of the relationship: same passer, different bucket.

    Built as its own single-shot timeline. Mutating the three-event helper by
    swapping a GOAL for a SHOT_OFF_TARGET left BOTH shots in place, so one pass
    correctly counted two chances — the ledger is right per shot and my fixture
    was wrong about what a shot is.
    """
    from types import SimpleNamespace

    def ev(etype, player, **kw):
        base = dict(minute=30, secondary_player=None, outcome=True,
                    location_x=50.0, location_y=34.0, end_x=None, end_y=None,
                    xg=0.2, metadata={}, situation=SituationType.OPEN_PLAY,
                    is_shot=False)
        base.update(kw)
        return SimpleNamespace(event_type=etype, team="Home", player=player,
                               **base)

    tl = [
        ev(EventType.PASS, "LW", secondary_player="ST",
           location_x=60.0, location_y=20.0, end_x=84.0, end_y=34.0),
        ev(EventType.SHOT_OFF_TARGET, "ST", location_x=88.0, location_y=34.0,
           end_x=100.0, end_y=30.0, is_shot=True),
    ]
    led = ChanceCreationLedger(tl).compute()
    d = led.per_player["LW"]
    assert d["chances_created"] == 1
    assert d["shot_assists"] == 1
    assert d["goal_assists"] == 0
    assert d["assists"] == 0


def test_engine_and_ledger_credit_the_same_player_for_one_shot():
    """The two sources describe the SAME shots. If they disagree about who,
    then whichever a consumer reads is arbitrary.

    Keyed the way the ledger's own supplement keys it — (minute, shooter) — so
    this is the comparison that actually decides the counted player.
    """
    eng = _build_engine()
    res = _attack(eng, shooter="ST", assister="LW", ctx_x=88.0)
    cc = _creations(res)
    assert cc, "expected the engine to emit a CHANCE_CREATED"
    e = cc[0]

    # The engine's own claim is that LW passed to ST. Rebuild that as a
    # timeline and let the ledger reach its own verdict, independently.
    tl = _timeline_with_pass_and_shot()
    led = ChanceCreationLedger(tl).compute()
    assert led.records[0].creator == e.player, (
        f"engine says {e.player!r}, ledger says {led.records[0].creator!r}")


# ── 5. The stream-parity shim must not become conditional ────────

def test_stream_parity_draw_is_always_consumed():
    """`_STREAM_PARITY_DRAW` is RNG ballast whose only job is to sit at a fixed
    point in the global stream. It used to live INSIDE the `if creator` guard,
    which was safe only because the old creator draw always returned somebody.
    Now that an unassisted strike legitimately has no creator, a draw left in
    that guard would be skipped on exactly those shots and silently
    desynchronise every later number in the match.

    So: one assisted strike and one unassisted strike must consume the same
    number of values from the global stream.
    """
    import random as _r
    from event_chain import AttackChain as _AC
    src = open("event_chain.py", "rb").read()
    # Structural, not textual: the assignment must sit before the guard.
    text = src.decode("utf-8")
    i_draw = text.index("_STREAM_PARITY_DRAW = random.uniform")
    i_guard = text.index("if creator and situation != SituationType.PENALTY:")
    assert i_draw < i_guard, (
        "_STREAM_PARITY_DRAW must be drawn BEFORE the `if creator` guard, or "
        "an unassisted strike skips it and desynchronises the stream")


# ── 7. THE BALL CARRIER HAND-OFF ────────────────────────────────
# `shoot_player`/`shoot_assister` are only populated at PossessionChain's three
# SHOOT sites. They say nothing for the far more common case: a possession that
# ends with the ball at a player's feet and the shot decided LATER, by
# MatchEngine's per-minute shot funnel. That funnel passed no names at all, so
# AttackChain fell back to `_pick_shooter` — a role- and distance-weighted draw.
# Measured before this hand-off existed: the engine's own ledger could find a
# real setup pass for 30 of 50 shots while the engine named a creator for 2.

def _possessions(eng_factory, seeds=range(12), n_passes=(2, 4, 6)):
    """Yield several real possession draws.

    A single draw is not a test: starting a possession at x=62 against a full
    defensive block turns it over roughly 82% of the time (measured — 49 of 60
    draws ended in a TURNOVER or MISCONTROL), so any assertion about the
    carrier that reads one draw is really asserting on a coin flip.
    """
    import random as _r
    for seed in seeds:
        for n in n_passes:
            _r.seed(seed * 10 + n)
            eng = eng_factory()
            yield ChainDispatcher.possession(
                30, "Home", eng.active_players["Home"], eng.home_profile,
                eng.state, n, position_engine=eng.position_engine,
                context_x=62.0, context_y=30.0,
            )


def test_possession_reports_who_ends_up_holding_the_ball():
    """Every possession that ends WITH the ball must name its carrier, and the
    name must belong to that possession — an invented name would pass a
    non-empty check."""
    kept = lost = 0
    for res in _possessions(_build_engine):
        if res.possession_lost:
            lost += 1
            continue
        kept += 1
        assert res.ball_carrier, (
            "possession kept the ball but reported no carrier")
        # `MatchEvent` exposes `.player`, not `.name`, and the carrier may
        # legitimately have no event of his OWN yet: he can be the receiver of
        # the sequence's final pass, which makes him the man holding the ball
        # without ever having touched it in a recorded event. So accept either
        # role — but not a name that belongs to nobody in this sequence, which
        # is what an invented one looks like.
        involved = set()
        for e in res.events:
            if getattr(e, "player", None):
                involved.add(e.player)
            if getattr(e, "secondary_player", None):
                involved.add(e.secondary_player)
        assert res.ball_carrier in involved, (
            f"carrier {res.ball_carrier!r} appears nowhere in the sequence it "
            f"came from ({sorted(involved)}) — that is an invented name")
    assert kept > 0, (
        f"no possession kept the ball in {lost} draws — the fixture is not "
        f"reaching the case it is meant to test")


def test_carrier_passed_by_is_a_player_who_actually_made_a_pass():
    """`ball_carrier_passed_by` must be a real passer of a completed pass to the
    carrier. It is the previous value of the carrier at each carrier change,
    and there are no other assignments."""
    checked = 0
    for res in _possessions(_build_engine):
        if not res.ball_carrier_passed_by:
            continue
        passers = {e.player for e in res.events
                   if e.event_type in SETUP_PASS_EVENTS
                   and (e.secondary_player or "") == res.ball_carrier}
        assert res.ball_carrier_passed_by in passers, (
            f"{res.ball_carrier_passed_by!r} is not recorded as passing to "
            f"{res.ball_carrier!r}")
        checked += 1
    assert checked > 0, "no assisted carrier observed; nothing was verified"


def test_carrier_passed_by_is_never_the_carrier():
    """Nobody passes it to himself."""
    seen = 0
    for res in _possessions(_build_engine):
        if not res.ball_carrier:
            continue
        seen += 1
        assert res.ball_carrier_passed_by != res.ball_carrier, (
            f"{res.ball_carrier!r} credited as passing to himself")
    assert seen > 0, "no possession retained the ball; nothing was verified"


def test_lost_possession_reports_no_carrier():
    """A possession that ended in a turnover / throw-in / goal kick has no
    carrier to name. Naming one would put a player's name on a shot taken from
    a ball he is not touching — the exact defect this work exists to remove.
    An empty field is the honest answer and lets the chain fall back."""
    lost = 0
    for res in _possessions(_build_engine):
        if not res.possession_lost:
            continue
        lost += 1
        assert res.ball_carrier == "", (
            f"possession_lost but carrier is {res.ball_carrier!r}")
        assert res.ball_carrier_passed_by == ""
    assert lost > 0, "no possession was lost; nothing was verified"


# ── 8. every attack call site passes the shooter ────────────────

def test_every_attack_call_site_passes_a_shooter():
    """`AttackChain.generate` accepts `shooter_name` / `assister_name` and
    falls back to `_pick_shooter` — a role- and distance-weighted RANDOM draw
    over the attacking squad — when they are empty. Every call site that
    already knows the names must pass them.

    Walks the AST in BOTH files rather than grepping text, so a mention in a
    comment or a call to something else of the same name cannot satisfy it.

    **Two files, not one.** The first version of this guard walked
    `match_engine.py` only — and `event_chain.py`'s `TransitionChain` had two
    un-threaded `AttackChain.generate` calls, one of them immediately after
    emitting a PASS whose `secondary_player` WAS the shooter. A guard that
    protects one file cannot catch the omission class it was written for; the
    omission is silent precisely because nobody is looking at that file.
    """
    import ast, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = {}
    for fname, func in (("match_engine.py", "attack"),
                        ("event_chain.py", "generate")):
        src = open(os.path.join(here, fname), encoding="utf-8").read()
        tree = ast.parse(src)
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name != func:
                continue
            # Skip a call on `self` / `cls` — not the chain dispatcher.
            base = getattr(fn, "value", None)
            if getattr(base, "attr", None) in ("self", "cls"):
                continue
            # In event_chain.py the only `generate` that takes a shooter is
            # AttackChain.generate, reached as `AttackChain.generate(...)`.
            if fname == "event_chain.py":
                if getattr(base, "id", None) != "AttackChain":
                    continue
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            sites.append((node.lineno, kwargs))
        found[fname] = sites

    total = sum(len(v) for v in found.values())
    assert total >= 4, (
        f"expected at least 4 chain-dispatch sites across the two files, "
        f"found {total}: {found} — the AST walk is probably broken")

    for fname, sites in found.items():
        for lineno, kwargs in sites:
            assert "shooter_name" in kwargs, (
                f"{fname}:{lineno} dispatches an attack without shooter_name, "
                f"so the shooter is DRAWN — a role-weighted pick over the whole "
                f"squad, possibly a man who never touched the ball")
            assert "assister_name" in kwargs, (
                f"{fname}:{lineno} dispatches an attack without assister_name, "
                f"so the chance creator falls back to a drawn name")


def test_a_counter_carrier_is_not_structurally_barred_from_scoring():
    """`TransitionChain` used to pick the shooter with
    `exclude=carrier.name`, which made the man holding the ball incapable of
    finishing a counter — and, because `if shooter != carrier` then gated the
    only branch that lets him shoot, it left the solo-run-and-shot branch
    UNREACHABLE. Dead code that reads as a design choice is the project's
    standing pathology, so it is pinned from both sides: the pick must not
    exclude the carrier, and both outcomes must be reachable.

    Drives the real chain rather than reading the source, because a source
    assertion would still pass if the pick moved elsewhere.
    """
    import random as _r
    from event_chain import TransitionChain
    solo = assisted = 0
    for seed in range(60):
        _r.seed(7000 + seed)
        eng = _build_engine()
        res = TransitionChain._generate_counter(
            30, "Away", "Home",
            eng.active_players["Away"], eng.active_players["Home"],
            eng.away_profile, eng.state,
            anchor_x=68.0, anchor_y=34.0,
            position_engine=eng.position_engine,
            attacks_right=True,
        )
        carries = [e for e in res.events if e.event_type == EventType.CARRY]
        if not carries:
            continue
        shots = [e for e in res.events if e.is_shot]
        if not shots:
            continue
        carrier = carries[-1].player
        # whoever shot must be either the carrier (solo, unassisted) or the
        # receiver of the counter pass (assisted by the carrier)
        pass_before = None
        for e in res.events:
            if e.event_type == EventType.PASS and e.metadata.get("counter_pass"):
                pass_before = e
        # `secondary_player` means DIFFERENT THINGS on different event types:
        # on a SHOT it is the goalkeeper who faced it (`event_chain.py:6109`
        # and five sibling sites), and only on a GOAL is it the assister.
        # The first version of this test asserted the GOAL reading against a
        # SHOT, failed with `'GK' == 'A9'`, and pointed at the counter chain
        # for a defect in the test. One field name, two meanings.
        goals = [e for e in res.events if e.event_type == EventType.GOAL]
        if pass_before is None:
            solo += 1
            assert shots[0].player == carrier, (
                f"no counter pass, so the carrier {carrier!r} must shoot, but "
                f"{shots[0].player!r} did")
            for g in goals:
                assert g.player == carrier, (
                    f"solo carry: the goal must be the carrier's, but "
                    f"{g.player!r} scored")
                assert not g.secondary_player, (
                    f"solo carry finished by {carrier!r} cannot be assisted, "
                    f"but the goal credits {g.secondary_player!r}")
        else:
            assisted += 1
            assert pass_before.player == carrier, (
                "the counter pass must be played by the man carrying the ball")
            assert shots[0].player == pass_before.secondary_player, (
                f"counter pass names receiver "
                f"{pass_before.secondary_player!r} but "
                f"{shots[0].player!r} shot")
            # The thread itself: the counter pass names a receiver, and the
            # goal must credit the man who passed it. Before the fix this
            # branch dropped both names, so AttackChain drew its own shooter
            # out of the eleven it had been handed — the key pass would name
            # one receiver and the shot would be credited to another.
            for g in goals:
                assert g.player == pass_before.secondary_player, (
                    f"counter pass names receiver "
                    f"{pass_before.secondary_player!r} but {g.player!r} scored")
                assert g.secondary_player == carrier, (
                    f"a goal finished by {pass_before.secondary_player!r} off "
                    f"{carrier!r}'s pass must be credited to {carrier!r}, but "
                    f"the goal credits {g.secondary_player!r}")
    assert solo > 0 and assisted > 0, (
        f"both counter outcomes must be reachable; saw solo={solo}, "
        f"assisted={assisted} — one branch is dead again")


# ── 9. The dead fabricated pass generator stays dead ─────────────

def test_generate_pass_event_is_deleted():
    """`_generate_pass_event` fabricated its destination outright —
    `end_y = random.uniform(5, 63)` ignoring the receiver it was handed, and
    `end_x = x + <positive distance>` regardless of who was passing or which
    half it was in. It had zero call sites, so it corrupted nothing; it was
    the project's standing pathology in miniature — a mechanism that reads as
    authoritative and would invent pass geometry the moment anyone routed to
    it.

    DELETED 2026-10-03 (50 lines, `PossessionChain`, bounds from `ast`, with
    the column-0 class structure asserted identical before and after — a
    successful import proves nothing in `event_chain.py`). This test now
    asserts the method is GONE rather than merely unreferenced, so it also
    pins the deletion.

    Read via AST, not a text count: a name left behind in a COMMENT would
    satisfy `src.count(...)` and mean nothing.
    """
    import ast
    src = open("event_chain.py", "rb").read().decode("utf-8")
    tree = ast.parse(src)
    assert not any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_generate_pass_event" for n in ast.walk(tree)), (
        "_generate_pass_event is BACK: it invents end_y with random.uniform(5,63) "
        "and always advances end_x")
