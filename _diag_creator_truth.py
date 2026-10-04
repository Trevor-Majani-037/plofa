"""IS CHANCE CREATION REAL, AND DO THE THREE QUANTITIES AGREE? (2026-10-02)

This is the test that matters, and it is deliberately NOT a re-run of the
ledger. The ledger (`chance_creation.py`) awards a chance created to a
completed PASS that results in a shot. The engine emits its own
CHANCE_CREATED events as a fallback for shots the ledger's scan misses. Those
are two independent sources describing the SAME shots, so they must agree.

Three questions, in order of how much they would matter:

  A. IS THE CREATOR REAL?  A fabricated creator is one who never passed the
     ball. So: scan backward from each CHANCE_CREATED for a completed pass by
     that player to that shooter. The old `_pick_creator` was a role- and
     distance-weighted DRAW and failed this badly — the named creator had
     frequently never touched the ball.

  B. DO THE TWO SOURCES AGREE?  For a shot both describe, the engine's
     CHANCE_CREATED and the ledger's scan must name the SAME player. This is
     the relationship the user asked not to break: chances created, shot
     assists and assists are three views of one fact.

  C. IS THE IDENTITY INTACT?  chance_creation.py defines
     chance created = goal assists + shot assists. Check it holds per player,
     because `_finalize_shot_assists` DERIVES shot assists by subtraction — a
     silent change in either input moves all three.

Also reports the counts, because the honest answer is allowed to be smaller
than the fabricated one and that is not a regression.

Run: .venv\\Scripts\\python.exe _diag_creator_truth.py [n]
"""
import random
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from match_engine import EventType
from chance_creation import (
    ChanceCreationLedger, SETUP_PASS_EVENTS, POSSESSION_BREAK_EVENTS, SHOT_EVENTS,
)

from _diag_chance_coords import build_pair

PASS_LIKE = SETUP_PASS_EVENTS


def _backward_pass_to(tl, idx, team, passer, receiver, window=24):
    """Did `passer` make a completed pass to `receiver` just before `idx`?

    Independent of the ledger: same possession rules, written separately, so a
    bug in one does not silently excuse the other.
    """
    for j in range(idx - 1, max(-1, idx - window), -1):
        e = tl[j]
        if e.event_type in POSSESSION_BREAK_EVENTS and e.team != team:
            return False
        if e.event_type not in PASS_LIKE:
            continue
        if e.team != team:
            return False
        if not getattr(e, "outcome", True):
            continue
        if e.player == passer and (e.secondary_player or "") == receiver:
            return True
    return False


def analyse(res):
    tl = res.timeline
    led = ChanceCreationLedger(tl).compute()

    cc = [e for e in tl if e.event_type in (
        EventType.CHANCE_CREATED, EventType.BIG_CHANCE_CREATED)]

    # ── A. IS THE CREATOR REAL? ───────────────────────────────────────
    real = fake = unknown = 0
    fake_detail = []
    for e in cc:
        idx = tl.index(e)
        shooter = e.secondary_player or ""
        if not e.player or not shooter:
            unknown += 1
            continue
        if _backward_pass_to(tl, idx, e.team, e.player, shooter):
            real += 1
        else:
            fake += 1
            fake_detail.append((e.minute, e.player, shooter))

    # ── B. DO THE TWO SOURCES AGREE? ──────────────────────────────────
    # Key on (minute, shooter) exactly as the ledger's own supplement does,
    # so this is the comparison that actually decides the counted player.
    ledger_creator = {}
    for r in led.records:
        if r.creator:
            ledger_creator.setdefault((r.minute, r.shooter), r.creator)
    agree = disagree = only_engine = only_ledger = 0
    pairs = []
    for e in cc:
        shooter = e.secondary_player or ""
        key = (e.minute or 0, shooter)
        lc = ledger_creator.get(key)
        if lc is None:
            only_engine += 1
            pairs.append((e.minute, e.player, shooter, "(ledger: no key pass)"))
        elif lc == e.player:
            agree += 1
        else:
            disagree += 1
            pairs.append((e.minute, e.player, shooter, lc))
    for key, lc in ledger_creator.items():
        if not any((e.minute or 0) == key[0] and (e.secondary_player or "") == key[1]
                   for e in cc):
            only_ledger += 1

    # ── C. IS THE IDENTITY INTACT? ────────────────────────────────────
    ident_ok = ident_bad = 0
    bad = []
    for p, d in led.per_player.items():
        cc_n = d.get("chances_created", 0)
        ga = d.get("goal_assists", 0)
        sa = d.get("shot_assists", 0)
        if cc_n == ga + sa:
            ident_ok += 1
        else:
            ident_bad += 1
            bad.append((p, cc_n, ga, sa))

    # ── D. GOAL ASSIST vs LEDGER CREATOR ───────────────────────────────
    # The first version of this probe had a single "unassisted" bucket that
    # conflated two DIFFERENT things: a goal carrying no assist on the event,
    # and a goal the ledger had no creator for. They have opposite meanings —
    # the first is the engine's claim, the second is the ledger's — and mixing
    # them produced a confident-looking "7 of 7 unassisted" that was really two
    # unanswered questions. Never share a counter.
    gast = [g for g in res.goals if g.event_type != EventType.OWN_GOAL]
    g_no_assist = g_assisted = g_agree = g_dis = g_no_rec = 0
    for g in gast:
        eng = g.secondary_player or ""
        if not eng:
            g_no_assist += 1
            continue
        g_assisted += 1
        rec = next((r for r in led.records
                    if r.outcome == "goal" and r.shooter == g.player
                    and abs((r.minute or 0) - (g.minute or 0)) <= 1), None)
        if rec is None:
            g_no_rec += 1
        elif rec.creator == eng:
            g_agree += 1
        else:
            g_dis += 1

    # ── E. IS THE SHOOTER THE MAN WHO HAD THE BALL? ────────────────────
    # The decisive test of the CAUSAL shooter fix, and the one the drawn
    # shooter failed hardest. Walk back from each shot to the last event that
    # touched the ball for that team: a completed pass means the shooter must
    # be its `secondary_player` (the receiver), a carry or dribble means the
    # shooter must be its `player`. A shooter who is neither was drawn, not
    # tracked. Independent of the ledger, so the two can disagree honestly.
    TOUCH = SETUP_PASS_EVENTS | {EventType.CARRY, EventType.DRIBBLE_SUCCESS}
    shots = [e for e in tl if e.event_type in SHOT_EVENTS]
    sh_real = sh_drawn = sh_unprovable = 0
    drawn_detail = []
    for e in shots:
        idx = tl.index(e)
        expected = None
        for j in range(idx - 1, max(-1, idx - 24), -1):
            prev = tl[j]
            if prev.event_type in POSSESSION_BREAK_EVENTS and prev.team != e.team:
                break
            if prev.team != e.team or prev.event_type not in TOUCH:
                continue
            expected = (prev.secondary_player
                        if prev.event_type in SETUP_PASS_EVENTS else prev.player)
            break
        if not expected:
            sh_unprovable += 1
        elif expected == e.player:
            sh_real += 1
        else:
            sh_drawn += 1
            drawn_detail.append((e.minute, e.player, expected))

    return {
        "shots": len(led.records), "ledger_creators": len(ledger_creator),
        "cc": len(cc), "real": real, "fake": fake, "unknown": unknown,
        "fake_detail": fake_detail, "agree": agree, "disagree": disagree,
        "only_engine": only_engine, "only_ledger": only_ledger,
        "pairs": pairs, "ident_ok": ident_ok, "ident_bad": ident_bad,
        "bad": bad, "goals": len(gast), "g_no_assist": g_no_assist,
        "g_assisted": g_assisted, "g_agree": g_agree, "g_dis": g_dis,
        "g_no_rec": g_no_rec,
        "n_shots": len(shots), "sh_real": sh_real, "sh_drawn": sh_drawn,
        "sh_unprovable": sh_unprovable, "drawn_detail": drawn_detail,
        "per_player": led.per_player,
    }


# ── THE BEFORE-CASE ───────────────────────────────────────────────
# Section E ("is the shooter the man who had the ball?") is the whole point of
# the carrier hand-off, but 87% on its own is not evidence of anything. The
# contrast has to come from the code that was there before.
#
# It CANNOT come from reverting the source and running both arms in one
# process: module-level brain / mind caches survive `simulate()`, so the second
# arm inherits the first's state and the comparison is noise (AGENTS.md, fifth
# occurrence). It also cannot come from a git stash, because that would risk
# leaving production modified while a background match is running.
#
# So un-thread it from INSIDE the probe, with no source edit at all:
# strip the two kwargs on the way into `ChainDispatcher.attack` and
# `AttackChain` sees exactly what it saw before — empty names, hence
# `_pick_shooter`, the role- and distance-weighted draw. Each arm runs in its
# own process.
_NO_THREAD = {"on": False}
_ORIG_ATTACK = None


def install_no_thread():
    """Make `ChainDispatcher.attack` behave as it did before the hand-off."""
    global _ORIG_ATTACK
    from event_chain import ChainDispatcher

    # `attack` is a STATICMETHOD, so `ChainDispatcher.attack` is a plain
    # function with no `__func__`. My first wrapper assumed a classmethod,
    # passed `cls` as a leading positional, and shifted every argument by one
    # — which surfaced as the deeply misleading
    #   "got multiple values for argument 'position_engine'"
    # rather than as the arity bug it actually was. Read the descriptor out of
    # the class __dict__ instead of inferring it from the bound object.
    raw = ChainDispatcher.__dict__["attack"]
    inner = raw.__func__          # true for BOTH classmethod and staticmethod
    _ORIG_ATTACK = inner

    if isinstance(raw, classmethod):
        def stripped(cls, *a, **kw):
            kw.pop("shooter_name", None)
            kw.pop("assister_name", None)
            return inner(cls, *a, **kw)
        ChainDispatcher.attack = classmethod(stripped)
    else:
        def stripped(*a, **kw):
            kw.pop("shooter_name", None)
            kw.pop("assister_name", None)
            return inner(*a, **kw)
        ChainDispatcher.attack = staticmethod(stripped)

    _NO_THREAD["on"] = True
    # Prove the two things that matter, rather than asserting on the wrapper's
    # own signature — which is `(*a, **kw)` by construction and therefore says
    # nothing at all. (It did, in fact: the first version of this check failed
    # while the wrapper was working exactly as intended.)
    import inspect
    raw_params = list(inspect.signature(inner).parameters)
    assert "shooter_name" in raw_params and "assister_name" in raw_params, (
        f"{raw_params} — the descriptor found is not ChainDispatcher.attack, "
        f"so this probe would silently measure nothing")
    assert ChainDispatcher.__dict__["attack"] is not raw, "wrapper not installed"
    print(f"  ** BEFORE-CASE: attack is a {type(raw).__name__}; "
          f"shooter/assister stripped **", flush=True)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    n = int(argv[0]) if argv else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    tot = Counter()
    ident_bad_all, pairs_all, drawn_all = [], [], []
    if "--no-thread" in flags:
        install_no_thread()
        print("  ** BEFORE-CASE: the carrier hand-off is DISABLED "
              "(shooter/assister stripped) **", flush=True)
    for i in range(n):
        h, a = pairs[i % len(pairs)]
        random.seed(3000 + i)
        print(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
        res = build_pair(h, a).simulate()
        r = analyse(res)
        for k in ("shots", "ledger_creators", "cc", "real", "fake", "unknown",
                  "agree", "disagree", "only_engine", "only_ledger",
                  "ident_ok", "ident_bad", "goals", "g_no_assist", "g_assisted",
                  "g_agree", "g_dis", "g_no_rec", "n_shots", "sh_real",
                  "sh_drawn", "sh_unprovable"):
            tot[k] += r[k]
        ident_bad_all += r["bad"]
        pairs_all += r["pairs"]
        drawn_all += r["drawn_detail"]
        print(f"      {res.home_goals}-{res.away_goals}  "
              f"CC {r['cc']}  real {r['real']}  fake {r['fake']}  "
              f"sources agree {r['agree']} disagree {r['disagree']}  | "
              f"shooter real {r['sh_real']}/{r['n_shots']} "
              f"drawn {r['sh_drawn']}",
              flush=True)

    print("\n" + "=" * 74)
    print("VERDICT")
    print(f"  ledger shots {tot['shots']}, of which the scan found a creator "
          f"for {tot['ledger_creators']}")
    print(f"  engine CHANCE_CREATED events {tot['cc']}")
    print(f"  A. creator REALLY passed the ball   {tot['real']}/{tot['cc']}"
          f"   (fabricated {tot['fake']}, unusable {tot['unknown']})")
    print(f"  B. engine vs ledger, same player    {tot['agree']} agree, "
          f"{tot['disagree']} DISAGREE")
    print(f"     engine-only {tot['only_engine']}  ledger-only {tot['only_ledger']}")
    print(f"  C. chances_created == goal_assists + shot_assists: "
          f"{tot['ident_ok']} players ok, {tot['ident_bad']} broken")
    print(f"  D. goals {tot['goals']}: engine emitted an assist for "
          f"{tot['g_assisted']}, none for {tot['g_no_assist']}")
    print(f"     of those assisted, ledger creator AGREES {tot['g_agree']}, "
          f"DISAGREES {tot['g_dis']}, ledger has no record {tot['g_no_rec']}")
    _prov = tot['sh_real'] + tot['sh_drawn']
    print(f"  E. shooter IS the man who had the ball   {tot['sh_real']}/{_prov}"
          f"   (drawn {tot['sh_drawn']}, unprovable {tot['sh_unprovable']}"
          f" of {tot['n_shots']} shots)")
    if pairs_all:
        print("\n  disagreements / engine-only (minute, engine creator, shooter, ledger):")
        for m, a, s, l in pairs_all[:20]:
            print(f"    {m:>3}  {a:<22} -> {s:<22} | {l}")
    if ident_bad_all:
        print("\n  broken identities (player, cc, ga, sa):")
        for p, c, g, s in ident_bad_all[:20]:
            print(f"    {p:<22} cc={c} ga={g} sa={s}")
    if drawn_all:
        print("\n  drawn shooters (minute, shooter, who actually had the ball):")
        for m, a, e in drawn_all[:20]:
            print(f"    {m:>3}  {a:<22} (ball was with {e})")


if __name__ == "__main__":
    main()
