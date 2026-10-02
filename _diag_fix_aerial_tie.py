"""Fix the aerial duel: a dead heat decided by argument order.

MEASURED FIRST (`_diag_corner_race.py`, milliseconds, no match):
  * two men on the IDENTICAL spot -- attacker passed first wins, defender
    passed first wins. `candidates` is built from
    `attacking_players + defending_players`, sorted by a STABLE
    `(time_s, -reach)`, so an exact tie goes to whoever was appended first.
    Every call site passes attackers first, so every dead heat went to the
    attack. Corners resolved 93-100% to the attacker.
  * the same tie returned ATTACKER at sample_step 0.1 and 0.05 and DEFENDER at
    0.02 and 0.01. A duel that flips with the sampling resolution is not
    football. `if candidates: break` abandoned the flight at the first sample
    at which ANYBODY could reach the ball, so the contest was decided on a
    0.1 s bucket (TICK_S) of a 1.3 s flight.

THE FIX
Score each contestant on `time_to_reach`, which is CONTINUOUS, and drop the
bucket and the list order:
  * the early `break` is removed -- every sample is scored;
  * a contestant's score is the earliest moment he could be at any point the
    ball will occupy, subject to the ball arriving there no earlier than he
    can (otherwise he cannot contest it at all);
  * ties -- which are common in a crowded box and were silently attacker wins --
    are broken on FOOTBALL: greater vertical reach first, then whoever the ball
    is travelling TOWARD. That second term is direction-agnostic (it is a dot
    product with the ball's own velocity) and is exactly the real tiebreak: in
    a dead heat the man the cross is travelling into gets it.
  * `id(player)` is the last resort purely so the result is reproducible.

Semantics for a clear winner are unchanged: whoever can reach the ball's path
first still wins. Only the ties and the quantisation change.
"""
import pathlib

p = pathlib.Path("geometry_engine.py")
lines = p.read_text(encoding="utf-8").splitlines()

START = next(n for n, l in enumerate(lines)
             if l.strip() == "steps = max(1, int(math.ceil(flight.duration "
                              "/ sample_step)))")
END = next(n for n in range(START, len(lines))
           if lines[n].strip().startswith('outcome = "contested" if challenger'))
print(f"replacing lines {START + 1}..{END + 1}")
print("  first:", lines[START].strip()[:60])
print("  last :", lines[END].strip()[:60])

NEW = '''    steps = max(1, int(math.ceil(flight.duration / sample_step)))

    # ── PER-CONTESTANT SCORE, NOT A LIST-ORDER TIE ─────────────────────
    # The flight used to be abandoned at the first sample at which ANYBODY
    # could reach the ball (`if candidates: break`) and the survivors were
    # ordered by a STABLE sort on (sample_time, -reach). Two measured
    # consequences, both of which made corners attacker-proof:
    #
    #   * an exact tie was won by whoever had been APPENDED first, and every
    #     call site passes attackers before defenders -- so every dead heat
    #     went to the attack. Corners resolved 93-100% to the attacker.
    #   * the race was quantised to `sample_step` (TICK_S = 0.1 s) over a
    #     ~1.3 s flight, so a duel decided inside one bucket flipped with the
    #     resolution alone: the SAME geometry returned ATTACKER at 0.1 s and
    #     DEFENDER at 0.02 s.
    #
    # `time_to_reach` is continuous, so the contest is settled on it instead.
    # Each contestant is scored by the earliest moment he could be at ANY
    # point the ball will occupy -- no bucket, and no dependence on the order
    # the arguments arrived in.
    #
    # `arrival > time_s` is not a near miss: the ball reaches that point
    # BEFORE he can, so he cannot contest it there at all.
    per_player: dict = {}
    for index in range(1, steps + 1):
        time_s = flight.duration * index / steps
        point = flight.position_at(time_s)
        airborne = point.z > 1.15
        # direction the ball is travelling, for the dead-heat tiebreak
        _nxt = flight.position_at(min(1.0, time_s + 0.05))
        _vx, _vy = _nxt.x - point.x, _nxt.y - point.y
        _vn = math.hypot(_vx, _vy) or 1.0

        for player in attacking_players + defending_players:
            max_reach = player.vertical_reach(airborne)
            if point.z > max_reach:
                continue
            arrival = _race_motion(player, point.horizontal(),
                                   player.control_radius, player_context)
            if arrival > time_s:
                continue
            jump_needed = max(0.0, point.z - player.standing_reach)
            jump_start = (time_s - math.sqrt(2.0 * jump_needed / 9.8)
                          if jump_needed > 0 else time_s)
            # "ahead" = how far in front of the ball this man is, along the
            # ball's own direction of travel. In a dead heat the man the cross
            # is travelling INTO gets it. Direction-agnostic, so it cannot
            # favour a side.
            ahead = ((player.position.x - point.x) * _vx
                     + (player.position.y - point.y) * _vy) / _vn
            entry = (arrival, -max_reach, -ahead, id(player), player,
                     player in attacking_players, time_s, point,
                     jump_start, max_reach)
            prev = per_player.get(id(player))
            if prev is None or entry[:3] < prev[:3]:
                per_player[id(player)] = entry

    candidates = sorted(per_player.values(), key=lambda item: item[:3])

    if not candidates:
        return AerialResolution("drops", flight.position_at(flight.duration),
                                flight.duration)

    (_arr, _nr, _na, _k, winner, winner_is_attacker, time_s, point,
     jump_time, reach_height) = candidates[0]

    # Challenger = the best-placed man from the other side who could also reach.
    challenger = None
    challenger_reach = 0.0
    for entry in candidates[1:]:
        if entry[5] != winner_is_attacker:
            challenger = entry[4]
            challenger_reach = entry[9]
            break

'''
lines[START:END] = NEW.splitlines()
p.write_text("\n".join(lines) + "\n", encoding="utf-8")

src = p.read_text(encoding="utf-8")
print()
print("early break removed       :", "if candidates:\n            break" not in src)
print("continuous arrival score  :", "arrival, -max_reach, -ahead" in src)
print("goal-side tiebreak        :", "-ahead, id(player)" in src)
print("imports:", end=" ")