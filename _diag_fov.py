"""HOW MUCH OF THE PITCH DOES EACH ROLE ACTUALLY SEE?

The user is right that this is one of the project's best features: **no player
sees everything.**  This probe measures it, because a feature nobody can
quantify is a feature nobody can protect.

THERE ARE TWO LAYERS, and which one is live depends on the runner:

  * `perception.perceive` — the DNA-scaled FOV: a radius between
    `range_min 8.0` and `range_max 45.0` (scaled by the player's vision and
    by his role's profile), a **150-degree horizontal cone centred on the
    forward direction** — so an actor BEHIND the player is unseen — positional
    noise up to `noise_base 3.0`, and a per-role relevance top-k.  This is
    the path the scratch runner uses.
  * `cognition.mind.PlayerMind` — the cognition `VisionSystem`: a fixed 25 m
    radius and **at most 2 teammates and 2 opponents**.  This is the path the
    PRODUCTION runner uses (`auto_run_match.py` registers a mind per player),
    and it is the tighter of the two.

This wraps whichever path is live and records, per decision, how many of the
twenty outfield players on the pitch the player actually perceived, and how
many of the unseen ones were BEHIND him — the part that makes the FOV a
football constraint rather than a blur.

Usage:
    python _diag_fov.py [--seed 777] [--minds]
"""
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RECORDS = []


def _record(pos, seen_t, seen_d, total, behind):
    RECORDS.append({"pos": pos, "seen": seen_t + seen_d, "total": total,
                    "behind": behind})


def _count_behind(player, x, y, teammates, defenders, position_engine,
                   attacks_right, seen_ids):
    """How many of the UNSEEN actors were behind the perceiver."""
    behind = 0
    for other in list(teammates or []) + list(defenders or []):
        if id(other) in seen_ids:
            continue
        try:
            tp = position_engine.get_position(other.name)
        except Exception:
            continue
        if tp is None:
            continue
        dx = tp[0] - x
        forward = dx if attacks_right else -dx
        if forward < 0:
            behind += 1
    return behind


def _wrap_fov():
    """The scratch-runner path: brain_integration imported `perceive` by name,
    so the MODULE attribute is the wrong reference — the importer's is right."""
    import brain_integration as BI
    orig = BI.perceive

    def patched(player, x, y, teammates, defenders, position_engine,
                under_pressure, attacks_right, game_state, minute=45.0,
                team_possession=True, score_diff=0, config=None,
                return_scene=False):
        vec, scene = orig(player, x, y, teammates, defenders, position_engine,
                          under_pressure, attacks_right, game_state, minute,
                          team_possession, score_diff, config,
                          return_scene=True)
        try:
            seen = list(scene.get("teammates") or []) + \
                list(scene.get("defenders") or [])
            _record(getattr(player, "position", ""),
                     len(scene.get("teammates") or []),
                     len(scene.get("defenders") or []),
                     len(teammates or []) + len(defenders or []),
                     _count_behind(player, x, y, teammates, defenders,
                                   position_engine, attacks_right,
                                   {id(a) for a in seen}))
        except Exception:
            pass
        return vec, scene

    BI.perceive = patched
    return orig


def _wrap_cognition():
    """The production path: the mind's own gate."""
    from cognition.mind import PlayerMind
    orig = PlayerMind.perceive

    def patched(self, player, x, y, teammates, defenders, position_engine):
        vis_t, vis_d = orig(self, player, x, y, teammates, defenders,
                            position_engine)
        try:
            seen = list(vis_t or []) + list(vis_d or [])
            _record(getattr(player, "position", ""), len(vis_t or []),
                     len(vis_d or []),
                     len(teammates or []) + len(defenders or []),
                     _count_behind(player, x, y, teammates, defenders,
                                   position_engine, True,
                                   {id(a) for a in seen}))
        except Exception:
            pass
        return vis_t, vis_d

    PlayerMind.perceive = patched
    return orig


def _auto_minds():
    """Give every player a mind on first sight, the way the production
    runner does (`auto_run_match.py` registers one per player).  Without it
    the scratch runner takes the vanilla path and the cognition gate never
    fires."""
    from cognition import mind as M
    orig = M.get_mind

    def patched(name):
        m = orig(name)
        if m is None:
            m = M.new_mind(view_radius=25.0)
            M.register_mind(name, m)
        return m

    M.get_mind = patched
    return orig


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--minds", action="store_true",
                    help="register a mind for every player (the PRODUCTION "
                         "path: a fixed 25 m, at most 2 teammates and 2 "
                         "opponents)")
    args = ap.parse_args()

    orig_fov = _wrap_fov()
    orig_cog = _wrap_cognition()
    orig_get = _auto_minds() if args.minds else None
    try:
        import random
        import numpy as np
        import run_match
        random.seed(args.seed)
        np.random.seed(args.seed)
        run_match.run()
    except Exception as exc:
        print(f"match failed: {exc}")
    finally:
        import brain_integration as BI
        BI.perceive = orig_fov
        from cognition.mind import PlayerMind
        PlayerMind.perceive = orig_cog
        if orig_get is not None:
            from cognition import mind as M
            M.get_mind = orig_get

    by_pos = collections.defaultdict(list)
    for r in RECORDS:
        by_pos[r["pos"]].append(r)

    out = []
    def w(line=""):
        out.append(line)
        print(line)

    w(f"perception samples: {len(RECORDS)}   "
      f"(path: {'COGNITION VisionSystem' if args.minds else 'DNA-scaled FOV cone'})")
    w("")
    w(f"{'pos':<6} {'decisions':>9} {'seen':>6} {'of':>4} {'share':>7} "
      f"{'unseen behind':>14}")
    for pos in sorted(by_pos):
        rows = by_pos[pos]
        seen = st.mean(r["seen"] for r in rows)
        total = st.mean(r["total"] for r in rows)
        behind = st.mean(r["behind"] for r in rows)
        w(f"{pos:<6} {len(rows):>9} {seen:>6.1f} {total:>4.0f} "
          f"{100.0*seen/max(total,1):>6.1f}% {behind:>14.1f}")
    if RECORDS:
        s = st.mean(r["seen"] for r in RECORDS)
        t = st.mean(r["total"] for r in RECORDS)
        b = st.mean(r["behind"] for r in RECORDS)
        w("")
        w(f"OVERALL: a player perceives {s:.1f} of {t:.0f} outfield players "
          f"({100.0*s/max(t,1):.1f}%), and misses {b:.1f} of them because "
          f"they are BEHIND him.")

    with open("_diag_fov.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n[written to _diag_fov.txt]")


if __name__ == "__main__":
    main()
