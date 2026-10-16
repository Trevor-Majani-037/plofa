"""
NEGATIVE CONTROL FOR tests/test_sub_quality.py
===============================================

WHY THIS EXISTS
---------------
Every guard in this project has, at some point, passed for the wrong
reason: the AST guard that walked one file when the omission was in
another; the jump guard inserted after the assignment it was meant to
precede, so its elapsed time was always zero; the "every multiplier is
wired" test that flagged two working mechanisms as dead; the mutation
control that rejected four mutants whose anchors were provably present
because it defaulted them to the wrong file.

A test that cannot fail is not a test, and the only way to find out is to
run it against code that is deliberately broken.

WHAT IT DOES
------------
Applies one mutation at a time to the real source files, runs the tests
that are supposed to notice, and asserts they go RED. Then restores every
file byte-exactly from a pre-control copy and asserts the restore.

HOW TO READ THE OUTPUT
----------------------
`CAUGHT`   the named tests failed, i.e. the guard works
`MISSED`   the tests still passed — that guard is decoration and must be
           fixed or the mutation removed from this list
`ANCHOR`   the mutation string was not found; the guard is VACUOUS and the
           real problem is that the source no longer matches what the
           tests assume

Run:  .venv\\Scripts\\python.exe _diag_sub_guard.py
"""

import ast
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

SM = "squad_manager.py"
ARM = "auto_run_match.py"

# (id, file, anchor string, replacement string, [pytest node ids])
# The anchor must exist or the mutation is a no-op that then reports a
# clean pass — the single most dangerous outcome this control can produce.
MUTATIONS = [
    (
        "M01", SM,
        "return sorted(pool, key=lambda p: -sub_quality(p, self))",
        "return sorted(pool, key=lambda p: sub_quality(p, self))",
        ["tests/test_sub_quality.py::TestBenchRanking::test_ranked_bench_is_quality_descending"],
    ),
    (
        "M02", SM,
        """        if not want_pos:
            return sorted(pool, key=lambda p: -sub_quality(p, self))""",
        """        if not want_pos:
            return list(pool)""",
        ["tests/test_sub_quality.py::TestBenchRanking::test_ranked_bench_is_quality_descending"],
    ),
    (
        # THE ORIGINAL DEFECT, restored verbatim: `_pick_sub` ignores the
        # ranking and takes the first unused bench player. An earlier
        # version of this mutant reversed the candidate list inside
        # `_apply_quality_rule`, which changed nothing — the scoreline rule
        # filters out the bad candidate either way, so the mutation was not
        # a defect at all, just a reordering of a set the rule then filters.
        "M03", SM,
        """            ranked = self._ranked_bench(bench, [pos_off] + ADJACENT_POS.get(pos_off, []))""",
        """            ranked = [p for p in bench if not getattr(p, "_used_as_sub", False)]""",
        ["tests/test_sub_quality.py::TestBenchRanking::test_pick_sub_prefers_the_better_of_two_same_position_subs"],
    ),
    (
        # Attacking budget ignored — the trailing half of the result.
        "M04", SM,
        """    if d_atk > budget_atk:""",
        """    if False:""",
        ["tests/test_sub_quality.py::TestScorelineRule::test_trailing_never_accepts_an_offensive_downgrade"],
    ),
    (
        # Defensive budget ignored — the Wharton headline result removed.
        "M05", SM,
        """    if d_def > budget_def:""",
        """    if False:""",
        ["tests/test_sub_quality.py::TestScorelineRule::test_leading_never_accepts_a_defensive_downgrade"],
    ),
    (
        # The bug I introduced and then fixed: ramp from zero at the
        # threshold, which silently moves 47' to 67'.
        "M06", SM,
        """    return SUB_REFERENCE_QUALITY_LOSS + SUB_QUALITY_TOLERANCE_PER_MIN * (
        minute - threshold_minute
    )""",
        """    return SUB_QUALITY_TOLERANCE_PER_MIN * max(0, minute - threshold_minute)""",
        ["tests/test_sub_quality.py::TestScorelineRule::test_leading_accepts_an_offensive_loss_only_after_47"],
    ),
    (
        # 'Never' encoded as the match length again.
        "M07", SM,
        "SUB_ACCEPT_DEFENSIVE_LOSS_LEADING_MIN  = SUB_NEVER   # never, per Wharton",
        "SUB_ACCEPT_DEFENSIVE_LOSS_LEADING_MIN  = 90",
        ["tests/test_sub_quality.py::TestScorelineRule::test_a_threshold_no_minute_can_reach_stays_at_zero"],
    ),
    (
        # Batch GATE side removed — `window_available` starts refusing a
        # second change at a stoppage already used.
        "M08", SM,
        """            return True                      # batch, free""",
        """            return False                     # batch, free""",
        ["tests/test_sub_quality.py::TestWindows::test_a_batch_at_one_stoppage_costs_one_window"],
    ),
    (
        # Batch ACCOUNTING side removed — `window_available` is a
        # different function from `_spend_window`, so M08 alone left the
        # window tally untested.
        "M20", SM,
        """        if self._last_window_minute.get(team, -99) == minute:
            return False                       # already opened this stoppage""",
        """        if False:
            return False                       # already opened this stoppage""",
        ["tests/test_sub_quality.py::TestWindows::test_a_batch_at_one_stoppage_costs_one_window"],
    ),
    (
        # Half-time window starts counting.
        "M09", SM,
        """        if minute == self.HALF_TIME_MINUTE:
            return False                       # free, untracked""",
        """        if False:
            return False                       # free, untracked""",
        ["tests/test_sub_quality.py::TestWindows::test_half_time_is_a_free_window"],
    ),
    (
        # Window count shared between teams.
        "M10", SM,
        """        self.windows_used[team] = self.windows_used.get(team, 0) + 1""",
        """        self.windows_used[self.home_team] = self.windows_used.get(self.home_team, 0) + 1""",
        ["tests/test_sub_quality.py::TestWindows::test_windows_are_counted_per_team"],
    ),
    (
        # Substitute arrives at a full tank again.
        "M11", SM,
        """        recovered = base + SUB_SEATED_RECOVERY_PER_MIN * float(minute)
        return max(0.0, min(100.0, recovered))""",
        """        return 100.0""",
        ["tests/test_sub_quality.py::TestSubEntryStamina::test_a_sub_does_not_arrive_at_one_hundred"],
    ),
    (
        # Soul term switched off — the permanently-dead-mechanism trap.
        "M12", SM,
        "        + 0.15 * (min(100.0, 100.0 * soul))",
        "        + 0.15 * 0.0",
        ["tests/test_sub_quality.py::TestQualitySignal::test_a_soul_scores_above_an_ordinary_player_of_equal_rating"],
    ),
    (
        # Missing attribute read as zero instead of the pool mean.
        "M13", SM,
        """    except (TypeError, ValueError):
        return default""",
        """    except (TypeError, ValueError):
        return 0.0""",
        ["tests/test_sub_quality.py::TestQualitySignal::test_a_non_numeric_attribute_degrades_to_neutral_not_zero"],
    ),
    (
        # Unreadable position becomes "" — matches nothing, bench undrawable.
        "M14", SM,
        """    return "CM"


# Adjacent positions""",
        """    return ""


# Adjacent positions""",
        ["tests/test_sub_quality.py::TestBenchRanking::test_a_player_whose_position_is_unreadable_is_still_selectable"],
    ),
    (
        # Cross-team contamination restored: test the accumulated list.
        "M15", SM,
        """            if not team_subs:
                # ── 2. TACTICAL PRE-PLANNED SUBS ─────────────""",
        """            if not subs_to_make:
                # ── 2. TACTICAL PRE-PLANNED SUBS ─────────────""",
        ["tests/test_sub_quality.py::TestDecisionLoop::test_one_teams_substitution_does_not_block_the_other"],
    ),
    (
        # Position normalisation dropped — "CB1" no longer matches "CB".
        "M16", SM,
        """            base = pos.rstrip("0123456789").upper()
            return base or pos.upper()""",
        """            return pos.upper()""",
        ["tests/test_sub_quality.py::TestBenchRanking::test_a_numbered_position_resolves_to_its_role_family"],
    ),
    (
        # The production runner overrides the limit back to three.
        "M17", ARM,
        "MAX_SUBS  = 5",
        "MAX_SUBS  = 3",
        ["tests/test_sub_quality.py::TestWindows::test_the_production_runner_does_not_override_it_back_to_three"],
    ),
    (
        # The season recovery stops reaching the engine.
        "M18", ARM,
        "            player.dna._starting_stamina = starting",
        "            pass",
        ["tests/test_sub_quality.py::TestStartingStaminaIsPublished::test_the_season_recovery_actually_reaches_the_engine"],
    ),
    (
        # The doubled drain comes back: a third process_action on the primary
        # actor, which is exactly the shape of the defect that was removed.
        "M19", "match_engine.py",
        """                    if event.secondary_player:
                        self.sub_controller.process_action(
                            event.secondary_player, action_key, event.team, minute,
                            drain_mult=press_tax, is_secondary=True,
                        )""",
        """                    if event.secondary_player:
                        self.sub_controller.process_action(
                            event.secondary_player, action_key, event.team, minute,
                            drain_mult=press_tax, is_secondary=True,
                        )
                    if event.player:
                        self.sub_controller.process_action(
                            event.player, action_key, event.team, minute,
                            drain_mult=press_tax,
                        )""",
        ["tests/test_sub_quality.py::TestStaminaIsDrainedOnce::test_absorb_chain_has_exactly_two_drain_calls"],
    ),
(
        # The `is_batched` ordering bug: computed AFTER `_spend_window` has
        # already written this minute, so every change looked batched.
        "M21", SM,
        """        was_batched = (minute == self._last_window_minute.get(team, -99)
                       and minute != self.HALF_TIME_MINUTE)""",
        """        was_batched = True""",
        ["tests/test_sub_quality.py::TestWindows::test_a_change_reports_the_batch_flag_as_it_was_at_the_time"],
    ),
    (
        # The window tally no longer saturates, so a match can report five
        # windows against a limit of three.
        "M22", SM,
        """        if self.windows_used.get(team, 0) >= self.MAX_WINDOWS:
            self.windows_exceeded[team] = self.windows_exceeded.get(team, 0) + 1
            self._last_window_minute[team] = minute
            return False""",
        """        if False:
            return False""",
        ["tests/test_sub_quality.py::TestWindows::test_the_window_tally_saturates_and_flags_the_excess"],
    ),
    (
        # A rotated-in player's penalty silently disappears. Anchored on the
        # READ of ROTATION_MAX_PENALTY, not on a fragment that also appears
        # in the audit.
        "M23", "roster_loader.py",
        "ROTATION_MAX_PENALTY * rot",
        "0.0 * rot",
        ["tests/test_sub_quality.py::TestRotation::test_rotation_actually_demotes_the_most_expensive_man"],
    ),
    (
        # The load never leaves the season state. The anchor must carry its
        # indentation, because `**load,` also appears inside the
        # `standing()` closure - and a match count of 2 against a mutation
        # that changes both would silently only change one.
        "M25", ARM,
        """                **load,
            )
            continue""",
        """            )
            continue""",
        ["tests/test_sub_quality.py::TestRotation::test_the_season_load_reaches_every_availability_site"],
    ),
    (
        # The audit is wired to nothing.
        "M26", "roster_loader.py",
        """    if not load_map:
        return""",
        """    if True:
        return""",
        ["tests/test_sub_quality.py::TestRotation::test_the_audit_names_the_man_it_promoted"],
    ),
    (
        # Turnaround awareness gone. The previous version of this mutation
        # was a NO-OP (old == new), which is worse than a weak mutation: it
        # reports a clean PASS for doing nothing.
        "M27", ARM,
        """            out["days_since_last_match"] = (
                match_date - date.fromisoformat(str(last))
            ).days""",
        """            out["days_since_last_match"] = 0""",
        ["tests/test_sub_quality.py::TestRotation::test_days_since_last_match_is_computed_from_the_calendar"],
    ),
    (
        # The LEVEL branch removed again.
        "M28", SM,
        """                else:
                    # LEVEL ON THE SCORE - the branch that did not exist.""",
        """                elif False:
                    # LEVEL ON THE SCORE - the branch that did not exist.""",
        ["tests/test_sub_quality.py::TestLevelScoreline::test_a_level_team_can_still_change_players"],
    ),
    (
        # The considered trigger becomes a coin flip again.
        "M29", SM,
        """        self.game_state_stats["considered"] += 1
        worth = score >= GAME_STATE_THRESHOLD""",
        """        worth = random.random() < 0.08""",
        ["tests/test_sub_quality.py::TestConsideredTrigger::test_no_random_draw_remains_in_the_game_state_trigger"],
    ),
    (
        # Pressure ordering flattened - a leading team is as desperate as a
        # chasing one.
        "M30", SM,
        """    if goal_diff < 0:
        # Behind. Full pull, and faster than level.
        return min(1.0, 0.65 + 0.35 * ramp)""",
        """    if goal_diff < 0:
        # Behind. Full pull, and faster than level.
        return 0.30 + 0.25 * ramp""",
        ["tests/test_sub_quality.py::TestConsideredTrigger::test_pressure_orders_trailing_above_level_above_leading"],
    ),

]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    py = str(REPO / ".venv" / "Scripts" / "python.exe")
    backups = {}
    originals = {}

    for _id, fname, *_ in MUTATIONS:
        p = REPO / fname
        if fname not in originals:
            originals[fname] = sha(p)
            backups[fname] = p.with_suffix(p.suffix + ".subguard.bak")
            shutil.copy2(p, backups[fname])

    print("=" * 78)
    print("NEGATIVE CONTROL — tests/test_sub_quality.py")
    print("=" * 78)
    print(f"  {len(MUTATIONS)} mutations. Each must turn a test RED.\n")

    results = []
    for mid, fname, anchor, replacement, nodes in MUTATIONS:
        path = REPO / fname
        src = path.read_text(encoding="utf-8")

        if src.count(anchor) != 1:
            results.append((mid, "ANCHOR", f"{src.count(anchor)} matches for {anchor[:44]!r}"))
            continue

        mutated = src.replace(anchor, replacement, 1)
        path.write_text(mutated, encoding="utf-8")

        try:
            try:
                ast.parse(mutated)
            except SyntaxError as exc:
                results.append((mid, "ANCHOR", f"mutation does not compile: {exc}"))
                continue

            proc = subprocess.run(
                [py, "-X", "utf8", "-m", "pytest", *nodes,
                 "-q", "--no-header", "-x"],
                cwd=REPO, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=900,
            )
            if proc.returncode != 0:
                results.append((mid, "CAUGHT", "tests went red as required"))
            else:
                results.append((mid, "MISSED", f"tests still passed: {' '.join(nodes)[:70]}"))
        finally:
            path.write_text(src, encoding="utf-8")

    caught = sum(1 for r in results if r[1] == "CAUGHT")
    missed = sum(1 for r in results if r[1] == "MISSED")
    anchor = sum(1 for r in results if r[1] == "ANCHOR")

    for mid, status, detail in results:
        mark = {"CAUGHT": "ok  ", "MISSED": "FAIL", "ANCHOR": "FAIL"}[status]
        print(f"  {mark} {mid}  {detail}")

    print()
    print(f"  caught {caught}/{len(MUTATIONS)}   missed {missed}   "
          f"anchor/vacuous {anchor}")

    ok_restore = True
    print("\n  byte-exact restore:")
    for fname, digest in originals.items():
        now = sha(REPO / fname)
        same = now == digest
        ok_restore &= same
        print(f"    {fname:<22} {'IDENTICAL' if same else 'MISMATCH'}")
        (REPO / f"{fname}.subguard.bak").unlink(missing_ok=True)

    verdict = (missed == 0 and anchor == 0 and ok_restore and caught == len(MUTATIONS))
    print(f"\n  VERDICT: {'ALL MUTATIONS CAUGHT, TREE RESTORED' if verdict else 'PROBLEM'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())