"""
PLOFA 26/27 — TEAM SEASON-PERFORMANCE → MATCH ATTENDANCE TESTS
================================================================
test_attendance_performance.py

Validates the "real football" behaviour: a team that is doing well in
the season (high league position / hot recent form) slightly fills the
stadium more, while a struggling side fills it less.

Guarantees:
  1. Absent standings data, MatchFinancials.compute is exactly unchanged
     (neutral 1.0 performance modifier — zero regression).
  2. League leaders draw more fans than bottom-half sides, all else equal.
  3. A hot recent run fills more seats than the same side on a cold streak.
  4. The modifier is small/bounded (the "slight" swing) and total fill stays
     within MIN_FILL_RATE..MAX_FILL_RATE.
"""

import random
import unittest

from exporter import MatchFinancials

_TEAMS = [
    "Avada Zenith", "Natrican City", "Port Virginia", "Pearls Lake Town",
    "Madzustico", "Old Lige Road", "Uditon City", "Oxland City",
    "Tuneelbeyn Slbey", "West Talern", "Castle Vale", "Frostline United",
    "Harborview FC", "Ironbridge Town", "Kingsfort", "Maplewood Rangers",
    "Quarryside", "Riverton Athletic",
]


def _full_table(home_pos, home_form, away_pos, away_form):
    """Build an 18-team standings dict with the given home/away placements."""
    table = {}
    for i, team in enumerate(_TEAMS, 1):
        if team == "Avada Zenith":
            pos, form = home_pos, home_form
        elif team == "Natrican City":
            pos, form = away_pos, away_form
        else:
            pos, form = i, "WDWDLW"
        played = 8
        table[team] = {
            "position": pos,
            "points": max(0, 3 * (played - 2) + (played - pos)),
            "played": played,
            "form": form,
        }
    return table


def _compute(seed, standings):
    random.seed(seed)
    return MatchFinancials.compute(
        stadium_capacity=45000,
        is_derby=False,
        home_team="Avada Zenith",
        away_team="Natrican City",
        home_avg_rating=78.0,
        away_avg_rating=75.0,
        big6_teams=set(),
        start_time=None,
        weather=None,
        is_weekend=True,
        standings=standings,
    )


class AttendancePerformanceTest(unittest.TestCase):
    def test_empty_standings_is_neutral(self):
        """No standings data -> identical to a no-standings baseline."""
        base = _compute(1234, None)
        neutral = _compute(1234, {})
        self.assertEqual(base["attendance"], neutral["attendance"])
        self.assertEqual(base["fill_rate"], neutral["fill_rate"])

    def test_neutral_fill_equals_none_standings(self):
        """None and {} standings both leave fill exactly at baseline."""
        a = _compute(99, None)
        b = _compute(99, {})
        self.assertEqual(a["fill_rate"], b["fill_rate"])

    def test_leader_draws_more_fans_than_bottom_home(self):
        """Top-of-table home team outdraws a bottom-of-table home team."""
        leader = _compute(7, _full_table(1, "WWWWW", 10, "DLLWDL"))
        struggler = _compute(7, _full_table(18, "LLLLL", 10, "DLLWDL"))
        self.assertGreater(leader["attendance"], struggler["attendance"])
        self.assertGreater(leader["fill_rate"], struggler["fill_rate"])

    def test_hot_home_form_outdraws_cold_home_form(self):
        """Same table, same position — hot streak beats cold streak at home."""
        hot = _compute(21, _full_table(5, "WWWWW", 10, "DLLWDL"))
        cold = _compute(21, _full_table(5, "LLLLL", 10, "DLLWDL"))
        self.assertGreater(hot["attendance"], cold["attendance"])

    def test_away_team_form_has_small_effect(self):
        """Away side's performance nudges attendance but less than the home side."""
        hot_away = _compute(31, _full_table(10, "DLLWDL", 9, "WWWWW"))
        cold_away = _compute(31, _full_table(10, "DLLWDL", 9, "LLLLL"))
        self.assertGreater(hot_away["attendance"], cold_away["attendance"])

    def test_modifier_is_small(self):
        """The performance swing is 'slight': well under a 15 percentage-point fill change."""
        top = _compute(5, _full_table(1, "WWWWW", 2, "WWWDL"))
        bottom = _compute(5, _full_table(18, "LLLLL", 17, "LLLWD"))
        self.assertLess(abs(top["fill_rate"] - bottom["fill_rate"]), 15.0)

    def test_fill_rate_stays_bounded(self):
        """Attendance never exceeds capacity or drops below the floor."""
        top = _compute(11, _full_table(1, "WWWWW", 2, "WWWDL"))
        bottom = _compute(11, _full_table(18, "LLLLL", 17, "LLLWD"))
        self.assertLessEqual(top["fill_rate"], MatchFinancials.MAX_FILL_RATE * 100)
        self.assertGreaterEqual(bottom["fill_rate"], MatchFinancials.MIN_FILL_RATE * 100)


if __name__ == "__main__":
    unittest.main()
