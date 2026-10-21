"""THE ATTENDANCE MODEL — the gate.

WHAT THIS FILE EXISTS FOR
-------------------------
The user: *"the fill rate or fans are just hand tuned numbers, that's bad…
the fans should be scaled based on city populations or something realistic."*

The diagnosis was right and it was measurable.  The crowd was

    attendance = stadium_capacity × fill_rate

with `fill_rate` read from a HAND-TUNNED table keyed on a "match tier"
(derby 0.95, big6_vs_big6 0.97, big6_home 0.90, standard 0.78,
low_interest 0.62), plus a squad-quality term that was measurably INERT
(the squads' average DNA rating runs 67.7-74.2, and the old term moved the
fill rate by less than two percentage points across that whole spread), a
kickoff-time term, a weather term, a standings term, and a random residual
of ±4-6% of the CAPACITY.

Measured over the 26/27 season: mean occupancy **85.0%**, seven clubs
pinned at the 98.5% ceiling on every one of their home matches.  The real
Premier League 24/25: **96.8% mean occupancy**, 40,474 per match, and
sixteen of twenty clubs between 95.3% and 99.9%.

WHAT THIS FILE PINS
-------------------
That the benchmark is real and self-consistent; that the model is centred
on it; that the ground is the ceiling; that the old hand-tuned table is
GONE (structurally, not by comment); and — the non-vacuity case — that the
model really does distinguish a leading club from a bottom one.
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attendance_model as AM  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    """The benchmark is the model's foundation, so it is checked first."""

    def test_the_benchmark_table_is_internally_consistent(self):
        """Every row's occupancy must be its own attendance over capacity."""
        for club, _stadium, cap, att, occ in AM.PL_24_25:
            derived = 100.0 * att / cap
            self.assertLess(
                abs(derived - occ), 0.15,
                f"{club}: table says {occ}%, the numbers give {derived:.2f}%")

    def test_the_benchmarks_totals_are_consistent(self):
        """The league mean must be the total over the matches."""
        mean = AM.PL_TOTAL_SPECTATORS_24_25 / AM.PL_MATCHES_24_25
        self.assertLess(
            abs(mean - AM.PL_MEAN_ATTENDANCE_24_25), 2.0,
            f"the stated mean {AM.PL_MEAN_ATTENDANCE_24_25} is not "
            f"{AM.PL_TOTAL_SPECTATORS_24_25}/{AM.PL_MATCHES_24_25} = {mean:.1f}")

    def test_the_benchmark_mean_occupancy_is_the_mean_of_its_own_rows(self):
        """The headline 96.8% must be what the twenty rows average to."""
        mean_occ = sum(o for *_r, o in AM.PL_24_25) / len(AM.PL_24_25)
        self.assertLess(
            abs(mean_occ / 100.0 - AM.PL_MEAN_OCCUPANCY_24_25), 0.01,
            f"the rows average {mean_occ:.2f}%, the constant says "
            f"{AM.PL_MEAN_OCCUPANCY_24_25 * 100:.2f}%")

    def test_the_cities_do_not_split_the_crowd(self):
        """THE FACT THAT KILLS THE "scale by city population" IDEA.

        Seven of the twenty benchmark clubs are London clubs, and between
        them they drew 256,000 a matchday from one city — every one of
        them between 95.4% and 99.9% full.  A shared city does not split a
        crowd, so a per-city population would add a hand-tuned number
        without adding realism.
        """
        london = ("Arsenal FC", "Tottenham Hotspur", "Chelsea FC",
                  "West Ham United", "Fulham FC", "Brentford FC",
                  "Crystal Palace")
        occ = {c: o for c, _s, _cap, _a, o in AM.PL_24_25 if c in london}
        self.assertEqual(len(occ), 7,
                         f"the benchmark has {len(occ)} London clubs, not 7")
        for club, o in occ.items():
            self.assertGreaterEqual(o, 95.4, f"{club} at {o}% breaks it")
        total = sum(a for c, _s, _cap, a, _o in AM.PL_24_25 if c in london)
        self.assertGreater(total, 250_000,
                           f"the London clubs drew {total:,} a matchday")


class ModelShapeTests(unittest.TestCase):
    def test_the_ground_is_the_ceiling(self):
        """No club in the benchmark, or in the model, draws more than the
        ground holds — with one named, real exception: Villa Park's
        published capacity excludes standing areas."""
        for club, _s, cap, att, _o in AM.PL_24_25:
            if club in AM.CAPACITY_ARTEFACT_CLUBS:
                continue
            self.assertLessEqual(att, cap, f"{club} drew {att:,} into {cap:,}")
        for cap in (8_000, 45_000, 87_000, 100_000):
            for score in (0.0, 0.25, 0.5, 0.75, 1.0):
                out = AM.attendance_for(capacity=cap, home_score=score,
                                        away_score=score)
                self.assertLessEqual(out["attendance"], cap)

    def test_the_capacity_artefact_is_named_not_hidden(self):
        """The one row that breaks the ceiling must be declared, so that
        anything asserting the invariant excludes it DELIBERATELY."""
        breakers = [c for c, _s, cap, att, _o in AM.PL_24_25 if att > cap]
        self.assertEqual(sorted(breakers),
                         sorted(AM.CAPACITY_ARTEFACT_CLUBS),
                         "a club draws more than its stated capacity and is "
                         "not declared as an artefact")

    def test_the_model_is_centred_on_the_benchmark(self):
        """A mid-strength club fills its ground at about the benchmark's
        league mean."""
        out = AM.attendance_for(capacity=50_000, home_score=0.5,
                                away_score=0.5)
        self.assertLess(
            abs(out["fill_rate"] / 100.0 - AM.PL_MEAN_OCCUPANCY_24_25), 0.02,
            f"the neutral crowd is at {out['fill_rate']}%, the benchmark "
            f"mean is {AM.PL_MEAN_OCCUPANCY_24_25 * 100:.1f}%")

    def test_a_leader_draws_more_than_a_bottom_club(self):
        """NON-VACUITY: the model must actually discriminate."""
        top = AM.attendance_for(capacity=50_000, home_score=1.0,
                                away_score=0.5)
        bottom = AM.attendance_for(capacity=50_000, home_score=0.0,
                                   away_score=0.5)
        self.assertGreater(top["attendance"], bottom["attendance"])
        self.assertGreater(top["fill_rate"], bottom["fill_rate"])

    def test_the_spread_matches_the_benchmarks_typical_band(self):
        """The model's own spread must sit inside the benchmark's band for
        the sixteen clubs that are not structural exceptions."""
        low = AM.base_occupancy(0.0)
        high = AM.base_occupancy(1.0)
        self.assertGreaterEqual(
            low, AM.PL_TYPICAL_OCCUPANCY_LOW - 0.02,
            f"the weakest club sits at {low:.3f}, below the benchmark's "
            f"typical floor {AM.PL_TYPICAL_OCCUPANCY_LOW}")
        self.assertLessEqual(
            high, AM.PL_TYPICAL_OCCUPANCY_HIGH,
            f"the strongest club sits at {high:.3f}, above the benchmark's "
            f"ceiling {AM.PL_TYPICAL_OCCUPANCY_HIGH}")

    def test_the_model_never_pins_at_its_ceiling(self):
        """The old model pinned seven clubs at exactly 98.5% on every home
        match.  A model that can reach its own ceiling can do that again,
        so the ceiling must be out of reach."""
        for score in (0.0, 0.25, 0.5, 0.75, 1.0):
            occ = AM.base_occupancy(score)
            self.assertLess(occ, AM.OCCUPANCY_CEILING,
                            f"score {score} reaches the ceiling")

    def test_the_squad_rating_is_a_weak_signal_and_says_so(self):
        """The measured spread of the squads' average DNA rating is
        67.7-74.2.  The market score must reflect that the rating barely
        moves — if it dominated, the model would be a rating model."""
        weak = AM.market_score(67.7, 0.0)
        strong = AM.market_score(74.2, 0.0)
        self.assertLess(
            abs(weak - strong), 0.20,
            "the squad rating moves the market score by "
            f"{abs(weak - strong):.3f} — the measured rating spread is only "
            "6.5 points, so it must be a weak signal")
        lo = AM.market_score(70.0, -1.0)
        hi = AM.market_score(70.0, 1.0)
        self.assertGreater((hi - lo), 2 * (strong - weak))


class TheOldTableIsGoneTests(unittest.TestCase):
    """STRUCTURAL, because a comment saying the table is gone proves
    nothing — and the house rule is that a name inside a comment is not a
    call site."""

    def _exporter_source(self):
        import exporter
        with open(exporter.__file__, encoding="utf-8", newline="") as f:
            return f.read()

    def test_the_hand_tuned_fill_rate_table_is_gone(self):
        """The old table was a dict literal of tier -> fill rate, inside
        `MatchFinancials.compute`.  No such table may exist any more."""
        src = self._exporter_source()
        tree = ast.parse(src)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "MatchFinancials":
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (isinstance(t, ast.Name)
                                and t.id == "fill_rate_map"):
                            offenders.append(sub.lineno)
        self.assertFalse(
            offenders,
            f"exporter.py still builds a hand-tuned fill-rate table at "
            f"line(s) {offenders}")

    def test_the_old_tier_fill_rate_table_is_not_rebuilt_under_another_name(self):
        """The old table was a dict of tier -> fill rate, and the values
        were FRACTIONS (0.62-0.97).  It must not reappear under a different
        variable name.

        The first version of this guard banned the six NUMBERS (0.95, 0.97,
        0.90, 0.85, 0.78, 0.62) as literals inside the class — and it
        fired on `random.uniform(0.90, 1.10)`, which is the TICKET PRICE
        variation and has nothing to do with the crowd.  A guard that fires
        on unrelated code gets switched off, and then it fires on nothing.

        The second version banned any mapping keyed by the tier names — and
        it fired on `TIER_TICKET_PRICES`, which is the legitimate PRICING
        table (the same tiers, but the values are currency, not fractions).
        So the guard checks BOTH: the tier names AND that the values are
        occupancy fractions."""
        src = self._exporter_source()
        tree = ast.parse(src)
        tier_names = {"derby", "big6_vs_big6", "big6_home", "big6_away",
                      "standard", "low_interest"}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "MatchFinancials":
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Dict):
                    continue
                keys, vals = set(), []
                for k, v in zip(sub.keys, sub.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
                    if isinstance(v, ast.Constant) and isinstance(v.value, float):
                        vals.append(v.value)
                if not (tier_names & keys):
                    continue
                if vals and all(0.5 <= x <= 1.0 for x in vals):
                    offenders.append(sub.lineno)
        self.assertFalse(
            offenders,
            f"a tier->fill-rate mapping still exists inside MatchFinancials "
            f"at line(s) {offenders}")

    def test_the_model_module_is_the_one_the_exporter_uses(self):
        """The exporter must call the sourced model, not roll its own."""
        src = self._exporter_source()
        self.assertIn("attendance_model.attendance_for", src,
                      "the exporter does not call attendance_model.attendance_for")
        self.assertIn("attendance_model.market_score", src,
                      "the exporter does not call attendance_model.market_score")


class WiringTests(unittest.TestCase):
    def test_the_exporter_uses_the_model(self):
        from exporter import MatchFinancials
        import random
        random.seed(7)
        out = MatchFinancials.compute(
            stadium_capacity=60_000, is_derby=False,
            home_team="Home", away_team="Away",
            home_avg_rating=72.0, away_avg_rating=70.0,
            big6_teams=set(), start_time=None, weather=None,
            is_weekend=True, standings=None,
        )
        self.assertGreater(out["attendance"], 0)
        self.assertLessEqual(out["attendance"], 60_000)
        self.assertGreaterEqual(out["fill_rate"],
                                MatchFinancials.MIN_FILL_RATE * 100)
        self.assertLessEqual(out["fill_rate"],
                             MatchFinancials.MAX_FILL_RATE * 100)

    def test_the_absence_of_standings_is_exactly_neutral(self):
        """Zero regression: with no standings the crowd must be the same
        as with an empty table.  Both calls are seeded — the model has a
        small random residual, so an unseeded pair would differ by that
        and the test would be measuring the residual, not the standings."""
        from exporter import MatchFinancials
        import random
        random.seed(11)
        a = MatchFinancials.compute(
            stadium_capacity=45_000, is_derby=False, home_team="H",
            away_team="A", home_avg_rating=70.0, away_avg_rating=70.0,
            big6_teams=set(), start_time=None, weather=None,
            is_weekend=True, standings=None)
        random.seed(11)
        b = MatchFinancials.compute(
            stadium_capacity=45_000, is_derby=False, home_team="H",
            away_team="A", home_avg_rating=70.0, away_avg_rating=70.0,
            big6_teams=set(), start_time=None, weather=None,
            is_weekend=True, standings={})
        self.assertEqual(a["attendance"], b["attendance"])
        self.assertEqual(a["fill_rate"], b["fill_rate"])


if __name__ == "__main__":
    unittest.main()
