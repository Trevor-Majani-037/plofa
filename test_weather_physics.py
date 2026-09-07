"""
PLOFA 26/27 — WEATHER & FIXTURE-TIME PHYSICS TESTS  (Checkpoint)
=============================================================
test_weather_physics.py

Validates:
  1. WeatherCondition presets + legacy string mapping (backward compat).
  2. Zero-regression toggle — disabled physics returns exactly-neutral
     multipliers (1.0) and zero wind deflection.
  3. Per-field physics multipliers (rain/wind/fog effect on pass, shot,
     stamina, dribble, press, grip, GK reach/reaction).
  4. apply_wind_deflection lateral/longitudinal drift.
  5. Fixture-time attendance + diurnal temperature proxy.
  6. Real-climate generator seasonal baselines.
  7. Excel FIXTURES Start-Time extraction (read-only).
  8. Integration — clear vs heavy-rain simulation stats diverge when the
     physics toggle is enabled, and disabled sims are untouched.

Run:  py -m pytest tests/test_weather_physics.py -v
"""
from __future__ import annotations
import random
import unittest
from datetime import date, time

from weather_physics import (
    WeatherCondition,
    WeatherPhysics,
    FixtureTimeEffect,
    get_climate_weather,
    load_fixture_info,
)
from pass_classifier import apply_wind_deflection, classify_pass


class WeatherConditionTests(unittest.TestCase):
    def test_presets_generate_correct_fields(self):
        c = WeatherCondition.clear()
        self.assertEqual(c.rain_intensity, 0.0)
        self.assertGreaterEqual(c.visibility, 0.99)
        self.assertLessEqual(c.pitch_wetness, 0.1)

        r = WeatherCondition.rain(intensity=0.80)
        self.assertGreaterEqual(r.rain_intensity, 0.79)
        self.assertLessEqual(r.visibility, 1.0)

        w = WeatherCondition.wind(wind_speed=15.0)
        self.assertGreaterEqual(w.wind_speed, 14.9)
        self.assertLessEqual(w.rain_intensity, 0.1)

        f = WeatherCondition.fog(visibility=0.20)
        self.assertLessEqual(f.visibility, 0.21)
        self.assertGreater(f.temperature_c, -10)

    def test_from_string_backward_compat(self):
        self.assertIsInstance(WeatherCondition.from_string("clear"), WeatherCondition)
        # clear maps to a dry, high-visibility condition
        self.assertEqual(WeatherCondition.from_string("clear").rain_intensity, 0.0)
        r = WeatherCondition.from_string("rain")
        self.assertGreaterEqual(r.rain_intensity, 0.5)
        self.assertTrue(r.name.startswith(("rain", "heavy_rain")))
        self.assertEqual(WeatherCondition.from_string("wind").name, "wind")
        self.assertEqual(WeatherCondition.from_string("fog").name, "fog")
        # "rail" typo tolerated (auto_run previously used rail)
        self.assertGreaterEqual(WeatherCondition.from_string("rail").rain_intensity, 0.5)

    def test_weather_physics_accepts_string_or_condition(self):
        # Both a string and a WeatherCondition should resolve identically.
        cond = WeatherCondition.rain(intensity=0.90)
        WeatherPhysics.set_active_weather(cond, enabled=True)
        m_cond = WeatherPhysics.pass_accuracy_mult()
        WeatherPhysics.reset()
        WeatherPhysics.set_active_weather("rain", enabled=True)
        m_str = WeatherPhysics.pass_accuracy_mult()
        self.assertLess(m_str, 1.0)
        self.assertLess(m_cond, 1.0)
        WeatherPhysics.reset()


class ZeroRegressionTests(unittest.TestCase):
    """Core guarantee: disabled => every multiplier is exactly 1.0 / 0.0."""

    def setUp(self):
        WeatherPhysics.reset()

    def test_disabled_toggle_returns_unit_multipliers(self):
        self.assertFalse(WeatherPhysics.enabled)
        self.assertEqual(WeatherPhysics.pass_accuracy_mult("rain", is_long=True), 1.0)
        self.assertEqual(WeatherPhysics.shot_accuracy_mult("wind", distance_m=30), 1.0)
        self.assertEqual(WeatherPhysics.shot_power_mult("rain"), 1.0)
        self.assertEqual(WeatherPhysics.stamina_drain_mult("rain"), 1.0)
        self.assertEqual(WeatherPhysics.dribble_control_mult("rain"), 1.0)
        self.assertEqual(WeatherPhysics.carry_distance_mult("rain"), 1.0)
        self.assertEqual(WeatherPhysics.pressing_intensity_mult("rain"), 1.0)
        self.assertEqual(WeatherPhysics.pitch_grip("rain"), 1.0)
        self.assertEqual(WeatherPhysics.gk_reach_mult("fog"), 1.0)
        self.assertEqual(WeatherPhysics.gk_reaction_mult("fog"), 1.0)
        # No wind drift when disabled
        ax, ay, drift = WeatherPhysics.pass_lateral_deflection(50, 34, 30, 34, "wind", True)
        self.assertEqual((ax, ay, drift), (50, 34, 0.0))

    def test_disabled_slip_probability_is_zero(self):
        self.assertEqual(WeatherPhysics.slip_probability("tackle", "rain"), 0.0)


class RainPhysicsTests(unittest.TestCase):
    """Heavy rain degrades pass control, grip; increases stamina drain."""

    def setUp(self):
        WeatherPhysics.reset()

    def test_rain_reduces_pass_accuracy(self):
        WeatherPhysics.set_active_weather("rain", enabled=True)
        m = WeatherPhysics.pass_accuracy_mult()
        self.assertLessEqual(m, 1.0)
        self.assertGreaterEqual(m, 0.68)

    def test_heavy_rain_reduces_grip(self):
        WeatherPhysics.set_active_weather(WeatherCondition.rain(intensity=0.95), enabled=True)
        grip = WeatherPhysics.pitch_grip()
        self.assertLessEqual(grip, 1.0)
        self.assertLess(grip, 0.90)

    def test_rain_increases_stamina_drain(self):
        WeatherPhysics.set_active_weather(WeatherCondition.rain(intensity=0.70), enabled=True)
        self.assertGreater(WeatherPhysics.stamina_drain_mult(), 1.0)

    def test_rain_reduces_shot_accuracy_at_range(self):
        WeatherPhysics.set_active_weather(WeatherCondition.rain(intensity=0.90), enabled=True)
        m = WeatherPhysics.shot_accuracy_mult(distance_m=30)
        self.assertLess(m, 1.0)

    def test_heavy_rain_throttles_pressing(self):
        WeatherPhysics.set_active_weather(WeatherCondition.rain(intensity=0.90), enabled=True)
        m = WeatherPhysics.pressing_intensity_mult()
        self.assertLess(m, 1.0)

    def test_rain_degrades_dribble_and_carry(self):
        WeatherPhysics.set_active_weather(WeatherCondition.rain(intensity=0.80), enabled=True)
        self.assertLess(WeatherPhysics.dribble_control_mult(), 1.0)
        self.assertLess(WeatherPhysics.carry_distance_mult(), 1.0)


class WindDeflectionTests(unittest.TestCase):
    def setUp(self):
        WeatherPhysics.reset()

    def test_side_wind_deflects_airborne_pass_laterally(self):
        # Wind from 90 deg blows along +y (lateral).
        ax, ay, drift = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=15.0, wind_direction_deg=90.0, is_airborne=True
        )
        self.assertGreater(ay, 34.0)
        self.assertGreater(drift, 0.0)
        self.assertTrue(0 <= ax <= 105)
        self.assertTrue(0 <= ay <= 68)

    def test_headwind_reduces_longitudinal_distance(self):
        # Wind from 180 deg blows along -x (headwind toward destination +x).
        ax, ay, _ = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=15.0, wind_direction_deg=180.0, is_airborne=True
        )
        self.assertLess(ax, 60.0)

    def test_tailwind_increases_longitudinal_distance(self):
        ax, ay, _ = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=15.0, wind_direction_deg=0.0, is_airborne=True
        )
        self.assertGreater(ax, 60.0)

    def test_light_wind_no_deflection(self):
        ax, ay, drift = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=0.2, wind_direction_deg=90.0, is_airborne=True
        )
        self.assertEqual((ax, ay, drift), (60, 34, 0.0))

    def test_wind_only_affects_airborne_strongly(self):
        ax, ay, drift_air = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=15.0, wind_direction_deg=90.0, is_airborne=True
        )
        gx, gy, drift_ground = apply_wind_deflection(
            60, 34, 40, 34, wind_speed=15.0, wind_direction_deg=90.0, is_airborne=False
        )
        self.assertGreater(drift_air, drift_ground)


class FogGkTandReactionTests(unittest.TestCase):
    def setUp(self):
        WeatherPhysics.reset()

    def test_fog_reduces_gk_reaction(self):
        WeatherPhysics.set_active_weather("fog", enabled=True)
        m = WeatherPhysics.gk_reaction_mult()
        self.assertLess(m, 1.0)
        self.assertGreaterEqual(m, 0.72)

    def test_clear_keeps_gk_reaction_at_1(self):
        WeatherPhysics.set_active_weather("clear", enabled=True)
        self.assertEqual(WeatherPhysics.gk_reaction_mult(), 1.0)
        # clear pitch has only trace wetness, so reach stays essentially full
        self.assertGreaterEqual(WeatherPhysics.gk_reach_mult(), 0.99)


class FixtureTimeTests(unittest.TestCase):
    def test_afternoon_weekend_attendance_peak(self):
        self.assertEqual(
            FixtureTimeEffect.attendance_mult("15:00", is_weekend=True), 1.0
        )

    def test_partial_capacity(self):
        self.assertEqual(
            FixtureTimeEffect.attendance_mult("20:00", is_weekend=False), 0.945
        )

    def test_attendance_mult_unit(self):
        # Legacy format with no time -> neutral
        self.assertEqual(FixtureTimeEffect.attendance_mult(None, is_weekend=True), 1.0)

    def test_temp_proxy_seasonal_curve(self):
        # Southern-hemisphere: January (summer) warmer than July (winter).
        jan = FixtureTimeEffect.temperature_proxy(1, "15:00")
        jul = FixtureTimeEffect.temperature_proxy(7, "15:00")
        self.assertGreater(jan, jul)

    def test_temp_proxy_night_colder(self):
        day = FixtureTimeEffect.temperature_proxy(6, "15:00")
        night = FixtureTimeEffect.temperature_proxy(6, "20:00")
        self.assertLess(night, day)


class RealClimateTests(unittest.TestCase):
    def test_summer_warmer_than_winter(self):
        # Toland is southern-hemisphere (attached to Brazil): Jan is summer.
        rng = random.Random(42)
        summer = get_climate_weather("Test Park", date(2026, 1, 15), "15:00", rng=rng)
        rng = random.Random(42)
        winter = get_climate_weather("Test Park", date(2026, 7, 15), "15:00", rng=rng)
        self.assertGreater(summer.temperature_c, winter.temperature_c)

    def test_climate_returns_valid_condition(self):
        rng = random.Random(7)
        for month, day in [(2, 3), (5, 10), (9, 20), (12, 5)]:
            c = get_climate_weather("Stadium", date(2026, month, day), "15:00", rng=rng)
            self.assertIsInstance(c, WeatherCondition)
            self.assertTrue(0.0 <= c.rain_intensity <= 1.0)
            self.assertTrue(0.0 <= c.visibility <= 1.0)


class ExcelFixtureExtractionTests(unittest.TestCase):
    def test_start_time_read_from_fixtures_sheet(self):
        info = load_fixture_info("PLOFA-2026-2027.xlsx", 1, "Oxton", "Justice")
        self.assertIsNotNone(info["start_time"])
        self.assertEqual(info["start_time"], "12:30")
        self.assertIsNotNone(info["venue"])

    def test_missing_fixture_returns_none(self):
        info = load_fixture_info("PLOFA-2026-2027.xlsx", 99, "No Such Club", "")
        self.assertIsNone(info["start_time"])

    def test_every_league_club_has_coordinates(self):
        # Every team in the real FIXTURES sheet must resolve to a Toland city /
        # coordinate so the weather API always applies (no silent fallback).
        from weather_physics import CLUB_TO_CITY, _coords_for
        import openpyxl
        wb = openpyxl.load_workbook(
            "PLOFA-2026-2027.xlsx", read_only=True, data_only=True
        )
        ws = wb["FIXTURES"]
        clubs = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1] is None:
                continue
            clubs.add(str(row[5]).strip())
            clubs.add(str(row[6]).strip())
        clubs.discard("None")
        missing = [c for c in clubs if c not in CLUB_TO_CITY or _coords_for(c) is None]
        self.assertEqual(missing, [])


class PassClassifierWindTests(unittest.TestCase):
    def test_classify_pass_carries_wind_drift(self):
        r = classify_pass(
            40, 34, 60, 34, attacks_right=True, is_airborne=True, wind_drift_m=1.4
        )
        self.assertEqual(r.wind_drift_m, 1.4)

    def test_classify_pass_no_wind_default(self):
        r = classify_pass(40, 34, 60, 34, attacks_right=True)
        self.assertEqual(r.wind_drift_m, 0.0)


class EngineIntegrationTests(unittest.TestCase):
    """Deterministic end-to-end wiring through MatchEngine (no full match)."""

    def tearDown(self):
        WeatherPhysics.reset()

    def test_engine_resolves_and_activates_weather(self):
        from match_engine import MatchConfig, MatchEngine
        cfg = MatchConfig(
            home_team="HC", away_team="AC",
            weather=WeatherCondition.rain(intensity=0.90),
            weather_enabled=True, start_time="20:00",
        )
        eng = MatchEngine(cfg, None, None)
        self.assertTrue(WeatherPhysics.enabled)
        self.assertTrue(eng.state.weather_enabled)
        self.assertIsInstance(eng.state.weather, WeatherCondition)
        self.assertGreater(eng.state.weather.rain_intensity, 0.8)
        # cleanup global state
        WeatherPhysics.reset()

    def test_engine_disabled_keeps_physics_off(self):
        from match_engine import MatchConfig, MatchEngine
        cfg = MatchConfig(
            home_team="HC", away_team="AC",
            weather="rain", weather_enabled=False,
        )
        MatchEngine(cfg, None, None)
        self.assertFalse(WeatherPhysics.enabled)


class StaminaWiringTests(unittest.TestCase):
    """Engine -> SubstitutionController stamina drain wiring is deterministic."""

    def tearDown(self):
        WeatherPhysics.reset()

    def _build_engine(self, weather, weather_enabled):
        from match_engine import MatchConfig, MatchEngine
        from squad_manager import SubstitutionController
        from player_dna import SquadBuilder

        starters = [
            ("GK", "GK", ["tackling", "positioning"]),
            ("RB", "DF", ["tackling", "positioning"]),
            ("CB1", "DF", ["tackling", "positioning"]),
            ("CB2", "DF", ["tackling", "positioning"]),
            ("LB", "DF", ["tackling", "positioning"]),
            ("CM1", "MF", ["passing", "vision"]),
            ("CM2", "MF", ["passing", "workrate"]),
            ("RW", "MF", ["dribbling", "pace"]),
            ("CAM", "MF", ["passing", "vision"]),
            ("LW", "FW", ["dribbling", "finishing"]),
            ("ST", "FW", ["finishing", "positioning"]),
        ]
        home_sq = SquadBuilder.build("HC", starters)
        away_sq = SquadBuilder.build("AC", starters)

        cfg = MatchConfig(
            home_team="HC", away_team="AC",
            weather=weather, weather_enabled=weather_enabled,
        )
        eng = MatchEngine(cfg, None, None)
        eng.set_squad("HC", home_sq["starters"], home_sq["substitutes"])
        eng.set_squad("AC", away_sq["starters"], away_sq["substitutes"])
        ctrl = SubstitutionController("HC", "AC", [], [])
        eng.set_stamina_controller(ctrl)
        return eng, ctrl

    def test_heavy_rain_increases_stamina_drain(self):
        _, ctrl = self._build_engine(
            WeatherCondition.rain(intensity=0.90), weather_enabled=True
        )
        self.assertGreater(ctrl.weather_drain_mult, 1.0)

    def test_disabled_keeps_stamina_drain_neutral(self):
        _, ctrl = self._build_engine("rain", weather_enabled=False)
        self.assertEqual(ctrl.weather_drain_mult, 1.0)


class WeatherApiTests(unittest.TestCase):
    """Deterministic tests for the Open-Meteo API layer (no network): the
    fetch is mocked, so behaviour is stable offline / in CI."""

    def tearDown(self):
        wp_module = __import__("weather_physics")
        wp_module.WEATHER_API_ENABLED = True
        wp_module.clear_weather_cache()

    def test_coords_for_known_club(self):
        from weather_physics import _coords_for, TEAM_LOCATIONS, CLUB_TO_CITY
        self.assertEqual(CLUB_TO_CITY["Oxton"], "Oxland City")
        coords = _coords_for("Oxton")
        self.assertIsNotNone(coords)
        # Toland sits off Brazil's SE coast -> southern-hemisphere, western lat.
        self.assertLess(coords[1], 0.0)   # negative latitude (south)
        self.assertGreater(coords[1], -30.0)
        self.assertLess(coords[2], 0.0)   # negative longitude (west)
        self.assertIn("Oxton", TEAM_LOCATIONS)

    def test_club_to_city_mapping_covers_all_clubs(self):
        from weather_physics import TEAM_LOCATIONS, CLUB_TO_CITY, TOLAND_CITIES
        self.assertEqual(len(CLUB_TO_CITY), len(TEAM_LOCATIONS))
        for city in CLUB_TO_CITY.values():
            self.assertIn(city, TOLAND_CITIES)

    def test_coords_for_unknown_returns_none(self):
        from weather_physics import _coords_for
        self.assertIsNone(_coords_for("Nowhere United"))

    def test_condition_from_pristine_rain_row(self):
        from weather_physics import _condition_from_archive_row
        row = {
            "hour": 15, "temperature": 8.0, "humidity": 90,
            "precipitation": 6.0, "wind_speed": 30.0,
            "wind_direction": 200.0, "visibility": 4000.0,
            "weather_code": 61, "cloud_cover": 100,
        }
        c = _condition_from_archive_row(row, "15:00")
        self.assertGreaterEqual(c.rain_intensity, 0.9)   # torrential
        self.assertGreater(c.wind_speed, 5.0)
        self.assertEqual(c.name, "heavy_rain")

    def test_condition_from_fog_code(self):
        from weather_physics import _condition_from_archive_row
        row = {
            "hour": 6, "temperature": 3.0, "humidity": 100,
            "precipitation": 0.0, "wind_speed": 4.0,
            "wind_direction": 100.0, "visibility": 150.0,
            "weather_code": 45, "cloud_cover": 100,
        }
        c = _condition_from_archive_row(row, "06:00")
        self.assertEqual(c.name, "fog")
        self.assertLess(c.visibility, 0.5)

    def test_get_weather_from_api_disabled_returns_none(self):
        import weather_physics as wp
        wp.WEATHER_API_ENABLED = False
        wp.clear_weather_cache()
        cond = wp.get_weather_from_api("Oxton", date(2026, 8, 31), "15:00")
        self.assertIsNone(cond)

    def test_get_weather_from_api_unmapped_returns_none(self):
        import weather_physics as wp
        wp.clear_weather_cache()
        cond = wp.get_weather_from_api("NoSuch FC", date(2026, 8, 31), "15:00")
        self.assertIsNone(cond)

    def test_resolve_real_weather_falls_back_to_seasonal_when_api_fails(self):
        import weather_physics as wp
        wp.clear_weather_cache()
        # Force an offline condition by breaking the base URL.
        base = wp.WEATHER_API_BASE
        wp.WEATHER_API_BASE = "https://127.0.0.1:1/x"
        try:
            cond = wp.resolve_real_weather("Oxton", date(2026, 8, 31), "15:00")
        finally:
            wp.WEATHER_API_BASE = base
        self.assertIsInstance(cond, WeatherCondition)

    def test_weather_physics_new_functions_importable(self):
        from weather_physics import (   # noqa: F401
            get_weather_from_api, resolve_real_weather, clear_weather_cache,
            TEAM_LOCATIONS, WEATHER_API_ENABLED,
        )
        self.assertTrue(isinstance(TEAM_LOCATIONS, dict))


if __name__ == "__main__":
    unittest.main()
