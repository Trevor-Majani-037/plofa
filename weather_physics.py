"""
PLOFA 26/27 — WEATHER & FIXTURE-TIME PHYSICS ENGINE
===================================================
weather_physics.py

Provides real-world physical and physiological modulations based on:
  1. WeatherCondition: rain intensity, wind speed & direction, visibility,
     pitch wetness, and ambient temperature.
  2. FixtureTimeEffect: kickoff time attendance impact and diurnal temperature
     proxy by calendar month.
  3. Real-climate generator based on venue and seasonal norms.

Toggle Guarantee:
  All physics calculations evaluate to 1.0 (neutral) and zero deflection unless
  WeatherPhysics.enabled is True or active weather with enabled flag is passed.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. WEATHER CONDITION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeatherCondition:
    """
    Physical conditions for a match.
    
    Fields:
      rain_intensity: 0.0 (bone dry) to 1.0 (torrential downpour)
      wind_speed    : wind velocity in m/s (0 = calm, 5 = breeze, 12 = strong, 20 = gale)
      wind_direction: wind direction in degrees (0 = blowing West->East (+x),
                      90 = North->South (+y), 180 = East->West (-x), 270 = South->North (-y))
      visibility    : 0.0 (zero visibility) to 1.0 (crystal clear)
      pitch_wetness : 0.0 (dry) to 1.0 (waterlogged/saturated surface)
      temperature_c : ambient air temperature in Celsius
      name          : descriptive preset label (clear, rain, wind, fog, etc.)
    """
    rain_intensity: float = 0.0
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    visibility: float = 1.0
    pitch_wetness: float = 0.0
    temperature_c: float = 15.0
    name: str = "clear"

    def summary(self) -> str:
        """Human-readable weather summary."""
        parts = [self.name.capitalize()]
        parts.append(f"{self.temperature_c:.1f}°C")
        if self.rain_intensity > 0.05:
            parts.append(f"Rain: {self.rain_intensity * 100:.0f}%")
        if self.wind_speed > 2.0:
            parts.append(f"Wind: {self.wind_speed:.1f} m/s @ {self.wind_direction:.0f}°")
        if self.visibility < 0.90:
            parts.append(f"Vis: {self.visibility * 100:.0f}%")
        if self.pitch_wetness > 0.10:
            parts.append(f"Pitch Wet: {self.pitch_wetness * 100:.0f}%")
        return " | ".join(parts)

    @classmethod
    def clear(cls, temperature_c: float = 18.0) -> WeatherCondition:
        return cls(
            rain_intensity=0.0,
            wind_speed=2.5,
            wind_direction=45.0,
            visibility=1.0,
            pitch_wetness=0.05,
            temperature_c=temperature_c,
            name="clear",
        )

    @classmethod
    def rain(
        cls,
        intensity: float = 0.60,
        pitch_wetness: Optional[float] = None,
        temperature_c: float = 12.0,
    ) -> WeatherCondition:
        wetness = pitch_wetness if pitch_wetness is not None else min(1.0, intensity * 1.1)
        return cls(
            rain_intensity=intensity,
            wind_speed=6.0,
            wind_direction=90.0,
            visibility=max(0.60, 1.0 - intensity * 0.4),
            pitch_wetness=wetness,
            temperature_c=temperature_c,
            name="rain" if intensity < 0.75 else "heavy_rain",
        )

    @classmethod
    def wind(
        cls,
        wind_speed: float = 13.0,
        wind_direction: float = 90.0,
        temperature_c: float = 13.0,
    ) -> WeatherCondition:
        return cls(
            rain_intensity=0.05,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            visibility=0.95,
            pitch_wetness=0.10,
            temperature_c=temperature_c,
            name="wind",
        )

    @classmethod
    def fog(
        cls,
        visibility: float = 0.30,
        temperature_c: float = 7.0,
    ) -> WeatherCondition:
        return cls(
            rain_intensity=0.10,
            wind_speed=1.5,
            wind_direction=0.0,
            visibility=visibility,
            pitch_wetness=0.35,
            temperature_c=temperature_c,
            name="fog",
        )

    @classmethod
    def from_string(cls, val: str, temperature_c: float = 15.0) -> WeatherCondition:
        """Map legacy string name to a rich WeatherCondition."""
        if not val:
            return cls.clear(temperature_c)
        s = str(val).strip().lower()
        if "rain" in s or "rail" in s:
            return cls.rain(intensity=0.60, temperature_c=temperature_c)
        if "wind" in s:
            return cls.wind(wind_speed=12.5, temperature_c=temperature_c)
        if "fog" in s or "mist" in s:
            return cls.fog(visibility=0.30, temperature_c=temperature_c)
        return cls.clear(temperature_c)


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEATHER PHYSICS MULTIPLIERS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class WeatherPhysics:
    """
    Physical simulations and parameter scalers driven by WeatherCondition.
    
    Safety:
      WeatherPhysics.enabled is False by default. When disabled, every
      multiplier returns 1.0 (neutral) and wind deflections return (0.0, 0.0).
    """
    enabled: bool = False
    _active_weather: Optional[WeatherCondition] = None

    @classmethod
    def is_active(cls, weather: Optional[Union[WeatherCondition, str]] = None) -> bool:
        """Return True only if physics is enabled and valid weather is provided."""
        if not cls.enabled:
            return False
        return weather is not None or cls._active_weather is not None

    @classmethod
    def set_active_weather(cls, weather: Optional[Union[WeatherCondition, str]], enabled: bool = True):
        cls.enabled = enabled
        if isinstance(weather, str):
            cls._active_weather = WeatherCondition.from_string(weather)
        else:
            cls._active_weather = weather

    @classmethod
    def reset(cls):
        cls.enabled = False
        cls._active_weather = None

    @classmethod
    def _resolve_condition(cls, weather: Optional[Union[WeatherCondition, str]] = None) -> Optional[WeatherCondition]:
        if weather is not None:
            if isinstance(weather, str):
                return WeatherCondition.from_string(weather)
            return weather
        return cls._active_weather

    # ── PASS ACCURACY & SPREAD ────────────────────────────────────────────────
    @classmethod
    def pass_accuracy_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
        is_long: bool = False,
        is_airborne: bool = False,
    ) -> float:
        """
        Accuracy multiplier for passes.
        Rain and pitch water reduce ball control (-5% to -12%).
        Strong wind adds drag/turbulence for airborne/long deliveries (-5% to -18%).
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0
        # Rain and pitch wetness effect
        mult -= w.rain_intensity * 0.08
        mult -= w.pitch_wetness * 0.04

        # Wind effect on airborne / long passes
        if is_airborne or is_long:
            if w.wind_speed > 5.0:
                wind_penalty = min(0.18, ((w.wind_speed - 5.0) / 20.0) * 0.16)
                mult -= wind_penalty

        return max(0.68, min(1.0, mult))

    @classmethod
    def pass_lateral_deflection(
        cls,
        to_x: float,
        to_y: float,
        from_x: float,
        from_y: float,
        weather: Optional[Union[WeatherCondition, str]] = None,
        is_airborne: bool = False,
        pitch_x: float = 105.0,
        pitch_y: float = 68.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate aerodynamic wind deflection for a ball in flight.
        Returns:
          (adjusted_x, adjusted_y, lateral_drift_m)
        Deflection is clamped so the ball does not exceed boundary limits unphysically.
        """
        if not cls.is_active(weather):
            return to_x, to_y, 0.0
        w = cls._resolve_condition(weather)
        if not w or w.wind_speed < 1.0:
            return to_x, to_y, 0.0

        dx = to_x - from_x
        dy = to_y - from_y
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            return to_x, to_y, 0.0

        # Estimated ball speed (m/s) and flight time (s)
        # Airborne passes travel at ~20 m/s; ground passes at ~15 m/s
        ball_speed = 20.0 if is_airborne else 15.0
        flight_time = dist / ball_speed

        # Wind velocity components
        # 0 deg = +x, 90 deg = +y
        rad = math.radians(w.wind_direction)
        w_vx = w.wind_speed * math.cos(rad)
        w_vy = w.wind_speed * math.sin(rad)

        # Drag coupling: airborne balls have high aerodynamic coupling;
        # ground passes have low wind coupling due to turf friction.
        k_drag = 0.09 if is_airborne else 0.02
        drift_x = w_vx * flight_time * k_drag
        drift_y = w_vy * flight_time * k_drag

        adj_x = max(0.0, min(pitch_x, to_x + drift_x))
        adj_y = max(0.0, min(pitch_y, to_y + drift_y))
        lateral_drift = math.hypot(adj_x - to_x, adj_y - to_y)

        return adj_x, adj_y, lateral_drift

    # ── SHOT POWER & SHOT ACCURACY ────────────────────────────────────────────
    @classmethod
    def shot_accuracy_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
        distance_m: float = 18.0,
    ) -> float:
        """
        Shot on-target probability multiplier.
        Slick/wet ball causes scuffed contacts; crosswinds deflect long-range shots.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0
        # Wet ball slippage
        mult -= w.rain_intensity * 0.07
        mult -= w.pitch_wetness * 0.03

        # Wind buffeting at long range
        if distance_m > 16.0 and w.wind_speed > 6.0:
            wind_pen = min(0.12, ((w.wind_speed - 6.0) / 18.0) * 0.12)
            mult -= wind_pen

        return max(0.72, min(1.0, mult))

    @classmethod
    def shot_power_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Shot power/speed multiplier.
        Waterlogged pitches and heavy air slightly slow down low drives.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0 - (w.pitch_wetness * 0.06) - (w.rain_intensity * 0.03)
        return max(0.88, min(1.0, mult))

    # ── STAMINA DRAIN RATE ────────────────────────────────────────────────────
    @classmethod
    def stamina_drain_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Stamina drain multiplier.
        Muddy/wet pitches increase mechanical work.
        Heat (>22°C) causes thermoregulatory strain.
        Freezing cold (<3°C) adds shivering/respiratory tax.
        Heavy rain adds drag.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0
        # Pitch resistance (wet mud/standing water)
        mult += w.pitch_wetness * 0.10
        mult += w.rain_intensity * 0.05

        # Thermal tax
        if w.temperature_c > 22.0:
            mult += min(0.16, (w.temperature_c - 22.0) * 0.015)
        elif w.temperature_c < 4.0:
            mult += min(0.10, (4.0 - w.temperature_c) * 0.015)

        return max(1.0, min(1.30, mult))

    # ── DRIBBLE CONTROL & CARRY DISTANCE ──────────────────────────────────────
    @classmethod
    def dribble_control_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """Close-control quality multiplier on slick or sticky turf."""
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        return max(0.80, 1.0 - (w.pitch_wetness * 0.12) - (w.rain_intensity * 0.05))

    @classmethod
    def carry_distance_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """Carry distance advance factor (water drag slightly truncates stride/pushes)."""
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        return max(0.85, 1.0 - (w.pitch_wetness * 0.08))

    @classmethod
    def rolling_decel_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Rolling-resistance multiplier for ground balls (1.0 = dry neutral).

        Wet / waterlogged turf grips the ball harder, so ground passes bite
        and die earlier: a saturated pitch adds up to ~45% rolling friction on
        top of the dry-grass baseline that possession physics applies
        (``BALL_ROLLING_DECEL``). This is the weather side of the ground-pass
        deceleration model — the same "we faked it with a constant, now it's
        derived" fix as the ballistic long-ball work.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        return max(1.0, min(1.45, 1.0 + w.pitch_wetness * 0.40 + w.rain_intensity * 0.15))

    # ── PRESSING WILLINGNESS ──────────────────────────────────────────────────
    @classmethod
    def pressing_intensity_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Pressing intensity factor.
        Managers throttle full-pitch gegenpressing in torrential rain or scorching heat.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0
        if w.rain_intensity > 0.50:
            mult -= (w.rain_intensity - 0.50) * 0.22
        if w.temperature_c > 26.0:
            mult -= (w.temperature_c - 26.0) * 0.015
        return max(0.80, min(1.0, mult))

    # ── PITCH GRIP & SLIPPING ─────────────────────────────────────────────────
    @classmethod
    def pitch_grip(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """Traction coefficient (1.0 = optimal dry studs grip, 0.65 = slick/mud)."""
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        return max(0.65, 1.0 - (w.pitch_wetness * 0.35))

    @classmethod
    def slip_probability(
        cls,
        action: str,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Probability that a high-acceleration action slips on wet turf.
        Tackles, sharp cuts, and sprints are most vulnerable.
        """
        if not cls.is_active(weather):
            return 0.0
        w = cls._resolve_condition(weather)
        if not w or w.pitch_wetness < 0.20:
            return 0.0

        grip = cls.pitch_grip(w)
        traction_loss = 1.0 - grip
        if action in ("tackle_aggressive", "tackle", "slide_tackle"):
            return min(0.12, traction_loss * 0.25)
        elif action in ("dribble", "sprint", "cut_back"):
            return min(0.08, traction_loss * 0.18)
        return min(0.03, traction_loss * 0.08)

    # ── GOALKEEPER POSITIONING & REACH ────────────────────────────────────────
    @classmethod
    def gk_reach_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Goalkeeper diving push-off reach multiplier.
        Slick grass reduces firm push-off for diving reaches (-4% to -10%).
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        mult = 1.0 - (w.pitch_wetness * 0.07) - (w.rain_intensity * 0.04)
        return max(0.88, min(1.0, mult))

    @classmethod
    def gk_reaction_mult(
        cls,
        weather: Optional[Union[WeatherCondition, str]] = None,
    ) -> float:
        """
        Goalkeeper visual tracking / reaction time multiplier.
        Fog and poor visibility delay tracking of incoming strikes.
        """
        if not cls.is_active(weather):
            return 1.0
        w = cls._resolve_condition(weather)
        if not w:
            return 1.0

        if w.visibility < 0.90:
            delay = (0.90 - w.visibility) * 0.28
            return max(0.72, 1.0 - delay)
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIXTURE-TIME LAYER
# ─────────────────────────────────────────────────────────────────────────────

class FixtureTimeEffect:
    """
    Evaluates kickoff time effects on attendance and diurnal temperature variation.
    """

    @staticmethod
    def parse_time_str(start_time: Optional[Union[str, time, datetime]]) -> Optional[time]:
        """Normalize start time input to datetime.time."""
        if start_time is None:
            return None
        if isinstance(start_time, time):
            return start_time
        if isinstance(start_time, datetime):
            return start_time.time()
        s = str(start_time).strip()
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                pass
        return None

    @classmethod
    def attendance_mult(
        cls,
        start_time: Optional[Union[str, time, datetime]],
        is_weekend: bool = True,
        is_derby: bool = False,
    ) -> float:
        """
        Attendance fill-rate multiplier based on kickoff slot.
        - Saturday 15:00 is standard peak (1.0).
        - Early 12:30 kickoff is slightly lower (0.96-0.97).
        - Evening floodlit (19:45, 20:00) on weekdays sees commuter constraints (0.94-0.96).
        - Derbies are immune to kickoff timing dropoffs (1.0).
        """
        if is_derby:
            return 1.0

        t = cls.parse_time_str(start_time)
        if t is None:
            return 1.0

        hour = t.hour + (t.minute / 60.0)

        # Early kickoff (12:00 - 13:00)
        if hour < 13.5:
            return 0.965

        # Traditional afternoon (14:00 - 16:30)
        if 14.0 <= hour <= 16.5:
            return 1.00

        # Twilight kickoff (17:00 - 18:30)
        if 16.5 < hour <= 18.5:
            return 0.985 if is_weekend else 0.965

        # Night floodlit kickoff (>= 19:00)
        if hour > 18.5:
            return 0.980 if is_weekend else 0.945

        return 1.0

    @classmethod
    def temperature_proxy(
        cls,
        month: int,
        start_time: Optional[Union[str, time, datetime]] = None,
    ) -> float:
        """
        Approximate Toland / southeast-Brazil coastal ambient match temperature
        (°C) by calendar month and kickoff diurnal cycle.
        """
        # Monthly mean afternoon baseline (°C) — Sao Paulo / SE-Brazil norm.
        monthly_baselines = {
            1: 25.5,   # January   (summer, hot & humid)
            2: 25.5,   # February
            3: 24.5,   # March
            4: 22.0,   # April
            5: 19.5,   # May
            6: 18.0,   # June      (mild "winter")
            7: 17.5,   # July
            8: 18.5,   # August
            9: 19.5,   # September
            10: 21.0,  # October
            11: 22.5,  # November
            12: 24.0,  # December
        }
        base_temp = monthly_baselines.get(month, 22.0)

        t = cls.parse_time_str(start_time)
        if t is None:
            return base_temp

        hour = t.hour + (t.minute / 60.0)
        # Diurnal curve: peak warmth around 14:00-15:00; cooler morning and evening
        if hour <= 12.5:
            temp_offset = -2.0
        elif 13.0 <= hour <= 16.0:
            temp_offset = +1.0
        elif 16.0 < hour <= 18.5:
            temp_offset = -1.0
        else:  # >= 19:00 night match under floodlights
            temp_offset = -4.0

        return round(base_temp + temp_offset, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. REAL-CLIMATE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_climate_weather(
    venue: str,
    match_date: date,
    start_time: Optional[Union[str, time, datetime]] = None,
    rng: Optional[random.Random] = None,
) -> WeatherCondition:
    """
    Produce a realistic WeatherCondition for a venue on match_date.
    Uses seasonal UK climate distributions with controlled stochastic variation.
    """
    r = rng if rng is not None else random

    month = match_date.month
    temp_c = FixtureTimeEffect.temperature_proxy(month, start_time)
    temp_c += r.uniform(-2.0, 2.5)

    # Seasonal weather probabilities for Toland / SE-Brazil:
    # (p_clear, p_rain, p_wind, p_fog) — humid with a tropical wet season.
    if month in (12, 1, 2):      # Summer — hot & wet season
        weights = [0.30, 0.55, 0.12, 0.03]
    elif month in (3, 4, 5):      # Autumn
        weights = [0.38, 0.46, 0.12, 0.04]
    elif month in (6, 7, 8):      # Winter — milder, drier, foggy mornings
        weights = [0.50, 0.30, 0.14, 0.06]
    else:                         # Spring (9, 10, 11)
        weights = [0.40, 0.44, 0.12, 0.04]

    wtype = r.choices(["clear", "rain", "wind", "fog"], weights=weights)[0]

    if wtype == "rain":
        intensity = r.uniform(0.25, 0.90)
        wind_spd = r.uniform(3.0, 10.0)
        wind_dir = r.uniform(0.0, 360.0)
        pitch_wet = min(1.0, intensity * r.uniform(0.9, 1.25))
        vis = max(0.55, 1.0 - intensity * 0.45)
        return WeatherCondition(
            rain_intensity=round(intensity, 2),
            wind_speed=round(wind_spd, 1),
            wind_direction=round(wind_dir, 1),
            visibility=round(vis, 2),
            pitch_wetness=round(pitch_wet, 2),
            temperature_c=round(temp_c, 1),
            name="rain" if intensity < 0.70 else "heavy_rain",
        )
    elif wtype == "wind":
        wind_spd = r.uniform(10.0, 18.0)
        wind_dir = r.uniform(0.0, 360.0)
        rain_int = r.uniform(0.0, 0.20)
        pitch_wet = round(rain_int * 0.8, 2)
        return WeatherCondition(
            rain_intensity=round(rain_int, 2),
            wind_speed=round(wind_spd, 1),
            wind_direction=round(wind_dir, 1),
            visibility=0.95,
            pitch_wetness=pitch_wet,
            temperature_c=round(temp_c, 1),
            name="wind",
        )
    elif wtype == "fog":
        vis = r.uniform(0.20, 0.45)
        wind_spd = r.uniform(0.5, 2.5)
        return WeatherCondition(
            rain_intensity=0.08,
            wind_speed=round(wind_spd, 1),
            wind_direction=0.0,
            visibility=round(vis, 2),
            pitch_wetness=0.35,
            temperature_c=round(temp_c - 1.5, 1),
            name="fog",
        )
    else:  # clear
        wind_spd = r.uniform(1.0, 5.0)
        wind_dir = r.uniform(0.0, 360.0)
        return WeatherCondition(
            rain_intensity=0.0,
            wind_speed=round(wind_spd, 1),
            wind_direction=round(wind_dir, 1),
            visibility=1.0,
            pitch_wetness=0.04,
            temperature_c=round(temp_c, 1),
            name="clear",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIXTURE EXCEL LOADER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_fixture_info(
    excel_path: str,
    matchday: int,
    home_team: str,
    away_team: str,
) -> Dict[str, Optional[Union[str, int, float]]]:
    """
    Extract fixture metadata ('Start Time', 'Venue', 'Capacity') from the
    'FIXTURES' sheet in PLOFA Excel.
    """
    import os
    if not os.path.exists(excel_path):
        return {"start_time": None, "venue": None, "capacity": None}

    try:
        import openpyxl
        wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
        if "FIXTURES" not in wb.sheetnames:
            return {"start_time": None, "venue": None, "capacity": None}

        ws = wb["FIXTURES"]
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
        if not header_row:
            return {"start_time": None, "venue": None, "capacity": None}

        col_map = {str(h).strip(): i for i, h in enumerate(header_row) if h is not None}
        md_idx = col_map.get("Matchday")
        home_idx = col_map.get("Home")
        away_idx = col_map.get("Away")
        time_idx = col_map.get("Start Time")
        venue_idx = col_map.get("Venue")
        cap_idx = col_map.get("Capacity")

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or row[home_idx] is None:
                continue
            r_md = row[md_idx] if md_idx is not None else None
            r_home = str(row[home_idx]).strip() if home_idx is not None else ""
            r_away = str(row[away_idx]).strip() if away_idx is not None else ""

            # Match matchday and team names
            if r_md == matchday and r_home.lower() == home_team.strip().lower() and (
                not away_team or r_away.lower() == away_team.strip().lower()
            ):
                raw_time = row[time_idx] if time_idx is not None else None
                start_time_str = None
                if raw_time is not None:
                    if isinstance(raw_time, time):
                        start_time_str = raw_time.strftime("%H:%M")
                    elif isinstance(raw_time, datetime):
                        start_time_str = raw_time.strftime("%H:%M")
                    else:
                        start_time_str = str(raw_time).strip()

                venue_val = str(row[venue_idx]).strip() if venue_idx is not None and row[venue_idx] else None
                cap_val = int(row[cap_idx]) if cap_idx is not None and row[cap_idx] else None

                return {
                    "start_time": start_time_str,
                    "venue": venue_val,
                    "capacity": cap_val,
                }
    except Exception:
        pass

    return {"start_time": None, "venue": None, "capacity": None}


# ─────────────────────────────────────────────────────────────────────────────
# 6. REAL-WORLD WEATHER API (Open-Meteo — 100% free, no key, no signup)
# ─────────────────────────────────────────────────────────────────────────────
# Toland is a fictional nation attached to Brazil's SE coast. WEATHER_API_ENABLED
# controls whether we pull REAL historical Brazilian-Atlantic coastal weather for
# each venue from Open-Meteo. When disabled (or offline, or a venue has no mapped
# coordinates), the seasonal climate model (get_climate_weather) is used as a
# seamless fallback so the simulation never breaks.
#
# How locations resolve:
#   * TOLAND_CITIES   — your 10 named Toland cities, each pinned to a real
#                       southeast-Brazil coastal coordinate (so Open-Meteo
#                       returns genuine warm/tropical/Atlantic weather).
#   * CLUB_TO_CITY    — each club -> its Toland city.
#   * TEAM_LOCATIONS  — derived map: club -> (label, lat, lon).
# Edit TOLAND_CITIES / CLUB_TO_CITY to reflect Toland's true geography; every
# change flows through automatically to the weather lookup.
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_API_ENABLED: bool = True
WEATHER_API_BASE: str = "https://archive-api.open-meteo.com/v1/archive"
_WEATHER_CACHE: Dict[Tuple[str, str], Optional["WeatherCondition"]] = {}

# Real southeast-Brazil coastal anchors used for Toland's cities.
# (label, latitude, longitude)
TOLAND_CITIES: Dict[str, Tuple[str, float, float]] = {
    "Avada":            ("Avada (Toland) @ Sao Paulo",    -23.5505, -46.6333),
    "Natrican City":    ("Natrican (Toland) @ Rio",       -22.9068, -43.1729),
    "Port Virginia":    ("Port Virginia (Toland) @ Santos", -23.9608, -46.3322),
    "Pearls Lake Town": ("Pearls Lake (Toland) @ SJC",    -23.1791, -45.8872),
    "Madzustico":       ("Madzustico (Toland) @ Vitoria", -20.3155, -40.3128),
    "Old Lige Road":    ("Old Lige (Toland) @ Bauru",     -22.3145, -49.0585),
    "Uditon City":      ("Uditon (Toland) @ Campinas",    -22.9099, -47.0626),
    "Oxland City":      ("Oxland (Toland) @ Ribeirao",    -21.1775, -47.8103),
    "Tuneelbeyn Slbey": ("Telbey (Toland) @ Sorocaba",    -23.5015, -47.4586),
    "West Talern":      ("West Talern (Toland) @ Londrina", -23.3106, -51.1628),
}

# club -> Toland city (Tier 1 & 2). Confirmed to cover every club in the
# PLOFA-2026-2027.xlsx FIXTURES sheet (18 clubs + the "Avada Zenith" capital club).
CLUB_TO_CITY: Dict[str, str] = {
    "Allburn": "Avada",          "Brim City": "Avada",
    "Buelton": "Avada",          "Claw": "Avada",
    "Seafcea": "Avada",          "FCTP2": "Avada",
    "Ganester": "Avada",         "Green Park City": "Avada",
    "Stade Den 08": "Avada",     "Football Venna": "Avada",
    "Draz": "Avada",             "Peiden": "Avada",
    "Titans": "Avada",           "Avada Zenith": "Avada",
    "Club Chovers": "Pearls Lake Town",  "Pearls": "Pearls Lake Town",
    "Triumpher": "Pearls Lake Town",     "Warters": "Pearls Lake Town",
    "Oxton": "Oxland City",      "Tryox City": "Oxland City",
    "Windteam": "Oxland City",   "Seed Grenet": "Oxland City",
    "Play City": "Port Virginia", "Quilos": "Port Virginia",
    "Seaton": "Port Virginia",
    "Natrican": "Natrican City",
    "Justice": "Madzustico",     "Vorlians": "Madzustico",
    "Lige-8": "Old Lige Road",   "Rodice": "Old Lige Road",
    "Uditon": "Uditon City",     "Red Wolves": "Uditon City",
    "Telbey": "Tuneelbeyn Slbey",
    "Trendboys": "West Talern",  "Niente": "West Talern",
}

# Derived: club -> (label, lat, lon)
TEAM_LOCATIONS: Dict[str, Tuple[str, float, float]] = {
    club: TOLAND_CITIES[city]
    for club, city in CLUB_TO_CITY.items()
    if city in TOLAND_CITIES
}


def _coords_for(venue: Optional[str]) -> Optional[Tuple[str, float, float]]:
    if not venue:
        return None
    key = str(venue).strip()
    if key in TEAM_LOCATIONS:
        return TEAM_LOCATIONS[key]
    return None


def _nearest_hour(time_obj: Optional[Union[str, time, datetime]]) -> int:
    t = FixtureTimeEffect.parse_time_str(time_obj)
    return t.hour if t else 15


def _fetch_archive(
    lat: float, lon: float, match_date: date, hour: int,
    timeout: float = 15.0,
) -> Optional[dict]:
    """Query Open-Meteo historical archive (no key required). Returns the
    hourly row closest to the kickoff hour, or None on any failure."""
    import urllib.request
    import urllib.parse
    import json
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": match_date.isoformat(),
            "end_date": match_date.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                      "wind_speed_10m,wind_direction_10m,visibility,"
                      "weather_code,cloud_cover",
            "timezone": "Europe/London",
        }
        url = WEATHER_API_BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "plofa/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return None
        target = match_date.strftime("%Y-%m-%dT") + f"{hour:02d}:00"
        idx = 0
        for i, t in enumerate(times):
            if t <= target:
                idx = i
            else:
                break
        row = {
            "hour": hour,
            "temperature": _get(hourly, "temperature_2m", idx),
            "humidity": _get(hourly, "relative_humidity_2m", idx),
            "precipitation": _get(hourly, "precipitation", idx),
            "wind_speed": _get(hourly, "wind_speed_10m", idx),
            "wind_direction": _get(hourly, "wind_direction_10m", idx),
            "visibility": _get(hourly, "visibility", idx),
            "weather_code": _get(hourly, "weather_code", idx),
            "cloud_cover": _get(hourly, "cloud_cover", idx),
        }
        return row
    except Exception:
        return None


def _get(hourly: dict, key: str, idx: int):
    try:
        vals = hourly.get(key) or []
        if idx < len(vals):
            return vals[idx]
    except Exception:
        pass
    return None


def _condition_from_archive_row(row: dict, start_time) -> WeatherCondition:
    """Convert an Open-Meteo hourly row into a WeatherCondition."""
    temp = row["temperature"]
    temp_c = temp if temp is not None else 15.0

    precip = row["precipitation"] or 0.0            # mm in the hour
    # 0.0 dry, ~2.5 heavy, >= 5 torrential
    if precip >= 5.0:
        intensity = 1.0
    elif precip >= 2.5:
        intensity = 0.80
    elif precip >= 1.0:
        intensity = 0.55
    elif precip > 0.0:
        intensity = 0.30
    else:
        intensity = 0.0

    wind_kmh = row["wind_speed"] if row["wind_speed"] is not None else 0.0
    wind_ms = wind_kmh / 3.6                       # km/h -> m/s
    wind_dir = row["wind_direction"] if row["wind_direction"] is not None else 90.0

    wcode = row["weather_code"]
    # WMO codes: 0-1 clear, 2-3 partly, 45/48 fog, 51-57 drizzle,
    # 61-67 rain, 71-77 snow, 80-82 showers, 95-99 thunderstorm
    wetness = min(1.0, intensity * 1.1)
    if wcode is not None and (wcode in (45, 48)):
        visibility = 0.30                                # fog
        rain_fog = 0.10
        wetness = max(wetness, 0.35)
        name = "fog"
    elif wcode is not None and (51 <= wcode <= 57):
        visibility = 0.90
        rain_fog = 0.35
        name = "rain"
    elif wcode is not None and wcode in (71, 73, 75, 77, 85, 86):
        visibility = 0.60
        rain_fog = min(0.5, intensity)
        wetness = 0.9                                  # snow/sleet
        name = "snow"
    elif intensity > 0.0:
        visibility = max(0.55, 1.0 - intensity * 0.45)
        rain_fog = intensity
        name = "rain" if intensity < 0.75 else "heavy_rain"
    else:
        visibility = 0.95
        rain_fog = 0.05
        name = "wind" if wind_ms > 8.0 else "clear"

    return WeatherCondition(
        rain_intensity=round(rain_fog, 2),
        wind_speed=round(wind_ms, 1),
        wind_direction=round(wind_dir, 1),
        visibility=round(visibility, 2),
        pitch_wetness=round(wetness, 2),
        temperature_c=round(float(temp_c), 1),
        name=name,
    )


def get_weather_from_api(
    venue: Optional[str],
    match_date: date,
    start_time: Optional[Union[str, time, datetime]] = None,
    timeout: float = 15.0,
) -> Optional[WeatherCondition]:
    """Fetch real historical weather for a venue/date from Open-Meteo.

    Returns a WeatherCondition, or None if the API is disabled, offline, or
    the venue has no mapped coordinates.
    """
    if not WEATHER_API_ENABLED:
        return None
    coords = _coords_for(venue)
    if coords is None:
        return None

    hour = _nearest_hour(start_time)
    cache_key = (str(venue).strip(), match_date.isoformat())
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]

    row = _fetch_archive(coords[1], coords[2], match_date, hour, timeout=timeout)
    if row is None:
        _WEATHER_CACHE[cache_key] = None
        return None

    cond = _condition_from_archive_row(row, start_time)
    _WEATHER_CACHE[cache_key] = cond
    return cond


def resolve_real_weather(
    venue: Optional[str],
    match_date: date,
    start_time: Optional[Union[str, time, datetime]] = None,
    rng: Optional[random.Random] = None,
    use_api: Optional[bool] = None,
) -> WeatherCondition:
    """
    Best-effort real weather:
      1. Live Open-Meteo historical archive (if enabled & venue has coords).
      2. Otherwise the seasonal climate model.
    Never raises / never blocks the sim: any API failure falls back silently.
    """
    effective = WEATHER_API_ENABLED if use_api is None else use_api
    if effective:
        cond = get_weather_from_api(venue, match_date, start_time)
        if cond is not None:
            return cond
    return get_climate_weather(venue, match_date, start_time, rng)


def clear_weather_cache() -> None:
    _WEATHER_CACHE.clear()
