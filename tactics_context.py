"""TACTICS-AS-CONTEXT — schema-v3 manager/team-instruction feature block.

PLOFA V2 audit Step 6: inject EffectiveTactics / manager instructions as
context features on the on-ball brain input, so the SAME situation is
read differently per the team's live tactical dials (a chasing team's
forward pass is not the same decision as a protecting team's).

Architecture
------------
- ``TACTICS_CONTEXT_D = 12`` fixed-dimension block appended AFTER the
  shared 24-d block and the schema-v2 role tail:
      v3 input = 24 shared + role block (7/8) + tactics context (12)
    → 43-d (GK/CB/FB/DM/CM families) / 44-d (AM/WING/ST).
- Features are duck-typed off the caller's ``team_profile`` (an
  ``EffectiveTactics`` in the match engine, a raw ``TeamProfile`` in
  harnesses): every read uses ``getattr`` with a neutral 0.5 default, so
  ANY object (or a partial one) produces a valid, deterministic block and
  a ``None`` profile returns ``None`` → caller zero-pads (dead inputs,
  mirroring unknown-role handling).
- The block encodes FOOTBALL meaning, not just numbers:
    index 0  tempo                 (dials)
    index 1  directness
    index 2  defensive_line
    index 3  press_intensity
    index 4  possession_target
    index 5  shots_per_sequence
    index 6  big_chance_ratio
    index 7  posture_attack_bias   (see_it_out ≈ 0 … all_out_chase = 1)
    index 8  stance_aggression     (SEE_IT_OUT ≈ 0 … ALL_OUT_CHASE = 1)
    index 9  style_attack_bias     (park_the_bus ≈ 0 … ultra_attacking 1)
    index 10 style_directness_bias (tiki_taka ≈ 0 … route_one = 1)
    index 11 style_press_bias      (park_the_bus ≈ 0 … gegenpressing = 1)
- Deterministic: identical inputs → identical floats.  Evolution
  synthesises TACTIC STATES via ``random_tactics_context(rng)`` so a
  challenger trains across the full instruction dial (same seed rule as
  the role/perception corpus).
- ``V3_INPUT_D`` = shared 24 + role + 12 per family.  The schema gate
  (``brain_schema``) treats ``v3_tactics_context`` like v2 but widens the
  allowed input to {43, 44}; the h1/h2/out tail must stay 32/32/10.

PURE-DATA / lazy-import discipline like role_features: only ``numpy`` at
module load.  Nothing pulls the engine in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from role_features import V2_INPUT_D, V2_SHARED_D


TACTICS_CONTEXT_D = 12

V3_INPUT_D: Dict[str, int] = {
    fam: V2_INPUT_D[fam] + TACTICS_CONTEXT_D for fam in V2_INPUT_D
}

# ─────────────────────────────────────────────────────────────
# MAPPING TABLES
# ─────────────────────────────────────────────────────────────

# Posture string -> attacking bias (0 = see it out, 1 = all-out chase).
# The engine appends suffixes: "all_out_chase+man_down" and
# "+fatigued" — parsed below.
POSTURE_ATTACK_BIAS: Dict[str, float] = {
    "see_it_out": 0.10,
    "protect_lead": 0.30,
    "baseline": 0.50,
    "tense_level": 0.55,
    "pushing": 0.75,
    "all_out_chase": 1.00,
}

# FormationStance -> aggression scalar (same semantic axis as posture).
STANCE_AGGRESSION: Dict[str, float] = {
    "see_it_out": 0.10,
    "protect_lead": 0.30,
    "baseline": 0.50,
    "tense_level": 0.55,
    "pushing": 0.75,
    "all_out_chase": 1.00,
}

# TeamStyle value -> (attack, directness, press) biases.
# Hand-calibrated PRIOR on what each authored style means on the pitch;
# these are PERCEPTUAL/context dials, never action rules.
STYLE_PROFILES: Dict[str, tuple[float, float, float]] = {
    "ultra_attacking":       (0.95, 0.80, 0.60),
    "attacking":             (0.85, 0.70, 0.60),
    "balanced":              (0.55, 0.50, 0.50),
    "defensive":             (0.40, 0.45, 0.50),
    "ultra_defensive":       (0.15, 0.30, 0.35),
    "gegenpressing":         (0.60, 0.55, 1.00),
    "tiki_taka":             (0.55, 0.15, 0.55),
    "park_the_bus":          (0.10, 0.20, 0.30),
    "route_one":             (0.50, 1.00, 0.45),
    "wing_play":             (0.60, 0.60, 0.50),
    "vertical_tiki_taka":    (0.70, 0.60, 0.60),
    "fluid_counter":         (0.70, 0.80, 0.50),
    "structured_possession": (0.50, 0.30, 0.50),
}

_DIAL_ATTRS = (
    "tempo", "directness", "defensive_line", "press_intensity",
    "possession_target", "shots_per_sequence", "big_chance_ratio",
)


def _clamp0(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _style_value(style: Any) -> str:
    """Normalise a TeamStyle enum / string / None to a profile key ('' unknown)."""
    if style is None:
        return ""
    v = getattr(style, "value", style)
    if not isinstance(v, str):
        return ""
    return v


def _posture_value(posture: Any) -> str:
    """Normalise a posture string; split '+man_down'/'+fatigued' suffixes."""
    if posture is None:
        return "baseline"
    s = str(posture)
    return s.split("+", 1)[0].strip() or "baseline"


def _stance_value(stance: Any) -> str:
    """Normalise a FormationStance enum / string to its value ('' unknown)."""
    if stance is None:
        return ""
    v = getattr(stance, "value", stance)
    if not isinstance(v, str):
        return ""
    return v.lower()


def _build_block(tempo, directness, defensive_line, press_intensity,
                 possession_target, shots_per_sequence, big_chance_ratio,
                 posture: str, stance: str, style_key: str) -> List[float]:
    """Assemble the 12-d block from already-normalised inputs."""
    posture_bias = POSTURE_ATTACK_BIAS.get(posture, 0.50)
    # Suffixes (man_down / fatigued) nudge the blocK slightly calmer.
    if "+man_down" in posture or "+fatigued" in posture:
        posture_bias = _clamp0(posture_bias - 0.05)
    stance_bias = STANCE_AGGRESSION.get(stance, 0.50)
    attack, direct, press = STYLE_PROFILES.get(style_key, (0.50, 0.50, 0.50))
    return [
        _clamp0(tempo), _clamp0(directness), _clamp0(defensive_line),
        _clamp0(press_intensity), _clamp0(possession_target),
        _clamp0(shots_per_sequence), _clamp0(big_chance_ratio),
        posture_bias, stance_bias, attack, direct, press,
    ]


def tactics_context_block(team_profile: Any) -> Optional[List[float]]:
    """12-d tactics-context block from a duck-typed team profile.

    Returns ``None`` when ``team_profile`` is ``None`` (caller zero-pads,
    consistent with unknown-role v2 tails).  Any other object — an
    ``EffectiveTactics``, a raw ``TeamProfile``, a ``SimpleNamespace``, a
    ``dict`` — is read with neutral 0.5 defaults for missing dials, so a
    partial profile still yields a deterministic, bounded block.
    """
    if team_profile is None:
        return None
    if isinstance(team_profile, dict):
        def _g(key: str, default: float = 0.5) -> float:
            v = team_profile.get(key, default)
            return float(v) if v is not None else default
        dials = tuple(_g(a) for a in _DIAL_ATTRS)
        posture = _posture_value(team_profile.get("posture", "baseline"))
        stance = _stance_value(team_profile.get("stance", ""))
        style_key = _style_value(team_profile.get("style"))
    else:
        def _g(key: str, default: float = 0.5) -> float:
            return float(getattr(team_profile, key, default))
        dials = tuple(_g(a) for a in _DIAL_ATTRS)
        posture = _posture_value(getattr(team_profile, "posture", "baseline"))
        stance = _stance_value(getattr(team_profile, "stance", None))
        style_key = _style_value(getattr(team_profile, "style", None))
    return _build_block(*dials, posture=posture, stance=stance,
                        style_key=style_key)


def random_tactics_context(rng: Any) -> List[float]:
    """Synthesise a plausible random tactic state for the evolution corpus.

    Consumes ONLY ``rng`` (like the role/perception corpus builders) so
    the legacy/non-tactics corpus path stays byte-identical when the
    tactics feature is off.  Dial values are drawn from football-plausible
    ranges; style/posture/stance are sampled from the real tables.
    """
    families = list(STYLE_PROFILES.keys())
    style_key = rng.choice(families)
    posture = rng.choice(sorted(POSTURE_ATTACK_BIAS.keys()))
    stance = rng.choice(sorted(STANCE_AGGRESSION.keys()))
    # dials: bias each toward the style's profile with spread
    attack, direct, press = STYLE_PROFILES[style_key]
    tempo = _clamp0(0.5 + (direct - 0.5) * 0.5 + rng.uniform(-0.2, 0.2))
    directness = _clamp0(direct + rng.uniform(-0.15, 0.15))
    defensive_line = _clamp0(0.5 + (1.0 - direct) * 0.3 + rng.uniform(-0.2, 0.2))
    press_intensity = _clamp0(press + rng.uniform(-0.15, 0.15))
    possession_target = _clamp0(0.5 + (1.0 - direct) * 0.4 + rng.uniform(-0.15, 0.15))
    shots_per_sequence = _clamp0(attack + rng.uniform(-0.15, 0.15))
    big_chance_ratio = _clamp0(attack * 0.6 + rng.uniform(-0.1, 0.2))
    return _build_block(
        tempo, directness, defensive_line, press_intensity,
        possession_target, shots_per_sequence, big_chance_ratio,
        posture=posture, stance=stance, style_key=style_key,
    )


def build_v3_vector(shared: Any, role: Optional[List[float]],
                    tactics: Optional[List[float]]) -> np.ndarray:
    """Concatenate shared v1 + role block (if any) + tactics block (if any).

    Mirrors ``role_features.build_v2_vector``: a ``None`` tail segment is
    simply omitted (caller zero-pads to the declared input width when a
    schema demands that segment for an unknown-role/poorly-typed player).
    """
    parts: list[np.ndarray] = [np.asarray(shared, dtype=np.float64)]
    if role is not None:
        parts.append(np.asarray(role, dtype=np.float64))
    if tactics is not None:
        parts.append(np.asarray(tactics, dtype=np.float64))
    return np.concatenate(parts)