"""Brain schema registry — versioning + strict validation for serialized brains.

PLOFA V2 audit Step 2: every serialized brain carries a metadata block so the
integration layer can detect a v1-era 24-d net vs a future perception-trained
net and route sensors accordingly.  Loaders accept v1 (no ``meta`` block) and
REJECT incompatible schemas loudly.

FILE FORMAT
-----------
v1 (legacy, ``brains/``):  no ``meta`` block — only ``{arch, w1, b1, w2, b2,
w3, b3}`` (+ ``kind`` on off-ball/press/defensive brains).  Loaded as v1.

v2 (new, ``brains_v2/``):  an extra ``meta`` block::

    "meta": {
        "arch_version": 1,
        "sensor_schema": "v1_24d",
        "normalization_version": "v1",
        "dna_schema": "v1",
        "training_method": "ga_surrogate",
        "generation": 80,
        "fitness": 0.9450,
        "seed": 123,
        "created": "2026-09-11"
    }

The versioned fields gate loading; ``generation/fitness/seed/created`` are
informational lineage and never gate.  A future perception-trained net would
declare ``sensor_schema`` other than ``v1_24d`` and the registry would refuse
it to a v1 loader (and route it to a perception-aware preprocessor instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────

BRAIN_FORMAT_VERSION = 2          # meta-aware file format
ARCH_VERSION = 1                  # architecture family version (weights layout)
SENSOR_SCHEMAS = (
    "v1_24d",             # shared 24-d vector (all legacy brains)
    "v2_role_features",   # 24-d shared + role-family block (7 or 8 dims)
    "v3_tactics_context", # 24-d + role block (7/8) + tactics-context block (12)
    "v4_offball_cert",    # off-ball 24-d + ball-certainty slot (25-d, press gate)
)
NORMALIZATION_VERSIONS = ("v1",)  # input normalization families
DNA_SCHEMAS = ("v1",)             # DNA attribute families
TRAINING_METHODS = (
    "ga_surrogate",   # evolved against FitnessSurrogate (on-ball)
    "ga_offball",     # evolved against OffBallSurrogate (off-ball conscience)
    "ga_team_press",  # evolved against TeamPressSurrogate (XI press controller)
    "ga_defensive",   # evolved against DefensiveActionSurrogate (which-act)
    "unknown",        # legacy / unspecified
)
BRAIN_KINDS = (
    "",                       # plain on-ball FootballBrain (no kind key)
    "offball_press_gate",     # OffBallBrain
    "team_press_engagement",  # TeamPressBrain
    "defensive_action",       # DefensiveActionBrain
)

# Expected architecture (in, h1, h2, out) per kind at ARCH_VERSION 1.
ARCH_BY_KIND: Dict[Tuple[int, str], Tuple[int, int, int, int]] = {
    (1, ""):                      (24, 32, 32, 10),
    (1, "offball_press_gate"):    (24, 32, 32, 1),
    (1, "team_press_engagement"): (24, 32, 32, 1),
    (1, "defensive_action"):      (24, 32, 32, 4),
}

# Default scratch namespace for meta-aware (v2) files — production ``brains/``
# is never written by the versioning machinery.
BRAINS_V2_DIR = "brains_v2"


class BrainSchemaError(Exception):
    """Raised when a serialized brain's schema is incompatible — loudly."""


# ─────────────────────────────────────────────────────────────
# META DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class BrainMeta:
    arch_version: int = ARCH_VERSION
    sensor_schema: str = "v1_24d"
    normalization_version: str = "v1"
    dna_schema: str = "v1"
    training_method: str = "unknown"
    # Lineage (informational, not gating)
    generation: Optional[int] = None
    fitness: Optional[float] = None
    seed: Optional[int] = None
    created: Optional[str] = None
    role_family: Optional[str] = None  # e.g. "ST" / "WING" / "FB" for v2_role_features

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrainMeta":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def legacy_meta() -> BrainMeta:
    """Meta assumed for a v1 file that carries no ``meta`` block at all."""
    return BrainMeta(arch_version=ARCH_VERSION, sensor_schema="v1_24d",
                     normalization_version="v1", dna_schema="v1",
                     training_method="unknown")


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise BrainSchemaError(msg)


def _resolve_meta(data: Dict[str, Any]) -> Tuple[BrainMeta, bool]:
    """Return (meta, is_v1).  Missing ``meta`` block ⇒ v1 legacy meta."""
    raw = data.get("meta")
    if raw is None:
        return legacy_meta(), True
    if not isinstance(raw, dict):
        raise BrainSchemaError(
            "brain 'meta' block must be a dict, got "
            f"{type(raw).__name__}")
    return BrainMeta.from_dict(raw), False


def validate(data: Dict[str, Any]) -> BrainMeta:
    """Parse and strictly validate a serialized brain's schema.

    Returns the resolved :class:`BrainMeta` (legacy v1 meta if the file has
    no ``meta`` block).  Raises :class:`BrainSchemaError` on ANY mismatch:
    unknown kind, arch version, sensor schema, normalization version, DNA
    schema, or training method; or an ``arch`` inconsistent with the kind.
    """
    kind = data.get("kind", "")
    _check(kind in BRAIN_KINDS,
           f"unknown brain kind {kind!r} (expected one of {BRAIN_KINDS})")

    meta, is_v1 = _resolve_meta(data)

    _check(meta.arch_version == ARCH_VERSION,
           f"incompatible arch_version {meta.arch_version!r}; "
           f"this loader supports only {ARCH_VERSION!r}")
    _check(meta.sensor_schema in SENSOR_SCHEMAS,
           f"incompatible sensor_schema {meta.sensor_schema!r}; "
           f"known: {SENSOR_SCHEMAS}")
    _check(meta.normalization_version in NORMALIZATION_VERSIONS,
           f"incompatible normalization_version "
           f"{meta.normalization_version!r}; known: {NORMALIZATION_VERSIONS}")
    _check(meta.dna_schema in DNA_SCHEMAS,
           f"incompatible dna_schema {meta.dna_schema!r}; known: {DNA_SCHEMAS}")
    _check(meta.training_method in TRAINING_METHODS,
           f"incompatible training_method {meta.training_method!r}; "
           f"known: {TRAINING_METHODS}")

    # Architecture must match the kind's expected layout at this arch version.
    expected = ARCH_BY_KIND.get((meta.arch_version, kind))
    if expected is not None:
        actual = tuple(data.get("arch", ()))
        _check(len(actual) == 4,
               f"arch {actual!r} malformed for kind {kind!r} at "
               f"arch_version {meta.arch_version} (expected {expected})")
        if meta.sensor_schema == "v2_role_features" and kind == "":
            # Role-feature on-ball brains widen ONLY the input (24+7 / 24+8);
            # the h1/h2/out tail must stay the canonical 32/32/10.
            try:
                from role_features import V2_INPUT_D
                allowed_in = sorted(set(V2_INPUT_D.values()))
            except Exception:  # noqa: BLE001  (role module unavailable -> fall back)
                allowed_in = [31, 32]
            _check(actual[1:] == expected[1:] and actual[0] in allowed_in,
                   f"arch {actual} incompatible with sensor_schema "
                   f"'v2_role_features' at arch_version "
                   f"{meta.arch_version} (tail {expected[1:]} exact, "
                   f"input in {allowed_in})")
        elif meta.sensor_schema == "v3_tactics_context" and kind == "":
            # Tactics-context brains stack the role tail AND the manager
            # context block: 24 + role (7/8) + tactics (12) -> 43 / 44.
            # Tail must still be the canonical 32/32/10.
            try:
                from tactics_context import V3_INPUT_D
                allowed_in = sorted(set(V3_INPUT_D.values()))
            except Exception:  # noqa: BLE001  (tactics module unavailable -> fall back)
                allowed_in = [43, 44]
            _check(actual[1:] == expected[1:] and actual[0] in allowed_in,
                   f"arch {actual} incompatible with sensor_schema "
                   f"'v3_tactics_context' at arch_version "
                   f"{meta.arch_version} (tail {expected[1:]} exact, "
                   f"input in {allowed_in})")
        elif meta.sensor_schema == "v4_offball_cert" and kind == "offball_press_gate":
            # Honest off-ball press gate: the 24-d off-ball vector + ONE
            # ball-certainty slot (slot 24) so the conscience can condition
            # press/hold on how sure the runner is of the ball position.
            # Tail must stay the canonical 32/32/1 for the binary gate.
            _check(actual[0] == 25 and actual[1:] == (32, 32, 1),
                   f"arch {actual} incompatible with sensor_schema "
                   f"'v4_offball_cert' at arch_version "
                   f"{meta.arch_version} (expected [25, 32, 32, 1])")
        else:
            _check(actual == expected,
                   f"arch {actual} incompatible with kind {kind!r} at "
                   f"arch_version {meta.arch_version} (expected {expected})")

    return meta


def check_shapes(data: Dict[str, Any], arrays: List[Any]) -> None:
    """Cross-check the weight array shapes against the declared ``arch``.

    ``arrays`` = [w1, b1, w2, b2, w3, b3].  This is the second line of
    defence after the registry rule check: a file that declares the right
    kind but ships wrong-shaped weights is still rejected loudly.
    """
    arch = tuple(data.get("arch", ()))
    if len(arch) != 4:
        raise BrainSchemaError(
            f"brain file declares malformed arch {arch!r} (need [in,h1,h2,out])")
    wanted = [
        (arch[0], arch[1]), (arch[1],),
        (arch[1], arch[2]), (arch[2],),
        (arch[2], arch[3]), (arch[3],),
    ]
    for arr, want in zip(arrays, wanted):
        if isinstance(want, tuple) and len(want) == 1:
            ok = arr.ndim == 1 and arr.shape[0] == want[0]
        else:
            ok = tuple(arr.shape) == want
        if not ok:
            raise BrainSchemaError(
                f"weight array shape {tuple(arr.shape)} does not match "
                f"declared arch {arch} (expected {want})")


def brain_meta_dict(training_method: str = "unknown",
                    generation: Optional[int] = None,
                    fitness: Optional[float] = None,
                    seed: Optional[int] = None,
                    created: Optional[str] = None,
                    **overrides: Any) -> Dict[str, Any]:
    """Convenience factory for a full v2 meta block (format BRAIN_FORMAT_VERSION).

    ``overrides`` let callers substitute any :class:`BrainMeta` field —
    e.g. ``sensor_schema=...`` for a future perception-trained net, or
    ``arch_version=...`` (which will be rejected by v1 loaders).
    """
    meta = BrainMeta(training_method=training_method,
                     generation=generation, fitness=fitness,
                     seed=seed, created=created, **overrides)
    d = meta.to_dict()
    d["format_version"] = BRAIN_FORMAT_VERSION
    return d