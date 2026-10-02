"""
Club Philosophy layer (Checkpoint 30).

A ClubPhilosophy is the persistent, owner-driven identity that sits ABOVE
the per-match TeamProfile style.  Every manager who inherits the club adopts
it (with an adherence factor); every player's DNA bends toward it.  It is
NOT a per-match tactical button -- it is a long-term cultural constant that
gives the club a genuine, distinguishable way of playing.

Cruyff's axiom -- "if we have the ball, they can't score" -- lives here as
``hunger`` (possess at all costs) and ``starve`` (the opponent must not hold
the ball).  Mourinho's bus lives as the opposite: near-zero hunger,
near-zero patience, block-first.

Lever A (closed-loop split), Lever B (opponent starve) and Lever C
(patience → pass volume) all read from ClubPhilosophy fields blended into
TeamProfile at construction time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClubPhilosophy:
    """Persistent, club-level identity vector.

    All knobs are 0..1 unless otherwise noted.  A ``magnitude`` (0..1)
    controls how strongly the philosophy overrides the club's per-match
    style DNA; ``adherence`` (0..1) controls how faithfully the current
    players/coach buy in.

    ``identity_strength`` = magnitude × adherence and is the effective
    pull on every knob.
    """
    name: str

    # ── core identity dials ────────────────────────────────────────
    hunger: float    = 0.50   # Cruyff dial: desire to monopolise the ball
    patience: float  = 0.50   # willingness to circulate instead of forcing
    starve: float    = 0.00   # how hard we squeeze the OPPONENT's time on ball

    # ── DNA targets the philosophy pulls toward ────────────────────
    press_target:    float = 0.50
    line_target:     float = 0.50
    width_target:    float = 0.50
    tempo_target:    float = 0.50
    directness_target: float = 0.50
    shots_target:    float = 0.12   # shots per sequence (0.02-0.25)

    # ── scaling ────────────────────────────────────────────────────
    magnitude:  float = 0.50   # how distinctive the club's identity is
    adherence:  float = 0.80   # how fully the current squad buys in

    # ── metadata ───────────────────────────────────────────────────
    preferred_style: Optional[str] = None   # TeamStyle key hint
    description: str = ""

    # ── derived ────────────────────────────────────────────────────
    @property
    def identity_strength(self) -> float:
        return max(0.0, min(1.0, self.magnitude * self.adherence))


# ══════════════════════════════════════════════════════════════════════
#  ARCHETYPE LIBRARY
# ══════════════════════════════════════════════════════════════════════

ARCHETYPES: dict[str, ClubPhilosophy] = {}


def _register(phi: ClubPhilosophy) -> ClubPhilosophy:
    ARCHETYPES[phi.name] = phi
    return phi


# ── POSSESSION MONOPOLISTS ──────────────────────────────────────────

TOTAL_FOOTBALL = _register(ClubPhilosophy(
    name="TotalFootball",
    hunger=0.90, patience=0.90, starve=0.45,
    press_target=0.72, line_target=0.70, width_target=0.75,
    tempo_target=0.58, directness_target=0.25, shots_target=0.07,
    magnitude=0.92, adherence=0.95,
    preferred_style="tiki_taka",
    description=(
        "Cruyff/Ajax/Barcelona total football.  If we have the ball they "
        "can't score.  High hunger, high patience, suffocating press as a "
        "unit.  Wide build-up, patient circulation, relentless final-third "
        "recovery.  The philosophy IS the style."
    ),
))

POSITIONAL_DOMINANCE = _register(ClubPhilosophy(
    name="PositionalDominance",
    hunger=0.95, patience=0.88, starve=0.50,
    press_target=0.74, line_target=0.78, width_target=0.70,
    tempo_target=0.60, directness_target=0.22, shots_target=0.06,
    magnitude=0.95, adherence=0.96,
    preferred_style="tiki_taka",
    description=(
        "Pep Guardiola positional play -- hyper-high line, suffocating "
        "press after loss, surgical patient build-up, and relentless "
        "midfield rotation.  The most dominant possession identity in "
        "the modern game.  900 passes is the floor, not the ceiling."
    ),
))

# ── PRESSING / COUNTERPRESSING ──────────────────────────────────────

GEGENPRESSING = _register(ClubPhilosophy(
    name="Gegenpressing",
    hunger=0.45, patience=0.25, starve=0.90,
    press_target=0.95, line_target=0.75, width_target=0.65,
    tempo_target=0.95, directness_target=0.62, shots_target=0.13,
    magnitude=0.88, adherence=0.92,
    preferred_style="gegenpressing",
    description=(
        "Klopp's 8-second rule: win the ball back within 8 seconds of "
        "losing it, or press the opponent into a mistake.  The starve "
        "dial is the highest in the library -- this team's opponent "
        "barely touches the ball before being harried into a mistake."
    ),
))

# ── DEFENSIVE / ANTI-FOOTBALL ───────────────────────────────────────

LOW_BLOCK = _register(ClubPhilosophy(
    name="LowBlock",
    hunger=0.12, patience=0.15, starve=0.10,
    press_target=0.22, line_target=0.06, width_target=0.40,
    tempo_target=0.32, directness_target=0.48, shots_target=0.05,
    magnitude=0.85, adherence=0.90,
    preferred_style="park_the_bus",
    description=(
        "Mourinho / Terry / Catenaccio.  Every man is a defender.  We "
        "don't concede.  Deep block, low press, long clearances on the "
        "rare recovery.  The philosophy is NOT to score -- it is to NOT "
        "concede and punish on the counter when the opponent overcommits."
    ),
))

# ── DIRECT / VERTICAL ───────────────────────────────────────────────

ROUTE_ONE = _register(ClubPhilosophy(
    name="RouteOne",
    hunger=0.18, patience=0.08, starve=0.08,
    press_target=0.32, line_target=0.18, width_target=0.60,
    tempo_target=0.82, directness_target=0.95, shots_target=0.15,
    magnitude=0.80, adherence=0.85,
    preferred_style="route_one",
    description=(
        "Long diagonals, early crosses, set-piece dominance.  We get "
        "it forward as fast as possible.  Patience is a sin.  The "
        "philosophy is functional -- aerial threat, second balls, "
        "territory."
    ),
))

VERTICAL_TRANSITION = _register(ClubPhilosophy(
    name="VerticalTransition",
    hunger=0.35, patience=0.22, starve=0.50,
    press_target=0.70, line_target=0.55, width_target=0.55,
    tempo_target=0.88, directness_target=0.85, shots_target=0.14,
    magnitude=0.80, adherence=0.88,
    preferred_style="fluid_counter",
    description=(
        "High-octane counterattacking: press hard, win it, break "
        "vertically at speed.  Third-man runs, overlapping fullbacks, "
        "and a striker who lives on the shoulder.  Controlled chaos."
    ),
))

# ── ENTERTAINING / CHAOS ────────────────────────────────────────────

CHAOS_BALL = _register(ClubPhilosophy(
    name="ChaosBall",
    hunger=0.45, patience=0.12, starve=0.80,
    press_target=0.85, line_target=0.65, width_target=0.55,
    tempo_target=0.92, directness_target=0.80, shots_target=0.16,
    magnitude=0.75, adherence=0.80,
    preferred_style="ultra_attacking",
    description=(
        "Anfield at night / Simeone's Vicente Calderon.  Terrifying "
        "press, breakneck transitions, and zero patience.  We don't "
        "circulate -- we attack.  The philosophy is controlled chaos: "
        "opponents are suffocated, then hit on the break."
    ),
))

# ── MID-TABLE / NOISE ───────────────────────────────────────────────

MID_TABLE_PRAGMATISM = _register(ClubPhilosophy(
    name="MidTablePragmatism",
    hunger=0.30, patience=0.35, starve=0.12,
    press_target=0.45, line_target=0.35, width_target=0.45,
    tempo_target=0.55, directness_target=0.55, shots_target=0.11,
    magnitude=0.35, adherence=0.70,
    preferred_style="balanced",
    description=(
        "A typical mid-table club.  No strong identity, some pragmatic "
        "defending, occasional spells of nice football.  The philosophy "
        "exists but is faint -- magnitude 0.35 means the style DNA "
        "dominates and the identity only subtly biases the output."
    ),
))

DEFENSIVE_SOLIDARITY = _register(ClubPhilosophy(
    name="DefensiveSolidarity",
    hunger=0.20, patience=0.25, starve=0.10,
    press_target=0.30, line_target=0.25, width_target=0.40,
    tempo_target=0.40, directness_target=0.55, shots_target=0.06,
    magnitude=0.50, adherence=0.75,
    preferred_style="defensive",
    description=(
        "Lower-league survival: compact, hard to break down, no "
        "pretensions of dominance.  Clean sheets first."
    ),
))


# ══════════════════════════════════════════════════════════════════════
#  RESOLUTION HELPERS
# ══════════════════════════════════════════════════════════════════════

# Maps common manager `tactical_philosophy` labels → archetype names
_MANAGER_LABEL_MAP: dict[str, str] = {
    "gegenpress":  "Gegenpressing",
    "gegenpressing": "Gegenpressing",
    "low block":   "LowBlock",
    "low_block":   "LowBlock",
    "park the bus": "LowBlock",
    "tiki taka":   "TotalFootball",
    "positional":  "PositionalDominance",
    "possession":  "PositionalDominance",
    "route one":   "RouteOne",
    "route_one":   "RouteOne",
    "direct":      "RouteOne",
    "counter":     "VerticalTransition",
    "vertical":    "VerticalTransition",
    "fluid counter": "VerticalTransition",
    "mid table":   "MidTablePragmatism",
    "pragmatic":   "MidTablePragmatism",
    "balanced":    "MidTablePragmatism",
    "chaos":       "ChaosBall",
    "ultra attacking": "ChaosBall",
    "defensive":   "DefensiveSolidarity",
}


def resolve_philosophy(
    club: str,
    club_override: Optional[dict] = None,
    manager_label: Optional[str] = None,
) -> Optional[ClubPhilosophy]:
    """Resolve a ClubPhilosophy from the layered hierarchy.

    Priority: per-club override dict > manager label > None.
    The override dict may contain: archetype (str), magnitude (float),
    adherence (float) to customise the base archetype.
    """
    archetype_name: Optional[str] = None
    mag_adj: Optional[float] = None
    adh_adj: Optional[float] = None

    if club_override is not None:
        archetype_name = club_override.get("archetype")
        mag_adj = club_override.get("magnitude")
        adh_adj = club_override.get("adherence")
    elif manager_label is not None:
        key = manager_label.strip().lower()
        archetype_name = _MANAGER_LABEL_MAP.get(key)

    if archetype_name is None:
        return None

    base = ARCHETYPES.get(archetype_name)
    if base is None:
        return None

    # Shallow copy with optional adjustments
    phi = ClubPhilosophy(
        name=base.name,
        hunger=base.hunger, patience=base.patience, starve=base.starve,
        press_target=base.press_target, line_target=base.line_target,
        width_target=base.width_target, tempo_target=base.tempo_target,
        directness_target=base.directness_target, shots_target=base.shots_target,
        magnitude=mag_adj if mag_adj is not None else base.magnitude,
        adherence=adh_adj if adh_adj is not None else base.adherence,
        preferred_style=base.preferred_style,
        description=base.description,
    )
    return phi


def list_archetypes() -> list[str]:
    """Return sorted archetype names."""
    return sorted(ARCHETYPES.keys())
