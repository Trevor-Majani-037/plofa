"""Step-8 (audit item 4 follow-through / "demote, don't delete"): rule layers
become *proposers* feeding the NeuralDecisionBrain as bounded priors, never
hard overrides.

This module is the SAME discipline as Step-7's decision journal:

  1. Deterministic: no input(), no RNG, no network. Every proposal is a pure
     function of (player, x, y, geometry, dna profile, fatigue, pressure,
     game_state) -- identical inputs always produce identical proposals.
  2. Byte-identical when off: the brain's caller passes ``rule_proposals=()``
     by default, and ``NeuralDecisionBrain.decide`` keeps
     ``chosen_intent`` authority 100% for itself. The proposals merely ride
     INTO ``_generate_candidates`` as *additional perceived candidates* with a
     ``rule_proposal`` tag; the chosen intent is still sampled from the full
     perceived set by the brain's own softmax. Heavy/scope-seeking proposals
     always lose unless the player's DNA/geometry genuinely favours them --
     the hybrid's whole point ("real football has no hard rule a footballer
     must obey").
  3. Explainable: every proposal carries a ``rule_source`` (which tactical
     instruction/coach layer wrote it), and when the brain picks a proposal
     the journal entry records ``obeyed_rule_source`` so ``why(player,
     minute)`` can answer "he obeyed the wide-combo instruction".

Why this is a SEPARATE module and NOT folded into decision_brain.py:
  - ``decision_brain.py`` is the sport engine's tested boundary (its 20+
    invariants run green and its on-ball behaviour is calibrated for realism
    -- shot/touch volumes, DNA-looking play). We do NOT bolt a second
    decision authority into it.
  - This module is the *thin, testable proposal supply*: pure geometry
    gating, possession-regression prior, attacking-matrix shot-window prior,
    wide-combo prior, gk-distribution prior. The brain stays the sole
    chooser; the proposals never set ``chosen_intent``.

The realism diagnostics (goals/shots/possession/pass completion/turnovers/
xG/sequence length/progressive passes) remain a SEPARATE consumer boundary
(Step-7 realism probes, NOT this module). This module only PROPOSES intents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Proposal envelope. A proposal is "what the coach/phase/geometry would like
# this player to consider next" -- never the outcome. The brain samples over
# perceived includes proposals exactly like any other candidate.
# ---------------------------------------------------------------------------
class RuleSource(str, Enum):
    TACTICAL_POSSESSION = "TacticalPossession"
    ATTACKING_MATRIX = "AttackingMatrix"
    WIDE_COMBO = "WideCombo"
    GK_DISTRIBUTION = "GKDistribution"


@dataclass(frozen=True)
class RuleProposal:
    intent: str                       # the intent the rule layer *proposes*
    objective_value: float = 0.0      # perceived goodness, on brain's own scale
    risk: float = 0.0
    target: Optional[str] = None
    rule_source: RuleSource = RuleSource.TACTICAL_POSSESSION
    note: str = ""


# ---------------------------------------------------------------------------
# Small pure helpers (same idiom as decision_brain's _clamp/_fraction).
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _fraction(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _tendency(profile: Any, name: str, default: float) -> float:
    dna = getattr(profile, "dna", profile)
    tendencies = getattr(dna, "tendencies", None) or {}
    return float(getattr(profile, name, tendencies.get(name, default)))


# ---------------------------------------------------------------------------
# 1. Possession-phase regression: propose recycling when the picture is
#    congested. The brain may override (a striker with a real window shoots).
# ---------------------------------------------------------------------------
def propose_phase_regression(
    minute: float,
    open_space: float,
    under_pressure: bool,
    plays_safe: float,        # 0..1 DNA tendency
    congestion: float,        # 0..1 density of bodies ahead
    regression_mode: Optional[str] = None,   # drop_to_gk|recycle|circulation
) -> List[RuleProposal]:
    if regression_mode is None:
        return []
    if open_space < 0.30 and congestion > 0.55:
        strength = 0.45 * plays_safe + 0.10 * (0.40 + congestion * 0.50)
        if regression_mode == "drop_to_gk":
            return [RuleProposal(
                "RECYCLE", 0.40, 0.05, None, RuleSource.TACTICAL_POSSESSION,
                "congestion forces a clean reset to the keeper"),
            ]
        return [RuleProposal(
            "SAFE_PASS", strength, 0.06, None, RuleSource.TACTICAL_POSSESSION,
            "phase regression: short recycle while the front stays congested"),
        ]
    return []


# ---------------------------------------------------------------------------
# 2. AttackingMatrix shot-window: propose SHOOT only when geometry/quality
#    justify a take. Calibration gate keeps scoreline realism (a boundary,
#    not a brain override).
# ---------------------------------------------------------------------------
def propose_shot(
    quality_score: float,      # 0..1 objective window goodness
    distance_to_goal: float,
    under_pressure: bool,
    calibration_floor: float = 0.70,
) -> List[RuleProposal]:
    if quality_score < calibration_floor:
        return []
    value = 0.20 + 0.45 * (1.0 - _fraction(distance_to_goal, 12.0, 30.0))
    if under_pressure:
        value *= 0.85
    return [RuleProposal(
        "SHOOT", _clamp(value, 0.05, 1.0), 0.55, None,
        RuleSource.ATTACKING_MATRIX, "a genuine shooting window the matrix opened",
    )]


# ---------------------------------------------------------------------------
# 3. Wide combo: propose CARRY/CROSS to a WIDE carrier when the flank is open
#    and their style leans wide. Brightest example of "coaching instruction as
#    a prior, not force".
# ---------------------------------------------------------------------------
def propose_wide_combo(
    position: str,
    flank_open: float,        # 0..1 openness of the outer channel
    pace_fraction: float,     # 0..1
    style_cross: float,       # 0..1 cross/delivery tendency
    open_space: float,
) -> List[RuleProposal]:
    if position not in ("LW", "RW", "LB", "RB"):
        return []
    if flank_open > 0.62 and open_space > 0.30:
        drive_value = 0.30 + 0.30 * pace_fraction + 0.20 * style_cross
        cross_value = 0.25 + 0.25 * style_cross
        out = [RuleProposal(
            "CARRY", _clamp(drive_value, 0.1, 0.9), 0.22, None,
            RuleSource.WIDE_COMBO, "wide corridor open; the combo favours the drive",
        )]
        if flank_open > 0.75:
            out.append(RuleProposal(
                "CROSS", _clamp(cross_value, 0.1, 0.9), 0.28, None,
                RuleSource.WIDE_COMBO, "byline window open for the delivery",
            ))
        return out
    return []


# ---------------------------------------------------------------------------
# 4. GK distribution: strong short-pass prior for a keeper told to play out
#    short (a coaching instruction) -- but the brain may still launch long.
# ---------------------------------------------------------------------------
def propose_gk_distribution(
    position: str,
    short_window: float,      # 0..1 openness of the short pass lane
    teammates_short: int,
    profile: Any,
) -> List[RuleProposal]:
    if position != "GK":
        return []
    if teammates_short <= 0:
        return []
    value = 0.25 + 0.45 * short_window
    return [RuleProposal(
        "SAFE_PASS", _clamp(value, 0.1, 0.8), 0.06,
        None, RuleSource.GK_DISTRIBUTION, "keeper distributing short per instruction",
    )]


# ---------------------------------------------------------------------------
# 5. fold: attach proposals into the brain's candidate stream.
# ---------------------------------------------------------------------------
def fold_proposals(
    base_candidates: List[Any],
    proposals: List[RuleProposal],
) -> List[Any]:
    """Append proposal-intents as additional perceived candidates. When
    proposals is empty this returns base_candidates *unchanged* (byte-identical
    when the gate is off)."""
    if not proposals:
        return base_candidates
    folded = list(base_candidates)
    for p in proposals:
        folded.append(_ProposalCandidate(p))
    return folded


@dataclass
class _ProposalCandidate:
    proposal: RuleProposal

    @property
    def intent(self) -> str:
        return self.proposal.intent

    @property
    def objective_value(self) -> float:
        return self.proposal.objective_value

    @property
    def risk(self) -> float:
        return self.proposal.risk

    @property
    def signature(self) -> str:
        return "rule::{0}::{1}".format(
            self.proposal.rule_source.value, self.proposal.intent,
        )
