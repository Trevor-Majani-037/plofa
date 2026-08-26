"""
PLOFA 26/27 — BLOCK AWARENESS MODULE
=====================================
block_awareness.py

Philosophy:
    Real midfielders don't just pass to available teammates — they read the
    opponent's defensive BLOCK and pass AROUND it. Modric's 79 passes orbit
    the opponent's compact mid-block like water around a rock. Tanaka's 96%
    accuracy comes from never forcing the ball through density.

    This module closes that gap. It detects the opponent's defensive shape
    from live player positions, identifies natural channels around it, and
    scores every candidate pass by how intelligently it navigates that shape.

    It is NOT a replacement for PositionEngine or AttackingMatrix. It is a
    MACRO layer that sits above their MICRO lane-clearance logic. Together:
        - AttackingMatrix asks: "Is THIS corridor open right NOW?"
        - BlockAwareness asks: "Does this pass go AROUND the block or THROUGH it?"

Architecture:
    BlockShape          — Detected opponent defensive structure (center, compactness, shape type)
    Channel             — A natural passing corridor (half-space, wide channel, deep recycle)
    BlockDetector       — Computes BlockShape from opponent spatial states
    BlockNavigationEngine — Scores passes by orbital navigation around the block
    HalfSpaceMagnet     — Pulls CAMs/CMs toward half-spaces when opponent block is compact
    RecycleTendency     — Rewards volume passers for keeping the ball moving around edges

Integration:
    1. MatchEngine creates one BlockDetector per team, updated every minute.
    2. EventChain._pick_receiver() calls BlockNavigationEngine.pass_orbital_score()
       as an additional multiplier on the existing label_weight.
    3. PositionEngine.DriftEngine queries HalfSpaceMagnet for CAM/CM drift targets.
    4. The result: CAMs cluster in half-spaces, CMs recycle around the block,
       and pass maps show the "orbital" pattern seen in Modric/Tanaka/Ødegaard.

Design reference: Checkpoint 29 — Block Awareness
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto

# ─────────────────────────────────────────────
# PITCH CONSTANTS (mirrors geometry_engine.py)
# ─────────────────────────────────────────────

PITCH_LENGTH: float = 105.0   # x-axis, 0 = left goal, 105 = right goal
PITCH_WIDTH: float = 68.0     # y-axis, 0 = bottom touchline, 68 = top
CENTER_X: float = 52.5
CENTER_Y: float = 34.0

# Defensive third / middle third / attacking third boundaries
DEF_THIRD_END: float = 35.0
MID_THIRD_END: float = 70.0

# ─────────────────────────────────────────────
# BLOCK SHAPE TYPE
# ─────────────────────────────────────────────

class BlockShapeType(Enum):
    """
    The opponent's defensive posture as a cohesive shape.
    Detected from spatial dispersion, not from team style labels.
    """
    COMPACT_MID_BLOCK = auto()   # Tight center, moderate depth — the classic "orbital" target
    SPREAD_PRESS = auto()        # High line, wide spread — no compact center to orbit
    LOW_BLOCK = auto()           # Very deep, very tight — channels are wide, not half-space
    HIGH_LINE = auto()           # Aggressive offside trap — space BEHIND is the channel
    BROKEN = auto()              # Transition, no coherent shape — play what's open


# ─────────────────────────────────────────────
# CHANNEL
# ─────────────────────────────────────────────

@dataclass
class Channel:
    """
    A natural corridor around or through the opponent's block.
    Channels are discovered, not hardcoded — they depend on where the
    opponent's density ISN'T.
    """
    name: str                     # e.g. "left_half_space", "right_wide", "deep_recycle"
    center: Tuple[float, float]   # (x, y) on pitch
    width: float                  # Lateral width of the channel (metres)
    accessibility: float          # 0.0–1.0, how open this channel is right now
    description: str = ""         # Human-readable why this channel exists


# ─────────────────────────────────────────────
# BLOCK SHAPE
# ─────────────────────────────────────────────

@dataclass
class BlockShape:
    """
    The opponent's defensive block as a single geometric object.
    Think of it as a "cloud" of defensive density with a center and a shape.
    """
    # Geometric center of the block (weighted by defensive importance)
    center_x: float
    center_y: float

    # How tight is the block? 0.0 = players spread across half the pitch,
    # 1.0 = all defensive players within a 15m radius cluster
    compactness: float

    # Lateral and longitudinal spread (metres, standard deviation of player positions)
    width: float      # spread along y-axis
    depth: float      # spread along x-axis

    # Classified shape type
    shape_type: BlockShapeType

    # Identified channels around/through the block
    channels: List[Channel] = field(default_factory=list)

    # Per-zone density: how many opponent players are in each coarse zone
    zone_density: Dict[str, float] = field(default_factory=dict)

    # Which opponent positions were used to compute this block
    contributing_positions: Set[str] = field(default_factory=set)

    # Timestamp (match minute) this block was computed
    minute: int = 0

    @property
    def radius(self) -> float:
        """Effective radius of the block cloud (metres)."""
        # A compact block has smaller radius; a spread block has larger
        return 8.0 + (1.0 - self.compactness) * 18.0

    @property
    def is_compact(self) -> bool:
        return self.compactness >= 0.55

    @property
    def is_spread(self) -> bool:
        return self.compactness <= 0.35

    def channel_named(self, name: str) -> Optional[Channel]:
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None


# ─────────────────────────────────────────────
# BLOCK DETECTOR
# ─────────────────────────────────────────────

class BlockDetector:
    """
    Computes a live BlockShape from the opponent team's current positions.

    Uses defensive-position-weighted center of gravity, so a DM and two CBs
    pull the block center deeper than the fullbacks do. This matches real
    football: the block is defined by the spine, not the width.
    """

    # Defensive importance weights per position — the "spine" matters most
    DEFENSIVE_WEIGHTS: Dict[str, float] = {
        "CB": 1.00,
        "CDM": 0.90,
        "DM": 0.90,
        "GK": 0.75,   # Sweeper keepers pull the block up
        "LB": 0.55,
        "RB": 0.55,
        "CM": 0.30,   # Only counts if sitting deep
        "CAM": 0.05,
        "LW": 0.05,
        "RW": 0.05,
        "ST": 0.05,
        "CF": 0.05,
    }

    # Positions that ALWAYS contribute to block detection
    CORE_POSITIONS: Set[str] = {"CB", "CDM", "DM", "LB", "RB", "GK"}

    # Compactness thresholds
    COMPACT_THRESHOLD: float = 0.55
    SPREAD_THRESHOLD: float = 0.35

    @classmethod
    def detect(
        cls,
        opponent_positions: Dict[str, Tuple[float, float]],
        opponent_positions_map: Dict[str, str],  # name -> position label
        attacks_right: bool = True,
        minute: int = 0,
    ) -> BlockShape:
        """
        Build a BlockShape from the opponent's live positions.

        Args:
            opponent_positions: {player_name: (x, y)}
            opponent_positions_map: {player_name: position_label}
            attacks_right: True if WE attack right (opponent defends right goal at x=105)
            minute: match minute for timestamp

        Returns:
            BlockShape representing the opponent's defensive block
        """
        # Collect weighted positions
        weighted_x, weighted_y, total_weight = 0.0, 0.0, 0.0
        player_coords: List[Tuple[float, float, float, str]] = []  # x, y, weight, pos

        for name, (x, y) in opponent_positions.items():
            pos = opponent_positions_map.get(name, "CM")
            weight = cls.DEFENSIVE_WEIGHTS.get(pos, 0.30)

            # CM only contributes if they're sitting deep (defensive third or deep midfield)
            if pos == "CM":
                if attacks_right:
                    # Opponent defends right side; deep means x > 60
                    if x < 60:
                        continue
                else:
                    if x > 45:
                        continue

            # CAM/LW/RW/ST almost never contribute unless they're tracking back unusually deep
            if pos in ("CAM", "LW", "RW", "ST", "CF"):
                if attacks_right:
                    if x < 75:  # Not tracking back deep enough
                        continue
                else:
                    if x > 30:
                        continue

            player_coords.append((x, y, weight, pos))
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        if total_weight < 0.01 or len(player_coords) < 3:
            # Not enough data — return a broken/neutral block
            return BlockShape(
                center_x=CENTER_X,
                center_y=CENTER_Y,
                compactness=0.0,
                width=30.0,
                depth=30.0,
                shape_type=BlockShapeType.BROKEN,
                minute=minute,
            )

        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight

        # Compute compactness from pairwise distances of CORE contributors
        core_coords = [(x, y) for x, y, w, p in player_coords if p in cls.CORE_POSITIONS]
        if len(core_coords) < 2:
            core_coords = [(x, y) for x, y, w, p in player_coords]

        compactness = cls._compute_compactness(core_coords)
        width, depth = cls._compute_spread(player_coords)
        shape_type = cls._classify_shape(compactness, depth, center_x, attacks_right)

        # Discover channels
        channels = cls._discover_channels(
            center_x, center_y, width, depth, compactness,
            player_coords, attacks_right
        )

        # Zone density
        zone_density = cls._zone_density(player_coords, attacks_right)

        contributing = set(p for _, _, _, p in player_coords)

        return BlockShape(
            center_x=center_x,
            center_y=center_y,
            compactness=compactness,
            width=width,
            depth=depth,
            shape_type=shape_type,
            channels=channels,
            zone_density=zone_density,
            contributing_positions=contributing,
            minute=minute,
        )

    @classmethod
    def _compute_compactness(cls, coords: List[Tuple[float, float]]) -> float:
        """
        Compactness = 1 - (avg_pairwise_distance / 35m), clamped 0-1.
        A perfectly compact block has all players within ~15m of each other.
        A spread block has players 40m+ apart.
        """
        if len(coords) < 2:
            return 0.0

        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                distances.append(math.hypot(dx, dy))

        avg_dist = sum(distances) / len(distances)
        # 15m avg = fully compact (1.0), 50m avg = fully spread (0.0)
        compactness = 1.0 - (avg_dist - 15.0) / 35.0
        return max(0.0, min(1.0, compactness))

    @classmethod
    def _compute_spread(
        cls, player_coords: List[Tuple[float, float, float, str]]
    ) -> Tuple[float, float]:
        """Return (width, depth) as standard deviations along y and x."""
        if len(player_coords) < 2:
            return 20.0, 20.0

        xs = [x for x, _, _, _ in player_coords]
        ys = [y for _, y, _, _ in player_coords]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)

        return math.sqrt(var_y) * 2, math.sqrt(var_x) * 2  # 2*SD ≈ 95% spread

    @classmethod
    def _classify_shape(
        cls, compactness: float, depth: float, center_x: float, attacks_right: bool
    ) -> BlockShapeType:
        """Classify the block based on compactness, depth, and center position."""
        if compactness < 0.25:
            return BlockShapeType.BROKEN

        if compactness < cls.SPREAD_THRESHOLD:
            return BlockShapeType.SPREAD_PRESS

        # For compact blocks, depth and center tell us if it's low or high
        if attacks_right:
            # Opponent defends right goal (x=105). Deep block = center_x > 80
            if center_x > 82 and depth < 20:
                return BlockShapeType.LOW_BLOCK
            if center_x < 65 and depth > 20:
                return BlockShapeType.HIGH_LINE
        else:
            # Opponent defends left goal (x=0)
            if center_x < 23 and depth < 20:
                return BlockShapeType.LOW_BLOCK
            if center_x > 40 and depth > 20:
                return BlockShapeType.HIGH_LINE

        return BlockShapeType.COMPACT_MID_BLOCK

    @classmethod
    def _discover_channels(
        cls,
        center_x: float, center_y: float,
        width: float, depth: float, compactness: float,
        player_coords: List[Tuple[float, float, float, str]],
        attacks_right: bool,
    ) -> List[Channel]:
        """
        Discover natural passing channels by looking for LOW opponent density.
        """
        channels: List[Channel] = []

        # Helper: count opponents in a rectangular zone
        def density_in_zone(x1, x2, y1, y2) -> float:
            count = sum(1 for x, y, _, _ in player_coords if x1 <= x <= x2 and y1 <= y <= y2)
            area = max(1.0, (x2 - x1) * (y2 - y1))
            return count / area * 100  # players per 100m²

        # ── HALF-SPACES ──────────────────────────────────────────────
        # The channels between CB and FB, or between DM and CM
        # These are the Modric/Tanaka orbital zones

        if attacks_right:
            # We attack right; opponent block is on their right (our left-to-right)
            # Left half-space (bottom side of block)
            left_hs_x1, left_hs_x2 = center_x - 15, center_x + 5
            left_hs_y1, left_hs_y2 = max(0, center_y - width - 5), center_y - 5
            left_hs_density = density_in_zone(left_hs_x1, left_hs_x2, left_hs_y1, left_hs_y2)

            right_hs_x1, right_hs_x2 = center_x - 15, center_x + 5
            right_hs_y1, right_hs_y2 = center_y + 5, min(PITCH_WIDTH, center_y + width + 5)
            right_hs_density = density_in_zone(right_hs_x1, right_hs_x2, right_hs_y1, right_hs_y2)
        else:
            left_hs_x1, left_hs_x2 = center_x - 5, center_x + 15
            left_hs_y1, left_hs_y2 = max(0, center_y - width - 5), center_y - 5
            left_hs_density = density_in_zone(left_hs_x1, left_hs_x2, left_hs_y1, left_hs_y2)

            right_hs_x1, right_hs_x2 = center_x - 5, center_x + 15
            right_hs_y1, right_hs_y2 = center_y + 5, min(PITCH_WIDTH, center_y + width + 5)
            right_hs_density = density_in_zone(right_hs_x1, right_hs_x2, right_hs_y1, right_hs_y2)

        # Accessibility = 1 - normalized density (0 = crowded, 1 = completely open)
        max_expected_density = 0.8  # players per 100m² threshold for "open"

        channels.append(Channel(
            name="left_half_space",
            center=((left_hs_x1 + left_hs_x2) / 2, (left_hs_y1 + left_hs_y2) / 2),
            width=12.0,
            accessibility=max(0.0, 1.0 - left_hs_density / max_expected_density),
            description="Channel between opponent CB and LB / DM and left CM",
        ))

        channels.append(Channel(
            name="right_half_space",
            center=((right_hs_x1 + right_hs_x2) / 2, (right_hs_y1 + right_hs_y2) / 2),
            width=12.0,
            accessibility=max(0.0, 1.0 - right_hs_density / max_expected_density),
            description="Channel between opponent CB and RB / DM and right CM",
        ))

        # ── WIDE CHANNELS ─────────────────────────────────────────────
        # Beyond the fullbacks — the "stretch" option
        wide_left_y = 8.0
        wide_right_y = PITCH_WIDTH - 8.0
        wide_x = center_x

        channels.append(Channel(
            name="left_wide",
            center=(wide_x, wide_left_y),
            width=10.0,
            accessibility=0.85,  # Wide is usually open unless opponent is very wide
            description="Touchline channel beyond opponent left back",
        ))

        channels.append(Channel(
            name="right_wide",
            center=(wide_x, wide_right_y),
            width=10.0,
            accessibility=0.85,
            description="Touchline channel beyond opponent right back",
        ))

        # ── DEEP RECYCLE ──────────────────────────────────────────────
        # Behind our own midfield, in front of our defence — the "reset" channel
        if attacks_right:
            deep_x = max(15.0, center_x - 25)
        else:
            deep_x = min(90.0, center_x + 25)

        channels.append(Channel(
            name="deep_recycle",
            center=(deep_x, CENTER_Y),
            width=20.0,
            accessibility=0.90,
            description="Deep central channel for possession reset",
        ))

        # ── FORWARD POCKETS (only if block is deep/compact) ───────────
        if compactness > 0.5:
            if attacks_right:
                fwd_x = center_x - 12
            else:
                fwd_x = center_x + 12
            channels.append(Channel(
                name="forward_pocket",
                center=(fwd_x, CENTER_Y),
                width=8.0,
                accessibility=0.40,  # Tight, but valuable
                description="Small gap just in front of opponent back line",
            ))

        return channels

    @classmethod
    def _zone_density(
        cls,
        player_coords: List[Tuple[float, float, float, str]],
        attacks_right: bool,
    ) -> Dict[str, float]:
        """Count opponent players in coarse zones."""
        zones = {"six_yard": 0, "box": 0, "edge": 0, "midfield": 0, "deep": 0}
        for x, y, _, _ in player_coords:
            if attacks_right:
                # Opponent defends right (x=105)
                if x >= 99:
                    zones["six_yard"] += 1
                elif x >= 83:
                    zones["box"] += 1
                elif x >= 70:
                    zones["edge"] += 1
                elif x >= 35:
                    zones["midfield"] += 1
                else:
                    zones["deep"] += 1
            else:
                if x <= 6:
                    zones["six_yard"] += 1
                elif x <= 22:
                    zones["box"] += 1
                elif x <= 35:
                    zones["edge"] += 1
                elif x <= 70:
                    zones["midfield"] += 1
                else:
                    zones["deep"] += 1
        return zones


# ─────────────────────────────────────────────
# BLOCK NAVIGATION ENGINE
# ─────────────────────────────────────────────

class BlockNavigationEngine:
    """
    Scores candidate passes by how well they navigate around the opponent's block.

    This is the core of the "orbital" passing pattern. A pass that arcs around
    the opponent's compact center gets a bonus. A pass that drives straight
    through the heart of their block gets a penalty (unless the passer has
    elite vision and is deliberately threading a needle).
    """

    # Maximum orbital bonus/penalty range
    ORBITAL_RANGE: float = 0.70   # pass weight can shift ±70%

    @classmethod
    def pass_orbital_score(
        cls,
        passer_pos: Tuple[float, float],
        receiver_pos: Tuple[float, float],
        block: BlockShape,
        passer_vision: float = 60.0,
        passer_risk_tolerance: float = 0.5,  # 0 = safe, 1 = aggressive
        passer_position: str = "",            # position label e.g. "CM", "CAM", "CDM"
        is_progressive: bool = False,
    ) -> float:
        """
        Compute a multiplier (0.45 – 1.55) for this pass based on block navigation.

        Args:
            passer_pos: (x, y) of the player with the ball
            receiver_pos: (x, y) of the candidate receiver
            block: the opponent's BlockShape
            passer_vision: 0-100, higher vision = better at finding orbital routes
            passer_risk_tolerance: 0.0-1.0 (derive from 1 - composure/100 for realism)
            passer_position: the passer's position label — CMs/CDMs get a stronger
                orbital preference because recycling IS their job; CAMs are threaders
            is_progressive: True if this is a forward pass into the final third

        Returns:
            float multiplier to apply to the pass's existing weight
        """
        # If block is broken or spread, orbital navigation doesn't apply
        if block.shape_type in (BlockShapeType.BROKEN, BlockShapeType.SPREAD_PRESS):
            return 1.0

        ax, ay = passer_pos
        bx, by = receiver_pos
        cx, cy = block.center_x, block.center_y

        # Vector from passer to receiver
        ab_x = bx - ax
        ab_y = by - ay
        ab_len = math.hypot(ab_x, ab_y)
        if ab_len < 1.0:
            return 1.0

        # Vector from passer to block center
        ac_x = cx - ax
        ac_y = cy - ay

        # Projection factor: where is the block center relative to the pass segment?
        # t = 0 → block center is at passer
        # t = 1 → block center is at receiver
        # 0 < t < 1 → block center is BETWEEN passer and receiver
        t = (ac_x * ab_x + ac_y * ab_y) / (ab_len ** 2)

        # Perpendicular distance from block center to the pass line
        # d = |AB × AC| / |AB|  (2D cross product magnitude)
        cross = abs(ab_x * ac_y - ab_y * ac_x)
        perp_dist = cross / ab_len

        # Block radius: the "danger zone" around the block center
        block_radius = block.radius

        # ── ORBITAL SCORE COMPUTATION ───────────────────────────────
        #
        # Core insight: reward passes that keep the ball away from the opponent's
        # defensive density. The reward depends on:
        #   1. How close the pass comes to the block center (perp_dist)
        #   2. Whether the block center is actually between passer and receiver (t)
        #   3. How compact the block is (compact blocks are more important to avoid)
        #   4. The passer's vision (better vision = better at finding orbital routes)

        # Base risk: how much does this pass intersect the block?
        # risk = 1.0 if pass goes directly through block center
        # risk = 0.0 if pass is far from block center
        proximity_risk = math.exp(-(perp_dist ** 2) / (2 * (block_radius * 0.6) ** 2))

        # If block center is NOT between passer and receiver, risk is lower
        # (the pass starts/ends near the block but doesn't traverse it)
        traversal_factor = 1.0 if 0.0 <= t <= 1.0 else 0.35
        risk = proximity_risk * traversal_factor

        # Compact blocks are more dangerous to pass through
        compactness_boost = 0.5 + 0.5 * block.compactness
        risk *= compactness_boost

        # Vision reduces the penalty for "through" passes — elite vision means
        # the passer CAN thread the needle, so we don't penalize as harshly
        vision_factor = 0.6 + 0.4 * (1.0 - passer_vision / 100.0)
        risk *= vision_factor

        # Risk tolerance: aggressive passers are willing to take block-on passes
        risk *= (1.0 - passer_risk_tolerance * 0.4)

        # Clamp risk
        risk = max(0.0, min(1.0, risk))

        # Convert risk to score: low risk = high score (bonus for going around)
        # High risk = low score (penalty for going through)
        score = 1.0 + cls.ORBITAL_RANGE * (0.5 - risk)

        # Progressive passes into the final third against a compact block:
        # slightly reduce the orbital bonus because SOME forward penetration
        # is necessary — you can't just orbit forever
        if is_progressive and block.is_compact:
            score = 1.0 + (score - 1.0) * 0.7

        # ── CM / CDM ORBITAL AMPLIFIER ──────────────────────────────
        # Central midfielders and holding midfielders are the block-orbiting
        # specialists. Their job IS to keep the ball moving around the edges
        # of the opponent shape (Modric, Tanaka, Rodri). CAMs are threaders;
        # wingers are stretchers. Only amplify for the recycling positions.
        if passer_position in ("CM", "CDM"):
            score = 1.0 + (score - 1.0) * 1.3

        return max(0.45, min(1.55, score))

    @classmethod
    def channel_bonus(
        cls,
        receiver_pos: Tuple[float, float],
        block: BlockShape,
        channel_names: Optional[List[str]] = None,
    ) -> float:
        """
        Bonus multiplier (1.0 – 1.25) if the receiver is positioned in an
        identified open channel. This rewards passes to the half-spaces and
        wide channels that naturally exist around a compact block.
        """
        if not block.channels:
            return 1.0

        rx, ry = receiver_pos
        best_bonus = 1.0

        target_channels = channel_names or [ch.name for ch in block.channels]

        for ch in block.channels:
            if ch.name not in target_channels:
                continue

            cx, cy = ch.center
            dist = math.hypot(rx - cx, ry - cy)

            # If receiver is inside this channel's zone, give a bonus
            # scaled by channel accessibility
            if dist < ch.width:
                bonus = 1.0 + 0.20 * ch.accessibility * (1.0 - dist / ch.width)
                best_bonus = max(best_bonus, bonus)

        return best_bonus

    @classmethod
    def recycle_tendency_score(
        cls,
        passer: "PlayerProfile",
        receiver: "PlayerProfile",
        block: BlockShape,
        receiver_pos: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Volume passers (Modric, Tanaka) get a bonus for passes that keep the
        ball moving around the block rather than forcing through it.

        This is a SEPARATE multiplier from orbital_score — it captures the
        *behavioral* preference of high-volume midfielders to recycle.

        Args:
            receiver_pos: the receiver's LIVE (x, y) from the PositionEngine.
                Without it the channel check cannot fire (a dummy coordinate
                is never inside a channel, which silently reduced this whole
                bonus to 1.0).
        """
        # Only apply against compact blocks
        if not block.is_compact:
            return 1.0

        # Identify "volume passer" profile
        vision = getattr(passer.dna.mental, "vision", 60.0) if hasattr(passer, "dna") else 60.0
        short_pass = getattr(passer.dna.passing, "short_passing", 60.0) if hasattr(passer, "dna") else 60.0
        work_rate = getattr(passer.dna.mental, "work_rate", 60.0) if hasattr(passer, "dna") else 60.0

        volume_score = (vision + short_pass + work_rate) / 3.0
        if volume_score < 60:
            return 1.0  # Not a volume passer

        # Volume passers prefer:
        #   - Lateral passes (same x-band, different y)
        #   - Backward passes that reset around the block
        #   - Passes to teammates who are also in orbital positions
        # They DIS-prefer:
        #   - Forward passes directly into the block center
        #   - Long diagonals through density

        # This is a subtle nudge, not a dramatic shift
        volume_factor = max(0.0, min(1.0, (volume_score - 70) / 30.0))  # 0.0 – 1.0

        # If receiver is sitting in a half-space / deep-recycle channel, boost
        channel_mult = 1.0
        if receiver_pos is not None:
            channel_mult = cls.channel_bonus(
                receiver_pos,
                block,
                channel_names=["left_half_space", "right_half_space", "deep_recycle"]
            )

        # The bonus is small but consistent — this is what produces 75+ passes
        return 1.0 + 0.20 * volume_factor * (channel_mult - 1.0)


# ─────────────────────────────────────────────
# HALF-SPACE MAGNET
# ─────────────────────────────────────────────

class HalfSpaceMagnet:
    """
    Pulls CAMs and CMs toward half-spaces when the opponent has a compact block.

    Real CAMs (Ødegaard) don't stand in the center of the pitch in front of a
    compact 4-4-2 block — they drift to the half-space where they can see both
    the block AND the channels around it. This module computes that drift target.
    """

    # Positions that respond to half-space magnetism
    MAGNET_POSITIONS: Set[str] = {"CAM", "CM", "CF"}

    # How strong the pull is (0.0 – 1.0, blended into drift)
    PULL_STRENGTH: float = 0.8

    @classmethod
    def drift_target(
        cls,
        player: "PlayerProfile",
        home_pos: Tuple[float, float],
        block: Optional[BlockShape],
        current_pos: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Compute a drift target that may pull the player toward a half-space.

        Args:
            player: the player whose drift we're computing
            home_pos: their formation home position (x, y)
            block: opponent's BlockShape (None = no effect)
            current_pos: their current position (x, y)

        Returns:
            (target_x, target_y) — may be the original home_pos or a half-space
        """
        if block is None or player.position not in cls.MAGNET_POSITIONS:
            return home_pos

        # Only pull against compact blocks
        if not block.is_compact:
            return home_pos

        # Find the best half-space channel
        best_channel = None
        best_dist = float("inf")

        for ch in block.channels:
            if "half_space" not in ch.name:
                continue
            if ch.accessibility < 0.15:
                continue  # Don't pull into a closed channel

            cx, cy = ch.center
            dist = math.hypot(current_pos[0] - cx, current_pos[1] - cy)
            if dist < best_dist:
                best_dist = dist
                best_channel = ch

        if best_channel is None:
            return home_pos

        # Compute the pull target: blend home position toward the channel center
        # We don't pull ALL the way to the channel — just enough to create
        # the half-space clustering seen in real pass maps
        hx, hy = home_pos
        cx, cy = best_channel.center

        # But: if the player is already in their preferred half-space side,
        # don't pull them across the pitch. CAMs have a preferred side based
        # on their natural tendency or the team's build-up side.
        # Simple heuristic: stay on the same side of center as home position
        if hy < CENTER_Y:
            # Player's home is on the left side — prefer left half-space
            left_ch = block.channel_named("left_half_space")
            if left_ch and left_ch.accessibility >= 0.3:
                cx, cy = left_ch.center
        else:
            right_ch = block.channel_named("right_half_space")
            if right_ch and right_ch.accessibility >= 0.3:
                cx, cy = right_ch.center

        # Blend: stronger pull when block is more compact
        blend = cls.PULL_STRENGTH * block.compactness

        target_x = hx + (cx - hx) * blend * 0.5  # Less x-pull than y-pull
        target_y = hy + (cy - hy) * blend

        return (target_x, target_y)

    @classmethod
    def winger_inversion_target(
        cls,
        player: "PlayerProfile",
        home_pos: Tuple[float, float],
        block: Optional[BlockShape],
    ) -> Optional[Tuple[float, float]]:
        """
        When opponent block is very compact, wingers may INVERT — cut inside
        to the half-space rather than hugging the touchline. This is what
        Saka, Foden, and Martinelli do against low blocks.

        Returns None if inversion is not appropriate.
        """
        if player.position not in ("LW", "RW") or block is None:
            return None

        if block.shape_type not in (BlockShapeType.COMPACT_MID_BLOCK, BlockShapeType.LOW_BLOCK):
            return None

        # Only invert if the winger has good vision and dribbling
        vision = getattr(player.dna.mental, "vision", 50.0) if hasattr(player, "dna") else 50.0
        dribbling = getattr(player.dna.technical, "dribbling", 50.0) if hasattr(player, "dna") else 50.0

        if vision < 65 or dribbling < 65:
            return None

        # Target: the half-space on their side, slightly deeper than a CAM
        if player.position == "LW":
            ch = block.channel_named("left_half_space")
        else:
            ch = block.channel_named("right_half_space")

        if ch is None or ch.accessibility < 0.20:
            return None

        cx, cy = ch.center
        # Wingers invert to slightly wider than the half-space center
        if player.position == "LW":
            return (cx, cy + 5)
        else:
            return (cx, cy - 5)


# ─────────────────────────────────────────────
# INTEGRATION HELPERS
# ─────────────────────────────────────────────

class BlockAwarenessIntegration:
    """
    One-stop helper for wiring BlockAwareness into existing PLOFA modules.
    Copy the integration snippets from this class into the relevant files.
    """

    # ── INTEGRATION POINT 1: MatchEngine ────────────────────────────
    # In MatchEngine.__init__() or per-minute update:
    #
    #   self.block_detector = BlockDetector()
    #   self.home_block: Optional[BlockShape] = None
    #   self.away_block: Optional[BlockShape] = None
    #
    # In MatchEngine._simulate_minute() or per-minute loop:
    #
    #   # Update block shapes from live positions
    #   home_positions = {p.name: self.position_engine.get_position(p.name)
    #                     for p in self.home_players}
    #   home_pos_map = {p.name: p.position for p in self.home_players}
    #   self.home_block = BlockDetector.detect(
    #       home_positions, home_pos_map, attacks_right=False, minute=minute
    #   )
    #   # (and similarly for away_block with attacks_right=True)
    #
    # ── INTEGRATION POINT 2: PositionEngine.DriftEngine ─────────────
    # In PositionEngine._compute_drift_target() or wherever home positions
    # are updated per minute:
    #
    #   # After computing the base drift target, apply half-space magnetism
    #   if block_awareness is not None:
    #       opponent_block = block_awareness.get_block_for(team_name)
    #       new_target = HalfSpaceMagnet.drift_target(
    #           player, home_target, opponent_block, current_pos
    #       )
    #       # Blend with existing drift (don't override completely)
    #       home_target = new_target
    #
    # ── INTEGRATION POINT 3: EventChain._pick_receiver ──────────────
    # In EventChain._pick_receiver(), inside label_weight(), AFTER the
    # existing receive_option_quality multiplier but BEFORE the final return:
    #
    #   # ── CHECKPOINT 29: BLOCK NAVIGATION ───────────────────────
    #   if block_shape is not None and position_engine is not None:
    #       px, py = position_engine.get_position(passer.name)
    #       rx, ry = position_engine.get_position(p.name)
    #       orbital = BlockNavigationEngine.pass_orbital_score(
    #           (px, py), (rx, ry), block_shape,
    #           passer_vision=passer.dna.mental.vision,
    #           passer_risk_tolerance=passer.dna.tendencies.risk_tolerance,
    #           is_progressive=(rx > px + 8 if attacks_right else rx < px - 8)
    #       )
    #       base *= orbital
    #
    #       # Channel bonus: extra reward for hitting open half-spaces
    #       channel_mult = BlockNavigationEngine.channel_bonus(
    #           (rx, ry), block_shape
    #       )
    #       base *= channel_mult
    #
    #   # Volume recycler bonus (for Modric/Tanaka types)
    #   if block_shape is not None:
    #       recycle = BlockNavigationEngine.recycle_tendency_score(
    #           passer, p, block_shape
    #       )
    #       base *= recycle
    #
    # ── INTEGRATION POINT 4: WingerBehavior ─────────────────────────
    # In winger_behavior.py, when computing winger target positions:
    #
    #   # Check if inversion is appropriate against compact block
    #   if block_shape is not None:
    #       invert_target = HalfSpaceMagnet.winger_inversion_target(
    #           player, home_pos, block_shape
    #       )
    #       if invert_target is not None and random.random() < 0.25:
    #           # 25% chance to invert against compact block
    #           target = invert_target
    #
    # ── INTEGRATION POINT 5: MatchEngine state access ───────────────
    # Block shapes need to be accessible to EventChain. The cleanest way:
    #
    #   class MatchState:
    #       ...
    #       home_block: Optional[BlockShape] = None
    #       away_block: Optional[BlockShape] = None
    #
    #   Then in EventChain methods that receive match_state:
    #       block = match_state.home_block if attacking_team == home_team else match_state.away_block
    #
    #   Or pass it explicitly through the call chain.
    pass


# ─────────────────────────────────────────────
# EXAMPLE / SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a compact mid-block (opponent sitting in a 4-4-2, defending right goal)
    opponent_positions = {
        "Opp GK": (95.0, 34.0),
        "Opp LB": (78.0, 55.0),
        "Opp CB1": (82.0, 28.0),
        "Opp CB2": (82.0, 40.0),
        "Opp RB": (78.0, 13.0),
        "Opp CDM": (72.0, 34.0),
        "Opp CM1": (65.0, 25.0),
        "Opp CM2": (65.0, 43.0),
        "Opp LW": (60.0, 55.0),
        "Opp RW": (60.0, 13.0),
        "Opp ST": (58.0, 34.0),
    }
    opponent_pos_map = {
        "Opp GK": "GK", "Opp LB": "LB", "Opp CB1": "CB", "Opp CB2": "CB",
        "Opp RB": "RB", "Opp CDM": "CDM", "Opp CM1": "CM", "Opp CM2": "CM",
        "Opp LW": "LW", "Opp RW": "RW", "Opp ST": "ST",
    }

    block = BlockDetector.detect(opponent_positions, opponent_pos_map, attacks_right=True, minute=30)

    print(f"Block center: ({block.center_x:.1f}, {block.center_y:.1f})")
    print(f"Compactness: {block.compactness:.2f}")
    print(f"Shape: {block.shape_type.name}")
    print(f"Radius: {block.radius:.1f}m")
    print(f"Channels:")
    for ch in block.channels:
        print(f"  {ch.name}: ({ch.center[0]:.1f}, {ch.center[1]:.1f}) accessibility={ch.accessibility:.2f}")

    # Test orbital scores for different passes
    print("\n--- Orbital Pass Scores ---")
    passer = (45.0, 34.0)  # Our CM in center circle

    test_passes = [
        ("Through center (BAD)", (65.0, 34.0)),      # Straight through block
        ("Left half-space (GOOD)", (60.0, 22.0)),    # Around left side
        ("Right half-space (GOOD)", (60.0, 46.0)),   # Around right side
        ("Wide left (GOOD)", (55.0, 8.0)),           # Touchline
        ("Deep recycle (NEUTRAL)", (25.0, 34.0)),    # Back to defense
    ]

    for name, recv in test_passes:
        score = BlockNavigationEngine.pass_orbital_score(
            passer, recv, block, passer_vision=75.0, passer_risk_tolerance=0.3
        )
        print(f"  {name}: {score:.2f}x")

    # Test half-space magnet
    print("\n--- Half-Space Magnet ---")
    from dataclasses import dataclass as dc
    @dc
    class FakePlayer:
        position: str
        name: str = "Test"
    class FakeDNA:
        pass
    FakePlayer.dna = FakeDNA()

    cam = FakePlayer("CAM")
    home = (55.0, 34.0)
    current = (55.0, 34.0)
    target = HalfSpaceMagnet.drift_target(cam, home, block, current)
    print(f"CAM home={home} -> drift_target={target}")

    cm = FakePlayer("CM")
    home_cm = (45.0, 25.0)
    target_cm = HalfSpaceMagnet.drift_target(cm, home_cm, block, home_cm)
    print(f"CM home={home_cm} -> drift_target={target_cm}")