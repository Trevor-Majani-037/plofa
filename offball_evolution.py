"""Off-ball conscience evolution — GA for the binary press/hold gate.

The conscience net (OffBallBrain, 24->32->32->1) replaces the per-position
Bernoulli press trigger in MatchEngine._offball_move_player.  Evolution is
grounded in real-match data: the OffBallSurrogate (offball_probe.py) maps
(position, situation bucket, decision) -> expected defensive-episode score,
learned from collected press decisions against the heuristic baseline.

Synthetic fitness resembles brain_evolution.synthetic_fitness but for the
binary decision: for each random-but-realistic off-ball state it asks the
net's press probability p, picks the argmax decision, and rewards the
surrogate's expected success of that decision weighted by confidence (a net
that commits firmly to the good choice scores higher).  An indecision
penalty applies near p ~ 0.5.

Outcome state generation mirrors what the live engine actually presents at
the decision point: the player is defending (team_possession=0), ball within
~12.5 m of the runner, with teammates/defenders around the ball and runner.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain_sensors import extract_offball_sensors
from football_brain import OffBallBrain, INPUT_SIZE
from perception import (get_perception_config, _mental_scale,
                        _bearing_diff, _forward_angle)


# ─────────────────────────────────────────────────────────────
# SYNTHETIC OFF-BALL STATE GENERATION
# ─────────────────────────────────────────────────────────────

def random_offball_state(rng: random.Random, position: str) -> Dict[str, Any]:
    """Generate a random-but-realistic state for a DEFENDING off-ball player.

    Ball sits within the chase-trigger radius of the runner.  All coordinates
    are metres on a 105x68 pitch; the runner forwards are measured from the
    ball, defenders cluster around the ball carrier (the runner presses a
    carrier, not a loose ball).
    """
    pos = position
    # Home-zone x for the runner (defending shape).
    if pos in ("ST", "CF", "LW", "RW", "CAM"):
        rx = rng.uniform(55, 100)
    elif pos in ("CM", "CDM"):
        rx = rng.uniform(35, 75)
    else:  # CB, LB, RB, GK-outfield
        rx = rng.uniform(15, 60)
    ry = rng.uniform(5, 63)

    attacks_right = rng.random() < 0.5
    # Ball within trigger radius 12.5 m of the runner.  ~40% of the time the
    # ball is BEHIND the runner (the "arrived late / ball leaked behind"
    # geometry) so the conscience actually trains on the phantom-chase
    # population the diagnostic measured — a ball the runner presses without
    # seeing (see perceive_ball: cone covers only +/-fov/2, bloat grows
    # uncertainty ~8m/sim-minute for stale reads).  Without this oversample
    # the cert axis is <9% of states and the 25th input is too sparse to
    # learn.
    direction = 1.0 if attacks_right else -1.0
    if rng.random() < 0.40:
        bx = rx - direction * rng.uniform(0, 12.0)
    else:
        bx = rx + direction * rng.uniform(0, 12.0)
    by = ry + rng.uniform(-8, 8)
    bx = max(1.0, min(104.0, bx))
    by = max(1.0, min(67.0, by))

    minute = rng.uniform(5, 90)
    game_state_names = ["LEVEL", "HOME_AHEAD_1", "AWAY_AHEAD_1",
                        "HOME_CRUISE", "AWAY_CRUISE", "LEVEL"]
    game_state = type("GS", (), {"name": rng.choice(game_state_names)})()

    # Teammates: already goal-side of ball or goal-side of runner.
    n_teammates = rng.randint(3, 8)
    teammates = []
    for _ in range(n_teammates):
        tx = bx + (rng.uniform(-15, 5) if attacks_right else rng.uniform(-5, 15))
        tx = max(0, min(105, tx))
        ty = max(0, min(68, by + rng.uniform(-20, 20)))
        teammates.append((tx, ty))

    # Defenders (opposition ball carrier's team): cluster around the ball.
    n_defenders = rng.randint(1, 5)
    defenders = []
    for _ in range(n_defenders):
        dx = max(0, min(105, bx + rng.uniform(-8, 8)))
        dy = max(0, min(68, by + rng.uniform(-12, 12)))
        defenders.append((dx, dy))

    cfg = get_perception_config()
    st = {
        "rx": rx, "ry": ry, "bx": bx, "by": by,
        "attacks_right": attacks_right, "minute": minute,
        "game_state": game_state, "teammates": teammates, "defenders": defenders,
    }
    px, py, sigma = _honest_ball_for(st, rng, cfg) if cfg.ball_vision \
        else (st["bx"], st["by"], 0.0)
    st["px"], st["py"], st["sigma"] = px, py, sigma
    return st


def _honest_ball_for(st: Dict[str, Any], rng: random.Random,
                     cfg: Any) -> Tuple[float, float, float]:
    """The PERCEIVED ball + running uncertainty (sigma, m) for a synthetic
    state, mirroring ball_vision.perceive_ball against a DNA-less runner.

    The evolution distribution now matches deployment: a runner inside his
    view cone/range reads the TRUE ball with a small distance noise; a blind
    runner reads his "last known" with uncertainty growing at
    ball_stale_growth per out-of-sight minute (drawn 1..4 for the dense
    blind population the diagnostic actually measured — ~66% of reads).

    Returns ``(px, py, sigma)``.  sigma=0 for fresh reads (certainty 1).
    """
    rx, ry = st["rx"], st["ry"]
    attacks_right = st.get("attacks_right", True)

    power, acc = _mental_scale(_NO_DNA_VIEW)
    radius = float(cfg.ball_radius) * (0.5 + 0.5 * power)
    dist = float(np.hypot(st["bx"] - rx, st["by"] - ry))
    fov_rad = np.deg2rad(float(cfg.ball_fov_deg))
    bearing = _bearing_diff(rx, ry, st["bx"], st["by"],
                            _forward_angle(attacks_right))
    visible = (dist <= radius) and abs(bearing) <= fov_rad / 2.0

    if visible:
        sigma = float(cfg.ball_noise) * (1.0 - acc) * (0.4 + 0.6 * dist / max(radius, 1.0))
        return st["bx"], st["by"], sigma

    sigma = float(cfg.ball_noise) * (1.0 - acc) + \
        float(cfg.ball_stale_growth) * rng.uniform(1.0, 4.0)
    px = st["bx"] + rng.gauss(0.0, sigma)
    py = st["by"] + rng.gauss(0.0, sigma)
    return px, py, sigma


class _NoDNAView:
    """Synthetic runner with no DNA — mental helpers default to 60."""

    dna = None


_NO_DNA_VIEW = _NoDNAView()


class _DummyPositionEngine:
    def __init__(self, positions: Dict[str, Tuple[float, float]]):
        self.positions = positions
    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def _coords_to_players(coords: List[Tuple[float, float]], prefix: str):
    class FakePlayer:
        def __init__(self, name, position):
            self.name = name
            self.position = position
    return [FakePlayer(f"{prefix}{i}", "CM") for i, (x, y) in enumerate(coords)]


def _sensors_for(state: Dict[str, Any], position: str) -> np.ndarray:
    t_coords = state["teammates"]
    d_coords = state["defenders"]
    all_names = {}
    for i, (tx, ty) in enumerate(t_coords):
        all_names[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(d_coords):
        all_names[f"d{i}"] = (dx, dy)
    pe = _DummyPositionEngine(all_names)
    teammates = _coords_to_players(t_coords, "t")
    defenders = _coords_to_players(d_coords, "d")

    class _Runner:
        def __init__(self):
            self.position = position
            self.dna = None

    return extract_offball_sensors(
        _Runner(), state["rx"], state["ry"],
        state["px"], state["py"],
        teammates=teammates, defenders=defenders, position_engine=pe,
        attacks_right=state["attacks_right"], game_state=state["game_state"],
        minute=state["minute"], ball_sigma=state.get("sigma", 0.0),
    )


# ─────────────────────────────────────────────────────────────
# SYNTHETIC FITNESS — evidence-gated conscience reward
# ─────────────────────────────────────────────────────────────

# Heuristic off-ball press probabilities (match_engine._PRESS_PROB).  The
# conscience may ONLY override these where decision-local data supports it;
# elsewhere it is rewarded for reproducing the heuristic frequency, so an
# always-press collapse is impossible.
_HEUR_PRESS_PROB = {
    "GK": 0.0, "CB": 0.80, "LB": 0.80, "RB": 0.80,
    "CDM": 0.75, "CM": 0.75, "CAM": 0.60,
    "LW": 0.45, "RW": 0.45, "ST": 0.50, "CF": 0.50,
}

# Both decisions need at least this many real samples in the cell before the
# conscience is allowed to override the heuristic (evidence gate).
_EVIDENCE_MIN = 5


def synthetic_conscience_fitness(
    brain: OffBallBrain,
    player_position: str,
    n_states: int = 400,
    seed: int = 0,
    surrogate: Optional[Any] = None,
) -> float:
    """Score a conscience brain over random off-ball defending states.

    Three weighted terms, all in [0, 1]:

      1. Evidence term (40%): per-state surrogate credit, shaped by
         CERTAINTY.  The certainty slot (1/(1+sigma)) tells the conscience
         how sure the runner is, so evidence credit for PRESSING is gated on
         knowing where the ball is:
             press credit  ~ expected_success(press)
                           * clamp((cert - 0.15)/0.4)   (blind -> ~0)
             hold credit   ~ expected_success(hold) * (0.8 + 0.2*cert)
         (pressing blind forfeits the evidence credit; holding blind is safe.)

      2. Calibration term (30%): the DECISION frequency over all states must
         reproduce the calibrated heuristic press rate:
             1 - |P(press) - _HEUR_PRESS_PROB[pos]|
         (this replaces the old logit-mean penalty, which silently let nets
         collapse to always/never while averaging p == heur).

      3. Blind-cap term (30%): blind reads (cert < 0.35) must NOT press at
         the omnisscient base rate — the phantom-chase artifact this whole
         seam exists to kill.  Penalize every blind press above ~60% of the
         calibrated rate:
             1 - max(0, P(press | blind) - 0.6 * _HEUR_PRESS_PROB[pos])

    Without a surrogate the evidence term degrades to the neutral 0.5.
    """
    rng = random.Random(seed)
    heur = _HEUR_PRESS_PROB.get(player_position, 0.6)
    heur = max(0.05, heur)
    evidence_rewards: List[float] = []
    decision_rows: List[Tuple[float, float]] = []  # (cert, press?1:0)

    for _ in range(n_states):
        st = random_offball_state(rng, player_position)
        sensors = _sensors_for(st, player_position)
        p = brain.forward(sensors)
        cert_raw = st.get("sigma", 0.0)
        cert = 1.0 if not cert_raw else 1.0 / (1.0 + cert_raw)
        decision = 1.0 if p > 0.5 else 0.0

        if surrogate is not None and surrogate.is_evidence_cell(
            sensors, player_position, min_both=_EVIDENCE_MIN):
            reward = surrogate.expected_success(
                sensors, "press" if decision else "hold", player_position)
            if decision:
                reward = reward * max(0.0, min(1.0, (cert - 0.15) / 0.4))
            else:
                reward = reward * (0.8 + 0.2 * cert)
            evidence_rewards.append(reward)
        elif surrogate is None:
            evidence_rewards.append(0.5)  # evidence term degrades to neutral
        # silent cells: no per-state reward; the batch terms handle them
        decision_rows.append((cert, decision))

    if not decision_rows:
        return 0.5

    ev_mean = statistics.mean(evidence_rewards) if evidence_rewards else 0.5
    freq_all = sum(d for _, d in decision_rows) / len(decision_rows)
    calib = 1.0 - abs(freq_all - heur)

    blind_rows = [d for c, d in decision_rows if c < 0.35]
    freq_blind = sum(blind_rows) / len(blind_rows) if blind_rows else 0.0
    blind_cap = 0.6 * heur
    blind_term = 1.0 - max(0.0, freq_blind - blind_cap)

    score = 0.4 * ev_mean + 0.3 * calib + 0.3 * blind_term
    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────
# POPULATION EVOLUTION
# ─────────────────────────────────────────────────────────────

@dataclass
class OffBallEvolutionResult:
    best_brain: OffBallBrain
    best_fitness: float
    generation: List[float]
    mean: List[float]


def evolve_conscience(
    position: str,
    population_size: int = 32,
    generations: int = 40,
    n_states: int = 400,
    elitism: float = 0.2,
    crossover_top: float = 0.5,
    base_mutate_rate: float = 0.15,
    mutate_decay: float = 0.97,
    seed: int = 0,
    verbose: bool = True,
    surrogate: Optional[Any] = None,
) -> OffBallEvolutionResult:
    rng = random.Random(seed)
    pop = [OffBallBrain.random(seed=i) for i in range(population_size)]

    best_overall: Optional[OffBallBrain] = None
    best_fitness_overall = -1.0
    gen_best: List[float] = []
    gen_mean: List[float] = []

    mutate_rate = base_mutate_rate

    for gen in range(generations):
        fitnesses = []
        for brain in pop:
            f = synthetic_conscience_fitness(
                brain, position, n_states=n_states,
                seed=seed + gen * 1000 + len(fitnesses),
                surrogate=surrogate)
            fitnesses.append(f)

        best_f = max(fitnesses)
        mean_f = statistics.mean(fitnesses)
        gen_best.append(best_f)
        gen_mean.append(mean_f)

        best_idx = fitnesses.index(best_f)
        if best_f > best_fitness_overall:
            best_fitness_overall = best_f
            data = pop[best_idx].serialize()
            best_overall = OffBallBrain.deserialize(data)

        if verbose:
            print(f"  gen {gen+1:3d}  best={best_f:.4f}  mean={mean_f:.4f}  "
                  f"mut_rate={mutate_rate:.3f}")

        order = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
        n_elite = max(1, int(population_size * elitism))
        next_pop = [pop[i] for i in order[:n_elite]]
        n_parents = max(2, int(population_size * crossover_top))
        parents = [pop[i] for i in order[:n_parents]]

        while len(next_pop) < population_size:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = p1.crossover(p2)
            child = child.mutate(rate=mutate_rate, strength=0.3)
            next_pop.append(child)

        pop = next_pop
        mutate_rate *= mutate_decay

    return OffBallEvolutionResult(
        best_brain=best_overall if best_overall is not None else pop[0],
        best_fitness=best_fitness_overall,
        generation=gen_best,
        mean=gen_mean,
    )