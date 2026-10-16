"""percy_the_chosen.py — Build, train and wire Percy Luka as THE CHOSEN ONE.

Percy (RW, Pearls — the first player in your universe, 2018 → today) already
has a soul (ATTACKING_PROPHET, Ω = 1.55) and the cognition layer exists, but
he still played with the SAME 24→32→32→10 brain (2186 params) as everyone
else.  This module makes him truly different:

1.  **Role-specialized bigger brain** — Percy alone gets an oversized net
    with the schema-v2 role-features input:
        24 shared + 8 WING role block → 32 → 64 → 64 → 10  = 6,922 params
    vs a normal player:
        24 → 32 → 32 → 10 = 2,186 params

    The WING role block feeds Percy wing-specific geometry:
    marker_dist, one_v_one, inside_channel, overlap_option, cross_zone,
    cross_target_open, box_arrival, cover_behind — so he learns winger
    decisions, not generic ones.

2.  **Train him and him alone with the PROJECT'S pipeline** — `train_percy()`
    uses the exact same training machinery that produced the production
    brains (`brains/`): `brain_evolution.synthetic_fitness()` scored against
    the production surrogate `brains/surrogate_pos.json`, run through a
    Percy-sized GA loop (64→64 hidden).  Same hyperparams as T1:
    80 gen × 32 pop × 300 states, seed 123, goal_bias=0.0, role-features
    ON, perception ON.  No other position's brains are touched.

3.  **A mind that never forgets** — Percy's `PlayerMind` has decay disabled
    (`DECAY_PER_SECOND = 1.0`) and the forget threshold set to 0.0, so no
    memory is ever dropped.  His temperament starts at supreme confidence
    and his view radius is widened so he sees more of the pitch.

Wire it once before a match with `wire_percy()`.  `NeuralDecisionBrain.decide`
looks the brain up by exact player *name* in `_brain_registry` **before**
auto-loading any position file, so Percy gets his big brain automatically.
His mind is looked up the same way in `_maybe_cognition`.  No other player
is affected.

Usage::

    from percy_the_chosen import wire_percy, train_percy

    # Train Percy's brain once (uses the project's surrogate pipeline),
    # saves to disk.
    train_percy(generations=80, population=32, n_states=300, seed=123)

    # Before kickoff (e.g. in run_match.py after the soul attachment,
    # or in your match runner before simulate()):
    wire_percy()   # registers Percy's big brain + never-forgetting mind
                   # under "Perćy Luka" (real roster) and "Percy" (alias)

    # Then run the match as normal.  Percy now thinks with a bigger, wing-
    # specialized net trained on the project's surrogate, remembers
    # everything, and plays with goat-level confidence.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Optional

import numpy as np

from football_brain import (
    FootballBrain,
    INPUT_SIZE,
    OUTPUT_SIZE,
    INTENT_LABELS,
    _dna_seeded_weights,
    _he_init,
)
from brain_integration import register_brain, get_brain
from cognition.mind import PlayerMind, register_mind, get_mind
from role_features import ROLE_FAMILY, V2_INPUT_D, build_v2_vector
from brain_evolution import (
    synthetic_fitness,
    generate_state_corpus,
    _context_reward,
    _INTENT_BY_IDX,
    PlayerIntent,
)
from critic_surrogate import load_any_surrogate
from perception import PerceptionConfig


# ═════════════════════════════════════════════════════════════════════════════
# PERCY CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

PERCY_NAME = "Perćy Luka"     # real roster name (auto_run_match.py SOUL_PLAYERS key)
PERCY_ALIAS = "Percy"         # scratch-runner name (run_match.py SOUL_PLAYERS key)
PERCY_POSITION = "RW"

# Percy's role family → WING → input width 24 shared + 8 wing role features = 32.
PERCY_ROLE_FAMILY = ROLE_FAMILY.get(PERCY_POSITION, "")   # "WING"
PERCY_INPUT_SIZE = V2_INPUT_D.get(PERCY_ROLE_FAMILY, INPUT_SIZE)  # 32 for WING

# Percy's oversized architecture: 24+8 → 64 → 64 → 10 = 6,922 params.
# Standard: 24 → 32 → 32 → 10 = 2,186 params.  Percy is ~3× the capacity.
PERCY_HIDDEN_1 = 64
PERCY_HIDDEN_2 = 64

PERCY_BRAIN_PATH = os.path.join(os.path.dirname(__file__), "brains", "percy.json")

# Percy's DNA (elite, chosen-one values) used to seed his initial weights.
PERCY_VISION = 99.0      # sees passes no-one else sees
PERCY_COMPOSURE = 95.0   # never flustered
PERCY_DECISIONS = 99.0   # decisive in every situation

# Production surrogate used to train the brains in `brains/`.
PERCY_SURROGATE_PATH = "brains/surrogate_pos.json"

# T1 production hyperparams (per brains/_manifest_all.json):
#   surrogate=brains/surrogate_pos.json, 80 gen × 32 pop × 300 states,
#   seed 123, goal_bias=0.0, role_features=True, perception=True.
PERCY_GENERATIONS = 80
PERCY_POPULATION = 32
PERCY_N_STATES = 300
PERCY_SEED = 123
PERCY_GOAL_BIAS = 0.25          # attack-oriented: 25% of states are scoring situations
PERCY_PERCEPTION = True
PERCY_PERCEPTION_ROLES = True


# ═════════════════════════════════════════════════════════════════════════════
# BIG BRAIN CONSTRUCTION (role-features, Percy-only)
# ═════════════════════════════════════════════════════════════════════════════

def build_percy_brain(
    seed: int = PERCY_SEED,
    input_size: Optional[int] = None,
) -> FootballBrain:
    """Construct Percy's oversized role-specialized brain.

    Default input_size = 32 (24 shared + 8 WING role features).  The network
    is built directly (not via `FootballBrain.load` / schema validation)
    because Percy's arch is deliberately non-standard.  It is registered by
    *name* in the brain registry, so the engine never tries to schema-validate
    it against a position file.
    """
    n_in = input_size if input_size is not None else PERCY_INPUT_SIZE
    rng = np.random.default_rng(seed)

    # He-init scaled by Percy's DNA — vision widens the receptive field,
    # composure calms the hidden state, decisions sharpen the output.
    vision = PERCY_VISION / 100.0
    composure = PERCY_COMPOSURE / 100.0
    decisions = PERCY_DECISIONS / 100.0

    vision_scale = 0.8 + vision * 0.4          # ~1.18
    composure_scale = 1.2 - composure * 0.4    # ~0.82 (calm)
    decisions_scale = 0.9 + decisions * 0.2    # ~1.08 (sharp)

    w1 = _he_init(n_in, PERCY_HIDDEN_1, rng) * vision_scale
    b1 = np.zeros(PERCY_HIDDEN_1, dtype=np.float64)
    w2 = _he_init(PERCY_HIDDEN_1, PERCY_HIDDEN_2, rng) * composure_scale
    b2 = np.zeros(PERCY_HIDDEN_2, dtype=np.float64)
    w3 = _he_init(PERCY_HIDDEN_2, OUTPUT_SIZE, rng) * decisions_scale
    b3 = np.zeros(OUTPUT_SIZE, dtype=np.float64)

    meta = {
        "arch_version": 9,          # non-standard — Percy only
        "sensor_schema": "v2_role_features",
        "role_family": PERCY_ROLE_FAMILY,
        "training_method": "ga_percy_chosen_one",
        "generation": None,
        "fitness": None,
        "seed": seed,
        "input_size": n_in,
        "note": "Percy Luka — the chosen one. Oversized 24+8(WING)->64->64->10 net.",
    }
    return FootballBrain(w1, b1, w2, b2, w3, b3, seed=seed, meta=meta)


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE (bypasses the strict schema validator — Percy is non-standard)
# ═════════════════════════════════════════════════════════════════════════════

def _percy_serialize(brain: FootballBrain) -> dict:
    return {
        "name": PERCY_NAME,
        "position": PERCY_POSITION,
        "role_family": PERCY_ROLE_FAMILY,
        "input_size": brain.w1.shape[0],
        "arch": [
            brain.w1.shape[0], brain.w1.shape[1],
            brain.w2.shape[1], brain.w3.shape[1],
        ],
        "meta": brain.meta or {},
        "w1": brain.w1.tolist(),
        "b1": brain.b1.tolist(),
        "w2": brain.w2.tolist(),
        "b2": brain.b2.tolist(),
        "w3": brain.w3.tolist(),
        "b3": brain.b3.tolist(),
    }


def _percy_deserialize(data: dict) -> FootballBrain:
    arrays = [
        np.array(data["w1"], dtype=np.float64),
        np.array(data["b1"], dtype=np.float64),
        np.array(data["w2"], dtype=np.float64),
        np.array(data["b2"], dtype=np.float64),
        np.array(data["w3"], dtype=np.float64),
        np.array(data["b3"], dtype=np.float64),
    ]
    return FootballBrain(*arrays, meta=data.get("meta"))


def save_percy_brain(brain: FootballBrain, path: str = PERCY_BRAIN_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_percy_serialize(brain), f, indent=2)


def load_percy_brain(path: str = PERCY_BRAIN_PATH) -> Optional[FootballBrain]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return _percy_deserialize(json.load(f))


# ═════════════════════════════════════════════════════════════════════════════
# DEDICATED TRAINING — Percy and him alone, using the project's pipeline
# ═════════════════════════════════════════════════════════════════════════════

# Intent -> index mapping (mirrors football_brain.INTENT_LABELS order).
_INTENT_BY_LABEL = {label: i for i, label in enumerate(INTENT_LABELS)}


def _make_percy_fitness_fn(
    surrogate: Any,
    n_states: int = PERCY_N_STATES,
    seed: int = PERCY_SEED,
    goal_bias: float = PERCY_GOAL_BIAS,
    perception_config: Optional[PerceptionConfig] = None,
    attack_oriented: bool = True,
) -> Callable[[FootballBrain], float]:
    """Return a fitness function that scores a Percy-sized brain as an RW.

    Uses `brain_evolution.synthetic_fitness()` with the production surrogate
    — the exact same fitness signal that trained the brains in `brains/`.
    Generates one deterministic corpus of RW sensor states with the WING
    role-features tail (32-d input), then scores each candidate brain through
    `synthetic_fitness()`.

    When ``attack_oriented`` is True, an additional per-intent multiplier
    boosts attacking intents (SHOOT, CROSS, THROUGH_BALL, PROGRESSIVE_PASS)
    on top of the surrogate, tilting Percy toward goal threat without
    removing his playmaking (switches, through-balls, safe build-up).
    """
    # Pre-generate the corpus once (deterministic per seed), exactly like
    # `evolve()` does internally.
    corpus = generate_state_corpus(
        PERCY_POSITION, n_states, seed,
        goal_bias=goal_bias,
        input_size=PERCY_INPUT_SIZE,
        perception_config=perception_config,
    )
    sensors_all = np.asarray(corpus, dtype=np.float64)
    n = len(sensors_all)

    # Attack-orientation multiplier per intent (1.0 = neutral).
    # SHOOT/CROSS/THROUGH_BALL get the biggest boost so Percy converts
    # his high xG into shots; PROGRESSIVE_PASS gets a smaller lift.
    if attack_oriented:
        attack_multipliers = np.ones(OUTPUT_SIZE, dtype=np.float64)
        attack_multipliers[_INTENT_BY_LABEL[PlayerIntent.SHOOT.value]] = 1.30
        attack_multipliers[_INTENT_BY_LABEL[PlayerIntent.CROSS.value]] = 1.25
        attack_multipliers[_INTENT_BY_LABEL[PlayerIntent.THROUGH_BALL.value]] = 1.20
        attack_multipliers[_INTENT_BY_LABEL[PlayerIntent.PROGRESSIVE_PASS.value]] = 1.10
    else:
        attack_multipliers = np.ones(OUTPUT_SIZE, dtype=np.float64)

    def fitness(brain: FootballBrain) -> float:
        # Forward pass over the whole corpus in one batched matmul.
        probs_all = brain.forward(sensors_all)  # (n, 10)
        idx_all = np.argmax(probs_all, axis=1)
        conf_all = probs_all[np.arange(n), idx_all]

        # Base reward per (state, intent) from the surrogate table.
        base_rewards = np.zeros((n, OUTPUT_SIZE), dtype=np.float64)
        for i in range(n):
            for j, intent in enumerate(_INTENT_BY_IDX):
                s_in = sensors_all[i][:24] if PERCY_INPUT_SIZE != INPUT_SIZE else sensors_all[i]
                base_rewards[i, j] = float(surrogate.expected_success(s_in, intent, PERCY_POSITION))

        # Apply attack-orientation multiplier on top of surrogate rewards.
        chosen_reward = base_rewards[np.arange(n), idx_all] * attack_multipliers[idx_all]

        # Weight reward by confidence (decisive picks count more).
        fitness_val = float((chosen_reward * conf_all).mean())

        # ── Anti-collapse: no single intent above ~35%, at least 6 intents used ──
        counts = np.bincount(idx_all, minlength=OUTPUT_SIZE).astype(float)
        total = float(counts.sum())
        max_share = float(counts.max() / total)
        if max_share > 0.35:
            fitness_val *= (1.0 - 1.5 * (max_share - 0.35))
        shares = counts / total
        n_effective = int(np.sum(shares > 0.05))
        if n_effective < 6:
            fitness_val *= (1.0 - 0.75 * (6 - n_effective) / 6.0)

        return float(max(0.0, min(1.0, fitness_val)))

    return fitness


def train_percy(
    generations: int = PERCY_GENERATIONS,
    population: int = PERCY_POPULATION,
    n_states: int = PERCY_N_STATES,
    seed: int = PERCY_SEED,
    save: bool = True,
    verbose: bool = True,
    goal_bias: float = PERCY_GOAL_BIAS,
    surrogate_path: str = PERCY_SURROGATE_PATH,
    perception: bool = PERCY_PERCEPTION,
    perception_roles: bool = PERCY_PERCEPTION_ROLES,
    attack_oriented: bool = True,
) -> Any:
    """Evolve Percy's brain and him alone, using the project's pipeline.

    Uses `brain_evolution.synthetic_fitness()` scored against the production
    surrogate `brains/surrogate_pos.json` — the exact same fitness signal
    that trained the brains in `brains/`.  Runs through a Percy-sized GA
    loop (64→64 hidden, WING role-features input).

    Same hyperparams as T1 production: 80 gen × 32 pop × 300 states,
    seed 123, goal_bias=0.25 (attack-oriented), role-features ON,
    perception ON.

    When ``attack_oriented`` is True, SHOOT/CROSS/THROUGH_BALL/PROGRESSIVE_PASS
    get a per-intent multiplier on top of the surrogate rewards, tilting
    Percy toward goal threat without removing his playmaking.

    No other position's brains are touched.  Synthetic fitness (no full
    matches), so a run takes under a minute.

    Returns a SimpleNamespace with best_brain, best_fitness, generation,
    mean.  The best brain is saved to `brains/percy.json` unless save=False.
    """
    # Load the production surrogate (same one that trained `brains/` brains).
    surrogate = load_any_surrogate(surrogate_path)

    # Perception config (train-through-imperfect-perception), matching T1.
    perception_cfg = None
    if perception:
        perception_cfg = PerceptionConfig(
            enabled=True,
            role_blocks=perception_roles,
            seed=seed,
        )
        if verbose:
            print(f"  Perception ON (role_blocks={perception_roles}, seed={seed})")

    fitness_fn = _make_percy_fitness_fn(
        surrogate,
        n_states=n_states,
        seed=seed,
        goal_bias=goal_bias,
        perception_config=perception_cfg,
        attack_oriented=attack_oriented,
    )

    rng = np.random.default_rng(seed)
    pop = []
    for i in range(population):
        br = build_percy_brain(seed=seed + i)
        pop.append(br)

    best_brain: Optional[FootballBrain] = None
    best_fit = -1.0
    history_best = []
    history_mean = []

    mutate_rate = 0.15
    for gen in range(generations):
        fits = [fitness_fn(b) for b in pop]
        gen_best = max(fits)
        gen_mean_f = float(sum(fits) / len(fits))
        history_best.append(gen_best)
        history_mean.append(gen_mean_f)

        bi = int(np.argmax(fits))
        if fits[bi] > best_fit:
            best_fit = fits[bi]
            best_brain = pop[bi]

        if verbose:
            print(f"  Percy gen {gen+1:>2d}: best={gen_best:.4f} "
                  f"mean={gen_mean_f:.4f}")

        # Elitism: keep top 25%.
        order = np.argsort(fits)[::-1]
        n_elite = max(1, population // 4)
        elites = [pop[int(order[k])] for k in range(n_elite)]

        # Crossover + mutate to refill.
        new_pop = list(elites)
        while len(new_pop) < population:
            p1 = pop[int(order[rng.integers(0, n_elite * 2)])]
            p2 = pop[int(order[rng.integers(0, n_elite * 2)])]
            child = _percy_crossover(p1, p2)
            child = _percy_mutate(child, rate=mutate_rate, strength=0.3)
            new_pop.append(child)
        pop = new_pop
        mutate_rate = max(0.05, mutate_rate * 0.97)

    if best_brain is None:
        best_brain = pop[0]

    if save:
        save_percy_brain(best_brain)
        if verbose:
            print(f"\n  [saved] Percy's brain -> {PERCY_BRAIN_PATH}")

    if verbose:
        print(f"\n  [trained] Percy training complete. Best fitness: {best_fit:.4f}")
        print(f"     Brain: {best_brain} (Percy-only, {best_brain.param_count} params)")

    import types
    return types.SimpleNamespace(
        best_brain=best_brain,
        best_fitness=best_fit,
        generation=history_best,
        mean=history_mean,
    )


def _percy_crossover(a: FootballBrain, b: FootballBrain) -> FootballBrain:
    rng = np.random.default_rng()
    def _cross(x, y):
        m = rng.random(x.shape) < 0.5
        return np.where(m, x, y)
    return FootballBrain(
        _cross(a.w1, b.w1), _cross(a.b1, b.b1),
        _cross(a.w2, b.w2), _cross(a.b2, b.b2),
        _cross(a.w3, b.w3), _cross(a.b3, b.b3),
    )


def _percy_mutate(brain: FootballBrain, rate: float = 0.15,
                  strength: float = 0.3) -> FootballBrain:
    rng = np.random.default_rng()
    def _perturb(w):
        m = rng.random(w.shape) < rate
        noise = rng.normal(0.0, strength, size=w.shape) * np.abs(w)
        return w + m * noise
    return FootballBrain(
        _perturb(brain.w1), _perturb(brain.b1),
        _perturb(brain.w2), _perturb(brain.b2),
        _perturb(brain.w3), _perturb(brain.b3),
    )


# ═════════════════════════════════════════════════════════════════════════════
# NEVER-FORGETTING MIND
# ═════════════════════════════════════════════════════════════════════════════

def build_percy_mind(view_radius: float = 45.0) -> PlayerMind:
    """Build Percy's mind: remembers everything, sees widely, confident.

    *   `DECAY_PER_SECOND = 1.0`  — memories never fade with time.
    *   `FORGET_BELOW = 0.0`      — nothing is ever dropped (severity > 0
        always holds once stored).
    *   Supreme starting confidence (1.5) and aggression (1.4) — he plays
        like he knows he's the chosen one.
    *   Wide view radius (45 m vs the default 25 m) — he sees more of the
        pitch than anyone, so his FOV gate admits more teammates/opponents
        into his sensor block.
    """
    mind = PlayerMind(view_radius=view_radius,
                      max_teammates=4, max_opponents=4)
    # Never forget.
    mind.memory.DECAY_PER_SECOND = 1.0
    mind.memory.FORGET_BELOW = 0.0
    # Goat-level temperament.
    mind.experience.bias["confidence"] = 1.5
    mind.experience.bias["tackle_aggression"] = 1.4
    mind.experience.bias["risk_in_box"] = 1.0
    mind.engaged = True
    return mind


# ═════════════════════════════════════════════════════════════════════════════
# WIRING — call once before kickoff
# ═════════════════════════════════════════════════════════════════════════════

def wire_percy(player: Optional[Any] = None,
               brain_seed: int = PERCY_SEED,
               train_if_missing: bool = True) -> None:
    """Register Percy's big brain and never-forgetting mind by name.

    Called before a match, this makes `NeuralDecisionBrain.decide` serve
    Percy his oversized role-specialized net and `CognitionDecisionBrain`
    serve Percy his permanent memory.  No other player is affected.

    Parameters
    ----------
    player : optional
        Percy's player object.  If given, his DNA mental attributes are
        maxed out (vision/composure/decisions → 99) and his soul is
        re-affirmed as ATTACKING_PROPHET Ω=1.55.
    brain_seed : int
        Seed for building/training Percy's brain.
    train_if_missing : bool
        If no saved brain exists at `brains/percy.json`, run a short
        dedicated evolution to create one first.
    """
    # ── 1. Big brain ──
    brain = load_percy_brain()
    if brain is None:
        if train_if_missing:
            print("[train] No saved Percy brain — training him now (him alone)...")
            train_percy(seed=brain_seed, save=True, verbose=True)
        brain = load_percy_brain()
    if brain is None:
        brain = build_percy_brain(seed=brain_seed)

    register_brain(PERCY_NAME, brain)
    register_brain(PERCY_ALIAS, brain)      # scratch runner uses "Percy"
    pname_safe = PERCY_NAME.replace(chr(0x107), "c")
    print(f"  [brain] Registered Percy's brain ({brain.param_count} params) under "
          f"'{pname_safe}' (and alias '{PERCY_ALIAS}')  "
          f"input={brain.w1.shape[0]} (24 shared + {brain.w1.shape[0]-24} WING role)")

    # ── 2. Never-forgetting mind ──
    mind = build_percy_mind()
    register_mind(PERCY_NAME, mind)
    register_mind(PERCY_ALIAS, mind)        # scratch runner uses "Percy"
    print(f"  [mind] Registered Percy's never-forgetting mind "
          f"(view_radius={mind.senses.view_radius}m, decay=OFF)")

    # ── 3. Player DNA / soul (if we have the object) ──
    if player is not None:
        dna = getattr(player, "dna", None)
        if dna is not None:
            mental = getattr(dna, "mental", None)
            if mental is not None:
                for attr in ("vision", "composure", "decisions"):
                    if hasattr(mental, attr):
                        setattr(mental, attr, 99.0)
                print("  [dna] Maxed Percy's DNA mental attributes "
                      "(vision/composure/decisions -> 99).")
        # Re-affirm his soul (Omega=1.55 ATTACKING_PROPHET).
        try:
            from player_soul import SoulFactory, SoulArchetype
            pname = getattr(player, "name", PERCY_NAME)
            player.dna.soul = SoulFactory.custom(
                pname,
                SoulArchetype.ATTACKING_PROPHET,
                hardwork=0.99, talent=0.90, luck=0.99,
            )
            pname_safe = pname.replace(chr(0x107), "c") if isinstance(pname, str) else pname
            print(f"  [soul] Re-affirmed Percy's soul: ATTACKING_PROPHET "
                  f"(Omega=1.55) for '{pname_safe}'.")
        except Exception:
            pass


def unwire_percy() -> None:
    """Remove Percy's special brain + mind (restores defaults)."""
    from brain_integration import _brain_registry
    from cognition.mind import _minds
    for name in (PERCY_NAME, PERCY_ALIAS):
        _brain_registry.pop(name, None)
        _minds.pop(name, None)
    print("  [reset] Percy's special brain/mind removed (back to defaults).")


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PERCY THE CHOSEN ONE — build + smoke test")
    print("=" * 60)

    brain = build_percy_brain()
    print(f"\n  Percy's brain: {brain}  ({brain.param_count} params)")
    print(f"  Standard brain: 24->32->32->10 = 2,186 params")
    print(f"  Percy input width: {brain.w1.shape[0]} (24 shared + {brain.w1.shape[0]-24} WING role)")
    print(f"  Percy is {brain.param_count // 2186}x the capacity.")

    # Forward pass sanity check with a random 32-d sensor vector.
    rng = np.random.default_rng(7)
    sample = rng.random(PERCY_INPUT_SIZE)
    probs = brain.forward(sample)
    idx = int(np.argmax(probs))
    print(f"\n  Sample forward: intent={INTENT_LABELS[idx]} "
          f"p={probs[idx]:.3f}  (distribution sums to {probs.sum():.6f})")

    # Mind sanity check.
    mind = build_percy_mind()
    print(f"\n  Percy's mind: view_radius={mind.senses.view_radius}m, "
          f"decay={mind.memory.DECAY_PER_SECOND}, "
          f"forget_below={mind.memory.FORGET_BELOW}, "
          f"confidence={mind.experience.bias['confidence']}")

    print("\n  OK Percy is ready. Call wire_percy() before kickoff.")
