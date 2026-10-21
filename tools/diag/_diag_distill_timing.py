"""How long does the distillation take, and how good is it? (timing probe)

    .venv\\Scripts\\python.exe tools\\diag\\_diag_distill_timing.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from brain_distill import distill_brain  # noqa: E402
from critic_surrogate import load_any_surrogate  # noqa: E402
from perception import PerceptionConfig  # noqa: E402

SUR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "brains", "surrogate_pos_v7merged.json")


def main():
    sur = load_any_surrogate(SUR)
    cfg = PerceptionConfig(enabled=True, role_blocks=True, seed=123)
    print(f"surrogate: {SUR}")
    print(f"{'pos':<5}{'states':>8}{'epochs':>9}{'seconds':>10}{'agree':>8}"
          f"{'intents':>9}{'max share':>11}{'SHOOT':>8}")
    for pos in ("ST", "CM", "CB"):
        for n, e in ((2000, 120), (800, 80)):
            t = time.time()
            _b, rep = distill_brain(pos, sur, n_states=n, seed=123, epochs=e,
                                    perception_config=cfg)
            el = time.time() - t
            print(f"{pos:<5}{n:>8}{e:>9}{el:>10.1f}{rep['agreement']:>8.3f}"
                  f"{len(rep['mix']):>9}{max(rep['mix'].values()):>11.3f}"
                  f"{rep['mix'].get('SHOOT', 0.0):>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
