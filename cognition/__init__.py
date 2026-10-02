"""Cognition layer for PLOFA — the TOLAND mind on top of the neural body.

Subpackages:

    senses      — VisionSystem FOV gate (which actors are perceived)
    memory      — episodic MemorySystem (decay / forget / recall)
    experience  — ExperienceModel temperament (risk/aggression/confidence)
    mind        — PlayerMind composition + registry + event observer
"""

from .senses import VisionSystem
from .memory import MemorySystem, MemoryEvent
from .experience import ExperienceModel
from .mind import (
    PlayerMind, register_mind, get_mind, clear_minds, new_mind,
)

__all__ = [
    "VisionSystem", "MemorySystem", "MemoryEvent", "ExperienceModel",
    "PlayerMind", "register_mind", "get_mind", "clear_minds", "new_mind",
]