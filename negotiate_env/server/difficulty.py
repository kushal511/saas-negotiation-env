"""Difficulty level configuration for NegotiateEnvironment.

Easy:   large vendor flexibility, no constraint drift, 30 turns
Medium: moderate vendor floor, competitor pressure, 40 turns (default)
Hard:   tight vendor floor, budget constraints, drift enabled, 50 turns

Long-horizon planning: Extended turn counts for super long-horizon planning.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    max_turns: int
    enable_drift: bool
    floor_multiplier: float   # multiplied against vendor_floor_price (higher = harder)
    concession_scale: float   # scales how much the AE concedes per turn (lower = harder)


DIFFICULTY_CONFIGS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        max_turns=30,            # Extended for long-horizon planning (was 12)
        enable_drift=False,
        floor_multiplier=0.90,   # floor is 10% lower than scenario default
        concession_scale=1.5,    # AE concedes 50% more generously
    ),
    "medium": DifficultyConfig(
        max_turns=40,            # Extended for long-horizon planning (was 10)
        enable_drift=True,
        floor_multiplier=1.0,    # scenario default
        concession_scale=1.0,
    ),
    "hard": DifficultyConfig(
        max_turns=50,            # Extended for long-horizon planning (was 7)
        enable_drift=True,
        floor_multiplier=1.10,   # floor is 10% higher (tighter margin)
        concession_scale=0.6,    # AE concedes 40% less
    ),
}


def get_difficulty(name: str) -> DifficultyConfig:
    name = name.lower()
    if name not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Unknown difficulty '{name}'. Choose from: {list(DIFFICULTY_CONFIGS)}")
    return DIFFICULTY_CONFIGS[name]
