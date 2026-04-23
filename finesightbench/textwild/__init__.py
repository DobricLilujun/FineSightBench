"""SynthText-style *text in the wild* tasks for FineSightBench.

Adds one perception task and two reasoning tasks that follow the existing
FineSight design logic (character pixel-height = ``TARGET_SIZES`` -> difficulty).
"""

from .generator import (
    generate_textwild_perception,
    generate_textwild_reasoning,
)

__all__ = [
    "generate_textwild_perception",
    "generate_textwild_reasoning",
]
