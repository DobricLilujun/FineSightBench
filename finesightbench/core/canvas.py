"""Canvas (background image) creation."""

from __future__ import annotations

from PIL import Image


def create_canvas(
    width: int = 512,
    height: int = 512,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create an RGB canvas filled with *bg_color*."""
    return Image.new("RGB", (width, height), bg_color)
