"""Canvas (background image) creation."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw


def create_canvas(
    width: int = 512,
    height: int = 512,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create an RGB canvas filled with *bg_color*."""
    return Image.new("RGB", (width, height), bg_color)


def create_textured_canvas(
    width: int = 512,
    height: int = 512,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    density: float = 0.02,
    dot_size_range: tuple[int, int] = (2, 6),
    palette: list[tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Create a canvas with random-coloured dot noise as background texture.

    ``density`` is the fraction of pixels covered by texture dots.
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    palette = palette or [
        (255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
        (255, 192, 203), (139, 69, 19), (128, 128, 128),
    ]
    n_dots = int((width * height) * density / (sum(dot_size_range) / 2) ** 2)
    for _ in range(n_dots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        s = random.randint(*dot_size_range)
        c = random.choice(palette)
        draw.ellipse([x, y, x + s, y + s], fill=c)
    return img
