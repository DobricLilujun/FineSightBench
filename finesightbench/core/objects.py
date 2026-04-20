"""Drawing primitives: letters, animals, blocks, shapes, dots."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── constants ────────────────────────────────────────────────────────────────
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

ANIMAL_TYPES = ["cat", "dog", "fish", "bird", "rabbit", "turtle"]

SHAPE_TYPES = ["circle", "triangle", "square", "star", "diamond", "pentagon", "hexagon", "cross"]

# Target sizes used across the benchmark (pixels)
TARGET_SIZES = [3, 5, 8, 12, 16, 24, 32, 48]

# ── pixel-art animal templates (7×7) ────────────────────────────────────────
_ANIMAL_TEMPLATES: dict[str, list[list[int]]] = {
    "cat": [
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
    ],
    "dog": [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
    ],
    "fish": [
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    "bird": [
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    "rabbit": [
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
    ],
    "turtle": [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ],
}

# ── font helper ──────────────────────────────────────────────────────────────
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TrueType font at *size*, with system-font fallback."""
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_SEARCH_PATHS:
        if Path(p).exists():
            font = ImageFont.truetype(p, size)
            _font_cache[size] = font
            return font
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ── random position helper ──────────────────────────────────────────────────
def random_position(
    canvas_w: int,
    canvas_h: int,
    target_size: int,
    margin: int = 10,
) -> tuple[int, int]:
    """Return a random (x, y) so the target fits inside the canvas."""
    lo_x = margin
    hi_x = max(lo_x + 1, canvas_w - target_size - margin)
    lo_y = margin
    hi_y = max(lo_y + 1, canvas_h - target_size - margin)
    return random.randint(lo_x, hi_x), random.randint(lo_y, hi_y)


# ── drawing functions ────────────────────────────────────────────────────────

def draw_letter(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    letter: str,
    color: tuple[int, int, int],
) -> None:
    """Render a single capital letter onto *image*.

    The letter is rendered at high resolution then scaled to *size*×*size*
    using nearest-neighbour interpolation so it stays crisp.
    """
    render_size = max(size * 4, 64)
    font = _get_font(render_size)
    tmp = Image.new("RGBA", (render_size * 2, render_size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((-bbox[0], -bbox[1]), letter, fill=(*color, 255), font=font)
    tmp = tmp.crop((0, 0, tw, th))
    tmp = tmp.resize((size, size), Image.NEAREST)
    image.paste(tmp, position, tmp)


def draw_animal(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    animal_type: str,
    color: tuple[int, int, int],
) -> None:
    """Render a pixel-art animal silhouette onto *image*."""
    template = np.array(_ANIMAL_TEMPLATES[animal_type], dtype=np.uint8)
    h, w = template.shape
    # scale template to target size
    scaled = np.zeros((size, size), dtype=np.uint8)
    for sy in range(size):
        for sx in range(size):
            oy = int(sy * h / size)
            ox = int(sx * w / size)
            scaled[sy, sx] = template[min(oy, h - 1), min(ox, w - 1)]
    # build RGBA patch
    patch = np.zeros((size, size, 4), dtype=np.uint8)
    for c in range(3):
        patch[..., c] = scaled * color[c]
    patch[..., 3] = scaled * 255
    patch_img = Image.fromarray(patch, "RGBA")
    image.paste(patch_img, position, patch_img)


def draw_block(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a solid square block."""
    draw = ImageDraw.Draw(image)
    x, y = position
    draw.rectangle([x, y, x + size - 1, y + size - 1], fill=color)


def draw_color_block(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a coloured square block (same as draw_block but name clarifies intent)."""
    draw_block(image, position, size, color)


def draw_shape(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    shape_type: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a geometric shape inscribed in *size*×*size* at *position*."""
    draw = ImageDraw.Draw(image)
    x, y = position
    cx, cy = x + size / 2, y + size / 2
    r = size / 2

    if shape_type == "circle":
        draw.ellipse([x, y, x + size - 1, y + size - 1], fill=color)

    elif shape_type == "triangle":
        pts = [(cx, y), (x, y + size - 1), (x + size - 1, y + size - 1)]
        draw.polygon(pts, fill=color)

    elif shape_type == "square":
        draw.rectangle([x, y, x + size - 1, y + size - 1], fill=color)

    elif shape_type == "diamond":
        pts = [(cx, y), (x + size - 1, cy), (cx, y + size - 1), (x, cy)]
        draw.polygon(pts, fill=color)

    elif shape_type == "star":
        pts = _star_points(cx, cy, r, r * 0.38, 5)
        draw.polygon(pts, fill=color)

    elif shape_type == "pentagon":
        pts = _regular_polygon(cx, cy, r, 5)
        draw.polygon(pts, fill=color)

    elif shape_type == "hexagon":
        pts = _regular_polygon(cx, cy, r, 6)
        draw.polygon(pts, fill=color)

    elif shape_type == "cross":
        arm = max(1, size // 3)
        draw.rectangle([x + arm, y, x + size - arm - 1, y + size - 1], fill=color)
        draw.rectangle([x, y + arm, x + size - 1, y + size - arm - 1], fill=color)

    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")


def draw_dot(
    image: Image.Image,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled circle (dot)."""
    draw = ImageDraw.Draw(image)
    x, y = position
    draw.ellipse([x, y, x + size - 1, y + size - 1], fill=color)


# ── geometry helpers ─────────────────────────────────────────────────────────

def _regular_polygon(
    cx: float, cy: float, r: float, n: int, start_angle: float = -math.pi / 2,
) -> list[tuple[float, float]]:
    """Vertices of a regular *n*-gon centred at (cx, cy)."""
    return [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n))
        for i in range(n)
    ]


def _star_points(
    cx: float, cy: float, outer_r: float, inner_r: float, n: int,
) -> list[tuple[float, float]]:
    """Vertices of an *n*-pointed star."""
    pts: list[tuple[float, float]] = []
    angle_start = -math.pi / 2
    for i in range(n * 2):
        r = outer_r if i % 2 == 0 else inner_r
        angle = angle_start + math.pi * i / n
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts
