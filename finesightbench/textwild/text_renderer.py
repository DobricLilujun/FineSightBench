"""Text rendering utilities with pixel-accurate character height control.

The "size" of a rendered word is defined as the cap-height of its glyphs in
pixels, matching the ``TARGET_SIZES`` convention used by the rest of
FineSightBench (4, 8, 12, 16, 24, 32, 48).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from finesightbench.core.colors import COLORS, TARGET_COLORS

# ── word bank (short, common, visually distinguishable) ─────────────────────
WORD_BANK: list[str] = [
    "HOME", "CITY", "ROAD", "STOP", "PARK", "CAFE", "OPEN", "EXIT",
    "TIME", "LIFE", "PLAY", "MOVE", "FAST", "SLOW", "HIGH", "COOL",
    "BOLD", "CALM", "WILD", "KIND", "LOUD", "SOFT", "EAST", "WEST",
    "MOON", "STAR", "FIRE", "RAIN", "SNOW", "LEAF", "TREE", "BIRD",
    "FISH", "LION", "BEAR", "WOLF", "FARM", "SHOP", "BANK", "HALL",
    "NOTE", "WORD", "BOOK", "PAGE", "MARK", "CODE", "DATA", "INFO",
]

# short keyword subset used to query "how many words contain X"
COUNTING_TARGETS: list[str] = list("AEIORSTN")

# ── font discovery ──────────────────────────────────────────────────────────
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _available_fonts() -> list[str]:
    return [p for p in _FONT_SEARCH_PATHS if Path(p).exists()]


def _get_font(font_path: str, px_size: int) -> ImageFont.FreeTypeFont:
    """Return a TrueType font targeting *px_size* cap-height.

    ``px_size`` here is the *requested* pixel font size; actual cap-height
    is slightly smaller.  We calibrate once per font.
    """
    key = (font_path, px_size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path, px_size)
    return _font_cache[key]


def _calibrate_font_for_cap_height(font_path: str, target_cap_px: int) -> ImageFont.FreeTypeFont:
    """Find the font size that produces the requested cap-height in pixels.

    Uses the bounding-box height of an uppercase 'H' as the cap-height proxy.
    """
    # Good initial guess: font size ≈ cap_height / 0.72 (typical for sans-serif).
    size = max(4, int(round(target_cap_px / 0.72)))
    for _ in range(5):
        font = _get_font(font_path, size)
        bbox = font.getbbox("H")
        h = bbox[3] - bbox[1]
        if h == 0:
            size += 1
            continue
        if h == target_cap_px:
            break
        # Linear adjustment.
        new_size = max(4, int(round(size * target_cap_px / h)))
        if new_size == size:
            break
        size = new_size
    return _get_font(font_path, size)


# ── text drawing ────────────────────────────────────────────────────────────
@dataclass
class TextItem:
    word: str
    position: tuple[int, int]       # top-left of the rendered bbox
    size_px: int                     # target cap-height
    color: str                       # named colour
    color_rgb: tuple[int, int, int]
    bbox: tuple[int, int, int, int]  # (l, t, r, b) on canvas
    stroke: bool


def _pick_readable_colors(rng: random.Random) -> tuple[str, tuple[int, int, int], tuple[int, int, int] | None]:
    """Pick a text colour and optional stroke colour for visibility on any bg.

    Returns (name, rgb, stroke_rgb).  A dark-on-light or light-on-dark stroke
    is applied to ensure small text remains readable on natural-scene bgs.
    """
    # Prefer saturated colours (skip near-neutral ones for stroke decision).
    palette = [c for c in TARGET_COLORS if c not in ("gray",)]
    name = rng.choice(palette)
    rgb = COLORS[name]
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    stroke_rgb: tuple[int, int, int] | None
    if luma > 160:
        stroke_rgb = (0, 0, 0)
    else:
        stroke_rgb = (255, 255, 255)
    return name, rgb, stroke_rgb


def _text_bbox(font: ImageFont.FreeTypeFont, word: str, stroke_w: int) -> tuple[int, int]:
    """Return (width, height) of the rendered word including stroke."""
    tmp = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(tmp)
    l, t, r, b = d.textbbox((0, 0), word, font=font, stroke_width=stroke_w)
    return r - l + 1, b - t + 1


def _place_non_overlapping_rects(
    canvas_w: int, canvas_h: int, rects: list[tuple[int, int]],
    rng: random.Random, margin: int = 4, max_attempts: int = 300,
) -> list[tuple[int, int]] | None:
    """Try to place *rects* (list of (w,h)) non-overlapping inside canvas.

    Returns list of (x,y) top-lefts, or ``None`` if it could not place them all.
    """
    placed: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    for (w, h) in rects:
        for _ in range(max_attempts):
            if canvas_w - w - margin <= margin or canvas_h - h - margin <= margin:
                return None
            x = rng.randint(margin, canvas_w - w - margin)
            y = rng.randint(margin, canvas_h - h - margin)
            box = (x - margin, y - margin, x + w + margin, y + h + margin)
            if all(
                box[2] < pb[0] or box[0] > pb[2]
                or box[3] < pb[1] or box[1] > pb[3]
                for pb in placed
            ):
                placed.append(box)
                positions.append((x, y))
                break
        else:
            return None
    return positions


def render_words_on_image(
    background: Image.Image,
    words: list[str],
    cap_height_px: int,
    rng: random.Random,
    font_path: str | None = None,
    stroke: bool = True,
) -> tuple[Image.Image, list[TextItem]] | tuple[None, None]:
    """Render *words* onto a copy of *background*, returning (image, items).

    ``cap_height_px`` is the uppercase-letter height in pixels; the font size
    is calibrated so the actual cap-height matches.  Returns (None, None) if
    words cannot be placed without overlap.
    """
    fonts = _available_fonts()
    if not fonts:
        raise RuntimeError("No usable TrueType font found on system.")
    font_path = font_path or rng.choice(fonts)
    font = _calibrate_font_for_cap_height(font_path, cap_height_px)

    # Stroke width scales modestly with size; tiny text uses no stroke
    # (a 1 px stroke on a 4 px glyph would destroy it).
    stroke_w = 0
    if stroke and cap_height_px >= 12:
        stroke_w = max(1, cap_height_px // 16)

    sizes_wh = [_text_bbox(font, w, stroke_w) for w in words]
    cw, ch = background.size
    positions = _place_non_overlapping_rects(cw, ch, sizes_wh, rng)
    if positions is None:
        return None, None

    img = background.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    items: list[TextItem] = []
    for word, (x, y), (w, h) in zip(words, positions, sizes_wh):
        color_name, color_rgb, stroke_rgb = _pick_readable_colors(rng)
        kwargs = {"font": font, "fill": (*color_rgb, 255)}
        if stroke_w > 0:
            kwargs["stroke_width"] = stroke_w
            kwargs["stroke_fill"] = (*stroke_rgb, 255) if stroke_rgb else None
        # Align draw origin with bbox top-left (PIL's textbbox may have neg offsets).
        l, t, _, _ = draw.textbbox((0, 0), word, font=font, stroke_width=stroke_w)
        draw.text((x - l, y - t), word, **kwargs)
        items.append(TextItem(
            word=word,
            position=(x, y),
            size_px=cap_height_px,
            color=color_name,
            color_rgb=color_rgb,
            bbox=(x, y, x + w, y + h),
            stroke=stroke_w > 0,
        ))
    return img, items
