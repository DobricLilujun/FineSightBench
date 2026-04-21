"""Color definitions, color-vision-deficiency simulation, and blur utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# ── named colours ────────────────────────────────────────────────────────────
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 180, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}

# Colours usable as *target* colours (visible on white bg)
TARGET_COLORS: list[str] = [
    "red", "green", "blue", "yellow", "cyan", "magenta",
    "orange", "purple", "pink", "brown", "black", "gray",
]

# ── CVD simulation matrices (Viénot 1999) ───────────────────────────────────
CVD_MATRICES: dict[str, np.ndarray] = {
    "protanopia": np.array([
        [0.56667, 0.43333, 0.00000],
        [0.55833, 0.44167, 0.00000],
        [0.00000, 0.24167, 0.75833],
    ]),
    "deuteranopia": np.array([
        [0.62500, 0.37500, 0.00000],
        [0.70000, 0.30000, 0.00000],
        [0.00000, 0.30000, 0.70000],
    ]),
    "tritanopia": np.array([
        [0.95000, 0.05000, 0.00000],
        [0.00000, 0.43333, 0.56667],
        [0.00000, 0.47500, 0.52500],
    ]),
}


def simulate_cvd(image: Image.Image, cvd_type: str) -> Image.Image:
    """Apply colour-vision-deficiency simulation to *image*.

    Parameters
    ----------
    cvd_type : str
        One of ``"protanopia"``, ``"deuteranopia"``, ``"tritanopia"``.
    """
    mat = CVD_MATRICES[cvd_type]
    arr = np.asarray(image, dtype=np.float64)[..., :3]
    transformed = arr @ mat.T
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return Image.fromarray(transformed, "RGB")


def apply_blur(image: Image.Image, radius: float = 12.0) -> Image.Image:
    """Return a Gaussian-blurred copy of *image*."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
