"""Overlay heatmaps onto images and save/display utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    cm = None  # type: ignore[assignment]


# ────────────────────────────────────────────────────────────────────────────
# core overlay
# ────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a 2-D heatmap onto *image* using a matplotlib colormap.

    Parameters
    ----------
    image : PIL.Image.Image
        Original image (any mode; will be converted to RGB).
    heatmap : ndarray, shape (H, W)
        Attention map with values in [0, 1].  Will be resized to match
        *image* dimensions using bilinear interpolation.
    colormap : str
        Matplotlib colormap name (e.g. ``"jet"``, ``"viridis"``,
        ``"turbo"``, ``"hot"``).
    alpha : float, 0–1
        Blending factor.  0 = only original image, 1 = only heatmap.

    Returns
    -------
    PIL.Image.Image
        RGB image with the coloured heatmap blended on top.
    """
    if cm is None:
        raise ImportError("matplotlib is required for overlay_heatmap")

    image = image.convert("RGB")
    w, h = image.size

    # resize heatmap to image resolution
    hmap_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    hmap_img = hmap_img.resize((w, h), Image.BILINEAR)
    hmap_arr = np.asarray(hmap_img, dtype=np.float64) / 255.0

    # apply colormap → RGBA float in [0, 1]
    cmap = plt.colormaps.get_cmap(colormap)
    colored = cmap(hmap_arr)  # (h, w, 4)
    colored_rgb = (colored[..., :3] * 255).astype(np.uint8)

    # blend
    img_arr = np.asarray(image, dtype=np.float64)
    overlay_arr = colored_rgb.astype(np.float64)
    blended = (1 - alpha) * img_arr + alpha * overlay_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, "RGB")


# ────────────────────────────────────────────────────────────────────────────
# side-by-side comparison
# ────────────────────────────────────────────────────────────────────────────

def side_by_side(
    image: Image.Image,
    heatmaps: dict[str, np.ndarray],
    *,
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize_per_image: tuple[float, float] = (4.0, 4.0),
    title_fontsize: int = 12,
) -> "plt.Figure":
    """Create a matplotlib figure with the original image and overlaid heatmaps.

    Parameters
    ----------
    image : PIL.Image.Image
    heatmaps : dict[str, ndarray]
        Mapping from label → 2-D heatmap.
    colormap : str
    alpha : float
    figsize_per_image : (w, h) per subplot
    title_fontsize : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    if plt is None:
        raise ImportError("matplotlib is required for side_by_side")

    n = 1 + len(heatmaps)
    fw = figsize_per_image[0] * n
    fh = figsize_per_image[1]
    fig, axes = plt.subplots(1, n, figsize=(fw, fh))
    if n == 1:
        axes = [axes]

    axes[0].imshow(np.asarray(image.convert("RGB")))
    axes[0].set_title("Original", fontsize=title_fontsize)
    axes[0].axis("off")

    for ax, (label, hmap) in zip(axes[1:], heatmaps.items()):
        overlay = overlay_heatmap(image, hmap, colormap=colormap, alpha=alpha)
        ax.imshow(np.asarray(overlay))
        ax.set_title(label, fontsize=title_fontsize)
        ax.axis("off")

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
# save helper
# ────────────────────────────────────────────────────────────────────────────

def save_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    path: str | Path,
    *,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> Path:
    """Overlay heatmap and save the result to *path*.

    Returns the resolved :class:`Path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = overlay_heatmap(image, heatmap, colormap=colormap, alpha=alpha)
    result.save(str(path))
    return path
