"""Visualisation utilities for FineSightBench datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from PIL import Image


def visualize_dataset(
    data_dir: str | Path,
    output_path: str | Path | None = None,
    num_display: int = 25,
    cols: int = 5,
    figscale: float = 3.0,
) -> Path:
    """Render a grid of sample images with their labels.

    Parameters
    ----------
    data_dir : path
        Directory containing ``labels.json`` and an ``images/`` sub-folder.
    output_path : path, optional
        Where to save the visualisation PNG.  Defaults to
        ``<data_dir>/visualization.png``.
    num_display : int
        Maximum number of samples to show.
    cols : int
        Number of columns in the grid.
    figscale : float
        Size multiplier per sub-plot.

    Returns
    -------
    Path to the saved visualisation image.
    """
    data_dir = Path(data_dir)
    labels_path = data_dir / "labels.json"
    with open(labels_path) as f:
        data = json.load(f)

    samples = data["samples"][:num_display]
    n = len(samples)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figscale, rows * figscale))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        if i < n:
            s = samples[i]
            img_path = data_dir / s["image_path"]
            img = Image.open(img_path)
            ax.imshow(img)
            # build concise caption
            task = s["task_type"]
            ans = s["answer"]
            diff = s.get("difficulty", "")
            size = ""
            targets = s.get("metadata", {}).get("targets", [])
            if targets:
                size = f"{targets[0]['size']}px"
            caption = f"{task}\n{size} [{diff}]\nA: {ans}"
            ax.set_title(caption, fontsize=7, pad=2)
        ax.axis("off")

    plt.tight_layout()
    if output_path is None:
        output_path = data_dir / "visualization.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved → {output_path}")
    return output_path


def visualize_by_task(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    samples_per_task: int = 10,
    cols: int = 5,
    figscale: float = 3.0,
) -> list[Path]:
    """Create one visualisation grid per task type.

    Returns list of saved paths.
    """
    data_dir = Path(data_dir)
    labels_path = data_dir / "labels.json"
    with open(labels_path) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = data_dir / "vis_by_task"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # group by task_type
    groups: dict[str, list] = {}
    for s in data["samples"]:
        groups.setdefault(s["task_type"], []).append(s)

    paths: list[Path] = []
    for task, items in groups.items():
        subset = items[:samples_per_task]
        n = len(subset)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figscale, rows * figscale))
        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i, ax in enumerate(axes):
            if i < n:
                s = subset[i]
                img_path = data_dir / s["image_path"]
                img = Image.open(img_path)
                ax.imshow(img)
                targets = s.get("metadata", {}).get("targets", [])
                size = f"{targets[0]['size']}px" if targets else ""
                diff = s.get("difficulty", "")
                ans = s["answer"]
                ax.set_title(f"{size} [{diff}]\nA: {ans}", fontsize=7, pad=2)
            ax.axis("off")

        fig.suptitle(task, fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"{task}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    print(f"[Visualize] Saved {len(paths)} task-level grids → {output_dir}")
    return paths
