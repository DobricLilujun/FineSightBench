"""Generator for **FineSight-Perception** dataset.

Each sample contains a single target placed on a clean canvas.
The model is asked to identify the target.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from finesightbench.core.canvas import create_canvas
from finesightbench.core.colors import COLORS, TARGET_COLORS
from finesightbench.core.objects import (
    ANIMAL_TYPES,
    LETTERS,
    SHAPE_TYPES,
    TARGET_SIZES,
    draw_animal,
    draw_block,
    draw_color_block,
    draw_letter,
    draw_shape,
    random_position,
)

# ── difficulty mapping ───────────────────────────────────────────────────────
def _difficulty(size: int) -> str:
    if size <= 5:
        return "extreme"
    if size <= 12:
        return "hard"
    if size <= 24:
        return "medium"
    return "easy"


# ── per-task generators ──────────────────────────────────────────────────────

def _gen_letter(canvas_w: int, canvas_h: int, size: int) -> dict[str, Any]:
    letter = random.choice(LETTERS)
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    pos = random_position(canvas_w, canvas_h, size)
    img = create_canvas(canvas_w, canvas_h)
    draw_letter(img, pos, size, letter, color_rgb)
    return {
        "image": img,
        "task_type": "letter_recognition",
        "question": "What letter is displayed in the image?",
        "answer": letter,
        "targets": [{
            "type": "letter",
            "value": letter,
            "size": size,
            "position": list(pos),
            "color": color_name,
            "color_rgb": list(color_rgb),
        }],
    }


def _gen_animal(canvas_w: int, canvas_h: int, size: int) -> dict[str, Any]:
    animal = random.choice(ANIMAL_TYPES)
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    pos = random_position(canvas_w, canvas_h, size)
    img = create_canvas(canvas_w, canvas_h)
    draw_animal(img, pos, size, animal, color_rgb)
    return {
        "image": img,
        "task_type": "animal_recognition",
        "question": "What animal is shown in the image?",
        "answer": animal,
        "targets": [{
            "type": "animal",
            "value": animal,
            "size": size,
            "position": list(pos),
            "color": color_name,
            "color_rgb": list(color_rgb),
        }],
    }


def _gen_block(canvas_w: int, canvas_h: int, size: int) -> dict[str, Any]:
    pos = random_position(canvas_w, canvas_h, size)
    img = create_canvas(canvas_w, canvas_h)
    draw_block(img, pos, size, COLORS["black"])
    return {
        "image": img,
        "task_type": "block_recognition",
        "question": "Is there a square block in the image? Describe what you see.",
        "answer": "yes",
        "targets": [{
            "type": "block",
            "value": "block",
            "size": size,
            "position": list(pos),
            "color": "black",
            "color_rgb": list(COLORS["black"]),
        }],
    }


def _gen_color_block(canvas_w: int, canvas_h: int, size: int) -> dict[str, Any]:
    color_name = random.choice([c for c in TARGET_COLORS if c != "black"])
    color_rgb = COLORS[color_name]
    pos = random_position(canvas_w, canvas_h, size)
    img = create_canvas(canvas_w, canvas_h)
    draw_color_block(img, pos, size, color_rgb)
    return {
        "image": img,
        "task_type": "color_block_recognition",
        "question": "What color is the block in the image?",
        "answer": color_name,
        "targets": [{
            "type": "color_block",
            "value": color_name,
            "size": size,
            "position": list(pos),
            "color": color_name,
            "color_rgb": list(color_rgb),
        }],
    }


def _gen_shape(canvas_w: int, canvas_h: int, size: int) -> dict[str, Any]:
    shape = random.choice(SHAPE_TYPES)
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    pos = random_position(canvas_w, canvas_h, size)
    img = create_canvas(canvas_w, canvas_h)
    draw_shape(img, pos, size, shape, color_rgb)
    return {
        "image": img,
        "task_type": "shape_recognition",
        "question": "What shape is displayed in the image?",
        "answer": shape,
        "targets": [{
            "type": "shape",
            "value": shape,
            "size": size,
            "position": list(pos),
            "color": color_name,
            "color_rgb": list(color_rgb),
        }],
    }


_GENERATORS = {
    "letter": _gen_letter,
    "animal": _gen_animal,
    "block": _gen_block,
    "color_block": _gen_color_block,
    "shape": _gen_shape,
}


# ── public API ───────────────────────────────────────────────────────────────

def generate_perception_dataset(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    num_per_config: int = 5,
    seed: int | None = 42,
) -> Path:
    """Generate the full FineSight-Perception dataset.

    Returns the path to the generated ``labels.json``.
    """
    if seed is not None:
        random.seed(seed)

    sizes = sizes or TARGET_SIZES
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    idx = 0

    for task_name, gen_fn in _GENERATORS.items():
        for size in sizes:
            for _ in range(num_per_config):
                result = gen_fn(canvas_size, canvas_size, size)
                image = result.pop("image")

                image_id = f"perception_{task_name}_{size}px_{idx:05d}"
                rel_path = f"images/{image_id}.png"
                image.save(img_dir / f"{image_id}.png")

                sample = {
                    "image_id": image_id,
                    "image_path": rel_path,
                    "dataset": "perception",
                    "task_type": result["task_type"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "difficulty": _difficulty(size),
                    "metadata": {
                        "canvas_size": [canvas_size, canvas_size],
                        "background_color": "white",
                        "background_color_rgb": [255, 255, 255],
                        "targets": result["targets"],
                    },
                }
                samples.append(sample)
                idx += 1

    dataset_meta = {
        "dataset_info": {
            "name": "FineSight-Perception",
            "version": "1.0",
            "description": "Fine-grained visual perception evaluation for VLMs",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": list(_GENERATORS.keys()),
            "target_sizes": sizes,
        },
        "samples": samples,
    }

    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False))
    print(f"[Perception] Generated {len(samples)} samples → {out}")
    return labels_path
