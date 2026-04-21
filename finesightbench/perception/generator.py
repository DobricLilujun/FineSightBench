"""Generator for FineSight-Perception dataset."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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

# Candidate vocabularies, inlined into prompts so the model knows the answer space.
_LETTER_CHOICES = ", ".join(LETTERS)

# ── ordering helpers ─────────────────────────────────────────────────────────

def _random_order() -> tuple[str, int]:
    """Return (direction_phrase, sort_axis) where axis 0=x (left→right), 1=y (top→bottom)."""
    if random.random() < 0.5:
        return "from left to right", 0
    return "from top to bottom", 1
_ANIMAL_CHOICES = ", ".join(ANIMAL_TYPES)
_SHAPE_CHOICES = ", ".join(SHAPE_TYPES)
_COLOR_BLOCK_CHOICES = ", ".join(c for c in TARGET_COLORS if c != "black")
_TARGET_COLOR_CHOICES = ", ".join(TARGET_COLORS)


def _difficulty(size: int) -> str:
    if size <= 5:
        return "extreme"
    if size <= 12:
        return "hard"
    if size <= 24:
        return "medium"
    return "easy"


def _place_non_overlapping(
    cw: int, ch: int, n: int, size: int,
    margin: int = 10, max_attempts: int = 200,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        for _ in range(max_attempts):
            x, y = random_position(cw, ch, size, margin)
            box = (x, y, x + size, y + size)
            overlap = any(
                not (box[2] < b[0] or box[0] > b[2]
                     or box[3] < b[1] or box[1] > b[3])
                for b in boxes
            )
            if not overlap:
                positions.append((x, y))
                boxes.append(box)
                break
        else:
            positions.append(random_position(cw, ch, size, margin))
    return positions


def _gen_letter(cw: int, ch: int, size: int, count: int = 1) -> dict[str, Any]:
    positions = _place_non_overlapping(cw, ch, count, size)
    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        letter = random.choice(LETTERS)
        color_name = random.choice(TARGET_COLORS)
        color_rgb = COLORS[color_name]
        draw_letter(img, pos, size, letter, color_rgb)
        targets.append({
            "type": "letter", "value": letter, "size": size,
            "position": list(pos), "color": color_name,
            "color_rgb": list(color_rgb),
        })
    if count == 1:
        q = (
            "What letter is displayed in the image? "
            f"Answer with a single uppercase letter from: {_LETTER_CHOICES}."
        )
        a = targets[0]["value"]
    else:
        direction, axis = _random_order()
        q = (
            f"List all the letters displayed in the image {direction}, "
            "separated by commas. Each letter is one of: "
            f"{_LETTER_CHOICES}."
        )
        a = ", ".join(t["value"] for t in sorted(targets, key=lambda t: t["position"][axis]))
    return {"image": img, "task_type": "letter_recognition",
            "question": q, "answer": a, "targets": targets}


def _gen_animal(cw: int, ch: int, size: int, count: int = 1) -> dict[str, Any]:
    positions = _place_non_overlapping(cw, ch, count, size)
    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        animal = random.choice(ANIMAL_TYPES)
        color_name = random.choice(TARGET_COLORS)
        color_rgb = COLORS[color_name]
        draw_animal(img, pos, size, animal, color_rgb)
        targets.append({
            "type": "animal", "value": animal, "size": size,
            "position": list(pos), "color": color_name,
            "color_rgb": list(color_rgb),
        })
    if count == 1:
        q = (
            "What animal is shown in the image? "
            f"Answer with one of: {_ANIMAL_CHOICES}."
        )
        a = targets[0]["value"]
    else:
        direction, axis = _random_order()
        q = (
            f"List all the animals shown in the image {direction}, "
            f"separated by commas. Each animal is one of: {_ANIMAL_CHOICES}."
        )
        a = ", ".join(t["value"] for t in sorted(targets, key=lambda t: t["position"][axis]))
    return {"image": img, "task_type": "animal_recognition",
            "question": q, "answer": a, "targets": targets}


def _gen_block(cw: int, ch: int, size: int, count: int = 1) -> dict[str, Any]:
    positions = _place_non_overlapping(cw, ch, count, size)
    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        draw_block(img, pos, size, COLORS["black"])
        targets.append({
            "type": "block", "value": "block", "size": size,
            "position": list(pos), "color": "black",
            "color_rgb": list(COLORS["black"]),
        })
    if count == 1:
        q = "Is there a square block in the image? Describe what you see."
        a = "yes"
    else:
        q = "How many square blocks are in the image?"
        a = str(count)
    return {"image": img, "task_type": "block_recognition",
            "question": q, "answer": a, "targets": targets}


def _gen_color_block(cw: int, ch: int, size: int, count: int = 1) -> dict[str, Any]:
    positions = _place_non_overlapping(cw, ch, count, size)
    img = create_canvas(cw, ch)
    pool = [c for c in TARGET_COLORS if c != "black"]
    targets = []
    used: list[str] = []
    for pos in positions:
        remaining = [c for c in pool if c not in used] or pool
        color_name = random.choice(remaining)
        used.append(color_name)
        color_rgb = COLORS[color_name]
        draw_color_block(img, pos, size, color_rgb)
        targets.append({
            "type": "color_block", "value": color_name, "size": size,
            "position": list(pos), "color": color_name,
            "color_rgb": list(color_rgb),
        })
    if count == 1:
        q = (
            "What color is the block in the image? "
            f"Answer with one of: {_COLOR_BLOCK_CHOICES}."
        )
        a = targets[0]["color"]
    else:
        direction, axis = _random_order()
        q = (
            f"List the colors of all the blocks in the image {direction}, "
            f"separated by commas. Each color is one of: {_COLOR_BLOCK_CHOICES}."
        )
        a = ", ".join(t["color"] for t in sorted(targets, key=lambda t: t["position"][axis]))
    return {"image": img, "task_type": "color_block_recognition",
            "question": q, "answer": a, "targets": targets}


def _gen_shape(cw: int, ch: int, size: int, count: int = 1) -> dict[str, Any]:
    positions = _place_non_overlapping(cw, ch, count, size)
    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        shape = random.choice(SHAPE_TYPES)
        color_name = random.choice(TARGET_COLORS)
        color_rgb = COLORS[color_name]
        draw_shape(img, pos, size, shape, color_rgb)
        targets.append({
            "type": "shape", "value": shape, "size": size,
            "position": list(pos), "color": color_name,
            "color_rgb": list(color_rgb),
        })
    if count == 1:
        q = (
            "What shape is displayed in the image? "
            f"Answer with one of: {_SHAPE_CHOICES}."
        )
        a = targets[0]["value"]
    else:
        direction, axis = _random_order()
        q = (
            f"List all the shapes displayed in the image {direction}, "
            f"separated by commas. Each shape is one of: {_SHAPE_CHOICES}."
        )
        a = ", ".join(t["value"] for t in sorted(targets, key=lambda t: t["position"][axis]))
    return {"image": img, "task_type": "shape_recognition",
            "question": q, "answer": a, "targets": targets}


_GENERATORS: dict[str, Callable] = {
    "letter": _gen_letter,
    "animal": _gen_animal,
    "block": _gen_block,
    "color_block": _gen_color_block,
    "shape": _gen_shape,
}


def generate_perception_dataset(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    num_per_config: int = 5,
    seed: int | None = 42,
    mode: str = "difficulty",
    counts: list[int] | None = None,
    target_size: int = 24,
) -> Path:
    """Generate the FineSight-Perception dataset.

    Modes
    -----
    * ``"difficulty"`` – iterate over ``sizes`` (default ``TARGET_SIZES``);
      size determines ``difficulty`` (1 target per sample).
    * ``"custom"`` – iterate over ``counts`` at a fixed ``target_size``;
      question adapts to multi-target setting.
    """
    if mode not in ("difficulty", "custom"):
        raise ValueError(f"mode must be 'difficulty' or 'custom', got {mode!r}")

    if seed is not None:
        random.seed(seed)

    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    idx = 0

    if mode == "difficulty":
        sizes = sizes or TARGET_SIZES
        config_axis = sizes
    else:
        counts = counts or [1, 2, 3, 5, 7]
        config_axis = counts

    for task_name, gen_fn in _GENERATORS.items():
        for cfg in config_axis:
            for _ in range(num_per_config):
                if mode == "difficulty":
                    size = cfg
                    result = gen_fn(canvas_size, canvas_size, size, count=1)
                    diff = _difficulty(size)
                    tag = f"{size}px"
                else:
                    size = target_size
                    result = gen_fn(canvas_size, canvas_size, size, count=cfg)
                    diff = _difficulty(size)
                    tag = f"n{cfg}"

                image = result.pop("image")
                image_id = f"perception_{task_name}_{tag}_{idx:05d}"
                rel_path = f"images/{image_id}.png"
                image.save(img_dir / f"{image_id}.png")

                sample: dict[str, Any] = {
                    "image_id": image_id,
                    "image_path": rel_path,
                    "dataset": "perception",
                    "task_type": result["task_type"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "difficulty": diff,
                    "generation_mode": mode,
                    "metadata": {
                        "canvas_size": [canvas_size, canvas_size],
                        "background_color": "white",
                        "background_color_rgb": [255, 255, 255],
                        "targets": result["targets"],
                    },
                }
                if mode == "custom":
                    sample["metadata"]["num_targets"] = cfg
                samples.append(sample)
                idx += 1

    dataset_meta = {
        "dataset_info": {
            "name": "FineSight-Perception",
            "version": "1.1",
            "description": "Fine-grained visual perception evaluation for VLMs",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": list(_GENERATORS.keys()),
            "generation_mode": mode,
            "target_sizes": sizes if mode == "difficulty" else [target_size],
            "target_counts": counts if mode == "custom" else None,
        },
        "samples": samples,
    }

    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False))
    print(f"[Perception] Generated {len(samples)} samples (mode={mode}) -> {out}")
    return labels_path
