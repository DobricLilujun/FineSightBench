"""Generator for FineSight-Reasoning dataset."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from finesightbench.core.canvas import create_canvas, create_textured_canvas
from finesightbench.core.colors import (
    COLORS,
    TARGET_COLORS,
    apply_blur,
    simulate_cvd,
)
from finesightbench.core.objects import (
    ANIMAL_TYPES,
    LETTERS,
    SHAPE_TYPES,
    TARGET_SIZES,
    draw_animal,
    draw_block,
    draw_color_block,
    draw_dot,
    draw_letter,
    draw_shape,
    random_position,
)


_VALUE_POOL: dict[str, list[str]] = {
    "letter": LETTERS,
    "animal": ANIMAL_TYPES,
    "block": ["block"],
    "color_block": ["color_block"],
    "shape": SHAPE_TYPES,
    "dot": ["dot"],
}

# Candidate vocabularies for prompt grounding.
_COLOR_BLOCK_CHOICES = ", ".join(
    c for c in TARGET_COLORS if c not in ("black", "white", "gray")
)
_LETTER_CHOICES = ", ".join(LETTERS)
_ANIMAL_CHOICES = ", ".join(ANIMAL_TYPES)
_SHAPE_CHOICES = ", ".join(SHAPE_TYPES)
_TARGET_COLOR_CHOICES = ", ".join(TARGET_COLORS)

_OBJ_TYPE_LABEL: dict[str, str] = {
    "letter": "letters",
    "animal": "animals",
    "block": "blocks",
    "color_block": "blocks",
    "shape": "shapes",
    "dot": "dots",
}

_OBJ_TYPE_CHOICES_HINT: dict[str, str] = {
    "letter": f" Each object is an uppercase letter from: {_LETTER_CHOICES}.",
    "animal": f" Each object is an animal from: {_ANIMAL_CHOICES}.",
    "shape": f" Each object is a shape from: {_SHAPE_CHOICES}.",
    "block": "",
    "color_block": "",
    "dot": "",
}


# ── ordering helpers ─────────────────────────────────────────────────────────

def _random_order() -> tuple[str, int]:
    """Return (direction_phrase, sort_axis) where axis 0=x (left→right), 1=y (top→bottom)."""
    if random.random() < 0.5:
        return "from left to right", 0
    return "from top to bottom", 1


def _difficulty(size: int) -> str:
    if size <= 5:
        return "extreme"
    if size <= 12:
        return "hard"
    if size <= 24:
        return "medium"
    return "easy"


def _quadrant(x: int, y: int, cw: int, ch: int) -> str:
    col = "left" if x < cw / 2 else "right"
    row = "top" if y < ch / 2 else "bottom"
    return f"{row}-{col}"


def _place_non_overlapping(
    canvas_w: int,
    canvas_h: int,
    sizes: list[int],
    margin: int = 10,
    max_attempts: int = 200,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for s in sizes:
        for _ in range(max_attempts):
            x, y = random_position(canvas_w, canvas_h, s, margin)
            box = (x, y, x + s, y + s)
            overlap = any(
                not (box[2] < pb[0] or box[0] > pb[2]
                     or box[3] < pb[1] or box[1] > pb[3])
                for pb in placed_boxes
            )
            if not overlap:
                positions.append((x, y))
                placed_boxes.append(box)
                break
        else:
            positions.append(random_position(canvas_w, canvas_h, s, margin))
    return positions


def _draw_target(
    image, obj_type: str, pos: tuple[int, int], size: int,
    color_rgb: tuple[int, int, int], value: str | None = None,
) -> None:
    if obj_type == "letter":
        draw_letter(image, pos, size, value or "A", color_rgb)
    elif obj_type == "animal":
        draw_animal(image, pos, size, value or "cat", color_rgb)
    elif obj_type == "block":
        draw_block(image, pos, size, color_rgb)
    elif obj_type == "color_block":
        draw_color_block(image, pos, size, color_rgb)
    elif obj_type == "shape":
        draw_shape(image, pos, size, value or "circle", color_rgb)
    elif obj_type == "dot":
        draw_dot(image, pos, size, color_rgb)


def _gen_comparison(cw: int, ch: int, base_size: int, count: int = 2) -> dict[str, Any]:
    """Two targets of different sizes. Two-part question: count + larger side."""
    obj_type = random.choice(["letter", "animal", "block", "shape", "dot"])
    value = random.choice(_VALUE_POOL.get(obj_type, ["x"]))
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]

    ratio = random.choice([0.5, 0.6, 0.75, 1.5, 1.7, 2.0])
    size_a = base_size
    size_b = max(3, int(base_size * ratio))
    if size_a == size_b:
        size_b = size_a + 2

    margin = 10
    ax = random.randint(margin, cw // 2 - size_a - margin)
    ay = random.randint(margin, ch - size_a - margin)
    bx = random.randint(cw // 2 + margin, max(cw // 2 + margin + 1, cw - size_b - margin))
    by = random.randint(margin, ch - size_b - margin)

    img = create_canvas(cw, ch)
    _draw_target(img, obj_type, (ax, ay), size_a, color_rgb, value)
    _draw_target(img, obj_type, (bx, by), size_b, color_rgb, value)

    if size_a > size_b:
        larger = "left"
    elif size_b > size_a:
        larger = "right"
    else:
        larger = "same"

    question = (
        "Answer two questions about the image. "
        "(1) How many objects are in the image? "
        "(2) Which one is larger, the object on the left or the one on the right? "
        "Answer in the format: '<count>; <left|right|same>'."
        + _OBJ_TYPE_CHOICES_HINT.get(obj_type, "")
    )
    answer = f"2; {larger}"

    return {
        "image": img,
        "task_type": "comparison",
        "question": question,
        "answer": answer,
        "sub_answers": {"count": 2, "larger": larger},
        "targets": [
            {"type": obj_type, "value": value, "size": size_a,
             "position": [ax, ay], "color": color_name,
             "color_rgb": list(color_rgb), "side": "left"},
            {"type": obj_type, "value": value, "size": size_b,
             "position": [bx, by], "color": color_name,
             "color_rgb": list(color_rgb), "side": "right"},
        ],
    }


def _gen_counting(cw: int, ch: int, base_size: int, count: int | None = None) -> dict[str, Any]:
    obj_type = random.choice(["letter", "animal", "block", "color_block", "shape"])
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    n = count if count is not None else random.randint(2, 9)
    positions = _place_non_overlapping(cw, ch, [base_size] * n)

    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        value = random.choice(_VALUE_POOL[obj_type])
        _draw_target(img, obj_type, pos, base_size, color_rgb, value)
        targets.append({
            "type": obj_type, "value": value, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        })

    type_label = {
        "letter": "letters", "animal": "animals", "block": "blocks",
        "color_block": f"{color_name} blocks", "shape": "shapes",
    }[obj_type]
    hint = _OBJ_TYPE_CHOICES_HINT.get(obj_type, "")

    return {
        "image": img,
        "task_type": "counting",
        "question": f"How many {type_label} are there in the image?{hint}",
        "answer": str(n),
        "targets": targets,
    }


def _gen_spatial(cw: int, ch: int, base_size: int, count: int = 1) -> dict[str, Any]:
    obj_type = random.choice(["letter", "animal", "block", "color_block", "shape"])
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    main_value = random.choice(_VALUE_POOL[obj_type])
    positions = _place_non_overlapping(cw, ch, [base_size] * count)

    img = create_canvas(cw, ch)
    targets = []
    for i, pos in enumerate(positions):
        if i == 0:
            v, c_name, c_rgb = main_value, color_name, color_rgb
        else:
            v = random.choice(_VALUE_POOL[obj_type])
            c_name = random.choice([c for c in TARGET_COLORS if c != color_name])
            c_rgb = COLORS[c_name]
        _draw_target(img, obj_type, pos, base_size, c_rgb, v)
        targets.append({
            "type": obj_type, "value": v, "size": base_size,
            "position": list(pos), "color": c_name, "color_rgb": list(c_rgb),
            "is_query": (i == 0),
        })

    main = targets[0]
    quad = _quadrant(main["position"][0] + base_size // 2,
                     main["position"][1] + base_size // 2, cw, ch)

    if count == 1:
        question = (
            "In which quadrant of the image is the object located? "
            "Answer: top-left, top-right, bottom-left, or bottom-right."
        )
    else:
        descr_value = main_value if obj_type not in ("block", "color_block") else "block"
        question = (
            f"In which quadrant of the image is the {color_name} {descr_value} located? "
            "Answer: top-left, top-right, bottom-left, or bottom-right."
            + _OBJ_TYPE_CHOICES_HINT.get(obj_type, "")
        )

    return {
        "image": img,
        "task_type": "spatial",
        "question": question,
        "answer": quad,
        "targets": targets,
    }


def _gen_cvd(cw: int, ch: int, base_size: int, count: int = 1) -> dict[str, Any]:
    """CVD interference is applied to the BACKGROUND only; target stays clean."""
    cvd_type = random.choice(["protanopia", "deuteranopia", "tritanopia"])
    bg = create_textured_canvas(cw, ch, density=0.04)
    bg = simulate_cvd(bg, cvd_type)

    color_pool = [c for c in TARGET_COLORS if c not in ("black", "white", "gray")]
    positions = _place_non_overlapping(cw, ch, [base_size] * count, margin=20)
    targets = []
    used_colors: list[str] = []
    for pos in positions:
        remaining = [c for c in color_pool if c not in used_colors] or color_pool
        color_name = random.choice(remaining)
        used_colors.append(color_name)
        color_rgb = COLORS[color_name]
        draw_color_block(bg, pos, base_size, color_rgb)
        targets.append({
            "type": "color_block", "value": color_name, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        })

    if count == 1:
        question = (
            "What is the color of the block in the image? "
            "(The background contains colored noise; ignore it and report "
            "the color of the block itself.) "
            f"Answer with one of: {_COLOR_BLOCK_CHOICES}."
        )
        answer = targets[0]["color"]
    else:
        direction, axis = _random_order()
        question = (
            f"The image contains {count} colored blocks on a noisy background. "
            f"List the colors of all blocks {direction}, separated by commas. "
            "(Ignore the background noise.) "
            f"Each color is one of: {_COLOR_BLOCK_CHOICES}."
        )
        sorted_t = sorted(targets, key=lambda t: t["position"][axis])
        answer = ", ".join(t["color"] for t in sorted_t)

    return {
        "image": bg,
        "task_type": "interference_cvd",
        "question": question,
        "answer": answer,
        "targets": targets,
        "extra": {"cvd_type": cvd_type, "interference_on": "background"},
    }


def _gen_blur(cw: int, ch: int, base_size: int, count: int = 1) -> dict[str, Any]:
    """Blur is applied to the BACKGROUND only; target stays crisp."""
    blur_radius = random.choice([8.0, 12.0, 16.0, 20.0])
    bg = create_textured_canvas(cw, ch, density=0.05)
    bg = apply_blur(bg, blur_radius)

    color_pool = [c for c in TARGET_COLORS if c not in ("black", "white", "gray")]
    positions = _place_non_overlapping(cw, ch, [base_size] * count, margin=20)
    targets = []
    used_colors: list[str] = []
    for pos in positions:
        remaining = [c for c in color_pool if c not in used_colors] or color_pool
        color_name = random.choice(remaining)
        used_colors.append(color_name)
        color_rgb = COLORS[color_name]
        draw_color_block(bg, pos, base_size, color_rgb)
        targets.append({
            "type": "color_block", "value": color_name, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        })

    if count == 1:
        question = (
            "What is the color of the block in the image? "
            "(The background is blurred and may contain distracting textures.) "
            f"Answer with one of: {_COLOR_BLOCK_CHOICES}."
        )
        answer = targets[0]["color"]
    else:
        direction, axis = _random_order()
        question = (
            f"The image contains {count} colored blocks on a blurred, textured "
            f"background. List the colors of all blocks {direction}, "
            "separated by commas. "
            f"Each color is one of: {_COLOR_BLOCK_CHOICES}."
        )
        sorted_t = sorted(targets, key=lambda t: t["position"][0])
        answer = ", ".join(t["color"] for t in sorted_t)

    return {
        "image": bg,
        "task_type": "interference_blur",
        "question": question,
        "answer": answer,
        "targets": targets,
        "extra": {"blur_radius": blur_radius, "interference_on": "background"},
    }


def _gen_chain(cw: int, ch: int, base_size: int, count: int | None = None) -> dict[str, Any]:
    n = count if count is not None else random.randint(3, 5)
    obj_type = random.choice(["letter", "shape", "dot"])
    positions = _place_non_overlapping(cw, ch, [base_size] * n)

    img = create_canvas(cw, ch)
    targets = []
    for pos in positions:
        color_name = random.choice(TARGET_COLORS)
        color_rgb = COLORS[color_name]
        value = random.choice(_VALUE_POOL[obj_type])
        _draw_target(img, obj_type, pos, base_size, color_rgb, value)
        targets.append({
            "type": obj_type, "value": value, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        })

    direction, axis = _random_order()
    sorted_targets = sorted(targets, key=lambda t: t["position"][axis])
    answer_order = [f"{t['color']} {t['value']}" for t in sorted_targets]

    identity_hint = {
        "letter": f" Each identity is an uppercase letter from: {_LETTER_CHOICES}.",
        "shape": f" Each identity is a shape from: {_SHAPE_CHOICES}.",
        "dot": " Each identity is simply 'dot'.",
    }[obj_type]

    return {
        "image": img,
        "task_type": "chain_reasoning",
        "question": (
            f"List all objects in the image {direction}. "
            "Describe each by its color and identity, in the form '<color> <identity>', "
            "separated by commas."
            f" Colors are drawn from: {_TARGET_COLOR_CHOICES}."
            f"{identity_hint}"
        ),
        "answer": ", ".join(answer_order),
        "targets": targets,
    }


_GENERATORS: dict[str, Callable] = {
    "comparison": _gen_comparison,
    "counting": _gen_counting,
    "spatial": _gen_spatial,
    "interference_cvd": _gen_cvd,
    "interference_blur": _gen_blur,
    "chain_reasoning": _gen_chain,
}

_CUSTOM_COUNT_TASKS: list[str] = [
    "counting", "spatial",
    "interference_cvd", "interference_blur",
    "chain_reasoning",
]


def generate_reasoning_dataset(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    num_per_config: int = 3,
    seed: int | None = 42,
    mode: str = "difficulty",
    counts: list[int] | None = None,
    target_size: int = 24,
) -> Path:
    """Generate the FineSight-Reasoning dataset.

    Modes
    -----
    * ``"difficulty"`` – iterate over ``sizes``; size determines difficulty.
    * ``"custom"`` – iterate over ``counts`` at fixed ``target_size``.
      ``comparison`` is skipped (always 2 targets).
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
        task_items = list(_GENERATORS.items())
    else:
        counts = counts or [2, 3, 5, 7, 9]
        config_axis = counts
        task_items = [(n, fn) for n, fn in _GENERATORS.items()
                      if n in _CUSTOM_COUNT_TASKS]

    for task_name, gen_fn in task_items:
        for cfg in config_axis:
            for _ in range(num_per_config):
                if mode == "difficulty":
                    size = cfg
                    result = gen_fn(canvas_size, canvas_size, size)
                    diff = _difficulty(size)
                    tag = f"{size}px"
                else:
                    size = target_size
                    result = gen_fn(canvas_size, canvas_size, size, count=cfg)
                    diff = _difficulty(size)
                    tag = f"n{cfg}"

                image = result.pop("image")
                image_id = f"reasoning_{task_name}_{tag}_{idx:05d}"
                rel_path = f"images/{image_id}.png"
                image.save(img_dir / f"{image_id}.png")

                sample: dict[str, Any] = {
                    "image_id": image_id,
                    "image_path": rel_path,
                    "dataset": "reasoning",
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
                if "extra" in result:
                    sample["metadata"]["extra"] = result["extra"]
                if "sub_answers" in result:
                    sample["metadata"]["sub_answers"] = result["sub_answers"]
                samples.append(sample)
                idx += 1

    dataset_meta = {
        "dataset_info": {
            "name": "FineSight-Reasoning",
            "version": "1.1",
            "description": "Fine-grained visual reasoning evaluation for VLMs",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": [t for t, _ in task_items],
            "generation_mode": mode,
            "target_sizes": sizes if mode == "difficulty" else [target_size],
            "target_counts": counts if mode == "custom" else None,
        },
        "samples": samples,
    }

    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False))
    print(f"[Reasoning] Generated {len(samples)} samples (mode={mode}) -> {out}")
    return labels_path
