"""Generator for **FineSight-Reasoning** dataset.

Task families
─────────────
1. comparison   – which of two targets is larger / smaller
2. counting     – how many targets of a given type
3. spatial      – locate a target (quadrant / relative position)
4. interference – colour-blindness & blur robustness
5. chain        – multi-target spatial chain reasoning
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from finesightbench.core.canvas import create_canvas
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

# ── helpers ──────────────────────────────────────────────────────────────────

_DRAW_FN: dict[str, Callable] = {
    "letter": draw_letter,
    "animal": draw_animal,
    "block": draw_block,
    "color_block": draw_color_block,
    "shape": draw_shape,
    "dot": draw_dot,
}

_VALUE_POOL: dict[str, list[str]] = {
    "letter": LETTERS,
    "animal": ANIMAL_TYPES,
    "block": ["block"],
    "color_block": ["color_block"],
    "shape": SHAPE_TYPES,
    "dot": ["dot"],
}


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
    """Return non-overlapping positions for targets with given *sizes*."""
    positions: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []  # x1,y1,x2,y2

    for s in sizes:
        for _ in range(max_attempts):
            x, y = random_position(canvas_w, canvas_h, s, margin)
            box = (x, y, x + s, y + s)
            overlap = False
            for pb in placed_boxes:
                if not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3]):
                    overlap = True
                    break
            if not overlap:
                positions.append((x, y))
                placed_boxes.append(box)
                break
        else:
            # fallback: place anyway
            positions.append(random_position(canvas_w, canvas_h, s, margin))
    return positions


def _draw_target(
    image, obj_type: str, pos: tuple[int, int], size: int,
    color_rgb: tuple[int, int, int], value: str | None = None,
) -> None:
    """Dispatch to the correct draw function."""
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def _gen_comparison(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    """Two targets of different sizes, left vs right – which is larger?"""
    obj_type = random.choice(["letter", "animal", "block", "shape", "dot"])
    value = random.choice(_VALUE_POOL.get(obj_type, ["x"]))
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]

    ratio = random.choice([0.5, 0.6, 0.75, 1.5, 1.7, 2.0])
    size_a = base_size
    size_b = max(3, int(base_size * ratio))
    if size_a == size_b:
        size_b = size_a + 2

    # place left / right halves
    margin = 10
    ax = random.randint(margin, cw // 2 - size_a - margin)
    ay = random.randint(margin, ch - size_a - margin)
    bx = random.randint(cw // 2 + margin, max(cw // 2 + margin + 1, cw - size_b - margin))
    by = random.randint(margin, ch - size_b - margin)

    img = create_canvas(cw, ch)
    _draw_target(img, obj_type, (ax, ay), size_a, color_rgb, value)
    _draw_target(img, obj_type, (bx, by), size_b, color_rgb, value)

    if size_a > size_b:
        answer = "left"
    elif size_b > size_a:
        answer = "right"
    else:
        answer = "same"

    return {
        "image": img,
        "task_type": "comparison",
        "question": "Which object is larger, the one on the left or the one on the right?",
        "answer": answer,
        "targets": [
            {"type": obj_type, "value": value, "size": size_a,
             "position": [ax, ay], "color": color_name, "color_rgb": list(color_rgb), "side": "left"},
            {"type": obj_type, "value": value, "size": size_b,
             "position": [bx, by], "color": color_name, "color_rgb": list(color_rgb), "side": "right"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. COUNTING
# ═══════════════════════════════════════════════════════════════════════════

def _gen_counting(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    obj_type = random.choice(["letter", "animal", "block", "color_block", "shape"])
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    count = random.randint(2, 9)
    sizes_list = [base_size] * count
    positions = _place_non_overlapping(cw, ch, sizes_list)

    img = create_canvas(cw, ch)
    targets = []
    for i, pos in enumerate(positions):
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

    return {
        "image": img,
        "task_type": "counting",
        "question": f"How many {type_label} are there in the image?",
        "answer": str(count),
        "targets": targets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. SPATIAL (find position / quadrant)
# ═══════════════════════════════════════════════════════════════════════════

def _gen_spatial(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    obj_type = random.choice(["letter", "animal", "block", "color_block", "shape"])
    value = random.choice(_VALUE_POOL[obj_type])
    color_name = random.choice(TARGET_COLORS)
    color_rgb = COLORS[color_name]
    pos = random_position(cw, ch, base_size)
    quad = _quadrant(pos[0] + base_size // 2, pos[1] + base_size // 2, cw, ch)

    img = create_canvas(cw, ch)
    _draw_target(img, obj_type, pos, base_size, color_rgb, value)

    return {
        "image": img,
        "task_type": "spatial",
        "question": "In which quadrant of the image is the object located? "
                    "Answer: top-left, top-right, bottom-left, or bottom-right.",
        "answer": quad,
        "targets": [{
            "type": obj_type, "value": value, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4-a. INTERFERENCE – colour-vision deficiency
# ═══════════════════════════════════════════════════════════════════════════

def _gen_cvd(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    color_name = random.choice([c for c in TARGET_COLORS if c not in ("black", "white", "gray")])
    color_rgb = COLORS[color_name]
    pos = random_position(cw, ch, base_size)
    cvd_type = random.choice(["protanopia", "deuteranopia", "tritanopia"])

    img = create_canvas(cw, ch)
    draw_color_block(img, pos, base_size, color_rgb)
    img = simulate_cvd(img, cvd_type)

    return {
        "image": img,
        "task_type": "interference_cvd",
        "question": "What is the original color of the block in the image? "
                    "(The image may have been altered by a color-vision simulation.)",
        "answer": color_name,
        "targets": [{
            "type": "color_block", "value": color_name, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        }],
        "extra": {"cvd_type": cvd_type},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4-b. INTERFERENCE – background blur
# ═══════════════════════════════════════════════════════════════════════════

def _gen_blur(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    color_name = random.choice([c for c in TARGET_COLORS if c not in ("black", "white", "gray")])
    color_rgb = COLORS[color_name]
    pos = random_position(cw, ch, base_size)
    blur_radius = random.choice([3.0, 5.0, 8.0, 12.0])

    img = create_canvas(cw, ch)
    draw_color_block(img, pos, base_size, color_rgb)
    img = apply_blur(img, blur_radius)

    return {
        "image": img,
        "task_type": "interference_blur",
        "question": "What color is the block in the image? (The image may be blurred.)",
        "answer": color_name,
        "targets": [{
            "type": "color_block", "value": color_name, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        }],
        "extra": {"blur_radius": blur_radius},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. CHAIN REASONING (multi-target spatial)
# ═══════════════════════════════════════════════════════════════════════════

def _gen_chain(cw: int, ch: int, base_size: int) -> dict[str, Any]:
    """Place 3-5 labelled targets; ask to list them left-to-right."""
    n = random.randint(3, 5)
    obj_type = random.choice(["letter", "shape", "dot"])
    sizes_list = [base_size] * n
    positions = _place_non_overlapping(cw, ch, sizes_list)

    img = create_canvas(cw, ch)
    targets = []
    for i, pos in enumerate(positions):
        color_name = random.choice(TARGET_COLORS)
        color_rgb = COLORS[color_name]
        value = random.choice(_VALUE_POOL[obj_type])
        _draw_target(img, obj_type, pos, base_size, color_rgb, value)
        targets.append({
            "type": obj_type, "value": value, "size": base_size,
            "position": list(pos), "color": color_name, "color_rgb": list(color_rgb),
        })

    # sort targets left-to-right by x position
    sorted_targets = sorted(targets, key=lambda t: t["position"][0])
    answer_order = [f"{t['color']} {t['value']}" for t in sorted_targets]

    return {
        "image": img,
        "task_type": "chain_reasoning",
        "question": "List all objects in the image from left to right. "
                    "Describe each by its color and identity.",
        "answer": ", ".join(answer_order),
        "targets": targets,
    }


# ── task registry ────────────────────────────────────────────────────────────
_GENERATORS: dict[str, Callable] = {
    "comparison": _gen_comparison,
    "counting": _gen_counting,
    "spatial": _gen_spatial,
    "interference_cvd": _gen_cvd,
    "interference_blur": _gen_blur,
    "chain_reasoning": _gen_chain,
}


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def generate_reasoning_dataset(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    num_per_config: int = 3,
    seed: int | None = 42,
) -> Path:
    """Generate the full FineSight-Reasoning dataset.

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

                image_id = f"reasoning_{task_name}_{size}px_{idx:05d}"
                rel_path = f"images/{image_id}.png"
                image.save(img_dir / f"{image_id}.png")

                sample: dict[str, Any] = {
                    "image_id": image_id,
                    "image_path": rel_path,
                    "dataset": "reasoning",
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
                if "extra" in result:
                    sample["metadata"]["extra"] = result["extra"]
                samples.append(sample)
                idx += 1

    dataset_meta = {
        "dataset_info": {
            "name": "FineSight-Reasoning",
            "version": "1.0",
            "description": "Fine-grained visual reasoning evaluation for VLMs",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": list(_GENERATORS.keys()),
            "target_sizes": sizes,
        },
        "samples": samples,
    }

    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False))
    print(f"[Reasoning] Generated {len(samples)} samples → {out}")
    return labels_path
