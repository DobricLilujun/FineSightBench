"""SynthText-style text-in-the-wild task generators.

Tasks
-----
Perception:
    * ``text_recognition`` — read a single word rendered on a natural-scene
      background at a controlled character pixel height.

Reasoning:
    * ``text_reading_chain`` — N words on a natural-scene background; list
      them from left→right or top→bottom.
    * ``text_counting_chain`` — N words on a natural-scene background; count
      words that contain a specified letter (sub-task also reports total N).

All tasks use character cap-height in pixels as the difficulty axis, matching
the rest of FineSightBench (``TARGET_SIZES``).
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from finesightbench.core.objects import TARGET_SIZES

from .backgrounds import list_backgrounds, sample_background
from .text_renderer import COUNTING_TARGETS, WORD_BANK, render_words_on_image


def _difficulty(size: int) -> str:
    if size <= 5:
        return "extreme"
    if size <= 12:
        return "hard"
    if size <= 24:
        return "medium"
    return "easy"


def _item_to_target_dict(item) -> dict[str, Any]:
    return {
        "type": "text",
        "value": item.word,
        "size": item.size_px,
        "position": list(item.position),
        "bbox": list(item.bbox),
        "color": item.color,
        "color_rgb": list(item.color_rgb),
        "stroke": item.stroke,
    }


def _ordering_direction(rng: random.Random) -> tuple[str, int]:
    if rng.random() < 0.5:
        return "from left to right", 0
    return "from top to bottom", 1


# ── single-sample generators ────────────────────────────────────────────────

def _gen_text_recognition(
    canvas_size: int, cap_px: int, bg_paths: list[Path], rng: random.Random,
) -> dict[str, Any] | None:
    word = rng.choice(WORD_BANK)
    # Try a few random backgrounds in case placement fails.
    for _ in range(4):
        bg = sample_background(bg_paths, canvas_size, rng)
        img, items = render_words_on_image(bg, [word], cap_px, rng)
        if img is not None:
            break
    else:
        return None
    question = (
        "What single English word is written in the image? "
        "Answer with the word in uppercase letters. Ignore any text that may "
        "naturally appear in the photograph and focus on the overlaid word."
    )
    return {
        "image": img,
        "task_type": "text_recognition",
        "question": question,
        "answer": word,
        "targets": [_item_to_target_dict(items[0])],
    }


def _gen_text_reading_chain(
    canvas_size: int, cap_px: int, n: int, bg_paths: list[Path], rng: random.Random,
) -> dict[str, Any] | None:
    words = rng.sample(WORD_BANK, n)
    for _ in range(6):
        bg = sample_background(bg_paths, canvas_size, rng)
        img, items = render_words_on_image(bg, words, cap_px, rng)
        if img is not None:
            break
    else:
        return None

    direction, axis = _ordering_direction(rng)
    sorted_items = sorted(items, key=lambda t: t.position[axis])
    answer = ", ".join(it.word for it in sorted_items)

    question = (
        f"The image contains {n} overlaid English words rendered on top of a "
        f"natural scene. List all the overlaid words {direction}, separated by "
        f"commas, in uppercase. Ignore any text that may naturally appear in "
        f"the background photograph."
    )
    return {
        "image": img,
        "task_type": "text_reading_chain",
        "question": question,
        "answer": answer,
        "targets": [_item_to_target_dict(it) for it in items],
        "extra": {"direction": direction, "order_axis": axis, "num_words": n},
    }


def _gen_text_counting_chain(
    canvas_size: int, cap_px: int, n: int, bg_paths: list[Path], rng: random.Random,
) -> dict[str, Any] | None:
    words = rng.sample(WORD_BANK, n)
    # Pick a query letter that actually partitions the set (avoid 0 or n).
    # Try several letters; if none partition, fall back to any with count >= 1.
    letters_shuffled = rng.sample(COUNTING_TARGETS, len(COUNTING_TARGETS))
    chosen_letter: str | None = None
    chosen_count = 0
    for L in letters_shuffled:
        c = sum(1 for w in words if L in w)
        if 0 < c < n:
            chosen_letter = L
            chosen_count = c
            break
    if chosen_letter is None:
        # Fallback: pick the most frequent letter among words.
        scored = [(L, sum(1 for w in words if L in w)) for L in COUNTING_TARGETS]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_letter, chosen_count = scored[0]

    for _ in range(6):
        bg = sample_background(bg_paths, canvas_size, rng)
        img, items = render_words_on_image(bg, words, cap_px, rng)
        if img is not None:
            break
    else:
        return None

    question = (
        f"The image contains {n} overlaid English words rendered on top of a "
        f"natural scene. Answer two questions. (1) How many overlaid words are "
        f"in the image in total? (2) How many of those words contain the "
        f"letter '{chosen_letter}'? Answer in the format: '<total>; <count>'. "
        f"Ignore any text that may naturally appear in the background."
    )
    answer = f"{n}; {chosen_count}"
    return {
        "image": img,
        "task_type": "text_counting_chain",
        "question": question,
        "answer": answer,
        "targets": [_item_to_target_dict(it) for it in items],
        "sub_answers": {"total": n, "with_letter": chosen_count, "letter": chosen_letter},
    }


# ── dataset-level generators ────────────────────────────────────────────────

def generate_textwild_perception(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    num_per_size: int = 100,
    seed: int = 42,
    bg_dir: str | Path | None = None,
) -> Path:
    """Generate the text-in-the-wild perception dataset (``text_recognition``).

    Total samples = ``len(sizes) * num_per_size`` (default 7 × 100 = 700).
    """
    rng = random.Random(seed)
    sizes = sizes or TARGET_SIZES
    bg_paths = list_backgrounds(bg_dir)

    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    idx = 0
    for cap_px in sizes:
        produced = 0
        attempts = 0
        while produced < num_per_size and attempts < num_per_size * 5:
            attempts += 1
            res = _gen_text_recognition(canvas_size, cap_px, bg_paths, rng)
            if res is None:
                continue
            image = res.pop("image")
            image_id = f"textwild_perception_text_recognition_{cap_px}px_{idx:05d}"
            rel_path = f"images/{image_id}.png"
            image.save(img_dir / f"{image_id}.png")
            samples.append({
                "image_id": image_id,
                "image_path": rel_path,
                "dataset": "perception",
                "task_type": res["task_type"],
                "question": res["question"],
                "answer": res["answer"],
                "difficulty": _difficulty(cap_px),
                "generation_mode": "difficulty",
                "metadata": {
                    "canvas_size": [canvas_size, canvas_size],
                    "background_source": "natural_scene",
                    "targets": res["targets"],
                },
            })
            idx += 1
            produced += 1

    meta = {
        "dataset_info": {
            "name": "FineSight-TextWild-Perception",
            "version": "1.0",
            "description": "Text-in-the-wild perception (SynthText-style word recognition)",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": ["text_recognition"],
            "generation_mode": "difficulty",
            "target_sizes": sizes,
        },
        "samples": samples,
    }
    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[TextWild-Perception] Generated {len(samples)} samples -> {out}")
    return labels_path


def generate_textwild_reasoning(
    output_dir: str | Path,
    canvas_size: int = 512,
    sizes: list[int] | None = None,
    counts: list[int] | None = None,
    num_per_config: int = 25,
    seed: int = 42,
    bg_dir: str | Path | None = None,
) -> Path:
    """Generate the text-in-the-wild reasoning dataset.

    Tasks: ``text_reading_chain`` + ``text_counting_chain``.
    Total samples per task ≈ ``len(sizes) * len(counts) * num_per_config``
    (default 7 × 4 × 25 = 700 per task; ~1400 total).
    """
    rng = random.Random(seed)
    sizes = sizes or TARGET_SIZES
    counts = counts or [2, 4, 6, 8]
    bg_paths = list_backgrounds(bg_dir)

    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    task_fns = {
        "text_reading_chain": _gen_text_reading_chain,
        "text_counting_chain": _gen_text_counting_chain,
    }

    samples: list[dict[str, Any]] = []
    idx = 0
    for task_name, fn in task_fns.items():
        for cap_px in sizes:
            for n in counts:
                produced = 0
                attempts = 0
                budget = num_per_config * 6
                while produced < num_per_config and attempts < budget:
                    attempts += 1
                    res = fn(canvas_size, cap_px, n, bg_paths, rng)
                    if res is None:
                        continue
                    image = res.pop("image")
                    image_id = f"textwild_reasoning_{task_name}_{cap_px}px_n{n}_{idx:05d}"
                    rel_path = f"images/{image_id}.png"
                    image.save(img_dir / f"{image_id}.png")
                    sample: dict[str, Any] = {
                        "image_id": image_id,
                        "image_path": rel_path,
                        "dataset": "reasoning",
                        "task_type": res["task_type"],
                        "question": res["question"],
                        "answer": res["answer"],
                        "difficulty": _difficulty(cap_px),
                        "generation_mode": "difficulty_x_count",
                        "metadata": {
                            "canvas_size": [canvas_size, canvas_size],
                            "background_source": "natural_scene",
                            "num_targets": n,
                            "targets": res["targets"],
                        },
                    }
                    if "extra" in res:
                        sample["metadata"]["extra"] = res["extra"]
                    if "sub_answers" in res:
                        sample["metadata"]["sub_answers"] = res["sub_answers"]
                    samples.append(sample)
                    idx += 1
                    produced += 1
                if produced < num_per_config:
                    print(
                        f"[TextWild-Reasoning] warn: only {produced}/{num_per_config} "
                        f"samples produced for {task_name} {cap_px}px n={n} "
                        f"(canvas too small for non-overlapping placement?)"
                    )

    meta = {
        "dataset_info": {
            "name": "FineSight-TextWild-Reasoning",
            "version": "1.0",
            "description": "Text-in-the-wild reasoning (reading order + letter-count chain)",
            "creation_date": datetime.now().isoformat(),
            "num_samples": len(samples),
            "task_types": list(task_fns.keys()),
            "generation_mode": "difficulty_x_count",
            "pixel_sizes": sizes,
            "count_configs": {t: counts for t in task_fns},
        },
        "samples": samples,
    }
    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[TextWild-Reasoning] Generated {len(samples)} samples -> {out}")
    return labels_path
