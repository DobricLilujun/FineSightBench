"""One-off script: generate TextWild samples and merge them into the
existing ``data/full_perception`` and ``data/full_reasoning`` datasets.

Produces / updates:
  * images/<...>.png
  * labels.json   (plain-text answers)
  * labels.jsonl  (JSON-structured answers + answer_format="json")
  * metadata.csv  (flat inspection table)
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from finesightbench.core.objects import TARGET_SIZES
from finesightbench.textwild import (
    generate_textwild_perception,
    generate_textwild_reasoning,
)

ROOT = Path("/home/snt/projects_lujun/FineSightBench")
CANVAS = 448   # match existing full_* datasets
TMP_PERC = ROOT / "data" / "_tw_perc_tmp"
TMP_REAS = ROOT / "data" / "_tw_reas_tmp"
DST_PERC = ROOT / "data" / "full_perception"
DST_REAS = ROOT / "data" / "full_reasoning"

LETTER_LIST = ", ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ── JSON-answer prompts (mirroring existing full_* style) ───────────────────

def _json_prompt_text_recognition() -> str:
    return (
        "What single English word is written in the image? "
        "Answer in JSON format only: {\"word\": \"<WORD>\"} where <WORD> is the "
        "uppercase word. Ignore any text that may naturally appear in the "
        "background photograph."
    )


def _json_prompt_text_reading_chain(n: int, direction: str) -> str:
    return (
        f"The image contains {n} overlaid English words rendered on top of a "
        f"natural scene. List all the overlaid words {direction}. "
        "Answer in JSON format only: {\"words\": [\"<W1>\", \"<W2>\", ...]} "
        "where each <Wi> is an uppercase word. Ignore any text that may "
        "naturally appear in the background."
    )


def _json_prompt_text_counting_chain(n: int, letter: str) -> str:
    return (
        f"The image contains {n} overlaid English words rendered on top of a "
        "natural scene. Answer two questions: (1) how many overlaid words are "
        f"in the image in total, and (2) how many of those words contain the "
        f"letter '{letter}'. Answer in JSON format only: "
        "{\"total\": <int>, \"with_letter\": <int>}. Ignore any text that may "
        "naturally appear in the background."
    )


def _json_answer(sample: dict) -> tuple[dict, str]:
    """Return (json_answer_obj, json_question_string) for a textwild sample."""
    t = sample["task_type"]
    if t == "text_recognition":
        return {"word": sample["answer"]}, _json_prompt_text_recognition()
    if t == "text_reading_chain":
        words = [w.strip() for w in sample["answer"].split(",")]
        n = sample["metadata"]["num_targets"]
        direction = sample["metadata"]["extra"]["direction"]
        return {"words": words}, _json_prompt_text_reading_chain(n, direction)
    if t == "text_counting_chain":
        total_str, cnt_str = sample["answer"].split(";")
        total = int(total_str.strip())
        cnt = int(cnt_str.strip())
        letter = sample["metadata"]["sub_answers"]["letter"]
        return {"total": total, "with_letter": cnt}, \
            _json_prompt_text_counting_chain(total, letter)
    raise ValueError(f"unknown task_type: {t}")


# ── merge helpers ───────────────────────────────────────────────────────────

def _copy_images(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("*.png"):
        shutil.copy2(p, dst_dir / p.name)


def _merge_labels_json(dst_json: Path, new_samples: list[dict], new_task_types: list[str]) -> None:
    data = json.loads(dst_json.read_text())
    existing_ids = {s["image_id"] for s in data["samples"]}
    added = 0
    for s in new_samples:
        if s["image_id"] in existing_ids:
            continue
        data["samples"].append(s)
        added += 1
    info = data["dataset_info"]
    info["num_samples"] = len(data["samples"])
    info["task_types"] = list(dict.fromkeys(info.get("task_types", []) + new_task_types))
    info["textwild_merged"] = True
    dst_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[labels.json] merged +{added} -> {dst_json} (total={info['num_samples']})")


def _merge_labels_jsonl(dst_jsonl: Path, new_samples: list[dict]) -> None:
    # Read existing to dedupe by image_id.
    existing_ids: set[str] = set()
    if dst_jsonl.exists():
        for line in dst_jsonl.read_text().splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["image_id"])
    added = 0
    with dst_jsonl.open("a") as f:
        for s in new_samples:
            if s["image_id"] in existing_ids:
                continue
            ans_obj, json_q = _json_answer(s)
            row = {
                "image_id": s["image_id"],
                "image_path": s["image_path"],
                "dataset": s["dataset"],
                "task_type": s["task_type"],
                "question": json_q,
                "answer": ans_obj,
                "difficulty": s["difficulty"],
                "generation_mode": s["generation_mode"],
                "metadata": s["metadata"],
                "answer_format": "json",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    print(f"[labels.jsonl] appended +{added} -> {dst_jsonl}")


def _append_metadata_csv(
    dst_csv: Path, new_samples: list[dict], has_num_targets: bool,
) -> None:
    # We assume the CSV already exists with a header.
    exists = dst_csv.exists()
    with dst_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            # Shouldn't happen for full_*, but be safe.
            header = ["image_id", "image_path", "task_type", "pixel_size"]
            if has_num_targets:
                header.append("num_targets")
            header += ["difficulty", "question", "answer"]
            w.writerow(header)
        added = 0
        for s in new_samples:
            size = s["metadata"]["targets"][0]["size"]
            row = [s["image_id"], s["image_path"], s["task_type"], size]
            if has_num_targets:
                row.append(s["metadata"].get("num_targets", 1))
            row += [s["difficulty"], s["question"], s["answer"]]
            w.writerow(row)
            added += 1
    print(f"[metadata.csv] appended +{added} -> {dst_csv}")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # ---- 1. Generate into tmp dirs (canvas 448 to match full_*) -----------
    print("=== Generating TextWild Perception ===")
    generate_textwild_perception(
        output_dir=TMP_PERC,
        canvas_size=CANVAS,
        sizes=TARGET_SIZES,      # [4,8,12,16,24,32,48]
        num_per_size=100,         # 7 * 100 = 700
        seed=42,
    )

    print("\n=== Generating TextWild Reasoning ===")
    generate_textwild_reasoning(
        output_dir=TMP_REAS,
        canvas_size=CANVAS,
        sizes=TARGET_SIZES,
        counts=[2, 4, 6, 8],
        num_per_config=25,        # 7 * 4 * 25 = 700 per task, ×2 tasks = 1400
        seed=43,
    )

    # ---- 2. Merge ---------------------------------------------------------
    print("\n=== Merging into data/full_perception ===")
    perc_data = json.loads((TMP_PERC / "labels.json").read_text())
    _copy_images(TMP_PERC / "images", DST_PERC / "images")
    _merge_labels_json(
        DST_PERC / "labels.json",
        perc_data["samples"],
        new_task_types=["text_recognition"],
    )
    _merge_labels_jsonl(DST_PERC / "labels.jsonl", perc_data["samples"])
    _append_metadata_csv(DST_PERC / "metadata.csv", perc_data["samples"],
                         has_num_targets=False)

    print("\n=== Merging into data/full_reasoning ===")
    reas_data = json.loads((TMP_REAS / "labels.json").read_text())
    _copy_images(TMP_REAS / "images", DST_REAS / "images")
    _merge_labels_json(
        DST_REAS / "labels.json",
        reas_data["samples"],
        new_task_types=["text_reading_chain", "text_counting_chain"],
    )
    _merge_labels_jsonl(DST_REAS / "labels.jsonl", reas_data["samples"])
    _append_metadata_csv(DST_REAS / "metadata.csv", reas_data["samples"],
                         has_num_targets=True)

    # ---- 3. Cleanup tmp ---------------------------------------------------
    shutil.rmtree(TMP_PERC, ignore_errors=True)
    shutil.rmtree(TMP_REAS, ignore_errors=True)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
