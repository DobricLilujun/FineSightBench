"""Unified VLM evaluation framework for FineSightBench val_data.

This module provides:
1) Dataset suitability/usability validation for data/val_data
2) Multi-model benchmark with accuracy metrics
3) Raw attention extraction and GIF visualization smoke tests
4) CLI + notebook-friendly API
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

import matplotlib

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    attn_implementation: str | None = "eager"
    max_new_tokens: int = 1024
    notes: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    # Qwen3-VL
    "Qwen3-VL-2B-Instruct": ModelSpec(
        name="Qwen3-VL-2B-Instruct",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "Qwen3-VL-4B-Instruct": ModelSpec(
        name="Qwen3-VL-4B-Instruct",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "Qwen3-VL-8B-Instruct": ModelSpec(
        name="Qwen3-VL-8B-Instruct",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "Qwen3-VL-30B-A3B-Instruct": ModelSpec(
        name="Qwen3-VL-30B-A3B-Instruct",
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    # Llama-4
    "Llama-4-Scout-17B-16E-Instruct": ModelSpec(
        name="Llama-4-Scout-17B-16E-Instruct",
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
        notes="Gated model on Hugging Face; approval required.",
    ),
    "Llama-4-Maverick-17B-128E-Instruct": ModelSpec(
        name="Llama-4-Maverick-17B-128E-Instruct",
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
        notes="Gated model on Hugging Face; approval required.",
    ),
    # InternVL3.5-Flash
    "InternVL3_5-1B-Flash": ModelSpec(
        name="InternVL3_5-1B-Flash",
        model_id="OpenGVLab/InternVL3_5-1B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-2B-Flash": ModelSpec(
        name="InternVL3_5-2B-Flash",
        model_id="OpenGVLab/InternVL3_5-2B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-4B-Flash": ModelSpec(
        name="InternVL3_5-4B-Flash",
        model_id="OpenGVLab/InternVL3_5-4B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-8B-Flash": ModelSpec(
        name="InternVL3_5-8B-Flash",
        model_id="OpenGVLab/InternVL3_5-8B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-14B-Flash": ModelSpec(
        name="InternVL3_5-14B-Flash",
        model_id="OpenGVLab/InternVL3_5-14B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-30B-A3B-Flash": ModelSpec(
        name="InternVL3_5-30B-A3B-Flash",
        model_id="OpenGVLab/InternVL3_5-30B-A3B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "InternVL3_5-38B-Flash": ModelSpec(
        name="InternVL3_5-38B-Flash",
        model_id="OpenGVLab/InternVL3_5-38B-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    # DeepSeek-VL2
    "deepseek-vl2-tiny": ModelSpec(
        name="deepseek-vl2-tiny",
        model_id="deepseek-ai/deepseek-vl2-tiny",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "deepseek-vl2-small": ModelSpec(
        name="deepseek-vl2-small",
        model_id="deepseek-ai/deepseek-vl2-small",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    # GLM-4.6V
    "GLM-4.6V-Flash": ModelSpec(
        name="GLM-4.6V-Flash",
        model_id="zai-org/GLM-4.6V-Flash",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "GLM-4.6V": ModelSpec(
        name="GLM-4.6V",
        model_id="zai-org/GLM-4.6V",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    # Gemma-4
    "gemma-4-E2B-it": ModelSpec(
        name="gemma-4-E2B-it",
        model_id="google/gemma-4-E2B-it",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "gemma-4-E4B-it": ModelSpec(
        name="gemma-4-E4B-it",
        model_id="google/gemma-4-E4B-it",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "gemma-4-26B-A4B-it": ModelSpec(
        name="gemma-4-26B-A4B-it",
        model_id="google/gemma-4-26B-A4B-it",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
    "gemma-4-31B-it": ModelSpec(
        name="gemma-4-31B-it",
        model_id="google/gemma-4-31B-it",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
    ),
}


def _norm_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_MODEL_NAME_ALIASES: dict[str, str] = {}
for _name, _spec in MODEL_SPECS.items():
    _MODEL_NAME_ALIASES[_norm_model_name(_name)] = _name
    # Also alias by the HF model_id (e.g. "google/gemma-4-E4B-it" -> "gemma-4-E4B-it")
    _MODEL_NAME_ALIASES[_norm_model_name(_spec.model_id)] = _name
    # And by just the repo name part after the slash
    if "/" in _spec.model_id:
        _MODEL_NAME_ALIASES[_norm_model_name(_spec.model_id.split("/", 1)[1])] = _name


def list_supported_models() -> list[str]:
    return list(MODEL_SPECS.keys())


def resolve_model_name(name: str) -> str:
    if name in MODEL_SPECS:
        return name
    key = _norm_model_name(name)
    if key in _MODEL_NAME_ALIASES:
        return _MODEL_NAME_ALIASES[key]
    raise KeyError(
        f"Unknown model '{name}'. Supported: {', '.join(list_supported_models())}"
    )


# -----------------------------------------------------------------------------
# Dataset loading and validation
# -----------------------------------------------------------------------------


@dataclass
class EvalSample:
    split: str
    label_file: str
    image_id: str
    image_path: str
    question: str
    answer: str
    task_type: str
    metadata: dict[str, Any]


def _iter_label_files(val_root: Path) -> list[Path]:
    return sorted(val_root.rglob("labels.json"))


def _split_name(val_root: Path, label_file: Path) -> str:
    rel_parent = label_file.parent.relative_to(val_root)
    return rel_parent.as_posix() if str(rel_parent) != "." else "root"


def _default_question_from_sample(sample: dict[str, Any]) -> str | None:
    if "animal" in sample:
        return (
            "What animal is shown in the image? "
            "Answer with one of: cat, dog, fish, bird, rabbit, turtle."
        )
    return None


def _to_eval_sample(val_root: Path, label_file: Path, sample: dict[str, Any]) -> EvalSample | None:
    image_rel = sample.get("image_path")
    image_id = sample.get("image_id")
    if not image_rel or not image_id:
        return None

    image_path = (label_file.parent / image_rel).resolve()
    question = sample.get("question") or _default_question_from_sample(sample)
    answer = sample.get("answer")
    if answer is None and sample.get("animal") is not None:
        answer = str(sample.get("animal"))

    if not question or answer is None:
        return None

    task_type = sample.get("task_type")
    if not task_type:
        if sample.get("animal") is not None:
            task_type = "animal_recognition"
        else:
            task_type = "unknown"

    return EvalSample(
        split=_split_name(val_root, label_file),
        label_file=str(label_file),
        image_id=str(image_id),
        image_path=str(image_path),
        question=str(question),
        answer=str(answer),
        task_type=str(task_type),
        metadata=sample.get("metadata") or {},
    )


def load_eval_samples(val_root: str | Path) -> list[EvalSample]:
    val_root = Path(val_root).resolve()
    samples: list[EvalSample] = []
    for label_file in _iter_label_files(val_root):
        data = json.loads(label_file.read_text())
        for sample in data.get("samples", []):
            rec = _to_eval_sample(val_root, label_file, sample)
            if rec is not None:
                samples.append(rec)
    return samples


def validate_val_dataset(val_root: str | Path) -> dict[str, Any]:
    """Validate val_data usability and return summary stats."""
    val_root = Path(val_root).resolve()

    summary: dict[str, Any] = {
        "val_root": str(val_root),
        "label_files": 0,
        "raw_samples": 0,
        "evaluable_samples": 0,
        "missing_required_fields": 0,
        "missing_images": 0,
        "corrupted_images": 0,
        "missing_questions_or_answers": 0,
        "split_counts": {},
        "task_counts": {},
        "issues": [],
    }

    for label_file in _iter_label_files(val_root):
        summary["label_files"] += 1
        split = _split_name(val_root, label_file)

        try:
            data = json.loads(label_file.read_text())
        except Exception as exc:
            summary["issues"].append(
                {
                    "type": "invalid_json",
                    "label_file": str(label_file),
                    "error": str(exc),
                }
            )
            continue

        samples = data.get("samples", [])
        summary["raw_samples"] += len(samples)

        for s in samples:
            if not s.get("image_id") or not s.get("image_path"):
                summary["missing_required_fields"] += 1
                summary["issues"].append(
                    {
                        "type": "missing_required_fields",
                        "label_file": str(label_file),
                        "sample": s,
                    }
                )
                continue

            rec = _to_eval_sample(val_root, label_file, s)
            if rec is None:
                summary["missing_questions_or_answers"] += 1
                continue

            image_path = Path(rec.image_path)
            if not image_path.exists():
                summary["missing_images"] += 1
                summary["issues"].append(
                    {
                        "type": "missing_image",
                        "split": split,
                        "image_id": rec.image_id,
                        "image_path": str(image_path),
                    }
                )
                continue

            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as exc:
                summary["corrupted_images"] += 1
                summary["issues"].append(
                    {
                        "type": "corrupted_image",
                        "split": split,
                        "image_id": rec.image_id,
                        "image_path": str(image_path),
                        "error": str(exc),
                    }
                )
                continue

            summary["evaluable_samples"] += 1
            summary["split_counts"][split] = summary["split_counts"].get(split, 0) + 1
            summary["task_counts"][rec.task_type] = (
                summary["task_counts"].get(rec.task_type, 0) + 1
            )

    summary["dataset_usable"] = (
        summary["evaluable_samples"] > 0
        and summary["missing_images"] == 0
        and summary["corrupted_images"] == 0
    )
    return summary


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9,;|\-\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = _SPECIAL_TOKEN_RE.sub(" ", t)
    t = t.replace("<", " ").replace(">", " ")
    t = t.replace("/", " ")
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def _parse_first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group(0))
    return None


def _parse_comparison(text: str) -> tuple[int | None, list[str] | None]:
    """Parse new comparison format: '<count>; <pos1>, <pos2>, ...'

    Positions are ordinals such as '1st', '2nd', '3rd', '4th', '5th'.
    """
    t = normalize_text(text)
    m = re.search(r"(\d+)\s*[;:]\s*((?:\d+(?:st|nd|rd|th)[\s,]*)+)", t)
    if m:
        count = int(m.group(1))
        ordinals = re.findall(r"\d+(?:st|nd|rd|th)", m.group(2))
        return count, ordinals
    count = _parse_first_int(t)
    return count, None


def _split_list(text: str) -> list[str]:
    t = normalize_text(text)
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return parts


def is_correct_prediction(pred: str, gt: str, task_type: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gt)

    if p == g:
        return True

    if task_type == "comparison":
        p_count, p_order = _parse_comparison(p)
        g_count, g_order = _parse_comparison(g)
        return p_count == g_count and p_order == g_order

    # Pure numeric answers (counting variants)
    if re.fullmatch(r"\d+", g):
        return _parse_first_int(p) == int(g)

    # Ordered list answers (custom / chain tasks)
    if "," in g:
        return _split_list(p) == _split_list(g)

    # yes/no style questions
    if g in {"yes", "no"}:
        return g in p.split()

    # Single-token targets (animals/shapes/colors/letters/quadrants)
    if len(g.split()) == 1:
        return g in p.split()

    # Fallback lenient check
    return g in p


def _aggregate_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    bucket: dict[str, list[bool]] = {}
    for row in rows:
        name = row.get(key, "unknown")
        bucket.setdefault(name, []).append(bool(row.get("correct", False)))

    out: dict[str, dict[str, float]] = {}
    for name, vals in bucket.items():
        n = len(vals)
        acc = float(sum(vals)) / float(n) if n else 0.0
        out[name] = {"n": n, "accuracy": acc}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


# -----------------------------------------------------------------------------
# Attention utilities
# -----------------------------------------------------------------------------


def find_visual_token_range(input_ids: torch.Tensor, tokenizer: Any) -> tuple[int, int]:
    ids = input_ids[0].tolist()

    def token_id_valid(tid: Any) -> bool:
        return tid is not None and isinstance(tid, int) and tid >= 0

    boundary_candidates = [
        ("<|vision_start|>", "<|vision_end|>"),
        ("<|IMAGE_START|>", "<|IMAGE_END|>"),
        ("<image>", "</image>"),
    ]

    vision_start_id = None
    vision_end_id = None
    for start_tok, end_tok in boundary_candidates:
        s_id = tokenizer.convert_tokens_to_ids(start_tok)
        e_id = tokenizer.convert_tokens_to_ids(end_tok)
        if token_id_valid(s_id) and s_id in ids and token_id_valid(e_id):
            vision_start_id = s_id
            vision_end_id = e_id
            break

    if vision_start_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if token_id_valid(pad_id) and pad_id in ids:
            first = ids.index(pad_id)
            last = len(ids) - 1 - ids[::-1].index(pad_id)
            return first, last + 1
        raise RuntimeError("Cannot locate visual token boundaries in input_ids.")

    start_pos = ids.index(vision_start_id) + 1
    end_pos = ids.index(vision_end_id, start_pos)
    return start_pos, end_pos


def infer_visual_grid(inputs: dict[str, Any], num_vis_tokens: int) -> tuple[int, int, int]:
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"][0].tolist()
        t_dim, grid_h, grid_w = int(thw[0]), int(thw[1]), int(thw[2])

        merge = 2
        h_vis = (grid_h // merge) * t_dim
        w_vis = grid_w // merge
        if h_vis * w_vis == num_vis_tokens:
            return t_dim, h_vis, w_vis

        h_vis = grid_h * t_dim
        w_vis = grid_w
        if h_vis * w_vis == num_vis_tokens:
            return t_dim, h_vis, w_vis

    side = int(math.sqrt(max(num_vis_tokens, 1)))
    side = max(side, 1)
    h_vis = side
    w_vis = max(1, num_vis_tokens // side)
    return 1, h_vis, w_vis


def raw_attention_heatmap(
    all_layer_attns: np.ndarray,
    query_pos: int,
    vis_start: int,
    vis_end: int,
    grid_h: int,
    grid_w: int,
    layer: int = -1,
    head_reduction: str = "mean",
) -> np.ndarray:
    attn = all_layer_attns[layer]
    if head_reduction == "max":
        attn_2d = attn.max(axis=0)
    else:
        attn_2d = attn.mean(axis=0)

    vec = attn_2d[query_pos, vis_start:vis_end].astype(np.float64)

    vmin, vmax = vec.min(), vec.max()
    if vmax > vmin:
        vec = (vec - vmin) / (vmax - vmin)
    else:
        vec = np.zeros_like(vec)

    expected = grid_h * grid_w
    if len(vec) != expected:
        side = int(math.sqrt(max(len(vec), 1)))
        side = max(side, 1)
        if side * side <= len(vec):
            vec = vec[: side * side]
            vec2d = vec.reshape(side, side)
        else:
            vec2d = np.zeros((side, side), dtype=np.float64)
        vec_img = Image.fromarray((vec2d * 255).astype(np.uint8)).resize(
            (grid_w, grid_h), Image.BILINEAR
        )
        return np.asarray(vec_img, dtype=np.float64) / 255.0

    return vec.reshape(grid_h, grid_w)


def overlay_attention(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet",
) -> Image.Image:
    image = image.convert("RGB")
    h_map = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    h_arr = np.asarray(h_map, dtype=np.float64) / 255.0
    colored = (
        matplotlib.colormaps.get_cmap(colormap)(h_arr)[:, :, :3] * 255
    ).astype(np.uint8)
    img_arr = np.asarray(image, dtype=np.float64)
    blended = (alpha * colored.astype(np.float64) + (1.0 - alpha) * img_arr).astype(
        np.uint8
    )
    return Image.fromarray(blended)


def make_attention_gif(
    image: Image.Image,
    tokens: list[str],
    heatmaps: list[np.ndarray],
    out_path: Path,
    fps: int = 3,
) -> Path:
    from PIL import ImageDraw as PILImageDraw

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))

    frames: list[Image.Image] = []
    running_text = ""
    w, h = image.size

    for tok, hmap in zip(tokens, heatmaps):
        running_text += tok
        blended = overlay_attention(image, hmap)

        frame = Image.new("RGB", (w, h + 36), (20, 20, 20))
        frame.paste(blended, (0, 0))
        draw = PILImageDraw.Draw(frame)
        draw.text((6, h + 8), running_text.strip(), fill=(255, 255, 255))
        frames.append(frame)

    if not frames:
        raise RuntimeError("No attention frames generated.")

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out_path


# -----------------------------------------------------------------------------
# Hugging Face VLM adapter
# -----------------------------------------------------------------------------


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class HuggingFaceVLM:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.spec = spec
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.local_files_only = local_files_only
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map: str = "auto" if torch.cuda.is_available() else "cpu"

    def _build_max_memory(self) -> dict[str, str] | None:
        """Leave VRAM headroom to reduce OOM from allocator/temporary buffers."""
        if not torch.cuda.is_available():
            return None

        max_memory: dict[str, str] = {}
        for i in range(torch.cuda.device_count()):
            total_bytes = torch.cuda.get_device_properties(i).total_memory
            # Keep ~10% free for runtime overhead and temporary tensors.
            usable_gib = max(int((total_bytes / (1024**3)) * 0.9), 1)
            max_memory[f"cuda:{i}"] = f"{usable_gib}GiB"
        # Allow CPU offload fallback if GPU-only placement cannot fit.
        max_memory["cpu"] = "256GiB"
        return max_memory

    def _load_processor(self) -> Any:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.spec.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)

        # InternVL processors require special token attributes on the tokenizer that
        # the fast TokenizersBackend doesn't expose. Patch them in before constructing
        # the processor via the slow-tokenizer path.
        if "internvl" in self.spec.model_id.lower():
            return self._load_internvl_processor(kwargs)

        return AutoProcessor.from_pretrained(self.spec.model_id, **kwargs)

    def _load_internvl_processor(self, kwargs: dict[str, Any]) -> Any:
        """Load InternVL processor, patching the tokenizer for missing image-token attrs."""
        from transformers import AutoTokenizer, AutoImageProcessor
        from transformers.models.internvl.processing_internvl import InternVLProcessor

        tok = AutoTokenizer.from_pretrained(self.spec.model_id, use_fast=False, **kwargs)

        # Map known InternVL special-token names → their string values
        for tok_str, attr_names in [
            ("<img>",         ["start_image_token"]),
            ("</img>",        ["end_image_token"]),
            ("<IMG_CONTEXT>", ["context_image_token"]),
            ("<video>",       ["video_token"]),
        ]:
            tok_id = tok.convert_tokens_to_ids(tok_str)
            if tok_id == tok.unk_token_id:
                tok.add_special_tokens({"additional_special_tokens": [tok_str]})
                tok_id = tok.convert_tokens_to_ids(tok_str)
            for attr in attr_names:
                if not hasattr(tok, attr):
                    setattr(tok, attr, tok_str)
                id_attr = attr.replace("_token", "_token_id")
                if not hasattr(tok, id_attr):
                    setattr(tok, id_attr, tok_id)

        img_proc = AutoImageProcessor.from_pretrained(self.spec.model_id, **kwargs)
        from transformers import AutoVideoProcessor
        vid_proc = AutoVideoProcessor.from_pretrained(self.spec.model_id, **kwargs)
        proc = InternVLProcessor(image_processor=img_proc, tokenizer=tok, video_processor=vid_proc)
        return proc

    def _candidate_loaders(self) -> list[Any]:
        loaders: list[Any] = [AutoModelForImageTextToText]
        if AutoModelForVision2Seq is not None:
            loaders.append(AutoModelForVision2Seq)
        loaders.append(AutoModelForCausalLM)
        # Last-resort: AutoModel (handles custom architectures like InternVLChatModel)
        from transformers import AutoModel
        loaders.append(AutoModel)
        return loaders

    def _load_model(self) -> Any:
        dtype = _resolve_torch_dtype(self.spec.dtype)
        base_kwargs: dict[str, Any] = {
            "trust_remote_code": self.spec.trust_remote_code,
            "local_files_only": self.local_files_only,
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "max_memory": self._build_max_memory(),
            "low_cpu_mem_usage": True,
        }
        if self.cache_dir is not None:
            base_kwargs["cache_dir"] = str(self.cache_dir)
        if self.spec.attn_implementation:
            base_kwargs["attn_implementation"] = self.spec.attn_implementation

        # InternVL uses remote-code classes that are missing transformers 5.x attrs.
        # Use a two-pass approach: first call registers the class in sys.modules (may fail),
        # then patch the class and retry.
        if "internvl" in self.spec.model_id.lower():
            return self._load_internvl_model(base_kwargs)

        errors: list[str] = []
        for loader in self._candidate_loaders():
            # try with full kwargs first
            try:
                model = loader.from_pretrained(self.spec.model_id, **base_kwargs)
                model.eval()
                return model
            except Exception as exc:
                errors.append(f"{loader.__name__} (full kwargs): {exc}")

            # fallback without attn_implementation
            try:
                kw = dict(base_kwargs)
                kw.pop("attn_implementation", None)
                model = loader.from_pretrained(self.spec.model_id, **kw)
                model.eval()
                return model
            except Exception as exc:
                errors.append(f"{loader.__name__} (fallback): {exc}")

        raise RuntimeError(
            "Unable to load model with supported loaders.\n" + "\n".join(errors)
        )

    def _load_internvl_model(self, base_kwargs: dict[str, Any]) -> Any:
        """InternVL remote-code model lacks transformers 5.x 'all_tied_weights_keys'.
        Strategy: first pass registers the class in sys.modules even if it fails;
        then patch the class attribute and retry."""
        import sys
        from transformers import AutoModel

        # Strip attn_implementation — InternVL's remote code doesn't accept it
        kw = {k: v for k, v in base_kwargs.items() if k != "attn_implementation"}

        # Pass 1: may fail on all_tied_weights_keys, but registers class in sys.modules
        try:
            model = AutoModel.from_pretrained(self.spec.model_id, **kw)
            model.eval()
            return model
        except (AttributeError, TypeError) as exc:
            if "all_tied_weights_keys" not in str(exc):
                raise RuntimeError(f"InternVL load failed unexpectedly: {exc}") from exc

        # Patch every InternVL-related class registered in sys.modules
        for _mod_name, _mod in list(sys.modules.items()):
            if "internvl" not in _mod_name.lower():
                continue
            for _attr in dir(_mod):
                try:
                    _cls = getattr(_mod, _attr, None)
                except Exception:
                    continue
                if not (isinstance(_cls, type) and hasattr(_cls, "__dict__")):
                    continue
                for _missing, _default in (
                    ("all_tied_weights_keys", {}),
                    ("_tied_weights_keys", []),
                ):
                    if not hasattr(_cls, _missing):
                        try:
                            setattr(_cls, _missing, _default)
                        except (TypeError, AttributeError):
                            pass

        # Pass 2: retry with patched classes
        model = AutoModel.from_pretrained(self.spec.model_id, **kw)
        model.eval()

        # Set required token IDs that InternVLChatModel.generate_flash needs
        if self.processor is not None:
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None:
                for tok_str, attr in (
                    ("<IMG_CONTEXT>", "img_context_token_id"),
                    ("<img>",         "img_start_token_id"),
                    ("</img>",        "img_end_token_id"),
                ):
                    if not hasattr(model, attr) or getattr(model, attr) is None:
                        tid = tok.convert_tokens_to_ids(tok_str)
                        setattr(model, attr, tid)

        return model

    def load(self) -> None:
        if self.processor is None:
            self.processor = self._load_processor()
        if self.model is None:
            self.model = self._load_model()

    def unload(self) -> None:
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Determine model dtype for floating-point casting
        model_dtype: torch.dtype | None = None
        if self.model is not None:
            try:
                model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                pass
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(target)
                # Cast float tensors to match the model dtype to avoid dtype mismatch errors
                if model_dtype is not None and v.is_floating_point() and v.dtype != model_dtype:
                    v = v.to(model_dtype)
                out[k] = v
            else:
                out[k] = v
        return out

    def _build_inputs(self, image: Image.Image, question: str) -> dict[str, Any]:
        assert self.processor is not None

        # InternVL requires an explicit <image> placeholder in the prompt text.
        if "internvl" in self.spec.model_id.lower():
            return self._build_internvl_inputs(image, question)

        # Most VLMs support chat template with multimodal content.
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch = self.processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                )
                return self._to_device(batch)
            except Exception:
                pass

        # Generic fallback
        batch = self.processor(
            text=[question],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        return self._to_device(batch)

    def _build_internvl_inputs(self, image: Image.Image, question: str) -> dict[str, Any]:
        """Build inputs for InternVL, which requires <image> placeholder in the prompt."""
        assert self.processor is not None
        img_token = getattr(self.processor, "image_token", "<image>")
        # Build prompt with image placeholder
        prompt = f"{img_token}\n{question}"
        if hasattr(self.processor, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch = self.processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                )
                return self._to_device(batch)
            except Exception:
                pass
        batch = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        )
        return self._to_device(batch)

    def _decode_answer_and_tokens(
        self,
        generated_ids: torch.Tensor,
    ) -> tuple[str, list[str]]:
        assert self.processor is not None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor tokenizer is missing; cannot decode tokens.")

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tokens = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in generated_ids.tolist()]
        return answer, tokens

    def _preprocess_internvl_image(
        self, image: Image.Image, input_size: int = 448, max_num: int = 12
    ) -> torch.Tensor:
        """Preprocess an image for InternVL using the official dynamic-tiling approach."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        img = image.convert("RGB")
        orig_w, orig_h = img.size
        aspect_ratio = orig_w / orig_h

        target_ratios = sorted(
            {
                (i, j)
                for n in range(1, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if 1 <= i * j <= max_num
            },
            key=lambda x: x[0] * x[1],
        )

        best_ratio = (1, 1)
        best_diff  = float("inf")
        area = orig_w * orig_h
        for ratio in target_ratios:
            target_ar = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - target_ar)
            if diff < best_diff or (
                diff == best_diff
                and area > 0.5 * input_size * input_size * ratio[0] * ratio[1]
            ):
                best_diff  = diff
                best_ratio = ratio

        target_w = input_size * best_ratio[0]
        target_h = input_size * best_ratio[1]
        blocks   = best_ratio[0] * best_ratio[1]

        resized = img.resize((target_w, target_h))
        tiles: list[Image.Image] = []
        for i in range(blocks):
            col = i % (target_w // input_size)
            row = i // (target_w // input_size)
            box = (
                col * input_size, row * input_size,
                (col + 1) * input_size, (row + 1) * input_size,
            )
            tiles.append(resized.crop(box))
        if len(tiles) != 1:
            tiles.append(img.resize((input_size, input_size)))  # thumbnail

        pixel_values = torch.stack([transform(t) for t in tiles])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_torch_dtype(self.spec.dtype)
        return pixel_values.to(device=device, dtype=dtype)

    def _predict_internvl(
        self,
        image: Image.Image,
        question: str,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """Use InternVL's native model.chat() API for correct inference."""
        assert self.model is not None and self.processor is not None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("InternVL processor has no tokenizer attribute.")
        pixel_values = self._preprocess_internvl_image(image)
        prompt = f"<image>\n{question}"
        generation_config = dict(
            max_new_tokens=self.spec.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        with torch.no_grad():
            response = self.model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response if isinstance(response, str) else str(response)

    def predict(
        self,
        image: Image.Image,
        question: str,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        assert self.model is not None and self.processor is not None

        # InternVL has a custom .chat() API that handles prompting correctly.
        if "internvl" in self.spec.model_id.lower():
            return self._predict_internvl(image, question, do_sample=do_sample, temperature=temperature)

        inputs = self._build_inputs(image, question)

        input_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.spec.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_dict_in_generate=True,
                output_attentions=False,
            )

        # Some models (e.g. InternVL) return only new tokens, not input+output.
        # Detect this by checking if the output is shorter than or equal to input_len.
        if out.sequences.shape[1] <= input_len:
            generated_ids = out.sequences[0]
        else:
            generated_ids = out.sequences[0, input_len:]
        answer, _ = self._decode_answer_and_tokens(generated_ids)
        return answer

    def predict_with_attention(self, image: Image.Image, question: str) -> dict[str, Any]:
        assert self.model is not None and self.processor is not None

        inputs = self._build_inputs(image, question)
        if "input_ids" not in inputs:
            raise RuntimeError("input_ids not found in processor output; cannot run attention test.")

        input_ids = inputs["input_ids"]
        input_len = int(input_ids.shape[1])

        with torch.no_grad():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=self.spec.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                output_attentions=False,
                return_dict_in_generate=True,
            )

        if gen_output.sequences.shape[1] <= input_len:
            generated_ids = gen_output.sequences[0]
        else:
            generated_ids = gen_output.sequences[0, input_len:]
        answer, tokens = self._decode_answer_and_tokens(generated_ids)

        full_ids = gen_output.sequences
        num_gen = int(full_ids.shape[1] - input_len)
        full_attention_mask = torch.ones_like(full_ids)

        fwd_kwargs: dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": full_attention_mask,
            "output_attentions": True,
            "return_dict": True,
        }

        # Pass multimodal tensors when available.
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            if key in inputs:
                fwd_kwargs[key] = inputs[key]

        # Qwen-style mrope support
        if "mm_token_type_ids" in inputs:
            orig_mm = inputs["mm_token_type_ids"]
            gen_mm = torch.zeros((1, num_gen), dtype=orig_mm.dtype, device=orig_mm.device)
            fwd_kwargs["mm_token_type_ids"] = torch.cat([orig_mm, gen_mm], dim=1)

        with torch.no_grad():
            fwd = self.model(**fwd_kwargs)

        raw_attns = None
        for attr in ("attentions", "decoder_attentions", "language_model_attentions"):
            raw_attns = getattr(fwd, attr, None)
            if raw_attns is not None:
                break

        if raw_attns is None:
            for key, value in fwd.items():
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    first = value[0]
                    if hasattr(first, "shape") and len(first.shape) == 4:
                        raw_attns = value
                        break

        if raw_attns is None:
            raise RuntimeError("No attention tensors found in forward outputs.")

        all_layer_attns = np.stack(
            [a.squeeze(0).float().cpu().numpy() for a in raw_attns], axis=0
        )

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor tokenizer is missing.")

        vis_start, vis_end = find_visual_token_range(input_ids, tokenizer)
        num_vis = vis_end - vis_start
        t_dim, grid_h, grid_w = infer_visual_grid(inputs, num_vis)

        heatmaps: list[np.ndarray] = []
        for i in range(len(generated_ids)):
            gen_pos = input_len + i
            hmap = raw_attention_heatmap(
                all_layer_attns=all_layer_attns,
                query_pos=gen_pos,
                vis_start=vis_start,
                vis_end=vis_end,
                grid_h=grid_h,
                grid_w=grid_w,
                layer=-1,
                head_reduction="mean",
            )
            heatmaps.append(hmap)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "answer": answer,
            "tokens": tokens,
            "heatmaps": heatmaps,
            "attention_shape": tuple(all_layer_attns.shape),
            "vis_range": (vis_start, vis_end),
            "grid": (t_dim, grid_h, grid_w),
        }


# -----------------------------------------------------------------------------
# Evaluation runners
# -----------------------------------------------------------------------------


def _sample_by_split(
    samples: list[EvalSample],
    max_samples_per_split: int | None,
    seed: int,
) -> list[EvalSample]:
    if max_samples_per_split is None:
        return samples

    grouped: dict[str, list[EvalSample]] = {}
    for s in samples:
        grouped.setdefault(s.split, []).append(s)

    rng = random.Random(seed)
    selected: list[EvalSample] = []
    for split in sorted(grouped.keys()):
        items = grouped[split]
        if len(items) <= max_samples_per_split:
            selected.extend(items)
        else:
            chosen = rng.sample(items, max_samples_per_split)
            selected.extend(chosen)
    return selected


def evaluate_model_on_val_data(
    model_name: str,
    val_root: str | Path,
    *,
    output_dir: str | Path = "outputs/vlm_eval",
    max_samples_per_split: int | None = 60,
    local_files_only: bool = True,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    run_attention_test: bool = True,
) -> dict[str, Any]:
    """Evaluate one model on val_data and run an attention visualization smoke test."""
    t0 = time.time()
    model_name = resolve_model_name(model_name)
    spec = MODEL_SPECS[model_name]

    val_root = Path(val_root).resolve()
    out_dir = Path(output_dir).resolve() / re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model_name": model_name,
        "model_id": spec.model_id,
        "status": "init",
        "local_files_only": local_files_only,
        "notes": spec.notes,
        "max_samples_per_split": max_samples_per_split,
        "val_root": str(val_root),
        "started_at": time.time(),
    }

    validation = validate_val_dataset(val_root)
    report["dataset_validation"] = validation

    samples = load_eval_samples(val_root)
    samples = _sample_by_split(samples, max_samples_per_split, seed)
    report["num_samples_selected"] = len(samples)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    adapter = HuggingFaceVLM(
        spec,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    try:
        adapter.load()
    except Exception as exc:
        report["status"] = "load_failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=2)

        report_path = out_dir / "report.json"
        report["elapsed_sec"] = time.time() - t0
        report_path.write_text(json.dumps(report, indent=2))
        return report

    attention_test: dict[str, Any] = {
        "status": "not_run",
        "reason": "",
        "gif_path": "",
        "attention_shape": None,
    }

    first_sample_for_attention = samples[0] if samples else None

    n_total = len(samples)
    for idx, sample in enumerate(samples):
        if idx % 10 == 0:
            print(f"  [{idx}/{n_total}] evaluating {sample.split}/{sample.task_type} ...", flush=True)
        row: dict[str, Any] = {
            "index": idx,
            "split": sample.split,
            "task_type": sample.task_type,
            "image_id": sample.image_id,
            "question": sample.question,
            "ground_truth": sample.answer,
        }

        try:
            image = Image.open(sample.image_path).convert("RGB")
            pred = adapter.predict(image, sample.question)
            correct = is_correct_prediction(pred, sample.answer, sample.task_type)
            row["prediction"] = pred
            row["correct"] = bool(correct)
            row["error"] = ""
        except Exception as exc:
            row["prediction"] = ""
            row["correct"] = False
            row["error"] = str(exc)
            errors.append(
                {
                    "image_id": sample.image_id,
                    "split": sample.split,
                    "task_type": sample.task_type,
                    "error": str(exc),
                }
            )

        rows.append(row)

    n_eval = len(rows)
    n_ok = sum(1 for r in rows if r.get("error") == "")
    n_correct = sum(1 for r in rows if bool(r.get("correct", False)))
    overall_acc = float(n_correct) / float(n_eval) if n_eval else 0.0

    report["status"] = "ok"
    report["num_evaluated"] = n_eval
    report["num_successful_inference"] = n_ok
    report["num_correct"] = n_correct
    report["accuracy"] = overall_acc
    report["accuracy_by_split"] = _aggregate_accuracy(rows, "split")
    report["accuracy_by_task"] = _aggregate_accuracy(rows, "task_type")
    report["num_errors"] = len(errors)

    # Attention smoke test.
    if run_attention_test and first_sample_for_attention is not None:
        try:
            image = Image.open(first_sample_for_attention.image_path).convert("RGB")
            attn = adapter.predict_with_attention(image, first_sample_for_attention.question)
            gif_path = out_dir / f"{first_sample_for_attention.image_id}_attention.gif"
            make_attention_gif(image, attn["tokens"], attn["heatmaps"], gif_path)

            attention_test = {
                "status": "passed",
                "reason": "",
                "gif_path": str(gif_path),
                "attention_shape": attn.get("attention_shape"),
                "num_tokens": len(attn.get("tokens", [])),
                "model_answer": attn.get("answer", ""),
                "question": first_sample_for_attention.question,
                "image_id": first_sample_for_attention.image_id,
            }
        except Exception as exc:
            attention_test = {
                "status": "failed",
                "reason": str(exc),
                "gif_path": "",
                "attention_shape": None,
            }

    report["attention_test"] = attention_test

    # Save detailed outputs.
    rows_path = out_dir / "predictions.jsonl"
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))

    report["predictions_path"] = str(rows_path)
    report["elapsed_sec"] = time.time() - t0

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    adapter.unload()
    return report


def evaluate_models_on_val_data(
    model_names: Iterable[str],
    val_root: str | Path,
    *,
    output_dir: str | Path = "outputs/vlm_eval",
    max_samples_per_split: int | None = 60,
    local_files_only: bool = True,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    run_attention_test: bool = True,
) -> dict[str, Any]:
    """Evaluate multiple models and return a summary table."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for name in model_names:
        report = evaluate_model_on_val_data(
            model_name=name,
            val_root=val_root,
            output_dir=output_dir,
            max_samples_per_split=max_samples_per_split,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            seed=seed,
            run_attention_test=run_attention_test,
        )
        reports.append(report)

    summary_rows: list[dict[str, Any]] = []
    for r in reports:
        summary_rows.append(
            {
                "model_name": r.get("model_name"),
                "model_id": r.get("model_id"),
                "status": r.get("status"),
                "accuracy": r.get("accuracy"),
                "num_evaluated": r.get("num_evaluated"),
                "num_successful_inference": r.get("num_successful_inference"),
                "attention_status": (r.get("attention_test") or {}).get("status"),
                "attention_gif": (r.get("attention_test") or {}).get("gif_path"),
                "error": r.get("error", ""),
            }
        )

    summary = {
        "val_root": str(Path(val_root).resolve()),
        "num_models": len(summary_rows),
        "models": summary_rows,
        "reports": reports,
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    summary["summary_json"] = str(summary_json)
    summary["summary_csv"] = str(summary_csv)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FineSightBench VLM evaluation framework")
    parser.add_argument(
        "--val-root",
        type=str,
        default="data/val_data",
        help="Path to val_data root",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list_supported_models(),
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vlm_eval",
        help="Directory for reports and predictions",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=60,
        help="Cap evaluated samples per split; use -1 for all",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading missing model files from Hugging Face.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-attention-test",
        action="store_true",
        help="Skip attention visualization smoke tests.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional HF cache dir override",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    max_samples = None if args.max_samples_per_split < 0 else args.max_samples_per_split
    summary = evaluate_models_on_val_data(
        model_names=args.models,
        val_root=args.val_root,
        output_dir=args.output_dir,
        max_samples_per_split=max_samples,
        local_files_only=not args.allow_download,
        cache_dir=args.cache_dir or None,
        seed=args.seed,
        run_attention_test=not args.no_attention_test,
    )

    print(json.dumps(summary["models"], indent=2, ensure_ascii=False))
    print(f"summary json: {summary['summary_json']}")
    print(f"summary csv : {summary['summary_csv']}")


if __name__ == "__main__":
    main()
