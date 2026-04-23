"""JSON-based evaluation metrics for VLM outputs.

Design goals (per requirements):

1. VLM outputs are expected to be JSON. Compare **per-field**, never
   compare the whole JSON blob as a single string. Each field has its own
   score and ``matched`` flag.
2. If the VLM output fails to parse as JSON, record it as a hallucination
   so callers can compute hallucination counts and rates.
3. For ordering tasks (``ordered_list`` fields), record positional
   accuracy — fraction of elements placed correctly — instead of
   all-or-nothing matching.

Public API
----------
- ``extract_json(text)``              → ``(obj | None, error_reason)``
- ``FieldSpec``                       → schema element
- ``evaluate_json_prediction(...)``   → per-sample ``JSONEvalResult``
- ``aggregate_json_results(...)``     → aggregate dict with accuracy,
  hallucination rate, per-field / per-task breakdowns.
- ``BUILTIN_SCHEMAS``                 → default schemas for FineSightBench
  task types (optional convenience).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _find_balanced_json(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` substring, or None."""
    opens = {"{": "}", "[": "]"}
    start = -1
    closer = ""
    for i, ch in enumerate(text):
        if ch in opens:
            start = i
            closer = opens[ch]
            break
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == text[start]:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def extract_json(text: str) -> tuple[Any | None, str]:
    """Best-effort JSON extraction from free-form VLM output.

    Returns ``(obj, "")`` on success, or ``(None, reason)`` on failure.
    """
    if text is None:
        return None, "empty_output"
    raw = text.strip()
    if not raw:
        return None, "empty_output"

    candidate = _strip_code_fences(raw)

    # Try direct parse first.
    try:
        return json.loads(candidate), ""
    except Exception:
        pass

    # Try to locate the first balanced JSON object/array.
    block = _find_balanced_json(candidate)
    if block is None:
        return None, "no_json_block_found"
    try:
        return json.loads(block), ""
    except Exception as exc:
        # Attempt a tolerant retry: replace single quotes, trailing commas.
        fixed = block.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed), ""
        except Exception:
            return None, f"json_decode_error: {exc}"


# ---------------------------------------------------------------------------
# Field schema & comparison
# ---------------------------------------------------------------------------


SCALAR = "scalar"
ORDERED_LIST = "ordered_list"
UNORDERED_SET = "unordered_set"
MAPPING = "mapping"  # dict[str, scalar]; compared key-by-key

_VALID_KINDS = {SCALAR, ORDERED_LIST, UNORDERED_SET, MAPPING}


@dataclass(frozen=True)
class FieldSpec:
    """Description of a single field inside the expected JSON object."""

    name: str
    kind: str = SCALAR
    path: tuple[str, ...] | None = None  # dotted path override (default = (name,))
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unknown FieldSpec kind: {self.kind!r}")

    @property
    def resolved_path(self) -> tuple[str, ...]:
        return self.path if self.path is not None else (self.name,)


def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return _MISSING
    return cur


_MISSING = object()


def _norm_scalar(v: Any, case_sensitive: bool) -> Any:
    if isinstance(v, str):
        s = v.strip()
        return s if case_sensitive else s.lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return v


def _to_list_of_scalars(v: Any, case_sensitive: bool) -> list[Any] | None:
    if v is None or v is _MISSING:
        return None
    if isinstance(v, list):
        return [_norm_scalar(x, case_sensitive) for x in v]
    if isinstance(v, str):
        # tolerate comma-separated string when a list was expected
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return [_norm_scalar(p, case_sensitive) for p in parts]
    return None


def _to_mapping(v: Any, case_sensitive: bool) -> dict[str, Any] | None:
    if not isinstance(v, dict):
        return None
    out: dict[str, Any] = {}
    for k, val in v.items():
        key = k if case_sensitive else str(k).strip().lower()
        out[key] = _norm_scalar(val, case_sensitive)
    return out


def _compare_scalar(pred: Any, gt: Any, spec: FieldSpec) -> dict[str, Any]:
    if pred is _MISSING:
        return {
            "kind": SCALAR,
            "expected": gt,
            "predicted": None,
            "missing": True,
            "matched": False,
            "score": 0.0,
        }
    p = _norm_scalar(pred, spec.case_sensitive)
    g = _norm_scalar(gt, spec.case_sensitive)
    ok = p == g
    return {
        "kind": SCALAR,
        "expected": gt,
        "predicted": pred,
        "missing": False,
        "matched": bool(ok),
        "score": 1.0 if ok else 0.0,
    }


def _compare_ordered_list(pred: Any, gt: Any, spec: FieldSpec) -> dict[str, Any]:
    gt_list = _to_list_of_scalars(gt, spec.case_sensitive) or []
    if pred is _MISSING:
        return {
            "kind": ORDERED_LIST,
            "expected": gt,
            "predicted": None,
            "missing": True,
            "matched": False,
            "score": 0.0,
            "positional_correct": 0,
            "expected_len": len(gt_list),
            "predicted_len": 0,
            "length_match": False,
        }
    pred_list = _to_list_of_scalars(pred, spec.case_sensitive)
    if pred_list is None:
        return {
            "kind": ORDERED_LIST,
            "expected": gt,
            "predicted": pred,
            "missing": False,
            "matched": False,
            "score": 0.0,
            "positional_correct": 0,
            "expected_len": len(gt_list),
            "predicted_len": 0,
            "length_match": False,
            "note": "predicted_value_not_a_list",
        }

    n_gt = len(gt_list)
    n_pred = len(pred_list)
    overlap = min(n_gt, n_pred)
    correct = sum(1 for i in range(overlap) if pred_list[i] == gt_list[i])
    denom = max(n_gt, n_pred) or 1
    score = correct / denom  # penalises both missing and extra items
    return {
        "kind": ORDERED_LIST,
        "expected": gt_list,
        "predicted": pred_list,
        "missing": False,
        "matched": bool(correct == n_gt and n_gt == n_pred),
        "score": score,
        "positional_correct": correct,
        "expected_len": n_gt,
        "predicted_len": n_pred,
        "length_match": n_gt == n_pred,
    }


def _compare_unordered_set(pred: Any, gt: Any, spec: FieldSpec) -> dict[str, Any]:
    gt_list = _to_list_of_scalars(gt, spec.case_sensitive) or []
    gt_set = set(gt_list)
    if pred is _MISSING:
        return {
            "kind": UNORDERED_SET,
            "expected": sorted(gt_set, key=str),
            "predicted": None,
            "missing": True,
            "matched": False,
            "score": 0.0,
        }
    pred_list = _to_list_of_scalars(pred, spec.case_sensitive)
    if pred_list is None:
        return {
            "kind": UNORDERED_SET,
            "expected": sorted(gt_set, key=str),
            "predicted": pred,
            "missing": False,
            "matched": False,
            "score": 0.0,
            "note": "predicted_value_not_a_list",
        }
    pred_set = set(pred_list)
    inter = gt_set & pred_set
    union = gt_set | pred_set
    jaccard = (len(inter) / len(union)) if union else 1.0
    return {
        "kind": UNORDERED_SET,
        "expected": sorted(gt_set, key=str),
        "predicted": sorted(pred_set, key=str),
        "missing": False,
        "matched": gt_set == pred_set,
        "score": jaccard,
        "intersection": len(inter),
        "expected_len": len(gt_set),
        "predicted_len": len(pred_set),
    }


def _compare_mapping(pred: Any, gt: Any, spec: FieldSpec) -> dict[str, Any]:
    gt_map = _to_mapping(gt, spec.case_sensitive) or {}
    if pred is _MISSING:
        return {
            "kind": MAPPING,
            "expected": gt_map,
            "predicted": None,
            "missing": True,
            "matched": False,
            "score": 0.0,
            "keys_correct": 0,
            "keys_total": len(gt_map),
        }
    pred_map = _to_mapping(pred, spec.case_sensitive)
    if pred_map is None:
        return {
            "kind": MAPPING,
            "expected": gt_map,
            "predicted": pred,
            "missing": False,
            "matched": False,
            "score": 0.0,
            "keys_correct": 0,
            "keys_total": len(gt_map),
            "note": "predicted_value_not_a_dict",
        }
    keys_correct = 0
    per_key: dict[str, bool] = {}
    for k, v in gt_map.items():
        ok = k in pred_map and pred_map[k] == v
        per_key[k] = bool(ok)
        if ok:
            keys_correct += 1
    score = keys_correct / len(gt_map) if gt_map else 1.0
    return {
        "kind": MAPPING,
        "expected": gt_map,
        "predicted": pred_map,
        "missing": False,
        "matched": gt_map == pred_map,
        "score": score,
        "keys_correct": keys_correct,
        "keys_total": len(gt_map),
        "per_key_correct": per_key,
    }


_COMPARATORS = {
    SCALAR: _compare_scalar,
    ORDERED_LIST: _compare_ordered_list,
    UNORDERED_SET: _compare_unordered_set,
    MAPPING: _compare_mapping,
}


# ---------------------------------------------------------------------------
# Fallback value extraction from raw text (for JSON parse failures)
# ---------------------------------------------------------------------------


def _extract_list_items_from_text(text: str) -> list[str] | None:
    """Try to extract list items from raw VLM output when JSON parsing fails.
    
    Attempts to find patterns like:
    - Quoted items: "item1", "item2"
    - Bullet points: • item1, • item2  or - item1, - item2
    - Numbered: 1. item1, 2. item2
    - Parenthesized: (item1), (item2)
    - Tags followed by text: <tag> item1, <tag> item2
    - Simple comma/semicolon-separated (fallback)
    
    Returns list of extracted strings (normalized lowercase) or None if not enough items.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Try to find quoted items first (most reliable)
    quoted = re.findall(r'["\']([^"\']{2,})["\']', text)
    if len(quoted) >= 1:
        return [_norm_scalar(q, False) for q in quoted]
    
    # Try parenthesized items: (item)
    parens = re.findall(r'\(([^)]{2,})\)', text)
    if len(parens) >= 1:
        return [_norm_scalar(p, False) for p in parens]
    
    # Try bullet/dash/number prefix: • item, - item, 1. item, etc.
    bullets = re.findall(r'(?:^|\n)\s*(?:•|-|·|\*|\d+\.)\s+([^\n]+?)(?:\n|$|,|;)', 
                        text + '\n', re.MULTILINE)
    if len(bullets) >= 1:
        return [_norm_scalar(b.strip(), False) for b in bullets if b.strip()]
    
    # Try tag-prefix format: <tag> item, <tag> item, ...
    # This handles patterns like: <color> Purple triangle, <color> Blue pentagon
    tag_pattern = re.findall(r'<[^>]+>\s+([^,<\n]{2,})', text)
    if len(tag_pattern) >= 1:
        items = [_norm_scalar(item, False) for item in tag_pattern if item.strip()]
        if len(items) >= 1:
            return items
    
    # Fallback: try comma or semicolon split, but only if text looks like a list
    # (multiple separators or short segments)
    if ',' in text or ';' in text:
        # Remove tag markers first
        cleaned = re.sub(r'<[^>]+>\s*', '', text)
        parts = re.split(r'[,;\n]', cleaned)
        items = [_norm_scalar(p.strip(), False) for p in parts 
                if p.strip() and len(p.strip()) > 1]
        if len(items) >= 1:
            return items
    
    return None


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


@dataclass
class JSONEvalResult:
    parse_ok: bool
    parse_error: str
    raw_pred: str
    pred_obj: Any
    field_results: dict[str, dict[str, Any]]
    overall_score: float        # mean of field scores; 0.0 on parse failure
    all_fields_matched: bool    # strict: every field matched exactly
    hallucination: bool         # True iff JSON failed to parse

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "raw_pred": self.raw_pred,
            "pred_obj": self.pred_obj,
            "field_results": self.field_results,
            "overall_score": self.overall_score,
            "all_fields_matched": self.all_fields_matched,
            "hallucination": self.hallucination,
        }


def evaluate_json_prediction(
    pred_text: str,
    gt_obj: dict[str, Any],
    schema: Iterable[FieldSpec],
) -> JSONEvalResult:
    """Evaluate a single VLM JSON output against a ground-truth dict.

    Parameters
    ----------
    pred_text:
        Raw VLM output (expected to contain JSON).
    gt_obj:
        Ground-truth object with the fields named in ``schema``.
    schema:
        Iterable of ``FieldSpec`` describing which fields to compare and how.

    Returns
    -------
    JSONEvalResult
    """
    schema_list = list(schema)
    pred_obj, err = extract_json(pred_text)

    if pred_obj is None:
        # JSON parse failed, but try fallback value matching for ORDERED_LIST fields
        field_results: dict[str, dict[str, Any]] = {}
        scores: list[float] = []
        all_matched = True
        
        for spec in schema_list:
            gt_val = _get_path(gt_obj, spec.resolved_path)
            
            # For ORDERED_LIST, try to extract values from raw text
            if spec.kind == ORDERED_LIST:
                extracted_items = _extract_list_items_from_text(pred_text)
                if extracted_items is not None:
                    # We have extracted values, compare them
                    res = _COMPARATORS[spec.kind](extracted_items, gt_val, spec)
                    res["extraction_fallback"] = True
                    field_results[spec.name] = res
                    scores.append(float(res["score"]))
                    if not res["matched"]:
                        all_matched = False
                    continue
            
            # Default: mark as missing
            res = _COMPARATORS[spec.kind](_MISSING, gt_val, spec)
            res["missing"] = True
            field_results[spec.name] = res
            scores.append(0.0)
            all_matched = False
        
        overall = sum(scores) / len(scores) if scores else 0.0
        
        # Only mark as hallucination if NO fields were successfully extracted
        has_any_success = any(
            r.get("extraction_fallback") or (not r.get("missing", True))
            for r in field_results.values()
        )
        
        return JSONEvalResult(
            parse_ok=False,
            parse_error=err or "unknown_parse_error",
            raw_pred=pred_text if pred_text is not None else "",
            pred_obj=None,
            field_results=field_results,
            overall_score=overall,
            all_fields_matched=all_matched,
            hallucination=not has_any_success,  # True if completely failed
        )

    field_results = {}
    scores: list[float] = []
    all_matched = True
    for spec in schema_list:
        gt_val = _get_path(gt_obj, spec.resolved_path)
        pred_val = _get_path(pred_obj, spec.resolved_path) if isinstance(pred_obj, dict) else _MISSING
        res = _COMPARATORS[spec.kind](pred_val, gt_val, spec)
        field_results[spec.name] = res
        scores.append(float(res["score"]))
        if not res["matched"]:
            all_matched = False

    overall = sum(scores) / len(scores) if scores else 1.0
    return JSONEvalResult(
        parse_ok=True,
        parse_error="",
        raw_pred=pred_text if pred_text is not None else "",
        pred_obj=pred_obj,
        field_results=field_results,
        overall_score=overall,
        all_fields_matched=all_matched,
        hallucination=False,
    )


# ---------------------------------------------------------------------------
# Aggregation across samples
# ---------------------------------------------------------------------------


def aggregate_json_results(
    rows: list[dict[str, Any]],
    *,
    result_key: str = "json_eval",
    group_keys: Iterable[str] = ("task_type",),
) -> dict[str, Any]:
    """Aggregate per-sample JSON eval results into a summary report.

    Each row in ``rows`` must contain ``row[result_key]`` which is either a
    ``JSONEvalResult`` or the dict produced by ``JSONEvalResult.to_dict()``.
    ``group_keys`` is a list of row keys used to build grouped breakdowns
    (e.g. ``("task_type",)`` or ``("split", "task_type")``).
    """
    group_keys = tuple(group_keys)

    def _get_result(row: dict[str, Any]) -> dict[str, Any]:
        r = row[result_key]
        if isinstance(r, JSONEvalResult):
            return r.to_dict()
        return r

    n = len(rows)
    hallucinations = 0
    all_matched = 0
    score_sum = 0.0

    # per-field
    field_hits: dict[str, int] = {}
    field_total: dict[str, int] = {}
    field_score_sum: dict[str, float] = {}
    field_positional: dict[str, dict[str, int]] = {}  # ordered_list only

    # per-group
    group_buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    for row in rows:
        res = _get_result(row)
        if res["hallucination"]:
            hallucinations += 1
        if res["all_fields_matched"]:
            all_matched += 1
        score_sum += float(res["overall_score"])

        for fname, fres in res["field_results"].items():
            field_total[fname] = field_total.get(fname, 0) + 1
            if fres.get("matched"):
                field_hits[fname] = field_hits.get(fname, 0) + 1
            field_score_sum[fname] = field_score_sum.get(fname, 0.0) + float(fres.get("score", 0.0))
            if fres.get("kind") == ORDERED_LIST:
                bucket = field_positional.setdefault(
                    fname, {"positional_correct": 0, "expected_total": 0, "predicted_total": 0}
                )
                bucket["positional_correct"] += int(fres.get("positional_correct", 0))
                bucket["expected_total"] += int(fres.get("expected_len", 0))
                bucket["predicted_total"] += int(fres.get("predicted_len", 0))

        key = tuple(str(row.get(k, "unknown")) for k in group_keys)
        group_buckets.setdefault(key, []).append(res)

    # Build group summaries recursively (only one level implemented here).
    groups_summary: dict[str, Any] = {}
    for key, bucket in sorted(group_buckets.items()):
        gn = len(bucket)
        g_hall = sum(1 for r in bucket if r["hallucination"])
        g_all = sum(1 for r in bucket if r["all_fields_matched"])
        g_score = sum(float(r["overall_score"]) for r in bucket) / gn if gn else 0.0
        label = " | ".join(f"{k}={v}" for k, v in zip(group_keys, key))
        groups_summary[label] = {
            "n": gn,
            "hallucinations": g_hall,
            "hallucination_rate": g_hall / gn if gn else 0.0,
            "all_fields_matched": g_all,
            "strict_accuracy": g_all / gn if gn else 0.0,
            "mean_overall_score": g_score,
        }

    per_field = {}
    for fname in sorted(field_total):
        tot = field_total[fname]
        per_field[fname] = {
            "n": tot,
            "correct": field_hits.get(fname, 0),
            "accuracy": field_hits.get(fname, 0) / tot if tot else 0.0,
            "mean_score": field_score_sum.get(fname, 0.0) / tot if tot else 0.0,
        }
        if fname in field_positional:
            pb = field_positional[fname]
            denom = max(pb["expected_total"], pb["predicted_total"])
            per_field[fname]["positional_accuracy"] = (
                pb["positional_correct"] / denom if denom else 0.0
            )
            per_field[fname]["positional_correct"] = pb["positional_correct"]
            per_field[fname]["positional_expected_total"] = pb["expected_total"]
            per_field[fname]["positional_predicted_total"] = pb["predicted_total"]

    return {
        "n": n,
        "hallucinations": hallucinations,
        "hallucination_rate": hallucinations / n if n else 0.0,
        "strict_accuracy": all_matched / n if n else 0.0,  # all fields matched
        "mean_overall_score": score_sum / n if n else 0.0,
        "per_field": per_field,
        "groups": groups_summary,
        "group_keys": list(group_keys),
    }


# ---------------------------------------------------------------------------
# Built-in schemas for FineSightBench task types (optional convenience)
# ---------------------------------------------------------------------------


BUILTIN_SCHEMAS: dict[str, list[FieldSpec]] = {
    # Perception single-field tasks
    "letter_recognition":      [FieldSpec("letter")],
    "animal_recognition":      [FieldSpec("animal")],
    "block_recognition":       [FieldSpec("present")],   # yes/no
    "color_block_recognition": [FieldSpec("color")],
    "shape_recognition":       [FieldSpec("shape")],

    # Reasoning — ordered chains
    "chain_reasoning":         [FieldSpec("objects", kind=ORDERED_LIST)],
    "comparison_chain":        [FieldSpec("objects", kind=ORDERED_LIST)],

    # Reasoning — counts + total
    "counting_chain": [
        FieldSpec("counts", kind=MAPPING),
        FieldSpec("total"),
    ],
    "blur_chain": [
        FieldSpec("counts", kind=MAPPING),
        FieldSpec("total"),
    ],
}


__all__ = [
    "SCALAR",
    "ORDERED_LIST",
    "UNORDERED_SET",
    "MAPPING",
    "FieldSpec",
    "JSONEvalResult",
    "extract_json",
    "evaluate_json_prediction",
    "aggregate_json_results",
    "BUILTIN_SCHEMAS",
]
