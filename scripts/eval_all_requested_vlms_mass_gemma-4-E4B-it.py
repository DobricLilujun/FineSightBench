import json
import logging
import random
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

# Suppress noisy but harmless warnings from PyTorch and transformers.
warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*", category=UserWarning)

# "Setting `pad_token_id` to `eos_token_id`" comes from the transformers logger,
# not warnings — silence it at the logging level.
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


from datasets import load_dataset
from tqdm.auto import tqdm

from finesightbench.evaluation.framework import (
    HuggingFaceVLM,
    MODEL_SPECS,
    list_supported_models,
    resolve_model_name,
)

# --- Config -------------------------------------------------------------------
HF_DATASET_IDS = [
    "Volavion/FineSightBench",
]
SPLITS = ["perception", "reasoning"]
# MODELS = [
#     "Qwen3-VL-2B-Instruct",
#     "InternVL3_5-1B-Flash",
#     "GLM-4.6V-Flash",
#     "gemma-4-E2B-it",
#     "Llama-4-Scout-17B-16E-Instruct",
# ]

MODELS = [
    # "Qwen3-VL-2B-Instruct",
    # "InternVL3_5-1B-Flash",
    # "GLM-4.6V-Flash",
    # "gemma-4-E2B-it",
    # "Llama-4-Scout-17B-16E-Instruct",
    "google/gemma-4-E4B-it"
]

# Decoding configurations to run for EVERY model.
# Each produces its own JSONL: <model>__<cfg_name>.jsonl
DECODING_CONFIGS = [
    {"name": "greedy",    "do_sample": False, "temperature": 1.0},
    {"name": "sample_t01", "do_sample": True,  "temperature": 0.1},
    {"name": "sample_t10", "do_sample": True,  "temperature": 1.0},
]

SAMPLES_PER_TASK = 2
SEED = 42
ALLOW_DOWNLOAD = True            # set False to force local cache only
OUTPUT_DIR = Path("outputs/vlm_eval_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def output_path_for(model_name: str, cfg_name: str) -> Path:
    """One JSONL per (model, decoding-config) so runs are independent and resumable."""
    return OUTPUT_DIR / f"{_safe_filename(model_name)}__{_safe_filename(cfg_name)}.jsonl"


print("Supported models :", list_supported_models())
print("Output dir       :", OUTPUT_DIR)



# --- Sample 2 items per (dataset_id, split, task_type) ------------------------
rng = random.Random(SEED)
selected: list[dict] = []   # list of small dicts carrying PIL image + metadata

for ds_id in HF_DATASET_IDS:
    ds = load_dataset(ds_id)
    for split in SPLITS:
        if split not in ds:
            print(f"  [skip] split '{split}' not in {ds_id}")
            continue
        by_task: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(ds[split]):
            by_task[row["task_type"]].append(idx)
        for task, idxs in by_task.items():
            rng.shuffle(idxs)
            SAMPLES_PER_TASK = len(idxs)
            for idx in idxs[:SAMPLES_PER_TASK]:
                row = ds[split][idx]
                selected.append({
                    "dataset_id": ds_id,
                    "split": split,
                    "row_index": idx,
                    "image_id": row.get("image_id"),
                    "task_type": row.get("task_type"),
                    "difficulty": row.get("difficulty"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "image": row["image"],       # PIL.Image
                })

print(f"\nSelected {len(selected)} samples in total")
_counts: dict[tuple, int] = defaultdict(int)
for s in selected:
    _counts[(s["dataset_id"], s["split"], s["task_type"])] += 1
for (d, sp, t), n in sorted(_counts.items()):
    print(f"  {d:28s} {sp:10s} {t:30s} {n}")


# --- Generate per-model; one JSONL per model; resumable ----------------------
# Each record is flushed immediately (one line at a time). On re-run we skip any
# (dataset_id, split, row_index) already present in the model's JSONL, so an
# interrupted run can be continued simply by re-executing this cell.

def _load_done_keys(path: Path, current_q_map: dict[tuple, str]) -> set[tuple]:
    """Read existing JSONL and return the set of (dataset_id, split, row_index)
    for rows that completed successfully (error is null) AND whose stored
    question matches the current dataset. Stale rows (different question text
    or previously-failed) are dropped and the file is rewritten in place so
    they will be regenerated on this run."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    kept: list[dict] = []
    stale = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("dataset_id"), rec.get("split"), rec.get("row_index"))
            if rec.get("error") is not None:
                stale += 1
                continue  # retry previously failed items
            cur_q = current_q_map.get(key)
            if cur_q is not None and rec.get("question") != cur_q:
                stale += 1
                continue  # question text changed upstream -> regenerate
            kept.append(rec)
            done.add(key)
    if stale:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f_out:
            for rec in kept:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
        print(f"  [resume] dropped {stale} stale/failed rows from {path.name}")
    return done


failures: list[dict] = []
totals: dict[str, dict[str, int]] = {}

# Map current (dataset_id, split, row_index) -> question text so we can detect
# stale JSONL rows whose question was regenerated upstream.
current_q_map: dict[tuple, str] = {
    (s["dataset_id"], s["split"], s["row_index"]): (s["question"] or "")
    for s in selected
}

for model_name in MODELS:
    try:
        resolved = resolve_model_name(model_name)
    except KeyError as exc:
        print(f"[skip] {model_name}: {exc}")
        failures.append({"model": model_name, "stage": "resolve", "error": str(exc)})
        continue

    spec = MODEL_SPECS[resolved]

    # Figure out which cfgs still have work to do before loading the model.
    cfg_todo: list[tuple[dict, Path, set[tuple], list[dict]]] = []
    for cfg in DECODING_CONFIGS:
        out_path = output_path_for(resolved, cfg["name"])
        done_keys = _load_done_keys(out_path, current_q_map)
        todo = [s for s in selected
                if (s["dataset_id"], s["split"], s["row_index"]) not in done_keys]
        print(f"\n=== {resolved} [{cfg['name']}]  ({spec.model_id}) ===")
        print(f"  file    : {out_path}")
        print(f"  done    : {len(done_keys)} / {len(selected)}")
        print(f"  todo    : {len(todo)}")
        if todo:
            cfg_todo.append((cfg, out_path, done_keys, todo))
        else:
            totals[f"{resolved}/{cfg['name']}"] = {"new": 0, "skipped": len(done_keys)}

    if not cfg_todo:
        continue

    # Load the model once, reuse across all cfgs.
    runner = HuggingFaceVLM(spec, local_files_only=not ALLOW_DOWNLOAD)
    try:
        t0 = time.time()
        runner.load()
        print(f"  loaded {resolved} in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"  [load-failed] {exc}")
        failures.append({
            "model": resolved,
            "stage": "load",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        })
        runner.unload()
        continue

    try:
        for cfg, out_path, done_keys, todo in cfg_todo:
            new_count = 0
            desc = f"{resolved}[{cfg['name']}]"
            # Append mode → resumable. Each record is written + flushed individually.
            with out_path.open("a", encoding="utf-8") as f_out:
                pbar = tqdm(todo, desc=desc, unit="sample", dynamic_ncols=True)
                for sample in pbar:
                    image = sample["image"].convert("RGB")
                    question = sample["question"] or ""
                    t0 = time.time()
                    try:
                        generated = runner.predict(
                            image, question,
                            do_sample=cfg["do_sample"],
                            temperature=cfg["temperature"],
                        )
                        err = None
                    except Exception as exc:
                        generated = ""
                        err = f"{type(exc).__name__}: {exc}"
                    latency = time.time() - t0

                    record = {
                        "model": resolved,
                        "model_id": spec.model_id,
                        "decoding": cfg["name"],
                        "do_sample": cfg["do_sample"],
                        "temperature": cfg["temperature"],
                        "dataset_id": sample["dataset_id"],
                        "split": sample["split"],
                        "row_index": sample["row_index"],
                        "image_id": sample["image_id"],
                        "task_type": sample["task_type"],
                        "difficulty": sample["difficulty"],
                        "question": sample["question"],
                        "reference_answer": sample["answer"],
                        "generated_text": generated,
                        "latency_s": round(latency, 3),
                        "error": err,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    new_count += 1
                    pbar.set_postfix(lat=f"{latency:.2f}s",
                                     err="yes" if err else "no",
                                     refresh=False)

            totals[f"{resolved}/{cfg['name']}"] = {
                "new": new_count, "skipped": len(done_keys),
            }
            print(f"  wrote {new_count} new rows → {out_path}")
    finally:
        runner.unload()

print("\n=== Summary ===")
for m, c in totals.items():
    print(f"  {m:50s} new={c['new']:4d}  resumed-skip={c['skipped']:4d}")
if failures:
    print(f"\nFailures: {len(failures)}")
    for f in failures:
        print(" -", f["model"], f["stage"], f["error"])