"""Evaluate Gemini 2.5 Flash Preview on FineSightBench via the Google Generative AI API.

Prerequisites
-------------
1. Install the Google Generative AI SDK::

       pip install google-generativeai

2. Set the ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) environment variable
   (or add it to a ``.env`` file in the project root)::

       export GOOGLE_API_KEY=AIza...

The script is resumable: on re-run it skips rows already present in the output
JSONL (unless they previously errored).
"""

import json
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from finesightbench.evaluation.framework import (
    API_MODEL_SPECS,
    GoogleVLM,
)

# --- Config -------------------------------------------------------------------
HF_DATASET_IDS = [
    "Volavion/FineSightBench",
]
SPLITS = ["perception", "reasoning"]

MODEL_NAME = "gemini-2.5-flash-preview"

# Decoding configurations.  temperature=0.0 is used for greedy inference.
DECODING_CONFIGS = [
    {"name": "greedy",     "do_sample": False, "temperature": 1.0},
    {"name": "sample_t01", "do_sample": True,  "temperature": 0.1},
    {"name": "sample_t10", "do_sample": True,  "temperature": 1.0},
]

SEED = 42
OUTPUT_DIR = Path("outputs/vlm_eval_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Delay between API requests (seconds) to stay within rate limits.
REQUEST_DELAY = 0.5


# --- API key loading ----------------------------------------------------------

def _parse_simple_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values[key] = value
    return values


def load_google_key_from_env() -> bool:
    token = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if token:
        # Normalise: always expose under both names.
        os.environ.setdefault("GOOGLE_API_KEY", token)
        os.environ.setdefault("GEMINI_API_KEY", token)
        print("Google API key source : environment")
        return True
    env_candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        env_values = _parse_simple_dotenv(env_path)
        token = env_values.get("GOOGLE_API_KEY") or env_values.get("GEMINI_API_KEY")
        if token:
            os.environ["GOOGLE_API_KEY"] = token
            os.environ["GEMINI_API_KEY"] = token
            print(f"Google API key source : {env_path}")
            return True
    print("Google API key source : not found — set GOOGLE_API_KEY before running.")
    return False


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def output_path_for(model_name: str, cfg_name: str) -> Path:
    return OUTPUT_DIR / f"{_safe_filename(model_name)}__{_safe_filename(cfg_name)}.jsonl"


load_google_key_from_env()
print("Model            :", MODEL_NAME)
print("Output dir       :", OUTPUT_DIR)


# --- Sample items per (dataset_id, split, task_type) --------------------------
rng = random.Random(SEED)
selected: list[dict] = []

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
            for idx in idxs:  # use all samples
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
                    "image": row["image"],
                })

print(f"\nSelected {len(selected)} samples in total")
_counts: dict[tuple, int] = defaultdict(int)
for s in selected:
    _counts[(s["dataset_id"], s["split"], s["task_type"])] += 1
for (d, sp, t), n in sorted(_counts.items()):
    print(f"  {d:28s} {sp:10s} {t:30s} {n}")


# --- Resumable JSONL helpers --------------------------------------------------

def _load_done_keys(path: Path, current_q_map: dict[tuple, str]) -> set[tuple]:
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
                continue
            cur_q = current_q_map.get(key)
            if cur_q is not None and rec.get("question") != cur_q:
                stale += 1
                continue
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


current_q_map: dict[tuple, str] = {
    (s["dataset_id"], s["split"], s["row_index"]): (s["question"] or "")
    for s in selected
}

# --- Generate -----------------------------------------------------------------
spec = API_MODEL_SPECS[MODEL_NAME]
failures: list[dict] = []
totals: dict[str, dict[str, int]] = {}

cfg_todo = []
for cfg in DECODING_CONFIGS:
    out_path = output_path_for(MODEL_NAME, cfg["name"])
    done_keys = _load_done_keys(out_path, current_q_map)
    todo = [s for s in selected
            if (s["dataset_id"], s["split"], s["row_index"]) not in done_keys]
    print(f"\n=== {MODEL_NAME} [{cfg['name']}] (API model: {spec.model_id}) ===")
    print(f"  file  : {out_path}")
    print(f"  done  : {len(done_keys)} / {len(selected)}")
    print(f"  todo  : {len(todo)}")
    if todo:
        cfg_todo.append((cfg, out_path, done_keys, todo))
    else:
        totals[f"{MODEL_NAME}/{cfg['name']}"] = {"new": 0, "skipped": len(done_keys)}

runner = GoogleVLM(spec)
try:
    runner.load()
    print(f"\nGoogle Generative AI client initialised for model '{spec.model_id}'.")
except Exception as exc:
    print(f"[load-failed] {exc}")
    failures.append({"model": MODEL_NAME, "stage": "load", "error": str(exc)})
    cfg_todo = []

try:
    for cfg, out_path, done_keys, todo in cfg_todo:
        new_count = 0
        desc = f"{MODEL_NAME}[{cfg['name']}]"
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
                    "model": MODEL_NAME,
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
                pbar.set_postfix(lat=f"{latency:.2f}s", err="yes" if err else "no",
                                 refresh=False)

                # Throttle to respect API rate limits.
                if REQUEST_DELAY > 0:
                    time.sleep(REQUEST_DELAY)

        totals[f"{MODEL_NAME}/{cfg['name']}"] = {
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
