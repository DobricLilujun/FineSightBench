"""Fix the `comparison_chain` / `chain_reasoning` prompts whose `obj_type`
is `dot`: the original `_OBJ_HINT` had an empty string for `dot`, so the
generated prompt mentioned `<color> <identity>` without ever defining what
`<identity>` is. The reference answers use the literal token ``dot``.

This script:
  1. Patches the source `_OBJ_HINT` dict in
     ``notebooks/obsolete/generate_reasoning_dataset.ipynb`` (for future
     regeneration).
  2. Rewrites every affected ``question`` in local files:
       data/full_reasoning/labels.json, labels.jsonl
       data/full_reasoning_large/labels.json, labels.jsonl
  3. Re-uploads the fixed ``question`` column to both Hugging Face repos
     via ``datasets.load_dataset(...).map(...).push_to_hub(...)``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DOT_HINT = " Identities are simply 'dot'."
TRIGGER_TASKS = {"comparison_chain", "chain_reasoning"}


def _needs_fix(row: dict) -> bool:
    if row.get("task_type") not in TRIGGER_TASKS:
        return False
    q = row.get("question") or ""
    if "List all dots" not in q:
        return False
    # Skip if an identity clause (from any variant of the prompt) is already there.
    if "identity is" in q.lower() or "identities are" in q.lower() or "each object is" in q.lower():
        return False
    return True


def _patch_question(q: str) -> str:
    return q.rstrip() + DOT_HINT


# ── 1. Source notebook ────────────────────────────────────────────────────────

def patch_notebook_source() -> None:
    nb_path = ROOT / "notebooks" / "obsolete" / "generate_reasoning_dataset.ipynb"
    text = nb_path.read_text()
    old = '"    \\"dot\\":        \\"\\",\\n"'
    new = '"    \\"dot\\":        \\" Identities are simply \'dot\'.\\",\\n"'
    if old in text:
        nb_path.write_text(text.replace(old, new, 1))
        print(f"  [src] patched _OBJ_HINT['dot'] in {nb_path.relative_to(ROOT)}")
    else:
        print(f"  [src] skip (_OBJ_HINT dot entry already non-empty or not found) in {nb_path.relative_to(ROOT)}")


# ── 2. Local label files ──────────────────────────────────────────────────────

def patch_local_files() -> None:
    for ds_dir in ("full_reasoning", "full_reasoning_large"):
        base = ROOT / "data" / ds_dir
        if not base.exists():
            print(f"  [local] skip {ds_dir}: directory not present")
            continue

        # labels.json ---------------------------------------------------------
        lj = base / "labels.json"
        if lj.exists():
            data = json.loads(lj.read_text())
            n_fixed = 0
            for sample in data.get("samples", []):
                if _needs_fix(sample):
                    sample["question"] = _patch_question(sample["question"])
                    n_fixed += 1
            lj.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"  [local] {ds_dir}/labels.json  fixed={n_fixed}")

        # labels.jsonl --------------------------------------------------------
        ljl = base / "labels.jsonl"
        if ljl.exists():
            lines = ljl.read_text().splitlines()
            out: list[str] = []
            n_fixed = 0
            for line in lines:
                if not line.strip():
                    out.append(line)
                    continue
                row = json.loads(line)
                if _needs_fix(row):
                    row["question"] = _patch_question(row["question"])
                    n_fixed += 1
                out.append(json.dumps(row, ensure_ascii=False))
            ljl.write_text("\n".join(out) + "\n")
            print(f"  [local] {ds_dir}/labels.jsonl fixed={n_fixed}")


# ── 3. Hugging Face repos ────────────────────────────────────────────────────

def push_hf(repo_id: str, num_shards: dict[str, int]) -> None:
    print(f"\n  [hf] loading {repo_id} ...")
    ds = load_dataset(repo_id)

    fix_counts: dict[str, int] = {}

    def _fix(batch, split_name: str):
        out_q = []
        fixed = 0
        for t, q in zip(batch["task_type"], batch["question"]):
            if (
                t in TRIGGER_TASKS
                and q
                and "List all dots" in q
                and "identity is" not in q.lower()
                and "identities are" not in q.lower()
                and "each object is" not in q.lower()
            ):
                out_q.append(q.rstrip() + DOT_HINT)
                fixed += 1
            else:
                out_q.append(q)
        fix_counts[split_name] = fix_counts.get(split_name, 0) + fixed
        return {"question": out_q}

    new_splits = {}
    for split in ds.keys():
        new_splits[split] = ds[split].map(
            lambda b, _s=split: _fix(b, _s),
            batched=True,
            batch_size=1000,
            desc=f"{repo_id}:{split}",
        )
    for split, n in fix_counts.items():
        print(f"  [hf] {repo_id}:{split}  rows_fixed={n}")

    from datasets import DatasetDict
    new_ds = DatasetDict(new_splits)

    print(f"  [hf] pushing to {repo_id} with num_shards={num_shards} ...")
    new_ds.push_to_hub(
        repo_id,
        num_shards=num_shards,
        commit_message="Fix chain_reasoning/comparison_chain prompts for dot identity",
    )
    print(f"  [hf] done: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    print("=== 1. Patch notebook source ===")
    patch_notebook_source()

    print("\n=== 2. Patch local label files ===")
    patch_local_files()

    print("\n=== 3. Push updates to Hugging Face ===")
    push_hf("Volavion/FineSightBench", num_shards={"perception": 8, "reasoning": 8})
    push_hf("Volavion/FineSightBench-Large", num_shards={"perception": 32, "reasoning": 32})

    print("\nAll done.")


if __name__ == "__main__":
    main()
