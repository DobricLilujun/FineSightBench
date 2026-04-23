"""Generate FineSightBench-Large — same design as FineSightBench but with
every base count multiplied by 10.

Outputs (all written fresh, won't touch the existing full_* dirs):
  * data/full_perception_large/   (35 000 base + 7 000 textwild = 42 000)
  * data/full_reasoning_large/    (25 200 base + 14 000 textwild = 39 200)

Strategy: reuse the existing generation notebooks by executing their cells
with the output dirs / per-config counts overridden, then reuse the
text-in-the-wild merge pipeline pointed at the new *_large dirs.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/snt/projects_lujun/FineSightBench")
sys.path.insert(0, str(ROOT))

import os

CANVAS = 448
# Scale factor × 10 by default; set FSB_LARGE_SCALE=0.01 (etc.) for quick dry-runs.
_SCALE = float(os.environ.get("FSB_LARGE_SCALE", "10"))
PERC_NUM_PER_CONFIG = max(1, int(round(100 * _SCALE)))
REAS_N_PER_CONFIG = max(1, int(round(20 * _SCALE)))
TW_PERC_NUM_PER_SIZE = max(1, int(round(100 * _SCALE)))
TW_REAS_NUM_PER_CONFIG = max(1, int(round(25 * _SCALE)))

DST_PERC = ROOT / "data" / "full_perception_large"
DST_REAS = ROOT / "data" / "full_reasoning_large"
TMP_TW_PERC = ROOT / "data" / "_tw_perc_large_tmp"
TMP_TW_REAS = ROOT / "data" / "_tw_reas_large_tmp"


# ── notebook-cell runner ────────────────────────────────────────────────────

def _load_cells(nb_path: Path) -> list[dict]:
    return json.loads(nb_path.read_text())["cells"]


def _run_cell(ns: dict, src: str, label: str) -> None:
    t0 = time.time()
    exec(compile(src, f"<{label}>", "exec"), ns)
    print(f"    · {label}  ({time.time()-t0:.1f}s)")


# ── 1. perception base (cells 2, 4, 6, 8, 14, 16) ──────────────────────────

def gen_perception_base() -> None:
    nb = _load_cells(ROOT / "notebooks" / "generate_perception_dataset.ipynb")
    ns: dict = {"__name__": "__main__"}

    print(f"\n=== Perception base → {DST_PERC} ===")
    # Cell 2 — imports (defines repo_root, etc.)
    _run_cell(ns, "".join(nb[2]["source"]), "perc[2] imports")

    # Cell 4 — config. Then override OUTPUT_DIR and NUM_PER_CONFIG.
    _run_cell(ns, "".join(nb[4]["source"]), "perc[4] config")
    ns["NUM_PER_CONFIG"] = PERC_NUM_PER_CONFIG
    ns["OUTPUT_DIR"] = DST_PERC
    ns["TOTAL_IMAGES"] = (
        len(ns["PIXEL_SIZES"]) * PERC_NUM_PER_CONFIG * len(ns["TASK_NAMES"])
    )
    DST_PERC.mkdir(parents=True, exist_ok=True)
    print(f"    · OVERRIDE  NUM_PER_CONFIG={PERC_NUM_PER_CONFIG}  "
          f"OUTPUT_DIR={DST_PERC}  TOTAL={ns['TOTAL_IMAGES']}")

    # Cell 6 — run generate_perception_dataset (writes images + labels.json)
    _run_cell(ns, "".join(nb[6]["source"]), "perc[6] generate")

    # Cell 8 — statistics (defines by_task, by_task_size used later)
    _run_cell(ns, "".join(nb[8]["source"]), "perc[8] stats")

    # Cell 10 uses matplotlib; we need TASK_ORDER for cell 14. Extract only it.
    src10 = "".join(nb[10]["source"])
    # Stop right after TASK_ORDER assignment.
    stop = src10.index("]", src10.index("TASK_ORDER")) + 1
    _run_cell(ns, src10[:stop], "perc[10] TASK_ORDER only")

    # Cell 14 — CSV export + summary print
    _run_cell(ns, "".join(nb[14]["source"]), "perc[14] metadata.csv")

    # Cell 16 — JSONL export. Override the hard-coded output dir.
    src16 = "".join(nb[16]["source"])
    src16 = src16.replace(
        'Path(_fb.__file__).resolve().parent.parent / "data" / "full_perception"',
        f'Path(r"{DST_PERC}")',
    )
    _run_cell(ns, src16, "perc[16] labels.jsonl")


# ── 2. reasoning base (cells 2, 4, 6, 8, 16, 18) ───────────────────────────

def gen_reasoning_base() -> None:
    nb = _load_cells(ROOT / "notebooks" / "generate_reasoning_dataset.ipynb")
    ns: dict = {"__name__": "__main__"}

    print(f"\n=== Reasoning base → {DST_REAS} ===")
    _run_cell(ns, "".join(nb[2]["source"]), "reas[2] imports")

    _run_cell(ns, "".join(nb[4]["source"]), "reas[4] config")
    ns["N_PER_CONFIG"] = REAS_N_PER_CONFIG
    ns["OUTPUT_DIR"] = DST_REAS
    DST_REAS.mkdir(parents=True, exist_ok=True)
    print(f"    · OVERRIDE  N_PER_CONFIG={REAS_N_PER_CONFIG}  "
          f"OUTPUT_DIR={DST_REAS}")

    # Cell 6 — generator function definitions
    _run_cell(ns, "".join(nb[6]["source"]), "reas[6] generators")

    # Cell 8 — run generation (writes images + labels.json)
    _run_cell(ns, "".join(nb[8]["source"]), "reas[8] generate")

    # Cell 10 — stats
    _run_cell(ns, "".join(nb[10]["source"]), "reas[10] stats")

    # Cell 16 — CSV export + summary (uses by_task_size/DISPLAY_TASK_ORDER)
    _run_cell(ns, "".join(nb[16]["source"]), "reas[16] metadata.csv")

    # Cell 18 — JSONL export. Patch output dir.
    src18 = "".join(nb[18]["source"])
    src18 = src18.replace(
        'Path(_fb.__file__).resolve().parent.parent / "data" / "full_reasoning"',
        f'Path(r"{DST_REAS}")',
    )
    _run_cell(ns, src18, "reas[18] labels.jsonl")


# ── 3. textwild merge (×10) ─────────────────────────────────────────────────

def merge_textwild() -> None:
    # Reuse helpers from the existing merge script
    from scripts import merge_textwild_into_full as merge_mod
    from finesightbench.core.objects import TARGET_SIZES
    from finesightbench.textwild import (
        generate_textwild_perception,
        generate_textwild_reasoning,
    )

    print("\n=== TextWild × 10 — Perception ===")
    shutil.rmtree(TMP_TW_PERC, ignore_errors=True)
    generate_textwild_perception(
        output_dir=TMP_TW_PERC,
        canvas_size=CANVAS,
        sizes=TARGET_SIZES,
        num_per_size=TW_PERC_NUM_PER_SIZE,
        seed=142,
    )

    print("\n=== TextWild × 10 — Reasoning ===")
    shutil.rmtree(TMP_TW_REAS, ignore_errors=True)
    generate_textwild_reasoning(
        output_dir=TMP_TW_REAS,
        canvas_size=CANVAS,
        sizes=TARGET_SIZES,
        counts=[2, 4, 6, 8],
        num_per_config=TW_REAS_NUM_PER_CONFIG,
        seed=143,
    )

    print("\n=== Merging into full_perception_large ===")
    perc_data = json.loads((TMP_TW_PERC / "labels.json").read_text())
    merge_mod._copy_images(TMP_TW_PERC / "images", DST_PERC / "images")
    merge_mod._merge_labels_json(
        DST_PERC / "labels.json", perc_data["samples"],
        new_task_types=["text_recognition"],
    )
    merge_mod._merge_labels_jsonl(DST_PERC / "labels.jsonl", perc_data["samples"])
    merge_mod._append_metadata_csv(
        DST_PERC / "metadata.csv", perc_data["samples"], has_num_targets=False
    )

    print("\n=== Merging into full_reasoning_large ===")
    reas_data = json.loads((TMP_TW_REAS / "labels.json").read_text())
    merge_mod._copy_images(TMP_TW_REAS / "images", DST_REAS / "images")
    merge_mod._merge_labels_json(
        DST_REAS / "labels.json", reas_data["samples"],
        new_task_types=["text_reading_chain", "text_counting_chain"],
    )
    merge_mod._merge_labels_jsonl(DST_REAS / "labels.jsonl", reas_data["samples"])
    merge_mod._append_metadata_csv(
        DST_REAS / "metadata.csv", reas_data["samples"], has_num_targets=True
    )

    shutil.rmtree(TMP_TW_PERC, ignore_errors=True)
    shutil.rmtree(TMP_TW_REAS, ignore_errors=True)


def main() -> None:
    t0 = time.time()
    gen_perception_base()
    gen_reasoning_base()
    merge_textwild()
    print(f"\n=== All done in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
