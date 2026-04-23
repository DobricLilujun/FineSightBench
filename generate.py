#!/usr/bin/env python3
"""CLI for generating and visualising FineSightBench datasets.

Usage examples
──────────────
  # Generate both datasets (default 5 samples per config)
  python generate.py all --output data

  # Generate only perception dataset
  python generate.py perception --output data/perception --num-per-config 10

  # Generate only reasoning dataset
  python generate.py reasoning --output data/reasoning

  # Visualise a generated dataset (overall grid)
  python generate.py visualize --data-dir data/perception --num-display 30

  # Visualise per-task grids
  python generate.py visualize --data-dir data/reasoning --by-task
"""

from __future__ import annotations

import argparse
import sys


def cmd_perception(args: argparse.Namespace) -> None:
    from finesightbench.perception import generate_perception_dataset
    from finesightbench.visualize import visualize_dataset, visualize_by_task

    path = generate_perception_dataset(
        output_dir=args.output,
        canvas_size=args.canvas_size,
        num_per_config=args.num_per_config,
        seed=args.seed,
    )
    # auto-visualise
    visualize_dataset(args.output, num_display=30)
    visualize_by_task(args.output, samples_per_task=16)


def cmd_reasoning(args: argparse.Namespace) -> None:
    from finesightbench.reasoning import generate_reasoning_dataset
    from finesightbench.visualize import visualize_dataset, visualize_by_task

    path = generate_reasoning_dataset(
        output_dir=args.output,
        canvas_size=args.canvas_size,
        num_per_config=args.num_per_config,
        seed=args.seed,
    )
    visualize_dataset(args.output, num_display=30)
    visualize_by_task(args.output, samples_per_task=16)


def cmd_all(args: argparse.Namespace) -> None:
    from finesightbench.perception import generate_perception_dataset
    from finesightbench.reasoning import generate_reasoning_dataset
    from finesightbench.visualize import visualize_dataset, visualize_by_task

    p_dir = f"{args.output}/perception"
    r_dir = f"{args.output}/reasoning"

    generate_perception_dataset(
        output_dir=p_dir,
        canvas_size=args.canvas_size,
        num_per_config=args.num_per_config,
        seed=args.seed,
    )
    generate_reasoning_dataset(
        output_dir=r_dir,
        canvas_size=args.canvas_size,
        num_per_config=args.num_per_config,
        seed=args.seed,
    )

    for d in [p_dir, r_dir]:
        visualize_dataset(d, num_display=30)
        visualize_by_task(d, samples_per_task=16)


def cmd_visualize(args: argparse.Namespace) -> None:
    from finesightbench.visualize import visualize_dataset, visualize_by_task

    if args.by_task:
        visualize_by_task(args.data_dir, samples_per_task=args.num_display)
    else:
        visualize_dataset(args.data_dir, num_display=args.num_display)


def cmd_textwild(args: argparse.Namespace) -> None:
    """Generate the SynthText-style text-in-the-wild perception+reasoning data."""
    from finesightbench.textwild import (
        generate_textwild_perception,
        generate_textwild_reasoning,
    )
    from finesightbench.visualize import visualize_dataset, visualize_by_task

    p_out = f"{args.output}/textwild_perception"
    r_out = f"{args.output}/textwild_reasoning"

    if args.split in ("perception", "all"):
        generate_textwild_perception(
            output_dir=p_out,
            canvas_size=args.canvas_size,
            num_per_size=args.num_per_size,
            seed=args.seed,
            bg_dir=args.bg_dir,
        )
        try:
            visualize_dataset(p_out, num_display=30)
            visualize_by_task(p_out, samples_per_task=16)
        except Exception as e:  # visualisation is best-effort
            print(f"[textwild] perception visualisation skipped: {e}")

    if args.split in ("reasoning", "all"):
        generate_textwild_reasoning(
            output_dir=r_out,
            canvas_size=args.canvas_size,
            num_per_config=args.num_per_config,
            seed=args.seed,
            bg_dir=args.bg_dir,
        )
        try:
            visualize_dataset(r_out, num_display=30)
            visualize_by_task(r_out, samples_per_task=16)
        except Exception as e:
            print(f"[textwild] reasoning visualisation skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FineSightBench – dataset generation & visualisation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- shared arguments ---
    gen_parent = argparse.ArgumentParser(add_help=False)
    gen_parent.add_argument("--output", "-o", default="data", help="Output directory")
    gen_parent.add_argument("--canvas-size", type=int, default=512)
    gen_parent.add_argument("--num-per-config", "-n", type=int, default=5,
                            help="Samples per (task, size) combination")
    gen_parent.add_argument("--seed", type=int, default=42)

    sub.add_parser("perception", parents=[gen_parent], help="Generate perception dataset")
    sub.add_parser("reasoning", parents=[gen_parent], help="Generate reasoning dataset")
    sub.add_parser("all", parents=[gen_parent], help="Generate both datasets")

    vis = sub.add_parser("visualize", help="Visualise a generated dataset")
    vis.add_argument("--data-dir", "-d", required=True, help="Dataset directory with labels.json")
    vis.add_argument("--num-display", "-n", type=int, default=25)
    vis.add_argument("--by-task", action="store_true", help="One grid per task type")

    tw = sub.add_parser(
        "textwild",
        help="Generate SynthText-style text-in-the-wild perception+reasoning data",
    )
    tw.add_argument("--output", "-o", default="data", help="Output root directory")
    tw.add_argument("--canvas-size", type=int, default=512)
    tw.add_argument("--split", choices=["perception", "reasoning", "all"], default="all")
    tw.add_argument("--num-per-size", type=int, default=100,
                    help="Perception: samples per pixel size (default 100).")
    tw.add_argument("--num-per-config", type=int, default=25,
                    help="Reasoning: samples per (size, count) config (default 25).")
    tw.add_argument("--bg-dir", default=None,
                    help="Directory of background images. If omitted, the "
                         "Stanford Background Dataset is auto-downloaded.")
    tw.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dispatch = {
        "perception": cmd_perception,
        "reasoning": cmd_reasoning,
        "all": cmd_all,
        "visualize": cmd_visualize,
        "textwild": cmd_textwild,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
