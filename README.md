# FineSightBench
FineSight-Perception / FineSight-Reasoning

## Abstract Of this repository

Recent vision-language models (VLMs) have demonstrated impressive multimodal understanding and reasoning capabilities, enabling neural networks to perceive aspects of the visual world and produce language-based analysis grounded in visual input. These advances have also made applications such as AI-based video dialogue increasingly feasible. Despite this progress, the fine-grained visual perception ability of VLMs remains poorly understood. In particular, it is still unclear how small or subtle the visual details are that VLMs can reliably perceive, how much useful information they can extract from images, and to what extent such information can support accurate reasoning and language expression. In general, VLMs are often believed to operate at a patch-level granularity rather than a truly pixel-level resolution. To address this gap, we introduce a novel benchmark for evaluating the fine-grained visual perception capabilities of VLMs. Using this dataset, we systematically investigate how small pixel-level objects state-of-the-art VLMs can effectively perceive, and we provide preliminary analyses to improve the interpretability of the observed behaviors. Furthermore, we argue that “seeing” and “seeing, reasoning, and describing correctly” are two distinct levels of visual competence. Motivated by this distinction, we construct a second benchmark focused on visual reasoning, which evaluates VLMs’ sensitivity to visual tokens and their ability to reason over and verbalize fine-grained visual information. In addition, we propose a novel plug-and-play module, which can be viewed as a pair of “glasses” for VLMs, implemented as a decoder that enhances their ability to capture visual details without any additional fine-tuning of the original model. Experimental results show that our method substantially improves fine-grained visual perception on the proposed benchmarks, while also yielding stronger performance on visually intensive reasoning tasks, particularly in detailed observation and spatial reasoning

## VLM Evaluation Framework (val_data)

The repository now includes a unified evaluation framework for VLM benchmarking on `data/val_data`.

Main entry points:

- Python API: `finesightbench.evaluation.framework`
- CLI: `python -m finesightbench.evaluation.framework`

What it does:

1. Validates dataset usability (labels, image existence, image readability)
2. Benchmarks model accuracy (overall, by split, by task)
3. Runs a raw-attention visualization smoke test and saves an attention GIF
4. Saves per-model reports and a cross-model summary CSV/JSON

Supported model presets:

- `Qwen3-VL-2B-Instruct`
- `Llama-4-Scout-17B-16E-Instruct`
- `InternVL3_5-1B-Flash`
- `deepseek-vl2-tiny`
- `GLM-4.6V-Flash`
- `gemma-4-E2B-it`

Example CLI:

```bash
python -m finesightbench.evaluation.framework \
	--val-root data/val_data \
	--models Qwen3-VL-2B-Instruct InternVL3_5-1B-Flash \
	--max-samples-per-split 60 \
	--output-dir outputs/vlm_eval \
	--allow-download
```

Generated notebooks:

- `notebooks/eval_qwen3_vl_2b_instruct.ipynb`
- `notebooks/eval_llama_4_scout_17b_16e_instruct.ipynb`
- `notebooks/eval_internvl3_5_1b_flash.ipynb`
- `notebooks/eval_deepseek_vl2_tiny.ipynb`
- `notebooks/eval_glm_4_6v_flash.ipynb`
- `notebooks/eval_gemma_4_e2b_it.ipynb`
- `notebooks/eval_all_requested_vlms.ipynb`