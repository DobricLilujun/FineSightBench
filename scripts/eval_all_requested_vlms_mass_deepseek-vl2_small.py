import json
import logging
import os
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
# not warnings - silence it at the logging level.
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


import torch
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

from finesightbench.evaluation.framework import MODEL_SPECS, list_supported_models, resolve_model_name


# --- Config -------------------------------------------------------------------
HF_DATASET_IDS = [
    "Volavion/FineSightBench",
]
SPLITS = ["perception", "reasoning"]

MODELS = [
    # "deepseek-vl2-tiny",
    "deepseek-vl2-small",
]

# Decoding configurations to run for EVERY model.
# Each produces its own JSONL: <model>__<cfg_name>.jsonl
DECODING_CONFIGS = [
    {"name": "greedy", "do_sample": False, "temperature": 1.0},
    {"name": "sample_t01", "do_sample": True, "temperature": 0.1},
    {"name": "sample_t10", "do_sample": True, "temperature": 1.0},
]

SEED = 42
ALLOW_DOWNLOAD = True  # set False to force local cache only
OUTPUT_DIR = Path("outputs/vlm_eval_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def output_path_for(model_name: str, cfg_name: str) -> Path:
    """One JSONL per (model, decoding-config) so runs are independent and resumable."""
    return OUTPUT_DIR / f"{_safe_filename(model_name)}__{_safe_filename(cfg_name)}.jsonl"


def _parse_simple_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _load_hf_token() -> tuple[str, str]:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("ACCESS_TOKEN")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        return token, "environment"

    roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for root in roots:
        for name in (".env", "env"):
            env_path = root / name
            if not env_path.exists():
                continue
            values = _parse_simple_dotenv(env_path)
            token = (
                values.get("HF_TOKEN")
                or values.get("HUGGINGFACE_HUB_TOKEN")
                or values.get("HUGGINGFACE_TOKEN")
                or values.get("ACCESS_TOKEN")
            )
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                return token, str(env_path)

    return "", "not found (.env/env); public-only access"


class DeepSeekVL2Runner:
    """DeepSeek-VL2 dedicated runner with transformers-5 compatibility patches."""

    def __init__(self, spec, hf_token: str = "", allow_download: bool = True) -> None:
        self.spec = spec
        self.hf_token = hf_token or None
        self.allow_download = allow_download
        self.model = None
        self.vl_processor = None
        self.tokenizer = None

    def _patch_transformers_for_deepseek(self) -> None:
        import transformers.utils.import_utils as _iu
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
        import transformers.models.llama.modeling_llama as _llama

        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: False
        # deepseek_vl2 imports this symbol from modeling_utils (transformers 4 layout).
        if not hasattr(_mu, "is_flash_attn_2_available") and hasattr(_iu, "is_flash_attn_2_available"):
            _mu.is_flash_attn_2_available = _iu.is_flash_attn_2_available
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
        if not hasattr(_llama, "LlamaFlashAttention2"):
            _llama.LlamaFlashAttention2 = _llama.LlamaAttention

        from transformers.cache_utils import DynamicCache as _DynCache

        if not hasattr(_DynCache, "seen_tokens"):
            _DynCache.seen_tokens = property(
                lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0
            )
        if not hasattr(_DynCache, "get_max_length"):
            _DynCache.get_max_length = lambda self: None
        if not hasattr(_DynCache, "get_usable_length"):
            _DynCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
                self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
            )
        if not hasattr(_DynCache, "from_legacy_cache"):
            _DynCache.from_legacy_cache = classmethod(lambda cls, past: cls())

        from transformers.models.llama.modeling_llama import (
            LlamaAttention as _LlamaAttn,
            LlamaRotaryEmbedding as _LlamaRoPE,
        )

        if getattr(_LlamaAttn.forward, "__name__", "") == "_patched_llama_forward_fsb":
            return

        _orig_llama_fwd = _LlamaAttn.forward

        def _patched_llama_forward_fsb(
            self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            if position_embeddings is None:
                if not hasattr(self, "_fs_rope"):
                    self._fs_rope = _LlamaRoPE(self.config).to(hidden_states.device)
                if position_ids is None:
                    bsz, seq_len = hidden_states.shape[:2]
                    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
                cos, sin = self._fs_rope(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            result = _orig_llama_fwd(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values if past_key_values is not None else past_key_value,
            )
            out, attn_w = result[0], result[1] if len(result) > 1 else None
            return out, attn_w, past_key_value

        _LlamaAttn.forward = _patched_llama_forward_fsb

    def _patch_siglip_attention_fallback(self) -> None:
        """Force DeepSeek SigLIP attention to use PyTorch SDPA instead of xformers.

        The bundled deepseek_vl2 code imports xformers inside forward(); with the
        current torch/xformers/triton build mix this crashes during JIT.
        """
        from deepseek_vl2.models import siglip_vit as _siglip_vit

        _SiglipAttn = _siglip_vit.Attention
        if getattr(_SiglipAttn.forward, "__name__", "") == "_patched_siglip_forward_fsb":
            return

        def _patched_siglip_forward_fsb(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, channels = x.shape
            qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)

            if not self.qk_norm:
                q, k, v = qkv.unbind(2)  # [B, N, H, D]
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
            else:
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = self.q_norm(q), self.k_norm(k)

            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        _SiglipAttn.forward = _patched_siglip_forward_fsb

    @staticmethod
    def _normalize_rope_scaling(config) -> None:
        """Bridge transformers>=4.50 rope_scaling schema to deepseek_vl2 expectations."""
        language_cfg = getattr(config, "language_config", None)
        if language_cfg is None:
            return
        rope_scaling = getattr(language_cfg, "rope_scaling", None)
        if not isinstance(rope_scaling, dict):
            return

        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        # "default" in newer transformers means no rope scaling.
        if rope_type in (None, "default"):
            language_cfg.rope_scaling = None
            return

        normalized = dict(rope_scaling)
        normalized["type"] = rope_type
        language_cfg.rope_scaling = normalized

    def load(self) -> None:
        self._patch_transformers_for_deepseek()

        from transformers import GenerationMixin
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2ForCausalLM

        # Apply after deepseek_vl2 import so we can patch its SigLIP class method.
        self._patch_siglip_attention_fallback()

        DeepseekVLV2ForCausalLM.all_tied_weights_keys = {}
        DeepseekV2ForCausalLM.all_tied_weights_keys = {}
        for cls in (DeepseekV2ForCausalLM, DeepseekVLV2ForCausalLM):
            if GenerationMixin not in cls.__mro__:
                cls.__bases__ = (GenerationMixin,) + cls.__bases__

        self.vl_processor = DeepseekVLV2Processor.from_pretrained(
            self.spec.model_id,
            token=self.hf_token,
            local_files_only=not self.allow_download,
        )
        self.tokenizer = self.vl_processor.tokenizer

        config_cls = getattr(DeepseekVLV2ForCausalLM, "config_class", None)
        if config_cls is None or not hasattr(config_cls, "from_pretrained"):
            raise RuntimeError("DeepseekVLV2ForCausalLM has no usable config_class")

        model_config = config_cls.from_pretrained(
            self.spec.model_id,
            token=self.hf_token,
            trust_remote_code=True,
            local_files_only=not self.allow_download,
        )
        self._normalize_rope_scaling(model_config)

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            self.spec.model_id,
            config=model_config,
            token=self.hf_token,
            dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=not self.allow_download,
        )

        # transformers 5 can skip custom checkpoint weights for deepseek_vl2.
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        snap = snapshot_download(
            self.spec.model_id,
            token=self.hf_token,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=not self.allow_download,
        )
        shards = sorted(Path(snap).glob("model*.safetensors"))
        state = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        self.model.load_state_dict(state, strict=False)

        self.model.to(dtype=torch.bfloat16)
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        self.vl_processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_inputs(self, image: Image.Image, question: str):
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare = self.vl_processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt="",
        )
        return prepare.to(DEVICE, dtype=torch.bfloat16)

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str, do_sample: bool, temperature: float) -> str:
        prepare = self._build_inputs(image, question)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare)
        attn_mask = prepare.attention_mask
        embed_layer = self.model.language.get_input_embeddings()
        eos_id = self.tokenizer.eos_token_id

        generated: list[int] = []
        cur_embeds, cur_mask = inputs_embeds, attn_mask
        for _ in range(int(self.spec.max_new_tokens)):
            out = self.model.language.model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = self.model.language.lm_head(out.last_hidden_state[:, -1, :])

            if do_sample:
                t = max(float(temperature), 1e-5)
                probs = torch.softmax(logits / t, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(logits.argmax(dim=-1).item())

            if next_id == eos_id:
                break
            generated.append(next_id)

            next_emb = embed_layer(torch.tensor([[next_id]], device=DEVICE)).to(cur_embeds.dtype)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=DEVICE)],
                dim=1,
            )

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


print("Supported models :", list_supported_models())
print("DeepSeek models  :", MODELS)
print("Device           :", DEVICE)
if DEVICE == "cuda":
    print("GPU              :", torch.cuda.get_device_name(0))
print("Output dir       :", OUTPUT_DIR)
HF_TOKEN, HF_TOKEN_SOURCE = _load_hf_token()
print("HF token source  :", HF_TOKEN_SOURCE)


# --- Select all items per (dataset_id, split, task_type) ----------------------
rng = random.Random(SEED)
selected: list[dict] = []  # list of small dicts carrying PIL image + metadata

for ds_id in HF_DATASET_IDS:
    ds = load_dataset(ds_id, token=HF_TOKEN or None)
    for split in SPLITS:
        if split not in ds:
            print(f"  [skip] split '{split}' not in {ds_id}")
            continue
        by_task: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(ds[split]):
            by_task[row["task_type"]].append(idx)
        for task, idxs in by_task.items():
            rng.shuffle(idxs)
            for idx in idxs:
                row = ds[split][idx]
                selected.append(
                    {
                        "dataset_id": ds_id,
                        "split": split,
                        "row_index": idx,
                        "image_id": row.get("image_id"),
                        "task_type": row.get("task_type"),
                        "difficulty": row.get("difficulty"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "image": row["image"],  # PIL.Image
                    }
                )

print(f"\nSelected {len(selected)} samples in total")
_counts: dict[tuple, int] = defaultdict(int)
for s in selected:
    _counts[(s["dataset_id"], s["split"], s["task_type"])] += 1
for (d, sp, t), n in sorted(_counts.items()):
    print(f"  {d:28s} {sp:10s} {t:30s} {n}")


# --- Generate per-model; one JSONL per model; resumable ----------------------
# Each record is flushed immediately (one line at a time). On re-run we skip any
# (dataset_id, split, row_index) already present in the model's JSONL, so an
# interrupted run can be continued simply by re-executing this script.

def _load_done_keys(path: Path, current_q_map: dict[tuple, str]) -> set[tuple]:
    """Read existing JSONL and return completed (dataset_id, split, row_index)."""
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
    if not resolved.startswith("deepseek-vl2"):
        print(f"[skip] {resolved}: this script only supports DeepSeek-VL2 models")
        continue

    cfg_todo: list[tuple[dict, Path, set[tuple], list[dict]]] = []
    for cfg in DECODING_CONFIGS:
        out_path = output_path_for(resolved, cfg["name"])
        done_keys = _load_done_keys(out_path, current_q_map)
        todo = [
            s
            for s in selected
            if (s["dataset_id"], s["split"], s["row_index"]) not in done_keys
        ]
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

    runner = DeepSeekVL2Runner(spec, hf_token=HF_TOKEN, allow_download=ALLOW_DOWNLOAD)
    try:
        t0 = time.time()
        runner.load()
        print(f"  loaded {resolved} in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"  [load-failed] {exc}")
        failures.append(
            {
                "model": resolved,
                "stage": "load",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }
        )
        runner.unload()
        continue

    try:
        for cfg, out_path, done_keys, todo in cfg_todo:
            new_count = 0
            desc = f"{resolved}[{cfg['name']}]"
            with out_path.open("a", encoding="utf-8") as f_out:
                pbar = tqdm(todo, desc=desc, unit="sample", dynamic_ncols=True)
                for sample in pbar:
                    image = sample["image"].convert("RGB")
                    question = sample["question"] or ""
                    t0 = time.time()
                    try:
                        generated = runner.predict(
                            image,
                            question,
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
                    pbar.set_postfix(
                        lat=f"{latency:.2f}s",
                        err="yes" if err else "no",
                        refresh=False,
                    )

            totals[f"{resolved}/{cfg['name']}"] = {
                "new": new_count,
                "skipped": len(done_keys),
            }
            print(f"  wrote {new_count} new rows -> {out_path}")
    finally:
        runner.unload()

print("\n=== Summary ===")
for m, c in totals.items():
    print(f"  {m:50s} new={c['new']:4d}  resumed-skip={c['skipped']:4d}")
if failures:
    print(f"\nFailures: {len(failures)}")
    for f in failures:
        print(" -", f["model"], f["stage"], f["error"])
