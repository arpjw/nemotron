"""Pinned Nemotron QLoRA interface for one-edge book-graph repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from edit_study import prompt

MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
MODEL_REVISION = "791833e92ebddb0bc2c1007f6d2b6764f886a2ae"
SYSTEM_PROMPT = "You are a graph search assistant. Choose one edge flip and return only exact JSON."
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 32
SOURCE_FILES = ("book.py", "edit_study.py", "edit_model.py", "edit_train.py",
                "edit_infer.py", "edit_evaluate.py", "edit_baseline.py")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: file_hash(directory / name) for name in SOURCE_FILES}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prompt_for_row(row: dict) -> str:
    rows = [int(word, 16) for word in row["rows_hex"]]
    return prompt(rows, (row["loss"], row["red_violations"], row["blue_violations"]))


def encode_record(tokenizer: Any, row: dict) -> dict[str, list[int]]:
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt_for_row(row)}],
        tokenize=True, add_generation_prompt=True)
    u, v = row["teacher_action"]
    completion = tokenizer.encode(json.dumps({"u": u, "v": v}, separators=(",", ":")),
                                  add_special_tokens=False)
    if not prefix or not completion or tokenizer.eos_token_id is None:
        raise ValueError("invalid tokenization")
    completion += [tokenizer.eos_token_id]
    ids = prefix + completion
    if len(ids) > MAX_LENGTH:
        raise ValueError(f"task {row['id']} exceeds {MAX_LENGTH} tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + completion}


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               trust_remote_code=False, local_files_only=True)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("missing native chat template or EOS")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(*, training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        local_files_only=True, use_safetensors=False, weights_only=True,
        quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.config.use_cache = not training
    return model


def attach_trainable_adapter(model: Any) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if set(TARGET_MODULES) - available:
        raise RuntimeError("expected projection modules missing")
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                        target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("unexpected trainable weights")
    return model


def validate_adapter(directory: Path, train_path: Path) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    expected = {"status": "completed", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "train_sha256": file_hash(train_path),
                "adapter_sha256": file_hash(directory / "adapter_model.safetensors"),
                "adapter_config_sha256": file_hash(directory / "adapter_config.json")}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("adapter provenance mismatch")
    config = json.loads((directory / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("peft_type") != "LORA":
        raise ValueError("adapter config mismatch")
    return manifest
