"""Pinned model loading and response-only encoding. No import-time GPU work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
MODEL_REVISION = "791833e92ebddb0bc2c1007f6d2b6764f886a2ae"
MAX_LENGTH = 1024
SYSTEM_PROMPT = "Answer using only the supplied excerpt. Return only the requested JSON object."
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")


def prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise ValueError("chat template must produce a nonempty token list")
    return ids


def encode_record(tokenizer: Any, record: dict[str, Any], max_length: int = MAX_LENGTH) -> dict[str, list[int]]:
    """Encode only prompt and target. Reject overflow; never truncate supervision."""
    if not 1 <= max_length <= MAX_LENGTH:
        raise ValueError("max_length must be between 1 and 1024")
    prompt = prompt_tokens(tokenizer, record["prompt"])
    target = record["target"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty JSON string")
    from .scoring import parse_response
    if parse_response(target) is None:
        raise ValueError("target must satisfy the strict response schema")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    completion = tokenizer.encode(target, add_special_tokens=False)
    if not completion:
        raise ValueError("target produced no tokens")
    completion = completion + [tokenizer.eos_token_id]
    ids = prompt + completion
    if len(ids) > max_length:
        raise ValueError(f"record {record.get('id', '<unknown>')} requires {len(ids)} tokens; cap is {max_length}")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt) + completion}


def load_records(path: Path, allowed_split: str | None = None) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"line {number} lacks a string id")
            if row["id"] in seen:
                raise ValueError(f"duplicate id {row['id']}")
            seen.add(row["id"])
            if allowed_split is not None and row.get("split") != allowed_split:
                raise ValueError(f"only {allowed_split} split may be used here")
            records.append(row)
    if not records:
        raise ValueError("empty dataset")
    return records


def load_frozen_records(path: Path, *, training: bool = False, allow_sealed: bool = False) -> list[dict[str, Any]]:
    """Verify the entire frozen dataset and reject disguised or sealed split inputs."""
    from .data import verify_dataset
    manifest = verify_dataset(path.parent)
    if path.name not in manifest["files"]:
        raise ValueError("data must be a manifest-listed split file")
    split = path.stem
    if training and split != "train":
        raise ValueError("training requires the frozen train split")
    if split in ("test", "robustness") and not allow_sealed:
        raise ValueError("sealed evaluation requires --allow-sealed")
    rows = load_records(path, allowed_split=split)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("missing model EOS or chat template")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base(*, training: bool = False) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a CUDA GPU supporting BF16 is required")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, quantization_config=quantization, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", use_safetensors=False, weights_only=True, local_files_only=True)
    model.config.use_cache = not training
    return model


def attach_trainable_adapter(model: Any) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(LORA_TARGETS) - present
    if missing:
        raise RuntimeError(f"checkpoint lacks expected LoRA targets: {sorted(missing)}")
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("expected only LoRA adapter weights to be trainable")
    return model


def validate_adapter(path: Path, *, expected_training_sha256: str | None = None) -> dict[str, Any]:
    """Refuse an adapter with unrecorded or incompatible model provenance."""
    metadata = json.loads((path / "run_manifest.json").read_text())
    if metadata.get("model_id") != MODEL_ID or metadata.get("model_revision") != MODEL_REVISION:
        raise ValueError("adapter model provenance mismatch")
    if metadata.get("status") != "completed":
        raise ValueError("adapter run did not complete")
    if metadata.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("adapter system prompt mismatch")
    if expected_training_sha256 is not None and metadata.get("training_sha256") != expected_training_sha256:
        raise ValueError("adapter training dataset does not match current frozen train split")
    if sha256_file(path / "adapter_config.json") != metadata.get("adapter_config_sha256"):
        raise ValueError("adapter config checksum mismatch")
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("peft_type") != "LORA":
        raise ValueError("adapter configuration mismatch")
    weights = path / "adapter_model.safetensors"
    if sha256_file(weights) != metadata.get("adapter_sha256"):
        raise ValueError("adapter weight checksum mismatch")
    return metadata


def install_termination_handler() -> None:
    """Let normal exception cleanup mark CLI manifests failed on watchdog SIGTERM."""
    import signal
    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
