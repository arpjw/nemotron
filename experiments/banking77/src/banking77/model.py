"""Pinned Nemotron runtime and taxonomy-constrained classification helpers."""
from __future__ import annotations
import hashlib
import json
import signal
from pathlib import Path
from typing import Any

MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
MODEL_REVISION = "791833e92ebddb0bc2c1007f6d2b6764f886a2ae"
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 32
SYSTEM_PROMPT = "Classify the banking request. Reply with exactly one category from the supplied taxonomy."
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(Path(__file__).with_name(name)) for name in ("model.py", "train.py", "inference.py", "evaluation.py", "data.py")}


def runtime_versions() -> dict[str, str]:
    import importlib.metadata
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub")}


def config_hash(model: Any) -> str:
    return hashlib.sha256(model.config.to_json_string().encode()).hexdigest()


def load_taxonomy(directory: Path) -> list[str]:
    labels = json.loads((directory / "taxonomy.json").read_text())
    if not isinstance(labels, list) or len(labels) != 77 or len(set(labels)) != 77 or any(not isinstance(x, str) or not x or x != x.strip() for x in labels):
        raise ValueError("taxonomy must contain exactly 77 unique canonical labels")
    return labels


def prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("nonempty public prompt required")
    ids = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or not ids:
        raise ValueError("invalid chat tokenization")
    return ids


def encode_record(tokenizer: Any, row: dict, taxonomy: list[str]) -> dict[str, list[int]]:
    # Only public prompt and supervision target are tokenized.
    prefix = prompt_tokens(tokenizer, row["prompt"])
    if row["target"] not in taxonomy:
        raise ValueError("training target outside canonical taxonomy")
    completion = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if tokenizer.eos_token_id is None or len(completion) < 2:
        raise ValueError("missing label tokens or EOS")
    ids = prefix + completion
    if len(ids) > MAX_LENGTH:
        raise ValueError("training example exceeds context cap; no truncation allowed")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + completion}


class LabelTrie:
    """Constrain generated suffix to exactly one tokenized canonical label + EOS."""
    def __init__(self, tokenizer: Any, taxonomy: list[str]):
        if tokenizer.eos_token_id is None:
            raise ValueError("EOS required")
        self.eos = tokenizer.eos_token_id
        self.paths = [tuple(tokenizer.encode(label, add_special_tokens=False) + [self.eos]) for label in taxonomy]
        if not self.paths or len(set(self.paths)) != len(self.paths) or any(len(path) < 2 or self.eos in path[:-1] for path in self.paths):
            raise ValueError("taxonomy tokenization must be unique, nonempty, and EOS-free")
        if max(map(len, self.paths)) > MAX_NEW_TOKENS:
            raise ValueError("canonical label exceeds generation cap")

    def allowed(self, suffix: list[int]) -> list[int]:
        prefix = tuple(suffix)
        options = {path[len(prefix)] for path in self.paths if len(path) > len(prefix) and path[:len(prefix)] == prefix}
        if not options:
            if prefix in self.paths:
                return [self.eos]
            raise ValueError("generation left the canonical label trie")
        return sorted(options)

    def callback(self, prompt_length: int):
        def permitted(batch_id: int, tokens: Any) -> list[int]:
            return self.allowed(tokens.tolist()[prompt_length:])
        return permitted


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("missing native tokenizer metadata")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(training: bool = False) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, local_files_only=True, use_safetensors=False, weights_only=True, device_map={"": 0}, torch_dtype=torch.bfloat16, quantization_config=quantization, attn_implementation="sdpa")
    model.config.use_cache = not training
    return model


def attach_adapter(model: Any) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if set(TARGET_MODULES) - names:
        raise RuntimeError("native model lacks expected projection layers")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora_" not in name for name, _ in parameters):
        raise RuntimeError("unexpected trainable parameters")
    return model


def validate_adapter(path: Path, data_directory: Path) -> dict:
    metadata = json.loads((path / "run_manifest.json").read_text())
    expected = {"status": "completed", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "system_prompt": SYSTEM_PROMPT, "train_sha256": sha256_file(data_directory / "train.jsonl"),
                "taxonomy_sha256": sha256_file(data_directory / "taxonomy.json"),
                "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(path / "adapter_config.json")}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("adapter provenance mismatch")
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("peft_type") != "LORA":
        raise ValueError("adapter model configuration mismatch")
    return metadata


def install_termination_handler() -> None:
    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
