"""Identical deterministic base/adapter generation; labels never enter model inputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .model import (MAX_LENGTH, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, load_base, load_frozen_records,
                    load_tokenizer, prompt_tokens, sha256_file, validate_adapter, install_termination_handler)
from .train import positive_int, write_json


def generate_response(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 192) -> str:
    import torch
    if not 1 <= max_new_tokens < MAX_LENGTH:
        raise ValueError("max_new_tokens must be between 1 and 1023")
    ids = prompt_tokens(tokenizer, prompt)
    if len(ids) + max_new_tokens > MAX_LENGTH:
        raise ValueError("prompt plus generation budget exceeds 1024-token cap")
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                use_cache=True)
    return tokenizer.decode(result[0, len(ids):], skip_special_tokens=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for predictions and provenance")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--allow-sealed", action="store_true", help="acknowledge final test/robustness inference; no subsequent tuning")
    parser.add_argument("--limit", type=positive_int, help="smoke-only prefix subset; recorded in manifest")
    parser.add_argument("--max-new-tokens", type=positive_int, default=192)
    parser.add_argument("--max-seconds", type=positive_int, default=7200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens >= MAX_LENGTH:
        raise ValueError("generation budget must be below 1024 tokens")
    started = time.monotonic()
    rows = load_frozen_records(args.data, allow_sealed=args.allow_sealed)
    total_rows = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    adapter_metadata = validate_adapter(args.adapter, expected_training_sha256=sha256_file(args.data.parent / "train.jsonl")) if args.adapter else None
    tokenizer = load_tokenizer()
    # Validate every prompt before paying the model loading cost.
    for row in rows:
        if len(prompt_tokens(tokenizer, row["prompt"])) + args.max_new_tokens > MAX_LENGTH:
            raise ValueError(f"prompt {row['id']} exceeds context budget")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                "dataset_sha256": sha256_file(args.data), "adapter_sha256": adapter_metadata["adapter_sha256"] if adapter_metadata else None,
                "adapter_training_sha256": adapter_metadata["training_sha256"] if adapter_metadata else None,
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
                "quantization": "NF4-double-BF16", "rows": len(rows), "dataset_rows": total_rows,
                "allow_sealed": args.allow_sealed, "subset": len(rows) != total_rows, "selected_ids": [row["id"] for row in rows]}
    write_json(args.output / "inference_manifest.json", manifest)
    try:
        model = load_base()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("inference wall-time limit reached")
                prediction = generate_response(model, tokenizer, row["prompt"], args.max_new_tokens)
                stream.write(json.dumps({"id": row["id"], "prediction": prediction}) + "\n")
                stream.flush()
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        predictions_sha256=sha256_file(args.output / "predictions.jsonl"))
        write_json(args.output / "inference_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "inference_manifest.json", manifest)
        raise


if __name__ == "__main__":
    install_termination_handler()
    main()
