"""Single-GPU response-only QLoRA training. Invoke behind the budget watchdog."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from .model import (MAX_LENGTH, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, attach_trainable_adapter,
                    encode_record, load_base, load_frozen_records, load_tokenizer, sha256_file, install_termination_handler)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=positive_int, default=100)
    parser.add_argument("--gradient-accumulation", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-seconds", type=positive_int, default=7200)
    parser.add_argument("--preflight-only", action="store_true", help="tokenize all rows without loading model weights")
    return parser


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    started = time.monotonic()
    rows = load_frozen_records(args.data, training=True)
    tokenizer = load_tokenizer()
    encoded = [encode_record(tokenizer, row) for row in rows]
    summary = {"rows": len(rows), "max_tokens": max(len(x["input_ids"]) for x in encoded), "supervised_tokens": sum(sum(label != -100 for label in x["labels"]) for x in encoded)}
    if args.preflight_only:
        print(json.dumps(summary))
        return
    # Exclusive directory prevents accidental overwrite of a prior adapter.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "system_prompt": SYSTEM_PROMPT, "max_length": MAX_LENGTH, "microbatch": 1,
                "seed": args.seed, "max_steps": args.max_steps, "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate, "training_sha256": sha256_file(args.data), **summary}
    write_json(args.output / "run_manifest.json", manifest)
    try:
        import importlib.metadata
        import torch
        from transformers import set_seed
        set_seed(args.seed)
        model = attach_trainable_adapter(load_base(training=True))
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        manifest["trainable_parameters"] = sum(p.numel() for p in parameters)
        manifest["versions"] = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["cuda"] = torch.version.cuda
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        randomizer = random.Random(args.seed)
        order = list(range(len(encoded)))
        randomizer.shuffle(order)
        position = 0
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(args.gradient_accumulation):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("training wall-time limit reached")
                if position == len(order):
                    randomizer.shuffle(order)
                    position = 0
                example = encoded[order[position]]
                position += 1
                batch = {k: torch.tensor([v], dtype=torch.long, device="cuda:0") for k, v in example.items()}
                loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite training loss")
                loss_sum += loss.detach().float().item()
                (loss / args.gradient_accumulation).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if not torch.isfinite(norm).item():
                raise RuntimeError("nonfinite gradient norm")
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in parameters):
                raise RuntimeError("all adapter gradients are missing or zero")
            optimizer.step()
            progress = {"step": step + 1, "loss": loss_sum / args.gradient_accumulation,
                        "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - started}
            with (args.output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
            print(json.dumps(progress), flush=True)
        if not all(torch.isfinite(p).all().item() for p in parameters):
            raise RuntimeError("adapter contains nonfinite parameters after optimizer updates")
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        manifest.update(status="completed", completed_steps=args.max_steps,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        elapsed_seconds=time.monotonic() - started,
                        adapter_sha256=sha256_file(args.output / "adapter_model.safetensors"),
                        adapter_config_sha256=sha256_file(args.output / "adapter_config.json"))
        write_json(args.output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    install_termination_handler()
    main()
