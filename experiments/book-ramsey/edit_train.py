"""Train a fresh response-only QLoRA adapter on frozen graph-repair actions."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from edit_model import (MODEL_ID, MODEL_REVISION, attach_trainable_adapter, encode_record,
                        file_hash, load_base, load_tokenizer, source_hashes, write_json)
from edit_study import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.accumulation < 1 or args.max_seconds < 1 or \
            (args.smoke_steps is not None and args.smoke_steps < 1):
        parser.error("positive accumulation, deadline and smoke steps required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("finite positive learning rate required")
    rows = load_split(args.data, "train")
    tokenizer = load_tokenizer()
    examples = [encode_record(tokenizer, row) for row in rows]
    preflight = {"rows": len(rows), "max_tokens": max(len(item["input_ids"]) for item in examples),
                 "planned_steps": math.ceil(len(rows) / args.accumulation)}
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    if args.output.exists():
        parser.error("output already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "train_sha256": file_hash(args.data / "train.jsonl"), "seed": args.seed,
                "learning_rate": args.learning_rate, "accumulation": args.accumulation,
                "smoke_steps": args.smoke_steps, "source_sha256": source_hashes(), **preflight}
    write_json(args.output / "run_manifest.json", manifest)
    try:
        import torch
        from transformers import set_seed
        set_seed(args.seed)
        model = attach_trainable_adapter(load_base(training=True))
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        random.Random(args.seed).shuffle(examples)
        steps = seen = 0
        for start in range(0, len(examples), args.accumulation):
            if args.smoke_steps is not None and steps >= args.smoke_steps:
                break
            window = examples[start:start + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for example in window:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("training deadline")
                batch = {key: torch.tensor([value], device="cuda:0", dtype=torch.long)
                         for key, value in example.items()}
                loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite loss")
                (loss / len(window)).backward()
                losses.append(float(loss.detach()))
                seen += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in parameters):
                raise RuntimeError("zero adapter gradients")
            optimizer.step()
            steps += 1
            with (args.output / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": steps, "seen": seen,
                                         "loss": sum(losses) / len(losses),
                                         "gradient_norm": float(norm),
                                         "seconds": time.monotonic() - started}) + "\n")
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        manifest.update(status="completed", steps=steps, seen=seen,
                        epoch_complete=seen == len(examples), elapsed_seconds=time.monotonic() - started,
                        adapter_sha256=file_hash(args.output / "adapter_model.safetensors"),
                        adapter_config_sha256=file_hash(args.output / "adapter_config.json"))
        write_json(args.output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__,
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
