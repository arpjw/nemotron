"""One bounded epoch of response-only QLoRA intent classification."""
from __future__ import annotations
import argparse
import importlib.metadata
import math
import random
import time
from pathlib import Path
from .model import (MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, MAX_LENGTH, attach_adapter,
                    encode_record, install_termination_handler, load_base, load_taxonomy,
                    load_tokenizer, sha256_file, write_json, source_hashes, runtime_versions, config_hash)


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def accumulation_windows(examples: list, accumulation: int):
    if accumulation <= 0:
        raise ValueError("positive accumulation required")
    for start in range(0, len(examples), accumulation):
        yield examples[start:start + accumulation]


def main(argv: list[str] | None = None) -> None:
    from .data import load_frozen_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gradient-accumulation", type=positive, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-seconds", type=positive, default=5400)
    parser.add_argument("--smoke-steps", type=positive, help="truncate epoch for engineering smoke only")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.data.name != "train.jsonl":
        raise ValueError("training requires manifest-listed train.jsonl")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("finite positive learning rate required")
    started = time.monotonic()
    rows = load_frozen_records(args.data, allowed_split="train")
    taxonomy = load_taxonomy(args.data.parent)
    tokenizer = load_tokenizer()
    encoded = [encode_record(tokenizer, row, taxonomy) for row in rows]
    if not encoded:
        raise ValueError("empty training data")
    summary = {"examples": len(rows), "max_tokens": max(len(item["input_ids"]) for item in encoded),
               "planned_epoch_steps": math.ceil(len(rows) / args.gradient_accumulation)}
    if args.preflight_only:
        print(summary)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                "train_sha256": sha256_file(args.data), "taxonomy_sha256": sha256_file(args.data.parent / "taxonomy.json"),
                "data_manifest_sha256": sha256_file(args.data.parent / "manifest.json"), "max_length": MAX_LENGTH,
                "seed": args.seed, "learning_rate": args.learning_rate, "gradient_accumulation": args.gradient_accumulation,
                "source_sha256": source_hashes(), "microbatch": 1, "smoke_steps": args.smoke_steps, **summary}
    write_json(args.output / "run_manifest.json", manifest)
    try:
        import torch
        from transformers import set_seed
        set_seed(args.seed)
        model = attach_adapter(load_base(training=True))
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        manifest.update(versions=runtime_versions(), model_config_sha256=config_hash(model),
                        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                        trainable_parameters=sum(p.numel() for p in parameters))
        random.Random(args.seed).shuffle(encoded)
        torch.cuda.reset_peak_memory_stats()
        steps = seen = 0
        for window in accumulation_windows(encoded, args.gradient_accumulation):
            if args.smoke_steps is not None and steps >= args.smoke_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for example in window:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("training deadline")
                batch = {key: torch.tensor([value], device="cuda:0", dtype=torch.long) for key, value in example.items()}
                loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite loss")
                total_loss += loss.detach().float().item()
                (loss / len(window)).backward()
                seen += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in parameters):
                raise RuntimeError("zero adapter gradients")
            optimizer.step()
            steps += 1
            with (args.output / "training_log.jsonl").open("a") as stream:
                import json
                stream.write(json.dumps({"step": steps, "examples_seen": seen, "loss": total_loss / len(window),
                                         "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - started}, allow_nan=False) + "\n")
            print(f"step={steps} examples={seen} loss={total_loss / len(window):.6f}", flush=True)
        if not all(torch.isfinite(p).all().item() for p in parameters):
            raise RuntimeError("nonfinite final adapter")
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        manifest.update(status="completed", completed_steps=steps, examples_seen=seen, epoch_complete=seen == len(rows),
                        elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
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
