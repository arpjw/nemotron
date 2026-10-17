"""Frozen zero-shot transfer inference for the selected v1 graph adapter."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from edit_model import (MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT,
                        file_hash, load_base, load_tokenizer, validate_adapter,
                        write_json)
from transfer_v2 import load_split, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v1-train", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.output.exists() or args.max_seconds < 1:
        parser.error("new output directory and positive deadline required")
    rows = load_split(args.data, args.split)
    adapter_sha = None
    if args.adapter is not None:
        info = validate_adapter(args.adapter, args.v1_train)
        adapter_sha = info["adapter_sha256"]
        if adapter_sha != "606c4efd0a33a808399daaab7a580ad3253181d1c65cece5667083153104e2cf":
            raise ValueError("not the selected v1 adapter")
    args.output.mkdir(parents=True)
    manifest = {"status": "started", "version": 2, "split": args.split,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "adapter_sha256": adapter_sha,
                "dataset_sha256": file_hash(args.data / f"{args.split}.jsonl"),
                "rows": len(rows), "generation": {"do_sample": False,
                                                     "max_new_tokens": MAX_NEW_TOKENS}}
    write_json(args.output / "inference_manifest.json", manifest)
    started = time.monotonic()
    try:
        import torch
        tokenizer = load_tokenizer()
        model = load_base(training=False)
        if args.adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        with (args.output / "predictions.jsonl").open("w") as stream:
            for row in rows:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("transfer inference deadline")
                tokens = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt(row)}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to("cuda:0")
                with torch.inference_mode():
                    result = model.generate(tokens, attention_mask=torch.ones_like(tokens),
                                            do_sample=False, num_beams=1,
                                            max_new_tokens=MAX_NEW_TOKENS,
                                            pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(result[0][tokens.shape[-1]:], skip_special_tokens=True)
                stream.write(json.dumps({"id": row["id"], "raw": raw}, ensure_ascii=False) + "\n")
                stream.flush()
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        predictions_sha256=file_hash(args.output / "predictions.jsonl"))
        write_json(args.output / "inference_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__,
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "inference_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
