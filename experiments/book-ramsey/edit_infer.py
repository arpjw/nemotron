"""Greedy one-action inference on the frozen graph-repair development split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from edit_model import (MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT,
                        file_hash, load_base, load_tokenizer, prompt_for_row,
                        source_hashes, validate_adapter, write_json)
from edit_study import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--limit", type=int, help="engineering smoke only")
    args = parser.parse_args()
    if args.split == "test" and (not args.allow_test or args.selection is None):
        parser.error("test requires --allow-test and a frozen development selection")
    if args.max_seconds < 1 or (args.limit is not None and args.limit < 1):
        parser.error("positive deadline and limit required")
    rows = load_split(args.data, args.split)
    selected = rows[:args.limit] if args.limit is not None else rows
    if args.output.exists():
        parser.error("output already exists")
    args.output.mkdir(parents=True)
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "split": args.split, "dataset_sha256": file_hash(args.data / f"{args.split}.jsonl"),
                "rows": len(selected), "full_split": len(selected) == len(rows),
                "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
                "source_sha256": source_hashes(), "adapter_sha256": None}
    if args.adapter is not None:
        info = validate_adapter(args.adapter, args.data / "train.jsonl")
        if not info.get("epoch_complete") and args.limit is None:
            parser.error("formal inference requires a complete training epoch")
        manifest["adapter_sha256"] = info["adapter_sha256"]
    if args.split == "test":
        selection = json.loads(args.selection.read_text())
        if selection.get("version") != 1 or selection.get("model_id") != MODEL_ID or \
                selection.get("model_revision") != MODEL_REVISION or \
                selection.get("dev_dataset_sha256") != file_hash(args.data / "dev.jsonl") or \
                not selection.get("dev_report_sha256") or not selection.get("adapter_sha256") or \
                (args.adapter is not None and
                 selection.get("adapter_sha256") != manifest["adapter_sha256"]):
            raise ValueError("test selection does not bind model, data, and adapter")
        manifest["selection_sha256"] = file_hash(args.selection)
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
            for row in selected:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("inference deadline")
                tokens = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt_for_row(row)}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
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
