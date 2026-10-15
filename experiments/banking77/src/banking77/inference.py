"""Comparable primary trie-constrained labels and raw-format diagnostics."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any
from .model import (MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, MAX_LENGTH, MAX_NEW_TOKENS,
                    LabelTrie, install_termination_handler, load_base, load_taxonomy,
                    load_tokenizer, prompt_tokens, sha256_file, validate_adapter, write_json, source_hashes, runtime_versions, config_hash)
from .train import positive


def predict(model: Any, tokenizer: Any, prompt: str, trie: LabelTrie) -> dict[str, str]:
    import torch
    ids = prompt_tokens(tokenizer, prompt)
    if len(ids) + MAX_NEW_TOKENS > MAX_LENGTH:
        raise ValueError("prompt and output budget exceed context cap")
    tokens = torch.tensor([ids], device=model.device, dtype=torch.long)
    common = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "do_sample": False, "num_beams": 1,
              "max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": tokenizer.eos_token_id,
              "pad_token_id": tokenizer.pad_token_id, "use_cache": True}
    with torch.inference_mode():
        constrained = model.generate(**common, prefix_allowed_tokens_fn=trie.callback(len(ids)))
        raw = model.generate(**common)
    return {"constrained": tokenizer.decode(constrained[0, len(ids):], skip_special_tokens=True),
            "raw": tokenizer.decode(raw[0, len(ids):], skip_special_tokens=True)}


def main(argv: list[str] | None = None) -> None:
    from .data import load_frozen_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--allow-sealed", action="store_true")
    parser.add_argument("--frozen-data", type=Path, help="original frozen train/dev directory required for official test")
    parser.add_argument("--limit", type=positive, help="engineering smoke only; cannot enter full comparison")
    parser.add_argument("--max-seconds", type=positive, default=3600)
    args = parser.parse_args(argv)
    if args.data.name not in ("dev.jsonl", "test.jsonl"):
        raise ValueError("inference requires frozen dev or explicitly unsealed test")
    if args.data.name == "test.jsonl" and not args.allow_sealed:
        raise ValueError("official test requires --allow-sealed and frozen test export")
    started = time.monotonic()
    sealed = args.data.stem == "test"
    selection = None
    if sealed:
        if args.limit is not None or args.frozen_data is None:
            raise ValueError("official test forbids --limit and requires --frozen-data")
        from .sealed_test import load_test_records
        from .test_guard import validate_frozen_training
        rows = load_test_records(args.data)
        selection = json.loads((args.data.parent / "selection.json").read_text())
        validate_frozen_training(selection, args.frozen_data)
    else:
        rows = load_frozen_records(args.data, allowed_split="dev")
    total = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    taxonomy = load_taxonomy(args.data.parent)
    adapter = validate_adapter(args.adapter, args.frozen_data if sealed else args.data.parent) if args.adapter else None
    tokenizer = load_tokenizer()
    trie = LabelTrie(tokenizer, taxonomy)
    for row in rows:
        if len(prompt_tokens(tokenizer, row["prompt"])) + MAX_NEW_TOKENS > MAX_LENGTH:
            raise ValueError("prompt exceeds context budget")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                "dataset_sha256": sha256_file(args.data), "taxonomy_sha256": sha256_file(args.data.parent / "taxonomy.json"),
                "adapter_sha256": adapter["adapter_sha256"] if adapter else None,
                "adapter_epoch_complete": adapter["epoch_complete"] if adapter else None,
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "modes": ["taxonomy_trie", "raw"]},
                "source_sha256": source_hashes(), "quantization": "NF4-double-BF16", "rows": len(rows), "dataset_rows": total,
                "subset": len(rows) != total, "selected_ids": [row["id"] for row in rows], "sealed": args.data.stem == "test"}
    if sealed:
        from .test_guard import validate_test_binding, reserve_test_inference
        validate_test_binding(selection, manifest, adapter=args.adapter is not None)
        manifest["selection_sha256"] = sha256_file(args.data.parent / "selection.json")
        manifest["test_guard_sha256"] = sha256_file(Path(__file__).with_name("test_guard.py"))
        receipt = reserve_test_inference(args.data.parent, args.output, adapter=args.adapter is not None, metadata=manifest)
        manifest["test_inference_receipt"] = receipt.name
        manifest["test_inference_receipt_sha256"] = sha256_file(receipt)
    write_json(args.output / "inference_manifest.json", manifest)
    try:
        model = load_base()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        import torch
        manifest.update(versions=runtime_versions(), model_config_sha256=config_hash(model),
                        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        with (args.output / "predictions.jsonl").open("x") as stream:
            for row in rows:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("inference deadline")
                result = predict(model, tokenizer, row["prompt"], trie)
                stream.write(json.dumps({"id": row["id"], **result}) + "\n")
                stream.flush()
        manifest.update(status="completed", predictions_sha256=sha256_file(args.output / "predictions.jsonl"), elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "inference_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "inference_manifest.json", manifest)
        raise


if __name__ == "__main__":
    install_termination_handler()
    main()
