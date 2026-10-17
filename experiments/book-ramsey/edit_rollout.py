"""Bounded exploratory iterative inference with exact book feedback."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from book import verify
from edit_evaluate import parse_action
from edit_model import (MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT,
                        file_hash, load_base, load_tokenizer, source_hashes,
                        validate_adapter, write_json)
from edit_study import BLUE_BOOK, N, RED_BOOK, edge_list, flip, load_split, prompt, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.split == "test" and (not args.allow_test or args.selection is None):
        parser.error("test requires --allow-test and frozen selection")
    if not 1 <= args.max_steps <= 8 or args.max_seconds < 1 or \
            (args.limit is not None and args.limit < 1):
        parser.error("max steps 1..8 and positive deadline/limit required")
    rows = load_split(args.data, args.split)
    selected = rows[:args.limit] if args.limit else rows
    adapter = validate_adapter(args.adapter, args.data / "train.jsonl")
    if not adapter.get("epoch_complete"):
        parser.error("complete training epoch required")
    if args.split == "test":
        selection = json.loads(args.selection.read_text())
        if selection.get("model_id") != MODEL_ID or \
                selection.get("model_revision") != MODEL_REVISION or \
                selection.get("adapter_sha256") != adapter["adapter_sha256"] or \
                selection.get("dev_dataset_sha256") != file_hash(args.data / "dev.jsonl"):
            raise ValueError("selection mismatch")
    if args.output.exists():
        parser.error("output exists")
    args.output.mkdir(parents=True)
    manifest = {"status": "started", "split": args.split, "rows": len(selected),
                "full_split": len(selected) == len(rows), "max_steps": args.max_steps,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "adapter_sha256": adapter["adapter_sha256"],
                "data_sha256": file_hash(args.data / f"{args.split}.jsonl"),
                "source_sha256": source_hashes(), "rollout_sha256": file_hash(Path(__file__)),
                "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}}
    write_json(args.output / "run_manifest.json", manifest)
    started = time.monotonic()
    try:
        import torch
        from peft import PeftModel
        tokenizer = load_tokenizer()
        model = PeftModel.from_pretrained(load_base(training=False), args.adapter, is_trainable=False)
        model.eval()
        summaries = []
        with (args.output / "trajectories.jsonl").open("w") as stream:
            for row in selected:
                state = [int(word, 16) for word in row["rows_hex"]]
                current = score(state)
                best_loss = current[0]
                visited = {tuple(state)}
                history = []
                stop = "budget"
                for _ in range(args.max_steps):
                    if time.monotonic() - started >= args.max_seconds:
                        raise TimeoutError("rollout deadline")
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt(state, current)}],
                        tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        generated = model.generate(tokens, attention_mask=torch.ones_like(tokens),
                                                   do_sample=False, num_beams=1,
                                                   max_new_tokens=MAX_NEW_TOKENS,
                                                   pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(generated[0][tokens.shape[-1]:], skip_special_tokens=True)
                    action = parse_action(raw)
                    if action is None:
                        history.append({"raw": raw, "action": None, "loss": current[0]})
                        stop = "malformed"
                        break
                    flip(state, *action)
                    current = score(state)
                    best_loss = min(best_loss, current[0])
                    history.append({"raw": raw, "action": list(action), "loss": current[0]})
                    if current[0] == 0:
                        if not verify(N, edge_list(state), RED_BOOK, BLUE_BOOK)["valid"]:
                            raise AssertionError("score and independent checker disagree")
                        stop = "certified"
                        break
                    identity = tuple(state)
                    if identity in visited:
                        stop = "cycle"
                        break
                    visited.add(identity)
                record = {"id": row["id"], "initial_loss": row["loss"],
                          "best_loss": best_loss, "success": best_loss == 0,
                          "verifier_calls": sum(step["action"] is not None for step in history),
                          "stop": stop, "steps": history}
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                summaries.append(record)
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        trajectories_sha256=file_hash(args.output / "trajectories.jsonl"),
                        successes=sum(item["success"] for item in summaries),
                        mean_best_loss=sum(item["best_loss"] for item in summaries) / len(summaries),
                        mean_verifier_calls=sum(item["verifier_calls"] for item in summaries) / len(summaries))
        write_json(args.output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__,
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
