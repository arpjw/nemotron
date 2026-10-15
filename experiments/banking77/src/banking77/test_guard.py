"""Selected-checkpoint and one-run guards for released official test artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from .model import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, MAX_NEW_TOKENS, sha256_file


def validate_test_binding(selection: dict, metadata: dict, *, adapter: bool) -> None:
    expected = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "modes": ["taxonomy_trie", "raw"]}}
    for key, value in expected.items():
        if selection.get(key) != value or metadata.get(key) != value:
            raise ValueError(f"selected official-test protocol mismatch: {key}")
    wanted = selection["adapter_sha256"] if adapter else None
    if metadata.get("adapter_sha256") != wanted:
        raise ValueError("official test requires the selected adapter or untouched base")
    if adapter and metadata.get("adapter_epoch_complete") is not True:
        raise ValueError("official-test selected adapter must complete its epoch")


def validate_frozen_training(selection: dict, frozen: Path) -> None:
    from .data import verify_dataset
    verify_dataset(frozen)
    if sha256_file(frozen / "manifest.json") != selection["data_manifest_sha256"]:
        raise ValueError("test selection references a different frozen training experiment")


def reserve_test_inference(directory: Path, output: Path, *, adapter: bool, metadata: dict) -> Path:
    """Keep reservations after failure; another inference requires explicit review."""
    name = "selected-adapter" if adapter else "untouched-base"
    receipt = directory / f"inference-{name}.receipt.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"status": "reserved", "role": name, "output": str(output.resolve()),
                   "selection_sha256": sha256_file(directory / "selection.json"),
                   "dataset_sha256": sha256_file(directory / "test.jsonl"),
                   "adapter_sha256": metadata.get("adapter_sha256")}, stream, indent=2)
        stream.write("\n")
    return receipt


def clean_test_rows(rows: list[dict], audit: dict) -> list[dict]:
    ids = {row["id"] for row in rows}
    clean, quarantined = audit["clean_ids"], audit["quarantined_ids"]
    if (len(set(clean)) != len(clean) or len(set(quarantined)) != len(quarantined)
            or set(clean) & set(quarantined) or set(clean) | set(quarantined) != ids):
        raise ValueError("invalid clean-subset audit partition")
    return [row for row in rows if row["id"] in set(clean)]


def validate_test_receipt(directory: Path, metadata: dict) -> None:
    role = "selected-adapter" if metadata.get("adapter_sha256") else "untouched-base"
    name = f"inference-{role}.receipt.json"
    if metadata.get("test_inference_receipt") != name:
        raise ValueError("official inference receipt role mismatch")
    path = directory / name
    if sha256_file(path) != metadata.get("test_inference_receipt_sha256"):
        raise ValueError("official inference receipt hash mismatch")
    receipt = json.loads(path.read_text())
    expected = {"status": "reserved", "role": role,
                "selection_sha256": metadata["selection_sha256"],
                "dataset_sha256": metadata["dataset_sha256"],
                "adapter_sha256": metadata.get("adapter_sha256")}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("official inference receipt binding mismatch")
    # Original absolute output is provenance only: exported results may relocate.
    if not isinstance(receipt.get("output"), str) or not Path(receipt["output"]).is_absolute():
        raise ValueError("official inference receipt output is invalid")
