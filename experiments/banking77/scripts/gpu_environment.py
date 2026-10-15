#!/usr/bin/env python3
"""Check the paid GPU environment; optionally prefetch only the pinned checkpoint."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from banking77.model import MODEL_ID, MODEL_REVISION, sha256_file

EXPECTED = {"torch": "2.6.0", "transformers": "4.51.3", "peft": "0.15.2", "accelerate": "1.6.0", "bitsandbytes": "0.45.5"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch", action="store_true", help="download pinned checkpoint into HF_HOME after all GPU checks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing environment report")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("use the reviewed Python 3.11 environment")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name} version mismatch: expected {expected}")
    import torch
    import bitsandbytes as bnb
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("exactly one visible BF16-capable CUDA GPU is required")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 22 * 1024 ** 3:
        raise RuntimeError("at least 22 GiB reported GPU memory is required")
    cache = os.environ.get("HF_HOME")
    if not cache or not Path(cache).is_absolute():
        raise RuntimeError("set absolute HF_HOME on the persistent volume")
    Path(cache).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if free < 12 * 1024 ** 3:
        raise RuntimeError("need at least 12 GiB free after installing dependencies, before checkpoint prefetch")
    # Exercise the installed quantized CUDA kernel and its input-gradient path.
    layer = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4").cuda()
    inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = layer(inputs)
    output.float().square().mean().backward()
    if not torch.isfinite(output).all().item() or inputs.grad is None or not torch.isfinite(inputs.grad).all().item() or not inputs.grad.abs().sum().item():
        raise RuntimeError("NF4 CUDA forward/backward failed")
    del layer, inputs, output
    torch.cuda.empty_cache()
    report = {"status": "environment_passed", "versions": versions, "python": sys.version.split()[0],
              "cuda": torch.version.cuda, "gpu": device.name, "vram_bytes": device.total_memory,
              "free_disk_bytes_before_prefetch": free, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "full_model_smoke_passed": False, "prefetched": False}
    if args.prefetch:
        from huggingface_hub import snapshot_download
        location = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                     allow_patterns=["*.json", "pytorch_model.bin", "tokenizer.model", "*.txt", "*.pdf"],
                                     max_workers=4)
        weights = Path(location) / "pytorch_model.bin"
        if not weights.is_file() or weights.stat().st_size != 8381129342:
            raise RuntimeError("pinned checkpoint weight file missing or wrong size")
        weight_sha = sha256_file(weights)
        if weight_sha != "e851a12d65d56e90ed99659e7439f2b5fc080db478500ee3390f809131426854":
            raise RuntimeError("pinned checkpoint weight SHA256 mismatch")
        report.update(prefetched=True, checkpoint_sha256=weight_sha, checkpoint_cache=location, free_disk_bytes_after_prefetch=shutil.disk_usage(cache).free)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
