# Model runtime and validation

The training runtime pins `nvidia/Nemotron-Mini-4B-Instruct` revision `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. It uses native Transformers support without remote model code. The requirements list pins the main runtime packages, not every transitive dependency; `pip check` and the GPU smoke test remain required in the selected container. Use Python 3.11 and the PyTorch 2.6.0 CUDA 12.4 wheel. CUDA driver compatibility must be checked on the actual GPU host.

## Training behavior

`python -m finmini.train --data data/frozen/train.jsonl --output runs/pilot-adapter --max-steps 100`

Run with `PYTHONPATH=src` or an installed package. The output directory must be new. The script verifies the entire frozen dataset manifest, hashes, and generator/scorer sources before model use; it accepts only the manifest-listed training file and records marked `train`, rejects duplicate IDs, uses only prompt and target for tokenization, and validates the strict target schema. The native prompt template includes a fixed system instruction. Prompt labels are -100; every response token and the terminating EOS are supervised. Examples exceeding 1,024 tokens fail explicitly. There is no silent truncation or cross-example packing.

QLoRA uses NF4, double quantization, BF16 compute, rank 16, alpha 32, dropout 0.05, microbatch one, and eight accumulation steps. All six linear projection families in the native Nemotron attention/MLP blocks are targeted, with embeddings/output head frozen. Only adapter parameters are optimized. The runtime rejects missing target modules, unexpected trainable base weights, nonfinite losses/gradients and zero adapter gradients. An explicit step cap and process wall-time checks bound work; an external watchdog must terminate a hung process and a separate provider controller must terminate the pod.

Successful runs save safetensors adapters and a manifest recording model revision, data SHA256, runtime versions, seed, actual steps, GPU, CUDA, peak memory, and adapter checksum. Failed runs are marked failed. The inference loader rejects incomplete adapters or provenance/checksum mismatches.

## Evaluation hooks

`python -m finmini.inference --data data/frozen/dev.jsonl --output runs/base-dev`

`python -m finmini.inference --data data/frozen/dev.jsonl --output runs/adapter-dev --adapter runs/pilot-adapter`

Both use the same pinned tokenizer, system prompt, quantization, greedy decoding, and 192-token output cap. Inference receives only each prompt; it never passes target/reference metadata into the model. It writes `{id,prediction}` JSONL plus a completion/provenance manifest. Score the predictions with the independent scorer. Do not use incomplete inference manifests as completed experiments. The final test set belongs to the frozen evaluation stage, not repeated development. Inference on test or robustness requires the explicit `--allow-sealed` flag; renamed split files are rejected. Completed manifests bind prediction file SHA256 and the system prompt.

## Completed local evidence

2026-10-15: ten model tests passed in a disposable Python 3.11.15 CPU environment using torch 2.6.0, Transformers 4.51.3 and PEFT 0.15.2. Nine tests exercise masking, exact token boundaries, invalid schema/splits/duplicates, adapter integrity, and deterministic continuation-only generation using mocks, plus CLI subset provenance and label isolation. One real tiny native Nemotron test verifies shifted response-only loss against manual cross entropy, nonzero adapter gradients, an optimizer step, safetensors save/reload, and identical restored logits. No pretrained weights are used by the tiny test.

The actual pinned tokenizer was downloaded separately and successfully encoded a short financial extraction example: 50 prompt tokens, 33 supervised tokens including EOS, 83 total. A complete tokenizer-only audit of all 3,600 frozen examples also passed: the longest full sequence is 498 tokens and the largest prompt plus 192-token generation budget is 671. Per-split hashes and lengths are in `TOKENIZER_AUDIT.json`. No model predictions or quality evaluation were performed on sealed sets. No full checkpoint weights were downloaded. Missing optional dependencies skip the tiny CPU integration test explicitly; they do not count as passing it.

## Outstanding GPU evidence

Bitsandbytes import/quantization, CUDA kernels and BF16, full checkpoint loading, real gradient flow, VRAM fit, throughput, full adapter save/reload, and task improvement remain unverified. The small CPU model validates implementation plumbing but cannot establish these properties. The proposed first GPU smoke is two optimizer steps followed by adapter reload and generation, under the external budget/watchdog controls. No scientific result is implied by passing software tests.

Primary sources: [native model implementation](https://github.com/huggingface/transformers/tree/v4.51.3/src/transformers/models/nemotron), [pinned checkpoint](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct/tree/791833e92ebddb0bc2c1007f6d2b6764f886a2ae), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
