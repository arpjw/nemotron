# BANKING77 model runtime

This experiment compares a one-epoch QLoRA Nemotron Mini adapter against the identical untouched checkpoint on 77 intent labels. The primary metric is macro-F1 from taxonomy-constrained greedy decoding. Raw greedy generations are saved separately to report malformed-output rate and raw accuracy; formatting improvements are not relabeled as primary classification gains. TF-IDF baselines provide context only.

All canonical publisher labels, including capitalization and punctuation, are preserved. A token trie restricts the primary decoder to a complete canonical label followed by EOS. Both model conditions use the same tokenizer, system prompt, full-taxonomy prompt, quantization, greedy settings and 32-token output cap. This is a constrained-generation classifier, not an exhaustive label-likelihood ranking model.

## Local validation

Before official-test release, the real pinned tokenizer was checked on all 2,308 training and 1,993 development rows. Maximum full training sequence is736 tokens; maximum prompt plus output allowance is760, below1024. The longest canonical label including EOS is17 tokens. No official test text was parsed at that audit stage. See TOKENIZER_AUDIT.json for data hashes.

The optional real tiny-model CPU integration verifies native Nemotron response-only loss, nonzero adapter gradients, save/reload equality and taxonomy-trie generation. It downloads no pretrained weights. The other tests check publisher taxonomy preservation, overflow rejection, one-epoch coverage and final accumulation-window scaling, strict format diagnostics, and distinct paired accuracy/macro-F1 statistics.

## Fresh-pod supervised stages

Use a fresh supervised resource with an operator-maintained dollar cap, resource review deadline and deletion/export plan. The pilot did not have a provider-enforced spending reservation. Training and inference timeouts are process checks, not billing controls. An outer process-group timeout remains necessary for hung CUDA calls or interpreter shutdown. There is no automatic provisioning or combined deadline controller in this package.

Use Python3.11 with the pinned requirements-gpu.txt. Install PyTorch2.6.0 from the CUDA12.4 wheel index into a fresh venv before the other requirements, then check dependencies. The adapted scripts/gpu_environment.py verifies the one-GPU environment and NF4 kernel, optionally prefetching the immutable model .bin with size/SHA256 checks. Keep HF_HOME on the pod's ephemeral disk and export all needed artifacts before deletion. Model loading is local_files_only=True to prevent Transformers' background safetensors conversion thread. Set HF_HUB_OFFLINE=1 after prefetch.

From the project root, with PYTHONPATH=src and the reviewed venv interpreter:

```bash
python scripts/gpu_environment.py --prefetch --output environment.json
python -m banking77.train --data data/frozen/train.jsonl --output runs/smoke-adapter --smoke-steps 2 --gradient-accumulation 1 --max-seconds 300
python -m banking77.inference --data data/frozen/dev.jsonl --output runs/base-smoke --limit 20 --max-seconds 300
python -m banking77.inference --data data/frozen/dev.jsonl --output runs/adapter-smoke --adapter runs/smoke-adapter --limit 20 --max-seconds 300
```

Inspect measured stage latency, GPU memory and disk before authorizing the full sequence within the original resource deadline. Two decodes per row across1,993 dev rows can dominate runtime. Subset smoke outputs cannot enter a formal comparison. After timing review, the full stages are:

```bash
python -m banking77.inference --data data/frozen/dev.jsonl --output runs/base-dev --max-seconds 2400
python -m banking77.train --data data/frozen/train.jsonl --output runs/epoch-adapter --max-seconds 2400
python -m banking77.inference --data data/frozen/dev.jsonl --output runs/adapter-dev --adapter runs/epoch-adapter --max-seconds 2400
python -m banking77.evaluation --data data/frozen/dev.jsonl --predictions runs/adapter-dev/predictions.jsonl --baseline runs/base-dev/predictions.jsonl --output runs/dev-comparison.json
```

The example individual ceilings are not additive authorization: their sum can exceed a two-hour resource limit. The supervisor must allocate a finite remaining budget before starting each stage and stop in time to export and delete. Never run all stages blindly. One epoch consumes all2,308 examples exactly once:289 optimizer steps with accumulation8, including a correctly weighted final window of4. Smoke runs are marked epoch_complete=false and are rejected as formal full-epoch candidates.

## Provenance and decision

Training manifests bind the frozen training/taxonomy/data manifest, source code files, runtime versions, actual model config, seed, steps, seen rows, memory and adapter/config checksums. Inference records matching source/runtime provenance and prediction-file hashes. Comparisons reject incomplete or subset runs and unequal base/adapter decoder code or package versions. SIGTERM attempts to mark active runs failed; a hard kill can leave status started, which is also rejected.

The prospective development gate in GATE_V2.json requires at least0.75 constrained macro-F1 and at least0.05 absolute macro-F1 improvement over untouched Nemotron with the paired group-bootstrap95% lower bound above0. Accuracy intervals are separately named. The official test stayed sealed until the completed development comparison passed and the checkpoint selection was recorded. Release used the separate gated sealed_test exporter and an explicit selected-checkpoint receipt. Passing development is single-seed evidence on this task, not broad financial expertise or a guarantee of test improvement.


## Selected official-test comparison

Only after the development gate passes and checkpoint selection is recorded, the supervised sealed_test exporter may produce `data/sealed-official-test`. Both official inference conditions must use the same final inference.py/model.py/runtime. Complete the development comparison before uploading these sealed-capable source revisions, because development provenance binds the earlier decoder source hashes.

```bash
python -m banking77.inference --data data/sealed-official-test/test.jsonl --frozen-data data/frozen --allow-sealed --output runs/base-test --max-seconds 3600
python -m banking77.inference --data data/sealed-official-test/test.jsonl --frozen-data data/frozen --allow-sealed --adapter runs/epoch-adapter --output runs/adapter-test --max-seconds 3600
python -m banking77.evaluation --data data/sealed-official-test/test.jsonl --allow-sealed --predictions runs/adapter-test/predictions.jsonl --baseline runs/base-test/predictions.jsonl --output runs/test-comparison.json
```

These commands are examples, not authorization or sufficient resource time. The supervisor must check remaining deadline and export reserve first. Official inference rejects subsets, verifies the original frozen training manifest and selected adapter, and binds the chosen checkpoint, prompt and decoder to selection.json. An exclusive receipt permits one run per model condition; failures retain the receipt for explicit review. Preserve these receipts with the exported results. The report includes every official row and a separately named clean-subset sensitivity comparison from the frozen overlap/conflict audit. Neither permits checkpoint reselection. The local guard tests use synthetic bytes only.
