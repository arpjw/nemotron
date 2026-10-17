# Reproducing the graph-repair study

Use a CUDA GPU with at least 24 GB VRAM and a Python environment containing
Torch 2.8.0+cu128, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0,
bitsandbytes 0.45.5, safetensors 0.5.3, tokenizers 0.21.1, and
huggingface-hub 0.30.2. The measured run used Python 3.12.3 and an RTX 4090.
The pinned Nemotron revision must be cached before the training and inference
scripts, which load model files locally. For example:

```sh
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("nvidia/Nemotron-Mini-4B-Instruct", revision="791833e92ebddb0bc2c1007f6d2b6764f886a2ae")'
```

Do not publish the downloaded base,
adapter, tokenizer, or generated data without a separate rights review.

The ten external graph6 records are available from
`https://users.cecs.anu.edu.au/~bdm/data/sr261034.g6` and are checksum
verified by the generator. Run from this directory:

```sh
python3 edit_study.py --source sr261034.g6 --output data/edit-v1
python3 -m unittest test_edit_study.py -q
python3 edit_baseline.py --data data/edit-v1 --split dev --output runs/edit-dev-cpu.json
python3 edit_train.py --data data/edit-v1 --output runs/unused --preflight-only
python3 edit_train.py --data data/edit-v1 --output runs/adapter-seed20261016 --seed 20261016
python3 edit_infer.py --data data/edit-v1 --split dev --output runs/base-dev
python3 edit_infer.py --data data/edit-v1 --split dev --adapter runs/adapter-seed20261016 --output runs/adapter-dev
python3 edit_evaluate.py --data data/edit-v1 --split dev --base runs/base-dev --adapter runs/adapter-dev --output runs/dev-report.json --selection-output runs/selection.json
```

The selection command fails unless the adapter exceeds the frozen development
gate. Only then run each test inference once:

```sh
python3 edit_infer.py --data data/edit-v1 --split test --allow-test --selection runs/selection.json --output runs/base-test
python3 edit_infer.py --data data/edit-v1 --split test --allow-test --selection runs/selection.json --adapter runs/adapter-seed20261016 --output runs/adapter-test
python3 edit_evaluate.py --data data/edit-v1 --split test --base runs/base-test --adapter runs/adapter-test --output runs/test-report.json
```

The secondary, exploratory rollout is described in `EDIT_ROLLOUT_PROTOCOL.md`.
It was run on development first, then test after the development repair gate:

```sh
python3 edit_rollout.py --data data/edit-v1 --split dev --adapter runs/adapter-seed20261016 --output runs/rollout-dev
python3 edit_rollout.py --data data/edit-v1 --split test --allow-test --selection runs/selection.json --adapter runs/adapter-seed20261016 --output runs/rollout-test
```

Paths under `runs/` are local examples, not a request to rerun the historical
held-out comparison. Preserve the frozen data and receipts. A new study should
use versioned splits and a fresh held-out source.
