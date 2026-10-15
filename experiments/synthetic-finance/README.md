# Synthetic finance snippet experiment

This experiment trains a 4-bit QLoRA adapter for strict JSON answers to fictional financial snippets. It is an exploratory extraction and arithmetic pilot, not a model of real financial expertise.

The corrected 400-example development result is summarized in [results/SYNTHETIC_FINANCE.md](../../results/SYNTHETIC_FINANCE.md). The tuned model reached 42% exact correctness after one epoch, while a deterministic parser reached 100%. The earlier generator version was invalidated for shortcut leakage and is excluded from the repo's frozen data.

The exact synthetic examples, manifests, source, tests, and training scripts are here. Run the local unit suite from this directory with `PYTHONPATH=src python -m unittest discover -s tests -q`. GPU dependencies and run protocol are in `requirements-gpu.txt` and `docs/MODEL_RUNTIME.md`. The selected adapter and detailed logs remain in the local evidence archive; they are not source files and are not licensed for redistribution here.

The pinned base model is `nvidia/Nemotron-Mini-4B-Instruct` revision `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. See the [model card](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) and its separate license before using or distributing derivatives.
