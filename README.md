# Nemotron small-model post-training experiments

Independent research on narrow post-training tasks using the pinned `nvidia/Nemotron-Mini-4B-Instruct` checkpoint. This repository is private while the results, redistribution rights, and publication copy are reviewed. It is not an NVIDIA project or a general finance assistant.

## Experiments

| Experiment | Task | Current evidence |
|---|---|---|
| [Synthetic finance snippets](experiments/synthetic-finance/) | Extract and calculate values from fictional snippets | One-epoch adapter reached 42% exact correctness on 400 development examples, from 0% for the untouched model. Gains were concentrated in abstention and formatting; numeric reasoning remained weak. A deterministic parser solved all 400, so this is not a useful finance-expertise claim. |
| [BANKING77 intents](experiments/banking77/) | Classify a banking-support query into one of 77 publisher labels | One-epoch QLoRA on 2,308 capped training examples. On 1,993 development examples, constrained macro-F1 was 0.82978 versus 0.39915 for untouched Nemotron. The selected checkpoint passed the prospective development gate. The publisher's 3,080-row official test is being evaluated once. |

The BANKING77 development comparison uses the same prompt, model revision, and taxonomy-constrained decoder in both conditions. A paired complete-group bootstrap estimated a macro-F1 gain of 0.43063 with a 95% interval of [0.41004, 0.45460]. This interval describes the fixed development split and does not capture training-seed or domain uncertainty. A TF-IDF logistic comparator scored 0.82852 macro-F1 on the same development data; the goal here is improvement over untouched Nemotron, not superiority over that smaller classifier.

## Layout

- `experiments/synthetic-finance/`: original synthetic task, code, frozen examples, tests, and protocol.
- `experiments/banking77/`: publisher-backed intent task, code, frozen train/development examples, tests, prospective gate, and attribution.
- `results/`: reviewed aggregate findings and checksum references. Model weights, full runtime logs, and official-test references stay in the local evidence archive.
- `writing/`: unpublished article draft.
- [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md): remaining release decisions and verification.

## Reproduction boundaries

The pinned Nemotron model revision is `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. GPU dependency versions and exact commands are documented in each experiment's `docs/MODEL_RUNTIME.md`. The full local suites use Python 3.11 with `requirements-validation.txt`; run `PYTHONPATH=src python -m unittest discover -s tests -q` from each experiment directory. The pull-request workflow runs smaller standard-library protocol suites. BANKING77 frozen train/development records are included under the publisher's CC BY 4.0 terms; the raw publisher test and released official references are intentionally outside the source bundle. A fresh acquisition must verify the publisher commit and hashes in `experiments/banking77/data/raw/source_manifest.json`.

## Scope and rights

The observed results describe one training seed on a historical benchmark. We cannot verify that BANKING77 was absent from Nemotron pretraining. Neither experiment establishes investment skill, production banking reliability, calibrated confidence, or out-of-taxonomy handling.

The BANKING77 data carry [CC BY 4.0 attribution](experiments/banking77/DATA_ATTRIBUTION.md) and a separate [dataset license](experiments/banking77/DATA_LICENSE). Nemotron weights and adapters are governed separately by the pinned checkpoint's [NVIDIA model license](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct). No adapter weights or tokenizer copies are in this source repository. A code license has not been selected; [public release](PUBLIC_RELEASE_CHECKLIST.md) requires that decision and a rights review.
