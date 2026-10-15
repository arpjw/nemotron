# A small model, a narrow task, and a useful comparison

**Unpublished working draft. Official-test outcome pending.** All BANKING77 numbers below are development results. Do not publish this version as the final experiment report. Final billing is also pending; account credit and spending limits are not experiment costs.

I started with a practical question: what can a small post-training experiment actually buy?

The model was NVIDIA's Nemotron-Mini-4B-Instruct. The researcher reported $100 of Runpod credit, but that was room to experiment, not a target to spend. The work used bounded GPU runs, saved artifacts, and explicit stopping points.

The first task exposed a limitation. The second produced a substantial improvement over the untouched model on a specific classification benchmark. A simple CPU classifier also did surprisingly well.

## First, the model learned the format

The initial task used synthetic financial documents. Questions asked the model to extract a value, calculate a difference, or calculate growth, returning a strict JSON answer with supporting evidence. Some questions lacked enough information and required abstention.

A data audit caught leakage in the first generator, including an entity identifier tied to row position. That version's results were excluded. After the generator was corrected and the data regenerated, a 100-step adapter answered 147 of 400 development questions exactly. A full 300-step epoch reached 168 of 400, or 42%.

The full-epoch model produced valid JSON on every example. It still got the numeric value right on only 105 of the 320 answerable questions. More training increased the exact-answer count by 21, but the answerable-value count increased by only two.

That distinction mattered. A model that consistently returns a well-formed answer can still be unreliable at the work inside the answer.

We separately tested deterministic calculation. A parser operating on the public document and question answered all 400 development examples correctly. A system combining the earlier model's evidence choices with a calculator reached 351 of 400. Those are tool-assisted or parser results, not arithmetic performed by the model itself.

The perfect parser score also revealed how limited the synthetic benchmark was. It was useful for testing formats, evidence handling, and failure cases. It could not establish broad financial reasoning. We did not run its sealed test or robustness split.

## A task with a clearer boundary

The next experiment used [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), a dataset of banking-support queries labeled with 77 intents. Examples distinguish requests about activating a card, a pending transfer, an unrecognized withdrawal, or a failed top-up. The task and dataset are described in the [original paper](https://aclanthology.org/2020.nlp4convai-1.5/).

The question became concrete: could a small adapter make Nemotron better at assigning these exact labels?

We formed a development split from the publisher's training data, grouped normalized duplicates, removed repeated examples, and quarantined a conflicting-label group. Training used at most 30 examples per intent: 2,308 examples in total. Development contained 1,993 examples. The official test was reserved for evaluating the selected checkpoint afterward.

The model trained for one epoch using QLoRA, which keeps the base model quantized and trains a comparatively small set of adapter weights. The run used rank-16 adapters, a 1,024-token context cap, microbatch size one, and gradient accumulation of eight. Training completed 289 optimizer steps in about 19.5 minutes on one RTX 4090. Peak reserved GPU memory was about 12.6 GB in decimal units.

Those measurements describe the training stage. Downloading the checkpoint, checking the environment, running both models, exporting artifacts, and supervising the resource are additional parts of the experiment.

## Make the baseline comparison fair

Both the untouched model and the adapter received the same prompt and full label taxonomy. For the primary comparison, the decoder constrained both models to valid taxonomy labels. A separate unconstrained generation recorded formatting behavior.

This prevents a misleading result in which the tuned model appears better mainly because it learned the required output format. The primary decoder is constrained greedy generation, not exhaustive ranking of all 77 label probabilities.

We chose macro-F1 as the primary metric so that each intent receives equal weight. Before GPU training and evaluation, the advancement criterion required at least 0.75 macro-F1, an improvement of at least 0.05 over untouched Nemotron, and a positive lower bound on the paired macro-F1 improvement interval.

## What changed on development

| Development metric | Untouched Nemotron | One-epoch adapter |
|---|---:|---:|
| Constrained macro-F1 | 0.3991 | 0.8298 |
| Constrained accuracy | 40.79% | 83.04% |
| Correct constrained predictions | 813/1,993 | 1,655/1,993 |
| Malformed unconstrained outputs | 271 | 2 |

The macro-F1 gain was 0.4306. A paired bootstrap over query groups gave a 95% interval of [0.4100, 0.4546]. This measures sampling uncertainty on the development split; it does not measure variation across training seeds or new domains.

The adapter corrected 885 baseline errors and introduced 43 errors on examples the baseline had answered correctly. Another 295 examples remained wrong for both. All 77 classes improved in F1, though three lost some recall.

Some improvements were easy to interpret. “What is the procedure for activating this card?” moved from `card_acceptance` to the correct `activate_my_card`. “Can you tell me how many currencies I can hold?” moved from `supported_cards_and_currencies` to `fiat_currency_support`.

Other cases remained difficult. “The top-up did not go through” moved from the correct `top_up_failed` to `pending_top_up`. Confusing a failed transaction with a pending one could matter in a deployed support system. High average accuracy does not erase that distinction.

The most frequent remaining errors involved neighboring categories: bank-transfer top-up charges versus transfer fees, unrecognized cash withdrawals versus pending withdrawals, and reverted top-ups versus failed top-ups.

These examples were selected after inspecting development outcomes. They illustrate behavior; they are not an independent test.

## The CPU baseline deserves a place in the table

A word-and-character TF-IDF classifier with logistic regression, trained on the same 2,308 examples, reached **0.8285 macro-F1 and 82.74% accuracy** on development. The adapter reached **0.8298 and 83.04%**.

That small numerical difference is not evidence of a meaningful advantage over the simpler classifier. The experiment's objective was to improve Nemotron relative to its untouched checkpoint, and the development evidence supports that objective. For a product that only needs this label assignment, the simpler classifier remains a serious option. This experiment did not establish an operational cost or latency advantage for either system.

## What this result does and does not establish

The adapter became substantially better at matching BANKING77's intent labels on the held-out development split. The comparison used identical constrained decoding, so output formatting alone does not explain the gain.

The result does not establish a general finance model. We did not test investment analysis, accounting judgment, mathematical reasoning on unfamiliar documents, calibrated uncertainty, or rejection of requests outside the taxonomy. A constrained classifier is required to pick a label even when none is appropriate.

BANKING77 has been public since 2020. We cannot verify that it was absent from the base model's pretraining. Exact duplicate handling also does not eliminate semantic overlap. Some queries have ambiguous category boundaries, and matching the benchmark label is not always equivalent to a uniquely correct interpretation.

This was one seed and one bounded epoch. Development supported checkpoint selection; it was not the final confirmatory evaluation.

## Official test: pending

**Editor placeholder:** Add the completed paired official-test result only after both model runs, prediction integrity checks, selection-receipt checks, and overlap audit are verified. Report the unchanged full official denominator and the separately labeled clean-subset sensitivity analysis. Include macro-F1, accuracy, paired macro-F1 interval, and malformed-output counts. Do not select or tune a checkpoint after viewing official results.

**Editor placeholder:** Add verified resource teardown and distinguish final provider billing from runtime-based cost estimates. Do not use the $100 account balance or $10 reservation as a reported experiment cost.

For now, the useful lesson is about scope. A few thousand examples and a short adapter run changed a small model's behavior substantially on a narrow language task. The earlier synthetic experiment showed why that improvement has to be measured at the task level: correct JSON, correct evidence, correct arithmetic, and correct intent classification are different capabilities.

---

## Internal evidence map, remove or convert before publication

- Synthetic corrected runs and calculator results: `results/SYNTHETIC_FINANCE.md`; full predictions and calculator report remain in the local evidence archive. Exploratory only; initial leaked results excluded.
- BANKING77 design and limits: `experiments/banking77/docs/PROTOCOL.md` and `experiments/banking77/docs/GATE_V2.json`.
- Verified completed training: local evidence archive `epoch-adapter/run_manifest.json`, `training_log.jsonl`, and saved adapter/config checksums. Binary adapters are excluded from Git.
- Primary development results and intervals: local evidence archive `dev-comparison.json`; reviewed analysis in `results/BANKING77_DEV_ERROR_ANALYSIS.md`.
- CPU comparator: local evidence archive `results/cpu-baselines/report.json`.
- Model pinned revision: `nvidia/Nemotron-Mini-4B-Instruct`, `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`.
- BANKING77 dataset attribution: PolyAI, CC BY 4.0; preserve `experiments/banking77/DATA_ATTRIBUTION.md` and `DATA_LICENSE` in any associated release. Dataset license is separate from model licensing.
- No official-test outcome was read or incorporated into this draft. It remains unpublished; the private source repository is a working archive, not a public result.
