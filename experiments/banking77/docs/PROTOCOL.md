# BANKING77 bounded adaptation protocol

## Task and evidence

Route a short banking-support query to one of 77 publisher intent labels. This is banking language understanding, not investment analysis or real-world banking deployment. The publisher reports 10,003 training queries and 3,080 test queries. The repository explicitly licenses its datasets under CC BY 4.0, and the official card agrees. Dataset origin and annotation details in the card are incomplete; do not claim verified production-customer provenance.

Primary sources, checked October 15, 2026:

- [Publisher dataset repository](https://github.com/PolyAI-LDN/task-specific-datasets)
- [Official PolyAI dataset card](https://huggingface.co/datasets/PolyAI/banking77/blob/main/README.md)
- [Publisher license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE)
- [Original paper](https://aclanthology.org/2020.nlp4convai-1.5/)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)

CC BY permits redistribution and adaptation, including commercial use, subject to attribution and other conditions. Preserve `DATA_LICENSE` and `DATA_ATTRIBUTION.md`. Dataset licensing does not determine the separate model's license.

## Immutable acquisition

Pinned publisher commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`. Every acquired file matched the publisher Git blob SHA-1 independently obtained from GitHub tree metadata. Git blob SHA-1 includes a Git object header and is distinct from the raw-byte SHA-256 below. Exact source URLs, lengths and both hashes are in `data/raw/source_manifest.json`.

| File | Raw SHA-256 |
|---|---|
| train.csv | b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b |
| test.csv | d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d |
| categories.json | 53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63 |
| LICENSE | 7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661 |

The official test CSV was downloaded and hashed as opaque bytes. It has not been parsed, displayed, transformed into prompts or evaluated. The publisher's 3,080 count is source-reported, not independently counted by opening this sealed file.

## Frozen development design

Only official training rows were parsed. Normalize by Unicode NFKC, casefold, and join word tokens; hash normalized text as group identity. Keep one lexically deterministic representative per label-consistent group, quarantine groups with conflicting labels, then partition groups separately per intent. This removes 31 duplicate rows and quarantines one conflicting-label group. No group crosses train/development.

Seed `20261015`. For each intent, sort groups by SHA-256 of seed, label and group ID. The first nearest-integer 20% become development. From the remainder choose at most 30 groups for training. This yields **2,308 training and 1,993 development examples**: 76 intents have 30 training examples; `contactless_not_working` has 28. Other unused training groups are excluded from this first experiment. All 77 intents appear in both used splits. Split ordering is independently hashed rather than label-sorted. IDs, group hashes and gold labels are not inserted into prompts.

Training reads only the generated public prompt and its label target. Inference receives only the prompt. The prompt contains the same full taxonomy for base and adapter and requests one exact publisher label. Preserve publisher case and punctuation, including `Refund_not_showing_up` and `reverted_card_payment?`.

`verify_dataset` checks frozen file hashes, source-independent group identity, disjoint groups, label/target consistency and exact prompt reconstruction. It rejects changed preparation code. `load_frozen_records` exposes train/development only and rejects aliases and test paths. The immutable manifest is a reproducibility record, not protection against deliberate replacement of both data and manifest.

## Baselines and measurements

Before GPU adaptation, fit two locked CPU baselines on exactly the capped 2,308 training examples:

1. Majority label, with publisher taxonomy order breaking count ties.
2. Union of word TF-IDF (1–2 grams) and character-within-word TF-IDF (3–5 grams), each min_df=2, at most 50,000 features, sublinear_tf=True, followed by logistic regression C=4, lbfgs, max_iter=1000. Vocabulary and IDF are fitted exclusively on training text. No development hyperparameter search. Nonconvergence fails loudly.

Also evaluate the untouched Nemotron checkpoint with exactly the same taxonomy prompt and decoding as the adapter. Primary model predictions use a taxonomy token trie to constrain greedy decoding to an exact publisher label plus EOS, with a 32-token limit. This is constrained greedy generation, not exhaustive sequence-probability ranking of all labels. A separate raw greedy prediction records unconstrained format behavior and valid-label rate. Compare primary constrained macro-F1 for the advancement gate; do not describe an improvement driven only by raw formatting as semantic understanding.

Primary metric: macro-F1 over all 77 fixed labels. Secondary metrics: accuracy, per-intent precision/recall/F1, confusion matrix and valid-label rate. Missing or malformed model predictions fail; they must not be dropped from the denominator. Save IDs, raw predictions, immutable training/development/taxonomy hashes, model revision and adapter hash, configuration, library versions and code hashes. Evaluate base and adapter with identical prompts and decoding. Paired intervals should resample complete query groups; one training seed does not establish robustness across seeds.

## Prospective gate and spending

Protocol amendment 2, October 15, 2026, made before BANKING77 GPU training or model evaluation at the user's direction: the primary objective is improvement over untouched Nemotron. This supersedes the original .85 / +.05 over base / +.02 over TF-IDF gate preserved in the immutable data manifest. The unchanged data manifest remains a record of data preparation; `docs/GATE_V2.json` records the active gate and binds it to that manifest.

The advancement gate requires development constrained macro-F1 at least **0.75**, an absolute macro-F1 gain of at least **+0.05 over untouched Nemotron**, and a **paired group bootstrap 95% lower bound above zero for that macro-F1 gain**. All three must hold. Use 2,000 paired group resamples, seed 871, and the fixed full 77-label taxonomy in every resample; do not substitute an accuracy interval. These are prospective design decisions, not a forecast. TF-IDF remains a contextual efficiency comparator and is not a success requirement. Report its score even if it exceeds the tuned model.

Begin with one bounded QLoRA epoch on the capped training subset. Permit at most one additional prespecified development-selected run. Use a $10 stage reservation inside the remaining overall account budget, with supervised time/cost enforcement and export/teardown checks owned by the runner. No paid resource was created by this data/baseline implementation. Do not infer a fresh $100 allocation from this protocol.

Select the checkpoint using development results only, then freeze checkpoint, prompt, decoding and evaluation code. Evaluate untouched Nemotron and that single selected adapter once on the official test, using the same macro-F1 gain, absolute-quality and paired-interval criteria for the final claim. Do not tune after test results or select between checkpoints on test. Before opening test, implement an explicit sealed-test loader and automated normalized train/test-overlap audit; preserve the official test set unchanged and report overlap counts plus a separately labeled clean-subset sensitivity result. The current loader still rejects test paths. Development intervals support selection only; the untouched test comparison provides the final benchmark evidence, subject to contamination caveats.

## Limitations and leakage risks

BANKING77 has been public since 2020. We cannot verify its exclusion from Nemotron pretraining, so report historical benchmark performance rather than contamination-free generalization. Exact normalized duplicate handling is implemented; semantic paraphrase grouping and official train/test overlap auditing are not yet implemented. Before a final claim, add an automated integrity audit that does not expose test labels or model outcomes, and report its findings.

Fine-grained category boundaries can be ambiguous. Do not silently relabel difficult development examples. A future independently authored, independently reviewed paraphrase/negation challenge can probe transfer, but it must be created without inspecting official test content and should be reported separately. A simple pattern parser is unlikely to cover arbitrary query wording, but that is a hypothesis; the TF-IDF comparator tests whether a simpler learned classifier already suffices.

## Optional independently authored challenge

Before any challenge predictions, a human author may write 40 new queries covering 10 intents selected in advance from the taxonomy, four per intent: ordinary paraphrase, negation or correction, distractor detail, and informal wording. A second human reviewer independently assigns the exact publisher label using taxonomy definitions and permitted training examples. Neither author nor reviewer may use official test examples, model-generated queries or model-generated labels. Resolve disagreement before evaluation; record author/reviewer attestations and exclusions, then freeze text, labels and hashes. Group variants from one scenario together for uncertainty estimates.

This challenge is optional and currently uncreated. It must not delay the bounded GPU pilot or be silently replaced with agent-authored gold labels. Evaluate the frozen base and selected adapter once, report accuracy and macro-F1 over the ten prespecified intents separately, and list its small sample size and author subjectivity. It is a descriptive transfer check, not a replacement for the 77-intent official benchmark or an additional model-selection set.

## Implemented one-time official-test release

`python -m banking77.sealed_test` requires explicit `--release-official-test` plus `--frozen`, `--raw`, `--report`, `--selection`, `--output` and `--gate` paths. It validates the completed development comparison, recomputes the advancement decision, verifies full development provenance, and checks the selection receipt before reading any test bytes. The receipt must contain exactly the original development report SHA-256, frozen manifest SHA-256, selected adapter SHA-256, system prompt, generation settings, model ID and revision. These fields must match the passed comparison. This module does not create a passing report or select a checkpoint.

The exporter verifies the official test's pinned raw SHA-256 and Git blob SHA-1 before CSV decoding. It preserves all 3,080 official rows and exact publisher labels, checks normalized-text overlap against used frozen train/development, and records duplicate and conflicting-label groups. Overlap and conflict rows are quarantined from a separate clean-subset sensitivity analysis; the official benchmark denominator remains unchanged. Unused publisher training rows are outside this overlap audit. Gold labels appear only in reference files, never inference prompt construction.

A persistent exclusive reservation beside the raw file prevents a second export, including silent retries after failure. The complete export directory is published by a same-filesystem atomic rename after all files and hashes are ready. An interrupted release requires explicit operator investigation; do not delete the reservation automatically. The export includes `test.jsonl`, taxonomy, audit, selection receipt, the exact development report bytes, active gate and a manifest. Report, selection and gate bytes are captured once before validation and those exact snapshots are exported, preventing later source edits from changing the validated release. The test loader likewise verifies and parses captured file bytes, and enforces the exporter code SHA-256 as well as file hashes, selection, labels and prompts. Changes to exporter code after release require explicit review rather than silently loading a release made by different code. Hashes provide reproducibility and accidental-change detection, not authentication against deliberate coordinated replacement.

Only synthetic fixtures have exercised this exporter during implementation. The official CSV remains unopened. Inference and evaluation must enforce `--allow-sealed`, receipt-matching adapter/prompt/decoder, exclusive result paths and no post-test tuning. Export itself does not run either model.
