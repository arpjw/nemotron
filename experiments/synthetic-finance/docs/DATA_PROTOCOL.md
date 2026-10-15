# Frozen financial snippet experiment

This first experiment tests a small model's ability to extract financial figures, identify the requested periods, perform one arithmetic operation, cite supplied evidence, and abstain when necessary. All facts, entities and text are original synthetic material. No SEC filing text, paid dataset, teacher-model output or external dataset is included. Results cannot establish financial advice quality, investment performance, or competence on real corporate filings.

## Data contract

Every JSONL row has `id`, `group_id`, `split`, `layout`, `task`, `input`, `prompt`, `target`, `response`, and `metadata`. Training reads only `prompt` and `target`. Inference reads only `id` and `prompt`. `response` is the reference object; `target` is its canonical JSON serialization. `metadata` contains latent facts for audit, never model inputs. `input` contains only entity, amount scale, numbered source lines, and question. `build_prompt` explicitly allowlists these fields.

Model outputs must have exactly four keys:

```json
{"status":"ok","value":"1250.00","unit":"USD","evidence":["L1","L3"]}
```

For missing, conflicting or undefined information:

```json
{"status":"unanswerable","value":null,"unit":null,"evidence":[]}
```

Values are two-decimal strings. Monetary answers normalize to USD dollars. Growth is `(new-old)/old*100`, expressed in `percent`; a nonpositive old value requires abstention. Rounding occurs once with Decimal ROUND_HALF_UP. Parentheses denote negative values. Questions cover revenue extraction, revenue difference, and revenue percentage change, with operating-income distractors. Source evidence order is irrelevant, but duplicate or fabricated identifiers fail.

## Splits and freeze

Default seed: 1729. Counts: train 2,400; development 400; final test 600; robustness 200. Exactly 20% of each default split requires abstention. Each split has three unique renderer families and disjoint fictional entities and latent record identities. Renderer assignment varies independently of task. Final test evaluates unseen layouts; robustness adds compact formats, mixed punctuation, and quoted malicious instructions. Its score is separate from the primary test score. All splits remain synthetic and share the same underlying task generator.

Generation is deterministic. A group hash excludes entity identity and rendering order and represents underlying facts and the requested operation. Validation rejects duplicate group IDs, example IDs and normalized prompts, checks labels against the oracle, checks target serialization, and reconstructs public prompts and rendered facts. Every example is currently its own group; future paraphrases must preserve group identity and remain in the same split. Dataset generation refuses an existing destination. Each file's bytes, size, count, generator code hash and scorer code hash are recorded in `manifest.json`; verification detects drift. The manifest is a reproducibility mechanism, not cryptographic protection against someone replacing both files and manifest.

Engineering tests may inspect synthetic labels for correctness. They must not use model scores on final test or robustness to guide training, prompt selection, hyperparameter selection or checkpoint selection. The CLI requires `--allow-sealed` for final scoring, and inference should require equivalent acknowledgement. After seeing final scores, any model changes informed by them make that test exploratory. Freeze a fresh independent test for another confirmatory claim.

## Evaluation

Primary metric is exact correctness: valid output schema plus correct status, two-decimal value, unit, and evidence set. Empty/missing model outputs count as failures. Extra prediction IDs and duplicate IDs fail evaluation. JSON key ordering and whitespace do not affect correctness. Extra prose, unknown keys, duplicate keys, nonfinite values, wrong types, repeated evidence IDs and invalid JSON fail.

Reports include denominator and count for each metric, task/layout slices, numeric accuracy on answerable records, and abstention precision/recall. When no abstentions are predicted, precision is null rather than a fabricated perfect score. The paired comparison bootstraps complete latent groups with a fixed seed and reports a 95% percentile interval. It captures finite evaluation-set uncertainty, not training-seed uncertainty or real-world distribution shift.

Compare the adapter with the untouched checkpoint using identical prompts, greedy decoding and generation limits. Also run a fixed few-shot base baseline with examples selected exclusively from train, if supported by the execution pipeline and sequence budget. If few-shot is omitted, report the primary comparison explicitly as against the zero-shot base model. A latent-record oracle is a data validation check, not an independent model baseline. A parser baseline would help assess whether an LLM is necessary, but is not implemented in this initial package.

Proposed criterion, to lock before any model score: at least a 10 percentage point gain over the strongest implemented base baseline on exact correctness, paired interval lower bound above zero, and at most a 5 percentage point drop in correct abstention rate on unanswerable cases. Report all metrics even if the criterion fails. A single training seed supports feasibility only. Do not spend on training merely to reproduce a near-perfect base result.

## Commands

From the project root with the package on PYTHONPATH or installed:

```sh
python -m finmini.data --output data/frozen
python -m finmini.data --output data/frozen --verify
python -m finmini.scoring --data data/frozen/dev.jsonl --predictions runs/base/predictions.jsonl --output runs/base/dev-score.json
```

Scoring optional `--baseline path/to/predictions.jsonl` adds the paired interval. Prediction JSONL rows have exactly `id` and `prediction`, where `prediction` is the raw model output string. Use a fresh output report path for every evaluation.

## Optional future real-data test

The SEC offers unauthenticated company-facts APIs. An extension could render selected factual numerical disclosures in original wording, preserving issuer, filing accession, dates, units and source hashes. This would still measure our rendering of real facts rather than parsing original filings. Corporate filing prose is not assumed public domain. Follow the SEC request policy with a truthful contact User-Agent, caching and conservative request rate. No SEC data is acquired or included here.

Sources consulted on October 15, 2026:

- [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC access policy](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [Copyright Office on facts and expression](https://www.copyright.gov/help/faq/faq-protect.html)
- [FinQA README, including documented historical formatting leakage](https://github.com/czyssrs/FinQA/blob/main/README.md)
- [FinQA repository license](https://github.com/czyssrs/FinQA/blob/main/LICENSE)

Official `--baseline` comparisons additionally require a completed `inference_manifest.json` beside each prediction file. The scorer checks data and prediction hashes, pinned model revision, system prompt, quantization, selected IDs and full-set row counts. Baseline and adapter must use identical greedy decoding settings; the baseline must have no adapter and the candidate must record an adapter checksum. Partial smoke runs can receive ordinary descriptive scores but cannot pass this paired-comparison gate.

## Generator 1.0.1 correction before capability evaluation

An additional audit found three shortcuts in generator 1.0.0: public entity suffixes exposed the row index used to assign abstentions, the total number of fact lines exposed missing/conflicting cases, and operating income was a fixed multiple of revenue. Generator 1.0.1 uses opaque deterministic entity identifiers, exactly seven financial fact lines in every example, and independently sampled distractor values. Answerable extraction examples also vary irrelevant prior-period revenue records, so counting revenue rows alone does not determine whether extraction is answerable. New regression tests check these properties. The original frozen dataset is retained as `data/frozen-v1-engineering-only`, and its original verification modules are in `archives/pilot-v1/src/finmini`.

Only `data/frozen` with generator version 1.0.1 is the current capability dataset. Earlier runs, if any, are engineering checks and must not support a clean capability claim. All four current JSONLs and the manifest were regenerated together; neither old training rows nor old model predictions may be combined with the corrected test set to claim the registered comparison.
