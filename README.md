# Nemotron small-model post-training experiments

Independent research on narrow post-training tasks using the pinned `nvidia/Nemotron-Mini-4B-Instruct` checkpoint. This is an independent project, not an NVIDIA project or a general finance assistant. The source release excludes model weights, adapters, official-test references, and operational receipts.

## Experiments

| Experiment | Task | Current evidence |
|---|---|---|
| [Synthetic finance snippets](experiments/synthetic-finance/) | Extract and calculate values from fictional snippets | One-epoch adapter reached 42% exact correctness on 400 development examples, from 0% for the untouched model. Gains were concentrated in abstention and formatting; numeric reasoning remained weak. A deterministic parser solved all 400, so this is not a useful finance-expertise claim. |
| [BANKING77 intents](experiments/banking77/) | Classify a banking-support query into one of 77 publisher labels | One-epoch QLoRA on 2,308 capped training examples. On the publisher's 3,080-row official test, constrained macro-F1 was 0.84443 versus 0.41337 for untouched Nemotron. The paired gain was 0.43105, 95% interval [0.41517, 0.44919]. |
| [Graph counterexamples](experiments/graph-counterexamples/) | Construct exact counterexamples to synthetic graph inequalities | A fresh-adapter Runpod pilot completed: 10/30 held-out successes versus 0/30 for the untouched base. A post hoc check found that empty or complete graphs solve every adapter success, so this does not establish nontrivial graph search. See the measured results and limits. |
| [Book Ramsey search](experiments/book-ramsey/) | Search for a 27-vertex coloring avoiding red \(B_4\) and blue \(B_9\) | Exact checker and CPU baselines are implemented; no 27-vertex certificate was found. A parity count narrows the necessary red edge range to 157–171. |
| [Book graph repair](experiments/book-ramsey/EDIT_RESULTS.md) | Choose edge flips to repair perturbed known book colorings | On 26-vertex B4/B9 starts, the adapter reduced exact loss by 5.17 on 150 held-out cases. A [zero-shot transfer test](experiments/book-ramsey/TRANSFER_V2_RESULTS.md) on a separate 25-vertex B6/B6 source gave +3.86 versus −6.09 for the untouched base. Cheap exact search remained stronger and faster for complete repairs; neither study found a new Ramsey bound. |

The [BANKING77 result report](results/BANKING77.md) separates checkpoint selection on development from the one-time official comparison. Both models used the same prompt, model revision, and taxonomy-constrained decoder. The paired official interval describes this fixed benchmark and does not capture training-seed or domain uncertainty. A TF-IDF logistic comparator scored 0.82852 macro-F1 on development; the goal here is improvement over untouched Nemotron, not superiority over that smaller classifier.

The book graph-repair study has a [published article](https://aryasomu.com/writing/nemotron-graph-repair.html), [article source](writing/BOOK_GRAPH_REPAIR.md), and [public release record](experiments/book-ramsey/PUBLIC_RELEASE.md). Its results concern one-step local repair; the selected adapter and per-state predictions remain in the private evidence archive.

The trained adapter is [Nemotron Mini 4B Banking](https://huggingface.co/arpjw/Nemotron-Mini-4B-Banking) on a separate Hugging Face model page. That page contains the adapter, model card, notices, and model-license texts; this source repository contains no model weights. The accompanying article is [I post-trained Nemotron 4B into an Average Banker](https://aryasomu.com/writing/nemotron-average-banker.html).

## Layout

- `experiments/synthetic-finance/`: original synthetic task, code, frozen examples, tests, and protocol.
- `experiments/banking77/`: publisher-backed intent task, code, frozen train/development examples, tests, prospective gate, and attribution.
- `experiments/graph-counterexamples/`: synthetic graph task, exact verifier, frozen splits, random baselines, and separate QLoRA path.
- `experiments/book-ramsey/`: exact book verifier, derived search bounds, bounded CPU and SAT baselines, and their checked records.
- `results/`: reviewed aggregate findings and checksum references. Model weights, full runtime logs, and official-test references stay in the local evidence archive.
- [`writing/ARTICLE.md`](writing/ARTICLE.md): source copy for the published article.
- [`writing/figures/`](writing/figures/): light and dark SVG figures, figure-data provenance, and the generation script.
- [MODEL_RELEASE.md](MODEL_RELEASE.md): terms and packaging checks for the separate adapter page.
- [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md): record of the checks completed before publication.

## Reproduction boundaries

The pinned Nemotron model revision is `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. GPU dependency versions and exact commands are documented in each experiment's `docs/MODEL_RUNTIME.md`. The full local suites use Python 3.11 with `requirements-validation.txt`; run `PYTHONPATH=src python -m unittest discover -s tests -q` from each experiment directory. The pull-request workflow runs smaller standard-library protocol suites. BANKING77 frozen train/development records are included under the publisher's CC BY 4.0 terms; the raw publisher test and released official references are intentionally outside the source bundle. A fresh acquisition must verify the publisher commit and hashes in `experiments/banking77/data/raw/source_manifest.json`.

The synthetic test and robustness files are included for reproducibility, although neither was scored in the pilot. They are now public and cannot serve as untouched future holdouts.

## Scope and rights

The BANKING77 results describe one training seed on a historical benchmark. We cannot verify that BANKING77 was absent from Nemotron pretraining. The experiments do not establish investment skill, production banking reliability, calibrated confidence, or out-of-taxonomy handling.

Original code is [MIT licensed](LICENSE). Original prose is [CC BY 4.0 licensed](LICENSE-WRITING.md). The included BANKING77 train/development derivative retains the publisher's separate [CC BY 4.0 license](experiments/banking77/DATA_LICENSE), [attribution, and change notice](experiments/banking77/DATA_ATTRIBUTION.md). Nemotron weights and adapters have separate [NVIDIA model terms](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct); neither weights nor tokenizer copies are in this source repository. The source license does not grant rights to distribute those materials.

For reproducibility corrections, use [GitHub Issues](https://github.com/arpjw/nemotron/issues). See [SECURITY.md](SECURITY.md) for sensitive reports; do not post private banking data or credentials in issues.
