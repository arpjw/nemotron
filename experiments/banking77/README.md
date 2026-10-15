# BANKING77 Nemotron intent-classification experiment

This experiment adapts the pinned 4B Nemotron Mini checkpoint to route banking-support queries into 77 publisher intent labels. It tests a narrow classification task, not financial analysis or production support safety.

The frozen capped design uses 2,308 training examples and 1,993 development examples from the publisher's official training partition. A one-epoch QLoRA adapter reached **0.82978 development macro-F1**, versus **0.39915** for untouched Nemotron under the same taxonomy-constrained decoding. The paired group-bootstrap 95% interval for the gain is **[0.41004, 0.45460]**. The prospective gate in `docs/GATE_V2.json` passed. The selected checkpoint is undergoing its single official-test comparison; final test scores are not yet in this source snapshot.

`docs/PROTOCOL.md` records the original prospective design and therefore contains pre-execution wording. Current execution results are tracked in the root `results/` reports. The official-test references and inference receipts are withheld from this source repository; the local evidence archive preserves them and their checksums.

## Reproduction

From this directory, run `PYTHONPATH=src python -m unittest discover -s tests -q` in the pinned CPU validation environment. GPU requirements are listed in `requirements-gpu.txt`; commands and provenance checks are in `docs/MODEL_RUNTIME.md`. The frozen train/development records are included. Reacquire raw publisher files only from the pinned commit and verify every hash in `data/raw/source_manifest.json` before regenerating data. The official test is intentionally excluded from this repository.

The dataset is [CC BY 4.0](DATA_LICENSE); preserve [attribution and the change notice](DATA_ATTRIBUTION.md). The model has a separate NVIDIA license. Historical benchmark contamination, neighboring-label ambiguity, and one-seed uncertainty limit the claim.
