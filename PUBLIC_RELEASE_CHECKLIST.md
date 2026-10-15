# Public release checklist

This repository is private. The checklist records work needed before changing visibility or publishing an adapter, article, or model card. Checked items have direct evidence in this repository or the local experiment archive. The release decision is separate from completing the experiments.

## Research record

- [x] Pin the Nemotron checkpoint revision and BANKING77 publisher commit.
- [x] Freeze training/development splits, protocol, active gate, prompts, and deterministic decoding.
- [x] Compare the selected adapter with untouched Nemotron under the same development procedure.
- [ ] Preserve the one-time official-test release and both complete official inference receipts in the local evidence archive.
- [ ] Finish the official paired evaluation, independent checksum replay, and final limitations review.
- [ ] Separate exploratory synthetic results, development selection, and official held-out results in every public claim.
- [ ] State that one seed and a historical benchmark do not establish broad finance ability or contamination-free transfer.

## Rights and attribution

- [x] Include BANKING77's CC BY 4.0 license, attribution, source commit, and change notice.
- [ ] Choose a license for the original code and writing; keep it separate from data and model licenses.
- [ ] Review the pinned NVIDIA model license and its distribution terms before publishing adapters or tokenizer files.
- [ ] Check any model card, checkpoint name, and article wording for clear independent-project attribution.
- [ ] Decide whether the frozen train/development derivative remains in the public source release; preserve CC BY attribution if it does.

## Repository hygiene

- [x] Keep raw publisher test references, weights, tokenizers, fitted `.joblib` models, runtime logs, pod receipts, and credentials out of tracked source.
- [ ] Run a full tracked-file secret and personal-path scan immediately before any visibility change.
- [ ] Verify the public bundle has no SSH host, personal filesystem path, account identifier, or unreviewed operational receipt.
- [ ] Run the CPU test suite in a fresh environment and the documented GPU replay if affordable and scientifically justified.
- [ ] Check that published result tables match the immutable manifests and hashes in the local evidence archive.
- [ ] Confirm the repository default branch, README, links, citation, and issue/security contact are appropriate for public readers.

## Publication

- [ ] Resolve editorial placeholders in `writing/ARTICLE_DRAFT.md` from final official results and final billing evidence.
- [ ] Independently review the article's benchmark and finance claims against the result report.
- [ ] Decide whether to publish source, result report, adapter, and article together or in separate steps.
- [ ] Obtain explicit approval for changing repository visibility or publishing the article.
