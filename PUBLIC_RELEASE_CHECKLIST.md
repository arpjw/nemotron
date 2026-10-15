# Public release checklist

This checklist records the checks before changing repository visibility or publishing an adapter, article, or model card. Checked items have direct evidence in this repository or the local experiment archive. The source release and website publication are separate decisions.

## Research record

- [x] Pin the Nemotron checkpoint revision and BANKING77 publisher commit.
- [x] Freeze training/development splits, protocol, active gate, prompts, and deterministic decoding.
- [x] Compare the selected adapter with untouched Nemotron under the same development procedure.
- [x] Preserve the one-time official-test release and both complete official inference receipts in the local evidence archive.
- [x] Finish the official paired evaluation, independent checksum replay, and final limitations review in `results/BANKING77.md`.
- [x] Separate exploratory synthetic results, development selection, and official held-out results in the result reports and article.
- [x] Review every final public claim in the article and root summary for the documented one-seed, historical-benchmark, and finance-scope limits.

## Rights and attribution

- [x] Include BANKING77's CC BY 4.0 license, attribution, source commit, and change notice.
- [x] License original code under MIT and original prose under CC BY 4.0; keep both separate from data and model terms.
- [x] Keep NVIDIA weights, tokenizers, and adapters out of this source release; review NVIDIA's pinned terms before any future model distribution.
- [x] Record the pinned model license and the separate adapter-release conditions in `MODEL_RELEASE.md`; do not apply the source MIT license to model artifacts.
- [x] Prepare an adapter-only Hugging Face model card and package with the archived artifact hashes, attribution, notices, and license copies, outside this Git repository.
- [x] Upload the adapter-only package to a private Hugging Face model page and verify its file list, sizes, and adapter hash.
- [x] Identify both license texts at the pinned base revision (June 2024 Open Model License metadata and August 2024 Community Model License attachment), and include both with attribution and modification notices in the adapter candidate.
- [x] Check checkpoint name and article wording for clear independent-project attribution.
- [x] Retain the frozen train/development derivative in the public source release with PolyAI CC BY 4.0 attribution and the change notice.

## Repository hygiene

- [x] Keep raw publisher test references, weights, tokenizers, fitted `.joblib` models, runtime logs, pod receipts, and credentials out of tracked source.
- [x] Scan this local release candidate and all 82 Git-history blobs for personal paths, private keys, credential assignments, SSH-host patterns, and blocked artifact names (October 15, 2026); none were found.
- [x] Run the hosted offline protocol workflow on the private default branch: [passing final-evidence run](https://github.com/arpjw/nemotron/actions/runs/38822192105).
- [x] Run a final tracked-file and Git-history secret and personal-path scan immediately before any visibility change.
- [x] Verify the final public bundle has no SSH host, personal filesystem path, account identifier, or unreviewed operational receipt.
- [x] Run both full CPU suites in a fresh Python 3.11 environment and independently replay the official evaluation from archived predictions.
- [x] Check that result tables match the immutable manifests and hashes in the local evidence archive.
- [x] Confirm the `main` default branch, README, relative links, `CITATION.cff`, GitHub Issues, and `SECURITY.md` are appropriate for public readers.

## Publication

- [x] Finalize `writing/ARTICLE.md` and its site-format draft; identify runtime-based costs as estimates pending final provider billing.
- [x] Review the article's benchmark and finance claims against the result report.
- [x] Prepare the source and aggregate result report for publication; keep adapter binaries outside the release.
- [x] Obtain explicit approval for changing repository visibility and publishing the article (October 15, 2026).
- [x] Obtain explicit approval to make the private Hugging Face model page public; its link is in the article and source README (October 15, 2026).
