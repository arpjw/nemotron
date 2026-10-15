# Synthetic GPU pilot status

The supervised RTX 4090 pilot completed, its attributed pod was deleted, and the deletion was verified by exact-ID read and complete listing. The roughly 81-minute runtime at the quoted rates implies about $1.22 of compute and ephemeral-disk cost. Final provider billing remains unconfirmed. Detailed operational receipts and local SSH information stay outside this source repository.

Remote NF4/BF16 preflight, a real two-step training smoke, adapter save/reload, and the 100-step and 300-step development runs passed. The 300-step epoch took 1,188.74 seconds and reached 168/400 exact answers, with 400/400 valid JSON but only 105/320 correct numeric values on answerable examples. The parser-only comparator reached 400/400 on this synthetic development set. See [the reviewed summary](../../../results/SYNTHETIC_FINANCE.md).

The initial synthetic generator had row-index leakage and its measured outputs were invalidated. The corrected data are in `../data/frozen/`. Both corrected model runs are exploratory development results from one seed. The sealed synthetic test and robustness splits were not evaluated, and these results do not establish broad finance capability.

The local evidence archive retains the full adapter hashes, raw predictions, source snapshots, environment report, and invalidated-run records. This repository deliberately omits large binary and operational artifacts. [RUNPOD.md](../RUNPOD.md) describes the sanitized supervised procedure. The automated launcher remains fail-closed for unattended paid creation.
