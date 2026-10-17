# Book graph-repair public source release

This release presents a bounded result about learned local repair on
perturbations of known book-Ramsey colorings. It does not claim a new Ramsey
bound, general graph-search ability, or a speed advantage over CPU search.

## Publishable source scope

- Exact verifier, dataset generators, CPU controls, inference and evaluation
  scripts, tests, frozen protocols, and aggregate result reports.
- Pinned external graph source links and checksums with
  [attribution](DATA_ATTRIBUTION.md).
- The v1 one-action result and exploratory rollout, explicitly separated.
- The v2 cross-task transfer result once its base and adapter predictions,
  controls, and independent checks are complete.

## Keep private for this release

- Adapter binaries, NVIDIA base weights and tokenizer copies.
- Frozen generated examples, per-state predictions, trajectories, and
  certificates in the local private evidence archive.
- Runtime logs, billing receipts, SSH details, personal paths, and keys.

## Final checks before a public GitHub change or article

- [x] Finish v2 inference and exact evaluation; state failures as clearly
  as successes and keep the unopened v1 test unchanged.
- [x] Check every number in the public result note against frozen report
  files and record adapter, source, and dataset hashes.
- [x] Run focused source tests and independent exact replay of any saved
  zero-loss certificate.
- [x] Scan every proposed tracked file and the public bundle for secrets,
  personal paths, operational receipts, and large artifacts.
- [x] Check the final staged diff and current repository visibility.
- [x] Obtain approval to publish the public source and article copy
  (October 16, 2026). Adapter/model upload remains outside this release.

The existing root `PUBLIC_RELEASE_CHECKLIST.md` applies to the earlier
BANKING77 release. It does not clear this book study for publication.
