# Frozen cross-task transfer evaluation

## Question and source

Does the selected v1 Nemotron Mini 4B B4/B9 edit adapter choose useful
one-edge repairs when both graph order and book parameters change? The
[SRG(25,12,5,6) graph6 collection](https://users.cecs.anu.edu.au/~bdm/data/sr251256.g6)
contains fifteen 25-vertex colorings valid for red B6 and blue B6. Source
SHA-256: `f55f86eb48cd61c3315c62341cc3a8c59289c58504edb3ab859d7ef2068b9aa4`.
This is a separate source from the ten v1 26-vertex B4/B9 graphs. Keep the
raw graph6 file outside the repository and attribute Brendan McKay's
CC BY 4.0 collection if derived examples are distributed.

## Freeze before model evaluation

`transfer_v2.py` uses seed 2026101602 and splits graph6 records 0–7 for
development and 8–14 for held-out test, before random vertex relabeling.
Generate 150 distinct invalid starts per split using 3–8 uniform edge
flips. Reject zero-loss starts. Freeze the source and data SHA-256 hashes in
the manifest. The exact teacher checks all 300 possible edge flips and
chooses the least next excess-page loss, breaking ties lexicographically.
No v1 adapter or prompt tuning is allowed after development results.

Use the selected v1 adapter, SHA-256
`606c4efd0a33a808399daaab7a580ad3253181d1c65cece5667083153104e2cf`,
and the same untouched pinned base, revision
`791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. Replace only the explicit
task facts in the user prompt: K25, red B6/blue B6, 25 rows, and vertex
bounds. Keep the v1 system prompt, chat template, deterministic decoding,
and 32-token limit. Run full development and test inference once per model.
This is zero-shot task transfer, not new training.

## Outcomes and interpretation

Primary outcome is mean exact one-flip excess-page loss reduction, with
malformed output scored as zero reduction. Also report valid action rate,
strict improvement count, one-flip repairs, paired adapter-minus-base
bootstrap interval with 2,000 task draws (seed 871), and verifier calls. Report a
separate 2,000-draw bootstrap (seed 872) that resamples source graph6 records as
clusters to reflect the small number of distinct anchors. One-action
controls are random edge, random shown violating spine, best of twelve shown
spines, and exhaustive teacher. Report wall-clock inference and CPU search
separately. A five-pass greedy CPU repair control uses at most 1,500 candidate
checks per start.

All starts are perturbations of known valid SRGs. Success here supports
transfer across this one source and parameter pair only. It does not show
general graph search or a new Ramsey bound. The v1 test remains frozen and
must not be reopened or retuned from this study.
