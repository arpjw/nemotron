# Exploratory training-seed sensitivity check

The v1 confirmatory comparison selected seed 20261016 on development and
opened the held-out test once. To measure one concrete limitation of that
result, train two additional fresh adapters from the same pinned base and
frozen training split, changing only the seeds to 20261017 and 20261018.
Keep one epoch, rank 16, accumulation eight, AdamW 2e-4, and the same
response-only action target. Score the complete frozen development split
with the same greedy decoder and exact evaluator. Record per-seed adapter
hashes, losses, valid actions, and mean exact loss reduction.

Do not choose a new adapter, retune hyperparameters, or score the held-out
test again using these replications. These development scores describe seed
sensitivity on the same tasks and cannot provide a multi-seed held-out
interval. The original test report remains tied to seed 20261016.
