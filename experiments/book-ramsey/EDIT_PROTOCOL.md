# Graph-repair study v1

## Question and frozen data

Can a fresh Nemotron Mini 4B adapter choose useful single-edge repairs for
26-vertex red/blue colorings that avoid red B4 and blue B9? The ten external
SRG(26,10,3,4) graph6 records are a calibration set. Records 0–7 generate
training states, record 8 development states, and record 9 test states.
The split occurs before independent random vertex relabeling. Each state is
perturbed by 3–8 uniformly sampled edge flips. States with zero violation
are rejected. This split tests transfer across these ten records, not across
different Ramsey parameter pairs or graph orders.

The private local dataset has 1,200/150/150 train/development/test states,
seed 20261016. Its manifest pins the source file and all split hashes. The
teacher action minimizes the exact one-flip excess-page loss over all 325
edges, breaking ties lexicographically. The teacher is a costly algorithmic
reference, not a proof that the action leads to a solution.

## Outcomes and comparators

The primary one-action outcome is mean reduction in exact excess-page loss,
counting malformed model output as zero reduction. Secondary outcomes are
valid action rate, strict loss improvement, and exact valid coloring after
one flip. The untouched base, one uniformly random flip, one uniformly
chosen spine among the twelve shown in the prompt, and the 325-call
exhaustive teacher use the same frozen starting states. The model gets the
current state, exact aggregate feedback, and the twelve largest violating
spines with excess counts, then makes one proposal. The controls can compute
the same feedback from the supplied graph. Record verifier calls and GPU
inference time separately.

Development selection requires positive adapter mean loss reduction that
exceeds untouched base and both one-flip controls. Only then score the test
split once with the frozen adapter. Paired task bootstrap intervals use 2,000
draws and seed 871. They condition on this one source collection, dataset
seed, and training seed. A later multi-step rollout must be declared as a
separate secondary analysis with matched proposal budgets.

CPU search controls run on the development states with budgets 8, 325, and
1,625 candidate verifications per state. Random walk, greedy best-flip, and
annealing all start from the same rows. A loss of zero is checked again by
the independent exact book verifier. These controls expose the local-search
difficulty but are not time-matched to GPU decoding.

## Model and resource gate

Use `nvidia/Nemotron-Mini-4B-Instruct` revision
`791833e92ebddb0bc2c1007f6d2b6764f886a2ae`, with a **new** rank-16
NF4 QLoRA adapter. Do not initialize from the BANKING77 adapter or the earlier
graph counterexample adapter. One response-only epoch, microbatch one,
accumulation eight, AdamW 2e-4, seed 20261016, greedy 32-token action output.
Preflight token lengths and one smoke optimizer step before the formal run.
The old graph and book records remain unchanged.

The prior graph-project authorization is a $40 total Runpod cap. The first
pilot used about $0.25 estimated compute, with final billing unposted at that
time. Before a new pod, check current billing, remaining balance, resource
price, SSH, and active resources. Keep a conservative combined spend ceiling
below $40, supervise the pod, export all adapter and prediction receipts, and
verify pod and volume deletion. No public adapter release or deployment is
authorized by this protocol.

## Interpretation

This is a local repair benchmark around known 26-vertex constructions. A
successful edit can be independently certified, but it does not settle the
27-vertex B4/B9 gap. A zero-loss failure is not evidence of impossibility.
The model is distilled from exhaustive one-step search; any claimed benefit
must be assessed against random and search controls, verifier calls, and
inference cost. The source graph6 is externally hosted and must be fetched
and checksum verified for reproduction. Dataset and adapter rights require
review before release.
