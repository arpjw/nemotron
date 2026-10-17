# Graph counterexample protocol v1

## Question and frozen design

Does one response-only QLoRA epoch improve the pinned Nemotron Mini 4B checkpoint's **single-shot** success rate on synthetic graph inequalities with unseen ordered invariant pairs? The fixed generator seed is `20261015`. The train, development, and test sizes are 120, 30, and 30. Ordered invariant pairs are allocated to disjoint splits before tasks are generated. Within each split, the generator covers each allocated pair in turn and varies graph order and coefficients. Every generated target is an independently checked counterexample. The generator uses sampled graphs to choose a high-tail threshold; the verifier uses exact computation and accepts any valid graph, not just the stored target.

The primary outcome is the fraction of all tasks for which greedy generation returns a schema-valid graph of the required order with positive margin. Malformed output and incorrect or missing graphs count as failures. Secondary diagnostics are schema validity, positive margin, and edge count among successes. A one-proposal random graph baseline is the budget-matched comparator; 32-proposal random search is a stronger computational reference. Paired task bootstrap intervals use 2,000 draws and seed 871. The sample is small, so intervals describe these generated tasks and do not capture variation across training seeds or generator seeds.

## Model protocol

Use base `nvidia/Nemotron-Mini-4B-Instruct` at revision `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. Start a **new** rank-16 QLoRA adapter; do not continue from the BANKING77 adapter. One epoch, response-only loss, microbatch 1, accumulation 8, AdamW learning rate 2e-4, seed 20261015, no truncation beyond 1,024 tokens. Base and candidate use the same system prompt, task prompt, NF4 quantization, greedy decoding, and 256-token output cap. Record complete per-task outputs and source, data, model, and adapter hashes. Check training smoke and tokenizer lengths before a full run.

Do not choose hyperparameters on the test split. The one prespecified pilot runs on development; if it improves success rate over untouched Nemotron, freeze the resulting adapter and compare both once on test. If it does not, report that result and treat further revisions as a new protocol. No deployment or public adapter release is included. Review NVIDIA's pinned model terms before distributing weights or adapters.

## Resource gate

Runpod work requires an explicit total dollar cap and live supervision. The authorized cap for this project is **$40**, including compute and storage. Check account access, GPU price, capacity, and active resources before provisioning. Actively supervise the pod and terminate it well before the cap; process `--max-seconds` only guards code, not billing. Export adapter and inference evidence before deleting the pod. Verify teardown and billing afterward. Confirm SSH or another authenticated transfer route before starting a pod.

## Mathematical and evidential limits

Integer invariants are computed exactly for simple graphs with at most eight vertices. A positive margin refutes the stated inequality at that graph order. A negative or zero margin does not establish the inequality. A finite exhaustive search at one order still would not prove a general conjecture. The task family is synthetic and small; beating a random baseline here would not establish general theorem-proving ability. A model may learn threshold and graph-density heuristics. Future stress tests should vary generator seeds, graph orders, invariant expression grammar, and candidate budgets before broader claims.
