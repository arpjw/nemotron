# Nemotron graph counterexample pilot

## Frozen question and outcome

The prespecified question was whether one response-only QLoRA epoch improves the single-shot success rate of `nvidia/Nemotron-Mini-4B-Instruct` at revision `791833e92ebddb0bc2c1007f6d2b6764f886a2ae` on unseen ordered pairs of graph invariants. The BANKING77 adapter was not used. The 120/30/30 synthetic train/development/test splits and exact verifier were frozen before the GPU run. The development gain was positive, so the adapter was selected before either model was scored on test.

| Split | Untouched base | New graph adapter | One random graph | 32-graph random search |
| --- | ---: | ---: | ---: | ---: |
| Development, 30 tasks | 2/30 (6.7%) | 12/30 (40.0%) | 8/30 (26.7%) | 25/30 (83.3%) |
| Held-out test, 30 tasks | 0/30 (0%) | 10/30 (33.3%) | 5/30 (16.7%) | 27/30 (90.0%) |

The adapter minus base success-rate difference was +10/30 on both splits. The paired task percentile bootstrap intervals were +4/30 to +16/30 on development and +5/30 to +15/30 on test (2,000 draws, seed 871). The adapter's test advantage over one random proposal was +5/30, with a paired interval of -1/30 to +11/30, which includes zero. The 32-proposal random search was much stronger than single-shot generation. These intervals condition on this one generated task set and do not measure variation across training or task-generation seeds.

Schema validity improved from 22/30 to 30/30 on development and from 18/30 to 30/30 on test. Failures, including malformed or wrong-order graphs, remained in the denominator. Every successful graph was checked with exact integer invariants. For example, test task `test-0017` asked for a graph with `2χ(G) > ν(G) + 7` on seven vertices. The adapter returned `K7`, for which `χ=7`, `ν=3`, and the violation margin is `14 - 3 - 7 = 4`.

## Post hoc diagnostic: extremal templates

After the held-out comparison, we checked the empty and complete graph at each requested order. This was not a prespecified model comparator. On development the empty graph solves 8/30 tasks and the complete graph solves 20/30; on test they solve 17/30 and 10/30. Their two-proposal union solves 25/30 on each split. **Every adapter success on both splits is in that union.** The adapter solves 0/5 tasks on each split that neither extreme solves, while 32-proposal random search solves 3/5 such development tasks and 5/5 such test tasks. The pilot therefore shows a gain over the untouched language model, but provides no evidence that the adapter learned to search past these two templates.

## Run and provenance

- One epoch completed on a secure Runpod RTX 4090: 120 examples, 15 optimizer steps, rank-16 NF4 QLoRA, BF16, learning rate `2e-4`, accumulation 8, seed `20261015`. Maximum tokenized example length was 255, below the 1,024-token limit. The training command took 34.85 seconds and saved adapter SHA-256 `0f64011740c008ba93c7c01e8a43375263a7bffeaae802139481da8641899517`.
- Base and adapter used the same greedy decoder with an explicit attention mask and 256 new-token cap. The initial base engineering pass without the explicit mask was discarded; both formal model passes use the corrected `infer.py` source hash recorded in their manifests.
- The development report SHA-256 is `3e9fb0e9b6bbe21157173eb1d3d142fdc2526f0b22f3e48074190f7745fdc99a`. The selection receipt binds that report, data split, model revision, and adapter hash. The held-out report SHA-256 is `ec468f5299ae6bbb3fc92f6502edbbf2ed8ab17271c19a6d50f8ea43a69f8263`.
- The source and frozen data remain in the experiment repository. The private evidence archive contains the adapter, complete per-task predictions, training log, inference manifests, selection receipt, reports, and environment versions. Its SHA-256 is `7847e53c5496c70a803d169fb4f631c01a3f1a3862ea078815dbf8eac5031a95`. Local CPU replays produced byte-identical development and test reports. Seven unit tests passed locally and on the pod.

## Runpod setup and cost

The pilot ran on a single Runpod RTX 4090 under the authorized $40 total cap.
After evidence export, the pod and volume were deleted; fresh resource lists
showed no active pods or volumes. Roughly 17 minutes at the quoted $0.89/hour
implies about $0.25 of compute. Billing had not posted a final record at the
last check, so this is an estimate. Operational identifiers and the complete
run receipt remain in the private evidence archive.

## Interpretation

This is evidence of improvement over the same model's untouched base on this small synthetic, bounded-size task set. It is not evidence of general theorem proving or a new graph-theory discovery. The task generator uses sampled graphs to set thresholds and stores known witnesses; train and evaluation share a grammar. Ordered invariant pairs are disjoint across splits, but reversed pairs can recur: 5 of the 8 test ordered-pair families have the reverse direction in training. The prompt also describes `αω` as "their product" after listing several invariants, which could be read ambiguously. A new study should freeze clearer prompts and unordered-family splits, require witnesses beyond simple templates, and preregister multiple training and generator seeds with comparable proposal budgets. No adapter was published or deployed.
