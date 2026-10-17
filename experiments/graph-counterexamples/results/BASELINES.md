# CPU development baselines

The frozen development split contains 30 synthetic tasks. Each task asks for a counterexample graph with an exact positive integer margin. A fixed random graph sampler used seed `871 + task index` and graph densities drawn from `{0.18, 0.35, 0.50, 0.65, 0.82}`.

| Method | Proposals per task | Verified witnesses | Success rate |
|---|---:|---:|---:|
| Random graph | 1 | 8 / 30 | 0.2667 |
| Random search | 32 | 25 / 30 | 0.8333 |

These are development engineering baselines, not model outcomes. The 32-proposal run uses 32 times as many verifier queries as one model response, so it is a computational reference rather than the budget-matched primary comparison. The 30 tasks are one deterministic generator seed and do not support a general graph-reasoning claim.

Frozen development SHA-256: `bf5deb18972b3cace7e2daab3e0013b1b12f8b156f5fec1cc8c7e6e340168fbb`. One-proposal predictions SHA-256: `c896527ccd57b5d4bd84a853f58691bad80b54c332d5381ad8bfced5dc1cbf84`. Thirty-two-proposal predictions SHA-256: `4658788ad97fea66f92d1b0805609fc82b4dea7675a5638575b722726c97e401`. Complete local predictions and comparison are under ignored `runs/` paths and can be regenerated with the README commands.
