# Synthetic finance snippet pilot

This exploratory task uses fictional snippets with deterministic labels. It tests strict JSON extraction and simple calculations, not knowledge of real companies or investment decisions.

| Development result, 400 examples | Untouched Nemotron | 100-step adapter | 300-step adapter |
|---|---:|---:|---:|
| Exact status, value, unit, and evidence | 0/400 | 147/400 (36.75%) | 168/400 (42.00%) |
| Valid strict schema | 0/400 | 399/400 | 400/400 |

The 300-step adapter's answerable numeric values were correct on 105/320 examples. A deterministic parser solved 400/400 on the same development set, and a model-evidence plus calculator system solved 351/400. Much of the model's measured gain came from abstention and output formatting. The result does not establish reliable arithmetic or broad finance ability.

The first synthetic generator version had shortcut leakage and was invalidated before this pilot. The corrected benchmark and adapter were tested on one GPU with bounded cost; the pod was deleted and absence verified. Estimated runtime cost was about $1.22, with final billing not confirmed. The sealed synthetic test and robustness sets were not used for a confirmatory claim. The local evidence archive retains raw predictions, manifests, checksums, and adapter bytes outside this source repository.

Reproduce the task and inspect its limits through `experiments/synthetic-finance/`. The experiment's original protocol and source are preserved there; the [root README](../README.md) records current status.
