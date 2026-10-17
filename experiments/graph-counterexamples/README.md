# Nemotron graph counterexample hunter

Can a small post-trained model construct counterexamples to unfamiliar graph inequalities? This independent experiment pairs a pinned Nemotron Mini 4B base model with an exact graph verifier. It uses only synthetic graphs. It does not claim to discover a new theorem or prove a surviving conjecture.

The model receives a claim of the form `a f(G) ≤ b g(G) + c` and a vertex count. It returns one JSON edge list. The verifier computes the integer margin `a f(G) - b g(G) - c`; a positive margin is a valid counterexample. Supported features are vertex and edge count, maximum degree, independence number, clique number, chromatic number, maximum matching size, and the product of independence and clique numbers. All invariants are computed exactly for at most eight vertices.

## Current state

- The generator, frozen 120/30/30 train/development/test splits, strict graph parser, exact scorer, random baselines, interactive puzzle mode, model inference, and one-epoch QLoRA path are implemented.
- CPU unit tests, random baselines, a full one-epoch QLoRA run, and the prespecified development and test comparison have been run. See [the measured Runpod pilot](results/RUNPOD_PILOT_PUBLIC.md).
- The test split is held for a single comparison after development selection. The source includes its targets for reproducibility, so it is not a secret benchmark.

## Try it locally

From this directory, without model dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
PYTHONPATH=src python -m graph_ce.play --data data/frozen --index 0
PYTHONPATH=src python -m graph_ce.baseline --data data/frozen --split dev --budget 1 --output runs/random1-dev/predictions.jsonl
PYTHONPATH=src python -m graph_ce.baseline --data data/frozen --split dev --budget 32 --output runs/random32-dev/predictions.jsonl
PYTHONPATH=src python -m graph_ce.evaluate --data data/frozen --split dev --predictions runs/random32-dev/predictions.jsonl --baseline runs/random1-dev/predictions.jsonl --output runs/random-comparison.json
```

Outputs under `runs/` are ignored by Git. The frozen splits are deterministic from seed `20261015`, and `data/frozen/manifest.json` records SHA-256 digests. `load_split` rechecks every split and rejects target, count, duplicate-task, and invariant-family leakage. Do not regenerate the frozen splits in place. To change the task design, write a new version and keep this record.

## GPU pilot

The training code uses the same pinned Nemotron Mini 4B base revision and QLoRA stack as the BANKING77 experiment, with a new adapter. The banking adapter was not used to initialize graph training. See [the protocol](docs/PROTOCOL.md) for the frozen comparison, runtime, spending gate, and claim limits.

After preparing a Python 3.11 CUDA environment with `requirements-gpu.txt` and caching the pinned base model:

```bash
PYTHONPATH=src python -m graph_ce.train --data data/frozen --output runs/graph-epoch --preflight-only
PYTHONPATH=src python -m graph_ce.infer --data data/frozen --split dev --output runs/base-dev --max-seconds 3600
PYTHONPATH=src python -m graph_ce.train --data data/frozen --output runs/graph-epoch --max-seconds 3600
PYTHONPATH=src python -m graph_ce.infer --data data/frozen --split dev --adapter runs/graph-epoch --output runs/adapter-dev --max-seconds 3600
PYTHONPATH=src python -m graph_ce.evaluate --data data/frozen --split dev --predictions runs/adapter-dev/predictions.jsonl --baseline runs/base-dev/predictions.jsonl --require-model --output runs/dev-comparison.json
```

These are individual stages, not a single unattended command or spending authorization. Run the model only under an operator-enforced total budget and deadline. Training and inference code fail closed on incomplete runs. If development improves, freeze the choice with `python -m graph_ce.select --dev-report runs/dev-comparison.json --output runs/selection.json`. Test inference requires `--allow-test --selection runs/selection.json` for both base and adapter under the same decoder.

## Limitations

The task generator sets thresholds using sampled graphs, so every task has a known witness and the benchmark studies constructive search rather than deciding arbitrary graph conjectures. The graphs are small, and generated train and evaluation tasks share the same grammar even though their ordered invariant pairs differ. Random-search difficulty depends on its sampling distribution and proposal budget. A successful edge list is a certified finite counterexample; failure to find one gives no proof that the inequality holds.
