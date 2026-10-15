# BANKING77 selected adapter versus untouched Nemotron

The study tested one narrow 77-intent banking-support classification task. The base model was `nvidia/Nemotron-Mini-4B-Instruct` at revision `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`. One rank-16 QLoRA adapter completed one epoch, 289 optimizer steps, on 2,308 examples. The publisher source commit was `57ec275d8078af65b7731c2a98be812d844a6d6b`. The frozen development comparison selected the checkpoint before the publisher's official test was released to the evaluation code. The official comparison ran once with the selected adapter and untouched base. Both used the same prompt and taxonomy-constrained greedy decoder.

| Split and metric | Untouched base | Selected adapter | Paired gain |
|---|---:|---:|---:|
| Development macro-F1, 1,993 rows | 0.39915 | 0.82978 | 0.43063, 95% CI [0.41004, 0.45460] |
| Development accuracy | 40.79% (813/1,993) | 83.04% (1,655/1,993) | 42.25 percentage points |
| Official macro-F1, **all 3,080 rows** | **0.41337** | **0.84443** | **0.43105, 95% CI [0.41517, 0.44919]** |
| Official accuracy | 42.53% (1,310/3,080) | 84.87% (2,614/3,080) | 42.34 percentage points |

The paired macro-F1 interval used 2,000 percentile-bootstrap resamples of complete normalized-query groups and all 77 fixed labels. It measures sampling variation on the benchmark, not uncertainty across training seeds or future domains. The development TF-IDF logistic comparator scored 0.82852 macro-F1, close to the adapter's 0.82978. This experiment targeted improvement over untouched Nemotron, not superiority over a cheap CPU classifier.

The [aggregate per-class official scores](BANKING77_PER_CLASS.json) show higher F1 for the adapter on all 77 labels. At per-class F1 of at least 0.8, six base labels and 55 adapter labels met the threshold. This distribution is a descriptive view of the fixed test outcome, not another model-selection criterion. The JSON records only per-class aggregates and the SHA-256 of the archived source report; it contains no test queries or per-row predictions.

The unconstrained output check found 396 malformed responses from the base and five from the adapter among 3,080 official rows. The primary score above uses constrained outputs for both. The overlap/conflict audit flagged five rows: three overlapped capped train or development records and two belonged to a conflicting-label group within the test. The overlap check did not include unused publisher training rows. Retaining all five gives the primary all-row result. Excluding them only as a sensitivity analysis gives macro-F1 0.41317 for base and 0.84445 for adapter, gain 0.43128, 95% CI [0.41547, 0.44954].

The local evidence archive contains both complete official inference receipts, per-row predictions, selected adapter, official export, and evaluation source. The adapter SHA-256 is `047303c0755892431269f3fc9c02a6327d08f3ae51b7c495936a3ee2a646bfdb`. The official base prediction SHA-256 is `aa1aa357e8fb6befc2c88f84a638501ca253096c0fbff1bfcdb7e6325c394f20`; the adapter prediction SHA-256 is `6b77376411740c0ed53ce8d233e3dd1d75916082403baa5de4b0f19b96e44da6`. An independent local evaluation replay matched every per-row outcome and all metrics within 1e-12. No test-driven checkpoint or decoding change followed the official comparison. Raw official references, model weights, and operational receipts remain outside Git.

The paid RTX 4090 pod was deleted after artifact export; Runpod returned 204 on deletion, 404 for the exact pod afterward, and an empty complete pod list. Quoted GPU cost was $0.89/hour plus about $0.0082/hour for disk. The runtime-based estimate for this pod is approximately $2.03; final provider billing is unconfirmed. The earlier synthetic pilot cost estimate is about $1.22, for approximately $3.25 combined.

This is one seed on a historical in-taxonomy benchmark. We cannot verify that BANKING77 was absent from base-model pretraining. The classifier was not evaluated for out-of-taxonomy rejection, calibrated confidence, broad financial reasoning, unfamiliar documents, or production banking reliability. Dataset and model rights remain separate from the MIT license for this project's original code.
