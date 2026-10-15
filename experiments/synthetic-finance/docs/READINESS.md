# GPU pilot status, 2026-10-15

## Paid pilot completed and pod deleted

The supervised pilot on RTX 4090 Secure pod `61tt0ldq65ewok` is closed. It was created at **2026-10-15 19:33:37 UTC** (12:33:37 PM Pacific), and deletion was verified at **20:55:05 UTC** (1:55:05 PM Pacific), before the 21:33:37 UTC deadline. Evidence: delete API **204**, subsequent get **404**, and a complete empty pod list with no next page. See `../results/pilot-20261015/pod_receipt.json`.

The creation-to-deletion-verification interval was about **81 minutes 28 seconds**. At $0.89/hour compute plus approximately $0.0082/hour for 60 GB container disk, estimated runtime cost is **about $1.22**. This is a rate-times-duration estimate, not confirmed final billing. The $10 stage cap was the authorized ceiling, not the amount spent.

Remote 4-bit preflight passed. Two smoke training steps, adapter saving, and adapter reloading also passed. Reported peak reserved GPU memory was **12.57 GB**. The reported adapter checksum was abbreviated as `43cc...d224`; this document does not substitute that abbreviation for the full artifact hash.

These checks establish that the remote training path can load and update the model. They do not establish task-quality improvement.

## Completed corrected-data development results

The 300-step one-epoch follow-up completed in **1,188.74 seconds**. Its reported adapter checksum is abbreviated as `17ada7...3c24`; verify the complete hash against the exported manifest.

| Metric | 100 steps | 300-step epoch |
|---|---:|---:|
| Exact answers | 147/400 (36.75%) | 168/400 (42.00%) |
| Valid schema | 399/400 | 400/400 |
| Correct value on answerable rows | 103/320 | 105/320 (32.8125%) |

The epoch model got status correct on **365/400** and correctly abstained on **63/80** unanswerable rows. Exact task scores were extraction **104/134**, difference **42/133**, and growth **22/133**. Arithmetic remains a bottleneck: the longer run increased correct answerable values by only two, despite perfect schema compliance.

The parser-only calculator scored **400/400** on development data; the model-assisted calculator scored **351/400** in the earlier 100-step analysis. This benchmark is too synthetic to support a broad capability claim. Assisted results include calculator help and are not unaided model reasoning. Both model runs are exploratory development results from one training seed. No sealed-test or robustness evaluation was run, and no general finance capability is established.

See `../results/pilot-20261015/RESULTS.md` for the concise result record.

## Data correction and prior validation

The original synthetic dataset contained row-index leakage in entity names. The initial measured run was stopped, and the dataset was rebuilt before the 100-step results above. Old-data outputs remain invalid for the intended quality comparison. Preserve each run's matching dataset manifest and source; never combine old predictions with corrected labels.

Earlier local validation passed 48 tests without skips in a disposable Python 3.11 CPU environment. Those tests did not detect the original entity-name leakage, so passing tests alone does not establish benchmark validity.

## Still unresolved

- Sealed-test and robustness splits remain unevaluated. Development results are limited to the synthetic task above.
- Complete measured-run artifacts, export, and checksum verification.
- Final billing reconciliation remains pending; pod deletion is verified.
- Available Runpod credit: the $100 account balance remains user-reported. MCP OAuth and the existing SSH key provide the active execution path despite absent CLI credentials.
- Unattended operation: the standalone automated launcher remains blocked because no durable automatic teardown integration has been verified. The completed pilot was actively supervised in this conversation.

Retain local raw predictions, manifests, source snapshots, environment reports, and full adapter checksums. Preserve incomplete/invalid runs with explicit labels; never combine old-data predictions with corrected labels. The ephemeral pod is deleted and cannot supply missing files.

The operational procedure is in `../RUNPOD.md`.
