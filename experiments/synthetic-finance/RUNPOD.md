# Supervised Runpod procedure

The synthetic pilot used one ephemeral RTX 4090 pod under a $10 stage cap and a quoted GPU rate below $1/hour. The user-reported account balance was not independently verified. The pilot finished and the attributed pod's deletion was confirmed by both an exact-ID read and a complete pod listing. Its estimated runtime charge was about $1.22; final provider billing is separate.

This document records the repeatable procedure without publishing personal paths, SSH details, or operational receipts. It is not an unattended launcher.

1. Review the frozen manifest, model revision, model license, dependency pins, expected VRAM and disk use, exact study objective, and maximum charge before creating any resource.
2. Create one explicitly priced GPU pod with ephemeral disk. Record its unique ID, actual quoted rate, creation time, and a hard operator deadline in a private receipt. Do not retry an uncertain create request without checking for the exact resource.
3. Verify disk, CUDA, package versions, model revision, and one real forward/backward smoke before the full training run. Keep all timeouts finite.
4. Run untouched and selected model inference using identical prompts, decoding and evaluation. Preserve the prospective data split and do not select a checkpoint using the sealed test.
5. Copy adapter, predictions, manifests, logs and environment receipts out of ephemeral storage. Verify checksums locally.
6. Delete the exact resource that was created. Confirm deletion by an exact-ID read and a complete list. Reconcile final billing when available.

`scripts/runpod_launch.py` remains fail-closed for unattended creation. Paid runs require a supervising operator or a separately reviewed durable controller. Never put API credentials in the training container or source repository.
