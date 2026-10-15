# Nemotron experiment guidance

- Read `README.md`, `PUBLIC_RELEASE_CHECKLIST.md`, and the relevant experiment's README and protocol before edits.
- Preserve frozen examples, manifests, gate definitions, and recorded result artifacts. Make a new version when changing a protocol; do not silently rewrite historical evidence.
- Keep BANKING77 official-test references and one-run receipts out of the source repository. The selected checkpoint was chosen on development data before the official comparison. Do not tune on official-test results.
- Separate synthetic extraction, BANKING77 intent classification, and broad finance claims. Report exact denominators, decoder, model revision, seed, and uncertainty scope.
- Model, data, and code rights are separate. Keep BANKING77 attribution and license with any derivative data; review the pinned NVIDIA license before adapter distribution.
- Do not commit raw credentials, pod receipts, SSH hosts, large model artifacts, or personal paths. Check `git status` and tracked files before each push.
- Paid GPU work needs an explicit bounded cap, active supervision, artifact export, and verified resource teardown.
- Run the narrowest relevant unit suite after code changes. The full local suites are `PYTHONPATH=src python -m unittest discover -s tests -q` from each experiment directory.
