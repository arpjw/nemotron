# Book Ramsey exploratory search

The [April 2026 Ramsey survey](https://www.combinatorics.org/ojs/index.php/eljc/article/download/DS1/pdf/) lists \(27\le R(B_4,B_9)\le28\). A 27-vertex graph passing `book.py` would settle that one-step gap at 28. This search has **not** found such a graph or proved it impossible. A separate model study below repairs perturbed known 26-vertex examples; it does not resolve the 27-vertex question.

`book.py` checks a JSON certificate containing exactly `n` and sorted `red_edges`. It computes the largest red and blue book sizes and reports whether a coloring avoids the requested books. `search.py`, `regular_search.py`, `feasible_search.py`, and `block_search.py` perform bounded heuristic runs for the 27-vertex \(B_4/B_9\) target. `circulant_scan.py` exhausts one small structured family. `sat_search.py` builds an independent SAT encoding using the optional open-source `python-sat` package. See `PARITY_BOUND.md` for the 157–171 necessary edge range and `DEGREE_LEMMA.md` for an additional necessary condition in the regular slice.

The checked records under `search-results/` contain no valid 27-vertex certificate. Three 12-regular switch searches reached loss 109; three balanced block-circulant searches reached loss 108; three parity-constrained flip searches reached loss 112. Both 90-second SAT searches timed out. The ordinary degree-12 circulant family was exhausted at 1,716 graphs with no certificate. Saved candidates are exploratory evidence, not bounds on \(R(B_4,B_9)\).

The search uses Brendan McKay's ten [SRG(26,10,3,4) graph6 examples](https://users.cecs.anu.edu.au/~bdm/data/sr261034.g6) as calibration and some initial states. Download that file to `sr261034.g6` beside these scripts before running. The expected SHA-256 is `f0b4b4ad4507835703444c0f0990733e2f7cc31524cf91d04ba4a109f1fd52ac`.

Example:

```sh
python3 search.py --seconds 20 --steps 10000000 --out search-results
python3 regular_search.py --seconds 45 --out search-results
python3 feasible_search.py --seconds 30 --out search-results
python3 block_search.py --seconds 30 --out search-results
python3 circulant_scan.py
python3 book.py certificate.json --red-book 4 --blue-book 9
```

For the checker, `certificate.json` must contain exactly `n` and sorted `red_edges`. The search result files include extra metadata; extract those two fields to check one. The saved candidates in `search-results/` are **invalid** unless the verifier reports `"valid": true`. The `loss` field sums common-neighbor excesses; near-zero is not a certificate. The search is heuristic and its timeout does not establish nonexistence.

The SAT script requires `python-sat` in a local environment. For example, after installing it in a virtual environment, run `python sat_search.py --n 27 --seconds 90`. A timeout is inconclusive. A solver's `unsat` response should be independently checked with a proof log before making a mathematical upper-bound claim.

## Graph-repair study

The separate [v1 edit protocol](EDIT_PROTOCOL.md) freezes a supervised
single-edge repair experiment around the ten known 26-vertex graphs. Its
source, dataset, and adapter are distinct from the unsuccessful 27-vertex
search and the earlier graph-counterexample pilot. The [measured results](EDIT_RESULTS.md)
show a held-out mean exact loss reduction of +5.17 for the selected adapter,
versus −6.69 for the untouched base. An exploratory eight-step rollout
repaired 11/150 test starts, while five-pass CPU greedy search repaired
67/150 with many more verifier calls. The dataset generator, CPU controls,
model path, and exact evaluator are in `edit_*.py`; [runtime instructions](EDIT_RUNTIME.md)
document reproduction. The source graph6 file and generated examples stay
outside the repository.

The [v2 transfer protocol](TRANSFER_V2_PROTOCOL.md) evaluates the same frozen
adapter without retraining on a separate SRG(25,12,5,6) graph collection and
the B6/B6 task. The 25-vertex source and derived rows remain outside this
repository. See [data attribution](DATA_ATTRIBUTION.md) and the separate
[public-release candidate checklist](PUBLIC_RELEASE.md).
The [held-out transfer result](TRANSFER_V2_RESULTS.md) gives +3.86 mean
one-action exact-loss reduction on 150 starts, compared with −6.09 for the
untouched base. An exact twelve-candidate control reaches +11.07 and CPU
greedy search repairs 71/150; the adapter has no end-to-end search advantage.
