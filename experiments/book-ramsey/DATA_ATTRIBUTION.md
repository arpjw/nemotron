# External graph data and derived examples

The graph-repair studies use Brendan McKay's graph6 collections:

- [SRG(26,10,3,4), ten graphs](https://users.cecs.anu.edu.au/~bdm/data/sr261034.g6), SHA-256 `f0b4b4ad4507835703444c0f0990733e2f7cc31524cf91d04ba4a109f1fd52ac`.
- [SRG(25,12,5,6), fifteen graphs](https://users.cecs.anu.edu.au/~bdm/data/sr251256.g6), SHA-256 `f55f86eb48cd61c3315c62341cc3a8c59289c58504edb3ab859d7ef2068b9aa4`.

McKay's [collection page](https://users.cecs.anu.edu.au/~bdm/data/)
states that its data files, except those indicated as compiled by someone
else, are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The graph page credits the strongly regular graph collection to McKay,
Ted Spence, and/or others. Credit the named source and retain this notice
when redistributing the files or derived records.

We generated derived repair examples by relabeling vertices, flipping
3–8 sampled edges, computing exact book violations, and labeling an
exhaustive best one-edge action. The source files and generated examples are
not included in this source repository. The scripts fetch neither file
automatically; users obtain the graph6 data from the linked collection and
verify the pinned checksums before generation.
