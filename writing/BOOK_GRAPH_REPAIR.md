# I taught Nemotron to flip graph edges. A small search still wins.

I wanted to see whether a 4B language model could learn a useful move in a
combinatorial search problem. I gave it a red and blue coloring of a complete
graph, told it which edges violated a book-Ramsey constraint, and asked for
one edge to flip. An exact checker scored the move. That made a plausible
looking answer irrelevant unless it changed the graph in the right way.

A book \(B_k\) is \(k\) triangles sharing one edge. On 26 vertices, I worked
with colorings that avoid red \(B_4\) and blue \(B_9\). The graph is small
enough that I can try all 325 possible single-edge flips, but large enough
that a random flip usually makes the constraints worse. Ten known strongly
regular graphs from [Brendan McKay's collection](https://users.cecs.anu.edu.au/~bdm/data/graphs.html)
provided valid starting constructions. I randomly relabeled them, flipped
3–8 edges, and trained the model to choose the edge with the lowest exact
next-step violation loss.

I started from the pinned Nemotron Mini 4B Instruct base and trained one
rank-16 QLoRA adapter for one epoch on 1,200 examples. Eight source graphs
generated training states, one generated development states, and the last
generated 150 held-out test states. The split happened before relabeling.
The adapter was selected on development before the one-action held-out run.

On the 26-vertex test, the adapter's proposed flip reduced exact loss by
5.17 on average and improved 105/150 starts. Untouched Nemotron increased
loss by 6.69 on average and improved 19/150. In an exploratory eight-step
rollout, the adapter repaired 11/150 starts. A five-pass CPU greedy search
repaired 67/150. The model used far fewer exact candidate checks, but its
inference took much longer in the practical timing comparison. The
[v1 protocol and report](../experiments/book-ramsey/EDIT_RESULTS.md) give the
full denominators and limits.

The more interesting question was whether the learned choice was tied to
that one setup. I froze a second test using a separate collection of fifteen
25-vertex strongly regular graphs. These are valid for red \(B_6\) and blue
\(B_6\). I changed the explicit order and book parameters in the prompt,
kept the same adapter, and did no additional training. Eight source graphs
generated 150 development starts; seven different graphs generated 150
held-out starts.

On that new held-out task, the adapter reduced exact loss by 3.86 on average
and improved 104/150 starts. Untouched Nemotron increased loss by 6.09 and
improved 21/150. The paired adapter-minus-base reduction was 9.95, with a
95% interval of [7.91, 11.96] from resampling tasks. Resampling the seven
source graphs as clusters gave [9.21, 10.84]. Those intervals describe this
collection and the selected training seed, not arbitrary graph problems.
All 600 base and adapter predictions across development and test were
independently replayed with a separate set-based loss calculation.

The cheap controls set the ceiling on the practical claim. Checking the
twelve largest violating edges and taking the best one gave +11.07 mean
one-flip loss reduction on the transfer test. Five-pass CPU greedy search
repaired 71/150 starts, and each repair has an exact certificate. The
adapter did not completely repair any test graph in one move, as expected
from starts made by several flips. The full [transfer result](../experiments/book-ramsey/TRANSFER_V2_RESULTS.md)
reports all controls, source checksums, inference settings, and timing.

I think the result is a narrow but real transfer of *local action selection*.
It is consistent with a reusable repair heuristic, although the experiment
cannot identify the model's internal rule. It does not make the model a
competitive graph-search engine. The training labels came from exhaustive
one-step search, and the exact search controls remain stronger on these
small graphs. Nor is this a new Ramsey result: all valid constructions used
or repaired here have 25 or 26 vertices, and the separate 27-vertex
\(B_4/B_9\) gap remains open in this project.

The [source and protocols](../experiments/book-ramsey/README.md) document
how to regenerate the tasks and repeat the training and evaluation. The
exact selected adapter and per-state evidence remain private, so this
source release alone cannot replay the reported model scores byte for byte.
The raw graph6 files are external. [Data attribution](../experiments/book-ramsey/DATA_ATTRIBUTION.md)
and model terms are separate from the license for this project's code.
