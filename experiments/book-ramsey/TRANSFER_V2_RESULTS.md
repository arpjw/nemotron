# Cross-task graph-repair transfer: K25, B6/B6

The selected v1 B4/B9 adapter was evaluated without retraining on a
separate 25-vertex SRG(25,12,5,6) graph collection. The fifteen source
graph6 records were split before relabeling: eight generated 150 development
starts and seven generated 150 held-out test starts. Each start received
3–8 random edge flips from a valid B6/B6 coloring. Source and frozen data
hashes are in [the protocol](TRANSFER_V2_PROTOCOL.md) and local manifest.

The primary outcome is mean reduction in exact excess-page loss after one
action. Malformed output contributes zero reduction. A positive number is
better; strict improvement fractions use all 150 starts. The model used the
same pinned base, selected v1 adapter, system prompt, greedy decoder, and
32-token limit, changing only the explicit task facts in the user prompt.

| One-action method | Development reduction; improvements | Held-out test reduction; improvements | Exact candidate checks |
| --- | ---: | ---: | ---: |
| Untouched Nemotron | −4.93; 24/150 | −6.09; 21/150 | 1 for valid output |
| Random edge | −8.82; 8/150 | −8.51; 6/150 | 1 |
| Random visible spine | +0.40; 71/150 | +0.39; 64/150 | 1 |
| Selected v1 adapter | **+2.60; 85/150** | **+3.86; 104/150** | 1 for valid output |
| Best of twelve visible spines | +10.18; 141/150 | +11.07; 144/150 | 12 |
| Exhaustive one-flip teacher | +13.33; 150/150 | +13.33; 150/150 | 300 |

The adapter produced valid actions on 149/150 development and 150/150 test
starts, versus 131/150 and 134/150 for the untouched base. On test, the
paired adapter-minus-base mean reduction was +9.95. The 2,000-draw paired
task bootstrap interval was [7.91, 11.96], seed 871. A separate bootstrap
resampling the seven source graph6 records as clusters gave [9.21, 10.84],
seed 872. Both intervals describe this frozen source collection and one
selected training seed, not arbitrary graph families. All seven test source
records had positive adapter-minus-base mean reductions. No one-action
method completely repaired a start, consistent with the 3–8 flip setup.

The five-pass CPU greedy comparator found 76/150 development and 71/150
test valid colorings, averaging 1,356 candidate checks per start in each
split. Exact certificates for all 147 successes were saved privately and
checked with `book.py`. The complete test CPU greedy run took 8.0 seconds
locally; the adapter's one-action test inference took 127.1 seconds on a
remote RTX 4090. These are different machines and different action budgets,
so the times are practical observations, not a normalized speed ratio.
They do not support an end-to-end speed advantage for the model.

Independent set-based replay matched all 600 development and test base and
adapter action scores. Four inference manifests each recorded 150 completed
predictions. The held-out report SHA-256 is
`8fb092df8e86b264a2546543961947fbae4e13876a1f2e294ea09532fad3cc2c`;
the test data SHA-256 is
`9b73fd1359160e89d91bff3386c697eaaecd2741a371e980dd3c18e65a9b02ae`.
The selected v1 adapter SHA-256 is
`606c4efd0a33a808399daaab7a580ad3253181d1c65cece5667083153104e2cf`.
Per-state predictions, derived data, and certificates remain outside Git.

This is evidence of local action-selection transfer across one source,
order, and book-parameter change. It does not show broad transfer,
competitive graph search, a new Ramsey bound, or reliable repair from
arbitrary starts. The adapter was distilled from exhaustive one-step
labels in v1, and the exact twelve-candidate control remains much stronger
on the new task. The original v1 test was not reopened.
