# Nemotron 26-vertex book-graph repair

A fresh adapter from the pinned Nemotron Mini 4B base was trained on 1,200
one-edge repair examples generated around eight known SRG(26,10,3,4)
graph6 records. One separate record generated 150 development starts; a
different record generated 150 held-out test starts. Each start was made by
3–8 random edge flips from a valid 26-vertex \(B_4/B_9\) coloring. The exact
teacher evaluated all 325 possible flips per training state. The adapter
was selected on development before a single one-action test comparison.

The primary measure is mean change in exact excess-page loss after one
proposed flip. Positive reduction is better; malformed output contributes
zero reduction. Every strict improvement count includes all 150 starts.

| Method | Development reduction; improvements | Test reduction; improvements | Candidate evaluations |
| --- | ---: | ---: | ---: |
| Untouched base | −7.12; 17/150 | −6.69; 19/150 | 1 for a valid action |
| Random edge | −8.76; 13/150 | −7.26; 17/150 | 1 |
| Random shown violating spine | −0.02; 56/150 | +0.05; 61/150 | 1 |
| Selected adapter, seed 20261016 | **+6.32; 108/150** | **+5.17; 105/150** | 1 for a valid action |
| Best of twelve shown spines, added control | +10.56; 139/150 | +10.85; 143/150 | 12 |

The held-out paired adapter-minus-base mean reduction was +11.86, with a
95% paired-task bootstrap interval [9.70, 13.97], 2,000 draws and seed 871.
The adapter was valid on 149/150 test starts. No one-step method reached a
valid final coloring. An independent set-based calculation replayed all 300
base and adapter held-out actions and matched the exact scores.

Under a separately declared exploratory eight-step rollout, the adapter
repaired 14/150 development and 11/150 test starts, averaging 4.83 and
4.61 candidate checks per start. Independent replay checked every trajectory
and all 25 exact certificates. Eight-check random walk and annealing found
0/150 on each split. Five-pass greedy search found 75/150 development and
67/150 test solutions, averaging about 1,447 and 1,495 checks. The model
uses fewer verifier calls but took 548 seconds for the development rollout
on an RTX 4090; all three 1,625-check CPU controls together ran locally in
29 seconds. Hardware differs, so this timing is a practical comparison.

Two additional fresh training seeds were scored on development only:
20261017 yielded +5.68 mean reduction and 102/150 improvements; 20261018
yielded +6.23 and 104/150. The test result concerns only the originally
selected seed. These figures do not give a multi-seed test interval.

This study supports useful local action selection on perturbations of known
26-vertex constructions. It does not settle the 27-vertex \(B_4/B_9\) gap,
show transfer to other graph orders or book sizes, or establish a speed
advantage over CPU search. See `EDIT_PROTOCOL.md`,
`EDIT_ROLLOUT_PROTOCOL.md`, `EDIT_REPLICATION_PROTOCOL.md`, and
`EDIT_RUNTIME.md` for the frozen design and reproduction commands. Adapters,
derived data, per-task predictions, and operational receipts are held in a
private local evidence archive, outside this source repository.
