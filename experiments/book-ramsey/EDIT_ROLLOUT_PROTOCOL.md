# Exploratory eight-step graph-repair rollout

This secondary protocol was written after the one-action development result
and before any model rollout. It does not change the frozen v1 one-action
protocol or its held-out comparison. The same selected adapter, graph starts,
prompt, and greedy decoder are used. At most eight model proposals are made
per state. Every syntactically valid action is applied, even if it raises the
loss, and the next prompt contains newly computed exact feedback. A malformed
action or repeated graph state stops the trajectory. Report the best exact
loss and whether any visited coloring passes the independent book verifier.

The model uses one candidate-loss evaluation per valid action, plus the
independent check for any zero-loss certificate. Compare against the existing
random-walk and annealing controls at eight candidate evaluations. The
325- and 1,625-call CPU searches are stronger compute references. GPU
generation time and verifier calls must be reported separately.

First run all 150 development states. Run the held-out test rollout once only
if development finds at least one exact valid coloring. Any test rollout is
exploratory because the one-action test has already been opened. A failed
search does not prove the Ramsey gap impossible. Neither a successful
26-vertex repair nor a 27-vertex near miss settles the 27-vertex gap.
