# E00: Measure the lever before pulling it

A methods note rather than an experiment. It records why one of the runs below had to be
discarded, and the check now run before any parameter sweep.

**What happened.** The first diversity grid on the biphase task (E05) treated "recurrence
present or absent" as the manipulated variable: 16 runs, seeds averaged. The manipulation
did nothing. Gain normalisation had been applied to the afferent weights and not to the
recurrent ones, leaving the external drive 6968 times stronger, and removing `W_rec`
altogether shifted the decoder's class scores by a relative 4e-4. Every pair of
conditions meant to differ returned bit-identical accuracies. The grid was not answering
the question, it was incapable of answering it, and the table it produced looked like an
ordinary set of results.

**The check.** Before sweeping a variable, confirm that it moves the output: one forward
pass with the manipulation applied, one without, and the relative change in the decoder's
class scores. Below roughly 1e-2 the sweep returns noise however many seeds it averages.
Two forward passes and a subtraction, against several hours of GPU time and a conclusion
about frequency diversity that would have been wrong.

**Why it is easy to miss.** A null result and a broken instrument produce the same table.
They differ only in a measurement the table does not contain: whether the manipulated
variable is mechanically connected to the readout at all. In an electrophysiology rig the
equivalent is a stimulus artefact check or a dead-channel test, run before believing an
absence of response. The same discipline applies to simulations, which fail more quietly:
nothing crashes, no warning appears, the accuracies simply stop depending on the thing
being varied.

Three instances of the same pattern occurred here. A flat `W_in` made every task
untrainable while presenting as a learning-rate problem (E02). A ridge estimator that
scaled its own bias column deflated accuracies on perfectly separable activity (E05,
provenance note). The gain asymmetry above made a recurrent network feedforward. Each fix
was one line; finding the line was the work.

**Current practice.** `test_drive_balance_and_recurrence_leverage` pins the afferent to
recurrent drive ratio and the leverage of `W_rec`, so the regression to feedforward cannot
recur silently. New sweeps compute the corresponding one-pass check before the loop, and
the runner reports it (`leverage` column in
`results/probe_mechanism_rec_flat_vs_norm.txt`).
