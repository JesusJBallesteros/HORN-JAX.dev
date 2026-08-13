# E02: Input gain and the statistic the decoder reads

**Question.** With the dynamics validated, nothing would train. Why, and which summary of
the population response makes the label recoverable by a linear decoder?

**Diagnosis.** Steady-state response scales as `1/(2ζω²)`, so fast units are quiet units.
With a flat `W_in` the population reaches amplitudes of order 1e-6, the decoder's class
probabilities are uniform, the loss sits at exactly `ln(n_classes)`, the value of pure
guessing, and gradients are around 1e-4. The signature is indistinguishable from a
learning-rate problem, and no learning rate fixes it.

**Two changes.**

1. Scale each row of `W_in` by `2ζω²`, so a drive produces an O(1) response whatever the
   unit's frequency (`input_gain="normalised"`, now the default).
2. Summarise each unit's response by its rms rather than its time-average. A sinusoidal
   response averages to nearly zero, and the mean-to-rms ratio falls with frequency, so
   averaging over time discards most of the signal from the units responding most
   strongly.

**Result.** Frequency discrimination, 3 classes, identical models differing only in the
statistic handed to the decoder:

| statistic | test acc |
|---|---|
| time-mean | 0.34 |
| final state | 0.35 |
| **rms (response power)** | **1.00** |

![pooling ablation](../results/freqdisc_pooling_ablation.png)

**Limit of this result.** Response power discards phase, so a bank of bandpass filters
with a power readout would do as well. It establishes that the plumbing works, not that
the oscillatory state space is being used. Building a task where a power readout cannot
succeed became E05.

**Reproduce.** Notebook 02 §1-4, or `horn.tasks.freq_batch` with `horn.training.train`,
varying `pool` over `{"mean", "rms", "last"}`.
