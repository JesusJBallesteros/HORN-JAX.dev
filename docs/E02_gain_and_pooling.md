# E02: Input gain and readout pooling

**Question.** With a correct core, why did nothing train, and which pooling makes the
oscillator states legible to a linear readout?

**Diagnosis.** Steady-state response scales as `1/(2ζω²)`, so fast units are quiet units.
With a flat `W_in` the population produces states of order 1e-6, the softmax is uniform,
loss sits at exactly `ln(n_classes)` and gradients are ~1e-4. It looks exactly like a
learning-rate problem and is not.

**Fixes.**

1. Scale each row of `W_in` by `2ζω²` (`input_gain="normalised"`, now the default).
2. Pool with `rms`, not `mean`: a sinusoidal response time-averages to nearly zero, and the
   mean/rms ratio falls with frequency, so mean-pooling discards most of the signal from the
   units working hardest.

**Result.** Frequency discrimination, 3 classes, identical model except for pooling:

| pool | test acc |
|---|---|
| mean | 0.34 |
| last | 0.35 |
| **rms** | **1.00** |

![pooling ablation](../results/freqdisc_pooling_ablation.png)

**Caveat that became E05.** `rms` discards phase entirely, so this result is reachable by a
bank of bandpass filters and a power readout. It validates the plumbing, not the
architecture's claim to a richer state space.

**Reproduce.** Notebook 02 §1–4, or `horn.tasks.freq_batch` + `horn.training.train` with
`pool` in `{"mean", "rms", "last"}`.
