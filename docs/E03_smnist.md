# E03: Sequential MNIST

**Question.** With a working input gain and readout, where does a single-layer population
land on sequential MNIST, presented row by row and pixel by pixel?

**Setup.** 96 oscillators row-wise, 128 pixel-wise, natural frequencies log-spaced across
the usable band, ζ = 0.15, decoder reading both the time-mean and the rms (`meanrms`,
since pixel intensities carry a DC component), fitted with Adam. The usable band follows
from sequence length alone, at a ratio of about L/10:

| presentation | L | dt | usable band | ratio | 1:6 nesting representable? |
|---|---|---|---|---|---|
| row-wise | 28 | 10 ms | 3.6-10.0 Hz | 2.8 | no |
| pixel-wise | 784 | 1.28 ms | 1.0-78.4 Hz | 78 | yes |

**Result.**

| presentation | test acc | wall time |
|---|---|---|
| row-wise | **0.897** | 28 s |
| pixel-wise | **0.794** | 22 min |

![training curves](../results/smnist_training.png)

Row-wise converges quickly and smoothly. Pixel-wise is noisier, and gradient magnitudes
grow toward the end of training, the expected stiffness of fitting through several
hundred timesteps of oscillatory dynamics. Gradient clipping and a learning-rate schedule
are the first things to try there.

**Not claimed.** These are not comparable to published HORN results: one small
single-layer population, defaults throughout, no stacking or tuning, and the published
work uses different sizes and training budgets. The numbers exist as an internal
reference for the manipulations in E04 and E06.

**Reproduce.** Notebook 02 §5-7. MNIST downloads once to `data/mnist.npz`
