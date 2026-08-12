# E03: Sequential MNIST

**Question.** Where does a single-layer HORN with a linear readout land on sMNIST, row-wise
and pixel-wise?

**Setup.** 96 oscillators (row-wise) / 128 (pixel-wise), log-spaced frequency bank inside the
usable band, ζ=0.15, `pool="meanrms"` (pixels have a DC component), Adam. The usable band is
set by sequence length alone (ratio ≈ L/10):

| task | L | dt | usable band | ratio | 1:6 nesting? |
|---|---|---|---|---|---|
| row-wise | 28 | 10 ms | 3.6–10.0 Hz | 2.8 | does not fit |
| pixel-wise | 784 | 1.28 ms | 1.0–78.4 Hz | 78 | fits |

**Result.**

| task | test acc | wall time (CPU/GPU box) |
|---|---|---|
| row-wise | **0.897** | 28 s |
| pixel-wise | **0.794** | 22 min |

![training curves](../results/smnist_training.png)

Row-wise trains fast and cleanly. Pixel-wise is noisier: gradient norms grow toward the end
of training, which is the documented stiffness of oscillatory BPTT and the first candidate
for clipping/schedule work.

**Not claimed.** No comparison against published HORN numbers is made: this is one small
single-layer model, no stacking, no tuning beyond the defaults, and published results use
different sizes and budgets. The number is a baseline for the repo's own ablations
(E04, E06), not a leaderboard entry.

**Reproduce.** Notebook 02 §5–7. MNIST downloads once to `data/mnist.npz`; the loader raises
rather than substituting synthetic data.
