# E04: Frozen vs learned oscillator constants

**Question.** Do learned ω and ζ beat a fixed, well-initialised bank, and if so, where do
they move?

**Setup.** Row-wise sMNIST as in E03, two conditions identical except that one zeroes the
gradient on `log_omega`/`log_zeta` (`freeze_oscillators`), holding the bank at its
initialisation.

**Result.**

| condition | test acc |
|---|---|
| frozen ω, ζ | 0.8715 |
| learned ω, ζ | **0.8971** (+0.026) |

![frozen vs learned](../results/smnist_row_frozen_vs_learned.png)

Where the bank went is more informative than the accuracy gap:

- **ω moves down, a lot.** 3.57–10.0 Hz at init → 0.58–7.23 Hz after training; median
  |Δf|/f = 38%, max 84%. The task wants slower units than the usable-band heuristic
  provides: gradient descent buys memory by lowering frequency.
- **ζ collapses by an order of magnitude.** 0.15 → 0.007–0.061. The network trades
  amplitude normalisation for memory horizon, which is exactly the two-jobs-one-knob
  conflict identified in the findings: ζ sets memory in cycles *and* sets gain.

**Reproduce.** Notebook 02 §8, or `horn.training.train(freeze_osc=True/False)` on the
row-wise task.
