# Experiment log

One page per experiment. Each page states the question, the result, and the exact command
or notebook cell that produced it. Figures live in `results/`; these pages are the index
that keeps them interpretable after the run that made them has scrolled away.

| # | Experiment | Question | Status | Result |
|---|---|---|---|---|
| E01 | [Core validation](E01_core_validation.md) | Does the integrator match the closed-form physics? | done | 5/5 physics tests pass; energy flat at ζ→0 |
| E02 | [Gain and pooling](E02_gain_and_pooling.md) | Why did nothing train, and which readout pooling works? | done | `W_in` scaling by 2ζω² + `rms` pooling; mean 0.34 / last 0.35 / **rms 1.00** |
| E03 | [Sequential MNIST](E03_smnist.md) | Where does a single-layer HORN land on sMNIST? | done | row-wise **0.897**, pixel-wise **0.794** |
| E04 | [Frozen vs learned ω, ζ](E04_frozen_vs_learned.md) | Do learned oscillator constants beat a fixed bank? | done | learned 0.897 vs frozen 0.872 (+0.026); ω migrates down 38% median |
| E05 | [Biphase / phase readout](E05_biphase.md) | Is there a task where power pooling provably loses? | probe done, trained grid pending | at init: `W_rec=0` at chance everywhere; recurrence + amplitude reach 1.00 |
| E06 | Nested bands vs flat heterogeneity | Does structured 1:6 banding + slow-phase gating beat a bag of frequencies? | planned | see `HORN_repo_plan.md`, conditions A–D |
| E07 | Spiking readout under quantisation | Does phase coding degrade more gracefully than a continuous readout? | planned | pending |

## Rules

- Every figure in `results/` is produced by a script or notebook that writes it via
  `horn.paths.results(...)`, never a relative path.
- A result is not a result until the page records the command, the seed, and the number.
- `results/pytest.txt` is the committed record of the last full suite run.
