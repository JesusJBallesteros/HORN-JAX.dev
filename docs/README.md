# Experiment log

One page per experiment: the question, the result, and the command or notebook cell that
produced it. Figures live in `results/`; these pages keep them interpretable once the run
that made them has scrolled away.

| # | Experiment | Question | Status | Result |
|---|---|---|---|---|
| E00 | [Measure the lever](E00_measure_the_lever.md) | How to tell a null result from a broken instrument? | methods note | one forward pass per manipulated variable, before any sweep |
| E01 | [Core validation](E01_core_validation.md) | Does the integrator match the closed-form physics? | done | 5/5 physics tests pass; energy flat as ζ→0 |
| E02 | [Gain and readout statistic](E02_gain_and_pooling.md) | Why did nothing train, and what should the decoder read? | done | `W_in` scaled by 2ζω²; time-mean 0.34, final 0.35, **rms 1.00** |
| E03 | [Sequential MNIST](E03_smnist.md) | Where does a single-layer population land? | done | row-wise **0.897**, pixel-wise **0.794** |
| E04 | [Frozen vs fitted ω, ζ](E04_frozen_vs_learned.md) | Do fitted oscillator constants beat a fixed bank? | done | 0.897 vs 0.872; ω down 38% median, ζ collapses 0.15 → 0.007-0.061 |
| E05 | [Biphase](E05_biphase.md) | Is there a task where a power readout must fail? | separability done under both gain conventions; fitted grid inconclusive at pilot scale | at initialisation `W_rec=0` at chance everywhere; gain fix raised recurrence leverage 3.6e-3 → 0.37 |
| E06 | [Readout precision](E06_readout_precision.md) | Does the analog readout collapse reproduce, and does the task's code matter? | done, both tasks | sMNIST: collapse and recovery reproduced, 0.31 → 0.70 at 3 bits. Biphase decoder needs 14 bits against sMNIST's 6, while its information survives lower precision |
| E07 | Nested bands vs flat heterogeneity | Does banding at a fixed ratio with slow-phase gating beat unstructured frequencies? | designed, not started | conditions A-D at matched parameter count |
| E08 | Spiking readout under quantisation | Does a spike-timing code read the surviving information more robustly? | designed, not started | E06 is the baseline it has to beat |

## Conventions for these pages

- Figures in `results/` are written by a script or notebook through
  `horn.paths.results(...)`, never a relative path.
- A result is recorded with the command, the seed and the number, or it is not recorded.
- `results/pytest.txt` holds the last full suite run.
