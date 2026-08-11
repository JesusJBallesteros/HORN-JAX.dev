# Project context

Notes for whoever (or whatever) picks this up next. Committed to the repo so it travels
with the code rather than living in a chat session on one machine.

## What this is

A from-scratch JAX implementation of Harmonic Oscillator Recurrent Networks (HORN), built
as a code sample for an application to Natural Intelligence GmbH (NISYS, Frankfurt —
Felix Effenberger and Wolf Singer), whose architecture is based on HORN. The application
call closes early September 2026.

Reference: Effenberger et al., *An analog-electronic implementation of a harmonic
oscillator recurrent network*, arXiv:2509.04064. Reference implementation: `brainmass`.

## Conventions that matter

- **`core.py` works in rad/s.** That is the form of the ODE. Everything user-facing is in
  **Hz**, converted at the boundary by `w_rads()`, `make_units()`, `drive_sine()`.
  `w = 2*pi*f`. Getting this wrong is a factor of 6.28 in every timescale.
- **Integration is semi-implicit (symplectic) Euler.** Velocity updates first, position
  uses the *new* velocity. Explicit Euler injects energy and diverges at zeta=0;
  `test_energy_conserved_when_undamped` catches it if the two lines are ever swapped.
- **`omega` and `zeta` are stored as logs** so gradient descent cannot drive them negative.
  Negative damping is exponential blow-up.
- **Readout reads `(x, v/omega)`, not `(x, v)`.** Since v ~ omega*x, raw velocity is ~600x
  position at 100 Hz and would dominate the readout purely through units.

## Findings so far (notebook 01, then confirmed in 02)

1. **Amplitude collapse.** Steady-state response scales as `1/(2*zeta*omega^2)`, so fast
   units are quiet units. A heterogeneous bank is badly scale-mismatched at the readout
   before training starts. Per-band `zeta` compresses the spread (352x -> 158x across
   8-150 Hz) but only partially.
2. **Memory horizon is `1/(zeta*omega)`, not `1/zeta`.** Measured in *cycles* it is
   `1/(2*pi*zeta)` and depends on zeta alone. So zeta sets memory in cycles, omega sets it
   in seconds — and zeta is simultaneously the amplitude-normalisation knob, which is a
   conflict.
3. **`dt` is bounded by the fastest unit.** A learned omega drifting upward can destabilise
   the solver with no parameter looking obviously wrong.
4. **Tracking a moving input goes as `(zeta*f)^2`.** Slow, lightly damped units cannot keep
   up with a sweep and never reach steady state.
5. **Usable frequency ratio is `L/10`, set by sequence length alone.** Row-wise MNIST
   (28 steps, ratio 2.8) *cannot* represent a 1:6 nesting. Pixel-wise (784, ratio 78) can.
   The nested experiment needs a long-sequence task.

## Two things that had to be fixed before anything trained (notebook 02)

Both were predicted by finding #1 above, and both look like a learning-rate problem:

- **Input gain.** With a flat `W_in`, the population produces states of order 1e-6, the
  softmax is uniform, loss sits at exactly `ln(n_classes)` and gradients are ~1e-4. Fix:
  scale each row of `W_in` by `2*zeta*omega^2` (`input_gain="normalised"`, the default).
- **Pooling.** A sinusoidal response time-averages to ~zero, and the mean/rms ratio *falls*
  with frequency, so mean-pooling discards most from the units working hardest. On the
  frequency-discrimination task: mean 0.34, last 0.32, **rms 1.00**. Use `pool="rms"` for
  zero-mean oscillatory input, `"meanrms"` when there is a DC component (e.g. pixels).

## Layout

```
horn/core.py            dynamics: init_params, step, run_sequence, energy
horn/model.py           sequence model: init_net, forward, loss_and_acc, usable_band
tests/                  13 tests. test_dynamics = physics; test_model = plumbing
notebooks/01_test_core  validation against closed-form solutions
notebooks/02_sequence_training  readout, training loop, sMNIST
HORN_repo_plan.md       the research plan, incl. the nested-oscillation proposal
```

## State

- [x] Core dynamics, validated against analytic solutions (free decay, impulse, steady-state
      amplitude and phase, dt convergence)
- [x] Sequence layer + linear readout + training loop
- [x] Frequency discrimination trained to ceiling
- [ ] **Sequential MNIST — next. Needs a GPU box; download caches to `data/mnist.npz`.**
- [ ] Frozen vs learned (omega, zeta) — wired up, needs real data to say anything
- [ ] Stacked layers
- [ ] Nested banded omega with cross-frequency modulation vs flat heterogeneity, matched
      parameters, on a long-sequence task
- [ ] Spiking readout: does phase coding degrade more gracefully than a continuous readout
      under quantisation of the hidden state?

## Gotchas

- MNIST download can fail behind restrictive networks; the loader falls back to synthetic
  data **loudly**. If you see the warning banner, the numbers are meaningless.
- Notebook 01 uses a simple `sys.path` insert and must be launched from `notebooks/`.
  Notebook 02 searches upward for the repo and works from anywhere (and on Colab).
- The frequency-discrimination task at a 1-2000 Hz band over 1 s implies L = 20,000 steps,
  which is 25x longer than pixel-wise MNIST. Watch BPTT memory; raise `dt` if it bites.
- Set `git config core.autocrlf input` (or use the `.gitattributes`) when moving between
  Windows and WSL, or every file shows as fully modified.
