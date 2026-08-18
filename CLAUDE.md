# Project context

Notes for whoever (or whatever) picks this up next. Committed to the repo so it travels
with the code rather than living in a chat session on one machine.

## What this is

A from-scratch JAX implementation of Harmonic Oscillator Recurrent Networks (HORN), built
as a code sample for an application to Natural Intelligence GmbH (NISYS, Frankfurt:
Felix Effenberger and Wolf Singer), whose architecture is based on HORN.

Reference: Effenberger et al., *An analog-electronic implementation of a harmonic
oscillator recurrent network*, arXiv:2509.04064. Reference implementation: `brainmass`.

## Conventions that matter

- **`core.py` works in rad/s.** That is the form of the ODE. Everything user-facing is in
  **Hz**, converted at the boundary by `w_rads()`, `make_units()`, `drive_sine()`.
  `w = 2*pi*f`. Getting this wrong is a factor of 6.28 in every timescale.
- **Where the tanh sits is a model choice, and it is a flag.** `drive="output"`
  (default) is `f = W_in u + W_rec tanh(x)`; `drive="input"` is
  `f = eps*tanh(W_rec x + W_in u + b)`, the reference model. They are different models,
  not different spellings: under "output" an uncoupled unit is a LINEAR filter, under
  "input" it is already nonlinear. The whole E05 falsifying control depends on the
  first. Default stays "output" because that clean null is the better instrument; see
  the header of `core.py` for the argument and docs/E05 for the measurement.
- **Integration is semi-implicit (symplectic) Euler.** Velocity updates first, position
  uses the *new* velocity. Explicit Euler injects energy and diverges at zeta=0;
  `test_energy_conserved_when_undamped` catches it if the two lines are ever swapped.
- **The solver's hard limit is `dt*omega < 2`,** not `2*pi/omega`. Measured: 1.99 runs,
  2.01 gives NaN. Nothing guards a *learned* omega drifting across it. Well below the
  limit is still wanted: position amplitude is inflated by `1/sqrt(1-(dt*omega/2)^2)` and
  resonance sits at `arcsin(dt*omega/2)/(pi*dt)`, so at ten samples per period a unit is
  5.3% too loud and 1.5% too fast. Both depend on omega, so a heterogeneous bank is
  biased unevenly across itself.
- **`omega` and `zeta` are stored as logs** so gradient descent cannot drive them negative.
  Negative damping is exponential blow-up.
- **Readout reads `(x, v/omega)`, not `(x, v)`.** Since v ~ omega*x, raw velocity is ~600x
  position at 100 Hz and would dominate the readout purely through units.
- **Gain normalisation must be applied to every drive, not just the external one.** The
  factor `2*zeta*omega^2` exists so that a drive produces an O(1) response. Recurrent input
  is a drive. Applying it to `W_in` alone leaves the network feedforward in all but name -
  see `rec_gain` in `init_net` and the State entry below. Applying it to BOTH is one
  per-unit gain on the whole drive, which is what the paper calls excitability `eps`;
  under `drive="input"` it is spelled that way, and it must sit outside the tanh because
  a gain inside would saturate rather than scale.
- **Before sweeping over a variable, measure that it moves the output.** One forward pass
  with the variable switched off, compared against one with it on. A relative change below
  ~1e-2 means the sweep will return noise no matter how many seeds it averages.

## Findings so far (notebook 01, then confirmed in 02)

1. **Amplitude collapse.** Steady-state response scales as `1/(2*zeta*omega^2)`, so fast
   units are quiet units. A heterogeneous bank is badly scale-mismatched at the readout
   before training starts. Per-band `zeta` compresses the spread (352x -> 158x across
   8-150 Hz) but only partially.
2. **Memory horizon is `1/(zeta*omega)`, not `1/zeta`.** Measured in *cycles* it is
   `1/(2*pi*zeta)` and depends on zeta alone. So zeta sets memory in cycles, omega sets it
   in seconds, and zeta is simultaneously the amplitude-normalisation knob, which is a
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
  decoder's class probabilities are uniform, loss sits at exactly `ln(n_classes)`,
  the value of pure guessing, and gradients are ~1e-4. Fix:
  scale each row of `W_in` by `2*zeta*omega^2` (`input_gain="normalised"`, the default).
- **Pooling.** A sinusoidal response time-averages to ~zero, and the mean/rms ratio *falls*
  with frequency, so mean-pooling discards most from the units working hardest. On the
  frequency-discrimination task: mean 0.34, last 0.32, **rms 1.00**. Use `pool="rms"` for
  zero-mean oscillatory input, `"meanrms"` when there is a DC component (e.g. pixels).

## Layout

```
horn/core.py            dynamics: init_params, step(drive=...), run_sequence, energy
horn/model.py           sequence model: init_net, forward, loss_and_acc, usable_band
horn/tasks.py           freq_batch (plumbing check), biphase_batch (the phase question)
horn/training.py        train / evaluate, freeze_osc and freeze_rec controls
horn/data.py            MNIST: IDX parsing, npz cache. Raises rather than faking data.
horn/paths.py           REPO / DATA_DIR / RESULTS_DIR, anchored to the package
horn/report.py          Report(): tees stdout to results/<name>.txt and saves matching
                        .json/.png with a provenance header (timestamp + commit hash)
experiments/            probe_mechanism (separability at init, --drive), run_diversity (the grid),
                        readout_precision (in-loop vs sampled quantisation, E06)
tests/                  34 tests. test_dynamics = physics; test_model = plumbing;
                        test_tasks = task construction. See TESTING.md.
docs/                   experiment log: E00 (methodology), E01-E06 (run), E07-E08
                        (designed, not started), reading.md (annotated bibliography)
notebooks/01_test_core  validation against closed-form solutions
notebooks/02_sequence_training  readout, training loop, sMNIST
results/                curated figures and run records, committed on purpose
HORN_repo_plan.md       private planning notes (gitignored). The parts worth keeping are
                        now in docs/E07, docs/E08 and docs/reading.md
```

## State

- [x] Core dynamics, validated against analytic solutions (free decay, impulse, steady-state
      amplitude and phase, dt convergence)
- [x] Sequence layer + linear readout + training loop
- [x] Frequency discrimination trained to ceiling
- [x] Sequential MNIST: row-wise 0.897 (96 osc), pixel-wise 0.794 (128 osc). See docs/E03.
- [x] Frozen vs learned (omega, zeta): 0.872 vs 0.897; omega migrates down 38% median,
      zeta collapses 0.15 -> 0.007-0.061. See docs/E04.
- [x] Biphase task built + probed at init: W_rec=0 at chance everywhere, recurrence +
      engaged tanh -> 1.00. See docs/E05 and horn/tasks.py. NOTE: probe ridge had a
      normalisation bug (scaled bias column); fixed, table regenerated, conclusions held.
- [x] Readout precision (docs/E06): in-loop quantisation, info survives to 3 bits with a
      retrained ridge, float readout collapses <14 bits, agreement bottoms at 0.26 vs the
      analog paper's 0.28. Sampled-only quantisation is nearly harmless.
- [x] Readout precision on sMNIST: float32 0.861, agreement bottoms at 0.13 and passes
      through 0.30 at 3 bits against the analog paper's 0.284; retrained ridge lifts 3-bit
      accuracy 0.308 -> 0.696. Results in `results/readout_precision_smnist.{txt,json,png}`.
      Written into docs/E06 beside the biphase run, stated as the E08-hypothesis inversion.
- [x] **`rec_gain` fix.** `input_gain="normalised"` multiplied W_in by `2*zeta*omega^2` and
      nothing multiplied W_rec, so external drive outweighed recurrent drive ~5000:1 and the
      network was effectively FEEDFORWARD. Zeroing W_rec changed the decoder's class scores
      by a relative 4e-4, so
      "recurrence on/off" was an inert variable and the first biphase grid returned
      bit-identical accuracies for conditions meant to differ. `init_net(rec_gain=...)` now
      offers "flat" (default, reproduces the old runs) and "normalised" (same factor applied
      to W_rec). Measured: drive ratio 6968:1 -> 8:1, recurrence leverage 3.6e-3 -> 0.37.
      Guarded by `test_drive_balance_and_recurrence_leverage`.
- [x] Biphase grid re-run under both gains, artefacts named by condition:
      `results/diversity_biphase_n24_L200_recflat.*` and `..._recnormalised.*`, plus
      `results/probe_mechanism_rec_flat_vs_norm.{txt,json,png}`. Interpreted into docs/E05
      (gain-bug section) and docs/E00 (the methodology lesson, now its own page).
- [x] **Drive placement made selectable, and E05 rescoped.** The reference model
      (Effenberger PNAS, Materials and Methods) squashes the summed drive,
      `eps*tanh(W_rec x + W_in u + b)`; this repo squashes the recurrent output,
      `W_in u + W_rec tanh(x)`. The difference decides what an uncoupled unit is, and
      E05's falsifying control rested on it without saying so: under the reference
      placement `W_rec = 0` scores 0.757-0.795, not chance, because `tanh(W_in u)`
      supplies the cubic cross-term by itself. Sharpest at `w_scale=0.1`, where the
      state-side `nonlin` diagnostic reads 2.2e-03 - the tanh on the state is doing
      nothing and the uncoupled bank still reaches 0.757.
      `core.step(drive=...)` now selects, `HORNParams` gained `log_eps` and `bias`
      (None under "output", so that pytree is unchanged and old runs reproduce
      bit-for-bit), and the flag is threaded through `model.forward`,
      `training.train/evaluate_*` and `probe_mechanism.py --drive`. Guarded by
      `test_drive_placement_decides_linearity` (superposition, the real distinction)
      and `test_drive_placement_changes_the_model` (plumbing + leaf count). Records:
      `results/probe_mechanism_input_rec_normalised.*` and
      `..._output_rec_flat_vs_norm.*`, the latter identical to the pre-flag table.
      Written up in docs/E05, "Which model the control belongs to".
- [ ] Stacked layers
- [ ] Nested banded omega with cross-frequency modulation vs flat heterogeneity, matched
      parameters, on a long-sequence task (E07 in docs/, designed but not started)
- [ ] Spiking readout: does phase coding degrade more gracefully than a continuous readout?
      (E08 in docs/; E06 is the continuous baseline it must beat)

## Next in line, explicitly

1. **The properly sized diversity grid, when GPU time is cheap.** The remaining gap in
   E05: ridge on standardised features at init reaches 1.000, the trained readout at
   pilot scale sits at chance. Before or with the scale-up, condition the readout the way
   the probe conditions its features: standardise the pooled features (a norm before the
   affine readout in `model.forward`, or z-scoring inside `_pool`), since that mismatch,
   not representation, is the suspected gap. Then:

       python experiments/run_diversity.py --n-osc 64 --steps 800 --seeds 5
       python experiments/run_diversity.py --n-osc 64 --steps 800 --seeds 5 --zeta 0.02

   under `rec_gain="normalised"` only (flat is the documented broken instrument). If
   heterogeneous still does not separate from homogeneous with a conditioned readout at
   real scale, that IS reportable, as a constraint on the diversity claim at this network
   size, and docs/E05 already has the slot for it.

Stacked layers, the nested architecture and a spiking readout stay unstarted
deliberately. E07 and E08 are now written up as full designs in docs/, each with its
predictions and its falsifying outcome recorded before any run, which says more than a
half-built version would. They start after the grid above is settled, not before.

## The open tension

Everything *trained* so far is reachable with a bank of bandpass filters and a power readout,
so the headline claim that oscillators carry information in amplitude *and phase* needed a task
where power provably fails. That task now exists: the biphase (horn/tasks.py, docs/E05), whose
class-conditional power spectra are matched by construction. The at-init probe already delivered
the sharp result, with a correction to the prediction: the falsifying control is the
architecture (`W_rec = 0` at chance at every amplitude), not the pooling. An engaged tanh
converts biphase into internal power and rms climbs to 0.65. That control is a property of
`drive="output"`: it is a statement about linear filter banks, and under the reference
placement it does not exist. Scoped in docs/E05 and in the docstrings that asserted it
unqualified (`horn/tasks.py`, `experiments/run_diversity.py`).

What remains is the trained heterogeneous-vs-homogeneous grid (`run_diversity.py`), which asks
whether frequency diversity wins at matched parameter count on a task where it is a
precondition rather than a nice-to-have.

## Gotchas

- MNIST download can fail behind restrictive networks. The loader now **raises** rather than
  substituting synthetic data: a warning banner scrolls off, and a plausible accuracy number
  computed on noise is worse than a crash. If offline, drop `mnist.npz` into `data/`.
- **`ModuleNotFoundError: No module named 'horn'`** used to happen because VS Code's
  `jupyter.notebookFileRoot` defaults to the *workspace* folder, so with the workspace opened
  one level above the repo the package sat below cwd and no `sys.path` walk could find it.
  **Fixed properly:** the project is now installable. `pip install -e ".[dev,notebooks]"` from
  the repo root, and `import horn` resolves everywhere with no path manipulation. If it recurs,
  the kernel is on the wrong interpreter; check it points at `.venv/bin/python`.
- After adding a NEW module to `horn/`, no reinstall is needed (editable installs track the
  source tree). After changing `pyproject.toml` dependencies, reinstall.
- Notebook setup cells print the resolved `REPO`; glance at it the first time on any machine.
- **Never write output to a relative path.** Editable install means scripts run from anywhere,
  so `plt.savefig("x.png")` lands in whatever directory the shell was in. Use
  `from horn.paths import results` and `plt.savefig(results("x.png"))`. Same for the MNIST
  cache: `load_mnist()` with no argument defaults to `<repo>/data`, so it is downloaded once
  rather than once per directory you happen to launch from.
- The frequency-discrimination task at a 1-2000 Hz band over 1 s implies L = 20,000 steps,
  which is 25x longer than pixel-wise MNIST. Backpropagating gradients through that many
  timesteps is memory-hungry; raise `dt` if it bites.
- **The MNIST cache is version-stamped** (`CACHE_VERSION` in `data.py`). A `.npz` written by an
  older loader is detected and rebuilt rather than half-read (the first version of this failed
  with `KeyError: 'xtr'` from the middle of `load_mnist`). Bump the version whenever key names,
  dtype or scaling change.
- Set `git config core.autocrlf input` (or use the `.gitattributes`) when moving between
  Windows and WSL, or every file shows as fully modified.
- **Scale features BEFORE appending a bias column.** The ridge estimators in
  `probe_mechanism.py` and `readout_precision.py` once scaled a column of ones by its own
  (zero) spread, creating a 1e9 column that poisoned the normal equations and deflated
  accuracies on perfectly separable features. Fixed in both; the E05 table was regenerated
  (`results/probe_mechanism.txt` is the committed record). If a linear probe reports
  chance on features a trained readout classifies, suspect the estimator before the data.