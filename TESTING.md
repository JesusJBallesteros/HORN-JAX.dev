# Testing

## Running

From anywhere in the repo:

```bash
pytest
```

That is all. `testpaths` and `addopts` in `pyproject.toml` supply the rest. Output is `-v --durations=5` by
default: one line per test, plus the five slowest.

Common variations:

```bash
pytest tests/test_dynamics.py                       # one file
pytest tests/test_model.py::test_forward_shapes     # one test
pytest -k shapes                                    # any test whose name matches
pytest -x                                           # stop at the first failure
pytest --lf                                         # only the tests that failed last run
pytest -q                                           # quiet, just the count
pytest --collect-only                               # list tests without running them
pytest -s                                           # let print() through (normally captured)
pytest | tee results/pytest.txt                     # keep a copy
```

**Verbosity.** Each `-v` is +1 and each `-q` is −1, and
`addopts` already contributes `-v`. So from this repo's baseline:

| typed | net level | output |
|---|---|---|
| `pytest` | +1 | one line per test |
| `pytest -q` | 0 | dots, grouped by file |
| `pytest -qq` | −1 | just the final count |
| `pytest -vv` | +2 | per test, plus extra detail on failures |

## What is tested, and why these things

The suite is 32 tests in five files: physics in one place, plumbing in
another, etc, so a failure points immediately at which kind of problem it is.

### `/test_dynamics.py`: the physics (5 tests)

Checks the model against closed-form solutions rather than against itself. An uncoupled,
unforced HORN unit is a textbook damped harmonic oscillator, so ground truth exists.

- **`test_matches_analytic_underdamped`**: trajectory against the exact solution.
- **`test_energy_conserved_when_undamped`**: at ζ→0 mechanical energy must not drift. This is
  the test that distinguishes semi-implicit from explicit Euler, and it is the single most
  valuable test in the repo. Swap the two integration lines in `core.py` and it fails loudly.
- **`test_overdamped_decays_without_oscillating`**: at ζ>1, decay with no zero crossing.
- **`test_heterogeneous_frequencies_are_independent`**: uncoupled units keep their own period,
  verified by FFT. This is the property the whole nested-frequency idea depends on.
- **`test_gradients_flow`**: reverse-mode autodiff survives the full `lax.scan`.

If an oscillator implementation is wrong it is almost always the integrator, and the first two
catch that.

### `/test_model.py`: the plumbing (10)

Shapes, batch independence, gradient reach into every parameter including `log_omega` and
`log_zeta`, the freeze mechanism, and the `usable_band` arithmetic. Two guards were added
with the rec-gain fix: `test_drive_balance_and_recurrence_leverage` pins the external-to-
recurrent drive ratio and the leverage of `W_rec` on the decoder's class scores, so the
network cannot silently regress to feedforward (the E00 lesson), and
`test_rec_gain_rejects_unknown_values` closes the config surface.

### `/test_tasks.py`: task construction (7)

Guards the properties the biphase experiment rests on, so a change to the task cannot silently
invalidate the result: class power spectra are matched, the biphase survives a global time
shift, the global phase carries nothing, classes are actually distinguishable, homogeneous and
heterogeneous conditions have matched parameter counts, and additive noise cannot leak the
label. Plus one check that `freq_batch` still works.

### `/test_data.py`: the loader (6)

Constructs IDX files from raw bytes so the parsing path is genuinely exercised (a header
bug once shipped here). Cache validation gets three tests: stale layouts are rejected, a
stale cache is deleted rather than half-read, and the cached path works offline with
train-statistics standardisation.

### `/test_paths.py`: path anchoring (4)

Guards the rule that output goes to the repo, never the working directory. One test
`chdir`s to a temporary directory, reloads `horn.paths`, and asserts nothing moved and
nothing was written.

## Writing a new test

Put it in the file matching its kind, name it `test_<what it checks>`, and make the name say
what is true when it passes. Prefer asserting against something external (an analytic solution,
a conservation law, a dimensional argument) over asserting against the code's own output, which
only shows the code has not changed.

Two patterns worth reusing:

- **Assert a property, not an output.** `test_power_spectrum_is_matched_across_classes` checks
  the *construction* the biphase result depends on, so a change to the task cannot silently
  invalidate the experiment downstream of it.
- **Perturb one element and assert the others are unmoved.** `test_batch_is_independent` changes
  one sequence in a fixed-shape batch. The obvious alternative (run a batch of 4, then one of
  its rows alone, and compare) fails on GPU for a reason unrelated to correctness: float32
  matmuls default to TF32 on NVIDIA hardware and different batch shapes dispatch to different
  kernels. A test that trips on numerical noise is worse than no test.

## When something fails

```bash
pytest -x --lf          # rerun just the failures, stop at the first
pytest --tb=long        # full traceback
pytest --tb=short       # one frame
pytest -s -k thing      # let prints through for a single test
pytest --pdb            # drop into the debugger at the point of failure
```

`--lf` ("last failed") reads `.pytest_cache/`, which is gitignored. After a green run it falls
back to running everything, which is why it shows all 32 rather than nothing.

Tolerances in the physics tests are set to the accuracy the integrator can actually deliver, not
to whatever made them pass. Semi-implicit Euler is first order, so error scales with `dt`. If a
tolerance failure appears after changing a timestep, that is the test working correctly.
