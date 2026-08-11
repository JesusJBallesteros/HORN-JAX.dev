# Testing

## Running

From anywhere in the repo:

```bash
pytest
```

That is all. `testpaths` and `addopts` in `pyproject.toml` supply the rest, so there is no
`-m pytest`, no path argument, no `cd` to the right directory. Output is `-v --durations=5` by
default: one line per test, plus the five slowest.

Common variations:

```bash
pytest tests/test_dynamics.py                       # one file
pytest tests/test_model.py::test_forward_shapes     # one test
pytest -k energy                                    # any test whose name matches
pytest -k "gradient and not model"                  # boolean name filter
pytest -x                                           # stop at the first failure
pytest --lf                                         # only the tests that failed last run
pytest -q                                           # quiet, when you want just the count
pytest --collect-only                               # list tests without running them
pytest -s                                           # let print() through (normally captured)
pytest | tee results/pytest.txt                     # keep a copy
```

**Verbosity is a running total, not a switch.** Each `-v` is +1 and each `-q` is −1, and
`addopts` already contributes `-v`. So from this repo's baseline:

| you type | net level | what you get |
|---|---|---|
| `pytest` | +1 | one line per test |
| `pytest -q` | 0 | dots, grouped by file |
| `pytest -qq` | −1 | just the final count |
| `pytest -vv` | +2 | per test, plus extra detail on failures |

This catches people out: `pytest -v` here nets to 0 and looks *less* verbose than the default.

## Why `python tests/test_dynamics.py` does nothing

A test file contains only `def test_*` functions and no top-level calls. Running it with
`python` imports the module, defines the functions, and exits — exit code 0, no output. Nothing
is broken; there is simply no code at module level asking to be run.

`pytest` is the thing that *finds* functions named `test_*` and calls them. `demo.py` behaves
differently only because it has statements at module level that execute on import.

## What is tested, and why these things

The suite is 21 tests in four files. The split is deliberate: physics in one place, plumbing in
another, so a failure tells you immediately which kind of problem you have.

### `tests/test_dynamics.py` — the physics (5)

Checks the model against closed-form solutions rather than against itself. An uncoupled,
unforced HORN unit is a textbook damped harmonic oscillator, so ground truth exists.

- **`test_matches_analytic_underdamped`** — trajectory against the exact solution.
- **`test_energy_conserved_when_undamped`** — at ζ→0 mechanical energy must not drift. This is
  the test that distinguishes semi-implicit from explicit Euler, and it is the single most
  valuable test in the repo. Swap the two integration lines in `core.py` and it fails loudly.
- **`test_overdamped_decays_without_oscillating`** — at ζ>1, decay with no zero crossing.
- **`test_heterogeneous_frequencies_are_independent`** — uncoupled units keep their own period,
  verified by FFT. This is the property the whole nested-frequency idea depends on.
- **`test_gradients_flow`** — reverse-mode autodiff survives the full `lax.scan`.

If an oscillator implementation is wrong it is almost always the integrator, and the first two
catch that.

### `tests/test_model.py` — the plumbing (8)

Shapes, batch independence, gradient reach into every parameter including `log_omega` and
`log_zeta`, the freeze mechanism, and the `usable_band` arithmetic.

### `tests/test_data.py` — the loader (4)

Constructs IDX files from raw bytes so the parsing path is genuinely exercised. It exists
because a header bug once shipped here: the version that validated the dtype byte against zero
rejected every valid MNIST file. No network access required.

### `tests/test_paths.py` — path anchoring (4)

Guards the rule that output goes to the repo, never the working directory. One test `chdir`s to
a temporary directory, reloads `horn.paths`, and asserts nothing moved and nothing was written.

## Writing a new test

Put it in the file matching its kind, name it `test_<what it checks>`, and make the name say
what is true when it passes. Prefer asserting against something external — an analytic solution,
a conservation law, a dimensional argument — over asserting against the code's own output, which
only tells you the code has not changed.

Two patterns worth reusing:

- **Build the wire format by hand** rather than mocking past it. `_idx_blob` in `test_data.py`
  constructs real IDX bytes. Mocking the parse is how the header bug survived.
- **Perturb one element and assert the others are unmoved.** `test_batch_is_independent` changes
  one sequence in a fixed-shape batch. The obvious alternative — run a batch of 4, then one of
  its rows alone, and compare — fails on GPU for a reason unrelated to correctness: float32
  matmuls default to TF32 on NVIDIA hardware and different batch shapes dispatch to different
  kernels. A test that trips on numerical noise is worse than no test.

## When something fails

```bash
pytest -x --lf          # rerun just the failures, stop at the first
pytest --tb=long        # full traceback
pytest --tb=short       # one frame
pytest -s -k thing      # let prints through for the test you care about
pytest --pdb            # drop into the debugger at the point of failure
```

`--lf` ("last failed") reads `.pytest_cache/`, which is gitignored. After a green run it falls
back to running everything, which is why it shows 21 passed rather than nothing.

Tolerances in the physics tests are set to the accuracy the integrator can actually deliver, not
to whatever made them pass. Semi-implicit Euler is first order, so error scales with `dt` — if a
tolerance failure appears after changing a timestep, that is the test working correctly.
