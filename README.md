# HORN-JAX.dev

A from-scratch JAX implementation of Harmonic Oscillator Recurrent Networks (HORN).

Written to understand the architecture by building it rather than by reading about it. The
first goal was a correct core, validated against physics. The standing goal is to play with
nested oscillations — bands of frequencies in fixed ratios, with slow phase modulating fast
amplitude — and eventually with spiking readouts.

## What HORN is

**The unit.** A standard RNN unit holds a scalar hidden state pushed around by a sigmoid,
tanh, or a set of learned gates. A HORN unit is instead a damped, driven harmonic oscillator
with a two-dimensional state — position x and velocity ẋ — evolving according to

```math
x'' + 2ζω x' + ω² x = f(x, u)
f(x, u) = W_rec · tanh(x) + W_in · u
```

where ω is the unit's natural frequency and ζ its damping. Units are coupled through a learned
weight matrix, so f mixes the oscillator's own state with input drive and with the states of
every other oscillator in the population.

**What that buys.** In a gated RNN, memory is an engineering artifact — LSTMs need forget gates
because nothing in a tanh unit naturally persists. In an oscillator, persistence is free: an
underdamped oscillator rings. ζ sets how long, so it is a tunable memory horizon; ω sets which
input frequencies a unit responds to, so a population with varied ω is a filter bank.

## What this repo implements

- The core dynamics, integrated with semi-implicit (symplectic) Euler, validated against
  closed-form solutions
- A single-layer sequence model with a linear readout and four pooling modes
- A training loop, and the two fixes that were required before anything trained at all
- Frequency-discrimination as a warm-up task, trained to ceiling

**What it deliberately does not implement:** stacked layers, the nested-band architecture, the
spiking readout. Those are the interesting questions and they are listed as open below rather
than half-built.

## Layout

```text
horn/core.py                     dynamics: init_params, step, run_sequence, energy
horn/model.py                    init_net, forward, loss_and_acc, usable_band, freeze_oscillators
horn/data.py                     MNIST: IDX parsing, caching, no synthetic fallback
horn/paths.py                    repo-anchored paths, so output never lands in the cwd
tests/test_dynamics.py           5 tests — physics against closed-form solutions
tests/test_model.py              8 tests — shapes, gradients, plumbing
<<<<<<< HEAD
tests/test_data.py               4 tests — IDX format, caching, scaling
=======
tests/test_data.py               6 tests — IDX format, cache validation, scaling
>>>>>>> e3b3bd2 (Fixed staled data error and documented)
tests/test_paths.py              4 tests — path anchoring, no import side effects
notebooks/01_test_core.ipynb     validation against analytic solutions
notebooks/02_sequence_training.ipynb   readout, the two fixes, training, sMNIST
demo.py                          damping regimes + frequency bank -> results/demo.png
CLAUDE.md                        working context and findings, for whoever picks this up
SETUP.md                         WSL2 + CUDA + JAX environment setup
TESTING.md                       how to run the suite, and what each test is for
```

## Conventions that will bite you if you miss them

- **`core.py` works in rad/s; everything user-facing is in Hz**, converted at the boundary.
  Getting this wrong is a factor of 6.28 in every timescale.
- **ω and ζ are stored as logs**, so gradient descent cannot drive them negative. Negative
  damping is exponential blow-up.
- **The readout reads (x, v/ω), not (x, v).** Since v ~ ωx, raw velocity is ~600× position at
  100 Hz and would dominate the readout purely through units.
- **Integration order matters.** Velocity updates first; position uses the *new* velocity.
  Swap those two lines and you have explicit Euler, which injects energy and diverges at ζ=0.

## What I learned

Ordered roughly by how much they changed what I did next.

1. **Amplitude collapse, and the two fixes it forced.** Steady-state response scales as
   `1/(2ζω²)`, so fast units are quiet units. With a flat `W_in` the population produces states
   of order 1e-6, the softmax is uniform, the loss sits at exactly `ln(n_classes)` and gradients
   are ~1e-4. It looks exactly like a learning-rate problem and is not. Scaling each row of
   `W_in` by `2ζω²` fixes it. Separately, a sinusoidal response time-averages to nearly zero,
   and the mean/rms ratio *falls* with frequency — so mean-pooling discards most of the signal
   from the units working hardest. On frequency discrimination: mean 0.34, last 0.32, rms 1.00.

2. **The usable frequency ratio is set by sequence length alone**, at roughly `L/10`. A unit
   must complete at least one cycle within the sequence, and integration needs ~10 samples per
   period; those two constraints leave a ratio of `L/(min_cycles × steps_per_period)`. Row-wise
   MNIST (28 steps, ratio 2.8) therefore *cannot* represent a 1:6 nesting, whatever else you do
   to it. This ruled out a task before I wasted a week on it.

3. **ζ sets memory in cycles; ω sets it in seconds.** The envelope time constant is `1/(ζω)`,
   but measured in *cycles* it is `1/(2πζ)` and depends on ζ alone. That is awkward, because ζ
   is simultaneously the amplitude-normalisation knob — one parameter with two jobs that pull
   in different directions.

4. **The surprise: a slow filter is a narrow one.** A sweeping input dwells in a unit's
   resonance peak for a time ∝ ζω while the unit needs `1/(ζω)` to fill, so the fraction of
   steady state reached goes as `(ζω)²`. Slow, lightly damped units never catch up. This is the
   time-frequency uncertainty principle appearing inside a network layer, and it constrains what
   any fixed oscillator bank can do on short sequences.

## Honest limitations

- **Sequential MNIST has not been run.** The code paths exist in notebook 02 §5–7 and are
  untested against real data. Nothing here is comparable to published HORN numbers yet.
- **The one task trained to ceiling is phase-free.** Frequency discrimination reaches 1.00 with
  `pool="rms"`, which discards phase entirely. So the repo's headline mechanism — that
  oscillators carry information in phase as well as amplitude — is *not currently exercised by
  any task in it*. The `rms`-versus-`mean` ablation is the right instrument; it needs a task
  where `rms` loses.
- **Single layer only.** No claim here bears on depth.
- **No comparison against `brainmass`.** Independent implementation validated against physics,
  not against the reference.

## Reproducing

```bash
git clone <this repo> && cd horn-jax
uv venv --python 3.12 && source .venv/bin/activate

uv pip install -e ".[dev,notebooks]"        # CPU
# uv pip install -e ".[cuda,dev,notebooks]" # NVIDIA GPU on WSL2, see SETUP.md

<<<<<<< HEAD
pytest                                      # 21 passed — see TESTING.md
=======
pytest                                      # 23 passed — see TESTING.md
>>>>>>> e3b3bd2 (Fixed staled data error and documented)
python demo.py                              # writes results/demo.png
jupyter lab notebooks/                      # 01 then 02
```

The `-e` is what makes `import horn` work from anywhere — notebooks, scripts, tests, any
working directory — with no `sys.path` manipulation. MNIST downloads on first use and caches to
`data/mnist.npz`. See `SETUP.md` for WSL2 + CUDA.

## Open

- [x] Core dynamics, validated against analytic solutions
- [x] Sequence layer, linear readout, training loop
- [x] Frequency discrimination trained to ceiling
- [ ] Sequential MNIST, row-wise and pixel-wise, against published numbers
- [ ] A task where phase carries the signal, so `rms` pooling *loses*
- [ ] Frozen vs learned (ω, ζ) — wired up, needs real data
- [ ] Stacked layers
- [ ] Nested banded ω with cross-frequency modulation vs flat heterogeneity, matched parameters
- [ ] Spiking readout: does phase coding degrade more gracefully than a continuous readout
      under quantisation of the hidden state?

## References

- Effenberger et al., *An analog-electronic implementation of a harmonic oscillator recurrent
  network*, [arXiv:2509.04064](https://arxiv.org/abs/2509.04064)
- Effenberger, Carvalho, Dubinin & Singer, *The functional role of oscillatory dynamics in
  neocortical circuits: A computational perspective*, PNAS 2025
  ([link](https://www.pnas.org/doi/10.1073/pnas.2412830122))
- Reference implementation: [`brainmass`](https://brainmass.readthedocs.io/reference/horn.html)
