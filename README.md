<<<<<<< HEAD
# JAX-HORN

A JAX implementation of Harmonic Oscillator Recurrent Networks (HORN).
First step is to implement a functional core and test:
different units with varied dumped levels (zeta)
different units with varied frequencies (omega)

GOAL is to get a basic understanding of it all... TOP GOAL is to play with an idea of nested-HORN+SNN, for the fun of it.

## What this is 
(see Refs)
=======
# horn-jax

A from-scratch JAX implementation of Harmonic Oscillator Recurrent Networks (HORN).

> **Note to self — rewrite this README before publishing.** Right now it describes the scaffold. By the time you send the application it should describe *your* work: what you built, what you found, what surprised you, and what the limitations are. The README is the part Felix will actually read.

## What this is
>>>>>>> 7d97265 (HORN core in JAX with tests)

Each unit is a damped, driven harmonic oscillator:

```
x'' + 2ζω x' + ω² x = f(x, u),    f(x, u) = W_rec · tanh(x) + W_in · u
```

<<<<<<< HEAD
State is `(x, v)`. Integration uses semi-implicit (symplectic) Euler: velocity updates first, then position uses the *new* velocity. Plain explicit Euler injects energy into the system and makes the undamped case diverge (for this, the energy-conservation test).

Parameters `ω` and `ζ` are stored as logarithms so they stay strictly positive. Negative damping means exponential growth.
=======
State is `(x, v)`. Integration uses semi-implicit (symplectic) Euler: velocity updates first, then position uses the *new* velocity. Plain explicit Euler injects energy into the system and makes the undamped case diverge — the energy-conservation test catches this.

Parameters `ω` and `ζ` are stored as logs so they stay strictly positive under gradient descent. Negative damping means exponential blow-up, and it is an easy failure to hit otherwise.
>>>>>>> 7d97265 (HORN core in JAX with tests)

## Layout

```
horn/core.py           dynamics: init_params, step, run_sequence, energy
tests/test_dynamics.py correctness tests against physics
demo.py                damping regimes + frequency bank -> demo.png
SETUP.md               WSL2 + CUDA + JAX environment setup
```

<<<<<<< HEAD
## Quick check
=======
## Quick start
>>>>>>> 7d97265 (HORN core in JAX with tests)

```bash
python -m pytest tests/ -q     # 5 passed
python demo.py                 # writes demo.png
```

<<<<<<< HEAD
## Testing

The implementation is checked for:

1. **Analytic solution.** An uncoupled, unforced unit is a damped harmonic oscillator with a known closed form. The trajectory is compared directly against it.
2. **Energy conservation.** At `ζ→0` with no drive, mechanical energy must not drift. This is what distinguishes a symplectic integrator.
3. **Overdamped behaviour.** At `ζ>1` the trajectory must decay to zero.
4. **Frequency independence.** Uncoupled units with different `ω` keep their own period (check FFT)
5. **Differentiability.** Gradients flow through the full scan, stay finite.

If an oscillator implementation is wrong, it is almost always wrong in the integrator and 1 & 2 will fail.

## STEPS

-  X Core dynamics, validated against analytic solution
- [] Sequence layer + linear readout
- [] Sequential MNIST, compared against published HORN numbers
- [] Stacked multi-layer network
- [] Nested frequency structure. Banded `ω` with cross-frequency amplitude modulation, tested against flat heterogeneity at matched parameter count
- [] + SpikeNNs ??

The last items are the playful questions
=======
## The tests, and why these tests

The implementation is checked against physics rather than against itself:

1. **Analytic solution.** An uncoupled, unforced unit is a damped harmonic oscillator with a known closed form. The trajectory is compared directly against it.
2. **Energy conservation.** At `ζ→0` with no drive, mechanical energy must not drift. This is what distinguishes a symplectic integrator from a naive one.
3. **Overdamped behaviour.** At `ζ>1` the trajectory must decay without crossing zero.
4. **Frequency independence.** Uncoupled units with different `ω` each keep their own period, verified by FFT. Tolerance is one FFT bin — asserting tighter would be testing the frequency grid, not the dynamics.
5. **Differentiability.** Gradients flow through the full scan and stay finite.

Tests 1 and 2 are the ones that matter. If an oscillator implementation is wrong, it is almost always wrong in the integrator, and both of these fail loudly when it is.

## Roadmap

- [x] Core dynamics, validated against analytic solution
- [ ] Sequence layer + linear readout
- [ ] Sequential MNIST, compared against published HORN numbers
- [ ] Stacked multi-layer network
- [ ] **Nested frequency structure** — banded `ω` with cross-frequency amplitude modulation, tested against flat heterogeneity at matched parameter count

The last item is the actual research question. See `HORN_repo_plan.md`.
>>>>>>> 7d97265 (HORN core in JAX with tests)

## References

- Effenberger et al., *An analog-electronic implementation of a harmonic oscillator recurrent network*, [arXiv:2509.04064](https://arxiv.org/abs/2509.04064)
- Reference implementation: [`brainmass`](https://brainmass.readthedocs.io/reference/horn.html)
<<<<<<< HEAD
- The functional role of oscillatory dynamics in neocortical circuits: A computational perspective. PNAS. 2025. Felix Effenberger, Pedro Carvalho, Igor Dubinin, and Wolf Singer (https://www.pnas.org/doi/10.1073/pnas.2412830122)
  
=======
>>>>>>> 7d97265 (HORN core in JAX with tests)
