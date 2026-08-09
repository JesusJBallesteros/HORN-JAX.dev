"""Correctness tests for the HORN core.

These do not check the code itself. They check the physics of the model, which is what matters.
The HORN is a recurrent neural network, but it is also a physical system: a population of coupled
damped harmonic oscillators. The physics is what makes the model useful.
An uncoupled, unforced HORN unit is just a damped harmonic oscillator, and that has a known
closed-form solution, so we can compare against ground truth.

When an oscillator implementation is broken it is nearly always the integrator that is broken.

Run with:  python -m pytest tests/ -q
"""

import jax                # for grad and lax.scan in the differentiability test
import jax.numpy as jnp   # JAX arrays: what the model consumes
import numpy as np        # plain NumPy: fine for computing the EXPECTED answers,
                          # because those never need to be differentiated or GPU-run
from horn.core import (
    HORNParams, HORNState, init_params, init_state, run_sequence, step, energy,
)


def _isolated(n_osc, omega, zeta):
    """Build a HORN with all coupling switched off -> n_osc independent oscillators.

    Zeroing W_in and W_rec removes the drive term entirely, so f(x, u) = 0 and each
    unit reduces to a textbook damped harmonic oscillator. That is what makes an
    analytic comparison possible.
    """
    return HORNParams(
        W_in=jnp.zeros((n_osc, 1)),                    # no external drive
        W_rec=jnp.zeros((n_osc, n_osc)),               # no recurrent coupling
        log_omega=jnp.log(jnp.full((n_osc,), omega)),  # same omega for every unit (log-stored)
        log_zeta=jnp.log(jnp.full((n_osc,), zeta)),    # same zeta for every unit (log-stored)
    )


def test_matches_analytic_underdamped():
    """TEST 1 - the trajectory must match the closed-form solution.

    Release the oscillator from x=1 with zero velocity and no drive, then compare
    against the exact solution of x'' + 2*zeta*omega*x' + omega^2*x = 0.
    """
    omega, zeta, dt, T = 2.0, 0.05, 1e-4, 20000  # small dt: we are testing the model,
                                                 # not tolerating integration error
    p = _isolated(1, omega, zeta)                          # one isolated oscillator
    s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))    # released from rest at x=1
    _, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)     # zero input for T steps

    # --- the exact solution, for comparison ---
    t = np.arange(1, T + 1) * dt              # time of each recorded sample.
                                              # Starts at 1, not 0: run_sequence returns
                                              # the state AFTER each step, so the first
                                              # output corresponds to t=dt.
    wd = omega * np.sqrt(1 - zeta ** 2)       # DAMPED frequency. Damping slows the
                                              # oscillation slightly below omega; using
                                              # omega here instead of wd would cause a
                                              # slow phase drift and fail the test.
    analytic = np.exp(-zeta * omega * t) * (          # exponentially decaying envelope
        np.cos(wd * t)                                # cosine term from x(0)=1
        + (zeta * omega / wd) * np.sin(wd * t)        # sine term enforcing v(0)=0
    )

    err = np.max(np.abs(np.asarray(xs).ravel() - analytic))  # worst-case deviation
    assert err < 1e-2, f"max deviation from analytic solution: {err}"


def test_energy_conserved_when_undamped():
    """TEST 2 - the integrator must not manufacture energy.

    This is the test that distinguishes semi-implicit from explicit Euler.
    With zeta -> 0 and no drive there is nothing to add or remove energy, so
    E = 0.5*(v^2 + omega^2*x^2) must stay flat forever. Explicit Euler fails
    this: it pumps energy in every step and the amplitude grows without bound.
    """
    omega, dt, T = 1.0, 1e-3, 50000
    p = _isolated(1, omega, 1e-12)                        # zeta ~ 0. Not exactly 0 because
                                                          # log(0) = -inf; 1e-12 is
                                                          # numerically indistinguishable.
    s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))
    e0 = energy(p, s0).sum()                              # baseline energy at t=0

    def body(s, _):
        s, _o = step(p, s, jnp.zeros((1,)), dt)  # advance one step
        return s, energy(p, s).sum()             # carry the state, record the energy

    _, es = jax.lax.scan(body, s0, jnp.arange(T))         # energy trace over T steps
    drift = float(jnp.max(jnp.abs(es - e0)) / e0)         # worst RELATIVE deviation
    assert drift < 1e-3, f"energy drift {drift:.2e} too large"


def test_overdamped_decays_without_oscillating():
    """TEST 3 - the damping regimes must behave qualitatively correctly.

    zeta > 1 is overdamped: the system is too sluggish to overshoot, so it must
    crawl back to zero without ever crossing it. If the sign of the damping term
    were wrong, or omega and zeta were transposed, this test fails.
    """
    p = _isolated(1, 1.0, 3.0)                             # zeta = 3 -> firmly overdamped
    s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))
    _, xs = run_sequence(p, s0, jnp.zeros((4000, 1)), 1e-3)
    xs = np.asarray(xs).ravel()                            # to NumPy for easy assertions
    assert np.all(xs > 0), "overdamped oscillator crossed zero"   # never overshoots
    assert xs[-1] < xs[0], "overdamped oscillator did not decay"  # and does decay


def test_heterogeneous_frequencies_are_independent():
    """TEST 4 - each oscillator must keep its own frequency.

    This is the property the whole nested-frequency research idea depends on:
    a population with different omega must behave as a bank of independent
    filters, not smear into one shared frequency. Verified in the frequency
    domain rather than by eye.
    """
    omegas = jnp.array([1.0, 4.0])          # deliberately a 1:4 ratio, as in theta:gamma
    p = HORNParams(
        W_in=jnp.zeros((2, 1)),
        W_rec=jnp.zeros((2, 2)),            # uncoupled, so they CANNOT entrain each other
        log_omega=jnp.log(omegas),          # different frequency per unit
        log_zeta=jnp.log(jnp.full((2,), 1e-12)),  # ~undamped, so the peaks stay sharp
    )
    dt, T = 1e-3, 60000                     # 60 s of signal -> fine frequency resolution
    s0 = HORNState(x=jnp.ones((2,)), v=jnp.zeros((2,)))
    _, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)
    xs = np.asarray(xs)                     # (T, 2)

    freqs = np.fft.rfftfreq(T, dt)          # frequency axis in Hz for a real-valued FFT
    peak = [
        freqs[np.argmax(np.abs(np.fft.rfft(xs[:, i])))]  # Hz of the largest spectral peak
        * 2 * np.pi                                      # convert Hz -> rad/s to match omega
        for i in range(2)
    ]

    # Tolerance = one FFT bin. The FFT can only resolve frequencies to within
    # 2*pi/(T*dt) rad/s, so asserting tighter than that would be testing the
    # resolution of the frequency grid rather than the correctness of the dynamics.
    tol = 2 * np.pi / (T * dt)
    assert abs(peak[0] - 1.0) < tol, (peak, tol)
    assert abs(peak[1] - 4.0) < tol, (peak, tol)


def test_gradients_flow():
    """TEST 5 - the model must be differentiable end to end.

    A model can be numerically perfect and still untrainable if gradients vanish,
    explode, or hit a NaN somewhere in the scan. This checks that reverse-mode
    autodiff makes it through the whole recurrence intact.
    """
    key = jax.random.PRNGKey(0)                                    # fixed seed: reproducible
    p = init_params(key, in_size=3, n_osc=8)                       # a small ordinary network
    inputs = jax.random.normal(jax.random.PRNGKey(1), (25, 3))     # 25 timesteps of input

    def loss(p):
        _, xs = run_sequence(p, init_state(8), inputs)   # forward pass over the sequence
        return jnp.mean(xs ** 2)                         # arbitrary scalar loss - the VALUE
                                                         # is irrelevant, we only need
                                                         # something differentiable

    g = jax.grad(loss)(p)   # gradient w.r.t. EVERY field of HORNParams at once,
                            # returned as a HORNParams with the same structure

    # Walk the gradient pytree and check every array is finite.
    for name, leaf in zip(p._fields, jax.tree.leaves(g)):
        assert jnp.all(jnp.isfinite(leaf)), f"non-finite gradient in {name}"

    # Also confirm the gradient is not identically zero, which would mean the loss
    # is disconnected from the parameters and nothing would ever learn.
    assert float(jnp.linalg.norm(jax.tree.leaves(g)[0])) > 0
