"""Tests for the sequence model: shapes, pooling, gradients, freezing.
These are structural, core.py already guarantees the right dynamics.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from horn.model import (
    init_net, forward, loss_and_acc, freeze_oscillators,
    usable_band, log_spaced_bands, band_summary, TWOPI,
)

DT = 1e-3


def _net(key=0, in_size=4, n_osc=8, n_out=3):
    return init_net(jax.random.PRNGKey(key), in_size, n_osc, n_out,
                    f_hz=jnp.linspace(4.0, 20.0, n_osc), zeta=0.15)


def test_forward_shapes():
    """(T, B, in) must map to (B, n_out) for every pooling mode."""
    p = _net()
    x = jax.random.normal(jax.random.PRNGKey(1), (30, 5, 4))
    for mode in ["last", "mean", "rms"]:
        out = forward(p, x, DT, pool=mode)
        assert out.shape == (5, 3), f"{mode} gave {out.shape}"
        assert jnp.all(jnp.isfinite(out))


def test_readout_uses_velocity():
    """The readout reads (x, v). Zeroing the velocity half must change the output.

    Guards against an indexing slip where the second half of the feature vector
    is silently the same as the first, which would look fine but throw away
    exactly the state variable a normal RNN does not have.
    """
    p = _net()
    x = jax.random.normal(jax.random.PRNGKey(2), (30, 5, 4))
    base = forward(p, x, DT, pool="mean")
    n_osc = p.horn.log_omega.shape[0]
    W2 = p.readout.W.at[:, n_osc:].set(0.0)          # blank the velocity columns
    p2 = p._replace(readout=p.readout._replace(W=W2))
    assert not jnp.allclose(base, forward(p2, x, DT, pool="mean"))


def test_batch_is_independent():
    """Sequences in a batch must not leak into each other."""
    p = _net()
    a = jax.random.normal(jax.random.PRNGKey(3), (25, 1, 4))
    b = jax.random.normal(jax.random.PRNGKey(4), (25, 1, 4))
    together = forward(p, jnp.concatenate([a, b], axis=1), DT)
    alone_a = forward(p, a, DT)
    assert jnp.allclose(together[0], alone_a[0], atol=1e-5)


def test_gradients_finite_and_nonzero():
    """Gradients must reach every parameter group through the recurrence."""
    p = _net()
    x = jax.random.normal(jax.random.PRNGKey(5), (40, 6, 4))
    y = jnp.array([0, 1, 2, 0, 1, 2])
    g = jax.grad(lambda q: loss_and_acc(q, x, y, DT)[0])(p)
    for leaf in jax.tree.leaves(g):
        assert jnp.all(jnp.isfinite(leaf))
    for name, leaf in [("W_in", g.horn.W_in), ("W_rec", g.horn.W_rec),
                       ("log_omega", g.horn.log_omega), ("log_zeta", g.horn.log_zeta),
                       ("readout.W", g.readout.W)]:
        assert float(jnp.linalg.norm(leaf)) > 0, f"{name} received zero gradient"


def test_freeze_zeroes_only_oscillator_constants():
    p = _net()
    x = jax.random.normal(jax.random.PRNGKey(6), (40, 6, 4))
    y = jnp.array([0, 1, 2, 0, 1, 2])
    g = freeze_oscillators(jax.grad(lambda q: loss_and_acc(q, x, y, DT)[0])(p))
    assert jnp.all(g.horn.log_omega == 0)
    assert jnp.all(g.horn.log_zeta == 0)
    assert float(jnp.linalg.norm(g.horn.W_in)) > 0      # everything else still trains
    assert float(jnp.linalg.norm(g.readout.W)) > 0


def test_usable_band_ratio_depends_only_on_length():
    """f_max/f_min is set by sequence length, not by dt. This is the constraint
    that decides whether a nested band structure is representable at all."""
    for n, dt in [(784, 1/784), (784, 1e-3), (28, 1e-2)]:
        lo, hi = usable_band(n, dt, min_cycles=1.0, min_steps_per_period=10)
        assert np.isclose(hi / lo, n / 10.0, rtol=1e-6), (n, dt, hi/lo)
    lo28, hi28 = usable_band(28, 1e-2)
    assert hi28 / lo28 < 6, "28 steps cannot represent a 1:6 nesting"


def test_band_summary_roundtrip():
    """Frequencies must come back out in Hz exactly as they went in."""
    f = jnp.array([4.0, 8.0, 16.0, 32.0])
    p = init_net(jax.random.PRNGKey(7), 3, 4, 2, f_hz=f, zeta=0.1)
    s = band_summary(p)
    assert jnp.allclose(s["f_hz"], f, rtol=1e-5)
    # cycles rung = 1/(2*pi*zeta) and must be identical across units at fixed zeta
    assert jnp.allclose(s["cycles"], s["cycles"][0])


def test_log_spaced_bands_endpoints():
    b = log_spaced_bands(5, 2.0, 32.0)
    assert jnp.allclose(b[0], 2.0) and jnp.allclose(b[-1], 32.0)
    ratios = b[1:] / b[:-1]
    assert jnp.allclose(ratios, ratios[0])      # constant ratio = log spacing


def test_drive_balance_and_recurrence_leverage():
    """The bug this exists to prevent: an inert independent variable.

    `input_gain="normalised"` multiplies W_in by 2*zeta*omega^2. Nothing
    multiplied W_rec, so at 40 Hz the external drive outweighed the recurrent
    drive by ~5000:1 and the network was effectively feedforward. An experiment
    whose variable was "recurrence on/off" then compared a network against
    itself: zeroing W_rec moved the logits by 4e-4 relative, and the sweep
    returned bit-identical accuracies for conditions that were supposed to
    differ.

    Nothing here asserts that "flat" is wrong - it is kept as the default so
    earlier results reproduce. What is asserted is that the two settings are
    genuinely different, and that "normalised" gives recurrence enough leverage
    to be worth sweeping over.
    """
    import numpy as np
    from horn.model import forward, log_spaced_bands, usable_band

    L, dt, n_osc = 200, 2.5e-3, 32
    f_lo, f_hi = usable_band(L, dt)
    inputs = jax.random.normal(jax.random.PRNGKey(3), (L, 8, 1))

    def leverage(rec_gain):
        p = init_net(jax.random.PRNGKey(0), 1, n_osc, 3,
                     f_hz=log_spaced_bands(n_osc, f_lo, f_hi), zeta=0.05,
                     pool="meanrms", w_scale=3.0, rec_gain=rec_gain)
        zeroed = p._replace(horn=p.horn._replace(
            W_rec=jnp.zeros_like(p.horn.W_rec)))
        a = forward(p, inputs, dt, "meanrms")
        b = forward(zeroed, inputs, dt, "meanrms")
        rel = float(jnp.abs(a - b).max() / jnp.abs(a).max())
        ratio = float(jnp.abs(p.horn.W_in).mean() / jnp.abs(p.horn.W_rec).mean())
        return rel, ratio

    flat_rel, flat_ratio = leverage("flat")
    norm_rel, norm_ratio = leverage("normalised")

    # The historical default really is inert - this is the measurement, recorded
    # so that a future change which silently "fixes" it is visible as a failure.
    assert flat_rel < 1e-2, f"flat leverage {flat_rel:.2e} unexpectedly large"

    # The fix must give recurrence real influence.
    assert norm_rel > 0.05, (
        f"rec_gain='normalised' leverage is only {norm_rel:.2e}; recurrence is "
        "still inert and a sweep over it would measure nothing")
    assert norm_rel > 10 * flat_rel, "normalised is not meaningfully different from flat"
    assert norm_ratio < flat_ratio / 10, "W_rec was not rescaled"


def test_rec_gain_rejects_unknown_values():
    from horn.model import log_spaced_bands
    with pytest.raises(ValueError, match="rec_gain"):
        init_net(jax.random.PRNGKey(0), 1, 8, 3,
                 f_hz=log_spaced_bands(8, 1.0, 10.0), zeta=0.1, rec_gain="nope")
