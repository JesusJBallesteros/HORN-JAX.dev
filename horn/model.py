"""Sequence model built on the HORN core: oscillator layer + linear readout.

core.py gives you dynamics. To *learn* it needs
  (a) pooling: a way to turn a whole trajectory into a fixed-size vector
  (b) linear readout: a map from that vector to class scores
  (c) a loss, and gradients that survive the recurrence

UNITS
-----
Frequencies are specified in Hz, as in notebook 01, and converted to rad/s behind.
A sequence of L steps is treated as occupying a real duration
T_seq seconds, so dt = T_seq / L. That keeps omega interpretable in Hz and makes
band-structured initialisation (theta / gamma / ...) mean something.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from horn.core import HORNParams, HORNState, init_state, run_sequence

TWOPI = 2 * jnp.pi


class ReadoutParams(NamedTuple):
    """Linear map from pooled oscillator state to class logits."""
    W: jnp.ndarray   # (n_out, n_feat)  n_feat = 2*n_osc when reading x and v
    b: jnp.ndarray   # (n_out,)


class NetParams(NamedTuple):
    """Everything the optimiser updates."""
    horn: HORNParams
    readout: ReadoutParams


def usable_band(n_steps, dt, min_cycles=1.0, min_steps_per_period=10):
    """The frequency range a sequence of this length and resolution can represent.

    To consider:

      A unit must complete at least `min_cycles` within the sequence,
                 otherwise it never oscillates and is just a leaky integrator.
                 -> f_min = min_cycles / (n_steps * dt)

      Integration needs `min_steps_per_period` samples per period to
                 stay faithful. -> f_max = 1 / (min_steps_per_period * dt)

    The ratio f_max/f_min = n_steps / (min_cycles * min_steps_per_period) is
    therefore set by SEQUENCE LENGTH alone. A 1:6 nested band
    structure needs at least ~60 steps to be representable at all.
    """
    total = n_steps * dt
    f_min = min_cycles / total
    f_max = 1.0 / (min_steps_per_period * dt)
    return float(f_min), float(f_max)


def log_spaced_bands(n_osc, f_lo, f_hi):
    """Log-spaced natural frequencies across a band. Log, not linear, because
    what matters perceptually and dynamically is the ratio between frequencies."""
    return jnp.logspace(jnp.log10(f_lo), jnp.log10(f_hi), n_osc)


# Construction

def init_net(key, in_size, n_osc, n_out, f_hz, zeta, w_scale=0.1, readout_scale=1.0,
             input_gain="normalised", pool="rms"):
    """Build a one-layer HORN plus linear readout.

    f_hz : scalar or (n_osc,) array, in HERTZ.
    zeta : scalar or (n_osc,) array, damping ratio.

    input_gain:
      "normalised" - scale each row of W_in by 2*zeta*omega^2 so that every
                      oscillator's on-resonance response is O(1) regardless of
                      its frequency. REQUIRED in practice: the steady-state gain
                      is 1/(2*zeta*omega^2), so across a 10-100 Hz bank the
                      fastest units respond 100x more weakly than the slowest,
                      and with a flat W_in the whole population produces states
                      of order 1e-6. The readout then sees nothing, the softmax
                      is uniform, and no gradient flows.
      "flat"       - plain W_in. Just so the failure can be reproduced.
    """
    k_in, k_rec, k_out = jax.random.split(key, 3)

    f_hz = jnp.broadcast_to(jnp.asarray(f_hz, jnp.float32), (n_osc,))
    zeta = jnp.broadcast_to(jnp.asarray(zeta, jnp.float32), (n_osc,))

    omega = TWOPI * f_hz
    if input_gain == "normalised":
        gain = (2.0 * zeta * omega ** 2)[:, None]   # cancels the 1/(2*zeta*w^2) collapse
    elif input_gain == "flat":
        gain = 1.0
    else:
        raise ValueError(f"unknown input_gain: {input_gain}")

    horn = HORNParams(
        W_in=jax.random.normal(k_in, (n_osc, in_size)) * w_scale * gain,
        # 1/sqrt(fan-in) keeps total recurrent drive stable as the layer grows
        W_rec=jax.random.normal(k_rec, (n_osc, n_osc)) * (w_scale / jnp.sqrt(n_osc)),
        log_omega=jnp.log(omega),          # rad/s, log-stored for positivity
        log_zeta=jnp.log(zeta),
    )

    n_feat = 4 * n_osc if pool == "meanrms" else 2 * n_osc   # x and v/omega, doubled if meanrms
    readout = ReadoutParams(
        # 1/sqrt(n_feat) so initial logits are O(1) regardless of width
        W=jax.random.normal(k_out, (n_out, n_feat)) * (readout_scale / jnp.sqrt(n_feat)),
        b=jnp.zeros((n_out,)),
    )
    return NetParams(horn=horn, readout=readout)


# Forward

def _pool(xs, vs, mode):
    """Collapse a (T, B, n_osc) trajectory into (B, 2*n_osc) features.

    `vs` is expected to have ALREADY been divided by omega by the caller. For a
    harmonic oscillator v ~ omega*x, so at 100 Hz the raw velocity is ~600x the
    position and would dominate the readout purely through units. Using
    (x, v/omega) puts both on the same scale - they are the coordinates in which
    x^2 + (v/omega)^2 is proportional to energy.

    'last' : final (x, v). Phase-sensitive - the answer depends on where in its
             cycle each oscillator happened to stop, which for a free-running
             oscillator is close to arbitrary.
    'mean' : time-average of (x, v). Phase-cancelling for a pure oscillation,
             but for a DRIVEN system it is just a linear filter output, and it
             is far more stable to train. Default.
    'rms'  : sqrt(mean(x^2)) per unit - pure power, discards sign and phase
             entirely. Useful as an ablation: if 'rms' matches 'mean', the task
             is not using phase information.
    """
    if mode == "last":
        return jnp.concatenate([xs[-1], vs[-1]], axis=-1)
    if mode == "mean":
        return jnp.concatenate([xs.mean(0), vs.mean(0)], axis=-1)
    if mode == "rms":
        return jnp.concatenate([jnp.sqrt((xs ** 2).mean(0) + 1e-12),
                                jnp.sqrt((vs ** 2).mean(0) + 1e-12)], axis=-1)
    if mode == "meanrms":
        # Both. Safe default when you do not know whether the class signal lives
        # in the DC component or in the response power. Costs 2x readout width.
        return jnp.concatenate([xs.mean(0), vs.mean(0),
                                jnp.sqrt((xs ** 2).mean(0) + 1e-12),
                                jnp.sqrt((vs ** 2).mean(0) + 1e-12)], axis=-1)
    raise ValueError(f"unknown pool mode: {mode}")


def forward(params, inputs, dt, pool="mean"):
    """inputs: (T, B, in_size) -> logits: (B, n_out)."""
    T, B, _ = inputs.shape
    n_osc = params.horn.log_omega.shape[0]

    state = init_state(n_osc, batch=B)          # every sequence starts at rest

    # run_sequence returns only x per step; we need v too for the readout, so
    # re-run the scan here with a body that emits both.
    from horn.core import step as _step

    def body(carry, u):
        new, _x = _step(params.horn, carry, u, dt)
        return new, (new.x, new.v)

    _, (xs, vs) = jax.lax.scan(body, state, inputs)
    omega = jnp.exp(params.horn.log_omega)
    feats = _pool(xs, vs / omega, pool)      # v/omega, see _pool docstring
    return feats @ params.readout.W.T + params.readout.b


def loss_and_acc(params, inputs, labels, dt, pool="mean"):
    """Softmax cross-entropy plus accuracy. labels are integer class indices."""
    logits = forward(params, inputs, dt, pool)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss = -jnp.mean(jnp.take_along_axis(logp, labels[:, None], axis=-1))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, acc


# Freezing the oscillator constants

def freeze_oscillators(grads):
    """Zero the gradients on log_omega and log_zeta.

    Used for the frozen-vs-learned comparison: the oscillator bank stays exactly
    where it was initialised and only W_in, W_rec and the readout adapt.
    """
    horn = grads.horn._replace(
        log_omega=jnp.zeros_like(grads.horn.log_omega),
        log_zeta=jnp.zeros_like(grads.horn.log_zeta),
    )
    return grads._replace(horn=horn)


def band_summary(params):
    """Current (f_hz, zeta, tau_s, cycles_rung) per oscillator, for diagnostics."""
    omega = jnp.exp(params.horn.log_omega)
    zeta = jnp.exp(params.horn.log_zeta)
    return {
        "f_hz": omega / TWOPI,
        "zeta": zeta,
        "tau_s": 1.0 / (zeta * omega),
        "cycles": 1.0 / (TWOPI * zeta),   # depends on zeta alone
    }
