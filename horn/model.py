"""Sequence model built on the HORN core: oscillator layer + linear readout.

core.py gives the dynamics. To *learn*, the model needs
  (a) pooling: a way to turn a whole trajectory into a fixed-size vector
  (b) linear readout: a map from that vector to class scores
  (c) a loss, and gradients that survive the recurrence

UNITS
Frequencies are specified in Hz and converted to rad/s behind.
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
    total = n_steps * dt                             # sequence duration in seconds
    f_min = min_cycles / total                       # slowest unit that completes a cycle
    f_max = 1.0 / (min_steps_per_period * dt)        # fastest unit the timestep resolves
    return float(f_min), float(f_max)                # plain floats: these are config, not traced


def log_spaced_bands(n_osc, f_lo, f_hi):
    """Log-spaced natural frequencies across a band. Log, not linear, because what
    matters perceptually and dynamically is the RATIO between frequencies: 2-4 Hz is
    the same interval as 40-80 Hz, and linear spacing would crowd the fast end."""
    return jnp.logspace(jnp.log10(f_lo), jnp.log10(f_hi), n_osc)


# Construction

def init_net(key, in_size, n_osc, n_out, f_hz, zeta, w_scale=0.1, readout_scale=1.0,
             input_gain="normalised", rec_gain="flat", pool="rms"):
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

    rec_gain: THE SAME QUESTION, ASKED OF THE RECURRENT PATH.
      "flat"       - plain W_rec, scaled only by w_scale/sqrt(n_osc). This is the
                      historical default and it has a measured consequence: with
                      input_gain="normalised" the external drive outweighs the
                      recurrent drive by ~5000:1 at 40 Hz, so the network is
                      effectively FEEDFORWARD. Zeroing W_rec then changes the
                      logits by 4e-4 relative, and any experiment whose
                      independent variable is "recurrence on/off" measures
                      nothing. Kept as the default so earlier results reproduce.
      "normalised" - apply the same 2*zeta*omega^2 factor to W_rec. The
                      normalisation exists so that a DRIVE produces an O(1)
                      response; recurrent input is a drive, so it belongs here
                      too. This is what makes recurrence actually matter.

    `tests/test_model.py::test_drive_balance_and_recurrence_leverage` pins both.
    """
    k_in, k_rec, k_out = jax.random.split(key, 3)  # one key per weight matrix, so no
                                                   # two of them share random draws

    # Accept a scalar or a per-unit array for both constants; broadcasting here means
    # no downstream code has to branch on which was given.
    f_hz = jnp.broadcast_to(jnp.asarray(f_hz, jnp.float32), (n_osc,))
    zeta = jnp.broadcast_to(jnp.asarray(zeta, jnp.float32), (n_osc,))

    omega = TWOPI * f_hz                # Hz at the interface, rad/s inside the ODE
    # 1/(2*zeta*omega^2) is the on-resonance steady-state gain of the oscillator,
    # so multiplying a drive by its inverse cancels the collapse exactly.
    # [:, None] makes it a column so it multiplies each ROW of a weight matrix, i.e.
    # each oscillator gets its own gain, applied across all of its inputs.
    resonance_gain = (2.0 * zeta * omega ** 2)[:, None]

    if input_gain == "normalised":
        gain_in = resonance_gain
    elif input_gain == "flat":
        gain_in = 1.0                    # scalar: broadcasts to a no-op
    else:
        raise ValueError(f"unknown input_gain: {input_gain}")

    if rec_gain == "normalised":
        gain_rec = resonance_gain
    elif rec_gain == "flat":
        gain_rec = 1.0
    else:
        raise ValueError(f"unknown rec_gain: {rec_gain}")

    horn = HORNParams(
        W_in=jax.random.normal(k_in, (n_osc, in_size)) * w_scale * gain_in,
        # 1/sqrt(fan-in) keeps total recurrent drive stable as the layer grows
        W_rec=(jax.random.normal(k_rec, (n_osc, n_osc))
               * (w_scale / jnp.sqrt(n_osc)) * gain_rec),
        log_omega=jnp.log(omega),          # rad/s, log-stored for positivity
        log_zeta=jnp.log(zeta),
    )

    # Feature width must match what _pool will emit, or the readout matmul fails at
    # runtime: every mode stacks x and v/omega (2*n_osc), and "meanrms" stacks both
    # statistics of both (4*n_osc). Keep this in step with _pool.
    n_feat = 4 * n_osc if pool == "meanrms" else 2 * n_osc
    readout = ReadoutParams(
        # 1/sqrt(n_feat) so initial logits are O(1) regardless of width
        W=jax.random.normal(k_out, (n_out, n_feat)) * (readout_scale / jnp.sqrt(n_feat)),
        b=jnp.zeros((n_out,)),
    )
    return NetParams(horn=horn, readout=readout)


# Forward

def _pool(xs, vs, mode):
    """Collapse a (T, B, n_osc) trajectory into per-trial features (B, n_feat).

    This is the summary statistic the decoder actually reads: it never sees the
    time series, only one number per unit and channel. Which statistic is chosen
    decides which tasks are learnable at all (see docs/E02).

    `vs` is expected to have ALREADY been divided by omega by the caller. For a
    harmonic oscillator v ~ omega*x, so at 100 Hz the raw velocity is ~600x the
    position and would dominate the readout purely through units. Using
    (x, v/omega) puts both on the same scale - they are the coordinates in which
    x^2 + (v/omega)^2 is proportional to energy.

    Widths: 'last', 'mean', 'rms' give 2*n_osc; 'meanrms' gives 4*n_osc.

    'last'    : final (x, v). Phase-sensitive, but the answer depends on where in
                its cycle each oscillator happened to stop, which for a free-running
                oscillator is close to arbitrary.
    'mean'    : time-average of (x, v). Phase-cancelling for a pure oscillation,
                but for a DRIVEN system it is a linear filter output, and stable
                to train.
    'rms'     : sqrt(mean(x^2)) per unit, pure response power, discarding sign and
                phase. Useful as an ablation: if 'rms' matches 'mean', the task is
                not using phase information.
    'meanrms' : both, for input with a DC component (e.g. pixel intensities).
    """
    if mode == "last":
        # index -1 along time; keeps the instantaneous phase of every unit
        return jnp.concatenate([xs[-1], vs[-1]], axis=-1)
    if mode == "mean":
        # .mean(0) averages over time (axis 0), leaving (B, n_osc) per channel
        return jnp.concatenate([xs.mean(0), vs.mean(0)], axis=-1)
    if mode == "rms":
        # +1e-12 inside the sqrt: at exactly zero the derivative of sqrt is
        # infinite, so a silent unit would emit NaN gradients without it
        return jnp.concatenate([jnp.sqrt((xs ** 2).mean(0) + 1e-12),
                                jnp.sqrt((vs ** 2).mean(0) + 1e-12)], axis=-1)
    if mode == "meanrms":
        # Safe default when it is unclear whether the class signal lives in the
        # DC component or in the response power. Costs 2x readout width.
        return jnp.concatenate([xs.mean(0), vs.mean(0),
                                jnp.sqrt((xs ** 2).mean(0) + 1e-12),
                                jnp.sqrt((vs ** 2).mean(0) + 1e-12)], axis=-1)
    raise ValueError(f"unknown pool mode: {mode}")


def forward(params, inputs, dt, pool="mean"):
    """Run a batch of sequences and decode. inputs (T, B, in_size) -> scores (B, n_out).

    Time first, then batch: that is the axis order lax.scan iterates over.
    """
    T, B, _ = inputs.shape
    n_osc = params.horn.log_omega.shape[0]      # infer width from the parameters

    state = init_state(n_osc, batch=B)          # every sequence starts at rest, x = v = 0

    # core.run_sequence emits only x per step; the readout also needs v, so the scan
    # is repeated here with a body that returns both halves of the state.
    from horn.core import step as _step

    def body(carry, u):
        new, _x = _step(params.horn, carry, u, dt)
        return new, (new.x, new.v)              # carry the state, record (x, v)

    _, (xs, vs) = jax.lax.scan(body, state, inputs)   # xs, vs: (T, B, n_osc)
    omega = jnp.exp(params.horn.log_omega)
    feats = _pool(xs, vs / omega, pool)         # v/omega, see _pool docstring
    # Affine decoder. W is (n_out, n_feat), so W.T contracts the feature axis.
    return feats @ params.readout.W.T + params.readout.b


def loss_and_acc(params, inputs, labels, dt, pool="mean"):
    """Cross-entropy loss and accuracy. `labels` are integer class indices."""
    logits = forward(params, inputs, dt, pool)
    # log-softmax via logsumexp rather than log(softmax(...)): mathematically the
    # same, numerically safe, since exponentiating a large score would overflow.
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    # Pick out the log-probability assigned to each trial's true class and average.
    # At chance this equals ln(n_classes), which is the diagnostic used in E02.
    loss = -jnp.mean(jnp.take_along_axis(logp, labels[:, None], axis=-1))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)   # fraction correctly classified
    return loss, acc


# Freezing the oscillator constants

def freeze_oscillators(grads):
    """Zero the gradients on log_omega and log_zeta.

    Used for the frozen-vs-learned comparison: the oscillator bank stays exactly
    where it was initialised and only W_in, W_rec and the readout adapt.
    """
    # Operates on the GRADIENT tree, not the parameters: a zero gradient means the
    # optimiser leaves those entries untouched, so the bank stays at initialisation
    # while W_in, W_rec and the readout continue to adapt.
    horn = grads.horn._replace(
        log_omega=jnp.zeros_like(grads.horn.log_omega),
        log_zeta=jnp.zeros_like(grads.horn.log_zeta),
    )
    return grads._replace(horn=horn)          # NamedTuple._replace returns a copy


def band_summary(params):
    """Per-oscillator (f_hz, zeta, tau_s, cycles) for diagnostics and figures."""
    omega = jnp.exp(params.horn.log_omega)
    zeta = jnp.exp(params.horn.log_zeta)
    return {
        "f_hz": omega / TWOPI,            # back to Hz for reporting
        "zeta": zeta,
        "tau_s": 1.0 / (zeta * omega),    # envelope decay time in SECONDS
        "cycles": 1.0 / (TWOPI * zeta),   # the same horizon in CYCLES: omega cancels,
                                          # so this depends on zeta alone (docs/E04)
    }
