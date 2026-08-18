"""Synthetic tasks.

Two tasks:
`freq_batch`    classify which of several frequencies a noisy sinusoid carries.
                Solved perfectly by power pooling. It is a plumbing check, a 
                magnitude FFT plus logistic regression does the same.

`biphase_batch` the actual question.

THE BIPHASE TASK
Every stimulus is the same two tones, at f and 2f, at the same amplitudes:

    s(t) = sin(2*pi*f*t + p) + sin(2*pi*2f*t + 2p + psi)

`p` is a global phase drawn uniformly per example. The class is `psi`, the
BIPHASE: the phase of the second harmonic relative to twice the phase of the
first. This is the quantity a bispectrum measures, the cross-frequency-coupling 
or quadratic phase coupling.

Three properties:
1. **The power spectrum is identical across classes.** Both tones have fixed
   amplitude; only their relative phase changes. `pool="rms"` is a test.

2. **`psi` survives a global time shift.** Shifting t -> t + tau sends
   p -> p + 2*pi*f*tau and the second tone's phase to 2*(p + 2*pi*f*tau) + psi.
   The `2p` structure is what makes this work: psi is invariant, so the label
   cannot be read off any absolute time reference. A readout of the final state
   alone is therefore also at chance, since the global phase is random.

3. **Extracting `psi` requires specific nonlinearity.** psi appears in the signal
   only in cross-terms between the two tones. Expanding a cubic nonlinearity in 
   a sum of tones at f and 2f, the term in 
   cos^2(2*pi*f*t + p) * cos(2*pi*2f*t + 2p + psi) 
   contains a component at zero frequency proportional to cos(psi). 
   A DC offset that depends on the biphase.

 Prediction: a bank of INDEPENDENT LINEAR resonators feeding a linear readout
 cannot do this at all, whatever the pooling, because no linear function of
 per-unit features contains a product of two units.

 The word LINEAR is load-bearing and was missing from the first version of this
 note. It holds under core.step(drive="output"), where the external path carries
 no nonlinearity and W_rec = 0 leaves a genuine filter bank. It does NOT hold
 under drive="input", the reference model's placement, where tanh(W_in u) squashes
 the stimulus before it reaches any oscillator and supplies the cubic cross-term
 on its own: the same uncoupled bank then scores 0.76-0.80 rather than chance.
 See docs/E05, "Which model the control belongs to".

MEASURED AT INITIALISATION, BEFORE TRAINING
Held-out accuracy of a ridge readout on the pooled features of an UNTRAINED
network, 3 classes, chance 0.333 (see `experiments/probe_mechanism.py`):

    rms|x|    W_rec        mean     rms     last
    0.028     free/off     0.323   0.365   0.320     linear regime: nothing
    0.279     free         1.000   0.372   0.687
    0.279     zero         0.323   0.365   0.320     filter bank: nothing
    2.789     free         1.000   0.652   1.000
    2.789     zero         0.323   0.365   0.320     filter bank: still nothing

Two conditions are required, and neither alone suffices:

  * **recurrence.** With W_rec = 0 the network is at chance at EVERY amplitude,
    which is the prediction above confirmed;
  * **amplitude.** With states of order 0.01 the tanh is linear to a part in
    400, and a linear recurrent network is still just a filter bank.

The control that falsifies is therefore the ARCHITECTURE (W_rec = 0), not the
pooling - in this model. Under the reference placement there is no such control,
because an uncoupled unit is not linear to begin with. `rms` is only chance-level while the system is linear; once the
nonlinearity is engaged it converts biphase into power too, which is why rms
climbs to 0.652 at large amplitude. That is a real effect.

The stimulus power spectrum is still matched, but the network's internal spectrum 
is not.

Forming a product needs units at BOTH f and 2f, so heterogeneity is
a precondition rather than an advantage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

TWOPI = 2 * jnp.pi

# THREE classes, evenly spaced. The count is derived, not chosen for convenience.
#
# The only DC (time-averaged) terms a nonlinearity can produce from tones at f
# and 2f are those where the frequencies cancel: a copies of the fundamental and
# b of the harmonic with a + 2b = 0. For an ODD nonlinearity such as tanh the
# total order |a| + |b| = 3|b| must be odd, so b = +-1 (order 3) or b = +-3
# (order 9), giving DC terms proportional to sin(psi) and sin(3*psi).
#
# With four evenly spaced biphases (0, pi/2, pi, 3pi/2) both of those are
# ZERO for psi = 0 and psi = pi, so those two classes are indistinguishable to
# anything reading a time average, and the ceiling is 75% rather than 100%. It
# would look like a partial failure of the model when it is a property of the
# stimulus set.
#
# Three evenly spaced values give sin(psi) = 0, +0.866, -0.866: all distinct.
DEFAULT_BIPHASES = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)


def homogeneous_bands(n_osc: int, f_lo: float, f_hi: float) -> jnp.ndarray:
    """Every unit at the geometric mean of the band.

    The matched control for `log_spaced_bands`: same unit count, therefore the
    same parameter count exactly, and the same mean frequency in log space. The
    only thing that differs is the spread, which is the variable under test.

    Geometric rather than arithmetic mean because frequency is perceived and
    behaves multiplicatively: the midpoint of 1-100 Hz is 10, not 50.
    """
    return jnp.full((n_osc,), float(np.sqrt(f_lo * f_hi)), dtype=jnp.float32)


def freq_batch(key, n, classes=(8.0, 32.0, 150.0), n_steps=20000, dt=5e-5,
               noise=0.3):
    """Noisy sinusoid at one of `classes` Hz, random phase. -> (L, n, 1), (n,)."""
    k_y, k_p, k_n = jax.random.split(key, 3)     # separate keys: label, phase, noise
    classes = jnp.asarray(classes, jnp.float32)

    y = jax.random.randint(k_y, (n,), 0, len(classes))   # class index per trial
    phase = jax.random.uniform(k_p, (n,)) * TWOPI        # random start phase, so absolute
                                                         # phase carries no label information
    t = jnp.arange(1, n_steps + 1) * dt                  # time axis in seconds

    # Outer-product broadcasting: t[:, None] is (L, 1) and the per-trial terms are
    # (1, n), so the result is (L, n) without any explicit loop over trials.
    x = jnp.sin(TWOPI * classes[y][None, :] * t[:, None] + phase[None, :])
    x = x + noise * jax.random.normal(k_n, (n_steps, n))
    return x[..., None], y                        # trailing axis = input channel of width 1


def biphase_batch(key, n, f_hz=10.0, n_steps=400, dt=2.5e-3,
                  biphases=DEFAULT_BIPHASES, noise=0.1, amps=(1.0, 1.0)):
    """Two tones at f and 2f; the class is their biphase. -> (L, n, 1), (n,).

    Amplitudes are FIXED, not drawn, keeping the power spectrum identical across 
    classes and making the rms result a proof. Do not randomise them without
    re-running `tests/test_tasks.py::test_power_spectrum_is_matched_across_classes`.

    Noise is additive white, so it raises the floor equally for every class and
    cannot leak label information.
    """
    k_y, k_p, k_n = jax.random.split(key, 3)     # separate keys: label, phase, noise
    psi_values = jnp.asarray(biphases, jnp.float32)
    a1, a2 = amps                                # fixed amplitudes, never drawn

    y = jax.random.randint(k_y, (n,), 0, len(psi_values))
    p = jax.random.uniform(k_p, (n,)) * TWOPI          # global phase, per example
    psi = psi_values[y]                                # the label, as a phase
    t = jnp.arange(1, n_steps + 1) * dt

    fundamental = a1 * jnp.sin(TWOPI * f_hz * t[:, None] + p[None, :])
    # The 2*p is essential: it is what makes psi invariant to a time shift. Shifting
    # t by d adds 2*pi*f*d to the fundamental's phase and twice that to the harmonic's,
    # so the combination phi2 - 2*phi1 = psi is left untouched. Without the factor of
    # two, psi would be recoverable from absolute timing and the task would be trivial.
    harmonic = a2 * jnp.sin(TWOPI * 2 * f_hz * t[:, None] + 2 * p[None, :]
                            + psi[None, :])

    x = fundamental + harmonic + noise * jax.random.normal(k_n, (n_steps, n))
    return x[..., None], y


def power_spectrum_by_class(key, n_per_class=256, **kw):
    """Mean power spectrum of each biphase class, for validity checking.

    Returns (freqs_hz, power) with power of shape (n_classes, n_freqs). If these
    rows are not equal to within sampling error, the task leaks the label into
    power and the rms condition stops being a control.
    """
    biphases = kw.get("biphases", DEFAULT_BIPHASES)
    dt = kw.get("dt", 2.5e-3)

    rows = []
    for i in range(len(biphases)):
        # One class at a time, by passing a single-element biphase tuple. The same
        # `key` is reused deliberately, so the trials differ ONLY in psi and the
        # comparison is not confounded by different noise draws.
        single = dict(kw, biphases=(biphases[i],))
        x, _ = biphase_batch(key, n_per_class, **single)
        # rfft over time (axis 0) of the single input channel; squared magnitude = power
        spec = np.abs(np.fft.rfft(np.asarray(x)[:, :, 0], axis=0)) ** 2
        rows.append(spec.mean(axis=1))           # average over trials within the class

    freqs = np.fft.rfftfreq(np.asarray(x).shape[0], dt)   # bin centres in Hz
    return freqs, np.stack(rows)                          # (n_classes, n_freqs)


# np.angle measures phase against a COSINE; biphase_batch builds the stimulus
# from SINES. Since sin(a) = cos(a - pi/2), each measured phase is short by pi/2,
# and the biphase combination phi2 - 2*phi1 therefore carries a constant offset:
#
#   phi1 = p - pi/2                      phi2 = 2p + psi - pi/2
#   phi2 - 2*phi1 = psi + pi/2
#
# A constant phase offset from a convention mismatch is one of the ways the 
# cross-frequency-coupling literature produces spurious results, so it is worth 
# being explicit rather than tuning a threshold until a test passes.
_SIN_TO_COS_BIPHASE_OFFSET = np.pi / 2


def biphase_of(signal: np.ndarray, f_hz: float, dt: float) -> float:
    """Recover psi from a signal built by `biphase_batch`, for diagnostics.

    Returns the value in the SINE convention, so that
    `biphase_of(biphase_batch(..., biphases=(psi,))) == psi`.

    Useful as a check that the stimulus carries what it claims to, and as the
    measurement to apply to the network's own units when asking whether
    cross-frequency coupling has emerged internally.
    """
    n = len(signal)
    spec = np.fft.rfft(signal)                   # real FFT: non-negative frequencies only
    freqs = np.fft.rfftfreq(n, dt)
    # Nearest bin to f and 2f. Nearest rather than exact because f need not fall on a
    # bin centre; if it does not, the recovered phase is slightly biased, which is why
    # the tests use frequencies that divide the sequence length evenly.
    i1 = int(np.argmin(np.abs(freqs - f_hz)))
    i2 = int(np.argmin(np.abs(freqs - 2 * f_hz)))
    # The biphase proper: phase of the harmonic minus twice the phase of the fundamental.
    raw = np.angle(spec[i2]) - 2 * np.angle(spec[i1])
    # Remove the sine-versus-cosine convention offset, then wrap into [0, 2pi).
    return float(raw - _SIN_TO_COS_BIPHASE_OFFSET) % (2 * np.pi)
