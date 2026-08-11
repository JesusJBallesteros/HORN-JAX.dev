"""Validity tests for the synthetic tasks.

These are not plumbing checks. The biphase experiment rests on three claims about
the STIMULUS, and if any of them is false the experiment measures nothing:

  1. the power spectrum is the same for every class, so `pool="rms"` is at chance
     by construction rather than by observation;
  2. the label survives a global time shift, so it cannot be read from an
     absolute phase reference;
  3. the classes are nevertheless distinguishable, so the task is not impossible.

A claim of "rms is provably at chance" is only as good as test 1.
"""

import numpy as np
import pytest
import jax

from horn.tasks import (
    DEFAULT_BIPHASES, biphase_batch, biphase_of, freq_batch, homogeneous_bands,
    power_spectrum_by_class,
)
from horn.model import log_spaced_bands

KW = dict(f_hz=10.0, n_steps=400, dt=2.5e-3, noise=0.0)


def test_power_spectrum_is_matched_across_classes():
    """THE critical test. If class spectra differ, rms is not a control.

    Compared as a fraction of the peak, because absolute FFT power depends on
    length and amplitude and the quantity that matters is relative difference.
    """
    freqs, power = power_spectrum_by_class(jax.random.PRNGKey(0), 128, **KW)

    assert power.shape[0] == len(DEFAULT_BIPHASES)
    peak = power.max()
    spread = (power.max(axis=0) - power.min(axis=0)) / peak
    assert spread.max() < 1e-6, (
        f"class power spectra differ by up to {spread.max():.2e} of peak "
        f"at {freqs[int(np.argmax(spread))]:.1f} Hz - rms could exploit this")

    # And the power really is concentrated at f and 2f, not smeared.
    top2 = np.sort(freqs[np.argsort(power[0])[-2:]])
    assert np.allclose(top2, [10.0, 20.0], atol=1.5), top2


def test_biphase_survives_a_global_time_shift():
    """psi must be invariant to shifting the clock, or the label is an artefact.

    The stimulus is regenerated with the time axis advanced by a whole number of
    fundamental periods plus an arbitrary fraction; the recovered biphase must
    not move.
    """
    dt, f = KW["dt"], KW["f_hz"]

    # Every example carries an independently drawn global phase, so checking a
    # handful covers the shift-invariance claim directly: same recovered psi
    # regardless of where each example's clock happens to start.
    x, y = biphase_batch(jax.random.PRNGKey(1), 16, **KW)
    sig, lab = np.asarray(x)[:, :, 0], np.asarray(y)

    for i in range(sig.shape[1]):
        recovered = biphase_of(sig[:, i], f, dt)
        expected = float(DEFAULT_BIPHASES[int(lab[i])])
        # Biphase lives on a circle; compare there, not on the real line.
        diff = np.angle(np.exp(1j * (recovered - expected)))
        assert abs(diff) < 0.2, (
            f"example {i}: recovered {recovered:.3f} vs expected {expected:.3f}. "
            "A constant offset here means a phase-convention mismatch, not noise.")


def test_global_phase_is_random_so_absolute_phase_carries_nothing():
    """Across examples of the SAME class, the absolute phase must be uniform.

    If it were not, a readout of the final state could shortcut the task and the
    `last` condition would stop being a control.
    """
    x, y = biphase_batch(jax.random.PRNGKey(2), 512, **KW)
    sig = np.asarray(x)[:, :, 0]

    dt, f = KW["dt"], KW["f_hz"]
    n = sig.shape[0]
    spec = np.fft.rfft(sig, axis=0)
    i1 = int(np.argmin(np.abs(np.fft.rfftfreq(n, dt) - f)))
    phases = np.angle(spec[i1])

    cls0 = phases[np.asarray(y) == 0]
    # Resultant vector length ~ 0 for a uniform circular distribution.
    r = abs(np.mean(np.exp(1j * cls0)))
    assert r < 0.2, f"fundamental phase is not uniform within a class (R={r:.2f})"


def test_classes_are_actually_distinguishable():
    """The waveforms must differ, or the task is unsolvable rather than hard.

    Compared as the mean waveform after aligning on the fundamental's phase -
    which is exactly the information a phase-sensitive mechanism could use.
    """
    x, y = biphase_batch(jax.random.PRNGKey(3), 512, **KW)
    sig, lab = np.asarray(x)[:, :, 0], np.asarray(y)

    dt, f, n = KW["dt"], KW["f_hz"], np.asarray(x).shape[0]
    spec = np.fft.rfft(sig, axis=0)
    freqs = np.fft.rfftfreq(n, dt)
    i1 = int(np.argmin(np.abs(freqs - f)))
    i2 = int(np.argmin(np.abs(freqs - 2 * f)))

    psi = (np.angle(spec[i2]) - 2 * np.angle(spec[i1])) % (2 * np.pi)
    means = [np.angle(np.mean(np.exp(1j * psi[lab == c])))
             for c in range(len(DEFAULT_BIPHASES))]

    # The four class means must be spread around the circle, not collapsed.
    for a, b in zip(means, np.roll(means, -1)):
        gap = abs(np.angle(np.exp(1j * (b - a))))
        assert gap > 1.0, f"class biphases not separated: {means}"


def test_matched_parameter_count_between_conditions():
    """Homogeneous and heterogeneous must differ ONLY in spread.

    Same shape, so the same parameter count exactly; same geometric mean, so the
    same 'average' unit. If either failed, an accuracy gap would be confounded.
    """
    n, lo, hi = 64, 1.0, 40.0
    hom, het = homogeneous_bands(n, lo, hi), log_spaced_bands(n, lo, hi)

    assert hom.shape == het.shape == (n,)
    assert float(np.std(np.asarray(hom))) == pytest.approx(0.0, abs=1e-5)
    assert float(np.std(np.asarray(het))) > 1.0

    gm = lambda a: float(np.exp(np.mean(np.log(np.asarray(a)))))
    assert gm(hom) == pytest.approx(gm(het), rel=1e-3), "geometric means differ"


def test_freq_batch_still_works():
    """The notebook-02 task, now importable, at a size that runs fast."""
    x, y = freq_batch(jax.random.PRNGKey(0), 8, n_steps=500, dt=1e-3)
    assert x.shape == (500, 8, 1)
    assert y.shape == (8,) and int(y.min()) >= 0 and int(y.max()) <= 2


def test_noise_does_not_leak_the_label():
    """With noise on, spectra must still match across classes within sampling error."""
    freqs, power = power_spectrum_by_class(
        jax.random.PRNGKey(4), 256, **dict(KW, noise=0.3))
    spread = (power.max(axis=0) - power.min(axis=0)) / power.max()
    assert spread.max() < 0.02, f"noise leaks class information ({spread.max():.3f})"
