"""Is the mechanism even available? Ask before spending GPU hours on training.

    python experiments/probe_mechanism.py

The biphase task requires the network to form a product between a unit near f and
one near 2f. Two things must hold for that product to exist at all:

  recurrence   W_rec must be non-zero, or the units never see each other
  amplitude    states must be large enough that tanh is measurably nonlinear;
               at |x| ~ 0.01 it is linear to a part in 400 and the recurrent path
               is just a linear filter

This script measures both, and then asks the only question that matters: are the
pooled features LINEARLY SEPARABLE by class, in an untrained network? If they are
not separable at initialisation, training the readout cannot help, and a sweep
would produce a grid of chance-level numbers and no information.

A note on how this is measured. The first version of this probe reported the
ridge classifier's TRAINING accuracy over 129 features and 600 samples, which
memorises: it reported 1.000 for conditions that are provably at chance. The
split below is the fix, and the lesson generalises - a separability probe with no
held-out data measures capacity, not structure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from horn.core import init_state, step as horn_step
from horn.model import init_net, log_spaced_bands, usable_band
from horn.tasks import DEFAULT_BIPHASES, biphase_batch

L, DT, N_OSC = 400, 2.5e-3, 64
F_HZ, NOISE, N_PER_CLASS = 10.0, 0.05, 400


def pooled_features(params, psi, n, key):
    """Run the network on one biphase class; return each pooling's features."""
    x, _ = biphase_batch(key, n, f_hz=F_HZ, n_steps=L, dt=DT, noise=NOISE,
                         biphases=(psi,))
    state = init_state(N_OSC, batch=n)

    def body(carry, u):
        new, _ = horn_step(params.horn, carry, u, DT)
        return new, (new.x, new.v)

    _, (xs, vs) = jax.lax.scan(body, state, x)
    omega = jnp.exp(params.horn.log_omega)
    xs, vs = np.asarray(xs), np.asarray(vs / omega)     # v/omega, as the readout sees it

    return {
        "mean": np.concatenate([xs.mean(0), vs.mean(0)], -1),
        "rms": np.concatenate([np.sqrt((xs ** 2).mean(0)),
                               np.sqrt((vs ** 2).mean(0))], -1),
        "last": np.concatenate([xs[-1], vs[-1]], -1),
    }, xs


def heldout_separability(feats, labels, n_classes, ridge=1e-2, seed=0):
    """Held-out accuracy of a ridge one-vs-rest readout.

    An upper bound on what any linear readout on these features could achieve,
    obtained without training the recurrent weights at all.
    """
    # Scale features FIRST, append the bias column after. Scaling a constant
    # column of ones by its (zero) spread creates a 1e9 column that poisons
    # the normal equations and deflates every accuracy this probe reports.
    Fs = feats / (feats.std(0, keepdims=True) + 1e-9)
    F = np.concatenate([Fs, np.ones((len(Fs), 1))], 1)

    idx = np.random.default_rng(seed).permutation(len(F))
    half = len(F) // 2
    tr, te = idx[:half], idx[half:]

    Y = np.eye(n_classes)[labels]
    A = np.concatenate([F[tr], np.sqrt(ridge * half) * np.eye(F.shape[1])])
    B = np.concatenate([Y[tr], np.zeros((F.shape[1], n_classes))])
    W = np.linalg.lstsq(A, B, rcond=None)[0]
    return float((np.argmax(F[te] @ W, 1) == labels[te]).mean())


def main():
    f_lo, f_hi = usable_band(L, DT)
    n_classes = len(DEFAULT_BIPHASES)
    chance = 1.0 / n_classes
    sd = np.sqrt(chance * (1 - chance) / (n_classes * N_PER_CLASS // 2))

    print(f"biphase task: {n_classes} classes, chance {chance:.3f}, "
          f"held-out sd {sd:.4f}")
    print(f"band {f_lo:.2f}-{f_hi:.1f} Hz, stimulus at {F_HZ:.0f} and "
          f"{2*F_HZ:.0f} Hz\n")
    print(f"{'w_scale':>7} {'W_rec':>6} {'rms|x|':>8} {'nonlin':>8} | "
          f"{'mean':>6} {'rms':>6} {'last':>6}")
    print("-" * 62)

    for w_scale in [0.1, 1.0, 3.0, 10.0]:
        for recurrence in ["free", "zero"]:
            params = init_net(jax.random.PRNGKey(0), 1, N_OSC, n_classes,
                              f_hz=log_spaced_bands(N_OSC, f_lo, f_hi),
                              zeta=0.05, pool="meanrms", w_scale=w_scale)
            if recurrence == "zero":
                params = params._replace(horn=params.horn._replace(
                    W_rec=jnp.zeros_like(params.horn.W_rec)))

            store, labels = {k: [] for k in ("mean", "rms", "last")}, []
            for c, psi in enumerate(DEFAULT_BIPHASES):
                feats, xs = pooled_features(params, psi, N_PER_CLASS,
                                            jax.random.PRNGKey(100 + c))
                for k in store:
                    store[k].append(feats[k])
                labels += [c] * N_PER_CLASS
            labels = np.array(labels)

            flat = xs.ravel()
            nonlin = (np.abs(np.tanh(flat) - flat).mean()
                      / (np.abs(flat).mean() + 1e-12))
            acc = {k: heldout_separability(np.concatenate(v), labels, n_classes)
                   for k, v in store.items()}

            print(f"{w_scale:>7.1f} {recurrence:>6} "
                  f"{np.sqrt((xs**2).mean()):>8.3f} {nonlin:>8.1e} | "
                  f"{acc['mean']:>6.3f} {acc['rms']:>6.3f} {acc['last']:>6.3f}")

    print(f"\nchance is {chance:.3f}. W_rec=zero rows should sit there at EVERY")
    print("amplitude - that is the filter-bank control. If they do not, the claim")
    print("that a bank of independent resonators cannot represent a biphase is")
    print("wrong, and the experiment needs rethinking before it is run.")


if __name__ == "__main__":
    main()
