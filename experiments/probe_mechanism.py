"""Is the mechanism even available?

    python experiments/probe_mechanism.py                 # both gain settings
    python experiments/probe_mechanism.py --rec-gain flat # just the old one

The biphase task requires the network to form a product between a unit near f and
one near 2f. Two things must hold for that product to exist at all:

  recurrence   W_rec must be non-zero, or the units never see each other
  amplitude    states must be large enough that tanh is measurably nonlinear;
               at |x| ~ 0.01 it is linear to a part in 400 and the recurrent path
               is just a linear filter

This script measures both, then asks the only question that matters: are the
pooled features LINEARLY SEPARABLE by class, in an untrained network? If they are
not separable at initialisation, training the readout cannot help, and a sweep
would produce a grid of chance-level numbers and no information.

IT RUNS TWICE
`rec_gain="flat"` (the default) leaves the recurrent drive ~5000x weaker than the
external drive, because input_gain="normalised" multiplies W_in by 2*zeta*omega^2 
and nothing multiplies W_rec. Zeroing W_rec then changes the logits by 4e-4 relative:
the "recurrence on/off" variable does nothing, and the sweep built on it measured nothing.

`rec_gain="normalised"` applies the same factor to W_rec. Both are run here, in
that order, so the fix can be judged against the thing it was meant to fix rather
than asserted. Artefacts are named `..._recflat` and `..._recnormalised`.

I left them so I can compare the two, but the flat version is known to be broken and
should be ignored.

MEASUREMENT NOTE
An early version reported the ridge classifier's TRAINING accuracy over 129
features and 600 samples, which memorises: it reported 1.000 for conditions that
are provably at chance. A second version scaled a constant bias column by its
zero spread, creating a 1e9 column that poisoned the normal equations. Both are
fixed below. A separability probe without held-out data measures capacity, not
structure; one that standardises a constant column measures floating point.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from horn.core import init_state, step as horn_step
from horn.model import forward, init_net, log_spaced_bands, usable_band
from horn.report import Report
from horn.tasks import DEFAULT_BIPHASES, biphase_batch

L, DT, N_OSC = 400, 2.5e-3, 64
F_HZ, NOISE, N_PER_CLASS = 10.0, 0.05, 400
W_SCALES = (0.1, 1.0, 3.0, 10.0)


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
    # column of ones by its (zero) spread creates a 1e9 column that poisons the
    # normal equations and deflates every accuracy this probe reports.
    scaled = feats / (feats.std(0, keepdims=True) + 1e-9)
    F = np.concatenate([scaled, np.ones((len(scaled), 1))], 1)

    idx = np.random.default_rng(seed).permutation(len(F))
    half = len(F) // 2
    tr, te = idx[:half], idx[half:]

    Y = np.eye(n_classes)[labels]
    A = np.concatenate([F[tr], np.sqrt(ridge * half) * np.eye(F.shape[1])])
    B = np.concatenate([Y[tr], np.zeros((F.shape[1], n_classes))])
    W = np.linalg.lstsq(A, B, rcond=None)[0]
    return float((np.argmax(F[te] @ W, 1) == labels[te]).mean())


def recurrence_leverage(params, f_lo, f_hi, n_classes, rec_gain):
    """How much does switching recurrence off actually change the output?

    This is the number that exposed the problem. If it is ~1e-4, "W_rec on/off"
    is not an independent variable and any experiment using it is void.
    """
    x, _ = biphase_batch(jax.random.PRNGKey(7), 32, f_hz=F_HZ, n_steps=L,
                         dt=DT, noise=NOISE)
    zeroed = params._replace(horn=params.horn._replace(
        W_rec=jnp.zeros_like(params.horn.W_rec)))

    a = forward(params, x, DT, "meanrms")
    b = forward(zeroed, x, DT, "meanrms")
    rel = float(jnp.abs(a - b).max() / (jnp.abs(a).max() + 1e-12))

    drive_ratio = float(jnp.abs(params.horn.W_in).mean()
                        / (jnp.abs(params.horn.W_rec).mean() + 1e-12))
    return rel, drive_ratio


def run_one(rep, rec_gain, n_classes, f_lo, f_hi):
    """The full amplitude x recurrence grid for one gain setting."""
    rep.print(f"\n{'='*72}")
    rep.print(f"rec_gain = {rec_gain}")
    rep.print(f"{'='*72}")
    rep.print(f"{'w_scale':>7} {'W_rec':>6} {'rms|x|':>8} {'nonlin':>8} "
              f"{'W_in/W_rec':>11} {'leverage':>9} | {'mean':>6} {'rms':>6} {'last':>6}")
    rep.print("-" * 88)

    rows = []
    for w_scale in W_SCALES:
        base = init_net(jax.random.PRNGKey(0), 1, N_OSC, n_classes,
                        f_hz=log_spaced_bands(N_OSC, f_lo, f_hi),
                        zeta=0.05, pool="meanrms", w_scale=w_scale,
                        rec_gain=rec_gain)
        lev, drive_ratio = recurrence_leverage(base, f_lo, f_hi, n_classes, rec_gain)

        for recurrence in ["free", "zero"]:
            params = base
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

            rep.print(f"{w_scale:>7.1f} {recurrence:>6} "
                      f"{np.sqrt((xs**2).mean()):>8.3f} {nonlin:>8.1e} "
                      f"{drive_ratio:>11.0f} {lev:>9.1e} | "
                      f"{acc['mean']:>6.3f} {acc['rms']:>6.3f} {acc['last']:>6.3f}")

            rows.append(dict(rec_gain=rec_gain, w_scale=w_scale,
                             recurrence=recurrence, rms_x=float(np.sqrt((xs**2).mean())),
                             nonlin=float(nonlin), drive_ratio=drive_ratio,
                             leverage=lev, **{f"acc_{k}": v for k, v in acc.items()}))
    return rows


def plot(rows, n_classes, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = list(dict.fromkeys(r["rec_gain"] for r in rows))
    fig, axes = plt.subplots(1, len(gains) + 1,
                             figsize=(4.6 * (len(gains) + 1), 3.8))
    chance = 1.0 / n_classes

    for ax, gain in zip(axes, gains):
        sub = [r for r in rows if r["rec_gain"] == gain]
        for recurrence, style in [("free", "-o"), ("zero", "--s")]:
            s = [r for r in sub if r["recurrence"] == recurrence]
            ax.semilogx([r["w_scale"] for r in s], [r["acc_mean"] for r in s],
                        style, label=f"W_rec {recurrence}")
        ax.axhline(chance, color="k", ls=":", lw=1, label=f"chance {chance:.2f}")
        ax.set(xlabel="w_scale", ylabel="held-out separability (mean pooling)",
               title=f"rec_gain = {gain}", ylim=(0, 1.05))
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)

    # The diagnostic that exposed the problem, on its own axis.
    ax = axes[-1]
    for gain in gains:
        s = [r for r in rows if r["rec_gain"] == gain and r["recurrence"] == "free"]
        ax.loglog([r["w_scale"] for r in s], [r["leverage"] for r in s], "-o",
                  label=f"rec_gain = {gain}")
    ax.axhline(1e-2, color="C3", ls=":", lw=1)
    ax.text(0.98, 1.3e-2, "below this line, recurrence is not\nan independent variable",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color="C3")
    ax.set(xlabel="w_scale", ylabel="relative logit change when W_rec -> 0",
           title="Recurrence leverage")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")

    plt.tight_layout()
    rep.save_fig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-gain", nargs="+", default=["flat", "normalised"],
                    choices=["flat", "normalised"],
                    help="run these in order; default runs both for comparison")
    ap.add_argument("--tag", default="probe_mechanism")
    args = ap.parse_args()

    f_lo, f_hi = usable_band(L, DT)
    n_classes = len(DEFAULT_BIPHASES)
    chance = 1.0 / n_classes
    sd = np.sqrt(chance * (1 - chance) / (n_classes * N_PER_CLASS // 2))

    suffix = "_vs_".join(g[:4] for g in args.rec_gain) if len(args.rec_gain) > 1 \
        else args.rec_gain[0]
    name = f"{args.tag}_rec_{suffix}"

    with Report(name, "biphase separability at initialisation, no training") as rep:
        rep.print(f"biphase task: {n_classes} classes, chance {chance:.3f}, "
                  f"held-out sd {sd:.4f}")
        rep.print(f"band {f_lo:.2f}-{f_hi:.1f} Hz, stimulus at {F_HZ:.0f} "
                  f"and {2*F_HZ:.0f} Hz")
        rep.print(f"L={L}, dt={DT*1e3:.2f} ms, n_osc={N_OSC}, "
                  f"{N_PER_CLASS} examples per class")

        rows = []
        for rec_gain in args.rec_gain:
            rows += run_one(rep, rec_gain, n_classes, f_lo, f_hi)

        rep.print(f"\nchance is {chance:.3f}. Two things to read:")
        rep.print("  * W_rec=zero rows must sit at chance at EVERY amplitude - that is")
        rep.print("    the filter-bank control, and the claim the experiment rests on.")
        rep.print("  * `leverage` is how much zeroing W_rec changes the logits. Below")
        rep.print("    ~1e-2 the recurrence variable does nothing and any sweep over it")
        rep.print("    is measuring noise, however many seeds it averages.")

        rep.save_json(rows)
        plot(rows, n_classes, rep)


if __name__ == "__main__":
    main()
