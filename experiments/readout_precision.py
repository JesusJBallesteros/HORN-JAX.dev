"""Readout precision: reproduce the analog paper's in simulation.

    python experiments/readout_precision.py                 # biphase, no data needed
    python experiments/readout_precision.py --task smnist   # row-wise sMNIST, downloads MNIST
    python experiments/readout_precision.py --quick         # test

THE QUESTION
Effenberger's paper [arXiv:2509.04064](https://arxiv.org/abs/2509.04064) transferred a trained HORN
to analog hardware. The dynamics matched, yet the digital readout agreed with its twin on only
28.39% of predictions; retraining a linear readout on the analog dynamics recovered full performance.
The information survived the precision loss, the readout did not.

This script reproduces that 'failure mode' with quantisation as the precision-loss stand-in:

  1. train a HORN + affine readout at float32
  2. re-run inference with the STATE quantised to n bits inside the recurrent loop,
     as analog hardware would evolve it, not just sampled coarsely at the output
  3. per bit depth, report
       acc_orig   : the float-trained affine readout applied to quantised dynamics
       acc_ridge  : a ridge readout retrained on the quantised features
       agreement  : fraction of predictions matching the float model, the paper's metric

The prediction, if the paper's account is right: acc_orig collapses well before the
dynamics stop being informative, and acc_ridge holds until only a few levels remain.

Quantisation is inside the loop on (x, v), symmetric uniform, ranges calibrated from a
float run (99.9th percentile per state variable). Pooling happens downstream on the
quantised trajectory, as a digital post-processing stage would.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from horn.core import init_state, step as horn_step
from horn.model import _pool, init_net, log_spaced_bands, usable_band
from horn.paths import results
from horn.tasks import DEFAULT_BIPHASES, biphase_batch
from horn.training import train


# quantize

def quantize(z, scale, bits):
    """Symmetric uniform quantiser to 2^bits levels over [-scale, scale]."""
    levels = 2 ** bits - 1                      # e.g. 3 bits -> 8 levels -> 7 intervals
    zc = jnp.clip(z, -scale, scale)             # saturate rather than wrap; wrapping would
                                                # turn a large excursion into a small one
    # Map [-scale, scale] onto [0, 1], round to the nearest level, then map back.
    return jnp.round((zc + scale) / (2 * scale) * levels) / levels * 2 * scale - scale


def run_traj(params, inputs, dt, bits=None, scales=None):
    """Trajectories (xs, vs), with the state optionally quantised INSIDE the loop.

    bits=None gives the full-precision reference run. Otherwise the state is rounded
    at every timestep, which is the analog case: the rounded value is what carries
    into the next step, so the error enters the dynamics and compounds.
    """
    n_osc = params.horn.log_omega.shape[0]
    state = init_state(n_osc, batch=inputs.shape[1])   # inputs are (T, B, in_size)

    def body(carry, u):
        new, _ = horn_step(params.horn, carry, u, dt)
        if bits is not None:
            # x and v get separate scales because their magnitudes differ by ~omega;
            # one shared scale would waste every level on one of the two.
            new = new._replace(x=quantize(new.x, scales[0], bits),
                               v=quantize(new.v, scales[1], bits))
        return new, (new.x, new.v)

    _, (xs, vs) = jax.lax.scan(body, state, inputs)
    return xs, vs


def features(params, xs, vs, pool):
    """Pool a trajectory into decoder features, using the model's own _pool."""
    omega = jnp.exp(params.horn.log_omega)
    # np.asarray pulls the result out of JAX: everything downstream is host-side
    # linear algebra with no tracing or gradients needed.
    return np.asarray(_pool(xs, vs / omega, pool))


def ridge_split(feats, labels, n_classes, seed=0):
    """Held-out accuracy of a retrained ridge readout, lambda selected internally.

    Same estimator family as probe_mechanism.py (bias column, per-feature
    scaling, the readout never sees its evaluation data), but with the
    regulariser chosen on a validation split instead of fixed. A fixed strong
    lambda underfits here: after training, the class signal can sit in
    low-variance feature directions that heavy shrinkage discards.
    """
    # Scale features FIRST, append the bias column after. Scaling a constant
    # column of ones by its (zero) spread creates a 1e9 column that poisons the
    # normal equations; that bug produced chance-level ridge accuracies on
    # perfectly separable features before it was caught.
    Fs = feats / (feats.std(0, keepdims=True) + 1e-9)
    F = np.concatenate([Fs, np.ones((len(Fs), 1))], 1)
    idx = np.random.default_rng(seed).permutation(len(F))
    n = len(F)
    tr, va, te = idx[:n // 2], idx[n // 2:3 * n // 4], idx[3 * n // 4:]
    Y = np.eye(n_classes)[labels]

    def fit(lam):
        # lstsq, not solve: at 1 bit the features can go constant, and a
        # singular system should degrade to chance, not crash the sweep
        A = np.concatenate([F[tr], np.sqrt(lam * len(tr)) * np.eye(F.shape[1])])
        B = np.concatenate([Y[tr], np.zeros((F.shape[1], n_classes))])
        return np.linalg.lstsq(A, B, rcond=None)[0]

    # Select lambda on the validation split, report on the untouched test split, so
    # the reported number is not inflated by having chosen the best of six on it.
    best, best_va = None, -1.0
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        W = fit(lam)
        acc_va = float((np.argmax(F[va] @ W, 1) == labels[va]).mean())
        if acc_va > best_va:
            best, best_va = W, acc_va
    pred = np.argmax(F[te] @ best, 1)
    return float((pred == labels[te]).mean()), te, pred


# tasks

def setup_biphase(quick):
    L, dt, n_osc = 400, 2.5e-3, 64
    f_lo, f_hi = usable_band(L, dt)
    n_classes = len(DEFAULT_BIPHASES)
    params = init_net(jax.random.PRNGKey(0), 1, n_osc, n_classes,
                      f_hz=log_spaced_bands(n_osc, f_lo, f_hi),
                      zeta=0.05, pool="mean", w_scale=3.0)

    def batch_fn(key, n):
        return biphase_batch(key, n, f_hz=10.0, n_steps=L, dt=dt, noise=0.05)

    steps = 60 if quick else 400
    n_eval = 300 if quick else 1200
    return params, batch_fn, dt, "mean", n_classes, steps, n_eval


def setup_smnist(quick):
    from horn.data import load_mnist                       # downloads on first use
    xtr, ytr, xte, yte = load_mnist()
    L, T = 28, 0.28
    dt = T / L
    f_lo, f_hi = usable_band(L, dt)
    n_osc = 96
    params = init_net(jax.random.PRNGKey(0), 28, n_osc, 10,
                      f_hz=log_spaced_bands(n_osc, f_lo, f_hi),
                      zeta=0.15, pool="meanrms")
    # (N, 28, 28) -> (28, N, 28): rows become timesteps, so the image is presented as
    # a sequence of 28 row-vectors. transpose(1, 0, 2) puts time on the leading axis.
    XTR = jnp.asarray(xtr.transpose(1, 0, 2))
    YTR = jnp.asarray(ytr)
    XTE = jnp.asarray(xte.transpose(1, 0, 2))
    YTE = jnp.asarray(yte)

    def batch_fn(key, n):
        idx = jax.random.randint(key, (n,), 0, XTR.shape[1])   # sample with replacement
        return XTR[:, idx, :], YTR[idx]                        # index the BATCH axis

    steps = 100 if quick else 1500
    n_eval = 500 if quick else 2000
    # Stashed on the function object so main() can reach the real held-out split
    # without threading another return value through both setup functions.
    setup_smnist.eval_data = (XTE[:, :n_eval, :], YTE[:n_eval])
    return params, batch_fn, dt, "meanrms", 10, steps, n_eval


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["biphase", "smnist"], default="biphase")
    ap.add_argument("--bits", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    setup = setup_biphase if args.task == "biphase" else setup_smnist
    params, batch_fn, dt, pool, n_classes, steps, n_eval = setup(args.quick)

    print(f"training ({args.task}, {steps} steps) ...")
    params, hist = train(params, batch_fn, dt, pool, steps=steps,
                         batch=64, lr=3e-3)

    # evaluation set, fixed across all bit depths
    if args.task == "biphase":
        X, Y = batch_fn(jax.random.PRNGKey(999), n_eval)
    else:
        X, Y = setup_smnist.eval_data
    Y = np.asarray(Y)

    # Full-precision reference run. It serves three purposes: the accuracy ceiling,
    # the predictions that "agreement" is measured against, and the quantiser range.
    xs, vs = run_traj(params, X, dt)
    # 99.9th percentile, not the max: one outlier trajectory would otherwise stretch
    # the range and waste every level on values the population never visits.
    scales = (float(jnp.quantile(jnp.abs(xs), 0.999)),
              float(jnp.quantile(jnp.abs(vs), 0.999)))
    feats = features(params, xs, vs, pool)
    logits = feats @ np.asarray(params.readout.W).T + np.asarray(params.readout.b)
    pred_float = np.argmax(logits, 1)       # the twin's predictions, per the paper's metric
    acc_float = float((pred_float == Y).mean())
    print(f"float32 test acc {acc_float:.3f}   "
          f"(chance {1/n_classes:.3f})   scales x {scales[0]:.3g} v {scales[1]:.3g}")

    def readout_acc(f):
        """Apply the UNCHANGED full-precision decoder to degraded features.

        Returns (accuracy, agreement with the full-precision model). Agreement is
        the paper's metric and is not the same as accuracy: a model can disagree
        with its twin while still being right, and both can be wrong together.
        """
        logit = f @ np.asarray(params.readout.W).T + np.asarray(params.readout.b)
        pred = np.argmax(logit, 1)
        return float((pred == Y).mean()), float((pred == pred_float).mean())

    records = []
    print(f"\n{'bits':>4} {'in-loop:':>9} {'orig':>6} {'ridge':>6} {'agree':>6}"
          f"  {'sampled:':>9} {'orig':>6} {'ridge':>6}")
    for b in args.bits:
        # in-loop: the state itself evolves at n bits, the analog situation
        xs_q, vs_q = run_traj(params, X, dt, bits=b, scales=scales)
        fq = features(params, xs_q, vs_q, pool)
        acc_orig, agree = readout_acc(fq)
        acc_ridge, _, _ = ridge_split(fq, Y, n_classes)

        # sampled: float dynamics, quantised only at observation, the ADC control.
        # If only this mattered, precision would be a measurement problem, not a
        # dynamics problem; pooling over T steps averages most of it away.
        fs_ = features(params, quantize(xs, scales[0], b),
                       quantize(vs, scales[1], b), pool)
        acc_s, _ = readout_acc(fs_)
        acc_s_ridge, _, _ = ridge_split(fs_, Y, n_classes)

        records.append({"bits": b, "acc_orig": acc_orig, "acc_ridge": acc_ridge,
                        "agreement": agree, "acc_sampled": acc_s,
                        "acc_sampled_ridge": acc_s_ridge})
        print(f"{b:>4} {'':>9} {acc_orig:>6.3f} {acc_ridge:>6.3f} {agree:>6.3f}"
              f"  {'':>9} {acc_s:>6.3f} {acc_s_ridge:>6.3f}")

    tag = args.task + ("_quick" if args.quick else "")
    out = results(f"readout_precision_{tag}.json")
    out.write_text(json.dumps({"task": args.task, "acc_float": acc_float,
                               "chance": 1 / n_classes, "scales": scales,
                               "records": records}, indent=2))
    print(f"\nwrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bits = [r["bits"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(bits, [r["acc_orig"] for r in records], "o-",
            label="in-loop, float-trained readout", color="tab:red")
    ax.plot(bits, [r["acc_ridge"] for r in records], "o-",
            label="in-loop, retrained ridge", color="tab:green")
    ax.plot(bits, [r["acc_sampled"] for r in records], "s--",
            label="sampled only, float-trained readout", color="tab:orange")
    ax.axhline(acc_float, ls="--", c="k", lw=1, label=f"float32 ({acc_float:.2f})")
    ax.axhline(1 / n_classes, ls=":", c="k", lw=1, label="chance")
    ax.set_xlabel("state precision (bits)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Precision inside the dynamics vs at the output")
    ax = axes[1]
    ax.plot(bits, [r["agreement"] for r in records], "o-", color="tab:blue")
    ax.axhline(1 / n_classes, ls=":", c="k", lw=1)
    ax.set_xlabel("state precision (bits)")
    ax.set_ylabel("agreement with float model")
    ax.set_ylim(0, 1.02)
    ax.set_title("The paper's metric: prediction agreement (in-loop)")
    fig.suptitle(f"task: {args.task}", fontsize=10)
    fig.tight_layout()
    fig.savefig(results(f"readout_precision_{tag}.png"), dpi=130)
    print(f"wrote {results(f'readout_precision_{tag}.png')}")


if __name__ == "__main__":
    main()
