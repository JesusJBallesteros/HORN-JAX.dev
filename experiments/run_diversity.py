"""Frequency heterogeneity, can a HORN read a biphase?

    python experiments/run_diversity.py --quick        # ~2 min, smoke test
    python experiments/run_diversity.py                # the real grid
    python experiments/run_diversity.py --task freq    # the phase-free control task

1. Does a population with heterogeneous omega beat a homogeneous one at MATCHED
   parameter count?
2. Is there a task where power pooling provably loses, so that phase is doing
   work rather than being asserted to?

The label is the phase of the second harmonic relative to twice the phase of the first. 
Recovering it requires a product of a unit tuned near f
with one tuned near 2f, which needs BOTH:
a nonlinearity to form the product
 and
units at BOTH frequencies to form it from.

So heterogeneity is a precondition.

PREDICTIONS
`experiments/probe_mechanism.py` measures separability at INITIALISATION, before
any training. Two of three original predictions were wrong:

  * `W_rec = 0` is at chance at every amplitude. A bank of
    independent resonators cannot represent a biphase.
  * `rms` is at chance only while the system is LINEAR. Once the nonlinearity is
    engaged it converts biphase into power and rms climbs to 0.65.
  * `last` is NOT at chance once recurrence is engaged, it reaches 1.00. The
    random global phase does not protect it, because the population encodes the
    biphase in a way a linear readout can reach.

So the H1 being W_rec = 0, and the experiment:

    W_rec = 0, any pooling, any amplitude   -> chance      (filter bank)
    W_rec free, states of order 0.1+        -> above chance
    heterogeneous > homogeneous             -> the diversity question

AMPLITUDE IS NOT A FREE PARAMETER HERE. At w_scale = 0.1 the states are ~0.014
and tanh is linear to a part in 400; nothing works and the result would be a grid
of chance values that says nothing about diversity. Default raised accordingly.

MATCHING
Homogeneous and heterogeneous differ only in the spread of omega: same n_osc, so
identical parameter count, and the same geometric-mean frequency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from horn.model import init_net, log_spaced_bands, usable_band
from horn.paths import results
from horn.tasks import DEFAULT_BIPHASES, biphase_batch, freq_batch, homogeneous_bands
from horn.training import binomial_sd, evaluate_stream, train


def build(key, cfg, omega_kind, recurrence):
    """One network. `omega_kind` and `recurrence` are the two variables."""
    bands = (log_spaced_bands if omega_kind == "heterogeneous" else homogeneous_bands)
    params = init_net(key, 1, cfg["n_osc"], cfg["n_classes"],
                      f_hz=bands(cfg["n_osc"], cfg["f_lo"], cfg["f_hi"]),
                      zeta=cfg["zeta"], pool=cfg["pool_widest"],
                      w_scale=cfg["w_scale"], rec_gain=cfg["rec_gain"])

    if recurrence == "off":
        # Zero AND frozen (see the caller) = a bank of genuinely independent
        # resonators. This is the filter-bank control: no coupling means no
        # products between units, so no biphase can be represented at all.
        params = params._replace(
            horn=params.horn._replace(W_rec=jnp.zeros_like(params.horn.W_rec)))
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["biphase", "freq"], default="biphase")
    ap.add_argument("--n-osc", type=int, default=64)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--zeta", type=float, default=0.05)
    # 0.1 (the model default) leaves the recurrent tanh linear and the biphase
    # mechanism unavailable - see probe_mechanism.py. 3.0 gives rms|x| ~ 0.4.
    ap.add_argument("--w-scale", type=float, default=3.0)
    # "flat" reproduces the run that found nothing: with input_gain="normalised"
    # the external drive is ~5000x the recurrent drive, so W_rec on/off changes
    # the logits by 4e-4 and the recurrence variable is inert. "normalised"
    # applies the same 2*zeta*omega^2 factor to W_rec. Run both, in that order.
    ap.add_argument("--rec-gain", choices=["flat", "normalised"], default="normalised")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=400, help="sequence length L")
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--f-hz", type=float, default=10.0, help="fundamental")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--pools", nargs="+", default=["rms", "meanrms", "last"])
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid for a smoke test, not for reporting")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.quick:
        args.n_osc, args.steps, args.seeds, args.n_steps = 24, 60, 2, 200
        args.eval_n, args.pools = 256, ["rms", "meanrms"]

    f_lo, f_hi = usable_band(args.n_steps, args.dt)
    n_classes = len(DEFAULT_BIPHASES) if args.task == "biphase" else 3

    if args.task == "biphase":
        def batch_fn(key, n):
            return biphase_batch(key, n, f_hz=args.f_hz, n_steps=args.n_steps,
                                 dt=args.dt, noise=args.noise)
        needed = (args.f_hz, 2 * args.f_hz)
    else:
        def batch_fn(key, n):
            return freq_batch(key, n, n_steps=args.n_steps, dt=args.dt)
        needed = (8.0, 32.0, 150.0)

    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"task={args.task}  L={args.n_steps}  dt={args.dt*1e3:.2f} ms  "
          f"T={args.n_steps*args.dt:.2f} s")
    print(f"usable band {f_lo:.2f}-{f_hi:.1f} Hz (ratio {f_hi/f_lo:.0f})")
    print(f"stimulus needs {needed} Hz -> "
          f"{'all inside' if all(f_lo <= f <= f_hi for f in needed) else 'OUTSIDE THE BAND'}")
    print(f"chance = {1/n_classes:.3f}, eval sd at n={args.eval_n} is "
          f"{binomial_sd(1/n_classes, args.eval_n):.4f}\n")

    cfg = dict(n_osc=args.n_osc, n_classes=n_classes, f_lo=f_lo, f_hi=f_hi,
               zeta=args.zeta, w_scale=args.w_scale, rec_gain=args.rec_gain,
               pool_widest="meanrms")

    records = []
    grid = list(itertools.product(args.pools, ["heterogeneous", "homogeneous"],
                                 ["free", "off"], range(args.seeds)))
    t_start = time.time()

    for i, (pool, omega_kind, recurrence, seed) in enumerate(grid, 1):
        # The readout width depends on the pooling, so the net must be built with
        # the pooling it will be trained with.
        cfg_run = dict(cfg, pool_widest=pool)
        params = build(jax.random.PRNGKey(seed), cfg_run, omega_kind, recurrence)

        params, hist = train(params, batch_fn, args.dt, pool, steps=args.steps,
                             batch=args.batch, lr=args.lr, seed=seed,
                             freeze_rec=(recurrence == "off"), log=False)
        acc = evaluate_stream(params, batch_fn, args.dt, pool, n=args.eval_n)

        records.append(dict(pool=pool, omega=omega_kind, recurrence=recurrence,
                            seed=seed, acc=acc, final_loss=hist["loss"][-1],
                            gnorm_final=hist["gnorm"][-1], secs=hist["secs"]))
        print(f"[{i:>3}/{len(grid)}] {pool:>7} | omega {omega_kind:<13} | "
              f"W_rec {recurrence:<4} | seed {seed} -> acc {acc:.3f}  "
              f"({hist['secs']:.0f}s)")

    print(f"\ntotal {time.time() - t_start:.0f}s")
    summarise(records, n_classes, args.eval_n)

    tag = args.tag or (f"{args.task}_n{args.n_osc}_L{args.n_steps}"
                       f"_rec{args.rec_gain}")
    out = results(f"diversity_{tag}.json")
    out.write_text(json.dumps({"config": vars(args), "n_classes": n_classes,
                               "f_lo": f_lo, "f_hi": f_hi,
                               "records": records}, indent=2))
    print(f"\nwrote {out}")
    plot(records, n_classes, results(f"diversity_{tag}.png"), tag)
    print(f"wrote {results(f'diversity_{tag}.png')}")


def summarise(records, n_classes, eval_n):
    chance = 1.0 / n_classes
    sd = binomial_sd(chance, eval_n)
    print(f"\n{'pool':>8} {'omega':>14} {'W_rec':>6}   mean    sd     sigma above chance")
    print("-" * 68)
    for pool in dict.fromkeys(r["pool"] for r in records):
        for omega_kind in ["heterogeneous", "homogeneous"]:
            for rec in ["free", "off"]:
                accs = [r["acc"] for r in records
                        if r["pool"] == pool and r["omega"] == omega_kind
                        and r["recurrence"] == rec]
                if not accs:
                    continue
                m, s = float(np.mean(accs)), float(np.std(accs))
                # Sigma against the sampling noise of a single evaluation. Crude,
                # but enough to separate "clearly above chance" from "noise".
                print(f"{pool:>8} {omega_kind:>14} {rec:>6}   {m:.3f}  {s:.3f}   "
                      f"{(m - chance) / sd:>6.1f}")


def plot(records, n_classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = list(dict.fromkeys(r["pool"] for r in records))
    conds = [("heterogeneous", "free"), ("heterogeneous", "off"),
             ("homogeneous", "free"), ("homogeneous", "off")]
    labels = ["het / W_rec free", "het / W_rec off", "hom / W_rec free", "hom / W_rec off"]

    fig, ax = plt.subplots(figsize=(1.9 * len(pools) + 4, 4))
    width = 0.8 / len(conds)

    for j, ((omega_kind, rec), lab) in enumerate(zip(conds, labels)):
        means, errs = [], []
        for pool in pools:
            accs = [r["acc"] for r in records if r["pool"] == pool
                    and r["omega"] == omega_kind and r["recurrence"] == rec]
            means.append(np.mean(accs) if accs else np.nan)
            errs.append(np.std(accs) if accs else 0.0)
        xs = np.arange(len(pools)) + (j - (len(conds) - 1) / 2) * width
        ax.bar(xs, means, width, yerr=errs, capsize=3, label=lab)

    ax.axhline(1.0 / n_classes, color="k", ls=":", lw=1.2,
               label=f"chance = {1/n_classes:.2f}")
    ax.set(xticks=np.arange(len(pools)), ylabel="test accuracy",
           title=f"Frequency diversity and phase readout: {title}", ylim=(0, 1.02))
    ax.set_xticklabels(pools)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
