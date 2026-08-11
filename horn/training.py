"""Training loop, lifted out of notebook 02 so experiments and tests can reuse it.

Nothing here is HORN-specific beyond the signature of `loss_and_acc`. It exists
because a training loop that lives only inside a notebook cell cannot be called
from a sweep, cannot be tested, and gets silently re-typed with small differences
every time it is needed.

`freeze_rec` is the addition that notebook 02 did not have. Zeroing the gradient
on W_rec holds the population at its initialisation - all-to-all coupling still
present but never adapted. Setting W_rec to zero outright at init AND freezing it
gives a bank of genuinely independent resonators, which is the control the
biphase task needs: a filter bank cannot form products between units, so it
cannot represent a biphase at all.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from horn.model import loss_and_acc


def make_train_step(opt, dt, pool, freeze_osc=False, freeze_rec=False):
    """Build the jitted update. Flags are closed over, so each combination
    compiles once rather than branching inside the traced function."""

    @jax.jit
    def train_step(params, opt_state, x, y):
        (loss, acc), grads = jax.value_and_grad(
            lambda q: loss_and_acc(q, x, y, dt, pool), has_aux=True)(params)

        horn_g = grads.horn
        if freeze_osc:
            horn_g = horn_g._replace(log_omega=jnp.zeros_like(horn_g.log_omega),
                                     log_zeta=jnp.zeros_like(horn_g.log_zeta))
        if freeze_rec:
            horn_g = horn_g._replace(W_rec=jnp.zeros_like(horn_g.W_rec))
        grads = grads._replace(horn=horn_g)

        # Pre-clip norm. After clipping it is pinned at the threshold and says
        # nothing; the gap between mean and max is the whole story of training
        # oscillatory dynamics.
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, acc, gnorm

    return train_step


def train(params, batch_fn, dt, pool, steps=400, batch=64, lr=3e-3, clip=1.0,
          freeze_osc=False, freeze_rec=False, seed=0, eval_every=50, log=True):
    """Train on an endless stream of freshly generated batches.

    Note that `batch_fn` synthesises new data every call, so there is no fixed
    training set: reported training accuracy is already a held-out estimate, just
    a noisy one at this batch size, and overfitting is impossible. That changes
    how the curves should be read - a train/test gap here is sampling noise, not
    memorisation.
    """
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    opt_state = opt.init(params)
    step_fn = make_train_step(opt, dt, pool, freeze_osc, freeze_rec)

    key = jax.random.PRNGKey(seed)
    hist = {"step": [], "loss": [], "acc": [], "gnorm": []}
    t0 = time.time()

    for i in range(steps):
        key, k_batch = jax.random.split(key)
        x, y = batch_fn(k_batch, batch)
        params, opt_state, loss, acc, gnorm = step_fn(params, opt_state, x, y)

        if i % eval_every == 0 or i == steps - 1:
            hist["step"].append(i)
            hist["loss"].append(float(loss))
            hist["acc"].append(float(acc))
            hist["gnorm"].append(float(gnorm))
            if log and (i % (eval_every * 4) == 0 or i == steps - 1):
                print(f"    step {i:>5}  loss {float(loss):.4f}  "
                      f"acc {float(acc):.3f}  |g| {float(gnorm):.2e}")

    hist["secs"] = time.time() - t0
    if log:
        print(f"    [{hist['secs']:.1f}s]")
    return params, hist


def evaluate_stream(params, batch_fn, dt, pool, n=2048, batch=256, seed=999):
    """Accuracy over freshly generated examples.

    For a synthetic task the honest evaluation is new samples from the generator,
    not a held-out slice - there is no finite dataset to hold out from. `seed` is
    fixed so the same evaluation set is used across conditions, which removes one
    source of variance from any comparison between them.
    """
    key = jax.random.PRNGKey(seed)
    correct = total = 0
    for start in range(0, n, batch):
        key, k = jax.random.split(key)
        size = min(batch, n - start)
        x, y = batch_fn(k, size)
        correct += float(loss_and_acc(params, x, y, dt, pool)[1]) * size
        total += size
    return correct / total


def evaluate_arrays(params, X, Y, dt, pool, chunk=500):
    """Accuracy over a fixed (T, N, F) array, in chunks to bound memory."""
    correct = 0
    for i in range(0, X.shape[1], chunk):
        size = X[:, i:i + chunk].shape[1]
        correct += float(loss_and_acc(params, X[:, i:i + chunk], Y[i:i + chunk],
                                      dt, pool)[1]) * size
    return correct / X.shape[1]


def chance_level(n_classes: int) -> float:
    return 1.0 / n_classes


def binomial_sd(p: float, n: int) -> float:
    """Standard deviation of an accuracy estimate, for judging whether a number
    is above chance. At n=2048 and 4 classes this is 0.0096, so anything under
    about 0.27 is not distinguishable from 0.25."""
    return float(np.sqrt(p * (1 - p) / n))
