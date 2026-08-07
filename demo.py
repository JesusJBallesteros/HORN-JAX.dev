"""Visual smoke test: what damping and frequency actually do.

Produces two panels:
  LEFT  - the same oscillator at four damping ratios, showing how zeta sets
          the memory horizon of a unit.
  RIGHT - four oscillators at different omega, showing that a heterogeneous
          population is a filter bank.

Run:  python demo.py    ->  writes demo.png
"""

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")          # non-interactive backend: renders straight to file.
                               # Required when running headless (WSL without an X server),
                               # and MUST be set before importing pyplot.
import matplotlib.pyplot as plt

from horn.core import HORNParams, HORNState, run_sequence


def isolated(n, omega, zeta):
    """A HORN with coupling zeroed out -> n independent oscillators.

    Same helper as in the tests: with W_in and W_rec both zero the drive term
    vanishes and each unit is a pure damped harmonic oscillator, which is what
    we want to visualise.
    """
    return HORNParams(
        W_in=jnp.zeros((n, 1)),                                             # no input drive
        W_rec=jnp.zeros((n, n)),                                            # no coupling
        log_omega=jnp.log(jnp.asarray(omega, jnp.float32) * jnp.ones(n)),   # omega -> log
        log_zeta=jnp.log(jnp.asarray(zeta, jnp.float32) * jnp.ones(n)),     # zeta  -> log
    )


dt, T = 1e-3, 8000                  # 1 ms steps for 8 s of simulated time
t = jnp.arange(T) * dt              # time axis in seconds, for plotting
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))   # one row, two panels

# ---------------------------------------------------------------------------
# LEFT PANEL: damping controls how long a unit remembers
# ---------------------------------------------------------------------------
for zeta, label in [(0.02, "underdamped  z=0.02"),   # rings for a long time = long memory
                    (0.3,  "underdamped  z=0.3"),    # a few visible oscillations
                    (1.0,  "critical     z=1.0"),    # fastest return with no overshoot
                    (3.0,  "overdamped   z=3.0")]:   # sluggish crawl, never crosses zero
    p = isolated(1, 2.0, zeta)                              # one unit, omega fixed at 2
    s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))     # released from rest at x=1
    _, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)      # no input: pure free decay
    ax[0].plot(t, xs[:, 0], label=label, lw=1.4)            # column 0 = the single unit

ax[0].set(title="Damping controls memory horizon", xlabel="time (s)", ylabel="x")
ax[0].legend(fontsize=8, frameon=False)
ax[0].axhline(0, color="k", lw=0.5)   # zero line, to make overshoot easy to see

# ---------------------------------------------------------------------------
# RIGHT PANEL: heterogeneous omega = a bank of filters
# ---------------------------------------------------------------------------
omegas = jnp.array([1.0, 2.0, 4.0, 8.0])   # octave spacing, so ratios are easy to read off
p = isolated(4, omegas, 0.05)              # four units, one per frequency, lightly damped
s0 = HORNState(x=jnp.ones((4,)), v=jnp.zeros((4,)))   # all four released together
_, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)    # xs is (T, 4): one column per unit

for i, w in enumerate(omegas):
    ax[1].plot(t, xs[:, i] - 2.5 * i,      # subtract 2.5*i to stack the traces vertically
               lw=1.2, label=f"w={float(w):.0f}")

ax[1].set(title="Heterogeneous omega = a learned filter bank",
          xlabel="time (s)", yticks=[])    # y ticks are meaningless once traces are offset
ax[1].legend(fontsize=8, frameon=False)

plt.tight_layout()                  # stop labels overlapping between panels
plt.savefig("demo.png", dpi=130)    # write to disk (no plt.show(): headless backend)
print("wrote demo.png")
