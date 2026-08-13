"""Visual test: what damping and frequency actually do.

Produces two panels:
  LEFT  - the same oscillator at four damping ratios, showing how zeta sets
          the memory horizon of a unit.
  RIGHT - four oscillators at different natural frequencies, showing that a
          heterogeneous population is a filter bank.

Run:  python demo.py    ->  writes results/demo.png

The figure goes to results/ inside the repo, wherever this is run from - see
horn/paths.py for why that is not the same as the working directory.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend: renders straight to file.
                               # Required when running headless (WSL without an X server),
                               # and MUST be set before importing pyplot.
import matplotlib.pyplot as plt

from horn.core import HORNParams, HORNState, run_sequence
from horn.paths import results

TWOPI = 2 * np.pi

def w_rads(f_hz):   # Conversion Hz -> rad/s
    return TWOPI * np.asarray(f_hz, dtype=float)

def isolated(n, f_hz, zeta):
    """A HORN with coupling zeroed out -> n independent oscillators.

    Same helper as in the tests: with W_in and W_rec both zero the drive term
    vanishes and each unit is a pure damped harmonic oscillator, which is what
    this figure needs to show.

    NOTE the argument is in HERTZ and is converted to rad/s on the way in, per the
    repo-wide convention. Passing a rad/s value here would run the unit 6.28x fast.
    """
    return HORNParams(
        W_in=jnp.zeros((n, 1)),                                             # no input drive
        W_rec=jnp.zeros((n, n)),                                            # no coupling
        log_omega=jnp.log(jnp.asarray(w_rads(f_hz), jnp.float32) * jnp.ones(n)),   # omega -> log
        log_zeta=jnp.log(jnp.asarray(zeta, jnp.float32) * jnp.ones(n)),     # zeta  -> log
    )


dt, T = 5e-5, 40000                  # time step and number of steps, so 2 seconds total
t = jnp.arange(T) * dt              # time axis in seconds, for plotting
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))   # one row, two panels, size (inches)

# LEFT PANEL: damping controls how long a unit remembers
for zeta, label in [(0.02, "underdamped  z=0.02"),   # rings for a long time = long memory
                    (0.2,  "damped  z=0.2"),    # a few visible oscillations
                    (1.0,  "critical     z=1.0"),    # fastest return with no overshoot
                    (2.5,  "overdamped   z=2.5")]:   # sluggish crawl, never crosses zero
    p = isolated(1, 8, zeta)                              # one unit, held at 8 Hz
    s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))     # released from rest at x=1
    _, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)      # no input: pure free decay
    ax[0].plot(t, xs[:, 0], label=label, lw=1.4)            # column 0 = the single unit

ax[0].set(title="Damping controls memory horizon", xlabel="time (s)", ylabel="x")
ax[0].legend(fontsize=8, frameon=False)
ax[0].axhline(0, color="k", lw=0.5)   # zero line, to make overshoot easy to see

# RIGHT PANEL: heterogeneous natural frequencies = a bank of filters.
# These are HERTZ, not rad/s: `isolated` converts. Octave spacing (each double the
# last) so the ratios can be read straight off the traces.
freqs_hz = jnp.array([2.0, 4.0, 8.0, 32.0])
p = isolated(4, freqs_hz, 0.02)            # four units, one per frequency, lightly damped
s0 = HORNState(x=jnp.ones((4,)), v=jnp.zeros((4,)))   # all four released together
_, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)    # xs is (T, 4): one column per unit

for i, f in enumerate(freqs_hz):
    ax[1].plot(t, xs[:, i] - 2.5 * i,      # subtract 2.5*i to stack the traces vertically
               lw=1.2, label=f"f={float(f):.0f} Hz")

ax[1].set(title="Heterogeneous frequencies = a filter bank",
          xlabel="time (s)", yticks=[])    # y ticks are meaningless once traces are offset
ax[1].legend(fontsize=8, frameon=False)

plt.tight_layout()                  # stop labels overlapping between panels
out = results("demo.png")           # always inside the repo, never the cwd
plt.savefig(out, dpi=130)           # write to disk (no plt.show(): headless backend)
print(f"wrote {out}")
