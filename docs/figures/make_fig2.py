"""Figure 2: what omega and zeta actually control.

Generated from the dynamics in horn/core.py rather than drawn, so the decay curves are
measurements. Panels (b) and (c) are the closed-form expressions the code is validated
against in E01, plotted over the range the experiments use.

    python docs/figures/make_fig2.py     ->  docs/figures/fig2_tuning.png
"""

import matplotlib
matplotlib.use("Agg")                 # headless; must precede the pyplot import
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from horn.core import HORNParams, HORNState, run_sequence
from horn.paths import REPO

TWOPI = 2 * np.pi
OUT = REPO / "docs" / "figures" / "fig2_tuning.png"


def isolated(n, f_hz, zeta):
    """Uncoupled, undriven units: both weight matrices zero, so each is a pure
    damped harmonic oscillator. f_hz is in HERTZ and converted here."""
    return HORNParams(
        W_in=jnp.zeros((n, 1)),
        W_rec=jnp.zeros((n, n)),
        log_omega=jnp.log(jnp.asarray(TWOPI * np.asarray(f_hz, float), jnp.float32)
                          * jnp.ones(n)),
        log_zeta=jnp.log(jnp.asarray(zeta, jnp.float32) * jnp.ones(n)),
    )


def steady_state_amplitude(f_drive, f0, zeta):
    """Closed-form response of a damped driven oscillator to a sinusoid at f_drive.
    On resonance this reduces to 1/(2*zeta*omega^2), the amplitude-collapse law."""
    w, w0 = TWOPI * f_drive, TWOPI * f0
    return 1.0 / np.sqrt((w0 ** 2 - w ** 2) ** 2 + (2 * zeta * w0 * w) ** 2)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))

    # (a) free decay at several damping ratios, integrated by the real solver
    dt, T = 5e-5, 40000                       # 2 s at 20 kHz
    t = np.arange(T) * dt
    for z, lab in [(0.02, "ζ=0.02"), (0.1, "ζ=0.1"), (0.35, "ζ=0.35"),
                   (1.0, "ζ=1.0 critical"), (2.5, "ζ=2.5 over")]:
        p = isolated(1, 8, z)                 # one unit at 8 Hz
        s0 = HORNState(x=jnp.ones((1,)), v=jnp.zeros((1,)))   # released from x=1
        _, xs = run_sequence(p, s0, jnp.zeros((T, 1)), dt)     # no drive: free decay
        ax[0].plot(t, np.asarray(xs)[:, 0], lw=1.3, label=lab)
    ax[0].set_title("(a) ζ sets the memory horizon\nsame unit at 8 Hz, released from x=1",
                    fontsize=10)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("x")
    ax[0].axhline(0, c="k", lw=.5); ax[0].legend(fontsize=7.5, frameon=False)

    # (b) resonance curves: which frequency a unit answers to, and how loudly
    fd = np.logspace(0, np.log10(200), 400)
    for f0, z, c in [(8, 0.05, "tab:blue"), (8, 0.3, "tab:cyan"),
                     (32, 0.05, "tab:orange"), (100, 0.05, "tab:red")]:
        ax[1].loglog(fd, steady_state_amplitude(fd, f0, z), c=c, lw=1.5,
                     label=f"f₀={f0} Hz, ζ={z}")
    ax[1].set_title("(b) ω sets which frequency a unit answers.\n"
                    "Fast units are quiet: gain ∝ 1/(2ζω²)", fontsize=10)
    ax[1].set_xlabel("drive frequency (Hz)"); ax[1].set_ylabel("steady-state amplitude")
    ax[1].legend(fontsize=7.5, frameon=False); ax[1].grid(alpha=.25, which="both")

    # (c) the same horizon in two units, which is the parameterisation conflict
    zet = np.logspace(-2.3, 0, 200)
    ax[2].loglog(zet, 1 / (TWOPI * zet), "k-", lw=2.0, label="in CYCLES = 1/(2πζ)")
    for f0, c in [(2, "tab:blue"), (8, "tab:green"), (32, "tab:orange"), (100, "tab:red")]:
        ax[2].loglog(zet, 1 / (zet * TWOPI * f0), c=c, lw=1.2, ls="--",
                     label=f"in SECONDS, f₀={f0} Hz")
    ax[2].set_title("(c) ζ sets memory in cycles, ω sets it in seconds.\n"
                    "ζ is also the amplitude knob: one parameter, two jobs", fontsize=10)
    ax[2].set_xlabel("ζ"); ax[2].set_ylabel("memory horizon")
    ax[2].legend(fontsize=7, frameon=False, loc="lower left")
    ax[2].grid(alpha=.25, which="both")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
