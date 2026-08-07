"""Harmonic Oscillator Recurrent Network (HORN) - minimal JAX implementation.

THE PHYSICS
-----------
Every unit in this network is a damped, driven harmonic oscillator obeying

    x'' + 2*zeta*omega*x' + omega**2 * x  =  f(x, u)

    x      position          - this is the unit's "activation", what the readout sees
    x'     velocity          - the second half of the state; a normal RNN has no equivalent
    omega  natural frequency - which input frequencies this unit resonates with
    zeta   damping ratio     - how fast the ringing dies away = the unit's memory horizon
    f      drive             - recurrent input from other units + external input

Contrast with a standard RNN: a tanh/LSTM unit holds one scalar and needs
learned gates to retain information. An oscillator holds TWO numbers (position
and velocity) and retains information for free, because an underdamped
oscillator rings. Temporal structure is intrinsic to the substrate rather than
bolted on with gating machinery.

NOTE ON JAX
-----------
JAX is functional: arrays are immutable and functions must have no side effects.
So there is no `self.state` being mutated in place. Instead every function takes
state in and returns new state out. This is what allows jax.jit, jax.grad and
jax.vmap to work, and it is the main adjustment coming from MATLAB or NumPy.
"""

from typing import NamedTuple   # NamedTuple = immutable struct; JAX treats it as a "pytree",
                                # meaning jax.grad can differentiate w.r.t. all its fields at once
import jax                      # core JAX: grad, jit, random, lax.scan
import jax.numpy as jnp         # NumPy-compatible API that runs on GPU and is differentiable
                                # (use jnp, never np, for anything JAX must trace through)


class HORNParams(NamedTuple):
    """The learnable parameters of one HORN layer.

    Everything here is updated by gradient descent. Grouping them in a
    NamedTuple means jax.grad(loss)(params) returns gradients in exactly the
    same shape, so an optimiser can walk both structures in lockstep.
    """
    W_in: jnp.ndarray       # (n_osc, in_size)  how external input drives each oscillator
    W_rec: jnp.ndarray      # (n_osc, n_osc)    how oscillators drive each other = the coupling
    log_omega: jnp.ndarray  # (n_osc,)          LOG of natural frequency, one per oscillator
    log_zeta: jnp.ndarray   # (n_osc,)          LOG of damping ratio, one per oscillator

    # Why store the LOG of omega and zeta rather than the values themselves?
    # Gradient descent is unconstrained: it will happily push a parameter negative.
    # A negative zeta means NEGATIVE DAMPING, i.e. energy pumped into the oscillator
    # every step, i.e. exponential blow-up and NaNs within a few hundred steps.
    # Storing the log and exponentiating on use makes negative values unreachable:
    # exp() of any real number is strictly positive. This is a standard trick and
    # it removes an entire class of training failure.


class HORNState(NamedTuple):
    """The dynamic state of the oscillator population at one instant in time.

    This is NOT learned - it is recomputed every forward pass, and reset
    between sequences.
    """
    x: jnp.ndarray  # position of each oscillator
    v: jnp.ndarray  # velocity of each oscillator (= dx/dt)


def init_params(key, in_size, n_osc, omega=1.0, zeta=0.1, w_scale=0.1):
    """Create and randomly initialise the parameters of one HORN layer.

    Args:
        key:     a JAX PRNG key. JAX has no global random seed - randomness is
                 explicit and threaded through by hand, so results are reproducible.
        in_size: dimensionality of the external input at each timestep.
        n_osc:   how many oscillators in this layer.
        omega:   natural frequency. A float gives every unit the same frequency
                 (homogeneous); pass an array of shape (n_osc,) for a heterogeneous
                 population - which is the starting point for the nested-frequency
                 experiment.
        zeta:    damping ratio. <1 underdamped (rings), =1 critical, >1 overdamped.
        w_scale: standard deviation for weight initialisation.
    """
    k_in, k_rec = jax.random.split(key)   # split one key into two independent keys.
                                          # Reusing the same key twice would give
                                          # W_in and W_rec identical random values.

    # Promote omega/zeta to full (n_osc,) arrays so downstream code never has to
    # branch on "is this a scalar or a vector". broadcast_to does this without
    # actually copying memory.
    omega = jnp.broadcast_to(jnp.asarray(omega, jnp.float32), (n_osc,))
    zeta = jnp.broadcast_to(jnp.asarray(zeta, jnp.float32), (n_osc,))

    return HORNParams(
        # Input weights: plain Gaussian, scaled small so the network starts gently driven.
        W_in=jax.random.normal(k_in, (n_osc, in_size)) * w_scale,

        # Recurrent weights: additionally divided by sqrt(n_osc). Each oscillator sums
        # n_osc inputs, and the variance of a sum of n independent terms grows like n,
        # so dividing by sqrt(n) keeps the total drive roughly constant as the layer
        # grows. Without this, big layers explode at initialisation.
        W_rec=jax.random.normal(k_rec, (n_osc, n_osc)) * (w_scale / jnp.sqrt(n_osc)),

        log_omega=jnp.log(omega),   # store logs, per the reasoning in HORNParams above
        log_zeta=jnp.log(zeta),
    )


def init_state(n_osc, batch=None):
    """Zero initial state: every oscillator at rest, at the origin.

    batch=None  -> shape (n_osc,)         single sequence
    batch=N     -> shape (N, n_osc)       N sequences processed in parallel
    """
    shape = (n_osc,) if batch is None else (batch, n_osc)  # pick shape based on batching
    return HORNState(x=jnp.zeros(shape), v=jnp.zeros(shape))


def step(params, state, u, dt=0.1):
    """Advance the oscillator population by one timestep.

    Args:
        params: HORNParams, the learned weights and oscillator constants.
        state:  HORNState, current (x, v).
        u:      external input this step, shape (in_size,) or (batch, in_size).
        dt:     integration timestep. Smaller = more accurate but more compute.
                Rule of thumb: dt must be well below the period of your FASTEST
                oscillator, i.e. dt << 2*pi/omega_max. If dt is too large the
                integration goes unstable and you get NaNs regardless of the learning rate.

    Returns:
        (new_state, x) - the new state, and the position as this step's output.
        Returning a 2-tuple in this order is exactly what jax.lax.scan expects.
    """
    omega = jnp.exp(params.log_omega)   # recover true frequency from its log (always > 0)
    zeta = jnp.exp(params.log_zeta)     # recover true damping from its log (always > 0)

    # --- the drive term f(x, u) -------------------------------------------------
    # Two contributions summed:
    #   u @ W_in.T          external input projected onto the oscillators
    #   tanh(x) @ W_rec.T   recurrent input from every other oscillator
    #
    # Why transpose? W_in is (n_osc, in_size) and u is (..., in_size). Writing
    # u @ W_in.T contracts over in_size and yields (..., n_osc). Crucially this
    # form works UNCHANGED whether u is (in_size,) or (batch, in_size), because
    # matmul broadcasts over leading axes. No separate batched code path needed.
    #
    # Why tanh on the recurrent path? It bounds the coupling. Purely linear
    # coupling makes the whole network a linear dynamical system - elegant, and
    # exactly solvable, but it cannot compute anything a linear filter cannot.
    # The saturating nonlinearity is where expressive power comes from.
    drive = u @ params.W_in.T + jnp.tanh(state.x) @ params.W_rec.T

    # --- the equation of motion, rearranged for acceleration --------------------
    # From  x'' + 2*zeta*omega*x' + omega^2 * x = f   solve for x'':
    #       x'' = f - 2*zeta*omega*x' - omega^2 * x
    #             ^         ^                ^
    #             |         |                +-- restoring force: pulls x back to 0.
    #             |         |                    This is the spring; it makes it oscillate.
    #             |         +-- damping/friction: opposes velocity, bleeds energy out.
    #             +-- external + recurrent drive
    accel = drive - 2.0 * zeta * omega * state.v - (omega ** 2) * state.x

    # --- SEMI-IMPLICIT (SYMPLECTIC) EULER INTEGRATION ---------------------------
    # The order of these two lines is the single most important detail in this file.
    v = state.v + dt * accel   # 1) update velocity using the acceleration
    x = state.x + dt * v       # 2) update position using the *NEW* velocity, not the old one
    #
    # Explicit Euler would use the OLD velocity on line 2 (x + dt*state.v). That
    # version systematically ADDS energy to an undamped oscillator: amplitude grows
    # without bound even with zero drive, which is physically nonsense and quietly
    # corrupts long sequences. Using the new velocity makes the scheme symplectic,
    # so energy stays bounded over arbitrarily many steps.
    # tests/test_dynamics.py::test_energy_conserved_when_undamped catches this if
    # the two lines are ever swapped.

    return HORNState(x=x, v=v), x


def run_sequence(params, state, inputs, dt=0.1):
    """Run the network over a whole input sequence.

    Args:
        inputs: (T, in_size) or (T, batch, in_size). TIME MUST BE THE FIRST AXIS -
                that is the axis lax.scan iterates over.

    Returns:
        (final_state, xs) where xs has shape (T, ...) - the position at every timestep.
    """
    def body(carry, u):
        # `carry` is the state threaded from one step to the next;
        # `u` is one slice of `inputs` along the leading (time) axis.
        return step(params, carry, u, dt)

    # jax.lax.scan is a compiled loop. A Python `for` loop would also work and give
    # identical numbers, but it would unroll into T copies of the graph - so a
    # 1000-step sequence compiles for minutes and eats memory. scan compiles the
    # body ONCE and reuses it, and it knows how to reverse-mode differentiate
    # through the recurrence, which is what makes backprop-through-time tractable.
    return jax.lax.scan(body, state, inputs)


def energy(params, state):
    """Mechanical energy of each oscillator: E = 0.5*(v^2 + omega^2 * x^2).

    Kinetic (0.5*v^2) plus potential (0.5*omega^2*x^2), exactly as for a mass on
    a spring. Not used in training - it exists as a correctness probe. With zero
    damping and zero drive this quantity must stay constant; if it drifts, the
    integrator is wrong. This is a far sharper test than eyeballing a trajectory.
    """
    omega = jnp.exp(params.log_omega)                        # true frequency from its log
    return 0.5 * (state.v ** 2 + (omega ** 2) * state.x ** 2)  # KE + PE, per oscillator
