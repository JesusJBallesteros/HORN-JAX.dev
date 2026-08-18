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

WHERE THE NONLINEARITY SITS
---------------------------
The drive is nonlinear, and there are two places to put the nonlinearity. Both
are standard, and they are not the same model:

    drive="output"   f = W_in u + W_rec tanh(x)              <- the default here
    drive="input"    f = eps * tanh(W_rec x + W_in u + b)    <- Effenberger et al.

"output" saturates what a unit EMITS, and the receiver sums those bounded signals
linearly. "input" saturates what a unit RECEIVES: everything arriving, stimulus
included, is summed first and squashed afterwards. In the rate-model literature
these are the voltage form and the activity form. They are not interchangeable,
and here even less so than usual, because the nonlinearity sits inside a
second-order operator with a different omega per unit.

The consequence that matters is what ONE UNCOUPLED unit is:

    "output"   a purely LINEAR filter. The stimulus reaches x through a plain
               matrix, so nothing bends it until another unit's output arrives.
               Every nonlinearity in the network is relational.
    "input"    already nonlinear. A static squash followed by a resonator, before
               any coupling exists at all.

"output" is the default because it makes W_rec = 0 a genuinely linear filter
bank, and that is the falsifying control the biphase experiment is built on
(docs/E05). Under "input" the control is unavailable: tanh(W_in u) already
contains the cubic cross-term the task asks for, so an uncoupled bank scores
0.795 where the linear one scores chance. "A filter bank cannot represent a
biphase" is therefore a statement about THIS model, not about the reference one,
which is a fact about the instrument and belongs in the open rather than in a
footnote. tests/test_dynamics.py::test_drive_placement_decides_linearity pins the
distinction as superposition, which is what it actually is.

"input" exists so that the comparison can be run.

One second-order consequence: under "input" the drive is capped at eps however 
large W_in grows, so the amplitude-collapse fix of E02, scaling W_in by 2*zeta*omega^2,
does nothing there. The factor has to move OUTSIDE the tanh, which is precisely 
what eps is. See model.init_net.

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

    # Used by drive="input" ONLY, and left None under drive="output". A None field
    # is an empty pytree node, so jax.grad and optax walk straight past it and the
    # output form is byte-for-byte what it was before these two were added.
    log_eps: jnp.ndarray = None    # (n_osc,) LOG of excitability: the gain OUTSIDE the
                                   #          tanh, the reference model's eps. Log-stored
                                   #          for the same reason omega and zeta are.
    bias: jnp.ndarray = None       # (n_osc,) offset INSIDE the tanh. The reference model
                                   #          carries b_hh and b_ih separately, but they
                                   #          enter one tanh additively, so only their sum
                                   #          is identifiable: one vector here, not two.

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


def init_params(key, in_size, n_osc, omega=1.0, zeta=0.1, w_scale=0.1,
                drive="output"):
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
        drive:   which placement the parameters are being built for. "input" needs
                 log_eps and bias, so they are allocated here; "output" leaves them
                 None. This is the low-level constructor and it works in rad/s, so
                 eps starts at 1 rather than at 2*zeta*omega^2 - the Hz-facing
                 model.init_net is where the resonance normalisation belongs.
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

        # None under "output": an absent pytree leaf, invisible to grad and optax.
        log_eps=jnp.zeros((n_osc,)) if drive == "input" else None,   # log(1) = 0
        bias=jnp.zeros((n_osc,)) if drive == "input" else None,
    )


def init_state(n_osc, batch=None):
    """Zero initial state: every oscillator at rest, at the origin.

    batch=None  -> shape (n_osc,)         single sequence
    batch=N     -> shape (N, n_osc)       N sequences processed in parallel
    """
    shape = (n_osc,) if batch is None else (batch, n_osc)  # pick shape based on batching
    return HORNState(x=jnp.zeros(shape), v=jnp.zeros(shape))


def step(params, state, u, dt=0.1, drive="output"):
    """Advance the oscillator population by one timestep.

    Args:
        params: HORNParams, the learned weights and oscillator constants.
        state:  HORNState, current (x, v).
        u:      external input this step, shape (in_size,) or (batch, in_size).
        dt:     integration timestep. Smaller = more accurate but more compute.
                The hard stability limit of symplectic Euler is dt*omega < 2:
                measured, dt*omega = 1.99 survives and 2.01 gives NaN, regardless
                of the learning rate. Stay well below it, because approaching the
                limit is not merely less accurate but systematically biased: the
                position amplitude is inflated by 1/sqrt(1 - (dt*omega/2)^2) and
                the resonance sits at arcsin(dt*omega/2)/(pi*dt) rather than at
                omega. At the ten-samples-per-period edge `model.usable_band`
                allows, that is +5.3% in amplitude and +1.5% in frequency. Both
                depend on omega, so in a heterogeneous bank they land unevenly
                across the population.
        drive:  where the nonlinearity sits: "output" (default) or "input". See
                WHERE THE NONLINEARITY SITS at the top of this file. It is a
                Python string rather than a traced value, so each setting compiles
                its own branch and the choice costs nothing per step.

    Returns:
        (new_state, x) - the new state, and the position as this step's output.
        Returning a 2-tuple in this order is exactly what jax.lax.scan expects.
    """
    omega = jnp.exp(params.log_omega)   # recover true frequency from its log (always > 0)
    zeta = jnp.exp(params.log_zeta)     # recover true damping from its log (always > 0)

    # --- the drive term f(x, u) -------------------------------------------------
    # Both placements combine the same two contributions:
    #   u @ W_in.T          external input projected onto the oscillators
    #   x @ W_rec.T         recurrent input from every other oscillator
    # and both put a tanh somewhere. They differ ONLY in where, which is the whole
    # point of the flag - see WHERE THE NONLINEARITY SITS at the top of this file.
    #
    # Why transpose? W_in is (n_osc, in_size) and u is (..., in_size). Writing
    # u @ W_in.T contracts over in_size and yields (..., n_osc). Crucially this
    # form works UNCHANGED whether u is (in_size,) or (batch, in_size), because
    # matmul broadcasts over leading axes. No separate batched code path needed.
    #
    # Why a tanh at all? It bounds the coupling. Purely linear coupling makes the
    # whole network a linear dynamical system - elegant, and exactly solvable, but
    # it cannot compute anything a linear filter cannot. The saturating
    # nonlinearity is where expressive power comes from.
    if drive == "output":
        # Saturation on what each unit SENDS. The external path stays linear, so
        # an uncoupled unit is exactly a linear filter and superposition holds.
        f = u @ params.W_in.T + jnp.tanh(state.x) @ params.W_rec.T
    elif drive == "input":
        # Saturation on what each unit RECEIVES. Stimulus and recurrent input are
        # summed FIRST and squashed together, so the two mix inside the
        # nonlinearity and not only through the network.
        if params.log_eps is None or params.bias is None:
            raise ValueError(
                "drive='input' needs log_eps and bias, and both are None on "
                "parameters built for the default output form. Build the network "
                "with model.init_net(drive='input') or core.init_params(drive='input').")
        # eps multiplies OUTSIDE the tanh, and it has to: |tanh| <= 1 caps the
        # drive, so a gain folded into W_in would saturate rather than scale. This
        # is the same 2*zeta*omega^2 factor that input_gain and rec_gain apply in
        # the output form, moved to the only place it can still act.
        f = jnp.exp(params.log_eps) * jnp.tanh(
            state.x @ params.W_rec.T + u @ params.W_in.T + params.bias)
    else:
        raise ValueError(f"unknown drive: {drive!r} (expected 'output' or 'input')")

    # --- the equation of motion, rearranged for acceleration --------------------
    # From  x'' + 2*zeta*omega*x' + omega^2 * x = f   solve for x'':
    #       x'' = f - 2*zeta*omega*x' - omega^2 * x
    #             ^         ^                ^
    #             |         |                +-- restoring force: pulls x back to 0.
    #             |         |                    This is the spring; it makes it oscillate.
    #             |         +-- damping/friction: opposes velocity, bleeds energy out.
    #             +-- the drive f built above, under whichever placement was asked for
    accel = f - 2.0 * zeta * omega * state.v - (omega ** 2) * state.x

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


def run_sequence(params, state, inputs, dt=0.1, drive="output"):
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
        return step(params, carry, u, dt, drive)

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
