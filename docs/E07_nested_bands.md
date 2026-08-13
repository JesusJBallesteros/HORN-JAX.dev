# E07: Nested bands versus flat heterogeneity

Designed, not started. Written down because the design is the part worth having on
record; a half-built version would say less.

**Question.** Published HORNs draw natural frequencies from a spread, which is
heterogeneity without structure. Cortex does something narrower: frequencies stand in
roughly fixed ratios, and the phase of the slow rhythm modulates the amplitude of the
fast one. Does that structure earn its keep at matched parameter count, and if so, is it
the banding or the cross-frequency modulation that does the work?

## Why this design

Two properties of the biological motif are absent from an additive coupling matrix with
unstructured frequencies:

- **Fixed ratios rather than a spread.** Theta-gamma nesting packs a small, roughly
  integer number of fast cycles into one slow cycle, and that near-integer relation is
  what allows an ordered sequence of slots rather than a mixture.
- **Multiplicative, asymmetric coupling.** Slow phase gates fast gain, not the reverse.
  A summed coupling term does not express gain modulation natively.

E05 supplies a mechanistic reason to expect the second to matter. Recovering a
cross-frequency phase relation requires a product between units near f and near 2f. With
additive coupling that product only appears once the tanh is driven hard, which was
measured: at low amplitude the population is at chance whatever the coupling. A
multiplicative pathway supplies the product directly, at any amplitude.

## Constraints already measured

The repo's own findings fix most of the free parameters before anything is built:

| constraint | consequence for this design |
|---|---|
| usable frequency ratio ≈ L/10 (E03) | a 1:6 nesting needs L of a few hundred; row-wise MNIST (ratio 2.8) cannot host it, pixel-wise MNIST (78) and the biphase family can |
| steady-state response ∝ 1/(2ζω²) (E02) | each band needs its own input-gain normalisation, or the fast band is silent at the decoder before fitting starts |
| dt bounded by the fastest unit (findings) | the fast band sets the timestep, and therefore the cost |
| recurrent drive needs the same normalisation (E00, E05) | any gating pathway must be checked for leverage before it is swept |

## Conditions, at matched parameter count

| condition | frequencies | coupling |
|---|---|---|
| A | homogeneous | additive only |
| B | unstructured heterogeneous | additive only |
| C | banded at a fixed ratio | additive only |
| D | banded at a fixed ratio | additive plus slow-phase gating of fast-band gain |

C against B isolates structured frequency organisation. D against C isolates
cross-frequency modulation specifically. That two-step decomposition is what makes a
positive result interpretable rather than a horse race.

Parameter matching is the delicate part: the gating pathway adds one scalar per band
pair, so the bookkeeping has to state explicitly whether those scalars are counted
against the control, and the answer should be fixed before the runs rather than after.

## Predictions, recorded before running

1. D beats C beats B beats A on a task with hierarchical temporal structure, and the
   margin grows with the depth of that structure.
2. The advantage of D over C survives at low drive amplitude, where the additive
   conditions lose their product term.
3. On a task without nested temporal structure, C and D show no advantage over B. If
   they do, the effect is not what it claims to be.

## What would falsify it

D indistinguishable from C at matched parameters on a task built to need nesting. That
would say the coupling matrix discovers whatever structure it needs, and the prior is
not worth building in at this scale.

## Prerequisite

The E05 grid has to be settled first, at proper scale and with a conditioned decoder.
Adding conditions to a comparison that does not yet separate its existing two would
compound an unresolved problem rather than answer a new question.

**Status.** Not started. No code beyond the band helpers already in `horn/tasks.py`
(`homogeneous_bands`) and `horn/model.py` (`log_spaced_bands`).
