# E08: Spiking readout under precision loss

Designed, not started. E06 turned this from a hunch into a question with a measured
baseline attached, and also inverted the hypothesis it started from.

**Question.** Every readout in this repo is continuous and, as E06 shows,
precision-hungry in the regime that matters. Does a spike-timing code read the same
dynamics more robustly, or does it only move the fragility somewhere else?

## What E06 changed about the premise

The original expectation was that phase coding would degrade more gracefully than an
amplitude code. Measured, the opposite holds for the decoder mapping: the phase-coded
biphase needs about 14 bits of state precision before the fitted decoder recovers, while
sMNIST needs about 6. The information behaves the other way round, surviving to 3 bits on
the biphase and eroding on sMNIST.

So the claim this experiment has to defend is narrower and more interesting than the one
it started with: not "phase is robust", but that a readout which reads *time of event*
rather than *value of state* should be insensitive to precisely the degradation that
breaks the affine decoder, because rounding the state perturbs amplitudes continuously
while leaving zero crossings and their ordering comparatively intact.

## The design

**Substrate.** A resonate-and-fire unit is a HORN unit with a threshold, so the dynamics
module barely changes: add a threshold and a reset to the existing state update. The
balanced formulation of these units is reported to train stably through backpropagation
over hundreds of timesteps, with far fewer events and parameters than conventional
recurrent spiking networks, which is the practical starting point.

**Readout.** Classify from the phase of fast-band events relative to a slow reference,
rather than from pooled amplitude. In a nested network (E07) the slow band supplies that
reference natively, which is the sense in which E07 and E08 compose: the nesting is not
decoration, it is what makes the spike phase readable.

**Protocol.** Exactly E06's, so the curves are directly comparable: sweep in-loop state
precision, report accuracy of the readout fitted at full precision, accuracy of a readout
refitted at each depth, and agreement with the full-precision model.

## Predictions, recorded before running

1. The spike-phase readout's accuracy curve is flatter than the continuous one across bit
   depth, with the gap widest on the phase-coded task where the continuous decoder is
   worst.
2. Its collapse threshold sits well below the continuous readout's 14 bits on the biphase.
3. It loses absolute accuracy at full precision, because thresholding discards
   information. The claim is about the slope, not the intercept.

## What would falsify it

A spike-phase readout that collapses at the same bit depth as the continuous one, which
would mean the fragility is in the dynamics decorrelating rather than in the readout's
dependence on precise values, and no change of readout can fix it. E06's separation
between information survival and mapping survival is what makes this outcome possible
rather than merely conceivable.

## Why it matters beyond the benchmark

A spiking readout is effectively a one-bit interface, which addresses directly the
precision mismatch that broke the digital readout on the analog hardware in
arXiv:2509.04064. If the slope claim holds, the recommendation is to co-design the
readout with the substrate rather than to digitise more finely.

**Status.** Not started. E06 (`experiments/readout_precision.py`) is the baseline it has
to beat, and its protocol is the one to reuse.
