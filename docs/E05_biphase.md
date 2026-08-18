# E05: Biphase, a task on which a power readout must fail

**Question.** Everything trained up to E02 is reachable with a bank of bandpass filters
and a power readout, so the claim that these networks use phase as well as amplitude was
untested. This needs a task where the label lives in phase alone.

**The stimulus.** Two tones at f and 2f, fixed amplitudes, on every trial:

```
s(t) = sin(2πft + p) + sin(2π·2f·t + 2p + ψ)
```

`p` is a random global phase and the class is `ψ`, the biphase. The power spectrum is
therefore identical across classes by construction
(`test_power_spectrum_is_matched_across_classes`), so a power readout is at chance by
design, and a global time shift leaves `ψ` unchanged, so reading the final state fails
too. Recovering `ψ` requires a product between a component near f and one near 2f, so
some nonlinearity is a precondition rather than an option. *Where* that nonlinearity sits
decides whether frequency heterogeneity is a precondition too, and that turns out to be a
property of the model and not of the task; see
[Which model the control belongs to](#which-model-the-control-belongs-to). The quantity is
the biphase of bispectral analysis, familiar from cross-frequency coupling measures in
EEG.

**Separability at initialisation, before any fitting.** Ridge decoder on the pooled
activity of an untrained population, 3 classes, chance 0.333
(`experiments/probe_mechanism.py`):

| rms\|x\| | W_rec | time-mean | rms | final | reading |
|---|---|---|---|---|---|
| 0.028 | free/off | 0.323 | 0.365 | 0.320 | linear regime, nothing |
| 0.279 | free | 1.000 | 0.372 | 0.687 | recurrence with mild nonlinearity, solved |
| 0.279 | zero | 0.323 | 0.365 | 0.320 | filter bank, nothing |
| 2.789 | free | 1.000 | 0.652 | 1.000 | strongly driven, rms climbs |
| 2.789 | zero | 0.323 | 0.365 | 0.320 | filter bank, still nothing |

Two of the three predictions written before the run were wrong. The control that
falsifies is the architecture, `W_rec = 0` at chance at every drive amplitude, and not
the choice of statistic. Once the tanh is engaged the population converts the biphase
into internal response power, so the rms decoder climbs to 0.65: the stimulus spectrum
remains matched across classes, but the population's own spectrum does not.

That result is real and it is also narrower than it reads. It holds for the model in this
repository, in which the nonlinearity sits on the output path; it does not hold for the
one in the reference paper. The section below is that qualification, measured rather than
conceded.

*Provenance.* The table was regenerated after two corrections: a normalisation fault in
the ridge estimator, where a scaled bias column distorted the normal equations and
deflated some values, and a drift between the quoted rms|x| and what the current task
code produces. `results/probe_mechanism.txt` is the committed record. The conclusions did
not change.

## The gain asymmetry that made recurrence inert

The first fitted grid returned a signature that should have been impossible: `W_rec` free
and `W_rec` removed gave bit-identical accuracies, seed by seed
(`results/diversity_biphase_n24_L200.txt`, kept as the record of the failure). Gain
normalisation multiplies `W_in` by `2ζω²` so that an afferent drive produces an O(1)
response, but nothing multiplied `W_rec`, and recurrent input is also a drive. Measured,
the afferent drive was 6968 times the recurrent one, and removing `W_rec` shifted the
decoder's class scores by a relative 4e-4. The population was feedforward in all but
name, and the manipulated variable was inert ([E00](E00_measure_the_lever.md)).

`init_net(rec_gain=...)` now selects between "flat", which reproduces the earlier
behaviour, and "normalised", which applies the same `2ζω²` factor to `W_rec`. Drive ratio
6968:1 to 8:1; leverage of `W_rec` on the decoder output 3.6e-3 to 0.37. Held by
`test_drive_balance_and_recurrence_leverage`.

**Separability under both conventions**
(`results/probe_mechanism_rec_flat_vs_norm.txt`): the control holds either way, with
`W_rec = 0` at chance at every amplitude. Under the normalised convention the recurrent
path converts biphase into internal power far more readily, rms and final-state decoders
reaching 1.000 from w_scale 1.0 upward, where the flat convention needed w_scale 10 for
rms to reach 0.65. The initialisation result is unchanged and sharper.

**Fitted grid under both conventions**
(`results/diversity_biphase_n24_L200_recflat.*` and `..._recnormalised.*`): under the flat
convention the two recurrence conditions remain bit-identical, now for an understood
reason. Under the normalised convention the manipulation moves the output, but all
conditions sit at chance within noise. This is not a null result on frequency diversity.
The runs are pilot scale, 24 oscillators, 60 fitting steps, 2 seeds, and the fitted linear
decoder reads raw pooled activity while the ridge decoder that reaches 1.000 at
initialisation standardises it first. The distance between those two is decoder
conditioning and fitting budget, not representation.

## Which model the control belongs to

The drive is nonlinear, and there are two places to put the nonlinearity. This repository
puts it on the output path; Effenberger et al. put it on the input path:

```
drive="output"   f = W_in u + W_rec tanh(x)              this repo
drive="input"    f = eps * tanh(W_rec x + W_in u + b)    PNAS 2025, Materials and Methods
```

The difference is not cosmetic, and it is not about how strongly the tanh is engaged. It
decides what a single uncoupled unit *is*. Under `"output"` the stimulus reaches the
oscillator through a plain matrix, so with `W_rec = 0` the network is exactly a linear
filter bank and superposition holds to 1e-5
(`test_dynamics.py::test_drive_placement_decides_linearity`). Under `"input"` the stimulus
is squashed before it arrives, so an uncoupled unit is already a static nonlinearity
followed by a resonator, and `tanh(W_in u)` contains the cubic cross-term that produces a
biphase-dependent DC term all by itself.

**The same probe, same task, same seeds, under the reference placement**
(`results/probe_mechanism_input_rec_normalised.txt`), chance 0.333:

| w_scale | rms\|x\| | W_rec | time-mean | rms | final |
|---|---|---|---|---|---|
| 0.1 | 0.027 | free | 0.950 | 0.365 | 0.723 |
| 0.1 | 0.027 | zero | **0.757** | 0.367 | 0.720 |
| 1.0 | 0.126 | free | 1.000 | 1.000 | 1.000 |
| 1.0 | 0.126 | zero | **0.790** | 0.688 | 0.792 |
| 3.0 | 0.151 | free | 1.000 | 1.000 | 1.000 |
| 3.0 | 0.151 | zero | **0.795** | 0.678 | 0.795 |
| 10.0 | 0.162 | free | 1.000 | 1.000 | 1.000 |
| 10.0 | 0.162 | zero | **0.772** | 0.677 | 0.788 |

The falsifying control fails. `W_rec = 0` sits at 0.76 to 0.80 at every amplitude where
the same rows sit at 0.323 under `"output"`. The sharpest row is the first: at
`w_scale = 0.1` the state is 0.027 and the `nonlin` diagnostic reads 2.2e-03, meaning the
tanh *on the state* is doing nothing at all, and the uncoupled bank still reaches 0.757,
because the nonlinearity that matters is on the input and never touches the state.
`nonlin` is the right diagnostic for one placement and the wrong one for the other, which
is worth remembering before reading that column again.

**What this does and does not change.** Every number in the table above the fold stands.
The mechanism claim stands: recovering a biphase needs a product between components at f
and 2f, and something has to form it. What was overstated is the scope. *A bank of
independent linear resonators cannot represent a biphase* is the claim the measurement
supports, and it is a claim about linear filter banks. *A HORN with `W_rec = 0` cannot
represent a biphase* is what the doc previously implied, and in the reference model it is
false. Frequency heterogeneity is a precondition under `"output"` and a convenience under
`"input"`, which also means the diversity grid is asking a slightly different question in
each.

**Why the flag rather than a switch of models.** `"output"` stays the default, and not
only for continuity: it is the better instrument for the question. Because `W_rec = 0`
under it is genuinely linear, it supplies a clean null that isolates what the *population*
contributes over and above pointwise nonlinearity, and the reference form has no
equivalent. That is a defensible reason to have departed, and it is now written down in
`horn/core.py` where the departure lives, rather than being inferable only by diffing
against the paper. `drive="input"` exists so the comparison is run rather than argued.

One consequence to know before switching: under `"input"` the drive is capped at `eps`
however large `W_in` grows, so the amplitude-collapse fix of [E02](E02_gain_and_pooling.md)
(scaling `W_in` by `2ζω²`) cannot work there. The factor has to move outside the tanh,
which is exactly what `eps` is. Noticing that is also the answer to a loose end: setting
`rec_gain="normalised"` in the output form puts the same `2ζω²` on both paths, i.e. one
per-unit gain on the whole drive. The correction found by measurement in the section above
and the excitability parameter named in the paper are the same quantity, factored
differently. It was never missing.

*Artefacts.* Runs are now named by placement, `probe_mechanism_<drive>_rec_<gains>.*`. The
records written before the flag existed keep their old names
(`probe_mechanism_rec_flat_vs_norm.*`); `probe_mechanism_output_rec_flat_vs_norm.*` is the
regeneration under the current code and is identical to them, which is the regression
check that the flag changed nothing about the default path.

**Status.** The initialisation result carries the claim under both conventions. The
fitted grid does not yet reproduce it. What remains is a properly sized run with more
oscillators, more steps, more seeds, and standardised activity at the decoder or an
equivalent normalisation layer. The pilot numbers are not evidence in either direction.

**Reproduce.**

```bash
python experiments/probe_mechanism.py                              # both gain conventions
python experiments/probe_mechanism.py --drive input --rec-gain normalised   # reference form
python experiments/run_diversity.py --quick                        # pilot, ~2 min
python experiments/run_diversity.py                                # full grid, GPU recommended
```
