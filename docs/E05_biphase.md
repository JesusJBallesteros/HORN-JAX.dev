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
too. Recovering `ψ` requires a product between units responding near f and near 2f, which
makes a nonlinearity and frequency heterogeneity preconditions rather than options. The
quantity is the biphase of bispectral analysis, familiar from cross-frequency coupling
measures in EEG.

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

**Status.** The initialisation result carries the claim under both conventions. The
fitted grid does not yet reproduce it. What remains is a properly sized run with more
oscillators, more steps, more seeds, and standardised activity at the decoder or an
equivalent normalisation layer. The pilot numbers are not evidence in either direction.

**Reproduce.**

```bash
python experiments/probe_mechanism.py         # both conventions, writes the comparison record
python experiments/run_diversity.py --quick   # pilot, ~2 min
python experiments/run_diversity.py           # full grid, GPU recommended
```
