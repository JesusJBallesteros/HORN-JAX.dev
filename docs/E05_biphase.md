# E05: Biphase, a task where power pooling must lose

**Question.** E02 left the repo's headline claim unexercised: everything trained so far is
reachable by a filter bank plus a power readout. Is there a task where phase carries the
label, so `rms` pooling provably loses and recurrent phase coupling provably wins?

**The task.** Every stimulus is the same two tones at f and 2f with fixed amplitudes:

```
s(t) = sin(2πft + p) + sin(2π·2f·t + 2p + ψ)
```

`p` is a random global phase; the class is `ψ`, the biphase. By construction the power
spectrum is identical across classes (`test_power_spectrum_is_matched_across_classes`),
so any power readout is at chance *by construction*, and a global time shift leaves `ψ`
invariant, so a last-state readout is at chance too. Extracting `ψ` requires a product of
units tuned near f and 2f: a nonlinearity, and heterogeneity, as preconditions.

**Probe result (at initialisation, before any training).** Ridge readout on pooled features
of an untrained network, 3 classes, chance 0.333 (`experiments/probe_mechanism.py`):

| rms\|x\| | W_rec | mean | rms | last | reading |
|---|---|---|---|---|---|
| 0.014 | free/off | 0.318 | 0.365 | 0.337 | linear regime: nothing |
| 0.139 | free | 1.000 | 0.365 | 0.930 | recurrence + mild nonlinearity: solved |
| 0.139 | zero | 0.317 | 0.365 | 0.337 | filter bank: nothing |
| 1.389 | free | 1.000 | 0.648 | 1.000 | strongly driven: even rms climbs |
| 1.389 | zero | 0.317 | 0.365 | 0.337 | filter bank: still nothing |

Two of three original predictions were wrong, in an informative way. The falsifying control
is the **architecture** (`W_rec = 0` is at chance at every amplitude), not the pooling. Once
the tanh is engaged the network converts biphase into internal power, so `rms` climbs to
0.65: the stimulus spectrum is still matched, but the network's internal spectrum is not.

**Status.** Probe done. The trained heterogeneous-vs-homogeneous grid
(`experiments/run_diversity.py`) has not been run to completion; it is the next experiment,
and its output (`results/diversity_*.json/png`) is not yet in the repo.

**Reproduce.**

```bash
python experiments/probe_mechanism.py
python experiments/run_diversity.py --quick   # smoke test, ~2 min
python experiments/run_diversity.py           # the real grid
```
