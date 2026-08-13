# E06: Readout precision, the analog failure mode in simulation

**Question.** Carvalho et al. (arXiv:2509.04064) transferred a fitted HORN onto analog
hardware. The dynamics matched their digital twin, but the digital readout agreed with it
on only 28.4% of trials, and refitting a linear readout on the analog activity restored
performance. Can that be reproduced in simulation with precision as the manipulated
variable, and does the outcome depend on how the task encodes its label?

**Method.** Fit the network and its linear decoder at full precision, then re-run
inference with the state rounded to n bits, sweeping n. Two regimes:

- **in-loop**, the state (x, v) rounded at every timestep, as an analog substrate would
  evolve it;
- **observation only**, full-precision dynamics rounded at readout, the equivalent of a
  coarse ADC on an otherwise clean recording.

Reported per bit depth: accuracy of the decoder fitted at full precision, accuracy of a
ridge decoder refitted on the degraded activity, and agreement with the full-precision
model's trial-by-trial predictions, which is the metric the paper uses. Two tasks, chosen
because they encode the label differently: the biphase (phase-coded by construction, E05)
and row-wise sMNIST (the paper's task, amplitude-coded in practice).

## Biphase (full precision 1.000, chance 0.333)

| bits | in-loop, original | in-loop, refitted | agreement | observation only |
|---|---|---|---|---|
| 1 | 0.258 | 0.820 | 0.258 | 0.836 |
| 3 | 0.258 | 1.000 | 0.258 | 0.842 |
| 6 | 0.258 | 0.990 | 0.258 | 0.889 |
| 10 | 0.278 | 1.000 | 0.278 | 1.000 |
| 12 | 0.680 | 1.000 | 0.680 | 1.000 |
| 14 | 0.982 | 1.000 | 0.982 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 | 1.000 |

![readout precision, biphase](../results/readout_precision_biphase.png)

## Row-wise sMNIST (full precision 0.861, chance 0.100)

| bits | in-loop, original | in-loop, refitted | agreement | obs. only | obs. refitted |
|---|---|---|---|---|---|
| 1 | 0.117 | 0.142 | 0.128 | 0.260 | 0.826 |
| 2 | 0.191 | 0.450 | 0.200 | 0.295 | 0.848 |
| 3 | 0.308 | 0.696 | 0.303 | 0.546 | 0.876 |
| 4 | 0.362 | 0.772 | 0.358 | 0.796 | 0.884 |
| 5 | 0.682 | 0.846 | 0.716 | 0.854 | 0.888 |
| 6 | 0.845 | 0.872 | 0.929 | 0.857 | 0.878 |
| 8 | 0.863 | 0.856 | 0.989 | 0.861 | 0.880 |
| 16 | 0.861 | 0.884 | 0.998 | 0.861 | 0.880 |

![readout precision, sMNIST](../results/readout_precision_smnist.png)

## Reading

**The published pattern reproduces on the published task.** On sMNIST the decoder fitted
at full precision collapses under in-loop rounding while a refitted ridge decoder
recovers most of the performance, 0.308 to 0.696 at 3 bits. Refitting the readout
recovers what the substrate preserved. The paper's 28.4% was a single hardware operating
point rather than a bit depth, so what is comparable is the shape of the failure, not the
value; this curve happens to pass through that agreement level near 3 bits.

**The two tasks fail in opposite ways.** The expectation going in, and the premise of the
planned spiking-readout work (E08), was that phase coding might degrade more gracefully.
For the decoder mapping it does the reverse: sMNIST is back to full-precision accuracy by
6 bits, the biphase needs 14. Lightly damped, strongly coupled dynamics are what the
biphase requires, and per-step rounding acts on them as state noise, so trajectories
decorrelate from the full-precision run and the specific decoder directions fitted at
full precision stop pointing at the label. Class statistics survive, which is why the
refitted decoder reaches 1.000 at 3 bits.

**The information behaves the other way round.** On the biphase it survives intact to 3
bits, refitted decoder 1.000. On sMNIST it erodes, refitted decoder 0.696 against 0.884
at full precision, because fine amplitude gradations carry digit identity and rounding
removes them. Phase coding protects the information and endangers the mapping; amplitude
coding does the opposite. Any robustness claim for a spike-timing readout now has a
measured baseline rather than an assumed one.

**Rounding the observation is nearly harmless.** With a refitted decoder, 1-bit
observation gives 0.826 on sMNIST and 0.836 on the biphase, because summarising over the
sequence averages out observation noise much as trial averaging does in a recording. The
loss of precision matters where the dynamics evolve, not where they are measured. That is
an argument for designing the readout together with the substrate rather than for
digitising more finely.

**Caveats.** One seed per task and one population size each, so the contrast between
tasks is a pattern to test rather than an established result. The sMNIST full-precision
value here (0.861, 2000-trial evaluation) sits below E03's 0.897 because the evaluation
set and run differ. Provenance headers in `results/readout_precision_*.txt`.

**Reproduce.**

```bash
python experiments/readout_precision.py                 # biphase, CPU, ~3 min
python experiments/readout_precision.py --task smnist   # downloads MNIST once
```
