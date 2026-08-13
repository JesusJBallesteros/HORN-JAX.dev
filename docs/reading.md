# Reading list

The papers this repo is built against, annotated with what each one contributes here.
Grouped by the role they play rather than by date. All are open access.

## The architecture itself

**Effenberger, Carvalho, Dubinin & Singer (2025).** *The functional role of oscillatory
dynamics in neocortical circuits: a computational perspective.* PNAS 122.
[link](https://www.pnas.org/doi/10.1073/pnas.2412830122)

The HORN thesis: populations of damped harmonic oscillators outperform non-oscillating
recurrent architectures on learning speed, noise tolerance and parameter efficiency, and
heterogeneous frequencies, conduction delays and modularity each add performance without
adding parameters. The last of those claims is what E05 and E07 interrogate.

**Carvalho, Ulmann, Singer & Effenberger (2025).** *An analog-electronic implementation of
a harmonic oscillator recurrent neural network.* [arXiv:2509.04064](https://arxiv.org/abs/2509.04064)

The hardware result and, more usefully, an instructive failure: the dynamics transferred
faithfully, yet the digital readout agreed with the analog system on only 28.4% of trials,
and refitting a linear readout recovered performance. The information was in the dynamics;
the readout was the fragile part. E06 reproduces that pattern in simulation with precision
as the controlled variable.

**Reference implementation:** [`brainmass`](https://brainmass.readthedocs.io/reference/horn.html).
Not used here. This repo is written from the equations so that agreement with the physics,
rather than agreement with another implementation, is the correctness criterion.

## Oscillators as sequence models

**Rusch & Mishra (2021).** *Coupled Oscillatory Recurrent Neural Network (coRNN).* ICLR.
[pdf](https://arxiv.org/pdf/2010.00951)

Second-order oscillator dynamics discretised into a recurrent network, with proven bounds
on hidden-state gradients. The formal account of why oscillatory dynamics mitigate
exploding and vanishing gradients, which is the stiffness visible in the pixel-wise
training curves in E03.

**Rusch & Rus (2025).** *Oscillatory State-Space Models (LinOSS).* ICLR.
[pdf](https://arxiv.org/pdf/2410.03943)

Forced harmonic oscillators as a linear state-space model with parallel scans: stable
under a mild condition on the state matrix, universal, and effective on sequences of tens
of thousands of steps. The scaling path if backpropagation through time stops being
viable, which is the boundary noted at the end of E07.

## The biological motif

**Lisman & Jensen (2013).** *The theta-gamma neural code.* Neuron 77, 1002-1016.
[free full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/)

The canonical statement of nesting as a code: items represented in successive gamma
subcycles within one theta cycle, giving an ordered multiplexed representation. This is
the template E07 copies, and the reason the frequency ratio is treated as structure rather
than spread.

**Yan et al. (2025).** *Efficient and robust temporal processing with neural oscillations
modulated spiking neural networks (Rhythm-SNN).* Nature Communications.
[link](https://www.nature.com/articles/s41467-025-63771-x)

The nearest existing work: heterogeneous oscillatory signals periodically gate spiking
units, reducing firing rates, improving robustness, and easing gradient flow through
periodic shortcuts. The rhythms are imposed externally rather than arising from the
network's own state, which is the specific difference E07 and E08 would test.

## Spike-timing readouts

**Izhikevich (2001).** *Resonate-and-fire neurons.* Neural Networks 14, 883-894.
[pdf](https://www.izhikevich.org/publications/resfire.pdf)

The bridging primitive: a spiking unit whose subthreshold behaviour is a damped
oscillation, which is a HORN unit with a threshold. Sensitive to input timing and to
resonance, and no more expensive than integrate-and-fire.

**Higuchi, Bohte & Otte (2024).** *Balanced resonate-and-fire neurons.* ICML.
[pdf](https://arxiv.org/pdf/2402.14603)

Makes those units trainable at scale: better task performance with a fraction of the
events and parameters of conventional recurrent spiking networks, and stable optimisation
over hundreds of timesteps. The practical starting point for E08.

**Frady & Sommer (2019).** *Robust computation with rhythmic spike patterns.* PNAS 116.
[pdf](https://arxiv.org/pdf/1901.07718)

Phase-to-timing mapping: complex-valued attractor networks realised as spiking networks
whose fixed points are stable, perturbation-tolerant periodic spike patterns. The
theoretical basis for expecting a timing code to degrade gracefully, which E08 tests
against the measured continuous baseline in E06.
