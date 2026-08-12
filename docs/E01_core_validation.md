# E01: Core validation against closed-form physics

**Question.** Is the integrator correct, not merely running: does an uncoupled, unforced
unit reproduce the textbook damped harmonic oscillator?

**Method.** An isolated HORN unit has an exact solution, so the tests compare trajectories
against ground truth rather than against the code's own output: free decay, energy
conservation at ζ→0 (the test that separates semi-implicit from explicit Euler), overdamped
decay without zero crossing, spectral independence of heterogeneous units, and end-to-end
gradient flow. Notebook 01 adds impulse response, steady-state amplitude and phase, and dt
convergence against the analytic curves.

**Result.** All pass. See `results/pytest.txt` for the committed run.

![damping regimes and filter bank](../results/demo.png)

Left: ζ sets the memory horizon. Underdamped rings, overdamped crawls. Right: units with
different ω keep their own frequency (uncoupled), which is the property every
nested-frequency idea downstream depends on.

**Reproduce.**

```bash
pytest tests/test_dynamics.py
python demo.py                  # writes results/demo.png
jupyter lab notebooks/01_test_core.ipynb
```
