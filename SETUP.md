# Local setup — Windows + NVIDIA GPU + WSL2 + JAX

Budget about an hour, most of it waiting on downloads. Do the steps in order; step 3 is the one people skip and then spend an afternoon debugging.

---

## 1. Update your NVIDIA driver (in Windows)

Install the latest Game Ready or Studio driver from nvidia.com, or via GeForce Experience. Reboot.

> **The one rule that matters:** install the GPU driver on **Windows only**. Never install an NVIDIA Linux driver inside WSL. WSL2 passes the Windows driver through a translation layer and exposes it as `libcuda.so` automatically. Installing a Linux driver inside WSL is the single most common way to break CUDA passthrough, and it is hard to diagnose afterwards.

## 2. Install WSL2

In **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when prompted. On first launch Ubuntu asks for a username and password — these are local to WSL and unrelated to your Windows account.

Confirm you are on version 2:

```powershell
wsl -l -v      # VERSION column must read 2
```

## 3. Verify GPU passthrough

Open Ubuntu and run:

```bash
nvidia-smi
```

You should see your card, driver version, and CUDA version. **If this fails, stop and fix it before installing anything else** — nothing downstream will work. Usual causes: driver too old, WSL1 instead of WSL2, or a Linux NVIDIA driver installed inside WSL (see step 1).

## 4. Base tooling

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload so uv is on PATH
```

`uv` handles Python versions and virtual environments, and is dramatically faster than conda or pip for this. If you would rather use conda, miniforge works fine — the rest of these instructions adapt directly.

## 5. Create the project

```bash
cd ~                       # NOTE: your home dir, NOT /mnt/c/...
mkdir horn-jax && cd horn-jax
uv venv --python 3.12
source .venv/bin/activate
uv pip install "jax[cuda12]" optax numpy matplotlib pytest ruff
```

> **Keep the project inside the WSL filesystem** (`~/horn-jax`), not on the Windows side (`/mnt/c/Users/...`). Cross-filesystem I/O in WSL is roughly an order of magnitude slower, and it will make every test run and data load feel sluggish for no reason.

## 6. Verify JAX sees the GPU

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Expected: `[CudaDevice(id=0)]`. If you get `[CpuDevice(id=0)]`, JAX installed but cannot reach CUDA — recheck `nvidia-smi` in WSL, then reinstall with `uv pip install --force-reinstall "jax[cuda12]"`.

A real check that the GPU is actually being used:

```bash
python -c "
import jax, jax.numpy as jnp, time
x = jnp.ones((4096, 4096))
jax.block_until_ready(x @ x)          # compile first
t = time.time(); jax.block_until_ready(x @ x); print(f'{time.time()-t:.4f}s')
"
```

Well under 0.05s means the GPU is doing the work.

## 7. Editor

Install **VS Code** on Windows plus the **WSL** extension, then from Ubuntu:

```bash
cd ~/horn-jax && code .
```

VS Code runs its UI on Windows and its language server inside Linux. Select the interpreter at `~/horn-jax/.venv/bin/python`.

## 8. Drop in the starter code

Copy the contents of `horn_starter/` into `~/horn-jax/`, then:

```bash
python -m pytest tests/ -q      # expect: 5 passed
python demo.py                  # writes demo.png
```

If those pass, your environment is correct and you are ready to start on Week 1.

---

## Optional: the reference implementation

To check yourself against the published HORN:

```bash
uv pip install brainmass
```

Being able to say *"I implemented it independently and validated against the reference"* is worth considerably more than either alone. Do not read their source until your own version passes the tests — you lose the value of the exercise.

---

## Common problems

| Symptom | Cause and fix |
|---|---|
| `nvidia-smi` fails in WSL | Driver too old, or a Linux NVIDIA driver was installed inside WSL. Update on Windows; never install inside WSL. |
| `jax.devices()` shows CPU only | CUDA plugin missing. `uv pip install --force-reinstall "jax[cuda12]"`. |
| Everything is inexplicably slow | Project sits on `/mnt/c/`. Move it to `~/`. |
| `XlaRuntimeError: out of memory` | JAX preallocates 75% of VRAM by default. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or reduce batch size. |
| Gradients become NaN during training | Expected with oscillatory dynamics. Clip gradients, lower the learning rate to 1e-4, and check `zeta` has not gone negative — the log-parameterisation in `core.py` prevents this. |
| WSL eats all your RAM | Create `C:\Users\<you>\.wslconfig` with `[wsl2]` and `memory=16GB`, then `wsl --shutdown`. |

## Hardware expectations

Small HORNs are genuinely small. A 100-oscillator network on sequential MNIST trains in minutes on any recent NVIDIA card, and would still be tractable on CPU. Your GPU matters for the Week 3 sweep — four architectural conditions across several seeds is where a GPU turns a day into an hour. Do not over-plan for scale you will not need.
