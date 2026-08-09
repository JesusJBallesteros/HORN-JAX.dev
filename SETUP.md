# Local setup — Windows + NVIDIA GPU + WSL2 + JAX
## 1. Install WSL2
```powershell
wsl --install -d Ubuntu-24.04
```
## 2. Verify GPU passthrough
Open Ubuntu and run:
```bash
nvidia-smi
```
You should see your card, driver version, and CUDA version.

## 3. Base tooling
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload so uv is on PATH
```
`uv` handles Python versions and virtual environments, and is dramatically faster than conda or pip for this. If you would rather use conda, miniforge works fine — the rest of these instructions adapt directly.

## 4. Create the project
```bash
cd ~                         # NOTE: your home dir, NOT /mnt/c/...
mkdir horn-jax && cd horn-jax
uv venv --python 3.12
source .venv/bin/activate
uv pip install "jax[cuda12]" optax numpy matplotlib pytest ruff
```
> **Keep the project inside the WSL filesystem** (`~/horn-jax`), not on the Windows side (`/mnt/c/Users/...`).

## 5. Verify JAX sees the GPU
```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```
If JAX is installed but cannot reach CUDA, recheck `nvidia-smi` in WSL, then reinstall with `uv pip install --force-reinstall "jax[cuda12]"`.

Check that the GPU is actually being used:
```bash
python -c "
import jax, jax.numpy as jnp, time
x = jnp.ones((4096, 4096))
jax.block_until_ready(x @ x)          # compile first
t = time.time(); jax.block_until_ready(x @ x); print(f'{time.time()-t:.4f}s')
"
```
Well under 0.05s means the GPU is doing the work.

## 6. Editor
Install **VS Code** on Windows plus the **WSL** extension, then from Ubuntu:

```bash
cd ~/horn-jax && code .
```
VS Code runs its UI on Windows and its language server inside Linux. Select the interpreter at `~/horn-jax/.venv/bin/python`.

## 7. Drop in the starter code
Copy the contents of `horn_starter/` into `~/horn-jax/`, then:

```bash
python -m pytest tests/ -q      # expect: 5 passed
python demo.py                  # writes demo.png
```
If those pass, your environment is correct.

---

## Optional: the reference implementation
To check yourself against the published HORN:
```bash
uv pip install brainmass
```

## Hardware expectations
Small HORNs are genuinely small. A 100-oscillator network on sequential MNIST trains in minutes on any recent NVIDIA card, and would still be tractable on CPU. GPU matters for the architectural conditions across several seeds, where a GPU turns a day into an hour.
